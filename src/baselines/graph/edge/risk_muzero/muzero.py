# Risk MuZero (graph/edge) training loop
#
# Implements the full graph-edge Risk MuZero training pipeline:
#   Config           — Pydantic config dataclass
#   run_experiment   — Top-level entry point
#
# Training pipeline (inside run_experiment)
# -----------------------------------------
# 1. Initialise environment, network heads, optimiser, and replay buffer.
# 2. Prefill the buffer with `learning_start` iterations of MCTS self-play.
# 3. Run `max_num_iters` training iterations, each consisting of:
#      a. Evaluation (every `eval_interval` iterations)
#      b. MCTS self-play data collection (`selfplay`)
#      c. `train_epochs_per_iter` gradient steps (`learning_step`)
#      d. Periodic target-network update
#
# Seven network components are trained jointly under a single parameter tree:
#   ObsEncoder          — graph observation → node / edge / aux embeddings
#   ProjectionHead      — embeddings → self-consistency projection
#   PolicyHead          — node embeddings → action logits
#   QValueHead          — node embeddings → value quantile distribution
#   RewardHistoryHead   — observation → accumulated-return quantile distribution
#   DynamicsHead        — (embeddings, action) → next embeddings + reward dist

from functools import partial
from typing import Literal

import chex
import flashbax as fbx
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import pgx
from jax_tqdm import (
    loop_tqdm,  # type: ignore
    scan_tqdm,  # type: ignore
)
from pydantic import BaseModel

import _mctx as mctx
import lib.util as util
from lib.auto_reset_with_iteration import auto_reset
from lib.make_env import make_env
from lib.quantile_losses import batched_quantile_huber_loss
from src.baselines.graph.edge.risk_muzero.networks import make_network_apply_fns
from src.baselines.graph.edge.risk_muzero.util import (
    ExItTransition,
    get_train_targets,
    init_model_and_optim,
    make_buffer,
    scale_gradient,
)
from src.datasets.sbm.cvar_estimation import estimate_distortion_risk

# Sentinel gumbel scale that triggers uniform-random action selection during
# buffer prefill instead of running MCTS.
WARMUP_GUMBEL_SCALE = 10.0


class Config(BaseModel):
    seeds: list[int] = []
    seed: int = 23
    env_name: str = "stochastic-bipartite-matching"  # type: ignore
    use_legal_actions: bool = False
    num_hidden: int = 64
    discount: float = 1.0
    distortion: Literal["cvar", "pow", "wang"] = "cvar"
    alpha_cvar: float = 0.25
    num_quantile_samples: int = 1024

    qtransform: Literal[
        "qtransform_by_parent_and_siblings",
        "qtransform_completed_by_mix_value",
    ] = "qtransform_by_parent_and_siblings"
    is_naive: bool = False

    # Training
    num_simulations: int = 32
    vf_coeff: float = 0.25
    sc_coeff: float = 10.0
    lr: float = 5e-4
    min_lr: float = 1e-5  # Minimum learning rate
    lr_linear_decay: bool = True  # Whether to linearly decay the learning rate
    lr_anneal_iterations: int = 2000  # Number of iterations to decay the learning rate
    max_grad_norm: float = 5.0
    optim_eps: float = 1e-5
    n_step: int = 5
    target_tau: float = 1.0
    target_update_interval: int = 5  # Update every X epochs
    gumbel_start: float = 3.0
    gumbel_end: float = 1.0
    gumbel_anneal_iterations: int = 30  # Number of iterations to anneal gumbel scale

    huber_param: float = 1.0
    num_quantiles: int = 64

    # Buffer
    eval_num_actors: int = 1024
    selfplay_batch_size: int = 32
    train_batch_size: int = 256
    train_epochs_per_iter: int = 20  # For mountain car, this should be 100+
    sample_sequence_length: int = 6
    max_num_steps: int = 32
    total_buffer_size: int = 32 * 8 * 32  # selfplay_batch_size * max_num_steps
    learning_start: int = 8  # Iters of max_num_steps to prefill the buffer

    # Placeholders for dynamic values
    num_actions: int = -1
    is_state_vector: bool = False

    # Logging
    eval_interval: int = 10
    max_num_iters: int = 2000


def run_experiment(args):
    """Run the full graph-edge Risk MuZero training loop.

    Builds the environment, initialises the model and replay buffer, runs
    self-play to fill the buffer, then alternates between self-play and
    gradient updates for ``args.max_num_iters`` iterations.

    Returns:
        json_logs: Scanned dict of per-iteration metrics.
        params:    Final network parameters.
    """
    # Make env
    env, is_state_vector, num_actions = make_env(
        env_name=args.env_name, use_legal_actions=args.use_legal_actions
    )
    args.num_actions = num_actions
    args.is_state_vector = is_state_vector

    # Create MCTS search function
    distortion_fn = {
        "cvar": util.cvar_distortion,
        "pow": util.pow_distortion,
        "wang": util.wang_distortion,
        "sqrt": util.sqrt_utility,
    }[args.distortion]
    search_fn = partial(
        mctx.risk_search,
        utility_fn=partial(distortion_fn, alpha=args.alpha_cvar),
        discount_factor=args.discount,
        num_quantile_samples=args.num_quantile_samples,
        sample_history=not args.is_naive,
    )
    qtransform_fn = (
        mctx.qtransform_by_parent_and_siblings
        if args.qtransform == "qtransform_by_parent_and_siblings"
        else mctx.qtransform_completed_by_mix_value
    )

    # Make and initialize the model
    (
        init_model,
        representation_apply,
        projection_apply,
        policy_apply,
        critic_apply,
        reward_history_apply,
        recurrent_inference,
    ) = make_network_apply_fns(args)
    params, optimizer, opt_state = init_model_and_optim(env, init_model, args=args)
    target_params = jax.tree.map(lambda x: jnp.copy(x), params)

    # Make trajectory buffer
    buffer_fn, buffer_state = make_buffer(env, args)

    # Make tau hats for quantile regression
    tau_hats = util.make_tau_hats(num_quantiles=args.num_quantiles)

    @jax.jit
    def root_fn(
        params: optax.Params,
        env_state: pgx.State,
        step_count: jnp.ndarray,
    ):
        """Root function for MCTS search."""
        initial = env_state.terminated | env_state.truncated | step_count == 0  # (b,)

        # We auto reset during selfplay, but NOT during recurrent_fn.
        # If this env_state is terminated, and we're using it as a root then we need
        # to manually reset the termination flag here.
        def reset_root(env_state: pgx.State):
            """Reset the root state if it is terminated or truncated."""
            return jax.lax.cond(
                (env_state.terminated | env_state.truncated),
                lambda: env_state.replace(  # type: ignore
                    terminated=jnp.bool_(False),
                    truncated=jnp.bool_(False),
                    rewards=jnp.zeros_like(env_state.rewards),
                    _step_count=jnp.zeros_like(env_state._step_count),  # type: ignore
                ),
                lambda: env_state,
            )

        env_state = jax.vmap(reset_root)(env_state)

        node_embeddings, aux_embedding, edge_embeddings = representation_apply.apply(
            params,
            env_state.observation["node_features"],  # type: ignore
            env_state.observation["edge_features"],  # type: ignore
            env_state.observation["senders"],  # type: ignore
            env_state.observation["receivers"],  # type: ignore
            env_state.observation["aux"],  # type: ignore
        )  # (b, num_actions, hidden), # (b, hidden), # (b, num_edges, hidden)
        logits = policy_apply.apply(params, node_embeddings, aux_embedding)  # (b, num_actions)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)

        value_dist = critic_apply.apply(
            params, node_embeddings, aux_embedding
        )  # (b, num_quantiles)

        # Zero out historical return if this is the initial state.
        return_history_dist = reward_history_apply.apply(
            params,
            env_state.observation["node_features"],  # type: ignore
            env_state.observation["edge_features"],  # type: ignore
            env_state.observation["senders"],  # type: ignore
            env_state.observation["receivers"],  # type: ignore
            env_state.observation["aux"],  # type: ignore
        )
        return_history_dist = (1 - initial[:, None]) * return_history_dist  # (b, num_quantiles)

        return mctx.RiskRootFnOutput(
            return_history=return_history_dist,  # type: ignore
            prior_logits=logits,  # type: ignore
            value=value_dist,  # type: ignore
            embedding=(  # type: ignore
                node_embeddings,
                edge_embeddings,
                aux_embedding,
                env_state.observation["senders"],
                env_state.observation["receivers"],
                step_count,
            ),
        )

    @jax.jit
    def recurrent_fn(
        params: optax.Params,
        _rng_key: jnp.ndarray,
        action: jnp.ndarray,
        obs_embedding: tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ],
    ):
        (
            node_embeddings,
            edge_embeddings,
            aux_embedding,
            senders,
            receivers,
            step_count,
        ) = obs_embedding
        node_embeddings, edge_embeddings, reward_dist, aux_embedding = recurrent_inference.apply(
            params,
            node_embeddings,
            edge_embeddings,
            aux_embedding,
            senders,
            receivers,
            action,
        )

        logits = policy_apply.apply(params, node_embeddings, aux_embedding)  # (b, num_actions)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)

        value_dist = critic_apply.apply(
            params, node_embeddings, aux_embedding
        )  # (b, num_quantiles)

        return mctx.RecurrentFnOutput(
            prior_logits=logits,  # type: ignore
            value=value_dist,  # type: ignore
            reward=reward_dist,  # type: ignore
            discount=jnp.ones((reward_dist.shape[0],)) * args.discount,  # type: ignore
        ), (
            node_embeddings,
            edge_embeddings,
            aux_embedding,
            senders,
            receivers,
            step_count + 1,
        )

    # Define eval loop
    @jax.jit
    def evaluate(rng_key: jnp.ndarray, params: hk.Params):
        """Evaluate the model by running selfplay and computing the average reward."""
        rng_key, subkey = jax.random.split(rng_key)
        batch_size = 512
        keys = jax.random.split(subkey, batch_size)

        total_instances = env._instances.num_nodes.shape[0] // 2  # type: ignore
        num_instances_per_batch = 512
        num_batches = (total_instances) // num_instances_per_batch
        assert total_instances % num_instances_per_batch == 0

        @loop_tqdm(num_batches)
        @jax.jit
        def eval_one_batch(batch_idx, carry):
            rng_key, acc_mean, acc_risk, acc_opt_risk, acc_opt_expected = carry

            # Instances for this batch
            iteration = (
                jnp.arange(num_instances_per_batch, dtype=jnp.int32)
                + batch_idx * num_instances_per_batch
            )
            iteration = iteration.reshape((-1, 1))
            iteration = jnp.repeat(iteration, batch_size // iteration.shape[0], axis=1)
            iteration = iteration.reshape((-1,))

            offset = jnp.zeros(batch_size, dtype=jnp.int32)
            state = jax.vmap(env.init_v2, in_axes=(0, 0, 0, None, None))(  # type: ignore
                keys, iteration, offset, 1, 1
            )

            # ---- optimal risk for this batch ----
            opt_risk_batch = jnp.mean(state._x.optimal_cvar_value)  # type: ignore
            opt_expected_batch = jnp.mean(state._x.cvar_100)  # type: ignore
            step_fn = jax.vmap(env.step)
            ep_return = jnp.zeros_like(state.rewards)
            step = jnp.array(0)
            max_steps = env._instances.num_nodes[0].astype(jnp.int32)  # type: ignore

            def cond_fn(tup):
                state, _, _, step, _ = tup
                still_running = ~state.terminated.all()
                if max_steps is None:
                    return still_running
                return jnp.logical_and(still_running, step < max_steps).all()

            @jax.jit
            def loop_fn(tup):
                state, R, key, step, actions = tup

                # Initialize the root
                step_count = jnp.full((batch_size,), step, dtype=jnp.int32)
                root = root_fn(params, state, step_count)

                search_output = mctx.gumbel_muzero_policy(
                    params=params,
                    rng_key=key,
                    root=root,
                    recurrent_fn=recurrent_fn,
                    num_simulations=args.num_simulations,
                    invalid_actions=~state.legal_action_mask,
                    qtransform=qtransform_fn,
                    search_fn=search_fn,
                    gumbel_scale=0.0,
                )
                action = search_output.action  # (batch_size,)

                # Step environment
                key, subkey = jax.random.split(key)
                keys = jax.random.split(subkey, batch_size)
                state = step_fn(state, action, keys)

                # Write action into buffer at position `step`
                actions = actions.at[step].set(action)

                return state, R + state.rewards, key, step + 1, actions

            max_steps = 30
            selecting_fors = jnp.arange(90, 120, 1)  # nleft + right to nleft + 2*right
            actions_init = jnp.zeros((max_steps, batch_size), dtype=jnp.int32)
            carry_init = (state, ep_return, rng_key, step, actions_init)
            state, R, rng_key, _, actions_taken = jax.lax.while_loop(cond_fn, loop_fn, carry_init)

            # Edges are [selecting_for, action]
            actions_taken = actions_taken.swapaxes(0, 1)  # (b, max_steps)
            selecting_fors = jnp.broadcast_to(selecting_fors, actions_taken.shape)
            edge_indices = jax.vmap(env.find_edge_indices)(  # type: ignore
                state.observation["senders"],  # type: ignore
                state.observation["receivers"],  # type: ignore
                selecting_fors,
                actions_taken,
            ).squeeze(axis=2)  # type: ignore # (512, 10)
            edge_types = jnp.take_along_axis(  # type: ignore
                state.observation["edge_types"],  # type: ignore
                edge_indices,
                axis=1,
            )

            # ---- empirical risk for this batch ----
            R_risks = jax.vmap(estimate_distortion_risk, in_axes=(0, 0, None, None, None))(
                jax.random.split(rng_key, batch_size),
                edge_types,
                10_000,
                args.alpha_cvar,
                args.distortion,
            )
            R = jax.vmap(estimate_distortion_risk, in_axes=(0, 0, None, None))(
                jax.random.split(rng_key, batch_size),
                edge_types,
                10_000,
                1.0,
            )
            R_mean = jnp.mean(R)
            R_risk = jnp.mean(R_risks)

            return (
                rng_key,
                acc_mean + R_mean,
                acc_risk + R_risk,
                acc_opt_risk + opt_risk_batch,
                acc_opt_expected + opt_expected_batch,
            )

        # Run across all batches
        rng_key, total_mean, total_risk, total_opt_risk, total_opt_expected = jax.lax.fori_loop(
            0, num_batches, eval_one_batch, (rng_key, 0.0, 0.0, 0.0, 0.0)
        )

        # Average across batches
        R_mean = total_mean / num_batches
        R_risk = total_risk / num_batches
        R_opt_risk = total_opt_risk / num_batches
        R_opt_expected = total_opt_expected / num_batches
        return R_mean, R_risk, R_opt_risk, R_opt_expected

    # Define selfplay loop
    def selfplay(
        rng_key: jax.Array,
        params: hk.Params,
        buffer_state: fbx.trajectory_buffer.TrajectoryBufferState,
        gumbel_scale: jnp.ndarray,
        env_state: pgx.State,
        episode_stats: dict[str, jnp.ndarray],
    ) -> tuple[
        pgx.State,
        dict[str, jnp.ndarray],
        fbx.trajectory_buffer.TrajectoryBufferState,
        ExItTransition,
    ]:
        @scan_tqdm(args.max_num_steps)
        def step_fn(
            carry: tuple[
                pgx.State,
                dict[str, jnp.ndarray],
            ],
            iter_data: jnp.ndarray,
        ) -> tuple[tuple[pgx.State, dict[str, jnp.ndarray]], ExItTransition]:
            state, episode_stats = carry
            _, key = iter_data
            key1, key2, key3 = jax.random.split(key, 3)  # (2,), (2,)
            observation = state.observation

            def search():
                # Initialize the root
                root = root_fn(
                    params,
                    state,
                    step_count=episode_stats["episode_length"],
                )

                # Run MCTS search
                search_output = mctx.gumbel_muzero_policy(
                    params=params,
                    rng_key=key1,
                    root=root,
                    recurrent_fn=recurrent_fn,
                    num_simulations=args.num_simulations,
                    invalid_actions=~state.legal_action_mask,
                    # Revisit: qtransform mix is very greedy
                    qtransform=qtransform_fn,
                    search_fn=search_fn,
                    gumbel_scale=gumbel_scale,
                    # max_num_considered_actions=8,
                    # temperature=temperature_scale,
                )
                action = search_output.action  # (b,)
                search_policy = search_output.action_weights
                return action, search_policy

            def random_selection():
                """Randomly select a valid action with equal probability."""
                valid_actions = state.legal_action_mask  # shape: (batch_size, num_actions)
                batch_size, num_actions = valid_actions.shape

                # Create random scores for actions, but mask invalid ones to -inf
                random_scores = jax.random.uniform(key1, (batch_size, num_actions))
                masked_scores = jnp.where(valid_actions == 1, random_scores, -jnp.inf)

                # Pick the argmax of the masked scores — effectively random among valids
                action = jnp.argmax(masked_scores, axis=1)

                # Create a uniform distribution over valid actions
                counts = valid_actions.sum(axis=1, keepdims=True)
                search_policy = valid_actions / counts

                return action, search_policy

            # If gumbel scale above 10, we assume this is a warmup step and
            # randomly pick an action with equal probability logits
            action, search_policy = jax.lax.cond(
                gumbel_scale >= WARMUP_GUMBEL_SCALE,
                random_selection,
                search,
            )  # type: ignore

            keys = jax.random.split(key2, state.observation["node_features"].shape[0])
            state = jax.vmap(auto_reset(env.step, env.init_v2))(state, action, keys)  # type: ignore

            # Create transition
            transition = ExItTransition(
                step_count=episode_stats["episode_length"],  # (b,)
                # This history is [...obs (history_length), next_obs]
                done=state.terminated,  # (b,)
                action=jnp.asarray(action),  # (b,)
                reward=state.rewards[:, -1],  # (b,)
                search_policy=jnp.asarray(search_policy),  # (b, num_actions)
                obs=observation,  # (b, *obs_shape)
                info=episode_stats,
            )

            # Update episode stats
            episode_stats["episode_return"] += jnp.sum(state.rewards, axis=1)
            episode_stats["episode_length"] += 1
            episode_stats["is_terminal_step"] = state.terminated

            # Reset stats for terminal steps
            episode_stats = jax.tree_util.tree_map(
                lambda x: jnp.where(state.terminated, jnp.zeros_like(x), x),
                episode_stats,
            )
            return (state, episode_stats), transition

        # Run self-play for max_num_steps per batch
        rng_key, sub_key = jax.random.split(rng_key)
        key_seq = jax.random.split(sub_key, args.max_num_steps)

        (env_state, episode_stats), traj_batch = jax.lax.scan(
            step_fn,  # type: ignore
            (env_state, episode_stats),
            (jnp.arange(args.max_num_steps), key_seq),  # type: ignore
        )

        # Switch the time and batch axes
        traj_batch = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), traj_batch
        )  # (b, t, ...)
        # Add the batch to the buffer
        buffer_state = buffer_fn.add(
            buffer_state,
            traj_batch,
        )  # type: ignore

        return env_state, episode_stats, buffer_state, traj_batch

    @jax.jit
    def selfplay_scan_fn(carry, iteration):
        """Scan function for self-play to prefill buffer."""
        (
            rng_key,
            buffer_state,
            opt_state,
            params,
            target_params,
            env_state,
            episode_stats,
            eval_reward,
        ) = carry

        # Run self-play
        rng_key, subkey = jax.random.split(rng_key)
        env_state, episode_stats, buffer_state, traj_batch = selfplay(
            subkey,
            params,
            buffer_state,
            jnp.asarray(
                WARMUP_GUMBEL_SCALE
            ),  # Flag for random action selection to prefill the buffer
            env_state,
            episode_stats,
        )

        return (
            rng_key,
            buffer_state,
            opt_state,
            params,
            target_params,
            env_state,
            episode_stats,
            eval_reward,
        ), traj_batch

    # Define training loop
    def learning_step(params, target_params, opt_state, buffer_state, key):
        """Perform a learning step using the buffer data."""

        def loss_fn(
            params: hk.Params,
            obs: jnp.ndarray,
            s_node_target: jnp.ndarray,
            s_edge_target: jnp.ndarray,
            s_aux_target: jnp.ndarray,
            policy_target: jnp.ndarray,
            a_seq: jnp.ndarray,
            r_target: jnp.ndarray,
            v_target: jnp.ndarray,
            dt: jnp.ndarray,
            historical_reward_targets: jnp.ndarray,
            initials: jnp.ndarray,
        ):
            # Generate root embeddings
            root_node_reps, root_aux_embedding, root_edge_embeddings = representation_apply.apply(
                params,
                obs["node_features"][:, 0],  # type: ignore
                obs["edge_features"][:, 0],  # type: ignore
                obs["senders"][:, 0],  # type: ignore
                obs["receivers"][:, 0],  # type: ignore
                obs["aux"][:, 0],  # type: ignore
            )
            obs = jax.tree_util.tree_map(lambda x: x[:, 1 : args.sample_sequence_length + 1], obs)

            @jax.remat  # type: ignore
            def unroll_fn(carry, targets):
                (
                    total_loss,
                    node_embeddings,
                    edge_embeddings,
                    aux_embedding,
                    mask,
                ) = carry
                (
                    a,
                    r_target,
                    s_node_target,
                    s_edge_target,
                    s_aux_target,
                    policy_target,
                    v_target,
                    done,
                    obs_i,
                    historical_reward_target,
                    initial,
                    step,
                ) = targets

                # HISTORICAL REWARD LOSS
                historical_reward = reward_history_apply.apply(
                    params,
                    obs_i["node_features"],  # type: ignore
                    obs_i["edge_features"],  # type: ignore
                    obs_i["senders"],  # type: ignore
                    obs_i["receivers"],  # type: ignore
                    obs_i["aux"],
                )
                historical_reward_loss = batched_quantile_huber_loss(
                    dist_src=historical_reward,
                    tau_src=tau_hats,
                    dist_target=historical_reward_target,
                    huber_param=args.huber_param,
                    stop_target_gradients=True,
                )
                chex.assert_shape(historical_reward_loss, (historical_reward.shape[0],))
                # Zero out loss for root initial step (it isn't used at inference, and not bootstrapped)
                historical_reward_loss = (1 - initial) * historical_reward_loss
                historical_reward_loss = jnp.mean(historical_reward_loss)

                # ACTOR LOSS
                policy_logits_pred = policy_apply.apply(params, node_embeddings, aux_embedding)
                policy_loss = optax.softmax_cross_entropy(
                    logits=policy_logits_pred, labels=policy_target
                )
                # Zero the policy loss for absorbing states
                policy_loss = policy_loss * mask

                # CRITIC LOSS
                # Zero the policy targets after done as absorbing
                v_target = v_target * mask[:, None]
                v_pred = critic_apply.apply(
                    params, node_embeddings, aux_embedding
                )  # (b, num_quantiles)
                value_loss = batched_quantile_huber_loss(
                    dist_src=v_pred,
                    tau_src=tau_hats,
                    dist_target=v_target,
                    huber_param=args.huber_param,
                    stop_target_gradients=True,
                )
                chex.assert_shape(value_loss, (v_target.shape[0],))

                # Scale the gradients of the obs_embedding
                node_embeddings = scale_gradient(node_embeddings, 0.5)
                edge_embeddings = scale_gradient(edge_embeddings, 0.5)

                (
                    next_node_embeddings,
                    next_edge_embeddings,
                    reward_pred,
                    next_aux_embedding,
                ) = recurrent_inference.apply(
                    params,
                    node_embeddings,
                    edge_embeddings,
                    aux_embedding,
                    obs["senders"][:, 0],  # type: ignore
                    obs["receivers"][:, 0],  # type: ignore
                    a,
                )

                # REWARD LOSS
                # Zero reward targets after done as absorbing
                r_target = r_target * mask
                r_target = r_target[:, None]  # (b, 1)
                reward_loss = batched_quantile_huber_loss(
                    dist_src=reward_pred,
                    tau_src=tau_hats,
                    dist_target=r_target,
                    huber_param=args.huber_param,
                    stop_target_gradients=True,
                )
                chex.assert_shape(reward_loss, (r_target.shape[0],))

                # SELF-CONSISTENCY LOSS
                # We want to ensure that the next state embedding is consistent with the target state embedding
                next_node_proj, next_edge_proj, next_aux_proj = projection_apply.apply(
                    params,
                    next_node_embeddings,
                    next_edge_embeddings,
                    next_aux_embedding,
                )  # (b, num_hidden)

                next_node_proj = next_node_proj.reshape((next_node_proj.shape[0], -1))
                next_edge_proj = next_edge_proj.reshape((next_edge_proj.shape[0], -1))
                next_aux_proj = next_aux_proj.reshape((next_aux_proj.shape[0], -1))
                s_node_target_ = s_node_target.reshape((s_node_target.shape[0], -1))
                s_edge_target_ = s_edge_target.reshape((s_edge_target.shape[0], -1))
                s_aux_target_ = s_aux_target.reshape((s_aux_target.shape[0], -1))

                unit_next_node_proj = next_node_proj / jnp.linalg.norm(
                    next_node_proj, axis=-1, keepdims=True
                )  # (b, num_hidden)
                unit_next_edge_proj = next_edge_proj / jnp.linalg.norm(
                    next_edge_proj, axis=-1, keepdims=True
                )  # (b, num_hidden)
                unit_next_aux_proj = next_aux_proj / jnp.linalg.norm(
                    next_aux_proj, axis=-1, keepdims=True
                )  # (b, num_hidden)

                unit_s_node_target = s_node_target_ / jnp.linalg.norm(
                    s_node_target_, axis=-1, keepdims=True
                )
                unit_s_edge_target = s_edge_target_ / jnp.linalg.norm(
                    s_edge_target_, axis=-1, keepdims=True
                )
                unit_s_aux_target = s_aux_target_ / jnp.linalg.norm(
                    s_aux_target_, axis=-1, keepdims=True
                )

                sc_node_loss = 1 - jax.vmap(jnp.dot)(unit_next_node_proj, unit_s_node_target)
                sc_edge_loss = 1 - jax.vmap(jnp.dot)(unit_next_edge_proj, unit_s_edge_target)
                sc_aux_loss = 1 - jax.vmap(jnp.dot)(unit_next_aux_proj, unit_s_aux_target)

                sc_loss = sc_node_loss + sc_edge_loss + sc_aux_loss
                sc_loss = sc_loss * mask  # No self-consistency loss for absorbing states

                curr_loss = {
                    "actor_loss": policy_loss,
                    "value_loss": value_loss,
                    "reward_loss": reward_loss,
                    "historical_reward_loss": historical_reward_loss,
                    "self_consistency_loss": sc_loss,
                    "node_self_consistency_loss": sc_node_loss * mask,
                    "edge_self_consistency_loss": sc_edge_loss * mask,
                    "aux_self_consistency_loss": sc_aux_loss * mask,
                }
                total_loss = jax.tree_util.tree_map(
                    lambda x, y: x + y.mean(), total_loss, curr_loss
                )

                # Update the mask
                mask = jnp.where(done, jnp.zeros_like(mask), mask)
                return (
                    total_loss,
                    next_node_embeddings,
                    next_edge_embeddings,
                    next_aux_embedding,
                    mask,
                ), curr_loss

            targets = (
                a_seq,
                r_target,
                s_node_target[:, 1 : args.sample_sequence_length + 1],
                s_edge_target[:, 1 : args.sample_sequence_length + 1],
                s_aux_target[:, 1 : args.sample_sequence_length + 1],
                policy_target,
                v_target,
                dt,
                obs,
                historical_reward_targets,
                initials,
            )
            targets = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, 0, 1), targets
            )  # (t, b, ...)
            targets = (*targets, jnp.arange(0, args.sample_sequence_length))

            init_total_loss = {
                "actor_loss": jnp.array(0.0),
                "value_loss": jnp.array(0.0),
                "reward_loss": jnp.array(0.0),
                "historical_reward_loss": jnp.array(0.0),
                "self_consistency_loss": jnp.array(0.0),
                "node_self_consistency_loss": jnp.array(0.0),
                "edge_self_consistency_loss": jnp.array(0.0),
                "aux_self_consistency_loss": jnp.array(0.0),
            }
            init_mask = jnp.ones((v_target.shape[0],))
            (losses, _, _, _, _), _ = jax.lax.scan(
                unroll_fn,
                (
                    init_total_loss,
                    root_node_reps,
                    root_edge_embeddings,
                    root_aux_embedding,
                    init_mask,
                ),
                targets,
            )
            # Divide by the number of unrolled steps to ensure
            # a consistent scale across different unroll lengths
            losses = jax.tree_util.tree_map(lambda x: x / (args.sample_sequence_length - 1), losses)
            return (
                losses["actor_loss"]
                + args.vf_coeff * losses["value_loss"]
                + losses["reward_loss"]
                + losses["historical_reward_loss"]
                + args.sc_coeff * losses["self_consistency_loss"],
                losses,
            )

        # Sample a batch from the buffer and compute targets
        key, subkey = jax.random.split(key)
        batch = buffer_fn.sample(buffer_state, subkey).experience
        (
            policy_targets,
            reward_targets,
            value_targets,
            s_node_targets,
            s_edge_targets,
            s_aux_targets,
        ) = get_train_targets(
            batch,
            target_params,
            representation_apply,
            critic_apply,
            args,
        )

        B, S = batch.obs["node_features"].shape[:2]  # (b, t)
        obs = jax.tree_util.tree_map(lambda x: x.reshape((B * S, *x.shape[2:])), batch.obs)
        # Generate history targets
        initials = batch.step_count == 0  # (b, t)
        initials = initials.reshape((-1,))  # (b * t,)
        historical_reward_prev = jax.lax.stop_gradient(
            reward_history_apply.apply(
                target_params,
                obs["node_features"],  # type: ignore
                obs["edge_features"],  # type: ignore
                obs["senders"],  # type: ignore
                obs["receivers"],  # type: ignore
                obs["aux"],
            )
        )  # (b * t, num_quantiles)
        rewards = batch.reward.reshape((-1,))  # (b * t)
        historical_reward_targets = (
            rewards[:, None] + (1 - initials[:, None]) * historical_reward_prev
        )
        historical_reward_targets = historical_reward_targets.reshape((B, S, -1))
        initials = initials.reshape((B, S))
        historical_reward_targets = historical_reward_targets[:, : args.sample_sequence_length]

        obs = jax.tree_util.tree_map(lambda x: x[:, : args.sample_sequence_length + 1], batch.obs)
        initials = initials[:, 1 : args.sample_sequence_length + 1]

        (loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params,
            obs,
            s_node_targets[:, : args.sample_sequence_length + 1],
            s_edge_targets[:, : args.sample_sequence_length + 1],
            s_aux_targets[:, : args.sample_sequence_length + 1],
            policy_targets,
            batch.action[:, : args.sample_sequence_length],
            reward_targets,
            value_targets,
            batch.done[:, : args.sample_sequence_length],
            historical_reward_targets,
            initials,
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params=params, updates=updates)
        return new_params, new_opt_state, (loss, losses)

    def train_loop_body(
        carry, iteration
    ):  # -> tuple[tuple[Any, TrajectoryBufferState[Any], Any, Any], Any]:
        (
            rng_key,
            buffer_state,
            opt_state,
            params,
            target_params,
            env_state,
            episode_stats,
            last_eval_reward,
        ) = carry
        # Split key for this iteration
        rng_key, subkey = jax.random.split(rng_key)

        def eval_fn():
            R, R_risk, R_opt_risk, R_opt_expected = evaluate(subkey, params)
            jax.debug.print(
                "Iter {i} / {max_num_iters}, Eval Reward: {r}, Eval {distortion} risk: {r_risk} ({r_opt_risk}), Opt Expected: {r_opt_expected}",
                i=iteration,
                max_num_iters=args.max_num_iters,
                r=R.mean(),
                distortion=args.distortion,
                r_risk=R_risk.mean(),
                r_opt_risk=R_opt_risk.mean(),
                r_opt_expected=R_opt_expected.mean(),
            )
            return R.mean(), R_risk.mean(), R_opt_expected.mean()

        # Run evaluation conditionally
        eval_R = jax.lax.cond(
            iteration % args.eval_interval == 0,
            eval_fn,
            lambda: last_eval_reward,
        )

        # Print/logging (must be done outside JAX or via `jax.debug.print`)
        gumbel_scale = jnp.clip(
            args.gumbel_start
            - (args.gumbel_start - args.gumbel_end) * (iteration / args.gumbel_anneal_iterations),
            a_min=args.gumbel_end,
            a_max=args.gumbel_start,
        )
        current_lr = util.linear_schedule(
            iteration,
            args.lr,
            args.lr_anneal_iterations,
            args.min_lr,
        )
        jax.debug.print(
            "Iteration {i} started. Gumbel scale: {gumbel_scale}, LR: {lr}",
            i=iteration,
            gumbel_scale=gumbel_scale,
            lr=current_lr,
        )

        # Self-play data collection
        env_state, episode_stats, buffer_state, traj_batch = selfplay(
            subkey,
            params,
            buffer_state,
            gumbel_scale,
            env_state,
            episode_stats,
        )

        # Log the training stats
        episode_returns = traj_batch.info["episode_return"] * traj_batch.info[
            "is_terminal_step"
        ].astype(jnp.float32)  # type: ignore
        episode_lengths = traj_batch.info["episode_length"] * traj_batch.info[
            "is_terminal_step"
        ].astype(jnp.int32)  # type: ignore
        total_terminations = jnp.sum(
            traj_batch.info["is_terminal_step"].astype(jnp.int32)  # type: ignore
        )
        average_return = jnp.sum(episode_returns) / (total_terminations + 1e-8)
        average_length = jnp.sum(episode_lengths) / (total_terminations + 1e-8)
        jax.debug.print(
            "Iter {i} / {max_num_iters}, Train Avg Return: {avg_return:.2f}, Avg Length: {avg_length:.2f}",
            i=iteration,
            max_num_iters=args.max_num_iters,
            avg_return=average_return,
            avg_length=average_length,
        )

        # Training step
        rng_key, subkey = jax.random.split(rng_key)

        # Learning step for epochs
        def scan_fn(carry, epoch):
            params, target_params, opt_state, rng_key = carry
            rng_key, subkey = jax.random.split(rng_key)
            params, opt_state, (loss, losses) = learning_step(
                params, target_params, opt_state, buffer_state, subkey
            )

            # Update target parameters
            target_params = jax.lax.cond(
                epoch % args.target_update_interval == 0,
                lambda: jax.tree_util.tree_map(
                    lambda target, online: (
                        args.target_tau * online + (1 - args.target_tau) * target
                    ),
                    target_params,
                    params,
                ),
                lambda: target_params,
            )

            return (params, target_params, opt_state, rng_key), (loss, losses)

        (params, target_params, opt_state, _), (loss, losses) = jax.lax.scan(
            scan_fn,
            (params, target_params, opt_state, subkey),
            jnp.arange(args.train_epochs_per_iter),
        )
        # Log the losses
        jax.debug.print(
            "Iter {i} / {max_num_iters}, Loss: {loss:.4f}, Actor Loss: {actor_loss:.4f}, "
            "Value Loss: {value_loss:.4f}, Self-consistency Loss: {self_consistency_loss:.4f} (node: {node_self_consistency_loss:.4f}, edge: {edge_self_consistency_loss:.4f}, aux: {aux_self_consistency_loss:.4f}), R: {reward_loss:.4f}, RH: {reward_history_loss:.4f}",
            i=iteration,
            max_num_iters=args.max_num_iters,
            loss=jnp.mean(loss),
            actor_loss=jnp.mean(losses["actor_loss"]),
            value_loss=jnp.mean(losses["value_loss"]),
            self_consistency_loss=jnp.mean(losses["self_consistency_loss"]),
            node_self_consistency_loss=jnp.mean(losses["node_self_consistency_loss"]),
            edge_self_consistency_loss=jnp.mean(losses["edge_self_consistency_loss"]),
            aux_self_consistency_loss=jnp.mean(losses["aux_self_consistency_loss"]),
            reward_loss=jnp.mean(losses["reward_loss"]),
            reward_history_loss=jnp.mean(losses["historical_reward_loss"]),
        )

        carry = (
            rng_key,
            buffer_state,
            opt_state,
            params,
            target_params,
            env_state,
            episode_stats,
            eval_R,
        )
        R, R_risk, R_opt_expected = eval_R
        json_log = {
            "num_steps": (iteration + 1) * args.max_num_steps,
            "training_steps": (iteration + 1) * args.train_epochs_per_iter,
            "iteration": iteration,
            "loss": jnp.mean(loss),
            "actor_loss": jnp.mean(losses["actor_loss"]),
            "value_loss": jnp.mean(losses["value_loss"]),
            "self_consistency_loss": jnp.mean(losses["self_consistency_loss"]),
            "reward_loss": jnp.mean(losses["reward_loss"]),
            "reward_history_loss": jnp.mean(losses["historical_reward_loss"]),
            "train_average_return": average_return,
            "train_average_length": average_length,
            "gumbel_scale": gumbel_scale,
            "learning_rate": current_lr,
            "last_eval_reward": R,
            "last_eval_risk": R_risk,
            "last_eval_opt_expected": R_opt_expected,
        }
        return carry, json_log

    # Run scan
    rng_key = jax.random.PRNGKey(seed=args.seed)
    init_rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, args.selfplay_batch_size)

    iteration = jnp.zeros(args.selfplay_batch_size, dtype=jnp.int32)
    offset = jnp.arange(args.selfplay_batch_size, dtype=jnp.int32)
    env_state = jax.vmap(env.init_v2, in_axes=(0, 0, 0, None, None))(  # type: ignore
        keys, iteration, offset, args.selfplay_batch_size, 0
    )

    episode_stats_init = {
        "episode_return": jnp.zeros((args.selfplay_batch_size,)),
        "episode_length": jnp.zeros((args.selfplay_batch_size,), dtype=jnp.int32),
        "is_terminal_step": jnp.zeros((args.selfplay_batch_size,), dtype=bool),
    }

    initial_carry = (
        init_rng_key,
        buffer_state,
        opt_state,
        params,
        target_params,
        env_state,
        episode_stats_init,
        (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),  # last_eval_reward
    )

    # Self-play to fill the buffer
    initial_carry, traj_batch = jax.lax.scan(
        selfplay_scan_fn, initial_carry, jnp.arange(args.learning_start)
    )

    iterations = jnp.arange(args.max_num_iters)
    final_carry, json_logs = jax.lax.scan(train_loop_body, initial_carry, iterations)

    return json_logs, params


if __name__ == "__main__":
    import os

    import numpy as onp

    def save_logs(logs):
        """
        Save logs to a specified file path.
        """
        log_file_path = "./logs/rmz/sbm/logger.log"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        numpy_logs = onp.array(logs)
        onp.save(log_file_path, numpy_logs)

    args = Config()
    print(args)
    run_experiment(args)
