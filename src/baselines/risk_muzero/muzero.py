# Risk MuZero training loop
#
# Implements the full Risk MuZero training pipeline:
#   Config           — Pydantic config dataclass
#   run_experiment   — Top-level entry point
#
# Training pipeline (inside run_experiment)
# -----------------------------------------
# 1. Initialise environment, network heads, optimiser, and replay buffer.
# 2. Prefill the buffer with `learning_start` iterations of MCTS self-play
#    (random action selection when gumbel_scale >= WARMUP_GUMBEL_SCALE).
# 3. Run `max_num_iters` training iterations, each consisting of:
#      a. Evaluation (every `eval_interval` iterations)
#      b. MCTS self-play data collection (`selfplay`)
#      c. `train_epochs_per_iter` gradient steps (`learning_step`)
#      d. Periodic target-network update
#
# Network heads (all sharing one parameter tree):
#   FeatureExtractor  — representation network
#   DynamicsHead      — GRU transition model (next state + reward dist)
#   PolicyHead        — policy logits
#   ValueHead         — quantile value distribution
#   RewardHistoryHead — discounted accumulated-return distribution
#   ProjectionHead    — embedding projection for self-consistency loss
#
# The unrolled loss combines policy, value, reward, reward-history, and
# self-consistency terms over `sample_sequence_length` steps.

from functools import partial
from typing import Literal

import chex
import flashbax as fbx
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import pgx
from jax_tqdm import scan_tqdm  # type: ignore
from pgx.experimental import auto_reset
from pydantic import BaseModel

import _mctx as mctx
import lib.util as util
from lib.history_buffer import (
    EnvHistory,
    history_reset,
    history_reset_at_done,
    history_step,
    make_batch_history,
)
from lib.make_env import make_env
from lib.quantile_losses import batched_quantile_huber_loss
from src.baselines.risk_muzero.networks import make_network_apply_fns
from src.baselines.risk_muzero.util import (
    ExItTransition,
    get_train_targets,
    init_model_and_optim,
    make_buffer,
    scale_gradient,
)

# Sentinel gumbel scale that triggers uniform-random action selection during
# buffer prefill instead of running MCTS.
WARMUP_GUMBEL_SCALE = 10.0


class Config(BaseModel):
    seeds: list[int] = []
    seed: int = 0
    env_name: str = "grid-risk"  # type: ignore
    use_legal_actions: bool = False
    num_hidden: int = 256
    discount: float = 1.0
    distortion: Literal["cvar", "wang", "pow"] = "cvar"
    alpha_cvar: float = 1.0
    num_quantile_samples: int = 1024

    qtransform: Literal[
        "qtransform_by_parent_and_siblings",
        "qtransform_completed_by_mix_value",
    ] = "qtransform_by_parent_and_siblings"
    is_naive: bool = False

    # Training
    num_simulations: int = 32
    vf_coeff: float = 0.25
    sc_coeff: float = 2.0
    lr: float = 5e-3
    min_lr: float = 1e-3  # Minimum learning rate
    lr_linear_decay: bool = True  # Whether to linearly decay the learning rate
    lr_anneal_iterations: int = 200  # Number of iterations to decay the learning rate
    max_grad_norm: float = 5.0
    optim_eps: float = 1e-5
    n_step: int = 5
    target_tau: float = 1.0
    target_update_interval: int = 5  # Update every X epochs
    gumbel_start: float = 3.0
    gumbel_end: float = 1.0
    gumbel_anneal_iterations: int = 30  # Number of iterations to anneal gumbel scale
    history_length: int = 10  # Number of previous steps to consider in the history

    huber_param: float = 1.0
    num_quantiles: int = 64

    # Buffer
    eval_num_actors: int = 1024
    selfplay_batch_size: int = 32
    train_batch_size: int = 1024
    train_epochs_per_iter: int = 20  # For mountain car, this should be 100+
    sample_sequence_length: int = 6
    max_num_steps: int = 24
    total_buffer_size: int = 32 * 8 * 24  # selfplay_batch_size * max_num_steps
    learning_start: int = 8  # Iters of max_num_steps to prefill the buffer

    # Placeholders for dynamic values
    num_actions: int = -1
    is_state_vector: bool = False

    # Logging
    eval_interval: int = 25
    max_num_iters: int = 200


def run_experiment(args):
    """Run the full Risk MuZero training pipeline.

    Initialises the environment, network heads, optimiser, and replay buffer,
    then runs MCTS self-play to prefill the buffer before entering the main
    training loop.  Returns the JSON log dict and final network parameters.

    Args:
        args: ``Config`` instance with all hyperparameters.
    """
    # Make env
    env, is_state_vector, num_actions = make_env(
        env_name=args.env_name, use_legal_actions=args.use_legal_actions
    )
    args.num_actions = num_actions
    args.is_state_vector = is_state_vector

    # Create MCTS search function
    search_fn = partial(
        mctx.risk_search,
        utility_fn=partial(util.cvar, alpha=args.alpha_cvar),
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

    # --- JIT-compiled functions ---

    @jax.jit
    def root_fn(
        params: hk.Params,
        env_state: pgx.State,
        history_state: EnvHistory,
        step_count: jnp.ndarray,
    ):
        """Root function for MCTS search."""
        initial = env_state.terminated | env_state.truncated | step_count == 0  # (b,)

        # We auto reset during selfplay, but NOT during recurrent_fn.
        # Auto-reset means that we return an INITIAL STATE with a terminated flag - persisting
        # the terminated flag will corrupt several computations below so:
        # If this env_state is terminated, and we're using it as a root then we need
        # to manually reset the termination flag here.
        # I'm not going to say how long it took to find this bug :D
        def reset_root(env_state: pgx.State):
            """Reset the root state if it is terminated or truncated."""
            return jax.lax.cond(
                (env_state.terminated | env_state.truncated),
                lambda: env_state.replace(  # type: ignore
                    terminated=jnp.bool_(False),
                    truncated=jnp.bool_(False),
                    rewards=jnp.zeros_like(env_state.rewards),
                ),
                lambda: env_state,
            )

        env_state = jax.vmap(reset_root)(env_state)

        obs = history_state.obs[:, -1]
        obs_embedding = representation_apply.apply(params, obs)
        logits = policy_apply.apply(params, obs_embedding)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        value_dist = critic_apply.apply(params, obs_embedding)

        return_history_dist = reward_history_apply.apply(params, history_state.obs[:, 1:])
        return_history_dist = (1 - initial[:, None]) * return_history_dist  # (b, num_quantiles)
        return mctx.RiskRootFnOutput(
            return_history=return_history_dist,  # type: ignore
            prior_logits=logits,  # type: ignore
            value=value_dist,  # type: ignore
            embedding=(obs_embedding, step_count),  # type: ignore
        )

    @jax.jit
    def recurrent_fn(
        params: hk.Params,
        _rng_key: jnp.ndarray,
        action: jnp.ndarray,
        obs_embedding: tuple[jnp.ndarray, jnp.ndarray],
    ):
        """Expansion function for MCTS: apply dynamics model and predict value/policy."""
        embedding, step_count = obs_embedding
        next_obs_embedding, reward_dist = recurrent_inference.apply(params, embedding, action)

        logits = policy_apply.apply(params, next_obs_embedding)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        value_dist = critic_apply.apply(params, next_obs_embedding)

        return mctx.RecurrentFnOutput(
            prior_logits=logits,  # type: ignore
            value=value_dist,  # type: ignore
            reward=reward_dist,  # type: ignore
            discount=jnp.ones((reward_dist.shape[0],)) * args.discount,  # type: ignore
        ), (next_obs_embedding, step_count + 1)

    # Define eval loop
    @jax.jit
    def evaluate(rng_key: jnp.ndarray, params: hk.Params):
        """Evaluate the model by running selfplay and computing the average reward."""
        key, subkey = jax.random.split(rng_key)
        batch_size = args.eval_num_actors
        keys = jax.random.split(subkey, batch_size)
        state = jax.vmap(env.init)(keys)
        step_fn = jax.vmap(env.step)
        step = jnp.array(0)
        max_steps = 12 if "grid" in args.env_name else None

        key, subkey = jax.random.split(key)
        history_reset_keys = jax.random.split(subkey, batch_size)
        eval_history_state = make_batch_history(
            batch_size=args.eval_num_actors,
            num_before=args.history_length,
            num_actions=env.num_actions,
            obs_shape=env.observation_shape,
            gamma=args.discount,
        )
        eval_history_state = history_reset(
            eval_history_state, state.observation, history_reset_keys
        )  # type: ignore

        def cond_fn(
            tup: tuple[jax.Array, pgx.State, EnvHistory, jax.Array, jax.Array],
        ) -> jax.Array:
            """Loop while not all envs are done and all below max_steps."""
            _, state, _, _, step = tup
            still_running = ~state.terminated.all()
            if max_steps is None:
                return still_running
            return jnp.logical_and(still_running, step < max_steps).all()

        def body_fn(
            val,
        ) -> tuple[jax.Array, pgx.State, EnvHistory, jax.Array, jax.Array]:
            key, state, eval_history_state, R, step = val

            # Initialize the root
            step_count = jnp.full((batch_size,), step, dtype=jnp.int32)
            root = root_fn(params, state, eval_history_state, step_count)

            # Run MCTS search
            search_output = mctx.gumbel_muzero_policy(
                params=params,
                rng_key=key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=args.num_simulations,
                invalid_actions=~state.legal_action_mask,
                # Revisit: qtransform mix is very greedy
                qtransform=qtransform_fn,
                gumbel_scale=0.0,
                search_fn=search_fn,
            )
            action = search_output.action  # (b,)

            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, batch_size)
            state = step_fn(state, action, keys)

            # Update history
            eval_history_state = history_step(
                eval_history_state,
                state.observation,
                action,
                state.rewards[:, -1],
                state.terminated,
            )

            R = R + state.rewards[:, 0]
            return key, state, eval_history_state, R, step + 1

        _, _, _, R, _ = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (key, state, eval_history_state, jnp.zeros((batch_size,)), step),
        )
        R = R.reshape((-1,))
        R_mean = jnp.mean(R)
        R_cvar = util.cvar(R, alpha=0.25)
        return R_mean, R_cvar

    # Define selfplay loop
    def selfplay(
        rng_key: jax.Array,
        params: hk.Params,
        buffer_state: fbx.trajectory_buffer.TrajectoryBufferState,
        history_state: EnvHistory,
        gumbel_scale: jnp.ndarray,
        env_state: pgx.State,
        episode_stats: dict[str, jnp.ndarray],
    ) -> tuple[
        pgx.State,
        dict[str, jnp.ndarray],
        fbx.trajectory_buffer.TrajectoryBufferState,
        EnvHistory,
        ExItTransition,
    ]:
        """Collect `max_num_steps` of MCTS self-play and add to the replay buffer.

        When ``gumbel_scale >= 10`` (warmup), actions are sampled uniformly
        at random instead of running MCTS, to cheaply prefill the buffer.
        """

        @scan_tqdm(args.max_num_steps)
        def step_fn(
            carry: tuple[
                pgx.State,
                EnvHistory,
                dict[str, jnp.ndarray],
            ],
            iter_data: jnp.ndarray,
        ) -> tuple[tuple[pgx.State, EnvHistory, dict[str, jnp.ndarray]], ExItTransition]:
            state, history_state, episode_stats = carry
            _, key = iter_data
            key1, key2, key3 = jax.random.split(key, 3)  # (2,), (2,)
            observation = state.observation

            # Reset done histories (these envs have reset)
            history_reset_keys = jax.random.split(key3, state.observation.shape[0])
            history_state = history_reset_at_done(
                history_state,
                state.observation,
                history_reset_keys,
                state.terminated,
            )

            def search():
                # Initialize the root
                root = root_fn(
                    params,
                    state,
                    history_state,
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
                )
                action = search_output.action  # (b,)
                search_policy = search_output.action_weights
                return action, search_policy

            def random_selection():
                """Randomly select an action with equal probability."""
                action = jax.random.randint(
                    key1, (state.observation.shape[0],), 0, args.num_actions
                )
                search_policy = (
                    jnp.ones((state.observation.shape[0], args.num_actions)) / args.num_actions
                )
                return action, search_policy

            # If gumbel scale above 10, we assume this is a warmup step and
            # randomly pick an action with equal probability logits
            action, search_policy = jax.lax.cond(
                gumbel_scale >= WARMUP_GUMBEL_SCALE,
                random_selection,
                search,
            )  # type: ignore

            keys = jax.random.split(key2, state.observation.shape[0])
            state = jax.vmap(auto_reset(env.step, env.init))(state, action, keys)

            # Update history
            history_state = history_step(
                history_state,
                state.observation,
                action,
                state.rewards[:, -1],
                state.terminated,  # If this (next) state is terminal
            )

            # Create transition
            transition = ExItTransition(
                step_count=episode_stats["episode_length"],  # (b,)
                # This history is [...obs (history_length), next_obs]
                obs_history=history_state.obs,  # (b, history_length + 1, *obs_shape)
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
            return (state, history_state, episode_stats), transition

        # Run self-play for max_num_steps per batch
        rng_key, sub_key = jax.random.split(rng_key)
        key_seq = jax.random.split(sub_key, args.max_num_steps)

        (env_state, history_state, episode_stats), traj_batch = jax.lax.scan(
            step_fn,  # type: ignore
            (env_state, history_state, episode_stats),
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

        return env_state, episode_stats, buffer_state, history_state, traj_batch

    @jax.jit
    def selfplay_scan_fn(carry, iteration):
        """Scan function for self-play to prefill buffer."""
        (
            rng_key,
            buffer_state,
            history_state,
            opt_state,
            params,
            target_params,
            env_state,
            episode_stats,
            eval_reward,
        ) = carry

        # Run self-play
        rng_key, subkey = jax.random.split(rng_key)
        env_state, episode_stats, buffer_state, history_state, traj_batch = selfplay(
            subkey,
            params,
            buffer_state,
            history_state,
            jnp.asarray(
                WARMUP_GUMBEL_SCALE
            ),  # Flag for random action selection to prefill the buffer
            env_state,
            episode_stats,
        )

        return (
            rng_key,
            buffer_state,
            history_state,
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
            init_obs: jnp.ndarray,
            s_target: jnp.ndarray,
            policy_target: jnp.ndarray,
            a_seq: jnp.ndarray,
            r_target: jnp.ndarray,
            v_target: jnp.ndarray,
            dt: jnp.ndarray,
            obs_history: jnp.ndarray,
            historical_reward_targets: jnp.ndarray,
            initials: jnp.ndarray,
        ):
            # Generate root embeddings
            root_embeddings = representation_apply.apply(params, init_obs)

            @jax.remat  # type: ignore
            def unroll_fn(carry, targets):
                total_loss, obs_embedding, mask = carry
                (
                    a,
                    r_target,
                    s_target,
                    policy_target,
                    v_target,
                    done,
                    obs_history,
                    historical_reward_target,
                    initial,
                    step,
                ) = targets

                # HISTORICAL REWARD LOSS
                historical_reward = reward_history_apply.apply(params, obs_history)
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
                policy_logits_pred = policy_apply.apply(params, obs_embedding)
                policy_loss = optax.softmax_cross_entropy(
                    logits=policy_logits_pred, labels=policy_target
                )
                # Zero the policy loss for absorbing states
                policy_loss = policy_loss * mask

                # CRITIC LOSS
                # Zero the policy targets after done as absorbing
                v_target = v_target * mask[:, None]
                v_pred = critic_apply.apply(
                    params, obs_embedding
                )  # (b, num_quantiles) or (b, num_actions, num_quantiles)
                value_loss = batched_quantile_huber_loss(
                    dist_src=v_pred,
                    tau_src=tau_hats,
                    dist_target=v_target,
                    huber_param=args.huber_param,
                    stop_target_gradients=True,
                )
                chex.assert_shape(value_loss, (v_target.shape[0],))

                # Scale the gradients of the obs_embedding
                obs_embedding = scale_gradient(obs_embedding, 0.5)
                next_obs_embedding, reward_pred = recurrent_inference.apply(
                    params, obs_embedding, a
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
                next_obs_proj = projection_apply.apply(
                    params, next_obs_embedding
                )  # (b, num_hidden)
                unit_next_obs_proj = next_obs_proj / jnp.linalg.norm(
                    next_obs_proj, axis=-1, keepdims=True
                )  # (b, num_hidden)
                unit_s_target = s_target / jnp.linalg.norm(s_target, axis=-1, keepdims=True)
                sc_loss = 1 - jax.vmap(jnp.dot)(unit_next_obs_proj, unit_s_target)
                sc_loss = sc_loss * mask  # No self-consistency loss for absorbing states

                curr_loss = {
                    "actor_loss": policy_loss,
                    "value_loss": value_loss,
                    "reward_loss": reward_loss,
                    "historical_reward_loss": historical_reward_loss,
                    "self_consistency_loss": sc_loss,
                }
                total_loss = jax.tree_util.tree_map(
                    lambda x, y: x + y.mean(), total_loss, curr_loss
                )

                # Update the mask
                mask = jnp.where(done, jnp.zeros_like(mask), mask)
                return (total_loss, next_obs_embedding, mask), curr_loss

            targets = (
                a_seq,
                r_target,
                s_target,
                policy_target,
                v_target,
                dt,
                obs_history,
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
            }
            init_mask = jnp.ones((v_target.shape[0],))
            (losses, _, _), _ = jax.lax.scan(
                unroll_fn,
                (init_total_loss, root_embeddings, init_mask),
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
        batch = buffer_fn.sample(buffer_state, key).experience
        (policy_targets, reward_targets, value_targets, s_targets) = get_train_targets(
            batch,
            target_params,
            representation_apply,
            critic_apply,
            args,
        )

        # Compute value for each obs
        B, S = batch.obs.shape[:2]  # (b, t)
        obs = jnp.reshape(
            batch.obs_history, (-1, *batch.obs_history.shape[2:])
        )  # (b*t, *obs_shape)
        # Generate history targets
        initials = batch.step_count == 0  # (b, t)
        initials = initials.reshape((-1,))  # (b * t,)
        historical_reward_prev = jax.lax.stop_gradient(
            reward_history_apply.apply(params, obs[:, :-1])
        )  # (b, num_quantiles)
        rewards = batch.reward.reshape((-1,))  # (b * t)
        step_counts = batch.step_count.reshape((-1,))  # (b * t,)
        historical_reward_targets = ((args.discount) ** step_counts)[:, None] * rewards[:, None] + (
            1 - initials[:, None]
        ) * historical_reward_prev
        historical_reward_targets = historical_reward_targets.reshape((B, S, -1))
        initials = initials.reshape((B, S))
        historical_reward_targets = historical_reward_targets[:, : args.sample_sequence_length]

        # Each transition history is [...obs (history_length), next_obs] so we exclude next_obs.
        obs_history = jax.tree_util.tree_map(
            lambda x: x[:, 1 : args.sample_sequence_length + 1, :-1], batch.obs_history
        )
        initials = initials[:, 1 : args.sample_sequence_length + 1]

        (loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params,
            batch.obs[:, 0],
            s_targets[:, 1 : args.sample_sequence_length + 1],
            policy_targets,
            batch.action[:, : args.sample_sequence_length],
            reward_targets,
            value_targets,
            batch.done[:, : args.sample_sequence_length],
            obs_history,
            historical_reward_targets,
            initials,
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params=params, updates=updates)
        return new_params, new_opt_state, (loss, losses)

    def train_loop_body(carry, iteration):
        """Single training iteration: evaluation, self-play, and gradient steps."""
        (
            rng_key,
            buffer_state,
            history_state,
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
            R, R_cvar = evaluate(subkey, params)
            jax.debug.print(
                "Iter {i} / {max_num_iters}, Eval Reward: {r}, Eval CVaR: {r_cvar}",
                i=iteration,
                max_num_iters=args.max_num_iters,
                r=R.mean(),
                r_cvar=R_cvar.mean(),
            )
            return R.mean(), R_cvar.mean()

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
        env_state, episode_stats, buffer_state, history_state, traj_batch = selfplay(
            subkey,
            params,
            buffer_state,
            history_state,
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
            "Value Loss: {value_loss:.4f}, Self-consistency Loss: {self_consistency_loss:.4f}, Reward Loss: {reward_loss:.4f}, Historical Reward Loss: {historical_reward_loss:.4f}",
            i=iteration,
            max_num_iters=args.max_num_iters,
            loss=jnp.mean(loss),
            actor_loss=jnp.mean(losses["actor_loss"]),
            value_loss=jnp.mean(losses["value_loss"]),
            self_consistency_loss=jnp.mean(losses["self_consistency_loss"]),
            reward_loss=jnp.mean(losses["reward_loss"]),
            historical_reward_loss=jnp.mean(losses["historical_reward_loss"]),
        )

        carry = (
            rng_key,
            buffer_state,
            history_state,
            opt_state,
            params,
            target_params,
            env_state,
            episode_stats,
            eval_R,
        )

        R, R_cvar = eval_R
        json_log = {
            "num_steps": (iteration + 1) * args.max_num_steps,
            "training_steps": (iteration + 1) * args.train_epochs_per_iter,
            "iteration": iteration,
            "loss": jnp.mean(loss),
            "actor_loss": jnp.mean(losses["actor_loss"]),
            "value_loss": jnp.mean(losses["value_loss"]),
            "self_consistency_loss": jnp.mean(losses["self_consistency_loss"]),
            "reward_loss": jnp.mean(losses["reward_loss"]),
            "historical_reward_loss": jnp.mean(losses["historical_reward_loss"]),
            "train_average_return": average_return,
            "train_average_length": average_length,
            "gumbel_scale": gumbel_scale,
            "learning_rate": current_lr,
            "last_eval_reward": R,
            "last_eval_cvar": R_cvar,
        }
        return carry, json_log

    # Run scan
    rng_key = jax.random.PRNGKey(seed=args.seed)
    init_rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, args.selfplay_batch_size)
    env_state = jax.vmap(env.init)(keys)

    # Make history buffer
    history_state = make_batch_history(
        batch_size=args.selfplay_batch_size,
        num_before=args.history_length,
        num_actions=env.num_actions,
        obs_shape=env.observation_shape,
        gamma=args.discount,
    )
    history_reset_rng = jax.random.split(init_rng_key, args.selfplay_batch_size)
    history_state = history_reset(history_state, env_state.observation, history_reset_rng)  # type: ignore

    episode_stats_init = {
        "episode_return": jnp.zeros((args.selfplay_batch_size,)),
        "episode_length": jnp.zeros((args.selfplay_batch_size,), dtype=jnp.int32),
        "is_terminal_step": jnp.zeros((args.selfplay_batch_size,), dtype=bool),
    }

    initial_carry = (
        init_rng_key,
        buffer_state,
        history_state,
        opt_state,
        params,
        target_params,
        env_state,
        episode_stats_init,
        (jnp.array(0.0), jnp.array(0.0)),  # last_eval_reward
    )

    # Self-play to fill the buffer
    initial_carry, _traj_batch = jax.lax.scan(
        selfplay_scan_fn, initial_carry, jnp.arange(args.learning_start)
    )

    iterations = jnp.arange(args.max_num_iters)
    final_carry, json_logs = jax.lax.scan(train_loop_body, initial_carry, iterations)

    trained_params = final_carry[4]
    return json_logs, trained_params


if __name__ == "__main__":
    args = Config()
    print("Experiment started with config:")
    print(args)
    run_experiment(args)
