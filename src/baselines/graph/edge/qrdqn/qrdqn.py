# Graph QR-DQN training loop (edge-featured variant)
#
# Implements the full QR-DQN training pipeline for graph-structured
# environments (stochastic bipartite matching and max-independent-set).
# Differs from the flat QR-DQN in that:
#   - Observations are graph dicts (node features, edge features, senders,
#     receivers, aux scalars) rather than flat vectors or images.
#   - The replay buffer stores flat transitions (one step at a time)
#     since the partial solution conveys history.
#   - auto_reset_with_iteration is used so each episode advances to the
#     next problem instance in the dataset.
#   - Evaluation computes Monte-Carlo CVaR via edge-type simulation.
#
# Training pipeline (inside run_experiment)
# -----------------------------------------
# 1. Initialise environment, model, optimiser, and replay buffer.
# 2. Prefill the buffer with ``learning_start`` iterations of random play.
# 3. Run ``max_num_iters`` training iterations, each consisting of:
#      a. Evaluation (every ``eval_interval`` iterations)
#      b. Self-play data collection (``selfplay``)
#      c. ``train_epochs_per_iter`` gradient steps (``learning_step``)
#      d. Periodic target-network update

import chex
import flashbax as fbx
import jax
import jax.numpy as jnp
import optax
import pgx
from jax_tqdm import loop_tqdm  # type: ignore
from pydantic import BaseModel

import lib.util as util
from lib.auto_reset_with_iteration import auto_reset
from lib.make_env import make_env
from lib.quantile_losses import quantile_huber_loss
from src.baselines.graph.edge.qrdqn.network import create_graph_qr_network
from src.baselines.graph.edge.qrdqn.util import (
    Transition,
    calc_eps,
    init_model_and_optim_edge,
    make_buffer,
)
from src.datasets.sbm.cvar_estimation import estimate_distortion_risk


class Config(BaseModel):
    seeds: list[int] = []
    seed: int = 159
    env_name: str = "sbm:datasets/stochastic_bm/instances_60_30_180"
    max_num_steps: int = 32
    max_num_iters: int = 2000
    eval_interval: int = 25
    eval_num_actors: int = 1024
    selfplay_batch_size: int = 32

    hidden_size: int = 64
    num_quantiles: int = 64
    distortion: str = "cvar"  # cvar, pow, wang
    alpha_cvar: float = 1.0
    huber_param: float = 1.0

    buffer_batch_size: int = 256
    buffer_size: int = 32 * 32 * 64

    epsilon_start: float = 1.0
    epsilon_finish: float = 0.1
    epsilon_anneal_time: int = int(2_000)
    learning_start: int = 64  # Iters of max_num_steps to prefill the buffer

    gamma: float = 1.0
    lr: float = 1e-4
    min_lr: float = 1e-5
    optim_eps: float = 1e-5
    lr_linear_decay: bool = True
    lr_anneal_iterations: int = 2000  # 5000
    max_grad_norm: float = 0.5
    target_tau: float = 1.0
    target_update_interval: int = 5  # 1000
    train_epochs_per_iter: int = 20

    # Placeholders for dynamic values
    num_actions: int = -1
    is_state_vector: bool = False


def run_experiment(args):
    """Run a full Graph QR-DQN experiment and return logged metrics.

    Args:
        args: ``Config`` instance with all hyperparameters.

    Returns:
        json_logs: Dict of per-iteration metrics (loss, returns, epsilon, …).
        _params:   Final online network parameters.
    """
    # --- Initialization ---
    _env, _is_state_vector, num_actions = make_env(env_name=args.env_name, use_legal_actions=False)
    args.num_actions = num_actions  # type: ignore

    # Make and initialize the model
    _model = create_graph_qr_network(args)
    _params, _optimizer, _opt_state = init_model_and_optim_edge(_env, _model, args)
    _target_params = jax.tree.map(lambda x: jnp.copy(x), _params)

    # Make trajectory buffer
    _buffer_fn, _buffer_state = make_buffer(_env, args)

    # Make tau_hats for quantile loss
    tau_hats = util.make_tau_hats(args.num_quantiles)

    # Derive graph structure constants from the first instance.
    # num_l: left nodes (original + dummy); num_r: right nodes.
    # Right node indices in the graph run from num_l to num_l + num_r - 1.
    _node_types_0 = _env._instances.node_types[0]  # type: ignore
    _num_l = int(jnp.sum(_node_types_0 == 0))
    _num_r = int(jnp.sum(_node_types_0 == 1))

    # Define eval loop
    @jax.jit
    def evaluate(rng_key: jnp.ndarray, params: optax.Params):
        """Evaluate the model by running selfplay and computing the average reward."""
        rng_key, subkey = jax.random.split(rng_key)
        batch_size = 512
        keys = jax.random.split(subkey, batch_size)

        total_instances = _env._instances.num_nodes.shape[0] // 2  # type: ignore
        num_instances_per_batch = 512
        num_batches = (total_instances) // num_instances_per_batch
        assert total_instances % num_instances_per_batch == 0

        @loop_tqdm(num_batches)
        def eval_one_batch(batch_idx, carry):
            rng_key, acc_mean, acc_risk, acc_opt_cvar, acc_opt_expected = carry

            # Instances for this batch
            iteration = (
                jnp.arange(num_instances_per_batch, dtype=jnp.int32)
                + batch_idx * num_instances_per_batch
            )
            iteration = iteration.reshape((-1, 1))
            iteration = jnp.repeat(iteration, batch_size // iteration.shape[0], axis=1)
            iteration = iteration.reshape((-1,))

            offset = jnp.zeros(batch_size, dtype=jnp.int32)
            state = jax.vmap(_env.init_v2, in_axes=(0, 0, 0, None, None))(  # type: ignore
                keys, iteration, offset, 1, 1
            )

            # ---- optimal CVaR and optimal expected value for this batch ----
            opt_cvar_batch = jnp.mean(state._x.optimal_cvar_value)  # type: ignore
            opt_expected_batch = jnp.mean(state._x.cvar_100)  # type: ignore
            step_fn = jax.vmap(_env.step)
            ep_return = jnp.zeros_like(state.rewards)
            step = jnp.array(0)
            max_steps = _env._instances.num_nodes[0].astype(jnp.int32)  # type: ignore

            def cond_fn(tup):
                state, _, _, step, _ = tup
                still_running = ~state.terminated.all()
                if max_steps is None:
                    return still_running
                return jnp.logical_and(still_running, step < max_steps).all()

            def loop_fn(tup):
                state, R, key, step, actions = tup
                qr_out = _model.apply(
                    params,
                    state.observation["node_features"],  # type: ignore
                    state.observation["edge_features"],  # type: ignore
                    state.observation["senders"],  # type: ignore
                    state.observation["receivers"],  # type: ignore
                    state.observation["aux"],  # type: ignore
                )

                q_vals = jnp.where(
                    state.legal_action_mask,
                    qr_out.q_values,
                    jnp.full_like(qr_out.q_values, -1000.0),
                )
                greedy_action = jnp.argmax(q_vals, axis=-1)  # type: ignore

                key, _rng = jax.random.split(key)
                keys = jax.random.split(_rng, state.observation["node_features"].shape[0])
                state = step_fn(state, greedy_action, keys)

                # Write action into buffer at position `step`
                actions = actions.at[step].set(greedy_action)
                return state, R + state.rewards, key, step + 1, actions

            max_steps = _num_r
            selecting_fors = jnp.arange(_num_l, _num_l + _num_r)  # right-node indices
            actions_init = jnp.zeros((max_steps, batch_size), dtype=jnp.int32)
            carry_init = (state, ep_return, rng_key, step, actions_init)
            state, R, rng_key, _, actions_taken = jax.lax.while_loop(cond_fn, loop_fn, carry_init)

            actions_taken = actions_taken.swapaxes(0, 1)  # (b, max_steps)
            selecting_fors = jnp.broadcast_to(selecting_fors, actions_taken.shape)
            edge_indices = jax.vmap(_env.find_edge_indices)(  # type: ignore
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

            R_risks = jax.vmap(estimate_distortion_risk, in_axes=(0, 0, None, None, None))(
                jax.random.split(rng_key, batch_size),
                edge_types,
                10000,
                0.25,
                args.distortion,
            )
            R = jax.vmap(estimate_distortion_risk, in_axes=(0, 0, None, None))(
                jax.random.split(rng_key, batch_size),
                edge_types,
                10000,
                1.0,
            )
            R_mean = jnp.mean(R)
            R_risk = jnp.mean(R_risks)

            return (
                rng_key,
                acc_mean + R_mean,
                acc_risk + R_risk,
                acc_opt_cvar + opt_cvar_batch,
                acc_opt_expected + opt_expected_batch,
            )

        # Run across all batches
        rng_key, total_mean, total_risk, total_opt_cvar, total_opt_expected = jax.lax.fori_loop(
            0, num_batches, eval_one_batch, (rng_key, 0.0, 0.0, 0.0, 0.0)
        )

        # Average across batches
        R_mean = total_mean / num_batches
        R_risk = total_risk / num_batches
        R_opt_cvar = total_opt_cvar / num_batches
        R_opt_expected = total_opt_expected / num_batches
        return R_mean, R_risk, R_opt_cvar, R_opt_expected

    @jax.jit
    def selfplay(
        rng_key: jax.Array,
        params: optax.Params,
        buffer_state: fbx.trajectory_buffer.TrajectoryBufferState,
        env_state: pgx.State,
        episode_stats: dict[str, jnp.ndarray],
        step_num: int,
    ):
        """Collect ``max_num_steps`` of epsilon-greedy experience and add to buffer.

        Actions are selected greedily with probability ``1 - epsilon`` and
        uniformly at random from legal actions otherwise. Environments are
        auto-reset on termination, advancing to the next problem instance.
        Transitions are added to the flat replay buffer one step at a time.

        Returns:
            env_state:     Updated environment state.
            episode_stats: Running episode statistics (reset on termination).
            buffer_state:  Updated replay buffer.
            traj_batch:    Collected transitions, shape ``(time, batch, ...)``.
        """

        def step_fn(
            carry: tuple[
                pgx.State,
                dict[str, jnp.ndarray],
                fbx.trajectory_buffer.TrajectoryBufferState,
            ],
            iter_data: jnp.ndarray,
        ) -> tuple[
            tuple[
                pgx.State,
                dict[str, jnp.ndarray],
                fbx.trajectory_buffer.TrajectoryBufferState,
            ],
            Transition,
        ]:
            state, episode_stats, buffer_state = carry
            _, key = iter_data
            key1, key2 = jax.random.split(key)  # (2,), (2,)
            observation = state.observation

            # Epsilon-greedy action selection
            qr_out = _model.apply(
                params,
                observation["node_features"],
                observation["edge_features"],
                observation["senders"],
                observation["receivers"],
                observation["aux"],
            )  # (b, num_nodes)

            # Mask out illegal actions if needed
            q_vals: jnp.ndarray = jnp.where(
                state.legal_action_mask,
                qr_out.q_values,
                jnp.full_like(qr_out.q_values, -100.0),
            )  # type: ignore
            greedy_action = jnp.argmax(q_vals, axis=-1)

            noise = jax.random.uniform(key1, q_vals.shape)  # (b, num_nodes)
            masked_noise = jnp.where(
                state.legal_action_mask, noise, jnp.full_like(noise, -1.0)
            )  # (b, num_nodes)
            random_action = jnp.argmax(masked_noise, axis=-1)  # (b)

            epsilon = calc_eps(
                step_num,
                args.epsilon_start,
                args.epsilon_finish,
                args.epsilon_anneal_time,
            )  # (b,)
            action = jax.lax.cond(
                jax.random.uniform(key2) < epsilon,
                lambda: random_action,
                lambda: greedy_action,
            )

            keys = jax.random.split(key2, state.observation["node_features"].shape[0])
            state = jax.vmap(auto_reset(_env.step, _env.init_v2))(state, action, keys)  # type: ignore

            # Update episode stats
            episode_stats["episode_return"] += state.rewards[:, -1]
            episode_stats["episode_length"] += 1
            episode_stats["is_terminal_step"] = state.terminated

            # Create transition
            transition = Transition(
                done=state.terminated,  # (b,)
                action=jnp.asarray(action),  # (b,)
                reward=state.rewards[:, 0],  # (b,)
                obs=observation,  # (b, *obs_shape)
                info=episode_stats,
            )
            buffer_state = _buffer_fn.add(
                buffer_state,
                transition,
            )  # Add transition to the buffer

            # Reset stats for terminal steps
            episode_stats = jax.tree_util.tree_map(
                lambda x: jnp.where(state.terminated, jnp.zeros_like(x), x),
                episode_stats,
            )
            return (state, episode_stats, buffer_state), transition

        # Run self-play for max_num_steps per batch
        rng_key, sub_key = jax.random.split(rng_key)
        key_seq = jax.random.split(sub_key, args.max_num_steps)

        (env_state, episode_stats, buffer_state), traj_batch = jax.lax.scan(
            step_fn,  # type: ignore
            (env_state, episode_stats, buffer_state),
            (jnp.arange(args.max_num_steps), key_seq),  # type: ignore
        )

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
            eval_R,
        ) = carry

        # Run self-play
        (
            env_state,
            episode_stats,
            buffer_state,
            traj_batch,
        ) = selfplay(rng_key, params, buffer_state, env_state, episode_stats, iteration)

        return (
            rng_key,
            buffer_state,
            opt_state,
            params,
            target_params,
            env_state,
            episode_stats,
            eval_R,
        ), traj_batch

    @jax.jit
    def learning_step(rng_key, params, target_params, opt_state, buffer_state):
        """Perform one gradient step on a batch sampled from the replay buffer.

        Samples a flat transition batch, computes 1-step distributional targets
        using the target network (Double DQN style), evaluates the
        quantile-Huber loss, and applies an Adam update.

        Returns:
            params:    Updated online network parameters.
            opt_state: Updated optimiser state.
            loss:      Scalar mean loss for this step.
        """
        learn_batch = _buffer_fn.sample(buffer_state, rng_key).experience
        qr_next_out = _model.apply(
            target_params,
            learn_batch.second.obs["node_features"],
            learn_batch.second.obs["edge_features"],
            learn_batch.second.obs["senders"],
            learn_batch.second.obs["receivers"],
            learn_batch.second.obs["aux"],
        )  # (b, num_nodes)
        greedy_actions = jnp.argmax(qr_next_out.q_values, axis=-1)
        qr_next_target = jnp.take_along_axis(
            qr_next_out.q_dist,
            jnp.expand_dims(greedy_actions, axis=(-1, -2)),  # (b, num_quantiles, 1)
            axis=-1,
        ).squeeze(-1)

        qr_next_target = (
            learn_batch.first.reward[:, None]
            + (1 - learn_batch.first.done[:, None]) * args.gamma * qr_next_target
        )  # (b, )

        def _loss_fn(params: optax.Params) -> jnp.ndarray:
            qr_out = _model.apply(
                params,
                learn_batch.first.obs["node_features"],
                learn_batch.first.obs["edge_features"],
                learn_batch.first.obs["senders"],
                learn_batch.first.obs["receivers"],
                learn_batch.first.obs["aux"],
            )
            chosen_action_q_dists = jnp.take_along_axis(
                qr_out.q_dist, learn_batch.first.action[:, None, None], axis=-1
            ).squeeze(-1)

            losses = jax.vmap(quantile_huber_loss, in_axes=(0, None, 0, None, None))(
                chosen_action_q_dists,
                tau_hats,
                qr_next_target,
                args.huber_param,
                True,
            )

            chex.assert_shape(losses, (args.buffer_batch_size,))
            loss = jnp.mean(losses)
            return loss

        loss, grads = jax.value_and_grad(_loss_fn)(params)
        updates, opt_state = _optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    # Define scan function for training
    @jax.jit
    def learn_scan_fn(carry, epoch):
        """Scan function for training."""
        rng_key, params, target_params, opt_state, buffer_state = carry
        rng_key, subkey = jax.random.split(rng_key)

        # Perform a learning step
        params, opt_state, loss = learning_step(
            subkey, params, target_params, opt_state, buffer_state
        )

        # Update target parameters
        target_params = jax.lax.cond(
            (epoch + 1) % args.target_update_interval == 0,
            lambda: jax.tree_util.tree_map(
                lambda target, online: args.target_tau * online + (1 - args.target_tau) * target,
                target_params,
                params,
            ),
            lambda: target_params,
        )

        return (rng_key, params, target_params, opt_state, buffer_state), loss

    @jax.jit
    def eval_fn(subkey, iteration, params):
        R, R_risk, R_opt_cvar, R_opt_expected = evaluate(subkey, params)
        jax.debug.print(
            "Iter {i} / {max_num_iters}, Eval Reward: {r}, Eval {distortion} risk: {r_risk} ({r_opt_cvar}), Opt Expected: {r_opt_expected}",
            i=iteration,
            max_num_iters=args.max_num_iters,
            r=R.mean(),
            distortion=args.distortion,
            r_risk=R_risk.mean(),
            r_opt_cvar=R_opt_cvar.mean(),
            r_opt_expected=R_opt_expected.mean(),
        )
        return R.mean(), R_risk.mean(), R_opt_expected.mean()

    @jax.jit
    def train_loop_body(carry, iteration):
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

        # Training step
        rng_key, subkey = jax.random.split(rng_key)

        # Run evaluation conditionally
        eval_R = jax.lax.cond(
            iteration % args.eval_interval == 0,
            eval_fn,
            lambda *_: last_eval_reward,
            subkey,
            iteration,
            params,
        )

        # Self-play data collection
        env_state, episode_stats, buffer_state, traj_batch = selfplay(
            subkey,
            params,
            buffer_state,
            env_state,
            episode_stats,
            iteration,
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

        # Perform learning steps
        initial_carry = (rng_key, params, target_params, opt_state, buffer_state)
        (rng_key, params, target_params, opt_state, buffer_state), losses = jax.lax.scan(
            learn_scan_fn,
            initial_carry,
            iteration * args.train_epochs_per_iter + jnp.arange(args.train_epochs_per_iter),
        )

        # Log losses
        current_lr = util.linear_schedule(
            iteration * args.train_epochs_per_iter,
            args.lr,
            args.lr_anneal_iterations * args.train_epochs_per_iter,
            args.min_lr,
        )
        current_epsilon = calc_eps(
            iteration,
            args.epsilon_start,
            args.epsilon_finish,
            args.epsilon_anneal_time,
        )
        jax.debug.print(
            "Iter {i} / {max_num_iters}, Loss: {loss:.4f} (LR: {lr:.6f}, EPS: {eps:.4f}), Train Avg Return: {avg_return:.2f}, Avg Length: {avg_length:.2f}",
            i=iteration,
            max_num_iters=args.max_num_iters,
            loss=jnp.mean(losses),
            lr=current_lr,
            eps=current_epsilon,
            avg_return=average_return,
            avg_length=average_length,
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
            "loss": jnp.mean(losses),
            "train_average_return": average_return,
            "train_average_length": average_length,
            "epsilon": current_epsilon,
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
    env_state = jax.vmap(_env.init_v2, in_axes=(0, 0, 0, None, None))(  # type: ignore
        keys, iteration, offset, args.selfplay_batch_size, 0
    )
    episode_stats_init = {
        "episode_return": jnp.zeros((args.selfplay_batch_size,)),
        "episode_length": jnp.zeros((args.selfplay_batch_size,), dtype=jnp.int32),
        "is_terminal_step": jnp.zeros((args.selfplay_batch_size,), dtype=bool),
    }
    initial_carry = (
        init_rng_key,
        _buffer_state,
        _opt_state,
        _params,
        _target_params,
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

    return json_logs, _params


if __name__ == "__main__":
    import os

    import numpy as onp

    def save_logs(logs):
        """
        Save logs to a specified file path.
        """
        log_file_path = "./logs/graph_qrdqn/sbm/logger.log"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        numpy_logs = onp.array(logs)
        onp.save(log_file_path, numpy_logs)

    json_logs = run_experiment(Config())
    save_logs(json_logs)
