# QR-DQN training loop
#
# Implements the full QR-DQN training pipeline:
#   Config           — Pydantic config dataclass
#   run_experiment   — Top-level entry point
#
# Training pipeline (inside run_experiment)
# -----------------------------------------
# 1. Initialise environment, model, optimiser, and replay buffer.
# 2. Prefill the buffer with `learning_start` iterations of random play.
# 3. Run `max_num_iters` training iterations, each consisting of:
#      a. Evaluation (every `eval_interval` iterations)
#      b. Self-play data collection (`selfplay`)
#      c. `train_epochs_per_iter` gradient steps (`learning_step`)
#      d. Periodic target-network update
# All inner loops use `jax.lax.scan` for XLA compatibility.

import chex
import flashbax as fbx
import jax
import jax.numpy as jnp
import optax
import pgx
from pgx.experimental import auto_reset
from pydantic import BaseModel

import lib.util as util
from lib.make_env import make_env
from lib.quantile_losses import quantile_huber_loss
from src.baselines.qrdqn.network import create_qr_network
from src.baselines.qrdqn.util import (
    Transition,
    calc_eps,
    get_value_target,
    init_model_and_optim,
    make_buffer,
)


class Config(BaseModel):
    seeds: list[int] = []
    seed: int = 23
    env_name: str = "grid-risk"
    max_num_steps: int = 24
    max_num_iters: int = 1000
    eval_interval: int = 1
    eval_num_actors: int = 1024
    selfplay_batch_size: int = 32

    huber_param: float = 1.0
    buffer_batch_size: int = 1024
    buffer_size: int = 32 * 24 * 100
    epsilon_start: float = 1.0
    epsilon_finish: float = 0.05
    epsilon_anneal_time: int = int(500)
    learning_start: int = 100  # Iters of max_num_steps to prefill the buffer
    n_step: int = 5

    num_quantiles: int = 64
    alpha_cvar: float = 0.25  # CVaR level

    gamma: float = 1.0
    lr: float = 1e-4
    min_lr: float = 1e-5
    optim_eps: float = 1e-5
    lr_linear_decay: bool = True
    lr_anneal_iterations: int = 1000
    max_grad_norm: float = 0.5
    target_tau: float = 1.0
    target_update_interval: int = 10
    train_epochs_per_iter: int = 20

    # Placeholders for dynamic values
    num_actions: int = -1
    is_state_vector: bool = False


def run_experiment(args):
    """Run a full QR-DQN experiment and return logged metrics.

    Args:
        args: ``Config`` instance with all hyperparameters.

    Returns:
        json_logs: Dict of per-iteration metrics (loss, returns, epsilon, …).
        _params:   Final online network parameters.
    """
    # --- Initialization ---
    _env, is_state_vector, num_actions = make_env(env_name=args.env_name, use_legal_actions=False)
    args.num_actions = num_actions

    # Make and initialize the model
    _model = create_qr_network(
        num_actions=args.num_actions,
        num_quantiles=args.num_quantiles,
        is_state_vector=is_state_vector,
        alpha_cvar=args.alpha_cvar,
    )
    _params, _optimizer, _opt_state = init_model_and_optim(_env, _model, args)
    _target_params = jax.tree.map(lambda x: jnp.copy(x), _params)

    # Make trajectory buffer
    _buffer_fn, _buffer_state = make_buffer(_env, args)

    # Make tau_hats for quantile loss
    tau_hats = util.make_tau_hats(args.num_quantiles)

    # --- JIT-compiled functions ---

    @jax.jit
    def evaluate(rng_key: jnp.ndarray, params: optax.Params):
        """Evaluate the model by running selfplay and computing the average reward."""
        rng_key, subkey = jax.random.split(rng_key)
        batch_size = args.eval_num_actors
        keys = jax.random.split(subkey, batch_size)

        state = jax.vmap(_env.init)(keys)
        step_fn = jax.vmap(_env.step)
        ep_return = jnp.zeros_like(state.rewards)  # (num_eval_envs, )
        step = jnp.array(0)
        max_steps = 12 if "grid" in args.env_name else None

        def cond_fn(
            tup: tuple[pgx.State, jnp.ndarray, jax.Array, jnp.ndarray],
        ) -> jax.Array:
            """Loop while not all envs are done and all below max_steps."""
            state, _, _, step = tup
            still_running = ~state.terminated.all()
            if max_steps is None:
                return still_running
            return jnp.logical_and(still_running, step < max_steps).all()

        def loop_fn(
            tup: tuple[pgx.State, jnp.ndarray, jax.Array, jnp.ndarray],
        ) -> tuple[pgx.State, jnp.ndarray, jax.Array, jnp.ndarray]:
            state, R, rng_key, step = tup
            q_out = _model.apply(params, state.observation)
            greedy_action = jnp.argmax(q_out.q_values, axis=-1)

            rng_key, _rng = jax.random.split(rng_key)
            keys = jax.random.split(_rng, state.observation.shape[0])
            state = step_fn(state, greedy_action, keys)

            return state, R + state.rewards, rng_key, step + 1

        # Loop until all environments are done
        state, R, _, _ = jax.lax.while_loop(cond_fn, loop_fn, (state, ep_return, rng_key, step))
        R = R.reshape((-1,))
        R_mean = jnp.mean(R)
        R_cvar = util.cvar(R, alpha=0.25)
        return R_mean, R_cvar

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
        uniformly at random otherwise.  Environments are auto-reset on
        termination.  The resulting trajectory batch is added to the replay
        buffer before returning.

        Returns:
            env_state:     Updated environment state.
            episode_stats: Running episode statistics (reset on termination).
            buffer_state:  Updated replay buffer.
            traj_batch:    Collected transitions, shape ``(batch, time, ...)``.
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
            key1, key2 = jax.random.split(key)
            observation = state.observation

            # Epsilon-greedy action selection
            q_out = _model.apply(params, observation)  # (b, num_nodes)

            greedy_action = jnp.argmax(q_out.q_values, axis=-1)  # (b, num_nodes)
            random_action = jax.random.randint(
                key1, shape=greedy_action.shape, minval=0, maxval=args.num_actions
            )
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

            keys = jax.random.split(key2, observation.shape[0])
            state = jax.vmap(auto_reset(_env.step, _env.init))(state, action, keys)

            # Update episode stats
            episode_stats["episode_return"] += jnp.sum(state.rewards, axis=1)
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

        # Switch the time and batch axes
        traj_batch = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), traj_batch
        )  # (b, t, ...)
        # Add the batch of experience sequences to the buffer
        buffer_state = _buffer_fn.add(
            buffer_state,
            traj_batch,
        )  # type: ignore

        return env_state, episode_stats, buffer_state, traj_batch

    @jax.jit
    def selfplay_scan_fn(carry, iteration):
        """Scan wrapper around ``selfplay`` used during buffer prefill."""
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

        Samples a trajectory batch, computes n-step distributional targets,
        evaluates the quantile-Huber loss, and applies an Adam update.

        Returns:
            params:    Updated online network parameters.
            opt_state: Updated optimiser state.
            loss:      Scalar mean loss for this step.
        """
        learn_batch = _buffer_fn.sample(buffer_state, rng_key).experience

        obs = learn_batch.obs[:, 0]
        action = learn_batch.action[:, 0]
        q_next_target = get_value_target(
            learn_batch, _model, target_params, args
        )  # (batch_size, num_quantiles)

        def _loss_fn(params: optax.Params) -> jnp.ndarray:
            q_out = _model.apply(params, obs)

            chosen_action_q_dists = jnp.take_along_axis(
                q_out.q_dist,
                action[:, None, None],
                axis=-1,
            ).squeeze(-1)  # (batch_size, num_quantiles)

            losses = jax.vmap(quantile_huber_loss, in_axes=(0, None, 0, None, None))(
                chosen_action_q_dists,
                tau_hats,
                q_next_target,
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
        R, R_cvar = evaluate(subkey, params)
        jax.debug.print(
            "Iter {i} / {max_num_iters}, Eval Reward: {r}, Eval CVaR: {r_cvar}",
            i=iteration,
            max_num_iters=args.max_num_iters,
            r=R.mean(),
            r_cvar=R_cvar.mean(),
        )
        return R.mean(), R_cvar.mean()

    @jax.jit
    def train_loop_body(carry, iteration):
        """One full training iteration.

        Steps:
          1. Conditionally evaluate the current policy.
          2. Collect self-play data and add it to the replay buffer.
          3. Run ``train_epochs_per_iter`` gradient steps.
          4. Conditionally update the target network.
          5. Log and return metrics.

        Returns:
            carry:    Updated training state tuple.
            json_log: Dict of scalar metrics for this iteration.
        """
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
            learn_scan_fn, initial_carry, jnp.arange(args.train_epochs_per_iter)
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
        R, R_cvar = eval_R
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
            "last_eval_cvar": R_cvar,
        }
        return carry, json_log

    # --- Buffer prefill ---
    rng_key = jax.random.PRNGKey(seed=args.seed)
    init_rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, args.selfplay_batch_size)
    env_state = jax.vmap(_env.init)(keys)
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
        (jnp.array(0.0), jnp.array(0.0)),  # last_eval_reward
    )

    # Run self-play in a scan to prefill the buffer
    initial_carry, traj_batch = jax.lax.scan(
        selfplay_scan_fn, initial_carry, jnp.arange(args.learning_start)
    )

    # --- Main training loop ---
    iterations = jnp.arange(args.max_num_iters)
    _final_carry, json_logs = jax.lax.scan(train_loop_body, initial_carry, iterations)

    return json_logs, _params


if __name__ == "__main__":
    import os

    import numpy as onp

    def save_logs(logs):
        """
        Save logs to a specified file path.
        """
        log_file_path = "./logs/qrdqn/logger.log"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        numpy_logs = onp.array(logs)
        onp.save(log_file_path, numpy_logs)

    json_logs = run_experiment(Config())
    save_logs(json_logs)
