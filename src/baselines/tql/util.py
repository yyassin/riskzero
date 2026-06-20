# TQL utilities
#
# Shared helpers used by the TQL training loop.
#
# Contents
# --------
# Data structures:   Transition
# Setup helpers:     init_model_and_optim, make_buffer
# Training helpers:  get_value_targets
#
# TQL (Trajectory Q-Learning) uses two network heads trained jointly:
# a QR-DQN head for single-observation action values of the future return,
# and a GRU-based TQL head that conditions on an observation history window
# to estimate the trajectory-level return distribution.

from functools import partial
from typing import NamedTuple

import flashbax as fbx
import jax
import jax.numpy as jnp
import optax

import lib.util as util


class Transition(NamedTuple):
    obs_history: jnp.ndarray  # History of observations
    act_history: jnp.ndarray  # History of actions
    done: jnp.ndarray  # If the final step in the sequence is done
    reward: jnp.ndarray  # The reward at the final transition
    discounted_accumulated_return: (
        jnp.ndarray
    )  # The discounted return up to second to last transition
    info: dict[str, jnp.ndarray]  # Additional info, e.g. episode return, length, etc.


def init_model_and_optim(env, model_init_fn, args):
    """Initialise model parameters and an Adam optimiser.

    Constructs a dummy observation repeated over ``history_length`` steps to
    initialise the joint (QR + TQL) model in a single ``init`` call.

    The learning rate is either constant or linearly decayed to ``min_lr``
    over ``lr_anneal_iterations * train_epochs_per_iter`` steps, depending
    on ``args.lr_linear_decay``.

    Returns:
        params:    Initial network parameters.
        optimizer: Optax optimiser (gradient clipping + Adam).
        opt_state: Initial optimiser state.
    """
    key1, key2 = jax.random.split(jax.random.PRNGKey(args.seed))
    obs = env.init(key1).observation
    obs = obs[jnp.newaxis, ...]
    # Repeat the obs over history length
    obs = jnp.repeat(obs, args.history_length, axis=0)

    params = model_init_fn.init(key2, obs[jnp.newaxis, ...])

    lr = (
        partial(
            util.linear_schedule,
            num_updates=args.lr_anneal_iterations * args.train_epochs_per_iter,
            lr=args.lr,
            min_lr=args.min_lr,
        )
        if args.lr_linear_decay
        else args.lr
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(learning_rate=lr, eps=args.optim_eps),  # type: ignore
    )
    opt_state = optimizer.init(params)
    return params, optimizer, opt_state


def make_buffer(env, args):
    """Create and initialise a Flashbax trajectory replay buffer.

    The buffer stores full ``n_step``-length trajectories (including
    observation history and action history windows) to support n-step returns.
    All buffer operations are JIT-compiled.

    Returns:
        buffer_fn:    Flashbax buffer object with JIT-compiled methods.
        buffer_state: Initial (empty) buffer state.
    """
    buffer_fn = fbx.make_trajectory_buffer(
        max_size=args.buffer_size,
        min_length_time_axis=args.n_step + 1,
        sample_batch_size=args.buffer_batch_size,
        sample_sequence_length=args.n_step + 1,
        period=1,
        add_batch_size=args.selfplay_batch_size,
    )
    buffer_fn = buffer_fn.replace(  # type: ignore
        init=jax.jit(buffer_fn.init),
        add=jax.jit(buffer_fn.add, donate_argnums=0),
        sample=jax.jit(buffer_fn.sample),
        can_sample=jax.jit(buffer_fn.can_sample),
    )

    # Make trajectory buffer
    dummy_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 1))
    dummy_transition = Transition(
        obs_history=jnp.repeat(
            dummy_state.observation,  # (1, *obs_shape)
            args.history_length + 1,
            axis=0,
        ),
        act_history=jnp.zeros((args.history_length + 1,), dtype=jnp.int32),
        done=jnp.array(False),
        reward=jnp.array(0.0),
        discounted_accumulated_return=jnp.array(0.0),
        info={
            "episode_return": jnp.array(0.0),
            "episode_length": jnp.array(0),
            "is_terminal_step": jnp.array(False),
        },
    )
    buffer_state = buffer_fn.init(dummy_transition)

    return buffer_fn, buffer_state


# --- Training Helpers ---


def get_value_targets(learn_batch, _qr_model, _tql_model, target_params, args):
    """Compute n-step distributional value targets for TQL.

    Produces targets for both network heads:

    - ``target_qr``: bootstrapped quantile targets for the QR head, computed
      using the TQL head's greedy action (Double DQN style) and an n-step
      Bellman backup.
    - ``target_tql``: targets for the TQL head formed by combining the
      discounted accumulated return from the history window with the
      bootstrapped QR target.

    Returns:
        target_qr:  Shape ``(batch, num_quantiles)``.
        target_tql: Shape ``(batch, num_quantiles)``.
    """
    # Compute the target
    batch = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(x, 0, 1), learn_batch
    )  # (b, t, *) -> (t, b, *)

    # Last step doesn't reset since we bootstrap from it (the preceding done is the reset if needed)
    # Setting to false here for logging clarity, this value is not read!
    dones = batch.done.at[-1].set(False)
    qr_next_out = _qr_model.apply(
        target_params,
        batch.obs_history[-1, :, -2],  # Second to last observation in the history is s
    )
    tql_next_out = _tql_model.apply(
        target_params,
        batch.obs_history[-1, :, :-1],  # Use history up to s
    )
    greedy_actions = jnp.argmax(tql_next_out.q_values, axis=-1)  # (b, )
    qr_next_target = jnp.take_along_axis(
        qr_next_out.q_dist,
        greedy_actions[:, None, None],  # (b, num_quantiles, 1)
        axis=-1,
    ).squeeze(-1)  # (b, num_quantiles)

    def body_fn(carry, i):
        ix = args.n_step - i - 1  # Reverse index for backward pass
        v = (
            batch.reward[ix, :, None] + (1 - dones[ix, :, None]) * args.gamma * carry
        )  # (batch_size,)
        return v, v

    _, qr_next_target = jax.lax.scan(
        body_fn,
        qr_next_target,
        jnp.arange(args.n_step),
    )
    target_qr = qr_next_target[-1, :]

    target_tql = (
        learn_batch.discounted_accumulated_return[:, 0, jnp.newaxis]  # (b, 1)
        + (1 - learn_batch.done[:, 0, jnp.newaxis])
        * (args.gamma ** learn_batch.info["episode_length"][:, 0, jnp.newaxis])
        * (target_qr - learn_batch.reward[:, 0, jnp.newaxis])
    )  # (b, num_actions)
    return target_qr, target_tql  # (b, num_quantiles), (b, num_quantiles)
