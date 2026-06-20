# QR-DQN utilities
#
# Shared helpers used by the QR-DQN training loop.
#
# Contents
# --------
# Epsilon schedule:  calc_eps
# Data structures:   Transition
# Setup helpers:     init_model_and_optim, make_buffer
# Training helpers:  get_value_target

from functools import partial
from typing import Any, NamedTuple

import flashbax as fbx
import jax
import jax.numpy as jnp
import optax

import lib.util as util

# ---------------------------------------------------------------------------
# Epsilon Schedule
# ---------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=(
        "epsilon_start",
        "epsilon_finish",
        "epsilon_anneal_time",
    ),
)
def calc_eps(
    t: int,
    epsilon_start: float,
    epsilon_finish: float,
    epsilon_anneal_time: float,
) -> jnp.ndarray:
    """Linearly anneal epsilon from ``epsilon_start`` to ``epsilon_finish``.

    Annealing begins at ``t=0`` and completes at ``t=epsilon_anneal_time``.
    The value is clipped so it never falls below ``epsilon_finish``.
    """
    return jnp.clip(
        ((epsilon_finish - epsilon_start) / epsilon_anneal_time)
        * (jnp.maximum(0, t))  # Only anneal after learning starts
        + epsilon_start,
        epsilon_finish,
    )


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: dict[str, Any] | None = None  # Additional info, e.g., episode stats


# ---------------------------------------------------------------------------
# Setup Helpers
# ---------------------------------------------------------------------------


def init_model_and_optim(env, model_init_fn, args):
    """Initialise model parameters and an Adam optimiser.

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

    The buffer stores full ``n_step``-length trajectories to support
    n-step returns.  All buffer operations are JIT-compiled.

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
        done=jnp.array(False),
        action=jnp.array(0),
        reward=jnp.array(0.0),
        obs=dummy_state.observation[0],
        info={
            "episode_return": jnp.array(0.0),
            "episode_length": jnp.array(0),
            "is_terminal_step": jnp.array(False),
        },
    )
    buffer_state = buffer_fn.init(dummy_transition)

    return buffer_fn, buffer_state


# ---------------------------------------------------------------------------
# Training Helpers
# ---------------------------------------------------------------------------


def get_value_target(learn_batch, _model, target_params, args):
    """Compute n-step distributional value targets for QR-DQN.

    Uses the target network with a greedy action selection (Double DQN style)
    and accumulates rewards over ``args.n_step`` steps via a backward scan.

    Returns:
        q_next_target: Target quantile values, shape ``(batch, num_quantiles)``.
    """
    batch = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(x, 0, 1), learn_batch
    )  # (b, t, *) -> (t, b, *)
    # Last step doesn't reset since we bootstrap from it (the preceding done is the reset if needed)
    # Setting to false here for logging clarity, this value is not read!
    dones = batch.done.at[-1].set(False)

    q_next_out = _model.apply(target_params, batch.obs[-1])  # (b, num_quantiles, num_actions)
    greedy_actions = jnp.argmax(q_next_out.q_values, axis=-1)  # (b, )
    q_next_target = jnp.take_along_axis(
        q_next_out.q_dist,
        jnp.expand_dims(greedy_actions, axis=(-1, -2)),  # (b, num_quantiles, 1)
        axis=-1,
    ).squeeze(-1)  # (b, num_quantiles)

    def body_fn(carry, i):
        ix = args.n_step - i - 1  # Reverse index for backward pass
        v = (
            batch.reward[ix, :, None] + (1 - dones[ix, :, None]) * args.gamma * carry
        )  # (b, num_quantiles)
        return v, v

    _, q_next_target = jax.lax.scan(
        body_fn,
        q_next_target,
        jnp.arange(args.n_step),
    )
    q_next_target = q_next_target[-1, :]
    return q_next_target  # (b, num_quantiles)
