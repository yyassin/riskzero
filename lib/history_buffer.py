# Sliding-window observation history buffer
#
# Provides a fixed-length history buffer (EnvHistory) that tracks the last
# ``num_before + 1`` steps of an environment episode.  Used to supply
# sequence context to the GRU historical networks.
#
# Key operations
# --------------
# make_history        — allocate a zeroed single-env history
# make_batch_history  — allocate a batch of zeroed histories
# reset               — reinitialise history with a new starting observation
# reset_at_done       — conditionally reset (only where done=True)
# step                — slide the window forward by one timestep
#
# Batched (vmapped) aliases: history_reset, history_step, history_reset_at_done

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct
from jax import random, vmap


@struct.dataclass
class EnvHistory:
    """Fixed-length sliding-window history for a single environment.

    Stores the last ``num_before + 1`` timesteps of observations, actions,
    rewards, done flags, and discounted cumulative returns.

    Fields
    ------
    gamma      : Discount factor (static).
    num_before : Number of past steps retained before the current one (static).
    num_actions: Size of the discrete action space (static).
    obs        : Observation window, shape ``(T, *obs_shape)``.
    a          : Action window, shape ``(T,)``.
    r          : Reward window, shape ``(T,)``.
    done       : Done-flag window, shape ``(T,)``.
    acc_r      : Discounted cumulative return at each step, shape ``(T,)``.
    curr_step  : Current episode step counter, shape ``(1,)``.
    """

    gamma: float = struct.field(pytree_node=False)  # won't be traced
    num_before: int = struct.field(pytree_node=False)  # won't be traced
    num_actions: int = struct.field(pytree_node=False)  # won't be traced
    obs: jnp.ndarray  # (T, *obs_shape)
    a: jnp.ndarray  # (T,)
    r: jnp.ndarray  # (T,)
    done: jnp.ndarray  # (T,)
    acc_r: jnp.ndarray  # (T,)
    curr_step: jnp.ndarray  # (1,)


def make_history(
    num_before: int, num_actions: int, obs_shape: tuple[int, ...], gamma: float = 0.99
) -> EnvHistory:
    """Allocate a zeroed ``EnvHistory`` for a single environment.

    The history window length is ``T = num_before + 1`` (past steps plus the
    current step).  All array fields are initialized to zero.
    """
    T = num_before + 1
    return EnvHistory(
        gamma=gamma,  # Default discount factor
        num_before=num_before,
        num_actions=num_actions,
        obs=jnp.zeros((T,) + obs_shape),
        a=jnp.zeros((T,), dtype=jnp.int32),
        r=jnp.zeros((T,), dtype=jnp.float32),
        done=jnp.zeros((T,), dtype=jnp.bool_),
        acc_r=jnp.zeros((T,), dtype=jnp.float32),
        curr_step=jnp.zeros((1,), dtype=jnp.int32),  # Current step in the history
    )


@jax.jit
def reset(hist: EnvHistory, obs: jnp.ndarray, key) -> EnvHistory:
    """Reset the history with a new observation."""
    T = hist.obs.shape[0]
    obs_hist = jnp.tile(obs[None, ...], (T, *([1] * obs.ndim)))  # (T, *obs_shape)
    a = random.randint(key, shape=(T,), minval=0, maxval=hist.num_actions, dtype=jnp.int32)
    return hist.replace(  # type: ignore
        obs=obs_hist,
        a=a,
        r=jnp.zeros((T,), dtype=jnp.float32),
        done=jnp.zeros((T,), dtype=jnp.bool_),
        acc_r=jnp.zeros((T,), dtype=jnp.float32),
        curr_step=jnp.zeros((1,), dtype=jnp.int32),  # Reset current step
    )


@jax.jit
def reset_at_done(hist: EnvHistory, obs: jnp.ndarray, key, done: jnp.ndarray) -> EnvHistory:
    """Conditionally reset the history for environments that just terminated.

    For each element, selects ``reset_hist`` where ``done=True`` and keeps
    the existing ``hist`` otherwise.
    """
    reset_hist = reset(hist, obs, key)

    # Choose between `reset_hist` and `hist` based on `done`
    return hist.replace(  # type: ignore
        num_before=hist.num_before,
        num_actions=hist.num_actions,
        obs=jnp.where(done, reset_hist.obs, hist.obs),
        a=jnp.where(done, reset_hist.a, hist.a),
        r=jnp.where(done, reset_hist.r, hist.r),
        done=jnp.where(done, reset_hist.done, hist.done),
        acc_r=jnp.where(done, reset_hist.acc_r, hist.acc_r),
        curr_step=jnp.where(done, reset_hist.curr_step, hist.curr_step),
    )


@jax.jit
def step(hist: EnvHistory, obs, a, r, done) -> EnvHistory:
    """Slide the history window forward by one timestep.

    Drops the oldest entry and appends the new ``(obs, a, r, done)`` tuple.
    The discounted cumulative return is updated as:

        acc_r[t] = acc_r[t-1] + gamma^curr_step * r
    """
    obs_hist = jnp.concatenate([hist.obs[1:], obs[None, :]], axis=0)
    a_hist = jnp.concatenate([hist.a[1:], a[None]], axis=0)
    r_hist = jnp.concatenate([hist.r[1:], r[None]], axis=0)
    done_hist = jnp.concatenate([hist.done[1:], done[None]], axis=0)
    acc_r_hist = jnp.concatenate(
        [hist.acc_r[1:], hist.acc_r[-1] + hist.gamma**hist.curr_step * r[None]], axis=0
    )
    curr_step_hist = hist.curr_step + 1
    return hist.replace(  # type: ignore
        obs=obs_hist,
        a=a_hist,
        r=r_hist,
        done=done_hist,
        acc_r=acc_r_hist,
        curr_step=curr_step_hist,
    )


def make_batch_history(
    batch_size: int,
    num_before: int,
    num_actions: int,
    obs_shape: tuple[int, ...],
    gamma: float = 0.99,
) -> EnvHistory:
    """Allocate a batch of zeroed ``EnvHistory`` structs."""
    return jax.vmap(lambda _: make_history(num_before, num_actions, obs_shape, gamma))(
        jnp.arange(batch_size)
    )


# --- Batched (vmapped) variants ---

history_reset = vmap(reset)
history_step = vmap(step)
history_reset_at_done = vmap(reset_at_done)
