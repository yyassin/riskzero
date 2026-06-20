# Graph QR-DQN utilities (node-only variant)
#
# Shared helpers used by the graph node-only QR-DQN training loop.
#
# Contents
# --------
# Epsilon schedule:  calc_eps
# Data structures:   Transition
# Setup helpers:     init_model_and_optim, make_buffer

from functools import partial
from typing import Any, NamedTuple

import flashbax as fbx
import jax
import jax.numpy as jnp
import optax

import lib.util as util


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: dict[str, Any] | None = None  # Additional info, e.g., episode stats


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


def init_model_and_optim(env, model_init_fn, args):
    """Initialise model parameters and an Adam optimiser for node-only graphs.

    Constructs a dummy observation (node features only, no edge features) to
    trace the model, then builds an optax optimiser with optional linear LR
    decay.

    Returns:
        params:    Initial network parameters.
        optimizer: Optax optimiser (gradient clipping + Adam).
        opt_state: Initial optimiser state.
    """
    key1, key2 = jax.random.split(jax.random.PRNGKey(args.seed))
    obs = env.init(key1).observation
    obs = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], obs)

    params = model_init_fn.init(
        key2, obs["node_features"], obs["senders"], obs["receivers"], obs["aux"]
    )

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
    """Create and initialise a Flashbax flat replay buffer.

    The buffer stores individual transitions (one step at a time) rather than
    full trajectories.  All buffer operations are JIT-compiled.

    Returns:
        buffer_fn:    Flashbax buffer object with JIT-compiled methods.
        buffer_state: Initial (empty) buffer state.
    """
    buffer_fn = fbx.make_flat_buffer(
        max_length=args.buffer_size,
        min_length=args.buffer_batch_size,
        sample_batch_size=args.buffer_batch_size,
        add_sequences=False,
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
        obs=jax.tree_util.tree_map(
            lambda x: x[0],
            dummy_state.observation,
        ),
        info={
            "episode_return": jnp.array(0.0),
            "episode_length": jnp.array(0),
            "is_terminal_step": jnp.array(False),
        },
    )
    buffer_state = buffer_fn.init(dummy_transition)

    return buffer_fn, buffer_state
