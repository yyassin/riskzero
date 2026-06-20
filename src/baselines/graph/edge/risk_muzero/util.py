# Risk MuZero (graph/edge) utilities
#
# Shared helpers used by the graph-edge Risk MuZero training loop.
#
# Contents
# --------
# Data structures:   ExItTransition
# Setup helpers:     scale_gradient, init_model_and_optim, make_buffer
# Training helpers:  batch_n_step_bootstrapped_returns, get_train_targets

from functools import partial
from typing import Mapping, NamedTuple

import chex
import flashbax as fbx
import jax
import jax.numpy as jnp
import optax

import lib.util as util


class ExItTransition(NamedTuple):
    step_count: jnp.ndarray  # (b,)  # Step count (distance from the root)
    done: jnp.ndarray  # (b,)
    action: jnp.ndarray  # (b,)
    reward: jnp.ndarray  # (b,)
    search_policy: jnp.ndarray  # (b, num_actions)
    obs: jnp.ndarray  # (b, *obs_shape)
    info: Mapping[
        str, float | int | bool | jnp.ndarray
    ]  # Additional info, e.g., episode return, length


def scale_gradient(g: jnp.ndarray, scale: float = 1) -> jnp.ndarray:
    """Scales the gradient of `g` by `scale` but keeps the original value unchanged."""
    return g * scale + jax.lax.stop_gradient(g) * (1.0 - scale)


def init_model_and_optim(env, model_init_fn, args):
    """Initialise model parameters and an Adam optimiser.

    Constructs dummy inputs matching the graph-edge observation structure
    and calls the joint ``init_model_fn`` in a single pass.

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
    obs = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], obs)

    dummy_action = jax.random.randint(jax.random.PRNGKey(args.seed), (1,), 0, env.num_actions)
    params = model_init_fn.init(
        key2,
        obs["node_features"],
        obs["edge_features"],
        obs["senders"],
        obs["receivers"],
        obs["aux"],
        dummy_action,
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


def make_buffer(
    env, args
) -> tuple[fbx.trajectory_buffer.TrajectoryBuffer, fbx.trajectory_buffer.TrajectoryBufferState]:
    """Create and initialise a Flashbax trajectory replay buffer.

    Stores full ``ExItTransition`` trajectories including graph observations
    and MCTS-improved search policies.  All buffer operations are JIT-compiled.

    Returns:
        buffer_fn:    Flashbax buffer object with JIT-compiled methods.
        buffer_state: Initial (empty) buffer state.
    """
    # Make trajectory buffer
    dummy_state = jax.vmap(env.init)(
        jax.random.split(jax.random.PRNGKey(0), 1)
    )  # (1, *env.observation_shape)
    dummy_transition = ExItTransition(
        step_count=jnp.array(0),
        done=jnp.array(False),
        action=jnp.array(0),
        reward=jnp.array(0.0),
        search_policy=jnp.zeros((args.num_actions,)),
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
    buffer_fn = fbx.make_trajectory_buffer(
        max_size=args.total_buffer_size,
        min_length_time_axis=args.sample_sequence_length + args.n_step,
        sample_batch_size=args.train_batch_size,
        sample_sequence_length=args.sample_sequence_length + args.n_step,
        period=1,
        add_batch_size=args.selfplay_batch_size,
    )
    buffer_fn = buffer_fn.replace(  # type: ignore
        init=jax.jit(buffer_fn.init),
        add=jax.jit(buffer_fn.add, donate_argnums=0),
        sample=jax.jit(buffer_fn.sample),
        can_sample=jax.jit(buffer_fn.can_sample),
    )
    buffer_state = buffer_fn.init(dummy_transition)
    return buffer_fn, buffer_state


def batch_n_step_bootstrapped_returns(
    r_t: jax.Array,
    discount_t: jax.Array,
    v_t: jax.Array,
    n: int,
    lambda_t: float = 1.0,
    stop_target_gradients: bool = True,
) -> jax.Array:
    """Stolen from rlax.
    Computes strided n-step bootstrapped return targets over a batch of sequences.

    The returns are computed according to the below equation iterated `n` times:

        Gₜ = rₜ₊₁ + γₜ₊₁ [(1 - λₜ₊₁) vₜ₊₁ + λₜ₊₁ Gₜ₊₁].

    When lambda_t == 1. (default), this reduces to

        Gₜ = rₜ₊₁ + γₜ₊₁ * (rₜ₊₂ + γₜ₊₂ * (... * (rₜ₊ₙ + γₜ₊ₙ * vₜ₊ₙ ))).

    Args:
        r_t: rewards at times B x [1, ..., T].
        discount_t: discounts at times B x [1, ..., T].
        v_t: state or state-action values to bootstrap from at time B x [1, ...., T].
        n: number of steps over which to accumulate reward before bootstrapping.
        lambda_t: lambdas at times B x [1, ..., T]. Shape is [], or B x [T-1].
        stop_target_gradients: bool indicating whether or not to apply stop gradient
        to targets.

    Returns:
        estimated bootstrapped returns at times B x [0, ...., T-1]
    """
    # swap axes to make time axis the first dimension
    r_t, discount_t, v_t = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(x, 0, 1), (r_t, discount_t, v_t)
    )
    seq_len = r_t.shape[0]
    batch_size = r_t.shape[1]

    # Maybe change scalar lambda to an array.
    lambda_t = jnp.ones_like(discount_t) * lambda_t  # type: ignore

    # Shift bootstrap values by n and pad end of sequence with last value v_t[-1].
    pad_size = min(n, seq_len)
    targets = jnp.concatenate([v_t[n:], jnp.array([v_t[-1]] * pad_size)], axis=0)

    # Pad sequences. Shape is now (T + n - 1,).
    r_t = jnp.concatenate([r_t, jnp.zeros((n - 1, batch_size))], axis=0)
    discount_t = jnp.concatenate([discount_t, jnp.ones((n - 1, batch_size))], axis=0)
    lambda_t = jnp.concatenate([lambda_t, jnp.ones((n - 1, batch_size))], axis=0)  # type: ignore
    v_t = jnp.concatenate([v_t, jnp.array([v_t[-1]] * (n - 1))], axis=0)

    # Work backwards to compute n-step returns.
    for i in reversed(range(n)):
        r_ = r_t[i : i + seq_len]
        discount_ = discount_t[i : i + seq_len]
        lambda_ = lambda_t[i : i + seq_len]  # type: ignore
        v_ = v_t[i : i + seq_len]
        targets = r_[:, :, None] + discount_[:, :, None] * (
            (1.0 - lambda_)[:, :, None] * v_ + lambda_[:, :, None] * targets
        )

    targets = jnp.swapaxes(targets, 0, 1)
    return jax.lax.select(stop_target_gradients, jax.lax.stop_gradient(targets), targets)


def get_train_targets(batch, target_params, representation_apply, critic_apply, args):
    """Compute policy, reward, value, and embedding targets from a sampled buffer batch.

    Value targets use n-step bootstrapped returns (via
    ``batch_n_step_bootstrapped_returns``) computed with the target network.
    Discounts are masked to zero out steps after the first terminal.

    Returns:
        policy_targets:  MCTS search-policy targets, shape ``(batch, seq, num_actions)``.
        reward_targets:  Masked reward targets, shape ``(batch, seq)``.
        value_targets:   n-step bootstrapped value targets,
                         shape ``(batch, seq, num_quantiles)``.
        s_node_targets:  Target node embeddings for self-consistency loss.
        s_edge_targets:  Target edge embeddings for self-consistency loss.
        s_aux_targets:   Target aux embeddings for self-consistency loss.
    """
    # Compute value for each obs
    B, S = batch.obs["node_features"].shape[:2]  # (b, t)
    obs = jax.tree_util.tree_map(lambda x: x.reshape((B * S, *x.shape[2:])), batch.obs)

    node_embeddings, aux_embedding, edge_embeddings = representation_apply.apply(
        target_params,
        obs["node_features"],  # type: ignore
        obs["edge_features"],  # type: ignore
        obs["senders"],  # type: ignore
        obs["receivers"],  # type: ignore
        obs["aux"],
    )  # (b*t, num_hidden)
    values = critic_apply.apply(
        target_params, node_embeddings, aux_embedding
    )  # (b*t, num_quantiles)
    values = jnp.reshape(values, (B, S, -1))  # (b, t, num_quantiles)

    # Return obs sequence dim
    s_node_targets = jnp.reshape(node_embeddings, (B, S, *node_embeddings.shape[1:]))
    s_edge_targets = jnp.reshape(edge_embeddings, (B, S, *edge_embeddings.shape[1:]))
    s_aux_targets = jnp.reshape(aux_embedding, (B, S, -1))

    # Mask after dones: cumulative sum along the row will be >1 after the first True
    discounts = jnp.ones_like(batch.reward) * args.discount  # (b, t)
    discounts = jnp.cumsum(batch.done, axis=1)
    discounts = (discounts < 1).astype(jnp.int32)

    value_targets = batch_n_step_bootstrapped_returns(
        batch.reward,
        discounts,
        values,
        n=args.n_step,
    )
    chex.assert_shape(value_targets, (B, S, args.num_quantiles))

    # We want to mask everything _after_ the first done.
    # This means shifting the discounts right by one and padding the first value as a 1
    # Note that this works because the first value is always a 1 since our mask starts after
    # the first TRUE, which can at worst be the first entry.
    discounts = jnp.concatenate(
        [
            jnp.ones((batch.obs["node_features"].shape[0], 1), dtype=jnp.int32),
            discounts[:, :-1],
        ],
        axis=1,
    )  # (b, t)
    value_targets = value_targets * discounts[:, :, None]
    chex.assert_shape(value_targets, (B, S, args.num_quantiles))

    # We want to mask all rewards after the first done.
    reward_targets = batch.reward * discounts
    policy_targets = batch.search_policy * discounts[:, :, jnp.newaxis]  # (b, t, num_actions)

    # Get unroll length
    value_targets = value_targets[:, : args.sample_sequence_length]
    reward_targets = reward_targets[:, : args.sample_sequence_length]
    policy_targets = policy_targets[:, : args.sample_sequence_length]

    return (
        policy_targets,
        reward_targets,
        value_targets,
        s_node_targets,
        s_edge_targets,
        s_aux_targets,
    )
