# Risk AlphaZero (graph/node) utilities
#
# Shared helpers used by the graph-node Risk AlphaZero training loop.
#
# Contents
# --------
# Data structures:   ExItTransition
# Setup helpers:     init_model_and_optim, make_buffer
# Training helpers:  get_train_targets

from functools import partial
from typing import Mapping, NamedTuple

import flashbax as fbx
import jax
import jax.numpy as jnp
import optax

import lib.util as util


class ExItTransition(NamedTuple):
    step_count: jnp.ndarray  # (b,)  # Step count (distance from the root)
    done: jnp.ndarray  # (b,) # Whether the next step is a reset
    action: jnp.ndarray  # (b,)
    reward: jnp.ndarray  # (b,)
    search_policy: jnp.ndarray  # (b, num_actions)
    obs: jnp.ndarray  # (b, *obs_shape)
    info: Mapping[
        str, float | int | bool | jnp.ndarray
    ]  # Additional info, e.g., episode return, length


def init_model_and_optim(env, model_init_fn, args):
    """Initialise model parameters and an Adam optimiser.

    Constructs dummy inputs matching the graph-node observation structure
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
    dummy_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 1))
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
        min_length_time_axis=args.n_step + 1,
        sample_batch_size=args.train_batch_size,
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
    buffer_state = buffer_fn.init(dummy_transition)
    return buffer_fn, buffer_state


def get_train_targets(batch, target_params, prediction_apply, args):
    """Compute policy and value targets from a sampled buffer batch.

    Value targets use n-step bootstrapped returns computed with the target
    network.  Discounts are masked to zero out steps after the first terminal.

    Returns:
        policy_targets: MCTS search-policy targets, shape ``(batch, num_actions)``.
        value_targets:  n-step bootstrapped value targets,
                        shape ``(batch, num_quantiles)``.
    """
    policy_targets = batch.search_policy[:, 0]

    # Switch batch and time axes
    batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), batch)  # (b, t, *) -> (t, b, *)

    # Last step doesn't reset since we bootstrap from it (the preceding done is the reset if needed)
    # Setting to false here for logging clarity, this value is not read!
    dones = batch.done.at[-1].set(False)
    # Compute value target from final history
    _, value_dist, _, _ = prediction_apply.apply(
        target_params,
        batch.obs["node_features"][-1],
        batch.obs["senders"][-1],
        batch.obs["receivers"][-1],
        batch.obs["aux"][-1],
    )  # (b, num_quantiles)

    def body_fn(carry, i):
        ix = args.n_step - i - 1  # Reverse index for backward pass
        v = (
            batch.reward[ix, :, None] + (1 - dones[ix, :, None]) * args.discount * carry
        )  # (batch_size,)
        return v, v

    _, value_targets = jax.lax.scan(
        body_fn,
        value_dist,
        jnp.arange(args.n_step),
    )
    # The initial is a value target
    value_targets = value_targets[-1, :]  # (b, num_quantiles)
    return policy_targets, value_targets
