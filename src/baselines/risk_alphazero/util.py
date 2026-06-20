# Risk AlphaZero utilities
#
# Shared helpers used by the Risk AlphaZero training loop.
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
    obs_history: jnp.ndarray  # (b, history_length + 1, *obs_shape)
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

    Constructs dummy inputs for all three network heads (prediction, reward,
    reward-history) and calls the joint ``init_model_fn`` in a single pass.

    The learning rate is either constant or linearly decayed to ``min_lr``
    over ``lr_anneal_iterations * train_epochs_per_iter`` steps, depending
    on ``args.lr_linear_decay``.

    Returns:
        params:    Initial network parameters.
        optimizer: Optax optimiser (gradient clipping + Adam).
        opt_state: Initial optimiser state.
    """
    dummy_obs = jnp.zeros((1, *env.observation_shape))  # (b, *obs_shape), b=1 for initialization

    # Repeat obs `args.history_length` times to get obs_history
    dummy_obs_history = jnp.repeat(
        dummy_obs[0][None, :],
        args.history_length,
        axis=0,
    )  # (history_length, *obs_shape)
    # Add batch dimension
    dummy_obs_history = dummy_obs_history[None, :]  # (1, history_length, *obs_shape)
    dummy_action = jax.random.randint(jax.random.PRNGKey(args.seed), (1,), 0, env.num_actions)
    params = model_init_fn.init(
        jax.random.PRNGKey(args.seed), dummy_obs, dummy_obs_history, dummy_action
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

    Stores full ``ExItTransition`` trajectories including observation history
    and MCTS-improved search policies.  All buffer operations are JIT-compiled.

    Returns:
        buffer_fn:    Flashbax buffer object with JIT-compiled methods.
        buffer_state: Initial (empty) buffer state.
    """
    # Make trajectory buffer
    dummy_state = jax.vmap(env.init)(
        jax.random.split(jax.random.PRNGKey(0), 1)
    )  # (1, *env.observation_shape)

    # Repeat obs `args.history_length` times to get obs_history
    obs_history = jnp.repeat(
        dummy_state.observation[0][None, :],
        args.history_length + 1,
        axis=0,
    )  # (history_length, *obs_shape)

    dummy_transition = ExItTransition(
        step_count=jnp.array(0),
        done=jnp.array(False),
        action=jnp.array(0),
        reward=jnp.array(0.0),
        search_policy=jnp.zeros((args.num_actions,)),
        obs=dummy_state.observation[0],
        obs_history=obs_history,
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


# --- Training Helpers ---


def get_train_targets(batch, target_params, prediction_apply, args):
    """Compute policy and value targets from a sampled buffer batch.

    The policy target is the MCTS-improved search policy stored in the buffer.
    The value target is an n-step bootstrapped quantile value computed by a
    backward scan over the trajectory, using the target network's value head
    for bootstrapping.

    Returns:
        policy_targets: MCTS search-policy targets, shape ``(batch, num_actions)``.
        value_targets:  n-step bootstrapped value targets, shape ``(batch, num_quantiles)``.
    """
    policy_targets = batch.search_policy[:, 0]

    # Switch batch and time axes
    batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), batch)  # (b, t, *) -> (t, b, *)

    # Last step doesn't reset since we bootstrap from it (the preceding done is the reset if needed)
    # Setting to false here for logging clarity, this value is not read!
    dones = batch.done.at[-1].set(False)
    # Compute value target
    _, values, _ = prediction_apply.apply(target_params, batch.obs[-1])  # (b,)

    def body_fn(carry, i):
        ix = args.n_step - i - 1  # Reverse index for backward pass
        v = (
            batch.reward[ix, :, None] + (1 - dones[ix, :, None]) * args.discount * carry
        )  # (batch_size,)
        return v, v

    _, value_targets = jax.lax.scan(
        body_fn,
        values,
        jnp.arange(args.n_step),
    )
    # The initial is a value target
    value_targets = value_targets[-1, :]  # (b,)
    return policy_targets, value_targets
