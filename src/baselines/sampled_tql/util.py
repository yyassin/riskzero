# Sampled TQL utilities
#
# Shared helpers used by the Sampled TQL training loop.
#
# Contents
# --------
# Data structures:   Transition
# Setup helpers:     init_model_and_optim, make_buffer
# Training helpers:  sampled_cvar_q_values, get_value_targets
#
# Sampled TQL extends QR-DQN with a second network head (RewardHistoryNetwork)
# that predicts a return distribution from the observation history window.
# The two heads' distributions are combined at target-computation time.

from functools import partial
from typing import NamedTuple

import flashbax as fbx
import jax
import jax.numpy as jnp
import optax

import lib.util as util


class Transition(NamedTuple):
    """A single environment transition, augmented with history context.

    Extends the standard QR-DQN ``Transition`` with fields needed by the
    GRU reward-history network and the Sampled TQL target computation.

    Fields
    ------
    obs_history                 : Sliding window of past observations,
                                  shape ``(history_length + 1, *obs_shape)``.
    obs                         : Current observation, shape ``(*obs_shape)``.
    act_history                 : Sliding window of past actions,
                                  shape ``(history_length + 1,)``.
    done                        : Whether the current step is terminal.
    reward                      : Scalar reward at the current step.
    discounted_accumulated_return: Discounted cumulative return from the
                                  history window up to (but not including)
                                  the current step.
    info                        : Episode statistics dict
                                  (``episode_return``, ``episode_length``,
                                  ``is_terminal_step``).
    step_count                  : Episode step index; used to mask the
                                  reward-history head at step 0 when no
                                  prior history is available.
    """

    obs_history: jnp.ndarray
    obs: jnp.ndarray
    act_history: jnp.ndarray
    done: jnp.ndarray
    reward: jnp.ndarray
    discounted_accumulated_return: jnp.ndarray
    info: dict[str, jnp.ndarray]
    step_count: jnp.ndarray


# --- Setup Helpers ---


def init_model_and_optim(env, model_init_fn, args):
    """Initialise model parameters and an Adam optimiser.

    Constructs a dummy observation repeated over ``history_length`` steps to
    initialise the joint (QR + reward-history) model in a single ``init`` call.

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
        obs=dummy_state.observation[0],
        act_history=jnp.zeros((args.history_length + 1,), dtype=jnp.int32),
        done=jnp.array(False),
        reward=jnp.array(0.0),
        discounted_accumulated_return=jnp.array(0.0),
        info={
            "episode_return": jnp.array(0.0),
            "episode_length": jnp.array(0),
            "is_terminal_step": jnp.array(False),
        },
        step_count=jnp.array(0),
    )
    buffer_state = buffer_fn.init(dummy_transition)

    return buffer_fn, buffer_state


# --- Training Helpers ---


def sampled_cvar_q_values(
    rng_key: jax.Array,
    qr_q_dist: jnp.ndarray,
    rh_dist: jnp.ndarray,
    num_actions: int,
    num_quantile_samples: int,
    alpha_cvar: float,
) -> tuple[jnp.ndarray, jax.Array]:
    """Compute CVaR Q-values by sampling from both distributional heads.

    The QR model predicts the distribution of future returns Z(x, a) and the
    reward-history model predicts the distribution of accumulated past rewards
    H(x).  The combined samples Z(x, a) + H(x) approximate the full-episode
    return distribution for each action, from which CVaR yields a
    risk-sensitive Q-value per action.

    Because ``sample_quantile_distribution`` operates per-distribution, the
    batch and action dims are flattened together before vmapping, then
    reshaped back after sampling.

    Args:
        rng_key:             JAX PRNG key (consumed; updated key is returned).
        qr_q_dist:           QR model output, shape ``(batch, num_quantiles, num_actions)``.
        rh_dist:             Reward-history model output, shape ``(batch, num_quantiles)``.
        num_actions:         Number of discrete actions.
        num_quantile_samples: Number of Monte-Carlo samples drawn per distribution.
        alpha_cvar:          CVaR confidence level in ``(0, 1]``.

    Returns:
        q_values: CVaR Q-values, shape ``(batch, num_actions)``.
        rng_key:  Updated PRNG key.
    """
    rng_key, sample_key = jax.random.split(rng_key)

    qr_dist = jnp.permute_dims(qr_q_dist, (0, 2, 1))  # (batch, num_actions, num_quantiles)
    qr_dist = qr_dist.reshape((-1, qr_dist.shape[-1]))  # (batch * num_actions, num_quantiles)

    sample_keys = jax.random.split(sample_key, qr_dist.shape[0])
    qr_samples = jax.vmap(util.sample_quantile_distribution, in_axes=(0, 0, None))(
        sample_keys, qr_dist, num_quantile_samples
    )  # (batch * num_actions, num_quantile_samples)
    qr_samples = qr_samples.reshape(
        (-1, num_actions, num_quantile_samples)
    )  # (batch, num_actions, num_quantile_samples)

    rng_key, sample_key = jax.random.split(rng_key)
    sample_keys = jax.random.split(sample_key, rh_dist.shape[0])
    rh_samples = jax.vmap(util.sample_quantile_distribution, in_axes=(0, 0, None))(
        sample_keys, rh_dist, num_quantile_samples
    )  # (batch, num_quantile_samples)

    samples = qr_samples + rh_samples[:, None, :]  # (batch, num_actions, num_quantile_samples)
    samples = samples.reshape(
        (-1, num_quantile_samples)
    )  # (batch * num_actions, num_quantile_samples)
    q_values = jax.vmap(util.cvar, in_axes=(0, None))(samples, alpha_cvar)
    q_values = q_values.reshape((-1, num_actions))  # (batch, num_actions)

    return q_values, rng_key


def get_value_targets(learn_batch, _qr_model, _reward_history_model, target_params, args, rng_key):
    """Compute n-step distributional value targets for Sampled TQL.

    Combines two network heads to form the target return distribution:

    - ``_qr_model`` produces a quantile distribution over future Q-values
      (bootstrapped from the next state).
    - ``_reward_history_model`` produces a quantile distribution over the
      accumulated reward from the history window.  This head is masked to
      zero at episode step 0 (no prior history).

    Samples are drawn from each head, summed to form a joint distribution
    over total returns, and CVaR is applied to select the greedy action
    (Double DQN style).  An n-step Bellman backup is then applied to the
    QR head's quantile estimates.

    Returns:
        target_qr: Target quantile values, shape ``(batch, num_quantiles)``.
    """
    batch = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(x, 0, 1), learn_batch
    )  # (b, t, *) -> (t, b, *)

    # Step 1: Get bootstrap targets from both network heads
    # Last step doesn't reset since we bootstrap from it (the preceding done is the reset if needed)
    # Setting to false here for logging clarity, this value is not read!
    dones = batch.done.at[-1].set(False)

    qr_out = _qr_model.apply(
        target_params,
        batch.obs[-1],
    )  # q_dist: (b, num_quantiles, num_actions)
    # The accumulated return at initial states is 0
    rh_out = (
        _reward_history_model.apply(
            target_params,
            batch.obs_history[-1, :, 1:],  # history up to (but not including) current obs
        )
        * (1 - (batch.step_count[-1] == 0))[:, None]  # mask out at episode step 0
    )  # (b, num_quantiles)

    # Step 2 & 3: Sample from both heads and compute CVaR Q-values
    q_values, rng_key = sampled_cvar_q_values(
        rng_key, qr_out.q_dist, rh_out, args.num_actions, args.num_quantile_samples, args.alpha_cvar
    )  # (b, num_actions)

    greedy_action = jnp.argmax(q_values, axis=-1)  # (b,)

    qr_next_target = jnp.take_along_axis(
        qr_out.q_dist,
        greedy_action[:, None, None],  # (b, 1, 1)
        axis=-1,
    ).squeeze(-1)  # (b, num_quantiles)

    # Step 4: N-step Bellman backup over the QR target
    def body_fn(carry, i):
        ix = args.n_step - i - 1  # reverse index for backward pass
        v = (
            batch.reward[ix, :, None] + (1 - dones[ix, :, None]) * args.gamma * carry
        )  # (b, num_quantiles)
        return v, v

    _, qr_next_target = jax.lax.scan(
        body_fn,
        qr_next_target,
        jnp.arange(args.n_step),
    )
    target_qr = qr_next_target[-1, :]

    return target_qr  # (b, num_quantiles)
