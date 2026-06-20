# Quantile regression loss functions
#
# Shared loss functions for distributional RL baselines.
#
# Contents
# --------
# Loss functions: huber_loss, quantile_huber_loss, batched_quantile_huber_loss

from functools import partial

import chex
import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("delta",))
def huber_loss(x: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    """Element-wise Huber loss.

    .. code-block::

        L(x) = 0.5 * x²                         if |x| <= delta
               0.5 * delta² + delta*(|x|-delta)  otherwise
    """
    chex.assert_type(x, float)
    abs_x = jnp.abs(x)
    quadratic = jnp.minimum(abs_x, delta)
    # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
    linear = abs_x - quadratic
    return 0.5 * quadratic**2 + delta * linear


@partial(jax.jit, static_argnames=("huber_param", "stop_target_gradients"))
def quantile_huber_loss(
    dist_src: jnp.ndarray,  # (num_quantiles,)
    tau_src: jnp.ndarray,  # (num_quantiles,)
    dist_target: jnp.ndarray,  # (num_quantiles,)
    huber_param: float = 0,
    stop_target_gradients: bool = True,
) -> jnp.ndarray:
    """Quantile-regression loss (QR-DQN objective).

    Computes the asymmetric Huber (or L1) loss weighted by the quantile
    regression residual.  Loss is averaged over target quantiles and summed
    over source quantiles, matching the QR-DQN paper.

    Args:
        dist_src:  Source quantile estimates, shape ``(num_quantiles,)``.
        tau_src:   Quantile levels for ``dist_src``, shape ``(num_quantiles,)``.
        dist_target: Target quantile estimates, shape ``(num_quantiles,)``.
        huber_param: Huber threshold ``delta``; if 0, plain L1 loss is used.
        stop_target_gradients: Whether to stop gradients through the sign term.
    """
    # Calculate quantile error.
    delta = dist_target[None, :] - dist_src[:, None]
    delta_neg = (delta < 0.0).astype(jnp.float32)
    delta_neg = jax.lax.select(stop_target_gradients, jax.lax.stop_gradient(delta_neg), delta_neg)
    weight = jnp.abs(tau_src[:, None] - delta_neg)

    # Calculate Huber loss.
    if huber_param > 0.0:
        loss = huber_loss(delta, huber_param)
    else:
        loss = jnp.abs(delta)
    loss *= weight

    # Average over target-samples dimension, sum over src-samples dimension.
    return jnp.sum(jnp.mean(loss, axis=-1))


@partial(jax.jit, static_argnames=("huber_param", "stop_target_gradients"))
def batched_quantile_huber_loss(
    dist_src: jnp.ndarray,  # (b, num_quantiles, num_actions)
    tau_src: jnp.ndarray,  # (num_quantiles,)
    dist_target: jnp.ndarray,  # (b, num_quantiles, num_actions)
    huber_param: float = 0,
    stop_target_gradients: bool = True,
) -> jnp.ndarray:
    """Batched quantile-regression loss over a mini-batch.

    Applies ``quantile_huber_loss`` independently for each element in the
    batch via ``jax.vmap``.

    Args:
        dist_src:    Source quantile estimates, shape ``(b, num_quantiles, num_actions)``.
        tau_src:     Quantile levels for ``dist_src``, shape ``(num_quantiles,)``.
        dist_target: Target quantile estimates, shape ``(b, num_quantiles, num_actions)``.
        huber_param: Huber threshold ``delta``; if 0, plain L1 loss is used.
        stop_target_gradients: Whether to stop gradients through the sign term.
    """
    return jax.vmap(
        quantile_huber_loss,
        in_axes=(0, None, 0, None, None),
        out_axes=0,
    )(dist_src, tau_src, dist_target, huber_param, stop_target_gradients)
