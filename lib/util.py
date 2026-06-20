from functools import partial

import chex
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

# TODO(yousef): cleanup unused, switch optimiser and initialized to 'z'.


def make_tau_hats(num_quantiles: int) -> jnp.ndarray:
    return (jnp.arange(0, num_quantiles) + 0.5) / float(num_quantiles)  # (num_quantiles, )


@partial(jax.jit, static_argnums=(2,))
def sample_quantile_distribution(
    key: chex.PRNGKey,
    quantile_distribution: jnp.ndarray,
    num_samples: int,
) -> jnp.ndarray:
    """Samples from the quantile distribution based on the tau_hats.

    Args:
        key: JAX PRNG key for random number generation. [1, 2]
        quantile_distribution: A 1D array representing the quantile distribution. [num_quantiles, ]
        tau_hats: A 1D array of quantile levels (tau values) for the quantile distribution. [num_quantiles, ]
        num_samples: The number of samples to draw from the quantile distribution.

    Returns:
        A 1D array of sampled quantiles from the quantile distribution. [num_samples, ]
    """
    # We assume the quantiles are evenly spaced between 0 and 1.
    quantile_levels = make_tau_hats(quantile_distribution.shape[0])
    sampled_taus = jax.random.uniform(key, shape=(num_samples,), minval=0.0, maxval=1.0)
    sampled_quantiles = jnp.interp(sampled_taus, quantile_levels, quantile_distribution)
    return sampled_quantiles


@jax.jit
def cvar(x: jnp.ndarray, alpha: float) -> jnp.ndarray:
    chex.assert_rank(x, 1)  # Ensure x is a 1D array

    # Sort the values in ascending order
    sorted_x = jnp.sort(x)

    # Take the bottom alpha fraction of the values
    num_values = sorted_x.shape[0]
    num_cvar_values = jnp.array(num_values * alpha, int)
    mask = jnp.arange(num_values) < num_cvar_values
    values_masked = jnp.where(mask, sorted_x, 0)

    # Calculate the CVaR as the mean of the bottom alpha fraction
    cvar_value = jnp.sum(values_masked) / jnp.maximum(num_cvar_values, 1)
    return cvar_value


@partial(jax.jit, static_argnums=(1, 2))
def distortion_risk(x: jnp.ndarray, distortion_fn, n_samples: int) -> jnp.ndarray:
    """Computes distortion risk measure for a given distortion function.

    Args:
        x: 1D array of values
        distortion_fn: Function that maps quantiles in [0,1] to distorted quantiles
        n_samples: Number of quantile samples to use for approximation

    Returns:
        The distortion risk measure value
    """
    chex.assert_rank(x, 1)  # Ensure x is a 1D array

    # Sort the values in ascending order
    sorted_x = jnp.sort(x)
    n = sorted_x.shape[0]

    # Generate uniform quantiles in [0, 1]
    quantiles = (jnp.arange(n_samples) + 0.5) / n_samples

    # Apply distortion function
    distorted_quantiles = jax.vmap(distortion_fn)(quantiles)

    # Map distorted quantiles to indices: round up and clamp
    # distorted_quantile * n gives position in [0, n], ceil and subtract 1 for 0-indexing
    indices = jnp.ceil(distorted_quantiles * n).astype(int) - 1
    indices = jnp.clip(indices, 0, n - 1)

    # Average the values at those indices
    risk_value = jnp.mean(sorted_x[indices])

    return risk_value


@partial(jax.jit, static_argnums=(2,))
def cvar_distortion(x: jnp.ndarray, alpha: float, n_samples: int = 1000) -> jnp.ndarray:
    """CVaR using distortion risk measure formulation.

    Args:
        x: 1D array of values
        alpha: CVaR level (e.g., 0.1 for bottom 10%)
        n_samples: Number of quantile samples for approximation

    Returns:
        The CVaR value
    """
    distortion_fn = lambda t: alpha * t
    return distortion_risk(x, distortion_fn, n_samples)


@partial(jax.jit, static_argnums=(2,))
def wang_distortion(x: jnp.ndarray, alpha: float, n_samples: int = 1000) -> jnp.ndarray:
    """Wang distortion risk measure.

    Uses the Wang transform: g(t) = Φ(Φ^{-1}(t) + α), where Φ is the standard normal CDF. This distortion translates the CDF of the standard Gaussian.

    Args:
        x: 1D array of values
        alpha: Risk preference parameter
               - Positive α: risk-seeking behavior
               - Negative α: risk-averse behavior
        n_samples: Number of quantile samples for approximation

    Returns:
        The Wang distortion risk measure value
    """
    from jax.scipy.stats import norm

    distortion_fn = lambda t: norm.cdf(norm.ppf(t) + alpha)
    return distortion_risk(x, distortion_fn, n_samples)


@partial(jax.jit, static_argnums=(2,))
def pow_distortion(x: jnp.ndarray, alpha: float, n_samples: int = 1000) -> jnp.ndarray:
    """Power distortion risk measure.

    Uses the power transform:
    - If α ≥ 0: g(t) = t^(1/(1+|α|))
    - If α < 0: g(t) = 1 - (1-t)^(1/(1+|α|))

    Args:
        x: 1D array of values
        alpha: Risk preference parameter
        n_samples: Number of quantile samples for approximation

    Returns:
        The power distortion risk measure value
    """
    exponent = 1 / (1 + jnp.abs(alpha))
    distortion_fn = lambda t: jnp.where(
        alpha >= 0,
        jnp.power(t, exponent),
        1 - jnp.power(1 - t, exponent),
    )
    return distortion_risk(x, distortion_fn, n_samples)


@partial(jax.jit, static_argnums=(1,))
def esr_risk(x: jnp.ndarray, utility_fn) -> jnp.ndarray:
    """Computes expected utility risk measure for a given utility function.

    Args:
        x: 1D array of values
        utility_fn: Function that maps scalar values to utility values

    Returns:
        The expected utility (mean of utility-transformed values)
    """
    chex.assert_rank(x, 1)  # Ensure x is a 1D array

    # Apply utility function to all values
    utilities = utility_fn(x)

    # Return the average utility
    return jnp.mean(utilities)


@jax.jit
def sqrt_utility(x: jnp.ndarray, alpha: float, n_samples: int = 1000) -> jnp.ndarray:
    """Square root utility function for risk-averse preferences.

    Applies utility function: u(x) = 10 * (√x - 1)

    Args:
        x: 1D array of values
        _alpha: Unused parameter for compatibility
        _n_samples: Unused parameter for compatibility

    Returns:
        The expected utility value
    """

    def utility_function(x):
        return 10 * (jnp.cbrt(x + 1) - 1)

    return esr_risk(x, utility_function)


@jax.jit
def eps_greedy_exploration(rng: jnp.ndarray, q_vals: jnp.ndarray, eps: float) -> jnp.ndarray:
    # A key for sampling random actions and one for picking
    rng_a, rng_e = jax.random.split(rng, 2)
    # Get the greedy actions
    greedy_actions = jnp.argmax(q_vals, axis=-1)  # (b,)
    chosen_actions = jnp.where(
        # With prob eps, choose a random action
        jax.random.uniform(rng_e, greedy_actions.shape) < eps,
        jax.random.randint(rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]),
        greedy_actions,
    )
    return chosen_actions


@jax.jit
def linear_schedule(count: int, lr: float, num_updates: int, min_lr: float) -> float:
    """Linear decay of the learning rate."""
    frac = 1.0 - (count / num_updates)
    return jnp.maximum(lr * frac, min_lr)  # type: ignore


@jax.jit
def calc_eps(
    t: int,
    epsilon_start: float,
    epsilon_finish: float,
    epsilon_anneal_time: float,
    learning_start: int,
) -> jnp.ndarray:
    return jnp.clip(
        ((epsilon_finish - epsilon_start) / epsilon_anneal_time)
        * (jnp.maximum(0, t - learning_start))  # Only anneal after learning starts
        + epsilon_start,
        epsilon_finish,
    )


@partial(jax.jit, static_argnums=(2,))
def _distort_value(q_dist: jnp.ndarray, tau: jnp.ndarray, alpha_cvar: float) -> jnp.ndarray:
    """Masks all quantile with tau > alpha, returns mean."""
    # q_dist: (num_quantiles, num_actions)
    # tau: (num_quantiles, )

    # Mask all quantiles with tau > alpha
    mask = jnp.expand_dims(tau, axis=-1) <= alpha_cvar  # (num_quantiles, num_actions)
    # Multiply q_dist with mask
    q_dist = jnp.where(mask, q_dist, jnp.zeros_like(q_dist))  # (num_quantiles, num_actions)
    # Calculate the distorted value by taking the mean of the quantiles that are below the cvar level
    q_dist_value = jnp.sum(q_dist, axis=0) / jnp.sum(mask, axis=0)
    return q_dist_value  # (num_actions, )


@partial(jax.jit, static_argnums=(2,))
def _batched_distort_value(q_dist: jnp.ndarray, tau: jnp.ndarray, alpha_cvar: float) -> jnp.ndarray:
    """Batched version of _distort_value."""
    batched_fn = jax.vmap(
        _distort_value,
        in_axes=(
            0,
            None,
            None,
        ),  # q_dist: (b, num_quantiles, num_actions), tau: (num_quantiles, )
        out_axes=0,
    )
    return batched_fn(q_dist, tau, alpha_cvar)


@partial(jax.jit, static_argnums=(2, 3))
def distort_value(
    q_dist: jnp.ndarray, tau: jnp.ndarray, distortion_fn, n_samples: int
) -> jnp.ndarray:
    """Distorts quantile distribution using a distortion function.

    Samples quantiles, applies distortion, interpolates Q-values at distorted
    quantiles, and averages them.

    Args:
        q_dist: Quantile distribution (num_quantiles, num_actions)
        tau: Quantile levels (num_quantiles,)
        distortion_fn: Function that maps quantiles in [0,1] to distorted quantiles
        n_samples: Number of quantile samples to use

    Returns:
        Distorted values for each action (num_actions,)
    """
    # Generate uniform quantiles in [0, 1]
    quantiles = (jnp.arange(n_samples) + 0.5) / n_samples

    # Apply distortion function
    distorted_quantiles = jax.vmap(distortion_fn)(quantiles)

    # Interpolate q_dist at distorted quantiles for each action
    def interpolate_action(q_values):
        # q_values: (num_quantiles,) for one action
        # Returns: (n_samples,) interpolated values
        return jnp.interp(distorted_quantiles, tau, q_values)

    # Apply to each action: (num_quantiles, num_actions) -> (n_samples, num_actions)
    interpolated_values = jax.vmap(interpolate_action, in_axes=1, out_axes=1)(q_dist)

    # Average over samples
    distorted_values = jnp.mean(interpolated_values, axis=0)  # (num_actions,)

    return distorted_values


@partial(jax.jit, static_argnums=(2, 3))
def batched_distort_value(
    q_dist: jnp.ndarray, tau: jnp.ndarray, distortion_fn, n_samples: int
) -> jnp.ndarray:
    """Batched version of distort_value.

    Args:
        q_dist: Batched quantile distribution (batch, num_quantiles, num_actions)
        tau: Quantile levels (num_quantiles,)
        distortion_fn: Function that maps quantiles in [0,1] to distorted quantiles
        n_samples: Number of quantile samples to use

    Returns:
        Distorted values for each batch and action (batch, num_actions)
    """
    batched_fn = jax.vmap(
        distort_value,
        in_axes=(0, None, None, None),
        out_axes=0,
    )
    return batched_fn(q_dist, tau, distortion_fn, n_samples)


@partial(jax.jit, static_argnums=(3,))
def cvar_distortion_q(
    q_dist: jnp.ndarray, tau: jnp.ndarray, alpha: float, n_samples: int = 1000
) -> jnp.ndarray:
    """CVaR distortion for a single quantile distribution.

    Args:
        q_dist: Quantile distribution (num_quantiles, num_actions)
        tau: Quantile levels (num_quantiles,)
        alpha: CVaR level (e.g., 0.1 for bottom 10%)
        n_samples: Number of quantile samples for approximation

    Returns:
        Distorted values (num_actions,)
    """
    distortion_fn = lambda t: alpha * t
    return distort_value(q_dist, tau, distortion_fn, n_samples)


@partial(jax.jit, static_argnums=(3,))
def batched_cvar_distortion(
    q_dist: jnp.ndarray, tau: jnp.ndarray, alpha: float, n_samples: int = 1000
) -> jnp.ndarray:
    """Batched CVaR distortion for quantile distributions.

    Args:
        q_dist: Batched quantile distribution (batch, num_quantiles, num_actions)
        tau: Quantile levels (num_quantiles,)
        alpha: CVaR level (e.g., 0.1 for bottom 10%)
        n_samples: Number of quantile samples for approximation

    Returns:
        Distorted values (batch, num_actions)
    """
    distortion_fn = lambda t: alpha * t
    return batched_distort_value(q_dist, tau, distortion_fn, n_samples)


@partial(jax.jit, static_argnums=(3,))
def wang_distortion_q(
    q_dist: jnp.ndarray, tau: jnp.ndarray, alpha: float, n_samples: int = 1000
) -> jnp.ndarray:
    """Wang distortion for a single quantile distribution.

    Args:
        q_dist: Quantile distribution (num_quantiles, num_actions)
        tau: Quantile levels (num_quantiles,)
        alpha: Risk preference parameter
               - Positive α: risk-seeking behavior
               - Negative α: risk-averse behavior
        n_samples: Number of quantile samples for approximation

    Returns:
        Distorted values (num_actions,)
    """
    distortion_fn = lambda t: norm.cdf(norm.ppf(t) + alpha)
    return distort_value(q_dist, tau, distortion_fn, n_samples)


@partial(jax.jit, static_argnums=(3,))
def batched_wang_distortion(
    q_dist: jnp.ndarray, tau: jnp.ndarray, alpha: float, n_samples: int = 1000
) -> jnp.ndarray:
    """Batched Wang distortion for quantile distributions.

    Args:
        q_dist: Batched quantile distribution (batch, num_quantiles, num_actions)
        tau: Quantile levels (num_quantiles,)
        alpha: Risk preference parameter
               - Positive α: risk-seeking behavior
               - Negative α: risk-averse behavior
        n_samples: Number of quantile samples for approximation

    Returns:
        Distorted values (batch, num_actions)
    """
    distortion_fn = lambda t: norm.cdf(norm.ppf(t) + alpha)
    return batched_distort_value(q_dist, tau, distortion_fn, n_samples)


@partial(jax.jit, static_argnums=(3,))
def pow_distortion_q(
    q_dist: jnp.ndarray, tau: jnp.ndarray, alpha: float, n_samples: int = 1000
) -> jnp.ndarray:
    """Power distortion for a single quantile distribution.

    Args:
        q_dist: Quantile distribution (num_quantiles, num_actions)
        tau: Quantile levels (num_quantiles,)
        alpha: Risk preference parameter
        n_samples: Number of quantile samples for approximation

    Returns:
        Distorted values (num_actions,)
    """
    exponent = 1 / (1 + jnp.abs(alpha))
    distortion_fn = lambda t: jnp.where(
        alpha >= 0,
        jnp.power(t, exponent),
        1 - jnp.power(1 - t, exponent),
    )
    return distort_value(q_dist, tau, distortion_fn, n_samples)


@partial(jax.jit, static_argnums=(3,))
def batched_pow_distortion(
    q_dist: jnp.ndarray, tau: jnp.ndarray, alpha: float, n_samples: int = 1000
) -> jnp.ndarray:
    """Batched power distortion for quantile distributions.

    Args:
        q_dist: Batched quantile distribution (batch, num_quantiles, num_actions)
        tau: Quantile levels (num_quantiles,)
        alpha: Risk preference parameter
        n_samples: Number of quantile samples for approximation

    Returns:
        Distorted values (batch, num_actions)
    """
    exponent = 1 / (1 + jnp.abs(alpha))
    distortion_fn = lambda t: jnp.where(
        alpha >= 0,
        jnp.power(t, exponent),
        1 - jnp.power(1 - t, exponent),
    )
    return batched_distort_value(q_dist, tau, distortion_fn, n_samples)
