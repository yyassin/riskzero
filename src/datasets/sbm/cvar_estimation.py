import jax
import jax.numpy as jnp

import lib.util as util

# ---------------------------------------------------------------------------
# Edge weight sampler
# ---------------------------------------------------------------------------


def sample_weights(edge_type, key):
    def type_0():
        w = jax.random.bernoulli(key, 0.4)
        return jnp.where(w, 20.0, -7.0)

    def type_1():
        w = jax.random.bernoulli(key, 0.7)
        return jnp.where(w, 6, -3.0)

    def type_2():
        return 0.0

    weight_fns = [type_0, type_1, type_2]
    return jax.lax.switch(edge_type, weight_fns)


# ---------------------------------------------------------------------------
# CVaR estimation
# ---------------------------------------------------------------------------


def estimate_distortion_risk(rng_key, edge_types, n_trials, cvar_alpha, distortion="cvar"):
    """Estimate a distortion risk measure of total matching weight by simulation.

    Parameters
    ----------
    rng_key    : JAX PRNGKey.
    edge_types : Integer array of edge types for the selected matching.
    n_trials   : Number of Monte-Carlo trials.
    cvar_alpha : Tail probability (e.g. 0.25 for CVaR_25).
    distortion : One of "cvar", "pow", "wang", "sqrt".

    Returns
    -------
    float : Estimated risk measure.
    """
    distortion_fn = {
        "cvar": util.cvar_distortion,
        "pow": util.pow_distortion,
        "wang": util.wang_distortion,
        "sqrt": util.sqrt_utility,
    }[distortion]

    with jax.default_device(jax.devices("cpu")[0]):
        rng_keys = jax.random.split(rng_key, edge_types.size * n_trials).reshape(
            edge_types.size, n_trials, 2
        )

        # For each edge, sample weights for all trials
        def simulate_edge(keys_for_edge, edge_type):
            return jax.vmap(lambda k: sample_weights(edge_type, k))(keys_for_edge)

        all_weights = jax.vmap(simulate_edge, in_axes=(0, 0))(rng_keys, edge_types)
        # Shape: (n_edges, n_trials)

        total_weights = jnp.sum(all_weights, axis=0)  # (n_trials,)

        risk_measure = distortion_fn(total_weights, alpha=cvar_alpha)
        return risk_measure


if __name__ == "__main__":
    edge_types = jnp.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    cvar = estimate_distortion_risk(jax.random.PRNGKey(23), edge_types, n_trials=100_000, cvar_alpha=0.25)
    print(cvar)
