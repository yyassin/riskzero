"""CVaR estimation utilities for stochastic maximum independent set (SMIS).

Node weight types
-----------------
Three stochastic node weight types are defined:

  Type 0  --  Bernoulli(0.3):  +40 (win) / -7  (loss)  [high-risk, high-reward]
  Type 1  --  Bernoulli(0.75): +10 (win) / -5  (loss)  [moderate-risk]
  Type 2  --  constant 0                               [zero-risk]
"""

import jax
import jax.numpy as jnp

import lib.util as util


def sample_weights(node_type, key):
    def type_0():
        w = jax.random.bernoulli(key, 0.3)
        return jnp.where(w, 40.0, -7.0)

    def type_1():
        w = jax.random.bernoulli(key, 0.75)
        return jnp.where(w, 10.0, -5.0)

    def type_2():
        return 0.0

    return jax.lax.switch(jnp.asarray(node_type), [type_0, type_1, type_2])


def sample_graph_weight_instances(node_types, n_samples, rng_key):
    """Draw *n_samples* independent weight realizations for all nodes.

    Parameters
    ----------
    node_types : jnp array of shape (n_nodes,).
    n_samples  : Number of Monte-Carlo scenarios.
    rng_key    : JAX PRNGKey.

    Returns
    -------
    weights : jnp array of shape (n_samples, n_nodes).
    """
    types_flat = jnp.repeat(node_types[None, :], n_samples, axis=0).reshape(-1)
    rng_key, subkey = jax.random.split(rng_key)
    keys = jax.random.split(subkey, types_flat.shape[0])
    weights = jax.vmap(sample_weights)(types_flat, keys)
    return weights.reshape((n_samples, node_types.shape[0]))


def estimate_distortion_risk(rng_key, node_types, n_trials, cvar_alpha, distortion="cvar"):
    """Estimate a distortion risk measure of total node weight by simulation.

    Parameters
    ----------
    rng_key    : JAX PRNGKey.
    node_types : Integer array of node types for the selected independent set.
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
        rng_keys = jax.random.split(rng_key, node_types.size * n_trials).reshape(
            node_types.size, n_trials, 2
        )

        # For each node, sample weights for all trials
        def simulate_node(keys_for_node, node_type):
            return jax.vmap(lambda k: sample_weights(node_type, k))(keys_for_node)

        all_weights = jax.vmap(simulate_node, in_axes=(0, 0))(rng_keys, node_types)
        # Shape: (n_nodes, n_trials)

        total_weights = jnp.sum(all_weights, axis=0)  # (n_trials,)

        risk_measure = distortion_fn(total_weights, alpha=cvar_alpha)
        return risk_measure


if __name__ == "__main__":
    node_types = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1])
    cvar = estimate_distortion_risk(
        jax.random.PRNGKey(23), node_types, n_trials=100_000, cvar_alpha=0.25
    )
    print(cvar)
