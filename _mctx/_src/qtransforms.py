"""Monotonic transformations for the Q-values."""

import chex
import jax
import jax.numpy as jnp

from _mctx._src import tree as tree_lib


def qtransform_by_parent_and_siblings(
    tree: tree_lib.Tree,
    node_index: chex.Numeric,
    *,
    epsilon: chex.Numeric = 1e-8,
) -> chex.Array:
    """Returns qvalues normalized by min, max over V(node) and qvalues.

    Args:
      tree: _unbatched_ MCTS tree state.
      node_index: scalar index of the parent node.
      epsilon: the minimum denominator for the normalization.

    Returns:
        Q-values normalized to be from the [0, 1] interval. The unvisited actions
        will have zero Q-value. Shape `(num_actions,)`.
    """
    chex.assert_shape(node_index, ())
    # Get the q-values q(s, a) for all children (aka: r + d*v(s')
    qvalues = tree.qvalues(node_index)  # (num_actions,)
    visit_counts = tree.children_visits[node_index]  # (num_actions,)
    chex.assert_rank([qvalues, visit_counts, node_index], [1, 1, 0])
    node_value = tree.node_values[node_index]  # (1,)

    # Get safe qvalues that are not zero for unvisited actions to compute min/max.
    safe_qvalues = jnp.where(visit_counts > 0, qvalues, node_value)  # (num_actions,)
    chex.assert_equal_shape([safe_qvalues, qvalues])

    min_value = jnp.minimum(node_value, jnp.min(safe_qvalues, axis=-1))  # (1,)
    max_value = jnp.maximum(node_value, jnp.max(safe_qvalues, axis=-1))  # (1,)

    # Set unvisited actions to zero by assigning min_value to them.
    completed_by_min = jnp.where(visit_counts > 0, qvalues, min_value)  # (num_actions,)
    normalized = (completed_by_min - min_value) / (
        jnp.maximum(max_value - min_value, epsilon)
    )  # (num_actions,)
    chex.assert_equal_shape([normalized, qvalues])
    return normalized  # (num_actions,)


# NOTE: qtransform_by_min_max left out intentionally.
def qtransform_completed_by_mix_value(
    tree: tree_lib.Tree,
    node_index: chex.Numeric,
    *,
    value_scale: chex.Numeric = 0.1,
    maxvisit_init: chex.Numeric = 50.0,
    rescale_values: bool = True,
    use_mixed_value: bool = True,
    epsilon: chex.Numeric = 1e-8,
) -> chex.Array:
    """Returns completed qvalues.

    The missing Q-values of the unvisited actions are replaced by the
    mixed value, defined in Appendix D of
    "Policy improvement by planning with Gumbel":
    https://openreview.net/forum?id=bERaNdoegnO

    The Q-values are transformed by a linear transformation:
      `(maxvisit_init + max(visit_counts)) * value_scale * qvalues`.

    Args:
      tree: _unbatched_ MCTS tree state.
      node_index: scalar index of the parent node.
      value_scale: scale for the Q-values.
      maxvisit_init: offset to the `max(visit_counts)` in the scaling factor.
      rescale_values: if True, scale the qvalues by `1 / (max_q - min_q)`.
      use_mixed_value: if True, complete the Q-values with mixed value,
        otherwise complete the Q-values with the raw value.
      epsilon: the minimum denominator when using `rescale_values`.

    Returns:
      Completed Q-values. Shape `[num_actions]`.
    """
    chex.assert_shape(node_index, ())
    qvalues = tree.qvalues(node_index)
    visit_counts = tree.children_visits[node_index]

    # Computing the mixed value and producing completed_qvalues.
    raw_value = tree.raw_values[node_index]
    prior_probs = jax.nn.softmax(tree.children_prior_logits[node_index])
    if use_mixed_value:
        value = _compute_mixed_value(
            raw_value,
            qvalues=qvalues,
            visit_counts=visit_counts,
            prior_probs=prior_probs,
        )
    else:
        value = raw_value
    completed_qvalues = _complete_qvalues(
        qvalues, visit_counts=visit_counts, value=value
    )

    # Scaling the Q-values.
    if rescale_values:
        completed_qvalues = _rescale_qvalues(completed_qvalues, epsilon)
    maxvisit = jnp.max(visit_counts, axis=-1)
    visit_scale = maxvisit_init + maxvisit
    return visit_scale * value_scale * completed_qvalues


def _rescale_qvalues(qvalues, epsilon):
    """Rescales the given completed Q-values to be from the [0, 1] interval."""
    min_value = jnp.min(qvalues, axis=-1, keepdims=True)
    max_value = jnp.max(qvalues, axis=-1, keepdims=True)
    return (qvalues - min_value) / jnp.maximum(max_value - min_value, epsilon)


def _complete_qvalues(qvalues, *, visit_counts, value):
    """Returns completed Q-values, with the `value` for unvisited actions."""
    chex.assert_equal_shape([qvalues, visit_counts])
    chex.assert_shape(value, [])

    # The missing qvalues are replaced by the value.
    completed_qvalues = jnp.where(visit_counts > 0, qvalues, value)
    chex.assert_equal_shape([completed_qvalues, qvalues])
    return completed_qvalues


def _compute_mixed_value(raw_value, qvalues, visit_counts, prior_probs):
    """Interpolates the raw_value and weighted qvalues.

    Args:
      raw_value: an approximate value of the state. Shape `[]`.
      qvalues: Q-values for all actions. Shape `[num_actions]`. The unvisited
        actions have undefined Q-value.
      visit_counts: the visit counts for all actions. Shape `[num_actions]`.
      prior_probs: the action probabilities, produced by the policy network for
        each action. Shape `[num_actions]`.

    Returns:
      An estimator of the state value. Shape `[]`.
    """
    sum_visit_counts = jnp.sum(visit_counts, axis=-1)
    # Ensuring non-nan weighted_q, even if the visited actions have zero
    # prior probability.
    prior_probs = jnp.maximum(jnp.finfo(prior_probs.dtype).tiny, prior_probs)
    # Summing the probabilities of the visited actions.
    sum_probs = jnp.sum(jnp.where(visit_counts > 0, prior_probs, 0.0), axis=-1)
    weighted_q = jnp.sum(
        jnp.where(  # type: ignore
            visit_counts > 0,
            prior_probs * qvalues / jnp.where(visit_counts > 0, sum_probs, 1.0),
            0.0,
        ),
        axis=-1,
    )
    return (raw_value + sum_visit_counts * weighted_q) / (sum_visit_counts + 1)
