"""Functions for Sequential Halving."""

import math

import chex
import jax.numpy as jnp


def score_considered(
    considered_visit, gumbel, logits, normalized_qvalues, visit_counts
):
    """Returns a score usable for an argmax."""
    # We allow to visit a child, if it is the only considered child.
    low_logit = -1e9
    logits = logits - jnp.max(logits, keepdims=True, axis=-1)
    # Mask out the actions that don't match the considered visit count to implement
    # sequential halving according to the sequence provided by the function below.
    penalty = jnp.where(visit_counts == considered_visit, 0, -jnp.inf)
    chex.assert_equal_shape([gumbel, logits, normalized_qvalues, penalty])
    return jnp.maximum(low_logit, gumbel + logits + normalized_qvalues) + penalty


def get_sequence_of_considered_visits(max_num_considered_actions, num_simulations):
    """Returns a sequence of visit counts considered by Sequential Halving.

    Sequential Halving is a "pure exploration" algorithm for bandits, introduced
    in "Almost Optimal Exploration in Multi-Armed Bandits": http://proceedings.mlr.press/v28/karnin13.pdf

    The visit counts allows us to implement Sequential Halving by selecting the best
    action from the actions with the currently considered visit count.

    In english: We have a budget of num_simulations divided _equally_ among log2(max_num_considered_actions) rounds.
    This function outputs a sequence of visit counts. To implement sequential halving, we MASK out the considered
    actions that do NOT match the current visit count. So a sequence of 0 0 0 0 0 0 0 0 means we consider 8 different
    actions initially (because once one is selected, its visit count is incremented, and it will be masked out in the next round).

    We need to impose an ordering to break ties, so we select the best from the unmasked actions.


    Args:
     max_num_considered_actions: The maximum number of considered actions.
       The `max_num_considered_actions` can be smaller than the number of
       actions.
     num_simulations: The total simulation budget.

    Returns:
      A tuple with visit counts. Length `num_simulations`.
    """
    if max_num_considered_actions <= 1:
        return tuple(range(num_simulations))
    log2max = int(math.ceil(math.log2(max_num_considered_actions)))
    sequence = []
    visits = [0] * max_num_considered_actions
    num_considered = max_num_considered_actions
    while len(sequence) < num_simulations:
        num_extra_visits = max(1, int(num_simulations / (log2max * num_considered)))
        for _ in range(num_extra_visits):
            sequence.extend(visits[:num_considered])
            for i in range(num_considered):
                visits[i] += 1
        # Halving the number of considered actions.
        num_considered = max(2, num_considered // 2)
    return tuple(sequence[:num_simulations])


def get_table_of_considered_visits(max_num_considered_actions, num_simulations):
    """Returns a table of sequences of visit counts.

    Args:
     max_num_considered_actions: The maximum number of considered actions.
       The `max_num_considered_actions` can be smaller than the number of
       actions.
     num_simulations: The total simulation budget.

    Returns:
      A tuple of sequences of visit counts.
      Shape [max_num_considered_actions + 1, num_simulations].
    """
    return tuple(
        get_sequence_of_considered_visits(m, num_simulations)
        for m in range(max_num_considered_actions + 1)
    )
