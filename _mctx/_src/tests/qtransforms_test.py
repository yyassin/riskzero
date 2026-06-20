"""Tests for `qtransforms.py`."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from _mctx._src import qtransforms
from _mctx._src import tree as tree_lib


class QtransformsTest(absltest.TestCase):
    def test_qtransform_by_parent_and_siblings(self):
        # Create a mock tree with a root node that has three children.
        tree = tree_lib.Tree(
            node_values=jnp.array(  # type: ignore
                [0.5, 0.6, 0.7, 0.9]
            ),  # V(root) = 0.5
            children_values=jnp.array(  # type: ignore
                [[0.6, 0.7, 0.9]]
            ),  # V(children) = [0.6, 0.7, 0.9]
            children_visits=jnp.array(  # type: ignore
                [[0, 2, 1]]
            ),  # Visits: [0, 2, 0]
            children_rewards=jnp.array(  # type: ignore
                [[0, 0.2, 0.3]]
            ),  # Rewards: [0, 0.2, 0.3]
            children_discounts=jnp.array(  # type: ignore
                [[0.9, 0.9, 0.9]]
            ),  # Discounts: [0.9, 0.9, 0.9]
            # ====The below does not matter for this test, but is needed to create a valid tree====
            parents=jnp.array([-1, 0, 0, 0]),  # Root has no parent. # type: ignore
            action_from_parent=jnp.array(  # type: ignore
                [-1, 0, 1, 2]
            ),  # Actions to reach children.
            children_index=jnp.array([[1, 2, 3]]),  # Children indices. # type: ignore
            children_prior_logits=jnp.array(  # type: ignore
                [[0.1, 0.2, 0.3]]
            ),  # Prior logits.
            node_visits=jnp.array(  # type: ignore
                [3, 0, 2, 1]
            ),
            raw_values=jnp.array([0.0, 0.0, 0.0, 0.0]),  # Raw values. # type: ignore
            embeddings=jnp.array(  # type: ignore
                [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
            ),  # Dummy embeddings.
            root_invalid_actions=jnp.array(  # type: ignore
                [0, 0, 0, 0]
            ),  # No invalid actions at root.
            extra_data=jnp.array([[0, 0, 0, 0]]),  # type: ignore
        )

        # Test the transformation for a specific node index.
        node_index = jnp.array(0)
        transformed_qvalues = qtransforms.qtransform_by_parent_and_siblings(
            tree, node_index
        )

        # V(node) = 0.5, qvalues = [0, 0.83, 1.11]
        # The max is 1.11 and min, among visited, is 0.5.
        # We expect the normalized qvalues to be:
        expected_qvalues = jnp.array([0.0, 0.54098, 1.0])
        np.testing.assert_allclose(
            transformed_qvalues, expected_qvalues, rtol=1e-5, atol=1e-5
        )

    def test_mix_value(self):
        """Tests the output of _compute_mixed_value()."""
        raw_value = jnp.array(-0.8)
        prior_logits = jnp.array([-jnp.inf, -1.0, 2.0, -jnp.inf])
        probs = jax.nn.softmax(prior_logits)
        visit_counts = jnp.array([0, 4.0, 4.0, 0])
        qvalues = 10.0 / 54 * jnp.array([20.0, 3.0, -1.0, 10.0])
        mix_value = qtransforms._compute_mixed_value(
            raw_value, qvalues, visit_counts, probs
        )

        num_simulations = jnp.sum(visit_counts)
        expected_mix_value = (
            1.0
            / (num_simulations + 1)
            * (
                raw_value
                + num_simulations * (probs[1] * qvalues[1] + probs[2] * qvalues[2])
            )
        )
        np.testing.assert_allclose(expected_mix_value, mix_value)

    def test_mix_value_with_zero_visits(self):
        """Tests that zero visit counts do not divide by zero."""
        raw_value = jnp.array(-0.8)
        prior_logits = jnp.array([-jnp.inf, -1.0, 2.0, -jnp.inf])
        probs = jax.nn.softmax(prior_logits)
        visit_counts = jnp.array([0, 0, 0, 0])
        qvalues = jnp.zeros_like(probs)
        with jax.debug_nans():
            mix_value = qtransforms._compute_mixed_value(
                raw_value, qvalues, visit_counts, probs
            )

        np.testing.assert_allclose(raw_value, mix_value)


if __name__ == "__main__":
    absltest.main()
