"""A data structure used to hold / inspect search data for a batch of inputs."""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Generic, TypeVar

import chex
import jax
import jax.numpy as jnp
from flax import struct

T = TypeVar("T")


# A number of aggregate statistics and predictions are extracted from the
# search data and returned to the user for further processing.
@chex.dataclass(frozen=True)
class SearchSummary:
    """Stats from MCTS search."""

    visit_counts: jnp.ndarray
    visit_probs: jnp.ndarray
    value: jnp.ndarray
    qvalues: jnp.ndarray


def unbatched_qvalues(tree: Tree, index: int) -> int:
    """Computes the unbatched Q-value for all children of a node.
    We define Q(v) = r + discount * v'"""
    # Ensure input is not batched.
    chex.assert_rank(tree.children_discounts, 2)
    return (
        tree.children_rewards[index]  # type: ignore
        + tree.children_discounts[index] * tree.children_values[index]  # type: ignore
    )  # (num_actions,)


def unbatched_trajectory_qvalues(tree: Tree, index: int) -> int:
    """Computes the unbatched trajectory Q-value for all children of a node.
    We define Q(v) = v'; the discounting is already accounted for in the value (during simulation)."""
    # Ensure input is not batched.
    chex.assert_rank(tree.children_discounts, 2)
    return (
        tree.children_values[index]  # type: ignore
    )  # (num_actions,)


def infer_batch_size(tree: Tree) -> int:
    """Recovers batch size from `Tree` data structure."""
    if tree.node_values.ndim != 2:
        raise ValueError("Input tree is not batched.")

    # Ensure that the tree is batched.
    chex.assert_equal_shape_prefix(jax.tree_util.tree_leaves(tree), 1)
    return tree.node_values.shape[0]


@struct.dataclass(frozen=True)
class Tree(Generic[T]):
    """State of a search tree.

    This `Tree` dataclass is used to hold and inspect search data for an input batch.
    Below, `B` denotes the batch dimension, `N` denotes the number of nodes in the tree,
    and `num_actions` is the number of action available (we assume they are fixed, they can be masked).


    Attributes:
        node_visits: `[B, N]` the visit counts for each node.
        raw_values: `[B, N]` the raw value for each node.
        node_values: `[B, N]` the cumulative search value for each node.
        parents: `[B, N]` the node index for the parent for each node.
        action_from_parent: `[B, N]` action to take from the parent to reach each node.
        children_index: `[B, N, num_actions]` the node index of the children for each action.
        children_prior_logits: `[B, N, num_actions]` the action prior logits of each node.
        children_visits: `[B, N, num_actions]` the visit counts for children for each action.
        children_rewards: `[B, N, num_actions, ...]` the immediate reward for each action. Can be quantiles.
        children_discounts: `[B, N, num_actions]` the discount between the
            `children_rewards` and the `children_values`.
        children_values: `[B, N, num_actions]` the value of the next node after the action.
        embeddings: `[B, N, ...]` the state embeddings of each node.
        root_invalid_actions: `[B, num_actions]` a mask with invalid actions at the
            root. In the mask, invalid actions have ones, and valid actions have zeros.
        extra_data: `[B, ...]` extra data passed to the search.
    """

    node_visits: jnp.ndarray  # [B, N]
    raw_values: jnp.ndarray  # [B, N]
    node_values: jnp.ndarray  # [B, N]
    parents: jnp.ndarray  # [B, N]
    action_from_parent: jnp.ndarray  # [B, N]
    children_index: jnp.ndarray  # [B, N, num_actions]
    children_prior_logits: jnp.ndarray  # [B, N, num_actions]
    children_visits: jnp.ndarray  # [B, N, num_actions]
    children_rewards: jnp.ndarray  # [B, N, num_actions, ...]
    children_discounts: jnp.ndarray  # [B, N, num_actions]
    children_values: jnp.ndarray  # [B, N, num_actions]
    embeddings: Any  # [B, N, ...]
    root_invalid_actions: jnp.ndarray  # [B, num_actions]
    extra_data: T  # [B, ...]

    # The following attributes are class variables (and should not be set on Tree instances).
    ROOT_INDEX: ClassVar[int] = 0
    NO_PARENT: ClassVar[int] = -1
    UNVISITED: ClassVar[int] = -1

    # We don't want to flatten the function in pytree.
    unbatched_q_value_fn: Callable[[Tree, int], int] = struct.field(
        pytree_node=False, default=unbatched_qvalues
    )

    @property
    def num_actions(self):
        """Number of actions available in the tree. Fixed across all nodes."""
        return self.children_index.shape[-1]

    @property
    def num_simulations(self):
        """Number of simulations in the tree.
        This is the number of nodes in the tree minus one, since the root node is not counted.
        """
        return self.node_visits.shape[-1] - 1

    def qvalues(self, indices):
        """Compute q-values for any node indices in the tree (batched or not)."""
        if jnp.asarray(indices).shape:
            return jax.vmap(self.unbatched_q_value_fn)(self, indices)
        else:
            return self.unbatched_q_value_fn(self, indices)

    def summary(self) -> SearchSummary:
        """Extract summary statistics for the root node."""
        # Get state-action values for the root nodes.
        chex.assert_rank(self.node_values, 2)
        value = self.node_values[:, Tree.ROOT_INDEX]
        batch_size = value.shape[0]
        root_indices = jnp.full((batch_size,), Tree.ROOT_INDEX)
        qvalues = self.qvalues(root_indices)  # (batch_size, self.num_actions)

        # Extract visit counts and induced probabilities for the root nodes.
        visit_counts = self.children_visits[:, Tree.ROOT_INDEX].astype(
            value.dtype
        )  # (batch_size, num_actions)
        total_counts = jnp.sum(visit_counts, axis=-1, keepdims=True)  # (batch_size, 1)
        visit_probs = visit_counts / jnp.maximum(
            total_counts, 1
        )  # (batch_size, num_actions)
        # If children are not visited, set equiprobable distribution.
        visit_probs = jnp.where(total_counts > 0, visit_probs, 1 / self.num_actions)
        return SearchSummary(
            visit_counts=visit_counts,  # type: ignore
            visit_probs=visit_probs,  # type: ignore
            value=value,  # type: ignore
            qvalues=qvalues,  # type: ignore
        )
