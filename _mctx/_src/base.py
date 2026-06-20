"""Core types used in mctx."""

from typing import Any, Callable, Generic, Optional, Protocol, Tuple, TypeVar

import chex
import jax
import jax.numpy as jnp

from _mctx._src import tree

# (Model) Parameters are arbitrary nested structures of `chex.Array`.
# A nested structure is either a single object, or a collection (list, tuple,
# dictionary, etc.) of other nested structures.
Params = chex.ArrayTree


# The model used to search is expressed by a `RecurrentFn` function that is used for EXPANSION.
# The `RecurrentFn` takes `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
# and the new state embedding.
@chex.dataclass(frozen=True)
class RecurrentFnOutput:
    """The output of a `RecurrentFn`.

    reward: `[B]` an approximate reward from the state-action transition.
    discount: `[B]` the discount between the `reward` and the `value`.
    prior_logits: `[B, num_actions]` the logits produced by a policy network.
    value: `[B]` an approximate value of the state after the state-action
      transition.
    """

    reward: chex.Array  # [B]
    discount: chex.Array  # [B]
    prior_logits: chex.Array  # [B, num_actions]
    value: chex.Array  # [B]


Action = chex.Array
RecurrentState = chex.ArrayTree
RecurrentFn = Callable[
    [Params, chex.PRNGKey, Action, Any],
    Tuple[RecurrentFnOutput, Any],
]


@chex.dataclass(frozen=True)
class RootFnOutput:
    """The output of a representation network.

    prior_logits: `[B, num_actions]` the logits produced by a policy network.
    value: `[B]` an approximate value of the current state.
    embedding: `[B, ...]` the inputs to the next `recurrent_fn` call.
    """

    prior_logits: chex.Array  # [B, num_actions]
    value: chex.Array  # [B]
    embedding: Any  # [B, ...]


@chex.dataclass(frozen=True)
class RiskRootFnOutput:
    """The output of a representation network.

    return_history: `[B, num_quantiles]` the quantile distribution of the returns so far.
    prior_logits: `[B, num_actions]` the logits produced by a policy network.
    value: `[B, num_quantiles]` an approximate value of the current state.
    embedding: `[B, ...]` the inputs to the next `recurrent_fn` call.
    """

    return_history: chex.Array  # [B, num_quantiles]
    prior_logits: chex.Array  # [B, num_actions]
    value: chex.Array  # [B, num_quantiles]
    embedding: Any  # [B, ...]


T = TypeVar("T")
RootT = TypeVar("RootT", RootFnOutput, RiskRootFnOutput)


@chex.dataclass(frozen=True)
class PolicyOutput(Generic[T]):
    """The output of the MCTS search policy.

    action: `[B]` the proposed action.
    action_weights: `[B, num_actions]` the targets used to train a policy network.
      The action weights sum to one. Usually, the policy network is trained by
      cross-entropy: `cross_entropy(labels=stop_gradient(action_weights), logits=prior_logits)`.
    search_tree: `[B, ...]` the search tree of the finished search.
    """

    action: chex.Array
    action_weights: chex.Array
    search_tree: tree.Tree[T]


# Action selection functions specify how to pick nodes to expand in the tree.
NodeIndices = chex.Array
Depth = chex.Array
RootActionSelectionFn = Callable[[chex.PRNGKey, tree.Tree, NodeIndices], chex.Array]
InteriorActionSelectionFn = Callable[
    [chex.PRNGKey, tree.Tree, NodeIndices, Depth], chex.Array
]
QTransform = Callable[[tree.Tree, chex.Array], chex.Array]
# LoopFn has the same interface as jax.lax.fori_loop.
LoopFn = Callable[
    [int, int, Callable[[Any, Any], Any], Tuple[chex.PRNGKey, tree.Tree]],
    Tuple[chex.PRNGKey, tree.Tree],
]
SampleBasedUtilityFn = Callable[[chex.Array], chex.Array]

# Accept subclasses of `RootFnOutput` for the root of the search tree.
SearchFnRootT = TypeVar("SearchFnRootT", contravariant=True)


class SearchFn(Protocol[SearchFnRootT]):
    def __call__(
        self,
        params: "Params",
        rng_key: chex.PRNGKey,
        *,
        root: SearchFnRootT,
        recurrent_fn: "RecurrentFn",
        root_action_selection_fn: "RootActionSelectionFn",
        interior_action_selection_fn: "InteriorActionSelectionFn",
        num_simulations: int,
        max_depth: Optional[int] = None,
        invalid_actions: Optional[chex.Array] = None,
        extra_data: Any = None,
        loop_fn: "LoopFn" = jax.lax.fori_loop,
    ) -> "tree.Tree[T]":  # type: ignore
        return tree.Tree(
            node_visits=jnp.zeros((0, 0), jnp.int32),
            raw_values=jnp.zeros((0, 0), jnp.float32),
            node_values=jnp.zeros((0, 0), jnp.float32),
            parents=jnp.zeros((0, 0), jnp.int32),
            action_from_parent=jnp.zeros((0, 0), jnp.int32),
            children_index=jnp.zeros((0, 0, 0), jnp.int32),
            children_prior_logits=jnp.zeros((0, 0, 0), jnp.float32),
            children_visits=jnp.zeros((0, 0, 0), jnp.int32),
            children_rewards=jnp.zeros((0, 0, 0), jnp.float32),
            children_discounts=jnp.zeros((0, 0, 0), jnp.float32),
            children_values=jnp.zeros((0, 0, 0), jnp.float32),
            embeddings=None,
            root_invalid_actions=jnp.zeros((0, 0), jnp.int32),
            extra_data=extra_data,
        )
