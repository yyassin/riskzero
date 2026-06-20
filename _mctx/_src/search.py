"""A JAX implementation of batched MCTS."""

import functools
from typing import Any, NamedTuple, Optional, Tuple, TypeVar

import chex
import jax
import jax.numpy as jnp

from _mctx._src import action_selection, base
from _mctx._src import tree as tree_lib

Tree = tree_lib.Tree
T = TypeVar("T")


# UTILITIES
# Utility function to set the values of certain indices to prescribed values.
# This is vmapped to operate seamlessly on batches.
def update(x, vals, *indices):
    """Utility function to set the values of certain indices to prescribed values.
    This is vmapped to operate seamlessly on batches.

    The *indices argument allows for setting multi-dimensional indices.

    Args:
        x: a JAX array to be updated.
        vals: values to set at the specified indices.
        *indices: indices at which to set the values.

    Example of multi-dimensional indices:
    >>> x = jnp.array([[1, 2, 3], [4, 5, 6]])
    >>> vals = jnp.array([10, 20])
    >>> indices = (jnp.array([0, 1]), jnp.array([1, 2]))
    >>> update(x, vals, *indices)
    array([[ 1, 10,  3],
              [ 4,  5, 20]])
    """
    return x.at[indices].set(vals)


batch_update = jax.vmap(update)


def update_tree_node(
    tree: Tree[T],
    node_index: chex.Array,
    prior_logits: chex.Array,
    value: chex.Array,
    embedding: chex.Array,
) -> Tree[T]:
    """Updates the tree at node index.

    Args:
      tree: `Tree` whose node is to be updated. Shape `[B]`.
      node_index: the index of the expanded node. Shape `[B]`.
      prior_logits: the prior logits to fill in for the new node, of shape
        `[B, num_actions]`.
      value: the value to fill in for the new node. Shape `[B]`.
      embedding: the state embeddings for the node. Shape `[B, ...]`.

    Returns:
      The new tree with updated nodes.
    """
    batch_size = tree_lib.infer_batch_size(tree)
    batch_range = jnp.arange(batch_size)
    chex.assert_shape(prior_logits, (batch_size, tree.num_actions))

    # When using max_depth, a leaf can be expanded multiple times so this isn't just 1.
    new_visit = tree.node_visits[batch_range, node_index] + 1
    updates = dict(
        children_prior_logits=batch_update(
            tree.children_prior_logits, prior_logits, node_index
        ),
        raw_values=batch_update(tree.raw_values, value, node_index),
        node_values=batch_update(tree.node_values, value, node_index),
        node_visits=batch_update(tree.node_visits, new_visit, node_index),
        # Update all dimensions of the embeddings.
        embeddings=jax.tree.map(
            lambda t, s: batch_update(t, s, node_index), tree.embeddings, embedding
        ),
    )
    return tree.replace(**updates)  # type: ignore


def instantiate_tree_from_root(
    root: base.RootFnOutput,
    num_simulations: int,
    root_invalid_actions: chex.Array,
    extra_data: Any,
) -> Tree:
    """Initializes tree state at search root."""
    chex.assert_rank(root.prior_logits, 2)  # (B, num_actions)
    batch_size, num_actions = root.prior_logits.shape  # type: ignore
    chex.assert_shape(root.value, [batch_size])

    # We can have at most `num_simulations` nodes, plus the root node.
    num_nodes = num_simulations + 1
    data_dtype = root.value.dtype
    batch_node = (batch_size, num_nodes)
    batch_node_action = (batch_size, num_nodes, num_actions)

    def _zeros(x):
        return jnp.zeros(batch_node + x.shape[1:], dtype=x.dtype)

    # Create a new empty tree state
    tree = Tree(
        node_visits=jnp.zeros(batch_node, dtype=jnp.int32),  # type: ignore
        raw_values=jnp.zeros(batch_node, dtype=data_dtype),  # type: ignore
        node_values=jnp.zeros(batch_node, dtype=data_dtype),  # type: ignore
        parents=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),  # type: ignore
        action_from_parent=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),  # type: ignore
        children_index=jnp.full(batch_node_action, Tree.UNVISITED, dtype=jnp.int32),  # type: ignore
        children_prior_logits=jnp.zeros(  # type: ignore
            batch_node_action, dtype=root.prior_logits.dtype
        ),
        children_values=jnp.zeros(batch_node_action, dtype=data_dtype),  # type: ignore
        children_visits=jnp.zeros(batch_node_action, dtype=jnp.int32),  # type: ignore
        children_rewards=jnp.zeros(batch_node_action, dtype=data_dtype),  # type: ignore
        children_discounts=jnp.zeros(batch_node_action, dtype=data_dtype),  # type: ignore
        embeddings=jax.tree.map(_zeros, root.embedding),  # type: ignore
        root_invalid_actions=root_invalid_actions,  # type: ignore
        extra_data=extra_data,  # type: ignore
    )

    # And fill the root node.
    root_index = jnp.full([batch_size], Tree.ROOT_INDEX)
    tree = update_tree_node(
        tree, root_index, root.prior_logits, root.value, root.embedding
    )
    return tree


# SEARCH
def search(
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    num_simulations: int,
    max_depth: Optional[int] = None,
    invalid_actions: Optional[chex.Array] = None,
    extra_data: Any = None,
    loop_fn: base.LoopFn = jax.lax.fori_loop,
):
    """Performs a full search and returns sampled actions.

    In the shape descriptions, `B` denotes the batch dimension.

    Args:
        params: params to be forwarded to root and recurrent functions.
        rng_key: random number generator state, the key is consumed.
        root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
            `prior_logits` are from a policy network. The shapes are
            `([B, num_actions], [B], [B, ...])`, respectively.
        recurrent_fn: a callable to be called on the leaf nodes and unvisited
            actions retrieved by the simulation step for EXPANSION, which takes as args
            `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
            and the new state embedding. The `rng_key` argument is consumed.
        root_action_selection_fn: function used to select final action at the root.
        interior_action_selection_fn: function used to select an action during simulation.
        num_simulations: the number of simulations.
        max_depth: maximum search tree depth allowed during simulation, defined as
            the number of edges from the root to a leaf node.
        invalid_actions: a mask with invalid actions at the root. In the
            mask, invalid actions have ones, and valid actions have zeros.
            Shape `[B, num_actions]`.
        extra_data: extra data passed to `tree.extra_data`. Shape `[B, ...]`.
        loop_fn: Function used to run the simulations. It may be required to pass
            hk.fori_loop if using this function inside a Haiku module.

        Returns:
            `SearchResults` containing outcomes of the search, e.g. `visit_counts`
            `[B, num_actions]`.
    """
    # Create action selection function that switches between root and interior selection.
    action_selection_fn = action_selection.switching_action_selection_wrapper(
        root_action_selection_fn=root_action_selection_fn,
        interior_action_selection_fn=interior_action_selection_fn,
    )

    # Do simulation, expansion, and backward steps in a loop, num_simulations times.
    batch_size = root.value.shape[0]  # type: ignore
    batch_range = jnp.arange(batch_size)
    max_depth = max_depth if max_depth is not None else num_simulations
    invalid_actions = (
        invalid_actions
        if invalid_actions is not None
        else jnp.zeros_like(root.prior_logits)
    )

    def body_fun(sim, loop_state):
        rng_key, tree = loop_state
        rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3)

        # Simulation is vmapped and expects batched rng keys.
        simulate_keys = jax.random.split(simulate_key, batch_size)  # (b,)

        # SIMULATION
        # Run simulation step. Retrieve index of node reached and action to evaluate.
        parent_index, action = simulate(
            simulate_keys, tree, action_selection_fn, max_depth
        )
        # A node first expanded on simulation `i`, will have node index `i`.
        # Node 0 corresponds to the root node.
        next_node_index = tree.children_index[batch_range, parent_index, action]
        next_node_index = jnp.where(
            next_node_index == Tree.UNVISITED, sim + 1, next_node_index
        )

        # EXPANSION
        # Expand the node and action selected by simulation, update the expanded leaves/transition.
        tree = expand(
            params,  # type: ignore
            expand_key,
            tree,
            recurrent_fn,
            parent_index,
            action,
            next_node_index,
        )

        # BACKWARD
        # Backpropagate the value of the expanded node to all its parents.
        tree = backward(tree, next_node_index)
        loop_state = rng_key, tree
        return loop_state

    # Allocate all necessary storage for the tree.
    tree = instantiate_tree_from_root(
        root,
        num_simulations,
        root_invalid_actions=invalid_actions,
        extra_data=extra_data,
    )

    # Loop for 0 to num_simulations - 1.
    _, tree = loop_fn(0, num_simulations, body_fun, (rng_key, tree))
    return tree


# SIMULATION
class _SimulationState(NamedTuple):
    """The state for the simulation while loop."""

    rng_key: chex.PRNGKey
    node_index: int
    action: int
    next_node_index: int
    depth: int
    is_continuing: bool


# simulation is vmapped over keys and trees
@functools.partial(jax.vmap, in_axes=[0, 0, None, None], out_axes=0)
def simulate(
    rng_key: chex.PRNGKey,
    tree: Tree,
    action_selection_fn: base.InteriorActionSelectionFn,
    max_depth: int,
) -> Tuple[chex.Array, chex.Array]:
    """Traverses the tree until reaching an unvisited action or `max_depth`.

    Each simulation **starts from the root** and keeps selecting actions traversing
    the tree until a leaf or `max_depth` is reached.

    Args:
        rng_key: random number generator state, the key is consumed.
        tree: _unbatched_ MCTS tree state.
        action_selection_fn: function used to select an action during simulation.
        max_depth: maximum search tree depth allowed during simulation.

    Returns:
        `(parent_index, action)` tuple, where `parent_index` is the index of the
        node reached at the end of the simulation, and the `action` is the action to
        evaluate from the `parent_index`.
    """

    def cond_fun(state: _SimulationState) -> bool:
        return state.is_continuing

    def body_fun(state: _SimulationState) -> _SimulationState:
        # Preparing the next simulation state.
        node_index = state.next_node_index
        rng_key, action_selection_key = jax.random.split(state.rng_key)
        # Select action.
        action = action_selection_fn(
            action_selection_key,
            tree,
            node_index,  # type: ignore
            state.depth,  # type: ignore
        )
        next_node_index = tree.children_index[node_index, action]
        # The returned action will be visited.
        depth = state.depth + 1
        is_before_depth_cutoff = depth < max_depth
        is_visited = next_node_index != Tree.UNVISITED
        # Terminate the simulation if the action hasn't been visited or we reach `max_depth``.
        is_continuing = jnp.logical_and(is_visited, is_before_depth_cutoff)
        return _SimulationState(
            rng_key=rng_key,
            node_index=node_index,
            action=action,  # type: ignore
            next_node_index=next_node_index,  # type: ignore
            depth=depth,
            is_continuing=is_continuing,  # type: ignore
        )

    # Start the simulation from the root node.
    node_index = jnp.array(Tree.ROOT_INDEX, dtype=jnp.int32)
    depth = jnp.zeros((), dtype=tree.children_prior_logits.dtype)
    initial_state = _SimulationState(
        rng_key=rng_key,
        node_index=tree.NO_PARENT,
        action=tree.NO_PARENT,
        next_node_index=node_index,  # type: ignore
        depth=depth,  # type: ignore
        is_continuing=jnp.array(True),  # type: ignore
    )
    # Simulate until reaching a leaf node or `max_depth`.
    end_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)
    # Returning a node with a selected action.
    # The action can be already visited, if the max_depth is reached.
    return end_state.node_index, end_state.action  # type: ignore


# EXPANSION
def expand(
    params: chex.Array,
    rng_key: chex.PRNGKey,
    tree: Tree[T],
    recurrent_fn: base.RecurrentFn,
    parent_index: chex.Array,
    action: chex.Array,
    next_node_index: chex.Array,
) -> Tree[T]:
    """Create and evaluate child nodes from given nodes and unvisited actions.

    Args:
        params: params to be forwarded to recurrent function.
        rng_key: random number generator state.
        tree: the MCTS tree state to update.
        recurrent_fn: a callable to be called on the leaf nodes and unvisited
            actions retrieved by the simulation step, which takes as args
            `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
            and the new state embedding. The `rng_key` argument is consumed.
        parent_index: the index of the parent node, from which the action will be
            expanded. Shape `[B]`.
        action: the action to expand. Shape `[B]`.
        next_node_index: the index of the newly expanded node. This can be the index
            of an existing node, if `max_depth` is reached. Shape `[B]`.

    Returns:
      tree: updated MCTS tree state.
    """
    batch_size = tree_lib.infer_batch_size(tree)
    batch_range = jnp.arange(batch_size)
    chex.assert_shape([parent_index, action, next_node_index], (batch_size,))

    # Retrieve states for nodes to be evaluated.
    embedding = jax.tree.map(lambda x: x[batch_range, parent_index], tree.embeddings)

    # Expand (s, a) -> s'. For s', get priors, value and immediate reward for (s, a).
    step, embedding = recurrent_fn(params, rng_key, action, embedding)
    chex.assert_shape(step.prior_logits, [batch_size, tree.num_actions])
    chex.assert_shape(step.reward, [batch_size])
    chex.assert_shape(step.discount, [batch_size])
    chex.assert_shape(step.value, [batch_size])

    # Update the expanded node.
    tree = update_tree_node(
        tree, next_node_index, step.prior_logits, step.value, embedding
    )

    # Return updated tree topology.
    return tree.replace(  # type: ignore
        # Update children_index[parent_index, action] to next_node_index.
        children_index=batch_update(
            tree.children_index, next_node_index, parent_index, action
        ),
        # Update children_rewards[parent_index, action] to step.reward.
        children_rewards=batch_update(
            tree.children_rewards, step.reward, parent_index, action
        ),
        # Update children_values[parent_index, action] to step.value.
        children_discounts=batch_update(
            tree.children_discounts, step.discount, parent_index, action
        ),
        # Update parents[next_node_index] to parent_index.
        parents=batch_update(tree.parents, parent_index, next_node_index),
        # Update action_from_parent[next_node_index] to action.
        action_from_parent=batch_update(
            tree.action_from_parent, action, next_node_index
        ),
    )


# BACKWARD
@jax.vmap
def backward(tree: Tree[T], leaf_index: chex.Numeric) -> Tree[T]:
    """Goes up and updates the tree until all nodes reached the root.

    Args:
      tree: the MCTS tree state to update, without the batch size.
      leaf_index: the node index from which to backpropagate the value.

    Returns:
      Updated MCTS tree state.
    """

    def cond_fun(loop_state):
        _, _, index = loop_state
        return index != Tree.ROOT_INDEX

    def body_fun(loop_state):
        # Here we update the value of our parent, so we start by reversing.
        tree, leaf_value, index = loop_state

        parent_index = tree.parents[index]
        count = tree.node_visits[parent_index]
        action = tree.action_from_parent[index]
        reward = tree.children_rewards[parent_index, action]

        # Update parent value
        bp_value = reward + tree.children_discounts[parent_index, action] * leaf_value
        parent_value = (tree.node_values[parent_index] * count + bp_value) / (
            count + 1.0
        )
        children_values = tree.node_values[index]

        # Update visit counts
        children_counts = tree.children_visits[parent_index, action] + 1

        tree = tree.replace(
            # Update node values with new parent value
            node_values=update(tree.node_values, parent_value, parent_index),
            # The expanded node was updated in the expansion. We just increment parent visits as we go up.
            node_visits=update(tree.node_visits, count + 1, parent_index),
            # Update children values with prior values (parent of this node is updated in the next iteration).
            children_values=update(
                tree.children_values, children_values, parent_index, action
            ),
            # Update visits of the children of this parent node.
            children_visits=update(
                tree.children_visits, children_counts, parent_index, action
            ),
        )

        return tree, bp_value, parent_index

    leaf_index = jnp.asarray(leaf_index, dtype=jnp.int32)
    loop_state = (tree, tree.node_values[leaf_index], leaf_index)
    tree, _, _ = jax.lax.while_loop(cond_fun, body_fun, loop_state)
    return tree
