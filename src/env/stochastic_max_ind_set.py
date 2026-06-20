# Stochastic Maximum Independent Set environment
#
# A graph-structured problem where a policy sequentially selects nodes to
# include in an independent set. Instances are loaded from a pre-generated
# dataset; node weights are stochastic, sampled from one of three distributions
# (types 0–2) defined in datasets/smis/cvar_estimation.py.
#
# At each step the policy selects one available node to add to the IS. A node
# becomes unavailable once it or any of its neighbors has been selected.
# An episode terminates once no available nodes remain.
#
# Action space : {0, …, num_nodes − 1}  (index of the node to select)
# Observation  : graph-structured dict containing node features,
#                senders, receivers, and auxiliary scalars.

import dataclasses
import os
from typing import Any, List, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from pgx import core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

from src.datasets.smis.cvar_estimation import sample_weights

# ---------------------------------------------------------------------------
# pgx monkey patch
# ---------------------------------------------------------------------------
# The graph experiments pass batches of problem instances to the environment.
# We extend core.Env with init_v2 here rather than modifying the installed
# library.


def _env_init_v2(
    self,
    key: PRNGKey,
    iteration: jnp.ndarray,
    offset: jnp.ndarray,
    num_envs: int,
    split: int,
) -> core.State:
    state = self._init(key, iteration, offset, num_envs, split)
    observation = self.observe(state)
    return state.replace(observation=observation)


core.Env.init_v2 = _env_init_v2  # type: ignore

INIT_NODES = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])
SENDERS = jnp.array([0, 1, 2, 2, 3, 3, 3, 4])
RECEIVERS = jnp.array([1, 2, 3, 4, 4, 5, 6, 5])


class EnvState(NamedTuple):
    adj: Array
    nodes: Array
    node_types: Array
    selected_nodes: Array
    available_nodes: Array
    senders: Array
    receivers: Array
    optimal_cvar_value: Array
    expected_value: Array
    cvar_100: Array
    key: Array


@dataclass
class Instance:
    num_nodes: jnp.ndarray
    node_types: jnp.ndarray
    optimal_cvar_value: jnp.ndarray
    expected_value: jnp.ndarray
    cvar_100: jnp.ndarray
    init_nodes: jnp.ndarray
    senders: jnp.ndarray
    receivers: jnp.ndarray


def load_stochastic_mis_instance(instance_id, dataset_path):
    instance_file = f"instance_{instance_id}.txt"
    instance_path = os.path.join(dataset_path, instance_file)

    with open(instance_path, "r") as f:
        lines = f.readlines()

    # lines[0]: cvar_100_estimate (E-value of the E-optimal IS)
    cvar_100 = jnp.array(float(lines[0].strip()), dtype=jnp.float32)

    # lines[1]: cvar_alpha_estimate (CVaR_alpha of the risk-averse IS)
    optimal_cvar = jnp.array(float(lines[1].strip()), dtype=jnp.float32)

    # lines[2]: expected_estimate (E-value of the risk-averse IS)
    expected_value = jnp.array(float(lines[2].strip()), dtype=jnp.float32)

    # lines[3]: number of nodes
    num_nodes = jnp.array(int(lines[3].strip()))

    # lines[4]: number of edges
    num_edges = int(lines[4].strip())

    # Remaining lines: edges
    edge_lines = lines[5 : 5 + num_edges]
    edges = [list(map(int, line.strip().split())) for line in edge_lines]

    # And then node types
    node_types = lines[5 + num_edges :]
    node_types = [list(map(int, line.strip().split()))[0] for line in node_types]
    node_types = jnp.array(node_types, dtype=jnp.int32)

    # Parse lines
    edges = np.array(edges)

    # Separate into columns
    senders = edges[:, 0]
    receivers = edges[:, 1]

    senders = jnp.array(senders)
    receivers = jnp.array(receivers)
    init_nodes = jnp.arange(num_nodes, dtype=jnp.int32)

    return Instance(
        num_nodes=num_nodes,
        node_types=node_types,
        optimal_cvar_value=optimal_cvar,
        expected_value=expected_value,
        cvar_100=cvar_100,
        init_nodes=init_nodes,
        senders=senders,
        receivers=receivers,
    )


def batch_stochastic_mis_instances(instances: List[Instance]) -> Instance:
    return Instance(
        num_nodes=jnp.stack([inst.num_nodes for inst in instances]),
        node_types=jnp.stack([inst.node_types for inst in instances]),
        optimal_cvar_value=jnp.stack([inst.optimal_cvar_value for inst in instances]),
        expected_value=jnp.stack([inst.expected_value for inst in instances]),
        cvar_100=jnp.stack([inst.cvar_100 for inst in instances]),
        init_nodes=jnp.stack([inst.init_nodes for inst in instances]),
        senders=jnp.stack([inst.senders for inst in instances]),
        receivers=jnp.stack([inst.receivers for inst in instances]),
    )


def make_adjacency_matrix(nodes: Array, senders: Array, receivers: Array) -> Array:
    """Create an adjacency matrix from the nodes and edges, as an undirected graph."""
    num_nodes = len(nodes)
    adjacency_matrix = jnp.zeros((num_nodes, num_nodes), dtype=jnp.bool_)
    adjacency_matrix = adjacency_matrix.at[senders, receivers].set(True)
    adjacency_matrix = adjacency_matrix.at[receivers, senders].set(True)
    return adjacency_matrix.astype(jnp.float32)


def new_state(state: EnvState, node: jnp.ndarray) -> EnvState:
    """Select a node. This will render that node selected.
    The new unavailable nodes are all its neighbors.
    """
    new_selected_nodes = state.selected_nodes.at[node].set(True)

    neighbor_mask = state.adj[node] > 0  # shape (N,), bool
    new_available_nodes: Array = jnp.where(neighbor_mask, False, state.available_nodes)

    # The node itself is also unavailable after selection
    new_available_nodes = new_available_nodes.at[node].set(False)

    return state._replace(
        selected_nodes=new_selected_nodes,
        available_nodes=new_available_nodes,
    )


def reward(state: EnvState, node: jnp.ndarray, rng_key: jnp.ndarray) -> jnp.ndarray:
    """Calculate the reward for selecting a node."""
    # If the node is available then the reward is based on its type. Otherwise, it's -5.
    reward = sample_weights(state.node_types[node], rng_key)
    return jax.lax.cond(
        state.available_nodes[node],
        lambda: reward,
        lambda: -1000.0,
    )


def is_terminal(state: EnvState) -> jnp.ndarray:
    """Check if the state is terminal."""
    # The state is terminal if there are no available nodes left.
    return jnp.all(~state.available_nodes)


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((2,), dtype=jnp.float32)
    rewards: Array = jnp.float32([0.0])
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    iteration: Array = jnp.int32(0)
    split: Array = jnp.int32(0)  # 0 is train, 1 is test
    offset: Array = jnp.int32(0)
    num_envs: Array = jnp.int32(1)
    _step_count: Array = jnp.int32(0)
    legal_action_mask: Array = jnp.ones((0,), dtype=jnp.bool_)
    _x: EnvState = EnvState(
        adj=make_adjacency_matrix(INIT_NODES, SENDERS, RECEIVERS),
        nodes=INIT_NODES,
        node_types=jnp.zeros((0,), dtype=jnp.int32),
        selected_nodes=jnp.zeros((0,), dtype=jnp.bool_),
        available_nodes=jnp.ones((0,), dtype=jnp.bool_),
        senders=SENDERS,
        receivers=RECEIVERS,
        optimal_cvar_value=jnp.array(0.0),
        expected_value=jnp.array(0.0),
        cvar_100=jnp.array(0.0),
        key=jax.random.PRNGKey(0),
    )

    @property
    def env_id(self) -> core.EnvId:
        return "max_ind_set"  # type: ignore

    @property
    def x(self) -> EnvState:
        return self._x

    def replace(self, **kwargs: Any) -> "State":  # type: ignore
        return dataclasses.replace(self, **kwargs)  # type: ignore[misc]


class StochasticMaxIndependentSet(core.Env):
    def __init__(self, instances: Instance, using_legal_actions: bool = True):
        super().__init__()
        self._instances = instances
        self._num_nodes = instances.num_nodes[0].astype(jnp.int32)
        self.using_legal_actions = using_legal_actions

    def _init(
        self,
        key: jnp.ndarray,
        iteration: jnp.ndarray = jnp.array(-1),
        offset: jnp.ndarray = jnp.array(0),
        num_envs: jnp.ndarray = jnp.array(1),
        _split: jnp.ndarray = jnp.array(0),  # 1 is test
    ) -> State:  # type: ignore
        num_instances = self._instances.init_nodes.shape[0] // 2
        idx = (iteration % (num_instances // num_envs)) + offset * (num_instances // num_envs)
        idx = jax.lax.cond(
            _split == 1,  # Test split
            lambda: idx + num_instances,  # Use the second half as unseen test
            lambda: idx,
        )
        init_nodes = self._instances.init_nodes[idx]
        senders = self._instances.senders[idx]
        receivers = self._instances.receivers[idx]
        node_types = self._instances.node_types[idx]

        return State(
            iteration=iteration,
            offset=offset,
            num_envs=num_envs,
            legal_action_mask=jnp.ones((self._num_nodes,), dtype=jnp.bool_),
            split=_split,
            _x=EnvState(
                adj=make_adjacency_matrix(init_nodes, senders, receivers),
                nodes=init_nodes,
                node_types=node_types,
                selected_nodes=jnp.zeros((self._num_nodes,), dtype=jnp.bool_),
                available_nodes=jnp.ones((self._num_nodes,), dtype=jnp.bool_),
                senders=senders,
                receivers=receivers,
                optimal_cvar_value=self._instances.optimal_cvar_value[idx],
                expected_value=self._instances.expected_value[idx],
                cvar_100=self._instances.cvar_100[idx],
                key=key,
            ),
        )

    def _observe(self, state: State, player_id: None = None) -> Array:  # type: ignore
        """The observation are node features, which are the selected nodes and available nodes. Alongside the senders and receivers.

        Auxiliary features are the ratio of unavailable nodes, the ratio of covered edges and a biasing 1 term.
        """

        node_features = jnp.stack([state.x.selected_nodes, state.x.available_nodes], axis=1).astype(
            jnp.float32
        )  # (NUM_NODES, 2)

        # There are 4 types of nodes, so we can use one-hot encoding
        node_types = state.x.node_types  # (NUM_NODES,)
        node_types_one_hot = jax.nn.one_hot(
            node_types, num_classes=3, dtype=jnp.float32
        )  # (NUM_NODES, 4)

        node_features = jnp.concatenate(
            [node_features, node_types_one_hot], axis=1
        )  # (NUM_NODES, 6)

        senders = state.x.senders  # (NUM_EDGES,)
        receivers = state.x.receivers  # (NUM_EDGES,)
        unavailable_ratio = jnp.sum(~state.x.available_nodes) / self._num_nodes  # (scalar)
        # Covered edges are those edges whose senders and receivers are both not selected.
        covered_edges = jnp.sum(
            ~jnp.logical_or(
                state.x.selected_nodes[state.x.senders],
                state.x.selected_nodes[state.x.receivers],
            )
        )
        covered_edges_ratio = covered_edges / state.x.senders.shape[0]  # (scalar)
        bias = jnp.array(1.0)  # (scalar)

        num_chosen = state._step_count / self._num_nodes
        aux = jnp.stack([unavailable_ratio, covered_edges_ratio, num_chosen, bias], axis=0).astype(
            jnp.float32
        )
        return {
            "node_features": node_features,
            "node_types": state.x.node_types,
            "senders": senders,
            "receivers": receivers,
            "aux": aux,
        }

    def _step(self, state: State, action: jnp.ndarray, key: jnp.ndarray) -> State:
        next_state = new_state(state.x, action)
        r = reward(state.x, action, key)
        is_term = is_terminal(next_state)

        # Terminal after num node steps, regardless of the state
        is_term = (state._step_count >= self._num_nodes) | is_term

        legal_action_mask = jax.lax.cond(
            self.using_legal_actions,
            lambda: next_state.available_nodes,
            lambda: jnp.ones((self._num_nodes,), dtype=jnp.bool_),
        )

        return state.replace(
            _x=next_state,
            rewards=jnp.array([r], dtype=jnp.float32),
            terminated=is_term,
            truncated=False,  # No truncation in this environment
            legal_action_mask=legal_action_mask,
        )

    @property
    def id(self) -> core.EnvId:
        return "max_ind_set"  # type: ignore

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 1


if __name__ == "__main__":
    # Load an instance from the dataset
    folder_path = "./datasets/maximum-independent-set"
    instance_ids = [0] * 10

    print("Loading instances...")
    loaded_instances = batch_stochastic_mis_instances(
        [load_stochastic_mis_instance(instance_id, folder_path) for instance_id in instance_ids]
    )
    print("Finished loading instances.")

    # Example usage
    env = StochasticMaxIndependentSet(instances=loaded_instances)
    key = jax.random.PRNGKey(0)
    state = env.init(key)

    # Play the game until terminal and log actions, observations, and rewards
    rng_key = jax.random.PRNGKey(0)
    while not state.terminated:
        rng_key, subkey = jax.random.split(rng_key)
        action = jax.random.choice(subkey, jnp.arange(0), p=state.legal_action_mask)
        state = env.step(state, action, subkey)

        observation = env.observe(state)
        print(
            f"Action: {action},\n Observation: {observation},\n Reward: {state.rewards}, Terminated: {state.terminated}"
        )
