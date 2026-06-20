# Stochastic Bipartite Matching environment
#
# A bipartite graph matching problem where a policy sequentially assigns each
# right node to a left node. Instances are loaded from a pre-generated dataset;
# edge weights are stochastic, sampled from one of three Bernoulli distributions
# (types 0–2) defined in datasets/sbm/cvar_estimation.py.
#
# At each step the policy selects which left node to match to the current right
# node. An episode terminates once all right nodes have been processed.
#
# Action space : {0, …, num_nodes − 1}  (index of the left node to match)
# - Legal actions : only left nodes that are connected to the current right node and not yet selected
# Observation  : graph-structured dict containing node features, edge features,
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

from src.datasets.sbm.cvar_estimation import sample_weights

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

# Defaults for testing; these will be overridden when loading actual instances.
INIT_NODES = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])
SENDERS = jnp.array([5, 5, 5, 6, 6, 6, 7, 7, 7])
RECEIVERS = jnp.array([0, 1, 4, 0, 1, 2, 2, 3, 4])
EDGE_TYPES = jnp.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
NODE_TYPES = jnp.array([0, 0, 0, 0, 0, 1, 1, 1])  # 0: left, 1: right
NUM_L = jnp.sum(NODE_TYPES == 0)  # Number of left nodes; 5
NUM_R = jnp.sum(NODE_TYPES == 1)  # Number of right nodes; 3


class EnvState(NamedTuple):
    adj: Array
    nodes: Array
    node_types: Array
    edge_types: Array
    selected_edges: Array
    selected_nodes: Array
    processed_nodes: Array
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
    edge_types: jnp.ndarray


def load_sbm_instance(instance_id, dataset_path):
    instance_file = f"instance_{instance_id}.txt"
    instance_path = os.path.join(dataset_path, instance_file)

    with open(instance_path, "r") as f:
        lines = f.readlines()

    # lines[0]: CVaR_100 estimate (expected value of the expected-optimal matching)
    cvar_100 = jnp.array(float(lines[0].strip()), dtype=jnp.float32)
    # lines[1]: CVaR_alpha estimate of the risk-averse matching
    cvar_weight = jnp.array(float(lines[1].strip()), dtype=jnp.float32)

    # lines[2]: expected value of the risk-averse matching
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
    edge_types = edges[:, 2]

    senders = jnp.array(senders)
    receivers = jnp.array(receivers)
    edge_types = jnp.array(edge_types)
    init_nodes = jnp.arange(num_nodes, dtype=jnp.int32)

    return Instance(
        num_nodes=num_nodes,
        node_types=node_types,
        optimal_cvar_value=cvar_weight,  # placeholder
        expected_value=jnp.array(expected_value, dtype=jnp.float32),
        cvar_100=cvar_100,
        init_nodes=init_nodes,
        senders=senders,
        receivers=receivers,
        edge_types=edge_types,
    )


def batch_sbm_instances(instances: List[Instance]) -> Instance:
    return Instance(
        num_nodes=jnp.stack([inst.num_nodes for inst in instances]),
        node_types=jnp.stack([inst.node_types for inst in instances]),
        edge_types=jnp.stack([inst.edge_types for inst in instances]),
        optimal_cvar_value=jnp.stack([inst.optimal_cvar_value for inst in instances]),
        expected_value=jnp.stack([inst.expected_value for inst in instances]),
        cvar_100=jnp.stack([inst.cvar_100 for inst in instances]),
        init_nodes=jnp.stack([inst.init_nodes for inst in instances]),
        senders=jnp.stack([inst.senders for inst in instances]),
        receivers=jnp.stack([inst.receivers for inst in instances]),
    )


def make_adjacency_matrix(
    nodes: Array, senders: Array, receivers: Array, edge_types: Array
) -> Array:
    """Create an adjacency matrix from the nodes and edges, as an undirected graph."""
    num_nodes = len(nodes)
    adjacency_matrix = jnp.full((num_nodes, num_nodes), -1, dtype=jnp.int32)
    adjacency_matrix = adjacency_matrix.at[senders, receivers].set(edge_types)
    adjacency_matrix = adjacency_matrix.at[receivers, senders].set(edge_types)
    return adjacency_matrix


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
        adj=make_adjacency_matrix(INIT_NODES, SENDERS, RECEIVERS, EDGE_TYPES),
        nodes=INIT_NODES,
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        selected_edges=jnp.zeros((0,), dtype=jnp.bool_),
        selected_nodes=jnp.zeros((0,), dtype=jnp.bool_),
        processed_nodes=jnp.zeros((0,), dtype=jnp.bool_),
        senders=SENDERS,
        receivers=RECEIVERS,
        optimal_cvar_value=jnp.array(0.0),
        expected_value=jnp.array(0.0),
        cvar_100=jnp.array(0.0),
        key=jax.random.PRNGKey(0),
    )

    @property
    def env_id(self) -> core.EnvId:
        return "bipartite_matching"  # type: ignore

    @property
    def x(self) -> EnvState:
        return self._x

    def replace(self, **kwargs: Any) -> "State":  # type: ignore
        # pgx adds this method dynamically at runtime; we re-declare it here
        # so static type-checkers (Pylance/pyright) can resolve it.
        return dataclasses.replace(self, **kwargs)  # type: ignore[misc]


def new_state(state: EnvState, node: jnp.ndarray, step_count: jnp.ndarray) -> EnvState:
    """Select a node. This will render that node selected.
    The new unavailable nodes are all its neighbors.
    """
    num_l = jnp.sum(state.node_types == 0)  # Number of left nodes

    def find_edge_index(x: int, y: int, senders: jnp.ndarray, receivers: jnp.ndarray):
        matching = ((senders == x) & (receivers == y)) | ((senders == y) & (receivers == x))
        indices = jnp.where(matching, size=1, fill_value=-1)[0]
        return indices  # returns -1 if no match found

    new_selected_nodes = state.selected_nodes.at[node].set(True)
    processed_nodes = state.processed_nodes.at[step_count + num_l].set(True)

    sender = state.nodes[step_count + num_l]
    receiver = state.nodes[node]
    edge_index = find_edge_index(sender, receiver, state.senders, state.receivers)
    new_selected_edges = state.selected_edges.at[edge_index].set(True)

    return state._replace(
        selected_nodes=new_selected_nodes,
        processed_nodes=processed_nodes,
        selected_edges=new_selected_edges,
    )


def reward(
    state: EnvState, step_count: jnp.ndarray, node: jnp.ndarray, rng_key: jnp.ndarray
) -> jnp.ndarray:
    """Calculate the reward for selecting a node."""
    num_l = jnp.sum(state.node_types == 0)  # Number of left nodes

    sender = state.nodes[step_count + num_l]
    receiver = state.nodes[node]
    edge_type = state.adj[sender, receiver]

    reward = sample_weights(edge_type, rng_key)
    return jax.lax.cond(
        state.selected_nodes[node],
        lambda: -1000.0,
        lambda: reward.astype(jnp.float32),
    )


def is_terminal(step_count: jnp.ndarray, num_r: jnp.ndarray) -> jnp.ndarray:
    """Check if the state is terminal."""
    # The state is terminal if there are no available nodes left.
    return step_count >= num_r


class StochasticBipartiteMatching(core.Env):
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
        split: jnp.ndarray = jnp.array(0),  # 0 = train, 1 = test
    ) -> State:  # type: ignore
        num_instances = self._instances.init_nodes.shape[0] // 2
        idx = (iteration % (num_instances // num_envs)) + offset * (num_instances // num_envs)
        idx = jax.lax.cond(
            split == 1,  # Test split
            lambda: idx + num_instances,  # Use the second half as unseen test
            lambda: idx,
        )

        init_nodes = self._instances.init_nodes[idx]
        senders = self._instances.senders[idx]
        receivers = self._instances.receivers[idx]
        node_types = self._instances.node_types[idx]
        edge_types = self._instances.edge_types[idx]

        num_l = jnp.sum(node_types == 0)  # Number of left nodes
        num_edges = senders.shape[0]  # Number of edges

        # Legal is all nodes connected to the first left node
        adj = make_adjacency_matrix(init_nodes, senders, receivers, edge_types)
        first_left_node_idx = num_l
        legal_action_mask = adj[first_left_node_idx, :] >= 0

        return State(
            iteration=iteration,
            offset=offset,
            num_envs=num_envs,
            legal_action_mask=legal_action_mask,
            split=split,
            _x=EnvState(
                adj=adj,
                nodes=init_nodes,
                node_types=node_types,
                edge_types=edge_types,
                selected_edges=jnp.zeros((num_edges,), dtype=jnp.bool_),
                selected_nodes=jnp.zeros((self._num_nodes,), dtype=jnp.bool_),
                processed_nodes=jnp.zeros((self._num_nodes,), dtype=jnp.bool_),
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
        num_l = jnp.sum(self._instances.node_types[0] == 0)  # Number of left nodes
        num_r = jnp.sum(self._instances.node_types[0] == 1)

        selecting_for = jnp.zeros((self._num_nodes,), dtype=jnp.bool_)
        selecting_for = selecting_for.at[state._step_count + num_l].set(True)
        node_features = jnp.stack(
            [selecting_for, state.x.selected_nodes, state.x.processed_nodes], axis=1
        ).astype(jnp.float32)  # (NUM_NODES, 2)

        # There are 2 types of nodes, so we can use one-hot encoding
        node_types = state.x.node_types  # (NUM_NODES,)
        node_types_one_hot = jax.nn.one_hot(
            node_types, num_classes=2, dtype=jnp.float32
        )  # (NUM_NODES, 2)

        node_features = jnp.concatenate(
            [node_features, node_types_one_hot], axis=1
        )  # (NUM_NODES, 3)

        # Edge features are just the types (there are 3). We add 1 because
        # type 0 edges are connected to the register nodes. Alongside the selected edges,
        # and if the edge is incident to the R node being selected for.
        sender_incident = state.x.senders == state._step_count + num_l
        receiver_incident = state.x.receivers == state._step_count + num_l
        incident = sender_incident | receiver_incident
        edge_features = jax.nn.one_hot(
            state.x.edge_types + 1, num_classes=4, dtype=jnp.float32
        )  # (NUM_EDGES, 4)
        edge_features = jnp.concatenate(
            [edge_features, state.x.selected_edges[:, None], incident[:, None]], axis=1
        )  # (NUM_EDGES, 4)

        senders = state.x.senders  # (NUM_EDGES,)
        receivers = state.x.receivers  # (NUM_EDGES,)
        bias = jnp.array(1.0)  # (scalar)

        num_chosen = (state._step_count + 1) / num_r
        aux = jnp.stack([num_chosen, bias], axis=0).astype(jnp.float32)
        return {
            "selecting_for": state._step_count + num_l,
            "node_features": node_features,
            "edge_features": edge_features,
            "edge_types": state.x.edge_types,
            "senders": senders,
            "receivers": receivers,
            "aux": aux,
        }

    def find_edge_index(
        self,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        receiver: jnp.ndarray,
        sender: jnp.ndarray,
    ) -> jnp.ndarray:
        matching = ((senders == receiver) & (receivers == sender)) | (
            (senders == sender) & (receivers == receiver)
        )
        indices = jnp.where(matching, size=1, fill_value=-1)[0]
        return indices  # returns -1 if no match found

    def find_edge_indices(
        self,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        receiver_batch: jnp.ndarray,
        sender_batch: jnp.ndarray,
    ) -> jnp.ndarray:
        def single_edge_index(receiver, sender):
            matching = ((senders == receiver) & (receivers == sender)) | (
                (senders == sender) & (receivers == receiver)
            )
            indices = jnp.where(matching, size=1, fill_value=-1)[0]
            return indices

        return jax.vmap(single_edge_index)(receiver_batch, sender_batch)

    def _step(self, state: State, action: jnp.ndarray, key: jnp.ndarray) -> State:
        num_l = jnp.sum(self._instances.node_types[0] == 0)  # Number of left nodes
        num_r = jnp.sum(self._instances.node_types[0] == 1)  # Number of right nodes

        next_state = new_state(state.x, action, state._step_count - 1)
        r = reward(state.x, state._step_count - 1, action, key)
        is_term = is_terminal(state._step_count, num_r)

        # Terminal after num node steps, regardless of the state
        is_term = (state._step_count >= self._num_nodes) | is_term

        L_NODES = state._x.node_types == 0  # Left nodes
        is_available = (
            L_NODES
            & ~next_state.selected_nodes
            & (state._x.adj[state._x.nodes[num_l + state._step_count], :] >= 0)
        )

        legal_action_mask = jax.lax.cond(
            self.using_legal_actions,
            lambda: is_available,
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
        return "bipartite_matching"  # type: ignore

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 1


if __name__ == "__main__":
    # Smoke-test using the built-in default graph (no dataset required).
    default_instance = batch_sbm_instances(
        [
            Instance(
                num_nodes=jnp.array(len(INIT_NODES)),
                node_types=NODE_TYPES,
                optimal_cvar_value=jnp.array(0.0),
                expected_value=jnp.array(0.0),
                cvar_100=jnp.array(0.0),
                init_nodes=INIT_NODES,
                senders=SENDERS,
                receivers=RECEIVERS,
                edge_types=EDGE_TYPES,
            )
        ]
    )

    env = StochasticBipartiteMatching(instances=default_instance)
    rng = jax.random.PRNGKey(0)

    rng, init_key = jax.random.split(rng)
    state = env.init_v2(init_key, iteration=jnp.array(0), offset=jnp.array(0), num_envs=1, split=0)  # type: ignore
    print(f"Initial legal actions: {state.legal_action_mask}")

    step = 0
    while not state.terminated:
        rng, step_key, choice_key = jax.random.split(rng, 3)
        action = jax.random.choice(
            choice_key, jnp.arange(len(INIT_NODES)), p=state.legal_action_mask.astype(jnp.float32)
        )
        state = env.step(state, action, step_key)
        print(
            f"Step {step}: action={int(action)}, reward={float(state.rewards[0]):.2f}, terminated={bool(state.terminated)}"
        )
        step += 1

    print(f"Episode complete after {step} steps.")
