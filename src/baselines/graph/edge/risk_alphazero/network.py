# Risk AlphaZero (graph/edge) networks
#
# Neural network modules used by the graph-edge Risk AlphaZero training loop.
#
# Contents
# --------
# FeatureExtractor     — GNN feature extractor with register nodes
# PredictionNetwork    — Policy logits + value quantile distribution head
# RewardHead           — Single-step reward quantile distribution head
# RewardHistoryHead    — Accumulated-return distribution head
# make_network_apply_fns — Returns Haiku-transformed apply functions + init

import haiku as hk
import jax
import jax.numpy as jnp
from haiku_geometric.nn import GraphConv


class FeatureExtractor(hk.Module):
    """GNN feature extractor with register nodes.

    Applies four ``GraphConv`` layers with layer normalization and residual
    connections.  A fixed number (1) of learnable register nodes are appended to
    the graph to act as a global pooling token.  The final output is a pair
    of (node_reps, pool_aux_concat) suitable for the downstream heads.
    """

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.conv1 = GraphConv(out_channels=hidden_channels)
        self.conv2 = GraphConv(out_channels=hidden_channels)
        self.conv3 = GraphConv(out_channels=hidden_channels)
        self.conv4 = GraphConv(out_channels=hidden_channels)

        self.node_features_proj = hk.Linear(hidden_channels)
        self.edge_features_proj = hk.Linear(hidden_channels)

        self.linear_node_features = hk.Linear(hidden_channels)
        self.linear_pool = hk.Linear(hidden_channels)
        self.linear_node = hk.Linear(hidden_channels)
        self.linear_aux = hk.Linear(hidden_channels)

        self.num_registers = 1
        self.register_node_init = hk.get_parameter(
            "register_node_init",
            shape=(self.num_registers, hidden_channels),
            init=hk.initializers.RandomNormal(),
        )

    def __call__(self, nodes, edges, senders, receivers, aux):
        nodes = self.linear_node_features(nodes)  # (nodes, hidden_channels)
        nodes = jax.nn.relu(nodes)  # (nodes, hidden_channels)
        nodes = jnp.concatenate(
            [nodes, jnp.ones((nodes.shape[0], self.hidden_channels))], axis=-1
        )  # (nodes, hidden_channels + 1)
        nodes = self.node_features_proj(nodes)  # (nodes, hidden_channels)
        nodes = jax.nn.relu(nodes)  # (nodes, hidden_channels)

        register_nodes = self.register_node_init  # (1, hidden)
        num_nodes = nodes.shape[0]
        nodes = jnp.concatenate([nodes, register_nodes], axis=0)  # (n_nodes + 1, hidden)

        # There are n_nodes + 1 nodes in the graph. We want to connect (register) nodes [n_nodes:] to all the prior nodes.
        receivers_ = jnp.arange(num_nodes)
        senders_ = jnp.full(num_nodes, num_nodes, dtype=jnp.int32)  # [num_node, num_node, ...]
        senders_ = jnp.concatenate([senders, senders_], axis=0)
        receivers_ = jnp.concatenate([receivers, receivers_], axis=0)

        # We've added n_nodes edges to the graph, these are type 0 edges.
        edge_dim = edges.shape[-1]
        edges_ = jnp.zeros((num_nodes, edge_dim), dtype=edges.dtype)
        edges_ = edges_.at[:, 0].set(1.0)  # Set the first edge type to 1
        edges = jnp.concatenate([edges, edges_], axis=0)
        edges = self.edge_features_proj(edges)  # (edges, hidden_channels)
        edges = jax.nn.relu(edges)  # (edges, hidden_channels)

        # Make the graph undirected by adding the reverse edges
        senders = jnp.concatenate([senders_, receivers_], axis=0)
        receivers = jnp.concatenate([receivers_, senders_], axis=0)
        # Repeat edges to match undirected edges
        edges = jnp.concatenate([edges, edges], axis=0)

        # Apply convolutions
        x = self.conv1(nodes, senders, receivers, edges)  # (n_nodes + 1, hidden_channels)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            jax.nn.relu(x)
        )  # (n_nodes + 1, hidden_channels)
        edges = edges + hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            jax.nn.relu(hk.Linear(self.hidden_channels)(edges))
        )  # (edges, hidden_channels)
        x = x + hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            jax.nn.relu(self.conv2(x, senders, receivers, edges))
        )  # (n_nodes + 1, hidden_channels)
        edges = edges + hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            jax.nn.relu(hk.Linear(self.hidden_channels)(edges))
        )  # (edges, hidden_channels)
        x = x + hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            jax.nn.relu(self.conv3(x, senders, receivers, edges))
        )  # (n_nodes + 1, hidden_channels)
        edges = edges + hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            jax.nn.relu(hk.Linear(self.hidden_channels)(edges))
        )  # (edges, hidden_channels)

        x = x + hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            jax.nn.relu(self.conv4(x, senders, receivers, edges))
        )  # (n_nodes + 1, hidden_channels)

        # Pool and concat
        pool_rep = jnp.sum(x, axis=0)  # (hidden_channels,)
        pool_rep = self.linear_pool(pool_rep)  # (out_channels,)
        aux_rep = self.linear_aux(aux)  # (out_channels)

        pool_aux_concat_ = jnp.concat(
            [pool_rep[jnp.newaxis, :], aux_rep[jnp.newaxis, :]], axis=-1
        ).reshape((-1))  # (out_channels * 2,)
        pool_aux_concat = jax.nn.relu(hk.Linear(self.hidden_channels)(pool_aux_concat_))

        node_rep = self.linear_node(x)  # (n_nodes + K, out_channels)
        # Repeat aux_rep to match the number of nodes
        aux_rep = jnp.repeat(
            aux_rep[jnp.newaxis, :], x.shape[0], axis=0
        )  # (n_nodes + K, out_channels)

        pool_rep = jnp.repeat(pool_rep[jnp.newaxis, :], x.shape[0], axis=0)
        node_pool_concat = jnp.concat(
            [pool_rep, node_rep], axis=-1
        )  # (n_nodes + K, out_channels * 2)
        node_pool_concat = jax.nn.relu(node_pool_concat)  # (n_nodes + K, out_channels * 2)
        node_pool_aux_concat = jnp.concat(
            [node_pool_concat, aux_rep], axis=-1
        )  # (n_nodes + K, out_channels * 3)

        node_embeddings = hk.Linear(
            self.hidden_channels,
        )(node_pool_aux_concat)  # (n_nodes + K, hidden_channels * 2)
        node_embeddings = jax.nn.relu(node_embeddings)  # (n_nodes + K, hidden_channels * 2)

        # Remove the register nodes from the representation
        node_reps = node_pool_aux_concat[: -self.num_registers]  # (n_nodes, out_channels * 3)
        return node_reps, pool_aux_concat


class PredictionNetwork(hk.Module):
    """Policy + value head for graph-edge observations.

    Wraps ``FeatureExtractor`` and produces (logits, v_dist, node_reps,
    pool_aux_concat) where logits are per-node policy logits and v_dist is
    a quantile value distribution over the global graph embedding.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_quantiles: int,
        name: str = "prediction_network",
    ):
        super().__init__(name=name)
        self.num_quantiles = num_quantiles
        self.hidden_channels = hidden_channels
        self.feature_extractor = FeatureExtractor(hidden_channels=self.hidden_channels)

    def __call__(self, nodes, edges, senders, receivers, aux):
        node_reps, pool_aux_concat = self.feature_extractor(
            nodes, edges, senders, receivers, aux
        )  # (n_nodes, hidden_channels * 3)

        # Q-dist
        v_dist = jax.nn.relu(hk.Linear(self.hidden_channels)(pool_aux_concat))
        v_dist = hk.Linear(
            self.num_quantiles,
            w_init=hk.initializers.UniformScaling(0.01),
            b_init=hk.initializers.UniformScaling(0.25),
        )(v_dist)  # (num_quantiles,)

        # Policy priors
        logits = hk.Linear(self.hidden_channels)(node_reps)  # (n_nodes, hidden_channels)
        logits = jax.nn.relu(logits)  # (n_nodes, hidden_channels)
        logits = hk.Linear(1)(logits).reshape((-1))  # (n_nodes,)

        return logits, v_dist, node_reps, pool_aux_concat


class RewardHead(hk.Module):
    """Single-step reward quantile distribution head.

    Takes the edge embedding of the selected action and outputs a quantile
    distribution over the immediate reward.
    """

    def __init__(self, hidden_channels: int, num_quantiles: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_quantiles = num_quantiles

    def __call__(
        self,
        selected_edge: jnp.ndarray,
    ):
        sa_embed = hk.Linear(self.hidden_channels)(selected_edge)
        sa_embed = jax.nn.tanh(sa_embed)  # (hidden_channels,)
        r_dist = hk.Linear(
            self.num_quantiles,
            w_init=hk.initializers.UniformScaling(0.01),
            b_init=hk.initializers.UniformScaling(0.25),
        )(sa_embed)  # (num_quantiles,)
        return r_dist


class RewardHistoryHead(hk.Module):
    """Accumulated-return quantile distribution head.

    Takes the global graph embedding and outputs a quantile distribution over
    the discounted accumulated return seen so far in the episode.
    """

    def __init__(self, hidden_channels: int, num_quantiles: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_quantiles = num_quantiles

    def __call__(
        self,
        obs_embedding: jnp.ndarray,
    ):
        obs_embedding_ = hk.Linear(self.hidden_channels)(obs_embedding)
        obs_embedding_ = jax.nn.tanh(obs_embedding)  # (hidden_channels,)
        obs_embedding_ = obs_embedding + obs_embedding_  # (hidden_channels,)
        r_dist = hk.Linear(
            self.num_quantiles,
            w_init=hk.initializers.UniformScaling(0.01),
            b_init=hk.initializers.UniformScaling(0.25),
        )(obs_embedding_)  # (num_quantiles,)
        return r_dist  # (num_quantiles,)


def make_network_apply_fns(args):
    """Build and return Haiku-transformed apply functions and a joint init.

    Returns:
        prediction_apply:      Apply fn for ``PredictionNetwork``.
        reward_apply:          Apply fn for ``RewardHead``.
        reward_history_apply:  Apply fn for ``RewardHistoryHead``.
        init_model:            Joint init fn that initialises all three heads.
    """

    def prediction_apply_fn(nodes, edges, senders, receivers, aux):
        prediction_network = PredictionNetwork(
            hidden_channels=args.num_hidden,
            num_quantiles=args.num_quantiles,
        )
        return jax.vmap(prediction_network)(nodes, edges, senders, receivers, aux)

    def reward_apply_fn(selected_edge):
        reward_head = RewardHead(
            hidden_channels=args.num_hidden,
            num_quantiles=args.num_quantiles,
        )
        return jax.vmap(reward_head)(selected_edge)

    def reward_history_apply_fn(obs_embedding):
        reward_history_head = RewardHistoryHead(
            hidden_channels=args.num_hidden,
            num_quantiles=args.num_quantiles,
        )
        return jax.vmap(reward_history_head)(obs_embedding)

    def init_model_fn(nodes, edges, senders, receivers, aux, selected_edge):
        prediction_network = PredictionNetwork(
            hidden_channels=args.num_hidden,
            num_quantiles=args.num_quantiles,
        )
        _logits, q_dist, _node_reps, _pool_aux_concat = jax.vmap(prediction_network)(
            nodes, edges, senders, receivers, aux
        )

        reward_head = RewardHead(
            hidden_channels=args.num_hidden,
            num_quantiles=args.num_quantiles,
        )
        r_dist = jax.vmap(reward_head)(selected_edge)

        reward_history_head = RewardHistoryHead(
            hidden_channels=args.num_hidden,
            num_quantiles=args.num_quantiles,
        )
        r_history_dist = jax.vmap(reward_history_head)(_pool_aux_concat)
        return _logits, q_dist, r_dist, r_history_dist

    prediction_apply = hk.without_apply_rng(hk.transform(prediction_apply_fn))
    reward_apply = hk.without_apply_rng(hk.transform(reward_apply_fn))
    reward_history_apply = hk.without_apply_rng(hk.transform(reward_history_apply_fn))
    init_model = hk.without_apply_rng(hk.transform(init_model_fn))
    return prediction_apply, reward_apply, reward_history_apply, init_model


if __name__ == "__main__":
    from types import SimpleNamespace

    args = SimpleNamespace(num_hidden=16, num_quantiles=8)
    rng = jax.random.PRNGKey(0)

    # Minimal graph: batch=2, 4 nodes, 3 edges, node_feat=5, edge_feat=4, aux_dim=2
    b, n_nodes, n_edges = 2, 4, 3
    node_feat, edge_feat, aux_dim = 5, 4, 2

    nodes = jnp.ones((b, n_nodes, node_feat))
    edges = jnp.ones((b, n_edges, edge_feat))
    senders = jnp.array([[0, 1, 2], [0, 1, 2]])
    receivers = jnp.array([[1, 2, 3], [1, 2, 3]])
    aux = jnp.ones((b, aux_dim))
    selected_edge = jnp.ones((b, edge_feat))

    prediction_apply, reward_apply, reward_history_apply, init_model = make_network_apply_fns(args)

    params = init_model.init(rng, nodes, edges, senders, receivers, aux, selected_edge)

    logits, v_dist, node_reps, pool_aux = prediction_apply.apply(
        params, nodes, edges, senders, receivers, aux
    )
    print(f"logits:     {logits.shape}")  # (b, n_nodes)
    print(f"v_dist:     {v_dist.shape}")  # (b, num_quantiles)
    print(f"node_reps:  {node_reps.shape}")  # (b, n_nodes, hidden*3)
    print(f"pool_aux:   {pool_aux.shape}")  # (b, hidden)

    r_dist = reward_apply.apply(params, selected_edge)
    print(f"r_dist:     {r_dist.shape}")  # (b, num_quantiles)

    r_hist = reward_history_apply.apply(params, pool_aux)
    print(f"r_hist:     {r_hist.shape}")  # (b, num_quantiles)
