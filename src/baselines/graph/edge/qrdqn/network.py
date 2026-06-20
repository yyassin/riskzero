# Graph QR-DQN neural network (edge-featured variant)
#
# Implements a Quantile-Regression DQN network over graph-structured
# observations using message-passing graph convolutions (haiku-geometric
# GraphConv). Unlike the node-only variant, edge features are explicitly
# projected and included in each convolution pass.
#
# Architecture
# ------------
# 1. Project node features and edge features to ``hidden_channels``.
# 2. Append ``num_registers`` learnable global register nodes, connected
#    to every real node via zero-type edges (a form of virtual global pooling).
# 3. Make the graph undirected by duplicating edges in both directions.
# 4. Four GraphConv layers with residual connections and LayerNorm.
# 5. Aggregate: sum-pool all node representations → global vector.
# 6. Per-node output: concatenate [global, node, aux] → Linear → num_quantiles.
# 7. Remove register nodes; reshape to (num_quantiles, num_nodes).
# 8. CVaR (or other distortion) aggregated across quantiles per action.

from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
from haiku_geometric.nn import GraphConv

import lib.util as util
from lib.util import (
    cvar_distortion_q,
    pow_distortion_q,
    wang_distortion_q,
)


class QRNetworkOutputs(NamedTuple):
    q_values: jnp.ndarray
    q_dist: jnp.ndarray


class GraphQNetwork(hk.Module):
    def __init__(
        self,
        hidden_channels,
        num_quantiles: int,
        alpha_cvar: float,
        distortion: str = "cvar",
        num_registers: int = 1,
    ):
        super().__init__()
        if distortion == "cvar":
            self.distortion_fn = cvar_distortion_q
        elif distortion == "pow":
            self.distortion_fn = pow_distortion_q
        elif distortion == "wang":
            self.distortion_fn = wang_distortion_q
        else:
            raise ValueError(f"Unknown distortion type: {distortion}")

        self.hidden_channels = hidden_channels
        self.num_quantiles = num_quantiles
        self.alpha_cvar = alpha_cvar

        self.tau_hats = util.make_tau_hats(num_quantiles)

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
        self.linear_value = hk.Linear(self.num_quantiles)

        self.num_registers = num_registers
        self.register_node_init = hk.get_parameter(
            "register_node_init",
            shape=(num_registers, hidden_channels),
            init=hk.initializers.RandomNormal(),
        )

    def __call__(self, nodes, edges, senders, receivers, aux):
        # Add a vector of ones to the node features
        nodes = self.linear_node_features(nodes)  # (nodes, hidden_channels)
        nodes = jax.nn.relu(nodes)
        nodes = jnp.concatenate([nodes, jnp.ones((nodes.shape[0], self.hidden_channels))], axis=-1)
        nodes = self.node_features_proj(nodes)  # (nodes, hidden_channels)
        nodes = jax.nn.relu(nodes)

        register_nodes = self.register_node_init  # (K=1, hidden)
        num_nodes = nodes.shape[0]
        nodes = jnp.concatenate([nodes, register_nodes], axis=0)  # (n_nodes + K, hidden)

        # There are n_nodes + K=1 (register) nodes in the graph.
        # We want to connect register nodes [n_nodes:] to all the prior nodes.
        receivers_ = jnp.arange(num_nodes)
        senders_ = jnp.full(num_nodes, num_nodes, dtype=jnp.int32)
        senders_ = jnp.concatenate([senders, senders_], axis=0)
        receivers_ = jnp.concatenate([receivers, receivers_], axis=0)

        # We've added (K=1) * n_nodes edges to the graph, these are type 0 edges.
        edge_dim = edges.shape[-1]
        edges_ = jnp.zeros((num_nodes, edge_dim), dtype=edges.dtype)
        edges_ = edges_.at[:, 0].set(1.0)  # One-hot index 0: reserved for register edges.
        # Real edges use types +1 (see _observe in stochastic_bipartite_matching.py),
        # so index 0 uniquely identifies register-to-node connections.
        edges = jnp.concatenate([edges, edges_], axis=0)
        edges = self.edge_features_proj(edges)  # (edges, hidden_channels)
        edges = jax.nn.relu(edges)

        # Make the graph undirected by adding the reverse edges
        senders = jnp.concatenate([senders_, receivers_], axis=0)
        receivers = jnp.concatenate([receivers_, senders_], axis=0)
        # Repeat edges to match undirected edges
        edges = jnp.concatenate([edges, edges], axis=0)

        # GraphConv layers with residual connections and LayerNorm
        x = self.conv1(nodes, senders, receivers, edges)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(jax.nn.relu(x))
        edges = edges + hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            jax.nn.relu(hk.Linear(self.hidden_channels)(edges))
        )
        x = x + hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            jax.nn.relu(self.conv2(x, senders, receivers, edges))
        )
        edges = edges + hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            jax.nn.relu(hk.Linear(self.hidden_channels)(edges))
        )
        x = x + hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            jax.nn.relu(self.conv3(x, senders, receivers, edges))
        )
        edges = edges + hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            jax.nn.relu(hk.Linear(self.hidden_channels)(edges))
        )
        x = x + hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            jax.nn.relu(self.conv4(x, senders, receivers, edges))
        )  # (n_nodes + K, hidden_channels)

        pool_rep = jnp.sum(x, axis=0)  # (hidden_channels,)
        pool_rep = self.linear_pool(pool_rep)  # (out_channels,)

        node_rep = self.linear_node(x)  # (n_nodes + K, out_channels)
        aux_rep = self.linear_aux(aux)  # (out_channels)
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
        values = self.linear_value(node_pool_aux_concat)  # (n_nodes + K, num_quantiles)

        # Remove the register nodes from the values
        values = values[: -self.num_registers]  # (n_nodes, num_quantiles)

        # Switch axes to get (b, num_quantiles, n_nodes)
        q_dist = jnp.swapaxes(values, 0, 1)  # (num_quantiles, n_nodes)
        q_dist = jnp.reshape(
            q_dist, (-1, self.num_quantiles, q_dist.shape[-1])
        )  # (b, num_quantiles, num_actions)

        q_values = jax.vmap(self.distortion_fn, in_axes=(0, None, None))(
            q_dist,  # (b, num_quantiles, num_actions)
            self.tau_hats,  # (num_quantiles,)
            self.alpha_cvar,
        )

        # Values are batched externally by vmap
        q_values = jax.lax.stop_gradient(q_values).squeeze(0)  # (num_actions)
        q_dist = q_dist.squeeze(0)  # (num_quantiles, num_actions)

        return QRNetworkOutputs(q_dist=q_dist, q_values=q_values)


def create_graph_qr_network(args):
    def model_fn(
        nodes: jax.Array,
        edges: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        aux: jax.Array,
    ) -> jax.Array:
        net = GraphQNetwork(
            hidden_channels=args.hidden_size,
            num_quantiles=args.num_quantiles,
            alpha_cvar=args.alpha_cvar,
            num_registers=1,
            distortion=args.distortion,
        )
        return jax.vmap(net)(nodes, edges, senders, receivers, aux)

    model = hk.without_apply_rng(hk.transform(model_fn))
    return model
