# Risk AlphaZero (graph/node) networks
#
# Neural network modules used by the graph-node Risk AlphaZero training loop.
#
# Contents
# --------
# FeatureExtractor       — SAGEConv GNN with register nodes
# PredictionNetwork      — Policy logits + value quantile distribution
# RewardHead             — Single-step reward quantile distribution
# RewardHistoryHead      — Accumulated-return quantile distribution
# make_network_apply_fns — Returns Haiku-transformed apply functions + init

import haiku as hk
import jax
import jax.numpy as jnp
from haiku_geometric.nn import SAGEConv


class FeatureExtractor(hk.Module):
    """SAGEConv GNN feature extractor with register nodes.

    Applies four ``SAGEConv`` layers with layer normalization and residual
    connections.  A learnable register node acts as a global pooling token.
    Returns (node_reps, pool_aux_concat) where ``node_reps`` has shape
    ``(n_nodes, hidden * 3)`` and ``pool_aux_concat`` has shape ``(hidden,)``.
    """

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.conv1 = SAGEConv(out_channels=hidden_channels)
        self.conv2 = SAGEConv(out_channels=hidden_channels)
        self.conv3 = SAGEConv(out_channels=hidden_channels)
        self.conv4 = SAGEConv(out_channels=hidden_channels)

        self.node_features_proj = hk.Linear(hidden_channels)

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

    def __call__(self, nodes, senders, receivers, aux):
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

        # Make the graph undirected by adding the reverse edges
        senders = jnp.concatenate([senders_, receivers_], axis=0)
        receivers = jnp.concatenate([receivers_, senders_], axis=0)

        # Apply convolutions
        x = self.conv1(nodes, senders, receivers)  # (n_nodes + 1, hidden_channels)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            jax.nn.relu(x)
        )  # (n_nodes + 1, hidden_channels)
        x = x + hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            jax.nn.relu(self.conv2(x, senders, receivers))
        )  # (n_nodes + 1, hidden_channels)
        x = x + hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            jax.nn.relu(self.conv3(x, senders, receivers))
        )  # (n_nodes + 1, hidden_channels)
        x = x + hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            jax.nn.relu(self.conv4(x, senders, receivers))
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

        # Remove the register nodes from the representation
        node_reps = node_pool_aux_concat[: -self.num_registers]  # (n_nodes, out_channels * 3)
        return node_reps, pool_aux_concat


class PredictionNetwork(hk.Module):
    """Joint policy-logits and value-distribution head.

    Wraps ``FeatureExtractor`` and produces:
    - ``logits``          — per-node action logits, shape ``(n_nodes,)``
    - ``v_dist``          — value quantile distribution, shape ``(num_quantiles,)``
    - ``node_reps``       — node embeddings, shape ``(n_nodes, hidden * 3)``
    - ``pool_aux_concat`` — global embedding, shape ``(hidden,)``
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

    def __call__(self, nodes, senders, receivers, aux):
        node_reps, pool_aux_concat = self.feature_extractor(
            nodes, senders, receivers, aux
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

    Takes the embedding of the selected node and produces a quantile
    distribution over the immediate reward.
    """

    def __init__(self, hidden_channels: int, num_quantiles: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_quantiles = num_quantiles

    def __call__(
        self,
        selected_node: jnp.ndarray,
    ):
        sa_embed = hk.Linear(self.hidden_channels)(selected_node)
        sa_embed_ = jax.nn.tanh(sa_embed)  # (hidden_channels,)
        sa_embed = hk.Linear(self.hidden_channels)(sa_embed_)
        sa_embed = jax.nn.tanh(sa_embed)  # (hidden_channels,)
        sa_embed = sa_embed + sa_embed_  # (hidden_channels,)
        r_dist = hk.Linear(
            self.num_quantiles,
            w_init=hk.initializers.UniformScaling(0.01),
            b_init=hk.initializers.UniformScaling(0.25),
        )(sa_embed)  # (num_quantiles,)
        return r_dist


class RewardHistoryHead(hk.Module):
    """Accumulated-return quantile distribution head.

    Takes the global observation embedding (``pool_aux_concat``) and outputs
    a quantile distribution over the discounted accumulated return.
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
    """Build Haiku-transformed apply functions and a joint init function.

    Returns:
        prediction_apply:     Apply function for ``PredictionNetwork``.
        reward_apply:         Apply function for ``RewardHead``.
        reward_history_apply: Apply function for ``RewardHistoryHead``.
        init_model:           Joint init function used to create parameters.
    """

    def prediction_apply_fn(nodes, senders, receivers, aux):
        prediction_network = PredictionNetwork(
            hidden_channels=args.num_hidden,
            num_quantiles=args.num_quantiles,
        )
        return jax.vmap(prediction_network)(nodes, senders, receivers, aux)

    def reward_apply_fn(selected_node):
        reward_head = RewardHead(
            hidden_channels=args.num_hidden,
            num_quantiles=args.num_quantiles,
        )
        return jax.vmap(reward_head)(selected_node)

    def reward_history_apply_fn(obs_embedding):
        reward_history_head = RewardHistoryHead(
            hidden_channels=args.num_hidden,
            num_quantiles=args.num_quantiles,
        )
        return jax.vmap(reward_history_head)(obs_embedding)

    def init_model_fn(nodes, senders, receivers, aux, action):
        prediction_network = PredictionNetwork(
            hidden_channels=args.num_hidden,
            num_quantiles=args.num_quantiles,
        )
        _logits, q_dist, node_reps, _pool_aux_concat = jax.vmap(prediction_network)(
            nodes, senders, receivers, aux
        )

        reward_head = RewardHead(
            hidden_channels=args.num_hidden,
            num_quantiles=args.num_quantiles,
        )
        selected_node = nodes[action]
        r_dist = jax.vmap(reward_head)(selected_node)

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

    args = SimpleNamespace(num_hidden=16, num_quantiles=10, num_actions=4)
    prediction_apply, reward_apply, reward_history_apply, init_model = make_network_apply_fns(args)
    nodes = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
    senders = jnp.array([[0, 1]])
    receivers = jnp.array([[1, 0]])
    aux = jnp.array([[0.1, 0.2]])
    params = init_model.init(jax.random.PRNGKey(42), nodes, senders, receivers, aux, jnp.array([0]))
    logits, v_dist, node_reps, pool_aux = prediction_apply.apply(
        params, nodes, senders, receivers, aux
    )
    print("logits:", logits.shape)  # (batch=1, n_nodes)
    print("v_dist:", v_dist.shape)  # (batch=1, num_quantiles)
    print("node_reps:", node_reps.shape)  # (batch=1, n_nodes, hidden*3)
    print("pool_aux:", pool_aux.shape)  # (batch=1, num_hidden)
