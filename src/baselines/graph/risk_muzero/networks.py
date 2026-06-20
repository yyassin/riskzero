# Risk MuZero (graph/node) networks
#
# Neural network modules used by the graph-node Risk MuZero training loop.
#
# Contents
# --------
# FeatureExtractor       — SAGEConv GNN feature extractor with register nodes
# ObsEncoder             — Encodes a graph observation into node / aux embeddings
# ProjectionHead         — Projects embeddings for self-consistency loss
# PolicyHead             — Policy logits head
# QValueHead             — Quantile value distribution head
# RewardHistoryHead      — Accumulated-return distribution head
# DynamicsHead           — Transition model (next state + reward dist)
# make_network_apply_fns — Returns Haiku-transformed apply functions + init

import haiku as hk
import jax
import jax.numpy as jnp
from haiku_geometric.nn import SAGEConv


class FeatureExtractor(hk.Module):
    """SAGEConv GNN feature extractor with register nodes.

    Applies four ``SAGEConv`` layers with layer normalization and residual
    connections.  A learnable register node acts as a global pooling token.
    Returns ``(node_embeddings, aux_embedding, pool_aux_concat)``.
    """

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.conv1 = SAGEConv(out_channels=hidden_channels)
        self.conv2 = SAGEConv(out_channels=hidden_channels)
        self.conv3 = SAGEConv(out_channels=hidden_channels)
        self.conv4 = SAGEConv(out_channels=hidden_channels)

        self.num_registers = 1
        self.register_node_init = hk.get_parameter(
            "register_node_init",
            shape=(self.num_registers, hidden_channels),
            init=hk.initializers.RandomNormal(),
        )

    def __call__(self, nodes, senders, receivers, aux):
        nodes = jax.nn.relu(hk.Linear(self.hidden_channels)(nodes))  # (nodes, hidden_channels)
        nodes = jnp.concatenate(
            [nodes, jnp.ones((nodes.shape[0], self.hidden_channels))], axis=-1
        )  # (nodes, hidden_channels + 1)
        nodes = jax.nn.relu(hk.Linear(self.hidden_channels)(nodes))  # (nodes, hidden_channels)

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
        aux_embedding = jax.nn.relu(hk.Linear(self.hidden_channels)(aux))  # (out_channels)
        pool = jnp.sum(x, axis=0)  # (hidden_channels,)
        pool_rep = jax.nn.relu(hk.Linear(self.hidden_channels)(pool))
        pool_aux_concat_ = jnp.concat(
            [pool_rep[jnp.newaxis, :], aux_embedding[jnp.newaxis, :]], axis=-1
        ).reshape((-1))  # (out_channels * 2,)
        pool_aux_concat = jax.nn.relu(hk.Linear(self.hidden_channels)(pool_aux_concat_))

        # Update node embeddings based on pool and aux
        pool_rep = jax.nn.relu(hk.Linear(self.hidden_channels)(pool))
        pool_rep = jnp.repeat(pool_rep[jnp.newaxis, :], x.shape[0], axis=0)
        aux_embedding = jax.nn.relu(hk.Linear(self.hidden_channels)(aux))  # (out_channels)
        aux_embedding = jnp.repeat(aux_embedding[jnp.newaxis, :], x.shape[0], axis=0)
        node_pool_aux_concat = jnp.concatenate([x, pool_rep, aux_embedding], axis=-1)
        node_embeddings = x + hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            jax.nn.relu(hk.Linear(self.hidden_channels)(node_pool_aux_concat))
        )

        aux_embedding = jax.nn.relu(hk.Linear(self.hidden_channels)(aux))  # (out_channels)

        return node_embeddings, aux_embedding, pool_aux_concat


class ObsEncoder(hk.Module):
    """Encodes a graph observation into (node_embeddings, aux_embedding).

    Wraps ``FeatureExtractor`` and discards the pooled global embedding,
    returning only the per-node and auxiliary embeddings.
    """

    def __init__(
        self,
        num_hidden: int,
        name: str = "obs_encoder",
    ):
        super().__init__(name=name)
        self.num_hidden = num_hidden

        self.feature_extractor = FeatureExtractor(
            hidden_channels=num_hidden,
        )

    def __call__(self, nodes, senders, receivers, aux) -> tuple[jnp.ndarray, jnp.ndarray]:
        node_embeddings, aux_embedding, _ = self.feature_extractor(
            nodes, senders, receivers, aux
        )  # (n_nodes, hidden_channels * 3)
        return node_embeddings, aux_embedding


class ProjectionHead(hk.Module):
    """Projects node and aux embeddings for self-consistency loss.

    Applies a two-layer MLP independently to node and aux embeddings,
    producing normalization-ready projections used in the self-consistency
    objective.
    """

    def __init__(self, num_hidden: int, num_out: int, name: str = "projection_head"):
        super().__init__(name=name)
        self.num_hidden = num_hidden
        self.num_out = num_out

    def __call__(
        self,
        node_embeddings: jnp.ndarray,
        aux_embeddings: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        node_embeddings = node_embeddings.astype(jnp.float32)
        node_embedding = jax.nn.relu(hk.Linear(self.num_hidden)(node_embeddings))
        node_proj = hk.Linear(
            self.num_out,
            w_init=hk.initializers.UniformScaling(0.01),
            b_init=hk.initializers.UniformScaling(0.25),
        )(node_embedding)

        aux_embeddings_ = aux_embeddings.astype(jnp.float32)
        aux_embeddings = jax.nn.relu(hk.Linear(self.num_hidden)(aux_embeddings_))
        aux_proj = hk.Linear(
            self.num_out,
            w_init=hk.initializers.UniformScaling(0.01),
            b_init=hk.initializers.UniformScaling(0.25),
        )(aux_embeddings)

        return node_proj, aux_proj


class PolicyHead(hk.Module):
    """Policy logits head.

    Reads per-node embeddings and the global aux embedding, then produces
    per-node action logits of shape ``(n_nodes,)``.
    """

    def __init__(self, num_hidden: int, name: str = "policy_head"):
        super().__init__(name=name)
        self.num_hidden = num_hidden

    def __call__(self, node_embeddings: jnp.ndarray, aux_embedding: jnp.ndarray) -> jnp.ndarray:
        # node_reps: (n_nodes, hidden_channels * 3), remove the register nodes
        node_embeddings = node_embeddings.astype(jnp.float32)[:-1]
        node_embeddings = jax.nn.relu(hk.Linear(self.num_hidden)(node_embeddings))

        pool_embedding = jnp.sum(node_embeddings, axis=0)  # (hidden_channels,)
        pool_embedding = jax.nn.relu(
            hk.Linear(self.num_hidden)(pool_embedding)
        )  # (hidden_channels,)

        aux_embedding = aux_embedding.astype(jnp.float32)
        aux_embedding = jax.nn.relu(hk.Linear(self.num_hidden)(aux_embedding))

        pool_aux_concat = jnp.concatenate(
            [pool_embedding, aux_embedding], axis=-1
        )  # (hidden_channels * 2,)

        # Repeat pool_aux_concat to match the number of nodes
        node_reps = jnp.repeat(pool_aux_concat[jnp.newaxis, :], node_embeddings.shape[0], axis=0)
        node_reps = jnp.concatenate([node_reps, node_embeddings], axis=-1)

        logits = hk.Linear(
            1,
            w_init=hk.initializers.UniformScaling(0.01),
            b_init=hk.initializers.UniformScaling(0.25),
        )(node_reps).reshape((-1,))  # (n_nodes,)
        return logits


class QValueHead(hk.Module):
    """Quantile value distribution head.

    Takes node and aux embeddings and returns a quantile distribution over
    the state value, shape ``(num_quantiles,)``.
    """

    def __init__(self, num_hidden: int, num_quantiles: int, name: str = "value_head"):
        super().__init__(name=name)
        self.num_hidden = num_hidden
        self.num_quantiles = num_quantiles

    def __call__(self, node_embeddings: jnp.ndarray, aux_embedding: jnp.ndarray) -> jnp.ndarray:
        # node_reps: (n_nodes, hidden_channels * 3), remove the register nodes
        node_embeddings_ = node_embeddings.astype(jnp.float32)[:-1]
        node_embeddings = jax.nn.relu(hk.Linear(self.num_hidden)(node_embeddings_))

        pool_embedding = jnp.sum(node_embeddings, axis=0)  # (hidden_channels,)
        pool_embedding = jax.nn.relu(
            hk.Linear(self.num_hidden)(pool_embedding)
        )  # (hidden_channels,)

        aux_embedding = aux_embedding.astype(jnp.float32)
        aux_embedding = jax.nn.relu(hk.Linear(self.num_hidden)(aux_embedding))

        pool_aux_concat = jnp.concatenate(
            [pool_embedding, aux_embedding], axis=-1
        )  # (hidden_channels * 2,)

        v_dist = jax.nn.relu(hk.Linear(self.num_hidden)(pool_aux_concat))
        v_dist = hk.Linear(
            self.num_quantiles,
            w_init=hk.initializers.UniformScaling(0.01),
            b_init=hk.initializers.UniformScaling(0.25),
        )(v_dist)  # (num_quantiles,)
        return v_dist  # (num_quantiles,)


class RewardHistoryHead(hk.Module):
    """Accumulated-return quantile distribution head.

    Re-runs ``FeatureExtractor`` from a raw observation to produce a quantile
    distribution over the discounted accumulated return.
    """

    def __init__(self, num_hidden: int, num_quantiles: int):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_quantiles = num_quantiles
        self.feature_extractor = FeatureExtractor(hidden_channels=num_hidden)

    def __call__(self, nodes, senders, receivers, aux):
        _, _, pool_aux_concat = self.feature_extractor(nodes, senders, receivers, aux)
        r_dist = hk.Linear(
            self.num_quantiles,
            w_init=hk.initializers.UniformScaling(0.01),
            b_init=hk.initializers.UniformScaling(0.25),
        )(pool_aux_concat)  # (num_quantiles,)
        return r_dist  # (num_quantiles,)


class DynamicsHead(hk.Module):
    """Transition model: (embeddings, action) → (next embeddings, reward dist).

    Adds a learnable action embedding at the selected node, then applies two
    ``SAGEConv`` layers to produce updated node embeddings together with a
    single-step reward quantile distribution of shape ``(num_quantiles,)``.
    """

    def __init__(
        self,
        num_hidden: int,
        num_quantiles: int,
        name: str = "dynamics_head",
    ):
        super().__init__(name=name)
        self.num_hidden = num_hidden
        self.num_quantiles = num_quantiles

        self.action_embedding = hk.get_parameter(
            "action_embedding",
            shape=(num_hidden,),
            init=hk.initializers.RandomNormal(),
        )

        self.conv1 = SAGEConv(out_channels=self.num_hidden)
        self.conv2 = SAGEConv(out_channels=self.num_hidden)
        self.num_registers = 1

    def __call__(
        self,
        node_embeddings,
        aux_embedding,
        senders,
        receivers,
        selected_node_index,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # These node embeddings include the register node
        node_embeddings_ = node_embeddings.astype(jnp.float32)

        # Node embeddings
        node_embeddings = node_embeddings.at[selected_node_index].add(self.action_embedding)
        node_embeddings = node_embeddings_ + jax.nn.relu(
            hk.Linear(self.num_hidden)(node_embeddings)
        )

        node_pool = jnp.sum(node_embeddings, axis=0)  # (hidden_channels,)
        node_pool = node_pool + jax.nn.relu(
            hk.Linear(self.num_hidden)(node_pool)
        )  # (hidden_channels,)

        selected_node = node_embeddings[selected_node_index]  # (hidden_channels,)
        sa_node_pool_aux_concat = jnp.concatenate(
            [node_pool, aux_embedding, selected_node], axis=-1
        )  # (hidden_channels * 3,)
        sa_embed = jax.nn.relu(hk.Linear(self.num_hidden)(sa_node_pool_aux_concat))
        sa_embed = selected_node + sa_embed  # (hidden,)
        r_dist = hk.Linear(
            self.num_quantiles,
            w_init=hk.initializers.UniformScaling(0.01),
            b_init=hk.initializers.UniformScaling(0.25),
        )(sa_embed).reshape((-1))  # (num_quantiles,)

        num_nodes = node_embeddings.shape[0] - 1
        # There are n_nodes + 1 nodes in the graph. We want to connect (register) nodes [n_nodes:] to all the prior nodes.
        receivers_ = jnp.arange(num_nodes)
        senders_ = jnp.full(num_nodes, num_nodes, dtype=jnp.int32)  # [num_node, num_node, ...]
        senders_ = jnp.concatenate([senders, senders_], axis=0)
        receivers_ = jnp.concatenate([receivers, receivers_], axis=0)

        # Make the graph undirected by adding the reverse edges
        senders = jnp.concatenate([senders_, receivers_], axis=0)
        receivers = jnp.concatenate([receivers_, senders_], axis=0)

        # Apply convolutions
        x = self.conv1(node_embeddings, senders, receivers)  # (n_nodes + 1, hidden_channels)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            jax.nn.relu(x)
        )  # (n_nodes + 1, hidden_channels)
        x = x + hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            jax.nn.relu(self.conv2(x, senders, receivers))
        )  # (n_nodes + 1, hidden_channels)

        aux_embedding = aux_embedding.astype(jnp.float32)
        aux_embedding = aux_embedding + jax.nn.relu(
            hk.Linear(self.num_hidden)(aux_embedding)
        )  # (hidden_channels,)

        # Update node embeddings based on pool and aux
        pool = jnp.sum(x, axis=0)
        pool = pool + jax.nn.relu(hk.Linear(self.num_hidden)(pool))
        pool = jnp.repeat(pool[jnp.newaxis, :], x.shape[0], axis=0)
        aux = jnp.repeat(aux_embedding[jnp.newaxis, :], x.shape[0], axis=0)
        node_pool_aux_concat = jnp.concatenate([x, pool, aux], axis=-1)
        node_embeddings = x + hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            jax.nn.relu(hk.Linear(self.num_hidden)(node_pool_aux_concat))
        )

        return (node_embeddings, r_dist, aux_embedding)


def make_network_apply_fns(args):
    """Build Haiku-transformed apply functions and a joint init function.

    Returns:
        init_model:           Joint init function used to create parameters.
        representation_apply: Apply function for ``ObsEncoder``.
        projection_apply:     Apply function for ``ProjectionHead``.
        policy_apply:         Apply function for ``PolicyHead``.
        critic_apply:         Apply function for ``QValueHead``.
        reward_history_apply: Apply function for ``RewardHistoryHead``.
        recurrent_inference:  Apply function for ``DynamicsHead``.
    """

    def representation_apply_fn(
        nodes: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        aux: jnp.ndarray,
    ):
        """Applies the representation model to the observation."""
        obs_encoder = ObsEncoder(num_hidden=args.num_hidden)
        return jax.vmap(obs_encoder)(nodes, senders, receivers, aux)

    def projection_apply_fn(
        node_embedding: jnp.ndarray,
        aux_embedding: jnp.ndarray,
    ) -> jnp.ndarray:
        """Applies the projection model to the observation embedding."""
        projection_head = ProjectionHead(num_hidden=args.num_hidden, num_out=args.num_hidden)
        return jax.vmap(projection_head)(node_embedding, aux_embedding)

    def critic_apply_fn(
        node_embedding: jnp.ndarray,
        aux_embedding: jnp.ndarray,
    ) -> jnp.ndarray:
        """Applies the critic model to the observation embedding."""
        value_head = QValueHead(
            num_hidden=args.num_hidden,
            num_quantiles=args.num_quantiles,
        )
        return jax.vmap(value_head)(node_embedding, aux_embedding)

    def reward_history_apply_fn(
        nodes: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        aux: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply the reward history head to the observation embedding."""
        net = RewardHistoryHead(
            num_hidden=args.num_hidden,
            num_quantiles=args.num_quantiles,
        )
        return jax.vmap(net)(nodes, senders, receivers, aux)

    def policy_apply_fn(
        node_embedding: jnp.ndarray,
        aux_embedding: jnp.ndarray,
    ) -> jnp.ndarray:
        """Applies the policy model to the observation embedding."""
        policy_logits = PolicyHead(num_hidden=args.num_hidden)
        return jax.vmap(policy_logits)(node_embedding, aux_embedding)

    def recurrent_inference_fn(
        node_embeddings,
        aux_embedding,
        senders,
        receivers,
        selected_node_index,
    ) -> jnp.ndarray:
        dynamics_head = DynamicsHead(
            num_hidden=args.num_hidden,
            num_quantiles=args.num_quantiles,
        )
        return jax.vmap(dynamics_head)(
            node_embeddings,
            aux_embedding,
            senders,
            receivers,
            selected_node_index,
        )

    def init_model_fn(nodes, senders, receivers, aux, selected_node_index):
        """Just for tracing through the entire model."""
        obs_encoder = ObsEncoder(num_hidden=args.num_hidden)
        node_embedding, aux_embedding = jax.vmap(obs_encoder)(
            nodes, senders, receivers, aux
        )  # (n_nodes, hidden_channels), (out_channels,)

        projection_head = ProjectionHead(num_hidden=args.num_hidden, num_out=args.num_hidden)
        projection = jax.vmap(projection_head)(node_embedding, aux_embedding)  # (hidden,)

        policy_head = PolicyHead(num_hidden=args.num_hidden)
        policy_logits = jax.vmap(policy_head)(node_embedding, aux_embedding)  # (n_nodes,)

        value_head = QValueHead(
            num_hidden=args.num_hidden,
            num_quantiles=args.num_quantiles,
        )
        value_dist = jax.vmap(value_head)(node_embedding, aux_embedding)  # (num_quantiles, n_nodes)

        reward_history_head = RewardHistoryHead(
            num_hidden=args.num_hidden, num_quantiles=args.num_quantiles
        )
        reward_history = jax.vmap(reward_history_head)(
            nodes, senders, receivers, aux
        )  # (num_quantiles,)

        dynamics_head = DynamicsHead(
            num_hidden=args.num_hidden,
            num_quantiles=args.num_quantiles,
        )
        dynamics_output = jax.vmap(dynamics_head)(
            node_embedding,
            aux_embedding,
            senders,
            receivers,
            selected_node_index,
        )
        return policy_logits, value_dist, reward_history, dynamics_output, projection

    init_model = hk.without_apply_rng(hk.transform(init_model_fn))
    policy_apply = hk.without_apply_rng(hk.transform(policy_apply_fn))
    representation_apply = hk.without_apply_rng(hk.transform(representation_apply_fn))
    projection_apply = hk.without_apply_rng(hk.transform(projection_apply_fn))
    critic_apply = hk.without_apply_rng(hk.transform(critic_apply_fn))
    reward_history_apply = hk.without_apply_rng(hk.transform(reward_history_apply_fn))
    recurrent_inference = hk.without_apply_rng(hk.transform(recurrent_inference_fn))

    return (
        init_model,
        representation_apply,
        projection_apply,
        policy_apply,
        critic_apply,
        reward_history_apply,
        recurrent_inference,
    )


if __name__ == "__main__":
    from types import SimpleNamespace

    args = SimpleNamespace(num_hidden=16, num_quantiles=10, num_actions=4)
    (
        init_model,
        representation_apply,
        projection_apply,
        policy_apply,
        critic_apply,
        reward_history_apply,
        recurrent_inference,
    ) = make_network_apply_fns(args)
    nodes = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
    senders = jnp.array([[0, 1]])
    receivers = jnp.array([[1, 0]])
    aux = jnp.array([[0.1, 0.2]])
    params = init_model.init(jax.random.PRNGKey(42), nodes, senders, receivers, aux, jnp.array([0]))
    node_reps, aux_embed = representation_apply.apply(params, nodes, senders, receivers, aux)
    print("node_reps:", node_reps.shape)  # (batch=1, n_nodes+1, hidden)
    print("aux_embed:", aux_embed.shape)  # (batch=1, hidden)
    logits = policy_apply.apply(params, node_reps, aux_embed)
    print("logits:", logits.shape)  # (batch=1, n_nodes)
    v_dist = critic_apply.apply(params, node_reps, aux_embed)
    print("v_dist:", v_dist.shape)  # (batch=1, num_quantiles)
    node_reps2, r_dist, aux_embed2 = recurrent_inference.apply(
        params, node_reps, aux_embed, senders, receivers, jnp.array([0])
    )
    print("next node_reps:", node_reps2.shape)  # (batch=1, n_nodes+1, hidden)
    print("r_dist:", r_dist.shape)  # (batch=1, num_quantiles)
