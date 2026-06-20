# Risk MuZero networks
#
# Neural network modules used by the Risk MuZero training loop.
#
# Contents
# --------
# FeatureExtractor       — CNN/MLP representation network
# ObsSeqEncoder          — GRU encoder over an observation-history window
# ActionEncoder          — Embeds a discrete action into the hidden space
# ProjectionHead         — Projects embeddings for self-consistency loss
# PolicyHead             — Policy logits head
# ValueHead              — Quantile value distribution head
# RewardHistoryHead      — Return-history distribution head (sequence input)
# DynamicsHead           — GRU-based transition model (next state + reward dist)
# make_network_apply_fns — Returns Haiku-transformed apply functions + init

import haiku as hk
import jax
import jax.numpy as jnp


class FeatureExtractor(hk.Module):
    """Representation network: encodes a raw observation into a hidden embedding.

    Uses a two-layer CNN followed by average-pooling and two linear layers for
    image observations, or two linear layers directly for state-vector inputs.
    """

    def __init__(self, num_hidden: int, is_state_vector: bool, name: str = "feature_extractor"):
        super().__init__(name=name)
        self.num_hidden = num_hidden
        self.is_state_vector = is_state_vector

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs.astype(jnp.float32)
        # If the state is an image, we use a CNN
        if not self.is_state_vector:
            x = hk.Conv2D(32, kernel_shape=3)(x)
            x = jax.nn.relu(x)
            x = hk.Conv2D(16, kernel_shape=3)(x)
            # Average pooling
            window_shape = (1, 2, 2, 1)
            strides = (1, 2, 2, 1)
            padding = "VALID"
            pooled = jax.lax.reduce_window(
                x,
                init_value=0.0,
                computation=jax.lax.add,
                window_dimensions=window_shape,
                window_strides=strides,
                padding=padding,
            )
            # Divide by window size to get average
            x = pooled / (2 * 2)  # (1, 5, 5, 32)

        x = x.reshape((x.shape[0], -1))
        x = hk.Linear(self.num_hidden)(x)
        x = jax.nn.tanh(x)
        return hk.Linear(
            self.num_hidden,
            w_init=hk.initializers.UniformScaling(0.01),
            b_init=hk.initializers.UniformScaling(0.25),
        )(x)


class ObsSeqEncoder(hk.Module):
    """GRU encoder over an observation-history window.

    Encodes a sequence of observations ``(batch, seq_len, *obs_shape)`` using a
    shared ``FeatureExtractor`` followed by a GRU, returning the last hidden
    state as a fixed-size embedding.
    """

    def __init__(
        self,
        num_hidden: int,
        is_state_vector: bool,
        name: str = "obs_seq_encoder",
    ):
        super().__init__(name=name)
        self.num_hidden = num_hidden
        self.is_state_vector = is_state_vector

        self.feature_extractor = FeatureExtractor(
            num_hidden=num_hidden,
            is_state_vector=is_state_vector,
        )

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        # obs: (1, history_length, *obs_shape)
        x = obs.astype(jnp.float32)

        # Join the sequence and batch dimensions to get latents
        batch, seq_len = x.shape[0], x.shape[1]
        x = x.reshape((-1, *x.shape[2:]))  # (batch * seq_len, *obs_shape)
        x = self.feature_extractor(x)  # (batch * seq_len, hidden_size)

        # Return the sequence dimension
        x = x.reshape((batch, seq_len, -1))  # (batch, seq_len, hidden_size)

        core = hk.GRU(self.num_hidden)
        rnn_initial_states = core.initial_state(batch)
        # Dynamic unroll the RNN core
        outs, _rnn_state = hk.dynamic_unroll(
            core,
            x,
            rnn_initial_states,
            time_major=False,
        )
        # Take the last time step (the current state) (batch, hidden_size)
        outs = outs[:, -1, :]  # type: ignore
        value = hk.Linear(self.num_hidden)(outs)  # (1, 128)
        value = jax.nn.tanh(value)

        return hk.Linear(
            self.num_hidden,
            w_init=hk.initializers.UniformScaling(0.01),
            b_init=hk.initializers.UniformScaling(0.25),
        )(value)


class ActionEncoder(hk.Module):
    """Embeds a discrete action index into the hidden space via a linear layer."""

    def __init__(self, num_hidden: int, name: str = "action_encoder"):
        super().__init__(name=name)
        self.num_hidden = num_hidden

    def __call__(self, action: jnp.ndarray) -> jnp.ndarray:
        action = action.astype(jnp.float32)
        return hk.Linear(
            self.num_hidden,
            w_init=hk.initializers.UniformScaling(0.01),
            b_init=hk.initializers.UniformScaling(0.25),
        )(action)


class ProjectionHead(hk.Module):
    """Projects a hidden embedding for the self-consistency loss."""

    def __init__(self, num_hidden: int, num_out: int, name: str = "projection_head"):
        super().__init__(name=name)
        self.num_hidden = num_hidden
        self.num_out = num_out

    def __call__(self, obs_embedding: jnp.ndarray) -> jnp.ndarray:
        obs_embedding_ = obs_embedding.astype(jnp.float32)
        obs_embedding = hk.Linear(self.num_hidden)(obs_embedding)
        obs_embedding = jax.nn.relu(obs_embedding)
        obs_embedding = obs_embedding_ + obs_embedding
        return hk.Linear(
            self.num_out,
            w_init=hk.initializers.UniformScaling(0.01),
            b_init=hk.initializers.UniformScaling(0.25),
        )(obs_embedding)


class PolicyHead(hk.Module):
    """Policy logits head: maps a hidden embedding to action logits."""

    def __init__(self, num_hidden: int, num_actions: int, name: str = "policy_head"):
        super().__init__(name=name)
        self.num_hidden = num_hidden
        self.num_actions = num_actions

    def __call__(self, obs_embedding: jnp.ndarray) -> jnp.ndarray:
        obs_embedding_ = obs_embedding.astype(jnp.float32)
        obs_embedding = hk.Linear(self.num_hidden)(obs_embedding)
        obs_embedding = jax.nn.relu(obs_embedding)
        obs_embedding = obs_embedding_ + obs_embedding
        return hk.Linear(
            self.num_actions,
            w_init=hk.initializers.UniformScaling(0.01),
            b_init=hk.initializers.UniformScaling(0.25),
        )(obs_embedding)


class ValueHead(hk.Module):
    """Quantile value distribution head, shape ``(batch, num_quantiles)``."""

    def __init__(self, num_hidden: int, num_quantiles: int, name: str = "value_head"):
        super().__init__(name=name)
        self.num_hidden = num_hidden
        self.num_quantiles = num_quantiles

    def __call__(self, obs_embedding: jnp.ndarray) -> jnp.ndarray:
        obs_embedding_ = obs_embedding.astype(jnp.float32)
        obs_embedding = hk.Linear(self.num_hidden)(obs_embedding)
        obs_embedding = jax.nn.relu(obs_embedding)
        obs_embedding = obs_embedding_ + obs_embedding

        return hk.Linear(
            self.num_quantiles,
            w_init=hk.initializers.UniformScaling(0.01),
            b_init=hk.initializers.UniformScaling(0.25),
        )(obs_embedding)  # (b, num_quantiles)


class RewardHistoryHead(hk.Module):
    """Return-history distribution head.

    Encodes an observation-history window via ``ObsSeqEncoder`` and outputs a
    quantile distribution over the discounted accumulated return seen so far,
    shape ``(batch, num_quantiles)``.
    """

    def __init__(
        self,
        num_hidden: int,
        num_quantiles: int,
        name: str = "reward_history_head",
    ):
        super().__init__(name=name)
        self.num_hidden = num_hidden
        self.num_quantiles = num_quantiles
        self.obs_encoder = ObsSeqEncoder(num_hidden=num_hidden, is_state_vector=True)

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        obs_embedding = self.obs_encoder(obs)
        obs_embedding = obs_embedding + jax.nn.tanh(hk.Linear(self.num_hidden)(obs_embedding))
        reward = hk.Linear(
            self.num_quantiles,
            w_init=hk.initializers.UniformScaling(0.01),
            b_init=hk.initializers.UniformScaling(0.25),
        )(obs_embedding)
        return reward  # (b, num_quantiles)


class DynamicsHead(hk.Module):
    """GRU-based transition model.

    Takes a hidden state embedding and action, and returns the next hidden state
    embedding and a quantile reward distribution, shape ``(batch, num_quantiles)``.
    Uses a residual connection from the input embedding to the next state.
    """

    def __init__(
        self,
        num_hidden: int,
        num_actions: int,
        num_quantiles: int,
        name: str = "dynamics_head",
    ):
        super().__init__(name=name)
        self.num_hidden = num_hidden
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles

    def __call__(
        self, obs_embedding: jnp.ndarray, action: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        obs_embedding = obs_embedding.astype(jnp.float32)
        action = action.astype(jnp.float32)

        one_hot_action = jax.nn.one_hot(action, num_classes=self.num_actions)
        action_embedding = hk.Linear(self.num_hidden)(one_hot_action)
        action_embedding = jax.nn.relu(action_embedding)
        next_state, rnn_output = hk.GRU(self.num_hidden)(action_embedding, obs_embedding)

        #  Add residual connection
        next_state = next_state + obs_embedding
        reward = hk.Linear(
            self.num_quantiles,
            w_init=hk.initializers.UniformScaling(0.01),
            b_init=hk.initializers.UniformScaling(0.25),
        )(rnn_output)
        next_state_embedding = hk.Linear(
            self.num_hidden,
            w_init=hk.initializers.UniformScaling(0.01),
            b_init=hk.initializers.UniformScaling(0.25),
        )(next_state)
        return next_state_embedding, reward


def make_network_apply_fns(args):
    """Construct Haiku-transformed apply functions and a joint init function.

    All network heads share a single parameter tree initialized by ``init_model``.

    Returns:
        init_model:           Joint init fn (initialises all heads).
        representation_apply: Apply fn for ``FeatureExtractor``.
        projection_apply:     Apply fn for ``ProjectionHead``.
        policy_apply:         Apply fn for ``PolicyHead``.
        critic_apply:         Apply fn for ``ValueHead``.
        reward_history_apply: Apply fn for ``RewardHistoryHead``.
        recurrent_inference:  Apply fn for ``DynamicsHead``.
    """

    def representation_apply_fn(
        obs: jnp.ndarray,
    ):
        """Applies the representation model to the observation."""
        obs_embedding = FeatureExtractor(
            num_hidden=args.num_hidden, is_state_vector=args.is_state_vector
        )(obs)
        return obs_embedding

    def projection_apply_fn(
        obs_embedding: jnp.ndarray,
    ) -> jnp.ndarray:
        """Applies the projection model to the observation embedding."""
        projection = ProjectionHead(num_hidden=args.num_hidden, num_out=args.num_hidden)(
            obs_embedding
        )
        return projection

    def critic_apply_fn(
        obs_embedding: jnp.ndarray,
    ) -> jnp.ndarray:
        """Applies the critic model to the observation embedding."""
        value = ValueHead(num_hidden=args.num_hidden, num_quantiles=args.num_quantiles)(
            obs_embedding
        )
        return value

    def reward_history_apply_fn(
        obs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply the reward history head to the observation embedding."""
        net = RewardHistoryHead(
            num_hidden=args.num_hidden,
            num_quantiles=args.num_quantiles,
        )
        return net(obs)

    def policy_apply_fn(
        obs_embedding: jnp.ndarray,
    ) -> jnp.ndarray:
        """Applies the policy model to the observation embedding."""
        policy_logits = PolicyHead(num_hidden=args.num_hidden, num_actions=args.num_actions)(
            obs_embedding
        )
        return policy_logits

    def recurrent_inference_fn(obs_embedding: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        dynamics_output = DynamicsHead(
            num_hidden=args.num_hidden,
            num_actions=args.num_actions,
            num_quantiles=args.num_quantiles,
        )(obs_embedding, action)
        return dynamics_output

    def init_model_fn(obs: jnp.ndarray, obs_history: jnp.ndarray, action: jnp.ndarray):
        """Just for tracing through the entire model."""
        obs_embedding = FeatureExtractor(
            num_hidden=args.num_hidden, is_state_vector=args.is_state_vector
        )(obs)
        projection = ProjectionHead(num_hidden=args.num_hidden, num_out=args.num_hidden)(
            obs_embedding
        )
        policy_logits = PolicyHead(num_hidden=args.num_hidden, num_actions=args.num_actions)(
            obs_embedding
        )
        value = ValueHead(num_hidden=args.num_hidden, num_quantiles=args.num_quantiles)(
            obs_embedding
        )
        reward_history = RewardHistoryHead(
            num_hidden=args.num_hidden, num_quantiles=args.num_quantiles
        )(obs_history)
        dynamics_output = DynamicsHead(
            num_hidden=args.num_hidden,
            num_actions=args.num_actions,
            num_quantiles=args.num_quantiles,
        )(obs_embedding, action)
        return policy_logits, value, dynamics_output, projection, reward_history

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
