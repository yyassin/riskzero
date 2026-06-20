# Risk AlphaZero networks
#
# Neural network modules used by the Risk AlphaZero training loop.
#
# Contents
# --------
# FeatureExtractor     — CNN/MLP feature extractor shared across heads
# ObsSeqEncoder        — GRU encoder over an observation-history window
# PredictionNetwork    — Policy + value head (single observation input)
# RewardHead           — Single-step reward distribution head
# RewardHistoryHead    — Return-history distribution head (sequence input)
# make_network_apply_fns — Returns Haiku-transformed apply functions + init

import haiku as hk
import jax
import jax.numpy as jnp


class FeatureExtractor(hk.Module):
    """Shared CNN/MLP feature extractor.

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


class PredictionNetwork(hk.Module):
    """Policy + value head conditioned on a single observation.

    Returns ``(logits, value_dist, obs_embedding)`` where ``logits`` are
    un-normalized policy logits, ``value_dist`` is a quantile value
    distribution, and ``obs_embedding`` is the feature vector passed to
    downstream heads.
    """

    def __init__(
        self,
        num_hidden: int,
        num_actions: int,
        num_quantiles: int,
        is_state_vector: bool,
        activation: str = "tanh",
    ):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_actions = num_actions
        self.is_state_vector = is_state_vector
        self.feature_extractor = FeatureExtractor(
            num_hidden=num_hidden, is_state_vector=is_state_vector
        )
        self.num_quantiles = num_quantiles
        self.activation = jax.nn.relu if activation == "relu" else jax.nn.tanh
        assert activation in ["relu", "tanh"]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.float32)
        x = self.feature_extractor(x)  # (1, hidden_size)

        # Value
        value = hk.Linear(self.num_hidden)(x)  # (1, 128)
        value = self.activation(value)
        value = hk.Linear(self.num_quantiles)(value)  # (b, num_quantiles)

        # Policy
        logits = hk.Linear(self.num_hidden)(x)  # (1, 128)
        logits = self.activation(logits)
        logits = hk.Linear(self.num_actions)(logits)  # (1, num_actions)
        return logits, value, x  # type: ignore


class RewardHead(hk.Module):
    """Single-step reward distribution head.

    Takes an observation embedding and a one-hot action and outputs a quantile
    distribution over the immediate reward, shape ``(batch, num_quantiles)``.
    """

    def __init__(
        self,
        num_hidden: int,
        num_actions: int,
        num_quantiles: int,
        name: str = "reward_head",
    ):
        super().__init__(name=name)
        self.num_hidden = num_hidden
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles

    def __call__(self, obs_embedding: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        obs_embedding = obs_embedding.astype(jnp.float32)
        action = action.astype(jnp.float32)

        one_hot_action = jax.nn.one_hot(action, num_classes=self.num_actions)
        action_embedding = hk.Linear(self.num_hidden)(one_hot_action)
        action_embedding = jax.nn.relu(action_embedding)

        combined = jnp.concatenate([obs_embedding, action_embedding], axis=-1)
        sa_embed = hk.Linear(self.num_hidden)(combined)
        sa_embed = jax.nn.tanh(sa_embed)
        sa_embed = sa_embed + obs_embedding

        reward = hk.Linear(
            self.num_quantiles,
            w_init=hk.initializers.UniformScaling(0.01),
            b_init=hk.initializers.UniformScaling(0.25),
        )(sa_embed)

        return reward  # (b, num_quantiles)


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


def make_network_apply_fns(args):
    """Construct Haiku-transformed apply functions and a joint init function.

    All three networks (prediction, reward, reward-history) share a single
    parameter tree initialized by ``init_model``.

    Returns:
        prediction_apply:      Apply fn for ``PredictionNetwork``.
        reward_apply:          Apply fn for ``RewardHead``.
        reward_history_apply:  Apply fn for ``RewardHistoryHead``.
        init_model:            Joint init fn (initialises all three heads).
    """

    def prediction_apply_fn(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the prediction network to the input."""
        net = PredictionNetwork(
            num_hidden=args.num_hidden,
            num_actions=args.num_actions,
            is_state_vector=args.is_state_vector,
            num_quantiles=args.num_quantiles,
        )
        return net(x)

    def reward_apply_fn(obs_embedding: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """Apply the reward head to the observation embedding and action."""
        net = RewardHead(
            num_hidden=args.num_hidden,
            num_actions=args.num_actions,
            num_quantiles=args.num_quantiles,
        )
        return net(obs_embedding, action)

    def reward_history_apply_fn(obs_history: jnp.ndarray) -> jnp.ndarray:
        """Apply the reward history head to the observation embedding."""
        net = RewardHistoryHead(
            num_hidden=args.num_hidden,
            num_quantiles=args.num_quantiles,
        )
        return net(obs_history)

    def init_model_fn(obs: jnp.ndarray, obs_history: jnp.ndarray, action: jnp.ndarray):
        """Initialise all three network heads in a single forward pass."""
        net = PredictionNetwork(
            num_hidden=args.num_hidden,
            num_actions=args.num_actions,
            is_state_vector=args.is_state_vector,
            num_quantiles=args.num_quantiles,
        )
        _logits, _value, obs_embed = net(obs)
        reward_net = RewardHead(
            num_hidden=args.num_hidden,
            num_actions=args.num_actions,
            num_quantiles=args.num_quantiles,
        )
        reward = reward_net(obs_embed, action)
        reward_history_apply_fn = RewardHistoryHead(
            num_hidden=args.num_hidden,
            num_quantiles=args.num_quantiles,
        )
        reward_history = reward_history_apply_fn(obs_history)
        return obs_embed, _logits, _value, reward, reward_history

    prediction_apply = hk.without_apply_rng(hk.transform(prediction_apply_fn))
    reward_apply = hk.without_apply_rng(hk.transform(reward_apply_fn))
    reward_history_apply = hk.without_apply_rng(hk.transform(reward_history_apply_fn))
    init_model = hk.without_apply_rng(hk.transform(init_model_fn))
    return prediction_apply, reward_apply, reward_history_apply, init_model
