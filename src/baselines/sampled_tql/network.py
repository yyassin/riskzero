# Sampled TQL neural networks
#
# Implements two network heads built on a shared CNN/MLP backbone:
#
# SharedBackbone
#   Vector input:  Flatten → Linear(512) → ReLU → Linear(256)
#   Image input:   Conv2D(16, 3×3) → ReLU → 3× ResBlock(16, 3×3)
#                  → Flatten → Linear(512) → ReLU → Linear(256)
#
# QRNetwork (action-value head)
#   SharedBackbone → ReLU → Linear(num_quantiles × num_actions)
#   → reshape to (b, num_quantiles, num_actions)
#   → CVaR distortion → q_values  (b, num_actions)
#
# RewardHistoryNetwork (reward-history head)
#   SharedBackbone applied to each step in the history window
#   → GRU(256) unrolled over the sequence
#   → last hidden state → residual Linear(256) → Linear(num_quantiles)
#
# Factory
#   create_qr_networks returns three hk.Transformed objects:
#   qr_model, reward_history_model, init_model (joint initialiser)

from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp

import lib.util as util


class QRNetworkOutputs(NamedTuple):
    q_values: jnp.ndarray
    q_dist: jnp.ndarray


class SharedBackbone(hk.Module):
    """Shared CNN/MLP feature extractor used by both network heads.

    Produces a ``(batch, 256)`` feature vector from either a flat vector
    observation or a spatial image observation (e.g. MinAtar).
    """

    def __init__(
        self,
        is_state_vector: bool,
    ) -> None:
        super().__init__()
        self.is_state_vector = is_state_vector

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.float32)

        # If the state is an image, we use a CNN
        if not self.is_state_vector:
            # x_in [batch_size, height, width, frames]
            x = hk.Conv2D(16, kernel_shape=3)(x)  # [padding=SAME]
            x = jax.nn.relu(x)

            # 3 residual blocks
            for _ in range(3):
                x_res = hk.Conv2D(16, kernel_shape=3, padding="SAME")(x)
                x_res = jax.nn.relu(x_res)
                x = x + x_res  # residual connection

        x = x.reshape((x.shape[0], -1))  # flatten
        x = hk.Linear(512)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(256)(x)

        return x  # (1, hidden_size)


class QRNetwork(hk.Module):
    """Quantile-regression action-value network.

    Reuses ``SharedBackbone`` for feature extraction, then projects to
    ``num_quantiles × num_actions`` quantile estimates.  CVaR distortion
    (at level ``alpha_cvar``) is applied to produce scalar ``q_values``
    used for action selection.
    """

    def __init__(
        self,
        num_actions: int,
        num_quantiles: int,
        is_state_vector: bool,
        alpha_cvar: float,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.is_state_vector = is_state_vector
        self.alpha_cvar = alpha_cvar

        self.tau_hats = util.make_tau_hats(num_quantiles)

        self.feature_extractor = SharedBackbone(
            is_state_vector=is_state_vector,
        )

    def __call__(self, x: jnp.ndarray) -> QRNetworkOutputs:
        x = x.astype(jnp.float32)

        x = self.feature_extractor(x)  # (b, hidden_size)
        x = jax.nn.relu(x)
        x = hk.Linear(self.num_quantiles * self.num_actions)(
            x
        )  # (b, args.num_quantiles * num_actions)
        q_dist = jnp.reshape(
            x, (-1, self.num_quantiles, self.num_actions)
        )  # (b, args.num_quantiles, num_actions)
        q_values = util.batched_cvar_distortion(
            q_dist, self.tau_hats, self.alpha_cvar
        )  # (b, num_actions)
        q_values = jax.lax.stop_gradient(q_values)  # (b, num_actions)
        return QRNetworkOutputs(q_dist=q_dist, q_values=q_values)


class RewardHistoryNetwork(hk.Module):
    """GRU-based network that predicts a return distribution from an observation history.

    Encodes each step in the history window with ``SharedBackbone``, unrolls a
    GRU over the sequence, and projects the final hidden state to
    ``num_quantiles`` quantile estimates of the cumulative reward.
    """

    def __init__(
        self,
        num_quantiles: int,
        is_state_vector: bool,
    ):
        super().__init__()
        self.num_quantiles = num_quantiles

        self.tau_hats = util.make_tau_hats(num_quantiles)
        self.feature_extractor = SharedBackbone(is_state_vector=is_state_vector)

    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        # Join the sequence and batch dimensions to get latents
        batch, seq_len = x.shape[0], x.shape[1]
        x = x.reshape((-1, *x.shape[2:]))  # (batch * seq_len, *obs_shape)
        x = self.feature_extractor(x)  # (batch * seq_len, hidden_size)
        x = jax.nn.relu(x)  # (batch * seq_len, hidden_size)

        # Return the sequence dimension
        x = x.reshape((batch, seq_len, -1))  # (batch, seq_len, hidden_size)

        core = hk.GRU(256)
        rnn_initial_states = core.initial_state(batch)

        # Dynamic unroll the RNN core
        outs, rnn_state = hk.dynamic_unroll(
            core,
            x,
            rnn_initial_states,
            time_major=False,
        )
        # Take the last time step as the current-state representation
        outs_ = outs[:, -1, :]  # type: ignore  # (batch, hidden_size)

        outs = hk.Linear(256)(outs_)  # (b, hidden_size)
        outs = outs_ + jax.nn.relu(outs)  # (b, hidden_size)
        reward = hk.Linear(
            self.num_quantiles,
            w_init=hk.initializers.UniformScaling(0.01),
            b_init=hk.initializers.UniformScaling(0.25),
        )(outs)

        return reward


def create_qr_networks(
    num_actions: int,
    num_quantiles: int,
    is_state_vector: bool,
    alpha_cvar: float,
) -> tuple[hk.Transformed, hk.Transformed, hk.Transformed]:
    """Wrap all network heads in Haiku transforms.

    Returns three stateless ``hk.Transformed`` objects:

    qr_model
        Action-value network.  Input: ``(b, *obs_shape)``.
        Output: ``QRNetworkOutputs(q_dist, q_values)``.
    reward_history_model
        GRU reward-history head.  Input: ``(b, seq_len, *obs_shape)``.
        Output: quantile estimates ``(b, num_quantiles)``.
    init_model
        Joint initialiser that calls both heads in one forward pass so a
        single ``init`` call produces shared parameters for both networks.
        Input: ``(b, history_length, *obs_shape)``.
    """

    def qr_model_fn(x: jnp.ndarray) -> QRNetworkOutputs:
        # x: (b, *obs_shape)
        network = QRNetwork(
            num_actions=num_actions,
            num_quantiles=num_quantiles,
            is_state_vector=is_state_vector,
            alpha_cvar=alpha_cvar,
        )
        return network(x)

    def reward_history_model_fn(x: jnp.ndarray) -> jnp.ndarray:
        # x: (b, seq_len, *obs_shape)
        network = RewardHistoryNetwork(
            num_quantiles=num_quantiles,
            is_state_vector=is_state_vector,
        )
        return network(x)

    def init_model_fn(x: jnp.ndarray) -> tuple[QRNetworkOutputs, jnp.ndarray]:
        # x: (b, history_length, *obs_shape)

        # This is used for initializing the model
        qr = QRNetwork(
            num_actions=num_actions,
            num_quantiles=num_quantiles,
            is_state_vector=is_state_vector,
            alpha_cvar=alpha_cvar,
        )

        reward_history = RewardHistoryNetwork(
            num_quantiles=num_quantiles,
            is_state_vector=is_state_vector,
        )

        qr_out = qr(x[:, 0, :])  # Use only the first frame for QR
        reward_history_out = reward_history(x)  # Use the entire history for reward prediction

        return qr_out, reward_history_out

    qr_model = hk.without_apply_rng(hk.transform(qr_model_fn))
    reward_history_model = hk.without_apply_rng(hk.transform(reward_history_model_fn))
    init_model = hk.without_apply_rng(hk.transform(init_model_fn))
    return qr_model, reward_history_model, init_model
