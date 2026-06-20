# QR-DQN neural network
#
# Implements a Quantile-Regression DQN (QR-DQN) network using Haiku.
#
# Architecture
# ------------
# Vector input:  Flatten → Linear(512) → ReLU → Linear(256) → ReLU
#                → Linear(num_quantiles × num_actions)
#
# Image input:   Conv2D(16, 3×3) → ReLU → 3× ResBlock(16, 3×3)
#                → Flatten → Linear(512) → ReLU → Linear(256) → ReLU
#                → Linear(num_quantiles × num_actions)
#
# The output distribution over returns is represented as `num_quantiles`
# quantile estimates per action.  CVaR is computed per-action.

from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp

import lib.util as util


class QRNetworkOutputs(NamedTuple):
    q_values: jnp.ndarray
    q_dist: jnp.ndarray


class QRNetwork(hk.Module):
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

    def __call__(self, x: jnp.ndarray) -> QRNetworkOutputs:
        """Forward pass through the network.

        Args:
            x: Input state, shape (batch_size, *obs_shape).
            Returns:
            QRNetworkOutputs containing:
                q_values: CVaR estimates for each action, shape (batch_size, num_actions).
                q_dist: Quantile estimates for each action, shape (batch_size, num_quantiles, num_actions).
        """
        x = x.astype(jnp.float32)

        # If the state is an image, we use a CNN
        # CNN torso for image-based observations (e.g. MinAtar)
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
        x = jax.nn.relu(x)
        x = hk.Linear(self.num_quantiles * self.num_actions)(
            x
        )  # (b, args.num_quantiles * num_actions)

        q_dist = jnp.reshape(
            x, (-1, self.num_quantiles, self.num_actions)
        )  # (b, num_quantiles, num_actions)
        q_values = util.batched_cvar_distortion(
            q_dist, self.tau_hats, self.alpha_cvar
        )  # (b, num_actions)
        q_values = jax.lax.stop_gradient(q_values)  # (b, num_actions)
        return QRNetworkOutputs(q_dist=q_dist, q_values=q_values)


def create_qr_network(
    num_actions: int,
    num_quantiles: int,
    is_state_vector: bool,
    alpha_cvar: float,
) -> hk.Transformed:
    """Wrap QRNetwork in a Haiku transform.

    Returns a stateless ``hk.Transformed`` object with ``init`` and ``apply`` methods.
    """

    def model_fn(x: jnp.ndarray) -> QRNetworkOutputs:
        network = QRNetwork(
            num_actions=num_actions,
            num_quantiles=num_quantiles,
            is_state_vector=is_state_vector,
            alpha_cvar=alpha_cvar,
        )
        return network(x)

    model = hk.without_apply_rng(hk.transform(model_fn))
    return model
