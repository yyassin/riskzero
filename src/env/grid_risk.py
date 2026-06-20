# GridRisk environment
#
# A 5×5 grid-world where a single agent navigates from the top-left corner
# (0, 0) to the goal cell (4, 4) by moving only right or down.
#
# Cell types
# ----------
#   N (Neutral)  — no reward
#   Y (Yellow)   — small stochastic reward (+10, p=0.75)
#   B (Blue)     — small deterministic reward (+2)
#   R (Red)      — large stochastic reward (+40, p=0.3)
#   O (Orange)   — penalty (−10, certain)
#   G (Goal)     — episode terminates; no additional reward
#
# Each non-neutral reward cell can only be collected once per episode.
# A per-step cost of −1 is applied on every transition.
#
# Action space: {0: down, 1: right}  (restricting to these two directions
# makes the MDP Markov — the agent can never revisit a cell.)
#
# Observation: normalized (row, col) position in [0, 1]².

import dataclasses
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import pgx
from pgx import core
from pgx._src.struct import dataclass
from pgx._src.types import Array
from pgx.experimental.wrappers import auto_reset

# Cell-type indices (must match GRID values and REWARD_LOOKUP order)
N, Y, B, R, O, G = 0, 1, 2, 3, 4, 5  # Neutral, Yellow, Blue, Red, Orange, Goal

# The grid
GRID = jnp.array(
    [
        [N, N, R, O, O],
        [N, B, N, R, O],
        [Y, N, B, N, R],
        [O, Y, N, B, N],
        [O, O, Y, N, G],
    ]
)

# Note: We restrict the action space to [right, down] only. This makes
# the environment markov as the agent can't return to a previous state.
# It is not markov otherwise, since rewards are collected only on the first visit to a cell.
GRID_SIZE = GRID.shape[0]
MOVEMENTS = jnp.array([[1, 0], [0, 1]], dtype=jnp.int32)  # down, right
NUM_ACTIONS = MOVEMENTS.shape[0]


# Define reward helpers indexed by cell type
REWARD_LOOKUP = jnp.array([0.0, 10.0, 2.0, 40.0, -10.0, 0.0])
REWARD_PROBS = jnp.array([1.0, 0.75, 1.0, 0.3, 1.0, 1.0])
REWARD_POS_TO_IDX = {
    (0, 2): 0,
    (1, 1): 1,
    (1, 3): 2,
    (2, 0): 3,
    (2, 2): 4,
    (2, 4): 5,
    (3, 1): 6,
    (3, 3): 7,
    (4, 2): 8,
}
NUM_REWARD_POSITIONS = len(REWARD_POS_TO_IDX)
# Maps (row, col) → reward-slot index; −1 for cells with no reward.
LOOKUP_TABLE = -jnp.ones((GRID_SIZE, GRID_SIZE), dtype=int)
for (x, y), idx in REWARD_POS_TO_IDX.items():
    LOOKUP_TABLE = LOOKUP_TABLE.at[x, y].set(idx)


class EnvState(NamedTuple):
    pos: Array  # shape (2,) — (row, col)
    remaining_rewards: Array  # shape (NUM_REWARD_POSITIONS,) — 1 if uncollected, 0 if collected
    key: Array  # PRNG key for stochastic rewards


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((2,), dtype=jnp.float32)
    rewards: Array = jnp.float32([0.0])
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    _step_count: Array = jnp.int32(0)
    legal_action_mask: Array = jnp.ones((NUM_ACTIONS,), dtype=jnp.bool_)
    _x: EnvState = EnvState(
        pos=jnp.array([0, 0], dtype=jnp.int32),
        remaining_rewards=jnp.ones((NUM_REWARD_POSITIONS,), dtype=jnp.int32),
        key=jax.random.PRNGKey(0),
    )

    def replace(self, **kwargs: Any) -> "State":
        # pgx adds this method dynamically at runtime; we re-declare it here
        # so static type-checkers (Pylance/pyright) can resolve it.
        return dataclasses.replace(self, **kwargs)  # type: ignore[misc]

    @property
    def env_id(self) -> core.EnvId:
        return "grid_risk"  # type: ignore

    @property
    def x(self) -> EnvState:
        return self._x


class GridRisk(core.Env):
    def __init__(self, use_legal_actions: bool = False):
        super().__init__()
        self._use_legal_actions = use_legal_actions

    def _init(self, key: jnp.ndarray) -> State:
        return State(
            _x=EnvState(
                pos=jnp.zeros((2,), dtype=jnp.int32),
                remaining_rewards=jnp.ones((NUM_REWARD_POSITIONS,), dtype=jnp.int32),
                key=key,
            )
        )

    def _observe(self, state: State, _player_id: None = None) -> Array:
        pos = state.x.pos / (GRID_SIZE - 1)
        return pos

    def _step(self, state: State, action: jnp.ndarray, key: jnp.ndarray) -> State:
        pos = state.x.pos
        delta = MOVEMENTS[action]
        new_pos = jnp.clip(pos + delta, 0, GRID_SIZE - 1)
        remaining = state.x.remaining_rewards
        state = state.replace(
            _x=EnvState(
                pos=new_pos,
                remaining_rewards=remaining,
                key=key,
            )
        )

        idx = LOOKUP_TABLE[new_pos[0], new_pos[1]]
        updated_remaining = jax.lax.cond(
            (idx >= 0),
            lambda: remaining.at[idx].set(0),
            lambda: remaining,
        )

        # _rewards and _is_terminal receive `state` (with new_pos but
        # *old* remaining_rewards) so that reward logic can check whether this
        # is the first visit to the cell before we mark it as collected.
        return state.replace(
            _x=state.x._replace(
                remaining_rewards=updated_remaining,
            ),
            legal_action_mask=jax.lax.cond(
                self._use_legal_actions,
                self._legal_action_mask,
                lambda _: jnp.ones((NUM_ACTIONS,), dtype=jnp.bool_),
                state,
            ),
            rewards=self._rewards(state),
            terminated=self._is_terminal(state),
        )

    def _legal_action_mask(self, state: State) -> Array:
        # Prevent the agent from moving up or left on boundaries.
        pos = state.x.pos
        mask = jnp.ones((NUM_ACTIONS,), dtype=jnp.bool_)
        mask = mask.at[0].set(pos[0] < GRID_SIZE - 1)  # down
        mask = mask.at[1].set(pos[1] < GRID_SIZE - 1)  # right
        return mask

    def _is_terminal(self, state: State) -> Array:
        cell_type = GRID[state.x.pos[0], state.x.pos[1]]
        return cell_type == G

    def _rewards(self, state: State) -> Array:
        cell_type = GRID[state.x.pos[0], state.x.pos[1]]

        reward = -1.0  # per-step cost
        idx = LOOKUP_TABLE[state.x.pos[0], state.x.pos[1]]

        def reward_logic():
            use_prob = REWARD_PROBS[cell_type]
            prng_key, _ = jax.random.split(state.x.key)
            give_reward = jax.random.bernoulli(prng_key, use_prob)
            r = jnp.where(give_reward, REWARD_LOOKUP[cell_type], 0.0)
            return r

        cond_reward = jax.lax.cond(
            (idx >= 0) & (state.x.remaining_rewards[idx] == 1),
            reward_logic,
            lambda: 0.0,
        )
        reward += jnp.where(cell_type == O, REWARD_LOOKUP[O], 0.0)
        reward += cond_reward
        return jnp.asarray([reward], dtype=jnp.float32)

    @property
    def id(self) -> core.EnvId:
        return "grid_risk"  # type: ignore

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 1


if __name__ == "__main__":
    RunnerState = tuple[
        pgx.State,  # env_state
        jnp.ndarray,  # last_obs
        jax.Array,  # rng
    ]

    class Transition(NamedTuple):
        done: jnp.ndarray
        action: jnp.ndarray
        reward: jnp.ndarray
        obs: jnp.ndarray

    jax.default_device(jax.devices("gpu")[0])
    env = GridRisk()

    def make_update_fn():
        def _update_step(runner_state: RunnerState) -> tuple[RunnerState, Transition]:
            """Update the network and environment state"""
            step_fn = jax.vmap(auto_reset(env.step, env.init))

            def _env_step(runner_state: RunnerState, unused: Any) -> tuple[RunnerState, Transition]:
                env_state, last_obs, rng = runner_state
                action = jax.random.randint(rng, (1,), 0, NUM_ACTIONS)

                jax.debug.print(
                    "Env state: {env_state}, action: {action}, legal actions: {legal_actions}",
                    env_state=env_state.observation,
                    action=action,
                    legal_actions=env_state.legal_action_mask,  # type: ignore
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)  # (2, ), (2, )
                keys = jax.random.split(_rng, env_state.observation.shape[0])  # (b, 2)
                env_state = step_fn(env_state, action, keys)  # Returns batched env state
                # Store the s, a, r, done in a Transition
                transition = Transition(
                    env_state.terminated,  # (b,)
                    action,  # (b,)
                    jnp.squeeze(env_state.rewards),  # (b,)
                    last_obs,  # (b, *env.observation_shape)
                )
                runner_state = (env_state, env_state.observation, rng)
                return runner_state, transition

            # Collect a trajectory
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, 12)
            return runner_state, traj_batch

        return _update_step

    update_fn = make_update_fn()
    jitted_update_fn = jax.jit(update_fn)

    # INIT ENV
    rng = jax.random.PRNGKey(0)  # (2,)
    rng, _rng = jax.random.split(key=rng)  # (2,), (2,)
    reset_rng = jax.random.split(_rng, 1)  # Tuple: (2,), (2,)
    env_state = jax.jit(jax.vmap(env.init))(reset_rng)  # Batched env state

    # INIT RUNNER STATE
    runner_state = (
        env_state,
        env_state.observation,
        rng,
    )  # (b, *env.observation_shape), (b,)
    # RUN ENV
    runner_state, traj_batch = jitted_update_fn(runner_state)
    print("Runner state:", runner_state)
    print("Trajectory batch:", traj_batch)
