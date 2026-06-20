import numpy as np
import pgx

from src.env.grid_risk import GridRisk
from src.env.space_invaders import RiskMinAtarSpaceInvaders
from src.env.stochastic_bipartite_matching import (
    StochasticBipartiteMatching,
    batch_sbm_instances,
    load_sbm_instance,
)
from src.env.stochastic_max_ind_set import (
    StochasticMaxIndependentSet,
    batch_stochastic_mis_instances,
    load_stochastic_mis_instance,
)

# Number of problem instances to load for graph environments. We interleave even/odd
# instance IDs so that the train/test split each contain a balanced mix of instances.
_NUM_INSTANCES = 1024


def _load_graph_instances(folder_path, load_fn, batch_fn, num_instances: int = _NUM_INSTANCES):
    """Load and batch ``num_instances`` problem instances from ``folder_path``.

    Instances are interleaved even/odd so that the first half (train split)
    and second half (test split) each contain a balanced mix of instance IDs.
    """
    ids = np.arange(num_instances)
    ids = np.concatenate([ids[ids % 2 == 0], ids[ids % 2 == 1]])
    print(f"Loading {num_instances} instances from {folder_path}...")
    instances = batch_fn([load_fn(i, folder_path) for i in ids])
    print("Done.")
    return instances


def make_env(env_name: str, use_legal_actions: bool = False) -> tuple[pgx.Env, bool, int]:
    """Construct an environment by name.

    Named environments
    ------------------
    ``grid-risk``            GridRisk 5×5 grid-world.
    ``space-invaders-risk``  Risk-sensitive MinAtar Space Invaders.
    ``sbm:<path>``           Stochastic Bipartite Matching; loads instances
                             from the directory at ``<path>``.
    ``smis:<path>``          Stochastic Max Independent Set; loads instances
                             from the directory at ``<path>``.

    Any other name is forwarded to ``pgx.make``.

    Returns
    -------
    env              : The constructed environment.
    is_state_vector  : True if observations are flat vectors (vs. image/graph).
    num_actions      : Size of the discrete action space.
    """
    if env_name == "grid-risk":
        env = GridRisk(use_legal_actions=use_legal_actions)
    elif env_name == "space-invaders-risk":
        env = RiskMinAtarSpaceInvaders()
    elif env_name.startswith("sbm:"):
        folder_path = env_name[len("sbm:") :]
        instances = _load_graph_instances(folder_path, load_sbm_instance, batch_sbm_instances)
        env = StochasticBipartiteMatching(instances=instances)
    elif env_name.startswith("smis:"):
        folder_path = env_name[len("smis:") :]
        instances = _load_graph_instances(
            folder_path, load_stochastic_mis_instance, batch_stochastic_mis_instances
        )
        env = StochasticMaxIndependentSet(instances=instances)
    elif env_name.startswith("mis:"):
        folder_path = env_name[len("mis:") :]
        instances = _load_graph_instances(
            folder_path, load_stochastic_mis_instance, batch_stochastic_mis_instances
        )
        env = StochasticMaxIndependentSet(instances=instances)
    else:
        env = pgx.make(env_name)  # type: ignore

    is_state_vector = env_name in ["grid-risk"]
    return env, is_state_vector, env.num_actions
