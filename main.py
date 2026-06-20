"""main.py — entrypoint for running risk-sensitive RL baselines.

Usage
-----
    python3 -m main --baseline tql --env grid-risk
    python3 -m main --baseline qrdqn --env space-invaders-risk --alpha 0.25
    python3 -m main --baseline qrdqn --env stochastic-bm --dataset datasets/stochastic_bm/instances_60_30_180
    python3 -m main --baseline qrdqn --env stochastic-mis --dataset datasets/stochastic_mis/instances_60_354
    python3 -m main --baseline sampled_tql --env grid-risk --alpha 0.25 --seed 42
    python3 -m main --baseline az_risk --env stochastic-bm --dataset datasets/stochastic_bm/instances_60_30_180 --alpha 0.25 --distortion wang
    python3 -m main --baseline mz_risk --env stochastic-mis --dataset datasets/stochastic_mis/instances_60_354 --alpha 0.25 --distortion cvar

Arguments
---------
--baseline      Algorithm: qrdqn | tql | sampled_tql | az_risk | mz_risk
--env           Environment short-name: grid-risk | space-invaders-risk |
                stochastic-bm | stochastic-mis
--dataset       Path to the dataset directory (required for graph environments).
                Passed to the env as sbm:<path> or smis:<path>.
--alpha-cvar    Override the CVaR confidence level from the config (optional).
--alpha         Override the risk alpha from the config (optional).
--distortion    Distortion measure for graph baselines: cvar | wang | pow (optional).
--seed          Override the random seed (optional).
                If omitted, the first seed in the config's ``seeds`` list is used.

Graph environments
------------------
When --env is ``stochastic-bm``, baselines are remapped to graph edge variants:
  qrdqn     → graph_qrdqn      (GraphConv, edge features)
  az_risk   → edge_az_risk     (Risk AlphaZero, graph/edge)
  mz_risk   → edge_mz_risk     (Risk MuZero, graph/edge)

When --env is ``stochastic-mis``, baselines are remapped to graph node variants:
  qrdqn     → graph_qrdqn_node (SAGEConv, no edge features)
  az_risk   → node_az_risk     (Risk AlphaZero, graph/node)
  mz_risk   → node_mz_risk     (Risk MuZero, graph/node)
"""

import argparse
import pathlib

import numpy as np
from pydantic_yaml import parse_yaml_file_as

from src.baselines.graph.edge.qrdqn import qrdqn as _graph_qrdqn
from src.baselines.graph.edge.risk_alphazero import alphazero as _edge_az_risk
from src.baselines.graph.edge.risk_muzero import muzero as _edge_mz_risk
from src.baselines.graph.qrdqn import qrdqn as _graph_qrdqn_node
from src.baselines.graph.risk_alphazero import alphazero as _node_az_risk
from src.baselines.graph.risk_muzero import muzero as _node_mz_risk
from src.baselines.qrdqn import qrdqn as _qrdqn
from src.baselines.risk_alphazero import alphazero as _az_risk
from src.baselines.risk_muzero import muzero as _mz_risk
from src.baselines.sampled_tql import tql as _sampled_tql
from src.baselines.tql import tql as _tql

# ---------------------------------------------------------------------------
# Baseline registry
# ---------------------------------------------------------------------------

BASELINES = {
    # Scalar / vector envs
    "qrdqn": (_qrdqn.Config, _qrdqn.run_experiment),
    "tql": (_tql.Config, _tql.run_experiment),
    "sampled_tql": (_sampled_tql.Config, _sampled_tql.run_experiment),
    "az_risk": (_az_risk.Config, _az_risk.run_experiment),
    "mz_risk": (_mz_risk.Config, _mz_risk.run_experiment),
    # Graph edge (stochastic-bm)
    "graph_qrdqn": (_graph_qrdqn.Config, _graph_qrdqn.run_experiment),
    "edge_az_risk": (_edge_az_risk.Config, _edge_az_risk.run_experiment),
    "edge_mz_risk": (_edge_mz_risk.Config, _edge_mz_risk.run_experiment),
    # Graph node (stochastic-mis)
    "graph_qrdqn_node": (_graph_qrdqn_node.Config, _graph_qrdqn_node.run_experiment),
    "node_az_risk": (_node_az_risk.Config, _node_az_risk.run_experiment),
    "node_mz_risk": (_node_mz_risk.Config, _node_mz_risk.run_experiment),
}

# User-facing baseline names (what the CLI accepts)
_USER_BASELINES = ["qrdqn", "tql", "sampled_tql", "az_risk", "mz_risk"]

# Graph envs: short name → (env_name prefix, config folder, remap dict)
_GRAPH_ENVS = {
    "stochastic-bm": (
        "sbm",
        "stochastic-bm",
        {
            "qrdqn": "graph_qrdqn",
            "az_risk": "edge_az_risk",
            "mz_risk": "edge_mz_risk",
        },
    ),
    "stochastic-mis": (
        "smis",
        "stochastic-mis",
        {
            "qrdqn": "graph_qrdqn_node",
            "az_risk": "node_az_risk",
            "mz_risk": "node_mz_risk",
        },
    ),
}

_BASELINE_ENV_REMAPS: dict[tuple[str, str], str] = {}

# Some baseline keys share a YAML config file name.
# Entries here override the filename used for YAML lookup.
_BASELINE_CONFIG_NAMES = {
    "graph_qrdqn_node": "graph_qrdqn",
    "edge_az_risk": "az_risk",
    "edge_mz_risk": "mz_risk",
    "node_az_risk": "az_risk",
    "node_mz_risk": "mz_risk",
}

# Maps CLI env names to their config folder names (where they differ)
_ENV_CONFIG_FOLDERS = {
    "space-invaders-risk": "space-invaders",
}

CONFIG_DIR = pathlib.Path("configs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_baseline_and_env(baseline: str, env: str, dataset: str | None) -> tuple[str, str]:
    """Remap baseline and build the env_name string.

    For graph envs, baseline names are remapped to their graph variant
    (edge-featured for SBM, node-only for SMIS) and env_name is constructed
    as ``<prefix>:<dataset>``.

    Returns
    -------
    baseline : Possibly remapped internal baseline key.
    env_name : The env_name string to set on the config.
    """
    if env in _GRAPH_ENVS:
        if dataset is None:
            raise ValueError(f"--dataset is required for graph env '{env}'")
        prefix, _, remap = _GRAPH_ENVS[env]
        env_name = f"{prefix}:{dataset}"
        if baseline in remap:
            baseline = remap[baseline]
    else:
        env_name = _BASELINE_ENV_REMAPS.get((baseline, env), env)

    return baseline, env_name


def _config_folder(env: str) -> str:
    """Return the config subdirectory for this env."""
    if env in _GRAPH_ENVS:
        return _GRAPH_ENVS[env][1]
    return _ENV_CONFIG_FOLDERS.get(env, env)


def _import_baseline(baseline: str):
    """Return (Config, run_experiment) for the chosen baseline."""
    if baseline not in BASELINES:
        raise ValueError(f"Unknown baseline '{baseline}'. Choose from: {list(BASELINES)}")
    return BASELINES[baseline]


def load_config(baseline: str, env: str):
    """Load the YAML config for the given baseline/env combination."""
    folder = _config_folder(env)
    config_filename = _BASELINE_CONFIG_NAMES.get(baseline, baseline)
    config_path = CONFIG_DIR / folder / f"{config_filename}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config found at {config_path}. Available baselines: {list(BASELINES)}"
        )
    Config, _ = _import_baseline(baseline)
    return parse_yaml_file_as(Config, config_path)


def save_logs(logs, baseline: str, env: str, seed: int, alpha_cvar: float) -> None:
    """Save numpy logs to logs/{env}/{baseline}/alpha_{alpha_cvar}/seed_{seed}.npy."""
    out_dir = pathlib.Path("logs") / env / baseline / f"alpha_{alpha_cvar}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_{seed}.npy"
    np.save(out_path, np.array(logs))
    print(f"Logs saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a risk-sensitive RL baseline.")
    parser.add_argument(
        "--baseline",
        required=True,
        choices=_USER_BASELINES,
        help="Algorithm to run.",
    )
    parser.add_argument(
        "--env",
        required=True,
        choices=["grid-risk", "space-invaders-risk", "stochastic-bm", "stochastic-mis"],
        help="Environment: grid-risk | space-invaders-risk | stochastic-bm | stochastic-mis",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the dataset directory (required for graph environments).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Risk alpha. Overrides the value in the config.",
    )
    parser.add_argument(
        "--distortion",
        type=str,
        default=None,
        choices=["cvar", "wang", "pow"],
        help="Distortion risk measure (graph baselines only). Overrides the value in the config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Defaults to the first entry in the config's seeds list.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    baseline, env_name = _resolve_baseline_and_env(args.baseline, args.env, args.dataset)

    if args.distortion is not None and args.env not in _GRAPH_ENVS:
        raise ValueError(
            f"--distortion is only supported for graph environments "
            f"({list(_GRAPH_ENVS)}), got '{args.env}'"
        )

    config = load_config(baseline, args.env)

    # --- Apply overrides ---
    config = config.model_copy(update={"env_name": env_name})

    if args.alpha is not None:
        config = config.model_copy(update={"alpha_cvar": args.alpha})

    if args.distortion is not None:
        config = config.model_copy(update={"distortion": args.distortion})

    # Resolve seed: explicit flag > first entry in seeds list > config default
    seed = args.seed
    if seed is None:
        if config.seeds:
            seed = config.seeds[0]
        else:
            seed = config.seed
    config = config.model_copy(update={"seed": seed})

    print(f"Running {baseline} on {env_name} (seed={seed}, alpha={config.alpha_cvar})")
    print("Config:", config)

    _, run_experiment = _import_baseline(baseline)
    json_logs, _ = run_experiment(config)

    save_logs(json_logs, baseline, args.env, seed, config.alpha_cvar)


if __name__ == "__main__":
    main()
