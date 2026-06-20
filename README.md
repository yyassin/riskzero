# RiskZero

[![OpenReview](https://img.shields.io/badge/OpenReview-Paper-blue.svg)](https://openreview.net/forum?id=4oeFD3fG5E)

**Authors**: Yousef Yassin and Junfeng Wen

## Abstract

_AlphaZero_ and _MuZero_ have demonstrated superhuman performance across a range of strategic tasks.
Yet their reliance on maximizing expected returns limits their use in real-world settings, where even high-return policies may incur rare but catastrophic failures.
We introduce _RiskZero_ — the first _MuZero_-family method for risk-sensitive decision-making with _zero_ prior knowledge of environment dynamics.
_RiskZero_ learns distributional quantities to estimate trajectory-level risk, guiding search toward policies that explicitly avoid rare but severe outcomes.
We establish theoretical convergence to optimal, stationary risk-sensitive policies and validate our approach on environments designed to test risk-sensitive learning from pixels, as well as on larger-scale combinatorial tasks.
Across all settings, _RiskZero_ consistently outperforms state-of-the-art risk-sensitive baselines and improves sample efficiency, providing a general framework for safer and more reliable model-based reinforcement learning under uncertainty.

---

## Setup

The project uses a GPU-enabled Docker devcontainer. You will need Docker with the NVIDIA Container Toolkit installed.

**1. Build the image** (requires a host with CUDA 12+ drivers):

```bash
./docker/build.sh            # CUDA 12.9.0 (default)
./docker/build.sh 13.2.0     # Blackwell / CUDA 13
```

**2. Open in VS Code devcontainer**

Open the repository in VS Code and select **Reopen in Container** when prompted. Alternatively, open the command panel (`ctrl+P`) and type `>Dev Containers: Rebuild Container`. The `postCreateCommand` runs automatically and calls `uv sync` to install all dependencies into `.venv`.

Upon successfully opening the container, you should see a terminal with the following output confirming JAX can detect the GPU:

```
JAX version : 0.6.0
All devices : [CudaDevice(id=0)]
CUDA status : OK — 1 GPU(s) detected
             cuda:0
Done. Press any key to close the terminal.
```

**You may also install manually** (outside Docker - I don't recommend this; installing Jax is not fun):

```bash
uv sync
source .venv/bin/activate
```

---

## Repository Structure

The repo is structured as follows - there is a lot of redundancy / repetition; this is 
done purposefully to reduce indirection and coupling across components.

```
configs/            # YAML configs for each baseline × environment
datasets/           # Stochastic combinatorial problem instances are stored here
src/
  baselines/        # All baseline implementations
    tql/            # Trajectory Q-Learning
    sampled_tql/    # Sampled TQL
    qrdqn/          # QR-DQN
    risk_alphazero/ # Risk AlphaZero (scalar/vector envs)
    risk_muzero/    # Risk MuZero    (scalar/vector envs)
    graph/
      edge/         # Graph edge baselines (Stochastic Bipartite Matching)
      qrdqn/        # Graph QR-DQN (node, Stochastic MIS)
      sampled_tql/  # Graph Sampled TQL (node, Stochastic MIS)
      risk_alphazero/ # Graph Risk AlphaZero (node, Stochastic MIS)
      risk_muzero/    # Graph Risk MuZero   (node, Stochastic MIS)
  env/              # Environment definitions
  datasets/         # Dataset generation scripts
_mctx/              # Custom version of the mctx lib with patched risk-sensitive search
main.py             # Unified training entrypoint
```

---

## Running Experiments

All baselines are launched through `main.py`.

```
python3 -m main --baseline <alg> --env <env> [options]
```

**Options**

| Flag | Description |
|---|---|
| `--baseline` | `tql`, `sampled_tql`, `qrdqn`, `az_risk`, `mz_risk` |
| `--env` | `grid-risk`, `space-invaders-risk`, `stochastic-bm`, `stochastic-mis` |
| `--dataset` | Path to dataset directory (required for graph envs; see [Generating Datasets](#generating-datasets)) |
| `--alpha` | Risk level α (overrides config default) |
| `--distortion` | Distortion measure: `cvar`, `wang`, `pow` (graph envs only) |
| `--seed` | Random seed (overrides config default) |

Logs are saved to `logs/{env}/{baseline}/alpha_{α}/seed_{seed}.npy`.

### Examples

**Scalar / pixel environments:**

```bash
# Grid-Risk with Trajectory Q-Learning
python3 -m main --baseline tql --env grid-risk

# Space Invaders with QR-DQN
python3 -m main --baseline qrdqn --env space-invaders-risk --alpha 0.25

# Grid-Risk with RiskAlphaZero
python3 -m main --baseline az_risk --env grid-risk --alpha 0.25

# Space Invaders with RiskMuZero
python3 -m main --baseline mz_risk --env space-invaders-risk --alpha 0.25
```

**Stochastic Bipartite Matching (graph, edge features):**

```bash
python3 -m main --baseline az_risk --env stochastic-bm \
    --dataset datasets/stochastic_bm/instances_60_30_180 \
    --alpha 0.25 --distortion cvar

python3 -m main --baseline mz_risk --env stochastic-bm \
    --dataset datasets/stochastic_bm/instances_60_30_180 \
    --alpha 0.25 --distortion wang
```

**Stochastic Maximum Independent Set (graph, node features):**

```bash
python3 -m main --baseline az_risk --env stochastic-mis \
    --dataset datasets/stochastic_mis/instances_60_354 \
    --alpha 0.25 --distortion cvar

python3 -m main --baseline mz_risk --env stochastic-mis \
    --dataset datasets/stochastic_mis/instances_60_354 \
    --alpha 0.25 --distortion cvar
```

### Configs

Default hyperparameters live in `configs/{env}/{baseline}.yaml`. You may edit some fields directly on the command line via the flags above, or edit the YAML before running.

---

## Generating Datasets

You will need to generate the instances for combinatorial environments. Both Erdős–Rényi (ER) and Barabási–Albert (BA) graph families are supported.

### Stochastic Bipartite Matching (SBM)

```bash
python3 -m src.datasets.sbm.create_stochastic_er_dataset \
    --n-left <N_LEFT> --n-right <N_RIGHT> --m-edges <M_EDGES> \
    --num-instances 1024 --save-dir datasets/stochastic_bm/instances_<N_LEFT>_<N_RIGHT>_<M_EDGES>

python3 -m src.datasets.sbm.create_stochastic_ba_dataset \
    --n-left <N_LEFT> --n-right <N_RIGHT> --m-edges <M_EDGES> \
    --num-instances 1024 --save-dir datasets/stochastic_bm/instances_ba_<N_LEFT>_<N_RIGHT>_<M_EDGES>
```

Paper notation → `(n_left, n_right, edge_density%)` maps to `--m-edges` as follows:

| Paper | `--n-left` | `--n-right` | `--m-edges` |
|---|---|---|---|
| (30, 10, 10%) | 30 | 10 | 30 |
| (30, 10, 15%) | 30 | 10 | 45 |
| (30, 10, 20%) | 30 | 10 | 60 |
| (60, 30, 10%) | 60 | 30 | 180 |
| (60, 30, 15%) | 60 | 30 | 270 |
| (60, 30, 20%) | 60 | 30 | 360 |

### Stochastic Maximum Independent Set (SMIS)

```bash
python3 -m src.datasets.smis.create_stochastic_er_dataset \
    --n-nodes <N_NODES> --m-edges <M_EDGES> \
    --num-instances 1024 --save-dir datasets/stochastic_mis/instances_<N_NODES>_<M_EDGES>

python3 -m src.datasets.smis.create_stochastic_ba_dataset \
    --n-nodes <N_NODES> --m-edges <M_EDGES> \
    --num-instances 1024 --save-dir datasets/stochastic_mis/instances_ba_<N_NODES>_<M_EDGES>
```

Paper notation → `(n_nodes, edge_density%)` maps to `--m-edges` as follows:

| Paper | `--n-nodes` | `--m-edges` |
|---|---|---|
| (30, 20%) | 30 | 87 |
| (30, 30%) | 30 | 130 |
| (30, 40%) | 30 | 174 |
| (60, 20%) | 60 | 354 |
| (60, 30%) | 60 | 531 |
| (60, 40%) | 60 | 708 |

---

## Acknowledgements

I'm super grateful for the following repos, which we build upon, for their clean and performant code.
In particular, this project probably wouldn't exist without mctx and jax - praise vmap and being able to batch the search 🙏

- [mctx](https://github.com/google-deepmind/mctx) — Monte Carlo tree search in JAX
- [JAX](https://github.com/jax-ml/jax) — High-performance numerical computing
- [pgx](https://github.com/sotetsuk/pgx) — JAX-native game environments
- [rlax](https://github.com/google-deepmind/rlax) — RL building blocks in JAX
- [haiku](https://github.com/google-deepmind/dm-haiku) - Deep learning primitives in JAX
- [haiku-geometric](https://github.com/alexOarga/haiku-geometric) — Graph neural networks for Haiku

---

## Citation

```bibtex
@inproceedings{yassin2026riskzero,
  title         = {RiskZero: Plan More to Risk Less with a Learned Model},,
  author        = {Yassin, Yousef and Wen, Junfeng},
  booktitle     = {Forty-third International Conference on Machine Learning},
  year          = {2026},
  organization  = {PMLR}
}
```
