"""
Create stochastic Barabási–Albert bipartite matching datasets.

Graph structure
---------------
An (n_left + n_right) × n_right bipartite graph is built as follows:

1. A bipartite graph with preferential attachment is sampled: for each right
   node, m = m_edges // n_right left nodes are selected with probability
   proportional to their current degree - preferential attachment.
2. n_right dummy left nodes are appended, each connected to exactly one
   distinct right node via a type-2 (zero-weight) edge.  This guarantees
   every right node has at least one incident edge, and effectively
   guarantees a skipping option for each right node.

Requirement: m_edges must be divisible by n_right.

Node indices are laid out as:
  [0,              n_left)             – original left nodes
  [n_left,         n_left + n_right)   – dummy left nodes
  [n_left+n_right, n_left+2*n_right)   – right nodes

Edge types
----------
Three stochastic edge types are defined in sbm/cvar_estimation.py:

  Type 0  –  Bernoulli(0.4): +20 (win) / –7  (loss)  [high-risk, high-reward]
  Type 1  –  Bernoulli(0.7): +6  (win) / –3  (loss)  [moderate-risk]
  Type 2  –  constant 0                              [dummy / zero-risk]

For random edges the type is drawn as:
  raw ~ Uniform{0, …, 11}  →  0–4 ⟹ type 0,  5–6 ⟹ type 1,  7–11 ⟹ type 2.
So, basically, 5/12 type-0, 2/12 type-1, 5/12 type-2 among random edges.
Dummy edges are always type 2.

Solver options
--------------
Two formulations are available for computing the "risk-averse" matching label:

  "max_weight" –  Maximum-weight bipartite matching LP using fixed deterministic
                  scores from RISK_MAPPING.  Fast; does not require weight samples.
                  The assumption here is that the CVaR solution will be well-approximated
                  by prioritizing the moderate-risk type-1 edges over the high-risk type-0 edges,
                  both prioritizing them over the zero-reward type-2 edges.

                  When compared to the MILP, which should be exact, this solution is often
                  on par or better (since solving the MILP exactly would require a large
                  number of samples, which is computationally expensive).

  "cvar_milp"  –  CVaR MILP.  A sample-average approximation of CVaR_alpha is
                  solved directly over n_weight_samples weight scenarios.

In both cases an additional expected-value matching (using E_MAPPING) is solved
as a baseline, and the realized CVaR / expected value of each solution is
estimated by Monte-Carlo simulation via estimate_cvar.

Output file format
------------------
Each instance is a plain-text file with the following lines:

  Line 1            : cvar_100_estimate   (E-value of the E-optimal matching)
  Line 2            : cvar_alpha_estimate (CVaR_alpha of the risk-averse matching)
  Line 3            : expected_estimate   (E-value of the risk-averse matching)
  Line 4            : n_nodes
  Line 5            : n_edges
  Lines 6 …         : u v type            (one edge per line)
  Remaining lines   : node_type           (one per node; 0 = left, 1 = right)
"""

import argparse
import os

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pulp
from tqdm import tqdm

from src.datasets.sbm.cvar_estimation import estimate_distortion_risk, sample_weights

# ---------------------------------------------------------------------------
# Edge-type score mappings used by the max-weight LP
# ---------------------------------------------------------------------------

# Expected value of each edge type: E[type0]=3.8, E[type1]=3.3, E[type2]=0
E_MAPPING = np.array([3.8, 3.3, 0.0])
RISK_MAPPING = np.array([1, 100, 2])  # This is the RISK AVERSE STRAT


def get_maximum_matching(n_left, n_nodes, edges, edge_types, mapping):
    """Solve a maximum-weight bipartite matching LP.

    Parameters
    ----------
    n_left     : Number of left-partition nodes (indices 0 … n_left-1).
    n_nodes    : Total number of nodes (left + right).
    edges      : Sequence of (u, v) edge pairs.
    edge_types : Integer array of edge types, one per edge.
    mapping    : Score array indexed by edge type.

    Returns
    -------
    selected_indices : list[int]  Indices into *edges* of matched edges.
    total_weight     : float      Objective value.
    """
    # Decision variables for edge selection
    prob = pulp.LpProblem("Maximum_Bipartite_Matching", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{u}_{v}", cat="Binary") for u, v in edges]

    # Objective: Maximize the sum of weighted edges
    prob += pulp.lpSum(mapping[edge_types[k]] * x[k] for k in range(len(edges)))

    # Constraints: Each left node can be matched to at most one right node
    for i in range(n_left):
        prob += pulp.lpSum(x[k] for k, (u, _v) in enumerate(edges) if u == i) <= 1

    # Constraints: Each right node can be matched to at most one left node
    for j in range(n_left, n_nodes):
        prob += pulp.lpSum(x[k] for k, (_u, v) in enumerate(edges) if v == j) <= 1

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    total_weight = pulp.value(prob.objective)
    selected_indices = [k for k in range(len(edges)) if x[k].value() == 1.0]
    return selected_indices, total_weight


def sample_graph_weight_instances(edge_types, n_samples, rng_key):
    """Draw *n_samples* independent weight realizations for all edges.

    Parameters
    ----------
    edge_types : jnp array of shape (n_edges,).
    n_samples  : Number of Monte-Carlo scenarios.
    rng_key    : JAX PRNGKey.

    Returns
    -------
    weights : jnp array of shape (n_samples, n_edges).
    """
    n_edges = edge_types.shape[0]
    types_flat = jnp.repeat(edge_types[None, :], n_samples, axis=0).reshape(-1)
    rng_key, subkey = jax.random.split(rng_key)
    keys = jax.random.split(subkey, types_flat.shape[0])
    weights = jax.vmap(sample_weights)(types_flat, keys)
    return weights.reshape((n_samples, n_edges))


def get_cvar_bm(n_left, n_nodes, edges, weight_samples, cvar_alpha):
    """Solve the CVaR bipartite matching MILP via sample-average approximation.

    Maximises CVaR_alpha of the total matching weight over the provided
    weight scenarios.

    Parameters
    ----------
    n_left         : Number of left-partition nodes.
    n_nodes        : Total number of nodes (left + right).
    edges          : Sequence of (u, v) edge pairs.
    weight_samples : Array of shape (n_samples, n_edges) with weight realizations.
    cvar_alpha     : CVaR tail probability (e.g. 0.25 for CVaR_25).

    Returns
    -------
    selected_indices : list[int]  Indices into *edges* of matched edges.
    total_weight     : float      CVaR objective value.
    """
    n_samples = weight_samples.shape[0]
    p_i = 1.0 / n_samples  # Equal probability weight per scenario

    prob = pulp.LpProblem("CVaR_Bipartite_Matching", pulp.LpMaximize)

    # CVaR auxiliary variables: v is the VaR threshold; t_i captures the
    # shortfall of scenario i below v (t_i = max(0, v - W_i(x))) - we will
    # subtract the expected shortfall from v in the objective to get CVaR.
    v = pulp.LpVariable("v", lowBound=None)
    t = [pulp.LpVariable(f"t_{i}", lowBound=0) for i in range(n_samples)]

    # Decision variables for edge selection
    x = [pulp.LpVariable(f"x_{u}_{v_}", cat="Binary") for u, v_ in edges]

    # Objective: CVaR_alpha = v - (1/alpha) * E[shortfall]
    prob += v - (1.0 / cvar_alpha) * p_i * pulp.lpSum(t)

    # Constraints: t_i >= v - W_i(x)  (linearises the shortfall)
    for i in range(n_samples):
        prob += t[i] >= v - pulp.lpSum(weight_samples[i, k] * x[k] for k in range(len(edges)))

    # Constraints: Each left node can be matched to at most one right node
    for i in range(n_left):
        prob += pulp.lpSum(x[k] for k, (u, _v) in enumerate(edges) if u == i) <= 1

    # Constraints: Each right node can be matched to at most one left node
    for j in range(n_left, n_nodes):
        prob += pulp.lpSum(x[k] for k, (_u, v) in enumerate(edges) if v == j) <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    total_weight = pulp.value(prob.objective)
    selected_indices = [k for k in range(len(edges)) if x[k].value() == 1.0]
    return selected_indices, total_weight


# ---------------------------------------------------------------------------
# Instance label computation
# ---------------------------------------------------------------------------


def get_instance_labels(
    init_rng_key,
    edges,
    edge_types,
    n_left,
    n_nodes,
    cvar_alpha,
    solver="max_weight",
    n_weight_samples=256,
):
    """Compute CVaR labels for a single graph instance.

    Two matchings are evaluated:
      - Risk-averse matching: found via *solver*; its CVaR_alpha and expected
        value are recorded.
      - Expected-value (EV) matching: found via get_maximum_matching with
        E_MAPPING; its expected value is recorded as a baseline.

    Parameters
    ----------
    init_rng_key    : JAX PRNGKey.
    edges           : Sequence of (u, v) edge pairs.
    edge_types      : Integer array of edge types, one per edge.
    n_left          : Left-partition size including dummy nodes (n_left_total).
    n_nodes         : Total number of nodes (left + right).
    cvar_alpha      : CVaR tail probability.
    solver          : "max_weight" or "cvar_milp".
    n_weight_samples: Number of scenarios (only used by "cvar_milp").

    Returns
    -------
    (cvar_alpha_est, cvar_100_est, expected_est) : tuple of float
        cvar_alpha_est – CVaR_alpha of the risk-averse matching.
        cvar_100_est   – Expected value of the EV-optimal matching.
        expected_est   – Expected value of the risk-averse matching.
    """
    rng_key, subkey = jax.random.split(init_rng_key)

    # --- Risk-averse matching ---
    if solver == "cvar_milp":
        weight_samples = sample_graph_weight_instances(edge_types, n_weight_samples, subkey)
        risk_indices, _ = get_cvar_bm(
            n_left=n_left,
            n_nodes=n_nodes,
            edges=edges,
            weight_samples=weight_samples,
            cvar_alpha=cvar_alpha,
        )
    else:  # "max_weight"
        risk_indices, _ = get_maximum_matching(
            n_left=n_left,
            n_nodes=n_nodes,
            edges=edges,
            edge_types=edge_types,
            mapping=RISK_MAPPING,
        )

    risk_types = edge_types[risk_indices]
    cvar_alpha_est = float(
        estimate_distortion_risk(subkey, edge_types=risk_types, n_trials=100_000, cvar_alpha=cvar_alpha)
    )
    expected_est = float(
        estimate_distortion_risk(subkey, edge_types=risk_types, n_trials=100_000, cvar_alpha=1.0)
    )

    # --- Expected-value matching (baseline) ---
    ev_indices, _ = get_maximum_matching(
        n_left=n_left,
        n_nodes=n_nodes,
        edges=edges,
        edge_types=edge_types,
        mapping=E_MAPPING,
    )
    cvar_100_est = float(
        estimate_distortion_risk(subkey, edge_types=edge_types[ev_indices], n_trials=100_000, cvar_alpha=1.0)
    )

    print(
        f"  risk-averse edge types: {risk_types}  |  "
        f"CVaR_{cvar_alpha}={cvar_alpha_est:.2f}  "
        f"E={expected_est:.2f}  E_baseline={cvar_100_est:.2f}"
    )
    return cvar_alpha_est, cvar_100_est, expected_est


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def create_graph(n_left, n_right, m_edges, seed):
    """Sample a stochastic Barabási–Albert bipartite graph.

    For each right node, m = m_edges // n_right edges are added one at a time.
    Each edge selects a left node with probability proportional to its current
    degree + 1 (the +1 seeds exploration of degree-zero nodes), and the degree
    is updated immediately after each selection — matching the standard BA model.
    Multi-edges within a round are prevented by zeroing out already-selected nodes.

    Node layout
    -----------
    [0,              n_left)           – original left nodes
    [n_left,         n_left + n_right) – dummy left nodes (one per right node)
    [n_left+n_right, n_left+2*n_right) – right nodes

    Parameters
    ----------
    n_left  : Number of original left nodes.
    n_right : Number of right nodes.
    m_edges : Number of random edges (must be divisible by n_right).
    seed    : Integer seed for reproducibility.

    Returns
    -------
    G_nodes      : NetworkX NodeView.
    G_edges      : NetworkX EdgeView.
    edge_types   : np.ndarray, shape (n_edges_total,), dtype int.
    node_types   : np.ndarray, shape (n_nodes_total,), dtype int
                   (0 = left partition, 1 = right partition).
    n_left_total : Updated left-partition size (= n_left + n_right).
    """
    assert m_edges % n_right == 0, "m_edges must be divisible by n_right"
    rng = np.random.default_rng(seed)
    m = m_edges // n_right  # edges added per right node
    n_left_total = n_left + n_right

    # --- Preferential-attachment edges ---
    left_degrees = np.zeros(n_left, dtype=int)
    random_edges_list = []
    for right_idx in range(n_right):
        right_node = n_left_total + right_idx  # final right-node index
        selected_left = []
        for _ in range(m):
            probs = (left_degrees + 1).astype(float)
            # Exclude already-selected nodes (no multi-edges)
            for s in selected_left:
                probs[s] = 0.0
            probs /= probs.sum()
            chosen = int(rng.choice(n_left, p=probs))
            selected_left.append(chosen)
            left_degrees[chosen] += 1
        random_edges_list.extend((ln, right_node) for ln in selected_left)
    random_edges = np.array(random_edges_list)

    # --- Dummy edges: dummy left node (n_left + i) → right node (n_left_total + i) ---
    dummy_edges = np.array([(n_left + i, n_left_total + i) for i in range(n_right)])
    edges = np.vstack((random_edges, dummy_edges))

    # Build NetworkX bipartite graph
    G = nx.Graph()
    G.add_nodes_from(range(n_left_total), bipartite=0)
    G.add_nodes_from(range(n_left_total, n_left_total + n_right), bipartite=1)
    G.add_edges_from(edges)

    # Assign edge types to random edges:
    #   raw ~ Uniform{0,…,11}: 0–4 → type 0, 5–6 → type 1, 7–11 → type 2
    raw = rng.integers(0, 12, size=m_edges)
    random_edge_types = np.where(raw < 5, 0, np.where(raw < 7, 1, 2))
    dummy_edge_types = np.full(n_right, 2, dtype=int)
    all_edge_types = np.concatenate((random_edge_types, dummy_edge_types))

    # Reorder edge types to match NetworkX edge iteration order
    edge_types_ordered = []
    for u, v in G.edges:
        idx = np.where((edges[:, 0] == u) & (edges[:, 1] == v))[0][0]
        edge_types_ordered.append(all_edge_types[idx])
    edge_types = np.asarray(edge_types_ordered)

    node_types = np.array([0] * n_left_total + [1] * n_right)

    return G.nodes, G.edges, edge_types, node_types, n_left_total


# ---------------------------------------------------------------------------
# Dataset file I/O
# ---------------------------------------------------------------------------


def write_instance_file(
    filename,
    n_nodes,
    edges,
    edge_types,
    node_types,
    cvar_alpha_estimate,
    cvar_100_estimate,
    expected_estimate,
):
    """Write a single graph instance to a plain-text file.

    File format
    -----------
    Line 1          : cvar_100_estimate   (E-value of the EV-optimal matching)
    Line 2          : cvar_alpha_estimate (CVaR_alpha of the risk-averse matching)
    Line 3          : expected_estimate   (E-value of the risk-averse matching)
    Line 4          : n_nodes
    Line 5          : n_edges
    Lines 6 …       : u v type            (one edge per line)
    Remaining lines : node_type           (one per node; 0=left, 1=right)
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(f"{cvar_100_estimate}\n")
        f.write(f"{cvar_alpha_estimate}\n")
        f.write(f"{expected_estimate}\n")
        f.write(f"{n_nodes}\n")
        f.write(f"{len(edges)}\n")
        for i, (u, v) in enumerate(edges):
            f.write(f"{u} {v} {edge_types[i]}\n")
        for nt in node_types:
            f.write(f"{nt}\n")


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def create_stochastic_ba_dataset(
    num_instances,
    n_left,
    n_right,
    m_edges,
    save_dir,
    cvar_alpha=0.25,
    solver="max_weight",
    n_weight_samples=256,
    init_seed=0,
):
    """Generate a dataset of stochastic BA bipartite matching instances.

    Parameters
    ----------
    num_instances    : Number of graph instances to generate.
    n_left           : Number of original left nodes.
    n_right          : Number of right nodes.
    m_edges          : Number of random edges per graph (must be divisible by n_right).
    save_dir         : Directory in which instance files are written.
    cvar_alpha       : CVaR tail probability for labelling.
    solver           : "max_weight" or "cvar_milp" (see get_instance_labels).
    n_weight_samples : Scenarios used by the "cvar_milp" solver.
    init_seed        : Base random seed; instance i uses seed init_seed + i.
    """
    n_nodes_total = n_left + 2 * n_right  # original left + dummy left + right

    for i in tqdm(range(num_instances)):
        seed = init_seed + i
        nodes, edges, edge_types, node_types, n_left_total = create_graph(
            n_left=n_left, n_right=n_right, m_edges=m_edges, seed=seed
        )

        cvar_alpha_est, cvar_100_est, expected_est = get_instance_labels(
            init_rng_key=jax.random.PRNGKey(seed),
            edges=edges,
            edge_types=edge_types,
            n_left=n_left_total,
            n_nodes=n_nodes_total,
            cvar_alpha=cvar_alpha,
            solver=solver,
            n_weight_samples=n_weight_samples,
        )

        instance_path = os.path.join(save_dir, f"instance_{i}.txt")
        write_instance_file(
            instance_path,
            n_nodes_total,
            edges,
            edge_types,
            node_types,
            cvar_alpha_est,
            cvar_100_est,
            expected_est,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate stochastic Barabási–Albert bipartite matching datasets."
    )
    parser.add_argument(
        "--n-left", type=int, default=60, help="Number of original left-partition nodes."
    )
    parser.add_argument("--n-right", type=int, default=30, help="Number of right-partition nodes.")
    parser.add_argument(
        "--m-edges",
        type=int,
        default=180,
        help="Number of random edges per graph (must be divisible by --n-right).",
    )
    parser.add_argument(
        "--num-instances", type=int, default=1024, help="Number of instances to generate."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="datasets/stochastic_bm/instances_60_30_180_ba",
        help="Output directory for instance files.",
    )
    parser.add_argument(
        "--cvar-alpha",
        type=float,
        default=0.25,
        help="CVaR tail probability (e.g. 0.25 for CVaR_25%%).",
    )
    parser.add_argument(
        "--solver",
        choices=["max_weight", "cvar_milp"],
        default="max_weight",
        help="Solver for the risk-averse matching label.",
    )
    parser.add_argument(
        "--n-weight-samples",
        type=int,
        default=256,
        help="Number of weight scenarios (cvar_milp only).",
    )
    parser.add_argument("--init-seed", type=int, default=0, help="Base random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    # Example invocations (run from the workspace root):
    #
    #   # Max-weight solver (fast, default):
    #   python3 -m src.datasets.sbm.create_stochastic_ba_dataset \
    #       --n-left 60 --n-right 30 --m-edges 180 \
    #       --num-instances 256 --save-dir datasets/stochastic_bm/instances_60_30_180_ba
    #
    #   # CVaR MILP solver:
    #   python3 -m src.datasets.sbm.create_stochastic_ba_dataset \
    #       --n-left 60 --n-right 30 --m-edges 180 \
    #       --num-instances 256 --save-dir datasets/stochastic_bm/instances_60_30_180_ba_cvar \
    #       --solver cvar_milp --cvar-alpha 0.25 --n-weight-samples 512

    args = parse_args()
    create_stochastic_ba_dataset(
        num_instances=args.num_instances,
        n_left=args.n_left,
        n_right=args.n_right,
        m_edges=args.m_edges,
        save_dir=args.save_dir,
        cvar_alpha=args.cvar_alpha,
        solver=args.solver,
        n_weight_samples=args.n_weight_samples,
        init_seed=args.init_seed,
    )
