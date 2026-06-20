"""
Create stochastic Erdos-Renyi maximum independent set datasets.

Graph structure
---------------
A G(n_nodes, m_edges) Erdos-Renyi random graph is sampled (gnm model).
For each non-type-2 node a paired "dummy" type-2 neighbor is added,
guaranteeing at least one zero-risk node in every neighborhood, representing
a skip option in the decision problem.

Node indices and types
----------------------
Nodes are assigned one of three weight types (see smis/cvar_estimation.py):

  Type 0  --  high-risk, high-reward  (Bernoulli(0.3): +40 / -7)
  Type 1  --  moderate-risk           (Bernoulli(0.75): +10 / -5)
  Type 2  --  zero-risk               (constant 0)

Type assignment: of n_nodes, the first 30 (in a random permutation) become
type 2, the next 15 become type 1, and the remainder become type 0.
Each non-type-2 node gets an extra type-2 dummy neighbor appended.

So effectively a 25%/25%/50% type distribution is imposed on the original nodes.

Solver options
--------------
Two formulations are available for solving for the risk-averse independent set:

  "max_weight" --  Maximum-weight MIS LP using fixed scores from RISK_MAPPING.
                   Fast; does not require weight samples.
                   The assumption here is that the CVaR solution will be well-approximated
                   by prioritizing the moderate-risk type-1 edges over the high-risk type-0 edges,
                   prioritizing both over the zero-reward type-2 edges.

  "cvar_milp"  --  CVaR MILP.  A sample-average approximation of CVaR_alpha is
                   solved directly over n_weight_samples weight scenarios.

Output file format
------------------
Each instance is a plain-text file with the following lines:

  Line 1            : cvar_100_estimate   (E-value of the E-optimal IS)
  Line 2            : cvar_alpha_estimate (CVaR_alpha of the risk-averse IS)
  Line 3            : expected_estimate   (E-value of the risk-averse IS)
  Line 4            : n_nodes
  Line 5            : n_edges
  Lines 6 ...       : u v                 (one edge per line)
  Remaining lines   : node_type           (one per node)
"""

import argparse
import os

import jax
import networkx as nx
import numpy as np
import pulp
from tqdm import tqdm

from src.datasets.smis.cvar_estimation import (
    estimate_distortion_risk,
    sample_graph_weight_instances,
)

# ---------------------------------------------------------------------------
# Node-type score mappings used by the max-weight LP
# ---------------------------------------------------------------------------

# Expected value of each node type: E[type0]=7.1, E[type1]=6.25, E[type2]=0
E_MAPPING = np.array([7.1, 6.25, 0.0])
RISK_MAPPING = np.array([1, 100, 2])


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------


def get_mis(n_nodes, edge_set, weights, mapping):
    """Solve a maximum-weight independent set LP.

    Parameters
    ----------
    n_nodes  : Number of nodes.
    edge_set : Sequence of (u, v) edge pairs defining adjacency constraints.
    weights  : Integer array of node types, one per node.
    mapping  : Score array indexed by node type.

    Returns
    -------
    selected_nodes : list[int]  Node indices in the independent set.
    total_weight   : float      Objective value.
    """
    prob = pulp.LpProblem("Maximum_Independent_Set", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n_nodes)]

    # Objective: maximise sum of node scores according to the mapping
    prob += pulp.lpSum(mapping[weights[i]] * x[i] for i in range(n_nodes))

    # Constraints: no two adjacent nodes in the IS
    for i, j in edge_set:
        prob += x[i] + x[j] <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    selected_nodes = [i for i in range(n_nodes) if x[i].value() == 1.0]
    total_weight = sum(weights[i] for i in selected_nodes)
    return selected_nodes, total_weight


def get_cvar_mis(n_nodes, edge_set, weight_samples, cvar_alpha):
    """Solve the CVaR MIS MILP via sample-average approximation.

    Maximises CVaR_alpha of the total IS weight over the provided weight
    scenarios.

    Parameters
    ----------
    n_nodes        : Number of nodes.
    edge_set       : Sequence of (u, v) edge pairs.
    weight_samples : Array of shape (n_samples, n_nodes) with weight realizations.
    cvar_alpha     : CVaR tail probability (e.g. 0.25 for CVaR_25).

    Returns
    -------
    selected_nodes    : list[int]  Node indices in the independent set.
    total_cvar_weight : float      CVaR objective value.
    """
    n_samples = weight_samples.shape[0]
    p_i = 1.0 / n_samples

    # CVaR auxiliary variables: v is the VaR threshold; t_i captures the shortfall in scenario i
    prob = pulp.LpProblem("Maximum_CVaR_Independent_Set", pulp.LpMaximize)
    v = pulp.LpVariable("v", lowBound=None)
    t = [pulp.LpVariable(f"t_{i}", lowBound=0) for i in range(n_samples)]
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n_nodes)]

    # Objective: CVaR_alpha = v - (1/alpha) * E[shortfall]
    prob += v - (1.0 / cvar_alpha) * p_i * pulp.lpSum(t)

    # Constraints: t_i >= v - W_i(x)
    for i in range(n_samples):
        prob += t[i] >= v - pulp.lpSum(weight_samples[i, j] * x[j] for j in range(n_nodes))

    # Constraints: no two adjacent nodes in the IS
    for i, j in edge_set:
        prob += x[i] + x[j] <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    selected_nodes = [i for i in range(n_nodes) if x[i].value() == 1.0]
    total_cvar_weight = pulp.value(prob.objective)
    return selected_nodes, total_cvar_weight


# ---------------------------------------------------------------------------
# Instance label computation
# ---------------------------------------------------------------------------


def get_instance_labels(
    init_rng_key,
    n_nodes,
    edge_set,
    node_types,
    cvar_alpha,
    solver="max_weight",
    n_weight_samples=256,
):
    """Compute CVaR labels for a single graph instance.

    Two independent sets are evaluated:
      - Risk-averse IS: found via *solver*; its CVaR_alpha and expected
        value are recorded.
      - Expected-value (EV) IS: found via get_mis with E_MAPPING; its
        expected value is recorded as the cvar_100 baseline.

    Parameters
    ----------
    init_rng_key    : JAX PRNGKey.
    n_nodes         : Number of nodes.
    edge_set        : Sequence of (u, v) edge pairs.
    node_types      : Integer array of node types, one per node.
    cvar_alpha      : CVaR tail probability.
    solver          : "max_weight" or "cvar_milp".
    n_weight_samples: Number of scenarios (only used by "cvar_milp").

    Returns
    -------
    (cvar_alpha_est, cvar_100_est, expected_est) : tuple of float
        cvar_alpha_est -- CVaR_alpha of the risk-averse IS.
        cvar_100_est   -- Expected value of the EV-optimal IS.
        expected_est   -- Expected value of the risk-averse IS.
    """
    rng_key, subkey = jax.random.split(init_rng_key)

    # --- Risk-averse IS ---
    if solver == "cvar_milp":
        weight_samples = sample_graph_weight_instances(node_types, n_weight_samples, subkey)
        risk_nodes, _ = get_cvar_mis(
            n_nodes=n_nodes,
            edge_set=edge_set,
            weight_samples=weight_samples,
            cvar_alpha=cvar_alpha,
        )
    else:  # "max_weight"
        risk_nodes, _ = get_mis(
            n_nodes=n_nodes,
            edge_set=edge_set,
            weights=node_types,
            mapping=RISK_MAPPING,
        )

    risk_types = node_types[risk_nodes]
    cvar_alpha_est = float(
        estimate_distortion_risk(
            subkey, node_types=risk_types, n_trials=100_000, cvar_alpha=cvar_alpha
        )
    )
    expected_est = float(
        estimate_distortion_risk(subkey, node_types=risk_types, n_trials=100_000, cvar_alpha=1.0)
    )

    # --- Expected-value IS (baseline) ---
    ev_nodes, _ = get_mis(
        n_nodes=n_nodes,
        edge_set=edge_set,
        weights=node_types,
        mapping=E_MAPPING,
    )
    cvar_100_est = float(
        estimate_distortion_risk(
            subkey, node_types=node_types[ev_nodes], n_trials=100_000, cvar_alpha=1.0
        )
    )

    print(
        f"  risk-averse node types: {risk_types}  |  "
        f"CVaR_{cvar_alpha}={cvar_alpha_est:.2f}  "
        f"E={expected_est:.2f}  E_baseline={cvar_100_est:.2f}"
    )
    return cvar_alpha_est, cvar_100_est, expected_est


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def create_graph(n_nodes, m_edges, seed=0):
    """Sample a stochastic Erdos-Renyi MIS graph.

    Parameters
    ----------
    n_nodes  : Number of base nodes (before dummy nodes are added).
    m_edges  : Number of edges to sample (gnm model).
    seed     : Random seed for graph sampling and type assignment.

    Returns
    -------
    nodes      : Node view of the final graph.
    edges      : Edge view of the final graph.
    node_types : Integer array of node types, one per node.
    """
    G = nx.gnm_random_graph(n=n_nodes, m=m_edges, seed=seed, directed=False)

    # Assign node types.  The first 30 nodes in a random permutation become type 2,
    # the next 15 become type 1, and the remainder become type 0.
    all_nodes = np.arange(n_nodes)
    all_nodes_permutation = np.random.permutation(all_nodes)
    types_2 = all_nodes_permutation[:30]
    types_1 = all_nodes_permutation[30:45]
    types_0 = all_nodes_permutation[45:]

    node_types = np.zeros(n_nodes, dtype=np.int32)
    node_types[types_0] = 0
    node_types[types_1] = 1
    node_types[types_2] = 2

    # Build the augmented graph.  For every non-type-2 node we attach a
    # dedicated type-2 "dummy" neighbor (labelled "{i}_add").  This
    # guarantees that every node always has at least one zero-risk neighbor
    # available as a skip option during the MIS decision process.
    new_G = nx.Graph()
    for i in G.nodes:
        if node_types[i] < 2:
            new_G.add_edge(i, f"{i}_add")
    # Copy over all original ER edges.
    for u, v in G.edges:
        new_G.add_edge(u, v)
    # Isolated nodes (degree-0 in G) would be missed by the edge loops above;
    # add them explicitly so every original node appears in the final graph.
    for i in range(n_nodes):
        if i not in new_G.nodes:
            new_G.add_node(i)

    # Relabel all nodes to contiguous integers 0 … |V|-1.
    new_ids = {old_id: i for i, old_id in enumerate(new_G.nodes)}
    new_node_types = []
    for node_id in new_G.nodes:
        if "_add" in str(node_id):
            # Dummy nodes are always type 2 (zero-risk).
            new_node_types.append(2)
        else:
            # Carry over the type assigned to the original node.
            new_node_types.append(int(node_types[int(node_id)]))

    new_G = nx.relabel_nodes(new_G, new_ids)
    return new_G.nodes, new_G.edges, np.asarray(new_node_types)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_instance_file(filename, cvar_100, cvar_alpha_est, expected_est, edges, node_types):
    """Write a dataset instance to *filename*.

    File format mirrors the SBM convention:
      Line 1 : cvar_100_estimate
      Line 2 : cvar_alpha_estimate
      Line 3 : expected_estimate
      Line 4 : n_nodes
      Line 5 : n_edges
      Lines 6+: u v   (one edge per line)
      Remaining: node_type  (one per node)
    """
    n_nodes = len(node_types)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(f"{cvar_100}\n")
        f.write(f"{cvar_alpha_est}\n")
        f.write(f"{expected_est}\n")
        f.write(f"{n_nodes}\n")
        f.write(f"{len(edges)}\n")
        for u, v in edges:
            f.write(f"{u} {v}\n")
        for t in node_types:
            f.write(f"{t}\n")


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------


def create_stochastic_er_dataset(
    n_nodes,
    m_edges,
    num_instances,
    save_dir,
    cvar_alpha=0.25,
    solver="max_weight",
    init_seed=0,
):
    """Generate *num_instances* ER-graph MIS instances and write them to disk.

    Parameters
    ----------
    n_nodes       : Number of base nodes per instance.
    m_edges       : Number of edges to sample (gnm model).
    num_instances : Total number of instances to generate.
    save_dir      : Output directory.
    cvar_alpha    : CVaR tail probability for labels.
    solver        : "max_weight" or "cvar_milp".
    init_seed     : Starting random seed (instance i uses seed init_seed + i).
    """
    for i in tqdm(range(num_instances)):
        seed = init_seed + i
        nodes, edges, node_types = create_graph(n_nodes=n_nodes, m_edges=m_edges, seed=seed)
        edges = list(edges)

        rng_key = jax.random.PRNGKey(seed)
        cvar_alpha_est, cvar_100_est, expected_est = get_instance_labels(
            init_rng_key=rng_key,
            n_nodes=len(nodes),
            edge_set=edges,
            node_types=node_types,
            cvar_alpha=cvar_alpha,
            solver=solver,
        )

        instance_path = os.path.join(save_dir, f"instance_{i}.txt")
        write_instance_file(
            instance_path,
            cvar_100=cvar_100_est,
            cvar_alpha_est=cvar_alpha_est,
            expected_est=expected_est,
            edges=edges,
            node_types=node_types,
        )


if __name__ == "__main__":
    # Example invocations (run from the workspace root):
    #
    #   # Max-weight solver (fast, default):
    #   python3 -m src.datasets.smis.create_stochastic_er_dataset \
    #       --n-nodes 60 --m-edges 354 \
    #       --num-instances 1024 --save-dir datasets/stochastic_mis/instances_60_354
    #
    #   # CVaR MILP solver:
    #   python3 -m src.datasets.smis.create_stochastic_er_dataset \
    #       --n-nodes 60 --m-edges 354 \
    #       --num-instances 1024 --save-dir datasets/stochastic_mis/instances_60_354_cvar \
    #       --solver cvar_milp --cvar-alpha 0.25

    parser = argparse.ArgumentParser(description="Generate stochastic ER MIS dataset.")
    parser.add_argument("--n-nodes", type=int, default=60)
    parser.add_argument("--m-edges", type=int, default=354)
    parser.add_argument("--num-instances", type=int, default=1024)
    parser.add_argument(
        "--save-dir", type=str, default="./datasets/stochastic_mis/instances_60_354"
    )
    parser.add_argument("--cvar-alpha", type=float, default=0.25)
    parser.add_argument(
        "--solver", type=str, default="max_weight", choices=["max_weight", "cvar_milp"]
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    create_stochastic_er_dataset(
        n_nodes=args.n_nodes,
        m_edges=args.m_edges,
        num_instances=args.num_instances,
        save_dir=os.path.abspath(args.save_dir),
        cvar_alpha=args.cvar_alpha,
        solver=args.solver,
        init_seed=args.seed,
    )
