"""
graph_intelligence.py
----------------------
Matches the team pipeline exactly:

    Road Graph
        -> NetworkX Graph Processing
        -> Graph Intelligence (Degree, Betweenness, Closeness, Eigenvector Centrality)
        -> Critical Road Ranking
        -> Road Resilience Simulation (Road Blockage / Disaster Scenarios)
        -> Road Resilience Analysis (Connectivity, GCC, Path Length, Global Efficiency, Accessibility)
        -> Results Export

This module is validated on a synthetic road-like topology (Stage: Validate).
"""

import json
import numpy as np
import networkx as nx


def build_synthetic_grid(rows: int = 12, cols: int = 12, seed: int = 42) -> nx.Graph:
    """
    12x12 grid graph as a synthetic proxy for a road network, with a few
    shortcut edges added and enough edges pruned (while staying connected)
    to create genuine bottlenecks like a real, sparse road network has.
    """
    rng = np.random.default_rng(seed)
    G = nx.grid_2d_graph(rows, cols)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    nodes = list(G.nodes())
    for _ in range(4):
        u, v = rng.choice(nodes, size=2, replace=False)
        G.add_edge(int(u), int(v))

    edges = list(G.edges())
    rng.shuffle(edges)
    removed = 0
    target_removals = int(0.28 * len(edges))
    for u, v in edges:
        if removed >= target_removals:
            break
        if G.degree(u) > 1 and G.degree(v) > 1 and nx.is_connected(G):
            G.remove_edge(u, v)
            if nx.is_connected(G):
                removed += 1
            else:
                G.add_edge(u, v)

    return G


def compute_graph_intelligence(G: nx.Graph) -> dict:
    """Degree, Betweenness, Closeness, Eigenvector Centrality."""
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, normalized=True)
    closeness = nx.closeness_centrality(G)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=2000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = nx.eigenvector_centrality_numpy(G)
    return {
        "degree": degree,
        "betweenness": betweenness,
        "closeness": closeness,
        "eigenvector": eigenvector,
    }


def _normalize(d: dict) -> dict:
    vals = np.array(list(d.values()))
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:
        return {k: 0.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def critical_road_ranking(metrics: dict, weights: dict | None = None) -> list:
    """
    Combines all four normalized centrality metrics into one composite
    criticality score per node, then ranks nodes descending.
    """
    if weights is None:
        weights = {"degree": 0.2, "betweenness": 0.4, "closeness": 0.2, "eigenvector": 0.2}

    normalized = {name: _normalize(vals) for name, vals in metrics.items()}
    nodes = metrics["degree"].keys()
    composite = {
        n: sum(weights[m] * normalized[m][n] for m in weights)
        for n in nodes
    }
    return sorted(composite.items(), key=lambda kv: kv[1], reverse=True)


def simulate_blockage(G: nx.Graph, order: list) -> dict:
    """
    Removes nodes in the given order (a disaster/blockage scenario) and
    records Road Resilience Analysis metrics at every step:
    Connectivity (GCC fraction), Average Path Length, Global Efficiency,
    Accessibility (fraction of original node pairs still reachable).
    """
    G2 = G.copy()
    n0 = G2.number_of_nodes()
    total_pairs0 = n0 * (n0 - 1) / 2

    gcc_frac, avg_path_len, global_eff, accessibility = [], [], [], []

    def record():
        if G2.number_of_nodes() == 0:
            gcc_frac.append(0.0); avg_path_len.append(0.0)
            global_eff.append(0.0); accessibility.append(0.0)
            return
        comps = list(nx.connected_components(G2))
        largest = max(comps, key=len)
        gcc_frac.append(len(largest) / n0)

        sub = G2.subgraph(largest)
        avg_path_len.append(nx.average_shortest_path_length(sub) if len(largest) > 1 else 0.0)
        global_eff.append(nx.global_efficiency(G2))

        reachable_pairs = sum(len(c) * (len(c) - 1) / 2 for c in comps)
        accessibility.append(reachable_pairs / total_pairs0 if total_pairs0 > 0 else 0.0)

    record()
    for node in order:
        if node in G2:
            G2.remove_node(node)
        record()

    return {
        "connectivity_gcc": gcc_frac,
        "avg_path_length": avg_path_len,
        "global_efficiency": global_eff,
        "accessibility": accessibility,
    }


def average_random_blockage(G: nx.Graph, trials: int = 20) -> dict:
    """Averages all four resilience-analysis curves over many random blockage orders."""
    all_runs = []
    for s in range(trials):
        rng = np.random.default_rng(s)
        order = list(G.nodes())
        rng.shuffle(order)
        all_runs.append(simulate_blockage(G, order))
    keys = all_runs[0].keys()
    return {k: list(np.mean([run[k] for run in all_runs], axis=0)) for k in keys}


def resilience_gap(targeted: dict, random_avg: dict, metric: str = "connectivity_gcc") -> float:
    """AUC(random) - AUC(targeted) on a chosen resilience metric, normalized x-axis."""
    t, r = targeted[metric], random_avg[metric]
    m = min(len(t), len(r))
    x = np.linspace(0, 1, m)
    return float(np.trapezoid(r[:m], x) - np.trapezoid(t[:m], x))


def export_results(path: str, G: nx.Graph, metrics: dict, ranking: list, scenarios: dict) -> None:
    payload = {
        "graph": {"nodes": list(G.nodes()), "edges": [list(e) for e in G.edges()]},
        "metrics": {name: {str(k): v for k, v in d.items()} for name, d in metrics.items()},
        "critical_road_ranking": [{"node": n, "score": s} for n, s in ranking],
        "scenarios": scenarios,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    G = build_synthetic_grid(12, 12)
    print(f"Synthetic grid: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    metrics = compute_graph_intelligence(G)
    ranking = critical_road_ranking(metrics)
    print("\nTop 5 critical nodes (composite score):")
    for node, score in ranking[:5]:
        print(f"  node {node}: {score:.3f}")

    targeted_order = [n for n, _ in ranking]
    targeted = simulate_blockage(G, targeted_order)
    random_avg = average_random_blockage(G, trials=20)

    gap = resilience_gap(targeted, random_avg)
    print(f"\nResilience Gap (connectivity, targeted vs avg-random, 20 trials): {gap:.4f}")
    idx30 = int(0.3 * len(targeted_order))
    print(f"Global efficiency at 30% blockage (targeted): {targeted['global_efficiency'][idx30]:.4f}")
    print(f"Accessibility at 30% blockage (targeted): {targeted['accessibility'][idx30]:.4f}")

    export_results("smoke_test_export.json", G, metrics, ranking,
                    {"targeted": targeted, "random_avg": random_avg})
    print("\nResults exported to smoke_test_export.json")
    print("Smoke test complete — pipeline runs end-to-end on synthetic topology.")
