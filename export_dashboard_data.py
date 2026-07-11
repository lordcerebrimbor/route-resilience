"""
export_dashboard_data.py
--------------------------
Runs the full graph_intelligence pipeline on the synthetic validation
network and writes a single JSON payload consumed by dashboard.html.
"""

import json
import networkx as nx
from graph_intelligence import (
    build_synthetic_grid,
    compute_graph_intelligence,
    critical_road_ranking,
    simulate_blockage,
    average_random_blockage,
    resilience_gap,
)

GRID = 12
G = build_synthetic_grid(GRID, GRID, seed=42)
metrics = compute_graph_intelligence(G)
ranking = critical_road_ranking(metrics)
ranked_nodes = [n for n, _ in ranking]

targeted = simulate_blockage(G, ranked_nodes)
degree_order = sorted(dict(G.degree()), key=lambda n: G.degree(n), reverse=True)
degree_scn = simulate_blockage(G, degree_order)
random_avg = average_random_blockage(G, trials=20)

gap = resilience_gap(targeted, random_avg)

pos = {n: [n % GRID, GRID - 1 - (n // GRID)] for n in G.nodes()}

payload = {
    "meta": {
        "grid": GRID,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "resilience_gap": round(gap, 4),
    },
    "nodes": [{"id": n, "x": pos[n][0], "y": pos[n][1]} for n in G.nodes()],
    "edges": [[u, v] for u, v in G.edges()],
    "metrics": {name: {str(k): v for k, v in d.items()} for name, d in metrics.items()},
    "ranking": [{"node": n, "score": round(s, 4)} for n, s in ranking[:10]],
    "scenarios": {
        "targeted": targeted,
        "degree": degree_scn,
        "random_avg": random_avg,
    },
    "targeted_order": ranked_nodes,
}

with open("dashboard_data.json", "w") as f:
    json.dump(payload, f)

print(f"Exported dashboard_data.json — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, gap={gap:.4f}")
