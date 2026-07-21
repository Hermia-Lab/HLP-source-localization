"""
Minimal example demonstrating all non-learning baselines on a toy hypergraph.

GCNSI is excluded here because it requires PyTorch and a training set;
see the docstring of ``GCNSIModel`` in ``src/baselines.py`` for usage.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.baselines import (
    run_hcc,
    run_hbc,
    run_hmcsm,
    run_slbic,
    run_lpsi1,
    run_lpsi2,
)

# ------------------------------------------------------------------
# Toy hypergraph (same as in toy_example.py)
# ------------------------------------------------------------------
incidence_matrix = np.array([
    [1, 0, 0, 0],  # node 0
    [1, 0, 0, 1],  # node 1
    [1, 1, 0, 0],  # node 2
    [0, 1, 0, 0],  # node 3
    [0, 1, 1, 0],  # node 4
    [0, 0, 1, 0],  # node 5
    [0, 0, 1, 1],  # node 6
    [0, 0, 0, 1],  # node 7
])

# Infection snapshot: 1 = infected, -1 = susceptible.
infection_state = np.array([1, 1, 1, -1, 1, -1, 1, -1])

K = 2  # number of sources to identify

print("=" * 50)
print("Toy hypergraph — 8 nodes, 4 hyperedges, K =", K)
print("Infected nodes:", list(np.where(infection_state == 1)[0]))
print("=" * 50)

# 1. HCC
res = run_hcc(incidence_matrix, infection_state, K)
print(f"\nHCC  predicted sources: {res['predicted_sources']}")

# 2. HBC
res = run_hbc(incidence_matrix, infection_state, K)
print(f"HBC  predicted sources: {res['predicted_sources']}")

# 3. HMCSM
res = run_hmcsm(incidence_matrix, infection_state, K, beta=0.3, repeats=5)
print(f"HMCSM predicted sources: {res['predicted_sources']}")

# 4. SLBIC (K is determined automatically)
res = run_slbic(incidence_matrix, infection_state, alpha=4.0)
print(f"SLBIC predicted sources: {res['predicted_sources']}")

# 5. LPSI-1 (K is determined by local-maxima rule)
res = run_lpsi1(incidence_matrix, infection_state, alpha=0.5)
print(f"LPSI-1 predicted sources: {res['predicted_sources']}")

# 6. LPSI-2 (dual pairwise channels + local-maxima rule)
res = run_lpsi2(incidence_matrix, infection_state, rho=0.5, mu=0.5, alpha=0.5)
print(f"LPSI-2 predicted sources: {res['predicted_sources']}")
