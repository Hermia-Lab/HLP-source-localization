import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.hlp_core import run_hlp


# Incidence matrix: rows are nodes, columns are hyperedges.
# H[i, e] = 1 means node i belongs to hyperedge e.
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

result = run_hlp(
    incidence_matrix=incidence_matrix,
    infection_state=infection_state,
    K=2,
    rho=0.5,
    mu=0.5,
    alpha=0.5,
    strategy="HLP-G",
)

print("Final scores C:")
print(np.round(result["C"], 4))
print("Predicted sources:", result["predicted_sources"])
