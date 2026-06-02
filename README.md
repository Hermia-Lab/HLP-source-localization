# HLP Core Implementation

This repository provides the core implementation of HLP, including:

- construction of initial hyperedge labels `B0`,
- construction of initial node labels `J0`,
- hyperedge-level label propagation,
- node-level label propagation,
- hybrid score fusion `C`,
- source selection strategies HLP-G and HLP-L.

The released code focuses on the main algorithmic workflow of HLP rather than the full experimental pipeline.

## Quick Start

```bash
pip install -r requirements.txt
python examples/toy_example.py
```

## Input Format

The input hypergraph is represented by an incidence matrix with shape `(num_nodes, num_hyperedges)`.

- `H[i, e] = 1`: node `i` belongs to hyperedge `e`.
- `infection_state[i] = 1`: node `i` is infected.
- `infection_state[i] = -1`: node `i` is susceptible.

## Main API

```python
from src.hlp_core import run_hlp

result = run_hlp(
    incidence_matrix=incidence_matrix,
    infection_state=infection_state,
    K=2,
    rho=0.5,
    mu=0.5,
    alpha=0.5,
    strategy="HLP-G",
)
```
