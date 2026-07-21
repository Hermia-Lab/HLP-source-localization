# HLP Core Implementation

This repository provides the core implementation of **HLP** (Hybrid Label Propagation for source localisation on hypergraphs) together with seven baseline methods used in the experimental comparison.

## Included Algorithms

**HLP** (`src/hlp_core.py`)
- Construction of initial hyperedge labels `B0` and node labels `J0`
- Hyperedge-level and node-level label propagation
- Hybrid score fusion `C`
- Source selection strategies HLP-G and HLP-L

**Baselines** (`src/baselines.py`)
- HCC — Higher-order Closeness Centrality
- HBC — Higher-order Betweenness Centrality
- HMCSM — Higher-order Monte Carlo-based Soft Boundary Estimation Method
- SLBIC — Source Localisation Based on Infection Cluster (two-stage: SLBNE + SLBIC)
- LPSI-1 — Label Propagation based Source Identification (node-level only)
- LPSI-2 — LPSI with dual pairwise channels via the line graph of the projected graph
- GCNSI — Graph Convolutional Networks based Source Identification

The released code focuses on the main algorithmic workflow of each method rather than the full experimental pipeline.

## Quick Start

```bash
pip install -r requirements.txt
python examples/toy_example.py
```

GCNSI additionally requires PyTorch and PyTorch Geometric:
```bash
pip install torch torch_geometric
```

## Input Format

The input hypergraph is represented by an incidence matrix with shape `(num_nodes, num_hyperedges)`.

- `H[i, e] = 1`: node `i` belongs to hyperedge `e`.
- `infection_state[i] = 1`: node `i` is infected.
- `infection_state[i] = -1`: node `i` is susceptible.

## Main API

### HLP

```python
from src.hlp_core import run_hlp

result = run_hlp(
    incidence_matrix=H, infection_state=Y,
    K=2, rho=0.5, mu=0.5, alpha=0.5, strategy="HLP-G",
)
```

### Baselines

```python
from src.baselines import run_hcc, run_hbc, run_hmcsm, run_slbic, run_lpsi1, run_lpsi2

# Each returns {"predicted_sources": [...], "node_scores": array}
result = run_hcc(H, Y, K=2)
result = run_hbc(H, Y, K=2)
result = run_hmcsm(H, Y, K=2, beta=0.3)
result = run_slbic(H, Y, alpha=4.0)       # K determined automatically
result = run_lpsi1(H, Y, alpha=0.5)       # K determined by local-maxima rule
result = run_lpsi2(H, Y, rho=0.5, mu=0.5, alpha=0.5)
```

### GCNSI (requires PyTorch)

```python
from src.baselines import GCNSIModel

model = GCNSIModel(hidden_channels=32)
model.train_model(train_H_list, train_Y_list, train_labels_list, num_epochs=100)
result = model.predict(H, Y, K=2)
```
