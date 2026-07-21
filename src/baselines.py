"""
Baseline methods for source localization on hypergraphs.

This file contains core implementations of seven baseline algorithms:

1) HCC   — Higher-order Closeness Centrality
2) HBC   — Higher-order Betweenness Centrality
3) HMCSM — Higher-order Monte Carlo-based Soft Boundary Estimation Method
4) SLBIC — Source Localization Based on Infection Cluster (two-stage: SLBNE + SLBIC)
5) LPSI-1 — Label Propagation based Source Identification (node-level only)
6) LPSI-2 — LPSI with dual pairwise channels via the line graph of the projected graph
7) GCNSI — Graph Convolutional Networks based Source Identification

Input convention
----------------
incidence_matrix : shape (num_nodes, num_hyperedges), H[i, e] = 1 if node i ∈ hyperedge e.
infection_state  : length num_nodes, 1 = infected, -1 = susceptible.

All ``run_*`` entry points return a dict with at least ``predicted_sources`` and
``node_scores``.  Additional intermediate results are included where informative.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable, Dict, List, Literal, Optional, Sequence, Set

import networkx as nx
import numpy as np
import scipy.sparse as sp

ArrayLike = Sequence[float] | np.ndarray


# =====================================================================
# Common utilities
# =====================================================================

def _to_dense(incidence_matrix: ArrayLike | sp.spmatrix) -> np.ndarray:
    if sp.issparse(incidence_matrix):
        return incidence_matrix.toarray().astype(float)
    return np.asarray(incidence_matrix, dtype=float)


def build_projected_adjacency(incidence_matrix: ArrayLike | sp.spmatrix,
                              binary: bool = True) -> np.ndarray:
    """Project the hypergraph to a simple node graph via clique expansion."""
    H = _to_dense(incidence_matrix)
    A = H @ H.T
    np.fill_diagonal(A, 0.0)
    if binary:
        A[A > 0] = 1.0
    return A


def get_node_neighbors(incidence_matrix: ArrayLike | sp.spmatrix) -> Dict[int, Set[int]]:
    """Return neighbours of each node in the projected node graph."""
    A = build_projected_adjacency(incidence_matrix, binary=True)
    return {i: set(np.flatnonzero(A[i])) for i in range(A.shape[0])}


def _normalize_symmetric(A: np.ndarray) -> np.ndarray:
    """Symmetric normalisation D^{-1/2} A D^{-1/2}."""
    degree = A.sum(axis=1)
    d_inv_sqrt = np.zeros_like(degree, dtype=float)
    nz = degree > 0
    d_inv_sqrt[nz] = np.power(degree[nz], -0.5)
    return d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]


def _infected_candidates(infection_state: ArrayLike) -> List[int]:
    Y = np.asarray(infection_state).reshape(-1)
    return [i for i, y in enumerate(Y) if y == 1]


def _projected_networkx_graph(incidence_matrix: ArrayLike | sp.spmatrix) -> nx.Graph:
    """Build a NetworkX graph from the clique-expanded projection."""
    A = build_projected_adjacency(incidence_matrix, binary=True)
    G = nx.Graph()
    G.add_nodes_from(range(A.shape[0]))
    rows, cols = np.where(A > 0)
    G.add_edges_from(zip(rows.tolist(), cols.tolist()))
    return G


def _build_s_adjacency(incidence_matrix: ArrayLike | sp.spmatrix,
                       s: int = 1) -> np.ndarray:
    """
    Build the s-adjacency matrix for the hypergraph.

    Two nodes are s-adjacent if they co-occur in at least *s* common
    hyperedges.  The entry A_s[i, j] is 1 when the overlap count >= s.
    """
    H = _to_dense(incidence_matrix)
    overlap = H @ H.T
    np.fill_diagonal(overlap, 0.0)
    A_s = (overlap >= s).astype(float)
    return A_s


def _s_adjacency_graph(incidence_matrix: ArrayLike | sp.spmatrix,
                       s: int = 1,
                       node_subset: Optional[Set[int]] = None) -> nx.Graph:
    """Return a NetworkX graph from the s-adjacency matrix, optionally
    restricted to a node subset (e.g. infected subgraph)."""
    A_s = _build_s_adjacency(incidence_matrix, s)
    N = A_s.shape[0]
    G = nx.Graph()
    nodes = range(N) if node_subset is None else sorted(node_subset)
    G.add_nodes_from(nodes)
    for i in nodes:
        for j in nodes:
            if j > i and A_s[i, j] > 0:
                G.add_edge(i, j)
    return G


# =====================================================================
# 1. HCC — Higher-order Closeness Centrality
# =====================================================================

def _hcc_scores(incidence_matrix: ArrayLike | sp.spmatrix,
                infection_state: ArrayLike,
                s_max: int = 3,
                alpha: float = 1.0) -> np.ndarray:
    """
    Higher-order closeness centrality on the infected sub-hypergraph.

    .. math::

        C_C^H(i) = \\sum_{s=1}^{s_{\\max}} \\alpha^s
        \\sum_{j \\neq i,\\, v_j \\in O} \\frac{1}{d_v^s(i,j)}

    where :math:`d_v^s(i,j)` is the shortest-path distance in the
    *s*-adjacency graph restricted to infected nodes, and :math:`\\alpha`
    weights different topological orders.
    """
    N = _to_dense(incidence_matrix).shape[0]
    Y = np.asarray(infection_state).reshape(-1)
    infected = set(i for i in range(N) if Y[i] == 1)

    if not infected:
        return np.zeros(N)

    scores = np.zeros(N, dtype=float)

    for s in range(1, s_max + 1):
        G_s = _s_adjacency_graph(incidence_matrix, s=s, node_subset=infected)
        dist = dict(nx.all_pairs_shortest_path_length(G_s))
        weight = alpha ** s

        for v in infected:
            total = 0.0
            for u in infected:
                if u != v:
                    d = dist.get(v, {}).get(u, None)
                    if d is not None and d > 0:
                        total += 1.0 / d
            scores[v] += weight * total

    return scores


def run_hcc(incidence_matrix: ArrayLike | sp.spmatrix,
            infection_state: ArrayLike,
            K: int,
            s_max: int = 3,
            alpha: float = 1.0) -> dict:
    """
    HCC: rank infected nodes by higher-order closeness centrality and
    return the top-K as predicted sources.

    Parameters
    ----------
    K : int
        Number of sources to return (must match the true source count).
    s_max : int
        Maximum s-order to consider (s = 1 corresponds to the standard
        projected graph).
    alpha : float
        Weighting factor for different s-orders.
    """
    scores = _hcc_scores(incidence_matrix, infection_state,
                         s_max=s_max, alpha=alpha)
    candidates = _infected_candidates(infection_state)
    ranked = sorted(candidates, key=lambda v: scores[v], reverse=True)
    return {"predicted_sources": ranked[:K], "node_scores": scores}


# =====================================================================
# 2. HBC — Higher-order Betweenness Centrality
# =====================================================================

def _hbc_scores(incidence_matrix: ArrayLike | sp.spmatrix,
                infection_state: ArrayLike,
                s_max: int = 3,
                beta: float = 1.0) -> np.ndarray:
    """
    Higher-order betweenness centrality on the infected sub-hypergraph.

    .. math::

        C_B^H(i) = \\sum_{s=1}^{s_{\\max}} \\beta^s
        \\sum_{q \\neq u} \\frac{\\sigma_{qu}^s(v_i)}{\\sigma_{qu}^s}

    where :math:`\\sigma_{qu}^s` is the number of shortest *s*-paths between
    nodes :math:`v_q` and :math:`v_u`, and :math:`\\sigma_{qu}^s(v_i)` counts
    those passing through :math:`v_i`.  This is computed via the standard
    betweenness centrality on the *s*-adjacency graph restricted to infected
    nodes.
    """
    N = _to_dense(incidence_matrix).shape[0]
    Y = np.asarray(infection_state).reshape(-1)
    infected = set(i for i in range(N) if Y[i] == 1)

    if not infected:
        return np.zeros(N)

    scores = np.zeros(N, dtype=float)

    for s in range(1, s_max + 1):
        G_s = _s_adjacency_graph(incidence_matrix, s=s, node_subset=infected)
        # NetworkX betweenness_centrality computes exactly
        # sum_{q!=u} sigma_{qu}(v) / sigma_{qu}, normalised by default.
        # We use normalized=False to get the raw count.
        bc = nx.betweenness_centrality(G_s, normalized=False)
        weight = beta ** s

        for v in infected:
            scores[v] += weight * bc.get(v, 0.0)

    return scores


def run_hbc(incidence_matrix: ArrayLike | sp.spmatrix,
            infection_state: ArrayLike,
            K: int,
            s_max: int = 3,
            beta: float = 1.0) -> dict:
    """
    HBC: rank infected nodes by higher-order betweenness centrality and
    return the top-K as predicted sources.

    Parameters
    ----------
    K : int
        Number of sources to return (must match the true source count).
    s_max : int
        Maximum s-order to consider.
    beta : float
        Weighting factor for different s-orders.
    """
    scores = _hbc_scores(incidence_matrix, infection_state,
                         s_max=s_max, beta=beta)
    candidates = _infected_candidates(infection_state)
    ranked = sorted(candidates, key=lambda v: scores[v], reverse=True)
    return {"predicted_sources": ranked[:K], "node_scores": scores}


# =====================================================================
# 3. HMCSM — Higher-order Monte Carlo Soft Boundary Estimation Method
# =====================================================================

def _default_simulate(sources: Set[int],
                      all_nodes: Set[int],
                      neighbors: Dict[int, Set[int]],
                      beta: float,
                      target_count: int) -> Set[int]:
    """
    A simplified SI-like simulation on the projected node graph.

    Each round, every currently-infected node attempts to infect each
    susceptible neighbour independently with probability *beta*.
    The process runs until the infected set reaches *target_count* or
    no new infections occur.

    In the full experimental pipeline, HMCSM supports multiple propagation
    models (SI, SI-Gillespie, IC, LT).  This default implementation provides
    a representative SI variant; users may supply their own ``simulate_fn``
    via :func:`run_hmcsm`.
    """
    infected = set(sources)
    susceptible = all_nodes - infected

    while len(infected) < target_count:
        new = set()
        for u in list(infected):
            for v in neighbors.get(u, set()):
                if v in susceptible and random.random() < beta:
                    new.add(v)
                    if len(infected) + len(new) >= target_count:
                        infected.update(new)
                        return infected
        if not new:
            break
        infected.update(new)
        susceptible -= new
    return infected


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.sum(a & b)
    union = np.sum(a | b)
    return intersection / union if union > 0 else 0.0


def run_hmcsm(incidence_matrix: ArrayLike | sp.spmatrix,
              infection_state: ArrayLike,
              K: int,
              beta: float = 0.3,
              repeats: int = 3,
              simulate_fn: Optional[Callable] = None) -> dict:
    """
    HMCSM: model-adaptive multi-source greedy localisation.

    For each candidate infected node, simulate propagation from that node and
    compare the simulated infection pattern with the observed snapshot using
    Jaccard similarity.  The first source is the node with the highest average
    match score.  Remaining sources are selected greedily: at each step, the
    candidate that maximises the match score when added to the current source
    set is chosen.

    Parameters
    ----------
    simulate_fn : callable, optional
        ``simulate_fn(sources, all_nodes, neighbors, beta, target_count) -> set``
        Custom propagation simulator.  Defaults to a simplified SI model on
        the projected graph.
    """
    H = _to_dense(incidence_matrix)
    N = H.shape[0]
    Y = np.asarray(infection_state, dtype=np.int8).reshape(-1)
    infected_nodes = np.where(Y == 1)[0]

    if len(infected_nodes) == 0:
        return {"predicted_sources": [], "node_scores": np.zeros(N)}

    neighbors = get_node_neighbors(incidence_matrix)
    all_nodes = set(range(N))
    target_count = int(N * 0.3)
    sim_fn = simulate_fn or _default_simulate
    node_scores = np.zeros(N, dtype=float)

    # Phase 1: score every infected node as a single-source candidate.
    for v in infected_nodes:
        total = 0.0
        for _ in range(repeats):
            sim_infected = sim_fn({v}, all_nodes, neighbors, beta, target_count)
            sim_vec = np.zeros(N, dtype=np.int8)
            sim_vec[list(sim_infected)] = 1
            jac = _jaccard(Y, sim_vec)
            total += math.exp(-(jac - 1.0) ** 2)
        node_scores[v] = total / repeats

    first = infected_nodes[np.argmax(node_scores[infected_nodes])]
    found: List[int] = [int(first)]

    # Phase 2: greedy selection of remaining sources.
    for _ in range(1, K):
        best, best_score = None, -1.0
        remaining = [n for n in infected_nodes if n not in found]
        for v in remaining:
            current = found + [v]
            total = 0.0
            for _ in range(repeats):
                sim_infected = sim_fn(set(current), all_nodes, neighbors, beta, target_count)
                sim_vec = np.zeros(N, dtype=np.int8)
                sim_vec[list(sim_infected)] = 1
                jac = _jaccard(Y, sim_vec)
                total += math.exp(-(jac - 1.0) ** 2)
            avg = total / repeats
            if avg > best_score:
                best_score = avg
                best = v
        if best is not None:
            found.append(int(best))
        else:
            break

    # Fill up if fewer than K sources found.
    if len(found) < K:
        for idx in np.argsort(node_scores)[::-1]:
            if Y[idx] == 1 and idx not in found:
                found.append(int(idx))
            if len(found) == K:
                break

    return {"predicted_sources": found, "node_scores": node_scores}


# =====================================================================
# 4. SLBIC — Source Localisation Based on Infection Cluster
# =====================================================================

def _slbne(G: nx.Graph,
           infected: List[int],
           alpha: float = 4.0) -> tuple:
    """
    Stage 1 (SLBNE): compute neighbourhood entropy for each infected node
    and extract candidate sources as local maxima of NE.
    """
    V_I = set(infected)
    xi, IE, eta, AE, NE = {}, {}, {}, {}, {}

    for i in V_I:
        nbrs = list(G.neighbors(i))
        n_i = len(nbrs)
        if n_i == 0:
            xi[i], IE[i] = 0.0, 0.0
            continue
        u_i = sum(1 for n in nbrs if n not in V_I)
        xi_val = ((n_i - u_i) / n_i) * (1 / (1 + math.exp(-n_i)))
        xi[i] = xi_val
        IE[i] = -xi_val * math.log2(xi_val) if xi_val > 0 else 0.0

    for j in V_I:
        total = 0.0
        for t in G.neighbors(j):
            if t in V_I:
                deg_t = G.degree(t)
                if deg_t > 0:
                    total += (1.0 / deg_t) * xi[t]
        eta[j] = total

    for i in V_I:
        nbrs = list(G.neighbors(i))
        n_i = len(nbrs)
        ae = 0.0
        for j in nbrs:
            if j in V_I and eta.get(j, 0) > 0 and n_i > 0:
                psi = (1.0 / n_i) / eta[j]
                if psi > 0:
                    ae -= psi * math.log2(psi)
        AE[i] = ae
        NE[i] = ae - alpha * IE[i]

    C_s: Set[int] = set()
    for i in V_I:
        nbr_ne = [NE[j] for j in G.neighbors(i) if j in V_I]
        if not nbr_ne or NE[i] > max(nbr_ne):
            C_s.add(i)

    return C_s, NE


def _slbic_stage2(G: nx.Graph,
                  infected: List[int],
                  C_s: Set[int],
                  NE: Dict[int, float]) -> Set[int]:
    """
    Stage 2 (SLBIC): partition the infected subgraph into clusters around
    candidates from SLBNE, then refine each cluster's representative by
    cohesion score.
    """
    if not C_s:
        return set()

    G_I = G.subgraph(infected)

    # Infection Community Division (ICD)
    clusters: Dict[int, Set[int]] = {c: {c} for c in C_s}
    assigned = set(C_s)

    one_hop = set()
    for c in C_s:
        for n in G_I.neighbors(c):
            if n not in assigned:
                one_hop.add(n)

    def _common_nbr_ratio(u: int, v: int) -> float:
        nu = set(G_I.neighbors(u))
        nv = set(G_I.neighbors(v))
        union = nu | nv
        return len(nu & nv) / len(union) if union else 0.0

    def _similarity(n1: int, n2: int) -> float:
        direct = _common_nbr_ratio(n1, n2)
        indirect = sum(_common_nbr_ratio(i, j)
                       for i in G_I.neighbors(n1) for j in G_I.neighbors(n2))
        return direct + indirect

    for v in one_hop:
        best_c = max(C_s, key=lambda c: _similarity(c, v))
        clusters[best_c].add(v)
        assigned.add(v)

    for v in set(G_I.nodes()) - assigned:
        best_c, mx = None, -1
        for c, nodes in clusters.items():
            e = sum(1 for u in nodes if G_I.has_edge(v, u))
            if e > mx:
                mx, best_c = e, c
        if best_c is not None:
            clusters[best_c].add(v)

    # Cohesion-based refinement
    C_sl: Set[int] = set()
    for _, cluster_nodes in clusters.items():
        G_cl = G_I.subgraph(cluster_nodes)
        best_node, best_coh = None, -float("inf")
        for v in cluster_nodes:
            paths = nx.single_source_shortest_path_length(G_cl, v)
            ne_v = NE.get(v, 0.0)
            if len(paths) <= 1:
                coh = ne_v * 0.5
            else:
                M = max(paths.values())
                counts: Dict[int, int] = {}
                for d in paths.values():
                    counts[d] = counts.get(d, 0) + 1
                apn = len(counts)
                part1 = sum(((M + 1 - d) / (M + 1)) * (c / apn)
                            for d, c in counts.items())
                coh = part1 * 0.5 + ne_v * 0.5
            if coh > best_coh:
                best_coh, best_node = coh, v
        if best_node is not None:
            C_sl.add(best_node)

    return C_s | C_sl


def run_slbic(incidence_matrix: ArrayLike | sp.spmatrix,
              infection_state: ArrayLike,
              alpha: float = 4.0) -> dict:
    """
    SLBIC: two-stage source localisation via neighbourhood entropy (SLBNE)
    followed by infection community division and cohesion refinement (SLBIC).

    The number of predicted sources is determined automatically by the
    algorithm (not a fixed K).
    """
    H = _to_dense(incidence_matrix)
    N = H.shape[0]
    Y = np.asarray(infection_state).reshape(-1)
    infected = [i for i in range(N) if Y[i] == 1]

    if not infected:
        return {"predicted_sources": [], "node_scores": np.zeros(N)}

    G = _projected_networkx_graph(incidence_matrix)
    C_s, NE = _slbne(G, infected, alpha=alpha)
    predicted = _slbic_stage2(G, infected, C_s, NE)

    scores = np.zeros(N)
    for i in range(N):
        scores[i] = NE.get(i, 0.0)

    return {"predicted_sources": sorted(predicted), "node_scores": scores}


# =====================================================================
# 5. LPSI-1 — Label Propagation based Source Identification (node-only)
# =====================================================================

def _lpsi_node_propagation(incidence_matrix: ArrayLike | sp.spmatrix,
                           infection_state: ArrayLike,
                           alpha: float = 0.5) -> np.ndarray:
    """
    Node-level label propagation on the projected graph.

    Steady-state: F = (1 - alpha) (I - alpha P)^{-1} Y,
    where P = D^{-1/2} A D^{-1/2}.
    """
    A = build_projected_adjacency(incidence_matrix, binary=True)
    P = _normalize_symmetric(A)
    Y = np.asarray(infection_state, dtype=float).reshape(-1)
    I = np.eye(P.shape[0])
    return (1 - alpha) * np.linalg.solve(I - alpha * P, Y)


def _select_local_maxima(incidence_matrix: ArrayLike | sp.spmatrix,
                         scores: np.ndarray,
                         infection_state: ArrayLike) -> List[int]:
    """Select infected nodes that are strict local maxima in the projected graph."""
    nbrs = get_node_neighbors(incidence_matrix)
    Y = np.asarray(infection_state).reshape(-1)
    peaks: List[int] = []
    for v in range(len(Y)):
        if Y[v] != 1:
            continue
        if all(scores[v] > scores[u] for u in nbrs[v]):
            peaks.append(v)
    return sorted(peaks, key=lambda v: scores[v], reverse=True)


def run_lpsi1(incidence_matrix: ArrayLike | sp.spmatrix,
              infection_state: ArrayLike,
              alpha: float = 0.5) -> dict:
    """
    LPSI-1: node-level label propagation on the projected graph with
    local-maxima source selection.
    """
    scores = _lpsi_node_propagation(incidence_matrix, infection_state, alpha=alpha)
    predicted = _select_local_maxima(incidence_matrix, scores, infection_state)
    return {"predicted_sources": predicted, "node_scores": scores}


# =====================================================================
# 6. LPSI-2 — Dual pairwise channels via the line graph
# =====================================================================

def _build_line_graph_propagation(A: np.ndarray) -> tuple:
    """
    Build the line graph L(G) from the projected adjacency matrix A.

    Each edge (i, j) with i < j in G becomes a node in L(G).
    Two L(G) nodes are adjacent iff the corresponding edges share an endpoint.

    Returns
    -------
    edge_list : list of (int, int)
        Edges of G, defining the node ordering in L(G).
    P_L : ndarray
        Normalised adjacency matrix of L(G).
    """
    edges = []
    N = A.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] > 0:
                edges.append((i, j))

    M = len(edges)
    if M == 0:
        return edges, np.zeros((0, 0))

    # Build adjacency of L(G)
    node_to_edges: Dict[int, List[int]] = defaultdict(list)
    for idx, (u, v) in enumerate(edges):
        node_to_edges[u].append(idx)
        node_to_edges[v].append(idx)

    W = np.zeros((M, M), dtype=float)
    for indices in node_to_edges.values():
        for a in indices:
            for b in indices:
                if a != b:
                    W[a, b] = 1.0

    row_sum = W.sum(axis=1, keepdims=True)
    P_L = np.divide(W, row_sum, out=np.zeros_like(W), where=row_sum != 0)
    return edges, P_L


def _edge_initial_labels(edges: List[tuple],
                         infection_state: ArrayLike) -> np.ndarray:
    """Initial label for each edge = mean infection state of its two endpoints."""
    Y = np.asarray(infection_state, dtype=float).reshape(-1)
    return np.array([(Y[u] + Y[v]) / 2.0 for u, v in edges])


def _map_edge_scores_to_nodes(edges: List[tuple],
                              edge_scores: np.ndarray,
                              num_nodes: int) -> np.ndarray:
    """Average edge scores back to nodes."""
    totals = np.zeros(num_nodes)
    counts = np.zeros(num_nodes)
    for idx, (u, v) in enumerate(edges):
        totals[u] += edge_scores[idx]
        totals[v] += edge_scores[idx]
        counts[u] += 1
        counts[v] += 1
    return np.divide(totals, counts, out=np.zeros(num_nodes, dtype=float), where=counts != 0)


def run_lpsi2(incidence_matrix: ArrayLike | sp.spmatrix,
              infection_state: ArrayLike,
              rho: float = 0.5,
              mu: float = 0.5,
              alpha: float = 0.5) -> dict:
    """
    LPSI-2: dual pairwise-channel control.

    Channel 1 — node-level LP on the projected graph G  (same as LPSI-1).
    Channel 2 — edge-level LP on the line graph L(G) of the projected graph.

    The two channels are fused as:
        C_i = alpha * node_score_i + (1 - alpha) * mean edge score of i's edges.

    Source selection uses the same local-maxima rule as LPSI-1 / HLP-L.
    """
    # Node channel
    A = build_projected_adjacency(incidence_matrix, binary=True)
    P_G = _normalize_symmetric(A)
    Y = np.asarray(infection_state, dtype=float).reshape(-1)
    N = len(Y)
    I_n = np.eye(N)
    node_scores = (1 - mu) * np.linalg.solve(I_n - mu * P_G, Y)

    # Edge channel via line graph
    edges, P_L = _build_line_graph_propagation(A)
    if len(edges) > 0:
        B0 = _edge_initial_labels(edges, infection_state)
        I_e = np.eye(P_L.shape[0])
        edge_scores = (1 - rho) * np.linalg.solve(I_e - rho * P_L, B0)
        edge_to_node = _map_edge_scores_to_nodes(edges, edge_scores, N)
    else:
        edge_scores = np.array([])
        edge_to_node = np.zeros(N)

    C = alpha * node_scores + (1 - alpha) * edge_to_node
    predicted = _select_local_maxima(incidence_matrix, C, infection_state)

    return {
        "predicted_sources": predicted,
        "node_scores": C,
        "node_channel": node_scores,
        "edge_channel": edge_to_node,
    }


# =====================================================================
# 7. GCNSI — Graph Convolutional Networks based Source Identification
# =====================================================================

def _require_torch():
    """Lazy import of PyTorch and PyG; raises a clear error if missing."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv
        from torch_geometric.utils import from_scipy_sparse_matrix
        from torch_geometric.data import Data
        return torch, nn, F, GCNConv, from_scipy_sparse_matrix, Data
    except ImportError as e:
        raise ImportError(
            "GCNSI requires PyTorch and PyTorch Geometric. "
            "Install them with: pip install torch torch_geometric"
        ) from e


def generate_gcnsi_features(infection_state: ArrayLike,
                            adj_csr: sp.spmatrix,
                            alpha: float = 0.5) -> np.ndarray:
    """
    Build the 4-dimensional node feature matrix used by GCNSI.

    d1 = Y  (raw infection state)
    d2 = (1 - alpha) (I - alpha S)^{-1} Y
    d3 = (1 - alpha) (I - alpha S)^{-1} Y+   (Y+ = max(Y, 0))
    d4 = (1 - alpha) (I - alpha S)^{-1} Y-   (Y- = min(Y, 0))

    where S = D^{-1/2} A D^{-1/2}.
    """
    from scipy.sparse import diags, eye
    from scipy.sparse.linalg import spsolve

    Y = np.asarray(infection_state, dtype=float).reshape(-1)
    N = len(Y)

    W = adj_csr.tocsc()
    degrees = np.maximum(np.array(W.sum(axis=1)).flatten(), 1e-10)
    D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
    S = (D_inv_sqrt @ W @ D_inv_sqrt).tocsc()

    I = eye(N, format="csc")
    A_mat = (I - alpha * S).tocsc()

    d1 = Y.copy()
    d2 = (1 - alpha) * spsolve(A_mat, Y)
    Y_pos = np.maximum(Y, 0).astype(float)
    d3 = (1 - alpha) * spsolve(A_mat, Y_pos)
    Y_neg = np.minimum(Y, 0).astype(float)
    d4 = (1 - alpha) * spsolve(A_mat, Y_neg)

    return np.stack([d1, d2, d3, d4], axis=1).astype(np.float32)


class GCNSIModel:
    """
    Wrapper around the GCNSI two-layer GCN for source identification.

    The model is a node-level binary classifier: for each node it outputs a
    probability of being a source.  Training and inference operate on
    PyG ``Data`` objects built from the incidence matrix and infection state.

    Example
    -------
    >>> model = GCNSIModel(hidden_channels=32)
    >>> model.train_model(train_incidence_list, train_infection_list,
    ...                   train_labels_list, num_epochs=100)
    >>> result = model.predict(incidence_matrix, infection_state, K=2)
    """

    def __init__(self, hidden_channels: int = 32, dropout: float = 0.3,
                 lr: float = 1e-5, weight_decay: float = 5e-4,
                 feature_alpha: float = 0.5):
        torch, nn, F, GCNConv, _, _ = _require_torch()
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.feature_alpha = feature_alpha
        self._torch = torch
        self._nn = nn
        self._F = F
        self._GCNConv = GCNConv
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._build_model().to(self.device)

    def _build_model(self):
        torch, nn, F, GCNConv = self._torch, self._nn, self._F, self._GCNConv

        class _Net(nn.Module):
            def __init__(inner, in_ch, hid_ch, drop):
                super().__init__()
                inner.conv1 = GCNConv(in_ch, hid_ch)
                inner.conv2 = GCNConv(hid_ch, hid_ch)
                inner.fc = nn.Linear(hid_ch, 1)
                inner.drop = drop

            def forward(inner, x, edge_index):
                x = F.relu(inner.conv1(x, edge_index))
                x = F.dropout(x, p=inner.drop, training=inner.training)
                x = F.relu(inner.conv2(x, edge_index))
                return torch.sigmoid(inner.fc(x)).squeeze(-1)

        return _Net(4, self.hidden_channels, self.dropout)

    def _to_pyg(self, incidence_matrix, infection_state, labels=None):
        torch = self._torch
        _, _, _, _, from_scipy, Data = _require_torch()

        H = _to_dense(incidence_matrix)
        A = H @ H.T
        np.fill_diagonal(A, 0.0)
        A[A > 0] = 1.0
        adj_csr = sp.csr_matrix(A)

        Y = np.asarray(infection_state, dtype=float).reshape(-1)
        X = generate_gcnsi_features(Y, adj_csr, alpha=self.feature_alpha)

        edge_index, _ = from_scipy(adj_csr)
        if edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()

        data = Data(x=torch.from_numpy(X), edge_index=edge_index)
        if labels is not None:
            data.y = torch.from_numpy(np.asarray(labels, dtype=np.float32))
        return data

    def train_model(self,
                    incidence_list: list,
                    infection_list: list,
                    labels_list: list,
                    num_epochs: int = 100,
                    batch_size: int = 8):
        """
        Train the GCN on a list of (incidence_matrix, infection_state, label_vector) triples.

        Each label_vector has 1 for source nodes and 0 otherwise.
        """
        torch = self._torch
        from torch_geometric.loader import DataLoader as PyGLoader

        graphs = [self._to_pyg(H, Y, lbl)
                  for H, Y, lbl in zip(incidence_list, infection_list, labels_list)]
        loader = PyGLoader(graphs, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        criterion = self._nn.BCELoss()

        self._model.train()
        for _ in range(num_epochs):
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = self._model(batch.x, batch.edge_index)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()

    def predict(self, incidence_matrix, infection_state, K: int) -> dict:
        """Run inference and return the top-K predicted sources."""
        torch = self._torch
        data = self._to_pyg(incidence_matrix, infection_state).to(self.device)

        self._model.eval()
        with torch.no_grad():
            scores = self._model(data.x, data.edge_index).cpu().numpy()

        predicted = np.argsort(scores)[-K:].tolist()
        return {"predicted_sources": predicted, "node_scores": scores}
