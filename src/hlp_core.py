"""
Core implementation of HLP: Hybrid Label Propagation for source localization on hypergraphs.

This file only contains the main algorithmic workflow:
1) construction of initial hyperedge labels B0,
2) construction of initial node labels J0,
3) hyperedge-level label propagation,
4) node-level label propagation,
5) hybrid score fusion,
6) HLP-G / HLP-L source selection.

Input convention
----------------
incidence_matrix: shape (num_nodes, num_hyperedges), where H[i, e] = 1 if node i belongs to hyperedge e.
infection_state: length num_nodes, where 1 denotes infected and -1 denotes susceptible.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Literal, Sequence, Set

import numpy as np
import scipy.sparse as sp


ArrayLike = Sequence[float] | np.ndarray
Strategy = Literal["HLP-G", "HLP-L"]


def _to_dense_incidence(incidence_matrix: ArrayLike | sp.spmatrix) -> np.ndarray:
    """Convert a dense/sparse incidence matrix to a NumPy array."""
    if sp.issparse(incidence_matrix):
        return incidence_matrix.toarray().astype(float)
    return np.asarray(incidence_matrix, dtype=float)


def build_hyperedge_labels(incidence_matrix: ArrayLike | sp.spmatrix,
                           infection_state: ArrayLike) -> np.ndarray:
    """
    Construct initial hyperedge labels B0.

    For each hyperedge e, the label is the average infection state of its incident nodes:
        B0_e = sum_{v in e} Y_v / |e|.

    Here Y_v = 1 for infected nodes and Y_v = -1 for susceptible nodes.
    """
    H = _to_dense_incidence(incidence_matrix)
    Y = np.asarray(infection_state, dtype=float).reshape(-1)

    if H.shape[0] != len(Y):
        raise ValueError("infection_state length must equal the number of nodes.")

    hyperedge_sizes = H.sum(axis=0)
    label_sum = H.T @ Y
    B0 = np.divide(label_sum, hyperedge_sizes, out=np.zeros_like(label_sum), where=hyperedge_sizes != 0)
    return B0


def build_hyperedge_propagation_matrix(incidence_matrix: ArrayLike | sp.spmatrix) -> np.ndarray:
    """
    Build the hyperedge-level propagation matrix based on overlapping hyperedges.

    The unnormalized coupling between two hyperedges is their overlap size:
        W_ab = |e_a ∩ e_b|.

    The diagonal is removed, and each row is normalized so that hyperedge scores
    are propagated among overlapping hyperedges.
    """
    H = _to_dense_incidence(incidence_matrix)
    W = H.T @ H
    np.fill_diagonal(W, 0.0)

    row_sum = W.sum(axis=1, keepdims=True)
    P_L = np.divide(W, row_sum, out=np.zeros_like(W, dtype=float), where=row_sum != 0)
    return P_L


def propagate_hyperedge_labels(P_L: np.ndarray,
                               B0: ArrayLike,
                               rho: float = 0.5) -> np.ndarray:
    """
    Hyperedge-level label propagation.

    Closed-form steady-state solution:
        B* = (1 - rho) (I - rho P_L)^(-1) B0.
    """
    if not 0 <= rho < 1:
        raise ValueError("rho must be in [0, 1).")

    P_L = np.asarray(P_L, dtype=float)
    B0 = np.asarray(B0, dtype=float).reshape(-1)

    I = np.eye(P_L.shape[0])
    return (1 - rho) * np.linalg.solve(I - rho * P_L, B0)


def build_node_labels(infection_state: ArrayLike) -> np.ndarray:
    """
    Construct initial node labels J0.

    In HLP, the node-level initial labels are directly given by the observed
    infection snapshot: 1 for infected nodes and -1 for susceptible nodes.
    """
    return np.asarray(infection_state, dtype=float).reshape(-1)


def build_projected_node_adjacency(incidence_matrix: ArrayLike | sp.spmatrix,
                                   binary: bool = True) -> np.ndarray:
    """
    Project the hypergraph to a node graph.

    Two nodes are adjacent if they appear in at least one common hyperedge.
    When binary=True, repeated co-occurrences are binarized.
    """
    H = _to_dense_incidence(incidence_matrix)
    A = H @ H.T
    np.fill_diagonal(A, 0.0)
    if binary:
        A[A > 0] = 1.0
    return A


def normalize_adjacency_symmetric(A: ArrayLike) -> np.ndarray:
    """
    Symmetric normalization of the projected node adjacency matrix:
        P_G = D^(-1/2) A D^(-1/2).
    """
    A = np.asarray(A, dtype=float)
    degree = A.sum(axis=1)
    d_inv_sqrt = np.zeros_like(degree, dtype=float)
    nonzero = degree > 0
    d_inv_sqrt[nonzero] = np.power(degree[nonzero], -0.5)
    return d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]


def propagate_node_labels(P_G: np.ndarray,
                          J0: ArrayLike,
                          mu: float = 0.5) -> np.ndarray:
    """
    Node-level label propagation.

    Closed-form steady-state solution:
        J* = (1 - mu) (I - mu P_G)^(-1) J0.
    """
    if not 0 <= mu < 1:
        raise ValueError("mu must be in [0, 1).")

    P_G = np.asarray(P_G, dtype=float)
    J0 = np.asarray(J0, dtype=float).reshape(-1)

    I = np.eye(P_G.shape[0])
    return (1 - mu) * np.linalg.solve(I - mu * P_G, J0)


def map_hyperedge_scores_to_nodes(incidence_matrix: ArrayLike | sp.spmatrix,
                                  hyperedge_scores: ArrayLike) -> np.ndarray:
    """
    Project propagated hyperedge scores back to nodes by averaging the scores of
    each node's incident hyperedges.
    """
    H = _to_dense_incidence(incidence_matrix)
    B = np.asarray(hyperedge_scores, dtype=float).reshape(-1)

    if H.shape[1] != len(B):
        raise ValueError("hyperedge_scores length must equal the number of hyperedges.")

    incident_count = H.sum(axis=1)
    score_sum = H @ B
    return np.divide(score_sum, incident_count, out=np.zeros_like(score_sum), where=incident_count != 0)


def fuse_scores(incidence_matrix: ArrayLike | sp.spmatrix,
                hyperedge_scores: ArrayLike,
                node_scores: ArrayLike,
                alpha: float = 0.5) -> np.ndarray:
    """
    Fuse node-level and hyperedge-level evidence into final node scores C.

        C_i = alpha * J*_i + (1 - alpha) * mean_{e incident to i} B*_e.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1].")

    J = np.asarray(node_scores, dtype=float).reshape(-1)
    B_to_V = map_hyperedge_scores_to_nodes(incidence_matrix, hyperedge_scores)

    if len(J) != len(B_to_V):
        raise ValueError("node_scores length must equal the number of nodes.")

    return alpha * J + (1 - alpha) * B_to_V


def get_node_neighbors(incidence_matrix: ArrayLike | sp.spmatrix) -> Dict[int, Set[int]]:
    """Return neighbors of each node in the projected node graph."""
    A = build_projected_node_adjacency(incidence_matrix, binary=True)
    return {i: set(np.flatnonzero(A[i])) for i in range(A.shape[0])}


def _infected_candidates(infection_state: ArrayLike) -> List[int]:
    Y = np.asarray(infection_state).reshape(-1)
    return [i for i, y in enumerate(Y) if y == 1]


def select_sources_hlp_g(final_scores: ArrayLike,
                         infection_state: ArrayLike,
                         K: int) -> List[int]:
    """
    HLP-G: select the global top-K infected nodes according to final score C.
    """
    scores = np.asarray(final_scores, dtype=float).reshape(-1)
    candidates = _infected_candidates(infection_state)
    ranked = sorted(candidates, key=lambda node: scores[node], reverse=True)
    return ranked[:K]


def select_sources_hlp_l(incidence_matrix: ArrayLike | sp.spmatrix,
                         final_scores: ArrayLike,
                         infection_state: ArrayLike,
                         K: int | None = None) -> List[int]:
    """
    HLP-L: select infected nodes that are strict local maxima in the projected graph.

    A node v is selected as a local peak if:
        C_v > C_u for every projected-graph neighbor u of v,
    and v is infected in the observed snapshot.

    If K is provided and the number of local peaks exceeds K, only the top-K peaks
    by C score are returned. If fewer than K peaks exist, all available peaks are returned.
    """
    scores = np.asarray(final_scores, dtype=float).reshape(-1)
    infected = set(_infected_candidates(infection_state))
    neighbors = get_node_neighbors(incidence_matrix)

    local_peaks: List[int] = []
    for node in infected:
        if all(scores[node] > scores[neighbor] for neighbor in neighbors[node]):
            local_peaks.append(node)

    local_peaks = sorted(local_peaks, key=lambda node: scores[node], reverse=True)
    if K is not None:
        return local_peaks[:K]
    return local_peaks


def run_hlp(incidence_matrix: ArrayLike | sp.spmatrix,
            infection_state: ArrayLike,
            K: int,
            rho: float = 0.5,
            mu: float = 0.5,
            alpha: float = 0.5,
            strategy: Strategy = "HLP-G") -> dict:
    """
    Run the complete HLP workflow on one hypergraph and one infection snapshot.
    """
    B0 = build_hyperedge_labels(incidence_matrix, infection_state)
    P_L = build_hyperedge_propagation_matrix(incidence_matrix)
    B_star = propagate_hyperedge_labels(P_L, B0, rho=rho)

    J0 = build_node_labels(infection_state)
    A = build_projected_node_adjacency(incidence_matrix)
    P_G = normalize_adjacency_symmetric(A)
    J_star = propagate_node_labels(P_G, J0, mu=mu)

    C = fuse_scores(incidence_matrix, B_star, J_star, alpha=alpha)

    if strategy == "HLP-G":
        predicted_sources = select_sources_hlp_g(C, infection_state, K)
    elif strategy == "HLP-L":
        predicted_sources = select_sources_hlp_l(incidence_matrix, C, infection_state, K=K)
    else:
        raise ValueError("strategy must be either 'HLP-G' or 'HLP-L'.")

    return {
        "B0": B0,
        "B_star": B_star,
        "J0": J0,
        "J_star": J_star,
        "C": C,
        "predicted_sources": predicted_sources,
    }
