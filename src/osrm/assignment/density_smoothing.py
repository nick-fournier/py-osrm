"""Neighbor-based density smoothing for mesoscopic traffic assignment.

Smooths per-link density using topological neighbors (links sharing a node),
weighted by link length. This approximates local spillback effects without
requiring zone definitions.

See docs/traffic_assignment_design.md §3.8 for theory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class DensitySmoothingConfig:
    """Configuration for density smoothing.

    Attributes
    ----------
    method : str
        Smoothing method: "neighbor" (default) or "none".
    beta : float
        Smoothing parameter in [0, 0.5]. Higher = more smoothing.
    passes : int
        Number of smoothing passes. Multi-pass widens spatial reach.
    """

    method: str = "neighbor"
    beta: float = 0.3
    passes: int = 1


class DensitySmoothing:
    """Neighbor-based density smoothing using sparse adjacency matrix.

    The smoothing kernel is:
        k̃_e = (1 - β) * k_e + β * Σ_{j ∈ N(e)} w_ej * k_j

    where N(e) are topological neighbors (edges sharing a node) and
    w_ej are length-proportional normalized weights.
    """

    def __init__(self, config: DensitySmoothingConfig | None = None) -> None:
        self.config = config or DensitySmoothingConfig()
        self._W: sparse.csr_matrix | None = None

    def build_adjacency(
        self,
        edge_ids: np.ndarray,
        lengths: np.ndarray,
    ) -> None:
        """Build the sparse adjacency matrix from edge topology.

        Two edges are neighbors if they share a node (either endpoint).
        Weights are proportional to the neighbor's length, row-normalized.

        Parameters
        ----------
        edge_ids : np.ndarray
            (N, 2) uint64 — (from_osm_id, to_osm_id) per edge.
        lengths : np.ndarray
            (N,) float64 — link lengths in meters.
        """
        n = len(edge_ids)
        if n == 0:
            self._W = sparse.csr_matrix((0, 0), dtype=np.float64)
            self._has_neighbors = np.empty(0, dtype=bool)
            return

        # Build node → edge_ordinals mapping
        node_to_edges: dict[int, list[int]] = {}
        for i in range(n):
            for node in (int(edge_ids[i, 0]), int(edge_ids[i, 1])):
                node_to_edges.setdefault(node, []).append(i)

        # Build adjacency: edges sharing a node are neighbors
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for edge_list in node_to_edges.values():
            for i in edge_list:
                for j in edge_list:
                    if i != j:
                        rows.append(i)
                        cols.append(j)
                        vals.append(lengths[j])

        if not rows:
            self._W = sparse.csr_matrix((n, n), dtype=np.float64)
            self._has_neighbors = np.zeros(n, dtype=bool)
            return

        W = sparse.coo_matrix(
            (vals, (rows, cols)), shape=(n, n), dtype=np.float64
        )
        # Combine duplicates (same pair may appear via multiple shared nodes)
        W = W.tocsr()

        # Row-normalize
        row_sums = np.array(W.sum(axis=1)).flatten()
        # Track which edges actually have neighbors
        self._has_neighbors = row_sums > 0
        row_sums[row_sums == 0] = 1.0
        inv_sums = sparse.diags(1.0 / row_sums)
        self._W = inv_sums @ W

    def smooth(
        self,
        density: np.ndarray,
    ) -> np.ndarray:
        """Apply density smoothing.

        Parameters
        ----------
        density : np.ndarray
            (N,) float64 — per-edge density (veh/km).

        Returns
        -------
        np.ndarray
            Smoothed density, same shape.
        """
        if self.config.method == "none" or self._W is None:
            return density.copy()

        beta = self.config.beta
        k = density.copy()
        mask = self._has_neighbors
        for _ in range(self.config.passes):
            neighbor_avg = self._W.dot(k)
            k[mask] = (1.0 - beta) * k[mask] + beta * neighbor_avg[mask]
        return k
