"""Bi-parabolic flow-density Volume Delay Function (Fournier et al.).

Implements a macroscopic fundamental diagram (MFD) with two parabolic
branches in q-k (flow-density) space, joined with C¹ continuity at
critical density k_c. The model requires only free-flow speed (v_f) and
jam density (k_j) per link — k_c is derived as k_j / 3.

See docs/traffic_assignment_design.md §3 for full derivation.
"""

from __future__ import annotations

import numpy as np


class BiParabolicVDF:
    """Vectorized bi-parabolic VDF operating on NumPy arrays.

    Parameters
    ----------
    kc_ratio : float
        Ratio k_c / k_j. Default 1/3 per Fournier et al.
    min_speed_kmh : float
        Floor speed for oversaturated links (avoids zero/negative speeds).
    """

    def __init__(
        self,
        kc_ratio: float = 1.0 / 3.0,
        min_speed_kmh: float = 0.01,
    ) -> None:
        self.kc_ratio = kc_ratio
        self.min_speed_kmh = min_speed_kmh

    def critical_density(
        self,
        k_j: np.ndarray,
        kc_ratio: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """k_c = kc_ratio * k_j"""
        r = kc_ratio if kc_ratio is not None else self.kc_ratio
        return r * k_j

    def capacity_flow(
        self,
        v_f: np.ndarray,
        k_j: np.ndarray,
        kc_ratio: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """q_c = v_f * k_c / 2"""
        k_c = self.critical_density(k_j, kc_ratio)
        return v_f * k_c / 2.0

    def density_to_speed(
        self,
        k: np.ndarray,
        v_f: np.ndarray,
        k_j: np.ndarray,
        kc_ratio: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Evaluate VDF: density → speed for both branches.

        Uncongested (k ≤ k_c): v(k) = q_c * (2*k_c - k) / k_c²
        Congested (k > k_c):   v(k) = q_c * [1 - (k - k_c)² / (k_j - k_c)²] / k

        Parameters
        ----------
        k : array-like
            Density (veh/km), same shape as v_f and k_j.
        v_f : array-like
            Free-flow speed (km/h).
        k_j : array-like
            Jam density (veh/km).
        kc_ratio : array-like or float, optional
            Per-link k_c/k_j ratio.  Falls back to ``self.kc_ratio``.

        Returns
        -------
        np.ndarray
            Speed (km/h), clipped to [min_speed_kmh, v_f].
        """
        k = np.asarray(k, dtype=np.float64)
        v_f = np.asarray(v_f, dtype=np.float64)
        k_j = np.asarray(k_j, dtype=np.float64)

        k_c = self.critical_density(k_j, kc_ratio)
        q_c = v_f * k_c / 2.0

        # Uncongested branch: linear speed-density
        v_uncongested = q_c * (2.0 * k_c - k) / k_c**2

        # Congested branch: parabolic in q-k, divided by k
        denom_k = np.where(k > 0, k, 1.0)
        denom_kj = np.where(k_j > k_c, (k_j - k_c) ** 2, 1.0)
        v_congested = q_c * (1.0 - (k - k_c) ** 2 / denom_kj) / denom_k

        v = np.where(k <= k_c, v_uncongested, v_congested)
        return np.clip(v, self.min_speed_kmh, v_f)

    def flow_to_density(
        self,
        q: np.ndarray,
        v_f: np.ndarray,
        k_j: np.ndarray,
        kc_ratio: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Closed-form flow → density inversion (uncongested branch only).

        k(q) = k_c * (1 − √(1 − q/q_c))

        Flows exceeding q_c are clamped to q_c (returns k_c). This is
        the analytic inverse of the uncongested parabolic branch. For
        assignment use ``demand_to_density`` which handles all q ≥ 0.

        Parameters
        ----------
        q : array-like
            Flow (veh/hr).
        v_f : array-like
            Free-flow speed (km/h).
        k_j : array-like
            Jam density (veh/km).
        kc_ratio : array-like or float, optional
            Per-link k_c/k_j ratio.

        Returns
        -------
        np.ndarray
            Density (veh/km), capped at k_c.
        """
        q = np.asarray(q, dtype=np.float64)
        v_f = np.asarray(v_f, dtype=np.float64)
        k_j = np.asarray(k_j, dtype=np.float64)

        k_c = self.critical_density(k_j, kc_ratio)
        q_c = self.capacity_flow(v_f, k_j, kc_ratio)

        ratio = np.clip(q / np.where(q_c > 0, q_c, 1.0), 0.0, 1.0)
        return k_c * (1.0 - np.sqrt(1.0 - ratio))

    def flow_to_speed(
        self,
        q: np.ndarray,
        v_f: np.ndarray,
        k_j: np.ndarray,
        kc_ratio: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Full pipeline: flow → density → speed. Single pass, no iteration."""
        k = self.flow_to_density(q, v_f, k_j, kc_ratio)
        return self.density_to_speed(k, v_f, k_j, kc_ratio)

    def demand_to_density(
        self,
        q: np.ndarray,
        v_f: np.ndarray,
        k_j: np.ndarray,
        kc_ratio: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Extended q → k mapping for flow-based assignment.

        Two-piece monotone function defined for ALL q ≥ 0:

        q ≤ q_c:  k = k_c · (1 − √(1 − q/q_c))       [exact MFD inverse]
        q > q_c:  k = k_c + (k_j − k_c) · √(1 − q_c/q) [asymptotic extension]

        The first piece is the standard uncongested inverse. The second
        piece maps oversaturated demand (q > q_c) into the congested
        density range (k_c, k_j), approaching k_j asymptotically as
        q → ∞. The two pieces are C⁰ continuous at q = q_c (both
        yield k_c).

        See docs/traffic_assignment_design.md §3.7.3 for derivation.
        """
        q = np.asarray(q, dtype=np.float64)
        v_f = np.asarray(v_f, dtype=np.float64)
        k_j = np.asarray(k_j, dtype=np.float64)

        k_c = self.critical_density(k_j, kc_ratio)
        q_c = self.capacity_flow(v_f, k_j, kc_ratio)

        # Uncongested branch: exact inverse
        safe_qc = np.where(q_c > 0, q_c, 1.0)
        ratio = np.clip(q / safe_qc, 0.0, 1.0)
        k_under = k_c * (1.0 - np.sqrt(1.0 - ratio))

        # Oversaturated branch: asymptotic extension toward k_j
        safe_q = np.where(q > 0, q, 1.0)
        k_over = k_c + (k_j - k_c) * np.sqrt(np.maximum(1.0 - q_c / safe_q, 0.0))

        return np.where(q <= q_c, k_under, k_over)

    def demand_to_speed(
        self,
        q: np.ndarray,
        v_f: np.ndarray,
        k_j: np.ndarray,
        kc_ratio: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Full pipeline: demand → density (extended) → speed."""
        k = self.demand_to_density(q, v_f, k_j, kc_ratio)
        return self.density_to_speed(k, v_f, k_j, kc_ratio)

    def density_to_flow(
        self,
        k: np.ndarray,
        v_f: np.ndarray,
        k_j: np.ndarray,
        kc_ratio: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Evaluate q(k) for both branches.

        Uncongested: q(k) = q_c * k * (2*k_c - k) / k_c²
        Congested:   q(k) = q_c * [1 - (k - k_c)² / (k_j - k_c)²]
        """
        k = np.asarray(k, dtype=np.float64)
        v_f = np.asarray(v_f, dtype=np.float64)
        k_j = np.asarray(k_j, dtype=np.float64)

        k_c = self.critical_density(k_j, kc_ratio)
        q_c = v_f * k_c / 2.0

        q_uncongested = q_c * k * (2.0 * k_c - k) / k_c**2
        denom = np.where(k_j > k_c, (k_j - k_c) ** 2, 1.0)
        q_congested = q_c * (1.0 - (k - k_c) ** 2 / denom)

        q = np.where(k <= k_c, q_uncongested, q_congested)
        return np.clip(q, 0.0, None)
