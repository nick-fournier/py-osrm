"""Bi-parabolic flow-density Volume Delay Function (Fournier et al.).

Implements a macroscopic fundamental diagram (MFD) with two parabolic
branches in q-k (flow-density) space, joined with C¹ continuity at
critical density k_c. The model requires only free-flow speed (v_f) and
jam density (k_j) per link — k_c is derived as k_j / 3.

An optional exponential tail replaces the congested parabola for
k > k_s = splice_ratio × k_j, providing C¹-continuous decay that
eliminates the gradient singularity at k_j.

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
    splice_ratio : float or None
        If set, fraction of k_j at which to splice an exponential tail
        onto the congested branch. The tail matches value and slope
        (C¹ continuity) and decays asymptotically — no zero crossing.
        Recommended value: 0.85. Set to ``None`` to disable (pure
        bi-parabolic).
    """

    def __init__(
        self,
        kc_ratio: float = 1.0 / 3.0,
        min_speed_kmh: float = 5.0,
        splice_ratio: float | None = None,
    ) -> None:
        self.kc_ratio = kc_ratio
        self.min_speed_kmh = min_speed_kmh
        self.splice_ratio = splice_ratio

    def critical_density(
        self,
        k_j: np.ndarray,
    ) -> np.ndarray:
        """k_c = kc_ratio * k_j"""
        return self.kc_ratio * k_j

    def capacity_flow(
        self,
        v_f: np.ndarray,
        k_j: np.ndarray,
    ) -> np.ndarray:
        """q_c = v_f * k_c / 2"""
        k_c = self.critical_density(k_j)
        return v_f * k_c / 2.0

    def density_to_speed(
        self,
        k: np.ndarray,
        v_f: np.ndarray,
        k_j: np.ndarray,
    ) -> np.ndarray:
        """Evaluate VDF: density → speed.

        Three regions:
        1. Uncongested (k ≤ k_c): v(k) = q_c * (2*k_c - k) / k_c²
        2. Congested (k_c < k ≤ k_s): v(k) = q_c * [1 - (k-k_c)²/(k_j-k_c)²] / k
        3. Exponential tail (k > k_s): v(k) = v_s * exp(-B*(k - k_s))
           where v_s and B are derived from C¹ continuity at k_s.

        If ``splice_ratio`` is None, region 2 extends to all k > k_c
        (original bi-parabolic behaviour).

        Parameters
        ----------
        k : array-like
            Density (veh/km), same shape as v_f and k_j.
        v_f : array-like
            Free-flow speed (km/h).
        k_j : array-like
            Jam density (veh/km).

        Returns
        -------
        np.ndarray
            Speed (km/h), clipped to [min_speed_kmh, v_f].
        """
        k = np.asarray(k, dtype=np.float64)
        v_f = np.asarray(v_f, dtype=np.float64)
        k_j = np.asarray(k_j, dtype=np.float64)

        k_c = self.critical_density(k_j)
        q_c = v_f * k_c / 2.0

        # Uncongested branch
        v_uncongested = q_c * (2.0 * k_c - k) / k_c**2

        # Congested branch (parabolic)
        denom_k = np.where(k > 0, k, 1.0)
        R = np.where(k_j > k_c, (k_j - k_c) ** 2, 1.0)
        v_congested = q_c * (1.0 - (k - k_c) ** 2 / R) / denom_k

        v = np.where(k <= k_c, v_uncongested, v_congested)

        # Exponential tail: replace congested branch for k > k_s
        if self.splice_ratio is not None:
            k_s = self.splice_ratio * k_j
            tail_mask = k > k_s

            if np.any(tail_mask):
                # Value at splice point
                k_s_safe = np.where(k_s > 0, k_s, 1.0)
                f_s = 1.0 - (k_s - k_c) ** 2 / R
                v_s = q_c * f_s / k_s_safe

                # Derivative at splice point: dv/dk|_{k_s}
                fp_s = -2.0 * (k_s - k_c) / R
                dvdk_s = q_c * (fp_s * k_s - f_s) / k_s_safe**2

                # Exponential coefficients: v_tail = v_s * exp(-B*(k - k_s))
                # C¹ match: -v_s * B = dvdk_s  →  B = -dvdk_s / v_s
                v_s_safe = np.where(v_s > 0, v_s, 1e-10)
                B = -dvdk_s / v_s_safe
                B = np.maximum(B, 0.0)  # ensure decay (not growth)

                v_tail = v_s * np.exp(-B * (k - k_s))
                v = np.where(tail_mask, v_tail, v)

        return np.clip(v, self.min_speed_kmh, v_f)

    def flow_to_density(
        self,
        q: np.ndarray,
        v_f: np.ndarray,
        k_j: np.ndarray,
    ) -> np.ndarray:
        """Closed-form flow → density inversion (uncongested branch only).

        k(q) = k_c * (1 − √(1 − q/q_c))

        Flows exceeding q_c are clamped to q_c (returns k_c). This is
        the analytic inverse of the uncongested parabolic branch. For
        assignment, use k = q / v instead (see AssignmentLoop._update_state).

        Parameters
        ----------
        q : array-like
            Flow (veh/hr).
        v_f : array-like
            Free-flow speed (km/h).
        k_j : array-like
            Jam density (veh/km).

        Returns
        -------
        np.ndarray
            Density (veh/km), capped at k_c.
        """
        q = np.asarray(q, dtype=np.float64)
        v_f = np.asarray(v_f, dtype=np.float64)
        k_j = np.asarray(k_j, dtype=np.float64)

        k_c = self.critical_density(k_j)
        q_c = self.capacity_flow(v_f, k_j)

        ratio = np.clip(q / np.where(q_c > 0, q_c, 1.0), 0.0, 1.0)
        return k_c * (1.0 - np.sqrt(1.0 - ratio))

    def flow_to_speed(
        self,
        q: np.ndarray,
        v_f: np.ndarray,
        k_j: np.ndarray,
    ) -> np.ndarray:
        """Full pipeline: flow → density → speed. Single pass, no iteration."""
        k = self.flow_to_density(q, v_f, k_j)
        return self.density_to_speed(k, v_f, k_j)

    def density_to_flow(
        self,
        k: np.ndarray,
        v_f: np.ndarray,
        k_j: np.ndarray,
    ) -> np.ndarray:
        """Evaluate q(k) = k * v(k) for all branches."""
        v = self.density_to_speed(k, v_f, k_j)
        return np.clip(np.asarray(k, dtype=np.float64) * v, 0.0, None)
