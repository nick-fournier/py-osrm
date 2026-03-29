"""Fractional link loading for multi-bin trip distribution.

Distributes route flow across time bins based on when the vehicle actually
traverses each link, using cumulative travel time offsets from OSRM
per-link duration annotations.

See docs/traffic_assignment_design.md §4.3 for the algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class LinkBinAssignment:
    """A single (link_index, bin_index, fraction) assignment."""

    link_idx: int
    bin_idx: int
    fraction: float


class FractionalLoader:
    """Distributes route flow across time bins.

    Parameters
    ----------
    bin_width_s : float
        Width of each time bin in seconds (e.g. 3600 for 1-hour bins).
    """

    def __init__(self, bin_width_s: float = 3600.0) -> None:
        if bin_width_s <= 0:
            raise ValueError("bin_width_s must be positive")
        self.bin_width_s = bin_width_s

    def distribute_route(
        self,
        link_durations_s: np.ndarray,
        departure_time_s: float,
    ) -> List[LinkBinAssignment]:
        """Distribute a route's links across time bins.

        Parameters
        ----------
        link_durations_s : array-like
            Per-link travel time in seconds.
        departure_time_s : float
            Departure time in seconds from epoch (or period start).

        Returns
        -------
        list of LinkBinAssignment
            (link_idx, bin_idx, fraction) tuples. Fractions for each link
            sum to 1.0.
        """
        link_durations_s = np.asarray(link_durations_s, dtype=np.float64)
        assignments: List[LinkBinAssignment] = []
        cum = departure_time_s

        for i, d in enumerate(link_durations_s):
            if d <= 0:
                continue
            enter = cum
            exit_ = cum + d
            b_enter = int(enter // self.bin_width_s)
            # If exit lands exactly on a bin boundary, it belongs to the
            # previous bin (the vehicle has left by that instant).
            if exit_ > 0 and exit_ % self.bin_width_s == 0:
                b_exit = int(exit_ // self.bin_width_s) - 1
            else:
                b_exit = int(exit_ // self.bin_width_s)

            if b_enter == b_exit:
                assignments.append(LinkBinAssignment(i, b_enter, 1.0))
            else:
                for b in range(b_enter, b_exit + 1):
                    bs = b * self.bin_width_s
                    be = (b + 1) * self.bin_width_s
                    overlap = min(exit_, be) - max(enter, bs)
                    assignments.append(LinkBinAssignment(i, b, overlap / d))

            cum = exit_

        return assignments

    def distribute_route_to_arrays(
        self,
        link_durations_s: np.ndarray,
        departure_time_s: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Same as distribute_route but returns parallel arrays.

        Returns
        -------
        link_indices : np.ndarray (int)
        bin_indices : np.ndarray (int)
        fractions : np.ndarray (float64)
        """
        assignments = self.distribute_route(link_durations_s, departure_time_s)
        if not assignments:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )
        link_idx = np.array([a.link_idx for a in assignments], dtype=np.int64)
        bin_idx = np.array([a.bin_idx for a in assignments], dtype=np.int64)
        fractions = np.array([a.fraction for a in assignments], dtype=np.float64)
        return link_idx, bin_idx, fractions

    def accumulate_route_flow(
        self,
        edge_ordinals: np.ndarray,
        link_durations_s: np.ndarray,
        departure_time_s: float,
        volume: float,
        flow_bins: np.ndarray,
    ) -> None:
        """Accumulate fractional flow into a (n_edges, n_bins) array.

        Parameters
        ----------
        edge_ordinals : np.ndarray
            (n_links,) int — edge ordinals for each link in route.
        link_durations_s : np.ndarray
            (n_links,) float — per-link duration in seconds.
        departure_time_s : float
            Departure time in seconds.
        volume : float
            Number of trips for this route (will be converted to veh/hr
            by dividing by bin_width).
        flow_bins : np.ndarray
            (n_edges, n_bins) float — accumulator. Modified in-place.
        """
        volume_rate = volume / (self.bin_width_s / 3600.0)  # convert to veh/hr
        link_idx, bin_idx, fractions = self.distribute_route_to_arrays(
            link_durations_s, departure_time_s
        )

        n_bins = flow_bins.shape[1]
        for li, bi, frac in zip(link_idx, bin_idx, fractions):
            if li < len(edge_ordinals) and 0 <= bi < n_bins:
                ei = edge_ordinals[li]
                if 0 <= ei < flow_bins.shape[0]:
                    flow_bins[ei, bi] += volume_rate * frac
