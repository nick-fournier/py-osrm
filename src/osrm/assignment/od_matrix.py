"""OD-matrix demand adapter for traffic assignment.

Converts an OD matrix + coordinate lists into a demand bucket
of (origin, destination, volume, departure_time) trips.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class DemandTrip:
    """A single OD trip with volume and departure time."""
    origin: Tuple[float, float]       # (lon, lat)
    destination: Tuple[float, float]  # (lon, lat)
    volume: float                     # vehicles per time period
    departure_time_s: float = 0.0     # seconds from epoch
    trip_id: Optional[str] = None     # unique ID for ABM runs


class ODMatrixAdapter:
    """Converts OD matrix + zone coordinates into a trip list.

    Parameters
    ----------
    origins : sequence of (lon, lat)
        Origin zone centroids.
    destinations : sequence of (lon, lat)
        Destination zone centroids.
    matrix : np.ndarray
        (n_origins, n_destinations) vehicle counts per time period.
    departure_time_s : float
        Departure time for all trips in this matrix.
    min_volume : float
        OD pairs below this threshold are skipped.
    """

    def __init__(
        self,
        origins: Sequence[Tuple[float, float]],
        destinations: Sequence[Tuple[float, float]],
        matrix: np.ndarray,
        departure_time_s: float = 0.0,
        min_volume: float = 0.1,
    ) -> None:
        self.origins = list(origins)
        self.destinations = list(destinations)
        self.matrix = np.asarray(matrix, dtype=np.float64)
        self.departure_time_s = departure_time_s
        self.min_volume = min_volume

        if self.matrix.shape != (len(self.origins), len(self.destinations)):
            raise ValueError(
                f"Matrix shape {self.matrix.shape} doesn't match "
                f"origins ({len(self.origins)}) × destinations ({len(self.destinations)})"
            )

    @property
    def total_demand(self) -> float:
        return float(self.matrix.sum())

    @property
    def n_od_pairs(self) -> int:
        return int(np.sum(self.matrix >= self.min_volume))

    def trips(self) -> List[DemandTrip]:
        """Generate trip list from the OD matrix."""
        result: List[DemandTrip] = []
        for i, origin in enumerate(self.origins):
            for j, dest in enumerate(self.destinations):
                vol = self.matrix[i, j]
                if vol >= self.min_volume:
                    result.append(DemandTrip(
                        origin=origin,
                        destination=dest,
                        volume=vol,
                        departure_time_s=self.departure_time_s,
                    ))
        return result

    @classmethod
    def from_uniform_random(
        cls,
        coordinates: Sequence[Tuple[float, float]],
        total_demand: float,
        seed: int = 42,
        departure_time_s: float = 0.0,
    ) -> "ODMatrixAdapter":
        """Create a synthetic OD matrix with uniform random demand.

        Useful for benchmarking and smoke tests.
        """
        rng = np.random.default_rng(seed)
        n = len(coordinates)
        matrix = rng.uniform(0, 1, (n, n))
        np.fill_diagonal(matrix, 0.0)
        matrix = matrix / matrix.sum() * total_demand
        return cls(
            origins=coordinates,
            destinations=coordinates,
            matrix=matrix,
            departure_time_s=departure_time_s,
        )
