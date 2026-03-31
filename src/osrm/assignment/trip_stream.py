"""Trip-stream demand adapter for matrix-free loading scaffolds.

This is the matrix-free counterpart to :mod:`od_matrix`. It does not yet
implement a fully stateful online hill-climber; instead it provides the
batching and time-slicing primitives that a future hill-climber solver will
consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence

from osrm.assignment.od_matrix import DemandTrip


@dataclass
class TripBatch:
    """A batch of trips sharing a time slice."""

    trips: List[DemandTrip]
    batch_index: int
    departure_bin: Optional[int] = None


class TripStreamAdapter:
    """Normalize a stream of trips into ordered batches or time slices.

    Parameters
    ----------
    trips : iterable of DemandTrip
        Input trips from an activity-based or other matrix-free demand model.
    sort_by_departure : bool
        If True, sort trips by ``departure_time_s`` before batching.
    """

    def __init__(
        self,
        trips: Iterable[DemandTrip],
        *,
        sort_by_departure: bool = True,
    ) -> None:
        ordered = list(trips)
        if sort_by_departure:
            ordered.sort(key=lambda t: t.departure_time_s)
        self._trips = ordered

    @property
    def n_trips(self) -> int:
        """Total number of trips in the stream."""
        return len(self._trips)

    def trips(self) -> List[DemandTrip]:
        """Return all trips as a materialized list."""
        return list(self._trips)

    def iter_batches(self, batch_size: int) -> Iterator[TripBatch]:
        """Yield fixed-size batches preserving trip order."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for i in range(0, len(self._trips), batch_size):
            yield TripBatch(
                trips=self._trips[i:i + batch_size],
                batch_index=i // batch_size,
                departure_bin=None,
            )

    def iter_time_slices(
        self,
        *,
        bin_width_s: float,
        max_batch_size: Optional[int] = None,
    ) -> Iterator[TripBatch]:
        """Yield batches grouped by departure-time bin.

        Parameters
        ----------
        bin_width_s : float
            Width of each departure-time slice in seconds.
        max_batch_size : int, optional
            If provided, split each time slice into multiple sub-batches no
            larger than this size.
        """
        if bin_width_s <= 0:
            raise ValueError("bin_width_s must be positive")
        if max_batch_size is not None and max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")

        current_bin: Optional[int] = None
        current: List[DemandTrip] = []
        batch_index = 0

        def flush(bin_idx: Optional[int], chunk: Sequence[DemandTrip]) -> Iterator[TripBatch]:
            nonlocal batch_index
            if not chunk:
                return
            if max_batch_size is None:
                yield TripBatch(list(chunk), batch_index=batch_index, departure_bin=bin_idx)
                batch_index += 1
                return
            for i in range(0, len(chunk), max_batch_size):
                yield TripBatch(
                    list(chunk[i:i + max_batch_size]),
                    batch_index=batch_index,
                    departure_bin=bin_idx,
                )
                batch_index += 1

        for trip in self._trips:
            bin_idx = int(trip.departure_time_s // bin_width_s)
            if current_bin is None:
                current_bin = bin_idx
            if bin_idx != current_bin:
                yield from flush(current_bin, current)
                current = []
                current_bin = bin_idx
            current.append(trip)

        if current:
            yield from flush(current_bin, current)
