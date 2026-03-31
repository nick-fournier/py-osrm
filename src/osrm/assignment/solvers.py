"""High-level assignment solver front doors.

These classes separate the two long-term assignment modes:

- ``MatrixAssignmentSolver``: OD-matrix / zone-based assignment.
- ``MatrixFreeHillClimber``: matrix-free trip-stream loading with
  wrapper-side batching and network updates.

Both intentionally reuse the shared assignment core (`AssignmentLoop`,
`NetworkState`, `BiParabolicVDF`, CSV customization, reporting helpers)
instead of duplicating logic.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np
import osrm as osrm_module

from osrm.assignment.assignment_loop import (
    AssignmentConfig,
    AssignmentLoop,
    AssignmentResult,
)
from osrm.assignment.network_state import NetworkState
from osrm.assignment.od_matrix import DemandTrip, ODMatrixAdapter
from osrm.assignment.trip_stream import TripBatch, TripStreamAdapter

logger = logging.getLogger(__name__)

# Minimum seconds between progress log messages (avoids spam on fast runs)
_LOG_INTERVAL_S = 2.0


# ---------------------------------------------------------------------------
# Slice ledger — tracks per-slice density contributions for rerouting
# ---------------------------------------------------------------------------

@dataclass
class SliceLedgerEntry:
    """One slice's contribution to network state.

    Stored after each HC batch so that reroute epochs can subtract the
    old contribution, re-route trips, and add the new one.
    """

    batch_index: int
    trips: List[DemandTrip]
    density: np.ndarray    # per-edge density contribution (veh/km)
    volume: np.ndarray     # per-edge volume contribution (vehicles)
    tstt: float            # batch TSTT (veh-seconds)


@dataclass
class SliceLedger:
    """Ordered collection of slice contributions.

    Invariant: ``sum(entry.density for entry in entries)`` equals the
    total density accumulated on the network (before clamping to k_j).
    """

    entries: List[SliceLedgerEntry] = field(default_factory=list)

    def append(self, entry: SliceLedgerEntry) -> None:
        self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]

    @property
    def total_density(self) -> Optional[np.ndarray]:
        """Sum of all slice density contributions (unclamped)."""
        if not self.entries:
            return None
        return sum(e.density for e in self.entries)

    @property
    def total_volume(self) -> Optional[np.ndarray]:
        if not self.entries:
            return None
        return sum(e.volume for e in self.entries)

    def pad_all(self, n_edges: int) -> None:
        """Extend all entries to *n_edges* if the network grew."""
        for entry in self.entries:
            if len(entry.density) < n_edges:
                pad = n_edges - len(entry.density)
                entry.density = np.append(entry.density, np.zeros(pad))
                entry.volume = np.append(entry.volume, np.zeros(pad))


class MatrixAssignmentSolver:
    """Matrix-based assignment front door.

    This is the canonical wrapper for OD-matrix workflows today. It is a thin
    façade over :class:`AssignmentLoop`, but gives the architecture an explicit
    home for future shortest-path-tree / bush / Table-accelerated logic without
    polluting the matrix-free API.
    """

    def __init__(self, base_path: str, config: Optional[AssignmentConfig] = None) -> None:
        self.base_path = base_path
        self.config = config or AssignmentConfig()

    def _make_loop(self) -> AssignmentLoop:
        return AssignmentLoop(self.base_path, self.config)

    def solve(
        self,
        demand: ODMatrixAdapter | Sequence[DemandTrip],
        *,
        state_patch=None,
        progress_callback=None,
    ) -> AssignmentResult:
        """Solve a matrix-style assignment problem.

        Parameters
        ----------
        demand : ODMatrixAdapter or sequence of DemandTrip
            Either an adapter that can materialize trips from an OD matrix, or
            a pre-built trip list.
        """
        trips = demand.trips() if hasattr(demand, "trips") else list(demand)
        return self._make_loop().run(
            trips,
            state_patch=state_patch,
            progress_callback=progress_callback,
        )


@dataclass
class HillClimberBatchResult:
    """Metrics for one matrix-free loading batch."""

    batch_index: int
    departure_bin: Optional[int]
    n_trips: int
    batch_tstt: float
    route_time_s: float
    customize_time_s: float
    engine_time_s: float
    max_k_over_kj: float
    mean_speed_kmh: float


@dataclass
class RerouteSliceSnapshot:
    """Metrics captured after rerouting a single slice within an epoch."""

    epoch: int
    slice_index: int
    tstt: float
    max_k_over_kj: float
    mean_speed_kmh: float
    route_time_s: float = 0.0
    customize_time_s: float = 0.0
    engine_time_s: float = 0.0


@dataclass
class RerouteEpochResult:
    """Metrics for one reroute epoch (full pass through all slices)."""

    epoch: int
    slices_rerouted: int
    routes_changed: int
    epoch_time_s: float
    gap: Optional[float]
    max_k_over_kj: float
    mean_speed_kmh: float
    slice_snapshots: List["RerouteSliceSnapshot"] = field(default_factory=list)


@dataclass
class HillClimberResult:
    """Final result of the wrapper-side hill-climber MVP."""

    network_state: Optional[NetworkState]
    batch_results: List[HillClimberBatchResult]
    total_time_s: float
    n_trips: int
    slice_ledger: Optional[SliceLedger] = None
    epoch_results: List[RerouteEpochResult] = field(default_factory=list)

    @property
    def n_batches(self) -> int:
        return len(self.batch_results)

    def log_as_dict(self) -> dict:
        return {
            "batch_index": [r.batch_index for r in self.batch_results],
            "departure_bin": [r.departure_bin for r in self.batch_results],
            "n_trips": [r.n_trips for r in self.batch_results],
            "batch_tstt": [r.batch_tstt for r in self.batch_results],
            "route_time_s": [r.route_time_s for r in self.batch_results],
            "customize_time_s": [r.customize_time_s for r in self.batch_results],
            "engine_time_s": [r.engine_time_s for r in self.batch_results],
            "max_k_over_kj": [r.max_k_over_kj for r in self.batch_results],
            "mean_speed_kmh": [r.mean_speed_kmh for r in self.batch_results],
        }


class MatrixFreeHillClimber:
    """Matrix-free loading MVP over the shared assignment core.

    The MVP is fully wrapper-side: it batches trips in Python, updates one
    mutable network state, writes fresh segment-speed CSVs after each batch,
    and recreates the OSRM engine between batches. No OSRM core patch is
    required.

    Once the OSRM multi-period patch lands, this solver can be upgraded to:
    - route by ``departure_time`` / ``departure_period`` directly,
    - keep multiple period-specific metric sets in one engine,
    - avoid Python-side slice orchestration and repeated engine reloads.
    """

    def __init__(
        self,
        base_path: str,
        config: Optional[AssignmentConfig] = None,
        *,
        default_batch_size: int = 1000,
    ) -> None:
        if default_batch_size <= 0:
            raise ValueError("default_batch_size must be positive")
        self.base_path = base_path
        self.config = config or AssignmentConfig()
        self.default_batch_size = default_batch_size

    def _make_loop(self) -> AssignmentLoop:
        return AssignmentLoop(self.base_path, self.config)

    def run_batch(
        self,
        trips: Sequence[DemandTrip],
        *,
        state_patch=None,
        progress_callback=None,
    ) -> AssignmentResult:
        """Run one frozen-cost trip batch through the shared assignment core."""
        return self._make_loop().run(
            list(trips),
            state_patch=state_patch,
            progress_callback=progress_callback,
        )

    def run_independent_batches(
        self,
        stream: TripStreamAdapter | Iterable[DemandTrip],
        *,
        batch_size: Optional[int] = None,
        state_patch=None,
        progress_callback=None,
    ) -> List[AssignmentResult]:
        """Execute a trip stream as isolated frozen-cost assignment runs.

        This preserves the original scaffold semantics for tests and debugging.
        """
        adapter = stream if isinstance(stream, TripStreamAdapter) else TripStreamAdapter(stream)
        size = batch_size or self.default_batch_size
        results: List[AssignmentResult] = []
        for batch in adapter.iter_batches(size):
            results.append(
                self.run_batch(
                    batch.trips,
                    state_patch=state_patch,
                    progress_callback=progress_callback,
                )
            )
        return results

    def run_stream(
        self,
        stream: TripStreamAdapter | Iterable[DemandTrip],
        *,
        max_batch_size: Optional[int] = None,
        state_patch=None,
        progress_callback=None,
        max_epochs: int = 0,
        gap_threshold: float = 0.01,
        gap_sample_floor: int = 10_000,
    ) -> HillClimberResult:
        """Run a stateful wrapper-side hill-climber over ordered trip batches.

        Trips are grouped by departure-time slice using ``config.bin_width_s``.
        If ``max_batch_size`` is provided, each departure slice is further
        micro-batched and loaded sequentially. Each batch:

        1. Routes on the current network state
        2. Accumulates additional density/volume onto the shared network state
        3. Recomputes VDF speeds
        4. Re-customizes OSRM and reloads the engine

        After the initial load, if ``max_epochs > 0``, reroute epochs iterate
        through slices oldest-first, subtracting old density, re-routing on
        the updated state, and adding new density.  Stops when the sampled
        Wardrop gap falls below ``gap_threshold`` or ``max_epochs`` is reached.

        This is the intended MVP for matrix-free loading before the OSRM
        multi-period patch is available.
        """
        started = time.monotonic()
        adapter = stream if isinstance(stream, TripStreamAdapter) else TripStreamAdapter(stream)
        all_trips = adapter.trips()
        if not all_trips:
            return HillClimberResult(
                network_state=None,
                batch_results=[],
                total_time_s=0.0,
                n_trips=0,
            )

        loop = self._make_loop()
        logger.info(
            "HC started: %d trips, epochs=%d",
            len(all_trips), max_epochs,
        )
        engine = loop._create_engine()

        snapped_trips = loop._snap_trips(engine, all_trips)
        snapped_stream = TripStreamAdapter(snapped_trips, sort_by_departure=False)
        state = loop._discover_network(engine, snapped_trips)
        if state_patch:
            state_patch(state)
        loop.smoother.build_adjacency(state.edge_ids, state.length_m)

        batch_results: List[HillClimberBatchResult] = []
        ledger = SliceLedger()
        _last_log = time.monotonic()
        for batch in snapped_stream.iter_time_slices(
            bin_width_s=self.config.bin_width_s,
            max_batch_size=max_batch_size,
        ):
            t_route = time.monotonic()
            batch_density, batch_volume, batch_tstt = loop._route_and_accumulate(
                engine,
                batch.trips,
                state,
            )
            route_time = time.monotonic() - t_route

            if len(batch_density) < state.n_edges:
                pad = state.n_edges - len(batch_density)
                batch_density = np.append(batch_density, np.zeros(pad))
                batch_volume = np.append(batch_volume, np.zeros(pad))
            # Keep ledger entries aligned with any network growth
            ledger.pad_all(state.n_edges)
            if state_patch:
                state_patch(state)
            loop.smoother.build_adjacency(state.edge_ids, state.length_m)

            # Record contribution BEFORE clamping (unclamped is the true
            # additive contribution; clamping happens on the total).
            ledger.append(SliceLedgerEntry(
                batch_index=batch.batch_index,
                trips=list(batch.trips),
                density=batch_density.copy(),
                volume=batch_volume.copy(),
                tstt=batch_tstt,
            ))

            state.density_vpkm = np.clip(
                state.density_vpkm + batch_density,
                0.0,
                state.jam_density,
            )
            loop._update_state(state)

            t_cust = time.monotonic()
            csv_path = loop.writer.write_from_state(state, only_changed=True)
            osrm_module.customize(
                self.base_path,
                segment_speed_file=str(csv_path),
                verbosity="ERROR",  # OSRM C++ always quiet
            )
            customize_time = time.monotonic() - t_cust

            t_engine = time.monotonic()
            del engine
            engine = loop._create_engine()
            engine_time = time.monotonic() - t_engine

            max_k_over_kj = float(np.max(state.density_vpkm / np.maximum(state.jam_density, 1e-9)))
            mean_speed = float(np.mean(state.speed_kmh))
            batch_result = HillClimberBatchResult(
                batch_index=batch.batch_index,
                departure_bin=batch.departure_bin,
                n_trips=len(batch.trips),
                batch_tstt=batch_tstt,
                route_time_s=route_time,
                customize_time_s=customize_time,
                engine_time_s=engine_time,
                max_k_over_kj=max_k_over_kj,
                mean_speed_kmh=mean_speed,
            )
            batch_results.append(batch_result)

            now = time.monotonic()
            if now - _last_log >= _LOG_INTERVAL_S:
                logger.info(
                    "E0:S%d TSTT=%.0f speed=%.1f km/h k/kj=%.2f",
                    batch.batch_index, batch_tstt, mean_speed, max_k_over_kj,
                )
                _last_log = now

            if progress_callback:
                progress_callback(batch_result)

        # --- Reroute epochs (tail-eating) ---
        epoch_results: List[RerouteEpochResult] = []
        for epoch_idx in range(max_epochs):
            epoch_start = time.monotonic()
            routes_changed = 0
            slice_snapshots: List[RerouteSliceSnapshot] = []

            for slice_idx in range(len(ledger)):
                entry = ledger[slice_idx]

                # 1. Subtract this slice's density contribution
                state.density_vpkm = np.clip(
                    state.density_vpkm - entry.density,
                    0.0,
                    state.jam_density,
                )
                loop._update_state(state)

                # 2. Customize OSRM with reduced-state speeds
                t_cust = time.monotonic()
                csv_path = loop.writer.write_from_state(state, only_changed=True)
                osrm_module.customize(
                    self.base_path,
                    segment_speed_file=str(csv_path),
                    verbosity="ERROR",  # OSRM C++ always quiet
                )
                rr_customize_time = time.monotonic() - t_cust

                t_eng = time.monotonic()
                del engine
                engine = loop._create_engine()
                rr_engine_time = time.monotonic() - t_eng

                # 3. Re-route this slice's trips (randomize OD order)
                shuffled_trips = list(entry.trips)
                random.shuffle(shuffled_trips)
                t_route = time.monotonic()
                new_density, new_volume, new_tstt = loop._route_and_accumulate(
                    engine, shuffled_trips, state,
                )
                rr_route_time = time.monotonic() - t_route

                # Pad if network grew during rerouting
                if len(new_density) < state.n_edges:
                    pad = state.n_edges - len(new_density)
                    new_density = np.append(new_density, np.zeros(pad))
                    new_volume = np.append(new_volume, np.zeros(pad))
                ledger.pad_all(state.n_edges)

                # Track route changes (density shifted materially)
                density_delta = np.abs(new_density - entry.density)
                if density_delta.sum() > 0.01 * entry.density.sum():
                    routes_changed += 1

                # 4. Update ledger entry with new contribution
                entry.density = new_density.copy()
                entry.volume = new_volume.copy()
                entry.tstt = new_tstt

                # 5. Add new density back
                state.density_vpkm = np.clip(
                    state.density_vpkm + new_density,
                    0.0,
                    state.jam_density,
                )
                if state_patch:
                    state_patch(state)
                loop.smoother.build_adjacency(state.edge_ids, state.length_m)
                loop._update_state(state)

                # Capture per-slice snapshot
                snap_k = float(
                    np.max(state.density_vpkm / np.maximum(state.jam_density, 1e-9))
                )
                snap_speed = float(np.mean(state.speed_kmh))
                slice_snapshots.append(RerouteSliceSnapshot(
                    epoch=epoch_idx + 1,
                    slice_index=slice_idx,
                    tstt=new_tstt,
                    max_k_over_kj=snap_k,
                    mean_speed_kmh=snap_speed,
                    route_time_s=rr_route_time,
                    customize_time_s=rr_customize_time,
                    engine_time_s=rr_engine_time,
                ))

                now = time.monotonic()
                if now - _last_log >= _LOG_INTERVAL_S:
                    logger.info(
                        "E%d:S%d rerouting (%d/%d slices)",
                        epoch_idx + 1, slice_idx, slice_idx + 1, len(ledger),
                    )
                    _last_log = now

            # Customize once more after full epoch
            csv_path = loop.writer.write_from_state(state, only_changed=True)
            osrm_module.customize(
                self.base_path,
                segment_speed_file=str(csv_path),
                verbosity="ERROR",  # OSRM C++ always quiet
            )
            del engine
            engine = loop._create_engine()

            # Gap check via sampled Table
            gap = self._sampled_gap(
                engine, ledger, state, loop,
                sample_floor=gap_sample_floor,
            )

            max_k_over_kj = float(
                np.max(state.density_vpkm / np.maximum(state.jam_density, 1e-9))
            )
            mean_speed = float(np.mean(state.speed_kmh))

            epoch_result = RerouteEpochResult(
                epoch=epoch_idx + 1,
                slices_rerouted=len(ledger),
                routes_changed=routes_changed,
                epoch_time_s=time.monotonic() - epoch_start,
                gap=gap,
                max_k_over_kj=max_k_over_kj,
                mean_speed_kmh=mean_speed,
                slice_snapshots=slice_snapshots,
            )
            epoch_results.append(epoch_result)
            gap_str = f"{gap:.6f}" if gap is not None else "n/a"
            logger.info(
                "E%d: gap=%s, changed=%d/%d, speed=%.1f km/h, %.1fs",
                epoch_idx + 1, gap_str, routes_changed, len(ledger),
                mean_speed, epoch_result.epoch_time_s,
            )

            if progress_callback:
                progress_callback(epoch_result)

            if gap is not None and gap < gap_threshold:
                break

        total_time = time.monotonic() - started
        logger.info(
            "HC complete: %d slices, %d epochs, %.1fs",
            len(batch_results), len(epoch_results), total_time,
        )

        return HillClimberResult(
            network_state=state,
            batch_results=batch_results,
            total_time_s=total_time,
            n_trips=len(snapped_trips),
            slice_ledger=ledger,
            epoch_results=epoch_results,
        )

    def _sampled_gap(
        self,
        engine,
        ledger: SliceLedger,
        state: NetworkState,
        loop: AssignmentLoop,
        *,
        sample_floor: int = 10_000,
    ) -> Optional[float]:
        """Compute Wardrop relative gap from (sampled) Table API.

        Collects unique OD pairs from the ledger, samples up to
        ``sample_floor`` if the total exceeds that threshold, then
        compares shortest-path travel times (from Table API) against
        assigned route travel times (from current state speeds).

        Returns
        -------
        float or None
            Relative gap: sum(assigned - shortest) / sum(shortest).
            None if no OD pairs could be evaluated.
        """
        import random

        # Collect unique OD pairs with their total assigned volume
        od_pairs: dict[tuple[float, float, float, float], float] = {}
        for entry in ledger:
            for trip in entry.trips:
                key = (trip.origin[0], trip.origin[1],
                       trip.destination[0], trip.destination[1])
                od_pairs[key] = od_pairs.get(key, 0.0) + trip.volume

        if not od_pairs:
            return None

        # Sample if needed
        all_keys = list(od_pairs.keys())
        if len(all_keys) > sample_floor:
            sampled_keys = random.sample(all_keys, sample_floor)
        else:
            sampled_keys = all_keys

        # Build trips for Table API
        sample_trips = [
            DemandTrip(origin=(k[0], k[1]), destination=(k[2], k[3]),
               volume=od_pairs[k])
            for k in sampled_keys
        ]

        # Get shortest-path times via routing on current (congested) state
        raw_results = loop._batch_route_raw(engine, sample_trips)

        sum_shortest = 0.0
        sum_assigned = 0.0
        evaluated = 0

        for trip_idx, raw in enumerate(raw_results):
            if raw is None:
                continue
            routes = raw["routes"]
            if not routes:
                continue
            trip = sample_trips[trip_idx]
            shortest_time = float(routes[0]["duration"])

            # Assigned time: compute from current state speeds along the
            # route edges.  This uses the VDF-based speeds (which may differ
            # slightly from OSRM's quantized speeds).
            route = routes[0]
            assigned_time = 0.0
            for leg in route["legs"]:
                ann = leg["annotation"]
                nodes = ann["nodes"]
                for i in range(len(nodes) - 1):
                    idx = state.edge_ordinal(int(nodes[i]), int(nodes[i + 1]))
                    if idx is not None:
                        v = max(state.speed_kmh[idx], 1.08)
                        assigned_time += state.length_m[idx] / (v / 3.6)
                    else:
                        # Edge not in state — use OSRM annotation speed
                        speeds = ann.get("speed", [])
                        spd = (speeds[i] * 3.6) if i < len(speeds) and speeds[i] > 0 else 1.08
                        dists = ann.get("distance", [])
                        dist = dists[i] if i < len(dists) else 0.0
                        assigned_time += dist / (spd / 3.6)

            sum_shortest += trip.volume * shortest_time
            sum_assigned += trip.volume * assigned_time
            evaluated += 1

        if evaluated == 0 or sum_shortest < 1e-9:
            return None

        return (sum_assigned - sum_shortest) / sum_shortest

    def iter_time_slices(
        self,
        stream: TripStreamAdapter | Iterable[DemandTrip],
        *,
        max_batch_size: Optional[int] = None,
    ) -> List[TripBatch]:
        """Expose time-sliced batching for future stateful hill-climbing."""
        adapter = stream if isinstance(stream, TripStreamAdapter) else TripStreamAdapter(stream)
        return list(
            adapter.iter_time_slices(
                bin_width_s=self.config.bin_width_s,
                max_batch_size=max_batch_size,
            )
        )
