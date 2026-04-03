"""Unified traffic assignment solver.

``TrafficAssignmentSolver`` handles both OD-matrix and matrix-free
trip-stream assignment.  It reuses the shared assignment core
(``AssignmentLoop``, ``NetworkState``, ``BiParabolicVDF``, CSV
customization, reporting helpers).
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import osrm as osrm_module

from osrm.assignment.assignment_loop import (
    AssignmentConfig,
    AssignmentLoop,
    AssignmentResult,
    RoutedTripPath,
)
from osrm.assignment.network_state import NetworkState
from osrm.assignment.od_matrix import DemandTrip, ODMatrixAdapter
from osrm.assignment.trip_stream import TripBatch, TripStreamAdapter

logger = logging.getLogger(__name__)

@dataclass
class RouteAssignment:
    """One route fragment currently carrying some share of an OD's volume."""

    edge_indices: List[int]
    density_contribution: List[float]  # actual current fragment contribution
    volume_fraction: float
    assigned_cost_s: float


@dataclass
class ODLedgerEntry:
    """One routed demand unit in the greedy-loaded network."""

    origin: Tuple[float, float]
    destination: Tuple[float, float]
    total_volume: float
    departure_time_s: float
    routes: List[RouteAssignment] = field(default_factory=list)
    current_gap: Optional[float] = None
    refinement_visits: int = 0

    @property
    def assigned_cost_s(self) -> float:
        """Current volume-weighted assigned cost across active route fragments."""
        if not self.routes:
            return 0.0
        return float(sum(
            route.assigned_cost_s * route.volume_fraction for route in self.routes
        ))

    def compress_routes(self) -> None:
        """Merge identical route fragments within this OD entry."""
        merged: dict[Tuple[int, ...], RouteAssignment] = {}
        for route in self.routes:
            signature = tuple(route.edge_indices)
            existing = merged.get(signature)
            if existing is None:
                merged[signature] = RouteAssignment(
                    edge_indices=list(route.edge_indices),
                    density_contribution=list(route.density_contribution),
                    volume_fraction=route.volume_fraction,
                    assigned_cost_s=route.assigned_cost_s,
                )
                continue

            total_fraction = existing.volume_fraction + route.volume_fraction
            if total_fraction > 0.0:
                existing.assigned_cost_s = (
                    existing.assigned_cost_s * existing.volume_fraction
                    + route.assigned_cost_s * route.volume_fraction
                ) / total_fraction
            existing.volume_fraction = total_fraction
            existing.density_contribution = [
                a + b
                for a, b in zip(
                    existing.density_contribution,
                    route.density_contribution,
                )
            ]
        self.routes = list(merged.values())


@dataclass
class ODLedger:
    """Ordered collection of routed demand units for sampled refinement."""

    entries: List[ODLedgerEntry] = field(default_factory=list)
    _index: dict[Tuple[Tuple[float, float], Tuple[float, float]], int] = field(
        default_factory=dict, init=False, repr=False,
    )

    def append(self, entry: ODLedgerEntry) -> None:
        key = (entry.origin, entry.destination)
        existing_idx = self._index.get(key)
        if existing_idx is None:
            self._index[key] = len(self.entries)
            self.entries.append(entry)
            return

        existing = self.entries[existing_idx]
        old_total = existing.total_volume
        new_total = old_total + entry.total_volume
        if new_total <= 0.0:
            return

        existing.total_volume = new_total
        scale = old_total / new_total if old_total > 0 else 0.0
        for route in existing.routes:
            route.volume_fraction *= scale

        added_scale = entry.total_volume / new_total
        for route in entry.routes:
            route.volume_fraction *= added_scale
            existing.routes.append(route)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]



@dataclass
class MSAIterationResult:
    """Metrics for one MSA convergence iteration."""

    iteration: int
    alpha: float
    aon_tstt: float
    link_tstt: float
    relative_gap: Optional[float]
    state_change_norm: float
    max_k_over_kj: float
    median_k_over_kj: float
    mean_speed_kmh: float
    median_speed_kmh: float
    max_speed_kmh: float
    min_speed_kmh: float
    median_tt_s: float
    max_tt_s: float
    route_time_s: float
    customize_time_s: float
    engine_time_s: float
    iteration_time_s: float
    n_routes: int


@dataclass
class HillClimberBatchResult:
    """Metrics for one loading batch."""

    batch_index: int
    departure_bin: Optional[int]
    n_trips: int
    batch_tstt: float
    network_tstt: float
    route_time_s: float
    customize_time_s: float
    engine_time_s: float
    max_k_over_kj: float
    median_k_over_kj: float
    mean_speed_kmh: float
    median_speed_kmh: float
    max_speed_kmh: float
    min_speed_kmh: float
    median_tt_s: float
    max_tt_s: float


@dataclass
class HillClimberResult:
    """Final result of the assignment solver."""

    network_state: Optional[NetworkState]
    batch_results: List[HillClimberBatchResult]
    total_time_s: float
    n_trips: int
    od_ledger: Optional[ODLedger] = None
    msa_results: List[MSAIterationResult] = field(default_factory=list)

    @property
    def n_batches(self) -> int:
        return len(self.batch_results)

    def log_as_dict(self) -> dict:
        return {
            "batch_index": [r.batch_index for r in self.batch_results],
            "departure_bin": [r.departure_bin for r in self.batch_results],
            "n_trips": [r.n_trips for r in self.batch_results],
            "batch_tstt": [r.batch_tstt for r in self.batch_results],
            "network_tstt": [r.network_tstt for r in self.batch_results],
            "route_time_s": [r.route_time_s for r in self.batch_results],
            "customize_time_s": [r.customize_time_s for r in self.batch_results],
            "engine_time_s": [r.engine_time_s for r in self.batch_results],
            "max_k_over_kj": [r.max_k_over_kj for r in self.batch_results],
            "median_k_over_kj": [r.median_k_over_kj for r in self.batch_results],
            "mean_speed_kmh": [r.mean_speed_kmh for r in self.batch_results],
            "median_speed_kmh": [r.median_speed_kmh for r in self.batch_results],
            "max_speed_kmh": [r.max_speed_kmh for r in self.batch_results],
            "min_speed_kmh": [r.min_speed_kmh for r in self.batch_results],
            "median_tt_s": [r.median_tt_s for r in self.batch_results],
            "max_tt_s": [r.max_tt_s for r in self.batch_results],
        }


class TrafficAssignmentSolver:
    """Unified traffic assignment solver.

    Supports both OD-matrix and matrix-free trip-stream workflows:

    - ``solve()``: OD-matrix / zone-based assignment via AssignmentLoop
    - ``run_stream()``: matrix-free batched loading with MSA convergence

    The MSA convergence loop operates on link-level density state:
    freeze -> route all demand (AON) -> blend with alpha=1/(m+1) -> update VDF.
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

    @staticmethod
    def _normalize_coord(coord) -> Tuple[float, float]:
        return (float(coord[0]), float(coord[1]))

    def _append_od_entries(
        self,
        od_ledger: ODLedger,
        trips: Sequence[DemandTrip],
        routed_paths: Sequence[RoutedTripPath],
    ) -> None:
        """Append greedy-load route assignments into the OD ledger."""
        for path in routed_paths:
            trip = trips[path.trip_index]
            od_ledger.append(ODLedgerEntry(
                origin=self._normalize_coord(trip.origin),
                destination=self._normalize_coord(trip.destination),
                total_volume=float(trip.volume),
                departure_time_s=float(trip.departure_time_s),
                routes=[RouteAssignment(
                    edge_indices=list(path.edge_indices),
                    density_contribution=list(path.density_contribution),
                    volume_fraction=1.0,
                    assigned_cost_s=float(path.duration_s),
                )],
            ))

    def solve(
        self,
        demand: ODMatrixAdapter | Sequence[DemandTrip],
        *,
        state_patch=None,
        progress_callback=None,
    ) -> AssignmentResult:
        """Solve an OD-matrix assignment problem via AssignmentLoop.

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

    def _customize_and_reload(
        self,
        loop: AssignmentLoop,
        state: NetworkState,
        engine,
    ) -> Tuple[object, float, float]:
        t_cust = time.monotonic()
        csv_path = loop.writer.write_from_state(state, only_changed=True)
        logger.info("Customizing OSRM...")
        osrm_module.customize(
            self.base_path,
            segment_speed_file=str(csv_path),
            verbosity="ERROR",
        )
        customize_time = time.monotonic() - t_cust
        logger.info("Customized in %.2fs", customize_time)

        t_engine = time.monotonic()
        del engine
        engine = loop._create_engine()
        engine_time = time.monotonic() - t_engine
        return engine, customize_time, engine_time

    def _run_convergence(
        self,
        *,
        engine,
        loop: AssignmentLoop,
        state: NetworkState,
        trips: list,
        initial_volume: np.ndarray,
        state_patch,
        progress_callback,
        max_od: int,
        max_rounds: int,
        gap_threshold: float,
        method: str = "msa",
    ) -> Tuple[object, List[MSAIterationResult]]:
        """Run convergence iterations on link-level density state.

        After greedy warm start, iteratively:
          1. Freeze current density
          2. Route demand on frozen network (all-or-nothing)
          3. Blend: k = (1-α)k + αk̂
             - MSA: α = 1/(m+1)
             - FW:  α from Beckmann line search
          4. Clip, update VDF, customize, reload

        Parameters
        ----------
        max_od : int
            Maximum OD pairs routed per iteration.  When the trip list is
            larger, MSA sub-samples; FW always requires full-pass so the
            guard in ``run_stream`` rejects oversized networks.
        """
        msa_results: List[MSAIterationResult] = []
        use_fw = method == "fw"
        n_trips = len(trips)
        full_pass = n_trips <= max_od or use_fw
        if use_fw and n_trips > max_od:
            logger.info(
                "FW: full-pass routing required (%d trips, max_od=%d)",
                n_trips, max_od,
            )
        prev_volume = initial_volume.copy()

        for m in range(1, max_rounds + 1):
            iter_start = time.monotonic()

            # 1. Freeze current state
            prev_density = state.density_vpkm.copy()

            # 2. Build auxiliary loading (AON on frozen network)
            if full_pass:
                route_trips = trips
            else:
                n_sample = min(n_trips, max_od)
                route_trips = random.sample(trips, n_sample)

            t_route = time.monotonic()
            aon_density, aon_volume, aon_tstt, _routed_paths = (
                loop._route_and_accumulate_with_paths(engine, route_trips, state)
            )
            route_time = time.monotonic() - t_route

            # Pad AON arrays to match state size (new edges may have appeared)
            if len(aon_density) < state.n_edges:
                pad = state.n_edges - len(aon_density)
                aon_density = np.append(aon_density, np.zeros(pad))
                aon_volume = np.append(aon_volume, np.zeros(pad))
            if len(prev_density) < state.n_edges:
                pad = state.n_edges - len(prev_density)
                prev_density = np.append(prev_density, np.zeros(pad))

            # Scale up sampled density/volume to full-demand estimate
            if not full_pass:
                scale_factor = n_trips / len(route_trips)
                aon_density = aon_density * scale_factor
                aon_volume = aon_volume * scale_factor
                aon_tstt = aon_tstt * scale_factor

            # Compute Wardrop gap on FROZEN state before blending.
            # Both numerator and denominator use the same VDF link costs,
            # eliminating OSRM quantization bias.  Uses demand-based volumes
            # (not MFD throughput) so gap is always ≥ 0.
            if len(prev_volume) < state.n_edges:
                prev_volume = np.append(
                    prev_volume,
                    np.zeros(state.n_edges - len(prev_volume)),
                )
            relative_gap = loop._compute_relative_gap(
                state, prev_volume, aon_volume,
            )

            # 3. Blend: step size depends on method
            if use_fw:
                alpha = loop._fw_line_search(
                    prev_density, aon_density,
                    prev_volume, aon_volume,
                    state,
                )
            else:
                alpha = 1.0 / (m + 1)
            state.density_vpkm = np.clip(
                (1.0 - alpha) * prev_density + alpha * aon_density,
                0.0,
                state.jam_density,
            )
            prev_volume = (1.0 - alpha) * prev_volume + alpha * aon_volume

            # 4. Update derived quantities
            if state_patch:
                state_patch(state)
            loop.smoother.build_adjacency(state.edge_ids, state.length_m)
            loop._update_state(state)

            # Customize and reload
            engine, customize_time, engine_time = self._customize_and_reload(
                loop, state, engine,
            )

            # --- Metrics ---
            state_change_norm = float(
                np.linalg.norm(state.density_vpkm - prev_density)
                / max(np.linalg.norm(prev_density), 1e-9)
            )

            # Post-blend link-level TSTT for tracking
            post_speed_ms = np.maximum(state.speed_kmh / 3.6, 0.001)
            post_link_tt_s = state.length_m / post_speed_ms
            link_tstt = float(np.sum(state.flow_vph * post_link_tt_s))

            max_k_over_kj = float(
                np.max(state.density_vpkm / np.maximum(state.jam_density, 1e-9))
            )
            k_over_kj = state.density_vpkm / np.maximum(state.jam_density, 1e-9)
            median_k_over_kj = float(np.median(k_over_kj[k_over_kj > 0])) if np.any(k_over_kj > 0) else 0.0

            # Link speed statistics (only over links with nonzero density)
            active = state.density_vpkm > 0
            active_speeds = state.speed_kmh[active] if np.any(active) else state.speed_kmh
            mean_speed = float(np.mean(active_speeds))
            median_speed = float(np.median(active_speeds))
            max_speed = float(np.max(active_speeds))
            min_speed = float(np.min(active_speeds))

            # Link travel time statistics (only over active links)
            active_tt = post_link_tt_s[active] if np.any(active) else post_link_tt_s
            median_tt = float(np.median(active_tt))
            max_tt = float(np.max(active_tt))

            iter_time = time.monotonic() - iter_start

            iter_result = MSAIterationResult(
                iteration=m,
                alpha=alpha,
                aon_tstt=float(aon_tstt),
                link_tstt=link_tstt,
                relative_gap=relative_gap,
                state_change_norm=state_change_norm,
                max_k_over_kj=max_k_over_kj,
                median_k_over_kj=median_k_over_kj,
                mean_speed_kmh=mean_speed,
                median_speed_kmh=median_speed,
                max_speed_kmh=max_speed,
                min_speed_kmh=min_speed,
                median_tt_s=median_tt,
                max_tt_s=max_tt,
                route_time_s=route_time,
                customize_time_s=customize_time,
                engine_time_s=engine_time,
                iteration_time_s=iter_time,
                n_routes=len(route_trips),
            )
            msa_results.append(iter_result)

            if progress_callback:
                progress_callback(iter_result)

            method_label = "FW" if use_fw else "MSA"
            logger.info(
                "%s iter %d: α=%.3f Δk=%.4f gap=%s TSTT=%.0f "
                "v̄=%.1f v̂=%.1f v↓=%.1f km/h  "
                "t̃=%.1f t↑=%.1fs  k/kj=%.2f (%.1fs)",
                method_label, m, alpha, state_change_norm,
                f"{relative_gap:.4f}" if relative_gap is not None else "n/a",
                link_tstt,
                mean_speed, median_speed, min_speed,
                median_tt, max_tt,
                max_k_over_kj, iter_time,
            )

            # Convergence checks
            if relative_gap is not None and 0 <= relative_gap < gap_threshold:
                logger.info(
                    "%s converged at iteration %d: gap=%.6f < %.6f",
                    method_label, m, relative_gap, gap_threshold,
                )
                break

            if use_fw and alpha == 0.0:
                logger.info(
                    "FW no improvement at iteration %d (alpha=0), stopping", m,
                )
                break

            if state_change_norm < 1e-6:
                logger.info(
                    "%s converged at iteration %d: state change norm=%.2e",
                    method_label, m, state_change_norm,
                )
                break

        return engine, msa_results

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
        max_od: int = 100_000,
        max_rounds: int = 0,
        gap_threshold: float = 0.001,
        method: str = "msa",
    ) -> HillClimberResult:
        """Run assignment over ordered trip batches with convergence.

        Trips are grouped by departure-time bin using ``config.bin_width_s``.
        If ``max_batch_size`` is provided, each bin is further micro-batched and
        loaded sequentially. Each batch:

        1. Routes on the current network state
        2. Accumulates additional density/volume onto the shared network state
        3. Recomputes VDF speeds
        4. Re-customizes OSRM and reloads the engine

        After the greedy load, convergence iterations run when
        ``max_rounds > 0``. Each iteration:

        1. Freezes the current link-density state
        2. Routes all demand (or up to ``max_od`` OD pairs for MSA) on the
           frozen network
        3. Blends auxiliary density with current state: k = (1-α)k + αk̂
           - MSA: α = 1/(m+1) (diminishing step, guaranteed convergence)
           - FW:  α from Beckmann line search (optimal step)
        4. Re-customizes and reloads the engine

        Parameters
        ----------
        max_od : int
            Maximum OD pairs routed per convergence iteration.  MSA will
            sub-sample and scale up when demand exceeds this.  FW always
            requires full-pass; raises ``NotImplementedError`` if the trip
            list exceeds *max_od*.
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
        converge = max_rounds > 0
        n_records = len(all_trips)
        total_demand = sum(t.volume for t in all_trips)

        # FW requires full-pass routing — fail fast if network too large.
        if method == "fw" and converge and n_records > max_od:
            raise NotImplementedError(
                f"Frank-Wolfe requires full-pass routing but network has "
                f"{n_records:,} trip-records (max_od={max_od:,}). "
                f"Use method='msa' for large-scale sampled assignment."
            )

        if converge:
            logger.info(
                "Started: %d trip-records (%.0f total demand), "
                "method=%s, max_od=%d, rounds=%d, n_threads=%d",
                n_records, total_demand,
                method, max_od, max_rounds, loop.config.n_threads,
            )
        else:
            logger.info(
                "Started: %d trip-records (%.0f total demand), "
                "greedy-only, n_threads=%d",
                n_records, total_demand, loop.config.n_threads,
            )
        engine = loop._create_engine()

        snapped_trips = loop._snap_trips(engine, all_trips)
        snapped_stream = TripStreamAdapter(snapped_trips, sort_by_departure=False)
        state = NetworkState.empty()
        if state_patch:
            state_patch(state)
        planned_load_steps = len(
            self.iter_time_slices(snapped_trips, max_batch_size=max_batch_size)
        )

        batch_results: List[HillClimberBatchResult] = []
        od_ledger = ODLedger()
        running_network_tstt = 0.0
        accumulated_volume = np.zeros(0, dtype=np.float64)
        for batch in snapped_stream.iter_time_slices(
            bin_width_s=self.config.bin_width_s,
            max_batch_size=max_batch_size,
        ):
            t_route = time.monotonic()
            if hasattr(loop, "_route_and_accumulate_with_paths"):
                batch_density, batch_volume, batch_tstt, routed_paths = (
                    loop._route_and_accumulate_with_paths(
                        engine,
                        batch.trips,
                        state,
                    )
                )
            else:
                batch_density, batch_volume, batch_tstt = loop._route_and_accumulate(
                    engine,
                    batch.trips,
                    state,
                )
                routed_paths = []
            route_time = time.monotonic() - t_route

            if len(batch_density) < state.n_edges:
                pad = state.n_edges - len(batch_density)
                batch_density = np.append(batch_density, np.zeros(pad))
                batch_volume = np.append(batch_volume, np.zeros(pad))
            if state_patch:
                state_patch(state)
            loop.smoother.build_adjacency(state.edge_ids, state.length_m)

            t_ledger = time.monotonic()
            self._append_od_entries(od_ledger, batch.trips, routed_paths)
            ledger_time = time.monotonic() - t_ledger
            running_network_tstt += float(batch_tstt)

            state.density_vpkm = np.clip(
                state.density_vpkm + batch_density,
                0.0,
                state.jam_density,
            )
            if len(accumulated_volume) < state.n_edges:
                accumulated_volume = np.append(
                    accumulated_volume,
                    np.zeros(state.n_edges - len(accumulated_volume)),
                )
            accumulated_volume += batch_volume
            loop._update_state(state)

            t_cust = time.monotonic()
            csv_path = loop.writer.write_from_state(state, only_changed=True)
            logger.info("Customizing OSRM (batch %d)...", batch.batch_index)
            osrm_module.customize(
                self.base_path,
                segment_speed_file=str(csv_path),
                verbosity="ERROR",
            )
            customize_time = time.monotonic() - t_cust

            t_engine = time.monotonic()
            del engine
            engine = loop._create_engine()
            engine_time = time.monotonic() - t_engine

            max_k_over_kj = float(np.max(state.density_vpkm / np.maximum(state.jam_density, 1e-9)))
            median_k_over_kj = float(np.median(state.density_vpkm / np.maximum(state.jam_density, 1e-9)))

            # Speed and travel time statistics over active links
            active = state.density_vpkm > 0
            active_speeds = state.speed_kmh[active] if np.any(active) else state.speed_kmh
            mean_speed = float(np.mean(active_speeds))
            median_speed = float(np.median(active_speeds))
            max_speed = float(np.max(active_speeds))
            min_speed = float(np.min(active_speeds))
            post_speed_ms = np.maximum(state.speed_kmh / 3.6, 0.001)
            post_link_tt_s = state.length_m / post_speed_ms
            active_tt = post_link_tt_s[active] if np.any(active) else post_link_tt_s
            median_tt = float(np.median(active_tt))
            max_tt = float(np.max(active_tt))

            batch_result = HillClimberBatchResult(
                batch_index=batch.batch_index,
                departure_bin=batch.departure_bin,
                n_trips=len(batch.trips),
                batch_tstt=batch_tstt,
                network_tstt=running_network_tstt,
                route_time_s=route_time,
                customize_time_s=customize_time,
                engine_time_s=engine_time,
                max_k_over_kj=max_k_over_kj,
                median_k_over_kj=median_k_over_kj,
                mean_speed_kmh=mean_speed,
                median_speed_kmh=median_speed,
                max_speed_kmh=max_speed,
                min_speed_kmh=min_speed,
                median_tt_s=median_tt,
                max_tt_s=max_tt,
            )
            batch_results.append(batch_result)

            route_rate = (
                len(batch.trips) / route_time
                if route_time > 1e-9 else float("inf")
            )
            logger.info(
                "Load %d/%d: %s routes in %.1fs (%s routes/s) "
                u"v\u0305=%.1f km/h k/kj=%.2f (max %.2f) ledger=%.1fs",
                batch.batch_index + 1,
                planned_load_steps,
                f"{len(batch.trips):,}",
                route_time,
                f"{route_rate:,.0f}",
                median_speed,
                median_k_over_kj,
                max_k_over_kj,
                ledger_time,
            )

            if progress_callback:
                progress_callback(batch_result)

        msa_results: List[MSAIterationResult] = []
        if converge:
            engine, msa_results = self._run_convergence(
                engine=engine,
                loop=loop,
                state=state,
                trips=snapped_trips,
                initial_volume=accumulated_volume,
                state_patch=state_patch,
                progress_callback=progress_callback,
                max_od=max_od,
                max_rounds=max_rounds,
                gap_threshold=gap_threshold,
                method=method,
            )

        total_time = time.monotonic() - started
        method_label = method.upper()
        if msa_results:
            logger.info(
                "Complete: %d load steps, %d %s iterations, %.1fs",
                len(batch_results), len(msa_results), method_label, total_time,
            )
        else:
            logger.info(
                "Complete: %d load steps, greedy only, %.1fs",
                len(batch_results), total_time,
            )

        return HillClimberResult(
            network_state=state,
            batch_results=batch_results,
            total_time_s=total_time,
            n_trips=len(snapped_trips),
            od_ledger=od_ledger,
            msa_results=msa_results,
        )

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



# Backward compatibility aliases
MatrixAssignmentSolver = TrafficAssignmentSolver
MatrixFreeHillClimber = TrafficAssignmentSolver
