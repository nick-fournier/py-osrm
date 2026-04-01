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
from copy import deepcopy
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

_REFINEMENT_FOCUS_SHARE = 0.30
_REFINEMENT_TOP_SHARE = 0.25
_REFINEMENT_ALPHA_MIN = 0.10
_REFINEMENT_ALPHA_MAX = 0.80
_REFINEMENT_TARGET_REL_EXCESS = 0.25
_REFINEMENT_STABLE_ROUNDS = 2
_REFINEMENT_MIN_ROUTE_FRACTION = 1e-9
_REFINEMENT_DISCOVERY_REL_TOL = 0.05
_REFINEMENT_ACCEPT_TOL = 1e-6
_REFINEMENT_BACKTRACK_SHRINK = 0.5
_REFINEMENT_MAX_BACKTRACK_STEPS = 6

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

        scale = old_total / new_total if old_total > 0 else 0.0
        for route in existing.routes:
            route.volume_fraction *= scale

        added_scale = entry.total_volume / new_total
        for route in entry.routes:
            route.volume_fraction *= added_scale
            existing.routes.append(route)

        existing.total_volume = new_total
        existing.departure_time_s = min(existing.departure_time_s, entry.departure_time_s)
        existing.compress_routes()

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]

    @property
    def total_volume(self) -> float:
        return float(sum(entry.total_volume for entry in self.entries))


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
    network_tstt: float
    route_time_s: float
    customize_time_s: float
    engine_time_s: float
    max_k_over_kj: float
    mean_speed_kmh: float


@dataclass
class RefinementRoundResult:
    """Metrics for one sampled path-set refinement round."""

    round_index: int
    sampled_pairs: int
    accepted_updates: int
    sampled_gap: Optional[float]
    network_tstt: float
    worst_score: float
    route_time_s: float
    customize_time_s: float
    engine_time_s: float
    round_time_s: float
    max_k_over_kj: float
    mean_speed_kmh: float


@dataclass
class HillClimberResult:
    """Final result of the wrapper-side hill-climber MVP."""

    network_state: Optional[NetworkState]
    batch_results: List[HillClimberBatchResult]
    total_time_s: float
    n_trips: int
    od_ledger: Optional[ODLedger] = None
    refinement_results: List[RefinementRoundResult] = field(default_factory=list)

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

    @staticmethod
    def _route_signature(route: RouteAssignment) -> Tuple[int, ...]:
        return tuple(route.edge_indices)

    def _current_route_cost(self, route: RouteAssignment, state: NetworkState) -> float:
        cost_s = 0.0
        for edge_idx in route.edge_indices:
            if edge_idx < 0 or edge_idx >= state.n_edges:
                continue
            speed_kmh = max(state.speed_kmh[edge_idx], self.config.vdf_min_speed_kmh)
            cost_s += state.length_m[edge_idx] / (speed_kmh / 3.6)
        return cost_s

    def _current_entry_cost(self, entry: ODLedgerEntry, state: NetworkState) -> float:
        if not entry.routes:
            return 0.0
        return float(sum(
            route.volume_fraction * self._current_route_cost(route, state)
            for route in entry.routes
        ))

    def _sample_od_indices(
        self,
        od_ledger: ODLedger,
        sample_rate: float,
        focus_indices: Sequence[int],
    ) -> List[int]:
        n_total = len(od_ledger)
        if n_total == 0:
            return []
        sample_size = min(n_total, max(1, int(np.ceil(sample_rate * n_total))))
        selected: set[int] = set()

        focus_pool = [idx for idx in focus_indices if 0 <= idx < n_total]
        if focus_pool:
            n_focus = min(len(focus_pool), int(round(sample_size * _REFINEMENT_FOCUS_SHARE)))
            if n_focus > 0:
                selected.update(random.sample(focus_pool, n_focus))

        if len(selected) < sample_size:
            remaining = [idx for idx in range(n_total) if idx not in selected]
            need = min(sample_size - len(selected), len(remaining))
            if need > 0:
                selected.update(random.sample(remaining, need))
        return sorted(selected)

    def _sampled_table_costs(
        self,
        engine,
        entries: Sequence[ODLedgerEntry],
        *,
        max_pairs_per_call: int = 512,
    ) -> List[Optional[float]]:
        """Compute shortest-path durations for sampled ledger entries via Table.

        Chunk the sampled set so each Table call stays modest. This keeps the
        diagnostic sublinear without relying on one enormous all-sample matrix.
        """
        if not entries:
            return []

        result: List[Optional[float]] = []
        for start in range(0, len(entries), max_pairs_per_call):
            chunk = entries[start:start + max_pairs_per_call]
            coordinates: List[Tuple[float, float]] = []
            coord_to_index: dict[Tuple[float, float], int] = {}

            def coord_index(coord: Tuple[float, float]) -> int:
                idx = coord_to_index.get(coord)
                if idx is None:
                    idx = len(coordinates)
                    coord_to_index[coord] = idx
                    coordinates.append(coord)
                return idx

            source_indices = [coord_index(entry.origin) for entry in chunk]
            destination_indices = [coord_index(entry.destination) for entry in chunk]
            unique_sources = list(dict.fromkeys(source_indices))
            unique_destinations = list(dict.fromkeys(destination_indices))
            dest_position = {
                coord_idx: pos for pos, coord_idx in enumerate(unique_destinations)
            }

            table = engine.Table(
                coordinates=coordinates,
                sources=unique_sources,
                destinations=unique_destinations,
                annotations=["duration"],
                skip_waypoints=True,
            )
            durations = table.get("durations", [])
            rows_by_source = {
                source_coord_idx: durations[local_idx]
                for local_idx, source_coord_idx in enumerate(unique_sources)
                if local_idx < len(durations)
            }

            for src_idx, dst_idx in zip(source_indices, destination_indices):
                row = rows_by_source.get(src_idx)
                if row is None:
                    result.append(None)
                    continue
                dst_pos = dest_position[dst_idx]
                duration = row[dst_pos] if dst_pos < len(row) else None
                result.append(None if duration is None else float(duration))
        return result

    def _apply_density_scale(
        self,
        state: NetworkState,
        route: RouteAssignment,
        scale: float,
    ) -> None:
        for edge_idx, density_delta in zip(
            route.edge_indices, route.density_contribution,
        ):
            state.density_vpkm[edge_idx] += scale * density_delta

    def _route_base_density(self, route: RouteAssignment) -> List[float]:
        if route.volume_fraction <= _REFINEMENT_MIN_ROUTE_FRACTION:
            return list(route.density_contribution)
        inv = 1.0 / route.volume_fraction
        return [value * inv for value in route.density_contribution]

    def _entry_route_costs(
        self,
        entry: ODLedgerEntry,
        state: NetworkState,
    ) -> List[float]:
        return [self._current_route_cost(route, state) for route in entry.routes]

    def _entry_weighted_cost_from_costs(
        self,
        entry: ODLedgerEntry,
        route_costs: Sequence[float],
    ) -> float:
        if not route_costs:
            return 0.0
        return float(sum(
            route.volume_fraction * route_costs[idx]
            for idx, route in enumerate(entry.routes)
        ))

    def _entry_changed(
        self,
        before: ODLedgerEntry,
        after: ODLedgerEntry,
        *,
        fraction_tol: float = 1e-9,
    ) -> bool:
        before_map = {
            self._route_signature(route): route.volume_fraction
            for route in before.routes
        }
        after_map = {
            self._route_signature(route): route.volume_fraction
            for route in after.routes
        }
        if before_map.keys() != after_map.keys():
            return True
        return any(
            abs(before_map[signature] - after_map[signature]) > fraction_tol
            for signature in before_map
        )

    def _evaluate_sample(
        self,
        entries: Sequence[ODLedgerEntry],
        shortest_costs: Sequence[Optional[float]],
        state: NetworkState,
    ) -> Tuple[Optional[float], float, float, List[Tuple[float, float, int]]]:
        numerator = 0.0
        denominator = 0.0
        scored_entries: List[Tuple[float, float, int]] = []

        for local_idx, (entry, shortest_cost) in enumerate(
            zip(entries, shortest_costs),
        ):
            if shortest_cost is None or shortest_cost <= 0.0:
                entry.current_gap = None
                continue
            used_cost = self._current_entry_cost(entry, state)
            excess = max(0.0, used_cost - shortest_cost)
            relative_excess = excess / shortest_cost
            entry.current_gap = relative_excess
            numerator += entry.total_volume * excess
            denominator += entry.total_volume * shortest_cost
            if excess > 0.0:
                scored_entries.append(
                    (entry.total_volume * excess, relative_excess, local_idx)
                )

        sampled_gap = (numerator / denominator) if denominator > 0.0 else None
        return sampled_gap, numerator, denominator, scored_entries

    def _apply_entry_to_state(
        self,
        state: NetworkState,
        entry: ODLedgerEntry,
        scale: float,
    ) -> None:
        for route in entry.routes:
            self._apply_density_scale(state, route, scale)

    def _copy_entry_from(
        self,
        dest: ODLedgerEntry,
        src: ODLedgerEntry,
    ) -> None:
        dest.total_volume = src.total_volume
        dest.departure_time_s = src.departure_time_s
        dest.current_gap = src.current_gap
        dest.refinement_visits = src.refinement_visits
        dest.routes = deepcopy(src.routes)

    def _scaled_entry_proposal(
        self,
        entry: ODLedgerEntry,
        proposal: ODLedgerEntry,
        step_scale: float,
    ) -> ODLedgerEntry:
        """Interpolate between the current entry and a full proposal."""
        if step_scale <= 0.0:
            return deepcopy(entry)
        if step_scale >= 1.0:
            return deepcopy(proposal)

        candidate = deepcopy(entry)
        before = {
            self._route_signature(route): route for route in entry.routes
        }
        after = {
            self._route_signature(route): route for route in proposal.routes
        }
        signatures = list(dict.fromkeys(list(before.keys()) + list(after.keys())))

        candidate.routes = []
        for signature in signatures:
            before_route = before.get(signature)
            after_route = after.get(signature)
            before_fraction = 0.0 if before_route is None else before_route.volume_fraction
            after_fraction = 0.0 if after_route is None else after_route.volume_fraction
            new_fraction = before_fraction + step_scale * (after_fraction - before_fraction)
            if new_fraction <= _REFINEMENT_MIN_ROUTE_FRACTION:
                continue

            source_route = after_route if after_route is not None else before_route
            if source_route is None:
                continue
            base_density = self._route_base_density(source_route)
            assigned_cost_s = (
                after_route.assigned_cost_s
                if after_route is not None else
                before_route.assigned_cost_s
            )
            candidate.routes.append(RouteAssignment(
                edge_indices=list(source_route.edge_indices),
                density_contribution=[value * new_fraction for value in base_density],
                volume_fraction=new_fraction,
                assigned_cost_s=assigned_cost_s,
            ))

        candidate.compress_routes()
        total_fraction = sum(route.volume_fraction for route in candidate.routes)
        if total_fraction > 0.0:
            for route in candidate.routes:
                scale = 1.0 / total_fraction
                route.volume_fraction *= scale
                route.density_contribution = [
                    value * scale for value in route.density_contribution
                ]
        candidate.current_gap = proposal.current_gap
        candidate.refinement_visits = proposal.refinement_visits
        return candidate

    def _refresh_state(
        self,
        loop: AssignmentLoop,
        state: NetworkState,
        state_patch,
    ) -> None:
        state.density_vpkm = np.clip(
            state.density_vpkm,
            0.0,
            state.jam_density,
        )
        if state_patch:
            state_patch(state)
        loop.smoother.build_adjacency(state.edge_ids, state.length_m)
        loop._update_state(state)

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

    def _discovery_needed(
        self,
        best_known_cost: float,
        shortest_cost: float,
    ) -> bool:
        if shortest_cost <= 0.0:
            return False
        return best_known_cost > shortest_cost * (1.0 + _REFINEMENT_DISCOVERY_REL_TOL)

    def _propose_path_swap(
        self,
        entry: ODLedgerEntry,
        state: NetworkState,
        shortest_cost: float,
        discovered_path: Optional[RoutedTripPath],
    ) -> Tuple[ODLedgerEntry, bool]:
        """Propose a bounded share swap for one OD using known paths first.

        Returns
        -------
        proposal : ODLedgerEntry
            Proposed updated path set and shares.
        used_discovery : bool
            Whether a newly routed path was actually introduced.
        """
        proposal = deepcopy(entry)
        route_costs = self._entry_route_costs(proposal, state)
        if not route_costs:
            return proposal, False

        best_known_idx = int(np.argmin(route_costs))
        best_known_cost = route_costs[best_known_idx]
        target_is_new = False
        target_edge_indices = list(proposal.routes[best_known_idx].edge_indices)
        target_base_density = self._route_base_density(proposal.routes[best_known_idx])
        target_cost = best_known_cost

        if discovered_path is not None:
            discovered_signature = tuple(discovered_path.edge_indices)
            known_signatures = {
                self._route_signature(route) for route in proposal.routes
            }
            if (
                discovered_signature not in known_signatures
                and float(discovered_path.duration_s) + 1e-9 < best_known_cost
            ):
                target_is_new = True
                target_edge_indices = list(discovered_path.edge_indices)
                target_base_density = list(discovered_path.density_contribution)
                target_cost = float(discovered_path.duration_s)

        used_cost = self._entry_weighted_cost_from_costs(proposal, route_costs)
        improvement = used_cost - target_cost
        if improvement <= 0.0:
            return proposal, False

        visit_scale = 1.0 / max(1, proposal.refinement_visits + 1)
        rel_improvement = improvement / max(shortest_cost, 1e-9)
        alpha = min(
            _REFINEMENT_ALPHA_MAX,
            max(
                _REFINEMENT_ALPHA_MIN,
                (rel_improvement / _REFINEMENT_TARGET_REL_EXCESS) * visit_scale,
            ),
        )

        updated_routes: List[RouteAssignment] = []
        target_signature = tuple(target_edge_indices)
        target_added = False
        for route in proposal.routes:
            signature = self._route_signature(route)
            if signature == target_signature:
                new_fraction = route.volume_fraction + alpha * (1.0 - route.volume_fraction)
            else:
                new_fraction = route.volume_fraction * (1.0 - alpha)
            if new_fraction <= _REFINEMENT_MIN_ROUTE_FRACTION:
                continue

            base_density = (
                target_base_density
                if signature == target_signature else
                self._route_base_density(route)
            )
            updated_routes.append(RouteAssignment(
                edge_indices=list(route.edge_indices),
                density_contribution=[value * new_fraction for value in base_density],
                volume_fraction=new_fraction,
                assigned_cost_s=target_cost if signature == target_signature else route.assigned_cost_s,
            ))
            if signature == target_signature:
                target_added = True

        if not target_added:
            updated_routes.append(RouteAssignment(
                edge_indices=list(target_edge_indices),
                density_contribution=[value * alpha for value in target_base_density],
                volume_fraction=alpha,
                assigned_cost_s=target_cost,
            ))
            target_is_new = True

        proposal.routes = updated_routes
        proposal.compress_routes()
        total_fraction = sum(route.volume_fraction for route in proposal.routes)
        if total_fraction > 0.0:
            for route in proposal.routes:
                scale = 1.0 / total_fraction
                route.volume_fraction *= scale
                route.density_contribution = [
                    value * scale for value in route.density_contribution
                ]
        return proposal, target_is_new

    def _run_sampled_refinement(
        self,
        *,
        engine,
        loop: AssignmentLoop,
        state: NetworkState,
        od_ledger: ODLedger,
        state_patch,
        progress_callback,
        sample_rate: float,
        max_rounds: int,
        gap_threshold: float,
    ) -> Tuple[object, List[RefinementRoundResult]]:
        """Run sampled path-set refinement rounds on top of the greedy pass."""
        refinement_results: List[RefinementRoundResult] = []
        focus_indices: List[int] = []
        stable_rounds = 0

        for round_idx in range(max_rounds):
            round_start = time.monotonic()
            round_base_density = state.density_vpkm.copy()
            sample_indices = self._sample_od_indices(
                od_ledger, sample_rate, focus_indices,
            )
            sampled_entries = [od_ledger[idx] for idx in sample_indices]
            shortest_costs = self._sampled_table_costs(engine, sampled_entries)
            sampled_gap, sampled_excess, _, scored_entries = self._evaluate_sample(
                sampled_entries, shortest_costs, state,
            )
            if not scored_entries:
                logger.info(
                    "Refinement complete: no sampled offenders after %d round(s)",
                    round_idx,
                )
                break

            scored_entries.sort(reverse=True)
            refinement_count = max(1, int(np.ceil(len(scored_entries) * _REFINEMENT_TOP_SHARE)))
            selected_meta = scored_entries[:refinement_count]
            selected_globals = [
                sample_indices[local_idx] for _, _, local_idx in selected_meta
            ]
            proposals: dict[int, ODLedgerEntry] = {}
            discovery_batch: List[Tuple[int, ODLedgerEntry, float]] = []
            route_time = 0.0
            customize_time = 0.0
            engine_time = 0.0

            for _, _, local_idx in selected_meta:
                global_idx = sample_indices[local_idx]
                entry = sampled_entries[local_idx]
                shortest_cost = shortest_costs[local_idx]
                if shortest_cost is None or shortest_cost <= 0.0:
                    continue
                route_costs = self._entry_route_costs(entry, state)
                if not route_costs:
                    continue
                best_known_cost = min(route_costs)
                if self._discovery_needed(best_known_cost, shortest_cost):
                    discovery_batch.append((global_idx, entry, shortest_cost))
                    continue

                proposal, _ = self._propose_path_swap(
                    entry, state, shortest_cost, None,
                )
                if self._entry_changed(entry, proposal):
                    proposal.refinement_visits = entry.refinement_visits + 1
                    proposals[global_idx] = proposal

            if discovery_batch:
                for _, entry, _ in discovery_batch:
                    self._apply_entry_to_state(state, entry, -1.0)
                self._refresh_state(loop, state, state_patch)
                engine, dt_cust, dt_engine = self._customize_and_reload(
                    loop, state, engine,
                )
                customize_time += dt_cust
                engine_time += dt_engine

                discovery_trips = [
                    DemandTrip(
                        origin=entry.origin,
                        destination=entry.destination,
                        volume=entry.total_volume,
                        departure_time_s=entry.departure_time_s,
                    )
                    for _, entry, _ in discovery_batch
                ]
                t_route = time.monotonic()
                _, _, _, routed_paths = loop._route_and_accumulate_with_paths(
                    engine, discovery_trips, state,
                )
                route_time += time.monotonic() - t_route
                path_by_trip = {path.trip_index: path for path in routed_paths}

                for discovery_idx, (global_idx, entry, shortest_cost) in enumerate(
                    discovery_batch,
                ):
                    proposal, _ = self._propose_path_swap(
                        entry,
                        state,
                        shortest_cost,
                        path_by_trip.get(discovery_idx),
                    )
                    if self._entry_changed(entry, proposal):
                        proposal.refinement_visits = entry.refinement_visits + 1
                        proposals[global_idx] = proposal

            if len(round_base_density) < state.n_edges:
                round_base_density = np.append(
                    round_base_density,
                    np.zeros(state.n_edges - len(round_base_density)),
                )
            state.density_vpkm = round_base_density.copy()
            self._refresh_state(loop, state, state_patch)

            if not proposals:
                if discovery_batch:
                    engine, dt_cust, dt_engine = self._customize_and_reload(
                        loop, state, engine,
                    )
                    customize_time += dt_cust
                    engine_time += dt_engine
                max_k_over_kj = float(
                    np.max(state.density_vpkm / np.maximum(state.jam_density, 1e-9))
                )
                mean_speed = float(np.mean(state.speed_kmh))
                network_tstt = self._network_tstt_from_od_ledger(od_ledger, state)
                round_result = RefinementRoundResult(
                    round_index=round_idx + 1,
                    sampled_pairs=len(sample_indices),
                    accepted_updates=0,
                    sampled_gap=sampled_gap,
                    network_tstt=network_tstt,
                    worst_score=float(selected_meta[0][0]) if selected_meta else 0.0,
                    route_time_s=route_time,
                    customize_time_s=customize_time,
                    engine_time_s=engine_time,
                    round_time_s=time.monotonic() - round_start,
                    max_k_over_kj=max_k_over_kj,
                    mean_speed_kmh=mean_speed,
                )
                refinement_results.append(round_result)
                if progress_callback:
                    progress_callback(round_result)
                logger.info(
                    "Refinement halted at round %d: no improving path-set updates found",
                    round_idx + 1,
                )
                break

            improved = False
            accepted_updates = 0
            accepted_proposals: dict[int, ODLedgerEntry] = {}
            step_scale = 1.0

            for _ in range(_REFINEMENT_MAX_BACKTRACK_STEPS):
                candidate_proposals = {
                    global_idx: self._scaled_entry_proposal(
                        od_ledger[global_idx], proposal, step_scale,
                    )
                    for global_idx, proposal in proposals.items()
                }

                if len(round_base_density) < state.n_edges:
                    round_base_density = np.append(
                        round_base_density,
                        np.zeros(state.n_edges - len(round_base_density)),
                    )
                state.density_vpkm = round_base_density.copy()
                self._refresh_state(loop, state, state_patch)

                for global_idx in selected_globals:
                    candidate = candidate_proposals.get(global_idx)
                    if candidate is None:
                        continue
                    self._apply_entry_to_state(state, od_ledger[global_idx], -1.0)
                    self._apply_entry_to_state(state, candidate, 1.0)

                self._refresh_state(loop, state, state_patch)
                engine, dt_cust, dt_engine = self._customize_and_reload(
                    loop, state, engine,
                )
                customize_time += dt_cust
                engine_time += dt_engine

                evaluation_entries = [
                    candidate_proposals.get(global_idx, od_ledger[global_idx])
                    for global_idx in sample_indices
                ]
                post_shortest_costs = self._sampled_table_costs(engine, evaluation_entries)
                post_gap, post_excess, _, _ = self._evaluate_sample(
                    evaluation_entries, post_shortest_costs, state,
                )

                improved = post_excess < sampled_excess * (1.0 - _REFINEMENT_ACCEPT_TOL)
                if improved:
                    accepted_updates = len(candidate_proposals)
                    accepted_proposals = candidate_proposals
                    sampled_gap = post_gap
                    break

                step_scale *= _REFINEMENT_BACKTRACK_SHRINK

            focus_limit = max(len(proposals) * 2, int(len(sample_indices) * _REFINEMENT_FOCUS_SHARE))
            focus_indices = [
                sample_indices[local_idx]
                for _, _, local_idx in scored_entries[:focus_limit]
            ]

            if improved:
                for global_idx, proposal in accepted_proposals.items():
                    self._copy_entry_from(od_ledger[global_idx], proposal)
            else:
                accepted_updates = 0
                if len(round_base_density) < state.n_edges:
                    round_base_density = np.append(
                        round_base_density,
                        np.zeros(state.n_edges - len(round_base_density)),
                    )
                state.density_vpkm = round_base_density.copy()
                self._refresh_state(loop, state, state_patch)
                engine, dt_cust, dt_engine = self._customize_and_reload(
                    loop, state, engine,
                )
                customize_time += dt_cust
                engine_time += dt_engine

            network_tstt = self._network_tstt_from_od_ledger(od_ledger, state)

            if sampled_gap is not None and sampled_gap < gap_threshold:
                stable_rounds += 1
            else:
                stable_rounds = 0

            max_k_over_kj = float(
                np.max(state.density_vpkm / np.maximum(state.jam_density, 1e-9))
            )
            mean_speed = float(np.mean(state.speed_kmh))
            worst_score = float(selected_meta[0][0]) if selected_meta else 0.0
            round_result = RefinementRoundResult(
                round_index=round_idx + 1,
                sampled_pairs=len(sample_indices),
                accepted_updates=accepted_updates,
                sampled_gap=sampled_gap,
                network_tstt=network_tstt,
                worst_score=worst_score,
                route_time_s=route_time,
                customize_time_s=customize_time,
                engine_time_s=engine_time,
                round_time_s=time.monotonic() - round_start,
                max_k_over_kj=max_k_over_kj,
                mean_speed_kmh=mean_speed,
            )
            refinement_results.append(round_result)

            gap_str = (
                f"{sampled_gap:.6f}" if sampled_gap is not None else "n/a"
            )
            if improved:
                logger.info(
                    "R%d: sample=%d updated=%d gap=%s TSTT=%.0f speed=%.1f km/h %.1fs",
                    round_result.round_index,
                    round_result.sampled_pairs,
                    round_result.accepted_updates,
                    gap_str,
                    round_result.network_tstt,
                    round_result.mean_speed_kmh,
                    round_result.round_time_s,
                )
            else:
                logger.info(
                    "R%d rejected: sample=%d gap=%s TSTT=%.0f speed=%.1f km/h %.1fs",
                    round_result.round_index,
                    round_result.sampled_pairs,
                    gap_str,
                    round_result.network_tstt,
                    round_result.mean_speed_kmh,
                    round_result.round_time_s,
                )

            if progress_callback:
                progress_callback(round_result)

            if not improved:
                break

            if sampled_gap is not None and stable_rounds >= _REFINEMENT_STABLE_ROUNDS:
                break

        return engine, refinement_results

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
        sample_rate: float = 0.0,
        max_rounds: int = 0,
        gap_threshold: float = 0.01,
    ) -> HillClimberResult:
        """Run a stateful wrapper-side hill-climber over ordered trip batches.

        Trips are grouped by departure-time bin using ``config.bin_width_s``.
        If ``max_batch_size`` is provided, each bin is further micro-batched and
        loaded sequentially. Each batch:

        1. Routes on the current network state
        2. Accumulates additional density/volume onto the shared network state
        3. Recomputes VDF speeds
        4. Re-customizes OSRM and reloads the engine

        After the greedy load, sampled path-set refinement optionally runs when
        ``sample_rate > 0`` and ``max_rounds > 0``. Each refinement round:

        1. Samples a subset of the OD ledger
        2. Uses Table to diagnose current shortest-path costs
        3. Rebalances among known paths and discovers new paths only when needed
        4. Re-customizes and reloads once per accepted round

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
        sampled_mode = sample_rate > 0.0 and max_rounds > 0
        n_records = len(all_trips)
        total_demand = sum(t.volume for t in all_trips)
        # Unique ODs: count from first time-slice to avoid iterating all records
        load_steps = max(1, round(1.0 / sample_rate)) if sample_rate > 0 else 1
        unique_ods = n_records // load_steps if load_steps > 1 else n_records
        if sampled_mode:
            logger.info(
                "HC started: %d trip-records (~%d unique ODs, %.0f total demand), "
                "sample_rate=%.2f, rounds=%d",
                n_records, unique_ods, total_demand,
                sample_rate, max_rounds,
            )
        else:
            logger.info(
                "HC started: %d trip-records (~%d unique ODs, %.0f total demand), "
                "greedy-only",
                n_records, unique_ods, total_demand,
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

            self._append_od_entries(od_ledger, batch.trips, routed_paths)
            running_network_tstt += float(batch_tstt)

            state.density_vpkm = np.clip(
                state.density_vpkm + batch_density,
                0.0,
                state.jam_density,
            )
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
            mean_speed = float(np.mean(state.speed_kmh))
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
                mean_speed_kmh=mean_speed,
            )
            batch_results.append(batch_result)

            route_rate = (
                len(batch.trips) / route_time
                if route_time > 1e-9 else float("inf")
            )
            logger.info(
                "Load %d/%d: %.0f routes/s speed=%.1f km/h k/kj=%.2f",
                batch.batch_index + 1,
                planned_load_steps,
                route_rate,
                mean_speed,
                max_k_over_kj,
            )

            if progress_callback:
                progress_callback(batch_result)

        refinement_results: List[RefinementRoundResult] = []
        if sampled_mode:
            engine, refinement_results = self._run_sampled_refinement(
                engine=engine,
                loop=loop,
                state=state,
                od_ledger=od_ledger,
                state_patch=state_patch,
                progress_callback=progress_callback,
                sample_rate=sample_rate,
                max_rounds=max_rounds,
                gap_threshold=gap_threshold,
            )

        total_time = time.monotonic() - started
        if refinement_results:
            logger.info(
                "HC complete: %d load steps, %d refinement rounds, %.1fs",
                len(batch_results), len(refinement_results), total_time,
            )
        else:
            logger.info(
                "HC complete: %d load steps, greedy only, %.1fs",
                len(batch_results), total_time,
            )

        return HillClimberResult(
            network_state=state,
            batch_results=batch_results,
            total_time_s=total_time,
            n_trips=len(snapped_trips),
            od_ledger=od_ledger,
            refinement_results=refinement_results,
        )

    def _network_tstt_from_od_ledger(
        self,
        od_ledger: ODLedger,
        state: NetworkState,
    ) -> float:
        total_tstt = 0.0
        for entry in od_ledger:
            total_tstt += float(entry.total_volume) * self._current_entry_cost(entry, state)
        return float(total_tstt)

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
