"""Traffic assignment loop orchestrator.

Implements the iterative equilibrium assignment cycle:
  Route → Decompose → Accumulate → VDF → CSV → Customize → Reload

Supports Method of Successive Averages (MSA) for convergence.

See docs/traffic_assignment_design.md §4 and §7 for full specification.
"""

from __future__ import annotations

import enum
import logging
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import osrm as osrm_module
from osrm.assignment.density_smoothing import DensitySmoothing, DensitySmoothingConfig
from osrm.assignment.fractional_loading import FractionalLoader
from osrm.assignment.network_state import NetworkState
from osrm.assignment.od_matrix import DemandTrip, ODMatrixAdapter
from osrm.assignment.segment_speed_writer import SegmentSpeedWriter
from osrm.assignment.trip_stream import TripBatch, TripStreamAdapter
from osrm.assignment.vdf import BiParabolicVDF

logger = logging.getLogger(__name__)

_VERBOSITY_LEVELS = {
    "NONE": logging.WARNING + 10,   # effectively silent
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


class StopReason(enum.Enum):
    """Why the assignment loop terminated."""
    CONVERGED = "converged"
    STAGNATED = "stagnated"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class AssignmentConfig:
    """Configuration for the assignment loop.

    Parameters
    ----------
    method : str
        Convergence method: ``"msa"`` (Method of Successive Averages,
        fixed step 1/n) or ``"fw"`` (Frank-Wolfe with Beckmann line
        search).  Default ``"msa"``.
    fw_bisections : int
        Maximum bisection iterations for the FW line search.
    fw_line_search_tol : float
        Convergence tolerance for the FW line search bracket width.
    incremental_steps : tuple of float
        Demand fractions for incremental loading during the warm-up
        phase.  Each entry triggers one AON iteration at that fraction
        of full demand before the main FW/MSA loop begins.
        Default ``(0.25, 0.50, 0.75, 1.0)``.  Set to ``(1.0,)`` to
        disable (single full-demand loading, legacy behaviour).
    stagnation_tol : float
        Gap change threshold for stagnation detection.  If
        ``|gap[n] - gap[n-1]| < stagnation_tol`` for
        ``stagnation_window`` consecutive iterations, the loop stops
        with ``StopReason.STAGNATED``.  Set to 0 to disable.
    stagnation_window : int
        Number of consecutive near-constant gap iterations before
        declaring stagnation.
    """

    method: str = "msa"
    max_iterations: int = 50
    convergence_gap: float = 0.01
    bin_width_s: float = 3600.0
    min_speed_kmh: float = 1.0
    # OSRM stores speeds internally as integer decimetres/second
    # (0.1 m/s = 0.36 km/h resolution).  Its effective floor is
    # 0.3 m/s = 1.08 km/h — anything below is rounded up to this.
    # We match the VDF floor so that the cost OSRM routes on and the
    # cost the VDF reports are consistent for gridlocked links.
    vdf_min_speed_kmh: float = 1.08
    smoothing: DensitySmoothingConfig = field(
        default_factory=DensitySmoothingConfig
    )
    vdf_kc_ratio: float = 1.0 / 3.0
    # HCM standard: ~150 veh/km/lane (~6.7m spacing at standstill)
    default_jam_density_per_lane: float = 150.0
    default_n_lanes: int = 1
    speed_csv_dir: Optional[str] = None
    verbosity: str = "INFO"
    fw_bisections: int = 20
    fw_line_search_tol: float = 1e-6
    incremental_steps: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)
    stagnation_tol: float = 0.001
    stagnation_window: int = 3
    n_threads: int = -1  # -1 = cpu_count - 2; 0 = all cores; >0 = explicit cap

    def __post_init__(self):
        if self.n_threads == -1:
            self.n_threads = max(1, (os.cpu_count() or 4) - 2)


@dataclass
class IterationResult:
    """Metrics for one iteration (greedy loading or convergence)."""

    iteration: int
    phase: str = "convergence"          # "greedy", "incremental", or "convergence"
    alpha: float = 0.0                  # step size (1.0 for greedy)
    relative_gap: Optional[float] = None
    tstt: float = 0.0
    state_change_norm: float = 0.0
    n_oversaturated: int = 0
    max_k_over_kj: float = 0.0
    median_k_over_kj: float = 0.0
    mean_speed_kmh: float = 0.0
    median_speed_kmh: float = 0.0
    max_speed_kmh: float = 0.0
    min_speed_kmh: float = 0.0
    median_tt_s: float = 0.0
    max_tt_s: float = 0.0
    n_routes: int = 0
    route_time_s: float = 0.0
    customize_time_s: float = 0.0
    engine_time_s: float = 0.0
    iteration_time_s: float = 0.0
    # Legacy field aliases for backward compatibility
    aon_tstt: float = 0.0
    link_tstt: float = 0.0
    departure_bin: Optional[int] = None
    n_trips: int = 0
    batch_tstt: float = 0.0
    network_tstt: float = 0.0

    @property
    def batch_index(self) -> int:
        """Legacy alias for ``iteration``."""
        return self.iteration


@dataclass
class AssignmentResult:
    """Final result of the assignment loop."""

    converged: bool
    iterations: int
    final_gap: float
    network_state: NetworkState
    iteration_log: List[IterationResult]
    total_time_s: float
    stop_reason: StopReason = StopReason.MAX_ITERATIONS
    od_ledger: Optional[object] = None     # ODLedger for diagnostics
    n_trips: int = 0

    @property
    def batch_results(self) -> List[IterationResult]:
        """Legacy accessor: greedy-phase iterations."""
        return [r for r in self.iteration_log if r.phase == "greedy"]

    @property
    def msa_results(self) -> List[IterationResult]:
        """Legacy accessor: convergence-phase iterations."""
        return [r for r in self.iteration_log if r.phase == "convergence"]

    @property
    def n_batches(self) -> int:
        """Legacy accessor: number of greedy-phase batches."""
        return len(self.batch_results)

    def log_as_dict(self) -> Dict:
        """Convert iteration log to dict for plotting."""
        return {
            "iteration": [r.iteration for r in self.iteration_log],
            "phase": [r.phase for r in self.iteration_log],
            "relative_gap": [r.relative_gap for r in self.iteration_log],
            "tstt": [r.tstt for r in self.iteration_log],
            "max_density_delta": [r.state_change_norm for r in self.iteration_log],
            "step_size": [r.alpha for r in self.iteration_log],
            "mean_speed_kmh": [r.mean_speed_kmh for r in self.iteration_log],
            "median_speed_kmh": [r.median_speed_kmh for r in self.iteration_log],
            "min_speed_kmh": [r.min_speed_kmh for r in self.iteration_log],
            "max_speed_kmh": [r.max_speed_kmh for r in self.iteration_log],
            "median_tt_s": [r.median_tt_s for r in self.iteration_log],
            "max_tt_s": [r.max_tt_s for r in self.iteration_log],
            "route_time_s": [r.route_time_s for r in self.iteration_log],
            "customize_time_s": [r.customize_time_s for r in self.iteration_log],
            "engine_time_s": [r.engine_time_s for r in self.iteration_log],
        }


@dataclass
class RoutedTripPath:
    """Per-trip path data captured during routing.

    This is used by the matrix-free hill-climber to retain enough path-level
    information to support later selective healing without re-running a full
    all-trip assignment pass.
    """

    trip_index: int
    edge_indices: List[int]
    density_contribution: List[float]
    duration_s: float


@dataclass
class RouteAssignment:
    """One route fragment currently carrying some share of an OD's volume."""

    edge_indices: List[int]
    density_contribution: List[float]
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


class AssignmentSolver:
    """Unified traffic assignment solver.

    Supports both OD-matrix and matrix-free trip-stream workflows:

    - ``run()``: UE solver with incremental warm-up + MSA/FW convergence
    - ``run_stream()``: greedy time-slice loading + optional MSA/FW convergence
    - ``solve()``: OD-matrix assignment (thin wrapper around ``run()``)

    Parameters
    ----------
    base_path : str
        Path to the OSRM data files (must be partitioned for MLD).
    config : AssignmentConfig
        Assignment parameters.
    default_batch_size : int
        Default batch size for independent-batch mode.

    Example
    -------
    >>> solver = AssignmentSolver("network.osrm")
    >>> adapter = ODMatrixAdapter(origins, destinations, matrix)
    >>> result = solver.run(adapter.trips())
    >>> print(f"Converged: {result.converged}, Gap: {result.final_gap:.4f}")
    """

    def __init__(
        self,
        base_path: str,
        config: AssignmentConfig | None = None,
        *,
        default_batch_size: int = 1000,
    ) -> None:
        if default_batch_size <= 0:
            raise ValueError("default_batch_size must be positive")
        self.base_path = str(base_path)
        self.config = config or AssignmentConfig()
        self.default_batch_size = default_batch_size
        # Wire verbosity config to Python loggers
        level = _VERBOSITY_LEVELS.get(
            self.config.verbosity.upper(), logging.ERROR,
        )
        _assignment_loggers = [
            logger,
            logging.getLogger("osrm.assignment.plots"),
            logging.getLogger("osrm.assignment.solvers"),
        ]
        for lg in _assignment_loggers:
            lg.setLevel(level)
            lg.propagate = False
            if not lg.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S",
                ))
                lg.addHandler(handler)
        self.vdf = BiParabolicVDF(
            kc_ratio=self.config.vdf_kc_ratio,
            min_speed_kmh=self.config.vdf_min_speed_kmh,
        )
        self.smoother = DensitySmoothing(self.config.smoothing)
        self.loader = FractionalLoader(self.config.bin_width_s)
        self.writer = SegmentSpeedWriter(
            output_dir=self.config.speed_csv_dir,
            prefix="osrm_assignment",
        )
        self._engine: Optional[osrm_module.OSRM] = None

    def _create_engine(self) -> osrm_module.OSRM:
        """Create a fresh OSRM engine instance."""
        logger.info("Loading OSRM engine from %s", self.base_path)
        t0 = time.monotonic()
        eng = osrm_module.OSRM(
            storage_config=self.base_path,
            algorithm="MLD",
            use_shared_memory=False,
        )
        logger.info("Engine loaded in %.2fs", time.monotonic() - t0)
        return eng

    def _snap_trips(
        self,
        engine: osrm_module.OSRM,
        trips: List[DemandTrip],
    ) -> List[DemandTrip]:
        """Snap trip coordinates to OSRM waypoints.

        Calls Nearest once per unique coordinate, then replaces all
        trip origins/destinations with the snapped location.  This
        ensures routes start/end at actual network nodes, eliminating
        phantom partial-segment inconsistencies between link-level
        cost computation and OSRM route durations.
        """
        unique_coords: Dict[Tuple[float, float], list] = {}
        for trip in trips:
            for coord in (tuple(trip.origin), tuple(trip.destination)):
                if coord not in unique_coords:
                    unique_coords[coord] = list(coord)

        logger.info(
            "Snapping %d unique coordinates (%d trip-records)...",
            len(unique_coords), len(trips),
        )
        t0 = time.monotonic()

        # Snap each unique coordinate via Nearest
        snapped: Dict[Tuple[float, float], list] = {}
        for coord in unique_coords:
            try:
                result = engine.Nearest(coordinates=[list(coord)], number=1)
                if result.get("code") == "Ok" and result.get("waypoints"):
                    snapped[coord] = result["waypoints"][0]["location"]
                else:
                    snapped[coord] = list(coord)
            except Exception:
                snapped[coord] = list(coord)

        n_moved = sum(
            1 for c in unique_coords
            if abs(snapped[c][0] - c[0]) > 1e-8 or abs(snapped[c][1] - c[1]) > 1e-8
        )
        t_nearest = time.monotonic() - t0
        logger.info(
            "Nearest done in %.1fs (%d/%d adjusted)",
            t_nearest, n_moved, len(unique_coords),
        )

        # Rebuild trips with snapped coordinates
        logger.info("Rebuilding %d trip-records with snapped coords...", len(trips))
        t1 = time.monotonic()
        snapped_trips = []
        for trip in trips:
            snapped_trips.append(DemandTrip(
                origin=snapped[tuple(trip.origin)],
                destination=snapped[tuple(trip.destination)],
                volume=trip.volume,
                departure_time_s=trip.departure_time_s,
            ))
        logger.info(
            "Trip rebuild in %.1fs, snap total %.1fs",
            time.monotonic() - t1, time.monotonic() - t0,
        )
        return snapped_trips

    @staticmethod
    def _batch_route_raw(engine: osrm_module.OSRM, trips: List[DemandTrip]):
        """Route all trips via C++ BatchRoute, return raw C++ result objects.

        Returns a list parallel to *trips*; failed routes are None.
        """
        from osrm._params import RouteParameters as _RP, set_param as _sp

        params = []
        for t in trips:
            rp = _RP()
            rp.coordinates = [t.origin, t.destination]
            _sp(rp, "annotations", ["nodes", "distance", "duration", "speed"])
            params.append(rp)

        return engine._engine.BatchRoute(params)

    def _discover_network(
        self,
        engine: osrm_module.OSRM,
        trips: List[DemandTrip],
    ) -> NetworkState:
        """Route all trips via BatchRoute to discover network edges.

        IMPORTANT: ``engine`` must be a clean (uncustomized) OSRM instance
        so that annotation speed reflects the original profile speed from
        OSM ``maxspeed`` tags.  This speed is stored as ``freeflow_kmh``
        and is never updated — it is an immutable physical road attribute.
        """
        logger.debug("Discovering network edges from %d OD pairs...", len(trips))

        raw = self._batch_route_raw(engine, trips)
        route_results = [
            r.to_dict() for r in raw
            if r is not None
        ]
        route_results = [r for r in route_results if r.get("routes")]

        state = NetworkState.from_route_annotations(
            route_results,
            default_jam_density_per_lane=self.config.default_jam_density_per_lane,
            default_n_lanes=self.config.default_n_lanes,
        )
        logger.info("Discovered %d unique directed edges", state.n_edges)
        return state

    def _route_and_accumulate_with_paths(
        self,
        engine: osrm_module.OSRM,
        trips: List[DemandTrip],
        state: NetworkState,
    ) -> Tuple[np.ndarray, np.ndarray, float, List[RoutedTripPath]]:
        """Route all trips, accumulate link density and volume.

        Uses C++ accumulation when available — route results never cross
        the C++/Python boundary, eliminating the Python per-segment loop.

        Returns
        -------
        aon_density : np.ndarray
            All-or-nothing density (veh/km) from this iteration.
        aon_volume : np.ndarray
            All-or-nothing demand volume (vehicles) per link.
        aon_tstt : float
            AON total system travel time (vehicle-seconds).
        routed_paths : list[RoutedTripPath]
            Per-trip path records for refinement.
        """
        try:
            return self._route_and_accumulate_cpp(engine, trips, state)
        except Exception:
            logger.debug("C++ accumulation unavailable, using Python fallback")
            return self._route_and_accumulate_py(engine, trips, state)

    def _route_and_accumulate_cpp(
        self,
        engine: osrm_module.OSRM,
        trips: List[DemandTrip],
        state: NetworkState,
    ) -> Tuple[np.ndarray, np.ndarray, float, List[RoutedTripPath]]:
        """C++ fast path: route + accumulate in one native call."""
        from osrm.osrm_ext import batch_route_accumulate

        n = len(trips)
        coords = np.empty((n, 4), dtype=np.float64)
        volumes = np.empty(n, dtype=np.float64)
        for i, t in enumerate(trips):
            coords[i, 0] = t.origin[0]
            coords[i, 1] = t.origin[1]
            coords[i, 2] = t.destination[0]
            coords[i, 3] = t.destination[1]
            volumes[i] = t.volume

        bin_width_hr = self.config.bin_width_s / 3600.0
        default_jam = (self.config.default_jam_density_per_lane
                       * self.config.default_n_lanes)

        density, volume, tstt, raw_paths, new_edges = batch_route_accumulate(
            engine._engine,
            coords,
            volumes,
            state.edge_ids.astype(np.uint64),
            state.freeflow_kmh.astype(np.float64),
            bin_width_hr,
            self.config.min_speed_kmh,
            default_jam,
            self.config.default_n_lanes,
            self.config.n_threads,
        )

        # Register any newly discovered edges
        for from_id, to_id, length_m, speed_kmh in new_edges:
            state.register_edge(
                int(from_id), int(to_id), float(length_m),
                float(speed_kmh), default_jam, self.config.default_n_lanes,
            )

        # Convert C++ path tuples to RoutedTripPath objects
        routed_paths = [
            RoutedTripPath(
                trip_index=int(ti),
                edge_indices=ei.tolist(),
                density_contribution=dc.tolist(),
                duration_s=float(dur),
            )
            for ti, dur, ei, dc in raw_paths
        ]

        # Ensure arrays cover all edges (including newly registered ones)
        density = np.asarray(density, dtype=np.float64)
        volume = np.asarray(volume, dtype=np.float64)
        if len(density) < state.n_edges:
            density = np.append(density,
                                np.zeros(state.n_edges - len(density)))
            volume = np.append(volume,
                               np.zeros(state.n_edges - len(volume)))

        return density, volume, float(tstt), routed_paths

    def _route_and_accumulate_py(
        self,
        engine: osrm_module.OSRM,
        trips: List[DemandTrip],
        state: NetworkState,
    ) -> Tuple[np.ndarray, np.ndarray, float, List[RoutedTripPath]]:
        """Python fallback: original per-segment accumulation loop."""
        new_density = np.zeros(state.n_edges, dtype=np.float64)
        new_volume = np.zeros(state.n_edges, dtype=np.float64)
        tstt = 0.0
        bin_width_hr = self.config.bin_width_s / 3600.0
        routed_paths: List[RoutedTripPath] = []

        raw_results = self._batch_route_raw(engine, trips)

        for trip_idx, raw in enumerate(raw_results):
            if raw is None:
                continue
            trip = trips[trip_idx]
            routes = raw["routes"]
            if not routes:
                continue
            route = routes[0]
            route_duration_s = float(route["duration"])
            tstt += trip.volume * route_duration_s
            trip_edge_indices: List[int] = []
            trip_density: List[float] = []

            for leg in route["legs"]:
                ann = leg["annotation"]
                nodes = ann["nodes"]
                distances = ann["distance"]
                speeds = ann["speed"]

                for i in range(len(nodes) - 1):
                    from_id, to_id = int(nodes[i]), int(nodes[i + 1])
                    idx = state.edge_ordinal(from_id, to_id)
                    if idx is None:
                        dist = distances[i] if i < len(distances) else 0.0
                        spd = (speeds[i] * 3.6) if i < len(speeds) else 1.0
                        jam_d = (self.config.default_jam_density_per_lane
                                 * self.config.default_n_lanes)
                        idx = state.register_edge(
                            from_id, to_id, dist, spd, jam_d,
                            self.config.default_n_lanes,
                        )
                        if idx >= len(new_density):
                            pad = state.n_edges - len(new_density)
                            new_density = np.append(new_density, np.zeros(pad))
                            new_volume = np.append(new_volume, np.zeros(pad))

                    new_volume[idx] += trip.volume

                    seg_speed_kmh = (speeds[i] * 3.6) if i < len(speeds) else 0.0
                    if seg_speed_kmh < self.config.min_speed_kmh:
                        seg_speed_kmh = state.freeflow_kmh[idx]
                    density_delta = trip.volume / (seg_speed_kmh * bin_width_hr)
                    new_density[idx] += density_delta
                    trip_edge_indices.append(idx)
                    trip_density.append(density_delta)

            routed_paths.append(RoutedTripPath(
                trip_index=trip_idx,
                edge_indices=trip_edge_indices,
                density_contribution=trip_density,
                duration_s=route_duration_s,
            ))

        return new_density, new_volume, tstt, routed_paths

    def _route_and_accumulate(
        self,
        engine: osrm_module.OSRM,
        trips: List[DemandTrip],
        state: NetworkState,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Route all trips, accumulate link density and volume."""
        new_density, new_volume, tstt, _ = self._route_and_accumulate_with_paths(
            engine, trips, state,
        )
        return new_density, new_volume, tstt

    def _compute_relative_gap(
        self,
        state: NetworkState,
        blended_volume: np.ndarray,
        aon_volume: np.ndarray,
    ) -> float:
        """Compute Wardrop relative gap using VDF link costs.

        gap = Σ_a (V_blended · c_a) / Σ_a (V_aon · c_a) − 1

        Both numerator and denominator use the same VDF-derived link
        cost ``c_a = L_a / v_a``, eliminating systematic bias from OSRM
        speed quantization and turn-penalty overhead.

        At equilibrium, blended flows equal AON flows (everyone is
        already on shortest paths), so gap → 0.
        """
        link_time_s = state.length_m * 3.6 / np.maximum(state.speed_kmh, 1.0)

        numerator = float(np.sum(blended_volume * link_time_s))
        if numerator == 0:
            return 0.0

        denominator = float(np.sum(aon_volume * link_time_s))
        if denominator == 0:
            return 0.0

        return numerator / denominator - 1.0

    def _fw_line_search(
        self,
        k_current: np.ndarray,
        k_aon: np.ndarray,
        v_current: np.ndarray,
        v_aon: np.ndarray,
        state: NetworkState,
    ) -> float:
        """Find optimal step size via bisection on the Beckmann gradient.

        The Beckmann objective in volume space is:
            Z = sum_a V_a * c_a(k_a)

        The directional derivative uses the volume direction (dV) for
        the gradient (since AON minimizes volume-weighted cost), but
        density direction (dk) for evaluating costs:
            g(alpha) = sum_a c_a(k(alpha)) * dV_a
        """
        dk = k_aon - k_current
        dv = v_aon - v_current
        v_f = state.freeflow_kmh
        k_j = state.jam_density
        length_m = state.length_m

        def gradient(alpha: float) -> float:
            k_trial = np.maximum(k_current + alpha * dk, 0.0)
            speed = self.vdf.density_to_speed(k_trial, v_f, k_j)
            cost = length_m * 3.6 / np.maximum(speed, self.config.vdf_min_speed_kmh)
            return float(np.dot(cost, dv))

        g_lo = gradient(0.0)
        g_hi = gradient(1.0)

        # If direction doesn't reduce cost, don't step
        if g_lo >= 0.0:
            return 0.0

        # If full step still reduces cost, take it
        if g_hi <= 0.0:
            return 1.0

        lo, hi = 0.0, 1.0
        for _ in range(self.config.fw_bisections):
            if hi - lo < self.config.fw_line_search_tol:
                break
            mid = (lo + hi) / 2.0
            if gradient(mid) < 0:
                lo = mid
            else:
                hi = mid

        return (lo + hi) / 2.0

    def _update_state(self, state: NetworkState) -> None:
        """Update speed and flow from current density using VDF.

        Density is the primary state variable (set by MSA blending).
        Speed comes from the forward MFD: v(k), monotonically decreasing.
        Flow is derived: q = k × v.
        """
        # Smooth density
        smoothed = self.smoother.smooth(state.density_vpkm)

        # Forward MFD: density → speed (monotonic, always well-defined)
        state.speed_kmh = self.vdf.density_to_speed(
            smoothed, state.freeflow_kmh, state.jam_density
        )

        # Derive flow from fundamental identity: q = k × v
        state.flow_vph = state.density_vpkm * state.speed_kmh

    def run(
        self,
        trips: List[DemandTrip],
        progress_callback=None,
        state_patch=None,
        gap_every: int = 1,
        gap_sample_frac: float = 1.0,
    ) -> AssignmentResult:
        """Run the iterative assignment loop.

        Parameters
        ----------
        trips : list of DemandTrip
            Demand to assign.
        progress_callback : callable, optional
            Called with (iteration, gap, tstt) after each iteration.
        state_patch : callable, optional
            Called with (NetworkState,) immediately after discovery to
            patch lane counts or other attributes not available from
            OSRM annotations (e.g. lane count, jam density).
        gap_every : int
            Compute Wardrop relative gap every *N* iterations. The last
            iteration always computes gap regardless.  On skipped
            iterations the previous gap value is carried forward.
        gap_sample_frac : float
            Deprecated — kept for API compatibility.  Gap computation
            now uses VDF link costs (a single dot product) and no longer
            routes OD pairs, so sampling is unnecessary.

        Returns
        -------
        AssignmentResult

        Notes
        -----
        Network discovery must happen on a clean (uncustomized) OSRM
        instance so that annotation speed = freeflow.  Segment-speed
        customization permanently mutates OSRM edge weights (even
        re-partition does not undo it; only re-extract from OSM resets).
        Therefore freeflow_kmh is captured once at discovery and treated
        as immutable for the lifetime of the NetworkState.

        If ``run()`` is called multiple times on the same base path,
        the caller must re-extract/partition/customize beforehand, or
        use a separate base path per run.
        """
        t_start = time.monotonic()
        n_trips = len(trips)
        logger.info(
            "%s started: %d trips, max_iter=%d, n_threads=%d",
            self.config.method.upper(), n_trips, self.config.max_iterations,
            self.config.n_threads,
        )
        log: List[IterationResult] = []

        engine = self._create_engine()

        # Pre-snap trip coordinates to OSRM waypoints so routes
        # start/end at actual network nodes, not mid-segment phantom
        # points.  Eliminates numerator/denominator gap inconsistency.
        trips = self._snap_trips(engine, trips)

        # Start with an empty network — edges are discovered continuously
        # during routing via register_edge().  Annotation speed on a clean
        # engine (or on edges not in the segment-speed CSV) is always
        # freeflow, so first-seen speed is correct regardless of step.
        state = NetworkState.empty()

        if state_patch:
            state_patch(state)

        prev_density = np.zeros(state.n_edges, dtype=np.float64)
        prev_volume = np.zeros(state.n_edges, dtype=np.float64)

        # --- Incremental loading warm-up ---
        # Load demand in increasing fractions to avoid catastrophic
        # overshoot on bottleneck links in iteration 1.  Each step
        # routes at full demand but scales AON output, then replaces
        # (not blends) the state.  The main loop inherits a network
        # that has already seen partial congestion.
        inc_steps = self.config.incremental_steps
        n_inc = 0
        _last_log = time.monotonic()
        for step_frac in inc_steps:
            if step_frac >= 1.0:
                break  # 1.0 is handled by the main loop's first iteration
            n_inc += 1
            logger.debug(
                "=== Incremental step %d/%d (%.0f%% demand) ===",
                n_inc, len(inc_steps), step_frac * 100,
            )
            t_route = time.monotonic()
            aon_density, aon_volume, aon_tstt = self._route_and_accumulate(
                engine, trips, state,
            )
            route_time = time.monotonic() - t_route

            # Grow arrays if new edges discovered
            if len(prev_density) < state.n_edges:
                pad = state.n_edges - len(prev_density)
                prev_density = np.append(prev_density, np.zeros(pad))
                prev_volume = np.append(prev_volume, np.zeros(pad))
                if state_patch:
                    state_patch(state)
                self.smoother.build_adjacency(state.edge_ids, state.length_m)

            # Scale to fractional demand and replace state
            state.density_vpkm = np.clip(
                aon_density * step_frac, 0.0, state.jam_density,
            )
            blended_volume = np.maximum(aon_volume * step_frac, 0.0)
            prev_density = state.density_vpkm.copy()
            prev_volume = blended_volume.copy()

            self._update_state(state)

            # Customize OSRM with partial-demand speeds
            csv_path = self.writer.write_from_state(state, only_changed=True)
            logger.info("Customizing OSRM (incremental step %d)...", n_inc)
            osrm_module.customize(
                self.base_path,
                segment_speed_file=str(csv_path),
                verbosity="ERROR",
            )
            del engine
            engine = self._create_engine()

            logger.debug(
                "Incremental step %d: frac=%.2f, max_k/kj=%.2f, route=%.3fs",
                n_inc, step_frac,
                float(np.max(state.density_vpkm / state.jam_density)),
                route_time,
            )

        stop_reason: Optional[StopReason] = None
        stagnation_count = 0
        fw_zero_count = 0

        for n in range(1, self.config.max_iterations + 1):
            logger.debug("=== Iteration %d ===", n)

            # 2. Route all trips, get all-or-nothing density and volume
            t_route = time.monotonic()
            aon_density, aon_volume, aon_tstt = self._route_and_accumulate(
                engine, trips, state,
            )
            route_time = time.monotonic() - t_route

            # Grow prev arrays if new edges were discovered during routing
            if len(prev_density) < state.n_edges:
                pad = state.n_edges - len(prev_density)
                prev_density = np.append(prev_density, np.zeros(pad))
                prev_volume = np.append(prev_volume, np.zeros(pad))
                # Re-patch new edges (e.g. lane counts)
                if state_patch:
                    state_patch(state)
                self.smoother.build_adjacency(state.edge_ids, state.length_m)

            # 2b. Compute gap BEFORE blending — link costs match the
            #     network that OSRM routed on, so AON paths are truly
            #     shortest under these costs.
            is_last = n == self.config.max_iterations
            compute_gap = (n % gap_every == 0) or n == 1 or is_last
            t_gap = time.monotonic()
            if compute_gap:
                gap = self._compute_relative_gap(
                    state, prev_volume, aon_volume,
                )
            else:
                gap = log[-1].relative_gap if log else float("nan")
            gap_time = time.monotonic() - t_gap

            # 3. Blending: MSA (fixed 1/n) or FW (optimal step)
            if n == 1:
                alpha = 1.0
                state.density_vpkm = aon_density.copy()
                blended_volume = aon_volume.copy()
            elif self.config.method == "fw":
                alpha = self._fw_line_search(
                    prev_density, aon_density,
                    prev_volume, aon_volume,
                    state,
                )
                state.density_vpkm = prev_density + alpha * (aon_density - prev_density)
                blended_volume = prev_volume + alpha * (aon_volume - prev_volume)
            else:
                alpha = 1.0 / n
                state.density_vpkm = prev_density + alpha * (aon_density - prev_density)
                blended_volume = prev_volume + alpha * (aon_volume - prev_volume)

            # Cap density at jam density — k > k_j is unphysical (road is
            # full bumper-to-bumper).  Excess demand would queue upstream
            # (spillback) which we don't model.
            state.density_vpkm = np.clip(
                state.density_vpkm, 0.0, state.jam_density,
            )
            blended_volume = np.maximum(blended_volume, 0.0)

            max_delta = float(np.max(np.abs(state.density_vpkm - prev_density)))
            prev_density = state.density_vpkm.copy()
            prev_volume = blended_volume.copy()

            # 4. Update speed (from density) and flow (derived) via VDF
            self._update_state(state)

            # Compute blended TSTT from demand volume × link travel time
            link_time_s = state.length_m * 3.6 / np.maximum(state.speed_kmh, 1.0)
            tstt = float(np.sum(blended_volume * link_time_s))

            # 5. Count oversaturated links
            k_c = self.vdf.critical_density(state.jam_density)
            n_oversat = int(np.sum(state.density_vpkm > k_c))

            # 6. Write CSV and re-customize
            t_cust = time.monotonic()
            csv_path = self.writer.write_from_state(state, only_changed=True)
            logger.info("Customizing OSRM (iter %d)...", n)
            osrm_module.customize(
                self.base_path,
                segment_speed_file=str(csv_path),
                verbosity="ERROR",
            )
            customize_time = time.monotonic() - t_cust

            # 7. Reload engine
            t_engine = time.monotonic()
            del engine
            engine = self._create_engine()
            engine_time = time.monotonic() - t_engine

            # Compute speed and TT metrics for this iteration
            active = state.density_vpkm > 0
            active_speeds = state.speed_kmh[active] if np.any(active) else state.speed_kmh
            active_tt = link_time_s[active] if np.any(active) else link_time_s

            iter_result = IterationResult(
                iteration=n,
                phase="convergence",
                alpha=alpha,
                relative_gap=gap,
                tstt=tstt,
                state_change_norm=max_delta,
                n_oversaturated=n_oversat,
                max_k_over_kj=float(np.max(state.density_vpkm / np.maximum(state.jam_density, 1e-9))),
                median_k_over_kj=float(np.median((state.density_vpkm / np.maximum(state.jam_density, 1e-9))[state.density_vpkm > 0])) if np.any(state.density_vpkm > 0) else 0.0,
                mean_speed_kmh=float(np.mean(active_speeds)),
                median_speed_kmh=float(np.median(active_speeds)),
                max_speed_kmh=float(np.max(active_speeds)),
                min_speed_kmh=float(np.min(active_speeds)),
                median_tt_s=float(np.median(active_tt)),
                max_tt_s=float(np.max(active_tt)),
                n_routes=n_trips,
                route_time_s=route_time,
                customize_time_s=customize_time,
                engine_time_s=engine_time,
                iteration_time_s=time.monotonic() - t_route,
                link_tstt=tstt,
            )
            log.append(iter_result)

            now = time.monotonic()
            if now - _last_log >= 2.0:
                logger.info(
                    "Iter %d: gap=%.4f, TSTT=%.0f, alpha=%.4f",
                    n, gap, tstt, alpha,
                )
                _last_log = now
            else:
                logger.debug(
                    "Iter %d: gap=%.4f, TSTT=%.0f, alpha=%.4f, max_dk=%.1f, "
                    "oversat=%d, route=%.3fs, gap=%.3fs, customize=%.3fs, engine=%.3fs",
                    n, gap, tstt, alpha, max_delta, n_oversat,
                    route_time, gap_time, customize_time, engine_time,
                )

            if progress_callback:
                progress_callback(n, gap, tstt)

            # 9. Convergence / stagnation checks
            stop_reason: Optional[StopReason] = None

            if (compute_gap and self.config.convergence_gap > 0
                    and 0 <= gap < self.config.convergence_gap):
                stop_reason = StopReason.CONVERGED

            # Stagnation: gap delta below tolerance for N consecutive iters
            if (stop_reason is None and compute_gap
                    and self.config.stagnation_tol > 0 and len(log) >= 2):
                prev_gap = log[-2].relative_gap
                if abs(gap - prev_gap) < self.config.stagnation_tol:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                if stagnation_count >= self.config.stagnation_window:
                    stop_reason = StopReason.STAGNATED

            # FW-specific: alpha=0 means line search found no improvement
            if (stop_reason is None
                    and self.config.method == "fw" and alpha == 0.0):
                fw_zero_count += 1
                if fw_zero_count >= 2:
                    stop_reason = StopReason.STAGNATED
            else:
                fw_zero_count = 0

            if stop_reason is not None:
                break

        total_time = time.monotonic() - t_start
        del engine

        # Cleanup CSV
        self.writer.cleanup()

        if stop_reason is None:
            stop_reason = StopReason.MAX_ITERATIONS

        final_gap = log[-1].relative_gap if log else float("inf")
        logger.info(
            "%s complete: %s at iter %d, gap=%.6f, %.1fs",
            self.config.method.upper(), stop_reason.value,
            len(log), final_gap, total_time,
        )

        return AssignmentResult(
            converged=stop_reason == StopReason.CONVERGED,
            iterations=len(log),
            final_gap=log[-1].relative_gap if log else float("inf"),
            network_state=state,
            iteration_log=log,
            total_time_s=total_time,
            stop_reason=stop_reason,
        )

    # ------------------------------------------------------------------
    # Methods from the former TrafficAssignmentSolver
    # ------------------------------------------------------------------

    def _make_loop(self):
        """Backward-compatible factory — returns self."""
        return self

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
        """Solve an OD-matrix assignment problem.

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
        loop,
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
        loop,
        state: NetworkState,
        trips: list,
        initial_volume: np.ndarray,
        state_patch,
        progress_callback,
        max_od: int,
        max_rounds: int,
        gap_threshold: float,
        method: str = "msa",
    ) -> Tuple[object, List[IterationResult]]:
        """Run convergence iterations on link-level density state.

        After greedy warm start, iteratively:
          1. Freeze current density
          2. Route demand on frozen network (all-or-nothing)
          3. Blend: k = (1-α)k + αk̂
             - MSA: α = 1/(m+1)
             - FW:  α from Beckmann line search
          4. Clip, update VDF, customize, reload
        """
        msa_results: List[IterationResult] = []
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

            iter_result = IterationResult(
                iteration=m,
                phase="convergence",
                alpha=alpha,
                aon_tstt=float(aon_tstt),
                link_tstt=link_tstt,
                tstt=link_tstt,
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
        """Execute a trip stream as isolated frozen-cost assignment runs."""
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
    ) -> AssignmentResult:
        """Run assignment over ordered trip batches with convergence.

        Trips are grouped by departure-time bin using ``config.bin_width_s``.
        After the greedy load, convergence iterations run when
        ``max_rounds > 0``.
        """
        started = time.monotonic()
        adapter = stream if isinstance(stream, TripStreamAdapter) else TripStreamAdapter(stream)
        all_trips = adapter.trips()
        if not all_trips:
            return AssignmentResult(
                converged=False,
                iterations=0,
                final_gap=0.0,
                network_state=NetworkState.empty(),
                iteration_log=[],
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

        greedy_results: List[IterationResult] = []
        od_ledger = ODLedger()
        running_network_tstt = 0.0
        accumulated_volume = np.zeros(0, dtype=np.float64)
        for batch in snapped_stream.iter_time_slices(
            bin_width_s=loop.config.bin_width_s,
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

            batch_result = IterationResult(
                iteration=batch.batch_index,
                phase="greedy",
                alpha=1.0,
                departure_bin=batch.departure_bin,
                n_trips=len(batch.trips),
                batch_tstt=batch_tstt,
                network_tstt=running_network_tstt,
                tstt=running_network_tstt,
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
            greedy_results.append(batch_result)

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

        msa_results: List[IterationResult] = []
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
        iteration_log = greedy_results + msa_results
        if msa_results:
            logger.info(
                "Complete: %d load steps, %d %s iterations, %.1fs",
                len(greedy_results), len(msa_results), method_label, total_time,
            )
        else:
            logger.info(
                "Complete: %d load steps, greedy only, %.1fs",
                len(greedy_results), total_time,
            )

        _converged = bool(
            msa_results
            and msa_results[-1].relative_gap is not None
            and 0 <= msa_results[-1].relative_gap < gap_threshold
        )
        _final_gap = msa_results[-1].relative_gap if msa_results else 0.0

        return AssignmentResult(
            converged=_converged,
            iterations=len(iteration_log),
            final_gap=_final_gap,
            network_state=state,
            iteration_log=iteration_log,
            total_time_s=total_time,
            stop_reason=StopReason.CONVERGED if _converged else StopReason.MAX_ITERATIONS,
            od_ledger=od_ledger,
            n_trips=len(snapped_trips),
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
AssignmentLoop = AssignmentSolver
TrafficAssignmentSolver = AssignmentSolver
MatrixAssignmentSolver = AssignmentSolver
MatrixFreeHillClimber = AssignmentSolver

# Legacy result type aliases
MSAIterationResult = IterationResult
HillClimberBatchResult = IterationResult


@dataclass
class HillClimberResult:
    """Legacy result type — wraps AssignmentResult for backward compat."""

    network_state: Optional[NetworkState]
    batch_results: List[IterationResult]
    total_time_s: float
    n_trips: int
    od_ledger: Optional[ODLedger] = None
    msa_results: List[IterationResult] = field(default_factory=list)

    @property
    def n_batches(self) -> int:
        return len(self.batch_results)

    def log_as_dict(self) -> dict:
        return {
            "batch_index": [r.iteration for r in self.batch_results],
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
