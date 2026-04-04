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
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import osrm as osrm_module
from osrm.assignment.density_smoothing import DensitySmoothing, DensitySmoothingConfig
from osrm.assignment.fractional_loading import FractionalLoader
from osrm.assignment.network_state import NetworkState
from osrm.assignment.od_matrix import DemandTrip
from osrm.assignment.segment_speed_writer import SegmentSpeedWriter
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
    # VDF speed floor — kept very low so the cost function captures
    # the full gradient of the MFD congested branch.  The CSV writer
    # applies a separate (slightly higher) floor for OSRM ingestion.
    vdf_min_speed_kmh: float = 0.01
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
    """Metrics for one iteration."""

    iteration: int
    phase: str = "convergence"
    alpha: float = 0.0
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
    link_tstt: float = 0.0


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
    n_trips: int = 0

    def log_as_dict(self) -> Dict:
        """Convert iteration log to dict for plotting."""
        return {
            "iteration": [r.iteration for r in self.iteration_log],
            "phase": [r.phase for r in self.iteration_log],
            "relative_gap": [r.relative_gap for r in self.iteration_log],
            "tstt": [r.tstt for r in self.iteration_log],
            "max_flow_delta": [r.state_change_norm for r in self.iteration_log],
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
class StreamBatchResult:
    """Metrics for one batch in stream assignment."""

    batch_index: int
    n_trips: int
    route_time_s: float
    customize_time_s: float
    engine_time_s: float
    batch_time_s: float
    tstt: float
    queue_vehicles: float  # mean unserved demand per link per lane (veh/hr/lane)
    mean_speed_kmh: float
    min_speed_kmh: float
    n_oversaturated: int


@dataclass
class StreamResult:
    """Final result of stream assignment."""

    n_trips: int
    n_batches: int
    network_state: NetworkState
    batch_log: List[StreamBatchResult]
    total_time_s: float

    def log_as_dict(self) -> Dict:
        """Convert batch log to dict for plotting."""
        return {
            "batch": [r.batch_index for r in self.batch_log],
            "n_trips": [r.n_trips for r in self.batch_log],
            "tstt": [r.tstt for r in self.batch_log],
            "queue_vehicles": [r.queue_vehicles for r in self.batch_log],
            "mean_speed_kmh": [r.mean_speed_kmh for r in self.batch_log],
            "min_speed_kmh": [r.min_speed_kmh for r in self.batch_log],
            "n_oversaturated": [r.n_oversaturated for r in self.batch_log],
            "route_time_s": [r.route_time_s for r in self.batch_log],
            "customize_time_s": [r.customize_time_s for r in self.batch_log],
        }




class AssignmentSolver:
    """Unified traffic assignment solver.

    Uses incremental warm-up followed by MSA or Frank-Wolfe convergence
    to find user-equilibrium link flows.

    Parameters
    ----------
    base_path : str
        Path to the OSRM data files (must be partitioned for MLD).
    config : AssignmentConfig
        Assignment parameters.

    Example
    -------
    >>> solver = AssignmentSolver("network.osrm")
    >>> result = solver.assign_matrix(trips)
    >>> print(f"Converged: {result.converged}, Gap: {result.final_gap:.4f}")
    """

    def __init__(
        self,
        base_path: str,
        config: AssignmentConfig | None = None,
    ) -> None:
        self.base_path = str(base_path)
        self.config = config or AssignmentConfig()
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

    def _create_engine(self, *, quiet: bool = False) -> osrm_module.OSRM:
        """Create a fresh OSRM engine instance."""
        _log = logger.debug if quiet else logger.info
        _log("Loading OSRM engine from %s", self.base_path)
        t0 = time.monotonic()
        eng = osrm_module.OSRM(
            storage_config=self.base_path,
            algorithm="MLD",
            use_shared_memory=False,
        )
        _log("Engine loaded in %.2fs", time.monotonic() - t0)
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

    def _route_and_accumulate(
        self,
        engine: osrm_module.OSRM,
        trips: List[DemandTrip],
        state: NetworkState,
    ) -> Tuple[np.ndarray, float]:
        """Route all trips, accumulate link volume.

        Uses C++ accumulation when available — route results never cross
        the C++/Python boundary, eliminating the Python per-segment loop.

        Returns
        -------
        aon_volume : np.ndarray
            All-or-nothing demand volume (vehicles) per link.
        aon_tstt : float
            AON total system travel time (vehicle-seconds).
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
    ) -> Tuple[np.ndarray, float]:
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

        volume, tstt, new_edges = batch_route_accumulate(
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

        volume = np.asarray(volume, dtype=np.float64)
        if len(volume) < state.n_edges:
            volume = np.append(volume,
                               np.zeros(state.n_edges - len(volume)))

        return volume, float(tstt)

    def _route_and_accumulate_py(
        self,
        engine: osrm_module.OSRM,
        trips: List[DemandTrip],
        state: NetworkState,
    ) -> Tuple[np.ndarray, float]:
        """Python fallback: original per-segment accumulation loop."""
        new_volume = np.zeros(state.n_edges, dtype=np.float64)
        tstt = 0.0

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
                        if idx >= len(new_volume):
                            pad = state.n_edges - len(new_volume)
                            new_volume = np.append(new_volume, np.zeros(pad))

                    new_volume[idx] += trip.volume

        return new_volume, tstt

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
        link_time_s = state.length_m * 3.6 / np.maximum(
            state.speed_kmh, self.config.vdf_min_speed_kmh,
        )

        numerator = float(np.sum(blended_volume * link_time_s))
        if numerator == 0:
            return 0.0

        denominator = float(np.sum(aon_volume * link_time_s))
        if denominator == 0:
            return 0.0

        return numerator / denominator - 1.0

    def _fw_line_search(
        self,
        q_current: np.ndarray,
        q_aon: np.ndarray,
        state: NetworkState,
    ) -> float:
        """Find optimal step size via bisection on the Beckmann gradient.

        Everything operates in flow space.  The Beckmann objective is:
            Z = sum_a integral_0^{q_a} c_a(x) dx

        The directional derivative along dq = q_aon − q_current is:
            g(alpha) = sum_a c_a(q(alpha)) * dq_a
        """
        dq = q_aon - q_current
        v_f = state.freeflow_kmh
        k_j = state.jam_density
        length_m = state.length_m

        def gradient(alpha: float) -> float:
            q_trial = np.maximum(q_current + alpha * dq, 0.0)
            speed = self.vdf.demand_to_speed(q_trial, v_f, k_j)
            cost = length_m * 3.6 / np.maximum(speed, self.config.vdf_min_speed_kmh)
            return float(np.dot(cost, dq))

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
        """Derive density and speed from the current link flow.

        Flow (volume) is the primary state variable set by MSA/FW
        blending.  The extended monotone mapping ``demand_to_density``
        converts any non-negative flow to a density in [0, k_j), and
        ``density_to_speed`` gives the corresponding MFD speed.
        """
        # Extended mapping: flow → density (monotone for all q ≥ 0)
        state.density_vpkm = self.vdf.demand_to_density(
            state.flow_vph, state.freeflow_kmh, state.jam_density,
            kc_ratio=state.kc_ratio,
        )

        # Optional spatial smoothing (operates on density)
        smoothed = self.smoother.smooth(state.density_vpkm)

        # Forward MFD: density → speed
        state.speed_kmh = self.vdf.density_to_speed(
            smoothed, state.freeflow_kmh, state.jam_density,
            kc_ratio=state.kc_ratio,
        )

    def assign_matrix(
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

        prev_volume = np.zeros(state.n_edges, dtype=np.float64)

        # --- Incremental loading warm-up ---
        # Load demand in increasing fractions to avoid catastrophic
        # overshoot on bottleneck links in iteration 1.  Each step
        # routes at full demand but scales AON flow, then replaces
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
            aon_volume, aon_tstt = self._route_and_accumulate(
                engine, trips, state,
            )
            route_time = time.monotonic() - t_route

            # Grow arrays if new edges discovered
            if len(prev_volume) < state.n_edges:
                pad = state.n_edges - len(prev_volume)
                prev_volume = np.append(prev_volume, np.zeros(pad))
                if state_patch:
                    state_patch(state)
                self.smoother.build_adjacency(state.edge_ids, state.length_m)

            # Scale to fractional demand and replace state (flow-primary)
            state.flow_vph = np.maximum(aon_volume * step_frac, 0.0)
            prev_volume = state.flow_vph.copy()

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
                float(np.max(state.density_vpkm / np.maximum(state.jam_density, 1e-9))),
                route_time,
            )

        stop_reason: Optional[StopReason] = None
        stagnation_count = 0
        fw_zero_count = 0

        for n in range(1, self.config.max_iterations + 1):
            logger.debug("=== Iteration %d ===", n)

            # 2. Route all trips, get all-or-nothing volume
            t_route = time.monotonic()
            aon_volume, aon_tstt = self._route_and_accumulate(
                engine, trips, state,
            )
            route_time = time.monotonic() - t_route

            # Grow prev arrays if new edges were discovered during routing
            if len(prev_volume) < state.n_edges:
                pad = state.n_edges - len(prev_volume)
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

            # 3. Blending in flow space: MSA (fixed 1/n) or FW (optimal step)
            if n == 1:
                alpha = 1.0
                state.flow_vph = aon_volume.copy()
            elif self.config.method == "fw":
                alpha = self._fw_line_search(
                    prev_volume, aon_volume, state,
                )
                state.flow_vph = prev_volume + alpha * (aon_volume - prev_volume)
            else:
                alpha = 1.0 / n
                state.flow_vph = prev_volume + alpha * (aon_volume - prev_volume)

            state.flow_vph = np.maximum(state.flow_vph, 0.0)

            max_delta = float(np.max(np.abs(state.flow_vph - prev_volume)))
            prev_volume = state.flow_vph.copy()

            # 4. Derive density and speed from blended flow via VDF
            self._update_state(state)

            # Compute blended TSTT from demand volume × link travel time
            link_time_s = state.length_m * 3.6 / np.maximum(
                state.speed_kmh, self.config.vdf_min_speed_kmh,
            )
            tstt = float(np.sum(state.flow_vph * link_time_s))

            # 5. Count oversaturated links
            k_c = self.vdf.critical_density(state.jam_density, kc_ratio=state.kc_ratio)
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
            active = state.flow_vph > 0
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

    run = assign_matrix

    def assign_stream(
        self,
        trips: List[DemandTrip],
        *,
        batch_size: Optional[int] = None,
        state_patch=None,
        progress_callback=None,
    ) -> "StreamResult":
        """Forward-simulation assignment with incremental loading.

        Routes trips in batches, updating network state (VDF → customize)
        between batches.  Unserved demand (vehicles on links that exceed
        capacity) carries forward as a queue, adding to congestion in
        subsequent batches.

        Unlike ``assign_matrix()`` which iterates to Wardrop equilibrium,
        this performs a single forward pass — quality depends on batch
        granularity and the resulting customize frequency.

        Parameters
        ----------
        trips : list of DemandTrip
            All trips to assign.  If trips have varying
            ``departure_time_s``, they are sorted and routed in
            chronological order.  For single-period demand, all
            departure times can be 0.
        batch_size : int or None
            Number of trips per loading step.  After each batch, the
            network state is updated and OSRM is re-customized.
            Smaller batches = more accurate but more customize overhead.
            If ``None`` (default), auto-tuned from the first batch's
            network size: ``batch_size = max(100, n_edges // 4)``.
            On large networks trips are spatially distributed so coarse
            batches suffice; on small networks finer batches avoid
            route-dumping.
        state_patch : callable, optional
            Called with ``(NetworkState,)`` after discovery to patch
            lane counts or jam density.
        progress_callback : callable, optional
            Called with ``(batch_index, n_batches, queue_vehicles)``
            after each batch.

        Returns
        -------
        StreamResult
        """
        from osrm.assignment.trip_stream import TripStreamAdapter

        t_start = time.monotonic()
        n_trips = len(trips)

        engine = self._create_engine()

        # Sort trips by departure time
        adapter = TripStreamAdapter(trips, sort_by_departure=True)

        # Snap coordinates (done once for all trips)
        snapped_trips = self._snap_trips(engine, adapter.trips())

        # Re-wrap after snapping
        adapter = TripStreamAdapter(snapped_trips, sort_by_departure=False)

        # Auto-tune batch_size so each batch adds ~10 vehicles per link.
        # Route a probe sample to measure avg edges per route, then:
        #   trips_per_veh_per_link = n_edges / avg_edges_per_route
        #   batch_size = 10 × trips_per_veh_per_link
        if batch_size is None:
            sample_n = min(5000, n_trips)
            sample = snapped_trips[:sample_n]

            # Discover network topology from probe
            probe_state = self._discover_network(engine, sample)

            # Route probe to measure edge traversals per trip.
            # aon_volume is in trip.volume units; normalize by total
            # volume to get avg edges per unit of demand.
            probe_vol, _ = self._route_and_accumulate(
                engine, sample, probe_state,
            )
            total_volume = sum(t.volume for t in sample)
            total_traversals = float(np.sum(probe_vol))
            avg_edges_per_route = total_traversals / max(total_volume, 1.0)

            # Extrapolate edge count: probe discovers edges sub-linearly.
            # If S trips found E edges, full N trips find ~E*(N/S)^0.3.
            n_edges_probe = probe_state.n_edges
            if sample_n < n_trips:
                n_edges_est = int(
                    n_edges_probe * (n_trips / sample_n) ** 0.3
                )
            else:
                n_edges_est = n_edges_probe

            trips_per_1vpl = n_edges_est / max(avg_edges_per_route, 1.0)
            batch_size = max(100, int(10 * trips_per_1vpl))

            logger.info(
                "Auto-tuned batch_size=%d (~%d batches): "
                "%d probe edges → %d est edges, "
                "%.1f avg edges/route",
                batch_size, max(1, n_trips // batch_size),
                n_edges_probe, n_edges_est, avg_edges_per_route,
            )

        logger.info(
            "Stream assignment: %d trips, batch_size=%d, n_threads=%d",
            n_trips, batch_size, self.config.n_threads,
        )

        # Network starts empty — edges discovered during routing
        state = NetworkState.empty()
        if state_patch:
            state_patch(state)

        # Cumulative flow: accumulates across loading steps within a
        # period.  Each batch ADDS its demand to the running total.
        cumulative_volume = np.zeros(state.n_edges, dtype=np.float64)

        # Queue: vehicles on over-capacity links that spill into the
        # next time period.  Only meaningful for multi-period — within
        # a single period it's diagnostic (how much demand exceeds
        # capacity after all batches are loaded).
        queue_veh = np.zeros(state.n_edges, dtype=np.float64)

        batch_log: List[StreamBatchResult] = []

        batches = list(adapter.iter_batches(batch_size))
        n_batches = len(batches)
        logger.info("Split into %d loading batches", n_batches)

        for batch in batches:
            t_batch = time.monotonic()
            bi = batch.batch_index

            # 1. Route this batch against current (congested) weights
            t_route = time.monotonic()
            aon_volume, aon_tstt = self._route_and_accumulate(
                engine, batch.trips, state,
            )
            route_time = time.monotonic() - t_route

            # Grow arrays if new edges were discovered
            if len(cumulative_volume) < state.n_edges:
                pad = state.n_edges - len(cumulative_volume)
                cumulative_volume = np.append(cumulative_volume, np.zeros(pad))
                queue_veh = np.append(queue_veh, np.zeros(
                    state.n_edges - len(queue_veh)))
                if state_patch:
                    state_patch(state)
                self.smoother.build_adjacency(state.edge_ids, state.length_m)

            # 2. Accumulate: add this batch's demand to cumulative flow.
            #    aon_volume is in the same units as DemandTrip.volume
            #    (veh/hr for TNTP, vehicles for ABM with volume=1).
            cumulative_volume += aon_volume
            state.flow_vph = np.maximum(cumulative_volume, 0.0)

            # 3. VDF: flow → density → speed
            self._update_state(state)

            # 4. Unserved demand diagnostic: flow exceeding physical
            #    throughput.  For multi-period, this would carry forward.
            unserved_vph = state.unserved_demand
            unserved_per_lane = unserved_vph / np.maximum(state.n_lanes, 1)
            oversat_mask = unserved_vph > 0
            mean_queue_per_lane = (
                float(np.mean(unserved_per_lane[oversat_mask]))
                if np.any(oversat_mask) else 0.0
            )
            queue_veh = unserved_vph.copy()

            # 5. Write CSV and re-customize OSRM
            t_cust = time.monotonic()
            csv_path = self.writer.write_from_state(state, only_changed=True)
            osrm_module.customize(
                self.base_path,
                segment_speed_file=str(csv_path),
                verbosity="ERROR",
            )
            customize_time = time.monotonic() - t_cust

            # 6. Reload engine with updated weights
            t_engine = time.monotonic()
            del engine
            engine = self._create_engine(quiet=True)
            engine_time = time.monotonic() - t_engine

            # Metrics
            active = state.flow_vph > 0
            active_speeds = state.speed_kmh[active] if np.any(active) else state.speed_kmh
            link_time_s = state.length_m * 3.6 / np.maximum(
                state.speed_kmh, self.config.vdf_min_speed_kmh,
            )
            tstt = float(np.sum(state.flow_vph * link_time_s))

            batch_result = StreamBatchResult(
                batch_index=bi,
                n_trips=len(batch.trips),
                route_time_s=route_time,
                customize_time_s=customize_time,
                engine_time_s=engine_time,
                batch_time_s=time.monotonic() - t_batch,
                tstt=tstt,
                queue_vehicles=mean_queue_per_lane,
                mean_speed_kmh=float(np.mean(active_speeds)),
                min_speed_kmh=float(np.min(active_speeds)),
                n_oversaturated=int(np.sum(
                    state.density_vpkm > self.vdf.critical_density(state.jam_density, kc_ratio=state.kc_ratio)
                )),
            )
            batch_log.append(batch_result)

            if bi % max(1, n_batches // 10) == 0 or bi == n_batches - 1:
                logger.info(
                    "Batch %d/%d: %d trips, queue=%.0f veh/hr/lane, "
                    "mean_speed=%.1f km/h, oversat=%d, "
                    "route=%.1fs, cust=%.1fs",
                    bi + 1, n_batches, len(batch.trips),
                    mean_queue_per_lane, batch_result.mean_speed_kmh,
                    batch_result.n_oversaturated,
                    route_time, customize_time,
                )

            if progress_callback:
                progress_callback(bi, n_batches, mean_queue_per_lane)

        total_time = time.monotonic() - t_start
        del engine
        self.writer.cleanup()

        logger.info(
            "Stream complete: %d trips in %d batches, %.1fs, "
            "final queue=%.0f veh/hr/lane",
            n_trips, n_batches, total_time,
            batch_log[-1].queue_vehicles if batch_log else 0,
        )

        return StreamResult(
            n_trips=n_trips,
            n_batches=n_batches,
            network_state=state,
            batch_log=batch_log,
            total_time_s=total_time,
        )
