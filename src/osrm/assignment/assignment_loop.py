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


def _fmt_num(n: float) -> str:
    """Format a number with k/M suffix for compact display."""
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}k"
    return f"{n:.0f}"


def _batch_header(timing: bool = False) -> str:
    base = (
        " {:>7s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s}"
    ).format("Bat", "Per", "Routes", "Vol", "Left",
             "VCavg", "VCmax", "Speed", "Osat")
    if timing:
        base += "  {:>4s} {:>4s} {:>4s}".format("Rte", "Acc", "Cst")
    return base


def _batch_sep(timing: bool = False) -> str:
    return " " + "─" * (len(_batch_header(timing)) - 1)


def _fmt_eta(seconds: float) -> str:
    """Format ETA as compact string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _period_to_clock(period_bin: object, period_duration_s: float) -> str:
    """Convert period bin number to clock time string like '4:00'."""
    try:
        secs = int(period_bin) * period_duration_s
    except (TypeError, ValueError):
        return str(period_bin)
    h = int(secs // 3600) % 24
    m = int((secs % 3600) // 60)
    return f"{h}:{m:02d}"


def _fmt_batch_row(
    bi: int, total: int, period: object, n_routes: int, vol: float,
    vol_left: float, vc_avg: float, vc_max: float, speed: float,
    oversat: int, links: int, route_s: float, accum_s: float, cust_s: float,
    eta_s: float = 0.0, timing: bool = False,
    period_duration_s: float = 900.0,
) -> str:
    clock = _period_to_clock(period, period_duration_s)
    row = (
        " {:>7s} {:>5s} {:>5d} {:>5s} {:>5s} {:>5.3f} {:>5.3f} {:>5.1f} {:>5s}"
    ).format(
        f"{bi}/{total}", clock, n_routes, _fmt_num(vol), _fmt_num(vol_left),
        vc_avg, vc_max, speed, _fmt_num(oversat),
    )
    if timing:
        row += "  {:>4.1f} {:>4.1f} {:>4.1f}".format(route_s, accum_s, cust_s)
    return row


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
    # Trip explosion: split high-volume trips into multiple low-volume routes
    # with spatial jitter for path diversity.  0 disables explosion.
    max_vol_per_route: float = 5.0
    explosion_jitter_m: float = 150.0  # Gaussian sigma in meters
    # OD-cluster round-robin: max trips drawn per OD-bin per batch.
    max_trips_per_bin: int = 5
    # Sampled gap: fraction of period trips re-routed at period-final to
    # estimate Wardrop gap.  0 disables (no extra routing cost).
    gap_sample_frac: float = 0.0
    # Show per-batch timing breakdown (Route/Accum/Cust columns)
    log_timing: bool = False

    def __post_init__(self):
        if self.n_threads == -1:
            self.n_threads = max(1, (os.cpu_count() or 4) - 2)


def _explode_trips(
    trips: List[DemandTrip],
    max_vol: float,
    jitter_m: float,
    rng: np.random.Generator,
) -> List[DemandTrip]:
    """Split high-volume trips into multiple low-volume trips with spatial jitter.

    Each trip with volume > max_vol is split into ceil(volume/max_vol) copies,
    each with equal share of the volume and independent Gaussian jitter on
    origin and destination coordinates.  This produces diverse OSRM snap
    points → diverse routes → less artificial path concentration.
    """
    if max_vol <= 0:
        return trips

    M_PER_DEG_LAT = 111_320.0
    cos_lat = np.cos(np.radians(37.0))  # approximate for continental US
    M_PER_DEG_LON = 111_320.0 * cos_lat
    sigma_lat = jitter_m / M_PER_DEG_LAT
    sigma_lon = jitter_m / M_PER_DEG_LON

    result: List[DemandTrip] = []
    n_exploded = 0
    for trip in trips:
        if trip.volume <= max_vol:
            result.append(trip)
            continue
        n_copies = int(np.ceil(trip.volume / max_vol))
        copy_vol = trip.volume / n_copies
        n_exploded += 1
        for _ in range(n_copies):
            if copy_vol > 1.0:
                o_lon = trip.origin[0] + rng.normal(0, sigma_lon)
                o_lat = trip.origin[1] + rng.normal(0, sigma_lat)
                d_lon = trip.destination[0] + rng.normal(0, sigma_lon)
                d_lat = trip.destination[1] + rng.normal(0, sigma_lat)
            else:
                o_lon, o_lat = trip.origin
                d_lon, d_lat = trip.destination
            result.append(DemandTrip(
                origin=(o_lon, o_lat),
                destination=(d_lon, d_lat),
                volume=copy_vol,
                departure_time_s=trip.departure_time_s,
                trip_id=trip.trip_id,
            ))

    if n_exploded > 0:
        logger.info(
            "Trip explosion: %d trips → %d (max_vol=%.0f, %d exploded)",
            len(trips), len(result), max_vol, n_exploded,
        )
    return result


def _compute_cell_size_km(trips: List[DemandTrip]) -> float:
    """Derive OD grid cell size from median trip length."""
    if len(trips) < 10:
        return 5.0  # fallback
    # Sample up to 50k trips for speed
    sample = trips if len(trips) <= 50_000 else [
        trips[i] for i in np.random.default_rng(0).choice(len(trips), 50_000, replace=False)
    ]
    o = np.array([(t.origin[0], t.origin[1]) for t in sample])
    d = np.array([(t.destination[0], t.destination[1]) for t in sample])
    # Haversine approximation (flat-earth in km)
    lat_mid = np.radians((o[:, 1] + d[:, 1]) / 2)
    dx = (d[:, 0] - o[:, 0]) * np.cos(lat_mid) * 111.32
    dy = (d[:, 1] - o[:, 1]) * 111.32
    lengths_km = np.sqrt(dx**2 + dy**2)
    cell = float(np.median(lengths_km)) / 3.0
    return max(1.0, cell)  # floor at 1 km


def _od_bin_key(
    trip: DemandTrip, inv_cell_lon: float, inv_cell_lat: float,
) -> Tuple[int, int, int, int]:
    """Hash a trip into a 4D OD grid bin."""
    return (
        int(np.floor(trip.origin[0] * inv_cell_lon)),
        int(np.floor(trip.origin[1] * inv_cell_lat)),
        int(np.floor(trip.destination[0] * inv_cell_lon)),
        int(np.floor(trip.destination[1] * inv_cell_lat)),
    )


def _build_od_batches(
    trips: List[DemandTrip],
    period_duration_s: Optional[float],
    cell_size_km: float,
    max_per_bin: int,
) -> Tuple[List["TripBatch"], int]:
    """Build spatially-diverse batches via OD-grid round-robin.

    Returns (list_of_TripBatch, n_occupied_bins_max).
    """
    from osrm.assignment.trip_stream import TripBatch

    # Convert cell_size from km to degree increments
    cos_lat = np.cos(np.radians(37.0))  # approximate; good enough for binning
    cell_lon = cell_size_km / (111.32 * cos_lat)
    cell_lat = cell_size_km / 111.32
    inv_cell_lon = 1.0 / cell_lon
    inv_cell_lat = 1.0 / cell_lat

    # Group trips by departure period first
    if period_duration_s is not None:
        from itertools import groupby
        sorted_trips = sorted(trips, key=lambda t: t.departure_time_s)
        period_groups = []
        for dep_bin, grp in groupby(
            sorted_trips,
            key=lambda t: int((t.departure_time_s % (24 * 3600.0)) // period_duration_s),
        ):
            period_groups.append((dep_bin, list(grp)))
    else:
        period_groups = [(None, list(trips))]

    batches: List[TripBatch] = []
    batch_idx = 0
    max_bins = 0

    for dep_bin, period_trips in period_groups:
        # Bin trips by 4D OD grid
        bins: Dict[Tuple[int, int, int, int], List[DemandTrip]] = {}
        for t in period_trips:
            key = _od_bin_key(t, inv_cell_lon, inv_cell_lat)
            if key not in bins:
                bins[key] = []
            bins[key].append(t)

        n_bins = len(bins)
        if n_bins > max_bins:
            max_bins = n_bins

        # Round-robin: cycle through bins, take max_per_bin from each
        bin_iters = {k: iter(v) for k, v in bins.items()}
        bin_keys = list(bins.keys())
        exhausted: set = set()

        while len(exhausted) < n_bins:
            batch_trips: List[DemandTrip] = []
            for key in bin_keys:
                if key in exhausted:
                    continue
                it = bin_iters[key]
                for _ in range(max_per_bin):
                    try:
                        batch_trips.append(next(it))
                    except StopIteration:
                        exhausted.add(key)
                        break
            if batch_trips:
                batches.append(TripBatch(
                    trips=batch_trips,
                    batch_index=batch_idx,
                    departure_bin=dep_bin,
                ))
                batch_idx += 1

    return batches, max_bins


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
    effective_batch_size: int  # dynamic batch size used for this batch
    route_time_s: float
    customize_time_s: float
    engine_time_s: float
    batch_time_s: float
    tstt: float
    queue_vehicles: float  # mean unserved demand per link per lane (veh/hr/lane)
    total_unserved_vph: float  # network-wide total unserved (veh/hr)
    mean_speed_kmh: float
    min_speed_kmh: float
    p10_speed_kmh: float  # 10th percentile (flow-weighted)
    p50_speed_kmh: float  # 50th percentile (flow-weighted)
    n_oversaturated: int
    departure_bin: int = -1
    vc_cv: float = float('nan')            # V/C coefficient of variation
    mean_vc: float = float('nan')          # mean V/C of active links
    flow_stability: float = float('nan')   # ‖Δflow‖/‖flow‖ vs previous period-final
    sampled_gap: float = float('nan')      # sampled Wardrop relative gap


@dataclass
class AggregatedRoute:
    """A unique route with aggregated demand and optional trip IDs."""

    demand: float
    trip_ids: List[str] = field(default_factory=list)


@dataclass
class StreamResult:
    """Final result of stream assignment."""

    n_trips: int
    n_batches: int
    network_state: NetworkState
    batch_log: List[StreamBatchResult]
    total_time_s: float
    unserved_vph: np.ndarray  # per-link unserved demand at end of last period
    routes: Optional[Dict[str, AggregatedRoute]] = None  # polyline6 → route info
    period_csvs: Optional[List[str]] = None  # per-period speed CSV paths
    period_flows: Optional[np.ndarray] = None  # (n_periods, n_edges) 2D flow
    experienced_times: Optional[np.ndarray] = None  # per-trip travel time (s)
    phase2_tstt: Optional[float] = None  # TSTT from Phase 2 re-route

    def log_as_dict(self) -> Dict:
        """Convert batch log to dict for plotting."""
        return {
            "batch": [r.batch_index for r in self.batch_log],
            "n_trips": [r.n_trips for r in self.batch_log],
            "effective_batch_size": [r.effective_batch_size for r in self.batch_log],
            "tstt": [r.tstt for r in self.batch_log],
            "queue_vehicles": [r.queue_vehicles for r in self.batch_log],
            "total_unserved_vph": [r.total_unserved_vph for r in self.batch_log],
            "mean_speed_kmh": [r.mean_speed_kmh for r in self.batch_log],
            "min_speed_kmh": [r.min_speed_kmh for r in self.batch_log],
            "n_oversaturated": [r.n_oversaturated for r in self.batch_log],
            "route_time_s": [r.route_time_s for r in self.batch_log],
            "customize_time_s": [r.customize_time_s for r in self.batch_log],
            "vc_cv": [r.vc_cv for r in self.batch_log],
            "mean_vc": [r.mean_vc for r in self.batch_log],
            "flow_stability": [r.flow_stability for r in self.batch_log],
            "sampled_gap": [r.sampled_gap for r in self.batch_log],
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
        self._incr_customizer: Optional[osrm_module.IncrementalCustomizer] = None
        self._in_mem_customizer: Optional[osrm_module.InMemoryCustomizer] = None
        self._compressed_graph = None  # loaded lazily by run() / assign_stream()

    def _create_engine(self, *, quiet: bool = False) -> osrm_module.OSRM:
        """Create a fresh OSRM engine instance."""
        _log = logger.debug if quiet else logger.info
        _log("Loading OSRM engine from %s", self.base_path)
        t0 = time.monotonic()
        eng = osrm_module.OSRM(
            storage_config=self.base_path,
            algorithm="MLD",
            use_shared_memory=False,
            use_mmap=False,
        )
        _log("Engine loaded in %.2fs", time.monotonic() - t0)
        return eng

    def _customize_and_reload(
        self, csv_path: str, engine: "osrm_module.OSRM",
    ) -> tuple["osrm_module.OSRM", float, float]:
        """Customize OSRM with a speed CSV and update routing weights.

        Tries InMemoryCustomizer first (in-place metric swap, no engine
        reload), then IncrementalCustomizer, then full customize.

        Returns (engine, customize_time, engine_time).
        """
        t_cust = time.monotonic()

        # In-memory path (no engine reload needed)
        if self._in_mem_customizer is not None:
            raw_engine = getattr(engine, '_engine', engine)
            result = self._in_mem_customizer.recustomize(
                csv_path, raw_engine, filter_indices=[0],
            )
            customize_time = time.monotonic() - t_cust
            logger.debug(
                "InMemoryCustomizer: dirty=%d geom, %d edges, %d cells, %d clean | "
                "csv=%.3f copy=%.3f upd=%.3f acc=%.3f patch=%.3f cell=%.3f total=%.3fs",
                result["dirty_geometries"], result["edges_patched"],
                result["dirty_cells"], result["newly_clean"],
                result["phase1_csv_s"], result["phase1_copy_s"],
                result["phase1_update_s"], result["phase2_accum_s"],
                result["phase3_patch_s"], result["phase4_cell_s"],
                customize_time,
            )
            return engine, customize_time, 0.0

        # Try incremental path
        if self._incr_customizer is not None:
            try:
                result = self._incr_customizer.recustomize(
                    csv_path, n_threads=self.config.n_threads,
                )
                customize_time = time.monotonic() - t_cust
                t_eng = time.monotonic()
                del engine
                engine = self._create_engine(quiet=True)
                engine_time = time.monotonic() - t_eng
                logger.debug(
                    "Incremental customize: %d/%d cells in %.2fs + engine %.2fs",
                    result.dirty_cells, result.total_cells,
                    customize_time, engine_time,
                )
                return engine, customize_time, engine_time
            except Exception:
                logger.warning(
                    "IncrementalCustomizer failed, falling back to full customize",
                    exc_info=True,
                )
                self._incr_customizer = None

        # Full customize fallback
        osrm_module.customize(
            self.base_path,
            segment_speed_file=csv_path,
            verbosity="ERROR",
        )
        customize_time = time.monotonic() - t_cust
        t_eng = time.monotonic()
        del engine
        engine = self._create_engine(quiet=True)
        engine_time = time.monotonic() - t_eng
        return engine, customize_time, engine_time

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
                trip_id=trip.trip_id,
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
        *,
        return_routes: bool = False,
        departure_period: int = -1,
        period_duration: float = 0.0,
        n_periods: int = 0,
        period_flows: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, Optional[List[str]]]:
        """Route all trips, accumulate link volume.

        Uses C++ accumulation when available — route results never cross
        the C++/Python boundary, eliminating the Python per-segment loop.

        Parameters
        ----------
        departure_period : int
            Period index for this batch (-1 = disabled).
        period_duration : float
            Duration of each period in seconds (0 = disabled).
        n_periods : int
            Total number of periods for 2D attribution. When > 0, volume
            is returned as (n_periods, n_edges) with flow attributed to
            the period each segment falls in based on cumulative travel
            time.  When 0, returns 1D (n_edges,) as before.

        Returns
        -------
        volume : np.ndarray
            2D (n_periods, n_edges) when n_periods > 0, else 1D (n_edges,).
        aon_tstt : float
            AON total system travel time (vehicle-seconds).
        polylines : list of str or None
            Per-trip polyline6 geometry strings when return_routes=True.
        """
        return self._route_and_accumulate_cpp(
            engine, trips, state,
            return_routes=return_routes,
            departure_period=departure_period,
            period_duration=period_duration,
            n_periods=n_periods,
            period_flows=period_flows,
        )

    def _route_and_accumulate_cpp(
        self,
        engine: osrm_module.OSRM,
        trips: List[DemandTrip],
        state: NetworkState,
        *,
        return_routes: bool = False,
        departure_period: int = -1,
        period_duration: float = 0.0,
        n_periods: int = 0,
        period_flows: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, Optional[List[str]]]:
        """C++ fast path: route + accumulate in one native call.
        
        When period_flows is provided (writable 2D buffer), C++ accumulates
        directly into it — no intermediate sparse arrays or Python scatter.
        Returns None for volume in that case.
        Also stores trip_durations on self._last_trip_durations for callers.
        """
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

        # Build per-trip departure offsets (seconds into the period)
        departure_offsets = np.empty(0, dtype=np.float64)
        if departure_period >= 0 and period_duration > 0:
            period_start = departure_period * period_duration
            departure_offsets = np.array(
                [max(t.departure_time_s - period_start, 0.0) for t in trips],
                dtype=np.float64,
            )

        default_jam = (self.config.default_jam_density_per_lane
                       * self.config.default_n_lanes)

        # Pass compressed graph lookup if available
        compressed_kwargs = {}
        if self._compressed_graph is not None:
            compressed_kwargs = dict(compressed_graph=self._compressed_graph)

        volume, tstt, new_edges, route_geoms, trip_durations, route_ms, accum_ms = batch_route_accumulate(
            engine._engine,
            coords,
            volumes,
            state.edge_ids.astype(np.uint64),
            self.config.n_threads,
            return_routes,
            departure_period=departure_period,
            period_duration=period_duration,
            departure_offsets=departure_offsets,
            n_periods=n_periods,
            period_flows=period_flows,
            **compressed_kwargs,
        )

        logger.debug(
            "C++ batch_route_accumulate: %d trips, route=%.1fms, accum=%.1fms",
            n, route_ms, accum_ms,
        )

        # Register any newly discovered edges (vectorized)
        if len(new_edges) > 0:
            ne_arr = np.array(new_edges, dtype=np.float64)
            state.register_edges_batch(
                ne_arr[:, 0].astype(np.uint64),
                ne_arr[:, 1].astype(np.uint64),
                ne_arr[:, 2],
                ne_arr[:, 3],
                default_jam,
                self.config.default_n_lanes,
            )

        # Store per-trip durations for experienced_times collection
        self._last_trip_durations = np.asarray(trip_durations, dtype=np.float64)

        # In-place mode: volume is None, accumulation already done in C++
        if volume is None:
            return None, float(tstt), route_geoms

        volume = np.asarray(volume) if not isinstance(volume, tuple) else volume

        # For 1D (non-period) mode, pad to match state.n_edges
        if not isinstance(volume, tuple):
            volume = np.asarray(volume, dtype=np.float64)
            if len(volume) < state.n_edges:
                volume = np.append(volume,
                                   np.zeros(state.n_edges - len(volume)))

        return volume, float(tstt), route_geoms

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

        # Raw (unsmoothed) speed for unserved demand calculation
        state.speed_kmh_raw = self.vdf.density_to_speed(
            state.density_vpkm, state.freeflow_kmh, state.jam_density,
            kc_ratio=state.kc_ratio,
        )

        # Optional spatial smoothing (operates on density) — for routing only
        smoothed = self.smoother.smooth(state.density_vpkm)

        # Forward MFD: smoothed density → speed (used for OSRM customize)
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

        # Try to load compressed graph for correct freeflow speeds.
        # Falls back to empty state with on-the-fly edge discovery.
        state = NetworkState.empty()
        compressed_graph = None
        try:
            from osrm.osrm_ext import load_compressed_graph
            compressed_graph = load_compressed_graph(self.base_path)
            self._compressed_graph = compressed_graph
            logger.info(
                "Loaded compressed graph: %d driving edges, %d node pairs "
                "(%d non-driving geometries skipped)",
                compressed_graph.n_edges, compressed_graph.n_pairs,
                compressed_graph.skipped_non_driving,
            )
        except Exception as exc:
            logger.warning(
                "Could not load compressed graph (%s), falling back to discovery",
                exc,
            )

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

        # Initialize incremental customizer (caches partition/cells for
        # dirty-cell recustomization).  Falls back to full customize on error.
        try:
            self._incr_customizer = osrm_module.IncrementalCustomizer()
            self._incr_customizer.initialize(
                self.base_path,
                n_threads=self.config.n_threads,
                verbosity="WARNING",
            )
            logger.info("IncrementalCustomizer initialized")
        except Exception:
            logger.warning(
                "IncrementalCustomizer not available, using full customize",
                exc_info=True,
            )
            self._incr_customizer = None

        for step_frac in inc_steps:
            if step_frac >= 1.0:
                break  # 1.0 is handled by the main loop's first iteration
            n_inc += 1
            logger.debug(
                "=== Incremental step %d/%d (%.0f%% demand) ===",
                n_inc, len(inc_steps), step_frac * 100,
            )
            t_route = time.monotonic()
            aon_volume, aon_tstt, _ = self._route_and_accumulate(
                engine, trips, state,
            )
            route_time = time.monotonic() - t_route

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
            engine, _, _ = self._customize_and_reload(str(csv_path), engine)

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
            aon_volume, aon_tstt, _ = self._route_and_accumulate(
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
            csv_path = self.writer.write_from_state(state, only_changed=True)
            logger.info("Customizing OSRM (iter %d)...", n)
            engine, customize_time, engine_time = self._customize_and_reload(
                str(csv_path), engine,
            )

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
        period_duration_s: Optional[float] = None,
        return_routes: bool = False,
        state_patch=None,
        progress_callback=None,
    ) -> "StreamResult":
        """Forward-simulation assignment with incremental loading.

        Routes trips in batches, updating network state (VDF → customize)
        between batches.  Unserved demand (vehicles on links that exceed
        capacity) carries forward between time periods.

        Unlike ``assign_matrix()`` which iterates to Wardrop equilibrium,
        this performs a single forward pass — quality depends on batch
        granularity and the resulting customize frequency.

        Parameters
        ----------
        trips : list of DemandTrip
            All trips to assign.  If trips have varying
            ``departure_time_s``, they are sorted and routed in
            chronological order.
        period_duration_s : float or None
            Duration of each time period in seconds.  Trips are grouped
            by departure time into periods.  Unserved demand from period
            *i* carries forward as additional flow in period *i+1*.
            If ``None`` (default), all trips are a single period.
        return_routes : bool
            If True, capture polyline6-encoded route geometries from
            OSRM and aggregate them in the result.  Each unique route
            maps to an :class:`AggregatedRoute` with total demand and
            trip IDs (when ``DemandTrip.trip_id`` is set).
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
        t_start = time.monotonic()
        n_trips = len(trips)

        engine = self._create_engine()

        # Initialize InMemoryCustomizer for fast in-place metric swaps
        try:
            imc = osrm_module.InMemoryCustomizer()
            imc.initialize(
                self.base_path,
                threads=self.config.n_threads,
            )
            self._in_mem_customizer = imc
            logger.info("InMemoryCustomizer initialized")
        except Exception:
            logger.warning(
                "InMemoryCustomizer unavailable, using full customize",
                exc_info=True,
            )
            self._in_mem_customizer = None

        # Explode high-volume trips for path diversity, then build
        # spatially-diverse batches via OD-grid round-robin.
        # No explicit snap needed — OSRM Route internally snaps to nearest node.
        explosion_rng = np.random.default_rng(42)
        if self.config.max_vol_per_route > 0:
            trips = _explode_trips(
                trips, self.config.max_vol_per_route,
                self.config.explosion_jitter_m, explosion_rng,
            )
            n_trips = len(trips)

        # Data-driven cell size from median trip length
        cell_size_km = _compute_cell_size_km(trips)
        logger.info(
            "OD-grid cell size: %.1f km (median trip / 3)", cell_size_km,
        )

        # Build OD-cluster round-robin batches
        od_batches, n_od_bins = _build_od_batches(
            trips, period_duration_s, cell_size_km,
            self.config.max_trips_per_bin,
        )
        est_total_batches = len(od_batches)
        total_vol = sum(t.volume for trip_batch in od_batches for t in trip_batch.trips)

        logger.info(
            "Stream assignment: %d trips, %d OD bins, %d batches "
            "(max %d/bin), n_threads=%d",
            n_trips, n_od_bins, est_total_batches,
            self.config.max_trips_per_bin, self.config.n_threads,
        )
        if period_duration_s is not None:
            mins = int(period_duration_s / 60)
            n_per_day = int(24 * 3600 / period_duration_s)
            logger.info(
                "Period: %dmin (%ds), %d periods/day",
                mins, int(period_duration_s), n_per_day,
            )

        # IncrementalCustomizer was removed — always use full customize
        self._incr_customizer = None


        # Network starts empty — edges discovered during routing.
        # Load compressed graph lookup for correct freeflow speeds.
        state = NetworkState.empty()
        compressed_graph = None
        try:
            from osrm.osrm_ext import load_compressed_graph
            compressed_graph = load_compressed_graph(self.base_path)
            self._compressed_graph = compressed_graph
            logger.info(
                "Loaded compressed graph: %d driving edges, %d node pairs "
                "(%d non-driving geometries skipped)",
                compressed_graph.n_edges, compressed_graph.n_pairs,
                compressed_graph.skipped_non_driving,
            )
        except Exception as exc:
            logger.warning(
                "Could not load compressed graph (%s), falling back to discovery",
                exc,
            )
        if state_patch:
            state_patch(state)

        # Determine total periods for cross-period flow attribution
        n_periods = 0
        if period_duration_s is not None:
            # Cap at 24h worth of periods (departure times wrap modulo 24h)
            day_s = 24 * 3600.0
            n_periods = int(day_s / period_duration_s)

        # 2D period flow tracking: flow[period, edge]
        # Initialized lazily after first batch discovers edges
        period_flows: Optional[np.ndarray] = None
        period_csvs: List[str] = []

        # Legacy 1D tracking (single-period or no period_duration_s)
        queue_carryforward = np.zeros(state.n_edges, dtype=np.float64)

        batch_log: List[StreamBatchResult] = []
        route_demands: Optional[Dict[str, AggregatedRoute]] = None
        if return_routes:
            route_demands = {}

        batch_iter = iter(od_batches)

        # Track remaining volume for progress reporting
        trips_remaining = n_trips
        vol_remaining = total_vol

        # Cumulative flow within the current period (1D legacy path)
        cumulative_volume = queue_carryforward.copy()
        current_period_bin = None  # set from first batch
        n_batches = 0
        n_batches_in_period = 0
        all_trip_durations: List[np.ndarray] = []

        # Equilibrium quality tracking
        prev_period_final_flow: Optional[np.ndarray] = None
        _log = logger.info
        _header_logged = False

        for batch in batch_iter:
            if current_period_bin is None:
                current_period_bin = batch.departure_bin
            n_batches += 1
            n_batches_in_period += 1
            t_batch = time.monotonic()
            bi = batch.batch_index

            # Period transition: finalize previous period, carry forward
            if (
                period_duration_s is not None
                and batch.departure_bin != current_period_bin
            ):
                old_period = current_period_bin

                # ── Period-final equilibrium metrics ──────────────
                # Computed BEFORE state resets for the new period.
                if batch_log:
                    pf = batch_log[-1]  # last batch of completing period

                    # Flow stability: ‖Δflow‖/‖flow‖ vs previous period
                    if prev_period_final_flow is not None:
                        flow_now = state.flow_vph
                        norm_now = float(np.linalg.norm(flow_now))
                        if norm_now > 0:
                            delta = float(np.linalg.norm(
                                flow_now[:len(prev_period_final_flow)]
                                - prev_period_final_flow[:len(flow_now)]
                            ))
                            pf.flow_stability = delta / norm_now

                    prev_period_final_flow = state.flow_vph.copy()
                # ──────────────────────────────────────────────────

                if n_periods > 0 and period_flows is not None:
                    # Save per-period CSV for the completing period
                    p_idx = old_period if old_period is not None else 0
                    csv_path = self.writer.write_from_state(
                        state, suffix=f"_period{p_idx}", only_changed=True,
                    )
                    # Ensure period_csvs list is long enough
                    while len(period_csvs) <= p_idx:
                        period_csvs.append("")
                    period_csvs[p_idx] = str(csv_path)

                    # Simple carryforward: unserved demand seeds next period
                    unserved = state.unserved_demand
                    n_spill = int(np.sum(unserved > 0))
                    genuine = unserved > 1.0
                    n_genuine = int(np.sum(genuine))
                    total_genuine = float(np.sum(unserved[genuine]))

                    # Culprit details to DEBUG
                    if n_genuine > 0:
                        genuine_idx = np.where(genuine)[0]
                        genuine_unserved = unserved[genuine_idx]
                        top_order = np.argsort(genuine_unserved)[::-1][:20]
                        for rank, gi in enumerate(top_order):
                            ei = genuine_idx[gi]
                            u, v = int(state.edge_ids[ei, 0]), int(state.edge_ids[ei, 1])
                            logger.debug(
                                "    culprit #%d: edge=%d→%d vf=%.1f q_c=%.0f "
                                "flow=%.0f unserved=%.0f",
                                rank + 1, u, v,
                                state.freeflow_kmh[ei],
                                self.vdf.capacity_flow(
                                    state.freeflow_kmh[ei:ei+1],
                                    state.jam_density[ei:ei+1],
                                )[0],
                                state.flow_vph[ei],
                                unserved[ei],
                            )

                    new_period = batch.departure_bin
                    if new_period is not None and new_period < n_periods:
                        if period_flows.shape[1] < state.n_edges:
                            pad = state.n_edges - period_flows.shape[1]
                            period_flows = np.pad(period_flows, ((0, 0), (0, pad)))
                        period_flows[new_period, :state.n_edges] += unserved

                    # Set state to new period's accumulated flow for VDF
                    state.flow_vph = np.maximum(
                        period_flows[new_period, :], 0.0
                    )
                else:
                    n_genuine = 0
                    total_genuine = 0.0
                    # Legacy 1D: simple carryforward
                    unserved = state.unserved_demand
                    n_spill = int(np.sum(unserved > 0))
                    total_spill = float(np.sum(unserved))
                    if total_spill > 1.0:
                        n_genuine = n_spill
                        total_genuine = total_spill
                    queue_carryforward = unserved.copy()
                    cumulative_volume = queue_carryforward.copy()
                    state.flow_vph = np.maximum(cumulative_volume, 0.0)

                current_period_bin = batch.departure_bin

                # Re-customize for new period's starting state (includes
                # unserved carryforward).  The last batch already customized,
                # but that was on old-period flow — we need weights that
                # reflect the new period's baseline (unserved demand).
                self._update_state(state)
                self.writer.reset_delta()
                csv_path = self.writer.write_from_state(state, only_changed=True)
                engine, period_cust_time, _ = self._customize_and_reload(str(csv_path), engine)
                n_batches_in_period = 1  # this batch starts the new period

                # Log period transition with ETA
                elapsed_so_far = time.monotonic() - t_start
                avg_s = elapsed_so_far / n_batches if n_batches > 0 else 0
                remaining = max(0, est_total_batches - n_batches)
                eta_str = _fmt_eta(avg_s * remaining) if n_batches > 0 else ""
                if n_genuine > 0:
                    _log(
                        " → %s: %s veh/hr on %d links carried forward  [ETA %s]",
                        _period_to_clock(batch.departure_bin, period_duration_s or 3600.0),
                        _fmt_num(total_genuine), n_genuine, eta_str,
                    )
                else:
                    _log(
                        " → %s: 0 carried forward  [ETA %s]",
                        _period_to_clock(batch.departure_bin, period_duration_s or 3600.0),
                        eta_str,
                    )
                _log(_batch_header(self.config.log_timing))

            # 1. Route this batch against current (congested) weights
            t_route = time.monotonic()

            # Initialize period_flows lazily before first batch
            if n_periods > 0 and period_flows is None and state.n_edges > 0:
                period_flows = np.zeros(
                    (n_periods, state.n_edges), dtype=np.float64
                )

            # Grow period_flows if new edges were discovered
            if period_flows is not None and period_flows.shape[1] < state.n_edges:
                pad = state.n_edges - period_flows.shape[1]
                period_flows = np.pad(period_flows, ((0, 0), (0, pad)))
                if state_patch:
                    state_patch(state)
                self.smoother.build_adjacency(state.edge_ids, state.length_m)

            aon_volume, aon_tstt, batch_polylines = self._route_and_accumulate(
                engine, batch.trips, state,
                return_routes=return_routes,
                departure_period=(batch.departure_bin
                                  if batch.departure_bin is not None else -1),
                period_duration=(period_duration_s
                                 if period_duration_s is not None else 0.0),
                n_periods=n_periods,
                period_flows=period_flows,
            )
            route_time = time.monotonic() - t_route

            # Collect per-trip durations for experienced_times
            if hasattr(self, '_last_trip_durations'):
                all_trip_durations.append(self._last_trip_durations)

            # Aggregate polylines into route_demands dict
            if return_routes and batch_polylines is not None:
                for trip, polyline in zip(batch.trips, batch_polylines):
                    if not polyline:
                        continue
                    rec = route_demands.get(polyline)
                    if rec is None:
                        rec = AggregatedRoute(demand=0.0)
                        route_demands[polyline] = rec
                    rec.demand += trip.volume
                    if trip.trip_id is not None:
                        rec.trip_ids.append(trip.trip_id)

            # 2. Accumulate flow
            t_accum = time.monotonic()
            if n_periods > 0 and aon_volume is None:
                # In-place path: C++ already wrote into period_flows buffer.
                # Just update state.flow_vph for VDF.
                cp = current_period_bin if current_period_bin is not None else 0
                if cp < n_periods:
                    state.flow_vph = np.maximum(period_flows[cp, :], 0.0)
                else:
                    state.flow_vph = np.maximum(
                        np.sum(period_flows, axis=0), 0.0
                    )
            elif n_periods > 0 and isinstance(aon_volume, tuple):
                # Sparse COO fallback (new edges exceeded buffer, or first batch)
                sp_periods, sp_edges, sp_flows = aon_volume

                # Initialize period_flows now that we know edge count
                if period_flows is None:
                    period_flows = np.zeros(
                        (n_periods, state.n_edges), dtype=np.float64
                    )
                # Grow if needed
                if period_flows.shape[1] < state.n_edges:
                    pad = state.n_edges - period_flows.shape[1]
                    period_flows = np.pad(period_flows, ((0, 0), (0, pad)))

                np.add.at(period_flows, (sp_periods, sp_edges), sp_flows)

                # Current period's total flow for VDF
                cp = current_period_bin if current_period_bin is not None else 0
                if cp < n_periods:
                    state.flow_vph = np.maximum(period_flows[cp, :], 0.0)
                else:
                    # Fallback: sum all periods
                    state.flow_vph = np.maximum(
                        np.sum(period_flows, axis=0), 0.0
                    )
            else:
                # Legacy 1D path
                # Grow arrays if new edges were discovered
                if len(cumulative_volume) < state.n_edges:
                    pad = state.n_edges - len(cumulative_volume)
                    cumulative_volume = np.append(
                        cumulative_volume, np.zeros(pad)
                    )
                    queue_carryforward = np.append(
                        queue_carryforward,
                        np.zeros(state.n_edges - len(queue_carryforward)),
                    )
                    if state_patch:
                        state_patch(state)
                    self.smoother.build_adjacency(state.edge_ids, state.length_m)

                cumulative_volume += aon_volume
                state.flow_vph = np.maximum(cumulative_volume, 0.0)

            # 3. VDF: flow → density → speed
            accum_time = time.monotonic() - t_accum
            t_vdf = time.monotonic()
            self._update_state(state)

            # 4. Unserved demand: flow exceeding physical throughput
            unserved_vph = state.unserved_demand
            unserved_per_lane = unserved_vph / np.maximum(state.n_lanes, 1)
            oversat_mask = unserved_vph > 0
            mean_queue_per_lane = (
                float(np.mean(unserved_per_lane[oversat_mask]))
                if np.any(oversat_mask) else 0.0
            )
            total_unserved = float(np.sum(unserved_vph))
            vdf_time = time.monotonic() - t_vdf

            # 5. Write CSV and re-customize OSRM
            # Skip if this is the only batch in the period so far — the
            # period-transition customize will handle it with unserved
            # demand included, avoiding a redundant ~16s customize.
            customize_time = 0.0
            engine_time = 0.0
            csv_time = 0.0
            if n_batches_in_period > 1:
                t_csv = time.monotonic()
                csv_path = self.writer.write_from_state(state, delta=True)
                csv_time = time.monotonic() - t_csv
                engine, customize_time, engine_time = self._customize_and_reload(
                    str(csv_path), engine,
                )

            # Metrics — use flow-weighted mean speed (stable denominator)
            t_metrics = time.monotonic()
            active = state.flow_vph > 0
            total_flow = float(np.sum(state.flow_vph))
            if total_flow > 0:
                weighted_speed = float(
                    np.sum(state.speed_kmh * state.flow_vph) / total_flow
                )
            else:
                weighted_speed = float(np.mean(state.speed_kmh))
            active_speeds = state.speed_kmh[active] if np.any(active) else state.speed_kmh
            # Flow-weighted speed percentiles
            if total_flow > 0:
                a_flow = state.flow_vph[active] if np.any(active) else state.flow_vph
                si = np.argsort(active_speeds)
                cs = active_speeds[si]
                cf = np.cumsum(a_flow[si])
                cf /= cf[-1]
                p10_speed = float(cs[np.searchsorted(cf, 0.10)])
                p50_speed = float(cs[np.searchsorted(cf, 0.50)])
            else:
                p10_speed = float(np.percentile(active_speeds, 10))
                p50_speed = float(np.percentile(active_speeds, 50))
            link_time_s = state.length_m * 3.6 / np.maximum(
                state.speed_kmh, self.config.vdf_min_speed_kmh,
            )
            tstt = float(np.sum(state.flow_vph * link_time_s))

            # V/C coefficient of variation (utilization uniformity)
            q_c = self.vdf.capacity_flow(
                state.freeflow_kmh, state.jam_density,
                kc_ratio=state.kc_ratio,
            )
            vc = state.flow_vph / np.maximum(q_c, 1.0)
            max_vc = float(np.max(vc)) if len(vc) > 0 else 0.0
            vc_active = vc[active] if np.any(active) else vc
            vc_mean = float(np.mean(vc_active))
            vc_cv = float(np.std(vc_active) / max(vc_mean, 1e-9))
            metrics_time = time.monotonic() - t_metrics

            batch_result = StreamBatchResult(
                batch_index=bi,
                n_trips=len(batch.trips),
                effective_batch_size=len(batch.trips),
                route_time_s=route_time,
                customize_time_s=customize_time,
                engine_time_s=engine_time,
                batch_time_s=time.monotonic() - t_batch,
                tstt=tstt,
                queue_vehicles=mean_queue_per_lane,
                total_unserved_vph=total_unserved,
                mean_speed_kmh=weighted_speed,
                min_speed_kmh=float(np.min(active_speeds)),
                p10_speed_kmh=p10_speed,
                p50_speed_kmh=p50_speed,
                n_oversaturated=int(np.sum(
                    state.density_vpkm > self.vdf.critical_density(state.jam_density, kc_ratio=state.kc_ratio)
                )),
                departure_bin=(batch.departure_bin
                               if batch.departure_bin is not None else -1),
                vc_cv=vc_cv,
                mean_vc=vc_mean,
            )
            batch_log.append(batch_result)

            batch_vol = sum(t.volume for t in batch.trips)
            trips_remaining -= len(batch.trips)
            vol_remaining -= batch_vol

            # Spatial concentration: top V/C links (moved to DEBUG)
            n_active = int(np.sum(active))
            if n_active > 0:
                active_vc = vc[active]
                active_flows = state.flow_vph[active]
                active_qc = q_c[active]
                active_vf = state.freeflow_kmh[active]
                top_idx = np.argsort(active_vc)[-5:][::-1]
                top5_str = ", ".join(
                    f"{active_flows[i]:.0f}/{active_qc[i]:.0f}(vf={active_vf[i]:.0f})"
                    for i in top_idx
                )
                logger.debug("  top V/C: %s", top5_str)

            # ETA from running average batch time
            elapsed_s = time.monotonic() - t_start
            avg_batch_s = elapsed_s / n_batches if n_batches > 0 else 0
            eta_batches = max(0, est_total_batches - n_batches)
            eta_s = avg_batch_s * eta_batches

            # Table-format batch log with progress
            period_label = current_period_bin if current_period_bin is not None else "?"
            if not _header_logged:
                _log(_batch_header(self.config.log_timing))
                _log(_batch_sep(self.config.log_timing))
                _header_logged = True
            _log(_fmt_batch_row(
                n_batches, est_total_batches, period_label,
                len(batch.trips), batch_vol,
                vol_remaining, batch_result.mean_vc, max_vc,
                batch_result.mean_speed_kmh, batch_result.n_oversaturated,
                n_active, route_time, accum_time, customize_time,
                eta_s=eta_s, timing=self.config.log_timing,
                period_duration_s=period_duration_s or 3600.0,
            ))

            if progress_callback:
                progress_callback(bi, n_trips, mean_queue_per_lane)

        # Final period equilibrium metrics (no transition triggers these)
        if batch_log:
            pf = batch_log[-1]
            if prev_period_final_flow is not None:
                flow_now = state.flow_vph
                norm_now = float(np.linalg.norm(flow_now))
                if norm_now > 0:
                    delta = float(np.linalg.norm(
                        flow_now[:len(prev_period_final_flow)]
                        - prev_period_final_flow[:len(flow_now)]
                    ))
                    pf.flow_stability = delta / norm_now

        total_time = time.monotonic() - t_start
        del engine

        # Save final period's CSV if in multi-period mode
        if n_periods > 0 and period_flows is not None and current_period_bin is not None:
            p_idx = current_period_bin
            csv_path = self.writer.write_from_state(
                state, suffix=f"_period{p_idx}", only_changed=True,
            )
            while len(period_csvs) <= p_idx:
                period_csvs.append("")
            period_csvs[p_idx] = str(csv_path)

        # Don't clean up period CSVs — they're needed for Phase 2
        if not period_csvs:
            self.writer.cleanup()

        # Final unserved demand for reporting
        final_unserved = state.unserved_demand if state.n_edges > 0 else np.array([])

        logger.info(
            "Stream complete: %d trips in %d batches, %.1fs, "
            "final queue=%.0f veh/hr/lane, total_unserved=%.0f veh/hr",
            n_trips, n_batches, total_time,
            batch_log[-1].queue_vehicles if batch_log else 0,
            batch_log[-1].total_unserved_vph if batch_log else 0,
        )

        if n_periods > 0 and period_csvs:
            non_empty = [p for p in period_csvs if p]
            logger.info(
                "Saved %d per-period speed CSVs for Phase 2 multi-period "
                "customize",
                len(non_empty),
            )

        if return_routes and route_demands:
            logger.info(
                "Captured %d unique routes (from %d trips)",
                len(route_demands), n_trips,
            )

        # Build experienced_times from collected trip durations
        experienced_times = None
        if all_trip_durations:
            experienced_times = np.concatenate(all_trip_durations)

        return StreamResult(
            n_trips=n_trips,
            n_batches=n_batches,
            network_state=state,
            batch_log=batch_log,
            total_time_s=total_time,
            unserved_vph=final_unserved,
            routes=route_demands,
            period_csvs=period_csvs if period_csvs else None,
            period_flows=period_flows,
            experienced_times=experienced_times,
        )

    def reroute_time_dependent(
        self,
        trips: List[DemandTrip],
        phase1_result: "StreamResult",
        period_duration_s: float,
    ) -> "StreamResult":
        """Phase 2: re-route all trips with multi-period time-dependent weights.

        Takes per-period speed CSVs from a Phase 1 ``assign_stream`` run,
        builds a single OSRM engine with all period weights via
        ``customize_multi_period``, then re-routes every trip with
        ``departure_period`` and ``departure_offset`` so OSRM switches
        cell metrics and boundary weights mid-route as accumulated travel
        time crosses period boundaries.

        Parameters
        ----------
        trips : list of DemandTrip
            Same trips used in Phase 1 (with departure_time_s set).
        phase1_result : StreamResult
            Result from ``assign_stream`` with ``period_csvs`` populated.
        period_duration_s : float
            Duration of each period in seconds (must match Phase 1).

        Returns
        -------
        StreamResult
            Updated result with ``experienced_times`` (per-trip travel
            time in seconds) and ``phase2_tstt`` populated.
        """
        from osrm.preprocessing import customize_multi_period

        if not phase1_result.period_csvs:
            raise ValueError(
                "Phase 1 result has no period_csvs — run assign_stream "
                "with period_duration_s to generate per-period speed CSVs."
            )

        t_start = time.monotonic()

        # Build (period_index, csv_path) tuples, skipping empty entries
        period_speed_files = []
        for p_idx, csv_path in enumerate(phase1_result.period_csvs):
            if csv_path:
                period_speed_files.append((p_idx, csv_path))

        if not period_speed_files:
            raise ValueError("No non-empty period CSVs found in Phase 1 result.")

        logger.info(
            "Phase 2: customize_multi_period with %d period CSVs",
            len(period_speed_files),
        )

        # Multi-period customize: builds one engine with all period weights
        t_cust = time.monotonic()
        customize_multi_period(
            self.base_path,
            period_speed_files=period_speed_files,
            threads=self.config.n_threads or None,
            verbosity="ERROR",
        )
        cust_time = time.monotonic() - t_cust
        logger.info("Phase 2: customize took %.1fs", cust_time)

        # Load time-dependent engine
        t_eng = time.monotonic()
        engine = self._create_engine(quiet=True)
        eng_time = time.monotonic() - t_eng
        logger.info("Phase 2: engine load took %.1fs", eng_time)

        # Route all trips with time-dependent period params via C++ batch.
        from osrm.assignment.trip_stream import TripStreamAdapter
        adapter = TripStreamAdapter(trips, sort_by_departure=True)
        sorted_trips = adapter.trips()
        n_trips = len(sorted_trips)

        # Snap coordinates
        snapped = self._snap_trips(engine, sorted_trips)

        # Build coordinate, volume, and offset arrays for all trips
        coords = np.empty((n_trips, 4), dtype=np.float64)
        volumes = np.empty(n_trips, dtype=np.float64)
        offsets = np.empty(n_trips, dtype=np.float64)

        for i, t in enumerate(snapped):
            coords[i, 0] = t.origin[0]
            coords[i, 1] = t.origin[1]
            coords[i, 2] = t.destination[0]
            coords[i, 3] = t.destination[1]
            volumes[i] = t.volume
            offsets[i] = t.departure_time_s  # absolute offset from time 0

        # Use period 0 as base; offsets are absolute seconds from time 0,
        # and OSRM's GetPeriodForWeight computes the correct period from
        # departure_offset / period_duration.
        base_period = 0

        from osrm.osrm_ext import batch_route_accumulate

        logger.info("Phase 2: routing %d trips with time-dependent weights", n_trips)
        t_route = time.monotonic()

        # Pass compressed graph lookup if available
        compressed_kwargs2 = {}
        if self._compressed_graph is not None:
            compressed_kwargs2 = dict(compressed_graph=self._compressed_graph)

        vol, total_tstt, new_edges, _, trip_durations, route_ms, accum_ms = batch_route_accumulate(
            engine._engine,
            coords,
            volumes,
            phase1_result.network_state.edge_ids.astype(np.uint64),
            self.config.n_threads,
            False,  # return_routes
            departure_period=base_period,
            period_duration=period_duration_s,
            departure_offsets=offsets,
            n_periods=0,  # 1D volume — we only need per-trip durations
            **compressed_kwargs2,
        )

        experienced_times = np.asarray(trip_durations, dtype=np.float64)
        route_time = time.monotonic() - t_route
        total_time = time.monotonic() - t_start
        del engine

        logger.info(
            "Phase 2 complete: %d trips, tstt=%.0f, "
            "route=%.1fs, total=%.1fs",
            n_trips, total_tstt, route_time, total_time,
        )

        # Update the Phase 1 result with Phase 2 data
        phase1_result.experienced_times = experienced_times
        phase1_result.phase2_tstt = total_tstt
        return phase1_result
