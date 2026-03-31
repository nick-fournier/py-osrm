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
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import osrm as osrm_module
from osrm.assignment.density_smoothing import DensitySmoothing, DensitySmoothingConfig
from osrm.assignment.fractional_loading import FractionalLoader
from osrm.assignment.network_state import NetworkState
from osrm.assignment.od_matrix import DemandTrip, ODMatrixAdapter
from osrm.assignment.segment_speed_writer import SegmentSpeedWriter
from osrm.assignment.vdf import BiParabolicVDF

logger = logging.getLogger(__name__)


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
    verbosity: str = "ERROR"
    fw_bisections: int = 20
    fw_line_search_tol: float = 1e-6
    incremental_steps: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)
    stagnation_tol: float = 0.001
    stagnation_window: int = 3


@dataclass
class IterationResult:
    """Metrics for a single assignment iteration."""

    iteration: int
    relative_gap: float
    tstt: float
    max_density_delta: float
    n_oversaturated: int
    route_time_s: float
    customize_time_s: float
    engine_time_s: float = 0.0
    gap_time_s: float = 0.0
    step_size: float = 0.0


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

    def log_as_dict(self) -> Dict:
        """Convert iteration log to dict for plotting."""
        return {
            "iteration": [r.iteration for r in self.iteration_log],
            "relative_gap": [r.relative_gap for r in self.iteration_log],
            "tstt": [r.tstt for r in self.iteration_log],
            "max_density_delta": [r.max_density_delta for r in self.iteration_log],
            "step_size": [r.step_size for r in self.iteration_log],
            "route_time_s": [r.route_time_s for r in self.iteration_log],
            "customize_time_s": [r.customize_time_s for r in self.iteration_log],
            "engine_time_s": [r.engine_time_s for r in self.iteration_log],
            "gap_time_s": [r.gap_time_s for r in self.iteration_log],
        }


class AssignmentLoop:
    """Orchestrates iterative traffic assignment using OSRM.

    Parameters
    ----------
    base_path : str
        Path to the OSRM data files (must be partitioned for MLD).
    config : AssignmentConfig
        Assignment parameters.

    Example
    -------
    >>> loop = AssignmentLoop("network.osrm")
    >>> adapter = ODMatrixAdapter(origins, destinations, matrix)
    >>> result = loop.run(adapter.trips())
    >>> print(f"Converged: {result.converged}, Gap: {result.final_gap:.4f}")
    """

    def __init__(
        self,
        base_path: str,
        config: AssignmentConfig | None = None,
    ) -> None:
        self.base_path = str(base_path)
        self.config = config or AssignmentConfig()
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
        return osrm_module.OSRM(
            storage_config=self.base_path,
            algorithm="MLD",
            use_shared_memory=False,
        )

    def _route_single(
        self, engine: osrm_module.OSRM, trip: DemandTrip,
        alternatives: bool = False,
    ) -> Optional[dict]:
        """Route a single OD pair with annotations."""
        try:
            kwargs = dict(
                coordinates=[trip.origin, trip.destination],
                annotations=["nodes", "distance", "duration", "speed"],
            )
            if alternatives:
                kwargs["alternatives"] = True
            result = engine.Route(**kwargs)
            if result.get("code") == "Ok" and result.get("routes"):
                return result
        except Exception as e:
            logger.debug(f"Route failed for {trip.origin}→{trip.destination}: {e}")
        return None

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

        # Rebuild trips with snapped coordinates
        snapped_trips = []
        for trip in trips:
            snapped_trips.append(DemandTrip(
                origin=snapped[tuple(trip.origin)],
                destination=snapped[tuple(trip.destination)],
                volume=trip.volume,
            ))

        n_moved = sum(
            1 for c in unique_coords
            if abs(snapped[c][0] - c[0]) > 1e-8 or abs(snapped[c][1] - c[1]) > 1e-8
        )
        logger.info(
            "Snapped %d/%d unique coordinates (max shift: %.2fm)",
            n_moved, len(unique_coords),
            max(
                ((snapped[c][0] - c[0])**2 + (snapped[c][1] - c[1])**2)**0.5 * 111_000
                for c in unique_coords
            ) if unique_coords else 0,
        )
        return snapped_trips

    def _discover_network(
        self,
        engine: osrm_module.OSRM,
        trips: List[DemandTrip],
    ) -> NetworkState:
        """Route all trips with alternatives to discover network edges.

        IMPORTANT: ``engine`` must be a clean (uncustomized) OSRM instance
        so that annotation speed reflects the original profile speed from
        OSM ``maxspeed`` tags.  This speed is stored as ``freeflow_kmh``
        and is never updated — it is an immutable physical road attribute.

        Segment-speed customization permanently mutates OSRM edge weights
        (only re-extract from OSM resets them), so discovery cannot be
        repeated on a contaminated instance.

        Requests alternatives to capture non-shortest paths that may
        become attractive under congestion.
        """
        logger.info("Discovering network edges from %d OD pairs...", len(trips))
        route_results = []
        for trip in trips:
            result = self._route_single(engine, trip, alternatives=True)
            if result:
                route_results.append(result)

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
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Route all trips, accumulate link density and volume.

        Routes OD pairs in parallel using a thread pool (OSRM releases
        the GIL during C++ routing), then accumulates results
        sequentially to safely mutate shared NetworkState.

        Returns two per-link arrays:

        * **aon_density** (supply-side): average density contribution
          ``Δk = volume / (v_kmh × T_hr)`` using OSRM annotation speed.
        * **aon_volume** (demand-side): total vehicles routed through
          each link, always conserves with total demand.

        Any edges not yet in ``state`` are dynamically registered so
        that density is never silently dropped.

        Returns
        -------
        aon_density : np.ndarray
            All-or-nothing density (veh/km) from this iteration.
        aon_volume : np.ndarray
            All-or-nothing demand volume (vehicles) per link.
        aon_tstt : float
            AON total system travel time (vehicle-seconds).
        """
        new_density = np.zeros(state.n_edges, dtype=np.float64)
        new_volume = np.zeros(state.n_edges, dtype=np.float64)
        tstt = 0.0
        bin_width_hr = self.config.bin_width_s / 3600.0

        # Build OD dict for bulk_route
        od_dict = {
            "origin_lon": [t.origin[0] for t in trips],
            "origin_lat": [t.origin[1] for t in trips],
            "dest_lon": [t.destination[0] for t in trips],
            "dest_lat": [t.destination[1] for t in trips],
        }

        raw_results = osrm_module.bulk_route(
            engine, od_dict,
            annotations=["nodes", "distance", "duration", "speed"],
            raw_response=True,
            show_progress=False,
        )

        # Walk raw responses to accumulate density and volume
        responses = raw_results.get("_response", [])
        for trip_idx, response in enumerate(responses):
            if response is None:
                continue
            trip = trips[trip_idx]
            routes = response.get("routes")
            if not routes:
                continue
            route = routes[0]
            tstt += trip.volume * route["duration"]

            for leg in route["legs"]:
                ann = leg.get("annotation", {})
                nodes = ann.get("nodes", [])
                distances = ann.get("distance", [])
                speeds = ann.get("speed", [])

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
                    new_density[idx] += trip.volume / (seg_speed_kmh * bin_width_hr)

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
        log: List[IterationResult] = []

        # 1. Discover network on current OSRM state.
        # IMPORTANT: the engine MUST be clean (no prior segment-speed
        # customization) so annotation speed = freeflow from OSM maxspeed.
        # See class docstring for the freeflow invariant.
        engine = self._create_engine()

        # Pre-snap trip coordinates to OSRM waypoints so routes
        # start/end at actual network nodes, not mid-segment phantom
        # points.  Eliminates numerator/denominator gap inconsistency.
        trips = self._snap_trips(engine, trips)

        state = self._discover_network(engine, trips)

        if state_patch:
            state_patch(state)

        # Build smoothing adjacency once
        self.smoother.build_adjacency(state.edge_ids, state.length_m)

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
        for step_frac in inc_steps:
            if step_frac >= 1.0:
                break  # 1.0 is handled by the main loop's first iteration
            n_inc += 1
            logger.info(
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
            osrm_module.customize(
                self.base_path,
                segment_speed_file=str(csv_path),
                verbosity=self.config.verbosity,
            )
            del engine
            engine = self._create_engine()

            logger.info(
                "Incremental step %d: frac=%.2f, max_k/kj=%.2f, route=%.3fs",
                n_inc, step_frac,
                float(np.max(state.density_vpkm / state.jam_density)),
                route_time,
            )

        stop_reason: Optional[StopReason] = None
        stagnation_count = 0
        fw_zero_count = 0

        for n in range(1, self.config.max_iterations + 1):
            logger.info("=== Iteration %d ===", n)

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

            osrm_module.customize(
                self.base_path,
                segment_speed_file=str(csv_path),
                verbosity=self.config.verbosity,
            )
            customize_time = time.monotonic() - t_cust

            # 7. Reload engine
            t_engine = time.monotonic()
            del engine
            engine = self._create_engine()
            engine_time = time.monotonic() - t_engine

            iter_result = IterationResult(
                iteration=n,
                relative_gap=gap,
                tstt=tstt,
                max_density_delta=max_delta,
                n_oversaturated=n_oversat,
                route_time_s=route_time,
                customize_time_s=customize_time,
                engine_time_s=engine_time,
                gap_time_s=gap_time,
                step_size=alpha,
            )
            log.append(iter_result)

            logger.info(
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
                logger.info(
                    "%s at iteration %d (gap=%.6f)",
                    stop_reason.value.capitalize(), n, gap,
                )
                break

        total_time = time.monotonic() - t_start
        del engine

        # Cleanup CSV
        self.writer.cleanup()

        if stop_reason is None:
            stop_reason = StopReason.MAX_ITERATIONS

        return AssignmentResult(
            converged=stop_reason == StopReason.CONVERGED,
            iterations=len(log),
            final_gap=log[-1].relative_gap if log else float("inf"),
            network_state=state,
            iteration_log=log,
            total_time_s=total_time,
            stop_reason=stop_reason,
        )
