"""Traffic assignment loop orchestrator.

Implements the iterative equilibrium assignment cycle:
  Route → Decompose → Accumulate → VDF → CSV → Customize → Reload

Supports Method of Successive Averages (MSA) for convergence.

See docs/traffic_assignment_design.md §4 and §7 for full specification.
"""

from __future__ import annotations

import logging
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


@dataclass
class AssignmentConfig:
    """Configuration for the assignment loop.

    Parameters
    ----------
    method : str
        Convergence method: ``"msa"`` (Method of Successive Averages,
        fixed step 1/n) or ``"fw"`` (Frank-Wolfe with Beckmann line
        search for optimal step size). Default ``"msa"``.
    fw_bisections : int
        Maximum bisection iterations for the FW line search.
    fw_line_search_tol : float
        Convergence tolerance for the FW line search bracket width.
    """

    method: str = "msa"
    max_iterations: int = 50
    convergence_gap: float = 0.01
    bin_width_s: float = 3600.0
    min_speed_kmh: float = 1.0
    vdf_min_speed_kmh: float = 0.01
    smoothing: DensitySmoothingConfig = field(
        default_factory=DensitySmoothingConfig
    )
    vdf_kc_ratio: float = 1.0 / 3.0
    default_jam_density_per_lane: float = 200.0
    default_n_lanes: int = 1
    speed_csv_dir: Optional[str] = None
    verbosity: str = "ERROR"
    fw_bisections: int = 20
    fw_line_search_tol: float = 1e-6


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

    def log_as_dict(self) -> Dict:
        """Convert iteration log to dict for plotting."""
        return {
            "iteration": [r.iteration for r in self.iteration_log],
            "relative_gap": [r.relative_gap for r in self.iteration_log],
            "tstt": [r.tstt for r in self.iteration_log],
            "max_density_delta": [r.max_density_delta for r in self.iteration_log],
            "step_size": [r.step_size for r in self.iteration_log],
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

    def _discover_network(
        self,
        engine: osrm_module.OSRM,
        trips: List[DemandTrip],
    ) -> NetworkState:
        """Route all trips with alternatives to discover network edges.

        Uses freeflow routing (before any congestion) to build the
        edge registry from route annotations. Requests alternatives
        to capture non-shortest paths that may become attractive
        under congestion.
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

        for trip in trips:
            result = self._route_single(engine, trip)
            if result is None:
                continue

            route = result["routes"][0]
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
                        # Grow arrays to match
                        if idx >= len(new_density):
                            pad = state.n_edges - len(new_density)
                            new_density = np.append(new_density, np.zeros(pad))
                            new_volume = np.append(new_volume, np.zeros(pad))

                    # Demand-side volume (always conserves)
                    new_volume[idx] += trip.volume

                    # Density contribution: k = volume / (v × T)
                    seg_speed_kmh = (speeds[i] * 3.6) if i < len(speeds) else 1.0
                    seg_speed_kmh = max(seg_speed_kmh, self.config.min_speed_kmh)
                    new_density[idx] += trip.volume / (seg_speed_kmh * bin_width_hr)

        return new_density, new_volume, tstt

    def _compute_relative_gap(
        self,
        engine: osrm_module.OSRM,
        trips: List[DemandTrip],
        state: NetworkState,
        blended_volume: np.ndarray,
    ) -> float:
        """Compute Wardrop relative gap.

        gap = Σ_a (V_a · t_a) / Σ_rs (d_rs · π_rs) − 1

        where V_a is the demand-side volume on link a (always conserves
        with total demand), t_a = L_a / v_a is the link travel time at
        current density, and π_rs is the shortest-path cost on the
        updated network.

        Using demand volume (not MFD throughput) ensures the numerator
        accounts for all vehicles, even on oversaturated links.
        """
        # Link travel time from current density-derived speed
        link_time_s = state.length_m * 3.6 / np.maximum(state.speed_kmh, 1.0)

        # Numerator: total veh·s on network (demand-side)
        numerator = float(np.sum(blended_volume * link_time_s))

        if numerator == 0:
            return 0.0

        # Denominator: demand × shortest path cost on current network
        denominator = 0.0
        for trip in trips:
            result = self._route_single(engine, trip)
            if result and result.get("routes"):
                shortest_time = result["routes"][0]["duration"]
                denominator += trip.volume * shortest_time

        if denominator == 0:
            return 0.0

        return max(0.0, numerator / denominator - 1.0)

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

        At alpha=0, g(0) = sum_a c(k_current) * (V_aon - V_current) <= 0
        because AON routes on shortest paths, guaranteeing the improving
        direction property of Frank-Wolfe.
        """
        dk = k_aon - k_current
        dv = v_aon - v_current
        v_f = state.freeflow_kmh
        k_j = state.jam_density
        length_m = state.length_m

        def gradient(alpha: float) -> float:
            k_trial = k_current + alpha * dk
            k_trial = np.minimum(k_trial, k_j)
            k_trial = np.maximum(k_trial, 0.0)
            speed = self.vdf.density_to_speed(k_trial, v_f, k_j)
            cost = length_m * 3.6 / np.maximum(speed, self.config.vdf_min_speed_kmh)
            return float(np.dot(cost, dv))

        g_lo = gradient(0.0)
        g_hi = gradient(1.0)

        # If AON direction doesn't reduce cost, don't step
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
            OSRM annotations.

        Returns
        -------
        AssignmentResult
        """
        t_start = time.monotonic()
        log: List[IterationResult] = []

        # 1. Initial routing on freeflow to discover network
        engine = self._create_engine()
        state = self._discover_network(engine, trips)

        if state_patch:
            state_patch(state)

        # Build smoothing adjacency once
        self.smoother.build_adjacency(state.edge_ids, state.length_m)

        prev_density = np.zeros(state.n_edges, dtype=np.float64)
        prev_volume = np.zeros(state.n_edges, dtype=np.float64)

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

            # 3. Blending: MSA (fixed 1/n) or Frank-Wolfe (optimal step)
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

            # Cap density at jam density — a link cannot hold more than k_j
            state.density_vpkm = np.minimum(
                state.density_vpkm, state.jam_density
            )

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
            del engine
            engine = self._create_engine()

            # 8. Compute gap (on updated network)
            gap = self._compute_relative_gap(
                engine, trips, state, blended_volume,
            )

            iter_result = IterationResult(
                iteration=n,
                relative_gap=gap,
                tstt=tstt,
                max_density_delta=max_delta,
                n_oversaturated=n_oversat,
                route_time_s=route_time,
                customize_time_s=customize_time,
                step_size=alpha,
            )
            log.append(iter_result)

            logger.info(
                "Iter %d: gap=%.4f, TSTT=%.0f, alpha=%.4f, max_dk=%.1f, "
                "oversat=%d, route=%.1fs, customize=%.1fs",
                n, gap, tstt, alpha, max_delta, n_oversat,
                route_time, customize_time,
            )

            if progress_callback:
                progress_callback(n, gap, tstt)

            # 9. Convergence check
            if self.config.convergence_gap > 0 and gap < self.config.convergence_gap:
                logger.info("Converged at iteration %d (gap=%.4f)", n, gap)
                break

        total_time = time.monotonic() - t_start
        del engine

        # Cleanup CSV
        self.writer.cleanup()

        return AssignmentResult(
            converged=log[-1].relative_gap < self.config.convergence_gap if log else False,
            iterations=len(log),
            final_gap=log[-1].relative_gap if log else float("inf"),
            network_state=state,
            iteration_log=log,
            total_time_s=total_time,
        )
