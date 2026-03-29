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
    """Configuration for the assignment loop."""

    max_iterations: int = 50
    convergence_gap: float = 0.01
    bin_width_s: float = 3600.0
    min_speed_kmh: float = 5.0
    smoothing: DensitySmoothingConfig = field(
        default_factory=DensitySmoothingConfig
    )
    vdf_kc_ratio: float = 1.0 / 3.0
    default_jam_density_per_lane: float = 130.0
    default_n_lanes: int = 1
    speed_csv_dir: Optional[str] = None
    verbosity: str = "ERROR"


@dataclass
class IterationResult:
    """Metrics for a single assignment iteration."""

    iteration: int
    relative_gap: float
    tstt: float
    max_flow_delta: float
    n_oversaturated: int
    route_time_s: float
    customize_time_s: float


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
            "max_flow_delta": [r.max_flow_delta for r in self.iteration_log],
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
            min_speed_kmh=self.config.min_speed_kmh,
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
    ) -> Optional[dict]:
        """Route a single OD pair with annotations."""
        try:
            result = engine.Route(
                coordinates=[trip.origin, trip.destination],
                annotations=["nodes", "distance", "duration", "speed"],
            )
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
        """Route all trips once to discover network edges.

        Uses freeflow routing (before any congestion) to build the
        edge registry from route annotations.
        """
        logger.info("Discovering network edges from %d OD pairs...", len(trips))
        route_results = []
        for trip in trips:
            result = self._route_single(engine, trip)
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
    ) -> Tuple[np.ndarray, float]:
        """Route all trips, accumulate flow into a fresh array.

        Returns
        -------
        new_flow : np.ndarray
            All-or-nothing flow from this iteration.
        tstt : float
            Total system travel time (vehicle-seconds).
        """
        new_flow = np.zeros(state.n_edges, dtype=np.float64)
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
                durations = ann.get("duration", [])

                for i in range(len(nodes) - 1):
                    idx = state.edge_ordinal(int(nodes[i]), int(nodes[i + 1]))
                    if idx is not None:
                        # Convert volume (veh/period) to flow rate (veh/hr)
                        new_flow[idx] += trip.volume / bin_width_hr

        return new_flow, tstt

    def _compute_relative_gap(
        self,
        engine: osrm_module.OSRM,
        trips: List[DemandTrip],
        state: NetworkState,
    ) -> float:
        """Compute Wardrop relative gap.

        gap = Σ_a x_a · t_a / Σ_rs q_rs · π_rs - 1

        where x_a·t_a is link flow × link cost summed over all links,
        and q_rs·π_rs is demand × shortest path cost summed over all OD pairs.
        """
        # Numerator: total travel on current costs
        # t_e = length_m / (speed_kmh / 3.6) = length_m * 3.6 / speed_kmh
        link_time_s = state.length_m * 3.6 / np.maximum(state.speed_kmh, 1.0)
        bin_width_hr = self.config.bin_width_s / 3600.0
        # flow_vph * link_time_s gives veh·s/hr; multiply by bin_width_hr to get veh·s
        numerator = float(np.sum(state.flow_vph * link_time_s)) * bin_width_hr

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

    def _update_state(self, state: NetworkState) -> None:
        """Update density and speed from current flow using VDF."""
        state.density_vpkm = self.vdf.flow_to_density(
            state.flow_vph, state.freeflow_kmh, state.jam_density
        )

        # Smooth density
        smoothed = self.smoother.smooth(state.density_vpkm)

        # VDF: smoothed density → speed
        state.speed_kmh = self.vdf.density_to_speed(
            smoothed, state.freeflow_kmh, state.jam_density
        )

    def run(
        self,
        trips: List[DemandTrip],
        progress_callback=None,
    ) -> AssignmentResult:
        """Run the iterative assignment loop.

        Parameters
        ----------
        trips : list of DemandTrip
            Demand to assign.
        progress_callback : callable, optional
            Called with (iteration, gap, tstt) after each iteration.

        Returns
        -------
        AssignmentResult
        """
        t_start = time.monotonic()
        log: List[IterationResult] = []

        # 1. Initial routing on freeflow to discover network
        engine = self._create_engine()
        state = self._discover_network(engine, trips)

        # Build smoothing adjacency once
        self.smoother.build_adjacency(state.edge_ids, state.length_m)

        prev_flow = np.zeros(state.n_edges, dtype=np.float64)

        for n in range(1, self.config.max_iterations + 1):
            logger.info("=== Iteration %d ===", n)

            # 2. Route all trips, get all-or-nothing flow
            t_route = time.monotonic()
            aon_flow, tstt = self._route_and_accumulate(engine, trips, state)
            route_time = time.monotonic() - t_route

            # 3. MSA blending: q = q_old + (1/n)(q_aon - q_old)
            if n == 1:
                state.flow_vph = aon_flow.copy()
            else:
                alpha = 1.0 / n
                state.flow_vph = prev_flow + alpha * (aon_flow - prev_flow)

            max_delta = float(np.max(np.abs(state.flow_vph - prev_flow)))
            prev_flow = state.flow_vph.copy()

            # 4. Update density + speed via VDF
            self._update_state(state)

            # 5. Count oversaturated links
            q_c = self.vdf.capacity_flow(state.freeflow_kmh, state.jam_density)
            n_oversat = int(np.sum(state.flow_vph > q_c))

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
            gap = self._compute_relative_gap(engine, trips, state)

            iter_result = IterationResult(
                iteration=n,
                relative_gap=gap,
                tstt=tstt,
                max_flow_delta=max_delta,
                n_oversaturated=n_oversat,
                route_time_s=route_time,
                customize_time_s=customize_time,
            )
            log.append(iter_result)

            logger.info(
                "Iter %d: gap=%.4f, TSTT=%.0f, max_Δq=%.1f, oversat=%d, "
                "route=%.1fs, customize=%.1fs",
                n, gap, tstt, max_delta, n_oversat, route_time, customize_time,
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
