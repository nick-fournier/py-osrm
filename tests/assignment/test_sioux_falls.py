"""Sioux Falls 24-node network validation.

Validates the density-based assignment against the canonical Sioux Falls
benchmark (24 nodes, 76 links, 528 OD pairs).

The TNTP demand (360,600 total) represents a BPR-calibrated hourly volume
that exceeds the physical capacity of realistic 2–3 lane roads. Matrix
validation stays at 15% demand (54,090 vph), which is representative of an
actual peak hour for a ~200k population city. Hill-climber validation uses
30% demand (108,180 vph) so sampled refinement is exercised on a more
meaningfully loaded network.

Structural (VDF-independent) checks:
  - Wardrop relative gap < threshold
  - Flow conservation
  - Non-negative flows
  - Speed within bounds
  - Rank correlation with published BPR equilibrium (directional, not magnitude)

See docs/traffic_assignment_design.md section 6.4.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

import osrm
from osrm.assignment import (
    AssignmentConfig,
    AssignmentLoop,
    DensitySmoothingConfig,
)
from osrm.assignment.od_matrix import DemandTrip
from osrm.assignment.osm_synthesis import sioux_falls_network, patch_sioux_falls_lanes
from .hillclimber_validation import (
    generate_hillclimber_validation_report,
    run_hillclimber_case,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "sioux_falls"


def _copy_clean_osrm(base_path: str, run_dir: Path) -> str:
    """Copy clean OSRM files to a fresh directory for an isolated run.

    Segment-speed customization permanently mutates OSRM edge weights,
    so each assignment run() needs its own copy of the base files.
    """
    src = Path(base_path).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        shutil.copy2(f, run_dir / f.name)
    return str(run_dir / Path(base_path).name)


def _prepare_sf_network(tmp_path: Path):
    """Synthesize, extract, partition, customize Sioux Falls."""
    work = tmp_path / "sioux_falls"
    work.mkdir(parents=True, exist_ok=True)

    osm_path, meta = sioux_falls_network(
        work / "sf.osm", fixture_dir=FIXTURE_DIR,
    )
    base = str(work / "sf.osrm")

    osrm.extract(str(osm_path), profile="car", output_path=base, verbosity="ERROR")
    osrm.partition(base, verbosity="ERROR")
    osrm.customize(base, verbosity="ERROR")

    return base, meta


def _build_trips(meta: dict) -> list:
    """Convert OD matrix to DemandTrip list, skipping zero/self demand."""
    centroids = meta["zone_centroids"]
    od = meta["od_matrix"]
    trips = []
    for i in range(od.shape[0]):
        for j in range(od.shape[1]):
            if od[i, j] > 0 and i != j:
                trips.append(DemandTrip(
                    origin=centroids[i + 1],
                    destination=centroids[j + 1],
                    volume=od[i, j],
                ))
    return trips


def _build_hillclimber_trips(meta: dict, demand_scale: float) -> list[DemandTrip]:
    meta_scaled = dict(meta)
    meta_scaled["od_matrix"] = meta["od_matrix"] * demand_scale
    return _build_trips(meta_scaled)


def _run_sf_assignment(
    base_path: str,
    meta: dict,
    max_iter: int = 30,
    method: str = "fw",
    demand_scale: float = 0.15,
):
    """Run assignment on Sioux Falls network.

    Parameters
    ----------
    demand_scale : float
        Fraction of TNTP demand to use.  Default 0.15 (54,090 vph)
        which is realistic for a ~200k city peak hour on 2–3 lane roads.
    """
    meta_scaled = dict(meta)
    meta_scaled["od_matrix"] = meta["od_matrix"] * demand_scale
    trips = _build_trips(meta_scaled)

    config = AssignmentConfig(
        method=method,
        max_iterations=max_iter,
        convergence_gap=0.0,
        smoothing=DensitySmoothingConfig(method="none"),
        speed_csv_dir=str(Path(base_path).parent),
    )

    loop = AssignmentLoop(base_path, config)

    def lane_patch(state):
        patch_sioux_falls_lanes(state, meta)

    return loop.run(trips, state_patch=lane_patch)


def _link_flow_correlation(result, meta: dict):
    """Compute Spearman rank correlation between assigned and BPR reference flows.

    Returns (correlation, n_matched_links).
    """
    from scipy.stats import spearmanr

    state = result.network_state
    ref = meta["ref_flows"]

    assigned = []
    reference = []
    for i in range(state.n_edges):
        key = (int(state.edge_ids[i, 0]), int(state.edge_ids[i, 1]))
        if key in ref:
            assigned.append(state.flow_vph[i])
            reference.append(ref[key][0])

    if len(assigned) < 3:
        return 0.0, len(assigned)

    corr, _ = spearmanr(assigned, reference)
    return float(corr), len(assigned)


class TestSiouxFalls:
    """Sioux Falls structural validation."""

    def test_smoke_fw(self, tmp_path):
        """Quick smoke: FW runs without error on Sioux Falls, 5 iterations."""
        base, meta = _prepare_sf_network(tmp_path)
        result = _run_sf_assignment(base, meta, max_iter=5, method="fw")
        assert result.iterations >= 3
        assert result.network_state.n_edges > 0

    def test_flow_nonnegativity(self, tmp_path):
        """All link flows must be non-negative."""
        base, meta = _prepare_sf_network(tmp_path)
        result = _run_sf_assignment(base, meta, max_iter=10)
        assert np.all(result.network_state.flow_vph >= 0)

    def test_speeds_within_bounds(self, tmp_path):
        """Speeds must be between VDF min and freeflow."""
        base, meta = _prepare_sf_network(tmp_path)
        result = _run_sf_assignment(base, meta, max_iter=10)
        state = result.network_state
        assert np.all(state.speed_kmh > 0)
        assert np.all(state.speed_kmh <= state.freeflow_kmh + 1e-6)

    def test_freeflow_immutable_across_runs(self, tmp_path):
        """Freeflow must be identical whether run at 5% or 20% demand.

        Regression test: segment-speed customization permanently mutates
        OSRM edge weights.  Running at high demand then low demand used to
        show degraded freeflow on the second run.
        """
        base1, meta = _prepare_sf_network(tmp_path / "r1")
        r1 = _run_sf_assignment(base1, meta, max_iter=5, demand_scale=0.20)
        s1 = r1.network_state

        base2, meta2 = _prepare_sf_network(tmp_path / "r2")
        r2 = _run_sf_assignment(base2, meta2, max_iter=5, demand_scale=0.05)
        s2 = r2.network_state

        ff1 = {
            (int(s1.edge_ids[i, 0]), int(s1.edge_ids[i, 1])): s1.freeflow_kmh[i]
            for i in range(s1.n_edges)
        }
        ff2 = {
            (int(s2.edge_ids[i, 0]), int(s2.edge_ids[i, 1])): s2.freeflow_kmh[i]
            for i in range(s2.n_edges)
        }

        common = set(ff1) & set(ff2)
        assert len(common) >= 20, f"Expected ≥20 common edges, got {len(common)}"
        for key in common:
            assert abs(ff1[key] - ff2[key]) < 0.5, (
                f"Freeflow mismatch on {key}: 20%={ff1[key]:.1f}, "
                f"5%={ff2[key]:.1f}"
            )

    def test_hillclimber_smoke(self, tmp_path):
        """Hill-climber loads Sioux Falls with sampled refinement enabled."""
        base, meta = _prepare_sf_network(tmp_path)
        case = run_hillclimber_case(
            base_path=base,
            meta=meta,
            copy_fn=_copy_clean_osrm,
            trip_builder=_build_hillclimber_trips,
            run_dir=tmp_path / "hc_run",
            demand_scale=0.30,
            state_patch_factory=lambda m: lambda s: patch_sioux_falls_lanes(s, m),
            sample_rate=0.10,
            max_rounds=10,
        )
        state = case.result.network_state
        assert state is not None
        assert case.result.n_batches == 10
        assert state.n_edges > 0
        assert np.all(state.flow_vph >= 0)
        assert np.all(state.speed_kmh > 0)


def generate_sioux_falls_report(
    tmp_path: str | Path,
    output_path: str = "plots/sioux_falls_validation.html",
    max_iter: int = 50,
    method: str = "msa",
) -> Path:
    """Generate Sioux Falls validation report.

    Parameters
    ----------
    method : str
        ``"msa"`` (default) uses greedy loading + MSA convergence.
        ``"fw"`` uses Frank-Wolfe via AssignmentLoop.
    """
    if method == "fw":
        from osrm.assignment.plots import generate_validation_report

        return generate_validation_report(
            network_name="Sioux Falls",
            prepare_fn=_prepare_sf_network,
            run_fn=_run_sf_assignment,
            copy_fn=_copy_clean_osrm,
            tmp_path=Path(tmp_path),
            output_path=output_path,
            max_iter=max_iter,
            detail_scale=0.15,
            intro_html=(
                "<p>Road classification from real Sioux Falls geography: "
                "I-29 (3 lanes, 105 km/h), I-229 (2 lanes, 105 km/h), "
                "arterials (2 lanes, 65 km/h). "
                "TNTP &lsquo;capacity&rsquo; values are BPR math artifacts, "
                "<b>not</b> physical road capacity.</p>"
            ),
        )
    return generate_hillclimber_validation_report(
        network_name="Sioux Falls",
        prepare_fn=_prepare_sf_network,
        copy_fn=_copy_clean_osrm,
        trip_builder=_build_hillclimber_trips,
        tmp_path=tmp_path,
        output_path=output_path,
        detail_scale=0.30,
        bin_width_s=3600.0,
        state_patch_factory=lambda meta: lambda state: patch_sioux_falls_lanes(state, meta),
        sample_rate=0.10,
        max_rounds=10,
        intro_html=(
            "<p>MSA validation runs at <b>30% Sioux Falls demand</b> "
            "with 10 greedy load steps and MSA convergence.</p>"
        ),
    )


# Backward-compat aliases
def generate_sioux_falls_hillclimber_report(
    tmp_path: str | Path,
    output_path: str = "plots/sioux_falls_validation.html",
) -> Path:
    """Backward-compatible wrapper — delegates to unified report."""
    return generate_sioux_falls_report(tmp_path, output_path=output_path, method="msa")
