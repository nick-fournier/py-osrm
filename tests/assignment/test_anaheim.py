"""Anaheim 416-node network validation.

Validates the density-based assignment against the Anaheim benchmark
(416 nodes, 914 links, 38 zones, 104,694 total demand).

Unlike Sioux Falls, Anaheim's TNTP data has meaningful speed and capacity
columns that map directly to road classification.  This tests the generic
``tntp_to_osm()`` pipeline without network-specific geographic overrides.

Source: bstabler/TransportationNetworks (Jeff Ban & Ray Jayakrishnan, 1992).
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
from osrm.assignment.od_matrix import DemandTrip, ODMatrixAdapter
from osrm.assignment.osm_synthesis import tntp_to_osm, patch_lanes
from osrm.assignment.tntp import parse_net, parse_trips, load_node_coords, parse_flow

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "anaheim"


def _copy_clean_osrm(base_path: str, run_dir: Path) -> str:
    """Copy clean OSRM files to a fresh directory for an isolated run."""
    src = Path(base_path).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        shutil.copy2(f, run_dir / f.name)
    return str(run_dir / Path(base_path).name)


def _prepare_anaheim_network(tmp_path: Path):
    """Synthesize, extract, partition, customize Anaheim."""
    work = tmp_path / "anaheim"
    work.mkdir(parents=True, exist_ok=True)

    net = parse_net(FIXTURE_DIR / "Anaheim_net.tntp")
    n_zones, od_matrix = parse_trips(FIXTURE_DIR / "Anaheim_trips.tntp")
    node_coords = load_node_coords(FIXTURE_DIR / "anaheim_nodes.geojson")
    ref_flows = parse_flow(FIXTURE_DIR / "Anaheim_flow.tntp")

    osm_path, meta = tntp_to_osm(
        net, node_coords, od_matrix, work / "anaheim.osm",
        ref_flows=ref_flows,
        speed_units="ft/min",
    )
    base = str(work / "anaheim.osrm")

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
                o_zone = i + 1
                d_zone = j + 1
                if o_zone in centroids and d_zone in centroids:
                    trips.append(DemandTrip(
                        origin=centroids[o_zone],
                        destination=centroids[d_zone],
                        volume=od[i, j],
                    ))
    return trips


def _run_anaheim_assignment(
    base_path: str,
    meta: dict,
    max_iter: int = 30,
    method: str = "fw",
    demand_scale: float = 0.30,
):
    """Run assignment on Anaheim network.

    Parameters
    ----------
    demand_scale : float
        Fraction of TNTP demand to use.  Default 0.30 (31,408 vph).
        Anaheim's capacities are more physical than Sioux Falls, so
        higher scaling is feasible before widespread gridlock.
    """
    meta_scaled = dict(meta)
    meta_scaled["od_matrix"] = meta["od_matrix"] * demand_scale
    trips = _build_trips(meta_scaled)

    config = AssignmentConfig(
        method=method,
        max_iterations=max_iter,
        convergence_gap=0.0,
        smoothing=DensitySmoothingConfig(method="none"),
        verbosity="ERROR",
        speed_csv_dir=str(Path(base_path).parent),
    )

    loop = AssignmentLoop(base_path, config)

    def lane_patch(state):
        patch_lanes(state, meta)

    return loop.run(trips, state_patch=lane_patch)


def _link_flow_correlation(result, meta: dict):
    """Compute Spearman rank correlation between assigned and BPR reference flows."""
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


class TestAnaheim:
    """Anaheim structural validation."""

    def test_smoke_fw(self, tmp_path):
        """Quick smoke: FW runs without error on Anaheim, 5 iterations."""
        base, meta = _prepare_anaheim_network(tmp_path)
        result = _run_anaheim_assignment(base, meta, max_iter=5, method="fw")
        assert result.iterations == 5
        assert result.network_state.n_edges > 0

    def test_flow_nonnegativity(self, tmp_path):
        """All link flows must be non-negative."""
        base, meta = _prepare_anaheim_network(tmp_path)
        result = _run_anaheim_assignment(base, meta, max_iter=10)
        assert np.all(result.network_state.flow_vph >= 0)

    def test_speeds_within_bounds(self, tmp_path):
        """Speeds must be between VDF min and freeflow."""
        base, meta = _prepare_anaheim_network(tmp_path)
        result = _run_anaheim_assignment(base, meta, max_iter=10)
        state = result.network_state
        assert np.all(state.speed_kmh >= 0.01 - 1e-6)
        assert np.all(state.speed_kmh <= state.freeflow_kmh + 1e-6)

    def test_freeflow_immutable_across_runs(self, tmp_path):
        """Freeflow must be identical whether run at 10% or 40% demand."""
        base1, meta = _prepare_anaheim_network(tmp_path / "r1")
        r1 = _run_anaheim_assignment(base1, meta, max_iter=5, demand_scale=0.40)
        s1 = r1.network_state

        base2, meta2 = _prepare_anaheim_network(tmp_path / "r2")
        r2 = _run_anaheim_assignment(base2, meta2, max_iter=5, demand_scale=0.10)
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
        assert len(common) >= 100, f"Expected ≥100 common edges, got {len(common)}"
        for key in common:
            assert abs(ff1[key] - ff2[key]) < 0.5, (
                f"Freeflow mismatch on {key}: 40%={ff1[key]:.1f}, "
                f"10%={ff2[key]:.1f}"
            )

    def test_network_scale(self, tmp_path):
        """Verify OSRM discovers a meaningful fraction of the 914 links."""
        base, meta = _prepare_anaheim_network(tmp_path)
        result = _run_anaheim_assignment(base, meta, max_iter=5)
        state = result.network_state
        # 38 zones → many OD pairs → should discover substantial network
        assert state.n_edges >= 200, (
            f"Only {state.n_edges} edges discovered from 914-link network"
        )

    def test_incremental_loading_reduces_overshoot(self, tmp_path):
        """Incremental loading should reduce max density overshoot vs direct."""
        base, meta = _prepare_anaheim_network(tmp_path / "prep")

        # Direct loading (no warm-up)
        bp_direct = _copy_clean_osrm(base, tmp_path / "direct")
        meta_full = dict(meta)
        meta_full["od_matrix"] = meta["od_matrix"] * 1.0
        trips = _build_trips(meta_full)

        cfg_direct = AssignmentConfig(
            method="fw", max_iterations=20, convergence_gap=0.0,
            smoothing=DensitySmoothingConfig(method="none"),
            verbosity="ERROR",
            speed_csv_dir=str(tmp_path / "direct"),
            incremental_steps=(1.0,),
        )
        loop_direct = AssignmentLoop(bp_direct, cfg_direct)
        r_direct = loop_direct.run(
            trips, state_patch=lambda s: patch_lanes(s, meta),
        )
        max_ratio_direct = float(np.max(
            r_direct.network_state.density_vpkm
            / r_direct.network_state.jam_density
        ))

        # Incremental loading (default 4-step warm-up)
        bp_inc = _copy_clean_osrm(base, tmp_path / "inc")
        cfg_inc = AssignmentConfig(
            method="fw", max_iterations=20, convergence_gap=0.0,
            smoothing=DensitySmoothingConfig(method="none"),
            verbosity="ERROR",
            speed_csv_dir=str(tmp_path / "inc"),
        )
        loop_inc = AssignmentLoop(bp_inc, cfg_inc)
        r_inc = loop_inc.run(
            trips, state_patch=lambda s: patch_lanes(s, meta),
        )
        max_ratio_inc = float(np.max(
            r_inc.network_state.density_vpkm
            / r_inc.network_state.jam_density
        ))

        assert max_ratio_inc < max_ratio_direct, (
            f"Incremental ({max_ratio_inc:.2f}) should have lower max k/kj "
            f"than direct ({max_ratio_direct:.2f})"
        )


def generate_anaheim_report(
    tmp_path: str | Path,
    output_path: str = "docs/plots/anaheim_validation.html",
    max_iter: int = 50,
) -> Path:
    """Generate Anaheim validation report using the generic reporter.

    Call directly::

        from tests.assignment.test_anaheim import generate_anaheim_report
        generate_anaheim_report("/tmp/work")
    """
    from osrm.assignment.plots import generate_validation_report

    return generate_validation_report(
        network_name="Anaheim",
        prepare_fn=_prepare_anaheim_network,
        run_fn=_run_anaheim_assignment,
        copy_fn=_copy_clean_osrm,
        tmp_path=Path(tmp_path),
        output_path=output_path,
        max_iter=max_iter,
        detail_scale=1.00,
        sweep_scales=[0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00],
        vc_scales=[0.10, 0.20, 0.30, 0.50],
    )
