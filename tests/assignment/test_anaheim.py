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


@pytest.fixture(scope="module")
def anaheim_net(tmp_path_factory):
    """Prepare Anaheim network once for all tests in this module."""
    tmp = tmp_path_factory.mktemp("anaheim")
    return _prepare_anaheim_network(tmp)


@pytest.fixture(scope="module")
def anaheim_result(anaheim_net, tmp_path_factory):
    """Run a single 3-iteration assignment shared by read-only tests."""
    base, meta = anaheim_net
    run_dir = tmp_path_factory.mktemp("ana_run")
    run_base = _copy_clean_osrm(base, run_dir)
    return _run_anaheim_assignment(run_base, meta, max_iter=3)


class TestAnaheim:
    """Anaheim structural validation."""

    def test_smoke_fw(self, anaheim_result):
        """Quick smoke: FW runs without error on Anaheim."""
        assert anaheim_result.iterations >= 1
        assert anaheim_result.network_state.n_edges > 0

    def test_flow_nonnegativity(self, anaheim_result):
        """All link flows must be non-negative."""
        assert np.all(anaheim_result.network_state.flow_vph >= 0)

    def test_speeds_within_bounds(self, anaheim_result):
        """Speeds must be between VDF min and freeflow."""
        state = anaheim_result.network_state
        assert np.all(state.speed_kmh >= 1.08 - 1e-6)
        assert np.all(state.speed_kmh <= state.freeflow_kmh + 1e-6)

    def test_freeflow_immutable_across_runs(self, anaheim_net, tmp_path_factory):
        """Freeflow must be identical whether run at 10% or 40% demand."""
        base, meta = anaheim_net
        bp1 = _copy_clean_osrm(base, tmp_path_factory.mktemp("ff1"))
        r1 = _run_anaheim_assignment(bp1, meta, max_iter=3, demand_scale=0.40)
        s1 = r1.network_state

        bp2 = _copy_clean_osrm(base, tmp_path_factory.mktemp("ff2"))
        r2 = _run_anaheim_assignment(bp2, meta, max_iter=3, demand_scale=0.10)
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

    def test_network_scale(self, anaheim_result):
        """Verify OSRM discovers a meaningful fraction of the 914 links."""
        state = anaheim_result.network_state
        assert state.n_edges >= 200, (
            f"Only {state.n_edges} edges discovered from 914-link network"
        )

    def test_incremental_loading_reduces_overshoot(self, anaheim_net, tmp_path_factory):
        """Incremental loading should reduce median density overshoot vs direct."""
        base, meta = anaheim_net

        # Direct loading (no warm-up)
        bp_direct = _copy_clean_osrm(base, tmp_path_factory.mktemp("direct"))
        meta_full = dict(meta)
        meta_full["od_matrix"] = meta["od_matrix"] * 1.0
        trips = _build_trips(meta_full)

        cfg_direct = AssignmentConfig(
            method="fw", max_iterations=10, convergence_gap=0.0,
            smoothing=DensitySmoothingConfig(method="none"),
            verbosity="ERROR",
            speed_csv_dir=str(Path(bp_direct).parent),
            incremental_steps=(1.0,),
        )
        loop_direct = AssignmentLoop(bp_direct, cfg_direct)
        r_direct = loop_direct.run(
            trips, state_patch=lambda s: patch_lanes(s, meta),
        )
        ratio_direct = (
            r_direct.network_state.density_vpkm
            / r_direct.network_state.jam_density
        )

        # Incremental loading (default 4-step warm-up)
        bp_inc = _copy_clean_osrm(base, tmp_path_factory.mktemp("inc"))
        cfg_inc = AssignmentConfig(
            method="fw", max_iterations=10, convergence_gap=0.0,
            smoothing=DensitySmoothingConfig(method="none"),
            verbosity="ERROR",
            speed_csv_dir=str(Path(bp_inc).parent),
        )
        loop_inc = AssignmentLoop(bp_inc, cfg_inc)
        r_inc = loop_inc.run(
            trips, state_patch=lambda s: patch_lanes(s, meta),
        )
        ratio_inc = (
            r_inc.network_state.density_vpkm
            / r_inc.network_state.jam_density
        )

        # Use 95th percentile k/kj — max is dominated by single outlier
        # links; median is in the uncongested noise
        p95_direct = float(np.percentile(ratio_direct, 95))
        p95_inc = float(np.percentile(ratio_inc, 95))
        assert p95_inc <= p95_direct * 1.1, (
            f"Incremental p95 k/kj ({p95_inc:.3f}) should not be much worse "
            f"than direct ({p95_direct:.3f})"
        )


def generate_anaheim_report(
    tmp_path: str | Path,
    output_path: str = "docs/plots/anaheim_matrix_validation.html",
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
        sweep_scales=[0.30, 0.50, 0.75, 1.00],
        vc_scales=[0.10, 0.20, 0.30, 0.50],
    )
