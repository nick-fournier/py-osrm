"""Chicago Regional 12,982-node network scaling benchmark.

Validates the density-based assignment on the Chicago Regional benchmark
(12,982 nodes, 39,018 links, 1,790 zones, 1,360,428 total demand).

This is ~13× larger than Chicago Sketch and serves as the primary scaling
test for the traffic assignment module.  Tests are marked ``@pytest.mark.slow``
and skipped in CI by default.

Node coordinates are in Illinois State Plane East (EPSG:3435, US Survey
Feet) and were pre-converted to WGS84 in the fixture GeoJSON.

OD demand is stored as a compressed numpy archive (.npz) to keep the
fixture under 4 MB (the raw TNTP trips file is 64 MB).

Source: bstabler/TransportationNetworks — Chicago Area Transportation Study.
"""

import logging
import shutil
import time
from pathlib import Path

import pytest

import osrm
from osrm.assignment import (
    AssignmentConfig,
    AssignmentLoop,
    DensitySmoothingConfig,
)
from osrm.assignment.od_matrix import DemandTrip
from osrm.assignment.osm_synthesis import LinkClass, tntp_to_osm, patch_lanes
from osrm.assignment.tntp import parse_net, parse_trips, load_node_coords, parse_flow
from .hillclimber_validation import generate_hillclimber_validation_report

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "chicago_regional"
logger = logging.getLogger(__name__)

# ── classification overrides ──────────────────────────────────────────

def _regional_classify_override(link, dist_m, default_cls):
    """Override classification for centroid connectors and speed outliers.

    Same heuristics as Chicago Sketch — centroid connectors (link_type 3)
    become high-speed motorway links, and speeds are capped at 130 km/h.
    """
    if link.link_type == 3:
        return LinkClass(
            highway="motorway_link",
            n_lanes=max(2, default_cls.n_lanes),
            speed_kmh=100.0,
        )
    if default_cls.speed_kmh > 130:
        return LinkClass(
            highway=default_cls.highway,
            n_lanes=default_cls.n_lanes,
            speed_kmh=130.0,
        )
    return None


# ── network preparation ──────────────────────────────────────────────

def _copy_clean_osrm(base_path: str, run_dir: Path) -> str:
    """Copy clean OSRM files to a fresh directory for an isolated run."""
    src = Path(base_path).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        shutil.copy2(f, run_dir / f.name)
    return str(run_dir / Path(base_path).name)


def _prepare_regional_network(tmp_path: Path):
    """Synthesize, extract, partition, customize Chicago Regional."""
    work = tmp_path / "chi_regional"
    work.mkdir(parents=True, exist_ok=True)

    logger.info("Parsing TNTP network and demand...")
    t0 = time.monotonic()
    net = parse_net(FIXTURE_DIR / "ChicagoRegional_net.tntp")
    n_zones, od_matrix = parse_trips(FIXTURE_DIR / "ChicagoRegional_trips.npz")
    node_coords = load_node_coords(
        FIXTURE_DIR / "chicago_regional_nodes.geojson"
    )
    ref_flows = parse_flow(FIXTURE_DIR / "ChicagoRegional_flow.tntp")
    logger.info(
        "Parsed in %.1fs: %d links, %d zones, %.0f total demand",
        time.monotonic() - t0, len(net.links), n_zones, od_matrix.sum(),
    )

    logger.info("Synthesizing OSM XML...")
    t0 = time.monotonic()
    osm_path, meta = tntp_to_osm(
        net, node_coords, od_matrix, work / "chi_regional.osm",
        ref_flows=ref_flows,
        speed_units="auto",
        classify_override=_regional_classify_override,
    )
    logger.info("OSM synthesis in %.1fs", time.monotonic() - t0)
    base = str(work / "chi_regional.osrm")

    logger.info("Extracting...")
    t0 = time.monotonic()
    osrm.extract(
        str(osm_path), profile="car", output_path=base, verbosity="ERROR"
    )
    logger.info("Extracted in %.1fs", time.monotonic() - t0)

    logger.info("Partitioning...")
    t0 = time.monotonic()
    osrm.partition(base, verbosity="ERROR")
    logger.info("Partitioned in %.1fs", time.monotonic() - t0)

    logger.info("Customizing...")
    t0 = time.monotonic()
    osrm.customize(base, verbosity="ERROR")
    logger.info("Customized in %.1fs", time.monotonic() - t0)

    return base, meta


# ── trip builders ─────────────────────────────────────────────────────

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


def _build_hillclimber_trips(
    meta: dict, demand_scale: float
) -> list[DemandTrip]:
    meta_scaled = dict(meta)
    meta_scaled["od_matrix"] = meta["od_matrix"] * demand_scale
    return _build_trips(meta_scaled)


# ── assignment runners ────────────────────────────────────────────────

def _run_regional_assignment(
    base_path: str,
    meta: dict,
    max_iter: int = 30,
    method: str = "fw",
    demand_scale: float = 1.00,
):
    """Run assignment on Chicago Regional network."""
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
        patch_lanes(state, meta)

    return loop.run(trips, state_patch=lane_patch)


# ── report generators ────────────────────────────────────────────────

def generate_regional_report(
    tmp_path: str | Path,
    output_path: str = "plots/chicago_regional_validation.html",
    max_iter: int = 50,
    max_rounds: int = 20,
    method: str = "msa",
) -> Path:
    """Generate Chicago Regional validation report.

    Parameters
    ----------
    method : str
        ``"msa"`` (default) uses greedy loading + MSA convergence.
        ``"fw"`` uses Frank-Wolfe via AssignmentLoop.
    max_iter : int
        Max iterations for FW convergence.
    max_rounds : int
        Max iterations for MSA convergence.
    """
    if method == "fw":
        from osrm.assignment.plots import generate_validation_report

        return generate_validation_report(
            network_name="Chicago Regional",
            prepare_fn=_prepare_regional_network,
            run_fn=_run_regional_assignment,
            copy_fn=_copy_clean_osrm,
            tmp_path=Path(tmp_path),
            output_path=output_path,
            max_iter=max_iter,
            detail_scale=1.00,
            sweep_scales=[1.00],
            vc_scales=[],
            methods=["fw"],
        )
    return generate_hillclimber_validation_report(
        network_name="Chicago Regional",
        prepare_fn=_prepare_regional_network,
        copy_fn=_copy_clean_osrm,
        trip_builder=_build_hillclimber_trips,
        tmp_path=tmp_path,
        output_path=output_path,
        detail_scale=1.00,
        bin_width_s=3600.0,
        state_patch_factory=lambda meta: lambda state: patch_lanes(state, meta),
        sample_rate=0.10,
        max_rounds=max_rounds,
        intro_html=(
            "<p>Chicago Regional MSA validation at "
            "<b>100% demand</b> (1,360,428 vph across 1,790 zones, "
            "39,018 links). Primary scaling benchmark — 13× larger than "
            "Chicago Sketch.</p>"
        ),
    )


# Backward-compat alias
def generate_regional_hillclimber_report(
    tmp_path: str | Path,
    output_path: str = "plots/chicago_regional_validation.html",
) -> Path:
    """Backward-compatible wrapper — delegates to unified report."""
    return generate_regional_report(tmp_path, output_path=output_path, method="msa")


# ── pytest entry points ──────────────────────────────────────────────

@pytest.mark.slow
def test_chicago_regional_matrix(tmp_path):
    """Matrix-based FW assignment on Chicago Regional (slow)."""
    base, meta = _prepare_regional_network(tmp_path)
    result = _run_regional_assignment(base, meta, max_iter=10)
    assert result.n_iterations >= 1
    assert result.total_system_travel_time > 0


@pytest.mark.slow
def test_chicago_regional_hillclimber(tmp_path):
    """Hill-climber assignment on Chicago Regional (slow)."""
    base, meta = _prepare_regional_network(tmp_path)

    from osrm.assignment.solvers import MatrixFreeHillClimber, TripStreamAdapter
    from .hillclimber_validation import (
        compute_volume_threshold,
        slice_trips_by_departure,
        _median_link_speed_from_meta,
    )

    trips = _build_hillclimber_trips(meta, demand_scale=1.0)
    load_steps = 10
    median_speed = _median_link_speed_from_meta(meta) or 30.0
    vol_threshold = compute_volume_threshold(
        median_speed_kmh=median_speed, n_slices=load_steps,
    )
    sliced = slice_trips_by_departure(
        trips, n_slices=load_steps, bin_width_s=3600.0,
        volume_threshold=vol_threshold,
    )

    config = AssignmentConfig(
        bin_width_s=3600.0,
        smoothing=DensitySmoothingConfig(method="none"),
        speed_csv_dir=str(Path(base).parent),
    )
    solver = MatrixFreeHillClimber(base, config, default_batch_size=len(sliced))

    def lane_patch(state):
        patch_lanes(state, meta)

    result = solver.run_stream(
        sliced,
        state_patch=lane_patch,
        sample_rate=0.10,
        max_rounds=3,
        gap_threshold=0.05,
    )
    assert result.n_trips > 0
    assert len(result.batch_results) > 0
