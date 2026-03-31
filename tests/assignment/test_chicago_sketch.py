"""Chicago Sketch 933-node network benchmark.

Validates the density-based assignment on the Chicago Sketch benchmark
(933 nodes, 2950 links, 387 zones, 1,260,907 total demand).

Unlike Anaheim, Chicago Sketch has speed=0 for all links — freeflow speed
is derived from haversine distance / free_flow_time.  774 centroid
connectors (link_type 3) have fft=0 and capacity=49500; these are
classified as high-speed links so they do not constrain routing.

Node coordinates are in Illinois State Plane East (EPSG:3435, US Survey
Feet) and were pre-converted to WGS84 in the fixture GeoJSON.

Source: bstabler/TransportationNetworks (Hillel Bar-Gera, 1999).
"""

import shutil
from pathlib import Path

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

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "chicago_sketch"


def _chicago_classify_override(link, dist_m, default_cls):
    """Override classification for centroid connectors and speed outliers."""
    # Centroid connectors: link_type 3, fft=0, cap=49500
    if link.link_type == 3:
        return LinkClass(
            highway="motorway_link",
            n_lanes=max(2, default_cls.n_lanes),
            speed_kmh=100.0,
        )
    # Cap unrealistic derived speeds (noisy length/fft ratios)
    if default_cls.speed_kmh > 130:
        return LinkClass(
            highway=default_cls.highway,
            n_lanes=default_cls.n_lanes,
            speed_kmh=130.0,
        )
    return None


def _copy_clean_osrm(base_path: str, run_dir: Path) -> str:
    """Copy clean OSRM files to a fresh directory for an isolated run."""
    src = Path(base_path).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        shutil.copy2(f, run_dir / f.name)
    return str(run_dir / Path(base_path).name)


def _prepare_chicago_network(tmp_path: Path):
    """Synthesize, extract, partition, customize Chicago Sketch."""
    work = tmp_path / "chicago"
    work.mkdir(parents=True, exist_ok=True)

    net = parse_net(FIXTURE_DIR / "ChicagoSketch_net.tntp")
    n_zones, od_matrix = parse_trips(FIXTURE_DIR / "ChicagoSketch_trips.tntp")
    node_coords = load_node_coords(FIXTURE_DIR / "chicago_sketch_nodes.geojson")
    ref_flows = parse_flow(FIXTURE_DIR / "ChicagoSketch_flow.tntp")

    osm_path, meta = tntp_to_osm(
        net, node_coords, od_matrix, work / "chicago.osm",
        ref_flows=ref_flows,
        speed_units="auto",
        classify_override=_chicago_classify_override,
    )
    base = str(work / "chicago.osrm")

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


def _build_hillclimber_trips(meta: dict, demand_scale: float) -> list[DemandTrip]:
    meta_scaled = dict(meta)
    meta_scaled["od_matrix"] = meta["od_matrix"] * demand_scale
    return _build_trips(meta_scaled)


def _run_chicago_assignment(
    base_path: str,
    meta: dict,
    max_iter: int = 30,
    method: str = "fw",
    demand_scale: float = 1.00,
):
    """Run assignment on Chicago Sketch network.

    Parameters
    ----------
    demand_scale : float
        Fraction of TNTP demand to use.  Default 1.00 (1,260,907 vph).
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
        patch_lanes(state, meta)

    return loop.run(trips, state_patch=lane_patch)




def generate_chicago_report(
    tmp_path: str | Path,
    output_path: str = "docs/plots/chicago_sketch_matrix_validation.html",
    max_iter: int = 50,
) -> Path:
    """Generate Chicago Sketch validation report.

    Call directly::

        from tests.assignment.test_chicago_sketch import generate_chicago_report
        generate_chicago_report("/tmp/work")
    """
    from osrm.assignment.plots import generate_validation_report

    return generate_validation_report(
        network_name="Chicago Sketch",
        prepare_fn=_prepare_chicago_network,
        run_fn=_run_chicago_assignment,
        copy_fn=_copy_clean_osrm,
        tmp_path=Path(tmp_path),
        output_path=output_path,
        max_iter=max_iter,
        detail_scale=1.00,
        sweep_scales=[1.00],
        vc_scales=[],
        methods=["fw"],
    )


def generate_chicago_hillclimber_report(
    tmp_path: str | Path,
    output_path: str = "docs/plots/chicago_sketch_hillclimber_validation.html",
) -> Path:
    """Generate Chicago Sketch hill-climber validation report at full demand."""
    return generate_hillclimber_validation_report(
        network_name="Chicago Sketch",
        prepare_fn=_prepare_chicago_network,
        copy_fn=_copy_clean_osrm,
        trip_builder=_build_hillclimber_trips,
        tmp_path=tmp_path,
        output_path=output_path,
        detail_scale=1.00,
        n_slices=4,
        bin_width_s=3600.0,
        state_patch_factory=lambda meta: lambda state: patch_lanes(state, meta),
        intro_html=(
            "<p>Chicago Sketch hill-climber validation intentionally keeps "
            "<b>100% demand</b>. This is a full-load scalability and behavior "
            "check, not a reduced-demand proxy.</p>"
        ),
    )
