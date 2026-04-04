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
from osrm.assignment import AssignmentConfig, AssignmentSolver, DensitySmoothingConfig
from osrm.assignment.od_matrix import DemandTrip
from osrm.assignment.osm_synthesis import LinkClass, tntp_to_osm, patch_lanes
from osrm.assignment.tntp import parse_net, parse_trips, load_node_coords, parse_flow
from .validation import generate_validation_report

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


def _build_validation_trips(meta: dict, demand_scale: float) -> list[DemandTrip]:
    meta_scaled = dict(meta)
    meta_scaled["od_matrix"] = meta["od_matrix"] * demand_scale
    return _build_trips(meta_scaled)





def generate_chicago_report(
    tmp_path: str | Path,
    output_path: str = "plots/chicago_sketch_validation.html",
    max_rounds: int = 20,
    method: str = "msa",
) -> Path:
    """Generate Chicago Sketch validation report.

    Parameters
    ----------
    method : str
        ``"msa"`` (default) or ``"fw"`` — convergence method after greedy loading.
    max_rounds : int
        Max convergence iterations.
    """
    return generate_validation_report(
        network_name="Chicago Sketch",
        prepare_fn=_prepare_chicago_network,
        copy_fn=_copy_clean_osrm,
        trip_builder=_build_validation_trips,
        tmp_path=tmp_path,
        output_path=output_path,
        detail_scale=1.00,
        state_patch_factory=lambda meta: lambda state: patch_lanes(state, meta),
        max_rounds=max_rounds,
        method=method,
        intro_html=(
            f"<p>{method.upper()} validation at <b>100% Chicago Sketch demand</b>. "
            f"Full-load scalability and convergence check.</p>"
        ),
    )


def generate_chicago_stream_report(
    tmp_path: str | Path,
    output_path: str = "plots/chicago_sketch_stream.html",
    batch_size: int | None = None,
) -> Path:
    """Generate Chicago Sketch streaming assignment report."""
    import plotly.graph_objects as go
    from osrm.assignment.plots import (
        _add_congestion_map_section,
        _add_correlation_section,
        _add_mfd_section,
        _write_combined_report,
    )

    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    base, meta = _prepare_chicago_network(tmp_path)
    run_base = _copy_clean_osrm(base, tmp_path / "stream_run")

    trips = _build_trips(meta)
    total_demand = sum(t.volume for t in trips)

    config = AssignmentConfig(
        smoothing=DensitySmoothingConfig(method="none"),
        speed_csv_dir=str(Path(run_base).parent),
    )
    solver = AssignmentSolver(run_base, config)

    def lane_patch(state):
        patch_lanes(state, meta)

    result = solver.assign_stream(trips, batch_size=batch_size, state_patch=lane_patch)
    state = result.network_state

    node_coords = meta["nodes"]
    link_attrs = meta["link_attrs"]
    ref = meta.get("ref_flows", {})

    figs: list = []
    descriptions: list[str] = []

    _add_congestion_map_section(
        figs, descriptions, "Chicago Sketch (Stream)",
        node_coords, state, link_attrs, meta, 1.0,
    )

    blog = result.log_as_dict()
    labels = [f"Batch {b}" for b in blog["batch"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=blog["queue_vehicles"], name="Queue (veh/hr/lane)",
        mode="lines+markers",
        line=dict(color="#D32F2F", width=2.5),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=blog["mean_speed_kmh"], name="Mean speed (km/h)",
        mode="lines+markers",
        line=dict(color="#2E7D32", width=2.5),
        marker=dict(size=5), yaxis="y2",
    ))
    fig.update_layout(
        title="Stream Loading Progression",
        xaxis_title="Loading Batch",
        yaxis=dict(title="Mean queue per lane (veh/hr/lane)"),
        yaxis2=dict(title="Mean speed (km/h)", overlaying="y", side="right"),
        template="plotly_white",
    )
    figs.append(fig)
    descriptions.append(
        "<h2>Stream Loading Progression</h2>"
        f"<p>Incremental greedy loading of {total_demand:,.0f} vph "
        f"across {result.n_batches} batches.</p>"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=blog["route_time_s"], name="Route time",
        mode="lines+markers",
        line=dict(color="#1565C0", width=2.0),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=blog["customize_time_s"], name="Customize time",
        mode="lines+markers",
        line=dict(color="#8E24AA", width=2.0),
        marker=dict(size=5),
    ))
    fig.update_layout(
        title="Runtime per Batch",
        xaxis_title="Loading Batch",
        yaxis_title="Time (s)",
        template="plotly_white",
    )
    figs.append(fig)
    descriptions.append(
        "<h2>Runtime</h2>"
        "<p>Per-batch routing and customize timings.</p>"
    )

    _add_mfd_section(figs, descriptions, state, 1.0)

    if ref:
        _add_correlation_section(
            figs, descriptions, state, ref, link_attrs, 1.0,
        )

    out = Path(output_path)
    _write_combined_report(
        title="Chicago Sketch — Stream Assignment",
        intro=(
            f"<p>Streaming assignment on <b>Chicago Sketch</b>: "
            f"{len(trips):,} OD pairs, {total_demand:,.0f} vph total demand, "
            f"{result.n_batches} loading batches. "
            f"Total runtime: {result.total_time_s:.1f}s.</p>"
        ),
        figures=figs,
        descriptions=descriptions,
        path=out,
    )
    return out

