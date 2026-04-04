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


def _build_trips_multiperiod(meta: dict, n_periods: int = 2,
                              period_duration_s: float = 1800.0) -> list:
    """Build trips with even demand across multiple periods.

    Each period gets the full OD matrix (same demand rate).  Trips in
    period *k* have ``departure_time_s = k * period_duration_s``.
    """
    centroids = meta["zone_centroids"]
    od = meta["od_matrix"]
    trips = []
    for k in range(n_periods):
        dep = k * period_duration_s
        for i in range(od.shape[0]):
            for j in range(od.shape[1]):
                if od[i, j] > 0 and i != j:
                    o_zone, d_zone = i + 1, j + 1
                    if o_zone in centroids and d_zone in centroids:
                        trips.append(DemandTrip(
                            origin=centroids[o_zone],
                            destination=centroids[d_zone],
                            volume=od[i, j],
                            departure_time_s=dep,
                        ))
    return trips


def generate_chicago_spillover_report(
    tmp_path: str | Path,
    output_path: str = "plots/chicago_sketch_spillover.html",
    batch_size: int | None = None,
) -> Path:
    """Generate Chicago Sketch multi-period spillover validation report.

    Runs 2 × 30-min periods with equal demand.  Unserved demand from
    period 1 carries forward as starting flow in period 2.  Compares
    against a single-period baseline to show the effect of queue
    carryforward.
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from osrm.assignment.plots import (
        _add_congestion_map_section,
        _add_mfd_section,
        _write_combined_report,
    )

    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    base, meta = _prepare_chicago_network(tmp_path)

    period_s = 1800.0  # 30 min

    # ── Baseline: single-period (no spillover) ────────────────────
    baseline_base = _copy_clean_osrm(base, tmp_path / "baseline_run")
    baseline_trips = _build_trips(meta)
    total_demand_1p = sum(t.volume for t in baseline_trips)

    config = AssignmentConfig(
        smoothing=DensitySmoothingConfig(method="none"),
        speed_csv_dir=str(Path(baseline_base).parent),
    )
    solver_bl = AssignmentSolver(baseline_base, config)
    result_bl = solver_bl.assign_stream(
        baseline_trips, batch_size=batch_size,
        state_patch=lambda s: patch_lanes(s, meta),
    )

    # ── Multi-period: 2 × 30-min with spillover ──────────────────
    spill_base = _copy_clean_osrm(base, tmp_path / "spillover_run")
    spill_trips = _build_trips_multiperiod(meta, n_periods=2,
                                            period_duration_s=period_s)
    total_demand_2p = sum(t.volume for t in spill_trips)

    config_sp = AssignmentConfig(
        smoothing=DensitySmoothingConfig(method="none"),
        speed_csv_dir=str(Path(spill_base).parent),
    )
    solver_sp = AssignmentSolver(spill_base, config_sp)
    result_sp = solver_sp.assign_stream(
        spill_trips, batch_size=batch_size,
        period_duration_s=period_s,
        state_patch=lambda s: patch_lanes(s, meta),
    )

    # ── Build report ──────────────────────────────────────────────
    node_coords = meta["nodes"]
    link_attrs = meta["link_attrs"]
    figs: list = []
    descriptions: list[str] = []

    # §1 — Congestion map: spillover final state
    _add_congestion_map_section(
        figs, descriptions, "Chicago Sketch — Period 2 (with spillover)",
        node_coords, result_sp.network_state, link_attrs, meta, 1.0,
    )

    # §2 — Loading progression comparison
    blog_bl = result_bl.log_as_dict()
    blog_sp = result_sp.log_as_dict()

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "Queue (veh/hr/lane)", "Mean Speed (km/h)",
        "Oversaturated Links", "TSTT",
    ], vertical_spacing=0.12, horizontal_spacing=0.10)

    # Baseline traces
    bl_x = list(range(len(blog_bl["batch"])))
    sp_x = list(range(len(blog_sp["batch"])))

    fig.add_trace(go.Scatter(
        x=bl_x, y=blog_bl["queue_vehicles"], name="1-period",
        line=dict(color="#90CAF9", width=2, dash="dash"),
        legendgroup="baseline", showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sp_x, y=blog_sp["queue_vehicles"], name="2-period spillover",
        line=dict(color="#D32F2F", width=2.5),
        legendgroup="spillover", showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=bl_x, y=blog_bl["mean_speed_kmh"], name="1-period",
        line=dict(color="#90CAF9", width=2, dash="dash"),
        legendgroup="baseline", showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=sp_x, y=blog_sp["mean_speed_kmh"], name="2-period spillover",
        line=dict(color="#2E7D32", width=2.5),
        legendgroup="spillover", showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=bl_x, y=blog_bl["n_oversaturated"], name="1-period",
        line=dict(color="#90CAF9", width=2, dash="dash"),
        legendgroup="baseline", showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=sp_x, y=blog_sp["n_oversaturated"], name="2-period spillover",
        line=dict(color="#FF6F00", width=2.5),
        legendgroup="spillover", showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=bl_x, y=blog_bl["tstt"], name="1-period",
        line=dict(color="#90CAF9", width=2, dash="dash"),
        legendgroup="baseline", showlegend=False,
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=sp_x, y=blog_sp["tstt"], name="2-period spillover",
        line=dict(color="#6A1B9A", width=2.5),
        legendgroup="spillover", showlegend=False,
    ), row=2, col=2)

    # Mark period boundary in spillover traces
    n_bl = len(blog_bl["batch"])
    if n_bl < len(blog_sp["batch"]):
        for row in range(1, 3):
            for col in range(1, 3):
                fig.add_vline(
                    x=n_bl - 0.5, line_dash="dot", line_color="grey",
                    annotation_text="Period 2", annotation_position="top",
                    row=row, col=col,
                )

    fig.update_layout(
        title="Loading Progression: 1-Period vs 2-Period Spillover",
        height=600, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    figs.append(fig)
    descriptions.append(
        "<h2>Loading Progression Comparison</h2>"
        "<p>Dashed blue = single-period baseline. Solid = 2-period with "
        "queue carryforward. Vertical dotted line marks the start of period 2. "
        "Spillover from period 1 adds to period 2's congestion.</p>"
    )

    # §3 — Per-link flow comparison: baseline vs spillover final state
    s_bl = result_bl.network_state
    s_sp = result_sp.network_state

    # Match links by edge_id
    bl_edges = {
        (int(s_bl.edge_ids[i, 0]), int(s_bl.edge_ids[i, 1])): i
        for i in range(s_bl.n_edges)
    }
    sp_edges = {
        (int(s_sp.edge_ids[i, 0]), int(s_sp.edge_ids[i, 1])): i
        for i in range(s_sp.n_edges)
    }
    common = set(bl_edges) & set(sp_edges)

    bl_flow = np.array([s_bl.flow_vph[bl_edges[k]] for k in common])
    sp_flow = np.array([s_sp.flow_vph[sp_edges[k]] for k in common])
    bl_speed = np.array([s_bl.speed_kmh[bl_edges[k]] for k in common])
    sp_speed = np.array([s_sp.speed_kmh[sp_edges[k]] for k in common])

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        "Flow: Baseline vs Spillover", "Speed: Baseline vs Spillover",
    ])
    fig.add_trace(go.Scattergl(
        x=bl_flow, y=sp_flow, mode="markers",
        marker=dict(size=3, color="#1565C0", opacity=0.5),
        name="Links",
    ), row=1, col=1)
    max_flow = max(float(np.max(bl_flow)), float(np.max(sp_flow))) * 1.05
    fig.add_trace(go.Scatter(
        x=[0, max_flow], y=[0, max_flow],
        mode="lines", line=dict(color="grey", dash="dash", width=1),
        name="y=x", showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scattergl(
        x=bl_speed, y=sp_speed, mode="markers",
        marker=dict(size=3, color="#2E7D32", opacity=0.5),
        name="Links", showlegend=False,
    ), row=1, col=2)
    max_spd = max(float(np.max(bl_speed)), float(np.max(sp_speed))) * 1.05
    fig.add_trace(go.Scatter(
        x=[0, max_spd], y=[0, max_spd],
        mode="lines", line=dict(color="grey", dash="dash", width=1),
        name="y=x", showlegend=False,
    ), row=1, col=2)

    fig.update_xaxes(title_text="Baseline (1-period)", row=1, col=1)
    fig.update_yaxes(title_text="Spillover (2-period)", row=1, col=1)
    fig.update_xaxes(title_text="Baseline speed (km/h)", row=1, col=2)
    fig.update_yaxes(title_text="Spillover speed (km/h)", row=1, col=2)
    fig.update_layout(
        title="Per-Link State: Baseline vs Spillover",
        height=450, template="plotly_white",
    )
    figs.append(fig)

    # Summary stats
    bl_unserved = float(np.sum(s_bl.unserved_demand))
    sp_unserved = float(np.sum(result_sp.unserved_vph))
    spill_pct = sp_unserved / total_demand_1p * 100 if total_demand_1p > 0 else 0

    descriptions.append(
        "<h2>Per-Link Comparison</h2>"
        f"<p>{len(common)} common links compared. Points above y=x line "
        "have higher flow/lower speed under spillover.</p>"
        f"<p><b>Baseline unserved:</b> {bl_unserved:,.0f} veh/hr | "
        f"<b>Spillover final unserved:</b> {sp_unserved:,.0f} veh/hr "
        f"({spill_pct:.1f}% of per-period demand)</p>"
    )

    # §4 — MFD for spillover final state
    _add_mfd_section(figs, descriptions, result_sp.network_state, 1.0)

    # §5 — Summary table
    figs.append(go.Figure())  # placeholder
    descriptions.append(
        "<h2>Summary</h2>"
        "<table style='border-collapse:collapse; margin:1em 0;'>"
        "<tr style='border-bottom:2px solid #333;'>"
        "<th style='padding:6px 16px; text-align:left;'>Metric</th>"
        "<th style='padding:6px 16px; text-align:right;'>Baseline (1-period)</th>"
        "<th style='padding:6px 16px; text-align:right;'>Spillover (2-period)</th></tr>"
        f"<tr><td style='padding:4px 16px;'>Total demand</td>"
        f"<td style='padding:4px 16px; text-align:right;'>{total_demand_1p:,.0f} vph</td>"
        f"<td style='padding:4px 16px; text-align:right;'>{total_demand_2p:,.0f} vph (2×)</td></tr>"
        f"<tr><td style='padding:4px 16px;'>Batches</td>"
        f"<td style='padding:4px 16px; text-align:right;'>{result_bl.n_batches}</td>"
        f"<td style='padding:4px 16px; text-align:right;'>{result_sp.n_batches}</td></tr>"
        f"<tr><td style='padding:4px 16px;'>Runtime</td>"
        f"<td style='padding:4px 16px; text-align:right;'>{result_bl.total_time_s:.1f}s</td>"
        f"<td style='padding:4px 16px; text-align:right;'>{result_sp.total_time_s:.1f}s</td></tr>"
        f"<tr><td style='padding:4px 16px;'>Final queue (veh/hr/lane)</td>"
        f"<td style='padding:4px 16px; text-align:right;'>"
        f"{result_bl.batch_log[-1].queue_vehicles:.0f}</td>"
        f"<td style='padding:4px 16px; text-align:right;'>"
        f"{result_sp.batch_log[-1].queue_vehicles:.0f}</td></tr>"
        f"<tr><td style='padding:4px 16px;'>Final unserved (veh/hr)</td>"
        f"<td style='padding:4px 16px; text-align:right;'>{bl_unserved:,.0f}</td>"
        f"<td style='padding:4px 16px; text-align:right;'>{sp_unserved:,.0f}</td></tr>"
        f"<tr><td style='padding:4px 16px;'>Mean speed (km/h)</td>"
        f"<td style='padding:4px 16px; text-align:right;'>"
        f"{result_bl.batch_log[-1].mean_speed_kmh:.1f}</td>"
        f"<td style='padding:4px 16px; text-align:right;'>"
        f"{result_sp.batch_log[-1].mean_speed_kmh:.1f}</td></tr>"
        f"<tr><td style='padding:4px 16px;'>Oversaturated links</td>"
        f"<td style='padding:4px 16px; text-align:right;'>"
        f"{result_bl.batch_log[-1].n_oversaturated}</td>"
        f"<td style='padding:4px 16px; text-align:right;'>"
        f"{result_sp.batch_log[-1].n_oversaturated}</td></tr>"
        "</table>"
    )

    out = Path(output_path)
    _write_combined_report(
        title="Chicago Sketch — Multi-Period Spillover Validation",
        intro=(
            f"<p>Spillover validation on <b>Chicago Sketch</b>: "
            f"2 × 30-min periods with equal demand "
            f"({total_demand_1p:,.0f} vph each). "
            f"Unserved demand from period 1 carries forward as starting "
            f"flow in period 2. Baseline = single-period (no spillover).</p>"
        ),
        figures=figs,
        descriptions=descriptions,
        path=out,
    )
    return out

