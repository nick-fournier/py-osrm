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


def _build_trips_multiperiod(meta: dict, n_periods: int = 4,
                              period_duration_s: float = 900.0) -> list:
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

    Runs 4 × 15-min demand periods, then continues with empty drain
    periods (no new trips, only spillover) until the queue dissipates.
    Plots end-of-period metrics with period number on X axis.
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

    n_demand_periods = 4
    period_s = 900.0  # 15 min
    max_drain_periods = 20

    # Build demand trips: 4 periods of equal demand
    spill_trips = _build_trips_multiperiod(
        meta, n_periods=n_demand_periods, period_duration_s=period_s,
    )
    per_period_demand = sum(t.volume for t in _build_trips(meta))

    # Run with spillover
    spill_base = _copy_clean_osrm(base, tmp_path / "spillover_run")
    config = AssignmentConfig(
        smoothing=DensitySmoothingConfig(method="none"),
        speed_csv_dir=str(Path(spill_base).parent),
    )
    solver = AssignmentSolver(spill_base, config)

    result = solver.assign_stream(
        spill_trips, batch_size=batch_size,
        period_duration_s=period_s,
        state_patch=lambda s: patch_lanes(s, meta),
    )

    # Extract end-of-period metrics from batch log.
    # Detect period boundaries: when total_unserved_vph drops between
    # consecutive batches, that's a period reset (carryforward).
    blog = result.log_as_dict()
    n_batches = len(blog["batch"])
    unserved_series = blog["total_unserved_vph"]

    # Detect period boundaries: when total_unserved_vph drops between
    # consecutive batches, that's a period reset (carryforward).  During
    # demand loading, unserved only grows — any drop means period reset.
    period_end_indices = []
    for i in range(1, n_batches):
        if unserved_series[i] < unserved_series[i - 1] * 0.7:
            period_end_indices.append(i - 1)
    period_end_indices.append(n_batches - 1)  # final period

    period_metrics = []
    for p, end_idx in enumerate(period_end_indices):
        period_metrics.append({
            "period": p + 1,
            "label": f"P{p+1}",
            "has_demand": True,
            "queue_veh_hr_lane": blog["queue_vehicles"][end_idx],
            "total_unserved_vph": blog["total_unserved_vph"][end_idx],
            "mean_speed_kmh": blog["mean_speed_kmh"][end_idx],
            "n_oversaturated": blog["n_oversaturated"][end_idx],
            "tstt": blog["tstt"][end_idx],
        })

    # Drain phase: no new demand, queue discharges over time.
    # Each period, links discharge at their throughput rate.
    # queue_veh = unserved_rate × period_hr  (vehicles in queue)
    # discharged = throughput × period_hr    (vehicles that leave)
    # remaining  = max(0, queue_veh - discharged)
    state = result.network_state
    unserved = result.unserved_vph.copy()

    # Save peak state before mutating for MFD plot.
    import copy
    peak_state = copy.copy(state)
    peak_state.flow_vph = state.flow_vph.copy()
    peak_state.density_vpkm = state.density_vpkm.copy()
    peak_state.speed_kmh = state.speed_kmh.copy()

    from osrm.assignment.vdf import BiParabolicVDF
    vdf = BiParabolicVDF()

    period_hr = period_s / 3600.0
    queue_veh = unserved * period_hr  # convert rate to vehicles

    for drain_p in range(max_drain_periods):
        total_queue = float(np.sum(queue_veh))
        if total_queue < 1.0:
            break

        # Compute throughput at current queue level
        queue_rate = queue_veh / period_hr  # back to veh/hr for VDF
        state.flow_vph = np.maximum(queue_rate, 0.0)
        state.density_vpkm = vdf.demand_to_density(
            state.flow_vph, state.freeflow_kmh, state.jam_density,
            kc_ratio=state.kc_ratio,
        )
        state.speed_kmh = vdf.density_to_speed(
            state.density_vpkm, state.freeflow_kmh, state.jam_density,
            kc_ratio=state.kc_ratio,
        )
        state.speed_kmh = np.maximum(state.speed_kmh, 1.0)

        throughput = state.density_vpkm * state.speed_kmh  # veh/hr
        discharged = throughput * period_hr  # vehicles that leave
        queue_veh = np.maximum(queue_veh - discharged, 0.0)

        # Metrics for this drain period
        queue_rate_remaining = queue_veh / period_hr
        oversat_mask = queue_veh > 0
        per_lane = queue_rate_remaining / np.maximum(state.n_lanes, 1)
        mean_q = (
            float(np.mean(per_lane[oversat_mask]))
            if np.any(oversat_mask) else 0.0
        )
        active = state.flow_vph > 0
        active_speeds = state.speed_kmh[active] if np.any(active) else state.speed_kmh
        link_time_s = state.length_m * 3.6 / np.maximum(state.speed_kmh, 1.0)
        tstt = float(np.sum(state.flow_vph * link_time_s))

        period_metrics.append({
            "period": n_demand_periods + drain_p + 1,
            "label": f"D{drain_p+1}",
            "has_demand": False,
            "queue_veh_hr_lane": mean_q,
            "total_unserved_vph": float(np.sum(queue_rate_remaining)),
            "mean_speed_kmh": float(np.mean(active_speeds)),
            "n_oversaturated": int(np.sum(oversat_mask)),
            "tstt": tstt,
        })

    # ── Build report ──────────────────────────────────────────────
    node_coords = meta["nodes"]
    link_attrs = meta["link_attrs"]
    figs: list = []
    descriptions: list[str] = []

    # §1 — Congestion map: peak period final state
    _add_congestion_map_section(
        figs, descriptions,
        f"Chicago Sketch — End of Demand (Period {n_demand_periods})",
        node_coords, result.network_state, link_attrs, meta, 1.0,
    )

    # §2 — Period-level metrics (the key chart)
    x_labels = [m["label"] for m in period_metrics]
    demand_mask = [m["has_demand"] for m in period_metrics]
    n_demand = sum(demand_mask)

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "Queue (veh/hr/lane)", "Mean Speed (km/h)",
        "Oversaturated Links", "Total Unserved (veh/hr)",
    ], vertical_spacing=0.15, horizontal_spacing=0.10)

    colors = ["#D32F2F", "#2E7D32", "#FF6F00", "#6A1B9A"]
    keys = ["queue_veh_hr_lane", "mean_speed_kmh", "n_oversaturated",
            "total_unserved_vph"]
    y_titles = [
        "veh/hr/lane", "km/h", "links", "veh/hr",
    ]
    for idx, key in enumerate(keys):
        row, col = divmod(idx, 2)
        row += 1
        col += 1
        vals = [m[key] for m in period_metrics]
        fig.add_trace(go.Scatter(
            x=x_labels, y=vals,
            mode="lines+markers",
            line=dict(color=colors[idx], width=2.5),
            marker=dict(size=8),
            showlegend=False,
        ), row=row, col=col)

        fig.update_xaxes(title_text="Period", row=row, col=col)
        fig.update_yaxes(title_text=y_titles[idx], row=row, col=col)

        # Shade demand vs drain regions
        fig.add_vrect(
            x0=-0.5, x1=n_demand - 0.5,
            fillcolor="rgba(200,200,255,0.15)", line_width=0,
            row=row, col=col,
        )
        if len(period_metrics) > n_demand:
            fig.add_vrect(
                x0=n_demand - 0.5, x1=len(period_metrics) - 0.5,
                fillcolor="rgba(200,255,200,0.15)", line_width=0,
                row=row, col=col,
            )

    fig.update_layout(
        title="Per-Period Metrics: Demand → Drain",
        height=650, template="plotly_white",
    )
    figs.append(fig)

    n_drain = len(period_metrics) - n_demand
    final_unserved = period_metrics[-1]["total_unserved_vph"]
    peak_unserved_rate = period_metrics[n_demand - 1]["total_unserved_vph"]
    drain_pct = (1.0 - final_unserved / peak_unserved_rate) * 100 if peak_unserved_rate > 0 else 100
    descriptions.append(
        "<h2>Per-Period Progression</h2>"
        f"<p>{n_demand_periods} demand periods (P1–P{n_demand_periods}, "
        f"{int(period_s/60)} min each, {per_period_demand:,.0f} vph/period), "
        f"followed by {n_drain} drain periods (no new demand). "
        f"Blue shading = demand active, green shading = drain only.</p>"
        f"<p>Peak unserved: <b>{peak_unserved_rate:,.0f} veh/hr</b> → "
        f"after {n_drain} drain periods: <b>{final_unserved:,.0f} veh/hr</b> "
        f"({drain_pct:.1f}% reduction)</p>"
    )

    # §3 — MFD at peak (use saved peak state, not drain-mutated state)
    _add_mfd_section(figs, descriptions, peak_state, 1.0)

    # §4 — Summary table (no dummy figure)
    peak = period_metrics[n_demand - 1]
    final = period_metrics[-1]
    descriptions.append(
        "<h2>Summary</h2>"
        "<table style='border-collapse:collapse; margin:1em 0;'>"
        "<tr style='border-bottom:2px solid #333;'>"
        "<th style='padding:6px 16px; text-align:left;'>Metric</th>"
        "<th style='padding:6px 16px; text-align:right;'>Peak "
        f"(P{n_demand_periods})</th>"
        "<th style='padding:6px 16px; text-align:right;'>Post-Drain "
        f"({final['label']})</th></tr>"
        f"<tr><td style='padding:4px 16px;'>Total demand loaded</td>"
        f"<td style='padding:4px 16px; text-align:right;' colspan=2>"
        f"{per_period_demand * n_demand_periods:,.0f} vph "
        f"({n_demand_periods} × {per_period_demand:,.0f})</td></tr>"
        f"<tr><td style='padding:4px 16px;'>Queue (veh/hr/lane)</td>"
        f"<td style='padding:4px 16px; text-align:right;'>"
        f"{peak['queue_veh_hr_lane']:.0f}</td>"
        f"<td style='padding:4px 16px; text-align:right;'>"
        f"{final['queue_veh_hr_lane']:.0f}</td></tr>"
        f"<tr><td style='padding:4px 16px;'>Unserved (veh/hr)</td>"
        f"<td style='padding:4px 16px; text-align:right;'>"
        f"{peak['total_unserved_vph']:,.0f}</td>"
        f"<td style='padding:4px 16px; text-align:right;'>"
        f"{final['total_unserved_vph']:,.0f}</td></tr>"
        f"<tr><td style='padding:4px 16px;'>Mean speed (km/h)</td>"
        f"<td style='padding:4px 16px; text-align:right;'>"
        f"{peak['mean_speed_kmh']:.1f}</td>"
        f"<td style='padding:4px 16px; text-align:right;'>"
        f"{final['mean_speed_kmh']:.1f}</td></tr>"
        f"<tr><td style='padding:4px 16px;'>Oversaturated links</td>"
        f"<td style='padding:4px 16px; text-align:right;'>"
        f"{peak['n_oversaturated']}</td>"
        f"<td style='padding:4px 16px; text-align:right;'>"
        f"{final['n_oversaturated']}</td></tr>"
        f"<tr><td style='padding:4px 16px;'>Runtime</td>"
        f"<td style='padding:4px 16px; text-align:right;' colspan=2>"
        f"{result.total_time_s:.1f}s</td></tr>"
        "</table>"
    )

    out = Path(output_path)
    _write_combined_report(
        title="Chicago Sketch — Multi-Period Spillover Validation",
        intro=(
            f"<p>Spillover validation on <b>Chicago Sketch</b>: "
            f"{n_demand_periods} × 15-min demand periods "
            f"({per_period_demand:,.0f} vph each), then drain periods "
            f"until queue dissipates. Unserved demand from each period "
            f"carries forward as starting flow in the next.</p>"
        ),
        figures=figs,
        descriptions=descriptions,
        path=out,
    )
    return out

