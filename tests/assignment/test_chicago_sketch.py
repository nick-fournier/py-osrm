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
                              period_duration_s: float = 900.0,
                              weights: list[float] | None = None) -> list:
    """Build trips with demand distributed across multiple periods.

    Each period gets the full OD matrix scaled by its weight.  Trips in
    period *k* have ``departure_time_s = k * period_duration_s``.
    If *weights* is None, demand is split evenly.
    """
    centroids = meta["zone_centroids"]
    od = meta["od_matrix"]
    if weights is None:
        weights = [1.0] * n_periods
    assert len(weights) == n_periods
    trips = []
    for k in range(n_periods):
        w = weights[k]
        dep = k * period_duration_s
        if w <= 0:
            # Sentinel trip so period transition fires in assign_stream
            if centroids:
                first = next(iter(centroids.values()))
                trips.append(DemandTrip(
                    origin=first, destination=first,
                    volume=0.0, departure_time_s=dep,
                ))
            continue
        for i in range(od.shape[0]):
            for j in range(od.shape[1]):
                if od[i, j] > 0 and i != j:
                    o_zone, d_zone = i + 1, j + 1
                    if o_zone in centroids and d_zone in centroids:
                        trips.append(DemandTrip(
                            origin=centroids[o_zone],
                            destination=centroids[d_zone],
                            volume=od[i, j] * w,
                            departure_time_s=dep,
                        ))
    return trips


def generate_chicago_spillover_report(
    tmp_path: str | Path,
    output_path: str = "plots/chicago_sketch_spillover.html",
    batch_size: int | None = None,
) -> Path:
    """Generate Chicago Sketch multi-period spillover validation report.

    Runs demand periods with trapezoidal loading (0.125, 0.25, 0.25,
    0.125) followed by unloaded periods (zero demand) where spillover
    discharges through the normal assignment loop.
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
    n_unloaded_periods = 8
    period_s = 900.0  # 15 min
    demand_weights = [1/6, 1/3, 1/3, 1/6]  # trapezoidal, sums to 1.0
    demand_scale = 3.0  # scale base OD to produce meaningful congestion
    n_total_periods = n_demand_periods + n_unloaded_periods

    # Build demand trips: trapezoidal demand + empty unloaded periods
    all_weights = [w * demand_scale for w in demand_weights] + [0.0] * n_unloaded_periods
    spill_trips = _build_trips_multiperiod(
        meta, n_periods=n_total_periods, period_duration_s=period_s,
        weights=all_weights,
    )
    base_demand = sum(t.volume for t in _build_trips(meta))

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
    # Compute period boundaries from the trip list: group by period
    # and count how many batches each period occupies.
    blog = result.log_as_dict()
    n_batches = len(blog["batch"])
    batch_size_used = blog["n_trips"][0] if blog["n_trips"] else 1

    # Count trips per period to compute batches per period.
    import math
    trips_per_period: list[int] = [0] * n_total_periods
    for t in spill_trips:
        p_idx = int(t.departure_time_s / period_s)
        p_idx = min(p_idx, n_total_periods - 1)
        trips_per_period[p_idx] += 1

    batches_per_period = [
        max(1, math.ceil(n / batch_size_used)) if n > 0 else 1
        for n in trips_per_period
    ]
    # Build period end indices (cumulative sum - 1)
    period_end_indices = []
    cum = 0
    for bp in batches_per_period:
        cum += bp
        period_end_indices.append(min(cum - 1, n_batches - 1))

    def _time_label(period_idx: int) -> str:
        """Format period end-time as 'h:mm'."""
        total_min = int((period_idx + 1) * period_s / 60)
        h, m = divmod(total_min, 60)
        return f"{h}:{m:02d}"

    period_metrics = []
    for p, end_idx in enumerate(period_end_indices):
        period_metrics.append({
            "period": p + 1,
            "label": _time_label(p),
            "has_demand": p < n_demand_periods,
            "queue_veh_hr_lane": blog["queue_vehicles"][end_idx],
            "total_unserved_vph": blog["total_unserved_vph"][end_idx],
            "mean_speed_kmh": blog["mean_speed_kmh"][end_idx],
            "n_oversaturated": blog["n_oversaturated"][end_idx],
            "tstt": blog["tstt"][end_idx],
        })

    state = result.network_state

    # ── Build report ──────────────────────────────────────────────
    node_coords = meta["nodes"]
    link_attrs = meta["link_attrs"]
    figs: list = []
    descriptions: list[str] = []

    # §1 — Congestion map: final state
    _add_congestion_map_section(
        figs, descriptions,
        f"Chicago Sketch — End of Period {n_demand_periods}",
        node_coords, result.network_state, link_attrs, meta, 1.0,
    )

    # §2 — Period-level metrics
    x_labels = [m["label"] for m in period_metrics]
    n_demand = sum(1 for m in period_metrics if m["has_demand"])

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

        fig.update_xaxes(title_text="Time", row=row, col=col)
        fig.update_yaxes(title_text=y_titles[idx], row=row, col=col)

        # Shade demand vs unloaded regions
        if n_demand < len(period_metrics):
            fig.add_vrect(
                x0=-0.5, x1=n_demand - 0.5,
                fillcolor="rgba(200,200,255,0.15)", line_width=0,
                row=row, col=col,
            )
            fig.add_vrect(
                x0=n_demand - 0.5, x1=len(period_metrics) - 0.5,
                fillcolor="rgba(200,255,200,0.15)", line_width=0,
                row=row, col=col,
            )

    fig.update_layout(
        title="Per-Period Metrics",
        height=650, template="plotly_white",
    )
    figs.append(fig)

    weight_str = ", ".join(f"{w:.2f}" for w in demand_weights)
    final = period_metrics[-1]
    n_unloaded = len(period_metrics) - n_demand
    descriptions.append(
        "<h2>Per-Period Progression</h2>"
        f"<p>{n_demand_periods} demand periods ({int(period_s/60)} min each, "
        f"trapezoidal [{weight_str}] × {demand_scale:.0f}× base demand) "
        f"+ {n_unloaded} unloaded periods. "
        f"Unserved demand carries forward as starting flow in the next period. "
        f"Blue = demand, green = unloaded.</p>"
        f"<p>Final unserved: <b>{final['total_unserved_vph']:,.0f} veh/hr</b></p>"
    )

    # §3 — MFD at final state
    _add_mfd_section(figs, descriptions, state, 1.0)

    # §4 — Summary table
    total_demand = sum(w * demand_scale * base_demand for w in demand_weights)
    peak_idx = max(range(n_demand), key=lambda i: period_metrics[i]["total_unserved_vph"])
    peak = period_metrics[peak_idx]
    descriptions.append(
        "<h2>Summary</h2>"
        "<table style='border-collapse:collapse; margin:1em 0;'>"
        "<tr style='border-bottom:2px solid #333;'>"
        "<th style='padding:6px 16px; text-align:left;'>Metric</th>"
        "<th style='padding:6px 16px; text-align:right;'>Peak "
        f"({peak['label']})</th>"
        "<th style='padding:6px 16px; text-align:right;'>Final "
        f"({final['label']})</th></tr>"
        f"<tr><td style='padding:4px 16px;'>Total demand loaded</td>"
        f"<td style='padding:4px 16px; text-align:right;' colspan=2>"
        f"{total_demand:,.0f} vph (weights [{weight_str}])</td></tr>"
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
            f"{n_demand_periods} × {int(period_s/60)}-min demand periods "
            f"(trapezoidal [{weight_str}], {demand_scale:.0f}× base) "
            f"+ {n_unloaded_periods} unloaded periods. Unserved demand "
            f"carries forward as starting flow in the next.</p>"
        ),
        figures=figs,
        descriptions=descriptions,
        path=out,
    )
    return out

