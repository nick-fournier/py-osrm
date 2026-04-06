#!/usr/bin/env python3
"""Scale test: chi-regional with multi-period streaming assignment.

Runs the full assign_stream pipeline on the Chicago Regional TNTP
network (12,982 nodes, 39,018 links, 1,790 zones) with demand
distributed across a 24-hour day at 15-minute periods.

Produces an HTML report with congestion build-up, TT/FFTT ratio,
network loading vs. departures, link saturation, and assignment
convergence metrics.

Usage:
    uv run python scripts/scale_test_multi_period.py [--demand-scale 1.0]
"""

import argparse
import logging
import os
import resource
import shutil
import time
from pathlib import Path

import numpy as np

import osrm
from osrm.assignment import AssignmentConfig, AssignmentSolver, DensitySmoothingConfig
from osrm.assignment.osm_synthesis import tntp_to_osm, LinkClass, patch_lanes
from osrm.assignment.tntp import parse_net, parse_trips, load_node_coords, parse_flow
from osrm.assignment.od_matrix import DemandTrip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

FIXTURE_DIR = Path("tests/fixtures/chicago_regional")
N_PERIODS = 96
PERIOD_DURATION_S = 900.0  # 15 min


def _get_mem_mb():
    """Current process RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _regional_classify_override(link, dist_m, default_cls):
    if getattr(link, "link_type", 0) == 3:
        return LinkClass(
            highway="motorway_link", n_lanes=max(2, default_cls.n_lanes),
            speed_kmh=min(default_cls.speed_kmh or 100, 100),
        )
    if default_cls.speed_kmh and default_cls.speed_kmh > 130:
        return LinkClass(
            highway=default_cls.highway, n_lanes=default_cls.n_lanes,
            speed_kmh=130,
        )
    return None


def build_network(work: Path):
    """Build chi-regional OSRM MLD network. Returns (base_path, meta)."""
    logger.info("=== Building chi-regional OSRM network ===")

    t0 = time.monotonic()
    net = parse_net(FIXTURE_DIR / "ChicagoRegional_net.tntp")
    n_zones, od_matrix = parse_trips(FIXTURE_DIR / "ChicagoRegional_trips.npz")
    node_coords = load_node_coords(FIXTURE_DIR / "chicago_regional_nodes.geojson")
    ref_flows = parse_flow(FIXTURE_DIR / "ChicagoRegional_flow.tntp")
    logger.info("Parsed TNTP: %d links, %d zones, %.0f demand (%.1fs)",
                len(net.links), n_zones, od_matrix.sum(), time.monotonic() - t0)

    t0 = time.monotonic()
    osm_path, meta = tntp_to_osm(
        net, node_coords, od_matrix, work / "chi_regional.osm",
        ref_flows=ref_flows, speed_units="auto",
        classify_override=_regional_classify_override,
    )
    logger.info("OSM synthesis: %.1fs", time.monotonic() - t0)

    base = str(work / "chi_regional.osrm")

    t0 = time.monotonic()
    osrm.extract(str(osm_path), profile="car", output_path=base, verbosity="ERROR")
    extract_time = time.monotonic() - t0
    logger.info("Extract: %.1fs", extract_time)

    t0 = time.monotonic()
    osrm.partition(base, verbosity="ERROR")
    partition_time = time.monotonic() - t0
    logger.info("Partition: %.1fs", partition_time)

    t0 = time.monotonic()
    osrm.customize(base, verbosity="ERROR")
    customize_time = time.monotonic() - t0
    logger.info("Customize (single): %.1fs", customize_time)

    return base, meta


def demand_profile(n_periods: int) -> np.ndarray:
    """24h demand weight per period: AM/PM peaks, near-zero overnight."""
    hours = np.arange(n_periods) * (24.0 / n_periods)
    w = (
        0.50 * np.exp(-0.5 * ((hours - 8.0) / 1.2) ** 2) +
        0.40 * np.exp(-0.5 * ((hours - 17.5) / 1.5) ** 2) +
        0.08 * np.clip(np.cos(np.pi * (hours - 13.0) / 12.0), 0, 1)
    )
    return np.clip(w, 0, None)


def build_demand(
    meta: dict,
    demand_scale: float = 1.0,
) -> list[DemandTrip]:
    """Build trip list from OD matrix distributed across 24h.

    The TNTP demand (1.36M) represents peak-hour equilibrium demand.
    We normalize the demand profile so the peak hour's 4 periods sum to
    the TNTP demand × demand_scale, then scale off-peak proportionally.
    """
    centroids = meta["zone_centroids"]
    od = meta["od_matrix"]
    rng = np.random.default_rng(42)

    profile = demand_profile(N_PERIODS)

    # Identify peak hour (4 consecutive periods with max sum)
    period_sums = np.convolve(profile, np.ones(4), mode="valid")
    peak_start = int(np.argmax(period_sums))
    peak_hour_weight = profile[peak_start:peak_start + 4].sum()

    # Scale so peak hour sums to 1.0 × demand_scale of OD matrix
    # Each period's demand = od[i,j] * (profile[p] / peak_hour_weight) * demand_scale
    period_scale = (profile / peak_hour_weight) * demand_scale

    logger.info("Demand profile: peak hour periods %d-%d (%.1fh-%.1fh), "
                "scale range [%.3f, %.3f], total daily = %.1f× peak hour",
                peak_start, peak_start + 3,
                peak_start * 0.25, (peak_start + 4) * 0.25,
                period_scale.min(), period_scale.max(),
                period_scale.sum())

    trips = []
    for p in range(N_PERIODS):
        if period_scale[p] < 1e-6:
            continue
        dep_base_s = p * PERIOD_DURATION_S
        for i in range(od.shape[0]):
            for j in range(od.shape[1]):
                if od[i, j] > 0 and i != j:
                    o_zone, d_zone = i + 1, j + 1
                    if o_zone in centroids and d_zone in centroids:
                        vol = od[i, j] * period_scale[p]
                        if vol < 0.01:
                            continue
                        trips.append(DemandTrip(
                            origin=centroids[o_zone],
                            destination=centroids[d_zone],
                            volume=vol,
                            departure_time_s=dep_base_s + rng.uniform(0, PERIOD_DURATION_S),
                        ))

    total_demand = sum(t.volume for t in trips)
    n_od_pairs = len(set((t.origin, t.destination) for t in trips))
    logger.info("Built %d trips (%.0f total demand) from %d OD pairs across %d periods",
                len(trips), total_demand, n_od_pairs,
                len(set(int(t.departure_time_s // PERIOD_DURATION_S) for t in trips)))
    return trips


def run_assignment(
    base: str,
    meta: dict,
    trips: list[DemandTrip],
    work: Path,
):
    """Run assign_stream with production settings. Returns StreamResult."""
    logger.info("=== Running streaming assignment (%d trips) ===", len(trips))

    run_dir = work / "assignment_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy clean OSRM files so assign_stream can re-customize freely
    src = Path(base).parent
    for f in src.iterdir():
        shutil.copy2(f, run_dir / f.name)
    run_base = str(run_dir / Path(base).name)

    config = AssignmentConfig(
        smoothing=DensitySmoothingConfig(method="none"),
        speed_csv_dir=str(run_dir),
        verbosity="INFO",
    )
    solver = AssignmentSolver(run_base, config)

    def lane_patch(state):
        patch_lanes(state, meta)

    t0 = time.monotonic()
    result = solver.assign_stream(
        trips,
        period_duration_s=PERIOD_DURATION_S,
        state_patch=lane_patch,
    )
    total_time = time.monotonic() - t0

    logger.info("Assignment complete: %d batches in %.1fs (%.1f trips/s)",
                result.n_batches, total_time,
                result.n_trips / max(total_time, 0.001))

    return result



def generate_report(
    result,  # StreamResult from assign_stream
    trips: list[DemandTrip],
    total_time_s: float,
    output_path: str = "plots/scale_test_multi_period.html",
):
    """Generate an HTML report from streaming assignment results."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from osrm.assignment.plots import _write_combined_report

    figs = []
    descriptions = []
    hours = np.arange(N_PERIODS) * 0.25

    # ── Compute per-period demand from trips ─────────────────────────
    departures_per_period = np.zeros(N_PERIODS)
    demand_per_period = np.zeros(N_PERIODS)
    for t in trips:
        p = min(int(t.departure_time_s // PERIOD_DURATION_S), N_PERIODS - 1)
        departures_per_period[p] += 1
        demand_per_period[p] += t.volume

    # ── Derive data from StreamResult ────────────────────────────────
    state = result.network_state
    batch_log = result.batch_log
    period_flows = result.period_flows  # (n_periods, n_edges) or None

    if period_flows is not None and period_flows.ndim == 2:
        volume_per_period = np.sum(period_flows, axis=1)
        n_flow_periods = period_flows.shape[0]
    else:
        volume_per_period = np.zeros(N_PERIODS)
        n_flow_periods = 0

    # ── 1. Demand profile ────────────────────────────────────────────
    fig_demand = go.Figure()
    fig_demand.add_trace(go.Bar(
        name="Demand (veh)", x=hours.tolist(), y=demand_per_period.tolist(),
        marker_color="#1565C0", opacity=0.7,
        hovertemplate="Hour %{x:.1f}: %{y:,.0f} veh<extra></extra>",
    ))
    fig_demand.add_trace(go.Scatter(
        name="Trip count", x=hours.tolist(), y=departures_per_period.tolist(),
        mode="lines", line=dict(color="#E65100", width=2, dash="dash"),
        yaxis="y2",
    ))
    fig_demand.update_layout(
        template="plotly_white", height=350,
        xaxis_title="Hour of day",
        yaxis_title="Demand (vehicles)",
        yaxis2=dict(title="Trip count", overlaying="y", side="right"),
        xaxis=dict(dtick=2, range=[0, 24]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    figs.append(fig_demand)
    total_demand = demand_per_period.sum()
    peak_demand = demand_per_period.max()
    descriptions.append(
        "<h2>Demand Profile</h2>"
        f"<p>Total daily demand: <b>{total_demand:,.0f}</b> vehicles across "
        f"<b>{int(np.sum(departures_per_period > 0))}</b> active periods. "
        f"Peak period demand: <b>{peak_demand:,.0f}</b> veh. "
        f"Total trip records: <b>{len(trips):,}</b>.</p>"
    )

    # ── 2. Assignment convergence (batch metrics) ────────────────────
    if batch_log:
        batch_idx = [b.batch_index for b in batch_log]
        mean_speeds = [b.mean_speed_kmh for b in batch_log]
        queue_veh = [b.queue_vehicles for b in batch_log]
        n_oversat = [b.n_oversaturated for b in batch_log]
        tsstts = [b.tstt for b in batch_log]

        fig_conv = make_subplots(
            rows=2, cols=2, shared_xaxes=True,
            subplot_titles=["Mean Speed (km/h)", "Queue Vehicles (veh/hr/lane)",
                            "Oversaturated Links", "TSTT (veh·s)"],
            vertical_spacing=0.12, horizontal_spacing=0.10,
        )
        fig_conv.add_trace(go.Scatter(
            x=batch_idx, y=mean_speeds, mode="lines",
            line=dict(color="#1565C0", width=2),
        ), row=1, col=1)
        fig_conv.add_trace(go.Scatter(
            x=batch_idx, y=queue_veh, mode="lines",
            line=dict(color="#E65100", width=2),
        ), row=1, col=2)
        fig_conv.add_trace(go.Scatter(
            x=batch_idx, y=n_oversat, mode="lines",
            line=dict(color="#C62828", width=2),
        ), row=2, col=1)
        fig_conv.add_trace(go.Scatter(
            x=batch_idx, y=tsstts, mode="lines",
            line=dict(color="#43A047", width=2),
        ), row=2, col=2)
        fig_conv.update_xaxes(title_text="Batch", row=2, col=1)
        fig_conv.update_xaxes(title_text="Batch", row=2, col=2)
        fig_conv.update_layout(
            template="plotly_white", height=500, showlegend=False,
        )
        figs.append(fig_conv)

        final = batch_log[-1]
        descriptions.append(
            "<h2>Assignment Loading Profile</h2>"
            f"<p><b>{result.n_batches}</b> batches in "
            f"<b>{total_time_s:.0f}s</b> ({total_time_s/60:.1f} min). "
            f"Final state: mean speed <b>{final.mean_speed_kmh:.1f}</b> km/h, "
            f"min speed <b>{final.min_speed_kmh:.1f}</b> km/h, "
            f"<b>{final.n_oversaturated}</b> oversaturated links, "
            f"queue <b>{final.queue_vehicles:.1f}</b> veh/hr/lane.</p>"
        )

    # ── 3. Network loading vs departures (spillover) ─────────────────
    if n_flow_periods > 0:
        vpp = np.zeros(N_PERIODS)
        vpp[:min(n_flow_periods, N_PERIODS)] = volume_per_period[:min(n_flow_periods, N_PERIODS)]

        dep_scale = vpp.max() / max(demand_per_period.max(), 1)
        dep_scaled = demand_per_period * dep_scale

        fig_spill = go.Figure()
        fig_spill.add_trace(go.Bar(
            name="Network volume (actual)",
            x=hours.tolist(), y=vpp.tolist(),
            marker_color="#1565C0", opacity=0.7,
            hovertemplate="Hour %{x:.1f}: vol=%{y:,.0f}<extra>actual</extra>",
        ))
        fig_spill.add_trace(go.Scatter(
            name="Demand (scaled)", x=hours.tolist(), y=dep_scaled.tolist(),
            mode="lines", line=dict(color="#E65100", width=2.5, dash="dash"),
        ))
        fig_spill.update_layout(
            template="plotly_white", height=400,
            xaxis_title="Hour of day",
            yaxis_title="Total link-volume (veh·links)",
            xaxis=dict(dtick=2, range=[0, 24]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        figs.append(fig_spill)

        pure_spill = int(np.sum((vpp > 0) & (demand_per_period == 0)))
        if vpp.sum() > 0 and demand_per_period.sum() > 0:
            vol_c = np.average(np.arange(N_PERIODS), weights=np.clip(vpp, 1e-10, None))
            dep_c = np.average(np.arange(N_PERIODS), weights=np.clip(demand_per_period, 1e-10, None))
            lag_min = (vol_c - dep_c) * (PERIOD_DURATION_S / 60)
        else:
            lag_min = 0
        descriptions.append(
            "<h2>Network Loading vs. Trip Departures</h2>"
            "<p>Blue bars: actual link-volume per period from streaming "
            "assignment with VDF feedback. Dashed orange: demand (scaled). "
            "The rightward shift reveals <b>spillover</b>.</p>"
            f"<p>Volume centroid lags departures by <b>{lag_min:.1f} min</b>. "
            f"<b>{pure_spill}</b> periods have volume but zero departures.</p>"
        )

    # ── 4. Link saturation by period ─────────────────────────────────
    if period_flows is not None and period_flows.ndim == 2 and state is not None:
        n_edges = period_flows.shape[1]
        per_lane_cap = 1800.0
        edge_cap_vph = state.n_lanes[:n_edges].astype(float) * per_lane_cap
        edge_cap_per_period = edge_cap_vph * (PERIOD_DURATION_S / 3600.0)

        links_with_flow = np.zeros(N_PERIODS, dtype=int)
        links_saturated = np.zeros(N_PERIODS, dtype=int)
        links_over_50pct = np.zeros(N_PERIODS, dtype=int)
        max_vc_ratio = np.zeros(N_PERIODS, dtype=float)

        for p in range(min(n_flow_periods, N_PERIODS)):
            row = period_flows[p]
            links_with_flow[p] = int(np.sum(row > 0))
            vc = np.where(edge_cap_per_period > 0, row / edge_cap_per_period, 0)
            links_saturated[p] = int(np.sum(vc >= 1.0))
            links_over_50pct[p] = int(np.sum(vc >= 0.5))
            max_vc_ratio[p] = float(vc.max()) if len(vc) > 0 else 0

        fig_sat = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=["Link Utilization by Period", "V/C Ratio"],
        )
        fig_sat.add_trace(go.Scatter(
            name="Links with flow", x=hours.tolist(), y=links_with_flow.tolist(),
            mode="lines", line=dict(color="#1565C0", width=2),
            fill="tozeroy", fillcolor="rgba(21, 101, 192, 0.10)",
        ), row=1, col=1)
        fig_sat.add_trace(go.Scatter(
            name="Links > 50% V/C", x=hours.tolist(), y=links_over_50pct.tolist(),
            mode="lines", line=dict(color="#FF8F00", width=2),
            fill="tozeroy", fillcolor="rgba(255, 143, 0, 0.15)",
        ), row=1, col=1)
        fig_sat.add_trace(go.Scatter(
            name="Links >= 100% V/C", x=hours.tolist(), y=links_saturated.tolist(),
            mode="lines", line=dict(color="#C62828", width=2),
            fill="tozeroy", fillcolor="rgba(198, 40, 40, 0.15)",
        ), row=1, col=1)
        fig_sat.add_trace(go.Scatter(
            name="Max V/C ratio", x=hours.tolist(), y=max_vc_ratio.tolist(),
            mode="lines", line=dict(color="#E65100", width=2.5),
        ), row=2, col=1)
        fig_sat.add_hline(y=1.0, line_dash="dash", line_color="grey",
                          annotation_text="V/C = 1.0", row=2, col=1)
        fig_sat.update_xaxes(title_text="Hour of day", dtick=2, range=[0, 24], row=2, col=1)
        fig_sat.update_yaxes(title_text="Link count", row=1, col=1)
        fig_sat.update_yaxes(title_text="V/C ratio", row=2, col=1)
        fig_sat.update_layout(
            template="plotly_white", height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        figs.append(fig_sat)

        descriptions.append(
            "<h2>Link Saturation by Period</h2>"
            f"<p>Of {n_edges:,} discovered edges, "
            f"peak <b>{int(links_with_flow.max()):,}</b> have flow, "
            f"<b>{int(links_over_50pct.max()):,}</b> exceed 50% V/C, "
            f"<b>{int(links_saturated.max()):,}</b> fully saturated. "
            f"Peak V/C = <b>{float(max_vc_ratio.max()):.2f}</b>.</p>"
        )

    # ── 5. Summary ───────────────────────────────────────────────────
    figs.append(None)
    descriptions.append(
        "<h2>Summary</h2>"
        f"<p><b>Network:</b> chi-regional (12,982 nodes, 39,018 links, "
        f"1,790 zones)</p>"
        f"<p><b>Periods:</b> {N_PERIODS} x {PERIOD_DURATION_S:.0f}s "
        f"(24-hour day at 15-min intervals)</p>"
        f"<p><b>Total demand:</b> {total_demand:,.0f} vehicles</p>"
        f"<p><b>Trip records:</b> {len(trips):,}</p>"
        f"<p><b>Assignment:</b> {result.n_batches} batches in "
        f"{total_time_s:.0f}s ({total_time_s/60:.1f} min)</p>"
        f"<p><b>Peak RSS:</b> {_get_mem_mb():.0f} MB</p>"
    )

    out = Path(output_path)
    _write_combined_report(
        title=f"Multi-Period Scale Test - chi-regional, {N_PERIODS} periods",
        intro=(
            f"<p>Streaming assignment on the Chicago Regional network "
            f"({N_PERIODS} periods, 24h at 15-min intervals). "
            f"Total demand: {total_demand:,.0f} vehicles loaded via "
            f"<code>assign_stream</code> with autotune batching and "
            f"VDF feedback.</p>"
        ),
        figures=figs,
        descriptions=descriptions,
        path=out,
    )
    logger.info("Report written to %s", out)
    return out


def main():
    parser = argparse.ArgumentParser(description="Multi-period scale test")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Working directory (default: /tmp/scale_test)")
    parser.add_argument("--output", type=str, default="plots/scale_test_multi_period.html",
                        help="Report output path")
    parser.add_argument("--demand-scale", type=float, default=1.0,
                        help="Demand multiplier (default 1.0 = full peak-hour demand)")
    args = parser.parse_args()

    work = Path(args.work_dir or "/tmp/scale_test_multi_period")
    work.mkdir(parents=True, exist_ok=True)
    logger.info("Working directory: %s", work)
    logger.info("Cores available: %d", os.cpu_count() or 1)
    logger.info("Initial RSS: %.0f MB", _get_mem_mb())

    # 1. Build network
    base, meta = build_network(work)

    # 2. Build demand from OD matrix with 24h profile
    trips = build_demand(meta, demand_scale=args.demand_scale)

    # 3. Run streaming assignment
    t0 = time.monotonic()
    result = run_assignment(base, meta, trips, work)
    total_time = time.monotonic() - t0

    # 4. Generate report
    report_path = generate_report(
        result, trips, total_time,
        output_path=args.output,
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SCALE TEST SUMMARY: chi-regional, %d periods", N_PERIODS)
    logger.info("=" * 60)
    logger.info("Demand: %.0f veh (scale=%.2f), %d trips",
                sum(t.volume for t in trips), args.demand_scale, len(trips))
    logger.info("Assignment: %d batches in %.0fs",
                result.n_batches, total_time)
    if result.batch_log:
        final = result.batch_log[-1]
        logger.info("Final: mean_speed=%.1f km/h, oversaturated=%d, queue=%.1f",
                     final.mean_speed_kmh, final.n_oversaturated,
                     final.queue_vehicles)
    logger.info("Report: %s", report_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
