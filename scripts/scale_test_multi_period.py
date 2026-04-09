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
import math
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


def demand_profile(n_periods: int):
    """24h demand weights per period, split into AM/PM/midday components.

    Returns (am, pm, mid) arrays each of length n_periods.
    AM = home→work (OD as-is), PM = work→home (OD transposed),
    midday = symmetric blend.
    """
    hours = np.arange(n_periods) * (24.0 / n_periods)
    am = 0.50 * np.exp(-0.5 * ((hours - 8.0) / 1.2) ** 2)
    pm = 0.40 * np.exp(-0.5 * ((hours - 17.5) / 1.5) ** 2)
    mid = 0.25 * np.clip(np.cos(np.pi * (hours - 13.0) / 10.0), 0, 1)
    return np.clip(am, 0, None), np.clip(pm, 0, None), np.clip(mid, 0, None)


def build_demand(
    meta: dict,
    demand_scale: float = 1.0,
    min_volume: float = 0.5,
) -> list[DemandTrip]:
    """Build trip list from OD matrix distributed across 24h.

    The TNTP demand (1.36M) represents peak-hour equilibrium demand.
    AM periods use OD (home→work), PM periods use OD.T (work→home),
    midday uses a symmetric blend (OD + OD.T) / 2.

    Args:
        min_volume: Minimum vehicles per period to include an OD pair.
            Default 0.5 — keeps ~47% of demand while limiting trip count.
    """
    centroids = meta["zone_centroids"]
    od = meta["od_matrix"]
    od_t = od.T.copy()  # transposed: work→home
    rng = np.random.default_rng(42)

    am, pm, mid = demand_profile(N_PERIODS)
    total_profile = am + pm + mid

    # Identify peak hour (4 consecutive periods with max total weight)
    period_sums = np.convolve(total_profile, np.ones(4), mode="valid")
    peak_start = int(np.argmax(period_sums))
    peak_hour_weight = total_profile[peak_start:peak_start + 4].sum()

    # Scale so peak hour sums to 1.0 × demand_scale of OD matrix
    scale_factor = demand_scale / peak_hour_weight

    logger.info("Demand profile: peak hour periods %d-%d (%.1fh-%.1fh), "
                "daily = %.1f× peak hour, directional AM/PM/mid split",
                peak_start, peak_start + 3,
                peak_start * 0.25, (peak_start + 4) * 0.25,
                total_profile.sum() / peak_hour_weight)

    # Extract non-zero OD pairs from union of OD and OD.T
    t0 = time.monotonic()
    combined = od + od_t
    ii, jj = np.where((combined > 0) & ~np.eye(od.shape[0], dtype=bool))
    zone_o, zone_d = ii + 1, jj + 1  # TNTP zones are 1-indexed
    valid = np.array([z in centroids for z in zone_o]) & \
            np.array([z in centroids for z in zone_d])
    ii, jj = ii[valid], jj[valid]

    # Per-pair base volumes for each direction
    fwd_vols = od[ii, jj]      # home→work
    rev_vols = od_t[ii, jj]    # work→home
    origins = [centroids[z] for z in (ii + 1)]
    dests = [centroids[z] for z in (jj + 1)]
    n_pairs = len(ii)
    logger.info("OD pairs (union fwd+rev): %d (%.1fs)", n_pairs, time.monotonic() - t0)

    # Build trips per period with directional blending
    t0 = time.monotonic()
    trips = []
    total_demand_full = 0.0
    total_demand_kept = 0.0

    for p in range(N_PERIODS):
        w_total = total_profile[p]
        if w_total < 1e-8:
            continue

        # Blend: AM→forward, PM→reverse, midday→symmetric average
        w_am = am[p] * scale_factor
        w_pm = pm[p] * scale_factor
        w_mid = mid[p] * scale_factor
        vols = fwd_vols * (w_am + w_mid * 0.5) + rev_vols * (w_pm + w_mid * 0.5)

        total_demand_full += vols.sum()
        mask = vols >= min_volume
        n_keep = int(mask.sum())
        if n_keep == 0:
            continue
        kept_vols = vols[mask]
        total_demand_kept += kept_vols.sum()
        dep_times = p * PERIOD_DURATION_S + rng.uniform(0, PERIOD_DURATION_S, size=n_keep)
        kept_idx = np.where(mask)[0]
        trips.extend(
            DemandTrip(origins[k], dests[k], float(v), float(t))
            for k, v, t in zip(kept_idx, kept_vols, dep_times)
        )

    pct = 100 * total_demand_kept / max(total_demand_full, 1)
    logger.info("Built %d trips in %.1fs (%.0f of %.0f demand kept = %.1f%%, "
                "min_volume=%.1f)",
                len(trips), time.monotonic() - t0,
                total_demand_kept, total_demand_full, pct, min_volume)
    return trips


def run_assignment(
    base: str,
    meta: dict,
    trips: list[DemandTrip],
    work: Path,
    batch_size: int | None = None,
    gap_sample_frac: float = 0.0,
):
    """Run assign_stream with production settings. Returns StreamResult."""
    logger.info("=== Running streaming assignment (%d trips) ===", len(trips))

    run_dir = work / "assignment_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy clean OSRM files so assign_stream can re-customize freely
    src = Path(base).parent
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, run_dir / f.name)
    run_base = str(run_dir / Path(base).name)

    config = AssignmentConfig(
        smoothing=DensitySmoothingConfig(method="none"),
        speed_csv_dir=str(run_dir),
        verbosity="INFO",
        gap_sample_frac=gap_sample_frac,
    )
    solver = AssignmentSolver(run_base, config)

    def lane_patch(state):
        patch_lanes(state, meta)

    t0 = time.monotonic()
    result = solver.assign_stream(
        trips,
        batch_size=batch_size,
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
        p50_speeds = [b.p50_speed_kmh for b in batch_log]
        p10_speeds = [b.p10_speed_kmh for b in batch_log]
        queue_veh = [b.queue_vehicles for b in batch_log]
        n_oversat = [b.n_oversaturated for b in batch_log]
        tsstts = [b.tstt for b in batch_log]
        eff_bs = [b.effective_batch_size for b in batch_log]
        vc_cvs = [b.vc_cv for b in batch_log]

        # Identify period-final batches (last batch before departure_bin changes)
        period_final = []
        for i in range(len(batch_log)):
            if (i == len(batch_log) - 1
                    or batch_log[i + 1].departure_bin != batch_log[i].departure_bin):
                period_final.append(i)
        pf_idx = [batch_idx[i] for i in period_final]
        pf_mean = [mean_speeds[i] for i in period_final]
        pf_p50 = [p50_speeds[i] for i in period_final]
        pf_p10 = [p10_speeds[i] for i in period_final]
        pf_queue = [queue_veh[i] for i in period_final]
        pf_oversat = [n_oversat[i] for i in period_final]
        pf_tstt = [tsstts[i] for i in period_final]
        pf_vc_cv = [vc_cvs[i] for i in period_final]
        pf_flow_stab = [batch_log[i].flow_stability for i in period_final]
        pf_gap = [batch_log[i].sampled_gap for i in period_final]

        fig_conv = make_subplots(
            rows=4, cols=2, shared_xaxes=True,
            subplot_titles=["Speed (km/h)", "Queue Vehicles (veh/hr/lane)",
                            "Oversaturated Links", "TSTT (veh·s)",
                            "V/C CV (utilization uniformity)",
                            "Sampled Gap / Flow Stability",
                            "Effective Batch Size",
                            "Congestion Ratios (r signals)"],
            vertical_spacing=0.06, horizontal_spacing=0.10,
        )

        # All batches — dashed, thin, low opacity
        batch_traces = [
            (mean_speeds, "#1565C0", 1, 1),
            (queue_veh,   "#E65100", 1, 2),
            (n_oversat,   "#C62828", 2, 1),
            (tsstts,      "#43A047", 2, 2),
            (vc_cvs,      "#6A1B9A", 3, 1),
        ]
        for y, color, row, col in batch_traces:
            fig_conv.add_trace(go.Scatter(
                x=batch_idx, y=y, mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.25, name="batch", showlegend=False,
            ), row=row, col=col)

        # Period-final speed bands: mean, P50, P10
        fig_conv.add_trace(go.Scatter(
            x=pf_idx, y=pf_mean, mode="lines+markers",
            line=dict(color="#1565C0", width=2),
            marker=dict(size=4), name="Mean",
        ), row=1, col=1)
        fig_conv.add_trace(go.Scatter(
            x=pf_idx, y=pf_p50, mode="lines+markers",
            line=dict(color="#42A5F5", width=2, dash="dash"),
            marker=dict(size=3), name="P50",
        ), row=1, col=1)
        fig_conv.add_trace(go.Scatter(
            x=pf_idx, y=pf_p10, mode="lines+markers",
            line=dict(color="#EF5350", width=2, dash="dash"),
            marker=dict(size=3), name="P10",
        ), row=1, col=1)

        # Period-final for other core metrics
        pf_other = [
            (pf_queue,   "#E65100", 1, 2),
            (pf_oversat, "#C62828", 2, 1),
            (pf_tstt,    "#43A047", 2, 2),
        ]
        for y, color, row, col in pf_other:
            fig_conv.add_trace(go.Scatter(
                x=pf_idx, y=y, mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=4), showlegend=False,
            ), row=row, col=col)

        # V/C CV period-final
        fig_conv.add_trace(go.Scatter(
            x=pf_idx, y=pf_vc_cv, mode="lines+markers",
            line=dict(color="#6A1B9A", width=2),
            marker=dict(size=4), showlegend=False,
        ), row=3, col=1)

        # Sampled gap + flow stability (period-final only, may contain NaN)
        valid_gap = [(x, y) for x, y in zip(pf_idx, pf_gap)
                     if not math.isnan(y)]
        valid_stab = [(x, y) for x, y in zip(pf_idx, pf_flow_stab)
                      if not math.isnan(y)]
        if valid_gap:
            fig_conv.add_trace(go.Scatter(
                x=[p[0] for p in valid_gap],
                y=[p[1] for p in valid_gap],
                mode="lines+markers",
                line=dict(color="#D84315", width=2),
                marker=dict(size=5), name="Sampled Gap",
            ), row=3, col=2)
        if valid_stab:
            fig_conv.add_trace(go.Scatter(
                x=[p[0] for p in valid_stab],
                y=[p[1] for p in valid_stab],
                mode="lines+markers",
                line=dict(color="#00897B", width=2, dash="dash"),
                marker=dict(size=4), name="Flow Δ‖·‖",
            ), row=3, col=2)

        # Congestion ratio series (for diagnostic panel)
        mean_vcs = [b.mean_vc for b in batch_log]
        frac_gt80 = [b.frac_vc_gt80 for b in batch_log]

        fig_conv.add_trace(go.Scatter(
            x=batch_idx, y=eff_bs, mode="lines",
            line=dict(color="#7B1FA2", width=2),
            fill="tozeroy", fillcolor="rgba(123, 31, 162, 0.10)",
            showlegend=False,
        ), row=4, col=1)

        # Congestion ratios panel (row 4, col 2)
        fig_conv.add_trace(go.Scatter(
            x=batch_idx, y=mean_vcs, mode="lines",
            line=dict(color="#1565C0", width=2),
            name="Mean V/C",
        ), row=4, col=2)
        fig_conv.add_trace(go.Scatter(
            x=batch_idx, y=frac_gt80, mode="lines",
            line=dict(color="#E65100", width=2, dash="dash"),
            name="Frac V/C>0.8",
        ), row=4, col=2)
        fig_conv.update_xaxes(title_text="Batch", row=4, col=1)
        fig_conv.update_xaxes(title_text="Batch", row=4, col=2)
        fig_conv.update_layout(
            template="plotly_white", height=850,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        )
        figs.append(fig_conv)

        final = batch_log[-1]
        bs_min = min(eff_bs)
        bs_max = max(eff_bs)
        gap_str = (f"sampled gap <b>{final.sampled_gap:.4f}</b>, "
                   if not math.isnan(final.sampled_gap) else "")
        descriptions.append(
            "<h2>Assignment Loading Profile</h2>"
            f"<p><b>{result.n_batches}</b> batches in "
            f"<b>{total_time_s:.0f}s</b> ({total_time_s/60:.1f} min). "
            f"Batch size ranged from <b>{bs_min:,}</b> to <b>{bs_max:,}</b>. "
            f"Solid line = period-final state, dashed = intra-period batches. "
            f"Final state: mean speed <b>{final.mean_speed_kmh:.1f}</b> km/h, "
            f"min speed <b>{final.min_speed_kmh:.1f}</b> km/h, "
            f"<b>{final.n_oversaturated}</b> oversaturated links, "
            f"queue <b>{final.queue_vehicles:.1f}</b> veh/hr/lane, "
            f"V/C CV <b>{final.vc_cv:.2f}</b>, "
            f"{gap_str}"
            f"</p>"
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
    parser.add_argument("--min-volume", type=float, default=0.5,
                        help="Min vehicles/period to keep an OD pair (default 0.5)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Fixed batch size (default: autotune)")
    parser.add_argument("--gap-sample", type=float, default=0.0,
                        help="Fraction of period trips to re-route for gap estimate "
                             "(0 = disabled, 0.01-0.10 recommended)")
    args = parser.parse_args()

    work = Path(args.work_dir or "/tmp/scale_test_multi_period")
    work.mkdir(parents=True, exist_ok=True)
    logger.info("Working directory: %s", work)
    logger.info("Cores available: %d", os.cpu_count() or 1)
    logger.info("Initial RSS: %.0f MB", _get_mem_mb())

    # 1. Build network
    base, meta = build_network(work)

    # 2. Build demand from OD matrix with 24h profile
    trips = build_demand(meta, demand_scale=args.demand_scale,
                         min_volume=args.min_volume)

    # 3. Run streaming assignment
    t0 = time.monotonic()
    result = run_assignment(base, meta, trips, work,
                            batch_size=args.batch_size,
                            gap_sample_frac=args.gap_sample)
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
