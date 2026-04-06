#!/usr/bin/env python3
"""Scale test: chi-regional with 96 periods (24h @ 15-min).

Benchmarks the multi-period OSRM infrastructure:
  1. Build chi-regional OSRM network from TNTP data
  2. Generate 96 synthetic per-period speed CSVs (time-of-day profile)
  3. customize_multi_period with all 96 CSVs → timing
  4. Load engine → memory footprint
  5. Route sample trips across periods → throughput

Usage:
    uv run python scripts/scale_test_multi_period.py [--work-dir /path/to/workdir]
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
from osrm.preprocessing import customize_multi_period
from osrm.assignment.osm_synthesis import tntp_to_osm, LinkClass, patch_lanes
from osrm.assignment.tntp import parse_net, parse_trips, load_node_coords, parse_flow
from osrm.assignment.segment_speed_writer import SegmentSpeedWriter
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


def generate_period_csvs(base: str, meta: dict, work: Path):
    """Generate 96 speed CSVs with a time-of-day congestion profile.

    Profile: sinusoidal AM/PM peaks (periods 28-36 and 64-72)
    with freeflow overnight and moderate midday.
    """
    logger.info("=== Generating %d period speed CSVs ===", N_PERIODS)

    # Discover network edges via a quick route probe
    engine = osrm.OSRM(
        storage_config=base, algorithm="MLD", use_shared_memory=False,
    )

    # Build OD pairs from centroids
    centroids = meta["zone_centroids"]
    zone_ids = sorted(centroids.keys())

    # Sample some routes to discover edges
    rng = np.random.default_rng(42)
    sample_zones = rng.choice(zone_ids, size=min(200, len(zone_ids)), replace=False)

    from osrm.assignment.network_state import NetworkState
    state = NetworkState.empty()

    coords_list = []
    for i in range(0, len(sample_zones) - 1, 2):
        o_z, d_z = sample_zones[i], sample_zones[i + 1]
        if o_z in centroids and d_z in centroids:
            coords_list.append((centroids[o_z], centroids[d_z]))

    # Route to discover edges and their freeflow speeds
    logger.info("Discovering edges via %d probe routes...", len(coords_list))
    for o, d in coords_list:
        rp = osrm.RouteParameters()
        rp.coordinates = [o, d]
        rp.annotations = True
        rp.annotations_type = ["nodes", "speed", "distance"]
        result = engine.Route(rp)
        if result.get("code") == "Ok" and result.get("routes"):
            for leg in result["routes"][0]["legs"]:
                ann = leg["annotation"]
                nodes = ann["nodes"]
                speeds = ann.get("speed", [])
                dists = ann.get("distance", [])
                for si in range(len(nodes) - 1):
                    spd = speeds[si] * 3.6 if si < len(speeds) else 50.0
                    dist = dists[si] if si < len(dists) else 100.0
                    state.register_edge(
                        int(nodes[si]), int(nodes[si + 1]),
                        float(dist), max(float(spd), 1.0),
                        150.0,  # default jam density
                        2,      # default lanes
                    )

    del engine
    logger.info("Discovered %d edges", state.n_edges)

    # Generate congestion factor per period (0-95)
    # AM peak: periods 28-36 (7:00-9:00), PM peak: 64-72 (16:00-18:00)
    hours = np.arange(N_PERIODS) * 0.25
    # Smooth Gaussian peaks + raised-cosine midday shoulder
    am = 0.50 * np.exp(-0.5 * ((hours - 8.0) / 1.0) ** 2)
    pm = 0.60 * np.exp(-0.5 * ((hours - 17.0) / 1.2) ** 2)
    midday = 0.15 * np.clip(np.cos(np.pi * (hours - 12.5) / 8.0), 0, 1)
    congestion_factors = am + pm + midday

    logger.info("Congestion factors: min=%.2f, max=%.2f, mean=%.2f",
                congestion_factors.min(), congestion_factors.max(),
                congestion_factors.mean())

    writer = SegmentSpeedWriter(output_dir=str(work / "csvs"), prefix="period")
    csv_paths = []

    for p in range(N_PERIODS):
        # Speed = freeflow * (1 - congestion_factor * 0.7)
        # At peak: speeds drop to ~30% of freeflow
        factor = 1.0 - congestion_factors[p] * 0.7
        speeds = state.freeflow_kmh * factor
        speeds = np.clip(speeds, 1.0, None)

        csv_path = writer.write(state.edge_ids, speeds, suffix=f"_{p:03d}")
        csv_paths.append(str(csv_path))

    logger.info("Wrote %d CSVs to %s", len(csv_paths), work / "csvs")
    return csv_paths, congestion_factors


def benchmark_customize(base: str, csv_paths: list):
    """Run customize_multi_period with all 96 CSVs and time it."""
    logger.info("=== Benchmarking customize_multi_period (%d periods) ===",
                len(csv_paths))

    period_speed_files = [(p, path) for p, path in enumerate(csv_paths)]

    mem_before = _get_mem_mb()
    t0 = time.monotonic()
    customize_multi_period(
        base,
        period_speed_files=period_speed_files,
        verbosity="INFO",
    )
    cust_time = time.monotonic() - t0
    mem_after = _get_mem_mb()

    logger.info("customize_multi_period: %.1fs, mem delta: +%.0f MB",
                cust_time, mem_after - mem_before)

    # Check file sizes
    base_path = Path(base)
    for suffix in [".cell_metrics", ".mldgr"]:
        f = base_path.parent / (base_path.name + suffix)
        if f.exists():
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.info("  %s: %.1f MB", suffix, size_mb)

    return cust_time


def benchmark_engine_load(base: str):
    """Load the multi-period engine and measure memory."""
    logger.info("=== Benchmarking engine load ===")

    mem_before = _get_mem_mb()
    t0 = time.monotonic()
    engine = osrm.OSRM(
        storage_config=base, algorithm="MLD", use_shared_memory=False,
    )
    load_time = time.monotonic() - t0
    mem_after = _get_mem_mb()

    logger.info("Engine load: %.1fs, mem: %.0f MB (delta +%.0f MB)",
                load_time, mem_after, mem_after - mem_before)

    return engine, load_time, mem_after


def benchmark_routing(engine, meta: dict, period_duration_s: float):
    """Route sample trips at various departure periods and measure throughput."""
    logger.info("=== Benchmarking routing throughput ===")

    from osrm.osrm_ext import batch_route_accumulate

    centroids = meta["zone_centroids"]
    zone_ids = sorted(centroids.keys())
    rng = np.random.default_rng(99)

    # Build 20000 OD pairs for statistically meaningful durations
    n_pairs = 20000
    o_zones = rng.choice(zone_ids, size=n_pairs)
    d_zones = rng.choice(zone_ids, size=n_pairs)

    coords = np.empty((n_pairs, 4), dtype=np.float64)
    volumes = np.ones(n_pairs, dtype=np.float64)

    valid = 0
    for i in range(n_pairs):
        o, d = int(o_zones[i]), int(d_zones[i])
        if o == d or o not in centroids or d not in centroids:
            continue
        oc, dc = centroids[o], centroids[d]
        coords[valid] = [oc[0], oc[1], dc[0], dc[1]]
        valid += 1

    coords = coords[:valid]
    volumes = volumes[:valid]
    edge_ids = np.zeros((0, 2), dtype=np.uint64)
    logger.info("Built %d valid OD pairs for routing benchmark", valid)

    results = {}

    # ── freeflow baseline (period 0 with no congestion) ─────────────
    logger.info("  Routing freeflow baseline (period 0)...")
    _, _, _, _, ff_durations = batch_route_accumulate(
        engine._engine, coords, volumes, edge_ids,
        n_threads=0, return_routes=False,
        departure_period=0, period_duration=period_duration_s,
        departure_offsets=np.zeros(valid, dtype=np.float64),
        n_periods=0,
    )
    ff_durs = np.asarray(ff_durations)
    ff_routed = ff_durs > 0
    ff_mean = np.mean(ff_durs[ff_routed]) if np.any(ff_routed) else 1.0

    # ── sweep all periods for TT/FFTT profile ───────────────────────
    logger.info("  Sweeping all %d periods for TT/FFTT profile...", N_PERIODS)
    tt_ratio_by_period = np.ones(N_PERIODS)
    for p in range(N_PERIODS):
        _, _, _, _, durs_p = batch_route_accumulate(
            engine._engine, coords, volumes, edge_ids,
            n_threads=0, return_routes=False,
            departure_period=p, period_duration=period_duration_s,
            departure_offsets=np.zeros(valid, dtype=np.float64),
            n_periods=0,
        )
        durs_p = np.asarray(durs_p)
        mask = ff_routed & (durs_p > 0)
        if np.any(mask):
            tt_ratio_by_period[p] = np.mean(durs_p[mask]) / np.mean(ff_durs[mask])

    logger.info("  TT/FFTT range: [%.2f, %.2f]",
                tt_ratio_by_period.min(), tt_ratio_by_period.max())

    # ── spotlight periods for throughput + spillover ─────────────────
    test_periods = [
        ("off-peak (3am)", 12),
        ("AM peak (8am)", 32),
        ("midday (12pm)", 48),
        ("PM peak (5pm)", 68),
    ]

    spillover_data = {}  # departure_label → per-period volume totals

    for label, dep_period in test_periods:
        offsets = rng.uniform(0, period_duration_s, size=valid).astype(np.float64)

        t0 = time.monotonic()
        vol2d, tstt, new_edges, _, durations = batch_route_accumulate(
            engine._engine, coords, volumes, edge_ids,
            n_threads=0, return_routes=False,
            departure_period=dep_period,
            period_duration=period_duration_s,
            departure_offsets=offsets,
            n_periods=N_PERIODS,
        )
        elapsed = time.monotonic() - t0

        durs = np.asarray(durations)
        routed = np.sum(durs > 0)
        mean_dur = np.mean(durs[durs > 0]) if routed > 0 else 0

        throughput = routed / max(elapsed, 0.001)
        results[label] = {
            "period": dep_period,
            "routed": int(routed),
            "elapsed_s": elapsed,
            "throughput": throughput,
            "mean_duration_s": mean_dur,
        }

        # Per-period total volume for spillover visualization
        v2d = np.asarray(vol2d)
        spillover_data[label] = {
            "dep_period": dep_period,
            "period_volumes": np.sum(v2d, axis=1),  # shape (N_PERIODS,)
        }

        logger.info(
            "  %s (p=%d): %d/%d routed in %.2fs (%.0f routes/s), "
            "mean_dur=%.1fs, spill_periods=%d",
            label, dep_period, routed, valid, elapsed, throughput, mean_dur,
            np.sum(np.any(v2d > 0, axis=1)),
        )

    return results, tt_ratio_by_period, spillover_data


def generate_report(
    congestion_factors: np.ndarray,
    cust_time: float,
    load_time: float,
    mem_mb: float,
    routing_results: dict,
    tt_ratio_by_period: np.ndarray,
    spillover_data: dict,
    output_path: str = "plots/scale_test_multi_period.html",
):
    """Generate an HTML report with scale test results."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from osrm.assignment.plots import _write_combined_report

    figs = []
    descriptions = []

    # ── 1. Time-of-day congestion profile ──────────────────────────
    hours = np.arange(N_PERIODS) * 0.25
    fig_profile = go.Figure()
    fig_profile.add_trace(go.Scatter(
        x=hours.tolist(),
        y=congestion_factors.tolist(),
        mode="lines",
        fill="tozeroy",
        line=dict(color="#1565C0", width=2),
        fillcolor="rgba(21, 101, 192, 0.15)",
        hovertemplate="Hour %{x:.1f}: factor=%{y:.2f}<extra></extra>",
    ))
    fig_profile.update_layout(
        template="plotly_white", height=350,
        xaxis_title="Hour of day",
        yaxis_title="Congestion factor",
        xaxis=dict(dtick=2, range=[0, 24]),
        yaxis=dict(rangemode="tozero"),
    )
    figs.append(fig_profile)
    descriptions.append(
        "<h2>Time-of-Day Congestion Profile</h2>"
        "<p>Synthetic congestion factors applied to generate per-period "
        "speed CSVs. AM peak ~8:00, PM peak ~17:00. Factor of 0.6 means "
        "speeds drop to 58% of freeflow.</p>"
    )

    # ── 2. TT / FFTT ratio by time of day ─────────────────────────
    fig_ratio = go.Figure()
    fig_ratio.add_trace(go.Scatter(
        x=hours.tolist(),
        y=tt_ratio_by_period.tolist(),
        mode="lines",
        line=dict(color="#E65100", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(230, 81, 0, 0.10)",
        hovertemplate="Hour %{x:.1f}: TT/FFTT=%{y:.3f}<extra></extra>",
    ))
    fig_ratio.add_hline(y=1.0, line_dash="dash", line_color="grey",
                        annotation_text="freeflow")
    fig_ratio.update_layout(
        template="plotly_white", height=350,
        xaxis_title="Hour of day",
        yaxis_title="TT / FFTT",
        xaxis=dict(dtick=2, range=[0, 24]),
        yaxis=dict(rangemode="tozero"),
    )
    figs.append(fig_ratio)
    descriptions.append(
        "<h2>Experienced Travel Time Ratio (TT / FFTT)</h2>"
        "<p>Mean travel time across 20k OD pairs at each period divided by "
        "the freeflow mean. A ratio of 1.0 is uncongested; higher values "
        "reflect the congestion penalty from the synthetic speed profile. "
        f"Peak ratio: <b>{tt_ratio_by_period.max():.3f}</b>.</p>"
    )

    # ── 2. Infrastructure timing bar chart ─────────────────────────
    timing_labels = [
        f"customize\n({N_PERIODS} periods)",
        "engine load",
    ]
    timing_values = [cust_time, load_time]

    fig_timing = go.Figure()
    fig_timing.add_trace(go.Bar(
        x=timing_labels,
        y=timing_values,
        text=[f"{v:.1f}s" for v in timing_values],
        textposition="outside",
        marker_color=["#1565C0", "#2196F3"],
    ))
    fig_timing.update_layout(
        template="plotly_white", height=350,
        yaxis_title="Time (seconds)",
        yaxis=dict(rangemode="tozero"),
    )
    figs.append(fig_timing)
    descriptions.append(
        "<h2>Infrastructure Timing</h2>"
        f"<p><code>customize_multi_period</code> with {N_PERIODS} period "
        f"CSVs: <b>{cust_time:.1f}s</b>. Engine load (all period cell "
        f"metrics + weight deltas): <b>{load_time:.1f}s</b>. "
        f"Peak RSS: <b>{mem_mb:.0f} MB</b>.</p>"
    )

    # ── 3. Routing throughput by period ────────────────────────────
    r_labels = list(routing_results.keys())
    r_throughput = [routing_results[k]["throughput"] for k in r_labels]
    r_mean_dur = [routing_results[k]["mean_duration_s"] for k in r_labels]

    fig_route = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Routing Throughput", "Mean Trip Duration"],
    )
    fig_route.add_trace(go.Bar(
        x=r_labels, y=r_throughput,
        text=[f"{v:.0f}" for v in r_throughput],
        textposition="outside",
        marker_color="#43A047",
    ), row=1, col=1)
    fig_route.add_trace(go.Bar(
        x=r_labels, y=r_mean_dur,
        text=[f"{v:.0f}s" for v in r_mean_dur],
        textposition="outside",
        marker_color="#FF8F00",
    ), row=1, col=2)
    fig_route.update_yaxes(title_text="Routes/sec", row=1, col=1)
    fig_route.update_yaxes(title_text="Duration (s)", row=1, col=2)
    fig_route.update_layout(
        template="plotly_white", height=400, showlegend=False,
    )
    figs.append(fig_route)
    descriptions.append(
        "<h2>Routing Performance by Departure Period</h2>"
        "<p>Throughput (routes/s) and mean trip duration routing 20k OD "
        "pairs at different times of day against a 96-period multi-period "
        "OSRM engine. Throughput should be stable regardless of period; "
        "duration varies with congestion level.</p>"
    )

    # ── 5. Cross-period spillover stacked bar ─────────────────────
    fig_spill = go.Figure()
    colors = ["#1565C0", "#43A047", "#FF8F00", "#E65100"]
    for idx, (label, sd) in enumerate(spillover_data.items()):
        pvol = sd["period_volumes"]
        dep_p = sd["dep_period"]
        # Show periods around the departure with nonzero volume
        nonzero = np.where(pvol > 0)[0]
        if len(nonzero) == 0:
            continue
        p_lo, p_hi = max(0, nonzero[0]), min(N_PERIODS - 1, nonzero[-1])
        # Include 1 period padding each side for context
        p_lo = max(0, p_lo - 1)
        p_hi = min(N_PERIODS - 1, p_hi + 1)
        ps = np.arange(p_lo, p_hi + 1)
        ph = ps * 0.25  # convert to hours

        # Split into: departure period volume vs spillover volume
        dep_vol = np.where(ps == dep_p, pvol[ps], 0.0)
        spill_vol = np.where(ps != dep_p, pvol[ps], 0.0)

        fig_spill.add_trace(go.Bar(
            name=f"{label} — departure",
            x=ph.tolist(), y=dep_vol.tolist(),
            marker_color=colors[idx % len(colors)],
            opacity=0.9,
            legendgroup=label,
            hovertemplate="p=%{x:.1f}h vol=%{y:.0f}<extra>departure</extra>",
        ))
        fig_spill.add_trace(go.Bar(
            name=f"{label} — spillover",
            x=ph.tolist(), y=spill_vol.tolist(),
            marker_color=colors[idx % len(colors)],
            opacity=0.4,
            marker_line=dict(width=1, color=colors[idx % len(colors)]),
            legendgroup=label,
            hovertemplate="p=%{x:.1f}h vol=%{y:.0f}<extra>spillover</extra>",
        ))

    fig_spill.update_layout(
        template="plotly_white", height=450,
        barmode="stack",
        xaxis_title="Hour of day",
        yaxis_title="Total link-volume (veh·links)",
        xaxis=dict(dtick=1, range=[0, 24]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    figs.append(fig_spill)

    # Compute spillover fractions for description
    spill_fracs = {}
    for label, sd in spillover_data.items():
        pvol = sd["period_volumes"]
        dep_p = sd["dep_period"]
        total = pvol.sum()
        if total > 0:
            spill_fracs[label] = 1.0 - pvol[dep_p] / total
        else:
            spill_fracs[label] = 0.0
    spill_desc = ", ".join(f"{k}: {v:.0%}" for k, v in spill_fracs.items())
    descriptions.append(
        "<h2>Cross-Period Flow Spillover</h2>"
        "<p>When 20k trips depart in a single period, their routes may "
        "traverse links in subsequent periods (spillover). Solid bars show "
        "volume attributed to the departure period; translucent bars show "
        "volume spilling into neighboring periods. "
        f"Spillover fractions: {spill_desc}.</p>"
    )

    # ── 6. Summary table ──────────────────────────────────────────
    rows_html = ""
    for label, r in routing_results.items():
        rows_html += (
            f"<tr><td>{label}</td>"
            f"<td style='text-align:right'>{r['routed']}</td>"
            f"<td style='text-align:right'>{r['elapsed_s']:.2f}s</td>"
            f"<td style='text-align:right'>{r['throughput']:.0f}</td>"
            f"<td style='text-align:right'>{r['mean_duration_s']:.0f}s</td></tr>"
        )

    figs.append(None)
    descriptions.append(
        "<h2>Summary</h2>"
        "<table style='border-collapse:collapse; width:100%;'>"
        "<thead><tr style='border-bottom:2px solid #1565C0;'>"
        "<th style='text-align:left; padding:6px;'>Period</th>"
        "<th style='text-align:right; padding:6px;'>Routed</th>"
        "<th style='text-align:right; padding:6px;'>Time</th>"
        "<th style='text-align:right; padding:6px;'>Routes/s</th>"
        "<th style='text-align:right; padding:6px;'>Mean dur</th>"
        "</tr></thead><tbody>"
        f"{rows_html}"
        "</tbody></table>"
        "<br>"
        f"<p><b>Network:</b> chi-regional (12,982 nodes, 39,018 links, "
        f"1,790 zones)</p>"
        f"<p><b>Periods:</b> {N_PERIODS} × {PERIOD_DURATION_S:.0f}s "
        f"(24-hour day at 15-min intervals)</p>"
        f"<p><b>customize_multi_period:</b> {cust_time:.1f}s</p>"
        f"<p><b>Engine load:</b> {load_time:.1f}s, RSS={mem_mb:.0f} MB</p>"
    )

    out = Path(output_path)
    _write_combined_report(
        title=f"Multi-Period Scale Test — chi-regional, {N_PERIODS} periods",
        intro=(
            "<p>Benchmarks the multi-period OSRM infrastructure at scale: "
            f"{N_PERIODS} periods (24-hour day at 15-min intervals) on the "
            "Chicago Regional network (12,982 nodes, 39,018 links). "
            "Measures <code>customize_multi_period</code> runtime, engine "
            "memory footprint, and routing throughput at varying congestion "
            "levels.</p>"
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
    args = parser.parse_args()

    work = Path(args.work_dir or "/tmp/scale_test_multi_period")
    work.mkdir(parents=True, exist_ok=True)
    logger.info("Working directory: %s", work)
    logger.info("Cores available: %d", os.cpu_count() or 1)
    logger.info("Initial RSS: %.0f MB", _get_mem_mb())

    # 1. Build network
    base, meta = build_network(work)

    # 2. Generate period CSVs
    csv_paths, congestion_factors = generate_period_csvs(base, meta, work)

    # 3. Benchmark customize_multi_period
    cust_time = benchmark_customize(base, csv_paths)

    # 4. Benchmark engine load
    engine, load_time, mem_mb = benchmark_engine_load(base)

    # 5. Benchmark routing
    routing_results, tt_ratio, spillover = benchmark_routing(
        engine, meta, PERIOD_DURATION_S,
    )
    del engine

    # 6. Generate report
    report_path = generate_report(
        congestion_factors, cust_time, load_time, mem_mb,
        routing_results, tt_ratio, spillover,
        output_path=args.output,
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SCALE TEST SUMMARY: chi-regional, %d periods", N_PERIODS)
    logger.info("=" * 60)
    logger.info("Network: %d zones, %d links",
                len(meta["zone_centroids"]), meta["od_matrix"].shape[0])
    logger.info("customize_multi_period: %.1fs", cust_time)
    logger.info("Engine load: %.1fs, RSS: %.0f MB", load_time, mem_mb)
    for label, r in routing_results.items():
        logger.info("Route %-20s: %.0f routes/s, mean_dur=%.0fs",
                     label, r["throughput"], r["mean_duration_s"])
    logger.info("Report: %s", report_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
