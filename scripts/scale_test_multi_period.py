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
            highway="motorway_link", lanes=max(2, link.lanes or 2),
            maxspeed_kmh=min(link.free_flow_speed_kmh or 100, 100),
        )
    if link.free_flow_speed_kmh and link.free_flow_speed_kmh > 130:
        return LinkClass(
            highway=default_cls.highway, lanes=default_cls.lanes,
            maxspeed_kmh=130,
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
        rp.annotations_type = (
            osrm.osrm_ext.RouteAnnotationsType.Nodes
            | osrm.osrm_ext.RouteAnnotationsType.Speed
            | osrm.osrm_ext.RouteAnnotationsType.Distance
        )
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
    congestion_factors = np.ones(N_PERIODS)
    for p in range(N_PERIODS):
        hour = p * 0.25  # hours from midnight
        # AM peak
        am_factor = 0.5 * np.exp(-0.5 * ((hour - 8.0) / 1.0) ** 2)
        # PM peak
        pm_factor = 0.6 * np.exp(-0.5 * ((hour - 17.0) / 1.2) ** 2)
        # Midday baseline
        mid_factor = 0.15 if 9 <= hour <= 16 else 0.0
        congestion_factors[p] = max(0.0, am_factor + pm_factor + mid_factor)

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

    # Build 2000 OD pairs
    n_pairs = 2000
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

    # Test routing at different periods: off-peak, AM peak, PM peak
    test_periods = [
        ("off-peak (3am)", 12),    # period 12 = 3:00
        ("AM peak (8am)", 32),     # period 32 = 8:00
        ("midday (12pm)", 48),     # period 48 = 12:00
        ("PM peak (5pm)", 68),     # period 68 = 17:00
    ]

    for label, dep_period in test_periods:
        offsets = rng.uniform(0, period_duration_s, size=valid).astype(np.float64)

        t0 = time.monotonic()
        vol, tstt, new_edges, _, durations = batch_route_accumulate(
            engine._engine, coords, volumes, edge_ids,
            n_threads=0,  # all cores
            return_routes=False,
            departure_period=dep_period,
            period_duration=period_duration_s,
            departure_offsets=offsets,
            n_periods=0,
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

        logger.info(
            "  %s (p=%d): %d/%d routed in %.2fs (%.0f routes/s), "
            "mean_dur=%.1fs",
            label, dep_period, routed, valid, elapsed, throughput, mean_dur,
        )

    # Also test with 2D period attribution
    logger.info("  2D attribution benchmark (96 periods)...")
    offsets_2d = rng.uniform(0, period_duration_s, size=valid).astype(np.float64)
    t0 = time.monotonic()
    vol2d, tstt2d, _, _, _ = batch_route_accumulate(
        engine._engine, coords, volumes, edge_ids,
        n_threads=0,
        return_routes=False,
        departure_period=32,  # AM peak
        period_duration=period_duration_s,
        departure_offsets=offsets_2d,
        n_periods=N_PERIODS,
    )
    elapsed_2d = time.monotonic() - t0
    v2d = np.asarray(vol2d)
    nonzero_periods = np.sum(np.any(v2d > 0, axis=1))
    logger.info(
        "  2D (96 periods): %.2fs (%.0f routes/s), "
        "%d/%d periods have flow, shape=%s",
        elapsed_2d, valid / max(elapsed_2d, 0.001),
        nonzero_periods, N_PERIODS, v2d.shape,
    )

    return results


def generate_report(
    congestion_factors: np.ndarray,
    cust_time: float,
    load_time: float,
    mem_mb: float,
    routing_results: dict,
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
        "<p>Throughput (routes/s) and mean trip duration routing 2000 OD "
        "pairs at different times of day against a 96-period multi-period "
        "OSRM engine. Throughput should be stable regardless of period; "
        "duration varies with congestion level.</p>"
    )

    # ── 4. Summary table ──────────────────────────────────────────
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
    routing_results = benchmark_routing(engine, meta, PERIOD_DURATION_S)
    del engine

    # 6. Generate report
    report_path = generate_report(
        congestion_factors, cust_time, load_time, mem_mb,
        routing_results, output_path=args.output,
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
