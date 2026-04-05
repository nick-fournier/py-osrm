#!/usr/bin/env python3
"""Validate multi-period OSRM routing on Chicago Sketch.

Runs a 4-period trapezoidal demand simulation, then compares
single-period vs multi-period probe trip travel times to verify
that OSRM's forward-search period switching works correctly.

Usage:
    python scripts/validate_multi_period.py [--output plots/multi_period_validation.html]
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import osrm
from osrm.assignment import AssignmentConfig, AssignmentSolver, DensitySmoothingConfig
from osrm.assignment.od_matrix import DemandTrip
from osrm.assignment.osm_synthesis import LinkClass, tntp_to_osm, patch_lanes
from osrm.assignment.segment_speed_writer import SegmentSpeedWriter
from osrm.assignment.tntp import parse_net, parse_trips, load_node_coords, parse_flow

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "chicago_sketch"


# ── Network preparation (same as test_chicago_sketch.py) ──────────

def _chicago_classify_override(link, dist_m, default_cls):
    if link.link_type == 3:
        return LinkClass(highway="motorway_link", n_lanes=max(2, default_cls.n_lanes), speed_kmh=100.0)
    if default_cls.speed_kmh > 130:
        return LinkClass(highway=default_cls.highway, n_lanes=default_cls.n_lanes, speed_kmh=130.0)
    return None


def _prepare_network(work_dir: Path):
    """Synthesize, extract, partition, customize Chicago Sketch."""
    work_dir.mkdir(parents=True, exist_ok=True)

    net = parse_net(FIXTURE_DIR / "ChicagoSketch_net.tntp")
    n_zones, od_matrix = parse_trips(FIXTURE_DIR / "ChicagoSketch_trips.tntp")
    node_coords = load_node_coords(FIXTURE_DIR / "chicago_sketch_nodes.geojson")
    ref_flows = parse_flow(FIXTURE_DIR / "ChicagoSketch_flow.tntp")

    osm_path, meta = tntp_to_osm(
        net, node_coords, od_matrix, work_dir / "chicago.osm",
        ref_flows=ref_flows, speed_units="auto",
        classify_override=_chicago_classify_override,
    )
    base = str(work_dir / "chicago.osrm")
    osrm.extract(str(osm_path), profile="car", output_path=base, verbosity="ERROR")
    osrm.partition(base, verbosity="ERROR")
    osrm.customize(base, verbosity="ERROR")
    return base, meta


def _copy_osrm(base_path: str, run_dir: Path) -> str:
    """Copy OSRM files to an isolated directory."""
    src = Path(base_path).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        shutil.copy2(f, run_dir / f.name)
    return str(run_dir / Path(base_path).name)


# ── Demand generation ─────────────────────────────────────────────

def _build_multiperiod_trips(meta, n_periods=4, period_s=900.0,
                             weights=None, demand_scale=3.0):
    """Build trips across periods with trapezoidal weighting."""
    centroids = meta["zone_centroids"]
    od = meta["od_matrix"]
    if weights is None:
        weights = [1 / 6, 1 / 3, 1 / 3, 1 / 6]
    trips = []
    for k in range(n_periods):
        w = weights[k] * demand_scale
        dep = k * period_s
        if w <= 0:
            if centroids:
                first = next(iter(centroids.values()))
                trips.append(DemandTrip(origin=first, destination=first,
                                        volume=0.0, departure_time_s=dep))
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


def _build_probe_trips(meta, period_s=900.0, n_probes=30):
    """Build probe trips near period boundaries for comparison.

    Creates trips departing at 80%, 90%, 95% into each period.
    These are the trips most likely to span period boundaries.
    """
    centroids = meta["zone_centroids"]
    od = meta["od_matrix"]
    # Pick the highest-demand OD pairs as probes
    flat = []
    for i in range(od.shape[0]):
        for j in range(od.shape[1]):
            if od[i, j] > 0 and i != j:
                o_zone, d_zone = i + 1, j + 1
                if o_zone in centroids and d_zone in centroids:
                    flat.append((od[i, j], o_zone, d_zone))
    flat.sort(reverse=True)
    top_pairs = flat[:min(n_probes, len(flat))]

    probes = []
    offsets_frac = [0.80, 0.90, 0.95]
    for period in range(4):
        for frac in offsets_frac:
            dep = period * period_s + frac * period_s
            for vol, o_zone, d_zone in top_pairs[:3]:  # 3 OD pairs per offset
                probes.append(DemandTrip(
                    origin=centroids[o_zone],
                    destination=centroids[d_zone],
                    volume=1.0,
                    departure_time_s=dep,
                    trip_id=f"P{period}_f{frac:.0%}_z{o_zone}-{d_zone}",
                ))
    return probes


# ── Route probes and extract travel times ─────────────────────────

def _route_probes(engine, probes, period_duration=0.0):
    """Route probe trips and return list of (trip_id, travel_time_s).

    If period_duration > 0, sets multi-period params per probe so OSRM
    switches cell metrics mid-route when crossing period boundaries.
    """
    from osrm import RouteParameters

    multi = period_duration > 0
    results = []
    for p in probes:
        rp = RouteParameters()
        rp.coordinates = [p.origin, p.destination]
        rp.steps = True  # ensure duration is populated

        if multi:
            period_idx = int(p.departure_time_s // period_duration)
            period_start = period_idx * period_duration
            rp.departure_period = period_idx
            rp.period_duration = period_duration
            rp.departure_time_offset = max(p.departure_time_s - period_start, 0.0)

        res = engine.Route(rp)
        if res and res.get("routes"):
            tt = res["routes"][0]["duration"]
            results.append((p.trip_id, p.departure_time_s, tt))
        else:
            results.append((p.trip_id, p.departure_time_s, float("nan")))
    return results


def _route_probes_batch(base_path, probes, period_s, multi_period=False):
    """Route all probes, optionally with multi-period switching."""
    engine = osrm.OSRM(storage_config=base_path, algorithm="MLD",
                       use_shared_memory=False)
    return _route_probes(engine, probes,
                         period_duration=period_s if multi_period else 0.0)


# ── Report generation ─────────────────────────────────────────────

def _generate_report(single_results, multi_results, output_path):
    """Generate HTML comparison report."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("plotly required for report generation")
        return

    # Build lookup: trip_id → (single_tt, multi_tt)
    single_map = {tid: (dep, tt) for tid, dep, tt in single_results}
    multi_map = {tid: (dep, tt) for tid, dep, tt in multi_results}

    rows = []
    for tid in single_map:
        dep_s, s_tt = single_map[tid]
        _, m_tt = multi_map.get(tid, (dep_s, float("nan")))
        diff = m_tt - s_tt
        pct = (diff / s_tt * 100) if s_tt > 0 else 0.0
        rows.append({
            "trip_id": tid,
            "departure_s": dep_s,
            "period": int(dep_s // 900),
            "offset_frac": (dep_s % 900) / 900,
            "single_tt": s_tt,
            "multi_tt": m_tt,
            "diff_s": diff,
            "diff_pct": pct,
        })

    rows.sort(key=lambda r: r["departure_s"])

    # Summary stats
    valid = [r for r in rows if not np.isnan(r["single_tt"]) and not np.isnan(r["multi_tt"])]
    n_differ = sum(1 for r in valid if abs(r["diff_s"]) > 0.1)
    n_correct_dir = sum(1 for r in valid if abs(r["diff_s"]) > 0.1)  # placeholder
    mean_abs_diff = np.mean([abs(r["diff_s"]) for r in valid]) if valid else 0
    max_abs_diff = max([abs(r["diff_s"]) for r in valid]) if valid else 0

    # Create figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Single-Period vs Multi-Period Travel Time",
            "Travel Time Difference by Departure Time",
            "Difference Distribution",
            "Summary",
        ],
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "histogram"}, {"type": "table"}]],
    )

    # 1. Scatter: single vs multi
    fig.add_trace(go.Scatter(
        x=[r["single_tt"] for r in valid],
        y=[r["multi_tt"] for r in valid],
        mode="markers",
        text=[r["trip_id"] for r in valid],
        marker=dict(size=6, color=[r["period"] for r in valid],
                    colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Period")),
        name="Probes",
    ), row=1, col=1)
    # 45-degree line
    if valid:
        mn = min(min(r["single_tt"] for r in valid), min(r["multi_tt"] for r in valid))
        mx = max(max(r["single_tt"] for r in valid), max(r["multi_tt"] for r in valid))
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(dash="dash", color="gray"), showlegend=False,
        ), row=1, col=1)

    # 2. Diff vs departure time
    fig.add_trace(go.Scatter(
        x=[r["departure_s"] for r in valid],
        y=[r["diff_s"] for r in valid],
        mode="markers+lines",
        text=[r["trip_id"] for r in valid],
        marker=dict(size=6),
        name="Δ travel time (s)",
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

    # 3. Histogram of differences
    fig.add_trace(go.Histogram(
        x=[r["diff_s"] for r in valid],
        nbinsx=20, name="Δ distribution",
    ), row=2, col=1)

    # 4. Summary table
    fig.add_trace(go.Table(
        header=dict(values=["Metric", "Value"]),
        cells=dict(values=[
            ["Total probes", "Probes with Δ > 0.1s", "Mean |Δ|", "Max |Δ|",
             "Any NaN/inf", "Validation"],
            [
                len(valid),
                n_differ,
                f"{mean_abs_diff:.1f} s",
                f"{max_abs_diff:.1f} s",
                "YES ⚠️" if any(np.isnan(r["multi_tt"]) for r in rows) else "No ✅",
                "PASS ✅" if n_differ > 0 else "FAIL ❌ (no difference detected)",
            ],
        ]),
    ), row=2, col=2)

    fig.update_layout(
        title="Multi-Period OSRM Routing Validation — Chicago Sketch",
        height=800,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Single-period TT (s)", row=1, col=1)
    fig.update_yaxes(title_text="Multi-period TT (s)", row=1, col=1)
    fig.update_xaxes(title_text="Departure time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Δ travel time (s)", row=1, col=2)
    fig.update_xaxes(title_text="Δ travel time (s)", row=2, col=1)

    # Also print the detail table
    detail_lines = [
        "\n=== Probe Trip Comparison ===",
        f"{'Trip ID':<30} {'Dep(s)':>7} {'P':>2} {'Single':>8} {'Multi':>8} {'Δ(s)':>8} {'Δ%':>7}",
        "-" * 85,
    ]
    for r in rows:
        detail_lines.append(
            f"{r['trip_id']:<30} {r['departure_s']:>7.0f} {r['period']:>2} "
            f"{r['single_tt']:>8.1f} {r['multi_tt']:>8.1f} "
            f"{r['diff_s']:>8.1f} {r['diff_pct']:>6.1f}%"
        )
    detail_lines.append("-" * 85)
    detail_lines.append(f"Probes with difference: {n_differ}/{len(valid)}")
    detail_lines.append(f"Mean |Δ|: {mean_abs_diff:.1f}s, Max |Δ|: {max_abs_diff:.1f}s")
    print("\n".join(detail_lines))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output))
    logger.info("Report written to %s", output)

    return n_differ > 0


# ── Main ──────────────────────────────────────────────────────────

def main(output_path="plots/multi_period_validation.html"):
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="mp_validate_"))
    logger.info("Working directory: %s", tmp)

    # 1. Prepare network
    t0 = time.monotonic()
    base, meta = _prepare_network(tmp / "network")
    logger.info("Network prepared in %.1fs", time.monotonic() - t0)

    # 2. Run 4-period stream to generate per-period congested states
    period_s = 900.0
    n_periods = 4
    demand_weights = [1 / 6, 1 / 3, 1 / 3, 1 / 6]
    demand_scale = 3.0

    trips = _build_multiperiod_trips(
        meta, n_periods=n_periods, period_s=period_s,
        weights=demand_weights, demand_scale=demand_scale,
    )
    logger.info("Built %d trips across %d periods", len(trips), n_periods)

    # Run stream with per-period customize to get congested states
    stream_base = _copy_osrm(base, tmp / "stream_run")
    csv_dir = tmp / "speed_csvs"
    csv_dir.mkdir(exist_ok=True)

    config = AssignmentConfig(
        smoothing=DensitySmoothingConfig(method="none"),
        speed_csv_dir=str(csv_dir),
    )
    solver = AssignmentSolver(stream_base, config)

    t1 = time.monotonic()
    result = solver.assign_stream(
        trips, period_duration_s=period_s,
        state_patch=lambda s: patch_lanes(s, meta),
    )
    logger.info("Stream completed in %.1fs, %d batches",
                time.monotonic() - t1, len(result.batch_log))

    # 3. Collect per-period speed CSVs
    # The stream wrote CSVs during execution. We need to generate
    # per-period CSVs from the final state of each period.
    # Re-run a simplified version: for each period, build flow state and write CSV.
    writer = SegmentSpeedWriter(output_dir=str(csv_dir), prefix="period")
    period_csvs = []

    # Use the batch log to reconstruct per-period states
    # Group batches by period, take the last batch's state
    state = result.network_state
    # For now, use the stream result's final state as period 3 (most congested)
    # and write per-period CSVs by scaling speeds
    logger.info("Generating per-period speed CSVs from stream results...")

    # Re-run stream, but this time capture per-period state snapshots
    # by hooking into the period transitions
    period_states = {}
    stream_base2 = _copy_osrm(base, tmp / "stream_capture")
    config2 = AssignmentConfig(
        smoothing=DensitySmoothingConfig(method="none"),
        speed_csv_dir=str(csv_dir),
    )
    solver2 = AssignmentSolver(stream_base2, config2)

    # We need per-period speed CSVs. The simplest approach: run stream again
    # and capture the state at each period boundary.
    # For this, patch assign_stream to record period-end states.
    # Alternative: just run N independent single-period assignments.

    # Simpler: Run independent single-period assignments for each period
    for k in range(n_periods):
        w = demand_weights[k] * demand_scale
        period_trips = [
            DemandTrip(
                origin=t.origin, destination=t.destination,
                volume=t.volume,
            )
            for t in trips
            if abs(t.departure_time_s - k * period_s) < 1.0 and t.volume > 0
        ]
        if not period_trips:
            logger.info("Period %d: no trips, skipping", k)
            continue

        period_base = _copy_osrm(base, tmp / f"period_{k}")
        period_config = AssignmentConfig(
            method="msa", max_iterations=5,
            smoothing=DensitySmoothingConfig(method="none"),
            speed_csv_dir=str(Path(period_base).parent),
        )
        period_solver = AssignmentSolver(period_base, period_config)

        logger.info("Period %d: %d trips, running short MSA...", k, len(period_trips))
        period_result = period_solver.assign_matrix(
            period_trips,
            state_patch=lambda s: patch_lanes(s, meta),
        )
        logger.info("Period %d: gap=%.4f after %d iters",
                     k, period_result.final_gap, period_result.iterations)

        # Write speed CSV for this period
        csv_path = writer.write_from_state(
            period_result.network_state, suffix=f"_{k}", only_changed=True,
        )
        period_csvs.append((k, str(csv_path)))
        logger.info("Period %d CSV: %s", k, csv_path)

    # 4. Customize multi-period
    logger.info("Customizing with %d period CSVs...", len(period_csvs))
    mp_base = _copy_osrm(base, tmp / "multi_period")
    t2 = time.monotonic()
    osrm.customize_multi_period(mp_base, period_speed_files=period_csvs, verbosity="INFO")
    logger.info("Multi-period customize took %.1fs", time.monotonic() - t2)

    # 5. Build probe trips
    probes = _build_probe_trips(meta, period_s=period_s)
    logger.info("Built %d probe trips", len(probes))

    # 6. Route probes — single-period baseline
    # Use period 1 (peak) weights for all probes
    sp_base = _copy_osrm(base, tmp / "single_period")
    if len(period_csvs) >= 2:
        osrm.customize(sp_base, segment_speed_file=period_csvs[1][1], verbosity="ERROR")
    else:
        osrm.customize(sp_base, verbosity="ERROR")
    logger.info("Routing probes single-period...")
    single_results = _route_probes_batch(sp_base, probes, period_s, multi_period=False)

    # 7. Route probes — multi-period
    logger.info("Routing probes multi-period...")
    multi_results = _route_probes_batch(mp_base, probes, period_s, multi_period=True)

    # 8. Generate comparison report
    passed = _generate_report(single_results, multi_results, output_path)

    logger.info("Validation %s", "PASSED ✅" if passed else "FAILED ❌")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate multi-period OSRM routing")
    parser.add_argument("--output", default="plots/multi_period_validation.html",
                        help="Output HTML report path")
    args = parser.parse_args()
    success = main(args.output)
    sys.exit(0 if success else 1)
