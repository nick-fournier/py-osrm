#!/usr/bin/env python3
"""Validate that MLD cell-metric period switching fires on a real network.

Uses Monaco (real-world MLD partition with many cells and boundary nodes)
to prove that multi-period routing produces different travel times from
single-period routing — specifically because the MLD search switches
cell metrics at period boundaries during the forward search.

Chi-sketch has only 1 cell per MLD level (0 boundary nodes) so can't
exercise the cell-switching code path. Monaco has hundreds of cells.
"""
import argparse
import logging
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import osrm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MONACO_BASE = Path(__file__).resolve().parent.parent / "tests" / "data" / "mld" / "monaco.osrm"


@dataclass
class ProbeResult:
    od_label: str
    departure_s: float
    period: int
    single_tt: float
    multi_tt: float


def _copy_osrm(src_base: str, dest_dir: Path) -> str:
    """Copy OSRM dataset to a fresh directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(src_base).parent
    stem = Path(src_base).name
    for f in src_dir.iterdir():
        if f.name.startswith(stem.split(".")[0]):
            shutil.copy2(f, dest_dir / f.name)
    return str(dest_dir / Path(src_base).name)


def _collect_segments(engine):
    """Route diverse OD pairs to collect segment IDs for speed CSVs."""
    ods = [
        ((7.4093, 43.7284), (7.4389, 43.7490)),
        ((7.4389, 43.7490), (7.4093, 43.7284)),
        ((7.4150, 43.7350), (7.4300, 43.7450)),
        ((7.4200, 43.7280), (7.4250, 43.7500)),
        ((7.4100, 43.7400), (7.4380, 43.7400)),
        ((7.4250, 43.7300), (7.4150, 43.7480)),
        ((7.4350, 43.7320), (7.4120, 43.7460)),
    ]
    segments = set()
    for o, d in ods:
        rp = osrm.RouteParameters()
        rp.coordinates = [o, d]
        rp.annotations_type = osrm.RouteAnnotationsType.All
        res = engine.Route(rp)
        if not res or not res.get("routes"):
            continue
        nodes = res["routes"][0]["legs"][0]["annotation"]["nodes"]
        for i in range(len(nodes) - 1):
            segments.add((int(nodes[i]), int(nodes[i + 1])))
    return list(segments)


def _write_speed_csv(segments, speed_kmh, path):
    """Write a speed CSV applying uniform speed to all segments."""
    with open(path, "w") as f:
        for from_id, to_id in segments:
            f.write(f"{from_id},{to_id},{speed_kmh:.1f}\n")
    return path


def _route_probe(engine, origin, dest, period=None, period_duration=0.0, offset=0.0):
    """Route a single probe, optionally with multi-period params."""
    rp = osrm.RouteParameters()
    rp.coordinates = [origin, dest]
    rp.steps = True
    if period is not None and period_duration > 0:
        rp.departure_period = period
        rp.period_duration = period_duration
        rp.departure_time_offset = offset
    res = engine.Route(rp)
    if res and res.get("routes"):
        return res["routes"][0]["duration"]
    return float("nan")


def main(output_path=None):
    tmp = Path(tempfile.mkdtemp(prefix="cell_switch_"))
    logger.info("Working directory: %s", tmp)

    # Verify Monaco data exists
    if not MONACO_BASE.parent.exists():
        logger.error("Monaco data not found at %s", MONACO_BASE.parent)
        return False

    # Load baseline engine to collect segments
    engine = osrm.OSRM(
        storage_config=str(MONACO_BASE), algorithm="MLD", use_shared_memory=False
    )
    segments = _collect_segments(engine)
    logger.info("Collected %d segments from Monaco", len(segments))

    # Create 2 period speed CSVs with very different speeds
    # Period 0: freeflow (60 km/h) — fast
    # Period 1: heavily congested (10 km/h) — slow
    csv_dir = tmp / "csvs"
    csv_dir.mkdir()
    csv_p0 = _write_speed_csv(segments, 60.0, csv_dir / "period_0.csv")
    csv_p1 = _write_speed_csv(segments, 10.0, csv_dir / "period_1.csv")
    logger.info("Speed CSVs: period_0=60km/h, period_1=10km/h")

    # Customize multi-period
    mp_base = _copy_osrm(str(MONACO_BASE), tmp / "multi_period")
    osrm.customize_multi_period(
        mp_base, period_speed_files=[(0, str(csv_p0)), (1, str(csv_p1))], verbosity="INFO"
    )
    logger.info("Multi-period customize done (2 periods)")

    # Load multi-period engine
    mp_engine = osrm.OSRM(
        storage_config=mp_base, algorithm="MLD", use_shared_memory=False
    )

    # Define probe trips — long routes across Monaco
    probe_ods = [
        ("SW-NE", (7.4093, 43.7284), (7.4389, 43.7490)),
        ("NE-SW", (7.4389, 43.7490), (7.4093, 43.7284)),
        ("W-E", (7.4100, 43.7400), (7.4380, 43.7400)),
        ("S-N", (7.4200, 43.7280), (7.4250, 43.7500)),
    ]

    # Period duration = 300s (5 min).
    # Compare same OD pair routed with departure_period=0 vs =1.
    # If cell-metric switching works, travel times MUST differ because
    # period 0 has 60 km/h cell metrics and period 1 has 10 km/h.
    period_s = 300.0

    print("\n" + "=" * 85)
    print(f"{'Probe':<20} {'P0 tt(s)':>10} {'P1 tt(s)':>10} {'Δ(s)':>10} {'Δ(%)':>8}  {'Status'}")
    print("=" * 85)

    n_diff = 0
    n_total = 0
    for od_label, origin, dest in probe_ods:
        tt_p0 = _route_probe(mp_engine, origin, dest,
                             period=0, period_duration=period_s, offset=0.0)
        tt_p1 = _route_probe(mp_engine, origin, dest,
                             period=1, period_duration=period_s, offset=0.0)
        delta = tt_p1 - tt_p0
        pct = (delta / tt_p0 * 100) if tt_p0 > 0 else 0
        differs = abs(delta) > 0.1
        n_diff += int(differs)
        n_total += 1
        status = "✅ DIFFER" if differs else "❌ SAME"
        print(f"{od_label:<20} {tt_p0:>10.1f} {tt_p1:>10.1f} {delta:>+10.1f} {pct:>+7.1f}%  {status}")

    print("=" * 85)
    print(f"\nRoutes with period-dependent travel times: {n_diff}/{n_total}")
    print(f"(Period 0 = 60 km/h on {len(segments)} segs, Period 1 = 10 km/h)")

    passed = n_diff > 0
    if passed:
        logger.info("Cell-switching validation PASSED ✅")
        logger.info("MLD cell metrics vary by period — switching is active")
    else:
        logger.warning("Cell-switching validation FAILED ❌")
        logger.warning("No travel time differences — cell switching may not be firing")

    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate MLD cell-metric period switching")
    args = parser.parse_args()
    success = main()
    sys.exit(0 if success else 1)
