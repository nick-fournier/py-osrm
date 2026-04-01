#!/usr/bin/env python
"""Generate traffic assignment validation reports.

Usage::

    uv run python scripts/generate_reports.py                          # all default scenes (hc)
    uv run python scripts/generate_reports.py chi-sketch               # HC (default method)
    uv run python scripts/generate_reports.py chi-sketch --method matrix
    uv run python scripts/generate_reports.py chi-region --method matrix
    uv run python scripts/generate_reports.py vdf braess               # multiple scenes
    uv run python scripts/generate_reports.py --method all             # all methods for defaults

Scenes:
    vdf, braess, sioux, anaheim, chi-sketch, chi-region

Methods:
    hc       Hill-climber (default)
    matrix   Matrix-based Frank-Wolfe
    all      Both hc and matrix

Reports are written to docs/plots/.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import time
from pathlib import Path
import sys

# Ensure repo root is on sys.path so `tests.*` imports work
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ── scene registry ────────────────────────────────────────────────────
# Each scene maps to {method: (label, callable)}.
# Scenes that only support one method (vdf, braess) ignore --method.

def _scene_vdf():
    from osrm.assignment.plots import vdf_theory
    return vdf_theory()

def _scene_braess():
    from tests.assignment.test_braess import generate_braess_report
    return generate_braess_report(tempfile.mkdtemp())

def _scene_sioux_hc():
    from tests.assignment.test_sioux_falls import generate_sioux_falls_hillclimber_report
    return generate_sioux_falls_hillclimber_report(tempfile.mkdtemp())

def _scene_sioux_matrix():
    from tests.assignment.test_sioux_falls import generate_sioux_falls_report
    return generate_sioux_falls_report(tempfile.mkdtemp())

def _scene_anaheim_hc():
    from tests.assignment.test_anaheim import generate_anaheim_hillclimber_report
    return generate_anaheim_hillclimber_report(tempfile.mkdtemp())

def _scene_anaheim_matrix():
    from tests.assignment.test_anaheim import generate_anaheim_report
    return generate_anaheim_report(tempfile.mkdtemp())

def _scene_chi_sketch_hc():
    from tests.assignment.test_chicago_sketch import generate_chicago_hillclimber_report
    return generate_chicago_hillclimber_report(tempfile.mkdtemp())

def _scene_chi_sketch_matrix():
    from tests.assignment.test_chicago_sketch import generate_chicago_report
    return generate_chicago_report(tempfile.mkdtemp())

def _scene_chi_region_hc():
    from tests.assignment.test_chicago_regional import generate_regional_hillclimber_report
    return generate_regional_hillclimber_report(tempfile.mkdtemp())

def _scene_chi_region_matrix():
    from tests.assignment.test_chicago_regional import generate_regional_report
    return generate_regional_report(tempfile.mkdtemp())


SCENES = {
    "vdf": {
        "hc": ("VDF Theory", _scene_vdf),
    },
    "braess": {
        "hc": ("Braess Paradox Validation", _scene_braess),
    },
    "sioux": {
        "hc":     ("Sioux Falls Hill-Climber", _scene_sioux_hc),
        "matrix": ("Sioux Falls Matrix FW", _scene_sioux_matrix),
    },
    "anaheim": {
        "hc":     ("Anaheim Hill-Climber", _scene_anaheim_hc),
        "matrix": ("Anaheim Matrix FW", _scene_anaheim_matrix),
    },
    "chi-sketch": {
        "hc":     ("Chicago Sketch Hill-Climber", _scene_chi_sketch_hc),
        "matrix": ("Chicago Sketch Matrix FW", _scene_chi_sketch_matrix),
    },
    "chi-region": {
        "hc":     ("Chicago Regional Hill-Climber", _scene_chi_region_hc),
        "matrix": ("Chicago Regional Matrix FW", _scene_chi_region_matrix),
    },
}

DEFAULT_SCENES = ["vdf", "braess", "sioux", "anaheim", "chi-sketch"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate traffic assignment validation reports.",
    )
    parser.add_argument(
        "scenes", nargs="*", default=DEFAULT_SCENES,
        help=f"Scenes to generate (default: {' '.join(DEFAULT_SCENES)}). "
             f"Available: {', '.join(SCENES)}",
    )
    parser.add_argument(
        "--method", default="hc", choices=["hc", "matrix", "all"],
        help="Assignment method (default: hc)",
    )
    args = parser.parse_args()

    for scene in args.scenes:
        if scene not in SCENES:
            parser.error(
                f"Unknown scene: {scene!r}. Choose from: {', '.join(SCENES)}"
            )

    methods = ["hc", "matrix"] if args.method == "all" else [args.method]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger("generate_reports")

    for scene in args.scenes:
        for method in methods:
            if method not in SCENES[scene]:
                if args.method != "all":
                    logger.warning("%s has no '%s' method, skipping", scene, method)
                continue
            label, fn = SCENES[scene][method]
            logger.info("Generating %s...", label)
            t0 = time.perf_counter()
            path = fn()
            elapsed = time.perf_counter() - t0
            logger.info(
                "  → %s (%d KB, %.1fs)",
                path, path.stat().st_size / 1024, elapsed,
            )

    logger.info("Done.")


if __name__ == "__main__":
    main()
