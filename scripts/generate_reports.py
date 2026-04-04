#!/usr/bin/env python
"""Generate traffic assignment validation reports.

Usage::

    uv run python scripts/generate_reports.py                          # all default scenes (msa)
    uv run python scripts/generate_reports.py chi-sketch               # MSA (default method)
    uv run python scripts/generate_reports.py chi-sketch --method fw
    uv run python scripts/generate_reports.py vdf braess               # multiple scenes
    uv run python scripts/generate_reports.py --method all             # all methods for defaults
    uv run python scripts/generate_reports.py --max-iter 30            # more iterations

Scenes:
    vdf, braess, method-comparison, sioux, anaheim, chi-sketch, chi-region

Methods:
    msa      MSA convergence (default)
    fw       Frank-Wolfe convergence
    all      Both msa and fw

Reports are written to plots/.
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


# ── scene callables ──────────────────────────────────────────────────
# Each callable accepts **kw with max_iter passed from CLI.

def _scene_vdf(**_kw):
    from osrm.assignment.plots import vdf_theory
    return vdf_theory()

def _scene_braess(**_kw):
    from tests.assignment.test_braess import generate_braess_report
    return generate_braess_report(tempfile.mkdtemp())

def _scene_method_comparison(*, max_iter=20, **_kw):
    from tests.assignment.test_method_comparison import generate_method_comparison_report
    return generate_method_comparison_report(tempfile.mkdtemp(), max_rounds=max_iter)

def _scene_sioux_msa(*, max_iter=20, **_kw):
    from tests.assignment.test_sioux_falls import generate_sioux_falls_report
    return generate_sioux_falls_report(tempfile.mkdtemp(), method="msa", max_rounds=max_iter)

def _scene_sioux_fw(*, max_iter=20, **_kw):
    from tests.assignment.test_sioux_falls import generate_sioux_falls_report
    return generate_sioux_falls_report(tempfile.mkdtemp(), method="fw", max_rounds=max_iter)

def _scene_anaheim_msa(*, max_iter=20, **_kw):
    from tests.assignment.test_anaheim import generate_anaheim_report
    return generate_anaheim_report(tempfile.mkdtemp(), method="msa", max_rounds=max_iter)

def _scene_anaheim_fw(*, max_iter=20, **_kw):
    from tests.assignment.test_anaheim import generate_anaheim_report
    return generate_anaheim_report(tempfile.mkdtemp(), method="fw", max_rounds=max_iter)

def _scene_chi_sketch_msa(*, max_iter=20, **_kw):
    from tests.assignment.test_chicago_sketch import generate_chicago_report
    return generate_chicago_report(tempfile.mkdtemp(), method="msa", max_rounds=max_iter)

def _scene_chi_sketch_fw(*, max_iter=20, **_kw):
    from tests.assignment.test_chicago_sketch import generate_chicago_report
    return generate_chicago_report(tempfile.mkdtemp(), method="fw", max_rounds=max_iter)

def _scene_chi_sketch_stream(*, max_iter=20, **_kw):
    from tests.assignment.test_chicago_sketch import generate_chicago_stream_report
    return generate_chicago_stream_report(tempfile.mkdtemp())

def _scene_chi_sketch_spillover(*, max_iter=20, **_kw):
    from tests.assignment.test_chicago_sketch import generate_chicago_spillover_report
    return generate_chicago_spillover_report(tempfile.mkdtemp())

def _scene_chi_region_msa(*, max_iter=20, **_kw):
    from tests.assignment.test_chicago_regional import generate_regional_report
    return generate_regional_report(tempfile.mkdtemp(), method="msa", max_rounds=max_iter)

def _scene_chi_region_fw(*, max_iter=20, **_kw):
    from tests.assignment.test_chicago_regional import generate_regional_report
    return generate_regional_report(tempfile.mkdtemp(), method="fw", max_rounds=max_iter)

def _scene_chi_region_stream(*, max_iter=20, **_kw):
    from tests.assignment.test_chicago_regional import generate_regional_stream_report
    return generate_regional_stream_report(tempfile.mkdtemp())


# ── scene registry ────────────────────────────────────────────────────

SCENES = {
    "vdf": {
        "msa": ("VDF Theory", _scene_vdf),
    },
    "braess": {
        "msa": ("Braess Paradox Validation", _scene_braess),
    },
    "method-comparison": {
        "msa": ("MSA vs FW Method Comparison", _scene_method_comparison),
    },
    "sioux": {
        "msa": ("Sioux Falls MSA", _scene_sioux_msa),
        "fw":  ("Sioux Falls FW", _scene_sioux_fw),
    },
    "anaheim": {
        "msa": ("Anaheim MSA", _scene_anaheim_msa),
        "fw":  ("Anaheim FW", _scene_anaheim_fw),
    },
    "chi-sketch": {
        "msa": ("Chicago Sketch MSA", _scene_chi_sketch_msa),
        "fw":  ("Chicago Sketch FW", _scene_chi_sketch_fw),
        "stream": ("Chicago Sketch Stream", _scene_chi_sketch_stream),
        "spillover": ("Chicago Sketch Spillover", _scene_chi_sketch_spillover),
    },
    "chi-region": {
        "msa": ("Chicago Regional MSA", _scene_chi_region_msa),
        "fw":  ("Chicago Regional FW", _scene_chi_region_fw),
        "stream": ("Chicago Regional Stream", _scene_chi_region_stream),
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
        "--method", default="msa", choices=["msa", "fw", "stream", "spillover", "all"],
        help="Assignment method (default: msa)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=20,
        help="Maximum convergence iterations for MSA or FW (default: 20)",
    )
    args = parser.parse_args()

    for scene in args.scenes:
        if scene not in SCENES:
            parser.error(
                f"Unknown scene: {scene!r}. Choose from: {', '.join(SCENES)}"
            )

    methods = ["msa", "fw", "stream", "spillover"] if args.method == "all" else [args.method]

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
            logger.info("Generating %s (%d iters)...", label, args.max_iter)
            t0 = time.perf_counter()
            path = fn(max_iter=args.max_iter)
            elapsed = time.perf_counter() - t0
            logger.info(
                "  → %s (%d KB, %.1fs)",
                path, path.stat().st_size / 1024, elapsed,
            )

    logger.info("Done.")


if __name__ == "__main__":
    main()
