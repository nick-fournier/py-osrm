#!/usr/bin/env python
"""Generate all traffic assignment validation reports.

Usage::

    uv run python scripts/generate_reports.py              # all default reports
    uv run python scripts/generate_reports.py vdf           # VDF theory only
    uv run python scripts/generate_reports.py braess        # Braess unified validation
    uv run python scripts/generate_reports.py monaco        # Monaco hill-climber validation
    uv run python scripts/generate_reports.py sioux         # Sioux Falls hill-climber validation
    uv run python scripts/generate_reports.py anaheim       # Anaheim hill-climber validation
    uv run python scripts/generate_reports.py chi-sketch    # Chicago Sketch hill-climber validation
    uv run python scripts/generate_reports.py slice-sweep   # Cross-network slice sweep
    uv run python scripts/generate_reports.py sioux-matrix  # Sioux Falls matrix validation
    uv run python scripts/generate_reports.py anaheim-matrix   # Anaheim matrix validation
    uv run python scripts/generate_reports.py chi-sketch-matrix # Chicago Sketch matrix validation

Reports are written to docs/plots/.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import time
from pathlib import Path

# Ensure repo root is on sys.path so `tests.*` imports work
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def generate_vdf() -> Path:
    """Generate VDF theory validation report."""
    from osrm.assignment.plots import vdf_theory

    return vdf_theory()


def generate_braess() -> Path:
    """Generate unified Braess paradox validation report."""
    from tests.assignment.test_braess import generate_braess_report

    return generate_braess_report(tempfile.mkdtemp())


def generate_sioux_falls_matrix() -> Path:
    """Generate Sioux Falls matrix validation report."""
    from tests.assignment.test_sioux_falls import generate_sioux_falls_report

    return generate_sioux_falls_report(tempfile.mkdtemp())


def generate_anaheim_matrix() -> Path:
    """Generate Anaheim matrix validation report."""
    from tests.assignment.test_anaheim import generate_anaheim_report

    return generate_anaheim_report(tempfile.mkdtemp())


def generate_chicago_matrix() -> Path:
    """Generate Chicago Sketch matrix validation report."""
    from tests.assignment.test_chicago_sketch import generate_chicago_report

    return generate_chicago_report(tempfile.mkdtemp())


def generate_monaco() -> Path:
    """Generate Monaco hill-climber validation report."""
    from tests.assignment.test_assignment_loop import generate_monaco_hillclimber_report

    return generate_monaco_hillclimber_report(tempfile.mkdtemp())


def generate_sioux_falls() -> Path:
    """Generate Sioux Falls hill-climber validation report."""
    from tests.assignment.test_sioux_falls import generate_sioux_falls_hillclimber_report

    return generate_sioux_falls_hillclimber_report(tempfile.mkdtemp())


def generate_anaheim() -> Path:
    """Generate Anaheim hill-climber validation report."""
    from tests.assignment.test_anaheim import generate_anaheim_hillclimber_report

    return generate_anaheim_hillclimber_report(tempfile.mkdtemp())


def generate_chicago() -> Path:
    """Generate Chicago Sketch hill-climber validation report."""
    from tests.assignment.test_chicago_sketch import generate_chicago_hillclimber_report

    return generate_chicago_hillclimber_report(tempfile.mkdtemp())


def generate_slice_sweep() -> Path:
    """Generate cross-network slice convergence sweep report."""
    from tests.assignment.test_slice_sweep import generate_slice_sweep_report

    return generate_slice_sweep_report(tempfile.mkdtemp())


REPORTS = {
    "vdf": ("VDF Theory", generate_vdf),
    "braess": ("Braess Paradox Validation", generate_braess),
    "monaco": ("Monaco Hill-Climber Validation", generate_monaco),
    "sioux": ("Sioux Falls Hill-Climber Validation", generate_sioux_falls),
    "anaheim": ("Anaheim Hill-Climber Validation", generate_anaheim),
    "chi-sketch": ("Chicago Sketch Hill-Climber Validation", generate_chicago),
    "slice-sweep": ("Slice Convergence Sweep", generate_slice_sweep),
    "sioux-matrix": ("Sioux Falls Matrix Validation", generate_sioux_falls_matrix),
    "anaheim-matrix": ("Anaheim Matrix Validation", generate_anaheim_matrix),
    "chi-sketch-matrix": ("Chicago Sketch Matrix Validation", generate_chicago_matrix),
}

DEFAULT_REPORTS = ["vdf", "braess", "monaco", "sioux", "anaheim", "chi-sketch"]


def main() -> None:
    targets = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_REPORTS

    for key in targets:
        if key not in REPORTS:
            print(f"Unknown report: {key!r}. Choose from: {list(REPORTS.keys())}")
            sys.exit(1)

    # Enable assignment loop progress logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    for key in targets:
        label, fn = REPORTS[key]
        print(f"Generating {label}...", flush=True)
        t0 = time.perf_counter()
        path = fn()
        elapsed = time.perf_counter() - t0
        print(f"  → {path} ({path.stat().st_size / 1024:.0f} KB, {elapsed:.1f}s)")

    print("Done.")


if __name__ == "__main__":
    main()
