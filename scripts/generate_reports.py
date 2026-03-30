#!/usr/bin/env python
"""Generate all traffic assignment validation reports.

Usage::

    uv run python scripts/generate_reports.py          # all reports
    uv run python scripts/generate_reports.py vdf       # VDF theory only
    uv run python scripts/generate_reports.py anaheim   # Anaheim only
    uv run python scripts/generate_reports.py braess    # Braess only
    uv run python scripts/generate_reports.py sioux     # Sioux Falls only

Reports are written to docs/plots/.
"""

from __future__ import annotations

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
    """Generate Braess paradox validation report."""
    from tests.assignment.test_braess import generate_braess_report

    return generate_braess_report(tempfile.mkdtemp())


def generate_sioux_falls() -> Path:
    """Generate Sioux Falls validation report."""
    from tests.assignment.test_sioux_falls import generate_sioux_falls_report

    return generate_sioux_falls_report(tempfile.mkdtemp())


def generate_anaheim() -> Path:
    """Generate Anaheim validation report."""
    from tests.assignment.test_anaheim import generate_anaheim_report

    return generate_anaheim_report(tempfile.mkdtemp())


REPORTS = {
    "vdf": ("VDF Theory", generate_vdf),
    "braess": ("Braess Paradox", generate_braess),
    "sioux": ("Sioux Falls", generate_sioux_falls),
    "anaheim": ("Anaheim", generate_anaheim),
}


def main() -> None:
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(REPORTS.keys())

    for key in targets:
        if key not in REPORTS:
            print(f"Unknown report: {key!r}. Choose from: {list(REPORTS.keys())}")
            sys.exit(1)

    for key in targets:
        label, fn = REPORTS[key]
        print(f"Generating {label}...", end=" ", flush=True)
        t0 = time.perf_counter()
        path = fn()
        elapsed = time.perf_counter() - t0
        print(f"{path} ({path.stat().st_size / 1024:.0f} KB, {elapsed:.1f}s)")

    print("Done.")


if __name__ == "__main__":
    main()
