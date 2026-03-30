"""Generate OSRM segment-speed CSV files from network state.

Writes speed overrides to tmpfs (/dev/shm/) for zero-disk-I/O ingestion
by OSRM's customize step.

See docs/pyosrm_assignment_module.md §1.6 for I/O strategy.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np


class SegmentSpeedWriter:
    """Writes segment-speed CSV files for OSRM customize.

    Format: ``from_osm_node_id,to_osm_node_id,speed_km_h``

    Parameters
    ----------
    output_dir : str or Path
        Directory for CSV files. Defaults to /dev/shm/ (tmpfs) if
        available, otherwise the system temp directory.
    prefix : str
        Filename prefix for generated CSVs.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        prefix: str = "osrm_assignment",
    ) -> None:
        if output_dir is None:
            output_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None
            if output_dir is None:
                import tempfile
                output_dir = tempfile.gettempdir()
        self.output_dir = Path(output_dir)
        self.prefix = prefix

    def write(
        self,
        edge_ids: np.ndarray,
        speeds_kmh: np.ndarray,
        suffix: str = "",
        min_speed_kmh: float = 1.0,
    ) -> Path:
        """Write a segment-speed CSV.

        Parameters
        ----------
        edge_ids : np.ndarray
            (N, 2) uint64 — (from_osm_id, to_osm_id).
        speeds_kmh : np.ndarray
            (N,) float64 — speed in km/h per edge.
        suffix : str
            Optional suffix for the filename (e.g. period index).
        min_speed_kmh : float
            Floor speed to avoid zero/negative values in CSV.

        Returns
        -------
        Path
            Path to the written CSV file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{self.prefix}{suffix}.csv"
        path = self.output_dir / fname

        speeds = np.clip(speeds_kmh, min_speed_kmh, None)

        with open(path, "w") as f:
            for i in range(len(edge_ids)):
                f.write(
                    f"{int(edge_ids[i, 0])},{int(edge_ids[i, 1])},{speeds[i]:.4f}\n"
                )

        return path

    def write_from_state(
        self,
        network_state,
        suffix: str = "",
        only_changed: bool = False,
        tolerance_kmh: float = 0.5,
    ) -> Path:
        """Write CSV from a NetworkState, optionally only changed edges.

        Parameters
        ----------
        network_state : NetworkState
            Network state with current speeds.
        suffix : str
            Optional filename suffix.
        only_changed : bool
            If True, only write edges where speed differs from freeflow
            by more than tolerance_kmh. Reduces CSV size.
        tolerance_kmh : float
            Speed change threshold for only_changed mode.

        Returns
        -------
        Path
            Path to the written CSV file.
        """
        if only_changed:
            delta = np.abs(network_state.speed_kmh - network_state.freeflow_kmh)
            mask = delta > tolerance_kmh
            edge_ids = network_state.edge_ids[mask]
            speeds = network_state.speed_kmh[mask]
        else:
            edge_ids = network_state.edge_ids
            speeds = network_state.speed_kmh

        return self.write(edge_ids, speeds, suffix=suffix)

    def cleanup(self, suffix: str = "") -> None:
        """Remove a previously written CSV file."""
        fname = f"{self.prefix}{suffix}.csv"
        path = self.output_dir / fname
        if path.exists():
            path.unlink()
