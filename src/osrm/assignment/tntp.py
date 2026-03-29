"""TNTP (Transportation Network Test Problems) file parser.

Reads the standard TNTP format used by the TransportationNetworks
repository (github.com/bstabler/TransportationNetworks).

Supports:
  - Network files (*.net.tntp): link topology and attributes
  - Trip files (*.trips.tntp): OD demand matrices
  - Node files (*.node.tntp): node coordinates
  - Node GeoJSON files (*.geojson): node coordinates (Anaheim, etc.)
  - Flow files (*.flow.tntp): reference equilibrium solution
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class TNTPLink:
    """A single directed link from the TNTP network file."""

    init_node: int
    term_node: int
    capacity: float
    length: float
    free_flow_time: float
    b: float
    power: float
    speed: float
    toll: float
    link_type: int


@dataclass
class TNTPNetwork:
    """Parsed TNTP network."""

    n_zones: int
    n_nodes: int
    n_links: int
    first_thru_node: int
    links: List[TNTPLink]


@dataclass
class TNTPFlowEntry:
    """Reference equilibrium flow for a single link."""

    init_node: int
    term_node: int
    volume: float
    cost: float


def parse_net(path: str | Path) -> TNTPNetwork:
    """Parse a TNTP network file.

    Parameters
    ----------
    path : str or Path
        Path to the ``*.net.tntp`` file.

    Returns
    -------
    TNTPNetwork
        Parsed network with metadata and link list.
    """
    path = Path(path)
    text = path.read_text()

    n_zones = int(re.search(r"<NUMBER OF ZONES>\s*(\d+)", text).group(1))
    n_nodes = int(re.search(r"<NUMBER OF NODES>\s*(\d+)", text).group(1))
    n_links = int(re.search(r"<NUMBER OF LINKS>\s*(\d+)", text).group(1))
    first_thru = int(re.search(r"<FIRST THRU NODE>\s*(\d+)", text).group(1))

    links: List[TNTPLink] = []
    in_data = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("~") or not line:
            if "init_node" in line.lower():
                in_data = True
            continue
        if not in_data:
            if line.startswith("<END OF METADATA>"):
                in_data = True
            continue
        if line.startswith("~"):
            continue

        line = line.rstrip(";").strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            links.append(TNTPLink(
                init_node=int(parts[0]),
                term_node=int(parts[1]),
                capacity=float(parts[2]),
                length=float(parts[3]),
                free_flow_time=float(parts[4]),
                b=float(parts[5]),
                power=float(parts[6]),
                speed=float(parts[7]),
                toll=float(parts[8]),
                link_type=int(parts[9]),
            ))
        except (ValueError, IndexError):
            continue

    return TNTPNetwork(
        n_zones=n_zones,
        n_nodes=n_nodes,
        n_links=n_links,
        first_thru_node=first_thru,
        links=links,
    )


def parse_trips(path: str | Path) -> Tuple[int, np.ndarray]:
    """Parse a TNTP trips (OD demand) file.

    Parameters
    ----------
    path : str or Path
        Path to the ``*.trips.tntp`` file.

    Returns
    -------
    n_zones : int
        Number of zones.
    matrix : np.ndarray
        OD demand matrix of shape ``(n_zones, n_zones)``.
        Entry ``[i, j]`` is demand from zone ``i+1`` to zone ``j+1``.
    """
    path = Path(path)
    text = path.read_text()

    n_zones = int(re.search(r"<NUMBER OF ZONES>\s*(\d+)", text).group(1))
    matrix = np.zeros((n_zones, n_zones), dtype=np.float64)

    current_origin = None
    in_data = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("<END OF METADATA>"):
            in_data = True
            continue
        if not in_data:
            continue

        origin_match = re.match(r"Origin\s+(\d+)", line)
        if origin_match:
            current_origin = int(origin_match.group(1))
            continue

        if current_origin is None:
            continue

        # Parse "dest : flow ; dest : flow ; ..."
        pairs = re.findall(r"(\d+)\s*:\s*([\d.]+)", line)
        for dest_str, flow_str in pairs:
            dest = int(dest_str)
            flow = float(flow_str)
            if flow > 0:
                matrix[current_origin - 1, dest - 1] = flow

    return n_zones, matrix


def parse_nodes(path: str | Path) -> Dict[int, Tuple[float, float]]:
    """Parse a TNTP node coordinate file.

    Parameters
    ----------
    path : str or Path
        Path to the ``*.node.tntp`` file.

    Returns
    -------
    dict
        Mapping ``{node_id: (longitude, latitude)}``.
    """
    path = Path(path)
    nodes: Dict[int, Tuple[float, float]] = {}

    for line in path.read_text().splitlines():
        line = line.strip().rstrip(";").strip()
        if not line or line.lower().startswith("node"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            node_id = int(parts[0])
            x = float(parts[1])  # longitude
            y = float(parts[2])  # latitude
            nodes[node_id] = (x, y)
        except ValueError:
            continue

    return nodes


def parse_nodes_geojson(path: str | Path) -> Dict[int, Tuple[float, float]]:
    """Parse node coordinates from a GeoJSON FeatureCollection.

    Expects Point features with an ``"id"`` property (integer node ID)
    and ``[longitude, latitude]`` coordinates.

    Parameters
    ----------
    path : str or Path
        Path to a ``.geojson`` file.

    Returns
    -------
    dict
        Mapping ``{node_id: (longitude, latitude)}``.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    nodes: Dict[int, Tuple[float, float]] = {}
    for feat in data["features"]:
        nid = int(feat["properties"]["id"])
        lon, lat = feat["geometry"]["coordinates"][:2]
        nodes[nid] = (float(lon), float(lat))
    return nodes


def load_node_coords(path: str | Path) -> Dict[int, Tuple[float, float]]:
    """Load node coordinates, auto-detecting file format.

    Supported formats:
      - ``.geojson`` — GeoJSON FeatureCollection with Point features
      - ``.tntp`` / anything else — TNTP whitespace-delimited node file

    Parameters
    ----------
    path : str or Path
        Path to node coordinate file.

    Returns
    -------
    dict
        Mapping ``{node_id: (longitude, latitude)}``.
    """
    path = Path(path)
    if path.suffix.lower() == ".geojson":
        return parse_nodes_geojson(path)
    return parse_nodes(path)


def parse_flow(path: str | Path) -> List[TNTPFlowEntry]:
    """Parse a TNTP reference flow (equilibrium solution) file.

    Parameters
    ----------
    path : str or Path
        Path to the ``*.flow.tntp`` file.

    Returns
    -------
    list of TNTPFlowEntry
        Reference flows and costs for each link.
    """
    path = Path(path)
    entries: List[TNTPFlowEntry] = []

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.lower().startswith("from"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            entries.append(TNTPFlowEntry(
                init_node=int(parts[0]),
                term_node=int(parts[1]),
                volume=float(parts[2]),
                cost=float(parts[3]),
            ))
        except ValueError:
            continue

    return entries


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in meters between two WGS84 points."""
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return float(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
