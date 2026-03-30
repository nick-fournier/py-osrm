"""Synthesize minimal OSM XML files for test networks.

Generates valid .osm files that can be processed by OSRM extract.
Used for structural validation on networks with known theoretical properties.

Provides a generic ``tntp_to_osm()`` pipeline that maps TNTP network
attributes (speed, capacity) to OSM road classification, plus per-network
wrappers like ``sioux_falls_network()`` that inject geographic overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from xml.etree.ElementTree import Element, SubElement, ElementTree

import numpy as np


# ---------------------------------------------------------------------------
# Road classification from TNTP attributes
# ---------------------------------------------------------------------------

@dataclass
class LinkClass:
    """Road classification for a single TNTP link."""

    highway: str
    n_lanes: int
    speed_kmh: float


def classify_by_speed(speed_kmh: float, capacity: float, *,
                      per_lane_capacity: float = 1800.0) -> LinkClass:
    """Classify a TNTP link using speed and capacity.

    Speed determines highway tier; capacity determines lane count within
    that tier.  Works well for networks where the TNTP speed column
    contains real freeflow speeds (e.g. Anaheim).

    Parameters
    ----------
    speed_kmh : float
        Freeflow speed in km/h (derived from TNTP ``speed`` column or
        computed from distance / free-flow time).
    capacity : float
        TNTP link capacity in veh/h.
    per_lane_capacity : float
        Assumed per-lane capacity for lane estimation (default 1800 vph).
    """
    n_lanes = max(1, round(capacity / per_lane_capacity))

    if speed_kmh >= 100:
        highway = "motorway"
    elif speed_kmh >= 80:
        highway = "trunk"
    elif speed_kmh >= 60:
        highway = "primary"
    elif speed_kmh >= 40:
        highway = "secondary"
    else:
        highway = "tertiary"

    return LinkClass(highway=highway, n_lanes=n_lanes, speed_kmh=speed_kmh)


def write_osm(
    nodes: Dict[int, Tuple[float, float]],
    ways: List[Dict],
    path: str | Path,
) -> Path:
    """Write a minimal OSM XML file.

    Parameters
    ----------
    nodes : dict
        OSM node ID → (lon, lat).
    ways : list of dict
        Each dict has keys:
        - "id": int — OSM way ID
        - "nodes": list[int] — ordered node IDs
        - "tags": dict[str, str] — OSM tags (highway, maxspeed, oneway, lanes, name)
    path : str or Path
        Output file path.

    Returns
    -------
    Path to written file.
    """
    path = Path(path)

    root = Element("osm", version="0.6", generator="py-osrm-test")

    # Bounds (OSRM needs this)
    lons = [lon for lon, _ in nodes.values()]
    lats = [lat for _, lat in nodes.values()]
    SubElement(root, "bounds", {
        "minlat": str(min(lats) - 0.001),
        "minlon": str(min(lons) - 0.001),
        "maxlat": str(max(lats) + 0.001),
        "maxlon": str(max(lons) + 0.001),
    })

    for nid, (lon, lat) in nodes.items():
        SubElement(root, "node", id=str(nid), lon=str(lon), lat=str(lat), version="1")

    for way in ways:
        w = SubElement(root, "way", id=str(way["id"]), version="1")
        for nid in way["nodes"]:
            SubElement(w, "nd", ref=str(nid))
        for k, v in way.get("tags", {}).items():
            SubElement(w, "tag", k=k, v=v)

    tree = ElementTree(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(path), xml_declaration=True, encoding="UTF-8")
    return path


# ---------------------------------------------------------------------------
# Generic TNTP → OSM pipeline
# ---------------------------------------------------------------------------

#: Type for a per-link classification override callback.
#: ``(link, distance_m, default_class) -> LinkClass | None``.
#: Return a ``LinkClass`` to override, or ``None`` to keep the default.
ClassifyOverride = Callable[
    ["TNTPLink", float, LinkClass],
    Optional[LinkClass],
]


def tntp_to_osm(
    net: "TNTPNetwork",
    node_coords: Dict[int, Tuple[float, float]],
    od_matrix: np.ndarray,
    path: str | Path,
    *,
    ref_flows: Optional[List] = None,
    osrm_speed_factor: float = 0.8,
    jam_density_per_lane: float = 200.0,
    per_lane_capacity: float = 1800.0,
    speed_units: str = "auto",
    classify_override: Optional[ClassifyOverride] = None,
) -> Tuple[Path, Dict]:
    """Convert a TNTP network to OSM XML using attribute-based classification.

    Infers highway type, lane count, and freeflow speed from each link's
    TNTP ``speed`` and ``capacity`` columns via :func:`classify_by_speed`.
    An optional *classify_override* callback can modify any link's
    classification for networks requiring geographic knowledge (e.g. Sioux
    Falls interstate corridors).

    Parameters
    ----------
    net : TNTPNetwork
        Parsed TNTP network.
    node_coords : dict
        ``{node_id: (lon, lat)}`` in WGS84.
    od_matrix : np.ndarray
        OD demand matrix of shape ``(n_zones, n_zones)``.
    path : str or Path
        Output ``.osm`` file path.
    ref_flows : list of TNTPFlowEntry, optional
        Published equilibrium flows for validation.
    osrm_speed_factor : float
        OSRM car profile speed factor (default 0.8).  ``maxspeed`` tags
        are inflated by ``1/factor`` so OSRM's profile-reduced speed
        matches the intended freeflow speed.
    jam_density_per_lane : float
        Default jam density per lane in veh/km (default 200).
    per_lane_capacity : float
        Assumed per-lane capacity for lane estimation (default 1800 vph).
    speed_units : str
        How to interpret the TNTP ``speed`` column:
        ``"ft/min"`` (Anaheim), ``"km/h"``, or ``"auto"`` (detect from
        magnitude: values > 500 are assumed ft/min).
    classify_override : callable, optional
        ``(link, distance_m, default_class) -> LinkClass | None``.
        Called per link after default classification.  Return a
        ``LinkClass`` to override, or ``None`` to keep the default.

    Returns
    -------
    (osm_path, metadata)
        ``metadata`` contains ``nodes``, ``zone_centroids``, ``n_zones``,
        ``n_links``, ``od_matrix``, ``lane_map``, ``link_attrs``,
        ``ref_flows``, ``jam_density_per_lane``.
    """
    from osrm.assignment.tntp import haversine_m

    ways = []
    lane_map: Dict[Tuple[int, int], int] = {}
    link_attrs: Dict[Tuple[int, int], Dict] = {}

    for i, link in enumerate(net.links):
        if link.init_node not in node_coords or link.term_node not in node_coords:
            continue
        lon1, lat1 = node_coords[link.init_node]
        lon2, lat2 = node_coords[link.term_node]
        dist_m = haversine_m(lon1, lat1, lon2, lat2)

        # Derive freeflow speed
        speed_kmh = _tntp_speed_to_kmh(link, dist_m, speed_units)

        # Default classification from speed + capacity
        default_cls = classify_by_speed(speed_kmh, link.capacity,
                                        per_lane_capacity=per_lane_capacity)

        # Allow per-network overrides
        cls = default_cls
        if classify_override is not None:
            override = classify_override(link, dist_m, default_cls)
            if override is not None:
                cls = override

        # Compensate for OSRM car profile speed reduction
        maxspeed = max(10, round(cls.speed_kmh / osrm_speed_factor))

        key = (link.init_node, link.term_node)
        way_id = 1000 + i
        ways.append({
            "id": way_id,
            "nodes": [link.init_node, link.term_node],
            "tags": {
                "highway": cls.highway,
                "oneway": "yes",
                "maxspeed": str(maxspeed),
                "lanes": str(cls.n_lanes),
                "name": f"Link {link.init_node}-{link.term_node}",
            },
        })

        lane_map[key] = cls.n_lanes
        link_attrs[key] = {
            "capacity": link.capacity,
            "freeflow_time_min": link.free_flow_time,
            "distance_m": dist_m,
            "ff_speed_kmh": cls.speed_kmh,
            "n_lanes": cls.n_lanes,
            "b": link.b,
            "power": link.power,
        }

    osm_path = write_osm(node_coords, ways, path)

    n_zones = net.n_zones
    zone_centroids = {z: node_coords[z] for z in range(1, n_zones + 1)
                      if z in node_coords}

    ref_flow_map = {}
    if ref_flows:
        ref_flow_map = {
            (e.init_node, e.term_node): (e.volume, e.cost) for e in ref_flows
        }

    metadata = {
        "nodes": node_coords,
        "zone_centroids": zone_centroids,
        "n_zones": n_zones,
        "n_links": len(net.links),
        "od_matrix": od_matrix,
        "lane_map": lane_map,
        "link_attrs": link_attrs,
        "ref_flows": ref_flow_map,
        "jam_density_per_lane": jam_density_per_lane,
    }

    return osm_path, metadata


def _tntp_speed_to_kmh(link: "TNTPLink", dist_m: float,
                        speed_units: str) -> float:
    """Convert TNTP link speed to km/h.

    Falls back to distance/FFT when the TNTP speed column is zero or
    missing.
    """
    FT_PER_MIN_TO_KMH = 0.018288

    raw = link.speed
    if raw > 0:
        if speed_units == "ft/min" or (speed_units == "auto" and raw > 500):
            return raw * FT_PER_MIN_TO_KMH
        elif speed_units == "km/h":
            return raw
        else:
            # auto: small values assumed km/h
            return raw

    # Fallback: derive from distance and free-flow time
    if link.free_flow_time > 0 and dist_m > 0:
        return (dist_m / 1000.0) / (link.free_flow_time / 60.0)

    return 50.0  # last resort default


def patch_lanes(
    state,
    meta: dict,
    jam_density_per_lane: float | None = None,
) -> None:
    """Patch NetworkState with lane counts and jam density from metadata.

    Works for any network whose ``meta["lane_map"]`` maps
    ``(from_node, to_node) -> n_lanes``.
    """
    kj_lane = jam_density_per_lane or meta.get("jam_density_per_lane", 200.0)
    lane_map = meta["lane_map"]
    for i in range(state.n_edges):
        key = (int(state.edge_ids[i, 0]), int(state.edge_ids[i, 1]))
        if key in lane_map:
            lanes = lane_map[key]
            state.n_lanes[i] = lanes
            state.jam_density[i] = kj_lane * lanes


def braess_network(
    path: str | Path,
    with_shortcut: bool = True,
) -> Tuple[Path, Dict[str, any]]:
    """Generate the Braess paradox network as OSM XML.

    Network topology (diamond)::

        1 ──→ 3
        │     ↓ (shortcut, optional)
        ↓     ↓
        4 ──→ 2

    Designed for the bi-parabolic MFD to produce a clear Braess paradox.

    The classical paradox requires two types of links:

    - **Variable** (congestion-sensitive): 1→3 and 4→2.  1 lane, 80 km/h,
      ~30 km.  At demand D, travel time grows significantly vs D/2.
    - **Constant** (capacity-insensitive): 1→4 and 3→2.  4 lanes, 80 km/h,
      ~54 km (1.8× variable, via detour waypoints).  High capacity means
      travel time is nearly constant regardless of flow.
    - **Shortcut** 3→4: 1 lane, 80 km/h, ~3 km (10% of variable length).

    At D ≈ 2005 vph (94% of variable-link capacity at v_f ≈ 64 km/h):

    - Without shortcut: traffic splits 50/50.  Path cost ≈ 84.8 min.
    - With shortcut: all traffic uses 1→3→4→2.  Path cost ≈ 94.9 min.
    - Alternative (skip shortcut): ≈ 95.8 min > 94.9, so stable.
    - TSTT increases ~11.9%, margin ≈ 1.2 min against OSRM quantization.

    Origin = node 1, Destination = node 2.

    Parameters
    ----------
    path : str or Path
        Output .osm file path.
    with_shortcut : bool
        If True, include the 3→4 shortcut link.

    Returns
    -------
    (path, metadata) where metadata includes node coords and link info.
    """
    # 6 nodes: 4 main + 2 detour waypoints for constant links.
    # Variable links (1→3, 4→2): ~30 km direct.
    # Constant links (1→4, 3→2): ~54 km via detour waypoints (1.8× ratio).
    # Shortcut (3→4): ~3 km vertical (10% of variable length).
    # 30 km scale maximizes stability margin (~1.2 min) against OSRM
    # speed quantization while maintaining 11.9% paradox strength.
    nodes = {
        1: (7.1000, 43.7400),   # west (origin)
        3: (7.4725, 43.7535),   # center-north
        4: (7.4725, 43.7265),   # center-south
        2: (7.8451, 43.7400),   # east (destination)
        5: (7.2863, 43.5315),   # detour south (for 1→4 highway)
        6: (7.6588, 43.9485),   # detour north (for 3→2 highway)
    }

    ways = [
        # 1→3: variable, congestion-sensitive (1 lane, ~30 km)
        {
            "id": 101, "nodes": [1, 3],
            "tags": {
                "highway": "secondary", "oneway": "yes",
                "maxspeed": "80", "lanes": "1",
                "name": "Link 1-3 (variable)",
            },
        },
        # 1→5→4: constant cost highway (4 lanes, ~54 km via detour)
        {
            "id": 102, "nodes": [1, 5, 4],
            "tags": {
                "highway": "motorway", "oneway": "yes",
                "maxspeed": "80", "lanes": "4",
                "name": "Link 1-4 (highway)",
            },
        },
        # 3→6→2: constant cost highway (4 lanes, ~54 km via detour)
        {
            "id": 103, "nodes": [3, 6, 2],
            "tags": {
                "highway": "motorway", "oneway": "yes",
                "maxspeed": "80", "lanes": "4",
                "name": "Link 3-2 (highway)",
            },
        },
        # 4→2: variable, congestion-sensitive (1 lane, ~30 km)
        {
            "id": 104, "nodes": [4, 2],
            "tags": {
                "highway": "secondary", "oneway": "yes",
                "maxspeed": "80", "lanes": "1",
                "name": "Link 4-2 (variable)",
            },
        },
    ]

    if with_shortcut:
        # 3→4: shortcut — 10% of variable length (~3 km)
        ways.append({
            "id": 105, "nodes": [3, 4],
            "tags": {
                "highway": "secondary", "oneway": "yes",
                "maxspeed": "80", "lanes": "1",
                "name": "Shortcut 3-4",
            },
        })

    osm_path = write_osm(nodes, ways, path)

    # Build lane map keyed by (from_osm_id, to_osm_id) for each segment.
    # Multi-node ways produce multiple OSRM edges (one per consecutive
    # node pair), so we map every segment, not just first→last.
    lane_map = {}
    for way in ways:
        lanes = int(way["tags"].get("lanes", 1))
        way_nodes = way["nodes"]
        for j in range(len(way_nodes) - 1):
            lane_map[(way_nodes[j], way_nodes[j + 1])] = lanes

    metadata = {
        "origin": nodes[1],
        "destination": nodes[2],
        "nodes": nodes,
        "n_links": len(ways),
        "with_shortcut": with_shortcut,
        "variable_links": ["1→3", "4→2"],
        "constant_links": ["1→4", "3→2"],
        "shortcut_link": "3→4" if with_shortcut else None,
        "lane_map": lane_map,
    }

    return osm_path, metadata


def patch_braess_lanes(
    state,
    meta: dict,
    jam_density_per_lane: float = 200.0,
) -> None:
    """Patch NetworkState with correct lane counts for a Braess network.

    OSRM annotations preserve OSM node IDs, so we match edge_ids
    directly against the synthesis lane_map {(from, to): n_lanes}.
    """
    lane_map = meta["lane_map"]
    for i in range(state.n_edges):
        key = (int(state.edge_ids[i, 0]), int(state.edge_ids[i, 1]))
        if key in lane_map:
            lanes = lane_map[key]
            state.n_lanes[i] = lanes
            state.jam_density[i] = jam_density_per_lane * lanes


def sioux_falls_network(
    path: str | Path,
    fixture_dir: str | Path | None = None,
    osrm_speed_factor: float = 0.8,
    jam_density_per_lane: float = 200.0,
) -> Tuple[Path, Dict]:
    """Generate the Sioux Falls 24-node network as OSM XML.

    Reads TNTP fixture files and delegates to :func:`tntp_to_osm` with
    geographic overrides for I-29 and I-229 corridors.

    TNTP speed/FFT columns for Sioux Falls are arbitrary (README: "Link
    lengths are set equal to free flow travel times"), so classification
    is entirely from geographic knowledge rather than TNTP attributes.

    Parameters
    ----------
    path : str or Path
        Output .osm file path.
    fixture_dir : str or Path, optional
        Directory containing TNTP fixture files. Defaults to
        ``tests/fixtures/sioux_falls/`` relative to the repo root.
    osrm_speed_factor : float
        OSRM car profile speed reduction factor (default 0.8).
        maxspeed tags are inflated by 1/factor to compensate.

    Returns
    -------
    (osm_path, metadata) where metadata includes node coords, lane_map,
    zone centroids, OD matrix, and reference flows.
    """
    from osrm.assignment.tntp import (
        parse_net, parse_trips, load_node_coords, parse_flow,
    )

    if fixture_dir is None:
        fixture_dir = (
            Path(__file__).parent.parent.parent.parent
            / "tests" / "fixtures" / "sioux_falls"
        )
    fixture_dir = Path(fixture_dir)

    net = parse_net(fixture_dir / "SiouxFalls_net.tntp")
    n_zones, od_matrix = parse_trips(fixture_dir / "SiouxFalls_trips.tntp")
    node_coords = load_node_coords(fixture_dir / "SiouxFalls_node.tntp")
    ref_flows = parse_flow(fixture_dir / "SiouxFalls_flow.tntp")

    # Sioux Falls geographic overrides: I-29 and I-229 corridors.
    # TNTP attributes are meaningless for this network, so we override
    # every link with known road classification.
    i29_links = {
        (1, 3), (3, 1), (3, 12), (12, 3), (12, 13), (13, 12),
    }
    i229_links = {(7, 18), (18, 7)}

    def sf_override(link, dist_m, default_cls):
        key = (link.init_node, link.term_node)
        if key in i29_links:
            return LinkClass("motorway", 3, 105.0)
        elif key in i229_links:
            return LinkClass("motorway", 2, 105.0)
        else:
            return LinkClass("primary", 2, 65.0)

    return tntp_to_osm(
        net, node_coords, od_matrix, path,
        ref_flows=ref_flows,
        osrm_speed_factor=osrm_speed_factor,
        jam_density_per_lane=jam_density_per_lane,
        classify_override=sf_override,
    )


def patch_sioux_falls_lanes(
    state,
    meta: dict,
    jam_density_per_lane: float | None = None,
) -> None:
    """Patch NetworkState with lane counts for Sioux Falls network.

    Thin wrapper around :func:`patch_lanes` for backward compatibility.
    """
    patch_lanes(state, meta, jam_density_per_lane=jam_density_per_lane)
