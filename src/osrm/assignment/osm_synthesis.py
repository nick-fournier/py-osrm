"""Synthesize minimal OSM XML files for test networks.

Generates valid .osm files that can be processed by OSRM extract.
Used for structural validation on networks with known theoretical properties.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree.ElementTree import Element, SubElement, ElementTree

import numpy as np


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


def braess_network(
    path: str | Path,
    with_shortcut: bool = True,
) -> Tuple[Path, Dict[str, any]]:
    """Generate the Braess paradox network as OSM XML.

    Network topology (diamond):

        1 ──→ 3
        │     ↓ (shortcut, optional)
        ↓     ↓
        4 ──→ 2

    - Links 1→3 and 4→2: narrow (1 lane), congestion-sensitive
    - Links 1→4 and 3→2: wide (4 lanes), effectively constant cost
    - Shortcut 3→4: ~2 km, fast (80 km/h), 1 lane (if enabled)

    Origin = node 1, Destination = node 2.

    Nodes form a diamond at Monaco coordinates.
    Arterials are ~4 km; the shortcut is ~2 km.

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
    # Diamond layout: node 1 (west), 3 (center-north), 4 (center-south), 2 (east)
    # Realistic urban scale: ~4 km arterials, ~0.6 km shortcut
    nodes = {
        1: (7.350, 43.735),    # west (origin)
        3: (7.400, 43.744),    # center-north
        4: (7.400, 43.726),    # center-south
        2: (7.450, 43.735),    # east (destination)
    }

    ways = [
        # 1→3: narrow, congestion-sensitive (fast freeflow, 1 lane)
        {
            "id": 101, "nodes": [1, 3],
            "tags": {
                "highway": "secondary", "oneway": "yes",
                "maxspeed": "60", "lanes": "1", "name": "Link 1-3 (narrow)",
            },
        },
        # 1→4: wide arterial, effectively constant cost (3 lanes, moderate speed)
        {
            "id": 102, "nodes": [1, 4],
            "tags": {
                "highway": "primary", "oneway": "yes",
                "maxspeed": "40", "lanes": "3", "name": "Link 1-4 (wide)",
            },
        },
        # 3→2: wide arterial, effectively constant cost
        {
            "id": 103, "nodes": [3, 2],
            "tags": {
                "highway": "primary", "oneway": "yes",
                "maxspeed": "40", "lanes": "3", "name": "Link 3-2 (wide)",
            },
        },
        # 4→2: narrow, congestion-sensitive
        {
            "id": 104, "nodes": [4, 2],
            "tags": {
                "highway": "secondary", "oneway": "yes",
                "maxspeed": "60", "lanes": "1", "name": "Link 4-2 (narrow)",
            },
        },
    ]

    if with_shortcut:
        # 3→4: shortcut connecting street — fast but 1 lane
        ways.append({
            "id": 105, "nodes": [3, 4],
            "tags": {
                "highway": "secondary", "oneway": "yes",
                "maxspeed": "80", "lanes": "1", "name": "Shortcut 3-4",
            },
        })

    osm_path = write_osm(nodes, ways, path)

    # Build direct lane map keyed by (from_osm_id, to_osm_id).
    # OSRM annotations preserve OSM node IDs, so this maps directly
    # to NetworkState edge_ids.
    lane_map = {}
    for way in ways:
        from_id, to_id = way["nodes"][0], way["nodes"][-1]
        lane_map[(from_id, to_id)] = int(way["tags"].get("lanes", 1))

    metadata = {
        "origin": nodes[1],
        "destination": nodes[2],
        "nodes": nodes,
        "n_links": len(ways),
        "with_shortcut": with_shortcut,
        "narrow_links": ["1→3", "4→2"],
        "wide_links": ["1→4", "3→2"] + (["3→4"] if with_shortcut else []),
        "lane_map": lane_map,
    }

    return osm_path, metadata


def patch_braess_lanes(
    state,
    meta: dict,
    jam_density_per_lane: float = 130.0,
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

    Reads TNTP fixture files, computes real-world link distances from
    GPS coordinates, derives freeflow speeds, and maps TNTP links to
    realistic lane counts.

    Lane counts are assigned by capacity tier (TNTP capacities are BPR
    math artifacts, not physical):
    - capacity >= 10000: 3 lanes (major corridors / interstate)
    - capacity < 10000:  2 lanes (arterials / collectors)

    Jam density defaults to 200 veh/km/lane (5 m bumper-to-bumper
    spacing), which gives MFD capacity of ~2000 vph/lane at 60 km/h.

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
        parse_net, parse_trips, parse_nodes, parse_flow, haversine_m,
    )

    if fixture_dir is None:
        fixture_dir = (
            Path(__file__).parent.parent.parent.parent
            / "tests" / "fixtures" / "sioux_falls"
        )
    fixture_dir = Path(fixture_dir)

    net = parse_net(fixture_dir / "SiouxFalls_net.tntp")
    n_zones, od_matrix = parse_trips(fixture_dir / "SiouxFalls_trips.tntp")
    node_coords = parse_nodes(fixture_dir / "SiouxFalls_node.tntp")
    ref_flows = parse_flow(fixture_dir / "SiouxFalls_flow.tntp")

    ways = []
    lane_map = {}
    link_attrs = {}

    # Road classification based on real Sioux Falls geography.
    # Nodes 1,3,12,13 lie along the I-29 corridor (west side).
    # Nodes 7,18 lie along I-229 (eastern bypass).
    # TNTP "capacity" values are BPR math artifacts — ignored for lanes.
    i29_links = {
        (1, 3), (3, 1), (3, 12), (12, 3), (12, 13), (13, 12),
    }
    i229_links = {(7, 18), (18, 7)}

    for i, link in enumerate(net.links):
        lon1, lat1 = node_coords[link.init_node]
        lon2, lat2 = node_coords[link.term_node]
        dist_m = haversine_m(lon1, lat1, lon2, lat2)
        dist_km = dist_m / 1000.0

        key = (link.init_node, link.term_node)

        if key in i29_links:
            n_lanes = 3
            speed_kmh = 105.0
            highway = "motorway"
        elif key in i229_links:
            n_lanes = 2
            speed_kmh = 105.0
            highway = "motorway"
        else:
            n_lanes = 2
            # TNTP lengths and FFTs are arbitrary (README: "Link lengths are
            # set equal to free flow travel times").  Use a fixed arterial
            # speed consistent with Sioux Falls urban arterials (40-45 mph).
            speed_kmh = 65.0
            highway = "primary"

        # Compensate for OSRM car profile speed reduction
        maxspeed = max(10, round(speed_kmh / osrm_speed_factor))

        way_id = 1000 + i
        ways.append({
            "id": way_id,
            "nodes": [link.init_node, link.term_node],
            "tags": {
                "highway": highway,
                "oneway": "yes",
                "maxspeed": str(maxspeed),
                "lanes": str(n_lanes),
                "name": f"Link {link.init_node}-{link.term_node}",
            },
        })

        lane_map[key] = n_lanes
        link_attrs[key] = {
            "capacity": link.capacity,
            "freeflow_time_min": link.free_flow_time,
            "distance_m": dist_m,
            "ff_speed_kmh": speed_kmh,
            "n_lanes": n_lanes,
            "b": link.b,
            "power": link.power,
        }

    osm_path = write_osm(node_coords, ways, path)

    zone_centroids = {z: node_coords[z] for z in range(1, n_zones + 1)}

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


def patch_sioux_falls_lanes(
    state,
    meta: dict,
    jam_density_per_lane: float | None = None,
) -> None:
    """Patch NetworkState with lane counts for Sioux Falls network."""
    kj_lane = jam_density_per_lane or meta.get("jam_density_per_lane", 200.0)
    lane_map = meta["lane_map"]
    for i in range(state.n_edges):
        key = (int(state.edge_ids[i, 0]), int(state.edge_ids[i, 1]))
        if key in lane_map:
            lanes = lane_map[key]
            state.n_lanes[i] = lanes
            state.jam_density[i] = kj_lane * lanes
