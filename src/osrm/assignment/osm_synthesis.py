"""Synthesize minimal OSM XML files for test networks.

Generates valid .osm files that can be processed by OSRM extract.
Used for structural validation on networks with known theoretical properties.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree.ElementTree import Element, SubElement, ElementTree


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
    - Shortcut 3→4: short, wide (if enabled)

    Origin = node 1, Destination = node 2.

    Nodes are placed ~500m apart in a diamond at Monaco coordinates.

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
    # Diamond layout: node 1 (left), 3 (top), 4 (bottom), 2 (right)
    # Using Monaco-area coordinates for consistency with test data
    nodes = {
        1: (7.4200, 43.7350),  # left (origin)
        3: (7.4230, 43.7370),  # top
        4: (7.4230, 43.7330),  # bottom
        2: (7.4260, 43.7350),  # right (destination)
    }

    ways = [
        # 1→3: narrow, congestion-sensitive
        {
            "id": 101, "nodes": [1, 3],
            "tags": {
                "highway": "secondary", "oneway": "yes",
                "maxspeed": "50", "lanes": "1", "name": "Link 1-3",
            },
        },
        # 1→4: wide, effectively constant cost
        {
            "id": 102, "nodes": [1, 4],
            "tags": {
                "highway": "primary", "oneway": "yes",
                "maxspeed": "30", "lanes": "4", "name": "Link 1-4",
            },
        },
        # 3→2: wide, effectively constant cost
        {
            "id": 103, "nodes": [3, 2],
            "tags": {
                "highway": "primary", "oneway": "yes",
                "maxspeed": "30", "lanes": "4", "name": "Link 3-2",
            },
        },
        # 4→2: narrow, congestion-sensitive
        {
            "id": 104, "nodes": [4, 2],
            "tags": {
                "highway": "secondary", "oneway": "yes",
                "maxspeed": "50", "lanes": "1", "name": "Link 4-2",
            },
        },
    ]

    if with_shortcut:
        # 3→4: short, wide, low cost
        ways.append({
            "id": 105, "nodes": [3, 4],
            "tags": {
                "highway": "primary", "oneway": "yes",
                "maxspeed": "60", "lanes": "4", "name": "Shortcut 3-4",
            },
        })

    osm_path = write_osm(nodes, ways, path)

    metadata = {
        "origin": nodes[1],
        "destination": nodes[2],
        "nodes": nodes,
        "n_links": len(ways),
        "with_shortcut": with_shortcut,
        "narrow_links": ["1→3", "4→2"],
        "wide_links": ["1→4", "3→2"] + (["3→4"] if with_shortcut else []),
    }

    return osm_path, metadata
