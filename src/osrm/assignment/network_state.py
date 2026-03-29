"""Network state model for traffic assignment.

Maintains per-edge state arrays (flow, density, speed) indexed by directed
OSM node pairs. Provides O(1) edge lookup and vectorized state updates.

See docs/pyosrm_assignment_module.md §3 for the data model.

Important: freeflow_kmh is an immutable physical road attribute
-------------------------------------------------------------
Free-flow speed must be set once at network discovery time and never
overwritten.  On a clean (uncustomized) OSRM instance, annotation speed
equals the profile-derived speed from OSM maxspeed tags, so it is safe to
use as freeflow.  After any segment-speed customization, annotation speed
reflects congested conditions and must NOT be used for freeflow.

Lane count and jam density are NOT available from OSRM annotations and
must be supplied via the ``state_patch`` callback or an external OSM
reader.

TODO: Add an OSM PBF/XML reader to extract ``lanes`` tags directly,
removing the need for a state_patch callback for lane-dependent
attributes.  Alternatively, extend OSRM's annotation API to expose
lane count per segment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

# Default per-lane jam density (veh/km) by OSM highway class
DEFAULT_JAM_DENSITY_PER_LANE: Dict[str, float] = {
    "motorway": 150.0,
    "trunk": 140.0,
    "primary": 130.0,
    "secondary": 120.0,
    "tertiary": 120.0,
    "residential": 100.0,
    "unclassified": 100.0,
}

DEFAULT_LANES: Dict[str, int] = {
    "motorway": 3,
    "trunk": 2,
    "primary": 2,
    "secondary": 1,
    "tertiary": 1,
    "residential": 1,
    "unclassified": 1,
}


@dataclass
class NetworkState:
    """Per-edge assignment state, indexed by edge ordinal.

    Attributes
    ----------
    edge_ids : np.ndarray
        (N, 2) uint64 — (from_osm_id, to_osm_id) for each edge.
    length_m : np.ndarray
        (N,) float64 — link length in meters.
    freeflow_kmh : np.ndarray
        (N,) float64 — free-flow speed in km/h.
    jam_density : np.ndarray
        (N,) float64 — jam density in veh/km (total across all lanes).
    n_lanes : np.ndarray
        (N,) uint8 — number of lanes per direction.
    flow_vph : np.ndarray
        (N,) float64 — current assigned flow in veh/hr.
    density_vpkm : np.ndarray
        (N,) float64 — current density in veh/km.
    speed_kmh : np.ndarray
        (N,) float64 — current speed from VDF in km/h.
    """

    edge_ids: np.ndarray
    length_m: np.ndarray
    freeflow_kmh: np.ndarray
    jam_density: np.ndarray
    n_lanes: np.ndarray
    flow_vph: np.ndarray = field(init=False)
    density_vpkm: np.ndarray = field(init=False)
    speed_kmh: np.ndarray = field(init=False)
    _edge_index: Dict[Tuple[int, int], int] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        n = len(self.edge_ids)
        self.flow_vph = np.zeros(n, dtype=np.float64)
        self.density_vpkm = np.zeros(n, dtype=np.float64)
        self.speed_kmh = self.freeflow_kmh.copy()
        self._build_index()

    def _build_index(self) -> None:
        """Build hash map: (from_osm_id, to_osm_id) → edge ordinal."""
        self._edge_index = {
            (int(self.edge_ids[i, 0]), int(self.edge_ids[i, 1])): i
            for i in range(len(self.edge_ids))
        }

    @property
    def n_edges(self) -> int:
        return len(self.edge_ids)

    def edge_ordinal(self, from_id: int, to_id: int) -> Optional[int]:
        """Look up edge index by OSM node pair. Returns None if not found."""
        return self._edge_index.get((from_id, to_id))

    def reset_flow(self) -> None:
        """Zero out flow for a new assignment iteration."""
        self.flow_vph[:] = 0.0

    def accumulate_flow(
        self,
        edge_indices: np.ndarray,
        flow_increments: np.ndarray,
    ) -> None:
        """Add flow increments to edges by ordinal index.

        Parameters
        ----------
        edge_indices : np.ndarray
            (M,) int — edge ordinals to update.
        flow_increments : np.ndarray
            (M,) float64 — flow to add (veh/hr) per edge.
        """
        np.add.at(self.flow_vph, edge_indices, flow_increments)

    def accumulate_route(
        self,
        nodes: list,
        volume_vph: float,
    ) -> int:
        """Accumulate uniform flow from a route's node sequence.

        Parameters
        ----------
        nodes : list
            Sequence of OSM node IDs from OSRM route annotations.
        volume_vph : float
            Flow rate to add to each edge on the route (veh/hr).

        Returns
        -------
        int
            Number of edges successfully matched.
        """
        matched = 0
        for i in range(len(nodes) - 1):
            idx = self.edge_ordinal(int(nodes[i]), int(nodes[i + 1]))
            if idx is not None:
                self.flow_vph[idx] += volume_vph
                matched += 1
        return matched

    def register_edge(
        self,
        from_id: int,
        to_id: int,
        length_m: float,
        freeflow_kmh: float,
        jam_density: float,
        n_lanes: int,
    ) -> int:
        """Register a new edge discovered mid-assignment.

        The caller is responsible for providing a correct ``freeflow_kmh``.
        During iteration (after OSRM customization), annotation speed is
        congested — do NOT pass it here.  Use the network median freeflow
        or a config default instead.

        Returns the ordinal of the new (or existing) edge.
        """
        key = (int(from_id), int(to_id))
        existing = self._edge_index.get(key)
        if existing is not None:
            return existing

        idx = self.n_edges
        self.edge_ids = np.vstack([
            self.edge_ids,
            np.array([[from_id, to_id]], dtype=np.uint64),
        ])
        self.length_m = np.append(self.length_m, length_m)
        self.freeflow_kmh = np.append(self.freeflow_kmh, max(freeflow_kmh, 1.0))
        self.jam_density = np.append(self.jam_density, jam_density)
        self.n_lanes = np.append(self.n_lanes, np.uint8(n_lanes))
        self.flow_vph = np.append(self.flow_vph, 0.0)
        self.density_vpkm = np.append(self.density_vpkm, 0.0)
        self.speed_kmh = np.append(self.speed_kmh, max(freeflow_kmh, 1.0))
        self._edge_index[key] = idx
        return idx

    @classmethod
    def from_edges(
        cls,
        from_ids: np.ndarray,
        to_ids: np.ndarray,
        lengths_m: np.ndarray,
        freeflow_kmh: np.ndarray,
        jam_density: np.ndarray,
        n_lanes: np.ndarray,
    ) -> "NetworkState":
        """Construct from parallel arrays."""
        edge_ids = np.column_stack(
            [np.asarray(from_ids, dtype=np.uint64), np.asarray(to_ids, dtype=np.uint64)]
        )
        return cls(
            edge_ids=edge_ids,
            length_m=np.asarray(lengths_m, dtype=np.float64),
            freeflow_kmh=np.asarray(freeflow_kmh, dtype=np.float64),
            jam_density=np.asarray(jam_density, dtype=np.float64),
            n_lanes=np.asarray(n_lanes, dtype=np.uint8),
        )

    @classmethod
    def from_route_annotations(
        cls,
        routes: list,
        default_jam_density_per_lane: float = 130.0,
        default_n_lanes: int = 1,
    ) -> "NetworkState":
        """Build a NetworkState from OSRM route annotation results.

        IMPORTANT: This must be called on a **clean** (uncustomized) OSRM
        instance so that annotation speed reflects the original profile
        speed (derived from OSM ``maxspeed`` tags).  After any segment-speed
        customization, annotation speed is congested and must NOT be used
        as freeflow.

        Discovers all unique directed edges across all routes. Uses OSRM
        annotations for length and freeflow speed; applies defaults for
        jam_density and lanes (not available from OSRM annotations —
        supply via ``state_patch`` or an external OSM reader).

        Parameters
        ----------
        routes : list
            List of OSRM route result dicts, each having
            ``routes[0]["legs"][*]["annotation"]`` with ``nodes``,
            ``distance``, and ``speed`` arrays.
        default_jam_density_per_lane : float
            Per-lane jam density when road class is unknown.
        default_n_lanes : int
            Lanes per direction when not available.

        Returns
        -------
        NetworkState
        """
        edge_data: Dict[Tuple[int, int], Tuple[float, float]] = {}

        for route_result in routes:
            for route in route_result.get("routes", [route_result]):
                for leg in route.get("legs", []):
                    ann = leg.get("annotation", {})
                    nodes = ann.get("nodes", [])
                    distances = ann.get("distance", [])
                    speeds = ann.get("speed", [])
                    for i in range(len(nodes) - 1):
                        key = (int(nodes[i]), int(nodes[i + 1]))
                        if key not in edge_data:
                            dist = distances[i] if i < len(distances) else 0.0
                            spd = speeds[i] if i < len(speeds) else 0.0
                            spd_kmh = spd * 3.6  # m/s → km/h
                            edge_data[key] = (dist, max(spd_kmh, 1.0))

        n = len(edge_data)
        if n == 0:
            return cls(
                edge_ids=np.empty((0, 2), dtype=np.uint64),
                length_m=np.empty(0, dtype=np.float64),
                freeflow_kmh=np.empty(0, dtype=np.float64),
                jam_density=np.empty(0, dtype=np.float64),
                n_lanes=np.empty(0, dtype=np.uint8),
            )

        keys = list(edge_data.keys())
        from_ids = np.array([k[0] for k in keys], dtype=np.uint64)
        to_ids = np.array([k[1] for k in keys], dtype=np.uint64)
        lengths = np.array([edge_data[k][0] for k in keys], dtype=np.float64)
        speeds = np.array([edge_data[k][1] for k in keys], dtype=np.float64)
        n_lanes_arr = np.full(n, default_n_lanes, dtype=np.uint8)
        jam_density_arr = np.full(
            n, default_jam_density_per_lane * default_n_lanes, dtype=np.float64
        )

        return cls(
            edge_ids=np.column_stack([from_ids, to_ids]),
            length_m=lengths,
            freeflow_kmh=speeds,
            jam_density=jam_density_arr,
            n_lanes=n_lanes_arr,
        )
