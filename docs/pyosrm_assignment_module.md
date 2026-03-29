# py-osrm Assignment Module Specification

> **Parent document**: [Traffic Assignment Design](traffic_assignment_design.md)
> **Related**: [OSRM Multi-Period Patch](osrm_multi_period_patch.md)

---

## 1  OSRM traffic update surface

This section documents the mechanism that makes the entire design possible.

### 1.1  The hidden UpdaterConfig

`CustomizationConfig` (OSRM v6.0.0) embeds an `UpdaterConfig`:

```
customizer/customizer_config.hpp
├── unsigned requested_num_threads
└── updater::UpdaterConfig updater_config
        ├── std::vector<std::string> segment_speed_lookup_paths   ← CSV paths
        ├── std::vector<std::string> turn_penalty_lookup_paths    ← CSV paths
        ├── std::string tz_file_path
        ├── double log_edge_updates_factor
        └── std::time_t valid_now
```

**py-osrm today** (`src/customizerconfig_nb.cpp`) exposes only
`requested_num_threads` and `UseDefaultOutputNames`. The `updater_config`
sub-struct is completely hidden.

### 1.2  Segment-speed CSV format

Each line: `from_osm_node_id, to_osm_node_id, speed_km_h [, rate]`

```
50267,27780,32         ← duration and weight both updated from speed
34491,34494,3,         ← duration updated, weight unchanged (blank rate)
50267,27780,32,30.3    ← duration from speed, weight from length/rate
```

- `from`/`to` are **OSM node IDs** for directly connected nodes.
- Order matters: `A,B,speed` updates A→B only; B→A needs a separate line.
- Multiple CSV files are supported; last-one-wins on conflicts.

### 1.3  Turn-penalty CSV format

Each line: `from_osm_id, via_osm_id, to_osm_id, penalty_seconds [, weight_penalty]`

### 1.4  What `osrm::customize(config)` does internally

```
Customizer::Run(config)
  │
  ├─ Updater(config.updater_config)
  │    ├─ csv::readSegmentValues(segment_speed_lookup_paths)
  │    │    → SegmentLookupTable  (sorted vector, O(log n) lookup)
  │    ├─ csv::readTurnValues(turn_penalty_lookup_paths)
  │    │    → TurnLookupTable
  │    ├─ For each geometry (parallel via TBB):
  │    │    For each segment (u, v) in geometry:
  │    │      if segment_speed_lookup({u, v}) found:
  │    │        update duration = distance / (speed / 3.6)
  │    │        update weight   = distance / rate  (or from speed)
  │    └─ Write updated .osrm.geometry, .osrm.turn_*_penalties
  │
  ├─ Build edge-based graph from updated weights
  ├─ Re-compute MLD cell metrics
  └─ Write .osrm.cell_metrics, .osrm.mldgr
```

### 1.5  Engine refresh: two strategies

#### Strategy A: Destroy and recreate (simple, works with mmap mode)

The OSRM engine memory-maps `.osrm.*` files at construction. After customize
rewrites some of those files, a new engine instance must be created:

```python
# 1. Destroy old engine (releases memory maps)
del engine  # or engine = None

# 2. Run customize with segment-speed updates
osrm.customize(base_path,
    segment_speed_file="updated_speeds.csv")   # ← needs new wrapper API

# 3. Create fresh engine (re-maps updated files)
engine = osrm.OSRM(base_path, algorithm="MLD")
```

**Constraint**: The old engine **must** be destroyed before customize writes,
because the files are memory-mapped.

#### Strategy B: Shared-memory hot-swap (zero-downtime, production-grade)

OSRM has a built-in hot-swap mechanism via shared memory that avoids
engine destroy/recreate entirely:

1. **`osrm-datastore`** loads `.osrm` files into shared memory regions and
   updates a timestamp in a `SharedRegionRegister`.
2. When `use_shared_memory=True`, the engine creates a **`WatchingProvider`**
   with a background **`DataWatchdog`** thread that monitors for timestamp
   changes (`include/engine/data_watchdog.hpp`).
3. When the watchdog detects a timestamp change, it atomically swaps the
   `DataFacadeFactory`. Active requests continue on the old facade
   (`shared_ptr` ref-counting); new requests get the new data.

```python
# Initial setup
engine = osrm.OSRM(base_path, algorithm="MLD", use_shared_memory=True)
# (requires prior: python -m osrm datastore base_path)

# Update loop — engine stays alive
osrm.customize(base_path, segment_speed_file="updated_speeds.csv")
# python -m osrm datastore base_path   (or programmatic equivalent)
# → watchdog auto-detects, swaps facade, zero downtime
```

py-osrm already exposes `use_shared_memory` on `EngineConfig` and ships
`osrm-datastore` via `python -m osrm datastore`. The missing piece is a
**programmatic datastore reload** (currently CLI-only).

**Recommendation**: Use Strategy A for the prototype (simpler, no shared-memory
setup). Migrate to Strategy B for production or when customize latency
profiling shows that engine re-instantiation is a bottleneck.

### 1.6  Avoiding disk I/O: in-memory CSV strategies

The CSV file parser (`include/updater/csv_file_parser.hpp`) uses
`boost::iostreams::mapped_file_source` to memory-map the CSV. This opens
several options for avoiding real disk I/O:

| Strategy | Effort | Throughput | OSRM changes? |
|----------|--------|-----------|---------------|
| **tmpfs (`/dev/shm/`)** | None | ~zero-copy from RAM | No |
| **Named pipe (FIFO)** | Low | Streaming, no file | No |
| **In-memory `LookupTable` bypass** | Medium | Skip CSV entirely | Patch vendored OSRM |
| **Direct weight injection** | High | Skip Updater entirely | Significant OSRM changes |

**Recommended for prototype**: Write CSV to `/dev/shm/osrm_speeds.csv`. The
memory-mapped read is then purely from RAM — no disk I/O at all, no OSRM
changes required. This can be made transparent:

```python
import tempfile, os

def write_speeds_to_tmpfs(edge_ids, speeds_kmh):
    path = "/dev/shm/osrm_assignment_speeds.csv"
    with open(path, 'w') as f:
        for (from_id, to_id), speed in zip(edge_ids, speeds_kmh):
            f.write(f"{from_id},{to_id},{speed:.1f}\n")
    return path
```

**Future optimization**: Build `SegmentLookupTable` directly in C++ from NumPy
arrays, bypassing CSV parsing entirely. This requires patching the vendored
OSRM `Updater` to accept pre-built lookup tables, but does not require an
upstream PR.



## 2  Path-to-link accounting

Route annotations are the bridge between OSRM's path output and the assignment
engine's link-level state.

### 2.1  What OSRM returns with `annotations=["nodes", "distance", "speed"]`

```python
result = engine.Route(
    coordinates=[(lon1, lat1), (lon2, lat2)],
    annotations=["nodes", "distance", "speed"],
    overview="full",
    geometries="geojson"
)

leg = result["routes"][0]["legs"][0]["annotation"]
# leg["nodes"]     = [n0, n1, n2, n3, ...]   ← OSM node IDs (uint64)
# leg["distance"]  = [d01, d12, d23, ...]     ← per-segment meters (uint32)
# leg["speed"]     = [s01, s12, s23, ...]     ← per-segment m/s (float)
```

### 2.2  Edge identity

An **assignable edge** is defined by a directed pair of consecutive OSM node
IDs from the `nodes` annotation:

```
nodes = [n0, n1, n2, n3]
edges = [(n0, n1), (n1, n2), (n2, n3)]
```

These pairs map **directly** to the segment-speed CSV format. This is the
critical property that makes the entire design work without OSRM-core changes.

### 2.3  Per-edge data available from annotations

| Annotation | Type | Unit | Per-edge? |
|------------|------|------|-----------|
| `nodes` | uint64 | OSM node ID | yes (consecutive pairs = edges) |
| `distance` | uint32 | meters | yes |
| `duration` | uint32 | seconds | yes |
| `weight` | uint32 | routing weight | yes |
| `speed` | float | m/s | yes |
| `datasources` | uint8 | source ID | yes |

**Not available**: OSM way IDs, internal edge IDs, number of lanes, capacity.
These must come from external data or heuristics (supplied via the
``state_patch`` callback).

> **Freeflow speed invariant.**  On a *clean* (uncustomized) OSRM instance,
> annotation ``speed`` reflects the profile-derived speed from OSM ``maxspeed``
> tags — this IS the free-flow speed.  Segment-speed customization
> **permanently mutates** OSRM edge weights (even re-partition does not undo
> it; only a full re-extract from OSM resets).  Therefore ``freeflow_kmh`` is
> captured once at network discovery and treated as immutable.  Each
> ``AssignmentLoop.run()`` call must operate on a clean OSRM base path.
>
> **TODO — lane count from OSM.**  Lane count is not exposed by OSRM
> annotations.  Currently supplied via ``state_patch``.  Future work: add an
> OSM PBF/XML reader to extract ``lanes`` tags directly, or extend OSRM's
> annotation API to include lane count per segment.


## 3  Network state model

### 3.1  Link state variables

For each directed edge `(from_osm_id, to_osm_id)`, the assignment engine
maintains:

| Variable | Symbol | Unit | Source |
|----------|--------|------|--------|
| Length | L_e | meters | OSRM `distance` annotation (first iteration) |
| Free-flow speed | v_f,e | km/h | OSRM profile / OSM `maxspeed` tag |
| Jam density | k_j,e | veh/km | Estimated from lanes × per-lane jam density |
| Number of lanes | n_lanes | — | OSM `lanes` tag or heuristic by road class |
| Current flow | q_e | veh/hr | Accumulated from assigned paths |
| Current density | k_e | veh/km | Derived from q_e and v_e |
| Current speed | v_e | km/h | Evaluated from VDF(k_e) |

### 3.2  Default assumptions for missing data

OSM coverage of capacity-related attributes is incomplete. Defaults:

| Road class | Default lanes | Per-lane k_j (veh/km) | Per-lane capacity (veh/hr) |
|------------|--------------|----------------------|---------------------------|
| motorway | 3 | 150 | 2000 |
| trunk | 2 | 140 | 1800 |
| primary | 2 | 130 | 1600 |
| secondary | 1 | 120 | 1400 |
| tertiary | 1 | 120 | 1200 |
| residential | 1 | 100 | 800 |

These are starting points; calibration against observed data is expected.

### 3.3  Data structure sketch

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class NetworkState:
    """Per-edge assignment state, indexed by edge ordinal."""
    edge_ids: np.ndarray       # (N, 2) uint64 — (from_osm_id, to_osm_id)
    length_m: np.ndarray       # (N,) float64
    freeflow_kmh: np.ndarray   # (N,) float64
    jam_density: np.ndarray    # (N,) float64  — veh/km (total, all lanes)
    n_lanes: np.ndarray        # (N,) uint8
    flow_vph: np.ndarray       # (N,) float64  — veh/hr (current assignment)
    density_vpkm: np.ndarray   # (N,) float64  — veh/km (current)
    speed_kmh: np.ndarray      # (N,) float64  — from VDF
```

A hash map `(from_osm_id, to_osm_id) → edge_ordinal` provides O(1) lookup
during flow accumulation.



## 4  Concrete implementation gaps

### 4.1  py-osrm wrapper changes (Phase 1)

Required before any assignment work can begin:

| Gap | Location | Work |
|-----|----------|------|
| **Expose `updater_config`** on `CustomizationConfig` | `src/customizerconfig_nb.cpp` | Bind `segment_speed_lookup_paths` and `turn_penalty_lookup_paths` as read-write properties |
| **Python `customize()` must accept speed/penalty file args** | `src/osrm/preprocessing.py` | Forward `segment_speed_file` and `turn_penalty_file` kwargs to `updater_config` paths |
| **Expose `departure_period` / `departure_time`** | `src/osrm_nb.cpp` | Add to RouteParameters / TableParameters bindings (after OSRM core patch, Phase 5) |
| **tmpfs CSV writer** | `src/osrm/assignment.py` | Write segment-speed CSV to `/dev/shm/` for zero-disk-I/O (§1.6) |

### 4.2  New assignment module

A new `src/osrm/assignment/` package containing:

- `NetworkState` — per-edge state arrays, edge index, flow accumulation
- `BiParabolicVDF` — vectorized VDF evaluation, flow-to-density solver ([Traffic Assignment Design](traffic_assignment_design.md) §3)
- `DensitySmoothing` — neighbor-based smoothing ([Traffic Assignment Design](traffic_assignment_design.md) §3.8)
- `AssignmentLoop` — outer loop, multi-period routing, convergence control
- `PeriodConfig` — user-defined period mappings ([OSRM Multi-Period Patch](osrm_multi_period_patch.md) §6)
- `DemandAdapter` — abstract base, `ODMatrixAdapter`, `TripStreamAdapter`
- `SegmentSpeedWriter` — generates CSV to tmpfs from `NetworkState`

### 4.3  Performance-critical path (candidate for C++ extension)

The flow accumulation step (decompose millions of paths into per-edge flow
increments) is the most likely bottleneck. If Python + NumPy is too slow,
this should be moved to a C++ nanobind extension that:

1. Takes route annotation arrays (node IDs, distances) as input.
2. Performs hash-based edge lookup and atomic flow increment.
3. Returns updated flow array.

This can be added as a new `.cpp` source in the existing build without
touching OSRM core.


## 5  Assignment loop with multi-period routing

```
for iteration in 1..max_iter:
    # 1. Pre-customize all periods (parallelizable)
    for period in 0..N-1:
        write_speed_csv(period, network_state)
        osrm_customize(base_path, period_index=period)

    # 2. Reload engine (picks up all N metric sets)
    engine = reload_engine(base_path)

    # 3. Route ALL trips in one pass — each with its departure_period
    for trip in demand:
        period = period_config.time_to_period(trip.departure_time)
        result = engine.Route(trip.origin, trip.destination,
                              departure_period=period)
        accumulate_flow(result, network_state)

    # 4. Update network state (VDF, density smoothing)
    network_state.update_speeds()

    # 5. Check convergence
    if converged(network_state):
        break
```

All trips are routed in a single pass regardless of period count. The
engine handles period selection internally — no sequential
customize-per-period-per-batch. This is the key performance advantage
over the original "sequential customize" approach.

---

## 6  py-osrm file reference

| File | Role |
|------|------|
| `src/osrm_nb.cpp` | Main OSRM class binding (Route, Table, etc.) |
| `src/customizerconfig_nb.cpp` | CustomizationConfig binding (**needs extension**) |
| `src/osrm/preprocessing.py` | Python customize() wrapper (**needs extension**) |
| `src/osrm/bulk.py` | Bulk parallel routing (reusable for assignment) |
| `src/osrm/__init__.py` | Python OSRM wrapper class |
| `CMakeLists.txt` | Build config — links `osrm_customize` library |
| `docs/Fournier_Ped_Transit_priority_manuscript_v4.pdf` | Bi-parabolic VDF derivation (Eq. 11) |
