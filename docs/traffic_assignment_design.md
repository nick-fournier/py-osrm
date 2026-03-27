# Traffic Assignment Design Document

> **Status**: Design / feasibility — not yet implemented  
> **Goal**: Extend py-osrm into a dynamic traffic assignment platform leveraging OSRM's computational performance  
> **Long-term benchmark**: Supplant commercial tools such as Bentley OpenPaths for activity-based and OD-based assignment workflows

---

## 1  Executive summary

OSRM is a static shortest-path engine. Traffic assignment requires iterative,
demand-responsive network loading where link costs change as vehicles are
assigned. This document defines how to bridge that gap by treating OSRM as the
**path engine** inside an outer assignment loop that owns demand, network state,
and convergence control.

The central mechanism is OSRM's **MLD customization pipeline**, which already
supports segment-speed and turn-penalty CSV updates. These fields exist in the
C++ `CustomizationConfig.updater_config` struct but are **not yet exposed** by
the py-osrm Python binding. Exposing them is the first concrete implementation
task.

### Architectural decisions locked in

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Routing algorithm | **MLD only** | CH requires full re-contraction for cost updates; MLD customize is ~100× cheaper |
| First demand interface | **OD-matrix** | Easier to validate; matrix-free adapter comes second on the same core |
| Timing model | **Discrete time slices** | Frozen costs inside each slice; MLD refresh between slices |
| VDF family | **Bi-parabolic flow-density** (Fournier et al.) | Parameter-light (v_f, k_j); grounded in fundamental diagram; closed-form inverse; avoids BPR |
| Density model | **Mesoscopic with spatial smoothing** (§5.8) | Zone-averaged density avoids short-link volatility; handles spillover |
| CSV I/O | **tmpfs (`/dev/shm/`)** for prototype (§2.6) | Zero disk I/O; no OSRM changes needed; future: in-memory bypass |
| Engine refresh | **Destroy/recreate** (prototype), **shared-memory hot-swap** (production, §2.5) | Hot-swap is zero-downtime but requires `osrm-datastore` integration |
| Assignment logic location | **Wrapper side** (Python + C++ extension) | Avoid OSRM-core fork until profiling proves the boundary is the bottleneck |

---

## 2  How OSRM's traffic update surface works

This section documents the mechanism that makes the entire design possible.

### 2.1  The hidden UpdaterConfig

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

### 2.2  Segment-speed CSV format

Each line: `from_osm_node_id, to_osm_node_id, speed_km_h [, rate]`

```
50267,27780,32         ← duration and weight both updated from speed
34491,34494,3,         ← duration updated, weight unchanged (blank rate)
50267,27780,32,30.3    ← duration from speed, weight from length/rate
```

- `from`/`to` are **OSM node IDs** for directly connected nodes.
- Order matters: `A,B,speed` updates A→B only; B→A needs a separate line.
- Multiple CSV files are supported; last-one-wins on conflicts.

### 2.3  Turn-penalty CSV format

Each line: `from_osm_id, via_osm_id, to_osm_id, penalty_seconds [, weight_penalty]`

### 2.4  What `osrm::customize(config)` does internally

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

### 2.5  Engine refresh: two strategies

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

### 2.6  Avoiding disk I/O: in-memory CSV strategies

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

---

## 3  Path-to-link accounting

Route annotations are the bridge between OSRM's path output and the assignment
engine's link-level state.

### 3.1  What OSRM returns with `annotations=["nodes", "distance", "speed"]`

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

### 3.2  Edge identity

An **assignable edge** is defined by a directed pair of consecutive OSM node
IDs from the `nodes` annotation:

```
nodes = [n0, n1, n2, n3]
edges = [(n0, n1), (n1, n2), (n2, n3)]
```

These pairs map **directly** to the segment-speed CSV format. This is the
critical property that makes the entire design work without OSRM-core changes.

### 3.3  Per-edge data available from annotations

| Annotation | Type | Unit | Per-edge? |
|------------|------|------|-----------|
| `nodes` | uint64 | OSM node ID | yes (consecutive pairs = edges) |
| `distance` | uint32 | meters | yes |
| `duration` | uint32 | seconds | yes |
| `weight` | uint32 | routing weight | yes |
| `speed` | float | m/s | yes |
| `datasources` | uint8 | source ID | yes |

**Not available**: OSM way IDs, internal edge IDs, number of lanes, capacity.
These must come from external data or heuristics.

---

## 4  Network state model

### 4.1  Link state variables

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

### 4.2  Default assumptions for missing data

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

### 4.3  Data structure sketch

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

---

## 5  Bi-parabolic flow-density VDF

> **Reference**: The bi-parabolic formulation used here follows Fournier et al.,
> "Pedestrian and Transit Priority Zoning."
> See `docs/Fournier_Ped_Transit_priority_manuscript_v4.pdf` for full
> derivations and default parameter recommendations.

### 5.1  Why not BPR?

The Bureau of Public Roads (BPR) function `t = t_0 [1 + α(v/c)^β]` maps
volume-to-capacity ratio to delay. It has well-known problems:

- Arbitrary parameters (α=0.15, β=4 are conventional, not physical).
- Undefined behavior above capacity (extrapolates to infinity).
- No direct relationship to traffic flow theory.
- Density is not a state variable; you get delay as f(V/C) but no link to
  the fundamental diagram.

A speed-density model grounded in the fundamental diagram avoids these issues,
requires fewer arbitrary calibration constants, and — critically — uses density
as its state variable, which makes it compatible with mesoscopic modeling and
MFD-based smoothing (see §5.8).

### 5.2  Functional form (from Fournier et al.)

The model defines **flow as a function of density** using two parabolic
branches in q-k (flow-density) space — i.e., it IS a macroscopic fundamental
diagram (MFD). The parabolas join at critical density k_c with C¹ continuity
(both value and slope match; the slope at k_c is zero, which is the peak of
the MFD).

**Uncongested branch** (0 ≤ k ≤ k_c):

```
q(k) = q_c · k · (2k_c − k) / k_c²
```

This is a downward-opening parabola in q-k space with vertex at k = k_c,
q = q_c. At k = 0, q = 0.

**Congested branch** (k_c < k ≤ k_j):

```
q(k) = q_c · [1 − (k − k_c)² / (k_j − k_c)²]
```

This is also a downward parabola in q-k space with vertex at k = k_c, q = q_c.
At k = k_j (jam), q = 0.

**Derived speed-density** (from q = k · v):

```
Uncongested:  v(k) = q_c · (2k_c − k) / k_c²         (linear in k)
Congested:    v(k) = q_c · [1 − (k − k_c)² / (k_j − k_c)²] / k
```

Note: the uncongested speed-density is **linear** (Greenshields-like), not
parabolic. The parabola is in the flow-density plane, which is the more
physically meaningful space for MFD-based modeling.

**Travel time** on a link of length L:

```
t(k) = L / v(k) = L · k / q(k)
```

Where:
- v_f = free-flow speed (km/h) — speed at k → 0
- k_j = jam density (veh/km) — density at v = 0
- k_c = critical density at which flow is maximized
- q_c = capacity flow (veh/hr) — maximum throughput

### 5.3  Parameter relationships

At k = 0 (empty network):

```
v_f = lim_{k→0} v(k) = q_c · 2k_c / k_c² = 2 · q_c / k_c
  ⟹  q_c = v_f · k_c / 2
```

At k = k_c (capacity):

```
v_c = v(k_c) = q_c / k_c = v_f / 2
```

**C¹ continuity at k_c** (automatic by construction):

Both branches are parabolas with vertex at (k_c, q_c). The slope dq/dk = 0
at k_c for both branches, so the junction is smooth.

```
Uncongested: dq/dk = q_c · (2k_c − 2k) / k_c²      → at k_c: 0  ✓
Congested:   dq/dk = −2q_c · (k − k_c) / (k_j − k_c)²  → at k_c: 0  ✓
```

### 5.4  Default parameters (from Fournier et al.)

The bi-parabolic model is "parameter-light" — only two inputs are strictly
required per link:

- **v_f** — free-flow speed (from OSM / OSRM profile, readily available)
- **k_j** — jam density (from `n_lanes × per_lane_jam_density`, estimable)

**k_c** is derived, not calibrated. The paper recommends:

```
k_c = k_j / 3
```

which gives:

```
q_c = v_f · k_c / 2 = v_f · k_j / 6
v_c = v_f / 2
```

| Parameter | Typical vehicular default | Source |
|-----------|--------------------------|--------|
| v_f | From OSM maxspeed or OSRM profile | Available per-link |
| k_j (per lane) | 150 veh/km | Standard assumption |
| k_c | k_j / 3 | Fournier et al. recommendation |
| v_c | v_f / 2 | Derived |
| q_c (per lane) | 1500 veh/hr at v_f=60 km/h | Derived: v_f · k_j / 6 |

The paper also provides defaults for pedestrian and transit modes; those are
not needed for the vehicular assignment prototype but could extend to
multimodal assignment later.

### 5.5  From density to OSRM weights

The VDF produces speed v(k) in km/h. OSRM's segment-speed CSV takes speed in
km/h. The mapping is therefore direct:

```
density k_e  →  v(k_e) (km/h)  →  CSV line: "from,to,v_e"

Uncongested: v_e = q_c · (2k_c − k_e) / k_c²
Congested:   v_e = q_c · [1 − (k_e − k_c)² / (k_j − k_c)²] / k_e
```

OSRM internally converts speed to duration: `duration = distance / (speed / 3.6)`

### 5.6  From flow to density (closed-form inverse)

The assignment loop produces flows q_e (veh/hr). We need density k_e to
evaluate the VDF. Unlike many VDF formulations that require iterative solvers,
the parabolic q-k form yields a **closed-form inverse** via the quadratic
formula.

**Uncongested branch** (q ≤ q_c):

Starting from `q = q_c · k · (2k_c − k) / k_c²`, let u = k/k_c:

```
q/q_c = u · (2 − u) = 2u − u²
u² − 2u + q/q_c = 0
u = 1 − √(1 − q/q_c)     (taking the physically meaningful root)
```

Therefore:

```
k(q) = k_c · (1 − √(1 − q / q_c))
```

This is **exact, non-iterative, and vectorizable** — no Newton solver needed.

**Congested branch** (q ≤ q_c, congested side):

Starting from `q = q_c · [1 − (k − k_c)² / (k_j − k_c)²]`:

```
(k − k_c)² / (k_j − k_c)² = 1 − q/q_c
k = k_c + (k_j − k_c) · √(1 − q/q_c)
```

**Branch selection policy**:

- **If q_e ≤ q_c**: use the uncongested branch (lower k, higher v).
  This is the standard equilibrium assumption.
- **If q_e > q_c**: the link is oversaturated. See §5.7.

**Vectorized implementation** (production use over all edges):

```python
def flow_to_density_vectorized(q, v_f, k_j, k_c):
    """Closed-form flow-to-density inversion. No iteration needed."""
    q_c = v_f * k_c / 2.0
    ratio = np.clip(q / q_c, 0.0, 1.0)
    k = k_c * (1.0 - np.sqrt(1.0 - ratio))
    return k

def density_to_speed_vectorized(k, v_f, k_j, k_c):
    """Evaluate VDF: density → speed for both branches."""
    q_c = v_f * k_c / 2.0
    uncongested = q_c * (2.0 * k_c - k) / k_c**2
    congested = np.where(
        k > 0,
        q_c * (1.0 - (k - k_c)**2 / (k_j - k_c)**2) / k,
        0.0,
    )
    return np.where(k <= k_c, uncongested, congested)

def flow_to_speed_vectorized(q, v_f, k_j, k_c):
    """Full pipeline: flow → density → speed. Single pass, no iteration."""
    k = flow_to_density_vectorized(q, v_f, k_j, k_c)
    return density_to_speed_vectorized(k, v_f, k_j, k_c)
```

This closed-form solution is a major advantage over BPR-based or other VDF
formulations that require iterative inversion. For N edges, the entire
flow-to-speed conversion is a single vectorized NumPy operation — O(N) with
no loops.

### 5.7  Oversaturation policy

When assigned flow exceeds capacity, physical queuing occurs. For the first
prototype, we use a simple penalty:

- Clamp density at k_j (speed → 0 is not useful).
- Instead, set speed to a configurable floor (e.g., 5 km/h).
- Optionally report oversaturated links for diagnostics.

Spillback modeling (queues propagating upstream) is deferred but partially
addressed by the mesoscopic density smoothing in §5.8.

### 5.8  Mesoscopic density model: spatial smoothing

> **Design decision**: This is a **mesoscopic** model, not a purely
> link-based microsimulation. Pure per-link density is fragile: short links
> exhibit volatile densities, spillback crosses link boundaries, and
> isolated link states cannot represent network-level congestion effects.
>
> **Preferred approach**: topological neighbor-based smoothing. Zone-based
> aggregation is available as a fallback but is not the default.

#### The problem with raw link density

Given flow q_e on link e of length L_e, the instantaneous density is:

```
k_e = q_e / v_e = n_e / L_e
```

where n_e is the number of vehicles on the link. For short links (e.g., 50 m
urban segments), even a few vehicles produce extreme densities. Moreover,
when a downstream link reaches jam density, the physical spillback into
upstream links is not represented — each link is an island.

#### Primary approach: neighbor-based smoothing

Smooth each link's density with its immediate **topological neighbors** —
the links that share a node (upstream or downstream):

```
k̃_e = (1 − β) · k_e + β · Σ_{j ∈ N(e)} w_ej · k_j
```

where:
- N(e) is the set of links sharing a node with e
- w_ej are normalized weights (e.g., proportional to link length L_j)
- β ∈ [0, 0.5] is a smoothing parameter (default: 0.3)

**Why this is preferred**:

- **No zone definitions needed** — operates purely on network topology.
- **Computationally cheap** — one sparse matrix-vector multiply per iteration.
  The adjacency structure is static and can be precomputed as a CSR matrix.
- **Physically motivated** — spillback is a local phenomenon that propagates
  along connected links, not across arbitrary zone boundaries.
- **Multi-pass option** — applying the smoothing kernel N times is equivalent
  to diffusing density over N hops, giving tunable spatial reach without
  requiring explicit zone sizes.

**Vectorized implementation sketch**:

```python
# Precompute once from network topology
# W is (N_edges, N_edges) sparse CSR, row-normalized
# W[e, j] = L_j / Σ_{m ∈ N(e)} L_m  for j ∈ N(e), else 0

def smooth_density(k, W, beta=0.3, passes=1):
    """Neighbor-based density smoothing. O(nnz) per pass."""
    k_smooth = k.copy()
    for _ in range(passes):
        k_smooth = (1 - beta) * k_smooth + beta * (W @ k_smooth)
    return k_smooth
```

The smoothed density k̃_e then feeds the VDF:

```
v_e = VDF(k̃_e)
```

#### Fallback: zone-based aggregation (H3)

When zone-level aggregation is desired (e.g., for MFD diagnostics or
comparison with bathtub models), use **Uber H3 hexagonal cells** rather
than arbitrary grids or TAZs:

- H3 provides a standardized, hierarchical spatial index with consistent
  cell sizes at each resolution level.
- Resolution 7 (~5.16 km² per cell) or 8 (~0.74 km²) are reasonable
  starting points for urban networks.
- Each link is assigned to the H3 cell containing its midpoint.
- The `h3` Python package is lightweight and well-maintained.

For each H3 cell z, compute an aggregate density:

```
K_z = Σ_{e ∈ z} (k_e · L_e) / Σ_{e ∈ z} L_e
```

The zone-level density feeds the VDF to produce a **zone-level reference speed**:

```
V_z = VDF(K_z)     (using zone-average v_f and k_j)
```

Individual link speeds can then be blended:

```
v_e = λ · VDF(k_e)  +  (1 − λ) · v_f,e · (V_z / v_f,z)
```

- **λ = 1**: pure link-level (no zone influence).
- **λ = 0**: pure zone-level MFD (bathtub-like).
- This is primarily useful for **diagnostics and validation**, not as the
  default production smoothing strategy.

#### Implementation recommendation

```python
class DensitySmoothingConfig:
    method: str = "neighbor"    # "neighbor" (default), "h3", or "none"
    neighbor_beta: float = 0.3
    neighbor_passes: int = 1    # multi-pass for wider spatial reach
    h3_resolution: int = 8      # only used if method="h3"
    h3_blend_lambda: float = 0.7
```

#### Relationship to the MFD literature

The neighbor-based smoothing is inspired by the Macroscopic Fundamental
Diagram (MFD) / Network Fundamental Diagram (NFD) literature (Geroliminis &
Daganzo, 2008). The MFD insight — that network-level speed is a well-defined
function of network-level density, even when individual links vary widely —
motivates smoothing in general. The neighbor-based approach applies this
principle locally along the network graph rather than within arbitrary spatial
zones, which better respects the directional structure of traffic flow.

---

## 6  Temporal model: discrete time slices

### 6.1  Slice lifecycle (overview)

The assignment processes all demand across all time bins simultaneously, then
distributes flow to bins via travel-time offsets (see §6.3 for details).

```
For each outer iteration:

  1. ROUTE      Route all OD pairs on current (frozen) network costs
               Request annotations=["nodes", "distance", "duration"]
  2. DISTRIBUTE Assign fractional flow to (link, bin) pairs via time offsets
  3. PER-BIN    For each bin: aggregate flow → density → smooth → VDF → speeds
  4. CSV        Write updated speeds to /dev/shm/ (§2.6)
  5. CUSTOMIZE  Run osrm.customize() with segment-speed-file
  6. RELOAD     Refresh engine (Strategy A or B, §2.5)
  7. (optional) CONVERGE  Repeat until flow changes fall below threshold
```

### 6.2  Slice width

Recommended starting point: **15 minutes**. This balances:

- Temporal resolution (captures peak spreading).
- Computational cost (one customize + reload per slice).
- Statistical stability (enough demand per slice for meaningful flows).

Sensitivity analysis on slice width is a key validation task.

### 6.3  Multi-bin trips: fractional link loading

Vehicles that depart in slice t may traverse links that fall in slices
t, t+1, t+2, etc. The design handles this via **travel-time offset loading**:
each link on a route is assigned to the time bin when the vehicle would
actually be traversing it, based on cumulative travel time from departure.

#### Algorithm

OSRM returns per-link `duration` (seconds) in route annotations. For a trip
departing at time t_dep with route links [e₁, e₂, ..., e_n]:

```
cum_time = 0
for each link e_i with duration d_i:
    enter_time = t_dep + cum_time
    exit_time  = t_dep + cum_time + d_i
    bin_enter  = floor(enter_time / Δt)
    bin_exit   = floor(exit_time / Δt)

    if bin_enter == bin_exit:
        # Entire link traversal within one bin
        load link e_i into bin bin_enter with full weight
    else:
        # Link traversal spans bin boundary — split proportionally
        for bin_k in range(bin_enter, bin_exit + 1):
            bin_start = bin_k * Δt
            bin_end   = (bin_k + 1) * Δt
            overlap   = min(exit_time, bin_end) - max(enter_time, bin_start)
            fraction  = overlap / d_i
            load link e_i into bin bin_k with weight = fraction

    cum_time += d_i
```

This is exact given frozen costs, requires no vehicle state tracking,
and uses only the per-link duration annotations OSRM already provides.

#### Flow interpretation

Each link-bin pair accumulates a fractional vehicle count. The flow rate for
link e in bin t is:

```
q_e,t = (Σ fractional vehicles on e in bin t) / Δt    [veh/hr]
```

This means a 45-minute trip across a 15-minute bin structure correctly loads
links in 3 different bins proportional to time spent, rather than dumping
all flow into the departure bin.

#### Vectorized implementation sketch

```python
def distribute_route_to_bins(
    link_durations_s: np.ndarray,   # per-link duration in seconds
    departure_time_s: float,        # seconds from epoch
    bin_width_s: float,             # Δt in seconds
) -> list[tuple[int, int, float]]:
    """Return [(link_idx, bin_idx, fraction), ...] for flow accumulation."""
    assignments = []
    cum = departure_time_s
    for i, d in enumerate(link_durations_s):
        if d <= 0:
            continue
        enter = cum
        exit_ = cum + d
        b_enter = int(enter // bin_width_s)
        b_exit  = int(exit_ // bin_width_s)
        for b in range(b_enter, b_exit + 1):
            bs = b * bin_width_s
            be = (b + 1) * bin_width_s
            overlap = min(exit_, be) - max(enter, bs)
            assignments.append((i, b, overlap / d))
        cum = exit_
    return assignments
```

In production, this inner loop should be vectorized or moved to C++ for
large route sets. The key insight is that no state is maintained between
bins — the route annotations contain all the information needed.

#### Implications for the slice lifecycle

The slice lifecycle (§6.1) changes: instead of processing one bin at a time
independently, **all trips for all bins are routed first** on the current
network state, then flow is distributed across bins via the offset algorithm.
The updated lifecycle becomes:

```
1. ROUTE     Route ALL trips (all departure times) on current network costs
2. DECOMPOSE Extract per-link durations from annotations
3. DISTRIBUTE Assign fractional flow to (link, bin) pairs via time offsets
4. For each bin t = 0, 1, ..., T-1:
   a. AGGREGATE  Sum fractional flows for bin t → q_e,t per link
   b. DENSITY    Convert flow to density (§5.6)
   c. SMOOTH     Neighbor-based smoothing (§5.8)
   d. VDF        Evaluate bi-parabolic → speed per link for bin t
5. WRITE CSV  Write final speeds (e.g., last bin or weighted average) to /dev/shm/
6. CUSTOMIZE  Run osrm.customize()
7. RELOAD     Refresh engine
8. (optional) CONVERGE — repeat from step 1 until gap < ε
```

This is more faithful to dynamic assignment: network conditions in each bin
reflect only the vehicles actually present in that bin, not all vehicles
that departed during it.

### 6.4  Inner convergence loop (optional)

Within the outer loop, a single all-or-nothing loading may not reach
equilibrium. An inner loop blends successive assignments:

```
Repeat (outer iteration n):
  1. Route all demand → paths (with per-link durations)
  2. Distribute flow to (link, bin) pairs via time offsets
  3. Blend with previous iteration (MSA: q_new = q_old + (1/n)(q_aon - q_old))
  4. Per-bin: density → smooth → VDF → speeds
  5. Write CSV to /dev/shm/ → customize → reload
  Until: max |Δq_e| / q_e < ε  or  iteration limit reached
```

**Method of Successive Averages (MSA)** is the recommended first convergence
method. It is simple, well-understood, and sufficient for a prototype. More
sophisticated methods (Frank-Wolfe, path-based algorithms) can be added later.

---

## 7  Demand interfaces

### 7.1  Shared assignment core

Both demand modes feed the same engine:

```
                    ┌──────────────┐     ┌──────────────────┐
                    │  OD Matrix   │     │  Matrix-Free     │
                    │  Adapter     │     │  Trip Adapter    │
                    └──────┬───────┘     └──────┬───────────┘
                           │                     │
                           ▼                     ▼
                    ┌──────────────────────────────┐
                    │     Demand Bucket            │
                    │  [(origin, dest, volume,     │
                    │    departure_time), ...]     │
                    └──────────┬───────────────────┘
                               │
                               ▼
                    ┌──────────────────────────────┐
                    │     Assignment Core          │
                    │  Route → Decompose → Accum   │
                    │  → VDF → CSV → Customize     │
                    └──────────────────────────────┘
```

### 7.2  OD-matrix adapter (first)

Input: a matrix of shape `(n_origins, n_destinations)` with vehicle counts per
time slice. Origins and destinations are coordinates or zone centroids.

```python
# Example API sketch
assigner = osrm.TrafficAssignment(
    base_path="network.osrm",
    algorithm="MLD",
    vdf="bi-parabolic",
    slice_duration_minutes=15,
    density_smoothing="zone",     # "zone", "neighbor", or "none" (§5.8)
    smoothing_lambda=0.6,         # blending parameter
    csv_backend="tmpfs",          # "tmpfs" (/dev/shm/), "disk", or "memory"
    engine_refresh="destroy",     # "destroy" (Strategy A) or "hotswap" (Strategy B, §2.5)
)

# Load demand
assigner.load_od_matrix(
    matrix=demand_array,       # (n_orig, n_dest) or sparse
    origins=origin_coords,     # [(lon, lat), ...]
    destinations=dest_coords,  # [(lon, lat), ...]
    departure_time=0,          # slice index or timestamp
)

# Run assignment
results = assigner.run(
    max_iterations=50,
    convergence_gap=0.01,
)

# Inspect results
results.link_flows       # DataFrame: edge_from, edge_to, flow, density, speed
results.od_paths         # Path assignments per OD pair
results.convergence_log  # Gap metric per iteration
```

### 7.3  Matrix-free trip adapter (second)

Input: a stream of individual trips `(origin, destination, departure_time)`.
Trips are bucketed into time slices by the adapter.

```python
# Example API sketch
assigner.load_trips(
    trips=trip_dataframe,  # columns: origin_lon, origin_lat, dest_lon, dest_lat, departure_time
)
```

The assignment core is identical; only the demand ingestion differs.

---

## 8  Concrete implementation gaps

### 8.1  Must-build wrapper changes

These are required before any assignment work can begin:

| Gap | Location | Work |
|-----|----------|------|
| **Expose `updater_config`** on `CustomizationConfig` | `src/customizerconfig_nb.cpp` | Bind `segment_speed_lookup_paths` and `turn_penalty_lookup_paths` as read-write properties |
| **Python `customize()` must accept speed/penalty file args** | `src/osrm/preprocessing.py` | Forward `segment_speed_file` and `turn_penalty_file` kwargs to `updater_config` paths |
| **Engine destroy-and-reload helper** | `src/osrm/__init__.py` | Add `OSRM.reload()` or document the `del engine; customize(); engine = OSRM(...)` pattern |
| **tmpfs CSV writer** | `src/osrm/assignment.py` | Write segment-speed CSV to `/dev/shm/` for zero-disk-I/O (§2.6) |

### 8.2  New assignment module

A new `src/osrm/assignment.py` (or `src/osrm/assignment/` package) containing:

- `NetworkState` — per-edge state arrays, edge index, flow accumulation
- `BiParabolicVDF` — vectorized VDF evaluation, flow-to-density solver (§5)
- `DensitySmoothing` — zone-based and neighbor-based smoothing (§5.8)
- `AssignmentLoop` — outer time-slice loop, inner convergence loop
- `DemandAdapter` — abstract base, `ODMatrixAdapter`, `TripStreamAdapter`
- `SegmentSpeedWriter` — generates CSV to tmpfs from `NetworkState`

### 8.3  Performance-critical path (candidate for C++ extension)

The flow accumulation step (decompose millions of paths into per-edge flow
increments) is the most likely bottleneck. If Python + NumPy is too slow,
this should be moved to a C++ nanobind extension that:

1. Takes route annotation arrays (node IDs, distances) as input.
2. Performs hash-based edge lookup and atomic flow increment.
3. Returns updated flow array.

This can be added as a new `.cpp` source in the existing build without
touching OSRM core.

---

## 9  Expected performance characteristics

### 9.1  Customize latency

OSRM's MLD customize step is designed for fast updates. Expected latency
(order of magnitude):

| Network size | Customize time | Source |
|-------------|---------------|--------|
| Monaco (~5k edges) | < 1 second | Trivial |
| City (~100k edges) | 2–10 seconds | OSRM wiki estimates |
| Region (~1M edges) | 30–120 seconds | Depends on partition depth |

This is the per-iteration cost. For 50 inner iterations × 16 time slices,
a city-scale network would spend ~30–80 minutes just on customize calls. This
is the primary scalability concern and the strongest argument for eventually
moving to an in-memory weight update mechanism (which would require OSRM-core
changes).

### 9.2  Routing throughput

OSRM routing with GIL release and thread pool:

- Single query: ~1–5 ms (city scale)
- Bulk parallel: ~50k–200k routes/second (depending on path length and cores)
- Table queries: much faster for dense OD matrices

### 9.3  Memory footprint

- OSRM engine: ~1–4 GB for a large metro area
- NetworkState: ~100 bytes/edge × 1M edges = ~100 MB
- Path storage (if retained): potentially large; may need streaming

---

## 10  Implementation phases

### Phase 1: Expose traffic update surface

**Deliverable**: `osrm.customize()` accepts `segment_speed_file` and
`turn_penalty_file` arguments, forwarded to OSRM's `UpdaterConfig`.

**Scope**:
- Modify `src/customizerconfig_nb.cpp` to bind `updater_config` sub-fields
- Modify `src/osrm/preprocessing.py` `customize()` to accept and forward kwargs
- Add integration test: extract → partition → customize with speed file → route
  and verify changed travel times

### Phase 2: Network state and VDF

**Deliverable**: A `NetworkState` class that can be populated from OSRM route
annotations, a `BiParabolicVDF` (Fournier et al. Eq. 11) that evaluates
vectorized speed from density, and a `DensitySmoothing` module for mesoscopic
spatial averaging.

**Scope**:
- `NetworkState`: edge registry, flow accumulation, density conversion
- `BiParabolicVDF`: vectorized NumPy implementation (§5.2–5.7)
- `DensitySmoothing`: zone-based and neighbor-based smoothing (§5.8)
- `SegmentSpeedWriter`: generate CSV to tmpfs `/dev/shm/` (§2.6)
- Unit tests on toy networks with known analytical solutions

### Phase 3: Assignment loop (OD-matrix)

**Deliverable**: End-to-end assignment on Monaco with OD-matrix input,
discrete time slices, MSA convergence, and link-flow output.

**Scope**:
- `AssignmentLoop` orchestrator
- `ODMatrixAdapter` demand input
- Convergence reporting and diagnostics
- Integration test on Monaco

### Phase 4: Matrix-free adapter and validation

**Deliverable**: Trip-stream demand input on the same assignment core.
Validation against known equilibrium solutions on toy networks.

**Scope**:
- `TripStreamAdapter`
- Braess network validation (known UE solution)
- Sioux Falls benchmark (if feasible at this scale)

### Phase 5: Performance optimization

**Deliverable**: Profiling-driven optimization of the hot path.

**Scope**:
- Benchmark customize latency at scale (tmpfs CSV vs. disk, §2.6)
- C++ flow-accumulation extension if Python is bottleneck
- Evaluate shared-memory hot-swap (§2.5 Strategy B) — wrap
  `storage::Storage::Run()` or use subprocess `osrm-datastore`
- Evaluate feasibility of in-memory `LookupTable` bypass (skip CSV entirely)
- Evaluate feasibility of direct weight injection (OSRM-core spike)

---

## 11  Risk register

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| **Customize latency dominates runtime** at metro scale | High | High | Profile early on realistic networks; use tmpfs CSV (§2.6) to eliminate I/O; consider in-memory weight mutation as Phase 5 escalation |
| **OSM lacks lane/capacity data** for many links | Medium | High | Ship sensible defaults by road class; allow user overrides via enrichment CSV |
| **Engine re-instantiation has hidden side effects** (TBB thread pool, memory leaks) | Medium | Medium | py-osrm already has a TBB cleanup handler; test repeated create/destroy cycles; migrate to shared-memory hot-swap (§2.5 Strategy B) if problematic |
| **Flow-to-density solver diverges** for edge cases | Low | Medium | Clamp density to [0, k_j]; use robust Newton with bisection fallback |
| **Path decomposition is too slow in Python** for large networks | Medium | Medium | Move to C++ extension; the nanobind build system already supports adding new .cpp sources |
| **Bi-parabolic VDF produces unrealistic speeds** on certain link types | Medium | Low | Validate against observed speed-flow data; allow per-link VDF parameter overrides |
| **OSRM upstream changes break FetchContent build** | Low | Low | Pin to v6.0.0; upgrade deliberately |
| **Neighbor-based smoothing over-diffuses** on sparse networks | Medium | Medium | Cap passes at 2; expose β as tunable; validate against known congestion patterns |
| **Shared-memory hot-swap requires external osrm-datastore process** | Low | Medium | Prototype with Strategy A (destroy/recreate); add programmatic `storage::Storage::Run()` binding later |
| **Short-link density volatility** despite smoothing | Medium | Medium | Minimum link-length filter; merge very short links into preceding link for assignment purposes |

---

## 12  Open questions

1. **Slice width**: What is the right default? 15 minutes is proposed; should
   it be configurable per study?

2. **Inner convergence method**: MSA is proposed for simplicity. Should
   Frank-Wolfe or path-based equilibrium be planned from the start?

3. **Carry-over policy**: Should residual in-transit vehicles be tracked
   between slices from day one, or is the "all trips complete within-slice"
   approximation acceptable for the first prototype?

4. **External network enrichment**: What format should users provide
   supplementary link attributes (lanes, capacity, jam density) in? A CSV
   keyed on OSM node pairs? A GeoPackage?

5. **Oversaturation handling**: Simple speed floor, or should the prototype
   include basic queue tracking?

6. **Multi-class assignment**: Should the design anticipate multiple vehicle
   classes (car, truck, transit) from the start, or defer to a later phase?

7. **Density smoothing calibration**: What β (neighbor smoothing) value and
   how many passes? Should multi-pass be the default, or single-pass with
   higher β?

8. **Hot-swap vs. destroy/recreate**: Should the prototype invest in wrapping
   `storage::Storage::Run()` for programmatic hot-swap, or is subprocess
   `python -m osrm datastore` sufficient?

---

## Appendix A: Key file references

| File | Role |
|------|------|
| `src/osrm_nb.cpp` | Main OSRM class binding (Route, Table, etc.) |
| `src/customizerconfig_nb.cpp` | CustomizationConfig binding (**needs extension**) |
| `src/osrm/preprocessing.py` | Python customize() wrapper (**needs extension**) |
| `src/osrm/bulk.py` | Bulk parallel routing (reusable for assignment) |
| `src/osrm/__init__.py` | Python OSRM wrapper class |
| `CMakeLists.txt` | Build config — links `osrm_customize` library |
| OSRM `include/updater/updater_config.hpp` | UpdaterConfig with speed/penalty paths |
| OSRM `include/updater/source.hpp` | Segment/Turn/SpeedSource/PenaltySource structs |
| OSRM `include/updater/csv_file_parser.hpp` | CSV parser — uses `mapped_file_source` (§2.6) |
| OSRM `src/updater/updater.cpp` | CSV read → edge weight update logic |
| OSRM `src/customize/customizer.cpp` | Customizer::Run() — calls Updater then recomputes metrics |
| OSRM `include/engine/data_watchdog.hpp` | DataWatchdog for shared-memory hot-swap (§2.5) |
| OSRM `include/engine/datafacade_provider.hpp` | WatchingProvider / ImmutableProvider |
| `docs/Fournier_Ped_Transit_priority_manuscript_v4.pdf` | Bi-parabolic VDF derivation (Eq. 11) |

## Appendix B: Bi-parabolic VDF reference

> Source: Fournier et al., "Pedestrian and Transit Priority Zoning."
> See `docs/Fournier_Ped_Transit_priority_manuscript_v4.pdf`.

Full equations for copy-paste implementation:

```
Given: v_f (km/h), k_j (veh/km), k_c (veh/km, default k_j/3)

q_c = v_f · k_c / 2                               # capacity flow
v_c = v_f / 2                                   # critical speed

Uncongested flow-density (0 ≤ k ≤ k_c):
  q(k) = q_c · k · (2k_c − k) / k_c²

Congested flow-density (k_c < k ≤ k_j):
  q(k) = q_c · [1 − (k − k_c)² / (k_j − k_c)²]

Uncongested speed-density:
  v(k) = q_c · (2k_c − k) / k_c²              (linear in k)

Congested speed-density:
  v(k) = q_c · [1 − (k − k_c)² / (k_j − k_c)²] / k

Travel time on edge of length L:
  t(k) = L / v(k) = L · k / q(k)

Closed-form flow-to-density inverse (uncongested):
  k(q) = k_c · (1 − √(1 − q/q_c))

Closed-form flow-to-density inverse (congested):
  k(q) = k_c + (k_j − k_c) · √(1 − q/q_c)
```

Mesoscopic density smoothing (see §5.8):

```
Neighbor-based (default):
  k̃_e = (1 − β) · k_e + β · Σ_{j ∈ N(e)} w_ej · k_j
  v_e = VDF(k̃_e)
  Default: β = 0.3, passes = 1

H3 zone-based (fallback, for diagnostics):
  K_z = Σ_{e ∈ z} (k_e · L_e) / Σ_{e ∈ z} L_e
  V_z = VDF(K_z)
  v_e = λ · VDF(k_e) + (1 − λ) · v_f,e · (V_z / v_f,z)
```
