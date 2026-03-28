# Traffic Assignment Design Document

> **Status**: Design / feasibility — not yet implemented  
> **Goal**: Extend py-osrm into a dynamic traffic assignment platform leveraging OSRM's computational performance  
> **Long-term benchmark**: Supplant commercial tools such as Bentley OpenPaths for activity-based and OD-based assignment workflows

---

## 1  Executive summary

Traffic assignment requires iterative, demand-responsive network loading where
link costs change as vehicles are assigned. This document defines a
**DTA framework built on OSRM** with a targeted core patch that adds
**multi-period metric storage** — enabling time-dependent routing at full
OSRM query speed without requiring a second routing engine.

The framework supports **progressive capability milestones**:

1. **OSRM static assignment** — single period, state-scale, fast. Proves the
   VDF/density model. Uses MLD customize to update edge weights between
   iterations.
2. **OSRM multi-period DTA** — user-defined time periods (2–96), pre-customized
   per outer iteration. Queries select the correct metric set via a
   `departure_period` parameter. Full OSRM routing speed on all periods
   simultaneously.

### Core innovation: multi-period metric sets in OSRM

OSRM's MLD algorithm already stores **multiple metric sets** internally (one
per vehicle-class "exclude" profile). The `DataFacadeFactory` holds a
`vector<Facade>` and selects the right one per query. We extend this pattern
to add a **period dimension**: N user-defined time periods, each with its own
pre-customized cell metrics. At query time, `departure_period=k` selects
metric set `k`. The MLD search algorithm is unchanged.

This is a surgical C++ patch (~300 lines across ~6 OSRM files) that:
- Eliminates the need for Valhalla or any second engine
- Provides full OSRM query speed (2000–5000 QPS) on all periods
- Supports concurrent queries across different periods (thread-safe)
- Is potentially PR-able to OSRM upstream as a generally useful
  "time-of-day routing profiles" feature

### Architectural decisions locked in

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Routing engine** | **OSRM with multi-period patch** | Full query speed; surgical patch extends existing exclude-class mechanism; avoids second engine dependency |
| **Time periods** | **User-defined, N arbitrary** | Peak/offpeak (N=2), hourly (N=24), 15-min (N=96), custom — memory is the only constraint |
| First demand interface | **OD-matrix** | Easier to validate; matrix-free adapter comes second on the same core |
| Timing model | **Discrete time slices** | Frozen costs inside each slice; engine refresh between slices |
| VDF family | **Bi-parabolic flow-density** (Fournier et al.) | Parameter-light (v_f, k_j); grounded in fundamental diagram; closed-form inverse; avoids BPR |
| Density model | **Mesoscopic with spatial smoothing** (§6.8) | Neighbor-based smoothing avoids short-link volatility; handles spillover |
| CSV I/O (OSRM) | **tmpfs (`/dev/shm/`)** for prototype (§2.6) | Zero disk I/O; no OSRM changes needed; future: in-memory bypass |
| Engine refresh (OSRM) | **Destroy/recreate** (prototype), **shared-memory hot-swap** (production, §2.5) | Hot-swap is zero-downtime but requires `osrm-datastore` integration |
| Assignment logic location | **Wrapper side** (Python + C++ extension) | All assignment logic (VDF, density, smoothing, convergence) in py-osrm; OSRM core only gets the multi-metric patch |

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

## 3  OSRM multi-period routing

The central technical contribution: extending OSRM's MLD algorithm to store
and select among **N user-defined metric sets** at query time, enabling
time-dependent routing without a second engine.

### 3.1  How OSRM already handles multiple metric sets

OSRM's MLD engine already supports multiple metric sets via the "exclude
class" mechanism (vehicle-class restrictions). The architecture:

```
DataFacadeFactory
  └── vector<shared_ptr<const Facade>> facades   // one per exclude class
  └── Get(BaseParameters) → facades[exclude_index]
```

Each facade holds its own `CellMetricView` — a set of pre-computed cell
shortcut weights, durations, and distances. At query time, the factory
selects a facade based on `BaseParameters::exclude`. The MLD search
algorithm is oblivious — it just calls `GetCellMetric()` and gets the
right view.

**Key files in OSRM core:**

| File | What it does |
|------|--------------|
| `include/engine/datafacade_factory.hpp` | `DataFacadeFactory` — stores `vector<Facade>`, selects per query |
| `include/customizer/cell_metric.hpp` | `CellMetric` struct: `{weights[], durations[], distances[]}` |
| `include/customizer/files.hpp` | Read/write `.osrm.cell_metrics` — TAR with `/mld/metrics/{name}/exclude/{N}/` |
| `include/engine/datafacade/contiguous_internalmem_datafacade.hpp` | `GetCellMetric()` — returns one `CellMetricView` |
| `include/engine/routing_algorithms/routing_base_mld.hpp` | MLD search — calls `GetCellMetric()` in `relaxOutgoingEdges()` |
| `include/engine/api/route_parameters.hpp` | `RouteParameters` — no time fields currently |

### 3.2  What changes with `customize`

Only a subset of OSRM's data files change when edge weights are updated.
Everything else is structurally static:

| File | Changes? | Content |
|------|----------|---------|
| `.osrm.cell_metrics` | ✅ Yes | Cell shortcut weights/durations/distances |
| `.osrm.mldgr` | ✅ Yes | Multi-level graph with shortcut weights |
| `.osrm.geometry` | ✅ Yes | Per-edge durations/weights for annotations |
| `.osrm.turn_*_penalties` | ✅ Yes | Turn penalty weights |
| `.osrm.ebg`, `.osrm.partition`, `.osrm.cells` | ❌ No | Graph topology, cell structure |
| Everything else | ❌ No | Names, coordinates, indexes |

For Monaco: ~1.0 MB changes per period vs ~1.4 MB shared (~42% vs ~58%).

### 3.3  The multi-period patch

**Concept:** Add a `period` dimension parallel to the existing `exclude`
dimension. Store N × E metric sets (N periods × E exclude classes). Query
with `departure_period=k` or `departure_time=T` → facade factory selects
metric set `k`.

**Patch scope** (~350 lines across ~7 files):

1. **`include/engine/api/base_parameters.hpp`**
   - Add `std::optional<unsigned> departure_period` — explicit period index
   - Add `std::optional<double> departure_time` — epoch seconds, auto-mapped

2. **`include/engine/datafacade_factory.hpp`**
   - Extend facade vector: `facades[period * num_excludes + exclude_index]`
   - Period resolution logic (in priority order):
     ```cpp
     if (params.departure_period)
         period = *params.departure_period;        // explicit index
     else if (params.departure_time && period_schedule)
         period = period_schedule->resolve(*params.departure_time);
     else
         period = 0;                               // freeflow default
     ```
   - Load period schedule from data index at construction time

3. **`include/customizer/files.hpp`**
   - Extended TAR paths: `/mld/metrics/{name}/period/{P}/exclude/{E}/`
   - Backward-compatible: if no period entries, treat as single period

4. **`src/customize/customizer.cpp`**
   - Accept a `period_index` parameter (default 0)
   - Write metrics under the period-indexed path

5. **`include/engine/datafacade/contiguous_internalmem_datafacade.hpp`**
   - Load N period metric sets during initialization
   - `GetCellMetric()` returns the period-appropriate view (selected by
     the facade factory, not the search algorithm)

6. **Period schedule file** (new, small)
   - Stored alongside OSRM data as `.osrm.period_schedule`
   - Simple format: list of `(period_index, start_epoch, end_epoch)` tuples
   - Supports arbitrary period boundaries (hourly, custom, weekday/weekend)
   - Optional: if absent, only explicit `departure_period` is available
   - Written by the assignment tool or by hand for production routing

7. **`include/engine/routing_algorithms/routing_base_mld.hpp`**
   - **No changes.** The search algorithm still calls `GetCellMetric()` and
     gets whichever view the facade was initialized with.

**The MLD search algorithm is completely unchanged.** Period selection
happens at the facade factory level, identically to how exclude classes
work today.

**Two query modes serve different use cases:**

| Mode | Parameter | Use case |
|------|-----------|----------|
| Explicit index | `departure_period=3` | Assignment loop (wrapper controls periods) |
| Timestamp | `departure_time=1711648200` | Matrix-free ABM, production routing, navigation |

For the matrix-free travel model, an activity-based model feeds individual
trips with real departure timestamps. OSRM automatically routes each trip
on the correct period's congestion state — no period mapping in the
application layer. The same engine instance serves all time periods
concurrently.

### 3.4  Bidirectional search correctness and cross-period trips

OSRM's MLD uses true bidirectional Dijkstra (forward + backward heaps).
For time-dependent routing, backward search is theoretically invalid
because arrival time is unknown a priori.

**With frozen costs per period, this is not a problem.** Both forward
and backward search use the same static weights within a given period.
The approximation is only that a route spanning two periods uses the
departure period's weights throughout. For trips shorter than the period
duration, this is exact.

This is the same approximation used by every discrete-time DTA model
(TRANSIMS, DTALite, etc.) — costs are frozen within each time slice.

**Route finding vs. flow accumulation for cross-period trips:**

There are two distinct concerns when a trip spans multiple periods:

1. **Route choice** (which path?): Uses the departure period's weights
   only. The entire route is computed on one metric set. This is the
   frozen-cost approximation — the route does not anticipate congestion
   changes in later periods.

2. **Flow impact** (which periods does the vehicle congest?): Handled
   correctly by fractional link loading (§7.3). As the vehicle traverses
   its route, each link's flow is assigned to the period when the vehicle
   would actually be there, based on cumulative travel time from departure.

Over multiple outer iterations, this self-corrects: if a route was
suboptimal because a later period had worse congestion than the departure
period assumed, the next iteration's later-period weights will reflect
that added flow, and some trips will reroute.

**Approximation quality depends on period width vs. trip duration.** With
1-hour periods, a typical 30-minute urban commute completes within one
period (exact). A 90-minute cross-regional trip spans ~2 periods, with
only the tail portion subject to the approximation. See §7.2 for period
width calibration guidance.

### 3.5  User-defined period configurations

OSRM sees only indices `0..N-1`. All temporal semantics — what each period
represents, how departure times map to period indices — live in the Python
wrapper:

```python
class PeriodConfig:
    """Maps wall-clock time to OSRM period indices."""

    @staticmethod
    def peak_offpeak():
        """N=2: AM/PM peak vs everything else."""
        return PeriodConfig(breaks_h=[6, 9, 15, 19],
                            labels=["offpeak", "am_peak", "midday",
                                    "pm_peak", "evening"])

    @staticmethod
    def uniform(minutes=15):
        """N=1440/minutes: uniform bins across 24 hours."""
        n = 1440 // minutes
        return PeriodConfig.from_bin_count(n)

    @staticmethod
    def custom(specs):
        """Arbitrary user-defined periods.

        specs: list of (label, time_range) tuples
        e.g., [("weekday_am", "07:00-09:00"),
               ("weekend",    "Sat-Sun 00:00-24:00")]
        """
        ...

    def time_to_period(self, departure_time) -> int:
        """Map wall-clock time to period index."""
        ...
```

The same `PeriodConfig` drives both the customize step (which period's
speed CSV to use) and the query step (which `departure_period` to pass).

### 3.6  Sparse cell-delta storage

A naive multi-period implementation stores N full copies of all cell
shortcut tables. But most cells — those containing only local/residential
roads — have identical shortcuts across all periods. Only cells containing
congested arterials or highways produce different shortcut weights under
peak conditions.

**How MLD cells work:**

OSRM partitions the road network into a hierarchy of cells — nested
geographic regions:

- **Level 1 cells**: Small clusters (~50–500 nodes, a few city blocks).
- **Level 2+ cells**: Each higher level groups several lower-level cells
  (neighborhoods → districts → regions).

For each cell, OSRM pre-computes **shortcut weights** between all
**boundary nodes** — the nodes where roads cross cell boundaries. If a
cell has 10 entry points and 8 exit points, it stores a 10×8 matrix of
shortest-path costs through that cell.

At query time, the MLD search **never enters cells** at levels 1+. It
hops between boundary nodes using the pre-computed shortcuts. Detailed
node-by-node search only happens at level 0 near the origin and
destination.

**The sparse optimization:**

Instead of N full copies of every cell's shortcut table, store:
- **One base (freeflow) metric set** — full, shared across all periods
- **Per period**: a bitset of which cells differ + only those cells' shortcuts

```
Cell 17 (residential only):
  Freeflow: entry A → exit B = 45s     ← stored once
  AM peak:  entry A → exit B = 45s     ← bitset says "use base"
  PM peak:  entry A → exit B = 45s     ← bitset says "use base"

Cell 22 (contains I-580 interchange):
  Freeflow: entry A → exit B = 30s     ← stored
  AM peak:  entry A → exit B = 90s     ← bitset says "use override"
  PM peak:  entry A → exit B = 75s     ← stored as override
```

At query time, the search reaches a cell boundary and needs the shortcut
cost. One bitset check determines which table to read from:

```cpp
const auto &metric = cell_has_override.test(cell_id)
    ? period_overrides[period].GetCell(cell_id)
    : base_metrics.GetCell(cell_id);
```

**Performance impact: effectively zero.** This check happens once per cell
traversal — about 10–50 times per route. A bitset test is a single AND
instruction hitting L1 cache (~1 ns). On a route taking ~0.5 ms, this is
unmeasurable. The shortcut weight arrays accessed afterward remain
contiguous and cache-friendly regardless of which copy is used.

**The intra-cell edge weights** (mldgr, used for level-0 search near
origin/destination) still need full per-period storage, since we can't
predict which cells will contain trip endpoints. However, the mldgr is
smaller than cell metrics for large networks — cell metrics store
precomputed shortcut matrices across all hierarchical levels, while
the mldgr stores raw edge weights.

**Geometry** (per-edge durations used for route annotations) does not need
per-period storage. It is only read during route unpacking after the
search completes, and the correct durations can be reconstructed from the
chosen route's edge weights on the fly.

**Impact on preprocessing:**

Only the `customize` step changes — `extract` and `partition` are
untouched (cell boundaries are topological, independent of weights).

Current customize pipeline:
1. Read segment-speed CSV
2. Update edge weights
3. Recompute all cell shortcut tables
4. Write `.osrm.cell_metrics`

Multi-period customize with sparse output:
1. Compute freeflow shortcuts (period 0) — full, as today
2. For each additional period:
   a. Apply that period's segment-speed CSV
   b. Recompute cell shortcuts
   c. **Diff against freeflow** — compare shortcut arrays per cell
   d. Store only cells that produced different results + the bitset
3. Write base metrics + per-period sparse overrides

The diff detection is cheap: an element-wise comparison of the shortcut
weight arrays after each cell is processed. The core customize algorithm
(recomputing boundary-to-boundary shortest paths within cells) is
identical — it just runs N times and records which cells changed.

### 3.7  Memory and runtime costs

**Naive (full copy) memory per period:**

| Network | Total OSRM data | Per-period overhead (~42%) | Shared (~58%) |
|---------|-----------------|---------------------------|---------------|
| Monaco (~5K edges) | ~2.4 MB | ~1.0 MB | ~1.4 MB |
| Metro (~500K edges) | ~400 MB | ~170 MB | ~230 MB |
| California (~10M edges) | ~4 GB | ~1.7 GB | ~2.3 GB |

**With sparse cell-delta storage:**

Assuming ~15–25% of cells contain congested edges during peak periods
(arterials and highways only), cell metric overhead drops proportionally.
The mldgr (intra-cell edge weights) remains full-copy but is a smaller
fraction of total per-period data.

| Periods | Use case | Naive (CA) | Sparse (CA, ~20% cells) | Hardware |
|---------|----------|------------|-------------------------|----------|
| 2 | Peak/offpeak | ~3.4 GB | ~1.0 GB | Laptop |
| 4–6 | AM/midday/PM/evening | 7–10 GB | ~2–3 GB | Laptop/workstation |
| 24 | Hourly | ~41 GB | ~10–12 GB | Workstation |
| 96 | 15-minute bins | ~163 GB | ~35–45 GB | Server (64 GB) |

The sparse approach makes 24 periods workstation-feasible and 96 periods
server-feasible, where the naive approach would require HPC resources.

**Customize cost (pre-computation per outer iteration):**

```
N periods × customize_latency (once per outer iteration)

 2 periods × 5 min (CA) =  10 min/iter → 20 iters = 3.3 hours
 6 periods × 5 min (CA) =  30 min/iter → 20 iters = 10 hours
24 periods × 5 min (CA) = 120 min/iter → 20 iters = 40 hours

Note: period customizations are independent — parallelizable across cores.
With 6 cores: 6 periods × 5 min = 5 min/iter → 20 iters = 1.7 hours
```

Once customized, **all routing is at full OSRM speed (2000–5000 QPS/core)
with instant period selection.** Queries for different periods can run
concurrently — each facade is immutable and thread-safe.

### 3.8  Assignment loop with multi-period routing

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

### 3.9  Valhalla as optional alternative

Valhalla remains an option if OSRM's coarse-period model proves
insufficient for a specific use case. Valhalla provides native
`date_time` routing with per-edge speed profiles indexed by time-of-week,
at the cost of ~5–10× slower per-query routing (100–500 QPS vs 2000–5000).

The assignment core (VDF, density, smoothing, convergence) is designed to
be engine-independent, so a `ValhallaBackend` could be added as a future
extension without restructuring. However, with the multi-period OSRM
patch, Valhalla is no longer a required dependency for DTA.

---

## 4  Path-to-link accounting

Route annotations are the bridge between OSRM's path output and the assignment
engine's link-level state.

### 4.1  What OSRM returns with `annotations=["nodes", "distance", "speed"]`

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

### 4.2  Edge identity

An **assignable edge** is defined by a directed pair of consecutive OSM node
IDs from the `nodes` annotation:

```
nodes = [n0, n1, n2, n3]
edges = [(n0, n1), (n1, n2), (n2, n3)]
```

These pairs map **directly** to the segment-speed CSV format. This is the
critical property that makes the entire design work without OSRM-core changes.

### 4.3  Per-edge data available from annotations

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

## 5  Network state model

### 5.1  Link state variables

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

### 5.2  Default assumptions for missing data

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

### 5.3  Data structure sketch

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

### 5.4  Flow accumulation semantics: why occupancy is already correct

A common concern: if a trip traverses links A → B → C, does the vehicle
contribute to density on all three links simultaneously? The answer is **no**
— the standard flow-to-density conversion handles this automatically.

**Flow** is a rate: vehicles passing a point per unit time.

```
q_e = (number of trips using link e in bin t) / Δt    [veh/hr]
```

**Density** is derived from flow via the fundamental relationship:

```
k_e = q_e / v_e
```

The division by v_e is critical: it converts a *passage count rate* into an
*average spatial occupancy*. A vehicle traveling at 60 km/h on a 1 km link
occupies it for 1 minute. In a 15-minute bin, its average contribution to
the link's instantaneous vehicle count is 1/15. The math confirms:

```
q = 1 trip / 0.25 hr = 4 veh/hr
k = q / v = 4 / 60 = 1/15 veh/km  ← matches 1 vehicle × (1 min / 15 min) / 1 km
```

Slow links naturally get higher density contributions per vehicle (longer
dwell time), fast links get lower. **No additional normalization is needed.**

> **Implementation warning**: Do NOT accumulate density directly as vehicle
> counts per link. Always accumulate **flow** (trips/Δt), then convert to
> density via the VDF's inverse (§6.6) or the fundamental relationship.
> Direct count-based density would overcount by treating each vehicle as
> simultaneously present on every link of its route.

---

## 6  Bi-parabolic flow-density VDF

> **Reference**: The bi-parabolic formulation used here follows Fournier et al.,
> "Pedestrian and Transit Priority Zoning."
> See `docs/Fournier_Ped_Transit_priority_manuscript_v4.pdf` for full
> derivations and default parameter recommendations.

### 6.1  Why not BPR?

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
MFD-based smoothing (see §6.8).

### 6.2  Functional form (from Fournier et al.)

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

### 6.3  Parameter relationships

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

### 6.4  Default parameters (from Fournier et al.)

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

### 6.5  From density to OSRM weights

The VDF produces speed v(k) in km/h. OSRM's segment-speed CSV takes speed in
km/h. The mapping is therefore direct:

```
density k_e  →  v(k_e) (km/h)  →  CSV line: "from,to,v_e"

Uncongested: v_e = q_c · (2k_c − k_e) / k_c²
Congested:   v_e = q_c · [1 − (k_e − k_c)² / (k_j − k_c)²] / k_e
```

OSRM internally converts speed to duration: `duration = distance / (speed / 3.6)`

### 6.6  From flow to density (closed-form inverse)

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
- **If q_e > q_c**: the link is oversaturated. See §6.7.

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

### 6.7  Oversaturation policy

When assigned flow exceeds capacity, physical queuing occurs. For the first
prototype, we use a simple penalty:

- Clamp density at k_j (speed → 0 is not useful).
- Instead, set speed to a configurable floor (e.g., 5 km/h).
- Optionally report oversaturated links for diagnostics.

Spillback modeling (queues propagating upstream) is deferred but partially
addressed by the mesoscopic density smoothing in §6.8.

### 6.8  Mesoscopic density model: spatial smoothing

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

## 7  Temporal model: discrete time slices

### 7.1  Slice lifecycle (overview)

The assignment processes all demand across all time bins simultaneously, then
distributes flow to bins via travel-time offsets (see §7.3 for details).

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

### 7.2  Slice width

Recommended starting point: **1 hour**. This balances:

- **Frozen-cost accuracy**: Most urban trips (< 45 min) complete within one
  period, making the departure-period route choice exact. Only long
  cross-regional trips span multiple periods (see §3.4 for implications).
- **Computational cost**: Fewer periods = fewer customize passes per
  iteration. 24 hourly periods vs. 96 quarter-hour periods is a 4× savings
  in customize wall-clock (or cores needed for parallel customize).
- **Memory**: Per-period overhead scales linearly with N. At 24 hourly
  periods with sparse cell-delta storage (§3.6), California fits in ~10–12
  GB vs. ~35–45 GB for 96 bins.
- **Statistical stability**: Longer bins aggregate more trips, producing
  smoother flow estimates and better-conditioned density calculations.

**Calibration guidance:** The period width should cover the majority of trip
durations in the study area. Key metric: **what fraction of trips complete
within one period?**

| Period width | Trips within 1 period (typical metro) | N (24h) | Customize cost (CA, 6 cores) |
|---|---|---|---|
| 15 min | ~50–60% | 96 | ~80 min/iter |
| 30 min | ~75–85% | 48 | ~40 min/iter |
| **1 hour** | **~90–95%** | **24** | **~20 min/iter** |
| 2 hours | ~98% | 12 | ~10 min/iter |
| 3 hours | ~99% | 8 | ~7 min/iter |

For most DTA studies, 1-hour bins provide an excellent tradeoff. For
detailed peak-spreading analysis, 30-minute bins may be warranted. The
period width is user-configurable via `PeriodConfig` (§3.5).

Sensitivity analysis on period width is a key validation task.

### 7.3  Multi-bin trips: fractional link loading

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

The slice lifecycle (§7.1) changes: instead of processing one bin at a time
independently, **all trips for all bins are routed first** on the current
network state, then flow is distributed across bins via the offset algorithm.
The updated lifecycle becomes:

```
1. ROUTE     Route ALL trips (all departure times) on current network costs
2. DECOMPOSE Extract per-link durations from annotations
3. DISTRIBUTE Assign fractional flow to (link, bin) pairs via time offsets
4. For each bin t = 0, 1, ..., T-1:
   a. AGGREGATE  Sum fractional flows for bin t → q_e,t per link
   b. DENSITY    Convert flow to density (§6.6)
   c. SMOOTH     Neighbor-based smoothing (§6.8)
   d. VDF        Evaluate bi-parabolic → speed per link for bin t
5. WRITE CSV  Write final speeds (e.g., last bin or weighted average) to /dev/shm/
6. CUSTOMIZE  Run osrm.customize()
7. RELOAD     Refresh engine
8. (optional) CONVERGE — repeat from step 1 until gap < ε
```

This is more faithful to dynamic assignment: network conditions in each bin
reflect only the vehicles actually present in that bin, not all vehicles
that departed during it.

### 7.4  Inner convergence loop (optional)

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

## 8  Demand interfaces

### 8.1  Shared assignment core

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

### 8.2  OD-matrix adapter (first)

Input: a matrix of shape `(n_origins, n_destinations)` with vehicle counts per
time slice. Origins and destinations are coordinates or zone centroids.

```python
# Example API sketch
assigner = osrm.TrafficAssignment(
    base_path="network.osrm",
    algorithm="MLD",
    vdf="bi-parabolic",
    slice_duration_minutes=15,
    density_smoothing="zone",     # "zone", "neighbor", or "none" (§6.8)
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

### 8.3  Matrix-free trip adapter (second)

Input: a stream of individual trips `(origin, destination, departure_time)`.
Trips are bucketed into time slices by the adapter.

```python
# Example API sketch
assigner.load_trips(
    trips=trip_dataframe,  # columns: origin_lon, origin_lat, dest_lon, dest_lat, departure_time
)
```

The assignment core is identical; only the demand ingestion differs.

With the `departure_time` parameter in the OSRM patch (§3.3), the
matrix-free adapter becomes especially clean: each trip's real departure
timestamp is passed directly to OSRM, which automatically selects the
correct period's congestion state. No period bucketing is needed in the
application layer — OSRM's period schedule handles the mapping internally.

This is the ultimate target for activity-based models: individual agents
with continuous departure times, routed on time-appropriate congestion,
with flow feedback updating congestion for the next iteration. The same
OSRM engine instance serves all time periods concurrently.

---

## 9  Concrete implementation gaps

### 9.1  Must-build wrapper changes

These are required before any assignment work can begin:

| Gap | Location | Work |
|-----|----------|------|
| **Expose `updater_config`** on `CustomizationConfig` | `src/customizerconfig_nb.cpp` | Bind `segment_speed_lookup_paths` and `turn_penalty_lookup_paths` as read-write properties |
| **Python `customize()` must accept speed/penalty file args** | `src/osrm/preprocessing.py` | Forward `segment_speed_file` and `turn_penalty_file` kwargs to `updater_config` paths |
| **Engine destroy-and-reload helper** | `src/osrm/__init__.py` | Add `OSRM.reload()` or document the `del engine; customize(); engine = OSRM(...)` pattern |
| **tmpfs CSV writer** | `src/osrm/assignment.py` | Write segment-speed CSV to `/dev/shm/` for zero-disk-I/O (§2.6) |

### 9.2  New assignment module

A new `src/osrm/assignment.py` (or `src/osrm/assignment/` package) containing:

- `NetworkState` — per-edge state arrays, edge index, flow accumulation
- `BiParabolicVDF` — vectorized VDF evaluation, flow-to-density solver (§6)
- `DensitySmoothing` — zone-based and neighbor-based smoothing (§6.8)
- `AssignmentLoop` — outer time-slice loop, inner convergence loop
- `DemandAdapter` — abstract base, `ODMatrixAdapter`, `TripStreamAdapter`
- `SegmentSpeedWriter` — generates CSV to tmpfs from `NetworkState`

### 9.3  Performance-critical path (candidate for C++ extension)

The flow accumulation step (decompose millions of paths into per-edge flow
increments) is the most likely bottleneck. If Python + NumPy is too slow,
this should be moved to a C++ nanobind extension that:

1. Takes route annotation arrays (node IDs, distances) as input.
2. Performs hash-based edge lookup and atomic flow increment.
3. Returns updated flow array.

This can be added as a new `.cpp` source in the existing build without
touching OSRM core.

---

## 10  Testing, validation, and benchmarking

### 10.1  Test tiers

| Tier | Scope | Network | Purpose |
|------|-------|---------|---------|
| **Unit** | Individual functions | None (pure math) | VDF correctness, solver accuracy, CSV writer |
| **Component** | Subsystem integration | Synthetic 4–5 node | Flow accumulation, density smoothing, engine refresh |
| **Structural validation** | Known theoretical results | Braess (4 nodes), Nguyen-Dupuis (13 nodes) | Verify equilibrium properties hold |
| **Benchmark validation** | Published test networks | Sioux Falls (24 nodes, 76 links) | Convergence, Wardrop conditions, flow patterns |
| **Regional benchmark** | Runtime at scale | Monaco (~5k edges), metro (~100k+ edges) | Wall-clock profiling, memory, throughput |

### 10.2  Unit tests (pure Python, no OSRM dependency)

These test the mathematical components in isolation:

```
tests/assignment/
├── test_vdf.py              # Bi-parabolic VDF
├── test_flow_to_density.py  # Closed-form inverse
├── test_density_smoothing.py
├── test_fractional_loading.py
├── test_csv_writer.py
└── test_network_state.py
```

**VDF tests** (`test_vdf.py`):
- v(0) = v_f (free-flow speed at zero density)
- v(k_c) = v_f / 2 (critical speed)
- v(k_j) = 0 (jam density → zero speed) for congested branch
- q(k_c) = q_c = v_f · k_c / 2 (capacity flow at critical density)
- C¹ continuity: v(k_c⁻) = v(k_c⁺) and dv/dk matches at k_c
- Monotonicity: v is non-increasing on [0, k_j]
- Vectorized output matches scalar loop for random inputs

**Closed-form inverse tests** (`test_flow_to_density.py`):
- Round-trip: q → k(q) → q(k) = q for random flows in [0, q_c]
- k(0) = 0
- k(q_c) = k_c
- Boundary: k(q > q_c) clamped to k_c
- Numerical accuracy: |q - k·v(k)| < ε for all test cases

**Fractional loading tests** (`test_fractional_loading.py`):
- Single-bin trip: all flow in departure bin, fractions sum to 1.0
- Multi-bin trip: fractions sum to 1.0 per link
- Boundary alignment: trip exactly spanning bin boundary
- Zero-duration links: skipped correctly

**Density smoothing tests** (`test_density_smoothing.py`):
- Identity: β=0 returns original density
- Conservation: total density · length is preserved by smoothing
- Convergence: repeated passes converge to uniform density on connected graph
- Known topology: 3-link chain with handcalculated expected output

### 10.3  Structural validation: Braess network

> Source: `bstabler/TransportationNetworks/Braess-Example`
> 4 nodes, 5 links, 1 OD pair.

Braess's paradox: adding a zero-cost shortcut link between nodes 3→4 causes
the total system travel time to INCREASE under user equilibrium, because
selfish routing overloads the shortcut.

**What we validate** (VDF-independent — works with bi-parabolic):

1. **Without the shortcut link (4 links)**: equilibrium has symmetric flow
   split between the two paths 1→3→2 and 1→4→2.
2. **With the shortcut link (5 links)**: equilibrium shifts to overuse the
   path 1→3→4→2, and total system travel time increases.
3. **Wardrop condition**: at convergence, all used paths have equal cost.
   No unused path has lower cost.

**Implementation**: Synthesize a minimal OSM XML file with 4 nodes at
arbitrary coordinates and ways matching the Braess topology. Process through
OSRM extract/partition/customize. Run assignment with a single OD pair. The
test passes if conditions 1–3 hold.

### 10.4  Benchmark validation: Sioux Falls

> Source: `bstabler/TransportationNetworks/SiouxFalls`
> 24 nodes, 76 links, 528 OD pairs, published UE solution.
> Node coordinates available in `SiouxFallsCoordinates.geojson`.

The Sioux Falls network is the canonical traffic assignment benchmark.
Published solutions use BPR (α=0.15, β=4), so our bi-parabolic VDF will
produce **different** equilibrium flows. What we CAN validate:

**Structural properties** (VDF-independent):

1. **Wardrop conditions at convergence**: for every OD pair, all used paths
   have equal travel time, and no unused path is cheaper. Measured as
   relative gap:
   ```
   gap = Σ_a x_a · t_a / Σ_rs q_rs · π_rs − 1
   ```
   where π_rs is the shortest-path cost between r and s. Gap < 0.01 is
   the conventional target.

2. **Convergence monotonicity**: gap decreases (non-increasing) across
   MSA iterations. Failure indicates a bug in flow blending or VDF.

3. **Flow conservation**: Σ paths for each OD pair = demand for that pair.

4. **Link flow non-negativity**: all q_e ≥ 0.

**Qualitative comparison with BPR solution**:

5. **Correlation**: rank-order link flows should correlate strongly with
   published BPR equilibrium flows (r > 0.9). The VDF changes magnitudes,
   not the overall congestion pattern.

6. **Highly loaded links**: the same links that are congested in the BPR
   solution should also be congested in our solution (top-10 overlap).

**Implementation**: Synthesize an OSM XML from the GeoJSON coordinates and
TNTP link topology. Map TNTP capacity to lane count + k_j. Map TNTP
free-flow time to OSRM profile speeds. Process through OSRM, run full
assignment with the published OD matrix, and check conditions 1–6.

**TNTP-to-OSM bridge utility**:

```python
def tntp_to_osm(nodes_geojson, net_tntp, output_osm):
    """Convert TNTP network files to OSM XML for OSRM processing.

    Reads node coordinates from GeoJSON and link topology from TNTP net file.
    Produces a minimal .osm XML with nodes and one-way highway segments.
    Link attributes (capacity, free-flow time, lanes) are mapped to OSM tags.
    """
    ...
```

This utility is reusable for any TNTP network that has geographic coordinates.

### 10.5  Benchmark validation: Nguyen-Dupuis (optional)

> Source: `bstabler/TransportationNetworks/NguyenDupuis`
> 13 nodes, 19 links.

Smaller than Sioux Falls but with more route alternatives per OD pair,
making it useful for testing convergence behavior on overlapping paths.
Same validation properties as §10.4, items 1–4.

### 10.6  Empirical validation against observed traffic data

Proving equilibrium on toy networks shows the algorithm is correct. Matching
**observed real-world traffic** shows the model is *useful*. Several open
data sources enable this:

| Source | Coverage | Data type | OSM-compatible? |
|--------|----------|-----------|-----------------|
| **Caltrans PeMS** | California freeways | 5-min flow, speed, occupancy at ~40k detectors | Stations have lat/lon; map-matchable to OSM |
| **Uber Movement** | ~50+ global cities | Aggregated link speeds by hour-of-day | OSM segment IDs directly |
| **Nature unified dataset** (2024) | 20 US cities | Flow, speed, density, travel time | Built on OSM networks |
| **State DOT count programs** | US statewide | AADT and hourly counts at stations | Lat/lon; map-matchable |
| **graphhopper/open-traffic-collection** | Global directory | Links to open count/speed portals | Varies |

**Recommended empirical validation workflow**:

1. **Pick a city with both OSM coverage and open traffic counts** (e.g., a
   California metro with PeMS sensor coverage, or a city from the Nature
   unified dataset).
2. **Obtain or synthesize an OD matrix** — from Census LODES commute data,
   a gravity model, or the unified dataset's included demand.
3. **Run assignment** on the OSM network with our bi-parabolic VDF.
4. **Compare assigned link flows/speeds with observed data**:
   - GEH statistic (industry standard): GEH < 5 for >85% of count locations
     is considered a good model.
   - Speed RMSE by facility type.
   - Scatter plot of assigned vs. observed (visual sanity check).

```
GEH = √(2(M − C)² / (M + C))

where M = model flow, C = count (observed flow)
GEH < 5 is acceptable for individual links
```

This is a stretch goal — it requires external data procurement and OD matrix
estimation, which are projects in themselves. But it's the gold standard for
model credibility and would strongly differentiate this tool from academic
prototypes.

### 10.7  Regional runtime benchmarking

#### Tier 1: Monaco (already in repo)

- **Network**: `tests/data/monaco.osm.pbf` (~5k edges)
- **Purpose**: Fast CI-friendly benchmark. Validates end-to-end pipeline
  runs without errors. Measures per-iteration wall clock.
- **Demand**: Synthetic — uniform random OD sampling from network nodes.
- **Metrics**: total wall-clock, customize latency, routing throughput,
  memory high-water mark.

#### Tier 2: Metro-scale (offline benchmark)

- **Network**: A Geofabrik OSM extract for a mid-size metro area
  (e.g., Lyon, Stuttgart, Salt Lake City — ~100k–500k edges).
- **Purpose**: Stress-test scalability of customize loop, flow accumulation,
  and memory footprint. Not run in CI.
- **Demand**: Synthetic gravity model or a published OD matrix if available.
- **Metrics**:

| Metric | Target (metro scale) |
|--------|---------------------|
| Customize latency (per iteration) | < 30 seconds |
| Routing throughput | > 50k routes/second |
| Full assignment (20 iterations, single time bin) | < 30 minutes |
| Peak memory | < 8 GB |
| Convergence gap after 20 iterations | < 0.05 |

#### Tier 3: Region-scale (stretch goal)

- **Network**: A full US state or small European country (~1M+ edges).
- **Purpose**: Determine the ceiling of the OSRM-based approach before
  in-memory weight injection becomes necessary.
- **Demand**: Synthetic or LODES/Census commute flow data.

### 10.8  Expected performance characteristics

#### Customize latency

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

#### Routing throughput

OSRM routing with GIL release and thread pool:

- Single query: ~1–5 ms (city scale)
- Bulk parallel: ~50k–200k routes/second (depending on path length and cores)
- Table queries: much faster for dense OD matrices

#### Memory footprint

- OSRM engine: ~1–4 GB for a large metro area
- NetworkState: ~100 bytes/edge × 1M edges = ~100 MB
- Path storage (if retained): potentially large; may need streaming

### 10.9  CI integration

Unit tests (§10.2) and the Monaco smoke benchmark (§10.6 Tier 1) run in CI
on every PR. Structural validation (Braess, §10.3) also runs in CI — the
synthetic OSM is tiny and processes in seconds.

Sioux Falls validation (§10.4) runs in CI but with relaxed iteration limits
(5 iterations, gap < 0.1) for speed. Full convergence is an offline test.

Metro and region benchmarks are manual / scheduled nightly runs, not
blocking PR checks.

---

## 11  Implementation phases

### Phase 1: Expose traffic update surface

**Deliverable**: `osrm.customize()` accepts `segment_speed_file` and
`turn_penalty_file` arguments, forwarded to OSRM's `UpdaterConfig`.

**Scope**:
- Modify `src/customizerconfig_nb.cpp` to bind `updater_config` sub-fields
- Modify `src/osrm/preprocessing.py` `customize()` to accept and forward kwargs
- Add integration test: extract → partition → customize with speed file → route
  and verify changed travel times

### Phase 2: Network state, VDF, and unit tests

**Deliverable**: A `NetworkState` class that can be populated from OSRM route
annotations, a `BiParabolicVDF` (Fournier et al.) that evaluates
vectorized speed from density, a `DensitySmoothing` module, and full unit
test coverage (§10.2).

**Scope**:
- `NetworkState`: edge registry, flow accumulation, density conversion
- `BiParabolicVDF`: vectorized NumPy implementation (§6.2–5.7)
- `DensitySmoothing`: neighbor-based smoothing (§6.8)
- `SegmentSpeedWriter`: generate CSV to tmpfs `/dev/shm/` (§2.6)
- `FractionalLoader`: travel-time offset bin distribution (§7.3)
- Unit tests for all of the above (§10.2)
- TNTP-to-OSM bridge utility for test network synthesis (§10.4)

### Phase 3: Assignment loop, Braess, and Monaco (single period)

**Deliverable**: End-to-end assignment on Monaco with OD-matrix input,
MSA convergence, and link-flow output. Braess paradox structural validation.

**Scope**:
- `AssignmentLoop` orchestrator with fractional loading
- `ODMatrixAdapter` demand input
- Convergence reporting (relative gap, iteration log)
- Braess paradox validation (§10.3) — synthesize OSM, verify paradox manifests
- Monaco smoke benchmark (§10.6 Tier 1)
- CI integration for unit tests + Braess + Monaco

### Phase 4: Sioux Falls validation and matrix-free adapter

**Deliverable**: Sioux Falls benchmark validation (§10.4). Trip-stream demand
input on the same assignment core.

**Scope**:
- TNTP-to-OSM synthesis for Sioux Falls (24 nodes, 76 links, 528 OD pairs)
- Wardrop condition verification (relative gap < 0.01)
- Flow pattern correlation with published BPR equilibrium (r > 0.9)
- `TripStreamAdapter` for matrix-free demand
- Nguyen-Dupuis validation (optional)

### Phase 5: OSRM multi-period patch and DTA

**Deliverable**: The OSRM core patch (§3.3) enabling multi-period metric
storage and `departure_period` query parameter. Multi-period DTA on OSRM.

**Scope**:
- Implement the ~300-line OSRM core patch (§3.3):
  - `BaseParameters` → `departure_period`
  - `DataFacadeFactory` → period × exclude indexing
  - `customizer/files.hpp` → period-indexed TAR paths
  - `Customizer::Run()` → `period_index` parameter
  - `ContiguousInternalMemoryAlgorithmDataFacade` → multi-period load
- Backward compatibility: existing OSRM data (no period entries) loads as
  single period (period_index=0)
- `PeriodConfig` Python class for user-defined period mappings (§3.5)
- Multi-period assignment loop (§3.8)
- py-osrm binding updates: expose `departure_period` on Route/Table
- Integration tests: Monaco with 2–4 periods, verify different routes per
  period under different congestion states
- Prepare PR to OSRM upstream (clean commit, tests, documentation)

### Phase 6: Performance optimization and scaling

**Deliverable**: Profiling-driven optimization. Metro-scale and regional
runtime benchmarks.

**Scope**:
- Metro-scale benchmark (§10.7 Tier 2) — Geofabrik extract, synthetic demand
- Benchmark customize latency at scale (tmpfs CSV vs. disk, §2.6)
- Parallelize period customization across cores
- C++ flow-accumulation extension if Python is bottleneck
- Evaluate shared-memory hot-swap (§2.5 Strategy B) — wrap
  `storage::Storage::Run()` or use subprocess `osrm-datastore`
- Evaluate feasibility of in-memory `LookupTable` bypass (skip CSV entirely)
- Region-scale stretch test (§10.7 Tier 3) — California or similar

---

## 12  Risk register

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| **Customize latency dominates runtime** at state scale | High | High | Parallelize N period customizations across cores; use tmpfs CSV (§2.6); for 6 periods on 6 cores → same wall-clock as 1 period |
| **OSRM multi-period patch rejected upstream** | Medium | Medium | Patch is isolated and maintainable on a pinned fork (v6.0.0); feature is generally useful ("time-of-day profiles") improving acceptance odds |
| **N-period memory exceeds hardware** for fine-grained DTA | Medium | Low | Memory scales linearly with N; user chooses N based on hardware; 4–8 periods covers most use cases at <10 GB (CA) |
| **Bidirectional search approximation** for cross-period trips | Low | Medium | Same approximation as all discrete-time DTA models; trips shorter than period duration are exact; document the approximation and its bounds |
| **OSM lacks lane/capacity data** for many links | Medium | High | Ship sensible defaults by road class; allow user overrides via enrichment CSV |
| **Engine re-instantiation has hidden side effects** (TBB thread pool, memory leaks) | Medium | Medium | py-osrm already has a TBB cleanup handler; test repeated create/destroy cycles; migrate to shared-memory hot-swap (§2.5 Strategy B) if problematic |
| **Flow-to-density solver diverges** for edge cases | Low | Medium | Clamp density to [0, k_j]; use robust Newton with bisection fallback |
| **Path decomposition is too slow in Python** for large networks | Medium | Medium | Move to C++ extension; the nanobind build system already supports adding new .cpp sources |
| **Bi-parabolic VDF produces unrealistic speeds** on certain link types | Medium | Low | Validate against observed speed-flow data; allow per-link VDF parameter overrides |
| **OSRM upstream changes break FetchContent build** | Low | Low | Pin to v6.0.0; upgrade deliberately |
| **Neighbor-based smoothing over-diffuses** on sparse networks | Medium | Medium | Cap passes at 2; expose β as tunable; validate against known congestion patterns |
| **Short-link density volatility** despite smoothing | Medium | Medium | Minimum link-length filter; merge very short links into preceding link for assignment purposes |

---

## 13  Open questions

1. **Slice width calibration**: 1-hour periods are recommended as the default
   (§7.2). Should we run a sensitivity analysis across 15/30/60 min as part
   of the Sioux Falls validation?

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

8. **OSRM patch upstream strategy**: Submit the multi-period patch as a PR
   to Project OSRM, or maintain as a fork? What level of test coverage and
   documentation would maximize acceptance odds?

9. **Parallel customize implementation**: Use Python `multiprocessing` to
   run N period customizations concurrently, or a shell-level approach?
   Need to verify OSRM customize is process-safe for concurrent execution
   writing to different output paths.

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
| OSRM `src/customize/customizer.cpp` | Customizer::Run() — calls Updater then recomputes metrics (**patch target: period_index**) |
| OSRM `include/customizer/cell_metric.hpp` | CellMetric struct: `{weights[], durations[], distances[]}` |
| OSRM `include/customizer/files.hpp` | Read/write `.osrm.cell_metrics` TAR (**patch target: period-indexed paths**) |
| OSRM `include/engine/api/base_parameters.hpp` | BaseParameters (**patch target: departure_period**) |
| OSRM `include/engine/datafacade_factory.hpp` | DataFacadeFactory — `vector<Facade>`, per-query selection (**patch target: period × exclude**) |
| OSRM `include/engine/datafacade/contiguous_internalmem_datafacade.hpp` | MLD facade impl, `GetCellMetric()` (**patch target: multi-period load**) |
| OSRM `include/engine/routing_algorithms/routing_base_mld.hpp` | MLD search — `relaxOutgoingEdges()` (unchanged by patch) |
| OSRM `include/engine/data_watchdog.hpp` | DataWatchdog for shared-memory hot-swap (§2.5) |
| OSRM `include/engine/datafacade_provider.hpp` | WatchingProvider / ImmutableProvider / ExternalProvider |
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

Mesoscopic density smoothing (see §6.8):

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
