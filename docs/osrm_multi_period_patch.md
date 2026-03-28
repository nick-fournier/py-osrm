# OSRM Multi-Period Metric Patch Specification

> **Parent document**: [Traffic Assignment Design](traffic_assignment_design.md)
> **Related**: [py-osrm Assignment Module](pyosrm_assignment_module.md)

---

## 1  Overview and motivation

OSRM's MLD algorithm already supports multiple metric sets via the "exclude
class" mechanism (vehicle-class restrictions). This document specifies a
surgical C++ patch (~350 lines across ~8 files) that extends this mechanism
to add a **period dimension**: N user-defined time periods, each with its own
pre-customized cell metrics. At query time, `departure_period=k` selects
metric set `k`. The MLD search algorithm is unchanged.

This enables time-dependent routing at full OSRM query speed (2000–5000 QPS)
without requiring a second routing engine, and supports concurrent queries
across different periods (thread-safe).

---

## 2  How OSRM already handles multiple metric sets

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

## 3  What changes with `customize`

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

## 4  The multi-period patch

**Concept:** Add a `period` dimension parallel to the existing `exclude`
dimension. Store N × E metric sets (N periods × E exclude classes). Query
with `departure_period=k` or `departure_time=T` → facade factory selects
metric set `k`.

**Patch scope** (~350 lines across ~8 files):

1. **`include/engine/api/base_parameters.hpp`**
   - Add `std::optional<unsigned> departure_period` — explicit period index
   - Add `std::optional<double> departure_time` — epoch seconds, auto-mapped

2. **`include/server/api/base_parameters_grammar.hpp`**
   - Add Boost.Spirit parsing rules for `departure_period=` and
     `departure_time=` URL query parameters
   - Add to the `base_rule` alternation chain
   - Since `BaseParameters` is inherited by all services, this exposes
     the parameters on **every HTTP endpoint** (Route, Table, Match,
     Nearest, Trip) automatically — no per-service changes needed

3. **`include/engine/datafacade_factory.hpp`**
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

4. **`include/customizer/files.hpp`**
   - Extended TAR paths: `/mld/metrics/{name}/period/{P}/exclude/{E}/`
   - Backward-compatible: if no period entries, treat as single period

5. **`src/customize/customizer.cpp`**
   - Accept a `period_index` parameter (default 0)
   - Write metrics under the period-indexed path

6. **`include/engine/datafacade/contiguous_internalmem_datafacade.hpp`**
   - Load N period metric sets during initialization
   - `GetCellMetric()` returns the period-appropriate view (selected by
     the facade factory, not the search algorithm)

7. **Period schedule file** (new, small)
   - Stored alongside OSRM data as `.osrm.period_schedule`
   - Simple format: list of `(period_index, start_epoch, end_epoch)` tuples
   - Supports arbitrary period boundaries (hourly, custom, weekday/weekend)
   - Optional: if absent, only explicit `departure_period` is available
   - Written by the assignment tool or by hand for production routing

8. **`include/engine/routing_algorithms/routing_base_mld.hpp`**
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

## 5  Bidirectional search correctness and cross-period trips

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
   correctly by fractional link loading ([Traffic Assignment Design](traffic_assignment_design.md) §4.3). As the vehicle traverses
   its route, each link's flow is assigned to the period when the vehicle
   would actually be there, based on cumulative travel time from departure.

Over multiple outer iterations, this self-corrects: if a route was
suboptimal because a later period had worse congestion than the departure
period assumed, the next iteration's later-period weights will reflect
that added flow, and some trips will reroute.

**Approximation quality depends on period width vs. trip duration.** With
1-hour periods, a typical 30-minute urban commute completes within one
period (exact). A 90-minute cross-regional trip spans ~2 periods, with
only the tail portion subject to the approximation. See [Traffic Assignment Design](traffic_assignment_design.md) §4.2 for period
width calibration guidance.

## 6  User-defined period configurations — PeriodConfig

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

## 7  Sparse cell-delta storage

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

(recomputing boundary-to-boundary shortest paths within cells) is
identical — it just runs N times and records which cells changed.

## 8  Memory and runtime costs

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

## 9  Valhalla note

If OSRM's discrete-period model proves insufficient for a niche use case,
Valhalla could be added as an alternative backend. Valhalla supports native
`date_time` routing (100–500 QPS vs OSRM's 2000–5000). The assignment
core (VDF, density, smoothing, convergence) is engine-independent, so a
Valhalla backend would not require restructuring. With the multi-period
OSRM patch, Valhalla is not a planned dependency.

---

## 10  OSRM file reference

| File | Role |
|------|------|
| OSRM `include/updater/updater_config.hpp` | UpdaterConfig with speed/penalty paths |
| OSRM `include/updater/csv_file_parser.hpp` | CSV parser — uses `mapped_file_source` ([py-osrm Assignment Module](pyosrm_assignment_module.md) §1.6) |
| OSRM `src/updater/updater.cpp` | CSV read → edge weight update logic |
| OSRM `src/customize/customizer.cpp` | Customizer::Run() (**patch target: period_index**) |
| OSRM `include/customizer/cell_metric.hpp` | CellMetric struct: `{weights[], durations[], distances[]}` |
| OSRM `include/customizer/files.hpp` | Read/write `.osrm.cell_metrics` TAR (**patch target: period-indexed paths**) |
| OSRM `include/engine/api/base_parameters.hpp` | BaseParameters (**patch target: departure_period, departure_time**) |
| OSRM `include/server/api/base_parameters_grammar.hpp` | HTTP URL param parser (**patch target: grammar rules**) |
| OSRM `include/engine/datafacade_factory.hpp` | DataFacadeFactory — `vector<Facade>`, per-query selection (**patch target: period × exclude**) |
| OSRM `include/engine/datafacade/contiguous_internalmem_datafacade.hpp` | MLD facade impl, `GetCellMetric()` (**patch target: multi-period load**) |
| OSRM `include/engine/routing_algorithms/routing_base_mld.hpp` | MLD search — `relaxOutgoingEdges()` (unchanged by patch) |
| OSRM `include/engine/datafacade_provider.hpp` | WatchingProvider / ImmutableProvider / ExternalProvider |
