# C++ BatchRoute: Native Parallel Routing

## Problem

Python `ThreadPoolExecutor` achieves only ~2x speedup (15K routes/s) on 12 cores
due to per-call GIL re-acquisition and thread scheduling overhead. Each OSRM MLD
route takes ~0.06-0.1ms — faster than Python can dispatch threads.

The HTTP server (`osrm-routed`) achieves full CPU utilization because it uses a
pure C++ event loop with no Python overhead.

## Proposed Solution

Add a `BatchRoute` C++ method to the nanobind bindings that accepts a vector of
route requests, dispatches them via TBB `parallel_for` with the GIL released once,
and returns all results. Zero Python threading overhead.

## Architecture

```
Python                          C++ (nanobind)
──────                          ──────────────
                                
bulk_route(engine, od_dict)     
  → build RouteParameters[]  →  BatchRoute(params_vec)
                                  gil_scoped_release
                                  tbb::parallel_for(0, N, [&](i) {
                                    engine->Route(params[i], results[i]);
                                  });
  ← parse results[]          ←  return results_vec
```

## Implementation

### 1. C++ binding (`src/osrm_nb.cpp`)

Add after the existing `Route` method (~line 190):

```cpp
#include <tbb/parallel_for.h>

// ...inside the OSRM class definition...

.def("BatchRoute", [](OSRM* t, const std::vector<RouteParameters>& params_list) {
    // Validate all params first (with GIL held for error reporting)
    for (size_t i = 0; i < params_list.size(); ++i) {
        if (!params_list[i].IsValid()) {
            throw std::runtime_error(
                "Invalid Route Parameters at index " + std::to_string(i));
        }
    }

    // Pre-allocate results
    std::vector<json::Object> results(params_list.size());
    std::vector<osrm::engine::Status> statuses(params_list.size());

    {
        nb::gil_scoped_release release;
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, params_list.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    statuses[i] = t->Route(params_list[i], results[i]);
                }
            }
        );
    }

    // Build Python-friendly output: list of result dicts
    // Failed routes get None instead of throwing
    nb::list py_results;
    for (size_t i = 0; i < results.size(); ++i) {
        if (statuses[i] == osrm::engine::Status::Ok) {
            py_results.append(results[i]);
        } else {
            py_results.append(nb::none());
        }
    }
    return py_results;
},
"Route multiple OD pairs in parallel using native C++ threading (TBB).\n\n"
"Args:\n"
"    params_list: List of RouteParameters objects.\n\n"
"Returns:\n"
"    List of route result dicts (None for failed routes).\n")
```

### 2. Python wrapper (`src/osrm/bulk.py`)

Modify `bulk_route` to use `BatchRoute` when available:

```python
def bulk_route(osrm_instance, df, ..., **default_params):
    # ... existing row parsing and param building ...
    
    # Build RouteParameters objects
    params_list = []
    for row in rows:
        params = build_route_params(row, default_params)
        params_list.append(params)
    
    # Use native batch if available (single C++ call, TBB threading)
    if hasattr(osrm_instance, 'BatchRoute'):
        raw_results = osrm_instance.BatchRoute(params_list)
        # ... unpack into existing result format ...
    else:
        # Fallback to ThreadPoolExecutor (existing code)
        ...
```

### 3. CMake (`CMakeLists.txt`)

TBB is already linked as an OSRM dependency. Verify the include path is available:

```cmake
# Should already be present via OSRM's CMake config
find_package(TBB REQUIRED)
target_link_libraries(osrm_nb PRIVATE TBB::tbb)
```

Check: `grep -r "TBB\|tbb" CMakeLists.txt` — if not explicitly linked, OSRM's
imported targets should transitively include it.

## RouteParameters Construction

The key challenge is efficiently building `RouteParameters` C++ objects from Python.
Currently `RouteParameters` is already exposed via nanobind
(`src/parameters/routeparameter_nb.cpp`). Options:

### Option A: Build in Python (simple, some overhead)
```python
params = osrm.RouteParameters(
    coordinates=[(lon1, lat1), (lon2, lat2)],
    annotations=["nodes", "speed", "distance", "duration"],
)
```
93K object constructions in Python — maybe 0.5-1s overhead.

### Option B: Add a C++ batch constructor (fast, more code)
```cpp
.def("BatchRouteFromCoords", [](OSRM* t,
    const std::vector<std::pair<double,double>>& origins,
    const std::vector<std::pair<double,double>>& dests,
    bool annotations_nodes, bool annotations_speed) {
    // Build params internally in C++, then route
});
```
This avoids Python object construction entirely — just pass flat coordinate arrays.

**Recommendation**: Start with Option A. If profiling shows param construction is
a bottleneck, add Option B.

## Expected Performance

| Approach | Routes/s (Monaco 50K) | Scaling |
|---|---|---|
| Serial (baseline) | 6,000 | 1.0x |
| ThreadPoolExecutor 12w | 14,000 | 2.3x |
| Multi-engine (2) | 17,000 | 2.8x |
| **BatchRoute (TBB)** | **~40,000-60,000** | **~7-10x** |

The TBB estimate assumes near-linear scaling to 12 cores (6K × 10 = 60K) since
TBB's work-stealing scheduler has negligible overhead for tasks >0.05ms.

Conservative estimate: 5x improvement over current ThreadPoolExecutor.

## Testing

1. **Correctness**: Route results from `BatchRoute` must match serial `Route` for
   the same inputs (compare node sequences, durations, distances).

2. **Performance**: Benchmark on Monaco (50K routes) and Chicago Sketch (93K routes)
   comparing ThreadPoolExecutor vs BatchRoute wall time and CPU utilization.

3. **Error handling**: Failed routes (NoRoute) return None without crashing the batch.

4. **Edge cases**: Empty input, single route, all routes fail.

## Risks

- **OSRM thread safety**: `Route()` on a single OSRM instance must be safe for
  concurrent reads. The MLD graph is immutable during queries, so this should hold.
  Verify with ASAN/TSAN under load.

- **TBB availability**: TBB must be available at build time. It's an OSRM dependency
  so should always be present, but verify on CI (Linux, macOS, Windows).

- **Memory**: 93K `json::Object` results allocated simultaneously. Each route result
  is ~1-5KB, so ~100-500MB total. Should be fine on modern systems but worth monitoring.

- **`json::Object` thread safety**: OSRM's internal JSON type must be safe to write
  to independent instances concurrently. Since each result is a separate object, this
  should be fine (no shared state).

## Files to Modify

1. `src/osrm_nb.cpp` — Add `BatchRoute` method (~30 lines)
2. `src/osrm/bulk.py` — Use `BatchRoute` when available
3. `CMakeLists.txt` — Verify TBB link (likely already present)
4. `tests/test_bulk.py` — Add `BatchRoute` correctness + performance tests
5. `src/osrm/__init__.py` — Expose `BatchRoute` in the Python API if needed

## Future: Shortest-Path-Tree Assignment (OSRM Core PR)

### Motivation

Even with BatchRoute (51K routes/s), large networks are bottlenecked by
routing 93K+ individual OD pairs per iteration.  Commercial tools (EMME,
VISUM) avoid this entirely by computing **one shortest-path tree per origin
zone** and loading all destinations simultaneously.

For Chicago Sketch: **387 trees vs 93,000 routes = 240× fewer routing ops.**

### Approach

Add a `TableWithAnnotations` or `TreeAssign` method to OSRM core that:

1. Computes a shortest-path tree from a single origin to all destinations
   (OSRM's Table service already does this internally for the cost matrix)
2. Traces back each destination→origin path through the tree
3. Returns **per-link flow accumulation** rather than individual route geometries
4. Accepts a demand vector so accumulation happens entirely in C++

This is fundamentally a new OSRM service endpoint — a core PR, not a wrapper
change.  MLD's multi-level Dijkstra would need adaptation to emit the tree
structure (currently it only emits costs).

### Expected gain

- Per-iteration: **4.2s → 0.1–0.5s** (93K routes → 387 trees)
- 50-iteration Chicago Sketch: **210s → 5–25s**
- Enables Chicago Regional (39K links, 100K+ OD pairs) in real-time

### LOE

- ~500–1000 lines C++ in OSRM core (table service extension)
- Requires deep understanding of MLD/CH internals
- Separate OSRM fork or PR — not a py-osrm change
- Estimated: 1–2 weeks
