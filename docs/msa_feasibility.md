# Time-Dependent MSA Feasibility Analysis

## Question

Is it feasible to extend the current MSA/Frank-Wolfe equilibrium
solver (`assign_matrix`) to handle time-dependent routing — where
trips have varying departure times and the network has per-period
congestion — while converging to a meaningful equilibrium?

## Executive Summary

**Computationally feasible on small networks (chi-sketch), infeasible
at regional scale with OSRM's customize-per-iteration architecture.**
The main bottleneck is not the algorithm but the I/O: each MSA
iteration requires writing speed CSVs and re-customizing OSRM for
every active period. At 96 periods × 50 iterations, this dominates.
An iterative-stream approach (multiple forward passes with feedback)
is more practical and achieves similar results without formal
equilibrium guarantees.

## 1. Convergence Theory

### Classical static MSA

Wardrop user equilibrium (UE) requires:
- **Monotone cost functions**: link cost c_a(f) is non-decreasing in
  flow f. Our VDF satisfies this — `demand_to_density` is monotone,
  `density_to_speed` is monotone decreasing, so travel time
  `L/v(f)` is monotone increasing.
- **Separable costs**: link a's cost depends only on flow on link a,
  not on other links. Our VDF is purely per-link (no neighbor lookups).
- **Fixed demand**: the OD matrix doesn't change between iterations.

Under these conditions, MSA with step size α=1/n converges to UE.
Frank-Wolfe with line search converges faster (sublinear).

### Time-dependent extension

With varying departure times, the cost of a trip depends on:
1. Which links it traverses (path)
2. When it arrives at each link (departure time + accumulated travel time)
3. The congestion on each link **at the time of arrival**

This breaks the classical framework in two ways:

**Non-separability**: A trip's cost on link a at time t depends on
flows from OTHER trips on link a at time t. The flow at time t is
itself a function of routing decisions of trips that departed earlier.
This creates temporal coupling — link costs are no longer separable
across the full (link × time) space.

**Path-time coupling**: The same OD pair departing at different times
takes different paths and experiences different costs. The "all-or-
nothing" step routes each trip on its shortest time-dependent path,
but the resulting per-period flow decomposition depends on how
trips distribute across periods — which itself depends on congestion.

### Does MSA still converge?

**In theory**: No formal guarantee. The variational inequality
formulation of time-dependent UE (e.g., Friesz et al. 1993) requires
FIFO conditions (first-in-first-out on each link) and additional
regularity. Our MFD-based VDF naturally satisfies FIFO since
speed decreases monotonically with density.

**In practice**: MSA with 1/n step sizes typically stabilizes for
time-dependent problems, even without formal convergence proof.
The gap oscillates but doesn't diverge. This is well-documented in
the DTA literature (Peeta & Ziliaskopoulos 2001, Szeto & Lo 2006).
The practical convergence quality depends on:
- Period granularity (finer periods → smoother cost surface)
- Demand magnitude (heavier congestion → more oscillation)
- Step size schedule (1/n may be too aggressive; 1/√n can help)

**Our specific risk**: The MFD's congested branch (oversaturated
extension) introduces a steep cost gradient. Small flow changes near
capacity cause large speed changes. This amplifies oscillation.
The spatial smoother helps but doesn't eliminate the issue.

### Verdict on convergence

MSA will **stabilize** (bounded oscillation) but may not **converge**
to a tight gap for time-dependent problems. A relative gap of 1-5%
is achievable; sub-1% is unlikely without sophisticated algorithms
(e.g., gradient projection, path-based methods).

## 2. Architectural Changes Required

### NetworkState: 1D → 2D flow arrays

Current state (per-link only):
```
flow_vph:      float64[N_edges]
density_vpkm:  float64[N_edges]
speed_kmh:     float64[N_edges]
```

Time-dependent MSA requires per-link-per-period:
```
flow_vph:      float64[N_edges, N_periods]
density_vpkm:  float64[N_edges, N_periods]
speed_kmh:     float64[N_edges, N_periods]
```

Immutable properties (`length_m`, `freeflow_kmh`, `jam_density`,
`n_lanes`, `kc_ratio`) remain 1D — they're physical constants.

**Scope**: ~3 arrays need 2D conversion. The VDF is purely per-link
and broadcasts cleanly — a simple loop over periods suffices:

```python
for p in range(n_periods):
    state.density_vpkm[:, p] = vdf.demand_to_density(
        state.flow_vph[:, p], state.freeflow_kmh, state.jam_density)
    state.speed_kmh[:, p] = vdf.density_to_speed(
        state.density_vpkm[:, p], state.freeflow_kmh, state.jam_density)
```

### Flow accumulation

Each iteration routes ALL trips. A trip departing in period p that
traverses links in periods p, p+1, p+2 contributes volume to each
period's flow array. The OSRM route result gives per-link travel
times; Python decomposes into period contributions post-hoc.

### Gap computation

Current gap: `Σ(V_blend × c) / Σ(V_aon × c) - 1` summed over links.
Time-dependent: sum over (links × periods) using period-specific costs.

### MSA blending

Per-period blending, same formula:
```
flow_vph[:, p] = prev[:, p] + α × (aon[:, p] - prev[:, p])
```

### Refactoring estimate

| Component | Lines changed | Complexity |
|-----------|--------------|------------|
| NetworkState arrays | ~30 | Low |
| _update_state VDF loop | ~15 | Low |
| _compute_relative_gap 2D | ~10 | Low |
| _fw_line_search 2D | ~20 | Medium |
| Flow accumulation post-hoc | ~50 | Medium |
| CSV writer per-period | ~20 | Low |
| **Total** | **~145** | |

The refactoring itself is straightforward. The VDF is per-link,
broadcasts cleanly, and the MSA/FW math is identical — just
applied per-period. **This is not the bottleneck.**

## 3. Computational Cost

### Per-iteration breakdown

| Step | Chi-sketch | Chi-regional |
|------|-----------|--------------|
| Route all trips | ~88 trips/s × 1.3M = 14.8k routes ≈ 1s | 14.3k routes/s × 1.36M ≈ 95s |
| VDF per period | 2,950 edges × 4 periods ≈ <1ms | 39k edges × 96 periods ≈ 5ms |
| Write CSVs | 4 CSVs ≈ 10ms | 96 CSVs ≈ 200ms |
| Customize OSRM | 4 × ~0.1s ≈ 0.4s | 96 × ~0.5s ≈ 48s |
| Engine reload | 1 × ~0.1s | 1 × ~2s |
| **Total per iter** | **~2s** | **~145s** |

### Total for convergence

| Scenario | Iterations | Wall clock |
|----------|-----------|------------|
| Chi-sketch, 4 periods | 50 | ~100s (1.7 min) |
| Chi-regional, 4 periods | 50 | ~2.4 hrs |
| Chi-regional, 96 periods | 50 | **~2.0 hrs** |
| California, 96 periods | 50 | **~days** |

The customize step dominates at scale. OSRM customize reloads the
full edge-expanded graph, re-runs Dijkstra through every cell, and
writes results to disk — per period. This is inherently sequential
for each period's weight set.

**Comparison with AequilibraE**: On chi-regional, AequilibraE's
BFW solver runs at 0.806 s/iter (50 iterations = 40 seconds).
OSRM-based time-dep MSA is ~180× slower per iteration due to the
customize overhead. AequilibraE doesn't support time-dependent
routing natively, but the speed gap is instructive.

### The fundamental mismatch

OSRM was designed for **one-shot customize + many queries**. The
MLD cell precomputation amortizes over millions of route queries.
MSA inverts this: each iteration changes all link weights, requiring
full re-customization. The precomputation cost is paid N_iterations ×
N_periods times instead of once.

## 4. Alternatives

### A. Iterative stream (recommended)

Run `assign_stream` multiple times, each pass using the previous
pass's per-period weights as starting point:

```
for outer_iter in range(max_iters):
    result = solver.assign_stream(trips)  # generates per-period CSVs
    if converged(result, prev_result):
        break
    prev_result = result
```

**Advantages**:
- Uses existing code with minimal changes
- Each stream pass is a single forward simulation (~minutes, not hours)
- Per-period weights improve with each pass
- Natural ABM integration (each ABM iteration = one stream pass)
- No 2D flow array needed — periods processed sequentially

**Disadvantages**:
- No formal convergence guarantee
- Heuristic stopping criterion
- Path quality depends on batch granularity

**Expected behavior**: 2-4 outer iterations typically stabilize
per-period flows. Trips near period boundaries get progressively
better weights as the feedback loop tightens.

### B. Hybrid: static MSA per period + stream for cross-period

Run independent static MSA within each period (proven convergence),
then use stream to handle cross-period spillover:

1. Group trips by departure period
2. Run `assign_matrix` independently per period
3. Use per-period equilibrium flows as input to `customize_multi_period`
4. Route cross-boundary trips with time-dependent OSRM

**Advantages**:
- Per-period MSA has formal convergence guarantees
- Cross-period effects handled by the OSRM search switching
- Parallelizable across periods

**Disadvantages**:
- Ignores cross-period flow coupling during equilibration
- Spillover from period N to N+1 not captured in per-period MSA

### C. Full time-dependent MSA

As analyzed in sections 1-3 above. Computationally expensive,
convergence uncertain. Only practical for small networks or
few periods.

### D. AequilibraE hybrid — per-period equilibrium with custom VDF

Use AequilibraE's compiled Dijkstra-based solver for per-period
static equilibrium (BFW at 0.806 s/iter on chi-regional), then
feed the per-period equilibrium flows into OSRM's multi-period
metrics for time-dependent routing of cross-period trips.

```
for period in range(n_periods):
    aeq_result = aequilibrae_assign(period_demand, custom_vdf)  # ~40s
    period_speeds[period] = aeq_result.link_speeds

osrm.customize_multi_period(base, period_speeds)  # one-shot
engine = osrm.OSRM(base)  # single load, all periods
# Route with time-dependent switching for cross-period trips
```

**Custom VDF PR**: AequilibraE supports pluggable VDFs. A PR
contributing our bi-parabolic MFD with queue spillover extension
would give AequilibraE a physically-grounded congested branch
(density-based, not BPR power-law). This enables:
- Proper oversaturation handling (flow > capacity → queue)
- Unserved demand computation from the VDF itself
- MFD-consistent speed/density relationships

**Feasibility**: High. AequilibraE's VDF interface accepts
user-defined functions. The MFD VDF is purely per-link (no
neighbor lookups) and monotone in flow — both required properties
for AequilibraE's convergence guarantees. The queue spillover
extension (`demand_to_density` oversaturated branch) maps cleanly
to AequilibraE's link cost function interface.

**Architecture**:
- AequilibraE handles equilibrium convergence (fast, proven)
- OSRM handles realistic network routing (turn penalties, one-ways,
  access restrictions) and cross-period weight switching
- py-osrm orchestrates: AequilibraE → period speeds → OSRM customize
  → time-dependent routing → ABM feedback

This is arguably the best of all options: formal equilibrium per
period, sub-minute convergence at regional scale, physically
meaningful VDF, and cross-period routing via OSRM.

## 5. Recommendation

**Implement iterative stream (Option A)** as the immediate path.
It provides:
- Practical time-dependent assignment at regional scale
- Natural ABM integration
- Minimal code changes (stream already works)
- Per-period weight improvement through iteration

**Investigate AequilibraE hybrid (Option D)** as the high-quality
path. A custom VDF PR to AequilibraE would unlock per-period
equilibrium at 180× the speed of OSRM-based MSA, with formal
convergence guarantees. Combined with OSRM's multi-period routing
for cross-period trips, this is the most capable architecture.

**Defer full time-dependent MSA (Option C).** The 2D refactoring
is modest (~145 lines) but the computational cost makes it
impractical at regional scale with OSRM.

**Consider per-period hybrid (Option B)** as a middle ground if
AequilibraE integration is deferred.

## References

- Friesz, T.L. et al. (1993). "A variational inequality formulation
  of the dynamic network user equilibrium problem." Operations Research.
- Peeta, S. & Ziliaskopoulos, A.K. (2001). "Foundations of Dynamic
  Traffic Assignment." Networks and Spatial Economics.
- Szeto, W.Y. & Lo, H.K. (2006). "Dynamic Traffic Assignment:
  Properties and Extensions." Transportmetrica.
