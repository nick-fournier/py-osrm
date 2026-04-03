# Failed Experiments

Record of approaches tried and abandoned during development of the
density-based traffic assignment with bi-parabolic MFD VDF.

---

## Category A — VDF Shape Modifications

### 1. Conjugate Frank-Wolfe (CFW)

**Idea:** Add momentum from prior search directions to accelerate FW
convergence. Standard CFW formula (Mitradjieva & Lindberg 2013):
β = ∇Z·d_fw / (∇Z·(d_fw − d_prev)), conjugate direction = d_fw + β·d_prev.

**Implementation:** Added `_beckmann_gradient()` (link cost vector) and
`_cfw_beta()` with β capped at 0.5. Line search extended with
dk/dv override parameters to use conjugate direction.

**Result (Anaheim 100%, FW, 50 iter):** Gap 0.015, ρ 0.771,
max k/k_j 14.67 — **worse than vanilla FW on every metric** (baseline:
gap 0.005, ρ 0.813, max k/k_j 6.36).

**Why it failed:** The MFD's near-singular gradient at k_j creates a
non-smooth cost surface. β oscillates between 0 and ~1.0; the conjugate
direction overshoots, pushing density to 12–15× k_j on bottleneck links.
CFW assumes a smooth quadratic-like objective; the bi-parabolic MFD
violates that assumption near jam density.

**Commits:** `84c7a67` (added), `7947df8` (removed)

---

### 2. Stretched k_j (VDF smoothing, approach A)

**Idea:** Use k_j_eff = k_j × (1 + ε) in the congested branch equation so
the parabola reaches zero speed at k_j_eff instead of k_j.  At physical k_j,
speed is small but nonzero, gradient is finite.

**Result:** Did not improve gap. Density overshoots to 6× k_j regardless,
so the singularity at k_j_eff is reached just as easily.

**Why it failed:** FW's AON step loads entire OD volumes onto shortest paths
in one shot. Whether the singularity is at k_j or 1.1×k_j is irrelevant when
density is at 6×k_j. The problem is the *magnitude of overshoot*, not the
*location of the singularity*.

**Commits:** `ba5a3e9` (added), `0c2a779` (replaced)

---

### 3. Exponential Tail (VDF smoothing, approach B)

**Idea:** Replace the congested parabola for k > k_s = 0.85·k_j with an
exponential decay: v(k) = v(k_s) · exp(−B·(k − k_s)).  C¹-continuous splice,
asymptotic decay, no gradient singularity at any density.

**Implementation:** Added `splice_ratio` parameter to `BiParabolicVDF`.

**Result (Anaheim 100%, FW, splice=0.85):** Gap 0.012, ρ 0.812,
max k/k_j 6.37 — **slightly worse than vanilla FW** (gap 0.005).

**Why it failed:** Same root cause as stretched k_j. FW's overshoot is so
extreme that links land deep in the tail where all VDF shapes return
near-floor speed.  The VDF shape doesn't matter when density is 6× k_j.

**Key insight:** The convergence problem is not the VDF shape. It's the
loading strategy.

**Commits:** `89829ab` (added), `7947df8` (removed)

---

## Category B — Convergence Algorithm Failures

### 4. Density-Based MSA with Congested Routing Speed

**Idea:** Classic MSA blending on link density:
`k ← (1−α)k_prev + α·k_aon` where `k_aon = V/(v_routing × Δt)` and
`α = 1/(m+1)`.

**Result (Anaheim 100%, MSA, 10 iter):** TSTT monotonically increases
(68.8M → 80.6M), k/kj saturates to 1.00 by iteration 6, gap oscillates
between 0.3%–3.8% — never converges.

**Why it failed:** Using the *congested* routing speed in `k_aon` creates
a positive feedback loop.  When the network is congested, `v_routing` is low,
which inflates `k_aon = V/v_routing`.  MSA blends this inflated auxiliary
density into the state, making congestion worse, which lowers speed further.
The auxiliary density depends on the iterate itself — this violates the MSA
requirement that the auxiliary problem be independent of the current state.
Density ratchets monotonically toward jam density.

**Status:** UNRESOLVED — this is the core convergence bug.  Density-based
static equilibrium may be fundamentally problematic because the MFD
backward-bends (same flow maps to two densities).  A monotone cost function
in volume space (queue-delay surrogate) is the proposed fix.

---

### 5. Volume-Based MSA with Freeflow Conversion

**Idea:** Fix the density-based MSA feedback loop by converting volume to
density using *freeflow* speed instead of routing speed:
`k = V/(v_ff × Δt)`.  This makes `k ∝ V`, removing the state dependence.

**Result:** Systematically underestimates congestion.  At capacity flow
q_c = v_f·k_c/2, the freeflow conversion gives density = k_c/2, not k_c.
Congestion signal halved, Braess paradox weakened.

**Why it failed:** `k = V/v_ff` is only correct when the network is
uncongested.  Under congestion, vehicles travel slower so more of them
occupy the link at any instant.  The freeflow conversion ignores this
occupancy effect, producing density that is half the true value at capacity.
As user put it: "Seems like your plan doesn't work buddy."

**Commits:** Applied then reverted at `dbb6253`

---

### 6. Frank-Wolfe with Sampled Routing

**Idea:** Use FW with 10% trip sampling on large networks (Anaheim,
142 sampled trips) to reduce per-iteration routing cost.

**Result:** Beckmann gradient `g(0) = Σ c_a · dV_a` on 142 trips scaled
up 10× produces a random-sign gradient.  Line search finds alpha=0 → FW
stops after 1 iteration.

**Why it failed:** FW requires the full objective gradient to find a
descent direction.  Random sub-sampling introduces so much noise that the
gradient direction is essentially random — the Beckmann line search correctly
determines there's no descent direction.  FW fundamentally requires
full-pass routing.

**Fix applied:** Guard added rejecting networks > 100K trips from FW;
`full_pass=True` forced for FW.

---

### 7. v3 Path-Share Refinement (Simultaneous Best-Response)

**Idea:** After greedy loading, evaluate each OD's current path cost vs
shortest-path cost on frozen state.  For ODs with gap > threshold, propose
a path swap: shift volume fraction from old path to new.  Apply all
proposals simultaneously (Jacobi-style).

**Implementation:** ~750 lines: `_run_sampled_refinement()`,
`_propose_path_swap()`, `_evaluate_sample()`, per-OD ledger tracking.

**Result:** Correlated herding — all ODs see the same congested state, all
propose the same "fix" (shift to same alternate path), all applied at once.
Oscillation, no convergence.  Chicago Sketch stalled at ~1.3% gap, Braess
oscillated wildly.

**Why it failed:** Simultaneous stale-state updates in a network game create
coordination failures.  Every OD proposes a move based on the *same* frozen
snapshot, but the cumulative effect of all moves is not accounted for.  This
is the classic simultaneous best-response instability.  Needs sequential
(Gauss-Seidel) or damped (MSA) handling.

**Commits:** ~750 lines removed when replaced by MSA.

---

### 8. Reroute Epoch Convergence

**Idea:** After greedy loading, run "reroute epochs" — re-route a sample
of ODs on the current state, update density, repeat.  Track TSTT delta as
convergence metric.

**Result:** Appeared to work on Braess (+4.4% TSTT increase with shortcut),
but the convergence metric was fundamentally broken — it tracked
`|TSTT_n − TSTT_{n-1}|/TSTT_{n-1}` which converges to zero even when the
system is nowhere near equilibrium (just means TSTT stopped changing, not
that it reached Wardrop equilibrium).

**Why it failed:** TSTT stabilization ≠ equilibrium.  A system can stabilize
at any feasible state, not necessarily UE.  The apparent "success" on Braess
was coincidental — the greedy loading happened to reach near-UE for that
specific 3-link network.

**Commits:** Removed when replaced by MSA with proper Wardrop gap.

---

## Category C — Gap Computation Failures

### 9. Wardrop Gap from OSRM Table API Durations

**Idea:** Compute relative gap as
`gap = (assigned_time − shortest_time) / shortest_time` using OSRM's Table
API for shortest_time and VDF-derived costs for assigned_time.

**Result:** Negative gaps (~-0.035) because VDF float-precision speeds
produce different costs than OSRM's integer-quantized speeds (OSRM stores
speeds as integer decimetres/second, 0.36 km/h resolution).

**Fix applied:** Replaced with VDF-only gap computation where both
numerator and denominator use the same cost function, eliminating
quantization bias.

---

### 10. Wardrop Gap Using MFD Throughput

**Idea:** Numerator uses `Σ V × L/v_vdf` where V is MFD throughput
`flow_vph = k × v`.  When k > k_j (oversaturated), throughput < demand
volume, so numerator < denominator → gap < 0.

**Fix applied:** Clamped with `max(0, ...)` which *hid* the negative gaps
for months.  Eventually fixed by using demand-side volume (the volume that
*wants* to traverse the link) instead of MFD throughput (the volume that
*can* discharge).

---

## Category D — Network Design Failures

### 11. TNTP Capacity as Physical Lane Count

**Idea:** Use TNTP "capacity" column values (4,824–25,900) to derive
physical lane counts for network construction.

**Result:** Produces 13-lane urban streets.

**Why it failed:** TNTP capacity values are pure mathematical artifacts from
BPR reverse-engineering, not physical roadway properties.  They're
calibrated to reproduce BPR equilibrium flows, not to represent real
infrastructure.

---

### 12. Backward Speed Assignment on Braess Network

**Idea:** Braess diamond with variable links (congestion-sensitive) set
SLOW and constant links (robust) set FAST.

**Result:** 100% of flow uses shortcut regardless of demand — no mixed
equilibrium, no paradox.

**Why it failed:** Classical Braess requires variable links to be FAST
(high freeflow, but sensitive to congestion) and constant links to be SLOW
(low freeflow, but robust under load).  Had the assignment exactly backwards.
Flipping speeds fixed the paradox immediately.

---

### 13. S-Curve Waypoints for Braess Geometry

**Idea:** Add 8 waypoints per highway link to create smooth curved geometry
instead of ugly straight-line offsets.

**Result:** User: "networks look even more shitty."  Still didn't produce
correct paradox because the underlying speed assignment was backwards (#12).

---

## Category E — Architectural Failures

### 14. Dual Assignment Pipelines

**Status:** RESOLVED (consolidated in `d022264`)

**What happened:** Over months of iteration, two independent assignment
codepaths emerged:

- `AssignmentLoop.run()` — UE solver with incremental warm-up + MSA/FW
- `TrafficAssignmentSolver.run_stream()` — greedy time-slice loading + MSA/FW

Both had their own convergence logic, result types (`IterationResult` vs
`MSAIterationResult` vs `HillClimberBatchResult`), metric computation, and
report interfaces.  Tests covered one path but not the other.

**Why it happened:** Organic growth — `AssignmentLoop` was the original
matrix-based solver.  `TrafficAssignmentSolver` was added for the
matrix-free hill-climber use case.  Convergence logic was copied rather than
shared.

**Fix:** Merged into single `AssignmentSolver` class with unified
`IterationResult` (distinguished by `phase` field).  Deleted `solvers.py`
(843 lines).

---

### 15. OSRM Customize Irreversibility

**Status:** PARTIALLY MITIGATED

**Problem:** `osrm.customize()` with a segment-speed CSV permanently
mutates `.osrm` data files on disk.  Even re-extract from OSM won't undo
it (the cell metrics store the customized weights).

**Impact:** Every assignment iteration must copy the entire OSRM dataset to
get clean freeflow speeds for the next run.  For multi-period DTA with 96
time bins, this is untenable.

**Current mitigation:** File-copy workaround (`_copy_clean_osrm()`) for
single-period.

**Needed:** OSRM core change for in-memory weight overlays or a
pre-customize snapshot API.

---

## Lessons Learned

1. **VDF smoothing addresses the wrong bottleneck.** The gradient singularity
   at k_j only matters if density approaches k_j gradually.  With AON loading,
   density leaps past k_j by 6× in one iteration.

2. **CFW assumes smooth objectives.** The bi-parabolic MFD is C¹ at k_c but
   near-singular at k_j.  Momentum methods amplify oscillation on non-smooth
   surfaces.

3. **The loading strategy is the real lever**, not VDF shape or solver
   sophistication.  Incremental loading (25% → 100%) prevents catastrophic
   iteration-1 overshoot.

4. **Density-based static equilibrium may be fundamentally ill-posed.**
   The MFD backward-bends: same flow maps to two densities.  For q > q_c
   there is no physical density on the uncongested branch.  A monotone cost
   function in volume space is needed.

5. **Don't confuse TSTT stabilization with equilibrium.**  Any feasible
   flow pattern can stabilize.  Only the Wardrop gap (assigned cost vs
   shortest-path cost) measures actual equilibrium.

6. **Gap computation must use a single cost function.**  Mixing VDF
   float-precision costs with OSRM integer-quantized durations produces
   systematic bias and negative gaps.

7. **Simultaneous best-response is unstable in network games.**
   Jacobi-style path-swap proposals create correlated herding.  Use
   MSA-style blending or sequential updates.

8. **Organic growth creates parallel pipelines.**  Two solvers, two result
   types, two report interfaces — each sensible in isolation but
   unmaintainable together.  Consolidate early.
