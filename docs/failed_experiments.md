# Failed Convergence Experiments

Record of approaches tried and abandoned during development of the
density-based traffic assignment with bi-parabolic MFD VDF.

All experiments tested on **Anaheim** (416 nodes, 914 links, 104,694 vph)
at **100% demand, 50 iterations** unless noted.

Baseline for comparison (vanilla FW, no modifications):
- Gap: 0.005, ρ (Spearman vs BPR reference): 0.813, max k/k_j: 6.36

---

## 1. Conjugate Frank-Wolfe (CFW)

**Idea:** Add momentum from prior search directions to accelerate FW
convergence. Standard CFW formula (Mitradjieva & Lindberg 2013):
β = ∇Z·d_fw / (∇Z·(d_fw − d_prev)), conjugate direction = d_fw + β·d_prev.

**Implementation:** Added `_beckmann_gradient()` (link cost vector) and
`_cfw_beta()` with β capped at 0.5. Line search extended with
dk/dv override parameters to use conjugate direction.

**Result:** Gap 0.015, ρ 0.771, max k/k_j 14.67 — **worse than vanilla FW
on every metric.**

**Why it failed:** The MFD's near-singular gradient at k_j creates a
non-smooth cost surface. β oscillates between 0 and ~1.0; the conjugate
direction overshoots, pushing density to 12–15× k_j on bottleneck links.
Even with β capped at 0.5, the momentum amplifies oscillation rather than
damping it. CFW assumes a smooth quadratic-like objective; the bi-parabolic
MFD violates that assumption near jam density.

**Commits:** `84c7a67` (added), `7947df8` (removed)

---

## 2. Stretched k_j (VDF smoothing, approach A)

**Idea:** Use k_j_eff = k_j × (1 + ε) in the congested branch equation so
the parabola reaches zero speed at k_j_eff instead of k_j. At physical k_j,
speed is small but nonzero, gradient is finite.

**Result:** Did not improve gap. Density overshoots to 6× k_j regardless,
so the singularity at k_j_eff is reached just as easily. The cliff is just
moved, not removed.

**Why it failed:** FW's AON step loads entire OD volumes onto shortest paths
in one shot. Bottleneck links get 6× k_j in iteration 1. Whether the
singularity is at k_j or 1.1×k_j is irrelevant when density is at 6×k_j.
The problem is the *magnitude of overshoot*, not the *location of the
singularity*.

**Commits:** `ba5a3e9` (added to theory report), `0c2a779` (replaced by
3-way comparison)

---

## 3. Exponential Tail (VDF smoothing, approach B)

**Idea:** Replace the congested parabola for k > k_s = 0.85·k_j with an
exponential decay: v(k) = v(k_s) · exp(−B·(k − k_s)), where
B = −v'(k_s)/v(k_s). C¹-continuous splice, asymptotic decay, no zero
crossing, no gradient singularity at any density.

**Implementation:** Added `splice_ratio` parameter to `BiParabolicVDF`.
Vectorized tail computation with per-link v_s, B coefficients.

**Result (Anaheim 100%, FW, splice=0.85):** Gap 0.012, ρ 0.812,
max k/k_j 6.37 — **slightly worse than vanilla FW** (gap 0.005).

**Why it failed:** Same root cause as stretched k_j. The exponential tail
does eliminate the gradient singularity (mathematically correct), but FW's
overshoot is so extreme that links are deep in the tail where all VDF shapes
return near-floor speed anyway. The VDF shape doesn't matter when density is
6× k_j — every function returns ~0.01 km/h there.

**Key insight:** The convergence problem is not the VDF shape. It's the
loading strategy: iteration 1 dumps 100% of demand onto freeflow shortest
paths, creating catastrophic bottleneck density that no VDF smoothing can
rescue.

**Commits:** `89829ab` (added), `7947df8` (removed)

---

## Lessons Learned

1. **VDF smoothing addresses the wrong bottleneck.** The gradient singularity
   at k_j matters only if density approaches k_j gradually. With FW's AON
   step, density leaps past k_j by 6× in one iteration.

2. **CFW assumes smooth objectives.** The bi-parabolic MFD is C¹ at k_c but
   has a near-discontinuity at k_j. Momentum-based methods amplify
   oscillation on non-smooth surfaces.

3. **The real lever is loading strategy**, not VDF shape or solver
   sophistication. Incremental demand loading (ramp from 25% → 100%) would
   prevent the initial overshoot that causes all downstream problems.
