# MZ early-onset dynamics — STATUS (final)

Scientific object (redirect per 2026-07-19 review): the pure **temporal** MZ D–a dynamics — inhibition-
efficacy depletion `D = 1 − z̄` vs firing-adaptation `a = η_m·m̄ / I_EE_scale`, and whether their timescale
competition lets repeated interictal events reach, and then recover from, run-off.

## Verdict (locked)

**In E1146's registered parameter range, per-neuron inhibition-efficacy depletion drives delayed run-off;
linear firing adaptation, by slowing depletion, produces graded prevention and a bounded sub-onset plateau —
it does NOT produce a repeatable onset–containment–recovery cycle.** This is a claim about the sampled range,
NOT a param-space impossibility (strength densified only at τ_adp=2 s; τ_adp swept at one η_m). A structural
claim would need a 2-D continuation along the plateau/run-off boundary — not done, deferred.

## What was measured + validated

- **Frozen slow-off event bar** (P0-2): the event-onset threshold is frozen once from the same-seed slow-off
  and reused for every z/m cell (no per-trajectory `af.max()` pollution). Regression-tested.
- **18-cell gap grid** (`analyze_mz_onset_dynamics.py` → `gap_dynamics_summary.json`): primary regime
  `zA_q75_tz5000` × `A_target=[0,0.001,0.0025,0.005,0.0075,0.01]` × seeds 1/3/4, T=20000 ms, no early-stop
  for z+m. a=0 run-off 3/3; **a=0.001 bounded plateau** (D→0.0566, ~65% of onset, recovered 0/3, events do
  not escalate); a≥0.0025 prevention with monotonically lower D_max (0.033/0.020/0.017/0.013). No frac recovers.
- **Run-off corridor**: all 7 run-offs (3 z-only + 4 short-τ z+m) cross at **D = 0.0869 ± 0.0018** (onset
  reference = D-at-first-crossing of the a=0 cells = 0.0873). `0.0975` is the z-only post-crossing peak D_max,
  NOT onset. → adaptation decides whether the trajectory REACHES the boundary, not where the boundary is.
- **Adaptation timing** (4 z+m run-offs): a at crossing ≈ 0.0005 (weak) vs a_max ≈ 0.0078 after run-off —
  feedback too late to contain. This is the mechanism of "prevents, cannot contain".
- **τ_adp sensitivity** (review §5, near-critical a=0.001, fixed η_m=0.00745): τ=2 s plateau 3/3; τ=1 s
  plateau 2 + run-off 1; τ=0.5 s run-off 3/3. Faster recovery weakens the brake — the plateau rises toward the
  corridor then tips into run-off; no (η_m, τ_adp) recovers. τ=1 s seed 3 `expanded_returned` is a seed-fragile
  boundary phenomenon (D stays ~0.077, no slow-state recovery), not a realized cycle.
- **Figures** (`figures/`): `mz_onset_natural_trajectories.png` (ratchet time-course + D–a plane with all five
  strengths + graded-prevention readout; run-off stars at first crossing) and `mz_onset_tau_sensitivity.png`
  (τ view clipped to D≤0.13, run-off markers at D-at-crossing, linear x). 中文 `figures/README.md`.

## Machine-readable

- `gap_dynamics_summary.json` top-level: `runoff_corridor` {n:7, D_mean:0.0869, D_sd:0.0018, D_min:0.0838,
  D_max:0.0899}, `adaptation_timing_in_m_runaways` {n:4, a_at_crossing_mean:0.00053, a_post_onset_max_mean:0.00777},
  `D_onset_ref:0.0873`, per-cell `cells`, per-frac `verdict`, `conclusion`.
- `focused_m_summary.csv`: 18 rows, 18 unique (seed, A_frac), τ_adp=2000 gap grid only (no τ-sweep, no smoke).

## Fixes (review-1 §3 + review-2 P1)

P0-2 frozen event bar; P1-2 real aggregator (no parallel-writer race) + conditional early-stop; onset ref →
D-at-first-crossing corridor; figure frac-key bug (0.0025/0.0075 were silently dropped) + analyzer/plotter
τ-contamination fix (glob matched `_tau` cells on the same A_frac) + run-off stars at crossing + D_max-vs-frac
panel + τ figure repairs; summary re-aggregated to 18 rows; regression tests for the frac-key, τ-exclusion, and
18-row aggregator. Structural claim softened to the registered range.

## Next-stage handoff (NOT part of this conclusion)

Deferred by the review — do not mix into the z+m verdict: frozen fast-system α₁ phase map (needs a documented
a-axis proxy-units convention; only the a=0 D-axis slice was explored — α₁<0 up to D≈0.13, run-off below it =
finite-amplitude/heterogeneous escape), empirical slow-flow/nullclines, ε_c nonlinear ignition, spatial
susceptibility, early-field bridge. The next mechanism lever (activity threshold / nonlinear adaptation / delay)
is the user's call — not added autonomously.

## Verification

`pytest -q tests/test_mz_slow_vars.py tests/test_topic4_mz_slowvars.py tests/test_topic4_mz_onset_dynamics.py`
— focused MZ suite green. `analyze_mz_onset_dynamics.py` is post-processing only (no SNN simulation).

## Provenance

Commits `ca54037`(spec) → `b34dc79`(upstream merge) → `e3b4748`/`3b3fe91`(trajectories) → `96159ab`(P0-2/P1-2)
→ `c7b72aa`(gap grid + verdict) → `3d5bc48`(τ sweep) → `6c878ae`(review-2 P1) → this consistency commit.
Calibration `results/.../mz_slowvars/calibration.json` (I_EE_scale=272.755, peak_m_tau2000=36.60).
No push, no merge, owned files only.
