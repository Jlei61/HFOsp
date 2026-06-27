# M3A-A2 axis-break autonomous exploration — plan & decision tree (2026-06-26)

**Goal (user, 8h autonomous mandate)**: produce **off-axis (axis-breaking), self-limiting, global
synchronous** seizure-like activity in the Stage-3 two-focus substrate. Determine whether it is a
parameter problem (under-used the existing mechanism) or a mechanism problem (need a new ingredient);
if parameters are exhausted, add a mechanism per this ladder and keep sweeping.

## What "the target state" means (the read-out)

Three source-space per-cell onset metrics (computed in `scripts/plot_a2p_synchronous_burst_figure.py`
`read_events`, re-used by the sweep):
- **off-axis** = the wiring's axial texture is broken: `isotropy` (perp/along spread) **> 0.7** (round)
  OR `grad_r2` (position predicts firing order) **< 0.3** (no directed front = whole sheet ~together).
- **global** = `n_fired` large (≈ whole sheet).
- **self-limiting** = the event TERMINATES: `tonic_fraction` (duty cycle above the event bar) **low**,
  AND ideally ≥2 discrete off-axis events (a recurring cycle), global rate not pinned high.

TARGET regime = `off_axis_SELF_LIMITING` (≥2 off-axis events, tonic_fraction < 0.5, rate < 60 Hz).
Failure modes = `axial_only` (parameters too weak), `runaway_tonic` (broke axis but never terminates),
`quiet_or_tiny`.

## Why the committed best point fails the target

`core_only` only drained the LOCAL core inhibition (`q_global` stayed 1.0). Events only grew ALONG the
axis (`grad_align`=1.0, every event). The "global" half of the Abbott local+global mechanism — the part
that globally disinhibits the off-axis surround — was never engaged.

## Mechanism ladder (escalate only when the lower rung is exhausted)

- **L0 (done)** core_only depletion + g_K → axial only. Never breaks axis.
- **L1 (Phase 1, running)** `two_tank`: engage the GLOBAL inhibitory tank (q_global drops with whole-
  sheet firing). Smoke (k0.8 d0.9 q_min0.1, T=600): q_global drained to floor, event went whole-field &
  round (isotropy 0.998, grad_r2 0.04, 32000 cells = **axis broken**) BUT 254 Hz **tonic** (not discrete).
  → Phase 1 maps q_min (partial vs full drain) × k_use × gk × drive for a discrete-off-axis sweet spot.
- **L2 (parameter, if L1 = off-axis but tonic)** strengthen self-limitation with existing code:
  stronger/faster g_K (gk_max 0.12–0.3, faster tau_k) + faster q_global RECOVERY (tau_rec 500–1000) so
  inhibition returns and the event cycles. Race: g_K must suppress firing fast enough for q_global to
  recover before tonic locks in.
- **L3 (mechanism, if L2 exhausted)** add **short-term excitatory synaptic depression on E→E**: a
  per-presynaptic resource that depletes with firing and recovers slowly → during a global burst the
  recurrent excitation collapses → the event self-terminates. This is the classic seizure-termination
  mechanism and the physically-motivated next ingredient. Requires an engine change (current accumulation
  in `src/snn_engine/`), so: isolated worktree, re-bless `engine_versions.json`, TDD before any sweep.
- **L4 (substrate, last resort)** if even L3 can't make it both global AND self-limiting, the AR=2 axial
  wiring may be the bottleneck → dynamically relax connectivity anisotropy, or accept the seizure state
  needs a less anisotropic substrate.

## Decision tree per phase

```
run sweep -> aggregate (plot_a2_axisbreak_summary.py) -> any off_axis_SELF_LIMITING cell?
  YES -> refine around it (multi-seed, longer T, onset-map confirm it's genuinely off-axis + terminates)
         -> if robust across seeds: TARGET MET, write up, build the main figure (axial vs off-axis).
  NO, but off_axis_oneshot / runaway_tonic exists -> axis breaks, termination is the gap -> go L2.
  NO off-axis at all (only axial/quiet) -> global tank didn't disinhibit enough -> push L1 harder first.
  L2 exhausted (no parameter makes off-axis self-limiting) -> MECHANISM problem -> L3.
```

## Discipline
- Every phase: hypothesis -> sweep -> table + figure -> verdict -> next. Update this doc + recap §6.x.
- Screen-level language; the target is "produced a candidate off-axis self-limiting regime", NOT
  "reproduced seizures". Multi-seed before any "robust" claim.
- Engine edits (L3+) only after parameter rungs (L1/L2) are genuinely exhausted; re-bless + TDD.
