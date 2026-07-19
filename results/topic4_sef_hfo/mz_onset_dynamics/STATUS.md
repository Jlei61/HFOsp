# MZ early-onset dynamics — STATUS (2026-07-19)

Scientific object (redirect per 2026-07-19 review): **pure temporal MZ D–a phase diagram** —
inhibition-efficacy depletion `D = 1 − z̄` vs firing-adaptation `a = η_m·m̄ / I_EE_scale`, and how
their timescale competition governs whether repeated interictal events reach run-off.
Space / source-sink / rotate-shuffle / early-field-bridge are **out of this figure** (demoted, review §7).

## ★ Latest verdict — z–m gap grid (review 2026-07-19 §6; SUPERSEDES the trajectory-era claims below)

Densified the previously-unsampled gap `0 < eta_m < 0.0745` (`A_target=[0,0.001,0.0025,0.005,0.0075,0.01]`
× seeds 1/3/4, T=20000ms, no early-stop for z+m, FROZEN slow-off event bar). Verdict
(`gap_dynamics_summary.json`, `analyze_mz_onset_dynamics.py`):
**graded prevention with a bounded sub-onset PLATEAU — NOT a containment–recovery cycle.**
- a=0 runaway 3/3 (run-off onset D≈0.087, corridor 0.0869±0.0018; post-crossing peak 0.0975). a=0.001
  (eta_m=0.0075): D **ratchets** to ~0.056 (**~65% of the 0.087 onset**), bounded,
  **recovered 0/3** (stays elevated); events do NOT escalate → plateau, not a discrete recruited event.
- a≥0.0025: prevention, D_max monotonically lower (0.036 / 0.020 / 0.017 / 0.013). No frac recovers.
- **Minimal linear spike-adaptation PREVENTS/STALLS onset (graded); it does not contain-and-recover.**

Two corrections to the trajectory-era claims below (both retracted):
1. NOT binary — an intermediate bounded-elevated regime exists at a=0.001 (was: "binary runaway↔prevention").
2. "interictal events persist in all non-runaway conditions" was a per-trajectory event-bar artifact; the
   frozen slow-off bar (P0-2 fix) shows strongly-suppressed cells are near-silent, not interictal-preserved.

Fixes landed (review-1): P0-2 frozen slow-off event bar, P1-2 real aggregator, conditional early-stop, relabel.
Finalization (review-2 P1): onset ref → D-at-first-crossing (run-off corridor 0.0869±0.0018, NOT post-crossing
0.0975 → a=0.001 = 65% not 57%); figure frac-key bug fixed (0.0025/0.0075 were silently dropped from the D–a
plane) + analyzer/plotter tau-contamination fixed + run-off stars moved to onset + D_max-vs-frac readout panel
+ tau figure clipped to D≤0.13 / D-at-crossing / linear x-axis / fixed-η_m title; summary re-aggregated to
18 rows (was a 2-row smoke artifact); plot-key + aggregator regression tests (38 pass). Structural claim
softened to "in the registered sampling range" (not a param-space ceiling). **Corridor finding: all 7 run-offs
cross at D≈0.087 — m decides whether the trajectory REACHES the boundary, not where it is; m is weak at
crossing (~0.0005) and only strong after (~0.008), too late to contain.** Commits `96159ab`, `c7b72aa`, +this.

Near-critical tau_adp sweep (review §5: 0.5/1 vs 2s at a=0.001) DONE. Faster adaptation recovery does NOT
enable a cycle — it weakens the brake: tau=2s plateau 3/3 (D~0.056); tau=1s plateau 2 + runaway 1
(seed4 D=0.58); tau=0.5s runaway 3/3 (D~0.75-0.83, ~477 Hz). The sub-onset plateau rises toward onset then
tips into full runaway; no (eta_m, tau_adp) recovers. **Locks the honest §6.6 conclusion: minimal linear
spike-adaptation only prevents/stalls onset (moving the stall-point toward onset then losing control), it
does not contain-and-recover a seizure.** Next mechanism lever (activity threshold / nonlinear adaptation /
delay) is the user's call — not added autonomously.

Branch `codex/topic4-mz-onset-dynamics` (off merged MZ base). Absolute-SNN-unit legs are unambiguous;
one proxy-units question (frozen a-axis) is deferred to the user — see Pending.

## P0 fixes (review §3) — status
- **P0-3 (no z+m trajectory)** — FIXED. `use_m` was off; the natural z+m grid is now run with the
  calibrated `eta_m = eta_m_from_frac(frac, I_EE_scale=272.755, peak_m=36.60)`, `tau_adp=2000`.
- **P0-1 (A-axis = mu_core proxy) / P0-2 (wrong scale)** — apply to the FROZEN grid only; not yet
  rebuilt (the old `cmd_operator_grid` mu_core proxy is untouched/unused; new frozen grid pending).

## Done + validated
- **Trajectory extractor** `natural_zm_trajectory` (D=1-z̄, a=η_m·m̄/I_EE, downsample) — TDD, 17/17
  file green. Wired into `cmd_focused_m` (saves per-cell continuous npz + slow-off reference).
  Commits `e3b4748` (code), `3b3fe91` (results+figure).
- **Natural z+m grid**: primary regime `zA_q75_tz5000` × adaptation {0,0.01,0.025,0.05,0.10,0.20} ×
  seeds {1,3,4}, T=15000ms. 21 trajectory npzs (`per_seed/traj_*.npz`), phenotype summary
  (`focused_m_summary.csv`).
- **Figure** `figures/mz_onset_natural_trajectories.png/pdf` (Panel A push-pull time course +
  Panel C D–a state plane), 中文 README.

## Key findings (validated)
1. **z alone → run-off, 3/3 seeds** at 9294/9499/9758 ms (matches locked onsets). The "push".
2. **Adaptation a ≥ 0.01 prevents run-off across all 3 seeds** — interictal events persist but each is
   capped, so z̄ barely depletes (D ≤ 0.014) and the D-buildup never reaches run-off. The "pull".
   In the D–a plane: z-only travels right along the D axis; adaptation redirects the path up the a axis.
3. **Seed-inconsistent at high adaptation**: a=0.10–0.20 fully suppresses seed 1 but seeds 3/4 stay
   interictal. Reported honestly; not smoothed.
4. **No strength yields a bounded/retriggerable run-off** — consistent with the review caveat that a
   clean z+m bounded seizure-like state may not exist yet. Reporting stays at
   "state-dependent early recruitment / run-off transition", NOT "seizure".
5. **Frozen D-axis α₁ (a=0)**: on the two-core scaffold with uniform inhibition-efficacy, α₁ rises with
   D but stays negative (stable) until D≈0.13–0.14 (then saturates to a runaway branch); leading mode
   goes global→local near D≈0.10. Natural z-only run-off (mean-D≈0.096) is BELOW this uniform-D linear
   boundary → run-off is not a simple linear crossing but finite-amplitude / heterogeneous escape.
   Caveat: mean-D vs local-D and uniform vs heterogeneous depletion not matched.

## Pending
- **[NEEDS USER DECISION] Frozen a-axis proxy-units convention** — the natural-trajectory `a` is
  absolute (`η_m·m̄/I_EE_scale`), but the frozen rate-field operator is proxy-normalized (muE~0.6).
  There is no canonical MZ-`a`→proxy-muE mapping in the code (the criticality `slow_to_ratefield.g_K`
  mapping is a different slow variable). Injecting `a·I_EE_scale` raw would blow up. Options: express
  the a-axis as a fraction of the operator's own excitatory reference (documented proxy), or a
  user-specified scale. Deferred; D-axis (a=0) done meanwhile.
- **Frozen 2D (D,a) α₁ map + leading frequency + boundary classification** (Panel B) — needs the
  a-axis decision; `state_operator` needs an optional `gK_field` passthrough (byte-identical default).
- **Empirical slow-flow + nullclines** (absolute units, SNN clamp; no proxy issue) — not started.
- **Sparse ε_c nonlinear ignition + continuation/hysteresis** (Panel D) — not started.
- **Full 4-panel figure + archive doc** `docs/archive/topic4/sef_hfo/mz_onset_dynamics_2026-07-19.md`.

## Provenance
- Config `config/topic4_mz_onset_dynamics.yaml`; calibration `results/.../mz_slowvars/calibration.json`
  (I_EE_scale=272.755, peak_m_tau2000=36.60). Frozen operator: w_ee_mult=1.05, ratio=1.0, q_floor=0.05,
  grid_n=12. Commits: `ca54037` (spec), `b34dc79` (upstream merge), `e3b4748`, `3b3fe91`.
- No push, no merge. Owned files only.
