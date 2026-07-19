# MZ early-onset dynamics — STATUS (2026-07-19)

Scientific object (redirect per 2026-07-19 review): **pure temporal MZ D–a phase diagram** —
inhibition-efficacy depletion `D = 1 − z̄` vs firing-adaptation `a = η_m·m̄ / I_EE_scale`, and how
their timescale competition governs whether repeated interictal events reach run-off.
Space / source-sink / rotate-shuffle / early-field-bridge are **out of this figure** (demoted, review §7).

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
