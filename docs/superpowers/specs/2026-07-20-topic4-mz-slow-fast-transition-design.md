# Topic 4 — MZ slow–fast dynamical transition (design / binding contract)

**Date:** 2026-07-20 · **Branch:** `codex/topic4-mz-slow-fast-transition` · **Baseline:** `codex/topic4-mz-onset-dynamics` @ 7477453
**Tier:** model-side mechanism analysis. Every phenotype is a *detection label*. We test whether a frozen fast
system crosses a repeatable **operational-runaway** boundary — NOT clinical seizure reproduction. The word
"seizure" does not enter any result sentence; only "operational runaway (120 Hz / 100 ms)".

---

## 0. Scientific object (plain language)

We have a spiking network of E1146 in which two *slow* quantities drift as interictal events repeat:
- **Disinhibition** `D(t) = 1 − z̄(t)` — inhibition efficacy gets used up (per-neuron `z_i` decays when the
  E-cell GABA current stays above `I_th_EI`).
- **Adaptation** `a(t) = η_m · m̄(t) / I_EE_scale` — firing fatigue accumulates (per-neuron `m_i` rises with
  each E spike, `η_m·m_i` is subtracted from the net current).

The onset-dynamics branch already drew the *temporal* D–a trajectories and found: pure disinhibition drives a
delayed run-off; linear adaptation slows depletion (graded prevention + bounded plateau) but never produces an
onset→containment→recovery cycle. That work characterized how the **slow state moves**.

**This task asks the orthogonal question:** freeze the slow state `S = {z_i, m_i}` at a moment on the natural
drift, then evolve **only** the fast system `X = {V_i, s^E_i, s^I_i, spikes, R_E}` (connectivity, threshold
landscape, noise statistics are fixed scaffold, not fast variables). Does the frozen fast system sit safely
below a tipping edge or fall over it — and does that edge behave like a **sharp threshold**, a
**finite-amplitude escape**, a **noise-driven escape**, or a **smooth crossover**? We do not presuppose which.

**Why this sidesteps the pending "a-axis proxy-units" decision:** freezing the *real per-neuron* `z_i, m_i`
and evolving the *real spiking network* keeps everything in absolute engine units. The deferred proxy-unit
convention only affected the mean-field rate-field α₁ map (`operator-grid`), which this task does NOT use.

---

## 1. The four conditions (locked cfg)

`η_m = 0.0074516` (= A_target 0.001 · I_EE_scale 272.75518960107513 / peak_m_tau2000 36.6036014019694, fixed
across τ per the onset-dynamics τ-sweep). `I_th_EI = 95.19851312666987`, `tau_z = 5000.0`, `dt = 0.1 ms`.
Substrate = `run_m4_phaseplane.build_substrate(seed)` (E1146 narrow, template_source twoend_equal, L20,
dens100, AR2). Seeds = **1, 3, 4**.

| id | label | MZSlowVarsConfig | natural behaviour (from onset-dynamics) |
|----|-------|------------------|------------------------------------------|
| `z_only`     | z-only reference | `use_z=T, use_m=F, tau_z=5000, I_th_EI=95.1985` | run-off 3/3 (onset 9293.6/9499.3/9757.9 ms) |
| `mz_runaway` | runaway | `use_z=T, use_m=T, tau_z=5000, I_th_EI=95.1985, tau_adp=500, eta_m=0.0074516`  | run-off 3/3 |
| `mz_edge`    | edge    | `…, tau_adp=1000, …` | plateau 2 + run-off 1 (seed-fragile) |
| `mz_plateau` | plateau | `…, tau_adp=2000, …` | bounded plateau 3/3 (D_max≈0.057, no crossing) |

Per-seed z-only operational-runaway **onset anchor** `O_s`: seed 1 = 9293.6 ms, seed 3 = 9499.3 ms,
seed 4 = 9757.9 ms (READ from onset-dynamics; never re-estimated). These anchor the matched-time checkpoints
for **all four** conditions.

---

## 2. Registered checkpoints

### 2.1 Matched-time (PRIMARY)
Anchored to each seed's z-only onset `O_s`, applied identically to all four conditions (union of spec §3 and
"主要工作" lists so neither reading is under-served):

| state key | time (ms) |
|-----------|-----------|
| `baseline_1000ms` | 1000 |
| `mid_fraction`    | 0.50 · O_s |
| `pre_onset_2000ms`| O_s − 2000 |
| `pre_onset_1000ms`| O_s − 1000 |
| `pre_onset_500ms` | O_s − 500 |
| `pre_onset_200ms` | O_s − 200 |
| `pre_onset_100ms` | O_s − 100 |
| `first_crossing`  | each condition's OWN first 120 Hz/100 ms crossing (censored if none within horizon) |

At each checkpoint we save the full frozen state (per-neuron `z_i, m_i` + fast-state `LoopState`) and record
the **actual** `(D, a)` the condition reached there. Plateau/edge are matched-time *controls*: their `(D, a)`
differ from runaway at the same wall-clock time — that difference IS the signal.

### 2.2 Matched-D (SECONDARY cross-check)
D targets `[0.02, 0.04, 0.06, 0.08]`. For each condition+seed, the first time `D(t) ≥ target` becomes a
checkpoint; run only the two frozen-fast probes (P_runaway, ε_c) there. Conditions that never reach a target
(e.g. plateau above ~0.057) are censored at that target — expected, and part of the result. Gives comparable
`P_runaway(D)` curves at matched D across conditions.

---

## 3. Frozen-fast-system probes (per checkpoint)

All three freeze `z_i, m_i` (via `MZOnsetProbe.set_branch(freeze=True)`) and continue ONLY the fast system.
No spatial maps, no source/sink, no off-axis: probes act on **all E** neurons or none.

### 3.1 Perturbation-free escape probability `P_runaway(state; T)`
Same frozen fast state, **N=20 independent future-noise branches**, horizon `T=500 ms`. `P_runaway` = fraction
of branches meeting the 120 Hz/100 ms operational-runaway criterion (`score_runaway`). Report Wilson 95% CI.
- Independence: each branch resumes from a **copy** of the checkpoint `LoopState` with `V, currents, rings, xi,
  z, m` identical, but `rng_state` replaced by a deterministic independent PCG64 stream keyed on
  `(base_seed, condition, state, branch_idx)` (`branch_rng_state`). Zero engine edits.
- ε = 0 (NO perturbation). This measures how often the frozen state runs away on noise variability alone.

### 3.2 Global nonlinear ignition threshold `ε_c(state)`
Deterministic, brief (10 ms), **global** threshold-lowering probe applied to **all E** neurons (units = vth-gap
`a·median(V_th − V_reset)`; consistent with the onset-dynamics ignition ladder). Future noise = the checkpoint's
native stream (fixed) so ε_c is a property of the state, not noise luck. Amplitude ladder
`[0.0, 0.025, 0.05, 0.10, 0.20]` + 2 bisection refinements. `ε_c` = smallest amplitude that ignites 120/100
runaway within horizon (`epsilon_c_from_ladder`); `ε_c=0` if it runs away with no probe (native noise);
right-censored if no rung ignites. This is the *inverse* question to §3.1 (minimal push vs. spontaneous luck).

### 3.3 Fast-rate recovery time `τ_rec(state)`
A single deterministic, explicitly **subthreshold** global pulse (fixed amplitude `a_rec = 0.02` vth-gap, below
the smallest ignition rung; verified non-igniting or the cell is flagged). Measure the time for the smoothed
E-rate excess to return to the frozen state's pre-pulse band (`pre_mean ± k·pre_std`, k=1) and stay. Censored if
it never returns (runs away or stays elevated). Near a tipping edge τ_rec should lengthen (critical slowing).

---

## 4. State-matched M/Z counterfactuals

Branch point = `pre_onset_100ms` fast-state (fixed for all five). Only the frozen slow fields `(z, m)` written
onto the copied slow object vary; connectivity, threshold landscape, and pre-branch noise are unchanged. "early"
= `mid_fraction` snapshot, "late" = `pre_onset_100ms` snapshot.

| branch | z source | m source | asks |
|--------|----------|----------|------|
| `native_zm`      | pre_100 | pre_100 | reference |
| `native_z_reset_m` | pre_100 | 0 (reset) | does removing accumulated adaptation change escape? |
| `reset_z_native_m` | 1 (reset) | pre_100 | does restoring inhibition efficacy prevent escape? |
| `late_z_early_m`   | pre_100 | mid | is instantaneous instability set more by z than m? |
| `early_z_late_m`   | mid | pre_100 | (symmetric contrast) |

For each: `P_runaway` (§3.1), `ε_c` (§3.2), `τ_rec` (§3.3). This isolates whether **z** controls proximity to
the escape boundary and whether **m** moves the boundary or merely slows arrival.

---

## 5. Result-neutral transition classification (spec §6)

A transparent, thresholded classifier `classify_transition(per_state)` over a condition's ordered checkpoints
(features: `D, a, P_runaway(+CI), ε_c(or censored), τ_rec(or censored)`, plus natural-crossing / plateau-outside
flags). No category is a gate; the function returns the label **and** the feature vector it used.

- `dynamical_tipping` — `P_runaway` steps ~0→~1 across a narrow D interval **and** `ε_c → ~0` near that D **and**
  `τ_rec` increases toward the edge **and** the natural runaway trajectory crosses there while plateau stays below.
- `finite_amplitude_escape` — `P_runaway(ε=0)` stays low across states (frozen states bounded under noise) **but**
  `ε_c` decreases with D and strong perturbation gives a non-returning high state (basin dependence).
- `noise_driven_escape` — `P_runaway` rises **smoothly** with state (no sharp step), no clear `ε_c` threshold,
  wide branch-to-branch CI.
- `smooth_crossover` — only rate/variance drift; `ε_c` flat/censored, `τ_rec` flat, `P_runaway` flat-low; no
  boundary behaviour.
- `unresolved` — too few resolved checkpoints.

Allowed terminology: "slow-state-controlled dynamical transition / tipping". FORBIDDEN: "thermodynamic phase
transition" (would need finite-size N scaling, not done), "seizure", "proves/reproduces onset".
**Not presupposed:** the onset-dynamics branch's D-axis α₁ hint leaned finite-amplitude, but the label here is
decided by the fast-system data, not inherited.

---

## 6. Reuse map (NO engine edits, NO reinvention)

From `src/topic4_mz_onset_dynamics.py` (import): `MZOnsetProbe`, `run_loop` (checkpoint/resume via `LoopState`),
`LoopState`, `score_runaway`, `epsilon_c_from_ladder`. From `src/topic4_mz_slowvars.py`: baseline helpers if
needed. Substrate from `run_m4_phaseplane`. The 6 guarded engine files
(`kick_probe/params/model/connectivity/connectivity_rot/lfp.py`) are read-only; their SHAs are recorded in
provenance and asserted unchanged (no re-bless).

**New, TDD'd, import-safe** in `src/topic4_mz_slow_fast_transition.py`:
`branch_rng_state`, `wilson_ci`, `recovery_time`, `classify_transition`, plus small pure helpers
(`state_step_schedule`, `matched_d_times`, counterfactual `(z,m)` assembly).

**Helper-reuse question-match check (CLAUDE.md §6.1):** the onset-dynamics `cmd_ignition` null asks "minimal
*focal* (source/sink/off-axis) probe to ignite". Our §3.2 asks "minimal *global* probe to ignite" — different
target set → we call the SAME primitives (`set_probe`, `epsilon_c_from_ladder`) with an all-E mask, not the
focal `_probe_targets`. §3.1 (P_runaway over noise) has NO onset-dynamics analogue (forks there reuse one noise
draw) → genuinely new.

---

## 7. New files + results layout

```
scripts/run_topic4_mz_slow_fast_transition.py     # pilot / run / aggregate subcommands (sims gated by --confirm-run)
scripts/plot_topic4_mz_slow_fast_transition.py     # 4-panel figure
src/topic4_mz_slow_fast_transition.py              # pure, import-safe, testable functions
config/topic4_mz_slow_fast_transition.yaml         # locked params
tests/test_topic4_mz_slow_fast_transition.py       # TDD
docs/archive/topic4/sef_hfo/mz_slow_fast_transition_2026-07-20.md   # archive verdict
results/topic4_sef_hfo/mz_slow_fast_transition/
  per_state/<condition>_seed<S>_<state>.json       # atomic per-(cond,seed,state) raw (P_runaway branches, ε_c ladder, τ_rec)
  per_state/<condition>_seed<S>_natural.npz         # natural D/a/rate trajectory + crossing + events
  counterfactual/<condition>_seed<S>.json           # (only conditions where relevant; runaway/edge/plateau + z_only)
  matched_d/<condition>_seed<S>.json                # secondary cross-check
  slow_fast_transition_summary.csv / .json          # aggregate (per state + per condition classification)
  figures/{mz_slow_fast_transition.png,.pdf,README.md}
  STATUS.md  provenance.json
```

Directory + file naming carries topic, not PR number (AGENTS.md). Atomic per-job writes; a **separate**
aggregate step (no parallel writer touches the shared CSV/JSON).

---

## 8. Resource discipline

- **Pilot first:** 1 seed × 1 condition × 1 checkpoint × few replays → measure peak RSS + wall/step. Print RSS.
- Parallel unit = one `(condition, seed)` job (does its full trajectory + all checkpoints + all forks serially
  inside the job). **No nested parallelism.** `OMP/MKL/OPENBLAS_NUM_THREADS=1`.
- Workers `W = min(nproc−2, floor((avail_RAM − margin) / peak_RSS_per_job))`, `margin = max(30 GB, 25% avail)`.
  Recomputed against **live** `available` RAM (other sessions/jobs are running). Back off on swap or growth.
- Idempotent `--resume` (skip completed per-(cond,seed,state) JSON). Long natural sims never repeated.
- Aggregate is a separate no-sim step after all jobs finish.

---

## 9. Figure (4 panels, one argument each — CLAUDE.md §7)

Read `docs/figure_style_guide.md` Topic 4 section first; reuse fonts/colors/linewidth/layout from
`scripts/plot_topic4_mz_onset_dynamics.py` + `plot_topic4_mz_onset_tau.py`. Fixed distinguishable colors for
runaway / edge / plateau / z-only. Mark crossing, right-censoring, seed + noise-replay uncertainty. No spatial
maps, no diagnostic clutter. PNG + PDF. `figures/README.md` in Chinese, per-panel argument + 关注点.

- **A** — onset-aligned `R_E, D, a` for the four conditions (natural trajectories).
- **B** — frozen-fast `P_runaway` vs natural slow state (D), crossing marked, CI bands.
- **C** — `ε_c` and `τ_rec` vs slow state (twin axes / stacked), censoring marked.
- **D** — counterfactuals: native / m-reset / z-reset / late-z-early-m / early-z-late-m on `P_runaway` (or `ε_c`).

---

## 10. Test contract (TDD, write first)

Pure functions:
1. `branch_rng_state(seed, cond, state, idx)` — deterministic; distinct `(idx)` → distinct PCG64 states; same
   inputs → identical state; type is PCG64 (swappable into `LoopState.rng_state`).
2. `wilson_ci(k, n)` — matches known reference values; `k=0`/`k=n` bounded in [0,1]; monotone.
3. `recovery_time(rate, dt, pulse_off_idx, band_lo, band_hi, …)` — synthetic decaying trace returns finite
   time; never-returning trace → None (censored); already-in-band → ~0.
4. `classify_transition(per_state)` — synthetic feature sets that must map to each of the 5 labels (one test
   each); asserts the returned feature vector is populated.
5. `state_step_schedule(O_s, dt)` / `matched_d_times(D_trace, targets)` — correct step indices; censored target
   → None.

Integration/smoke (tiny, `--confirm-run` gated or short T):
6. Frozen plateau baseline state with ε=0 over a few replays → low `P_runaway` (bounded); a global probe at high
   amplitude → runaway (sanity that the probe path fires). Engine SHAs unchanged assertion.
7. `run_loop` resume-from-copied-checkpoint with swapped `rng_state` yields a DIFFERENT trace than the native
   resume (independence sanity) while native resume reproduces the reference bit-for-bit.

---

## 11. Non-goals (explicit)

No spatial eigenmode, principal-axis decomposition, source/sink, off-axis, or field bridge. No modification of
the locked onset-dynamics runner/config/results. No push, no merge, no `git add -A`.
