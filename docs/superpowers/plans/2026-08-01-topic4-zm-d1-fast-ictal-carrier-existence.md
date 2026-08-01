# Topic 4 Z/M D1 fast ictal-carrier existence — implementation plan

**Date:** 2026-08-01
**Status:** READY TO EXECUTE AFTER REVIEW COMMIT
**Scope:** D1 only; frozen-slow fast-carrier existence on the unchanged current-based Z/M SNN
**Spec:** `docs/superpowers/specs/2026-08-01-topic4-zm-d1-fast-ictal-carrier-existence-design.md`

## Goal

Test one minimal, already implemented mechanism: whether an E-only per-neuron
dynamic-threshold increment can turn the Phase-C tonic high-rate continuation
into a bounded, non-tonic, spatially structured, perturbation-returning and
virtual-SEEG-readable fast carrier while preserving the original dynamic
interictal baseline.

The plan does **not** test spontaneous entry, native offset, postictal recovery,
multicycle behavior, stimulation efficacy or the full ictal lifecycle.  A D1 GO
only opens a later D2 spec.

## Fixed scientific and engineering boundaries

- Use the exact E1146 `twoend_equal` current-based per-neuron Z/M substrate:
  `NE=32000`, `NI=8000`, `tau_z=5000 ms`, `tau_adp=500 ms`, `eta_m=0.001`,
  recurrent-only `S_G` with `alpha_G=16` and `tau_S=80 ms`.
- Keep E→E graph, weights, anisotropy, orientation, STD/plasticity, external
  drive/noise law, base thresholds, resets and electrode geometry immutable.
- Disable the Phase-D conductance path.  Do not reopen its calibration lattice.
- The only new active mechanism is existing `slow_field.py` `phi_increment`:
  E-only spike jumps plus exact exponential recovery.
- Use the six locked settings `tau_phi={60,100,160} ms × f_phi={0.15,0.30}`.
  Convert Hz and ms correctly:

  `delta_phi_mV = f_phi * (V_th - V_reset) / ((tau_phi_ms/1000) * r_core_ref_hz)`.

- AI is not a primary acceptance requirement.  The primary object is an
  observation-consistent virtual-SEEG carrier with bounded non-tonic and spatial
  dynamics.
- No candidate-dependent threshold, window, mask, parameter insertion or
  automatic retuning is allowed.
- Every positive/negative cell must be resumable, provenance-bound and paired to
  its exact checkpoint and future-noise bank.

## Stop hierarchy

Stop at the earliest applicable state:

1. missing/changed source or real-data reference →
   `BLOCKED_input_or_observation_reference`;
2. all six phi settings fail baseline preservation →
   `NO_GO_D1_baseline_not_preserved`;
3. no seed-1 setting passes both bounded-mid phases →
   `NO_GO_D1_fast_carrier_not_formed`;
4. survivors are burst trains, tonic/saturated, non-spatial, fail vSEEG or fail
   perturbation return → the corresponding registered D1 negative;
5. only seed-1 passes → `virtual_seeg_carrier_candidate_seed1`;
6. replication, `dt/2` and perturbation return all pass →
   `fast_ictal_carrier_supported`.

No stop state may be relabeled as entry, offset, recovery, control or lifecycle.

---

## Task 1 — Close the upstream scientific audit

**Files**

- `src/topic4_lifecycle_feasibility.py`
- `scripts/run_topic4_lifecycle_feasibility.py`
- `tests/test_topic4_lifecycle_feasibility.py`
- `docs/archive/topic4/sef_hfo/zm_lifecycle_feasibility_screen_2026-08-01.md`
- `docs/topic4_sef_hfo.md`
- `docs/paper_overview.md`

- [ ] Keep the four scale calculations, but label them design-risk diagnostics.
- [ ] Remove every executable or prose path that converts them into a proof that
  the registered substrate cannot support a carrier or lifecycle.
- [ ] Unit-test that the machine verdict is only `diagnostic_risks_present` or
  `no_diagnostic_flags` and that all lifecycle fields remain unestablished.
- [ ] Run:

  ```bash
  python -m pytest -q tests/test_topic4_lifecycle_feasibility.py
  python scripts/run_topic4_lifecycle_feasibility.py
  ```

- [ ] Confirm the JSON version and archive wording agree exactly.
- [ ] Commit this audit independently before any D1 execution code.

**Acceptance:** engineering-green diagnostics with no hard non-existence claim.

## Task 2 — Lock D1 source, mechanism and unit contract

**Create**

- `src/topic4_zm_d1_contract.py`
- `scripts/lock_topic4_zm_d1.py`
- `tests/test_topic4_zm_d1_contract.py`

**Reuse/read only**

- `src/topic4_zm_fast_carrier_contract.py`
- `src/topic4_zm_fast_carrier_state.py`
- `results/topic4_sef_hfo/zm_phase_c_tonic_identity/phasec_input_manifest.json`
- `results/topic4_sef_hfo/zm_phase_c_tonic_identity/phasec_futility_verdict.json`
- `results/topic4_sef_hfo/zm_branch_decision/phase0/canonical_config.json`

- [ ] Write failing tests for exact seed `{1,3,4}` source hashes, source state
  names, population sizes, Z/M constants, E→E semantic hashes and disabled
  conductance path.
- [ ] Make the manifest fail closed if Phase-C evidence claims more than 59/60
  seed-1 C1 runs or a full three-seed negative.
- [ ] Implement the six phi rows using the explicit `tau_ms/1000` conversion;
  store source units, converted seconds, reference rate and resulting mV/spike.
- [ ] Test against a hand calculation so a factor-1000 regression fails.
- [ ] Hash the spec, plan, source checkpoints, exact noise banks, gate version,
  observation lock and parameter rows.
- [ ] Make publication write-once and idempotent only for identical content.
- [ ] Emit all D2–D6 fields as `not_tested` / `not_established`.

**Command**

```bash
python -m pytest -q tests/test_topic4_zm_d1_contract.py
python scripts/lock_topic4_zm_d1.py --check-only
```

**Acceptance:** one immutable D1 manifest; no SNN run is authorized before it
exists and validates from disk.

## Task 3 — Build and freeze the real-data observation sidecar

**Create**

- `src/topic4_zm_d1_observation.py`
- `scripts/lock_topic4_zm_d1_observation.py`
- `tests/test_topic4_zm_d1_observation.py`

**Canonical lineage**

- `scripts/paper_figures/plot_fig3_raw_spectral_context.py`
- `results/paper-ready-figure/fig3a_raw_spectral_context/figures/epilepsiae_1146_seizure_07_raw_spectral_context_summary.json`

- [ ] Reuse the same raw loader, CAR transform, 15 contacts, `SCL9`, frequency
  bands, baseline `[-120,-90)` and clinical `[0,10)` window.
- [ ] Recompute, do not hand-copy: duration above 6 dB, occupancy, maximum gap,
  mean/peak dB, active-contact fraction, dominant frequency and spectral entropy
  for 30–80, 80–150 and 1–150 Hz.
- [ ] Preserve the accepted means as a cross-check: about 23.34, 11.48 and
  16.22 dB respectively.  Fail if recomputation drifts outside a documented
  numerical tolerance.
- [ ] Store raw source path/hash, seizure ID, contact ordering, sampling rate,
  reference, code hash and exact feature definitions.
- [ ] Unit-test time masks, dB baseline, gap calculation, entropy and a missing
  source fail-closed path using synthetic arrays.
- [ ] State explicitly that this is a representative descriptive comparator,
  not a patient likelihood or cohort range.

**Command**

```bash
python -m pytest -q tests/test_topic4_zm_d1_observation.py
python scripts/lock_topic4_zm_d1_observation.py --check-only
```

**Acceptance:** immutable observation sidecar or a blocking verdict.  The D1
carrier run must not silently fall back to source-rate-only classification.

## Task 4 — Audit the existing phi hook and state migration

**Modify only if an audit proves necessary**

- `src/topic4_zm_fast_carrier_state.py`
- `src/topic4_zm_fast_carrier_runtime.py`
- `src/snn_engine/slow_field.py`

**Tests**

- `tests/test_zm_dynamic_threshold.py`
- `tests/test_topic4_zm_fast_carrier_state.py`
- `tests/test_topic4_zm_fast_carrier_runtime.py`
- new `tests/test_topic4_zm_d1_runtime.py`

- [ ] Prove `use_phi=False` returns the base threshold object/value unchanged
  and preserves the historical baseline hash.
- [ ] Prove enabled phi is E-only, I entries remain exact zero, decay is
  `exp(-dt/tau_phi)`, and every E spike adds exactly `delta_phi` once.
- [ ] Prove `slow.phi_increment` is the only inserted state field and is zero at
  dynamic-baseline start and frozen-state fork start.
- [ ] Prove the current-based path is used and `use_zm_conductance=False`.
- [ ] Prove frozen D1 means only `z` and `m` are frozen: membrane effects,
  `S_G`, phi, fast E/I state, delays and noise remain active.
- [ ] Add the uniform E-threshold diagnostic offset outside persistent state for
  the 50 ms return probe only; it must not alter I thresholds or write a state
  reset.
- [ ] Do not edit guarded E→E files.  If the hook is already sufficient, leave
  engine code untouched and add only tests/wrapper code.

**Command**

```bash
python -m pytest -q \
  tests/test_zm_dynamic_threshold.py \
  tests/test_topic4_zm_fast_carrier_state.py \
  tests/test_topic4_zm_fast_carrier_runtime.py \
  tests/test_topic4_zm_d1_runtime.py
```

**Acceptance:** byte-parity off path and scientifically exact enabled path.

## Task 5 — Implement dynamic-interictal baseline preservation

**Create**

- `src/topic4_zm_d1_baseline.py`
- `scripts/run_topic4_zm_d1_baseline.py`
- `tests/test_topic4_zm_d1_baseline.py`

- [ ] Write pure analyzers for paired event count, median duration, median core
  peak, all-sheet mean rate, peak active fraction, two-core readability,
  pathology-axis sign and phi inter-event decay.
- [ ] Use the exact native 8.5 s window and paired replay noise; do not infer a
  new onset window from candidate traces.
- [ ] Implement every ±20% and 80%-interval criterion literally and separately.
- [ ] Distinguish `invalid_baseline_setting` from a carrier negative.
- [ ] Save binned core/source/sink rates, all-sheet active fraction, virtual-SEEG
  traces, phi core/surround/maximum traces and resource receipt.
- [ ] Synthetic tests must catch prevention, changed event order, one-core loss,
  silent sheet, excessive phi carryover and false pass from division by zero.

**Command**

```bash
python -m pytest -q tests/test_topic4_zm_d1_baseline.py
python scripts/run_topic4_zm_d1_baseline.py --smoke --seed 1 --row 0
```

**Acceptance:** the smoke run is written to a smoke-only path and can never be
consumed by the production verdict.

## Task 6 — Implement frozen-state carrier runner

**Create**

- `src/topic4_zm_d1_runner.py`
- `scripts/run_topic4_zm_d1_carrier.py`
- `tests/test_topic4_zm_d1_runner.py`

- [ ] Load only the four locked Phase-C states:
  `bounded_mid__{rising,peak}` and `bounded_late__{rising,peak}`.
- [ ] Validate complete state/RNG/checkpoint hashes before allocation.
- [ ] Freeze per-neuron z and m values with their membrane effects active;
  initialize phi to zero; keep dynamic `S_G` and exact delay rings.
- [ ] Run 6 s: 1 s burn-in + 5 s adjudication, with no onset kick.
- [ ] Save enough raw/binned state to recompute all source, vSEEG, spatial,
  saturation, E/I-lag and phi metrics offline.
- [ ] Atomically publish NPZ, JSON and resource receipt.  Existing valid output
  is immutable; incomplete output is quarantined, never overwritten in place.
- [ ] Tests must catch wrong fork, wrong fast phase, z/m drift, phi nonzero start,
  noise reuse drift and adjudication leakage from burn-in.

**Command**

```bash
python -m pytest -q tests/test_topic4_zm_d1_runner.py
python scripts/run_topic4_zm_d1_carrier.py --smoke --seed 1 \
  --state bounded_mid__rising --row 0
```

## Task 7 — Implement carrier and spatial adjudication

**Create**

- `src/topic4_zm_d1_metrics.py`
- `tests/test_topic4_zm_d1_metrics.py`

**Reuse**

- `src/topic4_zm_carrier_gate_v2.py`
- Phase-C corrected-v2 saturation and morphology helpers where contracts match

- [ ] Call `carrier_gate_v2.1_revised_2026-07-24` rather than reimplementing it.
- [ ] Compute bounded persistence, occupancy, gap, modulation depth, periodic or
  relayed organization, full vSEEG Gate B, two-zone recruitment, axial
  first-passage, simultaneous-flash rejection and combined refractory saturation.
- [ ] Keep source carrier, virtual-SEEG carrier, spatial pattern, microscopic
  saturation and real-reference comparison as separate fields.
- [ ] AI/regularity is secondary and cannot veto an otherwise valid carrier.
- [ ] Distinguish HFO-like burst train, tonic/saturated, whole-sheet flash,
  silence, runaway and technically indeterminate traces.
- [ ] Synthetic fixtures must include at least: valid relayed carrier, valid
  non-AI carrier, burst train, tonic plateau, whole-sheet flash, runaway and a
  quiet-contact dB artifact.

**Command**

```bash
python -m pytest -q \
  tests/test_topic4_zm_carrier_gate_v2.py \
  tests/test_topic4_zm_d1_metrics.py
```

**Acceptance:** a positive requires all primary layers, not a firing-rate label.

## Task 8 — Lock and implement the perturbation-return test

**Create**

- `src/topic4_zm_d1_return.py`
- `scripts/lock_topic4_zm_d1_perturbation.py`
- `tests/test_topic4_zm_d1_return.py`

- [ ] Before opening candidate outcomes, calibrate on `A_native`
  `bounded_mid__rising` only.
- [ ] Use deterministic bisection for the smallest all-E 50 ms threshold uplift
  causing 50–70% paired core-spike reduction without ≥100 ms all-sheet rest.
- [ ] Freeze amplitude, all-E mask and exact fork-relative time 3.0 s.
- [ ] Implement return within 1 s, median period/frequency within 20%, same axial
  phase sign and phase-profile circular correlation ≥0.80.
- [ ] Keep `metastable_survival_without_return` distinct from attractor support.
- [ ] Prove the pulse does not touch z, m, phi state, currents, RNG or I thresholds.
- [ ] Label it a state-space robustness probe, never D5 control.

**Command**

```bash
python -m pytest -q tests/test_topic4_zm_d1_return.py
python scripts/lock_topic4_zm_d1_perturbation.py --check-only
```

## Task 9 — Build the fail-closed verdict and coverage model

**Create**

- `src/topic4_zm_d1_verdict.py`
- `tests/test_topic4_zm_d1_verdict.py`

- [ ] Encode the stop hierarchy at the top of this plan as a pure function.
- [ ] Require both bounded-mid phases before opening bounded-late.
- [ ] Require the registered seed-1 adjacent-checkpoint rule and both-phase
  perturbation returns.
- [ ] For replication require 5/6 per seed cell, ≥2/3 per fast phase, seeds 1
  and 3 support, seed 4 non-opposite, seeds 1/3 `dt/2`, and replicated return.
- [ ] A missing required artifact is `no_evidence`/blocked, never a scientific
  negative and never a default positive/negative enum.
- [ ] Ensure smoke, superseded, invalidated and technical-failure outputs cannot
  enter production coverage.
- [ ] Store all D2–D6 fields literally as not tested.

**Command**

```bash
python -m pytest -q tests/test_topic4_zm_d1_verdict.py
```

## Task 10 — Add adaptive parallel execution and resource receipts

**Create**

- `scripts/run_topic4_zm_d1_parallel.py`
- `tests/test_topic4_zm_d1_parallel.py`

**Reuse**

- `src/topic4_zm_phasec_resources.py`
- Phase-C crash-safe coordinator patterns

- [ ] Run one complete production-sized worker before parallel launch and record
  peak RSS, elapsed time, `VmSwap`, host memory and CPU count.
- [ ] Compute concurrency as the minimum of: 12 workers, available-memory
  budget using the existing 1.25× formula with 96 GB reserve, and logical CPUs
  minus 8.  Never hard-code 12 as safe.
- [ ] Set OMP/MKL/OpenBLAS/NumExpr threads to one in parent and workers.
- [ ] Poll process liveness and `VmSwap` at least every 5 s; stop launching new
  cells if reserve is breached, but do not kill peer-worktree processes.
- [ ] Resume only missing/technical-invalid cells.  Scientific failures remain
  terminal and cannot trigger retuning.
- [ ] Persist a coordinator receipt after every terminal cell so network loss is
  recoverable.
- [ ] Unit-test stale PID, partial file, mixed smoke/production, peer PID and
  memory-reserve calculations.

**Command**

```bash
python -m pytest -q tests/test_topic4_zm_d1_parallel.py
```

## Task 11 — Execute the cheap-first baseline gate

- [ ] Publish the real-data lock, perturbation lock and D1 manifest first.
- [ ] Run `A_native` and one production-sized phi row for resource calibration.
- [ ] Launch the remaining six paired seed-1 dynamic-baseline rows with adaptive
  concurrency.
- [ ] Analyze and publish `baseline_preservation_matrix.json`.
- [ ] If all six fail, publish `NO_GO_D1_baseline_not_preserved`, generate the
  baseline figure/README and stop all carrier execution.
- [ ] If one or more pass, freeze the eligible row list before looking at carrier
  outcomes.

**Production command shape**

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python scripts/run_topic4_zm_d1_parallel.py --phase baseline --confirm-run
```

## Task 12 — Execute seed-1 carrier and return gates

- [ ] Run every baseline-eligible row on bounded-mid rising and peak.
- [ ] Analyze each cell immediately, but do not change the panel.
- [ ] Only rows passing both mid phases proceed to bounded-late rising and peak.
- [ ] If no row survives, publish the correct D1 negative and stop.
- [ ] For survivors, run the fixed perturbation return on both phases and then
  candidate-only `S_G` clamp / no-`S_G` ablations.
- [ ] Freeze the seed-1 survivor list and verdict before replication.

```bash
python scripts/run_topic4_zm_d1_parallel.py --phase carrier-seed1 --confirm-run
python scripts/analyze_topic4_zm_d1.py --phase seed1
```

## Task 13 — Conditional replication and numerical confirmation

Run only if Task 12 has a seed-1 survivor.

- [ ] For seeds 1, 3, 4 run two fast phases × three locked future-noise banks.
- [ ] Do not recalibrate delta-phi by seed.
- [ ] Run independent `dt/2` confirmations for seeds 1 and 3.
- [ ] Replicate fixed perturbation return on seeds 1 and 3.
- [ ] Apply the exact replication rule in Task 9 and publish one final verdict.
- [ ] If the class changes at `dt/2`, emit `resolution_sensitive_fast_carrier`.

```bash
python scripts/run_topic4_zm_d1_parallel.py --phase replicate --confirm-run
python scripts/run_topic4_zm_d1_parallel.py --phase dt2 --confirm-run
python scripts/analyze_topic4_zm_d1.py --phase final
```

## Task 14 — Figures, archive, documentation and final acceptance

**Create**

- `scripts/plot_topic4_zm_d1.py`
- `tests/test_topic4_zm_d1_plot.py`
- `results/topic4_sef_hfo/zm_d1_fast_ictal_carrier/figures/README.md`
- `docs/archive/topic4/sef_hfo/zm_d1_fast_ictal_carrier_2026-08-01.md`

**Modify**

- `docs/topic4_sef_hfo.md`
- `docs/paper_overview.md`
- `docs/archive/topic4/INDEX.md`

- [ ] Plot baseline preservation + phi decay; source/E-I/vSEEG/spatial carrier;
  perturbation return; and complete coverage/verdict.
- [ ] Follow `docs/figure_style_guide.md` Topic-4 rules.  These are diagnostic
  carrier figures, not a Figure-5 lifecycle claim.
- [ ] Create the Chinese figure README after actual figures exist, with one
  `### filename` block and a final `**关注点**：` line per figure.
- [ ] Archive what was tested/how/verdict/claim boundary/ablation/resource and
  exact unresolved lifecycle legs.
- [ ] Keep Phase C, Phase D, diagnostic review and D1 evidence separate.
- [ ] Run visual QA on every PNG, not only file-existence tests.

**Final verification**

```bash
python -m pytest -q \
  tests/test_topic4_lifecycle_feasibility.py \
  tests/test_zm_dynamic_threshold.py \
  tests/test_topic4_zm_carrier_gate_v2.py \
  tests/test_topic4_zm_d1_contract.py \
  tests/test_topic4_zm_d1_observation.py \
  tests/test_topic4_zm_d1_runtime.py \
  tests/test_topic4_zm_d1_baseline.py \
  tests/test_topic4_zm_d1_runner.py \
  tests/test_topic4_zm_d1_metrics.py \
  tests/test_topic4_zm_d1_return.py \
  tests/test_topic4_zm_d1_verdict.py \
  tests/test_topic4_zm_d1_parallel.py \
  tests/test_topic4_zm_d1_plot.py

git status --short
ps -eo pid,ppid,rss,etime,args | rg 'topic4_zm_d1|run_topic4_zm' || true
```

## Required commit sequence

1. `fix(topic4): downgrade lifecycle scale screen to diagnostics`
2. `docs(topic4): lock D1 fast-carrier spec and execution plan`
3. `feat(topic4): lock D1 inputs and observation reference`
4. `feat(topic4): implement D1 baseline and frozen-carrier runners`
5. `feat(topic4): adjudicate D1 carrier and perturbation return`
6. `docs(topic4): archive D1 fast-carrier verdict`

Do not combine unreviewed scientific outputs with infrastructure commits.  Do
not push or merge unless explicitly authorized.  A clean worktree and green
tests establish engineering closure; the final JSON gate establishes the D1
scientific verdict.
