# Topic 4 state-conditioned spatial susceptibility — 8-hour autonomous implementation plan

> Status: EXECUTION PLAN, 2026-07-19.
> Design authority:
> `docs/superpowers/specs/2026-07-19-topic4-state-conditioned-spatial-susceptibility-design.md`.
> Worktree only:
> `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-mz-slowvars`.
> Expected starting branch/commit: `codex/topic4-mz-slowvars` / `66a4d93`.

## 0. Autonomy and wall-clock contract

The user authorizes up to approximately eight hours of bounded autonomous work on this task.
This authorization replaces the usual need to stop for user approval between the fast stage and
the task-scoped confirm stage. It does not authorize broader model sweeps or repository operations.

Time rules:

- record `START_TIME` in the run report;
- do not launch a new command expected to take more than 10 minutes after elapsed time reaches 7 h;
- reserve at least the final 45 minutes for figures, visual QA, STATUS, archive report, tests, and git audit;
- at 7 h 15 min, stop launching simulations regardless of remaining optional tasks;
- a running task with a reliable ETA under 30 minutes may finish; otherwise interrupt it safely and
  record `incomplete_runtime_budget`;
- never fill time with unregistered parameter exploration.

Priority order:

```text
P0 contract/tests/provenance
P1 primary candidate snapshots, atlas, required controls, primary figure/report
P2 second robust candidate sensitivity
P3 AR1/resolution/nonlinear spot checks
P4 optional comparison to dirty early-readout artifacts
```

## 1. Task 0 — enter and audit the dedicated worktree (0:00--0:20)

Run:

```bash
cd /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-mz-slowvars
git status --short
git branch --show-current
git log -1 --oneline
git worktree list --porcelain
```

Expected initial state:

- branch `codex/topic4-mz-slowvars`;
- HEAD `66a4d93` unless these plan docs were committed later;
- only the two 2026-07-19 spec/plan files may be new/modified;
- main checkout is dirty with unrelated Topic 5 work and must not be touched;
- `topic4-early-readout` is dirty and read-only.

If other unexpected files are dirty in this worktree, do not reset or overwrite them. Determine
whether they are user work. If overlap cannot be avoided, stop implementation and deliver an audit.

Read in this order:

```text
AGENTS.md
docs/topic0_methodology_audits.md
docs/topic4_sef_hfo.md
docs/topic4_m3_stage.md
docs/superpowers/specs/2026-07-19-topic4-state-conditioned-spatial-susceptibility-design.md
docs/superpowers/specs/2026-07-18-topic4-mz-per-neuron-slowvars-design.md
docs/archive/topic4/sef_hfo/mz_slowvars_discovery_2026-07-18.md
docs/superpowers/specs/2026-06-27-sef-hfo-m3b-spectral-phase-map-design.md
docs/superpowers/plans/2026-06-27-sef-hfo-m3b-spectral-phase-map-plan.md
docs/figure_style_guide.md  # Topic 4 section
```

Write `results/topic4_sef_hfo/state_conditioned_susceptibility/snapshot_contract.json` only after
the input paths and candidate rows have been verified.

## 2. Task 1 — baseline regression and cost audit (0:20--0:45)

Run the locked baseline tests:

```bash
pytest -q \
  tests/test_mz_slow_vars.py \
  tests/test_topic4_mz_slowvars.py \
  tests/test_topic4_m3b_spectral_phase.py
```

Record exact pass/fail count and wall time. Inspect, do not trust prose:

```text
results/topic4_sef_hfo/mz_slowvars/per_seed/multiseed_summary.json
results/topic4_sef_hfo/mz_slowvars/p3_candidates.json
results/topic4_sef_hfo/mz_slowvars/calibration.json
```

Confirm primary rows are `zA_q50_tz10000`, seeds 1/3/4, phenotype `runaway`, with finite onset.

Use existing `per_run.jsonl` wall times for scheduling. Do not rerun the whole discovery or RSS
audit unless the current environment differs materially. Set:

```bash
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
```

Maximum SNN worker count is 2. Spectral dense linear algebra runs serially unless a measured memory
audit proves safe.

Automatic gate:

- baseline tests green -> continue;
- baseline failure caused by this branch's current code -> diagnose/fix only if in scope;
- environment/dependency failure or unrelated failure -> record and stop before long runs.

## 3. Task 2 — implement an off-by-default slow-state snapshot observer (0:45--1:45)

Edit only:

```text
src/snn_engine/mz_slow_vars.py
tests/test_mz_slow_vars.py
```

Preferred design:

- add optional constructor argument `snapshot_steps=None`;
- store a normalized `{int_step: label}` mapping;
- keep an internal step counter;
- after the documented slow-state update, copy only `z[:NE]` and `m[:NE]` at registered steps;
- expose a small `snapshot_payload()` or read-only `snapshots` record;
- do not place large arrays into `MZSlowVarsConfig`;
- do not change the no-snapshot path's arithmetic or RNG calls.

Tests first:

1. no observer preserves exact output parity;
2. duplicate/invalid snapshot steps fail clearly;
3. requested steps capture once with exact shapes;
4. z/m bounds and E-only semantics hold;
5. memory shape is `n_snapshots x NE`, never `n_steps x NE`.

Run:

```bash
pytest -q tests/test_mz_slow_vars.py
```

Do not edit guarded engine files:

```text
src/snn_engine/kick_probe.py
src/snn_engine/params.py
src/snn_engine/model.py
src/snn_engine/connectivity.py
src/snn_engine/connectivity_rot.py
src/snn_engine/lfp.py
```

## 4. Task 3 — implement pure mapping, probe, and result-contract functions (1:45--3:00)

Create:

```text
src/topic4_state_conditioned_susceptibility.py
tests/test_topic4_state_conditioned_susceptibility.py
config/topic4_state_conditioned_susceptibility.yaml
```

The module must be import-safe and contain no simulations or file writes. Minimum pure functions:

```text
normalize_subject_coordinates(...)
bin_neuron_state_to_grid(...)
make_state_controls(...)
make_phase_paired_probe_dictionary(...)
embed_probe_in_rate_state(...)
batched_finite_time_response(...)
summarize_probe_atlas(...)
leading_probe_subspace_svd(...)
summarize_state_susceptibility(...)
```

Reuse from `src.topic4_m3b_spectral_phase`:

```text
Grid, Kernels, CoreMask, InhibitionField
build_kernels, make_core_mask, build_excitability_field
solve_operating_point, build_jacobian_dense
rate_eigenpairs, leading_subspace_indices, pair_loading
next_distinct_gap, globality, core_overlap
elongation_axis_score, off_axis_score
```

Do not duplicate the LIF transfer function, field RHS, Jacobian blocks, or eigensolver.

Required tests are the spec Gates C/D, including synthetic coordinate orientation, uniform-field
preservation, control invariants, cosine/sine phase invariance, batch equivalence, and unresolved
fail-closed behavior.

Run:

```bash
pytest -q \
  tests/test_topic4_state_conditioned_susceptibility.py \
  tests/test_topic4_m3b_spectral_phase.py
```

Cheap benchmark before scientific runs:

- one synthetic n=8 state;
- one n=12 baseline state;
- all registered probes at T=30 ms;
- print wall time and peak RSS;
- if n=12 single-state runtime exceeds 20 minutes or RSS is unsafe, keep n=8 as primary and record
  `resolution_limited`; do not silently reduce the grid.

## 5. Task 4 — build the gated runner and capture primary SNN snapshots (3:00--4:15)

Create:

```text
scripts/run_topic4_state_conditioned_susceptibility.py
```

Subcommands, all import-safe:

```text
audit-inputs
capture-snapshots
build-atlas
run-controls
run-nonlinear-spotchecks
all
```

Simulation-bearing subcommands require `--confirm-run`. `audit-inputs` does not.

The runner must read candidate configs/onsets from committed artifacts, not hard-code copied values.
It must support:

```text
--candidate zA_q50_tz10000
--seeds 1,3,4
--workers <=2
--max-wall-minutes
--resume
```

`--resume` may skip an existing stage only after schema, config hash, git/engine provenance, and
expected candidate/seed/state lists match. File existence alone is not a cache key.

Before full capture, run one short deterministic smoke on seed 1 and prove the snapshot observer does
not shift the trajectory. Then run the primary three seeds, maximum two workers:

```bash
python scripts/run_topic4_state_conditioned_susceptibility.py audit-inputs
python scripts/run_topic4_state_conditioned_susceptibility.py capture-snapshots \
  --candidate zA_q50_tz10000 \
  --seeds 1,3,4 \
  --workers 2 \
  --confirm-run
```

Automatic snapshot gate per seed:

- replay phenotype remains `runaway`;
- onset matches locked onset within 5 ms;
- five requested states captured or explicitly missing;
- z finite/in bounds and m exactly zero;
- coordinates/core/axis identical across states;
- all artifact provenance fields present.

If one seed fails, keep its failed artifact and continue other seeds. If all three fail for the same
reason, stop the atlas and write the diagnostic report.

## 6. Task 5 — build primary state-conditioned atlas and controls (4:15--5:45)

Run:

```bash
python scripts/run_topic4_state_conditioned_susceptibility.py build-atlas \
  --candidate zA_q50_tz10000 \
  --seeds 1,3,4 \
  --resume \
  --confirm-run

python scripts/run_topic4_state_conditioned_susceptibility.py run-controls \
  --candidate zA_q50_tz10000 \
  --seeds 1,3,4 \
  --resume \
  --confirm-run
```

The atlas must process states in the registered order and must write incremental per-state artifacts
atomically so interruption does not destroy completed work. Use a temporary file in the same output
directory followed by rename.

Required primary states/controls:

```text
states: baseline_1000ms, mid_fraction, pre_onset_500ms, pre_onset_100ms, onset
controls: real, uniform_mean, rotated_90, spatial_shuffle, z_blocked
T: 10, 30, 50, 75 ms
grid: n=12 primary; n=8 selected-state sensitivity
```

Scientific result handling:

- do not stop because the preferred direction is wrong, unchanged, or global;
- do not select a different candidate because seed 1 looks cleaner;
- unresolved state -> record `unresolved`, continue;
- all-seed baseline unresolved -> stop and report operator/calibration mismatch;
- uniform and real nearly identical -> report spatial pattern adds little beyond mean depletion;
- immediate rotation -> report it, but do not call it support for early scaffold reuse.

## 7. Task 6 — optional bounded extensions (5:45--6:30)

Only start if P0/P1 outputs are complete and at least 90 minutes remain.

Order:

1. AR1 baseline/onset selected-state control;
2. n=8 versus n=12 resolution audit;
3. nonlinear rate-field two-amplitude spot check;
4. second candidate `zA_q75_tz5000`, seeds 1/3/4;
5. read-only comparison to early-readout artifacts.

Do not start a full SNN mode-injection grid overnight. At most, implement an import-safe adapter and
one predeclared seed/state/probe smoke if all higher-priority tasks are complete.

For the nonlinear rate-field check, amplitudes must be fixed before seeing the output. Record failure
of the 10% scaling criterion; do not retune until it passes.

## 8. Task 7 — plot, visual QA, and README (6:30--7:15)

Create:

```text
scripts/plot_topic4_state_conditioned_susceptibility.py
```

The plotter reads artifacts only and never reruns a simulation. Produce the primary diagnostic PNG/PDF
defined in the spec. If controls are complete, produce the companion controls PNG.

After rendering:

1. open every PNG at original/high detail;
2. verify source/sink positions and axis direction against snapshot metadata;
3. verify identical coordinates and color ranges across state columns;
4. verify `eigenmode`, `probe`, and `response` labels are not mixed;
5. verify unresolved/not-run panels are visible;
6. verify all three seed trajectories are visible and not overwritten by the median;
7. inspect for clipping, unreadable colorbars, repeated titles, and mirrored geometry;
8. only then write `figures/README.md` in Chinese using the repository's required format.

Do not spend the final report window polishing aesthetics beyond legibility.

## 9. Task 8 — final scientific report and reproducibility audit (7:15--8:00)

Write:

```text
results/topic4_sef_hfo/state_conditioned_susceptibility/STATUS.md
docs/archive/topic4/sef_hfo/state_conditioned_spatial_susceptibility_2026-07-19.md
```

The report must answer, with artifact paths and denominators:

1. Were the locked MZ trajectories replayed without observer-induced drift?
2. Did `z` depletion become spatially patterned along the scaffold or mostly uniform?
3. How did axial, perpendicular, and global finite-time gains change within each seed?
4. Did preferred orientation remain stable, rotate, or become unresolved?
5. Did preferred spatial scale move toward lower `k` before onset?
6. Did real `z(x)` differ from uniform-mean, rotate, shuffle, and z-blocked controls?
7. Were changes consistent across seeds?
8. Did the true leading eigen-subspace and the non-normal optimal response tell different stories?
9. Which optional stages were `completed`, `failed`, or `not_run`?
10. What is the safest scientific claim, the largest remaining gap, and the single next experiment?

Mandatory wording boundary:

- use `runoff` or `operational runaway`, never seizure;
- call `z_i` a phenomenological inhibitory-efficacy variable;
- call Fourier/Gabor objects probes;
- distinguish true eigenmode from finite-time optimal response;
- distinguish engineering completion, numerical eligibility, scientific observation, and data bridge.

Final verification:

```bash
pytest -q \
  tests/test_mz_slow_vars.py \
  tests/test_topic4_mz_slowvars.py \
  tests/test_topic4_m3b_spectral_phase.py \
  tests/test_topic4_state_conditioned_susceptibility.py

git diff --check
git status --short
git diff --stat
git worktree list --porcelain
```

If targeted tests are green and time permits, run the full test suite only if its historical runtime
fits the remaining budget. A full-suite timeout does not invalidate targeted evidence; report it.

## 10. Local commit policy

Local commits are allowed to preserve overnight work; push, merge, rebase, and PR creation are not.
Use logical batches only after their tests pass:

1. `feat(topic4): add bounded MZ slow-state snapshot capture`
2. `feat(topic4): add state-conditioned spatial susceptibility analysis`
3. `docs(topic4): report state-conditioned susceptibility diagnostic`

Before each commit, inspect the exact staged file list. Do not stage unrelated files or any file from
another worktree. Results under the exact new output root may be force-added if repository ignore
rules require it and the artifacts are reasonably sized. Never force-add broad `results/` globs.

If a stage is incomplete or tests fail, leave changes uncommitted and report them rather than making a
misleading green commit.

## 11. Final handoff format

Return a concise Chinese handoff containing:

```text
一句话结论
完成到哪一层：engineering / numerical / scientific / bridge
primary observation across three seeds
required controls and their outcomes
tests and exact counts
new code/artifact/figure/report paths
local commits, if any
not-run/failed items and why
largest scientific gap
recommended next single action
final git/worktree status
```

Do not end with only “ran successfully.” The handoff must say what the model does and does not support.
