# Topic 4 state-conditioned spatial susceptibility — Design Spec

> Status: DESIGN LOCK for an autonomous overnight implementation, 2026-07-19.
> Execution base: `codex/topic4-mz-slowvars` at commit `66a4d93`, worktree
> `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-mz-slowvars`.
> Implementation plan:
> `docs/superpowers/plans/2026-07-19-topic4-state-conditioned-spatial-susceptibility-implementation.md`.
> This is an exploratory model-side mechanism/readout analysis. It is not a seizure validation.

## 0. Plain-language objective

The current Topic 4 model has three ingredients that have not yet been joined in one analysis:

1. the fixed anisotropic SNN scaffold supports a self-limited axial non-normal transient;
2. the M4-MZ `z_i` variable can move the same E1146 SNN from repeated interictal-like events to
   reproducible runoff;
3. an early-recruitment readout can place an interictal latency field and an onset-locked energy
   field on the same virtual-SEEG plane, but its current core-excluded association is not positive.

This task asks one narrower question:

> As inhibitory efficacy `z_i` evolves along one fixed, continuous, pre-existing MZ trajectory,
> does the finite-time spatial susceptibility of the fixed scaffold change in a way that preserves
> and strengthens the interictal propagation axis before runoff?

The expected object is a state-conditioned response map, not a new slow-variable parameter sweep.
No result direction is a success gate. A null, uniform-gain, off-axis, global, unresolved, or
seed-inconsistent result must be reported as observed.

## 1. Evidence and inputs that are already locked

### 1.1 MZ trajectory source

Canonical branch-local inputs:

```text
results/topic4_sef_hfo/mz_slowvars/calibration.json
results/topic4_sef_hfo/mz_slowvars/per_seed/multiseed_summary.json
results/topic4_sef_hfo/mz_slowvars/p3_candidates.json
docs/archive/topic4/sef_hfo/mz_slowvars_discovery_2026-07-18.md
```

Primary trajectory family:

```text
zA_q50_tz10000
seeds = 1, 3, 4
phenotype = runoff/runaway in 3/3 seeds
locked onset neighborhood ~= 4.7--4.9 s
```

Sensitivity trajectory family, only after the primary family is complete:

```text
zA_q75_tz5000
seeds = 1, 3, 4
phenotype = runoff/runaway in 3/3 seeds
locked onset neighborhood ~= 9.3--9.8 s
```

The seed-1 `zA_q75_tz10000` expanded-bounded run is not a primary trajectory because it reproduced
in only 1/3 seeds. It may be displayed as a labelled case-series sensitivity only if time remains.

### 1.2 Existing spectral machinery

Reuse, do not replace:

```text
src/topic4_m3b_spectral_phase.py
tests/test_topic4_m3b_spectral_phase.py
scripts/build_m3b_spectral_outputs.py
docs/superpowers/specs/2026-06-27-sef-hfo-m3b-spectral-phase-map-design.md
```

The accepted M3B result is layered:

- homogeneous Fourier dispersion is a background sanity check;
- the finite heterogeneous Jacobian is the true local linear operator;
- the leading eigenmode is predominantly global;
- the axial, self-limited signal lives in the non-normal finite-time response;
- no slow-state trajectory overlay has yet been validated.

This task extends that machinery. It must not relabel the leading eigenmode as axial merely because
the finite-time response is axial.

### 1.3 Optional early-readout reference

The dirty worktree
`/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-early-readout` is read-only for this task.
Its existing artifacts may be inspected for terminology and final comparison, especially:

```text
results/topic4_sef_hfo/early_recruitment_readout/STATUS.md
results/topic4_sef_hfo/early_recruitment_readout/early_recruitment_readout.json
results/topic4_sef_hfo/early_recruitment_readout/figures/
```

Do not edit that worktree, import its uncommitted source files, copy its scientific verdict into the
new verdict, or make it a required dependency. If the artifact is unavailable, record `not_run` and
continue.

## 2. Hard boundaries and non-goals

This overnight task must not:

- run a new broad `z`, `m`, `z+m`, STD, shunting, `g_K`, or `S_G` parameter sweep;
- start a 40 s acceptance run;
- optimize parameters against a field-correlation score;
- modify the six guarded SNN engine files;
- modify the main checkout, the early-readout worktree, paper Methods, or Topic 5 files;
- call runoff/runaway a seizure or ictal event;
- claim a complete seizure cycle, termination mechanism, or patient mechanism;
- treat Fourier/Gabor probes as eigenmodes of a heterogeneous finite operator;
- choose a preferred seed, state, time, `k`, direction, or control after seeing which looks best;
- hide unresolved states or wrong-sign/null results.

The task may add a snapshot observer to `src/snn_engine/mz_slow_vars.py` because that module is not
one of the six guarded engine files. The observer must be off by default and must preserve exact
simulation parity when unused.

## 3. Scientific objects and terminology

Use the following terms consistently.

### 3.1 Slow state

For a snapshot time `t_s`, the microscopic slow state is:

```text
s_slow(t_s) = {z_E(t_s), m_E(t_s), positions_E, core masks, scaffold axis}
```

For the primary `z`-only trajectory, `m_E == 0` is required and audited.

### 3.2 Coarse state field

Microscopic E-neuron values are binned onto one fixed coarse grid:

```text
z_bar(x, t_s) = mean z_i in grid cell x
m_bar(x, t_s) = mean m_i in grid cell x
```

The field is a sign-calibrated, normalized slow-to-rate mapping. It is not an exact microscopic
Jacobian of the 32,000-neuron threshold/reset network.

Mapping into the M3B rate field:

```text
q(x, t_s) = clip(z_bar(x, t_s), q_floor, 1)
gK_equiv(x, t_s) = eta_m * m_bar(x, t_s)   # only for m-enabled sensitivities
```

`q` scales I-to-E efficacy. `gK_equiv` is an additive E-only adaptation-current field passed through
the existing `gK_field` input with `eta_K=1`; it must not be renamed biological potassium current.

### 3.3 True finite-system eigenmode

At a frozen slow state, solve the corresponding rate-field operating point and form:

```text
J_s = dF/dx evaluated at x_star(s_slow)
J_s v_j = lambda_j v_j
```

`v_j` is a true eigenmode of the coarse finite heterogeneous operator. Report the leading invariant
subspace when the leading mode is a complex pair or near-degenerate group.

### 3.4 Fourier/Gabor probe

A probe is an input, not an eigenmode:

```text
b_k,theta,phi(x) = window(x-x0) * cos(k dot x + phi)
```

Use phase-paired cosine/sine probes. The global Fourier dictionary is a background/system-ID view;
the source-centered Gabor dictionary is the primary perturbation family because interictal events
are local, not whole-sheet plane waves.

### 3.5 Finite-time susceptibility

For readout `C` selecting the E-rate field and input embedding `B` into the E-rate row:

```text
R_s(T) = C exp(J_s T) B
G_s(k, theta, T) = ||R_s(T) b_k,theta||_2 / ||b_k,theta||_2
```

Primary windows are `T = 10, 30, 50 ms`; `75 ms` is a sensitivity. The primary visual map uses
30 ms because the existing non-normal axial response peaks near that scale.

The optimal response in the registered probe subspace is obtained by SVD of the batched response
matrix. Call its right vector the optimal probe combination and its left vector the optimal output
field. Do not call either an eigenmode.

## 4. Trajectory and snapshot contract

### 4.1 No parameter rediscovery

Read candidate configurations and per-seed `runaway_ms` from the committed artifacts. Do not refit
them. Replay the same candidate, seed, substrate, noise seed, and duration.

### 4.2 Snapshot labels

For every eligible seed, capture exactly these predeclared states:

```text
baseline_1000ms       = 1000 ms
mid_fraction          = 0.50 * locked runaway_ms
pre_onset_500ms       = runaway_ms - 500 ms
pre_onset_100ms       = runaway_ms - 100 ms
onset                 = runaway_ms
```

If a time is outside the simulated interval, save it as missing with a reason. Do not silently move
the time to a prettier event or quiet interval. The `onset` snapshot uses the locked prior-run onset,
not a retrospectively optimized spatial transition time.

Slow variables change little over the first 30--50 ms of runoff. Therefore the frozen `onset` state
is the primary state for predicting the early response. This round does not require an `onset+50ms`
microscopic snapshot after the early-stop boundary.

### 4.3 Observer requirements

The snapshot observer must:

- be `None`/off by default;
- copy only `z_E` and `m_E` at requested steps, not store `N x T` arrays;
- record whether capture occurs before or after the slow update at that step;
- use an integer step contract with `round(t_ms / dt)`;
- store every requested/actual time and any missing reason;
- leave I-cell values out of the saved payload or prove they remain `z=1`, `m=0`;
- preserve exact off-path parity in a deterministic fixture and one short `simulate_kick` run.

### 4.4 Snapshot artifact schema

Write one NPZ plus one JSON sidecar per candidate and seed:

```text
snapshots/<candidate>/seed_<seed>.npz
snapshots/<candidate>/seed_<seed>.json
```

NPZ minimum arrays:

```text
snapshot_labels
requested_time_ms
actual_time_ms
z_E                    # [state, NE]
m_E                    # [state, NE]
pos_E                   # [NE, 2]
core_mask_E             # [NE]
vth_E                   # [NE]
src_xy, snk_xy, axis_unit
```

JSON minimum fields:

```text
schema_version
candidate, seed, phenotype, locked_runaway_ms, replay_runaway_ms
dt_ms, L, subject, montage
config, source_artifact_paths
snapshot_update_convention
guarded_engine_sha256, git_sha, argv
missing_snapshots
```

The replay phenotype must remain `runaway`, and `abs(replay_runaway_ms-locked_runaway_ms) <= 5 ms`.
Otherwise mark the replay mismatch and do not silently use the shifted trajectory.

## 5. Coarse-grid and geometry contract

Primary diagnostic grid:

```text
n = 12
L_normalized = 5.0
coordinate_space = normalized_m3b
```

Map the E1146 sheet affinely into the centered normalized square while preserving:

- the subject long-axis direction;
- the relative source/sink positions and separation;
- the core membership mask;
- the same transform for every seed and state.

Use the established M3B Gaussian rate-field kernel scale (`ell_perp=0.6`, `AR=2`) as a coarse
surrogate. Record explicitly that its `k` units are normalized M3B units, not physical SNN
millimeters. A selected-state `n=8` versus `n=12` resolution sensitivity is required; `n=16` is
optional if runtime permits.

Every cell must record occupancy. Empty bins are invalid for direct averaging; use a documented
nearest valid-bin fill only if occupancy is below 100%, and emit the fill mask. With 32,000 E cells
on a 12x12 grid, widespread empty bins indicate a coordinate bug and are a stop condition.

The M3B axis `theta` must come from the transformed `axis_unit`; do not use the module default
45-degree constant when the E1146 registered sheet is horizontal.

## 6. Probe dictionary and readout metrics

### 6.1 Registered dictionary

Freeze before running scientific outputs:

```text
integer wave indices p,q in [-4,4]
exclude p=q=0 from directional peak selection
include p=q=0 separately as the global/uniform probe
phases = cosine, sine
centers = source core for primary Gabor; sink core as registered sensitivity
Gabor sigma = 1.0 normalized M3B unit
T = [10, 30, 50, 75] ms
```

Do not increase the `k` grid because the first result looks unresolved or move the Gabor center to
the strongest output location.

### 6.2 Per-state outputs

For each state and control, store:

- operating-point status, residual, mean/max E/I rates;
- leading eigenvalues, eigen residuals, next-distinct spectral gap;
- leading-subspace E loading, globality, core overlap, axis/off-axis scores;
- full phase-paired probe gain atlas for every `T`;
- registered axial gain, perpendicular gain, low-k/global gain;
- peak `k`, peak orientation modulo 180 degrees, peak gain;
- gain persistence `G(50)/G(30)` and `G(75)/G(30)`;
- probe-subspace leading singular value and optimal output field;
- state-field summaries: mean/min `z`, axis-corridor mean, off-axis mean, axis-minus-off-axis gap.

The primary across-state estimand is the within-seed change from `baseline_1000ms` to `onset`.
Aggregate seeds only after computing within-seed trajectories. Report all three seeds and their
median; never pool grid cells across seeds as independent samples.

## 7. Required controls

Run the following at each primary state unless explicitly marked selected-state only:

1. **real state field**: observed `z_bar(x,t)`;
2. **uniform-mean**: replace `z_bar` by its spatial mean;
3. **rotated-90**: rotate the slow field 90 degrees around sheet center while the scaffold stays fixed;
4. **spatial shuffle**: permute coarse-cell `z` with a fixed seed, preserving its histogram;
5. **z blocked**: set `z=1` everywhere;
6. **AR1 isotropic scaffold**: selected baseline/onset states only;
7. **resolution**: `n=8` versus `n=12` on primary seed baseline/onset only.

The controls answer different questions and must not be collapsed:

- uniform-mean separates global disinhibition from spatially patterned depletion;
- rotate/shuffle tests whether the spatial arrangement of depletion matters;
- z-blocked anchors the pre-depletion operator;
- AR1 tests whether direction comes from the anisotropic scaffold;
- resolution checks whether the apparent preferred mode is a coarse-grid artifact.

## 8. Engineering and numerical validity gates

These gates protect validity; they do not select a desired scientific result.

### Gate A — baseline regression

Before edits, run:

```bash
pytest -q \
  tests/test_mz_slow_vars.py \
  tests/test_topic4_mz_slowvars.py \
  tests/test_topic4_m3b_spectral_phase.py
```

If baseline tests fail, diagnose and report; do not start long simulations.

### Gate B — snapshot observer

Required tests:

- off-by-default exact parity;
- requested integer steps captured once;
- no `N x T` allocation;
- saved `z` finite and in `[0,1]`, `m>=0`;
- primary z-only snapshots have `m==0`;
- replay onset and phenotype match the locked artifact.

### Gate C — mapping

Required tests:

- known synthetic coordinates land in the expected `indexing="ij"` cells;
- the transformed axis has the expected orientation;
- coarse field preserves uniform inputs exactly;
- histogram/rotation/shuffle controls preserve their declared invariants;
- source and sink core locations map to the correct sides of the sheet.

### Gate D — operator/probes

Required tests:

- homogeneous/no-core spectrum agrees with sampled Fourier blocks on a tiny grid;
- dense-J JVP agrees with finite difference on a tiny grid;
- eigen residuals are `<1e-6` for reported modes;
- cosine/sine phase pairing is invariant to phase rotation;
- AR1 synthetic response has no fixed E1146-axis preference;
- batched probe response agrees with one-at-a-time response;
- no unresolved/saturated state is silently assigned a stable/axial label.

If a state operating point is unresolved, record it as `unresolved` and continue other states. If
baseline is unresolved for all seeds, stop the scientific atlas and deliver the diagnostic report.

### Gate E — nonlinear spot check

For selected primary-seed baseline/onset states, run a small rate-field perturbation at two amplitudes.
The response should scale within 10% over the registered early window to call it a linear-regime
validation. Failure is a reported nonlinear-boundary result, not permission to tune amplitude until
it passes. A full SNN re-probe is optional and lower priority than completing the controls.

## 9. Diagnostic figure contract

This is explicitly a Topic 4 diagnostic figure, so the standard single-row four-panel paper layout
does not apply yet. Produce one primary multi-row figure:

```text
columns: baseline | mid | pre-onset | onset
row 1: coarse z(x) field with source, sink, scaffold axis, and core outlines
row 2: Gabor/Fourier susceptibility in kx-ky at T=30 ms
row 3: registered strongest input probe -> finite-time E output field
row 4: within-seed/median trajectories of axial, perpendicular, global gain and persistence
```

Visual requirements:

- fixed coordinates, limits, color normalization, and axis direction across state columns;
- no mirroring to make source identity look consistent;
- separate labels for `eigenmode`, `probe`, and `finite-time response`;
- mark unresolved panels visibly rather than leaving them blank;
- show all three seeds in faint lines and the median prominently;
- no p-value stars or PASS/FAIL labels;
- output PNG and PDF;
- write `figures/README.md` in Chinese after rendering and visual inspection.

Optional companion figure: real/uniform/rotate/shuffle/AR1 controls at baseline and onset.

## 10. Output contract

All new outputs stay in:

```text
results/topic4_sef_hfo/state_conditioned_susceptibility/
├── snapshot_contract.json
├── snapshots/<candidate>/seed_<seed>.{npz,json}
├── coarse_fields/<candidate>/seed_<seed>.npz
├── susceptibility_atlas.json
├── susceptibility_arrays.npz
├── control_summary.json
├── numerical_audit.json
├── nonlinear_spotcheck_summary.json
├── STATUS.md
└── figures/
    ├── README.md
    ├── state_conditioned_susceptibility_diagnostic.png
    ├── state_conditioned_susceptibility_diagnostic.pdf
    └── state_conditioned_susceptibility_controls.png   # optional
```

Code/config/tests:

```text
config/topic4_state_conditioned_susceptibility.yaml
src/topic4_state_conditioned_susceptibility.py
scripts/run_topic4_state_conditioned_susceptibility.py
scripts/plot_topic4_state_conditioned_susceptibility.py
tests/test_topic4_state_conditioned_susceptibility.py
```

Archive report:

```text
docs/archive/topic4/sef_hfo/state_conditioned_spatial_susceptibility_2026-07-19.md
```

Every JSON must include schema version, exact upstream paths, git SHA, engine hashes, config hash,
candidate/seed/state lists, and whether an optional stage was `not_run`, `failed`, or `completed`.

## 11. Scientific interpretation contract

### Allowed wording if supported by artifacts

- repeated activity was accompanied by a state-dependent change in the coarse spatial susceptibility;
- the preferred early response retained, strengthened, weakened, broadened, or rotated relative to
  the fixed scaffold axis;
- spatially patterned depletion added information beyond its uniform mean, or did not;
- the change was or was not reproducible across the three locked seeds;
- the existing non-normal axial transient was modulated along a dynamic MZ trajectory;
- the result is a model-side mechanism candidate and diagnostic bridge.

### Forbidden wording

- the model reproduced a seizure;
- `z_i` proves chloride accumulation, GABA failure, or interneuron exhaustion;
- a Fourier/Gabor probe is the network eigenmode;
- a leading global eigenmode is axial because its transient response is axial;
- a single positive seed establishes the bridge;
- runoff termination or a full seizure cycle has been explained;
- a model/contact-space association proves the patient mechanism;
- an unresolved or not-run control passed.

### Result-neutral reporting

The final report must explicitly choose among observed descriptions, not among success labels:

```text
same-axis gain increase
same-axis gain decrease
orientation rotation
global/uniform amplification
spatial pattern adds no effect beyond mean z
seed-inconsistent
numerically unresolved
```

Multiple descriptions may coexist at different states. Do not force them into one verdict.

## 12. Completion definition

Minimum complete overnight deliverable:

1. snapshot observer and mapping tests pass;
2. primary `zA_q50_tz10000` seeds 1/3/4 replay and snapshot artifacts exist;
3. all five primary state fields are mapped or explicitly missing;
4. the n=12 real-field atlas and required real/uniform/rotate/shuffle/z-blocked controls exist;
5. eigen/probe numerical audits are written;
6. the primary diagnostic figure and Chinese README exist and have been visually inspected;
7. STATUS and archive report state the safe claim, largest gap, and exact not-run items;
8. no unrelated worktree files changed, no push/merge occurred, and final `git status` is reported.

If time remains, add the second robust trajectory family, AR1/resolution controls, and nonlinear
spot checks in that order. Do not sacrifice the primary report and provenance to start another long run.
