# FCXR-LC3 — field-preserving D-X lifecycle geometry and spatial response

Status: **LC3 REVISION 1.1 — EXECUTION AUTHORIZED**

Date: 2026-08-03

Upstream acceptance: `docs/archive/topic4/sef_hfo/fcxr_lc2_gx1_review_acceptance_2026-08-03.md`

## 1. Core scientific question

Does the unchanged RC1 + local-H + Z + X system contain an autonomous route in which repeated
interictal events build a spatial inhibitory-depletion field `D_i(x)=1-Z_i`, the two-dimensional
E/I network enters a bounded high state without kick, sustained load lowers presynaptic relay
availability `a_{X,j}`, and the joint recovery of D and X returns the network to the original
interictal statistical neighbourhood?

LC3 does not require low/high bistability at the same frozen D. A monostable branch exchange,
Hopf/SNIC-like transition, finite-amplitude basin crossing, non-normal amplification or noise-assisted
transition may all be valid. The required evidence is split into three independent axes:

1. **frozen geometry** proposes entry, survival and return boundaries;
2. **dynamic trajectories** determine whether the real slow flow traverses them;
3. **spatial responses** explain whether the patient-specific pathological axis participates.

No scientific result on one axis prevents measurement of the other two. Only exact-state failure,
numerical corruption or manifest/hash failure is a hard execution stop.

The target remains:

```text
sparse irregular returning IEDs
  -> no-kick local onset
  -> bounded high state
  -> autonomous offset
  -> postictal protection
  -> return to the pre-onset IED statistical neighbourhood
```

Even complete LC3 success establishes a lifecycle core, not full patient-like ictal morphology.
Per-cell M, 3--8 Hz envelope, 1--80 Hz broadband and E1146 observation matching remain a later phase.

## 2. Frozen assets and locked-out mechanisms

Retain without redesign:

- RC1 recurrent E-to-E conductance and smooth saturation;
- the existing anisotropic E-to-E pathological axis and two low-threshold cores;
- the current local H state, sensor and additive H actuator;
- the validated Z and presynaptic X implementations;
- the accepted E/I population, delays, connectivity, noise and current-based virtual-SEEG;
- the accepted workpoint, event and numerical classifiers.

H roles are fixed:

- nominal: GX1 `H1_ts1.25_r025`;
- sparse negative-control sentinel: `H6_ts1.25_r025`;
- candidate-only robustness: `H1_ts1.25_r050`, never used to select a lifecycle.

All values are read from and hashed against GX1 artifacts by point ID. No copied parameter constant may
become a second source of truth.

Locked out:

- explicit `S_D(D)`, `theta_H(D)` or a changed H sensor;
- shared X/H path, new E-to-E edges, K/A/ELR or recruited-area feedback;
- M or ictal morphology tuning;
- kick, reset or external parameter step in lifecycle acceptance;
- selecting from connection seed 3 or noise 402/403/404;
- paper-ready lifecycle figures.

If the current H architecture fails, a local E/I-balance H sensor is the next hypothesis in a separate
spec. It is not implemented here.

## 3. Evidence model and verdict axes

LC3 does not collapse all evidence into one top-level gate.

### 3.1 Temporal geometry

```text
DX_MAP_COMPLETE
DX_GEOMETRIC_PATH_PRESENT
DX_GEOMETRIC_PATH_ABSENT
DX_DYNAMIC_VECTOR_MISALIGNED
DX_MAP_UNRESOLVED
```

`DX_GEOMETRIC_PATH_PRESENT` means only that frozen entry/return brackets can be connected in
projection. It must never be called dynamically reachable before a no-kick trajectory traverses it.

### 3.2 Spatial mechanism

```text
AXIAL_LOCAL_RESPONSE
GLOBAL_DOMINANT_RESPONSE
TRANSVERSE_DOMINANT_RESPONSE
NONNORMAL_AXIAL_RESPONSE
SPATIAL_RESPONSE_UNRESOLVED
```

### 3.3 Dynamic lifecycle

```text
NO_SPONTANEOUS_ONSET
ONSET_ONLY
ONSET_BOUNDED_NO_OFFSET
OFFSET_REBOUND
TEMPORAL_LIFECYCLE_CANDIDATE
SPATIOTEMPORAL_LIFECYCLE_CANDIDATE
```

Temporal lifecycle positivity with a global spatial response is retained as
`TEMPORAL_LIFECYCLE_POSITIVE_SPATIAL_MECHANISM_NEGATIVE`, not erased. Axial spatial susceptibility
without temporal closure is retained as `SPATIAL_SUSCEPTIBILITY_POSITIVE_TEMPORAL_CLOSURE_NEGATIVE`.

## 4. E0 — provenance and exact-state contract

The execution lock must hash:

- GX1 candidate verdict, strip and X map;
- LC2 frozen-fork map and accepted baseline contract;
- the three LC1 Z-only scalar traces and relevant dynamic Z/X traces;
- H/Z/X mechanism module and LC3 runners;
- six blessed engine files.

Every frozen experiment starts from a complete prepared microstate. The state contract includes membrane
voltages, refractory clocks, spike/synaptic/delay buffers, currents, H/Z/X arrays, RNG state and every
mutable variable that affects continuation.

Before any scientific 40k row:

1. uninterrupted and forked continuations under identical controls have identical raster, trace and
   final-state hashes;
2. two child forks do not alias mutable arrays;
3. replacing only `D_i` or `a_{X,j}` changes only that frozen field at fork time;
4. off paths preserve byte parity;
5. all fields are finite, correctly shaped and in physical bounds.

An in-memory copy-on-write fork is allowed. A slow-variable-only snapshot is forbidden. Failure writes
`EXACT_FORK_BLOCKED.json` and is a hard stop.

## 5. E1 — field-preserving frozen D-X geometry

### 5.1 Full D fields, not scalar D

The archived Z-only files contain scalar D traces but not full fields. Use them only to select target
times, then deterministically replay the same no-kick runs with the existing sparse snapshot observer.

Primary field family:

- connection seed 1, q75 Z-only, after 2 s burn-in;
- `D_healthy` is the exact all-zero depletion control;
- targets `D10,D30,D50,D70,Dmax` are q10/q30/q50/q70/q99 of the archived scalar trace;
- for each target, snapshot the complete `D_i` field at the replay time whose spatial mean is nearest
  that target.

Replication field families:

- connection seed 1, q50 Z-only;
- connection seed 3, q75 Z-only.

For each replication trace, capture the complete field whose mean is nearest each primary target mean.
These fields test spatial-field robustness; they do not redefine the primary D axis.

Every field record includes source path and sha256, replay configuration and hash, target/actual time,
mean, q5/q50/q95, core A/B means, axial/off-axis means, spatial norm and field checksum. Deterministic
replay must reproduce the archived scalar trace at the selected times within a locked numeric tolerance;
otherwise report `D_FIELD_REPLAY_UNRESOLVED`.

A uniform field `D_i=mean(D_i)` is allowed only as a matched control at boundary landmarks. It is
never the primary D condition.

### 5.2 Uniform X coordinate and actual X fields

The coarse causal coordinate remains uniform:

```text
a_X = {1.0, 0.9, 0.8, 0.65, 0.5, 0.3, 0.1}
```

This measures the frozen mean relay surface. It does not represent the full dynamic X field. E3/E4 must
capture actual `a_{X,j}` fields. At relevant return boundaries compare, with the same D field:

1. the actual dynamic X field;
2. a deterministic permutation using `spatial_control_seed=731`;
3. a uniform field with the same mean.

No statement about natural X reachability is allowed from the uniform map alone.

### 5.3 Map budget

Nominal H1 full map:

```text
6 primary D fields x 7 uniform a_X levels x 2 prepared states = 84 rows
```

H6 sentinel map:

```text
D = {D_healthy,D50,Dmax}
a_X = {1.0,0.5,0.1}
2 prepared states
= 18 rows
```

H6 is a healthy-self-ignition/global-mode bad-data reference, not half of the primary state plane.

### 5.4 Canonical prepared states

The coarse map uses one exact state of each type:

- `low_canonical`: accepted returning-interictal state after at least 8 s at
  `(D_healthy,a_X=1)`;
- `high_canonical`: complete converged high state after at least `max(5 s,8 tau_H)` at
  `(D50,a_X=1)`. Analytic `H=2 theta` may begin preparation but is never the forked state.

All rows first run 1.5 s with a final-500 ms screen tail. Label changes and unresolved convergence get
a 5 s extension with a final-2 s tail.

### 5.5 Boundary probability and microstates

One state/noise defines a bracket, not a probability. Boundary cells are repeated with:

- low states: inter-event trough, pre-IED and post-IED;
- high states: tail-rate peak and trough; if high activity is fixed-like, use two tail states separated
  by at least 1 s and label them phase surrogates;
- noises `{401,405,406}`.

Low-state selection uses a locked canonical returning event after 8 s and its preceding gap. High-state
selection uses the final 2 s and fixed extrema/tie-breaking rules. Exact definitions must be tested before
opening map outcomes.

Only boundary cells with at least 3 microstates x 3 noises for the relevant basin may receive a
probability-like estimate or P=0.5 contour. Otherwise report an empirical bracket.

### 5.6 Geometry outputs

Report separately:

- low-start entry bracket `Sigma_entry`;
- high-start survival bracket `Sigma_survival`;
- high-to-low return bracket `Sigma_return`;
- sensitivity to actual, shuffled and uniform D/X fields;
- saturation and numerical diagnostics.

Same-D bistability is descriptive only and never a lifecycle prerequisite.

## 6. E2/E3 — slow vector field and dynamic reconnaissance

### 6.1 Short slow-flow probes

At 12--20 deterministically selected H1 landmarks, unfreeze Z and X for 300 ms while retaining the
prepared fast/H state. Include both sides of each observed entry/return bracket. If no bracket exists,
use the fixed 12-point grid:

```text
D = {D_healthy,D50,Dmax}
a_X = {1.0,0.65,0.3,0.1}
```

Estimate over 50--300 ms:

- mean `dot(D)` and `dot(a_X)`;
- core A/B, axial and off-axis field drifts;
- alignment with normals of any geometric brackets;
- whether the vector points toward or away from entry/return regions.

The vector field is a local drift diagnostic, not proof of a closed orbit.

### 6.2 Three no-kick reconnaissance trajectories always run

Regardless of map, vector-field or spatial labels, run three nominal trajectories:

- H = `H1_ts1.25_r025`;
- the archived q75 Z calibration and current unretuned LC1 X configuration, read by provenance;
- connection seed 1, noises `{401,405,406}`;
- M=K=A=ELR=0; no kick/reset/parameter step.

Time contract:

- run at least 32 s;
- onset search checkpoint at 20 s;
- if onset occurs by 32 s, continue at least 12 s after onset;
- after offset continue until 8 s of recovery observation;
- absolute cap 45 s;
- if no onset by 32 s, stop at 32 s.

Capture full `D_i` and `a_{X,j}` fields at pre-onset, onset, early high, late-high/pre-offset,
post-offset and recovered states when those states exist. Record actual trajectory in mean D-X
projection and spatial field coordinates.

These runs are reconnaissance, not parameter acceptance. A negative is still a completed result.

## 7. E4/E5 — spatial direct response before operator fitting

Use real dynamic states from reconnaissance whenever available. If no onset occurs, use the nearest
frozen-map landmarks and label that substitution.

### 7.1 Positive recruitment probes

Primary positive patterns:

- core A pulse and core B pulse (forward/reverse polarity);
- axial elongated patch;
- transverse elongated patch;
- global positive control;
- shuffled axial positive control.

Core/axial/transverse/shuffled local masks use the same active-cell count and match duration, positive
charge and RMS exactly. Core masks are subsampled deterministically when necessary. Because global
activation cannot match positive charge, RMS and cell count simultaneously, run and label two global
controls separately: charge-matched and RMS-matched. Never compare them through one shared threshold.

Use two pre-locked safe positive amplitudes derived from the healthy `I_ref` and a 10 ms pulse.
Report active-cell count, total positive charge and RMS for every arm. Primary readouts are first-passage
field, newly recruited area, core polarity, axial/off-axis expansion and finite-time gain.

### 7.2 Signed projected-response probes

Signed `+epsilon/-epsilon/sham` probes are a separate experiment with common random numbers. Use a
predeclared 8--12-dimensional physical basis containing global, core A, core B, axial symmetric,
axial antisymmetric, transverse, surround and deterministic random controls.

For response times 50/150/300/500 ms estimate:

```text
R_ij(T) = <basis_i, delta r produced by basis_j at T>
```

Use SVD to report dominant input/output direction, axial/global overlap and non-normal finite-time gain.
This projected matrix is the first operator-level object. Do not fit a 512-dimensional 16x16 E/I DMD in
the primary spatial stage.

Spatial results change interpretation, not authorization to run dynamic reconnaissance or lifecycle
exploration.

## 8. E6 — X calibration from measured boundaries

Run only when a high-state return/offset bracket exists. Define `a_off(D)` from the measured return
surface and compare it with the actual dynamic D-X trajectory and X fields.

Targets:

```text
ordinary returning IED: mean a_X > 0.9
high activity after 1--3 s: trajectory crosses a_off[D(t)]
postictal X protection lasts longer than D recovery to the safe region
```

Choose exactly one two-knob family before a maximum 3x3 calibration:

- asymptote cannot reach boundary: sensor gain + Hill midpoint;
- asymptote reaches but is too slow: sensor gain + rise time.

Do not simultaneously scan threshold, exponent, rise, decay and minimum.

## 9. E7 — dynamic lifecycle exploration

Use nominal `H1_ts1.25_r025` only, at most two X candidates and noises `{401,405,406}`:

```text
1 H x <=2 X x 3 noises = <=6 nominal trajectories
```

Use the same adaptive 32--45 s observation contract as reconnaissance. A candidate requires:

- at least 8 s sparse irregular returning IEDs before onset;
- no-kick onset and bounded non-clipped high activity;
- X change after onset and autonomous offset;
- postictal protection;
- at least 8 s recovery observation;
- return of event rate, IEI distribution, duration, participation, compactness and forward/reverse
  statistics to the pre-onset neighbourhood.

Return of mean rate alone is not recovery.

Only a frozen nominal candidate unlocks:

- `H1_ts1.25_r050` robustness;
- Z-frozen, X-off and H-off causal ablations;
- confirmation connection/noise seeds.

The robustness point never participates in nominal selection.

## 10. E8/E9 — candidate-only causality and formal operator

Causal expectations:

- Z-frozen: no equivalent spontaneous onset;
- X-off: high state prolonged or no autonomous offset;
- H-off: no equivalent bounded high state.

Formal spatial operator/eigenmode work is authorized only when at least one holds:

1. a real dynamic onset exists;
2. direct spatial response is reproducibly axis/global discriminative;
3. eigenvalue softening versus non-normal amplification must be separated.

First formalize the low-dimensional projected operator. A 16x16 E/I ridge/DMD operator is optional and
requires held-out prediction and bootstrap stability. Failure yields `OPERATOR_UNRESOLVED` without
altering temporal verdicts.

The final spatial test is whether the pre-onset response predicts the dynamic trajectory's starting
core, forward/reverse polarity and early first-passage field, not mode angle alone.

## 11. Hard stops, scientific routing and claim boundary

Hard stops only:

1. exact fork/state parity failure;
2. NaN, clip, numerical guard or resource-safety failure;
3. manifest, artifact or source-hash integrity failure.

Scientific negatives never stop E3 dynamic reconnaissance. In particular:

- geometric path absent -> still run reconnaissance;
- global/transverse response -> still run dynamics, but blocks a spatially valid claim;
- operator unresolved -> retain direct responses;
- spatial response unresolved -> retain temporal evidence;
- no onset -> report where the real D-X trajectory remained.

Only a temporal candidate plus axial/local spatial evidence may receive
`SPATIOTEMPORAL_LIFECYCLE_CANDIDATE`. Temporal positivity alone remains valuable but cannot answer the
full core scientific goal.

## 12. Resource, detachment and provenance

- T <20 s: at most two 40k workers; T >=20 s: exactly one worker.
- Before worker 2 require
  `MemAvailable >= 96 GiB + 2*1.35*RSS_single` and stable swap.
- Use bounded submission with at most `n_workers` pending jobs. Swap +256 MiB stops new submission;
  +512 MiB and rising terminates only LC3's newest worker.
- Pin OMP/OpenBLAS/MKL/NUMEXPR to one thread.
- Long stages use `setsid nohup`, exact launcher PID/SID, stage flock, wall guard, atomic
  RUNNING/DONE/FAILED stage sentinels and per-row DONE files. Wait by PID, never `pgrep -f`.
- Never signal sibling processes or modify sibling worktrees.
- Source/hash drift after lock blocks new simulations; analysis corrections preserve locked/current
  hashes and never rewrite raw cells.
- Figures are produced only from complete available evidence. Every generated figure directory gets a
  Chinese README after visual inspection; no placeholder figures.

## 13. Required outputs

```text
results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/
  execution_lock.json
  d_field_lock.json
  prepared_state_contract.json
  geometry_manifest.json
  geometry_map.json
  boundary_replication.json
  slow_vector_field.json
  reconnaissance_manifest.json
  reconnaissance_verdict.json
  spatial_probe_lock.json
  spatial_direct_response.json
  projected_response_operator.json
  x_field_sensitivity.json
  x_calibration.json                         # conditional
  lifecycle_manifest.json                    # conditional
  lifecycle_verdict.json                     # conditional
  causal_ablation.json                       # candidate-only
  formal_operator.json                       # conditional
  verdict_axes.json
  STATUS.md
  resource_log.jsonl
  figures/geometry_map.png
  figures/dynamic_trajectory.png
  figures/spatial_direct_response.png
  figures/lifecycle_candidate.png             # only if candidate exists
  figures/README.md
```

LC3 stops for review after the registered evidence program. It does not add morphology or produce the
final paper figure.
