# FCXR-LC3 — coupled D-X lifecycle geometry and spatial instability

Status: **DESIGN LOCK**

Date: 2026-08-03

Upstream acceptance: `docs/archive/topic4/sef_hfo/fcxr_lc2_gx1_review_acceptance_2026-08-03.md`

## 1. One scientific question

Does the unchanged RC1 + local-H + Z + X system already contain an autonomous path in which repeated
interictal events increase inhibitory depletion `D=1-Z`, a local or axial instability creates a bounded
high state, sustained load lowers relay availability `a_X`, and the joint recovery of `D` and `a_X`
returns the network to its original interictal statistical neighbourhood?

LC3 does **not** require low/high bistability at the same frozen `D`. A monostable branch exchange,
Hopf/SNIC-like transition, finite-amplitude basin transition, or noise-assisted crossing may all be
valid. The required object is a reachable onset surface, a reachable offset surface, and a return path.

The core acceptance trajectory remains:

```text
sparse irregular returning IEDs
  -> no-kick onset on the patient-specific pathological axis
  -> bounded high state
  -> autonomous X/D-mediated offset
  -> postictal protection
  -> return to the pre-onset IED statistical neighbourhood
```

## 2. Frozen model and scope

### 2.1 Model assets retained

- RC1 recurrent E-to-E conductance with smooth saturation;
- the existing anisotropic E-to-E pathological axis and two low-threshold cores;
- the existing local H equation and additive H actuator;
- the validated Z and presynaptic X implementations;
- the current E/I population, delays, connectivity, noise and current-based virtual-SEEG;
- the accepted workpoint/high-state numerical classifiers.

### 2.2 H anchors

No new H search is allowed.

- Primary: GX1 point `H1_ts1.25_r025`, because healthy low remains interictal while susceptible low and
  high enter finite high activity.
- Negative spatial/control family: `H6_ts1.25_r025`, because its healthy recurrent input lies near the H
  threshold and healthy low self-ignites.

All numerical values must be read from the GX1 lock/manifest by point ID and hashed into the LC3 lock;
they must not be copied into a second implementation constant.

### 2.3 Locked out

- explicit `S_D(D)` or `theta_H(D)`;
- changed H gain, H sensor, shared X/H path or new E-to-E edges;
- M, K, A, ELR, recruited-area feedback or morphology tuning;
- kick/reset/parameter step in lifecycle acceptance;
- 3--8 Hz, 1--80 Hz or patient morphology as a Phase-4 gate;
- 40k paper-ready lifecycle figures;
- use of confirmation seeds during parameter selection.

If the current H fails, a local E/I-balance H sensor is the primary *next hypothesis*. It is not
implemented in this sprint. Explicit D gating remains a mechanistic control, not the default repair.

## 3. Phase L0 — evidence and axis lock

### 3.1 GX1 acceptance lock

The LC3 execution lock must contain the GX1 verdict hash and these labels:

```text
GX1_MECHANISM_MAP_ACCEPTED
FINITE_H_HIGH_STATE_POSITIVE
D_SELECTIVE_ONSET_CANDIDATE
SAME_D_BISTABILITY_NOT_FOUND
X_OFFSET_PATH_REACHABLE
X_FIXED_D_DYNAMIC_RANGE_INSUFFICIENT
COUPLED_D_X_OFFSET_UNTESTED
DYNAMIC_LIFECYCLE_UNTESTED
SPATIAL_INSTABILITY_UNTESTED
```

It must also hash the GX1 strip, X map, LC2 frozen-fork map, baseline contract, H implementation module,
runner modules and the six blessed engine files.

### 3.2 D-axis rule

Use the already archived no-kick 24 s Z-only traces, not arbitrary uniform `D` values. The input set is:

- connection seed 1, q75 Z-only;
- connection seed 1, q50 Z-only;
- connection seed 3, q75 Z-only.

After a fixed 2 s burn-in, pool their finite spatial-mean `D_Z` samples. Define:

```text
D_healthy = 0
D10, D30, D50, D70 = pooled empirical quantiles
Dmax = maximum of the three per-run q99 values
```

Record each source path, sha256, sample count, time step, raw maximum and the six selected values in
`dx_axis_lock.json`. Require `0=D_healthy < D10 < D30 < D50 < D70 < Dmax`; otherwise block and report
`D_AXIS_UNRESOLVED`. The rule is fixed before inspecting LC3 outcomes.

Relay availability is fixed as:

```text
a_X = {1.0, 0.9, 0.8, 0.65, 0.5, 0.3, 0.1}
```

### 3.3 Seed split

- Development: connection seed 1, noise 401; boundary replication may use noise 405/406.
- Confirmation: connection seed 3 and noise 402/403/404 remain unseen until a candidate path and
  spatial-mode interpretation are locked.

The archived seed3 q75 trace contributes only to the pre-existing D-range definition; no LC3 response
from connection seed 3 is inspected during development.

## 4. Exact prepared-state contract

Each frozen `(D,a_X)` cell is run from two real fast-system microstates:

1. `low_prepared`: an accepted returning-interictal state after at least 8 s at
   `(D_healthy,a_X=1)`;
2. `high_prepared`: a converged finite-high state after at least `max(5 s,8 tau_H)` at
   `(D50,a_X=1)` under the same H family. Analytic `H=2 theta` may start the preparation, but the forked
   state is the complete converged endpoint, not that analytic initialization.

A prepared state includes membrane voltages, refractory clocks, synaptic/delay buffers, recurrent and
inhibitory currents, H, RNG state and every other variable that can affect the continuation. Before the
map, prove exact continuation parity: uninterrupted and checkpoint/fork continuations must have the same
raster hash, rate trace, H trace and final state under identical controls.

An in-memory copy-on-write fork is acceptable. A slow-variable-only snapshot is not. If the complete
state cannot be exposed without changing a blessed engine file, stop with `EXACT_FORK_BLOCKED`; do not
substitute analytic H initialization.

## 5. Phase L1 — frozen D-aX state plane

### 5.1 Registered map

Run the complete matrix:

```text
2 H families x 6 D levels x 7 a_X levels x 2 prepared states = 168 forks
```

All cells first run 1.5 s and use the final 500 ms as the screen tail. A scientific negative never stops
the matrix. Cells adjacent to a label
change, or with unresolved convergence, are extended to 5 s and repeated with development noise
405/406; the extended decision tail is the final 2 s. Both use the accepted 300 ms rolling workpoint
definition. The first single fork remeasures RSS and wall time.

### 5.2 Readouts

For every fork record:

- population E/I, core A/B, axial-band and off-axis rates;
- low/high/interictal classification and fixed/orbit/irregular morphology;
- H support, recurrent E/I currents and active area;
- hard clip, numerical guard, refractory-ceiling fraction and minimum effective membrane time constant;
- tail slope, tail occupancy, local perturbational gain and convergence uncertainty;
- whether low- and high-prepared starts converge to the same statistical state.

The primary empirical surfaces are:

- low-start entry surface `Sigma_on`: transition of `P_enter(D,a_X)`;
- high-start survival/return surface `Sigma_off`: transition of `P_remain(D,a_X)`.

A `P=0.5` contour is reported only where boundary cells were replicated. It is an empirical transition
surface, not automatically a bifurcation or eigenvalue crossing. Same-D bistability is a descriptive
map feature, never an entry gate.

### 5.3 State-plane verdicts

```text
DX_CLOSED_REACHABLE_PATH_CANDIDATE
DX_MONOSTABLE_ONSET_OFFSET_SURFACES_CANDIDATE
DX_OFFSET_SURFACE_PRESENT_X_CALIBRATION_REQUIRED
DX_NO_CLOSED_PATH_CURRENT_H
DX_MAP_UNRESOLVED
```

A reachable path requires that the archived dynamic D range can cross the low-state entry boundary and
that a joint decrease in `a_X` plus recovery in `D` can cross the high-state return boundary without
entering numerical or refractory saturation. It does not require a same-D hysteresis loop.

## 6. Phase L2 — early spatial instability audit

This phase begins after L1 identifies landmarks; it does not wait for a full lifecycle.

### 6.1 Landmark states

At minimum audit:

1. healthy low;
2. low state immediately before the empirical onset surface;
3. post-onset high;
4. high near the empirical offset surface;
5. returned low after the surface.

### 6.2 Primary causal response assay

Coarse-grain E and I activity onto a fixed `16 x 16` spatial grid. Apply matched, zero-mean or
energy-matched small current perturbations in the following pre-registered spatial patterns:

- axial forward and axial reverse;
- transverse;
- core A, core B and dual-core;
- global/isotropic;
- spatially shuffled axial control;
- one deterministic random matched control.

All patterns are normalized to the same RMS current over E cells and use
`spatial_control_seed=731` for the shuffled/random controls. Non-global patterns have their
E-cell weighted mean removed and receive a uniform compensation so net injected current is zero; the
global/isotropic positive control is RMS-matched but is not mean removed. Pattern construction is frozen
before outcomes.

Let `I_ref` be the median absolute recurrent-E current across E cells in the healthy prepared state;
non-finite or non-positive `I_ref` blocks the probe.
Use RMS amplitudes `{0.01,0.02,0.04}*I_ref`, a 10 ms pulse and common random numbers for `+epsilon`,
`-epsilon` and sham. Select the largest of the three amplitudes
that passes a pre-outcome linearity check: antisymmetry error and response/amplitude variation each
<=20%, with no state transition in the healthy reference. Report finite-time gain, first-passage map,
axis angle and axial/transverse/global gain ratios at 50, 150, 300 and 500 ms after pulse onset. These
direct paired responses are the primary spatial evidence.

### 6.3 Conditional effective operator

Bin the 16x16 E/I response at 10 ms. Fit ridge operators over normalized penalties
`{1e-4,1e-3,1e-2,1e-1,1}` with nested held-out perturbation trials. Report an operator only when median
held-out R2 is at least 0.30, the median absolute cosine of the leading right mode across bootstraps is
at least 0.80, and at least 80% of bootstraps agree on its global/axial/transverse category. Report
discrete-time eigenvalues, correctly transformed continuous-time
rates, leading left/right modes, participation ratio, global overlap, pathological-axis angle and
finite-time singular gain.

If operator identification is unstable, label `OPERATOR_UNRESOLVED` and retain only the direct
finite-time response result. Do not call a DMD mode a full 40k Jacobian eigenmode.

### 6.4 Spatial verdicts

```text
AXIAL_OR_LOCAL_INSTABILITY_CANDIDATE
GLOBAL_COMMON_MODE_FAILURE
TRANSVERSE_MODE_FAILURE
NONNORMAL_AXIAL_GAIN_CANDIDATE
SPATIAL_RESPONSE_UNRESOLVED
```

An axial/local candidate requires the maximum axial/core direct gain to exceed each global, transverse,
shuffled and random-matched control by at least 20% at two consecutive registered response times, with
the first-passage support overlapping the pathological axis or one registered core. Near offset, the
same response must decrease by at least 20% as X load increases. The desired direction is
`D up -> axial/local gain up` and `X load up -> the same response restabilizes`.
A global common mode reaching the boundary first blocks dynamic lifecycle acceptance under the current H
architecture, even if a temporal high/low trace can be produced.

## 7. Phase L3 — X calibration from the measured surface

L3 is unlocked only if L1 finds an offset surface. Define the required availability `a_off(D)` from that
surface. The target is:

```text
a_X after 1--3 s high activity < a_off[D(t)]
a_X during ordinary returning IEDs > 0.9
```

Before any grid, decide which two knobs are identifiable:

- if the X asymptote cannot reach the surface: sensor gain and Hill midpoint;
- if the asymptote is reachable but too slow: sensor gain and rise time.

Run at most a `3 x 3` calibration. Do not simultaneously scan threshold, exponent, rise, decay and
minimum. Derive the shortest postictal X decay satisfying
`T_X_protect > T_D_recover_to_safe` from the state-plane map; do not compare arbitrary 5 s/10 s values.

## 8. Phase L4 — no-kick dynamic pilot

L4 is unlocked only when:

1. L1 finds a closed reachable or calibratable D-X path;
2. L2 does not return a global/transverse spatial failure;
3. L3, when required, locks no more than two X candidates.

Keep `M=K=A=ELR=0`. Use primary `H1_ts1.25_r025` and the only allowed robustness neighbour
`H1_ts1.25_r050`, at most two X candidates and development noises `{401,405,406}`: no more than 12
nominal trajectories. Each trajectory is exactly 32 s and must contain, without kick/reset/parameter
step:

- at least 8 s sparse irregular returning IEDs;
- a spontaneous D-associated onset;
- a bounded non-clipped high state;
- X accumulation after onset;
- autonomous offset;
- postictal suppression while D recovers;
- at least 8 s return to the pre-onset IED statistical neighbourhood.

Recovery compares event rate, IEI distribution, duration, participation, compactness and forward/reverse
axis statistics; it is not equality to a fixed point or a fixed rhythm.

Only lifecycle candidates unlock matched `X-off`, `Z-frozen` and `H-off` ablations. Confirmation seeds
remain untouched until one nominal candidate and its causal interpretation are frozen.

## 9. Stop and branch rules

- `DX_NO_CLOSED_PATH_CURRENT_H`: stop. Draft a separate local E/I-balance H-sensor design.
- `GLOBAL_COMMON_MODE_FAILURE`: stop current H even if its temporal map closes.
- axial finite-time gain without reliable eigenvalues: continue under a non-normal interpretation, with
  no eigenvalue claim.
- surface reachable but dynamic onset absent: recalibrate the Z slow trajectory only; do not change H.
- onset but no offset: revise X/load architecture only.
- offset followed by rebound: calibrate X protection against measured D recovery.
- temporal lifecycle positive but spatial audit negative: report temporal component positive, core
  scientific lifecycle negative.
- only after temporal and spatial gates pass may a later sprint add per-cell M for morphology.

## 10. Resource and provenance contract

- T <20 s: at most two 40k workers; T >=20 s: exactly one worker.
- Before worker 2 require
  `MemAvailable >= 96 GiB + 2*1.35*RSS_single` and stable swap.
- Use a bounded-submission scheduler. Swap +256 MiB stops new submission; +512 MiB and rising terminates
  only LC3's newest worker. Never submit all futures upfront.
- Set OMP/OpenBLAS/MKL/NUMEXPR threads to one.
- Long stages use `setsid nohup`, exact launcher PID/SID, stage-scoped flock, wall guard,
  `RUNNING/DONE/FAILED` stage sentinels and per-cell DONE files. Wait by PID, never `pgrep -f`.
- Never signal sibling processes or edit sibling worktrees.
- Any source/hash drift after lock blocks new simulation; post-run analysis repairs must preserve both
  locked and current hashes.
- Figures are generated only after data exist and each figure directory gets a Chinese README after
  visual inspection.

## 11. Required outputs

```text
results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/
  execution_lock.json
  dx_axis_lock.json
  prepared_state_contract.json
  state_plane_manifest.json
  state_plane_map.json
  spatial_probe_lock.json
  spatial_response_map.json
  effective_operator.json                  # conditional
  x_calibration.json                       # conditional
  lifecycle_manifest.json                  # conditional
  lifecycle_verdict.json                   # conditional
  STATUS.md
  resource_log.jsonl
  figures/state_plane.png
  figures/spatial_instability.png
  figures/lifecycle_candidate.png           # only if candidate exists
  figures/README.md
```

The sprint may claim a temporal lifecycle candidate only after L4. It may claim the core scientific
lifecycle only when the temporal path and the axial/local spatial-transition gate both pass.
