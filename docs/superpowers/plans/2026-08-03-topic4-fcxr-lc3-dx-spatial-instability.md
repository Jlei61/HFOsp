# FCXR-LC3 D-X state plane and spatial instability — IMPLEMENTATION PLAN

Status: **IMPLEMENTATION PLAN — LOCKED, NOT STARTED**

Date: 2026-08-03

Design: `docs/superpowers/specs/2026-08-03-topic4-fcxr-lc3-dx-spatial-instability-design.md`

## 0. Execution graph

```text
T0 GX1 acceptance + artifact/hash preflight
 -> T1 D-axis derivation + complete registered manifests
 -> T2 exact prepared-state/fork contract
 -> T3 classifiers, spatial probes and bad-data tests
 -> T4 one-row resource smoke
 -> T5 complete 168-cell D-aX map
 -> T6 boundary extensions + empirical surfaces
 -> T7 spatial response and conditional operator audit
 -> Gate DX/SPATIAL
 -> T8 conditional X calibration
 -> T9 conditional no-kick lifecycle pilot
 -> T10 candidate-only causal ablations and confirmation
 -> figures/archive/STATUS
 -> STOP for review
```

T0--T7 are the main authorized diagnostic. T8--T10 are conditional and must not be launched merely
because a rate trace looks promising.

## 1. T0 — close GX1 and lock evidence

1. Require a clean committed GX1 acceptance with canonical label `GX1_MECHANISM_MAP_ACCEPTED`.
2. Verify the nine accepted mechanism labels and the corrected fixed-D X semantics.
3. Resolve and hash the GX1 strip/X map, LC2 frozen-fork map, LC1 Z-only traces, baseline contract,
   H mechanism module, runners and six blessed engine files.
4. Fail loudly on missing cross-worktree artifacts or hash mismatch; never silently select a similarly
   named file.
5. Write `execution_lock.json` before code capable of a new 40k run is invoked.

Tests must prove that same-D bistability is not a lifecycle prerequisite and that GX2/D-gate execution
remains unauthorized.

## 2. T1 — derive axes and write the whole matrix first

Implement a zero-simulation command that reads only the three locked 24 s Z-only traces and applies the
spec's burn-in/quantile rule. Unit-test pooling, quantiles, q99 maximum, ordering, NaN handling and sha256
provenance. Write `dx_axis_lock.json`.

Then write all 168 rows of `state_plane_manifest.json` before running any cell. Each row contains:

- H anchor ID and exact parameters read from GX1;
- D label/value and a_X value;
- prepared-state ID;
- connection/noise seed;
- 1.5 s screen duration/final-500 ms tail and possible 5 s/final-2 s extension status;
- output path, expected sentinel and resource tier.

Assert uniqueness and exact cardinality in tests. Do not inspect outcomes while constructing the grid.

## 3. T2 — exact prepared-state infrastructure

Prefer an in-memory fork in a new non-blessed LC3 runner. If serialization is used, enumerate and test
every mutable fast/slow/RNG field. Required tests:

1. baseline preparation lasts at least 8 s at `(D_healthy,a_X=1)` and passes the accepted interictal
   contract;
2. high preparation lasts at least `max(5 s,8 tau_H)` at `(D50,a_X=1)`, reaches a finite converged high
   branch and is not merely `H=2 theta`;
3. uninterrupted vs restored continuation has identical raster sha1, rate/H traces and final-state hash;
4. two forks from one state remain independent after mutation;
5. changing only D/a_X changes only the registered frozen controls at fork time;
6. no blessed engine edit and all off-by-default paths preserve byte parity.

Failure writes `EXACT_FORK_BLOCKED.json` and ends the sprint before T4.

## 4. T3 — adjudicators and bad-data regression

Build pure functions before simulations for:

- low/high/rest-like and fixed/orbit/irregular classification;
- tail convergence and uncertainty;
- empirical transition surfaces with boundary replication;
- direct paired spatial-response metrics;
- conditional operator validation and label withholding;
- lifecycle and multivariate recovery classification.

Required synthetic bad data include: returning IEDs, pulse comb, tonic refractory plateau, burst-silence,
global flash, transverse-first response, axial non-normal amplification without eigenvalue crossing,
unstable DMD modes and a fake rate recovery with wrong IED statistics. The archived HEO1 global common
mode must fail the spatial target if its compatible traces are available.

## 5. T4 — single-row smoke and OOM decision

Run one full 1.5 s primary-H/healthy-D/a_X=1/low-prepared row. Require finite output, no clip, valid
classifiers, exact sentinels and measured RSS/wall time.

Only enable worker 2 if:

```text
MemAvailable >= 96 GiB + 2*1.35*RSS_single
swap is stable
no sibling pressure violates the reserve
```

Use a producer that submits at most `n_workers` pending rows. At swap delta +256 MiB, stop producing new
rows and let active rows finish. At +512 MiB and rising, terminate only the newest LC3 worker, preserve
its FAILED sentinel, and continue no further submission. Threads are pinned to one.

## 6. T5 — complete frozen map

Launch with `setsid nohup`, exact PID/SID, stage flock and wall guard. Run breadth-first across H family,
D, a_X and initial state so an interruption does not leave only one side of the map. A scientific
negative does not stop the 168 rows.

Each row writes raw trace summary, classifier inputs, numerical diagnostics, resource record and a
per-row DONE sentinel atomically. Resume skips only rows with valid DONE plus matching manifest/hash.
Never trust a stale RUNNING file as completion.

## 7. T6 — boundary replication and state-plane verdict

Freeze the boundary-selection code before opening the aggregate. Extend only:

- adjacent cells with different low-start entry labels;
- adjacent cells with different high-start survival labels;
- unresolved convergence cells;
- the nearest cells on a possible closed path.

Run 5 s extensions with development noise 401 and replicate required boundary cells with 405/406. Fit a
P=0.5 empirical contour only where replication exists; otherwise report a bracket. Emit separate entry,
survival and return maps before a joint path verdict.

Do not stop because same-D low/high starts converge. Do stop before dynamics if the current H has no
closed or calibratable path.

## 8. T7 — spatial instability audit

### 8.1 Lock landmarks and amplitudes

Select landmarks from the frozen map by deterministic rules, not visual appeal. Build exactly nine
spatial patterns from geometry already locked upstream, with `spatial_control_seed=731`. Test the locked
`{0.01,0.02,0.04}*I_ref` RMS ladder with 10 ms pulses in a healthy reference; freeze the largest linear
amplitude before comparing landmark outcomes. Assert zero net current for all non-global patterns and
equal RMS energy across all patterns.

### 8.2 Run paired probes

For each landmark/pattern/amplitude, run `+epsilon`, `-epsilon` and sham with common random numbers.
Coarse-grain to the fixed 16x16 E/I grid. Store direct finite-time gain and first-passage outputs at
50/150/300/500 ms before any operator fitting.

### 8.3 Operator is conditional

Fit regularized projected operators over the locked ridge grid using nested held-out trials. Bootstrap
across perturbation/noise repetitions and enforce the spec's R2/cosine/category gates. If held-out
prediction or leading-mode stability fails, emit
`OPERATOR_UNRESOLVED`; do not search another basis after seeing the result.

The spatial gate uses direct causal responses. A stable operator can refine the interpretation but may
not rescue a global/transverse direct-response failure.

## 9. Gate after T7

Continue only under either:

```text
DX_CLOSED_REACHABLE_PATH_CANDIDATE
or DX_OFFSET_SURFACE_PRESENT_X_CALIBRATION_REQUIRED
```

and either:

```text
AXIAL_OR_LOCAL_INSTABILITY_CANDIDATE
or NONNORMAL_AXIAL_GAIN_CANDIDATE
```

`GLOBAL_COMMON_MODE_FAILURE`, `TRANSVERSE_MODE_FAILURE`, `DX_NO_CLOSED_PATH_CURRENT_H`, numerical
corruption or unresolved exact forks stop the sprint and route to a separate E/I-balance H-sensor spec.

## 10. T8 — conditional X calibration

Use the measured `a_off(D)` surface and archived IED/high-state load traces. Before a grid, write an
identifiability report choosing exactly one allowed two-knob pair. Run at most 3x3 points.

Select no more than two candidates that keep ordinary IED availability above 0.9 and reach the offset
surface after 1--3 s high activity. Compute the minimum postictal decay needed to protect D recovery.
No 40k dynamic trajectory is run during selection unless the required load summary cannot be obtained
from an exact prepared-state fork; such a deviation must be registered before execution.

## 11. T9 — conditional no-kick lifecycle pilot

Write all nominal rows first: `H1_ts1.25_r025`, optional locked neighbour `H1_ts1.25_r050`, up to two X
candidates and noises `{401,405,406}`, maximum 12 rows. Run exactly 32 s, strictly one 40k worker.

The aggregate reports separately:

- pre-onset returning-IED statistics;
- D/onset order;
- high-state boundedness and saturation diagnostics;
- onset-to-X order;
- X/D offset crossing;
- postictal protection;
- returning-IED multivariate distance.

A transient high state, a reset-dependent recovery, or return of mean rate alone cannot pass.

## 12. T10 — candidate-only causal tests

Only a frozen nominal candidate unlocks matched:

- `Z-frozen`: no spontaneous onset expected;
- `X-off`: high state significantly prolonged or no autonomous offset expected;
- `H-off`: no equivalent bounded high state expected.

After these pass, freeze parameters and run confirmation connection/noise seeds without reselection.
Failure of confirmation is reported; it does not reopen tuning on those seeds.

## 13. Outputs, figures and archive

Use the result root and filenames in the spec. Generate `state_plane.png` and
`spatial_instability.png` only from complete aggregate inputs. Generate `lifecycle_candidate.png` only
when a candidate exists; no placeholder. After visual inspection, write `figures/README.md` in Chinese,
2--4 sentences plus `关注点` per figure.

The archive must lead with:

1. completed stage;
2. temporal and spatial verdicts separately;
3. safe scientific claim and forbidden claim;
4. exact state/fork status;
5. map counts and seed coverage;
6. tests, hashes, resources, sentinels and residual processes;
7. one next recommendation only.

Stop for review even if a complete lifecycle candidate is found. Do not add M morphology or make the
paper-ready figure in this plan.
