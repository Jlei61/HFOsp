# FCXR-LC3 field-preserving lifecycle — IMPLEMENTATION PLAN

Status: **LC3 REVISION 1.1 — EXECUTION AUTHORIZED**

Date: 2026-08-03

Design: `docs/superpowers/specs/2026-08-03-topic4-fcxr-lc3-dx-spatial-instability-design.md`

## 0. Execution graph

```text
E0 exact fork + artifact/hash/numerical contracts
 -> E1 full D-field replay and 102-row geometry map
 -> E2 boundary microstate/noise/field replication
 -> E3 short Z/X slow-vector probes
 -> E4 three no-kick dynamic reconnaissance runs
      |                              |
      +-> E5 direct spatial response+
      +-> E6 X calibration when return boundary exists
                         |
                         v
 E7 <=6-run dynamic lifecycle exploration
 -> E8 candidate-only ablations/robustness/confirmation
 -> E9 conditional projected/full operator formalization
 -> figures/archive/STATUS
 -> STOP for review
```

Geometry, dynamic trajectories and spatial response are parallel evidence axes. E4 always runs after E0
engineering safety, even when geometry or spatial evidence is negative. Scientific labels route
interpretation; only exact-state, numerical/resource and manifest/hash failures stop execution.

## 1. E0 — lock evidence and prove exact continuation

### 1.1 Preflight and execution lock

Resolve paths from the GX1/LC2/LC1 artifacts named in the spec. Hash each exact file, the H/Z/X module,
all new runners and the six blessed engine files. Verify the accepted GX1 labels and that GX2/D-gate
execution remains unauthorized.

Write `execution_lock.json` before any new 40k simulation. Cross-worktree inputs are read-only.
Missing, ambiguous or hash-drifted input fails loudly.

### 1.2 Exact state inventory

Inventory every mutable object in the 40k simulation: neuron voltage/refractory state, spike and delay
buffers, recurrent/inhibitory currents, slow H/Z/X arrays, counters, RNG and adapter state. Implement an
in-memory fork or complete checkpoint in a new LC3 runner without changing blessed engines.

Tests:

1. uninterrupted and forked continuations are byte-identical under matched controls;
2. child forks do not alias arrays;
3. D-only and X-only interventions change only the registered field at fork time;
4. off paths are byte-identical to upstream;
5. invalid shapes/ranges/NaNs fail before stepping;
6. checkpoint metadata binds engine/module/source hashes.

Write `prepared_state_contract.json`. Do not proceed to E1 scientific rows until it passes.

### 1.3 Pure adjudicators first

Before outcomes, test:

- low/high/rest-like and fixed/orbit/irregular classification;
- canonical low/high microstate selectors;
- boundary adjacency and empirical brackets;
- three-axis verdict aggregation without science-induced early stop;
- adaptive dynamic observation window;
- direct positive and signed spatial probe accounting;
- resource scheduler soft/hard swap behavior.

Bad data include returning IED, tonic refractory plateau, burst-silence, global flash, transverse
response, fake mean-rate recovery and unstable operator modes.

## 2. E1 — acquire complete D fields and run primary geometry

### 2.1 Two-pass field replay

Use archived scalar traces to lock target times before replay. Replay seed1-q75, seed1-q50 and
seed3-q75 with the identical no-kick Z-only configurations and sparse `snapshot_steps`.

For primary seed1-q75 capture q10/q30/q50/q70/q99 nearest-time fields. For the two replication runs
capture nearest-mean fields relative to those primary targets. Also generate exact all-zero D and
uniform-mean controls. Assert replay scalar parity at every selected time and write `d_field_lock.json`
with complete provenance/checksums.

Because these are 24 s runs, execute strictly one 40k worker under `setsid nohup` and the long-run
resource contract. They may run sequentially; never overlap them with sibling long jobs.

### 2.2 Prepared states

Create and exact-fork:

- canonical low after >=8 s accepted interictal activity;
- canonical high after >=`max(5 s,8 tau_H)` at H1/D50/a_X=1;
- H6 low/high sentinel preparations.

Retain raw state hashes and validation traces.

### 2.3 Write manifests before map execution

Primary H1 manifest:

```text
6 D fields x 7 uniform a_X x 2 states = 84
```

H6 sentinel:

```text
3 D fields x 3 uniform a_X x 2 states = 18
```

Total `102` unique rows. Write them all before launching. Every row references a field checksum,
state hash, noise, duration/tail, output path and sentinel.

### 2.4 Smoke and bounded scheduler

Run one H1/healthy/a_X=1/low row for 1.5 s. Measure actual RSS/wall, numerical margins and artifact
completeness. Enable a second short worker only if:

```text
MemAvailable >= 96 GiB + 2*1.35*RSS_single
swap stable and sibling reserve respected
```

Amendment 2026-08-04 — the two-worker rule above is the floor, not the cap. Workers beyond two
additionally require the worst-case extended-row budget to fit:

```text
n * 1.35 * 3 * RSS_single <= MemAvailable - 96 GiB
n <= min(MAX_MAP_WORKERS = 8, cpu_count - 2)
```

Use a bounded producer with at most `n_workers` pending rows. Never submit the full map at once.

### 2.5 Geometry map

Run breadth-first over D/a_X/state, with H6 interleaved as sentinel. A scientific negative never stops
the 102 rows. Each cell writes atomic raw summary, numerical evidence and DONE sentinel.

Aggregate entry, survival and return separately. Do not calculate a probability or P=0.5 contour yet.

## 3. E2 — boundary replication and spatial-field controls

Freeze boundary selection rules before opening aggregate outcomes. Extend adjacent label changes and
unresolved cells to 5 s.

For each relevant H1 boundary run:

- low trough/pre-IED/post-IED states x noises 401/405/406;
- high peak/trough states x noises 401/405/406;
- primary D field plus matched q50 and seed3-q75 full fields;
- uniform-D matched-mean control at sentinel boundary cells.

Only adequately replicated boundaries receive probability-like estimates; all others remain brackets.
Record microstate and field effects separately.

Actual dynamic X-field controls are deferred until E4 captures X fields. After E4, add at relevant
return boundaries: actual X, seed731 permutation and uniform same-mean fields. This is a sensitivity
appendix to geometry, not post-hoc parameter selection.

## 4. E3 — short slow-vector field

Select 12--20 landmarks by the locked rule. When brackets exist, sample both sides of entry and return;
otherwise use the fixed 3x4 D/a_X grid.

From matched prepared states unfreeze Z/X for 300 ms. Estimate 50--300 ms slopes for mean D/a_X and
core/axis/off-axis field projections. Overlay vectors on geometric brackets, but label them local drift,
not a closed orbit.

Emit `slow_vector_field.json` and one of:

```text
DX_GEOMETRIC_PATH_PRESENT
DX_GEOMETRIC_PATH_ABSENT
DX_DYNAMIC_VECTOR_MISALIGNED
DX_MAP_UNRESOLVED
```

This label does not authorize or block E4.

## 5. E4 — three dynamic reconnaissance trajectories

Write exactly three rows before running:

- H1 r025 only;
- archived q75 Z and current unretuned LC1 X by artifact provenance;
- connection seed1, noises 401/405/406;
- no kick/reset/parameter step; M/K/A/ELR off.

Run one 40k worker. Minimum 32 s. At 20 s record onset-search status. If onset exists by 32 s, continue
at least 12 s after onset and, after offset, until 8 s recovery or the 45 s cap. If no onset by 32 s,
stop at 32 s.

Store full D_i/a_X,j snapshots at every available lifecycle landmark and record first-passage fields.
These runs are completed reconnaissance regardless of outcome. They answer where the real slow path
goes and which frozen projection assumptions fail.

## 6. E5 — direct spatial causality

Prefer exact states from E4: pre-onset, onset, early high and late-high/pre-offset. If unavailable,
choose nearest map landmarks deterministically and mark the substitution.

### 6.1 Positive recruitment

Build core A, core B, axial, transverse, global and shuffled-axial masks before responses. Equal-size
local masks match cell count, positive charge, RMS and duration. Run separately labelled charge-matched
and RMS-matched global controls. Use two healthy-safe amplitudes and 10 ms pulses.

Primary outputs: first passage, newly recruited area, core polarity, axial/off-axis expansion and
finite-time gain. Core A versus core B defines forward/reverse.

### 6.2 Signed projected response

Lock the 8--12 physical bases and seed731 random controls. Run +/-/sham with common random numbers at two
linear amplitudes. Compute the low-dimensional response matrix at 50/150/300/500 ms and its SVD.

Do not fit 512D DMD here. Emit spatial labels independently from temporal labels. Global/transverse or
unresolved spatial results do not block E6/E7.

## 7. E6 — conditional X calibration

If no high-state return/offset bracket exists, write `X_CALIBRATION_NOT_IDENTIFIABLE` and do not invent a
target. E7 then cannot add retuned X candidates, but the unretuned reconnaissance remains valid.

If a bracket exists, combine `a_off(D)`, local drift and actual E4 trajectory. Preselect one
two-knob family and run at most 3x3. Require IED mean availability >0.9 and high-state crossing in
1--3 s. Select at most two candidates and derive postictal protection from measured D recovery.

## 8. E7 — <=6 nominal lifecycle runs

Only H1 r025 participates. Cross at most two E6 X candidates with noises 401/405/406. If no calibrated
candidate exists, do not manufacture six rows; retain E4 as the dynamic result.

Use the adaptive 32--45 s contract. Aggregate pre/onset/high/offset/post/recovery separately and require
multivariate IED statistical return, not mean-rate return.

Temporal and spatial outcomes remain separate. A temporal candidate is reported even when spatial
mechanism is global or unresolved; it does not become a spatiotemporal candidate.

## 9. E8 — candidate-only causality, robustness and confirmation

Only after a nominal temporal candidate is frozen:

1. Z-frozen, X-off and H-off matched ablations;
2. H1 r050 robustness, not reselection;
3. connection seed3 and noise 402/403/404 confirmation without tuning.

Candidate failure on confirmation is reported and does not reopen parameter selection.

## 10. E9 — conditional operator/eigenmode formalization

Run only under a real onset, reproducible direct spatial contrast, or a need to distinguish eigenvalue
softening from non-normal gain.

First formalize the projected response operator with held-out trials and bootstrap. Only if stable,
optionally fit the 16x16 E/I operator. Withhold eigenvalue/eigenvector claims on unstable fits.

Test whether pre-onset response predicts the real onset core, polarity and early first-passage field.

## 11. Resource and nohup implementation

- T <20 s: maximum `MAX_MAP_WORKERS = 8` workers under the §2.4 sizing rule; T >=20 s: strict one worker.
- Threads: OMP/OpenBLAS/MKL/NUMEXPR=1.
- Swap baseline is sampled per stage. +256 MiB stops new submission; +512 MiB and rising terminates only
  the newest LC3 worker.
- Every long stage: `setsid nohup`, exact launcher PID/SID, stage flock, wall guard, atomic
  RUNNING/DONE/FAILED and per-row DONE files.
- Resume only from valid DONE plus matching manifest/hash. Stale RUNNING is not completion.
- Wait by PID. Never use `pgrep -f` and never signal sibling work.

Before every 40k stage, log MemAvailable, swap, sibling 40k count and chosen worker limit. If the memory
reserve fails, defer submission rather than changing the scientific matrix.

## 12. Figures, archive and stop

Generate only figures supported by completed evidence:

1. field-preserving geometry and brackets;
2. mean D-X path plus field drift;
3. direct spatial response;
4. lifecycle candidate only if one exists.

Write Chinese `figures/README.md` after visual inspection. Archive stage completion, three-axis
verdicts, exact-fork status, row/seed coverage, hashes, tests, resources and forbidden claims.

Stop for review after the registered program. Do not add M morphology, K/A/ELR or final paper figures.
