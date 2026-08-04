# FCXR-LC3 event-driven pivot — DESIGN

Status: **DESIGN, pending user sign-off**
Date: 2026-08-04
Supersedes the primary role of: `2026-08-03-topic4-fcxr-lc3-dx-spatial-instability-design.md`
(that spec's frozen `D × a_X` geometry is **demoted to a mechanism appendix**, not deleted)

## 1. Why the pivot

The frozen 102-cell geometry answered what it could answer and is now exhausted. It
established three assets that the earlier M4 line did not close:

- a bounded high branch exists that is neither runaway nor a refractory ceiling
  (102/102 rows clean, refractory-ceiling fraction 0.0000 everywhere);
- the complete per-cell wear field is enriched along a corridor joining the two
  cores (core/off-axis 2.6x, reproduced across connection seeds);
- `X` holds authority to close that high branch, with the frozen boundary between
  `a_X = 0.80` and `0.65`, identical at all six wear levels.

What it never measured is the bearing quantity:

> **how many interictal events does it take to push the system into a seizure?**

A frozen coordinate deletes exactly the history that question is about. From here the
primary analysis is an event-driven trajectory, and the frozen map is an appendix.

## 2. Corrected target

The hard target is the closed loop, not any single leg:

```
repeated IED -> D/Z accumulation -> entry into a bounded high state
             -> X termination -> Z recovery -> returning IED
```

### 2.1 Demoted

**High-state spatial localisation is a diagnostic, not a gate.** Wide recruitment
during the high state is allowed and expected; a multi-contact broad plateau does not
fail the lifecycle. The only spatial requirements are:

- onset must not be the whole array flashing simultaneously;
- an identifiable core/axis-related early entry;
- after that, broad recruitment is permitted.

### 2.2 The five hard gates

1. a robust, non-singleton parameter window (adjacent grid cells and multiple seeds);
2. autonomous termination;
3. return to the interictal statistical neighbourhood;
4. returning IED after offset;
5. the complete lifecycle in one continuous no-kick run.

### 2.3 Full acceptance statement

```
>=3 no-kick returning IED
  -> D/H accumulate and autonomously cross into a bounded high state
  -> 1-5 s bounded ictal state
  -> X depletes causally after onset
  -> autonomous offset
  -> postictal protection
  -> Z recovery -> X recovery
  -> returning IED statistics back in the frozen interictal neighbourhood
```

## 3. Primary measurements

### 3.1 `N_IED_to_onset`

Number of complete **returning** interictal events between a stable interictal start
and the first entry into a sustained high state.

Entry classes, reported verbatim, never collapsed:

- `ONE_SHOT` — 0 or 1 returning event before onset (explosion, not accumulation);
- `CUMULATIVE` — `>= 3` returning events before onset;
- `AMBIGUOUS_2` — exactly 2;
- `NO_ONSET` — no sustained high state within the record.

`>= 3` is the registered accumulation bar. The upper bound is deliberately not locked.

### 3.2 `Q_IED` cumulative dose

Event count alone is not load: one large event is not one small event. Two doses are
computed and **both reported**; neither may stand alone.

**Primary — active-fraction dose.** Uses the same frozen calibration that defines the
events, at the 1 ms bin the detector runs on:

```
dose_af(k)  = integral over event k of [ af(t) - floor_af ]+ dt          (ms, fraction-weighted)
Q_af(k)     = sum of dose_af(1..k)
```

`floor_af = 3.125e-05` and the event bar `0.03978125` come from the frozen LC1
baseline contract; nothing new is calibrated.

**Secondary — population-rate dose**, the form written in the redirect, at the full
0.05 ms integration step:

```
dose_rate(k) = integral over event k of [ r_E(t) - r_base ]+ dt          (Hz*ms)
Q_rate(k)    = sum of dose_rate(1..k)
```

`r_base` is the run's own pre-onset quiet median population rate, recorded explicitly
in the output so the choice is auditable rather than implicit.

The current-based form `Q^Z` over core/axis is **not** computed in this revision: the
inhibitory current is not retained during the run and adding it touches the stepping
loop. It is deferred with that reason recorded, not silently dropped.

### 3.3 Per-event slow state

For each event, the regional means of `D = 1 - z`, `H`, `X`, `y` over
`core_A / core_B / axial / off_axis`, taken from the nearest full-field snapshot on
each side of the event, with the snapshot lag recorded. Snapshot cadence is 250 ms and
interictal events are ~1 s apart, so each event has snapshots on both sides.

**Regional decomposition is mandatory, means alone are forbidden.** The slow-flow
probes already showed a mean whose sign is opposite to the cores': mean `X` drift was
`+0.033/s` while both cores depleted, because 85% of cells are off-axis. Any
per-event readout reported only as a whole-array mean would invert this result.

## 4. Required figures

**Figure A — event accumulation ladder.** Event index on the horizontal axis; per
event: core/axis `D_k`, `H_k`, core `X_k`, event dose, and whether it returned. Marks
the event at which the system stops returning.

**Figure B — event-indexed slow trajectory.** The `(D_axis, H_axis)` plane, but drawn
as the *actual* trajectory with one point per event coloured by event index, `X` on a
third panel. Shows whether `D` and `H` accumulate together, whether onset approaches a
boundary, and whether the post-offset trajectory returns to the pre-onset neighbourhood.

**Figure C — parameter phase diagram**, two bearing axes only: per-event `Z` depletion
strength / recovery ratio against `X` depletion / recovery timescale. Each cell carries
an outcome label, not low/high:

```
one_shot | IED_train_no_onset | onset_no_offset | offset_no_recovery | full_lifecycle
```

Passing cells additionally carry `N_IED_to_onset / T_ictal / T_recovery`. A robust
window means **adjacent cells and multiple seeds** are all `full_lifecycle` — not one
pretty point.

Per repo convention every figure directory gets a Chinese `figures/README.md` written
after visual inspection.

## 5. Staged execution

### Stage 1 — entry only

Freeze `a_X = 1` so termination cannot happen yet. Keep the two low-threshold cores,
the noise, dynamic `Z`, dynamic `H`. Scan **two** entry parameters only: per-event `Z`
depletion/recovery strength, and the effective `H` threshold position. No large
`D x X x H` grid.

Every trajectory records the full event ledger of §3.

**M4 is the entry positive control, not background.** Two arms sharing the two cores,
the noise, the `Z` configuration and the lifecycle adjudicator:

| Arm | Entry | Carrier | Exit |
|---|---|---|---|
| `M4-X` | dynamic `Z`/`q_I` | M4 shared-inhibition bounded state | `X` |
| `LC3-HX` | dynamic `Z`/`q_I` | `H` high branch | `X` |

Routing on the Stage-1 result:

- M4 enters, LC3 does not -> the defect is the `D -> H` coupling;
- neither enters -> the `Z` calibration or the IED substrate has drifted;
- both enter -> entry is locked, go straight to termination.

### Stage 2 — termination, only after entry is robust

Open dynamic `X`. Scan `X` depletion strength / rise time and `X` recovery time.
Targets: `X` roughly still during the IED phase; clear depletion once the high state
starts; autonomous offset within 1-5 s; a protection window after offset.

**Amendment 2026-08-04, measured on the first no-kick trajectory (noise 401).** The
axes above are not the ones the evidence points at, and Stage 2 is gated on an
arbitration before any of them is scanned.

That trajectory ran an ictal bout from 5.0 s to the end of its 45 s record with no
autonomous offset. Its 44.75 s state says:

- `X` is neither slow nor still falling. `tau_x_down = 500 ms` is 80 time constants
  inside a 40 s bout, and `X` has settled at its own set point, mean `0.394`. The
  sensor mean never reaches its gate (`59.4 Hz` against `y_gate = 76.64 Hz`), so the
  Hill drive never saturates. **Depth is the bottleneck, not speed**, and recovery
  time never enters the picture because `X` never turns around.
- **Every cell is already below the frozen map's `a_X = 0.65` return boundary** and
  the bout persists regardless, so "`X` cannot brake hard enough" is not established
  either.
- What differs is `D`. The frozen map's wear axis spans mean `D` in `[0, 0.097]` --
  the levels a 24 s interictal record reaches. The trajectory drove `D` to `0.663`,
  6.8x beyond the map's largest level, and flattened it: the 2.8x core-versus-off-axis
  enrichment present before onset was gone by 44.75 s, every region at `0.663`.

The map cannot arbitrate because its domain does not contain the trajectory. A 2x2
seeded from the byte-parity-verified late-bout state does, crossing `D` in
`{observed 0.663, map maximum 0.097}` with `X` in `{observed field, the 0.10 x_min
floor}`, each arm on the map's own screen/extend protocol and classifier so the
outcomes sit on the same scale as the 102 map rows
(`scripts/run_topic4_fcxr_lc3_dxprobe.py`).

Routing, registered before the arms ran:

- max-brake still high **and** map-`D` terminates -> `D` is the cause; `X` cannot
  terminate at real wear levels at any depth, and Stage 2 scans how to **bound the `D`
  excursion**, not `X`;
- max-brake terminates -> `X` can terminate, and Stage 2 scans how `X` reaches that
  depth naturally, i.e. `K_y` / `y_gate` / `x_min`, **not** `tau_x_down` or
  `tau_x_up`;
- control not high -> the frozen replica fails to reproduce the bout and the probe is
  void; neither branch may be read.

### Stage 3 — recovery

Acceptance is distributional, not "the mean rate came back". Compare against the frozen
LC1 baseline contract, which already holds the reference sample of 34 returning events:

- returning event count and rate against the band `0.086-3.15` events/s;
- duration distribution against the reference `8-19 ms` sample;
- participation against the reference `0.045-0.071` sample and `recruit_p90 = 0.0717`;
- pre-existing forward/reverse event structure;
- whether `D/H/X` return to the pre-onset statistical neighbourhood.

Recovery is not returning to an identical numeric point; it is being able once more to
produce IED under the original sparse, irregular statistics.

## 6. Claim boundary

Forbidden regardless of how good a trace looks:

- runaway; refractory tonic plateau; a global coherent oscillation; a kick-triggered
  high state; a brief trough; offset without returning IED; a frozen-state geometry
  presented as a lifecycle.

Additionally forbidden by this revision:

- reporting `N_IED_to_onset` without its entry class;
- reporting either dose alone;
- reporting per-event slow state as a whole-array mean without the regional split;
- calling a single passing grid cell a parameter window.

## 7. What happens to the current program

- E4 no-kick reconnaissance: **kept**, it is already an event-driven trajectory. Its
  output must be extended with the ledger of §3 before it can answer the bearing
  question. The run in flight answers only the binary "is there an onset".
- E5 direct spatial response: **demoted to diagnostic**, dequeued for now.
- E6/E7: **blocked on Stage 1**, since a termination knob is meaningless without entry.
- The frozen `D x a_X` map and the slow-vector field: **retained as appendix**, already
  complete and locked.
