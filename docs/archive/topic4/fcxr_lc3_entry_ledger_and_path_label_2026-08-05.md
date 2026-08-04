# FCXR-LC3 — entry is real and cumulative; what the frozen map can and cannot say about it

Date: 2026-08-05
Artifacts: `results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/`
  · `dynamic_reconnaissance/aggregate.json` (three no-kick trajectories)
  · `slow_vector_field/temporal_geometry_label.json` (the registered four-way label)
  · `entry_ledger/` (per-event ledgers)
  · `quiet_watch/` (frozen quiet state, watched past ignition)
  · `figures/` + `figures/README.md`

## Abstract

A block of worn tissue, given nothing but its own noise, discharged interictal
events that grew closer together until it entered a high state — three times out
of three, on three independent noise seeds, with 12, 15 and 7 events preceding
entry at 5.0 s, 6.0 s and 4.0 s. Nothing was kicked, reset or stepped.

Two things follow, and they pull in opposite directions.

The first is that entry replicates and it accumulates: it is not one event
tipping the tissue over, it is a train whose gaps shorten until the tissue does
not come back. That is the first half of the loop this stage set out to track.

The second is that **the frozen map cannot see this**. Every one of its 42 quiet
cells ran 1500 ms, against an ignition that takes 4000–6000 ms. So the map's
quiet side reports its own screen window, not the tissue, and the registered
temporal-geometry label is `DX_MAP_UNRESOLVED` rather than the
`DX_GEOMETRIC_PATH_ABSENT` its grid superficially suggests.

## 1. The three trajectories

| seed | onset | events before onset | all returning | first gaps | last gaps | relay floor |
|---|---|---|---|---|---|---|
| 401 | 5000 ms | 12 | 12 | 456 ms | 266 ms | 0.388 |
| 405 | 6000 ms | 15 | 15 | 383 ms | 263 ms | 0.388 |
| 406 | 4000 ms | 7 | 7 | 634 ms | 495 ms | 0.389 |

All three classify `ICTAL_LIKE_BOUNDED` / `RECON_HIGH_WITHOUT_OFFSET`, and all
three entry classes are `CUMULATIVE` — every seed clears the 3-event
accumulation bar by a wide margin.

The gaps shorten in 3 of 3 seeds (first-half against second-half mean:
0.456→0.279, 0.391→0.309, 0.795→0.372 s). **15 of 31 gaps fall below 0.317 s,
the closest spacing the frozen baseline ever produced.** That is a real
acceleration and it is not universal: seed 406 crosses the baseline floor once.
Report the counts, not "they all break through".

Claim boundary: reconnaissance. Three seeds at one point is replication of a
phenomenon, not a parameter acceptance, and none of these arms terminated.

## 2. The registered four-way label: `DX_MAP_UNRESOLVED`

The slow-vector stage completed without emitting its registered label. It is
derived post-hoc in `src/topic4_fcxr_lc3_pathlabel.py` from evidence already
frozen, and the plan states the label authorizes nothing either way, so a
post-hoc emission changes no gate.

What the frozen 84-cell H1 grid actually holds:

- **quiet side**: all 42 cells `INTERICTAL_WORKPOINT`, across 6 wear fields and
  7 relay levels. **All 42 ran 1500 ms; none were extended.**
- **high side**: the branch survives at relay ≥ 0.80 and dies at ≤ 0.65, in all
  six wear fields. 30 of 42 were extended to 5000 ms.

So the return bracket is present and adequately observed; the entry bracket's
existence is **untested**, because a quiet cell that was watched for 1500 ms has
not been given the time ignition takes. Reading those 42 cells as "the quiet
state never departs" would report the screen window.

`window_is_adequate` therefore checks the quiet window against the fastest
observed ignition **before** absence is allowed to mean anything, and the label
falls out as `DX_MAP_UNRESOLVED` with a 2500 ms shortfall on record.

### 2.1 The drift, on the mean and per region

The 12 slow-flow rows are all high-state; the quiet side has no vector either.
Among the six at or above the return bracket:

| readout | toward return |
|---|---|
| whole-array mean relay slope | **0 of 6** |
| core A | 6 of 6 |
| core B | 6 of 6 |
| along the axis / off axis | 0 of 6 |

The array mean rises at +0.033 to +0.067 per second while both cores fall by
0.004 to 0.042. **The mean and the cores point at opposite sides of the return
boundary**, so the mean alone erases the only component moving the right way.
Both are reported; neither stands alone.

## 3. What is running, and what it decides

- **`entry_ledger/`** — the two registered entry measurements, `N_IED_to_onset`
  and the cumulative dose `Q_IED`, plus per-event slow-state snapshots. These
  never reached disk: the three trajectories were launched before
  `build_event_ledger` was wired into the recon runner, and source cannot be
  hot-edited under a live run. The dose integrates the 1 ms active fraction the
  detector runs on, and only a 200×-decimated rate survives to the NPZ, so it is
  not recoverable afterwards. Rather than repeat 45 s at 110 GiB for a question
  that is over by 6 s, the identical registered preparation runs to 20 s — the
  recon runner's own onset checkpoint — and **refuses to publish unless its
  events reproduce the recorded ones exactly**.
- **`quiet_watch/`** — the frozen quiet state at the relay the trajectories
  actually sit at before onset (mean 0.99999979, so the a_X = 1.00 column is the
  matched one and the only matched one), watched 12000 ms: twice the slowest
  observed ignition. Departure is judged by the lifecycle detector over the
  whole record, not by the map's tail classifier, because a cell that ignites at
  5 s and self-terminates by 8 s settles quiet while having departed.
  **A departure means the map was under-screened. No departure means entry needs
  wear in motion**, and the frozen-geometry screen can never find it — which
  would retroactively explain every negative frozen screen on this line.

## 4. How far anything got

33 arms on disk: **9 stop at "entered, did not stop", 24 at "stopped, did not
come back", 0 reach the full loop.** Figure C draws each with what was held
still to get it there, and hatches every arm that started already in the high
state so its entry was never tested.

One thing that surfaced while drawing it: **a high branch that persists never
triggers the protocol's extension**, so those arms are only ever observed for
the 1.5 s screen however long a budget they were given — including the
registered control arm that was handed 12 s. This is the same window problem as
§2, in a second place. Figure C labels them "watched 1.5 s of 12 s allowed".
The registered setting's persistence beyond 1.5 s is known from the 45 s
trajectory, not from these arms.

## 5. The methodological thread, now three deep

Each of these was found by asking what the setup contributed to the answer.

1. **A window that opens on a transition measures the transition.** A 5 s mean
   starting with the collapse out of a discharge read as 3.07 Hz; measured with
   the collapse excluded it is 0.093 Hz and zero events.
2. **A clamped variable that would otherwise recover measures the clamp.** The
   silence below relay 0.380 was the clamp, not the tissue: released, the relay
   refills to 0.59-0.60 and leaves a 32-40 Hz train.
3. **A window shorter than the phenomenon measures the window.** All 42 frozen
   quiet cells ran 1500 ms against a 4000-6000 ms ignition.

The first two produced retractions; the third was caught before it entered a
conclusion, because the check was made a precondition in code rather than a
habit. `window_is_adequate` and the return test's count-before-rate ordering are
both that lesson compiled in.

## 6. Fields that must not be read

Carried forward from `fcxr_lc3_dx_arbitration_and_hill_sweep_2026-08-04.md` and
still true:

- `hill_sweep.json::n_terminating` — says 12; arms reaching interictal is 0. Use
  `resolved_label` in three tiers.
- per-arm `mechanism_consistent` — compares an instantaneous relay minimum
  against a threshold measured under a sustained clamp.
- the aggregate `hill_sweep.json` / `return_gate.json` **row lists** — each holds
  only the arms of the batch that wrote it last. Figure C reads the per-arm
  files; anything reading the aggregate silently drops the earlier sweeps.
- per-arm `is_control` — marks whichever arm a batch promoted when its custom
  grid contained no registered setting, so it does not identify the registered
  gate. Compare `y_gate` against `base_y_gate` numerically.
