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

Entry replicates and it accumulates: it is not one event tipping the tissue
over, it is a train whose gaps shorten until the tissue does not come back. The
events themselves stay ordinary-sized — 31 of the 34 sit inside the quiet
baseline's own participation range, and all 3 that exceed it are among the last
two before entry. What changes across the train is the timing and the wear it
leaves behind, not the size of the events.

**The frozen map had this all along and could not see it.** Every one of its 42
quiet cells ran 1500 ms against an ignition that takes 4000–6000 ms, so the whole
quiet side reported its own screen window; the registered temporal-geometry label
is therefore `DX_MAP_UNRESOLVED`, not the `DX_GEOMETRIC_PATH_ABSENT` its grid
superficially suggests. Watched to 12 s instead, a frozen wear field at mean
0.047 departs by itself at 7 s and one at 0.068 at 5 s, while a healthy field
does not depart at all.

So there is an entry bracket in the frozen geometry, the screen was 4.7× too
short to meet it, and what the trajectory's accumulation does is carry the tissue
across it — its whole-array wear at the last event before entry is 0.049, 0.054
and 0.034, straddling that 0.047.

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

## 3. The entry ledger: how many events, carrying how much

The two registered entry measurements never reached disk. The three trajectories
were launched before `build_event_ledger` was wired into the recon runner, and
source cannot be hot-edited under a live run; the dose integrates the 1 ms active
fraction the detector runs on, and only a 200×-decimated rate survives to the
NPZ, so it is not recoverable afterwards. Rather than repeat 45 s at 110 GiB for
a question that is over by 6 s, the identical registered preparation ran to 20 s
— the recon runner's own onset checkpoint.

All three re-simulations reproduce their recorded trajectories **exactly**: 174,
168 and 180 events, every onset time, duration and participation identical.

| seed | onset | `N_IED_to_onset` | `Q_af` | `Q_rate` | class |
|---|---|---|---|---|---|
| 401 | 5000 ms | 12 | 7.296 | 6820.6 | `CUMULATIVE` |
| 405 | 6000 ms | 15 | 11.933 | 11330.2 | `CUMULATIVE` |
| 406 | 4000 ms | 7 | 3.035 | 2764.3 | `CUMULATIVE` |

Per event, across the train (figure B):

- **wear rises in 3 of 3 seeds**, and the three whole-array curves lie almost on
  top of each other — the accumulation itself is highly reproducible across
  noise; what varies is how many events it takes to walk it;
- **it builds 3.1× faster in the cores than off-axis**, with the two cores and
  the tissue along the axis rising together (0.029 → 0.113-0.117 on seed 401)
  while off-axis tissue reaches 0.038;
- **the relay sits at 1.000 at the snapshot bracketing every event.** Entry is
  wear-driven and the relay takes no part in it.

### 3.1 A guard that failed two bit-identical seeds

The runner's own guard reported that seeds 401 and 406 did not reproduce their
recorded trajectories. They did. The shared prefix is bit-identical and the sole
difference is the last event: the 20 s run stops mid-event and records
19985-19999, while the 45 s run has the same event whole as 19985-20014.
Filtering both lists by "ends before the cut" keeps the truncated copy and drops
the whole one, so an identical trajectory counts one event too many.

The comparable span now ends one longest-event margin short of the cut, so
anything that could have been clipped is excluded from both sides
(`src/topic4_fcxr_lc3_reproduction.py`). Re-adjudication is a re-reading of event
lists already on disk, and it preserves the original verdict beside the corrected
one in `entry_ledger/reproduction_readjudicated.json` rather than overwriting it.

Two further defects surfaced while drawing the figures, both of the same family —
a rule that looked right and quietly answered a different question:

- the figures consumed seed 401's ledger while its own record said it had failed
  the guard. A ledger that does not reproduce the recorded trajectory is not a
  reading of it; the figures now honour the verdict and print any substitution;
- the regional panel compared "the cores" against `max(axial, off_axis)`, which
  picks the axis — and the axis moves *with* the cores. That framing prints a
  1.06× ratio over curves whose actual spread is 3.1×. Fastest and slowest
  regions are now read off the data.

## 4. The quiet side, watched past its own ignition time

The frozen quiet state at the relay the trajectories actually sit at before
onset (mean 0.99999979, so the a_X = 1.00 column is the matched one and the only
matched one), watched 12000 ms — twice the slowest observed ignition. Departure
is judged by the lifecycle detector over the whole record, not by the map's tail
classifier, because a cell that ignites at 5 s and self-terminates by 8 s settles
quiet while having departed.

| wear field | mean D | departs | tail | the 1500 ms screen said |
|---|---|---|---|---|
| `D_healthy` | baseline | **not within 12 s** | interictal | interictal workpoint |
| `D10` | 0.0473 | **7.0 s** | finite high orbit | interictal workpoint |
| `D30` | 0.0678 | **5.0 s** | finite high fixed | interictal workpoint |
| `D50` | 0.0819 | **2.0 s** | finite high orbit | interictal workpoint |
| `D70` | 0.0889 | **2.0 s** | finite high fixed | interictal workpoint |
| `Dmax` | 0.0972 | **2.0 s** | finite high fixed | interictal workpoint |

**Five of the six depart, and the screen called all six an interictal
workpoint.** At full relay only healthy wear holds. Every level drawn from the
interictal wear distribution is already on its way out; what differs is how long
it takes, from 7 s down to a 2 s floor set by the detector needing a couple of
1 s windows to call a bout. The map's quiet side at relay 1.00 is a screen
artifact almost everywhere, and the screen was 4.7× shorter than the slowest of
these departures.

So the entry bracket sits between healthy wear and 0.047, and the whole
interictal wear range above it is unstable rather than a set of workpoints.

**This is what the trajectory walks.** Its relay reads 1.000 at the snapshot
bracketing every pre-onset event, so it is on exactly this column of the map, and
its whole-array wear climbs from near zero to 0.049, 0.054 and 0.034 — from the
one level that holds into the range that does not.

This **retracts a statement made earlier the same day** in this file's first
draft: that no departure would mean entry needs wear in motion. Wear does not
need to be moving. A fixed field at 0.047 is already unstable; it simply takes
7 s to show, which no 1500 ms screen could see.

What the trajectory's accumulation does is *carry the tissue across that
bracket*. At the last event before entry its whole-array wear is 0.049, 0.054 and
0.034 on the three seeds — straddling `D10`'s 0.047.

**The shape control holds**, which is what licenses comparing them at all: the
frozen `D10` field is core 0.104 / axial 0.100 / off-axis 0.038, a 2.74× core
enrichment, against the trajectory's 0.113 / 0.038, a 2.96×. These are the same
spatial pattern at the same mean, not a uniform field standing in for a
structured one.

**Not licensed:** comparing the two latencies. The frozen cell's 7 s runs from
the moment its wear field is imposed; the trajectory's 5 s runs from the start of
the recording, with wear climbing throughout. The clocks start at different
events, so "the trajectory enters sooner" is not a statement these data support.

## 5. How far anything got

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

## 6. The methodological thread

**The setup contributes to the answer.** Three instances, each found by asking
what the apparatus was adding:

1. **A window that opens on a transition measures the transition.** A 5 s mean
   starting with the collapse out of a discharge read as 3.07 Hz; measured with
   the collapse excluded it is 0.093 Hz and zero events.
2. **A clamped variable that would otherwise recover measures the clamp.** The
   silence below relay 0.380 was the clamp, not the tissue: released, the relay
   refills to 0.59-0.60 and leaves a 32-40 Hz train.
3. **A window shorter than the phenomenon measures the window.** All 42 frozen
   quiet cells ran 1500 ms against a 4000-6000 ms ignition; watched to 12 s, two
   of them depart.

All three produced retractions, the third included: this file's own first draft
said that no departure would mean entry needs wear in motion, and `D10` departed.
What the code precondition (`window_is_adequate`) did buy was the *label* —
`DX_MAP_UNRESOLVED` was emitted before any cell was re-watched, and it turned out
to be exactly right where `DX_GEOMETRIC_PATH_ABSENT` would have been wrong.

**A rule that looks right can quietly answer a different question.** Four
instances today, all in adjudication code rather than simulation:

4. **A boundary filter applied to two records of different length.** "Ends before
   the 20 s cut" keeps the event this run truncated and drops the whole copy the
   45 s run holds — two of three bit-identical seeds reported as not reproducing.
5. **A consumer that does not read its own gate.** The figures used a ledger
   whose record said it had failed the reproduction guard.
6. **A grouping that contains its own contrast.** "Cores against the rest" with
   the rest taken as `max(axial, off_axis)` picks the axis, which moves *with* the
   cores: a 3.1× spread prints as 1.06×.
7. **A flag that names something other than what it appears to.** Per-arm
   `is_control` marks whichever arm a batch promoted when its grid held no
   registered setting, so it pointed at relay gate 75, not the registered 76.6386.

The first family is caught by asking what the apparatus adds. The second is
caught only by exercising the path on data where the right answer is known
independently — which is why the reproduction guard, the stage labels and the
figure titles are all tested against cases whose answers were worked out by hand.

## 7. Fields that must not be read

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

Added by this stage:

- `entry_noise*.json::reproduces_recorded_trajectory` — the runner's own field is
  wrong for seeds 401 and 406, both of which are bit-identical to their recorded
  trajectories. Read `entry_ledger/reproduction_readjudicated.json`, which keeps
  both verdicts.
- any frozen geometry cell's `resolved_label` **on the quiet side** — those cells
  ran 1500 ms against a 4000-6000 ms ignition, and two of the three re-watched so
  far depart. `INTERICTAL_WORKPOINT` there means "did not depart within 1500 ms",
  not "does not depart". Read `quiet_watch/` for the levels it covers.
- a frozen arm's `total_ms` as evidence of how long a **surviving** high branch
  was observed — a persisting branch never triggers the protocol's extension, so
  every such arm saw only the 1.5 s screen however long a budget it was given.
