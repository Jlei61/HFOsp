# FCXR-LC3 — the lifecycle on E1146's own geometry, and what "bidirectional" turned out to mean

**Date**: 2026-08-07
**Status**: two connection seeds landed; paper-ready figure + README written
**Figure**: `results/paper-ready-figure/fig_subject_snn_e1146_runaway/figures/`
**Records**: `results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/subject_runaway/seed{1,3}*.{json,npz}`

---

## Abstract

The request was to put the lifecycle on the patient's own electrode layout, start from the
interictal bidirectional working point, and let it run away. Both parts landed. On E1146's
geometry, with the two low-threshold cores at the centroids of that subject's two interictal
template source contacts (13.05 mm apart), the network runs away on its own from a train of
returning interictal events — at 5.0 s for connection seed 1 (12 pre-entry events) and 4.0 s for
seed 3 (10), both labelled `CUMULATIVE`, both ending at `wear_end ≈ 0.6635`.

The scientifically load-bearing part of this entry is **not** that number. It is what happened
when the figure was drawn and inspected: the two middle columns, keyed as they were by the sign
of an onset-vs-axis correlation, told a false story. Per-cell onset maps show that **most
pre-entry events are patches around one core, roughly 7–8 mm across, not transits between the two
cores 13 mm apart**. For such a patch the sign of that correlation reports which flank of the
ignition point is longer, which is a different question from where the event started. The two
events the figure had picked as "strongest forward" and "strongest reverse" both ignited on the
same side.

Re-keying the columns to source identity — which the four-column standard asks for anyway — gives
a cleaner and more honest result, and one that changes the headline: at seed 1 the two source
regions each ignite 5 of 12 events; **at seed 3 all 10 ignite at one source region and none at the
other**. Spontaneous two-source behaviour is connection-seed dependent, exactly as the style guide
already required to be stated, and now instantiated.

---

## 1. What ran

| | seed 1 | seed 3 |
|---|---|---|
| ran away | yes | yes |
| entry | 5.0 s | 4.0 s |
| entry class | `CUMULATIVE` | `CUMULATIVE` |
| returning events before entry | 12 | 10 |
| `wear_end` | 0.66347 | 0.66352 |
| `relay_end` | 0.39444 | 0.39211 |

Same substrate (`PP.build_substrate(seed)` = the E1146 subject placement), same noise seed (401),
same registered configuration, same montage (`narrow`, 15 contacts, 0 off-sheet), 45 s each. Only
the connection seed differs.

An 8 s capture re-simulated the opening window at each seed and kept the per-cell onset map of
every returning event before entry. Both captures reproduced the long run exactly (seed 1: 40
events within 7966 ms; seed 3: 50), so the maps belong to the same trajectory the ledger scored.

## 2. The measurement that changed the reading

`_event_direction` correlates each cell's first-spike time against its position along the
source→sink axis and reports the sign. Applied to the 12 seed-1 events it gives 5 forward and 7
reverse, which is where `bidirectional=True` came from.

Measuring the ignition site instead — the centroid of the earliest 10% of cells to fire — and the
along-axis extent of each event:

| kind | n | along-axis extent | ignition |
|---|---|---|---|
| patch around one core | 9 | ≈ 7–8 mm | 4 at the sink core, 5 at the source core |
| full-axis transit | 1 | −3.1 → 16.1 mm | sink core, and it does travel |
| near-simultaneous | 2 | full axis | mid-axis, \|corr\| < 0.08 |

The two cores are 13.05 mm apart. Nine of twelve events never cross. The correlation sign on
those nine tracks **which core ignited** (sink-core ignitions read positive, source-core
negative), not a direction of travel. The figure's original picks — the largest \|corr\| of each
sign — were event 0 (patch at the sink core) and event 6 (transit from the sink core): **both
ignited on the sink side**, and the star drawn on the source core for the first one contradicted
its own onset map. That contradiction is what visual inspection caught.

## 3. The rule now used, and why it is not fitted to the data

`src/topic4_fcxr_lc3_ignition.py`, 12 tests in `tests/test_topic4_fcxr_lc3_ignition.py`:

* ignition = centroid of the earliest `EARLY_Q = 10`% of cells to fire (NaN for cells that never
  fired), None below `MIN_CELLS = 20`;
* a source region owns the ignition when it is `CLOSER_BY = 2.0`× nearer than the other;
  otherwise the event is unattributed rather than pushed onto the nearer region.

The 2× is a stated convention, not a threshold tuned to separate these events. Attributed events
sit 0.74–2.94 mm from their region; unattributed ones sit ≥ 5.65 mm from the nearer one. Any
factor between 1.4 and 5 gives the identical partition, and a test pins that.

Selection for the two middle columns is then "the event whose ignition sits nearest that region",
and the star is that event's own ignition — per the style guide's `星号只标该事件的实际 source`.

## 4. Result, per source region

| | seed 1 | seed 3 |
|---|---|---|
| ignited at source region A | 5 | **0** |
| ignited at source region B | 5 | 10 |
| unattributed (mid-axis) | 2 | 0 |

Seed 3's tempA column is therefore empty, and says so on the geometry rather than as a blank box.

The old `interictal_directions` counts (5fwd/7rev and 10fwd/0rev) remain in the records and are
still what `bidirectional` is computed from. They are kept as-is; §7 says how they may be read.

## 5. What the electrode readout adds

Contact traces are the share of each contact's weighted neighbourhood recruited per 2 ms bin
(weights sum to 1 per contact), so rows are comparable and 1.0 means the whole neighbourhood
fired. An earlier reading of this file as "already normalised" was wrong: every column reaches
exactly 1.0 because during the discharge every contact's neighbourhood fully recruits, which is a
property of the runaway, not of scaling. The figure now plots the fraction directly instead of
rescaling each row to its own maximum — under the current data that rescaling was a no-op, but it
would have silently destroyed cross-row comparability on any run where a contact did not saturate.

Seed 1, half-recruitment (≥ 0.5):

* pre-entry, per event: 2–5 of 15 contacts (the three full-axis events: 8–9);
* during the discharge: 15 of 15;
* time to reach half recruitment after entry: on-axis shaft (0.1–3.8 mm off axis) median 0.15 s,
  off-axis shaft (6.5–6.9 mm) median 1.05 s — **with SCL9 an exception at 0.18 s**, so this is not
  a clean two-stage story and must not be written as one.

## 6. The methodological thread, continued

This entry adds a fourth instance to the family already recorded in
`fcxr_lc3_entry_ledger_and_path_label_2026-08-05.md` §6 — *an adjudication rule that looks right
can quietly answer a different question*:

* boundary filters across records of different length (the reproduction guard);
* groupings that contain their own contrast (`max(axial, off_axis)` picks the axis, which moves
  with the cores);
* flags that name something else (`is_control` marking a batch-promoted arm);
* bout-end-of-record read as termination (the brake adjudicator);
* **and now: a signed correlation along an axis read as direction of travel, when for a local
  patch it reports flank asymmetry around the ignition.**

The common shape: the statistic is computed correctly and the name is plausible, so nothing fails
loudly. What caught this one was the project rule that every figure is eyeballed after generation
— the star and the onset map disagreed on the same panel.

A second, smaller instance in this entry: I stated mid-analysis that `contact_trace` was already
normalised, on the evidence that all 15 column maxima were exactly 1.0. The producer
(`_contact_trace` with `wts = w / w.sum()`) shows it is raw and that 1.0 is full local
recruitment. Identical maxima are not evidence of normalisation when the ceiling is structural.

## 7. Fields that must not be read

* **`interictal_directions.n_forward` / `n_reverse` / `bidirectional`** — these are signs of an
  onset-vs-axis correlation. For the majority of these events, which do not cross between cores,
  the sign encodes which core ignited plus flank asymmetry. Do **not** read them as "events
  travelling forward" and "events travelling in reverse", and do not report `bidirectional=True`
  as "the model produces bidirectional propagation". The source-identity counts in §4 are the
  layer that supports a statement about the two source regions.
* **Cross-layer comparison with the accepted E1146 electrode-level counts.** That readout is
  electrode-level, endpoint-centroid, `k_dir`-based and runs **without** slow variables; this is
  per-cell onset **with** slow variables. Different measurements of differently-named things —
  event counts are not interchangeable, and "seed 3 flipped from bidirectional to unidirectional"
  is not a supported statement. (Note also the 2026-08-07 finding that published Fig4 artefacts
  ran `k_dir=3`, not the documented 2 — one more reason not to cross this boundary casually.)
* **The 1-second off-axis lag** — SCL9 breaks it (§5). It is a description of four contacts with
  one exception, not a propagation-speed result.
* **The figure as evidence about the patient.** Style-guide boundary: model substrate + two source
  regions + virtual electrode readout. Events of either source identity appearing here do not
  establish the real patient's mechanism.

## 8. What this does and does not close

Closes: the user's request — the lifecycle now runs on the subject's own layout, from the
interictal working point, and it runs away, at two connection seeds, with figures and a README.

Does not close: entry still ends in the same place the loop always ends — a sustained discharge
that does not terminate within the 45 s record. Nothing in this entry addresses termination or
recovery; the loop's last leg is where it was left in the 2026-08-06 entry.

Newly opened, and honestly small: whether **both** source regions igniting is common or rare
across connection seeds. n = 2 (one yes, one no). Any statement stronger than "it depends on the
connection seed" needs a seed sweep that has not been run.
