# Topic 4 rev5 implementation plan: target-informed Z/M early-energy bridge

Date: 2026-08-21
Status: authorized autonomous execution
Spec: `docs/superpowers/specs/2026-08-21-topic4-data-driven-zm-target-informed-early-energy-bridge-design.md`

## Execution rules

- Work only in `codex/topic4-data-driven-zm-ictal-transition`.
- Preserve rev4 discovery artifacts and historical Fig.5 outputs.
- Exhaust existing trajectories before new SNN runs.
- Primary fitting keeps E-to-E and E-to-I doses at `1.0`; 2%/5% are comparators only.
- Long runs use `nohup` plus a user systemd scope where available, one numerical thread per
  worker, measured-RSS worker count and at least 32 GiB free memory.
- A monitor checks process, memory, disk, hashes and completion at intervals of at least 600 s.
  It exits when the batch finishes and does not continuously poll.

## Task 0: audit and freeze inputs

1. Verify the rev4 substrate, model-ictal scorer, montage and completed candidate inventory.
2. Resolve the current Fig.3 checkpoint root and enumerate all complete exact-band E1146
   seizures.
3. Record that seizure 2 is display-only and the other 24 seizures form the target.
4. Hash every clinical checkpoint, frozen interictal record, model config and candidate artifact.
5. Fail closed on contact-order, band, window or fingerprint mismatch.

Outputs:

```text
target_informed_bridge_v1/input_audit.json
target_informed_bridge_v1/target_provenance.json
```

## Task 1: implement the clinical target producer

Create:

- `src/topic4_fig5_target_informed_bridge.py`
- `scripts/freeze_topic4_fig5_early_energy_target.py`
- `tests/test_topic4_fig5_target_informed_bridge.py`

The producer extracts 25 complete seizures using the existing Fig.3 spectral code, freezes
seizure 2 as display-only, and builds the 24-seizure pre/early target distribution. It writes raw
vectors, robust summaries, LOSO and split-half reliability, exact contact order and parity checks.

Tests cover exact-name alignment, no mirror/sign selection, display-seizure exclusion,
shaft-balanced errors, bootstrap determinism and synthetic static-axis/increment controls.

## Task 2: implement model readout and state-defined window

Add pure helpers for:

- paired Z/M-off robust-z baseline;
- 500 ms `10--150 Hz` Welch contact power;
- earliest state-qualified `W_read` on a 25 ms grid;
- energy, contact-field, increment and time-course distances;
- model-ictal fail-closed scoring;
- selection-aware target permutations.

The patient target cannot select a frame. Tests inject a later perfect-correlation frame and
require the earlier state-qualified window to remain selected.

## Task 3: zero-simulation rescore

Create `scripts/rescore_topic4_fig5_target_informed_candidates.py`.

Rescore:

- all completed full-dose Z/M-only threshold/adaptation candidates;
- the current exact/full-dose trajectory where evidence permits;
- historical 0--75% E-to-I morphology candidates as labelled comparators;
- 2% and 5% Fig.5 trajectories with full diagnostics.

Write raw component scores, missing evidence and whether a paired baseline exists. Do not invent
scores for incomplete recorder contracts.

Decision:

- if a full-dose candidate is model-ictal and bridge-evaluable, proceed directly to selection;
- otherwise launch only the minimal Z/M-only canary in Task 4.

## Task 4: minimal full-dose Z/M canary

Create a frozen candidate manifest before simulation. Start with the underexplored local plane:

```text
s_I   = 0.70, 0.80, 0.90
tau_z = 2500, 5000, 10000 ms
tau_m = 500 ms
G_m   = reference
EE/EI dose = 1.0/1.0
```

Reuse already completed exact duplicates. Run only missing cells on fit seed 1801. Every run saves
at least 1.5 s after transition, the full contact trace, global recruitment, population rate and
Z/M traces. A no-transition run continues to the fixed horizon.

If no candidate is model-ictal, use a second bounded adaptation refinement around the best broad
state:

```text
tau_m          = 250, 500, 1000 ms
G_m/G_m0       = 0.5, 1.0, 1.5
eta_m          = G_m/tau_m
```

Do not combine both grids into a blind 36-cell sweep. The second grid is frozen only after the
first grid's morphology results identify one centre without using patient bridge scores.

## Task 5: target-informed selection

For up to three model-ictal full-dose candidates, run the predeclared selection seeds
`1811,1812,1813` with common random numbers and the complete recorder. Rank by:

1. model-ictal eligible proportion;
2. median `J_bridge`;
3. worst-seed `J_bridge`;
4. distance from the exact Z/M reference.

Write `selection_results.json` and freeze one winner. If no candidate is eligible on at least
2/3 seeds, stop without a work-point claim.

## Task 6: frozen confirmation

Run the frozen winner on `1821,1822,1823`. No parameter, window, contact or target change is
allowed. Confirmation reports eligibility, bridge components, readout timing and uncertainty.

The primary development result requires at least 2/3 confirmation seeds to remain model-ictal
eligible. Clinical similarity remains continuous; no post-hoc patient threshold is introduced.

## Task 7: selection-aware nulls and comparators

Repeat the complete candidate/window scoring under:

- exact-contact permutations;
- within-shaft permutations;
- gradient-preserving surrogate targets;
- static-axis amplitude-only and uniform-energy controls.

Report the null minimum `J_bridge`. Rescore 2% and 5% comparators using the same target and readout
algorithm, but do not allow them to become the primary Z/M-only winner.

## Task 8: figures and GIF

Create a separate candidate package without overwriting the accepted Fig.5 directory.

Main figure:

1. continuous readout plus recruitment/rate strip;
2. projected Z/M trajectory;
3. interictal event and state-defined early energy field;
4. patient 24-seizure target and per-contact model comparison;
5. candidate/null and confirmation summary.

GIF:

```text
Z/M state | 2D activity | continuous 15-contact readout
```

Render the frozen Z/M-only winner and existing 2%/5% comparators. Use identical scales and no
post-transition per-contact renormalization. Generate PNG/PDF/SVG/GIF, metadata and Chinese
`README.md`; visually inspect actual outputs.

## Task 9: closeout

Write `final_report.md` and update the archive only after numeric and visual QA. Report:

- exact fitted parameters and readout rule;
- model-ictal morphology and frequency;
- target components and selection-aware null;
- Z/M-only versus edge-dose comparator distinction;
- fit, selection and confirmation seed counts;
- development-only and non-generalization boundary;
- any stopped stages and why.

Mark the goal complete only when the frozen result, figures and report are all present or when the
predeclared stop rule yields a defensible negative result with complete evidence.

