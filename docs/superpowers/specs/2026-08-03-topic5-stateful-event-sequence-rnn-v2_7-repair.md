# Topic 5 stateful event-sequence RNN v2.7 repair contract

**Status:** complete; `DERIVED_ACCEPTANCE_COMPLETE`  
**Scientific object:** unchanged event-history state tracking  
**Cohort:** the same 34 exploratory patients  
**Output root:** `results/topic5_stateful_event_sequence_rnn/v2_7/`

## 1. Purpose

V2.7 exists for one reason: remove the v2.6 early-stopping bias while preserving every
scientific and data choice. It must not become another architecture search or a new
state-shaping model.

The frozen question remains:

\[
E_{\le e}\longrightarrow \widehat D(E_{e+1:e+20}).
\]

V2.7 may update the final state-tracking effect sizes. It cannot establish event-driven
network change, graph recovery, plasticity or causal shaping.

## 2. Frozen from v2.6

The following must be byte-identical in meaning and, where practical, copied unchanged:

- dataset and source mappings;
- chronological source split and forbidden-input flags;
- train-only masked rank/template construction;
- event token, future descriptor and `H=20` target;
- RNN/GRU/LSTM profile grid, learning rates, optimizers, normalization, hidden sizes,
  layers, TBPTT lengths, update chunks and three final seeds;
- dense training anchors, dense validation selection and formal non-overlapping primary test
  anchors;
- static and fixed EWMA comparators;
- propagation-primary and participation-secondary scores;
- patient-first aggregation and support-stratified reporting.

No test result may change this grid or select an additional profile.

## 3. The only model-code change

V2.6 initializes early-stopping staleness against the epoch-minus-one static initialization.
V2.7 must maintain two independent records:

1. `best_trained_score/state/epoch`, initialized only after the first trained epoch;
2. `best_nested_score/state/epoch`, allowed to include epoch minus one for reporting.

Patience is updated only from `best_trained_score`. Epoch minus one never increments trained
staleness and never causes a run to stop at `minimum_epochs`.

The static initialization remains a reproducible comparator and nested fallback; it is not a
training checkpoint.

## 4. Required regression tests

1. A profile that remains worse than epoch minus one for the first eight epochs still trains
   until trained-checkpoint patience is exhausted.
2. A profile improving among trained epochs resets patience even if it has not yet beaten the
   static initialization.
3. Nested fallback may select epoch minus one; trained checkpoint may not.
4. All existing target, state-carry, RNN/GRU/LSTM chunk-parity and long-memory tests remain
   green.
5. V2.6 source/config/input hashes are recorded as parent provenance; V2.7 writes new hashes
   and never overwrites V2.6.

## 5. Execution and evaluation

The full validation screen is rerun because the bug affected 79/748 screen fits and could
alter patient-specific profile selection. The epoch-boundary extension is then rerun under
the repaired rule. Only after all 34 profiles and budgets are frozen are three-seed test,
state reset, memory curve, dense sensitivity, H40 and the chronology controls rerun.

Primary comparisons:

- trained recurrent model minus static repertoire;
- trained recurrent model minus fixed EWMA;
- true chronology minus source-coherent nulls.

The acceptance record must report all 34 patients and the predeclared support strata
`>=10`, `>=20` and `>=50` formal windows. Strata describe precision; they do not select a
post-hoc positive subgroup.

## 6. Acceptance outcomes

V2.7 is complete when:

- 34/34 patients have a validation selection or explicit failure record;
- every eligible patient has three finite test runs;
- no trained run stops because epoch minus one remained better;
- source/test leakage and forbidden-input checks pass;
- paired V2.6-to-V2.7 changes and every primary comparison are written to a derived
  acceptance state;
- v2.6 artifacts and hashes remain unchanged.

Possible scientific wording remains bounded:

- if recurrent minus static remains favorable: short-range state tracking is retained;
- if recurrent does not exceed EWMA: the minimal supported dynamics remain leaky recency;
- if precise effect sizes change: report the repaired values and retire v2.6 numbers from
  manuscript use;
- no V2.7 outcome alone licenses state shaping.

## 7. Relationship to V3.0

V2.7 repairs the training bias in the manuscript-facing observer estimate. Until that repair
is complete, the bounded wording is: **V2.6 supports short-range state tracking, with a
quantified training bias pending repair.** V3.0 asks whether valid future-blind event
innovations have time-directed and cumulative associations with later propagation-state
changes. The separate V3.1 contract asks whether a matched generative transition requires an
event input beyond observer correction.

V2.7 is not a scientific Gate for V3.0. V3.0 synthetic, train/validation and frozen human
analysis may proceed independently once the V3.0 design checklist is complete. V3.1 synthetic
implementation may also proceed, but its human test follows the frozen V3.0 handoff. Neither
successor may be implemented by adding capacity to V2.7.

## 8. Closure record

The 34-patient validation screen, boundary audit, 102 formal runs and all frozen controls
completed under the repair-only namespace. The trained RNN retained the static-repertoire
gain but did not exceed EWMA or coherent chronology nulls. V2.7 and V2.6 primary patient
effects were exactly identical, showing that the repair corrected patience provenance without
changing the selected best trained checkpoints. The final allowed object is within-recording
short-range state tracking; V2.7 is closed and must not be tuned further.
