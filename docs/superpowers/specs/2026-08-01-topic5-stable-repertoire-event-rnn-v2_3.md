# Topic 5 stable-repertoire event RNN v2.3 — frozen development contract

**Status:** frozen before the six-patient pilot  
**Scope:** exploratory development inside the previously defined `train80`; the old `heldout20` is not used  
**Scientific premise:** patient-specific propagation templates are reproducible across chronological split-half and odd/even source blocks (`23/30 strong`, `7/30 moderate`, `0 weak`). This premise authorizes an event-indexed model; it is not an evolving-field stopping gate.

## 1. Question and time axis

One recurrent step is one complete interictal event. The model asks whether the ordered history of complete events predicts the distribution of propagation patterns in a future event window:

\[
E_{e-L+1:e}\longrightarrow \mathcal D(E_{e+1:e+H}).
\]

It does not use within-event rank as the long-term recurrent clock and does not predict the next rank set. The safe positive claim is history-dependent modulation of a stable propagation repertoire. `network shaping`, plasticity and causal change are not implied by prediction alone.

## 2. Frozen representation

- Primary pilot: the six patients used in v0.1/v2.1 development.
- Input source: `dataset_v0_4` plus the audited event-to-source mapping from v2.2.
- Only `event_split == 0` is eligible. The old `heldout20` remains untouched.
- Selecting a source recording never re-admits ineligible events from that source: every final train/validation/test index is intersected with `event_split == 0` and asserted event by event.
- Within each patient, source recordings are ordered by their first absolute event time and split source-wise into 60% train, 20% validation and 20% exploratory test. No source appears in two partitions.
- Recurrent state resets at every source-record boundary. Cross-source IEI is never interpreted biologically.
- Stable modes are re-fit on train events only with masked rank features and fixed `K=2` for this pilot. Full-data PR-2 labels, A/B names, axis labels, geometry, SOZ, ictal data and SNN output are forbidden inputs.
- An event token contains its masked normalized-rank vector, participation mask and train-template mode indicator. Non-participating ranks are imputed using train-only statistics and remain distinguishable through the participation mask.

## 3. Future-window task

Primary horizon `H=20` events and history length `L=80` events are frozen. Histories and targets must remain within one source recording. Prediction anchors are spaced by `H`, so formal evaluation targets do not overlap.

The target is a future repertoire descriptor with three equally weighted families:

1. train-template mode occupancy;
2. contact mean normalized rank with train-only shrinkage;
3. contact participation probability.

The primary score is the mean of the three family MSE values. Mode Brier/MSE is the primary interpretable endpoint; rank and participation prevent a mode-only lookup from exhausting the event structure. `H=40` is a predeclared sensitivity, not a replacement chosen after seeing test results.

Future-window reliability is reported on validation windows by repeatedly splitting each
target window into two random halves. For occupancy, rank and participation separately,
report split-half Spearman correlation and the fraction of between-window variance left
after subtracting finite-event sampling noise. This is a measurement diagnostic; because
`H=20` was frozen before model scoring, it is not used to delete patients post hoc.

## 4. Ordered model ladder

All models use the same split, train-only template encoder, prediction anchors, targets and family-balanced score.

| ID | Model | Information available |
| --- | --- | --- |
| R0 | static repertoire | train target mean only |
| R1 | recent-window ridge | unordered descriptor of the most recent `H` events |
| R1-L | long-history ridge | unordered descriptor of all `L` history events; capacity-matched chronology control |
| R2 | discrete switching | train-only first-order transition between the two stable modes, propagated for `H` future events |
| R3 | linear event-state | each of the last `L` full event tokens updates a stable leaky linear state |
| R4 | GRU event-state | same event tokens, history, decoder and loss as R3; only the recurrent cell changes |

**Fair-increment amendment (v2.3.1):** direct R3/R4 additionally test whether a very
small state can replace the recent-window baseline, which confounds history value with
compression loss. The confirmatory ladder therefore nests recurrent state on R1:

\[
\widehat D_{future}=\widehat D_{R1-L(unordered\ history)}+Delta_{recurrent}(E_{e-L+1:e}).
\]

The long-history baseline uses exactly the same `L` events as the recurrent correction,
so the increment isolates ordering rather than access to more past events. The same
nested factorization is used for ordered, shuffled and circular controls. Direct
R3/R4 remain archived capacity diagnostics but cannot by themselves reject history.

R3 must be implemented and checked before R4 is run. A GRU gain is not interpreted until R0–R3 are valid.

## 5. Chronology controls

For R3/R4, two controls are mandatory:

- within-history order shuffle: same events and same future target, order destroyed;
- circular input-target shift within source: local marginals retained, correct history-to-future pairing destroyed.

The ordered-history increment is performance of real chronology minus the best order-destroyed control. A model beating static but not the controls only learns marginal repertoire composition.

## 6. Gates and allowed conclusions

- **C0 engineering:** train-only templates, disjoint sources, boundary resets, non-overlapping targets and untouched old heldout are verified by tests and artifacts.
- **C1 stable repertoire read-back:** the train-only `K=2` representation remains assignable in validation/test and does not collapse to a negligible component.
- **C2 predictive increment:** at the patient level, R3 or R4 improves the composite future-window score over the strongest R0–R2 baseline.
- **C3 chronology specificity:** the same model improves over both order-destroyed controls.
- **C4 recurrent necessity:** R4 must improve over R3 before nonlinear recurrent computation is claimed. If R3 is sufficient, the conclusion is a low-dimensional linear history state.

Outcomes are patient-first; seeds are merged within patient. Six-patient results are a pilot and cannot be called an independent cohort confirmation.

## 7. Explicit exclusions

This contract does not test event-driven graph plasticity, a changing contact-by-contact graph, SNN identifiability, ictal transfer, or biological learning. The v2.2 aggregate-block observability result remains an archived analysis of blockwise marginal descriptors; it is not an entry condition for this event-indexed RNN.
