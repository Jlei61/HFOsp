# Topic 5 stateful event-sequence RNN v2.6

**Status:** completed and frozen; accepted as the event-history state-tracking precursor  
**Cohort:** all 34 patients, exploratory nested chronological validation  
**Time step:** one complete interictal event  
**Forbidden inputs:** old heldout20, A/B or axis labels, geometry, SOZ, ictal data, SNN output

> **Closure adjudication (2026-08-03).** V2.6 establishes that a stable patient-specific
> repertoire has a measurable short-range, cross-event state: resetting state after every
> event worsens prediction, and the trained state mainly integrates the preceding 10--20
> events. It does not establish that an event changes the underlying pathological network.
> The trained recurrent family beats a fixed repertoire but not a matched leaky recency
> observer. This narrows the detectable dynamics to a simple low-dimensional leaky state; it
> does not terminate the evolving-state question. V2.6 is therefore accepted as a
> **state-tracking precursor**, not as a state-shaping or graph-identification result.

## 1. Scientific question

The model asks whether the true chronological sequence of complete interictal events carries a
patient-specific recurrent state that predicts the distribution of propagation events that
follows:

\[
h_e=F_\theta(h_{e-1},x_e),
\qquad
\widehat D_e=G_\theta(h_e)
\approx D(E_{e+1:e+H}).
\]

Stable split-half and odd/even propagation repertoire is an empirical premise. V2.6 does not
retest its existence. It tests whether expression of that repertoire has a learnable state
across events.

## 2. Correction relative to v2.5

V2.5 encoded fixed `L`-event windows independently and reset hidden state for every sample.
It therefore tested a window-reset recurrent encoder, not a continuously evolving event
state. Its result remains a narrow baseline result.

V2.6 runs once from the beginning to the end of each source recording. Hidden state resets
only at a true source/session boundary. Truncated backpropagation limits gradient length but
does not reset the forward state between chunks.

## 3. Observation and target

Train-only masked-rank templates define a `K=2` forecasting coordinate, not a universal
two-biological-mode claim. Event token `x_e` contains:

- masked normalized contact rank;
- contact participation;
- train-template mode indicator.

After observing event `e`, the target is the repertoire descriptor of the next `H=20`
events: occupancy, contact mean rank and participation. Predictions begin after a warm-up of
20 events.

Dense causal anchors are used for training **and for validation model selection / early
stopping**, because a spaced validation grid is too sparse to rank profiles stably on the
smaller patients. Formal anchors spaced by `H` are built for validation and test, are audited
for non-overlap on both splits, and define the **primary test endpoint**. Selection criterion
and primary endpoint therefore use different anchor grids; that difference is one declared
source of the formal-versus-dense disagreement and must be reported alongside the endpoint,
never used to pick whichever grid looks better after the fact.

## 4. Stateful recurrent family

The scientific model family is recurrent throughout. It includes trainable unidirectional:

- tanh RNN;
- GRU;
- LSTM.

These are parameterizations of the same event-state hypothesis, not separate scientific
branches. The screen varies hidden dimension, layers, train-only normalization, input/hidden
LayerNorm, optimizer, learning rate, TBPTT length and the number of TBPTT chunks accumulated
per optimizer update.

The primary head directly predicts future repertoire. It is not restricted to a residual of
an EWMA predictor. Readout bias is initialized to the train future-repertoire mean and readout
weights are zero, so epoch `-1` is a reproducible static state but not the final scientific
model.

## 5. Training contract

- State is carried continuously across consecutive TBPTT chunks within a source and detached
  only for gradient truncation.
- State is reset at source boundaries and split boundaries. No history is ever carried across
  sources, so the longest testable history is one source.
- Source order within each patient remains chronological; training may shuffle whole sources
  between epochs but never events within a true source.
- Checkpoints save the best trained validation epoch and a nested static-fallback state, the
  latter being `argmin` over the epoch `-1` static initialization and every trained epoch. It
  equals the raw epoch `-1` model only when no trained epoch beat it. The static comparator of
  §7 is built directly from the train future-repertoire mean and does not depend on this
  checkpoint.
- **Known v2.6 deviation, not repaired in this frozen run.** Early stopping counts staleness
  against the epoch `-1` static score rather than against the trained model's own best. A
  profile that has not beaten its own initialization within `minimum_epochs` therefore stops
  at the minimum budget instead of the full screen budget: 8 of 102 final runs and 79 of 748
  screen fits took that path. The bias is conservative for the RNN-versus-static claim and
  non-conservative for the RNN-versus-EWMA negative, so the latter carries a leave-those-
  patients-out check in the acceptance record. Any successor version must count staleness
  against the trained checkpoint and keep the epoch `-1` state as a reporting fallback only.
- Primary RNN evaluation uses the best **trained** validation checkpoint. A nested
  static-fallback score is secondary and cannot convert training failure into a scientific
  success or hide a harmful trained model.
- All losses, gradient norms, clipping fractions, best epochs, state norms, parameter counts
  and runtime are saved.

## 6. Parameter selection

All 34 patients participate in exploratory nested validation; no patient test source enters
selection.

For each patient:

1. architecture stage compares RNN/GRU/LSTM at hidden 16 and 32;
2. training stage refines the best validation architecture over learning rate, optimizer,
   normalization, TBPTT length, update batch and two-layer recurrence;
3. if a leading profile reaches its best checkpoint at epoch 35 or later under the 40-epoch
   screen, the top three unique validation profiles are re-trained for at most 100 epochs;
   the selected training budget is frozen together with the selected profile;
4. the patient-specific best validation profile is frozen;
5. three fixed seeds are evaluated on the untouched patient test source.

This patient-specific selection is necessary because event count, contact count and temporal
scale vary substantially across patients. Cohort inference folds one pre-test selected result
per patient.

## 7. Evaluation

Primary endpoint:

\[
\frac{1}{2}
\left(
\mathrm{MSE}_{occupancy}/V_{occupancy}
+
\mathrm{MSE}_{rank}/V_{rank}
\right).
\]

Participation is secondary. Report trained RNN versus:

- static train future-repertoire mean;
- one fixed continuous descriptor-EWMA ridge comparator (`decay=0.95`, `alpha=100`).

The comparator is an evaluation control, not an alternative modeling branch.

## 8. Chronology controls

After patient-specific profiles are selected from true validation chronology, the identical
profiles and budgets are run on:

- source-coherent event-block shuffle, block length `H`;
- source-level time reversal, with all future targets rebuilt from the reversed event
  sequence.

All surrogate sequences are rebuilt before stateful rollout. A chronology claim requires the
trained true-sequence model to beat the comparator and both coherent nulls. Circular target
re-pairing is not used: for a hidden state accumulated from the start of a source, circular
donors either leak already observed events into the target or, if the entire source is rotated,
preserve almost all chronological adjacency and do not form a meaningful null.

Two scope clauses bind every statement made from these nulls:

- Block length equals `H` equals the formal anchor spacing, so each formal target window is
  one intact block that the shuffle relocates as a unit. The null can only address order
  *between* windows; ordering *inside* a window is untouched and untested here.
- Both tails are reported. The registered direction is "true beats surrogate"; a surrogate
  that beats the truth means the two arms are not exchangeable — both arms' recurrent and
  comparator legs moved — and licenses no claim about the direction of real time.

The frozen null results come from the dedicated source-level runners
`scripts/run_topic5_stateful_event_rnn_v2_6_block_null.py` and
`scripts/run_topic5_stateful_event_rnn_v2_6_reversal_null.py`. Their current SHA-256 values
match the hashes stored in the completed null state files. They do not call the legacy
row-wise `shuffled_histories()` or target-only `circularly_shift_targets()` helpers retained
under the older stable-repertoire module.

## 9. Required calibration

1. A synthetic task with identical unordered event composition but long-range order must be
   learned by at least two recurrent cell types.
2. A state-persistence regression test must show that chunked rollout equals unchunked rollout
   when dropout is zero.
3. Resetting at every TBPTT chunk must fail the long-memory synthetic task or perform
   materially worse than state carry.
4. All 34 patients receive a test result or explicit failure record.

## 10. Interpretation boundary

The pre-specified strongest success would have supported a chronology-sensitive event-history
state. The accepted result is narrower: state tracking and short-range recency dependence.
Neither result by itself establishes biological synaptic plasticity or an evolving anatomical
graph. Failure bounds the calibrated stateful recurrent family; it does not negate the stable
propagation repertoire.

Three bounds travel with every reported result:

- **Horizon of the memory claim.** State resets at every source boundary, so "long-range" here
  means within one recording source. Taking each patient's median source length and then the
  median across the 34 patients gives 294 events. Nothing in this design can speak to
  processes spanning sources or days.
- **Grain of the chronology claim.** See §8: only between-window order is perturbed.
- **Support heterogeneity.** Formal test windows range from 1 to 1,096 per patient. Every
  cohort statement is reported both on all patients and stratified by minimum window count;
  a direction that survives only in the low-support tail is not a cohort result.

## 11. Observer versus updater boundary

V2.6 is an observer:

\[
E_{\le e}\longrightarrow \widehat z_e\longrightarrow D(E_{e+1:e+H}).
\]

The same unobserved slow state may generate both recent and future events. Consequently,
predictive history dependence does not identify the causal arrow
`event -> network-state change`. A successor model must estimate the pre-event state from
past events only, define the current event's innovation relative to that state, and test
whether this innovation predicts a later state residual beyond autonomous drift and the
same leaky observer. The physical-state transition and the observer's measurement update
must remain distinct in both equations and code.

Successor work is split deliberately:

1. V2.7 is a repair-only parallel rerun that fixes the epoch-minus-one early-stopping bias
   without changing the scientific model or parameter grid.
2. V3.0 is a new event-innovation low-rank state-space contract. It tests state-update
   association directly and is not a capacity extension of the GRU family.
