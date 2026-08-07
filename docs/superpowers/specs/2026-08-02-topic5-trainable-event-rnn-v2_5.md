# Topic 5 trainable event RNN v2.5 — 34-patient frozen model-family contract

**Status:** v2.5.4 joint-refinement amendment; frozen before the final 34-patient rerun  
**Development set:** the six previously used patients  
**Cohort execution set:** all 34 patients; every patient receives a result or an explicit adequacy flag  
**Eligible events:** old `event_split == 0` only; the old heldout20 remains forbidden

## 1. Scope correction

V2.4 tested a PCA projection followed by one scalar decay and a ridge correction. It did
not test a trainable recurrent operator. Its negative result is retained only for that
leaky-filter family.

V2.5 asks whether a genuinely trainable event-level recurrent state improves prediction of
the future stable propagation repertoire beyond matched unordered, EWMA and distributed-lag
models:

\[
E_{e-L+1:e}\longrightarrow \mathcal D(E_{e+1:e+H}).
\]

One recurrent step is one complete interictal event. No within-event next-rank recurrence,
ictal target, SNN output, geometry, SOZ or A/B/axis label is used.

## 2. Cohort and split contract

- Primary history is `L=20` and primary horizon is `H=20`. This gives a valid within-source
  forecasting task for 34/34 patients.
- `L=40` and `L=80` are predeclared sensitivity analyses and never replace the primary result.
- Sources remain atomic. Sources are ordered by the first eligible event time and contiguous
  source groups are assigned to train/validation/test by the number of available prediction
  windows, not merely by the number of files.
- If at least three source groups with windows exist, cut points are selected without using
  targets to approximate 60/20/20 window counts while keeping at least one window per split.
- If only two source groups with windows exist, the earlier group is reserved for
  train/inner-validation and the later group is test. The earlier source is split
  chronologically at 70% to create event-disjoint inner train and validation sequences; the
  held-out later source is never used for checkpoint selection.
- Formal test targets are non-overlapping. Training windows may use denser anchors, but no
  target crosses a source or split boundary.
- Histories are always strictly earlier than targets. All indices are rechecked against
  `event_split == 0` event by event.

## 3. Representation and endpoint

Train-only masked-rank `K=2` templates define a stable forecasting coordinate; `K=2` is a
coordinate, not a universal biological two-mode claim. Each event token contains masked
normalized rank, participation and train-template mode indicator. Future targets contain
mode occupancy, mean contact rank and participation over `H` events.

The primary score is the train-variance-standardized mean of occupancy and rank MSE.
Participation is secondary. Hyperparameters and checkpoints are selected only by the primary
propagation score.

## 4. Matched non-recurrent ladder

The development screen compares:

- static repertoire;
- unordered history ridge;
- full-token EWMA ridge;
- event-descriptor EWMA ridge;
- four-bin distributed-lag ridge.

The globally best matched baseline family and its decay/regularization are frozen from the
six-patient median validation score. Every recurrent model predicts a residual correction on
top of this same baseline.

## 5. Trainable recurrent ladder

The recurrent family includes causal, unidirectional `tanh-RNN`, `GRU` and `LSTM` cells.
The six-patient development screen explicitly varies:

- cell type;
- hidden size and number of recurrent layers;
- event-step resolution (`1` event or contiguous `5`-event mean blocks);
- train-only input normalization (`none`, z-score, robust median/IQR);
- optional input/hidden LayerNorm;
- optimizer (`Adam`, `AdamW`, `RMSprop`);
- learning rate;
- batch size;
- weight decay and gradient clipping.

The search is a frozen staged candidate list, not a post-test architecture zoo. Candidate
selection uses the median six-patient validation propagation gain, then parameter count as a
tie-breaker. The selected profile and all hashes are frozen before running the complete
cohort. Final cohort models use three frozen seeds. Each patient uses its own pre-test
validation sources only to choose the checkpoint epoch; train and validation are not pooled
for an unchecked fixed-epoch refit.

### V2.5.1 refit repair

The first v2.5 execution incorrectly forced the median development best epoch (about one) up
to 15 and then refit on train+validation without validation monitoring. Because the recurrent
readout starts at exactly zero residual, this converted a near-baseline checkpoint into a
systematically overfit model. That run is an engineering-invalid refit diagnostic, not a
scientific RNN result. V2.5.1 keeps the globally selected model/training profile but evaluates
the best patient-specific validation checkpoint, which is the same estimator used during the
development screen. The repaired rerun is exploratory because the bad test run was observed.

### V2.5.2 selection repair

The v2.5.1 screen table correctly stored each patient's baseline-minus-RNN validation gain,
but the aggregation code sorted profiles by the median absolute RNN score. Absolute scores
mix model performance with between-patient forecasting difficulty and violated the frozen
selection rule. V2.5.2 selects training and architecture profiles by descending median
within-patient validation gain, then validation score and parameter count only as tie-breakers.
No test score enters this repair. V2.5/v2.5.1 remain archived engineering diagnostics.

### V2.5.3 exact nested checkpoint

The residual readout is initialized to zero, so the untrained recurrent branch is exactly the
matched baseline. This epoch `-1` state must participate in validation checkpoint selection.
Earlier code first evaluated after one full training epoch and therefore forced a harmful
correction even when every trained checkpoint was worse than baseline. V2.5.3 initializes the
best checkpoint and validation score from the exact baseline state. A patient retains a
trained recurrent correction only when pre-test validation improves over that nested null.

### V2.5.4 bounded joint refinement

Because optimizer settings were first screened with a reference GRU and LayerNorm was added in
the later architecture stage, V2.5.4 performs one bounded joint validation-only refinement
around the selected combined profile. It varies learning rate, batch size, gradient clipping,
normalization, optimizer, hidden size, cell, layers and 5-event steps one factor at a time.
Selection again uses median within-patient validation gain. This closes the interaction gap;
no further hyperparameter expansion is allowed after the v2.5.4 cohort run.

## 6. Training adequacy

Every run saves:

- full train and validation loss curves;
- validation propagation curve;
- best epoch and checkpoint;
- gradient-norm and clipping summaries;
- finite-loss/finite-parameter checks;
- parameter count and runtime;
- selected normalization statistics.

A synthetic order task with identical unordered event composition must be solvable by at
least one screened recurrent profile while the unordered baseline remains at chance. Failure
blocks scientific interpretation and triggers an implementation/training audit.

## 7. Chronology controls

Only after a trainable recurrent profile is frozen are chronology controls evaluated:

1. source-coherent event-block shuffle followed by complete window reconstruction;
2. safe source-level circular history-target pairing with synchronized values and indices;
3. true chronology using the identical profile and training budget.

The RNN claim requires improvement over the strongest matched baseline and both coherent null
families. A model beating the baseline but not the nulls is a nonlinear history compressor,
not chronology-specific recurrent state.

## 8. Cohort reporting and stopping rules

- All 34 patients appear in the denominator table.
- The six development patients are descriptive and do not enter the primary P value.
- The other 28 define the extension inference; low-window patients remain present with an
  explicit support grade and sensitivity analysis rather than being silently deleted.
- Seeds are merged within patient before patient-level statistics.
- Report median delta, patient bootstrap CI, sign count, Wilcoxon and sign test.

Allowed outcomes:

- matched baseline sufficient: stable/recency repertoire prediction, no RNN claim;
- recurrent model wins but nulls do not: nonlinear compression only;
- recurrent model and coherent-null gates pass: low-dimensional chronology-sensitive
  event-history state;
- participation-only gain: recruitment-topography prediction, not propagation-state change.

Prediction does not establish activity-dependent plasticity, evolving biological connectivity
or causal network shaping.
