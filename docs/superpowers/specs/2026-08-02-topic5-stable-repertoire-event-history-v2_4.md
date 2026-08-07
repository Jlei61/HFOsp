# Topic 5 stable-repertoire event-history v2.4 — frozen repair and extension contract

**Status:** frozen before any v2.4 patient result is read  
**Development cohort:** the six patients used in v2.3.1  
**Locked extension cohort:** the other 28 patients in `dataset_v0_4`  
**Primary horizon:** `H=20`; `H=40` is sensitivity only  
**Eligible events:** old `event_split == 0` only

## 1. Scientific question

The stable patient-specific propagation repertoire is an empirical premise established by
masked split-half / odd-even analyses and checked again with train-only templates.  V2.4 asks:

\[
E_{e-79:e}\longrightarrow \mathcal D(E_{e+1:e+H}),
\]

and specifically whether the temporal organization of complete past events predicts the
future repertoire beyond a stable template, an unordered history estimate, simple recency
weighting and distributed-lag summaries.

The model is repertoire-conditioned forecasting.  It does not discover a biological graph,
test plasticity, predict seizure activity or establish causal shaping.

## 2. Frozen event and target representation

- One sequence step is one complete interictal event.
- Train-only `K=2` masked-rank templates define a common forecasting coordinate.  `K=2`
  is a coordinate inherited from the stable-repertoire analysis, not a universal biological
  claim of two exactly opposed modes.
- One event token contains masked normalized rank, participation and train-template mode.
- The future target contains mode occupancy, contact mean normalized rank and contact
  participation over `H` non-overlapping events.
- No A/B name, pathological axis, geometry, SOZ, ictal data or SNN output enters training.

## 3. Data split and confirmation boundary

Within each patient, complete source records are ordered and assigned 60/20/20 to
train/validation/test.  Every final event index is intersected again with `event_split == 0`.
State resets at source boundaries because the current source blocks are canonical recording
units and cross-block biological continuity is not established.

The six development patients may select implementation details and frozen hyperparameter
grids.  They never enter the primary cohort P value.  If the development release gates pass,
the identical code/config hashes are run once on the remaining 28 patients.  Test partitions
never select a model, patient, horizon, template grade or endpoint.

## 4. Frozen model ladder

All trainable baselines select hyperparameters on validation with the primary propagation
score and are refit on train+validation before the test partition is evaluated.

| ID | Model | Role |
| --- | --- | --- |
| B0 | static repertoire | stable patient scaffold |
| B1 | last-`H` ridge | recent equal-count history |
| B2 | unordered-80 ridge | more precise long-history composition |
| B3a | first-`H` ridge | distant equal-count control |
| B3b | random-`H` ridge | count-matched history control, repeated frozen seeds |
| B4a | full-token EWMA ridge | full-dimensional recency filter |
| B4b | event-descriptor EWMA ridge | simple descriptor recency filter |
| B5 | four-bin distributed-lag ridge | non-recurrent temporal profile |
| B6 | time/IEI nuisance ridge | duration, IEI, event rate and source progress |
| R1 | low-dimensional leaky event-history state | PCA state plus ridge |

GRU is excluded.  R1 is called a low-dimensional leaky history state, not a general RNN.
Low dimensionality is supported only if R1 exceeds the validation-selected B4/B5 comparator.

## 5. Coherent chronology nulls

Nulls are constructed once per source sequence and all overlapping windows are rebuilt from
that coherent pseudo-sequence.

### N1 source-level block shuffle

Events are partitioned into contiguous blocks of 20 events.  Blocks are permuted once per
source while event order inside each block is retained.  Window rows are never shuffled
independently.

### N2 safe source-level circular pairing

For every source, complete contiguous target windows are shifted by a frozen non-zero number
of target-window steps.  Target values, raw event indices, start/stop positions and times move
together.  A paired row is retained only when shifted target and observed history are disjoint
and separated by at least one full target horizon.  Multiple frozen safe shifts are run.

Every null artifact stores the source permutation/shift, original row, donor row and raw event
indices.  Contract checks use the shifted indices rather than the original metadata.

## 6. Scores and reliability

Family MSE is standardized by family variance estimated from training targets only.

- **Primary propagation score:** mean of standardized occupancy and rank MSE.
- **Secondary recruitment score:** standardized participation MSE.
- **Tertiary repertoire score:** mean of all three standardized family MSE values.

Future-window reliability is reported twice:

1. raw reliability, which includes stable contact main effects;
2. dynamic reliability after subtracting train-only family/contact means.

Neither reliability result is used to delete a patient after test prediction is seen.

## 7. Time-scale and source audit

For every patient and split report the number of source records and independent target
windows, the elapsed duration of 80-event histories and 20/40-event targets, median IEI,
event rate, source progress and cross-source gaps.  Event index is not treated as a common
biological time unit across patients.

## 8. Development release gates

The 28-patient locked extension is released only if all conditions hold in the six-patient
rerun:

1. all final indices are train80-only; source splits are disjoint; histories/targets and all
   shifted null pairs pass provenance and non-overlap checks;
2. every null is source-coherent and differs from the true sequence;
3. median test delta `R1 - validation-selected(B4a,B4b,B5)` is below zero for the primary
   propagation score;
4. median chronology increment is positive relative to both N1 and N2, with lower score
   meaning better prediction;
5. the propagation score does not show a systematic degradation hidden by participation;
6. code, config, template and input hashes are frozen before the extension run.

Failure invokes the prewritten stopping rule; the extension is not used to rescue the model.

## 9. Locked extension inference

The statistical unit is the patient.  Seeds/null replicates are first merged within patient.
For the remaining 28 patients report median delta, patient bootstrap 95% CI, positive count,
one-sided Wilcoxon signed-rank and sign test.  Epilepsiae and Yuquan directions are also shown
separately.  H=20 alone defines the primary claim; H=40 is sensitivity.

The primary claim requires both:

\[
R1 < \operatorname{best}_{validation}\{B4a,B4b,B5\}
\]

and a larger R1 improvement under true chronology than under both coherent null families.

## 10. Interpretation guardrails

- B2 beating B1 alone means that more events estimate the stable repertoire more precisely.
- Last-`H` beating random/first-`H` supports recency, not recurrent computation.
- B4/B5 matching R1 closes the analysis as a recency/distributed-lag effect.
- R1 beating B4/B5 but not coherent nulls supports compression, not chronology specificity.
- A participation-only effect is recruitment-topography prediction, not propagation-state
  prediction.
- A development-only or minority-patient result is heterogeneity, not a cohort mechanism.
- Prediction never establishes activity-dependent shaping or plasticity.

