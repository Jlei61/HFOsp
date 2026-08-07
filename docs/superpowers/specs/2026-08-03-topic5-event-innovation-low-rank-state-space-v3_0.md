# Topic 5 event-innovation impulse response and cumulative state displacement v3.0

- **Status:** revised association contract; implementation authorized; human test locked until
  Section 14 is frozen
- **Time step:** one complete interictal event
- **Primary scientific object:** a patient-specific propagation rank/precedence state evolving
  across events
- **Primary question:** whether valid event innovations are followed by time-directed and
  cumulative changes in the observable propagation state
- **Not a goal:** within-event next-rank prediction, proving that a GRU is necessary, or using
  SNN as an RNN Gate
- **Output root:** `results/topic5_event_innovation_impulse_response/v3_0/`

## 1. Frozen scientific question

Existing split-half and odd/even results establish a stable patient-specific propagation
repertoire. V2.6/V2.7 ask whether recent complete events help track the current expression of
that repertoire. V3.0 asks the distinct downstream question:

> After the current propagation state has been estimated using past events only, is the
> unexpected propagation content of an interictal event associated with a subsequent,
> time-directed change in that state; and do repeated aligned innovations produce a
> cumulative state displacement?

The full tested chain is:

\[
\text{stable backbone}
\rightarrow
\text{valid future-blind event innovation}
\rightarrow
\text{multi-horizon impulse response}
\rightarrow
\text{repeated-innovation accumulation}.
\]

A single-event association is necessary but insufficient for the original shaping
hypothesis. V3.0 remains an association/local-projection study. A separate V3.1 contract
performs the matched generative observer-versus-transition identification; independent
replication is still required before shaping language.

## 2. Separation from earlier RNN branches

This contract does not return to event-internal rank-step recurrence. One model step is one
complete event. It also does not repeat V2.2's unstratified block-delta screen or V2.6's
forecasting-only question.

- V2.6/V2.7: recent events improve estimation of a current slow repertoire state.
- V3.0 local response: an event innovation predicts a later state residual.
- V3.0 accumulation: repeated aligned innovations predict a larger and persistent state
  displacement.
- V3.1 generative identification: the event term must enter the latent transition, not only
  a more flexible observation/filtering update.

V2.7 repairs a known training bias in the observer precursor. It is not a scientific Gate
for synthetic, train/validation or human V3.0 work, and V3.0 must not be implemented by
adding capacity to V2.7.

## 3. Data, chronology and forbidden inputs

The analysis uses the canonical train80 event objects and preserves complete chronology:

- masked normalized contact rank and rank-set ties;
- participation mask;
- source ID, event index and absolute timestamp;
- canonical contact order and source provenance.

The old heldout20, externally supplied A/B or pathological-axis labels, geometry, SOZ,
ictal data, SNN output and patient outcome are forbidden. A train-only template occupancy
coordinate may be retained as a secondary readout; it is not a universal two-mode biological
label and does not define eligibility.

IEI, recent event rate, source progress, recording day and available sleep/medication
metadata are nuisance or timing sensitivities, not the primary propagation state.

Before model fitting, every source must be assigned to a **continuity unit** using recording
metadata rather than event density:

- genuinely independent sessions reset state;
- artificial file splits with compatible montage and verified continuous absolute time are
  joined, and the real inter-file gap is retained;
- unresolved source relationships remain separate and restrict interpretation to
  within-source dynamics.

The source audit must save start/stop time, gap, montage compatibility, continuity decision
and reason. Cross-source or cross-day shaping language is forbidden unless continuity is
explicitly established.

## 4. Observable propagation state

For event `e`, let `q_{e,c}` be the normalized rank of participating contact `c`; absent
contacts remain masked and tied contacts remain tied. The stable train-only rank backbone is
`mu_0`. Dynamic state is defined in contact rank coordinates:

\[
\mu_e=\mu_0+Ls_e,
\qquad
L\in\mathbb R^{C\times K}.
\]

The corresponding observable pairwise precedence field is

\[
G_{e,ij}
=
P(i\prec j\mid s_e)
=
\sigma(\mu_{e,j}-\mu_{e,i}).
\]

The primary state and score use rank and pairwise precedence. Train-only template occupancy
is secondary; contact participation is tertiary. This prevents stable recruitment topology
or a fixed `K=2` coordinate from carrying the propagation claim.

Low rank refers only to temporal variation around `mu_0`. Validation selection uses
`K in {1,2,3,4}`. `K=6`, `K=8` and the maximum estimable/full-rank model are mandatory
diagnostics. A low-dimensional claim requires held-out performance to saturate before these
higher-dimensional diagnostics, not merely for `K=4` to win inside a truncated grid.

Raw latent coordinates, `A`, `B`, hidden PCs and recurrent weights are not interpreted as a
contact graph. The coordinate-invariant scientific outputs are `mu_e`, `G_e` and their
observable impulse responses.

## 5. State estimation and family-specific innovations

All state estimates and innovations are future blind. For an anchor event `e`, the pre-event
state is estimated from events ending at `e-1`. A disjoint future rank field is estimated from
events `e+1:e+h`, with frozen horizons `h in {5,10,20,40}` and `h=20` the single-event
primary endpoint.

The observation model and its preprocessing are fit on training continuity units only.
Training innovations are constructed by source-level or blocked chronological cross-fitting;
random event-level folds are forbidden.

Innovations are family specific:

1. **Rank/precedence innovation, primary.** Rank residuals are defined only for participating
   contacts; pairwise residuals use only co-participating, non-tied pairs and reliability
   weights.
2. **Mode innovation, secondary.** The train-only mode readout uses a multinomial residual.
3. **Participation innovation, tertiary.** Recruitment uses a Bernoulli Pearson residual.

In compact notation,

\[
\nu_e=R^{-1/2}M_e\{y_e-E(y_e\mid\mathcal H_{e-1})\},
\]

where `M_e` is the family-specific validity/tie mask and `R` is estimated from training data.

### Innovation validity is mandatory

Before `nu_e` can be called an innovation, held-out residual predictability is tested against
a frozen observer ladder containing pre-20/pre-40/pre-80 summaries, four 20-event lag bins,
source progress, IEI and recent event rate. A residual family is innovation-valid only when
these past-only predictors do not exceed the 95th percentile of its calibrated blocked null
in cross-fitted data. Otherwise it is reported as **unresolved-state residual**, the observer
is expanded according to the frozen ladder, and no innovation-based claim is made for that
family.

The ladder is applied sequentially, not selected once and then declared valid. At each rung,
dimension and ridge strength are selected on validation within that rung; its blocked
training cross-fit residual is then tested against the **complete available past-only
feature ladder**. The first rung passing this stronger test is retained. If all rungs fail,
the family remains unresolved. Whiteness is evaluated in fixed contact-rank coordinates and
permuted coherently within continuity sequences; fold-specific latent coordinates and groups
containing fewer than two null blocks cannot certify an innovation.

## 6. Goal 1 — state and innovation measurement reliability

Before response testing, quantify:

- raw and contact-residualized rank-field reliability;
- pairwise precedence reliability with ties retained;
- mode and participation reliability as secondary families;
- innovation calibration and whiteness;
- source continuity and the real-time span represented by each event window.

This Goal does not decide whether the network is shaping. It establishes that the state
change and innovation to be related are measurable beyond finite-window noise and stable
contact main effects.

## 7. Goal 2 — multi-horizon local impulse response

For each valid event and horizon, estimate an autonomous state forecast from the pre-event
state and define the disjoint future residual:

\[
\Delta s_{e,h}
=
s^+_{e,h}-\widehat A_hs^-_e.
\]

The local-projection model is

\[
\Delta s_{e,h}
=
B_h\nu_e+\Gamma_h c_e+\epsilon_{e,h},
\qquad h\in\{5,10,20,40\},
\]

where `c_e` contains frozen time/event-rate nuisance covariates. The observable impulse is

\[
\mathcal J_h=L B_h,
\]

and is reported as the predicted change in `mu` and `G`, not as a raw latent matrix.

Required contrasts are:

- future versus duration-matched past response;
- true innovation pairing versus state- and source-progress-matched donor innovations;
- source-coherent block permutation at sizes `1,2,5,10,20,40`;
- safe lag shifts at `2h,3h,4h` with no wrap-around;
- nuisance-adjusted versus unadjusted response.

This Goal establishes at most an **event-innovation predictive association**, even when it is
future directed and null specific. It cannot by itself distinguish a true state transition
from improved state estimation.

## 8. Goal 3 — repeated-innovation accumulation

For `m in {5,10,20,40}`, construct the past-only cumulative innovation ending at event `e`:

\[
c_e^{(m)}
=
\sum_{j=0}^{m-1}w_j\nu_{e-j}.
\]

Event-count weights `w_j=1` are primary. An IEI-decay form
`w_j=exp[-(t_e-t_{e-j})/tau]` is a frozen sensitivity, with `tau` selected on validation only.

Let `s^-_{e-m+1}` be estimated from a disjoint window ending before the accumulation window,
and `s^+_{e,h}` from events after `e`. The cumulative response is

\[
\Delta s_{e,m,h}
=
s^+_{e,h}-\widehat A_{m+h}s^-_{e-m+1}
=
C_{m,h}c_e^{(m)}+\Gamma_{m,h}u_e^{\mathrm{nuis}}+\epsilon_{e,m,h}.
\]

Here `u_e^nuis` denotes the frozen timing/event-rate covariates. No event used to construct
`c_e^(m)` appears in either endpoint state window. Tests must
distinguish:

- vector magnitude/dose: larger aligned cumulative innovation predicts larger displacement;
- alignment: successive innovations pointing in similar propagation directions accumulate;
- cancellation: opposing innovations attenuate the displacement;
- persistence/decay across future horizons;
- true chronology versus order-shuffled and sign/direction-matched cumulative nulls.

For the primary equal-event-count exposure, permuting events *within the same exposure*
leaves the sum exactly unchanged and is therefore not an identifiable null. The primary
chronology null instead reassigns complete exposure sequences/windows while matching
pre-state, source progress, dose and alignment. Within-exposure order is tested only in the
predeclared IEI-decay sensitivity, where unequal temporal weights make order identifiable.

Primary cumulative anchors are separated by at least `max(m,h)` events. A single-event effect
without accumulation, alignment or dose response remains a short-range state-readout result.

## 9. Boundary to the V3.1 generative identification contract

V3.0 does not fit a recurrent state-transition mechanism on the human test set. Its strongest
allowed finding is a future-directed, cumulative association between valid innovations and
observable propagation-state displacement.

The separate V3.1 contract compares a shared-filter observer-only transition with an
event-driven transition and a discrete-switching control. V3.1 implementation and synthetic
calibration may start immediately, but human transition testing is opened only when frozen
V3.0 train/validation results show a stable Goal 2 or Goal 3 signal. This is a predeclared
scientific handoff, not an invitation to tune V3.0 after test inspection.

For patients with an innovation-valid primary family and at least 20 non-overlap validation
anchors, the handoff is `OPEN` when either of these cohort-level rules holds:

- **Goal 2 route:** median held-out propagation gain over autonomous prediction is positive,
  median true-minus-state-matched-null gain is positive, and median future-minus-past effect
  is positive;
- **Goal 3 route:** median cumulative-exposure gain is positive, median true-minus-matched-null
  gain is positive, and aligned exposure exceeds the cancellation control.

These are train/validation development criteria, not human-test claims. Patient support and
both dataset-specific directions are recorded continuously; no patient-level conjunctive
PASS/FAIL rule is used. If neither route opens, V3.1 human execution is `NOT_TRIGGERED` and no
model-capacity rescue is allowed.

## 10. Anchor, split and weighting contract

The 34-patient cohort has already informed design and is exploratory. It is analyzed in full;
patients are not split post hoc into development and confirmation groups. Independent data
are required for confirmation.

- Dense causal anchors are allowed for training, with each continuity unit assigned equal
  total weight and no long source dominating the loss.
- Validation selection and primary test use non-overlapping target anchors spaced by at least
  `h`; cumulative analyses use at least `max(m,h)`.
- Dense test anchors are sensitivity only and use continuity-unit moving-block bootstrap.
- Hyperparameters, dimensions and nuisance specifications are selected on validation only.
- Test is evaluated once after the model family and decision rules are frozen.
- Seeds/folds are combined within patient; the cohort unit is the patient.

No dense anchor count is interpreted as independent sample size.

## 11. Scores and cohort inference

The primary propagation score is the equal-family mean of:

1. train-normalized, reliability-weighted masked rank error;
2. reliability-weighted pairwise precedence Brier/log score on valid non-tied pairs.

Mode occupancy is secondary, participation is tertiary and the old three-family composite is
descriptive. A result carried only by participation is called recruitment-topography
prediction, not propagation-state change.

Primary contrasts are continuous patient-level effects:

- Goal 2: held-out gain of true rank/precedence innovation over autonomous prediction and its
  state-matched null;
- Goal 3: held-out cumulative dose/alignment effect relative to matched cumulative nulls.

Per-patient estimates include support and uncertainty but are not reduced to a conjunctive
PASS/FAIL Gate. Cohort reporting gives median effect, patient bootstrap 95% CI, favorable
patient count, two-sided Wilcoxon signed-rank and sign test. Dataset-specific directions and
heterogeneity are reported without result-defined subtypes.

## 12. Evidence levels and allowed claims

These are program-level levels. V3.0 alone can assign only Levels 0–2; Levels 3–4 require the
separate V3.1 identification and, for Level 4, independent confirmation.

### Level 0 — stable backbone

Repeated events sample a stable patient-specific propagation repertoire; no detectable
dynamic update is required.

### Level 1 — leaky observer

Recent events improve tracking of the current repertoire state, but valid innovations add no
specific future information.

### Level 2 — innovation predictive association

V3.0 innovation predicts a later residual. Future-versus-past and state-matched chronology
tests determine the strength of the temporal association, but do not identify a transition
mechanism.

### Level 3 — event-associated low-rank update

This level is unavailable from V3.0 alone. It additionally requires the separate V3.1
generative test: rank/precedence innovation has a stable future-directed observable impulse,
exceeds state-matched chronology nulls, and the event-driven transition improves over a
shared-filter observer-only model and discrete switching. The allowed wording is:

> Event innovations are associated with subsequent low-rank changes in the patient-specific
> propagation state.

### Level 4 — supports activity-dependent shaping

Level 3 replicates in a fully frozen independent dataset, and repeated innovations show
directional accumulation, dose response, cancellation and compatible IEI-sensitive decay.
Only then may the manuscript say the results **support activity-dependent shaping**. Causal
plasticity, synaptic formation and biological learning remain prohibited.

Failure of one supporting experiment lowers the evidence level; it does not retroactively
invalidate stable-backbone or state-tracking results.

## 13. SNN relationship

SNN is independent. It is not a Gate, target, prior, label source or ground truth for the
human RNN. After both analyses are frozen, their observable propagation principles may be
compared as a secondary discussion-level convergence test.

## 14. Human-test release checklist

Human test execution is released only after the following **design**, not outcome, items are
machine recorded:

1. source continuity manifest and reset/carry rule;
2. family-specific state, score and innovation schema;
3. blocked/source cross-fitting and innovation-validity thresholds;
4. dense-train/source-balanced and non-overlap validation/test anchor indices;
5. `K=1..4` selection plus `K=6/8/full-rank` diagnostics;
6. local-response horizons, cumulative windows and IEI sensitivity grid;
7. all null transformations, donor exclusions, statistics and V3.0 evidence rules;
8. explicit confirmation that the human run contains no V3.1 transition-model selection;
9. config, code, source-list and input hashes.

V2.7 completion is not on this checklist.

## 15. Frozen implementation amendments before human test

- Phase 0 completed on all 34 patients under
  `results/topic5_event_innovation_impulse_response/v3_0/phase0/`; all old heldout20 events
  were excluded and 34/34 continuity/anchor contracts passed.
- An early Phase 1 run that materialized test reliability was quarantined at
  `phase1_measurement_protocol_deviation_2026-08-03/` and is scientifically inadmissible.
  The accepted measurement artifact is validation-only at
  `phase1_measurement_validation_only/`.
- The primary conditional anchor is pre-20 / future-20. Pre-40/pre-80 and cumulative
  windows are analyzed when supported; lack of the longer history is an explicit
  `UNRESOLVED_INSUFFICIENT_HISTORY` state, not a patient failure and not an innovation pass.
- The primary cumulative window is 20 events. Windows 5/10/40 and IEI-decay taus
  60/300/1800/7200 seconds are frozen sensitivities.

## 16. Frozen final-fit and test semantics

The validation-only implementation selects the observer ladder, state dimension, observer
ridge and response ridge. Before test execution, the following final-fit rule is fixed:

1. the contact-rank basis is reconstructed deterministically from dense, source-balanced
   train rows only; it is not rotated or reselected with validation or test outcomes;
2. the selected observer is refitted on train plus validation events and emits future-blind
   test-event innovations using only past events from the same continuity unit;
3. response coefficients are refitted with the validation-selected ridge on the union of
   cross-fitted train response rows and train-observer residuals from validation; no test
   score changes dimension, ladder, ridge, horizon, cumulative window or null;
4. primary test anchors are non-overlapping and require at least 20 anchors per patient;
   dense test anchors are sensitivity only;
5. state-matched and cumulative donor nulls may use test pre-state, innovation, nuisance and
   source membership for matching, but never the future target used for scoring;
6. all fitting, matching and score scaling is completed before cohort aggregation, and every
   patient contributes at most one primary effect per route.

V3.0 identifies a rank-field state and therefore its primary endpoint is the frozen
rank-plus-precedence propagation score. It has no participation-probability or mode-occupancy
head. Those descriptor families remain v2.7 observer diagnostics and cannot be imported as
V3.0 evidence. A later head would be a new contract, not a post-test sensitivity or rescue.

The validation-only Goal 2 handoff is frozen as `NOT_OPEN`: among 17 innovation-valid
patients with at least 20 anchors, median propagation gain and median true-minus-state-matched
gain were negative, while future-minus-past was positive. Goal 3 remains pending; no human
test outcome was read for this amendment.
