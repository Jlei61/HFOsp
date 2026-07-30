# Topic 5 constructive within-event generation sufficiency v0.1

## 1. Scientific question

This contract asks one question:

> Given the observed first rank set of a held-out interictal event, are a
> patient-specific static contact scaffold, the frozen short-memory
> linear-state transition residual, and an independent event-progress/STOP
> process sufficient to generate the rest of the event and recover its global
> bidirectional propagation organization?

The model is an **algorithmic sufficiency test**, not a biological mechanism
and not a seizure predictor.

The three scientific objects remain separate:

\[
\text{where: static contact recruitment}
\neq
\text{how: within-event generation}
\neq
\text{when: real-time seizure entry}.
\]

The repository does contain a separate IEI/rate analysis in Topic 2:
lognormal IEIs, positive serial correlation and a broad seizure-centered rate
elevation. Those results are explicitly outside the present Paper 1 framework
and have not been connected to a persistent RNN state. They neither authorize
nor motivate a seizure-countdown claim here.

## 2. Frozen data, model and information seal

- Dataset: `results/topic5_interictal_rank_distribution/dataset_v0_4`.
- Cohort: 34 patients, 18 Epilepsiae and 16 Yuquan.
- Split: frozen chronological train80 / heldout20 within patient.
- Model checkpoints: target-blind selected `linear_state`, three seeds per
  patient, from
  `results/topic5_ordered_history_architecture_audit/formal/architecture_controls_formal_20260729/`.
- One model step is one within-event rank step.
- Every event resets; no state passes between events.
- No retraining, temperature tuning, architecture search or early-ictal
  optimization is permitted.
- A/B labels, template labels, physical axes, seizure labels and ictal values
  are forbidden during fitting and rollout.
- A/B and physical-axis quantities may be constructed from train80 interictal
  events only and used after generation as read-back endpoints.

## 3. Constructive generator

The frozen linear-state checkpoint supplies contact-transition residuals.
For patient \(p\), define:

\[
\ell_{t+1,c}
=
\underbrace{b_{p,c}}_{\text{static scaffold}}
+
\underbrace{
\ell^{\rm linear}_{t+1,c}
-
\ell^{\rm linear}_{0,c}
}_{\text{ordered transition residual}}.
\]

`b` is the smoothed train80 contact participation log-frequency. The
subtraction removes the checkpoint's no-prefix contact field, leaving a
history-dependent residual. Its scale is fixed to one.

STOP is not taken from the contact-state decoder. It is an independent
train80 event-progress hazard:

\[
h_p(t)
=
\frac{N_p(L=t)+a}
{N_p(L\geq t)+2a},
\qquad a=1,
\]

where \(L\) is the number of rank sets. This makes the three functional
components explicit:

\[
b_{p,c}
+
K_{\rm short\ history}
+
q_{\rm termination}(t).
\]

No heldout length or suffix is used to estimate `b`, the transition residual
or `h(t)`.

## 4. Source-conditioned free running

For each heldout event:

1. reveal only its first rank set;
2. reset the model and advance it once with that source set;
3. sample the next contact or STOP;
4. feed the sampled contact back into the model;
5. continue until STOP or all contacts have appeared.

The model never sees the true heldout suffix. Exact zero-tolerance ties are
retained in the revealed source; subsequent generated rank sets contain one
contact because ties are effectively absent in the primary encoding.

All conditions use the same heldout sources and the same pre-generated
uniform random numbers. Source contacts are not counted as model-predicted
success. Participation, endpoint and rank metrics must include a
suffix-conditioned version that excludes the revealed source contribution.

## 5. Frozen paired conditions

1. `full_constructive`: static scaffold + full linear-state residual +
   train80 progress hazard.
2. `static_only`: static scaffold + train80 progress hazard; ordered contact
   residual is zero.
3. `static_shuffle`: shaft-preserving permutation of `b`, with transition and
   progress retained.
4. `history_h1`: recompute the transition residual from the latest generated
   rank only, while retaining causal progress.
5. `history_h2`: recompute it from the latest two generated ranks.
6. `constant_stop`: replace the step-specific progress hazard by the single
   train80 marginal termination hazard.
7. `no_termination`: forbid STOP until every contact has appeared.

`history_h2` is the short-memory candidate supported by the v0.2 contact-loss
decomposition. `full_constructive` tests whether older exponentially decayed
state adds anything in free running. No condition is selected after seeing
heldout global metrics.

## 6. Posterior predictive endpoints

### 6.1 Event grammar

- rank-set/event length distribution;
- STOP hazard by generated rank;
- participant count;
- source-to-sink distance;
- sink-contact distribution.

### 6.2 Contact and sequence structure

- suffix participation MAE;
- contact-wise suffix rank Wasserstein error;
- pairwise precedence correlation and MAE;
- first-order transition residual agreement.

### 6.3 Train-only unsupervised global structure

For each patient, fit the read-back only on train80 human events:

- masked rank features;
- KMeans \(k=2\), reported continuously without treating two modes as a gold
  standard;
- train template stability and anticorrelation;
- nearest-template assignment for heldout human and generated events;
- mode prevalence, assignment margin and generated-template match.

The generated events are never reclustered to define an easier post-hoc
target. A sensitivity may recluster them, but it cannot enter a pass gate.

### 6.4 Independent train-derived physical axis

For geometry-complete patients, define an unsigned train80 propagation axis
as PCA1 of source-to-sink displacement vectors. This uses no A/B label.

Heldout and generated events are evaluated on:

- absolute concentration along the train axis;
- positive/negative side prevalence;
- signed displacement Wasserstein distance;
- sink-position distribution along the axis;
- support on both source sides.

The revealed source is part of the conditioning contract; successful sink
and direction reconstruction, not source recovery, carry the evidence.

## 7. Empirical variability reference

Absolute fidelity is assessed against human sampling variability:

- split heldout human events chronologically into two halves;
- match sample counts by deterministic subsampling;
- compute the same event, suffix, mode and axis endpoints;
- define each patient's empirical error floor from the heldout-half
  comparison.

Generated fidelity is `within empirical range` only when its error is no
larger than the empirical half-vs-half error plus 10%. This is descriptive per
patient and is never replaced by an arbitrary cohort average.

## 8. Component-specific predictions

- Removing/shuffling `b` should primarily damage suffix participation and sink
  topography.
- Removing ordered transition should primarily damage precedence, rank and
  bidirectional-mode/axis read-back.
- Replacing/removing the progress hazard should primarily damage length and
  STOP curves.

These are functional necessities within this model class, not biological
necessities.

## 9. Hierarchical sufficiency gates

### Gate A: engineering and leakage

- 102/102 checkpoint cells load;
- every generated event retains the revealed source and never repeats a
  contact;
- all paired conditions use identical source rows and uniforms;
- no A/B, axis or ictal field is read before rollout artifacts are frozen.

Failure blocks all scientific claims.

### Gate B: local constructive fidelity

At patient level, after seed collapse:

1. `full_constructive` improves suffix precedence correlation or rank error
   over `static_only` with two-sided Wilcoxon \(P<0.05\);
2. the full model is within the empirical range for at least two of suffix
   participation, suffix rank and precedence in at least half the cohort;
3. `constant_stop` or `no_termination` is worse for event length/STOP in the
   predicted direction.

If Gate B fails, retain only the existing one-step result and stop.

### Gate C: global bidirectional organization

Evaluated only where the train80 read-back is reliable and both source sides
have prespecified support:

1. generated-template match and signed displacement fidelity improve over
   `static_only` at patient level with \(P<0.05\);
2. at least half the eligible patients are within empirical range for both
   template/mode and signed-axis endpoints;
3. positive and negative directions are both represented without fitting
   direction labels to generated data.

Passing Gate B but failing Gate C means local recurrence is real but
insufficient for the global A/B repertoire.

Passing Gates B and C supports:

> A patient-specific static scaffold, short within-event recurrence and an
> event-progress termination process are algorithmically sufficient to
> generate the observed bidirectional event organization.

## 10. SNN gate

The existing SNN virtual-electrode events enter the same fingerprint pipeline
only if human Gates B and C both pass and a read-only inventory confirms:

- identical event detection semantics;
- identical contact/rank-set representation;
- adequate event count for full and prespecified mechanism lesions.

No new SNN parameter sweep is permitted. If either human gate fails, the SNN
fingerprint branch is `LOCKED_NOT_RUN`, not incomplete.

## 11. Stop rules and wording

- One-step positive, Gate B fail: supplementary short-order audit only.
- Gate B pass, Gate C fail: local generative grammar, not global sufficiency.
- Gates B/C pass: algorithmic/effective sufficiency, not unique or biological
  mechanism.
- SNN fingerprint pass after human gates: a biophysical sufficient
  implementation, not parameter identity and not evidence that interictal
  events trigger the next seizure.

Early-ictal prediction is not reopened. The existing negative ordered
increment remains a boundary result.
