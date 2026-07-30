# Interictal sequences are predictably structured but do not identify a physical-axis mechanism

> **Historical model-stage record.** 当前 manuscript-facing Supplementary
> 版本为 `figure6_static_contact_topography_bounded_result.md`。本文件只保留
> competitive-RNN 阶段的实现与边界。

We next asked whether the ordered contact recruitment observed within interictal
population events could identify a compact recurrent mechanism. We trained a
patient-specific recurrent observation model directly on masked contact-rank
sequences. The model was restricted to a symmetric axis-aligned propagation
operator, an observed event source, and propagation and delayed-competition
states; no template labels, seizure labels, or ictal measurements were
available during training. To avoid the set-size miscalibration of the earlier
Bernoulli formulation, the model predicted a single next contact using a
categorical likelihood conditional on event continuation.

Across 22 geometry-complete patients, the full model improved held-out
next-contact log likelihood over a fixed node-frequency baseline in every
patient (median NLL benefit, 0.078; 95% bootstrap CI, 0.043–0.113; 22/22
patients; FDR-adjusted \(P=1.4\times10^{-6}\)). Removing recurrent persistence
also reduced performance (median benefit, 0.020; 95% CI, 0.006–0.032; 18/22
patients). Thus, interictal rank sequences contained ordered historical
information beyond each contact's overall participation frequency.

The mechanistic restrictions, however, were not identified by these data.
Adding the delayed-competition state did not improve prediction over the
one-state model (median benefit, −0.00078), and neither the full axis model nor
the matched axis model without the source term showed a confidence-bounded
advantage over local isotropic propagation. The source-conditioned directional
term was likewise unsupported. The full model recovered approximately 58% of
the cohort-median improvement provided by an empirical ordered-history Markov
model, but exceeded that benchmark in only 3 of 22 patients. These results
therefore support a low-dimensional historical dependency in interictal
contact-rank sequences, but not the inference of a shared physical propagation
axis, delayed competition, or source-dependent reversal from the present
recurrent formulation.

## Figure caption

**Figure 6 | A constrained recurrent model captures interictal transition
history but not the proposed physical-axis mechanism.**
**A,** Model contract: a symmetric axis-aligned scaffold receives the observed
rank set and updates propagation and delayed-competition states.
**B,** Patient-level held-out categorical NLL benefit of the full model over
the fixed node-frequency baseline. Each point is one patient and the horizontal
bar denotes the median.
**C,** Ablations of recurrent persistence and the delayed-competition state.
History improved prediction, whereas the second state did not.
**D,** Incremental benefit of the full axis bundle, the matched axis-only
comparison, and the source-conditioned directional term. None met the
pre-registered confidence criterion.
**E,** Patient-level transition benefit recovered by the structured model
relative to an empirical ordered-history Markov benchmark. The dashed line
denotes equality.
**F,** Pre-registered interpretation gates. Predictive adequacy and historical
dependence passed; delayed competition, physical-axis structure, and
source-conditioned direction did not. Early-ictal targets remained sealed and
no cross-state transfer was performed.

## Claim boundary

This result supports self-supervised prediction of within-event contact order
and a compact historical dependency. It does not demonstrate recovery of
anatomical connectivity, a cellular excitation-inhibition mechanism, replay of
the two interictal templates, or prediction of early-ictal energy recruitment.
