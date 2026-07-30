# Topic 5 minimal sequence-kernel closeout v0.2

## 1. Scientific scope

This contract separates three objects that must not be pooled:

\[
\text{where: stable contact recruitment}
\neq
\text{how: within-event ordered transitions}
\neq
\text{when: inter-event real-time state}.
\]

The current event-reset models can address only `where` and `how`. Their
strongest permitted interpretation is:

\[
\text{stable patient-specific contact prior}
+
\text{short within-event ordered correction}.
\]

They cannot identify a seizure countdown, a continuous-time biological state,
or a unique recurrent mechanism.

The primary question is:

> After controlling the stable contact prior and the unordered recruited
> prefix, do the latest two to three rank sets improve held-out prediction of
> which contact is recruited next?

The architecture zoo is frozen. No additional GRU, nonlinear RNN, Dale-law or
recurrent-rank search is permitted in this closeout.

## 2. Frozen data and split

- Input: `results/topic5_interictal_rank_distribution/dataset_v0_4`.
- Cohort: 34 patients, with Epilepsiae and Yuquan reported separately.
- Nonparticipants remain missing; all ranks are participant-masked.
- Within patient: chronological train80 / heldout20.
- Patient is the inferential unit; seeds are collapsed within patient.
- Every event begins with a reset state.
- One model step is one within-event rank step, not seconds or minutes.
- Ictal targets remain unread until the target-free within-event outputs and
  model selection fields are frozen.

## 3. Exact likelihood decomposition

At decision \(d=(e,t)\), the model assigns one joint categorical distribution
over STOP and all eligible contacts:

\[
Z_d=\exp(s_d)+\sum_{j\in C_d}\exp(\ell_{d,j}),
\qquad
p_{\mathrm{stop},d}=\frac{\exp(s_d)}{Z_d}.
\]

For a nonterminal next set \(S_{d+1}\):

\[
\mathcal L_{\mathrm{total},d}
=
\underbrace{-\log(1-p_{\mathrm{stop},d})}_{\mathcal L_{\mathrm{continue},d}}
+
\underbrace{
-\log
\frac{\sum_{j\in S_{d+1}}\exp(\ell_{d,j})}
{\sum_{j\in C_d}\exp(\ell_{d,j})}
}_{\mathcal L_{\mathrm{contact}\mid\mathrm{continue},d}}.
\]

For a terminal decision:

\[
\mathcal L_{\mathrm{total},d}
=
\mathcal L_{\mathrm{STOP},d}
=-\log p_{\mathrm{stop},d}.
\]

The decomposition must reconstruct the original per-decision NLL within
floating-point tolerance. STOP probability must be computed from the joint
softmax; applying a sigmoid to the raw STOP logit is forbidden.

Two summaries are reported:

1. an additive event-balanced decomposition in which the contact contribution
   is zero on terminal decisions;
2. the primary contact-choice NLL, averaged only over nonterminal decisions
   within event and then patient.

All H1/H2/H3/full, shuffle and intervention comparisons must use the same
decision keys, candidate masks and event denominator. Conditions with
different eligible decision sets are not paired.

### Cardinality boundary

The frozen zero-tolerance encoding is audited before inference. The observed
inventory contains only 73 multi-contact ties among 5,902,546 rank sets
(approximately 0.0012%). Therefore next-set cardinality is not an estimable
primary endpoint in this encoding. It is reported as a descriptive data
contract, not fitted as a separate prediction head.

## 4. Nested predictive models

The conceptual decomposition is:

\[
\ell_{e,t+1,c}
=
\alpha_{p,c}
+
f_{\mathrm{set}}(U_{e,t},c)
+
f_{\mathrm{ord}}(S_{e,t-2:t},c).
\]

The accepted controls remain:

1. `static_contact_hazard`;
2. `unordered_prefix`;
3. `last_set_first_order`;
4. H1, H2, H3 and full ordered history;
5. within-event rank shuffle;
6. selected `linear_state`.

The primary ordered endpoint is the held-out reduction in
\(\mathcal L_{\mathrm{contact}\mid\mathrm{continue}}\), not the previously
pooled next-contact/STOP NLL.

## 5. Input-output invariant lag kernels

For the selected diagonal linear state:

\[
h_t=Ah_{t-1}+Bx_t+Gq_t+b,
\qquad
\ell_{t+1}=Ch_t+a,
\]

where \(x_t\) is the contact-set token and \(q_t\) contains causal scalar
progress covariates. Define:

\[
K_k=CA^kB,\qquad k=0,1,2,\ldots
\]

and map the token input back to contact space using the fitted patient contact
embeddings. Contact-logit rows and the STOP row are retained separately.
Because \(K_k\) is invariant to invertible hidden-state coordinate changes,
it is the main interpretation object; hidden PC axes are not.

Required analyses:

1. remove contact-identity input at lag 0, 1, 2 and \(3+\) while retaining the
   true candidate mask, progress and set-size covariates;
2. compute held-out changes in total, contact-choice and STOP NLL on identical
   decisions;
3. compare flattened \(K_k\) across seeds within patient;
4. report same-shaft and distance summaries only as secondary structure;
5. use an independently train-derived A/B axis only if an axis analysis is
   shown; an A/B label is never the fitting target.

## 6. Explicit FIR-H3 model

Fit a no-recurrent-state residual model:

\[
\ell_{t+1}
=
\ell_{\mathrm{unordered}}(U_t)
+
K_0x_t+K_1x_{t-1}+K_2x_{t-2}.
\]

Implementation contract:

- first fit the unordered baseline under the frozen training protocol;
- freeze its encoder, prefix model, decoder and patient contact offsets;
- fit only three lag-specific ordered projections on outer-patient train80
  events;
- do not recalibrate the ordered branch on heldout20;
- use the same candidate masks, STOP action and event-balanced loss as the
  linear state.

Interpretation is frozen:

- FIR-H3 equivalent to linear state: report a finite-memory conditional
  transition kernel and prefer FIR-H3 in the paper;
- linear state clearly better: retain its shared decay parameterization and
  interpret through \(K_k\);
- gain confined to STOP: report event-termination grammar, not propagation;
- gain in contact choice: report short-range propagation order information.

## 7. Input-output order

Construct a finite-horizon block Hankel matrix from the fitted lag kernels:

\[
\mathcal H=
\begin{bmatrix}
K_0&K_1&K_2\\
K_1&K_2&K_3\\
K_2&K_3&K_4
\end{bmatrix}.
\]

Report its singular spectrum and the rank retaining 90% and 95% squared
singular-value energy. This is an input-output predictive order, not a brain
manifold dimension. A low-order claim requires:

1. seed-stable leading singular spectrum; and
2. negligible held-out contact-choice loss after a prespecified low-order
   input-output truncation.

If only the spectrum is available, the result remains descriptive.

## 8. Dataset confirmation and robustness

The new decomposition and kernel endpoints are frozen using Epilepsiae as the
development dataset and evaluated on Yuquan without endpoint or hyperparameter
changes. The reverse direction is a sensitivity analysis.

Because both datasets participated in the earlier architecture audit, this is
called `cross-dataset confirmation of the new endpoints`, not untouched
external validation and not independent confirmation of the original model
selection.

Report:

- each dataset separately;
- patient heterogeneity versus event count, contact count and event length;
- effect size in nats and bits per decision;
- rank-set tolerance sensitivity at 1, 2, 5 and 10 ms, using the frozen raw
  lag values and the same chronological split.

Tolerance sensitivity may re-encode and re-evaluate the frozen model only if
contact identities and decision denominators remain auditable. It must not
retune the model.

## 9. Data-level matched contexts

Search heldout events for contexts sharing:

- the same unordered prefix;
- the same prefix length;
- the same candidate mask;
- different recent two-to-three-rank order.

First report the number of train-supported and heldout-evaluable contexts. A
data-level order/outcome test is run only if the frozen minimum support is met:
at least 20 patients and at least 50 heldout decisions per patient across
repeated context families. Otherwise this analysis is `INSUFFICIENT_SUPPORT`,
not negative.

## 10. Separate real-time `when` branch

No event-reset checkpoint can be reused as a real-time state model.

### Gate 0: seizure-specific target reliability

For seizure \(s\) in patient \(p\):

\[
Y_{p,s}=\mu_p+\delta_{p,s}.
\]

Audit patient-mean field reliability and seizure-specific residual reliability
using leave-one-seizure-out means and, where the cache permits, nonoverlapping
within-seizure time halves. If the lower confidence bound of residual
reliability is not above zero, dynamic ictal prediction is stopped.

### Gate 1: inter-event predictive state

Before any seizure target, test whether causal histories containing real event
times predict the next interictal event beyond:

1. patient static prior;
2. last event;
3. recent unordered event average;
4. event rate;
5. time of day;
6. IEI-only baseline;
7. block shuffle and patient-wise circular shift.

This stage is a feasibility and signal audit. It does not authorize a new GRU.
Only if Gate 0 and Gate 1 both pass may a separate inter-event model spec be
written.

## 11. Figure and manuscript position

This result is an Extended Data or Supplementary bounded computational result.
The six panels are:

1. additive task decomposition and event reset;
2. train-heldout static scaffold stability;
3. H1/H2/H3/full contact-choice horizon;
4. minimal architecture/FIR comparison;
5. invariant lag-kernel contribution and input-output order;
6. static, unordered and ordered early-ictal increments with the ordered
   boundary shown explicitly.

A preselected single-patient heatmap is supplementary, not a main cohort
panel. Literal escape sequences, overlapping labels and mathematically
duplicate interventions are forbidden.

## 12. Stop conditions and strongest allowed claim

Stop and repair on leakage, fingerprint drift, inconsistent decision keys,
candidate-mask mismatch, failure of exact NLL reconstruction, NaN or
incomplete seeds.

The strongest allowed result, only if contact-choice gain survives, is:

> Stable patient-specific recruitment structure is supplemented by a
> short-range, approximately two-to-three-rank conditional transition
> kernel. This event-indexed information does not establish a unique
> recurrent mechanism or a real-time pre-seizure state.
