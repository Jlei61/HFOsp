# Computational supplement: limits of direct symmetric-axis system identification

> 当前定位：supplementary bounded-negative result，不进入主文 Figure 6。  
> 冻结证据：v2.2.1 closeout，22 位 development-excluded geometry-complete
> patients，3 seeds。

## Terminology ledger

| Canonical term | Definition |
|---|---|
| symmetric-axis propagation-state RNN | v2.2 的非负、线性、单状态 recurrent model |
| effective propagation operator | contact-level model operator；不是 anatomical connectivity |
| empirical first-order Markov model | train80 conditional transition control |
| node-bias model | 无 history 的 eligible-prefix contact hazard control |
| rank-step persistence | 每个 observed rank step 的 state retention；不是生物时间常数 |
| early-ictal target | clinical-onset `[0,10] s`, `1–150 Hz` energy field；本阶段 sealed |

## One-sentence argument

In patient-specific interictal contact-rank sequences, first-order transition
structure was reproducible, but a nonnegative linear symmetric physical
scaffold used directly as the sole next-contact operator failed to capture
that structure, indicating that the missing component lies in the mapping from
structural scaffold to observed transitions rather than establishing the
absence of a shared pathological axis.

## Results draft

We next asked whether interictal rank sequences could identify a shared
physical propagation scaffold without using A/B template labels. In 22
development-excluded patients with complete contact geometry, an empirical
first-order Markov model improved held-out next-set likelihood over a
no-history node-bias control in 21 of 22 patients (median normalized NLL
benefit, 0.0108; 95% bootstrap CI, 0.0066–0.0155; one-sided Wilcoxon
\(P=4.77\times10^{-7}\)). Thus, event prefixes contained reproducible
information about the next recruited contact beyond each contact's marginal
hazard.

The constrained propagation models did not capture this information. Both the
local-isotropic model and the symmetric-axis model performed below the
node-bias control (median benefits, −0.0166 and −0.0166, respectively; one
positive patient for each model). The empirical Markov model outperformed the
full symmetric-axis model in all 22 patients (median NLL difference, 0.0290;
\(P=2.38\times10^{-7}\)). Adding axial anisotropy also did not improve the
constrained model over its isotropic counterpart for either the next-set or
future first-arrival endpoint. Because the full model class itself failed the
predictive-adequacy check, this comparison does not show that physical axes are
absent; it shows that the present linear propagation-state formulation did not
provide a valid observation model for the rank transitions.

Calibration analysis identified a concrete source of mismatch. The observed
non-terminal next rank set contained one contact on average, whereas both
constrained models predicted approximately 1.65 contacts. This overprediction
was strongest immediately after the observed source and decreased at later
rank steps. The local and axial kernels were also highly collinear across
patients (median Frobenius cosine, 0.979), and removing the learned axial
mixture changed held-out contact logits only weakly (median mean absolute
change, 0.0052). Accordingly, the near-identical axes obtained across random
seeds indicate reproducible optimization, not independent identification of a
patient-specific physical axis.

## Discussion draft

The negative result narrows the required bridge between the empirical data and
the network mechanism. A nearly symmetric structural scaffold can generate
directional events when combined with an observed source, a moving propagation
front, local competition and activity-dependent state. The v2.2 model instead
treated the symmetric nonnegative scaffold itself as the effective one-step
transition operator. Its failure is therefore compatible with the SNN result
and suggests that structural connectivity and observed transition hazards
should be separated explicitly. Before introducing another recurrent model,
we decompose the empirical Markov signal into local geometry, symmetric and
directed components, source-conditioned axial flow and multi-step history.

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Interictal prefixes contain next-contact information | Markov > node-bias, 21/22, \(P=4.77\times10^{-7}\) | supported |
| Current linear propagation class is predictively inadequate | full and isotropic < node-bias, 1/22 positive | supported |
| Axial anisotropy has no added value within this failed model class | full vs isotropic Claim 2 both FAIL | supported with model-class boundary |
| The shared pathological axis does not exist | not tested | forbidden |
| Early-ictal transfer failed | target sealed and exact sources absent | forbidden |

## Figure use

The four-panel closeout diagnostic may be used in Supplementary Results:

`results/topic5_symmetric_axis_propagation_state_v2_2/closeout_v2_2_1/figures/v2_2_1_closeout_diagnostics.pdf`

不要把原六块 Figure 6 作为主文图继续使用；其中 Claim 3/4 和 transfer 是按 stop
rule 锁定，不是阴性 data panels。

