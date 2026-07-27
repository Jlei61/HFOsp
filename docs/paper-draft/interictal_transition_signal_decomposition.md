# Computational supplement: decomposition of interictal transition structure

> 定位：Supplementary Results / model-development evidence。  
> 不进入当前主文 Figure 6；不构成 early-ictal transfer 结果。

## Results draft

To determine which features supported the reproducible Markov signal, we
decomposed patient-specific conditional transition residuals without using
interictal template labels or seizure data. Across 31 development-excluded
patients, the first-order transition model improved held-out next-contact
likelihood over a node-bias control in 30 patients (median normalized NLL
benefit, 0.0144; 95% bootstrap CI, 0.0108–0.0172; BH-adjusted
\(q=2.56\times10^{-9}\)). This gain was retained by the symmetric component of
the transition residual (30 of 31 patients; median benefit, 0.0148), whereas
adding an unconstrained skew-symmetric component provided no further benefit
(median increment, \(-4.35\times10^{-5}\); \(q=0.252\)).

The transition signal was not explained by local implantation geometry. In 22
patients with complete contact coordinates, a model containing only same-shaft
and Euclidean-distance terms did not improve consistently over node bias
(median benefit, 0.00044; \(q=0.168\)). The empirical transition residual
remained informative relative to this local-geometry model (median benefit,
0.00981; 21 of 22 patients; \(q=1.24\times10^{-5}\)). The gain also persisted
when evaluation was restricted to held-out prefixes containing a cross-shaft
next contact and all cross-shaft eligible contacts were included in the
likelihood (median benefit, 0.0157; 19 of 20 eligible patients;
\(q=1.31\times10^{-5}\)).

An axis-aligned residual provided a smaller but reproducible increment over
local geometry (median benefit, 0.00215; 20 of 22 patients;
\(q=4.37\times10^{-6}\)). This effect should not be interpreted as uniformly
enhanced propagation along a physical axis: the fitted axial coefficient was
negative in 14 of 22 patients, and the local and axial basis functions remained
highly collinear (median Frobenius cosine, 0.972). Conditioning a shared axial
direction term on the observed source position yielded a statistically
detectable but quantitatively small additional benefit (median,
\(1.73\times10^{-5}\); 15 of 22 patients; \(q=0.00861\)).

Multi-step history contributed substantially more. The last observed rank
improved over the event source alone (median benefit, 0.0184; 29 of 31
patients; \(q=5.12\times10^{-9}\)), and an ordered, decaying representation of
the full prefix further improved over the last rank (median benefit, 0.00681;
30 of 31 patients; \(q=2.56\times10^{-9}\)). Thus, the interictal sequences
contained history-dependent transition information that was not saturated by
a first-order Markov description.

## Discussion draft

These analyses clarify why the v2.2 propagation-state model failed and what a
subsequent model would need to represent. The useful signal was neither a
contact-frequency effect nor a simple consequence of same-shaft sampling.
Instead, it was predominantly symmetric, extended across shafts and depended
on ordered multi-step history. A physical-axis basis captured an additional
component, but its mixed coefficient sign indicates that the effective
next-contact residual combines propagation drive with competition or
refractory suppression. Structural scaffold and observed transition hazard
therefore cannot be identified with the same non-negative operator.

The decomposition supports development of a minimal recurrent observation
model with separate scaffold, source-conditioned direction and low-dimensional
history or competition states. It does not motivate a generic GRU or a return
to discrete A/B path labels. Nor does it establish a shared anatomical
pathological axis: the axial effect was modest, the source-conditioned
increment was small, and early-ictal targets remained sealed. Cross-state
transfer will additionally require independently adjudicated, seizure-specific
clinical-onset contacts, which are not yet available in the current cohort.

## Claim boundary

### Supported

- Interictal rank sequences contain transition information beyond node
  participation frequency and local implantation geometry.
- The effective transition residual is predominantly symmetric.
- Ordered multi-step history improves over first-order transition information.
- An axis-aligned component is detectable, with mixed sign across patients.

### Not supported

- A positive anatomical propagation axis has been recovered.
- Source-conditioned direction is the dominant transition signal.
- A recurrent model has already reproduced the interictal dynamics.
- The interictal scaffold predicts early-ictal energy recruitment.
