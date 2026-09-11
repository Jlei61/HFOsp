# Topic 4 rev21: frozen dual-core interictal substrate to Z/M model-ictal transition

Date: 2026-09-02

Status: execution contract

## 1. Motivation

Rev20 stopped the free-field search and tested a fixed binary dual-core Node
substrate. Among the confirmed low-dimensional connectivity variants,
`dualcore_s39 + Joint=1.25` was the only candidate that improved all three
predeclared interictal endpoints at once: held-out complete-event distribution,
two-template alignment and OOD fraction. It remains well above the matched
patient floor, so it is a useful development substrate rather than a complete
recovery of patient interictal activity.

Rev21 asks a narrower cross-state question:

> With Node geometry and learned EE/E-to-I expression frozen, can the per-cell
> Z/M slow variables move the same SNN from a low-activity interval that still
> expresses the two patient-supported interictal templates into a broad,
> sustained and faster model-ictal state?

This is not another field fit and not a connectivity search. The clinical ictal
target is not allowed to choose a Z/M work point. It is opened once only after
the model-internal work point is frozen.

## 2. Frozen substrate

The substrate of record is:

- binary Node field `dualcore_s39`;
- centers `(1.53773425799, 1.22641798807)` and
  `(18.60661213705, 2.41771341441)` mm;
- exactly 1,499 selected E neurons;
- `node_gain=1`, `signed_depth_shrinkage=1`;
- learned pathway expression `g_EE=0.625`, `g_EtoI=1.25`, the confirmed
  rev20 `Joint=1.25` candidate;
- EE ellipse angle `45 degrees`, aspect ratio `2`;
- topology, delays, per-target incoming pathway budgets, montage, detector,
  background spatial OU and virtual-contact readout frozen.

No rev21 candidate may alter Node position, radius, threshold mapping, EE/E-to-I
coefficient row, pathway dose, ellipse geometry, external-drive amplitude or
event detector.

## 3. Why seed handling is part of the science

Rev20 used multiple network seeds, but one integer controlled both realized
network topology and dynamical randomness. That design tests combined
robustness but cannot identify which source of randomness drives variability.

Rev21 separates:

- `topology_seed`: neuron placement, realized graph and frozen threshold
  realization;
- `dynamics_seed`: simulator RNG and spatial-OU trajectory.

For comparisons among candidates, the same topology/dynamics pairs are reused
as common random numbers. This is deliberate pairing, not repeated
initialization of an optimizer. Rev21 uses an explicit finite grid and no
CMA-ES, gradient descent or local-search update.

Before Z/M is enabled, a `3 topology x 4 dynamics` Z/M-off audit measures:

1. total variance of each interictal endpoint;
2. topology-associated variance;
3. dynamics-associated variance;
4. topology-by-dynamics residual variation;
5. whether both natural clusters remain present across the matrix.

If one topology alone determines the two-cluster result, the substrate is
reported as ensemble-dependent and the Z/M scan may continue only as a
development screen. No population-stable wording is allowed.

## 4. Slow-variable parameterization

The model remains

```text
tau_z dz_i/dt = H(I_th_EI - I_i^EI) - z_i
dm_i/dt       = -m_i/tau_adp + sum_k delta(t-t_i^k)
I_net,i       = I_i^E - z_i I_i^I - eta_m m_i
```

The fitted coordinates are chosen to separate timescale from steady adaptation
strength:

```text
s_I       = I_th_EI / I_th_EI,reference
s_M       = (eta_m * tau_adp) / (eta_m * tau_adp)_reference
tau_z     = disinhibition timescale
tau_adp   = adaptation recovery timescale
eta_m     = s_M * (eta_m * tau_adp)_reference / tau_adp
```

This prevents a tau-adaptation scan from silently changing both memory and
integrated adaptation gain.

The search is staged and finite:

1. coarse access map at fixed `tau_z=5000 ms`, `tau_adp=500 ms`:
   `s_I in {0.70,0.80,0.90,1.00}` and
   `s_M in {0.50,1.00,1.50,2.00}`;
2. only around model-internally eligible coarse points, a timescale map:
   `tau_z in {3000,5000,8000} ms` and
   `tau_adp in {250,500,1000} ms`, preserving the selected `s_M`;
3. Z-only, M-only and Z/M-off controls at the frozen finalist.

No parameter level is added after viewing clinical ictal data.

## 5. One uninterrupted trajectory and formal interictal events

Every run is one uninterrupted simulation. It records:

- the complete low-activity and pre-transition interval;
- at least 1,200 ms after the operational transition detector;
- population rate and active fraction;
- occupied-bin recruitment at 0.5, 1 and 2 mm;
- the 15-contact continuous readout;
- Z/M traces and h-weighted Z/M traces;
- the complete edge-supported causal-family interictal event table.

The formal interictal observation is exactly the rev20 complete returned,
temporally isolated, edge-supported causal family. Detector fragments cannot
define A/B samples. Only families whose complete return precedes model-ictal
onset enter interictal scoring.

There is no requirement for 20 events. Zero or one returned family is a valid
low-information outcome. Distribution distance requires at least two families;
natural KMeans is reported only when mathematically estimable. An unestimable
endpoint is never silently converted into failure or success.

## 6. Model-ictal qualification

The operational detector locates a candidate transition. Qualification then
uses a full 1,000 ms interval beginning 100 ms after the detector-adjusted onset.
Let `F_E(t)` be the fraction of E neurons active in 20 ms and `F_sheet(t)` the
fraction of occupied 1 mm bins in which at least half of local E neurons are
active. Primary bins contain at least 20 E neurons.

`MODEL_ICTAL_ELIGIBLE_REV21` requires:

1. transition onset occurs after 2,000 ms and early enough to record the full
   post-onset window;
2. at least 80% of the early window simultaneously has `F_E>=0.50` and
   `F_sheet>=0.50`;
3. median 20 ms-binned population E rate is at least twice the low-activity
   reference (the raw integration-step median is not used because sparse
   interictal firing makes it exactly zero);
4. median contact spectral centroid in 10-250 Hz increases by at least 5 Hz and
   by at least 25%;
5. state remains finite through the complete window.

Sensitivity reports use duty 70/80/90%, recruitment thresholds 0.4/0.5/0.6,
bin sizes 0.5/1/2 mm and onset shifts -100/0/+100 ms. Sensitivities cannot
replace the primary definition.

Returned high-rate bursts that repeatedly fall back to a mostly silent sheet do
not satisfy the broad sustained-state clause. High firing rate alone is not a
model-ictal state.

## 7. Interictal retention endpoints

The following are kept separate and shown separately:

1. **Complete distribution**: unconditional shaft-aware event-cloud distance
   against patient training during selection and held-out blocks after freeze.
2. **Two-template concordance**: natural model K=2, balanced alignment to the
   frozen patient direction classifier, cluster proportions and prototype
   matrix.
3. **OOD**: fraction over every complete returned pre-transition family;
   unreadable families count as OOD and are also reported separately.

Event rate, onset latency and estimability are sidecars. A lower OOD caused by
event suppression, unreadability or one-cluster collapse is a tradeoff, not an
improvement.

The 12-cell Z/M-off orthogonal audit defines the substrate reference
distribution. The coarse screen additionally runs Z/M-off on the same four
topology/dynamics cells as every active candidate. For each endpoint it reports
both (i) the paired candidate-minus-off difference and (ii) the candidate
aggregate relative to the 5th--95th percentile of the independent 12-cell off
reference. A candidate is `INTERICTAL_SUBSTRATE_RETAINED` only when all three
aggregate endpoints are estimable and lie on the admissible side of that 90%
reference support: distribution distance and OOD no higher than q95, and
two-template alignment no lower than q05. Pooled two-cluster presence is
required, but no per-network minimum event count is imposed. Paired differences
remain effect estimates and cannot by themselves rescue an out-of-support
candidate.

## 8. Work-point selection

Patient ictal vectors, Fig.3 bridge scores, seizure morphology and the historical
5% E-to-I visual candidate are forbidden before work-point freeze.

Coarse and timescale screens use fixed topology/dynamics pairs. The selection
order is:

1. highest fraction of `MODEL_ICTAL_ELIGIBLE_REV21` units;
2. `INTERICTAL_SUBSTRATE_RETAINED` preferred over non-retained;
3. smallest worst standardized deterioration among complete distribution,
   KMeans alignment and OOD;
4. broad-state and onset stability under neighboring parameter levels;
5. smallest log-distance from the reference Z/M parameters.

This order is frozen before the scan. No weighted patient-ictal objective is
used. At most one work point enters confirmation on a fresh
`3 topology x 4 dynamics` matrix.

If no point passes both state and retention criteria, the result is
`NO_CROSS_STATE_WORKPOINT_IN_FROZEN_ZM_GRID`. A model-ictal-only candidate may
be displayed as a mechanism boundary but cannot become the primary Fig.5
work point.

## 9. Post-freeze clinical bridge

Only after `WORKPOINT_FROZEN.json` is written and hashed may the producer read
the E1146 clinical ictal target. The bridge reports, without retuning:

- global energy increase;
- absolute early-ictal contact-energy organization;
- baseline-to-early incremental spatial redistribution;
- temporal evolution at baseline, pre-transition and early transition.

Model and clinical frequencies are not equated. The model endpoint is a
qualified model-ictal state. Clinical agreement is a retrospective continuous
evaluation, not a model eligibility gate.

## 10. Fig.5 deliverable

The main figure uses one confirmed uninterrupted trajectory and the accepted
Fig.5 grammar:

- continuous 15-contact readout showing returned interictal events and the
  transition into sustained broad high-frequency activity;
- projected Z/M trajectory with the selected transition marked;
- low-activity and early-ictal spatial recruitment/energy fields;
- interictal complete-family/KMeans sidecar linked to the same seed;
- multi-seed summary of state eligibility, complete distribution, two-template
  alignment and OOD;
- post-freeze clinical energy-gradient bridge.

Every figure directory contains a Chinese README. PNG and PDF are visually
checked; the representative trajectory cannot substitute for the multi-seed
table.

## 11. Claim boundary

This is a one-patient, development-only SNN experiment. Passing rev21 would
show that a frozen interictal-data-constrained substrate can, under a specified
Z/M parameter regime, express both patient-supported interictal templates and
then enter a model-internal ictal state. It would not identify Z/M as the
patient's biological seizure mechanism, prove anatomical cores, equate model
current with SEEG or establish patient generalization.
