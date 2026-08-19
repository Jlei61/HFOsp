# Topic 4 patient-specific continuous field and local-connectivity cohort v2

**Status:** `EXPLORATORY_FROZEN_BEFORE_RUN`

**Date:** 2026-08-19

**Plan:** [`2026-08-19-topic4-patient-specific-field-connectivity-cohort-v2.md`](../plans/2026-08-19-topic4-patient-specific-field-connectivity-cohort-v2.md)

## 1. Scientific question

For each patient independently, can training-block interictal events identify a
continuous node-excitability field and field-coupled local E-to-E/E-to-I weight
redistribution that reproduce both held-out propagation modes in that patient's
real implant geometry?

This replaces the old shared-field transfer question. The old cohort rotated or
reflected one E1146-derived substrate and read every montage through it. It did
not recover a patient-specific field.

## 2. Denominators and claim boundary

- 34 patients have frozen masked stable-K=2 train/held-out targets.
- 28 have a complete geometry-only 3-D-to-2-D projection and are eligible for
  patient-specific SNN fitting.
- 6 without complete geometry are audited and reported as
  `REAL_GEOMETRY_NOT_EVALUABLE`; canonical shaft rows must not substitute for
  anatomy.
- E1146 is fitted for reconstruction and figure parity but is excluded from the
  cross-patient primary statistic because it was the model-family development
  source.
- Held-out blocks never select field, edge coefficients, optimizer settings,
  detector settings, or stopping generation.
- With no new patient blind unit, the final result remains development-only.

The strongest possible positive statement is patient-specific recovery in real
implant readout. It is not recovery of tissue outside the sampled implant and
not identification of anatomical cores.

## 3. Patient-specific model

### 3.1 Continuous node field

The latent log field is a whole-sheet cubic B-spline surface:

\[
s_p(x,y)=\sum_{a,b} c_{p,ab}B_a(x)B_b(y),\qquad
h_p=\Pi_M\{\exp(s_p)\}.
\]

`Pi_M` is the existing exact field-mass projection. The node mechanism remains

\[
V_{\theta,i}=V_{\theta,0}-h_p(x_i,y_i)d_i.
\]

The numerical surface uses the existing 18 x 18 spline lattice, but the search
acts in a fixed 12-dimensional low-frequency whole-sheet Fourier subspace. The
basis is uniform on the 20 x 20 mm sheet and is constructed without contacts,
shaft identity, patient events, mode labels, SOZ, or inferred cores. There is no
K, component count, peak count, or contact-centred basis.

The initial field is a deterministic smooth-prior draw keyed by subject ID. The
ID changes the random start only; geometry and events do not place degrees of
freedom. The E1146 field is retained only as an explicit transfer baseline and
is not an optimizer warm start.

### 3.2 Local E-source redistribution

Existing AMPA edges are modulated separately for E-to-E and E-to-I using the
six observation-invariant pair features already implemented in
`src/topic4_local_connectivity.py`:

\[
\ell^P_{ts}=\boldsymbol\gamma^P\!\cdot
\phi(h_t,h_s,\Delta x,\Delta y),\qquad P\in\{EE,EI\}.
\]

For every postsynaptic target and pathway, weights are renormalized so that the
incoming total is unchanged. Topology, delays, GABA, neuron positions and all
non-AMPA pathways remain fixed. Raw logits are clipped at 0.65, bounding the
normalization-induced edge ratio to an interpretable range without adding a
post-hoc gate.

The fitted vector contains 12 field coordinates and 12 edge coordinates. This
is a node-plus-edge fit; pathway-specific effects are read out after fitting by
paired no-edge, EE-only, E-to-I-only and joint replays of the frozen winner.

### 3.3 Slow-state runtime

Z and M are active but frozen. They are not fitted to interictal modes. Their
role is to keep this scaffold compatible with the later interictal-to-ictal
transition line. Every simulation lasts 20,000 ms and delayed runaway is
invalid. Each winner is replayed with paired slow-off seeds as a mechanism
sensitivity, not as a second selection route.

## 4. Target and objective

Patient modes are defined from masked patient training events only. A model
event is represented by fixed contact identity, participation mask and
within-event normalized rank. Missing contacts remain missing; they are never
assigned phantom ranks.

For mode `k`, the shaft-balanced descriptor loss is

\[
D_{p,k}=\frac{1}{3}
\left(D_{\rm recruitment}+D_{\rm profile}+D_{\rm precedence}\right).
\]

The two-mode supervised term protects the weaker mode:

\[
D_{\rm sup}=\max(D_{p,A},D_{p,B}).
\]

Natural KMeans is part of training rather than a decorative endpoint. K=2 is
fitted to model events without patient labels, aligned to the two patient
training templates only after clustering, and contributes its weakest-mode
loss and seed-instability penalty.

The frozen exploratory objective is

\[
J_p=D_{\rm sup}
+0.75D_{\rm KMeans}
+0.50f_{\rm OOD}
+0.25(1-\mathrm{AMI})
+0.10P_{\rm support}
+0.02\log(1+R_h)
+0.01\|\gamma\|_2^2.
\]

`OOD` is the fraction of model events whose masked rank feature lies outside
the patient-training q95 distance to both frozen patient KMeans centres. It is
not waveform OOD and not anatomical OOD.

Insufficient events or a missing mode receive a continuous support penalty and
remain in the search history. Only simulation error, non-finite state,
provenance failure or runaway is invalid.

## 5. Search and confirmation

- Fit: one independent CMA-ES per eligible patient, 24 dimensions, population
  10, six generations, common network seed within each generation.
- Restart: one deterministic second smooth-prior start only if the first two
  generations contain no evaluable candidate.
- Selection: the four best distinct training candidates are rerun on two fresh
  network seeds; their mean training objective selects one winner.
- Confirmation: the frozen winner is run on four fresh network seeds and scored
  once against held-out patient blocks.
- Optimizer evidence: preserve every candidate, generation, seed and failure;
  report best-so-far curve, final sigma and selection regret. This supports an
  empirical best-in-search claim, never a global-optimum claim.

## 6. Final readouts

Each fitted patient receives the two Fig.4-style outputs:

1. continuous `h`, signed `Delta Vtheta`, mode-A/mode-B onset density and direct
   virtual-electrode waveforms;
2. event-by-contact rank heatmap, natural KMeans rank profiles, patient
   prototypes, 2 x 2 cross-fit matrix, OOD and per-network mode counts.

The cohort figure reports subject-level held-out quantities only:

- held-out weakest-mode loss versus matched within-shaft identity null;
- natural same-network K=2 recovery;
- OOD;
- field-only versus joint node-edge change;
- EE-only and E-to-I-only replay effects on mode-specific event yield and mode
  geometry.

Subject is the inference unit. Events and network seeds are nested replicates.

## 7. Interpretation branches

- Joint fit improves held-out KMeans and weakest-mode loss over field-only:
  local connectivity contributes beyond node accessibility.
- Field-only succeeds and edge replays change yield but not geometry: node
  controls nucleation; edge is a conditional amplifier.
- Training improves but held-out does not: search overfit or target sampling
  noise, not patient-specific recovery.
- Both fail with adequate events: the static node-edge family is insufficient
  at the tested budget.
- Most candidates have no events or runaway: optimization accessibility/runtime
  is unresolved before model-family capacity can be judged.

No branch permits calling spline extrema biological cores without lesion and
matched-relocation causality experiments.
