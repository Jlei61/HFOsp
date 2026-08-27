# Topic 4 rev14: causal-family static Node free-field refit

## 1. Motivation

The current data-driven Node substrate is not rejected, but it is not yet an
accepted reconstruction of the patient's two interictal propagation modes.
Rev13 corrected the observation unit from detector fragments to complete
edge-supported causal families. It then showed that an activity-dependent
zero-sum threshold redistribution did not add reproducible two-direction
capacity beyond the frozen static Node field and matched controls.

The corrected static baseline nevertheless exposes a more specific problem.
Across three fresh networks it produces stable pooled model-internal K=2
structure, but 51.6--57.7% of all isolated causal families are outside the
frozen patient-mode support and only 11.5--24.1% recruit both shafts. The pooled
KMeans balanced alignment is 0.875, yet only two of three individual networks
align well, and the contact-split patient cross-fit signed margin is negative
in all three networks. One patient-training mode also has substantially larger
four-layer error than the other. Thus the problem is not simply absence of
K=2. The model K=2 and the patient K=2 are not yet the same event organization.

The historical field lineage also mixes two ideas that must now be separated.
The runtime representation is continuous, but some selectable ancestors came
from a K=3 warm field, contact-target spline fitting, or a small library of
preselected bumps and donor-field interpolations. Stage-AK then crossed a
small number of frozen mean and dispersion fields; it did not jointly optimize
a free continuous Node field under the corrected event definition.

Rev14 therefore asks a narrower question:

> Can a contact-density-independent, whole-sheet continuous static Node field,
> learned from patient-training events only, make the same Node-only SNN
> express both patient propagation modes as isolated complete causal families?

EE, E-to-I, Z/M and all ictal endpoints remain off until this question is
answered and the Node field is frozen.

## 2. Frozen scientific boundaries

### 2.1 Allowed information during fit

- patient-training contact ranks, recruitment masks and recording blocks;
- the frozen patient-training direction classifier and patient-training noise
  floors;
- model-internal complete causal-family boundaries and whole-sheet onset maps;
- network safety, event support and overlap diagnostics;
- the uniform physical sheet and observation-free Fourier coordinates.

### 2.2 Forbidden information during fit

- patient held-out events, prototypes or scores;
- patient ictal data or Fig.3/Fig.5 bridge endpoints;
- natural model KMeans labels or KMeans quality;
- contact-density maps, shaft-density maps or contact locations as field basis
  functions;
- a fixed number, center, width or identity of Gaussian cores;
- EE, E-to-I or Z/M parameter changes.

Natural KMeans is generated only after a candidate is frozen. Patient held-out
is opened once after model-internal selection and cannot return information to
the fit.

## 3. Formal model event unit

One observation is one returned, source-evaluable, edge-supported directed
causal family. Families are sorted by onset time. Any family belonging to a
strict time-overlap connected component of size at least two is excluded from
the primary score, including transitive overlaps. The all-returned-family score
is retained only as a sensitivity analysis.

An isolated family enters the patient-training loss whenever at least three
contacts have finite onset ranks. OOD, ICL-only, SCL-only and missing-contact
events remain in the loss. They are not deleted by the old Fig.4
`formal_clean` mask. Missing recruitment is a modeled state in the recruitment
and precedence distances.

No fixed requirement such as 20 returned events is a biological gate. Event
support enters as a smooth penalty and an effective-sample diagnostic. A run
with too little support remains a valid poor observation unless it is an
engineering failure.

## 4. Phase 0 static-baseline diagnosis

Before any new simulation, rescore `exact_off` seeds 2311--2313 using the
event unit above. The producer must:

1. load patient-training arrays directly from the frozen training-target NPZ;
2. never materialize patient held-out arrays;
3. retain original NPZ event indices through every mask and contact reorder;
4. score each network separately and aggregate networks with equal weight;
5. report OOD and shaft participation without filtering them;
6. produce a Fig.4-ready bundle from the same isolated readable families;
7. mark natural KMeans and contact-split cross-fit as diagnostics only.

The diagnosis does not rescue rev13 and does not select a new field. It defines
the reference that every rev14 candidate must improve.

## 5. Free-field representation

### 5.1 Primary field

Rev14 returns to one static Node support field and the original signed-depth
mapping:

\[
h_i = \sigma\left(\frac{s(x_i,y_i)-\lambda}{\tau_h}\right),
\qquad
\sum_i h_i = N_{\rm core}^{\rm frozen},
\qquad
V_{\theta i}=V_{\theta0}-h_i d_i .
\]

`d_i`, `tau_h`, the total support mass, `node_gain`, topology, delays, noise and
readout remain frozen. The scalar threshold `lambda` is solved by the existing
mass projection. This is a continuous field and does not assume that a core is
a Gaussian component or that a fixed number of cores exists.

The latent sheet is a paired-phase Fourier field on the full 20 x 20 mm domain:

Let `L=20 mm`, `n=(n_x,n_y)` and

\[
k_n=\frac{\pi}{L}(n_x,n_y).
\]

Use one representative of every nonzero integer-frequency pair `n` and `-n`:
`n_x>0`, or `n_x=0` and `n_y>0`. The circular support is
`n_x^2+n_y^2 <= M^2`. The field is

\[
s(x,y)=\sum_{n\in\mathcal K_M}
\left[a_n\cos(k_n^\top x)+b_n\sin(k_n^\top x)\right].
\]

The physical field is not periodic and the SNN is not made toroidal. The
paired-phase convention uses a 2L period only to avoid forcing opposite sheet
boundaries to match. Modes are selected by a circular frequency support so the
prior is approximately rotation-neutral.

### 5.2 Hierarchical capacity

- M3 is the primary canary and has 28 real coefficients.
- If M3 is negative, only the M4 shell is released; M3 coefficients remain
  frozen. An M3 failure alone cannot reject continuous fields.
- Direct optimization of 18 x 18 spline coefficients is forbidden.
- B-spline coefficients may be used only as a storage/interpolation layer. It
  must satisfy whole-sheet `h` correlation at least 0.995, neuron-level
  `max|delta h| <= 0.01`, `RMS(delta V_theta) <= 0.01 mV`, and relative mass
  error at most `1e-9`.

The field basis is defined on a uniform sheet quadrature and is independent of
electrode coordinates. The patient objective may select coefficients, but it
may not change where basis functions exist or how strongly a location is
represented.

### 5.3 Regularization

Only field-function regularity is allowed:

\[
R_{\rm rough}=\sum_k \lVert k\rVert^4(a_k^2+b_k^2).
\]

There is no penalty on peak count, connected-component count, distance to an
electrode, distance to a shaft or distance to a historical core. Effective
area, peak count and compactness are reported as diagnostics, not optimized
targets. Coefficients are standardized to unit surface RMS within each shell.
`R_rough` is used only as a final tie-break within 1% of the stochastic
objective, not as a term that can automatically suppress the M4 shell.

## 6. Patient-training objective

For patient mode `k`, retain the existing shaft-balanced four-layer distance:

\[
D_k=\frac{1}{4}
\left(E_{{\rm rec},k}+E_{{\rm prec},k}
+E_{{\rm prof},k}+E_{{\rm cloud},k}\right),
\]

Each component is a matched 6-model-event versus 6-patient-event distance,
divided by the q95 of the corresponding patient-training cross-block 6-versus-6
floor. The same sampling and transformation are used on both sides. This is a
block-floor q95 ratio, not an excess-noise unit. Precedence includes the
not-jointly-recruited state, so a missing shaft cannot disappear from the
score. Each network uses 64 frozen bootstrap draws.

The primary exploratory fit objective is:

\[
J_{14}=\operatorname{LSE}_{0.25}(D_A,D_B)
+0.5D_{\rm JS}(\pi_M,\pi_P)
+0.25A_{\rm ambiguity}
+0.5L_{\rm contrast}
+0.5f_{\rm overlap}
+0.25L_{\rm support}.
\]

`f_overlap` is the fraction of returned/evaluable causal families removed by
the overlap-connected rule. For soft mode weights `w_ki`, define

\[
n_{{\rm eff},k}=\frac{(\sum_iw_{ki})^2}{\sum_iw_{ki}^2},
\]

and

\[
L_{\rm support}=
\frac12\sum_{k\in\{A,B\}}\frac{6}{n_{{\rm eff},k}+6}
+f_{\rm source\text{-}non\text{-}evaluable}
+f_{<3\ contacts}.
\]

The two fractions are calculated relative to returned causal families before
overlap removal. For a run with zero scored families, each four-layer mode
distance is set to the frozen missing-mode value 2.0, occupancy is `log(2)`,
contrast loss is 1.0, `n_eff=0` and the objective remains finite. Thus zero
events, source-non-evaluable events and a small set of attractive isolated
events cannot crash the optimizer or evade a penalty. No additional hard
patient-mode gates are added during exploration.

Signed causal displacement, source topology, OOD fraction and shaft support
are reported sidecars. They do not replace the patient-training four-layer
objective.

## 7. Search design

### 7.1 Observation-free M3 canary

Generate eight linearly independent, orthogonalized Sobol directions in the M3
coefficient space. Evaluate both signs at two surface-RMS levels, 0.8 and 1.4,
for 32 selectable fields.
Add two nonselectable benchmarks: uniform Node support and the frozen Stage-AK
`exact_off` field. The selectable fields are not initialized from v62, Stage-AK
or any contact-target fit.

Run all 34 fields on seed 2321 with common random numbers for 20 s. Continue the
best eight selectable fields on seeds 2322--2323. This is an instrument and
capacity canary spanning only an eight-dimensional section of M3, not evidence
of convergence or a negative capacity test.

### 7.2 Local refinement

If at least one M3 field improves the equal-network `J14` reference without
runaway or collapse of either soft mode, initialize a full 28-dimensional M3
CMA-ES search from the best canary field. Use population 16, three generations
and two fixed CRN fit networks, with no restart under the frozen primary
budget. These 96 runs are an
exploratory bounded optimization, not proof of convergence. Do not use natural
KMeans to update it.

If a usable M3 anchor is found but residual structure remains, freeze that
anchor and release only the 20-parameter M4 shell. If M3 does not yield a usable
anchor, freezing M3 and searching only the shell is forbidden; a later joint
M3+M4 search is required. A negative bounded canary is reported as
`NOT_SUPPORTED_WITHIN_FROZEN_M3_M4_BUDGET`, not as failure of all continuous
fields. Do not add Gaussian components or increase spline-grid freedom.

### 7.3 Frozen selection and confirmation

Freeze at most two candidates before new-network selection. Compare the two
candidates and the `exact_off` reference on six fresh networks with common
random numbers. The primary unit is the network; pooled events are display
only.

After selection is frozen, produce the two Fig.4 acceptance figures from the
same isolated families:

1. continuous Node field, one representative natural cluster-1 family, one
   cluster-2 family and the corresponding continuous 15-contact readout;
2. masked rank heatmap, rank distribution, model versus patient-training rank
   profiles and the patient-training contact-split cross-fit matrix.

Only then open the already-developmental patient held-out endpoint once. A
held-out failure cannot be repaired by returning to the field fit.

### 7.4 Fig.4 Node-freeze acceptance

Natural KMeans never enters the fit, but it is a required post-freeze
acceptance test. On every fresh network, run K=2 with eight initial-seed blocks
and require:

- at least six isolated readable families in each natural cluster and minority
  fraction at least 0.20;
- median KMeans seed AMI at least 0.90;
- balanced alignment with the frozen patient-training classifier at least
  0.70;
- positive contact-split cross-fit signed matrix margin;
- both natural clusters in the same network.

K2-versus-K1 held-out GMM density, silhouette and centroid valley gap are
reported diagnostics rather than additional blockers because their finite
sample behavior differs strongly at 20--30 events. At least four of six fresh
networks must satisfy all acceptance criteria above. The candidate must
also improve the weakest patient-training four-layer mode in at least four of
six paired networks without worsening the other mode by more than 10%. Failure
keeps the result at `STATIC_NODE_PATIENT_K2_PARTIAL` and blocks EE/E-to-I/Z/M.

## 8. Causal validation after field freeze

The continuous field has no predefined components. Intervention regions are
therefore defined algorithmically from the frozen field and training-only event
topology, not by a fitted core count. From identical checkpoints branch:

- sham;
- local attenuation of the top field-responsibility patch;
- mass-preserving relocation of that attenuation to a matched low-responsibility
  patch;
- matched random patch control.

Report mode-specific event rate, onset density and four-layer profile changes.
Selective effects support a model-internal role of the learned field location;
they do not identify a patient cellular core.

## 9. Result language

Possible outcomes are:

- `STATIC_NODE_PATIENT_K2_PARTIAL`: model-internal K=2 exists, but weakest
  patient mode, OOD or cross-shaft support remains poor;
- `STATIC_NODE_TRAINING_RECOVERY_CANDIDATE`: both patient-training modes improve
  on most selection networks, before held-out inspection;
- `STATIC_NODE_DEVELOPMENT_CONFIRMATION`: the frozen candidate also survives the
  one-time developmental held-out evaluation and Fig.4 diagnostics;
- `NOT_SUPPORTED_WITHIN_FROZEN_M3_BUDGET`: the bounded M3 search did not yield a
  usable anchor; no shell-only M4 claim is allowed;
- `NOT_SUPPORTED_WITHIN_FROZEN_M3_M4_BUDGET`: a usable M3 anchor was found, but
  the registered M4-shell extension did not produce reproducible two-mode
  improvement. Both states close only their registered budget, not the
  mathematical class of all continuous fields.

None of these states proves blind patient generalization, identifies a
biological core, or authorizes ictal claims. EE, E-to-I and Z/M remain a later
experiment on a frozen Node substrate.
