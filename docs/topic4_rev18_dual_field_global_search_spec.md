# Topic 4 rev18: robust multi-coordinate continuous dual-Node search

## 1. Scientific motivation

The rev17 exact dual-field anchor already generates stable natural `K=2`
partitions in the same fresh networks. Its KMeans balanced alignment with the
frozen patient direction labels was `1.00 / 0.85 / 0.96` on seeds
2371--2373. The remaining failure is different: many model events lie outside
patient support, weak mode A has few effective events, and the complete event
distribution remains too far from the patient target.

Rev17 perturbed one cosine coordinate at a time. Eighteen endpoints improved
the three fit networks, but none improved every fresh network. This is a
winner-selection and representation problem: a thresholded event process is
not locally linear enough for one-coordinate finite differences to provide a
reliable gradient.

Rev18 therefore searches multi-coordinate continuous fields directly. It does
not add Gaussian components or assign one core to each mode. `K=2` refers only
to event clustering; the Node substrate remains a pair of continuous spline
fields over the complete sheet.

## 2. Frozen Node mechanism

The exact rev17 anchor and mapping remain the zero point:

`Delta Vtheta_i = -h_mean,i mu_mean - h_disp,i (d_i - mu_disp)`.

Network topology, delays, incoming budgets, signed depth draw, noise law,
causal event segmentation and virtual-contact readout remain fixed. Learned
E-to-E redistribution, learned E-to-I redistribution and Z/M are exactly off.

## 3. Free-field coordinates

Each field receives a residual in the same 15-mode, uniform-sheet cosine basis
used by rev17. The 30-dimensional direction contains 15 mean-field and 15
dispersion-field coefficients. The basis is generated without contact,
shaft, patient-mode, manual-core or ictal coordinates.

Twenty-four scrambled Sobol directions are frozen before simulation. Each is
run with its antithetic sign; radii cycle through `0.08, 0.15, 0.25` sheet-RMS
units. Three rev17 directions are retained as sentinels, not as preferred
candidates. Together with the exact anchor, the screen contains 52 fields.

## 4. Fit simulation

All 52 fields run Node-only for 20 s on common fit networks 2391--2393. A late
runaway is invalid. One edge-supported causal root family remains one event;
contact observations cannot split or merge causal families.

For each network the screen reports:

- complete patient-training `J14`;
- mode A and B recruitment, precedence, profile and event-cloud losses;
- effective support and OOD fraction;
- natural masked-rank `K=2`, cluster sizes and alignment to the frozen patient
  direction classifier.

Natural KMeans is now an explicit fit protector because rev17 proved that the
anchor already possesses the desired same-network two-cluster repertoire.
Patient held-out blocks, source topology, intervention, figures and ictal data
remain closed.

## 5. Frozen robust objective

Each endpoint is computed per network. `J14` and weak-mode A are normalized to
the paired exact anchor. Mode B is penalized only beyond a 10% increase.
Effective support below six, a natural cluster below six events, KMeans
balanced alignment below 0.8, and OOD fraction enter continuous penalties.

The candidate loss is the equal-network mean plus `0.5` times the network
standard deviation and `0.25` times the worst-network value. This protects
against another field that wins only on one network. The six lowest-loss valid
candidates are nominated without additional post-hoc endpoint gates.

This screen is exploratory nomination, not Node acceptance. Fresh selection,
unseen confirmation, complete held-out event distribution, weakest-mode
improvement, source topology and same-checkpoint hotspot intervention remain
required before Node can freeze.

## 6. Interpretation boundary

Success in rev18 would show that an observation-invariant continuous Node
field can improve patient-training event distributions while retaining the
same-network two-mode repertoire. It would not identify biological cores,
establish patient-blind generalization, or justify changing EE, E-to-I or Z/M.
