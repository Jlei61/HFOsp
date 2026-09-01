# Topic 4 rev20-DC: frozen dual-core mechanism atlas

## 1. Motivation

The accepted starting point is no longer a free Node field. It is the frozen
binary field `dualcore_s39`: two fixed centers, 1,499 selected E neurons and the
historical signed-depth mapping `delta_Vtheta_i = -h_i d_i`. Under the older
detector-fragment readout this field produced both patient-supported directions
in 12/12 fresh networks, but it did not recover the full patient distribution
and about 46% of returned fragments were OOD.

The later event audit changed the formal observation unit. A threshold fragment
may be only one part of a longer causal event, and one physical event must not be
split into apparent A and B samples. The formal unit is now one complete,
returned, temporally isolated, edge-supported causal family. Therefore the old
dual-core result is a strong substrate candidate, not a valid response surface
for the next experiment.

This revision asks:

> With the geometry of `dualcore_s39` frozen, which low-dimensional Node and
> connectivity parameters actually control recovery of the patient's complete
> interictal event distribution, its two directional templates and the fraction
> of model events outside patient support?

No continuous/free field is fitted. Z/M and ictal targets remain off.

## 2. Frozen substrate

- field: two binary cores centered at `(1.53773425799, 1.22641798807)` and
  `(18.60661213705, 2.41771341441)` mm;
- reference Node budget: 1,499 E neurons, selected by distance to the nearer
  center with neuron-index tie breaking;
- reference Node mapping: `node_gain=1`, `signed_depth_shrinkage=1`;
- reference learned pathway expression: `g_EE=0.5`, `g_EtoI=1.0`;
- background spatial OU, detector, contacts, noise, delays and all non-varied
  parameters are frozen;
- Z/M is off for every run;
- patient ictal data are forbidden.

`Both` is the joint expression of the learned EE and E-to-I coefficient rows.
It is not a fourth pathway and is the formal meaning of the historical `BOS`
label.

## 3. Formal event unit

One observation is one complete returned edge-supported causal family. Detector
fragments only establish observability and cannot define family boundaries.
Strictly overlapping families are removed from the primary distribution score
as an overlap-connected component. All remaining returned families enter the
contact distribution, including zero-to-two-contact, single-shaft and OOD
families. Missing contacts are explicit states and are never dropped.

Natural KMeans uses only the readable subset after the family boundaries are
frozen. It cannot alter the event windows or the primary event population.

## 4. Parameter families

Exactly one coordinate changes from the reference unless the family is
explicitly `Both`.

| Family | Levels | Meaning |
|---|---|---|
| Node gain | 0.75, 0.875, 1.0, 1.125, 1.25 | scalar strength of `-h_i d_i` |
| Core budget | 0.75, 0.875, 1.0, 1.125, 1.25 x 1,499 | radius/number of binary Node neurons; centers fixed |
| Signed-depth heterogeneity | 0.0, 0.5, 1.0 | shrink `d_i` toward its h-weighted mean |
| EE expression | 0, 0.25, 0.5, 0.75, 1.0 | learned EE coefficient-row dose |
| E-to-I expression | 0, 0.5, 1.0, 1.25, 1.5 | learned E-to-I coefficient-row dose |
| Both expression | 0, 0.5, 0.75, 1.0, 1.25 | common multiplier of `(0.5, 1.0)` |
| EE ellipse angle | 0, 22.5, 45, 67.5, 90 degrees | long-axis direction of fixed-topology EE weight redistribution |
| EE aspect ratio | 1, 1.5, 2, 3, 4 | long/short scale ratio of fixed-topology EE weight redistribution |

The primary ellipse experiment preserves realized topology and delays. Existing
EE weights are multiplied by the ratio of the requested elliptical kernel to
the reference `(45 degrees, AR=2)` kernel and renormalized separately for every
E target. Thus the reference is an exact no-op and incoming EE budget is exact.
This tests effective recurrent geometry on the same graph. A small rebuilt-
topology sensitivity may be run later, but it cannot be pooled with the primary
fixed-topology contrasts.

## 5. Separated endpoints

### 5.1 Development selection endpoint

The only patient-dependent selection endpoint is an unconditional complete-
event distribution distance. For every formal model family, use the frozen
shaft-aware fixed-contact feature

`[recruitment mask, mask * normalized onset, ICL/SCL participation,
cross-shaft first-onset offset and validity]`.

Transform model events with the patient-training-only standardization, PCA and
fixed sliced-Wasserstein directions. Compare all model families with all
patient-training events without A/B labels. Report the raw distance and a
patient block-resampling floor matched to the model event count. Safety,
returned-family yield and overlap are model-internal sidecars.

No KMeans label, patient mode proportion, OOD threshold, held-out event or
patient ictal value may select a level.

### 5.2 Frozen validation endpoints

After a level is selected within each family using only the endpoint above:

1. **Held-out complete-distribution distance**: the same unconditional feature
   and projection contract against frozen recording-block held-out events.
2. **Two-template concordance**: natural model K=2 on all readable formal
   families, aligned to the frozen patient direction classifier. Report
   direction-balanced alignment, same-network two-cluster support and the full
   2 x 2 prototype correlation matrix. KMeans is not run only on in-support
   events.
3. **OOD fraction**: all returned isolated causal families are the denominator;
   unreadable events count as OOD, with unreadable and readable-support OOD
   shown separately.

The screen is aggregated twice from identical immutable worker artifacts. The
first pass cannot load the classifier or held-out endpoint and freezes one
level per family. Only after the selection JSON records the sealed aggregate
hash may the second pass expose the all-level validation curves. No selected
level can be changed after this opening.

The old mode-conditioned J14 components remain diagnostic. They cannot replace
the unconditional distribution endpoint or select a parameter.

## 6. Statistical unit and interpretation

The independent unit is the network seed, not an event. All contrasts are
paired to the reference candidate within seed. Screen summaries use equal
network weights. Confirmation uses paired network bootstrap intervals. Family
curves are exploratory; no collection of hard biological gates is added.

A parameter is called influential only when it changes the training
distribution monotonically or reproducibly across paired networks and the
direction is retained on the frozen validation endpoints. A lower OOD obtained
by suppressing events, collapsing one direction or making events unreadable is
reported as a tradeoff, not an improvement.

## 7. Execution stages

1. Rebuild `dualcore_s39` under the corrected causal-family producer on three
   fresh canary seeds.
2. Verify exact Node mask/hash reconstruction, reference ellipse no-op,
   per-target incoming budget conservation, event-family invariants and output
   provenance.
3. Render the Fig.4-style baseline KMeans/GIF audit from the same canary files.
4. If the baseline is scientifically evaluable, run the 31-candidate screen on
   four paired seeds.
5. Select at most one non-reference level per family using only the training
   complete-distribution endpoint, freeze it, and run 12 fresh paired seeds.
6. Open validation endpoints once, aggregate the response atlas and render the
   final mechanism figure.

There is no hard requirement for 20 events. Zero or low event yield is a valid
poor model result, not missing data. The canary stops only for engineering
failure, late runaway, or a baseline with no estimable complete-event
distribution in at least two of three seeds, because in that case a large
response scan would have no calibrated reference.

## 8. Claim boundary

This is a development-only mechanistic sensitivity analysis in one patient and
one SNN family. It can identify which model coordinates control patient-like
interictal distributions under the frozen dual-core hypothesis. It cannot prove
anatomical cores, identify biological synaptic changes, establish patient
generalization or make an ictal claim.
