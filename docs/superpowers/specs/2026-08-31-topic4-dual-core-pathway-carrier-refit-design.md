# Topic 4 rev17: frozen dual-core pathway refit and carrier calibration

## Motivation

Rev16 found a simple two-core Node field that produced both frozen patient propagation modes in 12/12 new networks, but 46% of returned events remained outside patient support. The pathway experiment then transferred EE and E-to-I coefficient rows learned on an older continuous field. It did not optimize either row on the new dual-core field. The transferred EE row reduced OOD mainly by suppressing event yield, the transferred E-to-I row shifted occupancy toward the patient's larger mode without reducing OOD, and their combination was antagonistic rather than additive.

The representative GIF also exposed a separate issue. Its apparent slow motion is a rendering effect: 5 ms simulation frames are shown at 8 fps. Absolute contact recruitment spans overlap the patient distribution. The stored firing-density envelope is dominated by about 23 Hz, but it has already undergone 5 ms Gaussian smoothing, which retains only about 17% of a 60 Hz amplitude. Raw spike-rate and synaptic-current carrier content therefore remain unaudited. The current OOD embedding also normalizes each event to 0--1 and cannot see either absolute duration or carrier frequency.

Rev17 therefore keeps the Node substrate fixed and separates three questions:

1. Do the two static cores make distinct contributions to the two event modes?
2. Can EE and E-to-I expression be recalibrated on this Node field to lower OOD without collapsing event yield or natural K=2 structure?
3. Can the same interictal work point retain patient-like recruitment timing and produce a native fast carrier, rather than only a bandpass visualization?

Z/M remains off. Ictal transition is a later experiment after the interictal substrate is frozen.

## Frozen Node and patient target

- Freeze `dualcore_s39`, including both centers, the 1,499-node binary mask, per-neuron threshold shifts, detector, virtual contacts, patient classifier and OOD thresholds.
- Both cores are present from time zero. Their confirmation-network populations are 752 and 747 E neurons.
- Keep topology, delays and per-target incoming pathway budgets fixed.
- Continue to count unreadable returned events as OOD.
- Do not use patient ictal data or Z/M outcomes for selection.

## Core participation audit

First reuse the stored 5 ms activity grids to estimate, for every supported event, the first activation time and early activity mass in each core. Report `P(core first | mode)` and its network-bootstrap interval.

If an association is present, run a minimal paired causal audit on three common network seeds:

```text
both cores
core 1 lesion with matched total-threshold control
core 2 lesion with matched total-threshold control
```

The audit is descriptive. A mode-specific core interpretation requires selective loss or latency shift of one mode; a global event-rate decrease alone does not establish it.

## Joint pathway response surface

The first pass keeps the two previously learned coefficient directions but refits their expression on the frozen dual-core Node:

```text
g_EE   in {0.00, 0.25, 0.50, 0.75, 1.00}
g_EtoI in {0.00, 0.50, 1.00, 1.25}
```

This is 20 candidates on three common-random-number network seeds for 8 s each. The existing four arms are embedded in the surface. Coefficient clipping, topology, delays and target-wise incoming budgets remain unchanged and are audited for every candidate.

Selection uses a frozen continuous development score:

```text
J_interictal = OOD_all_returned
             + 0.25 * abs(mode2_fraction - 0.691)
             + 0.20 * (1 - natural_KMeans_alignment)
             + 0.15 * abs(log(event_rate / Node_event_rate))
             + 0.15 * D_absolute_timing
```

`D_absolute_timing` is the mean absolute log ratio of model and patient median recruitment spans for the two frozen modes. All five components and the Pareto surface are reported; the composite score is not a new patient noise-floor distance.

The best four candidates then run on three new 12 s selection seeds. One work point is frozen before 12 fresh 20 s confirmation seeds.

## Fast-carrier branch

Carrier calibration starts only after the pathway work point is frozen.

1. Verify the raw-rate readout on a synthetic 62.5 Hz signal and on an archived SNN trajectory previously claimed to show a fast carrier. Failure means the readout must be repaired before simulation.
2. Record 1 ms population E/I rates and a current-based readout before the existing 5 ms observation smoothing. Also retain the 2 ms/5 ms virtual-contact firing-density envelope for exact Fig.4 display parity. A 30--80 Hz filtered trace remains a visualization, not the primary carrier endpoint.
3. If the frozen work point remains low-frequency, scan only:

```text
tau_d_AMPA in {2.0, 3.5} ms
tau_d_GABA in {8, 12, 18} ms
```

on three common seeds. Do not change Node, pathway coefficients or Z/M in this scan.
4. Report peak frequency, 20--150 Hz centroid, 30--80/5--30 power ratio, OOD, event yield, mode proportion and absolute timing for every cell. A fast carrier is acceptable only if the interictal repertoire remains comparable; frequency alone cannot rescue a high-OOD work point.

This is a model gamma-like carrier calibration, not a claim that the firing-density envelope is clinical HFO or SEEG.

## Visual contract

- GIF time labels state model milliseconds and playback slowdown.
- The formal lower panel uses the accepted Fig.4 fourth-order 30--80 Hz zero-phase filter and common amplitude scale, and is labeled as filtered firing-density activity.
- The GIF metadata states that its two onset-aligned event windows are not the same as Fig.4's single continuous two-event window.
- Mode examples remain algorithmically selected, and all-event OOD/KMeans statistics remain adjacent to the GIF.

## Claim boundary

Allowed: a frozen dual-core Node plus recalibrated local EE/E-to-I redistribution can be assessed for patient-support occupancy, two-mode organization, absolute recruitment timing and a model-internal fast carrier.

Not allowed: two anatomical patient cores, one-to-one core-mode causality before lesion evidence, clinical HFO reproduction, or patient causal identification of EE/E-to-I.
