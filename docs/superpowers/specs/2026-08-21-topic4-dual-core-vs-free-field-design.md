# Topic 4: hand dual core versus continuous free field

## Motivation

The current Figure 4 candidate uses a patient-fitted continuous Node field. A
hand-placed two-core field is scientifically simpler and visually more direct.
The historical comparison is not decisive because it mixed older single-axis
readouts, different event budgets, and earlier field families. This experiment
asks a narrower question under the current interictal contract:

> With identical Node mechanism, network realizations, background drive,
> detector, event readout and excitability budget, does the continuous free
> field explain the real patient interictal event distribution better than two
> manually placed cores?

This is a representation comparison, not a new fit and not a connectivity or
Z/M experiment.

## Frozen arms

### Continuous free field

Reuse the existing `node_baseline` artifacts from the final 12-network Figure 4
confirmation. The field is `v62_density_t050`; learned E-to-E and E-to-I edge
coefficients are zero, Z/M is off, and the fixed local spatial OU drive remains
active.

### Hand dual core

Use the two historically hand-placed centers:

- source core: `(4.19921432, 9.12890135)` mm;
- sink core: `(16.47920304, 3.96551153)` mm.

The original 1.5 mm radius covers a different number of E neurons in each
random network. To isolate geometry from total excitability, the primary arm
selects exactly the 1129 E neurons nearest to either frozen center. Its field is
binary. The effective cutoff radius and per-core allocation are recorded for
every network. The historical 1.5 mm count is reported as a geometry
sensitivity only and is not used for the primary score.

The hand arm otherwise copies `node_baseline` exactly: the same signed `d_i`,
same topology, delays, GABA, zero edge coefficients, spatial OU parameters,
network seeds, Poisson/OU seeds, detector and 20 s duration.

## Independent unit and event contract

- paired network seeds 1561-1572 are the independent units;
- the continuous arm reuses its immutable existing workers;
- only the hand arm is newly simulated;
- events must be detector-qualified and return to baseline;
- fixed patient direction assignment and OOD thresholds are used;
- no minimum of 20 returned events is imposed;
- the primary balanced comparison uses six events per mode per network;
- three events per mode is a predeclared sensitivity;
- unavailable network-mode cells remain unavailable rather than being pooled
  across networks.

## Patient distribution target

The fixed 15-contact identity and shaft-aware representation are primary. The
patient recording-block split from rev10-SA is rebuilt exactly. Model fields
were already frozen before this comparison, so held-out recording blocks are
used as the primary descriptive target and training blocks as continuity
sensitivity. Because these patient data have been inspected in earlier Topic 4
rounds, this remains development-only rather than a new blind test.

For each patient mode and each model network, report:

1. ICL and SCL recruitment probability error;
2. ICL-ICL, SCL-SCL and ICL-SCL precedence-distribution error;
3. separate ICL, SCL and cross-shaft profile error;
4. sliced-Wasserstein event-cloud distance in the patient-training embedding;
5. multishaft participation and SCL recruitment;
6. OOD fraction and mode support;
7. within-network natural KMeans agreement with the frozen patient direction;
8. mode-proportion Jensen-Shannon distance.

The four distribution families are shown separately. A floor-normalized
composite is secondary and cannot hide a failure of one shaft or one mode.

## Statistics

Network seed is the independent unit. Field differences are paired within
seed. Use 4096 paired network-bootstrap draws and report the median difference,
90% interval and probability that the continuous field has lower error. Event
resampling occurs within each network and mode. Patient uncertainty is sampled
by recording block, not by treating 30,000 events as independent.

No star or pass/fail label is added merely from pooled-event tests. A Wilcoxon
paired test may be reported as a compact diagnostic, but the effect size and
network-bootstrap interval carry the result.

## Interpretation

- Continuous lower across held-out recruitment, precedence and event-cloud
  metrics: the free field contributes explanatory power beyond two fixed cores.
- Similar performance with intervals spanning zero: the simpler dual-core
  representation is not distinguishable at 12 networks.
- Dual core lower on the patient target: the current free field is not
  justified by explanatory power and should not be presented as necessary.
- Tradeoff by mode or metric: neither field dominates; Figure 4 should present
  the free field as one development candidate, with dual core as a serious
  alternative.

No outcome establishes anatomical cores, patient-blind generalization,
clinical waveform equivalence or ictal dynamics.

## Required artifacts

- frozen config and candidate manifest;
- 12 hand-arm JSON/NPZ workers with hashes;
- source hashes for all reused continuous-arm workers;
- per-network and aggregate CSV/JSON metrics;
- a compact Nature-style comparison figure and a Fig.4-style KMeans figure;
- Chinese `figures/README.md`;
- implementation and scientific audit with an explicit safe claim.
