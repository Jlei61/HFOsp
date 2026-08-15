# Topic 4 rev11-NLC4: pathway-specific mechanism confirmation

## Scientific question

The frozen rev11-NLC substrate already has an exact paired four-arm decomposition:
Node, Node+E-to-E, Node+E-to-I and Node+both. The previous confirmation selected
and judged a composite score; mode-specific event yield and pathway dynamics were
not frozen endpoints. This experiment asks a narrower question on a new network
pool:

> Does E-to-E redistribution preferentially alter propagation organization while
> E-to-I redistribution preferentially alter TA-like/TB-like accessibility and
> inhibitory recruitment?

This is a patient-development mechanism experiment. It does not refit the Node
field or edge coefficients, and it is not a patient-blind test.

## Frozen substrate and arms

All four candidates are copied byte-for-byte from the accepted NLC3C manifest.
They share the same continuous Node field, network topology, delays, GABA,
detector and spatial OU process.

1. `node_baseline`: Node only;
2. `joint_04_ee_only`: Node plus the frozen E-to-E row;
3. `joint_04_etoi_only`: Node plus the frozen E-to-I row;
4. `joint_04_control`: Node plus both frozen rows.

The E-to-E and E-to-I incoming budgets remain separately conserved for every
postsynaptic target. Z/M and beta remain off. No coefficient search is allowed.

## Independent execution

- network seeds: 1581-1592;
- duration: 20,000 ms per arm-network pair;
- paired network, Poisson and spatial-OU seeds across arms;
- frozen absolute detector threshold: 0.0195703125;
- memory-bounded systemd/nohup execution with a measured-RSS sentinel;
- network seed is the independent unit.

The only hard invalid conditions are nonfinite output, structural-contract
violation, late runaway, missing pathway trace or stale provenance. Zero events
and missing modes are valid scientific outcomes.

## Frozen event labels

The patient-trained direction classifier is copied without refitting. Display
labels are `TA-like` for classifier label 0 and `TB-like` for label 1. Natural
KMeans clusters, when shown, remain `C1/C2`; they are not renamed TA/TB.

A mode-specific event is counted when it is:

- detector-qualified and returned;
- jointly recruits ICL and SCL;
- inside the frozen patient support;
- readable on at least three contacts.

Per network and arm, report TA-like count, TB-like count, total clean-event rate,
TB-like fraction and OOD fraction. Counts are normalized by the fixed 20 s
duration, not by the number of detected events.

## Pathway recorder

The simulator records group-level summaries every 1 ms without storing
per-neuron current traces:

- population E rate;
- population I rate;
- mean recurrent E current onto E targets (E-to-E readout);
- mean recurrent E current onto I targets (E-to-I readout);
- mean GABA current onto E targets.

Each clean event is aligned to detector onset on `[-20, +60] ms`. Currents and
rates are baseline-corrected using `[-20, -5) ms`. Frozen summary windows are:

- ignition: `[0, 10) ms`;
- local recurrence: `[10, 25) ms`;
- downstream relay: `[25, 50) ms`.

Event curves are averaged within network and mode first, then networks receive
equal weight. A network with no event in one mode contributes zero to the rate
endpoint and is absent only from that mode's event-aligned curve.

## Planned contrasts

For every endpoint, report paired network differences for:

- E-to-E only minus Node;
- E-to-I only minus Node;
- both minus Node;
- factorial interaction: both - E-to-E - E-to-I + Node.

Use 4,096 paired network-bootstrap draws and report the observed mean difference,
90% interval and directional probability. These are estimation endpoints, not a
stack of pass/fail gates.

The intended interpretation is supported only if the pattern is pathway-specific:

- E-to-E changes natural direction alignment or mode-conditioned propagation
  shape more than it changes TB-like occupancy;
- E-to-I changes TB-like rate/fraction and event-aligned I/GABA recruitment;
- the joint arm may be additive, redundant or antagonistic; synergy is not
  assumed.

If confidence intervals remain broad or effects do not separate by pathway, the
result is `PATHWAY_ATTRIBUTION_UNRESOLVED`, not evidence that the optimizer failed.

## Figure contract

The main candidate is a compact Fig.4C pathway-ablation panel. It contains four
short axes: `TA-like events`, `TB-like events`, `direction` and `OOD`. It shows
all 12 paired network values plus equal-network means and 90% bootstrap intervals.
The arm order is Node, +EE, +E-to-I, +both. No prose conclusion, status banner or
internal panel letter is drawn on the canvas.

The already accepted direct-readout and KMeans figures are not modified. The
existing 1561-1572 ablation can produce a clearly marked post-hoc candidate now;
the paper-facing panel must switch to the 1581-1592 frozen-endpoint result after
that run completes.

## Claim boundary

This experiment can identify a static-pathway mechanism candidate within the
frozen patient-development SNN. It cannot establish patient-blind generalization,
clinical waveform equivalence, an anatomical core, seizure entry/exit, or a
universal role for E-to-E or E-to-I connectivity.
