# Topic 4 rev11-NLC3C: frozen-substrate final confirmation

## Scientific question

Fresh-network selection retained `joint_04_control`: the continuous Node anchor
with frozen local E-to-E and E-to-I redistribution, but without the NLC2 Node
residual. NLC3C asks whether this static substrate produces a same-network
natural two-mode repertoire with patient-consistent rank geometry on a final,
longer and disjoint network pool.

No parameter is fitted or selected in this phase. The patient target remains a
development target, not a blind unit.

## Frozen four-arm decomposition

All arms use the same continuous Node field and differ only in which rows of
the selected two-by-six edge coefficient matrix are active:

1. `node_baseline`: neither pathway;
2. `joint_04_ee_only`: selected E-to-E row only;
3. `joint_04_etoi_only`: selected E-to-I row only;
4. `joint_04_control`: both selected rows.

The joint candidate is copied byte-for-byte from NLC3 selection. Single-pathway
arms are deterministic mechanism ablations, not newly optimized candidates.

## Simulation and engineering contract

- final network seeds 1561-1572, disjoint from NLC1/NLC2/NLC3;
- 20,000 ms per arm-network pair;
- paired network, OU and Poisson seeds across all four arms;
- one frozen absolute detector;
- Z/M off, beta closed, topology/delays/GABA frozen;
- incoming E budget conserved separately for every E and I target;
- late runaway and structural-contract violations invalidate an arm-network
  result;
- a measured-RSS sentinel determines memory-bounded systemd/nohup concurrency;
- completion emits a desktop notification.

## Primary readout

Network seed is the independent unit. The final report retains the NLC2 scalar
for paired ordering but adjudicates three interpretable statements separately:

1. `DIRECTIONAL_REPERTOIRE`: the joint arm has evaluable natural KMeans in at
   least 10/12 networks and the 90% network-bootstrap lower bound of balanced
   patient-direction alignment exceeds 0.5;
2. `PATIENT_GEOMETRY`: the 90% lower bound of contact-split patient rank margin
   is above zero;
3. `EDGE_INCREMENT`: the paired-network bootstrap probability that the joint
   scalar is lower than Node is at least 0.90.

Held-out K=2 minus K=1 GMM log likelihood, silhouette, cluster counts and AMI
are mandatory diagnostics, not extra blockers. The E-to-E-only and E-to-I-only
arms identify whether the joint effect is additive, redundant, synergistic or
antagonistic on natural alignment, patient margin, shaft-aware loss and OOD.

## Figure contract

Only NLC3C data can produce the two canonical final figures:

1. direct model-current waveforms with amplitude/time scales and mode shading;
2. natural within-network KMeans with model rank profiles, patient prototypes,
   per-network stability and patient-geometry matrix.

Both PNG and PDF are required, with a Chinese `figures/README.md`. Figures must
state that the signal is a 30-80 Hz model-current proxy, not clinical SEEG HFO.

## Claim boundary

A full three-part pass supports a patient-development, data-driven static
Node-plus-local-connectivity substrate. It does not establish patient-blind
generalization, a causal anatomical core, clinical waveform equivalence or an
ictal lifecycle. Z/M is considered only after this substrate and its hashes are
frozen; Z/M failure is then a seizure-interface failure, not an interictal
refit.
