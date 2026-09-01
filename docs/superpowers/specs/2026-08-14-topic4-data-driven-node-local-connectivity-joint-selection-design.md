# Topic 4 rev11-NLC3: fresh-network Node-connectivity selection

## Scientific question

The NLC2 fit pool searched a contact-invariant continuous Node residual jointly
with local E-to-E and E-to-I redistribution. NLC3 asks whether its apparent
improvement survives new network realizations, or whether the NLC1 parent
connectivity alone explains the fit-pool result.

This is a development selection experiment. It does not test patient-blind
generalization, complete interictal activity, core causality, or seizure
dynamics.

## Frozen candidate set

NLC3 copies, without interpolation or refitting:

1. the five NLC2 candidates frozen in `fresh_selection_shortlist_ids`;
2. the second NLC1 parent control `joint_03_control` when it is not already in
   the shortlist;
3. the continuous Node-only baseline.

Candidate Node coefficients, local E-to-E/E-to-I coefficients, hashes, and
metadata must be byte-equivalent to the NLC2 manifest. The shortlist is fixed
before the new networks are generated.

## Simulation contract

- six new paired network seeds: 1541-1546;
- 16,000 ms per candidate-network pair;
- one absolute detector threshold for all candidates;
- common network, OU, and Poisson seeds within each paired network;
- Z/M off, beta closed, topology/delays/GABA frozen;
- incoming E weight conserved separately for every E and I target;
- non-returned events excluded from shape scoring but retained in the safety
  audit.

The resource launcher must first measure one worker's peak RSS, then fill only
the memory-bounded number of systemd/nohup workers. Completion produces a
desktop notification.

## Selection readout

The scalar objective is copied unchanged from NLC2 and orders candidates only
after all six networks finish. It jointly protects:

- natural within-network KMeans direction alignment;
- contact-split patient rank geometry;
- the shaft-aware weakest-mode distance;
- mode-conditioned recruitment;
- OOD fraction and detector occupancy;
- perturbation energy.

Network seed is the independent unit. Event-pooled counts are descriptive.
The selection report must show paired network bootstrap intervals against the
Node baseline and both NLC1 parent controls. These intervals describe
uncertainty; they are not extra blockers.

## Decision boundary

- A residual-field candidate that ranks first on fresh networks is eligible
  for a longer frozen-candidate confirmation.
- A parent control that ranks first means local connectivity remains the
  supported substrate, while the added continuous Node residual is not
  confirmed.
- A Node-only winner means this local connectivity family does not replicate
  on the new pool.
- Any late runaway or failed conservation audit invalidates only the affected
  candidate-network pair and prevents that candidate from selection.

The final Fig.4-style waveform and KMeans panels are produced only from the
subsequent frozen-candidate confirmation, not from the NLC2 fit pool.
