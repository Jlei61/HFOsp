# Topic 4 rev10-R execution plan

## Scope

- [ ] Freeze rev10-SA V6.2 artifacts and hashes; development Node anchor is `t=0.05`,
  with `t=0.025/0.075` as sensitivities.
- [ ] Mark seeds 1041-1043 as consumed development history; use 1051-1054 for
  fit, 1061-1063 for selection, and 1071-1073 for confirmation.
- [ ] Keep patient held-out, `beta`, topology growth, slow variables, contact-
  conditioned bases, Gaussian components, and peak-count constraints closed.
- [ ] Treat this as exploratory route-capacity work with network seed as the
  independent unit.

## R0 graph-basis audit

- [ ] Build two-sided normalized directed graph SVD coordinates from the frozen
  E-to-E operator only; do not misinterpret target-row normalization as a
  forward transition matrix.
- [ ] Verify that no contact, shaft, patient event, label, field component, or
  field peak enters the builder signature or artifact provenance.
- [ ] Freeze rank `R=4`; use `R=2/6` only as numerical sensitivities.
- [ ] Drop the leading degree mode, standardize retained coordinates to unit
  RMS, and test deterministic sign, subspace-projector stability, and family
  invariance under basis rotation.

## R1 edge mapper

- [ ] Implement target-normalized `U Gamma V^T` edge flow over existing E-to-E
  edges, with a primary `4 x 4` coefficient matrix.
- [ ] Test `Gamma=0` exact no-op, topology/delay preservation, finite positive
  weights, incoming-E error `<=1e-9`, and unchanged E-to-I/GABA hashes.
- [ ] Report edge ratio, KL, ESS, outgoing-source influence, component-free
  graph-mode flow, and effective weighted-delay changes.

## R2 paired exploratory screen

- [ ] Freeze 32 nonzero symmetric scrambled-Sobol fields in rank-4 coefficient
  space, plus `Gamma=0`, with antithetic pairs.
- [ ] Launch one candidate-network per `systemd-run --user -> nohup` worker with
  all numeric thread counts at 1, bounded by measured RSS and available memory.
- [ ] Use common network/noise seeds and the common absolute detector.
- [ ] Score Node and Node+Edge per network with A/B joint+ID support,
  recruitment, three precedence classes, profile, event cloud, and OOD.

## R3 selection and confirmation

- [ ] Freeze at most six diverse fit Pareto fields before reading selection
  networks 1061-1063.
- [ ] Keep equal network weight; do not rank by pooled event count or KMeans AMI.
- [ ] Freeze at most three selection fields before fresh development networks
  1071-1073.
- [ ] Produce a Fig.4-style direct readout/KMeans figure and a network-level
  A/B joint+ID support figure with a Chinese figure README.

## Decision

- [ ] If mode A improves across fresh networks while B is preserved, freeze the
  candidate and run a paired Null/Node/Edge/Node+Edge confirmation.
- [ ] If only event yield rises, report conditional amplification without route
  recovery.
- [ ] If ranks 2/4/6 move edge structure but do not restore shared mode A, stop
  static edge refinement and specify a dynamic-state experiment.
- [ ] Compare CMA-ES/local/Sobol only after a known-good shared solution exists.
- [ ] Open `beta` only for a demonstrated radial-width or delay-scale residual.
