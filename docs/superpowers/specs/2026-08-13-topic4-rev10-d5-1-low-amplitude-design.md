# Topic 4 rev10-D5.1: low-amplitude continuous accessibility bracket

## Scientific question

D5 showed that continuous, observation-invariant afferent-rate heterogeneity can make both frozen A and B routes spontaneously accessible, but the lowest tested amplitude also increased detector occupancy substantially. D5.1 asks for the smallest tested amplitude that preserves same-network A/B access without changing the frozen Node field, edge topology, detector, classifier, or patient target.

This is not a search over cores. The spatial process is a zero-mean continuous OU field defined over the full neural sheet. Its law is translation invariant and uses no contact coordinates, shaft labels, patient events, D4 source locations, Node-field peaks, or Gaussian components. The matched permutation arm preserves the exact per-update rate-value multiset and OU trajectory while destroying spatial adjacency.

## Frozen bracket

- local and matched-permuted modes;
- `sigma_rate_per_ms = {0.1, 0.2, 0.35}`;
- `ell = 0.38 mm`, `tau = 20 ms`, update every `1 ms`;
- exact-off baseline;
- D5 fit networks `1271-1273` only;
- common detector and returned-only scoring;
- all static edge coefficients exactly zero.

The selection rule is fixed before execution: choose the smallest local amplitude with no runaway and clean A+B support in the same network in at least `2/3` fit networks. Do not select by the minimum objective score. Report event burden, fraction of time above the detector, returned fraction, peak active fraction, OOD, and the matched-permutation result. These are Pareto diagnostics, not retrofitted blockers.

## Decision

- A survivor is frozen with its matched permutation for fresh-network confirmation.
- No survivor closes this low-amplitude bracket; it does not authorize more cores, beta, or topology growth.
- The bracket itself is development-only and cannot replace Fig.4 acceptance.

## Fig.4 boundary

Only a later frozen run on unseen networks `1291-1296` may produce the accepted Fig.4 pair: direct model waveforms and KMeans/patient-mode consistency. Both modes must occur spontaneously within the same networks. A pooled split across different networks is insufficient.
