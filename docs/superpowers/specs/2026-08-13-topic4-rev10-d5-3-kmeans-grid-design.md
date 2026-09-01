# Topic 4 rev10-D5.3: canonical KMeans-guided continuous OU grid

## Scientific question

D5.2 established a narrower result than full patient recovery. On six fresh networks, the frozen continuous Node plus spatial-OU mechanism produced both supervised direction modes in every network and recovered a positive-diagonal, negative-crossed model-patient rank matrix. However, canonical Fig.4C masked-rank KMeans separated those directions with purity `0.674`, below the patient matched `q05=0.884`. The remaining question is whether this weak natural discreteness is caused by the chosen OU dose and temporal persistence, rather than by the optimizer, a missing core count, or a new edge family.

This experiment does not optimize the Node field and does not add cores. The frozen `h(x,y)` remains the continuous observation-invariant field from v6.2. The accessibility process remains a zero-mean stationary continuous random field over the entire sheet. Its law uses no contact positions, shaft identities, patient labels, D4 source points, Node peaks, or Gaussian components.

## Frozen grid

- `sigma = {0.025, 0.05, 0.075, 0.1}/ms`;
- `tau = {10, 20, 40} ms`;
- spatial correlation length `ell = 0.38 mm`;
- exact-off reference;
- three new development networks `1301-1303`;
- common absolute detector, returned-only events, frozen direction classifier;
- all static edge coefficients are exactly zero; `beta` and topology remain closed.

The `sigma=0.1/ms, tau=20ms` cell is the prior D5.2 anchor on new networks. The lower amplitudes test whether fewer ambiguous events sharpen the natural direction clusters. The time constants test whether brief versus persistent accessibility patches determine whether an event commits to one route or mixes both.

## KMeans selection contract

Canonical KMeans uses only fixed-contact, masked, within-event normalized ranks. It does not use the full shaft-aware PCA representation that was previously shown to cluster recruitment extent instead of direction.

A candidate is evaluable when it has no runaway and at least two networks each contain at least three returned, dual-shaft, patient-support events from both supervised directions. This is a statistical support requirement, not a biological pass/fail gate.

For each evaluable candidate, a hierarchical balanced bootstrap draws up to six events per direction from every eligible network. The primary quantity is median label-swap-invariant KMeans direction purity. The model-patient matrix is computed independently from frozen supervised A/B labels, and contributes through

```text
signed_margin = min(r_AA, r_BB, -r_AB, -r_BA).
```

The frozen continuous selection score is

```text
J = (1 - median_purity)
    + 0.125 * (1 - signed_margin)
    + 0.10 * OOD_fraction
    + 0.05 * detector_occupancy.
```

The smallest `J` is selected. Patient `q05` is reported but is not a hard blocker in this exploratory grid. This avoids manufacturing many retrospective gates while still making KMeans and patient consistency the main optimization target.

## Interpretation and next action

- If an evaluable candidate improves the D5.2 purity while preserving the signed patient matrix, freeze it before using seeds `1311-1313`, then confirm only the survivor on `1321-1326` with exact-off and exact-marginal permutation controls.
- If no candidate improves the D5.2 anchor, the result argues against simple OU dose/persistence as the source of the mode-A ambiguity. It does not prove that another optimizer would fail, because no high-dimensional optimizer is being tested here.
- Only a later fresh confirmation may replace the current Fig.4 pair. The accepted pair remains direct same-network A/B waveforms and canonical KMeans/patient consistency; grid figures are diagnostic only.

## Claim boundary

This remains patient-development work. It cannot establish patient-blind generalization, clinical waveform reproduction, full interictal-distribution recovery, or a causal core. It specifically tests whether continuous, observation-invariant stochastic accessibility can sharpen the two propagation directions already accessible in the frozen SNN substrate.
