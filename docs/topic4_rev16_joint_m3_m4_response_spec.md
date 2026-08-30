# Topic 4 rev16: joint M3+M4 continuous-Node response search

## 1. Motivation

Rev15 completed the full M3 coordinate-response tensor on three construction
networks and tested 12 frozen response-derived fields on three new networks.
No field passed the joint training-only gate. The nearest field improved the
weak patient mode A on all three new networks, but protected mode B on only one
network. Thus the failure is not a lack of A-directed signal: within M3, that
signal is coupled to a loss of B on unseen networks.

This result closes the local first-order M3 strategy. It does not show that a
continuous field is incapable of supporting both modes. The next minimal
capacity question is:

> Can a continuous Node field that jointly uses the complete M3 basis and the
> next M4 spatial shell improve mode A on every network while preserving mode
> B and both modes' readable support?

This is not a search for multiple explicit cores. The basis is a continuous
whole-sheet Fourier field. K, Gaussian centers and electrode locations do not
parameterize it.

## 2. Frozen scientific boundary

- Node is the only active mechanism. EE, E-to-I and Z/M remain off.
- Total field mass, signed `d_i`, topology, delays, noise, event family and
  20-s duration are unchanged.
- Candidate generation uses only the ordered Fourier basis on the 20 x 20 mm
  sheet. Patient events, contacts, shafts, KMeans, held-out data, figures and
  historical core locations cannot generate candidates.
- Patient-training labels and floors enter only the unchanged full-onset
  `J14_v1` scorer after simulation.
- Existing M3 results are a frozen response tensor. They are not a usable
  anchor and may not be frozen while only M4 is optimized.

## 3. M4-shell atlas

The complete M4 inventory contains 24 paired-phase modes and 48 real
coordinates. M3 contributes 14 modes and 28 coordinates. The new shell is the
set difference, containing 10 modes and 20 coordinates.

For each shell coordinate, set every M3 coefficient and every other shell
coefficient to zero, normalize the continuous latent field to centered
physical-sheet RMS 0.8, and freeze both exact signs. Run the 40 fields on the
same three construction networks, seeds 2331--2333, using common random
numbers. Existing same-seed `exact_off` runs are the paired reference and are
not rerun.

This 120-run atlas measures missing response directions. It cannot itself
select a field or establish two-mode recovery.

## 4. Joint response construction

After all 120 runs validate, concatenate the frozen M3 and M4-shell central
difference tensors into a `3 x 48` response tensor. Construct the same four
families used in rev15, but in the joint space:

1. mean A descent;
2. maximin A descent with B protected on all construction networks;
3. maximin A descent with B and both support gradients protected;
4. sparse coordinates with networkwise sign agreement.

Every constructed vector may contain both M3 and M4 coefficients. A vector
restricted to M4 around a failed fixed M3 field is forbidden. Candidate RMS is
limited to 0.4, 0.6 and 0.8; coefficients are frozen before fresh-network
simulation.

## 5. Fresh-network gate

Use new CRN seeds 2351--2353 and paired `exact_off`. A candidate progresses
only if all three networks satisfy:

- A loss is lower than paired `exact_off`;
- B loss is no greater than 110% of paired `exact_off`;
- equal-network effective support is at least six for A and B;
- no runaway or numerical invalidity.

No gate is relaxed because the M3 result was negative. Natural KMeans and the
two Fig.4 figures remain unavailable until one candidate passes this gate.

## 6. Interpretation

- A passing joint field shows that the missing capacity lay in the next smooth
  spatial shell; it does not identify anatomical cores or a causal patient
  mechanism.
- A field that improves A only by losing B repeats the rev15 tradeoff and does
  not progress.
- If the joint linear-response candidates also fail, close this bounded local
  gradient strategy. The next action is a residual/objective audit, not more
  generations, a relaxed gate, Gaussian cores, EE/E-to-I or Z/M.
- Held-out, source-topology intervention and final Node freeze retain the rev15
  post-selection contract and open only after natural KMeans acceptance.
