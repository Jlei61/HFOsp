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
difference tensors into a `3 x 48` response tensor. Construct six
training-only direction families in the joint space. Four preserve the rev15
diagnostics and two explicitly align candidate construction with the already
frozen complete `J14_v1` objective:

1. mean A descent;
2. mean `J14_v1` descent;
3. maximin A descent with B protected on all construction networks;
4. maximin A descent with B and both support gradients protected;
5. maximin `J14_v1` descent with A, B and both support gradients protected;
6. sparse coordinates with networkwise sign agreement.

Every constructed vector may contain both M3 and M4 coefficients. A vector
restricted to M4 around a failed fixed M3 field is forbidden. Candidate RMS is
limited to 0.4, 0.6 and 0.8; coefficients are frozen before fresh-network
simulation.

## 5. Fresh-network gate

Use new CRN seeds 2351--2353 and paired `exact_off`. A candidate progresses
only if all three networks satisfy:

- A loss is lower than paired `exact_off`;
- complete `J14_v1` loss is lower than paired `exact_off`;
- B loss is no greater than 110% of paired `exact_off`;
- equal-network effective support is at least six for A and B;
- no runaway or numerical invalidity.

No gate is relaxed because the M3 result was negative. Natural KMeans and the
two Fig.4 figures remain unavailable until one candidate passes this gate.
Among usable candidates, rank first by worst-network and then mean-network
`J14_v1` change, before A/B tie-breaks. This prevents a field that improves a
single mode while worsening the complete training target from being selected.

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

## 7. Frozen post-selection contract

The fresh-network aggregate chooses at most one `best_usable_anchor` before
natural KMeans, figures or held-out events are opened. Seeds 2351--2353 are
selection networks only. After one field is frozen, run only that field and
paired `exact_off` on unseen seeds 2361--2363, six 20-s Node-only simulations
in total. The field must repeat lower complete `J14_v1`, lower A loss, B loss
within 110% of paired exact, and effective support of at least six for both
modes on all three confirmation networks. Failure stops this field and does
not promote or rerank another candidate.

Natural KMeans and every later analysis reuse the exact seed-2361--2363
confirmation arrays. They never reopen the selection-network arrays. This
separates field choice from final same-network repertoire, held-out and
intervention evidence without adding another field search.

For each confirmation network, formal events are complete, non-overlapping
returned causal families that are readable on at least three contacts, recruit
both shafts and lie inside the frozen patient-training classifier support.
Masked normalized event ranks are clustered with K=2 without patient labels.
Cluster identities are mapped to frozen A/B labels only after fitting.

The selected Node progresses only if all three networks have at least three
events in each supervised mode and each natural cluster, all three have
natural-versus-supervised AMI at least 0.8, and the pooled patient-training
prototype matrix has positive diagonal and negative crossed cells. KMeans may
reject the selected field but cannot select another one.

The two canonical Fig.4 outputs use exactly this event set: direct continuous
electrode readout and masked-rank KMeans consistency. Diagnostic figures from
a rejected field must be explicitly marked diagnostic and cannot advance the
pipeline.

## 8. One-time held-out and source-topology audit

Only a confirmation- and post-selection-accepted field opens the frozen
patient held-out event set once. It is compared with paired same-seed
`exact_off`, with no field reranking. The eventwise prototype R2 is retained
but named explicitly: it asks
whether the two model mean prototypes explain held-out patient event variance
and does not measure model event-cloud dispersion. Progress therefore requires
positive and improved eventwise prototype R2; improved mode-conditioned,
shaft-balanced sliced-Wasserstein cloud loss for A, B and their weakest-mode
LSE; and improved total loss for A, B and the weakest-mode objective. The cloud
calculation uses every held-out recruitment/rank event vector, so a repeated
mean prototype cannot pass by itself.

Mode-specific early-source topology is evaluated only on complete returned,
source-evaluable causal families that also pass the frozen Fig.4 patient-mode
support contract: both shafts participate and the event is not classifier OOD.
An event that merely receives a forced A/B label is not eligible to define a
source template or an intervention representative. The continuous endpoint is
weakest-mode
cross-network cosine similarity multiplied by between-mode topology distance;
the null permutes mode labels within each network while preserving occupancy
for 4096 draws. Passing this zero-simulation audit permits the intervention but
does not yet freeze Node.

## 9. Same-checkpoint mode-selective intervention

For each of the three confirmation networks, choose one complete event per
native mode algorithmically. Branch the identical checkpoint and random stream
into sham, the two mode hotspots and their matched off-template controls. A
strong local E-threshold pulse begins 5 ms after the checkpoint; intervention
interpretation is limited to model-internal regional necessity.

The native representative event and all leave-one-network-out hotspot
templates use the same returned, source-evaluable, dual-shaft and in-support
event set as the source-topology audit. OOD or single-shaft events cannot enter
the hotspot definition even when the classifier assigns them an A/B label.

The target for one network is computed from the other two networks only. For
each mode, the hotspot maximizes that mode's early-source probability relative
to the competing mode; the two hotspots must be separated by at least 3 mm.
Off-template controls lie in the lower quartile of the union of both mode
templates and are matched to the hotspot over the actual 1.2-mm pulse disk,
not a mismatched 1-mm bin. Matching uses mean Node `h`, targeted E-neuron count
and baseline E rate, with an IQR-standardized L1 caliper of 2.0 and maximum
single-covariate difference of 1.0. An unmatched control or overlapping mode
hotspots can be reported descriptively but cannot support Node freezing.

The primary ordered endpoints are event survival and onset latency. A mode
hotspot is selective only when its effect on the predicted native mode exceeds
both its effect on the opposite mode and its adequately matched control. At
least two of
three networks must show the crossed selective pattern. The reconstructed
`h`, `Vtheta` and `delta-Vtheta` arrays must be exactly equal to the frozen
worker before any branch runs. Only this result may write the Node freeze
manifest; EE, E-to-I and Z/M remain off throughout.
