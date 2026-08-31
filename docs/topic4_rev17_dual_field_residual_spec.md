# Topic 4 rev17: exact dual-field residual Node atlas

## 1. Motivation

The accepted Node-only substrate does not use one field through
`Delta Vtheta_i = -h_i d_i`. It uses two independent continuous fields:

`Delta Vtheta_i = -h_mean,i mu_mean - h_disp,i (d_i - mu_disp)`.

The first term carries smooth mean excitability. The second gates the frozen
signed neuron-to-neuron dispersion while conserving its summed contribution.
The accepted anchor combines `stage_ag_g10_p` as the mean field and
`stage_ag_g08_m` as the dispersion field.

Rev16 did not preserve this mapping. Its selectable candidates replaced the
accepted field by an absolute Fourier surface and applied the historical
single-field `-h d` rule. Exact-off bypassed that rule. Candidate-versus-anchor
comparisons therefore changed both field geometry and Node mapping, so the
negative fresh-network result cannot establish that a continuous Node field
lacks two-mode capacity.

Rev17 repairs the experiment before any further optimizer, EE, E-to-I or Z/M
work. It asks whether small, observation-invariant continuous changes around
the exact accepted dual-field substrate reveal directions that improve the
weak patient mode while preserving the strong mode and event support.

## 2. Frozen substrate and zero point

The zero coordinate is the exact dual-field candidate recorded in the rev16
manifest. Both 18-by-18 cubic-spline coefficient tensors, the frozen signed
depth draw, network topology, noise law, detector, event boundary, contact
readout and Node budget are inherited.

Before any SNN run, seeds 2351--2353 must reproduce the archived exact-off
arrays after float32 serialization:

- `h_mean` exactly;
- `Delta Vtheta` exactly;
- all learned EE and E-to-I coefficients exactly zero;
- mapping type exactly `dual_continuous_mean_dispersion`.

Failure is a representation error and blocks the atlas. It is not a scientific
negative result.

## 3. Continuous residual coordinates

The residual basis is defined on the uniform 20-by-20 mm sheet and never
receives contact, shaft, patient-mode, manual-core or ictal coordinates. It is
the ordered low-frequency cosine span with maximum frequency three, projected
onto the stored 18-by-18 cubic-spline basis. The constant mode is excluded.

For each of the 15 spatial modes, rev17 applies an antithetic residual of
sheet-wide latent RMS `+/-0.15` to one channel at a time:

1. mean field perturbed, dispersion field exact;
2. dispersion field perturbed, mean field exact.

Together with exact zero, this gives 61 candidates. This stage measures local
finite-difference responses; it is not yet an optimizer and individual
directions are not required to pass a hard patient-fit gate.

## 4. Simulation and endpoints

All candidates run Node-only for 20 s on fit networks 2361--2363 with common
network/noise seeds. EE, E-to-I and Z/M remain off. Late runaway is invalid.

For each candidate and network report:

- complete training-target `J14`;
- mode A and mode B recruitment, precedence, profile and event-cloud losses;
- effective in-support events for both modes;
- returned-event yield, compound fraction and safety;
- mode-specific source topology and contrast alignment;
- finite-difference symmetry and cross-network sign consistency.

The response analysis may construct regularized joint directions across both
channels. It must protect mode B and minimum support explicitly; improving an
average score by collapsing one mode is not a useful direction.

Natural KMeans, patient held-out blocks and figures remain closed during atlas
construction and response-direction fitting.

## 5. Advancement and final Node freeze

The atlas advances if it is complete and yields at least one bounded joint
direction whose predicted response improves weak mode A and `J14` without
worsening mode B beyond 10% or pushing either mode below effective support six.
This is an exploratory nomination rule, not Node acceptance.

Nominated joint residuals are first run on fresh selection networks 2371--2373.
Each network must individually improve `J14` and weak mode A relative to the
paired exact dual anchor, retain mode B within 10% and retain effective support
six for both modes. Pooled compensation is not allowed. This stage freezes one
candidate for confirmation; it does not freeze Node and does not open natural
KMeans, patient held-out blocks or figures.

The selected candidate and exact dual anchor are then run once on unseen
confirmation networks 2381--2383. The training-target, safety and support rules
are re-established without reranking. Only after this confirmation passes may
the already frozen candidate undergo the following read-only acceptance layers:

- both modes occur in the same network;
- natural masked-rank KMeans has at least three events in each cluster and
  aligns with the frozen patient-mode classifier at AMI at least 0.8 in all
  three networks;
- the pooled model--patient matrix has positive diagonal and negative crossed
  cells;
- complete patient-training distribution and weakest mode remain improved;
- mode B is preserved;
- each mode has adequate support and distinct source topology.

Natural KMeans may accept or reject the frozen candidate but may not rerank the
field. If it passes, the same confirmation worker arrays are opened once for
patient held-out and source-topology evaluation. Relative to the paired exact
dual anchor, the candidate must improve positive held-out eventwise prototype
`R2`, both mode losses and their weakest-mode aggregate, and both mode-specific
event-cloud losses and their weakest-mode aggregate. Its mode-specific source
topology must exceed a within-network occupancy-preserving label-permutation
q95 and must improve over the exact dual anchor. Failure rejects the candidate;
there is no return to atlas ranking.

Passing this read-only audit permits, but is not itself, final Node freeze. The
last requirement is a same-checkpoint crossed intervention on confirmation
networks 2381--2383:

- choose one source-evaluable native event per mode and network by the
  within-network joint source-topology/contact-rank medoid rule;
- define each mode hotspot leave-one-network-out from the other two networks'
  earliest 10% onset support;
- compare sham, the predicted hotspot and a spatially separated off-template
  control matched over the actual pulse footprint for mean Node field,
  `Delta Vtheta`, E-neuron count and baseline E rate;
- raise local threshold by 20 mV for 70 ms from the identical checkpoint and
  random stream, then compare event survival first and nonnegative onset delay
  second;
- require the predicted-mode effect to exceed both the opposite-mode effect
  and its matched control in at least two of three networks for at least one
  mode.

Only that final intervention result can set `REV17_NODE_FIELD_FROZEN`. EE,
E-to-I and Z/M remain closed throughout the complete chain and may be opened
only after this status is established.

## 6. Figure acceptance

Formal figures cannot select or rerank the field. They are rendered only after
`REV17_NODE_FIELD_FROZEN` and must consume the exact confirmation/intervention
artifacts used by the audits.

1. The direct-readout view shows the continuous dual Node substrate, the two
   mode-specific onset-density fields and two temporally non-overlapping native
   events from one confirmation network.
2. The companion KMeans view reuses the accepted Figure 1E masked-rank painter
   and reports per-network support/AMI together with pooled model--patient
   profiles.
3. Source-topology and crossed hotspot effects are shown as causal-validation
   panels or a supplement from the same frozen artifacts; they cannot replace
   either of the two Fig.4 acceptance views.

## 7. Claim boundary

Rev17 is development-only. The atlas uses the frozen patient-training target
to estimate Node response and therefore is not patient-blind. A successful
atlas establishes only a locally useful response direction. Full success
requires fresh-network selection, unseen-network confirmation, natural KMeans,
held-out distribution improvement, distinct source topology and selective
same-checkpoint intervention. Even then the result is a viable continuous Node
substrate with model-internal regional necessity; it does not identify a
biological core, a causal patient mechanism or an interictal-to-ictal bridge.
