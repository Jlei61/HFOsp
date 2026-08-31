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

## 5. Advancement

The atlas advances if it is complete and yields at least one bounded joint
direction whose predicted response improves weak mode A and `J14` without
worsening mode B beyond 10% or pushing either mode below effective support six.
This is an exploratory nomination rule, not Node acceptance.

Nominated joint residuals are then run on fresh selection networks. Node can be
frozen only after all of the following are established on unseen networks:

- both modes occur in the same network;
- natural KMeans aligns with both patient modes;
- complete patient-training distribution and weakest mode improve;
- mode B is preserved;
- each mode has adequate support and distinct source topology.

Only after that freeze may patient held-out evaluation and same-checkpoint
hotspot intervention run. EE, E-to-I and Z/M remain closed until Node is frozen.

## 6. Claim boundary

Rev17 is development-only. The atlas uses the frozen patient-training target
to estimate Node response and therefore is not patient-blind. A successful
atlas establishes a viable continuous Node parameterization and a candidate
dual-mode substrate; it does not identify a biological core, a causal patient
mechanism or an interictal-to-ictal bridge.
