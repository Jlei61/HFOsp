# Topic 4 rev13: field-gated zero-sum Node recovery

## 1. Scientific motivation

Rev12 established three facts that must be kept separate.

First, the continuous data-driven Node field is active: it changes event yield,
patient-fit endpoints and causal propagation direction. Second, broad field,
signed-depth, scalar-gain and dual-channel scans all retain the same trade-off:
opening the second causal direction worsens mode-0 recruitment, profile and
event-cloud fit. Third, the same frozen SNN can express two opposite causal
waves under the Stage-AK and manual capacity controls, so the failure is not a
simple absence of network capacity.

The remaining minimal hypothesis is therefore dynamic accessibility within the
frozen Node substrate. A recently active part of the learned field may become
temporarily harder to reactivate while another part becomes relatively more
accessible. This must be tested before EE, E-to-I or Z/M are opened.

This is not licensed as a patient mechanism. The completed patient sequence
audit found unconditioned short-lag clustering, but that result was not robust
to a 10-s local-occupancy null and did not replicate after a true post-event
gap of at least 0.5 s in held-out blocks. Its frozen status is
`PATIENT_MODE_MEMORY_NOT_ROBUST_TO_GAP_CONTROL`; neither short-lived recovery
nor a persistent state is authorized. The canary below is therefore a bounded
model-capacity test only. Patient event history cannot define the runtime
state, select a time constant or amplitude, or switch the model between A and
B.

## 2. Frozen scientific question

Can a bounded, model-internal and activity-dependent redistribution of Node
thresholds allow one frozen continuous field and one frozen network to express
two independent causal propagation modes, without changing total signed Node
budget, recurrent connectivity, external-drive law or Z/M?

Only after model-internal capacity is established may the frozen patient
training target evaluate whether both modes match the complete Fig.4
recruitment, precedence, profile and event-cloud distributions.

## 3. Historical non-duplication

Rev10-D1 tested a different mechanism:

`a_i(t+dt) = exp(-dt/tau_a) a_i(t) + q_a S_i(t)`

with `a_i` subtracted from excitatory current. It was a raise-only local brake:
it could suppress a recently active region but could not lower the threshold of
an unused field region. Nineteen off/local/global candidates across three
networks completed without runaway; every candidate produced formal mode A in
0/3 networks. That family is closed and must not be rescanned.

The frozen spatial OU is also not the proposed mechanism. It redistributes
external E drive with a translation-invariant stochastic field. It opened both
supervised directions but failed natural KMeans/patient replication and greatly
increased activity occupancy. Rev13 adds no stochastic drive and draws no new
random numbers at runtime.

## 4. Controller equation

For E neuron `i`, define the frozen continuous support

`g_i = (h_mean_i + h_dispersion_i) / max_j(h_mean_j + h_dispersion_j)`.

This is the normalized union of the two already frozen Stage-AK Node channels;
it introduces no contact, shaft, patient-mode or new spatial basis. Let `a_i`
be a bounded recent-spike trace in mV:

`a_i(t+dt) = min(a_max, gamma a_i(t) + q S_i(t))`,

where `gamma=exp(-dt/tau_a)`, `S_i` is the current-step spike indicator and
`a_max=2 A_ref`. The state is updated after spike/reset and affects the next
step only.

Define

`a_bar_g = sum_i g_i a_i / sum_i g_i`,

`delta_theta_i = g_i (a_i - a_bar_g)`,

`Vtheta_eff_i = Vtheta_static_i + delta_theta_i`.

Then

`sum_i delta_theta_i = 0`.

This conserves only the unweighted sum of the added E-neuron threshold
offsets. It does not conserve firing rate, support-weighted threshold, or a
general physiological excitability budget. This is a
field-gated, field-wide redistribution, not purely local adaptation. It raises
thresholds in recently active field regions and lowers them elsewhere in the
same learned field.

`q` is not scanned independently. For a model-internal reference rate
`r_ref=50 Hz`,

`q = A_ref (1-gamma) / (r_ref dt / 1000)`.

This keeps the reference steady-state trace fixed when `tau_a` changes.
`A_ref=c sigma_node`, where `sigma_node` is the frozen support-weighted,
mean-centered standard deviation of the static Node threshold modulation. It
is not an uncentered RMS. The first canary uses `tau_a=250 ms` and
`c={0.1,0.2,0.4}`. The expected reference amplitudes are approximately
0.06--0.25 mV and the maximum dynamic threshold displacement is bounded near
0.5 mV. `tau_a={100,500 ms}` is a later sensitivity, not part of the first
screen.

## 5. Frozen controls

The first canary contains six arms per substrate:

1. `exact_off`: controller absent, not zero multiplied.
2. `zero_sum_c010`: zero-sum controller, `tau=250 ms`, `c=0.1`.
3. `zero_sum_c020`: same, `c=0.2`.
4. `zero_sum_c040`: same, `c=0.4`.
5. `raise_only_c020`: identical bounded trace but no compensating lowering of
   unused field regions.
6. `spatial_shift_c020`: deterministic half-sheet toroidal shifting of the
   trace through eight equal-count x bins, followed by the same zero-sum
   centering. At each step its dynamic field is rescaled to the
   support-weighted centered SD of unshifted `zero_sum_c020`. It preserves
   coarse spatial organization and instantaneous amplitude while breaking the
   learned activity-location alignment. Salt-and-pepper neuron shuffling is
   not an admissible scientific control.

The old D1 weak arm is a historical negative control and is not rerun.

One pre-existing data-driven substrate is the primary input, not a newly
optimized field:

- `stage_ak_mean_g10_p_disp_g08_m`, which is the same mapping as the Stage-AL
  endpoint `stage_al_m100_d100` and must not be counted twice.

This substrate was selected by a no-label full-sheet audit: all 6/6 historical
networks contain two causal clusters with opposite median displacement along
the substrate axis. The channel-swapped
`stage_ak_mean_g08_m_disp_g10_p` mapping is a later sensitivity and is run only
after a positive primary result.

The first engineering/model-internal pass uses one unused network for six
runs. Only if the controller is active without numerical or event-unit failure
are two additional unused networks run, for 12 more primary trajectories.

## 6. Model-internal canary endpoints

No patient label, contact identity or patient prototype may select an arm.
For each network and arm report:

- number of causal-root families and detector fragments per family;
- compound and multi-root fractions;
- contiguous-time held-out K=1 versus K=2 density evidence on the
  one-dimensional signed causal-family displacement along the frozen
  substrate axis;
- opposite median displacement signs, at least 0.70 directional consistency
  within each cluster, and recurrence of both signs in at least two of three
  contiguous time blocks;
- minority-cluster fraction;
- within-network transition matrix and run-length distribution;
- whether same-mode probability is below its occupancy-matched q05, which
  would indicate forced alternation;
- event rate, return, OOD-independent event support and runaway;
- topology separation and cross-network reproducibility;
- `sum(delta_theta)` error, threshold range and support-weighted trace
  saturation;
- support-weighted static-Node sign-flip fraction restricted to `g>=0.05` and
  static modulation magnitude at least `0.1 sigma_node`, with the unrestricted
  fraction retained only as a diagnostic;
- lowered-field mass.

Whole-sheet onset-map KMeans remains a label-invariant topology visualization
and later Fig.4 readout. Raw high-dimensional onset-map GMM density is not a
formal endpoint: replay on the six historical Stage-AK networks produced
pathologically large negative held-out K2-K1 values despite visibly separated
directional families.

An arm is not a capacity candidate if KMeans improvement is explained by
fragmenting one causal episode, if it induces anti-persistent alternation, if
the spatial-shift control performs similarly, or if it requires widespread
signed-depth reversal. Such a result is
`DUALMODE_BY_GENERIC_ALTERNATION`, not recovery.

The formal comparison event-count matches all paired arms within a network.
Only `zero_sum_c020` has coefficient-matched `raise_only_c020` and
`spatial_shift_c020` controls in this canary. `c=0.1` and `c=0.4` are
signal/dose arms; a positive result there is
`MATCHED_CONTROL_EXTENSION_REQUIRED`, not formal acceptance.

For a primary capacity candidate, at least 2/3 networks must show positive
held-out K2-K1 evidence above paired exact-off and both coefficient-matched
controls, opposite cluster directions, at least 20% occupancy in each cluster,
both directions in at least two of three time blocks, and returned-event yield
at least 50% of paired off. An arm is invalid if runaway occurs or its
compound-family fraction exceeds paired off by more than 0.15. These are
model-internal capacity rules, not patient acceptance.

## 7. Patient-scored acceptance

Only model-internally admissible arms are evaluated with the frozen patient
training target. Patient held-out remains closed. The formal endpoint remains
the same-network Fig.4 contract:

- natural K=2 support and label-invariant alignment to both patient modes;
- both causal directions present in the same network;
- recruitment, precedence, profile and event-cloud loss for both modes;
- weakest complete-patient endpoint improved over the paired static substrate;
- majority-network improvement, not pooled-event significance.

No KMeans-only result can rescue a worse complete patient distribution.

## 8. Engineering contract

- Add an independent `node_accessibility` engine argument; never reuse `slow`.
- `None` and exact-off paths are bitwise identical to rev12 Node-only and draw
  no random numbers.
- The controller never mutates `substrate.vtheta` in place.
- State update occurs after current-step spike/reset and affects the next step.
- Threshold arrays must be finite, shape-aligned and remain above
  `V_reset+1 mV`; bound failure invalidates the run rather than clipping it.
- Checkpoint schema stores controller kind, state, step and config hash;
  checkpoint/resume is bitwise exact.
- A checkpoint without controller state cannot resume a controller-enabled run.
- EE, E-to-I and Z/M are exactly off in rev13 canary and replication.

## 9. Claim boundary

A positive rev13 result would first show model capacity: a learned continuous
Node substrate plus bounded, zero-sum activity-dependent redistribution can
support two directional event families without generic alternation or a
spatially unmatched control. Because the patient gap-controlled audit did not
support robust mode memory, this controller is not claimed as a patient-derived
recovery mechanism. Patient alignment is evaluated only after the
model-internal arm is frozen. The result would not identify a patient cellular
mechanism, establish blind generalization, or authorize an ictal claim.
EE/E-to-I/Z/M remain a later cross-state experiment after Node is frozen.
