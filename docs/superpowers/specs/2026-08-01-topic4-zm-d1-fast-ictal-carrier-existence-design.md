# Topic 4 Z/M fast-dynamics discovery and lifecycle vertical-slice design

**Date:** 2026-08-01
**Revision:** 2 — post-development-strategy review
**Status:** DEVELOPMENT SCREEN + EARLY LIFECYCLE VERTICAL SLICE
**Independent line:** E→E graph, weights, kernel, anisotropy, STD and plasticity remain immutable

## 0. Decision

The scientific objective remains one fixed-parameter SNN that can eventually
support

\[
\mathcal I\xrightarrow{Z}\mathcal S
\xrightarrow{M}\mathcal P\xrightarrow{\mathrm{recovery}}\mathcal I,
\]

with a finite control input able to shorten an existing pathological episode
without resetting the network.

Revision 1 incorrectly treated model development as a confirmation pipeline.  It
required a deeply perturbation-returning frozen fast attractor, completely
serialised carrier, entry, offset, recovery and control, and placed extensive
observation/provenance infrastructure before the first informative SNN run.

Revision 2 keeps the E-only dynamic-threshold hypothesis, but changes the order:

1. **Stage A — fast-dynamics discovery:** immediately expose what \(\phi\) does
   to the old tonic branch;
2. **Stage B — reachable lifecycle vertical slice:** put the best phenotypes
   back into their own fixed-parameter trajectory from interictal baseline and
   examine entry, native-exit tendency, controlled exit and recovery together;
3. **Stage C — locked confirmation:** only a lifecycle-compatible candidate
   earns multi-seed/noise, \(dt/2\), real-data sidecar, ablation and publication
   infrastructure.

This is not an autonomous-lifecycle confirmation spec.  It is a bounded
development experiment designed to decide quickly whether the \(\phi\) mechanism
is worth confirming or whether the fast inhibitory/membrane class must change.

## 1. Accepted upstream evidence

### 1.1 Phase C

Accepted as `ACCEPT_PHASEC_POST_RESULT_FUTILITY_STOP`:

- C0 completed 153/153 tasks across seeds 1/3/4;
- C1 completed 59/60 seed-1 primary continuations;
- corrected-v2 classified 59/59 as `tonic_non_AI`;
- modulation depth was 0.025–0.045 versus the registered 0.20 marker;
- the missing run cannot rescue the registered C1 GO.

This stops further expansion of the same frozen \(Z/M/S_G\) morphology grid.  It
is not a three-seed negative and does not prove carrier non-existence.

### 1.2 Phase D

Accepted as `NO_GO_baseline_calibration_failed_zero_spike_dominance`:

- conductance, sAHP and dynamic-threshold hooks were implemented;
- current-based Arm-A state migration retained byte identity;
- the registered all-time conductance replacement eliminated E spikes during
  the dynamic interictal baseline, so carrier arms never ran.

This rejects that conductance replacement.  It does not test the E-only
dynamic-threshold mechanism on the original current-based Z/M substrate.

### 1.3 Scale diagnostics

The small \(m\) scale, latch-like \(z\) direction, short post-drive \(S_G\)
memory and high Phase-C refractory occupancy remain design risks, not
mathematical non-existence conditions.

## 2. Locked substrate and independence boundary

Use the exact current-based per-neuron Z/M family:

- E1146 `twoend_equal` 2-D anisotropic E/I SNN;
- \(N_E=32000\), \(N_I=8000\), \(L=20\) mm;
- existing heterogeneous two-core threshold substrate;
- per-neuron \(z_i,m_i\), `use_qI=False`, `use_gK=False`;
- \(\tau_z=5000\) ms, \(\tau_m=500\) ms, \(\eta_m=0.001\);
- recurrent-only \(S_G\), \(\alpha_G=16\), \(\tau_S=80\) ms;
- original virtual-electrode montage, pathology axis, delays and noise law;
- current-based membrane and synaptic-current semantics.

Locked throughout Stages A and B:

- all E→E edges, weights, anisotropy, orientation, STD and plasticity;
- external E/I drives and noise statistics;
- \(z/m\) equations and constants;
- base thresholds, reset and refractory periods;
- virtual-electrode geometry.

The Phase-D conductance path stays disabled.  No conductance homotopy or E→E
rescue is allowed on this line.

## 3. Mechanism under development

The primary mechanism is the already implemented E-only per-neuron threshold
increment

\[
\dot\phi_i=-\frac{\phi_i}{\tau_\phi}
+\Delta_\phi\sum_k\delta(t-t_i^k),
\qquad V_{\theta,i}=V^0_{\theta,i}+\phi_i.
\]

- \(\phi_i\) is local to one E neuron;
- I-cell thresholds remain unchanged;
- no global activity sensor, spatial recruitment mask or ictal-only equation is
  introduced;
- `use_phi=False` must preserve the existing path;
- in a genuine reachable trajectory, the same equation is active from \(t=0\).

The initial panel is

\[
\tau_\phi\in\{60,100,160\}\ \mathrm{ms},
\qquad f_\phi\in\{0.15,0.30\},
\]

with

\[
\Delta_\phi
\left(\frac{\tau_\phi}{1000}\right)
r_{\mathrm{core,ref}}
=f_\phi(V_\theta-V_{reset}).
\]

The `/1000` conversion is mandatory because \(\tau_\phi\) is stored in ms and
rate in Hz.  The six points represent two steady feedback strengths across three
recovery times; they are an initial discovery panel, not an exhaustive test of
dynamic threshold.

## 4. Two distinct state semantics

### 4.1 Arm A — branch-intervention fork

Load the four old Phase-C checkpoints:

- `bounded_mid__rising`;
- `bounded_mid__peak`;
- `bounded_late__rising`;
- `bounded_late__peak`.

Freeze their \(z_i,m_i\) fields, initialise \(\phi_i=0\), and allow fast E/I,
dynamic \(S_G\), delays and \(\phi\) to evolve.

This arm asks only:

> When \(\phi\) is introduced at an old tonic state, does it leave tonic,
> fragment, relay, oscillate, silence or run away?

It is an intervention diagnostic.  Because the old checkpoint was generated by
a no-\(\phi\) history, this arm cannot establish fixed-parameter reachability.

### 4.2 Arm B — reachable dynamic trajectory

Start from the original interictal initial condition with \(\phi\), \(z\), \(m\)
and \(S_G\) all active from \(t=0\).  Save this model's own state history and
checkpoints.  No old fast state is spliced onto it.

Only Arm B may support:

- `reachable_carrier_candidate`;
- spontaneous-onset claims;
- native-offset tendency;
- controlled-exit pilot;
- recovery to returning events.

## 5. Stage A — fast-dynamics discovery

### 5.1 Minimal sanity before compute

Before the 24-cell matrix, verify only:

1. correct Hz/ms \(\Delta_\phi\) calculation;
2. `use_phi=False` parity;
3. E-only action and exact I-cell zeros;
4. exponential decay plus one jump per E spike;
5. frozen \(z,m\) do not drift in Arm A.

Do not build a new real-data loader, generic parallel coordinator, full verdict
framework or formal figure package before this matrix.

### 5.2 Discovery matrix

Run seed 1, the six initial phi points, the four old checkpoints and one replay
future-noise bank: 24 production-scale continuations.  Each run is 6 s with the
first 1 s treated as switch-on transient and the following 5 s described.

Save only the traces needed to identify dynamics:

- core, surround, E and I population rates;
- modulation depth and event/gap structure;
- refractory occupancy;
- \(\phi\), \(S_G\) and frozen \(z/m\) checks;
- spatial active fraction, axial recruitment and kymograph;
- existing virtual-SEEG proxy and band-energy readout.

### 5.3 Phenotype taxonomy

Every run receives one descriptive phenotype:

| Phenotype | Meaning | Development response |
|---|---|---|
| `tonic` | adaptation did not leave the old branch | stronger adjacent gain only if a boundary is visible |
| `burst_train` | tonic broken but full-return gaps remain | adjacent time-scale refinement |
| `spatially_relayed_carrier` | bounded, sustained and spatially staggered | promote immediately to Stage B |
| `metastable_carrier_like` | finite carrier-like episode or perturbable state | promote to Stage B; do not reject for lack of return |
| `silence` | adaptation too strong | weaker adjacent gain only if paired with an active neighbor |
| `whole_sheet_oscillation` | common-mode synchrony dominates | diagnose fast inhibition; do not retune E→E |
| `runaway` | containment failed | phi alone is insufficient at this state |
| `technical_invalid` | provenance/numerical failure | rerun identically; never classify scientifically |

The existing v2.1 carrier gate is reported as an operational descriptor and
ranking aid.  It is not the sole Stage-A gate.  AI is secondary.

### 5.4 Limited local refinement

One bounded refinement round is allowed only when the initial matrix reveals a
coherent boundary or carrier-like near miss.

- stronger-gain neighbor: \(f_\phi=0.45\);
- weaker-gain neighbor: \(f_\phi=0.075\);
- faster-time neighbor: \(\tau_\phi=40\) ms;
- slower-time neighbor: \(\tau_\phi=240\) ms.

At most one point is opened per justified boundary direction and no more than
four new parameter combinations total.  The partner \(f_\phi\) or \(\tau_\phi\)
is chosen from the closest initial-panel cell before the new run is launched and
recorded in a discovery amendment.  No second refinement round is allowed.

If whole-sheet synchrony dominates every active phi point, one predeclared
fast-inhibition diagnostic may be used at the best phi point:

\[
\tau_{d,GABA}\in\{12,24\}\ \mathrm{ms}
\quad\text{versus canonical }18\ \mathrm{ms}.
\]

This is a two-direction mechanism diagnostic, not a new grid and not a positive
claim.  E→E remains untouched.  Any surviving combined candidate must later pass
its own baseline and ablation tests.

### 5.5 Stage-A selection

Promote at most two candidates.  Ranking prioritises:

1. bounded non-tonic activity without hard refractory saturation;
2. spatial relay/local-to-extended recruitment over common mode;
3. sustained virtual-SEEG occupancy/energy rather than isolated HFO bursts;
4. distance from both silence and runaway boundaries;
5. distinct candidates if two different phenotypes survive.

If the registered panel and allowed refinement contain no carrier-like state,
emit only

`NO_CARRIER_IN_REGISTERED_PHI_PANEL_AND_TESTED_STATES`.

This is not a mechanism-class no-go.

## 6. Stage B — reachable lifecycle vertical slice

Stage B runs before multi-seed confirmation.  For each of the top one or two
Stage-A candidates, start Arm B from interictal baseline with all equations
active from \(t=0\).

### 6.1 Baseline evaluation

Hard failures are limited to:

- persistent silence;
- runaway or whole-sheet plateau before a carrier can form;
- complete loss of returning events from both cores;
- loss of the pathology-axis recruitment geometry or conversion to simultaneous
  whole-sheet flash.

The following are continuous deviation scores, not discovery vetoes:

- event count and inter-event interval;
- event duration and amplitude;
- all-sheet rate and active fraction;
- core balance;
- \(\phi\) carryover between events.

The former ±20% rules and “\(\phi\) below 10% in 80% of intervals” are deferred
to Stage-C distributional validation.

### 6.2 Reachability and carrier identity

Run long enough to include the original escalation window and at least 10 s
after the first sustained high-activity episode, with a maximum developmental
horizon of 30 s.  Continuously checkpoint or retain a rolling state buffer; do
not turn \(\phi\) on at onset.

Ask:

1. do returning interictal events remain before escalation?
2. does the system enter the same phenotype family found in Arm A?
3. is the carrier bounded and spatially organised rather than tonic/common-mode?
4. what are the directions of \(\dot z\), \(\dot m\), \(\dot\phi\) and \(\dot S_G\)
   during onset, maintenance and decline?

A failed identity match means the frozen branch phenotype was unreachable; it
cannot be rescued by Arm-A replication.

If no matching high-activity episode occurs within the 30 s development horizon,
record `no_reachable_entry_within_development_horizon`.  This is a finite-horizon
development result, not proof that the carrier basin is globally unreachable.

### 6.3 Native and controlled exit in parallel

From matched reachable carrier states, fork two developmental branches:

**Native branch:** no intervention; measure whether activity terminates, whether
\(m\) moves toward an exit boundary, and whether returning events reappear.

**Controlled branch:** apply a 50 ms E-threshold uplift without resetting any
state.  Use doses

\[
u/(V_\theta-V_{reset})\in\{0,0.05,0.10,0.20\}
\]

at two fixed spatial scopes: the active pathological core and all E cells.  Use
the same saved carrier state and the three already available future-noise
continuations (replay plus two resamples) for paired comparisons.  Record,
rather than gate on, the descriptive pilot fractions

\[
P(\mathrm{return}\mid u),\qquad P(\mathrm{exit}\mid u),
\]

plus silence, rebound, recovery time and subsequent returning events.
With only one seed and three future-noise continuations these are susceptibility
descriptors, not control-efficacy estimates.

The control fork may be state-triggered because it is explicitly an
intervention experiment; the uncontrolled trajectory may not use a detector to
change its equations.

### 6.4 Stage-B development outcomes

Allowed outcomes are:

- `unreachable_frozen_phenotype`;
- `no_reachable_entry_within_development_horizon`;
- `reachable_carrier_no_exit_route`;
- `reachable_native_offset_no_recovery`;
- `spontaneous_onset_with_controllable_termination_candidate`;
- `autonomous_lifecycle_candidate`;
- `suppression_without_recovery`;
- `lifecycle_compatible_candidate`.

A metastable episode may be a positive lifecycle candidate.  Failure to return
after a strong perturbation is not a carrier-existence failure.  Conversely,
permanent silence after control is suppression, not recovery.

## 7. Stage C — locked confirmation

Only `lifecycle_compatible_candidate`,
`spontaneous_onset_with_controllable_termination_candidate`, or
`autonomous_lifecycle_candidate` may open Stage C.

Before confirmation:

- freeze the candidate equations and parameters;
- freeze validation seeds/noise/windows/phenotype definitions;
- rebuild and hash the representative E1146 seizure-7 Fig3-A observation
  sidecar;
- freeze sham and matched-energy control conditions;
- freeze ablations and resource policy.

Then perform:

- seeds 1/3/4 and multiple future-noise continuations;
- independent \(dt/2\) confirmation;
- longer pre/post interictal distribution comparison;
- real-data virtual-SEEG comparison;
- \(\phi\), \(S_G\) and any fast-I backup ablations;
- native-versus-controlled exit dose response;
- formal resource receipts, archive and paper-facing diagnostic figures.

The real-data sidecar is required for Stage-C claims but must not block Stage A
or Stage B mechanism development.

## 8. Claim boundaries

Stage A can claim only:

> Introducing E-only dynamic threshold at old Phase-C tonic checkpoints produced
> the observed phenotype map on the registered seed-1 panel.

Stage B can claim only the explicit development outcome.  In particular:

- controlled exit without native offset is not an autonomous lifecycle;
- offset without returning events is not recovery;
- an Arm-A phenotype is not reachable until reproduced on Arm B;
- one-seed vertical slices are candidates, not robust mechanisms.

Only Stage C may support a replicated carrier/lifecycle/control statement.

## 9. Resource and output contract

Stage A reuses existing state loading, carrier analysis and Phase-C resource
helpers.  Do not build a new generic scheduler or verdict package.

- set OMP/MKL/OpenBLAS/NumExpr threads to one;
- measure one worker before bounded parallel launch;
- retain at least 96 GB `MemAvailable` and eight logical CPUs;
- never kill or modify peer-worktree processes;
- publish each run atomically with a minimal JSON provenance record;
- resume missing or technical-invalid cells only.

Development root:

`results/topic4_sef_hfo/zm_fast_lifecycle_development/`.

Required before Stage B:

1. one phenotype matrix JSON/CSV;
2. one compact diagnostic figure with representative traces and spatial maps;
3. a short Chinese `figures/README.md`;
4. a candidate/refinement decision record.

Formal multi-panel figures and archive-grade evidence packaging are deferred to
Stage C.
