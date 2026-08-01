# Topic 4 Z/M lifecycle roadmap — D1 fast ictal-carrier existence design

**Date:** 2026-08-01
**Status:** LOCKED FOR D1 ONLY
**Parent evidence:** Phase C post-result futility stop; Phase D conductance baseline-gate NO-GO
**Independent line:** E→E graph, weights, kernel, anisotropy, STD and plasticity remain immutable

## 0. Decision and scope

The scientific objective is one fixed-parameter SNN that can eventually support

\[
\mathcal I\xrightarrow{Z}\mathcal S
\xrightarrow{M}\mathcal P\xrightarrow{\mathrm{recovery}}\mathcal I,
\]

where \(\mathcal I\) is the returning, irregular interictal regime, \(\mathcal S\)
is a bounded and spatially structured ictal carrier, and \(\mathcal P\) is a
postictal recovery corridor.  A finite control input may later shorten the
\(\mathcal S\to\mathcal P\) transition, but may not create recovery by reset.

This spec authorizes only **D1: fast-carrier existence on frozen slow states**.
It does not authorize endogenous entry, native offset, recovery, stimulation or
shared-scaffold causality.  D2–D6 are roadmap nodes and require separate specs.

The immediate question is:

> On the unchanged anisotropic Z/M SNN scaffold, can one baseline-compatible,
> neuron-local fast membrane feedback convert the Phase-C high-rate tonic branch
> into a sustained, bounded, perturbation-returning and virtual-SEEG-readable
> non-tonic carrier?

## 1. Accepted upstream evidence

### 1.1 Phase C

Accepted as `ACCEPT_PHASEC_POST_RESULT_FUTILITY_STOP`:

- C0 completed 153/153 tasks across seeds 1/3/4;
- all three seeds were `mixed_or_indeterminate_tonic_branch`;
- C1 completed 59/60 seed-1 primary continuations;
- corrected-v2 classified 59/59 as `tonic_non_AI`;
- modulation depth was 0.025–0.045 versus the registered non-tonic minimum 0.20;
- the one missing run cannot rescue any 5/6 cell or the seed-1-and-3 primary GO.

This rejects further expansion of the same frozen \(Z/M/S_G\) morphology grid.
It is not a three-seed bounded negative and does not prove carrier non-existence.

### 1.2 Phase D

Accepted as `NO_GO_baseline_calibration_failed_zero_spike_dominance`:

- the off-by-default conductance membrane, GABA conductance decomposition, sAHP
  representation and E-only threshold increment were implemented;
- five real state migrations and Arm-A continuation were byte-identical;
- the registered conductance replacement suppressed all E spikes during the
  8.5 s dynamic interictal baseline, so carrier arms were never opened.

This rejects that registered all-time conductance replacement.  It does not test
the dynamic-threshold mechanism by itself.

### 1.3 Lifecycle scale diagnostics

The 2026-08-01 calculations are retained as diagnostics, not a hard gate:

- current \(m\) scale is small relative to the single-cell reset gap;
- \(z\) cannot be the sole offset coordinate;
- \(S_G\) provides little post-activity memory;
- the Phase-C tonic core has high refractory occupancy.

None is a mathematical carrier/lifecycle non-existence proof.  D1 GO/NO-GO is
determined only by the behavioral and state-space gates below.

## 2. Lifecycle roadmap and authorization firewall

The full roadmap is fixed in this order:

1. **D1 — fast carrier:** frozen \(z,m\); establish an observation-consistent,
   bounded, spatially structured fast carrier.
2. **D2 — endogenous entry:** release \(z\); repeated returning events must alter
   onset hazard and enter the same D1 carrier without kick/timer/switch.
3. **D3 — native offset:** on the confirmed carrier, test whether \(m\) supplies
   an activity-dependent exit boundary; only then consider one backup variable.
4. **D4 — postictal recovery and multicycle:** return to the same distribution of
   interictal events and repeat at fixed parameters.
5. **D5 — controllable termination:** a finite, matched-energy intervention must
   shorten an existing native lifecycle by pushing it into the native recovery
   corridor, without reset or permanent silence.
6. **D6 — shared-scaffold causality:** rotate/weaken/randomize the pathological
   connection axis and separate geometry, entry, duration and recovery roles.

No D1 artifact may emit a positive field for D2–D6.

## 3. Locked substrate

Reuse the exact current-based per-neuron Z/M family:

- E1146 `twoend_equal` two-dimensional anisotropic E/I SNN;
- \(N_E=32000\), \(N_I=8000\), \(L=20\) mm;
- existing heterogeneous two-core threshold substrate;
- per-neuron \(z_i,m_i\), `use_qI=False`, `use_gK=False`;
- \(\tau_z=5000\) ms, \(\tau_m=500\) ms, \(\eta_m=0.001\);
- existing recurrent-only \(S_G\), \(\alpha_G=16\), \(\tau_S=80\) ms;
- calibrated seeds `{1,3,4}` and their existing exact checkpoint/noise lineage;
- original virtual-SEEG montage and pathology axis;
- current-based membrane and synaptic current semantics.

Locked against change in D1:

- all E→E edges, weights, kernel, anisotropy, orientation, STD and plasticity;
- E/I external drives and noise law;
- \(I_{th,EI}\), \(z/m\) equations and their constants;
- refractory periods, base thresholds and reset values;
- virtual-electrode geometry and carrier-gate thresholds;
- carrier observation windows after the input manifest is locked.

The Phase-D conductance path remains in the engine but is disabled throughout D1.
No conductance homotopy is a rescue arm in this spec.

## 4. The single new mechanism under test

D1 tests the already implemented, E-only, per-neuron threshold increment:

\[
\dot\phi_i=-\frac{\phi_i}{\tau_\phi}
+\Delta_\phi\sum_k\delta(t-t_i^k),
\qquad
V_{\theta,i}(t)=V^0_{\theta,i}+\phi_i(t).
\]

Properties:

- \(\phi_i\) is local to one E neuron; no spatial mask or global activity sensor;
- I-cell thresholds are unchanged;
- \(\phi_i=0\) is the exact baseline model state;
- the same equation is active from \(t=0\); no onset detector, timer or parameter
  switch is allowed;
- in D1 carrier forks, \(z_i,m_i\) are frozen but \(\phi_i\) remains dynamic;
- \(S_G\) is treated as a pre-existing 80 ms fast/intermediate loop and remains
  dynamic in the primary arm.  A frozen-\(S_G\) candidate ablation is required.

The hypothesis is not “AI is correct.”  The hypothesis is that neuron-local
threshold recovery can break the tonic fixed branch into a bounded population
carrier while spatial heterogeneity prevents whole-sheet synchronous reset.

## 5. Locked parameter panel

Use the six settings registered before the unexecuted Phase-D carrier stage:

\[
\tau_\phi\in\{60,100,160\}\ \mathrm{ms},
\qquad f_\phi\in\{0.15,0.30\}.
\]

At the locked seed-1 Phase-C tonic reference \(r_{core,ref}\), define

\[
\Delta_\phi\left(\frac{\tau_\phi}{1000}\right)r_{core,ref}
=f_\phi(V_\theta-V_{reset}).
\]

Here \(\tau_\phi\) is stored in ms and \(r_{core,ref}\) in Hz; the explicit
factor of 1000 is mandatory.  Thus \(\Delta_\phi\) is the mV increment per E
spike.  The implementation and manifest must record both the source units and
the converted \(\tau_\phi\) in seconds; a dimensionally inconsistent value is
an input-contract failure, not a runnable setting.

The resulting six numerical \(\Delta_\phi\) values are written before any D1
candidate run and reused unchanged for all seeds.  They are not recalibrated per
seed and no intermediate value is added after results are seen.

The analytic preflight must report, but not adjudicate:

- predicted tonic steady threshold shift;
- residual fraction \(e^{-T_{IED}/\tau_\phi}\) across the locked baseline IED
  interval distribution;
- exact `phi=0` default-path parity;
- maximum observed \(\phi\) in baseline and carrier forks.

## 6. Minimal arms

| Arm | Slow state | \(S_G\) | \(\phi\) | Purpose |
|---|---|---|---|---|
| `A_native` | dynamic for baseline; frozen \(z,m\) for carrier fork | dynamic | off | exact current-based control |
| `B_phi` | same as A | dynamic | one of six locked settings | primary D1 mechanism |
| `B_phi_SGclamp` | frozen \(z,m\) | frozen at checkpoint value | candidate setting | whether dynamic \(S_G\) is required |
| `B_phi_noSG` | frozen \(z,m\) | uncoupled, \(\alpha_G=0\) | candidate setting | whether containment is required |

The two ablations run only for settings that pass the seed-1 unperturbed carrier
screen.  They explain the mechanism; they cannot rescue a failed `B_phi` setting.

Forbidden D1 arms:

- conductance replacement or conductance homotopy;
- changes to \(m\), \(z\), \(S_G\) constants;
- H, P, q_I, g_K, pump, chloride or reset variables;
- E→E tuning;
- stimulation/termination optimization;
- large slow-state morphology grids.

## 7. Phase 0 — observation target lock

Before any candidate is inspected, write one immutable observation contract.

### 7.1 Existing operational carrier gate

Reuse `carrier_gate_v2.1_revised_2026-07-24` unchanged.  Its hard semantics are:

- source and virtual-SEEG macroepisode duration at least 2 s;
- occupancy at least 0.80;
- no full-return gap longer than 250 ms;
- no runaway, saturated sheet or simultaneous whole-field flash;
- at least two sustained 30–80 Hz contacts with temporally overlapping
  80–150 Hz or 1–150 Hz enhancement;
- real duration/duty/energy/spatial-extent separation from returning events;
- axial first-passage recruitment rather than a simultaneous flash.

### 7.2 Real-data reference sidecar

Freeze a descriptive, single-seizure reference from the already accepted
E1146 seizure-7 Fig3-A lineage:

- CAR reference, 15 locked contacts, `SCL9` spectral anchor;
- baseline `[-120,-90)` s;
- clinical early-ictal window `[0,10)` s;
- bands 30–80, 80–150 and 1–150 Hz;
- duration above 6 dB, occupancy, maximum gap, mean/peak dB, active-contact
  fraction, dominant frequency and spectral entropy.

The current accepted summary already reports clinical `[0,10)` means of about
23.34 dB (30–80 Hz), 11.48 dB (80–150 Hz) and 16.22 dB (1–150 Hz) on SCL9.
The new sidecar must derive the remaining features from the same raw lineage
before D1 outcomes are opened.

Because this is one representative seizure and the model LFP proxy is rectified
current amplitude, the sidecar is a directional observation comparator, not a
patient-level fitted likelihood.  D1 may claim `virtual_seeg_carrier_candidate`,
not `patient_matched_seizure`.

If the real reference cannot be rebuilt from its locked source, D1 stops as
`BLOCKED_observation_reference` rather than silently dropping the observation
layer.

## 8. Phase 1 — dynamic interictal baseline preservation

For every one of the six `B_phi` settings, run canonical dynamic Z/M from \(t=0\)
through the locked 8.5 s pre-escalation window with replay noise first.

A setting is baseline-eligible only if, relative to paired `A_native`:

- returning-event count, median duration and median core peak each stay within
  ±20%;
- at least one event from each registered core remains readable;
- event-order/pathology-axis sign is unchanged;
- all-sheet mean rate and peak active fraction stay within ±20%;
- no prevention, whole-sheet plateau or runaway occurs;
- post-event \(\phi\) decays to at most 10% of its event peak before the next
  returning event in at least 80% of eligible intervals.

Settings failing baseline are invalid, not carrier negatives.  If all six fail,
emit `NO_GO_D1_baseline_not_preserved` and stop.

## 9. Phase 2 — seed-1 frozen-state carrier screen

Use exactly four Phase-C forks:

- `bounded_mid__rising`;
- `bounded_mid__peak`;
- `bounded_late__rising`;
- `bounded_late__peak`.

Freeze the complete per-neuron \(z_i,m_i\) fields with membrane effects active.
Initialize \(\phi_i=0\), let dynamic \(S_G\) and \(\phi_i\) evolve, and use the
locked replay future noise.  There is no onset kick in this screen.

Each run lasts 6 s: 1 s burn-in followed by a 5 s adjudication window.  A transient
ringing response during burn-in cannot pass.

Cheap-first stop:

- run all baseline-eligible settings on bounded-mid rising/peak;
- a setting survives only if both phases pass the complete run-level source and
  virtual-SEEG gate;
- only survivors run bounded-late rising/peak;
- if no setting survives both checkpoints, emit
  `NO_GO_D1_fast_carrier_not_formed` and stop before additional seeds.

No near-miss parameter may be added.

## 10. Carrier acceptance

### 10.1 Primary endpoints

A run-level D1 carrier must satisfy all of the following:

1. **bounded persistence:** full 5 s analysis without runaway, silence or sheet
   plateau; sustained macroepisode ≥2 s, occupancy ≥0.80, gap ≤250 ms;
2. **non-tonic structure:** modulation depth ≥0.20 with either reproducible
   periodic/clonic organization or spatially relayed activity; AI is not required;
3. **virtual-SEEG carrier:** full revised v2.1 Gate B, including overlapping
   low-gamma/high-frequency enhancement and 4-D separation from returning events;
4. **spatial organization:** at least two separated zones, local-to-extended
   recruitment, no simultaneous flash, reproducible axial latency/phase sign;
5. **not microscopic hard saturation:** the candidate must not satisfy the locked
   combined refractory-saturation definition.  AI/regularity metrics are reported
   only as secondary mechanistic descriptors.

### 10.2 Fixed perturbation-return test

Before candidate outcomes are inspected, calibrate one 50 ms **uniform E-threshold
uplift** on `A_native` `bounded_mid__rising`.  During the pulse only,
\(V^0_{\theta,i}\mapsto V^0_{\theta,i}+u_{pert}\) for every E cell; I thresholds,
all currents, slow variables and RNG state are untouched.  Use a deterministic
bisection to find the smallest uplift reducing paired core spikes by 50–70%
without ≥100 ms all-sheet rest.  Freeze its amplitude and apply it at exactly
3.0 s after every fork (1 s burn-in + 2 s established analysis).  The spatial
mask is always all E cells; no candidate-specific mask or timing is allowed.

For every survivor:

- apply the pulse after ≥1 s of established carrier;
- remove it without resetting any fast/slow/RNG state;
- require return within 1 s to the same carrier class;
- median period/frequency must agree within 20%;
- spatial phase-sign must agree and phase-profile circular correlation must be
  at least 0.80.

Survival without return is `metastable_survival`, not attractor support.
This pulse is a state-space perturbation diagnostic only.  It is not D5
stimulation and cannot support a controllable-termination claim.

### 10.3 Seed-1 setting decision

A setting passes seed 1 only if:

- bounded-mid rising and peak both pass;
- bounded-late rising and peak both pass or one checkpoint passes while the
  adjacent checkpoint is technically indeterminate rather than tonic/runaway;
- the perturbation-return test passes in both fast phases;
- the same setting passed the dynamic interictal baseline gate.

## 11. Replication and numerical confirmation

Only seed-1 survivors are replicated.

For seeds 1, 3 and 4:

- two fast phases × three locked future-noise continuations;
- cell support requires at least 5/6 passes and at least 2/3 per fast phase;
- seeds 1 and 3 must both support the same setting/checkpoint carrier class;
- seed 4 must be concordant or indeterminate, not opposite silence/runaway;
- independent \(dt/2\) on seeds 1 and 3 must preserve carrier class and median
  frequency within 20%;
- perturbation return must replicate on seeds 1 and 3.

Only this stage may emit `fast_ictal_carrier_supported`.

## 12. Verdict vocabulary

Allowed D1 verdicts:

- `BLOCKED_input_or_observation_reference`;
- `NO_GO_D1_baseline_not_preserved`;
- `NO_GO_D1_fast_carrier_not_formed`;
- `NO_GO_D1_hfo_like_burst_train`;
- `NO_GO_D1_tonic_or_saturated`;
- `metastable_survival_without_return`;
- `resolution_sensitive_fast_carrier`;
- `seed_heterogeneous_fast_carrier`;
- `virtual_seeg_carrier_candidate_seed1`;
- `fast_ictal_carrier_supported`.

Every verdict must separately store:

- `interictal_baseline_preserved`;
- `source_carrier`;
- `virtual_seeg_carrier`;
- `spatial_pattern`;
- `perturbation_return`;
- `real_reference_comparison`;
- `entry=not_tested`;
- `offset=not_tested`;
- `recovery=not_tested`;
- `control=not_tested`;
- `lifecycle=not_established`.

## 13. Engineering and resource contract

- Reuse the tested `phi_increment` hook; no new guarded-engine mechanism should
  be needed unless audit proves an observation is missing.
- `use_phi=False` must retain byte parity and historical `BASELINE_SHA`.
- One immutable input manifest binds source checkpoints, exact RNG states,
  parameter grid, observation target, thresholds, code SHAs and panel IDs.
- Atomic per-run NPZ + JSON + resource receipt; resume only missing or technical-
  invalid runs; scientific failures are never retuned automatically.
- Set `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`.
- Measure one complete worker before parallel launch; keep at least 96 GB
  `MemAvailable`, reserve 8 logical CPUs, and use the existing 1.25× RSS formula.
- Sample worker `VmSwap` at least every 5 s and require a zero pre-publish snapshot.
- Never kill or alter peer-worktree processes.

## 14. Required outputs and figure boundary

Root: `results/topic4_sef_hfo/zm_d1_fast_ictal_carrier/`.

Required machine outputs:

- immutable D1 input manifest;
- real-data reference lock and provenance;
- baseline-preservation matrix;
- seed-1 carrier screen;
- fixed perturbation lock and return matrix;
- conditional multi-seed and \(dt/2\) summary;
- one fail-closed D1 verdict.

Required figures:

1. baseline event preservation and \(\phi\) decay;
2. arm/control source rate, E/I lag, virtual-SEEG energy and spatial kymograph;
3. perturbation-return phase portrait / return map;
4. parameter/checkpoint/seed coverage and verdict.

All figure directories require Chinese `README.md`.  No Figure-5 lifecycle layout
is authorized unless a later D4 multicycle lifecycle passes.  D1 figures are
carrier-existence diagnostics only.

## 15. Final claim boundary

If D1 passes, the strongest allowed statement is:

> On the unchanged anisotropic per-neuron Z/M SNN scaffold, a registered
> neuron-local threshold-recovery feedback supports a replicated, bounded,
> perturbation-returning and virtual-SEEG-readable fast ictal-carrier candidate
> at frozen slow states.

D1 cannot establish spontaneous entry, native offset, postictal recovery,
multicycle dynamics, stimulation efficacy or a complete ictal lifecycle.
