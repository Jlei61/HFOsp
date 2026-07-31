# Topic 4 Phase D — Z/M SNN-native fast-carrier repair design

**Date:** 2026-07-31

**Status:** LOCKED DESIGN — implementation not started

**Scope:** correct per-neuron Z/M SNN only; frozen fast-carrier gate first

**Independent line:** E→E weights, kernel, anisotropy, topology, STD and structural plasticity are frozen

---

## 1. Scientific decision

Phase C established a narrow but decisive result. Across 59/60 seed-1 primary frozen-state continuations:

- all corrected-v2 phenotypes were `tonic_non_AI`;
- pathology-core E rate was 435.5–442.6 Hz;
- all-sheet E rate was 140.1–160.2 Hz, below the whole-sheet runaway gate;
- fine-rate modulation depth was only 0.025–0.045, below the registered 0.20 non-tonic gate;
- every primary cell was already incapable of reaching the required 5/6 positive runs.

Therefore moving frozen \(z_i,m_i,S_G\) coordinates does not create the required carrier in seed 1. It moves the operating point on a local high-rate tonic branch. Continuing the same slow-field sweep cannot rescue the registered primary GO.

Phase D changes the **fast inhibitory/membrane feedback** while preserving the original spatial SNN substrate. Its first question is:

> At a frozen slow state already known to support bounded activity, can the native E/I spiking network support a sustained, bounded, spatially structured non-tonic carrier rather than a flat tonic plateau or an HFO-like burst train?

Entry, autonomous offset and recovery are conditional downstream questions. They are not tested until the fast-carrier gate passes.

---

## 2. Core scientific hypothesis

The current current-based membrane equation is approximately

\[
\tau_V\dot V_i=-V_i+I^E_i-z_i I^I_i-\eta_m m_i .
\]

For frozen \(z_i,m_i,S_G\), this is predominantly an additive shift of net drive. It can move a fixed point, suppress it, or drive it to a high-rate branch, but it does not necessarily change the slope and time scale of the fast membrane response enough to create a new oscillatory instability.

Phase D tests a more specific hypothesis:

1. local and weak-global GABA act as **conductances**, so inhibition changes both the membrane fixed point and effective membrane time constant;
2. existing delayed recurrent arrivals and AMPA/GABA synaptic filtering provide the fast negative-feedback loop;
3. a spike-history threshold \(\phi_i\) on a 60–160 ms scale supplies the intermediate feedback needed if conductance alone only produces another tonic fixed point;
4. seconds-scale \(z_i\) and the existing intermediate \(m_i\) feedback then move the network into and out of the fast carrier window.

The intended slow-fast geometry is:

\[
\text{interictal basin}
\xrightarrow[\text{seconds}]{z\downarrow,\;m\text{ state}}
\text{bounded fast carrier}
\xrightarrow[\text{subsecond--seconds}]{m\uparrow,\;z\uparrow}
\text{interictal basin}.
\]

Within the middle window, the fast subsystem may have a stable oscillatory orbit or a bounded phase-staggered carrier. The whole seizure need not be a permanent limit cycle.

### 2.1 Explicit falsification risk

Conductance conversion alone may only compress the existing tonic branch. That is expected to be common, not treated as an implementation failure. A Phase-D GO requires a new fast dynamical object: non-tonic carrier, perturbation return, and a bounded spatial pattern. Lower mean firing alone is not a GO.

### 2.2 Pre-result dynamical prediction

At fixed synaptic inputs and frozen slow state,

\[
\frac{\partial V_\infty}{\partial g_I}
=\frac{E_I-V_\infty}{g_\Sigma}<0,\qquad
\frac{\partial\tau_{\mathrm{eff}}}{\partial g_I}
=-\frac{C}{g_\Sigma^2}<0
\]

whenever \(V_\infty>E_I\). Conductance inhibition therefore changes both the
fast fixed point and its relaxation rate. By itself this commonly increases
damping; it is not assumed to create a Hopf bifurcation.

The dynamic-threshold arm adds a delayed negative-feedback pole
\(-1/\tau_\phi\) to the E/I fast block. The registered prediction is:

- arms B/C may lower and bound the tonic branch without making it oscillatory;
- if arm D alone develops a complex pair that crosses the stability boundary,
  the carrier should disappear when \(\Delta_\phi=0\) and return after a fixed
  perturbation;
- if B/C/D all remain low-modulation fixed branches, this inhibitory/membrane
  route is falsified at the tested checkpoints even if their mean rates differ.

The implementation must estimate this prediction numerically from paired
state forks: local finite-difference response/monodromy on the recorded
E-rate, I-rate, mean-\(V\), conductance and threshold-increment observables.
The sign of a fitted eigenvalue is supporting analysis only; the registered
waveform, spatial and perturbation-return gates remain decisive.

---

## 3. Substrate lock and independence

All production arms use the same canonical substrate:

- subject geometry: `epilepsiae_1146`;
- placement: `twoend_equal`;
- \(N_E=32000,\;N_I=8000\);
- original two low-threshold cores;
- original anisotropic spatial scaffold and virtual-SEEG montage;
- correct per-neuron \(z_i\) and \(m_i\);
- canonical noise bank and exact state checkpoints;
- current Phase-C seed-1 primary/visited frozen checkpoints.

The following are immutable in this line:

- E→E graph, edge weights and kernel;
- E→E anisotropy and orientation;
- E→E STD;
- core placement and threshold substrate;
- external-drive statistics except for the registered finite trigger/control;
- electrode geometry and readout filters.

This keeps Phase D independent of the peer E→E mechanism line. Any E→E change is a new spec, not a rescue arm.

---

## 4. Conductance equations

### 4.1 Unit-safe conversion

Current engine variables `I_E` and `I_I` are voltage-drive accumulators, not conductances. They must never be inserted directly into a denominator.

Define non-negative model conductances

\[
g^E_i=\kappa_E[I^E_i]_+,\qquad
g^{I,L}_i=\kappa_I[I^I_i]_+,
\]

where \(\kappa_E,\kappa_I\) are calibrated conversion factors with explicit units in the model coordinate. The calibration contract is in §6.

Use normalized leak units \(g_L=1\) and
\(C=\tau_{m,E}g_L\). This preserves the original E-cell membrane time
constant when synaptic/adaptation conductances are zero.

The E-cell membrane equation becomes

\[
C\dot V_i=
g_L(E_L-V_i)
+g^E_i(E_E-V_i)
+g^{I,\mathrm{eff}}_i(E_I-V_i)
+g_Mm_i(E_K-V_i).
\]

For I cells, the original membrane update is preserved in Phase D unless a separate I-cell conductance arm is later reviewed. This isolates the E-cell carrier mechanism.

The exact exponential update is

\[
g_{\Sigma,i}=g_L+g^E_i+g^{I,\mathrm{eff}}_i+g_Mm_i,
\]

\[
V_{\infty,i}=
\frac{
g_LE_L+g^E_iE_E+
g^{I,\mathrm{eff}}_iE_I+
g_Mm_iE_K
}{
g_{\Sigma,i}
},
\qquad
\tau_{\mathrm{eff},i}=\frac{C}{g_{\Sigma,i}},
\]

\[
V_i(t+\Delta t)=V_{\infty,i}+
\left[V_i(t)-V_{\infty,i}\right]
\exp\!\left(-\frac{\Delta t}{\tau_{\mathrm{eff},i}}\right).
\]

Initial model-coordinate reversals are:

- \(E_L=0\), preserving the existing leak coordinate;
- \(E_I=V_{\mathrm{reset}}\), the existing shunting-GABA reference;
- \(E_K=E_L\), making \(m_i\) an outward sAHP conductance;
- \(E_E=2\,\mathrm{median}(V_{\theta,E}^{0})-V_{\mathrm{reset}}=25\)
  mV in the current substrate, a locked model-coordinate choice symmetric to
  the threshold-reset gap.

These are model-coordinate choices, not claims about patient tissue reversal potentials.

The conductance hook consumes the **raw decomposed** `I_E`, `I_I` and
`I_E_rec` values. In conductance arms, the legacy slow-layer
`apply_currents()` result must not be passed into the membrane step:

- \(z_i\) is applied exactly once, inside \(g^{I,\mathrm{eff}}_i\);
- \(m_i\) is applied exactly once, through \(g_Mm_i(E_K-V_i)\);
- the old subtractive `-z*I_I-eta_m*m` path is disabled only for the
  conductance arm;
- recurrent excitation is not divided by \(S_G\) in the conductance arm.

This is a P0 scientific and engineering invariant. Passing an already
Z/M-corrected signed `I_net` into the conductance hook would double-apply the
slow variables and make the arm uninterpretable.

### 4.2 Local and weak-global GABA

The effective inhibitory conductance in the primary full arm is

\[
g^{I,\mathrm{eff}}_i
=z_i\left[(1-\gamma)g^{I,L}_i+\gamma g^{I,G}\right],
\]

\[
g^{I,G}(t)
=\left\langle g^{I,L}_j(t)\right\rangle_{j\in E}.
\]

The primary weak-global fraction is \(\gamma=1/6\); \(\gamma=0\) is the
local-only ablation. \(g^{I,L}\) has already passed through the canonical
GABA arrival delays and rise/decay filter, so the primary global component is
the instantaneous spatial mean of that already-filtered conductance. It must
not receive a second unregistered low-pass filter. The convex mixture keeps
the inhibitory budget comparable on a spatially uniform state.
\(g^{I,G}\) is a uniform/low-rank conductance, not a divisor on recurrent
excitation.

Sensitivity only, after the primary panel is frozen:

\[
g^{I,\mathrm{eff}}_i
=(1-\gamma)z_i g^{I,L}_i+\gamma g^{I,G},
\]

which tests whether local \(z_i\) depletion should spare the global GABA component. This sensitivity cannot replace a failed primary panel.

The existing \(S_G\) variable may be reused as an observation/control state, but in conductance arms it must not simultaneously divide `I_E_rec`. No arm may contain both interpretations of the same \(S_G\) signal.

### 4.3 Z and M semantics

The slow-state equations remain the correct Z/M equations:

\[
\tau_z\dot z_i=
H(I_{\mathrm{th},EI}-I^I_i)-z_i,
\]

\[
\dot m_i=-\frac{m_i}{\tau_m}+S_i(t).
\]

Only the membrane coupling changes:

- \(z_i\) scales GABA conductance rather than subtractive inhibitory current;
- \(m_i\) gates an \(E_K\)-reversal sAHP conductance rather than a signed linear current.

The locked starting time constants remain
\(\tau_z=5000\) ms and \(\tau_m=500\) ms. Phase D does not silently retune
them during carrier discovery.

The frozen-carrier screen holds \(z_i,m_i\) fixed. Dynamic Z/M is re-enabled only after carrier acceptance.

---

## 5. Intermediate fast feedback: dynamic threshold

Phase C already shows that static slow coordinates do not produce a non-tonic carrier. Therefore the dynamic-threshold arm is part of the preregistered minimal panel, not an unregistered rescue after seeing Phase-D outcomes. Here \(\phi_i\) is a **threshold increment**, not the absolute threshold:

\[
\dot\phi_i=-\frac{\phi_i}{\tau_\phi}
+\Delta_\phi S_i(t),
\qquad
V_{\theta,i}=V_{\theta,i}^{0}+\phi_i ,
\]

where \(S_i(t)=\sum_k\delta(t-t_i^k)\). Thus \(\Delta_\phi\) is the
threshold increment per spike, \(\phi_i(0)=0\), and the heterogeneous
per-neuron \(V_{\theta,i}^{0}\) substrate is preserved exactly. The
steady-shift calibration below is dimensionally consistent. This feedback is
applied to E cells only; I-cell thresholds and membrane update stay on their
locked baseline path.

This process is slower than E/I synaptic feedback but faster than the locked
\(m_i\) and \(z_i\) processes:

\[
\tau_{E/I}<\tau_\phi<\tau_m\ll\tau_z.
\]

Locked \(\tau_\phi\) panel: \(\{60,100,160\}\) ms.

\(\Delta_\phi\) is calibrated by a dimensionless steady-shift target at the seed-1 tonic reference:

\[
\Delta_\phi\,\tau_\phi\,r_{\mathrm{core,ref}}
=f_\phi\left(V_{\theta}-V_{\mathrm{reset}}\right),
\qquad
f_\phi\in\{0.15,0.30\}.
\]

This produces six predefined \((\tau_\phi,f_\phi)\) settings. No post-result interpolation or “best-looking” increment is allowed.

Dynamic threshold is not called the seizure mechanism. It is a falsifiable carrier-forming feedback arm.

---

## 6. Calibration contract

### 6.1 Analytic current-to-conductance anchor

Let \(V_{\mathrm{ref}}\) be the median free-E membrane voltage measured from
the locked slow-off pre-entry reference, before any Phase-D candidate is run.
Matching the local current-based coefficients at \(V_{\mathrm{ref}}\) gives:

\[
\kappa_E^{(0)}
=\frac{1}{E_E-V_{\mathrm{ref}}},\qquad
\kappa_I^{(0)}
=\frac{1}{V_{\mathrm{ref}}-E_I},\qquad
g_M^{(0)}
=\frac{\eta_m}{V_{\mathrm{ref}}-E_K}.
\]

These equalities make the conductance arm tangent to
\(+I_E-I_I-\eta_m m\) at the reference voltage while allowing the feedback
gain and \(\tau_{\mathrm{eff}}\) to vary away from that point. They are the
unit-safe initialization, not fitted seizure parameters.

Only three dimensionless scale factors multiplying
\(\kappa_E^{(0)},\kappa_I^{(0)},g_M^{(0)}\) may be baseline-calibrated, each
within \([0.8,1.2]\). \(E_E,E_I,E_K,g_L,C\) are not tuned. Calibration uses
only the slow-off reference and is completed before any frozen tonic
checkpoint outcome is inspected.

### 6.2 Baseline preservation

Before any carrier screen, validate the anchored conversion on the slow-off returning-interictal reference.

The calibration objective is multi-constraint:

1. slow-off median E rate and returning-event count stay within ±15% of current-based baseline;
2. pre-entry virtual-SEEG event ordering and two-source geometry remain readable;
3. median \(V_\infty\) at pre-entry matches the current-based reference within 0.5 mV;
4. inhibitory-to-excitatory effective charge ratio matches within ±15%;
5. median \(\tau_{\mathrm{eff}}\) remains between 0.25 and 1.0 of the original \(\tau_V\);
6. no baseline whole-sheet plateau or prevention.

Calibration uses seed 1 only and a fixed deterministic solver. The objective
is minimized lexicographically in the order listed above, with deterministic
tie-breaking by distance from scale factors \((1,1,1)\). It locks one
parameter set before carrier outcomes are inspected. A parameter set that
suppresses all returning events is invalid, not a successful carrier negative.

### 6.3 Matched inhibition budget

For local-only and local+weak-global arms, the mean inhibitory conductance on the spatially uniform reference must agree within 5%. This prevents “global works because it is stronger” from confounding spatial topology.

### 6.4 Existing shunt code

`kick_probe.membrane_step(shunt_gaba=True)` is useful unit-tested prior art, but it cannot be used unchanged because the current `slow is not None` path bypasses it and because it lacks simultaneous per-neuron Z/M plus local/global conductance decomposition. Phase D requires one new off-by-default E-cell membrane hook with explicit decomposed inputs. Default-off paths must remain byte-identical.

### 6.5 Conductance-consistent virtual SEEG

The observation layer cannot continue to call the old signed drive
`I_E-I_I` a membrane current after the membrane model changes. Conductance
arms must record:

\[
I^{\mathrm{syn}}_i(t)
=g^E_i(E_E-V_i)+g^{I,\mathrm{eff}}_i(E_I-V_i),
\]

plus the sAHP term separately. Virtual-SEEG carrier energy is computed from
the same fixed electrode kernels applied to this conductance-consistent
synaptic-current proxy. The old current-based proxy is retained only as a
paired continuity diagnostic; it cannot adjudicate the Phase-D 30–80 Hz
carrier gate. Sampling rate, reference and filter definitions remain fixed
across arms.

---

## 7. Minimal preregistered arms

All arms use identical seed, checkpoint, future noise and finite trigger.

| Arm | Membrane / inhibition | \(\phi\) | Purpose |
|---|---|---:|---|
| A | current-based Z/M+\(S_G\), frozen slow state | off | exact Phase-C baseline |
| B | conductance Z/M, local GABA only (\(\gamma=0\)) | off | conductance effect without global containment |
| C | conductance Z/M + weak-global GABA (\(\gamma=1/6\)) | off | primary membrane/topology test |
| D | arm C | six locked \((\tau_\phi,f_\phi)\) settings | full fast-carrier candidate |

Arm B/C are necessary ablations even if D is the only candidate. Arm D may be run in the same cheap seed-1 wave after calibration; it does not wait for a full B/C multi-seed screen.

Forbidden in Phase D carrier discovery:

- \(H\), \(P\), reset, actuator or persistence-gated termination;
- E→E tuning;
- large \(z,m,S_G\) grids;
- changing carrier thresholds after seeing outcomes;
- calling lower tonic rate a carrier.

---

## 8. Cheap-first state-fork screen

### 8.1 Checkpoints

Use exactly three seed-1 frozen slow states:

1. pre-entry returning-interictal reference;
2. Phase-C bounded-mid tonic checkpoint;
3. Phase-C bounded-late tonic checkpoint.

Each state uses two fast phases and one replayed future-noise stream for the first pass. Duration is 4 s post-fork, long enough to contain at least 20 cycles at 5 Hz and to reject transient onset ringing.

### 8.2 Trigger and controls

- tonic checkpoints start from their saved fast state without an extra kick;
- pre-entry state receives one fixed local finite trigger calibrated before carrier outcomes;
- trigger-off controls must remain in the returning-interictal regime;
- aligned/noise-replayed comparisons share exact random streams.

### 8.3 Immediate futility stop

If no arm-D setting at either tonic checkpoint passes all run-level carrier gates in both fast phases, Phase D is `NO_GO_fast_carrier_not_repaired` and stops at seed 1. Do not expand to seeds 3/4, dynamic Z/M or lifecycle.

If one or more settings pass, lock all passing settings before any replication. No near-miss setting is added later.

### 8.4 Fixed perturbation protocol

The attractor test does not choose a perturbation after a candidate is seen.
Before arms B–D are inspected, arm A at the bounded-mid checkpoint calibrates
one 50 ms E-cell inhibitory threshold pulse over the pathology core. Its
amplitude is the smallest value that reduces the paired 50 ms core spike count
by 50–70% without producing an all-sheet empirical-rest dwell lasting
\(\ge100\) ms. The resulting amplitude, mask, onset and duration are frozen
and reused unchanged for every arm, checkpoint, fast phase and seed.

The pulse changes only the threshold during its registered window. It does not
reset membrane, refractory, synaptic, delay, \(z\), \(m\), \(\phi\), random
state or future noise. A paired no-pulse continuation is required for every
perturbed run.

---

## 9. Carrier acceptance

A run-level candidate must satisfy all of the following after a 500 ms
settling exclusion. Sustained-onset, empirical-rest, occupancy and spectral
event definitions reuse the locked
`carrier_gate_v2.1_revised_2026-07-24` protocol; the
only registered observation change is that conductance arms use the
conductance-consistent §6.5 virtual-SEEG input. Gate thresholds are not
re-estimated from Phase-D candidates.

### 9.1 Bounded source activity

- no all-sheet runaway: all-sheet E rate never has a sustained ≥250 Hz interval for 100 ms;
- active area does not remain ≥0.50 for 500 ms;
- no empirical-rest dwell ≥100 ms;
- carrier occupancy ≥0.80 for at least 2 s;
- no monotonic late-tail escalation.

### 9.2 Non-tonic temporal structure

- fine-rate modulation depth ≥0.20;
- at least 10 cycles or 6 clonic bursts;
- cycle-period CV ≤0.30 for periodic candidates, or burst-interval CV ≤0.50 for clonic candidates;
- at least 50% of active core cells participate in ≥6 cycles;
- active-core \(\rho_{80}<0.80\), preventing a near-ceiling tonic core from passing through envelope modulation alone.

### 9.3 Virtual-SEEG carrier semantics

On at least one preregistered active contact:

- baseline-normalized 30–80 Hz energy forms a macroepisode lasting ≥2 s;
- macroepisode occupancy ≥0.80;
- no energy-baseline gap ≥100 ms;
- the same event is not explained by isolated 80–150 Hz HFO bursts with inter-burst returns to baseline.

This is the target “sustained high-frequency ictal energy”, not the previous HYP/HFO-like short-burst train.

### 9.4 Spatial pattern

- at least two zones separated by one virtual-SEEG readout width have occupancy ≥0.80;
- first-passage spread exceeds a simultaneous-flash null;
- whole-sheet activation is not simultaneous;
- phase or latency structure is reproducible across the two fast phases.

### 9.5 Attractor test

At the same frozen slow state:

1. apply the fixed §8.4 inhibitory pulse after ≥1 s of established carrier;
2. remove it without resetting slow or fast state;
3. require return to the same carrier class, frequency band and spatial phase pattern within 1 s.

Survival without return is metastability, not an accepted carrier attractor.

### 9.6 Seed-1 cell decision

A parameter/checkpoint cell passes only if both fast phases pass all gates. A candidate is promoted to replication only if the same arm-D setting passes bounded-mid and bounded-late, or passes one with the adjacent checkpoint indeterminate rather than opposite runaway/silence.

---

## 10. Replication and numerical confirmation

Only locked seed-1 candidates proceed:

1. seeds 3 and 4, two fast phases, three future-noise continuations;
2. 8 s confirmation;
3. independent \(dt/2\) on seeds 1 and 3;
4. unchanged virtual-SEEG gate;
5. conductance/local/global/threshold ablations.

Confirmed fast carrier requires:

- ≥5/6 run-level passes per seed;
- both fast phases contribute ≥2 passes;
- native seeds 1 and 3 support the same setting;
- third seed concordant or indeterminate, not opposite runaway/silence;
- \(dt/2\) agrees in carrier class and median frequency within 20%.

---

## 11. Dynamical analysis

For each accepted carrier and its ablations, record:

- \(V_\infty\), \(\tau_{\mathrm{eff}}\), \(g_E\), local/global \(g_I\), \(z\), \(m\), \(\phi\);
- E/I population phase lag;
- Poincaré return map of carrier peaks;
- perturbation return time and phase reset;
- local response/Jacobian or finite-difference monodromy on a reduced observable basis;
- spatial phase-gradient and active-area trajectory.

Required interpretation:

- B/C tonic, D periodic: \(\phi\)-mediated intermediate feedback created the carrier;
- C periodic, B tonic/runaway: weak-global conductance is necessary for containment;
- B/C/D tonic: conductance/threshold panel did not repair fast dynamics;
- D burst train with rest gaps: reject as HFO/HYP-like train;
- carrier disappears under \(dt/2\): resolution-sensitive, no GO.

The analysis must locate a fast carrier boundary before dynamic Z/M is attempted. A pretty waveform without a return map or perturbation return is insufficient.

---

## 12. Conditional lifecycle stage

Only a confirmed fast carrier authorizes dynamic Z/M.

### 12.1 Entry

Starting from the original slow-off returning-interictal regime, re-enable \(z_i,m_i\) and ask whether repeated spontaneous interictal events move the state across the frozen carrier boundary without a seizure-trigger reset.

### 12.2 Offset

Require endogenous slow drift to leave the carrier window and terminate activity without \(H\), \(P\), actuator or artificial clamp.

### 12.3 Recovery

After offset:

- \(z_i\) recovers and \(m_i,\phi_i\) relax;
- the network returns to the same irregular intermittent-event basin;
- no fixed interictal rhythm is required or desired;
- early retrigger is suppressed and late retrigger recovers.

### 12.4 Lifecycle claim

Only entry + bounded carrier + autonomous offset + irregular interictal recovery across seeds supports `recoverable_ictal_lifecycle_candidate`. Carrier alone is `fast_carrier_supported`, not lifecycle.

---

## 13. Engineering and resource contract

1. New conductance path is off by default and must preserve current engine byte parity.
2. Unit tests cover:
   - current-path parity;
   - exact \(V_\infty,\tau_{\mathrm{eff}}\);
   - \(z\) scaling all/local GABA arms;
   - \(m\) reversal coupling;
   - dynamic-threshold increment, recovery and preservation of the heterogeneous baseline threshold;
   - local/global budget matching;
   - no double application of \(S_G\);
   - raw-current decomposition and exactly-once Z/M coupling;
   - conductance-consistent virtual-SEEG current and sampling cadence.
3. Raw full spike rasters are reduced in worker memory and never accumulated by the coordinator.
4. One full seed-1 worker measures peak RSS before parallel launch.
5. Set `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`.
6. Keep ≥96 GB `MemAvailable`; worker `VmSwap` must remain 0.
7. Maximum 12 concurrent full-SNN workers even if the memory formula allows more.
8. Every atomic run publishes content-addressed observables, part JSON and resource receipt.
9. Resume reuses only exact manifest/task/receipt matches.
10. Peer worktrees and peer processes are inventoried and never killed.

---

## 14. Outputs

Root:
`results/topic4_sef_hfo/zm_fast_carrier_repair/`.

Required:

- calibration manifest and baseline-parity report;
- immutable arm/parameter manifest;
- per-run observables and resource receipts;
- seed-1 cheap-screen verdict;
- candidate replication and \(dt/2\) verdict if authorized;
- frozen-state carrier boundary and perturbation-return analysis;
- virtual-SEEG sustained-energy gate;
- figures/README.md;
- one final claim-boundary JSON.

The first reader-facing diagnostic follows the Topic-4 SNN visual language but is not a paper-ready lifecycle figure. A Figure-5 lifecycle layout is authorized only after the conditional lifecycle stage passes.

---

## 15. Verdict vocabulary

Carrier stage:

- `NO_GO_baseline_calibration_failed`;
- `NO_GO_fast_carrier_not_repaired`;
- `NO_GO_hfo_like_burst_train`;
- `resolution_sensitive_carrier`;
- `seed_heterogeneous_carrier`;
- `fast_carrier_supported`;
- `blocked_evidence`.

Lifecycle stage:

- `not_authorized_without_fast_carrier`;
- `entry_not_established`;
- `offset_not_established`;
- `recovery_not_established`;
- `recoverable_ictal_lifecycle_candidate`.

Every verdict separately reports:

- `fast_carrier`;
- `entry`;
- `offset`;
- `recovery`;
- `spatial_pattern`;
- `virtual_seeg_energy`;
- `claim_boundary`.

---

## 16. Claim boundary

Phase D may establish that one explicitly defined inhibitory/membrane feedback structure supports a bounded, perturbation-returning, spatially patterned SNN carrier on the correct Z/M substrate.

It may not by itself establish:

- clinical seizure mechanism;
- patient-specific biophysical conductances;
- a complete ictal lifecycle;
- that Abbott/Liou is proved;
- that E→E alternatives are false;
- that microscopic HFO carrier biophysics has been explained.

The core goal remains:

> preserve the original anisotropic spatial SNN and returning interictal events, then obtain a controllable transition into a sustained, bounded, spatially patterned ictal-energy carrier and back to the same irregular interictal basin without reset.
