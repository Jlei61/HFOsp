# Z/M minimal-carrier-subsystem → entry/offset branch decision — revision 3.1 (2026-07-26)

**Status: REVISION 3.1 LOCKED FOR PHASE 0–3 BRANCH DECISION.**
Branch: `codex/topic4-m4-snn-native-exit`.

> This revision supersedes revision 2 at the same path. It is a **branch-decision
> spec**, not a complete lifecycle implementation spec.
>
> Revision 2 correctly stopped immediate construction of a new excitable field, but
> made two overly strong assumptions: that a carrier must live in pure fast E/I, and
> that failure at visited states implies the fast subsystem has no nearby carrier.
> Revision 3 removed both assumptions. Revision 3.1 additionally treats carrier
> persistence probabilistically, moves functional-rank analysis before offset
> attribution, expands the neighbourhood representation, and requires checkpoint
> hooks in the canonical simulator rather than a copied integration loop. It asks
> for the **smallest dynamic carrier
> subsystem**, audits the local slow-state neighbourhood before choosing Branch F,
> and requires multi-seed, multi-fast-phase, paired-future-noise evidence.

Canonical upstream acceptance:
`docs/archive/topic4/sef_hfo/zm_carrier_exit_line_acceptance_2026-07-26.md`.

---

## 0. Scientific question and claim boundary

The current Z/M substrate provides a **candidate event-driven escalation and
recruitment sequence**:

\[
\text{returning interictal events}
\rightarrow z\downarrow
\rightarrow \text{denser recruitment}
\rightarrow \text{runaway}.
\]

Whether this constitutes ictal entry remains conditional on finding an acceptable
carrier. Adding the divisive shared inhibitory pool \(S_G\) converts runaway into a
bounded recurrent burst train, but the existing source-space and virtual-SEEG gates
classify it as HFO-like rather than a sustained ictal carrier.

The present question is:

> At slow states sampled from, or locally adjacent to, actual Z/M+\(S_G\)
> trajectories, what is the **smallest dynamic subsystem** among fast E/I,
> fast E/I+\(M\), fast E/I+\(S_G\), and fast E/I+\(M+S_G\) that supports a
> bounded, spatially organised, statistically stationary or metastable carrier?

The roles are hypotheses, not pre-assigned facts:

- \(Z\): candidate event-driven escalation/recruitment and entry coordinate;
- \(M\): possible carrier component, burst-shaping/refractory component, or offset
  coordinate;
- \(S_G\): possible carrier component, containment component, or reset generator;
- fast E/I: necessary substrate, not assumed sufficient;
- \(P\) or recruited-area \(A\): conditional independent exit coordinates, tested
  only if the existing system has a carrier but lacks an adequate offset.

The full branch decision must distinguish:

\[
\boxed{
\begin{aligned}
&\text{no carrier in the local fast/slow neighbourhood};\\
&\text{carrier exists, but the current slow trajectory misses it};\\
&\text{carrier exists, but existing slow coordinates cannot reach offset};\\
&\text{carrier/offset exist, but the observation readout does not match}.
\end{aligned}}
\]

## 1. Locked substrate, invariants, and scope

### 1.1 Canonical SNN substrate

Reuse the current Z/M-native SNN:

- E1146 `twoend_equal` anisotropic substrate;
- \(L=20\) mm, \(N=40000\), \(N_E=32000\), \(N_I=8000\);
- per-neuron `use_z=True`, `use_m=True`, `use_qI=False`,
  `use_gK=False`;
- lockpoint `zA_q75_tz5000__mA0p001_tau500`;
- \(\tau_z=5000\) ms, \(\tau_m=500\) ms, \(\eta_m=0.001\);
- \(I_{\mathrm{th},EI}=1.28\), with calibration provenance retained;
- \(S_G\) initial anchor uses the already tested \(\alpha_G=16\);
- no \(H\), no persistence current, no q_I/g_K fallback;
- E→E weights, kernel, anisotropy, topology, and pathology geometry remain
  unchanged.

Canonical code lineage:

- `scripts/run_zm_snn_native_exit.py`;
- `src/snn_engine/kick_probe.py`;
- `src/snn_engine/slow_field.py`;
- `src/topic4_zm_carrier_gate_v2.py`;
- existing E1146 current-based virtual-SEEG recorder.

### 1.2 Complete canonical-config lock

Phase 0 must serialize one canonical config and SHA256 covering:

- all neuron constants and threshold heterogeneity;
- all synaptic weights, rise/decay kinetics, and E→E STD/adaptive-threshold
  switches, including confirmation that they are off when expected;
- delay distribution and discrete delay bins;
- external and OU noise processes, means, variances, filters, and seeds;
- boundary condition and geometry;
- all \(Z/M/S_G\) equations, time constants, normalization, saturation, and
  coupling strengths;
- integration \(dt\), duration, early-stop behavior;
- source-space event/runaway/plateau detectors;
- current-based and rate-based readout settings and kernels.

Changing any locked field invalidates state reuse and creates a new experiment
family.

### 1.3 In scope

1. full simulator-state inventory and exact-resume parity;
2. empirical carrier-target and readout lock;
3. trajectory-derived anchor/state capture in multiple primary seeds;
4. minimal carrier-subsystem forks across natural fast phases and paired future
   noise;
5. local trajectory-neighbourhood continuation;
6. standardized slow-coordinate effective-rank diagnostic;
7. trajectory-conditioned modal/operator audit;
8. \(Z\)-entry and existing-slow-coordinate offset probability boundaries;
9. matched offline exit-driver comparison;
10. one fail-closed branch decision.

### 1.4 Out of scope

- no implementation of Branch T, F, M-calibration, P, or A;
- no new actuator or H tuning;
- no E→E modification;
- no large global parameter grid;
- no claim of a complete lifecycle;
- no migration of the rejected local-inhibition field into SNN;
- no paper-ready lifecycle figure before recovery is established.

## 2. Phase 0A — state inventory and freeze semantics

### 2.1 Required state inventory

Before implementing snapshots, create a machine-readable inventory with one row
per state:

| category | required examples | inventory fields |
|---|---|---|
| membrane | \(V_i\), refractory counters | shape, dtype, update order, time scale |
| synaptic | \(s_E,s_I,I_E,I_I,s_{E,\mathrm{rec}},I_{E,\mathrm{rec}}\) | current contribution, decay |
| delays | spike/delay ring buffers and cursor | bin definition, \(dt\) dependence |
| external drive | OU state, Poisson/noise generator or pre-generated stream | mean, variance, filter |
| slow Z/M | full per-neuron \(z_i,m_i\), last \(I_I^E\) sensor state | field semantics |
| shared pool | \(r_E^\mathrm{fast},\mu_G,S_G\), pool filter/drive | output vs internal state |
| optional hidden states | STD, adaptive threshold, moving averages | on/off and current effect |
| RNG | bit-generator type and complete state | simulator vs observer RNG |
| observer | LFP/readout filters and buffers | proof they do not affect dynamics |

The inventory must explicitly say whether each row is:

- simulator state;
- observer-only state;
- dynamic or frozen in each arm;
- included in snapshot/restore;
- dependent on \(dt\).

Any unclassified state that affects membrane current is a P0 stop.

### 2.2 Freeze semantics

For a frozen coordinate \(q\):

\[
q(t>t_f)=q(t_f),
\]

while its membrane-current effect remains active.

Forbidden implementations:

- resetting the coordinate to a default;
- freezing only its mean and discarding the spatial field;
- stopping trace output while hidden state continues to update;
- clearing its current contribution;
- letting an internal filter drift while presenting a frozen output without
  declaring that mixed semantics.

For \(S_G\), the primary freeze arm freezes \(r_E^\mathrm{fast}\), \(\mu_G\),
and \(S_G\) as one coordinate family. A separate diagnostic may freeze output
\(S_G\) while allowing its sensor state to evolve, but it must have a different
arm name and cannot support the primary carrier verdict.

## 3. Phase 0B — exact-resume and paired-noise contract

### 3.1 Exact-resume parity

Implement checkpointing in the canonical simulator through a minimal,
off-by-default state hook. A copied second integration loop is forbidden.
`save_state`, `load_state`, and `freeze_policy` may delegate serialization and
policy logic to new modules, but the timestep update order remains single-source
inside `simulate_kick`.

Because `kick_probe.py` is whole-file SHA guarded, any hook changes its guard
hash even when the default numerical path is byte-identical. The required order
is:

1. record old SHA and exact guarded diff;
2. add the smallest gated hook with no new default-path RNG draw, allocation,
   or floating-point operation;
3. prove default and checkpoint-disabled spike/current byte parity against the
   pre-edit engine plus exact split/resume parity;
4. only then update the `kick_probe.py` entry in `engine_versions.json`;
5. record old/new SHA, tests, and justification in provenance.

“Byte parity” must never be used to claim that re-blessing is unnecessary.

Split a short deterministic run at \(t_f\):

1. continuous reference from \(0\) to \(T\);
2. run \(0\to t_f\), serialize, restore, continue \(t_f\to T\).

Required equality:

- spike raster byte-identical;
- all simulator-state arrays byte-identical at the end;
- source-space traces and current-based virtual-SEEG byte-identical;
- RNG state progresses identically.

The parity test must include active \(Z/M/S_G\), nonempty delay buffers, nonzero
synaptic currents, and a non-refractory/nontrivial fork.

Failure is P0: no scientific forks run.

### 3.2 Future-noise controls

Each exact snapshot supports:

- `noise_replay`: exact future stream from the anchor;
- `noise_resample_1`;
- `noise_resample_2`;
- optional `mean_input_only`: matched external mean with stochastic fluctuations
  removed.

Noise must be pre-generated or counter-based so all arms receive matched external
input. Turning noise off by deleting both fluctuations and mean input is invalid.
Observer RNG is separate from simulator RNG.

### 3.3 \(dt/2\) semantics

A \(dt\) snapshot cannot be directly reused at \(dt/2\). For resolution
confirmation:

1. rerun the full anchor independently at \(dt/2\);
2. select a homologous state using the same trajectory-observable criterion;
3. fork from the \(dt/2\)-native state.

Direct delay/refractory/state interpolation is allowed only after an explicit,
separately tested conversion contract.

## 4. Phase 0C — empirical carrier target and readout validation

### 4.1 Reference classes

Before viewing new fork results, lock immutable inputs for:

1. real returning interictal group events;
2. real early-ictal windows;
3. matched synthetic sharp pulse-train nulls.

If these artifacts cannot be resolved with provenance, verdict =
`blocked_reference_artifacts`. Do not substitute model-derived thresholds.
This blocks an observation-matched claim, not the source-space dynamical audit:
if the source-space baseline/rest lock is complete, Phase 1 may proceed but the
top-level result must retain `observation_layer_blocked`.

### 4.2 Data-locked observation gate

All observation thresholds are derived and locked from the three reference
classes, including:

- macroepisode duration;
- duty cycle/occupancy;
- energy enhancement;
- spatial extent;
- number of independent active contacts;
- harmonic-comb concentration;
- spectral entropy or equivalent broadband-continuity statistic;
- instantaneous-frequency distribution and drift;
- burst-interval coefficient of variation;
- temporal phase coherence;
- wavefront-velocity variability;
- spatial phase entropy;
- axial first-passage structure.

For each metric, the model must:

- exceed a pre-registered upper null quantile where larger is more ictal-like
  (or lower null quantile where appropriate);
- lie within a broad empirical early-ictal interval;
- remain outside the returning-interictal/pulse-train region.

The quantiles, multiplicity rule, missing-data rule, and minimum usable sample
count are locked before model inspection. Historical `occupancy>=0.8` and
`duration>=2 s` remain diagnostic comparators only; they are not the new
primary empirical gate unless the data lock reproduces them.

The pulse-train null is supplemented by a synchronized global-oscillator null
matched in mean rate, dominant frequency, and energy. A fixed low-frequency
global rhythm cannot pass merely because it has high occupancy. The gate must
separate broadband recruiting activity from both harmonic pulse trains and
stationary whole-field oscillations.

### 4.3 Physical readout

- primary: spatially weighted synaptic/transmembrane-current proxy from the
  existing LFP recorder;
- sensitivity: rate-based envelope/proxy;
- if only the rate proxy is available, the strongest allowed label is
  `observation_proxy_carrier`.

“Multiple contacts” requires at least two active contacts separated by more
than one readout-kernel width. Two adjacent contacts receiving the same hotspot
do not establish spatially distributed recruitment.

### 4.4 Source-space gate and rest-state distance

A source-space carrier must:

- remain bounded below runaway and below a saturated plateau;
- persist statistically beyond returning-event lifetimes;
- retain spatial organisation beyond a single fixed hotspot;
- not repeatedly return to the pre-event rest/interictal distribution.

Troughs are allowed. Define a multivariate, baseline-standardized rest distance:

\[
d_\mathrm{rest}(t)=d\!\left[
r_\mathrm{core},r_\mathrm{surround},A_\mathrm{active},
E_\mathrm{vSEEG},H_\mathrm{spatial}
\right].
\]

A trough is not a reset if it is brief and remains outside the empirical
pre-event rest distribution. Threshold and dwell time are data-locked in
Phase 0C.

### 4.5 Source-only policy

A `source_space_carrier` may proceed to modal, entry, and offset diagnostics.
It may **not** authorize a new P/A actuator or a lifecycle implementation until
the observation layer is validated.

### 4.6 Phase 0D — one end-to-end vertical-slice smoke

Before building the full empirical/fork matrix, run one non-claim-bearing
seed-1 vertical slice:

\[
\text{short anchor}
\rightarrow\text{snapshot}
\rightarrow\text{restore}
\rightarrow\texttt{freeze\_all}
\rightarrow\text{short continuation}
\rightarrow\text{source/current-readout metrics}.
\]

The smoke checks interface shape, freeze behavior, readout continuity,
provenance, and metric production. It writes only to a smoke namespace and
cannot contribute evidence to any carrier or branch verdict.

## 5. Phase 1A — multi-seed trajectory anchors and natural microstates

### 5.1 Primary seeds and anchor eligibility

- cheap discovery: seed 1;
- primary confirmation seeds: `{1,3,4}`;
- a formal no-carrier verdict requires at least three seeds that actually form
  a bounded/contained anchor under the locked configuration;
- if fewer than three primary seeds form such an anchor, verdict =
  `insufficient_bounded_anchors`, not Branch F;
- additional seeds may be added only by a pre-registered extension, never
  cherry-picked after fork results.

Each anchor must contain returning events, escalation, and at least 4 s of the
contained burst-train state unless early runaway prevents it.

### 5.2 Slow-state bins

Select from trajectory observables, not fixed times:

- `pre_entry`;
- `onset_adjacent`;
- `bounded_early`;
- `bounded_mid`;
- `bounded_late`.

The bounded bins are defined along arc length/quantiles of the observed slow
trajectory, using full-field features rather than only time.

### 5.3 Natural fast-phase replication

Within each bounded slow-state bin, select real snapshots at:

- pre-burst/trough;
- rising/front;
- peak/active.

Snapshots must be naturally occurring states with similar slow coordinates;
membrane, synaptic, refractory, or delay states are never manually reset.

Record both the slow macrostate and fast phase. A carrier supported only by one
peak-phase snapshot is `isolated_carrier_candidate`, not a carrier window.

### 5.4 Snapshot contents

Save the complete Phase-0A inventory plus:

- state hash and canonical-config hash;
- seed, state-bin and fast-phase labels;
- physical time and trajectory coordinate;
- preceding event context;
- observer state;
- exact future-noise replay identifier.

The state lock is write-once and fail-closed.

## 6. Phase 1B — minimal carrier-subsystem audit

### 6.1 Fork arms

| arm | \(Z\) | \(M\) | \(S_G\) | question |
|---|---:|---:|---:|---|
| `dynamic_replay` | dynamic | dynamic | dynamic | exact-resume/trajectory control |
| `freeze_z` | frozen | dynamic | dynamic | fast E/I+\(M+S_G\) carrier? |
| `freeze_zm` | frozen | frozen | dynamic | fast E/I+\(S_G\) carrier? |
| `freeze_zsg` | frozen | dynamic | frozen | fast E/I+\(M\) carrier? |
| `freeze_all` | frozen | frozen | frozen | pure fast E/I carrier? |
| `dynamic_z_only` | dynamic | frozen | frozen | is \(Z\) drift alone sufficient to destabilize the frozen carrier manifold? |

Each scientific arm is crossed with `noise_replay`, `noise_resample_1`, and
`noise_resample_2`. `mean_input_only` is a diagnostic for the smallest positive
subsystem and matched negative control.

### 6.2 Burn-in and duration

Primary continuation length: 8 s after burn-in.

\[
T_\mathrm{burn}=
\max\left(250\ \mathrm{ms},
2\tau_{\max,\mathrm{dynamic\ carrier\ variable}}\right),
\]

unless a pre-registered stationarity detector requires longer. If \(M\) or
\(S_G\) remains dynamic, a universal 250 ms burn-in is forbidden.

Central positive candidates receive a 20 s confirmation and independent
\(dt/2\) homologous-anchor run.

### 6.3 Probabilistic stable/metastable carrier definition

For each continuation and each slow-state bin, estimate:

\[
P_\mathrm{carrier}(T)=
P(\text{the trajectory satisfies the source carrier state through }T)
\]

using the natural-fast-phase × paired-future-noise replicas. Report a
Jeffreys-prior beta-binomial posterior and interval rather than only a binary
pass/fail. Classify:

| class | locked interpretation |
|---|---|
| `stable_carrier` | posterior median \(P_\mathrm{carrier}(8s)>0.8\), bounded variance/drift, lifetime beyond the IED reference |
| `metastable_carrier` | \(0.3<P_\mathrm{carrier}(8s)\le0.8\), lifetime significantly longer than matched IEDs, no repeated rest-basin reset |
| `transient_carrier_like` | \(P_\mathrm{carrier}(8s)\le0.3\) or lifetime comparable to IEDs |
| `hfo_like_relaxation_train` | recurrent bursts repeatedly return to the rest/interictal distribution |

Threshold-edge posterior uncertainty produces `probabilistically_indeterminate`,
not an optimistic or pessimistic forced class.

A carrier window requires either a stable or metastable class and:

- compatible posterior carrier support in at least two adjacent slow-state bins;
- convergence across at least two natural fast phases after burn-in;
- confirmation in at least two of three eligible primary seeds; seed 1 alone is
  an `isolated_carrier_candidate` regardless of within-seed replication;
- no systematic monotonic drift in active area, energy, duty cycle, or slow
  carrier coordinates over the latter half;
- lifetime substantially longer than the data-locked returning-event
  distribution;
- source-space gate, and separately observation-level gate.

The smallest positive subsystem is determined by partial order:

1. `freeze_all` → `carrier_fast_only`;
2. otherwise `freeze_zm` and/or `freeze_zsg` →
   `carrier_fast_plus_sg` and/or `carrier_fast_plus_m`;
3. only `freeze_z` positive → `carrier_fast_plus_m_sg`.

If both one-variable subsystems pass, report both; do not force a unique winner.

### 6.4 Carrier taxonomy

- `carrier_fast_only`;
- `carrier_fast_plus_sg`;
- `carrier_fast_plus_m`;
- `carrier_fast_plus_m_sg`;
- `stable_source_space_carrier`;
- `stable_observation_carrier`;
- `metastable_source_space_carrier`;
- `metastable_observation_carrier`;
- `transient_carrier_like`;
- `probabilistically_indeterminate`;
- `isolated_carrier_candidate`;
- `hfo_like_relaxation_train`;
- `transient_active_state`;
- `saturated_plateau`;
- `runaway`;
- `no_carrier_in_visited_states`;
- `insufficient_bounded_anchors`;
- `no_evidence`.

If \(M\) is necessary in the smallest carrier subsystem, it cannot be treated
as an independent offset coordinate unless Phase 2B separates a slower
mean/load component from its carrier-period dynamics.

## 7. Phase 1C — local slow-state neighbourhood audit

This phase runs when visited states show no acceptable carrier or only an
isolated candidate. It prevents a false Branch-F decision.

### 7.1 Trajectory coordinates

Represent each visited slow state using preregistered full-field summaries:

\[
q(t)=
[z_\mathrm{core},z_\mathrm{surround},\Delta z_\parallel,
m_\mathrm{core},m_\mathrm{surround},\Delta m_\parallel,
S_G].
\]

Use three locked representations:

1. **coarse decision representation**: the seven summaries above, robustly
   standardized and reduced to the first two trajectory directions;
2. **full-field PCA interpretation**: vectorized \([z_i,m_i,S_G]\) fields,
   retaining \(u_1,u_2,u_3\) and reporting explained variance/spatial maps;
3. **pathology-axis projection**: preregistered parallel/perpendicular
   projections of \(z_i\) and \(m_i\), including axial gradient and
   core-boundary displacement.

Fit every basis using only locked anchor trajectories. The coarse
representation is the primary branch-decision coordinate; field PCA and axis
projection test whether a carrier window depends on a spatial direction hidden
by core/surround summaries.

### 7.2 Local continuation

Around selected `onset_adjacent`, `bounded_early/mid/late` anchors:

\[
q_\mathrm{test}=q_\mathrm{anchor}+a u_1+b u_2.
\]

The locked \((a,b)\) lattice is bounded by:

- interpolation between adjacent actually visited states where possible;
- no coordinate more than one robust trajectory SD from observed range;
- no clipping that silently changes the direction.

Construct fields by trajectory interpolation/PCA reconstruction, never by
independent arbitrary scalar changes. Run the coarse audit first, followed by a
matched local sensitivity in the first three full-field modes and pathology-axis
directions. Test the minimal carrier-subsystem arms with the same paired-noise
gate.

If the coarse and spatial representations disagree, verdict =
`representation_sensitive_no_branch`. This disagreement blocks Branch F and
requires a separately locked follow-up; it cannot be dismissed as
interpretation-only noise.

### 7.3 Branch decision

- carrier at visited states → Phase 1.5 and Phase 2;
- carrier absent at visited states but present locally →
  `Branch_T_slow_trajectory_repair`, requiring the local window in at least two
  of three eligible primary seeds;
- carrier absent across visited and local neighbourhood, with three eligible
  seeds, full replication, and no representation disagreement →
  `Branch_F_fast_carrier_repair`;
- insufficient evidence → stop without branch escalation.

Branch T may later adjust relative \(Z/M/S_G\) time scales, thresholds, or
coupling so the trajectory enters the existing carrier window. This spec does
not implement those changes.

## 8. Phase 1.5 — functional-rank then modal/operator audits

This is part of the main path, not reserved for Branch F.

### 8.1 Phase 1.5A — standardized slow-coordinate functional rank

Immediately after a carrier verdict, estimate central finite differences and
short-time impulse responses from existing slow coordinates \(q_j\) to
observables \(y_i\). Standardize:

\[
\widetilde S_{ij}
=
\frac{\sigma(q_j)}{\sigma(y_i)}
\frac{\partial y_i}{\partial q_j},
\]

where scales are robust ranges/SDs measured on locked trajectories.

Requirements:

- central differences under matched future noise;
- at least three frozen states;
- bootstrap over seeds and natural microstates;
- separate static-observable and impulse-response matrices;
- singular values and uncertainty intervals;
- unit-rescaling invariance.

Near rank-1 supports local functional collinearity only. It does not prove the
global slow manifold is one-dimensional. It changes the order of subsequent
questions:

- rank near 1: do not attribute offset to \(M\) alone; audit joint existing
  coordinates and prioritize a genuinely independent Phase-3 direction if no
  joint offset exists;
- rank 2–3: continue attribution of distinct entry, carrier, and offset
  directions.

### 8.2 Phase 1.5B — states and perturbations

At `pre_entry`, `onset_adjacent`, carrier, `offset_adjacent` if available, and
post-offset control, coarse-grain E/I activity to a locked spatial grid.

Apply equal-energy perturbations:

- axial mode;
- transverse mode;
- isotropic mode;
- core-localized mode;
- random matched control.

Use at least three perturbation amplitudes to identify a local linear range.
Fit the effective operator on a training subset and test prediction on held-out
perturbation shapes and time windows.

### 8.3 Tool matched to carrier type

- fixed point: eigenvalues/eigenvectors of the validated local operator;
- periodic carrier: Floquet/stroboscopic or Poincaré operator;
- stochastic/asynchronous carrier: DMD/Koopman/linear-response operator and
  finite-time singular gain.

Do not linearize only the time-averaged state of a periodic carrier.

### 8.4 Required outputs

\[
\alpha(q)=\max_j\operatorname{Re}\lambda_j,\qquad
G(T,q)=\|e^{A(q)T}\|_2,
\]

\[
\Delta_\parallel(q)=\alpha_\parallel(q)-\alpha_\perp(q).
\]

Also report:

- leading right and left modes;
- optimal finite-time input mode;
- angle to the pathology/E→E axis;
- axial/transverse propagation-speed ratio;
- held-out prediction error and amplitude-linearity range.

This distinguishes eigenvalue softening from non-normal transient
amplification and tests whether dominant mode organisation changes across
entry/carrier/offset.

## 9. Phase 2A — Z-entry probability boundary

Run only after a source-space carrier is identified.

### 9.1 Entry manifold

From matched pre-entry/interictal fast states:

- hold non-entry coordinates according to the smallest carrier subsystem;
- use actual \(z_i(t)\) fields or trajectory-manifold interpolation;
- start from matched rest/interictal initial states;
- apply the same locked IED-like perturbation or matched natural background;
- run paired future-noise continuations.

Estimate:

\[
P_\mathrm{enter}(q)=P(\text{enter carrier within }T).
\]

The empirical onset surface is \(P_\mathrm{enter}=0.5\), with bootstrap
uncertainty. Verify that the actual trajectory approaches/crosses the surface
in the entry direction.

Until this boundary exists, \(Z\) remains a candidate escalation coordinate,
not established ictal entry.

## 10. Phase 2B — existing slow-coordinate offset audit

### 10.1 Nested coordinate audits

Audit `bounded_early`, `bounded_mid`, and `bounded_late` carrier states in
three nested families:

1. **M alone**: actual \(m_i(t)\) fields/interpolation while \(Z/S_G\) are
   held consistently;
2. **M+\(S_G\)**: joint trajectory fields/coordinates, testing whether their
   combined direction supplies offset even if each alone is insufficient;
3. **M+\(Z\)-recovery coupling**: allow the actually specified \(Z\) recovery
   dynamics together with \(M\), with \(S_G\) treated according to the minimal
   carrier subsystem.

Use actual joint slow fields and interpolation:

\[
q_\mathrm{slow}(\lambda)
=(1-\lambda)q_\mathrm{slow}^\mathrm{early}
+\lambda q_\mathrm{slow}^\mathrm{late},
\qquad 0\le\lambda\le1,
\]

or PCA along the observed joint trajectory when more than one spatial
direction is needed. Whole-field scalar multiplication is sensitivity only,
not the primary atlas. If \(M\) is required by the carrier subsystem, the
M-alone family is descriptive rather than a valid removal/offset test.

Each state is launched from:

1. active carrier initial condition;
2. matched low/interictal initial condition.

This tests hysteresis and basin coexistence.

### 10.2 Joint probability boundary

For each nested family:

\[
P_\mathrm{remain}(q)=P(\text{remain carrier through }T).
\]

Define the empirical offset surface by \(P_\mathrm{remain}=0.5\), with
bootstrap uncertainty, and test:

\[
\nabla P_\mathrm{remain}\cdot\dot q(t)<0
\]

at actual exit-directed trajectory segments.

### 10.3 Outcomes and branches

- `existing_slow_offset_reached`: a boundary exists and the actual joint slow
  trajectory crosses it;
- `M_sufficient_and_reached`: the M-alone family is sufficient and reached;
- `M_SG_joint_offset_reached`: M+\(S_G\) supplies the reached boundary;
- `M_Z_recovery_offset_reached`: M+\(Z\) recovery supplies the reached boundary;
- `M_boundary_near_but_unreached`: boundary lies within a small pre-registered
  extension of actual \(M\) range → `Branch_M_calibration`;
- `M_boundary_far_unreached`: required \(M\) is far outside reachable range;
- `M_is_carrier_component`: dynamic \(M\) is necessary for carrier and no
  separable slower component is demonstrated;
- `M_shapes_but_no_offset_surface`;
- `no_M_evidence`.

Branch M-calibration is allowed only when a small, pre-registered change in
\(\eta_m\), \(\tau_m\), or high-activity drive could reach the nearby boundary
without preventing carrier formation. It is specified later and not
implemented here.

Only after all valid existing-coordinate families fail may
`M_boundary_far_unreached`, `M_is_carrier_component`, or
`M_shapes_but_no_offset_surface` proceed to Phase 3.

### 10.4 Distinct boundaries

Compare \(\Sigma_\mathrm{on}:P_\mathrm{enter}=0.5\) and
\(\Sigma_\mathrm{off}:P_\mathrm{remain}=0.5\). A lifecycle geometry requires
distinct surfaces or demonstrable hysteresis, not repeated crossing of one
threshold that regenerates the current burst train.

## 11. Phase 3 — matched offline exit-driver comparison

Run only if a source-space carrier exists and the valid existing-coordinate
families (\(M\), \(M+S_G\), and \(M+Z\)-recovery coupling) lack an adequate
offset.

### 11.1 Candidate observables

- \(D_\mathrm{mean}\): low-pass mean activity, **negative/control comparator**;
- \(D_\mathrm{load}\): local cumulative activity/ionic-load proxy;
- \(D_\mathrm{area}=\int A_\mathrm{recruited}(t)\,dt\);
- \(D_\mathrm{flux}=\int\max(0,dA_\mathrm{recruited}/dt)\,dt\);
- \(D_\mathrm{supra}\): carrier-selective suprathreshold-duration integral.

\(D_\mathrm{area}\) and \(D_\mathrm{flux}\) are reported separately: the first
integrates occupied territory over time; the second integrates newly recruited
territory.

### 11.2 Matched controls

Carrier and returning-event windows must be matched or reweighted on:

- duration;
- total spike count;
- total energy;
- peak rate;
- active-neuron count.

Also include spatial shuffle and temporal shuffle controls. A driver does not
pass merely because the carrier lasts longer.

### 11.3 Lexicographic acceptance

A driver becomes an actuator candidate only if it passes, in order:

1. carrier-vs-matched-IED separation;
2. persistence through carrier troughs;
3. reproducible accumulation within carrier;
4. incremental information beyond mean rate/total spike count;
5. decay compatible with postictal refractory and later \(Z\) recovery;
6. same-direction result in at least three seeds;
7. loss of effect under the appropriate shuffle.

Do not combine these into a tunable weighted score. Among passing drivers,
choose the simplest mechanism. P and A are never implemented together in the
first intervention.

Source-space carrier without observation validation may complete this
diagnostic, but cannot authorize actuator implementation.

## 12. Final branch-decision tree

```text
Phase 0
├─ 0A state inventory + freeze semantics
├─ 0B exact-resume parity + paired future noise
└─ 0C empirical carrier target + current/rate readout validation

Phase 1
├─ multi-seed, multi-fast-phase anchors
├─ minimal carrier subsystem:
│    E/I | E/I+SG | E/I+M | E/I+M+SG
├─ carrier at visited states
│    └─ Phase 1.5A functional rank
│         └─ Phase 1.5B modal/operator audit
│              └─ Phase 2A/2B
└─ no carrier at visited states
     ├─ local slow-state neighbourhood has carrier
     │    └─ Branch T: slow-trajectory repair
     └─ local neighbourhood has no carrier with adequate evidence
          └─ Branch F: fast-carrier repair

Phase 1.5
├─ A. standardized slow-coordinate functional rank
└─ B. effective spectrum / non-normal gain / spatial-mode audit

Phase 2A
└─ Z-entry probability boundary

Phase 2B
├─ existing slow-coordinate boundary reached
│    └─ existing-coordinate lifecycle attempt
├─ M boundary near but narrowly unreached
│    └─ Branch M-calibration
└─ M, M+SG, and M+Z-recovery have no usable offset
     └─ Phase 3

Phase 3
└─ matched offline P/load vs area/flux comparison
     └─ new spec implements one selected exit branch
```

## 13. Fail-closed verdict vocabulary

Top-level branch verdict is exactly one of:

- `blocked_state_inventory`;
- `blocked_exact_resume`;
- `blocked_reference_artifacts`;
- `insufficient_bounded_anchors`;
- `representation_sensitive_no_branch`;
- `carrier_at_visited_states`;
- `branch_T_slow_trajectory_repair`;
- `branch_F_fast_carrier_repair`;
- `branch_M_calibration`;
- `existing_M_lifecycle_candidate`;
- `phase3_driver_selection_required`;
- `observation_layer_blocked`;
- `no_evidence`.

Any missing required field, insufficient replica count, failed paired-noise
contract, or cache/config mismatch yields a blocked/no-evidence verdict. It
never defaults to Branch F.

## 14. Engineering and resource rules

1. Locks are write-once and fail-closed against spec SHA, git SHA, canonical
   config SHA, seed, state hash, arm, noise continuation, \(dt\), duration, and
   analysis version.
2. Smoke outputs never write production summaries.
3. Exact resume is a hard gate before scientific forks.
4. Snapshot serialization uses explicit schemas and versioning; no
   `allow_pickle=True`.
5. All trajectory-changing parameters are in provenance and filenames.
6. Missing acceptance fields raise or yield `no_evidence`.
7. Seed 1 is cheap discovery only; expensive expansion follows a positive or
   formally inconclusive gate.
8. Every \(dt/2\) result comes from its own anchor.
9. `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`.
10. Start with one 40k-neuron process and measure peak RSS.
11. Keep at least 96 GB `MemAvailable`, zero swap growth, and no more than two
    simultaneous full SNN workers unless a fresh RSS budget proves otherwise.
12. Crash-safe per-arm/per-state resume and durable resource logs are required.
13. New figure directories require Chinese `figures/README.md` and
    `results/FIGURE_INDEX.md` entry after figures exist.
14. The line remains independent of any E→E-modification worktree.

## 15. Deliverables and stopping point

This spec authorizes:

- full state/config inventory;
- exact snapshot/restore and paired-noise infrastructure;
- empirical carrier/readout lock;
- Phase 1 minimal-subsystem and neighbourhood audits;
- Phase 1.5 functional-rank and modal audits;
- Phase 2 entry/existing-slow-coordinate offset boundaries;
- conditional Phase 3 offline driver comparison;
- one archive report and one branch verdict.

It does **not** authorize implementing Branch T, F, M-calibration, P, A, or a
complete lifecycle. The correct stopping point is a defensible branch decision
that identifies whether the missing object is the carrier, the slow path, the
offset coordinate, or the observation layer.
