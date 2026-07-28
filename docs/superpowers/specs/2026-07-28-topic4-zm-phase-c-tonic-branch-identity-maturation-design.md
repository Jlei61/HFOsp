# Z/M Phase C — tonic-branch identity and carrier maturation design

**Status: LOCKED FOR PHASE C0–C1 ONLY (2026-07-28).**

Upstream branch-decision:
`docs/superpowers/specs/2026-07-26-topic4-zm-minimal-carrier-branch-decision-design.md`.

Upstream result:
`docs/archive/topic4/sef_hfo/zm_minimal_carrier_branch_decision_2026-07-28.md`.

This is a follow-up identity and neighbourhood experiment. It does not replace
the upstream result, alter the canonical Z/M substrate, or authorize a seizure
exit mechanism.

---

## 0. Scientific question and claim boundary

The upstream experiment established only that a sustained high-rate fast branch
can persist when trajectory-visited slow fields are clamped. Its population-rate
envelope is tonic, but that fact has two incompatible interpretations:

\[
\text{balanced asynchronous tonic activity}
\quad\text{or}\quad
\text{refractory-limited saturation}.
\]

Phase C asks two questions:

1. **C0 — branch identity:** is the clamped branch a non-saturated,
   asynchronous/irregular tonic source state, a refractory-limited plateau, or
   an unresolved/heterogeneous mixture?
2. **C1 — maturation neighbourhood:** within a small, preregistered
   slow-field neighbourhood supported by the observed Z/M/\(S_G\) trajectory,
   is there a contiguous bounded non-tonic carrier window?

The strongest result authorized by this spec is:

> A source-space tonic identity or a source-space carrier-maturation window
> exists in the tested frozen slow-field neighbourhood.

Phase C0/C1 acceptance is **not** evidence for:

- observation-matched ictal activity;
- spontaneous entry;
- autonomous offset;
- recovery to returning interictal events;
- a complete interictal–ictal lifecycle;
- an exit actuator.

The observation layer remains blocked until the real returning-event reference
contract is resolved. Current-based virtual-SEEG is descriptive in this spec.

---

## 1. Locked substrate and forbidden changes

Reuse the exact upstream canonical SNN family:

- E1146 `twoend_equal` anisotropic two-dimensional E/I substrate;
- \(N_E=32000\), \(N_I=8000\), \(L=20\) mm;
- per-neuron Z/M, `use_qI=False`, `use_gK=False`;
- lockpoint `zA_q75_tz5000__mA0p001_tau500`;
- \(\tau_z=5000\) ms, \(\tau_m=500\) ms, \(\eta_m=0.001\);
- \(S_G\) with \(\alpha_G=16\);
- the accepted canonical checkpoint/resume and future-noise bank;
- seeds `{1,3,4}` and their existing canonical config/state hashes.

The three seeds have separately calibrated \(I_{\mathrm{th},EI}\). They are
therefore a calibrated-connectome model family, not three pure RNG replicas at
one identical scalar operating point. All inference is first made within seed
and then combined by a seed-level rule. Raw rates or neuron-level samples must
not be pooled across seeds as independent observations.

Forbidden in Phase C:

- any E→E weight, kernel, anisotropy, topology, STD, or threshold-substrate
  change;
- changing \(I_{\mathrm{th},EI}\), \(\alpha_G\), Z/M equations, time constants,
  coupling strengths, external drive, noise law, refractory times, or neuron
  model;
- adding or tuning \(H\), \(P\), \(A\), q_I, g_K, persistence current, sAHP,
  pump current, chloride dynamics, or another actuator;
- changing a metric threshold, time bin, neuron panel, seed, fast phase,
  neighbourhood extent, or continuation duration after production results are
  viewed;
- expanding the grid after a negative result;
- selecting a visually favourable seed, state, contact, neuron subset, or
  future-noise realization;
- implementing Phase C2 or a lifecycle attempt.

No guarded-engine edit is allowed in Phase C. The existing full E raster,
opt-in I raster, current-based virtual-SEEG recorder, exact checkpoints, and
existing off-by-default threshold-perturbation hook are sufficient. Additional
observation must be implemented outside the guarded engine and must be
trajectory-transparent.

---

## 2. Locks written before production

**Pre-production contract amendment (2026-07-28):** the lock is an acyclic
two-stage chain, not one self-referential file:

1. `phasec_input_manifest.json`
   (`zm_phasec_input_v1_2026-07-28`,
   `production_authorized=false`) locks all upstream states, panels,
   thresholds, producer hashes, native configs and independent \(dt/2\)
   configs/anchors;
2. the native and \(dt/2\) coordinate manifests each point only to that input
   manifest and lock their own lossless float64 NPZ file and semantic hashes;
3. the sole production `phasec_manifest.json`
   (`zm_phasec_contract_v1.3_2026-07-28`,
   `production_authorized=true`) points forward to the input manifest and both
   coordinate manifests, including per-resolution/per-seed NPZ file and
   semantic hashes.

The final immutable Phase-C manifest must contain:

- this spec SHA, git SHA, engine SHA set, and upstream canonical-config SHAs;
- exact state paths and state hashes;
- primary seeds `{1,3,4}`;
- natural fast phases `{rising, peak}`;
- future-noise continuations
  `{noise_replay, noise_resample_1, noise_resample_2}`;
- continuation duration, burn-in, bin widths, neuron panels, core/surround
  masks, current panels, bootstrap settings, and all thresholds below;
- the complete C1 primary and shell coordinates before any C1 SNN run;
- intrinsic and empirical field-validity bounds;
- stop rules, resource rules, and output schema versions.

The manifest is write-once. Reuse requires exact SHA agreement. Missing or
mismatched fields yield a blocked verdict; no default is allowed.

Production runners reject the input manifest.  Final locking fails closed if a
coordinate manifest is absent or if any live production-producer hash differs
from the input lock.  This ordering removes the Phase-C-manifest ↔ coordinate-
manifest hash cycle.

Production results from an old run may be reused only if its saved raw
observables satisfy the new manifest exactly. Aggregate rates cannot substitute
for missing single-neuron spike, refractory, membrane, or current observables.

---

## 3. Phase C0 production matrix

### 3.1 Required coverage

For each seed `{1,3,4}`, use the canonical `bounded_mid` carrier witness and two
natural fast initial states:

- `bounded_mid__rising`;
- `bounded_mid__peak`.

Each is crossed with:

- `noise_replay`;
- `noise_resample_1`;
- `noise_resample_2`.

Thus the primary C0 identity matrix has:

\[
3\ \text{seeds}\times2\ \text{fast phases}\times3\ \text{noise futures}
=18\ \text{continuations}.
\]

Slow Z/M/\(S_G\) fields are frozen with their membrane effects active. Each
continuation is 8 s after a fixed 500 ms burn-in. A seed is identity-eligible
only if both fast phases and all three future-noise continuations complete.

The existing 2 s single-replay rhythm audit remains descriptive. It cannot
replace the C0 replication matrix.

### 3.2 Resolution confirmation

Any aggregate C0 identity supported at native \(dt\) receives an independently
generated \(dt/2\) homologous-anchor confirmation in at least two supporting
seeds. A native-\(dt\) checkpoint is never interpolated to \(dt/2\).

Native/\(dt/2\) disagreement yields `resolution_sensitive_identity`, not a
positive identity label.

### 3.3 Fixed observation panels

Neuron IDs are selected by a hash of:

`config_sha | seed | population | spatial_stratum | phase_c_version`.

The lock includes:

- all E neurons for streaming spike counts;
- all pathology-core E neurons for ceiling statistics;
- a fixed active-core/surround panel for ISI and membrane statistics;
- fixed core and surround E/I panels for effective-current traces;
- a fixed pairwise-correlation panel, selected without activity-result access.

If storage requires subsampling, the panel size and stratification are locked
before the smoke run and shared across all arms. No neuron is selected from its
observed firing rate after production.

Native and independent \(dt/2\) runs reuse the same activity-independent
anatomical neuron IDs.  The \(dt/2\) contract therefore records
`panel_selection_config_sha` as the parent native configuration SHA; it never
re-hashes/re-selects panels with the \(dt/2\) configuration SHA.

---

## 4. Phase C0 observables

### 4.1 Exact refractory ceiling

For population \(X\in\{E,I\}\):

\[
r_{\max,X}=\frac{1000}{\tau_{\mathrm{ref},X}}\ {\rm Hz},
\qquad
u_i(t)=\frac{r_i(t)}{r_{\max,X}}.
\]

The implementation must unit-test this formula against the exact engine
refractory update order. Under the currently locked E parameters,
\(r_{\max,E}=500\) Hz; the code must derive rather than hard-code it.

Use 250 ms sliding rate windows after burn-in. Define:

\[
\rho^{\rm core}_{80}
=
\operatorname{median}_t
\frac{
\#\{i\in E_{\rm core,active}:u_i(t)\ge0.8\}
}{
\#E_{\rm core,active}
}.
\]

`active` means at least 5 Hz over the post-burn interval. Report both
active-core and all-core denominators, all-E values, and active-neuron
fractions. The decisive quantity is active-core \(\rho_{80}\); a low whole-sheet
mean cannot hide a saturated local core.

Also report:

\[
f_{\rm ref}
=P\!\left(\mathrm{ISI}\le\tau_{\rm ref,E}+2dt\right).
\]

### 4.2 Paired local-susceptibility audit

From the same snapshot and future-noise stream, use the engine's existing
off-by-default `inhibitory_pulse` hook as a signed, source-core E-threshold
perturbation:

\[
\Delta V_{\rm th}\in\{\pm0.05,\pm0.10\}\ {\rm mV}.
\]

The target mask has length \(N\), is true only for source-core E neurons, and
the absolute perturbation window is derived from the resumed state's absolute
step. This diagnostic changes neither Z/M/\(S_G\), recurrent E→E, the external
drive distribution, nor the RNG stream.

Estimate the central slope of post-burn core rate for each amplitude, and
normalize it by the same probe applied to the seed-matched frozen `pre_entry`
state:

\[
G_{\rm rel}=\frac{g_{\rm carrier}}{g_{\rm pre-entry}}.
\]

The two amplitude estimates, in Hz/mV, must:

- have the same sign;
- differ by less than 25% relative to their mean;
- remain below the carrier runaway/plateau gates.

Failure gives `gain_unresolved`. It cannot be treated as zero gain.

### 4.3 Irregularity and asynchrony

Required spike observables:

- per-neuron ISI and local `CV2`;
- Fano factors at 5, 20, and 100 ms;
- active-neuron fraction;
- 5 ms pairwise spike-count correlations;
- a circular-time-shift null preserving each neuron's rate and autocorrelation.

The circular-shift null uses exactly 100 locked draws per block/pair stratum
(`pairwise_shift_null_draws=100`), recorded in every raw observable artifact;
missing or mismatched draw counts block C0 rather than silently changing the
null resolution.

The primary irregularity statistic is the seed/replicate median local `CV2`.
The primary synchrony statistic is the median pairwise count correlation plus
its position relative to the 97.5th percentile of the circular-shift null.

### 4.4 Required supportive diagnostics

The following are mandatory outputs but do not by themselves decide AI versus
saturation:

- effective E, effective I, recurrent-E, and net-current means, variances,
  ratios, cross-correlation, and optimal lag;
- membrane-potential distance to threshold and refractory occupancy;
- 1–2 ms population E/I rates and PSD;
- current-based virtual-SEEG PSD, spectral entropy, harmonic-comb
  concentration, and broadband continuity;
- active-area fraction, spatial entropy, active-centroid motion, core/surround
  rates, and axial kymograph;
- input-output slope at both diagnostic amplitudes.

`active-area fraction` is a spatial quantity, not the fraction of neurons that
fired.  On the locked \(16\times16\) E-rate grid, a bin is active when its
local mean E rate is at least 5 Hz.  The denominator is the set of
anatomy-occupied E grid bins (the occupied-bin count and whether all 256 bins
are occupied are stored explicitly).  C0 evaluates this at 25 ms; C1 applies
the same rule to its 2 ms E-rate grid.

Because the real observation reference remains incomplete, these diagnostics
cannot promote a source label to an observation-matched ictal label.

### 4.5 Uncertainty

Use a preregistered hierarchical bootstrap with 5,000 draws:

1. resample 500 ms time blocks within continuation;
2. resample neurons/pairs within the locked spatial strata;
3. resample the six fast-phase × future-noise continuations within seed.

Report seed-specific point estimates and 95% intervals. Seeds are the top-level
replication unit; neuron and time samples never inflate the seed count.

---

## 5. Phase C0 fail-closed taxonomy

### 5.1 Per-seed refractory saturation

A seed supports `refractory_saturated_branch` only when:

\[
\operatorname{LCB}_{95}(\rho^{\rm core}_{80})\ge0.50
\]

and at least one independent consequence of saturation holds:

\[
\operatorname{UCB}_{95}(G_{\rm rel})\le0.20
\]

or:

\[
\operatorname{LCB}_{95}(f_{\rm ref})\ge0.80.
\]

Both rising and peak phases must show the same direction, with at least two of
three future-noise continuations satisfying the run-level condition in each
phase.

The 0.8 ceiling defines proximity to a mechanistic hard limit. Fractions 0.50
and 0.80 require saturation to dominate the active core or its ISIs rather than
occur in a small tail.

### 5.2 Per-seed balanced asynchronous tonic candidate

A seed supports `balanced_AI_tonic_candidate` only when all hold:

\[
\operatorname{UCB}_{95}(\rho^{\rm core}_{80})\le0.20,
\]

\[
\operatorname{LCB}_{95}(G_{\rm rel})\ge0.50,
\]

\[
\operatorname{LCB}_{95}(\operatorname{median}CV2)\ge0.70,
\]

and:

- median 5 ms pairwise count correlation is below the circular-shift-null
  97.5th percentile;
- its absolute median is below 0.10;
- active-area fraction remains below the already locked whole-sheet plateau
  threshold of 0.50;
- there is no runaway, saturated whole-sheet plateau, or empirical-rest dwell.

Both fast phases and at least two of three future-noise continuations per phase
must agree. “Balanced” is a source-space candidate label; current diagnostics
must be reported, but the label does not claim a biophysical high-conductance
state or an ictal observation match.

### 5.3 Gray zones and conflicts

Any of the following yields `mixed_or_indeterminate_tonic_branch`:

- \(0.20<\rho^{\rm core}_{80}<0.50\) after uncertainty;
- \(0.20<G_{\rm rel}<0.50\) after uncertainty;
- nonlinear or sign-inconsistent gain probes;
- high ceiling occupancy without gain collapse/refractory locking;
- low ceiling occupancy without irregularity/asynchrony support;
- fast-phase disagreement;
- metric conflict.

Missing observables yield `C0_blocked_observables`; missing seed/phase/noise
coverage yields `C0_insufficient_coverage`.

### 5.4 Across-seed adjudication

An aggregate C0 identity requires:

- the same supported identity in at least two of three eligible seeds;
- the remaining seed is indeterminate or supports the same identity;
- no seed supports the opposite identity;
- required \(dt/2\) confirmation agrees.

Opposite seed identities yield `seed_heterogeneous_identity`. Majority voting
must not erase a mechanistically opposite third seed.

---

## 6. Phase C1 neighbourhood construction

C1 runs after a technically complete C0, including when C0 is mixed or
seed-heterogeneous. Seed disagreement must not suppress seed-specific
neighbourhood or modal analysis.

All bases are fit separately within seed from locked anchor trajectories.
Cross-seed PCA coefficients are never pooled.

### 6.1 Primary empirically supported neighbourhood

The primary neighbourhood uses complete slow fields, not independently varied
summary scalars.

For each seed, lock:

- exact full fields from `bounded_early`, `bounded_mid`, and `bounded_late`
  along the rising trajectory;
- the corresponding three fields along the peak trajectory;
- 50:50 convex interpolants between early–mid and mid–late within each phase
  trajectory.

This gives ten primary slow-field cells per seed: six actual fields and four
within-segment convex interpolants.

Every cell is launched from both canonical `bounded_mid__rising` and
`bounded_mid__peak` fast microstates and all three future-noise continuations.
Only the slow fields are replaced; membrane, synaptic, delay, refractory, and
RNG state remain exact.

These cells are called `primary_convex_reachable` in this spec: they are
empirically supported interpolations on the observed local slow-field manifold,
not proof that the dynamic slow flow visits every interpolated field.

### 6.2 Secondary local shell

Fit the locked full-field trajectory basis and align mode signs to the forward
trajectory derivative/pathology axis. Around each seed's bounded-mid slow
field, construct a fixed shell at exactly:

\[
\pm0.25\ \text{robust trajectory SD}
\]

along:

- the first two non-tangent full-field modes after the trajectory tangent;
- pathology-axis parallel field displacement;
- pathology-axis perpendicular field displacement.

The shell is locked before C1 production and is run regardless of the primary
scientific outcome once C1 starts. It must not be expanded to 0.5 or 1 SD after
a negative result.

Shell positives are `nearby_extrapolated_candidate` only. They cannot be called
a dynamically reachable maturation window.

The seven-coordinate empirical envelope is exactly:

\[
(z_{\rm core},z_{\rm surround},\Delta z_{\parallel},
  m_{\rm core},m_{\rm surround},\Delta m_{\parallel},S_G).
\]

Here the pathology axial coordinate \(a_i\) is centred and scaled to unit
population SD, and
\(\Delta z_{\parallel}=\sum_i a_i(z_i-\bar z)/\sum_i a_i^2\), with the
analogous definition for \(m\).  These two coordinates are field projections;
they are not core-minus-surround contrasts.  Definitions and engine units are
stored in each coordinate manifest.

The two non-tangent full-field modes are sign-aligned to the forward physical
trajectory derivative.  If that inner product is numerically zero, a fixed
activity-independent maximum-loading sign is used and explicitly flagged.
Every cell records reconstruction error and standardized distance from the
locked piecewise anchor manifold.

### 6.3 Physical-validity gate

Every reconstructed field must satisfy:

- no clipping;
- \(z_i\in[0,1]\), matching the actual Z/M engine clamp;
- \(m_i\ge0\);
- \(S_G\in[0,S_{\max}]\);
- every seven-coordinate summary lies within the seed-observed envelope plus
  0.25 IQR;
- full-field \(z/m\) values lie within the seed's locked 0.5–99.5 percentile
  envelope plus 0.25 IQR;
- the reconstruction error and distance from the anchor manifold are recorded.

Failure gives `invalid_physical_cell`. It is not silently projected, clipped,
or replaced. An invalid shell cell blocks a complete shell-negative verdict
but does not invalidate a complete primary-convex result.

Coordinate NPZ slow states and basis arrays are stored losslessly as float64.
The coordinate builder must prove semantic slow-state hash identity before and
after NPZ round-trip. Native seeds `{1,3,4}` use their six native anchors;
independent \(dt/2\) seeds `{1,3}` use their own six \(dt/2\) anchors. No
native coordinate, field, or checkpoint is interpolated to \(dt/2\).

---

## 7. Phase C1 carrier-phenotype taxonomy

Each C1 cell uses the same 8 s post-burn duration, two fast phases, and three
future-noise continuations as C0.

### 7.1 Common positive-carrier gate

A run can enter a C1 positive phenotype only if it:

- survives 8 s without runaway or a saturated whole-sheet plateau;
- does not satisfy empirical-rest dwell;
- has stationary or bounded-metastable late-half rate, active area, energy,
  and spatial entropy;
- retains activity in at least two zones separated by more than one
  readout-kernel width;
- is not a harmonic pulse train repeatedly resetting to rest.

For a cell-level positive:

- at least five of six fast-phase × noise continuations pass;
- Jeffreys posterior median \(P_{\rm phenotype}>0.8\);
- both fast phases contribute at least two passing noise continuations.

### 7.2 Phenotypes

`AI_tonic_window`:

**Pre-production clarification (2026-07-28):** this label is replaced by
`balanced_AI_tonic_cell`.  It:

- passes the common gate;
- satisfies every spike-only C0 AI condition at that cell;
- passes the same conditional \(G_{\rm rel}\), linearity, and uncertainty gate
  as C0;
- remains tonic at the population-envelope scale.

It is a terminal identity in the C1 atlas, not a non-tonic maturation
phenotype, and therefore cannot by itself support
`maturation_window_at_primary_convex_states`.

The C1 base matrix contains no gain perturbation.  After the complete base
primary and shell atlas exists, a cell becomes a locked
`spike_AI_screen_candidate` only if at least five of six base runs pass all
non-gain AI conditions and both fast phases contribute at least two passes.
All such cells are written once to `c1_gain_trigger_manifest.json` before any
C1 gain result is viewed.  For every triggered cell, run the complete
\(\{0,\pm0.05,\pm0.10\}\) mV carrier-gain set for both fast phases and all three
future-noise continuations; reuse the SHA-matched C0 pre-entry gain denominator.
No untriggered cell may be added later.

- nonlinear/sign-inconsistent gain or perturbation-induced runaway yields
  `tonic_gain_indeterminate` (scientific indeterminate, never zero gain);
- missing, truncated, or provenance-invalid conditional gain yields
  `C1_blocked_conditional_gain`;
- a triggered unresolved/blocked cell prevents a complete C1 negative, but
  cannot erase an independently complete periodic/clonic positive.

`periodic_non_tonic_carrier`:

- passes the common gate;
- contains at least ten post-burn cycles;
- fine-rate modulation depth
  \((P_{95}-P_{5})/\bar r\ge0.20\);
- cycle-period CV is at most 0.20;
- source phase structure is reproducible across the two fast initial phases.
- the cross-phase relative period difference is at most 0.20.

`clonic_or_bursting_carrier`:

- passes the common gate;
- contains at least six post-burn bursts;
- fine-rate modulation depth is at least 0.20;
- carrier-state occupancy is at least 0.80;
- no trough remains in the locked rest basin for the rest-dwell duration;
- burst-interval CV is at most 0.50.

`spatially_relayed_carrier` is an additional spatial modifier, not a standalone
temporal class. It requires:

- two active zones separated by more than one readout-kernel width;
- first-passage spread greater than the 97.5th percentile of a simultaneous
  flash/permuted-position null;
- reproducible direction or phase-gradient sign within seed.
- each separated zone has carrier-state occupancy at least 0.80.

`refractory_saturated` requires both active-core
\(\rho_{80}\ge0.50\) and refractory-locked ISI fraction \(\ge0.80\);
one threshold alone cannot create this label.

The locked implementation fields are
`c1_refractory_saturation_rho_min=0.50`,
`c1_refractory_isi_fraction_min=0.80`,
`c1_two_zone_occupancy_min=0.80`, and
`c1_periodic_cross_phase_period_rel_diff_max=0.20`.
The zone-separation scale is each seed's canonical
`params.Rr` in mm, stored as `readout_kernel_width_mm`; it is never hard-coded
from a guessed grid distance.

Other run/cell classes are:

- `tonic_non_AI`;
- `refractory_saturated`;
- `hfo_like_relaxation_train`;
- `rest_or_silence`;
- `runaway`;
- `probabilistically_indeterminate`;
- `invalid_physical_cell`;
- `missing`.

### 7.3 Maturation-window acceptance

A `maturation_window_at_primary_convex_states` requires:

- the same registered non-tonic phenotype
  (`periodic_non_tonic_carrier` or `clonic_or_bursting_carrier`) in at least
  two adjacent primary cells within seed;
- each cell passes the cell-level rule above;
- the same aligned slow-field direction supports it in at least two of three
  seeds;
- the third seed is either concordant or indeterminate, not an opposite
  saturation/runaway-only result;
- native and required \(dt/2\) confirmation agree.

An otherwise identical positive confined to the secondary shell is:
`maturation_candidate_in_secondary_shell`. It does not establish slow-path
reachability.

An isolated positive cell is `isolated_maturation_candidate`.

### 7.4 Negative coverage

`no_maturation_in_tested_primary_neighbourhood` is allowed only when:

- all three seeds are complete;
- all ten locked primary cells per seed are physically valid;
- every cell has both fast phases and all three future-noise continuations;
- every run has a terminal scientific class;
- every conditional-gain trigger has a terminal valid or scientific-
  indeterminate result, and no triggered cell remains technically blocked;
- no primary cell is missing, indeterminate, or representation-sensitive;
- no seed contains an isolated or contiguous non-tonic positive.

This is a bounded negative over the tested primary convex neighbourhood only.
It does not prove the SNN has no other carrier.

`no_maturation_in_tested_secondary_shell` additionally requires every locked
valid shell cell and all replication to complete. Invalid/missing shell cells
yield `secondary_shell_incomplete`, never a negative.

Coarse/full-field/pathology-axis disagreement yields
`representation_sensitive_maturation`. Seed disagreement yields
`seed_heterogeneous_maturation`. Neither defaults to fast-carrier repair.

---

## 8. Seed-specific modal analysis

Modal/operator analysis is routed per seed and per accepted phenotype:

- AI/stochastic candidate → DMD/Koopman/linear response and finite-time
  singular gain;
- periodic/clonic candidate → phase-conditioned/Poincaré or Floquet analysis;
- refractory-saturated branch → local gain and refractory sensitivity only;
  no seizure-mode claim;
- indeterminate identity → descriptive perturbation response only.

Carrier-class disagreement blocks one unified modal claim but does not cancel
valid seed-specific audits. Across seeds, compare leading spatial subspaces,
axis angles, and qualitative mode support; do not pool eigenvalues from
different calibrated operating points.

Modal results are explanatory. They cannot override C0/C1 taxonomy or authorize
a lifecycle mechanism.

---

## 9. Fail-closed Phase-C verdict vocabulary

Top-level Phase-C verdict is exactly one of:

- `C0_blocked_observables`;
- `C0_insufficient_coverage`;
- `resolution_sensitive_identity`;
- `refractory_saturated_branch_supported`;
- `balanced_AI_tonic_candidate_supported`;
- `mixed_or_indeterminate_tonic_branch`;
- `seed_heterogeneous_identity`;
- `C1_blocked_manifest`;
- `maturation_window_at_primary_convex_states`;
- `maturation_candidate_in_secondary_shell`;
- `isolated_maturation_candidate`;
- `no_maturation_in_tested_primary_neighbourhood`;
- `no_maturation_in_tested_secondary_shell`;
- `secondary_shell_incomplete`;
- `representation_sensitive_maturation`;
- `seed_heterogeneous_maturation`;
- `no_evidence`.

Every verdict must separately carry:

- `source_identity`;
- `primary_neighbourhood`;
- `secondary_shell`;
- `seed_specific_modal`;
- `observation_layer=blocked_reference_artifacts`;
- `entry=not_tested`;
- `offset=not_tested`;
- `recovery_lifecycle=not_established`;
- `phase_c2_authorized=false`;
- `actuator_authorized=false`.

No missing field may fall through to a positive or negative scientific verdict.

---

## 10. Engineering, resource, and recovery rules

1. Tests and synthetic fixtures precede any production SNN run.
2. The guarded engine must remain byte-identical. Observer wrappers must return
   exactly the underlying recorder output, and diagnostic-probe-disabled paths
   retain the already accepted byte parity.
3. One seed-1 smoke validates shape, units, refractory ceiling, panel identity,
   current signs, and output schema; smoke artifacts never enter production.
4. Every cell is one atomic JSON commit marker plus an immutable,
   content-addressed NPZ with exact state/noise/config provenance.  An NPZ
   orphaned before JSON publication cannot block an exact resume.
5. Coordinator merge rejects duplicate/conflicting rows and missing expected
   cells.
6. `--resume` is required to reuse exact terminal cells. Resume runs only
   missing or explicitly invalid technical cells; scientific failures are
   reused and never automatically rerun with changed settings.
7. The guarded upstream engine currently returns one transient full-E Boolean
   raster (and an opt-in I raster) per atomic continuation. Phase C may reduce
   that single inherited raster in memory, but must never persist it or retain
   multiple cells in the coordinator. Every cell exits after publishing only
   reduced, content-addressed observables. Concurrency is therefore derived
   from the measured peak RSS of a complete identity cell, not from a nominal
   streaming-memory estimate. Removing the inherited raster would require a
   separately reviewed guarded-engine change and is outside Phase C.
8. Set
   `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`.
9. Start with one full SNN worker and measure peak RSS. Set:

   \[
   W_{\max}
   =
   \min\left(
   N_{\rm CPU}-8,\,
   \left\lfloor
   \frac{\mathrm{MemAvailable}-96\ {\rm GB}}
   {1.25\,RSS_{\rm worker}}
   \right\rfloor
   \right).
   \]

   Recompute before each launch wave.
10. Keep at least 96 GB `MemAvailable`, require zero swap growth, and stop
    launches immediately if either guard fails.
11. Durable resource logs record PID, command, phase/cell, wall time, RSS,
    `MemAvailable`, swap, exit status, and artifact SHA.
12. Crashed cells are resumable; no monolithic all-grid process is allowed.
13. Peer-worktree processes are inventoried before launch and are never killed
    by this line.

---

## 11. Deliverables and stopping point

Required machine-readable outputs:

- immutable Phase-C manifest;
- C0 per-run/per-seed metrics and aggregate identity verdict;
- ceiling/gain/ISI/correlation/current/spatial summaries;
- primary and secondary C1 coordinate manifests;
- write-once C1 conditional-gain trigger manifest and all triggered gain parts;
- per-cell phenotype rows and coverage matrices;
- seed-specific modal-routing and operator summaries;
- one fail-closed Phase-C verdict with input SHAs;
- resource and resume audit.

Required figures:

1. core versus whole-sheet ceiling and gain audit;
2. ISI irregularity, refractory locking, and pairwise-correlation null;
3. current balance, membrane distance, fine E/I rate, and PSD;
4. C1 primary-convex phenotype atlas;
5. C1 secondary-shell atlas, clearly labelled extrapolative;
6. representative source/current-vSEEG/spatial traces;
7. seed-specific modal panels;
8. coverage and final Phase-C status.

All figures are diagnostic. Their `figures/README.md` must state that Phase C
does not establish entry, offset, recovery, observation matching, or lifecycle.

The execution stops after C0/C1 adjudication and archive. Any next mechanism
requires a separate reviewed spec:

- refractory saturation → fast-carrier repair;
- valid AI tonic branch → stage-transition/independent-offset design;
- primary-convex maturation window → slow-trajectory reachability design;
- mixed/heterogeneous result → observation/identity refinement.
