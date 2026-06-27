# M3B-next: mean-field / finite-Jacobian spectral phase map for SNN slow-state bifurcation — Design Spec

> Status: DESIGN, 2026-06-27.
> Supersedes M3B's next-step direction, **not** the completed M3B Round-1 bridge record.
> Previous M3B Round-1 plan/result: `docs/superpowers/plans/2026-06-21-sef-hfo-m3-local-w-propagation-operator-plan.md`.
> Brunel reference note: `docs/paper/brunel2026/review.md` and `docs/paper/brunel2026/paper.md`.
> Code starting assets: `src/sef_hfo_lif.py::{mean_field,lif_gains,closed_loop_leading,integrate_lif_field}`.

## 0. Current M3B Status: What We Already Did

M3B Round-1 was **not** a slow-variable bifurcation analysis. It was a model-to-SEEG
field bridge / instrument-probe line:

- Plan commit `19c3398` re-centered M3B on a bridge question: does the model's fixed
  anisotropic scaffold, observed through virtual SEEG, land inside the real interictal/ictal
  field distributions?
- Task 1 built a kick-driven LIF rate-field virtual-SEEG record and recovered the known
  45 degree E->E axis with axis error about 3.3 degrees, passing the 25 degree sanity gate.
- Task 2 showed an **interictal instrument-probe scaffold match**: model-to-real median
  field correlation about 0.844, placement percentile about 74%, beating channel and
  within-shaft geometry nulls.
- The ictal-early comparison was **placement-only**: model-to-ictal median correlation
  about 0.420, placement percentile about 72%, but it did **not** beat geometry nulls.
- Task 3 "same field, two gains" was inconclusive: the axis/shape was stable, but the
  gain sweep did not produce a real graded recruitment range.

Therefore the safe Round-1 claim is:

> A kick-rate-field instrument probe supports an interictal scaffold bridge; the ictal leg is
> placement-only; the gain sweep does not yet establish a phase transition.

That is useful, but it is **not** the M3B mechanism we now need. The new M3B-next should become
the **spectral phase-map** layer: a Brunel-style mean-field stability analysis adapted to a
finite, core-heterogeneous epilepsy sheet. A more precise working name is:

> **M3B-R2: spectral mechanism validation of the spontaneous / slow SNN bridge.**

It is not "one more M3B bridge plot." It asks whether the scaffold that Round-1 could read out
corresponds to intrinsic spatial modes of the core SNN, and whether slow variables move the system
through that modal landscape.

## 1. Core Re-Centering

Brunel's paper gives the right methodological skeleton:

1. coarse-grain the SNN to a spatial E/I mean-field or neural-field model;
2. solve an operating point;
3. linearize around that point;
4. read spatial modes and stability from eigenvalues;
5. validate the spectral prediction against spiking simulation.

The Jacobian in this spec is **not** a microscopic spike/reset Jacobian of every LIF neuron. That
object is ill-defined for threshold/reset dynamics and would be the wrong level for this question.
The modeling stack is:

```text
microscopic SNN
-> coarse-grained E/I rate-synapse field
-> linear spectrum / eigenmode
-> SNN + virtual-SEEG validation
```

The SNN validates whether the coarse-grained modes predict real spontaneous events, early axes,
return-to-baseline, and readout placement. It does not supply the linear operator directly in the
first pass.

The direct Brunel object is a translation-invariant plane-wave spectrum:

```text
delta nu(x,t) = nu1 exp(lambda t + i k dot x)
k -> lambda(k)
```

For our epilepsy problem, this is only the background sanity layer. Once a fixed pathological
core is inserted, the system is no longer translation-invariant. A Fourier `k` perturbation is
not an eigenmode. The correct main object becomes the finite heterogeneous Jacobian:

```text
delta dot z = J(core, q, g, nu_ext, slow_state, ...) delta z
J phi_m = lambda_m phi_m
```

`phi_m(x)` is now the spatial propagation/recruitment mode for the specific core, boundary,
and operating point. That is exactly what M3B needs: a phase map of **which spatial mode becomes
available as slow variables move the system**.

## 2. Boundary With M3A

M3A and M3B remain separate, but the interface changes:

- **M3A** tests whether SNN slow-variable dynamics can create spontaneous phenotype changes
  (`rest/interictal -> preictal-like -> R4a/R4b`) and emits slow-state traces.
- **M3A also owns the slow-to-rate mapping**: the sign/scale by which SNN slow variables become
  coarse rate-field coordinates.
- **M3B-next** builds the frozen-state spectral phase map and overlays M3A trajectories only after
  that mapping exists.
- M3B may say "the M3A slow trajectory moved through this spectral region"; it must not define
  seizure-like transition from eigenvalues alone.
- If M3A is negative or unavailable, M3B can still publish the frozen phase map as a model-side
  mechanism hypothesis, but must not claim an endogenous slow-state transition.

This is the better contract: slow variables are not the whole story; they are trajectories through
a pre-existing spatial dynamical landscape.

### M3A -> M3B Interface

> **2026-06-27 — CANONICAL CONTRACT SUPERSEDES THE FIELD-LEVEL DETAIL BELOW.** Authoritative field
> names, enums, schemas, and the overlay gate live in
> `docs/superpowers/specs/2026-06-27-sef-hfo-m3-interface-contract.md` (executable mirror
> `src/sef_hfo_m3_interface.py`, TDD `tests/test_sef_hfo_m3_interface.py`). A 4-lens review found 8
> fail-open blockers in the original text; where this section and the canonical contract disagree,
> the canonical contract wins. Corrections that matter here:
> - the binding id is the single canonical `slow_to_rate_mapping_id`, present as a COLUMN on
>   `phase_trajectory.csv` / `event_phase_samples.csv` and as `axes_built_from_slow_to_rate_mapping_id`
>   on `finite_jacobian_grid.json` (was missing — clause "same mapping" was unverifiable);
> - the 11 minimum columns are resolved by joining `phase_trajectory ⋈ event_phase_samples` on
>   `event_id` (sentinel `event_id == -1` for inter-event rows); `return_to_baseline` is the canonical
>   name (not `returned`);
> - overlay axes are the normalized `phase_x_core`/`phase_y_global` ∈ [0,1] with extent =
>   `phase_coord_ranges.json` (D1 — no raw-unit axes this round; the §3 raw-unit "examples" describe
>   the underlying slow variables, not axis units);
> - the four overlay conditions are recorded in a SCHEMA'd `m3a_interface_audit.json` (four required
>   booleans + `overlay_verdict ∈ {phase_map_trajectory, mechanism_candidate_only, refused}`); the
>   overlay artifacts/figure are written ONLY when `overlay_verdict == phase_map_trajectory`, never
>   merely because M3A files exist;
> - `cond3` permits ≤ 5 % out-of-range samples (`OUT_OF_RANGE_FRACTION_MAX = 0.05`); `cond4` reads
>   ONLY the A2 dynamic summary by strict equality (`rate_matched_control == "passed"` AND
>   `gate_A_trajectory == "PASS"`), never the A1 quasi-static answer.

M3B does not consume raw `z/phi/gK/e_GABA/q` traces directly. It consumes these M3A artifacts:

```text
slow_to_rate_mapping.json
phase_coord_ranges.json
phase_trajectory.csv
event_phase_samples.csv
dynamic_slowvars_summary.json
```

Minimum columns / fields:

```text
time_ms
event_id
event_stage  # pre, onset, peak, end, post_50ms, post_200ms, post_1s
phase_x_core
phase_y_global
phase_recovery
phase_coord_valid
phase_coord_out_of_range
slow_to_rate_mapping_id
R_class
return_to_baseline
```

M3B overlays a trajectory only when:

1. `slow_to_rate_mapping.json` has passed sign tests;
2. the M3B phase-map axes were built using the same mapping and ranges;
3. trajectory samples are inside the map range, or out-of-range samples are explicitly flagged;
4. M3A independently showed phenotype movement beyond event-rate-only heating.

If any of these fail, the artifact remains an M3A phenotype trace, not an M3B phase-map trajectory.

## 3. Objects and Notation

State vector on a coarse grid:

```text
z_i = [rE_i, rI_i, sEE_i, sEI_i, sIE_i, sII_i, optional frozen slow states]
```

Minimum first pass:

- E/I rates use the LIF-derived transfer `Phi_LIF(mu, sigma)` already implemented in
  `src/sef_hfo_lif.py`.
- `Phi_LIF`, `dPhi/dmu`, drive scaling, and `q_core/q_global` inhibition scaling are part of
  the numerical contract. If these local gains or signs are wrong, every eigenvalue downstream is
  contaminated.
- AMPA/GABA synaptic states are included, because the Hopf frequency depends on synaptic
  time constants.
- Delays are **not** in the first finite-Jacobian pass unless needed; add them later as
  delay characteristic equations or augmented delay-chain states.
- Slow variables are frozen parameters in M3B phase-map runs. Dynamic slow variables come
  from M3A and are only overlaid as trajectories.

Operating point:

```text
z_star = z_star(q_core, q_global, mu_core, g, nu_ext, frozen_slow, ...)
```

For a core-heterogeneous sheet this is not a spatially uniform rest state. It must be explicitly
defined before any eigenvalue is interpreted. The allowed sources are:

1. **Primary**: deterministic rate-field fixed point / integration-to-steady at frozen parameters.
2. **Sanity**: SNN pre-event baseline average mapped onto the coarse E/I state.
3. **Sensitivity**: quasi-steady baseline with slow variables frozen at M3A-supplied state samples.

Every spectrum must record which operating-point source was used. "Around the event" is too vague;
we need to know whether the linearization point is baseline/pre-ignition, post-event recovery, or
a frozen slow-state sample.

Parameter coordinates:

- `x = core excitability`: examples `Delta Vtheta_core`, `mu_core`, `1/q_core`,
  or core-specific external drive.
- `y = global excitability / disinhibition`: examples `1/q_global`, `nu_ext / nu_theta`,
  global GABA weakening.
- panels / facets: `g_eff`, `G`, `C_EE/C_EI`, `w_EE_mult`, recovery strength.

Core geometry variants:

- **Generic single core** is the cleanest first finite-Jacobian object and is the **primary atlas**
  geometry.
- **Two low-threshold cores on the E->E axis** connect to the current Fig4/5 SNN asset and the
  forward/reverse virtual-SEEG readout. Use it as validation / subject-facing geometry, but keep
  the first spectral atlas readable: do not claim balanced independent long-run dual-source
  dynamics unless the SNN actually shows it.

The leading spectral object:

```text
alpha_m = Re(lambda_m)
f_m = Im(lambda_m) / (2*pi)
```

Instability line:

```text
alpha_1 = 0
```

For discrete-time or fitted event maps, the equivalent check is `rho(M) = 1`, but the primary
continuous rate-field phase map should use `max Re(lambda_m)`.

## 4. Why `k`-Spectrum Is Still Useful but Not Sufficient

| Layer | Object | Purpose | Claim boundary |
|---|---|---|---|
| Homogeneous background | `lambda(k)` | Reproduce Brunel-style finite-k / Turing-Hopf logic; validate rate-field machinery | Background substrate only |
| Core-heterogeneous field | finite `J phi = lambda phi` | Main epilepsy phase map | Core-specific propagation / recruitment modes |
| SNN validation | spontaneous or finite-pulse SNN events | Tests whether spectral regions predict actual event phenotype | Final behavioral evidence |
| SEEG bridge | masked lag/rank readout | Compares observed model fields to patient fields | Data-facing readout, not the phase map itself |

M3B-next must start with the homogeneous `k` sanity check, but its main figure should be finite
Jacobian eigenmodes under fixed core heterogeneity.

Gate discipline for homogeneous `k` sanity:

- numerical root-search / convergence unresolved = **stop and fix**;
- numerically stable but no clean finite-k Brunel-like tendency = record as **background negative /
  calibration caveat**, but finite-core maps may continue, because fixed core heterogeneity can
  create localized or axial finite-system modes even when the homogeneous background lacks a
  pretty finite-k Hopf.

## 5. Mode Metrics

Each eigenmode should be summarized before any phase-map verdict. At minimum:

- `growth`: `alpha_m = Re(lambda_m)`.
- `frequency_hz`: `abs(Im(lambda_m)) / (2*pi)`, converted to Hz.
- `spectral_gap`: `alpha_1 - alpha_2`; small gap means mode competition / noisier directions.
- `core_overlap`: fraction of E-mode power inside the core mask, e.g.
  `sum_core |phi_E|^2 / sum_all |phi_E|^2`.
- `axis_score`: alignment with the E->E long axis, reported in two variants:
  `elongation_axis_score` from spatial second moments and `phase_gradient_axis_score` from
  Fourier/phase-gradient energy. Propagation ridges and wavevectors can differ by 90 degrees,
  so the two scores must not be silently collapsed before synthetic-mode calibration.
- `globality`: participation ratio and low-k energy; high values indicate broad/global recruitment.
  Participation ratio should be normalized by grid size, e.g.
  `(sum |phi_E|^2)^2 / (N * sum |phi_E|^4)`.
- `off_axis_score`: mode power or propagation component away from the E->E long axis.
- `core_controllability`: projection of a core-local perturbation onto the **left** eigenmode.
- `finite_time_gain`: `||exp(J*T) b_core|| / ||b_core||`, computed with `expm_multiply`
  for short windows relevant to HFO bursts.

Left/right eigenvectors are required because the operator can be non-normal:

```text
J phi_m = lambda_m phi_m
psi_m^T J = lambda_m psi_m^T
core_controllability_m = |psi_m^T b_core|
```

The right eigenvector says what the mode looks like; the left eigenvector says whether core noise
or a core perturbation can actually excite it. The last metric is not cosmetic either. Interictal
self-limited propagation may be:

```text
max_m Re(lambda_m) < 0
but finite_time_gain(T, b_core) is large
```

So non-normal transient growth is a **primary** M3B-R2 result, not a supplement. The `alpha_1=0`
contour is important, but it is not the only border between "silent" and "event-capable."

Metric implementation must be locked with synthetic modes before real eigensolver outputs are
interpreted. Synthetic tests should cover core-localized modes, elongated axial ridges,
phase-gradient waves, global low-k modes, off-axis modes, and a non-normal toy where
`Re(lambda)<0` but finite-time gain is high.

## 6. Expected Spectral Story

### Interictal-like region

- leading mode is core-localized plus axial propagation;
- `core_overlap` high;
- `axis_score` high;
- `globality` low to moderate;
- `spectral_gap` large enough that one mode dominates;
- `alpha_1 <= 0` or only weakly positive, with recovery able to return the system;
- `finite_time_gain` high enough for a transient event but not sustained recruitment.

### Preictal-like region

- `alpha_1` rises;
- `alpha_2` / `alpha_3` approach the leading mode;
- `spectral_gap` shrinks;
- low-k/global modes rise;
- axial mode can remain visible, but is no longer uniquely dominant;
- event direction becomes more sensitive to noise, ignition location, and readout geometry.

### Seizure-like / sustained recruitment region

- global or low-k recruitment mode approaches or exceeds the axial mode;
- `globality` increases;
- multiple modes may be near instability;
- weak recovery or high disinhibition produces R4a or R4b.

### Recovery / self-limiting region

- `phi`, `gK`, resource depletion, or inhibition recovery pushes `alpha_1(t)` back below zero;
- `finite_time_gain` falls after an event;
- system returns to stable/interictal axial region.

## 7. Phase-Map Design

Primary phase map:

```text
x-axis: core excitability
y-axis: global excitability / disinhibition
panels: g_eff or recurrent E/I ratio
```

Each grid point stores:

- operating point convergence and rates;
- leading 5-10 eigenpairs;
- left/right eigenvector normalization diagnostics;
- leading mode class: `stable`, `local`, `axial`, `mixed`, `global`, `runaway`, `unresolved`;
- `alpha_1 = 0` contour;
- axial-vs-global mode crossing contour;
- spectral gap contour;
- finite-time gain contour;
- optional M3A slow-state trajectory overlay.

Mode classes should be metric-based, not hand-labeled from pretty plots:

- `local`: high `core_overlap`, low `globality`, low axis extension.
- `axial`: high `axis_score`, moderate core controllability, propagating / elongated along E->E.
- `mixed`: axial and global scores both elevated, or gap small.
- `global`: high `globality` / low-k energy, broad participation.
- `runaway`: unstable high-rate branch or spectral prediction plus SNN tonic saturation.
- `unresolved`: no clean operating point or eigensolver failure; never silently called stable.

Required controls:

| Control | Purpose |
|---|---|
| no-core homogeneous background | Shows the core is not mathematically redundant. |
| isotropic E->E (`AR=1`) | Tests whether axial modes are scaffold-driven rather than boundary/readout artifacts. |
| off-axis core | Tests scaffold-core geometry, not just any local perturbation. |
| shuffled core thresholds | Excludes one random low-threshold pixel dominating the atlas. |
| frozen slow vs active/frozen `q, phi, gK` samples | Separates interictal propagation from slow-state recruitment. |
| no recovery (`phi/gK` off or weak) | Maps self-termination vs runaway boundary. |

## 8. Validation Contract

M3B-next is theory-first but cannot end at theory. Validation is staged:

1. **Homogeneous sanity**: existing `closed_loop_leading`-style dispersion should recover a finite-k
   Hopf / candidate mode in the expected parameter neighborhood, or explain why the current LIF
   gain estimate is too low.
2. **Finite-Jacobian unit tests**: known symmetric cases produce paired modes; removing the core
   approaches the homogeneous spectrum; increasing core drive raises core-overlap/growth.
3. **Rate-field dynamic check**: finite-pulse or noise-driven rate-field simulations from representative
   phase-map regions match predicted local/axial/global behavior.
4. **SNN spot checks**: selected grid points are run in the SNN with frozen slow state. Spectral
   class should predict return-to-baseline, R2/R3/R4a/R4b split, and early recruitment axis.
5. **Mode-to-readout projection**: candidate modes and SNN events must be pushed through the same
   virtual-SEEG observation layer used by M3B Round-1, producing model records for
   `compare_model_to_cohort` and geometry-null testing.
   A mock-mode readout schema smoke test should happen early, before scientific mode projection,
   so that late-stage bridge failures are scientific rather than adapter/schema surprises.
6. **M3A overlay**: if M3A provides `s_slow(t)`, its trajectory should move through phase-map regions
   in the same order as the observed phenotype changes.

No single eigenvalue can be called "the seizure." The phase map is a mechanism map; SNN behavior is
the phenotype test.

## 9. Data-Facing Role

Round-1 bridge results stay useful but become downstream:

- The spectral phase map predicts whether a state should generate a local/axial/global event.
- Candidate eigenmodes and generated SNN events are projected through the same virtual-SEEG readout:

```text
phi_m(x) or event field
-> virtual SEEG rank/envelope field
-> interictal / ictal cohort placement
-> geometry null
```

- Virtual-SEEG readout then asks whether those generated modes/events land in the real interictal
  or ictal field distributions.
- The new M3B bridge claim should require both: spectral mode class **and** observed event/readout
  behavior.

This avoids the old trap where a model field matched SEEG geometry but did not establish the
underlying bifurcation mechanism.

Claim ladder:

- Spectrum only: theoretical appendix / model mechanism hypothesis.
- Spectrum + SNN event match: model-side spontaneous mechanism.
- Spectrum + SNN event match + virtual-SEEG placement/null pass: true M3B model-to-patient bridge.

## 10. Allowed Claim Template

Allowed if supported:

> The epileptic core does not simply become "more excitable." Core excitability, global
> disinhibition, E/I gain, and anisotropic connectivity reshape the finite-system spatial
> propagation spectrum. In the interictal-like region, core perturbations preferentially
> excite a localized axial mode on the E->E scaffold, producing self-limited propagation.
> As slow variables move the system, the spectral gap shrinks and low-k/global recruitment
> modes approach the axial mode. When global or multi-mode growth exceeds recovery, the
> system transitions to sustained recruitment or runaway.

Not allowed:

- "W causes seizure."
- "A plane-wave `k` mode explains the fixed-core event" without finite-Jacobian evidence.
- "Eigenvalue > 0 proves clinical seizure onset."
- "M3B shows slow variables cause seizure" unless M3A produced a valid slow trajectory and SNN
  validation supports it.

## 11. Failure Modes to Report Honestly

- Homogeneous dispersion fails to reproduce the expected finite-k tendency: rate-field gain or
  susceptibility may be under-calibrated for Brunel-style background claims, but this is a caveat
  rather than an automatic stop for finite-core maps if the solver itself is numerically resolved.
- Finite Jacobian modes are dominated by trivial local threshold saturation and do not predict
  SNN propagation: the coarse mean-field is the wrong abstraction for M3.
- Axial mode never gives way to global/mixed modes across plausible slow-state range: no spectral
  support for the proposed interictal-to-seizure transition.
- Phase map predicts transition but SNN stays fixed-size/self-limited: recovery/noise/nonlinearity
  dominates the linear map.
- SNN transitions only to R4b tonic saturation, not R4a structured recruitment: this is toxicity,
  not a seizure-like bridge.

## 12. Canonical Outputs

Root:

```text
results/topic4_sef_hfo/m3b_spectral_phase_map/
```

Required artifacts:

- `STATUS.md`: short verdict with claim boundary.
- `homogeneous_dispersion.json`.
- `finite_jacobian_grid.json`.
- `mode_metrics.csv`.
- `control_summary.json`.
- `mode_readout_projection.json`.
- `m3a_interface_audit.json` when M3A handoff is attempted.
- `slow_trajectory_overlay.csv` when M3A is available.
- `snn_spotcheck_summary.json`.
- `figures/README.md`.

Required figures:

1. Brunel-style homogeneous `lambda(k)` sanity panel.
2. Example finite-core eigenmodes: local / axial / mixed / global.
3. Core-excitability x global-disinhibition phase map with `alpha_1=0`, gap, and mode-class overlays.
4. Non-normal finite-time gain / core controllability map.
5. Mode-to-virtual-SEEG projection and cohort/null placement.
6. M3A slow-state trajectory over the phase map, if available.
7. SNN spot-check grid: predicted spectral class vs observed R-class.
