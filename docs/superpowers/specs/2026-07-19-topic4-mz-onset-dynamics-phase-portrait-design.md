# Topic 4 MZ early-onset dynamics: projected phase diagram, nonlinear ignition, and causal bridge

Date: 2026-07-19

Status: **DESIGN LOCK — execute after the two active upstream tasks have committed or published stable artifacts**

Working family: `codex/topic4-mz-slowvars`

Scientific tier: model-side mechanism analysis. This is not seizure validation and does not replace the empirical Figure 3 result.

## 0. One-sentence objective

Test whether repeated interictal-like events move the per-neuron MZ inhibitory-efficacy state toward a regime in which the fixed E1146 scaffold has stronger axial finite-time susceptibility and a lower nonlinear ignition threshold, thereby producing an operational runaway whose early field retains the interictal spatial ordering; then test whether `m_i` opposes that transition.

The required causal chain is:

```text
repeated returning interictal-like events
    -> z_i depletion / effective disinhibition increases
    -> fixed fast network becomes easier to excite along the pre-existing scaffold axis
    -> nonlinear ignition threshold falls
    -> a subsequent endogenous fluctuation/event escapes the interictal basin
    -> early operational-runaway energy retains the interictal field ordering
    -> m_i raises the threshold or restores the state, if the peer push-pull hypothesis is correct
```

The task is not complete if it produces only a prettier trajectory, a parameter phenotype grid, or a linear eigenvalue map.

## 1. Relationship to active upstream work

Two upstream analyses are already active in the same MZ worktree and must be treated as separate evidence layers:

1. `topic4-mz-early-field-bridge`: held-out interictal timing template versus early operational-runaway field.
2. `topic4-state-conditioned-spatial-susceptibility`: frozen MZ slow-state mapping into the finite heterogeneous M3B operator.

This specification consumes their committed artifacts when available. It does not overwrite their code, re-stage their dirty files, or silently copy their scientific verdicts.

Their roles are:

| Layer | Question | What it cannot establish alone |
|---|---|---|
| Early-field bridge | Does interictal timing predict the early field? | Why the state became easier to ignite; causality of `z_i` |
| Susceptibility atlas | How does the frozen coarse linear operator change? | Nonlinear basin escape; event-driven depletion; `m_i` push-pull |
| This task | Does the MZ trajectory approach and cross a causal ignition boundary? | A complete seizure cycle or clinical mechanism |

If either upstream result is negative, complete this registered analysis without tuning the model to reverse that result. A positive spectral result cannot rescue a negative eligible field bridge; a positive field association cannot substitute for a causal state test.

## 2. Locked microscopic model

Only E neurons carry the peer-proposed slow variables:

$$
\tau_m^E \dot V_i^E
=-V_i^E+I_i^{E,E}-z_i I_i^{E,I}-\eta_m m_i.
$$

I neurons remain unchanged:

$$
\tau_m^I \dot V_i^I
=-V_i^I+I_i^{I,E}-I_i^{I,I}.
$$

Inhibitory efficacy:

$$
\tau_z\dot z_i
=H(I_{\mathrm{th}}^{E,I}-I_i^{E,I})-z_i,
\qquad z_i\in[0,1].
$$

Spike-frequency adaptation:

$$
\dot m_i=-\frac{m_i}{\tau_{\mathrm{adp}}}
+\sum_k\delta(t-t_i^k),
\qquad I_i^{\mathrm{adp}}=\eta_m m_i.
$$

Interpretation is locked:

- `z_i down` means reduced effective inhibition and is the candidate early-onset push;
- `m_i up` means a stronger E-cell adaptation current and is the candidate pull/recovery term;
- `z_i` is phenomenological inhibitory efficacy, not a direct measurement of chloride, GABA release, or interneuron exhaustion;
- `m_i` is not yet an established termination mechanism;
- all old `q_I`, `g_K`, `S_G`, shunting, and STD mechanisms remain off.

## 3. Current evidence boundary

The accepted MZ artifacts currently establish:

- slow-off returning interictal-like events in seeds 1/3/4;
- robust z-only operational runaway for `zA_q75_tz5000` and `zA_q50_tz10000` in all three seeds;
- m-only suppression in the sampled discovery cells;
- no robust expanded-returned state;
- the original Arm C nominal `weak/mid/strong` grid is invalid as a graded z-by-m interaction because its selected configurations collapsed.

Therefore the current safe hypothesis is:

> z-mediated loss of inhibitory efficacy is a candidate driver of early operational runaway; m-mediated adaptation is a competing brake whose ability to bound or reverse the transition remains untested in the correct realized-state regime.

Do not write that `z+m` already explains seizure onset or termination.

## 4. Primary candidates, seeds, and registered states

Primary trajectory:

```text
candidate = zA_q75_tz5000
seeds = 1,3,4
reason = robust 3/3 delayed runaway and the longest pre-transition interval for event-resolved analysis
```

Registered onset anchors from committed upstream artifacts must be read, not re-estimated for candidate selection. The known values are approximately 9294/9499/9758 ms, but the exact artifact values govern.

Sensitivity trajectory:

```text
candidate = zA_q50_tz10000
seeds = 1,3,4
reason = independent faster robust-runaway family
```

For each seed capture:

```text
baseline_1000ms
mid_fraction       = 0.50 * locked operational-runaway onset
pre_onset_500ms
pre_onset_100ms
onset
```

Also capture `event_pre` and `event_post` states for every eligible returning event in the primary pre-runaway trajectory:

```text
event_pre  = event onset - 20 ms
event_post = event offset + 100 ms
```

No state may be moved to a visually cleaner event after inspecting the target result.

## 5. Analysis coordinates

### 5.1 Slow-state coordinates

For a registered E-neuron set $\mathcal R$:

$$
D_z^{\mathcal R}(t)=1-\frac{1}{|\mathcal R|}\sum_{i\in\mathcal R}z_i(t),
$$

$$
A^{\mathcal R}(t)=\eta_m\frac{1}{|\mathcal R|}\sum_{i\in\mathcal R}m_i(t).
$$

Required regions:

- all E neurons;
- source core;
- sink core;
- registered scaffold-axis corridor;
- off-axis field;
- core-excluded field.

The primary projected phase coordinates are:

```text
x = D_axis       # disinhibitory push in the registered scaffold corridor
y = A_axis       # adaptation current in the same corridor
```

All-E and core/surround coordinates are required sensitivities. Do not choose the region that makes the trajectory look most separated.

### 5.2 Current-aware effective inhibition

Because `z_i` is driven by and multiplies `I_i^{E,I}`, arithmetic `mean(z_i)` is not automatically the effective inhibition seen by the membrane. For every registered state and spatial bin compute over a fixed 20-ms pre-state window:

$$
q_{\mathrm{eff}}(x,t)=
\frac{\sum_{i\in x}\sum_{u\in W_t} z_i(u)I_i^{E,I}(u)}
{\sum_{i\in x}\sum_{u\in W_t} I_i^{E,I}(u)+\epsilon}.
$$

Also save:

$$
p_{\mathrm{deplete}}(x,t)
=P(I_i^{E,I}\ge I_{\mathrm{th}}^{E,I}\mid i\in x,u\in W_t).
$$

The observer must accumulate only the registered windows; it must not allocate an `N x T` current matrix.

Required mapping audit:

- `z_bar` versus `q_eff` spatial Spearman and cosine;
- mean and maximum absolute difference;
- core, axis, and off-axis differences;
- whether the inferred preferred spatial orientation changes.

Use `q_eff` as the primary rate-field inhibition scale. Retain `z_bar` as the slow-state coordinate. If they disagree, report the disagreement rather than selecting whichever gives a stronger bridge.

### 5.3 Exact projected slow equations and closure boundary

Let $D=1-\langle z_i\rangle$ and $A=\eta_m\langle m_i\rangle$. Population averaging gives:

$$
\tau_z\dot D=P(I_i^{E,I}\ge I_{\mathrm{th}}^{E,I})-D,
$$

$$
\dot A=-\frac{A}{\tau_{\mathrm{adp}}}+\eta_m r_E.
$$

Candidate nullclines are therefore:

$$
D=P(I_i^{E,I}\ge I_{\mathrm{th}}^{E,I}\mid D,A),
$$

$$
A=\eta_m\tau_{\mathrm{adp}}r_E(D,A).
$$

These equations are exact averages but are not automatically a closed two-dimensional system: their right-hand sides can depend on spatial pattern, synaptic state, and history.

Rules for calling a plot a phase portrait:

1. estimate finite-difference drift in predeclared `(D,A)` bins;
2. require at least three independent visits from at least two seeds for an arrow;
3. require the drift sign to agree in at least two of three seeds for each displayed component;
4. draw an interpolated nullcline only through adjacent eligible bins with an observed sign change;
5. otherwise label the object `projected state trajectory`, not `phase portrait`.

## 6. Frozen fast-system phase diagram

At a frozen slow field $s=(q_{\mathrm{eff}}(x),A(x))$, solve the existing coarse E/I rate-synapse operating point and form:

$$
J(s)=\left.\frac{\partial F}{\partial X}\right|_{X^\ast(s)},
\qquad
\alpha_1(s)=\max_j\mathrm{Re}\,\lambda_j(J).
$$

The linear instability boundary is `alpha_1 = 0`. It must not be called the seizure boundary.

Finite-time E-rate susceptibility is:

$$
K_T(s)=P_{r_E}e^{J(s)T}E_{r_E},
$$

$$
G_{\mathrm{axis}}(s,T)=
\frac{\|K_T(s)b_{\mathrm{axis}}\|_2}{\|b_{\mathrm{axis}}\|_2}.
$$

Required windows: `T = 10, 30, 50 ms`; `75 ms` sensitivity.

Required outputs at every natural registered state:

- operating-point status and residual;
- `alpha_1`, imaginary frequency, next-distinct spectral gap;
- leading invariant-subspace globality and core overlap;
- axial, perpendicular, and uniform/global finite-time gain;
- `G_axis/G_perp` and gain persistence;
- input/output singular fields of the projected propagator;
- source/sink and core-excluded response summaries.

The accepted M3B distinction is binding: a global leading eigenmode and an axial non-normal response are different objects and must remain separately labelled.

### 6.1 Controlled `(D,A)` grid

The final MC diagram must be drawn in realized state coordinates, not in nominal `I_th`, `tau_z`, or `eta_m` parameter labels.

Use a small registered grid:

```text
D values: 9 equally spaced values from pooled primary baseline D_axis
          to 1.10 * pooled maximum onset D_axis, clipped to [0,0.95]
A/I_EE_scale values: [0.00, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
```

Field construction:

- primary z-pattern: pooled primary-onset depletion field normalized to mean depletion 1, then scaled to the requested `D`;
- primary adaptation field: spatially uniform `A`, so the push-pull axes remain interpretable;
- controls: uniform z, rotated-90 z, shuffled z, and z-blocked;
- patterned m is deferred until a valid natural z+m trajectory exists.

Run the coarse grid at `n=8` first for numerical audit, then `n=12` for final output. Grid failures are marked `unresolved`; parameters are not adjusted to fill a desired region.

## 7. Full-SNN nonlinear ignition boundary

Linear stability cannot determine whether the SNN escapes through a finite-amplitude/non-normal mechanism. Define the frozen-state critical perturbation:

$$
\varepsilon_c(s,x_0)=
\inf\{\varepsilon:\text{a registered deterministic local probe at }x_0
\text{ produces operational runaway within 500 ms}\}.
$$

### 7.1 State-matched replay without editing the guarded engine loop

Primary implementation uses complete replay from `t=0` with the same substrate and RNG seed. All branches are identical until a predeclared branch step. At that step the unguarded MZ slow object applies a registered transformation or probe.

Required proof:

- spike raster, population rate, `z`, `m`, and current summaries are bit-identical before the branch step;
- no branch configuration consumes additional RNG before the branch;
- the native no-action path remains bit-identical to the accepted MZ run;
- every intervention is off by default.

Do not modify or re-bless the six guarded engine files for the primary implementation. Exact checkpoint/resume is an optional performance optimization only after the replay implementation is scientifically complete.

### 7.2 Frozen-state protocol

At each of `baseline_1000ms`, `mid_fraction`, `pre_onset_500ms`, and `pre_onset_100ms`:

1. replay the native trajectory from `t=0`;
2. at the branch step freeze `z` and `m` at their current values;
3. continue for 500 ms under the same external-noise process;
4. apply a deterministic 10-ms local threshold-lowering probe through the MZ `threshold()` hook;
5. score operational runaway with the already locked `120 Hz / 100 ms` rule.

Probe locations:

- source core;
- sink core;
- one predeclared off-axis control at matched distance from sheet center.

Amplitude is normalized by the local threshold gap:

$$
a=\frac{\Delta V_\theta}{\mathrm{median}_{i\in\mathrm{target}}(V_{\theta,i}-V_{\mathrm{reset}})}.
$$

Registered amplitude ladder:

```text
a = [0, 0.025, 0.05, 0.10, 0.20]
```

If the ladder brackets a transition, run two fixed bisection refinements. If the zero-probe frozen control runs away, record `epsilon_c=0`. If the largest probe does not run away, record right-censoring; do not enlarge the range after seeing the result.

Required response metrics include onset latency, peak/mean E rate, participating fraction, spatial spread, return within the 500-ms horizon, and early source/contact fields where available.

The key classification is result-neutral:

- `linear_crossing`: `alpha_1 >= 0` near the state where `epsilon_c` reaches zero;
- `finite_amplitude_escape`: `alpha_1 < 0` while `epsilon_c` decreases materially;
- `uniform_global_amplification`;
- `axis-selective susceptibility`;
- `state-invariant threshold`;
- `seed-inconsistent`;
- `unresolved`.

No category is a success gate.

## 8. Causal z-field counterfactuals

At `pre_onset_500ms` and `pre_onset_100ms`, replay from `t=0` and branch into:

1. native dynamic z continuation;
2. native z frozen at the branch state;
3. uniform current-matched z, using the branch-state `q_eff`;
4. spatially shuffled z with the same histogram and a fixed independent shuffle seed;
5. z reset to one;
6. rotated-90 coarse z field mapped back to E neurons, if the mapping audit is valid.

Run each branch with zero probe and with the registered source-core probe nearest the native `epsilon_c` bracket.

Questions:

- Does resetting z prevent or delay operational runaway?
- Does uniform current-matched z preserve gain but remove the spatial preference?
- Does shuffle/rotation preserve the global transition but disrupt the scaffold-aligned early field?
- Is the natural spatial pattern informative beyond its global mean?

Report native versus counterfactual onset, `epsilon_c`, spatial gain, and early-field association separately. Association is not causality; the state-matched branch is the causal object.

## 9. Do interictal events drive the state or merely reveal it?

For the primary delayed-runaway trajectory, use the fixed slow-off-derived event detector. Select the last three eligible returning events that end at least 200 ms before the locked operational-runaway onset. Selection is chronological and independent of the target result.

Calibrate one deterministic inhibitory threshold pulse on slow-off events only:

- target = initiating core plus registered axis corridor;
- duration = 20 ms;
- choose the smallest predeclared amplitude from `[0.025,0.05,0.10,0.20]` threshold-gap units that reduces the event peak active fraction by at least 50% without suppressing activity for more than 200 ms;
- lock this amplitude before target-trajectory runs.

For each selected target event compare:

- event-suppression pulse beginning 5 ms before event onset;
- time-matched sham pulse in the nearest eligible quiet interval;
- no-pulse native replay.

Measure:

- removed event area/rate/participation;
- change in `D_axis`, `q_eff_axis`, and `p_deplete` at +100 and +500 ms;
- change in the next event and in operational-runaway onset;
- change in `epsilon_c` at the next registered probe state.

If a pulse cannot reduce an event without broadly silencing the network, report `event_suppression_unresolved`. Do not tune the pulse on the target onset delay.

This experiment is required before writing that interictal events causally deplete inhibition enough to trigger the transition.

## 10. Focused m test and repair of the two-dimensional MC plane

Do not reuse the invalid original Arm C labels as a dose response.

After the z-only phase/ignition analysis is complete, run a focused, explicitly post-discovery MC test:

```text
z regimes:
  zA_q75_tz10000   # weak/slow, seed-fragile bounded candidate
  zA_q75_tz5000    # robust delayed runaway

tau_adp:
  2000 ms          # one fixed peer-compatible time scale; avoid another broad grid

target adaptation-current fractions of I_EE_scale:
  [0.00, 0.025, 0.05, 0.10, 0.20]

seeds:
  1,3,4
```

Derive each `eta_m` from the existing slow-off calibration recipe. Verify that the realized adaptation-current levels are distinct; otherwise record calibration failure rather than relabelling duplicate cells weak/mid/strong.

For every cell report:

- realized `(D_axis,A_axis)` trajectory;
- phenotype using the existing MZ detector;
- operational-runaway onset or censoring;
- return/retrigger behavior where the trace is complete;
- `epsilon_c` at one matched pre-onset state;
- whether m raises the ignition threshold, merely suppresses all activity, or creates a bounded/retriggerable regime.

This focused test may support `m is a competing brake`. It supports `m bounds the recruited state` only if the effect is reproducible across seeds and is not equivalent to global suppression.

## 11. Integration with the field bridge

When the upstream early-field bridge is numerically eligible, align its per-seed target windows with this task's states. The final integrated analysis asks:

```text
Does D/q_eff increase?
Does G_axis/G_perp increase or remain scaffold-aligned?
Does epsilon_c decrease?
Does the early contact/source field agree with a held-out interictal template?
Do z reset/uniform/shuffle controls alter the predicted components?
```

Do not reduce these layers to one PASS/FAIL label. Report the five answers independently.

## 12. Figure contract

The existing Figure 5-style continuous trace plus two fields remains the observation-layer exemplar. This task produces a separate model-dynamics figure; do not overload the existing visualization.

Primary figure, one argument per panel:

### Panel A — registered MZ trajectory

Continuous E-rate/envelope with interictal events, `D_axis`, `A_axis`, and operational-runaway onset aligned in time.

Argument: repeated events occur while the slow state moves toward the transition.

### Panel B — projected `(D,A)` phase diagram

Background `alpha_1`, contours of `G_axis/G_perp`, eligible projected drift arrows/nullclines, and natural slow-off/z-only/m-only/z+m trajectories.

Argument: the natural trajectory approaches a specific fast-system stability/susceptibility region.

If closure fails, show trajectories without arrows/nullclines and title the panel `Projected MZ state trajectories`.

### Panel C — linear versus nonlinear boundary

Across registered states show `alpha_1`, axial/perpendicular/global gain, and source/sink/off-axis `epsilon_c`.

Argument: distinguish linear instability from finite-amplitude/non-normal escape.

### Panel D — causal z counterfactuals

Native, uniform, shuffle/rotate, and reset z: onset/threshold and early spatial response.

Argument: separate global loss of inhibition from spatially patterned susceptibility.

### Panel E — interictal-event intervention and field bridge

Event suppression effect on `D`, onset, and `epsilon_c`, plus the registered held-out template-to-early-field effect size across seeds.

Argument: test the full interictal-event-to-early-recruitment chain.

Visual requirements:

- read `docs/figure_style_guide.md` before plotting;
- no decorative mechanism cartoon;
- formulas belong in Methods/caption, not as a large empty panel;
- use the exact E1146 geometry, fixed axis orientation, and montage provenance;
- same limits and color normalization across matched state maps;
- show all three seeds; do not show only the best seed;
- no PASS/FAIL stamps or p-value stars;
- unresolved or censored values remain visible;
- produce PNG and PDF;
- visually inspect the final figures and write a Chinese `figures/README.md` afterward.

Full atlases, all state fields, and numerical controls go to the supplement/diagnostic figure, not the primary figure.

## 13. Code and artifact contract

New code should be isolated from the two active upstream tasks:

```text
config/topic4_mz_onset_dynamics.yaml
src/topic4_mz_onset_dynamics.py
scripts/run_topic4_mz_onset_dynamics.py
scripts/plot_topic4_mz_onset_dynamics.py
tests/test_topic4_mz_onset_dynamics.py
```

Changes to `src/snn_engine/mz_slow_vars.py` are allowed only after upstream snapshot-observer work is committed and merged into the execution branch. Add scheduled interventions off by default. Do not edit the six guarded engine files in the primary implementation.

Artifact root:

```text
results/topic4_sef_hfo/mz_onset_dynamics/
├── provenance.json
├── state_coordinate_audit.json
├── event_state_transitions.csv
├── projected_phase_grid.csv
├── spectral_state_summary.csv
├── nonlinear_ignition_summary.csv
├── z_counterfactual_summary.csv
├── event_suppression_summary.csv
├── focused_m_summary.csv
├── per_seed/
├── arrays/
├── STATUS.md
└── figures/
    ├── README.md
    ├── mz_onset_dynamics_main.png
    ├── mz_onset_dynamics_main.pdf
    └── mz_onset_dynamics_diagnostics.png
```

Archive report:

```text
docs/archive/topic4/sef_hfo/mz_onset_dynamics_2026-07-19.md
```

Every JSON must include schema version, exact upstream paths/hashes, branch and git SHA, config hash, engine hashes, candidate/seed/state lists, command line, completed/not-run/failed stages, and exact result censoring.

## 14. Tests and numerical validity

Before scientific execution run the relevant existing tests and record the exact count. Minimum new tests:

1. off-by-default scheduled-intervention parity;
2. native replay parity with the accepted MZ artifact;
3. bit-identical pre-branch spike/rate/slow traces;
4. freeze actually holds z/m constant after the branch;
5. uniform/reset/shuffle preserve their declared invariants;
6. independent shuffle RNG does not alter the network RNG before the branch;
7. deterministic threshold probe acts only on the registered E target and registered time window;
8. current-aware `q_eff` synthetic fixture;
9. coordinate/axis/core mapping fixture;
10. finite-Jacobian residual and projected-propagator checks;
11. right-censored/zero-threshold ignition classification fixtures;
12. phase-arrow/nullcline eligibility fixtures;
13. resume/idempotency and partial-artifact provenance tests.

Run one cheap low-density or short-duration smoke before full-density work. Measure wall time and peak RSS, choose safe concurrency, and use resumable per-seed/per-state artifacts. Never launch multiple full-density runs blindly.

## 15. Result-neutral completion levels

Report separately:

1. **engineering complete**: observers/interventions/tests/provenance are valid;
2. **state map complete**: registered MZ coordinates and current-aware inhibition are available;
3. **linear mechanism described**: frozen operator and finite-time gain are resolved or explicitly unresolved;
4. **nonlinear boundary described**: `epsilon_c` is measured or censored at registered states;
5. **causal z test complete**: native/uniform/shuffle/reset branches are compared;
6. **event-causality test complete**: event suppression or a clear unresolved reason is reported;
7. **m push-pull test complete**: focused realized-state grid is run without duplicate labels;
8. **integrated bridge described**: field association is aligned with state/gain/threshold results.

Do not collapse these levels into one overall PASS.

## 16. Allowed and forbidden claims

Allowed if directly supported:

- repeated interictal-like events were followed by measurable inhibitory-efficacy depletion;
- the frozen fast network became more or less susceptible along the registered scaffold axis;
- nonlinear ignition threshold decreased, remained constant, or changed inconsistently before operational runaway;
- the transition was consistent with linear instability or finite-amplitude escape;
- global versus spatial z counterfactuals had distinguishable or indistinguishable effects;
- suppressing registered interictal-like events delayed, did not change, or inconsistently changed depletion/onset;
- m raised the ignition threshold, globally suppressed activity, produced a bounded state, or had no reproducible effect;
- the model provides a bounded feasibility mechanism for `same scaffold, different state`.

Forbidden:

- the model reproduced a clinical seizure;
- operational runaway is a complete ictal event;
- `z_i` proves chloride accumulation, GABA failure, or interneuron exhaustion;
- a global leading eigenmode is axial because a finite-time response is axial;
- an eigenvalue crossing alone proves seizure initiation;
- a field correlation proves causal depletion;
- the invalid original Arm C nominal grid is a graded interaction;
- a single seed or direct-core-only effect establishes the model bridge;
- parameter tuning after reading the bridge/threshold result is confirmatory evidence;
- termination or recovery is explained without a reproducible bounded/returned state.

## 17. Stop and fallback rules

- If upstream active files are dirty, do not overwrite or stage them. Start with read-only audit and create a dedicated execution worktree only from a committed base.
- If the native replay does not match the accepted onset within the upstream tolerance, stop causal branches and diagnose replay parity.
- If current-aware mapping cannot be captured without changing dynamics, deliver the coordinate observer audit before running the phase map.
- If the rate-field operating point is unresolved for all registered states, skip the full coarse grid and complete the full-SNN ignition/counterfactual analysis.
- If a frozen no-probe SNN state is already runaway, record `epsilon_c=0`; do not move the state earlier.
- If the maximum registered probe does not ignite, record right-censoring; do not raise the amplitude post hoc.
- If phase-space closure support is insufficient, show projected trajectories and do not draw nullclines.
- If the eligible field bridge is negative, report that the current MZ mechanism does not reproduce the empirical bridge; do not add mechanisms to rescue it in this task.
- If event suppression is unresolved, retain the z-state causal test but do not claim events drive depletion.
- No push or merge. Commit only task-owned files in logical batches and report the exact branch/status.

## 18. Definition of done

The task is complete only when:

1. upstream provenance and worktree isolation are documented;
2. state-coordinate and `q_eff` audits exist for both trajectory families or exact missing reasons are written;
3. natural-state spectral/non-normal outputs exist across all three seeds;
4. the controlled `(D,A)` grid is completed or numerically fail-closed;
5. full-SNN `epsilon_c` is measured/censored at every registered primary state and location;
6. z counterfactuals are complete;
7. event suppression is complete or explicitly unresolved;
8. the focused m test is complete without duplicate realized-state labels;
9. the integrated model-data bridge table exists if upstream field artifacts are eligible;
10. main and diagnostic figures have been visually inspected and documented;
11. tests, commands, resource use, artifacts, safe claims, largest gap, and git status are written in `STATUS.md` and the archive report.

