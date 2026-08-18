# Topic 4 ZM-ITX: data-driven interictal-to-ictal transition under active Z/M

Date: 2026-08-17 (rev 2, 2026-08-18 after design review)
Branch: `codex/topic4-data-driven-zm-ictal-transition`
Worktree: `.worktrees/topic4-data-driven-zm-ictal-transition`
Base commit: `7393745c6777adaf88fbf0c5bc087e4c2f1c0a9e`

## Scientific question

On the frozen, patient-constrained Node + E→E + E→I substrate for `epilepsiae_1146`,
switching on the two per-neuron slow variables drives the self-limited interictal event
background into a sustained high-activity state. Prior work makes the bare version of that
statement uninformative, so the question this round asks is the narrower one:

> While the Fig.4 data-driven interictal repertoire is still present, does the frozen
> data-driven node-and-edge substrate organize **where** the pre-transition local
> susceptibility rises and **where** the transition ignites — or does it merely make an
> already-unstable system destabilize sooner?

Everything downstream follows from that framing. Time-to-transition is a *secondary* endpoint
whose role is pathway factorization, not the headline.

### Why latency cannot be the headline

`results/topic4_sef_hfo/data_driven_core_field_rev10_d/continuous_field_active_zm_d7_canary/d7_canary_verdict.json`
records that this fixed Z/M reference ran away on **98 of 98** worker runs spanning 49 distinct
continuous node fields on two development networks — `all_candidates_runaway_on_all_networks:
true`, `n_nonrunaway_workers: 0`, `safe_candidate_ids: []`. Onset times: median **7989 ms**,
min 5834, max 10291, q05 6458, q95 9981. Changing the node field across 49 candidates moved the
per-candidate mean onset only between 6362 and 9887 ms.

Boundary on that prior: D7 used a different node-field family (`d6_*` continuous fields), an
**exact no-op edge mapper**, and network seeds 1421/1422. This round uses the rev11-NLC local
connectivity mapper on 12 fresh seeds, so D7 is a strong prior on onset timing, not a guarantee.
It is why incidence is expected at ceiling and why "the Joint arm also runs away, slightly
sooner" would be a null result dressed as a finding.

### Operational definition

```
model ictal state = runaway_sustained
```

engine criterion, unchanged: 20 ms EMA of the population E rate ≥ 120 Hz sustained for
≥ 100 ms. The original engine field `runaway_early_stop_ms` is preserved verbatim and is
displayed as `model ictal onset`. Neither recovery nor termination is required.

**This label is operational only.** The project's own 2026-08-08 finding was that this engine's
sustained regime is a burst train re-igniting from population silence, not a continuous carrier.
That was determined at a different work point and **is recomputed on this round's trajectories**
(see *State characterization*); it may not be cited as-is.

## Frozen inputs (hash-verified before every run)

| Input | Path | sha256 |
|---|---|---|
| Substrate manifest | `results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc/frozen_substrate_confirmation/candidate_manifest.json` | `545b029d2d7947de5a27979e7166f6bc55ea2b01c64a67a9b736977ec960fcbb` |
| Z/M reference | `config/topic4_data_driven_snn_baseline_zm_v1.json` | `2b9586d274b85d9e3663557b5f4dfab7ac64292817667020503d144579ff8a91` |
| Substrate producer commit | `ff6cb0b782788c8d50f5342ce72c5a3b51623611` (ancestor of base commit) | — |

Four frozen arms, verbatim from `candidate_set.candidates` — no refitting:

| arm id | manifest `candidate_id` | pathways | role this round |
|---|---|---|---|
| Joint | `joint_04_control` | node + E→E + E→I | **the only arm in the primary experiment** |
| Node | `node_baseline` | node field only | latency-only, Phase 3 |
| Node+EE | `joint_04_ee_only` | + E→E | latency-only, Phase 3 |
| Node+EtoI | `joint_04_etoi_only` | + E→I | latency-only, Phase 3 |

Every formal arm runs with:

```
runtime_mode = active_z_plus_m
use_z = true, use_m = true
I_th_EI = 95.19851312666987
tau_z   = 5000 ms
tau_adp = 500 ms
eta_m   = 0.007451594355587098
```

`q_I`, `g_K`, `h_G`, EE-STD and every other slow protocol stay off. Topology, delays, I→E,
I→I, per-target incoming pathway budgets and the spatial OU accessibility process are frozen
exactly as in the rev11-NLC confirmation.

### Substrate facts recorded here so downstream text cannot drift

- Subject `epilepsiae_1146`; 15 contacts (`ICL1..ICL11`, `SCL6..SCL9`); `L = 20 mm`,
  density 100 /mm², `N_E = 32000`, `N_I = 8000`, `dt = 0.1 ms`.
- **`theta_EE = -22.805383965 deg`**, unit axis `(0.92182673, -0.38760221)`, from the patient's
  interictal stereotyped rank-gradient shared axis: source centroid `(4.199, 9.129)` → sink
  centroid `(16.479, 3.966)`, separation 13.32 mm. `AR = 2.0`. **The connectivity anisotropy
  axis is itself patient-derived**, which bounds what the spatial control can claim.
- Node field mass is exactly projected to `N_core_manual = 1129`, i.e. **3.53 % of the E
  population**. Any population-mean Z/M statistic is therefore dominated by background, not by
  the data-driven core — see *Slow-current fields*.
- Virtual contact readout is a **firing-density envelope** (2 ms frames, 5 ms smoothing),
  explicitly *not* a synaptic-current LFP. It is never called an SEEG voltage.
- In the Z/M-off reference the network sits above the common detector **41.2 %** of the time
  (`fraction_time_above_common_detector = 0.4115`). Any "the probe triggered an event" test must
  therefore be **probe-attributable** — an event present in the probe branch and absent from the
  paired sham branch — never "an event occurred".

### Existing control this round inherits

Under the identical frozen substrate with Z/M **off**, 48 runs (4 arms × seeds 1561-1572, 20 s)
produced **0 transitions**, ~105 detected and ~87 returned events per run
(`.../frozen_substrate_confirmation/workers/*.json`, `runaway == null` in 48/48). Entering the
sustained state is not something this substrate does on its own. No formal Z/M-off arm is run at
the formal seeds; this 48-run reference is the incidence control, and the fact that it uses
seeds 1561-1572 rather than 1811-1822 is stated wherever it is used. Three **canary-only**
Z/M-off runs at seeds 1801-1803 are run for a different purpose: they supply the same-seed
interictal reference distribution the Phase 1 gate compares against.

## Engine changes

All off by default, default path byte-identical, following the `dump_pathway_trace` precedent
in `src/snn_engine/kick_probe.py`.

```
post_runaway_record_ms = 0.0    # keep recording this long after detection, then stop
checkpoint_steps       = None   # absolute step indices at which to snapshot state
checkpoint_sink        = None   # callable(absolute_step, state)
resume_state           = None   # resume from a snapshot
time_offset_ms         = 0.0    # absolute clock origin for a resumed segment
```

`post_runaway_record_ms` stops at `min(detect_step + post_steps, nsteps)`; the hard duration cap
is never lifted to keep a runaway running for a figure.

Absolute time is mandatory for resumed segments: the spatial OU process advances on absolute
step indices, and the external drive, kick window, perturbation window and **forced-spike time**
are all judged on the absolute clock. A resumed segment allocates recorder arrays for the
continuation length only.

### Checkpoint contents (`src/snn_engine/checkpoint.py`, single enumeration point)

`V`, `ref`, `s_E`, `I_E`, `s_I`, `I_I`, `ring_sE`, `ring_sI`, external-drive OU scalar `xi`,
`net["rng"].bit_generator.state`, raster sampling indices `ras_keep`, early-stop EMA state,
`MZSlowVars.{z, m, _I_I_last, _step_index, accumulator}`, `SpatialOUDrive.{_state, _cached,
_next_step, _last_step, _rng.bit_generator.state}`, and the absolute step index. Capture is at
the top of a step, before any RNG draw, so resuming re-executes that step identically.

### Acceptance gates (all bit-level)

| Gate | Content |
|---|---|
| **A — default-path parity** | With `mz.mode = off`, checkpoint off, perturbation off, `post_runaway_record_ms = 0`, a fresh run of `joint_04_control` at seed 1561 reproduces the archived `.../workers/joint_04_control_seed_1561.npz` bit-for-bit. Engineering parity audit; does not conflict with all formal arms running Z/M on. |
| **B — sham reload** | Reload a checkpoint, continue 100 ms unperturbed, bit-identical to the original trajectory. |
| **C — perturbed reload** | Reload a checkpoint, inject the forced E packet, continue; bit-identical to a full run from `t = 0` with the same packet at the same absolute time. Gate B alone cannot catch state the unperturbed window never exercises; the existing full-rerun + `forced_spike_mask` path supplies the oracle. |

## Code layout

The rev11-NLC producer `scripts/run_topic4_rev10_r_edge_flow_worker.py` is **never modified**.
New code:

```
src/topic4_zm_ictal_transition.py                    substrate rebuild from the frozen manifest
src/snn_engine/checkpoint.py                         state capture/restore
src/topic4_zm_d4.py                                  covariant field+flow spatial transform
src/topic4_zm_state_characterization.py              what the high-activity state is
src/topic4_zm_recruitment.py                         local recruitment / spatial spread
src/topic4_zm_perturbation.py                        sites, packets, descendant-only response
src/topic4_zm_statistics.py                          paired bootstrap, censored latency, spatial null
scripts/run_topic4_zm_ictal_transition_worker.py     primary runs
scripts/run_topic4_zm_perturbation_worker.py         probe/sham pairs, incl. counterfactual splices
scripts/{launch,aggregate,audit,freeze}_topic4_zm_ictal_transition*.py
scripts/paper_figures/{plot_topic4_zm_ictal_transition_panels,build_main_figure_5}.py
config/topic4_data_driven_zm_ictal_transition_v1.json
```

Gate A proves the new rebuild path is the same substrate as the archived one.

## The perturbation endpoint, defined so it cannot be contaminated

Two endpoints, kept separate. Collapsing them was the review's first finding.

### E1 — sub-event finite response (PRIMARY)

```
susceptibility(site) = total descendant probe-minus-sham excess E spikes over 0-200 ms
```

**Descendant**: the directly injected spikes are removed before counting, by reusing
`src.topic4_forced_source_capacity.exclude_injected_packet_frame(forced, sham, packet_mask,
trigger_step=...)`, which replaces the injection frame's packet-neuron entries with the sham's.
Without this, a 256-cell packet contributes 256 excess spikes with zero recursive amplification
and would trivially clear any threshold on the order of 200.

E1 is only meaningful in a regime where the probe does **not** ignite. The dose is frozen so
that it does not.

### E2 — ignition (SECONDARY, nonlinear)

```
ignition(site) = the probe branch shows a probe-attributable detector-qualified event,
                 or reaches the model ictal criterion, within the response window
onset advance  = sham onset - probe onset
```

"Probe-attributable" means present in the probe branch and absent from the paired sham branch —
required because the unperturbed network is already above the detector 41 % of the time.

E2 is measured **only at the 6 representative sites**, **only after E1 has resolved**, and is
reported as a distinct endpoint. E1 numbers are never pooled with E2 numbers.

### Dose freeze

Ladder `{16, 32, 64, 128, 256}` E cells, calibrated on **baseline checkpoints only**, blind to
any pre-ictal or patient-derived quantity (the calibration script refuses any `--label` other
than `baseline`). Across `3 canary seeds × 6 representative sites = 18` units, the frozen dose
is the **largest** ladder rung satisfying **all** of:

```
0 / 18 units show a probe-attributable detector-qualified event
0 / 18 units reach the model ictal criterion
median descendant susceptibility over the 18 units >= 50 excess spikes
```

Largest, not smallest: within the no-ignition regime a bigger probe gives a better-conditioned
finite response. The 50-spike floor is a measurability floor on *descendant* spikes and is
unrelated to the packet size.

If no rung satisfies all three, the verdict is **`NO_SUBEVENT_PROBE_REGIME`** and Phase 3 does
not run. The finding reported is that this work point admits no sub-ignition probe, which is
itself informative and must not be worked around by loosening the ignition criterion.

## Attributing the susceptibility change: Z/M slow state versus fast-state proximity

A pre-ictal checkpoint differs from a baseline checkpoint in `z` and `m` **and** in membrane
potentials, synaptic currents, OU field state, and the fast-state residue of the most recent
event. A larger pre-ictal response therefore only licenses "response grows as the transition
approaches", not "the Z/M slow state drives the growth". The same confound bit the earlier
slow-fast line.

Five short branches at the canary checkpoints, 6 representative sites, 200 ms each, resolve it:

| id | fast state (V, currents, rings, OU, RNG) | `z` | `m` |
|---|---|---|---|
| `native_baseline` | baseline | baseline | baseline |
| `native_pre_ictal` | pre-ictal | pre-ictal | pre-ictal |
| `reset_z` | pre-ictal | **baseline** | pre-ictal |
| `reset_m` | pre-ictal | pre-ictal | **baseline** |
| `reset_zm` | pre-ictal | **baseline** | **baseline** |
| `slow_only` | **baseline** | pre-ictal | pre-ictal |

`reset_*` measure necessity, `slow_only` measures sufficiency.

**Boundary, stated in the report and the figure caption:** a spliced state is off-manifold — the
dynamics never visit "pre-ictal fast state with baseline `z`". These branches answer *which
variable carries the elevated responsiveness*, not *what would have happened*. They are a
counterfactual attribution test, not a trajectory.

**Without this block the permitted claim is only "pre-ictal susceptibility on a Z/M-active
trajectory".** With it, the claim may name the carrying variable.

## Spatial sampling and the primary spatial endpoint

Uniform sampling, because a per-network spatial correlation and its spatial null need a grid:

- **All 12 Joint networks** use the same frozen **7×7 grid** spanning `[3, 17] mm` in both sheet
  axes (2.333 mm spacing), at **both** the baseline and pre-ictal checkpoints, **200 ms
  sham/probe only**. No onset-advance continuation is run at grid points.
- The **6 representative sites** — patient source centroid, sink centroid, axis midpoint, two
  points ±4 mm from the midpoint along the axis normal, and the sheet centre — carry the dose
  calibration, the counterfactual splices, and E2.
- A grid site is dropped only if fewer than the frozen dose's packet size of E neurons lie
  within 1.0 mm; dropped sites are listed.

Primary spatial endpoint: the spatial relation of the susceptibility field to the substrate's
own structure —

```
node field h
outgoing E->E gain per E neuron   (post-mapping / pre-mapping outgoing weight)
outgoing E->I gain per E neuron
local ictal recruitment time      (see below)
```

each averaged over the E neurons within 1.0 mm of each grid site so it lives on the same grid.

**Collinearity is reported before interpretation.** `h`, the E→E gain and the E→I gain are all
functions of the same field and are expected to be strongly collinear; their pairwise Spearman
correlations are reported first, and three separate correlations are **never** presented as
three independent mechanisms. If the pairwise correlations exceed 0.7 the three are reported as
one *data-driven substrate structure* family with a single headline number plus partial
correlations as a diagnostic.

Per-network statistic: Spearman r on the 49 sites, with a **block circular-shift** null
(the covariate field is rigidly shifted on the 7×7 torus, preserving its autocorrelation); 49
distinct shifts give an exact but coarse per-network p, floor 1/49. The load-bearing test is the
cohort-level paired bootstrap over the 12 per-network r values, not the per-network p.

Also reported: hotspot compactness (top-quintile sites, mean pairwise distance versus random
equal-size subsets of the retained grid) at baseline and pre-ictal.

## Local recruitment, replacing first-spike onset density

A "first spike inside the 100 ms before detection" statistic is not a recruitment measure in
this network: with the observed background the great majority of E neurons fire at least once in
any 100 ms window, so the statistic is close to uniform noise. Replaced by:

1. Assign E neurons to fixed 1 mm spatial bins.
2. Compute each bin's smoothed local firing rate (5 ms kernel).
3. Threshold each bin against **its own** interictal baseline — the q99 of that bin's rate over
   a pre-onset reference window — requiring the excess to persist ≥ 15 ms.
4. Report per bin the local recruitment time; across bins the **10 % → 90 % spatial recruitment
   duration**, and the axial versus off-axial lag along the patient axis.

This is the measurement that separates **sequential local spread** from **near-simultaneous
whole-field ignition**, which is the distinction the earlier `q_I`/`g_K` line failed to make.

## Is it still the Fig.4 interictal repertoire?

If switching Z/M on produces high-OOD single-mode events that then run away, this round is not
about "the data-driven interictal repertoire going ictal". A **claim gate**, not a run blocker:

```
INTERICTAL_REPERTOIRE_RETAINED
```

computed over all returned events before onset with the frozen patient direction classifier:
out-of-distribution fraction, support for both modes, KMeans match — each compared against the
distribution across the 48 Z/M-off reference runs.

- Retained → the round may be written as *data-driven interictal modes → model ictal state*.
- Not retained → the wording is *low-activity background → high-activity state*, and every mode
  statement is dropped.

The runaway itself is never assigned a Mode 1/2 label.

## Endpoint tiers (fixed here, before any run)

```
Primary          E1: pre-ictal minus baseline descendant susceptibility
                 (sham-subtracted, per-neuron, mean over that network's retained grid sites)
Primary spatial  spatial relation of the susceptibility field to h / outgoing E->E gain /
                 outgoing E->I gain / local recruitment time, with collinearity reported first
Attribution      the five counterfactual splices: which variable carries the change
Secondary        E2 ignition and onset advance at the representative sites;
                 four-arm time-to-model-ictal
Descriptive      virtual-contact readout, projected Z/M trajectory, high-activity state
                 morphology, Mode 1/2 and OOD evolution, spatial recruitment duration
```

**Two orthogonal axes, never collapsed.** Tier is fixed above. Contact dependence is separate:
time-to-onset, per-neuron susceptibility, hotspot compactness, local recruitment time and
spatial recruitment duration are contact-independent; contact envelope burden and contact
recruitment are contact-dependent. A contact-dependent quantity may not carry a mechanism
conclusion for a substrate transform.

### Latency under censoring

```
primary latency endpoint (secondary tier): restricted ictal-free time over [0, 20 s]
secondary: paired onset-time difference among networks where both arms entered
also reported: proportion entering within 20 s
```

20 s is never used as an onset time. Given D7, incidence is expected at ceiling and is reported
as background.

## Controls

**Observation control (zero simulation).** The primary worker records the contact envelope for
the original montage **and** for seven pre-frozen D4-transformed montages in the same SNN run —
the spikes are already in memory, so this costs one extra envelope sampling per montage and no
simulation. Offline reconstruction from the original envelope alone is impossible, which is why
the montages must be declared before the run. This control answers **readout dependence only**
and may never support a mechanism conclusion.

**Substrate control — `matched spatial re-registration`.** The node field `h` and the two
directed flow coefficients of each pathway are transformed together by the **same** square-
symmetry element, keeping the network, the contacts, the anisotropy axis and every Z/M parameter
fixed.

The covariant coefficient transform is required because the flow features are signed and linear
in displacement (`src/topic4_local_connectivity.py:50-62`: `displacement = (source_xy -
target_xy) / length_scale`, features `interaction * displacement[:, 0]` and `[:, 1]`). Rotating
the field alone reverses the correspondence between field structure and the flow it drives.
Rotating `(c_x, c_y)` by the same matrix restores it, and because the group elements only swap
and negate components, the frozen bounds `±0.15` survive element-wise with no re-clipping.

**This is not an isometric copy of the substrate.** The field-and-flow *rule* is transformed as
a rigid unit; the realized random graph, its patient-derived anisotropic topology and the
contacts are not. The control therefore asks whether the node field must be **co-registered**
with the patient axis, the realized graph and the electrodes — not whether patient structure
matters at all.

Design: the formal control uses **`r180` on all 12 Joint seeds**, one transform, one
interpretation, paired 1:1 with the data-driven runs. Under `r180` the field's source end lands
on the patient's sink end while the undirected axis alignment is preserved. `r90` and one mirror
are run on the 3 canary seeds only and reported descriptively; **no claim is made that the
square's seven non-identity elements were surveyed at power.**

A control run that does not transition has a baseline checkpoint but no pre-ictal checkpoint; it
contributes to the censored latency endpoint and the baseline map, is excluded from the paired
contrast, and the excluded count is printed next to every control comparison. If more than half
the control runs fail to transition, the paired contrast is emitted as not evaluable rather than
computed on the survivors.

## Slow-current fields and the projected trajectory

Accumulated inside `MZSlowVars.apply_currents`, which already receives `I_I` and owns `z` and
`m` at exactly the instant the membrane equation consumes them. Off by default → byte-identical.

```
D_i = mean_t[ (1 - z_i(t)) * I_I,i(t) ]   over the sham window after the checkpoint
A_i = mean_t[ eta_m * m_i(t) ]            over the same window
net slow current = D_i - A_i
```

Product averages, not products of averages. Neuron fields map to a display grid with a uniform
grid and a fixed isotropic kernel; **contact-density weighting is forbidden**. Baseline and
pre-ictal `D - A` fields share one colour scale with static `h` overlaid as contours; static `h`
is not redrawn as its own field at two time points.

Projected trajectory — **`h`-weighted**, because the core is only 3.53 % of the E population and
a plain population mean would mostly report background:

```
x(t) = 1 - (sum_i h_i z_i(t)) / (sum_i h_i)
y(t) = eta_m * (sum_i h_i m_i(t)) / (sum_i h_i)
```

The unweighted population mean is drawn as a thin grey reference line on the same axes and its
full version goes to the supplement. Coloured by time, with baseline, pre-ictal and model ictal
onset marked. The panel title is `Projected Z/M trajectory`; it may only be called a phase
portrait if a 2-D drift field and nullclines are estimated, which this round does not do.

## State characterization (required, recomputed)

From the 500 ms post-detection recording on **this round's** trajectories: active/silent duration
distributions, burst interval, re-ignition rate, fraction of 20 ms windows with zero population
spiking, population peak rate, spatial recruitment fraction, 15-contact recruitment, and a
30–80 Hz band proxy with its amplitude change.

The band proxy is compared against a **length-matched 500 ms interictal window**, and the
resolution limit (500 ms gives ~15 cycles and ~2 Hz resolution at 30 Hz) is stated wherever the
number appears. Captions state that the state is defined operationally by the runaway threshold.

## Phase order

Staged so the cheapest decisive result comes first and a dead end costs three canary networks.

| Phase | Content | Gate to continue |
|---|---|---|
| **0** | Gates A/B/C; parallel montage recording; descendant-spike metric | all three gates bit-exact |
| **1A** | 3 Joint Z/M-on canary (seeds 1801-1803) + 3 same-seed Z/M-off canary | `INTERICTAL_BASELINE_AVAILABLE` in **≥ 2 of 3** networks |
| **1B** | dose freeze over the 18 baseline units; counterfactual splices; repertoire gate; local-recruitment audit | dose found, i.e. not `NO_SUBEVENT_PROBE_REGIME` |
| **2** | **Joint arm only**, 12 seeds; uniform 7×7 short response maps at baseline and pre-ictal | E1 resolved: the 90 % CI of the paired pre-minus-baseline difference **excludes 0** |
| **3** | Node / Node+EE / Node+EtoI latency arms (pass 1 only); E2 ignition and onset advance at the representative sites; `r180` spatial control | — |
| **4** | figures, report, freeze | — |

If Phase 2's interval includes 0, the round stops there and reports; the four arms, the spatial
control and every long onset-advance continuation are not run. If Phase 1A's gate fails,
the finding reported is that the current Z/M work point has no interpretable interictal residence
segment; **the baseline checkpoint is not moved earlier to rescue it.**

`INTERICTAL_BASELINE_AVAILABLE`, per network:

```
      model ictal onset >= 2500 ms
  AND >= 3 returned (self-limited) events occur before onset
  AND the baseline window [1500, 2000] ms shows no sustained high activity
      criterion: median of the 20 ms-EMA population E rate over that window
                 <= the 95th percentile of the same statistic across the
                    same-seed Z/M-off canary runs
```

All three clauses are evaluated and **every failing clause is reported**, not only the first.
Per D7's 5834 ms minimum across 98 runs, the first clause is expected to pass comfortably; in
the formal phase a network failing it is excluded from the perturbation analysis individually
rather than stopping the round.

Non-Joint arms carry latency only and therefore run **pass 1 only** — they need no
onset-relative checkpoints.

## Figure contract

Outputs under `results/paper-ready-figure/fig5/figures/`:

```
fig5-data-driven-zm-transition-candidate.{png,pdf,gif}
fig5-data-driven-zm-transition-main.{png,pdf}          6 panels A-F
fig5-data-driven-zm-transition-supplement.{png,pdf}    4 panels G-J
README.md
metadata.json
```

Both assemblies come from one set of producers; no simulation is re-run for the second layout.

```
A  frozen data-driven Node / E->E / E->I substrate + the Z/M mechanism
B  continuous 15-contact virtual readout and population activity
C  h-weighted projected Z/M trajectory, population mean as a thin grey reference
D  baseline vs pre-ictal slow-current fields (D - A), static h as contours
E  baseline vs pre-ictal perturbation response fields
F  susceptibility maps and the pre-minus-baseline change
G  counterfactual attribution: which of z / m / fast state carries the change
H  local recruitment map and the 10-90 % spatial recruitment duration
I  data-driven substrate vs the r180 re-registration control
J  pre-ictal Mode 1/2, KMeans match and OOD evolution
```

B–F carry the primary evidence; G–J are mechanism decomposition and robustness. The four-arm
latency panel and the observation control go to the supplement, not the main figure.

Fig4's red/blue mode colours are reused; the model ictal state uses a separate dark grey. No
PASS/FAIL text, no internal status codes, no long explanations inside the figure. The virtual
readout is one continuous trace, never spliced, and is never labelled a clinical SEEG voltage.
Panel redundancy is re-checked after the first render — E answers "where does activity go when
one fixed site is perturbed", F answers "which site is more sensitive"; if they collapse, one is
replaced.

## Claim boundary

Permitted, if supported:

> While the data-driven interictal repertoire is still present, local perturbation
> susceptibility rises before the model ictal transition, that rise is carried by the Z/M slow
> state rather than by fast-state proximity alone, and it is spatially organized along the
> frozen data-driven substrate.

If E1 grows but the counterfactual splices show the change is carried by fast state:

> Responsiveness grows as the transition approaches, but on this substrate the growth is not
> attributable to the accumulated Z/M slow state.

If E1 grows and is Z/M-carried but shows no spatial relation to the substrate:

> Z/M slow state raises pre-transition responsiveness without the data-driven node-edge
> structure organizing where that responsiveness rises.

If the `r180` control is indistinguishable:

> The co-registration of the node field with the patient axis, the realized graph and the
> electrodes is not a necessary organizing factor **at this power**.

**Because `theta_EE` is itself patient-derived and is held fixed under the transform, this round
cannot test whether patient spatial structure matters at all. It tests only mutual registration.
Writing the control's null result as "patient spatial structure is unnecessary" is forbidden.**

If the repertoire gate fails, every sentence above replaces "interictal repertoire" with
"low-activity background" and drops all mode language.

Never permitted, regardless of outcome:

```
clinical seizure reproduced
patient seizure onset predicted
complete seizure lifecycle recovered
Z/M identified as the patient biological mechanism
```

The prior LC3 non-carrier determination may not be restated as a result of this round; it is
context, and this round's own state characterization is what is reported.

## Execution discipline

New worktree only; the dirty main worktree is untouched. Long runs via `systemd-run --user` +
`nohup`. Numeric threads pinned to 1 per worker. Worker count derived from a measured memory
sentinel, capped at 8, always leaving ≥ 32 GiB available (prior peak RSS 14.6 GiB per 20 s
worker; the machine has 251 GiB total, 229 GiB available, 80 cores). The monitor polls every
600 s for memory, disk, worker state and module hashes. OOM, non-finite values, hash drift or a
checkpoint-replay mismatch stops new work; processes belonging to other worktrees are never
killed.

The frozen network cache directory no longer exists, so every seed is rebuilt. The rebuild guard
has been verified to pass at the base commit (`params.py`, `connectivity.py`,
`connectivity_rot.py` match the hashes in `node_kick_canary.json`; numpy 1.26.4), and the
archived seed-1561 pickle hash
`dba81d32d6c542bda4d1cfa0de196551c16f811a88c0864c7572a8db60852828` checks that a rebuilt network
is the same network.

Because the onset-relative checkpoints are defined against an onset not known in advance, each
**Joint** primary run is two passes: pass 1 records the onset while emitting only the 2000 ms
baseline checkpoint; pass 2 resumes from it and stops at `onset - 500 ms`, emitting the two
onset-relative checkpoints. Pass 2's overlap with pass 1 is asserted bit-identical — Gate B in
production. Non-Joint arms run pass 1 only.

`results/` is gitignored except three legacy files. Small decision artifacts (gate verdicts,
manifests, cohort summaries, figure files) are committed with `git add -f`; bulk `.npz` is not
committed.

Cost scales from the measured **94.5 s of wall clock per simulated second**. Taking D7's median
onset of ~8 s as the working prior, at 8 workers: gates ~1.2 h, Phase 1A ~0.4 h, Phase 1B ~0.3 h,
Phase 2 ~1.4 h, Phase 3 ~4.2 h, figures ~1 h — about **8.5 h**, of which only the first ~3.3 h is
spent before the Phase 2 gate decides whether the rest runs.

On completion: `DONE.json`, a desktop notification, a scientific report, the figure README,
provenance, and scoped commits. The report's first paragraph answers, in order: can this
continue, what is safe to conclude now, what is the largest gap, what is next.
