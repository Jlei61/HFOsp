# Topic 4 ZM-ITX: data-driven interictal-to-ictal transition under active Z/M

Date: 2026-08-17
Branch: `codex/topic4-data-driven-zm-ictal-transition`
Worktree: `.worktrees/topic4-data-driven-zm-ictal-transition`
Base commit: `7393745c6777adaf88fbf0c5bc087e4c2f1c0a9e`

## Scientific question

On the frozen, patient-constrained Node + E→E + E→I substrate for `epilepsiae_1146`,
switching on the two per-neuron slow variables (Z disinhibition, M adaptation) drives the
self-limited interictal event background into a sustained high-activity state. The question
this round answers is **not** "how fast does it get there" but:

> Does the local perturbation susceptibility rise before the transition, and is that rise
> spatially organized along the data-driven node field and the data-driven local edge gains?

Time-to-transition is retained as a *secondary* endpoint whose role is pathway
factorization (which of E→E / E→I changes reachability), not as the headline result.

### Operational definition (this round)

```
model ictal state = runaway_sustained
```

engine criterion, unchanged: 20 ms EMA of the population E rate ≥ 120 Hz sustained for
≥ 100 ms. The original engine field `runaway_early_stop_ms` is preserved verbatim and is
displayed as `model ictal onset`. Neither recovery nor termination is required.

**This label is operational only.** The project's own prior finding (2026-08-08,
`docs/topic4_sef_hfo.md` LC3 line) is that this engine's sustained high-activity regime is a
burst train that re-ignites from complete population silence, not a continuous carrier.
That determination was made on a different work point and **must be recomputed on this
round's trajectories** (see *State characterization* below); it may not be cited as-is.

## Frozen inputs (hash-verified before every run)

| Input | Path | sha256 |
|---|---|---|
| Substrate manifest | `results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc/frozen_substrate_confirmation/candidate_manifest.json` | `545b029d2d7947de5a27979e7166f6bc55ea2b01c64a67a9b736977ec960fcbb` |
| Z/M reference | `config/topic4_data_driven_snn_baseline_zm_v1.json` | `2b9586d274b85d9e3663557b5f4dfab7ac64292817667020503d144579ff8a91` |
| Substrate producer commit | `ff6cb0b782788c8d50f5342ce72c5a3b51623611` (ancestor of base commit) | — |

Four frozen arms, taken verbatim from `candidate_set.candidates` — no refitting:

| arm id | manifest `candidate_id` | pathways |
|---|---|---|
| Node | `node_baseline` | node field only |
| Node+EE | `joint_04_ee_only` | + E→E |
| Node+EtoI | `joint_04_etoi_only` | + E→I |
| Joint | `joint_04_control` | + E→E and E→I (primary) |

Every formal arm runs with:

```
runtime_mode = active_z_plus_m
use_z = true, use_m = true
I_th_EI = 95.19851312666987
tau_z   = 5000 ms
tau_adp = 500 ms
eta_m   = 0.007451594355587098
```

`q_I`, `g_K`, `h_G`, EE-STD and every other slow protocol stay off. Topology, delays,
I→E, I→I, per-target incoming pathway budgets and the spatial OU accessibility process are
frozen exactly as in the rev11-NLC confirmation.

### Substrate facts recorded here so downstream text cannot drift

- Subject: `epilepsiae_1146`; 15 contacts (`ICL1..ICL11`, `SCL6..SCL9`); sheet `L = 20 mm`,
  density 100 /mm², `N_E = 32000`, `N_I = 8000`, `dt = 0.1 ms`.
- **`theta_EE = -22.805383965 deg`, unit axis `(0.92182673, -0.38760221)`**, derived from the
  patient's interictal stereotyped rank-gradient shared axis: source centroid
  `(4.199, 9.129)` → sink centroid `(16.479, 3.966)`, separation 13.32 mm. `AR = 2.0`.
  **The connectivity anisotropy axis is itself patient-derived.** This constrains what the
  spatial null can claim (see *Substrate null*).
- Node field mass is exactly projected to `N_core_manual = 1129`, so any spatial transform
  of the field preserves field mass exactly.
- Virtual contact readout is a **firing-density envelope** (2 ms frames, 5 ms smoothing),
  explicitly *not* a synaptic-current LFP. It may never be called an SEEG voltage.

### Existing control this round inherits

Under the identical frozen substrate with Z/M **off**, 48 runs (4 arms × 12 network seeds
1561–1572, 20 s each) produced **0 transitions**, with ~105 self-limited events per 20 s run
(`.../frozen_substrate_confirmation/workers/*.json`, field `runaway == null` in 48/48).
Entering the sustained state is therefore not something this substrate does on its own.
No fresh Z/M-off arm is run; this 48-run reference is the cited control, and the fact that it
uses network seeds 1561–1572 rather than this round's 1811–1822 is stated wherever it is used.

## Engine changes

All four are off by default and the default path must stay byte-identical, following the
pattern already used by `dump_pathway_trace` in `src/snn_engine/kick_probe.py`.

```
post_runaway_record_ms = 0.0    # keep recording this long after detection, then stop
checkpoint_steps       = None   # absolute step indices at which to snapshot state
resume_state           = None   # resume from a snapshot
time_offset_ms         = 0.0    # absolute clock origin for a resumed segment
```

`post_runaway_record_ms` stops at `min(detect_step + post_steps, nsteps)`; the hard duration
cap is never lifted to keep a runaway running for a figure.

Absolute time is mandatory for resumed segments: the spatial OU process advances on absolute
step indices and the external drive, kick window and perturbation windows are all judged on
absolute `tm`. A resumed segment allocates recorder arrays for the continuation length only
(a 200 ms probe must not allocate 20 s of `E_spk_bool`).

### Checkpoint contents (`src/snn_engine/checkpoint.py`, single enumeration point)

`V`, `ref`, `s_E`, `I_E`, `s_I`, `I_I`, `ring_sE`, `ring_sI`, external-drive OU scalar `xi`,
`net["rng"].bit_generator.state`, raster sampling indices `ras_keep`, early-stop EMA state,
`MZSlowVars.{z, m, _I_I_last, _step_index}`, `SpatialOUDrive.{_state, _cached, _next_step,
_last_step, _rng.bit_generator.state}`, and the absolute step index. Capture point is the top
of a step, before any RNG draw, so resuming re-executes that step identically.

### Acceptance gates (all bit-level)

| Gate | Content |
|---|---|
| **A — default-path parity** | With `mz.mode = off`, checkpoint off, perturbation off, `post_runaway_record_ms = 0`, a fresh run of `joint_04_control` at seed 1561 reproduces the archived `.../workers/joint_04_control_seed_1561.npz` bit-for-bit on `onsets`, `ranks`, `event_t_on_ms`, `event_t_off_ms`, `event_returned`, `active_fraction`, `contact_envelope` and the spatial OU traces. This is an engineering parity audit and does not conflict with all formal arms running Z/M on. |
| **B — sham reload** | Reload a checkpoint and continue 100 ms with no perturbation; bit-identical to the original trajectory. |
| **C — perturbed reload** | Reload a checkpoint, inject the forced E packet, and continue; bit-identical to a full run from `t = 0` with the same packet at the same absolute time. Gate B alone cannot catch state that the unperturbed window never exercises (delay-ring slots, RNG advance under a different spike set); the existing full-rerun + `forced_spike_mask` path in `scripts/run_topic4_rev9l_forced_source_worker.py` supplies the oracle. |

## Code layout

The rev11-NLC producer script `scripts/run_topic4_rev10_r_edge_flow_worker.py` is **not
modified** — changing it would make that round non-reproducible in place. New code:

```
src/topic4_zm_ictal_transition.py                    substrate rebuild from the frozen manifest
src/snn_engine/checkpoint.py                         state capture/restore
scripts/run_topic4_zm_ictal_transition_worker.py     Phase 1/2 primary runs
scripts/run_topic4_zm_perturbation_worker.py         Phase 3 (loads net + checkpoint once,
                                                     loops all probe sites)
scripts/{launch,aggregate,audit,freeze}_topic4_zm_ictal_transition*.py
config/topic4_data_driven_zm_ictal_transition_v1.json
```

Gate A is what proves the new rebuild path is the same substrate as the archived one.

## Phases

### Phase 1 — canary and the one science gate

Fresh network seeds `1801-1803`, `joint_04_control` only, Z/M on, 20 s,
`post_runaway_record_ms = 500`. Confirms the NLC mapper composes with Z/M, that onset is
recordable, that the 15-contact readout / per-neuron Z, M / checkpoints are all present, and
measures memory and wall time. It also freezes the perturbation dose.

**This round's only new science blocker:**

```
INTERICTAL_BASELINE_AVAILABLE =
      model ictal onset >= 2500 ms
  AND >= 3 returned (self-limited) events occur before onset
  AND the baseline window [1500, 2000] ms shows no sustained high activity
      criterion: median of the 20 ms-EMA population E rate over that window
                 <= the 95th percentile of the same statistic across the 48
                 Z/M-off reference runs
```

If it fails, the finding reported is that the current Z/M work point has no interpretable
interictal residence segment, and a separate decision is taken about parameter calibration.
**The baseline checkpoint is not moved earlier to rescue the run.**

Dose calibration (baseline only, blind to any pre-ictal or patient quantity): packet sizes
`{16, 32, 64, 128, 256}` E cells; pick the **smallest** size whose median 50–200 ms excess E
spikes across the 6 representative sites is **≥ 200** and which triggers a detector-qualified
population event at **at most 1** of those 6 sites.

### Phase 2 — paired formal runs

Fresh network seeds `1811-1822`; 4 arms per network; Z/M on in all four; 20 s cap;
`post_runaway_record_ms = 500`. Network seed is the independent unit; the three non-Node arms
are compared to Node as paired differences with a network bootstrap 90 % CI.

Recorded per run: transition incidence by 20 s, model ictal onset, returned interictal event
rate before onset, Mode 1 / Mode 2 (TA-like / TB-like) occupancy before onset, OOD fraction
before onset, pre-ictal virtual-contact burden, neuron-level spatial onset density, and the
state-characterization block.

### Phase 3 — baseline and pre-ictal perturbation

Per network with a transition:

```
baseline checkpoint     = 2000 ms
pre-ictal checkpoint    = onset - 500 ms
sensitivity checkpoint  = onset - 1000 ms
```

Networks with onset < 2500 ms stay in the onset analysis but leave the perturbation analysis.
Networks with onset < 3500 ms lose only the sensitivity checkpoint.

From each checkpoint, two branches share the RNG stream (the forced packet consumes no RNG):
a **sham** continuation and a **matched local E-neuron packet** (the frozen dose, applied to
the packet-size nearest E neurons to the site). Every reported response is `probe - sham`.

Sites are frozen geometrically before Phase 1, from the sheet and the patient axis alone —
never from any run's output:

- **7×7 grid**, spanning `[3, 17] mm` in both sheet axes at 2.333 mm spacing, on seeds
  `1811-1813`. A site is dropped only if fewer than the dose's packet size of E neurons lie
  within 1.0 mm of it; dropped sites are listed.
- **6 representative sites**, on the other nine seeds and used for dose calibration in
  Phase 1: the patient source centroid, the patient sink centroid, the axis midpoint, two
  points at ±4 mm from the midpoint along the axis normal, and the sheet centre.

Per site, with `t = 0` at packet injection: excess E spikes 0–50 ms, excess E spikes
50–200 ms, response `r90` (radius holding 90 % of the excess spikes about the packet
centroid), virtual-contact excess energy, and ictal-onset advance = sham onset − probe onset.

**Canonical susceptibility scalar** (used for the primary endpoint and every map):

```
susceptibility(site) = total probe-minus-sham excess E spikes over 0-200 ms
```

The 0–50 / 50–200 ms split is a reported decomposition of that scalar, not a competing
definition. A network's susceptibility value is the **mean over its retained sites**; the
paired primary contrast is `pre-ictal - baseline` within network, with a network bootstrap
90 % CI over the 12 (or fewer) evaluable networks.

**Hotspot** = the set of grid sites in the top quintile of `susceptibility`; hotspot
compactness = mean pairwise distance among those sites, compared against the same statistic
for random equal-size site subsets drawn from the retained grid.

**Neuron-level ictal onset density** = the spatial density of E neurons whose first spike
inside the 100 ms window ending at the detection step falls in that window, i.e. where the
sustained state first recruits, expressed on the same display grid as the susceptibility maps
and independent of the contacts.

Onset advance is measured at **both** the baseline and the pre-ictal checkpoint. The pre-ictal
continuation is short; the baseline continuation is capped at absolute 20 s and recorded as
right-censored if the branch has not transitioned by then. The sensitivity checkpoint yields
response metrics only, no onset advance.

Products: baseline susceptibility map, pre-ictal susceptibility map, pre-minus-baseline map,
baseline response field, pre-ictal response field. Every paired map shares grid, dose and
colour scale.

### Nulls

**Observation null (zero simulation).** Re-read an already-computed trajectory with the
contacts moved / relabelled. Answers readout dependence only. It may never support a
mechanism conclusion.

**Substrate null (covariant D4).** Jointly transform the node field `h` **and** the two
directed flow coefficients `(source_target_flow_x, source_target_flow_y)` of each pathway by
the same element of the square's symmetry group, keeping the network, the contacts, the
anisotropy axis and every Z/M parameter fixed.

The flow features are signed and linear in displacement
(`src/topic4_local_connectivity.py:50-62`,
`displacement = (source_xy - target_xy) / length_scale`, features
`interaction * displacement[:, 0]` and `interaction * displacement[:, 1]`). Rotating the
field alone therefore reverses the correspondence between field structure and the flow it
drives — the transformed substrate would no longer be a copy of the original. Rotating
`(c_x, c_y)` covariantly restores it: for the 90°-multiple rotations and the mirrors, the two
components are only swapped and negated, so the frozen coefficient bounds `±0.15` are
preserved element-wise and no re-clipping is required. Field mass is preserved exactly by the
budget projection. The transformed substrate is thus a **strict isometric copy**.

Assignment: 12 paired null runs on the **Joint arm only**, one frozen non-identity D4 element
per network seed `1811-1822`, chosen so all seven elements are used once or twice. This yields
12 paired units against the 12 Joint data-driven runs at one quarter the cost of a 7 × 12
factorial. Per-element values are reported descriptively and **no claim is made that all seven
elements were surveyed at power**. Each null run also receives the 6-site perturbation
protocol at both checkpoints, because the primary endpoint is a susceptibility contrast, not a
latency.

A null run that does not transition within 20 s has a baseline checkpoint but no pre-ictal
checkpoint. Such a network contributes to the censored latency endpoint and to the baseline
susceptibility map, and is **excluded from the paired pre-minus-baseline contrast**, with the
excluded count reported next to every null comparison. If more than half the null runs fail to
transition, the paired susceptibility contrast is reported as not evaluable rather than
computed on the surviving half.

The 180° element is reported separately and first: under the covariant transform it preserves
field-to-undirected-axis alignment while reversing the field's directed sense relative to the
patient axis — the field's source end lands on the patient's sink end. Given that this project
is about two approximately reversed propagation templates, this element is a substantive
probe, not a filler control. It is named `axis-preserving, flow-consistent spatial transform`,
never "electrode-alignment-only control".

## Endpoint tiers (fixed here, before any run)

```
Primary          pre-ictal susceptibility - baseline susceptibility
                 (sham-subtracted, per-neuron, aggregated over probe sites)
Primary spatial  spatial relation of susceptibility hotspots to
                   - node field h
                   - outgoing E->E gain per E neuron (post-mapping / pre-mapping
                     outgoing weight; incoming budget is conserved by contract, so
                     only the outgoing side carries the mapper's effect)
                   - outgoing E->I gain per E neuron
                   - subsequent neuron-level ictal onset density
Secondary        four-arm time-to-model-ictal (serves pathway factorization)
Descriptive      virtual-contact readout, projected Z/M trajectory,
                 high-activity state morphology, Mode 1/2 and OOD evolution
```

**Two orthogonal axes, never collapsed.** Tier (primary / secondary / descriptive) is fixed
above. Contact dependence is a separate property: time-to-onset, per-neuron susceptibility,
hotspot compactness and neuron-level onset density are contact-independent; contact envelope
burden and contact recruitment are contact-dependent. A contact-dependent quantity may not
carry a mechanism conclusion for a substrate transform. Time-to-onset is contact-independent
*and* secondary; the two facts do not substitute for each other.

### Latency under censoring

```
primary latency endpoint (secondary tier):
    restricted ictal-free time over [0, 20 s]
secondary:
    paired onset-time difference among networks where both arms entered
also reported:
    proportion entering within 20 s
```

20 s is never used as an onset time. Incidence is expected at or near ceiling given the
existing 3/3 Z/M-on record, and is reported as background, not as a result.

## Slow-current fields and the projected trajectory

Accumulated inside `MZSlowVars.apply_currents`, which already receives `I_I` and owns `z` and
`m` at exactly the instant the membrane equation consumes them. Off by default → byte-identical.

```
D_i = mean_t[ (1 - z_i(t)) * I_I,i(t) ]   over the sham window after the checkpoint
A_i = mean_t[ eta_m * m_i(t) ]            over the same window
net slow current = D_i - A_i
```

Product averages, not products of averages. Neuron fields are mapped to a 2-D display grid
with a uniform grid and a fixed isotropic kernel; **contact-density weighting is forbidden**.
Baseline and pre-ictal `D - A` fields are shown with the static `h` overlaid as contours; the
static `h` is not redrawn as its own field at two time points.

Projected trajectory:

```
x(t) = 1 - mean_E[z_i(t)]
y(t) = eta_m * mean_E[m_i(t)]
```

coloured by time, with baseline, pre-ictal and model ictal onset marked. The panel title is
`Projected Z/M trajectory`; it may only be called a phase portrait if a 2-D drift field and
nullclines are actually estimated, which this round does not do.

## State characterization (required, recomputed)

Computed from the 500 ms post-detection recording on **this round's** trajectories:
active/silent duration distributions, burst interval, re-ignition rate, fraction of 20 ms
windows with zero population spiking, population peak rate, spatial recruitment fraction,
15-contact recruitment, and a 30–80 Hz band proxy with its amplitude change.

The 30–80 Hz proxy is compared against a **length-matched 500 ms interictal window**, and the
resolution limit (500 ms gives ~15 cycles and ~2 Hz resolution at 30 Hz) is stated wherever
the number appears. Figure captions state that the state is defined operationally by the
runaway threshold.

## Interictal-mode analysis

Only events that still return are classified, using the frozen patient direction classifier
from the manifest. Baseline versus the last 2 s before onset are compared on Mode 1 / Mode 2
share, KMeans match and OOD fraction. **The runaway itself is never assigned a Mode 1/2 label.**
The count of returned events inside the last 2 s is reported (it bounds this analysis) but is
not a gate.

## Figure contract

Outputs under `results/paper-ready-figure/fig5/figures/`:

```
fig5-data-driven-zm-transition-candidate.{png,pdf,gif}
fig5-data-driven-zm-transition-main.{png,pdf}          6 panels A-F
fig5-data-driven-zm-transition-supplement.{png,pdf}    4 panels G-J
README.md
metadata.json
```

Both assemblies are built from one set of producers; no simulation is re-run to make the
second layout.

```
A  frozen data-driven Node / E->E / E->I substrate + the Z/M mechanism
B  continuous 15-contact virtual readout and population activity
C  projected Z/M trajectory
D  baseline vs pre-ictal slow-current fields (D - A), static h as contours
E  baseline vs pre-ictal perturbation response fields
F  susceptibility maps and the pre-minus-baseline change
G  pathway-arm model ictal-onset latency
H  susceptibility growth vs onset advance
I  data-driven substrate vs covariant spatial transform
J  pre-ictal Mode 1/2, KMeans match and OOD evolution
```

Fig4's red/blue mode colours are reused; the model ictal state uses a separate dark grey.
No PASS/FAIL text, no internal status codes, no long explanations inside the figure. The
virtual readout is one continuous trace, never spliced, and is never labelled a clinical SEEG
voltage. Panel redundancy is re-checked after the first render — E answers "where does
activity go when one fixed site is perturbed", F answers "which site is more sensitive"; if
they collapse to the same construct, one is replaced.

## Claim boundary

Permitted, if supported:

> On a frozen patient-constrained node-edge substrate, Z/M slow state drives a self-limited
> interictal event background into a sustained high-activity state; before that state,
> local perturbation susceptibility increases and is spatially organized along the
> data-driven substrate.

If E→E / E→I only shift latency:

> Static connectivity modulates the reachability of the model ictal state without changing
> its spatial pattern.

If the covariant transform is indistinguishable:

> The mutual co-registration of the two patient-derived structures — the node field and the
> patient interictal propagation axis carried by the edge anisotropy — is not a necessary
> organizing factor for this transition **at this power**.

**Because `theta_EE` is itself patient-derived and is held fixed under the transform, this
round cannot test whether patient spatial structure matters at all. It tests only the mutual
registration of the node field, the patient axis and the electrodes. Writing the null result
as "patient spatial structure is unnecessary" is forbidden.**

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

New worktree only; the dirty main worktree is untouched. Long runs go through
`systemd-run --user` + `nohup`. Numeric threads pinned to 1 per worker. Worker count derived
from a measured memory sentinel, capped at 8, always leaving ≥ 32 GiB available (prior peak
RSS was 14.6 GiB per 20 s worker; the machine has 251 GiB total, 229 GiB available, 80 cores).
The monitor polls every 600 s for memory, disk, worker state and module hashes — no continuous
polling. OOM, non-finite values, hash drift or a checkpoint-replay mismatch stops new work;
processes belonging to other worktrees are never killed.

Storage: per-probe output is the aggregated per-neuron excess-spike field (32000 float32,
≈128 KB) plus scalar metrics — ~51 MB over ~400 probes. Raw per-neuron traces are kept for at
most 6 exemplars (~64 MB each). Checkpoints ~130 MB × 36 ≈ 4.7 GB. Total under 6 GB against
187 GiB free.

Estimated wall clock at 8 workers: gates ~1 h, Phase 1 ~0.5 h, Phase 2 ~2 h, Phase 3 ~5 h,
substrate null ~3.5 h, figures ~0.5 h — about 12.5 h unattended.

On completion: `DONE.json`, a desktop notification, a scientific report, the figure README,
provenance, and scoped commits. The report's first paragraph answers, in order: can this
continue, what is safe to conclude now, what is the largest gap, what is next.
