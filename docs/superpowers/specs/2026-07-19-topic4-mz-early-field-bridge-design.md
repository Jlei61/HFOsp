# Topic 4 MZ early-field bridge — design specification

Date: 2026-07-19

Status: overnight execution lock

Branch: `codex/topic4-mz-slowvars`
Primary question: can the spatial order of stable interictal-like events on the fixed E1146 scaffold predict the early energy field of an MZ disinhibition-driven operational runaway?

## 1. Decision

The next experiment is a **direct MZ field bridge**, not another slow-variable search.

The mandatory chain is:

```text
same-seed slow-off returning events
    -> held-out interictal timing templates on the fixed scaffold
    -> z-only delayed operational runaway
    -> onset-locked early activation/energy field
    -> template-to-field association + spatial null + three-seed consistency
```

Do not add a new global denominator, a new slow variable, a broad `z+m` scan, or a full state-conditioned spectral atlas before this chain is measured. The existing global-versus-local question is first treated as a **readout/ablation question**, not as a new term in the dynamics.

## 2. What is already done

### 2.1 MZ trajectory layer

The MZ branch already has:

- a fixed E1146 two-dimensional scaffold and 15-contact virtual-SEEG montage;
- same-seed slow-off runs with 38--40 returning events at 15 s;
- a robust delayed-runaway z-only candidate, `zA_q75_tz5000`, in seeds 1/3/4 with operational onset near 9.29/9.50/9.76 s;
- a faster z-only runaway sensitivity, `zA_q50_tz10000`, in seeds 1/3/4 with onset near 4.71--4.94 s;
- continuous population rate, active fraction, per-neuron spike raster during a run, and an LFP recorder;
- a readout-bundle writer for the accepted contact montage.

These facts establish a usable trajectory family. They do not yet establish an interictal-to-early-field relationship.

### 2.2 Existing Figure 5/readout layer

The separate read-only worktree `codex/topic4-early-readout` already contains:

- model-agnostic arrival/early-energy metrics;
- signed Spearman, field cosine, top-k overlap, all-contact and within-shaft permutation nulls;
- a Figure 5 plotting grammar using one continuous virtual-SEEG trajectory plus matched timing and energy fields;
- a single-seed M3 `q_I` Figure 5 candidate with 15-contact descriptive `Spearman=0.814`.

That candidate is useful infrastructure and a visual prototype, but it is not the MZ validation: it is one M3 pulse-driven trajectory, one displayed event, and its four SCL contacts have readable contact signals while 0/4 pass the local-neural-tissue recruitment gate.

The MZ implementation should selectively reuse the generic statistics and plotting contract. It must not overwrite the existing Figure 5 candidate or import unrelated dirty manuscript edits from the early-readout worktree.

### 2.3 M3B mechanism layer

M3B already supports a weaker, correctly bounded statement: the dominant eigenmode is mostly global, while axial structure appears in a non-normal finite-time response. It does not yet provide the projected optimal propagator

```text
K_T = P_rE exp(J T) E_rE
```

or an MZ spatial slow-state adapter. Therefore a full spectral interpretation is a later mechanism overlay, not tonight's primary bridge.

## 3. Known-invalid artifact quarantine

The original Arm C `z+m` discovery labels must not be used as a dose-response or interaction result:

- the nominal nine Arm C labels collapse to two unique z configurations;
- all three nominal m levels select the same m configuration;
- `weak/mid/strong`, `9/9`, or a graded z-by-m interaction is therefore not measured.

The current safe Arm C statement is only that one strong adaptation setting suppressed two z-only runaway configurations in seed 1. No part of this spec consumes Arm C as evidence. Repairing Arm C is a later experiment.

## 4. Scientific hypotheses

### H1 — stable interictal template

Returning slow-off events form at least one reproducible direction-conditioned timing field on the fixed scaffold. Reproducibility must be evaluated on held-out events, not on the events used to build the template.

### H2 — early-field bridge

For the z-only delayed-runaway trajectory, contacts or source-grid locations that are earlier in an eligible interictal template have higher activation/energy in a fixed early transition window.

Because the scaffold supports two directions, the primary direction-free statistic is a predeclared `maxAB` comparison whose spatial null repeats the same maximum operation. A source-anchored signed statistic is secondary and must be reported separately.

### H3 — state changes gain more than geometry

If H1 and H2 are supported, the bounded mechanistic interpretation is that the fixed scaffold supplies a reusable spatial response pattern while z-mediated loss of inhibitory efficacy increases recruitment gain. This is not yet proof that individual interictal events causally deplete z enough to trigger the transition.

## 5. Frozen candidates and seeds

Primary:

```text
candidate: zA_q75_tz5000
seeds: 1,3,4
T: 15000 ms maximum
slow variables: z on, m off
reason: longest robust pre-transition epoch and more detected event opportunities
```

Sensitivity, only after all primary outputs exist:

```text
candidate: zA_q50_tz10000
seeds: 1,3,4
reason: faster, stronger-depletion replication boundary
```

Do not promote `zA_q75_tz10000` to the primary bridge candidate: its expanded-bounded phenotype is present only in seed 1.

## 6. Fixed event detector

The current MZ runner recalculates its event threshold from every run's own maximum active fraction. That is invalid for cross-state comparison because the target runaway changes the threshold used to define its own events.

For each seed:

1. Run slow-off once with the accepted configuration and LFP recorder.
2. Compute the slow-off event bar exactly once using the existing slow-off rule:

   ```text
   floor_seed = P95(active_fraction in the registered baseline interval)
   event_bar_seed = floor_seed + CAL_FRAC * (max(active_fraction_slowoff) - floor_seed)
   ```

3. Save `floor_seed`, `event_bar_seed`, `CAL_FRAC`, bin width, baseline interval, and slow-off artifact hash.
4. Use that same `event_bar_seed` for slow-off, native z-only, and every later same-seed counterfactual.
5. Never recalibrate the bar from the z-only target run.

The existing phenotype labels remain historical outputs. The bridge has its own fixed-bar event inventory and does not silently rewrite the phenotype table.

## 7. Interictal timing templates

### 7.1 Primary source of events

The primary template uses **same-seed slow-off returning events**. This guarantees enough interictal support and tests the intended `same scaffold, different state` statement without requiring the z-only trajectory itself to contain many pre-runaway events.

Pre-runaway returning events from the continuous z-only trajectory are a secondary within-trajectory check. If a seed has too few eligible pre-runaway events, record `insufficient_support`; do not replace the fixed detector or choose another event after viewing the target energy field.

### 7.2 Per-event timing fields

Reuse the early-readout branch's 30--80 Hz burst-envelope peak-latency logic where possible.

For each fixed-bar returning event:

- use a window from event onset through `event offset + 40 ms`, capped before the next event and record end;
- subtract the per-contact slow-off quiet-envelope median;
- call a contact readable when its event peak exceeds its quiet median by `5 * MAD` and is at least 10% of the largest contact excess peak in that event;
- require at least six readable contacts;
- convert finite peak latencies to ordinal ranks; missing contacts remain missing and are never imputed;
- compute the Spearman association between contact axis coordinate and latency rank.

Direction labels are fixed by the registered model axis:

```text
A_to_B: axis-latency Spearman >= +0.30
B_to_A: axis-latency Spearman <= -0.30
unresolved: otherwise
```

The exact sign-to-endpoint mapping must be written into artifact metadata from `src_xy`, `snk_xy`, and `axis_unit`; it may not be inferred from the plotted field.

### 7.3 Train/held-out contract

Within each direction and seed, split eligible events chronologically by odd/even index:

- training template: per-contact median ordinal rank over training events;
- held-out score: Spearman between the frozen training template and every held-out event on matched finite support;
- eligible direction: at least three training and two held-out events, at least six shared contacts, and non-degenerate template variance.

Report event counts, held-out median/range, and sign consistency. Do not tune the direction threshold or split after seeing early-field associations.

Source-grid timing fields use the identical events and split. Bin E-neuron first-spike latency onto the fixed 24x24 grid; require a minimum of five active E neurons per occupied bin. Source-space is a projection-control layer, not a replacement for the contact result.

## 8. Transition onset and early fields

### 8.1 Two onset markers

Keep the existing operational runaway onset `t120`: the first 100-ms interval with at least 80% of 20-ms-smoothed E-rate samples above 120 Hz.

Add a baseline-relative recruitment onset without fitting the target field:

```text
r20_slowoff = 20-ms-smoothed slow-off E rate
theta_recruit_seed = P99.9(r20_slowoff)
r20_target = 20-ms-smoothed native z-only E rate
t_recruit = start of the contiguous r20_target > theta_recruit_seed component that contains t120
```

Allow at most 5 ms subthreshold gaps when defining the component. If `t120` or that component is absent, the seed is `onset_unresolved` and no early-field claim is made. Save both onsets and their difference.

This onset rule is fixed by slow-off and avoids defining early recruitment from the target run's own maximum. `t120` remains the reproducibility anchor; `t_recruit` is the primary field-locking time.

### 8.2 Fixed windows

Primary:

```text
early_0_50_ms relative to t_recruit
```

Registered sensitivities:

```text
early_0_100_ms
early_0_25_ms
early_25_50_ms
early_50_100_ms
```

If the trace ends before a complete window, mark that window ineligible. Never score a truncated window. A 100--300 ms window may be emitted only when naturally complete; it is not required overnight.

### 8.3 Contact energy field

Use the existing Figure 5 signal contract:

- filter virtual LFP to 30--80 Hz;
- take its envelope;
- define the per-contact baseline from slow-off quiet samples, never from runaway;
- form positive excess envelope;
- compute mean squared positive excess within each fixed early window.

Call this `virtual-LFP early-energy proxy`, not clinical broadband power.

### 8.4 Source-grid activation field

From the same E-neuron spike raster, bin neurons onto the fixed 24x24 grid and compute per-bin E firing rate in each early window. Subtract the corresponding slow-off quiet mean, clamp only the excess at zero, and square-average over time.

Source-grid and contact fields must share geometry provenance but remain separate observation levels.

### 8.5 Eligibility and dynamic range

Each field records:

- finite support size;
- standard deviation and robust `(P90-P10)/(abs(median)+eps)` dynamic range;
- recruited contact count or source-grid area above the slow-off P95 field value;
- whether numerical/trace truncation occurred.

A constant or near-constant field is `degenerate_field`, not a zero association.

## 9. Association and nulls

For each seed, space, and window, compute:

- `earliness_energy_spearman = corr(-template_rank, early_energy)`;
- standardized field cosine;
- earliest-quartile minus latest-quartile energy, normalized by the field IQR;
- top-k earliest/hottest overlap;
- support and dynamic-range diagnostics.

Compute every association on both `all eligible support` and a preregistered `direct-core-excluded`
support. In source space, exclude bins intersecting either low-threshold core. In contact space, reuse
the geometric/Gaussian core-loading definition from the existing early-readout adapter and save the
threshold in metadata. If exclusion leaves fewer than six contacts or six source bins, report
`insufficient_core_excluded_support`; do not fall back to the all-support result.

Separately audit local tissue participation around every virtual contact: the fraction of E neurons
within 1.5 mm that fire in the selected interictal event and early target window. A readable contact
signal must never be described as direct recruitment of the local tissue when this audit is negative.

For the two direction-conditioned templates:

```text
rho_A = association(A_to_B template, energy)
rho_B = association(B_to_A template, energy)
rho_maxAB = max(rho_A, rho_B)
```

`rho_maxAB` is eligible only when both registered direction templates are eligible. If only one
direction survives held-out validation, report its signed association as secondary and mark the
primary `maxAB` result unresolved.

The primary contact null permutes the target energy labels within shaft and recomputes `rho_A`, `rho_B`, and `rho_maxAB` on every permutation. Also report unrestricted-contact shuffle. Enumerate exactly when the constrained space is at most 50,000; otherwise use 10,000 fixed-seed permutations.

The source-grid null uses non-zero toroidal translations of the energy field, recomputing `maxAB`; this preserves more spatial autocorrelation than independent cell shuffling. Report the number of unique shifts.

Never report the better direction's ordinary one-template p-value as the `maxAB` p-value.

Across three seeds, report the median and range of every effect plus the number with the expected sign. With `n=3`, do not present a cohort significance claim.

## 10. Global versus local depletion

Do not add a global/local ratio to the model equation tonight.

If z snapshots are available, decompose the depletion field `d_i = 1-z_i` descriptively:

```text
global_fraction = N * mean(d)^2 / sum(d_i^2)
local_fraction  = sum((d_i-mean(d))^2) / sum(d_i^2)
```

The two fractions sum to one up to numerical error. Save them at `t_recruit-100 ms`, `t_recruit`, and `t120`. This measures whether depletion is mostly a uniform gain shift or a patterned local field; it does not feed back into the dynamics and is not evidence by itself that local pattern causes the early gradient.

## 11. Counterfactual hierarchy

### 11.1 What is feasible immediately

The native z-only run and matched slow-off run are common-random-number replays from `t=0`. They are not exact state forks. They can establish association and a broad z-necessity boundary, but not isolate a local pre-transition z pattern.

### 11.2 Required before a causal slow-field claim

A later engine task must snapshot and restore the full fast state, delay rings, slow state, recorder state, and RNG state, then prove bit-identical continuation before branching. Only after that audit may the following be called state-matched counterfactuals:

- native local z;
- spatially uniform mean-yoked z;
- spatially shuffled z with the same histogram;
- z reset to one;
- event-deletion/reset intervention to test whether interictal events drive, rather than merely reveal, the transition.

Do not attempt this invasive snapshot/resume change as a required overnight deliverable.

## 12. Later mechanism overlay

Only if the direct bridge is numerically eligible should MZ spatial snapshots be mapped into the coarse M3B rate field. The mapping is initially a `sign-preserving surrogate`, not a calibrated biophysical identity.

The correct finite-time object for the E-rate readout is:

```text
K_T = P_rE exp(J T) E_rE
```

Compute its singular values and input/output singular vectors on a frozen, fully specified operator. If a probe dictionary is used, orthonormalize it with rank-revealing QR/SVD before computing subspace gain. Do not call an SVD of responses to overlapping, non-orthogonal Gabor columns an optimal gain.

The desired later result is not “the leading eigenmode becomes axial.” It is: finite-time E-rate gain rises with depletion while the optimal/output spatial pattern remains concordant with the interictal scaffold. Existing M3B controls remain relevant, but this overlay is Phase B.

## 13. Overnight output contract

Output root:

```text
results/topic4_sef_hfo/mz_early_field_bridge/
```

Required artifacts:

```text
config_snapshot.yaml
provenance.json
per_seed/seed1/{slowoff,native,templates,bridge_metrics}.{npz,json}
per_seed/seed3/{slowoff,native,templates,bridge_metrics}.{npz,json}
per_seed/seed4/{slowoff,native,templates,bridge_metrics}.{npz,json}
cohort_summary.json
cohort_summary.csv
figures/mz_early_field_bridge_multiseed.png
figures/mz_early_field_bridge_seed1.png
figures/README.md
STATUS.md
docs/archive/topic4/sef_hfo/mz_early_field_bridge_2026-07-19.md
```

The per-seed Figure 5-style diagnostic reuses the existing field grammar but must preserve trajectory
provenance. If an eligible interictal event comes from the same native z-only trajectory, show the
single continuous trace and its exact event window. If the primary timing field comes from matched
slow-off, show slow-off and native traces as two explicitly labelled strips; never shade a slow-off
event on the native trajectory or imply that they are one recording. It must be labelled
`operational runaway` and `MZ diagnostic`; it must not overwrite the current canonical Figure 5
candidate.

The multiseed figure should prioritize scientific diagnosis over paper aesthetics:

- template held-out reproducibility;
- early timing-versus-energy association by seed and window;
- source/contact agreement;
- null distributions and support/dynamic range;
- unresolved states displayed explicitly.

Every generated figure must be visually inspected, and `figures/README.md` must be written in Chinese after inspection using the repository format.

## 14. Completion levels

Report four levels separately:

1. **engineering complete** — fixed-bar detector, reusable readout, tests, resumable artifacts;
2. **numerically eligible** — held-out template and complete non-degenerate early field exist;
3. **scientific observation** — direction, effect sizes, nulls, and seed consistency are reported regardless of sign;
4. **bridge supported** — at least two of three seeds have eligible held-out templates and positive contact `maxAB`, with source-space direction not contradictory and no result depending only on direct-core loading.

Level 4 is an overnight diagnostic criterion, not a cohort statistical proof and not seizure validation.

## 15. Claim boundaries

Allowed if supported:

- a fixed patient-specific scaffold expresses reproducible interictal timing fields;
- z-mediated loss of inhibitory efficacy moves the same model into an operational runaway whose early energy field is concordant with one registered interictal direction;
- this is a model-side feasibility bridge for `same scaffold, different state`.

Forbidden:

- calling operational runaway a clinical seizure;
- claiming termination, recovery, or a complete seizure cycle;
- claiming `z_i` is the unique biological seizure mechanism;
- claiming interictal events causally trigger the transition without event-deletion/reset;
- claiming local z pattern is causal without state-matched snapshot/resume counterfactuals;
- using the invalid Arm C nominal grid as a dose-response;
- interpreting virtual-LFP energy as measured clinical broadband SEEG power;
- choosing a direction, window, candidate, or seed because it gives the strongest correlation.

## 16. Stop rules

- If slow-off has fewer than six eligible returning events in all seeds under the fixed bar, stop target runs and report `template_input_blocked`.
- If a target replay does not reproduce the locked runaway phenotype/onset within 20 ms, inspect observer drift before continuing.
- If all seeds lack a complete 0--50 ms window, stop and report `readout_window_blocked`.
- If a result is negative or directionally opposite, complete the registered metrics/nulls and report it; do not retune.
- Stop launching expensive work at hour 6.5 and reserve the final 90 minutes for figures, visual QA, tests, provenance, and the report.

## 17. Next experiment after this spec

The next decision is conditional:

- bridge positive and stable -> implement exact state/RNG snapshot-resume and native/uniform/shuffle/reset z counterfactuals, then the projected-propagator overlay;
- bridge eligible but negative -> the current MZ loss-of-inhibition transition does not explain the empirical interictal-to-early-energy relation; do not add termination variables to rescue it;
- bridge blocked by sparse timing events -> improve observation/event sampling only, without changing dynamics;
- only after the bridge question is settled -> repair the Arm C selector/calibration and test whether graded m can bound runaway into a re-triggerable recruited state.
