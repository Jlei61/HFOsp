# Topic 4 rev4: data-driven interictal-to-model-ictal discovery audit and Fig.3 bridge

Date: 2026-08-20
Status: draft for collaborator review; no new simulation is authorized by this document
Branch: `codex/topic4-data-driven-zm-ictal-transition`
Supersedes: the unexecuted rev3 Fig.5 morphology-selection and Fig.3-bridge draft. It does not
rewrite historical execution contracts or results.

## 1. Motivation and scientific boundary

Fig.4 established a development-stage E1146 SNN substrate learned from interictal events: a
continuous node-excitability field plus data-driven local E-to-E and E-to-I redistribution.
Fig.5 asks whether activating the per-neuron slow variables Z and M on this substrate can produce
a model ictal state and, more importantly, whether the model state reuses the interictal
propagation structure before it is compared with the patient's ictal data.

Three questions must remain separate:

1. **Model state:** did the SNN enter sustained, broad, high-intensity and faster activity?
2. **Model-internal cross-state relation:** did that state retain and reuse the learned
   interictal repertoire beyond matched static-scaffold nulls?
3. **Clinical bridge:** after the work point is frozen without ictal supervision, how closely
   does its contact readout resemble the patient's Fig.3 energy, spatial and temporal target?

A trajectory may pass the first question and fail the second. It may pass both model-internal
questions and still fail the clinical bridge. These outcomes are scientifically distinct.

The current E1146 branch is not blind. The clinical target, candidate morphology and the 5%
E-to-I trajectory have already been viewed. This revision can therefore produce only a
`DEVELOPMENT_ONLY_RETROSPECTIVE_DISCOVERY_AUDIT`. Rewriting the selection rule cannot erase that
analyst exposure. A future claim of prospective discovery requires a patient or sealed seizure
unit not opened before work-point freeze.

## 2. Evidence architecture

### Layer 1: model-ictal qualification

`MODEL_ICTAL_ELIGIBLE_V2` asks only whether the model entered the intended dynamical endpoint.
It does not use any patient ictal quantity.

### Layer 2: model-internal cross-state discovery

`CROSS_STATE_DISCOVERY_ELIGIBLE` requires:

```text
MODEL_ICTAL_ELIGIBLE_V2
AND INTERICTAL_REPERTOIRE_RETAINED
AND MOTIF_REUSE_ABOVE_MATCHED_NULL
```

This is a claim gate, not a run blocker. Failure still permits a model-ictal figure, but not an
interictal-to-ictal discovery claim.

Counterfactual motif perturbation is an additional requirement for a model-internal mechanism
claim. It is not required merely to show a cross-state association.

### Layer 3: post-freeze clinical bridge

The Fig.3 energy, spatial and temporal endpoints are opened only after a work-point manifest is
frozen using Layers 1-2 and model robustness. They are evaluation endpoints, not candidate
selection variables.

## 3. Permitted and forbidden claims

This round may say:

> The interictal-data-constrained SNN enters a model ictal state. We then test whether its
> recruitment reuses the model's frozen interictal repertoire and, after freezing the work
> point, compare it retrospectively with the patient's Fig.3 ictal organization.

It may not claim that:

- model current is clinical SEEG voltage;
- model Hz equals patient Hz;
- Z or M is the identified biological mechanism in E1146;
- the model reproduces seizure termination or a complete clinical seizure lifecycle;
- E1146 is blind validation;
- high absolute spatial correlation proves an ictal transition;
- a representative seed establishes network-ensemble reproducibility;
- a model-internal motif perturbation proves patient biological causality.

## 4. Frozen substrate and experimental arms

Node positions, the continuous field `h`, topology, delays, pathway budgets, montage and learned
E-to-E/E-to-I coefficient bases remain frozen. The dosing implementation is:

```text
effective coefficient row = dose x frozen learned coefficient row
```

It scales the learned redistribution coefficients. It does not add edges or change the
target-wise incoming pathway budget.

Two arms must be named separately.

### Exact Fig.4 carry-over

The substrate, `I_th_EI`, E-to-E dose and E-to-I dose equal the accepted Fig.4 values by hash;
only Z/M feedback is activated. This is the only arm that tests "Fig.4 substrate + Z/M".

### Calibrated transition

This arm may alter slow-variable parameters, `I_th_EI` or pathway-expression doses. The current
visual candidate belongs here:

```text
I_th_EI       = 0.8 x 95.19851312666987
tau_z         = 5000 ms
tau_adp       = 500 ms
eta_m         = 0.007451594355587098
E_to_E_dose   = 1.0
E_to_I_dose   = 0.05
```

If exact carry-over fails but the calibrated arm passes, the allowed conclusion is that the
learned substrate required development-stage calibration to access the model ictal state.

## 5. Model-ictal qualification

### 5.1 Time landmarks

- `t_base`: `[500,1000] ms`, previously validated as low activity against same-seed passive-Z/M
  references;
- `t_op`: unchanged engine detector time, 20 ms causal EMA E-rate at least 120 Hz for 100 ms;
- `t_ictal = t_op - 100 ms`;
- `W_pre = [t_ictal-500,t_ictal)`;
- `W_early = [t_ictal+100,t_ictal+1100)`;
- `W_freq = [t_ictal+500,t_ictal+1000)`.

The candidate is not evaluable if the complete `W_early` interval is unavailable.

### 5.2 Primary qualification

Let `F_E(t)` be the fraction of E neurons active in 20 ms. Let `F_sheet(t)` be the fraction of
occupied 1 mm bins in which at least half of local E neurons are active. Primary bins contain at
least 20 E neurons and are equal-area weighted. Let `R_E(t)` be population E rate and
`f_contact` the median contact spectral centroid in the model-internal `10-250 Hz` band.

`MODEL_ICTAL_ELIGIBLE_V2` requires all of:

1. the unchanged operational detector is reached;
2. at least 80% of `W_early` windows satisfy `F_E >= 0.50` and `F_sheet >= 0.50`;
3. median `R_E(W_early) / R_E(t_base) >= 2.0`;
4. `f_contact(W_freq)-f_contact(t_base) >= max(5 Hz, spectral resolution)` and the ratio is
   at least `1.25`;
5. no non-finite state or simulator error occurs through `W_early`.

Recovery is not required. A sustained non-returning state is the model endpoint, but is called
`model ictal`, not a clinical seizure.

### 5.3 Qualification sensitivity

The 80% and frequency clauses describe the requested Fig.5 morphology and are development-stage,
not universal seizure criteria. Every candidate table also reports:

- broad-recruitment duty at `70/80/90%`;
- `F_E` and `F_sheet` thresholds `0.4/0.5/0.6`;
- bin sizes `0.5/1/2 mm` and minimum occupancy `10/20/40` E neurons;
- spectral centroid, peak frequency and broad-band rate-envelope diagnostics;
- onset shifts of `-100/0/+100 ms`.

The primary verdict remains fixed at the values above. Sensitivities reveal boundary dependence;
they do not provide a route to choose whichever definition makes a candidate pass.

## 6. Interictal repertoire retention

Reuse the already frozen Fig.4 direction classifier, OOD rule and reference distribution without
refitting. The historical `INTERICTAL_REPERTOIRE_RETAINED` contract remains the primary rule:

```text
at least 20 returned events before model-ictal onset
OOD fraction <= reference q95
at least 3 events in each frozen mode
natural-KMeans balanced alignment >= reference q05
```

The report must additionally include all returned-event distributions:

- event count and rate;
- frozen A/B counts and equal-event mode proportions;
- OOD fraction and classifier confidence;
- rank-profile similarity by mode;
- recruitment size, shaft participation and spatial range.

This gate is required for `CROSS_STATE_DISCOVERY_ELIGIBLE`, not for
`MODEL_ICTAL_ELIGIBLE_V2`. The runaway state itself is never assigned an A/B label.

## 7. Model-internal motif reuse

### 7.1 No subjective event selection

All returned events satisfying the frozen detector and classifier rules enter the analysis.
The displayed last qualifying event is fixed algorithmically by time. The word `clean` is not a
selection criterion. Supplementary output shows the full event distribution.

### 7.2 Primary reuse readouts

For each qualifying pre-transition event and the model early-ictal interval, calculate:

1. **rank reuse:** Spearman agreement between event contact-onset ranks and early-ictal contact
   first-passage ranks, using common recruited contacts and reporting coverage;
2. **precedence reuse:** weighted agreement between the event/frozen-mode contact-pair precedence
   matrix and early-ictal first-passage precedence;
3. **edge-flow reuse:** cosine similarity between model interictal recurrent E edge flow and
   early-ictal recurrent E edge flow, with pathway and delay strata retained;
4. **trajectory:** the above scores for every qualifying event versus time-to-transition, not
   only the final event.

Primary aggregation gives each network equal weight. Events are resampled inside network only
for within-network uncertainty.

### 7.3 Matched nulls

`MOTIF_REUSE_ABOVE_MATCHED_NULL` compares observed reuse against predeclared model-internal
nulls that preserve information unrelated to motif identity:

- within-shaft contact-label permutation preserving recruitment count;
- onset-time circular shift preserving event rate and transition time;
- learned-edge gain permutation within pathway x delay x distance bins, followed by exact
  target-wise budget renormalization;
- matched off-motif node sets preserving `h`, local E density, shaft, distance to montage and
  baseline rate.

The edge null is invalid unless topology, delays, incoming pathway budgets, edge-distance
distribution, source/target degree summaries and gain distribution pass a structural audit.
No SNN is run for that null until the audit passes on synthetic and frozen graphs.

The gate is exploratory: the observed network-level reuse statistic must exceed the q95 of its
matched null for rank/precedence and edge-flow families, with exact coverage reported. No patient
ictal target is read in this calculation.

### 7.4 Counterfactual dependence

For a model-internal mechanism claim, branch from the same pre-transition checkpoint and noise
state:

- motif-specific learned-edge attenuation;
- equal-budget matched random-edge attenuation;
- sham continuation.

Compare transition latency, model-ictal probability within the fixed horizon, onset location and
early-ictal field. A stronger motif-specific effect supports model-internal dependence. Without
this experiment, the result remains structural reuse, not mechanism.

## 8. Work-point selection without ictal supervision

Before `WORKPOINT_FROZEN.json` exists, allowed inputs are:

- Fig.4 interictal train/reference artifacts and frozen classifier;
- model-internal qualification, repertoire, reuse and safety metrics;
- parameter-neighbourhood and seed robustness;
- distance from exact Fig.4 parameters.

Forbidden inputs are patient ictal contact vectors, Fig.3 bridge scores, seizure-2 appearance and
aggregate clinical early-ictal fields.

Existing candidates are first rescored without simulation. Up to three model-internally eligible
candidates enter a fixed three-seed replication, chosen before those replications run. The
development figure work point is selected lexicographically:

1. highest model-ictal eligible proportion;
2. pass repertoire retention and matched-null motif reuse;
3. highest lower network-bootstrap bound of motif reuse;
4. smallest log-parameter distance from exact Fig.4 carry-over.

If no candidate passes Layer 2, the figure may use an eligible `MODEL_ICTAL_ONLY` candidate but
must not claim cross-state discovery. Clinical bridge metrics cannot break a tie.

Three seeds support a development figure and work-point freeze, not a population inference. Any
paper-level cross-state or pathway claim uses a predeclared 12-network set. Whether to run 12 is
decided from the intended claim, never from the first three effect directions.

## 9. Clinical Fig.3 bridge after freeze

### 9.1 Patient target

The locked patient feature is exact-name `1-150 Hz` log power, per-contact robust-z against
`[-120,-90] s`, with early ictal `[0,10] s`. The 24 eligible E1146 seizures define the aggregate;
seizure 2 is display-only. Its locked values `shared_a_signed=0.719127` and direct early-rank
correlation `0.570884` are parity checks, not targets for model selection.

The E1146 field is already strong pre-onset, approximately `0.7187`, and rises only to about
`0.7733` early ictally. Absolute spatial agreement therefore measures persistent scaffold as
much as transition organization.

### 9.2 Model contact feature and spectral estimation

The model primary bridge band is `10-150 Hz`; the signed `30-80 Hz` trace remains a display
proxy. Quantitative power uses at least 500 ms windows with multitaper or Welch estimation. A
100 ms window is not used to quantify 10 Hz power. Adjacent overlapping windows are never
treated as independent bootstrap units.

Primary uncertainty comes from fixed independent network/noise seeds. Time uncertainty uses
non-overlapping blocks within a trajectory. Required band sensitivities are patient `10-150 Hz`
and model `1-150 Hz` with detrending and a sufficiently long window. The report states whether
the spatial result depends on patient `1-10 Hz` power.

### 9.3 Three continuous domains

**Energy** reports global burden, positive-contact coverage and contact IQR, each compared with
the 24-seizure distribution.

**Absolute spatial organization** uses direct model-versus-patient early-contact Spearman. For a
comparable patient reference, each patient seizure is correlated with the template formed from
the other 23 seizures. The model is correlated with the all-24 development template and located
relative to that leave-one-seizure-out distribution. Patient split-half agreement is reported as
a reliability ceiling, not treated as the same statistic.

**Incremental spatial organization** is fully specified. For contact `i`:

```text
delta_M_i = (P_M_i(early) - P_M_i(pre)) - median_j(P_M_j(early) - P_M_j(pre))
delta_P_i = (P_P_i(early) - P_P_i(pre)) - median_j(P_P_j(early) - P_P_j(pre))
scale_i   = max(patient bootstrap IQR(delta_P_i), split-half MAD floor_i, epsilon)
D_increment = mean_i abs(delta_M_i - delta_P_i) / scale_i
```

No second scalar normalization is applied. Increment Spearman and cosine direction are reported
diagnostics. If the patient increment norm is near its split-half noise floor, direction is
marked not evaluable and only the scaled amplitude error is interpreted.

**Time** uses baseline, pre-transition and early landmarks and the four increments in global
energy and signed TA similarity. Exact seconds are not matched.

The full vector `D_energy`, `S_absolute`, `D_increment` and `D_time` is always shown. A descriptive
post-freeze summary may be:

```text
J_bridge = mean(D_energy, D_spatial, D_time)
           + LSE_0.25(D_energy, D_spatial, D_time)
```

`J_bridge` is never used to select or retune the discovery work point.

### 9.4 Retrospective target-informed sensitivity

If collaborators still want the historical question "which completed candidate best matches
E1146 ictal data?", it is a separately labelled `TARGET_INFORMED_SENSITIVITY`. The complete
candidate-selection procedure is repeated under matched patient-target surrogates to obtain a
selection-aware null for the minimum `J_bridge`. That result cannot be used as discovery evidence
or alter the frozen Fig.5 work point.

## 10. Zero-simulation controls

Before new SNN runs:

1. static-axis amplitude-only control must retain absolute correlation but fail increment/time;
2. uniform-energy control must improve energy but not spatial increment;
3. spatial-only control must improve spatial distance but not energy;
4. repeated returned bursts must fail model-ictal qualification;
5. SCL/contact-label censoring must be visible to the frozen repertoire metrics;
6. all motif nulls must preserve their declared nuisance structure and destroy motif identity.

## 11. Fig.5 contract

The main figure uses one uninterrupted trajectory and the accepted visual grammar.

### Panel A: continuous model readout

- exact 15-contact, two-shaft order;
- signed `30-80 Hz` model-current traces;
- aligned `F_E`, `F_sheet` and population-rate strip;
- at least 1 s of eligible broad high state after `t_ictal`;
- one pre-transition amplitude scale and no post-onset per-contact renormalization.

### Panel B: projected Z/M trajectory

Show `h`-weighted disinhibition `1-z` versus `h`-weighted adaptation `eta_m m`. Label it
`projected Z/M trajectory`, not a phase portrait unless a vector field is estimated.

### Panel C: interictal motif and early model-ictal field

- left: the algorithmically last qualifying returned event;
- right: early model-ictal `10-150 Hz` contact robust-z field;
- identical registered plane, electrode direction and contact labels;
- display the model-internal motif-reuse statistic; supplementary output shows all qualifying
  events and the time-to-transition distribution.

### Panel D: motif-conditioned susceptibility

The failed linear-response endpoint remains closed. Use the nonlinear packet threshold:

```text
n_crit(site,state) = smallest packet producing probe-attributable broad recruitment
susceptibility = 1 / n_crit
```

Primary sites are frozen from interictal training only: motif source/high-flow sites and matched
off-motif controls matched on `h`, E density, shaft, montage distance and baseline rate. Packet
ladder is `64,80,96,112,128` E neurons.

Directly injected neurons and the injection-source bin are excluded from broad-recruitment
qualification. Dose response must be monotone before a scalar `n_crit` is reported; otherwise the
full ladder is shown. Thresholds below 64 or above 128 are interval-censored. A sham transition is
a competing event: report sham latency and paired excess latency/probability, not ordinary
missingness.

The six-site canary is an engineering identifiability check. A 7x7 map is optional and cannot
replace the predeclared motif-versus-matched-control comparison.

## 12. Connectivity and counterfactual controls

After work-point freeze, compare:

```text
Exact Fig.4 carry-over + Z/M
Calibrated Node
Calibrated Node + generic distance/budget-matched connectivity
Calibrated Node + shuffled learned connectivity
Calibrated Node + learned E-to-E/E-to-I connectivity
```

Then run the paired `Node`, `Node+EE`, `Node+EtoI`, `Joint` factorial only if pathway inference is
an intended claim. Network seed is the independent unit. Report paired main effects and

```text
I_Y = Y(Joint) - Y(Node+EE) - Y(Node+EtoI) + Y(Node)
```

The scientific comparison is whether learned interictal structure outperforms matched generic or
shuffled structure, not merely whether adding an edge dose advances transition time.

## 13. Result interpretation

| Result | Permitted interpretation |
|---|---|
| model ictal fails | rate crossing is not an accepted Fig.5 model seizure |
| model ictal passes, repertoire fails | model high state exists; no interictal continuity evidence |
| repertoire passes, reuse not above null | interictal and ictal-like states coexist; structural reuse is unresolved |
| reuse passes, clinical bridge poor | model-internal cross-state reuse without patient ictal prediction |
| absolute field high, increment/time poor | persistent static scaffold expression |
| reuse and post-freeze bridge both strong | development-stage cross-state discovery candidate |
| motif counterfactual exceeds matched ablation | model-internal motif dependence candidate |
| prospective unseen unit replicates | evidence can begin to support generalizable cross-state prediction |

## 14. Required artifacts

```text
results/topic4_sef_hfo/data_driven_zm_ictal_transition/discovery_audit_v1/
  discovery_boundary.json
  exact_carryover_audit.json
  model_internal_candidate_rescore.csv
  workpoint_replication.json
  WORKPOINT_FROZEN.json
  repertoire_retention.json
  motif_reuse.json
  motif_null_audit.json
  clinical_bridge_postfreeze.json
  provenance.json

results/paper-ready-figure/fig5/figures/
  fig5-data-driven-zm-main.{png,pdf,svg}
  fig5-data-driven-zm-main-metadata.json
  README.md
```

Every artifact records source hashes, contact order, windows, bands, work-point parameters and
the `DEVELOPMENT_ONLY_RETROSPECTIVE_DISCOVERY_AUDIT` boundary.

## 15. Collaborator decisions requested

1. accept the three-layer evidence structure and forbid clinical ictal scores before work-point
   freeze;
2. accept exact Fig.4 carry-over versus calibrated transition as distinct arms;
3. accept repertoire retention plus matched-null motif reuse as the cross-state claim gate;
4. accept 500 ms spectral estimation and seed-level uncertainty for the Fig.3 bridge;
5. accept Panel D as motif-versus-matched-control susceptibility rather than a generic spatial
   heat map;
6. decide whether the paper needs a 12-network cross-state/pathway claim or only a representative
   development figure.
