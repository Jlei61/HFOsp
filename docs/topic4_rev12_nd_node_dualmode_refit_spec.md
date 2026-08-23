# Topic 4 rev12-ND: data-driven Node dual-mode identifiability and refit

## 1. Motivation and current evidence

Figure 4 currently uses a continuous patient-constrained Node excitability field
on a frozen SNN scaffold.  The field is not a set of hand-placed cores: it is a
continuous tensor B-spline surface projected to a fixed total excitability mass.

The current substrate establishes a useful but narrower result than originally
intended:

- natural KMeans finds two contact-order clusters in the same network;
- the dominant patient mode has a strong rank-profile match;
- the frozen Node field explains part of the held-out TA-TB contrast geometry;
- it does not explain the complete held-out patient event distribution;
- the weaker patient mode remains poorly reconstructed;
- a two-event neuron-level audit found one approximately dominant-source event
  and one event with a strong multi-source alternative.

Consequently, the present field is best interpreted as a mode-separation bias,
not yet as a stable dual-mode generator.  Contact ranks alone cannot distinguish
a travelling recruitment sequence from several spatially separated regions
whose activity is coupled in time.  Optimizing the old rank objective further
would preserve this ambiguity.

### Event-unit amendment (2026-08-22)

Fresh confirmation exposed a second ambiguity: the shared detector merges ON
segments across only 12 ms, while its own return decision observes a 50 ms
settling window.  A direct-readout panel therefore selected two fragments of one
wave packet, separated by 28 ms, and displayed them as opposite modes.  Across
the six selected-field confirmation networks, the 50 ms rule merged 64 adjacent
fragment pairs; 37 had received opposite patient-mode labels before merging.

For rev12-ND, a detector segment is no longer an event-level statistical unit.
Consecutive detector segments separated by at most 50 ms are first merged into
one `settled_episode`.  Contact recruitment, rank, patient assignment, natural
KMeans, source topology and all objective terms are then recomputed on the whole
episode.  The shared detector constants are not changed.  Sensitivities at
25/50/75/100 ms remain reported; 50 ms is primary because it is inherited from
the detector's frozen settling timescale, not selected from model fit.

### Population-excursion correction (2026-08-23; supersedes the 50 ms primary)

The 50 ms correction was necessary but insufficient.  In the Stage-C fit pool,
natural-KMeans direction purity for one apparent leading candidate changed from
0.93 at 50 ms to 0.65 at 100 ms and 0.55 at 150 ms.  One displayed blue event
also recruited only three contacts and was followed 113 ms later by a ten-contact
packet.  Thus the optimizer could still profit from splitting a recurrent
population excursion into favourable contact-rank fragments.

The primary event unit is now a `population_excursion`.  The absolute detector
threshold still identifies high-activity fragments, but fragment grouping and
event boundaries use only the latent whole-network active-fraction trace:

```text
high threshold = frozen common detector threshold
low threshold  = RETURN_FRAC * high threshold
reset time     = 5 * max(tau_m,E, tau_d,GABA) + maximum network delay
```

A new episode may begin only after activity remains below the low threshold for
the complete reset time.  Temporary dips shorter than this do not split an
episode.  The analysis window begins one fast time constant before the first
high-threshold crossing and ends at the start of the stable low-state dwell.
Contact geometry, virtual-contact amplitude and patient labels are forbidden
inputs to these boundaries.

The factor five is the primary fast-state decay reference.  Factors four and six
are mandatory sensitivities.  Candidate ranking, natural K=2 and both mode
prototypes must remain qualitatively stable across all three before any new
field optimization.  The 50 ms Stage-B/C results remain an invalidated
development audit and cannot seed selection or confirmation.

## 2. Scientific question

Can a continuous Node-only excitability field, with total field mass, topology,
delays, background drive, EE/E-to-I redistribution and Z/M all frozen, generate
two patient-aligned interictal modes in the same network such that:

1. both modes match the patient event distribution rather than only a mean rank
   profile;
2. the weaker mode cannot be hidden by the stronger mode;
3. each mode has a reproducible early spatial recruitment topology across
   events and networks;
4. mode-specific early regions have selective effects under same-checkpoint
   intervention.

This is a development refit.  The existing patient held-out blocks and Figure 4
diagnostics have already been inspected, so they are not blind validation.

## 3. Frozen mechanisms

The refit changes only the continuous Node field `h(x, y)`.

- arm: Node-only;
- learned EE redistribution: off;
- learned E-to-I redistribution: off;
- beta and topology growth: closed;
- Z/M and all other slow variables: off;
- topology, delays, GABA, spatial OU law and absolute detector: frozen;
- field mass: frozen to `N_core_manual`;
- `d_i` realization and the map `Vtheta_i = Vtheta_0 - h_i d_i`: frozen;
- duration: at least 20 s per formal candidate/network;
- late runaway: invalid, not a high score.

The accepted current Node field is the paired baseline.  No EE/E-to-I or Z/M
parameter may compensate for a failed Node refit.

## 4. Continuous field representation

No component count, peak count or electrode-centred basis is introduced.

The stored field remains an 18 x 18 tensor cubic B-spline.  Search is performed
through smooth residual control surfaces:

1. Stage A uses a 4 x 4 residual control grid interpolated onto the 18 x 18
   coefficient tensor;
2. Stage B optionally adds a 6 x 6 residual around the Stage-A result;
3. every candidate is projected to the same total field mass and `[0, 1]`
   physical field range;
4. roughness is reported and used only as a weak tie-break unless two candidates
   are otherwise equivalent.

These grids are numerical degrees of freedom, not biological cores.  They cover
the entire sheet and do not receive extra support near observed contacts.

## 5. Patient event representation

The model event unit is a `population_excursion`, not an unmerged threshold
fragment or fixed-gap `settled_episode`.  Every excursion stores its constituent
detector-fragment indices, high-threshold trigger interval, complete analysis
window, reset threshold and reset duration.  A figure or scorer that consumes
pre-group fragment ranks is invalid for rev12-ND.

For event `e` and contact `i`, retain fixed contact identity:

```text
x_e = [m_e, m_e * u_e]
```

where `m_e,i` is recruitment and `u_e,i` is the normalized contact onset rank.
Recruitment and rank each receive half the total weight.  Within each block,
ICL and SCL receive equal total weight, so the longer shaft cannot dominate.

The frozen patient TA/TB labels remain the primary development target.  Model
events are assigned with the patient-training classifier; OOD is reported and
is never silently deleted from the event-distribution loss.

## 6. Patient-alignment objective

For each mode `k`, compare model and patient training events at four levels:

```text
D_k = mean(D_rec,k, D_prec,k, D_profile,k, D_cloud,k)
```

- `D_rec`: shaft-balanced contact recruitment probabilities;
- `D_prec`: unordered contact-pair precedence with a third, not-jointly-
  recruited state;
- `D_profile`: weighted mode prototype error;
- `D_cloud`: sliced-Wasserstein distance in a patient-training-only embedding.

Every component is put in patient recording-block excess-noise units.  For each
mode, two different training blocks are sampled with six events per side.  The
block-to-block median is zero and its q95 is one.  A pooled-mode null is retained
as a diagnostic and is used as a fallback scale only if the block floor is
degenerate.  Raw distances are always stored beside normalized distances.  The
primary patient loss protects the weaker mode:

```text
J_patient = LSE_tau(D_TA, D_TB) + 0.25 * JS(pi_model, pi_patient)
```

The global mean distance and the full event-cloud held-out `R2` are reported
separately.  `R2` is not replaced by squared Spearman correlation.

Candidates with few events are scored with all available events and an explicit
uncertainty interval.  There is no arbitrary requirement for 20 returned events.
If one mode is absent, its finite penalty is the calibrated distance from that
patient mode to an all-contacts-unrecruited negative control, bounded below by
one excess-noise unit.  The candidate is not removed from the record.

## 7. Natural same-network repertoire

Patient assignment and natural KMeans answer different questions and remain
separate outputs.

For every network report:

- event count and returned-event rate;
- natural KMeans K=2 versus K=1 held-out likelihood;
- KMeans seed stability;
- alignment between natural clusters and frozen patient labels;
- counts of TA/TB events in that same network;
- equal-network mode occupancy and OOD fraction.

Pooled KMeans is descriptive only.  Network seed is the independent unit.

## 8. Mode-specific source topology

The SNN worker derives a 1 mm sheet-bin onset map for every returned event from
E-neuron spikes at 2 ms biological resolution.  A bin is recruited at the first
locally persistent activation above its own pre-event q99 baseline.

The source-topology representation contains:

- early 10% recruitment mask;
- normalized local onset-time map;
- connected-component count and dominant-component fraction;
- cross-validated one-source and two-source radial-model scores.

Single-source propagation is not a target.  The formal topology endpoints are:

```text
T_within,k = split-half similarity of mode-k early spatiotemporal templates
T_between  = distance between the two mode templates
```

The topology Pareto axis rewards high within-mode reproducibility and guards
against identical TA/TB source templates.  It does not reward a preselected
number of sources.  One/two-source scores remain mechanistic diagnostics.

## 9. Synthetic controls before simulation

The scoring implementation must pass:

1. copying patient mode prototypes with realistic within-mode noise improves
   every patient-distance component over a global-mean generator;
2. improving only the dominant mode cannot improve the worst-mode objective;
3. randomly permuting contact identity worsens recruitment/precedence/cloud;
4. a stable two-hotspot mode passes topology reproducibility even though it is
   not single-source;
5. random independent hotspots fail topology reproducibility;
6. two identical mode templates fail mode-topology separation;
7. changing only event count cannot create an artificial improvement after
   matched-sample scoring.
8. splitting one synthetic travelling event around a 20-50 ms subthreshold dip
   and then remerging it must reproduce the unsplit contact-rank event unit.
9. changing virtual-contact locations or gains must leave population-excursion
   boundaries exactly unchanged;
10. two packets without a complete fast-state reset must form one excursion,
    whereas a full low-state dwell must separate them;
11. the full-sheet movie must cover the entire analysis window so that repeated
    waves cannot be hidden by a fixed 100 ms display.

Failure of a control blocks long simulation because it means the objective does
not encode the scientific question.

## 10. Historical zero-simulation rescore

Before generating new candidates, rescore all compatible existing continuous
Node artifacts:

- current frozen Node baseline;
- rev10-SA V3-V6 spline fields;
- D6 `f05`, `f09`, primary and joint candidates;
- earlier continuous-field Sobol/interpolation candidates with compatible
  contact and detector contracts.

Report patient loss, weakest mode, event-cloud `R2`, contrast `R2`, natural K=2,
same-network mode coverage and OOD.  Formal-clean events are a historical
diagnostic only; complete returned events are the primary estimand.  Pool
records only when they share both the current spatial-OU contract and the exact
field hash, then count each network seed once, retaining its longest trajectory.
Record-level Pareto positions are not field evidence.  Source topology is
`NOT_RECORDED` for old artifacts without spike-level sidecars; it must not be
imputed.

If a historical candidate dominates the current baseline, it becomes a frozen
initialization.  Otherwise Stage A starts from the current field and the two
orthogonal historical directions that separately improved natural KMeans and
patient geometry.

## 11. Same-checkpoint causal canary

For a selected network/event, save a checkpoint 40 ms before onset and replay
the identical stochastic trajectory.  Apply a 15 ms E-neuron threshold pulse
to:

- the dominant early region;
- the strongest secondary early region;
- a matched off-template region with similar `h`, local E density and baseline
  activity;
- sham.

Primary paired outputs are event occurrence, latency, mode label, contact-rank
change and early topology change.  Intervention is interpreted only within the
model.  A mode-specific source region requires a larger effect on its predicted
mode than the matched control and the other mode.

## 12. Search and selection

### Fit

- common random numbers within each generation;
- 2-3 fit networks per candidate;
- Stage-A low-dimensional residual search before Stage-B refinement;
- patient training blocks only;
- no patient ictal data;
- no EE/E-to-I/Z/M compensation.

### Selection

Use fresh networks and a frozen Pareto rule over:

1. weakest patient-mode loss;
2. complete event-cloud fit;
3. same-network two-mode support;
4. source-topology reproducibility.

No single KMeans scalar selects the field.  The selected point is the fixed
knee of normalized Pareto coordinates, with field roughness as the last
tie-break.

No search resumes until the same shortlist and two-mode interpretation are
stable at four, five and six fast-state decay constants.  Instability is an
event-definition failure, not optimizer uncertainty.

### Confirmation

Run the frozen candidate and current Node baseline on paired fresh networks.
The result may close as a bounded negative result; optimization completion does
not imply scientific success.

## 13. Minimal scientific acceptance

Only three claim conditions are used:

1. complete held-out event-cloud `R2` is positive and improves over the frozen
   Node baseline without mode collapse;
2. both patient modes improve on paired networks, so the dominant mode cannot
   hide the weaker mode;
3. mode-conditioned source topology is reproducible and at least one predicted
   source region has a selective same-checkpoint intervention effect.

If these are not jointly established, the best candidate remains an exploratory
field and EE/E-to-I/Z/M remain closed.

## 14. Figure deliverables

The final package contains:

1. a Fig.4-style direct waveform panel with natural KMeans;
2. event-rank heatmap, patient/model prototypes and cross-fit matrix;
3. complete event-cloud and TA-TB contrast `R2`;
4. all-event mode-conditioned source-topology distributions;
5. same-checkpoint hotspot intervention effects.

Every figure directory includes a Chinese `README.md`.  Figures do not display
PASS/FAIL banners or internal status codes.

## 15. Claim boundary

Success would support a development-stage, model-internal statement that a
patient-constrained continuous excitability field can organize two reproducible
interictal propagation modes.  It would not identify anatomical cores, prove
patient causality, establish clinical generalization or validate an ictal
mechanism.
