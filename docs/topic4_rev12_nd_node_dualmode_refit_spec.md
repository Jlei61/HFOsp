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

### Spatiotemporal-cascade correction (2026-08-24; necessary but insufficient)

Fresh canaries showed that a global population reset is still only an outer
envelope.  One apparent pattern-2 excursion lasted about 0.8 s and contained
five detector fragments assigned to five disconnected sheet-level cascades;
one fragment was itself a two-origin compound.  Across all four canary runs,
16-18 returned population excursions per run contained multiple cascades.

The primary optimization event is therefore a `spatiotemporal_cascade` inside
the population-excursion envelope.  It is defined from the 2 ms by 1 mm
whole-sheet E-neuron activity movie, with no virtual-contact input:

1. a sheet bin is active when at least two E neurons fire in the frame;
2. active nodes connect within the same or adjacent sheet bin and in the same
   or immediately adjacent movie frame;
3. detector fragments whose dominant activity belongs to the same connected
   component form one cascade event, even if the high detector briefly drops;
4. a fragment whose largest component carries less than 70% of its activity is
   a `compound` observation and cannot be silently assigned a propagation
   direction.

The three-neuron bin threshold and 0.65/0.75 dominance thresholds are mandatory
sensitivities.  In the fresh canary they preserved the negative conclusion:
natural-KMeans patient-direction purity remained 0.56-0.59 and complete
held-out prototype R2 remained -0.35 to -0.44 for both historical fields.

The later root-preserving historical rescore materially tightened this negative
result.  Across all 18 historical continuous fields, every matched patient loss
remained above the patient block q95 floor and held-out prototype R2 remained
negative (approximately -0.58 to -0.78).  Natural KMeans still retained a
two-direction tendency for some fields, but 37-54% of detector fragments were
compound under the directed-lineage audit.  Therefore detector-window dual-mode
appearance is not an acceptable optimization endpoint; the directed lineage is
the primary event unit for every subsequent fit.

The optimization readout must also be lineage-restricted.  Cropping the original
whole-network contact envelope to a directed-root window is insufficient because
a concurrent independent root can still recruit a contact inside that window.
For the corrected readout, the selected root's 2 ms x 1 mm activity is projected
through the frozen 0.25 mm Gaussian contact sampler and 5 ms temporal smoothing.
The binned sampler must reproduce the original whole-network contact traces with
Pearson r >= 0.98 at every contact before its lineage-restricted ranks are used.
The unconditioned 15-contact traces remain a visual diagnostic only.

Under this final event/readout contract, all 18 historical fields still failed
patient-distribution reconstruction.  The best corrected historical candidate
had matched patient loss 1.161, OOD fraction 0.471 and held-out prototype R2
-0.836.  Across all 36 frozen trajectories, only 159 of 1,574 returned roots
recruited at least ten contacts.  This library is therefore an initialization
and negative control for a new Node fit, not an accepted substrate.
This correction removed grossly disconnected packets, but its 3-D connected
components were undirected.  Two independent roots that met later in time were
therefore merged retrospectively.  A full 18-field historical run under this
definition produced balanced KMeans/patient-direction alignment as high as
0.93, but that value is an intermediate diagnostic and cannot select a field.

### Directed-lineage correction (2026-08-24; superseded)

The current event identity is a `directed_spatiotemporal_lineage`.  The same
2 ms by 1 mm whole-sheet movie and two-neuron activity threshold are retained,
but ancestry is propagated forward in time:

1. contiguous active patches in a frame receive roots only from the immediately
   preceding frame's 3 x 3 spatial neighborhood;
2. a patch with one parent root inherits that root;
3. a patch reached by several roots is partitioned by a seeded watershed, so
   the roots remain distinct after collision;
4. only equidistant watershed boundaries are marked as collision mass;
5. a detector fragment enters a lineage only if one root explains at least 70%
   of active-bin mass, counting collision mass in the denominator;
6. a lineage is `returned` only if all constituent detector fragments satisfy
   the frozen return rule and the lineage ends before the recording boundary.

Virtual contacts and patient labels are read only after root identity and event
windows are frozen.  The definition is an operational directed lineage at the
movie resolution, not proof of a synaptic path.  Root collision, compound
fraction and threshold sensitivity remain explicit diagnostics.

The first implementation nevertheless inspected only the immediately preceding
2 ms movie frame.  In the exact-neuron 54-field fit this generated a median of
2,504.5 roots for only 120 detector fragments per network, and 47.2% of detector
fragments were compound.  A local frame that briefly fell below two active
neurons could therefore reset root identity even though membrane, inhibitory
synaptic and delayed-network state had not reset.  The resulting 108-run field
ranking is retained as a diagnostic but is formally invalid for selection.

### Persistent directed-lineage correction (2026-08-24; necessary but insufficient)

The primary unit is now a `persistent_directed_spatiotemporal_lineage`.  It keeps
the same whole-sheet movie, exact per-neuron contact sampler and root-preserving
collision rule, but a patch searches backwards to the nearest recent local
parent.  The parent memory is frozen from model timescales:

```text
causal memory = 5 * max(tau_m,E, tau_d,GABA) + maximum network delay
```

The factor five is inherited from the existing fast-state reset contract.  It is
not fit to contact ranks or patient labels.  A root can resume only within the
same or an adjacent 1 mm sheet bin; a spatially separate activation remains a
new root even inside the memory interval.  If several recent roots reach a patch,
the seeded watershed keeps them separate.  Only the nearest eligible parent
frame contributes ancestry, so a stale root cannot compete with an active one.

Before any further field fit, the event identity must pass four synthetic tests:
one travelling wave remains one root, two independent waves remain distinct after
collision, a short local interruption retains identity, and activity after the
memory expires receives a new root.  Three frozen fields on two networks then form
an event-identity canary.  Event partition, compound fraction and mode assignment
must be qualitatively stable for three, four and five fast-state time constants.
This is an event-definition canary only and cannot select a field.

The canary reduced immediate-frame root fragmentation, but 40-52% of detector
observations still contained several persistent roots and were excluded from the
patient objective.  This lets a field score well by producing a small favourable
single-root subset while complete multi-site events remain unseen.  Moreover,
grouping roots merely because they occur inside one detector window would make
the opposite error: an unusually long observation window could merge unrelated
roots.  Persistent roots are therefore retained as topology annotations, but no
longer define or exclude statistical event rows.

### Causal population-episode correction (2026-08-24; necessary but overmerging)

The statistical unit for KMeans and patient scoring is one complete population
excursion bounded by fast-state reset.  The high-threshold fragments are detected
from whole-network E active fraction, and consecutive fragments remain one event
until activity has stayed below `RETURN_FRAC * detector_threshold` for:

```text
reset time = c * max(tau_m,E, tau_d,GABA) + maximum network delay
```

with `c=5` primary and `c in {4,5,6}` mandatory sensitivities.  Contact position,
contact amplitude, contact recruitment and patient labels are absent from this
boundary calculation.  Every detector fragment belongs to exactly one episode.
The complete virtual-contact envelope within that episode supplies one onset-rank
row.  Thus a long or multi-site event cannot be split into favourable A/B rows,
and a multi-root event cannot disappear from the objective.

Persistent directed roots are computed on the same whole-sheet movie and stored
inside each episode.  Root count, root activity fractions, collisions and source
maps describe event topology only.  They do not split an episode, select a subset
of spikes, or impose a penalty merely because several roots exist.  If the 4/5/6
reset choices materially change episode partition, patient-mode assignment or
natural KMeans, event identity remains unresolved and field optimization stays
closed.

The six-trajectory canary failed that stability check.  Moving from four to six
fast-state constants reduced returned episode counts by roughly one half; boundary
ARI was 0.67-0.86 and patient-mode agreement on shared fragments was 0.54-0.91.
Representative 5-tau episodes lasted about 0.7-0.8 s and visibly contained six
or seven separate population packets.  A global quiet-state dwell therefore
overmerges observations in this active background and cannot be the optimization
unit.  These complete-excursion GIFs remain the required outer diagnostic.

### Persistent-root coactivity episode (2026-08-24; grouping retained, memory superseded)

The current statistical unit combines the stable part of the two corrections:

1. persistent roots retain local causal continuity across short detector dips;
2. every positive-mass root in a detector fragment is retained;
3. roots that are active in at least one common 2 ms movie frame form one complete
   observable ensemble;
4. ensembles across detector fragments merge only through a shared persistent
   root;
5. roots that are merely sequential inside one unusually long detector window
   remain separate observations;
6. a fragment may map to several observations only when it truly contains several
   temporally disjoint root ensembles; this multiplicity is stored explicitly.

Contact geometry and patient labels enter only after these groups are frozen.
Exact per-neuron contact readout uses the union of all roots in an ensemble and
excludes activity from concurrent roots outside that ensemble.  No dominance
threshold deletes an event.

The first canary used `4/5/6 * max(tau_m,E, tau_d,GABA) + maximum network
delay`.  It improved natural KMeans alignment over the global-excursion unit,
but remained overlong: 91-96% of observations contained several roots and
algorithmically selected examples lasted 120-330 ms.  The formula combined a
local 1 mm parent search with the maximum delay anywhere in the 20 mm network
and retained a PSP after only 0.7% of a membrane exponential remained.  This
memory definition is rejected before field optimization.

### Engine-derived root memory (2026-08-24; rejected grouping canary)

The coactivity grouping above is retained, but root memory is derived from the
excitatory pathway that can propagate E-neuron activity:

```text
root memory = E-to-E AMPA-to-membrane PSP tail support
            + local E-to-E axonal-delay support
```

The PSP support is computed with the same exponential-Euler AMPA gating,
synaptic-current and E-membrane updates as the simulator.  The primary endpoint
is the descending time at 10% of PSP peak; 20% and 5% are mandatory
sensitivities.  Delay support uses only E-to-E edges whose source and target lie
inside the same 1 mm parent cone and takes their maximum frozen delay.  For seed
2211 this gives 58.0 ms PSP support plus 9.4 ms local delay, rather than the old
100 ms plus 35.2 ms global delay.  These values use no contact geometry, patient
label or fit score.

Candidate summaries retained the worst matched patient loss and minimum natural-
KMeans alignment over the 5%/10%/20% PSP-tail definitions.  Shortening memory
did not repair event identity: 91-97% of events still contained several roots,
the median event contained 5-7 roots, and representative A/B windows showed
successive spatial packets rather than one propagation cause.  The failure is
the transitive coactivity grouping: if A overlaps B and B overlaps C, all three
become one observation even when A and C never share an initiation.  This event
unit is rejected and cannot reopen the field fit.

### Observable causal-root event (2026-08-24; local partition retained)

The formal model event is now one directed latent root.  Virtual-contact
detector fragments determine whether the root is observable, but they cannot
move its start or stop.  Fragments dominated by no single root remain explicit
`compound` observations and never enter the two patient direction classes.
They are retained as a field-quality endpoint rather than silently dropped.

Root continuation uses the descending half-maximum E-to-E PSP support plus the
maximum delay among local E-to-E edges compatible with the 1 mm movie cone.
The 80% and 20% PSP landmarks are mandatory memory sensitivities.  Root purity
uses 0.70 as primary and 0.60/0.80 as mandatory sensitivities.  The five
one-axis-at-a-time variants are all reported; no best event window, tail value
or purity threshold may be selected from patient loss or KMeans performance.

The required GIF shows all sheet activity while outlining only the causal root
used for contact ranks.  Concurrent activity must remain visible, but mixed
activity is labelled compound instead of being hidden or forced into A/B.

### Delayed E-to-E support amendment (2026-08-24; replay required)

The local 1 mm parent cone removes detector-window leakage but is not by itself
a complete causal graph.  A root may jump farther than one movie bin through a
frozen E-to-E edge and its axonal delay.  Conversely, edge existence alone is
not sufficient because the 25.6-million-edge graph contains many weak possible
routes.  The event producer therefore adds a contact-independent second pass.

For every local-root birth patch, delayed E-to-E support from each earlier root
family is estimated as:

```text
active source neurons * frozen delayed E-to-E weight
----------------------------------------------------
source-bin neuron count * target-bin incoming-E budget
```

The child joins the dominant earlier family only when that family explains at
least 70% of all candidate parent support and contributes at least 0.001 of the
target incoming-E budget.  Support 0.0003/0.001/0.003 and floor/nearest/ceil
delay-to-movie-frame mappings are frozen one-axis sensitivities.  None may be
selected by patient loss or KMeans.  Detector contacts still establish only
observability and cannot alter family boundaries.

A zero-simulation audit of four representative fields on both fit networks
changed three of 477 clean roots (0.63%) and recovered one of 983 detector
fragments from compound to a single supported family at the primary setting.
At the weakest support sensitivity, 4.2% of clean roots changed.  Thus the old
fit is not wholesale invalid, but its objective is not exactly identical to the
new estimand.  Stage-Q ranking remains provisional until deterministic exact-
spike replay of four fit-only representatives confirms that patient loss,
natural KMeans, direction and compound conclusions are stable.  Selection and
confirmation seeds remain closed during this replay.

The exact replay preserved every population activity, sheet movie and contact
envelope sample.  The leading field remained the same, but one recovered event
made the diagonal-GMM K2-vs-K1 auxiliary collapse while natural KMeans direction
alignment remained high.  This is an estimator instability, not a change in SNN
dynamics or evidence that the second mode vanished.  From the next fit onward,
the GMM K2 score is diagnostic only (`k2_support_weight = 0`).  The continuous
patient-distribution loss and balanced natural-KMeans/patient-direction
alignment remain in the objective.  The Stage-Q fields may be retained as a
geometry library, but their old scalar ranking is invalid for selection.

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

The model event unit used by KMeans and the patient objective is one
`causal_root_observation`, not an unmerged threshold fragment, fixed-gap
`settled_episode`, global quiet-state excursion or transitive multi-root
episode.  A detector fragment establishes that a latent root is observable but
cannot alter that root's start or stop.  A fragment without one root carrying
at least the frozen dominance fraction is retained as `compound` and does not
enter either patient direction class.  A figure or scorer that uses detector
fragment boundaries or forces compounds into A/B is invalid for rev12-ND.

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

Every objective draw compares exactly six model events with six patient events
from one eligible training block.  Component distances are divided by that
mode's patient block-to-block q95.  They are not clipped below the floor median,
so a field cannot obtain a zero loss merely by producing more events than the
floor calibration.  Raw distances are always stored beside normalized
distances.  The primary patient loss protects the weaker mode:

```text
J_patient = LSE_tau(D_TA, D_TB) + 0.25 * JS(pi_model, pi_patient)
```

The global mean distance and the full event-cloud held-out `R2` are reported
separately.  `R2` is not replaced by squared Spearman correlation.

There is no arbitrary requirement for 20 returned events.  A network with fewer
than six directed-lineage events in either patient-assigned mode receives a finite
missing-mode penalty and remains in the record.

The exploratory fit objective adds three continuous diagnostics:

```text
J_fit = J_patient
      + 0.50 * (1 - balanced KMeans/patient-direction alignment)
      + 0.25 * OOD fraction
      + 0.25 * compound detector-fragment fraction
      + 0.50 * (1 - causal direction score)
```

Compound fraction is a continuous penalty, not a hard exclusion.  Without this
term an optimizer can generate mostly mixed observations and obtain a good score
from a small lucky clean subset.  This penalty does not assert that patient
events have one biological source; it enforces the declared model estimand that
one optimization sample must have one identifiable latent causal root.

KMeans is computed after drawing the same event count from every network.
Patient held-out R2 and source topology do not select the fit library.
The formal candidate value is computed across the frozen one-axis-at-a-time
causal-memory, purity and delayed-edge-support variants.  It uses the
componentwise conservative envelope: maximum patient loss, OOD and compound
fraction, and minimum KMeans alignment.  No best-window selection is allowed.
Per-network held-out diagonal-GMM K2 support is stored as a diagnostic with zero
selection weight because one recovered event changed that statistic from
intermediate support to effectively zero without changing the trajectory,
silhouette or natural-KMeans direction alignment.

The causal direction score is computed from the root-restricted 1 mm onset map.
For each event, the centroid of the latest 20% recruited bins minus the centroid
of the earliest 20% gives an early-to-late displacement.  The patient-training
TA/TB rank-contrast gradient freezes a 2-D axis and opposite expected signs for
the two modes.  Signed cosine is first averaged across all evaluable events in
one mode and only then clipped below at zero.  The weaker corrected mode is the
network score and networks receive equal weight.  Event-wise clipping is
forbidden: it would give a positive score to a 50/50 mixture of forward and
reverse events even though that mode has no reproducible direction.  A same-
direction pair, stationary activation, transverse wave or sign-cancelling mode
therefore cannot pass merely because contact KMeans finds two clusters.

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

KMeans K=2 is not evidence for two modes by itself because it always returns two
clusters.  Held-out diagonal-GMM likelihood under K=2 versus K=1 is therefore
reported in every network, but receives zero selection weight after its
single-event instability audit.  Natural KMeans remains an auxiliary continuous
alignment endpoint, not proof of two biological generators.  Network seed is
the independent unit.

### Corrected-event Stage-S result and continuation decision

All 54 frozen fields were replayed on fit seeds 2241 and 2242 under the final
edge-supported causal-family event unit: 108/108 workers completed and none
failed.  The first aggregation reported `J_fit = 1.787` and weakest-mode causal
direction 0.566 for its leader.  Those two values are retrospectively invalid:
the implementation clipped each event before the mode mean.  Zero-simulation
correction reduced that field's direction score to 0.250 and increased its
objective to 1.945.  Patient loss 1.205, balanced natural-KMeans alignment
0.842, OOD 0.510 and compound fraction 0.632 are unaffected.  Selection and
confirmation seeds remain closed.

A zero-simulation full-onset-map audit replaced the early/late centroid summary
with event-wise Spearman monotonicity along the frozen patient-training axis.
The two direction scores correlated 0.928 across all 54 fields and preserved the
same leading neighborhood.  The failure is therefore not explained by the
centroid score hiding a stable reverse travelling wave.  The current smooth
field neighborhood lacks a robust weaker-direction solution under the corrected
event definition.

The next fit stage has two deliberately separate roles.  Eighteen selectable
continuous fields follow finite-difference response directions estimated from
the already completed fit library.  One non-selectable continuous-spline
approximation of the historical smooth two-core field is a rigid capacity
control.  It may use the historical source/sink geometry only because it cannot
become the data-driven answer.  If that control also fails to produce two
opposite causal modes, the fixed Node-only scaffold lacks demonstrated capacity
and further field optimization stops before EE, E-to-I or Z/M is opened.  If it
passes while the selectable proposals fail, the capacity exists but the current
patient objective/search directions have not recovered it.

Stage-T completed 38/38 field-network runs without runaway.  No selectable
proposal improved the corrected Stage-S objective.  Its pre-correction scalar
leader had patient loss 1.244, KMeans alignment 0.803, OOD 0.469, compound
fraction 0.565 and event-wise-clipped direction 0.520; mode-mean correction
reduced direction to 0.182 and increased `J_fit` to 2.011.  The non-selectable
smooth dual-core control retained a corrected direction score of 0.573 and
full-map monotonicity 0.467 across both networks, but had patient loss 1.411 and
OOD 0.679.  Thus the frozen Node scaffold has directional capacity, while the
current data-driven field family has not jointly recovered stable direction and
the patient event distribution.  This result does not license using the manual
control as an initialization for a claimed free-field recovery.

The correction was executed at commit `454b005d` without rerunning the SNN.
The immutable corrected sidecars are
`node_stage_s_edge_supported_field_refit/aggregate/fit_cascade_summary_mode_mean_direction.json`
(SHA-256 `1efc07fdebf112bdac3c139f78fa9c7b90f046d041941f5c5b4cba423517485b`)
and
`node_stage_t_causal_continuation_capacity/aggregate/fit_cascade_summary_mode_mean_direction.json`
(SHA-256 `26137f7a5a2030fbfc6411f54e8ec068677b358557bef1567cafb25dbfcc9dda`).
The original aggregates remain audit evidence and must not be used for field
ranking.

A zero-simulation coordinate audit then tested whether the four Stage-S Sobol
directions supplied a reproducible local gradient.  The first audit artifact
(`search_coordinate_diagnostic_v1`) is invalid because it projected raw spline
coefficients without applying both constant equivalences used by the simulator:
coefficient-mean removal in `continuous_surface` and spatial-constant removal by
the mass projection.  It is retained as provenance only.  The corrected v2
artifact performs every projection on the effective mean-free sheet surface.

The next fit-only screen, Stage-U, therefore does not continue along the failed
four-direction response surface.  It freezes the corrected Stage-S scalar leader
as a data-driven anchor and spans the uniform 20 mm sheet with all 15 nonconstant
two-dimensional cosine modes up to frequency index three.  These modes are
projected into the stored 18 x 18 spline in the effective mean-free surface
metric, are mutually orthonormal, and use no contact, shaft, patient-source or
manual-core coordinate.  Each mode is tested symmetrically at surface RMS 0.08
on fit networks 2241--2243.  This is a derivative screen, not a field-selection
round: no combination proposal is allowed until mode effects show cross-network
sign support.  The smooth manual capacity control is not simulated and only
audits generic span coverage.

Stage-U completed all 93/93 frozen runs without failure or runaway.  The anchor
remained the best aggregate field (`J_fit=2.0608`, patient loss 1.2313, natural
KMeans alignment 0.7303, OOD 0.5364, compound fraction 0.5759, causal-direction
score 0.1668).  Every single-mode field at either sign of RMS 0.08 had a worse
aggregate objective.  Some endpoints moved in useful but opposing directions:
`f01+` and `f02-` improved both patient-mode losses, whereas `f11+` and `f14+`
partly improved KMeans/direction, but none jointly improved the anchor.

The first Stage-U response sidecar selected six proposed RMS 0.16 checks from
the slope between the negative and positive arms.  That rule is not actionable:
one arm can be better than the opposite arm while both remain worse than the
anchor.  The sidecar is retained as audit provenance, but its outer-amplitude
follow-up is cancelled and must not be launched.

The replacement anchor-relative audit fits a diagonal local quadratic through
`-0.08`, `0` and `+0.08` for each generic mode.  It may nominate sparse canaries
only within coefficient L2 radius 0.06 and only when the surrogate predicts
positive aggregate changes in the full objective, both patient modes, natural
KMeans and causal direction with at least two-of-three network sign support for
the objective and both modes.  This is a retrospective trust-region diagnostic,
not a field optimizer: cross-mode interactions are unmeasured, predicted gains
are small (about 0.01 utility), and every nominated combination requires direct
SNN validation on the same fit networks before any fresh network is opened.

The direct Stage-W canary therefore contains the anchor, four single-direction
components needed for attribution (`f01+`, `f02-`, `f11+`, `f14+`) at reduced
RMS 0.03, and at most two sparse combinations nominated by the anchor-relative
audit.  It is limited to fit networks 2241--2243.  Stage-W closes as a bounded
negative result if no directly simulated candidate simultaneously improves the
aggregate objective, both patient-mode losses, natural KMeans and causal
direction, with at least two-of-three network support for the objective and both
mode losses; selection, confirmation, intervention, EE, E-to-I and Z/M stay
closed in that case.

Stage-W completed all 21/21 runs.  No candidate satisfied the frozen joint
advance rule.  The most informative near miss was `f14+0.03`: its aggregate
objective improved by 0.0155 utility, patient mode-0 and mode-1 utilities by
0.0156 and 0.0123, direction by 0.0478 and monotonicity by 0.0290.  Natural
KMeans alignment decreased by 0.0223, however, and paired network effects were
heterogeneous: total utility improved in only one of three networks and mode-1
loss improved in only one of three.  The aggregate gain was therefore driven by
network 2243 and cannot open selection.  Complete held-out event-cloud R2 moved
only from -0.907 to -0.856 and remains negative; this diagnostic was not used to
choose the next fit experiment.

Because three networks cannot distinguish a reproducible small effect from one
network-specific response, Stage-X freezes a paired uncertainty replication of
only the anchor and `f14+0.03` on new fit networks 2251--2259.  It does not
resume field search.  One joint decision is made after all nine pairs finish:
aggregate utility, both patient-mode utilities, natural KMeans and causal
direction must all improve, and utility plus both mode losses must improve in at
least six of nine paired networks.  Patient held-out quantities, source-topology
plots and historical Fig.4 diagnostics are reported but cannot affect this
decision.  Failure closes this local free-field basin; success permits only a
fresh-selection review, not Node-field acceptance.

Stage-X completed all 18/18 paired runs.  `f14+0.03` did not reproduce its
Stage-W advantage: the primary objective utility was negative on average, its
90% network-bootstrap interval included zero, and only four of nine networks
improved.  Aggregate patient mode-1 loss, natural KMeans alignment and causal
direction also worsened.  The frozen decision is therefore
`CLOSE_CURRENT_ANCHOR_LOCAL_BASIN`.  No further amplitude, seed or local
cosine continuation is permitted from this result.

### 7.1 Continuous patient-mode target audit

The closed local basin exposed a separate target problem.  The frozen patient
classifier already returns a continuous `P(TB)` for every causal-family event,
but the fit path thresholded it at 0.5 and then optimized a hard KMeans
alignment.  A small event-cloud displacement could therefore change both its
patient-mode membership and its scalar loss discontinuously.  KMeans is still
required for the Fig.4-style final validation, but it is not a suitable field
fit coordinate.

Stage-Y is a zero-simulation audit.  It retains the causal-family event unit and
replaces hard membership in the development score with patient-mapped
classifier probabilities.  For mode `k`, recruitment, precedence, profile and
event-cloud distances use probability weights `1-P(TB)` or `P(TB)`.  The two
mode losses remain protected by a smooth worst-mode LSE.  Three additional
continuous terms prevent a single or ambiguous event cloud from masquerading
as two modes:

```text
J_soft = LSE(D_TA, D_TB)
       + 0.50 * JS(soft occupancy, patient occupancy)
       + 0.25 * mean[4 P(TB)(1-P(TB))]
       + 0.50 * (1 - aligned mode-contrast amplitude).
```

The target must prefer an exact two-mode patient-training reconstruction over
an ambiguous continuous cloud, a one-mode generator and a contact-permuted
generator.  Exact replication of every event must leave the score unchanged.
All four controls passed.  The first 40-field zero-simulation rescore ranked
`stage_u_f02_m` first, but this is only a three-network development result and
does not freeze that field.  More importantly, Stage-X's paired soft-objective
utility was only +0.0021 with five positive and four negative networks, which
independently confirms that the local candidate has no stable advantage.

Natural KMeans, its silhouette and held-out K2-vs-K1 density evidence are now
final-validation diagnostics.  They cannot enter a global fit scalar.  The
continuous causal-direction and full-map monotonicity scores remain separate
Pareto axes so that a good contact-rank distribution cannot hide one-direction
or spatially synchronous sheet activity.  Patient held-out events remain
excluded from field fitting and ranking.

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
12. a synthetic travelling wave interrupted by a detector dip remains one
    cascade;
13. simultaneous spatially disconnected sources remain separate components and
    are marked compound if one detector fragment contains both;
14. cascade conclusions remain stable at two versus three active neurons per
    bin and 0.65/0.70/0.75 dominance.

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
stable across 80%/50%/20% excitatory-PSP support and 0.60/0.70/0.80 root
purity.  Instability is an
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

## 16. Stage-Z global soft-field screen

Stage-X closed the tested anchor-local basin and Stage-Y showed that hard
`P(TB)>=0.5` labels discard useful continuous information. Stage-Z therefore
tests a genuinely multidirectional but still continuous Node field. It is not
a new core-count model and does not use electrode locations to place basis
functions.

The search span consists of the 15 non-constant cosine modes with spatial
frequencies `kx,ky=0..3`, projected onto the frozen 18 x 18 spline field. These
modes are built on a uniform 20 x 20 mm sheet and receive no contact, shaft,
patient-source or manual-core coordinates. Sixteen scrambled-Sobol directions
are paired antithetically at sheet-space RMS radii 0.16, 0.34 and 0.52. The
frozen anchor and historical soft-score benchmark bring the fit screen to 34
fields. Every field is simulated on the same three new fit networks
2261--2263, giving 102 common-random-number runs.

Nomination is exploratory and Pareto-based. Its five axes are soft patient
training objective (minimize), soft causal direction, causal wave monotonicity,
cross-network source-topology reproducibility and between-mode source-topology
separation (maximize). Natural KMeans is a final validation diagnostic and
patient held-out data are unopened. Up to six nominees may pass to expanded fit
replication, with no more than two from one radius; best soft fit and best causal
direction are retained as sentinels. This screen cannot freeze Node or open
selection, intervention, EE, E-to-I or Z/M by itself.

Stage-Z's first resource canary showed that worker memory grows during the
trajectory: 20 workers reduced available memory from 230 GiB to 19 GiB before
any worker completed, so the controller stopped the batch before OOM. The
frozen rerun therefore uses at most 14 workers, a 14 GiB per-worker estimate,
at least 32 GiB reserved memory and a 600 s controller interval. This resource
correction changes no candidate, seed, simulation or selection rule.

Stage-Z completed 102/102 runs with 34/34 fields evaluable on three networks.
The screen exposed a fit-direction tradeoff rather than a frozen solution.
`stage_z_g09_m` had the lowest mean soft patient-training objective and improved
both mode losses in aggregate, but the improvement occurred in only two of
three networks and its causal-direction score was zero in two networks.
`stage_z_g05_p` had positive causal direction in all three networks, but worsened
the mean patient fit. Therefore Stage-Z does not freeze Node and does not open
held-out, KMeans selection, intervention, EE, E-to-I or Z/M.

Stage-AA copies the six candidates nominated by the predeclared Stage-Z Pareto
rule plus the unchanged anchor onto nine fresh fit networks 2271--2279. All
seven field hashes remain unchanged. This 63-run expansion measures whether the
fit-direction tradeoff is stable; it may reduce the fit shortlist but still
cannot select the final Node field or inspect patient held-out data.

Stage-AA completed 63/63 runs. `stage_z_g04_m` was the only candidate with
positive fresh-network mean utilities for the soft objective, both modes and
causal direction. Its 90% paired network-bootstrap intervals were positive for
mode 0 and direction, but not for the total objective or mode 1; mode 1 improved
in only 4/9 networks. It is therefore a balanced development anchor, not an
accepted Node field.

The Stage-Z diversity cap omitted `stage_z_g11_p` after two other RMS-0.52
fields filled that radius. This field was nevertheless Pareto-optimal, had the
second-lowest Stage-Z soft objective, jointly low mode-0/mode-1 losses and
positive causal direction. Before inventing a new interpolation family, one
fit-only recovery runs this already simulated candidate on the same nine
Stage-AA networks. This is an explicit development follow-up, not an independent
selection or confirmation result.

Stage-AB completed 9/9 runs without runaway or missing artifacts. On the fresh
networks, `stage_z_g11_p` improved mode 1 in 6/9 networks, but worsened the mean
soft objective and mode 0, improved causal direction in only 4/9 networks, and
reduced both cross-network topology reproducibility and between-mode topology
separation. All corresponding 90% paired bootstrap intervals except the
compound-event diagnostic crossed zero. Its formal status is therefore
`OMITTED_PARETO_MODE1_ONLY_NO_BALANCED_STABILITY`: it supplies a local mode-1
direction but cannot freeze Node.

Stage-AC is the final small response-surface experiment before another global
search is considered. Let `C_0` be the unchanged Stage-Z anchor coefficients,
and `C_4`, `C_11` and `C_5` be the coefficients of the balanced, mode-1 and
causal-direction donors. The frozen continuous field is

```text
C = C_4 + lambda_11 (C_11 - C_0) + lambda_05 (C_5 - C_0),
lambda_11 in {0, 0.25, 0.50},
lambda_05 in {0, 0.15, 0.30}.
```

The already run origin is omitted, leaving eight fields. This is coefficient
interpolation in one complete two-dimensional spline, not allocation of a
fixed number of cores. Every candidate preserves the coefficient budget, uses
no patient/contact coordinates for construction, and must remain below 0.54
RMS from the anchor on the uniform sheet grid. The same nine fit networks are
reused so the experiment estimates a local donor-dose response surface. It
cannot inspect held-out, use natural KMeans for selection, or open EE, E-to-I
or Z/M.

Stage-AC completed 72/72 runs. No field met the fit-only balanced advancement
rule. The best joint-mean cell, `stage_ac_m000_d030`, improved the total soft
objective in 7/9 networks, mode 1 in 6/9 and causal direction in 6/9, but mode 0
in only 4/9; it also did not outperform the `g04` center. Across the complete
factorial surface, increasing the mode-1 donor dose significantly worsened the
total objective and mode 0 without a stable mode-1 benefit. Increasing the
direction donor to 0.30 significantly improved causal direction but worsened
the total objective and mode 0. The status is
`LOCAL_INTERPOLATION_DIRECTION_ONLY_NO_BALANCED_NODE_FIELD`. Further dose
densification on these fit networks is closed.

A zero-simulation implementation audit also shows that signed threshold depth
does not explain this failure at the resolved spatial scale. Across the nine
networks, 1 mm maps of the continuous field have pairwise correlation about
0.999; the realized signed excitability map `-delta Vtheta` remains correlated
about 0.94--0.96 across networks and about 0.97 with `h` within a network. Thus
the field is reproduced consistently, but it does not consistently determine
both propagation modes.

Stage-AD therefore repeats the historical smooth dual-core spline only as a
non-selectable capacity control on the same nine networks. It uses the exact
historical field hash and cannot enter Pareto nomination, initialize a claimed
data-driven recovery or open held-out. If it expresses stable opposite causal
modes, the Node-only scaffold has directional capacity and the unresolved
problem is joint patient-distribution recovery. If it fails on nine networks,
the earlier two-network capacity result is not robust enough to justify further
Node-field search on this scaffold.

Stage-AD completed 9/9 runs without runaway. Both soft modes had effective
support in every network (the smaller per-mode minimum was 16.9 effective
events), and weakest-mode causal direction and causal monotonicity were
strictly positive in 9/9 networks. Natural KMeans was a secondary diagnostic:
all nine networks yielded two non-empty clusters and direction-balanced
alignment above 0.5. The formal status is
`NODE_ONLY_DIRECTIONAL_CAPACITY_POSITIVE_PATIENT_JOINT_RECOVERY_UNRESOLVED`.

This closes scaffold incapacity as the leading explanation, but it does not
validate the historical field as data-driven. Relative to the fit-only `g04-`
field on the same networks, the manual control improved causal direction by
0.154 (90% paired-bootstrap interval 0.084--0.228) and causal monotonicity by
0.151 (0.085--0.215), while worsening the patient soft objective by 0.225
(0.166--0.280). The mode-0 loss worsened by 0.266 (0.227--0.307), whereas the
mode-1 improvement was small and uncertain. Mean OOD remained 0.609. Therefore
the current failure is a joint-recovery problem: the scaffold can express two
causal directions, but the current data-driven search does not recover that
capacity while preserving the full patient event distribution.

No further interpolation or simulation is licensed directly from Stage-AD.
Stage-AE must first decompose the observed trade-off by recruitment,
within-shaft precedence, cross-shaft precedence, profile and event cloud using
only already generated fit artifacts. Any revised optimization target must be
derived from the patient mode axes and model causal trajectories, not from the
manual field geometry. The manual field remains a capacity control and may not
seed or anchor a claimed recovery.

Stage-AE completed without simulation. Relative to `g04-`, the manual control's
mode-1 causal direction increased by 0.183, while mode-0 causal direction
decreased by 0.064. All four mode-0 patient loss terms worsened: recruitment
0.370, precedence 0.234, profile 0.317 and event cloud 0.144. Mode-1 patient
losses moved slightly in the favorable direction, while its soft occupancy and
effective event count increased. Thus the observed trade-off is specifically a
shift toward a directionally coherent mode 1 at the expense of the full mode-0
distribution. This is not proof that patient matching and opposite causal
propagation are mathematically incompatible.

The old Stage-Z perturbation library cannot identify a trustworthy local search
gradient. Its 16 by 15 direction matrix is full rank but has condition number
69.8. Leave-one-direction Pearson correlation was 0.205 for mode 0, -0.179 for
mode 1 and 0.086 for mode-1 causal direction; cross-network gradient cosine was
also unstable. The status is
`OBSERVED_CROSS_MODE_TRADEOFF_NEW_PAIRED_DESIGN_REQUIRED`. Therefore another
field proposal cannot be calculated from the old three-network screen without
substantial overfitting.

Stage-AF is a fit-only orthogonal response calibration around `g04-`. It uses
the 15 observation-invariant low-frequency cosine basis functions already
frozen in Stage-Z, one positive and one negative perturbation per basis mode,
at one small common sheet-RMS radius. The design is exactly orthogonal and does
not use contact, manual-core or patient-source coordinates. It is run on the
nine opened fit networks with Node only. Its purpose is to estimate paired
network response slopes for the full patient objective, each patient mode, and
each mode's causal direction and monotonicity. It does not select or confirm a
Node field by itself.

Stage-AF completed 270/270 runs. The original training-average analysis found a
positive common direction, but that analysis reused each network both to fit
and to assess the direction. A stricter leave-one-network-out audit supersedes
that interpretation. The held-out direction improved the total soft objective
in 7/9 networks and mode 0 in 8/9, but mode 1 in only 4/9, mode-1 causal
direction in 2/9, and all seven required continuous endpoints in 0/9. The
median held-out joint margin was -0.394. The formal Stage-AF status is therefore
`CROSSVALIDATED_COMMON_DIRECTION_NOT_SUPPORTED`; no candidate may be generated
from the training-average gradient.

This negative result does not yet establish field-family incapacity. A separate
zero-simulation span audit found that the Stage-Z/AF 15-mode (`k<=3`) residual
span captures only 77.0% of the sheet-space difference between the data-driven
`g04-` field and the non-selectable smooth capacity control. More importantly,
that difference has sheet RMS 2.40, whereas Stage-Z stopped at 0.52 and Stage-AF
used 0.18. Thus all data-driven searches so far remained close to `g04-` on the
spatial scale at which Node-only directional capacity was observed.

Stage-AG is one bounded broad-span canary, not an unrestricted restart. It uses
the 35 non-constant uniform-sheet cosine modes with `kx,ky=0..5`, projected onto
the same continuous 18 x 18 spline field. Twelve scrambled-Sobol directions are
paired antithetically at sheet-RMS radii 0.8, 1.4 and 2.0 around `g04-`, together
with the unchanged center. Candidate generation receives no electrode,
patient-source or manual-field coordinates. The manual control is used only in
the preceding non-selectable span-resolution audit and supplies neither a
coefficient, direction nor initialization.

The 25 fields are run on six new fit networks with common random numbers. The
screen retains the full patient soft objective, both mode losses, weakest-mode
causal direction and monotonicity, and source-topology reproducibility and
separation as continuous Pareto axes. Natural KMeans and patient held-out remain
diagnostic or unopened. Stage-AG can nominate at most four fields for fresh
replication; it cannot freeze Node, open intervention, or alter EE, E-to-I or
Z/M. If broad candidates still show no joint patient/direction improvement,
continuous-field optimization closes and the Node mechanism or event target
must be revised before further SNN search.

Stage-AG completed 150/150 runs on six paired fit networks without failed
workers. A post-run network-level audit supersedes the aggregate Pareto
nomination as the scientific readout. None of the 24 perturbed fields had
positive mean utility simultaneously for the total patient objective, both
patient modes and both mode-specific causal directions. The best patient-fit
field, `stage_ag_g10_p`, improved the mean objective by 0.038, mode 0 by 0.064
and mode 1 by 0.017, but worsened mode-1 causal direction by 0.041 on average
and on all 6/6 networks. Conversely, the strongest mode-1 direction field,
`stage_ag_g08_m`, improved that direction by 0.479 while worsening the total
objective by 0.385, mode 0 by 0.338 and mode-0 direction by 0.681. The formal
status is
`BROAD_FIELD_SEARCH_PATIENT_DIRECTION_TRADEOFF_PERSISTS_NO_BALANCED_NODE_CANDIDATE`.

This closes insufficient coefficient span as the leading explanation. The
next experiment may audit the Node mapping itself, but may not add another
continuous-field basis or use the manual capacity field as an initializer.
The first bounded mapping audit tests whether frozen signed per-neuron depth
heterogeneity obscures the patient-derived coarse field. It must preserve the
field, total h-weighted threshold modulation, network topology, EE, E-to-I and
Z/M, and it must include the current mapping exactly as a bitwise-reference
arm. A negative result closes this microheterogeneity explanation rather than
licensing another field search.
