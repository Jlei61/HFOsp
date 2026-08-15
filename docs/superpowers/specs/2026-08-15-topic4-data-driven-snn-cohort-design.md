# Topic 4 data-driven SNN cohort: patient-conditioned continuous substrate

## Scientific question

The current rev11-NLC result is an E1146 development result. This cohort phase
asks whether the same data-driven model family can recover patient-consistent
interictal propagation repertoires across the masked stable-K=2 cohort.

The scientific unit is the subject. Events and network seeds are nested
replicates and must not be counted as independent cohort observations.

## Denominator contract

The upstream masked cohort contains 34 stable-K=2 subjects. Three separate
eligibility layers are frozen:

1. `PATIENT_TARGET_ELIGIBLE`: masked ranks, two train-derived modes, disjoint
   recording-block held-out data and at least six joint-valid contacts;
2. `CANONICAL_SHAFT_LAYOUT_ELIGIBLE`: every target contact name can be parsed
   into shaft identity and within-shaft ordinal, without reading event ranks;
3. `REAL_GEOMETRY_SENSITIVITY_ELIGIBLE`: real 3-D contact coordinates support
   a rank-2 geometry-only projection into the SNN sheet.

Every one of the 34 subjects remains in the primary canonical-layout analysis.
Real geometry is a sensitivity stratum, not the primary denominator. The live
audit found all 34 contact sets parseable, while only 28/34 have usable real
3-D geometry. The headline must therefore state `34 canonical-layout subjects;
28 real-geometry sensitivity subjects`, and must never imply that the
canonical layout is patient anatomy.

## Patient target

For each subject, recording blocks are split before KMeans. KMeans is fitted on
masked normalized ranks from training blocks only. Non-participating contacts
are midpoint-imputed only for clustering; recruitment masks remain explicit in
all scientific distances. Held-out blocks are transformed by the frozen train
centres and never refit.

The two train clusters are named TA/TB by maximum correlation with the already
frozen masked rank-displacement templates. This naming step cannot change
cluster membership. The per-mode target retains:

- contact recruitment probability;
- masked normalized rank profile;
- unordered-pair precedence states;
- a fixed event sample for event-cloud and KMeans checks;
- train and held-out recording-block counts;
- train-to-held-out prototype correlation and OOD distance.

`PATIENT_TARGET_ELIGIBLE` requires both modes to have at least 20 events on
both sides of the block split and a positive same-mode minus crossed-mode
held-out margin.

## Geometry and observation

The primary observation model is a canonical shaft layout. Contact names are
parsed into shaft identity and within-shaft ordinal; shafts occupy fixed
parallel rows and the ordinal axis is stretched to fill the usable sheet, which
mirrors how the real-geometry projection stretches its largest-variance axis.
A fixed physical contact pitch was rejected before any cohort simulation ran:
it left low-ordinal-span montages inside a 2-mm strip while their real-geometry
counterparts spread over 16 mm, so the canonical-versus-real contrast would
have confounded contact arrangement with montage extent on the very axis that
carries the contact-order claim. The layout may not read rank values, mode
labels, template direction, event counts, recruitment frequency or field peaks.
This supports a claim about internal contact-order structure, not anatomical
localization.

For the 28 subjects with usable coordinates, real 3-D geometry is projected by
deterministic PCA and isotropically fitted into the 20-mm sheet as a sensitivity
analysis. The cohort conclusion is observation-invariant only if the canonical
and real-geometry subject effects agree in direction. Disagreement is reported
as `OBSERVATION_LAYOUT_DEPENDENCE_UNRESOLVED`.

The same frozen readout kernel and detector are used for every candidate and
subject. Contact count may vary by subject; no E1146-specific shaft or
15-contact assumption is allowed in the cohort scorer. Recruitment errors are
first averaged within shaft and then across shafts. Precedence errors are first
averaged within unordered shaft-pair class and then across classes, so one
densely sampled shaft or a large cross-shaft pair count cannot dominate the
fit. A one-shaft subject is eligible for within-shaft endpoints, while its
cross-shaft endpoint is explicitly not applicable.

## Data-driven model family

All subjects share one candidate library generated before patient scoring:

- a continuous full-sheet Node field represented by tensor B-splines plus
  low-frequency spectral residuals;
- continuous local E-to-E and E-to-I source-flow redistribution;
- fixed topology and delays;
- incoming excitatory weight conserved per target and pathway;
- no component count, peak count or contact-conditioned field builder.

Each subject selects a candidate using only its patient-training target and the
fit/selection network pools. The selected candidate is then frozen for that
subject and evaluated on disjoint patient held-out blocks and fresh network
seeds. This is patient-conditioned selection from a shared family, not 34
unconstrained bespoke optimizations.

## Primary endpoints

The subject-level primary endpoint is the weakest-mode held-out patient
alignment, combining contact-rank profile, recruitment and precedence. The
same model readout is also scored against within-subject contact-identity
permutation nulls.

The within-shaft permutation null is requested at 64 draws, but four subjects
do not have 64 distinct within-shaft permutations: `epilepsiae_583` has 3,
`epilepsiae_1073` has 5, `epilepsiae_1077` has 11 and `epilepsiae_253` has 23.
For those the whole non-identity group is enumerated, giving an exact but
coarse null whose smallest reachable p-value is 1/4, 1/6, 1/12 and 1/24. The
effective null size and that floor are stored per subject, and the cohort
adjudication reports how many subjects could not reach conventional
significance on the null test alone. Padding these nulls back to 64 rows by
drawing with replacement is forbidden: it would advertise a resolution the
montage does not have.

Natural KMeans is mandatory but separate. It must show that the same network
contains two reproducible event clusters and that, after one-to-one alignment,
both clusters have positive patient-template margins. A supervised positive
matrix alone is not a KMeans pass.

Cohort inference reports subject-level medians and hierarchical intervals, but
the sign/Wilcoxon or subject bootstrap resamples subjects only. A positive
cohort claim requires:

1. patient alignment better than the matched contact-permutation null;
2. at least 60% of SNN-eligible subjects pass the frozen subject endpoint;
3. no result is driven only by pooled event counts or one high-yield network;
4. same-network K=2 support is reported separately from supervised geometry.

## Execution stages

1. build the 34-subject target and geometry eligibility audit;
2. run the frozen six-subject real-geometry transfer canary as a capacity
   preflight, not as the cohort result;
3. freeze the canonical shaft layout and shaft-balanced scorer for all 34;
4. audit memory, detector parity, event support and Fig.4-style output;
5. freeze the shared library and formal budget;
6. run all 34 canonical layouts plus the 28 real-geometry sensitivity layouts
   under managed systemd/nohup workers;
7. aggregate automatically, render PNG/PDF and notify on completion.

The formal run cannot begin while the rev11 pathway confirmation still owns
the memory budget.

## Figure contract

The final result is one cohort figure with:

1. the 34-subject canonical-layout cohort and 28-subject real-geometry
   sensitivity denominator;
2. subject-level held-out patient alignment versus matched null;
3. subject-level natural KMeans support and same-network dual-mode rate;
4. one median-performing subject shown with the accepted Fig.4 direct readout;
5. its paired Fig.4 KMeans heatmap, rank profiles and model-patient matrix.

The representative subject is chosen by distance to the cohort median before
rendering, not by the best score. The canvas and README must state the exact
scientific verdict and denominator.

## Claim boundary

A positive result may support:

> Across stable-bidirectional subjects, patient-conditioned
> continuous Node plus local-connectivity SNNs recovered held-out propagation
> geometry and same-network stereotyped event structure more closely than
> matched contact-identity nulls.

This wording is allowed only if the frozen subject endpoint passes in the
canonical-layout cohort and the real-geometry sensitivity does not contradict
it. It may not be shortened to “34 patients were reproduced”; the primary
layout is not anatomy. It does not establish clinical waveform identity, a
unique anatomical core, patient-blind generalization or seizure dynamics.
