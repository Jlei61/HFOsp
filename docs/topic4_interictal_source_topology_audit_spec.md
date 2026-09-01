# Topic 4 interictal source-topology audit

## Motivation

The frozen Figure 4 readout shows two reproducible contact-rank clusters, but a
rank reversal does not identify a unique propagating source.  The same contact
orders can arise when two or more spatially separated populations are coupled
by a shared network state and differ only in relative phase.  This audit tests
that alternative explanation before the model is described as reproducing a
single causal propagation route.

The patient Supplementary Video 1 and the current model-neuron GIF are not
direct visual comparators.  The patient video uses a 3 ms contact-envelope
average every 2 ms with 6 mm spatial smoothing and per-event normalization.
The model GIF uses raw E-neuron activity integrated over 10 ms with much finer
spatial smoothing.  Observation matching therefore precedes topology or
causality claims.

## Frozen questions

1. How much held-out patient event variance is captured by the model modes?
2. Does each model event have one spatial nucleation component followed by a
   continuous front, or several disconnected early components?
3. Is a one-source onset-time model sufficient, or does a two-source model
   improve out-of-sample prediction?
4. Does selective suppression of the earliest component alter event occurrence
   and contact order more strongly than suppression of secondary or matched
   control components?

## Observation-matched replay

For the accepted seed-1569 Node-only MTA/MTB pair, produce two synchronized
views without changing the simulated trajectory:

- contact-space view: 2 ms biological step, 3 ms frame average, fixed
  per-event participant q99 normalization, and the same 6 mm display smoothing
  as Supplementary Video 1;
- neuron-space event-excess view: 1 mm bins, 3 ms spike counts, each bin
  standardized against its own pre-event distribution.  Raw background
  activity is not interpreted as event recruitment.

The patient and model videos remain different observables.  Observation
matching makes their temporal and spatial resolution comparable; it does not
turn model current into clinical SEEG.

## Source-topology endpoints

For every clean event in all 12 frozen networks:

- `n_early_components`: 8-connected components among the earliest 10% of
  recruited 1 mm bins;
- `dominant_early_component_fraction`: fraction of earliest bins in the largest
  component;
- `front_continuity`: fraction of newly recruited bins adjacent to the already
  recruited set;
- `single_source_cv_r2`: cross-validated onset-time variance explained by
  `t(x)=a+distance(x,source)/v`;
- `two_source_delta_cv_r2`: out-of-sample gain from
  `t(x)=a+min(distance(x,s1),distance(x,s2))/v`;
- the same summaries separately for MTA and MTB, with network seeds as the
  independent units.

No single threshold is a blocker in this exploratory audit.  The endpoint
distributions and one-source versus two-source predictive comparison are the
result.

## Same-checkpoint causal replay

Only after the descriptive audit identifies stable early components, branch
from the same pre-event checkpoint and the same random state:

1. suppress the earliest component;
2. suppress the second component;
3. suppress a matched off-component region;
4. sham replay.

Compare event probability, latency, frozen mode label, contact-rank profile,
and early recruitment field.  A single causal source requires the earliest
component intervention to dominate the secondary and matched controls.  If a
different hotspot takes over with preserved mode identity, the result supports
a coupled multi-source substrate instead.

## Claim boundary

KMeans stability and contact-rank similarity establish reproducible kinematic
patterns only.  `single_source_propagation` is not used unless onset topology,
predictive source-model comparison, and same-checkpoint intervention agree.
Otherwise the accepted description is `phase-coupled multi-source event mode`.
