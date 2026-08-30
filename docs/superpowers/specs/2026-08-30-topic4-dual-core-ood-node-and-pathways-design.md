# Topic 4: OOD-guided dual-core Node recovery and pathway factorization

## Motivation

The current data-driven SNN work established that a frozen Node substrate can
produce two clusters of returned interictal event-like activity, but the
continuous field remains difficult to interpret and its weakest patient mode is
not recovered well. The historical two-core comparison is not a capacity test:
its centers were hand placed and never fitted under the current patient event
distribution or OOD classifier.

This round asks two ordered questions:

1. Can a strictly two-core, Node-only field recover the two patient interictal
   propagation modes with a low fraction of events outside the frozen patient
   support?
2. After that Node field is frozen, how do the previously learned local E-to-E
   and E-to-I redistributions change mode occupancy, support and OOD?

The first question must close before the second starts. Z/M is off throughout.
The result is development-only because the E1146 target and earlier model
results have already been inspected.

## Frozen patient target

- Use the existing 15-contact shaft-aware embedding and the frozen two-mode
  direction classifier without refitting.
- The patient modes are the same TA/TB modes used by the current Fig.2C
  contract. The locked illustrative exemplars are used only for visual
  comparison after model selection.
- The patient classifier assigns every readable model event to A or B and uses
  the frozen mode-specific Mahalanobis q99 thresholds for OOD.
- No patient ictal data, Z/M trajectory or Fig.5 endpoint is read in this round.
- The frozen E1146 shared-plane geometry and both TA/TB template-geometry files
  used to place virtual contacts are explicit hash-checked inputs; none is an
  implicit worktree-relative file.
- The network-cache source audit that fixes the engine commit, NumPy version and
  cache key is also an explicit hash-checked input.

## Two-core Node family

The field is an exact binary Node mask with two centers on the 20 x 20 mm SNN
sheet. For each network, the `N_core` E neurons nearest to either center are
selected. This gives two equal-depth cores and an exact excitability budget;
`N_core` controls their common effective radius. The signed frozen `d_i` values
are unchanged:

```text
Vtheta_i = Vtheta_0 - h_i d_i,   h_i in {0, 1}.
```

Search parameters are only:

```text
(x1, y1, x2, y2, N_core).
```

The centers are canonically ordered, remain inside the sheet, and must be
4--18 mm apart. Candidate generation also rejects geometries for which either
core receives less than 25% of the budget on a uniform reference sheet. There
is no orientation, covariance, filament, third component or contact-centered
prior. The historical hand dual core is included as an explicit anchor.

## Primary metric

For every detector-qualified event that returns to baseline:

```text
in_support = readable_at_3_or_more_contacts AND not_frozen_OOD
OOD_all_returned = 1 - n(in_support) / n(all_returned).
```

Thus an unreadable local event cannot disappear from the denominator. Also
report the historical readable-only OOD for continuity.

OOD is the primary continuous endpoint, but OOD alone can be gamed by producing
only one easy mode. Candidate ranking is therefore lexicographic:

1. both frozen patient modes are represented among in-support events;
2. lowest equal-network `OOD_all_returned`;
3. lowest weakest-mode normalized distance within the frozen shaft-aware
   patient support;
4. highest equal-network returned-event support.

The both-mode criterion is a representation requirement, not a demand for 20
events per network. Event-poor or one-mode candidates remain visible in the
tables and are not silently discarded.

## Search and confirmation

### Screen

- 48 scrambled Sobol candidates plus the historical hand-core anchor;
- two common-random-number network seeds;
- 8 s per network;
- no EE, E-to-I or Z/M modulation.

### Selection

- six mechanically selected candidates;
- three new common-random-number network seeds;
- 12 s per network.

### Confirmation

- one frozen candidate selected before confirmation;
- 12 new network seeds;
- 20 s per network;
- equal-network estimates and 4096-draw network bootstrap intervals.

The confirmation report must include OOD, A/B support, mode proportion,
recruitment, three precedence classes, separate shaft profiles, event-cloud
distance and natural KMeans agreement. Patient held-out recording blocks are a
development evaluation, not a new blind test.

## Fig.2C-style model movies

The frozen candidate is replayed without changing parameters. A 5 ms spatial
spike-count grid is stored for all confirmation seeds so the visual seed and
events can be selected algorithmically after the runs.

Select one network containing in-support returned events from both modes, then
select the lowest-OOD event in each mode subject to at least six ICL and two SCL
contacts. Produce:

- one side-by-side TA/TB GIF of the SNN E-neuron activity field using viridis;
- synchronized 15-contact virtual readout and cursor;
- frozen patient Fig.2C rank order as a static reference, not as a model field;
- first, middle and last frame PNGs plus metadata.

The GIF is a visual acceptance check. It cannot rescue a candidate that loses
the quantitative OOD or two-mode result, and it is not clinical SEEG.

## Frozen pathway factorization

Only after the Node candidate is frozen, run 12 new paired network seeds under:

```text
Node
Node + learned E-to-E redistribution
Node + learned E-to-I redistribution
Node + both redistributions
```

Reuse the previously frozen coefficient rows and local-connectivity mapper.
Topology, delays and each target's incoming pathway budget remain fixed. This
is a transfer/factorization experiment on the new Node field, not a pathway
refit. Primary endpoint remains `OOD_all_returned`; mode shares, natural KMeans,
returned-event rate and weakest-mode distribution errors are secondary.

Statistics use network seed as the independent unit. Report paired differences
and 90% network-bootstrap intervals. Pooled events are descriptive only.

## Claim boundary

Allowed:

> A fitted two-core Node field generated two returned event clusters and its
> event-level support relative to the frozen E1146 interictal distribution was
> quantified; frozen local E-to-E and E-to-I redistributions then produced
> model-internal changes in that repertoire.

Not allowed: anatomical core recovery, patient-blind generalization, clinical
wave reproduction, E-to-I causality in the patient, or seizure mechanism.
