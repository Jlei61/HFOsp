# Topic 4 rev15: M3 coordinate-response atlas for the weak patient mode

## 1. Motivation

Rev14 tested 32 absolute M3 Fourier fields generated from eight observation-free
Sobol directions. The best field improved paired `J14_v1` on two of three
networks, but did not produce a usable two-mode anchor. Equal-network effective
support changed from `A=4.80, B=7.72` for `exact_off` to `A=4.76, B=15.11`.
The best field left all four A-mode distances unchanged or slightly worse while
improving all four B-mode distances. Each network still generated roughly
19--22 classifier-A events, but only 3--6 fell inside patient-A support. The
failure is therefore an A-mode contact-topology mismatch, not an absence of
events and not evidence that the optimizer merely needs more generations.

The rev14 canary spanned only eight directions in a 28-dimensional M3 space.
Rev15 asks the next minimal capacity question:

> Does any individual coordinate of the complete M3 continuous field move the
> frozen Node-only SNN toward the patient A-mode distribution without erasing
> the already accessible B mode?

This is a response atlas and candidate-construction stage, not a convergence
claim. It must be completed before any new CMA-ES run.

## 2. Frozen boundaries

- The event unit, Node mapping, total field mass, signed `d_i`, topology,
  delays, noise and 20-s duration are unchanged from rev14.
- EE, E-to-I and Z/M remain off.
- Patient training A/B labels, training blocks, the frozen classifier and
  training noise floors may be used only by the post-run scorer.
- Candidate coordinates use only the uniform 20 x 20 mm Fourier basis. Contact
  positions, shaft identity, patient events, KMeans, figures and historical
  core geometry cannot generate the atlas.
- Patient held-out and ictal data remain unopened.
- `exact_off` and uniform Node are nonselectable paired references.

## 3. Coordinate atlas

M3 contains 14 paired-phase modes and 28 real coordinates. For each coordinate
`e_j`, construct the absolute fields

\[
s_{j,+}=+0.8\,\widetilde e_j,\qquad
s_{j,-}=-0.8\,\widetilde e_j,
\]

where `widetilde e_j` has unit centered RMS on the same frozen physical-sheet
quadrature used by rev14. After RMS normalization, coefficients are rounded to
13 decimal places before freezing. The 56 selectable fields, uniform Node and
`exact_off` are run on CRN seed 2331, for 58 runs total.

This atlas is not a set of proposed biological cores. A coordinate is a global
Fourier perturbation whose only role is to measure which spatial scales and
phases can alter A-mode recruitment, precedence, profile and event-cloud
structure.

## 4. Training-only response readout

Every run is rescored with the unchanged `J14_v1`. For each mode and each of
the four components, report the paired difference from same-seed `exact_off`.
Also report classifier counts, in-support readable counts, OOD, overlap,
ambiguity, contrast and event support.

Atlas ranking is lexicographic and exploratory:

1. lower A-mode four-component mean than paired `exact_off`;
2. B-mode mean no more than 10% above paired `exact_off`;
3. larger A in-support effective count;
4. lower `J14_v1`, then coordinate ID.

The 10% B tolerance prevents a coordinate that merely exchanges B for A from
being treated as a dual-mode direction. It is one construction rule, not a
claim-level gate. If no coordinate satisfies item 1, retain the nondominated
A/B response set as a negative atlas and do not launch a blind optimizer.

## 5. Combination stage

Only after the atlas is complete, construct combination directions from the
training-only coordinate responses. Freeze the algorithm before simulating
them:

- Let `g_Aj=(D_A(j,+)-D_A(j,-))/(2*0.8)` and define `g_B` identically.
  The dense A direction is `v_A=-g_A`.
- The B-protected direction removes only a harmful first-order B component:
  `v_AB=v_A-max(0,g_B dot v_A)g_B/(g_B dot g_B)`. A beneficial B component
  is retained.
- Sparse top-4 and top-8 directions use the atlas coordinates that actually
  improve A while keeping B within 110% of paired `exact_off`. Coordinates are
  ordered by observed A improvement, and each retained coefficient uses the
  A-improving sign with magnitude `abs(g_Aj)`.
- each direction at centered RMS 0.6, 0.8 and 1.0.

Duplicate or sign-equivalent fields are removed before launch. The
lexicographically best atlas coordinate and the best coordinate that improves
both A and B are included at their native RMS 0.8 as independent-replication
controls. Combination candidates are tested on fresh CRN seeds 2332--2333
together with `exact_off`. Networks are equally weighted. A new combination
must improve on both fresh networks; a single-coordinate control is judged on
all three networks including construction seed 2331. Effective support must be
at least six for both modes. Natural KMeans and held-out remain unavailable.

## 6. Interpretation

- A reproducible A-improving coordinate set means the original eight Sobol
  directions missed relevant M3 capacity.
- Coordinates that improve A only by destroying B show a static-field tradeoff,
  not dual-mode recovery.
- No A-improving coordinate or combination closes only this bounded M3 atlas;
  it does not reject continuous fields or justify activating connections.
- EE, E-to-I and Z/M remain blocked until one static Node candidate passes the
  post-freeze Fig.4 two-cluster acceptance contract.

## 7. Multinetwork response revision after the combination result

The 30-run combination replication completed without engineering failures, but
no candidate met all three training-only requirements. Five combinations
improved A on both fresh networks, yet none protected B on both networks and
none simultaneously retained equal-network effective support of at least six
for both modes. The best mean response (`dense_a`, RMS 1.0) reduced A by 0.345
and B by 0.032, but B worsened on seed 2333 and A support was 5.77. Therefore
the seed-2331 response gradient cannot be treated as a network-invariant field
direction.

The next bounded experiment repeats the same 56 signed coordinates, uniform
Node and `exact_off` on seeds 2332--2333. It adds no new field family and no new
patient input. The three-network response tensor will report, per coordinate:

- the sign and magnitude of A and B changes on every network;
- sign consistency and worst-network change;
- all four A/B distance components and effective support;
- a robust direction constructed only if the coordinate responses support it.

A later robust direction must be built from all three training networks and
then frozen before running on new selection networks. Natural KMeans, patient
held-out data and figures remain unavailable during construction. Failure of
the full coordinate tensor to yield a common A-improving/B-protecting direction
will close this local M3 response strategy; it must not be rescued by relaxing
the support threshold or activating EE, E-to-I or Z/M.

## 8. Post-selection and one-time final Node audit

Any response-derived direction is frozen before simulation on seeds
2341--2343. Training-only progression requires, on all three paired networks,
lower A loss than `exact_off`, B loss no greater than 110% of `exact_off`, and
equal-network effective support of at least six events for each mode. The field
is not yet frozen as the final Node substrate at this point.

The same complete, returned and temporally isolated causal families are then
used for the post-selection figures. Every network must contain both supervised
patient modes. Natural KMeans is fit to masked normalized ranks, independently
of the full-onset classifier; acceptance requires AMI at least 0.8 in all three
networks and a pooled patient-profile matrix with positive diagonal and
negative crossed cells. Natural KMeans cannot re-rank fields.

Only after this acceptance may the developmental held-out endpoint be opened
once. The frozen candidate is compared with same-seed `exact_off` using:

1. positive complete-event-cloud held-out `R2` that improves on `exact_off`;
2. lower held-out weakest-mode loss;
3. lower held-out A and B losses separately;
4. mode-conditioned source topology whose weakest-mode cross-network
   reproducibility times between-mode separation exceeds the 95th percentile
   of a within-network label-permutation null preserving network identity and
   mode occupancy.

Training-target losses remain diagnostics and cannot satisfy these held-out
clauses. Passing all four clauses permits a same-checkpoint hotspot
intervention; it does not itself freeze Node. The intervention is crossed:
each mode-derived hotspot is applied from identical checkpoints to both A and
B events, together with sham and its matched off-template location. Node is
frozen only if a mode hotspot affects its predicted event more than the
opposite-mode event and the matched spatial control on at least two of three
networks. Event survival and latency are primary ordered endpoints; rank and
source-topology displacement explain surviving events without adding separate
claim gates. EE, E-to-I and Z/M remain off throughout.
