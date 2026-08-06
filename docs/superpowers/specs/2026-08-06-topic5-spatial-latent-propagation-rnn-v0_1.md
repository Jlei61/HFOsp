# Topic 5 — Patient-specific Spatial Latent Propagation RNN v0.1 (SLP-RNN)

- **Status:** `OPEN_SCIENTIFIC_DISCOVERY_SPEC` — design frozen 2026-08-06, before any run
- **Output root:** `results/topic5_spatial_latent_propagation_rnn_v0_1/`
- **Branch:** `codex/topic5-spatial-latent-propagation-rnn-v0-1`
- **Interpreter:** `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python` (torch 2.5.1+cu124)
- **Geometry status:** `RETROSPECTIVE_GEOMETRY_PILOT` (see §3.3)

## 0. Revision log

**rev3 (2026-08-06, late execution)** — five amendments, each forced by something the
run surfaced rather than by a change of mind:

1. **H3 is not blocked.** rev2 read the recovery gate as removing three hypotheses. It
   removes two. Whether *learning* the connections predicts better than fixing them to
   nearest neighbours never needs the connections to be identified, only whether the freedom
   to choose them pays. H4 and H5 stay blocked; H3 is reported with its scope stated.
2. **The flow-ordering layer gets its own analysis (§7.3 below).** The gate certified it, and
   it is the only structural question left askable. It needs a second seed, so it could not
   run until seed 2 landed.
3. **Two controls added.** A geometry shuffle, which trains the learned arm against permuted
   node positions in the connection-cost term only, to test whether the spatial prior
   constrains anything; and a fully connected latent arm as a ceiling, to separate the cost
   of sparsity from the cost of the parameterisation.
4. **Batch size scales with the patient.** A fixed batch made an epoch one gradient update
   for a patient with 249 events and 68 for one with 69,000, so the epoch budget bought
   wildly different amounts of optimisation. Every patient now gets at least eight updates
   per epoch. Units fitted before this were discarded, not resumed.
5. **Arms are paired only on seeds they share.** Seeds land unevenly; a two-seed median
   against a one-seed value compares amounts of averaging as much as models.

### 7.3 The flow-ordering readout [rev3]

Licensed by the gate's third layer and by nothing else. For each patient the learned graph
gives, per tissue unit, how far along the axis its outgoing influence reaches. Refitting the
same patient from a different starting point tests whether that ordering is stable;
comparing different patients — restricted to the same contact-count band, both interpolated
onto a common normalised plane — tests whether it is theirs. Reported as relative ordering
and never as the graph. Montage size must be matched, or patients that differ only in
electrode count will look like patients that differ.

**rev2 (2026-08-06, during execution)** — the identifiability precondition H0 returned its
verdict before any patient result was read, and it removes three hypotheses from this run.
Four amendments, all recorded before the cohort was aggregated:

1. **H3, H4 and H5 are not answerable by this parameterisation.** On events generated from a
   known sparse spatial graph, the fitted adjacency ranks the true connections at median AUC
   0.482 against a pre-set floor of 0.60, and the global direction of travel comes out right
   in 3 of 7 runs against a floor of 0.80. Sweeping the observation ratio from 0.25 to 0.75,
   the wiring strength from 0 to 10, and the hidden width from 4 to 16 leaves all of them at
   chance. Any per-patient graph is therefore one arbitrary member of a large equally-fitting
   set. §7 tiers stand as written; the verdicts are `BLOCKED_BY_RECOVERY_GATE`, not negative.
2. **A third structural layer is added and is reportable.** Node-level *ordering* — which
   patches push activity further along the axis than others — recovers in 7 of 7 runs
   (median ρ = +0.39, sign test p = 0.008). It may be reported as relative ordering and never
   as the graph.
3. **Node count is a lower bound, not the value.** The supplied `min(64, max(24, 4C))` was
   written for the median montage; at 38 and 52 contacts it leaves nodes sparser than
   contacts and a contact reading a single node, which silently restores a per-contact
   parameter. The count now grows until every contact averages at least three nodes.
   Three patients needed it.
4. **Training schedule and batch size are fixed for fairness, not for speed.** Arms with no
   learnable graph train as one phase under the same stopping rule; the batch is 1024 with a
   matched learning rate, verified to reach the same validation loss as 256 on the
   representative patient. Every run records whether it converged or hit the epoch ceiling
   while still improving, and a run that hit the ceiling carries no negative verdict.

**rev1 (2026-08-06)** — first draft, transcribed from the user-supplied design and then
amended with six P1 fixes found by auditing the design against the frozen state of this
repository. Each fix is marked **[P1-n]** at its section. The user accepted three deviations
from the supplied design before any of this was written; they are recorded in
`INPUT_MANIFEST.json → accepted_deviations_from_pasted_plan` and summarised in §2.4.

## 1. Frozen scientific question

The task is unchanged from the existing Topic 5 interictal line:

> observed rank sets → next rank set of contacts, plus STOP.

The question is not "can an RNN predict the next contact". That is settled. The question is:

> On a patient's own physical contact plane, can local rate units — placed in tissue rather
> than at electrodes, and observed only through a fixed local virtual-SEEG kernel — form a
> patient-specific mesoscopic effective propagation graph, under an autoregressive objective
> and two generic wiring rules, that predicts held-out interictal propagation **at contacts
> the model never trained on**?

The load-bearing novelty is the last clause. Every earlier model in this line carries one
free parameter per contact, so a contact absent from training has no parameters and cannot
be predicted at all. Putting the state in space and making contacts pure observation ports
is what first makes that question askable.

## 2. What is already settled, and what this does not re-litigate

`results/topic5_rnn_overall_acceptance/FINAL_ACCEPTANCE.json` (2026-07-28) froze this line as
`ACCEPTED_AS_BOUNDED_SUPPLEMENTARY_COMPUTATIONAL_RESULT`. The relevant entries:

| already established | value |
|---|---|
| static contact participation topography reproduces across a chronological split | Spearman 0.893, 34/34 patients, p=1.8e-7 |
| ordered history helps over the last rank set alone | h2−h1 +0.017 (32/34), h3−h2 +0.011 (29/34) |
| the order itself carries information | ordered h3 − matched shuffle +0.026 (27/34), p=0.0036 |
| unbounded history adds nothing over history-3 | −0.00095, p=0.44 |
| **structured / path / axis / competition / source-direction mechanisms** | `NOT_SUPPORTED_BY_CURRENT_MODEL_FAMILIES` (6/6 sub-tests fail) |

The later v0.3 run (`topic5_patient_specific_source_conditioned_rnn_v0_3_final`,
306/306 units, 34 patients) adds the decisive baseline fact:

| comparison | median Δ contact NLL | patients better | p |
|---|---|---|---|
| structured − static | **+0.100** | 34/34 | 1.2e-10 |
| structured − ordinary GRU | **−0.121** | **0/34** | greater-p 1.0 |

So recurrence is worth a great deal over a static topography, and it is **not** confined to
local transitions — but every *structured* prior tested so far has cost accuracy relative to
an unconstrained GRU.

**Consequence for this spec [P1-1].** A model matrix whose only baseline is `STATIC_CONTACT`
would let a spatial model look successful while losing to a plain GRU by a wide margin.
`ORDINARY_GRU` is therefore a **mandatory arm in every primary comparison**, not an optional
control. No hypothesis in §7 may be declared supported on a static-only contrast.

### 2.4 Accepted deviations from the supplied design

| id | supplied design said | measured state | disposition |
|---|---|---|---|
| D1 | 31 primary patients | 31 is the coordinate-free cohort; with physical coordinates and exact-name alignment the cohort is **21** | accepted by the user 2026-08-06; run on the shared-axis cohort |
| D2 | "the patient's 2D propagation plane" | 9 of 21 planes lose ≥20% of transverse spread to the 2D projection | accepted; 2D stays primary, 3D placement is a pre-registered sensitivity arm (§4.5) |
| D3 | contact-node graph RNN as stage 1 | overlaps the persistent path-mode graph RNN frozen 2026-07-28 on the `do_not_tune` list | accepted; retained as a **baseline arm**, and no novel claim may rest on it |

## 3. Data contract

### 3.1 Events

Source: `results/topic5_interictal_rank_distribution/dataset_v0_4` via
`src.topic5_shared_propagation_field.load_subject_rank_events`. The record's manifest must
certify `target_values_read=false` and `ab_or_kmeans_labels_read=false`; the loader raises
otherwise. For patient `p`, event `e`:

- `group_ids[e, c] ∈ {−1, 0, 1, …}` — the rank step at which contact `c` is recruited, or −1;
- rank sets are the level sets of `group_ids[e, ·]`; ties inside a rank set are real and kept;
- a contact is recruited at most once per event.

Model input at step `t`: current rank multi-hot `x_t`, cumulative recruited mask, rank index.
Model output: `x̂_{t+1} ∈ [0,1]^{C_p}` and `p̂_STOP`.

**Never inputs:** A/B or KMeans labels, seizure labels, early-ictal fields, SOZ, SNN weights or
core positions, any static propagation template as a second prediction path.

### 3.2 Splits [P1-2]

The supplied design said "reuse the frozen train/heldout split". The repository's own record
contract states that `old_heldout20` *"has already been read by earlier RNN development and is
therefore not used for SPF-RNN model selection"*. Reusing it would make this run
non-confirmatory by construction.

Frozen rule:

- all optimisation, model selection and hyper-parameter choice use
  `record.development_split(0.15, 0.15)` **inside train80** → `train / validation / test`;
- `validation` is for early stopping and config selection only;
- `test` is untouched until the single formal config is frozen;
- `old_heldout20` is **burned**. It may be reported once, explicitly labelled
  `PREVIOUSLY_READ_NOT_CONFIRMATORY`, and may not carry any hypothesis verdict.

### 3.3 Geometry and its provenance

Source: `results/spatial_modulation/propagation_geometry/observation_readout/real_subjects`
(the `narrow` montage). One montage tree for the whole cohort — mixing trees inside a cohort
breaks the montage-consistency clause `src/sef_hfo_subject_placement.py` already enforces.
`narrow` is chosen because after exact-name intersection with the event contacts it strictly
contains `broad` (15 of 21) and carries the higher transverse PC1 on most patients.

Contact `c` gets `s_{p,c} = (along_axis_mm, signed_transverse_mm)`, plus the raw
`coord_mm` 3-vector for the sensitivity arm.

**Every plane in this cohort was estimated from the full recording**, not from train events
only. All outputs are therefore stamped `RETROSPECTIVE_GEOMETRY_PILOT`, and no result may be
described as prospective predictive geometry. A train-only axis re-fit is deferred to v0.2.

### 3.4 Alignment

Exact name match only, no fuzzy join. The usable contact set for patient `p` is the ordered
intersection of the event record's `contact_names` and the plane's channel names. A patient
enters the cohort at `≥ 8` jointly resolved contacts — below that the observation operator has
no interior and leave-one-contact-out leaves no field behind.

### 3.5 Frozen cohort

From `INPUT_MANIFEST.json`, fixed before any run:

- **primary: n = 21**, joint contacts min 8 / median 15 / max 52;
- pre-registered stratum **planar, n = 12** (transverse PC1 ≥ 0.80, not 1D sampled);
- pre-registered stratum **well-sampled, n = 18** (`n_events ≥ 2000`).

Both strata are declared now because an earlier Topic 5 RNN comparison changed sign once
low-support patients were removed. Primary statistics are reported on all 21; both strata are
reported alongside, always, whatever the direction.

**Development patients, fixed a priori:** `epilepsiae_1146` (repository's standing
representative subject), `epilepsiae_958` (large, planar, 16 contacts),
`yuquan_zhangkexuan` (Yuquan, non-planar, 26 contacts). Config selection reads only their
`validation` partitions, so they remain in the formal cohort without contaminating `test`.

## 4. The spatial latent field

### 4.1 Nodes

`M_p = min(64, max(24, 4 C_p))` latent rate units placed by farthest-point sampling inside the
patient's valid 2D domain (the convex region spanned by that patient's contacts, dilated by
the kernel width). Node positions are fixed once, with a recorded sampling seed. A node is a
coarse-grained rate state of a small patch of tissue, not a neuron.

### 4.2 Observation operator

```
H_{p,cm} = K_σ(‖s_{p,c} − r_{p,m}‖) / Σ_{m'} K_σ(‖s_{p,c} − r_{p,m'}‖)
```

with `K_σ` the continuous Gaussian kernel already used by the SNN virtual-SEEG readout,
truncated at `3σ` so support stays local. Rows sum to 1. `σ` is fixed per patient from the
existing readout, never fitted. Discretised 4–5 mm binning is forbidden.

### 4.3 Tied input and readout

```
u_t = Hᵀ x_t                    (observation injected into the field)
a_m = softplus(w_aᵀ h_m)        (node emission)
ℓ_{t+1} = b + α · H a           (contact logits)
```

The same `H` on both sides. No dense contact-to-contact path may exist anywhere in the graph;
if one did, the latent field could be bypassed and every structural claim would be vacuous.

### 4.4 Local microsteps

`K` internal microsteps per observed rank transition. Observation is injected at `k=0` only:

```
q_m^{(k)} = Σ_{n≠m} A_{nm} φ(h_n^{(k)}) / (Σ_{n≠m} |A_{nm}| + ε)
h_m^{(k+1)} = GRU_θ(h_m^{(k)}, [v_m^{(k)}, q_m^{(k)}]),   v^{(0)}=u_t, v^{(k>0)}=0
```

**[P1-4]** `K` and the wiring cost are not independent knobs. With `M_p` nodes over the plane,
`K` hops reach roughly `K` node spacings; a rank transition between two contacts further apart
than that can only be served by a long edge, which the wiring cost penalises. A sweep over
`K ∈ {1,3}` alone therefore confounds "too few hops" with "wiring cost too strong". Two
requirements follow:

1. the development sweep uses `K ∈ {1, 3, 6}`;
2. every run logs a **hop-reachability diagnostic** — the fraction of observed rank transitions
   whose two contacts' nearest nodes are within `K` hops on the thresholded graph. A
   configuration whose reachability is below 0.5 is reported as hop-limited, and a negative
   result from it may not be attributed to the wiring economy.

### 4.5 Sensitivity arm: 3D placement

`coord_mm` is available for every cohort patient. A pre-registered sensitivity arm places
nodes in 3D and uses 3D distances in both `H` and the wiring cost, with everything else
identical. It is not the primary. It exists to bound how much the 2D projection costs, which
matters because 9 of 21 planes discard ≥20% of transverse spread.

## 5. Learnable graph and wiring economy

```
A_{nm} = g_{nm} · tanh(w_{nm}),  n ≠ m,   g_{nm} ~ HardConcrete(α_{nm})
```

Edges may be positive or negative and are interpreted only as facilitatory or suppressive
*effective* influence — never as cellular excitation or inhibition.

Two generic rules, and only these two:

```
L_edge = ( (1/M_p) Σ_{n≠m} E[g_{nm}] − k_target )²            # connections are limited
L_wire = (1/M_p) Σ_{n≠m} E[g_{nm}] · |w_{nm}| · d̃_{nm}         # long connections cost more
```

`d̃` is distance normalised by the median inter-node distance. Distance enters the wiring loss
only — never a node feature, never the output head.

**Forbidden in the loss:** modularity, small-worldness, A/B routes, source/sink structure,
hubs, axis-aligned directionality, the SNN connectivity kernel, any static pathological field.
These may only ever be post-hoc observations.

## 6. Objective and structure formation

```
L = L_next + λ_stop L_stop + λ_wire L_wire + λ_edge L_edge
```

`L_next` is multi-label BCE over the next rank set; `L_stop` is BCE on STOP. Secondary and
always logged: cardinality-conditioned contact NLL, for comparability with the v0.3 line.

Three phases: (1) functional warm-up, gates open, `λ_wire ≈ 0`; (2) structure formation,
wiring penalty ramped in, Concrete temperature annealed; (3) topology freeze at
`P(g>0) > 0.5`, mask fixed, retained weights fine-tuned. The final graph must come from task
optimisation plus wiring economy, not from post-hoc magnitude pruning.

## 7. Hypotheses, with pre-registered tiers

Each runs and is reported independently. There is no composite all-or-none gate.

| id | statement | tier | primary contrast |
|---|---|---|---|
| **H0** | on SNN-generated virtual-SEEG events the learner recovers the known main propagation direction and lesion-sensitive path, without reading SNN connectivity | **identifiability precondition** | recovery vs chance |
| **H1** | recurrence beats a static contact topography | replication, not novel | `CONTACT_GRAPH_RNN` − `STATIC_CONTACT` |
| **H1b** | any graph/latent arm also beats an unconstrained GRU | **[P1-1] mandatory guard** | arm − `ORDINARY_GRU` |
| **H2** | the latent substrate predicts contacts it never trained on | **primary cohort claim** | leave-contact-out: `LATENT_LEARNED` − `CONTACT_GRAPH` |
| **H2s** | recovered structure is stable under sensor subsampling | secondary robustness | 100 / 80 / 60 % contacts |
| **H3** | a learned graph beats a fixed local graph at equal edge budget | secondary | `LATENT_LEARNED` − `LATENT_FIXED_LOCAL` |
| **H4** | graphs are patient-specific | secondary | within-patient vs between-patient similarity |
| **H5** | the specific topology is functionally necessary | secondary | targeted vs matched-random lesion |
| **H6** | modes A/B use different dynamic routes on one substrate | optional, post hoc | message-flow contrast |

**[P1-6]** The supplied design gated H2 on "leave-contact-out **or** sensor subsampling". A
disjunction of two tests inflates the false-positive rate and encodes the desired conclusion
rather than a decision rule. Split as above: leave-contact-out is the single primary for H2;
sensor subsampling is H2s, reported always, gate-bearing never.

### 7.1 Leave-contact-out, made well-posed [P1-3]

The readout is `ℓ = b + α H a`, and `b` is a free per-contact parameter. A contact held out of
training has no trained `b`, so as written the test is undefined — and silently defaulting it
to zero would bias the comparison in an uncontrolled direction.

Frozen rule: the leave-contact-out arm trains a **`no_bias` variant on both compared models**,
with `b ≡ 0` for every contact, so the contact-node and latent models are compared like for
like. Two levels, both reported:

- **weak holdout** — the target contact is removed from the loss but still visible in the rank
  prefix;
- **strong holdout** — the target contact is removed from the loss *and* from the prefix.

The contact-node model has no parameters at an unseen contact at all; its score there is
whatever its architecture can produce without a per-contact row, and that asymmetry is the
point of the test and must be stated in the report, not hidden.

### 7.2 Patient-specificity, made non-circular [P1-5]

Patients differ in contact count, plane extent and event count, so a low between-patient
similarity can come from those alone. H4 is therefore evaluated on
contact-count-banded pairs (bands: 8–12, 13–20, 21+), with an event-count-matched subsample
control. Similarity is computed on the flow field interpolated to normalised `(s̃, h̃)`, not on
raw edge indices, which are not comparable across patients.

## 8. Model matrix

| arm | nodes | graph | role |
|---|---|---|---|
| `STATIC_CONTACT` | contact | none | frequency floor |
| `ORDINARY_GRU` | one shared state | none | **[P1-1] strongest non-spatial competitor** |
| `CONTACT_GRAPH_RNN` | contact | learned | observation-level graph; baseline only (D3) |
| `LATENT_FIXED_LOCAL_RNN` | 2D latent | fixed kNN, matched edge count | is a local substrate already enough |
| `LATENT_LEARNED_SPATIAL_RNN` | 2D latent | learned under wiring economy | **primary model** |

Development-only controls: `LATENT_DENSE_RNN` (performance ceiling) and `COORDINATE_SHUFFLE`
(node coordinates permuted **in the wiring cost only**; `H` and the real observation geometry
untouched). No Transformer, no GAT, no second GRU layer, no hidden-size sweep.

The `COORDINATE_SHUFFLE` control must satisfy an identity-permutation acceptance test: with
the identity permutation it must reproduce `LATENT_LEARNED_SPATIAL_RNN` bit-for-bit. An
earlier Topic 5 round reported an order effect that turned out to be an ablation control that
was not isomorphic to the arm it was compared against; this test is the guard against that.

## 9. Statistics

Patient is the unit. Seeds are aggregated **within** patient and never counted as `n`.
Report for every comparison: median, bootstrap 95% CI, number of patients improved, paired
Wilcoxon, and the per-patient raw points. Both pre-registered strata (§3.5) are reported for
every primary comparison regardless of direction. Any bounded coverage (patients not finished,
seeds not reached) is logged explicitly — silent truncation reads as full coverage.

## 10. Claim boundaries

**Permitted if the results support them:** patient-specific spatially anchored mesoscopic
effective propagation graph; task-effective propagation substrate; learned local backbone;
task-selected long-range shortcut; in-silico targeted lesion.

**Forbidden regardless of results:** true anatomical connectivity; synaptic connectome; causal
human brain network; excitatory/inhibitory edges; proof of a biological propagation pathway;
any claim that interictal events causally shape the seizure network; equating the learned
graph with DTI or anatomy. Also forbidden: reporting a real-data negative as biological before
H0 passes, and describing the retrospective plane as prospective geometry.

A prediction gain does not by itself license a physiological reading of the graph. The verdict
is reported as a ladder — L1 recurrence value, L2 latent substrate value, L3 learned topology
value, L4 patient-specific reproducibility, L5 targeted structural necessity, L6 optional
mode-specific routing — with each level judged on its own evidence.
