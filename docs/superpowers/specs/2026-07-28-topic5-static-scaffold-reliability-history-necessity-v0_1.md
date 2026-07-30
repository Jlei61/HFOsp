# Topic 5 static-scaffold reliability and history-necessity v0.1

## 1. Scientific scope

This is a bounded, target-blind follow-up to the Topic 5 rank-distribution RNN.
It answers two questions only:

1. Is the interictal contact participation field a reproducible patient-level
   scaffold, and how many events are needed to estimate it?
2. Does ordered history deeper than one, two, or three rank sets improve
   held-out next-set prediction?

This experiment does not read any early-ictal value, clinical-onset source,
A/B label, inter-event interval, or seizure outcome. It does not fit a new
axis model and it does not select a new RNN architecture.

## 2. Frozen cohort and split

- Dataset: `dataset_v0_4`, with its existing SHA-256 fingerprints.
- Cohort: all 34 eligible patients (18 Epilepsiae, 16 Yuquan).
- Split: the existing chronological train80/heldout20 event split.
- Unit of inference: patient, after collapsing three random seeds.
- The heldout20 is never used for fitting, calibration, early stopping, event
  count selection, or model selection.

## 3. Static-scaffold reliability

For event set \(D\), define the contact participation field

\[
q_i(D)=|D|^{-1}\sum_{e\in D}\mathbf 1(i\text{ participates in }e).
\]

The primary reliability endpoint is the contact-wise Spearman correlation
between \(q_i(\mathrm{train80})\) and \(q_i(\mathrm{heldout20})\).

Secondary checks:

- chronological first-half versus second-half within train80;
- odd-event versus even-event within train80;
- top-quartile contact Jaccard and mean absolute field error;
- deterministic event-count saturation at
  \(n\in\{25,50,100,200,500,1000,2000\}\), when available, using 200 train80
  subsamples and the untouched heldout20 field as reference;
- within-shaft circular permutation null with 5,000 draws per patient.

These analyses establish repeatability of a static contact topography. They do
not establish a physical axis, a direction of propagation, or seizure
transfer.

## 4. History-necessity models

All models retain the frozen v0.4 contact encoder, decoder, likelihood,
LOSO shared-core training, heldout train80 local-offset calibration, optimizer,
coverage, and hyperparameters.

The only changed quantity is how many ordered rank-set identity tokens are
replayed before each prediction:

- `history_1_gru`: most recent rank set;
- `history_2_gru`: most recent two rank sets;
- `history_3_gru`: most recent three rank sets;
- `full_history_gru`: all prior rank sets, reused from the frozen formal run.

At prediction step \(t\), every finite-window model:

- masks every contact observed anywhere in the causal prefix;
- computes prefix progress from the observed prefix only;
- initializes a new hidden state and replays only
  \(\max(0,t-H),\ldots,t-1\);
- never uses final event length or final participant count.

The existing `last_set_first_order` control is retained as a simpler
first-order baseline, but it is not treated as architecture-matched to the
finite-window GRUs. Existing `rank_shuffle_gru`, `unordered_prefix`, and
`static_contact_hazard` results are also reused without retraining.

One matched sensitivity is frozen before cohort aggregation:
`history_3_rank_shuffle_gru` uses the identical history-3 architecture and
coverage but permutes rank assignments among each event's participating
contacts during fitting and heldout calibration. It is evaluated on the same
unshuffled heldout20 events. This isolates ordered recent history from model
capacity and participation-set information; it is an ablation, not an
independent replication.

## 5. Outcomes and contrasts

Primary outcome: heldout20 event-balanced next-set/STOP NLL.

Frozen patient-first contrasts:

1. `history_2_gru - history_1_gru`;
2. `history_3_gru - history_2_gru`;
3. `full_history_gru - history_3_gru`;
4. `full_history_gru - last_set_first_order`;
5. `full_history_gru - rank_shuffle_gru`.
6. `history_3_gru - history_3_rank_shuffle_gru` as a matched sensitivity.

Contrasts are reported as baseline NLL minus more-informed-model NLL, so
positive values favour the model with more ordered history. Report median,
paired patient bootstrap 95% CI, Wilcoxon signed-rank p value, and number of
positive patients. No multiple endpoints are compressed into one binary gate.

Interpretation is hierarchical rather than a single go/no-go:

- history beyond one set is supported if history-2 improves over history-1;
- history beyond two sets is supported if history-3 improves over history-2;
- unbounded history is supported only if full history improves over history-3;
- the full-history composite additionally requires gains over first-order and
  rank-shuffle.

Each statement requires the corresponding patient-level median-gain 95%
bootstrap CI to lie above zero. Failure at a deeper level does not erase a
shallower positive result, and no history result changes the independent
static-scaffold conclusion.

## 6. Engineering and leakage contract

- Three fixed seeds: 20260725, 20260726, 20260727.
- Formal coverage: one complete outer-train event cycle per patient and four
  complete heldout-train80 calibration cycles.
- No free rollout is required for finite-window models; the endpoint is
  teacher-forced heldout prediction.
- Each fold writes config, fingerprints, checkpoints, event-level NLL,
  training log, coverage, resource peak, and `ictal_target_read=false`.
- Unit tests must prove that a sufficiently long window matches the full GRU,
  that earlier-than-window order does not affect finite-window logits, and
  that all previously recruited contacts remain masked.
- Launch is refused below 32 GiB available RAM. GPU jobs are sharded with a
  resource watcher and resumable per-fold outputs.

## 7. Allowed conclusions

Allowed:

- the patient-level interictal participation scaffold is or is not
  reproducible;
- estimation of that field saturates at a given approximate event count;
- deeper ordered rank history does or does not provide held-out predictive
  information beyond short history and shuffled order.

Forbidden:

- claiming seizure prediction or early-ictal transfer;
- claiming recovery of A/B templates or a physical pathological axis;
- interpreting predictive gain as a biological latent state;
- changing hyperparameters after seeing these results.
