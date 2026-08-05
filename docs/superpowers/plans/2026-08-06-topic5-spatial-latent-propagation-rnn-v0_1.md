# Plan — Topic 5 Spatial Latent Propagation RNN v0.1

Spec: `docs/superpowers/specs/2026-08-06-topic5-spatial-latent-propagation-rnn-v0_1.md`
Output root: `results/topic5_spatial_latent_propagation_rnn_v0_1/`
Interpreter: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`

Every task states its verification. A task is done when its verification passes, not when the
code is written.

## Milestone A — input manifest ✅

`scripts/build_topic5_slp_input_manifest.py` → `INPUT_MANIFEST.json`.

Verify: manifest exists; `frozen_cohort.n_primary == 21`; strata `planar=12`,
`well_sampled=18`; `geometry_status == RETROSPECTIVE_GEOMETRY_PILOT`; dataset manifest
certifies `target_values_read=false`.

## Milestone B — spec and plan ✅

Verify: both documents committed; every P1 fix appears at its section.

## Milestone C — cache

`src/topic5_virtual_seeg_operator.py`, `scripts/build_topic5_slp_cache.py` →
`cache/<patient>/{plane_coordinates,latent_nodes,seeg_operator,events}.npz` + `provenance.json`.

- contacts in the frozen intersection order, exact-name;
- farthest-point node sampling in the dilated contact hull, recorded seed;
- `H` from the continuous Gaussian readout kernel truncated at 3σ, row-normalised;
- events split by `development_split(0.15, 0.15)`, stored separately.

Verify: T1–T4 below pass for all 21 patients; `H.sum(1) == 1`; no contact reads a node beyond
3σ; cache rebuild is byte-identical.

## Milestone D — models

`src/topic5_spatial_latent_rnn.py`, five arms plus two controls per spec §8.

Verify: T5–T13 pass; a 200-step overfit run on one patient drives train loss below the static
baseline for every arm.

## Milestone E — TDD suite

`tests/test_topic5_spatial_latent_rnn.py`. Sixteen tests, each mapped to a spec clause. These
are written **before** the corresponding implementation.

| id | test | guards spec clause |
|---|---|---|
| T1 | `H` rows sum to 1 for every patient | §4.2 |
| T2 | `H` support is local: every non-zero entry is within 3σ | §4.2 |
| T3 | contact order is the exact-name intersection, in record order | §3.4 |
| T4 | cache rebuild with the same seed is byte-identical | §4.1 |
| T5 | input and readout use the same `H` object | §4.3 |
| T6 | zeroing `A` degrades the latent model to node-wise, matching a hand-computed reference | §5 |
| T7 | no dense contact-to-contact path exists: perturbing one contact's input changes another's logit only through nodes | §4.3 |
| T8 | node coordinates never reach the prediction head — permuting them with wiring loss disabled leaves logits unchanged | §5 |
| T9 | `K=1` matches a hand-computed single microstep | §4.4 |
| T10 | free rollout reads no future rank; feeding shuffled future ranks changes nothing | §3.1 |
| T11 | rollout halts at STOP and emits nothing after | §3.1 |
| T12 | checkpoint resume reproduces the un-interrupted trajectory | Milestone H |
| T13 | topology freeze is reproducible from the same gate parameters | §6 |
| T14 | **`COORDINATE_SHUFFLE` with the identity permutation reproduces the learned arm bit-for-bit** | §8 |
| T15 | train / validation / test event index sets are disjoint and chronological, and `old_heldout20` never enters any of them | §3.2 |
| T16 | leave-contact-out `no_bias` variant has zero per-contact parameters on both compared models | §7.1 |

Verify: `pytest tests/test_topic5_spatial_latent_rnn.py -q` all green; `git diff --check` clean.

## Milestone F — recovery controls

Must pass before any real-data verdict is written.

- **F1 synthetic** — generate rank events from a known sparse spatial graph; check the learner
  recovers the top-edge ranking above chance.
- **F2 SNN positive control** — take the blessed E1146 SNN layout, generate virtual-SEEG rank
  events, train the model on those events only, and check it recovers the known main
  propagation direction and the lesion-sensitive path. The model must never read SNN
  connectivity.

Verify: F1 rank correlation with true edge importance above a pre-registered floor; F2 recovers
the main direction. On failure, debug implementation, loss scale and observation
identifiability — a real-data negative may **not** be called biological until these pass.

## Milestone G — development sweep and freeze

Development patients `epilepsiae_1146`, `epilepsiae_958`, `yuquan_zhangkexuan`, reading only
their `validation` partitions.

Sequential, not a full Cartesian product:

1. E1146 fast screen: node density `{2C, 4C}` capped 64 × microsteps `{1,3,6}` × edge budget
   `{4,6}` × wiring strength `{0.03, 0.1, 0.3}` of the task loss at ramp start;
2. keep two Pareto-reasonable configs;
3. run those two on the other two development patients, 2 seeds.

Selection: the Pareto knee of held-out NLL against total wiring cost, plus edge count, rollout
fidelity and the §4.4 hop-reachability diagnostic. Not the lowest NLL — that would select the
dense-like configuration and make the whole wiring-economy question vacuous.

Verify: `development/SWEEP_SUMMARY.json` lists every configuration run and every one dropped,
with its reason; exactly one `FROZEN_CONFIG.json` is written, with hashes, and the primary
patients are never retuned after it.

## Milestone H — formal cohort

21 patients × 5 arms × seeds, in priority order: all patients at seed 1, then seed 2, then
seed 3. Resumable; a patient that fails must not stop the others; `DONE.json` marks completion
and is never recomputed.

Per unit: `config.json`, `DONE.json`/`FAILED.json`, `checkpoint.pt`, `heldout_predictions.npz`,
`rollout_events.npz`, `graph_edges.csv`, `graph_metrics.json`, `node_flow.npz`,
`training_log.csv`, input and code hashes.

Verify: `EXPERIMENT_MATRIX.csv` reconciles planned against completed units; coverage is stated
explicitly in the report if incomplete.

## Milestone I — structure analysis

1. graph-formation snapshots at warm-up end, structure midpoint, freeze, final;
2. edge length, degree, wiring cost, hop reachability;
3. lesion importance per edge; top 5/10/20 % targeted lesion against distance-matched,
   weight-matched, distance-and-weight-matched random edges and degree-preserving rewiring;
4. 2D effective-flow vector field per patient, interpolated to normalised `(s̃, h̃)`;
5. within-patient across-seed vs between-patient similarity, **contact-count banded** and
   event-count-matched (spec §7.2);
6. leave-contact-out weak and strong holdout, `no_bias` on both arms (spec §7.1);
7. post-hoc A/B message-flow routing, no retraining.

Verify: each output has its own CSV/JSON with per-patient rows; no analysis silently drops a
patient without recording it.

## Milestone J — figures, statistics, report

Six panels A–F per spec, PNG + PDF + source data, full-resolution eyeball QA before commit,
and `figures/README.md` in Chinese per the repository standard.

Statistics: patient-level only, seeds aggregated within patient. Every primary comparison
reports median, bootstrap CI, patients improved, paired Wilcoxon, per-patient points, and both
pre-registered strata.

`CLOSEOUT_REPORT.md` sections: run matrix actually completed; units not completed and why;
synthetic recovery; SNN positive control; contact-RNN prediction; latent-RNN prediction;
accuracy–wiring trade-off; learned vs fixed local; graph reproducibility; cross-patient
variability; targeted lesion; leave-contact-out; A/B routing; safe claims; prohibited claims;
next minimal formal experiment. Plus the L1–L6 verdict ladder.

Verify: report exists; every hypothesis in spec §7 has an explicit verdict or an explicit
"not resolved, because …"; no forbidden phrase from spec §10 appears anywhere in it.

## Stop conditions

Stop and report rather than continue only on: unrecoverable missing data, demonstrated future
information leakage, corrupted input, or a readout misalignment. A negative scientific result
is **not** a stop condition — the remaining pre-registered experiments still run.
