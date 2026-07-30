# Figure 6 RNN axis/static-transfer bounded-negative draft

> **Historical model-stage record.** 当前 manuscript-facing Supplementary
> 版本为 `figure6_static_contact_topography_bounded_result.md`。本文件不再作为
> current RNN/static-transfer claim source。

## One-sentence argument

In patient-specific interictal rank sequences, a low-capacity recurrent model
captured reproducible history-dependent next-contact structure and broad
contact-level rank distributions, but neither recovered independently defined
A/B propagation axes nor transferred to the clinical-onset early-ictal energy
field.

## Terminology ledger

| Canonical term | Definition / use |
|---|---|
| interictal rank sequence | ordered contact-rank sets within one interictal population event |
| A/B propagation axis | independently frozen shared physical axis from the empirical A/B template analysis |
| RNN-selected axis | one of 32 physical directions selected by validation next-contact likelihood |
| node rank distribution | contact non-participation probability plus 10 joint normalized-rank probabilities |
| early-ictal static energy field | clinical-onset `[0,10] s`, 1–150 Hz baseline-normalized contact energy |
| source-free readout | cross-state prediction that does not use per-seizure onset contacts |

## Draft: Results

Interictal contact-rank sequences contained reproducible history-dependent
structure, but this structure did not identify the empirical A/B propagation
axis. In the full physical-axis cohort, the recurrent model improved held-out
next-contact likelihood over a fixed node-frequency model in all 22 patients
(median NLL benefit, 0.0781) and retained an ordered-history benefit (median,
0.0199; 18 of 22 patients). We therefore repeated physical-axis identification
only in the nine patients whose independently analysed A/B templates were
already supported as collinear. For each patient and seed, the model selected
one of 32 fixed physical directions using validation events before held-out
evaluation. Although the selected direction was identical across three seeds
in every patient, its alignment with the independently frozen A/B axis was
below the candidate-direction median (median alignment margin, −0.205; 95%
bootstrap CI, −0.437 to 0.052; 2 of 9 patients). The selected axis modestly
improved held-out next-contact likelihood over an isotropic model (median NLL
benefit, 0.00798; 8 of 9 patients), whereas adding the observed source side in
the six reversed-axis patients provided no stable benefit (median, 0.000041;
95% bootstrap CI, −0.00131 to 0.00184). Thus, the optimization reproducibly
selected a predictive physical direction, but not the independently defined
pathological propagation axis.

We next asked whether self-supervised rollouts preserved the contact-level
interictal field. Across 14 target-ready patients, the full recurrent model
reproduced the broad empirical ordering of contact participation (median
Spearman ρ, 0.922) and expected rank (median ρ, 0.742). However, its node rank
distribution remained nearly indistinguishable from that of the isotropic
model (median per-contact total-variation distance, 0.00467), while remaining
further from the empirical train-event distribution (median distance, 0.106).
Accordingly, the model retained which contacts typically participated and
whether they tended to occur early or late, but axis-dependent next-contact
increments produced little change in the long-run node distribution.

Finally, we froze all interictal representations before reading the
clinical-onset target and evaluated a patient-level leave-one-subject-out
readout of the early-ictal static energy field. The full model did not exceed
the patient-specific all-contact permutation null (median Spearman-ρ margin,
−0.153; 95% bootstrap CI, −0.436 to 0.601; 6 of 14 patients; FDR-adjusted
q=0.520). Neither recurrent history nor the physical-axis term improved
cross-state readout (both median increments, 0). The empirical train-event rank
distribution showed a positive descriptive trend, but its bootstrap interval
included zero and the corresponding P values were not corrected as a
pre-registered inferential family. These analyses therefore do not support
cross-state reuse by the current recurrent representation.

## Draft: Discussion

These findings separate sequence predictability from mechanism
identification. The positive next-contact and ordered-history results show that
interictal population events contain structured temporal information beyond
contact participation frequency. Nevertheless, even within patients selected
independently for a well-defined A/B axis, the direction favoured by the
prediction objective did not recover that axis. Seed-level reproducibility
therefore reflected a stable optimum rather than identification of the
pathological scaffold.

The failed early-ictal readout further localizes the model limitation. The
recurrent rollouts preserved coarse contact participation and rank order, yet
smoothed the empirical node distributions toward the isotropic solution. The
descriptive advantage of the uncompressed empirical distribution is consistent
with information loss during model rollout, but does not establish an
additional cohort-level interictal–ictal association. Thus, these negative
model results do not invalidate the independently observed empirical A/B
geometry or early-ictal field correspondence; instead, they show that a
next-contact-trained, low-capacity symmetric-axis recurrent model is
insufficient to unify those observations.

## Draft: Figure caption

**Figure 6 | A structured recurrent model captured interictal sequence
statistics but not the pathological axis or early-ictal field.**
**A,** Frozen analysis sequence: self-supervised next-contact learning was
completed before contact-level node rank distributions were frozen and the
early-ictal target was read. **B,** Patient denominators for the formal
physical-axis cohort, independently defined A/B-axis-positive subgroup,
clinical-onset target-ready cohort and their intersection. **C,** Patient-level
alignment margins relative to the median of the same 32 candidate directions
for the existing transition-selected axis and the RNN-selected axis. Lines
connect the same nine patients; horizontal bars denote medians. **D,** Held-out
NLL benefits of the selected axis over the isotropic model and of the source
term over the no-source model. **E,** Display-only example of empirical and
full-model node rank distributions; the patient was selected as the first
sorted member of the frozen axis-positive/target-ready intersection, without
reference to the displayed outcome. **F,** Patient-level Spearman-ρ margins
above the all-contact permutation-null median for the empirical distribution,
full recurrent model and model controls. Points denote patients; vertical
lines denote interquartile ranges and horizontal bars denote medians. All
cohort inference was performed after patient-level aggregation.

## 中文结构说明

- 这段 Results 先承认 RNN 已经完成的任务：next-contact 和 ordered history
  确实可预测；然后再区分“预测到一个方向”和“恢复了论文中的 A/B 病理轴”。
- 轴阳性 n=9 的结果不能写成亚组 rescue：虽然 axis term 对 NLL 有增益，但
  alignment 是阴性，因此只能叫 predictive physical direction。
- early-ictal 部分明确使用 clinical onset、`[0,10] s`、1–150 Hz static field；
  不使用 EEG onset，也不写成发作传播或早期预警。
- empirical_train80 的正向趋势只用于解释 RNN 可能丢失了信息，不能升级成新的
  primary significant result。
- 推荐位置是 supplementary bounded-negative computational result，而不是主文
  Figure 6 的机制中心。

## Claim–evidence map

| Claim | Evidence | Status |
|---|---|---|
| Interictal rank history is predictive | full > node in 22/22; ordered history median benefit 0.0199 | supported |
| The RNN recovers the empirical A/B pathological axis | alignment margin −0.205 in frozen n=9 | not supported |
| The RNN recovers broad node participation/rank order | median ρ=0.922 / 0.742 | supported, descriptive representation fidelity |
| RNN history/axis transfers to early-ictal field | Gate S/H/X all fail | not supported |
| Empirical interictal ranks may retain information lost by rollout | empirical > full descriptive trend; CI crosses zero | inferred, not a new cohort claim |
