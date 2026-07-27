# Topic 5 / Figure 6 v2.2 execution plan

> **合同**：
> `docs/superpowers/specs/2026-07-26-topic5-symmetric-axis-propagation-state-rnn-v2_2.md`
>
> **第一执行点**：human rank / geometry / target metadata inventory。
> **明确不做**：SNN exporter、synthetic benchmark、SNN recovery、dense GRU、
> restraint state、latent topology axis、eigen truncation。

## 0. 执行纪律

1. 旧 v0.7/v0.9/v1.0 已归档，不重跑 regression 或调参。
2. development 固定为 `epilepsiae_1077`、`epilepsiae_1146`、
   `yuquan_chengshuai`，三位均必须 geometry-complete。
3. 22 位 development-excluded geometry-complete 患者承担 physical-axis primary；
   31 位 development-excluded 患者只构成 sequence inventory。
4. early-ictal metadata 可先读；target values 在四项 unlock claims 通过前硬封存。
5. 每阶段生成 claim-specific 状态文件，不把所有科学结果压成一个总 boolean。
6. 每个 run 保存 resolved config、git commit、seed、日志、checkpoint、输入 fingerprint、
   peak RAM/VRAM 和 completion state。
7. 新结果写入
   `results/topic5_symmetric_axis_propagation_state_v2_2/`；旧结果不覆盖、不删除。
8. 新图目录在图实际生成后写中文 `figures/README.md`。

## Milestone A：冻结 lineage、队列与 target metadata

### A1. Upstream provenance

只验证以下旧线文件存在和 SHA256，不重跑 510 runs：

- `results/topic5_structured_axis_graph/persistent_path_mode_closeout_v1_0/`
- `docs/archive/topic5/persistent_path_mode_rnn_formal_result_2026-07-26.md`
- `results/topic5_interictal_rank_distribution/dataset_v0_4/`

输出：

`results/topic5_symmetric_axis_propagation_state_v2_2/provenance/upstream_manifest.json`

必须记录 `ictal_target_read=false`、v2.2 spec hash 和旧线
`complete_bounded_negative` 状态。

### A2. Human rank / geometry inventory

新增：

`scripts/audit_topic5_symmetric_axis_inputs_v2_2.py`

只读：

- `dataset_v0_4/subject_audit.csv`；
- per-subject rank NPZ/JSON metadata；
- contact coordinates、shaft IDs、event times 和 fingerprints。

输出：

```text
results/topic5_symmetric_axis_propagation_state_v2_2/input_audit/
├── subject_inventory.csv
├── development_cohort.json
├── physical_axis_formal_cohort.json
├── all_subject_sequence_cohort.json
└── INPUT_AUDIT_GATE.json
```

硬断言：

- 总 inventory = 34；
- geometry-complete = 25；
- development 恰为三位冻结患者且三位均 geometry-complete；
- development-excluded sequence = 31；
- development-excluded physical-axis = 22；
- development-excluded geometry-incomplete = 9；
- chronological splits 非空；
- phantom ranks 已用 participation mask 修复；
- `candidate_target_patient` 不参与路由。

任何断言失败即停止；不创建 topology fallback。

### A3. Target metadata inventory

新增：

`scripts/audit_topic5_early_ictal_target_metadata_v2_2.py`

只允许读取 path、schema、contact names、onset anchor 和样本计数，不读取 energy 或
recruitment values。逐患者、逐 seizure 记录：

- clinical-onset anchor；
- raw binary clinical-onset source set；
- early-ictal `1–150 Hz`、`[0,10] s` field artifact；
- exact contact join；
- non-source joined contact count；
- dynamic-rank producer；
- EEG-onset-only 标记；
- endpoint exclusion reason。

输出：

```text
results/topic5_symmetric_axis_propagation_state_v2_2/target_audit/
├── patient_inventory.csv
├── seizure_inventory.csv
├── endpoint_denominators.json
├── TARGET_METADATA_GATE.json
└── TARGET_VALUES_SEALED.json
```

`TARGET_VALUES_SEALED.json` 必须保持：

```json
{"energy_values_read": false, "recruitment_values_read": false}
```

structural denominator 必须是 development-excluded ∩ geometry-complete ∩
clinical-onset energy eligible，并按 spec §2.3 冻结最低 seizure/source/contact 条件。
target-value seal 解除后才允许按预注册 finite/nonconstant 规则形成 analysis denominator。

### A4. A/B read-back provenance

只登记、不读取为训练特征：

`results/interictal_propagation_masked/template_gradient_fields/per_subject/*.json`

审计：

- `axis_definition == template_propagation_axis_v2`；
- `axis_pair.shared_axis.status`；
- shared-axis contact exact join；
- artifact SHA256。

输出 `ab_axis_readback_inventory.csv`。它不能影响 development、formal eligibility 或
target unlock。

## Milestone B：精确数学模型与解析性测试

### B1. Core module

新增：

- `src/topic5_symmetric_axis_propagation_state_v2_2.py`
- `config/topic5_symmetric_axis_propagation_state_v2_2.yaml`

只实现：

- patient-centered geometry；
- fixed nearest-neighbour local scale；
- local/axis Gaussian kernels 与 per-kernel Frobenius normalization；
- \(W=gD^{-1/2}AD^{-1/2}\)；
- 单一 propagation state \(P\)；
- eligible-prefix node hazard bias；
- 受约束 scalar STOP；
- conditional-nonempty Bernoulli next-set likelihood；
- absorbing mean-field first-arrival rollout。

不得预留 restraint、MLP/GRU hook、future head、topology axis、low-rank branch 或
独立 forward/reverse axis。

### B2. Unit tests

新增：

- `tests/test_topic5_symmetric_axis_operator_v2_2.py`
- `tests/test_topic5_propagation_state_recurrence_v2_2.py`
- `tests/test_topic5_absorbing_rollout_v2_2.py`
- `tests/test_topic5_symmetric_axis_leakage_v2_2.py`
- `tests/test_topic5_symmetric_axis_aggregation_v2_2.py`

必须覆盖：

1. \(W=W^\top\)；
2. \(\mathbf u\rightarrow-\mathbf u\) 不改变 kernel；
3. row normalization 被拒绝；
4. 同一 toy symmetric graph 从两端初始化产生相反 signed displacement；
5. geometry-incomplete patient 无法进入 full-axis trainer；
6. local-isotropic 只令 \(\gamma=0\)；
7. event reset 清空 \(P\)；
8. 已参与 contact hazard 为 0；
9. node bias 使用 eligible-prefix hazard，不是 event participation；
10. terminal empty set 只由 scalar STOP 表示；
11. nonempty set likelihood 对 Bernoulli empty outcome 正确条件化；
12. STOP 后所有 future first-arrival mass 为 0；
13. event survival 与每触点 first-arrival 概率质量守恒；
14. \(H_{\mathrm{train}}=0\) 时仍有明确 \(H_{\mathrm{eval}}\) 和
    \(H_{\mathrm{transfer}}\)；
15. full/control 使用同一 event-first aggregation；
16. normalized NLL 使用 eligible contacts；
17. source-side thresholds 只由 train80 生成；
18. batch 不含 final event length、future participants、A/B 或 ictal values；
19. rank-step persistence 不接受时间单位标签；
20. full/control node-bias SHA256 完全一致。

这些是代码测试，不是新科学实验。

### B3. Trainer/analyzer skeleton

新增：

- `scripts/train_topic5_symmetric_axis_propagation_state_v2_2.py`
- `scripts/analyze_topic5_symmetric_axis_development_v2_2.py`
- `scripts/analyze_topic5_symmetric_axis_formal_v2_2.py`

trainer 只支持：

- development 60/20/20；
- physical-axis formal train80/heldout20；
- LOSO shared parameters；
- heldout-patient train-only \(\mathbf u,\gamma,g\)；
- `next_only`、`next_plus_rollout_h3`、`next_plus_rollout_h5`；
- spec §9 的五个 controls/sensitivities。

## Milestone C：三位 geometry-complete development 与冻结

### C1. Engineering smoke

每位 1 seed、每个 objective 短跑，检查：

- shape、gradient、determinism；
- finite log-space STOP/set likelihood；
- rollout mass conservation；
- no heldout/target leakage；
- CPU/GPU 一致性；
- peak memory；
- checkpoint/resume。

只修 bug 和资源设置，不增加候选。

### C2. Full development

固定：

```text
3 geometry-complete subjects
× 3 objectives
× 3 seeds (17, 29, 43)
```

按 spec §8 使用：

- 前 60% fit；
- 中间 20% objective selection；
- 最后 20% confirmation。

输出：

```text
results/topic5_symmetric_axis_propagation_state_v2_2/development/
├── run_inventory.csv
├── objective_comparison.csv
├── confirmation_metrics.csv
├── DEVELOPMENT_LOCK.json
└── figures/
    └── README.md
```

`DEVELOPMENT_LOCK.json` 必须包含：

- selected objective 与 \(H_{\mathrm{train}}\)；
- 已预定义的 \(H_{\mathrm{eval}}\)、\(H_{\mathrm{transfer}}\)；
- shared/patient parameter list；
- optimizer、early stopping、seeds；
- aggregation 与 metric fingerprint；
- input/config/code hashes；
- confirmation 读后不得修改的声明；
- two-\(W\) margin 10%。

confirmation 明显反向时停止，报告 development instability；不扩 grid。

## Milestone D：正式纯间期分析

### D1. 两个 formal inventories

生成：

```text
results/topic5_symmetric_axis_propagation_state_v2_2/formal/
├── PHYSICAL_AXIS_FORMAL_LOCK.json
└── ALL_SUBJECT_SEQUENCE_LOCK.json
```

`PHYSICAL_AXIS_FORMAL_LOCK.json` 固定：

- 22 个 LOSO folds；
- 每折 shared dynamics 只在其余 21 位 physical-axis formal 患者训练；
- heldout patient 只用自身 train80 拟合 \(\mathbf u,\gamma,g\)；
- full、local-isotropic、source-distance、random-axis、shaft-permutation、
  PCA1、two-\(W\)；
- null seed `20260726` 下的 256 random directions 与 256 shaft-preserving
  permutations；
- 3 seeds 与 selected objective；
- target seal。

`ALL_SUBJECT_SEQUENCE_LOCK.json` 固定：

- 31 位 development-excluded 患者；
- node-bias 与 Markov；
- participation/future-order descriptive outputs；
- 9 位 geometry-incomplete 患者绝不调用 full-axis trainer。

### D2. Parallel runner 与 monitor

新增：

- `scripts/run_topic5_symmetric_axis_formal_v2_2.sh`
- `scripts/monitor_topic5_symmetric_axis_formal_v2_2.py`

资源规则：

- 用 development 实测单进程 VRAM/RAM 后再决定并发；
- GPU 保留至少 20% 余量，系统内存保留至少 32 GiB；
- launcher 过滤空 subject 并断言 task count；
- atomic `run_state.json`；
- `COMPLETE` 跳过，残缺目录归档后重跑；
- `nohup` 日志和独立 monitor；
- 低资源时暂停派发，不杀运行中任务。

### D3. Claim-specific analysis

输出：

```text
results/topic5_symmetric_axis_propagation_state_v2_2/formal/analysis/
├── claim1_sequence_predictability.csv
├── claim2_axis_increment.csv
├── claim3_random_axis_specificity.csv
├── claim3_axis_secondary.csv
├── claim4_shared_scaffold.csv
├── ab_axis_readback.csv
├── physical_axis_patient_metrics.csv
├── all_subject_sequence_sensitivity.csv
└── INTERICTAL_CLAIM_SUMMARY.json
```

硬要求：

- 所有 prefix metric 先 event-first folding；
- normalized next-set NLL 除以 eligible contacts；
- source-left/right 使用 train80 Q25/Q75，中间 50% 排除；
- two-\(W\) 只分开 \(\gamma,g\)，不分开 axis；
- random-axis primary score 固定为 event-first normalized next-set NLL；
- non-inferiority 使用 paired
  \(M_p=\Delta_{\mathrm{two},p}-0.1\Delta_{\mathrm{axis},p}\) bootstrap；
- A/B read-back 只在全部 formal score 冻结后运行。

### D4. Target unlock

只有：

```text
claim2_next = PASS
claim2_future = PASS
claim3_random_axis = PASS
claim4_shared_scaffold = PASS
```

四项同时成立，才写：

`EARLY_ICTAL_VALUES_UNLOCKED.json`

否则保持 `TARGET_VALUES_SEALED.json`，生成 claim-specific bounded-negative 报告并停止。

## Milestone E：冻结的 early-ictal transfer

### E1. Target loader

只有 unlock 文件存在时运行。新增：

- `scripts/build_topic5_early_ictal_energy_transfer_index_v2_2.py`
- `scripts/run_topic5_frozen_early_ictal_energy_transfer_v2_2.py`
- `scripts/analyze_topic5_early_ictal_energy_transfer_v2_2.py`

loader 只消费 A3 冻结的 structural denominator。只有 spec §2.3 预注册的
finite/nonconstant value-QC 可以形成较小的 analysis denominator；其他
patient/seizure/contact attrition 一律 hard fail。

### E2. Primary transfer

每次 seizure：

1. 使用原始 binary clinical-onset source vector；
2. 加载该患者 formal LOSO fold 冻结的 full/local-isotropic \(W\)；
3. 使用 \(H_{\mathrm{transfer}}=N-|S_{\mathrm{source}}|\)；
4. 得到非 source contacts 的 cumulative participation field；
5. 与 `[0,10] s`、`1–150 Hz` robust-z energy ordering 比较。

固定比较：

- full；
- local-isotropic；
- source-distance-only；
- node-bias。

统计：

- seizure-level Spearman；
- patient 内 seizure median；
- patient-level full vs local-isotropic primary；
- exact same source/contact/seizure denominator；
- prediction constant 时记 Spearman 0；
- normalized-source、≥2 seizures、all-contact 和 EEG-onset-only 分别作 sensitivity。

输出：

```text
results/topic5_symmetric_axis_propagation_state_v2_2/early_ictal_transfer/
├── per_seizure.csv
├── per_patient.csv
├── primary_statistics.json
├── sensitivity_statistics.json
└── TRANSFER_CLAIM_SUMMARY.json
```

### E3. Secondary dynamic rank

仅在 A3 dynamic-rank metadata gate 通过时运行：

- later participation；
- later recruitment rank。

单独写入 `early_ictal_transfer/dynamic_rank_secondary/`。不得用它挽救 energy primary
阴性或修改 interictal model。

## Milestone F：Figure 6 与论文收口

### F1. Producer

新增：

`scripts/paper_figures/plot_fig6_symmetric_axis_propagation_state_v2_2.py`

输出：

```text
results/paper-ready-figure/fig6_symmetric_axis_propagation_state_v2_2/
├── figures/
│   ├── README.md
│   ├── fig6_symmetric_axis_propagation_state_v2_2.png
│   └── fig6_symmetric_axis_propagation_state_v2_2.pdf
└── fig6_symmetric_axis_propagation_state_v2_2_summary.json
```

Panels：

- A：同一 scaffold + 不同 source；
- B：单一 state + scalar STOP + absorbing rollout；
- C：22 人 full vs isotropic；
- D：random-axis specificity、稳定性与 A/B read-back；
- E：train-only source quantiles 下的 shared-\(W\)；
- F：clinical-onset early-ictal energy transfer。

target 未解封时，F 明确写 `target sealed`。

### F2. Scientific QA

逐项核对：

- development 三人均 geometry-complete；
- physical-axis formal 恰为 22 folds；
- 31 人只用于 sequence inventory/sensitivity；
- no topology fallback；
- no restraint state；
- node bias 是 eligible-prefix hazard；
- STOP 后 future mass 为 0 且 rollout 守恒；
- \(H_{\mathrm{train}}\)、\(H_{\mathrm{eval}}\)、\(H_{\mathrm{transfer}}\) 不混写；
- event-first aggregation；
- source-side threshold 只来自 train80；
- A/B 仅 post-hoc read-back；
- clinical onset 与 EEG onset 不混池；
- seizure → patient folding 正确；
- \(W\) 只称 effective propagation operator。

### F3. Visual QA 与文稿

- 300 dpi PNG + vector PDF；
- patient-level points、denominator、null、CI 和 claim status 可见；
- 无裁切、重叠或过密标签；
- 图后补中文 `figures/README.md`。

更新：

- `docs/paper-draft/figure6_symmetric_axis_propagation_state.md`
- `docs/archive/topic5/INDEX.md`
- `results/FIGURE_INDEX.md`

旧 structured path-mode Figure 6 保留为 supplementary/provenance。

## 交付顺序

```text
A metadata and denominator audit
  ↓
B exact model + STOP/rollout tests
  ↓
C 3 geometry-complete development cases
  ↓
D 22-fold physical-axis formal + 31-person sequence sensitivity
  ↓ Claim 2 + random-axis specificity + Claim 4
E frozen clinical-onset energy transfer
  ↓
F Figure 6 + manuscript
```

第一位执行者从 Milestone A2/A3 开始；不读取 target values，也不碰 SNN。
