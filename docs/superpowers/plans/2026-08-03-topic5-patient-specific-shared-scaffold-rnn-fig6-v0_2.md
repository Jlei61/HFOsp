# Topic 5 / Figure 6：shared-scaffold propagation RNN v0.2 execution plan

**合同**：
`docs/superpowers/specs/2026-08-03-topic5-patient-specific-shared-scaffold-rnn-fig6-v0_2.md`

**结果根目录**：
`results/topic5_patient_specific_shared_scaffold_rnn_v0_2/`

## 0. 执行纪律

1. 不覆盖 v0.1 或任何旧 RNN 结果。
2. 每个训练 unit 原子写 checkpoint、resolved config、seed、输入 hash、log、metrics、
   peak RAM/VRAM 和 `DONE.json`；失败 unit 可断点续跑。
3. target metadata 与 target values 分离；解除 seal 前先保存 model/field manifest。
4. 不因中间结果阴性停止后续预定分析，不设置 claim-chain gate。
5. GPU worker 数由 smoke 的峰值显存决定；保留至少 15% 显存余量。watcher 每 5 分钟写
   completion/failure/OOM/NaN summary，网络重连后从 `DONE.json` 恢复。
6. 新图生成后必须目视 QA，并同时写中文 `figures/README.md`。

## A. 输入与 denominator 冻结

### A1. Interictal inventory

新增 `scripts/audit_topic5_shared_scaffold_inputs_v0_2.py`：

- 读取 masked rank dataset 的 34 位患者；
- 审计 contact order、participation mask、event timestamp、shaft ID/ordinal；
- 生成每位患者 chronological fit60/val20/test20 索引；
- 断言 split 非空且 contact mapping 在三段完全一致；
- 保存 dataset/code/config SHA256。

输出：

```text
input_audit/subject_inventory.csv
input_audit/split_manifest.json
input_audit/input_fingerprints.json
```

### A2. Ictal metadata-only inventory

新增 `scripts/audit_topic5_shared_scaffold_ictal_metadata_v0_2.py`：

- 冻结 15 位 primary clinical-onset 患者；
- 冻结 `epilepsiae_1146` supportive；
- 记录 seizure ID、clinical-onset anchor、1--150 Hz `[0,10] s` artifact path、exact
  contact join 和 exclusion reason；
- 不反序列化 energy values。

输出 `target_audit/ictal_metadata_inventory.csv` 与：

```json
{"energy_values_read": false, "target_values_sealed": true}
```

## B. 模型、loss 与测试

### B1. Core implementation

新增：

- `src/topic5_shared_scaffold_rnn.py`
- `config/topic5_shared_scaffold_propagation_rnn_v0_2.yaml`
- `scripts/train_topic5_shared_scaffold_rnn_v0_2.py`

同一 trainer 必须通过
`--model {ordinary,structured,structured_rank_shuffle}` 切换训练模型；static 为同一数据
合同上解析估计的无历史基线，不另行优化。禁止三套脚本各自实现 scoring。统一实现：

- fit60 participation bias；
- event reset 与 causal prefix；
- contact/STOP/cardinality 三项 loss；
- ordinary dense GRU；
- structured symmetric rank-2 scaffold；
- validation checkpoint selection；
- test20 evaluation；
- expected first-arrival rollout。

### B2. Unit tests

新增 `tests/test_topic5_shared_scaffold_rnn_v0_2.py`，至少覆盖：

1. phantom/non-participating ranks 不进入输入；
2. fit/val/test chronology 与 patient isolation；
3. batch 不含 A/B、mean rank 或 ictal values；
4. structured \(W=W^\top\)，rank factor为 2，且无 dense bypass；
5. ordinary/structured 共用 static bias、candidate mask、loss 和 batch order；
6. event reset、已参与 contact mask、tie-set target；
7. contact loss 只在 continue decisions 上计算；
8. STOP 与 cardinality 可单独重算；
9. 同一个 symmetric scaffold 从两端初始化产生相反 rollout；
10. source pools、horizon 和 fields 的 hash 在 target read 前冻结；
11. all-contact/within-shaft permutation 每次重新执行绝对值与 two-direction max；
12. seizure-first、patient-first aggregation；
13. `epilepsiae_1146` 不进入 primary statistic。

## C. Smoke 与全队列训练

### C1. Smoke

在一位 Epilepsiae 和一位 Yuquan 患者各跑一个 seed、ordinary 与 structured：

- 验证 loss 下降、checkpoint 可恢复、rollout 概率质量有限；
- 记录单 unit 峰值显存/RAM 和吞吐；
- 用测得的峰值决定 worker 数，不修改科学超参数。

Smoke 只检查工程正确性，不要求效果阳性。

### C2. Formal training

并行运行：

```text
34 patients × 3 seeds × {ordinary, structured, structured-rank-shuffle}
+ 34 patient-specific static baselines
```

每个 unit 保存：

```text
per_subject/<patient>/<model>/seed_<seed>/
├── resolved_config.json
├── checkpoint.pt
├── train_log.jsonl
├── validation_metrics.json
├── test_metrics.json
├── rollout_test20.npz
└── DONE.json
```

runner 使用 `nohup` 或现有 tmux session，支持 `--resume` 和 `--workers`。新增
`scripts/watch_topic5_shared_scaffold_rnn_v0_2.py`，持续写
`monitor/status.json` 与 `monitor/status.log`。

## D. 间期分析与 field freeze

新增 `scripts/analyze_topic5_shared_scaffold_interictal_v0_2.py`：

- 汇总 34 人 static/ordinary/structured held-out contact NLL；
- 汇总 top-1、STOP、cardinality，但不与 contact endpoint 混合；
- 计算 rollout-vs-test20 participation、precedence 和 rank Wasserstein；
- 计算 structured vs ordinary、structured vs static patient-first statistics；
- 生成 empirical fit60 precedence 到 test20 的数据上限参照；
- 用 interictal-only metric 选择 Panel B cohort-median representative。

新增 `scripts/freeze_topic5_shared_scaffold_fields_v0_2.py`：

- 从 structured seed ensemble 构造一个 diffusion coordinate 与两端 source pools，ordinary 与
  structured 共享相同 source interventions；
- 用 fit60 长度分布冻结 \(H_p\)；
- 生成唯一 \(F^-_p,F^+_p\) participation-weighted first-arrival earliness fields；
- seed ensemble 先逐 contact 平均，再冻结；
- 写 `field_freeze/FROZEN_FIELD_MANIFEST.json` 和全部 SHA256。

Static baseline 只冻结一张 fit60 participation field。完成 field manifest 后才允许进入 E。

## E. Early-ictal target-free scoring

新增 `scripts/score_topic5_shared_scaffold_early_ictal_v0_2.py`：

1. 核验 `FROZEN_FIELD_MANIFEST.json` 后读取 target values，并把 seal 状态改写为只追加的
   `TARGET_UNLOCK_RECORD.json`；
2. 对每次 seizure 计算 structured/ordinary two-direction max-absolute Spearman；
3. 对 static field 计算单场 absolute Spearman；
4. 跑 5000 次 all-contact primary null，每次完整重做 absolute/max；
5. 跑 5000 次 within-shaft sensitivity；
6. seizure-first 后 patient-first 汇总；
7. 输出 15 位 primary 的群体统计，E1146 单列 supportive。

输出：

```text
early_ictal/seizure_scores.csv
early_ictal/patient_scores.csv
early_ictal/cohort_statistics.json
early_ictal/permutation_manifest.json
```

## F. Figure 6 A--E

新增 `scripts/plot_topic5_shared_scaffold_figure6_v0_2.py`，固定五块：

- **A**：shared-scaffold RNN 架构；同一个对称 \(W\)、两个 source sides、无 A/B/ictal
  输入。
- **B**：预先冻结的 E1146 source-minus/source-plus 上下排列；observed test20 与 generated
  contact-by-rank heatmap 并列，并显示逐 rank 时序。
- **C**：34 人的 held-out next-contact NLL paired comparison，加 rollout-test20
  precedence consistency；显示 patient-first statistics。
- **D**：E1146 的冻结 \(F^-\)、\(F^+\) 与患者内两次 seizure 中位 early-ictal broadband
  contact map；明确标为 illustrative/supportive。
- **E**：15 人 ordinary-vs-structured null-corrected early-ictal correspondence，显示
  all-contact primary null、个人 p95 计数和 within-shaft sensitivity；E1146 空心点单列。

图中不得把 empirical A/B 当训练标签，不得只展示训练曲线，不得从多种 rollout summary 中按
ictal score 选择 field。输出 PDF、SVG、600 dpi PNG 和逐 panel source-data CSV。

图实际生成后写 `figures/README.md`，逐 panel 用中文说明：测了什么、怎么测、结果支持什么、
不能解释为什么。

## G. 验收与收口

1. 跑新增测试和相关 Topic 5 回归测试。
2. 核对训练 unit 完成数、失败/OOM/NaN、重复 hash、target unlock 顺序。
3. 对 PDF/SVG/PNG 逐张目视 QA：文字遮挡、色标一致、统计数字与 JSON 一致、supportive
   标记明确。
4. 写白话结果报告与 paper-ready caption；按三条 claim 独立给结论，不生成总
   `hard_gate_pass`。
5. 更新 manifest/index；提交工作须按用户后续授权执行，本计划本身不 commit。
