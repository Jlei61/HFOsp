# Topic 5 / Figure 6 v2.1 execution plan

> **状态：SUPERSEDED；禁止执行。**
> 替代计划：
> `docs/superpowers/plans/2026-07-26-topic5-symmetric-axis-propagation-state-rnn-v2_2.md`。
> v2.1 的 development/geometry 分母、restraint、node bias 与 rollout 合同已撤回。

> **合同**：
> `docs/superpowers/specs/2026-07-26-topic5-symmetric-axis-propagation-state-rnn-v2_1.md`
>
> **第一执行点**：human rank / geometry / target metadata inventory。
> **明确不做**：SNN exporter、synthetic benchmark、directed generator、SNN recovery、
> dense GRU、latent topology axis、eigen truncation。

## 0. 执行纪律

1. 旧 v0.7/v0.9/v1.0 已归档，不重跑 regression 或调参。
2. 三位 development subjects 允许有限模型选择；31 位 development-excluded 患者是
   primary。
3. early-ictal metadata 可先读，target values 在 interictal Claim 2/4 通过前硬封存。
4. 每阶段生成 `*_GATE.json`，但不再把所有科学 claims 压成一个总 boolean。
5. 每个 run 保存 resolved config、git commit、seed、日志、checkpoint、输入 fingerprint、
   peak RAM/VRAM 和 completion state。
6. 新结果写入
   `results/topic5_symmetric_axis_propagation_state/`；旧结果不覆盖、不删除。
7. 新图目录在图实际生成后写中文 `figures/README.md`。

## Milestone A：冻结旧线与 metadata inventory

### A1. 复用旧线 manifest

直接消费：

- `results/topic5_structured_axis_graph/persistent_path_mode_closeout_v1_0/`
- `docs/archive/topic5/persistent_path_mode_rnn_formal_result_2026-07-26.md`

只验证文件存在和 SHA256，不重跑 75 项旧线测试，不重新生成 510 runs。

新增：

`results/topic5_symmetric_axis_propagation_state/provenance/upstream_manifest.json`

记录：

- v1.0 bounded-negative；
- `ictal_target_read=false`；
- v2.1 spec hash；
- 旧线只作 provenance。

### A2. Human rank / geometry inventory

新增：

`scripts/audit_topic5_symmetric_axis_inputs.py`

只读：

- `results/topic5_interictal_rank_distribution/dataset_v0_4/subject_audit.csv`
- per-subject NPZ/JSON metadata；
- contact coordinates、shaft IDs、event split 和 fingerprints。

输出：

```text
results/topic5_symmetric_axis_propagation_state/input_audit/
├── subject_inventory.csv
├── physical_axis_cohort.json
├── sequence_cohort.json
└── INPUT_AUDIT_GATE.json
```

硬断言：

- 总 inventory 恰好 34 人；
- development subjects 恰好 3 人；
- development-excluded 恰好 31 人；
- 当前预期 geometry-complete 全体 25 人；
- development-excluded physical-axis primary 当前预期 24 人；
- 每位患者 chronological split 与 event count 非空；
- phantom ranks 已 mask；
- `candidate_target_patient` 不参与路由。

若数字变化，停止并修订 denominator；不自动创建 topology fallback。

### A3. Target metadata inventory

新增：

`scripts/audit_topic5_early_ictal_target_metadata.py`

只允许读取 path/schema/name/time metadata，不读取 energy 或 recruitment 数值。逐患者、
逐 seizure 记录：

- clinical-onset anchor；
- clinical-onset contact set；
- early-ictal `1–150 Hz` field artifact；
- exact contact join；
- dynamic recruitment-rank producer；
- EEG-onset-only 标记；
- endpoint-specific exclusion reason。

输出：

```text
results/topic5_symmetric_axis_propagation_state/target_audit/
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

metadata audit 结束后立即冻结每个 endpoint 的预期分母，避免正式间期训练完成后才发现
target 不可执行。

## Milestone B：精确数学模型与 toy tests

### B1. Core module

新增：

- `src/topic5_symmetric_axis_propagation_state.py`
- `config/topic5_symmetric_axis_propagation_state_v2_1.yaml`

只实现 spec 中唯一公式：

- patient-centered geometry；
- fixed nearest-neighbour local scale；
- local/axis Gaussian kernels；
- per-kernel Frobenius normalization；
- \(A=(1-\gamma)\bar K_\mathrm{local}+\gamma\bar K_\mathrm{axis}\)；
- \(W=gD^{-1/2}AD^{-1/2}\)；
- propagation drive / restraint trace；
- independent contact hazards；
- empty-set STOP；
- first-arrival soft rollout。

不得预留 MLP/GRU hook、free function、future head、topology axis 或 low-rank branch。

### B2. Unit tests

新增：

- `tests/test_topic5_symmetric_axis_operator.py`
- `tests/test_topic5_propagation_state_recurrence.py`
- `tests/test_topic5_first_arrival_rollout.py`
- `tests/test_topic5_symmetric_axis_leakage.py`

必须覆盖：

1. \(W=W^\top\) 到数值精度；
2. \(\mathbf u\rightarrow-\mathbf u\) kernel 不变；
3. row normalization 被测试拒绝；
4. 同一个 toy symmetric graph 从两端初始化产生相反 signed displacement；
5. local-isotropic 只令 \(\gamma=0\)，不改变 node bias/state code；
6. 已参与 contact hazard 为 0；
7. event reset 清空两个 states；
8. STOP 等于空 next-set；
9. future participation、rank distribution 和 cumulative field 只由 first-arrival
   hazards 推导；
10. 训练 batch 不含 final event length、final participant count 或 heldout fields；
11. rank-step persistence 不接受时间单位标签；
12. full/control node-bias SHA256 完全相同。

这些是解析性代码测试，不是新的科学实验。

### B3. Trainer 与 analyzer skeleton

新增：

- `scripts/train_topic5_symmetric_axis_propagation_state.py`
- `scripts/analyze_topic5_symmetric_axis_development.py`
- `scripts/analyze_topic5_symmetric_axis_formal.py`

trainer 支持：

- development 60/20/20；
- formal train80/heldout20；
- LOSO shared parameters；
- heldout-patient train-only \(\mathbf u,\gamma,g\) calibration；
- `next_only`、`next_plus_rollout_h3`、`next_plus_rollout_h5` 三个且仅三个 objectives；
- full、node-bias、source-distance、local-isotropic、Markov 和 two-\(W\) sensitivity。

## Milestone C：三患者 development 与冻结

### C1. Engineering smoke

先每位 1 seed、每个 objective 只跑短覆盖，检查：

- shape/gradient/determinism；
- finite loss；
- no heldout leakage；
- CPU/GPU 一致性到容差；
- peak memory；
- checkpoint/resume。

只修 bug 和资源设置，不增加候选。

### C2. Full development

固定三患者 × 三 objectives × 三 seeds。按 spec §7：

1. 前 60% fit；
2. 中间 20% objective selection；
3. 最后 20% confirmation。

输出：

```text
results/topic5_symmetric_axis_propagation_state/development/
├── run_inventory.csv
├── objective_comparison.csv
├── confirmation_metrics.csv
├── DEVELOPMENT_LOCK.json
└── figures/
    └── README.md
```

`DEVELOPMENT_LOCK.json` 必须包含：

- selected objective 与 horizon；
- 选择规则的逐患者输入；
- shared/patient parameter list；
- optimizer、early stopping、seeds；
- input/config/code hashes；
- confirmation 已读后不得修改的声明；
- two-\(W\) non-inferiority margin：10% full-vs-isotropic benefit。

若 confirmation 明显反向，只报告 development instability；不得再扩 grid。此时停止，
重新找用户决定是否接受模型，而不是自动进入 31 人。

## Milestone D：正式纯间期分析

### D1. Formal lock

生成：

`results/topic5_symmetric_axis_propagation_state/formal/FORMAL_LOCK.json`

固定：

- 31 人 sequence cohort；
- 24 人预期 physical-axis primary cohort；
- 3 seeds；
- selected objective；
- all controls；
- claim-specific statistics；
- source-side minimum events；
- 256 random axes + 256 shaft-preserving coordinate permutations；
- target seal。

### D2. Parallel runner

新增：

- `scripts/run_topic5_symmetric_axis_formal.sh`
- `scripts/monitor_topic5_symmetric_axis_formal.py`

资源规则：

- 先用 development 实测单进程 VRAM/RAM；
- GPU 保留至少 20% 余量，系统内存保留至少 32 GiB；
- launcher 过滤空 subject，并断言 task count；
- atomic `run_state.json`；
- `COMPLETE` 跳过，残缺目录先归档后重跑；
- `nohup` 日志和独立 monitor；
- 低资源时暂停派发，不杀运行中任务。

### D3. Claim-specific analysis

输出目录：

```text
results/topic5_symmetric_axis_propagation_state/formal/analysis/
├── claim1_sequence_predictability.csv
├── claim2_axis_increment.csv
├── claim3_axis_nulls.csv
├── claim4_shared_scaffold.csv
├── physical_axis_patient_metrics.csv
├── all_subject_sequence_sensitivity.csv
├── development_excluded_statistics.json
├── full34_supportive_statistics.json
└── INTERICTAL_CLAIM_SUMMARY.json
```

分析分层：

- Claim 1：full vs node bias，sanity；
- Claim 2：full vs local-isotropic，next-set 与 future first-arrival 分开判定；
- Claim 3：split-half、random direction、shaft-permutation、PCA1；
- Claim 4：heldout 两侧、shared vs two-\(W\)、cross-side transfer；
- all-subject：31 人 coordinate-free Markov/participation sensitivity；
- full34：supportive，不能替代 development-excluded primary。

没有单一 `formal_gate_pass`。`INTERICTAL_CLAIM_SUMMARY.json` 分别记录
`claim1_status` 至 `claim4_status`。

只有 `claim2_next=PASS`、`claim2_future=PASS`、`claim4=PASS` 时，写：

`EARLY_ICTAL_VALUES_UNLOCKED.json`

否则写 target-sealed bounded-negative 报告并停止于 Milestone D。

## Milestone E：冻结的 early-ictal transfer

### E1. Energy target loader

只有 unlock 文件存在时才运行。新增：

- `scripts/build_topic5_early_ictal_energy_transfer_index.py`
- `scripts/run_topic5_frozen_early_ictal_energy_transfer.py`
- `scripts/analyze_topic5_early_ictal_energy_transfer.py`

loader 只消费 Milestone A 冻结 denominator。任何 patient/seizure/contact attrition
变化都 hard fail，不自动改分母。

### E2. Primary transfer

每次 seizure：

1. clinical-onset contact set 作为 source；
2. 加载冻结 interictal full/local-isotropic \(W\)；
3. 用相同 horizon rollout 得到 \(\widehat A_i\)；
4. 在非 source contacts 上与 clinical-onset `[0,10] s`、
   `1–150 Hz` robust-z energy ordering 比较。

固定比较：

- full；
- local-isotropic；
- source-distance-only；
- node-bias。

统计：

- seizure-level Spearman；
- patient 内先取 seizure median；
- patient-level full vs local-isotropic 为 primary；
- exact same source/contact/seizure denominator；
- all-contact 与 EEG-onset-only 分别作 sensitivity。

输出：

```text
results/topic5_symmetric_axis_propagation_state/early_ictal_transfer/
├── per_seizure.csv
├── per_patient.csv
├── primary_statistics.json
├── sensitivity_statistics.json
└── TRANSFER_CLAIM_SUMMARY.json
```

### E3. Secondary dynamic rank

仅当 A3 的 dynamic-rank metadata gate 通过时运行：

- later participation；
- later recruitment rank。

单独目录：

`early_ictal_transfer/dynamic_rank_secondary/`

不得用 secondary 阳性挽救 energy-field primary 阴性，也不得修改 interictal model。

## Milestone F：Figure 6 与论文收口

### F1. Producer

新增：

`scripts/paper_figures/plot_fig6_symmetric_axis_propagation_state.py`

输出：

```text
results/paper-ready-figure/fig6_symmetric_axis_propagation_state/
├── figures/
│   ├── README.md
│   ├── fig6_symmetric_axis_propagation_state.png
│   └── fig6_symmetric_axis_propagation_state.pdf
└── fig6_symmetric_axis_propagation_state_summary.json
```

Panels：

- A：同一 scaffold + 不同 source；
- B：精确 equations + unified hazard；
- C：full vs isotropic；
- D：axis stability/null；
- E：cross-direction shared-\(W\)；
- F：early-ictal energy-field transfer。

若 target 未解封，F 明确写 `target sealed`，不使用 dynamic-rank 或旧 SNN 图填空。

### F2. Scientific QA

逐项核对：

- physical-axis primary 只含 geometry-complete development-excluded 患者；
- 31 人与 34 人不混写；
- no topology fallback；
- node bias/control denominators 完全一致；
- no future-length leakage；
- state 只称 propagation/restraint；
- \(W\) 只称 effective propagation operator；
- clinical onset 与 EEG onset 不混池；
- seizure → patient folding 正确；
- claims 独立报告，无全局 boolean。

### F3. Visual QA 与文稿

- 300 dpi PNG + vector PDF；
- patient-level points 和 denominator 可见；
- null、effect direction、CI 和 claim status 清楚；
- 无裁切、重叠或过密标签；
- 图后补中文 README。

更新：

- `docs/paper-draft/figure6_symmetric_axis_propagation_state.md`
- `docs/archive/topic5/INDEX.md`
- `results/FIGURE_INDEX.md`

旧 structured path-mode Figure 6 保留为 supplementary/provenance，不覆盖。

## 交付顺序

```text
A metadata audit
  ↓
B exact model + toy tests
  ↓
C 3-subject development + DEVELOPMENT_LOCK
  ↓
D 31-subject formal interictal claims
  ↓ Claim 2 + Claim 4 only
E frozen early-ictal energy transfer
  ↓
F Figure 6 + manuscript
```

第一位执行者从 **Milestone A2/A3** 开始，不读取 target values，也不碰 SNN。
