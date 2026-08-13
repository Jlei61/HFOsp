# Topic 5.1 LBSS-RNN 执行计划 v0.2

> **2026-08-12 状态修正**：本计划已完成的空间模型使用 contact-dilated latent domain，仅作为历史
> sensitivity。后续执行由 `2026-08-12-topic5-lbss-full-tissue-rnn-v0-3.md` 接管。

> 对应 spec：`docs/superpowers/specs/2026-08-10-topic5-local-backbone-selective-shortcut-rnn-design.md`
>
> 状态：LBSS 物理几何子分析已完成。全 cohort/early-ictal 分母修复由
> `2026-08-11-topic5-rnn-full-cohort-field-transfer-correction.md` 接管；本计划中旧 10 人 target
> 交集不得再作为正式外部测试分母。

## Milestone A｜干净工作区与输入冻结

1. 从用户终审后的明确 base commit 创建独立 `codex/topic5-lbss-rnn-v0-1` worktree。
2. 复制 v0.4 的 21-patient/31-fit input cache 或只读引用；核对每个 cache hash。
3. 写入 `RUN_CONTRACT.json`：geometry、cohort、split、cell、optimizer、decoder、early endpoint。
4. 写入 `geometry_status=RETROSPECTIVE_TEST_INFORMED` 与 `edge_time_status=ORDINAL_NO_PHYSICAL_DELAY`。
5. 明确 target 历史状态：`TARGET_KNOWN_TO_PROJECT_BUT_WITHHELD_FROM_LBSS_TRAINING_AND_SELECTION`。
6. 将 launcher、model、trainer、scorer、field builder 复制到 immutable `run_snapshot/`，逐文件 hash。
7. 冻结每位患者 local/extra-local/nonlocal pool size、per-node candidate counts 和物理距离分布。

阶段图：

- `figures/stage_a_geometry_and_candidate_pools.png/.pdf`
- 每位代表患者显示 local backbone、extra-local pool、LR pool 和 K，不读取 target。

验收：

- 21 patients / 31 fits；
- 所有 H-supported nodes 位于单一 strongly connected component；
- contact-supported pairwise reachability = 1.0；
- 每个 fit 的 candidate pool ≥K；
- snapshot 与 source hash 一致。

该阈值只适用于历史 LBSS 物理机制子集，不再决定 RNN external benchmark 分母。正式
early-ictal primary 必须逐位匹配 Figure 3D 的 17 人/167 seizures；任何缺失都报错停止，不能
因 geometry-ineligible 自动缩小 cohort。

## Milestone B｜LBSS mask 与训练实现

实现以下模块：

1. symmetrized kNN candidate mask、双向 mask 和独立有向权重；
2. extra-local/LR candidate pool；
3. local mask 永久冻结的 masked recurrent operator；
4. 仅在指定 pool 内保持 K 条 active edges 的 SET；
5. L2/L3 同 seed 同初始 LR mask；
6. 语义阶段 snapshots：INIT、AFTER_WARMUP、REWIRE_1_3、REWIRE_2_3、MASK_FREEZE、FINAL；
7. order-shuffle 固定 rank 1，使用 rank 2…T tie-set derangement；
8. checkpoint 只能从 mask-freeze/共同 structural-phase 之后选择；
9. resume 恢复 optimizer、RNG、active mask、edge age、rewiring counter 和 freeze status；
10. 新生 edge 一个完整 rewiring interval 的 grace period；
11. source-first balanced proposal sampler 与 candidate-exposure audit。

单元测试：

- local mask 每 node 最低入/出度；
- local graph strong connectivity 与 pairwise reachability；
- edge count 与 K 恒定；
- local edges 从不被 prune；
- L2/L3 初始 mask 逐位相同；
- L0–L3 shared parameters 同 seed 逐位相同；
- L2/L3 local weights 和 initial added-edge weights 逐位相同；
- L1 只能从 extra-local pool 生长；
- L3 只能从 LR pool 生长；
- shuffle 保留 rank 1 与全部 tie sets；
- shuffle derangement、短事件未改变比例和 Kendall 破坏度正确；
- candidate proposal source frequency 不由 pool degree 静默偏置；
- best-checkpoint epoch 合格；
- resume 后 mask/RNG/counter 逐位一致；
- free rollout 不读取未来 set size；
- checkpoint restore 逐位一致。

阶段图：

- `figures/stage_b_mask_dynamics_toy_graph.png/.pdf`
- 用小图直接显示 local backbone 固定、K 条候选边在训练中替换。

## Milestone C｜小规模工程 smoke

使用预先固定的三个工程病例：

- E1073；
- E1146；
- Yuquan chenziyang（若不在 21 人物理队列，只作 shape/OOM smoke，不进入科学统计）。

运行 5 arms × 1 seed：

- shape/gradient/determinism；
- memory/throughput；
- snapshots；
- heldout scorer；
- distal bin denominator；
- field builder 不读取 target。

本阶段不根据科学效应修改 K、r_local 或模型；只修 bug/OOM。

阶段图：

- `figures/stage_c_training_and_distance_bins.png/.pdf`
- 展示训练 loss、active LR edge count 恒定、train-distance quantile 和 heldout bin counts。

### C2 Functional-shortcut detectability

在 2–3 个真实患者几何上植入已知 K 条 nonlocal effective shortcuts，生成 rank events，并以完全相同代码拟合 L0–L3 与 shuffle。只验证模型类别可检测性：distal benefit、attenuation specificity 和 true-order>shuffle；不要求 exact-edge recovery，也不作为真实数据启动 gate。

阶段图：

- `figures/stage_c2_functional_shortcut_detectability.png/.pdf`
- 展示已知 shortcut 条件下的 distal gain 与 attenuation dose response。

## Milestone D｜全 cohort target-free 训练

### D1 Core

先运行：

```text
L0_LOCAL_ONLY
L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL
L2_LOCAL_PLUS_RANDOM_LR
L3_LOCAL_PLUS_LEARNED_LR
C_L3_ORDER_SHUFFLED
```

31 fits × 5 arms × 3 seeds = 465 units；synthetic detectability 单元另列，不混入真实患者统计。

`C_L3_ORDER_SHUFFLED` 只在 train/validation events 上执行 derangement；所有 arm 共享未改动的真实 held-out test events、candidate masks 和评分决策。Aggregate 前逐 fit 核验 test-rank hash 完全相同。

Local/intermediate/distal 的 q50/q80 一律由未打乱的真实 train events 冻结；aggregate 同时核验各 arm 的 `distance_bin_reference_sha256` 和阈值逐位一致。

资源策略：

- 使用 GPU 多 worker，但以实测吞吐为准；
- 每个 unit 独立日志、DONE/FAILED；
- OOM 自动降并发，不改变 batch/模型合同；
- launcher 以 nohup/tmux 在 immutable snapshot 上运行；
- watcher 只监控与推进，不修改 active scripts。
- 所有恢复 OOM/retry 单独记账；最终要求 0 unresolved OOM，而不是历史上从未出现可恢复 OOM。

### D2 聚合

输出：

- `interictal_per_event.csv`；
- `interictal_per_fit_seed.csv`；
- `interictal_per_patient.csv`；
- `distal_transition_summary.json`；
- `rollout_diagnostics.json`；
- `training_trajectory_summary.json`。
- `order_shuffle_effective_strength.json`；
- `candidate_exposure_audit.json`；
- 每患者 local/intermediate/distal 的 transition count、实际毫米范围和 continuous distance–gain slope；
- Human–RNN–SNN common observables `delta_s` / `delta_h`。

阶段图：

- `figures/stage_d_interictal_distal_propagation.png/.pdf`
- 左：代表患者 observed/generated；
- 中：all/local/intermediate/distal NLL；
- 右：rollout reach 与 rank correlation。

阶段反思问题：

1. L3 是否只是在总体上增加容量，还是选择性改善 distal transitions？
2. L1/L2 是否已解释全部增益？
3. order-shuffle 是否保留相同参与集合却失去 distal benefit？
4. L1/L3 candidate exposure 是否可比，还是搜索空间本身解释了差异？

不根据答案停止后续冻结分析；各 claim 独立报告。

## Milestone E｜冻结 fields 与 pathway manifests

在读取 early-ictal target 前完成：

1. 每 arm/fit/seed 的 canonical-full 与 seed-removed fields；
2. non-collinear own_a/own_b fields 保持分开；
3. L3 与 shuffle 的 6 个语义阶段快照；
4. coarse source/target density grids、contact-space effective influence、distal reach contribution；
5. exact edge survival 只作 secondary；
6. arm-specific added-edge attenuation target sets 与 L3 local-subset controls；
7. matched-control descriptors；
8. `MODEL_FIELD_MANIFEST.json`；
9. `PATHWAY_MANIFEST.json`；
10. `ATTENUATION_MANIFEST.json`。

阶段图：

- `figures/stage_e_target_free_pathway_formation.png/.pdf`
- 真实顺序与 shuffle 的 endpoint-density/effective-influence/distal-reach trajectory；
- 不显示 early-ictal target。

冻结检查：

- field/pathway/attenuation source hashes；
- target file access log 为空；
- 代表患者和颜色/布局已冻结。

## Milestone F｜连续 attenuation

按各 arm 自己实际拥有的 added edges 运行 α=0.25/0.5/0.75/1.0：

- L1 learned extra-local，在 L1 内；
- L2 fixed random nonlocal，在 L2 内；
- L3 learned nonlocal，在 L3 内；
- L3 内 K 条匹配 local-backbone subset。

计算间期 endpoints：

- all/distal NLL；
- rollout reach；
- field fidelity。

每位患者记录有效 matched draws；少于 200 标记 `DESCRIPTIVE_ONLY`。

Local-subset matcher 固定为每 unit 最多 20,000 次无放回候选抽样；按 spec 的 weight、degree、spatial extent 与 H-support calipers 保留最多 500 个合法集合，并在 target-free 阶段预冻结 composite mismatch 最小的 16 个做精确四档 attenuation。所有 500 个 descriptors 和 16 个 target hashes 写入 manifest，不能按任何 attenuation endpoint 重新排序。

正式统计使用 attenuation slope/AUC 和 distal-selectivity `S`；双重解离使用 spec §10 的 `DD`。不为每个 alpha 分别追逐 P 值。200-draw 要求只用于 L3 local-subset controls。

本阶段同时生成所有 attenuated canonical-full、seed-removed、A/B/common/contrast fields，并冻结：

```text
ATTENUATED_FIELD_MANIFEST.json
target_access_count = 0
```

记录 checkpoint、edge-target、alpha、rollout、contact vector、support 和 producer hash。完成后 scorer 才能 unseal target。

阶段图：

- `figures/stage_f_attenuation_double_dissociation_interictal.png/.pdf`
- local 与 learned LR attenuation dose-response，配 matched controls。

## Milestone G｜冻结后 early-ictal benchmark

只在 Milestone E/F manifests 冻结后运行已有 scorer：

- clinical onset 0–10 s；
- 1–150 Hz broadband energy；
- synchronized all-contact null；
- within-shaft sensitivity；
- canonical-full primary；
- seed-removed key secondary；
- patient-first；
- 历史 LBSS physical exact join n=10，E1146 supportive；仅归档敏感性。
- 正式 RNN external benchmark 必须转到 Figure 3D 的 n=17/167 全母清单。

统计：

1. L3−L0；
2. L3−L1；
3. L3−L2；
4. seed-removed L3−L0/L1/L2；
5. 控制 interictal field fidelity 后的 model effect；
6. arm-specific added-edge attenuation AUC；
7. LR-vs-local double-dissociation；
8. learned-nonlocal attenuation 对 early-ictal margin 的 dose response。

Claim B、C 仍在 21 人 LBSS 机制子集中裁决。正式 cross-state 图和表明确标出
`n=17 patients / 167 seizures`；旧 `n=10 / 24 seizures` 只能标为 archived physical-subset sensitivity。

不允许：

- 看到 target 后改 K、threshold、checkpoint 或模型；
- 挑 early-ictal 最好的 seed；
- 把 21 人 LBSS 机制子集改写成 34 人间期 cohort；
- 把旧 n=10/24 cache 结果改写成 Figure 3D 的正式 n=17/167 外部测试；
- 将趋势写成确认迁移。
- target unseal 后重新运行模型、选择 edge target 或生成 field；
- scorer 写出任何新的模型 field。

阶段图：

- `figures/stage_g_frozen_lbss_early_ictal.png/.pdf`
- 同一患者 L0/L1/L2/L3 field 与 early-ictal field；
- cohort paired contrasts 和 attenuation。

## Milestone H｜Figure 6 候选与最终报告

主图六块：

| Panel | 产物 |
|---|---|
| A | local backbone + selective LR 模型合同 |
| B | 代表患者 local/distal observed vs generated |
| C | cohort distal propagation contrast |
| D | true-order vs shuffle 的 coarse endpoint density / effective influence / distal reach |
| E | frozen fields vs early-ictal benchmark |
| F | attenuation AUC 与正式 double-dissociation |

同时生成：

- 600-dpi PNG、PDF、SVG；
- source CSV/JSON；
- `figures/README.md` 中文逐图说明；
- `LBSS_RNN_FINAL_REPORT_ZH.md`；
- `FINAL_ACCEPTANCE.json`；
- reproducibility manifest。

视觉验收：

- 复用 paper-ready field renderer；
- TA 红、TB 蓝；
- 图内不靠长文字解释；
- 坐标、ticks、legend 主图尺度可读；
- 每张 PNG/PDF 逐张目视检查；
- 不使用 seed/fit 点冒充患者重复。

## Milestone I｜科学收口

最终报告逐项给出：

```text
CLAIM_A_LOCAL_BACKBONE_SUFFICIENT
CLAIM_B1_NONLOCAL_INCREMENT
CLAIM_B2_SELECTIVE_NONLOCAL_BENEFIT
CLAIM_C_TRUE_ORDER_SELECTS_FUNCTIONAL_SHORTCUT_ORGANIZATION
CLAIM_D1_EARLY_ICTAL_FIELD_CORRESPONDENCE
CLAIM_D2_SHORTCUT_SPECIFIC_CROSS_STATE_CONTRIBUTION
ATTENUATION_DOUBLE_DISSOCIATION
```

每项单独为 `SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE`，不设一个总 hard gate。

最后必须回答：

1. 学会传播是否需要 selective LR；
2. 若需要，增益是否只在 distal transitions；
3. learned LR 是否胜 random LR 和 extra-local；
4. 其干预是否产生功能双重解离；
5. frozen LR-dependent field 是否与 early-ictal 场存在额外对应；
6. 哪些结论是计算充分性，哪些不能外推到真实白质边/突触。

## 暂停条件

只有工程阻断才暂停并请用户决定：

- source cache/hash 不一致；
- candidate pool 小于 K 且影响多位患者；
- 具体定义为 ineligible patients >2，或任一当前 early-ictal primary 患者 ineligible；
- target 被训练/选择代码意外读取；
- 无法维持相同参数量/decoder；
- 反复 OOM 且降并发后仍无法运行。

科学阴性不是停止条件；完整执行后按 claim 分层收口。

## Definition of done

工程完成要求：

- all scheduled valid units completed；
- 0 unresolved OOM；所有恢复 OOM/retries 有日志；
- 所有产物时间与 producer/config/input hash freshness 一致；
- aggregation 不混入旧 cohort 或旧 source revision；
- 每个输入文件的 producer hash 通过；
- target unseal 后 source snapshot 不变；
- scorer 不生成新 field；
- figure source rows 可回溯到 patient-level tables；
- PNG/PDF/SVG 均完成逐图目视验收。
