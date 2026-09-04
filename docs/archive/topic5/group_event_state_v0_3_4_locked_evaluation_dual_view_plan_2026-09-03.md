# Group-Event State v0.3.4 — 执行计划（草案）

**状态：** `SUPERSEDED_BY_MULTIVIEW_PREDICTIVE_STATE_PLAN`
**日期：** 2026-09-03
**取代版本：** [`group_event_state_v0_3_4_multiview_predictive_state_plan_2026-09-03.md`](group_event_state_v0_3_4_multiview_predictive_state_plan_2026-09-03.md)。本文件保留用于追溯，不再作为执行依据。
**Spec：** `group_event_state_v0_3_4_locked_evaluation_dual_view_spec_2026-09-03.md`
**前置：** v0.3.3 可训练性收口复审修订（`group_event_state_v0_3_3_trainability_closeout_technical_2026-09-03.md` R0–R7）；Agent A eligibility 表；patient role lock。
**原则：** 三条 track 并行，只有 sealed violation / 真实泄漏 / canonical evaluator 对同一对象给出不同分数是全局硬停；其余失败按证据级别降级。任何 track 都不得因 development 读数回调参数。

## 0. 先后顺序（为什么这样排）

```text
Track A  基线阶梯 + 自校正验收  ──►  锁定评价 9 位 S_N（一次性 development）──►  optimism gap 报告
Track B  Training Lab 合同更新（rolling inner-val、预算阶梯、四段 adequacy、period_offset_control 已入卡）
Track C  S_G：合成 D3 Level-2 ──► 自有 recipe 搜索 ──► R1 人体训练卡 ──► 冻结 ──► cross-transfer / H2a / H2b
```

A 先于 C 的人体部分：C 的每个人体比较都要在 A 交付的自校正基线上做；B 的合同更新先于 A/C 的任何新训练。

## 1. Track A：基线阶梯与锁定评价（Agent A + C 合同，Agent B 提供 checkpoint）

### A1 基线阶梯实现（Agent C）

- [ ] `H_rate`：last IEI、5/30/120 min EWMA、**3/6/12/24 h 长窗口**、clock/session、coverage、因果自校正项（shrinkage 拟合，TRAIN 上选 W）。
- [ ] `H_mark`：在 `H_rate` 上加 extent/STOP EMA、contact/repertoire occupancy、multiband EMA。
- [ ] `H_nonlinear`：同 head 容量 MLP，只吃 `H_mark` 特征。
- [ ] 自校正验收脚本：三段（TRAIN / STATE_SELECTION / DEVELOPMENT）× 三窗 count/μ 比值，容差 `[0.8, 1.25]`；不合格层标 `MISCALIBRATED`。
- 验证：v0.3.3 的 9 位患者在新 `H_mark` 上 STATE_SELECTION 比值回到容差内（原来 E1096 2.0–2.2、E548 1.8–3.3、E1146 1.3–1.45）。
- 产物：`h_ladder_artifact_contract.json`、每患者 `h_ladder_<subject>.npz` + calibration 表。

### A2 canonical evaluator 扩展（Agent A）

- [ ] 每 anchor 表增加阶梯四层与全部对照臂列：`period_mean`、`segment_mean`、`times_only`、`mark_shuffle`、`linear_marked_ema`、`random_reservoir[k]`、`shift[j]`。
- [ ] seed × block nested bootstrap；≥32 shift 的 null 分位数；10–20 reservoir 分布。
- [ ] evaluator hash 登记进训练卡（关闭 `selection_metric_is_canonical=false`）。
- 验证：D0（H-only）假阳性≈5%；D1 上 `period_mean` 臂不得吃掉真状态的增量（planted 慢漂移 vs planted 快状态两种 DGP 分别校准）。
- 产物：`canonical_per_anchor_table.parquet` schema、`multi_shift_null.json`、`random_reservoir_null.json`。

### A3 锁定评价当前 9 位 `S_N`（一次性；supervisor 发 release）

- [ ] 冻结清单：9 张训练卡 sha + 45 个 learned checkpoint sha + 45 个 random checkpoint sha（已在 `incremental_summary.json`）。
- [ ] 一次性 development 读取，写 ledger（subject, endpoint=count_profile_30min, state_version=v0.3.3, sha, time）。
- [ ] 输出 `sn_locked_development_evaluation.json`、`hpo_optimism_report.json`（STATE_SELECTION − DEVELOPMENT，逐患者、逐臂）。
- 判读：按 spec §2 E1–E3 逐级；预期多数停在"基线水平失准"。
- 分歧点（spec §15.1）：是否只对 E1096/E1146/E548/E922 四位做 A3、保留其余 5 位为新基线下的 clean 评价——审阅决定后执行。

### A4 replication cohort

- [ ] 从 `eligibility_by_endpoint_horizon.json` 的可估集合中，排除已触碰 9 人与 tuning 2 人，按 support 排序取 ≤6 位；不足 6 位如实写明。
- 产物：`replication_cohort_manifest.json`。

## 2. Track B：Training Laboratory 合同更新（Agent B）

- [ ] **rolling inner-validation**：TRAIN 尾部滚动块用于配方与步数选择；STATE_SELECTION 只做最终候选比较与 optimism 记录。`DataView` 新增 `rolling_inner_val` phase；`assert_no_dev_test` 不变。
- [ ] 预算阶梯 `300 → 900 → 2700 → 8100`；best-step 落在末端自动延长（`budget_edge` → 追加 rung，不是分类结束）。
- [ ] 每模块独立 LR（encoder / write / head / state），三种 optimizer，三种 schedule；不移植任何其它目标的冻结配方（S_G 不得用 S_N 的 O1）。
- [ ] adequacy 四段：debug overfit（tiny slice 按配方自身预算）、convergence（plateau）、synthetic recovery（D1/D2/D3）、blocked generalization（rolling inner-val）。
- [ ] 训练卡：已含 `period_offset_control`（2026-09-03 入卡）；新增 `h_ladder_gains`（四层增量）、`rung0_fraction_beating_baseline`、`recommended_next_batch`。
- [ ] 停机：连续两轮搜索无改进 + plateau 才 `optimization_exhausted`；默认 `max_batches ≥ 3`。
- [ ] `S_N` recipe portfolio：在自校正基线上，两位 tuning 患者 R0/R1 消融；其余 7 位只在 A3 决策后重训。
- 测试：现有 `tests/test_group_event_state_v033_training_lab_*.py` 全绿 + 新增 rolling inner-val 泄漏测试（rolling 块必须早于 STATE_SELECTION 且不含 development）。

## 3. Track C：S_G、cross-transfer、H2a、H2b（Agent C + B）

### C1 S_G 合成门

- [ ] R1 token 进 S_G encoder；subset 目标用 conditional-Bernoulli DP 打分、并"按评分方式训练"（block 起点冻结状态、(anchor,event) 对、1/N_future 加权）。
- [ ] 合成 D3 Level-2：`ci_low>0` 三个独立 seed；D0 假阳性≈名义；D4 不错误合并。
- 未过 → 停在 C1，分类 encoder/objective/optimizer，不进人体。

### C2 S_G 人体（两位 tuning 患者）

- [ ] 自有 optimizer/LR/budget 搜索（Track B 合同）；`NO_LEARNING`（best step 在第 1–10 步且梯度范数 <1e-2）的配方不得冻结。
- [ ] 训练卡（含 `period_offset_control`、慢库对照）。
- [ ] `G-primary` vs `G-composite` 两臂；multiband 冻结 probe。
- 产物：`sg_state_registry.json`、`future_grammar_results.json`。

### C3 慢库对照（S_N 与 S_G 各一组）

- [ ] 6/12/24 h 时间常数臂，在自校正基线上 learned vs random；v0.3.3 复审第一版结果（`slow_bank_arms_on_{recal,mark}`）作为参照。
- 判读：learned 稳定胜出 → "依赖事件内容的慢分量"；打平 → 归基线。

### C4 冻结与 cross-transfer

- [ ] 冻结 `S_N`（Track A 或 B 的合格版本）与 `S_G`；within-view、cross-transfer 矩阵、CCA/RRR shared-private；容量匹配低秩投影。
- 产物：`cross_transfer_matrix.json`、`shared_private_state_report.json`。

### C5 current-event H2a

- [ ] 事件锚点；`H`、`H+S_N`、`H+S_G`、`H+[S_N,S_G]`、shifted、random；continue/STOP、positive size、contact identity | K、later continuation；primary `contact identity | K, prefix`。
- 产物：`h2a_current_event_results.json`。

### C6 frozen H2b（development-only）

- [ ] `B(t)` 按 spec §9 补齐（含长窗口 IED rate、time since last seizure、postictal/cluster、可得的 medication/stimulation）；5 min 网格 discrete hazard；5/15/30/60/120 min；patient-level Brier/log/calibration；按 seizure pattern 分层。
- [ ] early-ictal 低维 target 探索。
- 产物：`frozen_h2b_development.json`。

### C7 R2（探索，不进 Core）

- [ ] masked-contact、early-to-late、cross-band、held-out temporal block 四类 learning audit。

## 4. 本轮立即可执行（不需要 release）

以下不读 development、不改 target，可在审阅期间先做：

1. A1 的自校正验收脚本 + 在 v0.3.3 的 9 位患者上对现有 H_mark 出三段比值表（已有 STATE_SELECTION/TRAIN 两段：`offset_drift_control/*.json`）。
2. B 的 rolling inner-validation `DataView` phase + 泄漏测试。
3. C1 合成 D3 的 S_G encoder/objective 修复与 Level-2 重试（GPU）。
4. C3 慢库对照的第二版（自校正基线一旦可用即重跑；当前第一版在 `slow_bank_arms_on_{recal,mark}`）。

## 5. 交付物

```text
h_ladder_artifact_contract.json / h_ladder_<subject>.npz
canonical_per_anchor_table.parquet  multi_shift_null.json  random_reservoir_null.json
sn_locked_development_evaluation.json  hpo_optimism_report.json  replication_cohort_manifest.json
sg_state_registry.json  future_grammar_results.json  cross_transfer_matrix.json
h2a_current_event_results.json  shared_private_state_report.json  frozen_h2b_development.json
training cards（含 period_offset_control / h_ladder_gains / rung0 比例）
白话 / 技术 / 机器报告；核心图：阶梯增量、E1–E5 判定表、cross-transfer 矩阵、H2b skill/calibration
```

## 6. 硬停止与暂停

- 硬停：sealed 打开；真实泄漏；canonical evaluator 同对象不同分。
- 暂停等用户：A3 的分歧点（spec §15.1）；replication cohort 不足 6 位；需要外部临床标签。
- 不是停止：单患者阴性、某端点阴性、S_G 合成门未过（转 C1 修复）、某配方训练困难。
