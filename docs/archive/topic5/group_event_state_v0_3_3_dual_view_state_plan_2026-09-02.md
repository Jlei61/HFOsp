# Group-Event State v0.3.3 执行计划（复审修订稿）

**状态：** `V0_3_3_REVISED_DRAFT_FOR_REVIEW_DO_NOT_EXECUTE`

**复审前判定：** `V0_3_3_MAJOR_REVISION_BEFORE_EXECUTION`

**对应 spec：** [`group_event_state_v0_3_3_dual_view_state_spec_2026-09-02.md`](group_event_state_v0_3_3_dual_view_state_spec_2026-09-02.md)

**当前动作：** 只完成设计修订；等待用户审阅后才能启动新实验。

## 0. 执行原则

这轮不再按“先 assay 全部通过，再训练，再跑科学实验”的长串行链执行。三条工作流并行推进，但使用同一数据合同、同一 evaluator 和同一证据身份。

真正的全局硬停只有三条：

1. sealed partition 被触碰；
2. 出现时间、患者、发作、normalization 或 target 泄漏；
3. canonical evaluator 对同一 checkpoint/anchor 给出不同分数。

其他失败不封锁项目：结果降为 `DIAGNOSTIC`，训练不足交给 Training Laboratory，支持不足退出对应分母，objective mismatch 单独报告。

## 1. Core Definition of Done

本轮核心完成需交付：

1. canonical evaluator 一致，E1146 方向差异被定位；
2. Level 0–2 oracle 与 D0–D4 synthetic/power curve 完成；
3. `S_N` 和 `S_G` 分别获得 `TRAINING-ADEQUATE` 配方与 `training_card.json`；
4. R0/R1 在 2 位 tuning patients 完成选择，并锁定到 4 位 untouched development patients；
5. within-view、cross-transfer、shared/private、H1、H2a 完成；
6. frozen H2b risk 完成；
7. 白话版、技术版、机器 JSON、核心图和 figures README 完成；
8. sealed partition 仍未打开。

以下不进入核心完成条件：R2 waveform、D5、small shared producer、repaired gated、60/120 min、early ictal field/path、background observer、人体 H3。

## 2. 统一登记表和权限

### 2.1 模型登记

建立 `checkpoint_registry.json`，至少登记：

```text
H_rate baseline
H_mark baseline
S_N / R0
S_N / R1
S_G-primary / R0
S_G-primary / R1
S_G-composite / R0
S_G-composite / R1
```

每项记录 config hash、code commit、split hash、normalization hash、checkpoint hash、selected inner-validation step、training evidence label 和 sealed flag。

### 2.2 结果身份

任何结果必须带：

```text
DIAGNOSTIC
TRAINING-ADEQUATE
COMPARABLE
LOCKABLE
```

非 canonical evaluator 的人体结果可以并行产出，但只能标 `DIAGNOSTIC`，不得进入主汇总或作阴性结论。

### 2.3 三条工作流

| 工作流 | 负责 | 禁止 |
|---|---|---|
| A Evaluator & Assay | evaluator、E1146、DGP、power、eligibility、数据边界 | 调人体 LR、按结果换架构 |
| B Training Laboratory | T0–T6、搜索、训练卡、资源调度 | 改科学 endpoint、split、H、H2b 标签 |
| C Scientific Experiments | R0/R1、dual-view、sharedness、H1/H2a、frozen H2b、探索接口 | 临时改 optimizer/LR 或用 H2b 回选 state |

## 3. Workstream A：Evaluator、Assay 与数据合同

### A1. Canonical per-anchor evaluator

建立单一 per-anchor 表：

```text
subject / seed / checkpoint_hash / anchor_time / split
target / prediction_H / prediction_H_plus_state
shared_dispersion / mask / weight / per_anchor_NLL
eligibility / evidence_label
```

训练 branch、独立 evaluator、figure payload 均从同一表或同一纯函数重算，禁止各自复制 NLL、dispersion 或 reduction。

**验收：** 同一对象逐行分数在预设浮点容差内一致；测试覆盖 anchor permutation、mask、dispersion、intercept、weight 和 reduction。

### A2. E1146 discrepancy

对 v0.3.2 的 +0.1277 与 −0.3291 逐行 diff：

1. checkpoint/hash；
2. anchor set；
3. target 与 prediction；
4. dispersion/intercept；
5. block weight；
6. seed aggregation；
7. score sign/reduction。

输出首个产生差异的数据行和计算步骤，不能只给“已统一代码”的结论。

### A3. Oracle Level 0–2

- Level 0：真 state + 训练 output head；
- Level 1：真 event innovation + fixed leaky scan + readout；
- Level 2：synthetic mark channel + event encoder + readout。

每层保存 truth、prediction、continuous gain、failure location 和 false-positive readout。

### A4. D0–D4 与 power curve

- D0 H-only；
- D1 count-only；
- D2 grammar-only；
- D3 shared state；
- D4 independent states。

使用真实时间轴、coverage、split 和 support pattern。每次代码变化先 3 replicates smoke；夜间 10 replicates；里程碑版本 20–30 replicates。

效应以 oracle held-out deviance gain/block SNR 表示，输出连续 power curve，不用任意 β 或单个 pass count。

### A5. Eligibility

从 medium-oracle effect 的 power curve反推每个 endpoint/horizon 所需独立 blocks。资格计算必须复用真正 window builder 的 coverage segment，禁止用粗 session 数近似。

### A6. D5 与数据边界

D5 background-only 只运行少量预期失败例，验证 event-only state 的理论边界，不与 D0–D4 同等消耗。

同时冻结：

- target 不跨 gap/split/seizure；
- 发作/立即 postictal 不更新 state；
- autonomous flow 继续；
- gap/session reset；
- seizure hard reset 仅敏感性。

### A7. 交付

```text
canonical_evaluator.json
e1146_discrepancy_audit.json
oracle_level_0_2.json
d0_d4_power_curve.json
eligibility_by_endpoint_horizon.json
data_boundary_audit.json
```

## 4. Workstream B：Persistent Training Laboratory

Workstream B 长期运行，不因一次 staged search 结束。它对每个模型执行相同训练合同，但允许不同架构在等搜索预算下选择各自最佳超参数。

### B0. 训练请求接口

每个请求必须固定：

```text
scientific_target
input_view
state_family
split_hash
baseline_H
endpoint_and_reduction
search_budget
resource_ceiling
```

Training Laboratory 不得改这些字段，只返回训练状态和候选配置。

### B1. T0 数值/梯度路径

对 `S_N` 和 `S_G` 分别执行：

- tiny-slice overfit；
- oracle head；
- state/write Jacobian；
- optimizer membership；
- 每组参数 first-active step；
- gradient/update norm；
- clipping fraction；
- AMP 下小梯度检查；
- state-to-output modulation。

gate 从 step 1 可训练，较小 LR 配合全局 warm-up。任一参数组尚未开始训练的 checkpoint 不得 eligible。

### B2. T1 LR/optimizer/schedule

- parameter-group LR：log-uniform `[1e-5, 3e-3]`；
- optimizer：AdamW、Adam、RMSprop；
- schedule：constant、cosine、ReduceLROnPlateau；
- warm-up：0%、5%、10%。

先在 2 位 tuning patients 的 TRAIN/inner-validation 和 synthetic 上运行，不读取 development evaluation。

### B3. T2 初始化和归一化

- Xavier/orthogonal；
- write scale `0.01/0.1/1`；
- alpha `0.01/0.03/0.1`；
- gated bias；
- z-score vs robust input scaling；
- hidden none vs LayerNorm；
- state 不做 per-time LayerNorm。

所有统计量只用 TRAIN 固定。

### B4. T3 容量和时间结构

- depth 1/2/3；
- width 32/64/128；
- ReLU/GELU/SiLU；
- dropout 0/0.1；
- write width 2/4/8；
- time bank `{5,30,120}` vs `{10,60,180}` min；
- NB dispersion frozen vs low-LR；
- anchor-balanced vs event-balanced sampling。

`S_N` 的总状态维数为 6/12/24；选择 near-best 中最小者。fixed leaky 做完整 chronological scan。若运行 gated exploratory，TBPTT 为 30/60/120 min，chunk 边界 carry+detach、不 reset。

### B5. T4 多保真搜索

使用 ASHA/Hyperband：

1. low budget × 1 seed；
2. top candidates × 3 seeds；
3. top 2–3 configs × 5 seeds。

grace period 必须晚于全部参数激活，并至少包含一个 blocked validation interval。

### B6. T5 失败驱动调参

| 观察 | 下一动作 |
|---|---|
| tiny overfit 失败 | 查路径、容量、LR、归一化 |
| TRAIN 学会、inner-val 不学 | 正则化、容量、objective/support |
| count 学会、grammar 不学 | 调 `S_G` target/采样，不改 `S_N` |
| selected step 在预算末端 | 增加预算后再判断 |
| 两轮 search batch 无改善 | 停止盲搜，分类失败 |
| random reservoir 等价 | trained encoder 未提供可识别增量 |

### B7. T6 锁定训练卡

每个候选产生：

```text
training_card.json
learning_curves.parquet
parameter_group_diagnostics.json
seed_stability.json
capacity_and_rank.json
```

训练卡包含 best step、plateau、seed dispersion、gradient/update、clipping、state variance/rank、random-reservoir delta、shift-null 和 output modulation。

只有 tiny overfit、synthetic recovery、blocked inner-validation 均成立，且 checkpoint 非 warm-up/预算边界，才标 `TRAINING-ADEQUATE`。

### B8. 训练服务停止条件

单个模型在以下任一情形收口：

1. 得到可锁定的 training-adequate 配方；
2. validation 已稳定 plateau；
3. 连续两个 search batch 无改善；
4. 失败被定位为 objective/support/data，而非优化问题。

## 5. Workstream C：科学状态实验

### C1. 显式历史阶梯

同一 anchor 与 score 下建立：

```text
H_rate
H_mark
H_mark + state
```

count 与 grammar 各自使用相应 proper score，不混成平均 accuracy。

### C2. R0/R1 输入

- R0：summary token；
- R1：contact-resolved participation/delay/multiband/geometry token。

先在 2 位 tuning patients 比较，再锁定。R1 相对 R0 的结论只在搜索预算相同、训练卡均合格时成立。

### C3. `S_N`

目标固定为：

```text
[N_0–5, N_5–15, N_15–30]
```

主报告同时给 `N_0–30`。比较 write width 2/4/8，使用 inner-validation near-best 最小容量规则。

### C4. `S_G`

运行：

1. `G-primary`：subset identity only；
2. `G-composite`：subset primary + low-weight continue/size。

仅对 `N_future>0` anchor 评分；同时输出 first-future-event 和 block-average。记录多任务 gradient cosine；没有实测持续冲突时不启用 PCGrad。

### C5. Within-view 与 cross-transfer

冻结 state producer，在 later blocks 运行：

```text
S_N → count
S_N → grammar
S_G → grammar
S_G → count
```

所有 probe 使用同一 TRAIN-only 容量控制和同一 evaluator，不根据 cross-transfer 回改 state producer。

### C6. Shared/private

只用 TRAIN 拟合 regularized CCA 或 reduced-rank regression，得到：

```text
Z_shared / U_N / U_G
```

在 later blocks 分别评价 count、grammar、H2a、H2b。只有两个 within-view 均有效且双向 transfer/shared-subspace 都无效，才允许写“可分离 predictive states”。

### C7. H1/H2a

H1 在固定物理时间 anchor 上比较：

```text
H_rate → H_mark → H_mark+S_correct → H_mark+S_shifted
```

主 horizon 为 5/30 min。shift 为同 session block circular shift，偏移大于 target horizon。

H2a 使用冻结 pre-event state 与冻结 contact decoder 主干，低容量 adapter 将 state 接入 continue、size、subset heads；subset identity 为主端点，later continuation 为冻结 transfer。

这里分两步：`S_G` 训练时 decoder 主干固定、梯度经 state adapter 更新 producer；H2a 正式 transfer 时 producer 与 decoder 均冻结，只在 TRAIN 拟合容量受限 probe，再到 later blocks 评分。

### C8. 两位 tuning 到四位 untouched replication

在 2 位 tuning patients 上锁定：

- R0/R1；
- `S_N/S_G` target；
- state dimension；
- `G-primary/G-composite`；
- probe capacity；
- shift offsets；
- evaluator 与 checkpoint rule。

随后对 4 位 untouched development patients 单次运行，不再调参。每位患者先合并 seeds，再做 patient-first 汇总；同时给 per-patient 和 estimability。

### C9. Frozen H2b risk

在 seizure outcome 完全不参与 state 选择的前提下，训练单一离散 survival head：

```text
baseline
baseline + S_N
baseline + S_G
baseline + Z_shared
baseline + capacity-matched [S_N,S_G]
```

报告 Brier skill、log score、calibration；AUROC secondary。使用 forward held-out seizures/rolling origin。H2b 可以在 H1 阴性时运行，但不得借 H2b 结果回选 state。

### C10. 探索接口

- R2 masked/partial waveform learning；
- small shared producer：D3/D4 + 2 tuning patients；
- early ictal field/path：仅低维 target、支持足够患者；
- repaired gated：只在 fixed leaky training-adequate 后仍有明确容量证据时进入人体比较。

这些结果独立标记，不进入四位 replication 的核心结论。

## 6. 科学 Goals 与交付顺序

### Goal 1：证明尺子和训练能测到已知状态

交付 A1–A5、B1–B2。关键判断是 evaluator 一致、Level 0–2 可恢复、medium effect 的 power 与资格分母可计算。

### Goal 2：确定事件数据中有什么预测信息

交付 C1–C4。回答 `H_rate→H_mark→H_mark+S`，以及 R0/R1、count profile 和 conditional subset grammar 各自增加了什么。

### Goal 3：检验 sharedness

交付 C5–C6：within-view、bidirectional cross-transfer、shared/private 和小型 shared-producer 诊断。

### Goal 4：四位 untouched development replication

在 tuning 配方锁定后执行 C8。这一步不重新搜索。

### Goal 5：冻结 H2b risk

执行 C9，判断纯间期 predictive state 是否跨到发作风险；不以 H2b 选择模型。

### Goal 6：R2 与 early ictal field（探索）

可与核心并行，但不阻塞 Goal 1–5，也不进入 Core DoD。

### Goal 7：background 与 H3（后续）

只有在 event-only 边界清晰后另写 spec。H3 必须比较 common-drive/readout-only 与 explicit feedback，不从普通 hidden update 推因果。

## 7. 里程碑，不作为隐藏总 gate

| 里程碑 | 最小交付 | 允许动作 |
|---|---|---|
| M0 评分统一 | canonical evaluator + E1146 diff | 主结果可升级为 COMPARABLE |
| M1 训练实验室 | `S_N/S_G` training cards | 可判断 fixed leaky 是否训练到位 |
| M2 两人 tuning | R0/R1、目标和容量锁定 | 可冻结 replication manifest |
| M3 四人 replication | 单次 untouched 结果 | 可形成 development 结论 |
| M4 sharedness | cross + shared/private | 可分级命名 shared information |
| M5 frozen H2b | risk skill/calibration | 可判断跨任务价值 |

assay 尚未完成时可跑探索代码，标 `DIAGNOSTIC`；H1 尚未阳性时可跑 frozen H2b，但不得选择 state；R2 可并行，不拖住 R0/R1。

## 8. 资源与持久运行合同（批准执行后）

1. 使用 `nohup`/`setsid` 或 tmux，训练不依赖网络会话；
2. 每个 CPU worker 固定 `OMP_NUM_THREADS=1`；
3. 每种模型先测单作业 GPU peak 和 host RSS，再设不会 OOM 的并发；
4. GPU 填满但保留安全余量；患者、seed、synthetic replicate 可并行；
5. 同一 output key 只允许一个 writer；manifest 原子写入；
6. config hash 去重、断点续跑、日志和 checkpoint 可恢复；
7. 监督器只读状态，不拥有训练进程；
8. OOM 自动降 batch/启用 checkpointing 后重试，并记录 effective batch/optimizer steps；
9. 比较模型使用相同 effective search budget，而非相同 wall time。

## 9. 图和报告

### 9.1 核心图

1. **Assay and power：** Level 0–2、D0–D4 power、可估分母；
2. **Event-derived state：** `H_rate→H_mark→H_mark+S` 的 count/grammar proper-score gain；
3. **Sharedness：** within-view、cross-transfer、`Z_shared/U_N/U_G`；
4. **Frozen H2b：** risk skill 与 calibration。

### 9.2 辅助图

- Training Laboratory dashboard；
- R0/R1 消融；
- correct vs block-shift null；
- per-patient support/estimability；
- R2、small shared producer、early-field。

每个 `figures/` 目录同步中文 README；PNG/PDF 使用同一 payload 和 producer，并目视检查标签、分母和结论方向。

### 9.3 两份报告

- 白话版：问题 → 决定性对照 → 实际数据 → 结论 → 边界；
- 技术版：配置、hash、分母、逐患者/seed、训练卡、统计、复现命令、sealed 状态。

机器报告必须能区分工程 PASS、assay power、training adequacy、comparability 与科学结果。

## 10. 开始前清单

在用户批准本稿前：

- 不启动 v0.3.3 新训练；
- 不打开 sealed partition；
- 不运行人体 H3；
- 不把 v0.3.2 阴性升级为 fixed-leaky 架构阴性；
- 不把 R2、early field 或 gated model变成核心 blocker；
- 不 commit/push 本稿，除非用户明确要求。
