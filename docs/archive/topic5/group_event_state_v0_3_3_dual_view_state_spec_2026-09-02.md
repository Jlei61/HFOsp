# Group-Event State v0.3.3 Scientific Spec（复审修订稿）

## Assay-Calibrated Dual-View Predictive State

**状态：** `V0_3_3_REVISED_DRAFT_FOR_REVIEW_DO_NOT_EXECUTE`

**复审前判定：** `V0_3_3_MAJOR_REVISION_BEFORE_EXECUTION`

**日期：** 2026-09-02

**本稿用途：** 等待科学审阅；不得据此打开 sealed partition 或直接启动完整人体队列。

**继承：** v0.3.2 的群体事件数据底座、显式历史、时间分区、measurement manifest、seizure-label 隔离和 sealed lock。

**不继承：** v0.3.2 的 assay 通过判断、单一 count state、test-best-control、固定三人主队列，以及“训练跑完即说明架构无效”的解释。

## 0. 一句话科学目标

我们从连续群体间期事件中分别学习“未来事件负荷”和“未来事件空间 grammar”的预测状态；先证明评分器一致、网络确实学会，再检验两个视角含有多少共享信息，以及完全由间期任务训练的状态能否跨任务预测发作风险。

本轮不要求一个状态同时解释所有端点，也不把 RNN 的 hidden update 直接解释为 IED 对生理网络的塑形。

## 1. 术语与结论层级

| 规范术语 | 本稿定义 | 禁止替代说法 |
|---|---|---|
| `H_rate(t)` | IEI、多尺度事件数、clock/session、coverage 等负荷历史 | 无状态、无历史 |
| `H_mark(t)` | `H_rate` 加 extent/STOP、contact occupancy、multiband EMA | state |
| count-view state `S_N(t)` | 由未来负荷 profile 训练的候选状态 | susceptibility state |
| grammar-view state `S_G(t)` | 由未来事件的条件触点集合训练的候选状态 | propagation mechanism |
| conditional grammar state | 只在未来 block 确有事件时定义并评分的 `S_G` | 完整点过程状态 |
| shared predictive information | 两个视角之间存在可重复的跨任务或共享子空间信息 | 两个状态必为同一生理变量 |
| strong shared-state evidence | 双向 cross-transfer，并有 shared-subspace 支持 | shared architecture 自带的结果 |
| training adequacy | 已通过小样本过拟合、合成恢复、blocked inner-validation，且非 warm-up/预算边界选点 | 训练损失下降 |
| functional state | 由冻结未来预测读出定义的状态功能 | 某个 latent 坐标本身 |

所有人体结果另标四级证据身份：

1. `DIAGNOSTIC`：尺子或训练仍未完整合格；
2. `TRAINING-ADEQUATE`：该模型已证明能学指定目标；
3. `COMPARABLE`：评分器、分母、搜索预算与选择合同一致；
4. `LOCKABLE`：可冻结进入四位 untouched development replication。

## 2. 核心假设

### H1：群体间期事件历史能否形成跨真实时间的预测状态

相对 `H_rate` 与 `H_mark`，状态是否继续改善未来 5、30 分钟的：

- 事件负荷 profile；
- 在未来确有事件时的条件 contact grammar。

正确时刻状态还应优于保持自相关结构的同 session block shift。60/120 分钟只作支持充分时的探索，不进入本轮核心完成条件。

### H2a：状态是否改变具体事件的后续空间表达

给定当前事件已经出现的 early prefix，冻结 pre-event state 是否额外预测：

- 是否继续传播；
- 继续时的招募规模；
- 给定规模后具体招募哪些触点；
- later continuation / same-prefix branch。

主端点为 `subset identity | K,prefix`；STOP、size 和 later continuation 分项报告，不混成一个总分。

### H2b：纯间期状态是否跨任务预测发作风险

在临床与历史 baseline 之外，冻结的 `S_N`、`S_G`、共享成分或私有成分能否改善离散 seizure hazard。H2b 允许在 H1 尚弱时作为 development 探索运行，但不得反向选择 state、输入、超参数或 checkpoint。

### H3：IED 是否反过来塑造状态

H3 保留为后续独立机制问题。本轮不运行人体 H3，也不把 observer 看到事件后的信息更新称为生理反馈。

## 3. 统一科学图

```text
背景生理过程 / 临床与记录背景
                │
                ├──────────────→ H_rate(t), H_mark(t)
                │
                ▼
          潜在易感过程 S(t)
           ↙             ↘
 群体间期事件 X(t)      发作风险 Y(t)
           │
           └──── ? ───→ S(t+)       [H3，后续]
```

v0.3.3 只问 `X(t)` 是否含有超出显式历史的未来预测信息，以及负荷视角与 grammar 视角共享多少信息。它不预设 shared state，也不以 hidden-state 命名替代证据。

## 4. 数据、时间和发作边界

1. TRAIN、chronological inner-validation、development evaluation 按真实记录时间切分；任何 target 不跨 split、未记录 gap 或 seizure onset。
2. H1 与 H2b 主要使用固定物理时间网格，避免高 IED-rate 时段自动贡献更多 anchor；H2a 使用 event anchor。
3. 发作和 immediate postictal 事件不更新间期状态，状态只按其已定义的 autonomous flow 演化；真实 gap/session 边界才硬 reset。发作后 hard reset 仅作敏感性。
4. `time since last seizure`、seizure-cluster/postictal indicator 进入 H2b baseline，不偷偷进入间期 state producer。
5. normalization、contact support、vocabulary、cluster/target construction、patient adapter 与 checkpoint selection 只读取允许的时间前缀。
6. 旧 contact decoder 若复用，必须通过嵌套时间泄漏审计；34 位患者 × 3 seeds 是 inference bundles，不得写成 102 个独立预训练模型。
7. sealed partition 保持未打开；本轮全部是 development。

## 5. 显式历史阶梯

所有 state 增量均在同 anchor、同评分器下依次比较：

```text
H_rate  →  H_mark  →  H_mark + S
```

### 5.1 `H_rate`

- time since last IED；
- 1/5/30/120 min event count 或 EWMA；
- clock、session position、coverage/gap distance。

### 5.2 `H_mark`

在 `H_rate` 上增加：

- extent/STOP EMA；
- contact participation/occupancy；
- event repertoire occupancy；
- multiband expression EMA。

H2b baseline 另加入 time since last seizure、postictal/cluster、可获得的 medication/stimulation 和 sleep/background proxy。若只有 time-of-day，报告为 clock adjustment，不称控制了 vigilance。

## 6. 核心输入：R0 与 R1

### R0：summary token

复用 participation、tied-group、relative delay、dispersion、multiband 和 cross-band summary。R0 是解释性、低容量基线。

### R1：contact-resolved event token

保留逐触点 participation、精确 delay、逐频带能量与峰时、shaft/coordinates 和 validity mask，经低容量 contact encoder 聚合。R1 是本轮核心输入，用于检验 summary pooling 是否丢失病理网络信息。

### R2：within-event waveform token（探索）

R2 读取 event-core bipolar/CAR 波形与多频带时空 patch，但不进入核心 Definition of Done。它必须用 masked/partial prediction 证明学到了可泛化信息，例如：

- 遮住部分触点后预测 participation；
- 从其他触点或 early waveform 预测 held-out contact delay；
- 低频预测 held-out 高频表达，或反向；
- early segment 预测 later propagation。

简单重建输入中已经显式存在的标签不算 learning audit 通过。

## 7. Count-view state `S_N`

### 7.1 目标：future-burden profile

`S_N` 不再只预测一个低秩的 30 分钟总数，而预测：

```text
[N_0–5min, N_5–15min, N_15–30min]
```

`N_0–30min` 作为最直观的主要报告量，由上述分段共同决定。count likelihood 使用同一 canonical negative-binomial evaluator，并让有效观测区间内的“没有事件”进入目标。

### 7.2 状态与动力学

```text
event token → event encoder → event write
event write → 5/30/120 min fixed leaky bank → S_N(t)
H_rate/H_mark + S_N(t) → future-burden residual
```

真实 `dt` 只推动已声明的 leaky dynamics；未记录时间不能作为无事件证据。

### 7.3 容量选择

每个时间尺度的 write width 比较 `{2,4,8}`，总状态维数对应 `{6,12,24}`。只在 TRAIN/inner-validation 中选择“落在最佳表现容差内的最小容量”，并同时报告 supervised readout rank、state covariance rank 和 random-reservoir 增量。

## 8. Grammar-view state `S_G`

### 8.1 主目标只保留 subset identity

`S_G` 的主要训练目标为：

```text
contact subset identity | positive size K, early prefix
```

这是最接近“相同开头为何走向不同后续路径”的目标。

首轮只比较两个预先定义的训练臂：

1. `G-primary`：只训练 subset identity；
2. `G-composite`：subset 为主，continue/positive-size 以小权重辅助。

later continuation 作冻结读出；multiband expression 首轮只作冻结 probe，不与 subset 共同塑造 `S_G`。

### 8.2 条件 future-block 评分

在 anchor 时刻取得 `S_G(t)` 后，整个 future block 内使用同一个 anchor state；block 内后来发生的事件不得更新这个待评分状态。只对未来 block 确有事件的 anchor 定义 grammar loss：

```text
1[N_future > 0] × mean_event(subset loss)
```

同时报告：

- first-future-event grammar；
- future-block average grammar。

这明确回答条件事件形态，不把它伪装成完整的联合 point-process likelihood。未来是否有事件由 `S_N`/count 视角单独回答。

### 8.3 多任务冲突审计

`G-composite` 必须记录 subset、continue、size 各任务 gradient norm 与 cosine。只有持续冲突被实测确认时才启用 PCGrad；不把复杂多任务优化预先写入主架构。

## 9. Sharedness：分级证据，不做二元裁决

冻结 `S_N` 与 `S_G` 后，在 later development blocks 运行：

1. within-view：`S_N→count`、`S_G→grammar`；
2. cross-transfer：`S_N→grammar`、`S_G→count`；
3. shared/private：只用 TRAIN 拟合 regularized CCA 或 reduced-rank regression；
4. capacity-matched concatenation：组合状态先用 TRAIN-only 低秩投影匹配探针自由度。

shared/private 分解写为：

```text
S_N = A_N Z_shared + U_N
S_G = A_G Z_shared + U_G
```

`Z_shared`、`U_N`、`U_G` 均在 later blocks 独立评价 count、grammar、H2a 与 H2b。

### 9.1 允许结论

| 观察 | 允许命名 |
|---|---|
| 两个 within-view 都无增量 | 未发现可比较的 dual-view state |
| 只有一个 within-view 有增量 | 单视角 predictive state |
| 两个有效，只有单向迁移 | asymmetric shared information |
| 两个有效，双向迁移 | strong shared-state evidence |
| 双向迁移弱，但 `Z_shared` 在 later blocks 有增量 | partial shared predictive subspace |
| 两个有效、双向不迁移、`Z_shared` 也无增量 | separable predictive states |

### 9.2 小型 shared-producer control

另实现一个交替 count/grammar update 的小型 shared producer，只运行 D3/D4 synthetic 和两位高支持患者。它是诊断：只有在 D3 能合并、D4 不错误合并时，才允许未来扩展；不进入本轮四人 replication 主模型。

## 10. Canonical evaluator 与 assay

### 10.1 唯一评分器

同一 checkpoint、anchor、dispersion、mask、weight 与 reduction 下，训练 branch 和 evaluation branch 的逐 anchor NLL 必须在预设浮点容差内一致。E1146 的方向差异必须定位到唯一数据行或计算步骤。

这是除 sealed/leakage 外唯一会阻止“可比较人体结论”的硬边界；不妨碍标为 `DIAGNOSTIC` 的代码和探索运行。

### 10.2 Oracle cascade

| Level | 已知量 | 训练量 | 目的 |
|---|---|---|---|
| 0 | 真 state | output head | 校准 evaluator/head |
| 1 | 真 event innovation + fixed bank | readout | 校准 scan/anchor alignment |
| 2 | synthetic mark channel | encoder + readout | 校准 encoder/optimizer |

Level 3 realistic full assay 作为训练代理持续使用，但不作为本轮所有探索的总 gate。

### 10.3 Synthetic DGP

- D0：`H`-only；
- D1：count-only state；
- D2：grammar-only state；
- D3：shared count+grammar state；
- D4：两个独立 states；
- D5：background-only、event marks 不可见（少量预期失败，仅展示 event-only 边界）。

效应强度用 oracle held-out deviance gain/block SNR 定义。执行节奏为每次代码变更 3 次 smoke、夜间 10 次、里程碑 20–30 次；D5 不与 D0–D4 等量运行。

人体 eligibility 所需独立 block 数由 medium-oracle power curve 决定，不预先拍脑袋固定阈值。

## 11. Persistent Training and Optimization Contract

训练充分性不是一次调参 Phase，而是所有 `S_N`、`S_G`、R1、R2 和 repaired gated model 共用的持续服务。每个模型交付统一 `training_card.json`。

### T0：数值与路径诊断

- tiny-slice overfit；
- oracle-head fit；
- state/write Jacobian；
- optimizer parameter membership；
- 每组参数在 checkpoint eligible 前均获得有效梯度和更新；
- clipping、AMP 小梯度、state-to-output modulation 审计。

gate 从 step 1 可训练，使用较小 LR 与全局 warm-up；不得冻结 gate 后在其解冻前选择 checkpoint。

### T1：学习率、优化器与 schedule

- 每个参数组 LR log-uniform 搜索 `[1e-5, 3e-3]`；
- AdamW、Adam、RMSprop；
- constant、cosine、ReduceLROnPlateau；
- warm-up 0%、5%、10%。

不同架构可选择不同最佳超参数；公平指相同搜索预算和 inner-validation 合同，不是强迫相同 LR。

### T2：初始化与归一化

- Xavier/orthogonal；
- write scale `0.01/0.1/1`；
- `alpha_init=0.01/0.03/0.1`；
- gated bias；
- input z-score vs robust scaling；
- hidden none vs LayerNorm；state 不作逐时刻 LayerNorm。

所有 scaling 只由 TRAIN 固定。

### T3：容量与时间结构

- depth `1/2/3`、width `32/64/128`；
- ReLU/GELU/SiLU、dropout `0/0.1`；
- write width `2/4/8`；
- time bank `{5,30,120}` vs `{10,60,180}` min；
- NB dispersion frozen vs low-LR；
- anchor-balanced 与 event-balanced sampling。

fixed leaky 使用完整 chronological scan；gated model 的 TBPTT 比较 30/60/120 min，并在 chunk 边界 carry+detach、不 reset。

### T4：多保真搜索

ASHA/Hyperband 的 grace period 必须晚于全部参数开始更新且至少经过一次 validation。低预算单 seed，top 配方三 seed，最终 top 2–3 配方五 seed。

### T5：失败驱动调参

连续两个 search batch 无 inner-validation 改善，或 validation 已形成稳定 plateau 时停止继续盲搜；将失败分类为梯度路径、欠拟合、过拟合、objective mismatch、support 不足或数据噪声。

### T6：锁定前复核

训练卡必须包含 learning curves、best step、seed dispersion、gradient/update、clipping、state variance/rank、random-reservoir delta、shift-null、plateau 与 output modulation。

`TRAINING-ADEQUATE` 同时要求 tiny overfit、synthetic recovery、blocked inner-validation 增量，且 checkpoint 不在 warm-up 或预算边界。

## 12. 人体 development 设计

### 12.1 两类患者，职责不同

1. **2 位 tuning patients：** 只按结果无关的 support 指标预先选定，用于训练配方和输入分诊；
2. **4 位 untouched development patients：** 配方、输入、目标、容量、shift 与 evaluator 锁定后只运行一次，用于复现。

四位 replication patients 不参与超参数、checkpoint rule、state 维度或 endpoint 选择。最终 seed 先在患者内合并，患者为汇总单位。

### 12.2 Support 与 estimability

支持排序只使用连续有效时长、power curve 所需独立 blocks、事件数、contact/prefix 支持和 seizure 数。任何 patient/horizon 可因不可估计退出相应分母，但不得因结果难看剔除。

### 12.3 时间 null

主要 null 为同患者同 session 的 block circular shift，平移量大于 target horizon，保留状态自相关和粗时间结构。matched donor 仅作敏感性，只匹配 session、粗 time-of-day、coverage 和粗 recent-rate。

## 13. H1 与 H2a 的决定性比较

### H1

在固定时间 anchor 上依次报告：

```text
H_rate → H_mark → H_mark + S_correct → H_mark + S_shifted
```

count 与 conditional grammar 分开，5/30 min 分开，first-future-event 与 block-average grammar 分开。主结果不是 reset 首次失效的位置，而是相对 `H_mark` 的 held-out proper-score 增量随物理 horizon 的变化。

### H2a

训练 `S_G` 时，contact decoder 主干固定，梯度经低容量 state adapter 回到 state producer；这使状态学习“哪些历史信息有助于预测未来 subset”，而不让 decoder 随状态一起改写。进入 H2a transfer 后，state producer 与 decoder 主干全部冻结，只在 TRAIN 上拟合预先限定的低容量 probe，并在 later blocks 分别报告 continue、positive size、subset identity 和 later continuation，以 subset identity 为主端点。

## 14. 冻结 H2b：核心只保留 seizure risk

使用 5 min 固定时间网格的单一离散 survival 合同，统一导出 5/15/30/60/120 min 风险：

```text
baseline
baseline + S_N
baseline + S_G
baseline + Z_shared
baseline + capacity-matched [S_N,S_G]
```

评价 patient-level Brier skill、log score 与 calibration；discrimination 为 secondary。组合状态使用 TRAIN-only 低秩投影或等自由度 probe，防止“维度翻倍”成为免费优势。

state、共享分解和所有超参数在读取 seizure outcome 前冻结。H2b 结果不回写间期模型。

early-ictal field/path 降为探索：只在 seizure-level 支持足够时，用低维 TRAIN-seizure PCA coefficients、laterality 或 early-field entropy，并采用 forward held-out seizures 或 leave-one-seizure-cluster-out。

## 15. 三条并行工作流与权限

### Workstream A：Evaluator、assay 与数据合同

拥有 canonical evaluator、E1146 diff、D0–D5、power curve、eligibility、split/leakage/seizure boundary。不得调人体模型 LR 或按人体结果选架构。

### Workstream B：Persistent Training and Optimization

接收明确的科学 target 与数据接口，持续执行 T0–T6，为 `S_N` 和 `S_G` 分别给出 training-adequate 配方。不得改变 endpoint、split、显式历史或 H2b label discipline。

### Workstream C：Scientific State Experiments

拥有 R0/R1、dual-view、within/cross-transfer、shared/private、H2a、frozen H2b 和 R2 exploratory。不得自行临时改 LR/optimizer；所有训练请求进入 Workstream B。

三条线并行。只有 sealed violation、真实泄漏和 canonical evaluator 对同一对象给出不同分数是全局硬停；其余失败按证据标签降级，不封锁探索。

## 16. 核心与探索性 Definition of Done

### Core Definition of Done

1. canonical evaluator 数值一致，E1146 方向差异被唯一解释；
2. Level 0–2 oracle 与 D0–D4 power curve 完成；
3. `S_N`、`S_G` 各有独立的 `TRAINING-ADEQUATE` 配方与训练卡；
4. R0/R1 在 2 位 tuning patients 完成锁定，并在 4 位 untouched development patients 单次复现；
5. within-view、cross-transfer、shared/private、H1 与 H2a 完成；
6. frozen H2b risk development 完成；
7. 白话、技术、机器报告和核心图完成；
8. sealed partition 仍未打开。

### Exploratory，不进入 Core DoD

- R2 waveform；
- D5 background-only；
- small shared producer；
- repaired gated architecture（除非 fixed leaky training-adequate 后仍显示明确容量限制）；
- 60/120 min 长 horizon；
- early-ictal field/path；
- background observer 与人体 H3。

## 17. 预期核心图与允许结论

### 核心图

1. evaluator/oracle/power calibration；
2. `H_rate→H_mark→H_mark+S` 的 count 与 conditional-grammar 增量；
3. within-view、双向 cross-transfer 与 shared/private 矩阵；
4. frozen H2b risk skill 与 calibration。

training dashboard、R0/R1 消融、shift null、R2 和 early-field 进入辅助图。

### 结论边界

- training adequate 只说明模型学会了指定任务；
- within-view 阳性只说明相应预测状态存在；
- cross-transfer/shared-subspace 说明共享预测信息，不自动说明单一生理变量；
- frozen H2b 阳性支持跨任务癫痫易感信息，不等于临床可用预测器；
- event-only 阴性只约束当前 event-derived observation，不否定连续背景中的 susceptibility state；
- H3 未运行，不允许写 IED 驱动、塑形或因果改变网络。
