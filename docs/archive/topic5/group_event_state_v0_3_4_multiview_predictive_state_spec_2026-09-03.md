# Group-Event State v0.3.4 Scientific Spec

## Multi-view Interictal Predictive State

**状态：** `V0_3_4_SCIENTIFIC_CONTRACT_REVISED_TRAIN_ACTIVE`  
**日期：** 2026-09-03  
**取代：** `group_event_state_v0_3_4_locked_evaluation_dual_view_spec_2026-09-03.md`  
**范围：** 以群体间期事件为主学习序列，连续背景 SEEG 只作辅助观测；授权实现、synthetic 与 TRAIN/tuning，development outcome 和正式检验分区保持封存。

## 2026-09-03 锁定评价复审修订

冻结 checkpoint 的完整选择期对照显示：E1146、E922 的表面增益可被选择期常数完全重现；E583、E253 的常数甚至优于时变状态；E548 有常数之外增量，但主要由 5 个 seed 中的 2 个承担。五人的 correct-time 对 block-shift wrong-time 区间均跨零。

这个结果不否定 L1。阶段常数说明记录的不同时段具有不同的整体事件分布，属于阶段级或更慢背景信息；但它是用整个评分阶段计算的非因果输入上界，不能证明每个时刻都能从既往历史估计该水平，更不能单独升级为 L3 时刻特异状态。

复审后增加以下强制合同：

1. 冻结旧 decoder 后，先在 `STATE_TRAIN` 的早段拟合无状态 `train_mean_adapter`，只在 `STATE_TRAIN` 晚段选择并冻结；所有 state gain 改为相对该重标定基线。
2. `STATE_TRAIN` 内再做 chronological fit/inner split，并留出 30 min target embargo；checkpoint 只由 TRAIN-inner 选择。`STATE_SELECTION` 使用全部合格 anchor，只报告一次，不再参与 early stopping、recipe 选择或 checkpoint 选择。
   - 两道 embargo 都必须成立：fit target 早于 TRAIN-inner 起点；TRAIN-inner target 早于 `STATE_SELECTION` 起点。只看 anchor 标签、不检查 target 结束时刻不合格。
3. 每张训练卡原生报告 `train_mean_no_state`、因果 `rolling_prefix_level`、`selection_period_mean_noncausal_oracle`、learned correct-time 与 block-circular wrong-time。常数臂既是 L1 结果，也是区分 L1/L2/L3 的必要参照，不再事后补算。
4. synthetic calibration 必须同时包含 planted dynamic truth、无动态状态零真值、常数状态与 wrong-time；报告 oracle 上界、动态恢复和零真值假阳性。它只校准仪器，不替代人体可训练性。
5. E922 原 900-step 结果位于预算边界，修正版按 2700-step 复查；E1146 首个验证点即最优，记为训练/选择不稳定，不进入状态候选。E253 的旧 recipe 在 1–4 完成后重选，不能沿用“20/20 正增益”作为选择依据。

无状态重标定自身也必须训练充分。E548 的探针显示 `1e-3` 在 900 步仍位于边界；固定 `3e-3` 后于 575 步达到最优、775 步正常早停，且优于更快的 `1e-2/3e-2`。该固定学习率不进入 state recipe 搜索，所有 state seed 共享同一确定性重标定结果；若其他患者仍在重标定预算边界，先延长这条低维基线，不让 state 代替未收敛基线补水平。

## 0. 一句话目标

仅从预测时刻之前的群体间期事件学习患者内状态，先检验它能否在真实物理时间上持续预测未来一片事件的负荷、空间传播和频带/波形表达；再冻结该状态，检验它能否预测距下一次发作的距离和发作早期空间场；最后比较共同背景驱动与 IED 反馈模型，判断 IED 是否还对后续状态具有额外的反馈式影响。

## 1. 统一科学图

```text
背景生理过程（睡眠、昼夜、药物、记录环境）
                         │
                         ▼
                  慢易感状态 S(t)
                  ├──────────────► 发作风险与发作早期路径 Y(t)
                  ▼
            群体间期事件 X(t)
                  │
                  └──── ? ───────► S(t+)       H3
```

三个核心问题分别是：

1. **H1/H2a：**过去的群体 IED 是否能压缩成一个持续预测未来事件分布的状态；这个状态是否改变同样事件开头之后的传播、停止位置和频带表达。
2. **H2b：**这个完全由间期任务学到的状态是否能迁移到发作风险和发作早期空间场。
3. **H3：**在控制先前状态、真实时间、背景与近期事件率后，IED 的数量或内容是否仍需要一条指向后续状态的反馈边。

## 2. “常数信息”与五层证据

### 2.1 常数不是无信息

把一个评分阶段内的状态换成一个常数仍可改善预测，说明该阶段存在相对旧基线的**整体慢水平差异**。这可以是记录阶段、小时级背景或更慢状态的信息，不能被写成“没有额外信息”。

但需区分三种对象：

1. `selection_period_mean`：用整个评分阶段的输入状态求均值。它不看未来标签，但对阶段早期时刻使用了阶段后部的输入，属于**非因果的阶段信息上界**，不是可上线预测器。
2. `train_mean_adapter`：只从训练期估计的常数，属于可部署静态患者校准。
3. `rolling_prefix_level`：每个时刻只用此前已观测历史估计的慢水平，属于可因果使用的候选慢状态。

因此，常数臂不再作为“状态作废”开关，而是用来回答：信息只在阶段层面存在，还是能在每个预测时刻由过去历史估计。

### 2.2 五层证据是递进含义，不是串联 AND gate

| 层级 | 要回答的问题 | 主要比较 | 允许结论 |
|---|---|---|---|
| L1 阶段/慢背景信息 | 不同记录阶段是否有不同的未来事件分布 | 固定旧基线 vs `train_mean_adapter`、`rolling_prefix_level`、`selection_period_mean` | 存在阶段级或超慢背景预测信息；必须注明是否可因果估计 |
| L2 学到的事件历史记忆 | 序列模型是否利用了显式多尺度统计之外的历史 | `B_multiscale + learned state` vs `B_multiscale`；随机、times-only、mark-shuffle 分别解释来源 | 患者级或队列级事件历史候选记忆 |
| L3 时刻特异且持续的预测状态 | 正确时刻的状态是否优于保持自相关的错时轨迹，并能预测多个物理 horizon | correct-time vs block circular shift；5/30/120 min，合格者可加 6 h | 时刻特异的 persistent predictive state |
| L4 网络表达/间期 repertoire 状态 | 状态是否预测未来参与场、传播、STOP、频带/波形，而不只是事件数 | `S_P`、`S_F` 的 within-view、cross-view、same-prefix continuation | 网络表达状态；cross-transfer 只能称共享预测信息，除非联合功能证据也支持 |
| L5 发作相关的间期状态 | 冻结的间期状态是否迁移到发作距离及发作早期空间场 | 强临床/多尺度基线 vs baseline + frozen state | development-only 的 seizure-relevant interictal state |

高层未成立不删除低层证据。L1 可以成立而 L2–L5 未成立；单个患者可作为候选证据，但不得冒充队列结论。

H3 不塞进这五层：它是建立状态后的独立反馈机制问题。

## 3. 数据单位、时间与边界

### 3.1 主序列

一个 RNN timestep 对应一次完整群体间期事件，不对应单个触点，也不对应固定一分钟窗。事件按真实发生时刻排列；状态在事件间按真实 `dt` 演化。

### 3.2 事件 token 必须保留的信息

- 真实事件间隔和有效观测 exposure；
- participation mask；
- 10 ms 内 tied groups 与连续毫秒 centroid lag；
- 每触点多频带能量、峰时和跨频带 lag；
- bipolar/CAR 后的事件内波形 embedding；
- contact 坐标、shaft 与坏道/缺口 mask；
- 连续背景 SEEG 作为辅助 observation，不取代群体事件主序列。

所有 rank 派生量必须由 `eventsBool`/participation mask 重建；禁止直接使用 legacy phantom-contaminated `lagPatRank`。

### 3.3 两类 anchor

- **固定物理时间 anchor**（默认每 5 min）：用于慢状态、future block 和发作风险，避免高 IED rate 时段因事件多而被重复加权。
- **事件 anchor**：用于下一事件及 same-prefix continuation。

### 3.4 分区与边界

- split 按 chronological recorded time，不按事件数；
- target 必须完全落在本 split 内，split 间留出最大 horizon 的 embargo；
- future block、H3 exposure 和训练 chunk 不跨 session、真实缺口或 seizure；
- 发作后状态 reset，immediate postictal 期排除或独立报告；
- 缺失/检测无效时间不提供“没有 IED”的证据。

## 4. 候选状态视图与执行优先级

不再预设所有端点共享一个状态，但也不同时铺开三个同等规模的训练工程。**v0.3.4 的人体 Core 只以 `S_P` 为主状态**；`S_N` 保留为辅助负荷视图；`S_F` 在 `S_P` 证明目标与优化器至少能够学动之前只做数据接口和小型探索。

### 4.1 `S_N`：负荷/到达状态

预测未来事件数与 silence，保留为必要但非主故事的辅助视图。主 likelihood 为带 exposure mask 的离散 hazard 或 negative-binomial count likelihood。

### 4.2 `S_P`：空间传播状态（v0.3.4 Core）

预测未来事件的 participation、continue/STOP、positive extent、tied-group subset、传播 delay/direction 和 same-prefix continuation。

### 4.3 `S_F`：频带/波形表达状态（v0.3.4 Exploratory）

预测未来事件的 per-contact multiband energy、peak time、cross-band lag 和低维 waveform embedding。

`S_P` 必须先在与人体 scoring 同构的合成空间传播任务上恢复 planted signal，并通过人体 tiny-overfit/梯度更新检查，之后才进入人体主训练。合成只证明实现与目标可学习，不证明人体存在该信号。`S_F` 可并行完成 event token、target 和 frozen probe 接口，但在 `S_P` 人体训练学动前不启动大规模搜索。

只有在单视图先各自学到可复现信息后，才训练低维 shared/private readout；不因共享失败而否定单视图状态。

## 5. 模型与训练目标

### 5.1 状态骨干

第一版用低容量多尺度 leaky memory，固定 time-scale bank：

```text
5 min / 30 min / 120 min / 6 h
```

每个尺度若干通道，事件到来时由 event encoder 更新；事件间按真实 `dt` 衰减。自由 `tau` 只作 sensitivity，不能把学到的 `tau` 直接解释为生理时间常数。

### 5.2 真正的 multi-horizon producer

保留局部 next-event producer `P_local`，另训练面向未来时间块的 `P_slow`：

```text
L = L_local + λ5 L_5min + λ30 L_30min + λ120 L_120min
```

v0.3.4 Core horizon 为 **5 min 与 30 min**。120 min 与 6 h 只在真实 coverage/独立块可估时作 exploratory，不进入 Core Definition of Done。慢状态的定义来自跨物理时间的功能预测，不来自某个被命名为 slow 的 hidden block。

future-block loss 分开计算：

1. event count / silence；
2. conditional mark distribution（给定 block 内确有事件）；
3. participation/extent/STOP；
4. propagation lag/direction；
5. multiband/waveform expression。

不得把 count 增加造成的 occupancy 变化误算成 repertoire 改变。

### 5.3 多事件预测

除固定时间 future block 外，事件锚点同时预测未来 1/5/20 个事件的负荷、空间和频带表达。它用于判断候选状态影响的是单次局部连续性，还是附近一片事件的共同分布。

## 6. 强基线与解释性对照

### 6.1 `B_multiscale`

统一因果基线至少包含：

- event rate：1/5/30/120 min、3/6/12 h EWMA；
- time since last event；
- extent/STOP、participation/repertoire、multiband summary；
- time of day、session position、coverage；
- H2b 时另加 time since previous seizure、postictal/cluster；睡眠与药物信息只在真实可得时加入。

使用正则化 GLM 或与 residual head 容量匹配的低容量 MLP。形式、超参数与校准全部只在 TRAIN/rolling inner-validation 选择。development 上的 calibration 只报告，不再反向选模型。

### 6.2 必报但不作全局 gate 的对照

- `train_mean_adapter`：静态患者校准；
- `rolling_prefix_level`：可因果慢水平；
- `selection_period_mean`：非因果阶段信息上界；
- random reservoir：额外动态容量；
- times-only：只保留事件时刻；
- mark-shuffle：保留时刻、破坏事件内容；
- block circular shift：保留状态自相关、打破与未来的对齐。

这些对照分别定位信息来源。某一个未胜出只降低相应解释，不得自动关闭 H2b 或 H3。

## 7. 预训练 contact sequence decoder 的接口

### 7.1 冻结主干

复用既有每患者 contact sequence decoder，但先完成嵌套时间泄漏审计：normalization、contact vocabulary/order、patient offset、calibration、tied-group statistics、detector template 与 checkpoint selection 均不得使用最终评价时间。

零状态 adapter 必须逐事件复现旧 decoder 输出与评分。旧模型若使用旧 scoring，pilot 保持 parity；正式切换 exact subset likelihood 必须重新预训练 decoder，不能静默替换 objective。

### 7.2 状态如何调制事件形态

冻结 decoder 权重，在每个 tied-group recurrence step 注入低秩状态 adapter：

```text
h_k' = FiLM_or_additive_adapter(h_k, S(t_e^-))
```

状态分别调制：

1. continue vs STOP；
2. 继续时 positive group size；
3. 给定 size 的 contact subset/order；
4. later energy/waveform readout。

decoder 虽冻结，梯度仍可穿过它更新 state producer。状态网络负责记录跨事件上下文，decoder 负责解释一次事件如何展开。

## 8. H1/H2a 核心实验

### 8.1 H1：未来事件块

在同一固定时间 anchors 比较：

```text
B_multiscale
B_multiscale + P_local state
B_multiscale + P_slow state
correct-time P_slow
block-shifted P_slow
```

主图 Core 横轴为 5/30 min，纵轴为相对 `B_multiscale` 的 proper-score 改善；count 与 conditional content 分开画。120 min/6 h 使用不同视觉标记进入 exploratory panel，且必须同时画独立块数。

### 8.2 H2a：相同开头，后面是否分岔

匹配相同首发触点/前两个 tied groups、相近前 50–100 ms 波形和早期能量范围，比较：

```text
p(later recruitment | prefix, B_multiscale)
p(later recruitment | prefix, B_multiscale, frozen state)
```

分别报告 continue/STOP、positive extent、later subset、传播 lag、multiband expression。下一事件和未来 1/5/20 个事件均评估。

## 9. H2b：冻结间期状态的跨任务迁移

状态 producer 在看到任何 seizure label 之前冻结；seizure head 单独训练，发作梯度不得回流到状态。

两个并列主任务：

1. **发作风险/距离：**每 5 min 输出离散 survival hazard，得到 5/15/30/60/120 min 及更长区间的风险；评价 patient-level Brier skill、log score 和 calibration。
2. **发作早期空间场：**在 5 min、30 min、2 h、6 h lead time，用冻结状态预测发作最初 5–10 s 的 per-contact energy/recruitment field、first shaft/ROI、laterality 与 early propagation axis。

比较患者平均发作模式、最近 IED、多尺度历史和 frozen state；按 seizure pattern 分层，推断单位为患者/发作 episode，不是 grid row。

## 10. H3：独立的反馈模型比较

明确区分 observer 看见事件后的 belief update 与 IED 对生成状态的反馈。8 月 26 日长尺度复审与本轮常数偏移属于同一类失效：带反馈臂可能通过免费截距或阶段慢水平获益，而非利用 exposure。任何人体比较前必须先交付按真实 coverage segment 计算的完整非重叠 exposure/future blocks、有效独立窗口数和 exposure overlap。

所有反馈臂必须共享同一个已拟合截距与 `rolling_prefix_level`，或者给予 M0 完全等容量的截距/慢水平项；`selection_period_mean` 只能作为 `noncausal_input_oracle` 诊断。比较：

1. `M0_common_drive`：IED 只是状态读出，不反馈；
2. `M1_burden_feedback`：事件数/负荷对后续状态有 signed edge；
3. `M2_mark_feedback`：参与、extent、传播、频带/波形内容对后续状态有 signed edge。

在未见 future block 上比较 proper score，并报告 event-type-specific signed functional impulse response。事件尺度候选为 100/1000/10000，但人体只运行预先通过可估性审计的 `(subject, scale)` cell；只有 1–2 个有效独立窗口的旧长尺度结果一律标 `not_estimable`，不得进入方向计数。

人体最高允许措辞是 `event-feedback-like predictive dependence`，不直接写因果塑形。

## 11. 训练、模型选择与评价

### 11.1 session-preserving 训练

- 只 batch 不同 recorded sessions；
- session 内按物理时间上限与最大事件数双限做 TBPTT；
- chunk 前 burn-in 只重建状态、不计 loss；
- chunk 边界 carry state 后 detach，不 reset；
- 每个 epoch 从 session boundary 用当前参数重新 replay。

### 11.2 超参数搜索

- 模块级 LR、optimizer、schedule、width/depth、residual、normalization、state/write scale 均真实搜索；
- 预算阶梯 `300 → 900 → 2700`，曲线仍改善才到 `8100`；
- top 3 recipe × 5 seeds；患者级结果不能用单 seed；
- synthetic 只验证实现与可恢复性，不能替代人体可训练性；人体 tiny-overfit、梯度、更新量和 blocked generalization 必须单独报告。

### 11.3 患者角色按端点登记

- tuning：E253、E916；
- 不再设一张跨端点通用的 development sentinel 名单。现有可估性表显示：E1146/E583/E548/E922 的 30 min conditional grammar 独立块分别为 22/23/36/30，均达到现有 20 块要求；但 30 min count 分别只有 22/23/36/30，均低于 47 块要求；H2b development 发作数分别为 0/0/9/5。因此 `S_P` 可用这四位做锁定 evaluation，H2b 只可用 E548/E922，E1146/E583 不进入 H2b 分母；
- H2a 事件支持为 E1146 2709、E583 93、E548 7706、E922 8268，但尚无 power curve，故 E583 先标低支持、不得与其余三位等权解释；
- `S_N` count 若需要 development，只能在新模型锁定后从 count 可估集合另行登记，不能把上述四位当 30 min count 哨兵；
- multiband/waveform 与 early-field 尚无完整可估性字段，完成 power/support 表前标 `not_yet_measurable`，不按患者名猜；
- E1073、E1077、E818、E958 保持 untouched replication；
- 旧 9 人的 v0.3.3 selection 结果只作历史证据，不能先在旧失准 count 模型上消费 development 再用于新版选择。

## 12. 结果与结论合同

每位患者、每个 state view、每个 horizon 必报：

- baseline、train-mean、rolling-prefix、period-oracle、learned、random、times-only、mark-shuffle、shift；
- proper score 与 calibration；
- seed 分布、独立时间块数、覆盖时长；
- 是否属于 L1–L5 中哪一层及理由。

队列结论以患者为单位；患者级候选单独保留，不因队列未显著被抹掉。`not estimable`、`not learned`、`not supported` 分开。

## 13. Definition of Done

### Core

1. 新事件 token、固定时间 anchor、split/embargo/boundary 合同落盘；
2. `B_multiscale`、三类常数/慢水平对照和 state 对照可同表评分；
3. `S_P` 通过同构 synthetic recovery，再在 tuning 人体证明优化器确实更新，并在四位 grammar 可估哨兵锁定评价；`S_N` 为辅助视图，`S_F` 为 exploratory；
4. 5/30 min future-block 与 next 1/5/20-event 结果；120 min/6 h 为 exploratory；
5. frozen decoder 的 same-prefix H2a；
6. frozen-state H2b 风险与 early ictal field/path；
7. H3 先完成截距匹配、滚动慢水平和真实独立窗口审计；仅在可估 cell 上完成 M0/M1/M2 最小人体比较；
8. 白话、技术、机器报告与三张核心图；sealed 未打开。

### 不作为继续与否的 gate

- 单个 endpoint 阴性；
- 某个对照臂胜出；
- synthetic 未在第一版恢复；
- `S_N` 只解释 count；
- H1 未达队列显著。

只有真实泄漏、错误时间边界、同一对象 evaluator 不一致或不可恢复的产物污染才全局硬停。

## 14. 当前五层证据快照（v0.3.3 + v0.3.4 首批修正版）

| 层级 | 当前力度 | 已有数据 | 解释边界 |
|---|---|---|---|
| L1 阶段/慢背景信息 | **中等偏强** | 原基线增量 5/9；5 位可由 selection-period 常数重现；H_mark 在部分患者后段出现 1.3–3.3 倍双向水平失准 | 证明未来负荷分布随记录阶段变化；阶段常数是信息，但尚未证明可由当下之前历史稳定估计 |
| L2 学到的事件历史记忆 | **很弱，未建立** | 修正版 E548 的旧常数外阳性未复现；E922 的常数外增量很小且因果 rolling level 为负；E253 20-cell 初筛的中位常数外增量为负 | 仍可继续找患者异质候选，但当前没有可冻结的 learned-history 证据 |
| L3 时刻特异 persistent state | **未建立** | 旧五位 correct-time 对 wrong-time 区间均跨零；修正版 E548/E922 错时代价接近零；E253 对 shift 敏感但 learned state 不胜阶段常数 | 不能把阶段水平或错位敏感单独称为时刻特异状态 |
| L4 网络表达状态 | **未有效检验** | 旧 event-history 模型有 H2a development 阳性；v0.3.3 人体 `S_G` 6/6 cell 第 1 步即选中，实际未学习 | 旧证据保留，但新版多视图状态尚未复现 |
| L5 发作相关状态 | **未建立** | 两位 tuning、12 次 development 发作：5–60 min 无稳定增量，120 min 仅极小探索信号 | 尚未用训练充分的 `S_P/S_F` 做 frozen risk + early-field 迁移 |
