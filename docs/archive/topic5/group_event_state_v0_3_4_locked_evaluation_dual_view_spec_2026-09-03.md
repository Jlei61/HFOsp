# Group-Event State v0.3.4 Scientific Spec（草案）

## Locked-Evaluation Dual-View Predictive State

**状态：** `SUPERSEDED_BY_MULTIVIEW_PREDICTIVE_STATE_SPEC`
**日期：** 2026-09-03
**取代版本：** [`group_event_state_v0_3_4_multiview_predictive_state_spec_2026-09-03.md`](group_event_state_v0_3_4_multiview_predictive_state_spec_2026-09-03.md)。本文件保留用于追溯，不再作为执行依据。
**起草：** 复审者，基于用户 2026-09-03 的 v0.3.4 建议稿 + 当日 v0.3.3 可训练性收口复审（`group_event_state_v0_3_3_trainability_closeout_{plain,technical}_2026-09-03.md` 顶部修订块）。
**用途：** 等待科学审阅；不得据此打开 sealed partition、消费 development evaluation 或启动人体训练。
**继承：** v0.3.3 的群体事件数据底座、时间分区、canonical evaluator（`85437af9…`）、H_mark artifact 合同、Training Laboratory（T0–T6）、S_G synthetic recovery 结论、patient role lock。
**不继承：** v0.3.3 报告的"5/9 可靠增量 / 三位强候选"判读；"S_G 下一步直接人体训练"；单一 H_mark 作为唯一基线；同段 block-shift 作为唯一时刻对照；"state 成立/不成立"二元结论。

## 0. 一句话目标

锁定并在未触碰的 development 时段上独立评价当前 9 位患者的 `S_N`；把基线改成随时间自校正的阶梯；直接训练未来传播语法状态 `S_G`；再通过 within-view、多重时刻对照、count↔grammar cross-transfer、事件锚点 H2a 和 frozen H2b，判断"事件负荷状态、传播状态、发作易感信息"三者是共享、部分共享还是彼此分离。

## 1. 复审给本版定下的两条硬约束

这两条来自 2026-09-03 下午对 v0.3.3 收口的复审，是本版所有比较的前提：

1. **基线必须随时间自校正。** v0.3.3 的 9 位患者里，H_mark 在 STATE_SELECTION 把三个未来窗的事件数低估 1.3–3.3 倍（E1096 约 2 倍、E548 1.8–3.3 倍、E1146 1.3–1.45 倍），训练期接近 1。任何在这种基线上得到的"状态增量"首先是水平失准被补上。本版要求每个基线层在 TRAIN、STATE_SELECTION、DEVELOPMENT 三段都报告"实际事件数 ÷ 预测事件数"，偏离 1 超过预定容差的基线不得作为增量分母。
2. **常数偏移臂进入每一级评分。** 把状态换成"评分期一个常数向量"（只用输入）就能重现 v0.3.3 五位患者的可靠增量；同段 block-circular 错时置换保留期内均值、结构上检测不到常数。本版把 `period_offset_control` 作为与 shift null、random reservoir 并列的必报对照，并要求"常数之外的增量 > 0"才可进入 E1 以上任何一级。

常数偏移与"比模型时间常数更慢的变量"兼容，**不是"没有状态"的证明**；本版把"慢的事件率水平"划入基线阶梯（H0 的长窗口与因果自校正），把"依赖事件内容的慢状态"作为独立可检验命题（§6.3 慢库对照）。

## 2. 科学对象：五级证据层次（取代二元判决）

所有人体结果按下列五级逐级报告；每级是前一级的加严，不互相替代；任何一级未过不抹掉已过的级别。

| 级 | 名称 | 定义（同 anchor、同 evaluator） | 允许命名 |
|---|---|---|---|
| E1 | residual predictive feature | `H_top + S_correct > H_top`，且 `period_mean − learned > 0`（常数之外的增量区间在零之上）；`H_top` 是 §3 阶梯中最高的合格基线层 | "存在额外预测信息" |
| E2 | learned event-history memory | 在 E1 之上，learned producer 优于同容量 **random reservoir**、**times-only**（事件时间保留、marks 置零）、**mark-shuffle**（同段 30 min 块内打乱 marks）、**linear marked EMA**（无编码器的线性 marked 泄漏积分） | "事件历史记忆" |
| E3 | time-specific predictive state | 在 E2 之上，正确时刻状态位于 **多重 shift null**（≥32 个同段偏移量）分布的有利尾部（预注册分位数），并再次通过常数偏移臂 | "时刻特异的预测状态" |
| E4 | shared interictal state | `S_N`、`S_G` 冻结后至少单向稳定 cross-transfer；双向为最强；允许报告"部分共享"或"可分离" | "shared / partially shared / separable interictal predictive state" |
| E5 | seizure-relevant interictal state | 冻结间期状态在强 baseline `B(t)` 之外改善 seizure hazard 或低维 early-ictal 表达 | "与发作相关的间期状态（development-only）" |

禁止：在 E1 未过时使用"state"以外的任何生理名词；在 E4 未过时使用"shared state"；在 E5 未过时使用"易感状态"。

## 3. 基线阶梯（取代单一 H_mark）

```text
H0  H_rate       last IEI；事件率 EWMA（5/30/120 min）+ 长窗口（3/6/12/24 h）；clock/session；coverage；
                 因果自校正项 log(过去 W 小时已完成窗口的 实际事件数 / 预测事件数)，W ∈ {3, 6, 12} h（TRAIN 上选，shrinkage 拟合）
H1  H_mark       H0 + extent/STOP EMA + contact/repertoire occupancy + multiband EMA
H2  H_nonlinear  与 state residual head 同容量的 MLP，只输入 H_mark 特征，不输入事件序列
H3  H_mark + S   神经状态残差
```

主报告永远是四段阶梯 `H_rate → H_mark → H_nonlinear → H_mark + S`，每段的增量单独给区间。它回答四个问题：marks 是否比事件率更有用；显式 marked history 是否已足够；state 的优势是否只是非线性变换；事件序列记忆是否真的额外。

**自校正验收**：每个基线层在 TRAIN / STATE_SELECTION / DEVELOPMENT 的 count/μ 比值在 `[0.8, 1.25]` 内（三个未来窗分别），否则该层标 `MISCALIBRATED` 并不得作分母；v0.3.3 的 H_mark 在 9 位里至少 4 位不满足。

自校正项只允许用过去已观测事件数（因果），v0.3.3 复审用的 3 h 硬比值只是探针：它在 E1096/E1146 上拿回同量级改善，但在 E548/E1125 上过冲（基线反而变差 1.0–1.5 nats），所以本版要求用 shrinkage 拟合而不是硬比值。

## 4. 输入分支

- **R0 summary token**：v0.3.3 已用；解释性低容量基线。
- **R1 contact-resolved token**：逐触点 participation、精确 delay、逐频带能量/峰时、shaft/coordinates、validity mask，经低容量 contact encoder 聚合。**S_G 的必要输入**；S_N 也在两位 tuning 患者上做 R0 vs R1 消融。
- **R2 within-event waveform token**：探索分支；须先通过 masked-contact / early-to-late / cross-band / held-out temporal block 四类 learning audit，重建输入里已有的标签不算通过。

## 5. Count-view `S_N`

### 5.1 目标不变

`[N_0–5, N_5–15, N_15–30 min]` 三段 NB 残差，canonical evaluator，无事件进入目标；dispersion 冻结在 TRAIN。

### 5.2 Track A：锁定评价当前 9 位患者（v0.3.3 产物）

1. 冻结 9 位患者现有 recipe/checkpoint（card sha 已登记）。
2. 在 **untouched DEVELOPMENT_EVALUATION** 上一次性重算：`H_rate/H_mark/H_nonlinear` 阶梯；`H − learned`；常数偏移臂；≥32 个 shift；10–20 个 random reservoir；times-only；linear marked EMA；mark-shuffle。
3. seed × block nested bootstrap。
4. 输出 **HPO optimism gap**：STATE_SELECTION 读数 − DEVELOPMENT 读数，逐患者。已有的两条先例（E916 pilot +0.049 → −0.017；E253 pilot +0.010 → +0.488 且随机臂≈H）先写进报告作为参照。
5. development 读取写入 ledger（§10.3）；读取后这 9 个 `S_N` 版本不得再调参。
6. sealed 保持关闭。

Track A 不预期阳性：按复审，5 位的选择期增量已由常数重现，A 的价值是把 optimism gap 和基线失准量化成可引用的数字，并给 E1 一个诚实的分母。

### 5.3 S_N 的后续训练

只在 §3 自校正基线上重训；沿用 Training Laboratory 合同（§11）；两位 tuning 患者做 R0/R1 消融与 recipe portfolio；不再围绕 count 无限调参——连续两轮搜索无改进且曲线 plateau 即标 `optimization_exhausted` 并转向 S_G。

## 6. Grammar-view `S_G`

### 6.1 目标

主目标 `contact subset identity | positive K, early prefix`；分项报告 continue/STOP、positive size、later continuation；multiband expression 只作冻结 probe。第一轮两臂：`G-primary`（只 subset）与 `G-composite`（subset 为主，continue/size 小权重）。评分：block 起点冻结状态、整个 future block 复用同一状态、`1[N_future>0] × mean_event(subset loss)`，并同时报告 first-future-event 与 block-average grammar。

### 6.2 训练前置门（来自复审）

- S_G **必须自做** optimizer / LR-per-module / budget 搜索。v0.3.3 人体 O2 S1 六个 cell 全部在第 1 步被选中，原因是移植的 O1 配方把 pilot 学习率再乘 0.1（encoder 1.7e-6、adapter 4.2e-6），600 步等于没训；按选择期 NLL 在学习率档位里挑会选到"改动最少"的档。这类"最小改动即最优"的选择结果在本版视为 `NO_LEARNING`，不得冻结。
- 人体 S_G 启动前，必须先在合成 D3（shared count+grammar）上通过 Level 2 恢复（`ci_low>0` 且 D0 假阳性≈名义），并在 D4（两个独立状态）上不错误合并。v0.3.3 的 `encoder_objective_mismatch_under_frozen_scaffold_nuisance` 分类仍然有效，本版先修 encoder/objective（R1 输入、subset 目标的 conditional-Bernoulli 打分方式、按评分方式训练），再谈人体。

### 6.3 慢库对照（回答"会不会是更慢的状态"）

对 S_N 与 S_G 各做一组 6/12/24 h 时间常数的慢库臂，在自校正基线上比较 learned 与 random 慢库：learned 稳定胜出 → "依赖事件内容的慢分量"；打平 → 慢分量只是事件率水平，归基线。v0.3.3 复审已在 H_mark 与 3 h 硬比值基线上各跑一遍作为第一版（技术版 R3）。

## 7. H2a：事件锚点的当前事件传播

对事件 `e`，在已知 early prefix 后：

```text
p(next tied group | early prefix, H(t_e^-), S(t_e^-), G_p(t_e))
```

比较 `H`、`H+S_N`、`H+S_G`、`H+[S_N,S_G]`（容量匹配低秩投影）、block-shifted state、random state；输出 continue vs STOP、positive size、contact identity | K、later continuation；primary 为 `contact identity | K, prefix`。state producer 与 contact decoder 主干冻结，只拟合预先限定的低容量 probe。这是 event-anchor 问题，与 future count 无关；v0.3.3 已消费的 H2a development 评分（E253/E916：S_N 之外无可复现增量）作为参照写入。

## 8. Cross-transfer 与 shared/private

冻结 `S_N`、`S_G` 后在 later development blocks：within-view（`S_N→count`、`S_G→grammar`）、cross-transfer（`S_N→grammar`、`S_G→count`）、TRAIN-only regularized CCA / reduced-rank 分解 `S_N = A_N Z + U_N`、`S_G = A_G Z + U_G`，`Z/U_N/U_G` 分别评 count、grammar、H2a、H2b；组合状态先做 TRAIN-only 低秩投影匹配探针自由度。命名按 §2 E4 表。

## 9. H2b：冻结间期状态的发作风险（development-only）

只要 state 由纯间期目标训练、checkpoint 已锁定、未读 seizure label 调参，即可运行。5 min 固定网格离散 survival hazard，导出 5/15/30/60/120 min：

```text
B(t),  B(t)+S_N(t),  B(t)+S_G(t),  B(t)+[S_N,S_G]
```

`B(t)` 至少含 clock/session、sleep/background proxy、recent IED rate（含长窗口）、extent、time since last seizure、postictal/cluster、medication/stimulation（可得时）、coverage。评价 patient-level Brier skill、log score、calibration；discrimination 为 secondary；按 seizure pattern 分层。early-ictal 只用低维预定义 target（first recruited shaft/ROI、onset laterality、early 5–10 s contact-energy centroid、TRAIN-seizure PCA field coefficients），forward held-out seizures 或 leave-one-seizure-cluster-out。v0.3.3 的 H2b S_N-only（2 位 tuning、12 次 development 发作、5–60 min 无稳定增益、120 min 小的探索性信号）作为参照。event-only 阴性不否定 background physiological state。

## 10. 评价纪律

### 10.1 唯一 evaluator

同一 checkpoint/anchor/dispersion/mask/weight/reduction 下，训练侧与评价侧逐 anchor NLL 在浮点容差内一致；每张卡登记 evaluator hash（v0.3.3 的 `selection_metric_is_canonical=false` 状态在本版必须闭合）。

### 10.2 选择与评价分离

- 配方选择与 checkpoint 步数选择使用 **rolling inner-validation**（TRAIN 尾部滚动块），STATE_SELECTION 只用于最终候选之间的比较与 optimism 记录，不再同时承担配方、步数和报告区间三个角色。
- 每张卡报告 rung-0 "随机配置中已胜过基线的比例"；该比例接近 1 时（v0.3.3 E922 100%、E1146 77%）读作"任何非零修正都在补基线"，不是候选。

### 10.3 development ledger

每次读取 DEVELOPMENT_EVALUATION 记录 (subject, endpoint, state version, checkpoint sha, 时间)；同一 (subject, endpoint) 上，选择过程晚于该记录的任何模型版本标 `development_exposed`，其 development 读数只能作诊断。

### 10.4 不确定性

seed × block nested bootstrap；patient 为推断单位；≥32 shift 的 null 分布分位数；10–20 random reservoirs 的分布。

## 11. Training Laboratory 合同更新

- 每个模块（encoder / write / head / state）独立 LR ∈ [1e-5, 3e-3]（log-uniform）；AdamW/Adam/RMSprop；constant/cosine/plateau；warm-up 0/5/10%。
- 预算阶梯 `300 → 900 → 2700 → 8100` 步；top recipe 最佳点在预算末端就自动延长，不得据此宣布训练不足。
- 初始化：Xavier/orthogonal、write scale、residual head scale、非零 gate 初始化；归一化：z-score vs robust、hidden LayerNorm vs none、state 只允许 TRAIN-only 固定缩放。
- 容量：width/depth/state dim/write dim/tau bank/dropout/weight decay；采样：anchor / event-rate-balanced、count-extreme clipping/weighting、per-bin loss scale、NB dispersion 处理。
- training adequacy 重构为四段：debug overfit（tiny slice，按配方自身预算）、convergence（plateau 判据）、synthetic recovery（D1/D2/D3 而非 v0.3.2 proxy）、blocked generalization（rolling inner-validation）。
- 训练卡必报：curves、best-step 位置、逐模块 gradient/update、clipping、state/write scale、有效监督秩、seed variance、random-reservoir delta、**period_offset_control**、shift null、synthetic recovery、推荐的下一批搜索。
- 只有连续两轮搜索无改进且曲线 plateau 才标 `optimization_exhausted`；v0.3.3 的 7 位 broad 只跑了 1 个批次即停，本版不允许。
- 训练 agent 只能读 TRAIN、rolling inner-validation、synthetic、指定 tuning 患者；读到 DEVELOPMENT 后不得对同一版本回调。

## 12. 患者职责与分母

- tuning：E253、E916（已用）；
- 已触碰的 S_N 9 人：只进 Track A 锁定评价与自校正基线重训；
- untouched replication：E1073、E1077、E818、E958（patient role lock），本版扩到 6 位须只按 Agent A `eligibility_by_endpoint_horizon.json` 的 development 独立块数选（30 min count_profile 需 47 块，27 人中 8 人可估；30 min conditional grammar 需 20 块，12 人可估），不看任何结果。
- 任何患者/horizon 可因不可估退出分母，不得因结果剔除。

## 13. Definition of Done

### Core

1. §3 基线阶梯实现并通过自校正验收（三段 count/μ 在容差内）；
2. Track A：9 位 `S_N` 锁定评价 + optimism gap 报告；
3. `S_G`：合成 D3 Level-2 通过 → 自有 recipe 搜索 → 两位 tuning 患者 R1 训练卡；
4. within-view、multi-shift、E1–E3 三级对 `S_N`、`S_G` 分别判定；
5. cross-transfer matrix + shared/private；
6. current-event H2a；
7. frozen H2b hazard（development-only）；
8. 白话/技术/机器报告与核心图；sealed 未打开。

### Exploratory（不进 Core）

R2 waveform；early-ictal field/path；gated sensitivity；60/120 min horizon；小型 shared producer；人体 H3。

## 14. 允许 / 禁止表述

- 只有 E1 过 → "存在超出基线阶梯的预测信息"；不得写 state。
- E1–E3 过而 E4 未测 → "时刻特异的 count（或 grammar）预测状态"；不得写 shared。
- E4 单向 → "asymmetric shared information"；双向 → "strong shared-state evidence"。
- E5 development 阳性 → "development-only"，不得写 cohort confirmation。
- 常数偏移解释的增量 → "基线水平失准被补上；与更慢的率水平变量兼容"，不得写"无状态"，也不得写"状态"。

## 15. 留给审阅的分歧点

1. Track A 会消费 9 位患者 count endpoint 的 development 读取，之后这 9 人在自校正基线上重训的新 `S_N` 只能标 `development_exposed`；是否接受，或改为只对其中 4 位"原稿强候选"做 Track A、保留其余 5 位作为新基线下的 clean 评价？
2. 预算阶梯延长到 8100 步会放大 STATE_SELECTION 上的选择乐观；本版以 rolling inner-validation 隔离，是否足够？
3. 6 位 replication cohort 与 Agent A 的 8/27（count）/12/27（grammar）可估集合交集可能不足 6 位；不足时是否降为 4 位并写明？
4. 自校正项的形式（长窗口 EWMA 进 ridge vs 因果比值 shrinkage）由 Agent C 定；本版只规定验收（count/μ 在容差内）。
