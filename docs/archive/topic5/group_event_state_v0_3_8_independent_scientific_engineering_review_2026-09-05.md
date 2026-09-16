# 审阅结论：Group-Event State v0.3.8

日期：2026-09-05。性质：独立科学性与工程性审阅，基于现存合同、代码、checkpoint、逐 seed 卡片及原始 PNG/PDF。没有重训人体模型，没有修改原报告、原汇总或原结果。

## 1. 一句话判断

**已经做出了能测量部分历史预测信息的系统，但尚未闭合“从间期事件学到一个随时间演化、跨多种病理表达且关联发作的统一状态”。原报告的“E922 可重复发作候选”存在汇总错误，必须降级；“四个问题都获得支持”不能作为当前验收结论。**

这里按用户的原始科学目的审阅：从间期群体事件学习状态，证明其超越简单历史统计、具有时间特异性和多端点预测价值，再检验发作关联；事件是否反过来改变生理状态是另一个反馈假设。本文中的“病理状态”不能由 latent 向量、固定时间常数或梯度非零直接认定。

## 2. 完成程度

> **完成度：约 50/100。** 这是对原始科学目标的审阅评分，不是任务完成率或统计概率。

已完成的部分有实际证据：

- 两层 H1 共 165 张卡，H2a 共 150 张卡，H2b outcome 共 55 张卡；这 370 张主要结果卡都声明没有读取 development/sealed。读取路径确实截在登记的 `80pct` 之前。另有 30 张训练后 credit 审计、55 张随机背景审计、55 张 H2b freeze 卡和 30 个严格 decoder 单元；这些不是独立科学样本。
- 主监督器和登记子队列均为 `COMPLETE`，所查队列零待运行、零失败。H2a 的 30 张 `NOT_ESTIMABLE` 来自中尺度事件段不足，不能计作阴性。
- v0.3.7 的修复搜索真实存在：48 个 baseline 配方单元、264 个 observer 配方单元。优化器参数、初始化、宽度和预算确实被搜索过，不能说 v0.3.8 完全没有训练审计。
- H1 卡有首步梯度、训练期参数移动、FIT/INNER 曲线、选中步数、预算耗尽及耐心停止记录；E1125 的已选 event-only observer 权重实际偏离初始化。
- H2b 的全局 freeze marker 早于两队列首批 outcome 卡；源码先完成两层特征冻结再释放 outcome。这个顺序有实现与文件时间证据，不只是报告口头声明；文件时间不是不可篡改的全生命周期访问日志。
- 本次 CPU 复跑原有 91 项相关测试，全部通过；额外用真实卡片和小型确定性探针复现了下面的汇总与评分问题。

主要扣分：候选判定拼接不同 seed；训练与下游证据未绑定同一个已学习 checkpoint；same-prefix 目标不符；非线性演化未被单独检验；训练充分性、统计不确定性和 H2b 正对照尚不足以赋予阴性生物学解释。

## 3. P0 / P1 关键问题

### P0：E922 的“3/5 seeds 完整复现”不成立

**问题。** 汇总器先分别计算每个对照是否有至少三个 seed 为正，再对这些汇总布尔值取 AND。它没有先检查同一个 seed 是否同时胜过基线、常数和错时。

真实 E922 event-only 卡的三个增益如下，正号表示学习状态较好：

| seed | 超过 B_history | 超过 FIT 常数 | 正确时刻超过错时 | 三项同时通过 |
|---|---:|---:|---:|---|
| 20260903 | +0.001605 | +0.005556 | −0.004786 | 否 |
| 20260904 | +0.002046 | +0.002607 | +0.047623 | 是 |
| 20260905 | −0.048148 | −0.037233 | +0.092657 | 否 |
| 20260906 | +0.050216 | +0.054901 | −0.002017 | 否 |
| 20260907 | −0.004514 | −0.003149 | +0.014467 | 否 |

三个边际条件各自是 3/5，但交集只有 **1/5**；grid 的交集同样只有 1/5。因此可以说“几个边际增益的中位数为正”，不能说“同一检验链在 3/5 seeds 中复现”。同类问题还使 E958 的 0.5 h dual H1 被列为候选，实际同时满足原对照组合的是 2/5。

**为什么严重。** 它直接改变唯一发作候选与“四个问题都有支持”的结论。`scientific_closure.interpretation` 还把“All four questions … support”写成固定字符串，没有随判定结果生成。

**怎么改。** 以 `(patient, family, seed, checkpoint, horizon/endpoint)` 为键，先对同一卡片上的所有条件取交集，再统计 seed 数。随后才聚合患者。保留各边际结果作为诊断，重新生成报告和圈标。原有方向性 seed 门槛本身也不是显著性检验。

定位：`scripts/finalize_group_event_state_v038.py:216`、`:391`、`:725`；可重放证据见 `h2b_candidate_joint_gate_audit.json` 与 `reproduced_findings.json`。

### P1：支持“已学习状态”的证据链，混用了不同模型和未更新的 observer

**问题与影响。**

1. 长程 credit 脚本在第 49 行硬编码读取 `h1_root/event/...`，且只审计最后一个 **8 h future head**。E1125 的完整 **6 h H1 候选来自 dual**。同一个患者、同一套数据，并不代表同一个模型。现有结果尚未把 dual 的六小时预测增益与该 dual checkpoint 的旧事件 credit 连上。
2. E253 的 event-only 与 grid H1 在 **5/5 seeds 都选择第 0 步**。抽查 event seed20260903 的两个 observer 输入矩阵，保存值与按 seed 重建的初始化逐元素相同；H2a 随后仍把这个冻结 observer 作为状态输入。其 contact 增益说明下游 adapter 能利用固定过滤后的历史特征，不能证明上游 observer 已从人体预测任务学到状态。
3. E1077 dual 的 event 增量在 **5/5 seeds 回退父臂**，保存的 event observer 抽查也与初始化一致；主要预测增益来自 persistent background。该患者可保留“背景辅助的动态预测信息”候选，不能承担“由间期事件学习到状态”的核心证据。
4. E922 的最大 event 风险增益来自 seed20260906，而这个 seed 的 event observer 也保存了初始化权重。H2b 冻结的是 observer 原始轨迹经 PCA 压缩的特征，并没有要求它通过上游 H1 或 learned-checkpoint 资格。

**怎么改。** 增加统一证据表：患者、family、checkpoint hash、observer 自身选中参数变化、H1 endpoint、credit endpoint、H2a/H2b 读入 hash。明确区分 `trained_observer`、`initialised_observer_features` 和 `background_predictive_state`。对 E1125 的同一 dual checkpoint 直接补 event 与 background 两条路径的 credit/删历史测试，不能用 event-only 卡代替。

抽查也有正面结果：E1125 event seed20260903 的 burden/grammar 矩阵相对初始化最大改变量分别约 **0.04129 / 0.04501**；“某些 observer 确实被学习”可以保留。该事实不自动满足同一模型的全部假设。

定位：`scripts/audit_group_event_state_v038_trained_credit.py:49`、`:90`；`src/topic5_group_event_state/v037/h2b.py:319`；`reproduced_findings.json`。

### P1：same-prefix 目前不是严格的“前缀之后分叉”评分，而且缺少常数对照

**问题。** H2a 先选出前两组触点相同的事件，但后面调用完整事件的 `_mean_scores`。底层 `per_event_scores` 仍计算所有有效步骤，包括用第一组预测已经给定的第二组，以及已知前缀内的 STOP 决策。该目标也不能充分检验同前缀之后是否停止。

本次确定性探针保持所有前缀之后预测完全相同，只改善“已知第二组”的预测，就让原 grammar 分数改善 **0.22881**；真正后缀分数改善为 **0**。因此这不是仅有命名歧义，而是可复现的目标污染。

此外汇总器的 same-prefix 门槛只检查 B_mark 和错时，**没有该前缀子集上的 FIT-constant 对照**，却把它计入“至少两项都超过三个对照”的条件。

**怎么改。** 从给定的第二组处理完成后的预测开始评分，屏蔽所有已观察目标和前缀内 STOP；在完全相同的后缀事件集上评分正确、常数、错时、B_mark。将后缀 STOP、subset、route/order 分开；用“只改已知前缀、后缀不变”的回归探针防止复发。

E253 dual 的 contact 与 rich-mark 各自仍有 3/5 同 seed 方向通过，部分形态信息值得保留；event-only 所称的“两端点支持”则失去了可靠的第二个端点，且还存在上游 observer 未更新的问题。

定位：`src/topic5_group_event_state/v037/h2a.py:765`；`src/topic5_group_event_state/v034_spatial_state/we_decoder.py:130`；`scripts/finalize_group_event_state_v038.py:334`。

### P1：训练性检查存在，但尚不能叫“所有承重臂训练充分”

**问题。** v0.3.7 修复合同已经把“非父 checkpoint 比例”从训练性门槛里分离出去。这有合理之处：训练探索后回退父臂不等于代码没有训练。但当前剩下的“首步非零、参数动过、没撞预算”仍只是较弱的优化诊断。

- 首步梯度是 observer 和 head 混合后的总范数；参数变化是训练过程中任意参数的最大变化。它们没有证明选中 checkpoint 的每个 observer 分支都更新、梯度未长期衰减、或结果对合理优化配方稳定。
- E1096 event/grid 的随机对照各 **3/5 seeds 耗尽 900 步预算**。机器汇总中其 `training_interpretable=false`，H1 候选条件却只读取学习状态训练标志；B_mark 的标志也未用于最终候选放行。因此“超过充分训练的随机对照”没有被保障。
- H2a 没有写入统一的 `training_budget_exhausted` 字段。E253 grid 有两次训练到 120 epoch、最佳点为 97/109，尚未满足 40 epoch 耐心；这些有限预算结果不能自动当作科学阴性。
- H2b 使用 LBFGS，但卡片没有保存优化终止原因、最终梯度、实际迭代数、参数变化及对应端点的正对照检出率。H2a grammar 的泄漏 oracle 不能替代 H2b 的灵敏度检查。

**怎么改。** 给每个患者的状态臂与全部承重对照采用相同的充分性记录；先补报现有曲线和选中权重，再有针对性地补预算敏感性。尤其先处理 E1096 随机对照、E253 H2a grid 及关键 H2b readout。首步最优、晚期最优、训练损失下降但 INNER 持续上升应分别诊断，不能合并为阴性。

定位：`scripts/finalize_group_event_state_v038.py:107`、`:228`；`src/topic5_group_event_state/v037/h1_train.py:640`；`src/topic5_group_event_state/v037/h2a.py:289`；`src/topic5_group_event_state/v037/h2b.py:123`。

### P1：当前没有独立检验“非线性状态演化”，也没有检验 observer 更新的生理因果性

实际 CTSSM 的核心递推是固定指数衰减加线性输入：

`z_j(t) = exp(-Δt/τ_j) z_j(t_prev) + W_j m_event`。

因此，在线性未归一化支路上，先投影再累积等价于对固定 EWMA 历史进行线性投影。grammar 的质量归一化、LayerNorm、损失 link 以及事件内 adapter 确实包含非线性；**这不等于从数据学出了非线性状态转移规律**。正式代码没有启用可学习 tau，亦没有输入依赖的状态转移。

**怎么改。** 在相同数据、归一化、读出容量和训练充分性条件下，比较固定历史、学习投影的固定滤波器、线性可学习状态和非线性状态。先确认非线性增量是否可估，再决定是否扩大结构。不要直接从 42 维改为 128 维来回答动力学问题。

观察到事件后更新是 observer 的定义；旧事件存在梯度只是模型对历史敏感。H3 需要另外比较无反馈、仅 count/rate 反馈和 mark 特异反馈，并检验共同驱动与错时 placebo。v0.3.8 主监督器没有 H3 队列，也没有把此前 H3 的有效检验迁移到这条同 checkpoint 证据链。

## 4. 科学性：原始问题现在回答到哪里

| 原始科学问题 | 本次可保留结论 | 未达到的部分 |
|---|---|---|
| 间期事件历史能否提供动态预测信息 | 有患者内探索性证据；E1125 dual 六小时与 E1096 短尺度值得跟进 | E1096 随机对照预算未合格；E1077 主要来自背景；尚无稳定队列结论 |
| observer 是否实际被学到 | E1096、E1125 event-only 分别 4/5、5/5 选择非父 checkpoint；E1125 权重更新抽查确认 | E253 event/grid 全部回退初始化；不能跨患者或跨 family 转借学习证据 |
| 是否利用多小时历史 | 已选 event-only 模型对旧事件有真实 held-out loss 梯度，审计重建轨迹有 parity 检查 | 审计的是 8 h future head；没有证明同一 dual 六小时阳性需要六小时前历史；没有删历史后的预测损害检验 |
| 是否同时预测多种事件属性 | E253 dual 的 contact 与合并 rich-mark 有初步增益；E1125 dual H1 不限于 count | same-prefix 目标需修；rich mark 是 24 维合并 PCA 目标，不能逐一声称 waveform、频带和延迟都通过 |
| 是否发现非线性演化状态 | 模型输出含非线性处理 | 未建立“非线性转移比匹配线性/过滤历史更有解释力” |
| 发作前是否出现特异状态 | E922 有很弱、控制间不一致的探索性关联 | 完整对照交集只有 1/5；无可靠 early-field/path 支持；发作前特异性和统一病理状态均未建立 |
| 每次间期事件是否改变病理状态 | 本轮不能回答 | observer 更新不等于生理改变；需独立 H3，且人类观测数据最高通常只能支持反馈样预测依赖 |

其他必须保留的边界：

- E922 的结果本质上是同一套五分钟右删失 hazard likelihood 的增益；不能把“risk”与“distance”当成两个相互独立的阳性任务。其风险正向 seed 的绝对分数改善通常很小，且校准仍有明显偏差。
- 本次重新构造 E922 的风险支持，确有 3 次不同发作；错时比较仅剩 **9 个锚点、55 行 person-period、7 个阳性重复行**，这 7 行覆盖的也是这 3 次发作。不能把 55 或 7 当独立样本。三个 onset 是否属于独立发作簇仍需另外定义与验证。
- 非重叠 future windows 是有用的分母控制，但不保证生理过程统计独立。当前以 seed 方向投票与窗口数量形成候选，缺少按时间块/发作簇的区间、效应下限、重复选患者/端点后的校正和未触碰样本上的确认。
- `B_history` 在 H2b 被 PCA 压为 8 维，附加的 event state 为 4+8 维。增加状态可能恢复了基线压缩丢掉的信息；应增加不压缩或匹配信息/容量的强历史敏感性对照。event/grid 的迁移也没有绑定与其相同 observer 的初始化对照。
- 当前记录声明没有 vigilance 调整；时钟调整不能代替睡眠、药物变化等状态混杂控制。多端点预测成立时首先说明有共享预测信息，不能直接解释成疾病易感性或生理调制。
- 阴性必须保留 `NOT_ESTIMABLE`、优化受限、仪器受限和未建立之间的区别。本次不能把未通过项升级为“病理状态不存在”或“与发作无关”。

## 5. 工程性：实际训练配置和缺失信息

### 5.1 实际结构，不是泛泛的 64/128 层宽

下表中的 H1 尺寸以 E1125 seed20260903 的保存 checkpoint 为例，均从张量实读；跨患者输出字典可能改变，完整清单在审阅目录。

| 部件 | 实际维度/参数 | 含义 |
|---|---|---|
| event burden 输入 | `7 → 14`，98 个权重，无 bias | 7 个 tau，每个 2 个 burden 通道 |
| event grammar 输入 | `24 → 21`，504 个权重，无 bias | 7 个 tau，每个 3 个 grammar 通道 |
| event state | `14 burden + 21 composition + 7 mass = 42` | observer 自身共 602 个可训练参数；tau 是 buffer |
| background 输入 | `18 → 14`，252 个权重 | 背景状态为 14 个 composition + 7 个 mass = 21 维 |
| event/grid H1 heads | B_rate 10,560；B_mark 42,168；state 6,944；random 6,944；dispersion 4 | 所有读出阶段合计 66,620；这些参数不是一次全部更新 |
| dual H1 新增头 | current background 4,752；background state 5,544 | dual 全部头合计 76,916，另有两个 observer 共 854 参数 |
| E1125 的 rich-mark 状态头 | event 部分 `28 → 96` | 28 维 grammar+mass，4 horizons × 24 维 target |
| E253 事件内 decoder | 64 个 tissue 节点 × 每节点 1 维，循环矩阵 `[1,64,64]` | 这是事件内部网络，不是跨事件慢状态；30 个 decoder 的节点数为 64/66/90/96 |
| E253 dual H2a 接口 | 输入扩展后 `160 → 8 → {64,64,8,1}`，2,376 参数 | GELU 低秩调制；静态接口另有 411 参数；decoder 主体冻结 |

正则化前提也需说明：B_mark 在原始尺度上的固定历史系数与带 LayerNorm 的低秩状态头并非完全同类函数。不能仅按“都有线性层”认定对照已经隔离非线性或容量效应。

### 5.2 学习率、优化器、batch 和初始化

| 阶段 | 当前实际配置 | 已有诊断 / 缺口 |
|---|---|---|
| H1 event/grid | AdamW；state LR=0.001，head LR=0.003，WD=0.001；β=(0.9,0.999)，eps=1e−8；warm-up 100 steps | 固定配方 `strong_decay_warm_small`；不是本轮随结果临时改 LR |
| H1 dual | AdamW；event/background observer LR=0.0003，head LR=0.001，WD=0.0001；不做 state warm-up | 固定配方 `base_small`；current background LR=0.001 |
| H1 基线 | B_rate LR=0.003、上限3600；B_mark LR=0.002、上限1800、warm-up100、WD=0.0001 | `baseline_warm_init`，先于状态配方冻结 |
| H1 步数/批次 | 每步完整重算因果 carry segments，在全体 FIT anchors 上求目标；无普通独立样本 minibatch，无 TBPTT | state 上限1800，random上限900；每25步验证，16次无改进停止，clip=2；步数不能与 H2a epoch 混用 |
| H1 初始化 | 输入投影 Normal(std=0.02)；state readout std=0.01；B_mark readout std=0.01；B_rate 权重/偏置为0 | 新增臂显式保留零增量父臂；所以“动过”与“选中权重有更新”不同 |
| H2a | AdamW；静态LR=0.001，状态LR=0.0008，WD=0.0001；batch=512；rank=8；clip=1 | 状态最多120 epoch、patience40、warm-up5；静态与oracle最多80 |
| H2a 初始化 | down层 Xavier；输出调制矩阵初始为0 | 输出先动、随后上游获得梯度是可行路径；最终卡没保存逐层梯度和规范预算状态 |
| 严格 decoder | Adam LR=0.006，clip=5；batch按患者自适应，实为56–1024 | 30/30记录converged且未触顶，实际63–200 epoch；例E253 batch471，best64/run77 |
| H2b hazard | float64，LBFGS LR=0.5，max_iter160，strong_wolfe；6档L2由INNER选择 | 6个elapsed-time baseline参数；缺实际收敛记录与可重放的 fitted readout 权重 |
| H2a rich mark | 独立 ridge readout，6档正则由INNER选择 | 保存了选择结果，未在 adapter checkpoint 内保存完整三个ridge readout及其变换 |

H1 两队列 event/grid/dual 的主要状态阶段，实际训练 **400–1050 步**，选中点 **0–650 步**；常见 400 来自 `25×16` 耐心。这个范围只说明运行过程，并不证明找到了足够好的参数。event/grid long 的 state 各 13/30 回退父臂，dual long 的 event 14/30 回退父臂；这些值已经逐卡重算。

### 5.3 normalization、数据流和复现

- 事件负荷与 grammar 分开；参与掩码、真实 delay、tied groups 被显式读取，没有直接复用未屏蔽的 legacy phantom ranks。grammar 先按 burden 做 FIT 区间 ridge residualization，再做24维PCA。前缀 `<60pct` 包含 calibration+FIT，变换不使用 INNER/SELECTION。
- 连续 target/基线特征主要用 FIT median/MAD 标准化，clip到±12；observer 读出前用无可训练仿射参数的 LayerNorm。H2a context 和 H2b PCA也在FIT拟合。没有 batch normalization 的运行统计跨分区问题。
- H1 第一步全链/全锚点目标及 horizon 等权有对应实现；query 使用严格前驱事件，事件恰好位于未来窗左端进入未来目标。H2a pre-event 查询排除当前事件。这些数据流设计可以保留。
- `rich mark` 混合了频带、delay、waveform RMS/peak/line length等，之后压缩为24维。合并MSE有增益不代表每个物理端点都有增益。需要固定PCA与标准化后，另报原单位或明确标准化单位下的各端点。
- **60 张长队列 event/grid H1 卡的 producer hash 与当前 `h1_train.py` 不同；全部90张长队列卡缺当前新增的 dependency provenance。** 差异与后续空事件段修复相容，但不能仅凭这一解释确认历史结果等价，也不能据此认定60张全错。需保存旧源码快照或证明新旧在这些患者上逐轨迹/逐分数一致。
- checkpoint没有完整保存 grammar dictionary、residualization/PCA对象和所有随机observer；随机背景审计已遇到重建 drift，并允许排除部分 mixture/mark 端点。最终汇总仍未逐卡拒绝缺失或不匹配的完整依赖hash。目录和同一个Git HEAD不能替代完整来源链。
- 原始三张PNG及PDF栅格预览都已目视检查。动态曲线在0.5/2h把两队列合并，而6/8h只有长队列，与合同“长短两层不混总体效应”不一致；曲线变化不应解释为纯horizon效应。credit热图没标出仅event-only/8h，发作图沿用了E922错误候选，same-prefix圈标也需改。

## 6. 最小修改路线

1. **先改结论生成与评分语义。** 修同seed交集、硬编码结论、same-prefix后缀mask及常数对照；重新生成一个并行审阅修正版，旧报告保留。这个阶段大部分工作无需重训。
2. **补全训练验收表。** 对每位患者/每个承重臂输出θ_init→θ_selected、逐模块梯度和更新量、FIT/INNER各端点曲线、训练终止原因。将E1096随机对照预算不足等问题显式卡住。再补少量预先锁定的LR/预算敏感性，避免无目标地扩大GPU队列。
3. **用同一已学习checkpoint补H1/H1.5。** 优先E1125 dual：区分事件与背景路径；对真实6h与8h heads做旧历史梯度、跨年龄段删除/替换和预测损害检验，配同容量初始化observer及更短历史。预先固定协议，不按SELECTION效果继续调配方。
4. **同状态补多端点跨读出。** 修后缀评价；将contact、STOP、后缀route和rich-mark物理分量分别报告。E253作为形态读出例子，明确其上游训练状态；它不能代替E1125完成统一模型资格。
5. **最后做未触碰样本上的发作确认。** 先冻结患者、family、checkpoint、lead与端点，再开确认结局。补H2b readout收敛、对应正对照与按发作簇的区间；对睡眠/时钟、最近事件和强历史进行可用的控制。E922当前只能作为需确认的探索信号，不能继续按阳性反选状态。
6. **H3独立排期。** 在上述功能预测状态可解释后，固定M0/M1/M2、事件删改/错时placebo、公共驱动与物理时间支持。计算模型的事件更新不能替代这个步骤。

## 7. 下一步建议与可用表述

核心目标不是“让四个编号各出现一个阳性”，而是让**同一患者、同一已学习状态、同一冻结来源**形成可追踪的预测证据，再评估这种证据是否能扩展到发作与反馈。

当前可用表述：

> 间期事件和非事件背景的历史中存在可用于预测部分未来事件统计及触点表达的信息。部分患者的observer已发生有效参数更新并保留对旧观测的梯度敏感性。现有证据尚未建立同一非线性病理状态贯穿多端点事件表达与发作，也未建立事件对生理状态的反馈作用。

现在最有价值的动作是修评价与来源绑定，再补关键训练诊断；无需先把全部模型重跑，也不应直接扩成更大的64/128层宽搜索。

## 审阅证据与重放

审阅工作树：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-ges-v033-training-lab`，HEAD=`f27f048cd2372efeacd6801c9cf5326a974179db`；审阅开始时有98条未提交状态，v037/v038多文件未跟踪。没有把这个HEAD宣称为包含全部执行源码的完整提交。

本次只增加此独立归档与审阅目录：

`/data/hfosp_group_event_state_v0_3_8_core_expansion/independent_review_2026-09-05/`

- `audit_existing_artifacts.py`：逐卡训练、候选交集、参数维度、来源及E922真实结局支持重算；不训练模型。
- `reproduce_review_findings.py`：真实E922候选逻辑、只改前缀的评分反例、已选observer相对初始化的检查。
- `training_stages.csv`、`training_stage_summary.json`、`effective_h1_configs.json`：实际训练信息。
- `checkpoint_tensor_shapes.json`、`decoder_checkpoint_shapes.json`、`adapter_checkpoint_shapes.json`：实读权重维度；state_dict里的buffer不计作可训练参数。
- `source_card_manifest.json`：165张H1卡的hash与当前producer比较。
- `h1_candidate_joint_gate_audit.json`、`h2a_candidate_joint_gate_audit.json`、`h2b_candidate_joint_gate_audit.json`、`e922_actual_seizure_support.json`、`reproduced_findings.json`：主要复核证据。
- `figures/`：原PDF预览与中文README，非修正后的科学图。

复跑原回归：在上述工作树执行

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest -q tests/test_group_event_state_v037*.py tests/test_group_event_state_v038_audits.py
```

本次结果：**91 passed in 3.70s**。这91项未包含v038完整候选交集或严格same-prefix后缀目标测试，因此通过并不与本次发现冲突。
