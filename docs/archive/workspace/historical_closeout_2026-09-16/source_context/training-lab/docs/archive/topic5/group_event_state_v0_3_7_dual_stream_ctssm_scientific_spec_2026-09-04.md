# Group-Event State v0.3.7：双流连续时间状态 Scientific Spec

**日期：** 2026-09-04  
**状态：** `ARCHITECTURE_DECISION_FROZEN_IMPLEMENTATION_PENDING`  
**简称：** v3.7；仓库内规范版本写作 v0.3.7

## 1. 核心目标不因换架构而改变

v0.3.7 更换的是“如何记住很长历史”，不是“要回答什么”。整个项目仍回答三层问题：

1. **H1/H2a：** 过去的群体间期事件能否形成一个按真实时间持续、对得上时刻的预测状态；这个状态是否不仅预测事件多少，还预测稳定局部传播规则之上的 community occupancy、跨 community coupling、repertoire mixture、STOP/extent、路径和频带表达？
2. **H2b：** 这个只从间期数据学到并冻结的状态，能否预测距发作的时间，以及下一次发作早期的空间能量场和传播路径？
3. **H3：** 控制慢背景和发作前状态后，IED 的数量或内容是否仍对更远未来提供额外的、具有方向性的反馈式依赖？

不能用下一事件 rate 的改善代替上述问题，也不能用一个隐藏向量的存在代替功能性证据。

## 2. 两种“状态”必须从名字、代码和证据上分开

### 2.1 Observer state

`S_obs(t)` 是过去可见信息对未来的因果预测摘要。它可以在看到 IED 后更新，因为 observer 得到了新证据。其更新只表示“估计改变”，不表示脑网络被 IED 改变。

### 2.2 Generative physiological state

`Z_phys(t)` 只存在于 H3 的显式生成模型。H3 通过允许或禁止 event jump 的嵌套模型比较，判断 event-feedback 边是否有必要。

代码中禁止用同一个类、字段或 checkpoint 同时表示 `S_obs` 和 `Z_phys`。任何 observer perturbation 都不能单独承担 H3 结论。

## 3. 输入流与信息边界

### 3.1 群体事件主流

一次完整群体间期事件对应一个 event-time step。事件表示保留：

- participation 和 tied groups；
- 精确相对 delay；
- bipolar/CAR 波形摘要；
- 多频带能量、峰时与跨频带 lag；
- extent、duration、community 和 repertoire 信息。

事件 encoder 在严格早期时间块上训练/校准后冻结，并拆成：

- `u_N`：负荷信息，包括事件规模、总能量、extent、community 数量等；
- `u_G`：在控制负荷、近期 rate 和已知 context 后的条件 grammar 信息。

`u_G` 的连续分量使用只在训练块拟合的 cross-fitted residualization；类别型 repertoire/community 使用衰减 pseudo-count、归一化 composition 和 effective mass，避免事件率越高就机械放大 grammar state。

### 3.2 连续非事件背景流

每 5 分钟从此前 30–60 秒有效非事件 SEEG 产生一帧背景观测。首版使用可审计特征：多频带 power、aperiodic slope/offset、方差、自相关、line length、community connectivity、空间 dispersion、bipolar/CAR 一致性、坏道与 exposure。

主分析排除 IED 邻域；包含邻域版本只作敏感性。缺失背景只增加不确定度并传播已有状态，不制造“没有事件”的证据。

### 3.3 已知 context

在数据可得时显式加入 causal sleep probability、time-of-day、time since last seizure、postictal kernel、用药/刺激、住院日、植入后时间和 channel availability。它们是可解释协变量，不能藏进 latent 后再声称发现了新状态。

## 4. 状态架构

### 4.1 透明长历史基线

先对 `u_N/u_G` 建固定物理时间 EWMA bank：10 min、30 min、1、2、4、8、16 h。无信息事件不能额外擦除历史；时间尺度由墙钟决定，不随事件密度变化。32 h 只保留为架构敏感性：只有留出段显著长于 32 h 时才允许评价，不能进入当前人体主比较。

对类别 grammar 保存归一化 composition 和有效样本量。该基线叫 `B_mark`，是完整学习模型必须超过的科学基线，而不是临时 sanity check。

### 4.2 主 observer：event-impulse CT-DSSM

事件之间按真实 `dt` 传播：

`s_e^- = exp(A * dt_e) s_(e-1)^+`

事件作为观测脉冲写入：

`s_e^+ = s_e^- + B_N u_N,e + B_G u_G,e`

首版 `A` 固定或为有界稳定对角矩阵；禁止输入依赖的 `A(u_e)` 和按患者自由选择时间常数。不同事件的仿射转移用 associative scan 组合，完整反传，不使用会截断科学记忆的 TBPTT。

背景流在固定 5 分钟网格用独立连续时间 SSM 更新，形成 `Z_B(t)`。最终：

`S_obs(t) = [Z_B, M_N, M_G, R_fast, known_context, uncertainty]`

其中 `R_fast` 专门承接分钟级 refractory/burst；慢状态不被迫同时完成短程过滤。

### 4.3 时间尺度与预测 horizon 分开

10 min–16 h 是当前人体主分析的记忆基函数，不是生理结论；32 h 是显式标注的探索性架构敏感性。一个共享的 `S_obs` 同时进入独立低容量 heads，预测 0.5、2、6、8 h；12 h 仅在可估患者中扩展。禁止为每个 horizon 独立训练 producer 后再把四个模型称作“同一个持续状态”。

## 5. H1：未来负荷与条件 grammar

### 5.1 负荷状态

future burden 使用带 valid-exposure offset 的 Negative Binomial likelihood。未观测时间不当作 silence。比较顺序：

`constant -> B_rate -> B_mark -> S_event -> S_dual -> S_grid`

### 5.2 条件 grammar 状态

future grammar 必须条件于未来 event count 和 valid exposure，分开报告：

- community occupancy；
- cross-community coupling；
- repertoire mixture；
- continuous event-embedding distribution；
- conditional spatial/frequency field；
- STOP/extent distribution。

它回答“同样有这么多事件时，事件会以什么空间—频带方式表达”，不能被 count 提升冒充。

### 5.3 H1 的承重比较

同一 anchor、同一 horizon、同一结果分母上比较：

1. `B_mark` 与 `S_obs` 的嵌套增量；
2. correct-time state 与同 session block-circular shift；
3. event-only 与 background-only、dual-stream；
4. count 与 conditional grammar 分解；
5. 一个共享 producer 的跨 horizon 曲线。

只有超过强动态基线并且 correct-time 优于 shifted，才称时刻特异 predictive state。背景流吸收 event-history 增量时，结论是 IED 主要报告共同背景；吸收后仍有增量，才称未被背景解释的历史依赖。

## 6. H2a：状态如何调制一次事件的传播

使用按物理时间切分、通过泄漏审计的成熟 contact-sequence decoder。decoder 主体冻结；事件前状态经低维投影得到 `b_e`，在该事件的每一个 decoding step 保持不变并调制：

- continue/STOP；
- 继续时的 positive group size；
- contact subset logits；
- route/order transition；
- conditional band/waveform expression。

`b_e` 只能读取 `t_e^-` 以前的信息；当前事件完整 mark 不得进入。decoder 同时读取已经真实观察到的 event prefix，用于 same-prefix continuation。

状态时间语义固定为：`S_obs(t_e^-)` 不含第 `e` 次事件，供该事件的 H2a 解码；观察完整第 `e` 次事件后才形成 `S_obs(t_e^+)`，供之后的 H1/H2 预测。固定时间 anchor 只汇入严格早于 anchor 的事件。

主比较：prefix-only、prefix + `B_mark`、prefix + correct `S_obs`、prefix + shifted `S_obs`。并做三类结构分析：经验支持内 state interpolation、same-prefix state swap、decoder 局部 Jacobian/STOP sensitivity。零状态必须逐位复现冻结 decoder；泄漏 future-state oracle 必须证明接口有灵敏度。

H2a primary 是严格跨读出测试：`S_obs` producer 与事件内 decoder 主体都冻结，只训练 state-to-context projection 和低秩 modulation adapter。另设 interictal-only joint sensitivity，允许 decoder loss 穿过冻结 decoder 更新 producer；该版本必须重新评分 H1，且不能替代 primary。这样既检验既有状态能否跨读出，也不丢掉“事件形态目标是否能帮助学习状态”这一条探索线。

## 7. H2b：冻结的间期状态跨到发作

H2b 不把 seizure label 回传到 event encoder、memory core、时间尺度或 background encoder。

并列主任务：

1. 固定时间网格上的离散 survival hazard，输出 time-to-next-seizure 风险与距离；
2. 在 6 h、2 h、30 min、5 min 等 lead time 预测 held-out seizure 最初 5–10 秒的空间能量场、early recruitment vector、laterality、extent、传播轴和 IED-to-ictal reuse。

比较患者平均发作场、近期 IED、`B_mark`、event-only state、dual-stream state。统计单位是患者/发作，不是 grid row。无足够发作或独立窗口记为不可估，不记为阴性。

H2b 与 H1 不设结果 gate：即使 H1 对某一 endpoint 阴性，预先登记的 frozen producer 仍进入跨任务读出，避免只把 H1 漂亮状态送去发作任务。

## 8. H3：独立生成模型

H3 使用 `Z_phys` 的连续—离散随机生成模型，同时建模有效观测区间内的 IED 强度和 mark。比较三个容量匹配、其他部分相同的模型：

- `M0 common-drive`：事件是 `Z_phys` 的读数，不反馈；
- `M1 count feedback`：增加低秩 burden jump；
- `M2 mark feedback`：在 M1 上增加低秩 conditional-grammar jump。

主要 estimand 是未见 future block 的 log-score 增量，并分别报告 burden effect 和 mark-content effect。必须带动态慢水平/背景、截距匹配和 patient-level 独立块；mark 比较保持事件数与时刻不变，count 比较不能把 exposure count 匹配掉。

即使 M2 稳定超过 M0/M1，允许结论仍是“mark-specific directional dependence”；没有干预不能写成人体因果效应。

## 9. 模型梯与选择规则

| 模型 | 科学作用 |
|---|---|
| `M_old` | 旧门控 RNN，失败参照 |
| `B_rate` | 动态事件率/概要基线 |
| `B_mark` | 透明多尺度完整 mark 基线 |
| `S_event` | event-only diagonal CT-DSSM |
| `S_dual` | background + event 双流 CT-DSSM，主模型 |
| `S_grid` | 5 min 分层慢 SSM，计算折中 |

第一轮不让 low-rank `A`、Mamba、attention 或输入依赖 transition 参加主选择。只有 diagonal CT-DSSM 在相同数据、目标、seed 和优化预算下稳定超过 `B_mark`，才进入受控容量扩展。

## 10. 数据与泄漏合同

- base event decoder 只能来自其他患者预训练或本患者 calibration prefix；patient adapter、normalization、contact vocabulary/order、tied-group 统计和 checkpoint selection 均不得看最终 future block；
- split 按物理时间；同 session 内 chunk 只能 carry/detach，不能 reset 或乱序；CT-DSSM 主训练不做截断信用；
- 小于等于 10 分钟的人工切口可在状态轴桥接，但缺口保持零 exposure；发作边界不得静默跨越；
- 每个 tensor 记录最大可见时间，任何大于预测 anchor 的依赖直接判泄漏；
- horizon-specific head 可独立选择，shared producer 只能由间期 TRAIN/INNER 目标选择；
- development、seizure 和 sealed 结果不得反向选择 producer。

## 11. 不可妥协的仪器检查

1. 物理时间 impulse/gradient response 与设计的 `exp(A dt)` 一致；
2. 保持墙钟和 mark 分布、改变事件密度时，慢模式时间常数不变；
3. 插入 `u=0` 的事件不能擦除历史；
4. 提高 event rate 时 `S_N` 改变，但 normalized `S_G` 不机械放大；
5. 72 小时 replay 中状态范数、Jacobian 和输出有限；
6. 状态读出使用小非零初始化，第一步上游梯度非零；
7. gap 只传播状态/不确定度，不创造 exposure、anchor 或 silence evidence；
8. causal max-visible-time 审计通过；
9. 统计与裁决以患者为单位。

这些检查只判断仪器是否能测，不用合成数据替代人体可训练性或科学结论。

## 12. 允许的结论阶梯

| 结果 | 允许结论 |
|---|---|
| `B_rate` 最佳 | 可预测慢变化主要是 event burden |
| `B_mark > B_rate` | conditional event composition 有长历史信息 |
| `S_event` 不超过 `B_mark` | 状态可由少数透明多尺度统计描述 |
| `S_event > B_mark` 且 correct > shifted | 学习型、时刻特异的 event-history predictive state |
| background 消除 event gain | IED 主要报告共同背景 |
| dual-stream 后 event 仍有增量 | 未被已测背景解释的 event-history dependence |
| H2a same-prefix 增量 | 状态调制事件路径/终止，而非只调 rate |
| frozen H2b 增量 | 间期状态含跨任务的发作易感/路径信息 |
| M1 > M0 | burden feedback-like dependence |
| M2 > M1/M0 | mark-specific directional dependence |

只有新模型和强基线都通过仪器检查后仍无增量，才允许形成有解释力的 6–8 小时阴性。

## 13. 方法来源

本设计借用而不照搬以下思想：LFADS 的 initial condition/dynamics/input 分离与 contextual bias；HiPPO/S4/S5 的长历史连续时间表示；NCDSSM/GRU-ODE-Bayes 的 recognition 与 dynamic state 分离；Neural Hawkes/S2P2 的不规则 marked event likelihood；DPAD 的目标相关分阶段学习。关键参考包括 [LFADS](https://www.nature.com/articles/s41592-018-0109-9)、[HiPPO](https://proceedings.neurips.cc/paper_files/paper/2020/hash/102f0bb6efb3a6128a3c750dd16729be-Abstract.html)、[S5](https://arxiv.org/abs/2208.04933)、[NCDSSM](https://proceedings.mlr.press/v202/ansari23a.html)、[Neural Hawkes](https://proceedings.neurips.cc/paper/2017/file/6463c88460bd63bbe256e495c63aa40b-Paper.pdf) 和 [DPAD](https://www.nature.com/articles/s41593-024-01731-2)。
