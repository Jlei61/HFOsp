# Group-Event State v0.3.5 Scientific Spec

## Dynamic baseline, full-event state and step-wise frozen-decoder modulation

**状态：** `V0_3_5_EXECUTION_LOCKED`  
**日期：** 2026-09-03  
**取代范围：**取代 v0.3.4 中“固定重标定偏置 + 只在事件初态注入状态”的主实验设计；v0.3.4 结果保留为诊断证据，不按新合同重新解释。  
**数据范围：**只使用 development 体系中的 TRAIN、rolling-inner 与已登记 selection 分区；formal/sealed 分区保持关闭。

> **2026-09-04 长窗与共享状态修订：**状态可跨过不含已知发作的 `<=10 min` 短缺口继续传递，但 anchor 与有效观测时长仍严格来自原始覆盖区间；缺口中的秒数贡献零 exposure，不能被当作“没有事件”。horizon-specific 计数模型只作为 L0 难度/可估性基线，可以各自训练。完整状态阶段改为：`S_N(t)` 与 `S_G(t)` 各自只有一个跨 horizon 共享的 producer；2/6/8 h evaluator/head 可以独立拟合，但都读取同一条冻结状态轨迹。12 h 为较小子队列探索，24 h 仅作个案。禁止将 6 h 动态计数基线写成 H1 已完成，也禁止分别训练 `S_2h/S_6h/S_8h` 后拼成“持续状态”。

**执行完整性：**本版本按 W0–W6 全工作包执行。接口 smoke、synthetic recovery、单患者 canary 或任一单端点阳性，都不能替代人体多患者的动态基线、逐步 contact 调制、多时间尺度 H1/H2a、冻结 H2b 与显式反馈 H3。某个 patient×endpoint 不可估只限制该单元的解释，不取消其他已登记工作包。

## 0. 一句话目标

从连续群体间期事件中学习两个可区分、可组合的患者内过程：

1. 随真实时间缓慢变化的事件负荷/记录阶段水平；
2. 在负荷水平之外，随事件空间传播、精确延迟、多频带表达和事件内波形变化的网络表达状态。

两者共同调制一个预训练并冻结的患者内 contact-sequence decoder，而且状态必须在事件内部的每一步都能改变后续招募与停止，而不只改变事件起始隐变量。随后冻结间期状态，检验它是否迁移到发作距离和发作早期传播场；最后再比较 common-drive 与 event-feedback 模型。

## 1. 当前 v0.3.4 实际做了什么

### 1.1 当前慢状态的精确演化

每次完整群体事件先被编码成一个写入向量 `u_e`。同一个 `u_e` 被写入 5、30、120 min 三个固定时间尺度的漏积分器：

```text
s_tau(t_e-) = exp(-dt_e / tau) * s_tau(t_{e-1}+)
s_tau(t_e+) = s_tau(t_e-) + u_e
```

等价地，在一个连续记录段内：

```text
s_tau(t) = sum_{j: t_j <= t} exp(-(t-t_j)/tau) * u_j
```

这意味着：

- 事件之间只有固定指数衰减；
- 不存在学习得到的 state-to-state transition；
- 不存在连续背景输入；
- “一段时间没有事件”没有独立 likelihood，只表现为状态衰减；
- 三个尺度收到完全相同的事件写入，只是遗忘速度不同；
- 当前最长核心时间常数为 120 min。

因此它更准确的名称是 **marked leaky history bank**，而不是已经识别出的生理慢动力学。

### 1.2 当前状态如何进入 contact decoder

成熟的患者内 tissue/contact decoder 被冻结。v0.3.4 只在每次事件开始时使用：

```text
h0 = b_static + A * s(t_e-)
```

事件开始以后，decoder 按事件前缀递归展开，但 `s(t_e-)` 不再显式进入后续每一步。这个接口能检验“不同初始组织状态是否足够改变整场事件”，但有两个明显限制：

1. decoder 的递归会逐步冲淡 `h0`；
2. 状态不能在相同早期前缀后，直接改变某个 contact、STOP 或 later recruitment 的逐步决策。

### 1.3 v0.3.4 的安全结论

时间重切分后的成熟 contact decoder 本身有效；故意把未来参与场作为状态时，E253、E548、E583 的接口能检出改善。可是，在仅拟合静态重标定偏置后，当前跨事件 leaky state 在 E253、E548、E583 上没有增加可测的未来传播形态信息；E1146 连正对照也不敏感，E922 不可估。

这只是否定了以下组合：

> 三档固定漏积分历史 + 当前事件 token + 只注入 `h0` + 30 min contact-grammar 评分。

它不等于“间期事件没有时变网络状态”。

## 2. 为什么静态 bias 应当进入动态 baseline

### 2.1 静态 bias 不是坏东西

一个只用 TRAIN/calibration prefix 拟合的患者内静态偏置，表示预训练 decoder 与该患者/该记录阶段之间的平均差异。它是合法且必要的零阶校准：

```text
b(t) = b0
```

如果时变项学不到任何稳定信息，新模型应当精确退化回这个静态模型，而不是被迫比它更差。

### 2.2 但单一静态 bias 不可能覆盖整段记录

Topic 2 已在真实物理时间上证明，5 min binned event rate 具有多小时漂移：

- 相邻 IEI 相关在 30/30 患者为正，中位约 0.299；约 72% 可由 10 min 以上的慢漂移解释；
- 事件率没有单一主时间尺度，而是宽时间尺度叠加；
- Epilepsiae 队列的 8 h rate autocorrelation 中位仍为正；Yuquan 到 4 h 仍为正；
- 当前重点患者中，E253 的 0.5/2/8 h rate autocorrelation 为 0.655/0.539/0.402，E922 为 0.664/0.515/0.233，E1096 为 0.806/0.700/0.582；即使较弱的患者也常在 0.5–4 h 保留相关。

所以“固定 decoder + 一个全记录不变偏置”只适合表示平均水平，不能代表真实记录中随时间变化的负荷水平。

但必须保持结论边界：这些结果直接证明的是 **rate-level time variation**，不自动证明 contact sequence、延迟或能量场以同样方式变化。v0.3.5 必须把两件事分别测量。

### 2.3 v0.3.5 的嵌套定义

decoder 的时变条件写成：

```text
c(t) = b0 + W_rate q(t) + W_mark m(t)
```

其中：

- `b0`：TRAIN/calibration prefix 学到的静态患者/阶段偏置；
- `q(t)`：由事件到达与无事件区间因果估计的动态负荷基线；
- `m(t)`：由完整群体事件内容更新、并在负荷基线之外预测未来形态的状态。

这三个对象是嵌套关系：

```text
static calibration        W_rate=0, W_mark=0
dynamic rate baseline     W_mark=0
full predictive state     W_rate!=0, W_mark!=0
```

因此，动态基线学会记录阶段变化是预期中的好结果。科学问题不是“动态模型能否打败静态偏置”这一句，而是：

1. 过去信息能否在每个时刻因果估计当前水平；
2. 该水平是否只解释事件率，还是也解释事件形态；
3. 完整事件内容是否在 rate state 之外增加网络表达信息。

## 3. v0.3.5 的状态结构

### 3.0 两个共享 producer，而不是长窗计数替代状态

最终登记两个不同的 marked-history predictive state：

- `S_N(t)`：负荷状态。它读取完整的既往群体事件与有效 silence evidence，联合预测未来 5 min、30 min、2 h、6 h、8 h 的事件数/无事件概率。horizon-specific 负二项 head 可以不同，但 producer 和状态轨迹必须相同。
- `S_G(t)`：传播 grammar 状态。它读取完整事件的 contact、lag、波形和多频带 mark，在已经控制 `q(t)` 与未来事件数后，联合预测多个 horizon 的条件传播分布。它与 `S_N` 分开训练并分别冻结，避免高方差计数目标吞掉传播信息。

`q(t)` 是可解释动态基线，不等于 `S_N`；当前已经完成的 horizon-specific rate 作业只登记为 `H_N dynamic baseline`。`S_N/S_G` 可以共享冻结的 event encoder 前端，但各自拥有独立状态更新和科学目标。最终必须交叉读取 `S_N -> grammar`、`S_G -> burden`、`S_N+S_G`，判断两种状态是否真的可区分。

每位患者的 producer checkpoint 只依据训练期内的 multi-horizon INNER objective 选择一次。各 endpoint/horizon evaluator 在 producer 冻结后独立校准并在各自 holdout 报告。multi-horizon objective 对 patient×horizon×endpoint 先标准化、再等权汇总，不能让事件数最多或数值最大的 horizon 自动支配状态。

### 3.1 `q(t)`：动态负荷/阶段基线

使用 1 min 有效观测网格更新，5 min 网格评价。每个网格记录：

- 有效观测秒数；
- 群体事件数；
- time since last event；
- 近期 extent/STOP 和参与负荷的简单摘要；
- local clock 与 session position；
- IED-core-masked 连续背景 SEEG 的低容量辅助摘要/embedding；
- 可用时的 sleep/medication covariate，缺失时不伪造。

连续背景只作为固定时间网格上的辅助 observation，必须另报 `event-time-only` 与 `event-time+background`，不能取代群体事件主序列。

采用固定的物理时间 bank：

```text
2 min / 10 min / 30 min / 2 h / 8 h
```

短尺度覆盖 Topic 2 的事件级局部记忆，长尺度覆盖已观察到的 rate drift。24 h 只在长连续记录中作探索，不进入首轮核心模型。

第一版 `q(t)` 使用可解释的因果状态更新：

```text
q_tau(t_i) = rho_tau(dt) * q_tau(t_{i-1})
             + (1-rho_tau(dt)) * standardized_past_rate_innovation_i
```

`standardized_past_rate_innovation` 基于该患者自己的负二项 count model 和有效 exposure 计算，避免高事件率患者因为数值更大而获得更大的状态写入。缺失区间只推动时间，不提供“零事件”证据。

计数模型：

```text
N_i ~ NegativeBinomial(mu_i, dispersion_patient)
log(mu_i) = log(valid_exposure_i) + patient_intercept + f(q(t_i), clock, session_position)
```

`q(t)` 可以先用确定性 multiscale filter 形成强基线；再用小型有残差连接的 state-space update 学习其修正。静态 `b0` 是该模型所有动态权重为零时的精确特例。

### 3.2 `m(t)`：完整事件内容的网络表达状态

一个 timestep 仍是一场完整群体事件。每次事件输入不只包含 rank，而包含：

- participation mask 与 tied groups；
- 每触点连续毫秒 centroid lag；
- bipolar/CAR 后的事件内波形 embedding；
- 每触点多频带能量、峰时和跨频带 lag；
- contact 坐标、shaft 与缺失 mask；
- 当前 `q(t_e-)`，使 mark update 可以在已知负荷水平下解释事件内容。

事件编码器输出 mark innovation：

```text
v_e = EventEncoder(full_event_e, contact_scaffold, q(t_e-))
v_resid_e = v_e - E[v_e | q(t_e-), recent simple history]
```

`m(t)` 按真实时间在同一 time-scale bank 上传播，并由 `v_resid_e` 更新。第一版保留固定衰减作为稳定先验，但允许小型 gated residual transition：

```text
m(t_e-) = Decay(m(t_{e-1}+), dt)
m(t_e+) = m(t_e-) + Gate(m(t_e-), q(t_e-), v_resid_e) * Delta(v_resid_e)
```

这比旧版的改进是：不同时间尺度可收到不同写入，写入可依赖当前状态，且 mark state 被明确训练为 rate baseline 之外的残差。固定衰减仍只是模型先验，不能把某个通道直接命名为生理时间常数。

`m(t)` 在最终登记中对应 `S_G(t)`，但其目标不能只是一整个高维 contact subset。患者内局部传播 grammar 可以长期稳定，而慢变化表现在其上层组合，因此 `S_G` 同时预测四层条件对象：

1. **局部 grammar**：continue/STOP、继续时的 positive group size、给定大小后的 contact subset、later continuation 与 conditional multiband expression；
2. **community occupancy**：未来块内事件进入各固定患者内 community 的条件比例；community 只由 calibration/FIT 的 co-participation/scaffold 建立并冻结；
3. **跨 community coupling**：在事件数、group size 和局部前缀已知后，community 之间的募集/共现转移；
4. **repertoire mixture**：TRAIN-only event embedding dictionary 上的未来 mixture weights，并用连续 embedding-distribution score 作不依赖聚类数的并行读出。

所有块级 grammar 端点都条件于窗口内确实发生的事件，和 burden likelihood 分开。这样，事件率升高不会自动制造“所有 repertoire occupancy 都升高”的假状态。

### 3.3 pre-event 与 post-event 语义

- 预测当前事件形态时，只能用 `q(t_e-)`、`m(t_e-)`；当前事件自身不能先进入状态。
- 当前事件被完整观察后，才生成 `m(t_e+)`，用于预测之后的事件或未来时间块。
- future-block 评价中，anchor 之后的真实事件不得再写入状态。只允许状态按时间自主传播，避免 teacher forcing 把答案带进未来。
- next-1/5/20-event 的条件形态读出同样遵守这一点：使用 anchor 的 post-event state，并按 anchor 到目标事件的精确真实 `dt` 闭式演化到目标时刻；中间真实事件不写回。禁止把未衰减的 anchor state 直接复制给不同物理时长的目标。

## 4. 冻结 contact-sequence decoder 的逐步状态接口

### 4.1 保留成熟 decoder 的职责

患者内成熟 decoder 继续负责：给定已经发生的 tied-group/contact 前缀，预测下一组招募与 STOP。其主干权重在状态训练前冻结，状态网络只负责提供跨事件上下文。

v0.3.5 pilot 保持旧 decoder 的原训练 objective 与时间重切分，不能同时更换 scoring objective。新的 exact subset likelihood 如需采用，必须另行重训并单独比较。

### 4.2 状态在每一步进入，而不是只进入 `h0`

对事件内第 `k` 个 recurrence step：

```text
h_k_base = FrozenDecoderStep(h_{k-1}, prefix_k)
c_e      = concat(q(t_e-), m(t_e-))

gamma_k, beta_k = LowRankFiLM(c_e, step_fraction_k)
h_k_cond = h_k_base * (1 + gamma_k) + beta_k

contact_logits_k = FrozenContactReadout(h_k_cond)
                   + ContactSpecificLowRankShift(c_e, prefix_k)
stop_logit_k      = FrozenStopReadout(h_k_cond)
                   + StopShift(c_e, step_fraction_k)
```

必要约束：

- contact shift 必须是 contact-specific，不能给所有候选触点加同一常数后在归一化中抵消；
- FiLM/shift 以零初始化或极小初始化开始，使初始输出逐位复现 frozen decoder；
- adapter 为低秩残差，主干 decoder 永远冻结；
- decoder 冻结不阻断梯度，loss 仍穿过 decoder 和 adapter 更新 `q/m` producer；
- `c_e` 在该事件内保持为 pre-event state；事件内前缀由 decoder 自己递归更新；
- continue/STOP 与 contact recruitment 分开报告。

### 4.3 容量匹配的嵌套比较

所有臂使用同一套逐步 adapter 位置与参数量：

1. `static_only`：`c_e` 为 TRAIN 学到的常数；
2. `rate_dynamic`：`c_e=[q(t_e-),0]`；
3. `mark_dynamic`：`c_e=[0,m(t_e-)]`；
4. `rate_plus_mark`：`c_e=[q(t_e-),m(t_e-)]`；
5. `block_shift`：保持轨迹分布和自相关，错开其真实时刻；
6. `future_oracle`：仅作灵敏度上界，不进入科学结论。

比较的是同一 adapter 对不同因果输入的利用，不再让动态模型与一个缺少 adapter 的旧 decoder 比容量。

## 5. 训练目标：不再只预测下一事件

### 5.1 负荷目标

在固定物理时间 anchor 上预测未来：

```text
5 min / 30 min / 2 h
```

的 event count/silence。6–8 h 在具有足够连续 coverage 的患者中探索性报告。count likelihood 按有效 exposure 建模。

### 5.2 contact-sequence 与网络表达目标

对 anchor 之后：

- 下一 1、5、20 场事件；
- 未来 5、30、120 min 内所有可见事件；

分别评价：

1. continue/STOP；
2. 后续 contact/tied-group recruitment；
3. event extent；
4. 连续毫秒 lag/direction；
5. per-contact multiband energy、peak time 与 cross-band lag；
6. 低维事件内 waveform expression。

future block 内的事件数与“给定有事件时的形态”必须分开计分，避免 rate 变高自动改变 occupancy 后被误称为形态状态。

### 5.3 same-prefix continuation

这是 H2a 的决定性任务。对具有相同或相近早期前缀的事件，比较：

```text
p(later path | early prefix, static/rate baseline)
p(later path | early prefix, static/rate baseline, m(t_e-))
```

前缀至少包括首个 tied group，并在可估时扩展为前两个 tied groups、前 50–100 ms 波形与早期能量范围。主要终点是 later contacts、STOP/extent、later lag 与能量场，而不是首触点。

## 6. 患者事件率差异与统计口径

### 6.1 不把 event row 当独立样本

- 慢状态与风险使用固定 5 min 物理时间 anchor；
- 每个 anchor 等权，高事件率时间段不因事件更多而自动获得更大权重；
- contact-sequence loss 先在 anchor 内平均，再按独立时间块和患者汇总；
- 推断顺序为 block → seed → patient → cohort。

### 6.2 患者间 rate 不同不是要消掉的 nuisance

每位患者有自己的截距、离散度和因果慢水平。分析分成两问：

1. `q(t)` 本身能否解释患者内随时间变化的负荷和发作风险；
2. `m(t)` 是否在 `q(t)` 之外解释 contact sequence、延迟、能量和发作路径。

如果只有 `q(t)` 有用，允许结论是“存在患者特异的动态负荷/易感水平”；这仍是状态发现，但不是 network repertoire state。

### 6.3 时间尺度与可估性

统一候选尺度由 Topic 2 先验固定，不为每位患者事后挑最佳尺度。每个患者报告：

- 连续 coverage；
- 各 horizon 的完整 future blocks；
- 以该 horizon 为相关长度时的有效独立块数；
- 选择期与评价期的事件率范围。

样本少只扩大区间或标记 `not_estimable`，不按结果方向剔除。5/30 min 为首轮共同输出，2 h 为长状态主要探索；6–8 h 只在长连续患者中报告，不进入全队列 AND gate。

## 7. H1、H2a、H2b、H3 的递进解释

### 7.1 H1：有没有可因果追踪的动态状态

依次报告：

1. 静态患者/阶段校准；
2. 动态 rate baseline 相对静态校准的增量；
3. 完整 event-content state 相对 rate baseline 的增量；
4. correct-time 相对 block-shift；
5. 5 min、30 min、2 h 的持续预测曲线。

rate-only 阳性与 mark-residual 阳性是两种不同层级的发现，不互相取消。

### 7.2 H2a：状态是否改变事件传播形态

使用逐步调制的 frozen contact decoder，重点看 same-prefix 之后的 later recruitment/STOP，以及附近 1/5/20 场事件和固定 future block 的 conditional morphology。

### 7.3 H2b：间期状态能否跨任务预测发作

在 `q/m` producer 完全冻结后，训练两个独立 readout：

1. 固定 5 min 网格的单调离散 survival hazard，预测距下一次发作的分段风险；
2. 在 5 min、30 min、2 h、6 h lead 预测下一场发作最初 5–10 s 的 per-contact energy/recruitment field 与 early path。

必须并列比较 `rate only`、`mark only`、`rate+mark` 与临床/历史基线。发作 loss 不回流到间期 state producer。

### 7.4 H3：IED 是否还需要反馈边

H1/H2a/H2b 首轮完成后，再在同一 `q/m` 表示上比较：

- `M0_common_drive`：事件只是状态读出；
- `M1_rate_feedback`：事件负荷对后续状态有额外 signed update；
- `M2_mark_feedback`：事件空间/频带内容对后续状态有额外 signed update。

observer 因看到事件而更新 belief，不等于事件改变真实生理状态。H3 的人体最高措辞仍是 `event-feedback-like predictive dependence`。

## 8. 最少但承重的对照

核心只保留：

1. `static_only`；
2. `rate_dynamic`；
3. `rate_plus_mark`；
4. `block_shift`；
5. `future_oracle` 灵敏度上界。

`times-only` 与 `mark-shuffle` 只在 `rate_plus_mark` 出现增量后用于定位来源，不作为所有实验的前置 gate。reset 大网格、逐患者挑最佳 tau、无穷 null 扫描不进入 v0.3.5 Core。

## 9. 允许的结果命名

| 观察结果 | 允许结论 |
|---|---|
| TRAIN 静态偏置改善 | 患者/记录阶段校准差异 |
| 因果 `q(t)` 胜静态偏置 | 动态负荷/阶段状态 |
| `m(t)` 胜 `q(t)`，但错时不敏感 | event-content predictive memory，时刻专属性不足 |
| `m(t)` 胜 `q(t)` 且 correct 胜 block-shift | 时刻特异的网络表达预测状态 |
| same-prefix later path 改善 | state-dependent repertoire/propagation |
| frozen `q(t)` 改善 seizure risk | rate-linked interictal susceptibility state |
| frozen `m(t)` 在 `q(t)` 外改善 risk/field | network-expression susceptibility state |
| M1/M2 胜 M0 | event-feedback-like predictive dependence |

任何一层阴性不改写为“生物学不存在”。

## 10. 核心产出图

1. **动态基线图：**真实 5 min rate trace、静态水平、因果 `q(t)` 与 0.5–8 h autocorrelation；说明静态偏置是均值，动态基线追踪患者内漂移。
2. **H1/H2a 图：**横轴为 5 min、30 min、2 h，分别画 count 与 conditional morphology 相对静态/rate baseline 的增益；右侧为 same-prefix later-path panel；correct-time 与 block-shift 同图。
3. **H2b 图：**`rate only`、`mark only`、`rate+mark` 对 seizure hazard 与 early ictal field 的 lead-time 曲线。
4. **H3 图：**M0/M1/M2 的 held-out future-block score 与 signed impulse response；只在前述状态和独立窗口可估后生成。

每张图的机器输入为 per-subject long table，图目录实际生成后补中文 `README.md`；当前不显著结果照实进入同一接口。

## 11. 全局停止条件

只保留四类：

1. formal/sealed 分区被读；
2. normalization、baseline、checkpoint selection 或 event decoder 使用评价期未来信息；
3. target 跨 session、真实缺口、seizure 或 split；
4. 同一输入/配置得到不一致 evaluator 结果或并发污染产物。

单患者阴性、mark state 失败、2 h 不可估、H2b/H3 阴性都不是全局停止条件。

## 12. 完整执行后补充的 H3 数值可采信规则（2026-09-04）

该条是最终质检发现低独立块 ridge 外推后加入的工程可采信规则，不回写为事前注册假设，也不改变任何已运行的科学 arm。

- 对 `M1_burden_feedback`，分别计算 INNER 与 SELECTION MSE 相对嵌套父模型 `M0_common_drive` 的比值；
- 对 `M2_mark_feedback`，相对 `M1_burden_feedback` 计算同样比值；
- 任一分区 MSE 非有限或大于父模型 4 倍，该新增 feedback contrast 记为 `UNSTABLE/NOT_ADMISSIBLE`；
- M1 不稳定时，依赖该父模型的 M2 contrast 同样不可采信；
- 规则不读取 effect gain 的正负，因此偶然有利和偶然不利的发散拟合都会被排除；
- 所有原始 MSE、raw contrast 和失败原因仍进入机器长表，不删除结果。

该规则只阻止把数值发散当作生物效应，不把低效力或普通阴性改写为不可估。

## 13. 审阅修正（2026-09-04，代码审阅后追加）

本节记录完整执行报告发布后的代码审阅结论。它们改的是实现与工程规则，不改任何事前注册假设；带 P0 的一条触发 §11 全局停止条件 2，整条链在独立目录 `/data/hfosp_group_event_state_v0_3_5_causal/` 重跑。

1. **（P0）`q(t)` 的"session position"实现为 `(t − 段起点)/(段终点 − 段起点)`，用了覆盖段的结束时刻。** 目标覆盖段在发作排除处切开，其结束时刻多数正好等于下一次发作起点（E548 27/42、E922 21/29、E1125 14/21、E1146 13/27、E1096 9/23、E384 8/16、E253 7/21、E583 2/7 段）。该特征因此等价于"离下一次发作/断录还有多远"，违反 §3.1 "所有动态项只使用 anchor 之前的信息"。修正：`segment_elapsed_over_8h = min(t − 段起点, 8h) / 8h`——因果、有界，且尺度取自本 spec 已登记的时间尺度库中最长的常数（8 h），不是按结果调出来的；H3 `_common_time_features` 的两个段位置项同样改法。（过程中曾用 `log1p(t − 段起点)`，虽因果但标准化后左尾过重、破坏多位患者的拟合，已弃用并归档。）原始产物保留为 archive 证据，不得引用其依赖 q(t) 的数字。

泄漏影响已量化：对未来事件数预测很小（置零对照 5 min 中位 +0.1516 → +0.1484），对发作风险层明显（30 min 负荷增益最多下降 58%）。另新增：风险层在"完整随访观察点为零"时记 `NOT_ESTIMABLE`，因为此时合格样本全部由结局挑出。
2. **（P1）§8 block-shift 对照的同锚点约束。** 错时臂只能在有远距离供体的锚点上定义；正确时刻臂必须在同一批锚点上打分（新增 `correct_state_on_shift_support` / `mean_on_shift_support` / `q_plus_mark_state_on_shift_support` / `rate_dynamic_on_shift_support` 字段）。`correct_time_gain_over_shift` 一律定义为"错时臂 − 同支持集上的正确时刻臂"。
3. **（P1）§12 可采信规则改为双向。** 任一分区中子模型与嵌套父模型的 MSE 比值超出 [1/4, 4] 即不可采信；父模型发散而子模型有界（6 h 尺度 +20/+87 的"增益"）同样排除。规则仍不读取增益符号。
