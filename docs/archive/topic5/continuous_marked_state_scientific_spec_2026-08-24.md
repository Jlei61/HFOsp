# Epi-PRSSM-Raw v1 科学规范：连续观测的 marked state-space model

**版本：** v1.4（development；双时钟、长序列 warm-start 与 held-out preictal inference）
**日期：** 2026-08-24
**适用范围：** 34 位患者冻结队列的开发分区；正式检验分区保持封存。
**核心论证：** 在患者长序列中，检验“事件外连续 SEEG 所推断的唯一持续状态，是否在显式 renewal/burst 历史之外共同预测下一次 IED 的时刻与空间 mark；随后只增加 IED innovation→generator 的固定记忆路径，判断多个近期 IED 是否塑造未来状态，并区分这种记忆依赖真实经过时间还是仅依赖累积事件次数”。

## 1. 与原始科学问题的对应

- **H1：** 存在按真实时间持续演化、在显式 IED 历史之外预测未来 IED 的状态。
- **H2a：** 该状态改变下一次 IED 的 participation、order、STOP 与相同前缀后的 continuation。
- **H2b：** 冻结后的状态轨迹与间期到发作的转换相关；全部可用发作为主分析，高 IED 可观测性层为敏感性。
- **H3a：** 多个近期 IED 的 innovation 改变未来事件发生倾向或空间 repertoire；真实时间尺度与事件次数尺度必须分开识别。
- **H3b：** T2 特有的 exposure-sensitive 状态分量与后续发作模式相关。

H1–H3 是递进问题，不是 AND gate。timing-only、mark-only、raw-negative、T2-negative 都是可以独立收口的探索结果。

## 2. 术语账本

| 固定术语 | 符号 | 唯一含义 | 禁止混称 |
|---|---:|---|---|
| continuous background observation | \(O_k\) | anchor 前、IED core 被遮蔽的连续 SEEG 与显式低维特征 | state |
| observation embedding | \(o_k\) | encoder 对 \(O_k\) 的 64 维读出 | latent state |
| explicit event history | \(r(t)\) | 人工规定、可审计的 renewal/burst/session covariates | learned RNN memory |
| persistent generative state | \(z(t)\) | 全记录唯一一条可学习、可自主 rollout 的 8–16 维轨迹 | raw embedding、IEI memory |
| IED innovation | \(\eta_e\) | 实际 event load 减去 cross-fitted expected load | raw load |
| exposure state | \(s_e^{time}\) / \(s_e^{count}\) | 由 \(\eta_e\) 驱动、分别按真实时间或事件次数衰减的固定记忆 | unrestricted resource RNN |
| T1 | — | observation-only generator，IED 不直接推动 \(z\) | no-event model |
| T2 | — | 仅比 T1 多一条 \(u_\tau\to z\) generator edge | 更大任意网络 |

全文不得用 free event-history RNN/GRU 作为 \(r(t)\)，也不得把 encoder 输出直接命名为 state。

## 3. 数据与封条

1. 继承 `results/epi_prssm/v0_1/manifests/SPLIT_MANIFEST.json` 的患者级 chronological boundary。
2. 训练、标准化、encoder 预训练和超参数选择只可读取 `train_end_epoch` 以前数据；validation 只读 `train_end_epoch <= t < dev_end_epoch`；`t >= dev_end_epoch` 拒绝加载。
3. 后段 held-out 发作前 raw 即使不带 seizure label，也不得进入训练或自监督预训练。
4. marked-event stream 使用已修复 phantom-rank 的 participation mask、显式 tied group identity 与绝对时刻；生存积分只覆盖真实 recorded intervals。
5. continuous background branch 以 30 s 为 anchor；每个已知 IED 的主波形 \(\pm1\) s 主分析中置为 missing，并同时传递 mask。未经遮蔽 raw 仅作敏感性分析。
6. ictal interval 不进入 T1/T2 训练。preictal 的训练规则以 chronological training interval 为准；任何 H2b 评价发作及其前窗必须位于未参与训练的后段开发区间。
7. 第 3/6 条限制的是参数学习，不是评估时可观测性。H2b/H3b 评价时，冻结 observer 必须按真实时间因果读取每个 lead time 之前的 held-out preictal raw/background 与 IED history 来推断当时 (z)；禁止读取该 lead 之后或 onset 之后数据。另报在更早 anchor 后关闭 observation correction 的 open-loop sensitivity，用于区分“当下滤波读出”与“可自主推进的风险状态”。

## 4. R0.2 的定位与收口

R0.2 只叫 **Raw-SEEG encoder architecture triage**。它预测未来频谱，不直接检验 H1–H3。

截至 2026-08-24，三位预先指定患者上：

- Conformer 在 3/3 患者、全部四个 horizon 上均输给等容量 wide Transformer；
- 两类深度模型在 1/5/10 min 普遍输给显式 spectral feature AR；
- identity dynamics 与 learned damped rotation 的差异小且跨 horizon 不稳定；
- shuffled-target 臂落回患者均值附近，未提示未来泄漏。

因此不扩到 34 人。正在收尾的单个 identity 作业可保存；其余 Wave C 不再启动。该阴性只否定“当前频谱 forecasting 任务中的 Conformer 优势”，不否定 raw SEEG 对 IED 的信息。

## 5. B0–B3 信息桥接实验

### 5.1 问题

连续背景观测是否在显式 IED 历史之外，增加对下一次 IED exact time 与完整 spatial mark 的信息？Bridge 是必要性检验，不是 state discovery。

### 5.2 四臂

- **B0 history-only：** 仅 \(r(t)\)。
- **B1 spectral：** \(r(t)\) + masked-background 的显式 spectral/variance/autocorrelation features。
- **B2 raw：** \(r(t)\) + raw observation embedding。R0.2 只提供 encoder family/初始化；该 embedding 必须在本版 mask 与 joint IED timing+mark 目标下重新验收，旧 frequency-forecast latent 不得直接命名为 state。
- **B3 complementary：** \(r(t)\) + spectral + raw。

四臂使用完全相同的 exact-time head、mark head、训练区间、样本和容量控制。第一轮允许用小型 raw branch 做探索；paper-level raw claim 必须换成锁定的 wide Transformer family 并重跑。

### 5.3 输出与判定

统一报告 joint NLL，并拆出 timing NLL 和 mark NLL；另报 participation/order/STOP。Bridge 只归入：

1. no increment；
2. spectral-only；
3. raw beyond spectral；
4. complementary。

任何阳性仍可能来自睡眠、昼夜、session、阻抗或亚阈值事件；这些是解释边界，不是额外停止闸门。

## 6. 两类输入，但只有一个学习状态

### 6.1 连续观测

每 30 s 形成：

\[
O_k=[X_{t_k-30s:t_k}, S_k, M_k, \mathrm{coordinates}],\qquad
o_k=E_\phi(O_k)\in\mathbb R^{64}.
\]

\(X\) 为预处理 bipolar SEEG，\(S_k\) 为显式低维频谱、方差和自相关，\(M_k\) 为坏道、缺口、伪迹和 IED-core mask。\(o_k\) 只是一条观测。

### 6.2 显式事件历史

第一版固定：

\[
r(t)=[\log(1+\mathrm{time\ since\ last\ IED}),x_{30s}(t),x_{2m}(t),L_{last},\bar m_{recent},\mathrm{time/session\ covariates}].
\]

可增加的量必须是可解释、预先计算的 covariate；不得增加自由 recurrent state。

## 7. T1：observation-only persistent generator

\[
z_k^-=\mu+\exp(\Delta t_kK)(z_{k-1}^+-\mu),
\quad K=\Omega-Q,
\quad \Omega^\top=-\Omega,
\quad Q=\mathrm{diag}(\mathrm{softplus}(q)).
\]

观测只做 residual correction：

\[
c_k=\tanh W_c[o_k,z_k^-],\quad
g_k=\sigma W_g[o_k,z_k^-],\quad
z_k^+=(1-g_k)\odot z_k^-+g_k\odot c_k.
\]

这里使用有界的 GRU-style measurement correction。探索 smoke 已证明无界 residual addition 会在高 IED 频率患者中退化成事件计数器，即使 generator 本身稳定也会在 2,048 个事件内累积到 \(|z|>100\)。
observer 初始化为 identity/no-correction：candidate 权重与偏置为 0，gate 权重为 0 且偏置为 -4；这样状态变化必须由训练目标学出，而不是由随机初始化在密集事件序列上先饱和。

全记录只有一条 forward trajectory。TBPTT 可以 detach gradient，但不得重置 forward state。T1 中事件更新 \(r(t)\)，不直接 jump \(z(t)\)。

Recurrent timeline 必须包含 development 区间内每一个相邻 IED transition。若某一时刻的 raw/spectral observation 因缺口、伪迹或缓存不可用，则保留该事件与真实时间，只令 observation mask 为 0，由 generator 继续传播；不得像 Bridge 一样删除该行。每位患者同时报告 observation coverage，防止把“状态模型阴性”与“多数时刻根本没有观测”混为一谈。

## 8. timing 与 mark 的统一事件模型

arrival baseline 先独立拟合并冻结：

\[
\lambda(t)=\mathrm{softplus}[f_{base}(r(t))+w_\lambda^\top z(t)].
\]

复用当前 contact-RNN，状态只经低秩 adapter 进入：

\[
h^{contact}_{0,e}=h^{base}_{0,e}+U_z z(t_e^-).
\]

同一 tied group 必须按显式 `event_group_ids` 作为无序、无放回 subset 计算精确条件概率；不得把 tie 内成员伪造顺序，也不得用 `sum(logit) - m*logsumexp(logit)` 的有放回近似冒充 subset likelihood。STOP、participation、order/same-prefix continuation 分项持久化。

唯一训练目标为 joint marked point-process NLL：

\[
\mathcal L_{event}=-\sum_e\log\lambda(t_e)
+\int_{\mathcal T_{recorded}}\lambda(u)du
-\sum_e\log p(m_e\mid t_e,\mathcal H_{t_e}).
\]

不加入频谱重建、KL、contrastive 或 seizure loss；timing 与 mark 可分项报告，但不任意调 loss 权重。

## 9. T2：IED innovation-driven shaping

训练集内 cross-fit：

\[
\widehat L_e=\mathbb E[L_e\mid z(t_e^-),r(t_e),source,time/session],
\qquad \eta_e=L_e-\widehat L_e.
\]

固定时间尺度 exposure：

\[
\frac{du_\tau}{dt}=-\frac{u_\tau}{\tau},
\qquad u_\tau(t_e^+)=u_\tau(t_e^-)+\eta_e,
\]

\[
\frac{dz}{dt}=K(z-\mu)+B_\tau u_\tau.
\]

上式保留为连续 forcing 敏感性；承重的 clock comparison 必须使用相同作用算子。令：

\[
s_e^{time}=\exp[-(t_e-t_{e-1})/\tau]s_{e-1}^{time}+\eta_e,
\qquad
s_e^{count}=\exp[-1/N]s_{e-1}^{count}+\eta_e,
\]

并让两臂共享完全相同的 event-linked generator update：

\[
z(t_e^+)=z(t_e^-)+B_s s_e.
\]

其中 \(N=\tau/\widetilde{\Delta t}_{TRAIN}\)，\(\widetilde{\Delta t}_{TRAIN}\) 只用该患者 TRAIN 内相邻 IED 间隔中位数估计。这样 time/count 两臂有相同典型记忆长度、相同 \(B_s\) rank 与参数量，唯一差别是 exposure 是否读取实际 \(\Delta t\)。count arm 是“经过多少次 IED”而不是“经过多少分钟”的主替代解释。连续 \(u_\tau\to z\) forcing 只能在 clock 主比较收口后作为机制敏感性，不能与离散 count arm 直接比输赢。

T1 即 \(B_\tau=0\)。H3-S0 筛查先运行 \(\tau\to0\) 的 **current-event limit**，再独立运行秒级 burst controls、1 min fast control 和分钟至小时尺度；1440 min 只在记录覆盖足够时追加。current-event limit 在数值上只保留当前 IED innovation，不携带先前事件。每个 T2-real 都有同容量 time-shift/state-matched innovation placebo。

进入真正 T2 generator 时，不用数值极小的 \(\tau\) 逼近瞬时效应，而设置同参数量 event-jump control：

\[
z(t_e^+)=z(t_e^-)+B_0\eta_e.
\]

它等于上式取 \(s_e=\eta_e\)：只允许当前 IED 推动状态，不 carry exposure；cumulative time/count T2 与它共享 observer、decoder、\(K\)、初始化、event-linked update 和 forcing rank。participation innovation 使用固定低秩映射，current-event 与 cumulative arms 的 rank/参数量严格相同。

由于本队列事件间隔可短至数秒，1 min 不能充当“单个 IED”对照。判读分两层：

1. distributed exposure 稳定胜过 current-event limit，才支持 **多个近期 IED 的累积记忆**；
2. 真实时间臂进一步稳定胜过匹配典型记忆长度的事件次数臂，才支持已识别的 **physical-time clock**。

固定事件数开发曲线在看到完整结果前按以下方式冻结进入 R2：**50/100/200 events 为主窗口集合**，25 events 仅作 fast-history control，400 events 仅作 long-memory sensitivity。五档都完整报告，但不根据开发网格的最低 NLL 为每位患者或每类 exposure 另挑“最佳 N”；R2 的患者级主统计先在 50/100/200 三档分别给出，再报告跨三档方向是否一致，不把三档或两个 exposure source 合并成一个事后最优检验。

为避免从“分钟时钟”过度纠偏成“只允许事件次数”，R2 还预先保留 **10/60/360 min 的固定 physical-time sensitivity**。每个固定分钟臂必须同时运行患者内 rate-matched count arm，令 (N_i=\tau/\widetilde{\Delta t}_{TRAIN,i})；因此一组检查跨患者共同的事件数窗口，另一组检查跨患者共同的物理时间窗口，二者都不按当前 validation 结果筛选。固定分钟层为机制探索，不与 50/100/200-event 主集合合并成单一 p 值，也不阻断主集合。

截至本轮开发结果，3/5/10/20/30/60 min 的完整配对网格中，真实时间臂没有在预先拆开的 mark、participation、rank、STOP 或 timing 端点上形成稳定优势；对应患者中位记忆长度约 47/79/158/316/474/947 次 IED。唯一例外是 load 30 min 的 aggregate joint contrast（24/34，未校正 p=0.024），但它在 real-vs-history 直接比较及任何拆分端点上均不成立，不能承载 physical clock 结论。事件次数臂本身在约 47–474 次窗口仍多处胜过 current-event limit，尤其体现在 STOP，participation-exposure 还体现在 rank。因此当前只允许称 **recent-event-count accumulation / cumulative IED termination–extent memory**，不得称已识别的分钟级生理时间常数。

固定 25/50/100/200/400-event 网格随后用同一冻结 producer 完成 count 340 + rate-matched physical 340 cells；340/340 样本和 history endpoint 逐位一致，680 个产物均晚于 producer lock，早期 219 个可比产物重跑后 219/219 全字段完全一致。count arm 的 mark 同时胜 history 与 delayed placebo 的患者数，load 为 24/22/25/24/21，participation 为 25/24/21/20/18；相对 current-event limit 的 mark 方向在 load 为 26/25/29/26/24，participation 为 27/28/25/22/20。阳性主要由 STOP 承担；timing 无方向。actual-time arm 在 50/100/200-event 主集合中没有稳定胜 matched count，短窗口反而多为 count 更好。固定-N 对应的患者中位 physical labels 为 1.58/3.17/6.34/12.67/25.35 min（IQR 随 N 从 0.72–2.72 放大到 11.57–43.52 min），再次说明不能把单一分钟值当队列真值。mark 方向主要由 validation transitions ≥1000 的 24 人承担，低支持 10 人不稳定；因此它定位的是 **约 25–200 events 的支持度依赖宽记忆带**，而不是人人共享的单一点。

## 10. H3 的承重实验：correction-off post-event challenge

对每个 anchor event：

1. T1、T2-real、T2-placebo 从同一 pre-event \(z(t_e^-)\) 和同一真实时间线开始；
2. event 后关闭全部未来 raw/spectral observation correction；
3. 保留真实未来事件输入，以区分 generator forcing，而不是比较虚构时间线；
4. 在未来 5/10/20 个事件分别比较 timing 与 mark NLL。

train/validation 只是参数学习边界，不是生理轨迹边界。真正 R2 在进入 validation 前，必须用所有更早、真实可记录的 TRAIN 事件和 background observations 对 (z,s_e,r(t)) 做无梯度 causal warm-start；只有真实 recording gap/session boundary 才按合同 reset。不得因切分而把 persistent state 或 exposure 清零，也不得用 validation target 反向更新参数。当前 H3-S0 线性筛查在 split 边界统一 reset，因此它适合作为两时钟同条件的保守筛查，不替代该长序列验收。

只有 T2-real 在 correction-off 下稳定优于同容量 placebo，才支持 IED-driven generator update；distributed 臂进一步胜过同参数量 event-jump control，才支持累积暴露而非单事件解释；真实时间 distributed 臂再胜事件次数 distributed 臂，才能定位为物理时间尺度。若只改善 timing，称 event-propensity shaping；若 mark 增量仅由 STOP 承担，称 event-termination/extent shaping；只有 participation/order 或 same-prefix continuation 明确改善，才称 functional repertoire shaping。不得用 aggregate mark 掩盖端点来自 STOP。

## 11. H1/H2a 的最小干预证据

- real-time interval shuffle；
- matched wrong-time state swap；
- \(z\) clamp/reset；
- correction-off rollout 5/10/20 events；
- same-prefix continuation；
- filtered 与 correction-off 结果分开。

这些是状态解释所需的最小集合，不在探索期扩展成大规模防御测试矩阵。

## 12. H2b/H3b：冻结后的发作 probe

T1/T2 训练完成后冻结全部 backbone，再用低容量患者内 probe 检验 5/15/30/60/120 min lead。全部可用发作为主层，高可观测性为敏感性；按已有 seizure mode 做探索性分层。probe 可控制 time-of-day、session、IED rate、time since seizure，但不得用 seizure loss 反向改写 \(z\)。H2b/H3b 均为观察性方向证据，不写成因果。

## 13. 允许与禁止的结论

允许按证据层级写：

- `raw-informed predictive filter`：仅 filtering 阳性；
- `autonomous predictive state estimate`：correction-off、swap、reset 均支持；
- `event-propensity shaping`：T2-real 仅改善 timing；
- `event-termination/extent shaping`：T2-real 的 mark 增量主要或仅由 STOP 承担；
- `functional repertoire shaping`：T2-real 在同容量 placebo 之外改善 participation/order 或 same-prefix continuation，而不是只有 aggregate mark/STOP。
- `recent-event-count accumulation`：distributed exposure 胜 current-event jump，但真实时间不胜事件次数时钟；此时分钟数只用于把事件次数与该患者典型 IEI 对齐，不是生理时间常数。

禁止：把 forecastability 写成生理真值；把图拓扑增量写成因果流动；把 seizure association 写成 IED 导致发作；用正式检验分区调参或挑模型。
