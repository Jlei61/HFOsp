# Group-Event State v0.3：Marked point process 状态骨干与双任务迁移

**日期：** 2026-09-02  
**状态：** `REVISED_SCIENTIFIC_SPEC_AND_EXECUTION_PLAN`  
**版本关系：** v0.2 的 96 维端到端多任务网络降为 `P_local_multitask` 基线；v0.3 是新的主架构。旧结果不混入新模型选择，正式/封存分区继续关闭。

## 0. 核心问题

我们要从连续群体间期事件中学习一个跨事件、按真实时间演化的状态，并检验：

1. 它是否预测未来 IED 何时出现以及出现后如何在植入网络内传播；
2. 它是否在不看发作标签的情况下学成，并能冻结后跨任务预测发作风险；
3. 控制共同状态后，IED 是否仍需要一条反馈边来解释未来状态。

群体 IED 是从原始 SEEG 中提取的高信息病理 token，不只是旧 rank。事件内部波形、多频带能量和毫秒传播仍被保留；连续背景 SEEG 是独立辅助观测轴。

## 1. 生成模型：必须同时学习“发生”和“未发生”

仅在事件到来时更新状态，会让模型退化成复杂 IEI/近期计数编码器。v0.3 因此采用 marked temporal point process：

\[
\log p(\{t_e,M_e\})=
\sum_e\log\lambda_{\mathrm{IED}}(t_e\mid S(t_e^-),c(t_e))
-\int_{\mathcal O}\lambda_{\mathrm{IED}}(u\mid S(u),c(u))\,du
+\sum_e\log p(M_e\mid S(t_e^-),G_p(t_e)).
\]

- \(\lambda_{\mathrm{IED}}\) 预测事件何时发生；
- 积分项使有效记录中的“这段时间没有事件”也成为证据；
- \(M_e\) 是 event mark：tied groups、observed extent、contact identity、waveform 和 multiband expression；
- \(G_p(t)\) 是患者在该时刻实际可用的 implanted-contact scaffold；
- \(c(t)\) 是显式时钟、记录和临床协变量；
- \(\mathcal O\) 只包含真正有效的观测时间。

### 1.1 有效观测 mask

定义 \(r(t)\in\{0,1\}\)。只有 \(r(t)=1\) 的时间进入 timing/survival likelihood。

记录中断、坏段、检测器不可用、montage 不可解释期间：

- 真实时间仍推动 autonomous state flow；
- 不计入“没有 IED”的证据；
- 不产生虚构 event update；
- 长缺口后记录 observation age/state confidence，session 或长 gap 按合同 reset。

### 1.2 第一版数值实现

群体事件约每几秒一次，1 分钟二分类 hazard 会接近饱和，因此 primary 不使用“每分钟有/无事件”。

第一版采用有效观测小格上的 piecewise-constant intensity/count likelihood：

\[
N_j\sim\operatorname{Poisson}(\lambda_j\Delta_j),\qquad r_j=1,
\]

其中无事件小格的 \(N_j=0\) 自动提供 survival evidence，同一小格多个事件也不会丢失。默认约 1 秒网格；若 TRAIN 上占用率过高，再缩到 250–500 ms。步长只由 TRAIN 决定并冻结。

## 2. 状态骨干

### 2.1 两种状态不再混为一谈

- **within-event fast hidden：** 旧 contact decoder 内部 32 维 GRU hidden；每次新事件重置，只编码当前 prefix。
- **cross-event state \(S(t)\)：** 唯一跨事件保存的状态；负责真实时间上的历史记忆。

v0.3 不再使用 `64 fast + 32 slow = 96` 作为主状态。

### 2.2 真实时间 flow

保留真实时间衰减，但不让每个维度自由学习一个容易互相替代的 \(\tau\)。pilot 使用固定 log-spaced bank：

\[
\tau\in\{2,10,30,120,720\}\ \mathrm{min},
\]

每个尺度 4 个通道，共 20 个 nominal dimensions：

\[
S_e^-=\mu+\exp(-\Delta t_e/\tau)\odot(S_{e-1}^+-\mu).
\]

事件完整 waveform window 结束后才更新：

\[
S_e^+=S_e^-+U_\theta(S_e^-,X_e,B_e).
\]

这里的 update 首先解释为 observer 获得新信息后的状态估计更新，不自动解释为 IED 对生理系统的因果推动。

20 维只是容量设置。报告状态协方差谱、participation ratio、adapter effective rank 和跨 seed 子空间一致性，不给单个 latent coordinate 生理命名。

### 2.3 事件 token

每次群体事件输入：

- 有效触点上的参与集合和 tied groups；
- 精确 delay/centroid lag；
- bipolar/CAR 后的 event-core waveform embedding；
- ripple/fast-ripple 等频带能量、峰时和跨频带 lag；
- observed extent、spatial dispersion；
- detector confidence、reference、坏道和 coverage；
- 真实 \(\Delta t\)。

multiband/delay 可以进入 event encoder，但第一版不把所有端点都变成主训练头；它们主要作为 held-out future probe 检查状态是否真正保留这些信息。

### 2.4 背景 SEEG 是独立实验轴

预定义三臂：

1. `event_only`：primary；
2. `background_only`：独立比较；
3. `event_plus_background`：融合加强证据。

背景窗口固定按真实时间采样，在 anchor 前结束，使用 causal buffer、TRAIN-only normalization 和 missingness mask。是否有背景数据本身不得成为患者或记录块标识。

## 3. Contact grammar：先校准，再冻结

### 3.1 旧 checkpoint 的角色

现有 `FullHistorySequenceGRU` 有 34 人 × 3 seeds = 102 个 checkpoint，学习了事件内部 prefix → next contact/STOP 的 grammar。它们可用于：

- 初始化新 grammar；
- 验证旧语法可复现；
- 作为 `P_transductive_pretrained_grammar` 辅助分析。

它们不能直接作为 primary，因为旧参数化和新 likelihood 不一致，而且任何看过当前 outer-test 时段统计的 checkpoint 都会形成 transductive 风险。

### 3.2 Primary grammar-v0.3

对每个 outer chronological split：

1. 只用 outer TRAIN 初始化/训练 contact grammar；
2. 所有归一化、contact statistics、坏道支持集和校准只看 outer TRAIN；
3. inner validation 选择 epoch、正则和 adapter capacity；
4. outer TEST 只评分一次。

旧 checkpoint 只作 TRAIN 内初始化。若旧 package 与该 split 不兼容，则按同架构在 outer TRAIN 重训，不得静默读取 test。

### 3.3 新 tied-group likelihood

旧 logits 不能直接改名为新的概率量。应先在无状态条件下得到校准合格的 `grammar-v0.3`，再冻结 grammar 并训练状态。

每个 prefix step：

1. size head 输出下一 tied-group 大小 \(K\)；\(K=0\) 唯一表示 observed STOP；
2. 给定 \(K>0\)，contact 权重使用 exact fixed-cardinality conditional-Bernoulli likelihood：

\[
p(A\mid K)=\frac{\prod_{i\in A}w_i}{e_K(w_{\mathcal C_e})},\qquad |A|=K.
\]

它在“给定大小后除 cardinality 约束外条件独立”的模型内精确归一化、无序、无放回；不宣称是任意交互集合分布。

STOP 不再另建第二个 Bernoulli，避免重复计数。

### 3.4 observed network 边界

\(\mathcal C_e\) 只包含事件时刻有效、可解释的 implanted contacts。坏道从 likelihood 分母中移除；montage/reference 或有效触点集合变化时使用 \(G_p(t)\)。

报告固定使用：

- observed recruitment extent；
- observed STOP among available contacts；
- implanted-network propagation grammar。

H2a 主要空间端点是：在相同 prefix、group size 和 coverage 下，状态是否改变后续 contact identity，而不只重复预测 extent。

tie tolerance 在 pilot 上做小范围敏感性；它是测量分辨率检查，不是全队列 gate。

## 4. 状态如何进入冻结 grammar

Primary adapter 使用 LayerNorm 后的状态、低秩小范数映射和显式 gate：

\[
h_{e,0}=h_{\mathrm{base}}+\alpha_h A_h\widetilde S_e^-,
\]

\[
q_{e,k}=q_{\mathrm{base}}(h_{e,k})+\alpha_q A_q\widetilde S_e^-,
\]

\[
\ell_{K=0}=\ell_{K=0,\mathrm{base}}+
\alpha_{\mathrm{stop}}a_{\mathrm{stop}}^\top\widetilde S_e^-.
\]

- grammar 参数 `requires_grad_(False)` 并进入 `eval()`；
- 不使用 `no_grad()` 或 `detach()` 包住 decoder；
- gradient 可经 adapter 回到状态骨干；
- gate/adapter 从零或近零开始，使初始模型等于无状态 grammar。

完整 adapter 的增益不能自动解释成三个生理机制。`h-only/q-only/stop-only` 只在 pilot 做路径定位，不作扩大队列的 gate。

最重要的容量对照是：用完全相同的 adapter，把 \(S(t)\) 换成显式多尺度历史 \(H(t)\)。只有 learned state 在相同读出容量下超过 \(H(t)\)，才说明它包含近期统计以外的信息。

## 5. 间期训练目标

第一版主损失保持集中：

\[
\mathcal L_{\mathrm{state}}=
\mathcal L_{\mathrm{IED\ timing/survival}}
+\beta_1\mathcal L_{\mathrm{group\ size/STOP}}
+\beta_2\mathcal L_{\mathrm{contact\ identity}\mid K}.
\]

- timing/survival 训练模型使用完整有效时间，而不只是事件行；
- mark loss 使用事件发生前的 \(S(t_e^-)\)；
- event waveform 只能在事件结束后更新下一状态；
- delay、multiband、future count 作为 open-loop probes 或小权重 sensitivity，不再同时成为承重主头；
- 不加入 seizure loss、waveform reconstruction 或 latent consistency 来定义 primary interictal state。

## 6. 慢状态的功能评估

### 6.1 Rolling 与 open-loop 分开

- **Rolling filtered forecast：** 允许读取目标窗中新事件，评价在线更新性能。
- **Open-loop forecast：** 从 anchor \(S(t)\) 出发，只运行 autonomous flow，不读取目标窗内任何新事件，评价 anchor 状态是否已经包含远期信息。

对 5、30、120 分钟同时评分：

1. future event occurrence/count；
2. conditional contact grammar；
3. observed extent/STOP；
4. held-out delay/multiband summary。

不能只在未来恰好发生事件的样本上报告 mark；timing/count 与 conditional mark 必须联合呈现。

### 6.2 主要比较

同一批固定物理时间 anchors 上比较：

1. clock/current-event baseline；
2. multiscale interpretable history \(H(t)\)；
3. `event_only S(t)`；
4. `background_only S(t)`；
5. `event_plus_background S(t)`；
6. block-circular shifted \(S(t)\)。

matched wrong-time donor 只作辅助，避免把真正状态通过过度 matching 消掉。

## 7. Seizure-risk decoder

### 7.1 Primary 是冻结跨任务迁移

1. 只用间期 timing + grammar 训练整个状态系统；
2. 冻结状态骨干；
3. 训练低容量 seizure hazard decoder；
4. 比较 baseline vs baseline+\(S(t)\)。

联合 seizure fine-tune 只能称 supervised extension，不与 primary cross-task transfer 合并。

### 7.2 Hazard 而非五个独立 sigmoid

使用离散 survival intervals：

\[
[0,5],\ (5,15],\ (15,30],\ (30,60],\ (60,120]\ \mathrm{min},
\]

累计风险从条件 hazard 计算，天然满足时间单调性。primary horizon 固定为 30 分钟，其他 horizon 描绘风险曲线。

baseline 至少包括：

- 24 h clock phase、住院日和 session position；
- sleep/wake 或冻结背景状态 proxy；
- time since last IED；
- 1/5/15/30/60/120 min IED rate、recent extent 和 multiband summaries；
- time since last seizure、postictal/refractory 和 seizure cluster；
- medication/stimulation 信息（若数据可得）；
- coverage、有效触点数和 recording block。

住院 SEEG 不强行拟合 20–30 天 multidien rhythm。

### 7.3 评估

- chronological forward split，边界 purge 至少 120 min 加特征支持窗；
- patient、recording block、seizure cluster 为推断层级，anchor 不是独立样本；
- seeds 先在患者内合并；
- 报告 held-out log score、Brier、calibration、seizure sensitivity、time in warning 和 false alarms/day；
- 可选 secondary decoder 预测 early ictal spatial field/path。

冻结 \(S(t)\) 提高风险预测只说明间期表示含有可迁移信息，不单独证明同一个真实生理变量共同生成 IED 和 seizure。

## 8. H1–H3 的新判据

### H1：跨真实时间 predictive state

需要：

- marked timing + grammar 在 held-out data 有增量；
- learned \(S(t)\) 超过 capacity-matched multiscale \(H(t)\)；
- open-loop 30/120 min 仍有信息；
- correct-time 优于 block-shift。

只改善下一事件，称短程 filter。

### H2a：状态调制 implanted-network event grammar

主要看：

- identity conditional on group size；
- later continuation / same-prefix continuation；
- observed extent/STOP 单独报告。

只改善 STOP 时，只称 observed extent state。

### H2b：跨任务发作易感信息

冻结 interictal-only \(S(t)\) 在强 baseline 外改善 held-out seizure hazard。early ictal field/path 是更强 secondary 证据。

### H3：IED feedback

普通 post-event update 仍按 observer update 解释。H3 后续独立比较：

- common-drive/no-feedback；
- count-rate feedback；
- mark-specific feedback。

不以 v0.3 adapter 或 hidden jump 直接证明生理反馈。

## 9. 执行计划

### Phase 0：split 与 exposure 底座

- 建 chronological outer/inner split；
- 建有效观测 exposure mask、time-varying contact support 和 seizure/gap boundary；
- 用 TRAIN 选择 timing 网格；
- 锁定 event onset、feature-window end、\(S^-\)/\(S^+\) 语义。

### Phase 1：grammar-v0.3

- 旧 checkpoint 仅作 TRAIN 内初始化；
- 在 outer TRAIN 下拟合 size/STOP 与 conditional set calibration；
- held-out 检查 grammar calibration；
- 冻结成每患者 inference bundle；
- 旧 transductive checkpoint 仅作 supportive arm。

### Phase 2：marked state pilot

- 实现 count/intensity survival likelihood；
- 实现 event-only 20 维 fixed-timescale state；
- 实现 gated low-rank adapter；
- 3 位预定义患者 × 3 seeds；
- 检查 state 相对同容量 \(H(t)\)、open-loop horizon 和状态有效秩。

### Phase 3：H1/H2a development

- 扩到固定 6 人；
- fixed-time 5/30/120 min rolling/open-loop；
- 拆 timing、size/observed STOP、identity|K、continuation；
- 加 background-only 与 fused 两臂；
- patient/block-first 汇总，不以普通阴性停止其他 endpoint。

### Phase 4：H2b frozen transfer

- 冻结 interictal state；
- baseline vs baseline+state discrete seizure hazard；
- 30 min primary，完整风险曲线 secondary；
- seizure/cluster/patient 层级评估；
- 数据支持时增加 early ictal field/path。

### Phase 5：H3 独立模型比较

在 H1/H2b 之外单独实现 common-drive、count feedback、mark feedback；不把 observer update 当作 H3。

## 10. 资源与交付

- GPU worker 以患者/seed 为单位；先测最大患者显存，再在不 OOM 下并行；
- `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`；
- AMP 用于 event encoder/adapter，state flow、intensity integral 和 likelihood reduction 用 FP32；
- `nohup`/`setsid` 或 tmux 持久运行，原子 manifest/checkpoint，可断点续跑；
- OOM 只降 microbatch/并发，不改科学合同；
- primary 3 seeds；接近种子噪声时再补 2 seeds；
- 每阶段输出白话报告、技术报告、机器 JSON 和 checkpoint registry。

## 11. 最小验收

工程上必须确认：

- invalid exposure 不进入 survival；
- outer TEST 未参与 grammar/state 训练或选择；
- adapter 关闭时复现 frozen grammar；
- frozen grammar 权重不更新，但 state/adapter 有梯度；
- \(K=0\) 只计算一次 STOP；
- coverage mask 真正改变 likelihood support；
- event feature window 不泄漏到 pre-event state。

科学上下一轮只需回答三件事：

1. 加入无事件 survival evidence 后，learned state 是否超过同容量多尺度历史？
2. 同一个 anchor state 是否能 open-loop 预测 30–120 分钟的 timing 与 conditional event grammar？
3. 完全冻结的 interictal state 是否在强 baseline 外改善 30 分钟 seizure hazard？

这三项分别决定：它是否超出 IEI 编码器、是否真有慢时间跨度、是否具有跨任务临床意义。
