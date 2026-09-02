# Group-Event State v0.3：Marked point process 状态骨干与双任务迁移

**日期：** 2026-09-02  
**状态：** `V0_3_1_NESTED_PILOT_COMPLETE_DEVELOPMENT_ONLY`
**版本关系：** v0.2 的 96 维端到端多任务网络降为 `P_local_multitask` 基线；v0.3 是新的主架构。旧结果不混入新模型选择，正式/封存分区继续关闭。

**阶段性执行结果：** 3 位预定义患者 × 3 seeds 已完成。模型层 nested contract 通过，但旧全记录触点筛选仍使测量层属于 transductive development。当前模型在可比较患者中没有胜过 multiscale history，correct-time 也没有跨患者稳定胜过 block-shifted state；两位患者三 seeds 均选择第一个训练 epoch，另一位在预算边缘。因此本版验收为“仪器端到端可运行、H1/H2a 未建立且受优化/测量边界限制”，不是生物学阴性。详见同日白话版与技术版报告。

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

primary 直接在相邻真实事件时刻之间积分 intensity，并把每个有效记录段末尾的无事件尾巴纳入 survival likelihood；不使用容易饱和的一分钟二元标签。积分只跨连续有效 coverage，绝不跨记录 gap、发作或 postictal 排除段。

同时在固定物理时间 anchor 上加入 5、30、120 min future-count Poisson loss。前者训练局部 timing 与 silence，后者明确要求状态保存对未来一片事件分布有用的信息。两类 loss 分开报告，不混成 accuracy。

## 2. 状态骨干

### 2.1 两种状态不再混为一谈

- **within-event fast hidden：** 旧 contact decoder 内部 32 维 GRU hidden；每次新事件重置，只编码当前 prefix。
- **cross-event state \(S(t)\)：** 唯一跨事件保存的状态；负责真实时间上的历史记忆。

v0.3 不再使用 `64 fast + 32 slow = 96` 作为主状态。

### 2.2 真实时间 flow

保留真实时间衰减，但不让每个维度自由学习一个容易互相替代的 \(\tau\)。pilot 使用固定 slow bank：

\[
\tau\in\{5,30,120,360\}\ \mathrm{min},
\]

每个尺度 4 个通道，共 16 个 nominal dimensions：

\[
S_e^-=\mu+\exp(-\Delta t_e/\tau)\odot(S_{e-1}^+-\mu).
\]

事件完整 waveform window 结束后才更新：

\[
S_e^+=S_e^-+U_\theta(S_e^-,X_e,B_e).
\]

这里的 update 首先解释为 observer 获得新信息后的状态估计更新，不自动解释为 IED 对生理系统的因果推动。

16 维只是小型容量设置，不是 16 个自由生理时间常数。learnable \(\tau\) 只作为后续 sensitivity；主分析不给单个 latent coordinate 生理命名。

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

现有资产的准确说法是：**34 位患者的 inference bundles，每位重复 3 个优化 seeds**；不是 102 个独立患者或 102 个独立预训练任务。旧 `FullHistorySequenceGRU` 学习过事件内部 prefix → next contact/STOP grammar，可用于：

- 初始化新 grammar；
- 验证旧语法可复现；
- 作为 `P_transductive_pretrained_grammar` 辅助分析。

它们不能直接作为 primary，因为旧参数化和新 likelihood 不一致，而且任何看过当前 outer-test 时段统计的 checkpoint 都会形成 transductive 风险。

### 3.2 Primary grammar-v0.3：严格嵌套时间合同

每位患者按累计有效记录时间固定分成：

```text
0–16%  grammar fit
16–20% grammar inner-validation
20–70% interictal state training
70–80% development validation
80–100% development test（只评分一次）
```

patient offset、event/contact normalization、group-size statistics 和 calibration bias 都只能来自前 16%；grammar epoch 只由 16–20% 选择。state checkpoint 只由 70–80% 选择。最后 20% 不参与 normalization、超参数或 early stopping。

旧 checkpoint 在 primary 中**只提供网络宽度等架构超参数，不加载任何 learned weight 或 patient offset**。因此本 pilot 是“新 product-form grammar + 新 state”的探索性仪器，不宣称复现旧 scoring；旧 checkpoint parity 是独立辅助路线。

contact vocabulary 应来自不依赖未来事件的固定硬件 montage。当前缓存仍来自旧全记录 refine/packing 的触点筛选，因此本轮只能达到“模型层嵌套干净、上游测量层 transductive”的 development 级别。正式扩队列前必须用固定 montage 或 calibration-prefix 重建 vocabulary；这个限制不能靠模型内 split 消除。

### 3.3 新 tied-group likelihood

旧 logits 不能直接改名为新的概率量。应先在无状态条件下得到校准合格的 `grammar-v0.3`，再冻结 grammar 并训练状态。

每个 prefix step 只定义一个 \(K=0,\ldots,N\) categorical，并按下面方式等价分解报告：

1. 先报告 continue vs STOP，其中 \(K=0\) 唯一表示 observed STOP；
2. 若继续，再报告 \(K\mid K>0\)；
3. 给定 \(K>0\)，contact 权重使用 product-form fixed-cardinality likelihood：

\[
p(A\mid K)=\frac{\prod_{i\in A}w_i}{e_K(w_{\mathcal C_e})},\qquad |A|=K.
\]

ESP 只保证它在“给定大小后为 product-form 权重”的模型族内精确归一化、无序、无放回；不宣称是任意无序集合分布的 exact likelihood。

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
\boldsymbol\ell_{K}=\boldsymbol\ell_{K,\mathrm{base}}+
\alpha_{K}A_{K}\widetilde S_e^-.
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
+\beta_2\mathcal L_{\mathrm{contact\ identity}\mid K}
+\sum_{H\in\{5,30,120\}\mathrm{min}}\gamma_H\mathcal L_{\mathrm{future\ count},H}.
\]

- timing/survival 训练模型使用完整有效时间，而不只是事件行；
- intensity 写成患者 state-training period 平均事件率上的有符号状态调制，并约束动力学平衡点精确回到该平均率；不能让从未被训练访问的 latent 平衡点在 2 小时 open-loop 时产生任意事件率；
- pilot 以真实相邻事件区间做细粒度 survival 积分，而不是用会在本队列饱和的一分钟二元格；
- mark loss 使用事件发生前的 \(S(t_e^-)\)；
- event waveform 只能在事件结束后更新下一状态；
- future count 是小权重但预注册的慢状态训练目标；delay、multiband 仍作为 open-loop probes；
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

\[
P(T\le H)=1-\prod_{j\le H}(1-h_j).
\]

不训练五个互相独立的 binary heads。评价以每位患者的 Brier/Brier skill、survival log score 和 reliability 为主，AUROC/敏感度只作 secondary；不能把所有 grid anchors 混成一个跨患者分母。

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

- 建 16/4/50/10/20 的 nested physical-time split；
- 审计旧 patient offset、归一化、vocabulary、size prior、checkpoint selection 与 detector/template 来源；
- 建有效观测 exposure mask、time-varying contact support 和 seizure/gap boundary；
- 用 TRAIN 选择 timing 网格；
- 锁定 event onset、feature-window end、\(S^-\)/\(S^+\) 语义。

### Phase 1：grammar-v0.3

- 旧 checkpoint 只读架构超参数，不加载 learned weights；
- 在 calibration fit prefix 拟合 grammar，在 prefix tail 选 epoch；
- held-out 检查 grammar calibration；
- 冻结成每患者 inference bundle；
- 旧 transductive checkpoint 仅作 supportive arm。

### Phase 2：marked state pilot

- 实现 count/intensity survival likelihood；
- 实现 event-only 16 维 fixed slow-timescale state；
- TBPTT 同时限制最多 1024 events 和 30 min，chunk 边界 carry+detach、不 reset；每个 segment 前 5 min 只 burn-in，不计 loss；每个 epoch 从 segment boundary 重放；
- 加入固定 anchor 的 5/30/120 min future-count loss；
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
- development test 未参与 grammar/state normalization、训练或选择；
- patient calibration、state training、development validation/test 时间区间严格不重叠；
- 当前全记录触点筛选必须被机器审计标为 upstream transductive，不得伪装成完全 nested；
- adapter 关闭时复现 frozen grammar；
- frozen grammar 权重不更新，但 state/adapter 有梯度；
- \(K=0\) 只计算一次 STOP；
- continue、positive size 和 product-form subset 三项分别输出；
- TBPTT chunk 同时满足物理时间和事件数上限，边界 detach 但不 reset；
- coverage mask 真正改变 likelihood support；
- event feature window 不泄漏到 pre-event state。

科学上下一轮只需回答三件事：

1. 加入无事件 survival evidence 后，learned state 是否超过同容量多尺度历史？
2. 同一个 anchor state 是否能 open-loop 预测 30–120 分钟的 timing 与 conditional event grammar？
3. 完全冻结的 interictal state 是否在强 baseline 外改善 30 分钟 seizure hazard？

这三项分别决定：它是否超出 IEI 编码器、是否真有慢时间跨度、是否具有跨任务临床意义。
