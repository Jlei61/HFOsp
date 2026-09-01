# Group-Event State v0.2-A：Predictive-state identification（H1/H2a）

开始前完整阅读共同科学合同和工程附录。本线的任务不是证明“RNN 记住了历史”，而是直接训练并检验：仅由间期群体事件历史形成的状态，能否在真实时间上预测未来一片事件的 rate 和 network expression。

## 1. 三个核心问题

### A1：历史是否形成 future-block predictive state

同一固定时间网格比较：

1. current clock/event baseline；
2. `B_multiscale`；
3. `P_local`；
4. `P_slow`；
5. `P_slow` 的 within-session block-shift state。

主结果是相对 `B_multiscale` 的 future-block score 改善随 5/30/120 min horizon 的曲线。6 h 只在真实连续 coverage 足够时探索。

### A2：状态预测的是 rate，还是 repertoire

future block 的联合目标必须拆成：

\[
p(N_{t:t+\Delta}\mid S_t)
\]

和

\[
p(mark\mid N>0,S_t).
\]

conditional mark 至少包含 participation field、size/STOP、continuous event-embedding distribution、multiband field。TRAIN-only clusters 只作为可解释的 secondary endpoint。

### A3：same-prefix continuation

对早期前缀相似的事件，比较：

\[
p(later\ recruitment\mid prefix)
\quad\text{vs}\quad
p(later\ recruitment\mid prefix,S_t).
\]

前缀匹配至少考虑首发触点、前两个 tied groups、前 50–100 ms waveform 和早期能量范围；结局拆为是否继续传播、后续触点、STOP/extent 和后续 multiband expression。这是 H2a 最接近“状态改变病理网络走法”的主端点。

## 2. `P_slow` 训练规格

每个事件仍是一步，保留完整 waveform/multiband/participation/delay 输入。`P_slow` 与 `P_local` 使用相同 encoder 和基本容量，差别是加入 fixed-time future-block heads：

\[
\mathcal L_{slow}=\mathcal L_{local}
+\lambda_5\mathcal L_{5m}
+\lambda_{30}\mathcal L_{30m}
+\lambda_{120}\mathcal L_{120m}.
\]

首轮权重按 TRAIN 中各 loss 的初始梯度/尺度配平后冻结，不根据 development 结果调。长 horizon head 读取候选慢状态；local head 可读取 fast+slow。第一版不加 latent orthogonality、consistency 或 reconstruction loss。

fixed grid 每 5 min 一个 anchor；future targets 用 cumulative/sparse builder 即时索引，不物化大张量。所有 target 不跨 gap、split 或 seizure onset。

## 3. 主 baseline

`B_multiscale` 在 1/5/30/120 min 尺度构造 rate、time-since-last-event、size/STOP、participation、repertoire、band energy 的 EWMA，并加入 clock/session/coverage。线性 GLM 为主，小 MLP 为容量敏感性。

baseline 与 recurrent producer 必须使用相同 anchor、target、mask、normalization 和评分代码；不允许各自抽不同窗口后再相减。

## 4. 主 null 与诊断

### 4.1 承重 null

- `B_multiscale` vs `B_multiscale + state`；
- correct-time state vs within-session block circular shift，shift 严格大于 horizon。

### 4.2 敏感性

- matched wrong-time：只粗匹配 session、time-of-day、coverage、recent rate；
- event reset：1/100/1,000/full；
- physical reset：5/30/120 min/full；
- fast-only、slow-only、fast+slow 冻结读出；
- trained memoryless producer。

reset/latent 分解不承担慢状态主结论。不能把 K=100 与 full 不显著写成“100 次饱和”。

## 5. Session-preserving 训练和选择

- 按 recorded physical time 切 TRAIN/inner-validation/development-test。
- batch 并行不同 sessions；同 session 内 chunk carry state，只 detach 不 reset。
- checkpoint 由 TRAIN chronological inner-validation 的预注册组合目标选择；development-test 只评一次。
- 3 seeds 全配置；预先固定的主配置可加到 5 seeds。seed 不作患者分母。
- 先在固定 3 位长患者验收梯度、收敛和资源，再扩原 8 位语义复现，随后扩全部 development 可训练患者；不得按结果挑患者。

## 6. 执行计划

### A0：共同地基

1. 验证 causal warm-up、session carry、gap/seizure reset。
2. 实现 physical-time split、5-min grid 和稀疏 future-target builder。
3. 建立 `checkpoint_registry.json` schema 和 no-silent-fallback 校验。

### A1：三 producer

1. 构建 `B_multiscale` GLM/MLP。
2. 以修复后 session-preserving 方式重训 `P_local`。
3. 实现并训练 `P_slow`；确认 5/30/120 min heads 都有非零梯度和实际更新。
4. 全部写入 registry；旧 v0.1 checkpoint 只作 plumbing，不作承重结果。

### A2：A1/A2 主评估

在同一 fixed-grid anchor 上计算 count 与 conditional-mark score；运行 correct-time/block-shift。输出 patient-first、有效不重叠物理窗和 seed spread。

### A3：H2a continuation

在 event anchors 上运行 same-prefix continuation；前缀规则只用 TRAIN 冻结，报告每患者可匹配事件数和每个 endpoint。

### A4：少量诊断和收口

运行缩减 reset、fast/slow、memoryless、coarse matched donor；生成一张承重图和辅助图。白话报告必须说明当前找到的是 rate state、extent state 还是 repertoire state。

## 7. 首轮图和验收

唯一承重图：横轴 5/30/120 min，纵轴相对 `B_multiscale` 的 held-out future-block score；分面 count 与 conditional mark；曲线为 `P_local`、`P_slow`、correct-time、block-shift。same-prefix continuation 作为同图的小 panel 或紧邻补充 panel。

验收分三层：

- 工程：target/anchor/registry/session carry 正确；
- 仪器：模型实际更新、block shift 可计算、有效独立时间窗足够；
- 科学：`P_slow` 是否在 correct-time future block 超过 baseline/shift。

阴性边界：`P_slow` 阴性只反驳当前输入、容量和 multi-horizon objective 下的 predictive state；不能写“脑内没有慢状态”。

## 8. 允许结论

- 只下一事件改善：short-range predictive filtering。
- fixed-time count 改善、conditional mark 不改善：multiscale rate state。
- conditional participation/extent 改善：network-expression/extent state。
- same-prefix 后续触点/STOP 仍改善：state-dependent repertoire continuation。
- correct-time 不胜 shift：有预测码但时刻特异性不足。
