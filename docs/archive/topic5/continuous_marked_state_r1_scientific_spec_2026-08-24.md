# Continuous marked-state R1 Scientific Spec

**版本：** R1.0–R1.3
**日期：** 2026-08-24
**状态：** development-only frozen contract
**上游验收：** `continuous_marked_state_r0_1_acceptance_2026-08-24.md`

## 0. 唯一科学目标

R1 只回答一个问题：

> 连续背景 SEEG 所校正的同一个患者状态，是否在确定性 IED history 之外，直接改善下一次 IED 的发生时刻和完整 tied-contact recruitment mark？

R1 正式检验 H1 与 H2a。R1 不检验 IED 对未来状态的作用；任何 T2 edge、exposure 网格和 seizure probe 都不进入本版本。

## 1. 唯一学习状态与显式历史

每位患者只有一个学习状态：

\[
z_p(t)\in\mathbb R^{8}.
\]

它由规则背景 SEEG 观测校正，在观测间隔内连续演化，并在事件发生前调制 timing 和 mark likelihood。`state_dim=16` 只允许一次容量敏感性，不能形成新网格。

显式历史 `r(t)` 为确定性计算，包括：

- 距上次 IED 的时间；
- 30 s、2 min、10 min 的事件计数/指数迹；
- 最近事件 load、group count、participation 和 tied-group 摘要；
- session elapsed time、time-of-day；
- recording coverage 与 gap boundary；
- contact 静态几何、shaft identity 和 TRAIN-only 静态 repertoire baseline。

R1 禁止 free event-history RNN。这样 `r(t)` 负责已知短程历史，`z(t)` 才能被解释为背景观测带来的持续预测分量。

## 2. 时间与数据边界

### 2.1 规则背景观测

每 30 s 一个 anchor，读取严格位于 anchor 之前的 30 s：

\[
X_k\in\mathbb R^{C_p\times 30f_s}.
\]

输入包含：

- 所有可用 bipolar contacts 的预处理 raw SEEG；
- 只对 anchor 时刻已经发生的 IED core 使用 `±1 s` 因果 mask 做背景插补；mask 位置本身不得进入 observer；
- bad-channel、recording-gap 和 sample-valid mask；
- contact coordinates 与 shaft identity；
- 显式 spectrum、variance、mean absolute derivative、zero crossing、lag-1 autocorrelation 和 valid fraction。

anchor 必须与事件是否发生无关。当前 R0.1 的 60 s clock 只能作为兼容 smoke；R1 主合同是 30 s clock。

### 2.2 IED event

第 `e` 个事件表示为：

\[
\mathcal E_e=(t_e,S_{e,1},\ldots,S_{e,K_e}),
\]

其中每个 `S_{e,k}` 是显式 `group_ids` 定义的 tied contact subset。禁止用相同浮点 rank 猜 tie，也禁止给非参与 contact 读取 phantom rank。

`z(t_e^-)` 只能预测当前事件。当前事件的 group、load、rank 和 waveform 不得进入 `z(t_e^-)`。

### 2.3 development split

- 只使用既有 TRAIN 与 validation；
- validation 状态用全部较早 TRAIN history 做 no-grad causal warm-start，不在边界重置；
- 正式 test/sealed partition 不得读取、统计分母或用于选择；
- preictal 信号可在 development 的无标签背景 observer 中出现，但 seizure label/onset 不得作为输入。

## 3. Observation Transformer

R0.2 的 wide Transformer 只作为代码结构起点，不复用频谱 forecasting latent 或 checkpoint。

### 3.1 explicit branch

每个 contact 的显式背景特征经过共享小型 MLP：

\[
e^{explicit}_{k,i}=E_{explicit}(s_{k,i}).
\]

### 3.2 raw residual branch

每个 contact 的 30 s raw waveform 切成 250–500 ms patches，使用共享的两层 pre-norm temporal Transformer：

- `d_model=64`；
- 4 heads；
- 2 layers；
- recording/artifact sample mask 进入 attention pooling；既往 IED core 先做线性背景插补，IED mask stencil 不向网络暴露；
- 不跨 contact 混合，直到 spatial stage。

输出以零初始化残差进入融合：

\[
e_{k,i}=e^{explicit}_{k,i}+\alpha_{raw}e^{raw}_{k,i}
+E_{coord}(q_i)+E_{shaft}(i),\qquad \alpha_{raw}=0.
\]

`alpha_raw` 可以学习正负，不施加“raw 必须有用”的方向约束。

### 3.3 spatial stage

对变长 contact set 使用 1–2 层 masked spatial Transformer，并由 pool token 输出 64 维 `observation_embedding`。这个对象永远不叫 state。

## 4. Persistent controlled state

观测之间：

\[
z(t+\Delta t)^-=\mu+\exp(\Delta tK)[z(t)^+-\mu],
\]

\[
K=\Omega-Q,\quad \Omega^T=-\Omega,\quad
Q=\operatorname{diag}(\operatorname{softplus}(q)+q_{min}).
\]

背景观测更新：

\[
c_k=\tanh W_c[o_k,z_k^-],\quad
g_k=\sigma W_g[o_k,z_k^-],
\]

\[
z_k^+=(1-g_k)\odot z_k^-+g_k\odot c_k.
\]

该对象首先称为 `controlled persistent state`。只有后续证据达到相应层级时，才升级命名。

## 5. Recorded-interval timing likelihood

### 5.1 coverage contract

每位患者的记录 coverage 必须表示为不重叠、按时间排序的闭开区间：

\[
\mathcal R_p=\bigcup_j[a_j,b_j).
\]

区间直接来自 frozen block inventory/EDF header resolver；不能从事件密度推断。相邻或轻微舍入重叠的 block 按既有 session contract 处理。gap 不进入 survival integral。

现有 `recorded_intervals/*.npz` 的累计 recorded duration 只可用于 parity audit；R1 主 likelihood 必须保留每个 coverage segment 的真实起止位置。

### 5.2 deterministic history baseline

先在 TRAIN 拟合并冻结：

\[
\log\lambda_0(t|r(t)).
\]

R1 的 state residual 从精确零开始：

\[
\log\lambda(t)=\log\lambda_0(t|r(t))+w_\lambda^Tz(t),\qquad w_\lambda=0\text{ at init}.
\]

### 5.3 exact likelihood

在 recorded coverage 上：

\[
\log p(\{t_e\})=
\sum_e\log\lambda(t_e)-\int_{\mathcal R_p}\lambda(u)\,du.
\]

积分边界由 coverage boundary、observation anchor、event time 和 split boundary 联合切分。实现可用固定 Gauss–Legendre quadrature；训练与验证必须使用同一确定性节点。报告 event log-intensity、survival integral、每 recorded hour NLL 和 time-rescaling calibration。

Timing calibration 只限制 timing claim，不作为 mark/H2a 的总 gate。

## 6. Exact tied-group sequential mark likelihood

在第 `k` 个 tied group 前，eligible contacts 为尚未募集的集合 `E_{e,k}`。

### 6.1 group size / STOP

模型先预测：

\[
p(n_{e,k}|prefix,z,r,G),\qquad n=0\text{ 表示 STOP}.
\]

非终止 group size 必须满足 `1 <= n <= |E|`。

### 6.2 unordered without-replacement subset

对 eligible contact 产生 logits `ell_i`。给定 `n`：

\[
p(S|n,E)=
\frac{\exp(\sum_{i\in S}\ell_i)}
{e_n(\{\exp(\ell_j):j\in E\})},
\]

其中 `e_n` 是 elementary symmetric polynomial，以 log-space dynamic programming 精确计算。

这一定义保证：

- tied contacts 内没有虚假顺序；
- 已募集 contacts 不会再次被选；
- subset 无放回；
- 每个合法 size-`n` subset 的概率和严格为 1。

最终：

\[
\log p(m_e)=\sum_k\left[
\log p(n_{e,k})+
\mathbf1_{n_{e,k}>0}\log p(S_{e,k}|n_{e,k},E_{e,k})
\right].
\]

必须单独报告：group-size/STOP、contact identity、full mark，以及按 prefix depth 的 continuation。

## 7. 唯一训练目标

\[
\mathcal L_{event}=-\frac{1}{N_{event}}
\left[
\sum_e\log\lambda(t_e)-\int_{\mathcal R}\lambda(u)du
+\sum_e\log p(m_e|t_e)
\right].
\]

Timing 与 mark 是同一个事件 likelihood 的可加 log terms，不引入人工 loss weight。R1 禁止频谱重建、waveform reconstruction、KL、contrastive loss、seizure loss 和 latent consistency loss。

## 8. R1 对照臂

### 8.1 Stage 0 baselines

- `B_history`: deterministic timing history + static/history exact mark decoder；
- `B_explicit`: `B_history` + explicit background observation；
- `B_explicit_raw`: `B_explicit` + zero-initialized raw residual branch。

三臂使用相同事件、coverage、likelihood、split、contact decoder 和训练预算。

### 8.2 T1 state contrasts

- `T0_no_state`: observer 存在但不能写入 persistent state；
- `T1_explicit`: explicit observer 校正一个持续状态；
- `T1_explicit_raw`: explicit+raw observer 校正同一结构状态；
- `wrong_time_swap`: 同 session 内匹配 time-of-day、recent rate、last mark 和 coverage 后替换 state；
- `state_dim16`: 仅一次容量敏感性。

## 9. 三种 rollout 必须分名

1. **One-step next-event prediction**：同一 pre-event state，只评分下一个 timing 与 mark。
2. **Event-observed, raw-correction-off rollout**：anchor 后关闭背景 raw correction，但使用真实未来 IED timing/mark 更新确定性 history。它是 teacher-forced causal rollout，不叫 open-loop。
3. **Fully generative rollout**：模型自行采样 timing 与 mark，并用采样事件更新 history。只作 supportive。

## 10. H1/H2a 结论阶梯

| 证据 | 允许名称/结论 |
|---|---|
| filtered joint likelihood 改善 | predictive filter |
| filtered + matched wrong-time swap | time-specific predictive state estimate |
| correction-off 且保留已知外部条件仍改善 | controlled generative state |
| fully generative rollout 仍稳定 | autonomous generative model |
| exact mark identity/continuation 改善 | state-dependent recruitment field/repertoire prediction |

普通阴性不阻断后续探索。只有时间泄漏、future mark 泄漏、coverage 错误、subset law 不归一、split/sealed 违规或结果包混版会使对应结果失效。

## 11. R1.2 六人 pilot

冻结患者：

- `epilepsiae_620`；
- `epilepsiae_958`；
- `epilepsiae_139`；
- `yuquan_huanghanwen`；
- `yuquan_zhangjiaqi`；
- `yuquan_hanyuxuan`。

这六人覆盖事件量约 456–123,419、contact 数 7–22，并同时包含 Epilepsiae 与 Yuquan。小事件量患者不剔除；只在患者内报告估计不确定性。

R1.2 的输出是 H1/H2a 可识别性判断，不是队列结论。结构冻结后才进入 R1.3 的 34 人 development cohort。

## 12. 与 H3 的边界

R1 不允许任何 event innovation 写入 `z(t_e^+)`。当前事件只能更新确定性 history，不能通过学习 jump edge 改变 state。R1 完成后，R2 才从同一 T1 checkpoint 加一个低秩 event-innovation-to-state edge。
