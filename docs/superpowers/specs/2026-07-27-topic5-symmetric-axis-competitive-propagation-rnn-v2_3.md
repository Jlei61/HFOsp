# Topic 5 symmetric-axis competitive propagation RNN v2.3

> 状态：冻结设计合同；尚未进入正式训练
> 日期：2026-07-27
> 前置证据：transition decomposition v0.1 允许开发最小 recurrent observation
> model；early-ictal target 继续 sealed。

## 1. 唯一科学问题

本阶段只回答：

> 一个共享、正值、对称的 axis-aligned scaffold，加上 observed source 决定符号的
> 低容量方向项和 1–2 个传播/竞争历史状态，能否解释 template-free interictal
> contact-rank transitions？

模型不是为了超过 GRU，也不训练 A/B 分类。A/B 只可在所有间期 gate 结束后作
heldout read-back。

## 2. 为什么允许重新引入 recurrent state

冻结 transition decomposition 显示：

- Markov > node-bias：30/31；
- directed Markov > local geometry：21/22；
- 正式 cross-shaft endpoint：19/20；
- symmetric residual > node-bias：30/31；
- arbitrary skew increment：不显著；
- axis-aligned residual > local geometry：20/22；
- ordered full-prefix > last-rank：30/31。

因此 recurrent state 的依据是 ordered multi-step history，而不是“RNN 可能更强”。
source-conditioned增益虽经 FDR 后通过，但中位效应仅 \(1.73\times10^{-5}\)，只能
进入一个单 scalar、低容量 sensitivity term。

## 3. 数据与 cohort

### 3.1 输入

- masked `contact-rank` sequences；
- observed source rank set；
- fixed contact names 和三维坐标；
- 每个 event 开始时状态归零；
- 不输入 A/B、IEI、SOZ、seizure label 或 ictal values。

### 3.2 Tied ranks

全 34 人 864,163 个事件中，只有 25 个事件存在非 source 的 tied next rank。primary
训练和评估排除这 25 个事件，使用单一 next-contact categorical likelihood。排除表
必须按患者保存；原 conditional-nonempty set likelihood 只作包含 tied events 的
sensitivity。

### 3.3 Cohort

- development：
  `epilepsiae_1077`、`epilepsiae_1146`、`yuquan_chengshuai`；
- primary physical-axis：
  22 位 geometry-complete、development-excluded patients；
- supportive sequence cohort：
  31 位 development-excluded patients，但不作 physical-axis claim；
- chronological train80 / heldout20；
- train80 内再按 chronological train60 / validation20 选择 optimizer epoch；
- heldout20 不用于模型、epoch、方向或阈值选择。

## 4. 固定 scaffold

对患者 \(p\)，使用 transition decomposition 在 train80 中选择的 sign-invariant
axis \(\mathbf u_p\)。22 位 formal patients直接消费已冻结 axis；三位 development
patients 因未进入 decomposition formal cohort，使用同一 32-direction、train-only
procedure 单独生成 development axis。不在 v2.3 中扫描 heldout 或增加方向数。

\[
A_p(\gamma_p)
=
(1-\gamma_p)\bar K_{\mathrm{local},p}
+
\gamma_p\bar K_{\mathrm{axis},p},
\qquad 0\leq\gamma_p\leq1,
\]

\[
W^S_p
=
D^{-1/2}A_pD^{-1/2},
\qquad W^S_p=(W^S_p)^\top,\quad W^S_{ij}\geq0.
\]

其中：

- local scale 固定为患者 contact cloud 的 nearest-neighbour median；
- axis anisotropy ratio 固定为 2.0；
- kernel 在混合前分别做 Frobenius normalization；
- \(\gamma_p\) 是 patient-specific scalar；
- 不允许 dense residual matrix。

共同轴上的方向基函数为：

\[
W^A_{ij}
=
W^S_{ij}
\tanh\left(\frac{s_j-s_i}{\delta_p}\right),
\qquad W^A=-(W^A)^\top,
\]

其中 \(s_i=(\mathbf r_i-\bar{\mathbf r})^\top\mathbf u_p\)，
\(\delta_p\) 固定为 train-contact projection difference 的中位绝对值。

event source score只使用 observed source：

\[
d_e=\tanh\left(
\frac{\operatorname{mean}_{i\in S_0}s_i}
{\operatorname{std}_i(s_i)+\epsilon}
\right).
\]

## 5. 最小 recurrent dynamics

传播与竞争状态均为 contact-wise 向量：

\[
P_{t+1}
=
\rho_P P_t+W^S_px_t,
\]

\[
C_{t+1}
=
\rho_C C_t+W^S_px_t,
\qquad 0\leq\rho_P<\rho_C<1.
\]

下一 contact score：

\[
z_{t+1}
=
b
+
g_P P_{t+1}
-
g_C C_{t+1}
+
\beta\, d_e W^A_px_t
+
m_t,
\]

其中：

- \(b\) 为 train80 Beta(1,1) node hazard 的 fixed logit；
- \(g_P,g_C\geq0\)；
- \(\beta\) 为单一 patient-specific scalar；
- \(m_{i,t}=-\infty\) mask 已参与 contacts；
- 不允许 MLP、GRU cell、attention 或额外 contact mixing；
- \(\rho_P,\rho_C\) 在三位 development patients上选择后全队列共享；
- 每位 formal patient 只拟合
  \(\gamma_p,g_{P,p},g_{C,p},\beta_p\) 四个 scalars。

这两个状态只命名为 propagation drive 和 delayed competition，不命名为
excitation/inhibition。

## 6. 输出概率

### 6.1 Primary next-contact

条件于 event 继续：

\[
P(j_{t+1}=j\mid\mathrm{continue},\mathrm{prefix})
=
\operatorname{softmax}_{j\in\mathrm{eligible}}(z_{j,t+1}).
\]

它严格预测一个 next contact，从模型层面消除 v2.2 的 set-size overprediction。

### 6.2 STOP

STOP 固定使用 transition decomposition 接受的 LOSO control：

\[
\operatorname{logit}P(\mathrm{STOP})
=
c_0+c_n\cdot\mathrm{seen\_fraction}.
\]

不同 transition models在同一 scope 内共享完全相同的 STOP。primary contact
comparison 另行报告 conditional-on-continue categorical NLL，避免 STOP 掩盖
contact score。

### 6.3 完整 rollout

只有 primary heldout gates 通过后，才从同一 categorical transition + STOP
autoregressive rollout 得到：

- node participation probability；
- node first-arrival rank distribution；
- common path bundles；
- state trajectories。

rollout 不增加独立 decoder 或 future head。

## 7. 模型与对照

所有对照共享 node bias、STOP、event/prefix denominator 和 optimizer budget：

1. `node_bias_categorical`
2. `empirical_last_rank_markov`
3. `empirical_ordered_history_markov`
4. `local_isotropic_two_state`：\(\gamma=0,\beta=0\)
5. `axis_one_state_no_competition`：\(g_C=0,\beta=0\)
6. `axis_two_state_no_source`：\(\beta=0\)
7. `axis_two_state_source_full`
8. `axis_instantaneous_no_history`：\(\rho_P=\rho_C=0\)

不运行 dense GRU upper bound。

## 8. Development freeze

三位 development patients只允许选择：

- persistence pair：
  \((0.25,0.50)\)、\((0.50,0.75)\) 或 \((0.50,0.90)\)；
- learning rate：`3e-3` 或 `1e-2`；
- maximum epochs：200；
- patience：20；
- gradient clipping：5；
- batch events：从 512、1024、2048 中选不 OOM 的最大值。

optimizer 固定 AdamW，weight decay `1e-4`。选择标准为三患者 train60/validation20
的 patient-first categorical NLL；不得看 A/B、axis read-back 或任何 ictal target。

## 9. 正式 Claims 与停止规则

所有正式比较用 patient-level paired effects、bootstrap median CI、positive-patient
count、one-sided Wilcoxon 和同一 BH-FDR family。

### Claim A：predictive adequacy

\[
\text{full}>\text{node-bias categorical}.
\]

必须 median benefit > 0、CI 下界 > 0、超过半数患者为正且 q<0.05。失败则停止，
不分析 latent states。

### Claim B：历史状态必要

\[
\text{full}>\text{instantaneous no-history},
\]

且 full 相对 one-state/no-competition 至少一个正式 comparison 通过。失败则回退为
structured transition operator，不再称 RNN contribution。

### Claim C：axis scaffold 增量

\[
\text{full}>\text{local-isotropic two-state}.
\]

失败则停止 physical-axis system-identification claim；可保留 coordinate-free
history model。

为防止该比较被 full 独有的 source term 混淆，physical-axis 解释还必须同时通过
以下 matched safeguard：

\[
\text{axis two-state no-source}
>
\text{local-isotropic two-state}.
\]

该 safeguard 不新增模型或调参；两个条件均已在冻结模型表中。它只收紧
physical-axis claim，不改变 Claim D 的独立 secondary 地位。

### Claim D：source-conditioned direction

\[
\text{full}>\text{axis two-state no-source}.
\]

这是独立 secondary claim，不作为 Claim A–C 的总 gate。只有它通过后，才允许分析
同一 scaffold 的 source-side reversal；效应量必须和 decomposition 中的微小增益
一起报告。

### Empirical Markov benchmark

full 不要求超过 empirical ordered-history Markov。报告 full 恢复了多少 Markov
相对 node-bias 的 benefit，用作可解释性代价，不作机制 gate。

## 10. Latent-state analysis

仅在 Claim A–C 通过后开放：

- \(P_t\)、\(C_t\) 随 rank step 的 patient-first trajectories；
- node-level cumulative drive 与 heldout rank distribution；
- \(g_P P-g_C C\) 的正/负贡献；
- 同一 \(W^S\) 从两侧 observed source 初始化的 rollout；
- A/B labels 的 heldout read-back；
- axis coefficient mixed-sign 解释：是否由 delayed competition 产生，而不是负连接。

不能用训练集 A/B 调整模型或选 representative patient。

## 11. Early-ictal boundary

即使 Claim A–D 全部通过，只要 clinical-onset source registry 仍是 0 exact
per-seizure contacts，early-ictal transfer 继续为：

`BLOCKED_MISSING_EXACT_CLINICAL_ONSET_SOURCE_METADATA`

不允许使用 SOZ、患者级 focus、A/B source 或 energy-top contacts补位。

## 12. Go / no-go

- Claim A–C 通过：允许做 latent-state dynamics 和 source-side sensitivity；
- Claim A 通过但 B 失败：降级为非 recurrent operator；
- Claim A/B 通过但 C 失败：保留 coordinate-free history result，停止 physical-axis
  claim；
- Claim D 失败：禁止“source determines direction”；
- 任一结果都不自动开放 ictal target。
