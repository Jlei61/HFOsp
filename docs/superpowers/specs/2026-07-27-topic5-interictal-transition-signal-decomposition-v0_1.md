# Topic 5 interictal transition signal decomposition v0.1

> 状态：执行合同  
> 日期：2026-07-27  
> 前置结论：v2.2 已按预注册规则停在 Claim 2；本合同不重训 v2.2，也不读取
> early-ictal target values。

## 1. 唯一科学问题

本阶段只回答：

> 一阶 Markov 相对 node-bias 的 heldout 增益到底来自植入局部几何、对称转移、
> 有向转移、物理轴 residual，还是超过一阶的有序历史？

本阶段不是新 RNN，也不以提高 AUC/NLL 为目标。只有分解结果同时支持跨 shaft、
source-conditioned axis 和 multi-step history，才允许另写 v2.3 recurrent model
合同。

## 2. 冻结边界

- 输入只使用 masked contact-rank set 序列、contact names 和已冻结三维坐标。
- 34 人原始队列不变；3 位 development 患者继续排除。
- coordinate-free 分解使用 31 位 development-excluded 患者。
- physical-axis 分解只使用其中 22 位 geometry-complete 患者。
- 每位患者固定 chronological train80 / heldout20。
- 所有 node hazard、transition residual、shaft/distance coefficient、axis 和历史权重
  只使用 train80。
- A/B labels、IEI、SOZ、clinical/EEG onset、ictal energy values均不进入拟合或选择。
- v2.2 Claim 3/4 保持 `LOCKED_NOT_RUN`，不能借本阶段补跑。

## 3. 共同评分合同

所有分解模型共享：

1. 同一患者、事件、prefix 和 eligible-contact denominator；
2. 同一 tie-set conditional-nonempty Bernoulli likelihood；
3. 同一 terminal STOP 定义；
4. 同一 event-first、patient-first 聚合；
5. 同一 train80 node bias；
6. patient-level median effect、positive-patient count 和 one-sided Wilcoxon；
7. 所有正式比较（包括跨 shaft endpoint）进入同一个 BH-FDR family；探索性
   readout 单独标注。

STOP 不读取 contact identity 或未来长度。本阶段固定为 formal node-control 的
`c0 + c_n * seen_fraction`，在 LOSO 的其余 physical-axis 患者 train80 上拟合，
从而让不同 transition operator 的差异只来自 contact score。

## 4. Train-only transition residual

对患者 \(p\)、当前 rank set 中的 contact \(i\) 和 eligible contact \(j\)，先用
Beta-smoothed train80 counts 得到：

\[
b_j=\operatorname{logit} h_j,
\qquad
L_{ij}=\operatorname{logit} h_{ij}-b_j.
\]

tie set 使用平均 residual：

\[
\eta_{j,t+1}
=
b_j+\frac{1}{|S_t|}
\sum_{i\in S_t}L_{ij}.
\]

不把 \(L\) 命名为 anatomical connectivity；它只是 train-only conditional
log-hazard residual。

冻结数值合同：

- node hazard：Beta(1,1)；
- pair transition：向 node hazard 收缩的 concentration = 10；
- geometry residual：ridge = 1；
- axis candidates = 32；
- axis kernel anisotropy ratio = 2.0；
- source-direction scalar 搜索范围 = `[-4, 4]`。

## 5. 分解层级

### 5.1 基础模型

- `node_bias`
- `empirical_probability_markov`：已接受的一阶概率 Markov sensitivity
- `directed_logit_markov`：上述完整 \(L\)

先确认两种 Markov 参数化在 heldout 上方向一致。

### 5.2 局部采样几何

在 train-only pair residual 上拟合：

- `same_shaft_only`
- `distance_only`
- `same_shaft_plus_distance`

distance 使用患者内固定 nearest-neighbour scale 标准化；shaft 由 contact name
去除末尾 contact number 得到。模型只含少量 ridge-regularized scalar
coefficients，不允许自由 contact-mixing。

主要比较：

\[
\text{directed logit Markov}
>
\text{same-shaft + distance}.
\]

若不通过，Markov 阳性优先解释为局部传播/采样几何。

跨 shaft 正式 endpoint 只纳入真实 next set 含跨 shaft contact 的 heldout
prefix，并在所有 unseen、相对当前 rank set 为跨 shaft 的 eligible contacts 上计算
同一 conditional-nonempty likelihood。阳性与阴性 eligible contacts都必须进入
评分。单独的 positive-contact NLL 只作 descriptive calibration，不得参与
go/no-go。患者必须至少有 20 个 heldout events 和 50 个 heldout prefixes 含
跨-shaft next contact 才进入这一 endpoint；其他 physical-axis 分析不受影响。

### 5.3 对称与反对称成分

\[
L^S=(L+L^\top)/2,
\qquad
L^A=(L-L^\top)/2.
\]

评估：

- `symmetric_only`
- `skew_only`
- `symmetric_plus_skew`

主要比较：

\[
\text{symmetric + skew}>\text{symmetric only}.
\]

它检验的是有效 transition 是否需要方向性，不否定底层结构 scaffold 可以近似
对称。

### 5.4 物理轴 residual

在 `same_shaft_plus_distance` 之上，使用 train-only 候选方向拟合：

\[
K^{axis}_{ij}-K^{local}_{ij}.
\]

axis 只从预冻结的 32 个 sign-invariant Fibonacci directions 中选择；每个候选只在
train80 pair residual 上评分。报告：

- heldout axis residual benefit；
- selected axis 与 contact-cloud PCA1 的 cosine；
- local/axis feature collinearity。

若 axis residual 不优于局部几何，不允许进入 axis-RNN。

### 5.5 Source-conditioned directional component

对 train-only 选定轴 \(u_p\)，定义 source projection 的稳健连续分数：

\[
d_e=
\tanh\left[
\frac{s_e-\operatorname{median}_{train}(s)}
     {\operatorname{IQR}_{train}(s)+\epsilon}
\right].
\]

共同轴上的反对称方向基函数：

\[
K^A_{ij}
=
K^S_{ij}
\tanh\left(\frac{s_j-s_i}{\delta}\right),
\qquad K^A=-(K^A)^\top.
\]

只拟合一个 train80 scalar \(\beta_p\)：

\[
\eta=b+L_{local}x_t+\alpha K^Sx_t+\beta_p d_e K^Ax_t.
\]

正反事件共享全部参数；`d_e` 是 observed source 的连续量，不是 A/B label 或离散
path identity。

### 5.6 历史深度

在同一 train-only directed residual 上比较：

- `source_only`
- `last_rank`
- `last_2_ranks`
- `last_3_ranks`
- `unordered_full_prefix`
- `ordered_full_prefix`

`ordered_full_prefix` 的 decay 只可在 train80 内部的 chronological
train60/validation20 从 `{0.25, 0.5, 0.75}` 选择，然后在完整 train80 重新估计
residual，heldout20 只评估。

允许 recurrent state 的最低证据是：

\[
\text{ordered multi-step history}
>
\text{last-rank Markov}
\]

在患者层面方向稳定且经多重比较后通过。
`source_only` 同时作为必要参照，用于判断 last-rank 是否已包含超出事件起点的
即时 transition information。

## 6. Go / no-go

### 允许写 v2.3 spec

必须同时满足：

1. directed Markov 超过 `same_shaft_plus_distance`；
2. source-conditioned axis component 有正的 patient-level heldout benefit；
3. ordered multi-step history 超过 last-rank Markov；
4. 以上不是由少数患者或 topology-only 患者驱动。

### 只允许最小结构模型

只有 symmetric residual 相对 node-bias 和 axis residual 相对 local geometry
均阳性、但多步历史不超过 Markov 时，才只允许 transition operator，不允许 RNN。

### 停止 system-identification 主线

若 axis residual 与 cross-shaft residual均不稳定，停止从 contact-rank 序列反推
物理轴。Markov 阳性只保留为“事件内顺序非随机”的 supplementary self-supervised
结果。

## 7. 产物

输出根：

`results/topic5_interictal_transition_decomposition_v0_1/`

必须包含：

- `SCORING_CONTRACT_AUDIT.json`
- `patient_model_metrics.csv`
- `cohort_comparisons.csv`
- `operator_component_metrics.csv`
- `cross_shaft_positive_metrics.csv`
- `cross_shaft_prefix_metrics.csv`
- `cross_shaft_eligibility.csv`
- `CROSS_SHAFT_STATUS.json`
- `history_depth_metrics.csv`
- `DECOMPOSITION_STATUS.json`
- `per_subject/*.json`
- `figures/transition_signal_decomposition.png`
- `figures/transition_signal_decomposition.pdf`
- `figures/README.md`

## 8. Clinical metadata 独立边界

clinical-onset source 标注工作流与本合同并行但完全隔离。标注者不能看到本阶段
score 或 early-ictal energy。没有逐发作 exact source set 时，不论本阶段结果如何，
early-ictal transfer 都保持 blocked。
