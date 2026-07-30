# Topic 5 RNN internal-state reduction v0.1

## 1. 前提

v2.5 显示 full-history GRU 的 contact-level static readout 高于 static contact hazard，
但相对 rank-shuffle、unordered-prefix 和 first-order controls 的 all-contact-null margin
优势不稳定。结构化 RNN 的 history、axis 和 source 项没有独立迁移增量。

因此 v0.1 不再训练一个更复杂的新 RNN，也不直接把 hidden unit 命名为病理轴或 E/I。
它只问：

> full-history GRU 相对静态 contact prior 多保留的部分，是否由少数可重复、与事件进度
> 和未来 contact distribution 有明确关系的 hidden-state directions 承载。

## 2. 输入与冻结

- 模型：现有 34 人、3 seed 的 full-history GRU；
- 对照：rank-shuffle GRU、last-set first-order、static contact hazard；
- 数据：chronological heldout20 events；
- 不重新训练 GRU；
- 不使用 A/B 标签、ictal target、IEI、SOZ；
- hidden-state reduction、维度选择和 probe 全部在 interictal train/validation 内冻结，
  之后才读取 heldout20；
- patient-first、seed-within-patient collapse。

## 3. 分析

### 3.1 Hidden trajectory inventory

对每个 event prefix 保存：

- hidden state；
- 当前 rank index；
- 已参与 contact set；
- 下一 contact；
- remaining contact participation；
- event length；
- source-side projection仅在有几何患者中作描述。

每个事件状态在开始时清零；不得把最终事件长度作为输入。

### 3.2 低维性

在 train/validation hidden states 上计算：

- participation-weighted PCA effective rank；
- 解释 80%、90%、95% variance 所需维数；
- 跨 seed subspace angle；
- chronological split-half subspace stability。

PCA 只是压缩，不是机制命名。

### 3.3 可解释 probes

从冻结低维 state 预测：

1. 当前 event progress（仅当前可观测 rank index）；
2. 下一 contact；
3. future participation；
4. remaining normalized rank distribution。

所有 probe 与以下输入匹配比较：

- node bias only；
- current set only；
- last-set first-order；
- rank-shuffle GRU hidden state；
- full GRU hidden state。

只有 full hidden state 稳定超过 rank-shuffle 和 last-set，才称为 ordered-history state。

### 3.4 Causal rollout perturbation

对冻结 hidden state 的前两条稳定 direction 做小幅正负 perturbation，再从同一 prefix
free rollout。报告：

- 每个 contact participation change；
- early/late/endpoint mass change；
- rollout divergence；
- seed stability。

不允许根据 early-ictal target 选择 perturbation direction。

### 3.5 Target-sealed read-back

上述 direction 冻结后，才在 strict clinical-onset 16 人/106 seizure 中检查：

- direction-induced contact field 与 early-ictal energy 的 absolute similarity；
- coherent all-contact shuffle；
- 与未扰动 GRU、rank-shuffle GRU 的 paired difference。

这是静态 read-back，不是发作动态预测。

## 4. 验证矩阵

本阶段不使用单个 hard gate，也不因为一项阴性结果提前停止。所有分析均完成后，
按 patient-first effect size、bootstrap 95% CI、正向患者比例和 seed 稳定性共同判断。

### 4.1 冻结分层

每位患者按时间固定为：

- `train60`：原 train80 的前 75%；
- `validation20`：原 train80 的后 25%；
- `heldout20`：原合同封存的最后 20%。

PCA、probe 参数、正则化和方向符号只允许使用 `train60/validation20`。最终效应只在
`heldout20` 计算。原 GRU 的 patient-local calibration 已使用 train80，因此
`validation20` 只用于新 probe 的选择，不被描述为独立模型验证集。

### 4.2 低维性不是单一维数

同时报告：

- PCA effective rank 与 80/90/95% variance 维数；
- `k={1,2,4,8,16,32}` 的 heldout hidden reconstruction；
- 用这些重建 state 进行 next-set decoding 后的 NLL 保真度；
- 同维度随机子空间 sensitivity。

不根据某一个解释方差阈值宣布“存在”或“不存在”低维动力学。

### 4.3 ordered-history 增量

至少完成以下独立比较：

1. full GRU 与 rank-shuffle GRU；
2. full GRU 与 unordered-prefix、last-set、static controls；
3. 同一个 full GRU 内，将已观察 prefix 的 rank-set 顺序打乱，但保持 prefix
   成员、候选 contact、目标和 STOP 不变；
4. observable prefix features 之后，加入 full/rank-shuffle hidden state 的
   probe 增量；
5. 第 1、2、3 步和早/中/晚 prefix 分层。

主要连续指标是 event-first next-action NLL、future-participation Brier 和
remaining-rank score MSE，不把 top-1 accuracy 单独作为裁决指标。

### 4.4 稳定性

- 3 个既有 GRU seeds；
- 同一 heldout prefix 上的 raw linear CKA；
- 回归掉 progress、recruited set 和 last set 后的 residual CKA；
- chronological split-half PCA subspace overlap；
- patient bootstrap 20,000 draws；
- patient-level seed median，禁止把 seed 当独立样本。

### 4.5 扰动

同时测试 PCA variance directions 与 output-coupled directions。扰动幅度固定为
`±{0.25,0.5,1.0}` 个 train-state SD，报告 dose-response、contact field、
Jensen-Shannon divergence、STOP 变化和跨 seed 一致性。方向和符号不得由
early-ictal target 选择。

## 5. Early-ictal read-back

严格使用 Epilepsiae clinical-onset、`0–10 s`、`1–150 Hz` 的 16 人/106 次发作。
Yuquan EEG-onset 病例不混入 primary。

为避免五个 field 中取最大值成为唯一证据，分两层：

1. 固定 readout：`participation` 与 `endpoint_joint_mass` 分别报告；
2. 五 field 最大绝对相关只作 omnibus sensitivity，并在每次 permutation 中重新选 field。

primary null 是患者内 all-contact coherent permutation；within-shaft null 作为更强的
解剖 sensitivity。统计先按 seizure、再 seed、最后 patient 折叠。

需要明确标注：v2.5 已经读取过这一 target，因此本阶段属于同一数据集上的机制拆解，
不是新的独立确认。

## 6. 证据分级与后续

不输出二元 `GO/NO-GO`。完成全部实验后分为：

- **Tier A**：ordered-history 相对 rank-shuffle、unordered 和 first-order 均有稳定
  heldout 增量，低维方向可重复且有可迁移 contact loading；
- **Tier B**：hidden state 有稳定低维结构和预测增量，但 ordered-history 特异性有限；
- **Tier C**：主要收益可由 static/unordered contact scaffold 解释，hidden state
  仍可作为有效压缩器，但不命名为传播动力学。

无论落在哪一层，都保留完整定量结果。只有在 Tier A 时才设计下一版 constrained
state model；Tier B/C 不等同于整个 RNN 方向无价值，也不据此否定患者病理轴。
