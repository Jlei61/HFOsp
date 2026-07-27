# Topic 5 axis-positive RNN read-back and early-ictal static transfer v2.4

> 日期：2026-07-27
> 状态：冻结执行合同
> 前置：v2.3 已证明 next-contact 可预测且需要 ordered history，但 cohort-wide
> physical-axis 与 source-direction claims 未通过。

## 1. 科学问题

本实验把两个问题分开：

1. 在既有分析已经独立定义为 A/B 轴共线的患者中，仅以间期
   next-contact prediction 为目标的 RNN，能否选择与冻结 A/B shared axis
   对应的物理轴？
2. 不使用发作起始触点，仅使用纯间期模型冻结的 contact participation/rank
   distribution，能否跨患者读出 clinical onset 后 `[0,10] s`、1–150 Hz
   静态能量场？

第一问是 axis-positive subgroup 的构念 read-back，不把患者扩展到全队列。第二问是
source-free cross-state static readout，不是发作传播模拟或前瞻性发作预测。

## 2. 冻结输入

### 2.1 间期事件

- 数据：`results/topic5_interictal_rank_distribution/dataset_v0_4/`。
- 使用 masked contact-rank events。
- split：冻结的 chronological train80/heldout20；train80 内保持既有
  fit60/validation20。
- node bias、source distribution 和 event-length distribution 只能由 train80
  估计。
- non-source tied-rank events 沿用 v2.3 的25-event exclusion。

### 2.2 外部 A/B 轴定义

只消费已经冻结的：

- `results/topic5_ictal_recruitment/template_axis_field/axis_cohort.csv`
- `results/topic5_ictal_recruitment/template_axis_field/per_subject/*.json`

不得用 v2.3/v2.4 预测性能重新定义患者是否 axis-positive。

### 2.3 亚组

与 v2.3 physical-axis formal n=22 相交：

- primary construct-validity subgroup：
  `axis_pair_estimable & geometry_2d_supported & collinear_60deg`，冻结 n=9；
- directional secondary：
  上述患者中 `relation == reversed`，冻结 n=6；
- stability sensitivity：
  reversed 且 `strict_stability_pass`，冻结 n=5。

development 患者不进入统计。

## 3. Axis read-back

### 3.1 Stage A0：现有 transition-selected axis

v2.3 当前输入轴来自 train80 transition residual 的32方向离散搜索，不是 RNN
内部可学习参数。先报告：

- 该轴与冻结 `u_shared` 的 `abs(cosine)`；
- 相对同一32候选方向的 alignment percentile；
- selected alignment 减候选方向 median；
- selected axis 与 contact-cloud PCA1 的 `abs(cosine)`；
- 现有 full / axis-no-source / isotropic heldout NLL comparison。

该层只能称为 transition-selected axis read-back。

### 3.2 Stage A1：RNN-selected axis

为真正回答 RNN objective 是否能选择病例轴：

- 每位 primary subgroup 患者使用固定的32个 sign-invariant Fibonacci directions；
- 每个方向训练 `axis_two_state_no_source`；
- seeds 固定为 17、29、43；
- optimizer、persistence、batch size、maximum epochs、patience 全部继承 v2.3
  `DEVELOPMENT_FREEZE.json`，禁止重新调参；
- 每个 seed 仅按 validation20 categorical NLL 选择一个方向；
- heldout20 不参与方向选择、early stopping 或参数更新；
- 选定方向后再以相同 seed 拟合 `axis_two_state_source_full`，仅用于 n=6 reversed
  directional secondary。

主要指标：

1. RNN-selected axis 与冻结 `u_shared` 的 seed-median `abs(cosine)`；
2. selected alignment 减32方向 median；
3. selected-axis model 相对 local-isotropic 的 heldout NLL benefit；
4. 三 seed selected axes 的 pairwise sign-invariant cosine；
5. n=6 reversed 中 source-full 相对 no-source 的 heldout benefit。

alignment 是同一间期数据上的外部 read-back，因此只能称为 construct validity。
只有 alignment 与 heldout predictive increment 同时为正，才允许写
“RNN prediction objective recovered the pre-existing patient axis in the
axis-positive subgroup”。

## 4. 冻结节点级间期表征

在读取任何新 early-ictal target 数值前，为每位 target-metadata eligible formal
患者生成以下 contact-level distribution：

- 第0维：contact 不参与模拟事件的概率；
- 第1–10维：contact 参与且落在对应 normalized-rank bin 的联合概率；
- 11维之和必须为1。

每个模型 seed 运行5000次 free rollout：

- source contact/set 从 train80 empirical source distribution 采样；
- event length 从 train80 empirical event-length distribution 采样；
- 其余 contact 按模型 categorical next-contact distribution 无放回采样；
- 所有模型共享同一组 source、length 和 random-number streams。

冻结表示：

1. `full_fixed_axis`：v2.3 full；
2. `no_history`；
3. `local_isotropic`；
4. `node_only`；
5. `empirical_train80`；
6. `rnn_selected_axis`：仅在 n=9 axis-positive subgroup 作为 sensitivity。

三 model seeds 先在患者×contact×feature 内取中位数，再对每个 contact 的11维向量
做一次 simplex closure（除以该行之和；不改变分量次序）。随后写出：

- 每患者 NPZ；
- contact order；
- representation manifest；
- 所有输入/checkpoint/输出 SHA256；
- `TARGET_UNLOCK.json`。

只有 `TARGET_UNLOCK.json::status == FROZEN_INTERICTAL_REPRESENTATIONS` 且
`target_values_read == false` 时，才允许进入 §5。

## 5. Source-free early-ictal static readout

### 5.1 Target

- primary anchor：clinical onset；
- window：`[0,10] s`；
- frequency：1–150 Hz；
- feature：baseline-robust-z contact energy 的窗口均值；
- cache：
  `results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150/`；
- 每位患者先在 seizure 内按既有 cache 得到 contact field，再跨 seizure
  取 contact-wise median；
- exact contact-name join；
- 至少6个共同 contact；
- EEG onset 不替代 clinical onset。

逐发作 clinical-onset source contacts 不用于本任务。0/71 source registry 只继续
阻断 source-conditioned dynamic rollout。

### 5.2 Cohort

主分母由 v2.3 formal n=22 与完整 BB150 metadata/cache 取交集，在读取 target 数值前
由 sidecar、NPZ member inventory 和 exact contact-name join 冻结。预计 n≥8。

axis-positive 且 target-ready 的患者单独作为描述性 sensitivity，不能替代全体
target-ready 主分母。

### 5.3 Readout

- outer LOSO by patient；
- ridge `alpha=1.0` 固定；
- 输入标准化仅使用 outer-training patients；
- target 在患者内 median/MAD 标准化；
- 每位训练患者总 sample weight 相等；
- heldout 患者不校准 intercept，不重新训练 RNN；
- 每个 representation 独立走同一 LOSO。

每位 heldout 患者的指标是 contact-level Spearman rho。主 null 为患者内 coherent
all-contact target-label permutation，`n_perm=5000`、固定 seed。within-shaft null
只作解剖敏感性。

## 6. 统计与 gates

所有 cohort inference 以患者为单位，报告 median、95% bootstrap CI、正效应患者数、
one-sided Wilcoxon；同一 family 做 BH-FDR。

### Gate S：source-free static readout

`full_fixed_axis` 相对 all-contact permutation 同时满足：

- eligible patients ≥8；
- median rho margin >0；
- bootstrap CI lower >0；
- 超过半数患者为正；
- BH-FDR q<0.05。

### Gate H：history contribution

`rho(full_fixed_axis) > rho(no_history)` 满足同一方向性患者级标准。失败时只能写间期
节点分布与发作场相关，不能说 recurrent history 对跨状态读出有贡献。

### Gate X：axis contribution

`rho(full_fixed_axis) > rho(local_isotropic)` 为 secondary。失败不否定 Gate S/H，
只禁止将 static readout 解释成 physical-axis contribution。

### Gate A：axis-positive construct validity

n=9 中：

- RNN-selected axis alignment margin 的 cohort median >0；
- selected-axis heldout NLL 相对 isotropic 的 median benefit >0；
- 两者方向一致。

因本分析在 cohort-wide axis failure 后启动，Gate A 只能作为预先存在亚组上的
secondary construct-validity result，不升级为全队列 claim。

## 7. 禁止事项

- 不使用 early-ictal target 选择 RNN axis、模型、特征或 rollout 数；
- 不用 SOZ、patient-level focus、energy-top contacts 或 A/B source 替代临床
  onset source；
- 不把静态场 readout 写成逐秒发作传播、发作预警或 prospective forecasting；
- 不把 axis-positive subgroup 结果外推到22人或34人；
- 不把 contact、seizure、seed 当作 cohort 独立样本；
- Gate H/X 失败时不得只报告 Gate S 阳性并宣称 RNN mechanism。

## 8. 停止规则

- Stage A1 alignment 与 heldout predictive increment 均失败：停止继续扩展 learnable
  physical-axis model；
- Gate S 失败：cross-state static readout 阴性收口，不改 ridge、窗口或频带；
- Gate S 通过但 H 失败：保留 source-free interictal-field readout，删除 RNN-history
  机制措辞；
- Gate S/H 通过但 X 失败：支持 history-dependent cross-state readout，不支持物理轴
  贡献；
- 所有情况下 source-conditioned dynamic rollout 都保持
  `BLOCKED_MISSING_EXACT_CLINICAL_ONSET_SOURCE_METADATA`。

## 9. 执行偏差登记（append-only，不修改 gates）

`rnn_selected_axis` 原计划仅在 n=9 作为 static sensitivity，但 Stage A1 在五类主
representation 已完成 target-blind freeze、且 static target 已按 §5 读取后才结束。
为避免在看过 target 后补做新的 representation，不再追补这一项。正式 static
readout 只包含已在 target 读取前冻结的：

1. `full_fixed_axis`；
2. `no_history`；
3. `local_isotropic`；
4. `node_only`；
5. `empirical_train80`。

这不改变 Gate S/H/X 的定义与结果；`rnn_selected_axis` sensitivity 记为
`NOT_RUN_TO_PRESERVE_TARGET_BLIND_FREEZE`。

资源恢复期间曾短暂尝试在 GPU 上续跑 E958 的未完成 candidate。因为同一32方向
search 不能混合 CPU/GPU 数值路径，所有受影响 candidate 目录均移出 formal tree，
并从 clean CPU state 重训；正式聚合只接受 `resolved_config.device == cpu` 且32/32
candidate 均完整的 run。隔离目录只作工程审计，不进入任何选择或统计。
