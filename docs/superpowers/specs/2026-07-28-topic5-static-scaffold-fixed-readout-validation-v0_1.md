# Topic 5 interictal–early-ictal static scaffold fixed-readout validation v0.1

## 1. 科学对象

本合同不再问 RNN 是否自动恢复患者病理轴，也不把 early-ictal 结果解释成发作传播预测。
它只回答：

> 在不逐 seizure 选择最佳 field、固定场方向、支付空间平滑代价，并与正则化非递归
> estimator 公平比较后，间期事件是否仍提供可迁移到 clinical-onset 后早期宽频能量的
> 患者特异 static contact scaffold？

该 target 已在 v2.5 和 internal-state v0.1 中读取。本合同属于同一数据集上的严格内部
验证，不是独立 confirmatory replication。

## 2. 冻结队列和 target

- 数据集：Epilepsiae；
- anchor：逐发作 clinical onset；
- cohort：16 人、106 次发作；
- target：clinical onset 后 `[0,10] s` 的 `1–150 Hz` baseline-normalized
  contact energy；
- 每次发作至少 6 个 exact-joined contacts；
- seizure 先折叠、seed 再折叠、最后 patient-first inference；
- Yuquan EEG onset 不进入 primary；
- 不使用 ictal target 选择模型、平滑强度、field、方向或 confound 组合。

## 3. 唯一 primary readout

每个患者输出一个在任何 seizure target 未参与选择的固定场：

\[
\widehat f_i=P(i\ {\rm participates\ in\ an\ interictal\ event}).
\]

对每次 seizure 的 primary score 为预先定向的 signed Spearman：

\[
\rho_{p,s}=
\operatorname{Spearman}
\left(
\widehat f_p,
E^{\rm ictal}_{p,s}
\right).
\]

方向固定为正：更高的间期 participation 预期对应更高的 early-ictal energy。
`abs(rho)` 只作为 morphology sensitivity，不能替代 primary。

以下 field 降为 secondary，不允许逐 seizure 五选一：

- early joint mass；
- late joint mass；
- endpoint mass；
- weighted earliness。

## 4. 模型与非递归基线

### 4.1 原有冻结模型

1. raw empirical train80 participation；
2. static contact hazard；
3. unordered-prefix；
4. last-set first-order；
5. rank-shuffle GRU；
6. full-history GRU。

### 4.2 新增 target-free regularized baselines

1. **Beta-binomial participation shrinkage**：向患者内总体参与率收缩；
2. **shaft/geometry graph-Laplacian smoothing**：只平滑 participation；
3. **Dirichlet-smoothed contact×rank histogram**；
4. **low-rank non-recurrent contact×rank estimator**，候选 rank `1–4`；
5. **teacher-forced one-step aggregate field**，与现有 free-rollout field 配对。

所有 shrinkage、Laplacian penalty、Dirichlet concentration 和低秩维数只允许根据
interictal `train60/validation20` 的 heldout event likelihood/Brier 选择。禁止根据
early-ictal similarity 选择。

## 5. 空间 null 层级

### 5.1 论文兼容 primary null

5,000 次患者内 coherent all-contact permutation。每个 draw 对该患者所有 seizures
共享同一 contact permutation，然后 seizure-first 折叠。这一层与既有论文主统计一致。

### 5.2 强空间 sensitivity

1. **within-shaft circular shift**：16/16 患者可用；
2. **within-shaft reversal/random dihedral transform**：16/16 可用；
3. **equal-size shaft-profile permutation**：仅 2/16 可用，只作病例/可用子集敏感性，
   不形成全队列门；
4. **geometry-smooth surrogate**：13/16 坐标完整患者；RBF length scale 只从
   interictal field/geometry 估计，surrogate 做 rank matching，不读取 target。

各 null 独立报告，不把某一个 sensitivity 阴性转换成整个 static scaffold 的 no-go。

## 6. Free rollout 与 teacher forcing

现有 free rollout field 可能产生平滑和 shrinkage。必须并列导出：

1. empirical train80 participation；
2. free-rollout participation；
3. teacher-forced heldout20 one-step probability aggregate；
4. target-free smoothed empirical participation。

报告 field 间的 patient-wise Spearman、effective degrees of freedom 和相对 early-ictal
signed margin。若 GRU 与简单平滑场近似相同，结论应落在 regularized static scaffold，
而不是 ordered recurrent mechanism。

## 7. Contact confound 分层

### Tier 1：当前可直接完成

- shaft identity / within-shaft position；
- contact spacing 与局部 contact density；
- 3D coordinates（完整者 13/16）；
- raw interictal participation/HFO support；
- SOZ indicator（可用者 13/16）。

注意：raw participation 是本合同的 predictor-of-interest，不能在 static scaffold 主分析
中被当作 nuisance 回归掉。它只用于检验 GRU 是否提供超出 raw/smoothed participation
的增量。

### Tier 2：需单独构建

- baseline band power：复用 `build_topic5_v2_confound_maps.py` 从原始 seizure baseline
  重建，当前 0/16 已缓存；
- 只在 exact contact join 和 coverage 达标患者中报告。

### 当前不可用

- GM/WM label；
- artifact/rejection rate。

不得插补、从 energy 反推或宣称已控制。最终 wording 必须把这些列为未排除的 measurement
confounds。

## 8. 统计与证据分层

每个结果报告：

- patient median；
- bootstrap 95% CI；
- 正向患者数；
- paired Wilcoxon；
- 同一预定义 family 内 BH-FDR；
- null empirical P；
- 可用患者和排除原因。

不输出单一 hard `GO/NO-GO`。按独立 claim 分层：

### Claim S1：static scaffold correspondence

固定 signed participation 高于 all-contact null；within-shaft 和 geometry-smooth 作为强
敏感性决定措辞强度。

### Claim S2：GRU-specific increment

full GRU 是否超过 target-free 选择的最强 regularized non-recurrent baseline、
first-order 和 rank-shuffle。

### Claim S3：free-rollout contribution

free rollout 是否提供超出 teacher-forced aggregate 和简单平滑的增量。

### Claim S4：confound robustness

在可用 covariate/subcohort 中，signed correspondence 是否保留。每种 confound 单独报告，
不将缺失 covariate 当作阴性结果。

## 9. 与 RNN internal-state v0.1 的关系

- matched order perturbation 是独立的 interictal ordered-history 诊断，可保留；
- PCA effective rank、off-manifold direction perturbation 不作为 static scaffold 证据；
- participation-residualized PC1/PC2 early-ictal read-back 保留为 target-reused
  exploratory sensitivity；
- structured-axis RNN 冻结，不继续调 axis/source/history 参数；
- 只有 Claim S2/S3 显示 ordered recurrent model 的独立增量，才启动新的
  matched-prefix state-swap / on-manifold dynamics goal。
