# Topic 5 RNN 审阅后收口与 static scaffold 新 goal 报告

日期：2026-07-28

## 1. 总体判断

审阅意见的主方向成立，但必须用最新 internal-state v0.1 结果更新：

\[
\boxed{\text{可保留：patient-specific static contact correspondence}}
\]

\[
\boxed{\text{可保留：interictal rank order 会改变 GRU state/prediction}}
\]

\[
\boxed{\text{尚未建立：RNN-specific static transfer、physical axis 和 dynamic seizure transfer}}
\]

上一版报告把 ordered-state 和 target-reused early-ictal read-back 合称为 “Tier B+” 过强。
现在已经拆成三层：

1. static scaffold：同一数据集内的 sign-free morphology correspondence；
2. ordered-history：纯间期 matched perturbation 的可靠诊断；
3. internal-state early-ictal read-back：探索性、target-reused，不能作为独立确认。

## 2. 对审阅意见的逐条处理

| 审阅问题 | 最新状态 | 处理 |
|---|---|---|
| 每 seizure 五个 field 取最大 | 已解决 primary | v0.1 固定 participation/endpoint；五场只作旧 omnibus sensitivity |
| absolute correlation 方向不明确 | 已确认是关键缺口 | 新合同固定 positive signed participation，并完成首轮 signed null |
| all-contact null 可能过弱 | 部分解决并扩展 | 已新增 circular、dihedral、equal-size shaft 和 geometry-smooth null |
| full GRU 是否只是平滑器 | 尚未解决 | 新 goal 冻结 beta-binomial、Laplacian、Dirichlet、NMF baseline |
| free rollout 是否制造 shrinkage | 尚未解决 | 新 goal 加入 teacher-forced one-step aggregate |
| contact measurement confounds | 部分可做 | shaft/spacing/geometry/部分 SOZ 可用；baseline power 待长任务 |
| GM/WM、artifact rate | 不可用 | 明确不插补、不宣称已控制 |
| PCA 低维即机制 | 已降格 | 低维性只作描述；rank-shuffle 同样低维 |
| PCA off-manifold perturbation | 已降格 | 不作为 primary evidence |
| matched history test | 已补 | order-shuffle full−rank NLL 中位 0.0100，32/34 为正 |
| target 已多次读取 | 已明确 | 所有 early-ictal 新结果均标记 internal validation |
| structured-axis RNN 是否继续调 | 冻结 | 不再调 axis/source/history 参数 |

## 3. 上一 goal 的修复与验收

### 3.1 图文修复

- internal-state Figure 状态从 `paper_ready_candidate` 改为
  `supplementary_exploratory_candidate`；
- Panel A 明示 interictal directions target-blind，但 ictal residual read-back
  exploratory；
- Panel E 明示 contact 按 ictal energy 排序，只是 illustrative；
- Panel F 明示 target reused；
- archive 报告和 INDEX 不再把 state read-back 写成统一 Tier B+。

### 3.2 机器可读验收

`POSTREVIEW_ACCEPTANCE.json` 的正式状态为：

```text
execution_integrity: PASS
interictal_order_sensitivity: SUPPORTED
static_contact_correspondence: SUPPORTED_WITHIN_REUSED_DATASET
gru_specific_static_increment: NOT_ESTABLISHED
ordered_state_early_ictal_readback: EXPLORATORY_TARGET_REUSED
physical_axis_or_dynamic_seizure_mechanism: NOT_SUPPORTED_BY_CURRENT_MODEL_FAMILY
```

这里最后一项只否定当前模型归因，不否定患者可能存在病理轴。

### 3.3 最新 fixed participation 对照

在旧 absolute scoring 下，full GRU participation：

- absolute rho 中位 0.445；
- all-contact absolute margin 中位 0.214；
- 14/16 为正，P=0.000153。

但 full GRU 的 all-contact margin 相对：

- static contact hazard：中位 0，P=0.240；
- unordered prefix：0.030，P=0.070；
- last-set first-order：0.010，P=0.086；
- rank-shuffle GRU：−0.009，P=0.430。

因此审阅意见中“static scaffold 可以保留、不能归因 ordered GRU”的结论得到最新固定
readout 支持。此前 v2.5 中 full GRU 超过 static 的更强数字受到五场选择合同影响，不能
替代这一固定 participation 比较。

## 4. 新 goal 的输入审计

新合同：

> **Interictal–early-ictal static scaffold fixed-readout validation v0.1**

metadata-only audit 没有读取 target 数值，得到：

| 输入 | 可用分母 |
|---|---:|
| strict clinical-onset | 16 人 / 106 seizures |
| 现有六模型 participation fields | 16/16 |
| within-shaft circular | 16/16 |
| within-shaft reversal/dihedral | 16/16 |
| equal-size shaft-profile permutation | 2/16 |
| geometry-complete RBF null | 13/16 |
| shaft position / participation support | 16/16 |
| contact spacing 可构建 | 16/16 |
| SOZ labels | 13/16 |
| baseline band power cache | 0/16 |
| GM/WM | 0/16 |
| artifact/rejection rate | 0/16 |

因此等长 shaft swap 只能作两人 sensitivity；geometry-smooth 只能作 13 人子集。
all-contact 仍保留为与论文既有结果一致的 primary null，within-shaft/geometry 是强
sensitivity，不使用某一个 sensitivity 结果把整线二元裁决。

## 5. 已启动并完成的 Phase 1：固定 signed participation

### 5.1 合同

- 唯一 field：participation；
- 唯一方向：更高 participation 对应更高 early-ictal energy；
- patient statistic：seizure median 后的 signed Spearman；
- 6 个既有模型；
- 每类适用 null 5,000 coherent draws；
- target 为同一 16 人/106 seizures，因此仍是内部验证。

Phase 1 audit：

```text
status: PASS
rows: 480
patients: 16
models: 6
old/new signed rho max difference: 0
```

### 5.2 full GRU 的结果

| 统计 | n | 中位数 | 95% CI | 正向患者 | P |
|---|---:|---:|---:|---:|---:|
| observed signed rho | 16 | 0.237 | [−0.322, 0.504] | 11/16 | 0.126 |
| all-contact signed margin | 16 | 0.243 | [−0.322, 0.515] | 11/16 | 0.126 |
| within-shaft circular margin | 16 | 0.113 | [−0.119, 0.253] | 11/16 | 0.163 |
| within-shaft dihedral margin | 16 | 0.108 | [−0.075, 0.275] | 11/16 | 0.150 |
| geometry-smooth margin | 13 | 0.241 | [−0.400, 0.762] | 9/13 | 0.122 |
| all-contact absolute margin | 16 | 0.215 | [0.129, 0.351] | 14/16 | 0.000153 |

### 5.3 这意味着什么

原结果最稳的是：

> interictal participation field 与 early-ictal energy 存在 sign-free spatial
> morphology correspondence。

当前尚不能写：

> 高 interictal participation 的 contact 在全队列中一致具有更高 early-ictal energy。

5/16 患者的 signed rho 为负，另有患者接近 0；这不是简单降低效应量，而是明确的方向
异质性。absolute correlation 把正向和反向患者都计为 morphology match，因此显著性
明显更强。

### 5.4 full GRU 是否优于其他模型

在 all-contact signed margin 上：

- full − static：中位 0；
- full − empirical：0.013；
- full − unordered：0.010；
- full − last-set：0.014；
- full − rank-shuffle：0.065，未校正 P=0.0168，family FDR q=0.0838。

在 geometry-smooth 子集中，full − rank-shuffle 中位 0.083，FDR q=0.0537。

这些是边界性趋势，不足以称为 GRU-specific static transfer。它们值得在预定义的
regularized non-recurrent baselines 下继续量化，但不能靠再增加 seeds 升格。

## 6. 当前最可靠的整体科学结论

### 可以进入论文叙事

1. 间期 rank-set 序列含有稳定、患者特异的 contact-level scaffold；
2. GRU 的间期预测对真实 rank 顺序敏感，而不只是 contact membership；
3. interictal scaffold 与 early-ictal broadband energy 存在 sign-free static
   morphology correspondence；
4. 该 correspondence 目前更接近 patient-specific static scaffold readout，而不是
   directional replay 或 dynamic seizure prediction。

### 不能进入主结论

1. RNN 比正则化非递归 estimator 更好；
2. ordered long history 是 early-ictal transfer 的来源；
3. PC1/PC2 是患者病理轴或 E/I latent variables；
4. 当前模型预测了 clinical onset、发作传播顺序或个体 seizure；
5. static correspondence 已排除 baseline power、GM/WM、artifact sensitivity。

## 7. 新 goal 的剩余执行路线

### 下一步一：正则化非递归 baselines

在完全 target-free 的 interictal validation 中选择：

- beta-binomial participation shrinkage；
- shaft/geometry Laplacian smoothing；
- Dirichlet contact×rank histogram；
- NMF rank 1–4。

这一步回答 full GRU 是不是只相当于平滑器。

### 下一步二：free rollout vs teacher forcing

导出相同 heldout20 prefixes 的 teacher-forced one-step aggregate，与 empirical、
smoothed 和 free rollout 场比较。

### 下一步三：confound 层

- 快速层：shaft position、spacing/local density、geometry PCs、SOZ；
- baseline power：独立长任务；
- GM/WM、artifact rate：当前不可用，保留 limitation。

### 下一步四：方向异质性

不允许用 ictal target 逐患者翻转 sign。优先检查负向患者是否在预先存在的、target-free
变量上形成可解释分层，例如 participation field 的可靠性、contact denominator、
shaft geometry 和 interictal model calibration。若不能解释，结论保持 sign-free。

## 8. 当前状态

上一 goal 已完成并通过审阅后科学重构。新 goal 已创建，spec、plan、input audit 和
Phase 1 signed/null ladder 已完成；regularized baselines、teacher-forced decomposition
与 confound 层仍在进行中。

主要产物：

- `results/topic5_rnn_internal_state_reduction/POSTREVIEW_ACCEPTANCE.json`
- `results/topic5_static_scaffold_fixed_readout_validation/INPUT_AUDIT.json`
- `results/topic5_static_scaffold_fixed_readout_validation/PHASE1_EXISTING_FIELDS_SUMMARY.json`
- `results/topic5_static_scaffold_fixed_readout_validation/PHASE1_AUDIT.json`
- `docs/superpowers/specs/2026-07-28-topic5-static-scaffold-fixed-readout-validation-v0_1.md`
- `docs/superpowers/plans/2026-07-28-topic5-static-scaffold-fixed-readout-validation-v0_1.md`
