# Topic 5.2D v0.2 技术收口

> 日期：2026-08-18　|　spec：`docs/superpowers/specs/2026-08-17-topic5-capacity-constrained-structural-identifiability-v0-2-design.md`
> plan：`docs/superpowers/plans/2026-08-17-topic5-capacity-constrained-structural-identifiability-v0-2.md`
> 结果根：`results/topic5_capacity_constrained_history_motif_v0_2/`

## 1. 模型与目标

```text
prefix state      z_{e,q+1} = F_m z_{e,q} + B_m^T x_{e,q},  z in R^r,  r in {1,2,4,8}
encoder/readout   B_m = Q_m C^in_m,   R_m = Q_m C^out_m       (Q_m column-orthonormal)
direct family     dl_{e,t,h} = R_{m,h} z_{e,t}                (horizon-specific readout)
autonomous family z_{e,t+h} = F_m^h z_{e,t},  dl_{e,t,h} = R_m z_{e,t+h}
orderless bag     z^bag_{e,t} = C_bag^T Q^align^T S_{e,t}     (no F, no rank order)
unordered base    l^base_{e,t,h} = b_h + U_h V_h^T a_{e,t}
  a^min  = [x_{e,1}, t, |S_{e,t}|/C]
  a^full = [x_{e,1}, S_{e,t}, t, |S_{e,t}|/C]
loss              L_h = -log p(n_{e,t+h}) - log p(S_{e,t+h} | n, l_{e,t,h})
exact subset law  p(S|n) = prod_{i in S} w_i / e_n(w_available),  w = exp(l)
checkpoint        L_space = sum_{h in 1,2,3} w_h L_h + lambda_f L_suffix   (STOP excluded)
autonomous suffix f^{suffix,5} = 1 - prod_{h=1..5} (1 - p_h)   (prefix-only no-repeat mask)
```

冻结常数：`CHECKPOINT_HORIZONS=(1,2,3)`，`w_h=1/3`，`lambda_f=1.0`，训练期每个 horizon 权重 1/5，suffix 权重 1.0；这些是设计常数，对每条臂完全相同，不做任何按臂调参。

## 2. 输入、split 与防泄漏

- SEEG 复用 parent 的 28 人 `GEOMETRY_ONLY_PCA2` cache；split 逐位一致：`True`；`split == -1` 与 parent held-out 完全相同：`True`
- 25% ⊂ 50% ⊂ 100% 严格嵌套且按 recording block × 事件长度分层：`True`
- 候选集合约定：`teacher_forced_no_repeat_per_horizon`；suffix 掩码约定：`prefix_only_no_repeat`

## 3. 单元分母

- 预注册单元 3892 个，合格 3527 个，不合格 365 个：{'RANK_EXCEEDS_BASIS_DIMENSION': 241, 'ANGLE_NULL_INELIGIBLE': 124}

| block | planned | eligible |
|---|---:|---:|
| CAPACITY | 504 | 437 |
| CORE1 | 1484 | 1374 |
| CORE2 | 476 | 436 |
| F_FORM_SENSITIVITY | 336 | 296 |
| LEARNING | 728 | 656 |
| PREFIX2_SENSITIVITY | 168 | 150 |
| TIME_PROXY | 196 | 178 |

计划偏离（均在任何结果之前决定并冻结）：

- CORE2 direct family runs all four angle nulls (plan §F1 sketched one) so the patient-median angle null is a median in both families
- TIME_PROXY runs four angle nulls for the same reason
- LEARNING shares the geometry/shaft/free units between the end-to-end and fixed-basis curves because those bases do not depend on the training fraction
- spec §4.10 transition-form sensitivities and spec §5.1 prefix=2 sensitivity are enumerated here so they are frozen with the rest of the matrix

实际完成：complete 3527，unresolved 0，missing 0，出现非有限批次的单元 0，总训练墙钟 43.1 小时。

每个 horizon 的留出分母（prefix=3，development test）。每格是`总决策数 / 其中真正有空间选择的决策数`——当剩下的候选触点数不多于要选的个数时，那一步是被逼的，精确子集似然恒等于零、不携带空间信息，所以必须和总数一起看：

| patient | h1 | h2 | h3 | h4 | h5 |
|---|---:|---:|---:|---:|---:|
| epilepsiae_1077 | 1961 / 1961 | 859 / 859 | 240 / 0 | 0 | 0 |
| epilepsiae_1084 | 879 / 879 | 879 / 879 | 878 / 878 | 534 / 534 | 293 / 293 |
| epilepsiae_1096 | 16480 / 16480 | 10773 / 10773 | 5248 / 5248 | 1136 / 0 | 0 |
| epilepsiae_1125 | 3929 / 3929 | 2903 / 2903 | 1927 / 1927 | 983 / 0 | 0 |
| epilepsiae_1146 | 1492 / 1492 | 1492 / 1492 | 1492 / 1492 | 1492 / 1492 | 1492 / 1492 |
| epilepsiae_1150 | 1027 / 1027 | 1002 / 1002 | 416 / 416 | 142 / 142 | 40 / 40 |
| epilepsiae_139 | 1095 / 1095 | 491 / 491 | 195 / 195 | 48 / 0 | 0 |
| epilepsiae_253 | 6062 / 6062 | 4165 / 4165 | 2253 / 2253 | 920 / 920 | 135 / 0 |
| epilepsiae_384 | 2580 / 2579 | 2579 / 2578 | 1536 / 1533 | 656 / 655 | 199 / 199 |
| epilepsiae_442 | 320 / 320 | 320 / 320 | 320 / 320 | 320 / 320 | 320 / 320 |
| epilepsiae_548 | 838 / 838 | 837 / 837 | 652 / 652 | 457 / 457 | 303 / 303 |
| epilepsiae_583 | 940 / 940 | 427 / 427 | 157 / 157 | 32 / 0 | 0 |
| epilepsiae_590 | 598 / 598 | 598 / 598 | 595 / 595 | 593 / 593 | 587 / 587 |
| epilepsiae_620 | 2688 / 2688 | 2679 / 2679 | 1860 / 1860 | 1277 / 1277 | 810 / 810 |
| epilepsiae_635 | 328 / 328 | 110 / 110 | 14 / 14 | 2 / 0 | 0 |
| epilepsiae_922 | 10037 / 10037 | 7437 / 7437 | 5179 / 5179 | 2800 / 2800 | 1203 / 0 |
| epilepsiae_958 | 14811 / 14811 | 14811 / 14811 | 14811 / 14811 | 14811 / 14811 | 14802 / 14802 |
| yuquan_chengshuai | 3221 / 3221 | 2188 / 2188 | 1212 / 1212 | 566 / 566 | 170 / 0 |
| yuquan_huanghanwen | 55 / 55 | 52 / 52 | 20 / 20 | 8 / 8 | 3 / 3 |
| yuquan_litengsheng | 77 / 77 | 77 / 77 | 77 / 77 | 77 / 77 | 77 / 77 |
| yuquan_liyouran | 282 / 282 | 282 / 282 | 282 / 282 | 282 / 282 | 282 / 282 |
| yuquan_pengzihang | 2715 / 2715 | 2715 / 2715 | 2714 / 2714 | 1881 / 1881 | 1169 / 1169 |
| yuquan_songzishuo | 54 / 54 | 54 / 54 | 54 / 54 | 54 / 54 | 54 / 54 |
| yuquan_xuxinyi | 800 / 800 | 800 / 800 | 800 / 800 | 800 / 800 | 799 / 799 |
| yuquan_zhangbichen | 1005 / 1005 | 1005 / 1005 | 1005 / 1005 | 1005 / 1005 | 1005 / 1005 |
| yuquan_zhangjiaqi | 5739 / 5739 | 3146 / 3146 | 1529 / 1529 | 546 / 0 | 0 |
| yuquan_zhangkexuan | 954 / 954 | 954 / 954 | 954 / 954 | 954 / 954 | 954 / 954 |
| yuquan_zhaochenxi | 458 / 458 | 458 / 458 | 458 / 458 | 458 / 458 | 458 / 458 |
| E958 | 22841 / 22841 | 22505 / 22505 | 21631 / 21631 | 20183 / 20183 | 18177 / 18177 |
| E1084 | 1272 / 1272 | 897 / 897 | 512 / 512 | 231 / 231 | 82 / 82 |

只有 h1–h3 进入目标函数，那三档的情况是：h1: 30/30 位有分母，全部带空间信息；h2: 30/30 位有分母，全部带空间信息；h3: 30/30 位有分母，其中 1 位空间信息为零（epilepsiae_1077）。

## 3b. 冻结轴估计器的实际行为（判读前置）

`PATIENT_ALIGNED` 的轴按 spec §4.5 step 2 定义为 split-0 起点→late-field 位移外积的主特征向量。该统计量返回位移**二阶矩最大**的方向，因此在狭长植入上会返回contact-cloud 长轴。逐患者实测：

```json
{
 "what_this_measures": "the undirected angle between the axis the aligned basis was built from and two purely geometric axes of the same implantation; 0 deg means the trained axis carries nothing beyond where the electrodes were placed",
 "n_patients": 28,
 "gap_to_contact_cloud_axis_deg": {
  "median": 7.659740282475669,
  "min": 0.14038631492821044,
  "max": 64.15840420376306,
  "n_within_20_deg": 23,
  "n": 28
 },
 "gap_to_dominant_shaft_axis_deg": {
  "median": 28.43500410759615,
  "min": 0.14038631492821044,
  "max": 89.80403246299096,
  "n_within_20_deg": 12,
  "n": 28
 },
 "spearman_aspect_vs_gap_to_cloud": -0.866447728516694,
 "axis_stability_100_vs_25_percent_deg": {
  "median": 0.25591061185275366,
  "max": 20.749862531587652
 },
 "reading": "a small gap does not by itself invalidate the aligned arm — the direction-rotated null is still matched on kernel, anisotropy strength, rank and parameter count — but it does mean the aligned-versus-rotated contrast is largely a contrast between the implantation's long axis and a rotation of it, and it must be reported that way"
}
```

| patient | C | cloud aspect | gap→cloud axis (deg) | gap→dominant shaft (deg) |
|---|---:|---:|---:|---:|
| epilepsiae_139 | 7 | 35.81 | 0.1 | 0.1 |
| yuquan_zhangjiaqi | 7 | 43.49 | 0.3 | 0.3 |
| epilepsiae_922 | 8 | 6.86 | 0.7 | 1.5 |
| yuquan_chengshuai | 8 | 7.26 | 1.0 | 5.5 |
| epilepsiae_620 | 9 | 4.33 | 1.4 | 4.0 |
| epilepsiae_1096 | 7 | 1.52 | 1.5 | 0.3 |
| epilepsiae_1077 | 6 | 7.90 | 2.0 | 77.0 |
| yuquan_pengzihang | 12 | 2.84 | 2.8 | 4.9 |
| yuquan_liyouran | 17 | 2.66 | 3.1 | 73.6 |
| epilepsiae_253 | 8 | 2.24 | 3.5 | 43.1 |
| yuquan_litengsheng | 24 | 2.29 | 5.6 | 83.2 |
| yuquan_songzishuo | 38 | 1.78 | 6.7 | 27.9 |
| epilepsiae_548 | 11 | 1.86 | 7.1 | 73.7 |
| epilepsiae_958 | 16 | 1.71 | 7.3 | 40.9 |
| yuquan_zhangkexuan | 26 | 1.61 | 8.1 | 24.7 |
| epilepsiae_1146 | 15 | 1.41 | 8.5 | 28.9 |
| yuquan_huanghanwen | 10 | 1.53 | 8.7 | 51.6 |
| epilepsiae_1125 | 7 | 1.18 | 9.4 | 10.4 |
| yuquan_xuxinyi | 15 | 2.31 | 10.1 | 29.9 |
| epilepsiae_635 | 7 | 1.34 | 10.5 | 89.8 |
| yuquan_zhangbichen | 52 | 1.44 | 13.0 | 72.2 |
| epilepsiae_1084 | 11 | 1.11 | 14.2 | 60.8 |
| epilepsiae_442 | 15 | 1.61 | 18.2 | 11.3 |
| epilepsiae_590 | 16 | 1.56 | 20.5 | 85.1 |
| epilepsiae_384 | 9 | 1.14 | 24.1 | 56.2 |
| epilepsiae_1150 | 9 | 1.24 | 24.1 | 15.5 |
| epilepsiae_583 | 7 | 1.10 | 46.7 | 17.4 |
| yuquan_zhaochenxi | 26 | 1.05 | 64.2 | 2.2 |

该表不用于加权、排除或修正任何下游结果；它只固定 aligned-vs-rotated 对比的可读范围。方向 null 的匹配度另见 §4（旋转/对齐主奇异值比）。

## 4. Null 匹配实况

- 方向旋转角（弧度）：[0.3491, 0.6981, 1.0472, 1.3963, 1.7453, 2.0944, 2.4435, 2.7925]；合格患者 26/28（近一维几何不补造，也不用其它 null 顶替）
- 触点身份错位 null：每位 4 张，按 (shaft, 径向距离, degree) 分箱内置换，正交性与奇异值逐位保留
- 局部重连 null：每位 4 张，共 112 张；完全匹配 69 张，标记退化 24 张（近一维链状布局在保度数、保同轴性、保边长的约束下没有第二种接法）

## 5. 证据层逐条

| 层 | n | 中位 | 95% 区间 | 正/负/近零 | Wilcoxon p |
|---|---:|---:|---|---|---:|
| `E0_ceiling_all_ANGLE_ROTATED_AXIS_minus_aligned` | 24 | -0.00152 | [-0.01400, +0.00084] | 7/13/4 | 0.1140 |
| `E0_ceiling_all_GEOMETRY_LAYOUT_minus_aligned` | 26 | -0.07645 | [-0.13942, -0.01701] | 4/22/0 | 0.0001 |
| `E0_ceiling_all_IDENTITY_PERMUTED_minus_aligned` | 26 | +0.00822 | [-0.00001, +0.02976] | 15/7/4 | 0.1034 |
| `E0_ceiling_all_SHAFT_GRADIENT_minus_aligned` | 26 | -0.01216 | [-0.11877, +0.01254] | 11/15/0 | 0.0709 |
| `E0_ceiling_all_TRAIN_ONLY_FREE_PCA_minus_aligned` | 26 | -0.13875 | [-0.23429, -0.10834] | 0/26/0 | 0.0000 |
| `E0_ceiling_informative_ANGLE_ROTATED_AXIS_minus_aligned` | 14 | -0.00070 | [-0.00507, +0.00116] | 4/6/4 | 0.4263 |
| `E0_ceiling_informative_GEOMETRY_LAYOUT_minus_aligned` | 14 | -0.02108 | [-0.08283, -0.00776] | 2/12/0 | 0.0067 |
| `E0_ceiling_informative_IDENTITY_PERMUTED_minus_aligned` | 14 | +0.02772 | [+0.00445, +0.04153] | 10/2/2 | 0.0040 |
| `E0_ceiling_informative_SHAFT_GRADIENT_minus_aligned` | 14 | +0.00066 | [-0.03226, +0.02262] | 7/7/0 | 0.7609 |
| `E0_ceiling_informative_TRAIN_ONLY_FREE_PCA_minus_aligned` | 14 | -0.12029 | [-0.14742, -0.09871] | 0/14/0 | 0.0001 |
| `E1_aligned_ordered_minus_aligned_bag` | 26 | -0.00098 | [-0.00154, +0.00075] | 7/13/6 | 0.7454 |
| `E1_free_low_rank_minus_unordered_baseline` | 28 | +0.01016 | [+0.00454, +0.03734] | 24/2/2 | 0.0000 |
| `E2_direct_minus_autonomous_structure_effect` | 24 | -0.00084 | [-0.00149, +0.00068] | 6/12/6 | 0.5088 |
| `E2_direct_minus_autonomous_structure_effect_common_suffix5` | 24 | +0.00018 | [-0.00154, +0.00104] | 7/11/6 | 0.8115 |
| `E3_aligned_vs_H1_ANGLE_ROTATED_AXIS_AUTONOMOUS_SHARED_OPERATOR` | 24 | +0.00039 | [-0.00025, +0.00175] | 9/6/9 | 0.2405 |
| `E3_aligned_vs_H1_ANGLE_ROTATED_AXIS_DIRECT_HORIZON_UPPER_BOUND` | 24 | +0.00022 | [-0.00047, +0.00078] | 7/4/13 | 0.6033 |
| `E3_aligned_vs_H1_GEOMETRY_LAYOUT_AUTONOMOUS_SHARED_OPERATOR` | 26 | -0.00219 | [-0.01350, +0.00062] | 7/16/3 | 0.0435 |
| `E3_aligned_vs_H1_GEOMETRY_LAYOUT_DIRECT_HORIZON_UPPER_BOUND` | 26 | -0.00204 | [-0.00428, -0.00013] | 4/15/7 | 0.0254 |
| `E3_aligned_vs_H1_IDENTITY_PERMUTED_AUTONOMOUS_SHARED_OPERATOR` | 26 | +0.00164 | [-0.00017, +0.00444] | 14/6/6 | 0.0630 |
| `E3_aligned_vs_H1_IDENTITY_PERMUTED_DIRECT_HORIZON_UPPER_BOUND` | 26 | +0.00050 | [-0.00053, +0.00340] | 11/6/9 | 0.1987 |
| `E3_aligned_vs_H1_LOCALITY_REWIRED_AUTONOMOUS_SHARED_OPERATOR` | 26 | -0.00022 | [-0.00127, +0.00110] | 9/8/9 | 0.9602 |
| `E3_aligned_vs_H1_LOCALITY_REWIRED_DIRECT_HORIZON_UPPER_BOUND` | 26 | +0.00000 | [-0.00020, +0.00027] | 5/5/16 | 0.8791 |
| `E3_aligned_vs_H1_SHAFT_GRADIENT_AUTONOMOUS_SHARED_OPERATOR` | 26 | -0.00045 | [-0.00581, +0.00384] | 11/12/3 | 0.6528 |
| `E3_aligned_vs_H1_SHAFT_GRADIENT_DIRECT_HORIZON_UPPER_BOUND` | 26 | -0.00009 | [-0.00171, +0.00146] | 10/10/6 | 0.6348 |
| `E4_bypass_interaction` | 24 | -0.00058 | [-0.00378, +0.00200] | 9/11/4 | 0.6431 |
| `E4_delta_structure_U_FULL_SET` | 24 | +0.00039 | [-0.00025, +0.00175] | 9/6/9 | 0.2405 |
| `E4_delta_structure_U_MINIMAL` | 24 | -0.00035 | [-0.00241, +0.00198] | 9/10/5 | 0.8115 |
| `E5_capacity_rank1` | 26 | +0.00000 | [-0.00000, +0.00003] | 0/0/26 | 0.3313 |
| `E5_capacity_rank2` | 26 | -0.00002 | [-0.00042, +0.00050] | 6/7/13 | 0.8417 |
| `E5_capacity_rank4` | 24 | +0.00039 | [-0.00025, +0.00175] | 9/6/9 | 0.2405 |
| `E5_capacity_rank8` | 13 | +0.00124 | [-0.00067, +0.00450] | 7/3/3 | 0.1465 |
| `E5_end_to_end_fraction100` | 24 | +0.00039 | [-0.00025, +0.00175] | 9/6/9 | 0.2405 |
| `E5_end_to_end_fraction25` | 24 | +0.00023 | [-0.00034, +0.00157] | 9/3/12 | 0.2182 |
| `E5_end_to_end_fraction50` | 24 | -0.00050 | [-0.00169, +0.00046] | 6/8/10 | 0.2897 |
| `E5_fixed_basis_fraction100` | 24 | +0.00039 | [-0.00025, +0.00175] | 9/6/9 | 0.2405 |
| `E5_fixed_basis_fraction25` | 24 | -0.00025 | [-0.00095, +0.00051] | 7/7/10 | 0.8115 |
| `E5_fixed_basis_fraction50` | 24 | -0.00027 | [-0.00062, +0.00155] | 8/7/9 | 0.9888 |
| `E6_basis_transplant_delta_test_given_A` | 24 | +0.00028 | [+0.00004, +0.00251] | 10/1/13 | 0.0039 |
| `E6_basis_transplant_delta_test_given_N` | 24 | -0.00036 | [-0.00094, -0.00003] | 1/7/16 | 0.0025 |
| `E6_basis_transplant_transplant_interaction` | 24 | +0.00059 | [+0.00017, +0.00280] | 9/1/14 | 0.0014 |
| `E6_ordered_path_ablation_cost_H1_ANGLE_ROTATED_AXIS` | 24 | +0.00157 | [+0.00040, +0.00828] | 13/1/10 | 0.0002 |
| `E6_ordered_path_ablation_cost_H1_FREE_LOW_RANK` | 28 | +0.00476 | [+0.00148, +0.03228] | 21/1/6 | 0.0000 |
| `E6_ordered_path_ablation_cost_H1_PATIENT_ALIGNED` | 26 | +0.00303 | [+0.00053, +0.00989] | 15/0/11 | 0.0000 |
| `E6_prefix_order_cost_H1_ANGLE_ROTATED_AXIS` | 24 | +0.00015 | [+0.00002, +0.00118] | 8/1/15 | 0.0105 |
| `E6_prefix_order_cost_H1_FREE_LOW_RANK` | 28 | +0.00030 | [+0.00011, +0.00108] | 10/0/18 | 0.0000 |
| `E6_prefix_order_cost_H1_PATIENT_ALIGNED` | 26 | +0.00038 | [+0.00005, +0.00155] | 10/1/15 | 0.0020 |
| `E7_endpoint_endpoint_distance_mm` | 24 | -0.00098 | [-0.00503, +0.00093] | 7/10/7 | 0.3596 |
| `E7_endpoint_suffix_balanced_bce` | 24 | +0.00022 | [-0.00002, +0.00079] | 6/2/16 | 0.0604 |
| `E7_endpoint_total_nll_h1` | 24 | -0.00062 | [-0.00183, +0.00035] | 7/10/7 | 0.2405 |
| `E7_endpoint_total_nll_h2` | 24 | +0.00002 | [-0.00070, +0.00080] | 6/6/12 | 0.9888 |
| `E7_endpoint_total_nll_h3` | 24 | +0.00087 | [-0.00033, +0.00202] | 11/3/10 | 0.0691 |
| `E7_endpoint_total_nll_h4` | 24 | -0.00006 | [-0.00035, +0.00012] | 4/3/17 | 0.7151 |
| `E7_endpoint_total_nll_h5` | 21 | +0.00000 | [-0.00020, +0.00013] | 2/3/16 | 0.6475 |
| `E7_spectral_centroid_latency_proxy` | 24 | -0.00021 | [-0.00108, +0.00196] | 10/8/6 | 0.7257 |

**`E1_aligned_ordered_minus_aligned_bag_seed_aware`**（描述性，非队列统计）：

```json
{
 "label": "same contrast, resampling patients and both arms' training seeds",
 "n": 26,
 "median": -0.0009815685950773645,
 "median_ci95_seed_aware": [
  -0.0015774363160272475,
  0.0011690727967366144
 ],
 "crosses_zero": true,
 "n_runs_median": 3.0,
 "note": "resamples patients and, within each drawn patient, one training run; compare against median_ci95 in the matching patient-only entry"
}
```

**`E1_free_low_rank_minus_unordered_baseline_seed_aware`**（描述性，非队列统计）：

```json
{
 "label": "same contrast, resampling patients and the free arm's training seed",
 "n": 28,
 "median": 0.01015634268523069,
 "median_ci95_seed_aware": [
  0.004039345546024142,
  0.03984682425721675
 ],
 "crosses_zero": false,
 "n_runs_median": 3.0,
 "note": "resamples patients and, within each drawn patient, one training run; compare against median_ci95 in the matching patient-only entry"
}
```

**`E3_aligned_vs_H1_ANGLE_ROTATED_AXIS_AUTONOMOUS_SHARED_OPERATOR_seed_aware`**（描述性，非队列统计）：

```json
{
 "label": "H1_ANGLE_ROTATED_AXIS minus aligned (AUTONOMOUS_SHARED_OPERATOR), patients and seeds resampled together",
 "n": 24,
 "median": 0.0006340689071662187,
 "median_ci95_seed_aware": [
  -0.0009174116007960975,
  0.0018528989017060542
 ],
 "crosses_zero": true,
 "n_runs_median": 24.0,
 "note": "resamples patients and, within each drawn patient, one training run; compare against median_ci95 in the matching patient-only entry"
}
```

**`E3_aligned_vs_H1_ANGLE_ROTATED_AXIS_DIRECT_HORIZON_UPPER_BOUND_seed_aware`**（描述性，非队列统计）：

```json
{
 "label": "H1_ANGLE_ROTATED_AXIS minus aligned (DIRECT_HORIZON_UPPER_BOUND), patients and seeds resampled together",
 "n": 24,
 "median": 2.5440963001166494e-05,
 "median_ci95_seed_aware": [
  -0.0007647447088326264,
  0.0007805571553875945
 ],
 "crosses_zero": true,
 "n_runs_median": 24.0,
 "note": "resamples patients and, within each drawn patient, one training run; compare against median_ci95 in the matching patient-only entry"
}
```

**`E6_ordered_path_ablation_cost_H1_ANGLE_ROTATED_AXIS_seed_aware`**（描述性，非队列统计）：

```json
{
 "label": "ordered_path_ablation_cost_suffix_balanced_bce (H1_ANGLE_ROTATED_AXIS), patients and seeds resampled together",
 "n": 24,
 "median": 0.0015671550954856,
 "median_ci95_seed_aware": [
  0.0002430405952360475,
  0.006421994949986987
 ],
 "crosses_zero": false,
 "n_runs_median": 8.0,
 "note": "resamples patients and, within each drawn patient, one training run; compare against median_ci95 in the matching patient-only entry"
}
```

**`E6_ordered_path_ablation_cost_H1_FREE_LOW_RANK_seed_aware`**（描述性，非队列统计）：

```json
{
 "label": "ordered_path_ablation_cost_suffix_balanced_bce (H1_FREE_LOW_RANK), patients and seeds resampled together",
 "n": 28,
 "median": 0.0047606751085325005,
 "median_ci95_seed_aware": [
  0.0012828353737206499,
  0.0358532406698463
 ],
 "crosses_zero": false,
 "n_runs_median": 3.0,
 "note": "resamples patients and, within each drawn patient, one training run; compare against median_ci95 in the matching patient-only entry"
}
```

**`E6_ordered_path_ablation_cost_H1_PATIENT_ALIGNED_seed_aware`**（描述性，非队列统计）：

```json
{
 "label": "ordered_path_ablation_cost_suffix_balanced_bce (H1_PATIENT_ALIGNED), patients and seeds resampled together",
 "n": 26,
 "median": 0.00302648544311525,
 "median_ci95_seed_aware": [
  0.0005039278239659312,
  0.0098907283200619
 ],
 "crosses_zero": false,
 "n_runs_median": 3.0,
 "note": "resamples patients and, within each drawn patient, one training run; compare against median_ci95 in the matching patient-only entry"
}
```

**`E6_prefix_order_cost_H1_ANGLE_ROTATED_AXIS_seed_aware`**（描述性，非队列统计）：

```json
{
 "label": "prefix_order_cost_suffix_balanced_bce (H1_ANGLE_ROTATED_AXIS), patients and seeds resampled together",
 "n": 24,
 "median": 0.00015031236529137844,
 "median_ci95_seed_aware": [
  -5.142925865897485e-06,
  0.001079381045294
 ],
 "crosses_zero": true,
 "n_runs_median": 8.0,
 "note": "resamples patients and, within each drawn patient, one training run; compare against median_ci95 in the matching patient-only entry"
}
```

**`E6_prefix_order_cost_H1_FREE_LOW_RANK_seed_aware`**（描述性，非队列统计）：

```json
{
 "label": "prefix_order_cost_suffix_balanced_bce (H1_FREE_LOW_RANK), patients and seeds resampled together",
 "n": 28,
 "median": 0.00030366629801605,
 "median_ci95_seed_aware": [
  9.056036609035444e-05,
  0.00096468741299165
 ],
 "crosses_zero": false,
 "n_runs_median": 3.0,
 "note": "resamples patients and, within each drawn patient, one training run; compare against median_ci95 in the matching patient-only entry"
}
```

**`E6_prefix_order_cost_H1_PATIENT_ALIGNED_seed_aware`**（描述性，非队列统计）：

```json
{
 "label": "prefix_order_cost_suffix_balanced_bce (H1_PATIENT_ALIGNED), patients and seeds resampled together",
 "n": 26,
 "median": 0.00038344709108995,
 "median_ci95_seed_aware": [
  5.301803984686382e-05,
  0.0013656994653126
 ],
 "crosses_zero": false,
 "n_runs_median": 3.0,
 "note": "resamples patients and, within each drawn patient, one training run; compare against median_ci95 in the matching patient-only entry"
}
```

**`E8_coverage_descriptor_associations`**（描述性，非队列统计）：

```json
{
 "label": "exploratory association between coverage descriptors and the structure effect; descriptive only, never causal and never used to exclude a patient",
 "n": 24,
 "associations": {
  "n_contacts": {
   "n": 24,
   "spearman_rho": 0.05282734806933845,
   "spearman_p": 0.8063362202779776
  },
  "n_shafts": {
   "n": 24,
   "spearman_rho": 0.19909767225441252,
   "spearman_p": 0.350979605342437
  },
  "geometry_effective_dimension": {
   "n": 24,
   "spearman_rho": -0.027826086956521737,
   "spearman_p": 0.8973048186218164
  },
  "ratio_second_to_first": {
   "n": 24,
   "spearman_rho": 0.2860869565217391,
   "spearman_p": 0.17534039388346917
  },
  "recorded_SOZ_annotation_fraction": {
   "n": 21,
   "spearman_rho": -0.2285910174662676,
   "spearman_p": 0.31892478849516015
  },
  "n_recording_blocks": {
   "n": 24,
   "spearman_rho": 0.1675735486090396,
   "spearman_p": 0.4338190703594407
  }
 }
}
```

种子噪声底：218 条多种子臂，中位离散 0.004263239769132454，九成分位 0.023628758488525296。所有效应量必须对着它读。

## 6. Model-unseen 紧凑确认

- 锁定组合：{"block": "CORE1", "rank": 4, "data_fraction": 100, "basis_fraction": 100, "baseline_level": "U_FULL_SET", "f_form": "FULL", "prefix_len": 3, "time_head": false, "structures": ["H1_PATIENT_ALIGNED", "H1_ANGLE_ROTATED_AXIS", "H1_FREE_LOW_RANK"], "families": ["AUTONOMOUS_SHARED_OPERATOR", "DIRECT_HORIZON_UPPER_BOUND"]}
- 访问单元 708，锁外拒绝 2819
- 方向对照分母：26
- 结构效应：{"n": 24, "median": 0.0012505007338782237, "n_positive": 17, "n_negative": 7, "median_ci95": [0.0002751574179993277, 0.002230150394719299], "crosses_zero": false}
- 自由低维 vs 强抄近路：{"n": 28, "median": 0.008994136724351565, "n_positive": 20, "n_negative": 8, "median_ci95": [0.0004159113771715628, 0.02587041845256144], "crosses_zero": false}

## 7. 合成可辨识面

**实测结论（先读这一段，再读设计意图）**：功效块每格 24 个 montage 的二项检验**没有任何一格显著偏离掷硬币**，包括把 teacher 真轴直接交给 student、且事件以最强强度沿该轴推进的那一格（14/24, p=0.541）。该流程对已知强沿轴结构的检出力在本设计下未被建立，真实数据的结构层结果只能读作 uninformative，不能读作 negative。逐格数字与二项 p 见白话版同名小节。

S0 的 oracle-axis 臂把两种失败分开：同一 teacher / 同一数据 / 同一机器，只把 student 的轴从 spec 估计器换成 teacher 真轴。若 oracle 臂恢复而估计臂不恢复，瓶颈是轴估计器而非模型族或损失。该臂只存在于合成块，真实数据永远没有真轴。

- 格子数 237，失败 0，跳过 0
- 角色：calibrates how a real negative should be read; never a gate

```json
{
 "S0_correctness": {
  "S0_aligned_strong": {
   "n": 3,
   "P_aligned_beats_angle_null_autonomous": 0.3333333333333333,
   "P_aligned_beats_angle_null_direct": 0.6666666666666666,
   "median_structure_effect_autonomous": -0.0007157784921147936,
   "median_prefix_order_cost": 0.012125730161313797,
   "median_ordered_path_ablation_cost": 0.014747563114872797,
   "median_bypass_interaction": 0.012410204285475901
  },
  "S0_aligned_strong_high_bypass": {
   "n": 3,
   "P_aligned_beats_angle_null_autonomous": 1.0,
   "P_aligned_beats_angle_null_direct": 1.0,
   "median_structure_effect_autonomous": 0.0018524925379392876,
   "median_prefix_order_cost": 0.013418683259072473,
   "median_ordered_path_ablation_cost": 0.031489896421079466,
   "median_bypass_interaction": 0.0003711813467519587
  },
  "S0_aligned_strong_high_bypass_oracle_axis": {
   "n": 3,
   "P_aligned_beats_angle_null_autonomous": 0.3333333333333333,
   "P_aligned_beats_angle_null_direct": 0.6666666666666666,
   "median_structure_effect_autonomous": -0.00410158552692641,
   "median_prefix_order_cost": 0.0213045663739726,
   "median_ordered_path_ablation_cost": 0.03239368580005797,
   "median_bypass_interaction": 0.00421466682530669
  },
  "S0_aligned_strong_oracle_axis": {
   "n": 3,
   "P_aligned_beats_angle_null_autonomous": 0.6666666666666666,
   "P_aligned_beats_angle_null_direct": 0.6666666666666666,
   "median_structure_effect_autonomous": 0.00019606696234797383,
   "median_prefix_order_cost": 0.01758515105357028,
   "median_ordered_path_ablation_cost": 0.017709418402777732,
   "median_bypass_interaction": 0.011630648198510007
  },
  "S0_bypass_only": {
   "n": 3,
   "P_aligned_beats_angle_null_autonomous": 0.3333333333333333,
   "P_aligned_beats_angle_null_direct": 1.0,
   "median_structure_effect_autonomous": -0.0003713590463412242,
   "median_prefix_order_cost": 0.0027239877206308716,
   "median_ordered_path_ablation_cost": 0.02854927760471626,
   "median_bypass_interaction": 0.006489339793616988
  },
  "S0_bypass_only_oracle_axis": {
   "n": 3,
   "P_aligned_beats_angle_null_autonomous": 0.3333333333333333,
   "P_aligned_beats_angle_null_direct": 0.3333333333333333,
   "median_structure_effect_autonomous": -0.005129904570403188,
   "median_prefix_order_cost": 0.00219891018337659,
   "median_ordered_path_ablation_cost": 0.027942228178658324,
   "median_bypass_interaction": 0.004204631381565349
  },
  "S0_effect_zero": {
   "n": 3,
   "P_aligned_beats_angle_null_autonomous": 0.3333333333333333,
   "P_aligned_beats_angle_null_direct": 0.3333333333333333,
   "median_structure_effect_autonomous": -0.0005196239330151897,
   "median_prefix_order_cost": -0.0016933324574677222,
   "median_ordered_path_ablation_cost": -5.393228655270477e-05,
   "median_bypass_interaction": 0.002523531155915304
  },
  "S0_effect_zero_oracle_axis": {
   "n": 3,
   "P_aligned_beats_angle_null_autonomous": 0.6666666666666666,
   "P_aligned_beats_angle_null_direct": 0.6666666666666666,
   "median_structure_
```

## 8. ECoG 构造效度个案

```json
{
 "contract": "topic5_capacity_constrained_history_motif_v0_2_ecog_case_series",
 "captured_utc": "2026-08-18T14:37:01.862105+00:00",
 "reporting_rules": [
  "each subject is reported on its own; no pooled p-value across the pair",
  "E958 positive does not make an ECoG cohort mechanism",
  "E1084 disagreeing does not negate the SEEG cohort",
  "the physical-grid advantage is an observed-grid inductive bias, never a cortical synaptic graph"
 ],
 "microsteps_primary": 2,
 "microsteps_sensitivity": 1,
 "null_graph_indices": [
  0,
  4,
  8,
  12,
  16,
  20,
  24,
  28
 ],
 "subjects": {
  "E958": {
   "n_units": 127,
   "ECOG_CAPACITY": {
    "FREE_SAME_STATE_UPPER_BOUND|OBSERVED_GRID|AUTONOMOUS_SHARED_OPERATOR|U_FULL_SET|100|2": 7.827948359167297,
    "G1|DEGREE_AND_DISTANCE_REWIRED_GRID|AUTONOMOUS_SHARED_OPERATOR|U_FULL_SET|100|2": 7.834304972963858,
    "G1|IDENTITY_PERMUTED_GRID|AUTONOMOUS_SHARED_OPERATOR|U_FULL_SET|100|2": 7.833352875707691,
    "G1|OBSERVED_GRID|AUTONOMOUS_SHARED_OPERATOR|U_FULL_SET|100|2": 7.830315254284688,
    "G3|DEGREE_AND_DISTANCE_REWIRED_GRID|AUTONOMOUS_SHARED_OPERATOR|U_FULL_SET|100|2": 7.8330973819138965,
    "G3|IDENTITY_PERMUTED_GRID|AUTONOMOUS_SHARED_OPERATOR|U_FULL_SET|100|2": 7.826010814150739,
    "G3|OBSERVED_GRID|AUTONOMOUS_SHARED_OPERATOR|U_FULL_SET|100|2": 7.834201399672162
   },
   "ECOG_CORE": {
    "FREE_SAME_STATE_UPPER_BOUND|OBSERVED_GRID|AUTONOMOUS_SHARED_OPERATOR|U_FULL_SET|100|2": 7.827948359167297,
    "FREE_SAME_STATE_UPPER_BOUND|OBSERVED_GRID|AUTONOMOUS_SHARED_OPERATOR|U_MINIMAL|100|2": 8.173150614891227,
    "FREE_SAME_STATE_UPPER_BOUND|OBSERVED_GRID|DIRECT_HORIZON_UPPER_BOUND|U_FULL_SET|100|2": 7.630124684415382,
    "FREE_SAME_STATE_UPPER_BOUND|OBSERVED_GRID|DIRECT_HORIZON_UPPER_BOUND|U_MINIMAL|100|2": 7.7685978744044695,
    "G2|DEGREE_AND_DISTANCE_REWIRED_GRID|AUTONOMOUS_SHARED_OPERATOR|U_FULL_SET|100|2": 7.831871274959774,
    "G2|DEGREE_AND_DISTANCE_REWIRED_GRID|AUTONOMOUS_SHARED_OPERATOR|U_MINIMAL|100|2": 8.223623905911918,
    "G2|DEGREE_AND_DISTANCE_REWIRED_GRID|DIRECT_HORIZON_UPPER_BOUND|U_FULL_SET|100|2": 7.6380466612377695,
    "G2|DEGREE_AND_DISTANCE_REWIRED_GRID|DIRECT_HORIZON_UPPER_BOUND|U_MINIMAL|100|2": 8.01326893668947,
    "G2|IDENTITY_PERMUTED_GRID|AUTONOMOUS_SHARED_OPERATOR|U_FULL_SET|100|2": 7.833532319264995,
    "G2|IDENTITY_PERMUTED_GRID|AUTONOMOUS_SHARED_OPERATOR|U_MINIMAL|100|2": 8.222593656542255,
    "G2|IDENTITY_PERMUTED_GRID|DIRECT_HORIZON_UPPER_BOUND|U_FULL_SET|100|2": 7.638169638645678,
    "G2|IDENTITY_PERMUTED_GRID|DIRECT_HORIZON_UPPER_BOUND|U_MINIMAL|100|2": 8.01319503741379,
    "G2|OBSERVED_GRID|AUTONOMOUS_SHARED_OPERATOR|U_FULL_SET|100|2": 7.82536035328241,
    "G2|OBSERVED_GRID|AUTONOMOUS_SHARED_OPERATOR|U_MINIMAL|100|2": 8.117409005291556,
    "G2|OBSERVED_GRID|DIRECT_HORIZON_UPPER_BOUND|U_FULL_SET|100|2": 7.632162244939293,
    "G2|OBSERVED_GRID|DIRECT_HORIZON_UPPER_BOUND|U_MINIMAL|100|2": 7.879533801134672
   },
   "ECOG_DATA": {
    "G2|DEGREE_AND_DI
```

## 9. 随机 rollout（次级）

- 层级：secondary evaluation; instability here never changes the direct or autonomous held-out results
- 采样器：one split-1 temperature per patient fitted on the frozen unordered baseline, common random numbers across arms
```json
{
 "aligned": {
  "brier_three_step": 0.22904345393180847,
  "brier_five_step": 0.21681328117847443,
  "brier_full_suffix": 0.2589864432811737,
  "endpoint_distance_mm": 13.484539031982422,
  "length_bias": -0.37581270933151245,
  "between_template_cosine": 0.9849627614021301
 },
 "angle_null_median": {
  "brier_three_step": 0.2346801534295082,
  "brier_five_step": 0.21573633700609207,
  "brier_full_suffix": 0.2714732736349106,
  "endpoint_distance_mm": 14.572636604309082,
  "length_bias": -0.3881875276565552,
  "between_template_cosine": 0.9850772023200989
 },
 "free": {
  "brier_three_step": 0.22378437966108322,
  "brier_five_step": 0.2174198254942894,
  "brier_full_suffix": 0.2654673904180527,
  "endpoint_distance_mm": 13.397836685180664,
  "length_bias": 0.08356708288192749,
  "between_template_cosine": 0.9866629838943481
 },
 "unordered_baseline": {
  "brier_three_step": 0.23078545928001404,
  "brier_five_step": 0.21839101612567902,
  "brier_full_suffix": 0.26117604970932007,
  "endpoint_distance_mm": 13.507640838623047,
  "length_bias": -0.36362510919570923,
  "between_template_cosine": 0.9817290902137756
 }
}
```

## 10. 工程与科学合同审计

```json
{
 "contract": "topic5_capacity_constrained_history_motif_v0_2_closeout",
 "captured_utc": "2026-08-18T16:48:36.308304+00:00",
 "units_planned": 3892,
 "units_eligible": 3527,
 "unit_states": {
  "complete": 3527
 },
 "n_unresolved": 0,
 "retries_over_one": 0,
 "total_nonfinite_batches": 0,
 "total_wall_seconds": 155961.3988134861,
 "distinct_source_hashes": [
  "7219657d3911338c"
 ],
 "single_source_hash": true,
 "baseline_units_on_disk": 120,
 "baseline_all_bitwise_order_invariant": true,
 "baseline_bug_injection_detected": true,
 "baseline_units_with_testable_order_group": 60,
 "baseline_units_with_vacuous_order_group": 60,
 "baseline_min_gradient_updates": 248,
 "split_parity_all_pass": true,
 "model_unseen_equals_parent_heldout": true,
 "nested_subsets_all_pass": true,
 "horizon_denominator_rows": 236,
 "background_processes_still_running": 0
}
```

允许措辞：

- a low-dimensional ordered-history basis defined by the patient's training sequences and recording layout
- held-out suffix prediction with fewer state dimensions
- prefix-order perturbation and ordered-path ablation show the increment actually uses rank order (only if both are positive)
- one shared low-dimensional operator can generate several future steps (only if the autonomous family holds)

禁止措辞：

- the patient's true connectome
- the electrodes cover the seizure-onset zone or the propagation network
- a structural negative proves there is no directed propagation in the brain
- one ECoG patient proves a general local cortical mechanism
- train-time advantage equals online necessity
- a test-time swap equals a natural tissue lesion
- a direct-horizon positive equals shared propagation dynamics
- an aligned bag positive equals an ordered-history motif
- the SEEG basis transplant cost equals runtime graph dependence
- the low-dimensional state is an epilepsy-specific neural axis
- this interictal experiment recovers the previously negative seizure reuse
- a non-significant cohort median means every patient is null

科学合同关键项：

```json
{
 "two_baseline_levels_are_distinct_bypasses": {
  "U_MINIMAL": "start rank set + prefix length + recruited fraction + contact intercept",
  "U_FULL_SET": "the above plus the cumulative unordered contact set",
  "neither_reads": [
   "the last rank set",
   "the prefix ordering",
   "prefix centroid displacement",
   "mode labels",
   "anything future"
  ]
 },
 "autonomous_family_units_exposing_a_full_suffix_head": 0,
 "autonomous_family_shares_one_operator_and_one_readout": true,
 "structured_arms_with_mismatched_parameter_counts": [],
 "encoder_and_readout_share_one_frozen_basis": true,
 "orderless_bag_reads_no_rank_order": true,
 "null_families_reported_with_their_actual_matching": {
  "angle_grid_rad": [
   0.3490658503988659,
   0.6981317007977318,
   1.0471975511965976,
   1.3962634015954636,
   1.7453292519943295,
   2.0943951023931953,
   2.443460952792061,
   2.792526803190927
  ],
  "n_identity_nulls": 4,
  "n_rewire_nulls": 4,
  "angle_null_eligible_patients": 26,
  "rewire_nulls_flagged_degenerate": 24,
  "rewire_nulls_fully_matched": 69
 },
 "bases_built_per_rank": {
  "1": 780,
  "2": 780,
  "4": 726,
  "8": 393
 },
 "bases_ineligible_reasons": {
  "RANK_EXCEEDS_BASIS_DIMENSION": 441,
  "ANGLE_NULL_INELIGIBLE": 32
 },
 "stop_is_separate_from_the_spatial_checkpoint": true,
 "seeg_and_ecog_denominators_never_merged": {
  "seeg_root": "results/topic5_capacity_constrained_history_motif_v0_2",
  "ecog_root": "results/topic5_capacity_constrained_history_motif_v0_2/ecog_construct_validity",
  "ecog_matrix_present": true
 },
 "coverage_is_descriptive_only": true,
 "synthetic_is_an_interpretation_range_not_a_gate": true,
 "use_phase_audit_present": true,
 "basis_transplant_present": true,
 "split_minus_one_access_log_present": true
}
```

## 11. 图形审计

```json
{
 "contract": "topic5_capacity_constrained_history_motif_v0_2_figure_qa",
 "captured_utc": "2026-08-18T16:48:36.464279+00:00",
 "accepted_figure6_files_tracked": 24,
 "accepted_figure6_files_changed": [],
 "accepted_figure6_files_added": [],
 "accepted_figure6_untouched": true,
 "supplementary_assets": {
  "panelA_draft_topic5_strict_history_motif_v0_2.png": "280217d0e674e9092dd6ece1a79ae7251d70a3c683640aa539f14cb4b8a32fcc",
  "supp_fig6_topic5_strict_history_motif_v0_2.png": "b7b72fde17c732fca836691e0eb892923cb34f3e01a874e4368b98be66aa286d",
  "panelA_draft_topic5_strict_history_motif_v0_2.pdf": "d13dd2ddc1ab93200cbf0713db7c15ad39a261845e0dcafcd3366beed9d02773",
  "supp_fig6_topic5_strict_history_motif_v0_2.pdf": "c949f61709caf4d47d771c85f01139ee44c61d51e876bf240438b524a886c739",
  "panelA_draft_topic5_strict_history_motif_v0_2.svg": "7b8d6a7465d95154c004103f43f7a77a9bfdc11b25a08f354eec6d325742271f",
  "supp_fig6_topic5_strict_history_motif_v0_2.svg": "3eddf01418393f777e8096a84826cabfe86022e645f4fd07ef1f1bd0f07ec46f"
 },
 "supplementary_has_png_pdf_svg": true,
 "supplementary_readme_present": true,
 "supplementary_source_data_present": true,
 "supplementary_metadata_present": true
}
```

