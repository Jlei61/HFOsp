# Topic 1 → Topic 4 桥 — propagation-state attractor diagnostics（Step 0 + Step 1）

> **范围**：Topic 1 PR-2 stable_k=2 cluster geometry 上的 attractor-class 诊断（principal curve + GOF + KMeans-axis 夹角 + coordinate-free PR-2 label transition λ₂）。
> **当前状态**：Step 0 audit + Step 1 实测 + sensitivity 验证全部跑完；**未进入 Topic 1 / Topic 4 主文档主结论**（CLAUDE.md §5：sensitivity gates 通过才入主文档；Λ_gap Step 2 未跑）。
> **数据入口**：Topic 1 §3.1d cluster geometry；PR-2 stable_k=2 cohort + lagPat refine。
> **代码**：`src/topic4_attractor_diagnostics.py` + `scripts/audit_topic4_step0.py` + `scripts/run_attractor_step1.py` + `scripts/run_attractor_step1_sensitivity.py` + `scripts/augment_attractor_step1_kmeans_s.py` + `scripts/summarize_attractor_step1.py`。
> **测试**：`tests/test_topic4_attractor_diagnostics.py`（26 tests pass 2026-05-10）。
> **结果**：`results/topic4_attractor/`（per_subject JSON + step0_audit.csv + step1_cohort_summary.csv + step1_sensitivity.csv + 三个 summary MD）。

---

## 1. Locked design

- **Feature space** = `lagPatRank.T`，`NaN → 0.0`，与 PR-2 `compute_kmeans_cluster_stereotypy` 同合同（`src/interictal_propagation.py:1215`）。
  - **NaN→0 把 "rank order + participation pattern" 一同编码进 feature 空间**——这是 PR-2 state 定义的一部分（合同选择），不是单纯的生理量。后续科学解释必须明说。
- **Main cohort gate** = `stable_k = 2` AND `n_participating ≥ 6` after `n_events_eligible ≥ 100`。Step 0 锁定后给出 35 例。
- **PCA-3** → Hastie-Stuetzle smoothing-spline principal curve, 迭代上限 15 次（n>20k 时 stride 子采样）。
- **GOF 硬 gate**：`var_explained_curve > 0.6`，**作用域是 PCA-3 子空间**，非原始高维 X。
- **KMeans axis** = `centroid_1 - centroid_0`，centroid 用 PR-2 `labels` 映射到 Topic 4 eligible 子集后在 X 上重算。**Label 长度 mismatch 直接外排到 `excluded_from_h3_main`**（不用 template fallback），原因记录。

## 2. Step 0 cohort 切分（pre-locked）

总 subject 数：40（期望 20 yuquan + 20 epilepsiae）。

- `stable_k=2`：35 例 → 主分析合规
- `stable_k>2`：5 例 → 平行报，**不进 H3 推论池**
  - yuquan_huangwanling (k=4, n_ch=4), yuquan_zhaojinrui (k=5, n_ch=4),
    yuquan_zhourongxuan (k=5, n_ch=4), yuquan_zhangjinhan (k=6, n_ch=5),
    epilepsiae_818 (k=4, n_ch=5)
  - 全部 n_channels_union ≤ 5，自动落到 `n_participating ≥ 6` 闸门之外。Topic 4 `eligible_for_main = false` 与 PCA-3 几何 ill-posed 两条独立理由都把这部分排除。
- `unknown`：0 例。

Main 子集统计：n_events_eligible median=8371 [148, 165577]；n_blocks_with_events median=64 [9, 405]；n_channels_union median=11 [6, 52]。

## 3. Step 1 实测

进 H3 主分析：**34 / 35**（epilepsiae_1096 因 `pr2_label_event_index_drift` 排除：1 block 删除 + 28 block 事件数漂移，PR-2 label 对应旧 ordering）。

### 3.1 Principal curve — 在 PCA-3 子空间内的拟合

- `var_explained_curve` (in PCA-3): median = 0.953, range [0.565, 0.990]
- PC1 ratio: median = 0.303 [0.171, 0.585]
- top-3 cumulative ratio: median = 0.594 [0.352, 0.858]

**警示**：`var_explained_curve` 的分母是 top-3 PC 子空间方差，**不是原始 X 全方差**。top-3 cumulative ratio median 只占原始 X 总方差约六成，**不能由此结论"原始高维传播态是 1D manifold"**。

- GOF pass：33 / 34；GOF fail 1：epilepsiae_916（var=0.565, top3=0.858, n_ch=6）
- max_iter=15 内严格收敛：1 / 34。其余给最后一次迭代结果（max_iter sensitivity 见 §4）。

### 3.2 Principal curve 切向 vs PR-2 KMeans 主轴 — 单点 (s_median)

- 夹角 (at s_median): median = 83.0°, range [54.4°, 89.6°]
- 共线区 angle < 15°：0 例；非共线区 angle ≥ 30°：34 例

**警示**：当前夹角仅在 `s_median` 取曲线切向，是单点测量。Grid-wide / event-weighted 角度分布见 §4 sensitivity。

### 3.3 s_kmeans — PR-2-label-supervised 1D 投影

**重要 disclaimer**：s_kmeans 用 PR-2 label 算 centroid 再投影回的**监督坐标**。能确认"两个 PR-2 cluster 在 rank 空间可被一条轴分开"，**不能独立证明双稳态**。这部分是 cluster 几何 sanity，不是 H3 evidence。

- |Cohen's d| (per subject, |d_avg|): median = 4.01 [3.21, 6.31]
- midpoint-threshold accuracy: median = 1.000 [0.975, 1.000]

### 3.4 PR-2 label transition sanity — coordinate-free metastability test（**KEY**）

在 within-block 相邻事件对上算 2×2 PR-2 label transition matrix M，λ₂ = trace(M) − 1 ∈ [−1, 1]。λ₂ → 1 = 高 metastability（长 dwell 罕跳）；λ₂ → 0 = 无时间结构；λ₂ < 0 = 反相关（震荡）。Null：within-block label shuffle (n_perm=1000)，保 marginal cluster fraction，破时序。**这套测度不依赖任何 1D 坐标**，是直测 H3 metastable 假设的最朴素方法。

- λ₂ (observed): median = 0.044, range [−0.063, 0.198]
- z_λ₂ vs within-block shuffle: median = 1.2, range [−11.4, 19.7]
- empirical p (right-tail): median = 0.1169, min = 0.0010
- **p < 0.001 且 λ₂ > 0：10 / 34**

完整 per-subject λ₂ 表（34 行）见 `results/topic4_attractor/step1_summary.md` §4 表。

### 3.5 主曲线 vs KMeans 主轴几乎正交 — 现象与解读

**绝大多数 subject 的 principal curve 在 s_median 处切向与 PR-2 KMeans 主轴夹角 ≥ 50°，多数 70°–90°**。
解读（**目前是观察，不是结论**）：principal curve 抓的是 within-cluster 几何（每个 cluster 内部的 1D 延展），不是 cluster-to-cluster 分离方向。

对 H3 的影响：直接把 Λ_gap 跑在 principal curve 的 s 上，测的可能是 within-cluster trajectory 而不是 between-cluster 切换。**Step 2 主路径建议改为：**

1. 先看 §3.4 label transition λ₂ — 不依赖 1D 坐标，是最干净的 H3 直测；
2. 再跑 Λ_gap on `s_kmeans`（K=8 主，{6,10,12} sensitivity）作为 K-bin 加细版本；两者方向必须一致才信。
3. principal curve s 仅作 sensitivity，用来诊断 within-cluster 是否还有独立动力学层。

### 3.6 Cluster size imbalance（Step 3 估 M_label 时需关注）

- min(cluster) / total: median = 0.428, range [0.156, 0.497]

imbalance 越极端，minority-row 的 transition counts 越少，λ₂ 估计噪声越大。Step 3 报告里 `min_row_count` / `zero_row_count` 必须当 covariate 看。

## 4. max_iter & grid/event-wide angle sensitivity

Re-fit principal curve at max_iter ∈ {15, 30, 60} on the 34 main-cohort subjects。

| max_iter | n | converged | var_curve median (range) | angle@s_median | angle_grid | angle_event |
|---:|---:|---:|---|---:|---:|---:|
| 15 | 34 | 1 | 0.953 (0.565–0.990) | 83.0° | 81.0° | 82.7° |
| 30 | 34 | 3 | 0.949 (0.549–0.990) | 79.2° | 77.1° | 78.5° |
| 60 | 34 | 4 | 0.950 (0.634–0.990) | 71.5° | 67.0° | 69.8° |

**关键解读：**

1. **var_explained_curve 稳定**：max_iter 15→60 median 仅从 0.953 漂到 0.950。Main batch 数字可信。
2. **角度对 max_iter 敏感**：angle@s_median median 从 83° → 71.5°（差 ~12°）。"主曲线 ≈ 正交于 KMeans 主轴"的措辞需谨慎：稳健界至少 60°+ 而非 83°。
3. **Grid vs event vs single-point 三者每档差 < 5°**：single-point at s_median 不严重偏离 grid/event-weighted；问题不在采样位置，在主曲线尚未收敛。
4. **收敛差**：max_iter=60 仅 4/34 严格收敛（tol=1e-4）。Hastie-Stuetzle 迭代在 PR-2 rank 空间上行为不稳。某些 subject (e.g. epilepsiae_1077 var 0.658→0.549→0.733) 非单调。**含义**：principal curve s 作为 H3 主路径不可靠；§3.4 的 label-transition λ₂ 路径不依赖 curve 收敛，仍是最稳健的 H3 直测。

## 5. 范围声明与下一步

- **不进 Topic 1 / Topic 4 主文档主结论。** 当前所有数字是 Step 1 + sensitivity，**未跑 Step 2 Λ_gap**。CLAUDE.md §5：所有 sensitivity gates 通过才入主文档主结论。
- **跨 PR 提醒（CLAUDE.md §5 + Cross-PR Contract Lookups）**：本 PR 的 `stable_k=2` cohort 来自 `template_rank` adaptive cluster；下游若复用 `valid_mask`，必须从 raw bools 重新派生（PR-6 valid_mask 语义 caveat 同样适用于这里）。
- **下游路径建议**：
  - **Step 2 主**：label-transition λ₂ + 长 dwell-time 描述统计（dwell distribution / first-passage time）。不依赖任何 1D 坐标。
  - **Step 2 副**：Λ_gap on `s_kmeans` (K=8 主，{6,10,12} sensitivity)。方向必须与 λ₂ 一致才采信。
  - **Step 3 sensitivity**：principal curve s 上的 Λ_gap，仅作 within-cluster trajectory 诊断。
  - **报告口径**：H3 metastable switching 的 falsifiable claim 仅来自 §3.4 + Step 2 主路径；s_kmeans / principal curve 是 sanity / diagnostic。

## 6. 后续可能的归档拆分

- 当 Step 2 / Step 3 跑完，且 sensitivity 全部通过，可将 §3.4 + Step 2 结果在 Topic 1 主文档 §3.1e 开 "Cluster geometry → metastable label transitions" 桥接节，并把本 doc 拆为 step1_results / step2_results 两个独立 archive。
- 当模型层（PR-T4-* BHPN-fit）需要在数据侧给出 metastable label transition 作为 fit target 时，本 doc 升级为 model-data 对接合同的主参考。
