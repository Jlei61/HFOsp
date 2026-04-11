# 间期群体事件内部传播分析

## 1. 这是什么主题

这份文档只讨论一个问题：**单个间期群体事件内部，不同通道的激活顺序是否稳定、是否刻板、这种刻板性有多少是真实传播，有多少只是 channel identity bias。**

它不再挂在 `event_periodicity` 下面。原因很简单：

- `event_periodicity` 研究的是**群体事件之间**的时间结构（IEI / PSD / rate modulation）
- 本主题研究的是**群体事件内部**的空间时序结构（`lagPatRank / eventsBool / chnNames`）

把这两件事塞在同一套实验编号里，会把数据对象、统计问题和结果目录全搅烂。

## 2. 当前命名与边界

- 旧线名称：`event_periodicity` 里的 `PR-3`
- 新线名称：`interictal group-event internal propagation`
- 当前阶段：**PR-1**

### 核心科学态度

**我们不是在说一个 subject 只存在"一种"刻板的传播时序。**

我们的立场是：不论存在几种时序模式，主要的这几种时序模式在时间上是长时间稳定的（刻板的），因为其反映了癫痫病理网络本身的特征。老论文 Figure 5 已经展示过 E3 (subject 958) 存在 forward/reverse 两种主要模式（KMeans k=2 聚类，模式间 Spearman r = −0.91），这不是 bug 而是 feature。

PR-1 的目标：

1. **Mixture detection** — 确认是否存在多种传播模式（预期：是，与老论文一致）
2. **Per-cluster stereotypy** — 在每种模式内部，刻板性是否高？（这才是真正的检验）
3. **Identity-bias split** — 刻板性里有多少来自 detection ordering vs 真实传播
4. **Legacy MI** — 向后兼容老论文的 Matching Index 指标与置换检验
5. `n_participating` 分层 — 低参与事件是否在污染总体结论

## 3. PR-1 分析合同

### 3.1 mixture screen + cluster decomposition

- 对 sampled pairwise Kendall τ 跑 Hartigan dip test
- 用 `k=2` agglomerative clustering + silhouette 作为几何敏感性检查
- **KMeans(k=2) 聚类**（与老代码 `plotting_figKura_epilepsiae958Cluster.py` 一致）
  - 簇内 mean τ（raw + centered）— 检验每种模式内部的刻板性
  - 簇间 pattern correlation — 两种模式差异多大（E3 的 r = −0.91 是极端 case）
  - 簇大小比例

### 3.2 legacy MI（Matching Index）

- 实现老代码的 `return_hist_mean_rank()` 模板算法（滑窗 top-3-bin 加权均秩）
- 实现 `return_MI_matrixVer()` → 逐事件 MI 分布
- 行内 shuffle 置换检验（200 iter），报告 p 值
- MI ≡ Kendall τ（数学上完全等价），但保留 MI 命名以兼容老论文

### 3.3 identity-bias split

- `raw_tau`
- `centered_tau`
- `bias_fraction = (raw_tau - centered_tau) / raw_tau`

关键点：**centered rank 绝不能把 non-participating 的 0 算进通道均值。**

### 3.4 n_participating stratification

默认 bins：

- `3-4`
- `5-8`
- `9+`

目的不是做漂亮分层图，而是检查低参与事件是否主要贡献离散化噪声。

### 3.5 SOZ diagnostics

- SOZ-participating vs nonSOZ-only τ
- `soz_source_erased` 诊断

最后这个诊断很重要：如果一中心化，SOZ source node 大面积消失，说明 centered metric 在过度校正。

## 4. 代码地图

- `src/interictal_propagation.py`
  - `load_subject_propagation_patterns()`
  - `detect_propagation_mixture()`
  - `compute_centered_rank_tau()`
  - `compute_stereotypy_by_nparticipating()`
  - `compute_source_node_diagnostic()`
  - `compute_propagation_stereotypy()`
  - `run_subject_interictal_propagation_pr1()`
  - `summarize_propagation_cohort()`
- `scripts/run_interictal_propagation.py`
  - 批量生成 per-subject JSON + cohort summary
- `scripts/plot_interictal_propagation.py`
  - 生成 PR-1 cohort robustness 图
- `tests/test_interictal_propagation.py`
  - 覆盖 centered rank、mixture screen、`n_participating` 分层、SOZ source-erasure smoke case

## 5. 输出目录

结果全部写入：

- `results/interictal_propagation/pr1_subject_summary.json`
- `results/interictal_propagation/pr1_cohort_summary.json`
- `results/interictal_propagation/per_subject/`
- `results/interictal_propagation/figures/`

图目录中的 `README.md` 必须用中文解释每张图在看什么。

## 6. 与旧结果的关系

- `results/event_periodicity/phase2/exp5_stereotypy.json` 保留
- 但它现在只算**历史探索版前身**
- 新的正式入口与后续迭代，都以 `results/interictal_propagation/` 为准

## 7. PR-1 结果（2026-04-11，30 subjects verified）

**Status: COMPLETED**

### 7.1 Mixture screen

- **30/30 受试者 dip test p < 0.001** → 所有受试者的 pairwise τ 分布均为多模态
- Silhouette-based backup 未触发（因为严格 dip test 已全面阳性）
- 解读：30/30 多模态**确认**存在多种传播模式（与老论文 Fig5 E3 forward/reverse 观察一致），**不否认**刻板性 — 每种模式内部是否刻板需要簇内分析来回答
- 全体 mean τ = 0.089 之所以低，是因为混合了高内聚对（同模式）和低相关对（跨模式）

### 7.2 Identity-bias split

- Raw mean τ median = **0.089**（range 0.011–0.319）
- Centered mean τ median = **0.023**（range 0.004–0.224）
- **Bias fraction median = 0.652** → 约 65% 的表观刻板性来自通道检测排序偏差
- 2/30 受试者 bias_fraction = 0.000（huangwanling, zhangjinhan）→ 这些受试者的刻板性全部来自真实传播结构

### 7.3 n_participating stratification

| bin | n_subjects | median τ |
|-----|-----------|---------|
| 3-4 | 20 | 0.041 |
| 5-8 | 27 | 0.075 |
| 9+ | 16 | 0.098 |

τ 随参与通道数单调增长。低通道事件（3-4）的 τ 被离散化噪声拉低，但即使 9+ bin 的 median τ 也仅 0.098。

### 7.4 SOZ diagnostics

- 有效 SOZ/nonSOZ 配对：12 对
- SOZ median τ = 0.080，nonSOZ median τ = 0.068
- SOZ > nonSOZ 方向：7/12
- Wilcoxon (greater) p = 0.088 → 未达显著
- **SOZ source erasure**: 仅 3/30（10%）受试者在中心化后丧失 SOZ source node → centering 未过度校正

### 7.5 科学解读（初版，待簇内分析补充后修订）

以上结果表明：

1. **多模态性是普遍的**（30/30），但这与"刻板性"并不矛盾 — 一个 subject 可以有多种刻板模式。
2. **全体 mean τ 低**（0.089）是跨模式混合的数学必然，不能直接作为"刻板性弱"的证据。
3. **Identity bias 占全体 τ 的 65%** — 这个数字在簇内可能会不同，需要簇内重新计算才能下结论。
4. **SOZ 差异边际**（p = 0.088），但 n = 12 过小。
5. **Centering 没有过度校正**（10% SOZ source erasure），这个结论是稳固的。

**关键缺口**：当前分析没有做簇内 τ 计算，因此**既不能确认也不能否认"多种模式各自刻板"**这一核心假设。需要 KMeans 聚类 + 簇内 τ + legacy MI 来补全。

### 7.6 待补充分析

- KMeans(k=2) 簇内 τ — 如果簇内 τ 显著高于全体 τ，则确认多模式刻板性
- Legacy MI + 置换检验 — 向后兼容老论文指标
- 簇间 pattern correlation — 量化模式间差异（E3 式 forward/reverse 是极端情况还是普遍现象？）
- Figure 2 样式可视化 — lagPatRank heatmap + per-channel rank 分布

## 8. 当前状态

- 代码与独立脚本已拆出
- 独立结果目录已建立
- **PR-1 全量完成并验证**（30 subjects，0 errors）
- PR-1 的 cohort 结果以 `pr1_cohort_summary.json` 为准
- 图已生成并通过目视验证：`results/interictal_propagation/figures/pr1_interictal_propagation_robustness.png`

本文件只维护这个主题本身；涉及 IEI / PSD / rate modulation 的内容请回到 `docs/event_periodicity_analysis.md`。
