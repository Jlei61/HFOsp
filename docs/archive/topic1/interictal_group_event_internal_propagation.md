# 间期群体事件内部传播分析

> 状态：历史结果与详细合同文档
> 当前正式入口：`docs/topic1_within_event_dynamics.md`
> 用途：保留内部传播线的详细结果、实验合同、输出目录与阶段性结论；若只想快速获得当前正式口径，请先读正式入口。

## 1. 这是什么主题

这份文档只讨论一个问题：**单个间期群体事件内部，不同通道的激活顺序是否稳定、是否刻板、这种刻板性有多少是真实传播，有多少只是 channel identity bias。**

它不再挂在 `event_periodicity` 下面。原因很简单：

- `event_periodicity` 研究的是**群体事件之间**的时间结构（IEI / PSD / rate modulation）
- 本主题研究的是**群体事件内部**的空间时序结构（`lagPatRank / eventsBool / chnNames`）

把这两件事塞在同一套实验编号里，会把数据对象、统计问题和结果目录全搅烂。

## 2. 当前命名与边界

- 旧线名称：`event_periodicity` 里的 `PR-3`
- 新线名称：`interictal group-event internal propagation`
- 当前阶段：**PR-2.5 已验收**

### 核心科学态度

**我们不是在说一个 subject 只存在"一种"刻板的传播时序。**

我们的工作假设是：不论存在几种时序模式，主要的这几种模式都对应病理网络中的优选传播路径。PR-2 已经证明这些模式在**同一批事件云上**可重复分解，PR-2.5 进一步证明它们在 **split-half / odd-even block** 尺度上总体稳定；但这仍不等于已经回答了 day/night、seizure proximity 或 occupancy 漂移。老论文 Figure 5 已经展示过 E3 (subject 958) 存在 forward/reverse 两种主要模式（KMeans k=2 聚类，模式间 Spearman r = −0.91），这不是 bug 而是 feature。

PR-1 的目标：

1. **Mixture detection** — 确认是否存在多种传播模式（预期：是，与老论文一致）
2. **Per-cluster stereotypy** — 在每种模式内部，刻板性是否高？（这才是真正的检验）
3. **Identity-bias split** — 刻板性里有多少来自 detection ordering vs 真实传播
4. **Legacy MI** — 向后兼容老论文的 Matching Index 指标与置换检验
5. `n_participating` 分层 — 低参与事件是否在污染总体结论

## 3. PR-1 / PR-2 分析合同

### 3.1 mixture screen + cluster decomposition

- 对 sampled pairwise Kendall τ 跑 Hartigan dip test
- 用 `k=2` agglomerative clustering + silhouette 作为几何敏感性检查
- **PR-1 backward-compatible layer**：KMeans(`k=2`) 聚类（与老代码 `plotting_figKura_epilepsiae958Cluster.py` 一致）
  - 簇内 mean τ（raw + centered）— 检验每种模式内部的刻板性
  - 簇间 pattern correlation — 两种模式差异多大（E3 的 r = −0.91 是极端 case）
  - 簇大小比例
- **PR-2 accepted layer**：adaptive k-scan
  - `k=2..8` 扫描
  - multi-seed AMI 稳定性门控
  - 最小簇比例门控
  - `stable_k` 取通过门控的最高 silhouette
  - inter-cluster Spearman matrix + `candidate_forward_reverse`

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
  - `load_subject_propagation_events()` — `start_t` 排序 + 绝对时间重建 + block/event 合同
  - `load_subject_propagation_patterns()` — backward-compatible wrapper
  - `detect_propagation_mixture()`
  - `compute_centered_rank_tau()`
  - `compute_stereotypy_by_nparticipating()`
  - `compute_source_node_diagnostic()`
  - `compute_propagation_stereotypy()`
  - `compute_cluster_stereotypy()` — KMeans(k=2) + 簇内 τ + 簇间 correlation
  - `compute_adaptive_cluster_stereotypy()` — adaptive k-scan + AMI + silhouette + min-fraction gate
  - `build_cluster_templates()` — 从固定标签事件构建 cluster template
  - `assign_events_to_templates()` — 新事件投到固定模板
  - `compute_time_split_reproducibility()` — split-half / odd-even block 模板复现
  - `compute_legacy_mi()` — 老论文 MI 模板算法 + 行内 shuffle 置换检验
  - `run_subject_interictal_propagation_pr1()`
  - `summarize_propagation_cohort()`
- `scripts/run_interictal_propagation.py`
  - 批量生成 per-subject JSON + cohort summary
  - `--pr25`：基于现有 PR-2 JSON 增补跨时间复现层
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

## 7. PR-1 / PR-2 结果（2026-04-12，30 subjects full-batch accepted）

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

### 7.5 Cluster-aware stereotypy（PR-1 backward-compatible layer）

KMeans(k=2) 聚类后的簇内分析：

- **Within-cluster τ median = 0.250** → 从全体 0.089 跳到簇内 0.250，uplift median = +0.096
- **29/30 subject 的 within-cluster τ > overall τ** → 跨模式混合是全体 τ 低的主要原因
- **Inter-cluster Spearman r median = −0.374** → 两种模式整体倾向互补
- **11/30 (37%) 的 r < −0.5** → forward/reverse 双模式不是 958 (E3) 的特例，而是普遍存在
- 958 复现：inter-cluster r = **−0.915**（老论文 r = −0.91），簇比例 48%/52%
- 簇内 τ 最高的 subject：huangwanling (0.402), 1077 (0.401), 922 (0.381)

### 7.5b Adaptive clustering（PR-2 accepted layer）

- **30/30 全部找到 `stable_k`，0 fallback**
- `stable_k` 分布：
  - `k=2`: `27/30`
  - `k=4`: `2/30`（huangwanling, 818）
  - `k=6`: `1/30`（zhangjinhan）
- **Adaptive within-cluster τ median = 0.252**，相对 overall τ = 0.089 的 uplift median = **+0.100**
- **12/30 subject 有 `candidate_forward_reverse` 对，共 17 对**
- 代表性结果：
  - `958`: `stable_k=2`, inter-cluster `r = -0.915`，精确复现老论文 E3 双向传播
  - `huangwanling`: `stable_k=4`，出现两组独立互逆对 `(0,1)` 与 `(2,3)`，而且 raw = centered，说明这里的结构几乎全是真实传播
  - `zhangjinhan`: `stable_k=6`，存在 5 对候选互逆关系，提示少数 subject 的传播路径网络明显超过二模态
- **关键口径修正**：
  - `stable_k` 是“当前事件云的最佳稳定压缩”，不是“真实模式数”
  - AMI 证明的是**同一批事件上的算法稳定性**，不是**跨时间的生物学稳定性**
  - `candidate_forward_reverse` 是候选描述标签，不是最终机制判定

### 7.6 Legacy MI（已补全）

- **30/30 全部显著** (permutation test p < 0.05) → 完全复现老论文的 17/18 + 20/20
- MI median = **0.194**（range 0.048–0.431）
- MI ≡ Kendall τ（数学上等价），但 MI 使用固定模板，τ 使用全 pair，两者互补

### 7.5c Cross-time template reproducibility（PR-2.5 accepted layer）

- **30/30 可用 subject 全部完成 PR-2.5**
- reproducibility grade：
  - `strong`: `23/30`
  - `moderate`: `7/30`
  - `weak`: `0/30`
- split-half：
  - median template match corr = **0.899**
  - median assignment agreement = **0.865**
- odd/even block：
  - median template match corr = **0.985**
  - median assignment agreement = **0.882**
- **Forward/reverse 候选对：`11/12` 在时间切片中复现**
- moderate subject：
  - `1096`, `442`, `590`, `818`, `huanghanwen`, `litengsheng`, `zhangjinhan`
  - 其中 `818 (k=4)` 与 `zhangjinhan (k=6)` 仍是高 k 结构稳定性的重点风险点
  - `huanghanwen` 是唯一 forward/reverse 未通过跨时间复现的 subject

解读：

1. **PR-2.5 解决了 PR-2 最大的逻辑缺口**：现在我们不只是能说“同一批事件上 KMeans 很稳定”，而是可以说“模板在 split-half / blockwise 尺度上总体稳定”。
2. `k=2` 主导 subject 的模板几乎都表现出非常高的跨时间一致性。
3. 少数 `k>2` subject 的**方向结构**仍然存在，但**精确 cluster 边界**更容易漂。`818` 和 `zhangjinhan` 是后续鲁棒性复核的重点对象。
4. `forward/reverse` 现在已经不只是一次性几何标签；至少在当前 split-half / blockwise 验证下，它在有该结构的 subject 中是可持续复现的。

### 7.7 科学解读（PR-2.5 验收版）

1. **多模态性是普遍的（30/30），且与刻板性共存**。每种模式内部 τ ≈ 0.25（中位），比全体 τ = 0.089 高约 3 倍。全体 τ 低的原因是跨模式混合。
2. **`k=2` 是主导压缩，但不是唯一结构**。`27/30` subject 的 `stable_k=2`，但还有 `3/30` 明显需要 `k=4` 或 `k=6` 才能更好描述。
3. **Forward/reverse 不是 958 的孤例，而且现在已经过了最基本的时间复现关**。当前 `11/12` 带有候选互逆结构的 subject 都能在时间切片中复现该关系；但这仍然只是稳定的描述结构，不是最终机制解释。
4. **Legacy MI 全部显著**：完全复现老论文结论，没有 single failure。
5. **Identity bias 仍然重要**（全体 bias fraction = 65%），但在簇内水平，真实传播结构是主要贡献者。
6. **SOZ 差异仍不显著**（p = 0.088），这是当前最弱的环节。

**核心结论**：间期群体事件内部存在多种传播模式，而且每种模式内部是刻板的。PR-2 让我们可以更诚实地说：大多数 subject 以二模态为主，但少数 subject 明显更复杂；PR-2.5 则补上了最关键的一层，证明这些模板在 split-half / blockwise 尺度上总体稳定。老论文的 MI 显著性结论完全可复现，forward/reverse 候选结构也已经过了基础复现关。下一步不该再纠结“模板在不在”，而该转向“稳定模板的占比如何随时间漂移”。

### 7.8 推荐的下一步验证

- ~~**PR-3：固定模板的论文级 per-subject 图**~~ ✅ 已验收（2026-04-12）
- ~~**PR-4A：cluster occupancy 时间轨迹**~~ ✅ 已验收（2026-04-12）
  - `30/30` subject day/night summary；dominant fraction Wilcoxon p=0.124，entropy p=0.245，TV distance median=0.019
  - 固定模板投射 agreement median=0.888（3/30 < 0.8）
  - 结论：day/night occupancy 漂移整体很弱，描述性结果，不是强机制结论
- **高 k subject 的鲁棒性复核**
  - 对 `818`、`zhangjinhan` 这类 subject 做 `n_participating` 匹配子样本和 raw/centered 双版本模板比较
  - 防止把稀疏事件或 channel identity 残差误当成高维多模态
- **PR-4B：和 Topic 2 的慢调制做固定模板 coupling**
  - 模板本身已经稳定，下一步才能诚实地问：慢 rate state 改变的是模式占比，还是模式内部 stereotype 强度
- **seizure proximity 的固定模板占比轨迹**
  - day/night 已完成，下一个时间上下文应该优先问 seizure 邻近

### 7.9 对后续的影响

- 簇内 τ 和 MI 同时显著 → 可以直接用在新论文中
- Forward/reverse 双模式的生理解释仍然只能算候选方向（例如 day/night、seizure proximity 调控不同模式的出现频率）
- PR-4A 已经回答了"占比是否昼夜漂移"（答案：很弱），下一步转向 seizure proximity 和 Topic 2 耦合

## 8. 当前状态

- 代码与独立脚本已拆出
- 独立结果目录已建立
- **PR-2.5 全量完成并验收**（30 subjects，23 strong / 7 moderate / 0 weak）
- **PR-3 全量完成并验收**（30/30 per-subject propagation + MI heatmaps, 6-panel cohort summary）
- **PR-4A 全量完成并验收**（30/30 occupancy timelines + day/night group summary）
- PR-1 的 cohort 结果以 `pr1_cohort_summary.json` 为准
- PR-4A 结果以 `pr4a_temporal_dynamics.json` 为准
- 图已生成：
  - `results/interictal_propagation/figures/pr1_propagation_heatmap_examples.png` — Figure 2 样式 heatmap
  - `results/interictal_propagation/figures/pr1_propagation_cohort_summary.png` — 6-panel cohort summary
  - `results/interictal_propagation/figures/per_subject/*_propagation.png` — PR-3 per-subject 2×2 heatmap
  - `results/interictal_propagation/figures/per_subject/*_24h_timeline.png` — PR-4A occupancy timeline
  - `results/interictal_propagation/figures/pr4a_daynight_group_analysis.png` — PR-4A cohort day/night summary

本文件只维护这个主题本身；涉及 IEI / PSD / rate modulation 的内容请回到 `docs/topic2_between_event_dynamics.md`。
