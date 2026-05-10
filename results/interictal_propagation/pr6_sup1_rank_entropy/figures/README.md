# PR-6-sup1 — first-rank entropy 图说明

> Plan v3：`docs/archive/topic1/pr6_template_anchoring/pr6_supplementary_rank_entropy_plan_2026-05-10.md`
> Cohort：n=35 stable_k=2 ∩ rank_displacement v14 universe；19 subject 走完 subject-level Option B；15 subject 因 partial-participation 后 cluster `kept_events < 50` 被 `excluded_one_or_both_clusters`；1 subject (epi_1096) PR-2 label 长度对不上 `_valid_event_indices(min_participating=3)` 走 `pr2_labels_len_mismatch`。
> Tier：Topic 4 mechanism preflight，descriptive only，**不进 paper α**。
> 设计基线（feedback_figure_self_contained_paper_grade）：legend 描述科学量、不出现内部代号；单一 figure-level legend；坐标轴紧无白边；分类用感知可区分色（红 / 蓝 / 灰）。

### H_p_norm_cohort_overlay

主图。每 cluster 一条 H_p_norm 曲线（n=35 subject × 2 cluster − exit_reason 后剩 70 - 2 ≈ 68 条）插值到 normalized rank position x ∈ [0, 1]（x=0 最快通道、x=1 最慢通道）后叠加；按 forward/reverse template-pair classification 三色（红 = strong reversal、蓝 = suggestive reversal、灰 = no reversal）；cohort median 粗黑线 + IQR 灰带。**KMeans cluster_id 0/1 在本 cohort 上无系统排序**（c0 大者 16 个 / c1 大者 19 个 / 0 个相等），所以不分 panel。

**关注点**：(a) 整体 roof-shape——中间 H 接近 1.0，两端 dip 到 0.6-0.9；(b) cohort median 走势（黑线）证实形状一致；(c) 三色不分层——endpoint 稳定不是 reversal class 的副产物。

### delta_by_swapclass_box

`Δ = mean(H endpoints) − mean(H middle)` 按 strong / suggestive / no reversal 三组 box + 散点；Δ=0 参考虚线；标题直接讲结果。

**关注点**：(a) 三组 box 全部位于 Δ=0 以下；(b) 没有 outlier 跑到 Δ > 0 上半轴；(c) 组间 Kruskal-Wallis p=0.65（cohort_summary.json）—— 三组 indistinguishable。

### endpoint_pair_percentile_panel

3-panel：
- A：endpoint-pair percentile × Δ 散点。confluence prediction 应当点在 percentile=1, Δ>0 quadrant；实际全部点在 percentile≈0, Δ<0 quadrant。
- B：n_valid 直方图 + min_attainable_p_N1 floor 折线。说明小 n_valid（=6）floor 0.067 高于 0.05，硬阈值会按通道数歧视。
- C：is_endpoint_pair_max stacked bar by class。0/19 eligible subject 的端点对取得 max-Δ。

**关注点**：(a) panel A 数据集中在左下 = 否决 confluence prediction；(b) panel B 解释为什么不能用 p<0.05 硬阈值；(c) panel C 灰色 bar 占满全部 = 端点对在所有 subject 都 NOT max-Δ。

### swap_subset_per_subject

forward/reverse strong + suggestive 共 18 subject 单独 H_p_norm grid（4 行 × 5 列，2 空白 cell；T_0 蓝 / T_1 橙；class chip 在每 panel 右下角；subject-level excluded subject 标 "subject-level excluded (low kept events)"）。

**关注点**：(a) strong / suggestive reversal 标签 subject 也都出 roof-shape——swap 节点（端点 channel）真稳定；(b) excluded subject 仅由 §6.0 partial-participation 拒绝，不是 entropy 信号缺失；(c) paper-level 单 subject 例图候选（如 epilepsiae_1146 strong reversal、epilepsiae_958 strong reversal）。

### bridge_rank_displacement

横轴 = rank_displacement F_norm（0=无 swap，1=full reversal）；纵轴 = sup1 endpoint-pair percentile（1=端点对 Δ 最大）；按 reversal class 三色；★ = forward/reverse reproduced subset (rd v14)；x=2/3 红虚线 = rd 的 random reversal floor。

**关注点**：(a) 两量正交——sup1 信号集中纵轴 ≈ 0、rd 信号横轴 [0.4, 1.0] 全谱铺开，几乎独立；(b) 没有点跑右上角（高 F_norm + 高 percentile）= 两个 PR 测的不是同一信号；(c) reproduced ★ subject 在 sup1 percentile 一致 ≈ 0，confirms endpoint stability cross-PR robust。
