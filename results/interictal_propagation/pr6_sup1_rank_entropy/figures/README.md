# PR-6-sup1 — first-rank entropy 图说明

> Plan v3：`docs/archive/topic1/pr6_template_anchoring/pr6_supplementary_rank_entropy_plan_2026-05-10.md`
> Cohort：n=35 stable_k=2 ∩ rank_displacement v14 universe；其中 19 subject 走完 subject-level Option B；16 subject 在 §6.0 partial-participation 过滤后 cluster kept_events < 50 被 `excluded_one_or_both_clusters`。
> Tier：Topic 4 mechanism preflight，descriptive only，**不进 paper α**。

### H_p_norm_cohort_overlay

主图。每 subject H_p_norm 曲线插值到 normalized rank position x ∈ [0, 1]（x=0 最快、x=1 最慢）后叠加；按 §8 swap_class 三色（rust=strict, dust=candidate, gray=none）；cohort median 粗黑线 + IQR 灰带；cluster 0 / 1 双 panel；虚线参考 H=1.0 满熵。

**关注点**：(a) 整体是否 roof-shape——中间高（接近满熵）、端点低；(b) cohort median 在两端是否明显 dip 到 0.7-0.9；(c) §8 swap_class 三色是否都展示同样形状（无分层 bias）。

### delta_by_swapclass_box

`Δ_subject` (`= mean(H 端点) − mean(H 中段)`) 按 strict / candidate / none 分组 box + 散点；虚线 Δ=0 参考；y 轴标注"正 = confluence prediction，负 = endpoints determined"。

**关注点**：(a) 三组 box 全部位于 Δ=0 以下且不跨过；(b) 三组中位数差异是否显著（Kruskal-Wallis 见 cohort_summary.json）；(c) 没有 outlier 跑到 Δ > 0。

### endpoint_pair_percentile_panel

3-panel：
- A: `subject_combo_percentile × Δ_subject` 散点；x=1.0 参考线（端点对组合是 max）；按 swap_class 三色
- B: `n_valid` 直方图 + cluster-level `min_attainable_p_N1` 双 y 轴折线（小 n_valid floor 高，n=6 → 0.067）；红虚线 0.05 参考说明 floor heterogeneity
- C: `is_subject_combo_max` stacked bar by swap_class（True / False count）

**关注点**：(a) panel A 点密集在 percentile 0 附近 + Δ < 0 半轴 = endpoint 对接近 max ENTROPY-差最小，反向 confluence；(b) panel B 解释为什么不能用硬 p<0.05 阈值；(c) panel C 三组 True 计数全部 0（plan §7.2.3 floor + 实际反向方向共同导致）。

### swap_subset_per_subject

§8 strict + candidate 共 18 subject 各自 H_p_norm 曲线 grid（cluster 0 蓝 + cluster 1 红 / terracotta）+ subtitle 报 Δ_subj / percentile / is_max；excluded subject 标 "subject-level excluded"。

**关注点**：(a) 已被 §8 swap_class 标 strict/candidate 的 subject 是否都出 roof-shape，证明 swap 节点（端点 channel）真正稳定；(b) excluded subject 是否仅由 §6.0 partial-participation 拒绝（不是 entropy 信号本身缺）；(c) 为 paper-level 单 subject 例图候选。

### bridge_rank_displacement

桥接图。x = `rank_displacement F_norm`（0 = 无 swap, 1 = full reversal），y = `sup1 subject_combo_percentile`（1.0 = 端点对是 max）；按 §8 swap_class 三色；★ = `fwd_rev_reproduced` (rank_displacement v14 的 11 subject)；x=2/3 参考线（rd 的 random reversal floor）。

**关注点**：(a) 两量是否正交——sup1 信号集中 percentile ≈ 0、rd 信号在 F_norm 全谱铺开；(b) `fwd_rev_reproduced` ★ subject 是否 H_p_norm 也 robust；(c) 没有点跑到右上角（高 F_norm + 高 percentile）= 两个 PR 测的不是同一信号。
