# PR-6-sup1 — first-rank entropy 图说明

> Plan v3：`docs/archive/topic1/pr6_template_anchoring/pr6_supplementary_rank_entropy_plan_2026-05-10.md`
> Cohort：n=35 stable_k=2 ∩ rank_displacement v14 universe；19 subject 走完 subject-level Option B；15 subject 因 partial-participation 后 cluster `kept_events < 50` 被 `excluded_one_or_both_clusters`；1 subject (epi_1096) PR-2 label 长度对不上 `_valid_event_indices(min_participating=3)` 走 `pr2_labels_len_mismatch`。
> Tier：Topic 4 mechanism preflight，descriptive only，**不进 paper α**。
> 设计基线（feedback_figure_self_contained_paper_grade）：legend 描述科学量、不出现内部代号；单一 figure-level legend；坐标轴紧无白边；分类用感知可区分色（红 / 蓝 / 灰）。

### H_p_norm_cohort_overlay

Option B（all-valid-participating events）下的 H_p_norm 曲线 cohort 叠加。x = normalized rank position ∈ [0, 1]，0 = 最快通道、1 = 最慢通道；按 reversal class 三色；cohort median 粗黑线 + IQR 灰带。**KMeans cluster_id 0/1 在本 cohort 上无系统排序**（c0 大者 16 / c1 大者 19 / 0 相等），所以不分 panel。

**重要 caveat（user 2026-05-10 review 提出）**：本图的"屋顶形"被发现是**两个来源混合**——快端低熵是真生物信号，慢端低熵是 Option B selection artifact。详见 `H_p_norm_option_b_vs_with_absent`。本图只能支持"端点位置非 uniform"，**不能**支持"端点稳定 / source-sink stereotyped"。

**关注点**：(a) cohort 平均屋顶形；(b) 三色不分层——任何残留信号不是 reversal class 副产物；(c) 不能直接读"端点稳定"——需配合 with-absent 对照图。

### H_p_norm_option_b_vs_with_absent  ⭐ paper-grade re-analysis

3-panel 对照图，回应"如果考虑非参与通道，值会不会很低？"

- **A. Option B**（当前主图条件，drop_rate median = 0.98 selection 后）：roof-shape，两端 H ≈ 0.78
- **B. with-absent**（保留所有 events，把"该 rank 位置没人"当作 alphabet 第 n_valid+1 个 sentinel，按 log₂(n_valid+1) 归一化）：**单调下降**，快端 ≈ 0.85、慢端 ≈ 0.05
- **C. P("absent") at rank p**：慢端 95-100% events 在该 rank 位置空缺——解释了 panel B 慢端为何塌到接近 0

**重新解读**：

| 信号位置 | 在 Option B 下 | 真实来源（with-absent 揭示）|
|---|---|---|
| 快端低熵（rank≈1）| 0.78 | **真生物信号**：source channel 部分 stereotyped（effective alphabet ~6 of 8）|
| 慢端低熵（rank≈n_valid）| 0.78 | **Selection artifact**：~95% events 该位置就是 absent，Option B 才把它"挑"出"低熵" |

**关注点**：(a) panel A vs B 的形状对比是关键——roof 变 monotonic decrease；(b) panel C 慢端高 P("absent") 是因果解释；(c) with-absent 下 confluence prediction（端点最高熵）依然被否决，但**stable-pathway**（端点最低熵）只支持快端、不支持慢端。

### delta_by_swapclass_box

`Δ = mean(H endpoints) − mean(H middle)` 按 swap-strict / swap-candidate / swap-none 三组 box + 散点；Δ=0 参考虚线；标题直接讲结果。

**关注点**：(a) 三组 box 全部位于 Δ=0 以下；(b) 没有 outlier 跑到 Δ > 0 上半轴；(c) 组间 Kruskal-Wallis p=0.65（cohort_summary.json）—— 三组 indistinguishable。

### endpoint_pair_percentile_panel

3-panel：
- A：endpoint-pair percentile × Δ 散点。confluence prediction 应当点在 percentile=1, Δ>0 quadrant；实际全部点在 percentile≈0, Δ<0 quadrant。
- B：n_valid 直方图 + min_attainable_p_N1 floor 折线。说明小 n_valid（=6）floor 0.067 高于 0.05，硬阈值会按通道数歧视。
- C：is_endpoint_pair_max stacked bar by class。0/19 eligible subject 的端点对取得 max-Δ。

**关注点**：(a) panel A 数据集中在左下 = 否决 confluence prediction；(b) panel B 解释为什么不能用 p<0.05 硬阈值；(c) panel C 灰色 bar 占满全部 = 端点对在所有 subject 都 NOT max-Δ。

### swap_subset_per_subject

forward/reverse strong + suggestive 共 18 subject 单独 H_p_norm grid（4 行 × 5 列，2 空白 cell；T_0 蓝 / T_1 橙；class chip 在每 panel 右下角；subject-level excluded subject 标 "subject-level excluded (low kept events)"）。

**关注点**：(a) swap-strict / swap-candidate subject 也都出 roof-shape——但根据 with-absent 对照（见 `H_p_norm_option_b_vs_with_absent`），其慢端低熵是 selection artifact，不是真"端点稳定"；(b) excluded subject 仅由 §6.0 partial-participation 拒绝，不是 entropy 信号缺失；(c) paper-level 单 subject 例图候选（如 epilepsiae_1146 swap-strict、epilepsiae_958 swap-strict），但解读必须配合 with-absent 图。

### bridge_rank_displacement

横轴 = rank_displacement F_norm（0=无 swap，1=full reversal）；纵轴 = sup1 endpoint-pair percentile（1=端点对 Δ 最大）；按 reversal class 三色；★ = forward/reverse reproduced subset (rd v14)；x=2/3 红虚线 = rd 的 random reversal floor。

**关注点**：(a) 两量正交——sup1 信号集中纵轴 ≈ 0、rd 信号横轴 [0.4, 1.0] 全谱铺开，几乎独立；(b) 没有点跑右上角（高 F_norm + 高 percentile）= 两个 PR 测的不是同一信号；(c) reproduced ★ subject 在 sup1 percentile 一致 ≈ 0，confirms endpoint stability cross-PR robust。
