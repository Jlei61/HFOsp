# PR-6 rank displacement (masked) — continuous swap geometry figures

Figures on the Topic 0 phantom-rank masked feature tree. Cohort heatmap
regenerated on 2026-06-26 with `scripts/plot_rank_displacement.py
--masked-features --what cohort`; other panels are from 2026-05-22 D2 Batch 1.
Phantom-fix details: `docs/topic0_methodology_audits.md` §3.1。

### cohort_displacement_heatmap.{png,pdf}
Horizontal cohort rank displacement heatmap（columns = 34 stable_k=2 subjects
sorted by F_norm descending；rows = rank_T_a source→sink bins）。Rank axis is capped
at 24 bins; subjects with >24 joint-valid channels are median-binned along
rank_T_a order, not sorted by Δr. Colorbar is capped at ±24 with extend arrows,
so one high-channel outlier does not wash out the cohort。**关注点**：cohort F_norm
median masked 期望 ≈ 0.79（orig 0.80）；看 source→sink 方向上的红蓝反转是否和
bottom F_norm 轨道一致，尤其是 high-F_norm subjects。

### per_subject/<dataset>_<subject>_*.png
Per-subject rank displacement small-multiples（34 subject）。每张图列按 rank_T_a_dense
排序避免 sorting bias。**关注点**：Δr sign anchor 仅 subject 内部有效；reproduced cohort
6→11 subject 应展现 dashed reference 上方的 Δr 序列。

### swap_cardinality_heatmap.{png,pdf}
Swap node 数量 cohort heatmap（n=34, has_swap=9）。每行一个 subject，列 = swap node
count by decision_k。**关注点**：has_swap=9（masked）vs orig has_swap 比较；strict 子集
应集中在 decision_k=3–4。

### swap_clinical_soz_set_relation.{png,pdf} + swap_clinical_soz_overlap.{png,pdf}
PR-6 supp §9 swap × clinical SOZ set-relationship figures（typology + overlap matrix）。
**关注点**：S⊊E 优势是否保持（masked typology 仍多 partial / S⊊E），enrichment_over_lagPat
strict ∩ informative n=5 sign p masked ≈ 0.66（仍 NULL）；channel-selection circular
caveat: lagPat 已对 SOZ 富集。

### template_source_soz_overlap_top3.{png,pdf}
逐 subject 检查两个 stable_k=2 template 的 source 端 top-3 通道是否落在 clinical SOZ 内。
左图是每个 subject 的 T_a/T_b source1-3 SOZ 命中矩阵，右图是两个 template 的 source-SOZ
比例散点；exact channel list 写在同级 `../template_source_soz_overlap_top2_top3.csv`。
**关注点**：这是 source 端和 clinical SOZ 的重叠描述，不是全脑定位性能；lagPat 通道宇宙本身已偏向 SOZ，必须按 within-lagPat caveat 解释。
