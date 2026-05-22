# PR-6 rank displacement (masked) — continuous swap geometry figures

Figures regenerated on the Topic 0 phantom-rank masked feature tree
(2026-05-22 D2 Batch 1). Generator: `scripts/plot_rank_displacement.py
--masked-features`. Phantom-fix details: `docs/topic0_methodology_audits.md` §3.1。

### cohort_displacement_heatmap.{png,pdf}
Cohort rank displacement heatmap (rows = 34 stable_k=2 subjects sorted by Kendall
τ, columns = channels by template rank). swap_class strict markers indicate the
n_strict subset; dashed reference at 2/3。**关注点**：cohort F_norm median masked
期望 ≈ 0.79（orig 0.80），τ median ≈ −0.24（orig −0.20），ρ(F_norm, τ) ≈ −0.92
（强负相关持平）；PR-2.5 fwd/rev-reproduced subject 应仍聚集在 upper τ 带。

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
