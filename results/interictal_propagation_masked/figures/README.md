# interictal_propagation (masked) — PR-2/PR-3 figures

Figures regenerated on the Topic 0 phantom-rank masked feature tree
(2026-05-22 D2 Batch 1). Source data: `../per_subject/*.json` +
`../pr1_cohort_summary.json`. Generator: `scripts/plot_interictal_propagation.py
--masked-features`. Phantom-fix details: `docs/topic0_methodology_audits.md` §3.1.

### cohort_propagation_summary.png
6-panel cohort summary on masked features: stable_k distribution, within-cluster
τ vs overall τ uplift, identity bias raw/centered, mixture taxonomy, n_participating
stratification, SOZ comparison. Tier defaults to Tier 1 (n=33 primary). **关注点**：
stable_k 分布是否保持 dominant k=2（Tier 0 27/30, Tier 1 30/33, Tier 2 35/40 +
multi-mode minorities）；within-cluster τ uplift 是否方向保持；identity bias (raw vs
centered) 是否方向保持；masked rerun 期望加强 bias_fraction 87.9 → 92.2%。

### per_subject/<dataset>_<subject>_propagation.png
Per-subject 3-panel propagation summary (rank-time scatter colored by cluster +
template overlay + MI distribution). 40 subjects expected. **关注点**：每张图
template overlay 颜色对齐 cluster；右下角 MI distribution 是否仍 significant；
3 个 chengshuai/253/818 模板投射 agreement < 0.8 时叙事须谨慎。
