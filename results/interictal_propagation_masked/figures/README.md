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
Per-subject 4-panel propagation summary (4-quadrant 2×2 layout):
- **左上 (top-left)**: lagPatRank heatmap, 时间顺序; 底部加 Day/Night strip
- **左下 (bottom-left)**: KMeans k_best 聚类后的 heatmap, 按簇排序, 红色虚线分隔
- **右上 (top-right)**: per-channel rank distribution (原始通道顺序, stacked histogram)
- **右下 (bottom-right)**: per-cluster rank distributions (固定通道顺序, 两簇叠加)

**Topic 0 phantom-rank visualization fix (2026-05-22)**：左侧两个 rank heatmap
现在把非参与通道-事件 cell（eventsBool=False）渲染为 **lightgray**（不再是
viridis 着色 phantom int rank）。`_mask_phantom_cells()` + `_viridis_with_lightgray_bad()`
helper 把 bools=False 的 cell 设 NaN，cmap.set_bad('lightgray') 实现灰色渲染。
右上 / 右下两个分布图本就已经过 bools mask filtering，不受影响。

40 subjects expected. **关注点**：
1. 左侧 heatmap 灰色面积应跟通道参与率匹配（高参与 subject 灰色少；低参与
   subject 像 epilepsiae_620 / 818 / 916 / yuquan_zhourongxuan 灰色应占大半）
2. 聚类后 heatmap 簇内颜色一致性（同簇 vs 跨簇 rank pattern 应有视觉对比）
3. 右下角 MI distribution 是否仍 significant
4. 3 个 chengshuai/253/818 模板投射 agreement < 0.8 时叙事须谨慎
