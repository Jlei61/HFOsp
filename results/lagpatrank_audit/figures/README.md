# lagPatRank Phantom-Rank Audit — Diagnostic Figures

Source data: `../cohort_summary.csv` (n=40 subjects; all status=ok).
Code: `scripts/audit_kmeans_phantom_rank.py` +
`scripts/augment_lagpat_audit_masked_stable_k.py` +
`scripts/plot_lagpat_phantom_audit.py`.
Plan: `docs/superpowers/plans/2026-05-20-lagpatrank-phantom-rank-audit-and-fix.md`.

---

### ami_vs_noise_floor.png

主图：x = original feature 上跨 5 seed 的 pairwise median AMI（seed-jitter noise floor），
y = AMI(original@seed=0, masked@seed=0) = audit signal。对角线 = "audit AMI 等于 seed floor"
（cosmetic 通过区域）。灰色带 = cosmetic 阈值 (Δ ≥ -0.05)。

**关注点**：所有 40 个 subject 都远低于对角线。stable_k=2 cohort (n=35, 圆点)
median 音频 = 0.39，floor = 1.0；stable_k>2 cohort (n=5, 方块) 同样落在底部。
无 subject 进 cosmetic band。Cosmetic 出口被实证排除。

### phantom_fraction_vs_delta.png

x = phantom_fraction = `(~bools).sum() / bools.size`，y = `ami_audit_minus_floor`。
两条参考线 = pre-registered cosmetic (-0.05) / broad-rederivation (-0.15) 阈值。
标注 Spearman ρ。

**关注点**：所有 40 subject 都在 broad-rederivation 阈值 (-0.15) 之下；35/40 在
-0.50 之下。phantom fraction 越高，Δ 越负 (ρ = -0.42, p = 0.007)，符合
"more phantom → more cluster-identity damage" 的预期。phantom fraction 范围
[0.140, 0.458]，median 0.328 —— 即便最低 phantom 占比的 subject 仍越线。

### stable_k_confusion.png

行 = original stable_k，列 = masked stable_k 重新跑后的选择 (k_range=[2..6])。
对角线 = 重选 k 与原一致；off-diagonal = stable_k 翻转。

**关注点**：36/40 对角线 (k 选择不变)。4/40 翻转：
3 个高 k 翻转 (huangwanling 4→3, zhaojinrui 5→6, zhangjinhan 6→5) 都在 n_ch ≤ 5
的 stable_k>2 cohort 内 (这些 subject 已被 Topic 4 H3 主分析以独立结构理由排除)。
**唯一 stable_k=2 cohort 内的翻转：epilepsiae_916 (2→4)**，Δ=-0.81，
masked floor=1.0；material flip。

---

## 一句话验收

Cohort gate 走 **Broad re-derivation**。
- median Δ = -0.609 (stable_k=2 cohort median -0.609; stable_k>2 median -0.252)
- 35/35 stable_k=2 subject 越 broad 阈值
- 4/40 stable_k 翻转，1 个在主线 cohort 内 (epilepsiae_916: 2→4)
- 已写归档 `docs/archive/topic1/propagation/lagpatrank_phantom_audit_diagnostic_2026-05-20.md`
