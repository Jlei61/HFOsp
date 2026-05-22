# PR-6 template anchoring (masked) — endpoint geometry figures

Figures regenerated on the Topic 0 phantom-rank masked feature tree
(2026-05-22 D2 Batch 1). Generators:
- `scripts/plot_pr6_template_anchoring.py --masked-features --main`
- `scripts/plot_pr6_swap_cluster_rank_multiples.py --masked-features`

Phantom-fix details: `docs/topic0_methodology_audits.md` §3.1; PR-6 masked rerun
verdicts: `docs/archive/topic0/lagpat_phantom_rank/step5f_pr6_results_2026-05-21.md`.

### pr6_template_pair_geometry_main.png
PR-6 6-panel main figure: subject swap-vs-same paired test + node anatomy +
fwd/rev small multiples + endpoint geometry summary. **关注点**：H2 fwd/rev 子集
sign-test 是否 cleanly positive（masked 期望 8/8 vs orig 9/9）；H1 SOZ anchoring
cohort Wilcoxon 仍 NULL（masked p ≈ 0.388）；node anatomy h1_eligible Wilcoxon p
masked 期望 0.059（从 orig 0.014 进一步弱化）。

### pr6_supp_swap_cluster_rank_multiples.{png,pdf}
Forward/reverse template fwd/rev all subject small-multiples（PR-6 H2 cohort）。
每个 subject 一对面板（cluster A vs B），颜色编码节点 swap class：strict (rust) /
candidate (orange) / none (gray)。**关注点**：strict + candidate 节点是否在不同
subject 上呈现一致的 source ↔ sink swap 几何；强 fwd/rev 子集（n=8）是否每张面板
都有显著 swap 节点。

### pr6_supp_swap_cluster_rank_multiples_nonstrong.{png,pdf}
同上结构，但只包含 NON-strict subject（用于对比 baseline）。**关注点**：non-strict
subject 应该是 mixed / gradient geometry，不是干净的 swap，证明 strict tier 不是
方法学伪影。

### pr6_supp_rank_displacement_swap_strict.{png,pdf}
Variable-k swap classifier strict tier rank displacement panels（2×5 网格，10 个
strict subject）。每张 panel：列按 rank_T_a_dense 排序避免 sorting bias；Δr sign
anchor 仅 subject 内部有效。**关注点**：masked rerun 期望 strict 9/28（orig 10/35）
+ candidate 6/28（orig 8/35），分布大致保持；strict 子集主要落在 PR-2.5
fwd/rev-reproduced 集合。
