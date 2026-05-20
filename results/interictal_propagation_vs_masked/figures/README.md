# Step 5a Phantom vs Masked PR-2 — comparison figures

Source: `../pr2_comparison.csv` + `../pr2_comparison_summary.md`.
Plan: `docs/topic0_methodology_audits.md` §5 + `docs/archive/topic0/lagpat_phantom_rank/rerun_roadmap_2026-05-20.md`.

### label_jaccard_distribution.png
Left: distribution of PR-2 label-level Jaccard between original (phantom) and masked clusterings (best-permutation), restricted to subjects where chosen_k is unchanged. Low values → 'WHICH event in which cluster' changed substantially. **关注点**：median 是否 < 0.5。

Right: scatter of `ami_audit_minus_floor` (lagpatrank_audit Step 2) vs PR-2-level AMI on rerun output. Sanity-check that the two measurements agree on which subjects are most affected.

### cluster_fraction_shift.png
Per-subject max |orig − masked| cluster fraction after best-perm matching. Even when chosen_k is unchanged, the cluster size balance can shift dramatically (e.g. chengshuai: [0.458, 0.542] → [0.609, 0.391]). **关注点**：median shift 是否 > 0.10。
