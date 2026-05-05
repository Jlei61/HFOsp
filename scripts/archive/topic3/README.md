# scripts/archive/topic3

Topic 3（SOZ 空间归因 / where 问题）pivot 期间的一次性 sentinel preview 脚本。
findings 已归档到 `docs/archive/topic3/`，本目录脚本仅历史复现用。

## 脚本与对应文档

| 脚本 | 用途 | 对应归档 doc |
|---|---|---|
| `sentinel_t_er_onset_preview.py` | PR-T Step 2 sentinel 上 per-channel ER onset 时序 preview（不实现完整 Step 3 合同） | `docs/archive/topic3/pr_t3_1_data_driven_soz_audit_plan_2026-04-30.md`、`docs/archive/topic3/pr_t3_1_pivot_to_pr6a_er_ranking_2026-05-03.md` |

## 相关产出文档（建议先读）

- `docs/topic3_spatial_soz_modulation.md` — 当前主入口
- `docs/archive/topic3/pr_t3_1_data_driven_soz_audit_plan_2026-04-30.md` — PR-T-3-1 plan v2.2
- `docs/archive/topic3/pr_t3_1_pivot_to_pr6a_er_ranking_2026-05-03.md` — pivot 决策
- `docs/archive/topic3/per_subject_ictal_er_atlas.md` — per-subject ictal ER atlas
- `docs/archive/topic3/spatial_modulation_soz_analysis.md` — PR-1 (Yuquan) 结果
- `docs/archive/topic3/epilepsiae_three_tier_pr2_2026-04-27.md` — i/l/e 三层
- `docs/archive/topic3/epilepsiae_artifact_census_2026-04-27.md` — Epilepsiae 数据健康审计

## 状态

PR-T-3-1 当前生产入口是 `scripts/run_data_driven_soz.py` 与 `scripts/run_ictal_er_rank.py`
（A.1–A.4 已 land，commit 919a2cf 起）。本目录 sentinel 仅作 Step 2/3 间的
preview 历史记录。
