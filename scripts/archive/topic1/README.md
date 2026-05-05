# scripts/archive/topic1

Topic 1（事件内动力学）相关的一次性 sentinel / preview 脚本。
findings 已归档到 `docs/archive/topic1/`，本目录脚本仅历史复现用。

## 脚本与对应文档

| 脚本 | 用途 | 对应归档 doc |
|---|---|---|
| `sentinel_pr6a_step2.py` | PR-6-A Step 2 sentinel：548 / 11 子集上 gamma_ER vs broad_ER 多通道 z-ER trace 目检 | `docs/archive/topic1/pr6a_template_ictal_alignment_plan_2026-04-21.md`、`docs/archive/topic1/pr6a_step0-2_step3preview_review_2026-04-23.md`、`docs/archive/topic1/pr6a-1.md` |

## 相关产出文档（建议先读）

- `docs/topic1_within_event_dynamics.md` — 当前主入口
- `docs/archive/topic1/pr6_template_endpoint_anchoring_plan_2026-04-25.md`
- `docs/archive/topic1/pr6a_step0-2_step3preview_review_2026-04-23.md`
- `docs/archive/topic1/pr7_template_antagonistic_pairing_plan_2026-04-28.md`
- `docs/archive/topic1/pr7_template_pairing_results_2026-04-29.md`
- `docs/archive/topic1/pr7_addendum_p3_equivalence_2026-05-01.md`
- `docs/archive/topic1/pr_t4_1_bhpn_toy_plan_2026-05-01.md`
- `docs/archive/topic1/pr8_intra_event_spatial_polarity_plan_2026-04-30.md`

## 状态

PR-6-A 已并入 PR-6 template endpoint anchoring（accepted）。当前生产 PR-6 入口
是 `scripts/run_pr6_template_anchoring.py` 与 `scripts/plot_pr6_template_anchoring.py`。
本目录 sentinel 仅作回溯参考。
