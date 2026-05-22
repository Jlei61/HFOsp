# Topic 0 Archive Index — 方法学审计与数据合同

> **主入口**：`docs/topic0_methodology_audits.md`
> **范围**：跨 topic 的方法学问题归档；每个 audit 一个子目录。
> **不属于**：单个 topic 内部的科学结论 → 各 topic 自己的 archive。

每个子目录承载：
- `diagnostic_<DATE>.md` — 技术诊断（数据、cohort 数字、gate verdict）
- `plain_chinese_report_<DATE>.md` — 白话报告（给非技术读者 / 自己回看）
- `rerun_roadmap_<DATE>.md` — 重跑路线图 + checkpoint
- `step5{a..h}_*.md` — 每步 broad re-derivation 子结果
- `checkpoint_{a,b}_report_<DATE>.md` — advisor consult 报告
- `rerun_results_<DATE>.md` — 终稿整合 doc（一站式回看）
- 必要时：sensitivity 子归档、framework-revision 评估

## 子目录

### `lagpat_phantom_rank/` — lagPatRank phantom pseudo-rank（2026-05-20 确诊，**broad re-derivation 已完成 2026-05-22**；工程层 5i.6 default flip 进行中）

**入门顺序**（按读者角色）：

- 非技术 / 自回看 → `plain_chinese_report_2026-05-20.md`（白话）+ `rerun_results_2026-05-21.md` §1 一句话总判读
- 系统级 verdict → `rerun_results_2026-05-21.md`（终稿，整合 5a-5h+5g 所有 cohort 数字 + reconcile 表）
- 单步细节 → `step5{a,b,c,d.1,d.2,d.3,e,f,g,h}_*_2026-05-{20,21,22}.md`
- 历史 / 怎么发现的 → `diagnostic_2026-05-20.md` + `rerun_roadmap_2026-05-20.md`
- Checkpoint 报告 → `checkpoint_b_report_2026-05-21.md`
- Phase 0 进度追踪 → `phase0_progress_report_2026-05-21.md`

**完整文件清单**：

| 文件 | 角色 | 状态 |
|---|---|---|
| `diagnostic_2026-05-20.md` | 技术诊断（40-subject cohort gate verdict = broad re-derivation） | locked |
| `plain_chinese_report_2026-05-20.md` | 白话总览 | locked |
| `rerun_roadmap_2026-05-20.md` | Step 5a–5i 路线图 + checkpoint 规范 | locked |
| `step5a_pr2_results_2026-05-20.md` | PR-2 主聚类 re-cluster（35/40 stable_k=2，AMI 相关 0.961） | locked |
| `step5b_pr25_results_2026-05-20.md` | PR-2.5 split-half + odd-even（grade 31/9/0 → 26/12/2，fwd/rev 16/17 → 15/16） | locked |
| `step5c_pr3_results_2026-05-20.md` | PR-3 簇内 stereotypy（bias_fraction 87.9 → 92.2% 加强；PR-4 panel d 独立证据加强） | locked |
| `step5d.1_pr4a_results_2026-05-20.md` 及后续 5d.2.{0,1,2,3} | PR-4A/B/C 各步 cohort 数字 + L3 高置信子集 fragility loss | locked |
| `checkpoint_b_report_2026-05-21.md` | Checkpoint B advisor consult（5d 后；soft trigger, broad re-derivation 继续） | locked |
| `step5e_pr5_results_2026-05-21.md` | PR-5 / PR-5-B（dominant rate +65.46 → +65.66 events/h 主信号保持；3 secondary share / transition flip） | locked |
| `step5f_pr6_results_2026-05-21.md` | PR-6 全套（H1 NULL 不变；Step 6 swap_class concordance 0.69 → 0.82 加强；node anatomy h1_eligible secondary 翻 borderline） | locked |
| `step5g_pr7_results_2026-05-21.md` | PR-7 antagonistic temporal pairing（H1 NULL stays NULL；**P3 framework-flip gate CLEAR**：like-for-like orig 6-cohort verdict INCONCLUSIVE 完全保持，4/4 flag 一致；directional finding 留 PR-7 v2 跟踪）— **真版 2026-05-22 重写，agent v1 的下游 cohort 段为 fabricated** | locked |
| `step5h_topic4_attractor_results_2026-05-21.md` | Topic 4 attractor Step 1（cohort 34，少 916；GOF 97% 持平；**coordinate-free λ₂ 10/34 → 13/34 实质加强**） | locked |
| `rerun_results_2026-05-21.md` | **Phase 0 终稿整合 doc**（5a-5h + 5g + reconcile 表 + 论文叙事调整清单 + 5i 收口剩余任务） | locked |
| `phase0_progress_report_2026-05-21.md` | Phase 0 进度跟踪 | locked |

## 待写

- 当后续发现新方法学问题时，在 `docs/topic0_methodology_audits.md` §3 加新小节，并在本目录建对应子目录。
- 现有 lagPatRank Phase 0 收口：5i.6 default flip 完成 + missed-path code fix 完成后，本 INDEX `lagpat_phantom_rank/` 行从"工程层 5i.6 进行中"改为"全部封板"。
