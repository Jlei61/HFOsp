# Phase 2 实施 — 用户返回简报（autonomous 8h window 2026-05-23）

> 用户当时说："开始实现，我要离开8h左右，谨慎的做"
> 本简报：Claude (Opus 4.7) 在 8h 自治窗口内完成的工作 + 等用户拍板的事项

---

## 一句话

Topic 4 SEF-ITP Phase 2 (H3 mark-independence + H4 normalized rate/geometry instability) 实施完成，n=23 cohort run **finished** (`results/topic4_sef_itp/phase2_temporal_x_geometry/`)；**1 个 framework spec 决策点必须由用户拍板**才能闭环。

---

## 完成了什么

### 设计 + 计划

- `docs/superpowers/plans/2026-05-23-topic4-phase2-h3-h4-plan.md`（14 个 task TDD plan + 4 个 advisor catch 整合）
- 调用 advisor 一次（在写代码前）；catch 全部纳入

### 代码

- `src/sef_itp_phase2.py` —— 11 个 pure function + dataclass：H3 ingest 提取器、TOST equivalence (ported PR-7)、cohort LOO、H3 integrated verdict、epoch slicer (with Epilepsiae short-block tolerance)、per-epoch endpoint、Jaccard、I_rate (两个 null 方法)、I_geom、H4 cohort verdict
- `scripts/run_sef_itp_phase2.py` —— per-subject runner
- `scripts/summarize_sef_itp_phase2.py` —— cohort 汇总 + verdict
- `src/sef_itp_phase1.py` —— 抽出 `resolve_lagpat_subject_dir` 为公共函数（advisor catch D）

### 测试

- **39 个 Phase 2 单元测试 GREEN**
- 全 Phase 1 + Phase 2 + 集成测试 **104/104 GREEN**

### 数据产出

- `results/interictal_propagation_masked/template_pairing/per_subject_burst/*.json` —— 把 PR-7 burst diagnostic 从原 n=8 cohort 扩展到 Phase 1 n=23 (Task 0)
- `results/topic4_sef_itp/phase2_temporal_x_geometry/per_subject/*.json` —— 23 个 per-subject Phase 2 输出
- `results/topic4_sef_itp/phase2_temporal_x_geometry/cohort_summary.json` —— cohort verdict
- `results/topic4_sef_itp/phase2_temporal_x_geometry/cohort_subjects.csv` —— 平展数字表

### 文档

- `docs/archive/topic4/sef_itp_phase2/spec_amendment_2026-05-23.md` —— **科学决策提议**：H4 I_rate matched null
- `docs/archive/topic4/sef_itp_phase2/cohort_run_2026-05-23.md` —— cohort run 归档（数字已填，verdict 已写）
- 本简报 `USER_RETURN_BRIEFING.md`

---

## 关键决策点（等用户拍板）

### 1. H4 I_rate matched null — framework v1.0.5 §3.4 spec 数学退化 🛑

**问题**：framework v1.0.5 §3.4 写的 "shuffle epoch order, recompute std" 是数学退化的——std 对置换不变 → null variance = 0 → I_rate 必然未定义。这是 framework prose 层面的 spec 错误。

**implementation 做了什么**：实施了**两个** null 方法并行跑、并行报：
- `epoch_order_shuffle_literal` (framework v1.0.5 §3.4 原文，必然退化，仅作 audit)
- `circular_shift_within_block_proposed` (Phase 2 v1.0.0 提议；非退化)

**没做什么**：**framework doc 不动**（不能自己挑一个 null 然后偷偷改 pre-registered statistic）。

**用户必须决定**：
- 选哪个 null（circular_shift / Poisson / gamma / cross-epoch shuffle / 其他）
- 是否要 framework v1.0.6 升级 banner

详情：[`spec_amendment_2026-05-23.md`](spec_amendment_2026-05-23.md)

### 2. epoch_hours 默认值 = 0.5 (不是 framework 写的 1-2h)

Epilepsiae 自然 block 是 ~59min41s。Framework v1.0.5 §3.4 写 "1h-2h epoch" → Epilepsiae 在 1h epoch 下每 block 正好 1 epoch，circular-shift null 退化（block 内 single epoch 没自由度）。

**implementation**：默认改 0.5h（runner CLI `--epoch-hours` 可覆盖），同时给 slicer 加 `epoch_tolerance=0.1` 让 0.995h block 算作 1 个完整 epoch（在 1h epoch 设置下）。

**用户是否同意**这个默认值变更？需不需要 framework prose 把 epoch_hours 范围改为 "0.5-2h"？

### 3. endpoint stability OR vs AND combinator

advisor catch A：默认 OR 而非 AND（项目惯例 from `forward_reverse_reproduced` = split-half OR odd-even）。这个已经写进 plan、写进代码、写进测试。如果用户偏好 AND，是 5 行代码改动 + 1 个 test 反转。

---

## 复用纪律（CLAUDE.md §6.1 question-match check）

每一处 reuse 都做了显式 question-match 文档化：

| Phase 2 用了什么 | 来源 | match? |
|---|---|---|
| `tost_equivalence` | `scripts/pr7_addendum_p3_equivalence.py:123` | ✅ 同问题（cohort median TOST vs ±δ） |
| `compute_burst_diagnostic_with_nulls` | `src.template_temporal_pairing` | ✅ 同问题（lag-1 same + run-length vs N2） |
| PR-7 pairing `lift.N2.*.excess` | PR-7 per-subject JSON | ✅ 同问题（多窗口 mark-transition excess） |
| PR-6 `split_half_robustness.*.subject_mean_jaccard_endpoint` | PR-6 anchoring | ✅ 同问题（endpoint Jaccard recall） |
| `compute_local_endpoint`（new Phase 2 helper） | NOT PR-6 anchoring helper | ⚠️ 不同 gate（per-epoch local vs cohort-reproducibility）→ 写本地 thin helper |

---

## 提交历史

```
plan(topic4 phase2): H3 mark-independence + H4 normalized instability implementation plan
plan(topic4 phase2): incorporate advisor feedback before implementation
feat(topic4 phase2): module skeleton + SubjectPhase2Data dataclass
feat(topic4 phase2): H3 ingest extractors for PR-7 pairing/burst + PR-6 anchoring
feat(topic4 phase2): port tost_equivalence from PR-7 addendum
feat(topic4 phase2): H3 integrated verdict (SUPPORTED/NOT_SUPPORTED_*/CONTRADICTED)
feat(topic4 phase2): H4 epoch slicer (block-aware, time-preserving)
feat(topic4 phase2): H4 local endpoint extraction + Jaccard helper
feat(topic4 phase2): H4 I_rate two null methods + spec amendment proposal
feat(topic4 phase2): H4 I_geom normalized endpoint-geometry instability
feat(topic4 phase2): H4 cohort Wilcoxon + Cohen's d verdict
refactor(topic4 phase1): promote resolve_lagpat_subject_dir to public src module
feat(topic4 phase2): cohort TOST + leave-one-out (advisor catch C)
feat(topic4 phase2): runner script + Epilepsiae short-block tolerance
feat(topic4 phase2): cohort summarizer (H3 TOST + LOO + H4 Wilcoxon)
docs(topic4 phase2): cohort run archive draft (numbers pending background run)
results(topic4 phase2): full n=23 cohort run results landed  [LAST]
```

---

## 没做什么（DEFERRED）

- **framework doc edit** (`docs/topic4_sef_itp_framework.md`) — 因为 H4 I_rate null 是 science decision，advisor catch B 明确拦下
- **sensitivity sweep** (epoch_hours / endpoint_k 不同值)
- **figures** —— Phase 2 没有生图（数字层够用了；如果要 paper-grade 图，单独追加 PR）

---

## 推荐用户回来后第一步

读 `docs/archive/topic4/sef_itp_phase2/cohort_run_2026-05-23.md` §1-§3，看 H3/H4 verdict；再读 `spec_amendment_2026-05-23.md` 决定 H4 I_rate null。等这两件事确定后，可以一并 commit framework doc 更新 + 任何必要的措辞收紧。
