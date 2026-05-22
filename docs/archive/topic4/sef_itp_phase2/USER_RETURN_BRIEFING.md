# Phase 2 实施 — 用户返回简报（autonomous 8h window 2026-05-23）

> 用户当时说："开始实现，我要离开8h左右，谨慎的做"
> 本简报：Claude (Opus 4.7) 在 8h 自治窗口内完成的工作 + 等用户拍板的事项

---

## 一句话

**先讲科学**：H4 强支持 SEF-ITP 关于"事件率漂动 ≫ 几何漂动" 的核心区分性预测（n=21 有限，Wilcoxon p=9.5e-7，Cohen's d=+1.50，I_rate / I_geom 比值方向上稳健）；endpoint 通道集合时间二分稳定（Jaccard 0.71 / 0.86）；长尺度（60s+）模板挑选 compatible with mark-independent sampling within tested precision。

**然后讲细节**：但 sub-30s 短窗显示统计显著的**反聚集**（negative excess —— 不是 "memory" 而是 refractoriness 方向），加 run_length lift CI 上端擦出 ±0.05 等价带 0.001（在 framework 锁定的 δ=0.05 规则下） → **H3 integrated verdict = CONTRADICTED**（按 framework v1.0.5 §3.3 字面规则）。但失败方向与"独立触发假说"的相反方向（memory / recurrence）相反，所以并不构成"framework 整体证伪"。

**需要决策**：framework prose 需要 surgical clarification（详见 §7.2），**不是** rollback。**2 个核心决策点必须由用户拍板**才能闭环 framework doc v1.0.6 升级。

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

### 0. ⚠️ Cohort 输出位置（git ignored）

`results/topic4_sef_itp/phase2_temporal_x_geometry/{per_subject/*.json, cohort_summary.json, cohort_subjects.csv}` 是 `results/` 下的运行时 artifact，**不在 git** 里（项目惯例）。如果 fresh clone 想看数字，需重新跑：

```bash
python scripts/run_sef_itp_phase2.py --all --epoch-hours 0.5 --n-perm 1000 --seed 0
python scripts/summarize_sef_itp_phase2.py
```

archive 文档 `cohort_run_2026-05-23.md` 已经把核心数字写进文本。

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
