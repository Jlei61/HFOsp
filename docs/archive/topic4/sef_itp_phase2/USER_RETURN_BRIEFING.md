# Phase 2 实施 — 用户返回简报（autonomous 8h window 2026-05-23）

> 用户当时说："开始实现，我要离开8h左右，谨慎的做"
> 本简报：Claude (Opus 4.7) 在 8h 自治窗口内完成的工作 + 等用户拍板的事项

---

## 一句话（2026-05-23 user-return 后 v2 修订）

**先讲科学**：长尺度（60s/30min）模板挑选 compatible with mark-independent sampling within tested precision；endpoint 通道集合时间二分稳定（Jaccard 0.71/0.86）；participation-field-based per-epoch H4 强一致（n=21 有限，Wilcoxon p=9.5e-7，Cohen's d=+1.50） — 但 user 2026-05-23 指出 participation field ≠ rank-based propagation endpoint，**H4 v1.0 supplementary only，主线 H4 要重做（v1.1 rank-based endpoint geometry drift）**。

**短尺度细节（user 2026-05-23 三阶段解读完全成立）**：sub-30s 短窗显示统计显著的**反聚集**（negative excess —— 不是 "memory" 而是 refractoriness 方向），加 run_length lift CI 上端擦出 ±0.05 等价带 0.001（按 framework v1.0.5 §3.3 字面 AND-rule） → **H3 verdict = CONTRADICTED**。但失败方向与"独立触发假说"的相反方向（memory / recurrence）相反，从而**三阶段生理叙事**（burst → refractory → independent）反而是 framework 走得更具体的证据。

**需要 framework prose surgical clarification（不是 rollback、不是 rescue）**：v1.0.6 升级要保留 v1.0.5 字面 AND-rule 作为审计踪迹（CLAUDE.md §5 pre-registration 纪律），加新 sub-bullet "scale-stratified verdict: long-scale ≥60s independence + short-scale 10–30s anti-clustering descriptive (not predicted by v1.0.5 §3.3 but physiologically compatible refractoriness) + endpoint stability"；cohort 数字不变。同时 H4 主线 v1.1 重做（rank-based endpoint geometry drift + 4 spatial radius metrics + 9-subject decision-k drift），旧 participation-field 实现保留为 sensitivity。

---

## ✅ Stage A 文档纠偏完成 2026-05-23 (本次)

Stage A 6 项 doc-only 修订完成 — framework banner 已升 v1.0.6（不再 deferred）。下面"完成了什么 / 关键决策点 / 没做什么"部分反映 Phase 2 v1.0.0 implementation 完成时状态；Stage A 之后的状态见此处覆盖。

| # | 任务 | 文件 | 状态 |
|---|---|---|---|
| A1 | AGENTS.md 拨正 H2 tier 冲突 | `AGENTS.md` "Topic 4 H2 input source order" + "Pre-registered hypothesis tier" | ✅ 完成 |
| A2 | cohort_run archive — 顶部 banner (user-added) + body 保持 v1.0 unchanged | `docs/archive/topic4/sef_itp_phase2/cohort_run_2026-05-23.md` | ✅ 完成（顶部 banner 由 user 添加，3 个 user-return v2 catch 全部 cross-cite；body 严格 v1.0 不修改 per banner directive + CLAUDE.md §5 pre-registration；agent 曾尝试加 6 个 body cross-cite paragraphs 指向 framework v1.0.6 amendment，advisor 指出违反 "body 不修改" 严格解读后已 trim 还原 — 严格交付：banner-only cross-citation） |
| A3 | framework §3.3 H3 v1.0.6 surgical clarification | `docs/topic4_sef_itp_framework.md` §3.3 | ✅ 完成（三层 verdict: R1 long-scale ≥60s independence + R2 endpoint stability + biophysical sub-shape descriptive；原 v1.0.5 字面 6-AND rule 保留作 audit trail） |
| A4 | USER_RETURN_BRIEFING 改语言（本文档） | `docs/archive/topic4/sef_itp_phase2/USER_RETURN_BRIEFING.md` | ✅ 完成（本节加入） |
| A5 | H4 当前 implementation 降级 SUPPLEMENTARY + H4 v1.1 stub | `docs/topic4_sef_itp_framework.md` §3.4 v1.0.6 amendment | ✅ 完成（当前 implementation `events_bool.mean.top-k` = participation field, supplementary only；H4 v1.1 stub 写明 rank-based endpoint + per-side 4 spatial radius metrics + decision-k drift on 9/23 swap subjects + k=3 fall-back on 14/23 none-swap） |
| A6 | framework §3.5 H5 plan 修订 | `docs/topic4_sef_itp_framework.md` §3.5 v1.0.6 amendment | ✅ 完成（主问题改 "招募/空间扩展 vs 单纯 rate 升高"，4 primary 指标 swap-k Jaccard ↓ / per-side radius ↑ / decision-k Δ ↑ / rate Δ secondary；统计合同 time-of-day matched baseline ≥12h / min 30 events / power floor 6 subjects 沿用 v1.0.5 lock） |

**Framework banner v1.0.5 → v1.0.6 lock 完成**（原 deferred item，Stage A 由 cohort_run 顶部 banner explicit naming "v1.0.6 surgical clarification" + user 长 message ratify 触发）。

**Stage A 修订原则**（advisor 2026-05-23 catch 严格守住）：
- R1 字面遵循 banner "long-scale ≥60s independence"，含 **60s AND 1800s 两条**（不偷偷只用 1800s 最宽松的一条，那是 cherry-picking）
- 不重新发明 banner 没授权的 CONTRADICTED 触发条件（曾草稿 "mid-scale POSITIVE → CONTRADICTED memory variant" — advisor catch 后删除，保留为未来 user-ratify 候选）
- 旧 v1.0.5 prose 全保留作 audit trail，不删除（CLAUDE.md §5 pre-registration 纪律）

**Stage A 之后仍 deferred 决策（5 项）**：
1. **H4 I_rate matched null 选哪个**（spec_amendment_2026-05-23.md 提议 circular_shift；待 user ratify）— independent of v1.0.6 amendment
2. **endpoint stability OR vs AND combinator** — framework §3.3 v1.0.6 已写 OR；待 user 显式 spec 永久锁确认
3. **epoch_hours 默认值 0.5 vs framework prose 1–2h** — implementation 默认 0.5h；待 user 同意 framework prose 改 "0.5–2h"
4. **short-scale POSITIVE memory variant 子 verdict label** — framework §3.3 v1.0.6 当前不引入；未来 cohort 出现 short-scale POSITIVE memory 再 user ratify
5. **H5 SUPPORTED 阈值 X/3 primary metrics 取值** — framework §3.5 v1.0.6 已锁 "3 primary recruitment/expansion + 1 secondary rate Δ" 结构，但 X 值具体取 2/3 (weak majority) 还是 3/3 (strict all-primary) 待 user ratify；详见 framework §3.5 v1.0.6 X/3 threshold authorization scope lock 段

**Stage A 之后建议下一步**：
1. **读 `docs/topic4_sef_itp_framework.md` 顶部 v1.0.6 banner + §3.3/§3.4/§3.5 v1.0.6 amendment 段**（5–10 分钟）
2. **目视检查 4 个 deferred 决策点**（特别是 H4 I_rate null，会影响 Stage B B5 cohort 重跑数字）
3. **同意启动 Stage B (H4 v1.1 code + cohort 重跑)**：调 advisor 一次 review B1–B6 plan，进入 TDD 实施
4. **Stage C (H5 v1.0.6 Phase 3 plan)** 排在 Stage B 完成后单独立 PR

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

### 0. ⚠️ H4 v1.0 是 participation field drift 不是 rank-based endpoint drift（user 2026-05-23 catch）

当前 `compute_local_endpoint(events_bool, labels, k)` 用 `mean(events_bool[labels==c], axis=0)` 取 channel-wise 参与率 top-k/bottom-k —— **这是 participation field top-k 漂移，不是 propagation rank endpoint 漂移**。两件事不一样：
- 参与率 top-k = "哪些通道这段时间常常出现在事件里"
- 传播 rank top-k = "哪些通道在事件里最先点火"（这才是 endpoint 的本意）

**因此**：当前 H4 PASS 信号（p=9.5e-7，d=1.50）**只能解读为 "participation field 不漂"，不能直接支撑 "传播 endpoint 不漂"**。改口径作为 supplementary。主线 H4 v1.1 重做 — 在 src/sef_itp_phase2.py 写新 `compute_local_rank_endpoint(ranks, bools, labels, k, valid_mask)`，per-epoch 用 mean masked rank 取 source/sink，加 4 个空间半径漂动指标（centroid RMS / mean pairwise / min enclosing ball 分 source/sink + source-sink centroid distance），再加 9 subject 的 decision-k drift（rank-displacement swap_sweep per-epoch）。

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

---

## 2026-05-23 user-return 后 v2 加注（user catch 3 个真问题）

用户回来后 2026-05-23 晚指出 3 个真问题：

1. **H3 三阶段是科学故事，不是失败**：lag-1 同模板 burst → 10–30s anti-clustering refractory → 60s+ independent —— 这是模板携带具体物理时间尺度（refractory ~30–60 秒）的证据，恰恰说明 framework 走得更具体。framework v1.0.5 §3.3 字面 6-metric AND-rule 把这种生理上合理的非独立模式触发 CONTRADICTED 是 prose under-specification。**v1.0.6 surgical clarification**（保留原 AND-rule 作审计踪迹 + 加 scale-stratified verdict sub-bullet，cohort 数字不变；CLAUDE.md §5 pre-registration 纪律不允许 rewrite）。
2. **H4 当前实现用错了 endpoint 定义**：是 participation field top-k 不是 rank-based propagation endpoint。当前 PASS 信号改口径为 supplementary，主线 H4 v1.1 重做（rank-based endpoint + 4 spatial radius metrics + 9-subject decision-k drift）。
3. **AGENTS.md tier 已纠正**：user 直接修订 L119+L125 把 Topic 4 H2 spatial 层 = primary cohort claim 和 PR-6 原 8-subset mechanism sanity tier cohort boundary 明确写清楚。

执行路线：A4 (this doc) → A2 (cohort_run_2026-05-23 加 banner caveat, body 不动) → B0 (Stage B v1.1 plan + advisor) → B1-B4 (TDD compute_local_rank_endpoint + spatial radius + decision_k drift + runner) → B5 (cohort 重跑) → B6 (framework v1.0.6 consolidates A3+A5+A6 + 2026-05-24 archive)。Stage C (H5 v1.1 plan) 排在 Phase 3 启动前单独立 PR。
