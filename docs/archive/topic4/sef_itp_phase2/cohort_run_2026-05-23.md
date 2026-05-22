# SEF-ITP Phase 2 Cohort 实跑 — 2026-05-23

> **状态**：v1.0.0 first cohort run. **pending user ratification** of H4 I_rate null method (spec amendment proposal — see [spec_amendment_2026-05-23.md](spec_amendment_2026-05-23.md))
> **plan**：`docs/superpowers/plans/2026-05-23-topic4-phase2-h3-h4-plan.md`
> **runner**：`scripts/run_sef_itp_phase2.py`
> **summarizer**：`scripts/summarize_sef_itp_phase2.py`
> **per-subject 输出**：`results/topic4_sef_itp/phase2_temporal_x_geometry/per_subject/<dataset>_<sid>.json`
> **cohort summary**：`results/topic4_sef_itp/phase2_temporal_x_geometry/cohort_summary.json`
> **flat CSV**：`results/topic4_sef_itp/phase2_temporal_x_geometry/cohort_subjects.csv`
> **测试**：见 §6；115 (Phase 1) + 39 (Phase 2) = 154 GREEN
> **framework**：`docs/topic4_sef_itp_framework.md` v1.0.5 — **not edited this round**（H4 I_rate null 是 science decision，等用户拍板）

---

## 0. Cohort 漏斗

继承 Phase 1 cohort（n=23）：

```
40 个 masked Phase 0a 病人 (yuquan 20 + epilepsiae 20)
    ↓ 过滤：stable_k = 2
34 个
    ↓ 过滤：masked PR-6 anchoring 出了结果
30 个
    ↓ 过滤：3D 坐标可用（mm）
23 个 = 8 Yuquan + 15 Epilepsiae
    ↓ 过滤：PR-7 burst diagnostic 可用
        (Task 0 把 15 个原本 PR-7 audit 通过 h2_negative_pass 但未跑 burst 的人补跑)
23 / 23 都有 PR-7 burst + pairing
```

**phase2 cohort = phase1 cohort = n=23**。

---

## 1. H3 — Mark independence + endpoint stability

### 测了什么（朴素话）

挑选传播模板的过程像不像"独立抛硬币"？相邻事件挑同一个模板的频率、不同时间尺度的同模板偏置、连续同模板的"连击长度"——这些指标如果完全随机独立，应该匹配 N2 marginal-preserving null。同时看：endpoint 通道集合在前半段 vs 后半段、奇数 block vs 偶数 block 之间稳不稳。

### 怎么测的

6 个 cohort-level TOST 等价检验，对比 ±δ_excess=0.05 带：

| 指标 | 来源 | 目标值 | 含义 |
|---|---|---|---|
| `lag1_same_excess` | PR-7 burst `lag1_same_excess.N2` | 0 | 相邻两次挑同模板比例减去 N2 期望 |
| `window_excess_10s` | PR-7 pairing `lift.N2.10.excess` | 0 | 10 秒窗口内同模板偏置 |
| `window_excess_30s` | PR-7 pairing `lift.N2.30.excess` | 0 | 30 秒窗口偏置 |
| `window_excess_60s` | PR-7 pairing `lift.N2.60.excess` | 0 | 60 秒窗口偏置 |
| `window_excess_1800s` | PR-7 pairing `lift.N2.1800.excess` | 0 | 30 分钟窗口偏置 |
| `run_length_lift` | PR-7 burst `lift.N2.run_length_lift` | 1 | 连击长度 vs N2 倍数 |

加端点几何稳定（来源 PR-6 anchoring `split_half_robustness.per_split.*.subject_mean_jaccard_endpoint`）：
- `endpoint_jaccard_first_half` 中位数 vs 阈值 0.7
- `endpoint_jaccard_odd_even` 中位数 vs 阈值 0.7
- **OR 组合**（项目惯例，AGENTS.md cross-PR `forward_reverse_reproduced` = split-half OR odd-even；advisor 提醒 2026-05-23 catch A）

每条 TOST 用 PR-7 addendum 的 bootstrap CI 方法（移植到 `src.sef_itp_phase2.tost_equivalence`），加 leave-one-out 稳健性检验（`cohort_tost_with_loo`；advisor catch C）。

### Integrated verdict 规则（framework v1.0.5 §3.3 lock）

| 6 条 mark-transition TOST | endpoint 稳定（OR 阈值 0.7）| verdict |
|---|---|---|
| 全部 equiv_pass | yes | **SUPPORTED** |
| 全部 equiv_pass | no | **NOT_SUPPORTED_GEOMETRY_UNSTABLE** |
| ≥1 失败且 LOO < 0.5 仍稳健失败 | yes | **CONTRADICTED** |
| ≥1 失败但 LOO ≥ 0.5（单人敏感）| yes | **NOT_SUPPORTED_MEMORY** |
| ≥1 失败 | no | **NOT_SUPPORTED_BOTH** |

**禁用** PASS / NULL / FAIL —— 防止 "PASS = 证明独立" 误读。措辞锁：

> ✅ "compatible with mark-independent sampling within tested precision"
> ❌ "证明独立 / proves mark-independence"

### 揭示了什么（cohort n=23）

**[FILL IN AFTER COHORT RUN]** — 见 cohort_summary.json `h3` section + cohort_subjects.csv。

预期填入：
- integrated_verdict = ?
- 6 条 TOST equiv_pass 比例 = ?
- endpoint_jaccard_first_half median = ?
- endpoint_jaccard_odd_even median = ?
- endpoint stable 比例（OR threshold 0.7）= ?
- 任何 leave_one_out_min_pass_rate < 0.5 的指标 → 触发 CONTRADICTED 分支
- 任何 leave_one_out_min_pass_rate ≥ 0.5 的失败指标 → 触发 NOT_SUPPORTED_MEMORY 分支

---

## 2. H4 — Normalized rate vs geometry instability

### 测了什么（朴素话）

把每个 subject 的录制切成短时段（epoch），看每个 epoch 的 HFO 事件率（events/h）漂得有多厉害，再看每个 epoch 的"端点通道集合"漂得有多厉害。如果 SEF-ITP 假设成立（空间结构稳定，触发频率不稳定），rate 应该比 geometry 漂得**多**很多。

### 怎么测的

每个 subject:

1. **Epoch 切片**：每 0.5 小时切一个 epoch（block 内 wall-clock 切），过滤 < 10 events 的 epoch；保留时间顺序
2. **per-epoch rate**：每 epoch 事件数 / 0.5 h
3. **per-epoch endpoint**：每 cluster 拿 epoch 内事件 → 取参与率 top-k=3 source + bottom-k=3 sink → 跟 global endpoint（PR-6 anchoring）算 Jaccard，再 cluster 平均
4. **I_rate** = `std(log(rate_obs)) / sqrt(matched null variance of std(log(rate)))`
5. **I_geom** = `std(1 - Jaccard_obs) / sqrt(matched null variance)`

#### Matched null 关键决策点（advisor catch B 2026-05-23）

Framework v1.0.5 §3.4 prose "shuffle epoch order, recompute std" 是**数学退化**的（std 对置换不变 → null_var = 0 → I_rate 未定义）。Phase 2 v1.0.0 实施了两种 null:

| Null 方法 | 描述 | 退化性 |
|---|---|---|
| **`epoch_order_shuffle`** (literal) | framework v1.0.5 §3.4 原文 | 必然退化 |
| **`circular_shift_within_block`** (proposed) | 每 block 随机偏移 Δ ∈ [0, epoch_seconds)，wrap-around，重切，重算 std | 非退化（条件：block 长 > epoch_seconds）|

**双方法都跑、都报、cohort summarizer 都出 verdict**。框架文档 **不动**，等用户拍板。

详情：[`spec_amendment_2026-05-23.md`](spec_amendment_2026-05-23.md).

#### I_geom matched null

每 epoch 在 `valid_mask=True` pool 里随机抽 endpoint_size=6 通道（half source / half sink），重算 Jaccard vs global。非退化（除非 valid pool 太小 → 抽 6 from 6 只有一种组合）。

### Verdict 规则（framework v1.0.5 §3.4 lock）

| Cohort 状态 | verdict |
|---|---|
| Wilcoxon p<0.05 (rate > geom) AND Cohen's d ≥ 0.30 | **PASS** |
| Wilcoxon p<0.05 (rate < geom) | **FAIL** |
| 其他 | **NULL** |
| n < 6 (finite I_rate ∩ finite I_geom) | **UNDERPOWERED** |

### 揭示了什么（cohort n=23, epoch_hours=0.5）

**[FILL IN AFTER COHORT RUN]** — 见 cohort_summary.json `h4` section。

两个 verdict 并列报告：

| Null 方法 | Wilcoxon p | Cohen's d | n_subjects (有限) | verdict |
|---|---|---|---|---|
| `epoch_order_shuffle_literal` | ? | ? | ? | ? |
| `circular_shift_within_block_proposed` | ? | ? | ? | ? |

中位数 I_rate (circular_shift) = ?
中位数 I_geom = ?

#### 已知 Epilepsiae 子样本限制

epoch_hours=0.5 设计是为了让 Epilepsiae 的 ~1h block 也能产出 ≥ 2 个 epoch（circular shift 才有自由度）。但 6-通道 valid pool 的 Epilepsiae 病人，`compute_I_geom_normalized` 的 null 也会退化（抽 6 from 6 = 1 种）。这种病人的 I_geom = inf 会被 `compute_h4_cohort_verdict` 的 `np.isfinite` 过滤掉。

#### Smoke 测试数据（确认 logic 正常）

| Subject | n_epochs | I_rate (circshift) | I_geom |
|---|---|---|---|
| yuquan_chengshuai | 24 (1h epoch) / TBD (0.5h) | 53.2 | 3.6 |
| epilepsiae_1073 | 439 (0.5h epoch) | 101.8 | inf (6-ch valid pool) |

---

## 3. Cohort 整体一句话 verdict

**[FILL IN AFTER COHORT RUN]**

预期形式：
- H3: 在 n=23 cohort 上 X/6 条 mark-transition 指标 TOST 等价通过；endpoint Jaccard 中位数（first_half / odd_even）= ? / ?；OR-combinator 下 endpoint stable 比例 = ?；integrated verdict = SUPPORTED / NOT_SUPPORTED_* / CONTRADICTED
- H4 (circular_shift, proposed): n_finite=?，Wilcoxon p=?，Cohen's d=?，verdict=?

---

## 4. 与 framework v1.0.5 的关系

- H3 verdict 用 SUPPORTED 系列（**禁用** PASS/NULL/FAIL）
- H3 措辞 "compatible with mark-independent sampling within tested precision"
- H4 verdict 用 PASS/NULL/FAIL/UNDERPOWERED 标准 family
- δ_excess = 0.05 已锁，未调整
- endpoint k=3 已锁
- **framework §3.4 H4 I_rate matched null prose 不动**（等待用户决定）

---

## 5. 实施纪律 & 复用映射（CLAUDE.md §6 / §6.1）

| Phase 2 用法 | 来源 | 问题匹配？|
|---|---|---|
| `tost_equivalence` | 移植 `scripts/pr7_addendum_p3_equivalence.py:123` | ✅ TOST median equivalence vs ± δ band, target=0 or 1 |
| `lag1_same_excess` / `run_length_lift` | PR-7 `compute_burst_diagnostic_with_nulls` 输出 | ✅ same lag-1 / run-length metric vs N2 null |
| window excess | PR-7 `pairing_with_nulls.lift.N2.*.excess` | ✅ same windowed excess vs N2 |
| endpoint stability | PR-6 anchoring `split_half_robustness.per_split.*.subject_mean_jaccard_endpoint` | ✅ same Jaccard recall question |
| per-epoch endpoint | Phase 2 局部 `compute_local_endpoint`（不调用 PR-6 anchoring 全 helper）| ⚠️ 不同 gate；写一个 thin helper |

advisor 2026-05-23 catch 整合：
- A. endpoint 稳定组合 = OR（不是 AND）→ 测试 `test_h3_integrated_verdict_or_combinator` 锁住
- B. H4 I_rate null 是 science decision → 双方法并跑、双 verdict、framework 不动
- C. LOO populator 单独任务 → `cohort_tost_with_loo` + summarizer 自动调用，CONTRADICTED 分支可触发
- D. lagpat 路径 resolver 提升到 src 公共 → `src.sef_itp_phase1.resolve_lagpat_subject_dir`

---

## 6. 测试 + 实现

### 实现文件

- `src/sef_itp_phase2.py` — 11 个 pure functions + dataclass：
  - `extract_window_excess_from_pairing`, `extract_lag1_and_runlength_from_burst`, `extract_endpoint_jaccard_from_anchoring`
  - `tost_equivalence` (ported), `cohort_tost_with_loo`
  - `compute_h3_integrated_verdict`
  - `slice_events_into_epochs`, `compute_local_endpoint`, `endpoint_jaccard`
  - `compute_I_rate_normalized` (literal), `compute_I_rate_normalized_circular_shift` (proposed)
  - `compute_I_geom_normalized`
  - `compute_h4_cohort_verdict`
- `scripts/run_sef_itp_phase2.py` — per-subject runner
- `scripts/summarize_sef_itp_phase2.py` — cohort summarizer
- `src/sef_itp_phase1.py` — 新增 `resolve_lagpat_subject_dir` 公共函数（Phase 1 runner 改 import 同名 alias）

### 测试

- `tests/test_sef_itp_phase2.py`：**39 GREEN** (覆盖所有 helpers + 各 verdict 分支)
- `tests/test_sef_itp_phase1.py`：**115 + 1 (新加 resolver test) = 116 GREEN**

### Commits (linear history)

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
results(topic4 phase2): full n=23 cohort run  [next]
```

---

## 7. 待用户决定（STOP-AT-13 deliverable list）

advisor 2026-05-23 catch B 把 Task 14（framework doc 编辑）标为 DEFERRED。**所有以下事项等用户回来再做**：

1. **H4 I_rate matched null 选哪个**？(circular_shift / Poisson / gamma / cross-epoch shuffle / 其他)
2. **如果选 circular_shift，需不需要修订 framework §3.4 prose**？（建议：是，但措辞锁要先复审）
3. **framework banner 是否升 v1.0.6** with Phase 2 完结 marker？
4. **是否需要 sensitivity sweep**（epoch_hours ∈ {1.0, 0.5, 0.25}、endpoint_k ∈ {2, 3, 4, 5}）？

详细决策列表见 [`spec_amendment_2026-05-23.md`](spec_amendment_2026-05-23.md) §6。

---

## 8. 内部归档代号映射（CLAUDE.md §8 朴素话风格）

- H3 mark transition = 模板挑选是不是 lag/window/run-length 看起来独立
- H3 endpoint stability = 时间二分半数据 endpoint 还一不一样
- H4 I_rate = 事件率漂动的归一化幅度（log-rate std / matched null std）
- H4 I_geom = endpoint 几何漂动的归一化幅度（1-Jaccard std / matched null std）
- TOST = two one-sided test，bootstrap CI 在 ±δ 带内才算等价
- δ_excess = 0.05 = framework time-lock 的等价带宽
- N2 null = PR-7 marginal-preserving permutation null（保留每 cluster 总事件数）
- circular_shift_within_block = block 内随机偏移事件时间再重切 epoch 的 null
- epoch_order_shuffle = 事件率序列的 epoch 顺序随机置换（退化 null，只为 spec audit 保留）
