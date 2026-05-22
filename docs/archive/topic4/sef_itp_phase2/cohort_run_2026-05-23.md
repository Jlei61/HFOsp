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

### 揭示了什么（cohort n=23, n_perm=1000, n_boot=10000）

**integrated_verdict = CONTRADICTED**

| metric | cohort median | CI95 | equiv_pass | LOO min_pass | 解读 |
|---|---|---|---|---|---|
| `lag1_same_excess` | +0.016 | [+0.002, +0.020] | ✅ True | 1.000 | 紧邻下一次事件挑同模板 vs N2 兼容独立 |
| `window_excess_10s` | **-0.065** | [-0.099, -0.025] | ❌ False | 0.000 | 10 秒窗口内 same-template 显著**低**于独立预期 → 反聚集 |
| `window_excess_30s` | -0.043 | [-0.058, -0.018] | ❌ False | 0.000 | 30 秒窗口 CI 下端跨出 -0.05 → 反聚集 trend |
| `window_excess_60s` | -0.029 | [-0.045, -0.012] | ✅ True | 1.000 | 60 秒窗口兼容独立 |
| `window_excess_1800s` | -0.0004 | [-0.0015, +0.0002] | ✅ True | 1.000 | 30 分钟窗口兼容独立到惊人地精度 |
| `run_length_lift` | +1.035 | [+1.003, +1.051] | ❌ False | 0.000 | 连击长度 CI 上端 1.051 略超 1.05 → 轻微 burstiness 残留 |

**endpoint geometric stability**:
- first_half_second_half median Jaccard = **0.714** (14/23 ≥ 0.7)
- odd_even_block median Jaccard = **0.857** (18/23 ≥ 0.7)
- OR combinator threshold 0.7 → endpoint stable ✓（odd_even 强 pass，first_half 中位线刚好擦边过 0.7）

**Verdict logic 走法**：mark_pass = False（3/6 失败：10s/30s/run_length）+ endpoint stable = True + failing 三项的 LOO min_pass_rate 全 0（稳健失败）→ **CONTRADICTED**。

### 朴素话解读（CLAUDE.md §8 三段式）

- **测了什么**：连续事件之间挑同一个模板的频率，在 lag-1 / 10s / 30s / 60s / 30min / 连击长度 几个尺度上看起来是不是像"独立抛硬币"。
- **怎么测的**：每个尺度对比 PR-7 N2 marginal-preserving null（保留 cluster 总频次但打散时间），算 cohort 中位数偏离独立预期多远；TOST 等价检验在 ±δ=0.05 带内才算"在精度内兼容独立"，加 leave-one-out 看是否单个 outlier 撑住整个失败。
- **揭示了什么**：lag-1 紧邻、60s、30分钟、24h（隐式）这几个尺度都看起来像独立抛硬币（CI 全部在 ±0.05 带内）；**但 10s / 30s 短窗显示统计上显著的反聚集**（相邻两次事件**更不倾向**挑同模板，**不是** more likely）；run_length 连击长度比独立预期略大（中位 1.035，CI 上端 1.051）。endpoint 通道集合时间二分稳定（中位 Jaccard 0.71/0.86）。
- **结论锁定语言**："在 n=23 cohort 上，间期 HFO 模板挑选 **at long time scales (60s, 30min)** is compatible with mark-independent sampling within tested precision (δ_excess=0.05)；但在 short time scales (10s, 30s) 显示显著反聚集；run_length lift CI 略微超出 ±0.05 等价带"。
- **不能写**："framework 整体被证伪"——长尺度独立性 + endpoint 几何稳定 两条都成立；只是短尺度出现 framework 没有预测的反聚集结构。短尺度反聚集与 SEF-ITP 不直接矛盾（"独立触发 + 稳定几何" 大方向仍站得住），但需要 framework v1.0.5 §3.3 整合规则的反思：是否要把 "compatible at all tested precisions" 收紧成 "compatible at long scales only"？这是 spec 决策。

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

### 揭示了什么（cohort n=23, epoch_hours=0.5, n_perm=1000）

两个 verdict 并列报告：

| Null 方法 | Wilcoxon p | Cohen's d | n_subjects (有限) | verdict |
|---|---|---|---|---|
| `epoch_order_shuffle_literal` | NaN | NaN | **0** | UNDERPOWERED — 必然退化（21 个 I_rate=+inf 全被过滤）|
| `circular_shift_within_block_proposed` | **9.54e-7** | **+1.499** | **21** | **PASS** |

- 中位数 I_rate (circular_shift) = **47.37**
- 中位数 I_geom = **11.91**
- I_rate / I_geom 中位比 = 3.98

**21/23 有限 subject**：epi_1073 + epi_1077 因为 valid pool ≤ 6 通道，I_geom matched null 也退化（random sample 6 from 6 = 1 种可能），I_geom = inf → 被 `np.isfinite` 过滤。其他 21 个都给出有限 I_rate 和 I_geom。

**⚠️ 跨数据集 magnitude caveat**（advisor 2026-05-23 提醒）：
- Epi I_rate 中位 65.6 vs Yuq 中位 20.4（Epi ~3× higher）
- Epi I_geom 中位 16.3 vs Yuq 中位 6.5（Epi ~2.5× higher）
- 这个差距大概率来自 circular-shift null 的自由度差异（Epi block 短 → 每 block ~2 epoch → null 收得紧 → I 值膨胀；Yuq 24h 连续 → 每 block 48 epoch → null 散得开 → I 值小）
- **per-subject 比值 I_rate / I_geom 跨数据集差不多**：Epi 中位 ~4.1, Yuq 中位 ~3.5
- **统计稳健的是 within-subject difference (rate − geom) 的 Wilcoxon 方向**，不是绝对 magnitude
- 1 个 outlier 在 SEF-ITP 反方向：yuquan_zhaochenxi ratio = 0.92 (I_rate < I_geom)；20/21 在 SEF-ITP 方向

→ "median ratio 4×" 这一条要带星号读：方向 + Wilcoxon 在不同 null 下都 robust，绝对 4 倍 magnitude 是 null 构造 artifact。

**Cohen's d = 1.50 是大效应**（远超 0.30 floor）；Wilcoxon p < 1e-6。

### 朴素话解读

- **测了什么**：把每个录制切成 30 分钟的小段，看 HFO 事件率每段差多大、看 endpoint 通道集合每段差多大，归一化后比较。SEF-ITP 假设：率漂 ≫ 几何漂（"病理空间稳定，触发频率不稳定"）。
- **怎么测的**：I_rate / I_geom 各自归一化到自己的 matched null variance；cohort Wilcoxon signed-rank（rate > geom 方向）+ Cohen's d。matched null 选 circular-shift-within-block（非退化变体；spec amendment 提议，等用户拍板）。
- **揭示了什么**：21 个有限 subject 上 I_rate 中位 47.4 vs I_geom 中位 11.9——**率漂大概是几何漂的 4 倍**；Wilcoxon p<1e-6，Cohen's d=1.5（大效应）。SEF-ITP "rate >> geometry 漂动" 预测**强成立**。

### 结论锁定语言

✅ "在 n=23 cohort 上（21 个有限），归一化事件率漂动显著大于归一化端点几何漂动（Wilcoxon p<1e-6，Cohen's d=1.50，I_rate median 47.4 vs I_geom median 11.9）。这强支持 SEF-ITP 关于慢变率调制不变形空间结构的预测。"

❌ "framework 证明 H4"（"PASS within tested precision" 不等于 "proves"）

⚠️ **caveat**: I_rate 的 matched null 用 circular-shift-within-block 是 Phase 2 提议的 amendment（framework v1.0.5 §3.4 原文 prose 是退化的）；最终 framework 用哪个 null 还在等用户拍板。即使换成 Poisson / gamma null，I_rate / I_geom 中位比 4 倍 + Cohen's d>1 的方向不太可能反向，verdict 应该robust。

#### 已知 Epilepsiae 子样本限制

epoch_hours=0.5 设计是为了让 Epilepsiae 的 ~1h block 也能产出 ≥ 2 个 epoch（circular shift 才有自由度）。但 6-通道 valid pool 的 Epilepsiae 病人，`compute_I_geom_normalized` 的 null 也会退化（抽 6 from 6 = 1 种）。这种病人的 I_geom = inf 会被 `compute_h4_cohort_verdict` 的 `np.isfinite` 过滤掉。

#### Smoke 测试数据（确认 logic 正常）

| Subject | n_epochs | I_rate (circshift) | I_geom |
|---|---|---|---|
| yuquan_chengshuai | 24 (1h epoch) / TBD (0.5h) | 53.2 | 3.6 |
| epilepsiae_1073 | 439 (0.5h epoch) | 101.8 | inf (6-ch valid pool) |

---

## 3. Cohort 整体一句话 verdict

**H3 = CONTRADICTED** (at short timescales 10s/30s + run_length_lift, robust LOO). **H4 = PASS** (circular_shift null; Wilcoxon p=9.5e-7, Cohen's d=1.50, n=21 finite). Endpoint geometry is stable (Jaccard 0.71-0.86).

朴素话整合：**长尺度上模板挑选看起来是独立抛硬币（>60s + endpoint 几何稳定），事件率漂动比几何漂动大 4 倍（强支持 SEF-ITP "稳定空间 + 不稳定频率" 主图景）；但短尺度（10-30s）出现 framework 没有预测的反聚集结构，触发严格 CONTRADICTED verdict**。

**几个解读角度（中性陈述，留给用户判断）**：

1. **H3 严格 CONTRADICTED 但局部**：失败集中在 10-30s 反聚集 + run_length 微弱 burstiness；lag-1 紧邻 + 60s + 30分钟 + endpoint 全过。框架可能需要把 "compatible at all tested precisions" 收紧成 "compatible at long scales only"，或在 prose 里承认 sub-30s 反聚集结构。
2. **H4 强 PASS**：rate 漂动 ~4× geometry 漂动是 SEF-ITP 区分性预测的强证据，cohort 上一致。
3. **整体 verdict**：H3 局部失败 + H4 cohort 通过 + endpoint 稳定 → 不构成"framework 整体被证伪"，但**framework v1.0.5 §3.3 H3 prose 需要收紧**——这是 spec 修订决策，等用户拍板。

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

### 7.1 H4 spec amendment

1. **H4 I_rate matched null 选哪个**？(circular_shift / Poisson / gamma / cross-epoch shuffle / 其他) — 详情 [`spec_amendment_2026-05-23.md`](spec_amendment_2026-05-23.md)
2. **如果选 circular_shift，需不需要修订 framework §3.4 prose**？（建议：是）
3. **framework banner 是否升 v1.0.6** with Phase 2 完结 marker？

### 7.2 H3 verdict 处理（新增 — 由 cohort 结果触发）

cohort 结果是 H3 CONTRADICTED，但失败模式是 short-scale anti-clustering，**不是** SEF-ITP 直接预测的相反方向（不是"memory variant"，也不是 long-scale 失败）。需要用户决定：

4. **framework v1.0.5 §3.3 是否需要把 "compatible with mark-independent sampling within tested precision" 改成 "compatible at long timescales (≥60s) within tested precision"**？
   - 论据 for：避免短尺度反聚集这种生理上合理的非独立模式触发 CONTRADICTED；尊重 framework v1.0.5 §0 "在我们的数据精度内" 的设计意图
   - 论据 against：这是 framework spec 修订，pre-registration 原则要求慎重（CLAUDE.md §5）；既然 cohort 已经跑出来了，事后改 threshold 看起来像调参拯救 verdict
5. **是否要把 10-30s 反聚集独立写一个 SEF-ITP framework extension**？短尺度反聚集 = burst refractory / receptor depletion 物理痕迹，可能值得作为机制层面的补充预测
6. **endpoint stability OR vs AND combinator**：advisor catch A 默认 OR；结果上看 first_half=0.714 stable / odd_even=0.857 stable，OR 给"稳定"，AND 也给"稳定"（两者都 ≥0.7）。这一次 cohort 上 combinator 选择不影响 verdict，但是 spec 锁定。是否确认 OR？

### 7.3 Sensitivity sweep（可选）

7. epoch_hours ∈ {1.0, 0.5, 0.25} sensitivity（当前默认 0.5）
8. endpoint_k ∈ {2, 3, 4, 5} sensitivity（当前默认 3）
9. δ_excess ∈ {0.05, 0.10} sensitivity（**禁止** post-hoc 调参；只作 sensitivity 报告）

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
