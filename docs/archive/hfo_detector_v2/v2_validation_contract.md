# HFO Detector v2 — 三层验收契约

> **状态**：Phase 1 deliverable，2026-05-05 起生效（**阈值预注册时间点**，cohort
> detection 跑之前）。
> **配套文档**：
> - 算法定义：`docs/archive/hfo_detector_v2/v2_specification.md`
> - 执行计划：`docs/archive/hfo_detector_v2/v2_cohort_rebuild_plan_2026-05-05.md`

---

## ⚠️ 适用范围声明（必须先读）

> ⚠️ 本契约衡量的是 **pipeline 内部自洽**，不是 **生物学有效性**：
>
> - **Layer A** = detector 自己的 rule 自洽（duration window 命中率、side-rejection
>   比值、deterministic 跨次跑）。
> - **Layer B** = packing 在群体事件层稳定（rank 在 split-half / odd-even 上一致）。
> - **Layer C** = 下游 PR-1 / PR-2.5 在 v2 自身上 split-half / odd-even reproducibility。
>
> **没有任何一层声明 "v2 事件是真实生理 HFO"**。这只能由 ground-truth 标注 / 外部
> 独立测度（reviewer 标注 / 跨模态对照）解决。所以引用 v2 cohort 结论时必须用
> **"v2 内部自洽通过"** 而非 "事件已验证"，**"该 subject 的 propagation 在 v2 数据上
> 稳定"** 而非 "该 subject 真有 propagation"。

---

## 阈值预注册（locked 2026-05-05）

- **锁定时间**：**2026-05-05**，**在 cohort detection 跑之前**。锁定时间记入 git
  history（commit timestamp 即为锁定 stamp）。
- **失败 subject 处理**：
  - **不剔除主分析**。fail subject 进 `large_drift` 列表，与主分析并列报告。
  - cohort 论述用 "PASS rate" + "large_drift 列表" **双轨披露**，禁止只报 PASS rate
    不披露 fail。
- **敏感性曲线**：每个层 PASS rate 在阈值 **±20%** 上的曲线一并报告，写入
  `results/hfo_detector_v2/validation/cohort_summary.json`。
- **修订规则**：cohort 跑完后若发现阈值需调，必须新建 archive doc 注明 reason，并
  在新阈值下**重新计算所有 cohort PASS**。禁止悄悄改契约阈值。一旦改了阈值，原阈
  值下的 cohort PASS rate 必须同时披露作为对照。

---

## Layer A — 单通道事件质量（per subject）

**目标**：确认 detector 自己的 rule 在 v2 输出上自洽——duration filter 命中、
side-rejection 比值合理、跨次跑 deterministic。

| 指标 | 计算 | 角色 | PASS 条件 |
|------|------|------|-----------|
| `dur_in_band_frac` | events with `50 ms < dur < 200 ms` 的比例 | PASS gate | ≥ 0.99（duration filter 之后理论 100%；浮点 / 边界容差） |
| `peak_side_ratio_p25` | events 的 `pick_mean / side_mean` 25 分位 | PASS gate | ≥ 2.0（side rejection 阈值，所有保留 event 必须 ≥ 2，p25 < 2 说明实现漏洞） |
| `threshold_margin_p50` | events 的 `(env_max - threshold) / threshold` 中位数 | PASS gate | ≥ 0.5（事件应 ≥ 50% 余量，不只是擦边） |
| `timestamp_jitter_p99` | 同 subject 跑两遍后，每个 event `t_start` 与最近 event 差值的 99 分位 | PASS gate | ≤ 1 sample（~1 ms @ 1024 Hz）— deterministic 检验 |
| `strong_chn_count_match` | 同 subject 跑两遍后，**top-10 通道**的 `events_count` 完全一致比例 | PASS gate | = 1.0（GPU 应 deterministic）|

### Layer A 抽样策略

- 每个 record 取 **3 个 evenly-spaced 200s window**：first / middle / last。
- 选择基线：原计划 "first 200s only" 在 HFO 非平稳条件下覆盖不足；改为 first /
  middle / last 三段，让 Layer A 看到 record 内不同时段的 detector 自洽性。
- Recording 短于 `3 × CHUNK_SEC = 600s` 时回退到首段 `[0, 200s]` 单窗口。
- 实现：`scripts/v2_validate_layer_a.py::_window_starts_for_record`，单元测试在
  `tests/test_v2_validate_layer_a.py`。

### Layer A PASS 判定

每个 subject 五个 PASS gate 全部满足 → Layer A PASS；任何一项 fail → Layer A FAIL，
进 `large_drift` 列表。

---

## Layer B — 群体事件 packing 后质量（per subject）

**目标**：确认 packing 把单通道 event 合并成 group event 后，n_participating
分布合理、pack window 时长合理、并且 channel rank 在 split-half / odd-even 子集
上一致。

| 指标 | 计算 | 角色 | PASS 条件 |
|------|------|------|-----------|
| `n_participating_p50` | packed group events 的 `n_participating` 中位数 | PASS gate | ≥ 2（必须有跨通道协同，单通道事件不该构成 packing 主体） |
| `n_participating_p10` | 10 分位 | PASS gate | ≥ 1（不全是孤立通道事件） |
| `pack_window_width_p50` | packed group event window 时长（秒）中位数 | PASS gate | 50–500 ms（合理 group event 时长范围） |
| `splithalf_event_rank_corr` | 上下半各自 lagPat → channel rank 的 Spearman 相关 | PASS gate | ≥ 0.7（强通道 rank 稳定） |
| `oddeven_event_rank_corr` | 奇偶 event 各自 lagPat → channel rank 的 Spearman 相关 | PASS gate | ≥ 0.7 |
| `chunk_boundary_event_frac` | start ∈ `[n*200 - 2, n*200 + 2]` s 的 group events 占比（chunk 边界附近代理 merge_overhead） | descriptive only | 报告 + cohort 分布，不入 PASS |

### `merge_overhead` 降级说明

原计划用 `merge_overhead`（chunk 边界 merge 行为指标）作 PASS gate，但
`_legacy_rehist_events_by_packing` **不直接暴露 chunk-cross 的合并 telemetry**——
packing 内部把 chunk 边界事件按时间戳处理，外部只能看到结果不能看到合并过程。

降级方案：用 `chunk_boundary_event_frac`（chunk 边界 ±2 s 内 group event 占比）
作为同失败模式的代理——若 packing 因 chunk 边界 merge 出错，会表现为 "事件密集
聚集在 200s / 400s / 600s ... 边界附近"。该指标作为 **descriptive only**，不作
PASS 门槛；cohort 分布写入 `cohort_summary`，作为后续诊断 hint。

Layer B PASS 由前 5 个指标共同决定（**5/5 PASS gate**）。

---

## Layer C — 科学下游 v2 自身 reproducibility（per subject + cohort）

**目标**：确认下游 PR-1 / PR-2.5 propagation 分析在 v2 自身数据上 split-half /
odd-even 复现，证明科学结论不依赖具体一次跑的随机数值漂移。

调用：`src/interictal_propagation.py::compute_time_split_reproducibility`（line 1734）
作用于 v2 输出的 lagPat / propagation per-subject JSON。

| 指标 | 计算 | 角色 | PASS 条件 |
|------|------|------|-----------|
| `time_split_grade` | `compute_time_split_reproducibility` 整体评级（split-half 与 odd-even 的整体 grade） | PASS gate | `strong` 或 `moderate` |
| `forward_reverse_reproduced_strict` | split-half **AND** odd-even 都把 forward / reverse template 复现 | PASS gate | `True` |
| `forward_reverse_reproduced_lenient` | split-half **OR** odd-even（沿用 PR-2.5 历史定义） | descriptive only | 报告 + cohort 分布，不入 PASS |
| `stable_k_consistent` | adaptive cluster `stable_k` 在 split-half / odd-even 上一致或差 ≤ 1 | PASS gate | `True` |

### `_strict` vs `_lenient` 关键澄清

PR-2.5 历史 `forward_reverse_reproduced` 的接受定义是 **OR**（split-half ∥ odd-even）——
见 `docs/archive/topic1/interictal_group_event_internal_propagation.md` PR-2.5 段，
8/9 subjects 是 OR 结果。

v2 Layer C **强化**为 `_strict = AND`（split-half 与 odd-even 必须都复现），原 OR
字段保留并改名 `_lenient`，仅作描述与历史 cross-reference。

- **PASS 门槛用 `_strict`**：v2 是新 cohort，门槛比历史 PR-2.5 严，避免单一切分意外
  通过把 fail subject 漏到主分析。
- **cohort 报告必须把 strict 与 lenient 数都列出**：例如 "Layer C PASS
  (`_strict`) = 24/30；`_lenient` = 28/30；4 个 subject 在 strict 下 fail 但
  lenient 下 pass，列入 `large_drift`"。
- **不要把 strict / lenient 数混在一起报告**——读者必须看到两个数才能判断"v2 比
  PR-2.5 更严的代价"。

### Layer C PASS 判定

每个 subject 三个 PASS gate（`time_split_grade`、`forward_reverse_reproduced_strict`、
`stable_k_consistent`）**全部满足** → Layer C PASS；任何一项 fail → Layer C FAIL，
进 `large_drift` 列表。`_lenient` 字段不影响 PASS 判定，仅用于披露。

---

## Cohort 验收

**三层 PASS 数 / 总 subjects 各 ≥ 0.85**（容许 ~3 outlier on 20-subject Epilepsiae
cohort）。

- 三层独立判定：subject 可在 Layer A PASS、Layer B FAIL、Layer C PASS。
- `large_drift` 列表 = 任一层 fail 的 subject 集合。
- **失败 subject 与主分析并列报告**：cohort summary 必须有
  `pass_rate / fail_count / large_drift_subjects` 三栏并列；禁止静默剔除 fail
  subject 后只报 PASS rate。

### 报告 schema（`results/hfo_detector_v2/validation/cohort_summary.json`）

```json
{
  "locked_at": "2026-05-05",
  "cohort_size": 20,
  "layer_a": {
    "pass_count": 18,
    "pass_rate": 0.90,
    "pass_subjects": [...],
    "fail_subjects": [...],
    "metric_distributions": {...},
    "sensitivity_curve": {
      "threshold_minus_20pct": {...},
      "threshold_plus_20pct": {...}
    }
  },
  "layer_b": {...},
  "layer_c": {
    "pass_count_strict": 24,
    "pass_rate_strict": 0.80,
    "pass_count_lenient": 28,
    "pass_rate_lenient": 0.93,
    "fail_subjects_strict": [...],
    "subjects_strict_fail_lenient_pass": [...],
    "metric_distributions": {...},
    "sensitivity_curve": {...}
  },
  "large_drift_subjects": [...]
}
```

---

## 引用规范

- ✅ "v2 cohort 在 Layer A / B / C 上 18/20、19/20、24/30 subjects 通过自洽检查"
- ✅ "subject X 的 propagation 在 v2 数据上 split-half + odd-even 都 strict
  reproducible"
- ✅ "Layer C `_lenient` PASS = 28/30；`_strict` PASS = 24/30；4 个 subject 进
  `large_drift`"
- ❌ "v2 cohort 验证了事件是真实 HFO"（v2 任何一层都不声明这条）
- ❌ "subject X 在 v2 上有 propagation 因此该 subject 真有 propagation"
- ❌ "Layer C 通过 28/30"（缺 strict / lenient 区分；必须双轨披露）

---

## Cross-references

- `docs/archive/hfo_detector_v2/v2_specification.md` — v2 算法定义 + disclaimer 同款
- `docs/archive/hfo_detector_v2/v2_cohort_rebuild_plan_2026-05-05.md` — Phase 2–9
  实现细节（Layer A/B/C 提取器 TDD、cohort detection、synRefine + lagPat backfill、
  synchrony + propagation 重建、cohort 总结）
- `docs/archive/topic1/interictal_group_event_internal_propagation.md` — PR-2.5
  `forward_reverse_reproduced` 历史 OR 定义（v2 `_lenient` 兼容字段）
- `src/interictal_propagation.py::compute_time_split_reproducibility` — Layer C
  调用入口
- `results/_legacy_2021_readonly/README.md` — 21 年只读 detection artifact
