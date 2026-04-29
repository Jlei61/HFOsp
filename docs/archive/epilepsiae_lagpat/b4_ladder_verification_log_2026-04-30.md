# B.4 Ladder Verification Log — Epilepsiae lagPat backfill

> 创建：2026-04-30
> 计划档：[`epilepsiae_lagpat_backfill_plan_2026-04-29.md`](./epilepsiae_lagpat_backfill_plan_2026-04-29.md)
> 范围：plan §3 Task B.4 ladder 的 a/b/c 步骤实测验收日志。B.4.d 与 B.5 待跑。

## 状态总览

| step | 对象 | 记录数 | done / fail | median runtime | 验收 |
|---|---|---:|---:|---:|---|
| B.4.a | 253 / 25300102_0000 | 1 | 1 / 0 | 8.7 s（首跑）→ 8.9 s（`--force` 重跑） | PASS |
| B.4.b | 253 全 subject (512 Hz) | 268 | 267 + 1 skip / 0 | **11.0 s** | PASS |
| B.4.c | 548 全 subject (1024 Hz) | 147 | 147 / 0 | **49.4 s** | PASS |

`PASS` 的判据：`n_failed == 0`，cohort 内 `n_participating == 0` 列数 = 0，
`packedTimes` 与 lagPat 事件数对齐，min `n_participating` ≥ 1。

## 关键修复（B.4.a 复审触发）

`compute_lagpat_record` 之前会保留 `n_participating == 0` 的事件列（成因：
window 跨 200 s segment 边界、负 start、segment 太短）。修复：在 segment
循环之后用 `events_bool.sum(axis=0) > 0` 做 mask，同步过滤 `centroids`、
`events_bool` 与 `packed_times`，并在返回字典里加入过滤后的 `packedTimes`
键，让 `process_one_record` 落盘的 `*_packedTimes.npy` 与 `*_lagPat.npz`
事件数严格一致。

回归守卫位于 `tests/test_epilepsiae_lagpat_backfill.py
::test_compute_lagpat_record_shapes`：对每列 `n_participating > 0` 做 assert，
另在 `test_process_subject_writes_log_with_required_keys` 锁住 log JSON 字段。

提交：`7ec9236` (`fix(epilepsiae_lagpat): drop n_participating==0 events + add B.4.b subject loop`).

## B.4.a — 单 record 端到端（253 / 25300102_0000）

修复前：`n_events = 6`，per-event `n_participating = [0, 25, 19, 26, 19, 29]`（首列为空）。
修复后：`n_events = 5`，`n_participating = [25, 19, 26, 19, 29]`，最小值 19。
`packedTimes` 同步从 6 → 5 行；首条边界 window `[-0.165, 0.335]` 被丢弃。

事件 1（rank 头部）：
- HRB2 (lag=0.00 ms, rank 0)
- HLB4 (0.19 ms, rank 1)
- HRC3 (0.44 ms, rank 2)
- HRB3 (0.60 ms, rank 3)
- HRA1 (0.61 ms, rank 4) … 共 25 个参与通道

`start_t = 1079371059` → 2004-03-15 17:17:39 UTC。Unix epoch 范围 (1e9, 2e9) 通过；
note: 计划 §11 提到 “Epilepsiae 实际范围 2008-2012”，subject 253 实际是 2004，注记略偏。

## B.4.b — 253 全 subject (pure 512 Hz)

```
started_at  = 2026-04-29T16:17:13
completed_at = 2026-04-29T17:02:50   (wall 45 m 37 s)
n_records_total       = 268
n_records_done        = 267
n_skipped_existing    = 1            # 25300102_0000 来自 B.4.a --force
n_failed              = 0
median_record_seconds = 10.98        # min 0.32 s, max 13.09 s, mean 10.25 s, p90 11.22 s
```

cohort 体检（268 个 `_lagPat.npz` + `_packedTimes.npy`）：

```
records with empty event columns (BUG): 0
records with packedTimes/lagPat misalignment: 0
records with 0 events (legitimate, low signal): 24
total events across cohort: 4 936
events/record: min=0  max=143  median=6
min n_participating across all events: 13
unique chnNames: 29
```

通道集合（fixed 蒙太奇）：`HLA1-5, HLB1-5, HLC1-5, HRA1-5, HRB1-5, HRC1-4`（H + L/R + A/B/C + 1 位数字）。

## B.4.c — 548 全 subject (pure 1024 Hz)

```
started_at  = 2026-04-29T23:00:42
completed_at = 2026-04-30T01:00:39   (wall 1 h 59 m 57 s)
n_records_total       = 147
n_records_done        = 147
n_skipped_existing    = 0
n_failed              = 0
median_record_seconds = 49.40        # min 3.98 s, max 56.48 s, mean 48.96 s, p90 53.86 s
```

cohort 体检：

```
records with empty event columns (BUG): 0
records with packedTimes/lagPat misalignment: 0
records with 0 events (legitimate, low signal): 44
total events across cohort: 1 325
events/record: median=2
min n_participating across all events: 33
unique chnNames: 83
```

通道前缀（多 shaft 大网）：`GA*/GB*/GC*/GD* (32 grid)`、`HL1-10`、
`TBLA*/TBLB*/TBLC* + TBRA*/TBRB*/TBRC*`、`TLRA*/TLRB*/TLRC*`。

### 主验收 vs 次级期望

- **主**（plan §8 B.4.c）：sfreq 通用性 — `147/147 done, 0 failed`。**PASS**。
- **次**（plan §3 B.4.c "对比 253 chnNames overlap > 0"）：
  实测 overlap **= 0**，jaccard = 0.000。
  原因：253 与 548 是不同患者，电极标签**完全不共享 schema**：
  - 253: `H + L/R + A/B/C + N`（3 字母前缀）
  - 548: `HL + N`（无 A/B/C）+ 大量 grid / 颞底 / 颞外的 subject-private 标签
  含义：plan 的 “overlap > 0” 期望偏宽松假设；Epilepsiae 的电极命名在 patient
  之间是 private 的，跨 subject 不可对齐。**这是 informational，不是硬 fail**：
  Stage C 的通道相似度比较应理解为同一 subject 内 “新 lagPat 通道 vs 老 lagPat 通道” 的 jaccard，
  不是跨 subject。

## 运行成本估计修订

| sfreq | 测得 median | 来源 |
|---|---:|---|
| 512 Hz | 11.0 s/record | 253, n=267 |
| 1024 Hz | 49.4 s/record | 548, n=147 |

按 plan §1.1 的 sfreq 分布（16 纯 1024 + 1 纯 512 + 1 纯 256 已被 detector
跳过 + 2 mixed 1024+256 + 1 mixed 256+512 = 实际处理 ~3 600 records）：

```
253          268 records × 11.0 s ≈ 0.8 h
1024 Hz cohort   ~3 100 records × 49.4 s ≈ 42.5 h        (16 subject)
139 (512)         130 records × 11.0 s ≈ 0.4 h
mixed 1024 (384, 583)  128 records × 49.4 s ≈ 1.8 h
合计 single-thread  ≈ 45–50 h
N_JOBS=5            ≈ 9–10 h
```

→ plan §10b R1 的占位估计 “6 min/record × 3 700 / N_JOBS=5 ≈ 74 wall hours”
**实测偏高 7×**。R1 校验门槛 “实测 median 偏差 > 50% → 暂停重审 N_JOBS” 的
**触发条件已经满足**：1024 Hz 子集 49.4 s 比 6 min (≈ 360 s) 偏低 86%。但偏低
方向意味着工作量比预想小，无需限制并行度；Stage B.5 进入时**重新核对机器
内存** （147 records × CAR loader peak ~3 GB）即可，不需要重审 plan。

## Pending（不在本日志范围）

- **B.4.d** — 1 个 mixed-sfreq subject（384，65 records，仅 1024 Hz blocks 经过 detector）。
- **B.5** — Cohort batch driver + GNU parallel 脚本，全 cohort ~3 700 records。
- **Stage C / D** — 待 B.5 完成后开始。
