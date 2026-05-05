# scripts/archive/epilepsiae_lagpat

Epilepsiae 1024 Hz detector drift / lagPat backfill 期间产出的一次性诊断脚本。
findings 已归档到对应的 docs/archive 文档；保留脚本仅作为复现路径。

## 脚本与对应文档

| 脚本 | 用途 | 对应归档 doc |
|---|---|---|
| `diag_1024hz_smoke_635.py` | 1024 Hz subject 635 上游偏差 smoke：Path A vs legacy 第一个 200s chunk | `docs/archive/epilepsiae_lagpat/detector_drift_root_cause_2026-05-03.md` |
| `diag_1024hz_chunk_isolated.py` | notch 在整段 vs 单 chunk 的差异定位 | `docs/archive/epilepsiae_lagpat/detector_drift_root_cause_2026-05-03.md` |
| `diag_1024hz_event_envelopes.py` | legacy keep / Path A reject 事件包络对比（pick/side 比值） | `docs/archive/epilepsiae_lagpat/detector_drift_root_cause_2026-05-03.md` |
| `diag_1024hz_path_a_boundaries.py` | Path A 在 TLA1 上 pre-side / with-side 事件边界 | `docs/archive/epilepsiae_lagpat/detector_drift_root_cause_2026-05-03.md` |
| `diag_1024hz_float32.py` | cusignal float32 vs float64 复刻 legacy 计数测试 | `docs/archive/epilepsiae_lagpat/detector_drift_root_cause_2026-05-03.md` |

## 相关产出文档（建议先读）

- `docs/archive/epilepsiae_lagpat/detector_drift_root_cause_2026-05-03.md` — 检测漂移根因总结
- `docs/archive/epilepsiae_lagpat/legacy_replication_audit_2026-05-03.md` — Stage B 合同违例审计
- `docs/archive/epilepsiae_lagpat/stage_d_smoke_2026-05-02.md` — Stage D smoke
- `docs/archive/epilepsiae_lagpat/b4_ladder_verification_log_2026-04-30.md` — B4 ladder 验证
- `docs/archive/epilepsiae_lagpat/epilepsiae_lagpat_backfill_plan_2026-04-29.md` — 总计划

## 状态

`fix(epilepsiae_lagpat): Path A — detector legacy_align + FIR-801 notch (12/12)`（commit 6027281）已采纳；
当前生产入口是 `scripts/run_hfo_detection.py --dataset epilepsiae` 与
`scripts/run_epilepsiae_lagpat_backfill.py`。本目录脚本仅历史复现用。
