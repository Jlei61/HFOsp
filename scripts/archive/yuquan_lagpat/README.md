# scripts/archive/yuquan_lagpat

Yuquan 24h same-source 重对齐期间的 phase D / E / F-1 一次性运行脚本与 watchdog。
对应的检测漂移定位与重打包结果已归档；保留脚本仅作复现/取参考。

## 脚本与对应文档

| 脚本 | 用途 | 对应归档 doc |
|---|---|---|
| `_phaseD_run_references.sh` | Phase D：3 个 reference subject 用 `legacy_align=True` 重检测 | `docs/archive/yuquan_lagpat/yuquan_detector_drift_phaseD_results.md` |
| `_phaseE2_run_packing.sh` | Phase E-2：24 subject 同源 lagPat / packedTimes 批量打包 | `docs/archive/yuquan_lagpat/yuquan_lagpat_phaseB_results.md` |
| `_phaseE_to_F1_chain.sh` | Watchdog：Phase E 跑完后自动衔接 Phase F-1 | （chain wrapper，无独立 doc） |
| `_phaseF1_run_yuquan_main.sh` | Phase F-1：10 个 main-cohort subject 用 `legacy_align=True` 重检测 | `docs/archive/yuquan_lagpat/yuquan_detector_drift_phaseD_results.md` |

## 相关产出文档（建议先读）

- `docs/archive/yuquan_lagpat/yuquan_detector_drift_root_cause.plan.md` — 漂移根因 plan
- `docs/archive/yuquan_lagpat/yuquan_detector_drift_phaseD_results.md` — Phase D 结果
- `docs/archive/yuquan_lagpat/yuquan_lagpat_phaseA_results.md` / `..._phaseB_results.md`
- `docs/archive/yuquan_lagpat/yuquan_lagpat_backfill_validation.plan.md`
- `docs/archive/yuquan_lagpat/yuquan_24_same_source_contract_status.md`
- `docs/archive/yuquan_lagpat/dual_track_audit_2026-04-26.md`
- `docs/archive/yuquan_lagpat/wangyiyang_pack_top_n_decision.md`
- `docs/archive/yuquan_lagpat/yuquan_phaseB_gaolan_picked_drift_notes.md`

## 状态

24h same-source 对齐已 lock；当前生产入口是 `scripts/run_yuquan_lagpat_backfill.py`
与 `scripts/run_hfo_detection.py --dataset yuquan`。本目录的 `_phase*.sh` 是
当时手工排程的 shell wrapper，不是可重复的生产脚本。
