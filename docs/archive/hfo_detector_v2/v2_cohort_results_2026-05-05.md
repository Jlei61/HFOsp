# v2 cohort results (in progress)

Phase 3 cohort rebuild — Phase 3.4 not yet completed (CPU-fallback incident; see §3 below).

## Phase 3 status

- Task 3.1 ✅ — parallel script defaults updated (`results/hfo_detector_v2/`, `N_JOBS=1`)
- Task 3.2 ✅ — Epilepsiae GPU default set (`--gpu/--no-gpu` tristate; Yuquan stays CPU)
- Task 3.3 ✅ — 635 single-record smoke run completed (1 record, 97.6s wall, GPU verified via `conda run -n cuda_env`)
- Task 3.4 ⏳ **进行中** — 第一次启动 CPU fallback 已 quarantine + 修复 wrapper（见 §3）。**第二次启动 (2026-05-06 11:23 CST，PID 906605) 已确认 GPU**：cupy/cusignal preflight passed、bash → conda run → python 链路工作、GPU 利用率 sampling 看到 72% spike + 425–519 MiB memory engaged、12 records 完成于 6 min (= 30 s/record)。预计 ~33h cohort 完成。

## §1. 验证已确立的事实

| 事项 | 结果 | 来源 |
|------|------|------|
| 635 smoke detection produces well-formed `*_gpu.npz` | ✅ | Task 3.3 |
| 635 twice-run GPU determinism | `count_match=True`, `max_t_diff=0.0` | Phase 2 Task 2.3 (`results/hfo_detector_v2/validation/layer_a_determinism_635.json`) |
| 635 Layer A on smoke (3-window sample, 600s/3600s) | TLA1 `n=3, ratio_p25=2.108 ≥2.0 ✅`, `margin_p50=0.495` 边缘 fail (≥0.5) | `results/hfo_detector_v2/validation/layer_a_635.json` |
| Layer A path-resolution gap fixed | 635 records 1, skipped 0 | commit `c43c48d` |
| 29-test cohort regression suite | passed | commits `1adad11`, `d8e30d7` |

## §2. 真实运行时间预估（plan 10h 误差 11×）

`/mnt/epilepsia_data` 上 20 个 subject 的实际 record 数：

| subject | records | subject | records | subject | records | subject | records |
|---------|---------|---------|---------|---------|---------|---------|---------|
| 253 | 268 | 916 | 435 | 590 | 254 | 1096 | 165 |
| 548 | 147 | 922 | 114 | 620 | 256 | 1125 | 160 |
| 139 | 173 | 958 | 225 | 635 | 123 | 1146 | 117 |
| 384 | 130 | 583 | 206 | 1073 | 231 | 1150 | 161 |
| 1077 | 189 | 442 | 178 | 818 | 255 | 1084 | 252 |

**Total: 4,039 records**。GPU 单卡（RTX 3090）实测 throughput：

- **Single-record cold start (635 smoke, --smoke flag)**: 97.6 s（包含 cusignal 初始化 + 单 record 完整 detect）
- **Single-record warm rerun (Task 2.3 determinism, full 3600s record)**: 24.9 s
- **Cohort steady-state (subject 253 relaunch 2026-05-06, 12 records in 360 s)**: **30 s / record**

GPU 利用率 sampling at 1 Hz：5% (data load) → 72% spike (envelope+Hilbert kernel) → 0% (between records)。Memory 425–519 MiB（cusignal warm）。

**Wall clock 估计 ≈ 4,039 × 30 / 3600 ≈ 33 h ≈ 1.4 days**（plan 写的 10h 仍乐观，但比上次估算的 112h 大幅修正）。

> Note：原先 100s/record 估计来自 635 smoke 与 Task 2.3 单 record 重检的上界，**包含了 cusignal 初次 import 的开销**。Cohort steady-state 的初始化开销摊到所有 records 上后实际 throughput 是 30 s/record。最大单 subject (916, 435 records) ≈ 3.6 h。

## §3. 2026-05-06 CPU-fallback 事故

### 问题

Task 3.4 第一次启动时，`scripts/run_epilepsiae_detection_parallel.sh` 的 worker 用裸 `python` 调用 detection，但当前 shell PATH 上的 `python` (`/home/honglab/leijiaxin/anaconda3/bin/python`) **没有 cupy / cusignal**——`cuda_env` 才有。结果：

- `src/utils/bqk_utils.py:329` 触发 `RuntimeWarning: GPU requested but CuPy/cusignal not available. Using CPU.`
- 跑了 9 h 57 m 全部走 CPU fallback（`nvidia-smi` 0% GPU 使用率，仅 15 MiB GPU 内存）。
- 实际产物：subject 253 完整跑完 (268 records + `_refineGpu.npz`, 13645 s)，subject 548 部分 (53/147 records)，subject 139 启动后立即被 kill。

### CPU vs GPU 等价性验证

对同一 record `253/25300102_0000` 用 cuda_env GPU 重跑 (24.9 s)，对比已有的 CPU 产物：

- CPU: 29 channels, 4003 events
- GPU: **30 channels**, **4303 events** (+7.5%)
- count drift: **26 / 29 channels 不一致**（差值 -8 ~ +9）
- 即便 top-3 channels（HLC1, HRB1, HRB2），event count 都对不上 → shape 直接不匹配，无法做 timestamp diff

**结论：CPU 路径与 GPU 路径在 Epilepsiae 数据上不是 bit-equivalent。** Phase 1 spec (`v2_specification.md`) §4 的"GPU=CPU=float32=float64 deterministic 等价"基于 635 chunk 0 的 Path A 诊断（427/43 完全一致），但在 253 上失效。这条等价性声明的 generalization 范围比 spec 写的窄；不能用 CPU 产物当 v2 cohort 输入。

### 处置

- CPU 产物 quarantine 到 `results/_diag_quarantine/{253,548,139}_cpu_fallback_2026-05-06/`，**不保留为 v2 cohort 输入**。
- Wrapper 修补（`scripts/run_epilepsiae_detection_parallel.sh`）：
  1. PREFLIGHT 段——脚本启动前 `conda run -n cuda_env python -c "import cupy, cusignal"`，失败立即 `exit 2`。
  2. worker 用 `conda run -n cuda_env --no-capture-output python` 而非裸 `python`，确保子进程命中 cuda_env。
  3. 新增 `CONDA_ENV` 环境变量（默认 `cuda_env`），可显式覆盖。
- 645 smoke 的 GPU 产物保留：`results/hfo_detector_v2/635/63500102_0000_gpu.npz` 是用 `conda run -n cuda_env python ...` 显式启动的，cupy/cusignal 都已 import 到位，确认是 GPU 输出。`--skip-existing` 会跳过它。

### 启动前检查清单（写给后续 agent）

启动 cohort 前必须验证：

```bash
conda run -n cuda_env python -c "import cupy, cusignal; print('OK', cupy.__version__, cusignal.__version__)"
```

返回 `OK 13.6.0 23.08.00` 才能启动。如果换机器 / 换环境，重跑这条 check。Wrapper PREFLIGHT 段会自动 enforce，但人工再确认一次。

启动后 1 分钟内必须验证：

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

GPU 利用率 > 0%、内存使用 > 200 MiB 才算正常 GPU 跑动；都为 0 / 极低值就是 fallback 复发，立刻停。

## §4. 滚动 Layer A 描述（per-subject，**不计 cohort PASS rate**）

只记录单 subject 已完成的 Layer A 数字，不做 cohort PASS/FAIL 判定（按 user directive 留给 Phase 4 全 cohort 跑完后再做）。

### subject 253（2026-05-06 15:24 写入 `validation/layer_a_253.json`）

- 取样：268 records，每 record first/middle/last × 200s 三窗
- channels-with-events 总数：7,646（跨 record × ch）
- 总 events（去 NaN）：379,549；channels-with-events 中位 events/ch = 20

| metric | 中位 | p10 | per-channel pass rate (vs PASS 阈值) | 备注 |
|---|---|---|---|---|
| `dur_in_band_frac` | 1.000 | — | 97.8% (≥0.95) | 主带占比健康 |
| `peak_side_ratio_p25` (去 NaN) | 2.137 | 0.988 | 62.4% (≥2.0) | NaN 154/7646 = 2.0%，是 chunk 边界 edge events 的预期空值 |
| `threshold_margin_p50` | 0.917 | −0.283 | 74.4% (≥0.5) | 中位健康，但 10 分位为负 → 部分通道事件刚好压在 detector 阈值边缘 |

### subject 548（2026-05-07 11:47 写入 `validation/layer_a_548.json`，reboot 后重跑）

- 取样：147 records × 三窗
- channels-with-events 总数：10,164
- 总 events：294,348；中位 events/ch = 8

| metric | 中位 | p10 | per-channel pass rate | 备注 |
|---|---|---|---|---|
| `dur_in_band_frac` | 1.000 | — | 100% (≥0.95) | perfect |
| `peak_side_ratio_p25` (去 NaN) | **1.710** | 0.936 | 40.0% (≥2.0) | **中位低于 PASS 阈值 2.0**；NaN 19/10164 = 0.2% |
| `threshold_margin_p50` | 0.579 | −0.393 | 52.7% (≥0.5) | 中位刚过线 |

**观察**：548 信号质量比 253 低一档（ratio 1.710 vs 2.137，margin 0.579 vs 0.917）。83 ch 中有大量低 SNR 通道，median events/ch 也低（8 vs 20）。属真实 per-subject 差异，不是 bug；**Phase 4 是否将 548 计入 PASS 还需契约决策**（per-channel 中位 vs subject-level 中位 vs events-weighted）。

**观察（非结论）**：
- 中位线全部满足 PASS 阈值（dur≥0.95, ratio≥2.0, margin≥0.5）。
- per-channel pass rate 60–75% 区间，**这是按"每个通道是一个评估单元"统计**；很多 low-event 通道（< 5 events）的 ratio_p25 / margin_p50 估计本身就噪。
- Phase 4 cohort 框架需要**先锁定 PASS rate 的统计单位**：per-subject 中位 vs per-channel pass fraction vs weighted-by-events。这是 Phase 4 入口决策，**不在这里下定**。

### Layer A subject 间变异汇总（n=7，2026-05-09 更新）

| metric | 253 | 548 | 139 | 1077 | 1084 | 442 | 818 |
|---|---|---|---|---|---|---|---|
| ratio_p25 中位 | 2.137 | **1.710** | 2.207 | 2.312 | **1.483** | 2.245 | 2.215 |
| ratio per-ch pass | 62.4% | 40.0% | 91.5% | 98.4% | 34.3% | 99.6% | 98.9% |
| margin_p50 中位 | 0.917 | 0.579 | 1.021 | 1.052 | 0.779 | 0.856 | 0.819 |
| margin per-ch pass | 74.4% | 52.7% | 94.0% | 97.2% | 59.8% | 94.4% | 92.9% |
| events median/ch | 20 | 8 | 5 | 6 | 4 | 8 | 7 |

**subject-level 中位线**（7 subject）：
- ratio_p25 中位的中位 = 2.207；**5/7 ≥ 2.0**（548 + 1084 跌破）
- margin_p50 中位的中位 = 0.856；7/7 ≥ 0.5
- **subject-level 中位 PASS = 5/7 (ratio) + 7/7 (margin)**

**新观察（1084 加入后）**：
- 5 个 subject 中**两个**中位 ratio_p25 < 2.0，比例已达 40%
- 这两个（548, 1084）共同点：events median/ch 低（8, 4）；events 少 → ratio_p25 估计本身有更多 sampling noise
- 但 ch-w-events 总数都很大（10164, 16760）→ 不是数据稀缺，是单 ch event 稀疏
- Phase 4 入口决策更紧迫：events-weighted PASS 可能比 per-channel pass 更能反映真实信号质量

**关键洞察**：subject 间 per-channel pass rate 变异（2.5×）比 subject-level 中位变异（±20%）大得多。Phase 4 PASS rate 用 per-channel 单位会被 channel 多 / events/ch 少的 subject（如 548）拖低；用 subject-level 中位 + events-weighted 可能更稳定。**Phase 4 入口决策**，等更多 subject 跑出来再敲定。

### 待补 subject（2026-05-08 更新）

- ✅ 完成：253, 548, 139, 384, 1077
- ⏳ 进行中：1084（22:12:38 启动，08:25 进度 178/252，87 ch，1024 Hz，ETA ~12:30 CST）
- 排队：442, 818, 916, 922, 958, 583, 590, 620, 635, 1073, 1096, 1125, 1146, 1150（14 个）

### 滚动 cohort 进度表（per-subject 实测）

| subj | head on disk | processed | skipped | skip 原因 | ch | wall (s) | s/record |
|---|---|---|---|---|---|---|---|
| 253 | 268 | 268 | 0 | — | 29 | 8,646 | 32 |
| 548 | 147 | 147 | 0 | — | 83 | 21,084 | 143 |
| 139 | 173 | 130 | 43 | sfreq=256 Hz Nyquist | 41 | 5,229 | 40 |
| 384 | 130 | 65 | 65 | sfreq=256 Hz Nyquist | 93 | 2,822 (恢复后；新增 10 records) | 32 |
| 1077 | 189 | 189 | 0 | — | 121 | 40,694 | 215 |
| 1084 | 252 | ⏳ | — | — | 87 | — | — |

**关键观察**：
- s/record 主要由 **channel 数 × 持续时间** 决定，channel 数最大权重（253 vs 548 = 4.5× s/record gap 对应 ~2.9× channel gap）。
- **139 的 43 records skip 是合理 Nyquist 拒绝**，不是 bug：这些 records 在 `pat_13900*` 子树（不同 recording session），sfreq=256 Hz 不能满足 ripple band (80–200 Hz) Nyquist 下限。Detector hard-gate 在 `scripts/run_hfo_detection.py:386–392` 正确触发。
- Phase 4 Layer A cohort 应该以 **processed records** 为基底，不要把 Nyquist-skipped records 算进 PASS rate。
- v2_specification.md §4 GPU=CPU 等价性声明需要进一步验证（253 的 CPU vs GPU drift 已在 §3 记录）。

## §5. 下一步

- 不再做 Phase 5 / 6 / 7 的科学结论。Layer B / Layer C cohort 数字一律待 Phase 3.4 GPU cohort 完成后再产生。
- Phase 4 Layer A cohort 也只在 GPU cohort 完整后启动（per-subject 描述可滚动跑，cohort PASS rate 在 cohort 完成前无意义）。
- Phase 4 入口需要先锁定：per-subject Layer A 的 PASS 单元（per-channel / per-subject-median / events-weighted）。当前只记录 raw numbers。
- v2_specification.md §4 的"GPU=CPU 等价性"声明需要在 cohort 跑完后修订：appendix 加 "Empirical equivalence verified on 635 chunk 0; on 253 record 0 we observe channel-set + count drift" 的限定。这是真实数据上的 calibration，不是 retrospective 找借口。
