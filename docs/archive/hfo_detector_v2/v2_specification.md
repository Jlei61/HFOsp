# HFO Detector v2 — Algorithmic Specification

> **状态**：Phase 1 deliverable，2026-05-05 起生效。
> **配套文档**：
> - 验收契约：`docs/archive/hfo_detector_v2/v2_validation_contract.md`
> - 执行计划：`docs/archive/hfo_detector_v2/v2_cohort_rebuild_plan_2026-05-05.md`

---

## ⚠️ 适用范围声明（必须先读）

> ⚠️ **三层验收衡量的是 "pipeline 内部自洽"，不是 "生物学有效性"**。Layer A 验证
> detector 自己的 rule 自洽（duration window、side-rejection 阈值、deterministic）；
> Layer B 验证 packing 在群体事件层稳定（rank 在 split-half / odd-even 上一致）；
> Layer C 验证下游 PR-1 / PR-2.5 在 v2 自身上 split-half / odd-even reproducibility。
> **没有任何一层声明 "事件是真实生理 HFO"**——真假 HFO 的判定需要 ground-truth 标注
> 或外部独立测度（reviewer 标注 / 跨模态对照），不在 v2 验收范围内。引用 v2 cohort
> 结论时必须附带这条 disclaimer。

---

## 1. 定义

**HFO detector v2** 是 HFOsp 仓库自 2026-05-05 起生效的 Epilepsiae HFO 检测主路径。
它是一个 **deterministic、CPU/GPU 等价、内部可复现的现代 detector**，不再以
"对齐 21 年 cusignal 输出" 为目标。

- **输入**：单个 Epilepsiae block（`*.data` 原始 int16 + `*.head` 元数据）。
- **输出**：legacy-compatible `*_gpu.npz`，包含 per-channel events、channel names、
  start_time、reference_type、bipolar_pairs。
- **后端**：GPU（cusignal + CuPy）为默认；CPU（scipy）作 CI / 回归测试备份。两者在
  `cuda_env`（CuPy 13.6 / cusignal 23.08 / scipy）上经过 float32 / float64 双精度
  等价性验证（见 §4）。
- **算法层面**：与 commit `6027281` "Path A" 完全一致。v2 仅是 naming + framing 的
  重新定位——retire "legacy reproduction / 100% 复刻 21 年" 措辞。

主路径调用形式：

```python
from src.preprocessing import load_epilepsiae_block
from src.hfo_detector import HFODetector, HFODetectionConfig

pre = load_epilepsiae_block(
    data_path, head_path,
    reference="car",
    notch_freqs=[50.0, 100.0, 150.0, 200.0, 250.0],
    notch_filter_kind="fir_legacy",
)
cfg = HFODetectionConfig(
    band="ripple",
    bandpass=(80.0, 250.0),
    rel_thresh=2.0,
    abs_thresh=2.0,
    side_thresh=2.0,
    min_gap_ms=20.0,
    min_last_ms=50.0,
    max_last_ms=200.0,
    chunk_sec=200.0,
    chunk_overlap_sec=0.0,
    use_gpu=True,
    legacy_align=True,
)
result = HFODetector(cfg).detect(pre)
```

参考实现：`scripts/run_hfo_detection.py::detect_epilepsiae_subject`（lines ~290–340）。

---

## 2. 输入

### 2.1 Raw signal

- 文件对：`<stem>.data`（little-endian int16 interleaved samples）+ `<stem>.head`
  （ASCII key=value）。loader：`src/preprocessing.py::load_epilepsiae_block`。
- 仅保留 intracranial channel（`_epilepsiae_intracranial_indices`），过滤 scalp / aux。
- conversion factor 来自 `.head`，转 float64 µV。

### 2.2 Reference

- **CAR**（默认）：减去 intracranial channel 集合的 mean。
- bipolar 路径在 v2 cohort 中不启用（Epilepsiae 默认 CAR）。

### 2.3 Per-subject drop_chns 已 retire

- `sub_dropChns`（21 年 hand-picked bad-channel 黑名单）**不再** 在 v2 cohort 中使用。
- v2 走 "no manual drop_chns" 默认路径——所有 intracranial channel 全部进入检测。
- 旧黑名单仅作为历史只读参照保留在 `results/_legacy_2021_readonly/`，**不**作为 v2 contract。

---

## 3. 流水线

### 3.1 Notch filter（power-line + harmonics）

实现：`src/preprocessing.py::load_epilepsiae_block` `notch_filter_kind="fir_legacy"`
（line 619）。

```python
from scipy.signal import firwin, fftconvolve
nyq = sfreq / 2.0
for freq in [50.0, 100.0, 150.0, 200.0, 250.0]:
    if freq + 1 >= nyq:  # skip if upper transition reaches Nyquist
        continue
    b = firwin(801, [(freq - 1) / nyq, (freq + 1) / nyq], pass_zero=True)
    data = fftconvolve(data, b[None, :], mode="same", axes=-1)
```

- 滤波器：FIR firwin **801 tap**，bandstop（`pass_zero=True` 表示中心 stopband + DC pass）。
- 频率：`[50, 100, 150, 200, 250]` Hz（5 频率）。
- 卷积：**单 pass forward `fftconvolve`**（非 `filtfilt`，非 zero-phase）——这是
  legacy 21 年 `epilepsiae_detectHFOs.py:79` 的 verbatim 行为，v2 沿用。
- 当 `freq + 1 >= nyq`（如 sfreq=512 时 250 Hz 谐波）跳过。
- 带宽：±1 Hz（fixed 2 Hz bandwidth），所以 Q 随 f 递增（高谐波更窄）。

### 3.2 Multi-band envelope（80–250 Hz ripple band）

实现：`src/utils/bqk_utils.py::BQKDetector` + `_compute_subband_envelope_legacy_cpu`
/ `_compute_envelope_gpu`。

参数：
- `freqband = (80.0, 250.0)`
- `subband_width = 20.0`
- 9 个 sub-band：`(80,100), (100,120), (120,140), (140,160), (160,180),
  (180,200), (200,220), (220,240), (240,250)`（最后一段 10 Hz 宽，因为 250-240=10
  小于 subband_width）

每 sub-band：
1. **Bandpass**：`firwin(201, [low/nyq, high_clipped/nyq], pass_zero=False)`，单 pass
   forward `fftconvolve`（GPU `cusignal.convolution.fftconvolve`，CPU `scipy.signal.fftconvolve`）。
   - `high_clipped = min(high, nyq - 1.0)`。
   - `legacy_align=True` 强制走 FIR-201 forward 路径（见 line 420 / 487）。
2. **Hilbert**：`scipy.signal.hilbert`（CPU）/ `cusignal.hilbert`（GPU），`N=next_fast_len(n_samples)`。
3. **Envelope**：`np.abs(analytic)[..., :n_samples]`。

最终 envelope = 9 个 sub-band envelope **逐元素求和**（`envelope_sum += abs(analytic)`）。

### 3.3 Dual-threshold detection

实现：`src/utils/bqk_utils.py::find_high_enveTimes` 与 `_find_high_enveTimes_gpu`。

每通道独立做：
```text
mask = (env > rel_thresh * ch_median) AND (env > abs_thresh * whole_data_median)
```
- `rel_thresh = 2.0`（× per-channel envelope median）
- `abs_thresh = 2.0`（× whole-data 全通道 envelope median）
- AND 关系，必须同时满足。

### 3.4 Merge + duration filter

mask → `(start, end)` time ranges → 合并 → 时长过滤。

- **Merge gap**：相邻 range 间距 < `min_gap = 20 ms` 合并成一个 event。
- **Duration filter**：保留 `min_last_ms < dur < max_last_ms`，即
  **`50 ms < dur < 200 ms`**（开区间）。注意是严格小于，不含等号；这是 legacy 行为，
  v2 沿用。

### 3.5 Side-rejection denoise

实现：`find_high_enveTimes` lines 189–215（CPU）与 `_find_high_enveTimes_gpu` lines 565–592（GPU）。

对每个 candidate event `[t0, t1]`：
1. `dur = t1 - t0`
2. side window：`[t0 - dur, t0]` 与 `[t1, t1 + dur]`（与事件等长的左右各一个）
3. `pick = env[t0:t1]`，`side = concat(side_pre, side_post)`
4. **Empty side rule (`legacy_align=True`)**：若 `side_pre` 与 `side_post` 同时为空
   （事件贴在 chunk 边界），直接 **拒绝** 该 event。这是 legacy `find_high_enveTimes_cu`
   的 `side_mean=NaN` 路径的 verbatim 行为。
5. 否则 `side_mean = mean(side)`，`pick_mean = mean(pick)`：
   - 若 `side_mean <= 0` → 接受
   - 若 `pick_mean >= side_thresh * side_mean`（即 `pick_mean >= 2 * side_mean`）→ 接受
   - 否则拒绝
- **`side_thresh = 2.0`**。

### 3.6 Chunking

实现：`src/hfo_detector.py::HFODetector.detect`。

- `chunk_sec = 200.0`（200 秒 chunk）
- `chunk_overlap_sec = 0.0`（**无 overlap**；`legacy_align=True` 下强制为 0，line 67）
- chunks 起点：`np.arange(0, total_dur, 200.0)`，back-to-back
- chunk 间不做 boundary remerge（`legacy_align` 下禁用）
- 每个 chunk 独立做 envelope + threshold + merge + duration + side rejection

### 3.7 完整参数表

| 参数 | 值 | 出处 |
|------|-----|-----|
| reference | `car` | `load_epilepsiae_block(reference="car")` |
| notch FIR taps | 801 | `firwin(801, ...)` line 619 |
| notch freqs | `[50, 100, 150, 200, 250]` | `run_hfo_detection.py:340` |
| notch kind | `fir_legacy`（forward fftconvolve） | line 614 |
| bandpass | `(80, 250)` Hz | `run_hfo_detection.py:311` |
| sub-band width | 20 Hz（最末段 10 Hz） | `BQKDetector._build_filter_bank_ranges` |
| bandpass FIR taps | 201 | `firwin(201, ...)` lines 420, 487 |
| `pass_zero` (bandpass) | `False` | line 423 |
| `rel_thresh` | 2.0 | `run_hfo_detection.py:326` |
| `abs_thresh` | 2.0 | `run_hfo_detection.py:327` |
| `side_thresh` | 2.0 | `run_hfo_detection.py:328` |
| `min_gap_ms` | 20 | `run_hfo_detection.py:329` |
| `min_last_ms` | 50 | `run_hfo_detection.py:330` |
| `max_last_ms` | 200 | `run_hfo_detection.py:331` |
| 时长过滤区间 | `50 ms < dur < 200 ms` 开区间 | `bqk_utils.py:556–558` |
| `chunk_sec` | 200 | `run_hfo_detection.py:321` |
| `chunk_overlap_sec` | 0 | `run_hfo_detection.py:322`（`legacy_align=True`）|
| `legacy_align` | `True` | `run_hfo_detection.py:320` |
| `use_gpu` | `True`（默认） | `--use-gpu` CLI flag |

---

## 4. 后端等价性

### 4.1 验证范围

`cuda_env` 上完成的 sentinel 已经验证 **CPU = GPU = float32 = float64**，contract:

- **Filter coefficient diff**：CPU `scipy.firwin(201)` vs GPU `cusignal.firwin(201)`
  逐元素差 ~6e-18（机器 epsilon 级）。
- **Per-channel event count**：subject 635 chunk 0：
  - pre-side（不带 side rejection）：CPU = GPU = 427
  - with side rejection（`side_thresh=2`，`legacy_align=True`）：CPU = GPU = 43
  - float32 / float64 两路也完全一致

### 4.2 v2 contract

**v2 自身 deterministic on modern stack** = 等价性的契约目标。在同一台机器、同一
`cuda_env` 下，多次跑同一 input 必须 produce identical output（per-channel
events_count 完全一致；event timestamps 在 1 sample 以内 jitter）。

不要求与 21 年 niking314 上的 cusignal 22.x 输出 1:1 一致——见 §7。

### 4.3 Sentinel 文件指引

历史 sentinel 调试脚本归档在 `scripts/archive/epilepsiae_lagpat/`：
- `diag_1024hz_smoke_635.py` — subject 635 GPU/CPU pre-side / with-side smoke test
- `diag_1024hz_float32.py` — float32 vs float64 等价性
- `diag_1024hz_chunk_isolated.py` — chunk-level 隔离重放
- `diag_1024hz_path_a_boundaries.py` — chunk 边界事件命中率

这些脚本不再在 v2 主路径运行，仅作 contract 历史证据。

---

## 5. 输出（`*_gpu.npz` schema）

实现：`src/hfo_detector.py::save_detection_as_gpu_npz`（lines 313–366）。

```text
*_gpu.npz keys:
- whole_dets       : object array (n_channels,), each element = ndarray (n_events, 2) float64
                     [event_start_sec, event_end_sec]，绝对秒（block 内）
- chns_names       : object array (n_channels,) of str — primary channel names
                     (CAR 下是单极标签；bipolar 下是 "L1-L2" 风格)
- events_count     : int64 array (n_channels,)
- start_time       : float64 scalar — block start UTC epoch（来自 .head start_ts）
- reference_type   : str — "car" / "bipolar" / "none"
- bipolar_pairs    : object array of (left, right) tuples，CAR / monopolar 时为空
```

**注**：v2 输出的 `*_gpu.npz` 与 21 年 cusignal 输出的 `*_gpu.npz` 在 schema 上兼容
（key 命名 + dtype 完全一致），但 `whole_dets` 数值内容**不要求 1:1 一致**。

---

## 6. v2 vs Path A 的变化

**算法层面零变化**。v2 = Path A（commit `6027281` "fix(epilepsiae_lagpat): Path A —
detector legacy_align + FIR-801 notch (12/12)"）。两者跑同一 input 必须 produce
identical output。

变化的只是 naming + framing：
- 用 "v2 detector" / "v2 cohort" 替代 "Path A" / "legacy reproduction" / "100% 复刻"。
- `results/hfo_detector_v2/` 替代 `results/_legacy_2021_backup/` 作为 forward-going
  cohort artifact root。
- 验收契约由 "对齐 21 年" 改为 "v2 自身三层 reproducibility"（详见 validation contract）。

---

## 7. v2 vs 21 年 cusignal 的变化

21 年 detector 跑在 niking314 PC 上，stack 为 cusignal 22.x + 当年 CuPy + 当年 CUDA。
现代 cusignal 23.08 / cuFFT / CuPy 13.6 在 FFT 路径、kernel scheduling、
`fftconvolve` reduction order 上**不可控不可复刻**——具体差异可达 1–10% per-channel
event count，且无法用任何 deterministic 重放手段消除。

**v2 主动放弃 1:1 比对作为验收基准**。21 年 `*_gpu.npz` 仅以 read-only 形式归档在
`results/_legacy_2021_readonly/`，作为：
- 历史 paper 引用源
- per-subject 漂移诊断 anchor
- **不作为** v2 detector 的 1:1 validation target

任何要求 "v2 输出必须等于 21 年 cusignal 输出" 的论述都已退役。详见
`docs/archive/epilepsiae_lagpat/detector_drift_root_cause_2026-05-03.md`（已加 v2 banner）。

---

## 8. 验收范围声明（重申）

> ⚠️ **三层验收衡量的是 "pipeline 内部自洽"，不是 "生物学有效性"**。Layer A 验证
> detector 自己的 rule 自洽（duration window、side-rejection 阈值、deterministic）；
> Layer B 验证 packing 在群体事件层稳定（rank 在 split-half / odd-even 上一致）；
> Layer C 验证下游 PR-1 / PR-2.5 在 v2 自身上 split-half / odd-even reproducibility。
> **没有任何一层声明 "事件是真实生理 HFO"**——真假 HFO 的判定需要 ground-truth 标注
> 或外部独立测度（reviewer 标注 / 跨模态对照），不在 v2 验收范围内。引用 v2 cohort
> 结论时必须附带这条 disclaimer。

完整阈值与 PASS 条件见 `docs/archive/hfo_detector_v2/v2_validation_contract.md`。

---

## Cross-references

- `docs/archive/hfo_detector_v2/v2_validation_contract.md` — 三层 PASS 阈值预注册
- `docs/archive/hfo_detector_v2/v2_cohort_rebuild_plan_2026-05-05.md` — Phase 0–9 执行流程
- `results/_legacy_2021_readonly/README.md` — 21 年只读资产说明
- `src/preprocessing.py::load_epilepsiae_block` — 输入 + notch
- `src/hfo_detector.py::HFODetector` — 检测主类 + `save_detection_as_gpu_npz`
- `src/utils/bqk_utils.py::BQKDetector` — multi-band envelope + thresholding（CPU/GPU）
- `scripts/run_hfo_detection.py::detect_epilepsiae_subject` — Epilepsiae cohort 入口
