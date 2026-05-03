# Detector 层 5% 残差根因 — 深挖审计（2026-05-03）

> 计划：[`epilepsiae_lagpat_backfill_plan_2026-04-29.md`](./epilepsiae_lagpat_backfill_plan_2026-04-29.md)
> 上一阶段：[`legacy_replication_audit_2026-05-03.md`](./legacy_replication_audit_2026-05-03.md)
> 起因：用户拒绝接受 "detector 95% 与 legacy 一致" 的说法 — 既然代码理应一样，为什么会不一样？

## 答案 TL;DR

**新 pipeline 的 detector 在调用 site 上没用 `legacy_align=True`，也没显式传 5-频率 notch**，所以默认走的是：

| 维度 | Legacy | 新 pipeline 默认 |
|------|--------|------------------|
| Bandpass filter | **FIR firwin(201) forward fftconvolve**（单向）| **Butter-3 sosfiltfilt**（零相位双向）|
| Notch filter | **FIR firwin(801) forward fftconvolve**（每频率一次）| **IIR iirnotch(Q=30) filtfilt**（零相位）|
| Notch 频率 | **50, 100, 150, 200, 250 Hz**（5 个）| **50, 100, 150, 200 Hz**（4 个，缺 250）|
| Chunk size | `segment_time = 200s`（exact）| `chunk_sec = 600s` (GPU) 或 `50s` (CPU) |
| Chunk overlap | 0 (back-to-back) | 2.0s |
| Chunk-edge events | 当 side window 全空 → side_mean = NaN → reject | 当 side window 全空 → keep（除非 legacy_align）|
| 数值精度 | GPU `cusignal` cuFFT（可能 float32）| CPU `scipy.signal` FFT（float64）|

`legacy_align=True` 的 flag 已经实现，能解决前 5 个差异。**但 Epilepsiae 调用 site (`scripts/run_hfo_detection.py:313-326`) 没传它**。

剩下的最后 1 个差异（GPU vs CPU 数值精度）**在不同硬件上不可能 100% 复刻**。

## 实证（subject 253 record 0 vs legacy oracle）

Legacy gpu.npz 真值：4127 events, 29 chns。

| 配置 | 总 events | 比 legacy | 与 legacy 5% 内匹配的 chns |
|------|-----------|-----------|---------------------------|
| **Pre-Δ 默认**（Butter zero-phase + IIR notch 4 freqs）| 3942 | 0.95 (-5.5%) | 12 / 20 |
| **legacy_align=True + IIR notch 5 freqs** | 4258 | 1.03 (+3.2%) | — |
| **完全对齐**：FIR-201 bandpass + FIR-801 notch 5 freqs + chunk_overlap=0 + chunk_sec=200 + reject edge events | **4003** | **0.97 (-3.0%)** | **13 / 20** |

逐通道对照（Top 20 channels，完全对齐配置）：

```
chn          legacy   new(fully-aligned)   ratio
HLC1           1338            1333         1.00  ← perfect
HRB1            559             554         0.99
HRB2            404             401         0.99
HRA5            256             246         0.96
HLB1            209             204         0.98
HRC2            178             177         0.99
HRA4            170             165         0.97
HLB2            149             146         0.98
HLA2            136             133         0.98
HLA1            119             112         0.94
HLA3            114             111         0.97
HRC1             84              80         0.95
HLC2             46              44         0.96
HRB3             41              35         0.85  ← weak channel, harder to match
HRA2             37              37         1.00
HRA3             35              27         0.77  ← weak
HRB5             33              23         0.70  ← weakest
HLA4             31              27         0.87
HRA1             24              21         0.88
HLC5             21              17         0.81
```

**强通道（>100 events）逐个 0.94-1.00**，弱通道（<50 events）由于事件少、噪声放大效应明显，残差较大但仍单向（new < legacy）。

## 5 个 detector 层 deltas + 1 个未完全归因残差

### Δ5 — Notch filter 缺 250 Hz

`scripts/run_hfo_detection.py:355-359` 的 `load_epilepsiae_block(...)` 没传 `notch_freqs`，落到 `src/preprocessing.py:483` 的 default `[50.0, 100.0, 150.0, 200.0]`。

**Legacy** (`epilepsiae_detectHFOs.py:230`): `np.arange(50, 251, 50)` = `[50, 100, 150, 200, 250]`，**5 个频率**。

250 Hz 是 ripple band 的右边界，正好需要 notch 掉它的 50Hz 第 5 谐波。新 pipeline 缺这一项 → 250 Hz 附近残留 mains 谐波 → 在 80-250 Hz envelope 里多出虚假能量 → 影响阈值判定。

### Δ6 — Bandpass filter 实现差异

**Legacy** (`epilepsiae_detectHFOs.py:73-76`):
```python
def band_filt_cu(data, fs, freqs):
    b = cusignal.firwin(201, [freqs[0]/(fs/2), freqs[1]/(fs/2)], pass_zero=False)
    data = cusignal.convolution.fftconvolve(data, b[None, :], mode='same')
    return data
```
**FIR firwin(201) 单向 fftconvolve**（causal，有 ~100 sample group delay）。

**新 pipeline 默认** (`src/utils/bqk_utils.py:103-106`):
```python
def band_filt(data, fs, freqband):
    nyq = fs / 2
    b, a = butter(3, [freqband[0]/nyq, freqband[1]/nyq], btype='bandpass')
    return filtfilt(b, a, data, axis=-1)
```
**Butter-3 sosfiltfilt 零相位**（双向，effective order 6，更陡的过渡带）。

`HFODetectionConfig.legacy_align=True` 时切换到 FIR（`_compute_subband_envelope_legacy_cpu`，line 459）。但 Epilepsiae 调用站点 (`run_hfo_detection.py:313-326`) **没传 `legacy_align=True`**。

差异：Butter zero-phase 双向过滤更杀边带（ripple band 边界附近的弱信号），FIR 单向过滤更松。新 pipeline 默认偏严 → 少检 ~5% events。

### Δ7 — Notch filter 实现差异

**Legacy** (`epilepsiae_detectHFOs.py:79-83`):
```python
def notch_filt_cu(data, fs, freqs):
    for f in freqs:
        b = cusignal.firwin(801, [(f-1)/(fs/2), (f+1)/(fs/2)], pass_zero=True)
        data = cusignal.convolution.fftconvolve(data, b[None, :], mode='same')
    return data
```
**FIR firwin(801) 带阻 (pass_zero=True + 两个 cutoffs → notch)，每频率单向 fftconvolve 一次**。带宽固定 2 Hz，所有频率相同 → 高频谐波处的 Q 值 ≈ 50/100/150/200/250 / 2 = 25, 50, 75, 100, 125（高频 notch 更窄）。

**新 pipeline** (`src/preprocessing.py:589-591`):
```python
b, a = iirnotch(freq, 30, sfreq)
filtered = filtfilt(b, a, data)
```
**IIR iirnotch (Q=30 固定)，filtfilt 零相位双向**。Q=30 在所有频率上 → 高频 notch 比 legacy 宽 (1.7-4.2x)。

差异：新 pipeline 在 100-250 Hz 谐波附近移除更多信号能量 → ripple band 内总能量下降 → 全局 median 下降 → abs_thresh × global_median 更低 → 弱事件更易触发，但同时被滤掉的 ripple 真信号也更多。最终 net 效应是少检弱事件。

### Δ8 — Chunk-edge events 处理差异

**Legacy** (`epilepsiae_detectHFOs.py:159-163`):
```python
side_pre = tmp_enve[int((tr[0]-(tr[1]-tr[0]))*fs):int(tr[0]*fs)]
# ↑ 当 tr[0]-(tr[1]-tr[0]) < 0 时 → side_pre 是空 array
side_after = tmp_enve[int(tr[1]*fs):int((tr[1]+(tr[1]-tr[0]))*fs)]
side_mean = cp.mean(cp.concatenate([side_pre, side_after]))  # NaN if both empty
if pick_mean >= side_thresh*side_mean:  # NaN comparison → False → REJECTED
    after_denoise_list.append(tmp_tr)
```

**新 pipeline 默认** (`src/utils/bqk_utils.py:202-207`):
```python
if len(side_pre) == 0 and len(side_post) == 0:
    if legacy_align:
        continue  # rejected
    after_denoise_list.append(tmp_tr.tolist())  # KEPT
```

`legacy_align=True` 时已经修正。默认走 KEPT 分支 → 在 chunk 边界（每 200s 处）多检几个事件。

### Δ9 — Chunk overlap 差异

Legacy: `segment_time=200, tranges=back-to-back`（无 overlap）。
新 pipeline 默认 (`run_hfo_detection.py:323`): `chunk_overlap_sec=2.0`。

`legacy_align=True` 强制 chunk_overlap=0 (`hfo_detector.py:259`)。但默认未传 → 仍是 2.0。

差异：overlap 区域内的事件可能被检测两次（再 merge 时去重，但 merge 边界判定不同）。

### Δ10 — Chunk size 差异

Legacy: `segment_time=200` (固定 200s)。
新 pipeline (`run_hfo_detection.py:312`): `gpu_chunk = 600.0 if use_gpu else 50.0`。

差异：50s chunk vs 200s chunk → 更多 chunk 边界 → 更多 chunk-edge 事件需要处理。Δ8/Δ9 影响放大。

### 残余 ~3% 漂移：remaining residual not fully attributed

经过 Δ5+Δ6+Δ7+Δ8+Δ9+Δ10 完全修正后，subject 253 record 0 仍有 ~3% 残差：4003 vs 4127 = 0.97。这部分残差**当前未完全归因**——下面是几个可能的来源，但**没有逐一量化排除**：

可能贡献项（按怀疑度高低，未实测分解）：
1. **FFT 实现 / 数值精度**：Legacy 用 `cusignal` (GPU cuFFT)，新 pipeline 在 CPU 上用 `scipy.signal`。两条路径的 hilbert 长度选择不同（cusignal 默认 N=signal_length；scipy 我们用 `fftpack.next_fast_len` pad 到 fast-len 再裁剪），可能导致不同的 ringing 模式与边界效应。
2. **`np.median` 数值实现**：在 30M 样本上 partition 算法对实现敏感；global_median 的微小差异会被 `abs_thresh*median` 放大成阈值差异。
3. **可能仍存在的算法实现细节差异**：本次只对照了 detector 主流程；preprocessing 早期阶段（intracranial filter 顺序、CAR 计算 dtype、segment 拼接边界）若有未被发现的微差也会进入。
4. **GPU vs CPU 指令顺序累积误差**：浮点累加在 SIMD/CUDA 不同 reduce 模式下结果可能不同，量级 ~1e-6 但经过 hilbert + abs 可能被放大。

**为什么会影响事件计数**：Detector 是阈值判定 (`tmp_enve > rel_thresh*tmp_median AND tmp_enve > abs_thresh*whole_data_median`)。阈值边缘的事件（envelope ≈ threshold）只要浮点差异够大，就能 cross / not cross 阈值。强事件（envelope >> threshold）不受影响——这与观察一致：HLC1（1338 events）几乎完美匹配，HRB5（33 events）差距大。

**结论谨慎口径**：~3% 漂移**目前未量化分解**，不能断言 "100% 来自 GPU/CPU"。是否能进一步收窄需要后续单独诊断（例如：固定 GPU 实现做对照、profile 阈值边缘事件分布、对比 hilbert 中间结果）。本次 Path A 不做该诊断。

## 结论

**用户的 "为什么不一样" 答案分两层**：

1. **不是同样的代码**：默认配置下，新 pipeline 的 bandpass / notch / chunk 处理与 legacy 完全不同。这部分是 5 个 deltas (Δ5-Δ10) 的合计影响，**可以通过修改 Epilepsiae 调用 site 完全消除**（`legacy_align=True` + 5-freq notch + 200s chunk + 0 overlap）。

2. **即便算法对齐后仍有 ~3% 残差**：未完全归因（remaining residual not fully attributed）。可疑来源包括 FFT 实现 / hilbert N 选择 / median 数值精度 / SIMD 累加顺序，但本次没逐一量化分解。需要后续单独诊断。

## 下一步选择

### Path A — 完全对齐到 legacy（推荐）

工程改动：
1. `src/preprocessing.py`: 加 `notch_filter_kind="iir"|"fir_legacy"` 参数；`fir_legacy` 模式用 firwin(801) forward fftconvolve
2. `scripts/run_hfo_detection.py:313-326` Epilepsiae 调用 site：
   - `notch_freqs=[50, 100, 150, 200, 250]`
   - `notch_filter_kind="fir_legacy"`
   - `legacy_align=True` 传给 HFODetectionConfig
   - `chunk_sec=200.0`
   - `chunk_overlap_sec=0.0`
3. TDD 测试：FIR-801 notch 实现验证、调用 site verifyies 5 个参数

之后整 cohort 重新 detect → 重新 synRefine → 重新 lagPat，重跑 Stage C audit。

预期效果：detector 层残差 ~3%（未完全归因），refine/pack 层 0% 残差（已是 verbatim port）。Stage C bucket **expected to improve; must verify by Stage C rerun**——不能在重跑前断言 enter_full / 全 stable。

### Path B — 接受 5% detector 漂移

科学上可辩护：5% 是同一算法的不同实现间的合理变异，不是 bug。Topic 1 是 sensitivity audit，不是 strict parity，所以这点漂移可接受。

工程上不动 detector，只跑 refine/pack（已修复的 4 个 deltas）。但 detector 层的 5% 残差会被 refine/pack 放大到 chnNames 级差异（如 253 的左右翻转）。

### Path C — 折中：只修 Δ5（5-freq notch），不动 filter 实现

最小改动 — 只补缺的 250 Hz notch。filter 实现保持 zero-phase（更现代的工程选择）。

预期：filter implementation 差异仍在，~3-5% detector 漂移持续，但至少 250 Hz 谐波被 notch 干净。

## 选定路径

**Path A** — 用户态度明确："我们需要复刻老代码"。

Path A 的 risk：
- 覆盖现有 Stage D smoke artifacts（cohort rebuild 必然如此）
- 重跑 Stage C audit **期望** bucket 从 12/4/4 改善向更多 stable，但**必须以 rerun 结果为准**，不能预先断言
- 如果 rerun 后 253 仍在 large_drift（jaccard 仍低），那就是 detector 层未归因残差被 refine/pack 放大的真实信号差异——按 Topic 1 sensitivity audit 报告即可（不掩盖、不优化）

## 文件清单（本次 commit 范围）

新增：
- `docs/archive/epilepsiae_lagpat/detector_drift_root_cause_2026-05-03.md`（本文）
- `tests/test_load_epilepsiae_block_notch_kinds.py`（5 tests，FIR notch 数值正确性 + 默认 IIR 不变 + 越 Nyquist 跳过 + 错误参数）
- `tests/test_run_hfo_detection_epilepsiae_legacy_align.py`（7 tests，调用 site 5 个参数 + Yuquan 守护）

修改：
- `src/preprocessing.py`：`load_epilepsiae_block` 加 `notch_filter_kind="iir"|"fir_legacy"` 参数，`fir_legacy` 模式实现 firwin(801) forward fftconvolve（verbatim 复刻 legacy）
- `scripts/run_hfo_detection.py`：Epilepsiae 调用 site 默认 `legacy_align=True` + `notch_freqs=[50,100,150,200,250]` + `notch_filter_kind="fir_legacy"` + `chunk_sec=200.0` + `chunk_overlap_sec=0.0`
