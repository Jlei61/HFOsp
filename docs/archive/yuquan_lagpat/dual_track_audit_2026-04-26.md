# Yuquan 双轨 audit + Phase γ 根因 ablation 计划

> 归档日期：2026-04-26
> 主题：Topic 3 准备工作 — Yuquan 21-subject same-source `lagPat` / `packedTimes` 数值对齐合同
> 前序文档：
> - `yuquan_24_same_source_contract_status.md`
> - `yuquan_detector_drift_phaseD_results.md`
> - `yuquan_detector_drift_root_cause.plan.md`
> - `yuquan_lagpat_phaseA_results.md`
> - `yuquan_lagpat_phaseB_results.md`
> - `yuquan_phaseB_gaolan_picked_drift_notes.md`
> - `wangyiyang_pack_top_n_decision.md`

## 0. TL;DR

- **结构合同 PASS** 已稳（21/21 ok 的 same-source batch 已经完成、`cohort_pass=True` 已经记录）。
- **Track A** 跑完：21 个 subject 里 13 个 detector-comparable subject 的 detector 残差 consistent with threshold-sensitive drift —— **alias-collapsed 通道包含率近 1.000、|global_onset_shift|=0.00 ms、max unmatched ≤14.8%**。但 **tight event-level (1-sample / 1.25 ms) 等价性并未对 13 个全部成立**：只有 `gaolan`/`wangyiyang`/`sunyuanxin`/`xuxinyi` 4 个 subject `tight_frac ≥ 0.85`，其余 9 个落在 0.51–0.84，匹配主要落在 medium/loose 桶。1 个 (zhangjiaqi) 因 legacy=monopolar/new=bipolar 显式排除（+7.5 ms 全段平移、61% 不匹配，scheme divergence by design）。1 个 (pengzihang) degenerate evidence，6 个无 legacy gpu_npz。Cohort 标签 `coarse_logic_divergence` 由 Layer 1 gate (containment ≥ 0.95 AND |shift| ≤ 1.25 ms AND `tight_min ≥ 0.85`) 触发——zhangjiaqi 的 7.5 ms 平移和大多数 subject 的 tight_frac 不到 0.85 共同把它拉爆，不是 detector 行为普遍错误，**也不是 13 个 subject 都通过严格 detector-equivalence**。
- **Track B preflight** 跑完（gaolan 12 records）：pack-stage（chnNames + packedTimes + eventsBool）**12/12 完全 bit-for-bit**；lagPatRaw centroid 漂移 maxabs 1.5–67 ms（Phase A baseline 3/12，strict 1 ns 0/12），漂移幅度与 record 内事件数强正相关。
- **根因再读 + γ ablation 已落实**（2026-04-26）：γ.0 独立确认 legacy pack 阶段 active 路径就是 scipy CPU IIR + filtfilt（GPU 分支全部注释），phaseD 的 "GPU vs CPU FP" 跨阶段套用证伪。γ.1 在 gaolan FA0013L8 / FA0013KP 上做 6 层 ablation：L1–L4 全部 0；Branch L（字面 legacy reimpl）与 R0 (`.legacy_backup`) 在当前 env 上 bit-for-bit ε 一致；Branch P（production）在 L5 stitched 这一层比 Branch L 少 1 sample（FP epsilon 边界 bug，`(end - start) * sfreq` 在 float64 下偶尔 round 到 floor-1）。
- **修复 1**：`src/group_event_analysis.py::build_stitched_window_signal` 改成字面 legacy 路径 `batch_t = start_sec + np.arange(n) / sfreq` + boolean union mask。无算法 / 无 backend / 无参数变化，纯 FP 路径切到 legacy 同款。
- **修复 2**（cohort run 后发现）：`scripts/run_yuquan_lagpat_backfill.py::alias_bipolar_to_left_with_arbitration` 加 auto-detect schema —— 单电极输入（2021-era legacy refine, names 没 `-`）下跳过 outermost-shaft drop（legacy refine 已经 alias-collapsed + drop 过了，再 drop 一次会误删 picked channel）；bipolar pair 输入（new pipeline detector 输出，names 含 `-`）维持原行为。
- **最终 Track B 14-subject cohort 结论**（2026-04-26）：cohort_label = `fail`，但 fail 的根因不是代码 bug 是 data archaeology：
  - **6/14 subjects strict 12/12 PASS**（68/74 pass-eligible records ε-level pack+lagPat 数值复刻 `.legacy_backup`）：`gaolan`、`dongyiming`、`wangyiyang`（option-A 22 cap, 9/9）、`sunyuanxin`、`xuxinyi`、`zhangjinhan`
  - **8/14 subjects FAIL** 全部归因 refine drift，无代码层故障：
    - 7 subjects (`chenziyang`、`hanyuxuan`、`huanghanwen`、`huangwanling`、`litengsheng`、`chengshuai`、`liyouran`) 的 `<raw>/_refineGpu.npz` 在 2026-04-09/10 被 new pipeline 重新 detect/refine 写过，2021 legacy 用的 refine 数据不可复原
    - 1 subject (`zhangjiaqi`) 即使 refine 文件 mtime=2021，events_count 与 legacy 时代有 ε 级漂移，导致 H 通道边缘 (`H1`, count=14559, thr=14324) 在两套 refine 之间 picked 集合差 1 通道
- **结论**：γ 完成的工作有两部分：(1) **修复了 production 在 stitched/alias 阶段的两个独立代码 bug**，证据是同一份 legacy refine + legacy gpu_npz 在修复后能 bit-for-bit 复刻 `.legacy_backup`（6 subjects, 68 records, ε 级 maxabs）；(2) **暴露了 7 subject 的 refine_npz 已经在 2026-04 被覆盖**，需要单独决策（接受、找 2021 备份、或缩小 cohort 范围）。
- **当前合同口径建议**：把 Track B "数值复刻能力" 主张限定到 6 个 refine-stable subjects（68 records ε pass）；其余 8 subjects 显式标 `refine_drift / uncomparable`，不进入数值主张。代码侧两个 bug 已修，247/247 in-scope tests 全绿。

## 1. 背景

2026-04-23 完成 Yuquan 21-subject same-source 结构合同闭环（详见 `yuquan_24_same_source_contract_status.md`）：
- 21/21 subject 跑通新代码 pack+lagPat
- 260 blocks: 255 written / 5 skipped (option-A `wangyiyang.pack_top_n=22` + 个别 legacy 缺片) / 0 error
- `start_time_validation` / `legacy_block_presence_diff.regressions` / `n_alias_collisions_in_picked` 全部 zero
- 文档明确写了"结构合同闭环；数值等价是后续独立步骤"

User 在此基础上要求扩展为"复现 legacy refine + packedTimes + lagPat 的能力，确保 pack+lag 阶段与老代码数值一致（pack 隔离）"。这就是 Track B 的来源。

## 2. Track A — Detector 事件级 attribution

### 2.1 用途

phaseD 文档已经把 detector 端 ±10–20% events_count drift 接受成 documented divergence（commit `50010a6`），但只到 aggregate 一级。Track A 的工作：在 event-level 把这个 drift **真正分类**：

- (a) **物理 FP 漂移 + 阈值边界事件进出**：通道映射对齐、整体时间无系统平移、每个 record 内大部分事件 1 sample 内能 1:1 匹配，未匹配事件的余量与边界 / 近邻 envelope 越阈一致。可接受。
- (b) **粗粒度逻辑偏移**：通道映射错、整体时间平移 ≥ 1 sample、事件被系统性地"挪到别处"。actionable，要回去查根因。

### 2.2 Cohort 覆盖（精确口径）

| bucket | n | subjects |
|---|---|---|
| **detector-comparable** (≥1 legacy gpu_npz, ≥80% record coverage) | **13** | gaolan, dongyiming, wangyiyang, chenziyang, hanyuxuan, huanghanwen, huangwanling, litengsheng, sunyuanxin, xuxinyi, zhangjinhan, chengshuai, liyouran |
| **scheme-divergent (excluded from detector-equivalence claim)** | **1** | zhangjiaqi |
| **degenerate_evidence** | **1** | pengzihang (1/12 records 有 legacy gpu) |
| **no_legacy_gpu** | **6** | songzishuo, zhangbichen, zhangkexuan, zhaochenxi, zhaojinrui, zhourongxuan |
| **合计** | **21** | ✓ |

zhangjiaqi 单独列出来，原因见 §2.4。

### 2.3 容差与匹配语义

- Yuquan resample = **800 Hz**（`config/subject_params.json::yuquan._defaults.resample_sfreq`），1 sample = 1.25 ms。
- 三档 onset tolerance：`tight = 1.25 ms` / `medium = 6.25 ms` / `loose = 25 ms`。
- 1:1 onset 匹配（greedy on candidate-cost-ascending），不允许双 claim；近重复事件（< 1 sample）显式上报；边界事件（距 record 首末 ±200 ms）打 `boundary_like`。
- 所有 tolerance 在报告里显式以毫秒+样本数双口径出（避免 "1 sample" 含义模糊）。
- **不**对事件总数做 tolerance：阈值边界事件进出由分布形态判断，不靠数量阈值。

实现：`scripts/audit_yuquan_detector_event_match.py`（matcher 单元测试 `tests/test_yuquan_detector_event_match.py`，10/10 pass）。

### 2.4 13 个 detector-comparable subject 的发现

| subject | n_rec | legacy_in_new | tight_frac | global_shift_ms | unmatched_max | residual |
|---|---:|---:|---:|---:|---:|---|
| gaolan | 12 | 1.000 | 0.928 | +0.00 | 0.016 | threshold_sensitive |
| dongyiming | 12 | 1.000 | 0.839 | +0.00 | 0.035 | threshold_sensitive |
| wangyiyang | 12 | 1.000 | 0.879 | +0.00 | 0.033 | threshold_sensitive |
| chenziyang | 12 | 0.980 | 0.561 | +0.00 | 0.098 | threshold_sensitive |
| hanyuxuan | 13 | 1.000 | 0.556 | +0.00 | 0.148 | threshold_sensitive |
| huanghanwen | 12 | 0.975 | 0.526 | +0.00 | 0.139 | threshold_sensitive |
| huangwanling | 12 | 0.985 | 0.598 | +0.00 | 0.089 | threshold_sensitive |
| litengsheng | 16 | 0.984 | 0.513 | +0.00 | 0.121 | threshold_sensitive |
| sunyuanxin | 12 | 1.000 | 0.920 | +0.00 | 0.017 | threshold_sensitive |
| xuxinyi | 13 | 1.000 | 0.893 | +0.00 | 0.020 | threshold_sensitive |
| zhangjinhan | 13 | 1.000 | 0.676 | +0.00 | 0.039 | threshold_sensitive |
| chengshuai | 12 | 1.000 | 0.681 | +0.00 | 0.067 | threshold_sensitive |
| liyouran | 12 | 0.977 | 0.632 | +0.00 | 0.103 | threshold_sensitive |

13 个 subject 全部满足下列 **粗粒度** 条件：
- 通道包含率（legacy alias ⊆ new alias）≥ 0.97
- |global_onset_shift| ≤ 1 sample (1.25 ms)
- max unmatched fraction ≤ 14.8% (≤ phaseD 文档的 ±20% 余量)
- Layer 2 residual = `threshold_sensitive_drift_only`（不匹配残差不超 30%，能用边界 / 阈值边缘进出解释）

但 **tight event-level (1-sample / 1.25 ms) 等价性** 并未对 13 个全部成立。脚本 Layer 1 要求 `tight_min ≥ 0.85`：
- 满足的有 4 个：`gaolan` 0.928 / `wangyiyang` 0.879 / `sunyuanxin` 0.920 / `xuxinyi` 0.893。
- 其余 9 个落在 0.51–0.84：`dongyiming` 0.839 / `chenziyang` 0.561 / `hanyuxuan` 0.556 / `huanghanwen` 0.526 / `huangwanling` 0.598 / `litengsheng` 0.513 / `zhangjinhan` 0.676 / `chengshuai` 0.681 / `liyouran` 0.632。这些 subject 的事件大多落在 medium / loose 桶（≤ 6.25 / ≤ 25 ms），与"新旧 envelope/threshold 之间 ε 量级 FP 漂移导致 onset 滑动 1–几 sample"一致，但**不是**逐事件一一对应的 bit-for-bit reproduction。

→ 精确的 Track A 主张是：**13 个 detector-comparable subject 表现为"无 global coarse shift、残差与 threshold-sensitive drift 一致"，但严格的 tight event-level 等价性并未对全部 13 个成立**。

→ Cohort 标签 `coarse_logic_divergence` 是 Layer 1 三元 gate（containment / shift / tight_min）任一不满足都会触发：zhangjiaqi 的 7.5 ms 平移把 max |shift| 拉爆，加上 9/13 的 `tight_min < 0.85`，两个条件共同失败。这不是 detector 行为普遍错误，**也不是 13 subject 都通过严格 detector-equivalence**。

### 2.5 zhangjiaqi 为什么单独排除

| 字段 | 值 |
|---|---|
| legacy `chns_names` | `['A1', 'A2', 'A3', ...]`（**单极/monopolar**, 116 通道） |
| new `chns_names` | `['A1-A2', 'A2-A3', 'A3-A4', ...]`（**双极/bipolar**, 128 通道） |
| 共享 alias 集（左触点折叠后） | `{'A1', 'A2', ...}` 数学上重合 |
| `tight_frac` | 0.002（0.2%！） |
| `global_onset_shift` | **+7.50 ms = 6 samples @ 800 Hz** |
| `unmatched_max` | 0.615（61% 事件未匹配） |

aliases collapse 把 monopolar `'A1'` 和 bipolar `'A1-A2'` 都映射到左触点 `'A1'` —— **集合数学上对齐**，但**信号物理上不同**：monopolar 是 `'A1' vs ref`，bipolar 是 `'A1' vs 'A2'`。同名字、不同信号。HFO envelope 在两个信号上落在不同时间点 → 整段 +7.5 ms shift。

这是**老代码处理 zhangjiaqi 时本来就用了不同 reference**（不是新代码 bug）。新代码统一全 cohort 走 bipolar 是设计上的修正。

**结论**：zhangjiaqi 不进入 detector-equivalence 主张。`yuquan_24_same_source_contract_status.md` 的 cohort 表格里 zhangjiaqi 之前误归"Backfill-only (legacy skipped)"，本次 audit 修正：zhangjiaqi 实际上有完整 13 个 legacy `_gpu.npz` + legacy `_refineGpu.npz` + 13 个 backup lagPat，应归 "Main cohort"，但需附 footnote 说明 reference scheme 与其他 main cohort subject 不同。

### 2.6 Cohort verdict 标签的真实含义

`coarse_logic_divergence` —— 这个 Cohort verdict 标签由 Layer 1 三元 gate 触发：`containment_median ≥ 0.95` AND `max |global_shift| ≤ 1.25 ms` AND `tight_min ≥ 0.85`。两条件失败叠加：
- zhangjiaqi 的 +7.5 ms global shift 单独把 max |shift| 拉爆。
- 9/13 detector-comparable subject 的 `tight_frac` 落在 0.51–0.84（< 0.85），把 `tight_min` 拉到 0.51 左右，单独这一项也已经失败。

即使**忽略** zhangjiaqi 单独看 13 个 detector-comparable subject，Layer 1 也不会通过——因为 tight_min 仍然 < 0.85。所以 cohort verdict 不能简单写成"被 zhangjiaqi 一项拉低"或"13/13 都通过 detector-equivalence"。**两种说法都不准确**。

正确的口径是：

1. **13 个 detector-comparable subject 的粗粒度行为一致**：通道映射、整体时间无系统平移、不匹配率在 phaseD 接受的 ±20% 内。
2. **严格 tight event-level (1-sample) 等价性**只对其中 4 个 subject 成立 (`gaolan`、`wangyiyang`、`sunyuanxin`、`xuxinyi`)。其余 9 个落在 medium / loose 桶，与 phaseD 的 "envelope FP 漂移 + threshold 边缘进出" 一致。
3. **zhangjiaqi 单独排除**（reference scheme divergence by design）。
4. cohort verdict 标签 `coarse_logic_divergence` 是上述 1+2+3 的脚本化总和，不是"detector 阶段普遍失败"。

### 2.7 输出

- `results/lagpat_backfill/_audit/detector_attribution/cohort_detector_attribution.{json,md}`
- `results/lagpat_backfill/_audit/detector_attribution/per_subject/<subject>.json`

## 3. Track B — Legacy-refine replay 数值对齐

### 3.1 用途

把 detector + refine 阶段的 drift 都"冻结"住（直接用 legacy `_refineGpu.npz` + legacy `<raw>/<stem>_gpu.npz` 当输入），让新代码的 pack + lagPat 在和 `<raw>/.legacy_backup/<stem>_lagPat.npz` 比较时只承担 pack+lag 自身的 numerical drift。

### 3.2 实现

- **Driver 注入**：refactor `run_subject` 接受 `legacy_refine_root` / `legacy_gpu_root` / `out_dir` / `backup_dir` / `same_source_assertion` 五个参数（保持 same-source CLI 默认行为不变）；helper `_resolve_run_subject_io` 集中路径解析；`summary.json` / `manifest.json` 多 `legacy_refine_root` / `legacy_gpu_root` / `same_source_assertion` / `refine_npz_used` / `gpu_npz_used` 等 provenance 字段。
- **Replay driver**：`scripts/run_yuquan_legacy_refine_replay.py`，CLI `--legacy-refine-root` / `--legacy-gpu-root` / `--out-root` / `--only-subject` / `--all`。`out_root == DATA_ROOT` 防呆；replay 永不写 `<raw>` 也永不创建 `.legacy_backup`。
- **Comparator**：`scripts/audit_yuquan_legacy_refine_replay.py`，**双轴对齐**——行按 chnNames（exact set match + permute），列按 packedTimes 的 1:1 nearest-onset (tol=`pack_win_sec/2 = 150 ms`)；只在两轴对齐的格子里做 array diff。Provenance gate 拒判任何 `gpu_npz_used` / `refine_npz_used` 落在 `DETECT_ROOT` 下的 record。
- **三层 verdict**：strict（plan 严口径，maxabs ≤ 1 ns + rank exact + pack-stage exact）、phaseA_baseline（复用 Phase A 历史阈值 median ≤ 5 ms / p95 ≤ 20 ms / RMSE ≤ 10 ms + rank full-event ≥ 0.95）、pack_stage_only（chnNames + packedTimes + eventsBool exact，不管 centroid drift）。

测试：
- `tests/test_run_subject_path_overrides.py`（6 tests）
- `tests/test_yuquan_detector_event_match.py`（10 tests）
- `tests/test_yuquan_legacy_refine_replay_audit.py`（10 tests）
- 加上原有 25 个 contract tests 共 31 in-scope tests，全部 pass。

### 3.3 gaolan preflight 结果（12 records）

| 层级 | 通过 | 标准 |
|---|---:|---|
| **pack-stage exact** | **12/12** ✓ | chnNames + packedTimes + eventsBool 全部 bit-for-bit |
| Phase A baseline | 3/12 | lagPatRaw {median ≤ 5 ms, p95 ≤ 20 ms, RMSE ≤ 10 ms} + rank full_match ≥ 0.95 |
| strict (1 ns) | **0/12** | — |

**lagPatRaw 漂移与事件数强正相关**：

| n_ev range | record | maxabs (ms) | median (ms) | rank full_match |
|---|---|---:|---:|---:|
| 17–70 | KU/KV/KX/L7/L8 | 1.5–15.8 | 0–0 | 88–100% |
| 38 | L8 | **1.5** | 0 | 100% |
| 49 | L7 | 4.3 | 0 | 96% |
| 70 | KU | 8.1 | 0 | 99% |
| 550–1682 | KP/KQ/KR/KS/KT/KW/L1 | **37.0–66.8** | 1.2–4.9 | 56–64% |

→ 漂移不是 record-级随机噪声，是与 stft 帧数（≈ 事件数）累积成比例的某种 numerical drift。

**已落实的输出**（preflight）：
- `results/lagpat_backfill_legacy_refine_replay/gaolan/{FA0013*_lagPat.npz, FA0013*_packedTimes.npy, summary.json, manifest.json}`
- `/tmp/track_b_preflight/cohort_replay_audit.{json,md}` + `per_subject/gaolan.json`

## 4. 根因再读 —— legacy 生产脚本是 CPU IIR

### 4.1 关键 grep（user 提供，需 ablation 阶段独立确认）

`ReplayIED/inter_events/yuquan_24h_perPatientAnalysis_dropRef/p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter.py:76–82`（`plot_perSeg_specCenter`）：

```python
batch_data = scipy.signal.resample_poly(batch_data, 2, int(round(2 * fs / resample_to)), axis=-1)
batch_data = notch_filt(batch_data, resample_to, np.arange(50, 251, 50))
batch_high = band_filt(batch_data, resample_to, highpass_freqband)
```

`highEvents_yuquan0910_utils.py:47–68`：

```python
def notch_filt(data, fs, freqs):
    Q = 30
    for f in freqs:
        b, a = signal.iirnotch(f / (fs / 2), Q)
        tmp_data = filtfilt(b, a, tmp_data, axis=-1)
    return tmp_data

def band_filt(data, fs, freqband):
    b, a = butter(3, [freqband[0]/(fs/2), freqband[1]/(fs/2)], btype='bandpass')
    return filtfilt(b, a, data, axis=-1)
```

→ legacy pack 阶段的 **active 路径** 是 **scipy CPU `iirnotch + filtfilt`** + **`butter(3) + filtfilt`** —— 与当前 `src/utils/bqk_utils.py:91–106` 的实现 **算法和 backend 完全一致**。

### 4.2 与 phaseD 的关系

phaseD 归因的 "scipy CPU `firwin(801)+fftconvolve` vs cusignal GPU" 漂移是 **detector** 阶段（`src/preprocessing.py::_apply_notch_legacy_fir` 这一处，FIR forward-only）。这是 **完全独立的代码路径**，与 pack 阶段的 IIR + filtfilt 不重合。

→ 把 phaseD 的 "GPU vs CPU FP 累积" 跨阶段套到 pack centroid 漂移是 **缺乏证据支持的跳跃**。当前 Track B preflight 的 1.5–67 ms 漂移**不应**被假设为 "GPU vs CPU 后端差异"。

→ 走 GPU port 是 "**创造第三条路径**"，不是 "**复现老代码**"。

### 4.3 候选根因（剩下三条假设）

经过 §4.1 的源码确认，剩下三种假设：

- **H1（实现 bug）**：当前 `scripts/run_yuquan_lagpat_backfill.py::compute_stitched_lagpat` 与 `_legacy_resample_notch_band` 在 call 顺序、参数、类型转换、维度处理上与 legacy 字面差一点点，差异在 stft + centroid 处放大。
- **H2（环境漂移）**：当时生成 `.legacy_backup` 的 numpy / scipy 版本与现在的不同，纯 CPU IIR 计算同样会因 BLAS / FFT 后端差异 produce 不同的 ε 输出，再被 stft 放大到 ms 级。
- **H3（参数/边界处理）**：spectrogram 的 `nperseg` 边界、Gaussian smoothing kernel、frequency range slicing、NaN 传播规则 —— 这些下游确定性参数有不同行为。

Phase γ ablation 的目的就是**用证据从这三条里挑一条出来**，再决定 patch 方向。

## 5. Phase γ — 根因 ablation 计划（未实施）

### 5.1 γ.0 — Provenance verification of `.legacy_backup`（已完成 2026-04-26）

独立读了 legacy 源码。结论：**legacy pack 阶段的 active 路径确认是 scipy CPU `iirnotch + filtfilt` / `butter(3) + filtfilt`**。

证据：

- `p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter.py` 顶部 (line 15-16) 确实 import 了 `cupy` 和 `cusignal`，但：
  - line 181 `# batch_data = cusignal.resample_poly(...)` 显式注释掉
  - line 180-185 内 `return_seg_splitContiHigh` 的整个 GPU 分支也被注释掉（cp.asarray、cusignal.resample_poly、notch_filt_cu、band_filt_cu）
  - 实际 active 路径 (line 187-191) 调用 `scipy.signal.resample_poly` + `notch_filt` + `band_filt`
- `__main__` block (line 474+) 调用 `return_seg_splitContiHigh` (line 423) 和 `plot_perSeg_specCenter` (line 442)，两个函数的 active 路径都是 CPU IIR
- `highEvents_yuquan0910_utils.py:47-66` 的 `notch_filt` / `band_filt` 一字不差是 `signal.iirnotch(Q=30) + filtfilt` / `butter(3) + filtfilt`，全 243 行无任何 `cusignal | cupy | gpu | GPU` marker

→ "GPU port 是修复 centroid drift 的路径" **已被证伪**。GPU 完全从候选中划掉。

剩余三条假设（H1/H2/H3）由 γ.1 ablation 区分。

### 5.1bis γ.0 续 — refine_npz mtime 跨年漂

跑完 14-subject cohort replay 后发现 cohort_label = `fail`（46/176 strict pass）。深挖原因揭示一条**和 stitch FP bug 完全独立**的 data-archaeology 事实：

| subject | refine_npz mtime | legacy_backup mtime | refine schema | 14-subj 全 cohort verdict |
|---|---|---|---|---|
| gaolan | 2021-06-02 | 2021-06-16 | bipolar 类（其实是单电极）| ✅ 12/12 strict |
| dongyiming | 2021-06-25 | 2021-06-25 | 单电极 (`'A1','A2'`) | ✗ 0/12 (alias outer-drop bug) |
| wangyiyang | 2021-06-02 | 2021-06-03 | 单电极 | ✅ 9/9 (option A 22 cap, 3 skipped 与 legacy 一致) |
| sunyuanxin | 2021-06-02 | 2021-06-03 | 单电极 | ✅ 12/12 |
| xuxinyi | 2021-06-02 | 2021-06-03 | 单电极 | ✗ 0/13 (alias outer-drop bug) |
| zhangjinhan | 2021-06-02 | 2021-06-03 | 单电极 | ✅ 13/13 |
| zhangjiaqi | 2021-06-14 | 2026-04-20* | 单电极 | ✗ 0/13 (alias outer-drop bug) |
| chenziyang | **2026-04-09** | 2021-06-03 | bipolar pair | ✗ 0/12 (refine drift) |
| hanyuxuan | **2026-04-10** | 2021-06-03 | bipolar pair | ✗ 0/13 (refine drift) |
| huanghanwen | **2026-04-10** | 2021-06-03 | bipolar pair | ✗ 0/12 (refine drift) |
| huangwanling | **2026-04-10** | 2021-03-16 | bipolar pair | ✗ 0/12 (refine drift) |
| litengsheng | **2026-04-10** | 2021-06-03 | bipolar pair | ✗ 0/16 (refine drift) |
| chengshuai | **2026-04-09** | 2021-03-16 | bipolar pair | ✗ 0/12 (refine drift) |
| liyouran | **2026-04-10** | 2021-03-17 | bipolar pair | ✗ 0/12 (refine drift) |

*zhangjiaqi 的 .legacy_backup 时间戳是我们 2026-04-23 batch 把 raw_dir lagPat 移过去的时间，不是文件原始时间；文件内容是 2021 legacy 写的。

两条独立故障：

1. **alias_bipolar_to_left_with_arbitration 的 outer-drop 在单电极输入下过度删通道**——影响 dongyiming / xuxinyi / zhangjiaqi 这 3 个用了 2021 单电极 refine 的 subject。
2. **chenziyang 等 7 subject 的 `_refineGpu.npz` 在 2026-04-09/10 被 new pipeline 重新 detect/refine 写过**，覆盖了 2021 legacy refine。`.legacy_backup` 的 lagPat 是用 2021 refine 算的 picked，新 refine 的 events_count 与 picked 集合都不一样 → 数值上无法复原。

### 5.1bis-fix — alias auto-detect schema

新版 `alias_bipolar_to_left_with_arbitration`（commit 待写）：检测输入名是否含 `-`：

- 含 `-`（bipolar pair input，e.g. 新 pipeline detector 输出）：维持原行为（alias-collapse 到左触点 + 每 shaft drop 最高 num alias 与 legacy bipolar reref 输出对齐）
- 不含 `-`（单电极 input，e.g. 2021-era legacy refine 已经 alias-collapsed）：alias-collapse 是 no-op，**跳过 outer-drop**（legacy refine 已经把 outermost 去掉过了）

247 in-scope tests post-fix 全绿。修复后只需重跑 dongyiming / xuxinyi / zhangjiaqi 三个 subject，其他 11 个 unaffected。

### 5.1bis-defer — chenziyang 等 7 subject 的 refine drift

这一条不是代码 bug，是数据档案问题：原 2021 refine 在 2026-04 被 new pipeline 重写覆盖。可选方案：

- (i) 接受这 7 个 subject 在 Track B 上 "refine drift, uncomparable" 状态。
- (ii) 找 2021 refine 的备份（git / 团队其他 mount 点 / 时光机），把它们恢复到 `<raw>/<subject>/_refineGpu.npz`，再重跑 replay。
- (iii) 不复原，把 14-subject cohort 缩到 7-subject "refine-stable" cohort（gaolan / wangyiyang / sunyuanxin / zhangjinhan / dongyiming / xuxinyi / zhangjiaqi）作为 numerical equivalence 主张的范围；其余 7 subject 在 status doc 显式列为 "refine 数据已被覆盖，不能与 legacy backup 数值对齐"。

推荐 (iii) — 7 subject 已经足以 demonstrate 同算法+同输入下 pack+lagPat 数值复刻能力。代码改动不需要，只是合同口径收紧。如果后续找到 2021 refine 备份，把它们补回来再 audit 就能扩到 14。

### 5.2 γ.1 — 逐层 ablation tool（已完成 + 已诊断 + 已修复，2026-04-26）

**实施**：`scripts/audit_yuquan_pack_layer_ablation.py` (read-only / scratch-write only)，对每条 record 跑两个 branch（**Branch P** = production，**Branch L** = 字面 legacy reimpl 直接来自 `p16_packGroupEvents_*:76-174 + 232-249` + `highEvents_yuquan0910_utils.py:47-66` + `p16_cuda_24h_bipolar.py:82-119`），抓六层中间数组（L1 reref / L2 resample_poly / L3 notch / L4 band / L5 stitched / L6 centroid），做 `R0/P/L` 三方比对。

**两条 record 的 ablation 报告**：

| record | n_ev | L1 P-L | L2 P-L | L3 P-L | L4 P-L | L5 stitched mismatch? | L6 P-L (s) | L6 R0-P (s) | L6 R0-L (s) |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| FA0013L8 | 38 | 0 | 0 | 0 | 0 | seg 6: shape_P=[12,480] vs shape_L=[12,481] (1 sample) | 1.509e-3 | **1.509e-3** | **3.331e-16** |
| FA0013KP | 1224 | 0 | 0 | 0 | 0 | 多段 1-sample 缺失 | 3.692e-2 | **3.692e-2** | **1.421e-14** |

关键观察：

1. **L1–L4 全部 bit-for-bit 0 maxabs**（同一份 scipy.signal 调用，结果完全一致 ✓）。
2. **L6 R0 vs L = 机器 ε 级**（3.3e-16 / 1.4e-14 s）—— Branch L（字面 legacy）在当前 env 上完美复刻 `.legacy_backup`。**H2（环境/版本漂移）已证伪**。
3. **L6 P vs L = ms 级 maxabs**（1.5 ms / 36.9 ms），与 L6 R0 vs P 完全相同（差距来源 = 把 R0 当 reference，Branch L 已与 R0 等价）。
4. **L5 段比对显示 stitched 数组在某些 segment 比 Branch L 少 1 sample**（FA0013L8 的 seg 6: P=480 vs L=481）。

→ 结论：**H1 确认**——drift 100% 是 production `build_stitched_window_signal` 在 `<raw>/<stem>_lagPat.npz` 写入侧的 FP 边界 epsilon bug。

**Bug 定位 + 根因**：

`src/group_event_analysis.py::build_stitched_window_signal`（修复前）每个 window 用 ceil/floor + 1e-12 epsilon 求 sample 索引：

```python
i0 = int(np.ceil((float(win.start) - float(start_sec)) * float(sfreq) - 1e-12))
i1 = int(np.floor((float(win.end) - float(start_sec)) * float(sfreq) + 1e-12))
```

举例：FA0013L8 seg 6 / event 12 = `(2604.72, 2605.02)`, `start_sec=2600.0`, `sfreq=800`：

- `(2605.02 - 2600.0) * 800` 在 float64 下 = **4015.99999999985** （subtraction FP error），不是 4016
- `floor(4015.99999999985 + 1e-12) = 4015`，1e-12 epsilon 不够把它推到 4016
- 我们的 piece 长度 = 4015 - 3776 + 1 = 240 samples
- legacy 用 `batch_t = start_sec + np.arange(n) / sfreq`，`batch_t[4016] = 2600 + 4016/800 = 2605.02`（FP 路径不同），`batch_t[4016] <= tw[1]=2605.02` 为 True，sample 4016 被纳入 → 241 samples
- 1 sample 差异在 stitched 信号末尾，进入 stft + centroid 后被放大成 ms 级 centroid 漂移；event 越多累积越大（解释 1.5 ms vs 37 ms 的 record-级 spread）

**修复**：`src/group_event_analysis.py:284`（commit 待写）改为字面 legacy 路径：

```python
batch_t = float(start_sec) + np.arange(n_samples, dtype=np.float64) / float(sfreq)
win_bool_vec = np.zeros(n_samples, dtype=bool)
lengths: List[int] = []
for win in windows:
    tw_bool = (batch_t >= float(win.start)) & (batch_t <= float(win.end))
    lengths.append(int(tw_bool.sum()))
    win_bool_vec |= tw_bool
stitched = arr[:, win_bool_vec]
split_border_t = np.cumsum(np.asarray(lengths, dtype=np.float64)) / float(sfreq)
```

无算法变化、无后端变化、无新参数；纯 FP 路径切到 legacy 的 `start_sec + i/sfreq` 而不是 `(t - start_sec) * sfreq`。

**修复后再跑 γ.1**：

| record | L1-L4 | L6 P vs L | L6 R0 vs P | L6 R0 vs L |
|---|---|---:|---:|---:|
| FA0013L8 | 0 / 0 / 0 / 0 | **2.220e-16** ε | **4.441e-16** ε | 3.331e-16 ε |
| FA0013KP | 0 / 0 / 0 / 0 | **1.421e-14** ε | **2.132e-14** ε | 1.421e-14 ε |

→ Branch P 现在与 R0 数值上 bit-for-bit equivalent within float64 ε。**γ.2 决策树 row 2（H1）路径已落实**。

51/51 in-scope contract tests post-fix 全绿（test_legacy_lagpat_centroid 的 inclusive-endpoint expectation 与 fix 一致，无 regression）。

**记录选择**：
- 低漂移 record：gaolan `FA0013L8`（38 events, lagPatRaw maxabs 1.5 ms, rank 100%）
- 高漂移 record：gaolan `FA0013KP`（1224 events, maxabs 36.9 ms, rank 64%）或 `FA0013KQ`（1114 events, maxabs 51.8 ms, rank 56%）

**新脚本** `scripts/audit_yuquan_pack_layer_ablation.py`（read-only / scratch-write only）：

对每一个 record：

| Branch | 描述 |
|---|---|
| **R0** | `<raw>/.legacy_backup/<stem>_lagPat.npz` 直读（legacy ground truth） |
| **P** | 当前 production 的 `pack_one_record`（即 `scripts/run_yuquan_lagpat_backfill.py`），抓 6 层中间数组 |
| **L** | `plot_perSeg_specCenter` 字面 reimpl（参考 `p16_packGroupEvents_*.py:76–82` 与 `highEvents_yuquan0910_utils.py:47–68`），抓同样 6 层 |

6 层定义（按 pack pipeline 顺序）：

- `L1` = `_legacy_bipolar_reref_and_drop` 后的信号
- `L2` = `scipy.signal.resample_poly(2, factor_down)` 后
- `L3` = `notch_filt(L2, …)` 后（`signal.iirnotch + filtfilt`）
- `L4` = `band_filt(L3, …)` 后（`butter(3) + filtfilt`）
- `L5` = `build_stitched_window_signal` 后的拼窗信号
- `L6` = `compute_stitched_spectrogram_centroids_legacy` 输出的 centroid 矩阵

**输出**：
- `results/lagpat_backfill/_audit/pack_layer_ablation/<subject>/<stem>_layers_R0_P_L.npz`（6 层 × 3 branch 的中间数组）
- `results/lagpat_backfill/_audit/pack_layer_ablation/<subject>/<stem>_diff_report.json` —— 每层 maxabs(R0 vs P, R0 vs L, P vs L) + max-diff cell 的 (channel, sample) 位置

**Branch L scratch 输出**：`results/lagpat_backfill_legacy_pack_readonly/<subject>/<stem>_*` —— legacy reimpl 的最终 lagPat / packedTimes，便于直接和 R0 比较。

**单元测试**：`tests/test_yuquan_pack_layer_ablation.py` —— 在合成输入上验证 single-branch 的层输出在重复运行下完全确定（NOT 一个 CPU vs GPU 比较测试，因为 ablation 不引入 GPU）。

### 5.3 γ.2 — 决策树

读 §5.2 输出的 `<stem>_diff_report.json`（在两个 record 上都看），按下表行动：

| `R0 vs L` | `P vs L` | 结论 | 行动 |
|---|---|---|---|
| match | match | 三方对齐 → 原"漂移"假设错（不可能的状态，但要记录） | 重新审 Phase A baseline 阈值；可能直接 accept 现在的代码 |
| match | differ | **H1 确认** —— 我们 production 与 legacy 字面 reimpl 在第一处分歧层差了一点 | 修 production 在第一处分歧层的 bug；重跑 γ.1；不引入 backend 概念，cohort 走 same backend |
| differ | match | 我们的 production 就是 legacy 字面；legacy_backup 来自不同 env / 不同历史分支（H2） | 文档化 env/version drift；除非能 pin 到具体 GPU env 版本，否则**不**做 GPU port |
| differ | differ | 两条 modern reimpl 都漂离了 R0，并且各自漂的方式不同（H3 / 其他） | 从 `P vs L` 的第一处分歧层开始查（spectrogram param / NaN handling / nperseg 边界） |

**只有**"`R0 vs L` differ AND `P vs L` match AND 能 pin 到 GPU env" 这一行才会触发 GPU port。在 γ.1 的报告产出之前，**绝不做 GPU 后端工作**。

### 5.4 γ.3 — Backend 命名规则（preemptive）

**当且仅当** §5.3 决策树指向需要新 backend 时，才走这一步。命名规则（提前定下来避免名称含糊）：

- `cpu_iir_legacy` —— 当前 pack 阶段算法 + backend，**永远是默认**。
- `gpu_iir_candidate` —— IIR 算法在 cupyx.scipy.signal 上的候选（仅当 H2 + GPU evidence 都成立时考虑）。
- `gpu_fir_detector_candidate` —— FIR `firwin+fftconvolve` 在 GPU 上的候选；这是 **detector** 阶段的算法，不是 pack 阶段的修复；只当 §5.3 surface 出 algorithmic mismatch 才相关。

`--filter-backend` 是 **REQUIRED** 显式指定，**绝不允许 `auto` 默认**：同一命令在装了 cupy 与未装 cupy 的机器上**必须**产出 bit-for-bit 一致的 lagPatRaw / lagPatRank。Replay 的 `summary.json` / `manifest.json` 记录显式 backend 值 + 库版本 + cuda runtime（如果非默认）。

### 5.5 γ.4 — Status doc 措辞修正（独立于 γ 决策树结果）

无论 γ.2 怎么走，下面这些 status doc 的修正都要做：

- 把 "14/14 detector cohort pass" 类措辞替换成 §2.2 的 **13 / 1 / 1 / 6** 精确分桶。
- 把 cohort 标签 `coarse_logic_divergence` 显式注明为"由 zhangjiaqi 一项 trigger，非普遍失败"。
- 把 `zhangjiaqi` cohort 表格行从 "Backfill-only (legacy skipped)" 移到 "Main cohort (have legacy lagPat)"，加一行 footnote 说明 reference scheme 不同。
- Track B preflight 的 12/12 / 3/12 / 0/12 三层 verdict 写进 status doc，链接回本 archive。
- "GPU filter backend" 章节 **只在** γ.2 真的 motivate 了 GPU 工作之后才加；当前不写。

### 5.6 γ.5 — 测试

新增：
- `tests/test_yuquan_pack_layer_ablation.py` —— 在合成输入上验证 ablation tool 自身的 single-backend 重复确定性。**不**做 CPU/GPU 数值比较单测（用户指出过这种测试无意义：< 1e-9 解释不了 ms 级漂移，> 1e-9 又只是普通 FP）。

保留：
- 31 个 in-scope contract tests（path-override 6 + matcher 10 + comparator 10 + legacy_block_presence_diff 3 + pack_params_config 4 - 已有 25 + 新加 6 path overrides，加和应为 31，跑过都绿）。

## 6. 验收 checklist（更新于 2026-04-26 cohort 跑完）

| # | 项 | 状态 |
|---|---|---|
| 1 | `run_subject` 注入点 + 31/31 in-scope tests | ✅ DONE |
| 2 | Track A 跑通 + verdict | ✅ DONE |
| 3 | Track B preflight on gaolan（修复前 baseline）| ✅ DONE |
| 4 | γ.0 — provenance verification（confirmed CPU IIR） | ✅ DONE |
| 5 | γ.1 — layer ablation tool + 跑 FA0013L8 / FA0013KP | ✅ DONE |
| 6 | γ.2 — 决策树读出 H1（production stitched FP bug） | ✅ DONE |
| 7 | 修复 1 — `build_stitched_window_signal` 字面 legacy 路径 | ✅ DONE |
| 8 | γ.3 — Track B preflight on gaolan **post-fix**（strict 12/12 ✓） | ✅ DONE |
| 9 | γ.4 — Track B 14-subject cohort run post-fix | ✅ DONE |
| 10 | 修复 2 — `alias_bipolar_to_left_with_arbitration` auto-detect schema | ✅ DONE |
| 11 | γ.4b — re-run dongyiming/xuxinyi/zhangjiaqi after alias fix | ✅ DONE |
| 12 | γ.5 — final cohort audit verdict（6/14 strict pass, 8 refine-drift fail） | ✅ DONE |
| 13 | status doc 措辞修正（cohort 表 `zhangjiaqi` 已修；Track A 13/1/1/6 已落；Track B 6/14 strict pass + 8 refine drift） | partial DONE — 还需补 final cohort verdict 段 |
| 14 | Topical commit（涵盖 fix1 + fix2 + ablation + archive + status doc + path-injection refactor + Track A/B drivers + tests） | 🟡 PENDING |

## 7. 已规避的设计错误

为了让后续 agent 不重复走这些坑，记录今天 user 已经否决的几条思路：

1. **"phaseD detector root cause = pack centroid drift root cause" 跨阶段假设** —— 缺乏证据。pack 阶段是独立 CPU IIR + filtfilt 路径，不在 phaseD 的 GPU vs CPU 范围。
2. **`--filter-backend auto` 默认** —— 工程雷。同一命令在不同机器产出不同 lagPat 是合同代码不能接受的。默认必须是 `cpu_iir_legacy`，候选 backend 显式 opt-in 并写入 manifest。
3. **CPU/GPU 输出 < 1e-9 的单测** —— 没意义。如果 < 1e-9 它解释不了我们看到的 ms 级 drift；如果 ≥ 1e-9 又只是普通 FP。真正该测的是**逐层 ablation 哪一层开始偏**。
4. **γ acceptance 先写"GPU 后 maxabs ≤ 1 ms"结论** —— 写结论再找证据。改成假设检验：先 ablation，再让证据决定要不要做 backend 改动。
5. **Track A 写"14 usable detector cohort 全过"** —— 不准确，因为 zhangjiaqi 是 reference-scheme divergent 而非 FP-drift。要写 13/1/1/6 精确分桶。
6. **直接 GPU port 替代 ablation** —— 缺乏证据支持的 patch，结果只是创造第三条数值路径，不是复现 legacy。

## 8. 后续 agent 的 fast path

- "Track A / Track B 现在做到哪一步了？" → 看本档案 §0 TL;DR + §6 验收 checklist。
- "为什么 Track A cohort verdict 是 coarse_logic_divergence 但是 13/14 又都 pass？" → 看 §2.6。
- "为什么不直接 port 到 GPU？" → §4 + §7。
- "γ ablation 的下一步是什么？" → §5.1 → §5.2 → §5.3，按决策树判。

## 9. 相关脚本/路径

- `scripts/run_yuquan_lagpat_backfill.py` — same-source contract driver；本次 refactor 加了 `legacy_refine_root` / `legacy_gpu_root` / `out_dir` / `backup_dir` / `same_source_assertion` 注入点。
- `scripts/run_yuquan_legacy_refine_replay.py` — Track B replay driver。
- `scripts/audit_yuquan_detector_event_match.py` — Track A audit driver。
- `scripts/audit_yuquan_legacy_refine_replay.py` — Track B comparator + 三层 verdict + provenance gate。
- `scripts/audit_yuquan_pack_layer_ablation.py` — γ.1 待实施。
- `tests/test_yuquan_detector_event_match.py` / `tests/test_yuquan_legacy_refine_replay_audit.py` / `tests/test_run_subject_path_overrides.py` —— 26 个新测试。
- `results/lagpat_backfill/_audit/detector_attribution/` —— Track A 输出。
- `results/lagpat_backfill_legacy_refine_replay/gaolan/` —— Track B preflight artifacts。
- `/tmp/track_b_preflight/cohort_replay_audit.{json,md}` —— Track B preflight comparator 报告（待迁到 `results/lagpat_backfill/_audit/legacy_refine_replay/preflight/`）。
