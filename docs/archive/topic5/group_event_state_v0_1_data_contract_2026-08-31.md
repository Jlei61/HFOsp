# Group-Event State v0.1 — 数据合同（frozen 2026-08-31）

本文件是 v0.1 的**数据语义冻结件**。任何下游代码在读取字段前必须先在这里
确认该字段回答的是哪一层问题（CLAUDE.md §6.2）。归档代号密度高是有意的；
面向用户的解释见 plain-language 报告。

产出根目录：`results/epi_prssm/group_event_state/v0_1/`
缓存根目录：`/data/hfosp_group_event_state_v0_1/`（`/` 只剩 42 GB，缓存**不得**落在仓库盘）
运行环境：`/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`
（torch 2.5.1+cu124，与 Topic 5 既有 epi_prssm 线一致；`LD_LIBRARY_PATH` 需指向该 env 的 `lib`）

---

## 1. 一个"完整群体事件"的定义

一个时间步 = 一次 packed group event，**不是**单触点 IED、不是 rank step、
不是固定背景窗。事件由 legacy packer 定义：

- 事件时刻区间：`<record>_packedTimes*.npy` 的 `(t_start, t_end)`，单位为**记录内秒**；
- 事件内容：同一 packer 变体的 `<record>_lagPat*.npz`（`eventsBool` / `lagPatRaw` /
  `lagPatRank` / `lagPatFreq` / `chnNames` / `start_t`）。

**绝对时刻** = `start_t + packedTimes[:,0]`。已逐块验证 `start_t` 与
`block_start_epoch`（Epilepsiae SQL / Yuquan EDF header）**逐位相等**（|Δ| ≤ 1e-3 s，
41/41 患者、3896 blocks 全过）。

## 2. packer 变体配对（硬约束）

两个 legacy packer 对**同一记录给出不同事件表**：`FC10477Q` 老 packer 2965 行 / 6 通道，
`withFreqCent` 2601 行 / 8 通道。因此 packedTimes 必须**按变体配对**，不能按文件存在性选择：

| variant | lagPat 后缀 | packedTimes 后缀 | lagPatFreq |
|---|---|---|---|
| `withFreqCent` | `_lagPat_withFreqCent.npz` | `_packedTimes_withFreqCent.npy` | 有 |
| `legacy` | `_lagPat.npz` | `_packedTimes.npy` | 无 |

同一患者只允许使用**单一变体**（优先 `withFreqCent`）。cohort 实测：3812 blocks
`withFreqCent`，84 blocks `legacy`（7 位 Yuquan 患者只有老 packer 产物）。
回归测试：`tests/test_topic5_group_event_state_source_audit.py::test_packed_file_follows_the_lagpat_variant_not_file_existence`。

## 3. `lagPatRaw` 的层级与单位

`lagPatRaw` **不是**峰时刻、**不是**检测 onset、**不是**绝对记录时间。它是
`return_massCenterPat()` 在**分段拼接（stitched-per-segment）时间轴**上、对
`spectrogram(nperseg=0.05·fs, noverlap=0.8, mode='magnitude')` 做 gaussian 平滑后、
以 power³ 加权求得的**时频质心时刻，单位秒**。实测每块有 ~18（1h）/ ~36（2h）次
时间轴重置，佐证 200 s 分段。

**唯一可移植的用法**是 `eventsBool` 掩膜后的事件内相对差：

```
relative_delay[c, e] = lagPatRaw[c, e] − min_{c' ∈ participants(e)} lagPatRaw[c', e]
```

- 实测事件内 span 中位数 **28.9 ms**（p10 11.6 / p90 82.3 ms），**0.000%** 的事件超过
  packed core 时长 → 单位与掩膜规则已被数据自身证实。
- **phantom 污染**：非参与触点的 `lagPatRaw` 在 cohort 中位数 **100%** 是有限值
  （不是 NaN）。`lagPatRank` 同样 100% 有限（Topic 0 §3.1 已知）。任何未过
  `eventsBool` 的用法都在读伪值。

## 4. tied recruitment groups

producer 的质心分辨率 = 一个 spectrogram hop = `0.05·fs·0.2 / fs` = **10 ms**，与采样率无关。
定义 `TIE_TOLERANCE_SECONDS = 0.010`；一个事件的参与触点按 `relative_delay` 排序后，
相邻差 > 10 ms 处断开（single linkage）即为 tied recruitment group。
实测每事件 tied group 数中位数 **2.0**（p90 = 3.0），参与触点数中位数 5.0 →
**legacy 的 `argsort(argsort(x))` 在中位事件上把 5 个触点强行排成 5 个名次，
而可分辨的招募步只有 2 个**。这正是 v0.1 用 exact delay 取代 rank 的动因。
`legacy_rank` 仍然保留，仅作为 `a2` 低信息 ablation 输入。

## 5. detector montage（两队列语义不同，禁止混用）

| dataset | reference | lagPat 行含义 | 来源 |
|---|---|---|---|
| Yuquan | `bipolar` | **anode**（`E11` 实为 `E11−E12`） | `_gpu.npz::reference_type` + `bipolar_pairs` |
| Epilepsiae | 全局 CAR | 单触点（`GD8`） | producer `epilepsiae_detectHFOs.avg_rerefAndDrop_eeg`：`data − mean(data over retained intracranial)` |

- Epilepsiae CAR 通道集 = `_gpu.npz::chns_names`，实测等于 `.head::elec_names` 去掉非颅内道
  （958：97 → 96，剔除 `ECG`）。
- 83 个 Yuquan block 缺 `_gpu.npz`。montage 由标签规则重建，并**永久标记**
  `montage_provenance="derived_from_label_rule_no_gpu"`。规则合法性证据：
  (a) cohort 内 972 个**已记录** bipolar pair **零例外**全是同杆 `(n, n+1)`；
  (b) `pengzihang` 唯一保留 `_gpu.npz` 的 block 上，重建结果与记录值 **12/12 一致**；
  (c) 7 位患者的全部 lagPat 标签及其 `n+1` 邻居在各自 EDF 中**全部存在**（0 unresolvable）。
  cohort 计数：3813 blocks `gpu_npz` / 83 blocks derived。

## 6. 三个参考视图

| view | Yuquan | Epilepsiae |
|---|---|---|
| `detector` | `anode − cathode`（= 检测所用） | `contact − mean(96 intracranial)` |
| `bipolar` | **等于 detector**（manifest `bipolar_equals_detector=true`，不重复存盘） | `contact(n) − contact(n+1)`，缺则 `−(contact(n−1) − contact(n))` 并记 sign |
| `shaft_car` | `anode − mean(同杆全部触点)` | 同左 |

每个视图带自己的 reference token；模型侧 waveform 分支对每个 view 加可学习 embedding，
**禁止**把两个 montage 当同一物理信号拼接。

## 7. 频带（missing ≠ 0）

原生采样率下计算，Nyquist guard = 8 Hz，`high + 8 < fs/2` 才算 supported：

| band | Hz | 支持患者数 |
|---|---|---|
| `ied_low` | 1–30 | 41/41 |
| `gamma` | 30–70 | 41/41 |
| `low_ripple` | 80–110 | 41/41 |
| `ripple` | 80–150 | 41/41 |
| `fast_ripple` | 150–250 | **39/41**（`epilepsiae_253` / `epilepsiae_139` 为 512 Hz） |

不支持的频带**标为 missing 并置 mask**，不填 0。事件级：1,684,687 / 1,774,188 事件支持 fast ripple。

## 8. 端到端正确性证据（承重）

在原生数据上重建 detector 视图后，取 80–150 Hz Hilbert 包络，比较 event core 峰值
与 core 外基线中位数之比：

| | 参与触点 | 非参与触点 | corr(重建包络峰时, producer 质心) |
|---|---|---|---|
| `epilepsiae_958`（200 事件） | **15.79×** | 4.57× | median r = **+0.929**（94.5% 事件 r>0.5） |
| `yuquan_chengshuai`（200 事件） | **9.94×** | 3.60× | median r = **+0.566**（53.5% 事件 r>0.5） |

即：**指针、montage、事件对齐、时间轴四者同时正确**才可能出现这个结果。
（Yuquan 相关系数较低是预期的：其 core 0.5 s、质心为 power³ 加权而非包络 argmax。）

## 9. 采样窗与 shard 字段

- `core` = packedTimes 区间；`context` = core ± 0.25 s（存盘）；`filter_pad` = 再 ± 0.5 s（仅供滤波，丢弃）。
- **宽度按块取单一名义值**：`n_core = round(median(core_dur)·fs)`，
  `ctx_start = round(t0·fs) − round(0.25·fs)`，`ctx_stop = ctx_start + n_ctx`。
  独立 round `t0·fs` 与 `t1·fs` 会让恒定 0.11 s 的 core 在 112/113 samples 之间跳变，
  仅因 `t0` 的小数相位就丢掉一个块 **37%** 的事件（`epilepsiae_1073/107300102_0014`：236/377 → 修复后 376/377）。
- **事件永不因波形不可用而离开时钟**：只有 `has_waveform=False`（滤波窗越出块边界，
  实测每块 ≈1 个）。丢事件会静默缩短状态模型积分的 inter-event interval。

shard（`<record>.npz`，uncompressed，atomic rename）字段：
`event_abs_time / core_start_seconds / core_end_seconds / core_seconds_raw /
core_start_sample / core_stop_sample / ctx_start_sample / ctx_stop_sample /
has_waveform / contact_ok / participation / relative_delay_s / tied_group_id /
legacy_rank / legacy_freq_centroid / band_available / band_envelope (n,C,B,192 f16) /
band_features (n,C,B,5 f32) / cross_band_lag_s (n,C,10 f32) /
waveform_<view> (n,C,T f16) / background_time_s / background_features`。

`band_features` 列序：`peak_time_s, centroid_time_s, log_integrated_energy, width_s, log_peak_amplitude`（core 内计算）。
`cross_band_lag_s` = 10 个频带对的包络互相关峰值滞后，搜索范围截断在 ±core 时长。

## 10. 背景 SEEG（辅助观测，不是第二个时钟）

固定规则、与事件无关：30 s 网格上取 2 s 窗，**与任何 packed event core 重叠即丢弃该网格点**。
每锚点每触点特征：各 supported band 的 log power、log variance、lag-1 autocorrelation、
90% spectral edge。事件 `i` 只允许使用**结束时刻严格早于 `core_start(i)`** 的最近一个锚点，
并附带 `background_age`。

## 11. 发作标签的唯一合法用途

- **排除**：与 `[eeg_onset, eeg_offset]` 有交集的事件从 interictal 流中剔除
  （cohort：27,229 / 1,774,188 = 1.5%）。preictal 事件**全部保留**。
- **记账**：`time_to_next_seizure` / `time_since_prev_seizure` 逐事件保存，供 H2b 分层。
- **禁止**：任何 seizure label 进入 representation training 的输入。
- Yuquan seizure 来自 pr1 spatial-extent **detection**（非临床标注），且 10/21 位患者
  0 个检出事件——只能读作"未检出"，不可读作"无发作"。Epilepsiae 有 `pattern` /
  `classification` 字段，H2b 必须按 pattern 分层报告。

## 12. 序列与切分

- **序列 = contiguous coverage session**。断开条件：(a) 相邻 packed block 时钟间隙 > 2 s；
  (b) 中间存在**有记录但无 group-event 产物**的 block（该小时是"未观测"，不是"无事件"）。
  cohort：344 sessions（≥1k 事件 183 / ≥5k 77 / ≥10k 48）。
- 状态在 session 起点重置为可学习初值；**不跨 gap 传播状态**。
- 切分按患者自身时间顺序 70/10/20（train/val/test），test 是该患者最晚的一段。

## 13. 与既有 Topic 5 资产的关系

- 本线**不使用** `results/topic5_interictal_rank_distribution/dataset_v0_4`
  的已筛 definite-interictal 子集；直接扫描完整 artifact（`epilepsiae_958`：165,577 事件，
  高于旧线 123,419）。旧 34 人 cohort 成员关系保存在 `source_audit.json::historical_34_cohort`。
- 本线**不使用** `results/lagpat_broad*`（top_n=20 重打包）。理由：该分支会把
  large-narrow 患者**收窄**（`zhangbichen` 52→20），且事件表与 legacy 主线不同。
  记为可选扩展，不进入 v0.1。
