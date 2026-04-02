# Epilepsiae 数据集结构调研

**分析日期**: 2026-04-02  
**数据路径**: `/mnt/epilepsia_data`

---

## 核心判断

`Epilepsiae` 不是 Yuquan 那套 EDF 世界。这里真正的原始合同是：

- 原始信号：`*.data + *.head`
- 临床与发作元数据：`all_data_sqls/*.sql`
- 间期间期分析中间产物：`interilca_inter_results/all_data_lns/<subject>/all_recs/*`

真正该信的时间真值，不是老脚本里遍历 `.head` 拼出来的近似时间轴，而是 SQL 里的：

- `recording.begin/end/duration`
- `block.begin/end/gap`
- `seizure.eeg_onset/eeg_offset/clin_onset/clin_offset`

`.head.start_ts` 适合做块级校验，不该反过来当主真值。

---

## 1. 数据根概览

顶层目录：

```text
/mnt/epilepsia_data/
├── inv/
├── inv2/
├── inv_1_part/
├── epilepsiae_3patient/
├── all_data_sqls/
├── interilca_inter_results/
├── epilepsiae_dataFetch.py
├── epilepsiae_utils.py
└── readme
```

目录语义：

- `inv/`, `inv2/`, `inv_1_part/`, `epilepsiae_3patient/`: 原始患者数据
- `all_data_sqls/`: 患者、recording、block、seizure 等数据库导出
- `interilca_inter_results/all_data_lns/`: 老间期间期文章使用的 20 个患者中间产物

---

## 2. 全量规模

基于 `scripts/survey_epilepsiae_dataset.py` 的审计结果：

| 指标 | 数值 |
|---|---:|
| SQL subject 数 | 27 |
| recording 数 | 75 |
| block 数 | 5483 |
| seizure 条目数 | 542 |
| 有 `all_data_lns` 中间产物的 subject | 20 |
| 有 `sub_refineGpu.npz` 的 subject | 20 |

关键结论：

- 老间期间期文章里常说的 **20 个 subject**，只是**已有中间产物的 cohort**，不是挂载盘里的全量数据。
- 当前挂载盘里还有 **7 个只有原始数据/SQL、没有 `all_data_lns` 中间件**的 subject：
  - `115`, `264`, `273`, `375`, `565`, `862`, `970`

已有间期间期中间产物的 20 个 subject：

- `1073`, `1077`, `1084`, `1096`, `1125`, `1146`, `1150`, `139`, `253`, `384`
- `442`, `548`, `583`, `590`, `620`, `635`, `818`, `916`, `922`, `958`

---

## 3. 原始数据合同

### 3.1 文件组织

原始块按患者/住院/recording 分层，例如：

```text
/mnt/epilepsia_data/inv/pat_95802/adm_958102/rec_95800102/
├── 95800102_0000.data
├── 95800102_0000.head
├── 95800102_0000_gpu.npz
├── 95800102_0001.data
├── 95800102_0001.head
├── 95800102_0001_gpu.npz
└── ...
```

单个 block 的命名规律：

- `<recording_id>_<block_no:04d>.data`
- `<recording_id>_<block_no:04d>.head`
- 常见还有同名 `*_gpu.npz`

### 3.2 `.head` 字段

代表样例：`95800102_0000.head`

- `start_ts=2009-07-06 18:54:09.000`
- `sample_freq=1024`
- `duration_in_sec=3600`
- `num_channels=97`
- `elec_names=[..., ECG]`

最小可依赖字段：

- `start_ts`
- `duration_in_sec`
- `sample_freq`
- `num_channels`
- `elec_names`

### 3.3 采样率与通道规模

recording 级采样率分布：

- `1024 Hz`: 48 recordings
- `256 Hz`: 23 recordings
- `512 Hz`: 4 recordings

结论：

- 这套数据不是单一采样率，后续任何统一处理都必须按 recording 处理。
- 不能假定和 Yuquan 一样固定 `2000 Hz`。

recording 级通道数分布很散，常见值包括：

- `85`, `96`, `122`, `93`, `95`, `119`, `71`, `109`

说明：

- 不同 subject / recording 的电极配置差异明显。
- 任何“固定通道模板”思路都不靠谱。

---

## 4. modality 现状

SQL admission 里：

- `seeg = TRUE`: 27/27 subjects
- `ieeg = TRUE`: 27/27 subjects

但老代码真正的 modality 处理很粗糙，只做了一件事：

- 用 `eeg_chns` 黑名单把 scalp EEG / ECG / EOG / EMG 排除
- 剩下的全部当成 intracranial

这意味着：

- 我们可以可靠地区分“有 scalp/aux 通道”与“有 intracranial 通道”
- **不能**仅靠老代码可靠地区分 SEEG 和 ECoG

当前 survey 结果显示：

- 所有 subject 的 `.head` 都同时包含 EEG/aux 通道和 intracranial 通道
- 所以后续文档里更安全的说法应是：
  - `has_eeg = True`
  - `has_intracranial = True`
  - `SEEG/ECoG 细分仍需额外规则或元数据`

---

## 5. 时间轴与连续性

### 5.1 真值来源

时间轴建议优先级：

1. SQL `recording`
2. SQL `block`
3. `.head.start_ts`

原因很简单：

- `recording` 给长程开始/结束/净时长/总时长/完整度
- `block` 给块级 `gap`
- `.head` 只给块本身的本地元数据

### 5.2 `.head` 与 SQL 是否对齐

本次全量核对结果：

- block 数：5483
- `head.start_ts - sql.block.begin` 的绝对误差
  - median = `0.0s`
  - p95 = `0.0s`
  - max = `0.0s`

结论：

- 当前挂载数据上，`.head.start_ts` 与 SQL `block.begin` 是**精确对齐**的。
- 所以后续可以放心用 `.head` 做局部读取，但主时间轴仍建议由 SQL 驱动。

### 5.3 连续性不能一刀切

这套数据的“连续”分两层：

1. **recording 内 block 是否连续**
2. **subject 多个 recording 之间是否连续**

审计结果：

- `75` 个 recording 中，只有 `10` 个 recording 在 block 级别完全连续
- `27` 个 subject 中，只有 `5` 个 subject 在 recording 级别没有明显跨-recording gap
- `22` 个 subject 存在明显 inter-recording gap

典型大 gap：

- subject `958`: 最大 block gap = `28810s`
- subject `442`: 最大 block gap = `28806s`
- subject `384`: 最大 inter-recording gap = `28920s`
- subject `916`: 最大 inter-recording gap = `16096s`

结论：

- 这不是“24h 固定连续监测”那种干净数据。
- 后续任何 day/night、发作前窗口、interictal interval，都必须基于**绝对时间轴 + gap 感知**。

### 5.4 SQL commentary 很重要

`recording.commentary` 里有：

- `net duration`
- `gross duration`
- `% complete`

例如 `95800102`：

- `net = 242837s`
- `gross = 250432s`
- `96.97% complete`

这说明：

- recording 本身就允许存在缺失段
- “有 70 个 block” 不等于“完全连续”

---

## 6. Seizure 标注质量

### 6.1 SQL `seizure` 表字段

核心字段：

- `eeg_onset`
- `clin_onset`
- `eeg_offset`
- `clin_offset`
- `pattern`
- `classification`
- `vigilance`
- `recording`
- `block`

### 6.2 完整性

全量 seizure 条目：`542`

其中：

- 完整 EEG interval（有 onset 和 offset）：`523`
- 完整 clinical interval：`539`

说明：

- 标注整体比 Yuquan 的 EDF annotation 干净得多
- 但并不是 100% 完整，仍有少数条目缺 `eeg_offset` 或 `clin_offset`

### 6.3 `vigilance` 现状

分布：

- `A`: 223
- `2`: 166
- `?`: 137
- `1`: 14
- `R`: 2

结论：

- `vigilance` 能提供部分行为/觉醒状态信息
- 但标签含义需要额外对照原数据库说明，**不能直接当 day/night**

### 6.4 发作间 interval

本次 survey 已统一导出 `seizure_interval_from_prev_sec`，可直接做：

- subject 内相邻 seizure onset 间隔
- 发作前窗口筛选
- `>3h` 这类过滤

例如：

- `FR_1073` 的最小发作间隔约 `838.69s`
- `FR_1084` 的最小发作间隔约 `54.96s`

结论：

- 不少 subject 存在**密集成簇发作**
- 发作前分析必须显式过滤短间隔 seizure，不能默认所有 seizure 相互独立

---

## 7. 中间产物覆盖率

### 7.1 原始树与间期树的关系

原始树下经常已有：

- `*.data`
- `*.head`
- `*_gpu.npz`

而老间期间期结果集中在：

`/mnt/epilepsia_data/interilca_inter_results/all_data_lns/<subject>/all_recs/`

这里常见：

- `*_gpu.npz`
- `*_lagPat.npz`
- `*_lagPat_withFreqCent.npz`
- `*_packedTimes.npy`
- `*_packedTimes_withFreqCent.npy`
- `sub_refineGpu.npz`

### 7.2 覆盖率现状

recording 级 `lagPat` 覆盖：

- 全覆盖：33 recordings
- 部分覆盖：19 recordings
- 完全没有：23 recordings

subject 级同步性 manifest 分层：

- `ready_full_artifacts`: 16 subjects
- `ready_partial_artifacts`: 4 subjects
- `missing_interictal_artifacts`: 7 subjects

subject 级：

- 20 个 artifact subject 全都有 `sub_refineGpu.npz`
- 但只有 8 个 subject 的 `lagPat` 块数与 artifact `*_gpu.npz` 块数完全一致

典型“有 artifact，但覆盖不完整”的 subject：

- `1084`: `221 / 252`
- `1096`: `160 / 165`
- `1146`: `80 / 117`
- `818`: `222 / 255`

结论：

- “subject 在 `all_data_lns` 里存在” 不等于 “所有 block 都有 lagPat”
- 后续如果把 Epilepsiae 纳入同步性分析，必须先按 block/recording 检查资产完整度
- 同步性分析不该再手工挑人，应该直接消费规范化 manifest

---

## 8. 老代码里真正值得复用的入口

### 8.1 可复用

1. `ReplayIED/inter_events/epilepsiae_interictal/epilepsiae_utils.py::epilepsiae_block`
   - 作用：读取 `.head` 与 `.data`
   - 价值：原始 block 访问入口清楚

2. `ReplayIED/inter_events/epilepsiae_interictal/epilepsiae_cutIctal.py::return_timeStamp_fromDatetime`
   - 作用：解析 SQL 时间字符串
   - 价值：逻辑简单，可吸收进新代码

3. `ReplayIED/inter_events/epilepsiae_interictal/epilepsiae_cutIctal.py::find_corrFileAndT`
   - 作用：把 seizure 对齐到具体 block
   - 价值：思路可复用，但应该改成基于 SQL `block`，而不是只扫 `.head`

### 8.2 不该复用的坏味道

- 硬编码绝对路径 `/home/niking314/...`
- 同一个 helper 到处复制
- 默认假设 `all_recs` 中已有 `lagPat` / `packedTimes`
- 用 `eeg_chns` 黑名单硬充 modality 真值
- 直接从 `.head` 拼时间轴，而不吃 SQL `block.gap`

结论：

- 可以复用的是**数据契约和解析思路**
- 不该复用的是**老脚本组织方式**

---

## 9. 我们需要的统一格式

本次 survey 已经把后续真正要用的字段统一出来，建议以 CSV/Parquet 维持下面四张表：

1. `subject_inventory`
   - cohort 级总览

2. `recording_inventory`
   - recording 级时间轴、连续性、资产覆盖率

3. `block_inventory`
   - block 级真值表
   - 最关键字段：
     - `subject`
     - `recording_id`
     - `block_id`
     - `block_stem`
     - `block_start_epoch`
     - `block_end_epoch`
     - `gap_to_prev_sec`
     - `sample_rate_sql`
     - `head_start_epoch`
     - `head_sql_start_delta_sec`
     - `has_eeg`
     - `intracranial_channels`
     - `raw_gpu_exists`
     - `has_packed_times`
     - `has_lagpat`
     - `has_lagpat_freq`

4. `seizure_inventory`
   - seizure 级真值表
   - 最关键字段：
     - `subject`
     - `recording_id`
     - `block_id`
     - `eeg_onset_epoch`
     - `eeg_offset_epoch`
     - `clin_onset_epoch`
     - `clin_offset_epoch`
     - `has_complete_eeg_interval`
     - `has_complete_clin_interval`
     - `vigilance`
     - `classification`
     - `seizure_interval_from_prev_sec`

---

## 10. 对后续分析的直接约束

### 10.1 Day/Night

现在可以做 **显式时区** 的 wall-clock day/night，而且当前挂载数据上这件事已经不是拍脑袋：

- `27/27` 个 subject 的 `hospital=UKLFR`
- `27/27` 个 `patient_code` 都是 `FR_*`
- `src/epilepsiae_dataset.py` 已把默认规则收敛成正式接口：
  - `EpilepsiaeTimeConfig`
  - `resolve_epilepsiae_timezone()`
  - 默认映射：`UKLFR -> Europe/Berlin`
  - day/night 规则：`08:00-20:00` 记为 `day`，其余记为 `night`
  - 仍然保留 `timezone_overrides` / `recording_timezone_overrides`

因此当前挂载数据上的结论变成：

- `day_night_wall_clock_possible = True`
- `day_night_timezone_known = True`
- `day_night_reliable_without_override = True`

### 10.2 发作前同步性分析

要做 Figure 7B/C 那类分析，前提必须同时满足：

- 有可靠 seizure inventory
- 有 `packedTimes`
- 有 `lagPat`
- 已过滤短 seizure interval
- 已处理 recording/block gap

本次已经生成可直接喂给同步性分析的 subject 清单：

- `results/epilepsiae_sync_subject_manifest.csv`

默认准入规则：

- 至少 `2` 个完整 EEG seizure intervals
- 至少 `1` 个相邻 seizure interval `>= 3h`
- 已有 `lagPat + packedTimes`

manifest 分层语义：

- `ready_full_artifacts`: 可以直接进入同步性分析主队列
- `ready_partial_artifacts`: 可以分析，但必须透明标记资产覆盖不完整
- `missing_interictal_artifacts`: 只能等后续补 pipeline

本次已直接把 `interictal_synchrony` 接到 manifest：

- 新增接口：`src/interictal_synchrony.py::build_interictal_synchrony_from_legacy_lagpat()`
- 新增批量入口：`src/interictal_synchrony.py::run_epilepsiae_interictal_synchrony_from_manifest()`
- 新增脚本：`scripts/run_epilepsiae_interictal_synchrony.py`
- 实跑结果：
  - `ready_full_artifacts = 16` 个 subjects
  - 共处理 `2962` 个 `lagPat` blocks
  - 输出目录：`results/interictal_synchrony/epilepsiae_ready_full_artifacts`
  - 汇总表：`results/interictal_synchrony/epilepsiae_ready_full_artifacts/epilepsiae_ready_full_artifacts_interictal_sync_summary.csv`

### 10.3 Epilepsiae 是否能直接纳入当前项目

答案是：

- **可以纳入**
- 但必须先把它当成一套独立数据契约处理
- 不能拿 Yuquan 的 EDF pipeline 直接套

---

## 11. 当前已生成的 survey 结果

脚本：

- `scripts/survey_epilepsiae_dataset.py`

正式可复用接口：

- `src/epilepsiae_dataset.py`
  - `EpilepsiaeTimeConfig`
  - `resolve_epilepsiae_timezone()`
  - `survey_epilepsiae_dataset()`
  - `save_epilepsiae_inventory()`
  - `build_epilepsiae_sync_subject_manifest()`
  - `save_epilepsiae_sync_subject_manifest()`
- `src/interictal_synchrony.py`
  - `build_interictal_synchrony_from_legacy_lagpat()`
  - `run_epilepsiae_interictal_synchrony_from_manifest()`

输出：

- `results/epilepsiae_subject_inventory.csv`
- `results/epilepsiae_recording_inventory.csv`
- `results/epilepsiae_block_inventory.csv`
- `results/epilepsiae_seizure_inventory.csv`
- `results/epilepsiae_dataset_summary.json`
- `results/epilepsiae_sync_subject_manifest.csv`
- `results/interictal_synchrony/epilepsiae_ready_full_artifacts/`

---

## 最短结论

- `Epilepsiae` 全量是 **27 个 subject**，不是 20 个。
- 其中 **20 个**有老间期间期中间产物，**7 个**只有原始数据/SQL。
- 原始数据主合同是 `*.data + *.head + SQL`，不是 EDF。
- 时间真值应优先信 SQL `recording/block/seizure`；`.head` 只做块级校验。
- `head.start_ts` 与 SQL `block.begin` 当前是 **0 秒误差对齐**。
- 数据连续性不干净，gap 普遍存在，后续分析必须 gap-aware。
- seizure 标注总体可用，但不是每条 EEG/clinical interval 都完整。
- `vigilance` 不是 day/night；day/night 现在由显式 `Europe/Berlin` wall-clock 规则给出，并保留 override。
- `results/epilepsiae_sync_subject_manifest.csv` 已经把 subject 分成“可直接分析 / 资产不完整 / 缺中间产物”三类。
- `ready_full_artifacts` 的 `16` 个 subjects 已经实际跑完 `2962` 个 block 的同步性指标。
- 老代码里能复用的是解析思路，不是脚本组织方式本身。
