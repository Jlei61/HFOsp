# Yuquan lagPat Backfill Review, Validation, and Failure Contract

> Date: 2026-04-17
> Scope: Yuquan 11-subject lagPat / packedTimes backfill for Topic 1 and Topic 2
> Decision baseline: write assets into raw-data subject directories, back up existing lagPat for `gaolan` / `dongyiming` / `wangyiyang`, and generate all 11 subjects from the new `results/hfo_detection/<subject>/` detector outputs.

---

## 核心判断

**值得做，但不能糊里糊涂地做。**

这件事的真实目标不是“把 11 个 subject 跑出两个文件”，而是：

1. 用**旧 contract** 生成 `*_packedTimes.npy` + `*_lagPat.npz`
2. 证明这些资产在**范式上和老代码兼容**
3. 在不自欺欺人的前提下，把 Topic 1 / Topic 2 cohort 从 `30` 扩到 `41`

如果只做第 1 步，不做第 2 步，这不是科学扩容，这是批量造文件。

---

## 已批准的执行边界

### 输出位置

- 写回数据目录：`/mnt/yuquan_data/yuquan_24h_edf/<subject>/`
- 对已有旧资产的 3 个 subject：
  - `gaolan`
  - `dongyiming`
  - `wangyiyang`
- 先备份：
  - `*_lagPat.npz.legacy_backup`
  - `*_packedTimes.npy.legacy_backup`

### 数据来源

- 11 个 subject 统一使用：
  - `results/hfo_detection/<subject>/*_gpu.npz`
  - `results/hfo_detection/<subject>/_refineGpu.npz`
- 不混用数据目录中的旧 `_gpu.npz` 做正式回填

### 通道命名 contract

- 新 lagPat 一律写成**旧式左触点别名**，例如：
  - `A1-A2 -> A1`
- 如果 alias 后发生冲突，默认策略是：
  - 保留 `events_count` 更高者
- 但这不是“静默允许”的自由裁量。冲突必须进 QC 报告。

### 本轮不做的事

- 不把 Topic 3 / SOZ-AUC 混进来
- 不在同一个改动里顺手重写 Topic 1/2 统计逻辑
- 不改变 Topic 1/2 现有 loader contract
- 不生成“改良版” lagPat 格式

---

## 最大科学坑点

## 1. 41-subject cohort 仍然不是全同源

批准方案 B1 解决的是**11 个新增 Yuquan subject 内部一致**，不是**41 个总 cohort 完全同源**。

现状仍然是：

- 原有 30-subject Topic 1/2 cohort 主要来自旧检测 + 旧 packing / lagPat
- 新增 11 个 Yuquan subject 来自新检测 + 旧 contract 风格 packing / lagPat

这意味着正式表述必须诚实：

- 可以说：`41-subject extended cohort under legacy-compatible asset contract`
- 不可以说：`41 subjects all produced by the identical historical end-to-end pipeline`

### 科学后果

如果新增 11 个 subject 改变了：

- event count
- packed window density
- `n_participating`
- lag span
- propagation stereotypy
- periodicity peak behavior

你不能直接把变化归因于 sample size 增加。它也可能是 detector source drift。

### 结论

扩容后的 `41-subject` 结果必须先作为：

- **extended-cohort sensitivity analysis**

而不是立刻替换所有 `30-subject` 主结论。

---

## 2. 验证不能只做“新 detector vs 旧结果”

如果直接拿：

- 新 `_gpu.npz`
- 新 `_refineGpu.npz`
- 新生成 lagPat

去对比旧 lagPat，那么你同时混进了三类差异：

1. detector 差异
2. packing 差异
3. centroid / lag / rank 差异

这会让验证失去可解释性。

### 正确做法

验证必须拆成两个层级：

#### Level A: 算法同一性验证

固定老输入：

- 旧 `_gpu.npz`
- 旧 `_refineGpu.npz`
- 旧 `*_packedTimes.npy`
- 原始 EDF

只验证：

- picked channels 是否一致
- 新 spectrogram centroid 是否等价于老 `return_massCenterPat`
- 新 rank / bool 写出是否等价于旧 `lagPat`

这是**硬门槛**。不过这关，不能批量生成。

#### Level B: 端到端漂移验证

对 `gaolan` / `dongyiming` / `wangyiyang`，再用：

- 新 `_gpu.npz`
- 新 `_refineGpu.npz`

做 end-to-end 生成，与旧结果比较。

这一步不是要求“完全一致”，而是量化 detector source drift。

---

## 3. `lagPatRank` 不能偷偷现代化

老 lagPat 的脏包袱很明确：

- `lagPatRank` 不是“仅参与通道 rank”
- 它会给所有 picked channels 都写 rank
- 真正是否参与，要靠 `eventsBool` 屏蔽

这设计不优雅，但 Topic 1/2 现在就是吃这个 contract。

### 风险

如果回填时把 `lagPatRank` 改成新式语义：

- 非参与通道写 `-1`
- 或仅对参与通道排序

那么：

- `src/interictal_propagation.py`
- `src/event_periodicity.py`

虽然未必立刻报错，但统计语义会变。

### 合同

**本轮必须复刻旧 contract，不允许借机“修正”它。**

如果要改 contract，那是另一个 PR，必须全 cohort 重算，不准夹带私货。

---

## 4. `lagPatRaw` 必须保留旧语义：相对窗起点，不是 first-centroid 对齐

老代码的 `lagPatRaw` 是：

- spectrogram time centroid
- 相对 `packed window start`

不是：

- per-event min-subtracted relative lag

### 风险

如果误写成 first-centroid 对齐：

- Topic 1 absolute-lag validation 失真
- Topic 2 `packedTimes + lagPatRaw` 的 event-time reconstruction 失真
- 旧 plotting / summary 逻辑可能无声损坏

### 合同

- 写盘资产中的 `lagPatRaw` 必须保留旧语义
- 任何 min-subtracted 版本只允许在下游分析时临时构造，不得写回 legacy asset

---

## 5. alias collision 不是小问题

表面上看，`A1-A2 -> A1` 很自然。问题在于一旦 alias collision 出现，它就不是命名问题，而是：

- 你丢了一条双极对
- 并且这个丢弃是有统计后果的

### 允许的最小策略

- 先按 `events_count` 取高者

### 但必须额外做的事

每个 subject 必须输出 alias QC：

- 原始 bipolar channel 数
- alias 后 channel 数
- collision 数
- collision 明细
- 被丢弃通道及其 `events_count`

### 停止条件

满足任一条，停止该 subject，不进入正式回填：

1. alias collision 数 `> 0` 且涉及 picked channels
2. alias collision 导致被保留 / 被丢弃二者 `events_count` 比值 `< 1.2`
3. collision 发生在参考 subject (`gaolan` / `dongyiming` / `wangyiyang`)

原因很简单：这时“保留高计数者”已经不是无害近似，而是在改数据结构。

---

## 6. packed window 生成失败不能伪造空文件

老代码行为是：

- 如果一个 block 没有有效 packed windows，通常直接不写 `lagPat` / `packedTimes`

### 风险

如果新脚本为了整齐去写：

- 空的 `*_lagPat.npz`
- 空的 `*_packedTimes.npy`

会改变 downstream loader 的 block semantics。

### 合同

- 无有效 packed windows 的 block：**不写资产**
- 但必须写入 QC summary：
  - block 名
  - picked channel 数
  - candidate windows 数
  - filtered windows 数
  - drop reason

---

## 最大工程问题

## 1. 不能直接覆盖，必须原子写入

数据目录是 canonical root，不是试验场。

### 合同

每个 block 的写入顺序必须是：

1. 先生成临时文件
   - `*.tmp.npz`
   - `*.tmp.npy`
2. 完整校验 shape / keys / start_t
3. 若旧资产存在，先做 `.legacy_backup`
4. 最后原子 rename

任何异常都不允许留下：

- 半写文件
- 零字节文件
- 只写了一半的一对资产

---

## 2. 参数不能再散落在脚本里

真正影响科学语义的参数至少有这些：

- `pick_k`
- `pack_win_sec`
- `ext_ms=30`
- `chns_thr=0.5`
- `time_axis_hz=500`
- centroid power `=3`
- spectrogram config:
  - `50-300Hz`
  - `0.05s`
  - `80% overlap`
  - `gaussian sigma=1.5`

### 合同

- `pick_k` 和 `pack_win_sec` 进入 `config/subject_params.json`
- 其余 legacy-global 参数集中定义在回填脚本顶部或统一 helper 中
- 禁止在脚本不同分支里偷偷写不同默认值

---

## 3. `centroid_power` 必须显式锁死为 3

这是现成坑。

- 老代码：`spec ** 3`
- 新代码默认：`centroid_power=2.0`

### 合同

本轮回填脚本里必须显式传：

```python
centroid_power=3.0
```

不允许吃默认值。

如果后续有人把默认改回去，也不能影响本次回填。

---

## 4. `start_t` 必须双重校验

新 detector 结果里有 `start_time`，旧 lagPat 里有 `start_t`。

### 风险

如果时间基准有偏差：

- Topic 1 block ordering
- Topic 2 absolute event timeline
- seizure proximity / day-night downstream

都会被污染。

### 合同

每个 block 必须校验：

1. `gpu_npz["start_time"]`
2. EDF `meas_date.timestamp()`

二者差值必须 `< 1s`。

否则停止该 block，人工检查。

---

## 验证合同

## Phase A: 算法同一性硬验证

### 参考对象

- `gaolan`
- `dongyiming`
- `wangyiyang`
- 可再加 1 个老 cohort 中稳定 subject 作为补充，例如 `chengshuai`

### 输入固定为旧资产

- 数据目录旧 `_gpu.npz`
- 数据目录旧 `_refineGpu.npz`
- 旧 `*_packedTimes.npy`
- 原始 EDF

### 验证项 A1: picked channels

目标：证明新 wrapper 用老输入时，能恢复旧 `chnNames`

**通过标准**

- `picked_channels_new == lagPat["chnNames"]` exact match

**失败处理**

- 立即停止
- 不进入 centroid / lag 比较
- 先修 channel-selection / alias mapping

### 验证项 A2: packed windows

目标：证明新 `build_windows_from_detections` 与旧 packing 范式一致

**指标**

- `n_windows_exact_match`
- `median_abs_start_diff_ms`
- `p95_abs_start_diff_ms`
- overlap precision / recall

**通过标准**

- `n_windows` 完全一致，或差异只出现在边界 block 且有解释
- `median_abs_start_diff_ms <= 5`
- `p95_abs_start_diff_ms <= 20`
- precision / recall `>= 0.98`

**失败处理**

- 停止
- 不得进入批量回填
- 先修 packing 参数或边界处理

### 验证项 A3: eventsBool

目标：证明参与掩码范式一致

**指标**

- per-block Jaccard
- per-event exact-match rate

**通过标准**

- overall Jaccard `>= 0.98`
- exact-match rate `>= 0.95`

**失败处理**

- 停止
- 优先检查 overlap 判定边界和 alias mapping

### 验证项 A4: lagPatRaw

目标：证明 spectrogram centroid 语义一致

**比较方式**

- 仅在 `eventsBool == True` 的单元上比较

**指标**

- median absolute error
- p95 absolute error
- RMSE

**通过标准**

- median AE `<= 5 ms`
- p95 AE `<= 20 ms`
- RMSE `<= 10 ms`

**失败处理**

- 停止
- 优先检查：
  - `centroid_power`
  - frequency crop 边界
  - bandpass / resample path
  - window slicing 边界

### 验证项 A5: lagPatRank

目标：证明旧 rank contract 被正确复刻

**比较方式**

- 两层比较都要做：
  1. 旧 contract 下的整矩阵 exact match
  2. 仅参与通道的 rank match

**通过标准**

- exact-match rate `>= 0.95`
- participating-only match rate `>= 0.99`

**失败处理**

- 若 participating-only 通过但整矩阵不通过：
  - 说明 rank contract 没复刻好，停止
- 若 participating-only 都不过：
  - 说明 timing/order 本身错了，停止

---

## Phase B: 端到端漂移验证

### 参考对象

- `gaolan`
- `dongyiming`
- `wangyiyang`

### 输入改为新资产

- `results/hfo_detection/<subject>/*_gpu.npz`
- `results/hfo_detection/<subject>/_refineGpu.npz`

### 目的

量化“新 detector source + 旧 contract-style packing/lag”相对于旧全流程的漂移。

这一步不是要求完全一致，而是回答：

**漂移是否小到可以把新增 11 个 subject 放进 Topic 1/2 extended cohort？**

### 必做指标

对每个参考 subject 输出：

1. picked channel overlap
2. packed window count ratio
3. per-event `n_participating` distribution shift
4. lag span distribution shift
5. Topic 1 summary drift:
   - overall tau
   - cluster-aware tau
   - stable_k
6. Topic 2 summary drift:
   - group_n_events
   - IEI median
   - group PSD primary peak

### 允许结论

如果 end-to-end 漂移小：

- 可以推进 11-subject backfill
- 但 41-subject 结果仍标注为 extended cohort

如果 end-to-end 漂移大：

- 允许生成资产进入 staging
- **不允许**直接把 11 个 subject 并入正式 Topic 1/2 cohort summary

### 建议阈值

以下任一项在 `>=2/3` 参考 subject 上发生，即判定为“漂移过大”：

1. packed window count ratio 不在 `[0.67, 1.50]`
2. median `n_participating` shift `> 20%`
3. median lag span shift `> 20%`
4. Topic 1 `stable_k` 改变且 cluster fraction 结构明显重排
5. Topic 2 primary PSD peak 从“有”变“无”或从“无”变“有”

---

## 批量回填前置条件

只有当以下条件全部满足，才允许进入 11-subject 正式回填：

1. Phase A 全通过
2. Phase B 没有触发“漂移过大”停止条件
3. `pack_win_sec` 已完整进入配置
4. alias QC 机制已经实现
5. 临时写入 + 备份 + 原子 rename 已实现

少一条都不行。

---

## 批量回填后的 QC 合同

每个 subject 必须输出一个 QC JSON，至少包含：

- `subject`
- `n_blocks_total`
- `n_blocks_written`
- `n_blocks_skipped`
- `pick_k`
- `pack_win_sec`
- `n_raw_bipolar_channels`
- `n_alias_channels`
- `n_alias_collisions`
- `picked_channels`
- `event_count_per_block`
- `packed_window_count_per_block`
- `median_n_participating`
- `median_lag_span_ms`
- `write_status`

### subject-level 停止条件

满足任一条，该 subject 标记为失败，不并入 cohort：

1. `n_blocks_written == 0`
2. `n_alias_collisions > 0`
3. 超过 `30%` block 无有效 packed windows
4. `median_n_participating < 3`
5. `median_lag_span_ms <= 0`
6. 任一 block 写入后自检失败（keys / shape / start_t）

### cohort-level 停止条件

满足任一条，停止 Topic 1/2 全量重跑：

1. 11 个新增 subject 中失败 subject `>= 3`
2. 11 个新增 subject 的 `group_n_events` 中位数低于旧 10 个 Yuquan 的 `50%`
3. 新增 11 个 subject 中 `>= 4` 个在 Topic 1 或 Topic 2 中完全无可用事件

---

## 回填后的科学报告合同

正式对外或对内汇报时，必须同时给出三层结果：

1. **原 30-subject 主结果**
2. **41-subject extended-cohort sensitivity**
3. **reference 3-subject old-vs-new drift summary**

如果只报第 2 条，不报第 1 和第 3 条，那就是在洗数据来源差异。

---

## 推荐执行顺序

1. 补 `config/subject_params.json`
   - 加 `pack_win_sec`
2. 写 `validate_pack_against_legacy.py`
   - 先做 Phase A
3. 再做 Phase B
4. 写 `run_pack_group_events.py`
   - 仅当 A/B 都过
5. 批量回填 11 subject
6. 生成 subject QC + cohort QC
7. 只在 QC 通过后，才改 Topic 1/2 的 `YUQUAN_SUBJECTS`
8. 再重跑 Topic 1 / Topic 2

这顺序不能倒。

---

## 最终底线

这件事最容易犯的错有两个：

1. **工程上先批量写文件，再回头想怎么验证**
2. **科学上把“legacy-compatible asset contract”偷换成“legacy-equivalent cohort”**

第一个错会产出一堆垃圾文件。
第二个错会产出看起来很漂亮、实际上不能信的结论。

两种都不允许。
