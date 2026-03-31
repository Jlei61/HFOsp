# 玉泉24小时SEEG数据集结构分析

**分析日期**: 2026-01-12  
**数据路径**: `/mnt/yuquan_data/yuquan_24h_edf`

---

## 【核心判断】✅ 数据集结构清晰

这是一个**预处理完成的HFO检测数据集**，数据组织合理，每个文件职责明确。

---

## 1. 数据集概览

| 指标 | 数值 |
|------|------|
| 患者数量 | 21人 |
| EDF原始文件 | 260个 (约520小时记录) |
| GPU检测结果 | 176个 |
| 已检测HFO事件 | 约70万+ |
| 采样率 | 2000 Hz |
| 单文件时长 | 约2小时 |

**注意**: 有5个患者(pengzihang, songzishuo, zhangbichen, zhangkexuan, zhaochenxi, zhaojinrui, zhourongxuan)只有EDF原始文件，尚未完成GPU检测。

---

## 2. 目录结构

```
/mnt/yuquan_data/yuquan_24h_edf/
├── chengshuai/          # 患者1
│   ├── FC10477Q.edf                           # 原始SEEG信号 (2小时, 145通道)
│   ├── FC10477Q_gpu.npz                       # GPU检测的HFO事件
│   ├── FC10477Q_lagPat.npz                    # 滞后模式矩阵
│   ├── FC10477Q_lagPat_withFreqCent.npz       # 滞后模式+频率中心
│   ├── FC10477Q_packedTimes.npy               # 事件时间窗
│   ├── FC10477Q_packedTimes_withFreqCent.npy  # 事件时间窗+频率
│   ├── FC10477R.edf                           # 下一个2小时记录
│   ├── ...                                    # (共12个2小时记录)
│   ├── _refineGpu.npz                         # 患者级汇总
│   ├── hist_meanX.npz                         # 统计信息
│   ├── pick_chns.png                          # 筛选通道可视化
│   └── refine_hist.png                        # 统计直方图
├── chenziyang/          # 患者2
├── ...
└── zhourongxuan/        # 患者21
```

---

## 3. 文件类型详解

### 3.1 原始数据: `*.edf`

**内容**: SEEG侵入式脑电信号原始记录

```python
# 示例: FC10477Q.edf
采样率: 2000 Hz
通道数: 145个 (包含SEEG电极 + ECG/EMG)
时长: 7200秒 (2小时)
开始时间: 2019-09-20 22:19:56 (Unix: 1569017996.0)

通道命名:
- EEG A1-Ref, A2-Ref, ...    # 参考电极
- POL A3, A4, A5, ...        # 极性记录 (主要SEEG通道)
- POL K3, K5, K6, ...        # 不同电极串
- POL ECG, EMG1, EMG2        # 生理信号
```

---

### 3.2 HFO检测结果: `*_gpu.npz`

**内容**: GPU加速的高频振荡(HFO)事件检测结果

```python
键名: ['whole_dets', 'chns_names', 'events_count', 'start_time']

whole_dets:   # 每个通道检测到的事件时间段
  - shape: (120,)  # 120个有效通道
  - dtype: object
  - 内容: 每个元素是 [[start1, end1], [start2, end2], ...] (秒)
  - 示例: [[282.34875, 282.42375], [343.85875, 343.91125], ...]

chns_names:   # 通道名列表
  - shape: (120,)
  - dtype: <U3
  - 示例: ['A1', 'A2', 'A3', ..., 'K10']

events_count: # 每个通道的事件数量
  - shape: (120,)
  - dtype: int64
  - 示例: [13, 1, 1, 2, ..., 11528, 9801, ...]
  - 注意: 某些通道事件极多 (>10000), 可能是伪迹

start_time:   # Unix时间戳
  - scalar: 1569017996.0
```

**数据特点**:
- 只保留了**120个有效通道** (原始145个)
- 移除了参考电极和生理信号通道
- 事件数量在通道间差异极大: 0 ~ 11528

---

### 3.3 滞后模式矩阵: `*_lagPat.npz` / `*_lagPat_withFreqCent.npz`

**内容**: 用于分析事件传播模式的时空矩阵

```python
# *_lagPat.npz (基础版本)
键名: ['lagPatRaw', 'lagPatRank', 'eventsBool', 'chnNames', 'start_t']

lagPatRaw:    # 原始滞后值
  - shape: (8, 2601)  # 8个筛选通道 × 2601个事件
  - dtype: float64

lagPatRank:   # 滞后值排名
  - shape: (8, 2601)
  - dtype: int64

eventsBool:   # 事件布尔掩码
  - shape: (8, 2601)
  - dtype: float64

chnNames:     # 筛选后的通道
  - shape: (8,)
  - 示例: ['E11', 'K3', 'K5', 'K6', 'K7', 'K8', 'K9', 'K10']

# *_lagPat_withFreqCent.npz (增强版本)
额外包含:
lagPatFreq:   # 频率信息
  - shape: (8, 2601)
  - dtype: float64
```

**关键洞察**:
- 从120个通道**筛选出8个核心通道** → 降维90%
- 2601个事件是跨所有通道的**时间对齐事件池**
- `lagPatRaw[i, j]` 是每个事件内的“时间标量”，但其绝对值在事件间可能处于**拼接/累积时间轴**上（因此跨事件绝对值不直接可比）
  - 若要比较“传播时延”，应在**每个事件内部**做参考系统一：例如 `lag_rel = lagPatRaw - min(lagPatRaw)`（或对齐到最早通道）
  - `lagPatRank` 对应每个事件内的通道先后顺序（对 ms 级抖动非常敏感）
- `lagPatRaw` / `lagPatRank` 会对**每个 picked channel、每个 packed window**都给出一个值；真正告诉你“这个通道在该群体事件里是否参与”的，是同文件里的 `eventsBool`

### 3.3.1 `eventsBool` 的严格语义

`eventsBool[ch, ev] = 1` 的条件很朴素：

- 通道 `ch` 在群体事件窗口 `ev` 对应的 `packedTimes[ev]` 里
- 只要有任意一段独立 HFO 检测事件与该窗口重叠
- 就记为 1，否则为 0

这意味着：

- `eventsBool` 是**参与掩码**，不是强度、不是置信度、不是排序
- `lagPatRaw`/`lagPatRank` 本身**不会自动屏蔽**未参与通道
- 所以下游脚本若不用 `eventsBool` 过滤，就等于把“背景时频质心”也当成真实传播成员

---

### 3.4 群体事件时间窗: `*_packedTimes.npy`

**内容**: 由 picked channels 的独立 HFO 事件打包得到的群体事件窗口，不是“所有检测到的单通道事件边界”

```python
shape: (2601, 2)  # 2601个事件 × [start, end]
dtype: float64

示例:
[[  0.342,   0.842],   # 事件1: 0.342s ~ 0.842s (持续500ms)
 [  3.194,   3.694],   # 事件2
 [  5.660,   6.160],   # 事件3
 ...
 [7199.xxx, 7199.xxx]] # 最后一个事件
```

**老代码生成逻辑**:
1. 从 `*_gpu.npz` 的 `whole_dets` 出发，每个 picked channel 的独立 HFO 事件先左右各扩展 `ext=30ms`
2. 把这些扩展事件投影到统一时间轴，统计每个时刻被多少 picked channels 覆盖
3. 取覆盖通道数 `>= chnsThr * n_picked_channels` 的连续区间，老代码默认 `chnsThr=0.5`
4. 对每个过阈值区间取中心点，再扩成固定长度窗口 `packWinLen`
5. 若相邻两个候选窗口重叠，`pick_noOverlap_timeRanges()` 会把两个都删掉

**特点**:
- `packedTimes` 代表的是“群体事件定义”，不是原始 detector 逐通道事件
- 事件窗长度在不同记录中可能不同（常见 0.5s，也存在 0.3s），不要在代码里硬编码；应从 `packedTimes[:,1]-packedTimes[:,0]` 推断
- 时间相对于记录开始时刻 (start_time)
- 后续 `eventsBool`、`lagPatRaw`、`lagPatRank` 都建立在这些窗口上；这一步如果混入 spike/噪声事件，或把本应分开的事件错并，后续 pattern 会被系统性带偏

---

### 3.5 患者级汇总: `_refineGpu.npz`

**内容**: 患者级 `refine` 后的通道计数汇总，不是简单的原始 GPU 事件数求和

```python
键名: ['events_count', 'chns_names']

events_count: # 所有记录文件在 refine 后重新统计的事件数
  - shape: (120,)
  - dtype: int64

chns_names:   # 通道名
  - shape: (120,)
```

**关键区别**:

- `*_gpu.npz` 的 `events_count`:
  - 是单个 2h 记录里，每个通道原始 HFO 检测事件数
- `_refineGpu.npz` 的 `events_count`:
  - 不是把所有 `*_gpu.npz` 的计数直接相加
  - 老代码实际流程是：
    1. 先把所有 `*_gpu.npz` 的原始 `events_count` 跨记录求和
    2. 用 `mean + pickChn_thresh * std` 选一批 provisional channels
    3. 在每个 2h 记录上，仅用这批通道生成 packed group events
    4. 再用这些 packed windows 对 **所有通道** 做一次 `reHist_events_byPacking`
    5. 把这一步的 recount 结果跨记录求和，写入 `_refineGpu.npz`

结论：`_refineGpu.npz` 是 **“同步性约束后的患者级 recount”**，不是简单的 GPU 汇总。

---

### 3.6 统计信息: `hist_meanX.npz`

**内容**: 通道筛选和质量控制的统计量

```python
键名: ['hist_meanX', 'pick_chns']

hist_meanX:   # 基于 rank 直方图的模板位置统计，不是原始 lag 时间均值
  - shape: (8,)
  - dtype: float64
  - 示例: [6.17, 6.19, 3.99, 1.00, 3.01, 3.01, 5.03, 6.14]

pick_chns:    # 筛选出的8个核心通道
  - shape: (8,)
  - 示例: ['E11', 'K3', 'K5', 'K6', 'K7', 'K8', 'K9', 'K10']
```

---

## 4. 数据处理流程推断

```
[原始EDF 2000Hz, 145通道]
         ↓
   双极参考 + 去掉最外触点
         ↓
    重采样到 800Hz
         ↓
    带通滤波 (80-250Hz)
         ↓
    GPU加速HFO检测
      - rel/abs 双阈值
      - min_gap 合并
      - 50-200ms 时长过滤
      - side-band 去噪
         ↓
   [*_gpu.npz: 单通道独立候选事件]
         ↓
    跨记录原始 events_count 汇总
         ↓
    provisional channel pick (mean + k*std)
         ↓
    用 provisional channels 构建 packed group events
      - ext=30ms
      - 覆盖通道数 >= 50%
      - 取中心点扩成固定窗长
      - 删除重叠窗口
         ↓
    对所有通道做 refine recount
         ↓
   [_refineGpu.npz: 患者级 refine 后 recount]
         ↓
    再次通道筛选
         ↓
    计算群体事件参与矩阵 `eventsBool`
         ↓
    在每个 packed window 内计算 spectrogram 质心
         ↓
   [*_lagPat.npz: 8×2601 时频质心/排序矩阵]
         ↓
    频率中心计算
         ↓
   [*_lagPat_withFreqCent.npz]
```

---

## 5. 关键数据关系

```
whole_dets (120通道 × 变长事件列表)
    ↓ provisional channel pick
packedTimes (2601个统一事件 × [start, end])
    ↓ eventsBool: 哪些通道在该群体窗内参与
lagPatRaw (8通道 × 2601事件, spectrogram时间质心)
    ↓ 逐事件排序
lagPatRank (8通道 × 2601事件)
```

---

## 6. 老链路输入/输出合同表

| 资产 | 谁生成 | 上游依赖 | 下游消费者 | 语义合同 | 常见误读 |
| --- | --- | --- | --- | --- | --- |
| `<record>_gpu.npz` | `p16_cuda_24h_bipolar.py` | EDF、双极参考、800Hz重采样、80-250Hz检测包络 | `p16_refine_chns_bySyn.py`、`p16_packGroupEvents*.py` | `whole_dets` 是**单通道独立候选事件**；`events_count` 是该 2h 文件内的原始检测计数 | 把它当成“群体事件”或把 `events_count` 当成最终临床有效事件数 |
| `_refineGpu.npz` | `p16_refine_chns_bySyn.py` | 全部 `*_gpu.npz`、provisional picked channels、packed recount | `p16_packGroupEvents*_refine*.py`、当前 `core_channels.source=_refineGpu` 路径 | `events_count` 是**群体窗口约束后的 recount**，不是原始检测计数累加 | 把它当成“所有 GPU 文件 events_count 直接求和” |
| `<record>_packedTimes.npy` | `p16_packGroupEvents*.py` via `hfo_net.py::get_packedEventsTimes_overThresh` | 当前文件的 `whole_dets` + 患者级 `_refineGpu.npz` 选出的 picked channels | `get_packedEvents_bool()`、`return_seg_splitContiHigh()`、各类 `plotting_fig3/4/7/8*` | 每一行是**一个群体事件窗口**；窗口中心来自“过半通道覆盖区间”的中心点，再扩成固定窗长 | 把它当成单通道事件边界，或假设它独立于 picked channels 定义 |
| `<record>_lagPat.npz` | `p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool.py` | `packedTimes`、picked channels 高频信号、`eventsBool` | `merge24h_lagPat.py`、MI/variance/region/network 统计、论文图脚本 | `lagPatRaw` 是**群体窗口内的时频质心时间**；`lagPatRank` 是逐事件排序；`eventsBool` 标记参与通道 | 把 `lagPatRaw` 当成原始波形峰值、detector 起点，或把跨事件绝对值直接比较 |
| `<record>_lagPat_withFreqCent.npz` | `p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter.py` | 同上，外加频率质心计算 | `plotting_fig4_durDelayStats.py`、`plotting_fig4_seqsTemporalStats.py`、`plotting_fig8_HigherAndEarly.py` | 在 `lagPat` 基础上增加 `lagPatFreq`，用于时间-频率联合统计 | 以为它和旧版 `lagPat.npz` 完全同名同消费者；很多老图脚本其实还读旧名字 |
| `hist_meanX.npz` | `p16_merge24h_lagPat.py` | 全部 `lagPatRank` 拼接后的 rank 直方图 | 当前 `run_pipeline.py`、`visualize_run.py`、部分 core-channel 选择链路 | `hist_meanX` 是**24h rank 模板位置统计**，用来概括“谁更常靠前” | 把它当成原始 lag 时间均值或单事件统计 |

### 6.1 最小依赖链

```
EDF
  -> *_gpu.npz
  -> _refineGpu.npz
  -> *_packedTimes.npy
  -> *_lagPat.npz / *_lagPat_withFreqCent.npz
  -> hist_meanX.npz
```

### 6.2 真正该盯的断点

- `*_gpu.npz` 脏：说明单通道独立候选事件就不对，后面全错。
- `_refineGpu.npz` 脏：说明 provisional channels 或 recount 逻辑错了，packed 定义会漂。
- `*_packedTimes.npy` 脏：说明群体事件定义错了，`eventsBool` 和 `lagPat*` 一起连坐。
- `lagPatRaw` 漂：先怀疑 `packedTimes` 和 spectrogram 边界，不要先怪 rank 统计。

### 6.3 2026-03-31 EDF 起跑回归更新

这轮验证改了一个思路：不再只拿“现有老资产”打“新资产”，而是让**新老代码都从 EDF 起跑**，先做一个可完成的 `500s` smoke/regression，再看哪一层先漂。

#### 6.3.1 已证实的事实

- `cuda_env` 原先**缺少 `cusignal`**；现已通过 `conda install -n cuda_env -c rapidsai -c conda-forge cusignal` 补齐并验证导入成功。
- `resample_sfreq` 必须在配置里**显式写成 `800`** 才能对齐老代码。
  - 在一次验证中，`analysis.resample_sfreq: auto` 实际跑成了 `1000Hz`
  - 这不是老 yuquan 链路；所以凡是“从 EDF 起跑复现实验”的配置，都不应再用 `auto`
- 旧代码在当前软件栈上，从 EDF 起跑需要额外兼容层：
  - `mne.io.read_raw_edf(..., encoding='latin1')`，否则会被 EDF 注释通道的编码炸掉
  - 新版 `numpy` 下，老脚本直接 `np.savez(whole_dets=<ragged list>)` 会报错；这属于运行时兼容问题，不是算法差异
- 这些都属于**环境/依赖兼容问题**，不应误判成“新老算法逻辑不同”。

#### 6.3.2 500s EDF smoke/regression 结果

在 `chengshuai / FC10477Q` 上，使用 `500s` 截断做公平比较：

- 老代码（EDF → gpu → packed → lagPat）重跑成功：
  - `191` 个 packed windows
  - picked channels = `E11, K3, K5, K6, K7, K8, K9, K10`
- 新代码（EDF → pipeline）重跑成功：
  - `190` 个 packed windows
  - picked channels = `E11, K3, K5, K6, K7, K8, K9, K10`
- 两边 channel set 已经**完全一致**
- `packedTimes` 还没完全一致，但已经不是乱飞：
  - `178 / 190` 个新窗口能在 `5ms` 容差内找到老窗口对应项
  - 这些已匹配窗口的平均边界误差约 `0.86ms`

结论：

- 当前主要问题已经不是 “picked channels 完全错了”
- 真正剩下的是 **group window 边界仍有局部漂移**，这会继续传染到 `lagPatRaw / lagPatRank`

#### 6.3.3 一个必须正视的上游差异

这轮 EDF 起跑验证里，还抓到一个比 `190 vs 191` 更硬的信号：

- 老代码生成的 `*_gpu.npz` 是 `120` 个 bipolar 通道
- 新代码这次预处理日志显示：
  - `140 -> 130 bipolar channels`
  - `drop_shaft_edges` 后 `130 -> 110`

这说明：

- 新代码当前默认的 `drop_shaft_edges` 规则，**很可能不等价于**老代码真实的 `return_valid_chan_index + subs_drop_info` 合同
- 如果上游通道宇宙都不一样，后续 `packedTimes` 和 `lagPat*` 就不可能完全对齐

当前最值得追的根因，不是再去死抠某一个 `lagPatRaw` 小数点，而是先钉死：

- 老 `120` 通道名单到底是什么
- 新链当前少掉的 `10` 个通道是谁
- 它们是被 EDF 读入、有效通道筛选、双极映射，还是 edge-drop 规则删掉的

### 6.3 `eventsBool` 的消费者风险分级

- 明确正确使用 `eventsBool` 的代表：
  - `plotting_fig4_pairedDelay.py`
  - `plotting_fig4_durDelayStats*.py`
  - 这些脚本会先用 `eventsBool==0` 屏蔽未参与通道，再算传播时长、相邻延迟、频率中心等
- 部分使用 `eventsBool`，但模板统计仍混入未屏蔽 rank 的代表：
  - `plotting_fig7_networkDemo.py`
  - 它在 `bool_hist`/`rank1_count` 上用了 `eventsBool`，但 `hist_meanX` 风格的模板位置仍然来自未过滤的 `tmpHist`
- 基本忽略 `eventsBool`、直接消费 `lagPatRank` 的代表：
  - `p16_merge24h_lagPat.py`
  - `plotting_fig5_lagPat_variance.py`
  - `plotting_fig3_pickExamples.py`
  - 这类脚本更接近“rank 模板统计”，不是严格的“参与通道传播统计”

结论：

- 若任务是算传播时长、频率中心、参与通道数，必须尊重 `eventsBool`
- 若任务是复现老代码里的 `hist_meanX` / rank 模板链路，就得接受它带着“未参与通道也被硬排序”的历史包袱

### 6.4 模板链审计: `hist_meanX / MI / variance`

| 链路 | 代表脚本 | 实际输入 | 是否先用 `eventsBool` 屏蔽未参与通道 | 结论 |
| --- | --- | --- | --- | --- |
| `hist_meanX` | `p16_merge24h_lagPat.py` | 拼接后的 `whole_cat = concat(lagPatRank)` | 否 | **建立在未屏蔽 rank 上** |
| MI 模板 | `p16_quantify_MI.py` | `whole_cat = concat(lagPatRank)` | 否 | **建立在未屏蔽 rank 上** |
| 24h MI 变化 | `p16_MI_24hChange.py` / `p16_MI_24hChange_entropy.py` | 每个 2h 片段的 `lagPatRank`，模板来自 `return_hist_mean_rank()` | 否 | **建立在未屏蔽 rank 上** |
| variance / rank 分布 | `plotting_fig5_lagPat_variance.py` | 拼接后的 `whole_cat = concat(lagPatRank)` | 否 | **建立在未屏蔽 rank 上** |
| delay / duration / freq | `plotting_fig4_pairedDelay.py` / `plotting_fig4_durDelayStats*.py` | `lagPatRaw`/`lagPatFreq` + `eventsBool` | 是 | 这类统计更接近“真实参与通道”分析 |

审计说明：

- `p16_merge24h_lagPat.py` 虽然把 `eventsBool` 也读进来了，但生成 `hist_meanX` 时实际对 `whole_cat[ci]` 直接做直方图，没有先按 `eventsBool` 过滤。
- `p16_quantify_MI.py` 和 `p16_MI_24hChange*.py` 的模板函数 `return_hist_mean_rank()` 都直接对 `whole_cat_matrix[ci]` 做 `np.histogram`，再把这个模板与每个事件的整列 `lagPatRank` 比较。
- `plotting_fig5_lagPat_variance.py` 整条方差/模板分布链直接围绕 `whole_cat = concat(lagPatRank)` 展开，没有参与掩码。
- 因此，这条模板链更准确的名字应该是：**老式未屏蔽 rank 模板链**。

能支持什么结论：

- “在老实现定义下，某些通道在 rank 模板里更常靠前”
- “老实现定义下，群体事件排序模式在 24h 内具有一定重复性”
- “老实现定义下，模板与事件的 MI 高于随机置换”

不能过度宣称什么：

- 不能直接把它说成“只有真实参与通道的传播模板稳定”
- 不能把 `hist_meanX` 直接解释为生理上的绝对先导时延均值
- 不能把 MI/variance 直接解释为严格参与成员的传播稳定性
