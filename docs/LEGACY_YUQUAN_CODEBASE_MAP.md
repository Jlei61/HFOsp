# Legacy Yuquan Codebase Map

> 目的：把老代码、当前 `HFOsp`、以及 `/mnt/yuquan_data` 中间产物的关系钉死，减少以后追图、追结果、追生成链路时的盲搜。
>
> 范围：只针对玉泉 `yuquan_24h` 主线，不讨论其他数据集的旁支脚本。

---

## 1. 先认清三块地盘

### 1.1 权威数据根

- `/mnt/yuquan_data/yuquan_24h_edf`
- 这里是**权威中间产物仓**。论文图、老结果、现有复现实验，最终都要落回这里的病人目录。
- 典型文件：
  - `<record>.edf`
  - `<record>_gpu.npz`
  - `<record>_packedTimes.npy`
  - `<record>_lagPat.npz`
  - `<record>_lagPat_withFreqCent.npz`
  - `_refineGpu.npz`
  - `hist_meanX.npz`

### 1.2 当前工程

- `HFOsp/`
- 这是**可维护主线**。模块边界已经比老代码清楚得多：
  - `src/preprocessing.py`
  - `src/hfo_detector.py`
  - `src/group_event_analysis.py`
  - `src/network_analysis.py`
  - `scripts/run_pipeline.py`

### 1.3 老工程

- 老树真实路径：`/home/honglab/leijiaxin/HFOsp/ReplayIED`
- 历史探索里出现的老树名字是 `ReplayIED/`，不是现在仓库内的 `HFOsp/`
- 已确认的历史逻辑根有两条：
  - `ReplayIED/inter_events/yuquan_24h_perPatientAnalysis_dropRef/`
  - `ReplayIED/inter_events/epilepsiae_interictal/`
- 当前工作区里不一定挂载了这棵树。**如果看不到，就停下来问用户老工程实际路径，不要脑补。**

---

## 2. 老代码别按目录看，按逻辑看

老代码的问题不是算法，而是组织方式烂。真正该看的只有五层。

| 逻辑层 | 老代码主入口 | 产物/职责 | 当前 HFOsp 对应位置 |
| --- | --- | --- | --- |
| 预处理 + 检测 | `p16_cuda_24h_bipolar.py` | EDF 读取、双极参考、滤波、HFO 检测，写 `<record>_gpu.npz` | `src/preprocessing.py` + `src/hfo_detector.py` |
| 同步性约束 refine | `p16_refine_chns_bySyn.py` | 跨记录汇总计数、挑 provisional channels、packed recount，写 `_refineGpu.npz` | `src/group_event_analysis.py` 里的 `legacy_refine_*` |
| 群体事件打包 + lag/freq | `p16_packGroupEvents*.py`、`hfo_net.py` | 生成 `packedTimes`、spectrogram 质心、`lagPatRaw/Rank/Freq` | `build_windows_from_*`、`compute_centroid_matrix_spectrogram()`、`lag_rank_from_centroids()` |
| 合并/统计/网络 | `p16_merge24h_lagPat*.py`、`diffnet_prepareTXT.py`、`p16_lagPat_diffusionNet_comparison.py` | 24h 合并、diffnet/cascade、下游统计 | `src/network_analysis.py` + 新脚本 |
| 论文画图 | `plotting_fig4_*`、`plotting_fig5_*`、`plotting_fig8_*` | 只读中间产物出图 | `src/visualization.py` + `scripts/visualize_run.py`，但**不能假设图号一一对应** |

---

## 3. 以中间文件为中心溯源

别从“图名”找代码，先从中间文件找生成链路。真正稳定的是数据，不是脚本名。

| 中间文件 | 老代码生成阶段 | 关键老脚本 | 当前 HFOsp 对应/端口 | 备注 |
| --- | --- | --- | --- | --- |
| `<record>_gpu.npz` | 单通道 HFO 检测 | `p16_cuda_24h_bipolar.py` | `src/preprocessing.py` + `src/hfo_detector.py` | 键通常是 `whole_dets / chns_names / events_count / start_time` |
| `_refineGpu.npz` | subject 级同步性约束 recount | `p16_refine_chns_bySyn.py` | `legacy_refine_counts_from_detection_sets()`、`_legacy_rehist_events_by_packing()` | 不是简单加总原始 `events_count` |
| `<record>_packedTimes.npy` | 群体事件窗口打包 | `hfo_net.py::get_packedEventsTimes_overThresh`，由 `p16_packGroupEvents*.py` 驱动 | `build_windows_from_detections()` / `build_windows_from_packed_times()` | 真正的“群体事件定义”在这里 |
| `<record>_lagPat.npz` | 质心与时序 | `p16_packGroupEvents*.py::return_massCenterPat`，后续可能再由 `p16_merge24h_lagPat*.py` 汇总 | `compute_centroid_matrix_spectrogram()` + `lag_rank_from_centroids()` | 关键键：`lagPatRaw / lagPatRank / eventsBool / chnNames / start_t` |
| `<record>_lagPat_withFreqCent.npz` | 质心 + 频率中心 | `p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter.py` | `groupAnalysis` / `lag_freq` 导出链 | 这是很多论文图的直接输入 |
| `hist_meanX.npz` | 24h lag 模式汇总后得到的核心通道结果 | `p16_merge24h_lagPat.py` | `core_channels.source=hist_meanX` 的消费者在 `scripts/run_pipeline.py` | 写入键为 `hist_meanX / pick_chns` |

补充合同：

- `<record>_gpu.npz`：单通道独立候选事件，不是群体事件。
- `_refineGpu.npz`：群体窗口约束后的 recount，不是原始 `events_count` 跨文件直加。
- `<record>_packedTimes.npy`：群体事件定义本身；这一步脏了，后面 `eventsBool / lagPat*` 全部连坐。
- `<record>_lagPat.npz`：`lagPatRaw` 是时频质心时间，不是原始峰值或 detector 起点。
- `hist_meanX.npz`：24h rank 模板位置统计，不是 raw lag 时间均值。

---

## 4. 关键逻辑别再瞎找

### 4.1 `_gpu.npz`

看老代码：

- `p16_cuda_24h_bipolar.py`
- 相关工具函数通常在 `highEvents_yuquan0910_utils.py`

看新代码：

- `src/preprocessing.py`
- `src/hfo_detector.py`

### 4.2 `_refineGpu.npz`

看老代码：

- `p16_refine_chns_bySyn.py`

看新代码：

- `src/group_event_analysis.py`
  - `legacy_refine_channels_from_detections()`
  - `legacy_refine_counts_from_detection_sets()`
  - `_legacy_rehist_events_by_packing()`

这部分已经明确是对老 `refine` 逻辑的端口，不要再去别的地方猜。

### 4.3 `packedTimes`

看老代码：

- `hfo_net.py::get_packedEventsTimes_overThresh`
- `pick_noOverlap_timeRanges`
- 调用入口在 `p16_packGroupEvents*.py`

看新代码：

- `build_windows_from_detections()`
- `build_windows_from_packed_times()`

### 4.4 `lagPat`

看老代码：

- `p16_packGroupEvents*.py::return_massCenterPat`
- `p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter.py`
- `p16_merge24h_lagPat*.py`

看新代码：

- `compute_centroid_matrix_spectrogram()`
- `compute_centroid_matrix_from_envelope_cache()`
- `lag_rank_from_centroids()`

### 4.5 论文图

画图脚本多数只消费中间产物，不负责核心计算。找论文图时的顺序必须是：

1. 先认出图使用的是哪类资产：`gpu`、`packedTimes`、`lagPat`、`lagPatFreq`、网络结果
2. 再找对应的老画图脚本
3. 最后才看当前 `visualization.py` 有没有新版本替代

**不要反过来。**

---

## 5. 老代码项目结构，按“处理”和“画图”分

### 5.1 核心处理逻辑

这是必须优先读的脚本集合：

1. `p16_cuda_24h_bipolar.py`
2. `p16_refine_chns_bySyn.py`
3. `hfo_net.py`
4. `p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter.py`
5. `p16_merge24h_lagPat*.py`
6. `diffnet_prepareTXT.py`
7. `p16_lagPat_diffusionNet_comparison.py`
8. `highEvents_yuquan0910_utils.py`
9. `p16_subs_info.py`

### 5.2 画图逻辑

这些脚本默认只读产物，别把它们误认为“生成逻辑入口”：

- `plotting_fig4_pairedDelay.py`
- `plotting_fig4_durDelayStats.py`
- `plotting_fig4_regionHist.py`
- `plotting_fig4_regionHist_median.py`
- `plotting_fig4_seqsTemporalStats.py`
- `plotting_fig5_lagPat_variance.py`
- `plotting_fig8_seizureRelated_synchron.py`
- `plotting_fig8_resortConnMatrix_withKuramoto.py`

### 5.3 老树里的官方口径

`ReplayIED/inter_events/yuquan_24h_perPatientAnalysis_dropRef/readme` 里自己写的主线是：

1. `cuda_24h_bipolar.py` 检测高频
2. `refine_chns_bySyn.py` 同步性约束
3. `packGroupEvents_per2h_showSpecs_bipolar_refine_bool.py` 计算群体活动时序
4. `merge24h_lagPat.py` 24h 传播模式

这不是漂亮架构，但至少说明我们现在整理出的主链路没有跑偏。

---

## 6. 当前 repo 里最值得先读的文件

如果目标是“把老结果在新系统里接住”，优先顺序如下：

1. `docs/OLD_vs_NEW_algorithm_comparison.md`
2. `docs/yuquan_24h_dataset_structure.md`
3. `config/default.yaml`
4. `scripts/run_pipeline.py`
5. `src/group_event_analysis.py`
6. `src/preprocessing.py`
7. `src/hfo_detector.py`
8. `src/network_analysis.py`

理由：

- 前两个 docs 已经把老逻辑的大框架和资产结构压缩好了
- `config/default.yaml` 决定当前默认到底是不是 legacy 对齐路径
- `scripts/run_pipeline.py` 决定入口如何选用旧产物或重算
- `src/group_event_analysis.py` 是当前最接近老 `refine/packing/lagPat` 的核心

---

## 7. 追图时的标准动作

### 7.1 先问自己三个问题

1. 这张图到底在读哪类资产？
2. 这类资产的生成脚本是老代码还是新代码？
3. 现在工作区里有没有那棵老工程树？

### 7.2 标准排查顺序

1. 在 `/mnt/yuquan_data/yuquan_24h_edf/<subject>/` 确认相关产物是否存在
2. 从产物名回到上面的“中间文件溯源表”
3. 先看 `docs/OLD_vs_NEW_algorithm_comparison.md` 是否已记录老脚本名
4. 再去老工程里找脚本
5. 只有当老脚本缺失时，才看新代码有没有等价实现

---

## 8. 已知坑

### 8.1 `fig7`、`fig8` 不等于论文 Figure 7/8

当前 repo 的 `scripts/visualize_run.py` 里有 `fig7`、`fig8` 变量，但那只是脚本内部编号。它们不代表论文图号。

### 8.2 `hist_meanX.npz` 不是“可有可无的小缓存”

它是老代码核心通道选择的结果入口之一。当前 `run_pipeline.py` 和 `visualize_run.py` 都支持直接消费它。

### 8.3 `_refineGpu.npz` 不是原始 GPU 汇总

它是经过同步性约束后的 recount 结果。把它当原始事件数会把后续通道筛选逻辑搞歪。

### 8.4 `withFreqCenter` 路径会造成文件名漂移

`p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter.py` 写的是：

- `<record>_lagPat_withFreqCent.npz`
- `<record>_packedTimes_withFreqCent.npy`

而且同文件里原来的 `<record>_packedTimes.npy` 写出被注释掉了。很多 `plotting_fig*` 还在读旧名字，所以追图时必须确认当时跑的是哪一版 pack 脚本。

### 8.5 如果 `ReplayIED` 不在工作区，agent 必须停

这不是“再搜搜看”的问题。没有老树真实路径，就不要假装自己已经完成溯源。

---

## 9. 最短结论

- 对老项目，真正该维护的是**中间文件链路**，不是那堆散乱脚本本身。
- 对当前项目，真正要做的是把 `HFOsp` 明确成“可维护计算主线”，再让它能读懂老产物。
- 以后任何人或 agent 要追图、追结果、追生成逻辑，都必须先从 `/mnt/yuquan_data/yuquan_24h_edf` 开始，而不是先搜 `plotting_fig*`。
