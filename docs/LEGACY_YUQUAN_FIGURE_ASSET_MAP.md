# Legacy Yuquan Figure Asset Map

> 目的：把 `ReplayIED/inter_events/yuquan_24h_perPatientAnalysis_dropRef/` 里的论文图脚本，按“它读了什么资产、这些资产是谁生成的、当前 `HFOsp` 有没有替代”整理出来。
>
> 这份文档不追求列出每一个脚本的所有细枝末节，只保留能显著减少溯源时间的关键信息。

---

## 1. 先记住三条铁律

1. 老图脚本大多是**只读中间文件**，不是核心生成入口。
2. 追图时先认资产，再认脚本，最后才认图号。
3. `withFreqCenter` 这条线会改文件名，很多图脚本和产物名并不一致。

---

## 2. 老图最常读的资产类型

| 资产 | 典型内容 | 典型生产者 |
| --- | --- | --- |
| `<record>_gpu.npz` | 单通道 HFO 检测结果 | `p16_cuda_24h_bipolar.py` |
| `_refineGpu.npz` | patient 级同步性约束 recount | `p16_refine_chns_bySyn.py` |
| `<record>_packedTimes.npy` | 群体事件窗口 | `p16_packGroupEvents*.py` + `hfo_net.py` |
| `<record>_lagPat.npz` | lag 时序矩阵 | `p16_packGroupEvents*.py` |
| `<record>_lagPat_withFreqCent.npz` | lag + 频率中心 | `p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter.py` |
| `<record>_packedTimes_withFreqCent.npy` | 频率中心版本的 packed windows | 同上 |
| `hist_meanX.npz` | 24h 汇总后的核心通道结果 | `p16_merge24h_lagPat.py` |
| `*.mat` / `A_hat` | diffusion network / netRate 外部结果 | `prepareFor_netRate.py` 及外部 netrate 工具链 |
| `*_fdrP.npz` / `tmpRes_fig2_auc_yuquan.npz` / `machingIndex/*.npz` | 二次统计缓存 | 各自对应的 `plotting_fig*` 计算脚本 |

---

## 3. 高风险漂移

### 3.1 `withFreqCenter` 会改主产物名

`p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter.py` 写的是：

- `<record>_lagPat_withFreqCent.npz`
- `<record>_packedTimes_withFreqCent.npy`

而且旧的 `<record>_packedTimes.npy` 写出在脚本里被注释掉了。

结论：

- 很多老图脚本还在读 `<record>_lagPat.npz` + `<record>_packedTimes.npy`
- 你不能假设“跑了 withFreqCenter 版 pack 脚本”以后，所有旧图都会自动可复现

### 3.2 `_refineGpu.npz` 可能是病人目录级别，不是逐记录

有些脚本读的是病人目录下的 `_refineGpu.npz`，不是 `<record>_refineGpu.npz` 这种命名习惯。

### 3.3 `hist_meanX.npz` 很多脚本不直接读

它是核心通道选择结果，但很多图脚本并不直接加载它，而是通过先前合并/挑样本流程隐式使用。

---

## 4. 分组对照表

### 4.1 Fig2 类

| 代表脚本 | 主要用途 | 读取资产 | 上游生成脚本 | 当前 HFOsp 替代 |
| --- | --- | --- | --- | --- |
| `plotting_fig2_dataDemoNmethods.py` | 多方法示意 | `*_gpu.npz`, `*_packedTimes.npy` | `p16_cuda_24h_bipolar.py`, `p16_packGroupEvents*.py` | 有底层计算，无同图脚本 |
| `plotting_fig2_SEEGdataDemo_chap2.py` | SEEG + packed event 示例 | `*_gpu.npz`, `*_packedTimes.npy` | 同上 | 同上 |
| `plotting_fig2_extractHFOs.py` | 从检测中裁片段 | `*_gpu.npz` | `p16_cuda_24h_bipolar.py` | 可用新检测/裁剪链重做，但无现成一键图 |
| `plotting_fig2_synIndex.py` | 通道间同步指数 | 全目录 `*_gpu.npz` | `p16_cuda_24h_bipolar.py` | 新代码暂无同名统计图 |
| `plotting_fig2_AUC_groupingComp.py` | 原始 vs refine 的 AUC 比较 | `*_gpu.npz`, `*_refineGpu.npz` | cuda + refine | 新代码逻辑可复算，但无同图脚本 |
| `plotting_fig2_auc_groupingComp_purePlot_bar.py` | 只画 AUC 汇总 | `tmpRes_fig2_auc_yuquan.npz` | `plotting_fig2_AUC_groupingComp.py` | 无 |

### 4.2 Fig3 类

| 代表脚本 | 主要用途 | 读取资产 | 上游生成脚本 | 当前 HFOsp 替代 |
| --- | --- | --- | --- | --- |
| `plotting_fig3_lagPat.py` | lag pattern / histogram | `*_lagPat.npz` | `p16_packGroupEvents*.py` | `group_event_analysis` 可产 lag，图无直接替代 |
| `plotting_fig3_pickExamples.py` | 选“最像平均传播”的例子 | `*_lagPat.npz`, `*_packedTimes.npy` | pack + lag | 无 |
| `plotting_fig3_lag_3d.py` | 3D 电极传播图 | `*_lagPat.npz` + MRI/坐标文件 | pack + 外部 MRI 配准 | 无 |

### 4.3 Fig4 类

| 代表脚本 | 主要用途 | 读取资产 | 上游生成脚本 | 当前 HFOsp 替代 |
| --- | --- | --- | --- | --- |
| `plotting_fig4_durDelayStats.py` | 持续时间 / 频率中心统计 | `*_lagPat_withFreqCent.npz` | `...withFreqCenter.py` | 新代码已可产频心，但图无 |
| `plotting_fig4_pairedDelay.py` | 相邻延迟统计 | `*_lagPat_withFreqCent.npz` | 同上 | 无 |
| `plotting_fig4_FreqCenter80HzLowCheck.py` | 频率中心检查，甚至可重写 `withFreqCent` 产物 | `*_gpu.npz`, `*_refineGpu.npz` | cuda + refine | 这是诊断/修补脚本，不是主流程 |
| `plotting_fig4_regionHist.py` | 区域级传播统计 | `*_lagPat.npz` + 区域标签文件 | pack + 区域标签脚本 | 无 |
| `plotting_fig4_MI_demo.py` | Matching Index 单病人计算 | `*_lagPat.npz` | pack | 新代码暂无 MI 成熟替代 |
| `plotting_fig4_MI_stats_sigLine.py` | MI 群体统计画图 | `machingIndex/*.npz` | `plotting_fig4_MI_demo*.py` | 无 |

### 4.4 Fig5 类

| 代表脚本 | 主要用途 | 读取资产 | 上游生成脚本 | 当前 HFOsp 替代 |
| --- | --- | --- | --- | --- |
| `plotting_fig5_diffNetMatrix.py` | diffusion network 矩阵 | `*_lagPat.npz` + 外部 `A_hat.mat` | pack + `prepareFor_netRate.py` + 外部 netrate | `network_analysis.py` 不是同一路算法 |
| `plotting_fig5_lagPat_hist.py` | lag histogram | `*_lagPat.npz` | pack | 部分可替代，图无 |
| `plotting_fig5_lagPat_variance.py` | rank/lag 方差统计 | `*_lagPat.npz` | pack | 无 |

### 4.5 Fig6 类

| 代表脚本 | 主要用途 | 读取资产 | 上游生成脚本 | 当前 HFOsp 替代 |
| --- | --- | --- | --- | --- |
| `plotting_fig6_interDiff_ictalH2_computation.py` | diffNet vs ictal H2 相关 | `*_lagPat.npz` + 发作数据 + diffNet | pack + 发作预处理 + netrate | 无一键替代 |
| `plotting_fig6_diffH2_corrBar.py` | 画 FDR 后相关结果 | `*_fdrP.npz` | 上一脚本 | 无 |

### 4.6 Fig7 类

| 代表脚本 | 主要用途 | 读取资产 | 上游生成脚本 | 当前 HFOsp 替代 |
| --- | --- | --- | --- | --- |
| `plotting_fig7_realLagPat.py` | 真实 lag pattern 示例 | `*_lagPat.npz`, `*_packedTimes.npy` | pack + lag | 无 |
| `plotting_fig7_networkDemo.py` | lag + diffusion network 示意 | `*_lagPat.npz` + `A_hat.mat` | pack + netrate | 新网络模块不等价 |
| `plotting_fig7_compareRank1AndPropagation.py` | rank1 与传播关系 | `*_gpu.npz`, `*_refineGpu.npz`, `*_lagPat.npz`, `*_packedTimes.npy` | cuda + refine + pack | 无 |
| `plotting_fig7_compareHIAndRank1.py` | HFO 指标与 rank1 比较 | 同上 | 同上 | 无 |

### 4.7 Fig8 类

| 代表脚本 | 主要用途 | 读取资产 | 上游生成脚本 | 当前 HFOsp 替代 |
| --- | --- | --- | --- | --- |
| `plotting_fig8_seizureRelated_synchron.py` | 发作相关同步性比较 | `*_lagPat.npz`, `*_packedTimes.npy` | pack + lag | 当前 repo 无现成替代图 |
| `plotting_fig8_resortLagExamples.py` | 重排 lag 示例 | `*_lagPat.npz`, `*_packedTimes.npy` | pack + lag | 无 |
| `plotting_fig8_resortConnMatrix_finalPics.py` | 终版重排连接矩阵 | `*_lagPat.npz`, `*_packedTimes.npy`, `A_hat.mat` | pack + netrate | 无 |
| `plotting_fig8_moreAndEarly.py` | 传播早晚 vs 事件数关系 | `*_lagPat.npz`, `_refineGpu.npz` | pack + refine | 无 |
| `plotting_fig8_HigherAndEarly.py` | 传播早晚 vs 频率中心关系 | `*_lagPat_withFreqCent.npz` | withFreqCenter | 无 |

### 4.8 FigKura 类

| 代表脚本 | 主要用途 | 读取资产 | 上游生成脚本 | 当前 HFOsp 替代 |
| --- | --- | --- | --- | --- |
| `plotting_figKura_SeizureComp_10s_computation_COH.py` | 发作窗 COH/kuramoto/h2 对比 | `*_lagPat.npz` + 发作数据 + 仿真参数 | pack + 发作链 + 外部仿真 | 无 |
| `plotting_figKura_SeizureComp_10s_show.py` | 显示 FDR 结果 | `*_fdrP.npz` | computation 脚本 | 无 |
| `plotting_figKura_SeizureComp_countWholeTime.py` | 发作相关全时段统计 | `*_lagPat.npz` + 发作数据 | pack + 发作链 | 无 |

---

## 5. 怎么用这张表追一张图

### 5.1 如果图上写的是传播时序、rank、lag

优先看：

- `*_lagPat.npz`
- `*_packedTimes.npy`
- `plotting_fig3_*`
- `plotting_fig7_*`
- `plotting_fig8_resort*`

### 5.2 如果图上写的是频率中心、duration、paired delay

优先看：

- `*_lagPat_withFreqCent.npz`
- `*_packedTimes_withFreqCent.npy`
- `plotting_fig4_durDelayStats*.py`
- `plotting_fig4_pairedDelay.py`
- `plotting_fig8_HigherAndEarly.py`

### 5.3 如果图上写的是同步性、发作相关、Kuramoto

优先看：

- `plotting_fig8_seizureRelated_synchron.py`
- `plotting_figKura_*`
- 以及它们依赖的 `lagPat`, `packedTimes`, 发作数据目录

### 5.4 如果图上写的是 AUC、raw vs refine、SOZ

优先看：

- `plotting_fig2_AUC_groupingComp.py`
- `plotting_fig2_auc_*`
- `*_gpu.npz`
- `_refineGpu.npz`

### 5.5 如果图上写的是 diffusion network / A_hat / netRate

优先看：

- `plotting_fig5_diffNetMatrix.py`
- `plotting_fig7_networkDemo*.py`
- `plotting_fig6_interDiff_ictalH2_computation.py`
- 外部 netrate `A_hat.mat`

---

## 6. 当前 HFOsp 的现实情况

### 6.1 能替代的是计算主线

当前 `HFOsp` 已经能承接：

- 预处理
- HFO 检测
- legacy refine 逻辑
- packed windows
- spectrogram centroid / lag
- 新网络构建

关键入口：

- `src/preprocessing.py`
- `src/hfo_detector.py`
- `src/group_event_analysis.py`
- `src/network_analysis.py`
- `scripts/run_pipeline.py`

### 6.2 不能假装已经替代的是论文级绘图

当前 `src/visualization.py` 和 `scripts/visualize_run.py` 不是老 `plotting_fig*` 的一一映射。

结论：

- 新代码适合做**可维护计算主线**
- 老代码里的 `plotting_fig*` 仍然是追历史论文图的第一现场

---

## 7. 最短结论

- 追图先认资产，不认图号。
- `lagPat` / `packedTimes` 是老图世界的硬通货。
- `withFreqCenter` 会导致文件名和读图脚本漂移，这是最常见的坑。
- `HFOsp` 现在能接计算，但还没把老论文画图系统重建完。
