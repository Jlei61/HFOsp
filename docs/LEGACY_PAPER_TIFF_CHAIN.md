# Legacy Paper TIFF Chain — Panel 级链路表

> 目的：以 `ReplayIED/tiffs` 为论文图的权威顺序来源，建立 **panel 级** 的溯源表：
>
> `TIFF 文件 → Panel (A/B/C/D/E) → 画图脚本 → 输入资产 → 资产生成脚本 → 当前 HFOsp 状态`
>
> **用法**：追 `7b` 这种引用 → 直接查 Figure 7 → Panel B。

---

## 0. 铁律

1. **TIFF 编号 ≠ 脚本前缀**。`fig1_*.tif` 的内容来自 `plotting_fig2_*`，`fig7_*.tif` 右半来自 `plotting_fig8_*`。
2. **一张主图 = 多脚本拼成**。脚本各自产 panel，最终在排版软件拼成 TIFF。
3. **追图先认资产，再认脚本**。

---

## 1. TIFF 清单

主图 7 张，补图 13 张。


| #      | TIFF 文件名                               |
| ------ | -------------------------------------- |
| 1      | `fig1_prop_widder_画板 1_画板 1_画板 1.tif`  |
| 2      | `fig2_prop_hist_9p_画板 1_画板 1.tif`      |
| 3      | `fig3_periodicity_画板 1.tif`            |
| 4      | `fig4_hebb_kura_画板 1.tif`              |
| 5      | `fig5_958_bidir_画板 1.tif`              |
| 6      | `fig6_ictal_corr_画板 1.tif`             |
| 7      | `fig7_adapKura_画板 1.tif`               |
| S1–S13 | `fig_s1_画板 1.tif` … `fig_s13_画板 1.tif` |


---

## 2. 主图 Panel 级链路

### Figure 1 — `fig1_prop_widder`

> 视觉主题：SEEG 数据演示 + AUC + 同步性指数 + 归一化谱图


| Panel | 视觉内容                                                                       | 画图脚本                                                                        | 输入资产                                                              | 资产生成脚本                                                | HFOsp 状态  |
| ----- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------- | --------- |
| **A** | Yuquan Y1 & Epilepsiae E7 的 SEEG traces，红色标记检出 IEs，灰色标记 synchronous events | `plotting_fig2_dataDemoNmethods.py` 或 `plotting_fig2_SEEGdataDemo_chap2.py` | `*_gpu.npz`, `*_packedTimes.npy`, EDF                             | `p16_cuda_24h_bipolar.py`, `p16_packGroupEvents*.py`  | 计算可替代，图无  |
| **B** | ROC/AUC 曲线（Yuquan mean=0.857, Epilepsiae mean=0.772），含 AUC boxplot 嵌图      | `plotting_fig2_AUC_groupingComp.py` 或 `plotting_fig2_auc_curves_rawNsyn.py` | `*_gpu.npz`, `_refineGpu.npz` → 中间产物 `tmpRes_fig2_auc_yuquan.npz` | `p16_cuda_24h_bipolar.py`, `p16_refine_chns_bySyn.py` | 计算可替代，图无  |
| **C** | In-SOZ vs Out-SOZ Synchronization boxplot（Yuquan + Epilepsiae）             | `plotting_fig2_synIndex.py`                                                 | 全目录 `*_gpu.npz` → 中间产物 `synIndex/*.npz`                           | `p16_cuda_24h_bipolar.py`                             | 新代码暂无同名统计 |
| **D** | 多通道 HFO envelope + Normalized Spectrogram（Y1 + E7）                         | `plotting_fig2_dataDemoSpecEnve_comp.py`（GridSpec 2×3）                      | `*_gpu.npz`, EDF                                                  | `p16_cuda_24h_bipolar.py`                             | 计算可替代，图无  |


**置信度**：很高

---

### Figure 2 — `fig2_prop_hist_9p`

> 视觉主题：传播模式 heatmap + Matching Index 方法论 + MI 群体统计


| Panel | 视觉内容                                                                                                                       | 画图脚本                                                     | 输入资产                                                             | 资产生成脚本                                                                | HFOsp 状态  |
| ----- | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------- | --------- |
| **A** | Propagation Pattern heatmap（channels × Pop Events, First→Last rank），Y1 (24h, n=18005) + E7 (170.6h, n=6556)，含 day/night 条带 | `plotting_fig3_lagPat.py`（`fig.add_axes` 布局）             | `*_lagPat.npz`（`lagPatRaw`, `lagPatRank`, `chnNames`, `start_t`） | `p16_packGroupEvents*.py`                                             | lag 计算可替代 |
| **B** | MI 方法学示意（Perm 0/1, Orig Pats, Mean Pat）+ Permuted MI 分布曲线 vs MI(with Main sequence) 分布                                     | `plotting_fig4_MI_demo.py`（计算，含 `plt.figure('mi hist')`） | `*_lagPat.npz` → 中间产物 `machingIndex/*.npz`                       | `p16_packGroupEvents*.py`                                             | MI 无成熟替代  |
| **C** | 全体 Yuquan + Epilepsiae 的 MI violin/boxplot，含 day/night 显著性标记                                                               | `plotting_fig4_MI_stats_sigLine.py`                      | `machingIndex/*.npz`                                             | `plotting_fig4_MI_demo.py` / `plotting_fig4_MI_resultsAllPatients.py` | 无         |

**Panel 语义警告**:

- Figure 2 的 `hist_meanX / MI` 语义不是“严格参与通道传播模板”，而是**未屏蔽 `lagPatRank` 模板链**。
- `plotting_fig4_MI_demo.py`、`p16_quantify_MI.py`、`p16_MI_24hChange*.py` 都是直接对 `lagPatRank` 做模板和 MI，比对前没有先用 `eventsBool` 屏蔽未参与通道。
- 因此 Figure 2 更适合解释为“老实现定义下的 rank pattern 重复性/稳定性”，不应直接上升为“真实参与通道传播模板稳定性”。


**置信度**：很高

---

### Figure 3 — `fig3_periodicity`

> 视觉主题：周期性分析（PSD 分解 + power-law fit + 群体频率分布）


| Panel | 视觉内容                                                                                                   | 画图脚本                                                               | 输入资产                                                   | 资产生成脚本                                                           | HFOsp 状态 |
| ----- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------ | ---------------------------------------------------------------- | -------- |
| **A** | Epilepsiae E10：raw SEEG traces + SOZ (橙) + IEs (红) + Event-Windows (灰)，底部 ICL1 和 Pop 脉冲列               | 同 S7-A，demo 段可能来自 `plotting_figAdd_qusiPSDdecomp.py` 或专用 demo      | `*_gpu.npz`, `*_packedTimes.npy`, EDF                  | `p16_cuda_24h_bipolar.py`, `p16_packGroupEvents*.py`             | 计算可替代    |
| **B** | PSD 分解：ICL1 (Freq=2.11Hz) 和 Pop (Freq=2.10Hz)，显示 Original PSD / Aperiodic Fit / Full Fit / Period Freq | `plotting_figAdd_qusiPSDdecomp.py`（FOOOF `report()`）               | `*_perChn_events_histNpsd.npz`, `*_group_histNpsd.npz` | `plotting_figAdd_qusiPSDdecomp.py` 自身（写 `*_psd_hist_fitRes.npz`） | 无        |
| **C** | 群体级 PSD 曲线叠（Y1-Y18 + E1-E20），含 periodic peak 标注，底部 PeakHist 柱状图                                        | `plotting_figAdd_qusiPeriod_plotting_piled_stats.py`（GridSpec 3×1） | `*_psd_hist_fitRes.npz`                                | `plotting_figAdd_qusiPSDdecomp.py`                               | 无        |


**置信度**：很高

---

### Figure 4 — `fig4_hebb_kura`

> 视觉主题：Hopfield / Kuramoto 仿真 vs 经验 rank 对照


| Panel | 视觉内容                                                                                                                  | 画图脚本                                                                            | 输入资产                                 | 资产生成脚本                                         | HFOsp 状态                                     |
| ----- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------ | ---------------------------------------------- | -------------------------------------------- |
| **A** | 示意图：Hopfield Network（Hebbian matrix → binary → resorted）和 Hopfield Kuramoto Network（oscillators → coupled → resorted） | 外部排版工具手绘；`hop_kura_Vhebb.py` 产相关数据                                              | N/A（示意）                              | —                                              | —                                            |
| **B** | Epilepsiae E10 相位轨迹（Phase rad × Iterations），通道 ICL11→SCL6，虚线标事件边界                                                     | `epi.../plotting_figKura_AdapDiffnetComp.py`（`subplot(2,1,1                     | 2)`）                                 | `patientParams_brian2/{sub}_brian2Params.npz`  | `hop_kura_Vhebb.py` / `hop_kura_Vcomplex.py` |
| **C** | Simulation vs Empirical rank heatmap（Channels × Pop Events, First→Last 色标）                                            | `epi.../plotting_figKura_SeizureComp_10s_demo.py`（`'resort lagPat'`）            | `*_brian2Params.npz`, `*_lagPat.npz` | `hop_kura_Vhebb.py`, `p16_packGroupEvents*.py` | empirical lag 可替代                            |
| **D** | Simulation vs Empirical 每通道 rank 分布直方图 + mean rank overlay                                                            | `epi.../plotting_figKura_SeizureComp_10s_demo.py`                               | 同 Panel C                            | 同 Panel C                                      | 同 Panel C                                    |
| **E** | Empirical Mean Rank vs Simulation Mean Rank 散点图（r=1.00）                                                               | `epi.../plotting_figKura_SeizureComp_10s_demo.py`（`'corr'` + seaborn `regplot`） | 同 Panel C                            | 同 Panel C                                      | 同 Panel C                                    |


> `epi...` = `epilepsiae_interictal/`

**置信度**：高（Panel A 是手绘，无法追到单一脚本）

---

### Figure 5 — `fig5_958_bidir`

> 视觉主题：双向传播（concordant/opposite 仿真 + E3 实证 forward/reverse 聚类）


| Panel | 视觉内容                                                                | 画图脚本                                                                                  | 输入资产                                     | 资产生成脚本                         | HFOsp 状态 |
| ----- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------ | -------- |
| **A** | 示意：10 个 oscillator 相位排列 + connection matrix                         | `hop_kura_Vhebb.py`（`'pre patt'`, `'conn'`）或手绘                                        | 脚本内生成                                    | —                              | —        |
| **B** | Concordant 和 Opposite Kuramoto 仿真（Nodes × Iterations, Phase rad）    | `epi.../plotting_figKura_epilepsiae958Cluster.py` 或 `hop_kura_Vhebb.py`               | 内部仿真参数                                   | 仿真脚本                           | 无        |
| **C** | Epilepsiae E3 rank heatmap（原序 + 聚类后 Reverse/Forward 分离）             | `epi.../plotting_figKura_epilepsiae958Cluster.py`（`'lagPat ori'`, `'lagPat cluster'`） | `*_lagPat.npz`                           | epilepsiae 版 `packGroupEvents` | lag 可替代  |
| **D** | Forward & Reverse 代表事件：Raw Signal + High Freq + Spectrograms        | `plotting_fig5_diffNetExamples.py`（GridSpec 1×5）或 `plotting_fig7_realLagPat.py`       | `*_lagPat.npz`, `*_packedTimes.npy`, EDF | `p16_packGroupEvents*.py`      | lag 可替代  |
| **E** | Forward & Reverse 平均 rank profile + Pattern Correlation 散点（r=−0.91） | `epi.../plotting_figKura_epilepsiae958Cluster.py`（`subplot(1,4,...)`）                 | `*_lagPat.npz`                           | `p16_packGroupEvents*.py`      | lag 可替代  |


**置信度**：中—高

---

### Figure 6 — `fig6_ictal_corr`

> 视觉主题：发作数据 + Hebbian vs Ictal coherence 相关 + 群体统计 + 时间动态


| Panel | 视觉内容                                                                      | 画图脚本                                                          | 输入资产                                                                 | 资产生成脚本                                                            | HFOsp 状态 |
| ----- | ------------------------------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------- | -------- |
| **A** | Seizure Onset traces（ICL11→SCL6, 40–90s）                                  | `plotting_fig6_seizureData.py`                                | 发作目录 `*.npz`（szData, szT, fs, chns）                                  | 发作预处理链                                                            | 无        |
| **B** | Interictal Hebbian matrix 与 Ictal Coherence matrix 并排                     | `plotting_fig6_interDiff_ictalH2_computation.py`（`'gmatrix'`） | `*_lagPat.npz` → Hebbian; 发作数据 → H2/coherence; `A_hat.mat` → diffNet | `p16_packGroupEvents*.py`, `prepareFor_netRate.py` + netrate, 发作链 | 无        |
| **C** | Rank(Correlation) vs Rank(Hebbian) 散点（r=0.75, p=1.2e-38）                  | `plotting_fig6_interDiff_ictalH2_computation.py`（`'corr'`）    | 同 Panel B                                                            | 同 Panel B                                                         | 无        |
| **D** | 全体 Yuquan + Epilepsiae 的 Significant Percentile / Network Correlation 柱状图 | `plotting_fig6_diffH2_corrBar.py`（`add_subplot(111)`）         | `*_fdrP.npz`                                                         | `plotting_fig6_interDiff_ictalH2_computation.py`                  | 无        |
| **E** | Epilepsiae E10 (n=26 seizures) Correlation(R) vs Time(a.u.) 热图            | `plotting_figKura_SeizureComp_10s_show.py` 或相关时序分析脚本          | `*_fdrP.npz`, seizure computation 结果                                 | `plotting_figKura_SeizureComp_10s_computation_*.py`               | 无        |


**置信度**：很高

---

### Figure 7 — `fig7_adapKura`

> 视觉主题：adaptive Kuramoto 相图 + 发作前同步性趋势


| Panel | 视觉内容                                                                                                                           | 画图脚本                                         | 输入资产                                                         | 资产生成脚本                    | HFOsp 状态  |
| ----- | ------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------- | ------------------------------------------------------------ | ------------------------- | --------- |
| **A** | α-β 相图（R₂ colormap, Splay/Incoherent/Synchronous）+ Forward/Backward R₂ vs α 曲线 + 三个 oscillator raster 示例                       | `epi.../plotting_figKura_AdapDiffnetComp.py` | Kuramoto 仿真参数 / adaptive diffnet 搜索结果                        | 脚本内仿真                     | 无         |
| **B** | Epilepsiae E14：Synchronization vs Time before seizure onset 散点（r=0.1473, p=3.20e-14, n=14）+ 时间分箱 boxplot（−3240, −1800, −360 s） | `plotting_fig8_seizureRelated_synchron.py`   | `*_lagPat.npz`, `*_packedTimes.npy`, 发作时间标注（MNE annotations） | `p16_packGroupEvents*.py` | 数据链可部分替代  |
| **C** | 全 Epilepsiae 逐病人 Pearson Correlation strip（Significant/Negative 2/6, Significant/Positive 7/14）                                | `plotting_fig8_seizureRelated_synchron.py`   | 同 Panel B                                                    | 同 Panel B                 | 同 Panel B |


补充说明（Figure 7B/C 的实际逻辑链）：

1. `plotting_fig8_seizureRelated_synchron.py` 是“发作相关同步性比较”的历史入口，但论文里这张发作前趋势图的干净实现，实际更接近：
   `plotting_fig8_seizureRelated_synchron.py` / `plotting_fig9_seizureSynch.py` 生成按发作对齐的散点数据，再由 `plotting_fig9_preSeizureSyn_plot.py` 出最终 scatter + boxplot。
2. 事件级同步性不是直接用 `max(lag)-min(lag)`，而是对每个群体事件先取参与通道的 `lagPatRaw`（用 `eventsBool` 做掩码），再算：
   `pDelay = std(lagPatRaw_participating) / (max(lag)-min(lag))`
   最终画图用的是 `Synchronization = 1 - pDelay`。
3. `*_packedTimes.npy` 提供每个群体事件的真实时间戳；脚本把多个记录文件的 `packedTimes` 拼到同一绝对时间轴，再用发作起始时间做对齐。
4. 多个 seizure 的对齐方式不是按发作长度拉伸，而是固定取“发作前 1 小时”的群体事件窗口，并把 onset 记为 `0`。为避免窗口互相污染，只保留相邻发作起点间隔大于 `3h` 的 seizure；Panel B 标注里的 `n=14` 指的是该 subject 最终保留并参与对齐的 seizure 数，不是 subject 数。
5. Panel B 是单 subject（Epilepsiae E14，对应老代码里的 `sub='548'`）的事件级散点；黑线是对该 subject 所有对齐后事件做线性回归。图中的 `r` 是同一批散点的 Pearson correlation：`corr(time_before_onset, Synchronization)`。
6. Panel C 不是再算一次群体事件指标，而是把每个 subject 的 Pearson `r` 汇总后做 strip/scatter；脚本里对 20 个 Epilepsiae subject 做了这一步，图注里的 `7/14` 与 `2/6` 对应正相关与负相关两侧各自显著的病人数。
7. 这套脚本不能直接当通用工具复用。它硬编码了 Epilepsiae 数据根、`sub_list`、以及发作注释解析方式；只有在同时具备 `*_lagPat.npz`、`*_packedTimes.npy` 和可解析的 seizure onset 标注时，才可以迁移到别的 subject。对当前论文 Figure 7B/C，实际覆盖的是 Epilepsiae 这 20 个 subject，而不是玉泉全体 subject。

**置信度**：很高

---

## 3. 补图 Panel 级链路

### Figure S1

> 视觉主题：HFO / Spike / Coupled 分类样本 + t-SNE


| Panel | 视觉内容                                                                    | 画图脚本                                                                             | 输入资产            | 资产生成脚本                                                    |
| ----- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------- | --------------- | --------------------------------------------------------- |
| **A** | HFO / Spike / Coupled 三列：raw waveform + highpass 80-250Hz + spectrogram | `plotting_fig4_hfoSpike_clusterAnalysis_specs.py`                                | `*_cutSigs.npz` | `plotting_fig2_extractHFOs.py`, `p16_cuda_24h_bipolar.py` |
| **B** | t-SNE 散点（HFO 红, Coupled 绿, Spike 蓝）+ 代表波形 inset                         | `plotting_fig4_hfoSpike_clusterAnalysis.py`（`add_subplot(111, projection='3d')`） | cluster 分析结果    | `plotting_fig4_hfoSpike_clusterDataGen.py`                |


---

### Figure S2

> 视觉主题：HFO / Spike / Coupled 平均波形与频谱


| Panel | 视觉内容                                                                     | 画图脚本                                                                  | 输入资产 | 资产生成脚本 |
| ----- | ------------------------------------------------------------------------ | --------------------------------------------------------------------- | ---- | ------ |
| **A** | HFO/Spike/Coupled 平均波形（n=178/202/296）+ raw Spec + normalized Spec，3×3 网格 | `plotting_fig4_hfoSpike_clusterAnalysis_specs.py`（`subplot(1,2,...)`） | 同 S1 | 同 S1   |
| **B** | Normalized Energy vs Freq (Hz)，HFO/Spike/Coupled 三行                      | `plotting_fig4_hfoSpike_clusterAnalysis_specsNorm.py`                 | 同 S1 | 同 S1   |


---

### Figure S3

> 视觉主题：Raw vs Synchronized AUC 对比


| Panel | 视觉内容                                                                 | 画图脚本                                                            | 输入资产                          | 资产生成脚本                                                |
| ----- | -------------------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------- | ----------------------------------------------------- |
| **A** | Yuquan AUC Syn (mean=0.891) + Epilepsiae AUC Syn (mean=0.768) ROC 曲线 | `plotting_fig2_auc_curves_rawNsyn.py`                           | `*_gpu.npz`, `_refineGpu.npz` | `p16_cuda_24h_bipolar.py`, `p16_refine_chns_bySyn.py` |
| **B** | Raw vs Synchronized AUC paired bar（Yuquan *, Epilepsiae n.s.）        | `plotting_fig2_auc_groupingComp_purePlot_bar.py`（`subplots(1)`） | `tmpRes_fig2_auc_yuquan.npz`  | `plotting_fig2_AUC_groupingComp.py`                   |


---

### Figure S4

> 视觉主题：代表事件窗示例 + 3D 电极传播


| Panel  | 视觉内容                                                         | 画图脚本                                                          | 输入资产                                             | 资产生成脚本                            |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------- | ------------------------------------------------ | --------------------------------- |
| **上行** | Y1 + E7 raw traces + 选中 SOZ 通道的 waveform + spectrogram inset | `plotting_fig3_pickExamples.py`（GridSpec 1×5：raw, high, spec） | `*_lagPat.npz`, `*_packedTimes.npy`, `*_gpu.npz` | `p16_packGroupEvents*.py`         |
| **下行** | 3D 电极传播图（First→Last 色标，脑表面渲染）                                | `plotting_fig3_lag_3d.py`（PySurfer + Mayavi）                  | `*_lagPat.npz`, `chnXyzDict.npy`, MRI（nibabel）   | `p16_packGroupEvents*.py`, MRI 配准 |


---

### Figure S5

> 视觉主题：质心延迟与频率统计


| Panel | 视觉内容                                                      | 画图脚本                                                              | 输入资产                        | 资产生成脚本                 |
| ----- | --------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------- | ---------------------- |
| **左** | Consecutive Delay of Mass-Center Time（violin per patient） | `plotting_fig4_pairedDelay.py` 或 `plotting_fig4_durDelayStats.py` | `*_lagPat_withFreqCent.npz` | `...withFreqCenter.py` |
| **右** | Frequency of Mass-Center（violin per patient, 80-250Hz 范围） | 同左                                                                | 同左                          | 同左                     |


---

### Figure S6

> 视觉主题：First-Half-Rank 通道的脑区分布


| Panel | 视觉内容                                                             | 画图脚本                                            | 输入资产                                                        | 资产生成脚本                                                                         |
| ----- | ---------------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **A** | 个例（Y1, Y5）：脑区 bar + 3D 电极 UpStream 示意                            | `plotting_fig4_regionHist_median.py` + 3D 渲染    | `*_lagPat.npz`, region labels `<sub>.npy`, `chnXyzDict.npy` | `plotting_fig4_elecRegionLabels.py`（写 region labels），`p16_packGroupEvents*.py` |
| **B** | 群体 First-Half-Rank 通道 ratio by brain region（Yuquan + Epilepsiae） | `plotting_fig4_regionHist_median.py`（bar chart） | region labels, `*_lagPat.npz`                               | `plotting_fig4_elecRegionLabels.py`                                            |


---

### Figure S7

> 视觉主题：Power-law fit + Hist R-square


| Panel | 视觉内容                                                                              | 画图脚本                                                    | 输入资产                                  | 资产生成脚本                                               |
| ----- | --------------------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------- | ---------------------------------------------------- |
| **A** | Epilepsiae E10 raw traces + SOZ/IEs/Event-Windows（与 Fig 3A 类似但侧重展示 power-law 上下文） | demo 段，同 Fig 3A                                         | `*_gpu.npz`, `*_packedTimes.npy`, EDF | `p16_cuda_24h_bipolar.py`, `p16_packGroupEvents*.py` |
| **B** | ICL1 (γ=−2.14) + Pop (γ=−2.01) Power-law fit：log(Prob) vs log(T)                  | `plotting_figAdd_qusiPeriod_plotting.py`（`'hist expo'`） | `*_psd_hist_fitRes.npz`               | `plotting_figAdd_qusiPSDdecomp.py`                   |
| **C** | Hist R-square violin per patient（Per-Chn + Pop Event dots），Yuquan + Epilepsiae    | `plotting_figAdd_qusiPeriod_plotting.py`（`'hist r2'`）   | `*_psd_hist_fitRes.npz`               | `plotting_figAdd_qusiPSDdecomp.py`                   |


---

### Figure S8

> 视觉主题：Hebbian Network 构建与 resort


| Panel | 视觉内容                                                                  | 画图脚本                                                                          | 输入资产                                      | 资产生成脚本              |
| ----- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------- | ------------------- |
| **A** | Rank heatmap → Hebbian matrix 推导示意（Δθ → F(Δθ) 学习规则）                   | `epi.../plotting_figKura_SeizureComp_10s_demo.py`                             | `patientParams_brian2/*_brian2Params.npz` | `hop_kura_Vhebb.py` |
| **B** | Hebbian Network（原序）和 Resort Hebbian Network（rank 重排）矩阵 + rank profile | `epi.../plotting_figKura_SeizureComp_10s_demo.py`（`'aHat'` / `'resort aHat'`） | 同 Panel A                                 | 同 Panel A           |
| **C** | Network graph 可视化（NetworkX 节点布局）                                      | `epi.../plotting_figKura_SeizureComp_10s_demo.py`（`'graph'` / `'sort graph'`） | 同 Panel A                                 | 同 Panel A           |


---

### Figure S9

> 视觉主题：节点特异噪声下的 Kuramoto 仿真 + 经验 power-law 对照


| Panel | 视觉内容                                                                                                              | 画图脚本                                                              | 输入资产                                      | 资产生成脚本                                          |
| ----- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------- |
| **A** | Node-specific noise traces (SCL6→ICL1) + 网络示意 + common velocity 公式                                                | `epi.../plotting_figKura_patientSpecific_withNoise.py`（`'noise'`） | `patientParams_brian2/*_brian2Params.npz` | `hop_kura_Vhebb.py`                             |
| **B** | Phase trajectory（with noise, Channels × Iterations）+ Pop(Simulation) power-law fit + Pop(Empirical) power-law fit | `epi.../plotting_figKura_patientSpecific_withNoise_plot.py`       | `sim_res_all_*.npz`, `noisesignals_*.npz` | `plotting_figKura_patientSpecific_withNoise.py` |


---

### Figure S10

> 视觉主题：标准 Kuramoto 相图（α-J 空间，无 adaptive）


| Panel  | 视觉内容                                                                                                               | 画图脚本                                         | 输入资产  | 资产生成脚本 |
| ------ | ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------- | ----- | ------ |
| **整图** | α-J 相图 (R₂ colormap) + Kuramoto 方程 + Forward/Backward R₂ vs α 曲线 + Incoherent/Synchronous raster plot（Fig 7A 的展开版） | `epi.../plotting_figKura_AdapDiffnetComp.py` | 脚本内仿真 | 脚本内仿真  |


---

### Figure S11

> 视觉主题：CPU vs GPU 性能对比


| Panel | 视觉内容                                                                                               | 画图脚本                                                 | 输入资产                       | 资产生成脚本               |
| ----- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | -------------------------- | -------------------- |
| **A** | CPU vs GPU speedup bar（Whole-Pipeline 17.49×, Resampling 19.29×, Filtering 5.52×, Envelope 23.01×） | `compare_CPUGPU_plot_speedUp.py`（`subplot(1,N,i+1)`） | `./gpuT.npz`, `./cpuT.npz` | CPU/GPU benchmark 脚本 |
| **B** | CPU counts vs GPU counts per-channel bar（r=0.9996）                                                 | `compare_CPUGPU_plot_dets.py`（`subplot(1,2,1|2)`）    | `*_cpu.npz`, `*_gpu.npz`   | CPU/GPU detection 脚本 |


---

### Figure S12

> 视觉主题：Refine 前后对比流程


| Panel | 视觉内容                                                                                                    | 画图脚本                                                                  | 输入资产             | 资产生成脚本                     |
| ----- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ---------------- | -------------------------- |
| **左** | Before：Number of Events per Channel bar + Clinical SOZ overlay                                          | `p16_refine_chns_bySyn.py`（`'PRE'` 图 → `pick_chns.png`）               | `*_gpu.npz`      | `p16_cuda_24h_bipolar.py`  |
| **中** | Envelope (80-250Hz) 多通道展示 + "drop" 标注 + "determine population events" / "refine interictal events" 流程箭头 | `p16_inspectHighEvents_bipolar_dropChns.py`（`plot_dets_withNPZ`）或手动拼接 | `*_gpu.npz`, EDF | `p16_cuda_24h_bipolar.py`  |
| **右** | After：Refined Number of Events bar + Clinical SOZ overlay                                               | `p16_refine_chns_bySyn.py`（`refine_hist` 图 → `refine_hist.png`）       | `_refineGpu.npz` | `p16_refine_chns_bySyn.py` |


---

### Figure S13

> 视觉主题：PSD 分解方法详解


| Panel  | 视觉内容                                                                | 画图脚本                                                 | 输入资产                           | 资产生成脚本                             |
| ------ | ------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------ | ---------------------------------- |
| **上行** | Raw log(PSD) + Full Fit + Aperiodic Fit（Yuquan Y2 + Epilepsiae E10） | `plotting_figAdd_qusiPSDdecomp.py`（FOOOF `report()`） | `*_perChn_events_histNpsd.npz` | `plotting_figAdd_qusiPSDdecomp.py` |
| **中行** | log(PSD) − Aperiodic Fit 残差 + Found Peak 标记                         | 同上                                                   | 同上                             | 同上                                 |
| **下行** | z-score / significance histogram（z=−256/−245, p=0.000）              | 同上                                                   | 同上                             | 同上                                 |


---

## 4. 上游资产链总结

大多数 TIFF 最终都落在下面这几条主链上：


| #   | 生产脚本                                                     | 产物                                                            |
| --- | -------------------------------------------------------- | ------------------------------------------------------------- |
| 1   | `p16_cuda_24h_bipolar.py`                                | `*_gpu.npz`                                                   |
| 2   | `p16_refine_chns_bySyn.py`                               | `_refineGpu.npz`                                              |
| 3   | `p16_packGroupEvents*.py` + `hfo_net.py`                 | `*_packedTimes.npy`, `*_lagPat.npz`                           |
| 4   | `...withFreqCenter.py`                                   | `*_lagPat_withFreqCent.npz`, `*_packedTimes_withFreqCent.npy` |
| 5   | `p16_merge24h_lagPat.py`                                 | `hist_meanX.npz`                                              |
| 6   | `prepareFor_netRate.py` + 外部 netrate                     | `A_hat.mat`                                                   |
| 7   | `plotting_figAdd_qusiPSDdecomp.py`                       | `*_psd_hist_fitRes.npz`, `*_histNpsd.npz`                     |
| 8   | `plotting_fig4_MI_demo.py` / `..._resultsAllPatients.py` | `machingIndex/*.npz`                                          |
| 9   | `plotting_fig6_interDiff_ictalH2_computation.py`         | `*_fdrP.npz`, `diffNetData`, `h2Matrx`, `spearman`            |
| 10  | `hop_kura_Vhebb.py` / `hop_kura_Vcomplex.py`             | `patientParams_brian2/*_brian2Params.npz`                     |
| 11  | `plotting_figKura_patientSpecific_withNoise.py`          | `sim_res_all_*.npz`, `noisesignals_*.npz`                     |

补充语义：

- `*_lagPat.npz` 里同时有 `lagPatRaw / lagPatRank / eventsBool`，但不少模板脚本只直接消费 `lagPatRank`。
- `hist_meanX.npz` 是 `p16_merge24h_lagPat.py` 基于**未屏蔽 `lagPatRank` 直方图**提取的模板位置统计，不是原始 lag 时间均值。
- `machingIndex/*.npz` 这条 MI 资产链，同样建立在**未屏蔽 `lagPatRank` 模板**上。


---

## 5. 当前 HFOsp 替代状态

### 已经能接住的

- 预处理 + HFO 检测 → `*_gpu.npz` 等价
- legacy refine → `_refineGpu.npz` 等价
- packed windows → `*_packedTimes.npy` 等价
- spectrogram centroid / lag → `*_lagPat.npz` 等价

注意边界：

- 上面这句“等价”只覆盖**资产生成层**。
- 对直接依赖 `hist_meanX / MI / variance` 的论文面板，当前真正要对齐的不只是 `lagPat.npz` 文件形状，还包括老代码那条**未屏蔽 rank 模板链**的历史语义。

### 还没重建完的

- MI 统计全套（Fig 2B/C）
- diffusion / netrate 论文图（Fig 6B-E）
- Kuramoto / Hopfield 仿真链（Fig 4, 5, 7A, S8-S10）
- hfo/spike/coupled 聚类图（S1, S2）
- periodicity 论文级图（Fig 3, S7, S13）
- 发作相关图（Fig 6, 7B/C）
- 脑区分布图（S6）
- CPU/GPU benchmark 图（S11）

---

## 6. 最短结论

- **追 `7b`** → 查 Figure 7 → Panel B → `plotting_fig8_seizureRelated_synchron.py` → `*_lagPat.npz` + `*_packedTimes.npy` + seizure annotations
- **追某张图的某个 panel** → 在本表对应 Figure 段找到 Panel 行 → 读画图脚本列 → 读输入资产列 → 读资产生成脚本列
- TIFF 编号 ≠ 脚本前缀，永远从图面内容出发
- 凡是主要依赖 `hist_meanX / MI / variance` 的 panel，先默认它在讲**老式未屏蔽 rank 模板链**，除非脚本明确先用了 `eventsBool`

