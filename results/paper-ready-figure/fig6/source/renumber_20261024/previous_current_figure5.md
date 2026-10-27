# 当前 Figure 5

> **2026-10-11 最新重排候选（待作者目视检查）**：[完整图 PDF](../results/paper-ready-figure/fig5/candidates/reference_layout_20261011/figures/fig5-complete-layout.pdf) / [PNG](../results/paper-ready-figure/fig5/candidates/reference_layout_20261011/figures/fig5-complete-layout.png)。按作者参考图整理为左列 A 波形、B 慢变量、C 四帧空间图，右列 D 三维轨迹、E 模型/患者起始场、F 队列统计；原栅格行保留在来源包。A–E 沿用 `fig5_fig4f_historical_zm_20260930/right_column_offset_20261008` 的冻结图层（14.3–15.4秒显示窗）；F 沿用 `broadband_phase_statistics_20261009/two_model_tfr` 中作者指定的13人/140次候选dB统计（p=0.00708），不是旧版队列结果。示例场保留原归一化z定义，队列保持dB定义。producer为 `scripts/paper_figures/build_fig5_reference_layout.py`；[来源与布局核对](../results/paper-ready-figure/fig5/candidates/reference_layout_20261011/layout_manifest.json)。本次仅拼版，无新仿真或统计重算，未替换下述旧顶层登记入口。

2026-09-30用户要求系统调整后，当前显示入口为主目录[Fig5整图](../results/paper-ready-figure/fig5/figures/fig5-complete-layout.pdf)，身份仍为 `single_seed_zm_entry_and_onset_field`；新producer为[build_fig5_transition_focus.py](../scripts/paper_figures/build_fig5_transition_focus.py)。这次重排沿用原seed9108401空间Z/M轨迹与F的冻结数据，没有替换为其他rate或Z/G/K模型。

- A：15触点未滤波原始突触电流读出，保留持续电平，按进入前原始信号P95固定定标，显示8–11秒；仅一个波形轴。
- B：上Z蓝色、下有效adaptation橙色；All E实线、Core A虚线，两行与A同轴。
- C：8.370–8.670及9.370–9.670秒的Core A/B E-only raster窗。
- D：9.420、9.480、9.600、10.370秒的50 ms空间快照，中间两帧提前到核A活动的早期扩展。
- E：同一时间窗状态轨迹，状态1–5与A/B对应，其中2–5对应D。
- F：原模型/患者早期宽频场，放在E下方。

原Rest状态移除，原Interictal空间图移除，间期时刻继续以A/B竖线及C栅格保留。状态1移至窗内8.420秒的晚期间期事件，2和5保留原Pre-ictal与Onset，3和4前移至9.480和9.600秒。左列占比缩小，A–B留白增加，B两行压扁并靠拢；独立panel不带字母，完整拼版保留A–F。

**读出检查**：按作者要求，A直接绘制2000 Hz原始记录中8–11秒的全部6001个采样点；不滤波、不去均值、不去趋势或作CAR，保留持续电平。固定缩放使用0–9.420秒原始信号P95；已从绘图曲线逆变换核对原始样本。比较8–9.37秒与10.2–11秒，15触点内前后原始均值比值的中位数为10.30，波动标准差比为0.127；全体E平均率由29.30升至335.77 Hz，全体I由59.65升至432.59 Hz。该轨迹进入后是持续高电平和较小波动，不能由30–80 Hz曲线看起来平坦判断为静息，也不能由高放电率认定已经产生发作样周期振荡。详见[原始读出核对](../results/paper-ready-figure/fig5/validation/unfiltered_readout_check.json)；[原带通比较诊断](../results/paper-ready-figure/fig5/readout_audit/figures/readout_diagnosis.pdf)保留作辅助。

**模型身份**：本图仍为历史手放双核，不是Fig4 F2。另行启动[Fig4 F2原生Z/M实验](../results/topic4_sef_hfo/fig5_f2_native_zm_20260930/README.md)，保持`xy_left_20`实际拟合的网络、输入、拓扑2511及动力学种子847401，仅加入原Z/M。先做1秒原无Z/M轨迹前缀核验，再观察30秒；只用一个种子，其实时状态以实验目录为准，不能从本图推断F2已经复现过渡。

[详细图注](../results/paper-ready-figure/fig5/fig5_caption_detailed_zh.md) · [当前版本](../results/paper-ready-figure/fig5/current_version.json) · [单panel下载](../results/paper-ready-figure/fig5/fig5-single-panels-png-pdf-svg.zip) · [自查结果](../results/paper-ready-figure/fig5/publication_qa.json)

原70点参数图已暂移出主图，完整保留在[parameter_scan](../results/paper-ready-figure/fig5/parameter_scan/README.md)；此前A–F整包见[归档](../results/paper-ready-figure/archive/2026-09-30_pre_transition_focus_fig5/fig5/README.md)。它仍是单种子有限观察窗扫描，不是统一3000秒随访或正式分岔。

本次8–11秒与颜色/读出修改之前的版本另存于[当日归档](../results/paper-ready-figure/archive/2026-09-30_pre_8to11_readout_fig5/fig5/README.md)。

恢复旧图P95定标之前、带新增均值轴的版本保存在[读出修改前归档](../results/paper-ready-figure/archive/2026-09-30_pre_p95_readout_fig5/fig5/README.md)。

改为未滤波显示之前的P95带通版及其producer保存在[未滤波修改前归档](../results/paper-ready-figure/archive/2026-09-30_pre_unfiltered_readout_fig5/fig5/README.md)。

此前main整合工作树的current_figure5记录的是本次调整之前的版本；本文件及主目录current_version.json记录此次作者要求的显示变更。原生Z/M、空间rate、六群体rate和Z/G/K闭环的身份与结论继续分别保留。当前导出已自查，待作者目视检查。
