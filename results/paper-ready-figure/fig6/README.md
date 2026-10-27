# Figure 6（原 Figure 5）

2026-10-24 作者指定重新编号。当前入口见 [版本记录](current_version.json) 与 [当前说明](../../../docs/current_figure6.md)；以下原说明及内部 Fig5 文件名作为历史来源保留。

# Figure 5：间期活动至高活动态的过渡展示

2026-09-30按作者要求系统重排当前seed9108401空间Z/M Fig5。本图所有panel沿用同一条历史手放双核原生轨迹及原F模型/患者数据；Fig4 F2迁移另行在独立目录运行，不混入本图；参数扫描暂从主图移出，70点与19条长随访证据仍完整保留。

## 当前panel

| Panel | 内容 |
|---|---|
| A | 15触点未滤波突触电流读出（保留直流水平），按进入前第95百分位固定缩放，显示8.00–11.00秒 |
| B | 上行为Z、下行为有效adaptation ηM M；Z蓝色、ηM M橙色；All E实线、Core A虚线，两行与A严格共用时间轴 |
| C | 8.370–8.670及9.370–9.670秒的300 ms Core A/B E-only raster窗，编号1、2 |
| D | 9.420、9.480、9.600、10.370秒的四张50 ms原生E发放率快照，编号2–5 |
| E | 同一时间窗的原生Z–H_E–r_E轨迹，与A/B/D共享状态标记 |
| F | 原模型/患者早期宽频power-z场，置于E下方；数据与算法不变 |

删除原Rest状态；Interictal时刻仍以竖线标在A/B并保留C中栅格，不再单独画空间图。状态1改为窗内8.420秒的晚期间期事件；2、5保留原Pre-ictal/Onset，3、4改为更早的扩展阶段；对应表见[state_times.csv](state_times.csv)。A、B、E采用8–11秒，相当于进入起点后保留1.13秒；直接截取已存2000 Hz原始样本，不滤波、不去均值、不去趋势。A保留完整已记录频率内容与持续电平，F仍为原1–150 Hz CAR/基线robust-z，二者观测处理不同。

[完整PNG](figures/fig5-complete-layout.png) · [完整PDF](figures/fig5-complete-layout.pdf) · [单panel说明](figures/README.md) · [详细图注](fig5_caption_detailed_zh.md) · [全部单panel](fig5-single-panels-png-pdf-svg.zip) · [自查结果](publication_qa.json)

## 来源、验证与重建

`reference_fig5_metadata.json`冻结上一版的数据选择、F完整比较及原空间/栅格时窗。`fig5_metadata.json`记录本版映射、滤波、增益、真实快照、裁切及时间对齐；`source_snapshot.json`记录来源。已核对栅格中的固定神经元、第二窗时窗、原Pre-ictal/Onset快照逐格计数和F数据均不变；第一栅格窗移到8.370–8.670秒；中间两帧直接从原生计数生成，A/B三个轴横坐标完全一致。

单panel不带字母，完整拼版保留A–F；PNG/PDF/SVG原生导出。自查结果不代替作者目视验收，当前仍为待审候选。

当前 Fig6 已登记导出的校验与复制（不重跑仿真；历史原生重绘依赖共享机器来源）：

```bash
python scripts/paper_figures/build_fig6_current.py --output-dir /tmp/fig6-exports
```

原含70点参数图的整包归档至 `../archive/2026-09-30_pre_transition_focus_fig5/fig5/`；参数图和配套长时数据另保留于[parameter_scan](parameter_scan/README.md)，不再使用当前E的panel身份。其他独立候选保留原目录。

## 与Fig4 F2的区别

本图基于历史手放核场，不能称为Fig4 F2的仿真。独立F2 实验（共享机器目录 `results/topic4_sef_hfo/fig5_f2_native_zm_20260930/`）保持`xy_left_20`实际执行参数，先做无Z/M前缀复现，再加原Z/M观察30秒；只用一个固定种子。

本次左列宽度占比缩小，B两行压扁并靠拢，A–B留白增加；所有A/B状态仍严格共用横坐标。

## A的读出显示规则

按作者要求，直接显示15触点保存的原始电流proxy，保留8–11秒；不进行30–80 Hz或其他频带滤波，不去均值、不去趋势、不作共同平均参考、不作包络或平滑。按0–9.420秒原始信号的绝对值第95百分位作每触点固定缩放，分母下限仍为跨触点正P95中位数的15%，幅度/通道间距比为0.68/1.48；逐通道纵向错开只为排版。保存采样率为2000 Hz，绘出窗内全部采样点，不能据此声称读到了采样带宽之外的信号。

该读出为附近E细胞收到的兴奋和抑制突触电流绝对值之和的距离加权平均，在Z缩放之前记录；它不是局部神经元放电率，也不是患者实测SEEG。进入后持续电平明显升高，但高电平不要求伴随大幅周期波动；原始与带通数据的比较见[读出诊断](readout_audit/readout_audit.json)。主图A只有一个轴，B–F数据与显示保持不变。详细数值复核见[原始读出检查](validation/unfiltered_readout_check.json)。
