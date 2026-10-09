# 当前 Figure 4：三行 A–I 重排版

2026-10-09按作者最新草图调整，状态为`CURRENT_AUTHOR_DESIGNATED`，本次完整拼版待作者目视检查。首行A扩大并预留右侧物理机制连接图，中排B–E、底排F–I对齐；旧E/F传播showcase移出，旧底排G/H/I/J依次改为F/G/H/I。数据保持冻结，不重跑实验。

**2026-10-09继续收紧横向留白并放大坐标框。** 按包含标签、图例及色条的实际可见边界调整，中排相邻空隙为B/C=3.30、C/D=3.87、D/E=13.69 mm，底排F/G=4.99、G/H=5.20、H/I=3.22 mm，全部小于上一版；C/D原15.82 mm、G/H原17.20 mm。省下的空间用于把B/C/D/E/H/I从45×45扩大到50×50 mm，C不再靠缩窄来满足对齐。D/H与E/I左右边界逐列对齐，B/C整体边界与F/G组合对齐；F为84×50 mm，G为27×50 mm。两处行间可见留白约9.65、7.40 mm，字号与科学数据保持；画布300×232 mm。B横轴仍严格采用[0,2]，Mean rank与Participation贴合左右数据边界，色条间距1.5 mm。

A小框中心为(-8.5,0) mm，保持原1.675×1.314 mm视野；小框至最近电极触点约3.52 mm，两条虚线至最近触点约3.04 mm，均避开电极读出范围。底图仍来自同一冻结坐标，虚线连接左右两框；该局部回路示意适用于整个模型空间，不限定在电极附近。方向椭圆、可选core位置及右侧机制连接图仍待补充，本次完整拼版待作者目视检查。

- 机器入口：[current_version.json](../results/paper-ready-figure/fig4/current_version.json)，`layout_version=compact_ai_tighter_columns_larger_squares_v4`。
- 完整图：[预览](../results/paper-ready-figure/fig4/figures/fig4-complete-layout-preview.png)、[PNG](../results/paper-ready-figure/fig4/figures/fig4-complete-layout.png)、[PDF](../results/paper-ready-figure/fig4/figures/fig4-complete-layout.pdf)、[SVG](../results/paper-ready-figure/fig4/figures/fig4-complete-layout.svg)。
- 当前producer：[build_fig4_compact_ai.py](../scripts/paper_figures/build_fig4_compact_ai.py)。重建候选：`python scripts/paper_figures/build_fig4_compact_ai.py`；核对后同步当前包：同命令加`--publish`。旧`build_fig4_current_aj.py`不再是当前入口。
- [面板对应表](../results/paper-ready-figure/fig4/panel_map.md)、[registry](../results/paper-ready-figure/fig4/figure4_panel_registry.json)、[布局与数值检查](../results/paper-ready-figure/fig4/visual_qa.json)。

| Panel | 当前内容与冻结来源 |
|---|---|
| A | 局部E/I回路与同一冻结空间基底；zoom-in小框置于(-8.5,0) mm左侧，虚线避开电极并连接两框，右侧留给后续机制图 |
| B | 原第6–16阶段194个工作点的三种误差；第1–5阶段32点不显示，保留开发先验来源；去除旧E/F标记 |
| C | E1146一至四core、每组4次优化重复，共同前31个epoch的最佳训练loss均值±样本标准差（ddof=1） |
| D | 原D中的E→E角度13点扫描，三种误差共享Error轴；移除旧showcase的参考线及E/F标签 |
| E | 原D最新完成的位置密度：418个完成、413个可评分配置中低J_joint前20%的83个，核宽0.35 mm |
| F | 原G：连续30–80 Hz虚拟接触活动，MTA/MTB事件阴影与通道顺序保留 |
| G | 原H：模型与患者平均传播rank |
| H | 原I：模型—患者触点交叉匹配矩阵 |
| I | 原J：25位患者分别展示TA–MTA／TB–MTB相似度，红／蓝；右上角Subject／Median一列两行图例 |

中排与底排数据轴共同高50 mm；B/C/D/E/H/I均为等大的50 mm正方形，B/F左边界、C/G右边界、D/H与E/I对应列对齐，所有角标按行统一。B仍用原紫色阶段配色；位置图保留直电极杆、触点、无框图例、90% mass轮廓与竖直等高色条。旧D中的向外EE及核内EE扫描随草图简化移出当前主图，保留在归档。

科学口径保持：B后续评分不使用TA/TB标签，但继承早期标签辅助开发的候选池与几何先验；位置密度受定向加密影响，不能解释为参数后验或唯一收敛。I仍为25位患者在冻结417条件快照中按训练J选点的描述性相似度，患者FIT数据参与训练与选点；不是独立验证，也未换成后台J_joint新结果。矩阵H的触点分折检验与I的全模板描述性读出分开。

重建前核对`panel_b/c/d/e/i`及布局版本，不按旧字母或candidate目录名寻找替代品。C与密度输入均从旧当前指针核验后固定，全部底排数值及50项队列相似度已核对；待人工目视验收不等于数据验收失败，也不等于A机制图已完成。

上一版完整A–J包及来源保留在[归档](../results/paper-ready-figure/archive/2026-10-09_pre_compact_ai_fig4/fig4/README.md)，旧版说明见[历史文本](../results/paper-ready-figure/archive/2026-10-09_pre_compact_ai_fig4/fig4/current_figure4_before_relayout.md)。早期单面板修改历史均不覆盖本次A–I编号与producer。

本次前一版A–I整包保留在[横向间距收紧前归档](../results/paper-ready-figure/archive/2026-10-09_pre_tighter_columns_fig4/fig4/README.md)；旧局部回路图及B–I科学数值保持。


**2026-10-09 A机制重绘候选v2（按最新反馈）**：恢复原左侧E/I回路和中间完整网络，前两个组件与当前正式v4逐像素一致。将二维椭圆连接与触点局部采样合并为第三张图，放大ICL3邻域；去掉新增公式、长短轴符号和下方说明。右侧从F已展示的MTA事件10中取ICL5→ICL4→ICL3→ICL2→ICL1真实片段，用波峰点及虚线显示传播序列，未平移通道时间或逐通道归一化。B–I也逐像素保持。[独立A](../results/paper-ready-figure/fig4/candidates/a_circuit_sampling_sequence_20261009/figures/fig4-panela.png) · [完整拼版](../results/paper-ready-figure/fig4/candidates/a_circuit_sampling_sequence_20261009/figures/fig4-complete-layout-preview.png)。当前正式版仍为v4，新A待作者目视检查；前一A候选保留作历史。候选producer为`build_fig4_a_mechanism_candidate.py`，独立A为`draw_fig4_a_spatial_readout.py`。第三、四图采用F工作点的真实几何、连接核和信号；当前读出仍是加权E发放密度。
