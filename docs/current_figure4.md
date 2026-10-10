# 当前 Figure 4：三行 A–I 重排版

2026-10-10按作者要求将确认后的A机制图v10并入当前Figure 4，状态为`CURRENT_AUTHOR_DESIGNATED`；本次整体效果待作者目视检查。首行A展示回路、空间与采样读出，中排B–E、底排F–I对齐；B–I的24个PNG/PDF/SVG文件与上一正式版逐字节一致。旧E/F传播showcase移出，旧底排G/H/I/J依次改为F/G/H/I。数据保持冻结，不重跑实验。

**2026-10-09继续收紧横向留白并放大坐标框。** 按包含标签、图例及色条的实际可见边界调整，中排相邻空隙为B/C=3.30、C/D=3.87、D/E=13.69 mm，底排F/G=4.99、G/H=5.20、H/I=3.22 mm，全部小于上一版；C/D原15.82 mm、G/H原17.20 mm。省下的空间用于把B/C/D/E/H/I从45×45扩大到50×50 mm，C不再靠缩窄来满足对齐。D/H与E/I左右边界逐列对齐，B/C整体边界与F/G组合对齐；F为84×50 mm，G为27×50 mm。两处行间可见留白约6.20、7.40 mm，字号与科学数据保持；画布300×232 mm。B横轴仍严格采用[0,2]，Mean rank与Participation贴合左右数据边界，色条间距1.5 mm。

A包含放大的原局部E/I回路、中央二维sheet，以及右侧Local sampling和下方两组SEEG readout。左侧中央红色下行箭头与蓝色抑制竖线间距为0.22示意单位，z↓边框为0.60 pt、1.5/1.5细密虚线。

中央(-8.5,0) mm小框表示通用局部回路；电极附近仅有无填充灰色放大定位框。右侧在3.8×1.9 mm视野内显示五个错落的椭圆连接核和三个连续触点SL3/SL4/SL5，绿色高斯采样权重σ=0.25 mm仅在右侧显示。下方两组120 ms冻结burst与这些触点对应，源通道为ICL3/ICL4/ICL5。椭圆为局部连接核示意，读出是SNN发放密度代理，不是新增电位前向模型或实测SEEG电压。

- 机器入口：[current_version.json](../results/paper-ready-figure/fig4/current_version.json)，`layout_version=compact_ai_v4_a_left_circuit_spacing_v10`。
- 完整图：[预览](../results/paper-ready-figure/fig4/figures/fig4-complete-layout-preview.png)、[PNG](../results/paper-ready-figure/fig4/figures/fig4-complete-layout.png)、[PDF](../results/paper-ready-figure/fig4/figures/fig4-complete-layout.pdf)、[SVG](../results/paper-ready-figure/fig4/figures/fig4-complete-layout.svg)。
- 当前producer：[build_fig4_a_mechanism_candidate.py](../scripts/paper_figures/build_fig4_a_mechanism_candidate.py)。重建审阅包：`python scripts/paper_figures/build_fig4_a_mechanism_candidate.py`；脚本名含candidate不影响其当前producer身份。本次正式替换由`publish_fig4_a_mechanism.py`完成，旧版已归档；后续修改仍先核对当前指针，不能用旧compact或A–J发布入口覆盖。
- [面板对应表](../results/paper-ready-figure/fig4/panel_map.md)、[registry](../results/paper-ready-figure/fig4/figure4_panel_registry.json)、[布局与数值检查](../results/paper-ready-figure/fig4/visual_qa.json)。

| Panel | 当前内容与冻结来源 |
|---|---|
| A | 局部E/I回路、二维空间基底、局部椭圆连接与三个连续触点采样；下方为MTA/MTB两组冻结burst读出 |
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

重建前核对`panel_b/c/d/e/i`及布局版本，不按旧字母或candidate目录名寻找替代品。C与密度输入均从旧当前指针核验后固定，全部底排数值及50项队列相似度已核对；当前A已获作者指定并入整图，完整拼版仍待作者目视检查。

上一版完整A–J包及来源保留在[归档](../results/paper-ready-figure/archive/2026-10-09_pre_compact_ai_fig4/fig4/README.md)，旧版说明见[历史文本](../results/paper-ready-figure/archive/2026-10-09_pre_compact_ai_fig4/fig4/current_figure4_before_relayout.md)。早期单面板修改历史均不覆盖本次A–I编号与producer。

本次前一版A–I整包保留在[横向间距收紧前归档](../results/paper-ready-figure/archive/2026-10-09_pre_tighter_columns_fig4/fig4/README.md)；旧局部回路图及B–I科学数值保持。

**2026-10-10正式整合A v10**：左侧细节、中央sheet与右侧采样读出均采用作者刚确认的版本。正式入口已移除`pending_panel_a_revision`并登记`accepted_panel_a_revision`；A及完整PNG/PDF/SVG与v10审阅包一致。此前右侧留白的正式v4整包保存在[机制图整合前归档](../results/paper-ready-figure/archive/2026-10-10_pre_mechanism_fig4/fig4/README.md)。

**2026-10-10 A候选v13：淡紫色底纹、地图对齐与OPTM。** 三个问号候选区域加入淡紫色底纹（alpha=0.32），底纹与问号仍置于原神经元点下层。两个更新core改用E图每个core位置密度场的峰坐标：(3.3125,12.0625)/(16.4375,5.4375) mm，转换到A的坐标系后为(-6.6875,2.0625)/(6.4375,-4.5625) mm；采用物理平面位置，不跟随三维浮雕抬高后的屏幕峰顶。仅保留一条标有`OPTM`的虚线优化箭头，不逐一配对。候选区域、半径和箭头为示意；两个边际密度峰不代表一组联合最优配置，两组波形仍沿用原冻结工作点。中央sheet以外全图像素与正式v10一致，B–I不变。最新候选见`current_version.json.pending_panel_a_revision`，待作者目视检查；v11/v12保留作历史。[独立A](../results/paper-ready-figure/fig4/candidates/a_core_search_map_aligned_20261010/figures/fig4-panela.png) · [完整预览](../results/paper-ready-figure/fig4/candidates/a_core_search_map_aligned_20261010/figures/fig4-complete-layout-preview.png)。
