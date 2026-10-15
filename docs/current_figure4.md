# 当前 Figure 4：作者确认的 Paper Ready 定稿

2026-10-11作者明确指定本版为正式Paper Ready Figure 4，状态为`AUTHOR_ACCEPTED_FINAL`，版本为`core_search_final_20261011`，布局为`compact_ai_v4_a_core_search_central_shading_v15`。正式整图及A–I单图统一在`results/paper-ready-figure/fig4/figures/`，不存在待接受的A候选。

- [完整预览](../results/paper-ready-figure/fig4/figures/fig4-complete-layout-preview.png) · [PNG](../results/paper-ready-figure/fig4/figures/fig4-complete-layout.png) · [PDF](../results/paper-ready-figure/fig4/figures/fig4-complete-layout.pdf) · [SVG](../results/paper-ready-figure/fig4/figures/fig4-complete-layout.svg)
- [A–I单图说明](../results/paper-ready-figure/fig4/figures/README.md) · [机器入口](../results/paper-ready-figure/fig4/current_version.json) · [正式registry](../results/paper-ready-figure/fig4/figure4_panel_registry.json)
- [历史版本登记](archive/figures/figure4_revision_history_20261011.md) · [完整历史归档](../results/paper-ready-figure/archive/2026-10-11_fig4_finalization/README.md)

共享机器上的旧worktree也以`/home/honglab/leijiaxin/HFOsp/docs/current_figure4.md`和主目录`fig4/current_version.json`为准；离开共享机器时使用更新后checkout的同名文件。不能根据旧字母、candidate路径或旧producer推断当前版本。

A保留原局部E/I回路、中央二维sheet、右侧Local sampling及下方两组SEEG readout。中央有三个淡紫色问号区域与两个较深紫色core；上方两个问号区域分别以`OPTM`虚线箭头指向两个core，下方区域没有箭头和文字。最终core中心为(-6.6875,2.0625)和(6.4375,-4.5625) mm，对应E中两个边际位置密度峰平移(-10,-10) mm。

通用局部回路的zoom-in框仍在(-8.5,0) mm，远离电极。电极旁的中央取样框有浅绿底纹（alpha=0.22）；右侧Local sampling没有整框绿色背景，仅保留原σ=0.25 mm局部高斯采样权重。右侧3.8×1.9 mm视野内有五个错落椭圆连接核及连续触点SL3/SL4/SL5；下方两组120 ms冻结burst对应同样触点，源通道ICL3/ICL4/ICL5。电极杆保持刚直。左侧中央E/I竖线间距0.22示意单位，z↓采用0.60 pt、1.5/1.5细密虚线。

三行A–I、画布300×232 mm；B/C/D/E/H/I为50×50 mm正方形，D/H和E/I逐列对齐，B/C整体边界与F/G组合对齐。B类别轴为[0,2]、色条间距1.5 mm；横向留白与紧凑行距保持已接受版本。

| Panel | 当前内容与冻结来源 |
|---|---|
| A | 局部E/I回路、二维多core优化示意、局部椭圆连接与三个连续触点采样；下方MTA/MTB冻结burst读出 |
| B | 原第6–16阶段194个工作点的三种误差；第1–5阶段32点不显示，保留开发先验来源；去除旧E/F标记 |
| C | E1146一至四core、每组4次优化重复，共同前31个epoch的最佳训练loss均值±样本标准差（ddof=1） |
| D | 原D中的E→E角度13点扫描，三种误差共享Error轴；移除旧showcase的参考线及E/F标签 |
| E | 原D最新完成的位置密度：418个完成、413个可评分配置中低J_joint前20%的83个，核宽0.35 mm |
| F | 原G：连续30–80 Hz虚拟接触活动，MTA/MTB事件阴影与通道顺序保留 |
| G | 原H：模型与患者平均传播rank |
| H | 原I：模型—患者触点交叉匹配矩阵 |
| I | 原J：25位患者分别展示TA–MTA／TB–MTB相似度，红／蓝；右上角Subject／Median一列两行图例 |

A与完整拼图的PNG/PDF/SVG及预览逐字节匹配已确认v15；B–I的24个PNG/PDF/SVG逐字节保持原正式v10。此次只完成定稿登记和目录清理，没有重跑实验、改变数值或扩大科学结论。

三个候选区域、OPTM箭头及core数量变化为示意；两个边际密度峰不代表联合最优参数或临床SOZ。右侧读出为SNN发放密度代理，不是实测SEEG电压或新增电位前向模型。B继承早期标签辅助开发与患者几何先验；E为受定向加密影响的低损失搜索密度，不是参数后验。I为25患者冻结417条件快照中按训练J选点的描述性模板相似度，FIT数据参与训练与选点，不是独立验证；H的触点分折检验与I分开。

正式命令：`python scripts/paper_figures/build_fig4_current.py`，验证登记、数据合同与文件完整性。添加`--output-dir /tmp/fig4-export`可从随包`source/native_exports/`恢复31个已接受PNG/PDF/SVG，逐字节一致；无需旧候选、数据盘或旧worktree。这是冻结图文件恢复，不是从参数重绘或重跑SNN；原绘图代码及数值快照仍保留在`source/`，后续科学修改另开独立修订。

旧正式v10、58个历史候选、旧动画与重复导出全部归档至`results/paper-ready-figure/archive/2026-10-11_fig4_finalization/fig4/`。归档前后7,511个文件逐一校验一致；获接受的v15候选登记为`PROMOTED_SOURCE_ARCHIVED`，其余候选为`HISTORICAL_SUPERSEDED`。更早归档继续保留；目录内旧状态是历史记录，以当前指针及历史登记决定身份。
