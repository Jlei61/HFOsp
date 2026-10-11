# Figure 4 A–I 对应关系

当前身份：CURRENT_AUTHOR_DESIGNATED（用户指定，2026-09-28）。统一入口见 docs/current_figure4.md 与 fig4/current_version.json。

按 2026-09-23 用户提供的完整拼图核对。B 使用 9/18 的 226 个工作点 / 16 个阶段；C 使用 9/20 的 83 条件位置扫描，上方连接曲线未变。I 的原图在队列分析目录，本次一并导入 paper-ready 候选。

| 编号 | 子图标题 / 内容 | 原图 | 本次子图 |
|---|---|---|---|
| A | 局部 E/I 回路与患者电极空间基底 | [来源](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig4/figures/fig4-panela.png) | [PNG](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig4/candidates/complete_readability_20260923/figures/fig4-panela.png) |
| B | 参数搜索中的三类误差与搜索历程 | [来源](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig4/candidates/geometry_prior_patient_comparison_20260918/figures/fig4-panelb.png) | [PNG](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig4/candidates/complete_readability_20260923/figures/fig4-panelb.png) |
| C | EE 强度、方向及核位置的误差响应 | [来源](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig4/candidates/geometry_prior_expanded_response_20260920/figures/fig4-panela.png) | [PNG](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig4/candidates/complete_readability_20260923/figures/fig4-panelc.png) |
| D | 较高误差工作点的传播示例（lf_bo04_07） | [来源](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig4/candidates/geometry_prior_patient_comparison_20260918/figures/fig4-paneld.png) | [PNG](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig4/candidates/complete_readability_20260923/figures/fig4-paneld.png) |
| E | 较低误差工作点的传播示例（xy_left_20） | [来源](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig4/candidates/geometry_prior_patient_comparison_20260918/figures/fig4-panele.png) | [PNG](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig4/candidates/complete_readability_20260923/figures/fig4-panele.png) |
| F | 连续 30–80 Hz 虚拟接触活动 | [来源](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig4/candidates/geometry_prior_patient_comparison_20260918/figures/fig4-panelf.png) | [PNG](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig4/candidates/complete_readability_20260923/figures/fig4-panelf.png) |
| G | 模型与患者的平均传播 rank | [来源](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig4/candidates/geometry_prior_patient_comparison_20260918/figures/fig4-panelg.png) | [PNG](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig4/candidates/complete_readability_20260923/figures/fig4-panelg.png) |
| H | 模型—患者触点交叉匹配 | [来源](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig4/candidates/geometry_prior_patient_comparison_20260918/figures/fig4-panelh.png) | [PNG](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig4/candidates/complete_readability_20260923/figures/fig4-panelh.png) |
| I | 跨患者最佳训练 loss 随评估条件数的变化 | [来源](/data/hfosp/topic4_sef_hfo/cohort_three_observable_optimization_20260920/cohort_loss_improvement_20260923/figures/cohort_loss_trajectory.png) | [PNG](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig4/candidates/complete_readability_20260923/figures/fig4-paneli.png) |

A 保留局部回路内部文字，仅放大外部标题、坐标和 neuron 图例，并恢复完整的 0.65 pt 矢量虚线框；整图 285 × 264.5 mm 画布默认轴标签 10 pt，C/D/E 坐标标签增至 12 pt；普通刻度 9 pt、D/E 通道名 8.5 pt、F/G 通道名 7.5 pt。B 平行坐标恢复 42 × 45 mm，三维轴框恢复 51 × 51 mm；左侧纵轴为 Error、不带箭头，三维 Order error ε_o 使用随轴投影方向旋转的原生轴标签、紧邻刻度。G 图例以 7 pt 置于右上角，本次再上移 3 mm、跨在绘图区上边缘，其中 MTA/MTB 为模型实线、TA/TB 为患者虚线；F 图例以 8 pt 置于图内右上角覆盖少量曲线，取消此前为图例增加的纵轴留白；I 图例在图内，个体标签为 Subject，横轴按作者要求显示 Epochs；B 色条仍为 Iteration order；H 的 TA/TB、MTA/MTB 标签均加粗。

F/G/H/I 整行下移 6.5 mm，画布高度相应增加；与中排的可见留白匹配 A/B 至 C/D 的间距。H 左移 5 mm、I 左移 9 mm，收紧 H 两侧留白，I 绘图区距画布右边缘 11 mm。各绘图区尺寸保持不变，F/G/H/I 统一 40 mm 高且上下边界一致，H 保持正方形。C 上下两行分别与 D/E 包络和右侧两行原生快照总框等高并对齐，均为 40 mm；空间图保持等比例。验收数值来自渲染后的轴框，见 registry checks。

完整拼图与独立子图均提供 PNG/PDF/SVG；新增及放大的外部文字可编辑。D/E 在整图中共用 E 下方的横轴单位，E 右侧 x (mm) 居中于三列快照总框下方；独立 D 补回单位。旧正式图和原候选保留；本版为作者于 2026-09-28 指定的当前 Fig4；后续改图仍需目视检查。

短标记定义：B/C 的名称统一为 Mean rank error，共用 ε_r（平均 rank 误差）、ε_o（杆内先后概率误差），B 的 ε_p 为触点参与概率误差。B 平行坐标类别名下面和三维误差轴文字后面均标出对应符号，两处标签颜色统一：ε_r 深青灰 #3D5C6F、ε_o 珊瑚红 #E47159、ε_p 深橄榄绿 #6B7546；C 上方曲线与下方地图共用相同标记和误差配色：rank 为深青灰 #3D5C6F，order 为珊瑚红 #E47159，轴文字和曲线颜色一致。两个地图均从浅色渐变至对应参考色，范围保持 0–0.3。Outward E→E gain 是核内 E 神经元向核外 E 神经元的连接权重相对基线的乘数，1 表示基线。D/E 的 Ẽ 表示逐触点最大值归一化的模型包络；使用正文包络 E 的字母习惯，但此处是模型发放密度读出，不是患者电压或高频能量。当前本地 Fig2 canonical 仍为完整英文标签，未将本次短记号虚称为既有已锁定符号。

D/E 右侧 0–75 色标表示每 1 mm² 空间格、每 2 ms 时间窗内至少发放一次的 E 神经元数，色标上限仍为 75。来源 StreamingSpikes 使用 count > 0 后按网格计数；不是 Hz，也不是窗口内总 spike 次数。B 色条为参考紫色渐变的 Iteration order，曲线和三维散点共享该映射；B 刻度代表 16 个提案阶段。I 横轴按作者要求显示 Epochs，保留的刻度仍代表累计已评估条件的序号，不改写为梯度训练轮次；1/12/20/28 分别是基线、初始批次结束、两批自适应提案结束。I 的 Loss / baseline 保持原定义：每位患者截至该条件数的最低训练 loss 除以自身基线；Subject 为浅棕色（#D7B08A）个体轨迹，Median 为暖深棕色（#8C4B20）跨受试者中位数，图例同步。

F 约 396 ms 的 event 9 为模型 MTB（mode 0），补充与原 MTB 一致的蓝色背景 #287FA1、alpha 0.14；阴影为显示窗相对时间 336–426 ms，即参与触点招募起止 348–414 ms 的两端各加 12 ms。既有时间窗、事件分类和波形数组未改变。

科学读出保持原合同：F 为虚拟接触发放密度的带通读出；G 为冻结标签的平均 rank；H 是另一种触点交叉分组的匹配统计；I 是训练最优值轨迹，不据此增加机制恢复或优化器优越性的结论。
