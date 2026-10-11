# Paper-ready figure 唯一登记表

> **当前Fig3：作者已确认的A–E正式版（2026-10-10）。** [完整图及独立面板](../results/paper-ready-figure/fig3/figures/README.md)为`visual_alignment_compact_rows_20261010`、`AUTHOR_ACCEPTED_FINAL`；[当前说明](current_figure3.md)与[机器入口](../results/paper-ready-figure/fig3/current_revision.json)为统一入口。A为E10/SZ3，B为Y1/SZ6固定0–10 s、30–80 Hz；C/D/E承接原D、原E右半、原F。完整可见列宽包含色条及标签，两处行距均4.064 mm，画布308.11×254.24 mm。正式代码`build_fig3_current.py`从随包冻结原生图层重建，无需原始数据盘；旧A–F及原组装代码已归档，不能覆盖本版。

> **当前Fig1：作者已确认的完整定稿（2026-10-10，`y1_final_20261010`）。** 正式[完整图、A–F单独面板及7页PPTX](../results/paper-ready-figure/fig1/figures/README.md)统一位于`fig1/figures/`；[当前版本说明](current_figure1.md)和`fig1/current_revision.json`为共享入口。保留Y1真实脑图、A7/A9、18通道显示rank、原完整事件S³质心和40人统计，E/F整行上移6.35 mm的紧凑行距已接受。当前producer为`build_fig1_current.py`，冻结科学代码、必要数据备份和来源对应表随包保存；完整PNG/PDF可逐字节复现。此前整个fig1目录已归档至`archive/2026-10-10_fig1_finalization/fig1/`，[历史修订登记](archive/figures/figure1_revision_history_20261010.md)仅供追溯。

> **当前 Fig4（A机制图整合，2026-10-10）：三行A–I，`CURRENT_AUTHOR_DESIGNATED`。** 见[版本说明](current_figure4.md)及[机器入口](../results/paper-ready-figure/fig4/current_version.json)，producer为`build_fig4_a_mechanism_candidate.py`。A小框移至(-8.5,0) mm且虚线避开电极；B/C/D/E/H/I扩大到50×50 mm，C/D与G/H横向留白约3.87／5.20 mm，D/H与E/I逐列对齐、B/C整体与F/G组合对齐，B两端无空白延伸且色条间距1.5 mm，紧凑行距保持；右侧Local sampling及两组SEEG readout已并入；中排B/C/角度响应D/位置密度E，底排沿用旧G–J改为F–I；旧E/F传播showcase移出。B为第6–16阶段194点、C为四core均值±样本标准差、E为83/413低损失配置密度、I为25患者红蓝相似度，均保留最新冻结数据。当前拼版待作者目视检查；下方历史A–J说明不覆盖当前入口。

> A机制图v10（2026-10-10）已按作者要求并入当前正式包，登记于`fig4/current_version.json.accepted_panel_a_revision`；见[当前说明](current_figure4.md)。A包含放大的原局部E/I回路、中央二维sheet，以及右侧Local sampling和下方两组SEEG readout。左侧中央红色下行箭头与蓝色抑制竖线间距为0.22示意单位，z↓边框为0.60 pt、1.5/1.5细密虚线。中央(-8.5,0) mm小框表示通用局部回路；电极附近仅有无填充灰色放大定位框。右侧在3.8×1.9 mm视野内显示五个错落的椭圆连接核和三个连续触点SL3/SL4/SL5，绿色高斯采样权重σ=0.25 mm仅在右侧显示。下方两组120 ms冻结burst与这些触点对应，源通道为ICL3/ICL4/ICL5。椭圆为局部连接核示意，读出是SNN发放密度代理，不是新增电位前向模型或实测SEEG电压。B–I的24个PNG/PDF/SVG文件保持不变，完整拼版待作者目视检查。

> A候选v15（2026-10-11）：保留上方两条`OPTM`箭头及三个问号区域，下方不加箭头或OPTM文字；浅绿底纹仅在中央取样小框显示。右侧Local sampling恢复原背景及局部高斯采样权重，中央sheet以外全图像素与正式v10一致。替代v14，正式v10输出保持；见`fig4/current_version.json.pending_panel_a_revision`及[当前说明](current_figure4.md)，待作者目视检查。

> **Figure 4/SNN 设计口径（2026-09-09）**：新一轮采用[患者传播几何先验下的网络拟合](topic4_patient_geometry_prior_snn.md)，展示几何初始化、分布拟合及参数 × 观测作用。此为设计更新；下表现有 Figure 4/5 的资产身份和验收状态保持各自记录，运行中的 pilot 尚未替换正式图。

> 2026-09-18 新增 Figure 4 候选：患者几何先验下的参数响应、完整搜索历程和两个工作点的传播示例，独立 A–D 及完整拼版见下表。候选保存本次指定图的固定快照，扩大 XY 范围的运行不会自动覆盖它。

> 2026-09-23 新增作者 A–I 拼图的可读性候选：[完整图与独立子图](../results/paper-ready-figure/fig4/candidates/complete_readability_20260923/figures/README.md)，[编号、标题与原图对应](../results/paper-ready-figure/fig4/candidates/complete_readability_20260923/panel_map.md)。复用截图的组合：A 为正式机制/基底，B 为 9/18 的 16 阶段搜索，C 为 9/20 扩展位置响应，D–H 为 9/18 患者比较，I 为 9/23 队列训练 loss。统一完整画布字号，F/G 图例按最新反馈放入图内右上角，D/E 使用语义色 MTA/MTB，B 轴交点连合；完整图及 A–I 独立 PNG/PDF/SVG 均已生成。旧正式资产保留，本版待作者目视验收。

> 同日按作者进一步反馈修订：A 只放大外部 Local E/I circuit 标题、x/y 坐标文字与刻度、E/I neuron 图例，左侧 E→E/I→E 等内部文字和局部回路物理尺寸不变；B 恢复原轴框比例。C 使用 Outward E→E gain，两个双轴图均显示完整左右刻度；地图使用与上方误差轴对应的 ε_r/ε_o，D/E 包络使用 Ẽ，右侧原生场明确标为 E cells / mm² / 2 ms（每格每窗内至少发放一次的 E 神经元数量）。F/I 图例在图内，I 简化为 Patient / Median 和 Loss / baseline，H 使用 Spearman ρ 标题。F/G/H/I 上下边界一致，H 保持正方形；C 上下两行分别与 D/E 包络及右侧两行快照总框等高并对齐，各行为 40 mm。以实际渲染轴框验证；本版待作者目视验收。
> 最新细调：A 左侧外框补足裁切边距并增加 0.65 pt 矢量虚线，保证顶部边框在整图中可见；B 的平行坐标和三维误差轴统一使用 ε_r、ε_o、ε_p（rank、杆内顺序、参与误差）。C 与 D/E 共享坐标标签增至 12 pt，D/E 通道刻度增至 8.5 pt；数据轴比例和等高关系不变。随后按作者反馈将 H 的 TA/TB 标签加粗；G 图例缩至 7 pt 并放入图内右上角，F 图例以 8 pt 覆盖右上角少量曲线，取消为其增加的纵轴留白。I 按作者参考图改为浅紫色（#B1A9DA）患者轨迹和深紫色（#483D8B）中位数，图例同步。C 的 rank/order 误差最终使用深青灰 #3D5C6F / 珊瑚红 #E47159，曲线、轴文字及地图色条同步。H 的 MTA/MTB 纵轴标签也已加粗；I 横轴与 B 色条统一为 Iteration order，I 的刻度仍为累计评估条件序号 1/12/20/28，B 为提案阶段序号。E 右侧 x (mm) 已移至三列快照总框的水平中心。 G 图例随后再上移 3 mm，跨在绘图区上边缘，保持原轴框高度；B 横轴与三维轴标签统一按误差语义着色，ε_r 深青灰 #3D5C6F、ε_o 珊瑚红 #E47159（与 C 一致）、ε_p 深橄榄绿 #6B7546。B 左侧纵轴改为 Error、不带箭头，三维 Order error ε_o 随对应坐标轴旋转并紧邻刻度。 底排 F/G/H/I 在完整拼图中整体下移 6.5 mm，画布加高以保持字号和绘图区大小，两段行间可见留白匹配；H 左移 5 mm、I 左移 9 mm，收紧 H 周围占位，I 绘图区右侧保留 11 mm 边距。
>
> 2026-09-24 作者细调：B 的迭代色条、对应曲线和三维散点最终改用参考紫色渐变（#B1A9DA → #7367BE → #483D8B）；B/C 的 rank 误差名称统一为 Mean rank error，语义色保持一致，B 三维轴标签加粗。F 中约 396 ms 的 event 9 经源模型标签确认属于 MTB，补上相同蓝色、透明度及招募起止两端各加 12 ms 的阴影。I 图例改为 Subject / Median，分别使用浅绿 #A1D99B 与深绿 #006D2C，横轴显示 Epochs；数值仍为累计已评估条件序号 1/12/20/28。
>
> 2026-09-28 作者配色调整：I 使用接近橙色的暖棕色系，Subject 为浅棕 #D7B08A，Median 为深棕 #8C4B20；曲线、标记及图例同步，布局与数值保持不变。
>
> 状态：v9 主表及后续逐项更新。本文是 `results/paper-ready-figure/` 的唯一指代入口。
> 论文 panel 编号、资产语义、物理路径和科学状态必须分开记录；不能再仅凭目录名里的
> `fig2` / `fig3b` / `fig6` 判断它现在属于哪张论文图。

## 1. 指代规则

每个可引用图包使用四个字段：

1. `asset_id`：稳定语义 ID。只描述图在展示什么，不包含可能变化的 panel 编号。
2. `paper_slot`：当前论文位置，例如 `Fig3-A`。位置未锁定时写 `TBD`。
3. `status`：科学和出版状态。
4. `canonical_path`：当前唯一可消费路径。

状态按以下类别记录：

- `AUTHOR_ACCEPTED_FINAL`：作者明确确认并指定为当前正式版本。
- `LOCKED`：科学合同和可视版本均已锁定，可进入当前稿件。
- `CANDIDATE`：可作为候选，但总拼版或 panel 位置尚未最终锁定。
- `SUPPLEMENT`：只进入补图或补充计算材料。
- `SOURCE`：正式 panel 的源素材或公共 renderer 输出，不能单独当论文结论引用。
- `HISTORICAL`：已撤回、被替代或仅保留模型谱系；只能从 `archive/` 引用。

引用时优先写 `asset_id (paper_slot)`，例如
`ictal_phenotype_to_field_layout (Fig3-A–E)`，不要只写“fig3 图”或直接沿用脚本内部的 `fig3c_*` 名称。

## 2. 当前主文资产映射

| asset_id | paper_slot | status | canonical_path | producer / 说明 |
|---|---|---|---|---|
| `interictal_hfo_temporal_scaffold` | Fig1-A–F | `AUTHOR_ACCEPTED_FINAL` | `results/paper-ready-figure/fig1/figures/` | 作者2026-10-10接受Y1完整A–F；PNG/PDF及7页PPTX。当前代码、约13 MB科学输入备份和历史归档见`docs/current_figure1.md`。 |
| `interictal_spatial_scaffold` | Fig2-A–F | `CANDIDATE` | `results/paper-ready-figure/fig2/figures/` | 无角标 A–F 独立输出 + 带 A–F 角标的 `fig2-complete-layout`。2026-09-02 保持原拼板位置，仅收紧 A 的 2×2 内部留白、放大 B 主体文字、将 C 行标题改为 `TA samples` / `TB samples`；D 保持左侧 `TA field` / `TB field`。B/E/F 仍使用 all-event Timing+Space 合同 |
| `interictal_sequence_to_spatial_field_layout` | Fig2-A–D重排候选 | `PENDING_AUTHOR_VISUAL_REVIEW` | `results/paper-ready-figure/fig2/candidates/sequence_to_spatial_field_20261009/figures/` | 2026-10-09按作者示意：新A合并原C单事件readout、原A的E10局部解剖/二维投影、原D模板场；B原样保留；原E/F编号为C/D。1D/2D暂名Electrode-sequence template／Spatial propagation field，仍区分samples与冻结模板；四例沿用当前E1/E12/E5/Y9。矢量拼版producer为`build_fig2_sequence_field_layout.py`，源文件与metadata冻结在候选目录的`source_snapshot/`；旧A–F保留，待作者目视检查。 |
| `interictal_exemplar_rank_trajectory` | Fig2-A示例轨迹候选 | `PENDING_AUTHOR_VISUAL_REVIEW` | `results/paper-ready-figure/fig2/candidates/exemplar_rank_trajectory_20261009/figures/` | 2026-10-09作者认可前述四面板版式后，要求先给E10增加方向轨迹。左侧方向箭头拟合原单事件time–contact-row斜率，按作者反馈水平左移45 ms显示距离、上下各缩进一个通道，线杆与箭头头部合为一个轮廓；右侧TA红色／TB蓝色曲线为冻结rank场的early→late局部梯度流线，两模板共用由采样凸包面积确定的几何中心，沿用6 mm显示核并限制在凸包及display support内。不是唯一物理轨迹或速度估计；仅更新A候选及二维放大图，C其他患者暂未修改。最新按作者要求统一四张数据图的轴框为132 × 132 pt，上下两行左右严格对齐，图上方短标签改为TA/TB seq.和TA/TB traj.。左列仅底部显示Time (ms)，与右列shared TA axis (mm)共用380 pt文字基线；右侧字号独立缩小，横轴标签宽度限制在132 pt内。中间上方按2026-10-10最新反馈改为同一E10三维MRI中的斜切面：直接在冻结平面上采样T1纹理，保留真实ICL/SCL触点，平面内叠加两条白色TA/TB early→late方向箭头；两者来自冻结三维轴，法向分量均小于1e-10。浅虚线连接下方展开平面，保留向右转换箭头；按作者最新反馈移除“+ spatial information”文字与R/A/S坐标标记，切面改为与下方展开图呼应的浅青色并保留真实T1纹理，保留30%不透明度；三张正交MRI改在冻结分析原点相交，修正此前按全部局部触点均值取交点造成的约3.69 mm偏移。按原始LAS（左、前、上）方向显示，使用正交相机（elevation 22°、azimuth 45°、roll 0°、S向上），撤回透视相机；三张MRI组织切片均为100%不透明，只有分析平面保持30%不透明度。分析平面的完整青色（#4F8697）矢量轮廓加粗至2.1 pt，取消额外青色交线；体素LAS经原affine定位到物理RAS，绘图时只转换显示轴，电极、分析平面和模板轴不平移或重拟合。下方标签改为Early Projection，投影图底部裁切范围补足2 pt，并沿原四边坐标重画同色同宽的完整矩形边框，修复源图内部裁切造成的浅底边；原比例保持。按作者最新反馈找回旧双向箭头代码，白箭头分别从各自冻结p_early早端出发，不再共用尾端；末端只为避让触点沿原轴截短至至少2.5 mm间距。早端为模板轴摘要，不代表已定位的生物学源点，也不把1D时间/通道斜率当作3D角度。空间曲线按作者最新参考图改为线身连续收束到细长尖端的一体化轮廓，去掉中段箭头；每条曲线只有末端一个指向，彩色线芯与白色外沿各为一个完整填充轮廓，不再叠加独立箭头；仅为避让电极符号，把距触点中心不足2 mm的显示末端沿原曲线截短（本例TB晚端截短约2 mm）。作者澄清仅调整箭头／轨迹颜色：已恢复原viridis场、触点填色及色标，撤回浅灰蓝场版本；左侧方向箭头采用浅玫红（#EF8FB3）填充和白色细边，右侧TA／TB轨迹按2026-10-10最新反馈改为明亮珊瑚红（#F07862）／晴蓝（#55ADE9）线芯，移除深石墨蓝描边，仅保留加粗纯白（#FFFFFF）外沿；主图源坐标线芯宽1.3 pt、白色外轮廓的线身宽3.3 pt，单侧露白1 pt；末端尖部的底边放在距显示末端7 pt弧长的位置，外轮廓宽5.4 pt；头部统一用同一组方向与法向量构成平面三角形，修正原来头部沿曲线逐点旋转造成的扭折。外轮廓尖端仍在原显示终点，彩色尖端沿头部中轴缩进0.8 pt以留白，白色底层与彩色层各为完整轮廓，保持连接连续且无深色内线。完整积分曲线、通道顺序与1D拟合参数保持不变，显示末端不是生物学传播终点；producer为`plot_fig2_exemplar_trajectory.py`，三维切面helper为`fig2_embedded_plane.py`；新A仍待作者目视检查。 |
| `interictal_mri_orthogonal_orientation_check` | Fig2-A MRI参考观察方向核对 | `PENDING_AUTHOR_VISUAL_REVIEW` | `results/paper-ready-figure/fig2/candidates/mri_reference_orientation_20261010/figures/` | 2026-10-10按作者提供参考图调整观察方向：相机方位55°、仰角30°，近似匹配S向上、R左下、A右下；保持MNI/RAS原点(0,0,0) mm，同一T1/affine及切片数据。参考查看器R(+X)、S(+Y)、A(-Z)与本图RAS间是正规旋转，不交换解剖轴或镜像体数据。二维核对图统一采用参考的R=R（患者右侧在图右侧），矢状位前方在左。仅显示三张不透明MRI组织切片及真实交线，不含分析平面、电极或模板箭头；坐标与交线强度验证通过，PNG/PDF与二维解剖轮廓已由Agent检查，待作者检查。此前`mri_default_ras_origin_20261010/`版本保留，完整A候选未替换；producer为`plot_fig2_mri_orthogonal_views.py`。 |
| `interictal_mri_ra_plane_check` | Fig2-A RA平面独立核对 | `PENDING_AUTHOR_VISUAL_REVIEW` | `results/paper-ready-figure/fig2/candidates/mri_ra_plane_check_20261010/figures/` | 2026-10-10作者质疑RA平面后单独核对S=0的轴位纹理，保留当前相机及RAS原点。原始二维轴位与独立正向投影并排显示同一+R/+A坐标箭头，并输出隐藏另外两切面后的原渲染器结果；未发现R/A互换或前后翻转，独立坐标反解误差小于2e-14 mm。此项不代表作者对三维合成观感的认可；原三切面输出逐像素不变。PNG/PDF已检查，producer为`check_fig2_ra_plane.py`。 |
| `interictal_clear_mri_context` | Fig2-A MRI背景简化候选 | `PENDING_AUTHOR_VISUAL_REVIEW` | `results/paper-ready-figure/fig2/candidates/clear_mri_context_20261011/figures/` | 2026-10-11按作者要求简化遮挡：比较四种MRI背景后，选择后外侧观察（方位-125°、仰角25°），保留S≥0的上半张矢状面与A≥0的远侧半张轴位面，MRI原始交点保持RAS(0,0,0)。MRI组织保持100%不透明并适度减弱对比，真实分析切片30%不透明、青色边框突出；前方/下方背景移除，渲染检查中分析切片无遮挡。电极、分析平面与T1均沿用冻结输入，无平移、镜像或重拟合；去掉中间重复的TA/TB和杆名文字、方向箭头，触点细节由下方投影承接。已拼回A候选，左右数据图、色标及下方投影逐像素一致，仅改变上方MRI及相连虚线。PNG/PDF已检查，待作者目视验收；原候选及正式图未替换。producer为`build_fig2_clear_anatomical_context.py --compose --style open_two_planes`，同目录有四方案对照。 |
| `ictal_phenotype_to_field_layout` | Fig3-A–E | `AUTHOR_ACCEPTED_FINAL` | `results/paper-ready-figure/fig3/figures/` | 2026-10-10作者正式接受；E10/SZ3 broadband与Y1/SZ6 gamma两行均为raw/TFR、早期能量场、间期TA场。Y1固定0–10 s、30–80 Hz、自身TA平面、Y=−20～20 mm；C/D/E为原D、原E右半、原F。含色条的完整列宽对齐，行距4.064 mm。代码、来源与远端重建见`docs/current_figure3.md`。 |
| `ictal_field_scaffold` | 历史Fig3-A–F | `HISTORICAL` | `results/paper-ready-figure/archive/2026-10-10_pre_current_fig3_ae/fig3/figures/` | 2026-10-10被当前A–E正式版替代；旧独立面板、整图、metadata和组装代码保留供追溯。旧图号不是当前panel身份。 |
| `patient_geometry_prior_snn_compact_a_i` | **Fig4-A–I 当前版** | `CURRENT_AUTHOR_DESIGNATED` | `results/paper-ready-figure/fig4/figures/` | 2026-10-09按作者草图重排，对齐行列并移除旧E/F showcase。中排B/C/D/E为后先验搜索／core数量／角度响应／位置密度，底排旧G–J改为F–I；A框移至(-8.5,0) mm，虚线避开电极；B/C/D/E/H/I扩大到50 mm正方形，相邻横向留白收紧，B两端留白清除，紧凑行距保持；右侧机制图待补充。科学数据保持，见current_version.json与docs/current_figure4.md；人工目视待验收 |
| `fig4_core_count_mean_sd_c` | Fig4-C当前数据及样式来源 | `SOURCE_FOR_CURRENT_FIG4` | `results/paper-ready-figure/fig4/candidates/core_count_mean_sd_20261008/figures/e1146_one_two_three_four_core_loss.png` | 2026-10-08用户确认并要求同步到正式C；四种颜色、细线小点及均值±样本标准差（ddof=1），无单次轨迹、标题或图注，轴为Epochs／Loss。3／4 core两轮反馈优化完成；阴影表示优化重复间差异，独立噪声复测暂缓。完整拼版现为candidates/complete_aj_d_frameless_20261009/；B仅显示原第6–16阶段 |
| `fig4_core_count_panel_c` | A–J布局与旧C来源 | `SOURCE_FOR_CURRENT_FIG4` | `results/paper-ready-figure/fig4/candidates/core_count_panel_c_20260929/figures/` | 原C为E1146单核／双核初始采样loss，已由2026-10-08四组均值±标准差版替代；保留A–J布局来源。B仅保留左侧曲线及紫色色条，原C–I顺延为D–J。本目录D已作为2026-10-08正式D的数据及样式来源，正式图以上行机器入口为准 |
| `fig4_parameter_response_d_revision` | Fig4-D响应曲线数据及样式来源 | `SOURCE_FOR_CURRENT_FIG4` | `results/paper-ready-figure/fig4/candidates/core_count_panel_c_20260929/figures/fig4-paneld.png` | 三张曲线统一左侧Error轴与Mean rank／Order／Participation图例，核内EE七点扫描及E/F实际参数线保持原样。此来源中的历史52配置位置图已由2026-10-09完成快照替代；正式D需同时读取panel_d.density_source_package，不能整图回退至此旧来源。 |
| `fig4_completed_position_density_d` | Fig4-D当前位置密度来源 | `SOURCE_FOR_CURRENT_FIG4` | `results/paper-ready-figure/fig4/candidates/core_position_densification_complete_20261009/figures/core-position-search-context.png` | 2026-10-09作者确认并接入正式D；128个新增配置全部完成，合并历史池后418个完成、413个可评分，低J_joint前20%=83个（59历史+24新增）。0.35 mm高斯核、单色立体曲面、90%轮廓及直杆，触点置顶；主图40 × 40 mm、竖向等高色条。只描述搜索池低损失位置集中，局部加密影响密度，其他低损失区域保留；主图拼版待作者目视检查。 |
| `fig4_column_alignment_revision` | A/B/C/H列对齐候选 | `CANDIDATE_PENDING_AUTHOR_VISUAL_REVIEW` | `results/paper-ready-figure/fig4/candidates/core_count_panel_c_20260929/figures/fig4-complete-layout-preview.png` | A收窄至与D实际可见范围等宽，B/C共享空间并统一55.5 × 48.288 mm，与A空间图横轴和上边缘对齐；B/E/F/H绘图区及角标分别对齐。此整体列重排仍为候选；D内容已单独进入正式版。 |
| `data_driven_interictal_snn_fig4` | Fig4-A–G historical | `HISTORICAL_SOURCE_ONLY` | `results/paper-ready-figure/archive/2026-09-29_pre_template_recovery_fig4/fig4/figures/` | 无角标 A、C–G 独立 PNG/PDF + 带 A–G 角标的 `fig4-complete-layout`；B 使用右上现有留白并明确预留，当前无独立文件，后续用于 data-driven 参数对患者间期事件复现的影响。A 为已确认的 local E/I circuit + patient-specific substrate；C 为 Node field + Model TA/MTB；D 为模型/患者 rank profile；E 为 cross-fit matrix；F 为 recruitment-onset-span readout；G 为 34 人 formal cohort。原 KMeans heatmap/rank distribution 保留在 FigS7-E。科学口径不扩大 |
| `patient_geometry_prior_snn_parameter_response` | Fig4-A–D historical candidate | `SUPERSEDED_CANDIDATE` | `results/paper-ready-figure/fig4/candidates/geometry_prior_parameter_response_20260918/figures/` | 保留 2026-09-18 首版 A–D 候选及完整拼版；当前候选使用下一行的 D/E 重编号和新增 F/G/H。首版数据与文件未改动 |
| `patient_geometry_prior_snn_patient_comparison` | Fig4-A/B/D/E/F/G/H source | `SOURCE_FOR_CURRENT_FIG4` | `results/paper-ready-figure/fig4/candidates/geometry_prior_patient_comparison_20260918/figures/` | 2026-09-18 历史候选：B 标记 C/D→D/E，D=lf_bo04_07，E=xy_left_20；A 原样保留。F/G/H 均使用 E 的单拓扑2511/噪声847401、151个事件。F 为连续30–80 Hz接触发放密度读出；G 为冻结模型标签与患者 Timing+Space 标签下的平均rank；H 为重新计算的触点交叉匹配及杆内置换，不能当作新患者验证或机制恢复。独立PNG/SVG，无角标或总标题；当时未定义新C；当前A–J已由上行完整包替代。保留旧正式A–G资产，详细来源见同级registry与metadata |
| `patient_geometry_prior_snn_expanded_response` | Fig4-A/B alternative/source | `SOURCE_OR_ALTERNATIVE` | `results/paper-ready-figure/fig4/candidates/geometry_prior_expanded_response_20260920/figures/` | 2026-09-20 更新上一候选的A/B：A位置网格扩大为83个条件，另有独立放大图；B接入两轮扩大范围的82条新轨迹，由226→308个工作点、16→19个真实提案阶段。D/E标记仍对应上一候选的同一示例，D/E/F/G/H沿用上一目录。新旧误差使用相同FIT19770参考；颜色表示生成阶段，不是单调收敛。两条内存保护中断轨迹已补完，夜间实际42条完成，另8条未派发。PNG/SVG经Agent检查，待人工目视验收；不替换正式A–G或声称双模式恢复 |
| `data_driven_snn_dual_mode_validation` | Fig4-C historical candidate | `DIAGNOSTIC_ONLY` | `results/topic4_sef_hfo/data_driven_core_field_rev10_d/spatial_ou_accessibility_d5_2_confirmation/figures/` | 连续场 + MTA/MTB same-network readout 和 KMeans 核验图；natural KMeans 未复现患者 TA/TB（direction purity `0.674` < patient-matched q05 `0.884`，pooled MTB↔TB `−0.60`），已被独立冻结终点的 NLC pathway panel 替代，不得再作为当前 Fig.4C source |
| `data_driven_snn_d6_3_replication_diagnostic` | Fig4-C source（诊断，不可替换主文） | `DIAGNOSTIC_ONLY` | `results/topic4_sef_hfo/data_driven_core_field_rev10_d/continuous_field_kmeans_d6_3_fresh_replication/figures/` | 同规格第二套图，冻结连续场候选 `d62_a0p5_b0p5` 在 12 张全新网络上的复制臂；verdict `REV10D6_3_JOINT_CONTINUOUS_FIELD_NOT_REPLICATED`（patient cross-fit paired delta `−0.107`、K=2 支持仅 `4/12`）。图为 pooled 展示，不覆盖网络级复制失败，**不得替换主文 Fig.4** |
| `single_seed_zm_entry_and_onset_field` | Fig5-A–F | `CANDIDATE` | `results/paper-ready-figure/fig5/figures/` | 2026-09-30更新：A为15触点未滤波原始电流读出，保留持续电平，按进入前原始信号P95固定定标；B为蓝色Z、橙色adaptation，All E实线/Core A虚线，A/B/E共用8–11秒；C为8.370–8.670和9.370–9.670秒E-only raster；D为9.420/9.480/9.600/10.370秒的原生50 ms空间图；E状态标记同步；F仍为原模型/患者能量场。本图为历史手放核场，Fig4 F2+Z/M另行单种子运行。原70点参数图留于parameter_scan。producer为build_fig5_transition_focus.py；图注、读出诊断、PNG/PDF/SVG及单panel包已自查，待作者目视检查 |
| `data_driven_zm_transition_fig5` | Fig5-A–D historical | `HISTORICAL` | `results/paper-ready-figure/archive/2026-09-18_pre_single_seed_fig5/fig5/figures/` | 原seed1801 v5整包保留，已由上行A–F替换；其原`NOT_SUSTAINED_ICTAL_MORPHOLOGY`标签仅属于该历史版本，不移植到新轨迹 |

## 3. 当前补图 / 计算材料

| asset_id | paper_slot | status | canonical_path | 说明 |
|---|---|---|---|---|
| `interictal_single_event_propagation_video` | Supplementary Video 1 | `SUPPLEMENT` | `results/paper-ready-figure/supplementary-video-1.gif` | 2026-08-19 作者锁定，2026-08-30 v2 将最右 template-rank colorbar 统一为各模板内归一化 `0–1`。E10 单次 TA/TB 代表事件，30 帧、2 ms biological step、12.5 fps（80 ms/frame）；soft teal-to-navy HFO amplitude envelope + 冻结 template-rank field。只支持 frozen-axis 上的 representative raw-envelope timing cross-check，不支持 template-free、cohort、continuous-tissue traveling wave 或机制结论 |
| `fig3c_peri_onset_field_evolution_video` | Supplementary Video 2 | `SUPPLEMENT` | `results/paper-ready-figure/supplementary-video-2.gif` | 2026-09-03 作者锁定。E10/SZ3 的场动态配套（旧Figure 3C，现Figure 3A场部分）：固定 TA 间期场，右侧显示同一 shared plane 上从 −120 s 至 +20 s 的 1–150 Hz baseline-robust-z 发作场；下方为幅度感知模板表达量 `Q=max(|q_A|,|q_B|)` 与当前窗游标，红蓝点表示 TA/TB 主导。66 帧、10 s 滑窗、2 s 步长、固定 power-z 色标；0–10 s 帧通过静态旧Fig3C（现A场部分）数值一致性检查。只作代表病例动态可视化，不承担独立 cohort 或机制结论 |
| `interictal_event_phenotypes` | FigS1 | `SUPPLEMENT` | `results/paper-ready-figure/supp_fig1_interictal_event_phenotypes/figures/` | legacy 人工标注事件验证集；作者已锁定为正式补图入口 |
| `soz_auc` | FigS2 | `SUPPLEMENT` | `results/paper-ready-figure/supp_fig2_soz_auc/figures/` | raw vs synchronized SOZ AUC；作者已锁定为正式补图入口 |
| `k_scan_templates` | FigS3 | `SUPPLEMENT` | `results/paper-ready-figure/supp_fig3_k_scan_templates/figures/` | masked K=2–10 扫描与高阶示例；作者已锁定为正式补图入口 |
| `axis_geometry` | FigS4-A–D | `SUPPLEMENT` | `results/paper-ready-figure/supp_fig4_axis_geometry/figures/` | paired timing-versus-timing+space direction score / axis direction / held-out readback / clinical SOZ all-contact compactness；正式补图入口 |
| `early_seizure_phenotypes` | absorbed into Fig3-A/B | `OMITTED_FROM_SUPPLEMENT` | `results/paper-ready-figure/supp_fig5_early_seizure_phenotypes/figures/` | early-ictal spectral phenotype + gamma example 已进入主图 Fig3，不再作为独立补图；旧产物仅保留追溯 |
| `multiband_field_concordance` | FigS5 | `SUPPLEMENT` | `results/paper-ready-figure/supp_fig5_multiband_field_concordance/figures/` | 与 dense-grid method-sensitivity 输出同源；正式补图入口 |
| `data_driven_snn_nlc_pathway_confirmation` | FigS7-A–E | `SUPPLEMENT` | `results/paper-ready-figure/supp_fig7_nlc_pathway_confirmation/figures/` | A–D 为 1581--1592 共 12 张全新配对网络的冻结 local-connectivity ablation；E 为从主 Fig4 移入的 627-event masked-rank KMeans heatmap、逐触点 rank distribution 与共享色条。独立 panel 无角标，完整拼板带 A–E。定位仍是 development-only 的模型内部 pathway effect 与事件结构，不是患者因果连接机制或患者盲泛化 |
| `static_contact_topography` | FigS6 computational | `SUPPLEMENT` | `results/paper-ready-figure/archive/2026-08-13_non_main_figure_packages/fig6_static_contact_topography/figures/` | manuscript-facing RNN 计算材料之一，当前不与主图并列 |
| `ordered_history_architecture_audit` | FigS6 computational companion | `SUPPLEMENT` | `results/paper-ready-figure/archive/2026-08-13_non_main_figure_packages/fig6_ordered_history_architecture_audit/figures/` | bounded-history / architecture audit companion |

编号说明：empirical multiband figure 已前移为 FigS5；computational RNN account 保留 FigS6，原编号冲突已解除。

## 4. 不能由目录名直接升级的图

- `fig6_*` 的多数目录是模型迭代谱系，不等于当前主文 Figure 6。
  `fig6_structured_rank_rnn`、`fig6_persistent_path_mode_bounded_negative`、
  `fig6_symmetric_axis_propagation_state_v2_2` 等只能按各自 bounded / historical 合同引用。
- `fig_m3a_*`、`fig_m4_dynamic_qi`、`fig_stage4_*` 是机制筛选或 visual diagnostic。
  即使目录位于 `paper-ready-figure/`，也不能自动升级成 paper-facing panel。
- `fig3a_raw_spectral_context/`、`fig3b_interictal_ictal_shared_field/`、
  `fig3-sup-tspectral-field-concordance/`、`fig3_peri_onset_field_similarity/` 和
  `fig3f_ab_dominance_cohort/` 现在都是 Fig3 canonical panel 的 producer/source 包；论文组装只消费
  `fig3/figures/`。
- `fig3_ictal_field_concordance_grid_rebuild/` 与
  `fig3_ictal_field_concordance_grid_method_sensitivity/` 是计算与审计包；正式图应引用登记表中的
  panel 资产，而不是把整个 staging 目录叫作“Figure 3”。
- `fig4_subject_snn_e1146/`、`fig_subject_snn_epilepsiae_1146*/` 和其余
  `fig4_data_driven_core_field_*` 是 Figure 4 的 source/迭代包；当前论文组装使用 `fig4/current_version.json` 指向的三行 A–I 版。旧A–G图包已移至 `archive/2026-09-29_pre_template_recovery_fig4/fig4/` 保留作历史素材。
  历史 A–G 曾按作者 2026-09-03 顺序重排并有完整 producer；B 预留为空，A、C–F 来自
  `frozen_substrate_confirmation`，G 来自 34 人 cohort formal result。2026-08-19 的误改 A–G 版本已可恢复地归档到
  `results/paper-ready-figure/archive/2026-08-19_rejected_a_g_fig4/fig4/`；改动前 A–H 快照仍保存在
  `results/paper-ready-figure/archive/2026-08-19_pre_a_g_fig4/fig4/`。本次合并 A/B 前的完整 A–H 素材包保存在
  `results/paper-ready-figure/archive/2026-09-03_pre_combined_fig4_ab/fig4/`；改成本次 A–E 顺序前的 A–G 快照在
  `results/paper-ready-figure/archive/2026-09-03_pre_fig4_a_e_reorder/fig4/`；预留 B 之前的 A–F 快照位于
  `results/paper-ready-figure/archive/2026-09-03_pre_fig4_reserved_panel_b/fig4/`。pathway confirmation 与 KMeans panel 已整理为
  `data_driven_snn_nlc_pathway_confirmation (FigS7-A–E)`；其冻结上游结果仍保留在 Topic 4
  scientific root 中作为 provenance。`data_driven_snn_dual_mode_validation` 仍是历史诊断，不能替换当前主图。

## 5. 2026-08-09 第一批归档

以下顶层入口已移动到
`results/paper-ready-figure/archive/2026-08-09_stale_aliases/`，未删除：

| 原目录 | 原因 | 当前替代 |
|---|---|---|
| `fig2b_interictal_ictal_shared_field/` | 旧 Fig2-B 试制，语义已迁移并重新锁定 | `ictal_field_scaffold (Fig3-C)` |
| `fig3_raw_spectral_context/` | 未登记的过渡输出，含多个 seizure 版本 | `ictal_field_scaffold (Fig3-A/B)` |
| `fig3_sup2_raw_spectral_context/` | 与正式 raw spectral context 存在逐字节重复的历史别名 | `ictal_field_scaffold (Fig3-A/B)` |
| `fig3_sup_peri_onset_field_similarity/` | 旧单病例 supplement 路径 | `ictal_field_scaffold (Fig3-E)` |
| `fig3_sup_preictal_field_similarity/` | 旧 preictal-only 路径，已被统一窗口合同替代 | `ictal_field_scaffold (Fig3-E)` |
| `fig3f_shared_ab_dominance/` | 旧 shared-only 1/3 分母图，不是当前 17 人 Fig3-F 包 | `ictal_field_scaffold (Fig3-F)` |

另有三套无任何仓库引用的 E1146 2026-07-06 参数重跑输出移至
`results/paper-ready-figure/archive/2026-08-09_unused_subject_snn_reruns/`。有 sensitivity 合同或
archive 证据引用的 E1146 变体仍保留在顶层，未按文件名相似性一并移动。

归档目录只保留 provenance。任何 producer 若重新生成这些旧顶层路径，应视为路径回归错误，而不是恢复
canonical 输出。

## 6. 2026-08-09 Figure 1/2 panel 合同归档

- Figure 1 的旧 `panela*`、SOZ ROC 误占 `panelb*`、C/E 联合文件以及 `d1/d2` 命名，已整体移到
  `archive/2026-08-09_fig1_pre_panel_contract/`；B 的原始 HFO 素材移到
  `archive/2026-08-09_fig1_source_material/`。Fig1-A 是手绘，不在代码输出中保留。
- Figure 2 的 projection、direction、event-envelope、template-field 与 E+F 联合行旧目录，已移到
  `archive/2026-08-09_fig2_pre_panel_contract/`。当前唯一 panel 入口为 `fig2/figures/fig2-panel{a..f}`；
  Fig2-A 为 2×2：作者提供的 Y9 植入概览、E10 subject-specific T1 局部 cutaway、
  真实局部电极几何和含 6 mm Gaussian support 范围的二维投影；四格等宽等高、标题居中且不画流程箭头。Y9 overview
  内的红/蓝方向 glyph 不延伸到 E10 三块；Y9 与 E10 不得写成同一病例连续 zoom。
- 独立 panel 文件内部不画左上角字母；每张主图另输出 `fig1-complete-layout` / `fig2-complete-layout`，
  panel 字母只出现在完整排版版。
- 统一复现入口：`scripts/paper_figures/build_main_figures_1_2.py`。Fig2-C v12 为 4 个严格等间距
  frame（`0/+16/+32/+48 ms`），要求最终二维场使用的全部参与触点质心和 top-3 热点每一步均相反移动；静态中间场逐帧按 participant top-3 mean 显示相对包络并固定使用 `PowerNorm(gamma=0.5)`，GIF 则冻结 per-event complete-window q99，默认从已验收矢量 PDF
  重新导出 600 dpi PNG；只有显式 `--recompute-fig2c` 才重新读取原始 EEG。
- Fig2-C GIF 已锁定为 `interictal_single_event_propagation_video (Supplementary Video 1)`；正式入口
  直接位于 paper-ready 根目录，`fig2/figures/fig2-panelc.gif` 只作同字节 source sidecar。

## 7. 2026-08-13 非主图顶层归档

原顶层 60 个目录曾完整移动到
`results/paper-ready-figure/archive/2026-08-13_non_main_figure_packages/`。2026-08-19 根据作者纠正，已将
正式 FigS1–FigS7 七个 `supp_fig*` 包已恢复为顶层 canonical supplementary 入口。2026-09-02 新增明确的
`fig5/` 最新候选入口，并把其余 producer staging、source、comparison 与 visual-QA 包移入日期归档。

Figure 3/4 builder 已同步读取归档 source，因此归档不影响当前主图重建。后续 producer 若再次向顶层生成
`fig3_*`、`fig4_*`、`fig6_*` 或 `supp_fig*`，应视为输出路径回归；正式锁定新主图时统一建立
`fig5/` 或 `fig6/`，而不是恢复旧候选目录。

## 8. 2026-08-31 all-event Timing+Space 主图更新

Fig2-B/E/F 与 Fig3-C/D/E/F 已写入各自 canonical `figures/` 目录，并在同一状态下重建两张 complete layout。更新前的 Fig2/Fig3 完整图包保存在 `results/paper-ready-figure/archive/2026-08-31_pre_all_event_timing_plus_space/`；归档包含逐 panel、完整拼版、README、metadata、registry 与随图 null 资产，可整体恢复。
