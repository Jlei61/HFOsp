# Figure 2 独立 panel 输出

本目录保存 Figure 2A–F 的无字母逐 panel PNG/PDF，以及带 A–F 字母的 `fig2-complete-layout` 完整拼版。独立 panel PNG 均以 600 dpi 输出。

### fig2-panela.png / .pdf

紧凑的 2×2 展示作者提供的 Y9 植入概览、E10 subject-specific skull-stripped T1 局部三平面 cutaway、局部电极到冻结平面的三维投影，以及带 6 mm Gaussian 显示核范围的二维触点覆盖。E10 三块均标明 `ICL` / `SCL` 电极杆。四格不写叙述性标题，视图身份由 figure legend 说明；不画流程箭头或 legend。右下 viewport 为单元格的 84%，只收紧 panel 间距和外部留白，左下仍保留完整投影平面边界。

**关注点**：Y9 与 E10 是不同 representative subjects，2×2 顺序不是同一病例连续 zoom；Y9 overview 内已有的红/蓝方向 glyph 不延伸到 E10 三块。Gaussian 层只说明 support-limited display coverage，不是组织活动测量或 analysis scoring kernel；E10 历史 warp 类型仍不可重建。

### fig2-panelb.png / .pdf

左侧以 E10/E14 的同一组 fold-0 留出事件方向，比较仅由事件内时序得到的传播模板轴（虚线）与加入真实三维事件方向后得到的模板轴（实线）；红/蓝表示传播模板 A/B，而不是无含义的 mode 编号。右侧显示 26 名可评估患者的绝对留出方向得分、患者内配对变化和记录块内方向置换零模型。底部同一行叠加蓝色时序模型、橙色时序+三维方向模型的患者 bootstrap cohort-median 分布及灰色方向置换 cohort-median null。底部分布区的长横括号表示加入三维方向的模型相对零模型的检验，短横括号表示两个模型的患者内配对检验。配对小提琴分布进入 Supplementary Fig. 4B。

主图文字相对原输出统一放大 1.28 倍，并给底部分布标签增加局部垂直间距；panel 尺寸与完整拼板位置不变。

**关注点**：该 panel 支持真实三维电极信息提高患者内跨记录块的方向一致性；不是未见患者预测，也不证明连续组织轨迹、传播速度或机制因果。

### fig2-panelc.png / .pdf

E10 的 TA/TB 单事件 readout 与 4 个严格等间距的 participant-only HFO envelope 场。两行标题统一为 `TA samples` / `TB samples`。当前 canonical v13 使用固定 gamma=0.5、色盲友好的 soft teal-to-navy 包络场。静态时刻由 all-participant full-field selector 在 2 ms 网格上选择，要求每一步的全参与触点质心和 top-3 热点均相反移动。每幅静态场再按本帧最强三个参与触点的均值显示相对包络，避免完整窗尺度把有效后帧压成近白色；因此静态帧只读空间位置，不读帧间绝对幅度。

**关注点**：这是 raw-EEG-derived timing 在既有冻结轴上的 representative cross-check，不是独立验证。

### fig2-paneld.png / .pdf

E10 的冻结 TA/TB shared-plane rank fields，直接取自原 Fig. 2C 最右侧竖排内容，作为静态模板对照。两幅场使用同一物理平面和统一 6 mm display kernel；左侧行标题统一为 `TA field` / `TB field`，不再保留上方 template 标题，也不再调用旧的独立 D producer。

**关注点**：模板场与 panel C 的单事件 envelope 场含义不同，不得把插值表面写成真实组织传播轨迹。

### fig2-panele.png / .pdf

从最新 all-event、时序+真实三维方向重聚类后的 18 人 shared-axis 队列中显示四例：E1、E12、E5、Y9，分别覆盖 2、2、4、5 根电极杆，并能完整放入统一 70×70 mm 显示窗。除总触点数和电极杆数外，显示例还要求至少一根杆上有不少于 5 个连续触点；E12 的 HR3–HR10 提供 8 个连续触点，替代触点编号稀疏的 E20。四例均为 TA–TB 负相关且 channel permutation 单侧 p<0.05；E1、Y9 通过 BH-FDR，E12、E5 为名义显著的连续多触点形态案例。案例只负责形态传达，队列推断仍由 Fig. 2F 的完整 18 人承担。

**关注点**：逐列比较同一患者 TA 与 TB 的早晚场是否翻转；重点检查连续多触点上的梯度，而不是由单个离群 contact 决定视觉印象。不能把显示例当成独立抽样验证。

### fig2-panelf.png / .pdf

重聚类后 28 人可拟合 TA/TB 轴，其中 20 人满足 `|cos(TA,TB)|≥0.5` 的共线标准并可构造 shared axis；18 人同时具有受支持的二维几何，构成本 panel 的正式分母。上半部分展示逐患者 signed field correlation、observed IQR 横条和 raw cohort median 菱形；15/18 为负，负相关比例相对 50% 的一侧 exact sign test 为 `P=0.00377`（图内仅标 `**`）。下半部分使用同一 raw-r 坐标显示 full-contact shuffle 的 cohort-median null；下方 observed 菱形与上方 median 菱形严格同值、同 x 坐标，lower-tail permutation 为 `P=1.0×10⁻⁵`（图内仅标 `***`）。

**关注点**：上下两个菱形都表示 observed raw cohort median `r`，不能再将下方菱形解释成 `Δmedian`。`**` 检验负相关患者比例，`***` 检验 raw cohort median 相对 full-contact spatial null；subject-centered primary 统计仍保存在 metadata，且当前经验 P 与 raw-r display null 一致。

### fig2-complete-layout.png / .pdf

将 A–F 六个独立 panel 排为完整 Figure 2，并只在完整画布上添加 A–F 字母。

**关注点**：独立 panel 内不应重复出现字母；完整排版应保留各 panel 的相对信息层级和可读字号。
