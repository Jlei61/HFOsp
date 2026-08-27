# Figure 2 独立 panel 输出

本目录保存 Figure 2A–F 的无字母逐 panel PNG/PDF，以及带 A–F 字母的 `fig2-complete-layout` 完整拼版。独立 panel PNG 均以 600 dpi 输出。

### fig2-panela.png / .pdf

近方形 2×2 展示作者提供的 Y9 植入概览、E1146 subject-specific skull-stripped T1 局部三平面 cutaway、局部电极到冻结平面的三维投影，以及带 6 mm Gaussian 显示核范围的二维触点覆盖。E1146 三块均标明 `ICL` / `SCL` 电极杆。上排不写标题；下排只保留居中的 `Electrodes projection` 与 `2D local field`。不画流程箭头或 legend；右下 viewport 缩为单元格的 72%，左下保留完整投影平面边界。

**关注点**：Y9 与 E1146 是不同 representative subjects，2×2 顺序不是同一病例连续 zoom；Y9 overview 内已有的红/蓝方向 glyph 不延伸到 E1146 三块。Gaussian 层只说明 support-limited display coverage，不是组织活动测量或 analysis scoring kernel；E1146 历史 warp 类型仍不可重建。

### fig2-panelb.png / .pdf

左侧以 E1146/E548 的同一组 fold-0 留出事件方向对比仅时序模板轴（同色虚线）和时序--空间模板轴（同色实线），右侧显示 25 名可评估患者的绝对留出方向得分、患者内配对变化和记录块内方向置换零模型。底部同一行叠加蓝色 Timing、橙色 +Space 的患者 bootstrap cohort-median 分布及灰色方向置换 cohort-median Null；底部分布区的长横括号表示 +Space 相对零模型的检验，短横括号表示 +Space 相对 Timing 的患者内配对检验。配对小提琴分布进入 Supplementary Fig. 4B。

**关注点**：该 panel 支持真实三维电极信息提高患者内跨记录块的方向一致性；不是未见患者预测，也不证明连续组织轨迹、传播速度或机制因果。

### fig2-panelc.png / .pdf

E1146 的 TA/TB 单事件 readout、4 个严格等间距的 participant-only HFO envelope 场及时不变的冻结 template-rank field。当前 canonical v11 使用固定 gamma=0.5、色盲友好的 soft teal-to-navy 包络场；静态时刻由 all-participant full-field selector 在 2 ms 网格上选择，要求每一步的全参与触点质心和 top-3 热点均相反移动。每幅静态场再按本帧最强三个参与触点的均值显示相对包络，避免完整窗尺度把有效后帧压成近白色；因此静态帧只读空间位置，不读帧间绝对幅度。

**关注点**：这是 raw-EEG-derived timing 在既有冻结轴上的 representative cross-check，不是独立验证。

### fig2-paneld.png / .pdf

E1146 的冻结 TA/TB shared-plane rank fields，作为静态模板对照。两幅场使用同一物理平面和统一 6 mm display kernel。

**关注点**：模板场与 panel C 的单事件 envelope 场含义不同，不得把插值表面写成真实组织传播轨迹。

### fig2-panele.png / .pdf

四个锁定案例的 TA/TB shared-axis rank-field 配对展示；案例只用于说明可读的反向场形态，队列推断不由这四例承担。

**关注点**：患者选择和完整 12 人 denominator 写在 `fig2_panel_ef_metadata.json`，不能把 4 个显示例当独立抽样验证。

### fig2-panelf.png / .pdf

完整 shared-axis、二维几何可评估队列的逐患者 signed field correlation，以及 full-contact shuffle 的 cohort-median-shift null。

**关注点**：安全口径是 cohort median 比全触点随机化更负；不能升级成所有患者或所有 null 均显著。

### fig2-complete-layout.png / .pdf

将 A–F 六个独立 panel 排为完整 Figure 2，并只在完整画布上添加 A–F 字母。

**关注点**：独立 panel 内不应重复出现字母；完整排版应保留各 panel 的相对信息层级和可读字号。
