# Figure 4 panel 与完整排版

本目录是 Figure 4 的 canonical paper-facing 入口。独立 panel 不带左上角字母；只有 `fig4-complete-layout` 带 A–G。Panel B 按作者要求留空，等待后续补入 data-driven 参数对患者间期事件复现的影响，因此当前没有 `fig4-panelb` 独立文件。其余 PNG 均为 600 dpi，并提供同画面的 PDF。

### fig4-panela.png / .pdf

作者确认的组合 panel。左侧为 local E/I circuit 与慢变量示意，右侧为同一冻结 figdata 的 patient-specific E/I substrate 和患者触点几何；右图删除 `anisotropic E→E` corridor 及 `possible data driven core` 覆盖层。左框与右侧 SCL9 局部框使用相同的约 `1.68 × 1.31 mm` 视野；左图 scale bar 为 `0.5 mm`，右图显示 `−10–10 mm` 仿真坐标。每个触点周围的低透明度深绿色 halo 表示 Fig4F firing-density readout 的 Gaussian sampling footprint（`σ=0.25 mm`；理论 95% 权重半径 `0.61 mm`）。

**关注点**：两个框尺度匹配，但左侧仍是机制示意而非从右图裁出的组织图；绿色 halo 不是解剖边界。

### Panel B（预留，无独立文件）

完整拼板右上区域仅保留 B 角标和空白画布，后续用于展示 data-driven 模型中不同参数对患者间期事件复现的影响。当前版本不填入占位文字、示意数据或临时结果。

**关注点**：B 的空白是明确的作者布局合同，不代表缺失文件或构建失败。

### fig4-panelc.png / .pdf

冻结 data-driven Node field，以及 12 张网络 pooled clean events 的 Model TA 与 Model TB 空间模式。Node-field 色条使用独立列，刻度在左、`h` 在顶部，不与场图或 `y (mm)` 重叠；模型图坐标为 `x (mm)` / `y (mm)`，两个模型方图缩至各自网格单元的 74%。

**关注点**：MTA/MTB 是 development-level 模型模式，不是解剖 core。

### fig4-paneld.png / .pdf

模型 MTA/MTB 与患者 TA/TB 的 mean-rank profile。删除拥挤的不确定性带；legend 放到 panel 底部，不再从上方压低主体。

**关注点**：这是 E10 development 对照，不是 cohort 级模板恢复。患者 TA/TB 使用 all-event Timing+Space 标签；模型 MTA/MTB 与 SNN KMeans 保持冻结。

### fig4-panele.png / .pdf

12 张网络等权的 contact-split cross-fit Spearman 矩阵；患者 TA/TB 使用 all-event Timing+Space 标签，模型端保持冻结。两个 `***` 分别检验 MTA–TA 与 MTB–TB 是否高于各自的 within-shaft contact-permutation null。

**关注点**：这是 development-case 的 post-hoc null calibration，不支持 patient-blind 或真实几何泛化。

### fig4-panelf.png / .pdf

同一网络窗口的 30–80 Hz virtual-contact firing-density readout。MTA/MTB 阴影按参与触点的 recruitment-onset span（两侧各加 12 ms）绘制，不再用过长的 detector-event duration；比例文字和逐通道黑色 onset 点均已删除。

**关注点**：这不是 current、LFP 或临床 SEEG；MTB 阴影只标招募时间范围。

### fig4-panelg.png / .pdf

34 人 held-out cohort。左侧完整列出 eligible、loss 胜过 matched null、same-network two-mode pass 及两项同时满足的人数和比例；右侧为被试内 paired held-out loss。

**关注点**：`P=0.043` 只对应连续 loss 的配对检验；11/34 是描述性交集。

### fig4-complete-layout.png / .pdf

三行 A–G 完整排版：A 为组合机制/底物；B 使用既有右上留白并明确预留；C–E 为模型空间模式与患者模板 rank/cross-fit；F–G 为同网络 readout 与 cohort。panel 字母只在此文件中出现。原 KMeans heatmap/rank-distribution panel 保留在 Supplementary Fig. 7E。

**关注点**：当前安全口径仍是 development-level partial interictal substrate support；不能升级为临床因果机制、解剖 core、patient-blind real-geometry generalization 或 ictal lifecycle 证明。
