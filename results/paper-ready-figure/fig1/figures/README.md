# Figure 1 panel 与完整排版输出

Figure 1A 由主 builder 从旧 Supplementary Figure S6 TIFF 固定裁剪一张 representative SEEG 植入脑图；不绑定患者身份，也不重画科学元素。独立 panel 文件不写左上角 panel 字母；字母只出现在 `fig1-complete-layout` 完整排版中。

### fig1-panela.png / .pdf

复用 `ReplayIED/tiffs/fig_s6_画板 1.tif` 上排的脑表面与 SEEG 电极渲染，只做固定裁剪和分辨率转换。

**关注点**：A 只建立代表性的植入空间背景；图中不报告患者编号，彩色触点也不作为 Figure 1 的独立统计结论。

### fig1-panelb1.png / .pdf

严格复用 legacy 人工标注的 178 段 HFO，展示黑色叠加波形、黄色均值及 raw/normalized 平均谱。三行 x 轴均铺满完整 0–0.6 s，首末频谱 cell 仅延展绘图边界、不修改谱值。

**关注点**：红色计数标签应为 `HFO n = 178`，两张谱在 x 轴左右均不应出现白带。

### fig1-panelb2.png / .pdf

展示 Yuquan Y3 的三个真实群体 HFO 事件及 normalized spectrogram。B1/B2 的谱量统一为 Gaussian-smoothed magnitude；B2 保留原 50 ms Hamming 窗以维持群体事件的时间分辨率，红点取主峰 ≥70% 连通增强区的同图加权质心。左侧波形与右侧 spectrogram 使用相同时间范围、相同刻度和相同数据轴宽度；色条占独立窄列。

**关注点**：两块数据轴的时间尺度在物理宽度上必须一致；每个红点应落在对应通道的高频能量增强团内，左右外边界无白带，只有事件之间保留白色分隔线。

### fig1-panelc.png / .pdf

展示 Epilepsiae E7 的 masked 时间顺序热图、原始 overlapping rank ridgeline 与 day/night strip。

**关注点**：非参与触点必须保持空白；day/night strip 与事件时间顺序严格对齐；Day/Night 使用黑白方块在患者标题同一行单独画 legend，xlabel 只保留 `Population events (time-ordered)`；colorbar 使用顶部水平标题 `Heatmap rank / First → Last`。

### fig1-paneld.png / .pdf

患者内 masked shared-participant MI data vs permutation null；严格复用原 cohort producer 的 violin + box/IQR + whiskers + subject points，并恢复 data-vs-null 显著性括号。phantom ranks 已排除。

**关注点**：producer 对 40 个输入执行 `legacy_mi.masked=true` 硬检查；y 轴从 0 开始，括号显示 cohort-level data > null 检验。

### fig1-panele.png / .pdf

将同一位 Epilepsiae E7 的全量 6,556 个有效事件按 masked KMeans k=2 的 TA/TB 标签重排，并展示两类 mean-rank 轮廓。

**关注点**：TA/TB 两个 n 之和必须等于 6,556；TA/TB 顶部标签必须粗体显示，TA 固定为红色 `#B2182B`，TB 固定为蓝色 `#2166AC`，并与右侧 mean-rank 曲线一致；colorbar 标题固定放在色条上方。

### fig1-panelf.png / .pdf

Overall 与 within-template MI 配对散点，量化分模板后的 matching uplift。底层数值仍来自 masked `overall_tau` / `within_cluster_tau_mean` rank-concordance fields，但图面统一使用 MI 简写。右下小 panel 复用补充图 HFO AUC 的配对语法，以患者连线、均值柱和配对 Wilcoxon 括号直接比较 single-template 与 multi-cluster MI。

**关注点**：两轴从 0 开始；对角线下方保留灰区；右上角 Yuquan/Epilepsiae 图例必须使用较小字号并带白底细边框；右下不再放灰色摘要字，而应以窄竖向、非方形布局显示 40 名患者的配对 MI 分布和显著性括号。x 轴用居中的单行短标签 `Single` / `Multi`，分别表示 single-template MI / multi-cluster MI。

### fig1-complete-layout.png / .pdf

将 TIFF 提取的 A 与代码生成的 B–F panel 排为完整 Figure 1，并在完整画布上统一添加 A–F 字母。

**关注点**：独立 panel 内不应重复出现字母；完整排版中的字母位置和字号应统一。
