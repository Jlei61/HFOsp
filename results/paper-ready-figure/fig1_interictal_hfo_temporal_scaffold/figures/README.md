### fig1-panela1.png

严格复用 legacy 人工标注的 178 段 HFO，展示黑色叠加波形、黄色均值及 raw/normalized 平均谱。三行 x 轴均铺满完整 0–0.6 s，首末频谱 cell 仅延展绘图边界、不修改谱值。

**关注点**：标题应为红色 `HFO n = 178`，两张谱在 x 轴左右均不应出现白带。

### fig1-panela2.png

展示 Yuquan Y3 的三个真实群体 HFO 事件及 normalized spectrogram。A1/A2 的谱量统一为 Gaussian-smoothed magnitude；A2 保留原 50 ms Hamming 窗以维持群体事件的时间分辨率，红点取主峰 ≥70% 连通增强区的同图加权质心。

**关注点**：每个红点应落在对应通道的高频能量增强团内，左右外边界无白带，只有事件之间保留白色分隔线。

### fig1-panelb1.png

Yuquan refined-HFO count 对 clinical SOZ 的 subject-level ROC 汇总，现场重算 20 例。灰线为单被试，蓝线和阴影为 cohort mean 与 SEM。

**关注点**：本 panel 是 clinical anchor，不等于传播被限制在 clinical SOZ 内。

### fig1-panelb2.png

Epilepsiae refined-HFO count 对 clinical SOZ 的 subject-level ROC 汇总，纳入具备可用 SOZ 标签的 15 例。紫线和阴影为 cohort mean 与 SEM。

**关注点**：核对 n=15；无临床 SOZ 标签的病例不应被强行纳入。

### fig1-panelc.png

同一文件上下组合同一位 Epilepsiae E7 的 c1/c2。c1 展示 masked 时间顺序热图、原始 overlapping rank ridgeline 与 day/night strip；c2 将全量 6,556 个有效事件按 KMeans k=2 的 TA/TB 标签重排，并展示 mean-rank 轮廓。

**关注点**：上下两排来自同一患者，TA/TB 两个 n 之和必须等于 c1 的全量 n=6,556；TA/TB 之间使用白底灰色斜线断带并截断 x 轴线；右下 mean-rank 质心 marker 缩小。

### fig1-paneld1.png

患者内 masked shared-participant MI data vs permutation null；严格复用原 cohort producer 的 violin + box/IQR + whiskers + subject points，并恢复 data-vs-null 显著性括号。phantom ranks 已排除。

**关注点**：producer 对 40 个输入执行 `legacy_mi.masked=true` 硬检查；y 轴从 0 开始，括号显示 cohort-level data > null 检验。

### fig1-paneld2.png

Overall 与 within-template MI 配对散点，量化分模板后的 matching uplift。底层数值仍来自 masked `overall_tau` / `within_cluster_tau_mean` rank-concordance fields，但图面统一使用 MI 简写。画布只显示 median ΔMI，cohort 计数留给 caption/正文。

**关注点**：两轴从 0 开始；对角线下方恢复灰区；右上角图例解释蓝色 Yuquan、棕色 Epilepsiae；统计文字移入无数据的右下灰区。
