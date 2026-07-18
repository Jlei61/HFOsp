# MZ early-field bridge — 图说明

这两张图回答同一个问题：同一块 E1146 模型底物，安静自发（slow-off）时间期样事件里 15 个虚拟电极点亮的**先后顺序**，能不能预测只开 `z`（抑制效力消耗）后底物滑向"操作性失控放电"时、刚点火 0–50 ms 的**能量分布**。失控 = operational runaway，是模型代理，**不是临床发作**；能量是 virtual-LFP 30–80 Hz 包络能量代理，不是临床宽带功率。

### mz_early_field_bridge_seed1.png
单 seed 诊断图（复用 Topic 4 Figure-5 语法但为诊断变体）。上两条是**分别标注**的两段轨迹：slow-off（间期模板来源）与 z-only 操作性失控（红虚线 = t120 起始，粉带 = 0–50 ms 能量窗），绝不把 slow-off 事件阴影画到失控轨迹上。下两格是同一虚拟 SEEG 平面上的对照场：左 = maxAB 取胜方向（此 seed 为 B_to_A）的间期事件顺序（viridis，紫=早），右 = 早期失控能量（Blues，深=高）；灰箭头标 source→sink 轴。

**关注点**：左右两格是否在同一端"早=热"地对上（高轴/sink 端一端最早被点亮、也最热），即间期指纹是否预测了失控早期能量；标题里的 rho_maxAB 是这个对齐的相关系数。

### mz_early_field_bridge_multiseed.png
三 seed（1/3/4）诊断网格，五个互不重复的面板各答一问：Q1 模板在**留出事件**上是否可复现（held-out Spearman；圆=A→B、三角=B→A）；Q2 早期关联 rho_maxAB 随时间窗如何变、是否高过同杆内置换 null 的 p95（灰条）；Q3 电极读出与源空间读出是否同号（同象限=一致）；Q5 能量场支撑数与动态范围（无柱=退化场）；Q6 逐 seed 的 onset/模板/maxAB 资格与 within-shaft p（未跑/不合格显式标 MISSING）。标题只报三 seed 中位数/范围/同号计数，**不做 n=3 的 cohort p 值**。

**关注点**：Q1 里 B→A 三角是否贴近 1.0（模板可复现）、Q2 点是否稳定高过灰条（关联显著且跨窗稳健）、Q3 两 seed 是否都落右上（电极×源一致）、Q6 表里 maxAB 资格与 within-p 是否逐 seed 成立；任何反号、退化、MISSING 都据实显示。
