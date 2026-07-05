# 发作内 field 动力学 pilot — 图说明

**swap-positive broad 队列**（`epilepsiae_{139,253,1077,1096,1125,1150,620,635}`）逐次发作的 field
随发作进程（onset→offset）+ 终止前后的变化。**纯描述性 subject-level pilot，不下 cohort/机制结论。**
（原来的 narrow 6 个被试大多没有"走廊"，已降为对照，cache 留盘，不在此目录。）

每张 field 图都锚到该 subject **发作前**布局（与 `axis_alignment/.../field_vs_ictal_swap` 同一渲染：
viridis 场 + 触点按值着色 + **红圈=source in 模板A / 蓝圈=source in 模板B**（分布式 swap source 集合）+
SOZ）。走廊（轴向中段）= 两模板**最早 compact core**（top-2-3）连线段的中段、贴轴、非 source 的触点；
"走廊"的红蓝圈是显示用的整串 swap 集合，而轴/走廊的**数学**用紧凑 core（两者口径分开）。

> **重要 caveat（z-ER 中后期）**：所有 ictal 场是**示意性**的。baseline-robust-z（z-ER）是相对发作前
> 安静期归一的，越往发作中后期这个归一越不可靠。这里画的是**窗内 rank01**（相对空间形状，比绝对 z 稳），
> 但绝对幅度中后期不要当定量。轨迹曲线（占比/对齐/同步）同样在中后期偏示意。

## subject-level（每 subject 4 张 PNG + 1 个动画）

### \<ds\>_interictal_AB.png
间期传播场 **模板 A | 模板 B** 并排（发作前锚，大图，带 colorbar）。A、B 互为反向（swap）：A 的早端=红圈、
在 B 里变成晚端。
**关注点**：两模板是否清楚互为反向；红/蓝 source 落在轴两端；这是后面所有 ictal 场的参照布局。

### \<ds\>_mean_ictal.png
全部合格发作的**平均早期（0–10s）激活场**（同一布局）。
**关注点**：发作早期最亮的区相对间期 source（红/蓝圈）落在哪——靠近某一端还是别处。

### \<ds\>_progress.png
横轴 = 发作进程 0→100%（每发作压到同尺度）；灰细线=每次发作，黑粗线=分箱 median。四子图：场-轴对齐(maxAB)、
**轴向中段正质量占比**、**非轴向正质量占比**、同步。走廊为空的 subject（如 253 双侧植入）该子图标
`corridor n=0 (not measurable)`。
**关注点**：轴向中段占比是否随进程**下降**、非轴向是否**上升**、同步是否变化——以及灰线是否抱团（跨发作一致性）。

### \<ds\>_offset.png
按发作终止（offset=0，红虚线）对齐的窗 `[-60,-30,-10,0,+30]s`，散点=各发作各窗、黑线=median；已排除左缘
早于 onset 的短发作窗。
**关注点**：终止前 10–30s 是否有突变；offset 后（+30s）是否回落。

### \<ds\>_field_evolution.gif（动画，每 subject 一个）
该 subject 最长合格发作的 field 演化：[间期A | 间期B | 滑动的 ictal 场]，窗 10s、步 3s，从 onset 滑到
offset+30s。标题随帧显示时间/进程，并标注 POST-OFFSET。
**关注点**：直接看场怎么从间期形状演化、热区往哪走、终止前后变化（**记住中后期是示意**）。

## per-seizure（每条 dur≥40s 且非 parity_fail 的发作 3 张 PNG，在 `per_seizure/<ds>/`）

### \<ds\>_szN_evolution.png
单次发作的时刻演化行 [间期A | 间期B | ictal 0/25/50/75/100%]，同一布局 + source 圈。
**关注点**：单次发作内 field 是否保持同一轴还是方向漂移；热区的此消彼长。

### \<ds\>_szN_trajectory.png
单次发作各指标随进程（band=bb，仅 ictal_fraction≥0.5 的窗）。
**关注点**：本次发作走廊/非轴向/对齐/同步各自的走向。

### \<ds\>_szN_termination.png
单次发作终止对齐的场行 [间期A | 间期B | off -45/-20/-5/+15s]。
**关注点**：终止前后场的变化（不预设终止动力学）。

---
本目录 = **broad 队列 9**（8 swap + E916 非 swap；E916 证明走廊不必 swap，但发作太短中位 8s→无趋势贡献）。
narrow 平行批在 `../../field_dynamics_narrow/figures/`。

**初步观察（描述性，勿当结论）**：**趋势统计**（`../trend_stats.json`；每次发作 Spearman ρ(progress, 占比) →
每被试 Wilcoxon，发作=重复单位）：broad 轴向中段 `median ρ<0` 在 **5/8**、非轴向 `median ρ>0` 在 **8/8**
——**broad 有方向暗示**；但每被试 Wilcoxon 显著仅 **1/8**（轴向 1125 p=0.009、非轴向 1096 p=0.004），1150/620 反向。
**关键：narrow 平行批不复现甚至反向**（轴向 ρ<0 仅 3/7、非轴向 2/7；E1146 轴向 ρ=+0.52）→ **"轴向走廊变弱/非轴向
变强"不是稳健现象，依队列/substrate 而变**。`align_maxab` 多数中后期升高（轴共享在时间上保持，呼应 A 线）。
**pilot/exploratory，z-ER 中后期偏示意——不写成 cohort/机制规律；图/GIF 作 supplementary 可视化（"场会变"是真的，
"方向减弱"不稳健）。**
