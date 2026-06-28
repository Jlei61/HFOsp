# 发作内 field 动力学 — narrow 平行批（扩队列调查）

这是 broad 主目录（`../../field_dynamics/figures/`）的 **narrow substrate 平行批**，用来回答"非 swap 被试能不能也用
间期模板**端点**（每模板最早 2-3 个触点的 compact core）构轴 + 做统计 + 画 GIF"。**答案：能**（轴/走廊数学不必
swap）。队列 7 个：`epilepsiae_{1096,1125,1146,253,384,442,958}`（含非 swap 的 442、长发作的 1146 23sz）。

图的类型、口径、渲染形式、z-ER caveat 全部与 broad 主目录一致 —— **逐图说明见 `../../field_dynamics/figures/README.md`**。
本目录同样有：`<ds>_interictal_AB.png` / `_mean_ictal.png` / `_progress.png` / `_offset.png`、`<ds>_field_evolution.gif`、
`per_seizure/<ds>/*`。非 swap 被试的红/蓝圈 = 每模板**最早 top-K 端点集合**（template-earliest，替代 swap 集合）。

## 关键结果（扩队列把 broad 的暗示证否了）

broad 队列里"轴向走廊占比随发作下降、非轴向上升"有方向暗示（轴向 median ρ<0 在 5/8、非轴向 ρ>0 在 8/8）。
**narrow 平行批不复现、甚至多数反向**（`../trend_stats.json`）：

- 轴向 median ρ<0 仅 **3/7**（1125/384/958）；其余为正（1146 **+0.52**、442 **+0.37**、253 +0.40、1096 +0.19）。
- 非轴向 median ρ>0 仅 **2/7**（1146/442）；多数为负。
- 每被试 Wilcoxon 显著仅 958（轴向 p=0.008）。

→ **"轴向走廊随发作减弱"不是稳健现象，依队列/substrate/走廊厚度而变**（narrow 走廊多 n_axial_mid=2，薄、噪）。
**结论 = exploratory，方向假设阴性偏向；图/GIF 作 supplementary 可视化（"场会变"为真，"方向减弱"不稳健）。**
z-ER 中后期偏示意，场图/GIF 只作相对空间形状看。
