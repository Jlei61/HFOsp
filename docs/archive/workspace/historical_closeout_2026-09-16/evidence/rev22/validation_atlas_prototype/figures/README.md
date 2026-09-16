### validation_atlas_prototype.png

rev22 验证图谱第一层的版式原型，数据来自 rev20 的 232 条历史轨迹（每个候选按拓扑单元合并打分）。
列是 rev22 将继续研究的四个连接参数，行是六个选择盲验证端点中在 rev20 数据上"候选间差异
超过种子噪声"的那些（本次进入主图：heldout_D_support, heldout_D_order, heldout_D_time_ms, recall_fixed_budget, kmeans_balanced_alignment, ood_all_returned, heldout_composite）。空心圆是 4 网络 screen，实心菱形
是 12 网络确认，误差线是留一拓扑 jackknife 或单元均值的 90% 区间，灰带是参考点事件数下患者
自身对比的地板。固定预算 recall 使用 n_cov=22、r_cov=9.453（原型值，正式冻结在 Task 8）。

**关注点**：先看哪一列的曲线在多行同时离开灰带；单行改善、其他行变差的族属于 trade-off。

### validation_atlas_prototype_sidecar.png

未通过展示可辨识规则（候选间 q90−q10 除以噪声中位数 < 1）的端点行：无。
这些行完整保留，只是不进主图。

**关注点**：这些端点在 rev20 参数范围内分辨不开候选，不代表端点无意义。

### validation_family_matrix_prototype.png

第二层的版式原型：rev20 的确认候选相对参考点的配对差（按 12 个网络种子做配对 bootstrap，正值 = 更好），
每列一个端点，含次要的复合距离和事件产量列。绿色区间整体大于 0，红色整体小于 0，灰色跨 0。

**关注点**：一个候选要在多列同为绿色才算"有用"；只在一列绿、另一列红是 trade-off。

> **2026-09-03 更正（rev22 v5.1）**：图中「长轴方向」与「长短轴比」两列来自 rev20 的名义参考点 `(45°, 2)`，
> 该参考点与冻结图的真实核轴（配准患者轴 −22.8°）错位 68°，所以这两列的横轴标签不代表所写的几何变化
> （例如 `aspect=3.0` 实际把实现几何压圆并转轴）。两列仅作版式原型，不作为响应锚点；剂量两列不受影响。
