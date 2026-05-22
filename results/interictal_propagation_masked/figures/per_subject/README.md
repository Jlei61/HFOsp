### per-subject propagation figure
这组图采用 2x2 布局。左列宽，右列窄。
左上：原始 lagPatRank heatmap（时间顺序），底部附 Day/Night 条带（白=Day, 黑=Night）。左下：k_best 聚类后的 heatmap（按簇排序，红色虚线分隔，簇标注在顶部）。
右上：Per-channel rank distribution（原始通道顺序，stacked histogram）。右下：Cluster rank distributions（固定通道排序，两个簇用不同颜色叠加 rank 分布，直观对比不同传播模式的分布差异）。
**关注点**：先看聚类后 heatmap 的簇内颜色一致性和簇间差异，再看右下角分布中各簇的 rank 峰值位置差异（forward/reverse）。
