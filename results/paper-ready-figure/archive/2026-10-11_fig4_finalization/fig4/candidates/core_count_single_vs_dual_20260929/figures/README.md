### e1146_single_vs_dual_loss.png
E1146 单 core 与双 core 的最新训练 loss 曲线，另有同源 PDF/SVG。浅线及小点为四次随机优化各自的已完成前缀；粗线与阴影为两组全部重复共同完成的 epoch 上的中位数及四分位范围，使用共同原始 loss 尺度。
**关注点**：横轴放大至当前已评估范围，完整计划为每次96个epoch；目前属于初始采样，不代表优化完成或噪声复测结论。

### representative_single_vs_dual_loss.png
E635、E958、Y9（赵晨曦）的单核／双核训练结果，另有同源 PDF/SVG。每组两次随机优化，符号和短竖线在只有一个epoch时仍显示实际值、中位数及四分位范围，不连接或外推不存在的趋势。
**关注点**：各患者使用自身冻结loss尺度，不能直接将患者间纵轴数值当作病情或模型能力排序；当前首个配置尚不支持优化效果判断。

### single_vs_dual_loss_components.png
四位患者的rank分布、order/timing和participation三个分量，另有同源 PDF/SVG。每个epoch的三个分量来自同一个总loss最佳候选，复用原运行报告的选择算法。
**关注点**：误差分量不是分别选择最低值，某一项可随总loss下降而上升；本候选待作者目视检查。
