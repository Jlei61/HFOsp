# 六个共同 seed 双核几何的通道残差定位

本次使用已经收齐八个训练网络的六个几何，固定患者数据和观测方法，仅定位现有失败的空间／顺序组成，不改变正在运行的 v3 搜索。

## 关键发现

SCL9 在六个模型中的参与条件平均 rank 都晚于患者。患者平均 rank 为 0.325，六个模型的范围为 0.551–0.681。ICL4 等部分较后 contact 的平均 rank 则偏早。

在 SCL9 与 ICL4 都参与的事件中，患者 SCL9 先于 ICL4 的概率为 0.769，六个模型为 0.368–0.481。患者的 ICL4 减 SCL9 质心时滞中位数为 18.55 ms；六个模型的对应偏差均为负，范围 -39.91–-23.43 ms。这一对在每个模型都有至少十个共同参与事件。

这些是六个已测几何的描述性范围，不是置信区间。几何、网络 seed 与 contact 之间并不独立，不能将它们当作多个患者推断总体显著性。平均 rank 只定位误差，完整事件分布仍是拟合和验收对象。

## 对下一步的影响

当前拟合误差不仅是无符号轴角度的差异，还包含具体 contact 谁先谁后的偏差。单独强化双核连线与 E 长轴平行不能保证纠正这一点，因而继续保留方向派生验证、完整分布拟合的合同。

这六个几何不代表全部四维 XY 空间，因此还不足以证明 VTH 双核模型无解。v3 已开始围绕共同 seed 结果选择局部中心并继续随机位置搜索；下一批应检验这些残差是否能够随位置移动缩小，同时检查参与、rank、时滞全部分布，不能只盯这一对 contact 优化。

## 可复现输出

生产脚本为 `scripts/audit_topic4_xy_replicated_residuals.py`，汇总 `results/topic4_sef_hfo/joint_rank_space_dual_core_search_v2/replicated_residual_audit/summary.json` 包含患者／模型的逐 contact 和逐 pair 统计、数据与代码 SHA256。逐个 worker 的元数据和轨迹哈希已验证，事件数与冻结补算结果一致。

图由 `scripts/paper_figures/plot_topic4_xy_replicated_residuals.py` 生成，同目录 `figures/` 包含 PNG/PDF/SVG 与中文 README。PNG 和 PDF 已目视检查，色标两端标记了饱和范围；尚未代替作者验收。

Goal 保持进行中，当前没有通过验收的双核基底。
