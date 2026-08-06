### mz_spatial_autonomous_latch.png

这张图检验 frozen regional entry/exit geometry 能否由固定背景事件驱动的 autonomous Z–p–M 外慢环真正穿越。A–D 依次展示普通事件返回、Z-only 跨 fold 后继续加速、fast-M 提前退出与 four-return arm 的 support escape，以及对应 Z/A 慢轨迹；E 对照 base/half-dt outcome；F 给出机制判定。

红色 support escape 不是终止；本 producer 即使观察到至少 4 个无外驱 returns 后回到低尾，也只能标为 pending recovery/retrigger，不能直接叫完整 lifecycle。当前没有这类 finite-low-tail candidate，因此本图是 registered-margin clean no-go diagnostic，不是 paper-level seizure figure。

**关注点**：共享 regional Z 确实能产生区域性 entry；bath 资源由预注册 mask 固定，故 bath 未招募不能单独当 emergent containment。additive M 的有限退出发生得过早；一旦保留第 4 个 return，Z 已离开验证过的 bounded-CCO 窗口。
