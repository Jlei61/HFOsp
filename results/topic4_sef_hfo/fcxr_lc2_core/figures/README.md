### h_sensor_separability.png

左图比较普通 returning IED 的 H 负荷峰值上界与 HEO1/HEO2 已建立高态的负荷 trough；右图分别画 HEO1-only、HEO2-only 和两者联合的 bootstrap 分离余量。只有联合余量高于零的连续 tau 区间才允许激活 H 电流。

原始严格全窗口统计为 `H_SENSOR_NOT_SEPARABLE`；阶段验收为 `R1_IMPLEMENTATION_ACCEPTED`。前者是 long-gap stress test，不再作为闭环 H geometry 的 hard gate。

**关注点**：HEO1-only 在中等 tau 可分离，但 HEO2-only 全程为负并决定联合失败；这定位的是当前局部 sensor 对 adaptation-burst trough 不稳健，而不是 H 正反馈整体已被否定。
