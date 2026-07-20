### mz_inhibitory_reserve_periodic_oracle.png

这张图用 mapping 阶段 hash-locked 的 frozen CCO `U(t)`，在 CSV 的 exact 8-return 起止窗口内做分段常值解析 q 更新。A–C 检查 q 极值、周期均值与每-return contraction；D–E 展示代表性周期轨迹和 stroboscopic 收敛；F 锁定即使 hold gate 全通过，已有 entry-ordering no-go 仍然有效。

这里的 sensor 不随 q 反馈，bath mask 与 E→E 均未改变，因此不能把 periodic hold 写成 autonomous seizure lifecycle 或空间 containment。

**关注点**：先看 24 个 q×phase×dt 组合是否全部满足 `.8325<=q<=.850`、均值误差与 `rho<.9`；无论结果如何，都不能据此解锁 autonomous run。
