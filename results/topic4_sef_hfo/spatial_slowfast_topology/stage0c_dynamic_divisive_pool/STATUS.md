# Stage 0C 动态除法池状态

- 结论：`INCONCLUSIVE_NO_CONFIRMED_FINITE_FAST_OBJECT`
- 是否找到有限快态对象：`False`
- 参数点 / screen forks / confirm forks：189 / 3564 / 0
- alpha_G=0 复刻 Stage0B：`True`
- 双向 root-set 一致：`True`
- wall / peak RSS：1784.95 s / 0.311 GiB
- 有限候选：23 条 non-exact fork 在 rate classifier 上先被判为 oscillatory candidate，但全部因逐步 LUT 越界拒绝；另有 314 条长瞬态、111 条 bounded-indeterminate。
- 最强未决点：`z=0.80, alpha_G=12`，12 个不同 non-exact histories 收敛到约 2.079 Hz 的重复活动（tail mean 约 9.10 Hz，逐步 tail peak 约 99.56 Hz），但 tail 约 9.1% Euler states 的 `mu_E/mu_I` 低于 LUT 下界 -40 mV。因此它不是 accepted orbit，也未进入 12 s confirm 或 ablation。
- coverage 边界：锁定 alpha 轴为 `[0,1,2,4,8,12,16,24,32]`，开放区间 `(0,1)` 未采样；本轮不自行补点。
- 解释：动态池在当前 LUT 实现中呈现一个值得精确 transfer 复核的疑似有限振荡带，但当前实现无法裁决；所以不是 clean no-go，更不能写成已确认 limit cycle。

本阶段冻结 z、关闭 local recovery r、噪声、空间耦合和 dynamic phi。因此结果只裁决 M4 两级动态除法池能否在均匀快系统中造出有限对象，不能写成发作或恢复结论。下一阶段是否扩展 transfer support 或细化 `(0,1)`，需另行预注册；本轮没有启动。

性能 P2：当前 vectorized integration 每个 Euler step 仍逐对象重建并校验参数数组，导致 wall 约 30 min。它不改变本轮数值，但重复运行前应把参数数组移到积分循环外预计算。
