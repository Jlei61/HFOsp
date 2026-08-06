### mz_m_gated_reserve_coupled_canary.png

这张 2x3 图只展示预注册的 Segment-A center canary。A–C 给出六次固定背景事件后区域活动、真实耦合的 q/M 轨迹和 entry ordering；D–E 给出 persistence/latch 与无外驱 section returns；F 明确本次 stop-rule 判定。

当前真实耦合结果在 event 5 后已跨 entry fold；随后 persistence latch 在 event 6 之前 set，M 把第六次响应压低，所以 final pulse 后为 0 个 section returns。这说明 scalar preentry parity 只是 feed-forward sensor 假设，闭合 q-use feedback 后不成立。以实际 trigger event 5 对齐，则在 event 6 前可描述性地看到 4 个 core/annulus 配对 returns，且 event 6 immediate retrigger 被抑制；但 late recovery、same-basin return 和 late retrigger 尚未测试，因此不能称完整 lifecycle。B–D、retrigger、ablation 和其余 17 条路径仍不运行；bath 的 q 由固定 mask 强制保持，只能作为诊断。

**关注点**：C 中第 5 个点已低于 fold；D 中 latch 在第六次事件前开启；E 中四组竖线是 actual-entry-aligned returns，紫色虚线是 event 6。formal 结论仍是 numerically clean premature-entry no-go；四 returns + immediate retrigger suppression 只记为 descriptive lifecycle candidate。
