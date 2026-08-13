# FCXR-LC5v2.1 timescale–dose map — implementation plan

对应 active spec：
`docs/superpowers/specs/2026-08-13-topic4-fcxr-lc5v2p1-timescale-dose-map-design.md`

状态：**LOCKED FOR EXECUTION**

## T0 — 收口与唯一合同

1. 将旧 LC5v2 spec/plan 标为历史合同；
2. 把历史修订与结果移入 decision log；
3. 锁 machine-readable manifest、source/calibration/mechanism/noise hashes；
4. runner 只接受 manifest 中的 cell，不再暴露旧 stage 菜单。

## T1 — calibration parity

对 `tau={3,8,15}s`，在预锁的代表性 E-cell subset 上比较 1 ms calibration replay 与 0.05 ms
原方程，覆盖完整 `W_B/W_E`。subset 必须包括 core、axial、off-axis 与高率尾部。输出：

- temporal-q99 `p0` 差；
- `median Phi(W_E)` 差；
- q99-excess integral 差；
- cell index/category 与 source hash。

这只验证 calibration instrument，不调 `a_U/p0/Imax`。

## T2 — manifest-driven event-aligned runner

每格从 fresh t0 运行，18 s 后按当时完整 event ledger 判断：

- 已 onset：至少到 `onset+7 s`，上限 25 s；
- 未 onset 且 IED 保留：继续到 25 s；
- 每个 1 s chunk 保存 exact checkpoint、增量 spike 和资源行。

18 s 记录若已覆盖所需 onset 后 7 s，可经 hash 审计复用；18 s 无 onset 的 arm 不可复用为最终
no-onset 判决。

## T3 — 完整 3x3

运行 manifest 中 9 个非零 cell，加一条共享 pump-off control。完成整块前不根据科学标签删格。
每格输出 baseline metrics、onset/offset、lifecycle label、D/H/U/IU、saturation、achieved dose、
u-tail/release-time、输入与机制 hash。

最多 4 条并行；优先复用合格旧 arm，再按资源门分批补齐。

## T4 — 聚合与图

整块完成后生成：

1. `phase_map.json/csv`；
2. outcome、onset latency、saturation、achieved-dose 四张矩阵；
3. 代表性 saturation/contained/finite/no-onset 轨迹；
4. `figures/README.md` 中文逐图说明；
5. STATUS 与 archive，严格区分 containment、offset 与 lifecycle recovery。

## T5 — 最小后续

只按整张图选择 1--3 条：边界中点、最佳 contained 延长或单个 authority diagnostic。没有 finite/
contained 前不运行多 seed、M、eigenmode 或 70 s recovery。

## Definition of done

1. active spec、plan、manifest 三者一致且历史 runner 不能误触发；
2. 三个 tau 的完整窗口 calibration parity 有量化结果；
3. 共享 control 与 9 个非零格均有 DONE/FAILED；
4. 所有科学标签统一聚合后再解释；
5. 图、README、STATUS、archive、资源与进程审计齐全；
6. 无 contained/finite 时诚实收口，不以逐格 gate 或无限细化抢救。
