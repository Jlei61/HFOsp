# Continuous marked-state T2 超长尺度发现实验合同（2026-08-26）

## 1. 科学问题

本轮直接检验：IED 对状态的影响是否需要累计几千到上万次才出现。它不是单个 IED jump 的放大版，也不把名义窗口长度自动解释为真实生理记忆。

所有患者先按同一连续记录段的支持度选择，再训练任何新模型。正式检验分区保持封存；这是 development exploration，不产生队列 p 值或因果结论。

## 2. 全队列支持度审计

权威机器表：`results/epi_prssm/continuous_marked_state/r1/t2_long_total_effect/cohort_support/summary.json`，revision `t2_very_long_full_cohort_train_validation_contiguous_support_v2`。

训练和验证必须各自至少有窗口；每个窗口不得跨未记录缺口，并额外保留 1,000 次因果延迟对照的历史。

| 患者 | 最大主尺度 | TRAIN 窗口 | validation 窗口 | validation 中位时长 | 角色 |
|---|---:|---:|---:|---:|---|
| `yuquan_chengshuai` | 15,000 次 | 545 | 5,515 | 11.73 h | 事件数和真实时间都最长的主对象 |
| `yuquan_zhangjiaqi` | 10,000 次 | 11,905 | 4,715 | 5.93 h | 支持强，但既有 3 seed T1 全退化，只保留为仪器失败记录 |
| `yuquan_pengzihang` | 5,000 次 | 8,870 | 569 | 2.44 h | 中等时长、高事件数 |
| `yuquan_chenziyang` | 4,000 次 | 764 | 1,281 | 6.17 h | 较少事件但较长真实时间 |
| `yuquan_hanyuxuan` | 2,000 次 | 279 | 1,094 | 9.58 h | 低事件率、长时间对照 |
| `epilepsiae_922` | 3,000 次 | 1,379 | 3,749 | 0.96 h | 高事件率、短时间对照 |

程帅的 20,000 次虽有 1,060 个 validation 端点，但 TRAIN 为 0，不能拟合后再评分，因此不作为当前主实验。韩宇轩 3,000 次、陈子阳 5,000 次、E922 5,000 次同理降到上表的最大可训练尺度。

## 3. 前状态仪器

每位发现患者运行 7 个 seed 的 R1.2 frozen-explicit T1。只有 `selected_epochs > 0` 的 seed 进入 H3；退化 seed 记录为 `UNTESTABLE_T1_INSTRUMENT_DEGENERATE`，不记作 H3 阴性。不同患者互不 gate，一人的普通失败不停止其他患者。

这一 T1 比 formal R1.3 弱，作用是先建立可评分的 development 前状态，不得与原三位 formal R1.3 的 H1/H2a 证据合并。

## 4. 两种累计记忆

### 4.1 主分析：whole-window boxcar

在整个指定窗口内，每次 IED 的 exposure 权重相同：

\[
x_{e,N}=N^{-1/2}\sum_{j=e-N}^{e-1}\eta_j,
\]

其中 \(\eta_j\) 是只用 TRAIN 拟合的 load innovation。这个累计量不经过旧 T1 generator 衰减，所以 10,000 次就是实际使用 10,000 次，而不是只剩最近一小时。

固定 N 时，IED occurrence 数恒等于 N，与截距不可区分；因此固定 N 的可识别信息是累计 load/mark composition，不把它写成“事件次数本身的作用”。6 h 窗口中事件数会变化，可作为辅助 exposure。

### 4.2 敏感性：generator-weighted memory

保留旧算子作为“近期加权”敏感性，并强制报告实际权重：实际有效事件数、50%/90% 权重年龄及一小时前权重比例。不得用名义 N 给它命名为长时程机制。

韩宇轩 seed 0 的冒烟结果已经说明两者不同：名义 2,000 次横跨 9.58 h；旧算子 90% 权重只在最近 1.88 h，而 boxcar 的 90% 权重覆盖 8.72 h。

## 5. 对照和端点

主对比：

1. real cumulative exposure vs 同容量 intercept；
2. real cumulative exposure vs 因果延迟 1,000 次的 exposure。

延迟对照与超长窗口会高度重叠，属于局部时刻敏感性，不足以单独排除共同慢漂移。`real_minus_no_edge` 继续只报告为截距伪迹量，不作为 exposure 证据。

分别报告 timing、STOP、selecting size、contact subset。若某 decoder block 的 TRAIN 变化落在 scale floor，该端点单独标为弱仪器；不能用其他端点替它下结论。重叠窗口数不是样本量，同时报告端点跨度和按状态时间常数估算的有效独立窗口。

## 6. 执行与资源

- preparation 最多 2 个并发；T1 最多 4 个；H3 最多 3 个；每个 worker 的 OMP/MKL/OpenBLAS 线程固定为 1。
- 运行前至少保留 64–72 GiB 可用内存和 4–6 GiB 空闲显存；低于阈值自动等待。
- 原子 JSON、逐作业日志、完成结果自动跳过；`setsid` 后台运行，断线不终止。
- 当前近期加权队列 PID `3475030`；boxcar follow-up PID `3475637`，后者等待前者完成 T1 后自动启动。

输出：

- 近期加权：`results/epi_prssm/continuous_marked_state/r1/t2_very_long_discovery/`
- whole-window boxcar：`results/epi_prssm/continuous_marked_state/r1/t2_very_long_boxcar/`

## 7. 允许的结论

- 多患者、多 seed 的 real boxcar 同时胜过 intercept 和 delayed：支持“长窗口累计 IED composition 含有预测后续状态变化的信息”。
- 只有近期加权有利：支持近期历史，不支持几千上万次的整窗累计。
- boxcar 只在真实时间很长、事件较少患者有利：优先解释为 physical-time slow context；反之只在高密度患者有利，优先解释为 event-count accumulation。
- 当前实验仍是 total-effect predictive evidence。只有条件于高质量 pre-event state、加入更强的状态匹配/非重叠反事实并独立复现后，才讨论 IED shape state 的机制。
