# FCXR-LC4c：进入与自主退出首次同轨出现，但退出过晚且恢复未验收

日期：2026-08-10

分支：`codex/topic4-fcxr-lc3`

设计：`docs/superpowers/specs/2026-08-10-topic4-fcxr-lc4c-entry-offset-alignment-design.md`

计划：`docs/superpowers/plans/2026-08-10-topic4-fcxr-lc4c-entry-offset-alignment.md`

结果根：`results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4c_entry_offset_alignment/`

最终机器判决：`F2_NOMINAL_LIFECYCLE_INCOMPLETE`

## 0. 白话结论

LC4c 只做了两项解析锁定的修复：把 H 的进入阈值移动到既有 LC3 网格中 11 秒、29 次 returning IED 的进入锚点；把 exact-dead-zone 终止器的最大剂量从 88.744 提到 151.395，使它在上一条真实闭环高态所观察到的激活幅度上理论上能达到既往终止靶电流。死区、Hill 指数、激活/释放时间常数、Z/H/X 方程、空间轴、连接种子和噪声种子均不变。

进入修复明确成功。新鲜 15 秒轨迹无 kick、无 reset、无参数 step，于 11 秒进入，进入前有 29 次 self-terminating returning events；前 4 秒终止器执行电流严格为零，轨迹有限、零 clip、没有 refractory plateau。这不是从旧高态分叉，也不是用短暂刺激制造的进入。

70 秒 nominal 轨迹第一次在同一条无外部干预记录中同时出现了进入和自主退出：11 秒进入，66 秒退出，退出后 4 秒没有新事件且平均活动低于进入前。但这还不是合格 lifecycle。高态持续 55 秒而合同要求 1–5 秒；offset 后只剩 4 秒，短于 8 秒统计恢复窗，而且这 4 秒内没有 returning event。最后固定 8 秒里虽然有 36 个被 event detector 标记为 returned 的事件，但它们全部发生在 62–66 秒、即 offset 之前；把它们包装成 postictal returning IED 会把高态尾部混入恢复窗。

因此本轮的安全结论是：**entry coordinate 和 offset authority 已经在同一条自主轨迹里同时可见，但 offset latency 仍错一个数量级，distributional return 尚未被观察。** exact-final-D 12 秒确认按合同未启动。

## 1. C0：唯一候选

锁定候选：

- `theta_h_lc2=1.7317735254764568`，来自已执行 `theta_scale=1.1` 行；
- `g_m_max=151.3946389128093`；
- `deadzone=46.83235549926758`，`K=19.869522094726562`，`n=4`；
- `tau_adp=1000 ms`，`tau_a_on=100 ms`，`tau_a_off=10000 ms`；
- 匹配靶电流 `I_target=44.8619393917937`。

候选不是网格，也没有看 C1/C2 后再选参数。C0 判为 `ENTRY_OFFSET_REPAIR_IDENTIFIABLE`。

## 2. C1：进入对齐通过

15 秒 fresh-from-rest 轨迹：

- onset：11.0 s；
- onset 前 returning events：29；
- first 8 s 无 qualifying ictal bout；
- first 4 s executed current 最大值：0.0；
- finite、zero clip、`tau_eff_min=0.27469 ms`；
- refractory-ceiling fraction：0。

C1 判为 `C1_ENTRY_ALIGNED`。这证明 exact dead zone 没有通过基线泄漏替代 D/Z 累积进入。

首次 C1 运行曾在仿真结束后的判决阶段因事件字段误写为 `t_on_ms` 而失败；canonical producer 使用 `t_on`。该次没有科学判决，证据保留在 `superseded/c1_event_schema_failure_2026-08-10/`。修复只改读取字段和测试，候选、种子、时长和门槛均未变化，原协议重跑后通过。

## 3. C2 nominal：有自主退出，但太晚

70 秒 fresh-from-rest 轨迹：

- onset：11.0 s；
- onset 前 returning events：29；
- classifier high bout：`[11,65]`，offset=66.0 s；
- bout duration：55.0 s，合同为 1–5 s；
- autonomous offset：通过；
- 2 秒内无快速复燃：通过；
- postictal rate：2.419 Hz，低于 pre-onset 4.980 Hz；
- finite、zero clip、non-refractory；
- offset 后观察：4.0 s，returning events=0；
- final `D_mean=0.14560`，`X_mean=0.92213`，`a_mean=0.13437`。

终止器在高态 11–66 秒的平均执行电流为 23.768，中位数 24.000，最大值 39.594，即峰值达到靶电流的 88.3%，但仍长期与高态拉锯。offset 后 4 秒平均执行电流仍约 25.025，说明慢释放确实提供了短期 postictal protection；同时 D 从 offset 附近约 0.31 降到末端约 0.15，路径开始返回但尚未回到既有安全低-D 邻域。

与 LC4b 的“70 秒内无 offset”相比，本轮观察到 offset；但 LC4c 同时改变了 H 进入阈值和终止剂量，所以本 sprint 单独不能把 offset 因果全归给剂量增大。允许说“锁定组合出现自主 offset”，不能说已隔离证明某个单参数导致 offset。

## 4. 为什么不是完整 lifecycle

硬失败有三层：

1. carrier duration 55 秒，不是预注册的 1–5 秒；
2. offset 后只有 4 秒记录，不足以建立 8 秒统计恢复；
3. post-offset 没有任何 event，无法比较 event rate、duration、participation 和传播统计。

最后 8 秒 `[62,70]` 的 36 个事件都发生在 offset 前。机器 gate 正确把 `return_window_after_guard`、`return_window_interictal` 和 `returning_reference` 判为 false。exact-final-D continuation 也正确被锁住。

另外，本轮只使用 connection seed1/noise401。即使 nominal gate 通过，也只能称单 seed candidate；本轮更不能声称稳健生命周期或患者发作表型复现。

## 5. 下一步的最小机制含义

当前缺口已从“有没有终止权”收窄为“终止器何时达到足以跨 offset surface 的有效剂量”。前 5 秒高态内，激活均值到 bout 第 5 秒才约 0.113；按当前 `g_m_max` 只执行约 17.1，而既往匹配靶电流为 44.86。到稳定高态后平均电流约 24–26，网络可维持几十秒并最终随机/动力学退出。

下一轮如果继续，必须保持 exact dead zone、H 进入锚点和所有慢方程不变，只允许做一次预注册的 offset-latency 对齐：用本轮已存 activation trace 在 `onset+5 s` 的值解析反推剂量，使终止靶电流在 5 秒边界前可达。该候选必须重新跑 fresh 15 秒 entry gate 和 70 秒 nominal；不能从高态分叉冒充 lifecycle，也不能放宽 1–5 秒、8 秒恢复或 returning-reference 门。

## 6. 工程、图和资源

- 所有生产仿真均为 `setsid nohup`、独立 PID/SID、stage flock、source lock、RUNNING/DONE/STOP sentinel；C2 为单 worker、单线程。
- C1 wall time 2634.2 s，峰值 RSS 14.835 GiB。
- C2 wall time 16581.5 s（4 h 36 min），峰值 RSS 50.886 GiB。
- 启动时 swap 714.789 MiB，结束仍为 714.789 MiB；delta 0 MiB。
- nominal 不合格后 autopilot 写 `AUTOPILOT_STOP.json`；没有启动 exact-D confirm；无本任务残留进程。
- 图：`figures/lc4c_entry_offset_alignment_diagnostic.{png,pdf}`；中文说明 `figures/README.md`。只画实际完成的 C1 和 C2 nominal。

分层口径：engineering **green** · cumulative entry **accepted at one development seed** · autonomous offset **observed but late** · post-offset suppression **short-window positive** · distributional recovery **not tested** · exact-D return stability **not tested** · complete lifecycle **not established**。
