# FCXR-LC4d：单时点剂量放大未在 5 秒内终止空间高态

日期：2026-08-10

分支：`codex/topic4-fcxr-lc3`

设计：`docs/superpowers/specs/2026-08-10-topic4-fcxr-lc4d-offset-latency-alignment-design.md`

计划：`docs/superpowers/plans/2026-08-10-topic4-fcxr-lc4d-offset-latency-alignment.md`

结果根：`results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4d_offset_latency_alignment/`

最终机器判决：`OFFSET_LATENCY_REPAIR_INSUFFICIENT`

## 0. 白话结论

LC4c 已经在一条无 kick 轨迹中得到 11 秒进入和 66 秒自主退出，但高态持续 55 秒。LC4d 没有扫参数，只做一次解析修复：读取 LC4c 在 onset 后第 4 秒的激活均值，把最大执行增益从 151.395 提到 734.169，使旧轨迹上理论电流等于既往终止靶值 44.862。死区、Hill 指数、所有时间常数、Z/H/X、空间轴和种子均不变。

新鲜 18 秒轨迹仍在 11 秒自主进入，进入前 29 次 returning events，前 4 秒执行电流严格为零，数值有限、零 clip、没有 refractory plateau。这说明增益提升没有靠间期泄漏取消进入。

但高态一直延续到记录末端，至少 7 秒，超过锁定的 1–5 秒范围；没有自主 offset、2 秒保护窗或恢复证据。因此 autopilot 正确停止，没有启动 70 秒 nominal 和 exact-D。

失败不是简单的“电流仍然太小”。电流在 15.84 秒首次超过 44.862 靶值，峰值达到 51.417，但全网高态仍在。区域轨迹显示核心和轴带先被压低，活动和 H carrier 留在轴外：17.75 秒两核平均 `y≈1e-5, H≈0.085`，而轴外 `y≈13.30, H≈1.12`。安全结论是：**基于旧轨迹单时点激活度反推 cell-local 增益，在实际闭环中既被自身负反馈削弱，又未提供空间一致的终止权限。**

## 1. L0：解析候选与它的隐含假设

候选只改变：

```text
g_m_max: 151.3946389128093 -> 734.1686843528613
```

推导使用：

```text
I_target = 44.8619393917937
a_LC4c(t=15 s) = 0.06110576540231705
g_m_max = I_target / a_LC4c(15 s)
```

该等式在旧的 LC4c 轨迹上成立，但默认新增益不会改变 15 秒的激活度。L1 实测推翻了这个开环假设：新闭环在 15 秒的 `a_mean=0.025119`，执行电流仅 18.442，即靶值的 41.1%。负反馈压低放电，也压低了驱动自己的逐细胞负荷和激活度。

## 2. L1：进入保留，退出延迟未修复

- onset：11.0 s；
- onset 前 returning events：29；
- first 8 s 无 qualifying bout；
- first 4 s current 最大值：0.0；
- classifier bout：`[11,17]`，延续到记录末端；
- bout duration 下界：7.0 s；
- offset：未观察；
- max rate：386.875 Hz；mean rate：16.006 Hz；
- max executed current：51.417；
- `tau_eff_min=0.2674 ms`，clip=0，refractory fraction=0。

电流从 onset 到 onset+4 秒的平均值只有 4.020；onset+4 到记录末端平均 44.042。它在 15.84 秒才首次达到靶值，已经晚于 5 秒 offset 边界的有效余量。更重要的是，达到靶值后高态仍未消失，所以不能再把下一步写成同一公式的更大增益外推。

## 3. 空间读数：局部抑制伴随轴外持续

区域 `y/H` 读数从 onset 后发生交换：

| 时间 | 两核 mean y | 轴外 y | 两核 mean H | 轴外 H |
|---:|---:|---:|---:|---:|
| 13.0 s | 80.82 | 43.80 | 3.51 | 1.71 |
| 15.0 s | 55.61 | 68.27 | 3.06 | 3.37 |
| 17.0 s | 0.0054 | 25.03 | 0.357 | 1.63 |
| 17.75 s | 0.00001 | 13.30 | 0.085 | 1.12 |

这支持“cell-local 执行器先压低高负荷核心，而剩余 carrier 位于轴外”的读法。它还不是严格的空间迁移因果证明，因为没有 matched spatially coordinated actuator 对照；也不能据此否定 X、非局部 inhibition 或 recruited-area termination。

## 4. 为什么本 sprint 必须停

spec 明确写死：L1 高态超过 5 秒或跑到记录末端即 `OFFSET_LATENCY_REPAIR_INSUFFICIENT`，不得在同一 sprint 增加剂量。当前结果同时暴露闭环负反馈和空间逃逸；继续沿单一 `g_m_max` 轴外推会把两个问题混在一起，也不能回答完整 lifecycle 的真正缺口。

因此：

- L2 70 秒 nominal 未运行；
- exact-D 未运行；
- seed3/unseen noise 未运行；
- 没有 postictal suppression 或 returning-IED recovery 数据；
- 不称完整 lifecycle、bistability、患者样 ictal morphology 或稳健终止。

## 5. 下一步的最小机制问题

下一阶段不应再问“局部增益要多大”，而应先问：

> 在保持同一 D/Z 无 kick 进入、exact dead zone 和局部 carrier 的情况下，把同一持续负荷转成空间协调的终止作用，能否在不取消 1–5 秒 carrier 的前提下同时消除核心与轴外高态？

最小因果对照应使用同一总平均执行剂量，比较 cell-local 与空间共享/非局部执行；传感器仍由实际负荷驱动，不能读取 classifier 标签。只有空间协调臂在 matched dose 下终止，而 local 臂复现本轮轴外残留，才有权进入 70 秒 lifecycle 和 X/recovery 验收。

## 6. 工程与资源

- L1 由 `setsid nohup` 启动，独立 PID/SID、单 worker、单线程；
- wall time 3343.14 s（55.7 min）；峰值 RSS 16.687 GiB；
- swap 启动/结束均为 714.789 MiB，delta 0；
- MemAvailable 结束时 221.0 GiB；
- autopilot 在 L1 阴性后写 `AUTOPILOT_STOP.json` 并退出；无残留任务进程；
- 图：`figures/lc4d_offset_latency_screen.{png,pdf}`，中文说明见 `figures/README.md`。

分层口径：engineering **green after sentinel cleanup repair** · cumulative entry **preserved at one development seed** · cell-local one-point latency repair **bounded-negative** · spatially coordinated termination **not tested** · recovery **not tested** · complete lifecycle **not established**。
