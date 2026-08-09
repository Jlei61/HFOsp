# FCXR-LC4b：exact-dead-zone 保住基线与进入面，但未形成自主终止/恢复

日期：2026-08-10

分支：`codex/topic4-fcxr-lc3`

设计：`docs/superpowers/specs/2026-08-09-topic4-fcxr-lc4b-exact-deadzone-design.md`

结果根：`results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4b_deadzone_lifecycle/`

最终机器判决：`F2_NOMINAL_LIFECYCLE_INCOMPLETE`

## 0. 白话结论

LC4 的平滑 Hill 终止器即使把间期电流压到万分之一量级，仍会重排间期 IED 统计。LC4b 因此加入真正的低负荷死区：每个细胞的负荷低于间期与发作负荷分布之间的预注册阈值时，执行电流严格为零。

这个修改做对了两件关键的事。第一，同一底座、同一噪声下，候选与 actuator-off 的 12 秒间期群体率和 active fraction 逐位相同，终止器电流全程精确为零。第二，冻结 `D10` 时网络仍会进入高密活动，说明它没有把 D/Z 的进入面推没。

但 70 秒连续无 kick 轨迹没有闭环。网络在 5 秒进入，随后 65 秒都维持有界、非 refractory-ceiling 的高密自终止事件串，直到记录结束仍未自主 offset。最后 8 秒有 72 次短事件，即 9.0 次/秒；其中位时长 27.5 ms、中位参与度 0.137，三项均不在冻结的 returning-IED 参照范围内。因此没有 postictal suppression、没有统计恢复，exact-final-D 12 秒确认也按合同被锁住。

最直接的归因是剂量没有从离线负荷分布迁移到真实闭环高态。候选按既往终止臂配平到平均电流 44.862，但连续轨迹里执行电流最高只有 26.297，即 58.6%；`a_mean` 最高仅 0.296，而离线 ictal 标定期望 0.506。同时 X 的全场平均可用度最低仍为 0.714，两个核心的区域均值最低约 0.56--0.58，远高于 GX1 中真正能压回低态的 0.1 档。安全结论是：**exact dead zone 解决了 baseline leakage，却没有让当前 H/X 终止通路在自然闭环高态中达到足以跨 offset surface 的幅度。**

## 1. D0：候选锁定

唯一候选由冻结逐细胞负荷产物解析得到：

- `tau_m=1000 ms`；
- 间期最大负荷 35.9903，ictal settled 最小负荷 57.6744；
- `m0=46.8324`，正好位于两者之间；
- `K_excess=19.8695`，`n=4`；
- `tau_on=100 ms`，`tau_off=10000 ms`；
- 离线 ictal 平均激活 0.5055；
- `g_max=88.7437`，使离线平均电流配平到 44.8619。

间期所有存储细胞的激活精确为零，候选通过 `DEADZONE_IDENTIFIABLE`。未根据任何新仿真结果改阈值、指数、剂量或时间常数。

## 2. D1：配对基线严格同一

seed1/noise401、固定 `D_healthy`、relay=1、无 kick，运行 12 秒。actuator-off control 与 dead-zone candidate 的结果均为：

- 计分窗 8 个 returning IED，0.80/s；
- IEI CV 0.6053；
- duration 9 ms；
- participation 0.04544；
- population-rate trace 逐位相同；
- active-fraction trace 逐位相同；
- 候选 `a_i` 与执行电流全程精确为零；
- finite、zero clip、`tau_eff_min=0.2739 ms`。

因此 D1 为 `DEADZONE_BASELINE_INERT`。这直接修复了 LC4 的功能基线失败，不只是“电流很小”。

## 3. D2：D/Z 进入面仍可达

`D_healthy` 继续保持间期。候选在 frozen `D10` 下于第 2 个 1 秒窗进入 sustained high-density regime；actuator-off 的冻结阳性对照在第 7 秒进入。候选全程有限、零 clip，最大执行电流仅 1.431。

D2 判为 `ONSET_SURFACE_RETAINED`。它只说明 D/Z 进入面仍然存在，不说明已经得到合格发作 carrier 或闭环。

## 4. D3 nominal：进入后不退出

唯一 70 秒生产轨迹从 rest 出发，dynamic Z/H/X，dead-zone 执行器从 t=0 开启；无 kick、无 reset、无 parameter step。

- onset：5.0 s；
- onset 前 returning events：12；
- onset 前时长只有 5 s，未达到合同要求的 8 s；
- classifier bout：`[5,69]`，持续 65 s，跑到记录末端；
- autonomous offset：无；
- refractory ceiling fraction：0；
- finite、zero clip、`tau_eff_min=0.2747 ms`；
- 最后 8 s：72 个 self-terminating events，9.0/s，median duration 27.5 ms，median participation 0.1371；
- frozen reference：event rate 0.086--3.15/s，duration 8--22 ms，participation 0.0445--0.0795；
- `final_D_mean=0.2549`，`final_X_mean=0.8693`。

单个微事件会自己结束，不等于网络回到间期统计邻域。末窗的 72 个事件全部嵌在持续高密 regime 中，不能用 `n_returning=72` 包装成 returning-IED recovery。

## 5. 哪些结论成立，哪些不成立

允许写：

- exact low-load dead zone 在锁定种子/底座上严格保住了间期轨迹；
- D/Z frozen entry surface 仍可达；
- 一条无 kick 轨迹能在 5 秒进入数值有界、非 refractory-ceiling 的持续高密事件态；
- 当前终止器在真实闭环高态中只执行了离线配平目标的 58.6%，没有自主 offset；
- 本候选是清晰定位到 offset 的 bounded-negative。

禁止写：

- 完整 seizure lifecycle；
- autonomous termination、postictal suppression 或 recovery；
- returning IED 已恢复；
- exact-D recovery 失败（该确认根本未运行）；
- dead-zone cooperative termination 或 X 机制被一般性否定；
- 65 秒高密事件串已经符合真实 E1146 ictal morphology。

## 6. 图、工程与资源

- 图：`figures/lc4b_deadzone_lifecycle_diagnostic.{png,pdf}`；中文说明 `figures/README.md`。只画实际执行的 D1、D2、D3 nominal，没有给 gated exact-D 阶段造占位。
- 运行：所有 40k 阶段均为单 worker，`setsid nohup`、独立 session、PID、stage flock、source hash lock、RUNNING/DONE/STOP sentinel；网络断开不影响执行。
- 峰值 RSS：50.887 GiB（70 秒轨迹完整存储）；最低记录 MemAvailable 186.566 GiB；swap delta 0 MiB。
- 70 秒 nominal wall time 20644 s（约 5 h 44 min），在 6 h wall guard 内正常完成。
- 自动链在 nominal 不合格后写 `AUTOPILOT_STOP.json`，未启动 exact-D continuation；无残留本任务进程，未触碰 sibling worktree。

分层口径：engineering **green** · exact baseline **accepted** · frozen-D entry **accepted** · bounded high-density state **observed** · autonomous offset **absent** · recovery **not established** · complete lifecycle **not established**。
