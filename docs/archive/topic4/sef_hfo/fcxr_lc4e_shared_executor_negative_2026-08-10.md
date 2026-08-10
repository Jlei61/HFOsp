# FCXR-LC4e spatially shared executor — bounded-negative closeout

## 一句话结论

在完全相同的逐细胞负荷、dead zone、Hill 曲线、时间常数、增益和无 kick 入场轨迹上，只把终止电流从逐细胞局部分配改为全 E 细胞共享，确实消除了“核心被压住、轴外逃逸”的空间失败形态，但高态仍持续到 18 s 记录末端，没有自主终止。共享执行同时把峰值剂量从 51.4 压到 20.3，因此这轮关闭的是当前闭环实现，不是“活动依赖 relay depression”整个机制家族，也不是累计剂量匹配的空间因果检验。

## 测了什么

LC4d 的局部终止器可能只压住最活跃的核心，使高态转移到轴外组织。LC4e 保持每个细胞的负荷传感、激活曲线和群体平均剂量合同不变，只把已经计算出的激活度在全体 E 细胞之间均匀执行。该单轴对照问：空间分配本身是否是没有 offset 的原因。

## 因果与工程验收

- archived local 与 fresh shared 都在 11 s 自主进入，进入前有 29 个 returning events；
- 两臂终止电流均在 11.83 s 首次出现，在此之前电流严格为零；
- rate 和 activity-fraction 的全部 causal prefix 逐位相同；
- 40k、connection seed 1、noise 401，未使用 kick、reset、state fork 或 parameter step；
- 数值 finite、clip fraction 为 0、refractory ceiling fraction 为 0；
- 峰值 RSS 16.687 GiB，swap 只增加约 1.1 MiB，任务正常写出 DONE/STOP 后退出，无残留进程。

## 科学结果

两臂的高态都从 11 s 持续到 18 s 记录末端，下界 7 s，均无 offset。shared 臂的最终 core/axial/off-axis H 约为 3.12/3.34/2.50，而 local 臂约为 0.085/0.082/1.12。也就是说，共享执行不再把核心清空并把载体留在轴外；它留下的是一个更广泛存活的高态。

同时，shared 闭环的峰值执行电流仅 20.339，local 为 51.417。空间共享一开始压低了全体活动，也同步压低了驱动终止器的逐细胞负荷，因此执行器没有维持预期剂量。这个结果说明“局部空间逃逸”是真实失败形态，但不是唯一阻断。

## 允许和禁止的结论

允许：在该锁定候选和一个开发 seed 上，改变空间分配不足以产生自主 offset；LC4 cell-load actuator 不应继续通过提高同一增益抢救。

禁止：不能称为完整 lifecycle、不能声称 shared inhibition 普遍无效、不能否定 X relay 或 recruited-area termination，也不能把单 seed 架构比较写成患者机制。

## 路由（2026-08-10 审阅后订正）

按原 design §7 的 `both negative` 分支，LC4 具体实现停止，不再扫 `g_m_max`。但其科学解释收窄为：closed-loop sharing 消除了区域逃逸，同时因 sensor–actuator 自限减少累计剂量。后续若再研究 X，只允许做等剂量 yoked replay、trajectory-conditioned offset surface 或作用通道归因；这些是诊断，不再是 lifecycle 主路线。生命周期主路线转向独立的逐细胞 episode-load recovery coordinate，见 LC5 spec/plan。

## 产物

- `results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4e_spatially_shared_terminator/architecture_verdict.json`
- `results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4e_spatially_shared_terminator/latency_screen_traces.npz`
- `results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4e_spatially_shared_terminator/figures/lc4e_shared_executor_screen.png`
