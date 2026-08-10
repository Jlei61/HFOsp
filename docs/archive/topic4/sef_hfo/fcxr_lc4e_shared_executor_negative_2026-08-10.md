# FCXR-LC4e spatially shared executor — bounded-negative closeout

## 一句话结论

在完全相同的逐细胞负荷、dead zone、Hill 曲线、时间常数、增益和无 kick 入场轨迹上，只把终止电流从逐细胞局部分配改为全 E 细胞共享，确实消除了“核心被压住、轴外逃逸”的空间失败形态，但高态仍持续到 18 s 记录末端，没有自主终止。按预注册规则，这关闭 LC4 cell-load actuator 家族，下一步回到已经实测存在的 X relay offset surface。

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

## 路由

按 design §7 的 `both negative` 分支，停止 LC4 cell-load actuator 家族。下一步优先复用已归档的 X 冻结边界（高磨损下 X=0.395 维持、X=0.380 返回低态）和动态 hill-placement 结果，直接检验在当前无-kick D/Z entry 上，X 的已知可达深度能否在 1–5 s 内自主终止并提供足够的 postictal protection。不得再扫 `g_m_max`。

## 产物

- `results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4e_spatially_shared_terminator/architecture_verdict.json`
- `results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4e_spatially_shared_terminator/latency_screen_traces.npz`
- `results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4e_spatially_shared_terminator/figures/lc4e_shared_executor_screen.png`
