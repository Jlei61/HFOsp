# FCXR-LC5v2.1 — dose-boundary patch

状态：**EXECUTED — CLOSED（2026-08-14）**
日期：2026-08-13

> 11 格补点与唯一候选续跑均已完成；最终归档见
> `docs/archive/topic4/sef_hfo/fcxr_lc5v2p1_timescale_dose_map_closeout_2026-08-14.md`。

## 科学问题

在基础 3x3 已完成 8/9、且尚未查看 `tau15/Gamma0.020` 结果时，沿已经显现的动力学边界补一个固定
11 格块，检验 `ESCALATING_SATURATION` 与 `ENTRY_BLOCKED_WITH_IED` 之间是否存在
`CONTAINED_HIGH_NO_OFFSET` 或 `FINITE_EXCURSION_OFFSET`。

## 固定补点

- `tau=3 s`: `Gamma={0.030,0.040,0.060}`，检验短记忆的终止 authority；
- `tau=8 s`: `Gamma={0.006,0.007,0.008,0.009}`，填充 0.005--0.010 边界；
- `tau=15 s`: `Gamma={0.001,0.002,0.003,0.004}`，填充 0--0.005 进入边界。

机器唯一参数源为
`config/topic4_fcxr_lc5v2p1_boundary_patch.json`。11 格作为一个整体完成，不按单格科学结果提前停止，
也不在运行中追加点。

## 不变合同

机制、source、q99 deadband、fresh-t0、动态 Z/H、X=1、M=0、noise/connectivity、输入 hash、
18--25 s event-aligned 观察窗和标签完全沿用主 LC5v2.1 spec。三个 tau 的 `a_U/p0_i` 不重新拟合；
新 Gamma 的 `Imax` 仅按已锁的 `force/excess` 比例解析线性换算。

科学标签不是停机条件。只有 control/input、calibration/manifest/mechanism hash 或数值/资源错误可以停。
最多 4 个本补丁 arm 并行；启动时连同已有外部 40k worker 一并按当前 `MemAvailable` 计入安全余量。

本补丁仍只寻找 lifecycle scaffold。没有 offset 就不能声称 recovery、returning IED 闭环或患者样发作。
