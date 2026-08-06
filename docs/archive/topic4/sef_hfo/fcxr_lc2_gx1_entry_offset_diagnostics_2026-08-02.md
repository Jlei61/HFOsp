# FCXR-LC2-GX1 frozen entry/offset diagnostics — 2026-08-02

## 一句话结论

GX1 在不改方程、不接动态慢变量的条件下，分别检验现有 H 方程是否自带易感性选择窗，以及 X
理论最大关断是否有权把 H 高态拉回间期。正式结果是：

- S1：`NO_NATURAL_SELECTIVITY_WINDOW_IN_LOCKED_STRIP`（0/12 点通过，
  0 个点属于相邻窗）；
- X1：`X_PATH_REACHABLE_RANGE_INSUFFICIENT`；
- 原预注册局部路由：`LOCAL_D_DEPENDENT_H_GAIN_ONLY_X_RANGE_SEPARATE`；
- 经核心科学目标复审后，下一获准程序：`LC3_DX_STATE_PLANE_AND_SPATIAL_INSTABILITY_AUDIT`。

这不是一个笼统的双阴性。S1 在 H1 theta_scale=1.25 的 3 个 rho 点上都看到了同一分解：
健康 `D=0/H_low` 保持 4.2 Hz 的间期工作点，而易感 `D=0.15/H_low` 已经升到
54.8--91.5 Hz；易感高初值也维持 58.7--87.6 Hz。也就是说，现有方程已经出现
**D 选择性的单向点火**，但同一个易感 D 下低初值和高初值都落到高态，因此不是低/高双盆地。
这不等于 onset 机制失败：合法的 seizure onset 可以来自单稳态分支交换、Hopf、SNIC、
noise-assisted transition 或 slow-wave bursting，并不要求同一点迟滞。当前 H1 因而被验收为
`D_SELECTIVE_ONSET_CANDIDATE`，尚待 D-X 状态平面和空间稳定性检验。

X1 则给出清楚的权限括号：availability=1.0/0.5 仍为高态（尾段
99.9/78.2 Hz），
0.1/0.0 在末段连续 2 s 回到间期（尾段
0.061/0.058 Hz）。
所以 H 没有结构性绕过 X。同一 anchor 上归档的 LC2 E4 frozen-relay 负载臂为
0.872（尾段 97.6 Hz，仍为高态）、0.786（尾段 94.9 Hz，仍为高态），状态 `INSUFFICIENT_FOR_THIS_H_BRANCH`——即当前 relay 实际达到的负载区间不足以终止这条 H 分支，
能终止的 0.1 远在其之下，且不具有生理标定资格。
但该结论只在固定病理 `D=0.15` 下成立；完整系统可能通过 `X↑ -> rate↓ -> Z恢复 -> D↓`
共同跨过 offset surface，因此 `COUPLED_D_X_OFFSET_UNTESTED`，目前不能直接判定必须增强 X。

## 本轮结论覆盖不到的两处

1. **易感高初值臂不检验 H 的时间常数和阈值。** 该臂从 `H(0)=2*theta` 出发，门
   `sigmoid((h-theta)/k)` 在整段运行里被钉在 1（最小值
   0.999999990），于是 `rho*S_H(h)` 退化成常数附加电导 `rho`，`tau_H` 与
   `theta_H` 从方程里掉出去。12 个参数点在这一臂只产生
   3 条不同轨迹（按 rho 分组逐位相同）。因此"12 点都验证了高态可维持"
   是对覆盖度的高估：可维持性只在 rho 这一个轴上被验证过。真正区分 H 家族的是两条低初值臂。
2. **offset 臂没有从收敛的高分支出发。** 四条 X 臂都从 `H(0)=2*theta=2.225` 开始，
   而 availability=1 跑到的自洽高分支是 `H=6.817`（3.06 倍）。
   若从收敛值出发，H 衰减到 theta 以下多需要 0.71 s，
   仍显著短于 5.06 s 的记录长度与 2 s 的判定窗
   （余量充足=true），所以本轮结论在这个界内成立；但严格意义上被检验的是
   "高 H 起步能否被压住"，不是"已收敛发作能否被掐断"。

## 测了什么

S1 固定 connection seed 1 / noise 401，在 H1/H6 两个既有家族上扫描低于旧下界的三个 H 增益
和两个阈值尺度。每点同时要求健康低初值、易感低初值保持间期，且易感高初值保持有限高态。相邻
两点同时通过才算自然参数窗。

X1 从同一个解析高 H 初值出发，把 recurrent relay availability 冻结为 1、0.5、0.1、0，检验
现有 X 路径的理论最大终止权限。x=0 只是一条结构性因果探针，不是生理参数。

## 科学边界

本轮只允许说明 frozen entry/offset component control。没有接 dynamic Z/X，没有跑无 kick
lifecycle，没有测试 M 形态、K 招募、A/ELR，也没有比较真实 E1146 ictal morphology。因此不能称为
迟滞、双稳态、极限环或可恢复发作闭环。

## 核心科学目标复审与下一授权边界

GX1 原预注册决策表落在“no natural window + X path reachable”，局部路由原本指向显式
D-dependent H gain。然而，同一 D 的双盆地不是完整 lifecycle 的必要条件，而且 H1 已出现
健康低态到易感高态的单稳态型 candidate。直接加入显式 D gate 会把“Z 控制 onset”写进方程，
存在先射箭再画靶的风险。

因此正式授权改为：**保留当前 H1 方程，先完成 D-X 状态平面、early spatial mode audit 和
由状态平面约束的 dynamic no-kick pilot。** 只有 D-X 平面不存在可闭合路径、transition 在多 seed
下不稳，或 leading mode 是全局共同模态时，才允许改用 local E/I-balance H sensor。显式 D gate
只保留为后续 mechanistic control。GX2 `D gate × shared X/H path` 2×2 继续不得执行。

## 工程与资源

- strip trajectories: 36; X trajectories: 4;
- numerical safe: 40/40;
  numerically failed strip points: 0;
- the six blessed engine files were checked by the execution lock.  The module that implements the H gate
  and the frozen relay (`src/snn_engine/mz_slow_vars.py`) is **not** in that blessed set; it was last
  modified in `fe9674a2` (2026-08-02 01:30 +0800), before the GX1 lock, and `cmd_lock` now pins it under
  `mechanism_module_hashes` for future runs;
- long stages used setsid/nohup, exact PID watchdogs, stage locks and sentinels;
- S1 watchdog 5.933 h; X1 watchdog 1.008 h;
- peak single-cell RSS 11.236 GiB over 40 recorded trajectories;
  swap delta 0 MiB;
- the spec's `+256 MiB stop new submission` rule is **not** implemented: all futures are submitted up
  front, so only the `+512 MiB` hard stop is active, and it tears down the whole stage rather than the
  newest worker;
- final commit and test counts are recorded in `run_manifest.json` after sign-off.
