# FCXR-LC5v2.1 — matched-dose episode-memory timescale–dose map

状态：**DESIGN LOCK — ACTIVE**
日期：2026-08-13

## 1. 唯一科学问题

在 returning IED 可自然推动 `Z/H` 点火、但 pump-off 随后升级至 refractory plateau 的 E1146
40k SNN 中，始终在线的逐细胞 episode load `U_i` 能否把这条升级轨迹变成：

1. 非饱和的有限高态（containment）；或
2. 驻留 0.5--5 s 后自主 offset，并保留 postictal memory 的有限 excursion。

当前科学语义统一为：`Z/H` 负责点火与正反馈，`H+U` 的耦合共同决定高态是否有界以及是否退出。
不得再把 pump-off substrate 称为 bounded carrier；它是 **H-driven escalating high state**。

## 2. 机制边界

每个 E 细胞只积分自身 spike history：

\[
\Phi(u_i)=\frac{u_i^3}{1+u_i^3},
\]

\[
u_i(t+\Delta t)=\max\left[0,\ u_i(t)+a_U N_i^{spk}(t)
-\frac{\Delta t}{\tau_U}\Phi(u_i(t))\right],
\]

\[
I_{U,i}=I_{max}[\Phi(u_i)-p_{0,i}]_+.
\]

`p0_i` 是从 pump-off baseline 标定的 **cell-specific deadband instrument**，不是 Na/K pump 的
生物物理常数。模型禁止读取 population rate、发作标签、区域面积、坐标或人工 mask；空间结构只来自
冻结连接与每个细胞真实的放电历史。

## 3. 冻结对象

- E1146 connectivity、两个低阈值 core、RC1 recurrent saturation；
- dynamic Z、dynamic H、`X=1`、`M=0`；
- connection seed 1、noise seed 401、`dt=0.05 ms`、`h_U=3`；
- fresh `t=0`、`u_i(0)=0`，U load/current 从第一步始终在线；
- no kick、reset、parameter step 或 onset detector；
- calibration source 为封存 U1 pump-off stream，`W_B=[7,11)s`、`W_E=[12,14)s`；
- 每个 `tau_U` 独立锁 `a_U(tau)`、temporal-q99 `p0_i(tau)` 和 `Imax(tau,Gamma)`。

本实验比较的是：**matched baseline leakage 与 matched early-episode dose 条件下，不同 episode-memory
time scale 的动力学结果**，不是只改变一个裸 `tau_U`。

## 4. 唯一 active 矩阵

\[
\tau_U\in\{3,8,15\}\ \mathrm{s},\qquad
\Gamma_U\in\{0.005,0.010,0.020\}.
\]

完整完成 9 个非零格后统一解释；单格的 saturation、contained、finite、no-onset 都是结果，不是停止
整个实验块的 token。一条复用 pump-off control 即可。既有 arm 只有在 equation/config/calibration、
external-input prefix hash 和所需观察窗全部一致时才允许复用。

机器唯一参数源：
`config/topic4_fcxr_lc5v2p1_timescale_dose_map.json`。文档与 manifest 冲突即停止执行。

## 5. 观察窗

每格至少运行 18 s。一旦检测到 onset，观察至 `onset+7 s`；若 18 s 仍未 onset 且 returning IED
仍存在，则延长至最多 25 s：

\[
T_{end}=\min[25,\max(18,t_{on}+7)]\ \mathrm{s}.
\]

因此 18 s 的 `NO_NATURAL_ONSET` 不能被解释为 entry blocked。25 s 后才区分：
`DELAYED_ONSET`、`ENTRY_BLOCKED_WITH_IED` 与 `BASELINE_SUPPRESSED`。

## 6. 判读

核心动力学标签：

- `ESCALATING_SATURATION`；
- `CONTAINED_HIGH_NO_OFFSET`；
- `FINITE_EXCURSION_OFFSET`；
- `OFFSET_WITH_REBOUND`；
- `IMMEDIATE_SUPPRESSION`；
- `BURST_SILENCE_LOOP`；
- `DELAYED_ONSET` / `ENTRY_BLOCKED_WITH_IED` / `BASELINE_SUPPRESSED`；
- `NUMERICAL_FAIL`。

active arm 不要求 pre-onset spike bitwise equality。必须报告 returning-event count、IEI、duration、
participation/peak、onset latency 和与 control 的统计距离；bitwise parity 只属于 pump-off control。

每格另报告 `u50/u90/u99/u_max`、末窗 `du/dt`、估计 release time、achieved dose，以及
`U -> I_U -> rate/I_EE -> H` 时序。`CONTAINED_HIGH_NO_OFFSET` 是正结果，不得与普通阴性合并。

## 7. 只有三类硬停止

1. pump-off control/外源输入 prefix 不复现；
2. calibration、manifest 或机制 hash 不匹配；
3. numerical/resource failure。

科学标签不触发中途停机。3x3 完成后只允许一次有目的的边界实验：相邻 saturation 与
suppression/no-onset 之间补一个中点；或延长唯一最佳 contained arm；或做一个 authority diagnostic。

## 8. 后续解锁

- 最佳 contained arm：不改参数，延长至 onset 后 15--20 s；
- finite/延长后 offset：一对 dynamic-D / frozen-D，再进入 chunked recovery；
- 完整 lifecycle 后才做 independent noise/connection seed、held-out p0 leakage、M_i morphology、
  eigenmode 与论文图。

最终目标仍是 `IED -> natural onset -> finite non-saturated high state -> offset -> postictal ->
Z recovery -> returning IED distribution`，而不是只得到压低 rate 或延迟 onset。

## 9. 资源与工程

每个 40k arm 内部严格单 worker、线程数 1；最多 4 个独立 arm 并行。启动前按实测 RSS 的 1.5 倍
逐臂预算，`MemAvailable` 至少为总预算 3 倍；swap stage-baseline `+256 MiB` 停止新提交，
`+512 MiB` 保存 checkpoint 后结束当前 worker。所有长任务使用 `setsid nohup`、arm-scoped flock、
PID、RUNNING/DONE/FAILED sentinel、1 s rolling checkpoint 与 atomic bundle。
