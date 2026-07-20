# Topic 4：inhibitory-reserve 二维 frozen corridor（locked design v1.0）

日期：2026-07-20

状态：R0a cheap screen 待执行；本文件先锁方程、数值轴、资源上限与 stop rule。

工作线：`.worktrees/topic4-mz-divisive-lifecycle`

## 1. 目的与独立边界

目标不是再造一个 fast ictal attractor，而是检验一个抑制侧慢变量下限能否把外层轨迹限制在已存在的 bounded `CCO` strip 内，使 additive M 不再追逐持续上移的 exit boundary。

本线固定：

- current-based P=3 core–equal-area-annulus–bath scaffold；
- canonical M3B `K_EE/K_I`、shared global pool、`Amax=1.6 mV`；
- E→E weight、kernel、delay 与 recurrent divisor；
- event-locked additive exit 定义。

禁止修改：

- `W_EE`、E→E kernel/delay；
- AMPA membrane mode、recurrent conductance、`rec_sat_g`；
- conductance 线的 `y/x` relay 与 `src/snn_engine/mz_slow_vars.py`；
- dynamic threshold `phi`（除非 reserve 后只剩 tonic plateau，且另开条件节点）。

并行 `.worktrees/topic4-mz-conductance` 当前占用 recurrent E→E fast-gain shaping；本节点只处理 inhibitory efficacy 的外层慢流，两者不合并调参。

## 2. 方程

区域 effective inhibitory multiplier 写成：

\[
q_R=q_{res}+(q_0-q_{res})D_R,
\qquad 0\le D_R\le1,
\]

\[
\dot D_R=\frac{1-D_R}{\tau_{D,r}}
-\frac{D_RU_R(x)}{\tau_{D,d}},
\]

\[
U_R=\frac{w_cU_c+w_aU_a}{w_c+w_a},
\qquad
U_j=H_\epsilon(s^{EI}_j-s^{EI,0}_j).
\]

core 与 annulus 第一版共享一个 `D_R`；bath 暂固定 `q_B=q_0=.90`，只用于与既有 frozen oracle 同坐标比较。field lift 时必须移除该 bath mask。

等价 q 方程：

\[
\dot q_R=\frac{q_0-q_R}{\tau_{D,r}}
-\frac{(q_R-q_{res})U_R}{\tau_{D,d}}.
\]

在 frozen attractor 上周期平均：

\[
\overline{\dot q}=
\frac{q_0-q}{\tau_{D,r}}
-\frac{(q-q_{res})\overline U(q,A)}{\tau_{D,d}}.
\]

若暂把 `Ubar` 当常数，慢零流形为：

\[
q_N=\frac{\tau_{D,d}q_0+\tau_{D,r}\overline U q_{res}}
{\tau_{D,d}+\tau_{D,r}\overline U}>q_{res}.
\]

因此 `q_hold` 与 `q_res` 不是同一个量。R0 只找 fast safe strip；R1 才记录 `Ubar` 并反算 floor 参数。

## 3. 为什么原一维设计不成立

已有数值：

- `A=0` 的 Z-only trajectory 到 `q≈.722` 仍没有 transfer-support failure；
- warm-cycle pilot 在 `q=.80,A=0` 仍为 bounded CCO；
- four-return autonomous failure 位于 `(q,A)≈(.821886,.490231 mV)`；
- low-root fold 插值给 `A_SN(.821886)≈.490607 mV`。

所以失败边界属于二维 `(q,A)` geometry。不能再写
`q_support + margin <= q_res < q_entry`。

定义：

- `A_exit(q)`：established CCO 在全部 registered phase/dt 下进入同一 LLL basin 的最小 A；
- `A_fail(q)`：任一 phase 首次出现 transfer/bound/nonfinite failure 的 A；
- `I_safe`：同时存在 bounded CCO 与有余量 safe exit fiber 的连续 q 区间。

reserve 的 slow-fast 作用写成：

\[
G(q,A)=A-A_{exit}(q),
\qquad
\dot G=\dot A-A'_{exit}(q)\dot q.
\]

旧模型 `A'_exit<0, dot q<0`，M 追赶 moving target；reserve 若把 CCO 内 `dot q` 拉到约 0，则 `dot G≈dot A>0`，允许 M 穿过近似固定的 exit fiber。

## 4. R0a：base-dt cheap screen

固定四个 warm-cycle phase、`dt=.125 ms`、至少 4 个 prelude returns、event-locked A 与 matched `A=0` twin。

q 轴：

```text
.8555, .8550, .8525, .8500, .8475,
.8450, .8400, .8350, .8300, .8250, .8200
```

每个 q 先解 low-root `A_SN(q)`，再测试：

```text
A = 0
A = A_SN + {0, .01, .025, .05, .10, .20} mV
```

若 failure 先于 LLL，该 q 标 unsafe，不继续扩大 A。若找到 LLL bracket，只在 R0b 对 bracket 二分到 `.005 mV`。

资源合同：单进程、单 BLAS 线程、每批 `<=96 forks`、峰值 RSS `<2 GiB`；post-fork 按最多 3 个 q/批运行，不保存全轴高频 trace，只保存 summary/CSV、checkpoint/final states 与图所需代表性 trace。

## 5. R0b：candidate strip confirm

仅对 R0a 找到的候选 strip：

- 插入中点，使 q spacing `<=.0025`；
- `dt=.125/.0625 ms`；
- 四个 phase；
- `A_exit` bracket 两侧各一个点；
- 最近一个 lower-q failing anchor；
- exit 后恢复到 `q=.90,A=0`，验证同一 LLL basin。

每个 q 的 source CCO 必须：core/annulus `>=6 returns`、period CV `<=.01`、peak drift `<=.01`、Poincare closure `<=2e-5`、无 sustained ceiling、bath 0 returns/peak `<20 Hz`、无 support/bound/nonfinite failure。

每个 event-locked exit 必须：

- A 只在 core+annulus 各至少 4 returns 后打开；
- matched A=0 继续 bounded CCO；
- threshold 以上全部 phase/dt 回 LLL；
- 在 established CCO checkpoint 后，以原 `tau_m_up=225 ms`、joint-occupancy gate 平滑增长 M、同时固定 q；该 ramp 必须无 support/bound/nonfinite failure 地回 LLL；
- final fast RHS `<1e-8/ms`；
- 全程无 support/bound/nonfinite；
- phase/dt `A_exit` spread `<=.01 mV`；
- parameter restoration 回同一 LLL basin。

## 6. R0 通过门

必须存在宽度 `>=.005`、至少 3 个节点、spacing `<=.0025` 的连续区间 `I_safe`，且每个节点满足：

\[
A_{exit}(q)+\Delta A_{margin}<A_{fail}(q),
\qquad \Delta A_{margin}\ge.02\ {\rm mV}.
\]

若 `A_fail` 在注册轴内未观察到，必须至少验证 `A_exit+.02 mV` 仍安全，才能把 margin 写成 right-censored lower bound。

R0 只证明 fast geometry 对 reserve-compatible slow flow 有空间，不证明 autonomous lifecycle。

## 7. R1 解锁后的最小映射

R0 通过后才做：

1. 在 `I_safe` 下/中/上节点记录 `Ubar_CCO(q,A=0)`；
2. 由 q-nullcline反算各自 `q_res`；
3. 用固定背景事件 sensor replay 锁一个 `tau_D,d`，使最后一个 pulse 后 q 带余量跨 entry fold；
4. 固定原 four-return M arm `p_on=.115,tau_m_up=225 ms`，不先调 M；
5. old no-reserve arm 保留为已知 no-go 对照。

不把 `q_res`、`tau_D,d`、M 参数做三维网格。

## 8. Stop rules

任一项触发即关闭 reserve 主路线：

1. 只有单个精调 q 点，没有宽度 `>=.005` 的 safe strip；
2. safe exit 前先出现 support/bound/nonfinite failure；
3. `A_exit>Amax` 或距 failure `<.02 mV`；
4. 瞬时 A-step 安全，但 fixed-q smooth M ramp 越出 support；
5. base/half-dt 或四 phase 标签不一致；
6. sustained ceiling 或 bath recruitment；
7. q-nullcline无法落入 `I_safe`，同时 fixed-event replay 无法跨 entry fold；
8. reserve-only 仍在第 4 return 前离开 safe strip。

只有 reserve 把轨迹限制在 safe CCO strip、但最终只留下 tonic plateau 时，才解锁 local dynamic threshold `phi`。否则不加 `phi`，也不回头扫 E→E。

## 9. 允许与禁止的动力学表述

允许：localized real fold 进入、bounded CCO 内环、reserve-shaped q slow nullcline、event-locked additive exit、hybrid fold/SNIC-like slow-fast burst-loop hypothesis。

禁止：已证明 Hopf、torus、永久 ictal limit cycle、continuous spatial containment、full SNN seizure、或 reserve 已产生 complete lifecycle。
