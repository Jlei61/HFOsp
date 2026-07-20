# Topic 4 MZ slow-gated recurrent divisor：v2 discovery design

日期：2026-07-19

状态：已执行；得到 finite-window bounded-bursting opening，但 lifecycle no-go

分支：`codex/topic4-mz-divisive-lifecycle`
结果根：`results/topic4_sef_hfo/mz_divisive_lifecycle/`

## 1. 为什么这是新节点

v1 已经 clean no-go：current-based Z 加 40--200 ms activity-dependent recurrent-E divisor，没有产生
settled m-off ictal state；边界点只是 delayed runaway。并行 conductance line 的 I/M conductance 加线性 M 也只形成
runaway-to-prevention bracket。

历史机制去重如下：

- `h_G` 已测试持续高招募 sensor，但以减法电流接入，饱和 runaway 后无效；
- M4 `S_G` 已测试快速 recurrent-E divisor，能 bound 但不能 exit；
- M4-2 STD 已测试逐 presynaptic spike 的慢 recurrent availability，结果为 persist/fragment/suppress；
- M4-3A 已测试连续 activity load 到 leak-conductance shunt，结果为 runaway/fragment；
- conductance M 已测试逐细胞 spike 累积的 sAHP conductance，结果为 prevention 或低强度 burst train。

尚未原样测试的 conjunction 是：**普通 IED 完全不启动、只在持续招募 shoulder 启动的独立慢状态，且该状态直接
降低 recurrent E gain，而不是增加减法电流或 leak shunt。** 本轮只测试这个 conjunction，不重复上述机制。

并行 conductance 线的 excitatory term 仍为 current drive `I_E`；full `g_E(E_E-V)` 是该线的开放 fast-topology
lever，本分支不实现。

## 2. 锁定方程

保留 v1 的 `p=1` fast area pool：

\[
\Psi_i=\frac{[r_i-r_0]_+^n}{r_{50}^n+[r_i-r_0]_+^n},\qquad
A_G=\langle\Psi_i\rangle,
\]

\[
\tau_\mu\dot\mu_G=-\mu_G+A_G,\qquad
\tau_S\dot S_G=-S_G+S_{max}\mu_G.
\]

新增独立 slow gate：

\[
U_T(A_G)=
\frac{[A_G-A_0]_+^{n_T}}
{A_{50}^{n_T}+[A_G-A_0]_+^{n_T}},
\qquad
\tau_T\dot T_G=-T_G+U_T(A_G),
\]

并把 recurrent E coupling 改成：

\[
I^{rec,eff}_{E,i}=
\frac{I^{rec}_{E,i}}
{1+\alpha_fS_G+\alpha_TT_G}.
\]

其余 current-based Z 方程不变；M hard-off。`T_G` 是抽象的 slow recruited-state terminator，不能命名成 GABA、
sAHP、pump 或 seizure-termination physiology。

## 3. 参数锁

从已经落盘的同 seed trace 直接锁定 sensor：

- slow-off `p=1 AG_max=0.111708`；
- `p=1, alpha_f=2` delayed-runaway shoulder `AG_max=0.353514`（10 s）并在长跑达到 0.480378；
- 因此 `A0=0.15`：高于 ordinary IED ceiling，低于 recruited shoulder；
- `A50=0.10`, `n_T=4`, `T_max=1` 固定，不扫。

fast pool 固定：`p=1, alpha_f=2, tau_mu=30 ms, tau_S=80 ms`。Z 固定：
`I_th=1.6652801609959704, tau_z=10000 ms`。

## 4. 唯一允许的 5-cell screen

seed 1，spontaneous/no-kick，T=20 s：

1. `alpha_T=0` parity anchor；必须复刻约 14.64 s delayed runaway；
2. `alpha_T=4, tau_T=750 ms`；
3. `alpha_T=4, tau_T=2000 ms`；
4. `alpha_T=6, tau_T=750 ms`；
5. `alpha_T=6, tau_T=2000 ms`。

active cell 可保留 operational-runaway early stop；真正返回的轨迹必须完整跑到 20 s。只保存 metrics 与 downsampled
rate/AF/Z/SG/AG/TG/UT trace，不存 I spikes、LFP 或空间 movie。

## 5. 验收与 stop rule

工程 gate：

- `use_TG=False` 与 v1 byte parity；
- `use_TG=True, alpha_T=0` 与 v1 byte parity；
- `A_G<=A0` 时 `U_T=0`，`T_G` 只衰减；
- active `T_G` 只除 recurrent E，I cells 与 feed-forward E 不变；
- anchor 不复刻 delayed runaway则停止，不解释 active cells。

scientific candidate 必须同时满足：

1. Z 在无 kick/参数切换下自主进入；
2. 连续 recruited epoch 至少 1 s；
3. 不触发 operational runaway；
4. 无 reset 返回 same-seed slow-off rate band至少 2 s；
5. 20 s 内无 rebound runaway；
6. `T_G` 在 ordinary IED 段近零，先于 recruited-rate decay 上升。

若 4 个 active cells 全为 runaway、prevention、fragment train 或不恢复 plateau，则本节点 clean no-go；**不再调
`A0/A50/n_T/alpha_T/tau_T`**。只有出现候选才允许：

- seeds 3/4 复现；
- matched clamped-`T_G` control；
- frozen-`T_G` state forks，证明 `T_G=0` 时 trajectory 走向 high branch/runaway，而候选水平使该 high branch/route
  消失；
- 之后才做 early/late retrigger 与空间 readout。

## 6. 资源合同

沿用 v1 的 fork/COW、BLAS=1、launcher lock 与 96 GiB reserve。20 s worker 按 12 GiB 预算；5 cells 最多 5 workers，
预估上限 60 GiB。启动前必须检查 `MemAvailable`，不得与 conductance launcher 同时越过共同保留线；swap 增长则停止
新 wave。

## 7. Claim boundary

即便 5-cell 出候选，也只能称 `slow-gated recurrent-divisor lifecycle candidate`。未做 frozen state forks 前不能称
limit cycle、Hopf、bistability 或 ictal attractor；未做空间/LFP gate 前不能称 spontaneous seizure 或 tonic-clonic
event。

## 8. Execution outcome

锁定的 5-cell screen 已完成。`alpha_TG=0` 在 14639.7 ms 复刻 delayed runaway；四个 active cells
均把 runaway 改成有限窗 recruited bursting，但没有一格返回 slow-off。最佳 `alpha_TG=4,
tau_TG=750 ms` 在 20 s 内有约 5.05 Hz 节律，然而最后 3 s 的 `z_mean=-0.0218/s`、
`TG=+0.0378/s`，不能称 settled branch、attractor 或 limit cycle。严格重判：
`results/topic4_sef_hfo/mz_divisive_lifecycle/runs/20260719T162035.230785Z_6ce230e_e1acc35592_slow_gate/strict_audit.json`。
