# Topic 4 MZ slow-gated divisor + M exit：v3 bounded-state test

日期：2026-07-19

状态：已执行；locked eta ladder scoped no-go
分支：`codex/topic4-mz-divisive-lifecycle`

## 1. Gate provenance

v2 的 high-state-gated slow recurrent divisor 没有自己完成 lifecycle，但在预注册 5-cell screen 中产生了一个
允许继续做 discovery 的 finite-window m-off high-state candidate：

- cell：`alpha_fast=2, alpha_TG=4, tau_TG=750 ms`；
- seed 1，20 s，无 runaway；
- recruited 6094.9 ms，33 peaks，5.09 Hz，modulation 0.568；
- final rolling-1 s rate 67.56 Hz，last-3 s slope -0.534 Hz/s；
- 未返回 baseline。

证据：
`results/topic4_sef_hfo/mz_divisive_lifecycle/runs/20260719T162035.230785Z_6ce230e_e1acc35592_slow_gate/summary.json`。

这第一次近似满足 v1 的前置门：m-off 在 20 s 内不再 runaway，并有持续 recruited bursting；但慢变量仍漂移，
所以 v3 只能先作为 M-exit discovery test，不能预称 settled bounded branch。
本轮不是调 v2 sensor，也不是把 v2 的 lifecycle no-go 改判为 positive；它是一个独立的 mechanism-composition
节点。

## 2. 方程

Z、fast `S_G` 和 slow `T_G` 完全冻结为 v2 bounded cell。只打开原始 current-based per-neuron M：

\[
\dot m_i=-m_i/\tau_m+\sum_k\delta(t-t_i^k),
\]

\[
I^{net}_{E,i}=I^{ff}_{E,i}+
\frac{I^{rec}_{E,i}}{1+\alpha_fS_G+\alpha_TT_G}
-z_iI^I_i-\eta_m m_i.
\]

没有 conductance、`phi`、STD、qI/gK/hG、新 kernel 或参数 schedule。

## 3. 唯一允许的 6-cell screen

seed 1，spontaneous/no-kick，T=25 s，`tau_m=2000 ms`。

复用 v1 在 simulation 前已经注册的原 ladder，不新增中点：

`eta_m = [0, 0.00186, 0.00373, 0.00745, 0.01118, 0.01863]`。

- `eta_m=0` 是 25 s m-off longevity anchor；
- 所有 M-on cells 必须完整跑 25 s，不能用 high-rate early stop 截掉潜在 return；
- 只保存 metrics 与 downsampled traces。

## 4. Interpretation gate

首先要求 25 s m-off anchor 仍为 bounded high state且 endpoint settled；若它在 20--25 s delayed runaway，M-on
结果只能称 containment sensitivity，不能称 termination。

genuine M-exit candidate 必须同时满足：

1. 自主进入，recruited epoch >=1 s；
2. 无 operational runaway；
3. M-on 有明确 offset，随后 >=2 s 回到 same-seed slow-off rate band；
4. m-off 同期不返回；
5. 25 s 内无 rebound；
6. M build-up 先于 recruited-rate decay，且 `eta_m=0` 不出现同样的 decay。

出现候选后，取消任何剩余 discovery 调参，只做 seeds 3/4、M-off、matched-clamped-M/state-fork 与 early/late
retrigger。没有候选则该 exact composition clean no-go；**不扫 tau_m、不细化 eta_m、不改 T_G 阈值或强度。**

## 5. Resource contract

6 workers，25 s worker 按 15 GiB、1.2 safety factor预算；保留至少 96 GiB available memory。fork/COW、BLAS=1、
launcher lock、atomic outputs 继续沿用。启动时 conductance launcher 必须空闲；swap 增长则不启动新 wave。

## 6. Claim boundary

即使出现 return，也只能称 `M-dependent autonomous lifecycle candidate`；cross-seed、matched controls、state forks
和空间/LFP gate 未完成前，不称 seizure、ictal attractor、limit cycle、Hopf 或 tonic-clonic transition。

## 7. Execution outcome

6 个 25 s cells 已完整执行。m-off 继续自主进入并维持 recruited bursting 到末端，但 rate 末段仍
`+1.63 Hz/s`，`z_mean=-0.0175/s`、`TG=+0.0506/s`，因此只是更长的 finite-window high-state
candidate。五个非零 M cells 均未形成至少 1 s 的 recruited macro-state：最长 shoulder 依次为
500/220/190/120/0 ms。它们改变的是 entry/containment，而不是已进入高态后的 exit。

严格 post-hoc verdict 为 `no_strict_lifecycle_candidate_in_locked_eta_ladder`；按本 spec 不运行 seeds 3/4，
不细化 `eta_m`，也不扫描 `tau_m/T_G`。证据：
`results/topic4_sef_hfo/mz_divisive_lifecycle/runs/20260719T172358.336529Z_6ce230e_80a127d772_slow_gate_m/strict_audit.json`。
