# M3A-v2.2 carrier exploration under the sustained ramp+HOLD protocol — NEGATIVE (2026-06-29)

> **Status:** pilot-gate / necessary-condition SCREEN — **NOT** a seizure-mechanism validation.
> **Verdict: NEGATIVE / FAIL-CLOSED.** No clean partial-fill / returned-broken candidate.
> **Red lines (held):** tonic/multiburst is fail-closed (never ictal-like); a returned slow-off
> event = the protocol itself tamed it (not slow vars); NO "h_G proves recovery / axis-broken /
> closed-loop holds" claim is made. This is a descriptive screen.

## 一句话（朴素）

**测了什么** —— 上一版（M3A-v2 closed-loop）卡在"网络事件全或无、给不出局部可部分填充的事件"。
这一轮换了刺激方式（把背景驱动**缓慢加上去再按住不放**，给慢变量几百毫秒积分窗口），问两件事：
(1) 这个新协议能不能把"全或无"变成"局部、可部分填充、能自己回去"的事件；
(2) 推力（`q_I`）+ 刹车（`g_K`）载体在这种协议下，能不能造出一个"受控地拐到旁边、又能回到基线"的候选。

**怎么测** —— 3184 次真仿真、~5.7 小时、逐行存盘、每个 arm 重置噪声（配对可比）、引擎没碰。
slow-off 跑 3 衬底 × 30 seed × 8 个驱动强度（720 次）；`q_I+g_K` 在 primary 扫 80 组参数 × 12 seed
（1920 次）+ 在唯一出过干净事件的 backup·0.85 补扫 30 seed × 16 组参数（544 次）。判据是
**fail-closed 的**：多段 / 长 tonic 一律不当事件；候选必须是"干净单事件 + 返回 + 中等范围 +
轴向性下降 + 旁路/全局上升"。

**揭示了什么（在这个尺度上看起来）** —— **持续驱动没改掉"全或无"，载体也造不出候选。**
720 次 slow-off 里 718 次仍是"铺成一片 tonic 或压根没点着"；唯一的 2 次干净事件是 backup 最高驱动
下个别 seed 的**小沿轴 blip**（范围~6–8%、轴向性~0.97），不是拐到旁边；`q_I+g_K` 加到这些 blip 上
也没把它们变大变拐——范围还是~8%、轴向性还是~0.97。全程 **0 个 partial-fill 候选**。这跟独立的
clamp 复查（[[project_topic4_m3a_v2_1_qigk_clamp_verdict]]）**同向收敛**：是载体图景本身不够，不是够不着。

## 设置（可复现）

- **协议**：sustained `nu_signal_fn`（runner 级 ramp `r0→r_hold`、t0=50ms、t_ramp=200ms、之后 HOLD 到 T=500ms）。
  驱动**不撤**（return 必须内源）。引擎核心未碰（骑现有 `nu_signal_fn` 钩子）。
- **驱动脚本**：`scripts/run_m3a_v2_2_explore.py`（commit `fc65a61`）+ `scripts/run_m3a_v2_2_explore_followup.py`（`b87cd45`）。
  逐 run 存 `per_run.jsonl`（crash-safe、strict-JSON、全参数）；budget 8h soft / 10h hard。
- **结果目录**（gitignored）：
  - 主跑 `results/topic4_m3a_v2_2_explore/20260628_234520/`（2640 runs, 4.66h）。
  - 补跑 `results/topic4_m3a_v2_2_explore/20260629_042947_followup/`（544 runs, 0.99h）。
  - 图 `results/paper-ready-figure/fig_m3a_v2_2_dynamics/`（fail-closed visual diagnostic）。

## 结果（gate）

**Stage 1 — slow-off C1 / Exp-0（720 runs）**

| 衬底 | n | C1 失败模式保留 | C1 协议变温和 | 干净单事件 | 分类分布 |
|---|---|---|---|---|---|
| primary | 240 | 240 | 0 | 0 | 225 tonic + 15 no-event |
| sensitivity | 240 | 240 | 0 | 0 | 225 tonic + 15 no-event |
| backup | 240 | 238 | 2 | 2 | 223 tonic + 15 no-event + 2 expanded_axial |

- **C1：718/720（99.7%）"失败模式保留"**——持续驱动**没**把"全或无"变成可部分填充的事件。
- **Exp-0：全程 `UNCALIBRATED`**——没有任何 seed 同时具备"干净返回轴向锚点"和"失控锚点"，感应器标不了。
- 唯一 2 个干净事件：backup、r_hold=0.85、seed 22/30，`expanded_axial`，**R≈0.06–0.08（小）、S_axis≈0.97（高轴向）**、
  F_off≈0.19–0.30、returned——是间期尺度的**小沿轴 blip**，不是破轴/离轴。

**Stage 2 — `q_I+g_K` 载体（primary 1920 + backup 补 544 = 2464 runs，`use_hG=False`）**

- primary：1920 次 = 1520 tonic + 400 no-event；**0 干净单事件、0 partial-fill 候选**。
- backup·0.85（补跑闭口）：544 次 = 520 tonic + 24 `expanded_axial`；24 个"干净单事件"**全是同一批小 blip**
  （seed 22/30，R≈0.08，S_axis≈0.97 仍高），**0 partial-fill 候选**。最高 F_off=0.58 只出现在 **tonic（fail-closed）** 态、不是干净事件。
- **Stage 3（h_G smoke）：SKIPPED**——gate 未过（Exp-0 不可标定 + 无候选），按设计不跑闭环 h_G。

**L-sensitivity（L=10 vs L=16，排除"薄片太小导致动力学不一致"）**

主跑 L=10。补跑 **L=16**（411 runs, 1.79h，C1 ladder 全 3 衬底 + primary `q_I+g_K`；geometry 已入 artifact）
**给出一致负结论**：slow-off C1-A **100%**（240/240，比 L=10 的 99.7% 更彻底）、**0 干净事件**（L=10 backup 那 2 个在 L=16 也没出现）、Exp-0 仍 `UNCALIBRATED`、`q_I+g_K` **0 partial-fill 候选**（171 runs 全 tonic）。
→ **负结论 L-robust**，不是 sheet-size artifact。结果目录 `results/topic4_m3a_v2_2_explore/20260629_083937/`（run_config + per_run 含 `L/density/AR/g/l_EI/C_EI/nu`）。

## 机制判读（为什么 negative）

跟 [[project_topic4_m3a_v2_1_qigk_clamp_verdict]]（独立 clamp 复查，1560 摆法 0 受控旁路）**同向**：
- 载体只能"松 XOR 累"——`q_I` 去抑制把事件推大就铺成 tonic；`g_K` 疲劳只压不改道（与 M2 同构）。
- "拐到旁边招募" 与 "自己停下来" 在这套快衬底里**互斥**：能拐（off-axis 高）的只有失控 tonic，能返回的只有不拐的小沿轴 blip。
- 换持续协议给了积分窗口，但**没改掉这个互斥**——衬底的事件拓扑（recurrent E→E relay-to-completion）仍是瓶颈。

**口径**：这是"**当前载体 + 当前协议不闭合**"，**不是**"慢变量机制总体失败"，更**不是**任何发作/恢复主张。

## Go / No-Go（下一步）

- **NO-GO**：继续在这套 `q_I/g_K` 载体 + sustained 协议上调参 / 扫 h_G 闭环——载体图景已被两条独立线（clamp 复查 + 本次 sustained 扫）判为不足。
- **下一杠杆**（承接三岔 [[project_topic4_m3a_v2_spatial_field_plan]]）：**`D_EE`（E→E relay depression）或衬底/事件协议重做**——
  瓶颈在"接力网把波推到底"的衬底拓扑，不在恢复变量。h_G 载体（已实现、测试齐、字节奇偶守）保留备用，但**在拿到一个干净 partial-fill 候选之前不开闭环大扫**。

## Provenance / 测试

- 分支 `codex/topic4-m3a-v2-2`；HEAD 在 run dir 的 `git_head.txt`。
- 实现 + 探索测试：`pytest tests/test_m3a_v2_2_global_recovery.py` = 29 passed（含 @slow byte-parity / order-invariance / smoke）；v2.1 回归无 regression；引擎未碰。
- 全部 `per_run.jsonl` strict-JSON（`allow_nan=False`、非有限数→null）。
