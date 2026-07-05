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

### 附：h_G 全局恢复 runaway-transition 单轨迹 GIF（visual diagnostic，开环，非闭环大扫）

为了直观回答"把全局抑制恢复标量 `h_G` 打开后，间期 axis-like→runaway 的转变长什么样"，做了一个
**单轨迹** GIF（`results/paper-ready-figure/fig_m3a_v2_2_hG_runaway_transition/`），和 v2.1 的
`fig_m3a_v2_1_qigk_runaway_transition` **同一条轨迹**（同 substrate/seed/多脉冲驱动/`q_I` 载体），唯一区别是
`use_hG=True`。M50/B50/Pi50 由 `--probe`（eta_G=0 不反馈）测得的局部事件天花板 vs runaway 地板取**几何中点**
（M:0.031/0.373→0.11，B:0.508/0.592→0.55，Pi:0.300/0.997→0.55），使 `χ_G≈2e-4` 穿过局部事件、`≈0.52` 进入 runaway。

两件事：
- **全局性传感器工作正常**：`h_G` 在 130/400/670 ms 的局部沿轴事件期间基本不升（`χ_G≈0`），只在 runaway
  （onset 771 ms）后陡升到 0.94——它没误伤局部事件。
- **但减法式全局刹车拉不回 runaway**：`eta_G` 开环阶梯（0,2,4,6,8,12,20,40,80）**结构性无效**——runaway onset
  恒为 771 ms、末段 ~471 Hz、`h_G` 恒到 0.94，即便把 `eta_G·h_G` 加到 ~75 mV（reset 11→threshold 18 = 7 mV 跨度
  的 >10 倍）也纹丝不动。`I^net_i -= η_G h_G`（仅-E）是个减法常量，减不动一个已经饱和的 recurrent-excitation 雪崩。
  **印证主结论**：瓶颈在 recurrent E→E relay-to-completion 衬底，不在恢复变量。

这是**开环、单轨迹的可视化诊断**（把 §B6 减法耦合作用在一条已知 runaway 上看效果），**不是**被 gate 拦下的
**闭环** `h_G` partial-fill 大扫（后者仍 SKIPPED）；也**不**主张 recovery / 闭环成立。阶梯原始 9 行存
`fig.../figures/eta_G_ladder.json`。

另有一个 **q_I 载体 + 轴向 g_K 疲劳版（E1146 真实电极几何）** `results/paper-ready-figure/fig_m3a_v2_2_qI_runaway_transition_epilepsiae_1146/`：
同轨迹但顶部画**抑制资源 `q_I(t)`（mean+min）+ 轴向区域 `g_K` 疲劳**（legend 右上角）、`h_G` 关、去掉底注、左图 E→E 梯度画成**椭圆**。
直观看 `q_I`（min=轴向走廊先耗竭）掉到地板 + `g_K` 累积→局部沿轴小事件铺成全场 runaway（757 ms）。同一脚本
`--layout subject1146 --top qI --no-footer`。**机制岔路（重要）**：本图 `g_K` 膜耦合**关**（`eta_K=0`，只建起来可视化、不改轨迹）；
若 `g_K` **真耦合**到 nominal（`eta_K=1`、`gK_max=1`），它在**小事件期**就把核压住、**直接阻止 runaway**（实测 max~24 Hz、`q_I` 几乎不耗竭、无 runaway）——
这是 `g_K`=limit 成功限流（与 M2 "只压" 同向），是另一张图。注意这跟 eta_G 全局减法刹车不同：g_K 在**点火前**局部建起来才有效，**点火后**减法刹车仍拉不回。

## Go / No-Go（下一步）

- **NO-GO**：继续在这套 `q_I/g_K` 载体 + sustained 协议上调参 / 扫 h_G 闭环——载体图景已被两条独立线（clamp 复查 + 本次 sustained 扫）判为不足。
- **下一杠杆**（承接三岔 [[project_topic4_m3a_v2_spatial_field_plan]]）：**`D_EE`（E→E relay depression）或衬底/事件协议重做**——
  瓶颈在"接力网把波推到底"的衬底拓扑，不在恢复变量。h_G 载体（已实现、测试齐、字节奇偶守）保留备用，但**在拿到一个干净 partial-fill 候选之前不开闭环大扫**。

## Provenance / 测试

- 分支 `codex/topic4-m3a-v2-2`；HEAD 在 run dir 的 `git_head.txt`。
- 实现 + 探索测试：`pytest tests/test_m3a_v2_2_global_recovery.py` = 29 passed（含 @slow byte-parity / order-invariance / smoke）；v2.1 回归无 regression；引擎未碰。
- 全部 `per_run.jsonl` strict-JSON（`allow_nan=False`、非有限数→null）。
