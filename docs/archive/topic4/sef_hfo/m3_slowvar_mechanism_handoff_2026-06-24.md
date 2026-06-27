# HANDOFF — M3 slow-variable MECHANISM thread (Stage-3 follow-up, 2026-06-24)

> **给接手这条线的人/上下文**：本线的目标**不是** W、**不是** kick basin。目标是：找到一个**慢变量+新机制**，
> 在它变化时网络出现**≥2 个状态**（双稳 / 转变 — 不预先设计，看仿真定），能**区分"间期传播状态"和"发作"**
> （有 resting / 间期→发作前 pre-ictal 更好，但不强求）。**这是 M3 拆分后的第 1 个核心目标。** 第 2 个目标
> （W 有效表示 field + 连接数据）在另一条线，见 [[m3_b1_validation_recap_2026-06-24]] / 本仓 W 工作。

## 0. 为什么需要这条线（动机，别重蹈覆辙）

Stage 3 已大量仿真证明：**在当前机制设置（固定阈值 + 外部 drive 的 Brunel 波引擎）下，只能产生传播，
产生不了"空间自限 + 间期/发作双稳"的区别。** 这正是进入 M3 的动机。

**本次（2026-06-24）又踩了一次坑、确认了这一点**：static-μ pilot 把慢变量做成**最简单的阈值 permissivity**
（`V_th_eff = vth_core − ΔVth(μ)·h`，就是改 vth）。结果（results/topic4_sef_hfo/m3_static_mu/）：
- μ↑ 只让**自发事件频率**升（0.04→4.2→5.4 Hz），**事件大小/时长不变**（~12 bins、~60ms 平的）；
- 事件**全是 R3（大-回静）**，**无 R2→R3 梯度、无 R4a 持续招募、无双稳**；basin 退化（什么都回静、什么都不 escape）；
- core-h **不优于** uniform/shuffled → μ ≈ 全局加热，没特异门控。

**结论：单纯改 vth / drive（已测过的简单旋钮）不行，必须引入新机制。** 不要再扫 vth-μ 网格。

## 1. 好消息：新机制的基础设施**已经在引擎里**（off by default，未标定）

`src/snn_engine/slow_vars.py` —— "Page-4 epilepsy slow-variable layer (Zou & Lei deck 2026-06-01)"，
经 `src/snn_engine/model.py` 的 `simulate(..., slow=...)` 的 `SLOWVAR HOOK` 挂载，**默认关（slow=None ⇒ bit-parity）**，
**参数是 PLACEHOLDER（deck 没给表，需标定后才能下结论）**。三个慢变量（deck 方程已写）：

| 慢变量 | 机制 | 时标 | deck 方程要点 | 文献锚 |
|---|---|---|---|---|
| **disinhibition z** | Cl⁻ / STP 致抑制减弱 | ~5 s | `I_net = I_E − z·I_I`，`z→0` 当 I_I 超阈 | **Cl⁻/去极化 GABA → pre-ictal**（用户点名的"Cl 致异质性反转"；Huberfeld 等 pre-ictal discharge vs IID）|
| **adaptive threshold φ** | 阈值随放电自适应 | ~100 ms | `V_th_eff = slow.threshold(V_th)` per-neuron | 自适应阈值 / spike-frequency adaptation |
| **sAHP g_K** | K⁺ 介导外向流 | ~5 s | `I_net −= g_K`，`dg_K/dt=−g_K/τ+g_Kmax·S` | sAHP / 发作终止 + bursting |

另有 **GABA 反转**直接在 `kick_probe.py::membrane_step`：`shunt_gaba` + `e_gaba`（反转电位）+ `g_gaba_scale`
（conductance-based shunting：`V_inf=(I_E+g_I·e_gaba)/(1+g_I)`）。**慢慢把 `e_gaba` 从超极化推向去极化 = Cl⁻ 累积/
去极化 GABA 机制** —— 这是 pre-ictal discharge 的经典机制、且能**区分 pre-ictal discharge 与 IID**（用户要点）。

## 2. 引擎路径（**2026-06-24 audit 更正：simulate_kick 也接 slow**）

**早先本文误写"simulate_kick 没有 z/φ/g_K 钩子"——错。实测更正：**
- `kick_probe.py::simulate_kick(p, net, KICK_BOOST, slow=None, ...)` **本身就接 `slow=`**，并在循环里调
  `slow.apply_currents / slow.threshold / slow.step`（行 230-251）。`model.py::simulate()` 也接。**两条路都有。**
- `SlowVars`（slow_vars.py）的 `apply_currents/threshold/step` **已实现**（z/φ/g_K 方程都在），**只有参数是 placeholder**。
- **实测 gate 全过**（L8 smoke）：`slow=None` 与不传 slow **逐字节一致**（bit-parity）；`slow=SlowVars(z=0.3)` 改变放电
  （125→404988 spike，说明 z 起作用但 placeholder 没标定=直接 runaway，印证"标定前不下结论"）；静态 `e_gaba` 扫
  （`shunt_gaba=True, e_gaba=...`）改变活动、**不需要改 engine**。
- **结论**：M3A 可**直接复用现有 `simulate_kick` 路**（含我已建的 `run_m3_static_mu_spontaneous.py`，它就是 `simulate_kick(KICK_BOOST=0)`），
  只需传 `slow=` / `e_gaba`/`shunt_gaba`。**唯一缺口**：动态 `e_GABA`（Cl⁻ 累积，首选机制）**不在 SlowVars 里**
  （SlowVars 只有 z/φ/g_K；e_gaba 是 membrane_step 的静态参数）→ **dynamic e_GABA 需给 SlowVars 加一个 e_GABA 状态量
  + 让 membrane_step 收 per-neuron 时变 e_gaba**（off-by-default、保 slow=None bit-parity）。quasi-static e_GABA 不需要，直接扫参数。

## 3. 这条线该做什么（建议，PILOT-FIRST）

1. **标定 + 单变量扫**：逐个打开 z / φ / g_K（和 e_gaba 去极化扫），其余关，先标定 PLACEHOLDER 量级
   （时标已知 100ms/5s；幅度要扫）。**问：哪个慢变量能产生"间期传播 vs 发作"两个可区分状态（双稳/迟滞/转变）？**
2. **读出复用本仓已建的 R0–R4 分类器**（`src/sef_hfo_mu_basin.classify_event`，含 R4a-W对齐持续 vs R4b-全场tonic；
   **只有 R4a 算发作样**）+ 自发事件检测器（`detect_events/event_props/aggregate_spontaneous`）。这套不用重写。
3. **优先序（文献+用户指向）**：**disinhibition z / e_gaba 去极化（Cl⁻/pre-ictal）= 首选**（最贴"Cl 反转 + 区分
   pre-ictal/IID"）；其次 adaptive threshold φ（可能给自限↔持续的双稳）；再 sAHP g_K。
4. **判据（基本要求）**：慢变量某个区间出现**≥2 个可区分状态**，至少能分"间期传播"和"发作"。resting / pre-ictal
   作 bonus，不强求。**先看仿真长什么样，不预设双稳形态。**
5. **纪律**：slow=None ⇒ bit-parity（回归硬条件）；PLACEHOLDER 标定前不下结论；不改 detector 阈值追结果；
   R4b tonic ≠ 发作；先小 pilot（单网络 / L20 / 慢变量 3 档 / 长 T），跑通看趋势再扩。

## 4. 与 W 那条线的关系（为什么拆开但终点相连）

M3 的核心命题是 **"有效 W 随慢变量发生相变"**：慢变量（本线）提供机制 → 状态转变；W（另一线）是 field 的
**抽象表示 + 数据桥**（SEEG 间期模板 rank 投 2D ≈ 发作 rank 序列 ↔ 模型 E→E 兴奋梯度轴）。所以两线**并行**、
**终点在"W_eff 在状态转变时怎么变"汇合**。本线只需做到"有机制能产生≥2 状态"；W 长什么样、pre-ictal W 的判据，
交给 W 线。

## 5. 现状交接清单

- 已建可复用：`src/sef_hfo_mu_basin.py`（apply_mu / R0–R4 / R_event / 自发检测）、`run_m3_static_mu_spontaneous.py`
  （自发 no-kick runner = `simulate_kick(KICK_BOOST=0)`；simulate_kick **已接 slow=**，runner 只需加 `--slow-*` 透传即可用 z/φ/g_K；
  e_gaba/shunt_gaba 也只是透传参数）、`analyze_m3_static_mu_pilot.py`。
- 待做：给 runner 加 `--slow-mode/--e-gaba/--shunt-gaba` 透传；quasi-static 扫 e_gaba(参数,无需改engine) + z/φ/g_K(SlowVars 冻结值)；
  **标定 placeholder**（z=0.3 已 runaway）；找双稳。dynamic e_GABA 才需给 SlowVars 加 e_GABA 状态量。
- 别做：dynamic m 已被禁的是"无机制的动态 m"；这里的"慢变量动态"是**有机制的**（z/φ/g_K/e_gaba），是允许且正是目标。
  （澄清：之前"禁 dynamic m"是禁"凭空动态加热"；deck 的 z/φ/g_K 是有生理机制的慢变量，属本线正题。）

关联：[[project_topic4_sef_hfo_snn_stage3_plan]]、[[project_topic4_sef_hfo_pathology_mapping_spec]]
（Rich/Huberfeld/Lepeu 锚）、[[m3_static_mu_pilot_2026-06-24]]（本次 vth-μ 坑的完整记录）。
