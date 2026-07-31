# FCXR-HYB2 实施计划 —— 事件尺度、有界幅度的自主招募层（ELR）

日期：2026-07-31（**rev2，二轮审阅后**） · 分支 `codex/topic4-fcxr-hyb2`（**待建**）
**基点 `50f90c59`**（= 含 HYB2 spec + 本 plan 的那个 commit；**不是 `9c5c5b16`**，从那里开分支会
把 spec 与 plan 一起丢掉）

Spec：`docs/superpowers/specs/2026-07-31-topic4-fcxr-hyb2-event-limited-recruitment-design.md`

> **spec 与 plan 冲突 = 执行 blocker。** 不设"谁优先"；发现冲突必须**先同步修订两份文件**再执行。

**本文件在任何 HYB2 结果出现之前锁死。之后只允许追加"执行结果"，不允许改判据、阈值或档位。**

前代不可改写：`docs/archive/topic4/sef_hfo/fcxr_hyb1_baseline_disturbed_2026-07-31.md`
（`STOP_BASELINE_DISTURBED`）。HYB1 的 spec/plan 已标记为已执行并收口，**不得回填**。

---

## 0. 一句话

HYB1 否掉的是"**用 0.65 秒的浓度记忆承载招募**"——它和间期事件的**空白间隔**同量级，所以钾在事件
之间清不干净、逐级棘轮。HYB2 换一个**记忆只有事件长度（约 27 毫秒）、幅度有上限、活动停了自己就
衰减**的执行器，先问两件事：**(B0) 它在平时看不看得见？(A0) 它还留不留得住 B2.1 那个招募效应？**
两问之一失败就干净停机，**不烧 12 格**。

---

## 1. 授权边界（先说清楚，避免与"不得进 40k"自相矛盾）

本 plan 授权 **恰好一条** 40k 运行：**§5.1 的 24 s sensor-only calibration run**。
它的产物写进独立的 `calibration_lock.json`。**在那份 lock 落盘并回填附录 A 之前，
其余任何 40k 运行（B0 / A0 / 12 格 / 生命周期）一律不得启动。**

---

## 2. 锁定项 1 —— ELR 的完整离散方程（**无隐形自由参数**）

审阅 P0-1 指出 spec §3.2 只给了 `R_ε` 的函数**类**。以下把每一项写死；任何一项改动都会改变
`q_v`、`Q_on` 与 A0 的激活时刻。

| 项 | 锁定值 / 规则 |
|---|---|
| 网格 | 32×32 体素、`dx = 0.625 mm`、`L = 20 mm`（沿用 B2.1 / HYB1） |
| 聚合步长 | **`dt_R = 0.5 ms`**；`s_v(t)` = 该块内体素 E+I spike 数 ÷ 体素细胞数 ÷ `dt_R`（单位 Hz） |
| 空体素 | `n_v = 0` 是**采样空缺**：不产源、不进任何统计（沿用 B2.1 空体素契约） |
| `b_v` | **`b_v := Q99_t[s_v]`**，只用 **seed1 calibration 前 12 s**，仅占据体素 |
| deadband | **`R_ε(u) = u²/(u+ε)`（`u > 0`），否则严格 0**（HYB1 已实现并测过：C¹、背景以下严格零） |
| `ε_s` | **`:= 0.1 · median_v(b_v)`**（沿用 HYB1 规则） |
| `ε_q` | **`:= 0.1 · Q_on`**（同一 10% 约定，零自由度） |
| 积分 | **精确指数更新** `q ← q·e^{−dt_R/τ_R} + e·(1 − e^{−dt_R/τ_R})`（对分块常值 `e` **精确**，无条件稳定） |
| 两遍离线 | **pass 1** 由记录的 load 算 `b_v`；**pass 2** 重放同一条记录算 `q_v` → `Q_on`。**全离线，无新仿真** |
| 硬门 | `Q_on > 0` **且** `Q_scale > 0`，否则判 **`CALIBRATION_INVALID`** 并停止 |

$$e_v(t)=R_{\varepsilon_s}\big(s_v(t)-b_v\big),\qquad
\tau_R\dot q_v=-q_v+e_v,\qquad q_v(0)=0$$

$$u_v=R_{\varepsilon_q}(q_v-Q_{\mathrm{on}}),\qquad
R_{\mathrm{evt},v}=I_{R,\max}\tanh\!\big(u_v/Q_{\mathrm{scale}}\big)$$

膜执行器：`E` 与 `I` 收到**同一个附加 current**（不是 conductance——引擎会对 I 细胞丢弃 conductance）；
virtual-SEEG 仍为 **E-only** 读出；六个 blessed engine 文件**不得修改**。

---

## 3. 锁定项 2 —— 事件时间尺度与 `τ_R`（**用 GAP，不用 IEI**）

### 3.1 唯一来源（冻结 event bar）

**只用** `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-mz-fcxr-lc1/results/topic4_sef_hfo/`
`mz_full_conductance_spatial_relay/lifecycle_closure/baseline_contract_seed{1,3}.json`
（LC1 的 24 s slow-off 基线，bar 已冻结）。

> 上一版把根路径写成 `/.worktrees/...`，**那不是一个能解析的绝对路径**。已更正。
> 执行前必须做 **artifact preflight**：逐个 `os.path.exists` + 落 **sha256** 进 manifest，缺失即停。

**明令禁止**用 HYB1 自己那条 8 s 基线：它在 seed3 上给 duration 中位 39.0 ms，同配置的 LC1 24 s
给 11.0 ms——差 3.5 倍，纯粹是短窗**重新推导** event bar 的假象。

### 3.2 已核数值（零仿真）

| 量 | seed1 | seed3 | 锁定 |
|---|---:|---:|---:|
| n events (24 s) | 34 | 67 | — |
| duration 中位 / q90 / max (ms) | 10.0 / 19.7 / 22.0 | 11.0 / 14.0 / 22.0 | — |
| **`T_event,guard`（两 seed 事件时长最大值）** | | | **22.0 ms** |

> **命名订正**：22.0 是**最大值**，不是 q90（per-seed q90 = 19.7 / 14.0，pooled q90 = 15.0）。
> 取最大值而非 q90 的理由：`τ_R` 必须装得下**一个完整的最长事件**，用 q90 会截断 10% 的事件。

### 3.3 分母必须是 event gap

$$GAP_k := t_{\mathrm{on},k+1}-t_{\mathrm{off},k}$$

事件**结束之前** `q_v` 一直在被驱动，能用于衰减的只有 `GAP`。用 onset-to-onset 的 `IEI` 多算了约
一个事件时长，**系统性低估残留**。

**保守界（有效）**：`GAP_k = IEI_k − dur_k` 逐点成立且 `dur_k ≤ T_max` ⟹ `GAP_k ≥ IEI_k − T_max`
逐点成立 ⟹ **分位数在该平移下保持** ⟹ `GAP_05 ≥ IEI_05 − T_max`。

代入预审的 `IEI_05 ≈ 169.5 ms`：`GAP_05 ≥ 147.5 ms` ⟹ 上界 **36.81 → 32.03 ms**。

### 3.4 `τ_R` 的唯一选择规则（先写死，不扫）

$$\tau_R := \sqrt{T_{\mathrm{event,guard}}\cdot \frac{GAP_{05}}{\ln 100}}$$

- 可行区间 `[22.0, 32.03]` **非空**，几何中点 **`τ_R ≈ 26.55 ms`**（**预期值，不是锁值**；
  锁值由 §5.1 的 canonical `GAP_05` 重算后写回附录 A）；
- **余量只剩 10 ms**：`T_event,guard` 只要涨到 32.0 ms 区间就闭合。这是一个真实的窄区间，
  必须在 manifest 里显式记录余量；
- 区间为空 → **`DESIGN_BLOCKED_EVENT_TIMESCALE`**，停止，**不许**调 baseline、不许跳到 40k。

### 3.5 必须一并写入 manifest 的尾部残留（**不许藏**）

`exp(−GAP/τ_R)` 在四处：**`GAP_05`、`GAP_01`、实测最短 `GAP`**，以及 `GAP = 40 ms`。
预审在 `τ_R = 26.55 ms` 下：147.5 → 0.0039、100 → 0.0231、60 → 0.1043、40 → 0.2216。
**`GAP_05` 规则只覆盖 95% 的空白；最短那 5% 的残留必须可见。**

---

## 4. 锁定项 3 —— `Q_on`、`Q_scale`、`I_R,max`

### 4.1 calibration / validation 划分与 seed3 合同

一条 **T = 24 s** 的 sensor-only calibration run（§5.1），**seed1 前 12 s = calibration、
后 12 s = validation**。

**seed3 的合同（这一条我不照办审阅的建议，理由如下）**：

审阅建议"seed1 锁全部参数，seed3 直接沿用 seed1 的 `b_v/Q_on/Q_scale`"。
问题在于 **`b_v` 是逐体素的背景负荷，而 seed3 是不同的连接底座**——把 seed1 的 `b_v` 套到 seed3 上，
deadband 在空间上是**错配**的。那样 seed3 一旦失败，**无法区分"设计不成立"与"阈值贴错了地方"**，
正是本项目反复在抓的那类含混失败。

**锁定：规则在 seed1 上写死并冻结，seed3 用同一条冻结规则在自己的 calibration 半段上算出自己的
`b_v/Q_on/Q_scale`。** 没有任何参数是看着 seed3 的结果选的 ⟹ **规则是 out-of-sample 的**，
这才是要验证的东西。审阅担心的"seed3 变成第二份 in-sample calibration"由此排除。

**同时**把审阅的方案作为**免费的二级诊断**报告（同一条已记录的 load 上后处理，零额外仿真）：
把 seed1 的 `b_v/Q_on` 直接套到 seed3，看是否也通过。**它只诊断空间错配，不作为门。**

### 4.2 `Q_on`

1. 在 calibration 半段上，对每个**占据**体素记录 `q_v(t)` 的**逐事件峰值**
   （事件由同一冻结 bar 的 canonical onsets 划窗，窗 = `[t_on, t_on + 3τ_R]`）；
2. **`Q_on := 1.10 × max_{event, occupied voxel} q_v^{peak}`**；
3. 写入 `calibration_lock.json` 并冻结。**不得因 B0/A0/lifecycle 结果下调。**

### 4.3 `Q_scale` —— 解析规则（**spec 原文那条不可实现**）

按 §4.2，calibration 上 `q_v − Q_on ≤ 0` 恒成立，**没有任何样本落在 `Q_on` 之上**，
所以"从 calibration 解析 `q_v−Q_on` 的尺度"是空集。**锁定 `Q_scale := Q_on`**
（超出间期上界"一倍"时执行器达 `tanh(1) = 0.7616 × I_R,max`；零自由度）。

**不采用**"从 Gate A0 的 supra-interictal artifact 取尺度"——那条 artifact 同时是 A0 的被测输入，
用它定标会让 A0 的 ≥10% 判据部分自证。`Q_scale` 与 `I_R,max` **都不扫**。

### 4.4 `I_R,max` 单位链（**已核，逐位复现**）

`I_R,max := g·[E_K(K_{o,0}+δK*) − E_K(K_{o,0})]`，`g=1`、`δK* = 0.6715 mM`
（B2.1 匹配对照**闭环臂**窗 2 热点峰值，`b2_1_matched_control.json::arms.closed.peak2_mM`），
用已验收的 `src/topic4_fcxr_ion.E_K`（`RTF=26.64`、`K_i=140`、`K_{o,0}=4`）算得

**`I_R,max = 4.134151260609386` engine-drive units**（与 spec 逐位一致，差 < 1e-12）。

**force anchor，不是浓度解释。** `R_evt` 无 mM 单位；**禁止**称其为 extracellular potassium /
ion homeostasis / 患者离子机制。plan 只核 sha 与单位链，**不重新拟合**。

---

## 5. 锁定项 4 —— 唯一的 calibration run；A0 的 matched input

### 5.1 一条 run 同时落盘五样东西（**本 plan 授权的唯一 40k 运行**）

`seed = 1`、`T = 24 s`、**RC1 accepted 工作点、慢变量全 off、ELR off、无 kick**：

| 落盘 | 用途 |
|---|---|
| canonical event onsets **与 offsets**（同一冻结 bar） | §3.3 `GAP_05/01/min`、§4.2 划事件窗 |
| 逐体素 `s_v(t)`（`dt_R = 0.5 ms`） | `b_v`、pass2 离线重放 `q_v` → `Q_on` |
| 逐细胞 `I_I(t)`，100 ms × 240 帧（float32，约 30 MB） | §6 `S_Z` replay |
| 逐细胞 `p_i`（沿用 `zA_q75_tz5000` snapshot） | `D_Z` 权重 |
| 完整数值安全与事件统计 | **直接复用为 Gate B0 的 seed1 ELR-off 臂** |

> **它同时就是 B0 的 seed1 off 臂**——同一条确定性 24 s 轨迹，**不重复跑**（省约 25 min）。
> **§4 的 selection rule 必须在它跑之前写死**，不得据其结果修改规则。
> 它**不是** actuator run 也不是 lifecycle run。

seed3 的 24 s sensor-only run 同理，一条兼作 seed3 的 calibration 与 B0 off 臂。

### 5.2 Gate A0 的 matched input —— 唯一 provenance 与三分判决

- **输入 = LC1 的 `q50` Z-only 配置**（`I_th_EI = 1.6652801609959704`、`tau_z = 10000 ms`），
  `seed = 1`，**无 kick**，**`T = 9 s`**（不是 6 s）；
- 选它的理由（先写死）：`q50` 是**已验收**能把网络推到 supra-interictal 持续高负荷的档
  （LC1 E2：约 3 s 进 dense、约 6 s 进 ictal 窗、`D_Z→0.805`、末段 452.8 Hz）；
- **`T = 9 s` 的理由**：6 s 可能还攒不够"激活后 1 s"的窗口。9 s 给到 `t_gate` 最晚 8 s 仍有完整 1 s；
- **不再断言 q50"一定"越过 `Q_on`**——那个阈值此刻还不存在。改由下面的三分判决处理。

**`t_gate` 由两臂共同追踪的 counterfactual sensor 定义**：两臂都**完整跑 `q_v`**（同一代码路径），
**off 臂只把膜电流置零，不停掉 sensor 状态**。`t_gate := 首次 max_v q_v > Q_on`，两臂应**逐位相同**
——这既是 `t_gate` 的定义，也是结构性匹配的 sanity（沿用 B2.1 §2.3 的形态）。

**三分判决（先写死）**：

| 标签 | 条件 |
|---|---|
| `A0_INPUT_INSUFFICIENT` | `max_v q_v` 全程未越 `Q_on`，或 `t_gate` 后剩余 < 1000 ms |
| `A0_CEILING_CONFOUNDED` | off 臂在该 1 s 窗内 participants ≥ 90% E 细胞、或 occupied voxels ≥ 90% 占据体素、或触发 300 Hz early-stop |
| `A0_ELIGIBLE` | 以上都不成立 |

**只有 `A0_ELIGIBLE` 之后仍达不到 ≥10%，才允许判 `NO_GO_EVENT_LIMITED_ACTUATOR`。**

### 5.3 A0 的 ≥10% 判据实现

窗口 = `t_gate` 之后的 **1000 ms**。三个量沿用 B2.1 已实现的定义：
`window_participants`、`recruitment_radius_mm`、`participant_voxels`。

**通过 = 三者中至少两个相对 off 臂 ≥ +10%**，且 `max_v R_evt ≤ I_R,max`、零 clip、finite。
**禁止**为过门扫 `I_R,max`、恢复扩散项或加长 `τ_R`。

---

## 6. 锁定项 5 —— `S_Z` 开环累积耗竭轴与三个 `I_th_EI`

### 6.1 这条轴到底是什么（订正后的正确说法）

在**冻结的 slow-off 负荷轨迹**上离线 replay，`I_I(t)` 不受 `z` 影响，
**所以它结构上不可能表现"耗竭自限"**（自限是 `z↓ → I_I↓ → 落回阈下 → 恢复` 的闭环通路）。

更强的算术：若细胞在窗内阈上/阈下状态不变，所有阈上细胞走同一条 `z(t)=exp(−t/τ)`，于是
`S_Z(I_th) = a_p(I_th)·C(T_cal, τ)` —— **与 `h_Z` 严格成正比，不含新信息**。

**`S_Z` 唯一多出来的信息 = 窗内跨越阈值的细胞**，即它量的是**阈上时间占比**而不是某一瞬间的比例。
这比 `h_Z` 稳健，也把 1.64 倍窄区间上的三点分得更开，
**但它不是"整段耗竭响应"，也不是自限的度量。它只是一个参数坐标，用来把三个 Z 点均匀摆开；
它不预测这三点会不会产生不同的闭环分支——那个答案只来自 12 格短屏。**

### 6.2 两条因此必须写死的要求

- **`T_cal := 24 s`**（≥ `3·τ_Z_down = 15 s`）。实测 `C(T)`：`T_cal` = 3 / 5 / 15 / 24 s →
  0.248 / 0.368 / 0.683 / 0.793。**低于 15 s 时 `C` 仍在近线性段，换轴等于没换**（代码需告警）；
- replay 需要**逐细胞 `I_I(t)`**，**现有 artifact 里没有**，由 §5.1 一并落盘。

### 6.3 锁定规则

1. 固定 `τ_Z_down = 5000 ms`、`τ_Z_up = 20000 ms`；
2. 同一条 replay 上，用 `I_th = 95.19851312666987`（q75）与 `1.6652801609959704`（q50）
   定义弱 / 强端点 `S_Z^{q75}`、`S_Z^{q50}`；
3. 验证 `S_Z(I_th)` 在 `[q50, q75]` 上**严格单调**；
4. 在 `S_Z` 的 **25% / 50% / 75%** 处反解三个 `I_th_EI`；
5. 三值写入附录 A + manifest，**此后不得移动**；
6. 不单调或 replay 覆盖不到端点 → **`DESIGN_BLOCKED_Z_RESPONSE_AXIS`**，**不得**回二维网格。

### 6.4 与 HYB1 三档并列记录

HYB1 的三档（`h_Z` 几何四分点）= `I_th_EI` 96.30 / 72.35 / 46.80，其中 **H_LO 落在 q75 上
（比值 0.996）不是内点**。HYB2 三档必须与之并列，**并显式说明新的 25% 点是否仍贴着 q75**——
若仍贴着，那不是失败，但**同样不得称为内点**。

---

## 7. 两道前置门

### 7.1 Gate B0 — baseline invisibility

ELR off / on，无 kick，`T = 24 s`，seed 1 与 seed 3（off 臂复用 §5.1）。

| # | 子句 | 门 |
|---|---|---|
| 1 | `R_evt` 活跃占空比（**validation 半段 + seed3**） | ≤ 0.01 |
| 2 | **下一事件 onset 前 2 ms 窗**内 `q_v` 的 q99（跨事件×占据体素） | ≤ `0.01 · Q_on` |
| 3 | **`q_v` 无单调抬升**：`(q_floor,末段 − q_floor,首段)/Q_on` | ≤ 0.01 |
| 4 | returning event rate / IEI CV / duration / participation / silent fraction | 落在该 seed 的 `baseline_contract` 带内 |
| 5 | 数值安全 | 零 clip、finite、`tau_eff_min ≥ 2dt` |

> 子句 2 用**下一事件前的小窗**而不是整个 gap：事件刚结束时 `q_v` 本来就高，
> 把它算进"残留"会让这条**按定义必败**，而且它测的是衰减起点不是残留。
> 子句 3 用 **`Q_on` 归一化的差值**而不是比值：首段接近零时比值数值不稳。
>
> **`Q_on` = calibration 最大值 × 1.1 ⟹ 子句 1 在 calibration 半段上按构造必然通过。**
> **但子句 3 不是构造保证**——`q_v` 不棘轮正是 HYB1 栽的地方，**子句 3 是 B0 的主判据**。
> 报告时**只能写**"在 validation 半段与第二个 seed 上未观察到基线扰动"，
> **不得写**"已证明事件尺度执行器不打扰基线"。**风险主要在 A0。**

失败 → **`STOP_ELR_BASELINE_VISIBLE`**；禁止调 drive / 连接 / `Q_on` / `τ_R` / `I_R,max`。

### 7.2 Gate A0 — actuator efficacy（**本轮真正的风险点**）

见 §5.2–5.3。**plan 必须先写明 A0 在测什么**：

去掉扩散项之后，`R_evt` 在一个体素里**只取决于该体素自己的负荷**，所以执行器
**只能放大已经在活动的组织，不能点亮一个安静的体素**。招募范围仍可能扩大，但因果路径是
"被放大的体素经 recurrent scaffold 驱动邻居"——**一步突触，不是一个扩散场**。

B2.1 那 +24% 半径是在 `τ_K = 654 ms` 记忆 **加** `D_K` 扩散 **加** 200 ms 窗内累积下测到的；
HYB2 把记忆缩短约 **24 倍**、把扩散**删掉**、另加了 deadband / 阈值 / 饱和曲线。

> **A0 不过时只能写**："**当前这个短记忆、无扩散、阈值化的 ELR 未能保留 B2.1 的招募 extent 效应**"。
> **不得**写成"B2.1 的招募依赖跨事件浓度记忆"——**同时改了至少三件事**，单臂失败无法归因到任何一件。
> 真要单独归因记忆需要 memory × diffusion 对照；**本轮不扩成 2×2，只收紧措辞**。

---

## 8. 顺序与停机（任一上游门失败即停，不扩网格）

| # | 阶段 | 计算量 | 失败 → |
|---|---|---|---|
| 0 | 组件符号审计（沿用 HYB1 `TOPOLOGY_OK`，补 ELR 新增两条符号） | 0 | `TOPOLOGY_INPUT_UNRESOLVED` |
| 1 | artifact preflight（存在性 + sha256） | 0 | 缺失即停 |
| 2 | **§5.1 calibration run（本 plan 授权的唯一 40k 运行）** → `calibration_lock.json` | 2 × 24 s（seed1 + seed3，各兼 B0 off 臂） | `DESIGN_BLOCKED_EVENT_TIMESCALE` / `CALIBRATION_INVALID` |
| 3 | §6 `S_Z` 轴（**离线 replay，零仿真**） | 0 | `DESIGN_BLOCKED_Z_RESPONSE_AXIS` |
| 4 | **Gate B0**（只需补 ELR-on 两臂） | 2 × 24 s | `STOP_ELR_BASELINE_VISIBLE` |
| 5 | **Gate A0**（三分判决） | 2 × 9 s | `NO_GO_EVENT_LIMITED_ACTUATOR`（仅 eligible 后） |
| 6 | 12 格短屏（3 `S_Z` 点 × ELR{off,on} × X{off,on}，M off，无 kick） | 12 × 14 s | 按失败层归档 STOP |
| 7 | ≤2 survivor 长窗七门 | ≤2 × 24 s | 无 survivor → 不跑 M、不跑 confirmatory |
| 8 | `M` 固定一档（`tau_adp=250 ms` + force-matched 10%） | 2 × 24 s | waveform trade-off，不称成功 |
| 9 | seed3 + unseen noise | 视情况 | 不回调参数 |

**运行时预算（LC1 实测锚）**：24 s 有界 run ≈ 23–36 min，**24 s 失控 run = 4.3 h**。
阶段 2–5 合计约 **3–4 h**（1 worker）。

---

## 9. 资源合同（**完整恢复，不只是"沿用脚手架"**）

开工时重新记录：`nproc`、load、`MemAvailable`、swap used baseline、sibling 40k 进程数与 RSS、
六个 blessed 文件 sha256。

- `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = NUMEXPR_NUM_THREADS = 1`；
- **`T ≥ 20 s`：严格 1 worker**（§8 的阶段 2、4、6-24s、7、8 全部落在这一档）；
- `T < 20 s`：最多 2 worker，且**提交前**必须核 sibling 40k ≤ 2、`MemAvailable ≥ 96 GiB`
  **且** ≥ 2 × 实测单 run peak RSS、swap 不增长；
- swap 相对本 sprint baseline：**> +256 MiB 停止提交新任务**；**> +512 MiB 且继续增长** →
  只终止**自己最新的**任务并写 `RESOURCE_PAUSED.json`；
- **不杀任何 sibling / user 进程**；
- 每个 run 写 `resource_log.jsonl`；
- 长任务一律：

```
setsid nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python scripts/run_topic4_fcxr_hyb2.py --stage <stage> --confirm-run \
  > <run_root>/nohup_<stage>.log 2>&1 < /dev/null &
```

- **stage-scoped flock**（不是 FCXR 全局 build lock）+ `launcher_<stage>.pid` +
  `RUNNING_/DONE_/FAILED_<stage>.json`；per-cell DONE sentinel ⟹ 断线重连**不重复提交已完成 cell**；
- **wall-time kill guard = 3600 s / cell** + 科学 early-stop（连续 2 s 均值率 > 300 Hz 判 `RUNAWAY`，
  记录而非重跑）。

> **一条已知的自锁陷阱（HYB1 踩过，记下来别再犯）**：用 `pgrep -f "<stage 名>"` 写等待循环会
> **匹配到 waiter 自己的命令行**，导致链式启动永远空转。等待必须**按 PID**（`kill -0 <pid>`）。

---

## 10. 七门与坏数据回归（沿用 HYB1 已锁版本，不动）

七门与 q75 / q50 / q50-无-X 三条坏数据回归**逐字沿用** HYB1 plan §5，已在
`tests/test_topic4_fcxr_hyb1.py` 落地（39 tests 绿）。**ELR 不能替代任何一门，
也不得用七门结果反调 B0 / A0 的参数。**

Gate 4 空间腿仍按 HYB1 §5.1 的条件规则：`recruit ≥ 12/15` 与 onset 梯度**若都分不开**
structured event 与 synchronous negative control → **`UNRESOLVED`**，不发明方向结论。
（HEO2.1 已量过 48/48 个工作点 recruit ≥13/15，含纯同步态，所以这条腿大概率不判别。）

---

## 11. 工程与测试

**新写**：`src/snn_engine/event_limited_recruitment.py`、`src/topic4_fcxr_hyb2.py`、
`scripts/run_topic4_fcxr_hyb2.py`、`scripts/plot_topic4_fcxr_hyb2.py`、
`tests/test_event_limited_recruitment.py`、`tests/test_topic4_fcxr_hyb2.py`。

**复用不重造**：RC1 底座、`Z`（含 HYB1 已落地的非对称恢复）、`X`、`M`、`D_Z`/`D_X`、
生命周期分类器与七门、baseline 合同、stage-scoped flock / sentinel / resource_log、
B2.1 的 `E_K` 幅度锚、HYB1 已测的 deadband 实现。
**六个 blessed engine 文件不得修改，每阶段核 SHA。**

必测：

- `q_v` 在背景以下**严格**为 0（deadband 非 softplus）；`R_evt(q_v ≤ Q_on) ≡ 0`；
- `R_evt ≤ I_R,max` 对任意持续输入成立（`tanh` 有界性回归）；
- **精确指数更新**对分块常值输入与解析解一致；`dt_R` 改变时结果收敛；
- 活动停止后 `q_v` 按 `τ_R` 自主衰减，**无 hard reset、不读离线 event label**
  （**源码级断言**：模块内不得出现 `detect_events` / 事件标签 / 未来时刻索引）；
- **跨事件不累积**：给两串间隔 = `GAP_05` 的合成事件，第二串起点前 2 ms 窗的 `q_v` ≤ `0.01·Q_on`；
- **HYB1 棘轮回归（判决器资格测试）**：把 `τ_R` 换成 **654 ms** 重放同一合成输入，
  子句 3 的差值**必须 > 0.01**——**判决器要能重现 HYB1 的失败，才有资格判 HYB2 通过**；
- **A0 三分判决**的三条分支各有合成用例（未越阈 / 窗口不足 / off 臂饱和）；
- `t_gate` 在两臂上**逐位相同**；off 臂只置零电流、`q_v` 照常演化；
- E 与 I 都收到同一 current；`g_rel`/`g_rev` 不被触碰；absent 引擎属性保持 absent；
- 确定性、snapshot/restart、off-parity 逐比特；
- `S_Z` 单调性 + `C(T_cal)` 解析值回归；`T_cal < 15 s` 必须告警；
- `Q_on ≤ 0` / `Q_scale ≤ 0` → `CALIBRATION_INVALID`；artifact 缺失 → fail closed。

---

## 12. 允许 / 禁止

**允许**（且仅在对应门通过后）：

- B0 过 → "在 validation 半段与第二个 seed 上**未观察到**基线扰动"；
- A0 过 → "事件尺度有界执行器**保留了** B2.1 的招募 extent 效应"（只说 extent，不说 propagation）；
- A0 不过（且 `A0_ELIGIBLE`）→ "**当前这个短记忆、无扩散、阈值化的 ELR 未能保留该效应**"。

**禁止**：把 `R_evt` 称为细胞外钾浓度 / ion homeostasis / 患者离子机制；脚本 event reset；
为过门调 `Q_on` / `I_R,max` / `τ_R` / 恢复扩散；把 A0 阴性归因到"跨事件记忆"；
七门前称 seizure lifecycle；未做动力学分析称 limit cycle / Hopf / bistability；
未测 onset 梯度与延迟称 propagation；source/sink 成功措辞；把 engineering green 写成科学成功。

> **最后一条边界（审阅原话，保留）**：即使 B0/A0 都过，那也只说明**招募执行器可能合格**，
> **不代表已接近有界发作振荡**。核心科学目标要到七门同时成立才触及。

---

## 附录 A — 待 §5.1 跑完后回填（**回填前不得进入阶段 3 以后**）

| 项 | 状态 |
|---|---|
| `T_event,guard` | **已锁 22.0 ms**（LC1 24 s 冻结 bar，两 seed 事件时长最大值） |
| `GAP_05`、`GAP_01`、最短 `GAP` | 待回填（canonical onsets **与 offsets**；预审下界 ≥ 147.5 ms） |
| `τ_R`（§3.4 公式）＋ 四处尾部残留 ＋ 区间余量 | 待回填（预期 ≈ 26.55 ms，余量约 10 ms） |
| `b_v`（seed1 / seed3 各自）、`ε_s`、`Q_on`、`ε_q`、`Q_scale = Q_on` | 待回填 |
| seed1-`b_v`-套-seed3 的二级诊断 | 待回填（仅诊断，不作门） |
| `I_R,max` | **已锁 4.134151260609386**（逐位复现） |
| `S_Z^{q75}`、`S_Z^{q50}`、三个 `I_th_EI`（`T_cal = 24 s`） | 待回填 |
| 与 HYB1 三档（96.30 / 72.35 / 46.80）的并列对照 | 待回填 |
| 六个 blessed sha256 + 全部输入 artifact sha256 | 待回填 |
