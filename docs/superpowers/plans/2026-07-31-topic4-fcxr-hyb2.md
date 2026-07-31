# FCXR-HYB2 实施计划 —— 事件尺度、有界幅度的自主招募层（ELR）

日期：2026-07-31 · 分支 `codex/topic4-fcxr-hyb2`（**待建**） · 基点 `9c5c5b16`
Spec：`docs/superpowers/specs/2026-07-31-topic4-fcxr-hyb2-event-limited-recruitment-design.md`
（含 2026-07-31 六处审阅订正；**冲突时以本 plan 的数值为准**）

**本文件在任何 HYB2 结果出现之前锁死。之后只允许追加"执行结果"，不允许改判据、阈值或档位。**

前代不可改写：`docs/archive/topic4/sef_hfo/fcxr_hyb1_baseline_disturbed_2026-07-31.md`
（`STOP_BASELINE_DISTURBED`）。HYB1 的 spec/plan 已标记为已执行并收口，**不得回填**。

---

## 0. 一句话

HYB1 否掉的是"**用 0.65 秒的浓度记忆承载招募**"——它和间期事件 0.4–0.6 秒的间隔同量级，
所以钾在事件之间清不干净、逐级棘轮。HYB2 换一个**记忆只有事件长度（约 23 毫秒）、幅度有上限、
活动停了自己就衰减**的执行器，先问两件事：**(B0) 它在平时看不看得见？(A0) 它还留不留得住
B2.1 那个招募效应？** 两问之一失败就干净停机，**不烧 12 格**。

---

## 1. 锁定项 1 —— 事件时间尺度与 `τ_R`

### 1.1 唯一来源（冻结 event bar）

**只用** `/.worktrees/topic4-mz-fcxr-lc1/.../lifecycle_closure/baseline_contract_seed{1,3}.json`
（LC1 的 24 s slow-off 基线，bar 已冻结）。

**明令禁止**用 HYB1 自己那条 8 s 基线：它在 seed3 上给 duration 中位 39.0 ms，而同配置的 LC1 24 s
给 11.0 ms——差 3.5 倍，因为短窗会**重新推导** event bar。

### 1.2 已核数值（零仿真，已从上述 artifact 读出）

| 量 | seed1 | seed3 | 锁定值 |
|---|---:|---:|---:|
| n events (24 s) | 34 | 67 | — |
| duration 中位 (ms) | 10.0 | 11.0 | — |
| duration q90 (ms) | 19.7 | 14.0 | — |
| duration max (ms) | 22.0 | 22.0 | — |
| **`T_event,90`（保守 = 两 seed 的 max）** | | | **22.0 ms** |

**取最保守的 22.0 ms，不取 pooled q90 的 15.0 ms。** 理由：`τ_R` 必须长到装得下**一个完整的
较长事件**，用 pooled q90 会让 10% 的事件被截断。

### 1.3 `IEI_05` 必须重算（现有 artifact 不够）

`baseline_contract_*.json` 存了 duration 与 participation，**没有存事件起始时刻**，所以
`IEI_05` 只能从下采样 rate trace 近似（spec 预审给出 169.5 ms）。

**锁定**：`IEI_05` 由 §5 那条 calibration run 的 **canonical event onsets** 重算
（同一 `_events_from_res` + 同一冻结 bar），写入 manifest 后冻结。

### 1.4 `τ_R` 的唯一选择规则（先写死，不扫）

$$\tau_R := \sqrt{T_{\mathrm{event},90}\cdot \frac{IEI_{05}}{\ln 100}}\qquad
\text{（可行区间 } [T_{\mathrm{event},90},\ IEI_{05}/\ln 100] \text{ 的几何中点）}$$

- 若 `T_event,90 ≥ IEI_05/ln 100` → 区间为空 → 判 **`DESIGN_BLOCKED_EVENT_TIMESCALE`** 并停止，
  **不许**调 baseline、不许跳到 40k；
- 用预审的 `IEI_05 = 169.5 ms` 代入，区间 `[22.0, 36.8]` **非空**，`τ_R ≈ 28.5 ms`
  （**这是预期值，不是锁值**；锁值由 §5 重算后写回本文件）。

### 1.5 必须一并写入 manifest 的尾部残留（**不许藏**）

`exp(−IEI/τ_R)` 在三处：**`IEI_05`、`IEI_01`、以及实测最短 IEI**。
预审在 `τ_R = 23.5 ms` 下：169.5 ms → 0.0007、100 ms → 0.0142、60 ms → 0.0778。
**`IEI_05` 规则只覆盖 95% 的间隔；最短那 5% 的残留必须可见。**

---

## 2. 锁定项 2 —— calibration / validation 划分、`Q_on`、`Q_scale`

### 2.1 划分（先写死）

一条 **T = 24 s** 的 sensor-only calibration run（§5），按时间**前 12 s = calibration、
后 12 s = validation**。`Q_on` **只**用 calibration 半段；validation 半段与 **seed 3**
**只**用于 Gate B0 的检验，**不得**参与定阈。

### 2.2 `Q_on`

1. 在 calibration 半段上，对每个**被占据**体素记录 `q_v(t)` 的**逐事件峰值**
   （事件由同一冻结 bar 的 canonical onsets 划窗，窗 = `[t_on, t_on + 3τ_R]`）；
2. **`Q_on := 1.10 × max_{event, occupied voxel} q_v^{peak}`**（calibration 最大值 + 10% margin）；
3. 写入 manifest 并冻结。**不得因 B0/A0/lifecycle 结果下调。**

### 2.3 `Q_scale` —— 改用解析规则（**spec 原文那条不可实现**）

按 §2.2，calibration 上 `q_v − Q_on ≤ 0` 恒成立，**没有任何样本落在 `Q_on` 之上**，
所以"从 calibration 解析 `q_v−Q_on` 的尺度"是空集。**锁定解析规则**：

$$Q_{\mathrm{scale}} := Q_{\mathrm{on}}$$

即超出间期上界"一倍"时执行器达到 `tanh(1) = 0.7616 × I_{R,\max}`。**零自由度，只依赖已锁的 `Q_on`。**

**不采用**"从 Gate A0 的 supra-interictal artifact 取尺度"——那条 artifact 同时是 A0 的被测输入，
用它定标会让 A0 的 ≥10% 判据部分自证。

`Q_scale` 与 `I_R,max` **都不扫**。

---

## 3. 锁定项 3 —— `I_R,max` 单位链（**已核，逐位复现**）

$$I_{R,\max} := g\cdot\big[E_K(K_{o,0}+\delta K^\star) - E_K(K_{o,0})\big],\quad
g=1,\ \delta K^\star=0.6715\ \mathrm{mM}$$

用已验收实现（`src/topic4_fcxr_ion.E_K`，`RTF=26.64`、`K_i=140`、`K_{o,0}=4`）算得

**`I_R,max = 4.134151260609386` engine-drive units** —— 与 spec 的数字**逐位一致**（差 < 1e-12）。

- `δK* = 0.6715 mM` 的出处：B2.1 匹配对照**闭环臂**窗 2 的热点峰值
  （`ion_homeostasis/b2_1_matched_control.json::arms.closed.peak2_mM`）；
- 这是 **force anchor**，**不是浓度解释**。`R_evt` 的状态量无 mM 单位，
  **禁止**称其为 extracellular potassium / ion homeostasis / 患者离子机制；
- plan 只核 sha 与单位链，**不重新拟合**。

---

## 4. 锁定项 4 —— `S_Z` 开环累积耗竭轴与三个 `I_th_EI`

### 4.1 这条轴到底是什么（**审阅订正后的正确说法**）

`S_Z` 在**冻结的 slow-off 负荷轨迹**上离线 replay，`I_I(t)` 不受 `z` 影响，
**所以它结构上不可能表现"耗竭自限"**（自限是 `z↓ → I_I↓ → 落回阈下 → 恢复` 的闭环通路）。

更强的算术：若一个细胞在窗内阈上/阈下状态不变，所有阈上细胞走同一条 `z(t)=exp(−t/τ)`，于是
`S_Z(I_th) = a_p(I_th)·C(T_cal, τ)` —— **与 `h_Z` 严格成正比，不含新信息**。

**`S_Z` 唯一多出来的信息 = 那些在窗内跨越阈值的细胞**，即它量的是**阈上时间占比**而不是
某一瞬间的阈上比例。这比 `h_Z` 稳健，也把 1.64 倍窄区间上的三点分得更开，
**但它不是"整段耗竭响应"，也不是自限的度量。三点能否分开闭环行为，由 12 格短屏回答。**

### 4.2 两条因此必须写死的要求

- **`T_cal := 24 s`**（≥ `3·τ_Z_down = 15 s`）。实测 `C(T)`：`T_cal` = 3 / 5 / 15 / 24 s →
  0.248 / 0.368 / 0.683 / 0.793。**低于 15 s 时 `C` 仍在近线性段，换轴等于没换。**
- replay 需要**逐细胞 `I_I(t)`**。**现有 artifact 里没有**（HYB1 的 zaxis probe 只落盘了生存曲线
  与分位数）。由 §5 那条 calibration run 以 **100 ms 节奏 × 240 帧 × NE** 落盘（float32，约 30 MB）。

### 4.3 锁定规则

1. 固定 `τ_Z_down = 5000 ms`、`τ_Z_up = 20000 ms`；
2. 在同一条 replay 上，用 `I_th = 95.19851312666987`（q75）与 `1.6652801609959704`（q50）
   定义弱 / 强端点 `S_Z^{q75}`、`S_Z^{q50}`；
3. 验证 `S_Z(I_th)` 在 `[q50, q75]` 上**严格单调**；
4. 在 `S_Z` 坐标的 **25% / 50% / 75%** 处反解三个 `I_th_EI`
   （`S_Z^{q75} + f·(S_Z^{q50} − S_Z^{q75})`，`f = 0.25/0.50/0.75`）；
5. 三个值写入 plan 附录 + manifest，**此后不得移动**；
6. 不单调或 replay 覆盖不到端点 → **`DESIGN_BLOCKED_Z_RESPONSE_AXIS`**，**不得**回二维网格。

### 4.4 与 HYB1 的差别必须在 manifest 里并列

HYB1 的三档（`h_Z` 几何四分点）是 `I_th_EI` = 96.30 / 72.35 / 46.80，其中 **H_LO 落在 q75 上
（比值 0.996）不是内点**。HYB2 的三档必须与之并列记录，**并显式说明新的 25% 点是否仍贴着 q75**——
如果仍贴着，那不是失败，但**同样不得称为内点**。

---

## 5. 锁定项 5 —— 唯一的 calibration run 与 matched actuator input

### 5.1 一条 run 同时落盘四样东西（**只跑这一条**）

`seed=1`、`T = 24 s`、**RC1 accepted 工作点、慢变量全 off、ELR off、无 kick**，落盘：

| 落盘 | 用途 |
|---|---|
| canonical event onsets + durations（同一冻结 bar） | §1.3 `IEI_05`、§2.2 划事件窗 |
| 逐体素 `s_v(t)` 流式峰值统计 + `b_v` | `Q_on`、deadband |
| 逐细胞 `I_I(t)`，100 ms × 240 帧 | §4 `S_Z` replay |
| 逐细胞 `p_i`（沿用 `zA_q75_tz5000` snapshot） | `D_Z` 权重 |

**它不是 actuator run 也不是 lifecycle run。§2 的 selection rule 必须在它跑之前就写死，
不得据其结果修改规则。**

### 5.2 Gate A0 的 matched input —— 唯一 provenance

- **输入 = LC1 的 `q50` Z-only 配置**（`I_th_EI = 1.6652801609959704`、`tau_z = 10000 ms`），
  `seed = 1`，`T = 6 s`，**无 kick**；
- 选它的理由（先写死）：`q50` 是**已验收**能把网络推到 supra-interictal 持续高负荷的档
  （LC1 E2：`D_Z→0.805`、末段 452.8 Hz），因此它一定能把 `q_v` 推过 `Q_on`——
  A0 于是**只**检验执行器的招募效力，不再受"输入够不够强"的混淆；
- 两臂（ELR off / on）**同 substrate、同 seed、同初值**，在 ELR 首次启动前**逐比特一致**；
  有效性由此**结构性**保证，不由响应差异保证（复用 B2.1 §2.3 已订正的判据形态）；
- **6 s 而不是 24 s**：q50 的 24 s Z-only 曾跑 **4.3 小时**（runaway）。6 s + `RUNAWAY_RATE_HZ=300`
  科学 early-stop + **3600 s wall kill guard**。

### 5.3 A0 的 ≥10% 判据实现

窗口 = ELR 首次启动后的 **1000 ms**。三个量沿用 B2.1 已实现的定义：
`kick_participants`（改名 `window_participants`）、`recruitment_radius_mm`、`participant_voxels`。

**通过 = 三者中至少两个相对 off 臂 ≥ +10%**，且 `max_v R_evt ≤ I_R,max`、零 clip、finite。
失败 → **`NO_GO_EVENT_LIMITED_ACTUATOR`**，**禁止**为过门扫 `I_R,max`、恢复扩散项或加长 `τ_R`。

---

## 6. 两道前置门的判据（含审阅订正）

### 6.1 Gate B0 — baseline invisibility

ELR off / on，无 kick，`T = 24 s`，seed 1 与 seed 3。

| # | 子句 | 门 |
|---|---|---|
| 1 | `R_evt` 活跃占空比（**validation 半段 + seed3**） | ≤ 0.01 |
| 2 | 事件间隙内 `q_v` 残留 q99 | ≤ `0.01 · Q_on` |
| 3 | **`q_v` 的** pre-event floor：末段 / 首段 | ≤ 1.10 |
| 4 | returning event rate / IEI CV / duration / participation / silent fraction | 落在该 seed 的 `baseline_contract` 带内 |
| 5 | 数值安全 | 零 clip、finite、`tau_eff_min ≥ 2dt` |

> **子句 3 定义在 `q_v` 上，不是 `R_evt` 上**——间期 `R_evt ≡ 0`，比值 0/0 无定义。
> 这正是 HYB1 栽的那种"棘轮"，所以这一条是 B0 的**主判据**。

> **⚠️ B0 通过不是正面证据。** `Q_on` = calibration 最大值 × 1.1 ⟹ 子句 1、3 在 **calibration
> 半段上按构造必然通过**。有信息量的只有 validation 半段 + seed3 上的 1、2 条与子句 4。
> 报告时**只能写**"在 validation 半段与第二个 seed 上未观察到基线扰动"，
> **不得写**"已证明事件尺度执行器不打扰基线"。**风险全部集中在 A0。**

失败 → **`STOP_ELR_BASELINE_VISIBLE`**；禁止调 drive / 连接 / `Q_on` / `τ_R` / `I_R,max`。

### 6.2 Gate A0 — actuator efficacy（**本轮真正的风险点**）

见 §5.2–5.3。**必须在 plan 里先写明 A0 在测什么**：

去掉扩散项之后，`R_evt` 在一个体素里**只取决于该体素自己的负荷**，所以执行器
**只能放大已经在活动的组织，不能点亮一个安静的体素**。招募范围仍可能扩大，但因果路径是
"被放大的体素经 recurrent scaffold 驱动邻居"——**一步突触，不是一个扩散场**。

而 B2.1 那 +24% 半径是在 `τ_K = 654 ms` 记忆 **加** `D_K` 扩散 **加** 200 ms 窗内累积下测到的；
HYB2 把记忆缩短约 **28 倍**、把扩散**删掉**。**A0 很可能失败，而那会是一个干净的结果**——
它把"B2.1 的空间招募是否依赖跨事件浓度记忆"回答成"是"。

---

## 7. 顺序与停机（任一上游门失败即停，不扩网格）

| # | 阶段 | 计算量 | 失败 → |
|---|---|---|---|
| 0 | 组件符号审计（沿用 HYB1 的 `TOPOLOGY_OK`，只补 ELR 新增的两条符号） | 0 | `TOPOLOGY_INPUT_UNRESOLVED` |
| 1 | §5.1 calibration run（**锁 `τ_R` / `Q_on` / `Q_scale` / `S_Z` replay 输入**） | 1 × 24 s | `DESIGN_BLOCKED_EVENT_TIMESCALE` |
| 2 | §4 `S_Z` 轴（**离线 replay，零仿真**） | 0 | `DESIGN_BLOCKED_Z_RESPONSE_AXIS` |
| 3 | **Gate B0** | 4 × 24 s | `STOP_ELR_BASELINE_VISIBLE` |
| 4 | **Gate A0** | 2 × 6 s | `NO_GO_EVENT_LIMITED_ACTUATOR` |
| 5 | 12 格短屏（3 `S_Z` 点 × ELR{off,on} × X{off,on}，M off，无 kick） | 12 × 14 s | 按失败层归档 STOP |
| 6 | ≤2 survivor 长窗七门 | ≤2 × 24 s | 无 survivor → 不跑 M、不跑 confirmatory |
| 7 | `M` 固定一档 | 2 × 24 s | waveform trade-off，不称成功 |
| 8 | seed3 + unseen noise | 视情况 | 不回调参数 |

**运行时预算（LC1 实测锚）**：24 s 有界 run ≈ 23–36 min，**24 s 失控 run = 4.3 h** →
每个 cell **3600 s wall kill guard** + 连续 2 s 均值率 > 300 Hz 的科学 early-stop。
阶段 1–4 合计约 **3.5 h**（1 worker）或 **2 h**（2 worker，`T<20 s` 才允许）。

---

## 8. 七门与坏数据回归（沿用 HYB1 已锁版本，不动）

七门与 q75 / q50 / q50-无-X 三条坏数据回归**逐字沿用** HYB1 plan §5，已在
`tests/test_topic4_fcxr_hyb1.py` 落地（36 tests 绿）。**ELR 不能替代任何一门，
也不得用七门结果反调 B0 / A0 的参数。**

Gate 4 的空间腿仍按 HYB1 §5.1 的条件规则：`recruit ≥ 12/15` 与 onset 梯度**若都分不开**
structured event 与 synchronous negative control → **`UNRESOLVED`**，不发明方向结论。
（HEO2.1 已量过 48/48 个工作点 recruit ≥13/15，含纯同步态，所以这条腿大概率不判别。）

---

## 9. 工程与测试

**新写**：`src/snn_engine/event_limited_recruitment.py`、`src/topic4_fcxr_hyb2.py`、
`scripts/run_topic4_fcxr_hyb2.py`、`scripts/plot_topic4_fcxr_hyb2.py`、
`tests/test_event_limited_recruitment.py`、`tests/test_topic4_fcxr_hyb2.py`。

**复用不重造**：RC1 底座、`Z`（含 HYB1 已落地的非对称恢复）、`X`、`M`、`D_Z`/`D_X`、
生命周期分类器与七门、baseline 合同、stage-scoped flock / sentinel / resource_log、
B2.1 的 `E_K` 幅度锚。**六个 blessed engine 文件不得修改，每阶段核 SHA。**

必测：

- `q_v` 在背景以下**严格**为 0（deadband，非 softplus）；`R_evt(q_v ≤ Q_on) ≡ 0`；
- `R_evt ≤ I_R,max` 对任意持续输入成立（`tanh` 有界性回归）；
- 活动停止后 `q_v` 按 `τ_R` 自主衰减，**无 hard reset、不读离线 event label**
  （源码级断言：模块内不得出现 `detect_events` / 事件标签 / 未来时刻索引）；
- **跨事件不累积**：给两串间隔 = `IEI_05` 的合成事件，第二串起点的 `q_v` ≤ `0.01·Q_on`；
- **HYB1 棘轮回归**：把 `τ_R` 换成 654 ms 重放同一合成输入，地板比值**必须 > 1.10**
  （判决器要能重现 HYB1 的失败，才有资格判 HYB2 通过）；
- E 与 I 都收到同一 current；`g_rel`/`g_rev` 不被触碰；absent 引擎属性保持 absent；
- 确定性、snapshot/restart、off-parity 逐比特；
- `S_Z` 单调性 + `C(T_cal)` 解析值回归；`T_cal < 15 s` 时必须告警；
- 数值稳定：`dt_ion/τ_R` 与显式步长裕度。

---

## 10. 允许 / 禁止

**允许**（且仅在对应门通过后）：

- B0 过 → "在 validation 半段与第二个 seed 上**未观察到**基线扰动"；
- A0 过 → "事件尺度有界执行器**保留了** B2.1 的招募 extent 效应"（只说 extent，不说 propagation）；
- A0 不过 → "**B2.1 的空间招募依赖跨事件浓度记忆**"（这是一个干净的正结论，不是失败）。

**禁止**：把 `R_evt` 称为细胞外钾浓度 / ion homeostasis / 患者离子机制；脚本 event reset；
为过门调 `Q_on` / `I_R,max` / `τ_R` / 恢复扩散；七门前称 seizure lifecycle；
未做动力学分析称 limit cycle / Hopf / bistability；未测 onset 梯度与延迟称 propagation；
source/sink 成功措辞；把 engineering green 写成科学成功。

---

## 附录 A — 待 §5.1 跑完后回填（**回填前不得进 40k**）

| 项 | 状态 |
|---|---|
| `T_event,90` | **已锁 22.0 ms**（LC1 24 s 冻结 bar，两 seed max） |
| `IEI_05`、`IEI_01`、最短 IEI | 待回填（canonical onsets） |
| `τ_R`（§1.4 公式）＋ 三处尾部残留 | 待回填（预期 ≈ 28.5 ms） |
| `Q_on`、`Q_scale = Q_on` | 待回填 |
| `I_R,max` | **已锁 4.134151260609386**（逐位复现） |
| `S_Z^{q75}`、`S_Z^{q50}`、三个 `I_th_EI` | 待回填（`T_cal = 24 s`） |
| 与 HYB1 三档（96.30 / 72.35 / 46.80）的并列对照 | 待回填 |
