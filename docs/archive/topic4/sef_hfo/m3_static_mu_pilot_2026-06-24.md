# M3 static-μ **spontaneous-first** pilot with finite-pulse basin support — spec (2026-06-24 v2)

> **朴素一句话**：真实的间期→发作不是谁推一把，而是**慢变量(μ)慢慢爬高、网络自己的随机涨落越过边界、
> 自发地把事件从"小、点一下扩一点就回静"长成"大、回静"再到"不回静、持续招募"**。所以本 pilot 的**主问题
> 是自发的**：不 kick、长记录、扫 μ，看网络自己有没有从 R2→R3→R4a 迁移。kick basin 只是**辅助机制探针**
> （解释 basin 边界有没有收缩），**不是发作的定义**。5×5 下 B1b/c 看不出方向是分辨率问题，并行验证、不阻塞。

v1（finite-pulse 为 primary）被用户 review 推翻（2026-06-24）：把 kick 当主线 = 偷偷假设"癫痫是被外部触发的"。
本 v2 改为 **spontaneous-first + basin-support**。pivot 来源：[[m3_b1_validation_recap_2026-06-24]] §9 + 用户 review。

---

## 0. Gate（硬合同）

**ON**：静态（quasi-static）μ 经已有 per-neuron `V_th` 路径（μ=0 bit-parity，engine 不动）；**spontaneous
no-kick 长记录 = 主 claim gate**；finite-pulse kick basin = 辅助机制探针；R0–R4 分类（R4a vs R4b）；
横轴 `μ`(=ΔVth) / `K_min(μ)` / 事件 size 分布。高分辨率 `W_event` = **并行非阻塞**验证线。

**OFF（禁止）**：dynamic m(t)；hub / corridor / endpoint / W_escape；改 SNN engine；改 detector 阈值追结果；
**μ full grid**；**L32 confirm**；把高分辨率 W 当阻塞 gate；把 **R4b tonic runaway** 写成 seizure-like bridge
（**只有 R4a** W-aligned sustained/recurrent recruitment 算 synthetic bridge candidate）；把 `W_event` 写成
已证明的方向性算子。

**纪律**：μ=0 bit-parity（回归硬条件，已验证 md5 `1086c5ef…` 不变）；PILOT-FIRST（先 tiny，不全网格）；做扎实。

---

## 1. μ 耦合 + h 场（已实现，复用）

```
V_th_eff,i = vth_core_i − ΔVth(μ)·h_i ,    ΔVth(μ) = dvth_at_mu1 · μ ,  dvth_at_mu1 = 1.333 mV
```
μ ∈ [0,1] → ΔVth ∈ {0,0.2,…,1.2} mV。μ=0 ⇒ V_th_eff=vth_core ⇒ 逐字节一致。engine 不碰。
**h 决策（用户"两个都跑"）**：本轮 primary h = `core_susceptibility`（核阈值压低场，现成）；`uniform`/`shuffled`
= 对照；empirical `kmin_susceptibility`-h 等并行 dense sweep 后**后验复现**（不阻塞本轮）。
（实现：`src/sef_hfo_mu_basin.apply_mu` + runner `--mu/--dvth-at-mu1/--h-mode`，已提交 `5e8b5d0`。）
**自洽 sanity**：μ=.90 时核 ≈16.4mV，落在早先 mean-amplitude scan 的自发点火边界 ~16.3–16.9mV 上 → μ 网格
确实跨过自发 returned→escape 边界。

---

## 2. 两套分类输入（**必须分开，不可混**）

| 模式 | 输入 | 分类依据 |
|---|---|---|
| **spontaneous（§3 primary）** | **no-kick** 长记录里检测到的**每个自发事件**（不对齐任何 kick） | **raw** 群体活跃轨迹的事件检测 + 每事件 size/duration/回静/前沿 → R0–R4 |
| **kick-basin（§4 support）** | 对每个 (K,μ) 的 **kick-aligned** 响应 | raw `core_kick` 回静/失控 + **spontaneous-ignition flag**（**不用差分**，因高 μ 下 sham 本身自发） |

> ⚠️ **高 μ 下 `core_only`/no-kick 自发点火不是 contamination，是 spontaneous phenotype 本身**——
> 它正是我们要测的东西。所以 spontaneous 模式直接读它；kick-basin 模式用 raw+flag、不用 kick−sham 差分作主分类。

---

## 3. §3 PRIMARY — spontaneous no-kick μ 扫描（**主 claim gate**）

**问题**：不同 μ 下，网络**自己**是否从 R2（finite returned）过渡到 R3（large returned），再到 **R4a**
（sustained / recurrent recruitment）？

**做法**：core 场（含 μ）**不施加 kick**，跑长记录，检测 [0,T] 内**所有自发事件**，逐事件分类 + 聚合统计。
读出（per μ，聚合多 seed×多网络）：
```
event_rate (events/s)            event size 分布 (per-event 空间 extent / active mass)
event duration 分布              return_probability (回静事件占比)
R0/R1/R2/R3/R4a/R4b fraction     representative event diagnostics (median-size seed)
```
**成功标志（机制结果）**：μ↑ 时事件 **size/duration 分布右移**、return_probability 下降、出现 **R3**、（高 μ）
出现 **R4a**；`core_susceptibility`-h 比 `uniform`/`shuffled` **更早**出现 R3/R4a；`bare`（无核 ⇒ h=0）**不迁移**。
**W 在自发事件上重测**（与 kick-W、数据侧模板是否同构）= 关键 falsifiable（同一个 W 是否同时管自发+kick+不同 μ）。

**§3b 桥接**：`K=0`（no-kick）既是 spontaneous 的入口、也是 basin 的 K=0 端 → **同一量在两套读出里对齐**，
把 basin 曲线接到 spontaneous phenotype 上。

---

## 4. §4 SUPPORT — finite-pulse kick basin（**辅助机制探针，非发作定义**）

**问题**：同一 μ 下，basin 边界是否随 μ 收缩？（解释为什么自发会更容易跨过去。）
固定 source=center，扫 `K ∈ {0, 0.8, 1.0, 1.2, 1.4, 1.6}`（含 K=0 桥接），读：
```
P_return(K,μ)   P_escape(K,μ)   K_min_return(μ)   K_min_escape(μ)
```
μ↑ 时 `K_min_return`/`K_min_escape` 应下降（basin 收缩）。**kick 不是发作机制**，只是稳定性几何探针。
高 μ 下用 raw `core_kick` 回静/失控 + spontaneous flag 分类（§2），不用差分。

---

## 5. R0–R4 分类器（已实现 `src/sef_hfo_mu_basin.classify_event`，8 TDD）

R0 silent / R1 failed ignition / R2 finite returned (local) / R3 large returned (回静但变大) /
**R4a** sustained **且仍有传播前沿**（`sustained_front_score ≥ FRONT_THRESH`：持续相未全场饱和、有空间梯度/移动前沿）/
**R4b** tonic runaway（全场均匀饱和、无前沿）。**只有 R4a 可作 synthetic bridge candidate；R4b 一律不可。**
draft 阈值在 `DEFAULT_CAPS`（审计期可调），R4a/R4b 各配 TDD。

---

## 6. 第一轮：tiny pilots（**只 smoke 级，不全网格**）

### B. tiny spontaneous pilot（§3 primary）
```
substrate = n17.6   L = 20   h = core_susceptibility   no kick
μ ∈ {0, μ_mid, μ_high}   seeds = 3–5   T = 8–20 s
```
输出：event rate、size/duration 分布、return prob、R2/R3/R4a/R4b fraction、representative event diagnostics。
**目标只看 phenotype 是否随 μ 右移，不要求一次拿完整相图。**

### C. basin support（§4，同 3 个 μ 点）
```
K ∈ {0, 0.8, 1.0, 1.2, 1.4, 1.6}   source=center   seeds = 5–8
```
输出 P_return(K,μ)、P_escape(K,μ)、K_min_return(μ)、K_min_escape(μ)。高 μ core_only 自发=phenotype，用 raw+flag。

### D. 最少 h controls（只在 μ_mid/μ_high + spontaneous）
```
h ∈ {core_susceptibility, uniform, shuffled}
```
问：core-coupled μ 是否比 uniform/shuffled **更早**产生 R3/R4a？（否则 μ 只是全局加热、没门控病理场。）

> **L8 只许 smoke（验证 plumbing），不作为科学步骤**（删除 v1 的"先 L8 basin 再 L20"）。

---

## 7. 工程实现顺序（PILOT-FIRST）

1. **spontaneous-event detector**（纯函数 `src/sef_hfo_mu_basin.py` 扩展，TDD）：长 no-kick 群体活跃轨迹里
   检测事件 → 每事件 size/duration/回静/`sustained_front_score` → 聚合 rate/size 分布 + R 类占比。复用
   `classify_event` + `active_fraction_trace`。
2. **runner `--spontaneous` 模式**：no kick、长 T、core 场（含 μ）、检测+分类+落盘 per-event + 聚合。μ=0 仍 bit-parity。
3. **L8 smoke** 跑通 → **L20 tiny pilot B/C/D**（80 核并行）。
4. STATUS（§8）。**不扩 μ full grid、不 L32**——是否扩等 STATUS 后用户定。

**结果目录**：`results/topic4_sef_hfo/m3_static_mu/{spontaneous/, basin/, h_controls/, figures/+README}`。

---

## 8. STATUS（完成后必答的 6 问，给用户）

1. spontaneous μ 是否改变 event rate / size / duration？
2. 是否出现 R3？
3. 是否出现 **R4a**，还是只有 **R4b** tonic？
4. `K_min_return` / `K_min_escape` 是否随 μ 下降？
5. `core-h` 是否优于 `uniform`/`shuffled`？
6. 是否值得扩成正式 L20 μ grid？

---

## 9. 能写 / 不能写

能写：「有限幅自限事件稳定存在；局部病理核降低其招募阈值、不改其（kick 探针下、可分辨尺度的）早期形状；
据此测：静态慢 permissivity μ 升高时，网络**自发**事件是否从 returned 长到 sustained recruitment。」
不能写：「W_event 已证明方向性算子」「已复现间期—发作相变」；把 R4b 写成 bridge；把 kick basin 当发作定义。

关联：[[m3_b1_validation_recap_2026-06-24]]、[[m3_finescan_recap_2026-06-23]]、
[[project_topic4_sef_hfo_snn_heterogeneity_result]]（mean-amplitude 点火边界 ~16.3–16.9mV）、
[[project_topic4_sef_hfo_snn_stage3_plan]]（自发 template train 框架）。
