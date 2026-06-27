# M3：hub 门控的临界分支病理骨架 (hub-gated critical branching scaffold) — Design Spec

> ⚠️ **SUPERSEDED（2026-06-21）→ 已被 local-W 版取代为承重主线，本版降级为「已记录退路 fallback」。**
> 新主线：`docs/superpowers/specs/2026-06-21-sef-hfo-m3-local-w-propagation-operator-design.md`
> （承重机制从「空间墙：走廊 + 关着的枢纽」换成「同一条局部各向异性传播场 W + 慢易感度 μ 把它从亚临界推到超临界」；
> 自限从「空间沿轴短」重定义为「时间上回到静息 / 不持续招募」）。
> **取代原因（实测，非口味）**：本版 hub 在 worktree `.worktrees/topic4-m3` 跑过第一轮——空间拉开距离不 gate（间隔 0→0.4 漏只降约 6%）、
> 各向同性对照漏大降约 48%（沿轴各向异性自己把活动桥到外围）、度归一化门一开正反双向塌成单向。见
> `docs/archive/topic4/sef_hfo/m3_hub_scaffold_infra_status_2026-06-19.md`。
> **本版仍是合法退路**：当 local-W 给不出「亚临界 + 可读 + 自停」窗、或拿不到稳定双向模板时启用（新 spec §9 + §10）。
> 本版已跑出的基础设施（degnorm / σ 探针 / Layer-2 等价 / hub-diag / runner 接线）大量被 local-W 主线复用。
>
> 状态：DESIGN（待用户 review → writing-plans 出实现计划）。日期 2026-06-19。
> 上游：M2 收口 `docs/archive/topic4/sef_hfo/m2_stage_recap_2026-06-19.md`；Task 0 数据审计
> `results/topic4_sef_hfo/event_extent_audit/cohort_summary.json`（INCONCLUSIVE，AF=0.915，LR=0.561，
> contact-space sampling-dominated）；联合判据 `snn_5criteria_joint_verdict_2026-06-18.md` §7。
> 框架合同：`docs/topic4_sef_hfo.md`（工作点 lock 2026-06-03、§7.2 红线、tier 纪律）。

---

## 0. 一句话承诺（朴素表述，CLAUDE.md §8）

**要做什么**：让一个一个细胞放电的网络自己长出"间期 HFO 群体事件"，并同时满足五项——① 自发冒出来；
② 时间上自己停；③ 空间上自己停；④ 能沿一条固定路线传；⑤ 正反双向——而且能在一个慢变量改变时，
从"间期的局部事件"切换到"发作样的广泛传播"。

**怎么做**：上一阶段（M2）的思路是"在沿传播轴的方向上，让前方抑制先升起来把波刹停"。放电网络里这招失效
（波是全或无的，点着就沿轴传到组织边缘）；而真实数据的电极读出里事件足迹是**沿轴长的**（采到的范围里沿轴铺满约
92%、且跟随机按电极杆抽样几乎没差别）——这是**采样主导的 contact-space 事实**：它不支持把模型读出做成"沿轴一小段"，
但**组织层面到底有没有轴向自限，它判不了**。所以 M3 换承重机制：给病理通路一条**有限长的走廊**，
走廊尽头放一个**高门槛的广播枢纽**。平时枢纽关着，事件能在走廊里自发冒出来、沿走廊传、时间上自己停、
两头都能起，但**过不了枢纽**，所以不会烧到更大范围——这就是新的"空间上自己停"：不是沿轴短，而是"到枢纽为止、
不外溢成发作"。当慢变量（比如抑制储备下降）把枢纽门槛压低，同一条走廊就能跨过枢纽，枢纽的长程输出把活动广播到
更大范围——这就是间期→发作的相变。"枢纽难被点着、但一旦点着输出很广"用"**连得越多、门槛越高**"来实现。

**揭示什么（预期判据）**：只有**完整模型**（走廊 + 长程边 + 高门槛枢纽 + 度归一化门槛 + 慢变量）同时满足五项、
并且能相变，**且**模型的虚拟电极读出跟真实数据"沿轴长 + 像随机抽样"统计上对得上——才算这个机制看起来成立；
拆掉其中任何一块就垮，才说明不是单纯调参凑出来的。

> （内部归档代号：M3 = hub-gated critical branching scaffold；上游 M2 = dynamic inhibitory gate（失败）+ M1 = E→E recovery；
> Task 0 contact-space AF/LR two-layer calibration；criticality = recruitment-operator branching ratio σ，**非** resting-state max Re λ。）

---

## 1. 背景与动机（为什么是 M3，而不是再调 M2）

M2 测的是"前沿抑制刹车 + 兴奋疲劳能不能在放电网络里复现沿轴空间自限"。结论（`m2_stage_recap_2026-06-19.md`）：
平均率场模型里刹车有效（波传一小段就衰减熄灭、尺度固定不随系统变），但搬到 cm-scale 放电薄片里**无效**——
全或无波前一旦点着就沿轴传到边界，加的抑制只能压死点火、或把波垂直方向压窄，**挡不住沿轴推进**。
放电网络自然给的空间自限是**垂直于传播轴的**（贴杆窄波前），不是沿轴的。

Task 0（数据侧审计，n=23）独立地从另一头夹击了"沿轴一小段"这个目标：真实间期事件在采到的范围里
**沿轴铺满约 92%**（AF_cohort_median=0.915；23 人里只有 1 人像一小段），而且整个足迹（沿轴 + 侧向）
**≈ 随机按电极杆抽样的零分布**（沿轴 Δ Wilcoxon p=0.056 不显著，侧向 CI 跨过 0）。这是 **contact-space**
的事实——足迹主要由电极采样决定，既不能证明也不能否定**组织层面**的自限。它的作用是**校准模型**：
模型的虚拟电极读出也必须"沿轴长 + 像随机抽样"，**短一段反而算不对**（详见 §6 验收 layer 2）。

**两头收口的结论（非对称，别写成"两边都证否了轴向自限"）**：把空间自限的承重点放在"沿轴刹停 / 让模型读出变成一小段"上，
一头被**放电动力学硬顶回来**（M2 失败，确定），另一头被**数据读出约束住**（contact-space 足迹是沿轴长的 → 模型读出不能短一段；
但组织层面是否轴向自限本身判不了，是 constraint 不是 negative）。M3 把承重点换成"**有限走廊 + 关着的枢纽**"，同时满足这两侧——
事件在走廊里铺满（沿轴长，对得上数据读出）但跨不过枢纽（不外溢成发作）。这把顶层设计（间期/发作共享病理通路、慢变量触发相变）
落成一个具体可证伪的网络机制。

---

## 2. 设计决策（用户 2026-06-19 拍板的 3 项 + 推导出的第 4 项）

| # | 决策 | 选择 | 含义 |
|---|---|---|---|
| D1 | 起步方式 | **直接上最小 hub-SNN**（跳过中间 toy） | 不再做"先 toy 后 SNN"；避免 M2 那种"率场成了但放电层不迁移"的二次踩坑。代价：没有便宜的连续相图 → 由 D4 的结构探针补偿。 |
| D2 | Hub 实现 | **单图度归一化门槛** | 在一张异质 cm-薄片里嵌入 hub 节点 + 度归一化门槛 θ_i=θ0+α·k_i^out + 稀疏长程边；不走"两片+桥"的离散 caricature。更贴 Nature 谱半径框架，引擎手术更大。 |
| D3 | 与 M2 关系 | **两条线并行** | M2 faithful test（shunting + ahead-recruit, Tasks 1–7）与 M3 同时推进，各拿独立 verdict。需 worktree 隔离 + 不交叉依赖（§8 风险）。 |
| D4 | 便宜相图替代（D1 的推论） | **结构临界探针** | D1 拿掉了 toy 这层便宜动力学相图 → 用"从连接矩阵直接算招募/传播算子的分支比 σ"这个纯线代探针补回来（§5.3）。σ 是诊断/工作点选择器，**不**作闸门；SNN 两层验收才是闸门。 |

---

## 3. 五个边界约束（claim discipline — 写进 Global Constraints，违反 = 静默科学污染）

**C1. 「临界」只能指传播/分支算子层，不能指静息网络线性临界。** 项目 2026-06-03 已把真 LIF 工作点锁成
**"稳健稳定但可激"**（自洽点 max Re λ≈−0.05、loop gain≈0.58、k=0），并**明确更正过一次**——"近临界工作窗"
的措辞是错的，静息网**不**处在线性失稳边缘（见框架 banner + `lif_rate_field_theory_2026-06-03.md`）。
M3 的"临界"对象**必须**是**事件跨过枢纽的招募概率 / 分支比 σ / 招募算子谱半径接近 1**，绝不能写成
"把整张静息网调到 max Re λ≈0"。色散/λ(k) 这类线性诊断同框架一样**作诊断、不作闸门**。

**C2. 不引入平滑率场作主力。** M2 的硬教训是平滑率场里成立的东西不迁移到放电层。D1（直接上 SNN）正是为了
honoring 这条：M3 主力是放电网络；§5.3 的结构探针是**连接矩阵的线性代数**（全或无招募的线性化筛查），
**不是**平滑率场动力学。若日后想加率场 mean-field，只能作旁证，不能作 SNN 结论的替身。

**C3. 空间自限的承重机制换人，措辞从"内禀自限"改"结构封闭"，两层验收照搬并收紧 M2。** 承重机制：旧 = 前沿抑制刹停（放电层失败）→
新 = **有限走廊 + 关着的高门槛枢纽**。**措辞锁（审阅 §4）**：事件因"走廊有限长 + 枢纽关着"而停 = **结构封闭 / 闭门约束
（structural containment / closed-gate confinement）**，**不是内禀传播自限（intrinsic self-limitation）**——论文/报告语言只准用前者。
验收两层（§6）：layer 1（组织/全场）事件空间有界、随薄片放大不长大、不贴边、不强直、死在枢纽前（不招募全局区，**数值判据见 §6.5**）；
layer 2（虚拟电极）沿轴/侧向足迹在 **subject 层面与 Task 0 真实分布等价**（预注册容差，**非"不被拒"**，见 §6.2）。
**绝不能把目标写回"沿轴收到一小段"**——那是 layer 2 失败。

**C4. 间期→发作相变只能当"合成可行性桥"。** 框架 §7.2 红线：ictal-like recruitment 只作 synthetic feasibility
bridge，**不解释临床发作起始**。M3 的相变是"同一走廊在慢变量改变后能否从局部跨成广泛"的**动力学可行性**演示，
不是临床因果。报告里禁止出现"这解释了病人发作怎么起来"。

**C5. 慢变量复用现成层，hub 是真·新引擎。** 引擎已有慢变量层 `src/snn_engine/slow_vars.py`
（去抑制 z~5s、自适应阈值 φ~100ms、钾 sAHP g_K~5s，默认关，参数为占位值需标定）——M3 那个"压低枢纽门槛"
的慢变量**复用这层**（首选 φ 抬降阈值 / z 减弱抑制），不另起炉灶。hub/长程边是**真新手术**：现有连接
（`connectivity_rot.py`）是局部 + 各向异性距离核，无任何长程或 hub 概念；M2 加的是宽 I→E veto，
hub 要加的是**稀疏长程 E→E（broadcast 边）+ 度归一化高门槛**。

---

## 4. 架构总览

```
                 ┌─────────────────── cm-scale 放电薄片 (single graph) ───────────────────┐
                 │                                                                          │
   触发盆 A ───►  │   ████ 病理走廊 (anisotropic 高连接轴, 核 cell 低阈值) ████  ───►  触发盆 B │
                 │                              │                                           │
                 │                          [ HUB ]  ← 度归一化高门槛 θ_i=θ0+α·k_i^out      │
                 │                              │  长程 E→E broadcast 边 (稀疏)              │
                 │                              ▼                                           │
                 │                    ░░░ 全局区 (走廊外的更大薄片) ░░░                       │
                 └──────────────────────────────────────────────────────────────────────────┘
   间期: 慢变量低 → θ_hub 高 → 事件填满走廊、死在 HUB 前、不进全局区  (① ②③④⑤ 全在走廊内成立)
   发作: 慢变量高 → θ_hub 降 → 事件跨过 HUB → 长程边广播到全局区   (synthetic feasibility bridge, C4)
```

四个组成单元，每个有单一职责、可独立测试、可独立 ablate：

- **U1 病理走廊基底**（substrate）：各向异性高连接轴 + 两端触发盆 + 走廊外的全局薄片。决定 ④ 可传播、⑤ 双向。
- **U2 Hub 机制**（3 块默认关、逐比特一致的手术）：长程 E→E 边 + 度归一化门槛 + 慢变量门控。决定 ③ 空间自限 + 相变。
- **U3 结构临界探针**（纯线代，无动力学）：从 U1+U2 的连接矩阵算 σ_corridor / σ_crossing 的相图 → 选工作点。补 D1 拿掉的便宜相图。
- **U4 虚拟 SEEG 读出层**（复用现成 pipeline）：把放电事件读成 `*_lagPat_withFreqCent.npz` → 真实 propagation pipeline → layer 2 验收。

---

## 5. 各单元详细设计

### 5.1 U1 — 病理走廊基底 (substrate)

现有 lesion 模式：`twoend_equal`（两个等价灶，给双向）、`extended_patch`（单个大灶，stage4 走廊雏形）。
M3 需要一个**同时**有"走廊（各向异性轴）+ 两端触发盆（给 ⑤ 双向）+ 走廊外全局区（给相变的去处）"的基底。

- **走廊**：复用 `extended_patch` 的核 cell（低 V_th 病理斑）+ `connectivity_rot` 的各向异性（`theta_EE`, `AR`）
  让连接沿轴增强 → 事件沿轴传（④）。走廊**有限长**（核斑只覆盖薄片的一段，不到全片）。
- **两端触发盆**：走廊两端各放一个可自发点火的小核（类似 `twoend_equal` 的两灶），事件从任一端起、向另一端传
  → 正反双向（⑤）。这是 stage3 `twoend_equal` 与 stage4 `extended_patch` 的合并体，记为新 lesion 模式
  `corridor_twoend`。
- **全局区**：走廊核斑之外的薄片（普通 V_th，非病理）。间期事件**不应**招募它；发作时经 hub 长程边招募它。
- **Hub 位置**：走廊轴末端（靠近全局区的过渡带）的少数 E cell。`hub_select ∈ {axis_end, top_outdeg}`
  （首选 `axis_end` = 几何指定在轴端；`top_outdeg` = 按建好的局部出度选，作 sensitivity）。
  **必须 deterministic（审阅 §5）**：`axis_end` 给定基底/seed 唯一确定，**不得引入 seed-dependent hub 位置**；`top_outdeg` 给定建好的 net 也唯一确定。

> **U1.0 substrate feasibility gate（独立闸门，审阅 P1）**：在"hub 关、慢变量低"的纯走廊态，`corridor_twoend` 必须先**自己**
> 产生**足够、稳定、双向可读**的事件，才允许进入 hub 机制判定。⚠️ ⑤双向 + 刻板在旧 cm-SNN 里是 **partial / unstable**
> （M2 recap §5：两灶能给正/反，但多次事件路线越攒越散），**不是已解决事实**——这是要重新挣回的 gate，**不能假设**。
> **判据**：自发离散事件率 > 阈值（非 silent / 非 tonic）、fwd 与 rev 各 ≥ N_min 个干净可读事件、跨事件路线稳定度 > 阈值。
> **失败 = 停在"substrate fail"**，记录报告，**不得把基底失败算到 hub 机制头上**（否则混淆 substrate 与 hub 的责任）。
> 通过后才叠 U2，这样 U2 的"成功/失败"才分得清是 hub 机制还是基底本身（CLAUDE.md §7 单元纪律）。

### 5.2 U2 — Hub 机制（3 块手术，默认关 = 逐比特一致）

**手术 1：稀疏长程 E→E broadcast 边（`connectivity_rot.py`，需 re-bless）。**
- 新增参数：`hub_n=0`（hub cell 数）、`hub_select='axis_end'`、`hub_long_range_C=0`（每个 hub 的长程出边数）、
  `l_hub_long=None`（长程距离尺度，≫ 局部核宽）、`hub_gain=0.0`（长程边权重 = `hub_gain * w_EE`）。
- 实现：对选中的 hub E cell，额外采 `hub_long_range_C` 个**远处**（`l_hub_long` 尺度）E 目标，权重 `hub_gain*w_EE`，
  写进 `ampa_by_delay`（E→E）。镜像 M2 gate 的实现骨架（M2 加的是宽 I→E；这里是稀疏长程 E→E）。
- **默认 `hub_n=0` 或 `hub_gain=0` → 不采新边 → 无新 rng draw → spike SHA 不变（bit-parity）。** 引擎 guard
  re-bless 前必须先过 parity 测试。

**手术 2：度归一化门槛（runner 预变换，零 `simulate_kick` 改动）。**
- 关键工程点：`simulate_kick` 已经吃一个 per-neuron 的 `V_th_per_neuron` 向量。度归一化门槛 = 从建好的 net 算每个 E cell
  的出度 `k_i^out`（含长程边），在调 `simulate_kick` **之前**把 `V_th_per_neuron[i] += degnorm_alpha * k_i^out`。
- 新增参数：`degnorm_alpha=0.0`（默认 0 → 阈值不变 → 无副作用）；`degnorm_use ∈ {out_strength, in_strength, hybrid}`。
- **三方案预注册并列、不设单一 primary、不许看结果挑（审阅 P1）**：`out_strength`（θ∝输出强度 = "广播者难被点着"，护 hub 源端）；
  `in_strength`（θ∝输入强度 = homeostatic / input-normalized，更对应 Nature 那篇"减均值=全局抑制反馈"，且**护住接收 hub 长程输入的
  全局区 cell**、防其过早被招募）；`hybrid`（两者 max 或加权）。**out 与 in 护的是不同环节**（out 护 hub 难招募、in 护下游不早燃），
  **只用 out 单口径有让全局区过早招募的风险** → 主口径在 M3.0/M3.5 **预注册后再定，结果出来不许改**。
- **作用范围 = 全网 E cell（全局 homeostatic 归一化），不是 hub 专属**：每个 E cell 按自身的（out/in/hybrid）连接量抬高阈值；
  hub 因连接量最高**自然**收到最大抬升——这正是"hub 成为高门槛门"的来源（机制驱动，非手工指定）。
- 这是纯连接→阈值的 readout 变换，**不碰动力学、不碰 rng** → 天然 bit-parity，且让"连得越多门槛越高"显式成立。

**手术 3：慢变量门控相变（复用 `slow_vars.py`，C5）。**
- 第一版 = **两个静态条件**：`hub_theta_interictal`（高，间期）vs `hub_theta_ictal`（低，发作）——只作用在 hub cell 的
  V_th 上。先证明"高门槛 → 走廊内自限 / 低门槛 → 跨枢纽广播"，**再**把它接成慢变量。
- **与手术 2 的复合（写清，避免歧义）**：hub cell 的间期阈值 = θ0 + 手术 2 的度归一化抬升（高基线，门关）；
  手术 3 的慢变量在此基线上**向下叠一个 permissivity 增量**（只对 hub cell），把有效 θ_hub 从度归一化高基线压到
  `hub_theta_ictal`（门开）。即 `θ_hub_eff = θ0 + α·g_deg(hub) − (permissivity 慢项)`（`g_deg` = 预注册的 out/in/hybrid 度量），对应顾问的 θ_i=θ0+α·k − β·z。
- 第二版 = 接 `slow_vars` 的 φ（自适应阈值，~100ms）或 z（去抑制，~5s）作用于 hub cell，让 θ_hub 随慢状态平滑下降。
  `slow_vars` 参数是占位值，必须先标定（§8 风险）。慢变量**只调 permissivity（θ_hub / 长程增益），不得改走廊轴、
  模板几何、hub 位置**——否则不再是"同一病理通路的相变"（C4 + framework H4 不变形几何）。

**默认关 parity 测试粒度（审阅 §5）**：`hub_n=0`、`hub_gain=0`、`degnorm_alpha=0` 三个 toggle **分开各测一遍** spike SHA bit-parity
（不是只测"全关"），确保每块手术独立默认无副作用。

### 5.3 U3 — 结构临界探针（cheap phase map，替代 toy）

**目的**：D1 跳过 toy 后，需要一个便宜的"hub 门槛 × 长程增益 → 区域"地图来选 SNN 工作点，且必须在
**招募/传播算子层**（C1），不能是静息线性谱、不能是平滑率场（C2）。

**做法**（纯线代，建好网络后一次性算，无动力学）：
- 把全或无招募线性化成一个**招募算子** `M`：`M[j,i] ≈ P(活跃的 i 把 j 顶过阈值)`，由 E→E 权重 `W[j,i]` 与
  `(V_th[j] − 静息驱动)` 的间隙单调映射得到（screening 用的单调 link，非动力学拟合）。
- **分支比** σ = `M` 的最大特征值（或受限子集的平均行和）。
  - `σ_corridor` = 限制在走廊节点上的 σ：目标区要 **≳1**（事件能沿轴**可再生**传播，④）；事件停下来是因为**走廊有限长 + 枢纽关着 = 结构封闭**，**不是内禀自限**（措辞锁见 C3）。
  - `σ_crossing` = 走廊→hub→全局区这条跨越路径的 σ：**间期要 <1**（死在枢纽）、**发作要 >1**（广播）。
- **相图**：σ_crossing 作为 (`degnorm_alpha`/`θ_hub`, `hub_gain`) 的函数 → 等高线 σ_crossing=1 就是间期/发作的相变边界。
  选 SNN 工作点：间期候选 = σ_corridor≳1 且 σ_crossing<1；发作候选 = 同一点把 θ_hub 压到 σ_crossing>1。

**地位（写死）**：探针是**诊断 / 工作点选择器**，**不是闸门**——同框架"λ(k) 作诊断、finite-pulse 才是闸门"。
SNN 的两层验收（§6）才判 pass/fail。探针只负责"别在 SNN 上盲扫"。

**接口**：`src/topic4_hub_criticality.py`：`recruitment_operator(net, V_th, NE) -> sparse M`；
`branching_ratio(M, idx_subset) -> float`；`crossing_branching(M, corridor_idx, hub_idx, global_idx) -> float`；
`sigma_phase_map(net_builder, grid) -> dict`。纯函数 + TDD（小网络解析校验：链状网 σ=行和、断开 hub → σ_crossing=0）。

### 5.4 U4 — 虚拟 SEEG 读出层（复用现成 pipeline）

复用 `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py`（虚拟电极 montage → `*_lagPat_withFreqCent.npz`）
+ `scripts/run_model_contact_plane_readout.py::build_model_record`。模型事件经**真实** propagation pipeline
（masked lagPat，禁用旧 unmasked loader）读回方向/端点，再喂 §6 layer 2 的 Task 0 纯指标。
**禁止只看 raster 下结论**——所有 synthetic 结论必须经真实读出 pipeline。

---

## 6. 五项 + 相变的机制映射 + 验收

### 6.1 机制映射

| 要求 | M3 靠什么 | 验收锚点 |
|---|---|---|
| ① 自发 | 近临界**亚临界**走廊核 + 噪声（σ_corridor≳1 但有限长） | quiet rest + 自发离散事件（非 tonic/非 silent） |
| ② 时间自限 | 兴奋疲劳（保留 M1 E→E recovery `ee_std_u`） | 事件有限时长，事件间安静基线 |
| ③ 空间自限 | **有限走廊 + 关着的高门槛 hub**（U2，新承重） | layer 1：有界 + L 不变 + edge_margin>0 + 死在 hub 前（不招募全局区） |
| ④ 可传播 | 病理长轴各向异性连接（U1） | 沿轴清晰传播，fwd/rev 可读出 |
| ⑤ 刻板 + 双向 | 固定走廊/hub 拓扑 + 两端触发盆 | 双向 fwd/rev 干净；多次事件路线稳定（不越攒越散） |
| 相变 | 慢变量压低 θ_hub / 抬高长程增益（U2 手术 3，C4 合成桥） | σ_crossing 越过 1 + SNN 确认从"走廊内自限"切到"跨 hub 广播" |

### 6.2 两层验收（layer 1 框架照搬 M2；layer 2 **收紧** M2 faithful plan §6 Step 3b 的 event-level "不被拒=PASS" 误判，C3）

- **Layer 1（组织 / 全场自限）**：**三层都报，不得只看可读事件（审阅 §5）**——(a) 全部组织事件、(b) 可读子集 `n_part≥7`、
  (c) 局部不可读子集；只筛可读子集会把失败事件（runaway / 贴杆窄）藏掉。每事件量真实足迹 `reach_axis_mm`/`r95_mm`/`edge_margin_mm`。
  **Layer 1 PASS = 事件 FINITE**：足迹有界、L20≈L32（**L 不变**）、`edge_margin_mm>0`（不贴边）、事件间安静基线（非 tonic）、
  **死在枢纽前**（hub/global 数值判据见 §6.5）。**目标是 FINITE + 有界 + 结构封闭，不是"沿轴短"。**
- **Layer 2（虚拟电极足迹 ≈ Task 0，subject-level 等价检验）**：**不用 event-level KS/MW 的"不被拒"当 PASS（审阅 P0）**——
  Task 0 reference 是 9145 event 嵌套在 23 subject 内（**非独立**），且"不被拒"是低功效/分辨不开、不是匹配。**改 subject-level /
  hierarchical bootstrap**：每个模型"被试"（一个网络实现 / seed）算 subject-level median AF、median LR、obs−null gap、IQR，
  要求**都落在真实 subject 分布的预注册容差带内**（等价检验 / TOST 思路，**PASS = 在容差内，非 p>α**）。容差带由 Task 0
  真实 subject 间分布（per_subject median 的分位距）预注册。event-level KS/MW + overlap/effect-size **只作辅助描述**，不作 PASS。
  **短一段（subject-median AF≪0.9）= layer 2 FAIL。**

### 6.3 相变 demo（synthetic feasibility bridge，C4）

同一基底、同一走廊、同一 hub 位置，只动慢变量（θ_hub / 长程增益）：
- 间期工作点：事件填满走廊、死在 hub 前、全局区静默；
- 发作工作点：事件跨过 hub、长程边把放电广播到全局区（广泛招募，远超走廊）；
- 过渡：σ_crossing 越过 1 的边界由结构探针预测、由 SNN 确认。
口径锁：这是"同一病理通路能否相变"的**可行性**演示，**不**主张临床发作起始机制（C4）。

### 6.4 消融（证明机制成立、非调参）

只有完整模型同时满足五项 + 相变；逐一拆下列任一块，预期某项垮：

| 消融 | 操作 | 预期失败 |
|---|---|---|
| 去 hub 高门槛 | `degnorm_alpha=0`（hub 不再高阈） | 事件过早跨 hub → ③ 垮（铺满/runaway） |
| 去 hub 输出 | `hub_gain=0`（hub 无长程边） | hub 成死屏障 → 相变垮（发作态无广播） |
| 去长程边 | 不加长程 E→E（仅局部+轴） | 有刻板局部传播但发作态无广泛招募 → 相变垮 |
| 去度归一化 | 全网同阈（非 θ∝k） | 高出度节点因输入多过早招募 → 铺满/压死二分 |
| 固定慢变量 | θ_hub 冻结 | 有间期事件但无相变 |
| 打乱 hub 位置 | hub 随机重置 | ⑤ 刻板退化 / σ_crossing 失稳 |

### 6.5 机制诊断（P1，配合 layer 1/2，避免"压死也算成功"）

- **hub 是中继不是首点火**：hub cell 的首发时间应**晚于**走廊起点、**早于**全局区招募（onset→hub→wider 的时序），
  不是最早点火点（对标顾问报告检验 2）。
- **σ_crossing 结构预测 vs SNN 实测**：探针预测的相变边界与 SNN 实际跨越点是否吻合（吻合 = 机制可解释；
  不吻合 = 探针线性化失真，需诚实标注，SNN 为准）。
- **是哪些 cell 跨过 hub**：跨越事件应走 hub + 长程边（结构通路），不是全局抬升。
- **hub/global recruitment 数值判据（审阅 P1，替代"死在 hub 前"口语，喂间期 gate + 发作 bridge）**：定义
  `hub_recruited_fraction`（hub cell 被招募比例）、`global_E_spike_fraction`（全局区 E cell 放电比例）、
  `global_first_spike_after_hub_ms`（全局区首发相对 hub 首发的延迟）。**间期 gate**（Layer 1）要求 `hub_recruited_fraction`
  与 `global_E_spike_fraction` 都 < 预注册阈值（事件死在枢纽前、全局区 ≈ 本底）。**发作 bridge**（§6.3）要求 onset→hub→global
  时序成立：hub 先被招募、`global_first_spike_after_hub_ms > 0`（全局区在 hub 之后才燃）、`global_E_spike_fraction` 显著抬升。

---

## 7. 分阶段交付（pilot-first，每阶段有 go/no-go）

- **M3.0 基底 + 结构探针 + 预注册**（便宜，无动力学网格）：建 `corridor_twoend` 基底 + `src/topic4_hub_criticality.py`，
  算 σ_corridor/σ_crossing 相图 → 选间期 & 发作候选工作点。**此阶段（出任何 SNN 结果之前）预注册并落盘，结果出来不许改**：
  度归一化三方案（out/in/hybrid）的比较 + 主口径选取规则（审阅 P1）、Layer 2 的 subject-level 容差带（由 Task 0 真实 subject
  分布 per_subject median 分位距导出）、hub/global 数值阈值（§6.5）。**Gate**：相图里存在"σ_corridor≳1 且 σ_crossing<1"的区，
  且其邻域有 σ_crossing>1 的发作区。无此区 → 停，报告（拓扑不支持门控）。
- **M3.1 引擎手术**（默认关 bit-parity）：长程边（`connectivity_rot`）+ 度归一化阈值变换（runner）+ TDD + re-bless。
  **Gate**：默认关 spike SHA 不变（`hub_n=0`/`hub_gain=0`/`degnorm_alpha=0` 三 toggle 分开测）；开启后 σ/出度按预期变化。
- **M3.1b U1.0 substrate feasibility gate**（bare corridor，hub 全关 + degnorm 关）：跑纯走廊 `corridor_twoend` SNN，
  确认 §5.1 U1.0 判据成立（足够 + 稳定 + 双向可读，非 silent/tonic）。**Gate**：U1.0 通过才进 M3.2；**失败 = "substrate fail"**
  停报告，**不把基底失败算到 hub**（审阅 P1）。
- **M3.2 工作点 sanity**（仿 M2 Task 5）：度归一化 + hub 开后，基底仍自发点火、有安静 rest。
  **Gate**：若 degnorm 压死点火 → 在该层 re-tune drive（不在 hub pilot 里调）；若无 (drive, degnorm) 给离散事件 → 停报告。
- **M3.3 间期 pilot**（单工作点）：hub 关（θ_hub 高）→ 事件填满走廊、死在 hub 前、双向、有界。跑 layer 1 + layer 2。
  **Gate**：layer 1 有界 + 不招募全局区 + 双向保留，才进 M3.4。
- **M3.4 相变 pilot**：同工作点压低 θ_hub → 跨 hub → 广播。σ_crossing 越过 1 由 SNN 确认。
  **Gate**：相变可复现（间期/发作判别清晰）才进网格。
- **M3.5 网格 + 消融**（pilot 全过后才跑）：(θ_hub × hub_gain × degnorm_alpha) 网格 + §6.4 六个消融 + L 不变性（L20/L32）。
- **M3.6 verdict + archive**：§8 白话 abstract + 两层结果 + 相变 + 消融表，写 `docs/archive/topic4/sef_hfo/m3_*_result_<date>.md`，
  更新 `m2_stage_recap` 续篇 + memory。

---

## 8. 风险与控制

- **R1 复合高成本路径**（D1 跳 toy + D2 单图度归一化 + D3 并行）= 最高手术 + 无便宜动力学缓冲。
  控制：U3 结构探针补便宜相图；**铁律 pilot-first**（M3.0/3.2/3.3/3.4 四道 gate）；分块默认关 bit-parity 手术。
- **R2 度归一化改 E/I 平衡**：抬高均值阈值可能压低自发率。控制：M3.2 工作点 sanity 必须在叠 hub 前 re-verify
  自发性 + 安静 rest（同 M2 Task 5 4-cell sanity 区分"真自限"vs"压死点火/转 tonic"）。
- **R3 并行 worktree 共享引擎 guard**（D3）：M2 faithful（shunting/ahead-recruit）与 M3（hub）都改
  `kick_probe.py`/`connectivity_rot.py` 且各自 re-bless `engine_versions.json`，**在不同 worktree**。
  控制：**M3 用独立 worktree，从当前态分支，只加 hub 机制；不依赖 M2 在飞的 shunting 改动**；两树各 re-bless 自己的
  `engine_versions.json`；合并/reconcile 是后续单独步骤（用户决定），不在本 spec。M2 的 shunting/ahead-recruit 作为
  M3 的"侧向约束可选旋钮"是**未来可选 merge**，非依赖。
- **R4 单图度归一化可解释性差**（D2 的代价）：控制：三块手术独立 toggle（`hub_gain`/`degnorm_alpha`/长程边分开）
  + §6.5 机制诊断 + U3 σ 相图给可读的相变边界。
- **R5 慢变量参数是占位值**（C5）：`slow_vars` 的 τ/阈值未标定。控制：第一版用两静态条件绕开标定；接慢变量前先标定
  时间尺度（事件间隔量级），且只调 permissivity 不改几何。

---

## 9. 与 M2 faithful test 的并行协调（D3）

- 两条线各自独立 worktree、独立 verdict。M3 **不**消费 M2 在飞的 shunting/ahead-recruit 引擎改动。
- M2 faithful 若得 "STILL FAILS（full-field 仍不有界）" → 它 earns "前沿抑制机制不足以 bound full field" 的结论，
  与 M3 "换 hub 承重机制" 互补、不矛盾（两者本就分工：M2 测前沿抑制能否沿轴 bound；M3 测 hub 门控能否）。
- M2 的宽 I→E veto + 电导分流，作为 M3 的**侧向约束可选项**（M3 主力是 hub 管轴向/全局，侧向窄本来 SNN 自带），
  是 M3.5 之后的可选 sensitivity，不进 M3 主链。

---

## 10. Out-of-scope（红线，写死）

- ictal recruitment = **synthetic feasibility bridge**，不解释临床发作起始（C4 / framework §7.2）。
- 「临界」= 招募/传播算子分支比 σ≈1，**非** resting-state 线性 near-criticality（C1 / 工作点 lock）。
- **不**主张模型 hub = 病人真实解剖 hub。"候选 hub 在数据里是否少招募-但招募即大事件"（顾问报告 §13 数据侧检验）
  是**独立的、可选的**后续验证，**不在本 spec**（避免 scope 膨胀；列为 future）。
- 不引入平滑率场作 SNN 结论的替身（C2）。

---

## 11. 溯源（内部代号 / 路径）

- 上游失败：M2 dynamic inhibitory gate（`src/topic4_dynamic_inhibition_gate.py` rate；cm-SNN gate `connectivity_rot.build_connectivity_rot`
  `gate_scale/l_gate/c_gate`），recap `m2_stage_recap_2026-06-19.md`，rate 成 / SNN 不成。
- 数据校准：Task 0 `src/topic4_event_extent_audit.py` + `scripts/run_topic4_event_extent_audit.py`，
  `results/topic4_sef_hfo/event_extent_audit/cohort_summary.json`（AF=0.915, LR=0.561, INCONCLUSIVE, two-layer reference_distribution）。
- 引擎：`src/snn_engine/{connectivity_rot,kick_probe,params,slow_vars}.py`；guard `engine_versions.json`；
  lesion `twoend_equal`(双向)/`extended_patch`(走廊) → 新 `corridor_twoend`；M1 recovery `--ee-std-u`。
- 新增（本 spec）：`src/topic4_hub_criticality.py`（σ 探针）；`connectivity_rot` 长程 E→E 边（`hub_n/hub_gain/l_hub_long/hub_long_range_C`）；
  runner 度归一化阈值变换（`degnorm_alpha/degnorm_use`）+ hub θ 门控（`hub_theta_interictal/ictal` → slow_vars φ/z）。
- 框架合同：`docs/topic4_sef_hfo.md`（工作点 lock 2026-06-03、§7.2 红线、H4 几何不变形、tier 纪律）；
  联合判据 `snn_5criteria_joint_verdict_2026-06-18.md` §7。
- 文献 framing：Pachitariu/Stringer 2026 Nature（critically-normalized 连接 + 稀疏长程 → 宏观模式，谱半径≈1，
  仅作建模语言）；Moosavi & Truccolo 2023 PLOS Comp Biol（seizure spread 相图：excitability × global connectivity 控制）；
  Proix/Jirsa 2018 Nat Commun（局部 + 异质长程 + 多时间尺度 spread）；epilepsy pathological hub（PMC12646151，
  hub 可促可抑，支持"平时 barrier、跨过后 broadcaster"）。
```
