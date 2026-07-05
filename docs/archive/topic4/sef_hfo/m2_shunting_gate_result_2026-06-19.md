# M2 faithful test — conductance shunting + ahead-of-front recruitment gate (result)

> 2026-06-19. Worktree `.worktrees/topic4-m1` (branch topic4-snn-m1-recovery). Engine commits
> 14da64c / 00f1bbd / faa496c / 96a416f (all bit-parity-gated, SHA da5fc18c). Outputs:
> `results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/{m2_shunt_opsanity, m2_shunt_gate_pilot}/`.

## 朴素话摘要（测了什么 / 怎么测的 / 揭示了什么）

**测了什么。** 我们给一张兴奋-抑制神经元网（虚拟皮层片）装了一套"刹车"，想让它产生的间期放电事件**在组织空间里收住成一个有限的、不铺满整张片子的事件，同时事件还得活着**——有离散的起止、双向（既能从一端传向另一端、也能反过来）、能读出传播方向。刹车有两部分：一是把抑制从"减一个电流"改成"把细胞电压钳向静息"（强驱动也能压住）；二是让传播的前沿**提前点亮它前方**的抑制细胞（赶在波到达之前先布防）。

**怎么测的。** 先扫"分流强度"找一个能用的工作点——要求是"分流打开后网络还能自发产生离散的双向事件，事件之间还安静"。结果窗口很窄：分流太弱（≈0）网络跑飞成持续发放，太强（≥0.5）直接压死，只有中间一档（强度 0.25）勉强有离散双向事件，但事件之间的基线已经偏吵。然后在这个点上把"提前招募门"从弱到强（0.1→0.6）加上去，看事件会不会变成"有限但还活着"。同时直接量两件事的机制证据：抑制是不是**真在前沿之前**发放；分流是不是**真把轴向高驱动细胞钳到发放阈以下**。

**揭示了什么。** **不行——这套刹车在我们测的工作点里复现不出"有限+可读+双向"的事件。** 门砍的是事件**多频繁发放（率）**，不是事件**传多远（空间范围）**：门强度从 0 加到 0.6，事件峰值（率）单调降了 50 倍，但发放出来的事件最大轴向跨度始终维持 ~20–26mm（≈片子对角）——**发放的事件仍然铺满整张片子**。把读出加细（18 触点）解出 3 正向/1 反向可读双向事件，它们 reach 24.9mm、贴边，**还是铺满，不是有限收住**。所以门起的是"率压制"作用，不是"空间收住前沿"。机制诊断把原因说得更细：**提前招募确实工作了**（抑制在 80% 的轴向格子里领先前沿约 6.5ms）；但**分流钳制这条腿基本没工作**——在 g=0.25 和 g=0.5 两个强度下都只钳住 2–3% 的轴向高驱动细胞（前沿细胞上的抑制电流太小，分流电导拉不动电压到阈下）。真正压低/压死事件的是**宽抑制门的全局压制**，不是定点的分流钳制。所以诚实结论是：在当前读出口径、当前这片工作点上，"前沿分流刹车"做不出有限自限事件——提前招募腿成立、分流钳制腿缺席，净效应是全局压制，要么管不住（事件铺满），要么管过头（事件消失）。

**这不是"刹车机制不可能成立"的证明**，而是"在我们扫到的这片操作点（分流 0.25、门 0.1–0.6 等强度、固定宽核、单一驱动、L=20）里没找到能用的窗口，并且找到了一个机制层面的原因（钳制-存活权衡）"。

（内部归档代号：g_gaba_scale / gate_scale(I→E veto) / ei_gate_scale(E→I recruit) / reach_axis_mm / edge_margin / true_inter_event_floor / front_lead_by_axis I_lead_ms / clamp_check frac_axial_gated_by_shunt / 两层验收 Layer1 full-field finite + Layer2 virtual-SEEG AF/LR vs Task 0）

---

## 1. 两层验收结果（用户 2026-06-19 reframe 锁定的靶子）

| 层 | 判据 | 结果 |
|---|---|---|
| **Layer 1 全场/组织有限自限** | 事件有限：reach_axis 有界、edge_margin>0、不 tonic、不全 sheet | **FAIL（稳健）** — 事件在**所有**门强度下都铺满片子（max_reach ~20–26mm；细读出解出的可读双向事件 reach 24.9mm、贴边 edge<0）；门只降**率**（peak ↓50×）不降**空间范围**。无有限自限事件。 |
| **Layer 2 虚拟 SEEG 足迹 ≈ Task 0** | 模型每事件触点 AF/LR 与真实分布 KS 不可区分（α=0.01） | **MOOT / 不适用** — 门压制后几乎无可读模型事件；可测的极少数点 LR≪数据（0.09 vs 0.57），且 12 触点轴对齐 montage 与真实 ~20ch broad 池有 montage 粗度混淆，不能单独下结论 |

两层都没过。Layer 1 是这里的承重判据（Layer 2 因事件被压制 + montage 混淆而无法独立采信）。

## 2. 工作点筛选（Task 5，shunting g_gaba_scale 扫描，drive 0.6, L20, T8000）

| 工作点 | n_ev | fwd/rev | inter-event floor | peak | 读 |
|---|---|---|---|---|---|
| ref (current-LIF) | 22 | 9/6 | 0.0003 | 0.041 | GOLD：离散+安静+双向 |
| shunt g=0.0 | 1 | 0/0 | 0.060 | 0.479 | 跑飞/tonic（无抑制） |
| **shunt g=0.25** | 28 | 3/4 | 0.012 | 0.033 | **唯一候选**：离散+双向，但基线偏吵 |
| shunt g=0.5 | 9 | 0/0 | 0.0002 | 0.008 | 压制 onset（无方向） |
| shunt g=1.0 | 3 | 0/0 | 0.0001 | 0.0003 | 压死（峰值≈噪声底） |

分流窗口窄。g=0.25 是唯一勉强可用的点。

## 3. 门控试点（Task 6，g=0.25 + recovery + 门 0.1–0.6，drive 0.6, L20, T8000）

| 门强度 (veto=recruit) | n_ev | peak (率) | max_reach (范围) | 读 |
|---|---|---|---|---|
| 0 (control) | 17 | 0.0155 | 26.2mm | 事件铺满片子 |
| 0.1 | 12 | 0.0124 | 15.2mm | 率降、少事件 |
| 0.2 | 7 | 0.0098 | 25.0mm | 率降、事件仍铺满 |
| 0.3 | 1 | 0.0079 | 8.3mm | 率降到 1 个事件 |
| 0.4 | 17 | 0.0003 | 20.4mm | 率砸到噪声底 |
| 0.6 | 30 | 0.0003 | 22.6mm | 率砸到噪声底 |

**关键（率 vs 范围）：peak（率）随门强度单调下降 50×（0.0155→0.0003），但 max_reach（空间范围）在所有门强度下都维持 ~20–26mm（≈片子对角）。门砍的是"事件多频繁发放"（率），不是"事件传多远"（范围）——发放的事件仍然铺满片子。这是率压制，不是空间收住。**

**细读出复核（`--nc 9`, 18 触点，排除粗 montage 欠采样）：** gate=0.2 在细读出下解出 **3 forward / 1 reverse 可读双向事件**（说明粗读出的 readable=0 部分是 12 触点欠采样假象，不是没事件）——但这些可读事件 reach **24.9mm、贴边（edge<0）**，**仍然铺满片子，不是有限收住**；只有 1/7 事件 bounded-finite（reach<14 & edge>0）且 sub-readable。**所以即使把读出加细解出双向事件，也没有"可读 + 双向 + 有限"的收住事件——事件还是铺满片子。**

## 4. 机制诊断（front_lead + clamp，gate=0.2）

| 量 | 值 | 含义 |
|---|---|---|
| frac bins I ahead of E front | **0.80** (10 bins) | 提前招募**工作了**：抑制在 80% 轴向格子里领先前沿 |
| median I_lead | **6.55 ms** | 领先约 6.5ms |
| frac_axial_gated_by_shunt (g=0.25) | **0.02** | 分流在 g=0.25 几乎没钳住轴向高驱动细胞（2%） |
| frac_axial_gated_by_shunt (g=0.5) | **0.03** | 加倍分流强度也只钳住 3% —— 分流钳制**在任何测试强度下都几乎不 engage** |

**关键诊断（比"权衡"更准）**：分流钳制在 g=0.25 和 g=0.5 都只有 2–3%，说明**前沿高驱动细胞上的抑制电流 I_I 太小，导致分流电导 g_I=g_gaba_scale·I_I 根本拉不动 V 到阈下**——"分流钳住轴向放大器"这条腿**基本没工作**。事件被压低/压死的真正原因是**宽抑制门（veto）的全局压制**（整体兴奋性下降），不是定点的分流钳制。所以"前沿分流刹车"= ahead-recruit 腿成立（I 领先前沿 80%/6.5ms）+ shunting-clamp 腿基本缺席（2–3%）→ 净效应是全局压制。

## 5. 判定：STILL FAILS — "change mechanism" verdict EARNED（限定范围）

按 plan Task 7 分类：combo 在 shunting + ahead-recruit 下**只压率不收空间**——发放的事件（含细读出解出的双向事件）仍铺满片子，**且诊断确认 faithful 机制被实例化了**（前沿领先 0.80 / 6.5ms = 提前招募确实在做）→ 先前"change mechanism"的判断现在是**挣来的**（faithful 机制测过了、不足）。精确诊断：**ahead-recruit 腿成立**（I 领先前沿 80%/6.5ms）**，但 shunting-clamp 腿基本缺席**（g=0.25/0.5 都只钳 2–3%，前沿抑制电流太小，分流电导拉不动电压到阈下）；压制事件的是宽 veto 门的全局作用，不是定点分流钳制。所以这套"前沿分流刹车"在当前读出口径与工作点上做不出有限自限事件。

**口径锁（允许/禁止的话）：**
- 允许说：在 g=0.25 分流、门 0.1–0.6、固定宽核(l=1.5,C=150)、drive 0.6、L=20 的工作点里，"前沿分流刹车"**未复现**有限+可读+双向的自限事件；门表现为全局压制；提前招募机制本身被实例化（I 领先前沿 80%/6.5ms）但分流钳制基本不 engage（g=0.25/0.5 都只 2–3%），压制来自宽 veto 门的全局作用。
- **禁止说**："ahead-of-front 刹车机制不可能产生空间自限" / "M2 证伪了空间自限" / 把 Layer 2 的 montage-混淆数当独立证据。这是**限定工作点范围**的 NULL，不是不可能性证明。
- 与上游一致：M0（空间自限 lever 在 rate/mean-field 耗尽）+ M1（recovery 给时间自限不给可读空间自限）+ 现在 M2（SNN 前沿门也压制而非收住）→ 三层一致指向"在当前 read-out + 现有 magnitudes + flat-threshold 下，空间自限难以不靠压制实现"。

## 6. 诚实范围 + 下一步候选（未做，留给用户/未来）

测过的只是一片操作点。**没测**：veto 与 recruit 解耦（弱 veto + 强 recruit 可能 ahead 不压制）；更窄/更局部的门核；其它 drive / 分流 / e_gaba；非 flat-threshold 异质核下的门。这些任一可能改变结论。本结果是"在扫到的范围里没找到窗口 + 一个机制原因"，不是终判。

下一步候选（若继续）：① 解耦 veto/recruit 强度做 2D 小扫；② 把"轴向钳制"做成局部（只钳前沿邻域）而非全局宽核；③ 接受"读出口径下空间自限靠压制"为 Level-2 收口，转向其它可观测量（方向/时序）作为 M2 的真问题。

---

## 7. M2 验收口径锁（user review 2026-06-19，ACCEPTED）

**M2 验收通过。** 验收的精确陈述（只能这么说）：

> 在当前 SNN 基底上，**额外加入的"前沿分流刹车 / front shunting gate"在当前参数窗口和读出条件下，主要压低事件发生率（rate），而不是限制事件传播距离（reach）**，因此不能产生满足五项要求的空间自限间期事件。

**M2 加的"那一支"是什么**（拆解，便于不混淆）：在原 EI + 长轴 EE + E 疲劳网络外，叠加一条 **E→I→E 前沿抑制门控支路**——上游 E 提前招募前方 I（`ei_gate_scale`，**recruit leg 成立**：I 领先前沿 80%/6.5ms），前方 I 通过分流性 GABA（`g_gaba_scale`）把轴向 E 钳到阈下（`gate_scale`，**veto/clamp leg 基本缺席**：只钳 2–3% 前沿细胞）。净效应=宽域抑制/率压制，不是前沿截断。

**M2 ≠ Liou & Abbott 2020 full mechanism（关键，不可混为一谈）。** M2 只借用了 inhibitory restraint / inhibitory veto 的**概念**，方向与 Liou/Abbott **相反**：
- Liou/Abbott 2020 的核心是 **local recurrent projections + global feedback inhibition + usage-dependent exhaustion of inhibition + adaptation/AHP + chloride / inhibition-effectiveness 慢变量**；其传播机制更像"抑制约束被**活动依赖地耗竭/失效**后，那里才被 recruit"——是**刹车在某些位置被解除/耗竭**。
- M2 做的是**额外加一支更强的前沿刹车**。M2 **没有实现** Liou/Abbott 最关键的部分：usage-dependent inhibition exhaustion / chloride-mediated inhibitory failure / **spatially-localized inhibition defect（抑制约束的空间异质性失效）**。
- Liou 2018(Brain) 实验对照：ictal bursts 不传过 ~2–3mm，但远处可见 LFP + PV interneuron 活动（**"远处有 LFP/PV 活动 ≠ 远处 principal cells 被真正招募"**，feedforward inhibition）；intact inhibition→core 小 penumbra 大；**local inhibition defect→距 focus ~4–5mm 形成第二 focus**；contiguous inhibition breakdown→连续 Jacksonian-like 传播。M2 的"加前沿抑制→率降 reach 不降"与此一致：M2 缺的正是"局部抑制约束的空间异质性失效"。（以上 Liou 文献框定来自 user review，作为 M3 方向锚点；引用细节以原文为准。）

**三个禁止口径（硬锁）：**
1. ✗"前沿抑制不可能产生空间自限" → ✓"当前 **broad front-shunting gate** 没有产生沿轴空间自限"。
2. ✗"Liou/Abbott 机制失败 / 被证伪" → ✓"M2 借用了 inhibitory restraint 概念，但**没有实现** Liou/Abbott 的 full mechanism（尤其没有 usage-dependent inhibition exhaustion / chloride failure / spatially-localized inhibition defect）"。
3. ✗"空间自限已被证伪" → ✓"**'由均质前沿抑制增强产生沿轴自限'这一机制假设，在当前模型中被强烈削弱，足以支持换机制**"。

**验收不以 veto/recruit 解耦小扫为前置条件**（user 明确）。机制的自由度主要沿 **rate/amplitude** 移动、不沿 **propagation length** 移动；recruit leg 已证工作、veto leg 在可存活工作点几乎不工作、更强 veto 趋向压死、更弱 veto 不限 reach——继续解耦大概率只是再确认同一结论。M2 的目标是判断"这个方向（均质前沿抑制增强）是否值得作为**主机制**继续"——答案明确：**不值得**。§6 的三个候选降级为 appendix / M3 前 optional engineering check，不阻塞 M2 验收。

**→ 转 M3：方向是 Liou-direction（抑制约束 + 其空间异质性失效/耗竭），不是加更强的均质刹车。**
