# M3A-A2 — Abbott local+global resource (动态慢变量) DESIGN SPEC (2026-06-25)

> Scope: M3A-A2 PILOT design only。承接 A1/A1b/A1c。**不碰 M3B**（W / h(W) / SEEG bridge）；
> 不重用 W 做机制源头。本 spec 锁定"动态局部+全局抑制资源（Abbott-LG）"的机制方程、引擎落地、
> 衬底锚点、读出 schema、验收门、不变量合同与 pilot 范围。**实现计划另起**（writing-plans →
> `docs/superpowers/plans/2026-06-25-sef-hfo-m3a-a2-abbott-lg-plan.md`）。
>
> Provenance: 承 `m3a_quasistatic_slowvars_recap_2026-06-24.md`、`m3a_a1b_state_topography_2026-06-25.md`、
> `m3a_a1c_pilot_recap_2026-06-25.md`、`a1c_dynamic_global_feedback_spec_2026-06-25.md`。
> Worktree（实现期）：off `topic4-snn-m3-hub` HEAD 的新 A2 worktree（见 §10）。
> 用户决策锁（2026-06-25）：2 油箱（核+全局）/ 对称耗竭 / interictal 锚点先按绝对 tail-ratio 复核 /
> 新 A2 worktree / Abbott 合同引 handoff 方程 + A1b/A1c archive（`docs/paper/abbott_model.md` 物理缺失）。
>
> **v2（2026-06-25 用户 review 后，6 项保护已整合）**：#1 frozen-q 边界复核（Task-0b，§4.1）；#2 `q_global`
> 改真全局乘子（§2.1/§2.4）；#3 per-core `q_L/q_R` 复核假同步（§9.1 Task-3）；#4 `k_use` 从目标 q 反推
> （§7 #14）；#5 onset/offset 因果分开、回落=涌现非慢变量终止（§4.3.3）；#6 R-class（R4a/R4b）接入 gate
> （§4.3.2，复用 `classify_event`）。pilot 改分阶段硬停（§9）。

---

## 0. 朴素话（测什么 / 怎么测 / 揭示什么）

**测什么**：一片脑组织里埋了两个易兴奋的"病灶核"，平时背景噪声下各自冒一点会回静的小放电（间期样）。
我们给每个区域配一个"抑制油箱"——一个 0 到 1 之间的量，表示这个区域当下能用的抑制有多足。**放电越凶，
油箱漏得越快；安静时慢慢回灌**。两个油箱：一个套在病灶核上，一个是全网背景。核心问题：当核自己放电把
自己那箱抑制漏掉（局部去抑制），网络会不会**自己**（不戳）从零星小放电，滑进一段"大范围、两核同步招募"
的发作样活动，再滑回间期样——还是要么压根没动，要么一旦点着就回不来（失控）。

**怎么测**：把油箱接到引擎里现成的那条"按使用耗竭、缩放抑制"的慢变量通路上（这条通路本来就有一个逐神经元
版，叫 z；A2 只是把它从逐神经元推广成逐区域）。从 A1b 已经画好的状态地图里挑一个**间期样**的格子做起点
（那里有干净的、会回静的小放电基线），开长记录、不外部戳。盯一个总数 `rho(t)`：核相对全局把油箱漏得多狠
（油箱满时 `rho` 就等于 A1b 那个格子的静态坐标，漏了就升）。**如果油箱完全冻住（不漏），`rho` 不动、只看到
间期样小放电；实测让它能漏**，我们看 `rho(t)` 会不会自己升过 A1b 标好的"发作样"那条线（核抑制塌得比回灌快），
再降回来（油箱回灌）。

**揭示什么**：三种可能、都有信息——(1) 油箱漏得太弱 → 一直停在间期样（机制不够）；(2) 漏到某个强度 →
`rho(t)` 升过线、活动变成发作样、之后（靠快事件自限 + 油箱回灌）再滑回间期（**目标相变**：use-dependent
去抑制能自发驱动 间期→发作样 这一步；**回落是涌现**，不是慢变量主动终止）；
(3) 漏太狠 → 升过线就回不来（失控）。**这是一个机制 screen，不是"证明真实发作机制"**：能看到 (2) 只能说
"在这个模型、这个量级、这个读出下，看起来像 use-dependent 去抑制能驱动相变"；看不到则把方向交给下一档
配料（见 §8.4）。

（内部归档代号：M3A-A2, Abbott-LG, q_core/q_global regional inhibitory resource (q_global=真全局乘子,
q_core=核额外; per-core q_L/q_R), symmetric depletion, `rho(t)=lgr_static/(q_core·q_global)`,
RegionalResource on slow path, A1b `local_global_ratio`, frozen-q boundary (Task-0b), absolute
`tail_to_baseline_ratio` gate, R-class `classify_event` (R4a 有前沿/R4b tonic) gate, onset-leads(bout 级)/
offset-recovery(涌现) discriminator, k_use-from-q_target, twoend_equal Stage-3 core,
`slow_vars.py` SlowVars/z generalization, off-by-default byte parity。）

---

## 1. 背景与定位：A1 给了什么，A2 站在哪

三句话 recap（精度版见各 archive）：

- **A1 quasi-static**：在**均匀**衬底上冻结单个慢变量 → 没有间期基线（OFF=纯 R0），相变测不了。
  结论：要异质兴奋核衬底；`e_GABA` 静态最强但**不是 slow-var**（膜分流路，与 slow 路互斥）；静态 `z` 不可
  判负（应在动态起作用）。
- **A1b state topography**：在 **Stage-3 `twoend_equal` 双核**衬底上，固定结构旋钮（局部 loop 强度
  `core_ei_scale/core_ee_gain` × 全局 restraint `global_ei_scale`）→ 画出 **静默 / 间期样 / 发作样 / 失控**
  四态地图，**`local_global_ratio = (core_ee_gain/core_ei_scale)/global_ei_scale` 是状态主轴**。
- **A1c dynamic global feedback**：把全局抑制做成"随全网率升而升"的均匀刹车 → **不能干净终止核集中型失控**
  （要压住核就过压弱态；`i_global` 不领先率降 = 区分不开动态/静态）。结论：**均匀全局不是对的工具，机制要
  空间化（局部）或 use-dependent 耗竭**。

**A2 = 接 A1b 的状态轴 + 答 A1c 的空间缺口**：把 `local_global_ratio` 从静态坐标变成**动态** `rho(t)`，
由**区域活动驱动的抑制资源耗竭**推动。A1b 已证静态比例能形成四态；A2 测**动态 `rho(t)` 能不能自发穿过这些
已标好的边界**。空间化体现在：核有自己的油箱、由核自己的活动驱动（A1c 缺的就是这种局部targeting）。

---

## 2. 机制方程（Abbott-LG，2 油箱，对称耗竭）

### 2.1 状态变量

区域抑制资源（"油箱"），各 ∈ `[q_min, 1]`，**缩放对应区域 E 细胞收到的抑制电流**：

- `q_global(t)`：**全局乘子**——缩放**所有** E 细胞（核 + 背景）收到的抑制。这是 A1b `global_ei_scale`
  轴的动态对应：A1b 里 `global_ei_scale` 也作用于**全部** E（`a1b_weight_lesion`: `ls=full(global_ei_scale)`，
  核再 `×core_ei_scale`）。**[review-fix #2]** 早期把 `q_global` 只作用核外背景 = 与 A1b 全局轴不对应，已改真全局乘子。
- `q_core(t)`：**核额外乘子**——只在核 E 上再叠一层（= A1b `core_ei_scale` 的动态对应）。

**有效抑制（油箱 = A1b 静态权重之上的时变乘子）**：核 E `I_inh = (A1b 核抑制)·q_global·q_core`；
背景 E `I_inh = (A1b 背景抑制)·q_global`。`q_min ∈ (0,1)` 地板（默认 0.25，PLACEHOLDER）；初值全 1.0
（满箱 = A1b 静态态）。

**[review-fix #3] 两核分辨率**：核并集 `core_mask_E = (core_masks[0]|core_masks[1]) 的 E` 的**单一** `q_core`
仅用于 Task-1/2 的 **union smoke**。**一旦出现 R-excursion 或同步上升，必须用 per-core `q_L/q_R`**（左核活动
只耗 `q_L`、右核只耗 `q_R`，`q_global` 仍全局）复核——排除"共享 `q_core` 把左核事件同时去抑制右核 → 假同步"
（见 §9.1 Task-3）。

### 2.2 活动代理（驱动耗竭）

区域 `r` 的"使用强度" = 该区 E 细胞当步发放比例的短时低通（EMA，平滑 spiking 噪声）：

```
a_r(t)   = (该区当步 spike 的 E 数) / (该区 E 总数)         # 瞬时发放比例 ∈ [0,1]
ā_r(t)   = ā_r(t-dt) + α_a · (a_r(t) - ā_r(t-dt))           # EMA
α_a      = 1 - exp(-dt / τ_a)                                # τ_a 小（默认 100 ms，PLACEHOLDER）
```

> 选择记录：用**发放率**做"use"代理（贴 handoff `k_use·local_activity_region` + Abbott use-dependent）。
> 备选（敏感性，不在 pilot 主线）：用区域平均抑制电流 `I_I`（z 原版的代理 `H(g_th-I_I)`）。

### 2.3 耗竭 ODE（前向 Euler，对齐引擎现有 z 更新风格）

```
q_r ← q_r + dt · [ (1 - q_r) / τ_rec  -  k_use · ā_r(t) · q_r ]
q_r ← clip(q_r, q_min, 1.0)
```

- `(1-q_r)/τ_rec`：回灌项（朝满箱 1 恢复），`τ_rec` 默认 5000 ms（≈z 的 `tau_z`、deck ~5 s，PLACEHOLDER）。
- `-k_use·ā_r·q_r`：使用耗竭项（放电漏箱），`k_use` 是 pilot 的主扫旋钮（强度 ladder）。
- **对称**：`q_core`/`q_global` 同号同式，只是驱动活动 `ā_r` 来自各自区域（核活动 vs 全局活动）。

定点（活动恒定 `ā` 时）：`q* = 1/(1 + k_use·ā·τ_rec)`。**注意**：单纯对称耗竭是去抑制正反馈
（漏→更兴奋→更漏），**"发作样且回落"不是 ODE 自带的**——回落要靠衬底的快事件自限 + 发作样 bout 后的
**涌现网络静默**让油箱在间隙回灌（见 §8.2 风险）。

### 2.4 施加到膜电流（slow 路 `apply_currents`）

对每个 E 细胞 `i`（**[review-fix #2]** `q_global` 真全局、`q_core` 核额外）：

```
scale_i  = q_global · q_core   if i ∈ core_mask_E   else  q_global
I_net[i] = I_E[i] - scale_i · I_I[i]
```

per-core 模式：核 E 用 `q_global·q_L`（左核）/ `q_global·q_R`（右核）。**core-only 模式（Task-1）**：
`q_global ≡ 1`，只 `q_core` 动 → `scale_i = q_core`（核）/ `1`（背景）。I 细胞 `I_net = I_E - I_I` 不变。
满箱 `q=1` 时 `scale_i = 1.0` ⇒ `I_net = I_E - 1.0·I_I ≡ I_E - I_I` ⇒ 与 `slow=None` **逐位一致**
（IEEE `1.0·x==x`）——off-by-default parity 的根据（§5.2）。

### 2.5 读出坐标 `rho(t)`（A2 的主观测）

```
rho(t) = (core_ee_gain / (core_ei_scale · q_core(t))) / (global_ei_scale · q_global(t))
       = lgr_static / (q_core(t) · q_global(t))
其中 lgr_static = (core_ee_gain/core_ei_scale)/global_ei_scale   # = A1b 该格子的满箱坐标
```

油箱漏（`q↓`）⇒ `rho↑`；满箱 `rho = lgr_static`。配 §2.1 的 `q_global` 真全局乘子，这正是 `lgr` 沿**抑制轴**
的动态像（`core_ei_scale → core_ei_scale·q_core`，`global_ei_scale → global_ei_scale·q_global`）。

> 诚实边界：`rho` 是**模型地形坐标，不是真实生理量**（沿用 A1b 对 `local_global_ratio` 的标注）。

> **[review-fix #1] `rho` 只沿抑制轴动 ⇒ 边界必须 frozen-q 复核**：`q` 只缩放**抑制**，**不动
> `core_ee_gain`**（核内 recurrent E→E）。但 A1b 的局部 loop 轴 `l0→l1→l2` 是 `core_ei_scale` 和
> `core_ee_gain` **一起**变。所以 A2 在 `core_ee_gain` 固定于锚点值时把 `rho` 漏到 1.35，与 A1b 的 `l1_g1.0`
> （`core_ee_gain=1.15`）是**不同操作点**、只是 lgr 数值相同。⇒ A1b 的 1.35/1.86 **不能直接借作 A2 硬边界**，
> 必须 Task-0b 在**本锚点**用 frozen-q 实测复核（§4.1、§9.1 Task-0b）。

---

## 3. 衬底与锚点

### 3.1 衬底（Stage-3 `twoend_equal`，A1b 标定参数）

复用 `build_lesion_vth(... mode='twoend_equal')`：两个对称易兴奋核
`neg_xy/pos_xy = center ∓ sep_frac·half·axis_unit`，各 `sample_core_field(core_mean, core_std, core_r)`，
返回**逐核全网 bool masks**（喂 `build_sidecar` 算每核 onset）。

**A2 必须显式传 A1b pilot 标定值（不靠 runner 默认）**：`L=20, density=100, theta=45,
core_mean=17.5, core_std=1.0, core_r=1.5, sep_frac=0.7, drive(nu_ext)=0.6`。
（runner 默认是 `core_mean=17.0, core_std=1.5, sep_frac=0.6`——错的，会换衬底。）

结构旋钮（A1b，`a1b_weight_lesion → build_connectivity_rot`，**不改引擎、`scale=1` bit-parity**）：
局部 loop level `l0=(core_ei_scale 1.0, core_ee_gain 1.0) / l1=(0.85,1.15) / l2=(0.70,1.30)`；
全局 `global_ei_scale ∈ {0.7,1.0,1.3,1.6}`。**A2 的油箱是这些静态权重之上的时变乘子**（§2.4）。

### 3.2 锚点选择（Task-0：先按绝对 tail-ratio 复核，再锁）

A1b on-disk 实测（`status_a1b.json`，本 spec 引用值）：

| cell | state(disk) | evt Hz | glob Hz | core Hz | return\* | collis | r95 | **lgr** | nseed |
|---|---|---|---|---|---|---|---|---|---|
| **l0_g1.0** | interictal | 3.21 | 4.98 | 14 | 0.778 | 0.119 | 10.8 | **1.00** | 3 |
| **l1_g1.3** | interictal | 4.58 | 3.39 | 15.4 | 0.645 | 0.123 | 5.28 | **1.04** | 5 |
| **l2_g1.6** | interictal | 3.15 | 2.65 | 20 | 0.864 | 0.232 | 5.48 | **1.16** | 5 |
| l1_g1.0 | **seizure** | 3.62 | 8.2 | 35 | 0.901 | 0.339 | 9.94 | **1.35** | 5 |
| l2_g1.3 | interictal† | 3.45 | 4.94 | 32.8 | 0.875 | 0.326 | 7.88 | **1.43** | 5 |
| l2_g1.0 | **runaway** | 8.35 | 26.2 | 414 | 0.020 | 0.2 | 12.4 | **1.86** | 5 |

\* `return` 这里是 A1b 的 per-event-peak-relative `return_to_baseline_fraction`，**不是**绝对 tail-ratio；
A1 handoff 警告这些"returnable"标签在绝对 tail 上可能仍偏高。**Task-0 必须用 A1c 的绝对
`tail_to_baseline_ratio` 重判**。
† **[CORRECTION]** on-disk `l2_g1.3` = interictal（r95 7.88 < 8 gate），不是 recap 说的 seizure_like
（recap 用 r95 9.2）。⇒ **on-disk 唯一稳健 seizure_like 格子 = `l1_g1.0`（lgr 1.35）**。

**锚点选择规则（Task-0，预注册，[review-fix] clean-then-near-boundary）**：在候选 `{l0_g1.0, l1_g1.3, l2_g1.6}`
各跑满箱（`k_use=0`）长记录，**先筛 clean**（非零事件 + 绝对 `tail_to_baseline_ratio ≤ 1.5` + 多数 returned），
**再在 clean 候选里选离 seizure 边界最近**者为主锚（离太远 → 一直 R-stay → 信息量低）：

- **首选近边界 clean 主锚 = `l2_g1.6`**（lgr 1.16，离 1.35 最近 + return 0.864 三者最干净），若其绝对 tail clean。
- **`l0_g1.0`（lgr 1.00）= 干净归因参照锚，必跑**：`core_ee_gain=1.0`、满箱即纯静态、跨越最大，最能区分
  "机制驱动 vs 只是移到某静态工作点"——即使不做主锚也作归因对照。
- 备选 `l1_g1.3`。**Task-0 输出**：三候选绝对-tail 基线 + 锁定主锚 + 归因参照锚。
- 注意（衔接 §2.5/§4.1）：不同锚 `core_ee_gain` 不同（l0=1.0、l2_g1.6=1.30），故每个主锚的 `rho` 边界
  **各自** Task-0b frozen-q 复核，不可跨锚借用。

### 3.3 区域 mask 注入

`RegionalResource` 构造期接 `core_mask_E`（核 E bool，长度 NE），背景 = 补集——镜像 `φ` 接 `vth_field`
的现有模式（slow 路构造期注入逐神经元场）。**不依赖 `apply_currents(labels)` 携带区域信息**（`labels`
仅 E/I），区域 mask 存在对象里。

**异质核阈值场必须显式传（load-bearing，§6 boundary-param）**：runner 调 `simulate_kick(...,
slow=RegionalResource(...), V_th_per_neuron=<twoend_equal 核 vth 场>)`，`RegionalResource.threshold(base_vth)`
**原样返回** base_vth（不覆盖为 φ）。**漏传 `V_th_per_neuron` ⇒ `base_vth=p.V_th` 均匀 ⇒ 衬底退化为同质 ⇒
退回 A1 "无间期基线" 失败**——这是静默科学污染，必须有不变量测试（§7 #11）守。

---

## 4. 验收门 + 预注册结果（grounded in A1b lgr 阶梯）

### 4.1 `rho` 边界（A1b lgr = **起点假设**，Task-0b 在本锚点 frozen-q 实测复核）

A1b on-disk lgr 阶梯：间期 ~1.0–1.16 → **seizure 线 ~1.35（l1_g1.0）** → 失控 ~1.86（l2_g1.0）。
**[review-fix #1] 这是起点假设、不是 A2 硬边界**（因 `q` 只动抑制轴不动 `core_ee_gain`，§2.5）。**Task-0b
（frozen-q boundary）**：在主锚冻 `q` 使油箱积 `q_core·q_global ∈ {1.0, lgr/1.35, lgr/1.86}`，跑 no-kick 长记录，
看是否分别复现 间期 / 发作样 / 失控。**复现** → 沿用 1.35/1.86；**不复现** → 用 frozen-q 实测的 **A2-本征边界**
替代，后续所有动态判据对齐 A2 边界。下面数值按"若 A1b 边界成立"给（Task-0b 后锁定）：

归因参照锚 `l0_g1.0`（lgr=1.00）：跨 seizure 约需油箱积 ≤ 1.00/1.35 = 0.74、跨失控 ≤ 0.54
（发作样窗 ≈ [0.54, 0.74]，`q_min=0.25` 易达）。主锚 `l2_g1.6`（lgr 1.16）的窗由 Task-0b 标定。

### 4.2 三种预注册结果（都有信息）

| 结果 | 现象 | `rho(t)` | 含义 / 下一步 |
|---|---|---|---|
| **R-stay** | 一直间期样小放电 | 卡在锚起点、不过边界 `B` | 油箱漏太弱（`k_use` 不够）→ 加强；若全 ladder 都不过 = 此机制在此衬底不够 |
| **R-excursion**（目标） | 自发一段发作样 bout（两核同步↑、r95↑、广招募、**R4a**）夹在间期之间 | 升过本锚 Task-0b 边界 `B`、bout 内维持、bout 后回落 | use-dependent 去抑制驱动 onset/excursion；**回落是涌现**（screen 级"看起来像"） |
| **R-runaway** | 点着就回不来（tonic / tail 不回基线 / **R4b**） | 升过 `B` 后钉在失控带、不回落 | 漏太狠 / 缺对抗回落机制 → §8.4 下一档配料 |

### 4.3 R-excursion 的硬判据（验收，承 A1c §5.5 + Abbott 专属）

1. **基线锚定（前置，Task-0）**：主锚满箱（`k_use=0`）= **实测**干净自限间期基线：非零事件 + 绝对
   `tail_to_baseline_ratio ≤ 1.5`（= 回静/非失控，沿用 A1c gate）+ 多数事件 returned。**纯 R0（零事件）=
   停，重选锚**（A1 §4 gate）。注：tail 只分"回静 vs 失控"；**间期-vs-发作样靠 `rho` 带 + collis/r95，不靠
   tail**（间期与发作样都可"回静"，区别在招募范围/同步，不在 tail）。
2. **相变 + 回落（primary，[review-fix #6] R-class gate）**：某 `k_use` 档自发出现一个 **bout** 同时满足
   **全部四条**：(i) `rho(t)` 升过**本锚 Task-0b 标定的 seizure 边界**；(ii) 事件形态转发作样
   `collision/sync↑` + `r95/招募↑`；(iii) **R-class 达 `R4a`（持续招募且保有空间前沿）或 `R3→R4a-like` bout**
   ——复用 `src/sef_hfo_mu_basin.py::classify_event`，**不是 `R4b` 均匀 tonic**；(iv) **回落** 绝对
   `tail_to_baseline_ratio ≤ 1.5` 持续 ≥ 500 ms 且油箱回灌过 seizure 边界对应积。**`rho` 穿线只是坐标证据 (i)；
   真成功 = (i)+(ii)+(iii)+(iv) 同时**。单个孤立 tonic / `R4b` ≠ 成功。
3. **因果判据（[review-fix #5] onset / offset 分开，对齐对称耗竭机制）**：
   - **onset claim（慢变量领先 bout 级相变，不是领先第一下 spike）**：`q` 由 spike 使用驱动，不可能在毫无
     前序间期事件时凭空领先第一个 spike；"领先"定义在 **bout 级**：(a) `rho_pre`（bout 前慢变量值）**预测下一个
     bout 的 class**（hazard：`rho_pre` 越高，短时窗内出 R3/R4a bout 概率越高）；或 (b) `rho(t)` 在 bout 的
     广招募 / collision 抬升**之前**已升高。
   - **offset claim（[review-fix #5] 不要求 `q` 回灌领先活动下降）**：当前对称耗竭 ODE 在活动仍高时耗竭项仍在，
     `q` 回灌**只能跟在**活动下降之后（快事件自限先让活动掉 → `q` 才回灌）。所以 offset 成功 = **绝对
     `tail ≤ 1.5` + inter-bout 间隔内 `q` 回到接近满箱 + 系统重回间期样事件分布**，**不是** "q 领先率降"。
     **要证"慢变量导致终止" = 换机制**（加 `g_K` / 负反馈 / recovery，§8.4）；本机制只能 claim：use-dependent
     去抑制驱动 onset/excursion，**回落是涌现**（快自限 + 慢回灌）。
4. **anti-rules（硬）**：(a) "事件变多" ≠ 成功——必须是**跨边界的相变**（A1 quasi-static 教训：旋钮内率升
   但无 R-class 迁移 = null）；(b) "completed within budget" ≠ termination——用**绝对** tail gate，非
   per-peak-relative；(c) 单个孤立 tonic ≠ 发作样 bout；(d) `rho` 穿线但事件形态没变（collision/r95 不动）
   = 坐标假象，不算相变。
5. **seeds / 一致性**：pilot **3 seeds**（A1c pilot 同档；full 升 5）。逐 seed 报，看 sign-consistency，
   **不 pool-p**。

### 4.4 判读语言纪律（承 A1c P1-1）

允许写："在此衬底/量级/读出下，动态局部+全局抑制资源**对称耗竭** 能 / 不能自发驱动 间期→发作样→恢复
相变"。**禁止**写"证明真实发作机制""证明 Abbott 论文机制""W 导致发作""use-dependent 耗竭机制成立"。
screen 级 = "看起来像 / 不像 / 没看清"，不是 PASS/validated。

**[review-fix #5] 终止归因纪律**：最多写"use-dependent 去抑制能驱动 onset/excursion，回落是**涌现**
（fast self-limitation + slow refill）"；**禁止**写"慢变量导致发作终止"（那需 recovery 机制，不在本 pilot）。

---

## 5. 引擎落地（slow 路；**不改 `kick_probe.py`**）

### 5.1 落点：`src/snn_engine/slow_vars.py` 新增 `RegionalResource`

`simulate_kick` 已驱动完整 slow 环（`kick_probe.py:256 apply_currents` / `:261 threshold` /
`:285-286 step`，全部 `if slow is not None` 门控）。⇒ 新慢变量对象**零改 `kick_probe.py`**（M3B 正改的
文件）。两个已验证的关键事实（2026-06-25 firsthand）：

- **异质核 substrate hook 已存在**（`kick_probe.py:258-263`）：`base_vth = p.V_th if V_th_per_neuron is None
  else V_th_per_neuron`，`V_th_eff = slow.threshold(base_vth)`。⇒ A2 **不需新引擎 hook 接核**——runner 传
  `V_th_per_neuron = twoend_equal 核 vth 场`，`RegionalResource.threshold(base_vth)` **原样返回** base_vth，
  核异质阈值即生效（A1 recap 警告的"slow 路看不见核"是 hook 前的；该 hook = A1 plan Task 1，现已落地）。
- **q=1 parity 逐位精确**（已验证）：slow 路膜更新 `kick_probe.py:270 Vtmp = I_net + (V-I_net)*decay_V`
  与 `membrane_step` 默认体（`:84-85 I_net=I_E-I_I; return I_net+(V-I_net)*decay_V`）**同式**；满箱
  `q=1` 时 `apply_currents` 返回 `I_E - 1.0·I_I ≡ I_E - I_I`（IEEE `1.0·x==x`）⇒ `RegionalResource(q1,k0)`
  与 `slow=None`（走 `:278 membrane_step`）逐位一致。

`RegionalResource` 实现 slow 协议三方法：

```
class RegionalResource:                 # off-by-default：只有 runner 显式构造才进 slow 路
    __init__(self, N, V_th0, core_mask_E, cfg)
        # mode: 'core_only'(q_global≡1) / 'two_tank'(q_global 真全局乘子) / 'per_core'(q_L,q_R,q_global)
        # 状态：q_global=1.0, q_core=1.0(union) 或 q_L/q_R=1.0(per_core), ā_* EMA, _I_I_last
        # 存 core_mask_E（per_core 另存 左/右 masks）；背景=补集
    apply_currents(self, I_E, I_I, labels) -> I_net    # §2.4: 核 E ×q_global·q_core(或q_L/q_R), 背景 ×q_global
    threshold(self, V_th_base) -> V_th_base            # 原样返回（核异质阈值经 V_th_per_neuron 生效）
    step(self, spk, labels, dt)                        # §2.2 EMA + §2.3 ODE; per_core: 左核活动只耗 q_L
```

> 复用而非重发明（§6 helper-reuse）：`RegionalResource` 是 `SlowVars` 的 z-路**逐区域推广**
> （z 逐神经元缩放 `I_I`；此处逐区域缩放）。若用 `SlowVars` 子类更省，需保证 z 路 mutual-exclusion
> 不被破坏（A2 开 RegionalResource 时 `use_z/use_phi/use_gK` 全关）。

### 5.2 off-by-default byte parity（硬合同）

- `slow=None` → `kick_probe.py:255` 原路 `I_net=I_E-I_I`，不变。
- `RegionalResource(q=1, k_use=0)`（满箱冻结）→ `I_net=I_E-1.0·I_I` 逐位 == `slow=None`。
- **parity 测试自包含、不用 fixture**（避开 M3B 正改的 `tests/fixtures/a1c_parity_baseline.pkl`）：
  同一测试内跑 `simulate_kick(slow=None)` 与 `simulate_kick(slow=RegionalResource(q1,k0))`，
  断 `lfp_trace/rate_E` `array_equal` + `E_spk_bool` sha 等 + RNG state 等。

### 5.3 引擎校验表（re-bless）

`slow_vars.py` **不在** `engine_versions.json` watch list（6 文件：kick_probe/params/model/connectivity/
connectivity_rot/lfp）——**[recon-fix] 编辑 `slow_vars.py` 不触 guard、无需 re-bless**。A2 **不改任何 watched
文件**（只改 `slow_vars.py` + 新文件），故 `assert_versions`（T8）自动保持 green、**零 re-bless**。代价：
`slow_vars.py` 漂移不被 guard 抓——**由 A2 测试套件（parity + ODE-精确 + region-partition）充当 RegionalResource
的漂移哨兵**。（不把 `slow_vars.py` 加 watch list：会给 TDD 每改一次都要 re-bless 的摩擦 + 与 M3B 共享 registry。）

### 5.4 互斥守卫（承 A1c T6 模式）

`RegionalResource` 与以下**互斥**，构造/runner 期 fail-fast（`raise`）：A1c feedback（`feedback_gain>0`，
`kick_probe.py:195 assert slow is None`）、`shunt_gaba`/`e_gaba`（膜分流路）、`FrozenSlowVars`、
`SlowVars(use_z/phi/gK)`。**一次只开一个机制**（A1 纪律）。

---

## 6. 读出 schema（A1c superset + 慢变量层）

每 run `readout_*.json`（≥ A1c 字段）：

- `activity`（沿用 A1c）：`global_E_rate_mean_hz, global_E_rate_p95_hz, tonic_fraction,
  active_E_fraction_peak, core_E_rate_mean_hz, surround_E_rate_mean_hz, tail_to_baseline_ratio,
  baseline_abs_hz, peak_E_rate_hz, completed`。
- **`a2` block（新，RegionalResource on 时）**：`mode(core_only/two_tank/per_core), k_use, q_target,
  tau_rec, tau_a, q_min, q_core_min/q_global_min(轨迹最低; per_core 另 q_L_min/q_R_min),
  q_core_end/q_global_end, rho_static(=lgr), rho_peak, rho_p95, rho_core(=lgr/q_core), rho_surround(=1/q_global),
  seizure_boundary(Task-0b 标定值), n_boundary_crossings, frac_time_seizure_band, frac_time_runaway_band,
  onset_leads_rho_vs_recruit_ms(>0=领先), rho_pre_vs_next_bout_class(hazard 关联),
  q_recovered_interbout(bool)`。**[review-fix #5] 删 `leads_qrefill_vs_decay_ms` 作硬判据**——对称耗竭下回灌
  必跟随活动降、非领先；只描述记录 `qrefill_vs_decay_ms` 不作 gate。
- **慢变量 trace `a2_trace_*.npz`**：`q_core_bin, q_global_bin, rho_bin, a_core_bin, a_global_bin`
  （per_core 另 `q_L_bin/q_R_bin`，1 ms binned，长度 = T/1ms）+ `rate_E_hz`（全分辨率）。
- **per-event**（沿用 + 油箱值 + R-class）：`events[]` 各元素加 `q_core_pre/onset/peak/end,
  q_global_pre/onset/peak/end, rho_pre/onset/peak/end`（四相位快照）+ **`R_class`**（`classify_event` 标签）
  + 其输入 metrics `{event_detected, runaway, returned, sustained_front_score, active_peak, r95_ea, far_ea}`
  （**[review-fix #6]** 复用 `src/sef_hfo_mu_basin.py::classify_event`，`DEFAULT_CAPS` 与 A1 quasi-static 同）。
- 其余沿用 A1c：`tag, provenance, config, detector, n_events, stage3_source_counts, rep_event_index`。

聚合 `status_a2_pilot.json`：`base, tier("MECHANISM-SCREEN, NOT validation"),
anchor_selected, baseline_check{cell→{evt_hz, tail_abs, clean}},
per_kuse{k→{n_seeds, states[], n_excursion, n_runaway, n_stay, rho_peak_med, leads_med,
frac_seizure_band_med}}, verdict{R-stay/R-excursion/R-runaway per seed + 一致性}, caveat`。

状态分类（复用 A1c + A1b 阈，**描述性**）：`IGNITE_PEAK=0.05 (或 peakE≥3.0)`，绝对 `TAIL_GATE=1.5`；
bout 级 regime 用 `rho` 带（边界用 **Task-0b 标定值**，非硬编 1.35/1.86）+ 活动门（collis/r95）+ **R-class
（R4a/R4b/R3）** 三重确认（[review-fix #6]：R-class 是 phenotype 证据、`rho` 是坐标证据，二者须一致）。

---

## 7. 不变量合同（给实现计划 TDD，每条一测）

1. **off-by-default parity**：`RegionalResource(q1,k0)` 逐位 == `slow=None`（§5.2，自包含、无 fixture）。
2. **bounded**：任意步 `q_core,q_global ∈ [q_min,1]`（clip 生效，no overflow）。
3. **ODE 精确性**：给定 `(spk 序列, k_use, τ_rec, τ_a)`，`step` 后 `q/ā` == 手算前向 Euler（`atol=1e-9`，
   镜像 A1c T4/T5）。
4. **rho 一致性**：`rho(t) == lgr_static/(q_core·q_global)`，且满箱 `rho==lgr_static`；`lgr_static`
   与 A1b 公式一致。
5. **区域划分**：核 E 用 `q_core`、背景 E 用 `q_global`、I 细胞不缩放；mask 是 E 的不相交覆盖
   （`core_mask_E ⊕ 补集 == 全 E`）。
6. **no-NaN**：默认参数长记录无 NaN/inf。
7. **mutual-exclusion raise**：RegionalResource ∧（A1c feedback / shunt_gaba / SlowVars active）→ `raise`（§5.4）。
8. **engine blessed（T8 green）**：A2 不改 6 个 watched 引擎文件 ⇒ `assert_versions` 自动 pass；测试断言
   T8 green（[recon-fix] `slow_vars.py` 不在 watch list、无需 re-bless）。
9. **trace shape**：`a2_trace_*.npz` 各数组长度 == T/bin；per-event 四相位油箱键齐全。
10. **substrate parity guard**：A2 用 A1b 标定衬底参数（§3.1）；runner 拒绝默认 `core_mean=17.0` 跑 A2
    （或显式 echo 实际值进 provenance，防默认衬底污染）。
11. **异质核活着（heterogeneous core live）**：RegionalResource on 时，核 E 细胞的有效阈值 == 传入的
    `V_th_per_neuron` 核场（非均匀 `p.V_th`）；测试断言核 E 阈值场被用上（如：核 E 的 `V_th_eff` 低于背景、
    或满箱 A2 run 的核事件结构 == A1b 同格 run）。漏传 `V_th_per_neuron` → 同质退化 → 该测试必须 RED。
12. **[#2] `q_global` 真全局**：核 E 有效抑制 == base·`q_global`·`q_core`、背景 == base·`q_global`；
    测试取 `q_global=0.5, q_core=1`，断言核与背景的 `I_I` 缩放**都**变 0.5（漏掉核 = 旧 background-only bug = RED）。
13. **[#3] per-core 耗竭隔离**：per_core 模式下注入仅左核发放，断言 `q_L↓` 而 `q_R` 不动（且 `q_global` 由全局
    活动驱动）。union 模式仅 smoke；任何 sync 结论须 per-core 复核（§9.1 Task-3）。
14. **[#4] `k_use` 反推**：runner 接 `--q-target`（或 `--rho-target`），由 Task-0 实测 `ā_core` 经
    `k_use = (1/q_target − 1)/(ā·τ_rec)` 反推（不接受任意 `k_use` 作主 ladder）；测试断言反推公式 + 满箱 `ā` 取自
    Task-0 baseline（注：`q*` 用 baseline `ā` 定"静息工作点"，bout 内 `ā↑` 会进一步过冲，属预期）。
15. **[#6] R-class 复用合同**：per-event 喂 `classify_event` 的 metrics dict 含
    `{event_detected, runaway, returned, sustained_front_score, active_peak, r95_ea, far_ea}`，caps == A1
    `DEFAULT_CAPS`（R95_CAP=6, FRONT_THRESH=0.5）；测试断言已知 metrics → 期望标签。

---

## 8. 风险 / 诚实边界 / 下一档

### 8.1 占位参数

`q_min=0.25, τ_rec=5000, τ_a=100, k_use ladder` 全 PLACEHOLDER（`slow_vars.py` 既有 banner 同理）。
**任何 A2 结论都 contingent on 这组未生物标定的时间常数/强度**。pilot 报"在这组量级下"。

### 8.2 回落机制风险（**最关键**）

对称 2 油箱单纯耗竭是**去抑制正反馈**（漏→更兴奋→更漏），ODE 自身不带"发作样后回落"。R-excursion 的回落
**依赖涌现**：发作样 bout 驱动高活动 → 衬底快事件自限 + bout 后**网络瞬时静默**（refractory/exhaustion）
→ 静默期油箱回灌 → `rho` 落 → 回间期。**这是经验问题，pilot 才知道**：若实测 R-runaway（漏过线不回），
**不是 bug，是预注册结果**，含义见 §8.4。**[review-fix #5] 因果方向**：回灌**跟在**活动下降之后（不是领先）——
快事件自限先让活动掉、`q` 才回灌；故 offset 不能当"慢变量主动终止"的证据（§4.3.3 offset claim）。

### 8.3 `rho` 是模型坐标

`rho`/`local_global_ratio` 是地形坐标非生理量；边界是 Task-0b 在本锚标的**描述性**分界（起点 A1b lgr）。
"穿边界"是对着模型坐标，不是绝对生理阈。

### 8.4 预注册"下一档配料"（pilot NULL/runaway 的指定下一步，**不在本 pilot**）

- R-runaway（无回落）→ 加一个**对抗回落**慢过程：(a) 全局那半改 A1c 式负反馈刹车与局部耗竭**竞争**
  （fork#2 的另一支，要碰 `kick_probe`，M3B 协调后做）；或 (b) 引擎现成 `g_K` sAHP（活动累积、慢衰、
  发作终止子）作 recovery 项 = 经典 Liou/Abbott 兴奋+恢复对。
- R-stay（太弱）→ 加 `k_use` / 降 `τ_rec` / 或加 surround-ring 第三油箱（"核周刹车塌陷")。
- **均不在 pilot**：pilot 只对称 2 油箱、tiny。配料是 NULL 后交用户的下一轮。

### 8.5 与 M3B 隔离

A2 改 `slow_vars.py`（M3B 不改）+ 新文件（`src/sef_hfo_a2.py`、runner CLI、analyzer、tests、results dir）；
parity 自包含无 fixture；**[recon-fix] 与 M3B 零共享文件**（`slow_vars.py` 不在 watch list、无 re-bless、不碰
`engine_versions.json`；R-class 经 `src.topic4_propagation_operator.spatial_bins` + `src/sef_hfo_a2.py` 自算，
**不 import M3B 正改的 `run_m3_kick_calibration.py`**）。⇒ **新 A2 worktree off 当前 hub**，slow 路完全独立。

---

## 9. Pilot 范围（[review-fix] 分阶段 + 控制 + 失败分支，pilot-first 硬停）

公共：`T≈20000 ms`（≈4×τ_rec，够看 漏→bout→回灌 慢周期；A1c T=8s 太短），`τ_rec=5000` 单值，逐 seed 报、不 pool-p。

### 9.1 任务顺序（每个 Stage 间硬停 + 评估）

- **Task-0（anchor）**：三候选锚满箱（`k_use=0`）绝对-tail 复核 → 锁主锚 + 归因参照锚 `l0_g1.0`（§3.2）；
  顺带记录各锚 baseline `ā_core`（喂 Task-1 的 `k_use` 反推）。
- **Task-0b（frozen-q boundary，[review-fix #1]）**：主锚上**冻** `q`，跑油箱积 `q_core·q_global ∈
  {1.0, lgr/1.35, lgr/1.86}` × 2 seeds，看是否复现 间期/发作样/失控。**锁定本锚 A2-本征 `rho` 边界**（§4.1）。
  〔**Stage A** = Task-0 + 0b ≈ 3 + 3×2 = 9 runs；硬停门：基线 clean？边界可标？〕
- **Task-1（core-only dynamic，[review-fix #2A]）**：`q_global≡1`，只开 `q_core`，最干净测"核局部 use-dependent
  去抑制"。`k_use` ladder 由 **Task-0 `ā_core` 反推**，目标取相对 Task-0b 边界 `B` 三档（略低 / 到 `B` / 略高）：
  示例 `B=1.35` 时 `rho_target≈{1.15,1.35,1.65}`；core-only `q_global=1` 故 `q_core=lgr/rho_target`，对
  `l0_g1.0` 即 `q_target≈{0.87,0.74,0.61}`（[review-fix #4]）× **3 seeds** = 9 runs。〔硬停门：有
  R-stay/excursion/runaway 清楚趋势？〕
- **Task-2（two-tank dynamic）**：**仅当 Task-1 见趋势**。开 `q_core` + `q_global`（**真全局乘子** §2.1），
  3 `k_use`(目标反推) × 3 seeds = 9 runs。〔硬停门：两油箱是否改变结局 vs core-only？〕
- **Task-3（per-core confirm，[review-fix #3]）**：**仅当 Task-1/2 出 R-excursion 或同步上升**。用
  `q_L/q_R/q_global` 在出正例的 seed 重跑（≈3 runs），断同步**不是**共享 `q_core` 假同步（左核活动只耗 `q_L`）。
- **Task-4（event-accumulation 分析，无新 run）**：在 Task-1/2 输出上做 `rho_pre → 下个 bout class`（hazard）
  + `Δq_event → 下个 IEI / collision`，回答"多次间期事件是否经慢变量积累把网络推向发作样 bout"（§4.3.3 onset）。

### 9.2 最小控制（仅高价值，不扩网格）

- **C1 `k_use=0` 满箱基线**（已在 Task-0）：动态结果的锚，必须 clean。
- **C2 frozen-q matched-static**（出 R-excursion 的 seed）：取该动态 run 的 `q_pre-bout` 或 `q_min` 冻住重跑——
  答"动态结果是否只是到了某个静态工作点"。即便"是"也不坏（动态移动坐标、静态地形定表型），但叙事须分清。
- **C3 replay-q（time-shuffle，承 A1c 经验）**：把某正例 `q(t)` 轨迹记录，相位平移/时间打乱后作**外源 q** 重输入；
  打乱后 R-excursion 消失 = timing/积累相关；仍出现 = 主要靠平均低抑制工作点。防把 prevent-ignition/静态强效
  误读成动态机制。
- **C4 core-only vs global-only vs both**（一个中等 `k_use`）：`{q_core 动,q_global=1}` / `{q_core=1,q_global 动}` /
  `both` 各 1–2 seed，分清"核局部耗竭"还是"背景/全局耗竭"在驱动。

### 9.3 预注册失败分支（pilot 信号后交用户，**不在本 pilot 自动续**）

- **R-stay（rho 不过线、表型不变）**：先查 `q_min` 是否到目标区间——没到 = `k_use` 太弱/baseline 活动不足 →
  提 `k_use`、降 `τ_rec`、换近边界锚；**若 `q` 已过线但表型不变** = A1b 边界不适用于 A2 实现 / `rho` 非对坐标
  （Task-0b 应已先抓到）。
- **R-runaway（过线不回）**：很可能（对称耗竭=正反馈）。**不调 `k_use`，改加 recovery 机制**：`g_K` sAHP 或
  空间局部/surround 的 A1c 式负反馈（不再均匀全局），见 §8.4。
- **只率增、无表型迁移**：判 null（= 旧 static-vth μ 负结果重复），不能用率增写成功。
- **A2b dynamic e_GABA fallback**：若 Abbott-LG 只给 R-stay/R-runaway，下一条明确为 dynamic e_GABA / Cl⁻ 累积
  （A1a 示去极化 GABA 是最强静态同步候选）；与 `q` 油箱互斥，**不进本 pilot**，列正式 fallback。

### 9.4 pilot-first 硬停 + 规模

**Stage A（Task-0+0b ≈9 runs）→ 停评估 → Task-1 core-only（9 runs）→ 停评估**。这 ≈18 runs 是**第一可执行块**；
Task-2/3 + 控制 **gated on Task-1 见趋势**，由用户在 Stage 评估点决定续不续。**不做**：大网格、surround 第三
油箱、competition 刹车、`g_K` recovery、5-seed full、多 τ_rec/τ_a 扫——全是 pilot 信号后的下一轮。

---

## 10. 工程纪律 / 路径 / 交付

- **Worktree（实现期）**：新建 A2 worktree off `topic4-snn-m3-hub` HEAD（`using-git-worktrees`）；
  engine 已在 `src/snn_engine/` tracked ⇒ worktree 可行。
- **Results**：`results/topic4_sef_hfo/m3a_slowvars/a2_abbott_lg/`（与 `a1c_pilot/`、`a1b_grid/` 同级），
  含 `figures/README.md`（中文逐图，AGENTS.md 规范）。
- **图**（pilot 后目视）：(1) `rho(t)` + q_core/q_global + global/core/surround rate 时序；(2) 状态
  timeline（间期/发作样/失控带）；(3) 事件 phenotype scatter（油箱值 vs sync/r95/tail）；(4) `k_use`
  ladder 的 R-stay/excursion/runaway 汇总；(5) 代表 core_model 四联。
- **Docs**：本 spec（archive）；实现计划 → `docs/superpowers/plans/2026-06-25-sef-hfo-m3a-a2-abbott-lg-plan.md`；
  pilot 后 recap → `m3a_a2_abbott_lg_pilot_recap_<date>.md`。
- **Abbott 合同来源**：handoff Abbott-LG 方程 + A1b/A1c archive（`docs/paper/abbott_model.md` 物理缺失，
  dangling；不引其行号）。

关联：[[m3a_quasistatic_slowvars_recap_2026-06-24]]、[[m3a_a1b_state_topography_2026-06-25]]、
[[m3a_a1c_pilot_recap_2026-06-25]]、[[a1c_dynamic_global_feedback_spec_2026-06-25]]、
[[project_topic4_sef_hfo_observation_layer]]、[[project_topic4_m0_boundary_audit_2026-06-18]]。
