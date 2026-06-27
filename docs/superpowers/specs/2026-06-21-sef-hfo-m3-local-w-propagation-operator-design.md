# M3：W-kicked 标定的局部传播算子 + W-耦合慢易感度 (local-W propagation operator with W-coupled slow permissivity) — Design Spec

> 状态：DESIGN（2026-06-21，用户 review M3-final 整合版后定稿；待 writing-plans 同步出实现计划）。
> **取代** `docs/superpowers/specs/2026-06-19-sef-hfo-m3-hub-gated-critical-scaffold-design.md`（hub 门控版降级为已记录退路，见 §1.3 + §10）。
> 上游：M2 收口 `docs/archive/topic4/sef_hfo/m2_stage_recap_2026-06-19.md`；hub 第一轮实测
> `docs/archive/topic4/sef_hfo/m3_hub_scaffold_infra_status_2026-06-19.md`（worktree `.worktrees/topic4-m3`）；
> Task 0 数据审计 `results/topic4_sef_hfo/event_extent_audit/cohort_summary.json`（AF=0.915、LR=0.561、contact-space sampling-dominated）；
> 联合判据 `snn_5criteria_joint_verdict_2026-06-18.md` §7。框架合同 `docs/topic4_sef_hfo.md`（工作点 lock 2026-06-03、§7.2 红线、tier 纪律）。

---

## 0. 一句话承诺（朴素表述，CLAUDE.md §8）

**要做什么**：让一个一个细胞放电的网络自己长出"间期 HFO 群体事件"，并同时满足——① 自发冒出来；② 时间上自己停；
③ **不持续招募**（点一下传一段、然后整片回到安静，而不是变成停不下来的持续放电）；④ 能沿一条固定路线传；
⑤ 正反双向——而且能在一个慢变量（"易感度"）升高时，从"间期的局部事件"切换到"发作样的持续招募"。

**怎么做**：**不再人为指定走廊 / 端点 / 枢纽**。先在当前局部各向异性放电网络里，**轻轻踢一下、看活动往哪传、传多远**，
用这个响应测出一个可观测的**局部传播算子 W**（记 `W_kicked`）。间期事件 = 这条传播场在**低易感度（亚临界）**状态下的
自发、时间自限传播 excursion——点着、沿 W 主轴传一段、然后整片回到静息；M1 兴奋疲劳让每次事件后传播增益短时下降，
把事件在**时间上**收住。慢变量 `m` **不造新路径**，而是通过一张从 W 读出的"易感度地图" `h` 调节**同一条传播场**的
有效招募增益——把系统从"点一下自己熄"（亚临界）推到"点一下停不下来"（超临界）。

**揭示什么（预期判据）**：只有当 (a) 这个"亚临界、能时间自停、还能读出方向"的间期工作窗**确实存在**；
(b) `W_kicked` 能预测自发事件的传播先后顺序、且**优于纯几何距离 / 纯放电率**；(c) 同一条场在易感度升高后能**相变**到
持续招募、且持续招募仍沿同一条 W 主轴；(d) 对照实验（把 `m` 改成全场均匀、或把 `h` 空间打乱）**不能**复现这种
"轴向被保住"的相变——才算这套 local-W 机制看起来成立。任一条垮，老实记录，并退回已写好的 hub 退路（§10）。

> （内部归档代号：M3-final = local-W propagation operator + W-coupled slow permissivity；W_kicked = small-kick response operator；
> W_0^eff = μ=0 baseline **bundle** {W_resp, W_step, W_shape}（denoise(W_kicked@μ=0) 的三派生对象，**不是单一矩阵**；见 §5.2/C6）；h = W-derived susceptibility/recruitability map（**从未归一 W_resp 算**）；μ=β·m = permissivity scalar；
> Λ₀ = ρ(W_step)（**非** ρ(W_0^eff)；W_step = bundle 里按源活动归一的那个对象）；criticality = 招募/传播算子分支比 ρ≈1，**非** resting-state max Re λ；上游 M2 = dynamic inhibitory gate（放电层失败）、
> M1 = E→E recovery（成立）；hub 门控 = 已记录退路 fallback。）

---

## 1. 背景与动机（为什么从 hub 转 local-W）

### 1.1 三头夹击：空间墙立不住

**(a) M2 放电层硬顶（确定）。** 前沿抑制刹车在平均率场里能把波停在固定小范围，但搬到 cm 放电薄片**无效**：全或无波前
一旦点着就沿轴铺到组织边界；平均率场那种"波平滑传一小段就衰减熄灭"在放电层**根本不存在**（`m2_stage_recap_2026-06-19.md` §4）。

**(b) Task 0 数据侧约束（n=23）。** 真实间期事件在采到的范围里**沿轴铺满约 92%**（AF_median=0.915），整个足迹**≈ 按电极杆
随机抽样的零分布**。这是 contact-space 事实：足迹主要由电极采样决定。它的作用是**"沿轴短"既不被数据支持、数据本身也判不了**——
模型读出做成"沿轴一小段"反而算错（详见 §6 Layer 2）。

**(c) hub 第一轮实测（worktree `.worktrees/topic4-m3`，已跑、已提交）。** 旧 hub 版搭起来跑过第一轮，结论恰好是空间墙立不住的实测：
- **空间拉开距离不 gate**：固定走廊只把外围往外推（间隔 0→0.4），漏到外围只降约 **6%**（`gap_sweep/`）。
- **各向同性对照坐实机制**：把连接改成各向同性（AR=1）同样推间隔，漏**大降约 48%**——说明**让事件沿轴传的那条各向异性连接，
  正好把活动桥到外围**；你要沿轴传（④）就挡不住它（`8f41b46`）。
- **门控机制与双向打架**：那条"连得越多门槛越高"的度归一化门一开（alpha>0），**正反双向就基本塌成单向**（⑤ 退化）。

### 1.2 换承重机制 + 重定义"自己停"

把承重点从"空间墙（走廊 + 关着的枢纽）"换成"**同一条局部各向异性传播场 W + 慢易感度 μ 把它从亚临界推到超临界**"。
与此同时**重定义第③条**：

- 旧（hub）：③ = 事件**空间上**死在枢纽前、不外溢（structural containment）。
- 新（local-W）：③ = 事件**时间上**回到静息、不演变成持续招募（**subcritical temporal self-limitation / non-sustained recruitment**）。
  事件**沿轴铺多远是次要相关量、不是判据**——这与 Task 0（沿轴铺满、像随机抽样）自洽，也绕开 M2 那道"沿轴挡不住"的硬顶。

承重机制改成：**M1 兴奋疲劳**在低易感度时把事件**在时间上**收住（已成立）；**易感度 μ 升高** = 事件逃出 M1 的收住能力 → 持续招募。
间期/发作之别从"小走廊 vs 大范围"变成"**自终止 excursion vs 自维持招募**"，由临界量 ρ 越过 1 控制。

### 1.3 与 hub 版的关系：hub = 已记录退路，不删

hub 门控版（结构封闭）**不删**——它的设计推理 + worktree 已跑出来的基础设施都保留。它在两种情形下是合法退路：
(i) local-W 的间期工作窗（亚临界 + 可读 + 自停）跑不出来；或 (ii) 跑出来但拿不到稳定双向模板（§9 失败口径）。
旧 spec/plan 顶部加 superseded 横幅指向本 spec。

---

## 2. 设计决策

| # | 决策 | 选择 | 含义 |
|---|---|---|---|
| D1 | 起步方式 | **直接在 SNN 上做**，不退回率场 | honoring M2 教训（率场成的不迁移）。临界只定义在传播/招募算子层。 |
| D2 | W 的地位 | 机制对象 = `W_step`；**操作估计量 = `W_resp`（small-kick 响应）** | SNN 非线性，raw 连接矩阵 ≠ 真传播偏好。用 small-kick 直接测响应 `W_resp`（含 E→E 轴、E/I 平衡、局部阈值、M1 当前态、非线性）。**关键（审阅 §1，2026-06-21）：W 拆成三个对象，不可混用一个 row-normalized 矩阵**——`W_resp`（未归一，给 `h`）→ `W_step`（按源活动归一，给 `Λ₀=ρ`）→ `W_shape`（行/列归一，给主轴 + 顺序预测）。详见 §5.2。结构 `W_struct` 作便宜先验 + 交叉校验。 |
| D3 | 慢变量耦合 | μ **通过 `h(W)` 耦合**，不是全场均匀降阈/去抑制 | `m` 不造新结构，只调同一条 W 的有效招募增益。`h_i` = 从 W 读出的局部 recruitability。 |
| D4 | "自己停"定义 | **时间/临界**（回静息、不持续招募），**非**空间沿轴短 | 见 §1.2 + C3。 |
| D5 | 代码基底 | **复用现有 worktree `.worktrees/topic4-m3`** | 复用已跑出的 degnorm（=h 场结构先验）、branching_ratio（=ρ）、m3_acceptance（=Layer-2）、hub_diag（=reach 读出）、runner 的 `V_th_per_neuron` 接线。hub 专属件（长程边/crossing/hub-θ）留着但不进 local-W 主线（§10）。 |

---

## 3. 边界约束（claim discipline — 写进 Global Constraints，违反 = 静默科学污染）

**C1. 「临界」只能指传播/招募算子层，不能指静息网络线性临界。** 项目 2026-06-03 已把真 LIF 工作点锁成"稳健稳定但可激"
（自洽点 max Re λ≈−0.05），并明确更正过"近临界工作窗"的措辞。M3 的"临界"对象 = **`Λ₀ = ρ(W_step) ≈ 1`**（招募算子谱半径接近 1；`W_step` = μ=0 bundle `W_0^eff` 里按源活动归一的对象，见 §5.2/C6；**ρ 永远只吃 `W_step`，不吃 bundle 名 `W_0^eff`**），
绝不能写成"把静息网调到 max Re λ≈0"。**这一条与旧 C1 一致，且改进**：旧版从结构连接矩阵线性化算 σ；本版**直接测** W_kicked，
天然含非线性 / E-I / recovery 当前态，更接近真传播偏好。

**C2. 不引入平滑率场作主力。** `Λ_eff = ρ[D_μ(h)·W_step·D_x]` 是**线代筛查 / 便宜预测**（选工作点、画相变边界），
**SNN 跑出来的事件分类才是判据**——同框架"λ(k) 作诊断、finite-pulse 才是闸门"。ρ 相图不得当结论。若日后想加率场 mean-field，只能作旁证。

**C3. 自限措辞锁（与旧版不同，必须重写）。** 第③条 = **subcritical temporal self-limitation / non-sustained recruitment**
（亚临界时间自限 / 不持续招募）——事件在**时间上**回到静息、不变持续招募。**禁止**把它写成空间 "structural containment"（那是 hub 退路的词），
**也不再要求**沿轴短 / L 不变 / edge_margin>0 / 不贴边（旧 hub Layer 1 的那套数据约束不到的额外结构）。事件沿轴铺多远是**次要相关量**。
Layer 2 仍要求模型虚拟电极读出的足迹与 Task 0 真实分布（沿轴铺满、像随机抽样）**按被试等价**（§6.2，预注册容差、非"不被拒"）。

**C4. 间期→发作相变只能当"合成可行性桥"。** 框架 §7.2 红线：ictal-like recruitment 只作 synthetic feasibility bridge，
**不解释临床发作起始**。M3 的相变 = "同一条传播场在易感度改变后能否从自终止跨成自维持"的动力学可行性演示。报告禁止出现"这解释了病人发作怎么起来"。

**C5. 慢变量必须通过 `h(W)` 耦合；均匀 μ / 打乱 h 是对照不是机制。** 局部慢变量**不能**是全场均匀降阈、也**不能**是全场均匀去抑制——
否则它和传播场 W 没真正耦合。`m_i = μ·h_i`（沿场）vs `m_i = μ`（均匀）vs `m_i = μ·shuffle(h)`（打乱）是**头号对照**（§6.4）：
只有"沿场耦合"版能保住轴向 / 产生清楚相图，才说明慢变量经 W 起作用、而非全局兴奋性调参。

**C6. `W` 必须拆三对象 + 按"一步招募"标定，否则 ρ / h 都失真（审阅 §1，2026-06-21）。** 三条子约束：
- **(a) 不许用 row-normalized 矩阵算 `Λ₀` 或 `h`。** row-normalize（每行和=1）会让 `ρ≈1` 恒成立、与 `J_EE_scale`（增益旋钮）脱钩，
  且 `h_post=行和≈1`（全场抹平）——`Λ₀` 就不再是传播增益、`h` 就不再是 susceptibility。**这是 row-normalize → ρ 的隐藏陷阱。**
- **(b) `Λ₀=ρ(W_step)`，`W_step` 按"源活动质量"归一**（`W_step[p,q]=W_resp[p,q]/(源 q 的 kick 诱发活动 + ε)`）= "一个着火格点在一个传播代内平均招募多少 downstream 活动" → ρ 有分支比含义、可大于/小于 1。**不是**行/列和恒为 1。
- **(c) kick 幅度 + 一步窗必须标定**（§5.2 + 计划 Task 1.5）：响应窗 `[Δ1,Δ2]` 对应 first downstream recruitment（避开直接刺激 artifact、不含 self-sustained wave），kick 幅度落在 quasi-linear local regime（太小=全噪声、太大=直接触发全局 event）。标定值预注册后冻结。
worktree 的结构探针 `recruitment_operator` + `branching_ratio` 已按一步招募算子搭好，`W_step` 的 `ρ` **复用 `branching_ratio`** + 与结构 σ 交叉校验。

---

## 4. 架构总览

```
                ┌───────────────── cm-scale 放电薄片（无走廊/无端点/无 hub）─────────────────┐
   平滑 onset    │   局部各向异性 E→E（沿轴拉长，theta_EE/AR）+ 平滑阈值异质 + M1 recovery     │
   patch  ─────► │   ↑ 自发点火        ── 沿 W 主轴传播 ──►        ↑ 整片回到静息（间期 ρ<1）    │
                └──────────────────────────────────────────────────────────────────────────────┘
   测量：small-kick 在各空间格点踢一下 → 响应 = W_kicked → 𝒟 → W_0^eff bundle {W_resp, W_step, W_shape} → Λ₀=ρ(W_step)、主轴(W_shape)、易感度图 h(W_resp)
   间期（μ 低，Λ₀<1）：点着→沿 W 传一段→时间自停→回静息            （①②③④⑤ 在同一条场上）
   发作（μ 高，Λ_eff>1）：点着→持续/重复招募波→不回静息→沿同一 W 主轴广泛招募  （synthetic bridge, C4）
```

四个组成单元，每个单一职责、可独立测试、可独立 ablate：

- **U1 基底（substrate）**：现有局部各向异性 E→E + 一个平滑 onset patch + 平滑局部阈值异质 + M1 recovery。决定 ①④⑤。**无走廊/端点/hub。**
- **U2 传播算子 W**：`W_kicked` 测量 harness + 𝒟 去噪 → `W_0^eff` bundle（`W_resp`/`W_step`/`W_shape`）、主轴(`W_shape`)、`Λ₀=ρ(W_step)`、`h^post(W_resp)`。决定"W 是不是有用的机制对象"（§4.2 gate）。
- **U3 慢易感度 m**：静态 μ-clamp 先做（骑现有 `V_th_per_neuron`，`V_th_eff = V_th0 − Δθ·μ·h`）；第二版接 `slow_vars` φ/z。决定 ③ 的相变门控。
- **U4 虚拟 SEEG 读出层**：复用现成 pipeline（masked lagPat → 真实 propagation pipeline）。决定 Layer 2 验收。

---

## 5. 各单元详细设计

### 5.1 U1 — 基底（substrate）

**Primary 基底 = 单个平滑 onset patch**（审阅 §3，2026-06-21）+ `connectivity_rot` 各向异性（`theta_EE`, `AR`）。**不**新建走廊/端点/hub。
**`twoend_equal`（两等价灶）只作 legacy comparability control / fallback**——若单 patch 给不出双向，再用它探"双点火异质性是否足够"，**不进 local-W 主 claim**。
（理由：twoend_equal 把"两个端点"写进基底 = 变相预设 endpoint，与 D4「双向应从 W 近对称 + 单区随机点火自然涌现」冲突；主 claim 必须用单 patch，否则双向是 trivially built-in。）

- **onset patch**：`V_th0_i = V_th_base − Δ_onset·exp(−|x_i−x0|²/2σ_onset²) + ε_i`，`ε_i` = 小幅平滑随机阈值场（不是两个预设端点）。
  patch 角色只是让网络能自发点火，**不**等价完整 SOZ、**不**预设传播方向。
- **双向 / 有向怎么来**：不放两个手工端点。双向来自 (1) `W_0^eff` 沿轴近似对称；(2) onset patch 内不同 sublocation 随机点火；
  (3) M1 recovery 历史造成局部 recovery 不对称；(4) `μ·h_i` 轻度轴向不对称造成 direction bias。方向比例
  `DI = (N_{+θ}−N_{−θ})/(N_{+θ}+N_{−θ})`。
- **⚠️ 基底可行性是早期关键观测（不硬停，用户 2026-06-21 决定）**：纯基底（μ=0）必须先**自己**给足够、稳定、双向可读的自发离散事件。
  ⑤双向 + 刻板在旧 cm-SNN 是 **partial / unstable**（多次事件路线越攒越散），**不是已解决事实**。本阶段**早报这个可行性**作为后续解读语境，
  但**不**作 go/no-go 硬停——继续跑相图 / basin，最终由 §6 联合判据 + 对照给结论；若联合图景为负，退 hub（§10）。

### 5.2 U2 — 传播算子 W（三个对象，不可混用一个 row-normalized 矩阵；C6）

**测量响应 `W_resp`（未归一）**：baseline 低态（μ=0、M1 稳定 recovery）下，对每个空间 bin `q` 给**小幅** local kick
（`simulate_kick(kick_center=bin_q, KICK_BOOST=small, r_kick=small, V_th_per_neuron=baseline)`），多 seed 平均、coarse-bin：
`W_resp[p,q] = [ E(A_p(t+Δ1:t+Δ2) | kick@q) − E(A_p | sham) ]_+`（行 p=target/响应 bin，列 q=source/被踢 bin）。只做 baseline 减、取正、去对角、可加 reliability 阈值——**不 row-normalize**。
`A_p` = E spike count / population rate / HFO proxy / 虚拟电极激活（预注册 primary）。

**kick 幅度 + 一步窗标定（C6c，先于测量；计划 Task 1.5）**：对 3–5 个代表 bin 扫 `KICK_BOOST × win_ms`，定 (1) response∝kick 的近线性区；
(2) 直接刺激 artifact 窗 vs first downstream generation peak；(3) kick 是否直接触发 self-sustained wave。选 quasi-linear local regime、`[Δ1,Δ2]` 主含 first recruitment。预注册后冻结。

**三个派生对象（审阅 §1，2026-06-21）**：
- **`W_resp`（未归一）→ 易感度图 `h`**：`h_p^post = norm(Σ_q W_resp[p,q])`（target p 多易被招募；primary，因 `m`=postsynaptic permissivity）；
  `h_q^out = norm(Σ_p W_resp[p,q])`（source q 多易传出去；sensitivity）；`h^hybrid = ½(h^post+h^out)`。`norm` = 除以中位数。**h 必须从未归一的 `W_resp` 算**（row-normalize 会把 h 抹平到 1）。
- **`W_step`（按源活动归一）→ `Λ₀=ρ(W_step)`**：`W_step[p,q] = W_resp[p,q] / (src_mass[q] + ε)`，`src_mass[q]=W_resp[q,q]`（源 q 自身 kick 诱发活动）= "一个着火 bin 一代内平均招募多少 downstream 活动"。`ρ` 复用 `branching_ratio`，可 >/<1（真传播增益）。**不**行/列和恒 1。**防爆保护（审阅 P1，2026-06-21）**：低 `src_mass` 源 bin 会被小分母放大成假高增益——必须 (1) 设 `src_mass` reliability floor（`src_mass[q] < floor` 的源 bin **整列排除**、不参与 ρ，记 `valid_src` 掩码）、(2) 主口径分母 = `src_mass`，**sensitivity 口径** = 除以"注入期望 spike mass"（kick 注入 q 的预期 spike 量，对低响应源更稳健）；两口径 `ρ(W_step)` 跨 `J_EE_scale` 的相对排序须一致才算稳。floor 值 + 排除了哪些 bin 随结果一并报。
- **`W_shape`（行/列归一）→ 主轴 + 顺序预测**：仅用于 `principal_axis` + `ordering_predictivity` + 方向场。**不**作 `Λ₀` 主对象。

**层级错配警告（CLAUDE.md §6.1）**：worktree 的 `ee_degree(net, scheme)` 给的是**结构连接**导出的 h（先验 / 次要）；本版 primary `h` 必须从 **measured `W_resp`** 来——
两者形状同、问题不同（结构度数 vs 实测传播场嵌入），**别混用**；`ee_degree` 只作 `W_struct` 路径先验（`W_struct` = 现有 net + gap_factor，复用 `recruitment_operator`，便宜预筛 + 交叉校验）。

**binning / kick-radius sensitivity（审阅 §10）**：`W_resp` 是 coarse-bin 算子，须证非 binning artifact——扫 `n_bins_per_axis ∈ {4,5,6} × r_kick ∈ {small,medium} × edge-bin 含/不含`，
要求 **principal axis 稳定、h-map rank 相关稳定、`ρ(W_step)` 跨 `J_EE_scale` 的相对排序稳定、`ordering_predictivity` 方向一致**（不要求矩阵逐元相同）。

### 5.3 U3 — 慢易感度 m（先静态、后动态）

**静态版本先做（最少改引擎，骑现有 `V_th_per_neuron`，μ=0 逐比特一致）**：
- `μ = β_m·m`（静态 clamp）；`W_eff(μ) = D_μ(h)·W_step·D_x`，`D_μ(h) = diag(1+μ·h_i)`；`Λ_eff = ρ[D_μ(h)·W_step]`（线代预测；**Λ_eff 只吃 W_step**）。
- 第一版工程实现 = threshold / permissivity（最少改引擎）：`V_th_eff_i = V_th0_i − Δθ·μ·h_i`（调 `simulate_kick` 前 `V_th_per_neuron` 减一个 `μ·h` 增量）。
- 第二版 = inhibitory-restraint：`g_{I,i}^eff = g_{I,i}^0·(1−β_I·μ·h_i)`（或 `J_{IE}` 减弱）。`slow_vars` 参数是占位值、接前先标定。
- **⚠️ μ 经阈值耦合的已知风险**：μ via V_th↓ **同时**改"点火率"和"传播增益"（和 `drive` 一样的混淆——这正是 §6.1 不拿 drive 当 x 轴的理由）。
  对 y 轴可接受（易感度本就该两个都动），但"μ 真耦合到场"**只能靠对照证**（C5 + §6.4）。另：worktree 实测度归一化（按度数改阈值）一开就把双向打塌；
  本版 μ-耦合也是"按 h（≈度数）改阈值"、只是符号相反——**会不会同样扰双向是未知数，pilot 专门盯。**

**`h` 选择预注册（不许看结果挑）**：primary `h^post`；secondary `h^hybrid`；negative control `h_i=1`（均匀）+ `shuffle(h_i)`（打乱）。

### 5.4 U4 — 虚拟 SEEG 读出层（复用现成 pipeline）

复用 `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py`（虚拟电极 montage → `*_lagPat_withFreqCent.npz`）+
`scripts/run_model_contact_plane_readout.py::build_model_record`。模型事件经**真实** masked lagPat propagation pipeline 读回方向 / 端点，
再喂 §6 Layer 2 的 Task 0 指标。**禁止只看 raster 下结论。**

---

## 6. 五项 + 相变的机制映射 + 验收

### 6.1 机制映射

| 要求 | M3-final 靠什么 | 验收锚点 |
|---|---|---|
| ① 自发 | 亚临界但可激基底（`Λ₀<1`）+ onset patch + 噪声 | quiet rest + 自发离散事件（非 tonic / 非 silent） |
| ② 时间自限 | 兴奋疲劳（保留 M1 E→E recovery `ee_std_u`） | 事件有限时长、事件间安静基线、`Λ(t)` 事件后短时回落 |
| ③ **不持续招募** | 低 μ 时 `Λ_eff<1`（亚临界）→ excursion 回静息（**时间自限，非空间封闭**） | Layer 1：事件**时间上**回静息、不变持续/重复招募；事件间整片安静 |
| ④ 可传播 | 局部各向异性连接（U1）→ W 沿轴 | 沿 W 主轴清晰传播、fwd/rev 可读 |
| ⑤ 刻板 + 双向 | W 近对称 + onset 内随机点火 + recovery/μh 不对称 | 双向 fwd/rev 干净、多次事件路线稳定（不越攒越散） |
| 相变 | μ↑ 经 `h(W)` 抬有效招募增益 → `Λ_eff` 越过 1（U3，C4 合成桥） | basin `K_min(μ)` 下降 + SNN 确认从自终止切到自维持、且沿同一 W 主轴 |

### 6.2 两层验收

- **Layer 1（时间自限 / 不持续招募 — 重定义，C3）**：**三层都报，不得只看可读事件**——(a) 全部组织事件、(b) 可读子集 `n_part≥7`、
  (c) 局部不可读子集。每事件量：事件时长、事件后是否回静息、事件间是否安静、是否变 tonic/持续招募。**Layer 1 PASS = 事件在时间上 FINITE +
  回静息 + 事件间安静（非 tonic、非持续招募）。空间足迹（reach_axis_mm / r95）只作描述报告、不作 PASS 判据**（不再要求沿轴短 / L 不变 / 不贴边）。
- **Layer 2（虚拟电极足迹 ≈ Task 0，subject-level 等价检验）**：复用 `src/topic4_m3_acceptance.py`。**不用 event-level KS/MW 的"不被拒"当 PASS**——
  Task 0 reference 是 9145 event 嵌套 23 subject（非独立），"不被拒"是低功效。**用 subject-level / hierarchical bootstrap**：每个模型"被试"
  （一个网络实现 / seed）算 subject-level median AF、median LR、obs−null gap，要求**都落在真实 subject 分布的预注册容差带内**（等价检验 / TOST，
  PASS = 在容差内、非 p>α）。容差带由 Task 0 真实 per-subject median 分位距预注册。**与旧 hub 版不同：因为 ③ 不再要求空间短，模型事件沿轴铺满
  ≈ Task 0 反而是 local-W 的自然行为、是优点；这里检验的是"足迹分布形状对得上"，不是"模型读出要短"。**

### 6.3 相图与 basin（机制核心证据）

**主相图 `Λ₀ × μ`**（不是 `AR × m`——AR 是方向性参数、不是相变主控量）：
- `x = Λ₀ = ρ(W_step)`：工程上用**不改几何**的旋钮实现——primary `J_EE_scale` / `ee_gain_scale`；secondary `C_EE_scale`。
  **不**用 `l_EE`（易在 flood / collapse 间跳）、**不**用 `drive`（同时改点火与传播）。`AR` 固定（如 AR=2, θ=45°）作控制。
- **`Λ₀` 只在 μ=0 测一次（关键，避免 x 轴自漂）**：每个 ρ 旋钮值下，μ=0 测一次 `W_kicked` → 得 bundle `W_0^eff`（`W_resp`/`W_step`/`W_shape`）→ 定该列 `Λ₀=ρ(W_step)`（x 坐标）+ `h(W_resp)` 场；
  各 μ 用**同一个** baseline bundle 算 `Λ_eff=ρ[D_μ(h)·W_step]`（线代预测）+ 跑 SNN（真判据）。**别在每个 μ 重新踢。**
- `y = μ = β_m·m`。每点：跑 spontaneous SNN → 真实 lag/rank/KMeans pipeline → 分类 R0–R4 → 看 `Λ_eff` 是否预测 phenotype。

| 区域 | 条件 | SNN 表现 | 解释 |
|---|---|---|---|
| R0 | Λ₀ 低、μ 低 | silent / 仅噪声 | 传播场不足 |
| R1 | Λ₀ 中低 | 局部 HFO，但不可读 / 轴不稳 | 可点火但传播不可再生 |
| R2 | Λ₀ 合适、μ 低/中 | 自发、时间自限、刻板传播 | **interictal target** |
| R3 | Λ₀ 合适、μ 接近临界 | 大事件、长传播、时长增、failed recruitment | preictal-like |
| R4a | μ 高、沿 W 展开 | **W-aligned** 持续 / 重复招募波 | **seizure-like recruitment（唯一支持"同一场超临界展开"）** |
| R4b | μ 高或 Λ₀ 高、无传播结构 | 全场长时高 firing、无方向 | nonspecific tonic runaway（**不算 seizure-bridge**） |

**R4 必须分 R4a/R4b（审阅 §4，2026-06-21）**：只有 **R4a**（沿 W 主轴的持续招募）支持"同一传播场发作样展开"；R4b（全局强直 runaway、无传播结构）是退化态，**不得**当 bridge。
重点不是找漂亮点，而是 **R2→R3→R4a 是否连续、且 R4a 沿同一 W 主轴**。

**数值判据（预注册，避免 R2/R3/R4 主观，审阅 §5）**：群体活动 `A(t)=N_E^{-1}Σ_i s_i(t)`，baseline `μ_A,σ_A`。
- **return-to-baseline（间期 / 时间自限）**：事件后 `A(t) < μ_A + z·σ_A` 持续 `T_quiet`（预注册如 `z=2`、`T_quiet=200–500ms`）。
- **sustained recruitment（发作样）**：`A(t)>μ_A+z·σ_A` 持续 > `T_sustain`，**或**事件后无 quiet interval，**或**重复波 inter-wave interval < `T_gap`。
- **tonic/runaway 排除**：全场 firing 长时高于 baseline 且**无传播结构** → R4b，与 R4a 分开。
- **早相轴对齐（审阅 §6）**：R4a 的"沿 W"用 **early recruitment axis**（前 X ms / 前 Y% 招募 bin）对齐 `W_shape` 主轴 `θ_early≈θ(W_shape)`（轴只吃 `W_shape`），
  **非** whole-event 对齐——late phase 常同步化 / 全局化，不应惩罚机制。与真实数据"间期场 ≈ 发作早期激活粗空间场（非整场 replay）"一致。

**condition-on-ignition 分析（审阅 §7 + P1，2026-06-21，每个 μ 都做两套）**：μ via 阈值压低**同时**抬高点火率 + 传播增益。要证 μ 在门控 **W 的传播能力**（非只让网络更易点火）：
- (A) **spontaneous mode**：记 event rate、phenotype、R0–R4 分类。
- (B) **conditioned propagation mode（主口径 = fixed finite-pulse）**：**主分析**用**相同 onset patch + 相同 finite kick** 让每个 μ 的 ignition 完全相同（消除 onset location / event-selection bias），再比 propagation reach / duration / return-to-baseline / 早相轴对齐 / `P_escape(K,μ)`。**secondary**：只取初始 10–20ms active mass 匹配的自发 event 作旁证（仍可能混入 onset 位置 / 选择偏差，**不作主口径**）。这样即使 μ 提高 event rate，仍能证它改变了 **same-W propagation / escape probability**。

**第二相图 `recovery × μ`**（固定 Λ₀ 在 R2/R3 边界）：`y = R_rec`（`U` / `τ_rec` / `U·τ_rec`）。回答："同一条场上，慢易感度增强时，
强 recovery 能否把事件拉回静息？弱 recovery 是否让它逃逸成持续招募？" **这把'促传播慢变量'和'自限恢复机制'放进同一框架，是 ③ 的承重判据**（与主相图并列）。

**basin / separatrix（`K_min(μ)`）**：对每个 μ，从 onset patch 给不同强度 finite kick `K`，测 `P_escape(K,μ)`，定
`K_min(μ) = min K s.t. P_escape>0.5`。解释：`K_min` 大 / 不存在 = 只有间期 rest basin、事件是有限 excursion；`K_min` 随 μ 下降 = seizure
separatrix 变近、near-critical；`K_min≈0` = 自发即进持续招募。这比"看 trace 像不像发作"强。

**post-event `ΔΛ_x` 探针（审阅 §8，证 M1 是真·快速保护项，非只在公式里）**：同一网络测 baseline `W_resp` 与 **post-event**（自发事件结束后 50–200ms 做 small kick）`W_resp^post`，
算 `ΔΛ_x = ρ(W_step^post) − ρ(W_step^baseline)`。预期 `ΔΛ_x < 0`（事件后兴奋资源耗尽 → 传播增益短时下降 → 系统短期远离发作）。
若太贵，至少记事件后固定窗内 local evoked response / event probability 的压制。直接验证 `W_eff(t)=D_m(h)·W_step·D_x(t)` 里 `D_x`（M1 recovery）的快速负反馈。

### 6.4 消融（证明机制成立、非调参）

| 消融 | 操作 | 预期 |
|---|---|---|
| 去各向异性 | AR=1 | W_kicked 主轴 + 事件方向性下降 |
| 旋转长轴 | 改 theta_EE | W 主轴 + 事件读出方向跟着转 |
| 打乱 E→E 空间方向 | 保 degree/weight、打乱空间关系 | 模板稳定性下降 |
| 去 onset patch | Δ_onset=0 | 自发事件率降 |
| 去 M1 recovery | U=0 或 τ_rec→∞ | 时间自限下降、runaway 增 |
| **均匀 μ** | h_i=1 | 全局兴奋性效应，**轴向 specific 预测下降** |
| **打乱 h** | shuffle(h_i) | 相图与 W 不再对齐 |
| **reversed h** | h_i→1−h_i | 持续招募不再沿原 W 场 |
| threshold vs inhibition 实现 | V_th gating vs g_I weakening | 检查是否依赖具体实现 |
| 固定 μ=0 | 不允许易感度增加 | 有间期事件、但无相变 |

**头号消融（C5）：`D_m(h)` vs `D_m(uniform)` vs `D_m(shuffled-h)`**——直接检验核心担心：慢变量是否真和局部传播场耦合。
看的是**轴向 / 方向有没有被保住**，不是"能不能搞出发作样活动"。把它设成头号判据、不埋消融表尾。

### 6.5 真实数据侧预测（data-facing predictions，审阅 §11 — 提前写进 spec，避免纯 synthetic）

模型不是只跑 raster；它对**真实数据**做以下可证伪预测（探索 / 次要 tier，**不**违反 C4：是间期/preictal 结构对应，**非**临床发作起始因果）。
正式验收在 Phase 2（真实 readout），但预测在此**预先声明**：

1. **传播场 > HFO 率**：间期传播场（`W_shape` 主轴 / 顺序）对**发作早期激活**的预测应**优于** HFO 率地图。
2. **preictal 的 `m`-like 漂移**：preictal 事件应表现更长 duration、更大 active count、更高 propagation completeness、更低 relay failure（亚临界→近临界）。
3. **seizure-aligned mode bias**：若患者有正反两套模板，preictal / 高风险态下**更 seizure-aligned 的方向**应增加。
4. **post-event 保护 vs 促发**：大事件后短期 event probability / reach / completeness 是降是升，对应 `ΔΛ_post < 0`（保护，M1 主导）vs `> 0`（促发，慢易感度主导）。

这些把 synthetic 机制锚到真实队列，是 M3 从"工程可行性"升级到"机制实验"的桥（与 Topic 5 早期招募 / 轴对齐工作衔接）。

---

## 7. 分阶段交付（pilot-first，每阶段 go/no-go；存在性早报不硬停）

- **M3-0 冻结最小假设 + 预注册**（无动力学）：写本 spec 的最小约束（no corridor/endpoint/hub；W_0^eff via W_kicked；μ via h(W)；ictal=synthetic bridge）。
  预注册（出任何 SNN 动力学结果之前落盘）：`h` 三方案比较 + 主口径（h^post）、`A_p` primary 选择、响应窗 `[Δ1,Δ2]` 一步约定、Layer-2 容差带（复用 Task-0 per-subject median）、R0–R4 分类阈值。结果出来不许改。
- **M3-1 baseline 基底**（复用 worktree 基底）：固定 `drive≈0.6 / θ_EE=45° / AR=2 / M1 recovery on / 一个 onset patch`。输出自发事件率、return-to-baseline、时长、reach、方向、DI、KMeans 模板稳定。
  **早报基底可行性（含 ⑤双向稳定度）作语境，不硬停**（§5.1）。
- **M3-1.5 kick 幅度 + 一步窗标定**（C6c）：3–5 bin 扫 `KICK_BOOST × win_ms` → 选 quasi-linear local regime + first-generation 窗；标定值喂 M3-0 预注册冻结。
- **M3-2 估 `W` 三对象**（核心 gate）：small-kick harness → `W_resp`（→`h^post`）/ `W_step`（→`Λ₀=ρ`）/ `W_shape`（→主轴、顺序）+ binning/r_kick sensitivity。
  **Gate**：`W_shape` 预测自发事件传播先后顺序须**优于纯距离 / 纯放电率**——否则 W 不是有用机制对象（记录、并审视退 hub）。
- **M3-3 静态 `Λ₀ × μ` 相图**：`J_EE_scale × μ` 网格；每列 μ=0 测一次 `W_resp/W_step/h`；各 μ 算 Λ_eff + 跑 SNN（spontaneous + conditioned-on-ignition 两套）+ 真实 pipeline 分类 R0–R4（含 R4a/R4b、早相轴对齐）。
  **Gate**：是否存在 R2、R2→R3→R4a 是否连续、R4a 是否沿同一 W 主轴、均匀-μ / 打乱-h 对照是否不能解释。
- **M3-4 `recovery × μ` 相图**（③ 承重判据）：固定 Λ₀ 于 R2/R3 边界，扫 `μ × τ_rec/U`。强 recovery=间期自限、弱 recovery=near-critical/持续招募。
- **M3-4.5 post-event `ΔΛ_x` 探针**：事件后 50–200ms small kick 测 `ρ(W_step^post)−ρ(W_step^baseline)`，预期 <0（证 M1 快速保护，§6.3）。
- **M3-5 basin `K_min(μ)`**：finite-pulse escape；输出 `K_min(μ)`、escape 曲线、escape 事件轴 vs W 主轴。
- **M3-6 动态 `m_i(t)` pilot**（仅在 M3-3/M3-5 成功后）：`τ_m·ṁ_i = −(m_i−m_0) + η·h_i·[W_step·a(t)]_i − χ·a_i(t)`；
  比较 recovery-dominant（事件后 Λ↓、远离发作）vs permissivity-dominant（事件后 Λ↑、事件渐大）vs balanced。输出 `ΔΛ_post`。（动态 m 里 `η·h_i·[W_step·a(t)]_i` 的矩阵-向量积只吃 `W_step`。）
- **M3-7 真实 SEEG 读出验收**：复用 pipeline → lagPat / rank displacement / KMeans / split-half(odd-even) / endpoint enrichment；Layer 2 subject-level 等价（复用 acceptance）。

> **存在性早报不硬停（用户 2026-06-21 决定）**：M3-1/M3-2 的"亚临界可读自停窗是否存在"**早报作语境**，但不作 go/no-go 硬停；
> 继续跑 M3-3/4/5，最终由相图 + basin + 对照**联合判断**；联合图景为负 → 退 hub（§10），并老实记录（§9）。

---

## 8. 风险与控制

- **R1 亚临界可读自停窗可能很窄 / 不存在**（M2 反证：放电层全或无、没有平滑衰减态）。控制：①重定义自限为时间/临界（§1.2）绕开"沿轴挡不住"；
  ②M3-1/M3-2 早报存在性；③联合判据 + hub 退路（§10）。
- **R2 μ 经阈值耦合可能像度归一化一样扰双向**（worktree 实测度归一化打塌双向）。控制：M3-1/M3-3 专门盯 ⑤双向随 μ 的退化；必要时换 inhibition-weakening 实现（§5.3 第二版）。
- **R3 `W_kicked` 测量成本**（N_bins × seed 次 kick）。控制：coarse-bin、先小网格 pilot；结构 `W_struct` 先验先筛。
- **R4 ρ 当临界量的构造陷阱**（C6）。控制：响应窗一步约定 + 镜像 `recruitment_operator` 构造 + 与结构 σ 交叉校验。
- **R5 动态 m 需新引擎件**（"沿 W 累积活动"状态变量）。控制：先静态 clamp 证相图 / basin，动态 m 排在成功之后（M3-6）+ pilot-first。

---

## 9. 失败口径（写死，避免"再加结构直到成功"）

- 若 local-W 主线给不出稳定**双向**模板：**不**立刻加 hub。先写："局部各向异性 W_0^eff 足以支持自发、时间自限、有向传播 + 间期/发作相变，
  但不足以稳定双向模板；稳定双向可能需要本最小模型未施加的、患者特异的多点火异质性。"
- 若连**亚临界可读自停窗**都不存在（M2 反证成真）：记录"在当前读出 / 工作点 / 平坦阈值下，这套放电网络没有亚临界可读自停窗"，
  退回 hub 结构封闭（§10）——hub 此时是合法承重退路，不是失败。
- 一律 Level-2 收口语气：**"在当前读出 / 这些工作点 / 这套基底下"**，非"任何放电模型 / 任何参数都如此"。

---

## 10. Out-of-scope / 退路（红线，写死）

- ictal recruitment = **synthetic feasibility bridge**，不解释临床发作起始（C4 / framework §7.2）。
- 「临界」= 招募/传播算子分支比 `ρ≈1`，**非** resting-state 线性 near-criticality（C1）。
- 不引入平滑率场作 SNN 结论替身（C2）。
- 动态 `m_i(t)` 在静态相图 / basin 成功**之前**不做。
- **hub 门控（结构封闭）= 已记录退路**：spec `2026-06-19-...-hub-gated-...-design.md` + plan + worktree 已跑基础设施保留。
  仅在 §9 两种失败下启用，不在 local-W 主线。

---

## 11. 溯源（内部代号 / 路径）

- 上游：M2 dynamic inhibitory gate（放电层失败）`m2_stage_recap_2026-06-19.md`；M1 E→E recovery（成立，`--ee-std-u`）。
- hub 第一轮实测：`m3_hub_scaffold_infra_status_2026-06-19.md` + worktree `.worktrees/topic4-m3`（gap_sweep −6%、各向同性 −48%、度归一化打塌双向）。
- 数据校准：Task 0 `src/topic4_event_extent_audit.py` + `results/topic4_sef_hfo/event_extent_audit/cohort_summary.json`（AF=0.915、LR=0.561、two-layer reference）。
- 引擎：`src/snn_engine/{connectivity_rot, kick_probe, params, slow_vars}.py`；`simulate_kick(kick_center=, r_kick=, t_kick=, V_th_per_neuron=)`；guard `engine_versions.json`。
- **复用件（worktree）**：`src/topic4_degnorm.py::ee_degree`（结构先验 h）、`src/topic4_hub_criticality.py::{recruitment_operator, branching_ratio}`（W_struct + ρ）、
  `src/topic4_m3_acceptance.py`（Layer-2 subject 等价）、`src/topic4_hub_diag.py`（reach / 招募读出）、runner `V_th_per_neuron` 接线。
- **新增（本 spec）**：`src/topic4_propagation_operator.py`（W_kicked 测量 + 𝒟 + W_resp/W_step/W_shape 三对象 + h^post(W_resp) + Λ₀=ρ(W_step) + 主轴(W_shape)）；runner `--mu/--h-source/--h-control` h-耦合（骑 V_th，μ=0 bit-parity）；M3-3/4/5 相图 + basin 脚本。
- 框架合同：`docs/topic4_sef_hfo.md`（工作点 lock 2026-06-03、§7.2 红线、H4 几何不变形、tier 纪律）；联合判据 `snn_5criteria_joint_verdict_2026-06-18.md` §7。
- 文献 framing：Pachitariu/Stringer 2026 Nature（critically-normalized 连接、谱半径≈1，仅建模语言）；Moosavi & Truccolo 2023 PLOS Comp Biol（seizure spread 相图 excitability×connectivity）；Proix/Jirsa 2018 Nat Commun（局部 + 异质长程 + 多时间尺度 spread）。
```
