# M4-3 —— 病理轴 = 连续慢快坐标（sensor-free / mask-free 恢复变量）设计 spec（2026-07-09, rev2, 中文版）

> 状态：**DRAFT / 待用户 review**。rev2 按外部 review（2026-07-09, "conditional approve for rev2"）修 8 个 blocking + 数个
> 强化项，并把 review 里的定性原则**落成数值验收门 + 工程红线**（照 CLAUDE.md §6 / 「验收 gate 编码结论非存在」）。
> rev1→rev2 的承重改动一览见 §11。前置：M4-2 archive `docs/archive/topic4/m4_2_std_termination_p1_sweep_2026-07-08.md`
> （no-go + 机制 + 下一杠杆双分叉）；M4-1 pass-1 archive `docs/archive/topic4/sef_hfo/m4_pass1_divisive_shared_pool_acceptance_2026-07-09.md`。
>
> 本 spec 的所有引擎论断已核对真实代码（rev2 增补）：`slow_field.py`、`kick_probe.py`、`run_m4_dynamic_qi.py`、
> `sef_hfo_m4_termination.py` 的确切行为见 §8 的行号引用。**rev1 曾把几处"零改复用"写乐观了，rev2 据实修正。**

---

## 0. 一句话结论（第一性原理朴素话）

M4-2 证明：**快的、减法的、单一的**恢复变量（短时程突触抑制 `ee_std`）造不出"一次干净发作 → 可再触发间期"的
分离——只碎裂或压死。M4-3 换成一个 **连续存在、活动驱动、shunting（除法/电导型增益控制）** 的恢复变量 `n→a`，它正是
M4-2 no-go 指向的"更慢的离子型 / shunting 终止器"（Epileptor-2 的 Na/K-pump、TRESK 的去极化诱导 shunting、腺苷——共同点
不是"发作时才打开"，而是"每次活动都改慢变量，只是短 IED 改得少、持续发作改得多"）。

**M4-3A（committed 第一步）** 只问最小命门：**更慢 + shunting + 连续的恢复变量，能不能 clean-terminate M4-1 那个
有界持续态**（`λ_K=0`；复用 M4-2 的 `--p1-sweep` + 形态判读器，**扩**两遍 retrigger 为 early+late 两窗）。
**M4-3B / M4-3C（后续）** 才碰全景命门：把慢变量 kernel 从各向同性 Gaussian 推向**病理连接图诱导**的
`K_graph=F(W_EE)`，看能否复现数据里"间期传播模板 ≈ 发作早期宽带能量梯度"；再做完整闭环读出。**rev2 关键改动：M4-3B
不再完全 gated 在 M4-3A 成功之后**——加一个**低成本 M4-3B smoke**（3 个 `λ_K`、1–2 seed、只读出）与 M4-3A discovery
并行，避免"terminate 调很久、结果 graph-kernel 根本不对齐"的沉没成本（§9）。

---

## 1. 命门 & 重构

### 1.1 为什么不是"再加一个终止器"

M4-2 的 clean no-go 暴露的不是"疲劳强度不够"，而是机制类型不对：`ee_std` 是**快恢复**（碎裂：削完很快恢复→事件
未终止又点火）或**过强**（压死），中间没有"结束一次事件、又压住足够久不再点火"的窗口。继续"不能终止就加疲劳、
再加 gate / sensor / mask"是加法雪球，工程能跑出闭环但科学解释弱。

### 1.2 重构：轴 = 网络主导响应 / 非正规瞬态放大方向（对齐 M3B）

数据里的核心发现（Topic5 V2 phase1，cohort 承重）是：**间期 HFO 传播模板 ≈ 发作早期宽带能量的空间梯度**。
正确解释**不是**"发作时刚好也沿这条路走"，而是：**这条轴是 SOZ 内部病理网络对扰动的主导响应方向**。

**承重措辞（对齐 M3B，不要说成"轴向 eigenmode → critical Hopf"）：** M3B 谱线已立——主导本征模是**全局**的，
轴向结构在**非正规瞬态**里（骨架特异自限轴向，源空间 onset 梯度 R²≈0.87）。所以：
- **间期 HFO 传播** = 系统对短扰动的 impulse response 的 peak-timing（一个非正规瞬态放大方向）。
- **发作前 / 早期宽带能量** = 同一系统对噪声 / 局部活动的高增益响应的空间分布。
- 慢变量（`q_I`、`n/a`）**重塑的是这个瞬态在哪里、多大程度上放大**——不是"打开一个轴向本征模"。

M4-3 的目标闭环（**model 层面的假设**，数据支持程度见 §7）：
间期 replay → 轴向慢态漂移 → 发作前宽带梯度 → 有界发作招募 → 轴向恢复 / 爆后不应期 → 回到可 replay 间期。

---

## 2. 承重设计决策（锁）

| # | 决策 | 理由 / 核对 |
| --- | --- | --- |
| D1 | 恢复变量 = **连续、活动驱动、shunting** 的 `n→a`（无 ictal gate / sensor / mask）| §1；用户 2026-07-09 sign-off |
| D2 | `a` 主要作用 = **shunting / divisive 增益控制**；**铁律：`a` 绝不整体除 signed net current**（否则同时削弱 `-q_I I_I`/`-β_G S_G` 抑制 → 去抑制假象，review §1）。两条合法实现见 §3.2：**(A) 电导型 leak-shunt（首选，带 reversal，结构上不去抑制）** 或 **(B) 只除去极化驱动（current-based 后备）**。减法偏置 `-η_A a` 仅次要。 | review blocking #1；电导路已存在于 `kick_probe.py:86-88` 但在 `slow=None` 支路，M4 支路要**接线**（§8），非零改 |
| D3 | **M4-3A 先做，`λ_K=0`**（Gaussian 慢场）| 最小命门=终止；复用 M4-2 ~70% infra；graph-kernel 是 M4-3B |
| D4 | `q_I` 保留不动（局部抑制资源，pro-ictal）；**legacy `g_K` 在 M4 工作点已 hard-off**（`use_gK=False,k_K=0.0`，核对 `run_m4_dynamic_qi.py:184`）→ clean termination **不可能**来自 `g_K`，**无需**单设 g_K arm（review #7 的 ArmB 因此坍缩进 ArmA）| M4-1 已锁；核对代码后**简化**掉 review 的 ArmB |
| D5 | 判读器 `classify_termination`（mechanism-agnostic）**零改复用**；**但两遍 retrigger 需扩成 early+late 两窗**（review #4）——`run_cell_with_retrigger` 现在**单 offset**（且已是 late，≈offset+10s），要**加一个 early 短 offset 探针**，`retrigger_verdict` 把 `fail` 拆成 `attenuated`(=refractory) 与 `runaway` 两态。sweep-Pool/COW/provenance 复用 | 核对 `sef_hfo_m4_termination.py:121-165`；rev1 "两遍 retrigger 零改"**不成立**，据实修正 |
| D6 | 成功标准**分层**：M4-3A=终止；M4-3B=空间对齐（项目 primary，V2 phase1）；闭环=模型预测（§7）| 用户 sign-off；§5 sensitivity-gate 纪律 |
| D7 | 预注册 **clean no-go 合法**（照 M4-2）；**但 no-go ≠ 已证 `D_EE`**（M4-3A 仍锁 `λ_K=0` 各向同性 Gaussian 慢场；M4-2 自身 archive §5 把 deferred `g_K` arm 列为**第一**分叉、不是 `D_EE`）。no-go 只**加强怀疑**，且须先过 M4-3B smoke 才谈 `D_EE`（§5、§9）| review #9；对齐 M4-2 §4/§5 的实际措辞 |
| D8 | `n` 方程 **baseline-centered**（`ũ_n=[u_n-u_{n,0}]_+`，`u_{n,0}`=Arm0 安静间期长期均值的**固定常数**，非在线、非 ictal sensor）+ 全 clamp / denominator cap（§3.1/§3.2）| review #2；防 `a` 变常驻 tonic shunt、防数值伪象 |
| D9 | **T 分两级：discovery `T=15000`（快网格）；acceptance `T=40000`+ post-offset 静默 10–20s 无 rebound/runaway（候选 go cell 与边界 cell 必跑）** | review #5；M4-1 已知延迟失控（seed2 5781ms、某点 36.48s 才失控）——短窗会把 delayed-runaway 当 clean |
| D10 | **主分母 = Arm0 判为 `bounded-persist` 的 seed（计算得出，非硬编码）**；Arm0=`fragment`/`runaway` 的 seed（如 M4-2 的 seed 4）只作 stress/robustness，不进 primary go-fraction | review #6；M4-2 §0/§2.3 已这样分层（seed 1/3 persist；seed 4 fragment），rev2 把它变成**每 seed 跑 Arm0→分类→入选**的计算门 |
| D11 | **机制专属性 ablation（照 M4-1 除法-vs-减法）：** 候选 go cell 上跑 shunt-only(`η_A=0`) / subtractive-only(`α_A=0`，匹配平均削弱量) / hybrid。预期 shunt-only+hybrid→clean、subtractive-only→persist/suppress/fragment | review #12；与 M4-1 "真除法池 vs 等量减法 vs 冻结池"平行 |

---

## 3. 状态变量 & 方程

保留 M4-1/M4-2 的场：`q_I(x,t)`（局部抑制资源，耗竭→pro-ictal，`sigma_q=1.5`）、`S_G(t)`（除法共享池，把 runaway 压成
有界，**不负责终止**）。**M4-2 的 `ee_std` 在 M4-3A 里关掉**（`ee_std_u=0`，已被证 no-go）。**legacy `g_K` 在 M4 工作点已
hard-off**（D4），M4-3A 不动它。

### 3.1 新连续恢复变量 `n → a`（承重，无 sensor，baseline-centered）

活动驱动（`u_n = K_n^⋆ * r_E`；M4-3A 里 `K_n^⋆` = 现有 isotropic Gaussian，`λ_K=0`，`σ_n` 见 §3.3）。**baseline-center（D8）：**

$$
\tilde u_n(x,t)=\big[\,u_n(x,t)-u_{n,0}\,\big]_+,\qquad u_{n,0}=\big\langle u_n\big\rangle_{\text{Arm0 quiet baseline}}\ (\text{固定常数})
$$

$$
\frac{dn(x,t)}{dt} = -\frac{n(x,t)-n_0}{\tau_n} + k_n\,\tilde u_n(x,t) - \rho_n\,\Pi(n(x,t)),\qquad
\Pi(n)=\frac{[\,n-n_0\,]_+^{\,h}}{n_{50}^{\,h}+[\,n-n_0\,]_+^{\,h}},\qquad a(x,t)=a_{\max}\,\Pi(n)
$$

$$
\textbf{clamp（数值红线，D8）：}\quad n_{\min}\le n\le n_{\max},\qquad 0\le a\le a_{\max}
$$

- `n` = 抽象活动负荷（intracellular Na⁺ / metabolic burden / adenosine precursor / AHP-shunt 上游），**IED 也小幅推、
  持续发作大幅推**——区别来自占空比，不是 gate。
- **为什么 baseline-center（据实修正 review 的"会漂"）：** leak 项 `-(n-n_0)/τ_n` 已保证 `n` **有界收敛**（不会无限漂），
  但间期非零 firing 会把稳态 `a` 顶到一个**非零常驻值** → 分不清 `a` 是"活动负荷"还是"被参数调出来的 tonic 抑制"。
  减去固定 `u_{n,0}` 让 `a` 表征**活动的超额负荷**，安静间期 `a≈0`，`a` 可解释。`u_{n,0}` 是 homeostatic set point，
  **一次性从 Arm0 静息标定、写死为常数**，不随时间追踪状态——所以不是 ictal sensor。
- `Π(n)` = pump / conductance 激活的**连续非线性**，**不是发作传感器**。
- **[承重锁，见 §6 数值门] `n_50` 必须低到 IED 期 `a` 有可测上升**——否则 `Π` 变成"只有持续发作才跨过"的软 ictal gate，
  等于把 sensor 从后门放回来。这条 rev1 只有文字，**rev2 落成 `Δa_IED` 硬数值门（§6.1）**。

### 3.2 膜方程（`a` 主 shunting；`a` 绝不除 signed net current —— review blocking #1）

**铁律（写进代码注释 + 验收）：`a` 不得整体除 `I^{net}`。** rev1 §3.2 的写法

$$
\underbrace{I^{\text{net}}_{E,i}=\frac{\,I^{\text{ff}}_{E,i}+I^{\text{rec}}_{E,i}/(1+\alpha_G S_G)-q_I I^{\text{loc}}_{I,i}-\beta_G S_G\,}{1+\alpha_A a}-\eta_A a}_{\textbf{\;禁用：分母同时削弱了}\ -q_I I_I\ \textbf{和}\ -\beta_G S_G\ \textbf{抑制}\to\textbf{去抑制假象}}
$$

危险在于：`a↑` 时不仅削兴奋，也削**局部抑制 `q_I I_I` 与共享池减法 `β_G S_G`** → 可能反而去抑制；之后若看到
fragment/rebound/强振荡，分不清是真 shunting termination 还是 `a` 削了抑制造成的假象。两条合法实现（二选一）：

**(A) 电导型 leak-shunt（首选；复刻引擎已有的 `membrane_step` shunt 形，`kick_probe.py:86-88`）：**

$$
g_A(x_i,t)=\alpha_A\,a(x_i,t),\qquad 0\le g_A\le g_{A,\max}
$$

$$
V^{\infty}_i=\frac{I^{\text{net,undivided}}_{E,i}+g_A\,E_A}{1+g_A},\qquad
V_i \leftarrow V^{\infty}_i+(V_i-V^{\infty}_i)\,\big(\text{decay}_V\big)^{\,1+g_A}
$$

其中 `I^{net,undivided}_{E,i}` = 引擎当前 `apply_currents` 输出的净流（含 `-q_I I_I - β_G S_G`，**不再另除**），
`E_A` = shunt reversal（取在静息/抑制反转附近、**低于阈值**，复用引擎 `e_gaba`）。**为什么这样不去抑制：** `g_A↑` 把 `V`
拉向 `E_A`（静息），无论净流是净兴奋还是净抑制——净兴奋时压回静息（灭 runaway），净抑制时本就在阈下、拉到静息也
不点火。这是 review 的"推荐 A"（Chance/Reyes/Abbott 背景突触输入 divisive gain、Carandini/Heeger normalization）。

**(B) 只除去极化驱动（current-based 后备；review "推荐 B"）：**

$$
\boxed{\,I^{\text{net}}_{E,i}=\frac{\,I^{\text{ff}}_{E,i}+I^{\text{rec}}_{E,i}/(1+\alpha_G S_G)\,}{1+\alpha_A a(x_i,t)}\;-\;q_I(x_i,t)\,I^{\text{loc}}_{I,i}\;-\;\beta_G S_G(t)\;-\;\eta_A a(x_i,t)\,}
$$

$$
\textbf{denominator cap：}\quad 1\le 1+\alpha_A a\le D_{A,\max}
$$

`a` 只削**去极化驱动**（`I_ff` + 除过池的 recurrent），抑制项 `-q_I I_I - β_G S_G` **原样保留**（不被除）→ 无去抑制。
比 rev1 安全很多，但无 reversal-clamp（极端 `a` 只把兴奋清零 + 加超极化偏置，不是"拉到静息"）。

**选型指引（D2）：** 首选 (A)（忠于 D2 "shunting"、reversal 结构上防去抑制）；若 (A) 的接线 / parity 冒烟不顺，退 (B)。
两者 `a=0` 都必须**逐字节等旧引擎**。**§8 记录 blast radius：(A) 动 `slow_field.py`+`kick_probe.py` 内联 M4 支路
（须 re-bless）；(B) 只动 `slow_field.apply_currents`。** D11 的 shunt-only vs subtractive-only ablation 同时也**裁决 (A)/(B)
的机制差**是否承重。

### 3.3 `K^⋆` 与 `σ_n`（M4-3A 恒为 Gaussian；graph-kernel 是 M4-3B）

$$
K^\star=(1-\lambda_K)\,K_{\text{phys}}+\lambda_K\,K_{\text{graph}},\qquad K_{\text{graph}}=F(\tilde W_{EE})\ (\text{e.g. }\exp[\ell(\tilde W_{EE}-I)])
$$

**M4-3A 锁 `λ_K=0`**（纯 isotropic Gaussian，= 现有慢场）。**注意：`λ_K` / `K_graph` 引擎里当前不存在**（核对：grep `lambda_K`
空），是 M4-3B 要新建的东西（§9），**非**现成可调项。

**`σ_n` footprint（review #8）：** 恢复变量的**空间整合范围**默认取 `σ_n = σ_q = 1.5`（**宽**，= `q_I` 耗竭 footprint），
**不**默认继承窄的旧 `σ_K=0.5`（否则只压核心、边缘继续 rebound）。这是各向同性 Gaussian、**非** spatial mask、不告诉模型
"哪是轴"。次级 sweep `σ_n/σ_q ∈ {0.5, 1, 1.5}`（即 `σ_n ∈ {0.75, 1.5, 2.25}`）。

---

## 4. M4-3A 实验（committed 最小命门）—— 执行顺序 P0 → P1 → 40s 验收

**命门问题：** 更慢 + shunting + 连续的 `a`，能不能 clean-terminate M4-1 那个有界持续态（M4-2 快·减法·STD 做不到）？

### 4.0 P0 —— 离线 `n/a` trace 标定（**先做，不跑网络**；review #13）

跑大规模 SNN sweep 前，先拿 M4-1 Arm0 的**真实 rate trace**（4 类）离线喂 §3.1 的 `n/a` ODE，看 `a(t)` 是否满足下表；
不满足就先调 `τ_n, k_n, n_50, ρ_n, u_{n,0}`，**不浪费网络 sweep 预算**，也防 `n_50` 变软 ictal gate。

| 输入 trace（来自 M4-1 Arm0） | 期望 `a(t)` |
| --- | --- |
| quiet baseline（安静间期） | `a` 低且稳定（`≈0`，baseline-center 之后） |
| isolated IED（孤立棘波） | `a` 有**小但可测** transient（喂 §6.1 的 `Δa_IED` 门） |
| bounded persistent ictal（有界持续态） | `a` 在数秒内**明显累积** |
| post-offset quiet（熄灭后安静） | `a` **慢衰减**（给 postictal 不应期） |

P0 产出：标定好的一组 `(τ_n,k_n,n_50,ρ_n,u_{n,0},a_max)` + `Δa_IED`、`R_A`（§6.1）离线值，写进 P1 config 与 archive。

### 4.1 P1 —— `(α_A × τ_n)` 网格（discovery，`T=15000`）

- 工作点 = M4-2 同分母：`k_q=0.10, alpha_G=16`；`ee_std` off；`g_K` off（D4）；`λ_K=0`；`σ_n=σ_q=1.5`；`use_A` on。
- **主扫描平面 P1' = `(α_A shunting 强度 × τ_n 负荷恢复时标)`** @ 该有界点 + **每 seed 的 Arm0**（`a` off = M4-1 bounded persist
  基线；D10 的分母判据就从这里算）。次级：`(k_n × n_50)`、`σ_n/σ_q ∈ {0.5,1,1.5}`。
- **seed：** 至少 seed 1/3/4；**每 seed 都在同 run 内跑 Arm0（`cells[0]`, `a` off）→ 用 `classify_termination` 判 Arm0 类**；
  Arm0=`bounded-persist` 的 seed 入 **primary** go-fraction，Arm0=`fragment`/`runaway` 的（M4-2 里 seed 4）只作 stress（D10）。
- **判读复用 M4-2：** `classify_termination`（persist / terminate_clean / fade / fragment / suppress / rebound / runaway；
  full-sheet 1ms active-fraction；`terminate_clean` = 单个 ≥90%-peak 平台 ≥250ms 后陡降到静息尾）。
- **预期分类（先验，非验收）：** `a` 太弱→persist；太强→suppress；`τ_n` 恢复太快→fragment/rebound；合适→**terminate_clean**。

### 4.2 候选 go cell 的 40s acceptance（D9）+ early/late retrigger（D5）+ 机制 ablation（D11）

任一在 P1 判为 `terminate_clean` 的 cell（及其边界邻格）**必须**：

1. **40s 长跑（`T=40000`）+ post-offset 静默 10–20s 无 rebound/runaway** —— 排除 M4-1 式 delayed runaway。
2. **early+late 两窗 retrigger（§7 的 go 判据）：**
   - **late 探针**（现有单探针，≈`offset + 2×max(τ_n,τ_q)` ≈ offset+10s）：要 `reignite_bounded`（可再触发有界事件）。
   - **early 探针**（新增，`offset + 500–1000ms`）：要 `attenuated`（点不着 / 被压 = postictal 不应期），**不是** `runaway`。
   - 两探针都记录当刻 `(q_I, a)` 状态。
3. **机制 ablation（D11）：** shunt-only(`η_A=0`) / subtractive-only(`α_A=0`, 匹配平均削弱量) / hybrid，看是否 shunt 专属。

---

## 5. 三个科学锚（承重）

1. **墙可能是衬底不是终止器类型——但 M4-3A no-go 不能单独证 `D_EE`（review #9 / D7）。** M3A/M4/M4-2 三次收口"下一杠杆"。
   M4-3A 改的是**慢变量类型**（shunting vs 减法），**但仍锁 `λ_K=0`（各向同性 Gaussian 慢场）**。所以 M4-3A 若也 no-go：
   - **能说：** 在当前 Gaussian 慢场衬底上，只换恢复变量**类型**（shunting）拿不到 clean termination——加强"墙在衬底"的**怀疑**。
   - **不能说：** 已证 `D_EE` 是唯一剩余杠杆。M4-2 自身 archive §5 把 deferred **`g_K` arm** 列为**第一**分叉（Epileptor 谱系
     用 slow-K/pump 作主慢渗透终止变量），`D_EE`/衬底是第二分叉。**任何 `D_EE` 定论前，先跑 §9 的 M4-3B graph-kernel smoke
     （`λ_K∈{0,0.5,1}`）**——因为病理轴对齐可能恰恰要 `K_graph` 才出现。
2. **成功 tier 到数据（§7）。** 不把"完整发作周期闭环"当主判据。
3. **§1 措辞对齐 M3B。** 轴 = 非正规瞬态放大方向，非 leading eigenmode → critical。

---

## 6. 纪律锁（写进验收 —— 定性原则落成数值门）

### 6.1 `a` 被 IED 也真冲（sensor-free 硬门，取代 rev1 的纯文字；review #3）

对每个 Arm0 / `use_A` cell 自动做 **event-triggered `a` response**（`t_IED` = 间期 IED 事件时刻）：

$$
\Delta a_{\text{IED}}=\Big\langle a(t_{\text{IED}}{+}\Delta t_1:t_{\text{IED}}{+}\Delta t_2)\Big\rangle-\Big\langle a(t_{\text{IED}}{-}\Delta t_0:t_{\text{IED}})\Big\rangle
$$

**硬门（全部满足才算 sensor-free 合格）：**
- `Δa_IED > 0` **且** `Δa_IED ≥ 2·σ_baseline`（`σ_baseline` = 安静间期 `a` 波动标准差）；magnitude sanity `Δa_IED ≥ 0.5% · a_max`。
- `⟨a⟩_interictal < a_block`（`a_block` = 让标准 IED-kick 点不着的 `a` 水平，P0 测得）——否则 `a` 把所有 IED 压没，进不了 preictal。
- **占空比比值 `R_A`（正确的 sensor-free 判据）：**

$$
R_A=\frac{\Delta a_{\text{bounded ictal, 1s}}}{\Delta a_{\text{IED}}}\gg 1\quad(\text{起步 bar }R_A\ge 5\text{，P0 调})
$$

即：**短事件和持续发作驱动同一个 `a`，但持续高占空比对 `a` 的积分效果远大于单次 IED**——这才是"没有 sensor 也能区分"。
`Δa_IED` 若 ≈0（IED 完全不冲 `a`）= 软 ictal gate，**违反 D1，判 fail**。

### 6.2 clean-no-go 合法（预注册，D7）

M4-3A 若在参数网格 + primary seed 内无 `terminate_clean` → 报 no-go，**加强"墙在衬底"怀疑**（不是已证 `D_EE`，§5），
**不为闭环调参数雪球**，且先过 M4-3B smoke（§9）再谈下一杠杆。

### 6.3 复用不重造（据实，§8）

`q_I`/`g_K` 不动；`--p1-sweep` / `classify_termination` 复用不改；`run_cell_with_retrigger` **扩** early 窗、`retrigger_verdict`
**拆** attenuated/runaway（非零改，D5）。

---

## 7. 成功标准（分层，承重 —— 对齐数据 + postictal 生理）

**M4-3A（终止层，本 spec 主体）——go 判据改自 M4-2（review #4）：**

$$
\boxed{\text{go(cell)}=\underbrace{\text{terminate\_clean}}_{\text{40s 长跑确认}}\ \text{AND}\ \underbrace{\text{early retrigger}=\text{attenuated}}_{\text{postictal 不应期}}\ \text{AND}\ \underbrace{\text{late retrigger}=\text{reignite\_bounded}}_{\text{恢复后可再触发}}}
$$

且须：`a` on 有、Arm0 无（终止确由 `a` 带来）；跨 primary seed（Arm0=bounded-persist）有连通 go 区；过 §6.1 `Δa_IED` sensor-free 门。
**理由（review #4）：** M4-3 机制是 postictal shunting / recovery burden——真正像发作后的结果**不是**"熄灭后马上可再点"，而是
**early 不应期 + late 可再点**。若沿用 M4-2 的"熄了马上补一发就要点着"判据，会把一个**正确的 postictal 不应期**误判成 retrigger fail。
**这一层只回答"shunting 能否终止"，不回答轴对齐。**

**M4-3B（空间对齐层 = 项目 primary，`λ_K` 扫 + 负控；见 §9）：**
- **primary（数据承重，Topic5 V2 phase1）：** 模型从**自身间期事件**读出传播模板 `v_k(x)`（**不喂模板**），发作早期宽带
  能量梯度 `P_BB(x,t)` 与 `v_k` 的相关 `A_k(t)` 在 onset 附近对齐。**只有 `λ_K>0` 才出现这个对齐**（同一连接图既定快传播、
  又经 replay 重塑慢地形）= 强解释力。
- **数据"何时"口径：** V2 phase1 的答案是**发作前高平台 + 起始小上抬**，**不是戏剧性 ramp**——M4-3B 若跑出强 ramp
  是模型多产，按预测报，不硬凑 data-match。

**M4-3C（完整闭环 = 模型预测，非 data-match）：** interictal replay → preictal 梯度 → bounded ictal → postictal → 回可 replay 间期。
**发作前逐渐爬升 + 发作后沿轴恢复这两条腿，数据里 weak/negative**（V3p preictal 硬门全 0 阴；V2 phase2 发作前 state 0/16
沿轴显著；V3a 模态转移脆弱）→ **报成"模型预测，数据这条腿待验"，不 headline"复现完整发作周期"。**

---

## 8. 工程含义（复用 vs 新增 —— 全部核对真实代码）

**引擎新增（`slow_field.py` + 视 (A)/(B) 决定是否动 `kick_probe.py`）：**
- **`SpatialSlowField` 加 `n`/`a` 场**：照 `q_I` 模式（**Pattern A：`use_*` bool + `k_*=0.0` 零值旋钮 → OFF → byte-parity**）。
  config：`use_A / k_n / tau_n_field / rho_n / n50 / hill_h / a_max / alpha_A / eta_A / sigma_n_field / u_n0`；
  `__init__` alloc `self.n_field`（init 到 parity 值）+ 预算 `self._Kn = isotropic_gaussian(n,L,sigma_n_field)`；
  `step` 里 `if cfg.use_A and cfg.k_n != 0.0:` 门控 → `convolve_periodic` 驱动 → Hill `Π` → forward-Euler → `np.clip`（n/a 双 clamp）。
  **trace 用 slow-field 的 always-on 模式**（`self.trace_a_mean=[]` / `.append(float(self.a.mean()))`，照 `slow_field.py:310`
  的 `trace_qI_mean`），**不是** `kick_probe.py` 的 `dump_ee_std_trace` flag-hook（那是引擎局部 `x_dep` 用的）。
- **膜/电流（§3.2 二选一）：**
  - **(A) 电导型：** 在 M4 内联支路 `kick_probe.py:316-317`（`slow != None` 走 `Vtmp = I_net + (V-I_net)*decay_V`，**当前不调
    `membrane_step`**）里改成 `V_inf/(decay_V**(1+g_A))` shunt 形（复刻 `membrane_step` 的 `kick_probe.py:86-88`）；`g_A=α_A·a`
    须按 E 神经元位置采样（照 `qI_E = self.q_I[self._iyE,self._ixE]`，让 `slow` 暴露 `g_A_E` 或加采样法）。**动 `kick_probe.py` → 必须
    re-bless `engine_versions.json`。**
  - **(B) current-based：** 只改 `slow_field.apply_currents`（`slow_field.py:254-278`）——把去极化部分除 `(1+α_A a)`、抑制项原样、
    减 `η_A a`；`a` 在 `apply_currents` 内部按位置采样。**不动 `kick_probe.py`。** blast radius 更小 → 首实现更稳。
- **数值红线（review #14）：** ODE 用 forward-Euler + 全 clamp（n/a）+ denominator/conductance cap（`1≤1+α_A a≤D_A,max`；
  `0≤g_A≤g_A,max`）；`dt/τ` 单位统一（引擎 `ms`）。
- **诊断记录（review #14，候选 go cell 必存）：** 每 cell 记 `D_G=1+α_G S_G`、`D_A=1+α_A a`（或 `g_A`）的 mean/p95/max；
  候选 go cell 另存 `I_rec_E`、`I_rec_E/D_G`、`I_dep/D_A`、`q_I I_I` —— 证 clean termination 是 `a` 改了 recurrent gain、
  **不是**数值压死。trace 默认只存 `⟨a⟩/p95(a)/max(a)/⟨a⟩_{top-k rate bins}`（照 M4-2 downsample），全空间图只对候选 go / 机制图 cell 存。

**runner（`run_m4_dynamic_qi.py`）：** M4-3A 扫 `(α_A × τ_n)`——**复用 `--p1-sweep` 的 Pool/COW/provenance/OOM-safe(`--p1-workers 5`)/
fail-loud 机器**，只换 cell 参数（`ee_std_u × ee_std_tau_ms` → `α_A × τ_n`）；`ee_std` off、`use_A` on、`g_K` off。
**每 seed 的 Arm0 必在同 run 内重跑（D10）**——不引用旧 archive（一点 seed routing / hook 改动都可能改 baseline）；核对：现有
`--p1-sweep` 的 `cells[0]` 本就是 per-seed Arm0（`ee_std_u=0`），`--T 40000 --seed <s>` 即可，无结构改。
**retrigger 扩早窗（D5）：** `run_cell_with_retrigger`（`sef_hfo_m4_termination.py:121-165`）现单 offset（且已是 late）；加一个 early
短 offset 的同-seed pass（引擎只有 `t_kick`/`t_kick2` 两个 kick 槽、无 `t_kick3` → early/late 用**两次同-seed pass-2**，各自过
pre-probe identity 断言）；`retrigger_verdict`（`:100-118`）把 `fail` 拆 `attenuated`(fizzle) vs `runaway`(尾高)。

**parity 红线：** `use_A=False`（或 `k_n=0`）默认逐字节等旧引擎；改 `kick_probe.py`（实现 A）则 re-bless `engine_versions.json`。

---

## 9. M4-3B / M4-3C（rev2：M4-3B smoke 提前并行；full 闭环仍 gated）

### 9.1 M4-3B smoke（**低成本、与 M4-3A discovery 并行**；review #11）

**不**等 M4-3A 完全 terminate 成功。只做最小冒烟，判 graph-kernel 有没有戏：
- `K_graph=F(W_EE)`（E→E 连接 coarse-grain 到网格、row-normalize、graph 扩散核）；`λ_K∈{0, 0.5, 1}`；**只改 `K_q^⋆`**；1–2 seed。
- 看模型自发/早期 event 的 `P_BB` 是否开始和自身 HFO 模板对齐。若 `K_graph` 完全无效 → 不在 M4-3A 终止机制上过度投入（§5）。

### 9.2 M4-3B full（gated；graph-kernel λ_K 扫 + 三负控 —— review #10）

- 扫 `λ_K∈{0,0.25,0.5,0.75,1}`，分别用于 `K_q^⋆` / `K_n^⋆`；对照 = 只改 `K_q^⋆` / 只改 `K_n^⋆` / 都改。
  预期 `K_q^⋆` graph 分量对**发作早期宽带对齐**最关键，`K_n^⋆` 影响**沿轴恢复**。
- **三负控（必做，否则"用 `W_EE` 造传播又用 `W_EE` 造 kernel 当然对齐"不可反驳）：**
  1. **degree-preserving shuffle：** 保每节点 degree/权重分布、打乱空间位置 → 对齐消失 = 不是 degree/smoothing。
  2. **rotated / mismatched kernel：** `K_graph` 空间旋转 or 换另一 seed/subject 的 graph → 对齐下降 = 需 matched 病理图。
  3. **readout-only（不反馈）：** `K_graph` 只作分析投影、不进慢变量动力学 → 若只读出不出 early 宽带对齐、而反馈能 = graph-coupled
     slow variables 是机制，不只是分析坐标。

### 9.3 M4-3C full（gated）

每长仿真自动检测间期 HFO-like 事件 → peak-timing 出模板 → 追 `P_BB(x,t)` 与模板相关 → 追 `(q_∥, a_∥)`（**模型输出后的坐标
变换，非输入 mask**）闭环轨迹 → 发作后验证回到可 replay 态。

---

## 10. 执行顺序（承重 —— review "先小后大，不直接开大 sweep"）

```
P0  离线 n/a trace 标定（不跑网络）           → 定 (τ_n,k_n,n_50,ρ_n,u_n0,a_max) + Δa_IED/R_A 离线值
P1  M4-3A discovery 小网格 (α_A×τ_n), T=15000  → 每 seed 含 Arm0；找 candidate terminate_clean
    ┊（并行）M4-3B smoke: λ_K∈{0,0.5,1}, K_q* only, 1–2 seed, 只读出  → graph-kernel 有没有戏
40s candidate/边界 cell acceptance, T=40000 + post-offset 10–20s 静默 + early/late retrigger + D11 ablation
────────────────────────────────────────────────────────────────────
（gated）M4-3B full λ_K 扫 + 三负控   →  M4-3C full 闭环
```

---

## 11. rev1 → rev2 承重改动一览（8 blocking + 强化）

| # | rev1 | rev2 | 来源 |
| --- | --- | --- | --- |
| 1 | §3.2 `a` 除整个 signed net current | **铁律：绝不除 signed net**；(A) 电导 leak-shunt 首选 / (B) 只除去极化驱动 | review #1（blocking） |
| 2 | `n` 方程无 baseline | **baseline-centered `[u_n-u_{n,0}]_+`** + n/a clamp + denom cap | review #2（blocking） |
| 3 | "`a` 被 IED 冲" 纯文字 | **`Δa_IED`/`R_A`/`⟨a⟩<a_block` 硬数值门**（§6.1） | review #3（blocking） |
| 4 | retrigger 沿用 M4-2 单判据 | **go = terminate_clean + early attenuated + late reignite_bounded**（postictal 不应期） | review #4（blocking） |
| 5 | `T=15000` 作终判 | **discovery 15s / acceptance 40s + post-offset 静默** | review #5（blocking） |
| 6 | seed 1/3/4 不分层 | **primary = Arm0 bounded-persist seed（计算门）；seed 4 stress**（对齐 M4-2 §0/§2.3） | review #6（blocking） |
| 7 | `g_K` 未言明 | **核对：`g_K` 在 M4 op-point 已 hard-off** → ArmB 坍缩、无需 g_K arm（简化 review） | review #7 + 代码核对 |
| 8 | no-go → `D_EE` | **no-go 只加强怀疑；先过 M4-3B smoke；对齐 M4-2 的 gK-first 分叉** | review #9（blocking） |
| — | §8 "两遍 retrigger 零改" | 据实：`classify_termination` 零改，**retrigger harness 扩 early 窗 + verdict 拆 attenuated/runaway** | 代码核对 |
| — | (A) "复用 membrane_step" | 据实：shunt 形在 `slow=None` 支路，M4 支路要**接线 + re-bless**（非零改） | 代码核对 |
| + | `σ_n` 未定 | **`σ_n=σ_q=1.5` 默认（宽），非窄 `σ_K`；次级扫 {0.5,1,1.5}×σ_q** | review #8 |
| + | 无机制 ablation | **D11 shunt-only vs subtractive-only vs hybrid**（照 M4-1） | review #12 |
| + | 无 P0 | **P0 离线 `n/a` trace 标定先行** | review #13 |
| + | M4-3B 全 gated | **M4-3B smoke 与 M4-3A 并行**；full + **三负控** | review #10/#11 |

---

## 12. 本地锚 & 文献 & Framing 锁

**本地锚（含 rev2 核对的确切行号）：** M4-2 archive + spec（§7.2 no-go；§0/§2.3 seed 分层；§5 双分叉 gK/D_EE）；M4-1 pass-1
archive；M3B 谱线（轴=非正规瞬态，R²≈0.87，`src/topic4_m3b_spectral_phase.py`）；引擎 `slow_field.py`（`q_I` 步进
`:285-311`、apply `:254-278`、`sigma_q=1.5`/`sigma_K=0.5` `:56/:66`、trace `:310`）、`kick_probe.py`（M4 内联膜更新 `:316-317`、
现成电导 shunt `:86-88`、`dump_ee_std_trace` hook `:238-251`）；runner `run_m4_dynamic_qi.py`（`--p1-sweep` cell `:289-355`、
M4 op-point `use_gK=False` `:184`、per-seed Arm0=`cells[0]`）；判读器 `src/sef_hfo_m4_termination.py`（`classify_termination`
`:48-97`、`retrigger_verdict` `:100-118`、`run_cell_with_retrigger` `:121-165`）。
Topic5 数据锚：V2 phase1（间期↔早期宽带对齐，承重）、V2 phase2 / V3p / V3a（发作前 state / 逐渐爬升 / 模态转移，
weak/negative）——写 M4-3B/C 前**回 topic5 archive 复核这三个的确切口径**。

**文献（连续活动驱动慢变量，非 sensor）：** Chizhov/Chizhov-Zefirov Epileptor-2（PLoS CB 2018，连续 K/Na/pump，IED 聚集成
ictal、Na/K-pump 终止，含 STD 变量）；Krishnan/Bazhenov（离子动态介导自发终止 + 爆后 K⁺ 低于 baseline = postictal
depression）；TRESK（去极化诱导 shunting inhibition 参与 seizure termination + 延长 postictal）；adenosine（发作中升高，参与
arrest + 不应期）；shunting/normalization：Chance-Abbott-Reyes（背景突触输入 divisive gain modulation）、Carandini-Heeger
（normalization）；非正规瞬态放大：Hennequin et al. / Murphy-Miller（balanced network non-normal amplification）。

**Framing 锁：** 措辞用 "actual M4-3 **SIMULATION**"，绝不 "real data"。**别现在改论文摘要**（§7 那段升级版是 M4-3B primary
复现后的 paper claim；§6.2 sensitivity-gate：没跑不写）。任一结局（M4-3A go / clean no-go）都如实报；no-go **加强怀疑**（不是
已证）指回衬底，且先过 M4-3B smoke。
