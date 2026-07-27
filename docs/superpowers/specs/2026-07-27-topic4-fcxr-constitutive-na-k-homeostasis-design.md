# FCXR-ION：在 E1146 各向异性 E/I sheet 上引入构成性 Na/K 离子稳态

日期：2026-07-27（rev2，按 2026-07-27 审阅闭合 5 项 P0 + 4 项 P1）

状态：**DESIGN LOCK CANDIDATE — 方程与出处已闭合，无未定方程、无未定出处；剩 2 项 B0 待验证（引擎电压单位对应、方向读出的功率前置），见 §16 末尾。**

上一代际终局：`docs/archive/topic4/sef_hfo/mz_fcxr_pump_lifecycle_gate_Ia_2026-07-27.md`

实施计划（B0–B2）：`docs/superpowers/plans/2026-07-27-topic4-fcxr-constitutive-na-k-homeostasis-B0-B2.md`

rev2 相对 rev1 的实质改动都记在 §16，其中三项是**读源文献后改设计**，不是措辞调整。

---

## 0. 为什么换代际

上一版把泵当作**应当在间期隐形的附加终止器**并要求泵开 ≈ 泵关。它没通过：即使强度压到平均只扰动
阈值 2%，间期事件仍变长约 40%、源区放电份额降低约 11.5%。

上一版归档提的"加半激活位置"**已作废**：稳态质量平衡给 `E[phi_K(u)] ≈ a_load·tau_N·r`，
在跳变、清除时间与放电率固定时改半激活位置**不改变平均激活度**，且三者互相补偿会引入不可辨识。

本代际改的是前提：Na/K 泵本来就持续工作、参与维持静息电位与间期兴奋性。因此不再要求它对基线隐形，
而写成**构成性**离子稳态并**重新标定**间期工作点。要重新证明的不是"泵开=泵关"，而是：

> **离子化之后的网络，是否仍产生稀疏、不规则、且在两个注册核上都会起始的间期活动？**
>
> 注意措辞：**不是**「双向轴向传播」——该性质在 E1146 上未被确立，见 §9 B-real。

---

## 1. 科学假设与命名合同

| 慢坐标 | 职责 |
|---|---|
| `Z_i` | 局部抑制可靠性与 onset permissivity（保留，参数不动） |
| `[K]_o(x,t)` | 活动依赖的**局部正反馈**：招募与传播易感性 |
| `[Na]_i → J_pump` | 累积负荷 → **负反馈**：终止与爆后记忆 |

**onset 归因不得预设**：可能主要由抑制可靠性下降驱动、主要由钾积累驱动、或二者协同，必须由 B4 消融决定
（本 spec 不授权 B4）。

**命名合同**：本模型只称 **reduced ion-homeostatic SNN**。不得称 ion-conserving neuron model；
`Na_i`/`K_o` 的 mM 量纲是机制先验的量纲，**不是** E1146 的实测浓度估计。每个模型神经元代表一群未解析的
真实神经元，因此 `q_ion` 是"每个**模型** spike"的有效增量，不是真实单细胞的 Na 内流。

---

## 2. 明确不做什么

第一代际**不加入**：动态 `Cl_i`/KCC2、`Ca_i`、胞体—树突双室、体积变化、去极化阻滞、完整 HH 电流。
只加入六件事：per-cell `Na_i`、二维 `K_o(x,t)`、构成性泵、`K_o` 对膜的 Nernst 型调节、
胶质 + 储库清除、局部 `K_o` 扩散。

---

## 3. B0 参数与出处表（P0-1 闭合）

**主文献**：Ullah et al. 2009 network model（PMC2951284）
**配套单细胞约束**：Cressman et al. 2009（PMC2704057）
两者是同一套；**不再使用模糊的 "family" 表述，也不跨文献拼半激活常数。**

分类：`inherited` = 逐字取自上述两篇；`derived` = 由 inherited 项 + 本模型已测量的工作点解析导出，
**无自由度**；`effective` = 本约化模型自己的闭合假设，**不是**原 HH 网络的原方程，必须如此标注。

### 3.1 泵与离子参考量（全部 inherited）

| 参数 | 值 | 单位 | 出现的方程 | 出处 | 类型 |
|---|---:|---|---|---|---|
| `rho`（泵最大通量） | 1.25 | mM/s | `I_pump` | Cressman 2009，正常态 | inherited |
| `Na_half` | 25.0 | mM | `I_pump` | 同上 | inherited |
| `s_Na` | 3.0 | mM | `I_pump` | 同上 | inherited |
| `K_half` | 5.5 | mM | `I_pump` | 同上 | inherited |
| `s_K` | 1.0 | mM | `I_pump` | 同上（隐式斜率） | inherited |
| `Na_i0` | 18.0 | mM | 静息 | 同上 | inherited |
| `K_i0` | 140.0 | mM | 静息 | 同上 | inherited |
| `K_o0` | 4.0 | mM | 静息 = `k_o_inf` | 同上 | inherited |
| `beta`（胞内/胞外体积比） | 7.0 | — | `K_o` 方程 | 两篇一致 | inherited |
| `eps`（储库/浴清除） | 1.2 | 1/s | `I_diff = eps([K]_o − k_o_inf)` | Cressman 2009 | inherited |
| `k_o_inf` | 4.0 | mM | 同上 | 同上 | inherited |
| `G_glia` | 66.0 | mM/s | `I_glia` | Cressman 2009 | inherited |
| 胶质半激活 / 斜率 | 18.0 / 2.5 | mM | `I_glia` | 同上 | inherited |
| `D_K` | 2.5e-6 | cm²/s = 2.5e-4 mm²/s | Ullah 2009 Eq.(5) 离散拉普拉斯 | Ullah 2009 | inherited |
| Nernst 因子 `RT/F` | 26.64 | mV | `E_K` | 标准值（310 K） | inherited |

方程（逐字形式）：

```
I_pump = rho / (1 + exp((Na_half - [Na]_i)/s_Na)) / (1 + exp((K_half - [K]_o)/s_K))
I_glia = G_glia / (1 + exp((18.0 - [K]_o)/2.5))
I_diff = eps * ([K]_o - k_o_inf)
[K]_i  = K_i0 + (Na_i0 - [Na]_i)                      # Cressman 代数闭合
E_K    = (RT/F) * ln([K]_o / [K]_i)
```

由此**解析**得到静息量（无自由度）：

```
I_pump_0 = 0.02016 mM/s          E_K_0 = -94.71 mV
```

### 3.2 spike→ion 约化（本模型自己的 effective 闭合，含唯一一个 dial）

原参考是 HH 网络，`I_K`/`I_Na` 是真实电流，用 `0.33 mM·cm²/µcoul` 把电流密度换成浓度变化率。
本模型是 LIF，**没有**显式 Na/K 电流，因此必须自己定义每个模型 spike 的有效离子增量。
**这一段是 `effective`，不是继承。**

唯一的 dial 是无量纲的 **`f ∈ (0,1]`**：静息期泵负荷中由 spike 驱动的比例。

| 量 | 表达式 | 类型 |
|---|---|---|
| `q_ion`（每模型 spike 的胞内 Na 增量） | `3 * I_pump_0 * f / r0` | derived（给定 f） |
| `J_Na_rest`（静息 Na leak） | `3 * I_pump_0 * (1 - f)` | derived，**恒 ≥ 0** |
| `q_K`（每模型 spike 的胞外 K 增量） | `beta * q_ion` | effective：取每 spike 的 K 外流与 Na 内流**摩尔数相等**（最小电荷平衡闭合），再由 `beta` 换算到胞外体积 |

把 `q_ion` 写成 `f` 的函数的好处：`J_Na_rest ≥ 0` **由构造保证**（审阅要求的解析可行性门之一自动满足），
且 `f` 有明确物理含义与硬边界，不像原始的自由 `q_ion` 那样量纲含糊。

**B0 已完成的解析可行性核算**（表中用 `r0 = 3.838 Hz`；§7.1 把 primary 锁为同一条泵关轨迹实测的
`4.158 Hz`，两者差 8%，对下面的量级结论无影响）：

| `f` | `q_ion` (mM/spike) | 一次普通事件的 `ΔK_o` | 对应 `ΔE_K` | 占 `V_th=18` | 持续 50 Hz 的稳态 `K_o` | 对应 `ΔE_K` |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.01576 | 0.0215 mM | 0.143 mV | 0.8% | ~7.2 mM | +15.7 mV |
| 0.50 | 0.00788 | 0.0108 mM | 0.072 mV | 0.4% | ~5.7 mM | +9.2 mV |
| 0.25 | 0.00394 | 0.0054 mM | 0.036 mV | 0.2% | ~4.9 mM | +5.2 mV |

**这是本 spec 最重要的 B0 结论**：在**零自由参数**（`f=1` 即"静息泵负荷全部由 spike 驱动"）下，
一次普通间期事件只把钾抬 0.02 mM、对膜的影响不到阈值的 1%，而持续 50 Hz 会把钾推到约 7.2 mM、
钾反转电位移动 +15.7 mV（阈值的 87%）——约 **110 倍动态范围**，且落在这类模型报告的发作期钾区间内。
即"间期安静、高态强正反馈"这个必要条件在动笔仿真之前就已经解析成立。

`f` 的 B1 任务因此不是"找一个能出发作的值"，而是"找一个让单次事件的钾瞬态**可测但会恢复**、
重复事件簇能时间积分、且普通事件不产生持续积累的值"。primary 起点 `f = 1.0`，B1 只在 `{1.0, 0.5, 0.25}` 三点里选。

### 3.3 `rho` 的阶段权限（P0-2 闭合）

- **B0 固定 reference `rho` = 1.25 mM/s**，用于构成性质量平衡（`J_Na_rest`、`K_res`、Gate H 全都依赖它）；
- **B0–B2 一律不动 `rho`**；
- B4（未授权）最多标定 `eta_pump`，或一个**预锁范围内**的无量纲泵倍率；
- **若 B4 改动 `rho`，必须重新走 Gate H 与 Gate B，不得沿用旧 baseline。**

rev1 里"第一阶段标定 rho"与"B4 才确定 rho"的自相矛盾就此消除。

---

## 4. 方程（P0-3 闭合：显式有限体积形式）

### 4.1 每个细胞的胞内 Na

```
d[Na]_i/dt = J_Na_rest + q_ion * S_i(t) - 3 * I_pump_i
I_pump_i   = rho / (1+exp((Na_half-[Na]_i)/s_Na)) / (1+exp((K_half-[K]_o,g(i))/s_K))
```

系数 3 来自泵每循环外排 3 个 Na（与 Cressman 的 `-3 I_pump` 一致）。

### 4.2 胞外 K：显式有限体积

**归一化合同（rev1 的主要漏洞）**。设网格格 `g` 的面积 `A_g = Δx²`，等效层厚 `h`，
胞外体积 `V_o,g = alpha_o * A_g * h`；格内模型神经元数 `n_g`，全局均值 `n̄ = N / N_grid`。

**`beta` 是组织属性，恒为 7.0，不随 `n_g` 变化。** 模型神经元是组织的**子采样**
（sheet 密度 100 neurons/mm² 远低于真实密度），因此局部 `n_g` 的涨落是**采样噪声**，不代表真实的
局部体积比变化；把 `beta` 按 `n_g` 缩放会把采样噪声当成真实的钾涨落。

由此，格 `g` 的 K 源用**格内 per-cell 平均**，不用总和：

```
r̄_g(t) = (1/n_g) * sum_{i in g} S_i(t)          # 每细胞平均 spike 率；n_g = 0 -> 源项 = 0
Ī_pump,g(t) = (1/n_g) * sum_{i in g} I_pump_i    # 同上

d[K]_o,g/dt =  beta * q_ion * r̄_g(t)             # spike 驱动的 K 外流（= q_K * r̄_g）
             - 2 * beta * Ī_pump,g(t)            # 泵每循环回收 2 个 K
             - eps * ([K]_o,g - k_o_inf)         # 储库/血管清除（线性，inherited）
             - I_glia([K]_o,g) + I_glia(K_o0)    # 胶质缓冲，围绕静息中心化（inherited 形状）
             + (D_K / Δx²) * (sum_{nb} [K]_o,nb - n_nb * [K]_o,g)   # 局部扩散
```

**为什么保留胶质项而不是并进一条线性储库**（rev1 曾把两者并成一个 `(K_res - K_o)/tau_K`）：
胶质摄取是**饱和**的（`I_glia` 在高钾处趋于 `G_glia`），这正是钾积累失控的已知机制之一。
把它并进线性项会**删掉**路线 B 依赖的一个正反馈成分。因此按参考分开写。

`+ I_glia(K_o0)` 与 `- eps(K_o - k_o_inf)` 的组合使 `K_o = K_o0 = k_o_inf` 在**无 spike、泵处于静息**时
恰为不动点，无需再引入自由的 `K_res`（rev1 的 `K_res` 反推式随之删除）。

边界条件：**零通量（反射）**，`n_nb` 为该格实际存在的邻居数（角/边格分别为 2/3）。空格
（`n_g = 0`）源项与泵项为 0，但清除与扩散照常作用。

**网格不变性的正确表述**：由于 K 源用的是 per-cell 平均且 `beta` 是组织常数，细化网格**不改变**
总钾预算，也不改变粗粒化后的场；但**逐格场本身不期望逐点相同**（扩散离散化 + 采样噪声）。
Gate H 的检验因此是"总预算闭合 + 粗粒化场一致"，不是"逐格数值相同"。

### 4.3 与膜方程的耦合（受 §5 引擎钩子硬约束）

```
tau_m_a * dV_i/dt = F_FCXR_i
                    + g_K_ion * (E_K_i - E_K_0)                  # 钾反转移动 -> 兴奋性变化
                    - eta_pump * (I_pump_i - I_pump_0)           # 高于静息的电生性外向电流
```

两项都是**电流**量纲，**对 E 和 I 一视同仁**。减去 `I_pump_0` 只是把静息泵电流吸收进重新标定的
bias 平衡；**离子质量方程（§4.1/§4.2）始终使用完整的、非零的 `I_pump_i`**。

`g_K_ion`：**effective**。这是围绕静息电位的**线性化** —— 它抓住"钾反转移动导致静息处的钾电流改变"，
但**丢掉**了电导本身的分流效应（真正的 `g_K(E_K − V)` 会同时改变有效时间常数）。这一线性化对 I 细胞
是引擎强制的（§5），为一致性对 E 细胞也采用。B0–B2 锁 reference `g_K_ion = 1`（含义：静息钾电导
与漏电导同量级，1 mV 的钾反转移动产生 1 个引擎单位的驱动改变）；B3 才细化，**任何改动都要重跑 Gate B**。

⚠️ **B0 待验证项之一**：上述 `g_K_ion = 1` 依赖"1 个引擎电压单位 = 1 mV"。证据支持
（`V_L = 0` 坐标、`V_th = 18` 约等于生理阈上量、`E_E = 58` 对应静息 −58 mV 时的 0 mV AMPA 反转），
但**必须在 B0 用一次显式的单位核算确认**，因为它直接决定 `g_K_ion` 的量级。未确认前不得进入 B1。

---

## 5. 引擎钩子审计（2026-07-27 在代码中逐条核实，是硬约束）

| 需要的钩子 | 现状 | 证据 |
|---|---|---|
| 每步拿到 **E+I 全体** spike mask | **已有** | `slow.step(spk, labels, dt)` 收全长 `N` 掩码 |
| 慢对象自行决定更新频率 | **已有** | 慢对象自持步计数 |
| I 细胞的**附加电流**耦合 | **已有** | `cond_drive[NE:]` 生效 |
| I 细胞的**电导**耦合 | **不存在** | 电导分支执行 `Vtmp[~is_E] = cond_drive[~is_E] + …`，`g_rel[NE:]`/`g_rev[NE:]` 被**静默丢弃** |
| 虚拟电极读出范围 | **E-only** | 记录器只对 E 加权（`posE = pos[:NE]`） |

**设计锁定**：两个离子膜项对 E/I 都写成**附加电流**。**不得**把钾项写成 I 细胞的电导——那会被静默丢弃，
造出一个只作用于 E 的机制却报告成 E/I 都有。若 B3 之后证明 I 细胞必须走电导路径，
**另开受审阅的 guarded-engine-change spec**，不在科学 sprint 里改 blessed 引擎。

---

## 6. 与既有慢变量的组合协议（P0-4 闭合）

引擎只接受**一个** `slow=` 对象，其协议表面（已逐条枚举）为：

```
capability : uses_conductance_membrane()  uses_split_excitation()  uses_ee_relay()  uses_shunt()
attribute  : cfg.use_SG      ee_relay_send[...]      nE        q_I[:]
call       : membrane_terms(I_E, I_I, labels[, I_E_rec=...])
             apply_currents(I_E, I_I, labels[, I_E_rec])
             threshold(base_vth)
             step(spk, labels, dt)
```

因此**不新造一个平行的 slow 对象**，而是包装：

```python
class IonHomeostaticMZAdapter:
    """包装既有 MZSlowVars，保留 Z 与 FCXR 电导路径不变，只在其结果上叠加离子电流。"""
    def __init__(self, mz, ions): self.mz, self.ions = mz, ions
    # 四个 capability 谓词 + cfg / ee_relay_send / nE / q_I 一律直通 self.mz
```

**委托顺序（合同）**：

1. `membrane_terms(...)` → 先**原样调用** `self.mz.membrane_terms(...)` 拿到 `(drive, g_rel, g_rev)`；
   再对 **E 和 I 全体** 执行 `drive += g_K_ion*(E_K − E_K_0) − eta_pump*(I_pump − I_pump_0)`，
   用的是**上一个离子块**的 `E_K` / `I_pump`（因果：离子状态从下一块才生效）；
   `g_rel` / `g_rev` **一个字节都不动**。
2. `threshold(...)` → 直通。
3. `step(spk, labels, dt)` → **先**原样调用 `self.mz.step(...)`（既有 Z/M/X 顺序与逐步值不变），
   **再**累积本步的 E/I 网格 spike；每 `dt_ion / dt` 步更新一次离子状态。
4. `apply_currents(...)` → 直通（本代际只走电导膜路径，此路径不应被触发；若被触发则 `raise`，
   遵循"stub 必须响亮失败"的项目约定）。

**必需回归测试**：

- `adapter-off byte-parity`：`ions` 关闭时，整条 `simulate_kick` 与**裸** `MZSlowVars` 逐位相同；
- `existing Z/M/X update order unchanged`：复用本 sprint 已有的逐步值级别测试；
- `I-cell coupling is a CURRENT`：显式验证离子项在 **I 细胞**上确实生效（防止只作用于 E 却报告成 E/I 都有）；
- 六个 blessed 引擎文件 sha256 不变。

---

## 7. 工作点闭合与参数辨识顺序（P1 循环定义闭合）

### 7.1 `r0` 的出处与"一次闭合迭代"规则

`r0_E` / `r0_I` **锁定来源**：本 sprint 已落盘的 arm-C 泵关轨迹的后燃烧期块指标
`results/topic4_sef_hfo/mz_full_conductance_spatial_relay/pump_lifecycle/pump_baseline_equivalence.json`
→ `per_arm.pump_off.pooled.mean_rate_hz = 4.158 Hz`（E 群体平均）。
B0 的解析核算用的 `r0 = 3.838 Hz` 是 HEO1 的 slow-off 参考值；**B0 必须二选一并写死**，
primary 取 **4.158 Hz**（与将要对照的泵关臂同一条轨迹、同一段窗口），HEO1 值只作为一致性旁证
（两者差 8%，对 §3.2 的结论无影响）。

**循环定义的解法 —— 恰好一次闭合迭代，之后冻结**：

1. 用锁定的 `r0` 计算 `q_ion` / `J_Na_rest`；
2. B2 只调 `I_bias_E` / `I_bias_I` 重建间期；
3. 测得新的实际率 `r0'`；
4. **重算一次** `q_ion` / `J_Na_rest`，重跑一次；
5. **此后冻结，不再迭代。** 残差硬门：`|r0' − r0| / r0 ≤ 10%`，且离子变量在长 burn-in 后
   `|d⟨Na_i⟩/dt|` 与 `|d⟨K_o⟩/dt|` 的块间趋势不显著（用本 sprint 已有的块间容差机制判定）。
   超出即 NO-GO，不得靠继续迭代收敛。

**解析可行性门（进入任何仿真前必须成立）**：

```
J_Na_rest >= 0                     # 由 f in (0,1] 构造保证
K_o = k_o_inf 是无 spike 时的不动点   # 由 §4.2 的中心化形式构造保证
[Na]_i, [K]_i, [K]_o 全程 > 0
```

### 7.2 nuisance 参数（P1 闭合）

**只允许 `I_bias_E` 与 `I_bias_I` 两个。** rev1 里"必要时再加一个全局背景驱动尺度"的第三个 fallback
**已删除**：两个 bias 无法恢复 baseline 即判 **NO-GO**。

**禁止修改**：椭圆连接梯度、core 位置、合作增益、连接权重、`Z` 参数、源/汇标签、泵以外的终止机制。

### 7.3 五步顺序（B0–B2 授权，B3–B4 不授权）

- **B0（授权）**：锁参考家族与出处表（§3）、单位与空间闭合（§8）、`rho` 权限（§3.3）、
  电压单位核算（§4.3 ⚠️）、真实 E1146 目标产物（§9）。**不跑 40k。**
- **B1（授权）**：小网络定 `f`（三点里选）；`tau_K` 无需拟合（`eps` 是 inherited）。
- **B2（授权）**：40k，只调两个 bias，重建间期并验收两核起始位点（§9 B-real）。
- **B3（不授权）**：由冻结态局部钾微扰定 `g_K_ion`。
- **B4（不授权）**：由冻结高分支 exit 定 `eta_pump`（及预锁范围内的泵倍率）。

---

## 8. 空间、边界与积分（P1 闭合）

| 量 | 锁定值 | 依据 |
|---|---|---|
| sheet 物理尺寸 | **20 mm × 20 mm** | 引擎 `L = 20`，`params.py` 中 `rx`/`Rr`/`grid_spacing` 均以 mm 标注 |
| 神经元面密度 | 100 /mm²（**约化**密度，非真实计数） | 引擎 `density = 100`，`N = 40000` |
| **primary 网格** | **32 × 32，Δx = 0.625 mm** | 见下方论证 |
| sensitivity 网格 | 16×16、64×64 | — |
| 边界条件 | 零通量（反射） | — |
| `D_K` | 2.5e-4 mm²/s（= Ullah 的 2.5e-6 cm²/s），**物理值，不改名** | §3.1 |
| **primary `dt_ion`** | **0.5 ms** | 最快线性离子速率是 `eps = 1.2 /s`，0.5 ms 比它小三个数量级 |
| sensitivity `dt_ion` | 0.25、1、2 ms | — |
| 膜/突触 `dt` | 0.05 ms（不变） | — |

**primary 网格从 rev1 的 64×64 改为 32×32 —— 有量化理由，明确标注为偏离原始 brief**：

| 网格 | Δx (mm) | 每格神经元数 | 空格概率 | `D_K/Δx²` (1/s) | 占 `eps=1.2` 的比例 |
|---|---:|---:|---:|---:|---:|
| 16×16 | 1.250 | 156 | ~0 | 1.6e-4 | 0.01% |
| **32×32** | **0.625** | **39** | ~0 | 6.4e-4 | 0.05% |
| 64×64 | 0.3125 | 9.8 | 5.7e-5 | 2.6e-3 | 0.21% |
| 128×128 | 0.156 | 2.4 | **8.7e-2**（约 368 个空格） | 1.0e-2 | 0.85% |

K 源是**格内 per-cell 平均**，64×64 只有约 10 个细胞/格、128×128 只有 2.4 个且约 9% 的格全空，
采样噪声会直接污染钾场。32×32 给 39 个细胞/格，且 Δx = 0.625 mm 能用约 2.4 格分辨 1.5 mm 的核区。

**同一张表带出一个改设计的事实（读源文献后才知道）**：Ullah 自己的网格是**每个细胞一格**
（Δx = 10 µm），`D_K/Δx² = 2.5 /s`，与浴清除 `eps = 1.3 /s` **同量级**，所以在他们的模型里扩散是重要的。
在我们的 sheet 尺度上，`D_K/Δx²` 只有 `eps` 的 **0.01%–0.85%**，**扩散项实际上是惰性的**。三个后果：

1. `D_K` 保留物理值，**不需要** `D_K_eff` 这个名字（除非将来有人要人为放大它，那时必须改名）；
2. 停机条件"钾只能靠人为增大的扩散沿轴传播"变成**可解析预先排除**的，不需要靠仿真发现；
3. 各格近似独立，因此**网格分辨率就是钾反馈的空间局部性**——这是一个建模选择，不是数值旋钮，
   必须按上表的理由明示，不得事后调整。

**各向同性**：primary `D_x = D_y`。不得让钾沿行为学轴扩散更快——那等于把答案写进离子机制。
行为学轴只来自既有的椭圆连接、core 几何与事件起始位置。

**多时间尺度块内顺序**：用当前 `Na_i`/`K_o` 算 `E_K` 与泵电流 → 跑若干膜步 → 累积 E/I spike →
块末更新 `Na_i` → 更新 `K_o`（源、泵、清除、胶质）→ 半隐式或稳定显式更新扩散 →
**新离子状态从下一块才生效**。

---

## 9. 本代际授权的两道门

### Gate H：稳态与数值合同（`N≈1000`，再到 `N≈4000`）

**措辞已按审阅订正**：本模型**没有** `Na_o` 动态、使用**开放**储库、`K_i` 是代数近似，
因此**不得**暗示整个系统 ion-conserving。三项检验分别是：

1. **ODE balance residual**：无 spike 时 `|d[Na]_i/dt|`、`|d[K]_o/dt|` → 0，且静息不动点正确；
2. **finite-volume K budget closure**：一段窗口内 `Σ(源) − Σ(泵回收 + 清除 + 胶质) − Σ(扩散净通量)`
   与 `Δ(总胞外钾)` 的相对误差 < 1e-10；扩散在零通量边界下**净通量为零**；
3. **pump 3:2 flux identity**：同一 `I_pump_i` 在 Na 方程里出现系数 3、在 K 方程里出现系数 2β。

另需：**baseline 泵通量非零**（构成性，不是静默插件）；局部扰动能恢复；网格/`dt_ion`/checkpoint-restart
一致；**离子插件关闭时旧引擎 byte-parity**；不出现负浓度或触及 safety bound（safety bound 只作
fail-fast，不得充当动力学饱和器）。

**Gate H 失败不进入 40k。**

### Gate B：40k 新间期 substrate（预注册，P1 闭合）

**development 用 connectivity seed 1 + 一个显式 development noise seed；两个 bias 只在 development seed 上调。
confirmatory：connectivity {1, 3} × 3 个未见 noise seed = 6 条轨迹，不得用于调参。**

指标、事件与窗口定义**全部复用本 sprint 已实现并测试过的机制**（不重造）：事件判据用完整轨迹上的
canonical bar；块指标、块间容差（`k = 2`）、以及"容差宽于均值一半即标 UNDERPOWERED"的规则同上一 sprint。

#### B-real（优先级 1，binding）—— 面向真实 E1146

**先纠正 rev1 的一处事实错误（2026-07-27 查产物后订正）。** rev1 把"双向轴向传播"写成 E1146 的
真实目标并引了通道层产物。查证结果：

| 事实 | 证据 |
|---|---|
| E1146 **有**两个稳定的间期传播模板 | `interictal_propagation_masked/per_subject/epilepsiae_1146.json` → `adaptive_cluster.stable_k = 2` |
| 两个模板整体秩相关 = **−0.464** | 同上 `inter_cluster_corr_matrix` |
| E1146 **没有**已确立的正/反向模板对 | 同上 `adaptive_cluster.candidate_forward_reverse_pairs = **null**` |
| 通道层互换分数**未populated** | `rank_displacement/per_subject/epilepsiae_1146.json` → `pr6_swap_score = null`、`pr6_swap_null_p = null`、`fwd_rev_reproduced = null`、`fwd_rev_source = "unknown"`；该文件**没有** `primary_pair` 字段 |

因此 **Gate B 不得把"双向/正反向传播"作为面向真实数据的判据** —— 这个被试上它没有被确立。
两个模板整体秩相关只有 −0.464，远不是一对反向。

**改为可支撑的判据**：模型必须保留**锚定在两个注册核上的两个可区分的传播起始位点**
（即事件不塌缩到单一核），这正是 substrate 编码的内容（两个低阈值核放在两个模板的 source foci 上）。

| 判据 | 真实目标产物 | 层级（§6.2 纪律） |
|---|---|---|
| 两个稳定传播模板存在，模型两核对应其 source foci | `interictal_propagation_masked/per_subject/epilepsiae_1146.json` → `adaptive_cluster.stable_k`、`clusters[k]` | **模板层**（不是通道层） |
| 间期事件的稀疏性与不规则性 | 同上 `propagation_stereotypy` / `temporal_dynamics` | 模板层 |

> 层级纪律（防 §6.3 代词坍缩）：真实产物只支持"**存在两个稳定模板**"；模型侧读出的是
> "**事件在两个注册核上都会起始**"。二者不是同一层，**不得**写成"模型复现了通道层的角色互换"，
> 也**不得**写成"模型复现了双向传播"。

**方向读出必须换（rev1 的选择在已接受底座上不可执行）**。rev1 用本 sprint 的 `forward_event_fraction`
（源核平均首发是否早于汇核），它要求**两核同时有参与**。在已接受的 arm-C 泵关轨迹上实测：
22 个自终止事件里只有 **2 个**满足（逐块 `n_direction_events = 1,0,0,0,1`），其余块为 `NaN`，
且该指标已被标为检验力不足。**用它做 Gate B 会让已接受的底座自己不通过**，判据无效。

**替换为 initiation-site 读出**（对每个事件取最早发放的那一小部分参与细胞的质心，按到两核的距离归属），
它对**每个**事件都可评分，不要求两核同时参与，功率高一个量级。

**B0 硬前置**：该读出必须先在**已接受的 arm-C 泵关轨迹**上验证功率 —— 在 B2 计划的窗长内
可评分事件数 **≥ 20**，且两核归属都非零。**达不到就必须先加长窗口或换读出，不得带着一个
在自己底座上都测不出来的判据进入 B2。** 阈值只在这一前置通过之后、在 development seed 上预锁。

#### B-model（优先级 2，工程参考）—— 对照 arm-C 泵关轨迹

- 长 burn-in 后 `Na_i` / `K_o` 块间平稳，**无缓慢倒计时**；
- 稀疏、不规则的间期事件保留（事件率、间隔中位数、间隔变异系数落在泵关臂的块间容差内；
  UNDERPOWERED 的指标不作为等价证据）；
- 源 / 汇 / 轴外放电份额落在泵关臂容差内；
- **普通间期事件不产生全 sheet 钾波**：事件后 `K_o` 的空间标准差不得超过其均值抬升的预锁倍数，
  且远离事件的格 `ΔK_o` 必须 < 事件格的 10%；
- baseline 泵不饱和（`I_pump` 远离 `rho`）；
- 普通事件后 `Na_i` / `K_o` 可恢复。

---

## 10. 三套虚拟电极读出

```
V_synaptic       只含突触分量
V_synaptic_K     + 钾相关膜项
V_all            + 泵电生电流
```

主要数据一致性判据用 `V_synaptic` 或 `V_synaptic_K`，以证明宽带变化来自网络活动重构而非慢电流直接
制造低频功率；`V_all` 只作机制敏感性。读出空间范围仍是 **E-only**（引擎记录器限制，§5），
必须写进产物元数据。

---

## 11. 停机条件（任一触发即停并归档 bounded negative）

1. 无法建立稳定的**非零**泵静息平衡；
2. 离子变量在普通间期持续漂移（或 §7.1 的一次闭合迭代后残差超门）；
3. 普通间期事件产生全局钾波；
4. 恢复间期需要修改连接或合作增益（或需要第三个 nuisance 参数）；
5. 必须加入 M / X / H 才能终止；
6. 泵关与完整泵的 offset 无差异；
7. 终止主要来自 LIF 的不应期上限；
8. 钾只能靠人为增大的扩散沿轴传播（**§8 已解析预先排除，若出现说明实现有 bug**）；
9. 系统只能产生规则周期性的发作振荡器；
10. 高态进入 LIF 无法表示的去极化阻滞区；
11. 生命周期只在单一 connectivity/noise seed 中存在；
12. 需要动态 `Cl` / `Ca` 才能保住最基本的 baseline。

---

## 12. 执行边界

**授权**：B0（设计/单位/出处闭合）、B1（小网络定 `f`）、B2（40k 新间期 substrate）+ Gate H + Gate B。

**不授权**：B3、B4、三维 `(Z̄, K̄_o, N̄a_i)` 相图、动态生命周期、因果分解（含四臂泵分解、钾钳制、
Na 钳制、Z 消融、爆后 reset）、空间响应模态、数据一致性判据、以及任何 `Cl`/`Ca`/双室扩展。

只有新的构成性离子 substrate 在**真实间期目标**与**两个注册核的起始位点**上重新验收，才允许进入
三维相图与生命周期实验，届时另写 spec 与 plan。

**本 spec 完成后的下一步是写 implementation plan，不是启动 B1。**

---

## 13. 工程架构

新建 `src/snn_engine/ion_homeostasis.py`（**不**塞进 `mz_slow_vars.py`），
加上 §6 的 `IonHomeostaticMZAdapter`。状态：

```
Na_i_all        (N,)          E 与 I 都有
K_o_grid        (32, 32)
pump_flux_all   (N,)
E_K_all         (N,)
grid_spikes_E / grid_spikes_I
```

必需工程测试：

```
plugin off byte-parity            adapter-off byte-parity vs bare MZSlowVars
existing Z/M/X update order       I-cell coupling is a CURRENT
resting equilibrium               3:2 pump flux identity
single-spike Na/K update          finite-volume K budget closure
E and I both contribute           zero-flux boundary net flux = 0
empty-voxel handling              grid-resolution: total budget + coarse-grained field
multi-rate convergence            no negative concentrations
checkpoint/restart identity       analytic feasibility gate (J_Na_rest >= 0 etc.)
```

六个 blessed 引擎文件**仍然一个字节都不能改**；若证明必须改，先停并另写受审阅的
guarded-engine-change spec。

---

## 14. 允许的结论层级

- **Gate H only**：离子稳态与数值合同成立；尚未证明任何网络级性质。
- **Gate H + B**：构成性离子 substrate 重新达到间期工作点，并保留锚定在两个注册核上的两个起始位点；
  尚未证明相图或生命周期。**不得**写成"保留双向传播"——该性质在 E1146 上未被确立（§9 B-real）。

即使全部通过，安全表述也只能是 **reduced ion-homeostatic mechanism on an E1146-informed spatial
scaffold**；不得声称重建了患者真实离子浓度，不得声称"证明了发作由离子稳态终止"。

---

## 15. 为什么是 Ullah–Cressman 而不是 Epileptor-2

保持 Ullah–Cressman。它与当前问题结构一致：显式 E/I 网络、局部胞外钾、泵、胶质清除、空间扩散。

Epileptor-2 同样围绕钾驱动进入与 Na/泵终止，但它是 population/QIF 约化，带突触抑制，
并假设抑制性群体率与兴奋性群体率成比例，其参数围绕 ictal bursting 而非维持当前 40k 的 E/I baseline。

分工：Ullah–Cressman 提供泵动力学、浓度参考、体积比、扩散与化学计量；本模型自己声明并验证
spike→ion 的粗粒化（§3.2，标为 effective）；Epileptor-2 留给 B3/B4 的时间尺度/轨迹敏感性，
**不作为本轮 primary 常数来源**。

---

## 16. rev2 相对 rev1 的实质改动

**读源文献后改设计（三项）**

1. **胶质缓冲从"并进线性储库"改回显式饱和项**。rev1 把浴清除与胶质摄取并成一个线性 `(K_res−K_o)/tau_K`，
   那会删掉钾积累失控所依赖的饱和正反馈。现按参考分开写，并用中心化形式让静息自动成为不动点，
   `K_res` 反推式随之删除。
2. **primary 网格 64×64 → 32×32**，因为 K 源是格内 per-cell 平均，64×64 只有约 10 个细胞/格、
   128×128 有约 9% 空格，采样噪声会直接污染钾场（§8 表）。
3. **扩散在本 sheet 尺度上是惰性的**（`D_K/Δx²` 只有 `eps` 的 0.01%–0.85%，而 Ullah 自己的
   细胞尺度网格上二者同量级）。因此 `D_K` 保留物理值不改名，停机条件 8 变成可解析预先排除，
   且网格分辨率被明确定性为"钾反馈的空间局部性"这一建模选择。

**按审阅闭合的 P0/P1**

| 项 | 处理 |
|---|---|
| P0-1 参考家族未真正锁定 | §3 逐项出处表（论文/方程/数值/单位/inherited-derived-effective） |
| P0-2 `rho` 权限自相矛盾 | §3.3：B0 固定，B0–B2 不动，B4 改动必须重走 Gate H/B |
| P0-3 K 场不是完整有限体积方程 | §4.2 显式有限体积 + `beta` 恒定的子采样论证 + 空格/边界 + 网格不变性的正确表述；Gate H 措辞改为三项，不再暗示 ion-conserving |
| P0-4 新模块无法接入 | §6 `IonHomeostaticMZAdapter` 委托顺序 + 协议表面枚举 + 四项回归测试 |
| P1 `r0` 循环定义 | §7.1 锁定出处 + 恰好一次闭合迭代 + 残差硬门 + 解析可行性门 |
| P1 nuisance 数量不一致 | §7.2 删除第三个 fallback，只留两个 bias |
| P1 Gate B 未预注册 | §9 B-real / B-model 两层 + 真实产物路径 + 层级纪律 + 方向判据 + 2×3 confirmatory |
| P1 空间单位未闭合 | §8 表（mm、Δx、边界、`D_K`、`dt_ion` 全部单值锁定） |

**新增的 B0 待验证项（两项，未确认前不得进入 B1/B2）**：

1. `1 引擎电压单位 = 1 mV` 的显式核算（§4.3 ⚠️）——直接决定 `g_K_ion` 的量级；
2. initiation-site 方向读出的功率前置（§9 B-real）——必须先在已接受的泵关轨迹上做到可评分事件 ≥ 20。

**另一项 rev2 订正（查产物后发现的事实错误）**：rev1 把"双向轴向传播"写成 E1146 的真实目标并引用了
通道层产物。实际上该被试 `candidate_forward_reverse_pairs = null`、通道层互换分数未 populated，
两个模板的整体秩相关只有 −0.464。Gate B 的真实数据判据据此改为"两个稳定模板 + 两个注册核的起始位点"，
并明确禁止写成"复现了双向传播"。
