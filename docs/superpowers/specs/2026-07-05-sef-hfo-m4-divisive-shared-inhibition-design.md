# M4 — 共享抑制池 `S_G` 的除法归一化（divisive shared inhibition）设计 spec（2026-07-05, rev4, 中文版）

> 状态：**LOCKED as M4 mechanism screen**（首轮 consult 已并入，见下）。这是机制筛查，不是发作机制验证。
> 它取代"靠加大减法项 $-\eta_G h_G$ 来抢救 M3A-v2.2"的思路。

---

## 修订说明

### rev4（2026-07-05）—— 第二轮 consult：sensor 简化为单一自然 readout

把 §3 的三 sensor 加法驱动（`M/B/P95`→rev3 的 `P_soft`）**换成一个自然 readout**：每个位置先过招募非线性
$\Psi_G(r_E)$（Hill；亚阈背景不招募、超阈才招募），再 **p-范数**池化成 $A_G$。理由：$\langle\Psi_G(r_E)\rangle$
本身就是一个**软招募面积**度量（`B` 冗余）；非线性平均比线性平均 `M` 更像神经池招募函数（`M` 冗余）；p-范数
（p=2–4）比 percentile `P95` 平滑且能让强 focal/core 提前招募（`P95` 冗余、只留作 ablation/fallback）。净效果：
**少 3 个 sensor + 3 个权重 + 2 个门限**，一个招募非线性，Jacobian 更干净。并入 §3/§4/§10.2/§11。

同改：池激活变量 `u_G` 更名 `μ_G`（pool firing/activation）；**去掉多余的 $\Phi_G$**（$A_G\in[0,1]$ 已归一化）。
§10.2 补更直观的核加权表述：实际被削掉的 recurrent 电流
$\Delta I_{E,rec,i}=I_{E,rec,i}\,\dfrac{\alpha_G S_G}{1+\alpha_G S_G}\propto I_{E,rec,i}$——哪里 recurrent 大、哪里削得多，
所以即便 `S_G` 是全局标量、稳定化仍 **core-weighted**。**`p` 取代 rev3 的 $\rho$** 作 mean↔focal 的诊断旋钮。

> rev4 判断：p-范数是 **canonical `A_G`**（不是 fallback）。因 pilot 已显示临界模态 core-localized、且 focal-core
> 提前招募是设计要求；$p=1$ 纯均值对一个孤立 core 几乎不响应，所以默认 $p\in[2,4]$、并把 $p$ 当扫描量
> （$p=1$ 是均值极限）。

### rev3（2026-07-05）—— 并入首轮 consult 反馈

首轮 consult verdict：**主方向对、锁 M4，但把 "global Hopf" 的理论先验再降一级。** 核心纠偏：

> **不要说 M4 的目标是新造 global Hopf。M4 的目标是把 core-localized 的 runaway，经延迟除法池
> 改造成一个有界、可撤回、空间上可解释的 pool-coupled ictal-core attractor。global Hopf 是**可能**结果、
> 不是默认目标；它要由 mode-shape 分数**证明**，不能从 "`S_G` 是全局标量" 推出来。**

关键论证：**"`S_G` 是全局标量" ≠ "它耦合出来的 eigenmode 是全局形状"**。哪个模态走 Hopf，由该模态同时满足
"能被 sensor 读到"且"能被 `S_G` 的反馈形状打到"决定（§10.2 的耦合分 $\kappa_k$）。而除法反馈形状
$b\propto I_E^{\text{rec},0}$ 在 core-ignition 工作点本身就是 core-weighted，加上 sensor 里的聚焦腿也偏 core，
所以最可能是 **core-coupled Hopf（breathing core）**，不是纯 global Hopf。

并入的 5 处 revision：

1. **§10.3**：把 `λ_axis → λ_global` 单一路线改成 **mode-agnostic `λ_lead` + 形状分数**，默认先验是
   $\lambda_{\text{core}}$；预注册 4 种候选结局 H0/H1/H2/H3。
2. **§10.2**：新增 `S_G`-mode 耦合分 $\kappa_k=(w_k^\top b)(c^\top v_k)$（左/右本征向量 × 反馈形状 × sensor 读出），
   直接回答"`S_G` 主要耦合 core / axis / global / mixed 哪个模态"。
3. **§4**：显式固定延迟 $\Delta_G$ 会把 Pass-2 变成 DDE（超越特征方程），与 §10.2 的有限维增广 Jacobian
   **矛盾** → **第一版锁 $\Delta_G=0$**，相位滞后完全由两级低通 $\tau_\mu,\tau_S$ 提供；要加延迟走
   低通级联近似（仍有限维），真 DDE 谱 out-of-scope。
4. **§8.1**：Pass-1 主平面从 `q_axis × alpha_G` 改成 **`q_core × alpha_G`**（pilot 的领头模是 core-localized，
   扫 `q_axis` 会把真正控制变量平均掉）；`q_axis × alpha_G` 降为次级平面。
5. **§9.1 / §8.5**：**空间局部化有界核（region 3）是 primary target**；whole-field synchronized burst 只作
   secondary candidate。`S_G` **第一轮保持 uniform scalar**；若唯一成功是 whole-field rhythmic burst，
   下一杠杆是**低秩空间化池**（soft-core 读出 + broad-nonuniform 反馈），不是更强的均匀全局池（§8.5）。

另外并入两个技术改动：**§3** 把 `P95` 换成光滑 log-sum-exp 传感器 `P_soft`（percentile 对 frozen-Jacobian
不可导；`P_soft` 可导且 $\rho$ 扫描能区分 global-readout vs core-readout）；**§9.1** 把硬判据
"$B_\delta\le 1$ 持续" 放松成 **窗口/周期平均 $\overline{B_\Delta}\le 1+\epsilon$**（真 bounded oscillator 在每个
burst 上升相瞬时 $B_\delta>1$ 属正常，runaway 的标志是"持续尾抬 + 单调饱和 + 不恢复"）。

**两个原 consult 问题已解决：**（主）core-localized 起点 + 延迟均匀 `S_G` → 默认**不是** global Hopf、
而是 core-coupled Hopf（§10.3）；（次）whole-field burst = secondary、primary 锁 localized core（§8.5）。

### rev2 血缘（背景）

rev1 = `S_G` 除法骨架，整合 **Consult #1**（Chance/Abbott 除法：保留局部椭圆指数 E/I 核，新增 low-rank 共享池，
对 recurrent E 增益归一化、不是减标量）。rev2 并入 **Consult #2**（局部等效抑制资源 `R_L` + 延迟全局反馈 `G`、
相图、模态 $\lambda$ 读法、SSN/ISN + 平衡网络 Hopf + 癫痫文献）+ 读出拆两遍（§10）+ Pass-1 空间守卫（§9.1）。

**命名守卫：** `S_G` 的 "global" 是**机制变量**（一个池化抑制电流）；§10 里的 "global mode" 是**本征模形状**——
不能混。`S_G` / 本 M4 线不撞 `V3b`（Topic 5 cross-ref 里指 M3B 谱线）。`q_I` 保留为场名；Consult #2 的 `R_L`
是同一个对象（§2）。

---

## 0. 一句话结论

用 Abbott / 除法抑制的思路，落地成一个**共享抑制池对 recurrent E 输入做归一化**。承重改动是**除法**：

$$
I_E^{\text{rec}}\ \longrightarrow\ \frac{I_E^{\text{rec}}}{1+\alpha_G\,S_G}
$$

而**不是**减法 $I^{\text{net}}\to I^{\text{net}}-\eta_G h_G$。

- **局部 $q_I$** 决定轴向核 / 波前是否失去局部约束；
- **共享 $S_G$** 决定 recurrent E 增益是否在 runaway 饱和之前被归一化。

**最强科学目标（rev3 定调）：** 不是证明"全局同步 = 发作"，而是证明一个 **core-localized 的病理点火，
可以被共享除法抑制从 runaway 改造成有界、可撤回、空间上可解释的发作样 core**。global Hopf 是可能结果、
不是目标。

**交付顺序：** 先做一张**遍1 仿真相平面**（`q_core × alpha_G`），回答"在 transient 与 runaway 之间，是否存在
一个**非平凡的有界中间区**？"（go/no-go，§9.1）——**在**任何长闭环搜索**之前**，也**在**遍2 冻结-Jacobian
本征模刻画（§10.2）**之前**。

## 1. 为什么旧的 `h_G` 不够

M3A-v2.2 现有的全局变量是 E 细胞上的一个标量减法电流：

$$
I^{\text{net}}_{E}=I_E-q_I(x,t)\,I_I-\eta_K\,g_K(x,t)-\eta_G\,h_G(t)
$$

它把膜输入整体往下平移，但**不直接改变有效 recurrent E 增益**。一旦 recurrent E→E 接力饱和进 runaway，
一个标量减法有两个常见失败模式：**太弱**（拉不回已饱和的 recurrent avalanche）或**太强**（把整片压成静默、
把成核打碎）。M3A-v2.2 诊断观察到的正是这个边界：`h_G` 能**感知**全局性，但 E-only 的减法刹车**不能反转**
runaway 轨迹。

Chance / Abbott 的除法抑制针对**另一个对象——recurrent 放大本身**：把 recurrent 输入**除以**一个共享抑制
变量、保留 feedforward 不变（原型见 §附录 A）。这就是"新建 M4、而不是调 $\eta_G$"的核心科学理由。

**第一性原理重述：** 减法平移工作点、不改增益，所以自放大 recurrent 环只会"欠刹"（→ runaway）或"过刹"
（→ 静默），中间没有有界态。除法归一化**直接作用于增益**——这正是 stabilized-supralinear / inhibition-stabilized
把一个**强** recurrent 网络稳定在**有界高增益态**的办法。但注意（rev3）：除法的第一功能是**稳定 recurrent 放大**、
**不是**自动制造全局同步振荡；要出 Hopf 还需要足够的负反馈 loop gain + 相位滞后（§4 两级低通提供），且
Hopf 落在哪个空间模态由 §10.2 的 $\kappa_k$ 决定。

## 2. 状态变量

保留现有的两个空间慢变量场：

- $q_I(x,t)$：局部抑制资源 / 有效局部抑制标度（随局部活动耗竭、回充到 1）。
- $g_K(x,t)$：局部疲劳 / sAHP 恢复刹车（随局部 E 活动累积、衰减到 0）。

> **`q_I` == Consult #2 的 `R_L`。** Consult #2 把局部抑制归成一个"局部等效抑制资源" $R_L(x,t)$（揉合
> Cl⁻ / K⁺ / interneuron 可用度 / GABA 极性）。这跟引擎里 `q_I(x,t)` 是**同一个对象**。保留场名 `q_I`
> （不重命名、不 churn 代码）；文献口径里把 `R_L` 与 `q_I` 读作同义。

新增一个**快共享抑制池**（两个标量）：$\mu_G(t)$（池激活 / firing，第一级低通）、$S_G(t)$（共享抑制池输出，进膜的那个）。

旧的 `h_G` 只保留作**次级慢的爆后约束** $H_G(t)$（慢全局 refractory / 恢复臂，主筛查中关闭）。

命名规则：新除法池一律用 `S_G`；**不要**把 `h_G` 复用给这个机制（`h_G` 在 M3A-v2.2 里已表示均匀减法恢复）。

## 3. 传感器（sensors）—— 单一自然 readout（rev4）

不用三 sensor 加法驱动。用**一个自然 readout**：每个位置先过一个招募非线性 $\Psi_G$（Hill；亚阈背景不招募、
超阈才招募），再 p-范数池化。局部招募（per-location recruitment，$[\,\cdot\,]_+=\max(\cdot,0)$）：

$$
z_G(x,t)=\Psi_G\big(r_E(x,t)\big),\qquad
\Psi_G(r)=\frac{[r-r_0]_+^{\,n}}{r_{50}^{\,n}+[r-r_0]_+^{\,n}}
$$

池化驱动（p-范数；因 $\Psi_G\in[0,1]$，故 $A_G\in[0,1]$）：

$$
A_G(t)=\Big[\big\langle z_G(x,t)^{\,p}\big\rangle_x\Big]^{1/p}
$$

- $\Psi_G$：把率场逐点变成"这里被招募了吗"的软 0/1（$r_0$ 起点、$r_{50}$ 半招募、$n$ 陡度）。
- $\langle z_G\rangle_x$（$p=1$）本身就是**软招募面积**：小 core → $A_G$ 小、大面积 → $A_G$ 大——所以旧的 `B`（软面积）
  与 `M`（线性均值）都**冗余**、去掉（$\langle\Psi_G(r_E)\rangle\ne\Psi_G(\langle r_E\rangle)$，逐点招募再平均更像
  神经池招募函数）。
- **`p` 是 mean↔focal 旋钮**：$p=1$ 趋软面积/均值端，$p=2$–$4$ 让强 focal/core 提前招募（回收 `P95` 的动机、但
  **光滑可导、Jacobian 友好**）。**canonical 取 $p\in[2,4]$ 并把 $p$ 当扫描量**（pilot 显示临界模态 core-localized、
  focal 提前招募是设计要求；$p=1$ 对一个孤立 core 几乎不响应）。`p` 取代 rev3 的 $\rho$ 作 global↔core readout 诊断。
  旧 `P95` 只留作 ablation/fallback。
- 因 $A_G\in[0,1]$ **已归一化**，**不再需要 rev3 的 $\Phi_G$**（招募非线性已在 $\Psi_G$ 里、放在更合理的
  per-location 层）；仅当某轮 $A_G$ 未归一化，才在 §4 池前套一层 Hill $\Phi_G$。
- 只有当"强 focal core"与"大面积中等活动"在 $A_G$ 上不可区分、且你希望池对两者反应**不同**时，才加**第二个**读出
  $A_G=w_{\text{amp}}A_{\text{amp}}+w_{\text{area}}A_{\text{area}}$——**第一版不加**（YAGNI）。

## 4. 池动力学（pool dynamics）

**两级低通**（$\dot{(\cdot)}\equiv \mathrm d/\mathrm dt$；rev4：$A_G$ 已在 $[0,1]$、不再套 $\Phi_G$；**第一版 $\Delta_G=0$**）：

$$
\tau_\mu\,\dot\mu_G=-\mu_G+A_G(t),\qquad
\tau_S\,\dot S_G=-S_G+S_{\max}\,\mu_G
$$

界（clip）：$0\le\mu_G\le 1$，$0\le S_G\le S_G^{\max}$。$\mu_G$ 是池激活 / firing、$S_G$ 是进膜的池输出。

两级形式给出**延迟负反馈环**：E 活动升 → $A_G$ 升 → $\mu_G$ 升 → $S_G$ 升 → recurrent E 增益降（经 §5 除法）
→ E 活动降 → $S_G$ 降。这一环是预期能造出**有界振荡 / burst 包络**的部分。

> **延迟 $\Delta_G$ 的处理（承重）。** 第一版 $A_G$ **无显式延迟**（$\Delta_G=0$），相位滞后**完全由两级低通
> $\tau_\mu,\tau_S$ 提供**。若给 $A_G$ 加**真固定延迟** $\Delta_G>0$，Pass-2 特征方程里会出现 $e^{-\lambda\Delta_G}$，
> 就**不是**有限维 dense Jacobian 而是 **DDE / 延迟本征谱**——与 §10.2 的有限维增广 Jacobian 矛盾。要加延迟又
> 保持有限维：用**低通级联**（3–5 级）近似；真做 DDE 谱工程成本大、**out-of-scope**（第一版不做）。

时标：$\tau_{S_G}\sim 20$–$300$ ms（快，管包络 / 同步）；$\tau_{H_G}\sim 0.5$–$5$ s（慢，次级，管爆后
refractory / offset）；$q_I/g_K$ 保持现有时标。

> **时标经验锚（Consult #2 §5）：** Curot et al.（2023）报告人类 fast ripple 之后跟一段大范围放电**抑制**、
> 量级约 **200–800 ms**，与慢 GABA / down-state / feedforward inhibition 一致——这就是快 `S_G`（20–300 ms）与较慢
> `H_G`（0.5–5 s）的经验依据。快 / 慢分离对应 Consult #2 的 $G=G_{\text{fast}}+G_{\text{slow}}$。

## 5. 必需的膜方程（membrane equation）

E 细胞输入必须区分 **feedforward** 与 **recurrent** 兴奋：$I_E=I_E^{\text{ff}}+I_E^{\text{rec}}$。
定义除法因子 $D_G(t)=1+\alpha_G\,S_G(t)$。

**M4 主方程**（承重：只除 recurrent E 电流，feedforward 不动）：

$$
\boxed{\,I^{\text{net}}_{E,i}=I_{E,i}^{\text{ff}}+\frac{I_{E,i}^{\text{rec}}}{D_G(t)}-q_I(x_i,t)\,I_{I,i}-\eta_K\,g_K(x_i,t)\,}
$$

**可选 hybrid 臂**（额外一个**小**减法 $S_G$ 项 + 慢 $H_G$）：

$$
I^{\text{net}}_{E,i}=I_{E,i}^{\text{ff}}+\frac{I_{E,i}^{\text{rec}}}{1+\alpha_G S_G(t)}-q_I(x_i,t)\,I_{I,i}-\beta_G\,S_G(t)-\eta_K\,g_K(x_i,t)-\eta_H\,H_G(t)
$$

$-\beta_G S_G$ 可以作为一个小的共享抑制电流存在，但**它不是主机制**；主机制是对 recurrent E 输入的
**除法归一化**。

I 细胞在主筛查中不变：$I^{\text{net}}_{I,i}=I_{E,i}-I_{I,i}$。次级扫描**可以**让 `S_G` 也作用到 I 细胞，
但只在 E-only 主臂有明确 verdict **之后**（否则 `S_G` 一边压 E、一边经压 I 去抑制 E，机制不可解读）。

## 6. 在当前 SNN 里的工程含义

**不能**只改 `SpatialSlowField.apply_currents()` 就把 M4 正确实现，因为当前 `simulate_kick()` 只提供**一路合并的
AMPA 电流**。

**已对代码核实（2026-07-05）：** 在 `src/snn_engine/kick_probe.py::simulate_kick` 里，突触门 `s_E` 在单次 AMPA
低通**之前**就把 recurrent E→E ring 到达与外部驱动加在一起：

```text
s_E *= decay_sE                 # 约 L234  AMPA rise 衰减
s_E += ring_sE[slot]            # 约 L237  RECURRENT E->E / E->I 到达（delay ring）
s_E += ext * ext_incr           # 约 L246  EXTERNAL Poisson / kick / ramp 驱动
I_E   = s_E + (I_E - s_E)*decay_IE   # 约 L250  单路合并 AMPA 电流
```

所以到达 `slow.apply_currents(I_E, I_I, labels)`（约 L257）的 `I_E` 已经是 `external + recurrent` 合并。
所以到达 `slow.apply_currents(I_E, I_I, labels)`（约 L257）的 `I_E` 已经是 `external + recurrent` 合并。

**首选实现（parity-preserving，本轮已实现并锁定）：不字面拆 `s_E`，而是保留合并 `I_E` 一字节不动，另外并行追踪
一路"只含 recurrent 到达"的 `I_E_rec`，除法效果写成减去被削掉的 recurrent 电流：**

$$
\Delta I_{E,rec}=I_{E,rec}\,\frac{\alpha_G S_G}{1+\alpha_G S_G},\qquad
I^{\text{net}}_{E}=I_E-\Delta I_{E,rec}\ \equiv\ I_E^{\text{ff}}+\frac{I_E^{\text{rec}}}{1+\alpha_G S_G}
$$

代数等价：$I_E-I_{E,rec}\frac{\alpha_G S_G}{1+\alpha_G S_G}=(I_E-I_{E,rec})+\frac{I_{E,rec}}{1+\alpha_G S_G}=I_E^{\text{ff}}+I_E^{\text{rec}}/D_G$。
**这比字面拆 $s_E^{\text{ff}}/s_E^{\text{rec}}$ 再相加更利于 parity**：当 $\alpha_G S_G=0$，$\Delta I_{E,rec}$ **精确为 0**、合并 `I_E`
完全不动 → **逐字节 parity**；字面两路拆分因浮点非结合（$ac+bc\ne(a+b)c$）只能 allclose、拿不到精确 parity。

实现（已锁，见 `kick_probe.py` / `slow_field.py`）：(1) `simulate_kick` gated on `use_SG` 额外累一路 `s_E_rec`
（**只加 `ring_sE[slot]`，且必须在该 slot 被清零之前读**——否则 `I_E_rec` 读到 0、除法静默失效）→ `I_E_rec`
（同一个 `decay_IE` 低通）；合并 `I_E` 累加不变。(2) `use_SG` 时把 `I_E_rec` 传给 4-参 `apply_currents`，在 E 细胞减
$\Delta I_{E,rec}+\beta_{SG}S_G$（代码里 `beta_SG`，避开已存在的 `beta_G` 配置字段）。(3) 默认 `use_SG=False` 不追踪、
走原 3-参 `apply_currents` → 逐字节 parity。(4) `use_SG` + `alpha_G>0/beta_SG>0` 但缺 `I_E_rec` → **RuntimeError**
（防 `S_G` 建起来却对膜无效的静默假阴性）。

**不要除总的 `I_E`。** 除 feedforward 触发输入会把科学问题从"稳定 recurrent runaway"变成"衰减触发"。

## 7. 臂（arms）

| 臂 | 抑制 / 归一化项 | 目的 | 遍1 角色 |
| --- | --- | --- | --- |
| 0 | 现有 $q_I+g_K+h_G$ | 历史 negative / 减法基线 | **主基线** |
| 1 | $q_I+g_K+\beta_G S_G$（减法）| 只共享减法池 | **负对照**（"只会压活动"）|
| 2 | $q_I+g_K+\alpha_G S_G$（除法）| Abbott 式 recurrent 增益归一化 | **主信号** |
| 3 | 除法 $S_G$ + 小减法 $S_G$ | hybrid 池 | 延后（Pass-1 go 之后）|
| 4 | hybrid + 慢 $H_G$ | 爆后 refractory / 终止（次级）| 延后（Pass-1 go 之后）|

**遍1 最小消融 = 臂0 vs 臂2，臂1 作负对照。** 臂 3 / 4 只在遍1 返回 "go" 后跑。预期：臂1 只能靠**压活动**成功
→ 不够（Consult #2 "region 6"）；臂2 降 runaway、开**有界高增益区** → 主信号；臂3 拓宽有界窗；臂4 改善 no-rebound。

## 8. 相平面（phase planes）

**遍1 主 = §8.1（`q_core × alpha_G`）；§8.2 次级；§8.3 延后；§8.4 可选；§8.5 = `S_G` 空间结构的下一步预案。**

Consult #2 把同一个对象写成 2-D 相**图**，发作样态是一个**区域**、不是一个点——六个定性角：interictal-only /
near-critical IED / 局部有界核 / 同步有界振荡 / runaway / 压制。§8.1 是它的可操作实例。

### 8.1 `q_core` × `alpha_G`（遍1 主，rev3 从 `q_axis` 改来）

pilot 已发现领头线性模态是 **core-localized**（不是 axis mode），所以真正的控制变量是**核内**资源，不是
轴向平均资源。定义核内平均资源（$m_{\text{core}}(x)$ 是从 pilot ignition core / 领头模功率 / 实际
kick-triggered first-activation map **导出**的核 mask，非人为指定）：

$$
q_{\text{core}}=\frac{\sum_x m_{\text{core}}(x)\,q_I(x)}{\sum_x m_{\text{core}}(x)}
$$

扫 $q_{\text{core}}\in[q_{\min},1]$、$\alpha_G\in[0,\alpha_{\max}]$，把**同一个** IED 样脉冲分类为：decay /
near-critical transient / 局部有界核 / 有界同步发作样候选 / runaway / 压制（无成核）。

预期 M4 好处不只是"更少 runaway"，而是 transient 与 runaway 之间一个**非平凡的有界中间区**。这张平面回答：
① M4 能否把 core runaway 变成 bounded core？② 这个 bounded core 是否随后产生 axis / off-axis recruitment？

**次级平面：`q_axis × alpha_G`**（$q_{\text{axis}}=\langle q_I\rangle_{\text{axis}}$）保留——用来回答"轴向结构到底是
线性 softening 还是非线性足迹"，不丢轴向视角。

### 8.2 `alpha_G` × `tau_S`（遍1 次级）

**时标平面**。预期：全局反馈**太快** → 压制 / 碎片化；**中间** → 有界振荡 burst；**太慢** → 反馈到达前已
runaway。Consult #2 预测有界同步振荡只在**中间**的全局反馈延迟窗（延迟负反馈 Hopf 窗）里存在。仅当 §8.1
出现有界带时才跑。（注意 §4：延迟由 $\tau_\mu,\tau_S$ 提供、$\Delta_G=0$。）

### 8.3 `q_core` × `gK_core`（延后）

测可逆性 / 终止：$q_I\downarrow$ 起始 / 扩张；$g_K\uparrow$ 限界并终止；$q_I$ 恢复 + $g_K$ 衰减 → 爆后恢复。

### 8.4 `gamma_G` × `q_core`（延后 / **可选** —— 不是主机制）

> **Consult #1 明确建议不要重写局部核。** 下面 local-global 抑制核形式把局部 $q_I$ 约束与全局 $S_G$ 混进一个
> 抑制项，只作可选次级臂。M4 主机制是 §5 的**除法归一化**，它**不动**局部椭圆指数 E/I 核。

$$
I^{\text{inh}}_{\text{eff}}(x,t)=g_I\Big[(1-\gamma_G)\,q_I(x,t)\,(K_{I,L}*r_I)(x,t)+\gamma_G\,S_G(t)\Big]
$$

扫 $\gamma_G$ × $q_{\text{core}}$：$\gamma_G$ 太低 → 局部失守变 runaway；中间 → 局部有界核；太高 → 早全局同步 / 压制。

### 8.5 `S_G` 空间结构：第一轮 uniform，成功但退化则上低秩空间池（rev3）

**第一轮 `S_G` 保持 uniform scalar** —— 现在要测的是一个很清楚的新 lever（recurrent 增益归一化）；若第一轮就把
`S_G` 做成空间结构，结果会混入新的 spatial kernel / `D_EE` / local-global kernel 效应，机制解释变脏。

但**若 uniform `S_G` 的唯一成功结果是 whole-field rhythmic burst**（无空间结构），**不要把它当终点**。下一步引入
**低秩但空间化的池**（不是更强的均匀全局池）：

$$
K_G(x_i,x_j)=v_G(x_i)\,u_G(x_j)
$$

（这里 $u_G(x),v_G(x)$ 是低秩核的**空间**读出/投射权重，Consult #1 记法，**与池激活标量 $\mu_G(t)$ 无关**。）

- 第一轮：$u_G(x)=1,\ v_G(x)=1$（= 当前 uniform 池）。
- 第二轮：$u_G(x)=$ soft-core（p-范数 $\Psi_G$）读出（让核强招募全局反馈），$v_G(x)=v_0+v_{\text{peri}}\,K_{\text{broad}}(x,x_{\text{core}})$
  （**broad 但非均匀**的反馈，不一定完全均匀打回全场）。
- 或多低秩池：$S_0(t)$（uniform normalization）+ $S_{\text{peri}}(x,t)$（broad surround / peri-core restraint）。

目的：保住 region 3（空间局部化有界核），避免 region 4 退化成无空间结构的 common-drive oscillation。

## 9. 成功门（success gates）

### 9.0 完整发作样目标门（长期标准）

一个**完整**主候选必须**全部**满足：(1) 触发后自维持；(2) 有界（全局 E 率无单调饱和）；(3) 足够大范围招募；
(4) 同步或共享包络（可测群体节律 / 谱峰）；(5) **源空间读出**（区分 expanded axial 与 off-axis / 全局招募）；
(6) HOLD / 无外部释放下自发终止；(7) `S_G`/`H_G` 衰减后无 rebound；(8) 动态-vs-静态对照（matched-static 池不能
解释同一轨迹）。

保持长期 M3 红线：**expanded axial recruitment $\neq$ seizure-like state**。**primary target = 空间局部化、有界、
自维持、有组织招募的 core（Consult #2 region 3）**；whole-field synchronized burst 只作 secondary candidate（§8.5）。

### 9.1 遍1 预注册 go/no-go（带空间守卫）—— **跑之前锁死**

遍1 是 go/no-go **筛查**，任务：判断 divisive 臂是否开出一个**非平凡有界中间区**、值得投入遍2——同时**当场**
排除这块衬底已知会产的两个平凡有界态。

**逐格读数。** 对每个 `(q_core, alpha_G)` 格（一次仿真）算：`persist`（kick 后维持 $>T_{\min}$，否则 decay）、
$M_{\text{peak}}$/`sat`、$B_\delta$（spike-count 分箱经验分支比）、`act_frac`（招募面积分数）、$S_{\text{grad}}$
（源空间 onset 梯度分 = 是否有**空间点火序列**）、$F_{\text{off}}$（off-axis 招募分数）、`core_overlap`、
`globality`、`mode_class`$\in\{\text{core, axial, global, off-axis}\}$（复用 M2 形状分）。

**要排除的两个平凡有界态（拿真实实例标定阈值）：**

- **TRIVIAL-A（全场同步 / 全场 skirt）**：`act_frac` 高 且 `core_overlap` $>\theta_{\text{core}}$ 且
  `globality` $<\theta_{\text{glob}}$（低幅全场 skirt 骑在核加权功率上——衬底已知的相干全场招募被变成节律）。
- **TRIVIAL-B（expanded axial）**：onset 轴对齐 且 $F_{\text{off}}<\theta_{\text{off}}$ 且自限（退回核）
  （一个更大的间期轴向事件，不是发作）。

阈值 $\theta_{\text{core}},\theta_{\text{glob}},\theta_{\text{off}},T_{\min},A_{\min},\text{sat},K_{\min},\epsilon$
**不拍脑袋**。标定流程（跑一次然后锁）：从**当前**模型（臂0）产出 TRIVIAL-A / TRIVIAL-B 参考实例（全场铺开收尾 与
自限 expanded-axial，这块衬底本来就有，见 criticality M2 pilots, 2026-07-04），量其
`core_overlap`/`globality`/$F_{\text{off}}$/$S_{\text{grad}}$，把阈值设成**恰好排除**这些参考实例，扫描前锁进标定表。

**有界判据（rev3 放松）：** bounded 不用瞬时 $B_\delta\le 1$（真 bounded oscillator 在每个 burst 上升相瞬时
$B_\delta>1$ 属正常），改用**窗口 / 周期平均**——四条同时成立才算 bounded：窗口/周期平均分支比
$\overline{B_\Delta}\le 1+\epsilon$；全局率 $M(t)$ 无单调饱和；尾部回到 baseline；事件能量有限。
runaway 的真判据是**持续尾抬 + 单调饱和 + 不恢复**，不是瞬时 $B_\delta>1$。

**go(cell)** 要求全部：`persist` 且 `bounded` 且 `act_frac` $\ge A_{\min}$ 且 **非 TRIVIAL-A** 且 **非 TRIVIAL-B**。
直白讲：一个有界、自维持、大范围的事件，有真正的空间 onset 序列（$S_{\text{grad}}$ 存在），功率**不是**核加权全场
skirt，扩散**突破纯轴向**。

> **TRIVIAL-A vs 真正分布式同步爆（不要混）。** TRIVIAL-A 只排**低幅**全场 skirt（`core_overlap` 高、`globality`
> **低**）。一个**高幅分布式**同步爆（`globality` 高、非核加权）**不是** TRIVIAL-A，它通过 go(cell) 作 candidate。
> 但按 rev3 定调它是 **secondary**、不是 primary target（§8.5 / §9.0）；是否算发作样留给遍2 / gate 5 / gate 8。

**go(plane)**：至少 $K_{\min}$ 个**连通**的 go(cell)（一个**面积**、非单点、非数值边缘），且这个 go-区出现在
**臂2（除法）但不在臂1（纯减法）**。

**no-go**（任一）：只有臂1 开出有界区；唯一有界格是 TRIVIAL-A / TRIVIAL-B；go-格不成连通面积。**一个干净的 no-go
是合法结果**：说明除法池没加出非平凡有界态，**加强**长期"下一个杠杆 = `D_EE` / 换衬底"的结论（§12），而不是把
M4 悄悄证伪。

## 10. 读出（readouts）—— 两遍

### 10.1 遍1 仿真读出（自包含、现在就能跑）

直接从仿真轨迹算，不依赖 criticality 机器。至少报：$T_{\text{rec}}$（持续回到 baseline$+c\sigma$ 以下的时间）、
$B_\delta$（分支比）、$M_{\text{peak}}$、`PR / globality`、$S_{\text{axis}}$（源空间 onset-轴分）、$F_{\text{off}}$、
包络谱峰 / 节律分、`tail_to_baseline_ratio`。这些直接喂 §9.1。**near-critical 主张**只在恢复变慢、分支比逼近 1、
事件保持有界时才允许。

### 10.2 遍2 冻结-Jacobian 本征模读出 + `S_G` 耦合分（gated on criticality M1/M2 合并）

仅在遍1 "go" 之后、且 criticality 机器在 M4 base 上可用时跑。

**复用**（`src/topic4_m3b_spectral_phase.py` + `src/topic4_criticality.py`）：`solve_operating_point`（冻结
`q_I/g_K`）、`build_jacobian_dense` / `rate_eigenpairs` / `spectral_gap`、M1/M2 的复本征对不变子空间 loading +
**左本征向量**、M2 形状分（`core_overlap`、`globality`、`elongation_axis_score`、`axis_wavevector_alignment`、
`off_axis`）、`finite_time_gain` / 数值 abscissa。

**扩展**（真活）：`S_G` 是**快变量**、不能冻住，作为**状态**进快 Jacobian。增广快系统（$\Delta_G=0$，§4）：

$$
\dot{\delta r}=A_0\,\delta r+b\,\delta S_G,\qquad
\tau_\mu\,\dot{\delta\mu_G}=-\delta\mu_G+c^\top\delta r,\qquad
\tau_S\,\dot{\delta S_G}=-\delta S_G+S_{\max}\,\delta\mu_G
$$

其中 $A_0$ 是无-`S_G` 的率场 Jacobian；$c=\partial A_G/\partial r_E$ 是 sensor 对率场的读出梯度（rev4：由
$A_G=[\langle\Psi_G(r_E)^p\rangle]^{1/p}$，$c$ 权重偏向**高活动 cell**，即 core-ignition 时偏 core）；$b$ 是
`S_G` 反馈进率场的空间形状。**关键：$b$ 不是 spatially uniform**——由除法主方程：

$$
b_i=\frac{\partial}{\partial S_G}\!\left[\frac{I_{E,i}^{\text{rec}}}{1+\alpha_G S_G}\right]\Bigg|_{0}=-\frac{\alpha_G\,I_{E,i}^{\text{rec},0}}{(1+\alpha_G S_G^0)^2}\ \propto\ I_{E,i}^{\text{rec},0}
$$

有限变化的直观版（consult 定调）：实际被削掉的 recurrent 电流

$$
\Delta I_{E,rec,i}=I_{E,rec,i}-\frac{I_{E,rec,i}}{1+\alpha_G S_G}=I_{E,rec,i}\,\frac{\alpha_G S_G}{1+\alpha_G S_G}\ \propto\ I_{E,rec,i}
$$

哪里 recurrent E 输入大、哪里被削得多。在 core-ignition 工作点 $I_E^{\text{rec},0}$ 本身 core-weighted，所以
**即使 `S_G` 是全局标量，反馈 $b$ 与读出 $c$ 都 core-weighted → $\kappa_{\text{core}}$ 双重偏大**——这就是
rev-note "global 机制变量 ≠ global eigenmode" 的来源。

**`S_G`-mode 耦合分（新，回答"`S_G` 耦合哪个模态"）：** 对 $A_0$ 的每个模态（右本征向量 $v_k$、左本征向量 $w_k$）：

$$
o_k=c^\top v_k,\qquad p_k=w_k^\top b,\qquad \kappa_k=o_k\,p_k
$$

其中 $o_k=c^\top v_k$ 是 sensor 读到模态 $k$ 的强度、$p_k=w_k^\top b$ 是反馈形状 $b$ 打到模态 $k$ 的强度。
$\kappa_k$ 是一阶（弱耦合）估计：**$\kappa_k$ 大的模态才是被延迟反馈推成 Hopf 的候选**（真 Hopf 仍以增广
Jacobian 的实际本征值为准，$\kappa_k$ 告诉你**哪个无-`S_G` 模态**被池耦合）。若 $\kappa_{\text{core}}\gg
\kappa_{\text{axis}},\kappa_{\text{global}}$ → 不要再期待 global Hopf，正确表述是"M4 把 core-localized runaway
变成 pool-coupled bounded core oscillation"。

**Hopf** = 一对复共轭本征值穿 $\operatorname{Re}(\lambda)\to 0^-$、$\operatorname{Im}(\lambda)=\omega\neq 0$；
near-critical 读出：$\alpha_1=\operatorname{Re}(\lambda_1)\to 0$、$\tau=-1/\alpha_1\to\infty$、模态形状（形状分判它
是 core / axis / global / mixed）、非正规 `finite_time_gain`。**扫 sensor 的 $p$（§3）**看 `S_G` 是 global-
还是 core-readout。

**Framing 锁：** preliminary；verdict 措辞必须说 "actual M4 **SIMULATION** trajectory"、**绝不**说 "real data"。

### 10.3 主预测（rev3）：mode-agnostic `λ_lead`，默认 core-Hopf；预注册 4 结局

**不预设** $\lambda_{\text{axis}}\to 0$ 之后一定 $\lambda_{\text{global}}\to i\omega$。因为无-`S_G` 的冻结-Jacobian
pilot（criticality M2, 2026-07-04）发现当前失控起点的线性临界模态是 **core-localized nucleation**
（`core_overlap`$\approx 0.99$、`globality`$\approx 0.11$、~24 Hz 复对），双核评估仍塌到单核（"axial-corridor
takeover" 被证伪），轴向→全场是**非线性足迹**、不在线性谱里。

**核心表述（consult 定调，EN + 中文）：**

> Because the no-`S_G` frozen-Jacobian pilot identifies a core-localized critical mode, M4 does **not** assume an
> axial-to-global eigenmode sequence. The primary hypothesis is that the delayed divisive pool couples to whichever
> high-gain mode has the largest readout–feedback overlap ($\kappa_k$). Given the current substrate, the expected
> first success is a **pool-coupled bounded core oscillation**, not necessarily a true global Hopf. A true global
> Hopf is one possible Pass-2 outcome and must be demonstrated by mode-shape scores, not inferred from the global
> scalar nature of `S_G`.

中文：不要说 M4 目标是新造 global Hopf；说 M4 目标是把 core-localized runaway 经延迟除法池变成有界的
pool-coupled ictal-core attractor。global Hopf 是可能结果、不是默认目标。

**遍2 预注册 4 种候选结局**（用 $\kappa_k$ + 形状分裁，不看 `S_G` 是不是标量）：

- **H1 — core-Hopf / breathing-core（最可能，= M4 成功）：** $\lambda_{\text{core}}\to\sigma\pm i\omega$，模态仍
  core-localized、`globality` 低/中；仿真上是局部核有界振荡、或伴弱同步 envelope。尤其若它把原 runaway 改成
  自限 core event，视为**成功**。
- **H2 — mixed core-global Hopf（中等可能）：** $\lambda_{\text{mixed}}\to\sigma\pm i\omega$，模态有 core loading
  也有明显 distributed loading；仿真上核心先起、随后更大区域同步参与（最接近 region 3 → region 4）。
- **H3 — true global Hopf（可能，但不作默认先验）：** $\lambda_{\text{global}}\to i\omega$，模态高 `globality`、
  低 `core_overlap`；只有 global mode 本身接近 marginal、或 `S_G` 的 readout/feedback 对 uniform mode 耦合很强时
  才合理。
- **H0 — 无 Hopf、只是非线性收尾被重塑：** 增广 Jacobian 仍是 core-localized 实失稳，但仿真变 bounded → 说明
  `S_G` 主要靠非线性 clipping / gain compression 改写收尾、不是产生局部分岔。**不能叫 `λ_global → iω`**，但也不是坏结果。

任一结局都如实报。"仍核点火 / 无 Hopf" 会把下一杠杆指回 `D_EE` / 衬底，而不是继续调 `S_G`。

## 11. 实现顺序

Cheap-first（遍1 先；遍2 gated）：(1) `simulate_kick` AMPA 拆分测试（parity）；(2) 实现 `I_E_ff/I_E_rec`，
$\alpha_G=0$ byte-parity；(3) 纯 sensor helper `Psi_G/z_G/A_G`（p-范数，无 M/B/P95/Phi_G）；(4) 给 `SpatialSlowField`（或新池对象）加
`S_G`（$\Delta_G=0$）；(5) field-level 测试：`S_G` 会 build、有界、**只**除法缩放 recurrent E；(6) tiny SNN smoke：
$\alpha_G=0$ 等旧输出、高 $\alpha_G$ 降 runaway 倾向；(7) **导出 $m_{\text{core}}$ 核 mask + 标定 §9.1 守卫阈值**
（拿臂0 的 TRIVIAL-A/B 参考实例），锁死；(8) 跑 **遍1 相平面 §8.1（`q_core × alpha_G`）**（臂0 vs 臂2，臂1 对照）
→ §9.1 go/no-go；(9)（仅当 go、且 criticality M1/M2 合并后）跑 **遍2** §10.2（增广 Jacobian + $\kappa_k$）。

不要从大闭环网格开始。第一个科学产物是那张小的**遍1 相平面**，显示除法 recurrent-增益控制是否造出了减法 `h_G`
臂**缺失**的那个有界中间区。

## 12. 参考与本地锚

### 12.1 本地锚（本仓库）

- 当前方程：`docs/snn_core_model_equations.md` §B5（`q_I`/`g_K`）与 §B6（`h_G`）。
- 当前减法 `h_G` 实现：`src/snn_engine/slow_field.py::SpatialSlowField`。
- 合并-AMPA 电流点（拆分目标）：`src/snn_engine/kick_probe.py::simulate_kick`（约 L234–L257，2026-07-05 核实）。
- M3A-v2.2 negative 边界：`docs/archive/topic4/m3a_v2_2_carrier_exploration_2026-06-29.md`。
- criticality 机器（遍2 复用 / 扩展）：`src/topic4_m3b_spectral_phase.py`、`src/topic4_criticality.py`；M1 verdict
  仪器 + M2 模态分解 pilot（2026-07-04）发现当前临界模态是核点火。
- M3A 线长期"下一个杠杆"结论：M3A-v2/v2.1/v2.2 在均匀衬底全部 NEGATIVE 收口（"压死 XOR 耗尽"），累积结论是
  下一杠杆为结构连接 `D_EE` 或换衬底。M4（除法全局池）是**真正新**的杠杆；NULL M4 加强那个结论。

### 12.2 文献（Consult #1 + Consult #2）

**除法 / 归一化核心（机制）：** Chance FS, Abbott LF（Divisive inhibition in recurrent networks；除 recurrent、
不除 feedforward，§5 承重区分；也证明合适除法抑制可**稳定并加快** recurrent、保持近恒 gain）；Carandini M,
Heeger DJ（Normalization as a canonical neural computation；响应 = 驱动 / 池化活动）。

**局部兴奋 + 全局抑制 motif：** Wang X-J attractor / decision（common inhibitory pool → WTA）；Engel TA et al.；
Binas J et al.（Front. Comput. Neurosci. 2014）；Wang DL & Terman D（LEGION）；Lomp O et al.（Dynamic Field
Theory, Front. Neurorobot. 2016）。

**ISN / SSN 高增益稳定态：** Rubin DB, Van Hooser SD, Miller KD（SSN, Neuron 2015）；Holt AB, Miller KD,
Ahmadian Y（SSN gamma, PLOS Comput. Biol. 2024）。

**平衡网络临界 / Hopf：** Liang J et al.（Hopf in E-I balanced networks, Front. Syst. Neurosci. 2020；同步转变
附近 Hopf → limit cycle）；Meisel C et al.（Critical slowing down, PLOS Comput. Biol. 2015）；Ma Z et al.
（criticality as set point, Neuron 2019）。

**癫痫 —— 局部兴奋后跟全局抑制 / core-vs-penumbra（疾病侧锚）：** Curot J et al.（2023；人类 FR 后
200–800 ms 抑制 = `S_G`/`H_G` 时标锚）；Liou J-Y et al.（focal seizure model, 2020；local recurrent + global
feedback inhibition + usage-dependent exhaustion → territory expansion + widespread synchronization + rhythm
slowing）；Schevon CA et al.（Nat. Commun. 2012；**seizure core vs ictal penumbra**——核心超同步 firing、邻区
大场电位但低水平无结构 firing；快 feedforward inhibition = restraint —— 支持对"全场同步但无空间结构"的警惕）；
Proix T et al.（Epileptor field, Nat. Commun. 2018；发作既有慢 ictal wavefront 又有 faster coupled-oscillator /
SWD dynamics → 同步节律须放在空间招募 + 多时间尺度里解释，不单独作发作定义）。

---

## 附录 A —— Consult #1 核心论证（除法共享池）

- **不要重写局部 I→E 核，也不要把椭圆指数核换成 Gaussian。** 局部抑制仍是现有的
  $q_I(x,t)\cdot(K_{IE}^{\text{elliptic-exp}}*s_I)$；模型里的 Gaussian smoothing 只在**慢变量感知**（$K_q,K_K$）里，
  表示局部活动驱动的资源耗竭、不是突触连接。
- **Abbott 的关键是除法 recurrent 增益控制。** 原型率方程（只除 recurrent、保留 feedforward）：

$$
\tau_r\,\dot r_i=-r_i+I_i+\frac{\sum_j W_{ij}r_j}{R}
$$

  对应本模型 $I_E^{\text{rec}}\to \dfrac{I_E^{\text{rec}}}{1+\alpha_G S_G}$，外加至多小减法 $-\beta_G S_G$；承重是**除法**。
- **共享池写成 low-rank 核** $K_G(x_i,x_j)=v_G(x_i)\,u_G(x_j)$；第一版均匀（$u_G=v_G=1$，$A_G=\langle r_E\rangle_E$），
  可选 axis / soft-core 加权读出（让强核早招募池）。
- **第一版 `S_G` 只作用 E、不作用 I**（否则压 E 又经压 I 去抑制 E → 不可解读）。
- 消融 A（现有减法）、B（只减法共享池）、C（除法共享池）→ C 最可能产出中间有界发作样态。

## 附录 B —— Consult #2 核心论证（局部 `R_L` + 全局 `G`、相图、模态）

- **局部抑制是一个揉合的"等效抑制资源" `R_L(x,t)`**（= `q_I`）；发作态缺的不是更多局部细节，而是一个
  **活动依赖、延迟、共享 / 全局的抑制反馈 `G(t)`**。
- 发作样态需三个性质（runaway 都没有）：**有界**、**同步**、**可撤回**。
- **两个抑制环**：局部 `R_L` 决定病理模态是否逼近临界；全局延迟 `G` 决定那个失稳变 runaway 还是变有界、同步、
  可逆 attractor。`G` 必须滤波 / 延迟（$\tau_G\sim 100$–$800$ ms），可拆 $G_{\text{fast}}$（γ / 同步）+
  $G_{\text{slow}}$（refractory / offset）。
- **近临界不是"抑制弱"**：是强 recurrent 兴奋被强反馈抑制稳住（ISN/SSN 高增益边缘）。
- **模态解读**（遍2，§10.2/§10.3）：从有效 Jacobian 定义领头模态、按形状分类。有界同步振荡 $\approx$ Hopf /
  延迟反馈 limit cycle，**不是**活动爆炸。（rev3：默认领头模是 core、不是 axis/global。）
- **相图**（`R_L` × `g_G`）约 6 区；发作样态是一个**区域**、不是一个点——§8.1 是它的可操作实例。
- **判"发作样、非 runaway"的四个模型指标**：有界高活动 attractor；全局同步序参量

$$
S(t)=\Big|\tfrac{1}{N}\sum_x e^{i\phi_x(t)}\Big|
$$

  临界慢化（恢复时间↑、variance/autocorr↑）；轴-到-全局模态转移（**rev3 降级为 H2/H3 候选之一、非默认**）。
- 推荐版本：**B（除法归一化）+ C（延迟全局反馈）**——本 spec §5 + §4 就是 B+C。
- 完整参考并入 §12.2。
