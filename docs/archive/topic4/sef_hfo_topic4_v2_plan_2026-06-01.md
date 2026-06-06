# Topic 4 v2：SEF-HFO 空间易激场模型计划

> **状态**：v0.2 plan lock draft，2026-06-01。用户确认：Topic 4 主模型路线从旧的 HR/FHN toy modeling 转为 **Spatial Excitability Field model for interictal HFO propagation (SEF-HFO)**；2026-06-01 review 后收紧为 **gain-closed + pulse-validated + control-disciplined** 版本。
> **替代范围**：替代 `docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md` 中 HR 主、FHN sensitivity 的 Phase 4 建模主线；保留 `docs/topic4_sef_itp_framework.md` 中已经锁定的真实数据验收合同、cohort tier、phantom-rank 修复纪律、clinical SOZ 不作为拟合标签的红线。
> **不替代范围**：不重写 Topic 1 的实证发现，不解释 HFO 80-250/500 Hz carrier 的细胞生物物理，不把 template source 直接等同 clinical SOZ。
> **2026-06-06 细化指针 + superseding note**：病理→中观→参数映射纪律（强制链 + 方向工作点上算不预设）+ 第一轮调参（连接性 E→E 核「定往哪传」+ E 阈值异质性 `Var(V_th,E)`「定哪里点着」，LIF/SNN 并行）见 `docs/superpowers/specs/2026-06-06-sef-hfo-pathology-parameter-mapping-design.md`。**SUPERSEDES**：本 plan 下文凡出现"降低异质性 → 更易激"这类**方向性**语言，一律以新 spec 的"方向必须在工作点上计算、不预设"为准（异质性经斜率/曲率→闭环稳定→有限扰动余量进入解释，可被 disconfirm）。承接 2026-06-04 Step-1 同质率场 NULL，走向**条件式**（too-frequent→异质核 / non-returning→先补 recovery·surround，由 `joint_A_failmode` 定），非"直接进异质核"。

> **2026-06-02 review amendment（two-stage screen 重定位；本块为当前 governing 口径）**
>
> 经 2026-06-02 review（advisor 复核 + 用户 ratify），SEF-HFO v0.2 从「强机制确认模型」**重定位为 two-stage control-disciplined exploratory mechanism screen**：
> - **Stage 1 = exploratory mechanism screen**：不预设「低异质性必然 → 近临界」。先在**跟模型无关的真实数据框定的 baseline operating-point family** 上系统筛选哪些机制组合能产生稳定、自限、可复现的间期传播组织；通过筛选的**最小机制冻结**后才进入 Stage 2。
> - **Stage 2 = held-out consistency validation**（原 §4 Step 6「预测验证」改名）：在 held-out 上做一致性印证，**不**声称前瞻验证（各向异性轴旋转在真实数据无对应物；率变-几何不变即已有 H4）；**Stage 1 筛选用过的真实特征不得再作 Stage 2 验证目标**（disjoint screen/validation targets），或在未参与筛选的被试上验证。
>
> **冻结的是纪律与判据逻辑，不是参数值**（用户口径：建模本身探索性强，参数值留作待筛选的 hypothesis scaffold，不现在写死）。governing 纪律补丁（review A–E + 用户五条，统一编号）：
>
> 1. **operating-point family + 自洽稳态 + 不准抢救**（收紧 §1.3 第1条 gain-closed / §2.2）：工作点取**自洽稳态**（稳态发放率自洽解 r0 = F_eff(W·r0 + I)）；可接受族范围用背景发放率 / E-I 比例 / 平时离阈值距离等**模型无关真实数据先框死、动异质性前锁定**。报「可接受族里多大**比例**的工作点上『降低异质性 → 更易激（`eta_lin` 下降 / finite-pulse 阈值下降）』成立」，**不报「存在一个点成立」**。某些工作点不成立时**如实报适用范围，禁止用移动均值阈值 / 外部输入 / 连接强度抢救机制**。E、I gain 同时动，净效果正负由计算决定。
> 2. **finite-pulse 为线性稳定性之后的必要步**（§3.1 / §4 Step 0 拆分）：linear dispersion map 只定位小扰动 mode 和候选窗，不证明有限幅自限事件；Step 0 拆 **0a linear dispersion map + 0b finite-pulse response map**（extinction / local bump / self-limited propagation / runaway）；**只有落在 self-limited propagation 区的参数进入后续 synthetic HFO 分析**。
> 3. **recovery 变量从 rate 层起 + 跨阶段同构**（收紧 §3.1 / §3.4）：rate field 即加**可开关的最小恢复 / 不应变量**保证「点着→传播→熄灭」闭环；**0b 同时跑「纯抑制」与「加恢复变量」两套响应面，按实测哪套稳定给出自限传播定**，tie-break = 自限区间更宽（更不挑参数）+ 事件时长/空间范围更贴数据；选定后 **rate 与 LIF 用同一套**（LIF 映射为 g_K / adaptation current）。**恢复时间常数锚到真实 HFO 群体事件持续时长，不自由调**。Stage 4 须回验 LIF 粗粒化的群体 I-O 曲线与恢复时间常数与 rate 层一致（「同构」要验不是声明）。
> 4. **承重判别指标 = 方向随连接轴转、不随电极杆转**（替换 §4 Step 2 的「是否出现 forward/reverse」主判据）：线性电极几何本身就能产生稳定顺序与正反反向模板（各向同性扩散从一根直杆两端成核即给出 1→n 与 n→1），故 **identity bias / held-out rank stability / inter-template anti-correlation 很可能与 isotropic+aligned-shaft 对照共享，不作承重判据**。承重判别指标具体化：系统转「连接各向异性轴」与「电极杆轴」，模板方向角对二者回归，**预注册「连接角系数≈1、杆角系数≈0、CI 排除反转情形」才算过**；且 **isotropic+aligned-shaft 对照必须过不了这一条**（它即该判据的坏数据回归测试）。
> 5. **复用边界**（§3.5 / Stage 4）：模型只生成虚拟信号 + 事件候选；排序 / 聚类 / 模板稳定性 / identity bias / endpoint 全部**调用现有真实数据 pipeline 代码（PR-2 / PR-2.5 / PR-6）不改一行**，不为 synthetic data 另写更易通过的分析器（CLAUDE.md §6.1）。
>
> 受本 amendment 收紧的 **§1.3 / §4 Step 2 / §4 Step 6** 节首已加回指；§2.2 / §3.1 / §3.4 / §3.5 的 prose 由本 amendment 直接 govern，原文保留作 audit trail。**下一步**：Step 0a/0b 具体实施方案（exploratory mechanism screen 定位，预注册纪律 1–5，参数值留 scaffold）。

> **2026-06-03 amendment（user 讨论敲定；与 `docs/topic4_sef_itp_framework.md` 2026-06-03 amendment 同步）**
>
> 1. **「局部自限」→「时间离散自终止」（空间可填满 SOZ 邻域网格）**：建模目标不是空间次网格波，而是 SOZ 小邻域网格内时间离散且自终止的群体 HFO 传播事件；要锚传播 lags（模板时间结构）+ 事件包络时长，**不**锚 HFO 80–250Hz 载波频率（载波仍 out of scope）。
> 2. **recovery 措辞收硬**（收紧上节纪律 3）：recovery 不是首选生物解释，**但必须作为并列机制分支进入 Step 0b**（**不**写成暂不做）；纯抑制自限 vs 加恢复变量自限 report-both / no-auto-select，去留由真实事件时长 / 空间范围 / 自限区宽度决定。亚临界噪声触发（"Brunel 支"）为离散事件的概念首选路径。
> 3. **促临界 ↔ 稳态回拉拮抗 = 看整个参数方案的组织视角**（描述性，非新验收）：旋钮归两极——促临界（低阈值异质性 / patch、强 E→E、增益↑、去抑制、Cl⁻ 致 E_GABA↑）vs 回拉（sAHP、自适应阈值、强/快抑制、阈值异质性↑）；同一拮抗跑两尺度：事件内（快：可激传播 vs recovery / 空间约束掐断）+ 跨时间（慢：促临界漂移 vs 稳态）。recovery = 回拉极快端，与慢端稳态同源。
> 4. **homogeneous / heterogeneous（不绝对化）**：heterogeneous patch + surround = 间期自限事件的自然结构（促临界核被周围回拉约束住）；homogeneous = 单相极限，可能代表全回拉 / 全促临界 / 控制条件，**不直接等同「发作态」**。
> 5. **发作桥接降格 = H5 / Phase 3 候选机制**，不入纯间期主验收：间期事件可能既是噪声涨落、又可能留净促临界痕迹；自限安全余量被压低时某次涨落更易越界成持续招募；逐事件累积 / 悬停临界 / 发作后重置须用 seizure 间隔 / 风险率数据判（无记忆 vs 渐增）。**只作后续仿真 + H5 数据检验候选机制，不作为 Topic 4 主模型已解释 clinical seizure onset 的结论。**
> 6. **coworker1 LIF（`Jlibrary/ei_snn_scaffold/`，Brunel-exact current-based LIF）= Step 4 前置参考 / LIF 引擎可行性证据，非模型闭环**；缺：亚临界离散事件、低异质性 patch、synthetic 走真实 pipeline、controls 必须过不了、rate↔LIF 同构。
>
> **Step 0 验收判定（2026-06-03）**：Step 0a/0b **机器里程碑 accepted**（模块 + 21 tests + gate + runner，smoke tier 过；`docs/archive/topic4/sef_itp_phase4_v2/step0_results_2026-06-02.md`）。但**解锁 Step 1 的 formal gate 未过**：scaffold 占位参数下 `fraction_with_window=0`、0 候选工作点、深度稳定 / 主模 k=0、有限脉冲全 extinction —— 按纪律 2「只有落在 self-limited propagation 区的参数进入后续分析」**无参数入选**，Step 1 无候选窗可加噪声 → **Step 1 保持锁定**。解锁前置：(a) 数据锚定 operating-point family + 单位（背景率 / E-I 比 / 离阈距 + Brunel Table-1 时间常数 + 实测事件时长定恢复 / 包络 + 实测 lags 定传播速度），`provenance`→`data_locked`；(b) data_locked re-run 0a/0b 确认候选窗使有限脉冲给出 **self-limited PROPAGATION（移动波前，非静态 / 非全局闪）+ 正安全余量**，recovery off/on 并列、dt/L 敏感性稳定、报族内通过比例；(c) 很可能先把 rate 层换 LIF 有色噪声 transfer + Brunel 时间结构使色散预测有限-k Hopf / 传播态（rate↔LIF 同构交叉验证）。

---

## 0. 一句话假设

间期 HFO 群体事件是病理组织附近一块“局部低异质性、各向异性连接、近临界但仍亚阈值”的空间 E-I 易激斑块，在噪声触发下产生的自限性瞬态传播事件；同一斑块在仿真中可被推入持续招募状态，但这只是 synthetic feasibility bridge，不是临床发作起始机制的主结论。

白话版：

> 间期 HFO 群体事件不是每次随机走一条新路，而是同一块病理组织附近存在一条固定的传播高速路。噪声偶尔点燃这条路，活动沿轴传播后自限熄灭；慢变量把系统推得更危险时，事件出现得更频繁。至于发作样持续招募，第一版只问“同一模型能不能在仿真里进入这种状态”，不把它写成真实发作机制。

这个模型要解释的是通道先后顺序、模板稳定性、正反方向、端点几何、事件率调制和发作邻近招募之间的关系。它不解释单个 HFO 载波为什么是 ripple / fast ripple。

---

## 1. 模型边界

### 1.1 声称

- 群体 HFO event envelope 的中观传播组织。
- 虚拟 / 真实 SEEG 通道激活顺序。
- rank template、forward / reverse 模板、split-half / blockwise 稳定性。
- identity bias、source / sink / endpoint 几何。
- 事件率随慢变量或发作邻近状态改变，但传播几何相对稳定。
- 间期自限瞬态和发作样招募之间的同一参数族 synthetic feasibility bridge。

### 1.2 不声称

- 不解释 HFO 80-250/500 Hz carrier 的细胞层振荡机制。
- 不声称 template source 等于 clinical SOZ。
- 不声称“E/I imbalance causes epilepsy”这种过大结论。
- 不用 clinical SOZ 反向拟合模型参数。
- 不声称本模型解释 clinical seizure onset；发作样招募只作为仿真可行性，不作为 Topic 4 主结论。

### 1.3 v0.2 硬合同

> **2026-06-02 amendment 收紧**：本节 5 条在 two-stage screen 重定位下收紧，见顶部 2026-06-02 amendment —— 第1条 gain-closed 加 operating-point family + 自洽稳态 +「报比例不报存在 + 不准抢救」；第3条 control-disciplined 的承重判别指标具体化为「方向随连接轴转、不随电极杆转」带阈值回归判据（isotropic+aligned-shaft 对照必须过不了）；并入 recovery 从 rate 层起跨阶段同构、Stage 2 disjoint screen/validation targets。

这版计划的核心纪律是：

1. **Gain-closed**：`sigma_phi(x)` 不能口头等于近临界。它必须先进入有效 population transfer function，改变局部 gain，再进入线性化矩阵。只有实际计算显示当前工作点下 `eta_lin(x)` 下降，才能说低异质性 patch 更接近临界。
2. **Pulse-validated**：线性稳定性只负责给出小扰动地图和候选空间 mode。有限幅、自限传播事件必须用 finite-pulse nonlinear simulation 验证。
3. **Control-disciplined**：identity bias 和 `k=2` 不能单独作为机制成功证据。Full model 必须高于几何采样 negative controls；controls 不能通过主指标。
4. **K descriptive**：`k=2` 是描述性输出，不是 primary success criterion。主指标是 held-out rank stability、split-half / odd-even stability、inter-template anti-correlation、finite-pulse self-limited propagation、negative controls fail。
5. **Slow variable minimal**：第一版只用抽象慢变量 `q(t)` 调制 `eta(x,t)`。`z_I`、`g_K`、`E_GABA` 只能在 `q(t)` 版通过后作为 mechanism decomposition。

---

## 2. 核心机制

### 2.1 固定传播轴

E→E 连接核不是圆形，而是椭圆形。沿长轴方向连接更远、更容易传播。这条长轴是传播高速路；事件从一端成核表现为 forward，从另一端成核表现为 reverse。正反模板不是两套网络，而是同一条轴的两个方向。

重要口径：forward / reverse 不写成强 ping-pong 机制。更稳妥的解释是 **共享传播轴 + 随机成核端点**。这与真实数据中“几何相关，但短时间尺度模板依赖不强”的结果兼容。

### 2.2 局部低异质性斑块

第一版只在 threshold variance 上实现病理斑块：

```text
phi_i = phi_bar(x_i) + sigma_phi(x_i) * epsilon_i
epsilon_i ~ N(0, 1)
```

病灶 patch 内 `sigma_phi(x)` 降低。生物解释是：病灶附近同类神经元更相似，因此更容易被同一次 fluctuation 集体推过传播阈值。

但这不允许直接写成：

```text
sigma_phi(x) down => eta(x) down
```

严格路径必须是：

```text
sigma_phi(x)
  -> F_eff(h; x)
  -> G(x)
  -> lambda(k; x)
  -> eta_lin(x)
```

也就是说，阈值分布先进入 population input-output curve：

```text
F_a_eff(h; x) = integral f_a(h - phi) p(phi; phi_bar_a(x), sigma_phi,a(x)) dphi
```

再在当前工作点定义局部 gain：

```text
G_a(x) = dF_a_eff(h; x) / dh | h=h0(x)
```

最后把 `G_E(x), G_I(x)` 放入局部线性化矩阵，实际计算 `eta_lin(x)`。只有在选定工作点下计算得到 `d eta_lin / d sigma_phi > 0`，才能说降低异质性让该 patch 更接近临界。若 E 群体和 I 群体的 gain 同时改变，方向必须由计算决定，不准口头预设。

第一版不要同时改 `J_EE`、`J_IE`、`E_L`、adaptation、外部输入。否则模型会变成多参数调出来的拟合机器。

### 2.3 近临界但亚阈值

定义局部离失稳边界的距离：

```text
eta_lin(x) = - max_k Re(lambda(k; G_E(x), G_I(x), W_hat(k), tau_E, tau_I))
```

- `eta_lin(x) >> 0`：小扰动稳定，扰动很快消失。
- `eta_lin(x) approx 0+`：小扰动仍稳定但接近边界，是候选可激区。
- `eta_lin(x) < 0`：小扰动失稳，可能进入持续 pattern。

纪律：`eta_lin(x)` 只是线性地图，不证明有限幅事件存在。真正的间期工作窗必须再通过 finite-pulse response 证明：有限幅输入能触发可检测事件，且离 runaway 还有正安全边界。

### 2.4 噪声成核

背景输入和内源性波动提供点火源。系统靠近 `eta approx 0+` 时，小扰动可以触发一次传播事件，但抑制和恢复变量使事件自限终止。

### 2.5 慢变量只调率，不调路

第一版只保留一个抽象慢变量：

```text
eta(x,t) = eta_0(x) - a_q q(t)
```

`q(t)` 的唯一验收目标是：改变事件发生概率和事件率，但不显著改变传播几何。`z_I(t)`、`g_K(t)`、`E_GABA` 漂移和 Epileptor-like slow permittivity variable 全部放到 extension；只有抽象 `q(t)` 版通过后，才拆成具体生物变量。

---

## 3. 最小方程层级

### 3.1 Step 0-3 主体：二维 E-I rate field

先用 rate / neural field 找工作窗，不直接跑 SNN：

```text
tau_E * dr_E/dt = -r_E + F_E(W_EE * r_E - W_EI * r_I + I_E)
tau_I * dr_I/dt = -r_I + F_I(W_IE * r_E - W_II * r_I + I_I)
```

对均匀稳态线性化：

```text
r_b(x,t) = r_b^0 + delta r_b * exp(lambda t + i k x)
```

求 `max_k Re(lambda(k))`，先画 linear dispersion map。这个图只给候选工作区和候选空间 mode，不证明自限传播事件。

### 3.2 各向异性 E→E 核

```text
P_ij^EE = p0 * exp(
    -u_ij^2 / (2 * ell_parallel^2)
    -v_ij^2 / (2 * ell_perp^2)
)
ell_parallel > ell_perp
rho = ell_perp / ell_parallel < 1
```

`rho` 越小，传播轴越明确，forward / reverse 越干净。

### 3.3 抑制核

抑制核更宽，并保留全局刹车：

```text
W_I = g_I_bar * ((1 - gamma) * G(sigma_I) + gamma * U)
sigma_I > ell_parallel
```

作用是允许局部传播，但防止活动无限扩散。

### 3.4 Step 4 sensitivity：LIF E-I SNN

Rate field 通过验收后，再搬到 conductance-based LIF：

```text
C_m dV_i/dt =
  -g_L(V_i - E_L,i)
  -g_i^E(t)(V_i - E_E)
  -z_i(t) g_i^I(t)(V_i - E_GABA,i)
  -g_K,i(t)(V_i - E_K)
  +I_i_ext(t)
  +sigma_i xi_i(t)
```

发放可先用 rate 版，再用 Poisson / Bernoulli spiking 版。SNN 的角色是验证 rate 层机制是否能在 spiking 层保持，不是第一版主战场。

### 3.5 虚拟 SEEG

用突触电流 proxy 做 LFP / SEEG：

```text
LFP_e(t) = sum_i K_e(x_i) * (|I_i^E(t)| + |I_i^I(t)|)
A_e(t) = |Hilbert(Bandpass(LFP_e(t)))|
```

群体 HFO 事件定义为多个通道 envelope 在短时间窗内超过阈值。所有 synthetic data 必须走真实 PR pipeline，而不是人工挑图。

---

## 4. Step 0-6 执行路线

### Step 0a：线性稳定性分析

目标：找到小扰动稳定性地图和最易激空间 mode。

验收标准：

- `max_k Re(lambda(k)) << 0` 时小扰动快速消失。
- `max_k Re(lambda(k*)) <= 0` 且接近 0 时，标记为候选可激区；不直接宣称会产生事件。
- `max_k Re(lambda(k*)) > 0` 时小扰动失稳，标记为 sustained-pattern risk。
- 报告 `k*` 和可选 `omega*`，作为后续传播方向 / 空间尺度预测。

输出：

- phase diagram。
- `max_k Re(lambda(k))` heatmap。
- `k*` heatmap。
- interictal working point 和 ictal transition boundary。

### Step 0b：finite-pulse response map

目标：验证有限幅扰动的非线性响应，区分“可激但自限”和“直接招募失控”。

预注册 pulse family：

- `x0`：patch center、axis-end A、axis-end B、off-axis control 四类位置。
- `r`：固定 2-3 档空间半径，不事后连续调。
- `T`：固定 2-3 档持续时间，不事后连续调。
- `A`：在每个 `(x0, r, T)` cell 内单调扫描，估计阈值。

刺激：

```text
I_pulse(x,t) = A * 1{|x - x0| < r} * 1{0 < t < T}
```

响应分类：

| 类别 | 判据 | 含义 |
|---|---|---|
| Extinction | 脉冲后快速衰减 | 不可激 |
| Local bump | 局部激活但不传播 | 低可激 |
| Self-limited propagation | 传播一段距离后熄灭 | 间期候选区 |
| Runaway / recruitment | 持续扩散或全局招募 | ictal-like 失稳 |

定义：

```text
A_event(x0, r, T) = inf A: 出现可检测群体事件
A_runaway(x0, r, T) = inf A: 进入持续招募
```

真正的间期候选工作窗必须满足：

```text
A_event < infinity
A_runaway - A_event > positive safety margin
```

验收标准：

- 至少一个预注册 pulse family cell 落入 self-limited propagation。
- Runaway threshold 与 event threshold 之间有正安全边界。
- 报告完整 response surface；不能只挑一个好看的 pulse。

### Step 1：最小 2D rate field

目标：在 Step 0a/0b 锁定的候选工作窗里，验证噪声能产生离散、自限传播事件。

验收标准：

- 无噪声时没有持续事件。
- 加噪声后出现离散群体事件。
- 事件分类落在 Step 0b 的 self-limited propagation 区，不持续扩散。
- 事件率可调到真实数据同一量级。
- 事件持续时间和空间范围合理。

### Step 2：加入各向异性和几何 negative controls

目标：复现主数据结构：两个方向、稳定模板、高 identity bias。

> **2026-06-02 amendment（承重判别指标替换）**：下列主验收标准中 identity bias / held-out rank stability / inter-template anti-correlation **很可能与 isotropic+aligned-shaft 对照共享**（线性电极杆两端成核即给出正反反向模板），**不作承重判据**。承重判别指标改为「模板方向随连接各向异性轴转、随电极杆旋转不变」带阈值回归判据，且 isotropic+aligned-shaft 对照必须过不了这一条。详见顶部 2026-06-02 amendment 第4条。

主验收标准：

- held-out rank stability 高于 controls。
- split-half / odd-even template stability 高于 controls。
- inter-template rank anti-correlation 高于 controls。
- finite-pulse response 属于 self-limited propagation。
- full model 相对 negative controls 的主指标超过 controls 的 95% 区间，或达到预注册效应量。

必须包含的 controls：

| 控制 | 目的 |
|---|---|
| isotropic connection + aligned shaft | 检查仅电极排列是否足以产生高 identity bias |
| anisotropic connection + random shaft | 检查 anisotropy 是否被错误采样成稳定 rank |
| rotated shaft | 检查 rank 是否随采样几何旋转 |
| jittered contacts | 检查模板稳定性是否对微小电极扰动鲁棒 |
| shuffled contact identity | 检查 identity bias 是否只是统计 artifact |
| multiple shaft orientations | 检查真实 SEEG shaft geometry 是否主导结果 |

描述性输出：

- KMeans 后是否主导为 `k=2`。
- identity bias fraction 是否接近真实数据范围，例如 0.80-0.92。
- 加入第二条弱轴或第二个 patch 后，是否可出现少数 `k=4/6`。

`k=2` 和 raw identity bias 都不是 primary success criterion。

### Step 3：加入局部低异质性 patch

目标：把传播轴和病理位置连接起来。

验收标准：

- 事件更容易从低异质性 patch 附近成核。
- source / endpoint 与 patch 有统计关系。
- 改变 patch 位置，source density 随之移动。
- 改变各向异性轴，forward / reverse 轴随之旋转。
- 不强制 source 等于 clinical SOZ。

### Step 4：搬到 LIF E-I SNN

目标：证明 rate 层机制在 spiking 层仍成立。

验收标准：

- 单神经元 firing 不需要高度同步。
- 群体层面出现事件。
- 虚拟 SEEG 上能提取 onset rank。
- 用真实 PR-2 / PR-2.5 pipeline 跑 synthetic data。
- synthetic data 的 `k` 分布、within-template Kendall tau、split-half stability、identity bias 与真实数据同方向。

### Step 5：加入抽象慢变量

目标：验证“率变，几何不变”。

验收标准：

- `q(t)` 变化时，事件率升高。
- rank template 不大幅改变。
- KMeans cluster identity 保持稳定。
- forward / reverse 不需要强时间配对。
- 当 `eta(x,t) < 0` 时，仿真可进入 ictal-like recruitment；该结果只作为 synthetic feasibility bridge，不作为 clinical seizure onset claim。
- `z_I(t)`、`g_K(t)` 不进入第一版主分析；只能作为 `q(t)` 版通过后的分解实验。

### Step 6：回到真实数据做预测验证（→ 2026-06-02 改名 held-out consistency validation）

> **2026-06-02 amendment（改名 + 收紧）**：本步改名 **held-out consistency validation**，定位为一致性印证、**不**声称前瞻验证（各向异性轴旋转在真实数据无对应物，率变-几何不变即已有 H4）。新增硬约束：**Stage 1 筛选用过的真实特征不得再作本步验证目标**（disjoint screen/validation targets），或在未参与筛选的被试上验证。详见顶部 2026-06-02 amendment。

目标：避免模型过度灵活，只做预测验证，不做反向拟合。

验收标准：

- held-out endpoint 稳定。
- source / sink 比 middle channel 更接近某些病理标记。
- forward / reverse 共享几何轴。
- event-level mark dependence 不强也可以接受。
- endpoint 不等同 clinical SOZ 不算模型失败；真正失败是 endpoint 在 held-out 内部不稳定，或模型不能复现 identity bias / template stability。

---

## 5. 最重要的可证伪预测

1. **近临界窗预测**：只有在 `eta_lin approx 0+` 且 finite-pulse map 显示 `A_event < infinity`、`A_runaway - A_event` 有正安全边界的区域，模型才出现离散、自限、稳定传播事件。若远离临界或无安全边界也能产生同样模板，近临界假设不必要。
2. **各向异性轴预测**：改变 E→E 各向异性轴，forward / reverse 模板方向必须随之旋转。若模板方向不受轴控制，固定传播轴假设失败。
3. **低异质性 source 预测**：只有在 `sigma_phi -> F_eff -> G -> eta_lin` 实际计算显示该 patch 更接近候选可激窗后，事件成核点才应向该 patch 聚集。若降低 threshold variance 只增加全局同步，或 gain / stability 方向不支持接近临界，低异质性不能作为核心机制。
4. **identity bias 预测**：full model 的 held-out rank stability、split-half / odd-even stability 和 inter-template anti-correlation 必须高于几何采样 controls；raw identity bias 只作描述性输出。若 controls 也通过，说明模型只是在重现平凡采样几何。
5. **率变几何不变预测**：慢变量改变 event rate，但不应显著改变 rank geometry。若慢变量导致模板完全重排，和已有数据不一致。
6. **间期-发作桥接预测**：当抽象慢变量把 `eta(x,t)` 从 `0+` 推到 `<0`，自限传播事件可在仿真中转为持续 recruitment。若同一参数族不能进入 ictal-like recruitment，synthetic feasibility bridge 不成立；即使成立，也不直接声称解释 clinical seizure onset。

---

## 6. 文献 framing：具体机制多样，中观动力学收敛

这批文献不应该被写成“某篇论文证明了 SEF-HFO”。更稳妥、也更有用的 framing 是：

> 癫痫的具体细胞机制可能很多：离子稳态、Na/K pump、氯离子 / GABA、抑制失效、突触短时可塑性、胶质缓冲、结构连接和局部网络异质性都可能参与。但在我们关心的中观尺度上，它们可能殊途同归：让局部网络从“稳定但可激”进入“更接近临界、更难恢复、更容易被有限扰动点燃、更容易发生事件聚簇或空间招募”的状态。

### 6.1 这些文献支撑什么

1. **间期到发作可以被看成慢状态推动的动力学转变**。Epileptor / Jirsa 2014 用抽象分岔和慢变量统一 seizure onset / offset；Wendling 2005 的人类 TLE neural mass work 则提醒我们，转变不是简单“兴奋升高、抑制降低”，而是 pyramidal cells 与 interneuron populations 的相互作用随时间重组。
2. **不同细胞机制可以实现同一类慢变量或易激性变化**。Na/K concentration、pump、glia、diffusion、inhibition 和 glial-potassium dynamics 这些机制都能改变网络稳定性和发作易感性；Wei 2014 进一步把 spikes、seizures、spreading depression 放进同一 ionic-gradient 动力系统视角。
3. **空间传播需要场和连接结构，而不是单节点解释**。Proix 2018 的 Epileptor neural field、Naze 2015 的 coupled-network work、Wang 2016/2017 的 focal onset pattern modeling 都支持“局部状态 + 连接结构 + 周边组织可招募性”共同决定传播形态。
4. **线性稳定性不够，有限扰动响应必须单独测**。Chang 2018 的 network resilience 和 Maturana 2020 / Lepeu 2024 的 critical-transition framing 都支持一个保守说法：接近临界时，恢复速度和扰动后的回弹能力会改变；因此 Step 0b finite-pulse response map 不是装饰，而是主合同。
5. **HFO 是合理观测对象，但不能被偷换成“微型发作”**。Zijlmans 2010/2011 和 Weiss 2013 支持 HFO 与 epileptogenic tissue / seizure core readout 有关；但它们更偏 biomarker / localization 层，不能直接证明 interictal HFO propagation 等同 ictal spread。

### 6.2 这些文献启发 SEF-HFO 的具体位置

SEF-HFO 的贡献不是指定“钾、钠、氯、泵、胶质或抑制”哪一个才是真慢变量，而是把这些机制在中观层的共同后果抽象出来：局部易激性、恢复能力、有限幅扰动阈值和空间招募能力发生变化。我们建模的对象是 HFO 群体事件的通道先后顺序和空间传播组织；发作样持续招募只作为同一参数族的仿真可行性桥接。

因此文中安全写法是：

> We do not assume a single cellular mechanism linking interictal and ictal activity. Instead, we use a dynamical abstraction: diverse biophysical mechanisms may converge onto a shared change in local excitability, resilience, and finite-amplitude recruitment. Under this view, interictal group-HFO events are modeled as isolated excursions of a pathological excitable field, whereas ictal-like recruitment corresponds to slow-state-gated clustering or spatial expansion of similar elementary events.

### 6.3 引用分层

- **抽象动力学 / 慢变量**：Jirsa et al. 2014; Chizhov et al. 2018; Wendling et al. 2005.
- **离子与胶质等慢机制实现**：Cressman et al. 2009; Wei et al. 2014; Ho and Truccolo 2016.
- **空间场与传播**：Proix et al. 2018; Wang et al. 2016/2017; Naze et al. 2015; **Bachschmid-Romano, Hatsopoulos & Brunel 2026**（bioRxiv 2026.03.18.712701；与本模型方法**最接近的先例**：空间结构化 E-I spiking 网络、自洽 2×2 色散 `λ(k)`、外驱相图、各向异性 E→E 选传播轴、`|I^E|+|I^I|` LFP 代理。**关键区别**：其事件为近全局 Turing–Hopf 行波，SEF-HFO 承接其机制与方法，但目标是局部自限的间期 HFO 事件——0b 的 recovery off/on 正对应"亚临界近 Hopf"与"恢复变量局部脉冲"两条机制）.
- **恢复能力 / 有限扰动 / 临界转变**：Chang et al. 2018; Maturana et al. 2020; Lepeu et al. 2024.
- **HFO 观测对象与边界**：Zijlmans et al. 2010/2011; Weiss et al. 2013.

红线：这些文献只能支撑“机制抽象层的合理性”和“间期到发作的动力学启发”，不能写成“SEF-HFO 已解释 clinical seizure onset”，也不能写成“interictal HFO 就是微型发作”。

---

## 7. 论文 / PPT 图结构

1. **Figure 1：模型概念图**。低异质性 patch、各向异性传播轴、宽抑制背景、噪声成核、forward / reverse、虚拟 SEEG rank template。
2. **Figure 2：linear + pulse maps**。左侧为 `max_k Re(lambda(k))` 和 `k*`，右侧为 finite-pulse response map（extinction / local bump / self-limited propagation / runaway）。
3. **Figure 3：rate model event + controls**。噪声触发瞬态波、forward / reverse、rank template；并列展示 isotropic / random shaft / rotated shaft / jittered contacts controls。
4. **Figure 4：SNN synthetic SEEG**。raster、虚拟 SEEG traces、envelope、group event detection、rank template clustering。
5. **Figure 5：synthetic vs real cohort**。held-out rank stability、within-template Kendall tau、split-half / odd-even stability、inter-template anti-correlation、identity bias 描述值、forward / reverse fraction、endpoint stability。
6. **Figure 6：slow variable feasibility bridge**。抽象慢变量变化、事件率升高、template geometry 稳定、越阈后进入 ictal-like wavefront；标题和图注明确这是 synthetic feasibility。

---

## 8. Topic 4 v2 取代关系

- `docs/topic4_sef_itp_framework.md` 保留为 Topic 4 formal entry，但顶部状态和 §6.5 起应指向本 v2 plan。
- `docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md` 降级为历史路线 / sensitivity，不再是主模型路线。
- `docs/archive/topic4/sef_itp_phase4_v1/stage1_results_2026-05-28.md` 与 `stage1b_results_2026-05-28.md` 保留为 HR route 的历史探索证据。
- 新结果目录建议：`results/topic4_sef_hfo/`，子目录按阶段命名：`linear_stability/`, `rate_field/`, `anisotropy/`, `low_heterogeneity_patch/`, `lif_snn/`, `slow_variable_bridge/`, `synthetic_vs_real/`。

---

## 9. 立即下一步

最小闭环不是继续加机制，而是：

1. 先写 Step 0a/0b 实现计划：effective gain + linear dispersion map + finite-pulse response map。
2. 用 rate field 在 self-limited propagation 区跑出稳定 forward / reverse rank geometry，并同时跑 negative controls。
3. 确认 synthetic data 能走真实模板 pipeline，主指标高于 controls。
4. 再决定是否进入 LIF SNN。

如果 Step 0a/0b 找不到“可触发但不失控”的安全窗，或者 Step 2 controls 也能通过主指标，SNN 不启动。
