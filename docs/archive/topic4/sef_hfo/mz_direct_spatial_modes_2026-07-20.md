# Topic 4 — MZ 整网 spiking 直接空间响应（direct current-based SNN spatial modes）

Archive · 2026-07-20 · branch `codex/topic4-mz-direct-spatial-modes` · base `6c878ae`
Spec: `docs/superpowers/specs/2026-07-19-topic4-mz-direct-spatial-modes-design.md`
Tier = 模型本体机制分析（model-side mechanism），不是发作验证。每个 phenotype 是检测标签。

---

## 摘要（第一性原理）

**测了什么** — 我们拿 E1146 那张约 4 万个神经元、会自发放电的电流型 spiking 网络，在它从平静滑向
自发失控的三个时刻各看一眼：很安静的间期（约 0.1 Hz）、活动中等的中段（约 10 Hz）、快到失控前
100 ms（约 85 Hz，已很接近失控）。每个时刻把慢变量冻住 50 ms（只留快系统），给一个很小的电流扰
动，看整张网络在空间上怎么响应。

**怎么测的** — 两件事。第一，问这张真网络能不能用一个"干净的线性空间算子"描述：给两个大小的扰动
（一个 ε、一个 ε/2），如果响应是线性的，两次算出来的空间响应形状应该几乎一样；实测两次相差 40%–
260%，远超我们定的 15% 线，所以量不出干净的线性算子。第二，固定给源头核团打一个稍强的局部"踢一
下"（同样位置/宽度/强度/时刻），看这一踢的响应停在源头不动还是沿源→汇轴向铺开；同一踢在三个时刻各
做一遍、三条种子重复。

**揭示了什么** — 两件事，分层说清楚。(1) **承重结果——空间响应强烈依赖慢状态（可复现）**：固定的局部
源头"踢一下"，安静间期响应窝在源头附近、几乎不铺开；快到失控前，同样一踢的响应沿源→汇轴向**走廊**铺
开——而且走廊里**远端（靠近汇的那半段）**的响应比同距离的**离轴对照区**强约 2.3 倍（seed1/seed4），说明
不是四面八方乱铺，是真往汇的方向走；响应到达各处的时间随距离基本成线性（seed1 拟合 R²≈0.9，正斜率、多
阈值方向一致）——像沿轴逐步招募（只说"像招募"，没证明连续行波）；远端汇核团 50 ms 内始终没被点亮。
(2) **诊断结果——"这张真网络有没有一个线性空间算子"是个测量能不能辨识的问题，而且目前只能勉强、零星地
辨识，不能说有一个可复现的算子**：第一版审计（只 seed1、每格强度只有踢的 1/12、单条未来噪声）量出"到处
都非线性（0.4–2.6）"——那是测量假象。改用**集成平均 + 每格 RMS 匹配到踢的量级 + 平衡低波数 + 把 16 条独
立未来拆成两组各 8 条互相验证 + 检查有没有 fork 饱和**重审后，差异全面降到 0.06–0.36（确认第一版是假
象）；**但严格判据（全 16 条过线 且 两组各 8 条都过线 且 没有饱和）下，9 个"种子×时刻"里只有 2 个稳健
辨识**（seed3 的 baseline、seed4 的 midpoint），而且**这两个模式互不相同**（重叠≈0.05、朝向相反：seed4
midpoint 沿轴向但不落在走廊上，seed3 baseline 反而横着）。好几个点的"全 16 条平均"能压到 15% 线以下，但
拆成两组独立 8 条就各自越线——说明它们不稳健。所以：**"整网到处没有线性算子"不成立（是测量假象），但也
远没到"有一个可复现的、跨种子一致的轴向算子模式"**；现有证据只支持"在个别孤立的种子×时刻点上勉强辨识出
的模式各自有沿轴/横轴的取向倾向"，不是一个模式。真正可复现的是那个固定"踢"的状态条件化空间响应。

（内部归档代号：nonlinear_response_only；empirical finite-time SNN response operator；zA_q75_tz5000
[use_z, I_th_EI=95.199, tau_z=5000]；freeze z/m 50 ms；144-dim real Fourier basis；common random
numbers；fixed-kick frac=0.01×I_EE_scale；axis_corridor/source ratio；arrival-vs-distance fit；
locked runoff 9293.6/9499.3/9757.9 ms；D≈0.087 runoff corridor。）

---

## 1. 设计与本体（读取，不重估）

- 底物：`run_m4_phaseplane.build_substrate(seed)`。E1146 narrow / template_source / twoend_equal，
  L=20 mm，密度 100/mm²（NE≈32000 / NI≈8000），E→E AR=2 沿源→汇，g=3.6，ν_ext=0.6，dt=0.1 ms，
  两个低阈值核 r=1.5 mm @ 17.5±1.0 mV，背景阈值 18 mV，自发（无原生 kick）。
- 慢变量：仅 E 细胞 `z`（抑制效能）+ `m`（适应），`src/snn_engine/mz_slow_vars.py`。q_I/g_K/S_G/
  shunt/STD/conductance 全关。
- 主候选（锁定）：`zA_q75_tz5000`（use_z, I_th_EI=95.19851312666987, tau_z=5000），seeds 1/3/4。
  锁定 operational-runoff onset（READ，从 onset-dynamics config，不重估）：9293.6 / 9499.3 / 9757.9 ms。
- 三个主状态：baseline=1000 ms；midpoint=0.5×runoff；pre_onset=runoff−100 ms。主图只直接比较
  baseline 与 pre_onset。

## 2. 方法（direct-SNN 扰动）

- **checkpoint fork**：复用 `src.topic4_mz_onset_dynamics.run_loop`/`LoopState`（已捕获 V、refractory、
  AMPA/GABA 突触态与电流、delay rings、OU、rng_state、慢变量对象）。每条种子原生轨迹只 replay 一次
  （分段 + resume，持久化 checkpoint）。所有扰动从 checkpoint fork；主分析在 fork 后冻结 z/m 50 ms
  （隔离快系统）。
- **扰动载体**：`MZSpatialProbe(MZOnsetProbe)`，off-by-default 的逐 E-神经元加性电流 schedule（作用于
  E only，时长 1.0 ms，可正可负，幅度以 I_EE_scale=272.755 的比例表示）。电流加在 `apply_currents`
  返回的 I_net 上，引擎在两次 RNG 抽样之后消费它 → 不改抽样顺序 → common random numbers 天然成立
  （contract test C1/C3/C4，`tests/test_topic4_mz_direct_spatial_modes.py`）。
- **读出**：12×12 粗网格（复用 `topic4_state_conditioned_susceptibility` 的 cell 分配，但**不**用它的
  冻结-q 算子）。每格 E-rate = 该格 E 神经元发放数 / 该格 E 数 / T；空格标记；总 spike 质量守恒审计。
- **empirical operator**：完整 144-dim 实正交 2-D Fourier basis（Q^T Q = I，spec 要求全空间）；每个 basis
  pattern 做 ±ε fork，`K_T[:,j] = [Y_T(+εp)−Y_T(−εp)]/(2ε)`，`M = K P^T`，SVD → σ̂₁ / V₁ / U₁。
  **仅当线性资格通过时才算 SVD**。
- **线性审计**：对每个 ladder ε 比较 K(ε) 与 K(ε/2)，归一化差异 ≤15% 且不饱和才合格，取最大合格 ε；
  没有合格 → `nonlinear_response_only`（不扩大 ladder），只做 fixed-kick。
- **固定 kick**：源头 Gaussian 正电流（RMS=frac×I_EE_scale，frac=0.01），同一踢跨状态；读出 5/15/30/
  50 ms 局部图、轴向 kymograph、arrival-vs-distance（<4 点 fail-closed）、region/累积比。
- **并行**：COW fork Pool（连通性每种子建一次约 13 GB，只读共享；worker 只传 (j,sign)）。

## 3. 结果

### 3.1 状态活动度（no-probe 冻结 fork，seed1）

| state | 时刻 | 整网平均 E-rate | 峰值 | 50 ms 内 spike 数 | 冻结后自发失控？ |
|---|---|---|---|---|---|
| baseline | 1000 ms | 0.10 Hz | 1.2 Hz | 157 | 否 |
| midpoint | 4647 ms | 10.4 Hz | 31 Hz | 16 569 | 否 |
| pre_onset | 9194 ms | 84.8 Hz | 105 Hz | 135 693 | 否（冻结把它稳在临界前） |

冻结 z/m 后三个状态的 no-probe 控制在 50 ms 内都不自发失控（未触发 right-censoring），所以三个状态
都可测。

### 3.2 算子可辨识性：第一版审计（假象）→ 修正审计（可辨识窗口在中段）

**第一版审计（seed1 only，thin-input，单轨迹）** — 归一化差异 ‖K(ε)−K(ε/2)‖/‖K(ε/2)‖ over
ladder [0.001…0.01]×I_EE_scale：baseline nan/nan/0.661/0.813、midpoint 0.503/0.535/2.606/0.444、
pre_onset 1.432/1.004/0.864/1.012，全部远超 15% → 当时写成 `nonlinear_response_only`。

**这个"到处非线性"是测量假象**（2026-07-20 review 拦下，逐条核对属实）：(a) 只在 seed1 跑了审计，
seed3/4 复用了全局锁，不能代表三种子；(b) 每个状态只用一条随机未来、没有集成平均，30 ms baseline 响应
被整数 spike 的量化噪声淹没；(c) Fourier basis 每列 per-grid RMS 只有 1/12，同一个 ε 下算子输入比那个
固定"踢"弱约 12 倍；(d) "前 16 个模式"其实含单方向到 Nyquist 的高频列，不是平衡低波数。

**修正审计**（`audit` 子命令；平衡对称低波数 9 模式 + 每格 RMS 匹配到踢的量级 strength_frac×I_EE_scale +
每状态 **N=16 条独立未来** + 把 16 拆成两组各 8 条互验 + 检查 fork 饱和；**严格判据 = 全 16 过线 且 两半各
8 都过线 且 无饱和**，只有过判据才 SVD）。full = 全 16 条差异，repA/B = 两组独立 8 条各自的差异：

| state | seed | full | repA/B（8+8） | sat | identifiable |
|---|---|---|---|---|---|
| baseline | 1 | 0.362 | 0.429/0.449 | 0 | False |
| baseline | 3 | **0.059** | **0.081/0.091** | 0 | **True** |
| baseline | 4 | 0.156 | 0.217/0.165 | 0 | False |
| midpoint | 1 | 0.125 | 0.122/**0.159** | 0 | False（half 越线）|
| midpoint | 3 | 0.246 | 0.249/0.246 | 0 | False |
| midpoint | 4 | **0.062** | **0.074/0.080** | 0 | **True** |
| pre_onset | 1 | 0.135 | 0.191/0.176 | 0 | False（half 越线）|
| pre_onset | 3 | (N=16 收尾中) | — | 0 | False（N=8=0.178；N=16 与 seed1/4 同型必 False，落地后回填）|
| pre_onset | 4 | 0.125 | 0.164/0.177 | 0 | False（half 越线）|

可辨识 2 点的模式：seed3 baseline σ̂₁=4.15 **u1_axis=−0.32（横轴）** corridor_frac=0.35；seed4 midpoint
σ̂₁=8.57 **u1_axis=+0.52（沿轴）** corridor_frac=0.00（沿轴但不落走廊）。**两模式互不相同**：sign-invariant
重叠 |cos|≈**0.05**、朝向相反、σ̂₁ 差约 2 倍。

**结论（N=16 严格判据）**：(1) 差异从第一版 0.4–2.6 全面降到 0.06–0.36 → **第一版"整网到处非线性"是测量
假象，推翻**。(2) **但严格判据下 9 个种子×时刻只 2 个稳健辨识**（seed3 baseline、seed4 midpoint）；**split-
half 是判别关键**——好几个点全 16 平均能压过线（seed1 midpoint full 0.125、pre_onset seed1/4 full 0.13/0.13），
但拆两组各 8 就越线（0.16–0.19），说明不稳健；**全程 sat=0**。midpoint 只 1/3 稳健（不是 2/3）；只有 3 个离散
时刻，不足以定义连续"窗口"。(3) **两个可辨识模式互不相同、朝向相反 → 没有跨种子一致的轴向算子模式**；只能说
个别孤立点上勉强辨识出的模式各自有取向倾向（orientation tendency）。`corrected_audit_summary.json` + 每
realization K（`corr_Kr_*`）。（口径：低波数子空间上的经验有限时算子，**不是精确 full-SNN 本征模**。）

### 3.3 固定 kick 空间响应（同一源头踢，跨状态，三种子 1/3/4）

`fixed_kick_summary.json`（新代码重跑，arrival/saturation/distal 字段已更新）。每格 = |Δ E-rate|
区域均值（Hz）；arrival 判据 = 响应地板 + 正斜率 + 有限 R² + R²≥0.5 + 多阈值方向一致；distal/off =
走廊远端（靠汇那半）响应 ÷ 同距离离轴对照带：

| state | seed | norm | source | corridor | distal/off-axis | arrival |
|---|---|---|---|---|---|---|
| baseline | 1 | 1.83 | 1.163 | 0.122 | — | 不合格 |
| baseline | 3 | 0.49 | 0.000 | 0.034 | **0.67** | 合格(R²0.96,kymo7.8Hz) |
| baseline | 4 | 0.31 | 0.000 | 0.000 | — | 不合格(响应<地板) |
| midpoint | 1 | 0.99 | 0.628 | 0.053 | — | 不合格 |
| midpoint | 3 | **23.94** | 13.401 | 2.385 | — | 合格 |
| midpoint | 4 | **0.00** | 0.000 | 0.000 | — | 不合格(退化,已正确拒绝) |
| pre_onset | 1 | 2.35 | 0.132 | 0.178 | **2.29** | **合格 R²≈0.92 正斜率** |
| pre_onset | 3 | 2.90 | 0.227 | 0.221 | None(离轴带空) | **合格** |
| pre_onset | 4 | 2.42 | 0.048 | 0.234 | **2.34** | **合格** |

**主对比 baseline vs pre_onset（spec 主图口径）：**
- baseline：响应窝在源头（seed1 source 1.16 ≫ corridor 0.12）或近放电离散地板（seed3/4 norm 0.3–0.5）。
  轴向走廊响应近零、seed 波动大，没有往汇的方向铺开。
- pre_onset：轴向走廊响应三种子一致≈0.2，**且远端走廊/离轴对照 seed1/seed4 ≈2.3**（seed3 因几何离轴带为空
  无法算）——**说明响应是真往汇的方向走，不是四面八方乱铺**；arrival-vs-distance 三种子合格（seed1 R²≈0.92，
  正斜率、多阈值方向一致）。远端汇核团 50 ms 内始终为 0（没真到汇）。**这是"同刺激、不同慢状态、空间响应
  从局部转为往汇方向定向铺开"的直接 SNN 证据（跨三种子可复现）。**

**两个诚实警示：**
1. **midpoint 的固定-kick 响应是过渡不稳态，seed 极不一致**（norm 0.99/23.94/0.00；seed3 踢诱发一次大暴发
   src=13.4，seed4 净效应 0）→ 固定-kick 主对比只用 baseline vs pre_onset，midpoint 不进主图空间/幅度面板。
   （这跟 §3.2 无冲突：那里 midpoint 是**集成平均小扰动**下唯一稳健辨识出算子的那 1 个种子[seed4]，把过渡态
   抖动平掉了；这里是**单点强踢**对过渡态敏感。）
2. arrival 假性合格 bug **已按完整合同修复**（review 2026-07-20）：响应地板 `arrival_min_peak_hz`=2 + 拒绝
   常数到达 + **正斜率 + 有限 R² + R²≥0.5 + 多阈值方向一致**。修复后：seed4 midpoint/baseline 退化响应正确判
   ineligible；seed3 baseline 是**真实的小而干净的拟合**（kymo 7.8 Hz、R²0.96、正斜率），合格是对的——但它
   的 distal/off=0.67<1，**有轴向到达梯度却不偏向走廊**，所以"arrival 合格"≠"往汇方向定向"，两个指标要分开
   看。主图 Supplementary 1d 只画有强响应且 distal/off>2 的 pre_onset。

### 3.4 控制：z+m plateau vs D-matched z-only（P1）

在**同一个抑制耗竭水平 D** 下，比较"有适应（z+m plateau）"和"无适应但 D 相同（z-only）"两个状态对同一
源头踢的空间响应，看适应（m）是不是在 D 之外还改变空间响应。选点只用 D + 静息（population rate 低）+
时间，不看空间响应（spec §1）。`controls_summary.json`。

三种子（每种子 z+m 与 z-only 都 D 精确对齐）：

| seed | 匹配 D | z+m plateau norm (src) | z-only norm (src) | z-only/z+m norm |
|---|---|---|---|---|
| 1 | 0.0448 | 0.15 (0.045) | 0.90 (0.494) | 5.9×（z+m 更弱） |
| 3 | 0.0444 | **504.2 (175.8)** | 0.27 (0.000) | 0.0×（z+m 反而暴涨） |
| 4 | 0.0344 | 0.00 (0.000) | 9.80 (1.858) | ∞（z+m 死、z-only 强） |

**结论：seed 极不稳定，inconclusive，不成 cohort 主张。** seed1 看起来"适应把响应压下去"，但 seed3 相反
（z+m 反而被同一踢点燃成一次大暴发 norm=504，no-probe 控制稳定=censor resolved+kick_runaway=None，即这是
**踢诱发的大瞬态**，不是自发失控），seed4 又是 z+m 几乎无响应。三种子方向不一致。**根因**：D-matched 选到的
中间耗竭水平（D≈0.034–0.045）正好落在过渡不稳区（跟 §3.3 的 midpoint 一样 seed-variable）——这个 D 上同一
踢的效应对 seed 极敏感（有的种子点燃大暴发、有的压死、有的无响应）。所以"在匹配 D 下比较有/无适应的空间
响应"这条对照，在这组 D 上被过渡态不稳定性淹没，**量不出干净的适应效应**。这是诚实的 inconclusive/seed-
unstable，不是"适应抑制响应"（那只是 seed1 单点，未复现）。

（口径：这条对照是次级机制探针；三种子不一致 → 只登记为"过渡态 seed-unstable，未见可复现的适应空间效应"，
不写成 cohort 结论。arrival"合格"在近零/暴发响应上都不可解读。）

## 4. 与旧冻结-q 速率场结果的一致/不一致

- **唯一可复现的轴向证据是 fixed-kick，不是算子模式**：冻结-q 速率场是线性化，天然有干净算子/本征模，其
  "非正规瞬态沿轴"结论指向源→汇轴向。真 spiking 网络这边，**唯一跨三种子可复现的轴向信号是 fixed-kick**
  （pre_onset 远端走廊/离轴 seed1/seed4≈2.3），方向上与速率场结论一致——但这是那个"踢"的响应，不是一个线性
  算子的 U₁。
- **不能用算子模式去印证轴向**：修正审计里稳健辨识出的 2 个模式（seed3 baseline、seed4 midpoint）**互不
  相同、朝向相反**（重叠≈0.05；seed4 沿轴但不落走廊，seed3 反而横轴），所以**算子模式不构成"跨种子一致的
  轴向传播模式"**，不能拿来当第二条轴向证据。
- **本质差别**：速率场"全状态都有干净算子"在真 spiking 网络不成立——不是"SNN 没有算子"，而是**在这组测量
  设置下算子只能零星、勉强辨识，远没到可复现**（差异来源 spike/reset/delay/noise/非线性，spec §9）。**不
  调参强行让二者一致**；以直接 SNN 为本体结果，速率场留作理论 closure。

## 5. 数值/随机性/分辨率/稳健性 + review 修复（round-1/2）

- **算子可辨识性（N=16 严格判据）**：第一版审计只 seed1、thin-input、单轨迹 → "到处非线性 0.4–2.6" 是假象；
  修正审计（集成 + RMS 匹配 + 平衡低波数）把差异全面降到 0.06–0.36（确认假象）。**但严格判据 = 全 16 条过线
  且 两组独立 8 条都过线 且 无 fork 饱和 → 9 个种子×时刻里只 2 个稳健辨识**（seed3 baseline、seed4 midpoint）。
  **split-half 是判别关键**：好几个点全 16 条平均能压过线，但拆两组各 8 条就越线（如 seed1 midpoint full=0.125
  但 repB=0.159；pre_onset seed1/4 full=0.135/0.125 但半批 0.16–0.19）——所以它们不稳健。**全程 sat=0/576**
  （RMS 匹配强度没点燃冻结 fork），可辨识性不被饱和污染。每 realization 的 K 已存（`corr_Kr_*`），可复算。
- 随机性：common random numbers（±/no-probe 共享 checkpoint rng_state）+ 修正审计每状态 16 条独立未来、±共享
  每条未来；tiny C3 + full-net smoke idempotent。
- 分辨率：12×12 读出，occ_min=45（无空格），basis 正交残差 2e-15。低波数 = 平衡对称（1-D 频率 {0,1} 双轴 →
  9 个 2-D 模式），不是"前 N 列含 Nyquist"。
- **arrival bug 完整合同修复**：响应地板 + 拒绝常数到达 + 正斜率 + 有限 R² + R²≥0.5 + 多阈值方向一致。
- **within-window saturation / right-censoring bug 修复**：50 ms fork 装不下 100 ms 判据（原判据永不触发 →
  censor/*_runaway 恒 None 是假的）；改 within-window（120 Hz ≥ 20 ms，能触发）+ 字段改名 `*_saturation`。
- 种子稳健：**fixed-kick 主发现（pre_onset 往汇方向定向铺开、走廊≈0.2、distal/off≈2.3、arrival 合格）跨三
  种子可复现**；算子稳健辨识仅 2/9 孤立点、模式互不相同 = **非可复现**。

## 6. 结论口径（允许/禁止）

**允许**：同一 MZ spiking 骨架不同慢状态下有限时空间易感性不同；快到失控前同一局部踢的响应从局部转为**往汇
方向定向铺开**（远端走廊/离轴 seed1/4≈2.3，三种子走廊≈0.2 一致），到达时间随距离线性（compatible with 轴向
招募）；修正审计**推翻了"整网到处非线性"（是测量假象）**；在个别孤立的种子×时刻点上勉强辨识出的模式各自有
沿轴/横轴的**取向倾向**（orientation tendency）。

**禁止**：operational runoff = 临床发作起始；复现完整间期—发作—恢复；把稀疏辨识的低波数 V₁/U₁ 说成**精确
full-SNN 本征模**、或说成"跨种子一致的轴向传播模式/复用了同一条 kick 传播路径"（两个可辨识模式互不相同、
朝向相反）；说"整网到处没线性算子"（假象，已推翻）；说"midpoint 是可辨识窗口"（只有 3 个离散时刻、且
midpoint 只 1/3 稳健，不足以定义连续窗口）；把 pre_onset 说成"温和且可复现的非线性"（只能说三点在本次 N=16
估计里都没过 15% self-consistency 门）；σ̂₁>1 = 净放大；kymograph 证明行波；Hopf/fold/Floquet；按结果换
state/seed/ε/basis/T 救结论。

## 7. 最大局限与下一步（不自主启动新机制）

- **最大局限**：低波数经验算子只在 2/9 孤立种子×时刻点稳健辨识、且那两个模式互不相同 = **没有可复现的线性
  算子、没有一致的轴向模式**；仍不是精确 full-SNN 本征模。**fixed-kick 承重全部可复现结论**（状态条件化、往汇
  方向定向）；轴向招募是"像"非证明。
- **P1 plateau + D-matched 对照已撤下结论**：settle_ms 原没用上（已修 + `settled` flag），D-matched 中间 D
  （≈0.04）是过渡不稳区 → seed-unstable/withdrawn，不进科学结论；`controls_summary.json` 留作 exploratory。
- 下一步候选（**待用户定，不自主**）：(a) 先把统计/复现坐实（已做 N=4/8/16 + 两组独立 8 + split-half + 饱和
  检查），再考虑是否值得扩到全 144 维——直接扩维只放大计算量和 spike-count 噪声，不解决复现问题；(b) fixed-kick
  往汇方向定向可加空间置换 null；(c) native-dynamic 次级核验；(d) settled plateau + 避开过渡 D 重做适应对照。
- 下一步候选（**待用户定，不自主启动新机制**）：(a) 把可辨识算子从低波数扩到全 144 维、扫更多 T 看 σ̂₁(T)；
  (b) fixed-kick 沿轴招募加 within-window 时间分辨 + 空间置换 null；(c) native-dynamic（不冻结）次级核验；
  (d) 用 settled plateau + 避开过渡 D 重做适应对照。

## 8. 产物

科学根：`results/topic4_sef_hfo/mz_direct_spatial_modes/`（STATUS.md, provenance.json,
checkpoint_manifest.json, linearity_audit.json, fixed_kick_summary.json, empirical_operator_summary.json,
probe_scan_summary.json, numerical_audit.json, per_seed/, figures/）。
Paper-ready 候选：`results/paper-ready-figure/fig5_mz_direct_snn_spatial_modes_candidate/figures/`
（Supplementary 1 = fixed-kick 空间响应；Supplementary 2 = 可辨识性 + 轴向招募诊断）。
**未覆盖**旧速率场 `fig5_mz_spatial_dynamics_supplementary/`；未改 FIGURE_INDEX / main_figure_plan。
