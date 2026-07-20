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

**揭示了什么** — 两件事。(1) **空间响应强烈依赖慢状态**：固定的局部源头"踢一下"，安静间期响应就窝在
源头附近、几乎不铺开；快到失控前，同样一踢的响应沿源→汇轴向走廊铺开（走廊响应≈或>源头，三种子一致
≈0.2），且响应到达各处的时间跟距离基本成线性（拟合 R²≈0.9）——像沿轴逐步招募（只说"像招募"，没证明
连续行波）；远端汇核团 50 ms 内始终没被点亮。(2) **这张真网络有没有一个"线性空间算子"，是个"测量能不
能辨识"的问题，不是"存不存在"的问题**：第一版审计只在 seed1 上、用很细（每格强度只有踢的 1/12）的单
轨迹扰动量，量出来"到处都非线性"——但那是测量假象。改用**集成平均**（对每个状态平均多条独立的未来噪
声）**+ 把扰动强度按每格 RMS 匹配到那个"踢"的量级 + 只用平衡的低波数模式**重新审计后，**中等活动的
midpoint 状态在 3 个种子里有 2 个能干净辨识出算子**（差异 0.07–0.12，低于 15% 线），其最优输出模式**偏
向源→汇轴向**；seed3 的 baseline 也能辨识。安静 baseline 仍被放电离散性限制（种子间不稳，1/3），快到
失控的 pre_onset 三种子一致地"刚好过线一点点"（0.16–0.19，比 midpoint 更难线性化）。所以**"整张网络到处
没有线性算子"这个说法不成立**——修正审计说算子在中等活动区可辨识、且偏轴向；这跟旧"冻结速率场"近似有
干净算子的方向一致，只是真 spiking 网络的可辨识窗口是中段、两头（太静/太饱和）辨识不出。

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

**修正审计**（新 `audit` 子命令；平衡对称低波数 9 模式 + 每格 RMS 匹配到踢的量级 strength_frac×I_EE_scale
+ 对每个状态集成平均 8 条独立未来噪声，±共享每条未来=CRN；只有过线才 SVD）：

| state | seed1 | seed3 | seed4 | identifiable | σ̂₁(T30) / U₁ 轴向（可辨识处）|
|---|---|---|---|---|---|
| baseline | 0.429 | **0.081** | 0.217 | 1/3（量化受限、seed 不稳）| seed3: σ̂₁=4.06, u1_axis=−0.35 |
| midpoint | **0.122** | 0.249 | **0.074** | **2/3** | seed1 σ̂₁=87.5 u1_axis=+0.24；seed4 σ̂₁=8.86 u1_axis=+0.52 |
| pre_onset | 0.191 | 0.178 | 0.164 | 0/3（三种子一致刚过线）| — |

**结论（修正后）**：差异从第一版的 0.4–2.6 全面降到 0.07–0.43；**中等活动的 midpoint 有 2/3 种子干净辨识
出算子**，其最优输出模式偏源→汇轴向（u1_axis>0）；安静 baseline 被量化限制、seed 不稳（1/3）；pre_onset
三种子一致地"刚好过线一点点"（0.16–0.19，比 midpoint 更难线性化，是温和且可复现的非线性）。→ **"整网到处
无线性算子"不成立**；这是一个"测量可辨识性问题"，可辨识窗口在中段，两头（太静/太饱和）辨识不出。`corrected_
audit_summary.json`。（口径：这仍是低波数子空间上的经验有限时算子，不是精确 full-SNN 本征模。）

### 3.3 固定 kick 空间响应（同一源头踢，跨状态，三种子 1/3/4）

`fixed_kick_summary.json`。每格 = |Δ E-rate| 区域均值（Hz）：

| state | seed | response norm | source_core | axis_corridor | arrival fit |
|---|---|---|---|---|---|
| baseline | 1 | 1.83 | 1.163 | 0.122 | 不合格 |
| baseline | 3 | 0.49 | 0.000 | 0.034 | (合格但响应≈0，见下) |
| baseline | 4 | 0.31 | 0.000 | 0.000 | (合格但响应≈0，见下) |
| midpoint | 1 | 0.99 | 0.628 | 0.053 | 不合格 |
| midpoint | 3 | **23.94** | 13.401 | 2.385 | 合格 |
| midpoint | 4 | **0.00** | 0.000 | 0.000 | 不合格 |
| pre_onset | 1 | 2.35 | 0.132 | **0.178** | **合格 R²≈0.92 v≈0.06 u/ms** |
| pre_onset | 3 | 2.90 | 0.227 | **0.221** | **合格** |
| pre_onset | 4 | 2.42 | 0.048 | **0.234** | **合格** |

**主对比 baseline vs pre_onset（spec 主图口径）：**
- baseline：同一源头踢的响应窝在源头（seed1 source 1.16 ≫ corridor 0.12）或几乎量不到（seed3/4
  source=0，整网响应 0.3–0.5，接近放电离散性地板）。轴向走廊响应 0.122 / 0.034 / 0.000 → **近零、
  seed 波动大**，没有沿轴铺开。
- pre_onset：轴向走廊响应 0.178 / 0.221 / 0.234 → **三种子紧密一致（≈0.2）**，且 corridor ≈ 或 >
  source（比值 1.35 / 0.97 / 4.9，2/3 走廊超源头），arrival-vs-distance **三种子全合格**（seed1
  R²≈0.92）。**这是"同刺激、不同慢状态、空间响应从局部/近零转为沿轴一致铺开"的直接 SNN 证据（跨三种子
  可复现）。** 远端汇核团 remote_sink 在所有状态、50 ms 内始终为 0（没到远端）。

**两个诚实警示：**
1. **midpoint 的固定-kick 响应是过渡不稳态，seed 极不一致**（response norm 0.99 / 23.94 / 0.00；seed3
   踢引起一次大暴发 src=13.4，seed4 净效应 0）→ **固定-kick 主对比只用 baseline vs pre_onset，midpoint
   不进主图空间/幅度面板**。注意：这跟 §3.2 的"midpoint 是修正算子审计能辨识的窗口（2/3）"不矛盾——固定
   -kick 是一个强的单点踢（对过渡态敏感），修正审计是集成平均的小扰动线性响应（把过渡态的抖动平掉了）。
2. arrival 零/近零响应假性合格的 bug **已修复**（review 2026-07-20）：加了绝对响应地板 `arrival_min_peak_hz`
   + `fit_arrival_distance` 拒绝常数到达（零展布）。baseline seed3/4 现在正确判 ineligible；主图 Supplementary
   1d 只画有实响应的 pre_onset arrival。

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

- **一致的方向**：冻结-q 速率场（M3B）是线性化，天然有干净算子/本征模，其"非正规瞬态沿轴"结论指向源→汇
  轴向。修正审计里 midpoint 的**可辨识算子最优输出 U₁ 也偏轴向**（u1_axis +0.24/+0.52），且 fixed-kick 在
  pre_onset 沿轴铺开——两条独立证据都指向轴向。
- **不一致 / 需分层**：真 spiking 网络的**线性可辨识窗口在中段**（midpoint 2/3），两头辨识不出——安静
  baseline 被放电离散性限制（1/3、seed 不稳），近饱和 pre_onset 一致地温和非线性（0/3、0.16–0.19 刚过线）。
  速率场那种"全状态都有干净算子"在真网络里不成立；是**测量可辨识性 + 状态窗口**的差别，不是"SNN 没有算子"。
  差异来源 spike/reset/delay/noise/非线性（spec §9）。**不调参强行让二者一致**；以直接 SNN 为本体结果，
  速率场留作理论 closure。

## 5. 数值/随机性/分辨率/稳健性 + review 2026-07-20 修复

- **算子可辨识性**：第一版审计只 seed1、thin-input、单轨迹 → 假象（§3.2）；修正审计（集成 + RMS 匹配 +
  平衡低波数）后 midpoint 2/3 可辨识、σ̂₁/U₁ 已算（低波数子空间；不是精确本征模）。
- 随机性：common random numbers（±ε/no-probe 共享 checkpoint rng_state），tiny-net C3 通过 + full-net
  smoke idempotent；修正审计的集成对每状态平均 8 条独立未来噪声，±共享每条未来。
- 分辨率：12×12 读出，occ_min=45（无空格），basis 正交残差 2e-15。
- arrival：**bug 已修**（绝对响应地板 + 拒绝常数到达）；pre_onset seed1 R²≈0.92，baseline 现正确 ineligible。
- **within-window saturation / right-censoring bug 已修**：fork 窗 50 ms 装不下 100 ms 的 operational-
  runaway 判据（原判据永不触发 → censor/kick_runaway 恒 None 是假的）。改用 within-window 判据（120 Hz
  持续 ≥ saturation_dur_ms=20 ms，能在 50 ms 窗内触发），字段改名 `*_saturation` 诚实标注（不是 100 ms
  operational-runaway）。
- 种子稳健：fixed-kick 的 **pre_onset 轴向走廊响应三种子紧密一致（0.178/0.221/0.234≈0.2）+ arrival 合格 =
  主发现可复现**；baseline 近零、seed 波动大（都表示"无沿轴铺开"）；midpoint 固定-kick 过渡不稳（不进主
  对比）。算子可辨识性：midpoint 2/3、baseline 1/3、pre_onset 0/3（三种子一致刚过线）。

## 6. 结论口径（允许/禁止）

**允许**：同一 MZ spiking 骨架在不同慢状态下有限时空间易感性不同；快到失控前同一局部刺激响应更沿轴
铺开（走廊≈或>源头，三种子一致），沿轴到达时间随距离线性（compatible with 轴向招募）；**低波数经验算子
在中等活动（midpoint）可辨识、其最优输出偏轴向**；这是**测量可辨识性问题**（可辨识窗口在中段），不是
"SNN 没有算子"。

**禁止**：operational runoff = 临床发作起始；复现完整间期—发作—恢复循环；把 midpoint 的低波数 V₁/U₁ 说成
**精确 full-SNN 本征模**；说"三种子/整网到处都没有线性算子"（已被修正审计推翻）；把算子失败归因于"跨临界
非线性"（三状态都失败、无状态选择性——是量化噪声/输入过弱/单轨迹的测量问题）；σ̂₁>1 = 净放大；kymograph
证明连续行波；证明 Hopf/fold/Floquet；按结果换 state/seed/ε/basis/T 救结论。

## 7. 最大局限与下一步（不自主启动新机制）

- **最大局限**：算子只在低波数子空间 + 中段活动可辨识（midpoint 2/3），仍**不是精确 full-SNN 本征模**；
  baseline 量化受限、pre_onset 温和非线性。fixed-kick 承重"状态条件化空间易感性"这条主结论；轴向招募是
  "像"非证明。
- **P1 plateau + D-matched 对照已撤下结论**（review）：settle_ms 原来没用上（现已修，加 `settled` flag），
  且 D-matched 选到的中间 D（≈0.04）是过渡不稳区，量不出干净适应空间效应 → 登记为 seed-unstable/withdrawn，
  不进科学结论；已存 `controls_summary.json` 留作 exploratory。
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
