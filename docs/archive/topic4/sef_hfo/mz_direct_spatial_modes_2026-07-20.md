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

**揭示了什么** — 在这张网络本体上，靠很小的分布式扰动量不出干净的线性空间算子：安静时响应太弱、被
放电的离散性（一个个整数 spike）淹没，活跃时又太非线性（快饱和）。这跟我们之前用"冻结速率场"近似
算出来的干净线性算子/本征模**不一样**——真 spiking 网络在临界点附近的响应是非线性的。但固定的局部
"踢一下"看得很清楚：安静的间期，同一踢的响应就窝在源头附近、几乎不铺开；快到失控前，同样一踢的响
应沿源→汇的轴向走廊铺开（走廊上的响应比源头还强），而且响应到达各处的时间跟距离基本成线性关系（拟
合很好）——像是沿轴向逐步招募，但我们只说"像招募"，没证明是连续行波。远端汇核团在 50 ms 内始终没被
点亮。

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

### 3.2 线性审计 → operator 不可辨识（lock seed = 1）

15% 判定线；归一化差异 ‖K(ε)−K(ε/2)‖/‖K(ε/2)‖ over ladder [0.001, 0.0025, 0.005, 0.01]×I_EE_scale：

| state | ε=0.001 | 0.0025 | 0.005 | 0.01 | 判定 |
|---|---|---|---|---|---|
| baseline | nan（无响应） | nan | 0.661 | 0.813 | 不合格 |
| midpoint | 0.503 | 0.535 | 2.606 | 0.444 | 不合格 |
| pre_onset | 1.432 | 1.004 | 0.864 | 1.012 | 不合格 |

三个状态所有幅度都远超 15% → **全局 `nonlinear_response_only`**，不计算 operator SVD，不扩大 ladder
（spec §2.3 合规 fallback）。这是有效完成结果（spec §12：operator 不满足线性资格是有效完成态）。
安静 baseline 是被放电离散性淹没；活跃 midpoint/pre_onset 是真非线性（近饱和）。

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
1. **midpoint 是过渡不稳态，seed 极不一致**（response norm 0.99 / 23.94 / 0.00；seed3 的踢引起一次
   大暴发 src=13.4 但未持续失控，seed4 的踢净效应为 0）。spec 明确 midpoint 只用于模态轨迹；本轮无
   operator → 无模态轨迹 → midpoint 只进 §3.2 线性诊断（证明 operator 到处不可辨识），**不进主对比、
   不进主图空间/幅度面板**。
2. baseline 的 seed3/4 arrival 标为"合格"是**在近零响应上拟合噪声的假性合格**（arrival 阈值=10%×max|kymo|，
   max 很小的时候噪声也能跨 ≥4 位置）→ **baseline arrival 不可解读**；只在有实响应的 pre_onset 解读
   arrival。主图 Supplementary 2c 只画 pre_onset arrival。

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

- **不一致**：冻结-q 速率场（M3B）是一个线性化，天然有干净的线性算子、本征模、σ₁(T)、exp(JT) 响应；
  真 spiking 网络在注册幅度下**量不出**干净线性算子（近临界非线性 + 放电离散性）。以直接 SNN 为模型本体
  结果，速率场保留为理论 closure；差异来源于 spike/reset/delay/noise/非线性（spec §9）。**不调参强行
  让二者一致。**
- **定性一致的方向**：旧冻结-q 分析的"非正规瞬态沿轴"结论（骨架各向异性给方向）与这里 fixed-kick 在
  pre_onset 的沿轴铺开方向一致——都指向源→汇轴向走廊。但直接 SNN 说明这种沿轴响应在真网络里是非线性
  放大出来的，不是一个线性算子的 U₁。

## 5. 数值/随机性/分辨率/稳健性

- 线性：见 §3.2，全部不合格 → operator 降级（预注册资格失败，非"结果不好看"）。
- 随机性：common random numbers（±ε/no-probe 共享 checkpoint rng_state），tiny-net contract test
  C3 通过；full-net smoke 复现 idempotent。
- 分辨率：12×12 读出，occ_min=45（无空格），basis 正交残差见 numerical_audit.json。
- arrival：pre_onset **三种子全合格**（≥4 轴向位置，seed1 R²≈0.92）；baseline seed3/4 的"合格"是
  近零响应上拟合噪声，不可解读（§3.3 警示 2）。
- 种子稳健：三种子（1/3/4）全部完成、全部 `nonlinear_response_only`（口径一致）。**pre_onset 轴向走廊
  响应三种子紧密一致（0.178/0.221/0.234≈0.2）+ arrival 全合格 = 主发现可复现。** baseline 近零、
  seed 波动大（但都表示"无沿轴铺开"）。midpoint 是过渡不稳态，seed 极不一致（§3.3 警示 1），不进主对比。
  线性不可辨识（operator 缺失）三种子一致（lock seed=1 审计代表；seed3/4 复用 lock，不重审）。

## 6. 结论口径（允许/禁止）

**允许**：同一 MZ spiking 骨架在不同慢状态下有限时空间易感性不同；快到失控前同一局部刺激响应更沿轴
铺开（走廊 > 源头）；直接 SNN 与冻结-q 速率场不一致（真网络非线性，无干净线性算子）；沿轴到达时间随
距离线性（compatible with 轴向招募）。

**禁止**：operational runoff = 临床发作起始；复现完整间期—发作—恢复循环；V₁/U₁ 是精确 full-SNN
本征模（本轮根本没算 operator）；σ̂₁>1 = 净放大；kymograph 证明连续行波；证明 Hopf/fold/Floquet；
按结果换 state/seed/ε/basis/T 救结论。

## 7. 最大局限与下一步（不自主启动新机制）

- 最大局限：empirical operator 在注册幅度下不可辨识——线性算子这条路在真 spiking 网络本体上（这组
  幅度 + 1 ms 脉冲 + 50 ms 窗 + 冻结）走不通。fixed-kick 承重全部结论。轴向招募是"像"，非证明。
- P1 plateau + D-matched 控制**已做三种子**：inconclusive/seed-unstable（§3.4），中间耗竭水平（D≈0.04）
  是过渡不稳区，量不出干净适应空间效应。这本身印证了 §3.3 的 midpoint 不稳定——中间 D 状态对同一踢的响应
  对 seed 极敏感。
- 下一步候选（**待用户定，不自主启动新机制**）：(a) 若要 operator，需不同可辨识化设计（更强/更久探针、
  更粗读出、或选真正线性的工作点），超出本轮注册合同；(b) fixed-kick 的沿轴招募可加 within-window 时间
  分辨 + 空间置换 null 做更强判读；(c) native-dynamic（不冻结）次级核验；(d) 若要看适应的空间效应，得避开
  过渡不稳的中间 D，或换更稳的匹配量（不是 D）。

## 8. 产物

科学根：`results/topic4_sef_hfo/mz_direct_spatial_modes/`（STATUS.md, provenance.json,
checkpoint_manifest.json, linearity_audit.json, fixed_kick_summary.json, empirical_operator_summary.json,
probe_scan_summary.json, numerical_audit.json, per_seed/, figures/）。
Paper-ready 候选：`results/paper-ready-figure/fig5_mz_direct_snn_spatial_modes_candidate/figures/`
（Supplementary 1 = fixed-kick 空间响应；Supplementary 2 = 可辨识性 + 轴向招募诊断）。
**未覆盖**旧速率场 `fig5_mz_spatial_dynamics_supplementary/`；未改 FIGURE_INDEX / main_figure_plan。
