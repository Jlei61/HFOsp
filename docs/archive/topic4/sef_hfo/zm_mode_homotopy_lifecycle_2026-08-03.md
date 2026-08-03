# Z/M 状态选择性模态与 conductance homotopy：阶段验收

**日期：** 2026-08-03（§10 为 2026-08-04 续记）  
**分支：** `codex/topic4-m4-snn-native-exit`  
**科学 verdict：** `NO_DURABLE_CREDIBLE_ICTAL_CARRIER`

> §10 是同一 verdict 下的后续机制轮次，结论不变但把失效机制定位得更具体：局部持续慢兴奋 conductance 能稳定地消除 burst 间深间隙并保持 12 s 有界不失控，但它是**用消灭弛豫振荡换来的**——持续段是静止的高秩空间图样（末段放电起伏 ≤0.046、轴向 DMD 主模态 0 Hz 实模态），而不是 §7 要求的 non-tonic spatial orbit。

## 1. 一句话判断

本阶段已经在正确的 per-neuron Z/M 空间 SNN 上证明：延长 I→E 相位差可以打开一部分空间自由度，局部 H 可以填平 burst 间隙，Z 门控 conductance 可以在不重写高-z 间期路径的前提下介入；但三者组合后的状态序列仍是

\[
\text{fragmented burst train}
\rightarrow \text{短暂空间分层过渡}
\rightarrow \text{common-mode tonic plateau}
\rightarrow \text{runaway},
\]

而不是有界、空间组织化、可由 M 内生退出并回到间期的 ictal lifecycle。

## 2. 上一阶段 bounded-negative 的冻结结论

上一阶段的快抑制、原始加性 M 和全局 E 阈值刺激结论可以验收：

- 快抑制参数改变 `relaxation burst train` 与 `spreading plateau` 的分支身份，但没有创造 recovery basin；
- 原始 M 可以降低活动密度或改变分支表型，但配对的退出后 z 回弹只有约 0.06--0.09，远弱于修复前误读的数值；
- 有限刺激可在脉冲期近乎完全压制 E 群并产生约 192 ms E-silence，但 6/6 均无持久退出、无 returning interictal event；
- 因此这条 scalar suppression 路线停止，不再扩大 `tau_D/d*/g_M/tau_M/stimulation` 网格。

## 3. 轴向模态与 phase-lag 结果

### 3.1 empirical mode

锁定 fast anchor 的 virtual-SEEG gain 约 25.9 dB、PC1 约 0.892。局部 DMD 没有找到稳定的 phase-staggered complex mode；主病理方向是实的 0-Hz fixed spatial mode。因此原 burst train 的主要语义仍是固定空间图样的幅度开关，而不是 Hopf/traveling orbit。

### 3.2 I→E delay panel

将 I→E delay 从 1× 提高到 3×，PC1 可降至 0.851、有效秩升至 1.69，说明空间自由度确实被打开。但所有臂仍有：

- energy occupancy 0.20--0.27；
- post-onset deep-gap fraction 0.67--0.71；
- 无 offset；
- 无 returning event/distribution。

所以 phase lag 是一个空间方向的正信号，但没有产生持续高能 ictal carrier。machine verdict 为 `NO_CREDIBLE_ICTAL_CARRIER`。

## 4. baseline-preserving conductance homotopy

完整 conductance replacement 在 Phase D 已因 baseline zero-spike dominance 失败。本阶段没有重复该实验，而是实现：

\[
\tau_m\dot V_i=(1-\lambda_i)F_i^{native}+\lambda_iF_i^{cond},
\qquad
\lambda_i=\Lambda(z_i),
\]

其中高 z 细胞逐位走原 current-based Z/M 膜路径，只有 z 进入耗竭窗才平滑切换到 conductance 向量场。最终 engagement window 为 `z: 0.52→0.48`；这是根据第一轮 core-lambda 峰值仅 0.16 做的一次传感器 engagement 修正，不是 carrier-outcome 参数网格。

无 H 时，即使 core lambda 峰值约 0.47，结果仍是 occupancy 约 0.21、deep gap 约 0.78--0.80 的 burst train。conductance homotopy 本身提供负向恢复/分流方向，不是 carrier 正反馈。

## 5. H–M balance 的承重结果

固定 `rho_H=2`、`tau_H_up/down=250/1500 ms`、I→E delay 3×、weak-global GABA `gamma=1/6`，只改变 M 对 H 的关闭强度：

| arm | gain dB | occupancy | deep gap | PC1 | effective rank | 结论 |
|---|---:|---:|---:|---:|---:|---|
| no H | 21.2 | 0.208 | 0.775 | 0.950 | 1.26 | fragmented burst train |
| H + M30 | 33.6 | 0.339 | 0.700 | 0.953 | 1.25 | M 过早切碎 carrier |
| H + M45, 2.5 s | 34.0 | 0.486 | 0.150 | 0.940 | 1.31 | 近门过渡，未过 occupancy 0.50 |
| H + M60 | 36.2 | 0.715 | 0.000 | 0.991 | 1.06 | common-mode tonic plateau |
| H, M gate off | 41.2 | 0.799 | 0.000 | 0.985 | 1.09 | high-rate tonic plateau |

M45 的短臂一度看起来最接近联合门，但 12 s 轨迹证明它不是稳定 carrier：短臂是长臂的逐位前缀，长臂在约 1.8 s 后进入约 400 Hz plateau，并于 5.407 s 触发 runaway early-stop。长臂指标为：

- median virtual-SEEG gain 38.94 dB；
- occupancy 0.717；
- deep gap 0.038；
- PC1 0.990；effective rank 1.07；
- core mean 307.5 Hz；all-E mean 120.9 Hz；
- offset none；return none。

因此不能把短臂 M45 的近门数值升级为 ictal carrier candidate。它是从 burst train 通向 common-mode runaway 的过渡。

## 6. 当前能写与不能写

### 能写

- 在固定各向异性 E→E scaffold 上，I→E phase lag 能提高空间有效秩，但不足以消除 burst 间深间隙；
- 状态选择性 H 能把离散 burst train 转成连续高能状态；
- 原生 M gate 能控制 H–plateau 分支平衡；
- 当前组合中，连续能量与空间非共模不能同时稳定存在；
- M45 长跑进入 common-mode runaway，没有 native offset 或 interictal return。

### 不能写

- 不能写已存在可控 ictal lifecycle；
- 不能把 M45 的 2.5 s 近门过渡称为 seizure carrier；
- 不能把高 gain/高 occupancy 的 M60/M-off plateau 称为发作；
- 不能声称 H、conductance 或局部抑制普遍失败；
- 不能进入 stimulation/control 优化，因为可控制的 ictal basin 尚不存在。

## 7. 对下一步的动力学约束

本阶段已经排除了“再调一个 scalar slow gain 就能闭环”的主要变体。下一个机制必须直接改变 fast subsystem 的模态结构，使至少两个空间自由度在持续高能段共同存在，而不是继续沿单一 fixed/common mode 放大。

最小要求是：

1. frozen slow state 下先存在 bounded non-tonic spatial orbit；
2. 该 orbit 不能依赖 detector、定时开关或 E→E 拓扑重写；
3. M 的作用必须让该 orbit 消失/失稳，而不是只降低幅度或把它切碎；
4. 只有上述 fast carrier 通过后，才继续 native offset、recovery 和 control。

本线下一节点应收缩到真正的局部 E/I cluster 或 traveling/re-entrant fast instability，而不是再扫 H/M/延迟/conductance 标量。

## 8. 交付物

- phase-lag verdict：`results/topic4_sef_hfo/zm_mode_lifecycle/i2e_phase_lag_lifecycle_summary.json`
- homotopy verdict：`results/topic4_sef_hfo/zm_mode_lifecycle/conductance_homotopy_summary.json`
- 核心图：`results/topic4_sef_hfo/zm_mode_lifecycle/figures/conductance_homotopy_lifecycle.png`
- phase-lag 图：`results/topic4_sef_hfo/zm_mode_lifecycle/figures/i2e_phase_lag_lifecycle.png`
- 分析器：`scripts/analyze_topic4_zm_phase_lag_lifecycle.py`、`scripts/analyze_topic4_zm_conductance_homotopy.py`

所有新机制均 default-off；E→E topology/kernel/AR/direction/delay 未修改。

## 9. 定向 fast-modal follow-up（同日）

按“不再做标量网格”的约束，又做了三个单点因果测试，并与原 common-H 臂统一重算：

| arm | gain dB | occupancy | deep gap | PC1 | rank | core Hz | rho80 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| common H | 41.2 | 0.799 | 0.000 | 0.985 | 1.09 | 396.5 | 0.465 | 连续但 common-mode 高率平台 |
| I→E source-delay dispersion | 37.9 | 0.649 | 0.000 | 0.983 | 1.10 | 304.1 | 0.503 | 相位异质性未产生空间 orbit |
| local-fast + broad-slow GABA | 31.1 | 0.335 | 0.750 | 0.984 | 1.10 | 100.7 | 0.000 | 降率但切成同步 burst train |
| common-subtracted contrast H | 41.1 | 0.803 | 0.000 | 0.984 | 1.09 | 415.4 | 0.966 | hotspot saturation，不是模态选择 |

四臂在同一严格 gate 下均失败，verdict 为
`NO_CREDIBLE_CARRIER_IN_TARGETED_FAST_MODAL_PANEL`。这组结果把缺口进一步收缩：

- 单纯分散同一 I 群的输出延迟不够；
- 同一批 I spikes 驱动的快/慢双滤波只改变刹车强度和 burst 周期，不产生独立抑制相位；
- 去掉 H 的均匀分量也不等于放大一个 dynamical eigenmode，它会把正反馈集中成局部饱和。

因此下一步需要**两个真正独立的抑制性状态群**，而不是同一 I 群的两个滤波器。最小 SNN-native 候选是：在不改 E→E 的条件下，将现有 I 群确定性拆成 fast/local（PV-like）与 delayed/broad/slow（SOM-like）两群，使后者具有独立的 E→I 招募延迟和 I→E 输出通道。只有该结构在 frozen Z/M 下形成持续、非 rank-1、非饱和 carrier，才释放 M。

新增交付：

- 统一 verdict：`results/topic4_sef_hfo/zm_mode_lifecycle/targeted_fast_modal_summary.json`
- 统一图：`results/topic4_sef_hfo/zm_mode_lifecycle/figures/targeted_fast_modal_mechanisms.png`
- 分析器：`scripts/analyze_topic4_zm_modal_fast_mechanisms.py`

## 10. 局部持续慢兴奋 conductance：剂量带、连接种子复现与冻结工作点扫描（2026-08-04）

一句话：这个机制确实把 burst 之间的深间隙填掉了，但填掉的方式是**把弛豫振荡整个消灭、换成一个静止的高秩空间图样**，而不是在间隙里托住一个还在运动的 carrier；预注册 gate 分辨不出这两者，因为"深间隙 ≤ 0.20"这条对一条恒定放电给满分。

### 10.1 机制与工程层

`cd654801` 引入的机制是 opt-in、E-only 的局部慢兴奋性 conductance，复用既有 mode-H 状态与其 Z/M 门控：

\[
g_{H,i}=g_{H,\max}\,h_i\,S_\zeta(z_i)\,S_M(m_i),\qquad
V_\infty=\frac{I_{net}+g_H E_{exc}}{1+g_H},\quad \tau_{eff}\propto(1+g_H)^{-1}
\]

`g_max=0` 严格走原 current-based 膜更新路径。本轮全部臂取 `rho_mode_H=0`，即关闭乘性 H、只留新机制，因此 `persistent_g0` 臂是本剂量序列自带的 matched control（H 传感器在跑，两条慢兴奋通路都不耦合）。

分析器侧新增（均带 TDD，`tests/test_topic4_zm_pv_som_carrier_panel.py` 16 项、`tests/test_topic4_zm_carrier_state_specificity.py` 11 项）：

- `_label` 接受 `rho=0` 剂量序列与 replicate SOM wiring；无 `mode_H_persistent_g_max` 键的历史 run 默认 0（parity path），不被丢弃；
- `_gap_spatial_class` 从实测 gap 与 PC1 推导失效轴，不手工贴标签；
- `adjudicate` 为纯函数，区分"单一剂量跨 wiring 通过 / 每个 wiring 各有通过剂量 / 某 wiring 无任何通过剂量"三种情形；
- `sustained_core_cv` 报告末 1 s 放电相对起伏（**报告量，不入 gate**）。

### 10.2 剂量带（wiring 1，冻结 `bounded_late__peak`，2.5 s，seed1）

| g | gain dB | occ | gap | PC1 | rank | core Hz | CV(末1s) | 质心 bin/s | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 30.02 | 0.346 | 0.700 | 0.737 | 1.902 | 86.6 | 1.840 | 65.5 | — |
| 0.04 | 29.24 | 0.365 | 0.750 | 0.966 | 1.181 | 91.2 | 1.894 | 51.1 | — |
| 0.08 | 27.99 | 0.410 | 0.625 | 0.943 | 1.299 | 93.1 | 1.420 | 73.4 | — |
| 0.12 | 23.90 | 0.374 | 0.575 | 0.953 | 1.248 | 105.1 | 1.305 | 79.7 | — |
| 0.16 | 28.20 | 0.408 | 0.550 | 0.952 | 1.252 | 108.0 | 1.239 | 90.9 | — |
| 0.20 | 28.14 | 0.450 | 0.375 | 0.944 | 1.277 | 125.4 | 0.964 | 83.9 | — |
| 0.24 | 29.25 | 0.443 | 0.400 | 0.936 | 1.300 | 133.2 | 0.994 | 84.3 | — |
| 0.28 | 27.02 | 0.424 | 0.350 | 0.932 | 1.308 | 145.6 | 0.893 | 82.6 | — |
| **0.32** | **25.21** | **0.547** | **0** | **0.823** | **2.114** | **187.0** | **0.048** | **7.7** | **PASS** |
| 0.40 | 19.87 | 0.527 | 0 | 0.603 | 3.411 | 139.2 | 0.051 | 6.9 | — (gain) |
| 0.48 | 18.77 | 0.672 | 0 | 0.497 | 6.096 | 206.6 | 0.044 | 3.9 | — (gain) |
| 0.64 | 0.22 | 0.035 | 0.040 | 0.455 | 5.422 | 217.1 | 0.040 | 2.7 | — (gain, occ) |

两条约束反向夹逼：能量占空要求强度往上，场增益要求强度往下。wiring 1 上 0.28 因间隙与占空不过、0.40 因增益 19.87 不过，实测**只有 0.32 一个采样点通过**；可行窗口的两个边界分别落在 (0.28, 0.32] 与 [0.32, 0.40) 内，未进一步细分。高剂量端（0.48/0.64）间隙填得更平、有效秩升到 5–6，但宏观场信号塌掉（增益 18.8→0.2 dB，占空 0.035）——高放电率下的去同步态，电极上读不出场。

### 10.3 连接种子复现（g=0.32 与 0.40）

| wiring | g=0.32 | g=0.40 |
|---|---|---|
| 1 | **PASS** | — (gain 19.87) |
| 2 | **PASS** | **PASS** |
| 3 | — (occ 0.4925) | **PASS** |

三套 SOM 连接布线**各自都有能通过 gate 的剂量**（0.32 / 0.32 / 0.40），无 wiring 缺席。因此不能写成"carrier 依赖某一套特定连接"，正确表述是**能通过的剂量随连接布线移动**。每个 wiring 均配 `g=0` matched control，三条对照全部是 gap 0.75、CV≈1.9 的 burst train。

机器 verdict：`PERSISTENT_SLOW_EXCITATION_CARRIER_REPLICATES_AT_A_WIRING_SPECIFIC_DOSE`。

### 10.4 承重结果：过 gate 的方式是弛豫振荡消失，不是间隙被托住

末 1 s 放电相对起伏在 g=0.28→0.32 之间发生一步跳变，且与 gate 结果完全共变：

- g ≤ 0.28（全部不过）：CV 0.893–1.894，空间质心速度 51–91 bin/s；
- g ≥ 0.32（含全部 4 条过 gate 臂）：CV 0.040–0.071，质心速度 2.7–20.8 bin/s。

**全 panel 18 臂中没有任何一臂同时满足"gap ≤ 0.20"与"CV 与 burst 臂同量级"。** 冻结工作点扫描的时间序列（`carrier_state_specificity.png`）逐格印证同一形态：凡是过 gate 的臂，都是 burst 序列在窗口内被阻尼掉、收敛成一条平线。

这暴露 gate 的一个结构性盲点：`post_onset_deep_gap_fraction ≤ 0.20` 对一条恒定放电取满分，因此 §7 最小要求里的 **"bounded non-tonic spatial orbit"** 的 non-tonic 一半从来没有被编进 gate。`sustained_core_cv` 作为报告量补上该缺口，**不修改预注册 gate**。

### 10.5 冻结工作点扫描（固定 g=0.32，wiring 1）

| 工作点 | 冻结 z | Z 门开度 | gap | PC1 | rank | core Hz | 慢兴奋 g 峰 | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `bounded_mid__rising` | 0.408 | 0.993 | 0.100 | 0.867 | 1.627 | 174.5 | 0.2109 | — (occ) |
| `bounded_mid__peak` | 0.406 | 0.993 | 0.325 | 0.967 | 1.180 | 161.7 | 0.1753 | — |
| `bounded_late__rising` | 0.375 | 0.993 | 0 | 0.736 | 2.740 | 193.0 | 0.2115 | **PASS** |
| `bounded_late__peak` | 0.373 | 0.993 | 0 | 0.823 | 2.114 | 187.0 | 0.1840 | **PASS** |
| `pre_entry__natural` g=0.32 | 0.525 | **0.989** | 0.450 | 0.976 | 1.141 | 87.0 | 0.1216 | — |
| `pre_entry__natural` g=0 | 0.525 | 0.989 | — | 1.000 | 1.000 | 54.3 | 0 | — |

carrier 只覆盖轨迹的 **late 段**（`bounded_late__rising` / `__peak`），两个 `bounded_mid__*` 点不过。

**已撤回的一次 over-claim**：本节初稿把 `pre_entry__natural` 当作间期基线，并把该点上机制仍有 66% 参与度（0.1216 vs 0.184）读作"Z 门在间期不关闭"。直接测量门开度后撤回：该点 z 已降至 0.525，Z 门开度 0.989，**面板五个工作点的门全部近乎全开（0.989–0.993）**，即本扫描**没有采样到任何位于门关闭侧的状态**，无从检验特异性。该点上机制参与度的差异来自活动驱动的 `h` 累积与 M 门，不是 Z 门。机器 verdict 相应改为 `CARRIER_ON_THE_LATE_ARC_SELECTIVITY_UNTESTED`，并在 `adjudicate` 中加入门开度前置判据，使分析器在门已开时**拒绝**给出特异性结论。

需要注意 `pre_entry__natural` 是 onset 前约 1.35 s 的 pre-ictal checkpoint，不是 interictal baseline；要检验"进入"与"退出"，必须先注册一个 z 位于门以上的 checkpoint。

### 10.6 12 s durability（全部 4 条过 gate 臂）

四条 2.5 s 轨迹均通过 `_validate_short_prefix`，即短臂是长臂的逐位前缀（无时钟平移；dynamic run 的 equilibration 区间保留在轨迹内，与 §5 M45 长跑同一约定）。

| arm | gain dB | occ | gap | PC1 | rank | core Hz | CV(末1s) | 质心 bin/s | 12 s gate | tail label |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| g=0.32 / w1 | 19.51 | 0.552 | 0 | 0.483 | 4.904 | 205.1 | 0.0388 | 1.94 | — (gain) | `tonic_tail` |
| g=0.32 / w2 | 21.13 | 0.554 | 0 | 0.656 | 2.733 | 202.8 | 0.0372 | 1.97 | **PASS** | `tonic_tail` |
| g=0.40 / w2 | 19.47 | 0.587 | 0 | 0.764 | 2.649 | 208.5 | 0.0461 | 0.90 | — (gain) | `tonic_tail` |
| g=0.40 / w3 | 19.33 | 0.599 | 0 | 0.538 | 5.358 | 210.7 | 0.0433 | 1.61 | — (gain) | `tonic_tail` |

四条**均无 runaway**。轴向 DMD 在四条上给出同一结论：leading mode 一律为 **0 Hz 实模态**，growth −1.60 至 −6.26 /s；`pathological_mode_candidate` 或本身即 0 Hz，或为 18–20 Hz 但 growth −25 至 −27 /s（强阻尼）。即持续段不存在自持的复模态/行波。

`g=0.32 / w2` 是唯一在 12 s 仍形式上通过全部七条 gate 的臂，但其 `tail label` 为 `tonic_tail`、末段起伏 0.0372、空间质心速度 1.97 bin/s、leading DMD 0 Hz。它是一个**持久的不动点**，不是 carrier。

另外四条的 gain 从 2.5 s 的 22–25 dB 一致降至 12 s 的 19.3–21.1 dB：2.5 s 窗口包含高幅 burst 瞬态，持续段本身的场幅度更低。这解释了为何 2.5 s 面板比 12 s 更容易过 gain gate。

### 10.7 当前能写与不能写

**能写**

- 局部持续慢兴奋 conductance 能可靠地把 PV/SOM 衬底上的弛豫 burst train 转成**有界、空间非均匀（有效秩 2.6–5.4）、不饱和、12 s 不失控**的高活动态；
- 该转变在 3 套 SOM 连接布线上均可复现，只是通过 gate 的强度随布线移动（0.32 / 0.32 / 0.40）；
- 转变是**突变式**的：末段放电相对起伏在 g=0.28→0.32 之间从 0.893 跳到 0.048，空间质心速度从 82.6 掉到 7.7 bin/s；
- 该态在轴向 DMD 下的主导模态是 0 Hz 实模态且被阻尼，持续段没有自持的复模态；
- 两条 gate 反向夹逼可行强度：能量占空要求强度往上、场增益要求强度往下，高强度端因去同步导致宏观场信号塌掉（g=0.64：增益 0.22 dB、占空 0.035）。

**不能写**

- 不能写已获得 ictal carrier：唯一在 12 s 通过全部 gate 的臂 `g=0.32/w2` 末段是 `tonic_tail`、起伏 0.0372、质心速度 1.97 bin/s，动力学上是静止态；
- 不能把"深间隙降到 0"写成"间隙被托住"：同一批数据显示间隙消失与振荡消失同时发生，且**没有任何一臂**同时满足 gap ≤ 0.20 与 burst 量级的起伏；
- 不能声称该机制在间期关闭或不关闭：本轮五个冻结工作点的 Z 门开度全部在 0.989–0.993，面板内**不存在门关闭侧的状态**；
- 不能把 `pre_entry__natural` 当作 interictal baseline（它是 onset 前约 1.35 s 的 pre-ictal checkpoint）；
- 不能进入 M 因果 / native offset / stimulation：合格的 fast carrier 尚不存在，释放 Z/M 只会测到一个静止态被慢变量拖走。

### 10.8 对下一步的约束

§7 的最小要求仍未满足，且本轮把缺口定位得更准：不是"能量不够"或"空间自由度不够"，而是**持续高能与持续时间结构在本机制下互斥**。局部慢兴奋提供的是一个稳定不动点的吸引域，强度越大吸引越强；把振荡保住的唯一方式是把强度压回 0.28 以下，而那里间隙又回到 0.35 以上。

因此下一个机制必须在**不依赖提高局部兴奋标量强度**的前提下，让持续段保留一个自持的时间尺度——即需要一个真正的慢负反馈与快正反馈之间的相位竞争，而不是再加一层局部正反馈。另需注册一个 z 位于 Z 门关闭侧的 checkpoint，否则"进入 / 退出"在任何后续面板里都无法检验。

同时记录一条方法学缺口：预注册 gate 的 `post_onset_deep_gap_fraction ≤ 0.20` 对恒定放电取满分，§7 要求的 non-tonic 从未进入 gate。本轮以 `sustained_core_cv` 作为报告量补上，**未修改 gate**；后续若要把 non-tonic 提升为判据，需在下一份 plan 里预注册，不能事后追加。

### 10.9 交付物

- 面板 verdict：`results/topic4_sef_hfo/zm_mode_lifecycle/pv_som_carrier_summary.json`
- 工作点扫描 verdict：`results/topic4_sef_hfo/zm_mode_lifecycle/carrier_state_specificity_summary.json`
- 图：`figures/pv_som_carrier_panel.png`（15 行机制对照）、`figures/persistent_dose_response.png`（四条 gate + 非 gate 起伏诊断对剂量）、`figures/carrier_state_specificity.png`（工作点扫描）
- 分析器：`scripts/analyze_topic4_zm_pv_som_carrier.py`、`scripts/analyze_topic4_zm_carrier_state_specificity.py`
- 测试：`tests/test_topic4_zm_pv_som_carrier_panel.py`（16）、`tests/test_topic4_zm_carrier_state_specificity.py`（11）

本轮共 22 条冻结轨迹（18 条 2.5 s + 4 条 12 s），全部 seed1 噪声、`--freeze-zm`；E→E topology/kernel/AR/direction/delay 未修改；新机制 default-off。
