# Z/M 状态选择性模态与 conductance homotopy：阶段验收

**日期：** 2026-08-03  
**分支：** `codex/topic4-m4-snn-native-exit`  
**科学 verdict：** `NO_DURABLE_CREDIBLE_ICTAL_CARRIER`

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
