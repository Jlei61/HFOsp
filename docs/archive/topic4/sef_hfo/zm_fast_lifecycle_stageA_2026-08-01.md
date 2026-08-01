# Z/M fast-lifecycle Stage A：E-only dynamic-threshold phenotype discovery

## 1. Verdict

`NO_CARRIER_IN_REGISTERED_PHI_PANEL_AND_TESTED_STATES`

在原始 current-based、per-neuron Z/M、各向异性 E/I SNN 上，E-only 动态阈值成功打破了旧的 refractory-saturated tonic branch，但 24/24 个 seed-1 production cells 全部落入同一类 `burst_train`：短促高幅 burst 之间存在 302–422 ms 的低活动区，virtual-SEEG 能量约 79–82% 时间落入相对深谷。没有形成持续、高能量、空间接力的 ictal carrier，因此不进入 reachable lifecycle vertical slice。

这是否定当前注册的 homogeneous-phi panel，不是否定所有动态阈值、适应或 fast inhibitory mechanism。

## 2. 增加了什么

保持 E→E、Z/M 方程、S_G、连接几何、checkpoint 和 future-noise bank 不变，只在 E 神经元加入 spike-triggered dynamic threshold：

\[
V^{\mathrm{eff}}_{\theta,i}=V^{\mathrm{base}}_{\theta,i}+\phi_i,
\qquad
\dot\phi_i=-\frac{\phi_i}{\tau_\phi}
+\Delta_\phi\sum_k\delta(t-t_i^k).
\]

I 神经元的 \(\phi\) 始终严格为零。跳变量按旧 tonic reference rate 标定：

\[
\Delta_\phi=
\frac{f_\phi\,(V_{\theta,\mathrm{core}}-V_{\mathrm{reset}})}
{(\tau_\phi/1000)\,r_{\mathrm{ref}}},
\quad r_{\mathrm{ref}}=439.229\ \mathrm{Hz}.
\]

Stage A 是 branch intervention：从四个真实 Phase-C Z/M tonic checkpoints 出发，\(z,m\) 逐神经元冻结，\(S_G\) 和 \(\phi\) 动态演化。它只问 fast phenotype 被改造成什么，不回答该状态能否从间期自然到达。

## 3. 跑了什么

- seed：1；
- checkpoints：`bounded_mid/late × rising/peak` 四个原生状态；
- \(\tau_\phi=60,100,160\) ms；
- \(f_\phi=0.15,0.30\)；
- 共 24 cells，每个 6 s，丢弃前 1 s switch-on transient；
- future noise：同一个 locked replay bank；
- 最大并发：12 个单线程 SNN worker；
- 单 cell 峰值 RSS 约 6.8–7.4 GB；并发期间观测到的最低 MemAvailable 约 177 GB，swap 未增长；
- 24/24 receipts 的 code SHA、source/noise hashes、6 s duration、5 s post-burn、`z/m` 零漂移和 I-cell phi 零值均通过审计；收尾无残留仿真进程。

## 4. 核心结果

| 指标 | 24-cell 范围 / 中位数 | 含义 |
|---|---:|---|
| phenotype | 24/24 `burst_train` | 没有 carrier candidate |
| core mean rate | 37.91–45.71 Hz；median 41.97 | tonic 高率被压低，但不是持续高能量态 |
| core modulation depth | 7.19–8.32；median 8.01 | 由平坦 tonic 变成大幅 relaxation pulses |
| median inter-burst interval | 270–316 ms；median 291.5 | 参数改变没有打开新的时间结构 |
| longest rest dwell | 302–422 ms；median 328 | 存在完整低活动间隙 |
| virtual-SEEG energy floor / p95 | 0.155–0.182；median 0.167 | 持续能量地板很低 |
| virtual-SEEG deep-gap fraction | 0.79–0.82；median 0.80 | 主要仍是离散 HFO/burst train |
| core–surround rate correlation | 0.915–0.935；median 0.925 | 招募域仍以 common-mode 脉冲为主 |
| `rho80` | 0 in 24/24 | 成功消除了旧 tonic refractory saturation |
| active occupancy | 0.149–0.171；median 0.163 | 远低于持续 carrier 的 0.80 operational marker |

5/24 cells 在单次 burst 内通过轴向 first-passage relay 描述符，但它们仍有完整能量间隙，因此不能升级为 `spatially_relayed_carrier`。这说明“每个 burst 内有传播”与“多个区域接力维持连续 ictal energy”是两件不同的事。

## 5. 动力学解释

动态阈值做对的一点是：它确实移除了旧 tonic fixed/high-rate branch 的表面表型，且没有靠 refractory saturation 兜底。

但它没有在 fast E/I subsystem 内创造一个新的 non-tonic high-state attractor。当前闭环更像：

\[
\text{同步高率支}
\rightarrow \phi,S_G\uparrow
\rightarrow \text{跌入低率支}
\rightarrow \phi,S_G\downarrow
\rightarrow \text{再次同步点燃}.
\]

也就是说，慢反馈在高支和低支之间制造了一个大幅 relaxation cycle，而不是在 ictal high-state 内产生持续的较小振荡、cluster relay 或 metastable carrier。由于实际平均放电从参考的 439 Hz 降到约 42 Hz，\(\phi\) 的自洽平均值只达到约 0.09–0.21 mV；提高 \(f_\phi\) 主要成比例提高平均 \(\phi\)，改变 \(\tau_\phi\) 主要改变单次峰值，但 24 个 cells 的 burst interval、能量地板和 phenotype 基本不变。四个 Z/M checkpoints 也没有打开不同分支。

因此这不是参数网格太稀，而是当前 homogeneous spike-triggered feedback 的流向：它只能周期性关掉共同高态，再等共同恢复后重启。

## 6. Stage decision

- 不做 `f_phi=0.075/0.45` 或 `tau_phi=40/240 ms` refinement：初始矩阵没有 tonic→carrier→silence 边界，只有同一 burst-train attractor；继续沿相同参数方向只是在调 burst 幅度和间隔。
- 不进入 Stage B：没有 frozen carrier candidate 可供 reachability、native exit 或 controlled exit 检验。
- 不把这张诊断图拼成 lifecycle Figure 5：它缺少持续 ictal energy、native offset 和 recovery。

## 7. 下一杠杆

下一版不应继续增强同一个 homogeneous slow brake，而应直接改变 fast high-state 内部的 E/I 组织，同时继续冻结 E→E scaffold。最短路线是先做一个小型 fast-inhibition carrier gate：让 inhibitory recovery/kinetics 或局部 I-population state 在固定 Z/M 下打破 common-mode pulse reset，检验能否把“各区同时停、同时重启”变成“局部群体错相接力、宏观能量连续”。

该 gate 必须优先看两项：

1. virtual-SEEG energy floor 是否显著抬高且 deep-gap fraction 降低；
2. core/surround 与轴向分区是否从 common-mode flash 变成稳定相位差，而非仅把 burst 频率调快。

如果这两个量没有同时改善，就不扩大参数网格，也不进入 termination/recovery。

## 8. Artifacts

- runner：`scripts/run_topic4_zm_fast_lifecycle_development.py`
- analyzer：`scripts/analyze_topic4_zm_fast_lifecycle_development.py`
- matrix：`results/topic4_sef_hfo/zm_fast_lifecycle_development/stageA_phenotype_matrix.{json,csv}`
- figure：`results/topic4_sef_hfo/zm_fast_lifecycle_development/figures/fig_stageA_phi_phenotype_matrix.png`
- per-cell traces/receipts：`results/topic4_sef_hfo/zm_fast_lifecycle_development/discovery/seed1/`

