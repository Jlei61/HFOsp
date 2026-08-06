# MZ conductance + global inhibition：L=20 pilot

日期：2026-07-19
状态：本轮仿真完成；稳定工作点与 Z 阶梯通过；可恢复发作态未找到

## 1. 一句话结论

新的 conductance membrane 在 L=20 上找到了跨 seed 的 interictal 工作点，也复刻了“returning events 逐步消耗 Z，随后 runaway”的稳定阶梯；只有“保留 local GABA、额外加入不随 local Z 耗竭的 global conductance”能形成 runaway→prevention bracket，但在已限定的 M 轴上仍没有出现“持续高招募振荡后回到间期”的完整 lifecycle。

## 2. 方程与实现边界

E cells 使用

\[
\tau_m\dot V_i=-V_i+I_i^E+g_i^I(E_{GABA}-V_i)+g_i^M(E_K-V_i),
\]

并用 `V_match=18` 将旧 current proxy 做 force matching。主工作点使用 `E_GABA=V_L=0`；`E_GABA=V_reset=11` 的 pure-shunt sensitivity 在低增益下约 80 ms 即 runaway，不能复刻 current baseline。

global 轴分为两种，必须分开解释：

- replacement：`(1-beta) I_local + beta mean(I_I)`，保持均值但重分配空间抑制；
- additive：`I_local + beta mean(I_I)`，保留 local restraint，并在 conductance denominator 上增加 rank-1 global restraint。

这里的 global 项是 received-GABA surrogate，不是严格 presynaptic uniform kernel，也不是新增的秒级全局资源池。

## 3. 工作点

锁定候选：`gaba_gain=1.125, E_GABA=0, gamma=0`。

| seed | current returning / duration | conductance returning / duration | participation | peak rate | gate |
|---|---:|---:|---:|---:|---|
| 1 | 20 / 28 ms | 23 / 30 ms | 0.0375 | 45.31 Hz | 6/6 clauses |
| 3 | 22 / 32 ms | 31 / 32 ms | 0.0370 | 43.75 Hz | 6/6 clauses |

所有报告 cell `clip_fraction=0`；工作点最小 `tau_eff` 约 0.267–0.272 ms，高于 `2dt=0.2 ms`。

## 4. Z 稳定阶梯

`q75 I_th=95.1985, tau_z=2.5 s` 在两个 seed 都先产生多个 returning events，再 runaway：

| seed | pre-runaway returning | median ΔD | 正台阶比例 | event-locked 正增量 | event index–Dpre Spearman | runaway |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 18 | 0.01366 | 1.00 | 0.679 | 1.000 | 6081.5 ms |
| 3 | 14 | 0.01360 | 1.00 | 0.627 | 1.000 | 3712.6 ms |

`tau_z=5 s` 也在两个 seed 保持同方向阶梯：seed 1 在 8 s 内未 runaway，seed 3 在 29 个 returning events 后于 7085.4 ms runaway。安全表述是“稳定的 event-locked staircase-to-runaway route”；不能把 seed 1 的 8 s 无 runaway 写成永久稳定态。

## 5. global/local 竞争

replacement-total 和 replacement-protected 都失败：beta 增大反而更早 runaway。原因是 mean replacement 先拿走核心的强 local inhibition；仅仅保护 global 部分不被 Z 缩放不足以补偿。

additive + protected-global 形成清楚 bracket：

| beta | phenotype | returning events | runaway |
|---:|---|---:|---:|
| 1/12 | runaway | 19 | 7180.1 ms |
| 1/6 | interictal-like / near-prevention | 3 | 无 |
| 1/3 | suppress | 0 | 无 |

因此目前真正有用的更新不是“全局占比替换 local”，而是“保留 local，再额外加一个不随 local Z 一起耗竭的 global conductance”。它建立了动力学 bracket，但单独没有产生发作态。

## 6. M recovery screen

在 `beta=1/12` runaway 侧加入 M：

- 旧标定 5%/10%（fast 0.5 s、slow 2 s）全部进入 suppression/prevention；
- slow-M 细边界 `A_frac={0.25%,0.5%,1%,2.5%}` 仍未出现 lifecycle；
- 最弱 0.25% arm 有 14 bursts、2.5 Hz 谱峰、modulation 0.987，但连续高招募只有 9 ms，tail mean 2.124 Hz，远高于 baseline 上限 0.163 Hz，且无 late returning event。

所以最弱 arm 只能称为持续低强度 burst train，不能称为 ictal oscillation 或 recovery。更强 M 逐步进入 prevention/suppression。本轮安全结论是：**当前 additive-global + M 方程仍表现为 runaway↔prevention 的陡峭转换，没有解析出中间的可恢复发作态。**

## 7. 工程与资源验收

- 新 conductance hook 为 opt-in；旧 current、MZ、M4 shunt、STD、A1c、onset replay 联合回归 181 项通过；更新 engine manifest 后签名测试通过。
- 单进程 L=20 冒烟峰值 RSS 6.79 GiB；12 s worker 观测峰值约 9.19 GiB。
- 单 launcher + file lock + fork/COW；worker 硬上限 4，T≥20 s 上限 2；没有 OOM、swap failure、NaN、Inf 或 clipping。
- 每个 run 保存原子 JSON/NPZ、task hash、git 状态和输入文件 SHA。

## 8. 下一节点

本轮不应继续在同一 M 强度轴上做更细 tuning，现有结果已经重复给出 runaway→prevention。下一步若继续方程更新，应改变恢复变量的**非线性结构**，而不是只改增益：优先考虑 Abbott/Liou 路线中的慢适应激活阈值/饱和、或独立全局反馈状态，使 restraint 在高招募态内累积、在退出后衰减；同时保留本轮锁定的 `gain=1.125` 工作点和 Z staircase 作为回归基线。

机器可读汇总：`results/topic4_sef_hfo/mz_conductance/pilot_summary.json`。

## 9. 2026-07-20 时空图与结构反思

同一条 `beta=1/12 / Z on / M off` 连续轨迹的 paper-ready visual diagnostic 已补到
`results/paper-ready-figure/fig_mz_conductance_current_dynamics/figures/`。图中 returning event 保留清楚的
轴向 onset gradient，而 early runaway 的方向梯度塌缩且空间招募扩大；它支持“有序间期传播 → terminal runaway”的
时空变化，但仍不支持 bounded/recovered ictal state。

完整反思与下一版独立路线见：
`docs/archive/topic4/sef_hfo/mz_conductance_dynamics_reflection_and_next_model_2026-07-20.md`。
