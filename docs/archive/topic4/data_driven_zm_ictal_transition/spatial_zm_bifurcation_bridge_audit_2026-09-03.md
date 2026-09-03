# 空间 Z/M 相图与分叉桥审计（2026-09-03）

## 1. 结论先行

患者特异空间 SNN 已出现可复现的 near-saturated tonic high branch，但当前双初值扫描尚未找到
严格的 `LOW/TONIC_HIGH` 同点共存。现有结果更支持三段形态：低事件态、宽的 intermediate/
亚稳带、近饱和 tonic runaway。它是有限随机网络的经验相图起点，不是解析分叉图。

现有 `topic4_m3b_spectral_phase.py` 不能直接把这条经验曲线升级为 saddle-node：它与 SNN
共享基础 LIF、连接度、权重和突触时标，但没有使用患者映射后的真实连接算子，也没有把本轮
per-neuron M 作为自洽状态放进固定点与 Jacobian。直接叠加其特征值会造成参数对象错配。

## 2. 已冻结的 SNN 对象

- 组织片：20 × 20 mm，32,000 E + 8,000 I，density=100/mm²，substrate seed 1842；
- fast parameters：`g=3.6`、`C_EE=C_IE=800`、`C_EI=C_II=200`、
  `tau_AMPA=3.5 ms`、`tau_GABA=18 ms`、external ratio=0.6；
- patient field：E 阈值 mean=17.9805 mV、SD=0.1417 mV、range=14.6819–20.0430 mV；
  `h>=0.5` 的 E cells 为 629/32,000；
- learned edge map：拓扑、delay assignment、GABA 不变；25.6 M 条 E→E 与 6.4 M 条 E→I
  权重被患者场重分配，同时逐 target incoming budget 守恒。E→E outgoing gain range
  0.753–1.312，E→I range 0.671–1.462；
- coordinate：`q_init=q_min=q_clamp`、`freeze_q=True`；q 只缩放 E cells 上的 inhibitory
  current；
- M：`dm_i/dt=-m_i/tau_m + spikes_i`，`tau_m=12.5 ms`，E current 减去
  `eta_m*m_i`；本轮 primary `eta_m=0.02`，q-only control `eta_m=0`；
- environment：持续空间 OU，sigma=0.10/ms、tau=20 ms、ell=0.38 mm；无 kick；
- low/high arms 的 fast state 与 M 来自 seed-1842 的 200/600 ms checkpoints；同一点两臂
  使用逐位相同的未来 Poisson 与 OU 流。

## 3. 当前经验相图

### eta_m=0.02 的完整双初值 Stage 0

| q | low-start | high-start | 联合分类 |
|---:|---|---|---|
| 0.860 | LOW, 69.36 Hz | LOW, 49.71 Hz | L/L |
| 0.840 | LOW, 74.66 Hz | INTERMEDIATE, 82.20 Hz | L/I |
| 0.820 | LOW, 79.06 Hz | INTERMEDIATE, 96.96 Hz | L/I |
| 0.805 | INTERMEDIATE, 111.98 Hz | INTERMEDIATE, 205.55 Hz | I/I |
| 0.790 | INTERMEDIATE, 197.50 Hz | TONIC_HIGH, 384.85 Hz | I/H |
| 0.770 | INTERMEDIATE, 238.81 Hz | TONIC_HIGH, 396.43 Hz | I/H |

### eta_m=0 的定向边界探测

| q | initial state | median rate | first/second half | active-E | 结论 |
|---:|---|---:|---:|---:|---|
| 0.82500 | low | 74.33 Hz | 74.08 / 74.49 | 0.261 | LOW |
| 0.82500 | high | 86.25 Hz | 96.98 / 81.94 | 0.301 | INTERMEDIATE，原高平台回落 |
| 0.81250 | high | 100.75 Hz | 124.49 / 82.96 | 0.357 | INTERMEDIATE，继续回落 |
| 0.80625 | high | 185.86 Hz | 181.55 / 190.16 | 0.579 | 长寿命 intermediate，非 tonic |

`q=0.80625` 的 20-ms 平滑率在全部 800 ms 打分窗均高于 120 Hz，但 median active-E 只有
0.579，median rate 只有 185.86 Hz，因此不能把它并入 near-saturated tonic branch。它可能是
第三个定常/周期态，也可能是有限时程亚稳态；当前 1.2 s 单 seed 不能区分。

## 4. 为什么现在仍不能叫真正分叉

### P0：deterministic reduction 不是同一个模型对象

现有 M3B 是 6-field `[rE,rI,sEE,sEI,sIE,sII]` 高斯核 neural field，默认统一阈值；本轮
SNN 是患者逐细胞阈值加真实映射邻接。虽然 `scale_II=False` 能匹配“q 只作用 I→E”，但空间
operator 与 gain field 尚未匹配。

### P0：M 没有进入自洽固定点和 Jacobian

M3B 只接受外加 frozen `gK_field`。本轮线性 M 在均值闭合下应满足
`m*(x)=tau_m*rE(x)`（rate 用 spikes/ms），并给 E drive 加 `-eta_m*m*`；动态 Jacobian 还需
新增 M block。否则 `eta_m=0.02` 的 branch/fold 位置不是本轮系统的 branch/fold 位置。

### P1：OU 与有限时程会制造随机跃迁和长驻留

空间 OU 是零均值但有色、持续的 accessibility process。deterministic mean-field 至少要分别
报告 OU mean-off fixed point 与有效输入方差敏感性；并补做“从 high checkpoint 开始、OU 关掉后
是否仍保持”的 persistence control。仅在 OU 打开时出现的跃迁必须写 noise-induced/
metastable transition。

### P1：现有 branch solver 不是 pseudo-arclength continuation

`solve_branches` 只是多初值积分并聚类可达稳态；`check_low_branch_continuation_between` 是 warm
start/bisection。它们可能找到低/高可达支，但不能穿过 fold 追踪不稳定中支。因而即使画出 S 形
外观，也不能据此标 saddle-node。

### P1：delay 对 Hopf 判定不可忽略

固定点和零特征值的 fold 条件在 lambda=0 时不受传导 delay 相位影响，但 Hopf 的复特征值会受
delay 强烈影响。零 delay 的有限 Jacobian最多用于 saddle-node 候选筛查，不能排除/确认 Hopf。

## 5. 最小可执行路线

1. **经验层先收窄，不跑原 348 条盲网格。** 用定向二分分别找 high-start tonic survival edge
   与 low-start escape edge；只有两条边界分离才在夹层补 3 paired seeds 和 2.5 s 驻留。
2. **先做 q-only deterministic bridge。** `eta_m=0` 可把问题降成固定 q 的 fast subsystem；
   用实际阈值分布和实际 E/I coarse-grained weight operators 解固定点，避免先引入 M 闭合误差。
3. **再加入 self-consistent M。** 增加 `m` field、`m*=tau_m*rE` 固定点约束和 Jacobian block；
   扫 `q × eta_m`，但只在经验边界附近。
4. **真正分叉门。** pseudo-arclength 找到稳定低支、不稳定中支、稳定/高活动支；fold 处一个实
   特征值穿零，并与 SNN 边界方向一致。若找不到不稳定支，保留“经验 transition map”措辞。
5. **噪声与尺寸控制。** 至少做 OU-off persistence、OU sigma 梯度和两个网络 size；后者若无
   finite-size scaling，不能写 phase transition。

## 6. 当前产出

- 双初值聚合：`/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_zm_phase_diagram/stage0_sparse_2d_aggregate.json`
- 相图 pilot：`/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_zm_phase_diagram/figures/spatial_zm_phase_stage0_sparse_2d.{png,pdf,svg}`
- q-only 单臂边界点：同一结果根的 `phase_points/stage0{e,f}_*`；因为不完整配对，不进入正式聚合。

**允许口径**：存在 tonic runaway 高态支和明显初值敏感性；当前边缘是宽 intermediate/亚稳带。

**禁止口径**：已发现 saddle-node、已证明 bistability、已证明 phase transition、B 图折点就是
解析 fold。
