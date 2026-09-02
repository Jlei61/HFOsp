# rev21：冻结双-core 底物上的 Z/M 跨状态审计

**日期**：2026-09-03  
**状态**：

```text
SEED_FACTORIZATION_COMPLETE
/
COARSE_INTERICTAL_RETAINED_NEAR_STATE_ONLY
/
NO_CROSS_STATE_WORKPOINT_IN_FROZEN_TIMESCALE_GRID
/
CONFIRMATION_NOT_RUN
/
PATIENT_ICTAL_TARGET_UNOPENED
/
FIG5_NOT_GENERATED
```

## 一句话结论

`dualcore_s39 + Joint=1.25` 的自然 K=2 容量不是单一随机种子的偶然结果，但精确匹配质量高度依赖
网络拓扑与随机动力学的交互。在冻结 Node 和连接后，当前 Z/M 幅度及时间常数网格没有找到一个点，
能够同时保留间期双方向结构并稳定进入广泛、持续且频率升高的模型发作态。因此本轮不能冻结 Fig.5
工作点，也没有打开患者发作数据做反向挑选。

## 1. 为什么先拆 seed

本轮把随机性拆成 `topology_seed × dynamics_seed`，而不是每个候选只跑一个绑定固定 seed。Z/M-off
正交审计共 3×4=12 个单元，12/12 都可自然分成两簇且两簇均存在。方差分解显示：

| 端点 | topology | dynamics | interaction |
|---|---:|---:|---:|
| 完整分布距离 | 0.032 | 0.216 | 0.753 |
| 双模板一致性 | 0.377 | 0.074 | 0.550 |
| OOD | 0.005 | 0.433 | 0.562 |
| 事件数 | 0.011 | 0.047 | 0.943 |

这说明 K=2 容量跨 seed 存在，但事件数和具体质量主要由 topology×dynamics 配对决定。后续筛选因此
全部用相同配对 seed 做 common-random-number 比较，并预留 fresh 3×4 confirmation；没有把同一个
seed 的重复 KMeans 初始化当作生物学复现。

## 2. coarse 幅度图

冻结 `tau_z=5 s, tau_m=0.5 s`，扫描
`s_I={0.7,0.8,0.9,1.0} × s_M={0.5,1.0,1.5,2.0}`，每点 2×2 seed 单元。
68/68 条仿真完成，0 OOM、0 工程失败。

初版聚合把 3--4 s 即进入的 active 轨迹与 20 s Z/M-off 轨迹直接比较，事件数不等，结论无效。
修订后的正式口径按每个 seed cell 的 active 事件数从配对 off 轨迹重采样 128 次；完整分布另报同 N
patient floor。还修复了“只有一个冻结方向时 `nanmean` 仍给高 balanced alignment”的问题。

正式 coarse 结果没有 model-ictal 合格单元。四个点保留冻结间期底物；其中
`s_I=0.7, s_M=0.5` 的状态短缺最小，因此只作为预注册 timescale map 的 seed，不是工作点：

- pooled complete returned families：14；
- frozen-direction counts：13:1；
- natural KMeans alignment：0.462；
- OOD：0；
- complete distance：1.138，仍远高于同 N patient q95=0.463；
- model-ictal：0/4。

所以“保留”只表示没有明显劣于同事件数的冻结 off 底物，不表示完整复现患者分布。

## 3. timescale 图

在上述 seed 上冻结 Z/M 幅度，扫描
`tau_z={3,5,8}s × tau_m={0.25,0.5,1}s`，保持 `eta_m × tau_m` 不变。36/36 条仿真完成。

主要结果：

1. `tau_z=3 s` 明显抬高触点频率，中位变化约 +19 至 +28 Hz；但 broad-state duty 中位仅
   0.51--0.65。唯一一个 model-ictal 合格单元位于 `tau_z=3 s, tau_m=0.5 s`，只是 1/4 seed。
2. 该点在 4 个 seed cell 合计只有 4 个完整返回事件，冻结方向为 3:0；自然 KMeans 无法估计，
   因而不能证明发作前仍保留双模板。
3. `tau_z=5--8 s` 的部分点保留间期底物，但 0/4 seed 达到模型发作资格。最佳双方向计数也仍明显
   偏向 mode 0，例如 `tau_z=8 s, tau_m=0.25 s` 为 13:2。
4. 9 个时间常数点均未进入同 N patient complete-distribution floor。

因此当前结果不是“Z/M 不能产生 runaway”。它说明的是：在冻结双-core + Joint=1.25 和本轮网格内，
能够快速进入并升频的状态会挤掉可评价的间期双方向 repertoire；能够留下 repertoire 的慢状态又没有
形成多-seed 稳健的广泛持续高态。

## 4. 科学边界

- 这是一个有限、预注册的参数图，不是局部优化器；阴性结果不能归因于 CMA-ES 或初始化落入同一局部最优。
- 它不排除网格外的 Z/M 参数、不同的慢变量方程或重新调整 EE/E→I expression；但当前数据不支持继续
  无边界扩大 Z/M 搜索。
- frozen dual-core 本身仍是 `COMPLETE_PATIENT_DISTRIBUTION_NOT_RECOVERED`。rev21 的“retained”
  不能升级 rev20 的结论。
- 患者 ictal target 没有参与候选选择，也没有在失败后打开；因此没有 clinical bridge，也没有合格的
  data-driven Fig.5。

## 5. 下一步

若继续，优先问题不是再加 seed 或细化相同网格，而是解释为何 `tau_z` 加快同时缩短了可观察的双模板
间期阶段。最小的新机制比较应让慢变量的**进入时刻**与**高态形态**可分离，例如冻结低活动驻留期后再
启用同一 Z/M 动力学，或增加不读取患者发作数据的 slow-variable onset delay 对照。只有该对照证明
双模板保留而高态形态仍可形成，才值得重新开放多-seed确认；否则应把“同一静态底物连续连接间期与发作”
收为当前机制的阴性边界。

## 6. 产物

- seed audit：`results/topic4_sef_hfo/data_driven_dual_core_zm_transition/seed_audit/seed_factorization_audit.json`
- coarse：`results/topic4_sef_hfo/data_driven_dual_core_zm_transition/coarse/aggregate.json`
- timescale：`results/topic4_sef_hfo/data_driven_dual_core_zm_transition/timescale/aggregate.json`
- 诊断图：`results/topic4_sef_hfo/data_driven_dual_core_zm_transition/timescale/figures/rev21_zm_timescale_boundary.png`

