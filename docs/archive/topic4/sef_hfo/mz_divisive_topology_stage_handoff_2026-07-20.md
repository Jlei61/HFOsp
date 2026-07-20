# Topic 4 MZ-divisive / topology-first 阶段交接

日期：2026-07-20
分支：`codex/topic4-mz-divisive-lifecycle`
owner worktree：`.worktrees/topic4-mz-divisive-lifecycle`

## 1. 当前安全结论

1. 完整 current-based SNN 已能在无 kick 条件下从 returning interictal events 跨入持续到记录终点的 recruited activity，但该状态仍伴随 Z/`T_G` 漂移，不是已证实的稳定 ictal attractor，也未观察到 offset/recovery。
2. 当前 SNN 并非“没有空间 pattern”：operational onset 附近有 60-ms source-to-sink fast event sweep；但它不是秒级 local tissue-state recruitment front，也没有 wake、stall 或 annihilation。
3. 问题不是“additive/current-based 方程原则上不能发作”。Z 是局部乘性抑制，`T_G/S_G` 是 recurrent-E 除法增益，只有 M/候选 recovery 是加性负向输入。关键是 fast phase portrait 和 slow path 是否分别存在。
4. 原始六变量 reduced E/I fast block 在锁定范围仅有 low 与 `>100 Hz` saturation cliff。加入 delayed mean recurrent-E divisive pool 后，`z=0.85, alpha_G={15,16}` 出现约 605--609 ms 的闭合 period-1 orbit。
5. Stage 0E 的 shooting closure、base/half-`dt` 波形和 32 条多相位 finite-perturbation return 均强支持周期对象；但 bilinear-LUT Floquet Jacobian 没有 epsilon 平台，因而只能写“闭合且有强非线性回归证据的 mathematical periodic orbit，线性稳定性未解析”。

## 2. 阶段 gate

| 节点 | 结果 | 下游 |
|---|---|---|
| current SNN v1--v3 | operational recruitment，无 recovery | 只作失败边界图 |
| Stage 0A oracle | PASS，只验证 analyzer | 无机制 claim |
| Stage 0B six-variable fast block | `CLEAN_NO_GO_LOW_OR_SATURATION_CLIFF_ONLY` | 关闭 slow/space |
| Stage 0C dynamic divisor | 开出一条数值可信振荡线索 | 仅允许局部复核 |
| Stage 0D v1.1 | 175 unresolved + 5 survivors，预注册 open-basin 门未过 | 不据此排除周期 |
| Stage 0E | period-1 closure + nonlinear return PASS；Floquet epsilon platform FAIL | `STAGE0E_NUMERICAL_UNRESOLVED` |
| Stage 0F | 只复核 smooth/variational Floquet | 当前唯一开放节点 |
| Stage 1--3 | 未开放 | 禁止慢环/空间大扫描 |

## 3. 审阅反馈已收紧的要点

- global scalar 不能**单独**保留位置与局部历史，但它仍可通过局部核、延迟与非线性参与 pattern selection/extent containment；不应写成“global inhibition 不能影响空间 pattern”。
- 1D/2D 的 front 必须以 local frozen branch/orbit membership + 2--3 cycles dwell + onset/offset hysteresis 定义，不能用首次 spike 或 50-ms activity crossing，否则会把当前 fast event wave 误写成 slow tissue recruitment。
- 即使 Stage 0F 证实稳定，当前周期的 peak 仍约 96--98 Hz，`mu_E` 约跨 `-130...59 mV`；它首先只是 mathematical periodic orbit，不能直接叫 physiological ictal state。
- local `r(x,t)` 虽在方程中可仍是 additive subtraction，但它若只在 established recruitment 后启动、保存局部历史并把 frozen fast system 推过 exit surface，就能产生真正的状态切换；机制不由“加性/乘性”标签决定。

## 4. 主图与审计产物

- current-stage paper-ready diagnostic：`results/paper-ready-figure/fig5_mz_divisive_current_stage/figures/fig5_candidate_E1146_mz_divisive_current_stage.png`
- aggregate failure summary：同目录 `fig5_candidate_E1146_mz_divisive_failure_summary.png`
- 图说与 claim boundary：同目录 `README.md`
- 阶段反思：`docs/archive/topic4/sef_hfo/mz_divisive_current_stage_reflection_2026-07-20.md`
- topology-first 主合同：`docs/superpowers/specs/2026-07-20-topic4-topology-first-spatial-slow-fast-field-design.md`
- Stage 0E summary：`results/topic4_sef_hfo/spatial_slowfast_topology/stage0e_poincare_floquet_audit/stage0e_poincare_floquet_summary.json`

主图只支持：

> repeated returning spatial events precede operational recruitment; a distinct fast axial sweep occurs around onset, followed by heterogeneous recruited activity persisting to record end.

不支持 seizure lifecycle、slow tissue front、stable SNN attractor、self-termination 或 cohort mechanism。

## 5. 与并行 conductance 线的边界

- 本线：reduced rate/field，top-down 解析 fast topology、slow path 和 spatial front/wake 的必要条件。
- 并行线：full-conductance SNN，bottom-up 检验 reversal dependence、`tau_eff`、local/global GABA 与 presynaptic relay 能否自然实现该拓扑。
- 本线不改 LIF 膜电导，不实现 E->E presynaptic relay，不编辑 `.worktrees/topic4-mz-conductance`。两线在各自通过 fast-object/entry/exit/front gate 前不合并方程。

## 6. 其他 agent 的读取顺序

1. 先读本交接，不要从单张图或 `candidate_survives` 字段推断机制通过。
2. 再读 reflection 的 P0/P1 和 topology spec 的 stop rules。
3. 需要数值细节时才读 Stage 0D v1.1/Stage 0E summary；首次遗漏 return battery 的 Stage 0E archive 非权威。
4. 任何 Stage 1/space 工作都必须先检查 Stage 0F 最终 gate，不能用 Stage 0E 小谱半径点云绕过 epsilon-platform 失败。
