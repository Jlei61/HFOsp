# Stage 0F v1.1 smooth/variational certificate 验收

日期：2026-07-20

分支：`codex/topic4-mz-divisive-lifecycle`

权威结果：`results/topic4_sef_hfo/spatial_slowfast_topology/stage0f_smooth_transfer_variational_certificate_v1_1/`

## 1. 结论

**接受并冻结本次执行为 clean numerical-unresolved。**执行合同完整通过，但周期轨道的横向导数证书未通过；Stage 1、slow lifecycle 与空间仿真继续关闭，不修改 gate 后重跑 v1.1。

锁定 verdict：`STAGE0F_NUMERICAL_UNRESOLVED`。

## 2. 已通过的证据

- `alpha_G={15,16}` 的 smooth shooting 均收敛；event-restarted `P/P2` closure 约为 `10^-15`。
- smooth 与原 LUT 周期只差约 `0.082 ms`，波形残差约 `3.2--3.6e-4`。
- shooting 与 variational nominal map 的周期和完整 crossing state 一致；continuous/discrete transversality 均约 `0.0061 /ms`。
- chain-rule 与两档 boundary-aware RHS stencil 的矩阵相对差均小于 `2.3e-4`；谱半径约为 `1.15e-4/7.24e-5`。
- 两点 peak E-rate 为 `98.15/96.33 Hz`，无 transfer support、natural-bound 或 `>=100 Hz` 违反。
- execution/provenance、failure artifact 与图形合同通过；无 execution exception，peak RSS `0.192 GiB`，wall time `187.16 s`。

## 3. 唯一科学阻断

两个点都只失败于 `base_whole_return_jv`：

| `alpha_G` | chain vs `epsilon=3e-4` | chain vs `epsilon=1e-3` | epsilon ladder |
|---:|---:|---:|---:|
| 15 | 0.00761 | 1.31735 | 1.31598 |
| 16 | 0.01313 | 1.26433 | 1.26621 |

较小 epsilon 与 chain-rule 很接近，说明周期候选仍有强稳定线索；但预注册合同要求两档 whole-return Jv 同时一致。不能在看到结果后删除 `1e-3`、放宽阈值或只保留较小 epsilon。因此当前不能写“Floquet stability 已独立认证”，也不能把该周期升级为稳定 ictal attractor。

这不是“不稳定”的证据。安全表述是：

> 两个 frozen-fast 点均有强闭轨、有限扰动回归和内部变分一致性证据，但独立 whole-return 导数的尺度一致性仍 unresolved。

## 4. 下游判决

- **GO**：原样归档并提交 v1.1 canonical result、lock、STATUS、报告与诊断图。
- **STOP**：不调整或重跑 v1.1；不把较小 epsilon 的通过改写成总体 PASS。
- **NO-GO**：Stage 1 `(z,D_R)` continuation、slow recovery、1D/2D spatial workflow 与完整 SNN 映射。
- 如继续数值诊断，必须另立预注册 numerical-resolution arm，专门解析有限扰动曲率、event localization 与 epsilon convergence；它不能回写成 v1.1 的通过证据。

## 5. 产物

- summary：`stage0f_v1_1_variational_summary.json`
- point outcomes：`parameter_point_outcomes.json`
- execution lock：`EXECUTION_LOCK.json`
- 状态：`STATUS.md`
- 图：`figures/stage0f_v1_1_variational_certificate.png/.pdf`
- 图说：`figures/README.md`

该图是阶段性数值诊断，不是 paper-ready 的稳定 ictal-state 证据。
