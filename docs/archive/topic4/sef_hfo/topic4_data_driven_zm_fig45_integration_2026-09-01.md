# Topic 4 data-driven + Z/M Fig4/5 统一版

> 状态：Fig5A tonic/global-high 子图已锁定；Fig4 正式 artifact 与完整 Fig5 recovery 生命周期仍未锁定。

## 一句话判断

Topic 4 只保留一条 paper-facing 模型身份：**Fig4 是 data-driven Node + local-connectivity
间期底物，Fig5 是同一份冻结底物只增加 Z/M 慢变量**。当前 Fig4 仍只有 development-level
partial support；Fig5A 已由冻结 `tonic_b0_v2` 的 3 个全新 seed 锁定为持续、全局招募的
近饱和高态，但没有 recovery/termination，不能把该子图升级成完整 Fig5 生命周期或临床发作机制。

## 本次合并

- 主线基于 `799f5ccc`（data-driven Z/M）。
- 合入 `26bc4338` 的 dual-core OOD/pathway 代码、配置、测试和归档文档。
- dual-core 只作为 pathway/OOD 诊断：它改善了部分 OOD/Mode 2 指标，但没有恢复 natural KMeans、
  完整患者事件分布或 native HFO carrier，不能替代 Fig4/5 canonical model。
- 本次 merge 明确剔除 dual-core 分支的 28 个 `results/` 文件；运行产物只允许写到
  `/data/hfosp_topic4_fig45_artifacts`。
- spatial Z/M + stationary OU 的代码、测试与审计文档由 `1292e064` 合入；正式 Fig5A 产物外置于
  `fig5/data_driven_node_local_connectivity_plus_zm/paper_ready_fig5a/figures/`，配置逐文件锁定 SHA256。
- 干净主干重建时修复一条原分支已存在的 V6 冻结端点复现失败：V6 的 `t=0.25` 终点现在直接
  复用 hash-locked V5 coefficients，避免当前数值栈重复投影造成末位漂移；中间插值点和科学
  判据均未改变。

机器合同：`config/topic4_fig45_data_driven_zm_integration.json`；校验器：
`scripts/paper_figures/validate_topic4_fig45_integration.py`。

## Fig4 / Fig5 科学合同

### Fig4

模型身份是 `data_driven_node_local_connectivity`。必须并报 same-network 模式读出、fresh-network
或 held-out 确认、natural KMeans 与事件产率。只允许写成 data-driven substrate 的开发证据，
不允许写 patient-blind、解剖 core、native HFO carrier 或因果机制已成立。

### Fig5

模型身份是 `data_driven_node_local_connectivity_plus_zm`。必须先由
`audit_topic4_zm_exact_fig4_carryover.py` 证明网络、Node 场、EE/E-to-I 路径、delay 和 readout 与
Fig4 完全相同；唯一新增自由度是 Z/M。正式图还必须同时通过：全局 carrier、recovery、至少 3 个
fresh seed、同状态 PNG/PDF 目视验收。

当前分层状态：

- **Fig5A 子图已锁定**：`tonic_b0_v2` seeds 1841/1842/1843 为同一冻结参数族，3/3 通过
  tonic/global-high、数值稳定、stationary-OU 与 15/15 虚拟触点招募门；代表 seed 1842 按 onset
  中位数自动选取，PNG/PDF/SVG 与 metadata 同状态。
- **完整 Fig5 未锁定**：该轨迹进入近饱和高态后没有 recovery/termination；同时 Fig4 尚无已锁定
  paper artifact，因此 exact carryover 只能作为代码/配置来源关系，不能声称完成 artifact-to-artifact 闭环。
- **允许口径**：model-internal tonic/global-high transition。禁止写成临床发作重现、患者机制、恢复生命周期
  或生物学因果成立。

## 不进入统一版的线

- `topic4-m4-snn-native-exit`：carrier/recovery gate 失败，已移到仓库外归档。
- patient-specific cohort v2：只完成 8/28 人，没有 cohort 汇总，归档为未完成运行。
- spatial Z/M OU 的 Fig5A tonic endpoint 已纳入；旧 30–80 Hz 深调制门的阴性结论不撤回。
- Node rev18/rev19 历史链已在可恢复 bundle 中归档；当前只保留 rev20 活跃计算，运行中不纳入。
- RNN：迁出为独立 repo，不作为 Fig4/5 模型组成。

## 最小封板路线

1. rev20 完成后只审计最终 artifact，不在运行中移动 worktree。
2. 先锁定 Fig4 paper artifact，再执行 artifact-to-artifact exact-carryover 审计。
3. recovery 若要作为完整 Fig5 结论，必须用冻结机制与至少 3 个 fresh seed 单独通过；不得拿已锁定的
   tonic Fig5A 代替 recovery。
4. Fig5A 可先进入候选版式审阅；完整 Fig5 与 registry 的整体锁定等待上述两门完成。
