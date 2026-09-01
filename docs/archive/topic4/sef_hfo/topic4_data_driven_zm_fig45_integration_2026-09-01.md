# Topic 4 data-driven + Z/M Fig4/5 统一版

> 状态：代码合同已合并，artifact 尚未锁定；本文件不升级任何科学结论。

## 一句话判断

Topic 4 只保留一条 paper-facing 模型身份：**Fig4 是 data-driven Node + local-connectivity
间期底物，Fig5 是同一份冻结底物只增加 Z/M 慢变量**。当前 Fig4 只有 development-level
partial support，Fig5 没有通过全局 carrier、recovery 与三新 seed 联合门，因此还不能封板。

## 本次合并

- 主线基于 `799f5ccc`（data-driven Z/M）。
- 合入 `26bc4338` 的 dual-core OOD/pathway 代码、配置、测试和归档文档。
- dual-core 只作为 pathway/OOD 诊断：它改善了部分 OOD/Mode 2 指标，但没有恢复 natural KMeans、
  完整患者事件分布或 native HFO carrier，不能替代 Fig4/5 canonical model。
- 本次 merge 明确剔除 dual-core 分支的 28 个 `results/` 文件；运行产物只允许写到
  `/data/hfosp_topic4_fig45_artifacts`。
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
fresh seed、同状态 PNG/PDF 目视验收。当前状态是 `BLOCKED_NO_ELIGIBLE_ARTIFACT`。

## 不进入统一版的线

- `topic4-m4-snn-native-exit`：carrier/recovery gate 失败，已移到仓库外归档。
- patient-specific cohort v2：只完成 8/28 人，没有 cohort 汇总，归档为未完成运行。
- spatial Z/M OU 未提交改动：仍在原 worktree，未纳入本次 committed integration。
- Node rev18/rev19：仍在运行，最终 artifact 出来前不纳入。
- RNN：迁出为独立 repo，不作为 Fig4/5 模型组成。

## 最小封板路线

1. 等 Node rev18/rev19 完成，只审计最终 artifact；不得在运行中移动 worktree。
2. 从 Fig4 冻结底物生成 exact-carryover Fig5 候选，不再更换 Node/连接性来抢救 Z/M。
3. 先过 carrier + recovery + 3-seed gate，再生成 paper-ready PNG/PDF。
4. 图通过目视审阅后，才把 `artifact_state` 从 `NOT_LOCKED` 改成锁定状态，并更新
   `docs/paper_figure_registry.md`。
