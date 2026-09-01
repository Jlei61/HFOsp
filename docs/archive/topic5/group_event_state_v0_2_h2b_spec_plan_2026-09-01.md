# Group-Event State v0.2-B：H2b Scientific Spec 与执行计划

本文件可独立交给 Agent B。开始前必须先读
`group_event_state_v0_2_common_contract_2026-09-01.md`。H2b 与 H1/H2a 分开运行；H1 分层不是纳入 gate。

## 1. 科学问题

状态模型只在间期群体事件任务上学习并冻结。随后问：在同患者内，它是否能在简单近期 IED 历史、时钟和当前事件表示之外，预测：

1. 距离下一次发作还有多久；
2. 未来 5/15/30/60/120/360 min 是否进入发作；
3. 下一次发作属于哪种预先定义的发作模式/入口路线；
4. 发作临近时是缓慢靠近、稳定高风险区，还是突然离开常态流形。

## 2. 当前旧线边界

- 旧描述性对齐有发作前偏移线索，但高可观测层较弱，不能代替 held-out transfer。
- R1.7 H2b v0.2/v0.4 工程完成但 H2b 未建立：独立 held-out 发作少、persistent 未胜 memoryless、correct-time 未胜 matched wrong-time，且 assay 不够灵敏。
- 当前 group-event v0.1 的 `run_h2b_transfer.py` 只取发作前最后一个事件与远期 control 做 AUC；它不是逐事件 distance/hazard，真实 checkpoint 尚未形成承重结果。

这些都是设计输入，不是当前 group-event state 的生物学阴性。

## 3. 输入冻结与防泄漏

- encoder、fast/slow dynamics、event decoder、整条状态轨迹全部冻结；发作标签不能反向更新状态。
- 每个 anchor 只能读取时刻 t 及以前的信息。
- ictal-overlap 排除，preictal 间期事件保留；记录结束、缺口和失访进入 right censoring。
- seizure pattern/route 只能由 TRAIN 发作或既有临床标签定义；held-out 发作不参与建类。

## 4. 主任务：逐事件 distance/hazard

每个间期群体事件时刻 t 输出：

- `time_to_next_seizure` 的离散生存分布或 cause-specific hazard；
- 多 lead cumulative incidence；
- competing route probability（支持足够时）。

事件行用于拟合，不当作独立统计样本。评估以 chronological held-out seizure/risk set 为基本单元，再 patient-first 聚合。报告 time-dependent log loss/Brier、calibration 和 seizure-level ranking；AUC 只作辅助。

## 5. 比较臂

1. clock/session baseline；
2. recent history：rate、last IEI、size/STOP、participation burden、time of day；
3. current event embedding；
4. fast state；
5. slow state；
6. fast+slow；
7. fast+slow + current event（判断状态是否超出当前事件）；
8. matched wrong-time state。

所有状态臂在相同 risk-set rows 上直接作嵌套差值。不能从分别汇总的模型中相减。

## 6. matched wrong-time 与发作模式

wrong-time donor 采用共同合同的同 session 匹配，并额外排除主发作前后窗口。每个 anchor 5–10 donors，报告无 donor 的发作数。

发作模式分层顺序：

1. 已有临床/冻结 pattern；
2. TRAIN-only IED-to-ictal reuse、early extent/entropy、laterality 等预先定义路线；
3. 若单患者每类不足，只作逐发作可视化与描述，不强行合并成一个效应。

## 7. 执行计划

### B0：独立仪器实现

重写以逐事件 survival/hazard 为主的 runner；用 v0.1 checkpoint 只做 `plumbing_only`，验证 censoring、risk-set、route 和 matched donor。synthetic 只测代码能否恢复已知方向。

### B1：真实 support inventory

对所有有 seizure metadata 的患者逐一报告：记录小时、事件数、发作总数、各 split 发作数、可用 risk sets、pattern 数、censoring。不要用总发作数替代 held-out 分母。

### B2：v0.2 冻结轨迹运行

读取 shared manifest 中全部可用 checkpoint，不按 H1 好坏剔除；H1-short/H1-slow 只作事前解释分层。3 seeds 全跑；发作足够的固定患者增加到5 seeds。

### B3：距离、lead 和 pattern 并行分析

主报告先给 continuous distance/hazard，再给多 lead，最后给 route。哪一层不估计不阻止另外层的探索。

### B4：报告

白话版必须回答“状态是否真的让我们更早知道离发作多远”，同时给出独立发作分母；技术版提供逐发作预测、校准、matched donor、patient-first 差值和旧线对照。

## 8. 允许结论

- slow state 超过 history/current event，且 correct-time 更好：development cross-task susceptibility state。
- 只预测某一 route：route-specific transfer，不外推所有发作。
- 只有当前事件 embedding 有效：preictal event phenotype，不称 persistent state。
- assay/分母不足：H2b 未建立，不作生物学阴性。
- 任何阳性仍不是临床预测器或 IED 导致发作的因果证据。

