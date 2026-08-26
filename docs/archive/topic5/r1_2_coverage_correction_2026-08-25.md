# Continuous marked-state R1.2：记录覆盖与连续性更正

## 一句话

旧 R1 的 survival coverage 不是完整记录轴：它由“至少含一个 definite-interictal IED 的记录块”反推，因而把真实存在但没有入选事件的 SEEG 块误作缺失。R1.2 已改为从 raw-SEEG `dataset_manifest.parquet` 的完整块清单建立时间轴，再减去 ictal 与发作后 2 小时；旧 coverage 与基于它的 timing 数字不得替代 R1.2。

## 为什么会影响科学结论

点过程似然同时包含事件项和对所有已记录时间的 survival integral。如果漏掉“有记录但没有入选 IED”的区间，模型不会因在这些区间错误预测大量事件而受罚，时间通道会被系统性偏置。这不是小的分母误差，而是把事件密度反过来写进 exposure 定义。

黄瀚文给出了直接反证：raw cache 的 N4/N5 块连续存在，但旧 event-derived coverage 没有这些块。旧路径只接受 511 个 validation raw anchors；恢复完整块清单后，raw cache 有 790 个，随后按 R1.2 的 ictal/2 h postictal 合同排除 118 个，最终 admissible validation 分母为 672。

## R1.2 的冻结规则

1. 记录覆盖只来自 raw-SEEG 完整 block inventory，不来自 IED 是否出现。
2. preictal 事件保留；ictal 与 seizure offset 后 2 小时的事件和 raw corrections 排除。
3. survival integral 只积在真实记录且 admissible 的区间；记录缺口不计 exposure。
4. 发作是 T1 未建模的干预。发作/postictal 禁用区之后新开 latent/history session，不把发作前状态无条件传播到发作后。
5. 每个事件按严格的 `z(t-)` 评分；恰好结束于事件时刻的 raw anchor 只能在该事件之后校正状态。
6. raw observer 在 Bridge-E1 的 inner-TRAIN 选择后冻结；全锚点 cache 与 T1 不再重新选择 observer。
7. “可读 raw anchor”还必须保留至少两个非 IED-core 背景样本，且至少一半触点未被该分钟的
   artifact mask 排除。这个条件现在在冻结分母时用与 `read()` 相同的实现检查，不再等 cache
   完成后才发现分母变化。

## 六人 development 分母

最终机器可读分母位于 `results/epi_prssm/continuous_marked_state/r1/r1_2/manifests/R1_2_ADMISSIBLE_DENOMINATORS.json`。它逐患者同时记录 raw cache 原始锚点数、R1.2 admissible 锚点数、因 ictal/postictal 排除的数量，以及完整 TRAIN/validation 事件数。普通阴性不会阻止六人实验完成；本轮不扩展到 34 人、不运行 H3 event-to-state edge，也不打开 sealed partition。

分母冻结时另发现 958 有 18 个训练窗、ZJQ 有 15 个训练窗和 9 个 validation 窗被密集 IED
的 ±1 s core 覆盖到不足两个背景样本。这些窗不能按冻结的“去 IED 后观察背景”合同构造输入，
因此在模型训练前排除并计入 attrition；最终六人合计 60,930 个可读 admissible anchors。

## 旧结果的边界

旧 R1 的工程测试、exact mark likelihood 实现与 raw observation 可读性检查仍可作为开发证据；但凡依赖旧 coverage 的 timing baseline、sampled T1 或 recorded-hours 数字，均不能作为 R1.2 的正式比较。被发现后生成的错误 smoke cache已移动到 `r1_2/_invalidated_event_density_coverage_20260825/`，并带有 `INVALIDATION.json`，不会进入聚合。
