# Continuous Marked State H2b v0.2 验收裁决

> 本文件是 v0.2 的 post-review 权威裁决。它覆盖早期白话/技术报告中与本文件冲突的证据等级和命名，但不改写原始数值产物。

## 一句话裁决

v0.2 **通过工程、冻结边界和防泄漏验收，也通过“当前方案未建立 H2b 证据”的科学收口；不通过 H2b 科学问题的完成验收。** 它是合格的 negative pilot，不是完成的 negative experiment。

科学定位冻结为：`Module 1: frozen low-capacity probe falsification`。它只直接覆盖 susceptibility transfer 的一部分，不等于 dynamical continuity 或 organizational continuity。

## 1. 已通过的部分

- 最新已推送提交 `ca03c7ba55eabc61e7c45e7cdb75947d737b8d1b` 上，H2b 测试为 `112 passed`。
- v0.2 队列、15 个预期 probe、patient-first 汇总、phenotype availability、consumer-side 上游审计、机器审计和两份报告均已完成。
- 最终机器审计为 `PASS_COMPLETE`；这只表示 v0.2 工程链闭合，不表示 H2b 阳性。
- state/observer/generator/IED decoder 在读取 seizure task 前冻结，seizure loss 没有上游梯度路径。
- formal、sealed、H3/T2、physical clock 和 paper-ready figures 均未打开或运行。
- `B_history=[history]`、`B_observation=[history, observation]`、`B_state=[history, observation, persistent state]`。因此主量 `B_state-B_observation` 的确是嵌套增量，不是 state-only 与 observation-only 的替代模型比较。
- 上述设计矩阵已从实现本身核实：`risk_probe.py::FEATURE_PREFIXES` 对 `B_state` 明确同时解析 `history__`、`observation__` 和 `state__` 列；各臂使用同一 risk-set hash。
- 真正的 held-out 分母已公开；不可估计 permutation 被结构化标记，没有伪造零效应或 p 值。

## 2. 未通过的科学层级

### A. susceptibility transfer：未建立

30 min 暂合并各 development 支持层仅作诊断时，`state-observation` 为 7/8 患者方向有利，但唯一 nominal primary 患者只有 2 个最终 held-out risk sets，效应位于置换零带内。该方向不能升级为 cohort evidence。

### B. dynamical continuity：未支持

`persistent-memoryless` 为 5/8 方向有利，但唯一 nominal primary 患者方向不利。当前结果不能证明信息来自跨窗口持续记忆，也不能区分“生理上没有记忆”与“observer 覆盖了 generator carry”。

### C. time specificity：未建立

30 min `correct-time-wrong-time` 仅 1/5 方向有利。5 min descriptive 层 4/4 有利是 acute preictal signature 的探索线索，不是慢状态迁移证据。

### D. organizational continuity：不可估计

冻结 phenotype 只有 2/18 个患者×目标单元可估计。它不能支持 interictal-to-ictal phenotype transfer。

### E. 合并支持层的诊断图景：只作路线判断

把 30 min development 支持层合并仅用于诊断时，嵌套 `state-observation` 为 7/8 患者方向有利，`persistent-memoryless` 为 5/8，而 `correct-time-wrong-time` 仅 1/5。这个组合更像“表示中含宽泛情境信息”，尚不像“正确时刻专属、跨窗口持续的状态”。这些分母不构成统一确认队列，不能合并作 cohort p 值。

## 3. 为什么不能把 v0.2 称为完成的阴性实验

- `primary_chronological` 由总 eligible seizure 数命名，但唯一患者最终只评分 2 次发作；科学可估计性应由真实 OOF lead seizure 数与 assay power 决定。
- 因此旧标签只保留作软件分层兼容；科学报告一律改称 `primary-eligible by total support / chronological case series test n=2`。
- 当前一次性 chronological split 浪费了大部分可用发作；下一版应使用 past-only prequential evaluation。
- sign test 在 4–5 位患者层即使全部同向也无法达到常用双侧 0.05 阈值。
- 唯一 nominal primary 患者属于事前 H1-unstable 层，不能单独承担 persistent-state 假设。
- 单一 wrong-time donor 没有与慢状态自身时间常数对齐。
- v0.2 没有直接审计 generator drift、observation correction、open-loop horizon、有效秩、时间常数和 seed geometry。
- 没有证明任何迁移特异来自 IED objective，而不是一般背景建模。
- 没有在 outer-training clean interictal trajectory 上拟合并对 held-out seizure 作 OOS 投影的流形—流场检验。

## 4. 最终允许表述

> 在 development 数据上，当前冻结 R1.7B state 与低容量 risk-set probe 未建立时间专属、跨窗口持续的 H2b 状态迁移。若干患者的嵌套 state-containing 模型相对 observation comparator 呈有利方向，但唯一 nominal primary 患者仅有 2 个 held-out risk sets，效应位于时间保持置换零带内；持续状态未优于 memoryless state，30 min 正确时刻状态也未优于匹配错误时刻。因此，本轮限制的是当前 state instrument、probe 和数据支持量，而不是证明生理状态不存在。

## 5. 下一阶段决定

- v0.2：正式封存为 `engineering PASS / H2b NOT ESTABLISHED / negative pilot`。
- formal/confirmation：不升级。
- sealed partition：不打开。
- H3/T2：不进入。
- v0.3 development redesign：批准，先做 state instrument 与 assay qualification，再做 nested hazard、时间尺度和 OOS 流形—流场。
