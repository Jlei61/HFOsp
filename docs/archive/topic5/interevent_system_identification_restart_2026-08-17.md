# Topic 5 跨事件隐状态 / 系统辨识线重启审计（2026-08-17）

## 1. 一句话状态

这条线已经完成“短程隐状态可追踪”的 observer 层，但没有建立“单场事件驱动后续状态转移”。
本次已恢复一个独立、可复现的工程入口，并重新通过 synthetic identification；旧 34 人
`v3.1` 人体 transition 仍由预注册 `NOT_TRIGGERED` 闸门关闭，不能事后加容量重开。

## 2. 与事件内 next-sequence RNN 的边界

- 事件内线：一场事件内部按 contact/rank 展开，预测下一个序列元素。
- 本线：一个时间步是一整场完整间期事件；状态描述同一患者稳定传播 repertoire 的慢变表达。
- 本线不得把 hidden weight 解释成接触点网络，也不得把预测增量写成 plasticity 或 causal shaping。

## 3. 已完成阶段

### V2.7：state tracking observer

- 34/34 患者、3 seeds、102 个正式 run 和全部 controls 已验收。
- RNN 相对 static repertoire 改善，但没有稳定超过固定 EWMA。
- 每场事件清零 state 会变差，模型整合的主要是最近约 10–20 场事件。
- block shuffle、time reversal 和 H40 没有支持 chronology-specific 长程计算。

安全口径：稳定 repertoire 存在短程、within-recording 的 leaky state tracking。

### V3.0：innovation association / local projection

- 先用过去事件估计 pre-event state，再把当前完整事件中无法由过去预测的成分定义为 innovation。
- 只有 17/34 患者达到 `INNOVATION_VALID`；其余 9 人历史不足，8 人残差仍可由过去预测。
- 单事件 local response 主指标为负；20 场累积路线方向为正但未过冻结门，且高支持度分层翻负。
- 冻结 evidence level 为 Level 1 `leaky_observer`。

安全口径：近期完整事件有助于追踪当前状态，但有效 innovation 未提供路线一致的未来传播信息。

### V3.1：matched recurrent transition

- T1 observer-only 与 T2 event-driven transition 共用 observation、filter、state dimension 和训练合同；
  T2 唯一新增项是 transition 中的 `B * innovation`。
- T3 是离散 switching control；不是 GRU/Transformer architecture sweep。
- synthetic observer-only、event-driven 和 switching 三种真值均通过识别校准。
- V3.0 validation-only handoff 为 `NOT_TRIGGERED`，所以旧 34 人人体 transition 从未获准执行。

## 4. 本次重启动作

- 独立 worktree：`.worktrees/topic5-system-identification-restart`
- 分支：`codex/topic5-system-identification-restart`
- 复现检查：48 个 V3.0/V3.1 核心测试通过；扩展到 V2.3→V2.7 上游链后，
  clean worktree 为 194 passed / 2 skipped，6 项只因隔离 worktree 不含 ignored `results/`
  冻结产物而失败；同 6 项在原产物根目录复跑为 6/6 passed。
- synthetic 重跑：
  `results/topic5_event_innovation_state_space/v3_1/restart_synthetic_2026-08-17/`
- synthetic 状态：`SYNTHETIC_TRANSITION_IDENTIFICATION_COMPLETE`；`human_data_read=false`；
  shared-parameter equality、observer-only negative control、event-driven recovery、switching control 均通过。
- 当前 GPU 空闲；另一个 LBSS/motif worktree 的长期 CPU workers 未被终止或覆盖。

本次没有读取或重算 V3.0 人体 test，也没有启动 V3.1 人体拟合。

## 5. 当前最大缺口

最大缺口不是 RNN 容量，而是可辨识性与统计支持：

1. primary innovation 只有 17/34 可定义，且 Yuquan 仅 2/16；
2. 累积效应实测中位约 `+0.0041`，低于既有 cohort 汇总可探测下限 `+0.0069` 至 `+0.0083`；
3. 当前可探测下限只覆盖 patient-level aggregation，没有量化 patient 内 observer/response pipeline
   对真实 transition 的衰减；
4. 同一 34 人上直接运行 V3.1 会违反预注册 handoff，不能作为修复。
5. 6 个验收测试直接依赖工作树本地的 ignored `results/` 路径；代码已恢复到 git，
   但 clean checkout 的 artifact locator / read-only mount contract 仍需单独修复。

## 6. 下一阶段的冻结问题与门槛

下一阶段应是独立的 measurement-recovery / confirmation-readiness 合同，不是 `v3.1 rescue`。

### 首要问题

在保留真实 chronology、participation mask、tie 和 source boundary 的条件下，向合法的 observable
rank/precedence trajectory 注入已知 event-driven transition 后，现有 observer → innovation → response
流水线能否恢复其方向、大小和 horizon profile？

### 必须预先冻结

1. 注入发生在 observable future rank/precedence field，而不是只改 latent state；
2. participation mask、tie 与 contact order 的合法化规则；
3. 注入强度网格、随机种子、患者集合和 patient-first 汇总；
4. observer-only 零注入、方向反转、state-matched donor、source-coherent block null；
5. 恢复门：方向 cosine、效应衰减比、80% recovery threshold、false-positive rate；
6. 人体结果和旧 V3.1 transition 在设计与校准阶段均不可读取。

### 决策规则

- 若真实数据结构上的注入恢复失败：停止人体 transition，优先改测量或换数据，不增加 RNN 容量。
- 若恢复通过但旧队列效力不足：只形成独立队列/更长记录的确认合同，不重开旧 34 人。
- 只有新合同的 validation-only Gate 通过，才允许 one-shot matched transition test。

## 7. 当前允许结论

> 同一患者稳定的间期传播 repertoire 存在可由最近约 10–20 场完整事件追踪的短程隐状态；
> 现有 34 人探索性数据没有证明单场事件 innovation 驱动后续传播状态转移。

目前不能写：`event-driven transition identified`、`activity-dependent shaping`、
`causal plasticity`，也不能把本线重新解释成事件内 next-sequence mechanism。
