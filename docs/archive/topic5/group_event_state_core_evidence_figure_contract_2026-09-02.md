# 群体间期事件状态：核心证据图合同

## 一句话

这三张图不是按当前 pilot 的显著性临时拼出来的，而是先固定最终科学问题与接口：

1. 间期历史能否形成持续预测未来事件块的状态；
2. 该状态能否改变一次群体事件如何继续和招募；
3. 该状态能否迁移到发作，并且 IED 是否在共同状态之外反馈性改变未来状态。

当前未运行的 H2b/H3 只画坐标和比较关系，不填模拟数据。

## 图 1：H1 future-block predictive state

**科学问题**：停止读取未来事件后，时刻 `t` 的候选状态能否预测未来 5、30、120 分钟的一片群体事件，而不是只预测下一次事件。

**承重比较**：

- B：状态相对可解释 multiscale history baseline 的未来事件数 log-score 增量；
- C：正确时刻状态相对保留自相关的 block-shifted wrong-time state 的增量。

两个纵轴均统一为“正值支持状态”。每条细线是一位患者，粗菱形为患者中位；缺失保持缺失，绝不画成零。

**最后理想结果**：B、C 在 30 和 120 分钟仍稳定高于零。只有 5 分钟阳性只能称短程历史记忆，不能称慢状态。

**当前 pilot 读数**：可估患者中，状态在 B 全部未胜过 multiscale history；C 只有一位患者在 5、30 分钟为正，另一位接近零，且 120 分钟不稳定。因此当前不支持可复现的慢预测状态。

## 图 2：H2a state-dependent repertoire

**科学问题**：在事件开头相同或相近时，状态是否改变事件后面会不会继续、继续时招募多少触点、具体招募哪些触点。

**承重比较**：正确时刻状态与 wrong-time state 在三个条件 likelihood 上的差值：

- B：continue vs STOP；
- C：positive size，条件于继续；
- D：contact subset，条件于 size。

三张统计 panel 共用 y 轴。“只改善 STOP/size”最多支持 extent state；只有 subset 和后续 same-prefix continuation 也稳定改善，才接近 repertoire/path state。

**最后理想结果**：正确时刻状态在多个物理 horizon 上稳定改善 subset 与 same-prefix continuation，而不是只改善事件大小。

**当前 pilot 读数**：continue 与 size 的患者中位接近零；subset 在一位患者 30 分钟明显反向，其他点接近零。same-prefix 的最终接口已在 payload 中固定，但尚未运行。因此 H2a 当前未建立。

## 图 3：H2b transfer 与 H3 feedback

这是一张预先固定的最终结果接口，目前不填任何观察点。

- A：冻结的纯间期状态，相对 multiscale history 对发作风险的 Brier skill；
- B：同一状态对发作早期空间能量场/传播路径的预测增量；
- C：在共同 pre-event state 已控制后，count/rate feedback 与 mark-specific feedback 相对 no-feedback 的未来事件块 log-score 增量；
- D：不同 IED 类型对功能状态的有符号 impulse response，允许促进或抑制。

**最后理想结果**：H2b 的增量在较早 lead time 仍存在，并以 held-out seizure 为分母；H3 的 mark-specific feedback 在未见 future block 上胜过 no-feedback，且有可复现的有符号状态响应。

**当前状态**：H2b/H3 尚未在 v0.3 pilot 上运行。画布上的 `not yet run` 是真实缺失状态，不是阴性结果。

## 固定数据接口

payload 格式为 `group_event_state_core_evidence_v1`，由
`src/topic5_group_event_state/v03/core_evidence.py` 生成。稳定字段包括：

- `h1_future_block.rows`；
- `h2a_repertoire.rows` 与 `same_prefix.required_fields`；
- `h2b_transfer.{risk_rows,field_rows}`；
- `h3_feedback.{model_rows,impulse_rows}`。

producer 为 `scripts/paper_figures/plot_group_event_state_core_evidence.py`。输出写到
`results/group_event_state/core_evidence/`，包含 PNG、矢量 PDF、metadata、payload 与中文 README。

## 图形合同

- 180 mm 双栏宽；tick/label/legend 不低于 7 pt；
- 患者为基本统计单位，显示患者线、患者中位和零线；
- 所有承重增量统一为正值支持假设；
- 未运行、不可估和数值零严格分开；
- 不在图内使用内部臂名或工程缩写；
- 当前包是 candidate framework，不登记为 canonical 主图，也不打开正式检验分区。
