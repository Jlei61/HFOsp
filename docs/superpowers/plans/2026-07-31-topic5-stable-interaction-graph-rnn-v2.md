# Topic 5：Stable Interaction Graph RNN v2 执行计划

> **Gate correction（2026-07-31）**：Task 3b 只完成了 seen-distribution
> predictive stress test。它不能锁住 Task 5 的 structure stability 与
> unseen-start；后续执行顺序由
> `2026-07-31-topic5-stable-interaction-identifiability-v2_1.md` 接管。

## 1. 目标

把 v0.1 的“输出不反馈 autonomous trajectory”收口为历史 null，另立
contact-space feedback graph 合同。先证明工程和理想可辨识性，再碰人类
structure claim；SNN 全程独立。

## 2. Task 0：v0.1 closure 纠正

交付：

- v0.1 spec 顶部 closure note；
- Round 5 标为 `EXPLORATORY_COMPATIBILITY_CHECK_ONLY`；
- G0 标为 `REMOVED / NOT EVALUABLE`；
- M4 改名 `ALT-null`；
- M4-phase vs M3 标为 non-selection tie；
- 机器判决、人类归档和 index 同步；
- 不重跑任何 SNN 或 M4 扩展。

验收：全文搜索不再把 Round 5 写成 G0 negative，也不把 v0.1 扩大成稳定结构
阴性。

## 3. Task 1：数据可辨识性审计

新增入口：

`scripts/audit_topic5_sig_identifiability.py`

输出：

`results/topic5_stable_interaction_graph/development/identifiability_audit/`

步骤：

1. 只加载 frozen masked rank dataset；
2. 复用旧 train80 内 chronological development split；
3. 计算 start repetition、within-start suffix diversity、sender/intermediate
   coverage 和 unseen-start candidates；
4. 审计 `event_lag_raw` 实际数组及语义；
5. 输出 per-subject CSV、cohort JSON 和 fail-closed state；
6. 不读取几何、A/B、SOZ、ictal 或 SNN。

## 4. Task 2：通用 synthetic feedback graph

新增：

- `src/topic5_stable_interaction_graph.py`
- `scripts/run_topic5_sig_synthetic_benchmark.py`
- `tests/test_topic5_stable_interaction_graph.py`

顺序：

1. 实现 contact-space SIG0/SIG1；
2. 验证 `W[i,j]` 方向、`diag(W)=0`、candidate exclusion；
3. 验证真实过去的 likelihood factorization 与 generated-past free rollout；
4. 实现 paired intervention `I_eff`；
5. 冻结 12-contact 多起点/多分支 synthetic generator；
6. 运行 3 fit seeds；
7. 先写 per-run artifact，再聚合 G0-A，不允许人工挑 seed。

若 G0-A 失败，状态写 `BLOCK_HUMAN_PILOT` 并停止；不改阈值追结果。

首轮 G0-A 已因 influence 全 pair 排序一项失败。允许的唯一 remediation 是：

1. 保留原失败；
2. 做 endpoint/support/seed 诊断与 nested event-count curve；
3. 若曲线在冻结上限内越过原阈值，使用全新 graph/data seeds、相同阈值运行
   一次 G0-A2；
4. G0-A2 失败即停止，不再增加第三次确认。

## 5. Task 3：六患者 SIG-A pilot

仅在 G0-A 通过且数据审计合格后启动。

沿用 v0.1 target-blind 六患者与 development split，训练：

- M1-phase；
- M2-phase；
- M3；
- SIG0；
- SIG1。

每模型 3 seeds。第一轮不加入 event drive、process noise、多 fields、A/B
readout、发作 target 或 SNN。

主输出：

- held-out NLL/decision；
- free-rollout precedence、event-distance distribution 与 entropy；
- SIG1−SIG0 graph increment；
- 训练充分性和过/欠分散诊断。

## 6. Task 4：按诊断开放复杂度

- SIG1 within-start under-dispersed：开放 SIG2，`q={1,2,3}` 只用 inner
  validation 选择；
- 固定 `x_1,u_e` 后仍 under-dispersed：才评估 process noise；
- over-dispersed：先检查 graph shrinkage、temperature 与 phase balance；
- 任一复杂度升级必须保持相同 phase nuisance 和 free-rollout合同。

## 7. Task 5：结构稳定与 unseen-start

只对通过 generation adequacy 且审计合格的患者：

1. early/late 独立重训；
2. seed-folded `I_eff` stability；
3. phase-Markov、phase shuffle、static surrogate；
4. leave-one-start-group-out；
5. 未见 start 的 contacts 必须在训练中作为中间节点有支持。

## 8. Task 6：扩展规则

只有 G1–G4 全部通过，才：

- 扩展完整 development cohort；
- 冻结模型、阈值与统计代码；
- 选择另一数据集或真正未读 cohort 作 G5。

否则按最窄 bounded result 收口，不使用 SNN 或 outer heldout20 挽救。

## 9. 本轮最小完成定义

本轮必须完成：

1. v0.1 判决纠正；
2. v2 spec/plan 冻结；
3. 34 人 identifiability audit；
4. SIG 最小实现、单元测试和 G0-A synthetic 3-seed benchmark；
5. 根据机器 Gate 明确 `START_HUMAN_PILOT` 或 `BLOCK_HUMAN_PILOT`。

若 Gate 允许且时间足够，可继续运行六患者 pilot；否则停在有证据的工程边界。

## 10. 执行状态（2026-07-31）

本计划已按冻结顺序执行并收口：

| Task | 状态 | 结果 |
| --- | --- | --- |
| Task 0 | COMPLETE | v0.1 改为窄的 autonomous latent-trajectory bounded null；SNN Round 5 从 RNN Gate 删除 |
| Task 1 | COMPLETE | 34/34 可做 generation 评分，33/34 满足 unseen-start support |
| Task 2 | COMPLETE | 首轮 G0-A 保留失败；独立 graph/event seeds 的 G0-A2 在 9,600 events、未改阈值下通过；v0.2 统一训练充分性重跑后 PASS 不变 |
| Task 3a | COMPLETE | v0.2 的 36/36 fits 训练充分；SIG1 相对匹配 SIG0 的 NLL 与 precedence 同时改善 6/6 |
| Task 3b | COMPLETE / RECLASSIFIED | 54/54 强基线 fits 训练充分；1/6 来自 development-test、endpoint-specific oracle envelope，只说明 predictive dominance 未建立，不是 structure Gate |
| Task 3 主输出缺项 | NOT DONE | §5 列的“过/欠分散诊断”没有产出；因此 Task 4 的 SIG2 开放条件（within-start under-dispersed）**未被检验**，只是被 G1 未通过挡住 |
| Task 4 | NOT OPENED | G1 未通过，不以 event drive、process noise 或更多结构复杂度救模型 |
| Task 5 | REOPENED_UNDER_V2_1 | 先运行 structure stability、matched null 与 unseen-start；它们才直接裁决 stable/compositional structure |
| Task 6 | LOCKED_NOT_RUN | 不扩 34 人，不使用 outer heldout20 或 SNN 挽救 |

原机器状态 `COMPLETE_BOUNDED_DEVELOPMENT` 只作为 v2 历史 screen。v2.1 当前
安全结论是：

> emitted-contact feedback 相对 phase-only noGraph 有可重复增量；single fixed
> graph 没有在已见分布上取得相对灵活 mixture/template 的 predictive dominance。
> stable shared backbone 尚未裁决，必须由 matched-scale identifiability、结构
> 稳定性和可组合泛化检验。

机器状态：
`results/topic5_stable_interaction_graph/development/SIG_V2_DEVELOPMENT_STATE.json`。
