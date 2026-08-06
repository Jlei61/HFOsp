# Topic 5 direct early-ictal transfer v0.2 执行计划

对应 spec：`docs/superpowers/specs/2026-08-02-topic5-history-rnn-direct-early-ictal-transfer-v0_2.md`

## Milestone A：修正 v0.1 口径

1. 将 `ACCEPTED_BOUNDED_NEGATIVE_CLOSEOUT` 改为 `PROVISIONAL_BOUNDED_NEGATIVE_FOR_CURRENT_G1_TASK`。
2. 文档明确区分 static、M1、EventRNN 和 HistoryRNN 分别提供的信息。
3. 修正 static→unordered 与 dataset-specific chronology 数值归属。
4. 图题改为 target-blind next-event evaluation，early-ictal 箭头标 `not evaluated in v0.1`。

## Milestone B：G1 能力与训练审计

1. 运行 synthetic chronology positive control，3 seeds。
2. 在固定 development patients 上比较 3/10/30 cycles。
3. 从正式 checkpoint 汇总 state variance、readout norm、decay drift。
4. 对代表 fold 做 zero-state/state-field ablation。
5. 输出 `G1_DIAGNOSTIC_VERDICT.json`；只有 recoverability 和训练充分性均通过，才把 G1 阴性解释为当前 objective 下的信息阴性。
6. 若 c3→c10 仍变化但 c10→c30 已稳定，则 target-blind 重训 16 个 c10 checkpoints，并以其重跑 direct transfer；禁止继续沿用 undertrained c3 checkpoint 给 RNN 下结论。
7. 若 c10→c30 未稳定，则继续完成 16 个 c30 checkpoints，并在完全相同 direct-transfer 合同下比较 c10 与 c30；禁止按 early-ictal target 挑选训练预算。

## Milestone C：实现 direct transfer ladder

1. 保留 M0/M1。
2. 加入 0.5/2/6 h EWMA 和 multi-horizon fields。
3. 加入 true HistoryRNN 与 strict order-shuffled HistoryRNN。
4. target-patient LOSO ridge 只在 outer target patients 拟合。
5. 增加 patient-mean oracle、correct/wrong pairing 和 seizure-specific residual 输出。

## Milestone D：运行 16-fold direct transfer

1. 先跑 `epilepsiae_1146` shape/leakage smoke。
2. smoke 通过后并发 16 个 outer folds；并发数从 2 开始，依据显存和吞吐上调。
3. 每个 fold 保存 contact predictions、seizure metrics、wrong pairing、residual metrics 和 readout metadata。
4. 汇总 15 人 primary、16 人 supportive、10 人 primary pairing（11 人 supportive pairing）和 3 人 residual candidate。
5. 将首次 c3 direct 结果保留为训练预算 sensitivity；canonical summary 必须写明最终 checkpoint cycles，并与 c3 做患者配对比较。
6. 若启动 c30，最终必须同时给出 c3/c10/c30 的预算比较；只有 c10 与 c30 科学结论一致，才称为 training-budget robust。

## Milestone E：验收和画图

1. patient-first 统计 R2−M1、E2−M1、EM−M1、R2 true−shuffle。
2. 对每患者运行 5000 次 all-contact target-label shuffle；以 seizure median 折叠，报告 M0/M1/R2 等模型的 absolute margin 与 target headroom。
3. 生成 direct-G2 六联图：任务、target headroom、all-contact channel null、模型相对增量、时间/zero-state 对照、pairing/residual。G1 的 synthetic、state-utilization 和 c3/c10/c30 训练审计另放诊断图，不占用 direct-G2 主叙事。
4. 图逐张目视检查并写中文 README。
5. 更新 Topic 5 主文档和 archive，给出可写/不可写结论。

## 停止规则

- v0.1 next-event proxy 不再调参追阳性。
- synthetic recoverability 失败：先修模型/训练器，不解释 G1 阴性。
- direct transfer 无论阳性或阴性都运行到 patient-first 汇总；不由 G1 再次锁死。
- 16-fold 结束后不临时扩展 architecture zoo。
- frozen transfer 结果落地前不端到端微调 HistoryRNN；G3/pairing 是 secondary，不作为患者级 G2 的硬门。
