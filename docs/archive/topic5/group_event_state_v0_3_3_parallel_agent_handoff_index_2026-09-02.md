# Group-Event State v0.3.3：并行 Agent Handoff 索引

以下三份文件均是可直接复制给独立 Codex task 的完整 prompt。主 task 保留为 supervisor，不占用第四条科学工作流。

| 角色 | Handoff | 独占代码/结果职责 |
|---|---|---|
| Agent A：Evaluator & Assay | [`Agent A prompt`](group_event_state_v0_3_3_agent_a_evaluator_assay_handoff_2026-09-02.md) | canonical evaluator、E1146、D0–D5、power、eligibility |
| Agent B：Training Laboratory | [`Agent B prompt`](group_event_state_v0_3_3_agent_b_training_laboratory_handoff_2026-09-02.md) | T0–T6、训练 harness、搜索、training cards、训练队列 |
| Agent C：Scientific Experiments | [`Agent C prompt`](group_event_state_v0_3_3_agent_c_scientific_experiments_handoff_2026-09-02.md) | R0/R1、`S_N/S_G`、sharedness、H1/H2a、frozen H2b |
| 主 task：Supervisor | [`Supervisor runbook`](group_event_state_v0_3_3_training_supervisor_handoff_2026-09-02.md) | 资源租约、heartbeat、异常审阅、空闲 worker 补投、最终整合 |

## 共同前提

所有 agent 开始前必须完整阅读：

1. `/home/honglab/leijiaxin/HFOsp/AGENTS.md`；
2. `docs/topic0_methodology_audits.md`；
3. `docs/topic5_seizure_subtyping.md`；
4. `docs/archive/topic5/group_event_state_v0_3_2_post_review_corrections_2026-09-02.md`；
5. `docs/archive/topic5/group_event_state_v0_3_3_dual_view_state_spec_2026-09-02.md`；
6. `docs/archive/topic5/group_event_state_v0_3_3_dual_view_state_plan_2026-09-02.md`；
7. 本角色 handoff。

## 共同执行状态

当前 spec 是 `V0_3_3_REVISED_DRAFT_FOR_REVIEW_DO_NOT_EXECUTE`。因此：

- 没有 `V0_3_3_EXECUTION_RELEASE.json` 时，可以做只读审计、实现、单元测试、synthetic smoke 和资源 sentinel；
- 不得启动新的承重人体 development 训练；
- release 文件必须记录用户批准、spec/plan hash、base commit、sealed=false 和三条 worktree；
- release 后三条 agent 才能按各自权限启动 development-only 工作；
- sealed/formal、人体 H3 和 paper-ready Fig1–Fig4 始终不在本轮授权内。

## 共享目录

```text
大产物：/data/hfosp_group_event_state_v0_3_3/
小型索引：/home/honglab/leijiaxin/HFOsp/results/group_event_state/v0_3_3/
共享 manifest：.../v0_3_3/shared/
资源租约：.../v0_3_3/shared/resource_leases/
job request：.../v0_3_3/shared/job_requests/
job status：.../v0_3_3/shared/job_status/
```

每个 agent 使用独立 `codex/` branch/worktree 和独占结果子目录。共享文件必须按单条原子写入；禁止三个 agent 整体覆盖同一个 registry。

## 全局硬停

1. sealed partition 被读取；
2. 时间、患者、发作、normalization 或 target 泄漏；
3. canonical evaluator 对同一 checkpoint/anchor 给出不一致分数。

其它失败降低证据身份，不终止无关工作流。
