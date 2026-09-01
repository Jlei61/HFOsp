# Group-Event State v0.2：三 Agent 交接索引（重大修订版）

三份文件都是可直接复制给独立 agent 的完整 prompt：

| 角色 | 交接文件 | 独占结果根 |
|---|---|---|
| Agent A：建立 predictive state | `group_event_state_v0_2_agent_a_handoff_2026-09-01.md` | `.../v0_2/h1_h2a/` 与 shared registry 的 producer 条目 |
| Agent B：seizure transfer | `group_event_state_v0_2_agent_b_handoff_2026-09-01.md` | `.../v0_2/h2b/` |
| Agent C：event feedback | `group_event_state_v0_2_agent_c_handoff_2026-09-01.md` | `.../v0_2/h3/` |

共同科学层级：A 建状态；B 判断它是不是癫痫易感状态；C 检验 IED 是否还对状态有额外反馈。A 的结论不 gate B/C，但 B/C 只能读取 registry 中已完成且由间期 objective 选出的 producer，不得静默 fallback。

## 共享文件

1. `docs/archive/topic5/group_event_state_v0_2_common_contract_2026-09-01.md`
2. `docs/archive/topic5/group_event_state_v0_2_engineering_invariants_2026-09-01.md`
3. A/B/C 各自的 spec+plan。

## 共享写权限

- Agent A 拥有 core producer、future target builder 和 registry producer-entry 写权。
- Agent B/C 将 core 视为只读；需要字段时优先写 adapter，确需 schema 时只加字段并通知 A。
- registry 采用每 producer 原子条目/锁，不允许整文件“最后写者覆盖”。
- 三个 agent 分别使用独立 `codex/` branch/worktree，不在主 dirty worktree 直接开发。

## 当前运行边界

交接时 `/tmp/hfosp_group_event_state_v01` 仍有旧队列使用 GPU；每个 agent 开始时必须重新核实 PID/GPU，而不是相信这句可能过期的状态。旧队列不得停止、修改或复用输出目录。GPU 高利用时先做 CPU/代码/target/schema 工作；资源允许后按工程附录的 smoke 峰值动态扩 worker。
