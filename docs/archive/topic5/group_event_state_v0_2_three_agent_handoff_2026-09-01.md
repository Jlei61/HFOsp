# Group-Event State v0.2：三 Agent 交接索引

## 1. 每个 agent 共同先读

1. `docs/topic0_methodology_audits.md`
2. `docs/topic5_seizure_subtyping.md`
3. `docs/archive/topic5/group_event_state_v0_1_data_contract_2026-08-31.md`
4. `docs/archive/topic5/group_event_state_v0_2_common_contract_2026-09-01.md`
5. 自己负责的 spec+plan。

## 2. 分工

| Agent | 文件 | 独占结果根 |
|---|---|---|
| A：H1/H2a | `group_event_state_v0_2_h1_h2a_spec_plan_2026-09-01.md` | `.../v0_2/h1_h2a/` + `.../v0_2/shared/` |
| B：H2b | `group_event_state_v0_2_h2b_spec_plan_2026-09-01.md` | `.../v0_2/h2b/` |
| C：H3 | `group_event_state_v0_2_h3_spec_plan_2026-09-01.md` | `.../v0_2/h3/` |

建议从当前 v0.2 common commit 各建独立 `codex/` 分支/worktree。Agent B/C 不直接编辑 Agent A 的 core state 代码；需要字段时提出 additive schema 请求，或在自己模块中用 adapter。

## 3. 当前运行状态

- `/tmp/hfosp_group_event_state_v01` 的 v0.1 队列仍在运行，不能修改其代码或把新作业写入 `tag=main`。
- v0.1 中期 reset/wrong-time 可作旧语义诊断，不进入 v0.2 slow-state 结论。
- v0.2 warm fix 位于 `/tmp/hfosp_group_event_state_v02`；新作业必须用 v0.2 结果根/tag。
- 数据缓存不变且只读，不重新占用数十 GB 复制。

## 4. warm fix 验收

- `run_sequence` 显式接受/返回终态与 reset counter；
- validation/test 从当前 recorded session 真起点 replay；
- split pass + carry 与 uninterrupted pass 的 content/timing state 逐位一致；
- 固定 event-count warm cap 被禁用；
- 旧 checkpoint 的 validation selection 受 bug 影响，v0.2 承重结果必须重训。

定向测试：

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest \
  tests/test_topic5_group_event_state_streams.py \
  tests/test_topic5_group_event_state_no_leakage.py -q
```

## 5. 运行合同

- Python：`/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`
- 持久执行：nohup/setsid 或 tmux；单一 queue owner；日志、PID、STATUS、resume 命令落盘。
- CPU worker 线程均设 1；GPU 按实测显存提高 slot，遇 OOM 只降低资源参数，不删患者。
- 每个 run 原子写结果；完成依据是 result + source/config/checkpoint hash，不凭进程消失。
- 不触碰 formal/sealed、paper-ready figure 和主工作树无关 dirty changes。

## 6. 每个 agent 最终交付

1. `plain_language_report.md`：白话但不省略估计对象、分母、时序和证据边界；
2. `technical_report.md`：模型、切分、端点、患者×seed、独立分母、失败与复现命令；
3. machine JSON/CSV；
4. `CURRENT_HANDOFF.md`；
5. 图如生成，须有 `figures/README.md` 并目视检查 PNG/PDF。

