# 可直接交给 Agent A 的 Prompt：Evaluator、Assay 与数据合同

你接手 `/home/honglab/leijiaxin/HFOsp` 的 Group-Event State v0.3.3 Workstream A。你的职责是建立唯一评分器、解释 E1146 的方向差异、校准 synthetic power，并给出每个 endpoint/horizon 的可估分母。你不负责调人体神经网络，也不能根据人体效果选择架构。

## 1. 开始前

1. 完整阅读 handoff 索引列出的共同文档。
2. 检查 `git worktree list --porcelain`、当前分支/commit/dirty state、活动进程、GPU/CPU/RAM/磁盘和已有输出。
3. 不在主 dirty worktree 开发；从 supervisor 指定的 clean release commit 建 `codex/topic5-ges-v033-evaluator-assay` 独立 worktree。
4. 若 `V0_3_3_EXECUTION_RELEASE.json` 尚不存在，只做审计、实现、测试、synthetic smoke；不启动承重人体运行。
5. 旧 v0.3.2 与 Topic 4 活动作业只读；不得停止、覆盖或复用其输出键。

## 2. 科学职责

你必须让下列问题有唯一答案：

1. 同一 checkpoint/anchor 在训练、独立评价和画图时是否得到同一个 proper score？
2. v0.3.2 中 E1146 的 +0.1277 与 −0.3291 从哪一个具体数据行/步骤开始分叉？
3. 已知真 state、真 memory、可读 synthetic mark 时，Level 0–2 是否能够恢复？
4. D0–D4 不同真值和效应强度下，continuous gain、false positive 和 power 是什么？
5. 中等 oracle effect 至少需要多少独立 blocks，哪些患者/endpoint/horizon 真正可估？

## 3. 必须实现

### A1. Canonical evaluator

建立单一纯函数和 per-anchor schema：

```text
subject / seed / checkpoint_hash / anchor_time / split
target / prediction_H / prediction_H_plus_state
dispersion / mask / weight / per_anchor_NLL
eligibility / evidence_label
```

训练 branch、evaluation branch 和 figure payload 复用该实现。测试必须覆盖 anchor permutation、mask、dispersion、intercept、weight、sign 和 reduction。

### A2. E1146 逐行 diff

依次比较 checkpoint、anchor、target/prediction、dispersion/intercept、weight、seed aggregation、score sign/reduction。输出第一处分叉及其上下游影响；不得只写“重构后已一致”。

### A3. Oracle Level 0–2

- Level 0：真 state，只训练 output head；
- Level 1：真 event innovation + fixed leaky scan，只训练 readout；
- Level 2：synthetic mark channel，训练 encoder + readout。

每层保存 truth、prediction、held-out continuous gain 和 failure location。

### A4. D0–D4 power

- D0：H-only；
- D1：count-only；
- D2：grammar-only；
- D3：shared count+grammar；
- D4：independent states。

复用真实时间轴、coverage 和 split。每次改代码先 3 replicates smoke；nightly 10；里程碑 20–30。效应以 oracle held-out deviance gain/block SNR 定义，不用任意 β 或 pass count 代替 power curve。

### A5. D5 与 support

D5 background-only 仅少量预期失败。资格计算必须调用真正 window builder 的 coverage segment，不得用 session 数或滑动窗口总数冒充独立样本。

### A6. 数据边界

冻结并测试：target 不跨 gap/split/seizure；发作和立即 postictal 不更新间期 state；autonomous flow 继续；真实 gap/session reset；hard seizure reset 仅敏感性。

## 4. 写权限

你独占：

```text
/data/hfosp_group_event_state_v0_3_3/agent_a/
results/group_event_state/v0_3_3/evaluator_assay/
shared/evaluator_contract/
shared/eligibility/
```

不得编辑 Agent B 的训练 config 或 Agent C 的科学 endpoint。共享 registry 只写你拥有的 evaluator/eligibility 条目，原子 rename 后发布。

## 5. 并行和资源

- Python：`/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`。
- CPU worker 设 `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`。
- 先测 synthetic sentinel 的 peak RSS、I/O、wall time，再按 RAM/CPU/I/O 余量扩 worker。
- 不因两张 GPU 空闲就占用 GPU；本线优先 CPU。GPU oracle 若确有收益，先单作业测峰值并写 lease。
- 长任务用 `nohup`+`setsid` 或 tmux；独立 log、status、heartbeat、原子输出、幂等 resume。
- 不得 `pkill -f`；只用精确 PID/PGID 管理本线作业。

## 6. 状态与交付

每 5–10 分钟原子更新 `CURRENT_HANDOFF.md` 和 `agent_a.status.json`：当前 commit、已完成、正在跑、pending、失败、资源和下一步。

完成交付：

```text
canonical_evaluator.json
e1146_discrepancy_audit.json
oracle_level_0_2.json
d0_d4_power_curve.json
eligibility_by_endpoint_horizon.json
data_boundary_audit.json
plain_report.md
technical_report.md
```

允许结论必须分开写：工程一致性、assay power、人体可估性。测试全绿不等于状态存在。
