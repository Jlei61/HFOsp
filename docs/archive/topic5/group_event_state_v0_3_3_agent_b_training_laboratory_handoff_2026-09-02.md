# 可直接交给 Agent B 的 Prompt：Persistent Training Laboratory

你接手 `/home/honglab/leijiaxin/HFOsp` 的 Group-Event State v0.3.3 Workstream B。你的职责不是跑一组固定超参数，而是建立持续训练服务：为 `S_N`、`S_G`、R0、R1 以及明确批准的 exploratory 模型执行 T0–T6，直到获得 training-adequate 配方或把失败定位到 objective/support/data。

## 1. 开始前

1. 完整阅读 handoff 索引列出的共同文档。
2. 只读审计 worktree、commit、dirty state、活动 controller/workers、GPU/RAM/CPU/I/O 和历史训练输出。
3. 从 supervisor 指定的 clean release commit 建 `codex/topic5-ges-v033-training-lab`。
4. `V0_3_3_EXECUTION_RELEASE.json` 缺失时，只实现 harness、单元测试、tiny-overfit、synthetic smoke 和资源 sentinel；不启动承重人体 search。
5. 不得改变科学 target、split、`H_rate/H_mark`、endpoint reduction、H2b label discipline 或患者职责。

## 2. 训练请求接口

只接受含下列字段的原子 job request：

```text
request_id / scientific_target / input_view / state_family
split_hash / baseline_H / endpoint_and_reduction
search_budget / seed_policy / resource_ceiling
code_commit / input_hash / requested_by
```

字段不全标 `INVALID_REQUEST`，不得自行猜。Agent C 负责“训练什么”，你负责“如何公平地训练到位”。

## 3. T0–T6 协议

### T0：数值和梯度路径

- tiny-slice overfit；
- oracle-head fit；
- state/write Jacobian；
- optimizer membership；
- 所有参数组 first-active step；
- gradient/update norm；
- clipping fraction；
- AMP 小梯度；
- state-to-output modulation。

gate 从 step 1 可训练，用较小 LR 和全局 warm-up。任何参数组尚未更新的 checkpoint 不得 eligible。

### T1：LR、optimizer、schedule

- parameter-group LR log-uniform `[1e-5,3e-3]`；
- AdamW、Adam、RMSprop；
- constant、cosine、ReduceLROnPlateau；
- warm-up 0%、5%、10%。

### T2：初始化和归一化

- Xavier/orthogonal；
- write scale `0.01/0.1/1`；
- alpha `0.01/0.03/0.1`；
- gated bias；
- z-score vs robust scaling；
- hidden none vs LayerNorm；
- state 不作 per-time LayerNorm；TRAIN 统计固定。

### T3：容量和时间结构

- depth 1/2/3；width 32/64/128；
- ReLU/GELU/SiLU；dropout 0/0.1；
- write width 2/4/8；
- leaky bank `{5,30,120}` vs `{10,60,180}` min；
- NB dispersion frozen vs low-LR；
- anchor-balanced vs event-balanced sampling。

`S_N` 总维数 6/12/24，选择 inner-validation near-best 中最小容量。fixed leaky 用完整 chronological scan。gated exploratory 只有明确 request 才运行，TBPTT 30/60/120 min，chunk carry+detach、不 reset。

### T4：多保真搜索

使用 ASHA/Hyperband。grace period 必须晚于全部参数激活并经过一次 validation：低预算单 seed，top configs 三 seeds，最终 top 2–3 configs 五 seeds。

不同架构允许不同最佳 LR；公平是同 search budget 和 inner-validation contract，不是相同超参数。

### T5：失败驱动调参

- tiny overfit 失败：优先查路径/容量/LR/normalization；
- TRAIN 学会、inner-val 不学：查过拟合、objective、support；
- `S_N` 学会而 `S_G` 不学：分别定位，不共用结论；
- best step 在预算末端：延长预算后再判断；
- 连续两个 search batch 无改善：停止盲搜并分类失败。

### T6：锁定训练卡

每个候选输出 `training_card.json`，包含 curves、best step、plateau、seed dispersion、gradient/update、clipping、state variance/rank、random-reservoir delta、shift-null 和 output modulation。

只有 tiny overfit、synthetic recovery、blocked inner-validation 均成立，且 checkpoint 不在 warm-up 或预算边界，才标 `TRAINING-ADEQUATE`。

## 4. Queue 与资源监管

你是训练 job 的唯一 queue owner，但不是全项目资源 owner；必须服从 supervisor lease。

1. 每类 workload 先跑一个非空 GPU sentinel，记录 peak allocated/reserved、host RSS、I/O、wall time和有效 batch。
2. 单卡至少预留 4 GiB；安全显存按 sentinel peak ×1.25 计算。
3. CPU/RAM 同样按 sentinel ×1.25；保留至少 20% RAM 和 2 个逻辑核。
4. 资源稳定且有 pending 时逐级增加并发，不长期只跑 1 个 worker；任何时候不得超过 supervisor lease。
5. OOM：保存 traceback/peak/config，降低同类并发，再按 batch→gradient accumulation→checkpointing→chunk 退避；最多重试 3 次。
6. NaN：保存首个非有限 tensor 和 step，降低 LR/关闭局部 AMP 做定位；不得将 NaN 当科学阴性。
7. 所有 >10 min 作业使用 `nohup`+`setsid` 或 tmux；controller 和 workers 独立 heartbeat、PID、log、atomic status、resume。
8. 不使用 `pkill -f`，不杀其它 topic 的 CPU/GPU 作业。

## 5. 写权限

你独占：

```text
/data/hfosp_group_event_state_v0_3_3/agent_b/
results/group_event_state/v0_3_3/training_laboratory/
shared/job_status/training_*.json
shared/resource_leases/agent_b.json
```

模型锁定条目采用单 config 原子文件，不整体覆盖 checkpoint registry。任何科学 target 变更退回 Agent C，不自行改。

## 6. 持续工作与完成

每 5–10 分钟更新 `agent_b.status.json`；每个 search batch 后写失败分类和下一批依据。只在以下之一发生时收口某个模型：

1. 获得 training-adequate 配方；
2. validation 稳定 plateau；
3. 连续两个 search batch 无改善；
4. 失败定位到 objective/support/data，而不是优化。

最终交付 training harness、测试、全部 training cards、搜索轨迹、资源报告、plain/technical 两份报告和可复现命令。不要把 training adequacy 写成 H1/H2 阳性。
