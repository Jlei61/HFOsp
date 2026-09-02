# 主 Task Runbook：v0.3.3 训练监管与空闲 Worker 补投

主 task 不直接占据 Agent A/B/C 的科学职责。它持续审阅三条工作流的 manifests、训练卡和资源，确保机器有安全余量时继续推进预先批准的实验，直到用户回来或 Core DoD 完成。

## 1. 监管目标

1. 监控的不只是 PID，而是科学有效性：评分是否 canonical、训练是否越过 warm-up、分母是否可估、结果身份是否正确。
2. 保证 GPU/CPU/RAM/I/O 有安全余量时，approved queue 不长期空转。
3. 遇到 OOM/NaN/stale job 时精确恢复，不删除患者、不覆盖输出、不杀其它项目。
4. 防止 Agent A/B/C 越权：A 不调模型；B 不改 target；C 不私调 optimizer；H2b 不回选 state。

## 2. 每轮监控顺序

### 2.1 只读现场

```text
git worktree/branch/commit/dirty
agent status + heartbeat
job requests/status/leases
controller PID/PGID/log freshness
GPU utilization/free VRAM/processes
CPU load/runnable workers
MemAvailable/swap/iowait
free disk and output growth
sealed flag
```

必须识别其它 topic 的活跃作业；它们不属于本任务，不能停止或抢占。

### 2.2 科学审阅

每个新结果先问：

1. 哪个 exact config/hash/split？
2. evaluator 是否 canonical？
3. training card 是否 training-adequate？
4. 分母是患者/独立 block/held-out seizure，还是滑动行数？
5. 这是 within-view、cross-transfer、sharedness 还是 H2b？
6. 允许的最高结论是什么？

发现同一对象分数不一致、泄漏或 sealed violation，立即停止受影响队列并保留证据；其它队列继续。

## 3. 空闲资源时的补投优先级

只补投已经存在于 approved manifest 的 job，不临时发明新科学问题：

1. 恢复 `STALE/FAILED_RETRYABLE/OOM_RETRYABLE`；
2. Agent A 的 canonical/synthetic smoke 与缺失 D0–D4 replicates；
3. Agent B 已批准 training requests 的下一 ASHA rung、缺失 seed 或预算延长；
4. Agent C 依赖已满足的 R0/R1、within-view、cross-transfer、shared/private、H1/H2a；
5. frozen H2b risk；
6. 预登记的 R2/D5/small-shared exploratory。

以下永不自动补投：sealed/formal、人体 H3、未批准新 endpoint、看 dev 结果后新增的 horizon/患者/seed、early-field 大搜索、paper-ready Fig1–Fig4。

## 4. 动态 worker 规则

### 4.1 Sentinel

每种 workload class 先有非空 sentinel：CPU synthetic、CPU analysis、GPU training、GPU probe。记录 peak RSS/VRAM、wall time、threads、I/O 和临时磁盘。

### 4.2 安全上限

- worker 安全需求 = sentinel peak ×1.25；
- RAM reserve = `max(20% total RAM, 20 GiB)`；
- GPU reserve = 每卡至少 4 GiB；
- CPU reserve = 至少 2 logical cores，并考虑其它 topic 当前 load；
- disk free <10 GiB 时不启动新 job；
- 持续高 iowait/swap 增长时停止补投，让 active job 自然完成。

并发上限取 pending、CPU、RAM、GPU、I/O、config ceiling 和已有 leases 的最小值。稳定两个 heartbeat 周期且有余量时每次增加一个 slot；接近 reserve 时停止补投，不因瞬时 spike 杀 active job。

### 4.3 当前现场不是永久合同

交接时两张 RTX 3090 可能空闲，Topic 4 可能正在占用多核 CPU。每轮必须重测；不得把交接快照写死成资源事实。GPU 空闲但 CPU/I/O 饱和时，只能启动 data-loader 受限的 GPU job，不能盲目填满。

## 5. OOM、NaN 与 stale

### OOM

1. 保存 traceback、peak、batch/chunk/concurrency；
2. 标 `OOM_RETRYABLE`；
3. 同类并发减一；
4. 依次降低 batch、加 gradient accumulation、启 checkpointing、缩 chunk；
5. 最多 3 次；仍失败标 `RESOURCE_UNRESOLVED`，其它 jobs 继续。

### NaN

保存首个非有限 step/tensor/optimizer state；先关闭局部 AMP、降低相应参数组 LR做诊断。不得直接重启到同一配置，也不得称科学阴性。

### Stale

status=RUNNING 但 PID 不存在且 heartbeat 超时：保留旧 log/status，核对 atomic output 后只恢复未完成 job。不得以文件存在即 COMPLETE。

## 6. 持久运行

- controller 使用唯一 tmux session 或 `nohup`+`setsid`；stdin `/dev/null`；绝对路径与固定 Python。
- 每 60 秒写 heartbeat；前 20 分钟每 2–5 分钟人工/agent 审阅，稳定后每 10–20 分钟，不超过 30 分钟。
- 所有线程环境设 1；单一 queue owner；job key 包含 target/input/state/patient/seed/split/config/code/input hash。
- status 原子写入：`PENDING/RUNNING/COMPLETE/FAILED/OOM/NAN/INVALID/SKIPPED_EXISTING`。
- 网络断开不影响本地运行；恢复后先读 manifest，不从头重跑。

## 7. Supervisor 状态页

持续维护：

```text
/data/hfosp_group_event_state_v0_3_3/shared/SUPERVISOR_STATUS.json
/data/hfosp_group_event_state_v0_3_3/shared/SUPERVISOR_LOG.md
results/group_event_state/v0_3_3/CURRENT_HANDOFF.md
```

状态页至少包含：时间、commit、sealed、三 agent heartbeat、planned/running/complete/failed/OOM、GPU/RAM/CPU/I/O、当前 leases、最近科学审阅、下一批补投依据。

## 8. 何时暂停并等待用户

1. 需要打开 sealed/formal；
2. 需要改变核心 target、患者职责或 H2b 冻结纪律；
3. 同一泄漏/评分矛盾重复三轮仍无法修复；
4. 需要删除、覆盖或不可逆改变数据；
5. 需要外部权限、缺失挂载或临床解释。

资源忙、单个 OOM、一个模型阴性、某患者不可估都不是全局暂停理由。

## 9. 用户回来时的报告

先给白话版：

```text
这段时间真正学会了什么？
哪把尺子已可信？
哪些只是训练问题？
H1/H2a/H2b 各有什么新证据？
最大的剩余不确定性是什么？
```

再给技术版：所有 job、hash、训练卡、资源、错误恢复、逐患者/seed、分母、统计、图和复现命令。不得用“跑了多少 job”代替科学进展。
