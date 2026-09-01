# Agent A (Group-Event State v0.2, H1/H2a) — CURRENT_HANDOFF

最后更新：2026-09-01（会话进行中）

## 1. 一句话状态

固定 5 分钟时间网格上的"未来一段时间"预测装置已经建成并通过全部工程验收；
可解释多尺度基线（`B_multiscale`）已在**全部 27 位患者**上跑完；
两个循环状态生产者（`P_local` / `P_slow`）正在 GPU 上训练（162 个 job）。
**尚未产出任何 H1/H2a 科学结论。**

## 2. 代码位置

- worktree：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-group-event-state-v02-a`
- branch：`codex/topic5-group-event-state-v02-a`（base `codex/topic5-group-event-state-v0-2` @ f0c9e075）
- 新代码全部在 `src/topic5_group_event_state/v02/`；v0.1 模块只做了一处外科式修改
  （`train.py::estimate_stats` 增加可选 `positions=`，默认行为不变）。

| 模块 | 职责 |
|---|---|
| `v02/timeline.py` | 记录时间切分、carry segment（发作+发作后 60 min 断开）、5 min anchor 网格、独立窗口分母 |
| `v02/marks.py` | 四类 mark（participation / size-span / 多频带 / TRAIN 冻结的连续 repertoire 嵌入） |
| `v02/targets.py` | 前缀和 future-block target（不物化 dense 张量） |
| `v02/scoring.py` | 唯一评分入口（负二项 count / 伯努利 participation / 高斯连续量） |
| `v02/readout.py` | 嵌套增量读出、逐 family 的 ridge（TRAIN 内按时间分块 CV）、截距地板、block circular shift |
| `v02/baseline.py` | `B_multiscale` |
| `v02/producers.py` | session-preserving 训练、`P_slow` 固定时间 future heads、anchor/event 状态提取 |
| `v02/prefix.py` | H2a same-prefix continuation |
| `v02/aggregate.py` | patient-first 队列聚合 |
| `v02/registry.py` / `v02/runtime.py` | 每 producer 原子 registry 条目、租约、幂等结果 |

## 3. 正在运行的东西（**不要用 `pkill -f`**）

| 任务 | queue owner PID | 输出 | 日志 |
|---|---|---|---|
| producer 训练（162 job = 27 患者 × 2 producer × 3 seed） | **919885** | `/data/hfosp_group_event_state_v0_2/agent_a/producers/main/` | `/data/hfosp_group_event_state_v0_2/agent_a/logs/train_main.log` |
| 下游全链（等训练 manifest 出现后自动跑 registry → 嵌套评估 → H2a → A4 → 图） | **953195** | 见各阶段 tag | `/data/hfosp_group_event_state_v0_2/agent_a/logs/chain_after_training.log` |

- GPU：0 和 1，各 3 slot；单 job 实测峰值显存 0.76–2.93 GB。
- 租约：`results/epi_prssm/group_event_state/v0_2/shared/resource_leases/agent_a_train_main.json`
- STATUS：`/data/hfosp_group_event_state_v0_2/agent_a/producers/main/STATUS.json`
- 停止方式：`kill <queue owner pid>` 然后按 `job_logs/` 里记录的子进程 PID 逐个处理。
- 恢复方式：重跑同一条命令；`result.json` 的 `config_hash` 匹配即跳过。

## 4. 已完成并可复现的产出

```bash
cd /home/honglab/leijiaxin/HFOsp/.worktrees/topic5-group-event-state-v02-a
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python

# 回归测试（65 项：新增 40 + 既有 25）
$PY -m pytest tests/test_topic5_group_event_state_v02_*.py \
              tests/test_topic5_group_event_state_{contract,no_leakage,source_audit,streams}.py -q

# 全 27 人 B_multiscale（约 5 分钟，12 worker，纯 CPU）
$PY scripts/topic5_group_event_state/v02_run_future_block.py --cohort all --workers 12 --tag baseline_only

# 训练（GPU）
$PY scripts/topic5_group_event_state/v02_train_producers.py \
    --subjects <A1 优先的 27 人顺序> --producers P_local P_slow --seeds 1 2 3 \
    --gpus 0 1 --jobs-per-gpu 3 --max-epochs 24 --patience 4 --max-train-seconds 5400 --tag main
```

产出：`/data/hfosp_group_event_state_v0_2/agent_a/future_block/baseline_only/`（27 个 per-subject JSON + manifest）。

## 4.1 实测资源（决定并发的依据）

- 单 job 峰值显存 0.76–2.93 GB；每卡 3 个 slot（共 6 路）。
- 单 job 独占一卡时 `epilepsiae_1073` 每 epoch ~60 s；6 路并发时 ~100 s
  → 聚合吞吐 ≈ 3.6 倍单 job。再往上堆很可能不升反降，因此**不**按"还有显存"继续加。
- 大患者单 job 30–40 min，小患者 5–10 min；162 个 job 预计 **7–8 小时**。

## 5. 下一步（按顺序）

1. 训练结束后，用 `--state-dir .../producers/main/states/{P_local,P_slow}_seed{1,2,3}`
   重跑 `v02_run_future_block.py`，得到嵌套增量 + block-shift 零假设。
2. 跑 `v02_run_prefix.py`（H2a），输入同样的 6 个 state 目录。
3. A4 缩减诊断：event reset 1/100/1000/full、physical reset 5/30/120 min/full、
   fast-only / slow-only、memoryless、粗匹配错时 donor。
4. 承重图 + 两张辅助图 + `figures/README.md` + 目视验收
   （1–4 已由 `v02_after_training.sh` 串好，并已挂在训练完成后自动触发，PID 953195）。
5. 写 registry producer 条目；补完 plain / technical 报告的结果段。
6. 可选：`P_memoryless` seed 1（SP §4.2 敏感性臂，27 个 job）与
   `P_slow` seed 4/5（承重配置补到 5 seeds，54 个 job）。前者优先，后者按 GPU 时间决定。

## 6. 已知的、必须写进报告的限制

- **120 分钟 horizon 上每位患者的独立窗口只有 2–31 个**（5 分钟档是 45–865 个）。
  滑窗 anchor 数远大于此，不可当样本量。
- 发作 + 发作后 60 分钟排除掉了中位 **10.3%**、最多 **33.1%** 的间期事件
  （`epilepsiae_1146` 26 次发作）。这是合同要求的排除，不是数据缺陷。
- 4 位患者在 120 分钟档没有合格 anchor（记 `insufficient_coverage`，**不是阴性**）。
- `B_multiscale` 在 120 分钟 count 上有 14/23 位患者把正则化推到网格顶端，
  即"多尺度特征对两小时后的事件数没有增量"——这是基线自己的性质，要照实写。
- 4 个 (患者 × horizon × 端点) 单元里 `B_multiscale` 比截距还差，已标 `not_estimable`。

## 7. 未触碰的范围

- `/tmp/hfosp_group_event_state_v01`（v0.1 树与其结果）——只读，未修改、未停止、未复用。
- Agent B 的 `h2b/`、Agent C 的 `h3/` 结果根。
- formal / sealed 分区；paper-ready Fig1–Fig4。
- `/data/hfosp_group_event_state_v0_1/dataset`——只读复用。
