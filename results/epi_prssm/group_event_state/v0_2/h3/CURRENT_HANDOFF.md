# Agent C (H3 event feedback) — CURRENT_HANDOFF

最后更新：2026-09-02（已收官）

## 0. 一句话

判断间期群体事件到底只是慢状态的**读数**，还是它的**数量/内容**还会反过来进入
之后的状态演化。做法是训练三条只差一条边的模型，比它们对**没见过的未来时间块**的预测。

## 1. 工作区与分支

| 项 | 值 |
|---|---|
| worktree | `/tmp/hfosp_group_event_state_v02_c` |
| branch | `codex/topic5-group-event-state-v02-c` |
| base commit | `f0c9e0750cb59ee9691634d1c57a36487fdc421c`（`codex/topic5-group-event-state-v0-2`） |
| 独占结果根 | `results/epi_prssm/group_event_state/v0_2/h3/` |
| 大文件根 | `/data/hfosp_group_event_state_v0_2/agent_c/` |
| Python | `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python` |
| 线程环境 | `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1` |

**未触碰**：A 的 core（`src/topic5_group_event_state/`）、shared registry 主文件、
`/tmp/hfosp_group_event_state_v01` 队列与结果、formal/sealed 分区、paper-ready Fig1–Fig4。

## 2. 代码

```
src/topic5_group_event_state_h3/
  support.py     coverage segment / seizure+postictal cut / physical-time split / disjoint block tiling
  background.py  30 s 背景 anchor 表 + 5 min cell 池化 + clock 协变量
  features.py    冻结的 count / mark 事件词表；稀疏 future-block target
  stream.py      读 v0.1 冻结事件流（只复用 memmap，不复用其事件计数切分）
  timeline.py    cell / anchor / event 合并时间轴（同刻顺序 cell → anchor → event）
  models.py      M0 / M1 / M2（唯一差别 = 那条边）+ 精确闭式 rollout + future-block decoder
  train.py       共享训练协议、TBPTT、checkpoint 选择
  runtime.py     患者 → 张量的唯一装配路径（三个下游阶段共用）
  perturb.py     real / no-feedback / state-matched mark replacement (+2 secondary)
  impulse.py     逐事件 signed impulse response（闭式）
  innovation.py  C1 functional innovation trajectory
  analysis.py    block → patient → cohort 聚合与配对统计
  registry.py    读 A 的 registry；缺失即 not_available，不 fallback
  synthetic.py   已知反馈边的合成患者（仅校准仪器）
  io.py          原子写、payload hash、幂等 resume
scripts/topic5_group_event_state_h3/
  build_support.py  build_background.py  build_features.py
  run_h3_models.py  run_perturbation.py  run_impulse.py  run_innovation.py
  run_synthetic.py  queue_runner.py  aggregate_h3.py  make_figures.py
tests/test_topic5_group_event_state_h3.py   (23 tests)
```

## 3. 复现命令（按顺序）

```bash
cd /tmp/hfosp_group_event_state_v02_c
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python

# C0：支持度 / 背景 / 事件词表（CPU）
$PY scripts/topic5_group_event_state_h3/build_support.py
$PY scripts/topic5_group_event_state_h3/build_support.py --postictal-exclusion-s 0 --tag postictal0
$PY scripts/topic5_group_event_state_h3/build_background.py --workers 8
$PY scripts/topic5_group_event_state_h3/build_features.py --workers 6
$PY -m pytest tests/test_topic5_group_event_state_h3.py -q

# C2：仪器可识别性（合成，不作为人体分析的 gate）
$PY scripts/topic5_group_event_state_h3/run_synthetic.py --device cuda:0 --lr 3e-4 --max-epochs 60

# C3：人体主比较（27 患者 x 3 臂 x 3 seed）
SUBJ="$($PY - <<'P'
import json
d=json.load(open('results/epi_prssm/group_event_state/v0_2/h3/support/coverage_support_primary.json'))
print(' '.join(r['subject'] for r in sorted(d['subjects'], key=lambda r:-r['usable_hours_after_seizure_cuts'])))
P
)"
$PY scripts/topic5_group_event_state_h3/queue_runner.py --stage models --subjects $SUBJ \
    --seeds 0 1 2 --tag main --gpus 0 1 --slots-per-gpu 3 --lr 3e-4 --max-epochs 60 --max-train-seconds 3600
$PY scripts/topic5_group_event_state_h3/aggregate_h3.py --tag main

# C4：冻结模型上的最小扰动 + signed impulse response + C1 innovation
$PY scripts/topic5_group_event_state_h3/queue_runner.py --stage impulse      --subjects $SUBJ --seeds 0 1 2 --tag main --gpus 0 1 --slots-per-gpu 3
$PY scripts/topic5_group_event_state_h3/queue_runner.py --stage perturbation --subjects $SUBJ --seeds 0 1 2 --tag main --gpus 0 1 --slots-per-gpu 3 --extra --include-secondary
$PY scripts/topic5_group_event_state_h3/queue_runner.py --stage innovation   --subjects $SUBJ --seeds 0 1 2 --tag main --gpus 0 1 --slots-per-gpu 3

$PY scripts/topic5_group_event_state_h3/make_figures.py --tag main
```

## 3b. 本轮结论（一句话）

**H3 = instrument/data not estimable，既非阳性也非阴性。**
18 条预注册对比全部落在同档"重训噪声地板"以内（最大 0.61×）；
拟合出的事件边对读出的影响 27/27 位患者 < 1%（24/27 < 0.1%，队列中位 0.0006%）；
合成标定显示当反馈边真实占潜变量方差 47%–83% 时，本 assay 测出的增量比零真值那格还小。
详见 `reports/technical_report.md` §7–§8 与 `reports/plain_language_report.md` §4。

**已跑完**：486 个模型（27 患者 × 3 臂 × 6 seed，0 失败）、
impulse / perturbation / innovation 各 81 个、合成标定 ladder2、
32,058 个块的边界暴力核对（27/27 干净）、29 项回归测试。

## 4. 状态

见同目录 `STATUS.json`（队列所有者 PID/PGID、pending/running/failed 计数、心跳）与
`results/epi_prssm/group_event_state/v0_2/shared/resource_leases/agent_c.json`。

进程管理只用队列记录的精确 PID/PGID；**禁止 `pkill -f`**。

## 5. 已知边界（接手前必读）

- **120 分钟这一档只有 8/27 位患者有 ≥6 个不重叠的留出块；6 小时档 0/27**。
  这是 C0 在任何模型跑之前就定下的（`support/coverage_support_primary.json`），
  不是事后发现的。5 分钟 27/27、30 分钟 26/27。
- **背景观测的新鲜度与事件率反相关（已实测，比担心的小）**：2 秒背景窗只要与事件核重叠
  就被丢弃，所以事件密的患者每小时留下的干净背景窗少得多
  （13.3/h vs 稀疏患者 148.9/h，理论上限 120/h）。
  但因为每个 5 分钟 cell 允许取"该 cell 之前最近的一个 anchor"，
  **有效率仍是 0.965–1.000，没有患者的共同驱动臂真的缺输入**；
  代价只是密集患者的背景更"陈旧"。事件率与 cell 有效率的秩相关只有 −0.364。
  数值见 `support/background_coverage.json`，结论时必须与效应并列。
- **registry 绑定的 C1 轨迹目前是 `not_available`**：A 尚未发布任何 producer。
  C 自己冻结模型上的 innovation 写在**另一个 key** 下并标 `diagnostic_only_not_registry_bound`，
  不得当作 registry-bound 结果引用。
- 最高允许措辞：**event-feedback-like predictive dependence**。不得写成 IED 因果改变脑网络。
