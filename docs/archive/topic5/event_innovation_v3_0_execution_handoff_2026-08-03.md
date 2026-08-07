# Topic 5 V3.0 跨事件 innovation 分析执行 handoff（2026-08-03）

## 1. 任务目标与不可偏离边界

当前主线把**一整场间期事件作为一个时间步**。V3.0 检验：在用过去事件估计当前患者特异 rank/precedence state 后，一场完整事件中无法被过去预测的 rank-field innovation，是否与随后状态变化及多事件累计位移有关。

禁止把任务改回：

- 同一事件内部的 next-rank prediction；
- GRU/Transformer/更多 hidden dimension 的 architecture sweep；
- 从 raw latent weights 解释 contact graph；
- 用 SNN、ictal、SOZ、geometry、A/B axis 或 old heldout20 作为输入或 Gate；
- 用 V3.0 test 结果反开 V3.1 human transition 分支；
- 把预测关联写成 activity-dependent shaping 或 causal plasticity。

## 2. 已完成并冻结的内容

### V2.7 前置验收

- formal 34 patients x 3 seeds、dense、state reset、memory curve、block shuffle、time reversal、H40 均已完成。
- 验收产物：`results/topic5_stateful_event_sequence_rnn/v2_7/acceptance/ACCEPTANCE_STATE.json`。
- 冻结判决：`ACCEPTED_REPAIR_ONLY_STATE_TRACKING_FINAL`。支持短期 within-recording state tracking；EWMA 足以解释主要增量；不支持 chronology-specific shaping。

### V3.0 Phase 0–4

- Phase 0：34/34 完成，old heldout20 全部排除。
- measurement：validation-only 合同完成；早期误读 test reliability 的产物已隔离，不可使用。
- innovation validity：34 人中 17 人 rank innovation 有效；其余为 history support 不足或 residual 仍可被过去预测。
- Goal 2 local validation：`NOT_OPEN`。
- Goal 3 cumulative validation：`NOT_OPEN`。
- V3.0 synthetic：完成。
- V3.1 matched-transition synthetic：完成。
- 预 test handoff：`results/topic5_event_innovation_impulse_response/v3_0/V3_1_HANDOFF_STATE.json`，状态 `NOT_TRIGGERED`。
- 结论：**V3.1 human recurrent-transition 不执行，也不做 capacity rescue。**这是一项已完成的预注册停止判决，不是剩余工作。

### V3.0 human test release

- `results/topic5_event_innovation_impulse_response/v3_0/HUMAN_TEST_RELEASE_STATE.json` 状态为 `HUMAN_TEST_RELEASED`。
- release 后不得改动以下冻结主分析实现再声称同一次 test：
  - `scripts/run_topic5_event_innovation_v3_0_human_test.py`
  - `src/topic5_event_innovation_test_v3_0.py`
  - `config/topic5_event_innovation_v3_0.yaml`
  - V3.0 frozen spec/plan。
- 汇总验收规则已在任一 route aggregate 产生前写入
  `results/topic5_event_innovation_impulse_response/v3_0/human_exploratory/ACCEPTANCE_RULE_STATE.json`。
- **不要修改** `scripts/accept_topic5_event_innovation_v3_0.py` 后再运行验收；规则文件锁定了它的 SHA256。

## 3. 当前实时状态

### 已完成的人类 test route

Local route 已完成 34/34 并汇总：

- 产物：`results/topic5_event_innovation_impulse_response/v3_0/human_exploratory/local/LOCAL_TEST_STATE.json`
- eligible：17/34；
- propagation gain median = `-0.0007759218`，7/17 positive，two-sided Wilcoxon `p=0.328949`；
- true-minus-state-matched median = `+0.0000073893`，10/17 positive，`p=0.430679`；
- future-minus-past median = `+0.0001214024`，12/17 positive，`p=0.071411`。

因此 Goal 2 human test 不支持 Level 2。不能只取 future-minus-past 的正方向包装阳性。

### 已完成的 dense moving-block sensitivity

独立 sensitivity runner 没有改冻结主 runner：

- `scripts/run_topic5_event_innovation_v3_0_dense_bootstrap.py`
- `src/topic5_event_innovation_bootstrap_v3_0.py`

两条 route 均已 34/34 完成和汇总：

- local dense median = `-0.0006950206`，6/17 positive，Wilcoxon `p=0.159378`；
- cumulative dense median = `+0.0046887388`，11/17 positive，Wilcoxon `p=0.063828`。

这些是 overlapping dense-anchor sensitivity，不替代非重叠 primary，也不能反开 V3.1。

### 仍在运行的唯一主分析

Cumulative human primary 已完成 32/34。当前仍有两个单线程进程：

- `epilepsiae_1096`
- `epilepsiae_1073`

交接时它们的 PID 分别为 `2246446`、`2246448`，但新 agent 应以命令实时查询，不依赖旧 PID：

```bash
pgrep -af '^/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python scripts/run_topic5_event_innovation_v3_0_human_test.py'
find results/topic5_event_innovation_impulse_response/v3_0/human_exploratory/cumulative/per_subject -name '*.json' | wc -l
```

不要在这两个进程仍运行时重复启动相同患者。实测每个进程 RSS 约 0.38 GiB，CPU 100%，不是 OOM 或挂死。

## 4. 剩余执行步骤

### Step 1：收完 cumulative 34/34

等待上述两个进程退出并确认 artifact count = 34。如果进程已经消失但 artifact 缺失，只重跑缺失患者：

```bash
CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MALLOC_ARENA_MAX=2 \
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
scripts/run_topic5_event_innovation_v3_0_human_test.py \
  --kind cumulative --phase patients \
  --config config/topic5_event_innovation_v3_0.yaml \
  --subjects <missing_subjects>
```

### Step 2：只运行一次 cumulative aggregate

```bash
CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
scripts/run_topic5_event_innovation_v3_0_human_test.py \
  --kind cumulative --phase aggregate \
  --config config/topic5_event_innovation_v3_0.yaml
```

必须生成：

`results/topic5_event_innovation_impulse_response/v3_0/human_exploratory/cumulative/CUMULATIVE_TEST_STATE.json`

### Step 3：运行冻结 acceptance

不要修改 acceptance script。直接运行：

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
scripts/accept_topic5_event_innovation_v3_0.py
```

必须生成：

- `human_exploratory/HUMAN_EXPLORATORY_STATE.json`
- `human_exploratory/cohort_inference.json`
- `human_exploratory/evidence_level.json`
- `human_exploratory/patient_summary.csv`

冻结 evidence rule：

- 任一 route 的三个预注册 cohort median 全为正，且该 route 主增益的患者级双侧 Wilcoxon `p<=0.05`，才到 Level 2；
- 否则，因 V2.7 state tracking 已验收，落在 Level 1；
- human test 结果无论如何都不能把 pre-test `NOT_TRIGGERED` 改成 `OPEN`。

### Step 4：整体工程验收

```bash
CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest -q \
  $(rg --files tests | rg 'topic5_(event_innovation|stateful_event_rnn_v2_7)' | tr '\n' ' ')

git diff --check -- \
  scripts/accept_topic5_event_innovation_v3_0.py \
  scripts/run_topic5_event_innovation_v3_0_dense_bootstrap.py \
  src/topic5_event_innovation_bootstrap_v3_0.py \
  tests/test_accept_topic5_event_innovation_v3_0.py \
  tests/test_topic5_event_innovation_bootstrap_v3_0.py
```

当前最近一次 scoped suite 是 101 passed；bootstrap 新增测试单独是 3 passed。最终应合并重跑。

### Step 5：归档最终科学报告

新建中文 acceptance report，至少写清：

1. V2.7 已完成，不是未验收前置；
2. V3.0 有效 innovation 仅 17/34，Yuquan 仅 2 人；
3. local 和 cumulative primary 的连续 effect、CI、favorable count、Wilcoxon、sign test；
4. dense moving-block sensitivity 只作敏感性；
5. 最终 evidence level 和允许/禁止 wording；
6. V3.1 human `NOT_TRIGGERED` 是预 test 决定；
7. SNN 与这条 RNN 线独立。

建议路径：

`docs/archive/topic5/event_innovation_v3_0_acceptance_2026-08-03.md`

并在 `docs/archive/topic5/INDEX.md` 增加入口。不要回改冻结 spec/plan 的正文或 release 哈希对象；用 acceptance report 记录执行结果。

## 5. 最终验收条件

只有以下全部满足才算当前 goal 闭环：

- cumulative 34/34 artifact + aggregate 完成；
- frozen acceptance 四个产物存在且状态一致；
- V3.1 handoff 仍为 `NOT_TRIGGERED`；
- dense sensitivity 两 route 均 34/34；
- scoped tests 全绿；
- `git diff --check` 干净；
- 中文 acceptance report 与 INDEX 完成；
- 没有把 rank-event step 改回 within-event rank step；
- 没有把 association、observer 或 dense trend 升级成 shaping。

仓库是 dirty worktree，包含大量用户已有改动。不要清理、reset、checkout 或提交无关文件；除非用户另行要求，不要 commit。
