# Agent B (H2b seizure transfer) — CURRENT_HANDOFF

状态：**B0 进行中**（crosswalk + risk set + lead coverage 已完成；early ictal field 未开始）
最后更新：2026-09-01

---

## 1. 一句话现状

我们还没有任何"状态能不能预测发作"的结果——**Agent A 的 producer 一个都还没产出**。
这一轮做完的是**能不能做**这件事本身：把每一次发作对回它自己那段录音、逐次核对发作起点确实
落在那段录音里、然后数清楚"在发作前 5 分钟 / 30 分钟 / 2 小时 / 6 小时那一刻，我们手上到底有没有
一个能读状态的时刻"。答案是**有**：99 个可留出的发作事件里，75–98 个（随提前量不同）在那一刻有数据。

---

## 2. 运行环境与边界（开始时实测，不是转述）

| 项目 | 值 |
|---|---|
| worktree | `/tmp/hfosp_group_event_state_v02_b` |
| branch | `codex/topic5-group-event-state-v02-b` |
| base commit | `f0c9e0750cb59ee9691634d1c57a36487fdc421c`（`codex/topic5-group-event-state-v0-2`） |
| 主仓 dirty | 是（`codex/topic5-state-r1-5-h3-long`，与本线无关，未触碰） |
| Python | `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python` |
| 线程 | `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1` |

**开始时实测的活动队列（2026-09-01 22:04）**：

- v0.1 队列 **仍在跑**：`queue_runner.py` PID **863139**，27 subject × {b4_memoryless, b1_no_real_dt} × 3 seeds，
  GPU 0/1，8 jobs/GPU。GPU0 23.7/24 GiB、GPU1 14.9/24 GiB，两卡 util 99–100%。
  → 按工程附录 §5「GPU 已被其他队列持续高利用时不得继续堆作业」，**本线本轮不申请 GPU**。
- Topic 4 rev18 worker 亦在跑（`.worktrees/topic4-node-dualmode-rev17-continuation`）。
- 我**没有**停止、修改或复用任何上述队列的输出目录或 tag。

**Agent A registry：`not_available`**（`results/epi_prssm/group_event_state/v0_2/` 在主仓尚不存在）。
registry 搜索顺序（两处都查，先到先用，并记录命中位置）：

1. `/home/honglab/leijiaxin/HFOsp/results/epi_prssm/group_event_state/v0_2/shared/checkpoint_registry.json`
   （主仓 = v0.1 `MAIN_TREE` 约定，三个 agent 各自 worktree 都能看见）
2. `/tmp/hfosp_group_event_state_v02/results/.../shared/checkpoint_registry.json`（A 的 worktree，兜底）

## 3. 目录约定

| 用途 | 路径 |
|---|---|
| 共享（registry / lease，跨 agent 可见） | 主仓 `results/epi_prssm/group_event_state/v0_2/shared/` |
| 本线交付物（索引、小统计、报告、图） | 本 worktree `results/epi_prssm/group_event_state/v0_2/h2b/` |
| 大产物（risk sets、field、预测） | `/data/hfosp_group_event_state_v0_2/agent_b/` |

lease 已原子写入 `shared/resource_leases/agent_b.json`。**未** `pkill -f`，**未**抢占 A/C 租约。

---

## 4. 已完成

### B0.1 crosswalk（`support/seizure_crosswalk.{csv,json}`）

按 **recording code** 连接，不用 subject 字符串 inner join。逐 onset 核对其是否落在该录音自己的
block 时间跨度内。

- Epilepsiae 542 行 → matched **230**、recording_absent 26、incomplete_interval 5、
  subject_not_in_dataset 281（14 个不在本 27 人队列里的病人）。
- Yuquan 54 行 → matched **44**、subject_not_in_dataset 10（huanghanwen 2 + litengsheng 8，两人不在队列）。
- **零 `onset_outside_recording`、零歧义、零重复 id** —— 即"逐发作零误差审计"通过。
- 5 位 Yuquan 病人 0 条发作记录（chengshuai / hanyuxuan / huangwanling / zhangjiaqi / zhourongxuan）。
  按 v0.1 数据合同 §11，这只能读作**未检出**，不可读作**无发作**。

### B0.2 risk set + 发作边界（`/data/.../risk_sets/<subject>.csv`）

每 5 分钟一个固定时刻锚点，落在**状态覆盖**内；发作区间与其后 60 分钟不出锚点（30/120 分钟为敏感性）。
**两种覆盖分开**：锚点用「状态覆盖」（真正进了数据集的 block），结局是否可信用「监测覆盖」
（Epilepsiae 用 SQL block 表、Yuquan 用 EDF block）。监测断掉 → 该行 **right-censored**，
不会被当成"六小时内没发作"。

- 27,982 行；7,511 行在 6h 内确有发作；7,640 行删失；12,831 行确认超出 6h。

### B0.2b 发作成簇（重要，改变分母）

`yuquan_zhangjinhan` 的 **8 次"发作"全部落在同一个 block 内 3.6 分钟之内、时长全为 0.0 s** ——
这是检测器把**一次发作切成了 8 条**，不是 8 次独立发作。`sunyuanxin` 等也有成簇。
因此按「后一次发作若落在前一次的 60 分钟不应期内，则同属一个 episode」归并：

- 274 次发作 → **209 个 episode** → 留出 **99 个 episode**（滚动起点：每人前 ⌈n/2⌉ 个 episode 进 TRAIN）。
- 只有 episode 的**首发**才可能被预测（后续那几次的前置锚点必然落在自己 episode 的不应期里）。
- **15 条发作时长恰为 0.0 s**（Yuquan 13 条 + 部分），inventory 却标 `has_complete_eeg_interval=True` →
  已打标，offset 不可信，需在 B3 做时长敏感性。

### B0.2c 各提前量的可用锚点（决定 assay 能不能做）

| 提前量 | 有锚点的留出 episode | 病人数 |
|---|---:|---:|
| 5 min | 98 / 99 | 19 |
| 30 min | 89 / 99 | 19 |
| 2 h | 79 / 99 | 18 |
| 6 h | 75 / 99 | 19 |

→ **B1 与 B2 在工程上都可估计**，6h 档也不必靠补零。

---

## 5. 代码与测试

- `src/topic5_h2b_transfer/crosswalk.py` + `tests/test_topic5_h2b_crosswalk.py`（13 项）
- `src/topic5_h2b_transfer/risk_grid.py` + `tests/test_topic5_h2b_risk_grid.py`（21 项）
- `scripts/topic5_h2b_transfer/{build_seizure_crosswalk,build_risk_sets}.py`
- 全部 **34 passed**。每项测试对应合同里的一条 clause（见模块 docstring 的 C1–C8 / D1–D9）。

**刻意没有复用** `src/topic5_group_event_state/source_audit.py::seizure_index`：它按
`(dataset, subject)` 字符串建索引，回答的是"这个事件是不是发作期"（间期剔除），
与"这次发作属于哪一段有覆盖的录音"不是同一个问题（CLAUDE.md §6.1）。

---

## 6. 下一步（按顺序）

1. **B0.3 early ictal field**：发作后前 5 s（10 s 敏感性）逐触点归一化能量/募集场。
   决定：**不直接复用** `results/topic5_ictal_recruitment/ictal_field_long_cache` ——
   它只覆盖 27 人里的 13 人（Yuquan 只有 2 人），且 Epilepsiae 侧锚在**临床起始**而非 EEG 起始
   （topic5 caveat 9 要求按 EEG 起始锚定）。改为用已验证的原语重建，并把该 cache 当**对拍参照**。
2. B0.4 冻结 TRAIN-only route / field normalization。
3. B1 `plumbing_only`：用 v0.1 trajectory 打通 survival/field/censoring/schema（**绝不**写成 v0.2 人体结果）。
4. B2 读取 registry 全部 producer（缺失记 `not_available`，不 fallback）。

## 7. 复现命令

```bash
cd /tmp/hfosp_group_event_state_v02_b
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
P=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
$P -m pytest tests/test_topic5_h2b_crosswalk.py tests/test_topic5_h2b_risk_grid.py -q
$P scripts/topic5_h2b_transfer/build_seizure_crosswalk.py
$P scripts/topic5_h2b_transfer/build_risk_sets.py
```

## 8. 未触碰范围

formal/sealed 分区、paper-ready Fig1–Fig4、Agent A 的 producer 代码与 registry 条目、
`/tmp/hfosp_group_event_state_v01` 队列及其输出。
