# Agent B (H2b seizure transfer) — CURRENT_HANDOFF

状态：**阶段性验收已封存（历史版本），worktree 已关闭**
机器状态：`STAGE_CLOSED_ENGINEERING_ONLY__H2B_NOT_ESTABLISHED`
最后更新：2026-09-02

> **封存说明**：本阶段只验收**工程层与分母层**；一切涉及"冻结状态能否预测发作"的科学陈述
> **均已撤回**（原因见 `technical_report.md` §0 与阶段封存归档）。
> 归档：`docs/archive/topic5/group_event_state_v0_2_h2b_stage_closeout_2026-09-02.md`
> 分支 `codex/topic5-group-event-state-v02-b` 保留为历史记录，**未合并 main**。
> worktree `/tmp/hfosp_group_event_state_v02_b` 已移除；
> 大产物仍在 `/data/hfosp_group_event_state_v0_2/agent_b/`（22 MB）。
> 恢复方式：`git worktree add <path> codex/topic5-group-event-state-v02-b`

---

## 1. 一句话现状

**只验收工程层与分母层。** 一切"冻结状态能否预测发作"的科学陈述已撤回：
读的是 raw latent 而非合同 `S_func`；`P_local`/`P_slow` 的 cell 跨 27–29 种配置不可采纳；
`B_multiscale`、recent IED、block circular shift 三条对照未执行。
B1 判为 assay not estimable。详见 `technical_report.md` §0。

---

## 2. 环境与边界（实测，非转述）

| 项目 | 值 |
|---|---|
| worktree | `/tmp/hfosp_group_event_state_v02_b` |
| branch | `codex/topic5-group-event-state-v02-b`（base `f0c9e075`） |
| Python | `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`；线程全设 1 |
| 测试 | **105 passed**；改动**纯增量**（只有新增文件，0 个既有文件被改） |

**GPU：本线全程未申请。** 开工时 v0.1 队列占满两卡（99–100%），按工程附录不叠加；
该队列已于 22:33:44 自然跑完（`ALL QUEUES DONE`, 162/162），**不是被我停的**。
之后 Agent C 的 H3 队列接管 GPU。本线所有计算都是 CPU（8 worker，峰值 ~4.8 GB/worker）。

**Agent A registry**：`results/epi_prssm/group_event_state/v0_2/shared/checkpoint_registry.json`

| producer | status | B 侧可用性 |
|---|---|---|
| `B_multiscale` | complete (27) | ❌ 无逐时刻特征；且 `log_time_to_nearest_seizure` **含未来标签泄漏**，不得照搬导出 |
| `P_local` | complete (27) | ❌ **不可采纳**——cell 跨 **29** 种配置（单 seed 内 10 种） |
| `P_slow` | complete (27) | ❌ **不可采纳**——cell 跨 **27** 种配置（单 seed 内 9 种） |
| `P_memoryless` | complete (27) | ⚠️ 配置同质、可读，但只是对照，单独无科学意义 |

registry loader 已加**默认开启**的 fail-closed 校验（producer 内部配置同质性 + checkpoint 哈希）。
另：registry 声明的 `source_commit` 比其 artifact **晚 5 小时**，不可能是产出它们的代码。

---

## 3. 目录

| 用途 | 路径 |
|---|---|
| 共享（registry / lease / issues） | 主仓 `.../v0_2/shared/` |
| 本线交付物 | 本 worktree `.../v0_2/h2b/`（`support/`、`figures/`、`machine/`） |
| 大产物 | `/data/hfosp_group_event_state_v0_2/agent_b/`（22 MB） |

---

## 4. B0 结论（已完成）

- **crosswalk**：按录音编号连接、逐 onset 核对。274 次发作匹配，**零** onset 落在自己录音之外、
  零歧义、零重复。5 位 Yuquan 病人 0 条记录 = **未检出**，不是无发作。
- **成簇归并**：一位病人 8 次"发作"实为同一次（3.6 min 内、时长全 0）。
  274 次发作 → **209 个独立事件** → 留出 **99 个**。
- **各提前量可用锚点**：5min 98 / 30min 89 / 2h 79 / 6h 75（18–19 位病人）。
- **不应期敏感性**：30/60/120 min → 留出 106/99/80，**60 min 主口径不在悬崖上**（2h 档最敏感）。
- **锚点新鲜度**：上一次间期事件中位只早 5–14 s，几乎无超 1 h → 新鲜度不是瓶颈。
- **发作能量场**：264 ok / 10 dropped（10 条全部是窗口越出块边界）。按**脑电起点**锚定。
- **对拍**（168 次发作 / 11 人）：通道顺序处处一致；同锚点中位 **ρ=+0.9977**，
  原样发布口径 +0.8655。脑电起点在 **145/168** 次里早于临床起点（中位 5.0 s，最大 86.2 s）。
  ⚠️ **未解决**：8 次（全在 922）同锚点对拍 < 0.8，已排除信号强弱/电极数/基线长度/时间平移四种解释。
- **`block_id` 修复**：2 条记录指向比自己发作早 14 h 结束的块，已修并各救回 1 次发作。
- **坐标**：Epilepsiae 215 次全有标准空间坐标；**3 位 Yuquan 病人无坐标**（13 次发作）→ 算不了偏侧性。

### 承重的预注册数字（在读任何模型之前定死）

**静态"病人平均场"基线**（不含任何状态）预测留出发作头 5 s：**中位 ρ=+0.41，最高 +0.93**。
这是状态必须越过的线。单次发作自身的可复现度中位 +0.30（6/12 人低于 +0.30）——
**它不是上限**：平均能消掉单次噪声，静态基线在 9/12 人身上反而超过它。

---

## 5. B1 / B2：结论已撤回

两条主任务的仪器都已建成并跑通全队列（133 格、266 次运行、零失败），但**产出的数字全部撤回**：

- **B2**（发作早期空间场）：曾报"未见超过静态基线的增量"。撤回——所依据的 `P_local`/`P_slow`
  不可采纳，且读的是 raw latent 而非合同 `S_func`，关键对照亦未执行。
- **B1**（离发作还有多久）：**assay not estimable**——133 格仅 11 格越过"不得差于纯截距"闸门，
  19 人中 15 人零可用格。其负增量**不得**读作"状态有害"。

**不得引用本阶段任何 `P_local` / `P_slow` 数字。**

---

## 6. 已修的、会伪造结论的错（记下来防复发）

1. `np.savez` 给文件名补 `.npz`，打断原子写 → 封装 `save_npz_atomic` + 回归测试。
2. 打分器要求"发作档必须被完整观测" → **会丢掉真实发作**（2.5 h 发作 + 3 h 覆盖）。已放宽。
3. registry 读取器在缺请求 seed 时**静默回退** → 伪造三次重复。已改为拒绝并列出可用 seed。
4. 重跑子集**截断**全队列 status 表（271 → 60 行）→ 改为从盘上全部 JSON 重建。
5. 把"两次留出发作相似度"写成"预测器上限" → 同一张表当场证伪，已更正。
6. `eval_events` 报的是 5 min 网格行 → 916 上 1578 行仅 **40** 次发作；曾报 987 实为 **24**
   （**虚报 41 倍**）。已拆分为行数 / 独立发作数。
7. 温度网格最大 4.0、**不含**均匀权重 → "基线是严格嵌套特例"在模型选择中不成立。已加 `inf`。
8. 首版 registry 校验规则**过严** → 把记账哈希差异当缺陷，154 cell 全被拒，掩盖真问题。已改。

---

## 7. 下一步

1. 上游导出**冻结功能读出 `S_func`**（带名称、TRAIN 标准化统计、checkpoint hash）；主分析只读它。
2. 上游统一 `P_local`/`P_slow` 的 cell 配置（或说明 27–29 种差异来源），并修正 registry provenance。
3. 上游删除 `B_multiscale` 中一切由 `to_next` 派生的维度，重建**因果可用**基线后再导出。
4. 本线补 recent/current IED 与 **block circular shift** 两条对照；
   B1 补 calibration、seizure-level ranking、episode 内先汇总再 patient-first。
5. `epilepsiae_922` target provenance 闭合前，按已预声明的排除敏感性并行报告。

## 8. 复现

```bash
git worktree add /tmp/hfosp_group_event_state_v02_b codex/topic5-group-event-state-v02-b
cd /tmp/hfosp_group_event_state_v02_b
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
P=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
$P -m pytest tests/test_topic5_h2b_*.py -q                       # 105 passed
$P scripts/topic5_h2b_transfer/build_seizure_crosswalk.py
$P scripts/topic5_h2b_transfer/build_risk_sets.py
$P scripts/topic5_h2b_transfer/build_early_ictal_field.py --workers 8 --skip-existing
$P scripts/topic5_h2b_transfer/check_early_field_parity.py
$P scripts/topic5_h2b_transfer/summarize_field_targets.py
$P scripts/topic5_h2b_transfer/postictal_sensitivity.py
$P scripts/topic5_h2b_transfer/plot_early_field_qa.py
$P scripts/topic5_h2b_transfer/run_b1_plumbing.py     --subject epilepsiae_916 --producer P_slow --seed 1
$P scripts/topic5_h2b_transfer/run_b2_field_transfer.py --subject epilepsiae_916 --producer P_slow --seed 1
```

## 9. 未触碰

formal/sealed 分区、paper-ready Fig1–Fig4、Agent A 的 producer 代码与 registry producer 条目、
Agent C 的 H3 队列、v0.1 的输出目录与 tag。
