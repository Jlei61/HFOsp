# Agent B (H2b seizure transfer) — CURRENT_HANDOFF

状态：**B0 完成；B1/B2 仪器完成并已接上真 producer；科学结论一条都还不能下**
最后更新：2026-09-01

---

## 1. 一句话现状

支持度、目标、打分器、两条主任务的估计量都做完并跑通了，**而且已经接上 Agent A 的真 producer**。
但目前 `P_local` / `P_slow` 只覆盖 **27 人里的 2 人**，所以**任何方向的结论都不能下**——
现在能说的只有"这套东西能跑、分母有多大、什么样的阴性才算数"。

---

## 2. 环境与边界（实测，非转述）

| 项目 | 值 |
|---|---|
| worktree | `/tmp/hfosp_group_event_state_v02_b` |
| branch | `codex/topic5-group-event-state-v02-b`（base `f0c9e075`） |
| Python | `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`；线程全设 1 |
| 测试 | **98 passed**；改动**纯增量**（只有新增文件，0 个既有文件被改） |

**GPU：本线全程未申请。** 开工时 v0.1 队列占满两卡（99–100%），按工程附录不叠加；
该队列已于 22:33:44 自然跑完（`ALL QUEUES DONE`, 162/162），**不是被我停的**。
之后 Agent C 的 H3 队列接管 GPU。本线所有计算都是 CPU（8 worker，峰值 ~4.8 GB/worker）。

**Agent A registry**：`results/epi_prssm/group_event_state/v0_2/shared/checkpoint_registry.json`

| producer | status | B 侧可用性 |
|---|---|---|
| `B_multiscale` | complete (27 人) | ❌ `not_available`——只存结果，不存逐时刻的 111 维特征（已提 additive 请求） |
| `P_local` | partial (**2 人**) | ✅ 可读（916 三种子 / 253 一种子） |
| `P_slow` | partial (**2 人**) | ✅ 可读（916 三种子 / 253 两种子） |

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

## 5. B1 / B2 现状（仪器完成，结论未到）

### B1 生存
接上真 producer 后第一批数是每行 1–2 nat 的增益——**那是拟合外推，不是发现**。
三处 TRAIN-only 修正：特征标准化、岭系数按 TRAIN 内时序 CV 选、状态维度同法选。
并**实装**工程不变量「远坏于截距基线的拟合记为不可估计」——它确实触发：
修之前状态臂在 **9 格里有 6 格输给纯截距**。修之后 916 六格中五格可用，
增益 **−0.034 ~ +0.043 跨零**；253（46 个事件）仍全部不可估计。

### B2 早期空间场
估计量：用这位病人**自己**过去几次发作的场做加权平均，权重来自状态相似度；
两臂**同一批场**，只差权重是否由状态给（均匀权重 = 病人平均场基线，是状态臂温度→∞ 的严格特例）。
softmax 温度按 **TRAIN 内留一**选定后冻结。
- 916（20 个留出事件，基线 ρ≈0.92）：增益 −0.027 ~ +0.008，贴零；**基线本身没余量**。
- 253（**3 个**留出事件）：增益 −0.24 ~ +0.13，随种子乱跳 = 噪声。

**两位病人都不能用来说状态有用或没用。**

---

## 6. 已修的、会伪造结论的错（记下来防复发）

1. `np.savez` 会给文件名补 `.npz`，把「写临时文件再改名」的原子写打断 → 已封装 `save_npz_atomic` + 回归测试。
2. 打分器原本要求「发作那一档必须被完整观测」→ 会**丢掉真实发作**（2.5 h 的发作 + 3 h 覆盖）。已放宽到正确判据。
3. registry 读取器在「要 3 号种子、只有 1 号」时**悄悄给 1 号** → 伪造三次重复。已改为拒绝并列出可用种子。
4. 重跑子集会把全队列 status 表**截断**（271 行 → 60 行）→ 已改为从盘上全部 JSON 重建。
5. 我把「两次留出发作的相似度」写成"任何预测器的上限"——**同一张表当场证伪**，已在交付物里改正。

---

## 7. 下一步

1. 等 A 把 `P_local` / `P_slow` 铺到更多病人——**特别是静态基线低、有余量的那几位**
   （1073 基线 0.08 / 1096 −0.04 / 1125 0.11 / 548 0.30）。916 基线 0.93 没余量，
   在它身上得不到有信息的答案。
2. `B_multiscale` 的逐时刻特征到位后，把它接成正式对照臂（issue 已提）。
3. 922 那 8 次对拍偏低的原因仍未查明。
4. 收口：承重两图（生存增量 vs 提前量、空间场增量 vs 提前量，每点标留出发作数）+ 双报告。

## 8. 复现

```bash
cd /tmp/hfosp_group_event_state_v02_b
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
P=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
$P -m pytest tests/test_topic5_h2b_*.py -q                       # 98 passed
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
