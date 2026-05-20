# Topic 0：方法学审计与数据合同

> 状态：当前正式入口
> 范围：跨 topic 的方法学问题、数据合同 bug、需要回溯重跑的污染源。
> 不属于：单个 topic 内部的科学结论（→ Topic 1–5 主文档）。
> **优先级**：本 topic 的问题一旦未解决，下游 topic 的结论强度受限。`docs/paper_overview.md` 的 "一句话总论" 必须明示当前未结清的 Topic 0 问题。

---

## 1. 这个 topic 只回答什么

本 topic 回答 **"我们用的数据本身是否如其所言"**——而不是"数据告诉我们生物学是什么"。

它**只**记录：

1. 已发现的方法学 bug / 数据合同问题（legacy 继承的、本仓库引入的、未来发现的）
2. 每个问题的实证证据 + 影响范围
3. 修复合同 + 重跑路线图
4. 修复版结果与旧结果的并行存放约定（这样能"看出问题"）

它**不**记录：

- 任何 topic 内部的科学结论（→ 各 topic 主文档）
- 工程基础设施（→ AGENTS.md / CLAUDE.md）
- 文献综述 / framework discussion（→ `docs/paper1_framework_sba.md`）

---

## 2. 一句话当前状态

- **当前未结清问题**：1 个（`lagPatRank` phantom pseudo-rank，2026-05-20 确诊，broad re-derivation gate 已触发）。
- **当前进行中的重跑**：见 §5 重跑路线图。**Topic 1 / Topic 4 / PR-5 / PR-6 / PR-7 的所有数字背后挂有方法学 caveat**，直到 §5 走完。

---

## 3. 已记录的方法学问题

### 3.1 `lagPatRank` phantom pseudo-rank（已确诊 2026-05-20，重跑中）

**问题**：legacy 数据生成代码（`ReplayIED/.../hfo_net.py:289`）对**所有通道**排名次，包括"不参与"的通道——这些通道获得来自谱图噪声的虚假整数名次。HFOsp 把这些假名次直接喂给 KMeans，造成事件归类被假名次驱动。

**实证证据**（chengshuai/FC10477Y, n_ch=8, n_ev=3513）：
- 9064 个本该空缺的格子全部有 0–7 的整数名次
- 分布 U 形：rank=7 占 19.1%、rank=0 占 17.8%、中段 ~10%
- 修过版 vs 原版 KMeans 分类的事件归属一致程度 = 0.001（噪声基线 0.95）

**40-subject cohort 实测**（`results/lagpatrank_audit/cohort_summary.csv`）：
- cohort-median Δ（修过版 vs 原版） = **-0.599**
- **40/40** subject 都越 broad-rederivation 阈值（Δ < -0.10）
- 主线 stable_k=2 cohort 唯一最佳类别数翻转：epilepsiae_916（2→4）
- 3 个高 k 翻转都在 stable_k>2 cohort（这部分已被 Topic 4 H3 主分析独立排除）

**方法学定性**（不是 noise）：
phantom-driven clustering 在 seed 间**完全可重现**（noise floor median 0.997）——bug 制造了一组与真实激活顺序无关、但稳定、可重现的 cluster。性质上与 PR-4 panel d "簇内 86% identity bias" 同类。

**受影响 PR**：PR-2 / PR-2.5 / PR-3 / PR-4A / PR-4B / PR-4C / PR-4D / PR-5 / PR-5-B / PR-6 H1 / PR-6 H2 / PR-6 Step 6 / PR-6 supplementary rank displacement / PR-7 / Topic 4 attractor Step 1 / cluster_geometry PCA embedding。

**修复方式**：consumer 侧 masked re-rank + per-event 中点 (0.5) impute。不重跑 legacy producer（破坏 paper 字节级 reproducibility，且代价过高）。详见：
- 计划：`docs/superpowers/plans/2026-05-20-lagpatrank-phantom-rank-audit-and-fix.md`
- 诊断归档：`docs/archive/topic0/lagpat_phantom_rank/diagnostic_2026-05-20.md`
- 白话报告：`docs/archive/topic0/lagpat_phantom_rank/plain_chinese_report_2026-05-20.md`
- 重跑路线图：`docs/archive/topic0/lagpat_phantom_rank/rerun_roadmap_2026-05-20.md`

**重要 disclaimer**：修过版只去除了"端点名次假"的偏置，**不**解开"事件归类是被排序驱动还是被参与模式驱动"这道更深层的问题。后者由 cluster geometry / participation-pattern 框架的后续工作回答。

---

## 4. 命名约定 — 修复版结果如何与旧结果区分

为了让"修过版 vs 旧版"能并排对比，约定以下结构：

### 4.1 结果目录（默认 parallel dir）

修复版结果 = **旧结果路径 + `_masked` 后缀** 的并行目录：

```
results/interictal_propagation/             ← 旧（phantom-contaminated），不动
results/interictal_propagation_masked/      ← 新（masked re-rank）

results/topic4_attractor/                   ← 旧
results/topic4_attractor_masked/            ← 新

results/interictal_propagation/pr6_template_anchoring/   ← 旧
results/interictal_propagation_masked/pr6_template_anchoring/   ← 新
```

**理由**：
- 旧结果对旧归档 doc 是 evidence，不可删
- 修复版与旧版并存便于 diff、可视化对比、独立审计
- git ls-files 上 path 不重叠，blame / log 干净

### 4.2 图文件名（不加后缀）

修复版图直接放在 `_masked/` 子目录里，**文件名不重复加 `_masked`**——目录已经区分了，加后缀只会冗余：

```
results/interictal_propagation_masked/figures/cohort_propagation_summary.png   ✓
results/interictal_propagation_masked/figures/cohort_propagation_summary_masked.png   ✗
```

例外：当某张对比图直接放在原 dir 里（比如 PPT 文件夹），文件名加 `_v2masked` 后缀。

### 4.3 对比图（专门的 dir）

如果做 "before-vs-after" 并排对比图（典型场景：让评委一眼看出 phantom 把哪些事件挪到了哪一类），放在专门的 `_vs_masked/` dir：

```
results/interictal_propagation_vs_masked/figures/
  per_subject/yuquan_chengshuai_cluster_relabel.png   ← 显示 phantom 把哪些事件挪了类
  cohort_stable_k_flip.png
```

### 4.4 per-subject JSON 字段

per-subject JSON 顶层不动现有 schema；如果同时要在一个文件里携带旧+新结果，约定 key：

```json
{
  "adaptive_cluster": {...},                ← 旧（phantom-contaminated）
  "adaptive_cluster_masked": {...},         ← 新（masked）
  "audit": {                                ← 跨修复的对比汇总
    "ami_audit_minus_floor": -0.599,
    "stable_k_changed": false,
    "labels_relabel_jaccard": 0.34
  }
}
```

但**默认不混存**——修过版结果就完全独立写到 `_masked/` 目录的 JSON，避免老 reader 不小心读到混合 schema。

---

## 5. 重跑路线图

详见：`docs/archive/topic0/lagpat_phantom_rank/rerun_roadmap_2026-05-20.md`

按依赖顺序：

| Step | 内容 | 输出目录 | 状态 |
|---|---|---|---|
| 5a | PR-2 re-cluster on masked | `results/interictal_propagation_masked/` per_subject | 待启动 |
| 5b | PR-2.5 split-half / odd-even on masked | 同上 | 待启动 |
| **Checkpoint A** | advisor consult: stable_k flip + forward/reverse 复现集合变化 | — | — |
| 5c | PR-3 per-cluster MI on masked labels | 同上 | 待启动 |
| 5d | PR-4A/B/C/D on masked labels | 同上 + 各子 dir | 待启动 |
| **Checkpoint B** | advisor consult: PR-4B/D 方向是否反转 | — | — |
| 5e | PR-5 / PR-5-B on masked | `results/interictal_propagation_masked/template_share_switching/` | 待启动 |
| 5f | PR-6 endpoint anchoring / swap / Step 6 / rank displacement on masked | `results/interictal_propagation_masked/pr6_*/` | 待启动 |
| 5g | PR-7 antagonistic pairing on masked | `results/interictal_propagation_masked/pr7_*/` | 待启动 |
| 5h | Topic 4 attractor on masked | `results/topic4_attractor_masked/` | 待启动 |
| 5i | 主文档更新 + AGENTS.md cross-PR 合同更新 | `docs/topic1_*.md` / `docs/topic4_*.md` (新) | 待启动 |

每个 Step 跑完，更新 §3.1 的 "受影响 PR" 表，把对应行从"待重跑"挪到"重跑完，方向 = 一致 / 翻转 / NULL flip"。

**Checkpoint A / B 是 hard stops**：未通过 advisor consult 不进入下一阶段。

**PR-7 P3 翻转特别 flag**：如果 §5g 把 P3 从 INCONCLUSIVE 推到 PASS 或 FAIL，这是 framework-level revision，必须先停下来发起 `docs/paper1_framework_sba.md` 修订 review，**不允许**默默改写。

---

## 6. 文件清单

### 代码

- `src/lagpat_rank_audit.py` — masked re-rank helpers
- 修改：`src/interictal_propagation.py` — 4 个 KMeans 调用点加 `use_masked_features` 参数（待启动）
- 修改：`src/cluster_geometry.py:compute_pca_embedding` — 同参数（待启动）
- 修改：`src/topic4_attractor_diagnostics.py:build_rank_feature_matrix` — `mask_phantom` 参数（待启动）

### 测试

- `tests/test_lagpat_rank_audit.py`（5 tests pass 2026-05-20）

### 脚本

- `scripts/audit_kmeans_phantom_rank.py` — 40-subject AMI audit driver
- `scripts/augment_lagpat_audit_masked_stable_k.py` — masked stable_k 重选
- `scripts/plot_lagpat_phantom_audit.py` — 3 张诊断图

### 数据 / 图（已落地）

- `results/lagpatrank_audit/cohort_summary.csv`（40 行）
- `results/lagpatrank_audit/<sid>.json` × 40
- `results/lagpatrank_audit/figures/`：
  - `ami_vs_noise_floor.{png,pdf}`
  - `phantom_fraction_vs_delta.{png,pdf}`
  - `stable_k_confusion.{png,pdf}`
  - `README.md`

---

## 7. 与其他 topic 的边界

- Topic 1–5 主文档的科学结论引用本 topic 时，必须明示 "假设 §3.1 phantom-rank 重跑完成且方向一致"
- 任何下游 PR plan 在 trigger 前必须检查 §3 是否有未结清的 audit 直接影响该 PR
- AGENTS.md "Cross-PR Contract Lookups" 会增加本 topic 的 lookup 条目（待 §5i 更新）

---

## 8. 历史文档索引

- `docs/superpowers/plans/2026-05-20-lagpatrank-phantom-rank-audit-and-fix.md` — plan-of-record（含 noise-floor anchor gate 设计 + advisor checkpoint A/B + PR-7 framework flag）
- `docs/archive/topic0/lagpat_phantom_rank/diagnostic_2026-05-20.md` — 技术诊断归档（cohort 数字 + gate verdict）
- `docs/archive/topic0/lagpat_phantom_rank/plain_chinese_report_2026-05-20.md` — 白话报告（给非技术读者 / 自己回看用）
- `docs/archive/topic0/lagpat_phantom_rank/rerun_roadmap_2026-05-20.md` — Step 5a-5i 重跑路线图
- `docs/archive/topic1/propagation/lagpatrank_phantom_audit_diagnostic_2026-05-20.md` — **MOVED 2026-05-20** 到 `docs/archive/topic0/lagpat_phantom_rank/diagnostic_2026-05-20.md`（原 path 仅留 redirect stub）
