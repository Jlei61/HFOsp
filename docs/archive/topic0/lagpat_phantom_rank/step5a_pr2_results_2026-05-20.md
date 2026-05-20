# Step 5a — PR-2 修过版重跑结果（2026-05-20）

> 状态：Step 5a 已完成。Step 5b（split-half 复现）尚未启动。
> 主入口：`docs/topic0_methodology_audits.md`
> 路线图：`./rerun_roadmap_2026-05-20.md`
> 诊断（Step 2/3）：`./diagnostic_2026-05-20.md`
> 白话报告：`./plain_chinese_report_2026-05-20.md`
> 修过版结果：`results/interictal_propagation_masked/`（per_subject + cohort summary）
> 对比汇总：`results/interictal_propagation_vs_masked/`

---

## 1. 三段式朴素话

**测了什么** —— 在 40 个 subject 上，把"每个 HFO 群体事件被分到哪一类传播模式"这件事，**用原版（含假名次的）和修过版（去掉假名次的）两套同样的算法都跑一遍**，看两套分类的事件归属差多少。

**怎么测的** ——
1. 对每个 subject，先按当前 PR-2 主路径（KMeans on `lagPatRank`）原样跑一次，写到 `results/interictal_propagation/`（旧目录）。这步是历史已有的，不重新跑。
2. 再用同样的 KMeans，但 feature 矩阵改成"只在参与的通道之间重排名次，不参与的通道用中点 0.5 占位"，写到 `results/interictal_propagation_masked/`（并行目录）。
3. 两边跑完，对每个 subject 比三件事：
   - **类别数变没变**：原版 chosen_k vs 修过版 chosen_k
   - **类别大小变没变**：每个 cluster 的事件比例 (best-permutation 对齐后)
   - **具体哪个事件分到哪一类变没变**：Jaccard / exact agreement / AMI (best-permutation 对齐后)
4. 再把这个"PR-2 层面跑出来的差距"和上周 Step 2 audit 跑出来的"AMI 与 noise floor 距离"做相关——如果两者一致，说明 audit 不是孤立测量，consumer 端跑出来的差也吻合。

**揭示了什么** ——
- "类别数"层面：**结构稳健**。35 个主线 subject (k=2 cohort) 里 34 个 chosen_k 仍是 2；唯一翻转的是 epilepsiae_916（2→4，这个 subject 之前在 Topic 4 attractor 里也是 GOF 不通过的那一个）。
- "哪个事件分到哪一类"层面：**变化明显**。33 个可比较的 subject 里，事件归属 Jaccard 中位 0.71，6 个 subject < 0.5（约一半事件被重新归类），15 个 < 0.7。
- "类别大小"层面：**中等翻动**。7/33 subject 的 cluster 大小比例变化 > 10%（中位 0.047）。
- 跟 Step 2 audit 的对照：**Spearman ρ = 0.961（p ≈ 7e-20, n=35）**。consumer 端跑出来的差跟之前 audit 跑出来的差几乎完全一致——audit 不是测错，而且不存在 noise floor 把 audit ceiling 吃掉的隐藏问题。

代号补注：phantom-rank fix, masked KMeans features, `use_masked_features=True`, `compute_adaptive_cluster_stereotypy`, Step 5a per the rerun roadmap §5a。

---

## 2. Cohort 数字表

### 2.1 chosen_k / stable_k 翻转（4/40）

| sid | stable_k_orig | stable_k_masked | 备注 |
|---|---:|---:|---|
| `yuquan_huangwanling` | 4 | 3 | 高 k cohort（n_ch=4，已被 Topic 4 H3 主分析独立排除） |
| `yuquan_zhaojinrui` | 5 | 6 | 高 k cohort（n_ch=4，同上） |
| `yuquan_zhangjinhan` | 6 | 5 | 高 k cohort（n_ch=5，同上） |
| **`epilepsiae_916`** | **2** | **4** | **主线 stable_k=2 cohort 内唯一翻转**；之前是 Topic 4 attractor Step 1 GOF-fail subject (var_explained_curve=0.565)；这个 subject 的"两类"原本就站不稳 |

四个翻转的 sid 跟 Step 2 audit 在 `results/lagpatrank_audit/cohort_summary.csv` 里报的 `stable_k_changed=True` 集合完全一致——consumer 端鉴定与 audit 端鉴定吻合。

### 2.2 stable_k=2 cohort 内、chosen_k 不变的 subject 的事件归属差距（n=33）

排除 epilepsiae_1096：该 subject 在 lagpatrank_audit 里就标了 `pr2_label_event_index_drift`（旧 PR-2 label 写到磁盘的时候用的是更早的 event indexing，labels 大小与现在 loader 输出不匹配），不能逐事件 contingency。剩 33 个可比较。

| metric | median | p25 | p75 | min | max |
|---|---:|---:|---:|---:|---:|
| Jaccard (macro, best-perm) | **0.707** | 0.604 | 0.790 | 0.346 | 0.897 |
| exact agreement (best-perm) | **0.847** | — | — | 0.516 | 0.950 |
| AMI(orig labels, masked labels) | **0.372** | — | — | 0.001 | 0.700 |
| max cluster fraction shift | 0.047 | 0.018 | 0.082 | 0.000 | 0.226 |
| **n with Jaccard < 0.5** | **6/33** | — | — | — | — |
| **n with Jaccard < 0.7** | **15/33** | — | — | — | — |
| **n with fraction shift > 0.10** | **7/33** | — | — | — | — |

**解读**：即便"K=2"这个大数稳健（34/35），具体"哪个事件属哪一类" + "两类多少事件" 在大约一半 subject 上发生substantial 改动。下游所有用了"per-subject 每事件 cluster label" 的工作（PR-3/PR-4/PR-5/PR-6/PR-7/Topic 4 attractor）都得在修过版的 label 上重新算一遍。

### 2.3 audit vs PR-2 rerun 一致性

| metric | 值 |
|---|---|
| Spearman ρ(audit `ami_audit_minus_floor`, PR-2 rerun `AMI(orig, masked)`) | **0.961** |
| p | **6.65e-20** |
| n | 35 |

这条 sanity 把 Step 2 audit (`scripts/audit_kmeans_phantom_rank.py`) 的方法学结论从"自带 noise floor 锚定的低 cost 测量"升级到"PR-2 主管道层面的同方向、相近幅度的 consumer-side 验证"。**修过版 consumer pipeline 没有引入新偏置**：它跑出来的差只是 audit 已经测到的差的直接 manifestation。

---

## 3. 进入 5b 的条件

按路线图 §Checkpoint A 的合同：

- 主线 cohort stable_k flip 数 = **1**（仅 epilepsiae_916），**远低于** Checkpoint A 的红线 "3 个 flip 触发暂停审 masking 策略"
- chosen_k 在 34/35 主线 subject 上一致，"K=2 是间期刻板时序主导特征" 这个结构层结论不动
- audit ↔ PR-2 rerun 一致性极高（ρ=0.961），masking 策略本身没有引入新问题

→ **Checkpoint A 通过，可进入 Step 5b（split-half / odd-even 复现 on masked）**。

进入 5c (PR-3) / 5d (PR-4) 前需要先 5b 跑完，把 `forward_reverse_reproduced` 集合稳定下来再做 advisor consult（在 5b 跑完时正式做）。

---

## 4. 受影响 PR 状态更新（Topic 0 §3.1 表）

| 下游 | 之前状态 | 5a 后状态 |
|---|---|---|
| PR-2 主管道 (KMeans on rank) | YES | **重跑完，K 不变、labels 实质变** |
| PR-2.5 split-half / odd-even | YES | 待 5b |
| PR-3 per-cluster MI | YES | 待 5c |
| PR-4A/B/C/D | YES | 待 5d |
| PR-5/PR-5-B | YES | 待 5e |
| PR-6 全套 | YES | 待 5f |
| PR-7 | YES | 待 5g |
| Topic 4 attractor Step 1 | YES | 待 5h |
| cluster_geometry PCA embedding | YES | 待补 5x（Step 5a 不覆盖） |

---

## 5. 输出位置

- per-subject masked PR-2 JSON：`results/interictal_propagation_masked/per_subject/*.json` (40 files)
- cohort summary：`results/interictal_propagation_masked/pr1_cohort_summary.json`
- 对比 CSV：`results/interictal_propagation_vs_masked/pr2_comparison.csv` (40 rows + summary fields)
- 对比 summary md：`results/interictal_propagation_vs_masked/pr2_comparison_summary.md`
- 两张对比图：
  - `results/interictal_propagation_vs_masked/figures/label_jaccard_distribution.{png,pdf}` — 33 subject 上 Jaccard 直方图 + audit ↔ PR-2 rerun 相关散点
  - `results/interictal_propagation_vs_masked/figures/cluster_fraction_shift.{png,pdf}` — 33 subject 上 cluster fraction 最大变化直方图
  - + `figures/README.md`
- run log：`logs/step5a_pr2_masked_cohort.log`（约 5h40m wall, 14:55–20:37）

---

## 6. 下一步

立即可跑：
```
python scripts/run_interictal_propagation.py --pr25 --masked-features
```

这一步会：
- 加载 `results/interictal_propagation_masked/per_subject/*.json` 里 5a 写好的 masked PR-2 labels
- 对每个 subject 跑 split-half + odd-even split，每个 split 内用 `use_masked_features=True` 重新 KMeans，匹配到原 chosen_k templates，算 `forward_reverse_reproduced` (split-half OR odd-even 规则 per AGENTS.md cross-PR contract)
- 把 `time_split_reproducibility` 字段写回每个 subject JSON

预计 1–2 小时（split-half / odd-even 各跑一次 KMeans，per subject）。
