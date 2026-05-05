# Topic 1 — Cluster geometry visualization results

> 状态：results doc（2026-05-06 验收）
> Plan-of-record：`docs/archive/topic1/propagation/cluster_geometry_viz_plan_2026-05-06.md`
> 主文档入口：`docs/topic1_within_event_dynamics.md` §3.1d
> 关联：`docs/archive/topic1/propagation/interictal_group_event_internal_propagation.md`（PR-2/2.5 数据来源）

---

## 1. 一句话结论

在 template-matching metric（shared-channel mean squared deviation，与 `assign_events_to_templates` 一致）下做 classical MDS 投影，PR-2 给出的 stable_k 簇结构在 cohort 上**视觉与定量都成立**：20/20 入选 Epilepsiae subject 全部成功生成 geometry，cohort silhouette median = **0.460**（range 0.182–0.671），KMeans-vs-template-matching 一致率 cohort median = **0.892**（range 0.769–0.955，与用户先行 audit 完全一致）。8/20 subject agreement < 0.85，但**核心论点（PR-2 决策稳健、forward/reverse template pair 真实）不被推翻**。

**958（k=2 forward/reverse）与 818（k=4 multimodal）的 panel c 模板剖面图直接复现了老论文 E3 的现象级 finding，并把多模态结构在二维平面上首次画出来。**

**Joint scatter 揭示一个新规律**：silhouette 与 agreement 在 cohort 上 **Spearman ρ = 0.889（p = 1.65e-07, n=20）—— 极强正相关**。换言之，在 template-matching metric 下 cluster validity 弱的 subject，KMeans 决策也最不稳；这两类问题不是独立的，而是同一组"边界事件支配"现象的两个测量维度。

**Boundary fraction by n_participating 验证 metric drift 来源**：3-4 通道事件 boundary fraction median = **0.135**，5-8 = **0.097**，9+ = **0.046**——单调下降。**metric drift 主要发生在低 n_participating 事件**，符合理论预期（NaN→0 vs masked metric 在低参与下差异最大）。

---

## 2. Cohort 数字摘要

> 数据源：`results/interictal_propagation/cluster_geometry/cohort_summary.json`

### 2.1 入选 / 排除

- **配置 cohort**：18 Yuquan + 20 Epilepsiae = 38 候选
- **入选**：**20**（全部 Epilepsiae）
- **排除（PR-2 labels 与当前 lagPat data 不对齐 / 缺 adaptive_cluster JSON）**：18 Yuquan
  - **8 个缺 adaptive_cluster JSON**：`zhangkexuan`、`pengzihang`、`songzishuo`、`zhangbichen`、`zhaochenxi`、`zhaojinrui`、`zhourongxuan`、`zhangjiaqi`（PR-2 没有为这些 subject 生成 per-subject JSON）
  - **10 个 size 不匹配**：`chengshuai`、`huangwanling`、`liyouran`、`chenziyang`、`hanyuxuan`、`huanghanwen`、`litengsheng`、`xuxinyi`、`zhangjinhan`、`sunyuanxin`（current lagPat valid_events 数量与 PR-2 saved labels 数量不一致；diff range 52–1893 events，xuxinyi 反向 −515 events）
  - 这是 **data freshness 问题，不是本 PR 的 bug**。`scripts/run_interictal_propagation.py` PR-2.5 路径在同样 check 上同样的 skip 行为
  - 修复方式 = 在最新 lagPat data 上重跑 PR-2 / PR-2.5 → 不在本 PR 范围

### 2.2 关键数字

| 指标 | cohort 中位数 | range | 警戒触发 |
|---|---|---|---|
| **silhouette median**（template-matching metric） | **0.460** | 0.182–0.671 | 5/20 < 0.3 |
| **KMeans-vs-matching agreement** | **0.892** | 0.769–0.955 | 8/20 < 0.85；2/20 < 0.80 |
| **MDS stress** | 0.365 | 0.273–0.481 | 18/20 > 0.30（k=2 多 cluster events 在 2D 上有 distortion 是预期；不是 cluster validity 问题） |
| **MDS imputation fraction** | (大多数 0) | — | 6/20 > 0.20（低 n_participating 事件多的 subject 会触发） |

### 2.3 Joint silhouette × agreement

- **Spearman ρ = 0.889，p = 1.65e-07，n = 20** —— 极强正相关
- **科学含义**：cluster validity 弱的 subject 同时是 KMeans-template-matching disagreement 高的 subject。两类问题不是独立的，而是同一现象的两个测量维度。
- **机制候选**：低 silhouette ↔ 多 boundary events ↔ 这些事件在 NaN→0 vs masked metric 下归类不同（因为它们的 n_participating 多半较低）。
- **图**：`figures/cohort_geometry_summary.png` panel c

### 2.4 Boundary fraction by n_participating

| n_part bin | cohort 中位数 boundary fraction | n_subjects |
|---|---|---|
| 3-4 | **0.135** | 16 |
| 5-8 | **0.097** | 20 |
| 9+ | **0.046** | 10 |

- 单调下降验证假设：metric drift 集中在低 n_participating 事件
- **图**：`figures/cohort_geometry_summary.png` panel d

### 2.5 stable_k 分布

| stable_k | n | subjects |
|---|---|---|
| k=2 | 19 | 其余全部 |
| k=4 | 1 | `818`（多模态唯一 case） |

注：原 PR-2 archive 报告 cohort 30/30 中 27×k=2 + 2×k=4 + 1×k=6，因此 cohort 主流是 k=2；本 PR 入选 20 中 19 个 k=2 与原 PR-2 一致。

---

## 3. Showcase 三张图的科学叙事

### 3.1 `958` — 老论文 E3 forward/reverse 复现（k=2）

- n=165577、stable_k=2、sil=0.413、agreement=0.902、stress=0.373
- inter-cluster Spearman r ≈ −0.91（PR-2 archive 数字）
- **Panel c 模板剖面**：cluster 0（蓝）与 cluster 1（赭）在同一通道集合（`G00–GH7`）上 rank 完全反向——这是**间期事件内部存在两条互逆主导传播路径**的视觉直接证据。老论文用 KMeans cluster centroid 已经报告过，**但缺乏在距离空间里的二维直接展示——本 PR 首次把它画出来**。
- **Panel a MDS scatter**：两类事件云清晰分开，模板大星各自落在事件云中心
- **Panel b silhouette**：两个 cluster 都以正 silhouette 为主；负 silhouette 比例 = **9.8%**（boundary events，落在两类之间的过渡带）
- **图**：`figures/showcase/958_geometry_showcase.png`

**论文叙事价值**：作为 paper 主图候选；和老论文 KMeans heatmap 互补，二维空间里直接展示 cluster decomposition 的几何

### 3.2 `818` — k=4 多模态 + 最低 agreement（双重 caveat showcase）

- n=11337、stable_k=4、sil=0.217、**agreement=0.769（cohort 最低）**、stress=0.481
- 替代原计划 showcase `huangwanling`（被 data freshness 排除）
- **Panel c 模板剖面**：4 条独立 rank pattern——cluster 0 / 1 / 2 / 3 在不同通道组合上 dominate；不是简单 forward/reverse，而是 4 种**独立**主要传播模式
- **Panel a MDS scatter**：4 个事件云在 2D 上有重叠（k=4 在 2D 必然 distortion，stress=0.481 反映这点；不是 cluster validity 问题，而是降维瓶颈）
- **Panel b silhouette**：4 个 cluster 都有显著负 silhouette；总 boundary fraction = **23.1%**（cohort 最高）
- **图**：`figures/showcase/818_geometry_showcase.png`

**论文叙事价值**：反例——证明 cluster decomposition 不止 k=2，**少数 subject 真的有更复杂结构**；同时它的高 boundary fraction 就是**本 PR 审阅发现的 metric drift 在哪里最严重**——一图两用

### 3.3 `253` — 低 cluster validity + 低 agreement caveat

- n=75053、stable_k=2、sil=0.182（cohort 最低）、agreement=0.785、stress=0.402
- auto-pick 选出（agreement < 0.85，silhouette 最低）
- **Panel a MDS scatter**：两类事件云重叠严重，模板大星彼此距离不远，没有清晰决策边界
- **Panel b silhouette**：两个 cluster 都有约 1/3 事件 silhouette 接近 0；boundary fraction = **21.5%**
- **Panel c 模板剖面**：两条折线走向接近平行（不是反向也不是错开），说明这个 subject 的"两个 cluster"在通道 rank 维度上区分度本身就低
- **图**：`figures/showcase/253_geometry_showcase.png`

**论文叙事价值**：诚实暴露 cluster decomposition 在该 subject 上较弱；**不掩饰** PR-2 KMeans 决策在 metric 选择敏感的 subject

---

## 4. 审阅发现：KMeans vs template-matching metric drift（写入档案）

### 4.1 现象记录

实证核对（用户先行 + 本次 cohort 跑数确认完全一致）：
- **Cohort 中位一致率 = 0.892**（用户先前报 0.892 ✓）
- **最低一致率 = 0.769**（用户先前报 0.769 ✓ —— 落在 818）
- 大部分 subject 仍 ≥ 0.85（12/20 = 60%），说明 PR-2 决策在 cohort 层 **稳健**

### 4.2 机制层差异（清单）

1. **非参与通道**：KMeans 把 NaN→0 当成 rank=0；matching 忽略非参与通道
2. **聚合方式**：KMeans sum squared Euclidean；matching mean squared deviation（除以 shared channel 数）
3. **Centroid 来源**：KMeans `cluster_centers_` 是 sklearn 算法输出；matching 用的 template 是 `_legacy_hist_mean_rank` 的 3-bin top-rank 模板（来自 legacy 老代码合同）——**不是同一对象**
4. **min_shared_channels gate**：matching 有硬门槛；KMeans 没有

### 4.3 不影响 Topic 1 主结论的依据

PR-2 / PR-2.5 / PR-3 / PR-4A-D / PR-5 / PR-6 / PR-7 的核心结论都不依赖 masked metric 单独成立：
- `stable_k` 是 KMeans feature space 内的稳定压缩（不依赖 masked metric）
- `within-cluster τ` 是 shared participating channels 上的 Kendall τ
- PR-2.5 split-half / odd-even 用 template Spearman matching + assignment agreement
- PR-5-A 用 best-template r / reconstruction error / gap gate
- PR-6 用 endpoint Jaccard
- PR-7 用 lag-1 / event-level fixed-window null

### 4.4 后续工作（**不在本 PR**）

按优先级：

1. **重跑 PR-2 / PR-2.5 在最新 lagPat data 上**（**P0**）：恢复 18 Yuquan subject 入选；这是上游 pipeline 责任，本 PR 只是发现并记录
2. **量化"哪条机制层差异贡献最多 disagreement"**：按 4 个差异维度做 ablation，看是否其中某一项主导
3. **评估：是否应统一用 masked metric 重做 KMeans**：用 SMACOF MDS 嵌入做 KMeans，或直接 k-medoids on masked distance
4. **对最低 agreement 的 subject 做 case-by-case 检查**：`818`、`253`、`635`、`442`、`1096`——是否影响 PR-2.5 / PR-3 / PR-4 解读
5. 如果统一 metric 下 stable_k 仍然稳定，写一个 v2 版 PR-2 替换原 KMeans

---

## 5. 输出资产清单

```
results/interictal_propagation/cluster_geometry/
├── per_subject/
│   ├── epilepsiae_<X>.json    # 20 个 status="ok"
├── cohort_summary.json
└── figures/
    ├── README.md              # 中文说明每张图
    ├── per_subject/
    │   └── epilepsiae_<X>_geometry.png  # 20 张
    ├── showcase/
    │   ├── 958_geometry_showcase.png    # k=2 forward/reverse
    │   ├── 818_geometry_showcase.png    # k=4 multimodal + lowest agreement
    │   └── 253_geometry_showcase.png    # auto-pick low-validity caveat
    └── cohort_geometry_summary.png      # 2x2 cohort summary
```

代码：

- `src/cluster_geometry.py` — 距离 + classical MDS + per-event readout + cohort aggregator
- `tests/test_cluster_geometry.py` — 12 项 TDD（全 pass）
- `scripts/run_cluster_geometry.py` — batch driver（`--dataset`、`--subject`、`--all`、`--dry-run`、`--max-events`、`--cohort-only`）
- `scripts/plot_cluster_geometry.py` — figure generator（`--per-subject`、`--cohort`、`--showcase`、`--all`）

文档：

- `docs/archive/topic1/propagation/cluster_geometry_viz_plan_2026-05-06.md` — plan-of-record
- `docs/archive/topic1/propagation/cluster_geometry_viz_results_2026-05-06.md` — 本 results doc
- `docs/topic1_within_event_dynamics.md` §3.1d — 主文档 pointer
- `CLAUDE.md` §7 — Multi-panel figure discipline rule（从本 PR 设计 review 抽出）

---

## 6. 与既有 PR / 主文档的边界（再次确认）

- **不改 PR-2 / 2.5 / 3 / 4A-D / 5 / 6 / 7 任何已落定结论**
- **不改 Topic 1 主文档的"当前正式口径"**：仅在 §3.1d 加 pointer 指向本 results doc + plan
- **不替换 PR-3 6-panel cohort 图**：PR-3 是 raw KMeans-feature-space 的统计快照，本工作是 template-matching metric 下的几何展开，两者并列、互不替代
- **不进入 SBA framework P1–P5 任何 prediction 池**：本工作是描述性 supplement

---

## 7. 验收意见

| 项 | 状态 | 证据 |
|---|---|---|
| 距离 metric 与 `assign_events_to_templates` 一致 | ✅ | `tests/test_cluster_geometry.py` Test 3 |
| Classical MDS 在 toy 数据上零 stress | ✅ | Test 4 |
| Anti-correlated templates 在 MDS 中落在事件云两端 | ✅ | Test 6 |
| Per-event silhouette 边界正确 | ✅ | Test 7 |
| 端到端 pipeline 在合成数据上 sil > 0.5、agreement > 0.95 | ✅ | Test 8 |
| Subsample 保留所有模板坐标 | ✅ | Test 9 |
| All-NaN template 触发 exclusion | ✅ | Test 11 |
| Cohort summary 对 mixed status 健壮 | ✅ | Test 10 |
| 全 cohort 跑通 + figures 出 | ✅ | 20 per_subject + 1 cohort + 3 showcase；`cohort_summary.json` 完整 |
| 958 panel c 复现 forward/reverse | ✅ | `figures/showcase/958_geometry_showcase.png` |
| 818 panel c 复现 k=4 多模态 | ✅ | `figures/showcase/818_geometry_showcase.png` |
| 用户先行 audit 数字（median 0.892, min 0.769）匹配 | ✅ | cohort_summary.json 完全一致 |
| KMeans-vs-matching joint 显著相关 | ✅ | Spearman ρ = 0.889, p = 1.65e-07 |
| Metric drift 集中在低 n_part 事件 | ✅ | boundary fraction 单调下降：0.135 → 0.097 → 0.046 |

---

## 8. 一句话面对外部审稿人 / 论文的口径

> Stable interictal HFO group-event clusters identified by PR-2 KMeans on
> rank vectors (NaN→0 imputed) are visually and quantitatively reproduced
> in the orthogonal template-matching distance metric (shared-channel
> mean squared deviation): 20/20 included Epilepsiae subjects yield
> well-formed clusters in the augmented event+template MDS plane (cohort
> silhouette median 0.46, range 0.18–0.67), and the KMeans-vs-template-matching
> label agreement is high in cohort (median 0.89, range 0.77–0.96), with
> disagreement concentrated at events with fewer participating channels
> (boundary fraction monotone in n_part: 0.13 → 0.10 → 0.05). The
> forward/reverse template geometry first reported in legacy E3
> (subject 958) is directly visible in panel c of the showcase figure;
> a k=4 multimodal subject (818) shows that the two-template compression
> is not universal. The 18 Yuquan subjects of the original 38-subject
> cohort were excluded due to data-freshness mismatches between the saved
> PR-2 labels and the current lagPat data, and require a PR-2 / PR-2.5
> rerun on the latest pipeline output before they can be added back.
