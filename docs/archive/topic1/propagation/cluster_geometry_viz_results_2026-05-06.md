# Topic 1 — Cluster geometry visualization results

> 状态：results doc（2026-05-06 验收，2026-05-06 修订）
> Plan-of-record：`docs/archive/topic1/propagation/cluster_geometry_viz_plan_2026-05-06.md`
> 主文档入口：`docs/topic1_within_event_dynamics.md` §3.1d
> 关联：`docs/archive/topic1/propagation/interictal_group_event_internal_propagation.md`（PR-2/2.5 数据来源）

---

## 1. 一句话结论（已降调）

在 template-matching metric（shared-channel mean squared deviation，与 `assign_events_to_templates` 一致）下做 classical MDS、并在 KMeans-native feature space 做 PCA 互为对照，PR-2 给出的 stable_k 簇结构在入选 Epilepsiae cohort 上**整体可解释**，但存在**明确的 boundary-event drift**：silhouette 中位 = 0.460（range 0.182–0.671，5/20 < 0.3），KMeans 与 template-matching reassign 的事件级一致率中位 = 0.892（range 0.769–0.955，8/20 < 0.85）。drift 主要集中在低 n_participating 事件上（boundary fraction 单调下降 0.135 → 0.097 → 0.046）。

**958（k=2 forward/reverse）的 panel c 模板剖面、818（k=4 多模态）的 4-panel 结构、253（k=2 低-validity）的 boundary 重叠**——三张 showcase 各自展示一种典型情况，是这次 PR 的描述性贡献；不构成对 PR-2/2.5 主结论的推翻或新生物 finding。

**18 个 Yuquan subjects** 因 PR-2 saved labels 与当前 lagPat valid_events 数量不一致（10）或缺 adaptive_cluster JSON（8），在最新 lagPat data 上无法直接复用 PR-2 标签——本 PR **不在该 cohort 上提供完整 Topic 1 cohort 图**；这是 upstream data freshness 问题，列为 P0 follow-up（在最新 lagPat 上重跑 PR-2 / PR-2.5）。

---

## 2. Cohort 数字摘要

> 数据源：`results/interictal_propagation/cluster_geometry/cohort_summary.json`

### 2.1 入选 / 排除

- **配置 cohort**：18 Yuquan + 20 Epilepsiae = 38 候选
- **入选**：**20**（全部 Epilepsiae）
- **排除（写入 cohort_summary.excluded）**：18 Yuquan
  - **8 个 `adaptive_cluster_missing`**：`zhangkexuan`、`pengzihang`、`songzishuo`、`zhangbichen`、`zhaochenxi`、`zhaojinrui`、`zhourongxuan`、`zhangjiaqi`（PR-2 没有为这些 subject 生成 per-subject JSON）
  - **10 个 `label_size_mismatch`**：`chengshuai`、`huangwanling`、`liyouran`、`chenziyang`、`hanyuxuan`、`huanghanwen`、`litengsheng`、`xuxinyi`、`zhangjinhan`、`sunyuanxin`（current lagPat valid_events 数量与 PR-2 saved labels 数量不一致；diff range 52–1893 events，xuxinyi 反向 −515 events）
  - 这是 **data freshness 问题，不是本 PR 的 bug**；`scripts/run_interictal_propagation.py` PR-2.5 路径在同样 check 上同样的 skip 行为
  - 修复方式 = 在最新 lagPat data 上重跑 PR-2 / PR-2.5 → 不在本 PR 范围

### 2.2 关键数字（已确认 cohort_summary.json）

| 指标 | cohort 中位数 | range | 触发警戒 |
|---|---|---|---|
| **silhouette median**（template-matching metric） | **0.460** | 0.182–0.671 | 5/20 < 0.3 |
| **KMeans-vs-matching agreement** | **0.892** | 0.769–0.955 | 8/20 < 0.85；2/20 < 0.80 |
| **MDS stress** | 0.365 | 0.273–0.481 | 18/20 > 0.30——k=2 多 cluster events 嵌入 2D 必然有 distortion，**不**是 cluster validity 问题 |
| **MDS imputation fraction** | (大多数 0) | — | 6/20 > 0.20（低 n_participating 事件多的 subject 触发） |

### 2.3 Silhouette × agreement consistency check（不是新规律）

- Spearman ρ = **0.889**，p = 1.65e-07，n = 20
- **解读**：silhouette 与 agreement 都派生自同一组 d_within / d_min_other 距离差——event silhouette 为负 ≡ matching reassign 把它换给最近他簇 ≡ 与 KMeans label 不一致。所以 cohort 上正强相关本来就是定义上预期，**不是独立生物 finding**
- 用作 cohort 层 sanity check：如果方向反向或弱相关，反而说明 pipeline 有问题
- **图**：`figures/cohort_geometry_summary.png` panel c

### 2.4 Boundary fraction by n_participating

| n_part bin | cohort 中位数 boundary fraction | n_subjects |
|---|---|---|
| 3-4 | **0.135** | 16 |
| 5-8 | **0.097** | 20 |
| 9+ | **0.046** | 10 |

- 单调下降，符合预期：matching metric 只用共享通道，低 n_part 事件的距离判定本来就不稳；KMeans 用全 rank 向量（含 fallback rank）则不受 n_part 限制
- **图**：`figures/cohort_geometry_summary.png` panel d

### 2.5 stable_k 分布

| stable_k | n | subjects |
|---|---|---|
| k=2 | 19 | 其余 |
| k=4 | 1 | `818`（多模态唯一 case） |

注：原 PR-2 archive 报告 30/30 subject 中 27×k=2 + 2×k=4 + 1×k=6；本 PR 入选 20 中 19 个 k=2 与 PR-2 在重叠 cohort 上一致。

---

## 3. KMeans 与 template-matching metric 的真实差异（已修正）

之前文档的"非参与通道 NaN→0"描述**不准确**。实证核对（chengshuai、253）：lagPatRank 在非参与通道上**不是 NaN/0**，而是 legacy `return_massCenterPat` 给出的有限 fallback rank（取值 0..n_ch−1）。

**真实差异**：

| 阶段 | 距离 | 用什么通道 |
|---|---|---|
| PR-2 KMeans 聚类 | sklearn KMeans on `(events × n_ch)` 全 rank 矩阵；欧氏距离 sum；`np.where(isfinite, x, 0.0)` 是防御性 no-op（cohort 内 lagPat 全 finite） | **所有 n_ch 个通道**（含非参与通道的 fallback rank） |
| `assign_events_to_templates` 模板匹配 | `np.nansum(sq_diff) / max(n_valid, 1)` over shared channels | **只 bool=True 的共享通道** |

差异的根本来源：
- KMeans 把非参与通道的 fallback rank 也算进事件距离——这些 rank 的具体值由 legacy 老代码决定（`return_massCenterPat` 的填充逻辑）
- matching 完全忽略非参与通道——只看两边都参与的位置

**机制层差异（不止一处）**：
1. **通道集**：KMeans = 全集 `n_ch`，matching = 共享子集
2. **聚合**：KMeans = sum squared Euclidean，matching = mean squared deviation（除以 shared channel 数）
3. **Centroid 来源**：KMeans `cluster_centers_` ≠ legacy `_legacy_hist_mean_rank` 模板（本 PR 用的是后者）
4. **min_shared gate**：matching 有硬门槛（默认 3），KMeans 没有

**实测一致率**：cohort 中位 0.892、最低 0.769（与用户先前 audit 完全一致）。**不推翻** PR-2/2.5/3/4/5/6/7 任何核心结论（这些结论都不依赖 matching metric 单独成立），但 boundary events 的几何解释 metric 选择敏感。

---

## 4. Showcase 三张图

> 每张 = §5.1 4-panel 2×2 放大版（PCA all-events + silhouette + template profile + MDS audit）

### 4.1 `958` — k=2 forward/reverse（老论文 E3 复现）

- n=165577、stable_k=2、sil=0.413、agreement=0.902、stress=0.373
- inter-cluster Spearman r ≈ −0.91（PR-2 archive）
- **Panel c 模板剖面**：cluster 0（蓝）与 cluster 1（赭）在同一通道集合（`G00–GH7`）上 rank 完全反向——间期事件内部有两条互逆主导传播路径；老论文用 KMeans cluster centroid 已经报告，本 PR 在 PCA 二维空间里直接画出 + 在 panel c 用 channel-rank 折线显式
- **Panel a (PCA, all events)** + **Panel d (MDS audit, subsample)**：两种 metric 下都显示两类事件云清晰分开，模板大星各自落在事件云中心
- **Panel b silhouette**：负 silhouette 比例 ≈ **9.8%**（boundary events，落在两类之间的过渡带）

**论文叙事价值**：作为 paper 主图候选；PCA all-events 视图比老论文的 KMeans heatmap 更直观

### 4.2 `818` — k=4 多模态 + 最低 agreement

- n=11337、stable_k=4、sil=0.217、**agreement=0.769（cohort 最低）**、stress=0.481
- 替代原计划 showcase `huangwanling`（被 data freshness 排除）
- **Panel c 模板剖面**：4 条独立 rank pattern——不是简单 forward/reverse，而是 4 种独立主要传播模式
- **Panel a vs d**：4 个事件云在 PCA 视图中相对清晰；MDS 视图中由于 stress=0.481 有更多重叠（k=4 在 2D 必然 distortion）
- **Panel b silhouette**：4 个 cluster 都有显著负 silhouette；总 boundary fraction = **23.1%**（cohort 最高）

**论文叙事价值**：反例——证明 cluster decomposition 不止 k=2，少数 subject 真的有更复杂结构；同时它的高 boundary fraction 就是本 PR 审阅发现的 metric drift 在哪里最严重

### 4.3 `253` — 低 cluster validity caveat（auto-pick）

- n=75053、stable_k=2、sil=0.182（cohort 最低）、agreement=0.785、stress=0.402
- 由 plot 脚本从 `agreement < 0.85 ∩ n_events ≥ 200` 中按 silhouette 升序 auto-pick
- **Panel a/d**：两类事件云重叠严重，模板大星彼此距离不远，没有清晰决策边界
- **Panel b silhouette**：两个 cluster 都有约 1/3 事件 silhouette 接近 0；boundary fraction = **21.5%**
- **Panel c 模板剖面**：两条折线走向接近平行（不是反向也不是错开）——这个 subject 的"两个 cluster"在通道 rank 维度上区分度本身就低

**论文叙事价值**：诚实暴露 cluster decomposition 在该 subject 上较弱；不掩饰 PR-2 KMeans 决策在 metric 选择敏感的 subject

---

## 5. 输出资产清单

```
results/interictal_propagation/cluster_geometry/
├── per_subject/
│   ├── epilepsiae_<X>.json    # 20 个 status="ok"
│   └── yuquan_<X>.json        # 18 个 status="excluded"，excluded_reason 写明
├── cohort_summary.json        # n_subjects_included=20, n_subjects_excluded=18
└── figures/
    ├── README.md              # 中文说明每张图，列出实际 showcase picks
    ├── per_subject/
    │   └── epilepsiae_<X>_geometry.png  # 20 张 2×2
    ├── showcase/
    │   ├── 958_geometry_showcase.png    # k=2 forward/reverse
    │   ├── 818_geometry_showcase.png    # k=4 multimodal
    │   └── 253_geometry_showcase.png    # auto-pick low-validity caveat
    └── cohort_geometry_summary.png      # 2×2 cohort summary
```

代码：

- `src/cluster_geometry.py` — 距离 + classical MDS + **PCA on KMeans feature matrix** + per-event silhouette + cohort aggregator
- `tests/test_cluster_geometry.py` — 12 项 TDD（全 pass）
- `scripts/run_cluster_geometry.py` — batch driver；skipped subject 写入 per-subject JSON 让 cohort_summary 正确反映 excluded
- `scripts/plot_cluster_geometry.py` — figure generator（4-panel 2×2）；JSON NaN→null

文档：

- `docs/archive/topic1/propagation/cluster_geometry_viz_plan_2026-05-06.md` — plan-of-record（已修正 KMeans metric 描述、§5 panel layout = 2×2 + PCA panel、§3.4 flag-and-impute 与代码一致）
- `docs/archive/topic1/propagation/cluster_geometry_viz_results_2026-05-06.md` — 本 results doc（已降调）
- `docs/topic1_within_event_dynamics.md` §3.1d — 主文档 pointer（已降调）
- `CLAUDE.md` §7 — Multi-Panel Figure Discipline（rule-only，已删除特定 incident 描述）

---

## 6. 与既有 PR / 主文档的边界（再次确认）

- **不改 PR-2 / 2.5 / 3 / 4A-D / 5 / 6 / 7 任何已落定结论**
- **不改 Topic 1 主文档的"当前正式口径"**：仅在 §3.1d 加 pointer
- **不替换 PR-3 6-panel cohort 图**：PR-3 是 raw KMeans-feature-space 的统计快照，本工作并列、互不替代
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
| **Excluded subject 写进 cohort_summary** | ✅ | Runner 修订后；cohort_summary.excluded 含 18 Yuquan stub |
| **JSON 是标准 JSON（NaN→null）** | ✅ | `_sanitize_json_floats` + `allow_nan=False` |
| **PCA panel 展示 all events** | ✅ | `compute_pca_embedding` + 主图 panel a |
| 全 cohort 跑通 + figures 出 | ✅ | 20 per-subject ok + 18 excluded stub + 1 cohort + 3 showcase |
| 958 panel c 复现 forward/reverse | ✅ | `figures/showcase/958_geometry_showcase.png` |
| 818 panel c 复现 k=4 多模态 | ✅ | `figures/showcase/818_geometry_showcase.png` |
| 用户先行 audit 数字（median 0.892, min 0.769）匹配 | ✅ | cohort_summary.json 完全一致 |
| Metric drift 集中在低 n_part 事件 | ✅ | boundary fraction 单调下降：0.135 → 0.097 → 0.046 |

---

## 8. 一句话面对外部审稿人 / 论文的口径（已降调）

> Within the included Epilepsiae cohort (20 subjects), the PR-2 KMeans
> cluster decomposition is **mostly concordant** with template-matching
> reassignment (event-level agreement median 0.89, range 0.77–0.96,
> 8/20 below 0.85; cohort silhouette median 0.46 under the matching
> metric, range 0.18–0.67). Disagreement is concentrated at events with
> fewer participating channels (boundary fraction monotone in
> n_participating: 0.13 → 0.10 → 0.05), consistent with the structural
> difference between the two metrics (KMeans uses all n_ch channels with
> their legacy fallback ranks; matching uses only shared participating
> channels). The forward/reverse template structure first reported in
> legacy E3 (subject 958) is directly visible in panel c of the showcase
> figure. A k=4 multimodal subject (818) confirms that the two-template
> compression is not universal. The 18 Yuquan subjects of the original
> 38-subject cohort are excluded due to data-freshness mismatches between
> saved PR-2 labels and the current lagPat data, and require a PR-2 /
> PR-2.5 rerun on the latest pipeline output before they can be added back.
> **No new biological finding is claimed**; this is a descriptive
> visualization supplement to PR-2 / PR-2.5 with an internal-consistency
> caveat documented for follow-up.
