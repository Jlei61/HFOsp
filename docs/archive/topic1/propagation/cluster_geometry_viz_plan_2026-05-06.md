# Topic 1 — Cluster geometry visualization plan

> 状态：plan-of-record（2026-05-06 立项）
> 范围：Topic 1 PR-2 / PR-2.5 cluster decomposition 的描述性可视化补强；**不引入新假设、不改 PR-2/2.5 主结论**。
> 主文档入口：`docs/topic1_within_event_dynamics.md`（§7.7c PR-3 6-panel cohort 之后追加 §7.7e 指针指向本 archive）
> 关联：`docs/archive/topic1/propagation/interictal_group_event_internal_propagation.md`（§7.5b adaptive cluster 是数据来源）

---

## 1. 这次工作回答什么 / 不回答什么

**回答**：每个 subject 的间期事件，在 template-matching 距离空间下展开成 2D 时，是否能直观看到 PR-2 给出的 stable_k 个簇？簇间分离强度（silhouette）在 cohort 上分布如何？KMeans label 与 template-matching reassign 的一致率分布如何？label drift 是否集中在低 n_participating 事件上？

**不回答**（明确 out-of-scope）：
- 不重新聚类、不替换 PR-2 给出的 `adaptive_cluster.labels`
- 不引入"是否真有两类"的新统计假设——PR-2 已用 dip test + AMI + silhouette 给过判定
- 不深挖 KMeans (NaN→0 Euclidean) vs template-matching (masked mean squared deviation) 两 metric 的语义差异（仅记录现象，列为后续工作）
- 不做 cross-subject 投影（通道集合按 subject 分，跨 subject 投影需要专门通道对齐工作，本 PR 拒绝）
- 不做 centered-rank 投影（只画 raw rank，与 KMeans feature space 一致；centered 版本属于"identity-bias 之后传播是否真实"的另一个问题）

---

## 2. 科学动机 + 审阅发现

PR-2 用 KMeans 在 `(events × n_ch)` rank 矩阵上做了 stable_k 聚类；PR-2.5 用 split-half / odd-even block 验证了模板跨时间稳定性。当前论文叙事说"主要由两个传播模板压缩"——但仓库里**没有任何一张图**直接展示"这两类簇在距离空间里是否真的分得开"。读者只能信 silhouette / AMI 数字，不能看图核对。

审阅期间还暴露了一个内部不一致：

| 阶段 | 距离 |
|---|---|
| PR-2 KMeans 聚类 | `np.where(isfinite, rank, 0.0)` 后做 sklearn KMeans → **NaN→0 后的欧氏距离 sum**（非参与通道当 rank=0） |
| `assign_events_to_templates` 模板匹配 | `masked_ranks = np.where(bools>0, ranks, NaN)` → **shared-channel mean squared deviation**（只在两边都参与的通道上算） |

实证核对（用户先行验证）：用当前全局模板把事件重新 assign 回模板，与原始 KMeans label 的一致率 cohort 中位 **0.892**，最低 **0.769**。这说明 metric 不一致**会**影响边界事件和少数 subject 的图形解读，但 Topic 1 主结论仍稳定（stable_k、within-cluster τ、PR-2.5 split-half 模板 Spearman 都不依赖这个 masked metric 单独成立）。

本 PR 的两件事：
1. **主交付物**：在 template-matching metric 下做 cluster geometry 可视化（per-subject + cohort + showcase）
2. **审阅落档**：把 metric drift 现象作为 audit finding 写进 archive §6，列为后续独立工作；本 PR 不深挖

---

## 3. 数据合同

### 3.1 输入

每个 subject 复用既有产出，**不重跑** PR-2/2.5 任何步骤：

- `load_subject_propagation_events(subject_dir)` → `ranks`, `bools`, `channel_names`, `event_abs_times`, `block_ids`, `block_time_ranges`
- `pr1_subject_summary.json` 里的 `adaptive_cluster`：
  - `labels`（valid_events 上的 cluster id）
  - `clusters[k].template_rank`（argsort-of-argsort 整数）
  - `clusters[k].n_events`, `clusters[k].fraction`
  - `chosen_k`, `stable_k`
- `valid_events`：`_valid_event_indices(bools, min_participating=3)` 重算（与 PR-2 一致）

### 3.2 模板重建

不直接吃 `template_rank` 做距离运算（template_rank 是整数 rank，把非参与通道也做了 fallback fill）。改为**复用 `build_cluster_templates`**：

```python
templates_real = build_cluster_templates(ranks, bools, labels, chosen_k)
# templates_real shape: (chosen_k, n_ch), 非参与通道 = NaN
```

这给出的是 mean-rank-vector 模板，非参与通道保留 NaN —— 这是和 `assign_events_to_templates` 完全配对的输入。

### 3.3 距离 metric（**locked**，三处共用）

```
d(x, y) = sqrt( mean over channels c where bool_x[c]=1 AND bool_y[c]=1 of (rank_x[c] - rank_y[c])² )

# 模板的 bool: bool_T[c] = 1 当且仅当 templates_real[k, c] is finite
# 若 #shared channels < min_shared_channels (默认 3)：d = NaN（pair 被剔除）
```

适用：
- 事件 ↔ 事件
- 事件 ↔ 模板
- 模板 ↔ 模板

注意：
- 这是 mean squared deviation 的 sqrt（不是 sum）。和 `assign_events_to_templates:1656` 用的 `nansum / max(n_valid, 1)` 一致后开方
- min_shared_channels 默认 3，与 PR-2 的 `_valid_event_indices` 一致；可配置
- 距离对称、非负、自距 = 0、不必满足三角不等式（masked metric 一般不满足，MDS 上是已知现象，需在文档里明示）

### 3.4 二维投影

**Classical MDS** on 增广 `(E + k) × (E + k)` 距离矩阵（事件 + 模板都参与）。算法选择：

- 距离矩阵对角线强制为 0（任何点到自身距离 = 0）
- 缺失 off-diagonal pair（`d = NaN`，由 `min_shared_channels` gate 触发）的 imputation 规则：
  1. 先剔除 `valid_distance_fraction < 0.5` 的整行/整列对应的 events（这些 events 共享通道太稀，无法稳定嵌入；写入 `excluded_events` 列表）
  2. 剩下矩阵里仍有 NaN 的，用**中位 pairwise 距离**替代（cohort 内每 subject 自己算自己的中位数；不用全局），中位数比均值对异常稀疏 pair 不敏感
- 输出 2D 坐标 `Y` 形状 `(E + k, 2)`，其中前 E 行是事件、后 k 行是模板
- **Stress** 报告：`stress = sqrt(sum((D - D_Y)²) / sum(D²))`，其中 `D_Y` 是 2D 重建距离；`D` 用 imputation 后的矩阵计算（与 MDS 内部一致）；记入 cohort summary
- **Reproducibility**：classical MDS 是 closed-form，无随机种子依赖；唯一的随机性来自 §3.5 subsample 步骤（已 seed）

**为什么 classical MDS 而不是 SMACOF / UMAP / t-SNE**：
- classical MDS 给出与距离一致的 closed-form 解，可重复、无随机种子依赖
- SMACOF 是迭代版，本 PR 距离矩阵规模够小（E+k ≤ 1.5 万）+ 缺失值少（min_shared 已剔除），cmdscale 已足够；如某 subject stress > 0.3 才考虑切换 SMACOF
- 不用 UMAP / t-SNE：参考 §1 已说明（"漂亮分离"和距离 / 聚类决策不直接绑定，且模板 out-of-sample 投影非平凡）

### 3.5 内存预算

最大 subject 大概 1.5 万 events。`(E+k)² × 8 bytes ≈ 1.8 GB` —— 紧。两条防线：
- subject n_events > **N_MAX = 8000**：随机降采样到 8000 events + 全部模板进入主图（subsample 用 `np.random.default_rng(seed=0)`，固定 seed 保证 cohort 内可重复；选择策略：完全均匀随机，不分层）；附录单独存"全集 silhouette / agreement"数字（不画 MDS）
- 距离矩阵用 float32 不用 float64，省一半内存（精度对 MDS 足够）

---

## 4. Per-event readout 定义

每个 valid event `i` 输出：

| 字段 | 定义 |
|---|---|
| `kmeans_label` | PR-2 的 `adaptive_cluster.labels[i]` |
| `matching_label` | `assign_events_to_templates(ranks, bools, templates_real)` 给出的最近模板 id |
| `agreement` | `kmeans_label == matching_label`（bool） |
| `d_within` | `d(event_i, templates_real[kmeans_label[i]])` |
| `d_min_other` | `min over k' ≠ kmeans_label[i]: d(event_i, templates_real[k'])` |
| `silhouette` | `(d_min_other - d_within) / max(d_within, d_min_other)`，缺失 → NaN |
| `n_participating` | `bools[:, i].sum()` |
| `mds_x`, `mds_y` | classical MDS 二维坐标 |
| `block_id`, `event_abs_time` | 跟 PR-4D 一样附带，供未来 cross-link |

Silhouette 这里用的是 **template-prototype 版本**（每事件比对自己模板 vs 最近他模板），而**不是** sklearn 的 sample-to-cluster 平均距离版本。理由：原 sklearn silhouette 在 (NaN→0 KMeans labels, masked distance) 混合下的语义不干净；prototype 版本和"事件离自己模板有多近 / 离别模板有多远"的物理图像直接对应，正好回答 cluster validity 问题。

---

## 5. Panel 设计（Plan A'，每 panel 答一个独立问题）

### 5.1 Per-subject（30 张，1×3 行；输出 `figures/per_subject/<subject>_geometry.png`）

| Panel | 答的问题 | 内容 |
|---|---|---|
| **a** | 事件在距离空间长什么样？哪些事件 KMeans/matching 不一致？ | MDS scatter；事件按 `kmeans_label` 上色（Morandi blue/rust/sage…，按 stable_k 取调色板）；模板用大星黑边突出；`agreement = False` 事件用空心圆叠加；title 写 `subject n=… k=… sil_med=… stress=…` |
| **b** | 簇间分离的数值有多强？哪些事件归属可疑？ | Per-event silhouette 排序条；x = event index re-ordered 按 cluster + cluster 内按 sil 降序；y = silhouette；负 sil 用 Morandi rust 高亮；cluster 边界 vertical separator |
| **c** | 两类簇结构是什么意思？哪些通道早 / 晚？ | Cluster template profile：x = channel idx（**按 dominant cluster（n_events 最大那个）的 template rank 排序；并列时按 cluster_id 字典序**，cohort 内每 subject 自己定），y = template rank；每 cluster 一条折线 + 半透明 IQR 带（IQR 由 cluster 内事件的 rank 在每通道上的 25/75 分位计算）；非参与通道（template = NaN）的通道在该 cluster 折线上断开（不连线） |

### 5.2 Cohort summary（1 张，2×2；输出 `figures/cohort_geometry_summary.png`）

| Panel | 答的问题 | 内容 |
|---|---|---|
| **a** | Cohort 上每 subject 的 cluster validity（template-matching metric 下）？ | Per-subject silhouette median 排序 bar；dataset 上色（YQ blue / EPI terracotta）；shape: stable_k=2 圆 / 4 三角 / 6 方 |
| **b** | KMeans/matching 一致率每 subject 多少？ | Per-subject agreement ranked bar（subject 顺序与 a 不同，按 agreement 排）；< 0.85 用 Morandi rust 高亮 |
| **c** | silhouette 和 agreement 是否相关？ | Joint scatter `silhouette_med vs agreement`；marker shape = stable_k；Spearman ρ + p 在角落；颜色按 dataset |
| **d** | metric drift 主要发生在低 n_participating 事件上吗？ | Cohort-level boundary-event fraction（agreement = False 的事件比例），按 n_participating bin（3-4 / 5-8 / 9+）做 violin + scatter；cohort 内每 subject 一个数据点 |

### 5.3 Showcase（3 张大图；输出 `figures/showcase/<subject>_geometry_showcase.png`）

每张 = 单个 subject 的 §5.1 三 panel 放大版，加更密的轴标 + 模板形状注释。

候选：
- **`958`**：k=2、inter-cluster r = −0.91，老论文 E3 的 forward/reverse 复现，作为"教科书 case"
- **`huangwanling`**：k=4、raw_tau ≈ centered_tau（identity bias = 0），多模态最干净
- **第三个 subject 在 cohort 跑完后选**：取 `agreement < 0.80 ∩ n_events ≥ 200` 中 silhouette 最低的一个（突出 metric drift caveat 的真实案例）

---

## 6. Audit finding：KMeans vs template-matching label drift

**写入本 archive §6，作为可追溯档案；不在本 PR 深挖。**

实证核对结果（用户先行）：
- Cohort 中位一致率 = 0.892
- 最低一致率 = 0.769
- **未推翻** PR-2 / PR-2.5 主结论（stable_k、within-cluster τ、split-half 模板 Spearman matching、PR-5-A 的 best-template r / reconstruction error gate 都不依赖该 masked metric 单独成立）

机制层差异（清单，留给后续工作）：
1. **非参与通道处理**：KMeans 把 NaN→0 当 rank=0（feature 长度齐）；matching 忽略非参与通道（只在 shared 上计算）。低 n_participating 事件下两者差异最大。
2. **聚合方式**：KMeans 是 sum squared Euclidean（centroid 优化目标）；matching 是 mean squared deviation（除以 shared channel 数）。
3. **Centroid 来源**：KMeans 输出的 `cluster_centers_` 是 sklearn 算的；本 PR 用的 template 是 `_legacy_hist_mean_rank` 的 3-bin top-rank 模板（来自 legacy 老代码合同）——两者不是同一对象。
4. **min_shared_channels gate**：matching 有 `n_valid >= min_shared_channels` 硬门槛（默认 3），低 n_part 事件可能直接被 assign 成 −1；KMeans 阶段没有此 gate（feature 已被 NaN→0 强制成完整长度）。

后续工作（**本 PR 不做**，留作独立 follow-up）：
- 量化"哪些机制层差异贡献最多 disagreement"
- 评估：是否应统一用 masked metric 重做 KMeans（用 SMACOF MDS 嵌入做 KMeans，或直接 k-medoids on masked distance）
- 对低 agreement subject（如 < 0.85）做 case-by-case 检查，看是否影响 PR-2.5 / PR-3 / PR-4 结果解读
- 如果统一 metric 下 stable_k 仍然稳定，考虑写一个 v2 版 PR-2 替换原 KMeans（这一步影响面大，要谨慎）

---

## 7. 实现合同

### 7.1 代码入口

新建：

- `src/cluster_geometry.py`
  - `compute_masked_distance(x_rank, y_rank, x_bool, y_bool, min_shared) -> float` — 单 pair
  - `compute_event_template_distances(ranks, bools, templates_real, min_shared) -> np.ndarray` — `(n_events, n_clusters)` 矩阵；模板 NaN 当 `bool=0`
  - `compute_event_event_distances(ranks, bools, indices, min_shared) -> np.ndarray` — `(n_indices, n_indices)` 对称矩阵
  - `compute_augmented_distance_matrix(ranks, bools, templates_real, valid_events, min_shared) -> (D, n_events_used)` — `(E+k, E+k)`
  - `classical_mds(D, n_components=2) -> (Y, stress, eigenvalues)` — cmdscale 实现
  - `compute_per_event_silhouette(d_within, d_min_other) -> np.ndarray`
  - `compute_subject_geometry(subject_data, adaptive_cluster, min_shared, max_events_for_mds) -> dict` — 主入口，返回 §4 整套字段 + cohort 输入需要的 summary
  - `summarize_cohort_geometry(per_subject_results) -> dict` — cohort summary aggregator

- `scripts/run_cluster_geometry.py`
  - CLI: `--dataset {yuquan,epilepsiae,both}` `--subject <name>` `--all` `--dry-run` `--max-events <int>` `--min-shared <int>`
  - 每 subject 输出 `results/interictal_propagation/cluster_geometry/per_subject/<subject>_geometry.json`
  - cohort 写 `results/interictal_propagation/cluster_geometry/cohort_summary.json`

- `scripts/plot_cluster_geometry.py`
  - CLI: `--per-subject` `--cohort` `--showcase` `--subjects <list>`
  - 复用 `src/plot_style.py`（Morandi 调色板、`style_panel`、`add_significance_bracket`、`FS_*`、`DPI_PUB`）

- `tests/test_cluster_geometry.py`（TDD，§7.3）

### 7.2 输出契约

```
results/interictal_propagation/cluster_geometry/
├── per_subject/
│   └── <subject>_geometry.json     # 每 subject 一个；schema 见下
├── cohort_summary.json
└── figures/
    ├── README.md                   # 中文，每张图描述
    ├── per_subject/
    │   └── <subject>_geometry.png
    ├── showcase/
    │   ├── 958_geometry_showcase.png
    │   ├── huangwanling_geometry_showcase.png
    │   └── <selected_low_agreement>_geometry_showcase.png
    └── cohort_geometry_summary.png
```

Per-subject JSON schema（key 字段）：

```json
{
  "subject": "<name>",
  "dataset": "yuquan|epilepsiae",
  "stable_k": 2,
  "n_events_total": 1234,
  "n_events_used_for_mds": 1234,
  "subsampled": false,
  "stress": 0.123,
  "silhouette_median": 0.45,
  "silhouette_iqr": [0.30, 0.58],
  "agreement_overall": 0.892,
  "templates_real_finite_channels": [[...], [...]],
  "boundary_fraction_by_nparticipating": {"3-4": 0.18, "5-8": 0.07, "9+": 0.02},
  "events": [
    {"event_idx": 0, "kmeans_label": 0, "matching_label": 0, "agreement": true,
     "d_within": 1.2, "d_min_other": 2.4, "silhouette": 0.5,
     "n_participating": 5, "mds_x": -0.3, "mds_y": 0.2,
     "block_id": 0, "event_abs_time": 1234567.0}
  ],
  "templates_mds": [{"cluster_id": 0, "mds_x": ..., "mds_y": ...}, ...]
}
```

Cohort summary schema：

```json
{
  "n_subjects": 30,
  "per_subject": {
    "<name>": {"silhouette_median": ..., "agreement": ..., "stable_k": ..., "stress": ..., "n_events": ...}
  },
  "joint": {"spearman_silhouette_agreement": {"r": ..., "p": ...}},
  "boundary_fraction_by_nparticipating": {"3-4": [..., ...], "5-8": [...], "9+": [...]},
  "showcase_selected": {"958": "classic_forward_reverse", "huangwanling": "k4_clean", "<X>": "low_agreement_caveat"}
}
```

### 7.3 TDD 合同（≥ 8 项）

| # | 测试 | 失败信号 |
|---|---|---|
| 1 | `compute_masked_distance` 自距 = 0、对称、非负 | 距离公式实现错 |
| 2 | `compute_masked_distance` shared < min_shared → NaN | gate 漏写 |
| 3 | `compute_masked_distance` 与 `assign_events_to_templates` 在同一 (event, template) pair 上一致（`d²` vs mean squared 一致） | metric drift 引入 |
| 4 | `classical_mds` 在 toy 数据上 reconstruction error < 1e-10 | cmdscale 实现错 |
| 5 | `classical_mds` 输出 shape 正确、stress 在 [0,1] | shape/normalization bug |
| 6 | k=2 anti-correlated 模板（手工构造 inter-cluster Spearman r=-1）→ MDS 上两模板坐标的连线穿过事件云中心 | 模板与事件共空间投影错 |
| 7 | `compute_per_event_silhouette` 边界：`d_within=d_min_other` → silhouette = 0 | 公式实现错 |
| 8 | `compute_subject_geometry` 在 toy 数据上 agreement_overall = 1.0（手工保证 KMeans label 与 matching 等价） | pipeline 端到端 wiring 错 |
| 9 | n_events > max_events 时，subsampled=True 且模板坐标未被随机降采样剔除 | subsample 逻辑漏掉模板 |
| 10 | cohort summary aggregator：subject 缺失字段时不 crash，记录 `excluded` | 健壮性 |

### 7.4 失败合同

- **`stable_k=1` 或 `chosen_k != stable_k`（PR-2 fallback case）**：跳过该 subject、写 `excluded: "no_stable_k"`。当前 cohort 30/30 都有 stable_k，此分支理论上不会触发，但保留入口
- **某 cluster 模板向量整体全 NaN（该 cluster 所有事件在所有通道上都没参与）**：该 subject 退出，`excluded: "all_nan_template"`，cluster_id 写入 `failed_cluster`。这是退出条件，不是 warning。
- **某 cluster 部分通道 NaN（正常情况，参与稀疏）**：保留该 cluster；matching 距离用 shared-channel mean（如果某事件与该模板 shared < min_shared，该 (event, cluster) pair 距离 = NaN，但不影响其他 pair）
- **MDS 距离矩阵中 imputed pair 比例 > 20%**：仍出图但 cohort summary 写 `mds_imputation_warning: True`；该 subject 在 figure title 标黄色注释
- **MDS stress > 0.3**：仍出图但 figure title 标黄色 `WARN stress=...`；cohort summary 单列 `subjects_high_stress`
- **n_events_total < 50**：跳过该 subject、`excluded: "too_few_events"`
- **测试任意一项失败**：不允许进 cohort 跑

---

## 8. 与既有 PR / 主文档的边界

- **不改 PR-2 / 2.5 / 3 / 4A-D / 5 / 6 / 7 任何已落定结论**
- **不改 Topic 1 主文档的"当前正式口径"**：只在 §7.7c 后追加 `§7.7e — Cluster geometry visualization (template-matching metric, 2026-05-06)：指针 → 本 archive`
- **不替换 PR-3 6-panel cohort 图**：PR-3 是 raw KMeans-feature-space 的统计快照，本 PR 是 template-matching metric 下的几何展开，两者并列、互不替代
- **不进入 SBA framework P1–P5 任何 prediction 池**：本 PR 是描述性 supplement，不拆 H1/H2

---

## 9. 工作量 / 优先级

| 阶段 | 估计 |
|---|---|
| `src/cluster_geometry.py` + 测试（TDD） | 0.5 d |
| `scripts/run_cluster_geometry.py` | 0.25 d |
| `scripts/plot_cluster_geometry.py`（per-subject + cohort + showcase） | 1 d |
| Smoke test 1 subject + 调图 | 0.25 d |
| 全 cohort 跑（30 subjects） | < 1 h |
| 写 figures/README.md + Topic 1 主文档 §7.7e 指针 | 0.25 d |
| **总计** | **2.25 d** |

优先级 **P1**（Topic 1 描述性补强；不阻塞任何主线 PR）。

---

## 10. Plan-of-record 锚点

本 plan 一旦落盘，metric / panel layout / failure 合同 **不再修改**。如执行中发现需要变更，新增 `<plan>_addendum_<date>.md` 旁挂归档；本文件保持原状。
