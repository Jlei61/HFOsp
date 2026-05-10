# Topic5 PR-1 — Per-subject seizure subtyping with z-ER tensor (cohort 2026-05-10)

> **状态**：探索性 (exploratory)，2026-05-10 audit-rerun **完成**版本（第一份归档）。
> **范围**：每 subject 内部，把多个 ictal seizure 按 (channel × time-bin) z-ER 张量
> 表征聚类，并在视觉 + sentinel 层面验证算法捕获到的 within-subject 异质性。
> **不属于**：cross-subject seizure-type 标准化（cohort 内每 subject 独立聚类）；
> pre-ictal / outcome 关联（留 topic5 PR-2+）。

---

## 1. 动机与生物先验

v2.3 ictal ER timing atlas（topic5 PR-0）按 user 视觉巡视
（442 sz=9, 548 {13,14,24}, 916 {21,23,25}, 1077 sz=1）暴露 within-subject
seizure pattern 的非平凡异质性。把所有 seizure 平均成单一"主模式"
是一种方法学缺陷——下游 pre-ictal / outcome 分析必须先把异质性 carve out。

**生物先验**：Schroeder et al. 2020 *PNAS*（Newcastle Wang/Taylor lab）
在多 subject SEEG 上记录到 within-patient seizure pathway variability，
支持 "同一 patient 不同 seizure 走不同路径" 的存在性。本 PR **不直接复现**
Schroeder 的 dynamic-time-warping pipeline；只借用先验。

---

## 2. 方法（实现细节见 `src/`）

### 2.1 Feature：z-ER 张量

- 来源：`src/ictal_onset_extraction.py::compute_er` (fast/slow band amplitude
  ratio) + `baseline_zscore_er` (per-channel pre-onset baseline z-score)
- 张量形状：`(n_channels, n_time_frames)`，每帧 hop=0.1s，window=1.0s
- 时间分箱：5 个 bin `[(-200,-50), (-50,0), (0,50), (50,150), (150,200)]` s
  rel. clinical onset（`src/ictal_zer_features.py::DEFAULT_BINS_SEC`）
- 每 channel × bin → `nanmean(z_ER over frames in bin)`
- Flatten 成 `(n_channels × 5, n_seizures)` 矩阵（channel-major）
- **bands**：`gamma_ER` + `broad_ER` 各跑一次（`GAMMA_ER_BANDS` / `BROAD_ER_BANDS`）

### 2.2 Distance + Linkage

- Pairwise dissimilarity：`1 − Spearman ρ` over channel-bin features，
  pair-overlap 守门 `min_overlap = 5`
- Linkage：UPGMA (`scipy.cluster.hierarchy.linkage(method="average")`)
- k 选择：`silhouette_score`（precomputed dist），守门 `min_cluster_size`：
  至少 1 个 cluster size ≥ `min_subtype_size = 2`
- Outlier vs subtype split：cluster size < min_subtype_size → `outlier_flag=True`
  + `subtype_label = -1`；剩下按 size 降序重命名 0..k-1

### 2.3 Permutation null（gap_perm_k）

- `src/ictal_seizure_clustering.py::channelwise_permutation_null`
- 对每 channel-block（5 bins）独立 sample 一个 seizure index 排列，
  把这 5 行整体按该排列重排——**保留 within-channel multi-bin 协方差**
  （t_onset 路径用 per-row finite-value shuffle，仍兼容；通过 `bins_per_channel=1`）
- 重排后重算 D + within-cluster dispersion；
  `gap = median(log(WD_perm)) − log(WD_obs)`
- B = 50

### 2.4 Sentinel sanity（D3）

`scripts/cluster_ictal_seizures.py::FROZEN_SENTINEL` 4 个 case：
442 [9], 548 [13,14,24,25], 916 [21,23,25], 1077 [1]。每个 case 报：

- `outlier_jaccard` = J(user-marked outliers, algo singleton outliers)
- `subtype_jaccard` = J(user main set, algo largest subtype)
- `minority_jaccard` = J(user outliers, algo minority = anyone NOT in largest)

明示：sentinel 是 sanity，不是 independent validation。

---

## 3. Audit (2026-05-10) bug list & 修复

### Bug 1：`channelwise_permutation_null` 在 z-ER 路径破坏 5-bin 协方差

- **症状**：per-row shuffle 把同一 channel 的 5 个 bin 视作独立行各自换 seizure，
  破坏 within-channel multi-bin coupling
- **影响**：`gap_perm_k` 数值偏移方向不可预先解析，可能高估或低估 cluster
  vs null 紧密度
- **实证**（cohort 28 个 ok rows）：Δgap_perm 中位 −0.0007、abs_max 0.061、
  0 个 `|Δ|` > 0.10；`chosen_k`/`over_split_flag`/sentinel jaccard 全 0 flips
  —— bug 真实但本 cohort 上效应几乎为零（详见 §4.4）
- **修复**：`bins_per_channel` 参数；`>1` 时走 channel-block coherent shuffle
- **测试**：3 个新 TDD case（参数验证 + block shuffle 正确性 + per-row vs per-block 给不同 gap）

### Bug 2：`extract_zer_binned_for_subject` 缺少 channel-order check

- **症状**：每 seizure 单独走 z-ER 提取流程，downstream `stack_features_to_matrix`
  只 check feature 长度不 check name；某次 extraction 通道顺序漂了 → 静默污染
- **影响**：未发现实际 cohort 受影响（手动 audit 4 个 subject 显示 channel order 一致）
- **修复**：每次成功 extraction 严格 equality check 与首个；不一致 →
  `drop_reasons[i] = "channel_order_mismatch"` + `features[i] = None`
- **测试**：1 个 monkeypatch TDD case 注入第 3 个 seizure 通道顺序错位

### Bug 3：`over_split_flag` 旧 OR 规则过严

- **症状**：`sil < 0.2 OR n_subtypes/n_eff > 0.4` 在 cohort 上产 3/32 True，
  目视全部为真子型（548 gamma k=7, 548 broad k=3, 1146 broad k=3）
- **诊断**：silhouette 绝对阈值在高维 Spearman 距离空间下不可比
  （高维 Spearman 距离普遍被压缩到 ~0.5-1.0 区间，silhouette 天然偏低）
- **修复**：改 `over_split_flag = (gap_perm_k < 0.10) AND (n_subtypes / n_eff > 0.5)`
  —— gap_perm 是 null-relative 量跨 feature 可比，AND 要求 "弱 vs null + 真切碎"
  双满足
- **测试**：复用现有 unit test；audit-rerun 后 cohort `over_split_flag` 全 False

---

## 4. Cohort 数值（最终版，2026-05-10 audit-rerun 完成）

cohort z-ER audit-rerun 完成时间：2026-05-10 16:21（log
`results/run_logs/cohort_zer_audit_20260510_1045.log`）。CSV：`cohort_summary__zer_binned.csv`
（32 rows）；pre-audit 快照保留为 `cohort_summary__zer_binned__pre_audit_2026-05-10.csv`。

### 4.1 规模

- **16 subjects**（全 epilepsiae，无 yuquan v2.3 atlas）
- **32 subject-band rows**：28 ok / 4 insufficient_n
  - insufficient_n: `epilepsiae/1077` (gamma+broad), `epilepsiae/139` (broad), `epilepsiae/916` (broad)
- 0 subject failures
- 0 `channel_order_mismatch` drop（新加的 channel-order check 全 cohort 通过）

### 4.2 聚类质量分布（n=28 ok rows）

| 指标 | 中位 | 范围 |
|---|---|---|
| n_eff | 12.5 | [5, 40] |
| silhouette_k | **0.418** | [0.128, 0.573] |
| gap_perm_k (channel-block null) | **0.325** | [0.094, 0.737] |
| n_subtypes | 2.0 | [1, 5] |
| n_outliers | 0 | [0, 2] |

**20/28 (71%) subject-bands 找到 ≥2 morphological subtypes**。

### 4.3 `over_split_flag` 分布

- AND 规则 `gap_perm < 0.10 AND ratio > 0.5` cohort 命中 **0/32**
- 旧 OR 规则 `sil < 0.2 OR ratio > 0.4` 在 pre-audit 快照命中 3/32（548 gamma + 548 broad + 1146 broad），全部目视裁定为真子型

### 4.4 Bug-fix 实测影响（pre-audit vs post-audit）

| 指标 | 数值 |
|---|---|
| n compared (ok rows in both) | 28 |
| Δgap_perm 中位 | **−0.0007** |
| Δgap_perm 均值 | −0.0005 |
| Δgap_perm 绝对值最大 | 0.061 |
| n with `|Δ|` > 0.10 | **0** |
| `over_split_flag` flips | **0** |
| sentinel jaccard 变化 | **0** |

**结论**：channel-block null 的 bug 真实存在，但本 cohort 上对 gap_perm 数值
的影响 ≤ 0.06；对 cohort-level 决策（chosen_k、over_split、sentinel jaccard）
影响为零。Audit-rerun 数值是公开报告版，pre-audit 数值仅作回归对比保留。

### 4.5 Sentinel jaccards（不变，audit-rerun 后与 pre-audit 完全一致）

| Subject / Band | minority_jaccard | outlier_jaccard | subtype_jaccard |
|---|---|---|---|
| 442 / gamma | 1.00 | 1.00 | 1.00 |
| 442 / broad | 1.00 | 1.00 | 1.00 |
| 548 / gamma | 0.25 | 0.00 | 0.44 |
| 548 / broad | 0.43 | 0.00 | 0.79 |
| 916 / gamma | 0.00 | 1.00 | 0.85 (degenerate — user picks all filtered by status) |
| 1077 (both) | — | — | — (insufficient_n) |
| 916 / broad | — | — | — (insufficient_n) |

---

## 5. Sentinel 视觉裁定结论（用户 2026-05-09 / 2026-05-10）

| Subject | gamma_ER | broad_ER | 视觉裁定 |
|---|---|---|---|
| 442 (user=[9]) | k=2, sz=9 outlier | k=2, sz=9 outlier | **最干净 sentinel**：两 band 都把 sz=9 单列，与目视一致 |
| 1146 broad k=3 | — | k=3 (n_eff=7) sizes 3+2+2 | **教科书级 3 子型分离**，旧 OR 规则误报 over_split |
| 548 broad k=3 | — | minority=[13,14,16,20,23,24,26] 含 user-marked | **基本合理**，user [13,14,24] 落同一 minority 家族 |
| 548 gamma k=7 | k=5+2outlier，subtype 2=[13,14,26], subtype 4=[23,24] | — | **high-heterogeneity / fine subdivision candidate**：算法把 user [13,14,24] 拆成两个邻近族；不是过切但也不能确认真 7 类，需要 sensitivity |
| 916 (user=[21,23,25]) | k=2 | insufficient_n | **不能作为 sentinel**：user 标的 3 个全被 v2.3 status filter 过滤掉 |
| 1077 (user=[1]) | insufficient_n | insufficient_n | **不能作为 sentinel**：n_ok=3 < 5 |

**有效 sentinel = 442 + 548**（4 个 subject-band 全 recall=100%）。
542 gamma 的 fine subdivision 需要后续 sensitivity（min_subtype_size=3 /
不同 bin 设计 / common channel mask / bootstrap stability）才能 commit。

---

## 6. Per-subject 视觉骨架（PNG）

`results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/figures/per_subject/`
共 16 张 4-panel PNG。每张布局：

- 左：dendrogram (UPGMA on 1−Spearman) + sorted pairwise heatmap，行/列按
  dendrogram leaf 顺序，tick label 颜色 = subtype 颜色
- 右上：MDS 2D 散点 colored by subtype（outlier 用 X marker + 灰色），
  标题含 `k / sil / gap`
- 右下：cluster-stratified t_ER_onset matrix（rows=channel 按主子型 t_onset 升序，
  cols=seizure 按 subtype 分组，黑竖线分隔）

PNG 在 audit-rerun 完成后会重新渲染（标题里的 `gap=...` 数字会更新）。

诊断 grid（`figures/diagnostic/`）：6 张以 v2.3 atlas per-seizure thumbnail
为单元的 cluster-grouped grid，用于 "算法子型在视觉上互相一致 / outliers 真异类"
的目视裁定。详细每张含义见 `figures/README.md`。

---

## 7. `over_split_flag` 规则演化

| 版本 | 规则 | cohort 命中 | 视觉一致 |
|---|---|---|---|
| v1（旧） | `sil<0.2 OR ratio>0.4` | 3/32 (548×2 + 1146 broad) | ✗ 全部 false-positive |
| v2（当前） | `(gap_perm < 0.10) AND (ratio > 0.5)` | 0/32 (audit-rerun 待回填) | ✓ 与目视一致 |

**警告**：v2 规则的 gap_perm 输入必须来自 channel-block null（修复后）。
旧 per-row null 给出的 gap_perm 不能直接套 v2 规则。

`over_split_flag` 是 **descriptive post-hoc flag**，**不是过切检验**。
真正的过切判定需要：gap_perm（正确 null）+ 视觉 diagnostic + sensitivity 三方一致。

---

## 8. 已知 caveat

1. **gap_perm 重跑 vs 视觉论证**：cohort 大部分 subject gap_perm 数值小幅
   shifts（≤ 0.1）；over_split 命中率不变。但严格意义上"cohort-level 真子型断言"
   仍需要 sensitivity。
2. **共同 channel mask 缺失**：z-ER feature 用 nanmean per bin，不同 seizure 缺
   channel 走 NaN。channel coverage 高的 subject 基本无害，低 coverage subject 应
   做 sensitivity。
3. **916 / 1077 sentinel 失效**：是 v2.3 status filter / n_ok 门的副产物，
   不是聚类失败；不能作 recall/precision evidence。
4. **548 gamma fine subdivision 不能确认真 7 类**：需要 sensitivity battery
   (min_subtype_size=3 / 不同 bin / bootstrap stability) 才能 commit。
5. **Schroeder 2020 是先验，不是 pipeline 复现**：本聚类管道 (1−Spearman + UPGMA
   + silhouette + permutation null + outlier split) 全部本项目实现。
6. **Yuquan 缺席**：当前 cohort 16 个全是 epilepsiae，因为 yuquan v2.3 atlas 还
   没建。下一轮 cohort（含 yuquan）完成后回来更新。

---

## 9. 下游 PR 必须遵守

1. **Per-subtype 不 per-subject**：`result["per_band"][band]["subtype_label"]`
   是先验分组依据，下游 pre-ictal / outcome / propagation metric 必须按 subtype
   分别统计，不能在 subject 内对所有 seizure 平均。
2. **`subtype_size < 3` 处理规则**：cohort 有不少 size=2 的 subtype，
   小子型 statistical power 不足。下游分析需在自己的 README 标处理规则
   （pool with annotation / drop / 全 cohort CI）。
3. **JSON schema**：`results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/per_subject/<sid>__zer_binned.json`，
   含 `per_band[band].{subtype_label, outlier_flag, n_subtypes, subtype_sizes,
   centroids, gap_perm_k, over_split_flag, ...}`
4. **CSV**：`cohort_summary__zer_binned.csv`（32 subject-band rows，含
   `over_split_flag` 列与 `sentinel_minority_jaccard`）
5. **t_onset feature 已被 z-ER 取代**：`per_subject/*.json`（无 `__zer_binned`
   后缀）保留为历史归档，不再作为 PR-2+ 的 subtype 来源

---

## 10. 文件清单

### 代码

- `src/ictal_zer_features.py` — z-ER tensor extraction + binning + channel-order check
- `src/ictal_seizure_clustering.py` — pairwise dissim, UPGMA, k selection, channel(-block) permutation null, outlier/subtype split, sentinel jaccard, EEG-realign helpers, PR-1 propagation template match
- `src/ictal_seizure_plotting.py` — MDS, subtype color palette, sort orders
- `scripts/cluster_ictal_seizures.py` — CLI driver `per-subject / cohort / render`
- `scripts/diagnostic_cluster_grid.py` — cluster-grouped per-seizure thumbnail grid

### 测试

- `tests/test_ictal_seizure_clustering.py` — 33 tests (helpers + orchestrators + permutation null + bins_per_channel + ARI)
- `tests/test_ictal_zer_features.py` — 5 tests (binning + stack + channel-order check)
- `tests/test_ictal_seizure_plotting.py` — 8 tests (sort + MDS + palette + channel sort)
- 总计 **45 tests pass**

### 数据 / 图

- `results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/`
  - `per_subject/*__zer_binned.json` — 16 cohort z-ER cluster results
  - `cohort_summary__zer_binned.csv` — 32 subject-band rows
  - `cohort_summary__zer_binned__pre_audit_2026-05-10.csv` — pre-audit snapshot for diff
  - `figures/per_subject/*.png` — 16 per-subject 4-panel PNG
  - `figures/diagnostic/*.png` — 6 cluster grid 视觉诊断
  - `figures/README.md` — cohort 视觉骨架文档（中文）

### Run logs

- `results/run_logs/cohort_zer_20260509_2104.log` — pre-audit cohort run
- `results/run_logs/cohort_zer_audit_20260510_1045.log` — audit-rerun (channel-block null + ch-order check)

---

## 6. 2026-05-10 Yuquan Cohort Extension (PR-0.1)

Cohort grew from 16 → 25 by adding 9 yuquan audit_eligible subjects.
Engineering changes (commit chain `264cc25..fda465c`):

1. `scripts/build_yuquan_block_inventory.py` + `results/dataset_inventory/yuquan_block_inventory.csv` (new) — EDF probe → 115 block rows / 52 seizure rows
2. `src/yuquan_dataset.load_yuquan_record` — yuquan SEEG loader with intracranial filter (regex `^[A-Z]'?\d{1,3}$`) + CAR/bipolar
3. `src/ictal_onset_extraction.extract_seizure_window` — dual-dataset routing (replaces 2026-04 NotImplementedError); asymmetric join: yuquan seizure CSV uses `record`, block CSV uses `block_id`, both equal the EDF stem
4. `scripts/run_ictal_er_rank.py` — `SUPPORTED_DATASETS = {epilepsiae, yuquan}`, branched `_focus_rel_path` / `_count_seizures` / `_load_focus_rel`; `_load_focus_rel("yuquan")` normalizes the flat `{sid: [chans]}` to 3-tier `{i, l, e}` (l/e empty)
5. Canary on gaolan + huanghanwen — both produce v2.3 schema JSON without exception; both insufficient_n in z-ER (n_ok ≤ 1)
6. Full cohort run 16 → 25 in 147 min wall time
7. z-ER subtype rerun on 9 yuquan: 5 ok cells across 4 subjects (litengsheng broad k=2, sunyuanxin broad k=1, zhangkexuan gamma k=2, zhaojinrui gamma k=2 + broad k=1)

stable_k=2 was deliberately NOT used as a cohort gate. zhangjinhan
(stable_k=6) and zhaojinrui (stable_k=5) are kept because higher
interictal stable_k is a positive signal for within-subject seizure
heterogeneity — exactly what z-ER subtyping tests. Outcome confirms:
zhaojinrui (stable_k=5) is now one of the strongest z-ER cohort members
(both bands ok, gamma k=2). zhangjinhan came back insufficient_n
(small n_ok), but inclusion preserved cohort representativeness.

Yuquan SOZ JSON (`results/yuquan_soz_core_channels.json`) is normalized
into the 3-tier `{i, l, e}` surface that `classify_clinical_concordance`
expects (l/e empty for yuquan because yuquan has no l/e annotations).
This means yuquan `clinical_concordance` lands as `not_assessable` for
most cells — expected behavior, not a bug.

Cohort medians (50 subject-band rows, 33 ok):
- silhouette_k median = 0.444 (was 0.418 on n=16 pre-extension)
- gap_perm_k median   = 0.380 (was 0.327)
- yuquan ok subset (n=5): silhouette=0.495, gap_perm=0.552 — actually
  HIGHER than epilepsiae ok subset (silhouette=0.418, gap_perm=0.325).

Production-grade z-ER subjects post-extension (chosen_k ≥ 2 in any band):
- epilepsiae: 15 (1073, 1084, 1096, 1146, 1150, 139, 253, 442, 548, 583, 590, 635, 916, 922, 958)
- yuquan: 4 (litengsheng, sunyuanxin, zhangkexuan, zhaojinrui)
