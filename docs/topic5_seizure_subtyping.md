# Topic 5：Seizure-related analysis (per-subject subtype + 下游 pre-ictal / outcome)

> 状态：**探索性 (exploratory) — PR-0 + PR-1 落地，audit-rerun 完成 (2026-05-10)，yuquan 扩展 (2026-05-10 PR-0.1)**。PR-1 cohort z-ER 聚类有 1 张 cohort 主结果（`figures/per_subject/`，25 subjects），cohort-level "真子型断言" 仍依赖 sensitivity；PR-2+ 未启动。
> 范围：以 ictal seizure 本身为研究对象——subject 内的 seizure subtype carve-out + 下游 pre-ictal / outcome / propagation 关联。
> **不属于**：interictal 事件内部传播（topic1）、IEI/PSD（topic2）、spatial SOZ 归因（topic3）、模型层（topic4）。

---

## 1. 这个 topic 只回答什么问题

本 topic 回答以下三类问题：

1. **每 subject 内部，多个 ictal seizure 是否需要按 within-subject pathway 切分成 subtype？** (PR-1)
2. （未启动）切分出的 subtype 在 pre-ictal / propagation / outcome 层面是否表现出系统性差异？(PR-2+)
3. （未启动）subtype 与 SOZ propagation pattern 是否互相印证？

它**不**回答：

- interictal 群体事件内部传播的刻板性：那是 `docs/topic1_within_event_dynamics.md`
- 群体事件的 IEI / PSD：那是 `docs/topic2_between_event_dynamics.md`
- per-channel SOZ vs non-SOZ 慢调制：那是 `docs/topic3_spatial_soz_modulation.md`

---

## 2. 一句话当前结论

- **PR-0 (v2.3 Layer A ictal ER timing atlas)**：cohort = 25 (15 epilepsiae audit_eligible + 9 yuquan audit_eligible + sentinel-only epilepsiae/916; topic5 PR-0.1 2026-05-10 yuquan extension)。每 subject v2.3 schema，per-seizure PNG 全 cohort 渲完。User 视觉巡视暴露 within-subject seizure pattern 异质性（442 sz=9 / 548 {13,14,24,25} / 916 {21,23,25} / 1077 sz=1），是 PR-1 的直接动机。详见 `docs/superpowers/specs/topic5_pr0_*` (待整理) + `results/data_driven_soz/layer_a_ictal_er_rank/atlas_v2_3/figures/`。
- **PR-1 z-ER subtype 聚类（2026-05-10 audit-corrected exploratory 版；2026-05-10 yuquan-extended）**：25 subjects (16 epilepsiae + 9 yuquan)，50 subject-band rows，33 ok / 17 insufficient_n。yuquan ok 子集 (n=5 cells, 4 subjects: litengsheng broad k=2, sunyuanxin broad k=1, zhangkexuan gamma k=2, zhaojinrui gamma k=2 + broad k=1) silhouette median 0.495 / gap_perm median 0.552 — 实际优于 epilepsiae ok 子集 (silhouette 0.418 / gap_perm 0.325)。整体 cohort silhouette median 0.444、gap_perm median 0.380。`over_split_flag` (AND 规则 `gap_perm < 0.10 AND ratio > 0.5`) cohort 命中 **0/33 ok**。Bug-fix 实测影响：pre-audit 28 个 ok rows 上 Δgap_perm 中位 −0.0007、abs_max 0.061，**0 个 over_split_flag flip**，0 个 sentinel jaccard 变化。
- **PR-1 sentinel 视觉裁定**（user 2026-05-09 / 2026-05-10）：
  - 442 (user=[9])：**最干净 sentinel** ✅，gamma+broad 都把 sz=9 单列
  - 548 broad k=3：**基本合理**，user-marked [13,14,24] 落同一 minority 家族
  - 548 gamma k=7：**high-heterogeneity / fine subdivision candidate** ⚠️ ——不是过切但也未确认真 7 类，需要 sensitivity (min_subtype_size=3 / 不同 bin / bootstrap stability)
  - 1146 broad k=3：**教科书级 3 子型分离** ✅
  - 916 (user=[21,23,25])：**不能作为 sentinel** ✗（user 标的 3 个全被 v2.3 status filter 过滤）
  - 1077 (user=[1])：**不能作为 sentinel** ✗（n_ok=3 < 5）
  - 有效 sentinel = 442 + 548（4 个 subject-band 全 recall=100% on user-marked outliers）
- **2026-05-10 audit fix（重要）**：发现并修复 3 个 bug
  - `channelwise_permutation_null` 在 z-ER 路径破坏 5-bin 协方差 → 加 `bins_per_channel` 参数走 channel-block coherent shuffle
  - `extract_zer_binned_for_subject` 缺少 channel-order consistency check → 加严格 equality check + `channel_order_mismatch` drop_reason
  - `over_split_flag` 旧 OR 规则 (`sil<0.2 OR ratio>0.4`) 在高维 Spearman 下产假阳性 → 改 AND 规则 + 用 gap_perm 替代 silhouette
  - cohort z-ER audit-rerun 已完成（2026-05-10 16:21）；cohort-level 结论与 sentinel 视觉裁定全部保持有效，gap_perm 数值微调（中位 −0.0007）
- **下游 PR contract**：subtype_label 是先验分组依据，PR-2+ 必须 per-subtype 不 per-subject。

---

## 3. 核心证据链

### 3.1 PR-0：v2.3 Layer A ictal ER timing atlas

每 subject 一张 (gamma+broad) 主 atlas + 每 seizure 一张 per-seizure PNG。
schema：`pr_t3_1_layer_a_v2_3_timing`，detection_window=[-120, 30]s，
`channel_onsets[ch] = {frame_idx, t_onset_sec}`。
cohort：25 subjects (15 epilepsiae audit_eligible + 9 yuquan audit_eligible + sentinel-only 916; topic5 PR-0.1 2026-05-10 yuquan extension; gaolan, huanghanwen, litengsheng, pengzihang, sunyuanxin, xuxinyi, zhangjinhan, zhangkexuan, zhaojinrui)。
关键发现：user 视觉巡视暴露 within-subject seizure pattern 异质性，
直接催生 PR-1 z-ER subtyping。

完整说明：`results/data_driven_soz/layer_a_ictal_er_rank/atlas_v2_3/figures/README.md`

### 3.2 PR-1：z-ER subtype 聚类

**Method**：

- Feature: per-channel × 5-bin z-ER 张量，bins `[(-200,-50), (-50,0), (0,50), (50,150), (150,200)]` s rel. clinical onset
- Distance: `1 − Spearman` over channel-bin features (min_overlap=5)
- Linkage: UPGMA
- k 选择: silhouette + min_cluster_size 守门
- Outlier vs subtype: cluster size < 2 → outlier (`subtype_label=-1`); rest → subtype 0..k-1 by descending size
- Permutation null (gap_perm_k): channel-block coherent shuffle (5 bins of a channel move together)

代码：`src/ictal_zer_features.py` + `src/ictal_seizure_clustering.py`
驱动：`scripts/cluster_ictal_seizures.py {per-subject, cohort, render}`

**Cohort 数值（n=25, 50 subject-band rows, post yuquan-extension 2026-05-10）**：

| 指标 | 中位 | 范围 |
|---|---|---|
| n_eff | 9.0 | [5, 40] |
| silhouette_k | **0.444** | [0.128, 0.597] |
| gap_perm_k | **0.380** | [0.094, 0.737] |
| n_subtypes | 2.0 | [1, 5] |
| ari_gamma_vs_broad | — | 多数 ≥ 0.6（双 band 一致度高） |

yuquan ok cells (n=5): silhouette 0.495, gap_perm 0.552（高于 epilepsiae ok 子集 silhouette 0.418 / gap_perm 0.325）。

`over_split_flag` 在 33 ok cells 中的命中数：**0/33** (AND 规则 `gap_perm<0.10 AND ratio>0.5`)；当前 `cohort_summary__zer_binned.csv` 已可查询 `over_split_flag` 列。

**形态层面**：约 33/50 (66%) subject-bands 落 ok 状态; ok 子集中 **23/33 (70%) 找到 ≥2 morphological subtypes** (基于 n_subtypes ≥ 2 的 subject-band 数 / ok 数) → within-subject morphological
异质性是 cohort-level 真现象（biological prior 与 Schroeder 2020 *PNAS* pathway-variability
一致）；cohort-level "真子型率" 的 publication-grade 断言仍需 sensitivity (intersection-only
mask / bin 设计变化 / bootstrap stability) 才能 commit。

**Sentinel 详细**：见 §2 五行表，与 archive doc §5。

完整 method + bug fix + sentinel 表 + per-subject 数值：
`docs/archive/topic5/pr1_seizure_clustering/pr1_zer_cohort_2026-05-10.md`

### 3.3 PR-1 视觉骨架

`results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/figures/per_subject/`
共 16 张 4-panel PNG（gamma 上 / broad 下）：

- 左：dendrogram + sorted pairwise (1−Spearman) heatmap
- 右上：MDS 2D 散点 colored by subtype（outlier 用 X marker + 灰色）
- 右下：cluster-stratified t_ER_onset matrix（rows=channel, cols=seizure 按 subtype 分组）

诊断 grid（`figures/diagnostic/`）：6 张以 v2.3 atlas per-seizure thumbnail 为单元
的 cluster-grouped grid，用于目视裁定。

---

## 4. 已知 caveat

1. **gap_perm bug-fix 实测影响小**：cohort 28 个 ok rows 上 Δgap_perm 中位 −0.0007、
   abs_max 0.061，**0 个 over_split_flag flip**，0 个 sentinel jaccard 变化。bug 真实
   但本 cohort 上效应几乎为零；早先 sentinel 视觉裁定与 ARI 等所有 PR-1 结论保持有效。
2. **共同 channel mask 缺失**：z-ER feature 用 nanmean per bin 处理跨 seizure 缺通道，
   高 coverage subject 基本无害；低 coverage subject 应做 sensitivity (intersection-only mask)。
3. **916 / 1077 sentinel 失效**：是 v2.3 status filter / `n_ok < 5` 门的副产物，
   不是聚类失败；不能作 recall/precision evidence。
4. **548 gamma fine subdivision 不能确认真 7 类**：sensitivity battery
   (min_subtype_size=3 / 不同 bin / common channel mask / bootstrap stability)
   是 commit 前必跑项。
5. **Yuquan 部分覆盖**：cohort=25 含 9 yuquan audit_eligible (gaolan, huanghanwen, litengsheng, pengzihang, sunyuanxin, xuxinyi, zhangjinhan, zhangkexuan, zhaojinrui)。仍有 12 yuquan 因 n_seizures<2 被排除（chengshuai, dongyiming, hanyuxuan, huangwanling, liyouran, songzishuo, wangyiyang, zhangjiaqi, zhaochenxi, zhourongxuan, chenziyang, zhangbichen），ictal pool 不足以做 within-subject 聚类，无法补救。9 个 yuquan 中 4 个 (litengsheng, sunyuanxin, zhangkexuan, zhaojinrui) 在至少一个 band 上达到 ok 状态；其余 5 个落 insufficient_n（CUSUM 阈值 λ=100 cap 下 onset 未触发）。yuquan ok 子集的 z-ER 聚类质量 (silhouette 0.495 / gap_perm 0.552) 高于 epilepsiae ok 子集 (0.418 / 0.325)，但样本太小不足以做"yuquan vs epi"对比。
6. **方法溯源严格性**：Schroeder 2020 *PNAS* 是生物先验（within-patient pathway
   variability），**不是 pipeline 复现**。本 PR-1 聚类管道 (1−Spearman + UPGMA +
   silhouette + permutation null + outlier split) 全部本项目实现；Panagiotopoulou
   2022 *Brain Communications* 不能作为 pipeline 直接溯源。
7. **`over_split_flag` 是 descriptive flag，不是过切检验**：真正过切判定需要
   gap_perm（正确 null）+ 视觉 diagnostic + sensitivity 三方一致。

---

## 5. 历史文档索引

- `docs/archive/topic5/INDEX.md` — topic5 archive 索引
- `docs/archive/topic5/pr1_seizure_clustering/pr1_zer_cohort_2026-05-10.md` — PR-1 主结果文档（audit-corrected）
- `results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/figures/README.md` — cohort 视觉骨架文档
- `docs/superpowers/specs/topic5_pr1_seizure_clustering*.md` — PR-1 plan v2
- `docs/superpowers/plans/2026-05-10-topic5-pr0_1-yuquan-ictal-cohort-extension.md` — yuquan cohort 扩展 plan (2026-05-10)
- `docs/archive/topic5/bridge_q1/bridge_q1_results_2026-05-10.md` — Topic 1 × Topic 5 Bridge Q1 cohort exploratory result (verdict: NULL-locked, n=9; power floor identified)
- `docs/archive/topic5/bridge_q1prime/bridge_q1prime_results_2026-05-10.md` — Topic 1 × Topic 5 Bridge Q1' PIVOT case-series result (verdict: INDETERMINATE, N=4 strict + 548 candidate; 4 strict subjects show consistent positive Cramér V 0.25–0.67 median 0.486 but underpowered on p; channel-rank correspondence × swap-subset)
- `docs/archive/topic5/bridge_q1prime/q1prime_overnight_exploration_2026-05-10.md` — Q1' overnight 探索：full 25-subject cohort + per-seizure feature × delta_rho/subtype 相关性分析 (verdict: INDETERMINATE/WEAK-SIGNAL; median_onset_latency_sec 有方向性倾向 sign_p=0.039 uncorrected; Stage C 全 NULL; 2 subjects 有大效量 subtype 区分)

---

## 6. 下游 PR 必须遵守

1. **Per-subtype 不 per-subject**：从
   `results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/per_subject/<sid>__zer_binned.json`
   读 `result["per_band"][band]["subtype_label"]` 与 `outlier_flag`，按 subtype
   分别统计 pre-ictal / outcome / propagation 指标。**禁止**在 subject 内对所有 seizure 平均。
2. **`subtype_size < 3` 处理规则**：cohort 有不少 size=2 的 subtype，
   小子型 statistical power 不足。下游 PR 必须在自己的 README 标处理规则
   （pool with annotation / drop / 全 cohort CI）。
3. **band 选择**：`gamma_ER` vs `broad_ER` cohort 数值近似 (median ARI ≥ 0.6)；
   下游可任选其一作主分析，另一作 sensitivity；不能两 band 同时跑而不合并解释。
4. **t_onset feature 已被 z-ER 取代**：`per_subject/*.json` (无 `__zer_binned` 后缀)
   保留为历史归档，不再作为 PR-2+ 的 subtype 来源。

---

## 7. 文件清单

### 代码

- `src/ictal_zer_features.py` — z-ER tensor extraction + binning + channel-order check
- `src/ictal_seizure_clustering.py` — pairwise dissim, UPGMA, k selection, channel(-block) permutation null, outlier/subtype split, sentinel jaccard, EEG-realign helpers
- `src/ictal_seizure_plotting.py` — MDS, subtype color palette, sort orders
- `scripts/cluster_ictal_seizures.py` — CLI driver `per-subject / cohort / render`
- `scripts/diagnostic_cluster_grid.py` — cluster-grouped per-seizure thumbnail grid

### 测试

- `tests/test_ictal_seizure_clustering.py` (33 tests) + `tests/test_ictal_zer_features.py` (5 tests) + `tests/test_ictal_seizure_plotting.py` (8 tests) = **45 tests pass**

### 数据 / 图

- `results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/`
  - `per_subject/*__zer_binned.json` — 25 cohort z-ER cluster results (16 epilepsiae + 9 yuquan)
  - `cohort_summary__zer_binned.csv` — 50 subject-band rows
  - `cohort_summary__zer_binned__pre_audit_2026-05-10.csv` — pre-audit 快照 (n=16, 32 rows)
  - `figures/per_subject/*.png` — 25 per-subject 4-panel PNG
  - `figures/diagnostic/*.png` — 6 cluster grid 视觉诊断
  - `figures/README.md` — cohort 视觉骨架文档（中文）

### Run logs

- `results/run_logs/cohort_zer_20260509_2104.log` — pre-audit cohort run
- `results/run_logs/cohort_zer_audit_20260510_1045.log` — audit-rerun (channel-block null + ch-order check)
