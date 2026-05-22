# SEF-ITP Phase 1 Cohort 实跑 — 2026-05-22

> **状态**：preliminary，pending Yuquan coord 补全 + cohort 扩展 + sensitivity checks
> **runner**：`scripts/run_sef_itp_phase1.py` (commit `6c3a89e`)
> **per-subject 输出**：`results/topic4_sef_itp/phase1_spatial_geometry/per_subject/<dataset>_<sid>.json` (23 个)
> **cohort summary**：`results/topic4_sef_itp/phase1_spatial_geometry/cohort_summary.json`
> **图**：`results/topic4_sef_itp/phase1_spatial_geometry/figures/`（含中文 README）
> **plan**：`docs/archive/topic4/sef_itp_phase1/plan_2026-05-21.md`
> **framework**：`docs/topic4_sef_itp_framework.md` v1.0.2

---

## 0. Cohort 漏斗（实测）

```
40 个 masked Phase 0a 病人 (yuquan 20 + epilepsiae 20)
    ↓ 过滤：stable_k = 2
34 个
    ↓ 过滤：masked PR-6 anchoring 出了结果（valid 通道 ≥ 6 + SOZ JSON 等闸门）
30 个
    ↓ 过滤：3D 坐标可用（mm 单位）
23 个 = 8 Yuquan (fs_native_ras_mm) + 15 Epilepsiae (mni152_1mm)
```

**流失原因（按 cohort 大小排序）**：

| 闸门 | 流失 | 数据集 | 原因 |
|---|---|---|---|
| stable_k ≠ 2 | 6 | 混合 | 3 个 k=4，1 个 k=3，2 个 k=5/6 — 多模板结构，SEF-ITP 当前 H1/H2 假设 k=2 对偶模板 |
| PR-6 anchoring 没出 | 4 | 主要 yuquan | valid_channels < 6 或 SOZ JSON 空（epilepsiae_1125/620/916/384/yuquan_gaolan 子集）|
| 没坐标 | 7 | 全是 yuquan | chenziyang, hanyuxuan, huanghanwen, litengsheng, sunyuanxin, wangyiyang, xuxinyi — 原始 CT+MRI+reg.mat 齐，但电极点选未做（详见 §3.4 yuquan_coord_gap_investigation_2026-05-21.md）|

---

## 1. H6 — 参与场空间分隔

### 测了什么

每个病人，看高 HFO 参与率的通道（参与 ≥50% 事件）和低参与率的通道（< 50%），在大脑里是不是空间上**分开聚集**——是不是有一块"热区"和一块"冷区"，而不是高低参与率随机散布。

### 怎么测的

每个 subject 算三个指标：

- Moran's I（参与率的空间自相关，inverse-distance 权重）
- 高/低参与率两组的 centroid 距离
- 两组的 silhouette 分数

跑 1000 次 shaft-stratified shuffle（在每根电极杆内部 permute 参与率，跨杆分布保持）当 null。三个指标里 **≥2 个 p<0.05 → PASS；0 个 → NULL；1 个 → PARTIAL**。

### 揭示了什么（cohort n=23）

| 裁决 | 数 |
|---|---|
| PASS (≥2 显著) | 0 |
| PARTIAL (1/3 显著) | 3 |
| NULL (0/3 显著) | 10 |
| INSUFFICIENT_SPLIT (高或低组 < 2 通道) | 8 |
| EXCLUDED_SINGLE_SHAFT (subject 只有 1 根杆) | 2 |

**结论（preliminary）**：H6 在 cohort level **没有得到强支持**。能测到的 13 个 subject 里 10 个 NULL + 3 个 PARTIAL + 0 个 PASS。8 个 INSUFFICIENT_SPLIT 主要是因为通道数小 / 参与率阈值（0.5）下高低组之一 < 2。但即使把 PARTIAL 算 pass-like，3/13 = 23%，远远谈不上 cohort claim。

**两种可能的解读**：

1. **参与场确实没有跨电极杆的强空间组织** — 高/低参与率主要在同一根杆内梯度（这与 shaft 内部相关性大致匹配）。这个解读对 SEF-ITP 框架略有不利——意味着"病理易激场"如果存在，它的空间尺度可能小于杆间距（5–10 mm 量级），而不是跨杆的厘米尺度。
2. **shaft-stratified null 太严** — shaft-stratified 把"沿杆参与率梯度"当作 null 的一部分，等于扣掉了 H6 可能想抓的一个主要信号。如果用 unstratified random shuffle，cohort PASS 数应该更高，但那也意味着我们抓的是"参与率非均匀"而不是"跨杆空间聚集"，是更弱的 H6 claim。

我们选择保留 shaft-stratified null（严格版本）——这与 framework v1.0.2 §3.6 锁定的设计一致。

**敏感性 follow-up**：在补完 Yuquan 7 个坐标 → cohort=30 之后，重新跑一次；同时考虑 framework 里加一个"shaft-pooled H6"作为 sensitivity（不替换主分析）。

---

## 2. H1 — Endpoint 紧凑性 + 嵌入参与场

### 测了什么

每个 subject 每个 cluster（一共 23 × 2 = 46 cluster），看：

- 这个 cluster 的"起点通道"集合（PR-6 source，k=3）是不是空间上紧凑聚集
- "终点通道"集合（PR-6 sink，k=3）是不是空间上紧凑聚集
- 起点 + 终点的并集是不是嵌在"高参与率通道的几何中心"附近（必要条件 — 端点不能跑出参与场外）

### 怎么测的

三层独立检验：

1. **strict_source / strict_sink**：source/sink 集合的平均两两距离 vs 1000 个"匹配 shaft + 参与率 + HFO rate"的随机 3-通道子集（matched-null）。
2. **envelope**：端点到非端点中心的距离比，比例显著高于 1000 个随机端点子集 → 端点逃出参与场。
3. **整合裁决**：envelope FAIL → 整体 FAIL（必要条件）；envelope SKIPPED → INCOMPLETE_GATED_ON_COORDS；否则按 strict_source × strict_sink 真值表组合（PASS、partial_PASS、PASS_one_side_untestable、NULL、NULL_one_side_untestable、UNTESTABLE_BOTH_SIDES、FAIL_DIFFUSE、FAIL）。

### 揭示了什么（46 cluster）

| 类别 | 子裁决 | 数 |
|---|---|---|
| pass-like | PASS / partial_PASS / PASS_one_side_untestable | 1 / 2 / 6 = **9** |
| null-like | NULL / NULL_one_side_untestable | 1 / 1 = **2** |
| fail-like | FAIL / FAIL_DIFFUSE | 7 / 5 = **12** |
| untestable | INCONCLUSIVE_ENVELOPE_INDETERMINATE / UNTESTABLE_BOTH_SIDES | 19 / 4 = **23** |

**主要观察**：

- **50% 的 cluster 无法测**（INCONCLUSIVE_ENVELOPE_INDETERMINATE 主导）。原因：多数 subject 总通道数 ≤ 8，端点 k=3 + k=3 拿走 6 个之后 non_endpoint < 3，envelope 必要条件不能跑。这是**通道数限制**，不是 H1 假说被否。
- **能测的 cluster 里 fail-like (12) > pass-like (9)**：5 个 FAIL_DIFFUSE 是 strict layer 发现端点反而比 matched-null 更**散开**（anti-compact），7 个 FAIL 是 envelope 发现端点跑出参与场外。
- **6 个 PASS_one_side_untestable**：一边端点紧凑，另一边 INSUFFICIENT_NULL（matched-null 抽不出 100 个满足约束的样本）—— 这是 cohort 小 + 通道数小的复合效应。

**preliminary 结论**：当前精度下 H1 在 cohort level 不构成强支持，但**不能就此宣告假说否决**——

1. 一半 cluster 是 untestable，能测的部分 cohort n 也只有 23，统计功效太低
2. FAIL_DIFFUSE 信号（5/46）值得单独追踪——这是 strict layer 主动告诉我们"端点比同 shaft 同参与率的随机 3-通道子集更散开"，不是 null 信号
3. 补完 Yuquan 7 个坐标 + 扩展 cohort 到 30 之后必须重跑

---

## 3. H2 — 正反模板的 source/sink 反转几何

### 测了什么

每个 fwd/rev pair（一共 6 对，来自 23 subject 里有 candidate_forward_reverse_pairs 的 6 个），看：

- 正模板的 source 是不是和反模板的 sink **集合上**对应（Jaccard 反向 > 同向 → set-based reversal）
- 正模板的 source centroid 和反模板的 sink centroid 是不是**空间上**靠近（distance 反向 < 同向 → spatial reversal）

### 怎么测的

两条独立的 reversal index：

- **R_set = J_swap − J_same** > null upper 5% → set-based PASS
- **R_spatial = d_same / (d_swap + d_same)** > 0.5 且 null upper 5% → spatial PASS

null = 在 4 个端点集合 union 池里 shuffle role labels 1000 次。

整合裁决：两条都 PASS → PASS；一条 PASS → partial_PASS；其他 NULL。

### 揭示了什么（6 pair）

| 裁决 | 整合 | set-based | spatial |
|---|---|---|---|
| PASS | 4 | 4 | 5 |
| partial_PASS | 1 | — | — |
| NULL | 1 | 1 | — |
| EMPTY_SET | — | 1 | 1 |

**主要观察**：

- **5/6 pass-like**（4 PASS + 1 partial_PASS）—— 这是当前 cohort 最强的信号
- spatial reversal 5/5 都 PASS（剔除 1 个 EMPTY_SET），set-based 4/5 PASS
- spatial 比 set-based 还略稳，说明几何反转比集合反转更鲁棒

**preliminary 结论**：在能测到 fwd/rev pair 的少数 subject 里，**SEF-ITP 框架预测的"对偶端点 / 涟漪可从两端起"的几何反转特性高度成立**。这个信号方向上和 framework v1.0.2 §3.2 的 H2 PASS 预测完全一致。

**重要 caveat**（不能跳过）：

- n=6 太小，cohort-level 5/6 ratio 的二项检验 p = 0.109（α=0.5 null），**还不到 cohort claim 的统计门槛**
- 6 个 pair 来自哪 6 个 subject？需要核对是否集中在 dataset 一侧——如果 6 个全是 Epilepsiae，那"cohort claim"实际上只是"Epilepsiae 一侧 claim"
- PR-6 candidate_forward_reverse_pairs 的判定标准（spearman_r < 阈值）目前过严，导致很多 stable_k=2 subject 没出 pair

---

## 4. Cohort 整体一句话 verdict

**H2 在 6 对里 5 个 pass-like（最强信号但 n=6 不足以 cohort claim）；H1 一半 untestable + fail-like 12 vs pass-like 9（受通道数限制）；H6 cohort-level NULL/untestable 主导（弱信号）**。

framework v1.0.2 §0 一句话承诺里写："如果指纹大多数没看到 → 假说在我们的数据精度内被证伪"。**当前精度下 H1/H6 没看到强指纹**，但因为统计功效太低（n_testable 太小），**不能升级到 framework-level 证伪**。H2 的方向支持是真实的，但 n=6 还不够"cohort 支持"。

---

## 5. Pending 工作

按优先级：

1. **补 Yuquan 7 个 subject 的电极定位** → cohort 从 23 推到 30。这是当前 cohort 限制的最大单点突破。
2. **PR-2 candidate_forward_reverse_pairs 判定标准重审** → 让更多 stable_k=2 subject 进 H2 分析。
3. **Epilepsiae warp 类型核实** → normalization_certainty 从 `grid_confirmed_warp_type_unverified` 升级到 `mni_normalized_verified`。
4. **Sensitivity sweep**：H6 shaft-pooled null（不替换主分析，只作 sensitivity）；H1 endpoint k=2 / k=4（当前 k=3 lock）。
5. **Cohort summary 写进 paper_overview + topic4 主文档**——但只能在补完 §5.1 cohort 扩展之后写"接受结论"，目前只能写"preliminary，pending cohort 扩展"。

---

## 6. 内部归档代号映射（CLAUDE.md §8 朴素话风格）

- H6 = participation field spatial segregation
- H1 strict layer = matched-null endpoint compactness (within source, within sink)
- H1 envelope layer = endpoint ratio vs non-endpoint centroid (necessary condition)
- H2 set reversal = Jaccard-based source/sink role swap index
- H2 spatial reversal = centroid-distance-based source/sink swap index
- stable_k = adaptive cluster scan 锁定的 K（聚类数）
- PR-6 anchoring = endpoint definition via top-k template_rank with valid_mask filter
- fwd/rev pair = candidate_forward_reverse_pairs from PR-2 with spearman_r < threshold
- coord_units mm = main-analysis gate (voxel coord rejected by assert_coord_result_is_mm_for_main_analysis)
- normalization_certainty = honest tag for coord-comparability claim (subject_native / grid_confirmed_warp_type_unverified / mni_normalized_verified)
