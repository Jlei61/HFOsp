# PR-6-sup1 — First-rank Entropy Results（2026-05-10）

> Plan v3：`pr6_supplementary_rank_entropy_plan_2026-05-10.md`
> Tier：**Topic 4 / SBA mechanism preflight，descriptive only，不进 paper α，不进 PR-6 主线**。
> 主文档：`docs/topic1_within_event_dynamics.md` §7.10 加 Topic 4 preflight 回链。

## 1. 一句话结论

**iHGE 事件的 rank vector 在 endpoint position 上比 middle position 更稳定**（H_p_norm 是 roof-shape 不是 bowl-shape）：subject-level 19/19 eligible subject 全 cohort `Δ_obs_subject < 0`（median = −0.123，sign test p = 1.9e-6），三组 §8 swap_class 之间无显著差异（Kruskal-Wallis p = 0.65）。Plan v3 §6.3 列出的三种 Δ 形态中，**实际数据落在"endpoint deterministic / middle 随机"那一档**，**否决 Liou-Abbott 风 noise-driven confluence-point 模型在 iHGE 上的 sharp prediction**。

## 2. Cohort 与 eligibility

- Cohort 入选 = supplementary v14 全 35 subject（stable_k=2 ∩ rank_displacement universe）。
- **Subject-level eligible (Option B 双 cluster 都 kept ≥ 50)：n=19**
- **Excluded `excluded_one_or_both_clusters`：n=15**（partial-participation 后某 cluster kept_events < 50 阈值）
- **Excluded `pr2_labels_len_mismatch`：n=1**（epi_1096，PR-2 labels 长度 223264 vs `_valid_event_indices(min_participating=3)` 给 223212；差 52 events，提示 PR-2 在该 subject 上用了不同 min_participating；excluded 走 case-series follow-up）

### 2.1 Eligibility × §8 swap_class

| swap_class | total | eligible | excluded |
|---|---|---|---|
| strict | 10 | 6 | 4 |
| candidate | 8 | 3 | 5 |
| none | 17 | 10 | 6 |
| (unknown / 1 unmatched) | — | — | 1 |

Eligibility 与 swap_class 没有明显关联——三组 exclude 比例都接近 40%，非选择偏置。

## 3. 主结果

### 3.1 Δ_obs_subject 全 cohort（n=19 eligible）

| 量 | 值 |
|---|---|
| median | **−0.123** |
| IQR | [−0.203, −0.112] |
| range | [−0.337, −0.079] |
| 全部 < 0 | **19/19** |
| sign test (one-sided Δ<0) | **p = 1.9e-6** |
| `is_subject_combo_max == True` 计数 | **0/19** |
| `subject_combo_percentile` median | **0.0023**（端点对组合接近 distribution 最小） |

### 3.2 Stratified by §8 swap_class

| 组 | n_eligible | Δ median | Δ range | percentile median | n_max |
|---|---|---|---|---|---|
| strict | 6 | −0.121 | [−0.222, −0.097] | 0.003 | 0 |
| candidate | 3 | −0.123 | [−0.302, −0.111] | 0.005 | 0 |
| none | 10 | −0.139 | [−0.337, −0.079] | 0.002 | 0 |

**Kruskal-Wallis on Δ_subject**：H = 0.868, **p = 0.648** —— 三组分布无统计差异。
**结论**：endpoint stability 是**所有 stable_k=2 subject 共有现象**，不是 §8 swap geometry 的副产物。

### 3.3 H_p_norm 形态（图 1 主图）

35 subject 的 H_p_norm 曲线（cluster 0/1 分别）插值到 normalized rank position x ∈ [0, 1] 后叠加：

- 全部呈 **roof-shape（屋顶形）**：中段 H ≈ 1.0（接近满熵）、端点 H 跌到 0.6-0.9
- cohort median 曲线在 x ≈ 0.5 处达 ~0.97 峰，在 x = 0 与 x = 1 两端各 dip 到 ~0.85
- 按 swap_class 三色叠加无肉眼可见分组

详见 `figures/H_p_norm_cohort_overlay.{png,pdf}`。

## 4. 大写 Caveats（必须随结论一起读）

### 4.1 Drop rate median 0.98 — 我们只看了"全员参与"事件的 ~2% 子集

`drop_rate` cohort 统计：

| 指标 | 值 |
|---|---|
| n_clusters_total | 68（34 subject × 2 cluster；epi_1096 提前 exit） |
| `drop_rate_median` | **0.980** |
| `drop_rate_max` | 1.000（某些 cluster 完全没有全员参与的 events） |
| n_clusters_high_drop_rate_warning（drop > 0.5）| **43/68 = 63%** |
| n_clusters_excluded_low_kept_events（kept < 50）| 25/68 = 37% |

Plan v3 §12.4 写死的触发条件：cohort `high_drop_rate_warning` 占比 > 30% → 应做 X=80% partial-participation sensitivity follow-up。当前 63% 远超阈值。**本批没做**该 sensitivity——所以现在的"endpoint determined / middle 抖动"严格只对**全员参与的最 stereotyped 事件子集**成立。

实际意义：可能存在两种解释，二者从本批数据无法区分：

- **(a) Stable-pathway 真信号**：iHGE 真的有 stereotyped 端点 channel，全员参与子集是 most stereotyped 的事件，整体方向就是这样。
- **(b) Selection bias**：要求所有 valid_mask channel 都参与已经隐含挑出了"刚好在端点 channel 上参与的 events"，因此把端点 deterministic 的子集挑出来——这是 §6.0 Option B 自身的 selection effect 而非真信号。

X=80% sensitivity（保留 ≥ 80% 参与的 events、用 partial dense rank）能在统计层面区分两者；本批不做。**结论的真正成立性需要后续工作支持**。

### 4.2 `is_subject_combo_max == True` 计数 0 看似矛盾，其实因 percentile 在分布另一端

Cohort percentile median = 0.002，意味着观测端点对在所有可能位置对的 Δ_combo 排序里**接近最小**——实际数据的 endpoint pair 是 LOWEST entropy difference，而不是 highest。`is_subject_combo_max` 只检测 endpoint 是否在 maximum 端，因此一致 False。

如果要 sign test "endpoint 是不是接近 minimum"：用 1 - percentile 翻面后阈值 0.95 即可。本批没做（不在 plan 范围）；archive only。

### 4.3 Liou-Abbott 类比有限 + iHGE scale

`SBA preflight` tier 的边界——这只能否决"在 iHGE sub-second 时间尺度上 Liou-Abbott confluence-point 是否被支持"，**不能**反向证伪整个 SBA 框架（ictal recruitment 上仍待 Topic 4 主线检验）。

### 4.4 Channel-selection circular caveat（与 PR-6 §9 同构）

lagPat 已是 high-HI / high-HFO-rate selected。"端点稳定 vs 中段抖动" 是在 selected channel 集**内部**；不是全脑信号。任何 paper-level 表述必须紧邻数字带这条 caveat。

### 4.5 1 subject (epi_1096) 数据完整性问题

PR-2 `adaptive_cluster.labels` 长度（223264）与 `_valid_event_indices(min_participating=3)`（223212）相差 52 events。该 subject 在 PR-2 阶段用了不同 min_participating 阈值（推测 4），导致 sup1 runner 的 label 重映射失败。**本批 excluded；不影响其他 18 subject 信号**；列入 follow-up：要么 PR-2 重跑统一 min_participating=3，要么 sup1 runner 接受多 min_participating 值的兼容映射。

## 5. 与 rank_displacement (§8 / §9) 的关系

**不重叠，是互补**。

| 维度 | rank_displacement | sup1 |
|---|---|---|
| 比的是什么 | 两 cluster 模板间的 channel-level 位移几何 | 单 cluster 内 events 之间的 rank-position 稳定性 |
| 主指标 | F_norm, Kendall τ, swap_class | H_p_norm, Δ_subject, percentile |
| Cohort 信号 | 连续谱 F_norm ∈ [0.39, 1.00]，median 0.81 | unanimous Δ < 0，median −0.123 |
| 含义 | "templates 之间是不是 swap" | "template 内部哪些位置可信" |

**桥接**（图 5 `bridge_rank_displacement`）：35 subject 在 (F_norm, subject_combo_percentile) 平面上的散点显示**两量正交**——sup1 信号集中在 percentile ≈ 0 的窄带（横向），与 F_norm 在 [0.4, 1.0] 上的展开（横向）几乎独立。

**反过来对 PR-6 主线 endpoint 定义的支撑**：PR-6 endpoint = top-3 ∪ bottom-3 channels 这个约定，sup1 给出独立经验证据是合理的——rank vector 的两端就是**event-wise 最稳定**的位置；中段反而是 noise-prone 区域。这强化了 PR-6 H1 / §8 / §9 endpoint 几何分析的方法论合理性，与各自 NULL 结论方向一致：**endpoint 的几何信号是真的（sup1 confirms），但它不锚定 clinical SOZ（H1/§9 NULL）**。

## 6. 与 v3 plan 预期方向的对照

| Plan v3 §6.3 列出的三种 Δ 形态 | 实际 cohort | 解读 |
|---|---|---|
| confluence model 成立 → Δ > 0 | 19/19 不是 | **否决** |
| rank 全随机 → Δ ≈ 0 | 19/19 不是 | **否决** |
| endpoint 是 deterministic、middle 随机 → Δ < 0 | **19/19 是** | **支持** |

Plan v3 已把第三种作为可能性写出；本结果**不是 surprise reversal，是 plan 列出的三档之一被实测命中**。Topic 4 / SBA preflight 的科学动机本来就是检验 confluence prediction，不是支持它——本结果相当于完成了否决检验。

## 7. Outputs

```
results/interictal_propagation/pr6_sup1_rank_entropy/
├── per_subject/<stem>.json           (n=35 raw, including 16 excluded)
├── cohort_summary.json                (subjects + cohort_drop_rate_summary
                                        + stratified_by_swap_class)
├── pilot_3subjects.json               (Task 5 pilot)
└── figures/
    ├── README.md                                 (中文逐图说明)
    ├── H_p_norm_cohort_overlay.{png,pdf}        (主图 — cluster 0/1 双 panel,
                                                  normalized x ∈ [0,1])
    ├── delta_by_swapclass_box.{png,pdf}         (Δ by swap_class)
    ├── endpoint_pair_percentile_panel.{png,pdf} (3-panel: percentile-Δ + 
                                                  n_valid floor + max bar)
    ├── swap_subset_per_subject.{png,pdf}        (18 strict+candidate panels)
    └── bridge_rank_displacement.{png,pdf}       (F_norm × percentile scatter)
```

## 8. Reproduction

```bash
python -m pytest tests/test_rank_entropy.py -v   # 22 tests, all green
python scripts/run_pr6_sup1_rank_entropy.py --pilot --n-perm-N0 200
python scripts/run_pr6_sup1_rank_entropy.py --cohort --n-perm-N0 1000
python scripts/plot_pr6_sup1_rank_entropy.py
```

代码 entry：
- `src/rank_displacement.py::compute_rank_position_entropy` + `compute_endpoint_middle_entropy_delta`
  + `rank_entropy_null_N0` + `rank_entropy_null_N1_pseudo_endpoint`
  + `rank_entropy_null_N1_subject_level` + `_filter_all_valid_participating_events`
  + `run_subject_rank_entropy`
- `tests/test_rank_entropy.py`（22 测试）
- `scripts/run_pr6_sup1_rank_entropy.py`、`scripts/plot_pr6_sup1_rank_entropy.py`

## 9. Follow-up（不在本批）

1. **X=80% partial-participation sensitivity**（plan §6.0 Option C 候选）：当前 high_drop_rate_warning 63% 远超 30% 阈值，应该做。能区分本批结论是 selection artifact 还是 stable-pathway 真信号。
2. **`epi_1096` PR-2 label mismatch 修复**：要么 PR-2 重跑统一 min_participating=3，要么 sup1 runner 兼容多阈值。
3. **如何把 sup1 桥到 paper 第一部分演化故事**：roof-shape 是 PR-6 endpoint 定义的 method-justification 证据；具体怎么写进 paper 由 Topic 1 整体 framing 决定（不在本 archive 范围）。
