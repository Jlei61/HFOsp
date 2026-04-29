# Epilepsiae Topic 3 PR-2 — i/l/e gradient + binary i-vs-e fallback

> 归档日期：2026-04-27
> Stage：Topic 3 PR-2 (per-channel relaxed-refine 三层 i/l/e 梯度)
> 数据合同：新 pipeline `results/hfo_detection/<subject>/`（详见 `epilepsiae_artifact_census_2026-04-27.md`）
> 前序：Yuquan PR-1（n=9，detrend_fraction SOZ < non-SOZ p=0.129、iei_median 短 p=0.055）

## 0. TL;DR

- **三层 i/l/e 梯度：n=1 valid subject (`253`)，结构性不可执行**。8/16 subject 的 `focus_rel.l` 列表为空（lesion 通道未标注），其余 8 个里 5 个在 relaxed-refine k=0.0 + min_count=100 + min_rate=5/h 活动门槛后单一区域不到 3 通道。Wilcoxon 三对 paired test 全部 SKIPPED (n<3)。
- **二元 i vs e fallback (n=8) 大致 null**：
  - `iei_detrended_r` greater: median_diff=+0.013, 6/8 i>e, p=0.19（弱趋势，方向 consistent with Yuquan PR-1）
  - `iei_median` less: median_diff=−0.33s, 2/8 i<e, p=0.37（方向 consistent，强度 null）
  - `detrend_fraction` less: median_diff=−0.004, 4/8 i<e, p=0.58（null）
  - `event_rate` (confound): median_diff=+50 events/h, 7/8 i>e, p=0.078（**确认 SOZ event-rate confound**）
  - 其他 metric 全 null
- **整体结论**：Epilepsiae cohort 在 PR-2 这个口径下**未独立支持 i/l/e SOZ 梯度**。Yuquan PR-1 的 detrend_fraction / iei_median 边缘趋势在 Epilepsiae 上未能复现到统计显著程度，但**方向（iei_median 短、iei_detrended_r 高）保持一致**。
- **结构性瓶颈而非代码 bug**：focus_rel JSON 缺 lesion 标注是数据合同问题；二元 i vs e fallback 给出的 n=8 也只能给 underpowered Wilcoxon。任何后续 strengthen 必须从（a）补 focus_rel.l 标注（b）放宽活动门槛 这两个方向走。

## 1. 数据合同 + 运行参数

- subject 全集：20（来自 `results/hfo_detection/` 数字子目录）
- focus_rel 覆盖：16/20 有 (`results/epilepsiae_electrode_focus_rel.json`)；4 个 (1084/442/1146/818) 缺失 → 整 subject 的 channel 全 `region_label=unknown`，paired stats 自动剔除
- 数据：每 subject `*_gpu.npz`（含 `whole_dets / start_time / chns_names / events_count`）+ `_refineGpu.npz` from new pipeline（Stage 0 census 全 `new_verdict=ready`）
- 运行命令：`python scripts/run_spatial_modulation.py --dataset epilepsiae --refine-k 0.0`（min_count=100, min_rate=5/h, min_group_channels=3）
- 输出：
  - per-subject JSON: `results/spatial_modulation/per_channel_metrics/epilepsiae/<subject>_perchannel.json` × 20
  - 三层 cohort: `results/spatial_modulation/soz_comparison/epilepsiae/cohort_three_tier_statistics.json`
  - per-channel CSV: `results/spatial_modulation/soz_comparison/epilepsiae/cohort_i_l_e.csv`
  - 二元 i-vs-e fallback: `results/spatial_modulation/soz_comparison/epilepsiae/cohort_binary_i_vs_e.json`
  - sensitivity at min_group=2: `results/spatial_modulation/soz_comparison/epilepsiae/cohort_three_tier_statistics_min_group_2.json`
  - 14 张 PNG + summary CSV + 中文 README: `results/spatial_modulation/figures/epilepsiae/`

## 2. 结构瓶颈：focus_rel.l 缺失导致 8/16 subject 不可三层化

`results/epilepsiae_electrode_focus_rel.json` 各 subject 三层标注计数：

| subject | \|i\| | \|l\| | \|e\| | l=0 | 三层合理? |
|---|---:|---:|---:|---|---|
| 139  |  8 | 28 | 21 | no  | ✓ |
| 253  |  3 | 42 |  6 | no  | ✓ |
| 384  | 16 | 22 | 78 | no  | ✓ |
| 548  |  7 | 76 |  1 | no  | borderline (e=1) |
| 583  | 15 |  **0** | 46 | **YES** | ✗ |
| 590  | 17 |  **0** | 103 | **YES** | ✗ |
| 620  |  5 |  **0** | 54 | **YES** | ✗ |
| 635  | 22 |  **0** | 56 | **YES** | ✗ |
| 916  | 11 |  **0** | 94 | **YES** | ✗ |
| 922  | 19 |  **0** | 65 | **YES** | ✗ |
| 958  | 21 |  **0** | 96 | **YES** | ✗ |
| 1150 |  6 |  **0** | 118 | **YES** | ✗ |
| 1073 |  5 | 66 |  **0** | no | ✗ (e=0) |
| 1077 |  8 | 94 |  3 | no | borderline (e=3) |
| 1096 |  8 | 37 | 51 | no | ✓ |
| 1125 |  1 | 56 |  5 | no | borderline (i=1) |
| (1084 / 442 / 1146 / 818 缺 focus_rel) | – | – | – | – | – |

**8/16 (50%) subject 的 lesion 列表为空**。这不是 bug，是 SQL 标注本身没有这一类的 contact（典型情形是 SOZ + healthy reference 的二元设计）。剩下 8 个里有 4 个三层完整 (139/253/384/1096) + 3 个 borderline + 1 个 e=0 (1073)。

## 3. 活动门槛后 per-subject region 计数

`relaxed_refine_k=0.0 + min_count=100 + min_rate=5/h` 后，每 subject 实际进 cohort stats 的 channel 计数：

| subject | n_total | i | l | e | unknown | artifact | 三层 valid (≥3 each)? |
|---|---:|---:|---:|---:|---:|---:|---|
| 1073 | 29 | 3 | 23 | **0** | 0 | 3 | ✗ (e=0) |
| 1077 | 16 | 5 | 4 | **2** | 0 | 4 | ✗ (e=2) |
| 1084 | 22 | – | – | – | 15 | 7 | ✗ (focus_rel missing) |
| 1096 | 13 | **0** | 3 | **2** | 0 | 8 | ✗ (i=0, e=2) |
| 1125 | 29 | **1** | 22 | 5 | 0 | 1 | ✗ (i=1) |
| 1146 | 26 | – | – | – | 5 | 21 | ✗ (focus_rel missing) |
| 1150 | 22 | 6 | **0** | 13 | 0 | 3 | ✗ (l=0) |
|  139 | 13 | 5 | **1** | 7 | 0 | 0 | ✗ (l=1 after activity filter) |
|  253 | 12 | 3 | 6 | 3 | 0 | 0 | **✓** |
|  384 | 23 | **2** | **0** | 7 | 0 | 14 | ✗ (i=2, l=0) |
|  442 | 23 | – | – | – | 17 | 6 | ✗ (focus_rel missing) |
|  548 | 18 | 5 | 7 | **0** | 0 | 6 | ✗ (e=0) |
|  583 | 18 | 10 | **0** | 8 | 0 | 0 | ✗ (l=0) |
|  590 | 22 | 6 | **0** | 11 | 0 | 5 | ✗ (l=0) |
|  620 | 12 | 3 | **0** | 8 | 0 | 1 | ✗ (l=0) |
|  635 | 22 | 11 | **0** | 7 | 0 | 4 | ✗ (l=0) |
|  818 | 12 | – | – | – | 4 | 8 | ✗ (focus_rel missing) |
|  916 | 14 | **1** | **0** | 12 | 0 | 1 | ✗ (i=1, l=0) |
|  922 | 11 | 9 | **0** | **2** | 0 | 0 | ✗ (l=0, e=2) |
|  958 | 45 | 13 | **0** | 32 | 0 | 0 | ✗ (l=0) |

**三层 valid: 1/20 (5%) — `253` 是唯一通过的 subject**。
**Sensitivity at min_group_channels=2**: 2/20 (`253`, `1077`)；仍然 < 3 subject，无法做 cohort Wilcoxon。

→ 三层 cohort statistics 输出全部 SKIPPED (n<3)。

## 4. 二元 i vs e fallback (n=8)

剔除 l 维度后用 i vs e 二元 paired Wilcoxon（min_group_channels=3, focus_rel-missing 与 single-region-thin subject 自动剔除）：

valid subjects: `1150 / 139 / 253 / 583 / 590 / 620 / 635 / 958` (n=8)

| metric | direction | n | median_diff | i>e | Wilcoxon p | 解读 |
|---|---|---:|---:|---|---:|---|
| `iei_detrended_r`  | greater   | 8 | +0.0127 | 6/8 | **0.191** | 弱趋势，方向 consistent with Yuquan PR-1 detrend_fraction direction |
| `detrend_fraction` | less      | 8 | −0.0035 | 4/8 | 0.578 | null |
| `iei_median`       | less      | 8 | −0.3253 | 2/8 | 0.371 | 方向 consistent (i 更短)，强度 null |
| `iei_p02`          | less      | 8 | +0.0093 | 5/8 | 0.527 | null + 方向不符 |
| `iei_lag1_r`       | two-sided | 8 | +0.0044 | 5/8 | 0.742 | null |
| `iei_cv`           | two-sided | 8 | −0.2417 | 3/8 | 1.000 | null |
| `event_rate`       | (confound)| 8 | **+50.25** | **7/8** | **0.078** | **确认 SOZ event-rate confound** |

观察：
- `iei_detrended_r` 是 best signal：6/8 subject 上 i > e，median_diff=+0.013（与 Yuquan PR-1 方向一致：SOZ 在 detrend 后保留更多 short-range memory）。但 n=8 + p=0.19，underpowered，达不到 statistical significance。
- `iei_median` 同向：i 比 e 短 0.33s（中位），与 SOZ 高活动的 mechanism 一致；但只 2/8 subject 严格满足 i_med < e_med，强 signal 之下 cohort 散布大。
- `event_rate` 是已知 confound：i 比 e 高 50 events/h，**这一项明显呈方向**——它本身既是 SOZ 的 phenotype 也是其它 metric 的混淆源。任何 i-vs-e 差异的解读都要保留 "rate-confound 未控制" 这一层。
- **最 critical 的 takeaway**：iei_detrended_r 的方向一致性给出**与 Yuquan PR-1 同方向的微弱信号**（detrend_fraction 在 Yuquan 是 SOZ 较小，对应 detrended_r 在 SOZ 较大），但 Epilepsiae cohort underpowered 导致 p value 远离显著。**Yuquan PR-1 的边缘 trend 在 Epilepsiae 上没有独立复现到 p<0.05 程度**。

## 5. 与 Yuquan PR-1 对照

| 维度 | Yuquan PR-1 (n=9) | Epilepsiae PR-2 (binary i vs e, n=8) |
|---|---|---|
| n_subjects | 9 | 8 (binary fallback) / 1 (3-tier) |
| 主 metric | `detrend_fraction` SOZ < non-SOZ p=0.129 (7/9) | `iei_detrended_r` i > e p=0.191 (6/8) |
| 方向 | SOZ 慢漂占比更小 | i 去趋势后 lag-1 更高（同向） |
| 二级 metric | `iei_median` 更短 p=0.055 | `iei_median` 更短 p=0.371 |
| event_rate confound | 已知 | 确认 (p=0.078 i>e) |
| 总评 | 边缘 trend，未显著 | 同向 trend，underpowered，未显著 |

→ 两个 cohort 都给出 weak directional consistency；但 PR-1 / PR-2 都达不到 significance。**没有 cohort 上"SOZ-specific 慢调制"的强证据**。

## 6. 结论

1. **三层 i/l/e gradient 在 Epilepsiae 当前 focus_rel 标注下结构性不可执行**：50% subject 缺 lesion 标注，剩下 50% 又被活动门槛进一步淘汰，最终只有 `253` 一个 subject 满足 i/l/e ≥ 3 each。Page-trend / Bonferroni 三对检验全部 SKIPPED。
2. **二元 i vs e fallback 给出 underpowered null + 弱方向一致**（n=8, p>0.19 全 metric）。`iei_detrended_r` 6/8 i>e + `iei_median` 方向同 Yuquan PR-1，但都没达到 p<0.05。
3. **event_rate confound 在 i vs e 上 7/8 i>e（p=0.078），确认 SOZ 高活动**。这是 PR-1 / PR-2 共有的混淆源；Wilcoxon paired test 没控制 rate，差异解读必须保留这一层。
4. **代码合同** 完整：scripts/runner + tests + figures + 中文 README + cohort statistics + binary fallback 都按 plan 落地，未发现剩余 bug。

## 7. 后续可做（不在本 PR 内执行）

1. **补 focus_rel.l 标注**：找 SQL 或临床团队补充 8 subject 的 lesion contact 标注。如果能补到 4-5 subject，三层分析就能得到 n=4-5 valid，可以做有意义的 paired Wilcoxon。
2. **放宽活动门槛 sensitivity**：min_count=20 / min_rate=1/h 试一次，看能不能把 1077 / 1096 等 borderline subject 推过 3-channel-each gate。
3. **rate-controlled binary i vs e**：mixed-effects 或 inverse-rate weighting，在 i / e 之间做 event_rate-matched 比较。直接对应 PR-1 同样建议。
4. **per-subject 案例叙述**：对 `253`（唯一三层 valid）单独画 subject-level i/l/e 时序图，看 monotonic trend 在它身上是否真的存在；同时对 binary fallback 8 subject 中 i>e direction 一致的 6 个做 case series。
5. **Yuquan PR-1 + Epilepsiae PR-2 联合 meta-analysis**：把两个 cohort 的 paired diffs 合在一起，n=17，可以提高 power。但前提是 metric 定义和方向一致，本 PR 已 verified 方向一致。

## 8. 已规避的设计错误（沿用前序 PR 经验）

- **不**改 plan 把 "i > l > e gradient" 写成单一方向：per-metric direction map 已 ack（`iei_detrended_r` greater / `detrend_fraction`/`iei_median` less / `event_rate` confound）
- **不**用 `focus_rel.keys()` 当 subject 全集：用 `results/hfo_detection/` 数字子目录全集（覆盖 4 个 focus_rel-missing subject 也跑 per-channel），区分 "label=unknown" 和 "subject not run"
- **不**把 Epilepsiae 输出和 Yuquan 共用同一目录：`soz_comparison/epilepsiae/` + `figures/epilepsiae/` 单独建
- **不**写"i > l > e SOZ-specific gradient confirmed"：实际 n=1 + 二元 fallback null，结论按数据出
- **不**引入 Page trend 新依赖：用 binomtest(null=1/6) 实现 subject-level monotonicity sign test
- **不**因 l=0 subject 多就把这些数据"伪标注"成 e/i 凑数：focus_rel 是上游真值，缺就缺，不补

## 9. 输出 artifact 入口

- 主代码：`scripts/run_spatial_modulation.py`（PR-1 Yuquan + PR-2 Epilepsiae 复用）
- Plot：`scripts/plot_spatial_modulation.py --dataset epilepsiae`
- Tests: `tests/test_spatial_modulation_three_tier.py`（16 tests, all pass）
- 图 + 中文 README：`results/spatial_modulation/figures/epilepsiae/`
- per-subject JSON: `results/spatial_modulation/per_channel_metrics/epilepsiae/<subject>_perchannel.json`
- Cohort: `results/spatial_modulation/soz_comparison/epilepsiae/{cohort_three_tier_statistics.json, cohort_i_l_e.csv, cohort_binary_i_vs_e.json, cohort_three_tier_statistics_min_group_2.json}`
- Census 前序: `docs/archive/topic3/epilepsiae_artifact_census_2026-04-27.md`
