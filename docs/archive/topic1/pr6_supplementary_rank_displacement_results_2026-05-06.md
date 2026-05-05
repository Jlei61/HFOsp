# PR-6 Supplementary：Per-Channel Signed Rank Displacement Results

> **状态：supplementary to PR-6（2026-05-06）。** Continuous version of PR-6 Step 4b
> discrete swap_node count（n=6 forward/reverse-reproduced subset, sign-test p=0.031）。
> 不立独立 cohort claim；不开 H1/H2 gate。
>
> **上游**：`docs/archive/topic1/pr6_template_endpoint_anchoring_plan_2026-04-25.md` §15 Step 4b；
> `docs/archive/topic1/ping_pong_hypothesis_review_2026-04-28.md` §4.2 + §6 PR-8 candidate roadmap。
>
> **plan**：`docs/archive/topic1/pr6_supplementary_rank_displacement_plan_2026-05-06.md`
>
> **代码**：`src/rank_displacement.py`、`scripts/run_rank_displacement.py`、
> `scripts/plot_rank_displacement.py`、`tests/test_rank_displacement.py`。
>
> **artifacts**：`results/interictal_propagation/rank_displacement/`（per_subject/, cohort_summary.json, figures/）。

## 1. 度量定义

对每对 cluster template (T_a, T_b)（按 PR-2 cluster_id 配对，**T_a = 较小 cluster_id**；PR-6 valid_mask 取交集后 dense re-rank）：

- 逐通道有符号位移 `Δr(ch) = rank_T_b(ch) − rank_T_a(ch)` (channel ∈ joint_valid)
  - **sign 合同**：仅 subject 内部有效，不跨 subject 聚合方向
- 整体 footrule `F = Σ|Δr|`
- Diaconis-Graham 归一化 `F_norm = F / floor(n_valid² / 2)` ∈ [0, 1]，1 = 完全反向
- Kendall τ(rank_T_a, rank_T_b)，−1 = 完全反向
- SOZ split（baseline-corrected）：
  - `soz_channel_fraction = n_soz_joint / n_valid`（chance baseline）
  - `soz_contribution_fraction = Σ|Δr|_SOZ / F`
  - `soz_contribution_excess = contribution_fraction − channel_fraction`（关键比较量）
  - `soz_abs_mean`, `nonsoz_abs_mean`, `soz_minus_nonsoz_abs_mean`

完整数学定义见 plan §3。

## 2. Cohort

主可视化 cohort = `stable_k == 2` ∩ PR-6 anchoring 有 valid_mask 的 subject。**n_available = 23**（PR-6 anchoring per_subject JSON 实际数；30 个 PR-2 stable_k=2 subject 中 7 个 `pr6_missing`）。

forward/reverse-reproduced 标签用 OR 规则（`first_half_second_half OR odd_even_block`，CLAUDE.md cross-PR contract lookup）：

| group | n |
|---|---|
| reproduced (`OR` rule = True) | 6 |
| not reproduced / not testable | 17 |
| **total** | **23** |

reproduced 6 个 subject：`epilepsiae_1073`, `epilepsiae_139`, `epilepsiae_548`, `epilepsiae_635`, `epilepsiae_958`, `yuquan_chenziyang`。

## 3. 主结果

| 指标 | reproduced (n=6) | not reproduced (n=17) | 备注 |
|---|---|---|---|
| Kendall τ 中位数 | **−0.495** | −0.048 | reproduced 中位数远偏负（接近 −0.5）；not reproduced 中位数靠近 0 |
| F_norm 中位数 | **0.964** | 0.688 | reproduced 接近 1.0（完全反向）；not reproduced 接近 asymptotic random reference (≈ 2/3，**非精确基线**) |
| `soz_contribution_excess` 中位数 | +0.018 | +0.040 | 都接近 0；SOZ 在 Δr 上参与与通道占比一致，无明显 enrichment |
| `soz_minus_nonsoz_abs_mean` 中位数 | +1.200 | +1.750 | 单 SOZ 通道平均 \|Δr\| 略高于单 non-SOZ；两组同向，descriptive only |
| `soz_channel_fraction` 中位数 | 0.450 | 0.636 | 两组 SOZ 通道占比都 ≥45%；Yuquan SOZ 列表更宽是已知事实 |

**与 PR-6 离散 swap_node 一致性**：PR-6 Step 4b sign-test n=6, p=0.031（fwd/rev-reproduced subset 上）。本 supplementary 在同一 6 个 subject 上的 Kendall τ：

- 6/6 subject Kendall τ < 0（100% direction consistency）
- median τ = −0.495；range = [−0.767, −0.394]
- 所有 6 subject 的 PR-6 `pr6_swap_score` 都 ≥ 0.35（H2 swap 方向已建）

→ **continuous metric 与 PR-6 离散 swap_node 同向，6/6 subject 一致**。这不是新的独立检验（同一 cohort），是对 PR-6 离散结果的 continuous-axis 补充。

**Independent cohort sanity**：本 supplementary 的 reproduced n=6 是从 PR-2 / PR-2.5 `time_split_reproducibility.splits` (OR rule) 与 PR-6 `template_anchoring/per_subject/*.json` 的 `valid_mask` 交集独立 derive 出来的，**与 PR-6 plan §15 H2 cohort 定义** (`endpoint_defined ∩ forward_reverse_reproduced`) **完全相同**（PR-6 plan 中 "forward/reverse-reproduced subset n=6" 在 §15 Step 4b）。这是对 cluster_id × valid_mask alignment 合同正确性的 cross-check —— 两个独立 runner 在不同字段路径上聚到同一个 6 subject cohort。

per-subject Δr 痕迹（按 Kendall τ 升序，沿 T_a source→sink 排序）：

```
epilepsiae_958     τ=-0.767  Δr = [+14, +14, +11, +9, +4, +5, +3, -1, -1, +2, -6, -10, -7, -11, -11, -15]
epilepsiae_635     τ=-0.644  Δr = [+8, +6, +4, +1, +1, -2, +3, -6, -6, -9]
epilepsiae_139     τ=-0.524  Δr = [+4, +4, +4, 0, -4, -3, -5]
epilepsiae_1073    τ=-0.467  Δr = [+5, +2, +2, -3, -3, -3]
yuquan_chenziyang  τ=-0.467  Δr = [+6, +7, +5, +6, -3, -2, -2, -2, -6, -9]
epilepsiae_548     τ=-0.394  Δr = [+11, +7, +8, +6, -1, -4, -4, -3, -2, -4, -3, -11]
```

每行单调"红→蓝"梯度都来自数据本身，**不是** sorting bias —— 列轴严格按 rank_T_a_dense 排序。τ ≈ 0 的 subject（如 `yuquan_chengshuai` τ=0.000, `epilepsiae_922` τ=+0.429）在同一排序规则下颜色散乱无梯度，是反 sorting bias 的实证。

## 4. 图

- `results/interictal_propagation/rank_displacement/figures/cohort_displacement_heatmap.{png,pdf}` — stable_k=2 × 通道 Δr 热图（行按 τ 排序，列按 rank_T_a_dense 排序）
- `.../footrule_kendall_summary.{png,pdf}` — 2-panel descriptive summary
- `.../per_subject/<subject>_displacement.png` — 每 subject 详图（共 23 张）
- `.../figures/README.md` — 中文图说明

## 5. 解读边界（写死）

可以说：
- "Continuous-version footrule + Kendall τ 与 PR-6 离散 swap_node 同向，6/6 fwd/rev-reproduced subject 一致"
- "Forward/reverse-reproduced subject 的 Kendall τ 中位数 = −0.495，集中在 [−0.77, −0.39]；F_norm 中位数 = 0.964（接近完全反向）"
- "Not reproduced subject 的 F_norm 中位数 = 0.688，接近 asymptotic random reference（≈ 2/3，n→∞ 渐近）"
- "SOZ `contribution_excess` 中位数 reproduced = +0.018, not reproduced = +0.040 —— SOZ 在两组中都没有明显 enrichment（descriptive only）"

**不**可以说：
- ~~"反向 template 是抑制墙的反弹"~~ — HFO 80–250 Hz 不区分 E/I
- ~~"footrule_normalized 高 ⇒ 致痫"~~ — 没做任何疾病侧 outcome 检验
- ~~"forward template SOZ-leading"~~ — 这是 PR-8 v1（DEFERRED）的范围，本 supplementary 不做 SOZ 极性方向判读
- ~~"SOZ contribution_fraction = X% ⇒ SOZ 主导"~~ — 必须用 `contribution_excess` 或 `abs_mean` 比较，裸 fraction 受通道数 confound
- ~~"高于 random baseline 2/3 即代表反向"~~ — 2/3 是 n→∞ 渐近期望，不是精确基线，不能用作显著性 gate
- ~~"reproduced 中位数 −0.495 显著反向"~~ — n=6，没做 cohort-level Wilcoxon / sign-test；这是描述统计，不是 hypothesis test 结论
- ~~"signed Δr 在 SOZ 通道为正 / 负"~~ — signed Δr 只在 subject 内部有效，跨 subject 聚合方向无意义

## 6. 历史链接

- `docs/topic1_within_event_dynamics.md` §7 — Topic 1 主文档
- `docs/archive/topic1/pr6_template_endpoint_anchoring_plan_2026-04-25.md` — PR-6 plan（上游 swap_node 离散合同）
- `docs/archive/topic1/ping_pong_hypothesis_review_2026-04-28.md` — PR-8 candidate 来源
- `docs/archive/topic1/pr6_supplementary_rank_displacement_plan_2026-05-06.md` — 本 supplementary 的 plan
