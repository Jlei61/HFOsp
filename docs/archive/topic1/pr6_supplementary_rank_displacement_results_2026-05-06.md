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

## 2. Cohort（**PR-6 supplementary cohort，不是完整 stable_k=2 cohort**）

> **Scientific boundary（写死）**：本 supplementary 依赖 PR-6 endpoint-defined artifacts（`per_template[k].valid_mask`），**不是**完整 27 subject stable_k=2 rank-geometry 分析。这个选择是为了与 PR-6 离散 swap_node 同 cohort 比较（cross-check 需要），可以接受，**但不能包装成全 stable_k=2 cohort**。

PR-2 cohort 全景（30 subject）：

| stable_k | n | subject 列表 |
|---|---|---|
| 2 | **27** | `epilepsiae_{1073, 1077, 1084, 1096, 1125, 1146, 1150, 139, 253, 384, 442, 548, 583, 590, 620, 635, 916, 922, 958}`（19 个）+ `yuquan_{chengshuai, chenziyang, hanyuxuan, huanghanwen, litengsheng, liyouran, sunyuanxin, xuxinyi}`（8 个）|
| 4 | 2 | `epilepsiae_818`, `yuquan_huangwanling` |
| 6 | 1 | `yuquan_zhangjinhan` |

本 supplementary cohort = stable_k=2 ∩ PR-6 endpoint-defined → **n=23 / 27**：

- **23 nominal cohort**：PR-6 `template_anchoring/per_subject/*.json` 已有，本 supplementary consume
- **4 个 stable_k=2 subject 被排除**（PR-6 anchoring 缺失，原因可能是 SOZ JSON 空 / n_ch < 6）：`epilepsiae_1125`, `epilepsiae_384`, `epilepsiae_620`, `epilepsiae_916`
- **3 个 stable_k≠2 subject 不在范围内**：`epilepsiae_818` (k=4), `yuquan_huangwanling` (k=4), `yuquan_zhangjinhan` (k=6)

**这是 PR-6 supplementary cohort，覆盖 23/27 = 85% 的 stable_k=2 subject。** 完整 27-subject stable_k=2 rank geometry 需要绕过 PR-6 endpoint 合同 (例如直接用 PR-2 raw bools 推导 valid_mask)，那是另一个 PR 的工作，本 supplementary **不**做。

forward/reverse-reproduced 在 PR-2.5 内有三种 outcome（不是简单二分）：

| PR-2.5 outcome | 含义 | n |
|---|---|---|
| **TRUE** | `inter_cluster_corr_matrix` Spearman ρ < −0.5（候选）AND split-half OR odd-even reproducibility 通过 | **6** |
| **FALSE** | 候选（ρ < −0.5）但 reproducibility 测试失败 | 1（`yuquan_huanghanwen`，ρ = −0.527）|
| **None** | 非候选（ρ ≥ −0.5），未跑 reproducibility 测试 | 16 |

**关键**：本 supplementary 的 cohort 划分按 **PR-2.5 候选门槛**（fwd/rev cohort = TRUE+FALSE = n=7）vs **非候选**（None, n=16），**不**按裸的 reproduced/not-reproduced binary 划分（plan 早期版本采用过的错误划分会让 borderline ρ ≈ −0.4 但 F_norm ≈ 0.8 的 subject 错误归到"未反向"组）。

reproduced 6 个 subject：`epilepsiae_1073`, `epilepsiae_139`, `epilepsiae_548`, `epilepsiae_635`, `epilepsiae_958`, `yuquan_chenziyang`。candidate-fail 1 个：`yuquan_huanghanwen`。

## 3. 主结果（按本 supplementary 自己的指标分组，含 Mann-Whitney U）

**分组合同（pre-registered，写死）**：弃用任何 PR-2.5 派生的硬阈值（`reproduced/not-reproduced` 是 PR-2.5 二步嵌套的 binary；`ρ_inter < −0.5` 是 PR-2.5 候选门槛——都是另一个 PR 的硬阈值）。改为按**本 supplementary 自己计算的 footrule** 在 **Diaconis-Graham 渐近随机参考点 2/3** 处分组：

| group | 定义 | n |
|---|---|---|
| F_norm > 2/3（above asymptotic random）| 比纯随机排列的 D-G 渐近期望（≈2/3）更大的归一化 footrule | **15** |
| F_norm ≤ 2/3（around random）| 在或低于 D-G 渐近随机参考点 | **8** |

**为什么用 2/3 而不是 ρ_inter < −0.5 / `forward_reverse_reproduced`**：
- 2/3 是 Spearman footrule 在均匀随机置换下 n→∞ 的渐近期望（Diaconis-Graham 1977）—— **数学上的自然参考点**，不依赖任何 PR 的人为阈值
- ρ_inter < −0.5 是 PR-2.5 候选门槛，本质上是另一个 PR 的硬阈值；borderline subject（ρ_inter ∈ [−0.5, −0.3]）会被错误划到"非候选"组
- `forward_reverse_reproduced` 是 PR-2.5 在候选门槛之上又叠了一个 reproducibility 测试，**两步嵌套**让"未测候选"和"测了不过"在同一个 None bucket，更糟

| 指标 | F_norm > 2/3 (n=15) | F_norm ≤ 2/3 (n=8) | 统计 |
|---|---|---|---|
| F_norm 中位数 | **0.880** | 0.491 | (按 F_norm 分组测 F_norm 是循环论证，**不**做 MW-U) |
| Kendall τ 中位数 | **−0.333** | +0.295 | **MW-U U=6.0, p = 0.000548**（descriptive，非 PASS gate）|
| `soz_contribution_excess` 中位数 | +0.067 | −0.055 | MW-U p ≈ 0.048（borderline，descriptive only）|

**Spearman ρ(F_norm, Kendall τ) = −0.924, p = 3.4e−10, n=23**（continuous，principled）—— 两个指标高度反向相关，证明它们捕捉同一个底层几何（rank reversal degree）。

### 3.1 高 F_norm 组（n=15）的 PR-2.5 状态混合

PR-2.5 status 在 F_norm > 2/3 组内的分布：

- 6 reproduced（PR-2.5 候选 + reproducibility 通过）：`epi_1073, 139, 958, chenziyang, 635, 548`
- 1 candidate-fail（PR-2.5 候选但 reproducibility 失败）：`yuq_huanghanwen`
- 8 non-candidate（PR-2.5 ρ_inter ≥ −0.5）：`epi_1150, 1146, 583, hanyuxuan, 1077, liyouran, 590, chengshuai`

**这正是按 PR-2.5 阈值分组的盲点**：8 个 non-candidate subject 在我们指标上明确呈现"高于随机的 reversal"（F_norm > 2/3），但 PR-2.5 候选门槛把他们划在外面。borderline ρ_inter（如 `epi_1146` ρ=−0.464, `yuq_liyouran` ρ=−0.404, `epi_1077` ρ=−0.371）的 subject 仍带显著反向几何，只是没过 PR-2.5 二分门槛。

**重要 caveat**：MW-U p = 0.000548 并非 PASS gate；§0 禁区第 1 条写死本 supplementary 不开 cohort PASS 检验。这里只作描述。Spearman ρ = −0.924 反映 F_norm 与 Kendall τ 在本 cohort 上的高度共线性——两个指标基本测量同一件事（rank displacement / inverse rank concordance），所以"按 F_norm 分组测 τ"的 MW-U 在 effect size 上是基本可预测的，**不是独立证据**。它的意义在于：把分组从 PR-2.5 阈值移到 D-G 渐近参考之后，多了 8 个 borderline subject 进入"高反向"组，而这些 subject 的 τ 同样显著偏负。

### 3.2 与 PR-6 离散 swap_node 一致性 cross-check

PR-6 Step 4b sign-test n=6, p=0.031（fwd/rev-reproduced subset 上）。本 supplementary 在同一 6 个 subject 上的 Kendall τ：

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

### 3.3 SOZ enrichment（baseline-corrected, descriptive only）

| 指标 | F_norm > 2/3 (n=15) | F_norm ≤ 2/3 (n=8) | MW-U p（descriptive）|
|---|---|---|---|
| `soz_contribution_excess` 中位数 | +0.067 | −0.055 | p ≈ 0.048（borderline）|
| `soz_minus_nonsoz_abs_mean` 中位数 | +1.500 | +1.250 | — |
| `soz_channel_fraction` 中位数 | 0.500 | 0.578 | — |

**`soz_contribution_excess` borderline 高于 0 in F_norm > 2/3 组**：descriptive 提示在"高反向"组中，SOZ 通道在 Δr 上参与略高于其通道占比，但 effect size 很小（中位数差 ~0.12）且在 F_norm ≤ 2/3 组中反向（−0.055）。**不**作为 SOZ enrichment 的独立证据；§5.1 的通道选择 caveat 直接限制了任何 SOZ-vs-nonSOZ 比较的解读力度。

## 4. 图

- `results/interictal_propagation/rank_displacement/figures/cohort_displacement_heatmap.{png,pdf}` — stable_k=2 × 通道 Δr 热图（行按 τ 排序，列按 rank_T_a_dense 排序）
- `.../footrule_kendall_summary.{png,pdf}` — 2-panel descriptive summary
- `.../per_subject/<subject>_displacement.png` — 每 subject 详图（共 23 张）
- `.../figures/README.md` — 中文图说明

## 5. 解读边界（写死）

### 5.1 通道选择 caveat（关键方法学边界）

**lagPat npz 的通道集不是"该 subject 所有 SEEG/ECoG 通道"**，而是 legacy yuquan 高-HI gate + 高 HFO rate 选出来的子集（每 subject 通常 n_ch ∈ [6, 24]）。这意味着：

- footrule 和 Kendall τ 度量的是**该选定通道集之内**的 rank 排序，不是"真实"全脑反向程度
- 通道选择**偏宽**（混入旁观非病理通道）→ 旁观通道在两个 template 间的 Δr 接近随机 → footrule 偏低、方差偏大、F_norm 被稀释
- 通道选择**偏窄**（漏掉真传播通道）→ 反向几何被欠反映，Kendall τ 可能比真值更接近 0

**所以 footrule 和 Kendall τ 不一定能"正确反映该 subject 的正反向程度"**：它们反映的是 lagPat 选定通道集**之内**的两个 template 排序差异。在解读单 subject 数字时，这个 caveat 必须保留：

- 同一 subject 不同通道选择策略（例如改用 PR-1 ER-leading 通道、Topic 3 SOZ-audit 通道、或更严格的 high-HI 阈值）下的 F_norm 与 Kendall τ 会变化
- 跨 subject 比较 F_norm 时，通道集 size（n_valid）与 type（病理 vs 旁观比例）的差异是 confound，不能用单个 footrule 数字断言"A subject 比 B subject 更反向"

**本 supplementary 不重新选通道**（重新选属于另一个 PR 的工作），只在 lagPat 既定通道集之内做 supplementary 描述。任何 cohort-level claim 都必须在文字中带上这个 caveat。

### 5.2 二分组的方法学风险（Panel B/C 分组合同）

PR-2.5 `forward_reverse_reproduced` flag 的逻辑是**两步嵌套**：

1. 候选门槛：`inter_cluster_corr_matrix` Spearman ρ < −0.5 → 候选
2. 候选 subject 跑 reproducibility 测试（split-half + odd-even）→ TRUE / FALSE
3. 非候选 subject **未跑 reproducibility 测试** → flag = None

**早期版本的 plan / 第一版图把 `True` 与 `False ∪ None` 二分，会把"未测候选 + 测了不过"两类混淆**。在 cohort 数据中：

- `epilepsiae_1146` (ρ_inter=−0.464, F_norm=0.857, n_v=15) — 非候选，flag=None
- `epilepsiae_583` (ρ_inter=−0.286, F_norm=0.833, n_v=7) — 非候选
- `yuquan_hanyuxuan` (ρ_inter=−0.380, F_norm=0.810, n_v=22) — 非候选
- `epilepsiae_1077` (ρ_inter=−0.371, F_norm=0.778, n_v=6) — 非候选
- `yuquan_liyouran` (ρ_inter=−0.404, F_norm=0.764, n_v=17) — 非候选

这些 subject 都是 **borderline**：ρ_inter 在 [−0.5, −0.3] 之间，没过 PR-2.5 候选门槛但仍带明显反向几何。把它们与 ρ_inter ≈ 0 的真随机 subject（如 `yuquan_litengsheng` ρ=−0.141）一起塞进"non-reproduced"组会产生伪 bimodality。

**正确分组（采用）**：
- **Group A — 候选 cohort（ρ_inter < −0.5）**：n=7 = 6 TRUE + 1 FALSE（`yuquan_huanghanwen` ρ=−0.527）
- **Group B — 非候选（ρ_inter ≥ −0.5）**：n=16

非候选组里包含几个 borderline subject (ρ_inter ∈ [−0.5, −0.3])，他们的 F_norm 偏高是数据本身的连续谱使然，**不是分组错误**。对感兴趣 borderline subject 的更细分析需要绕开 PR-2.5 二分门槛，做 continuous metric vs continuous outcome 比较，本 supplementary 不做。

### 5.3 可以说 / 不可以说

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
