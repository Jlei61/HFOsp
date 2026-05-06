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

**Visualization contract（post-hoc，不是 pre-registered）**：本节描述 figure 的展示选择。设计经过两轮 user review 后从二分 violin 改为连续 scatter，目的是避免任何"循环展示"——即按一个阈值分组之后再画该指标的 violin（信息量等于 if-条件可视化）。当前 figure 的设计如下。

**Panel B —— ρ_inter vs F_norm scatter**（回答 "PR-2.5 硬阈值 ρ < −0.5 漏掉了哪些 continuous high-reversal subject"）：

- x 轴：`inter_cluster_corr_matrix` Spearman ρ_inter（PR-2.5 metric）
- y 轴：F_norm（D-G 归一化 footrule）
- 参考线：x = −0.5（PR-2.5 候选门槛，FYI），y = 2/3（D-G 渐近随机参考，FYI）—— **都是描述性参考，不是决策规则**
- marker：圆 = PR-2.5 reproduced (n=6)，X = candidate-fail (n=1, `huanghanwen`)，方 = non-candidate (n=16)

23 个 subject 的象限分布：

| 象限 | 含义 | n |
|---|---|---|
| TL（ρ < −0.5 ∩ F > 2/3）| 两个指标都判"高反向" | **7** (6 reproduced + 1 candidate-fail) |
| **TR（ρ ≥ −0.5 ∩ F > 2/3）** | **PR-2.5 硬阈值漏掉的 borderline subject** | **8** (`epi_1150, 1146, 583, hanyuxuan, 1077, liyouran, 590, chengshuai`) |
| BR（ρ ≥ −0.5 ∩ F ≤ 2/3）| 两个指标都判"around random" | 8 |
| BL（ρ < −0.5 ∩ F ≤ 2/3）| 异常情况（无） | 0 |

Spearman ρ(ρ_inter, F_norm) = **−0.963, p = 1.8e−13, n=23** — 两个 anti-correlation 指标共线性极高（这是 expected：都测两个 cluster template 的相反程度），所以 Panel B 的核心信息**不是** "ρ_inter 与 F_norm 谁更好"，而是 **"PR-2.5 在 ρ = −0.5 处砍一刀，正好砍掉了 8 个 F_norm 仍 > 2/3 的 subject"**。

**重要边界**：F_norm > 2/3 不是 "明确呈现真反向" —— 2/3 是 Diaconis-Graham 渐近随机期望（n→∞），**不是显著性阈值**。"F_norm > 2/3" 的正确读法是 "**比均匀随机置换的渐近期望更大的归一化 footrule**"，可作连续谱上的描述性切分点，**不**作"reversal 是否真存在"的判定。

**Panel C —— F_norm vs soz_contribution_excess scatter**（回答 "rank reversal 强的 subject 是否更 SOZ-driven"）：

- x 轴：F_norm
- y 轴：`soz_contribution_excess` (= contribution_fraction − channel_fraction，baseline-corrected)
- 参考线：x = 2/3（D-G 渐近随机，FYI），y = 0（SOZ 在 chance baseline，FYI）
- marker 同 Panel B

Spearman ρ(F_norm, soz_contribution_excess) = **0.193, p = 0.376, n=23** — 几乎不相关。

**这是诚实的 negative finding**：rank reversal 强（F_norm 高）的 subject **并没有**系统性地更高 SOZ enrichment。即使在 F_norm > 2/3 的 15 个 subject 中，soz_contribution_excess 也散布在 [−0.07, +0.19]，median ≈ +0.07，effect size 小。

**正确读法**：

✅ "Geometry reversal 在 cohort 上呈现连续谱（F_norm 0.39 → 1.00），PR-2.5 候选门槛 ρ < −0.5 把谱在 7 vs 16 处一刀切，漏掉 8 个 F_norm > 2/3 的 borderline subject"
✅ "高反向 subject 与 SOZ enrichment 之间无强关联（Spearman ρ = 0.19, p = 0.38）；在 lagPat 选定通道集内，rank reversal 不能由 SOZ-vs-nonSOZ 通道占比来解释"

❌ ~~"F_norm > 2/3 = 真实反向"~~ —— 2/3 是渐近随机期望，不是显著性阈值
❌ ~~"PR-2.5 cohort 与 non-candidate 在 F_norm 上显著分离（MW-U p ~ 1e−4）"~~ —— 按 F_norm 分组再测 F_norm 是循环论证，不要这样写

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

### 3.3 SOZ enrichment vs reversal（continuous，无分组）

cohort 上 23 个 subject 的 (F_norm, soz_contribution_excess) 散布范围：

- F_norm 范围：[0.39, 1.00]
- soz_contribution_excess 范围：[−0.36, +0.19]，median ≈ 0
- soz_channel_fraction median ≈ 0.55（cohort 通道集大约 55% 是 SOZ）

**Spearman ρ(F_norm, soz_contribution_excess) = 0.193, p = 0.376, n=23** —— 几乎不相关。

诚实结论：**rank reversal 强的 subject 并没有系统性的 SOZ enrichment**。即使在高 F_norm 端（F > 2/3, n=15），soz_contribution_excess 散布在 [−0.07, +0.19]，median ≈ +0.07，effect size 小且方向不齐。这意味着在 lagPat 选定通道集内，rank reversal 程度不能由 SOZ-vs-nonSOZ 通道占比解释。Panel C 把这个 negative finding 可视化清楚。

§5.1 的通道选择 caveat 直接限制了任何 SOZ-vs-nonSOZ 比较的解读力度——本 supplementary 用的 lagPat 通道集是 high-HI gate + 高 HFO rate 选出的，**不是**全脑或随机抽样，所以 SOZ vs nonSOZ 的 contribution_excess 反映的是该选定通道集内部的对比，**不是** "SOZ 是不是真的驱动反向"。

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

**当前 figure 的处理（visualization contract，详见 §3）**：弃用任何 binary 分组——既不用 PR-2.5 派生阈值（reproduced/not-reproduced 或 ρ_inter < −0.5），也不用本 supplementary 自己的 F_norm > 2/3 作为分组逻辑（按 F_norm 分组再画 F_norm violin 是循环展示）。改为 ρ_inter vs F_norm 与 F_norm vs SOZ excess 两张连续 scatter，PR-2.5 status 仅作 marker shape 描述性 overlay。这种设计避免了任何 "把 if 条件画一遍" 的低信息量展示。

borderline subject (ρ_inter ∈ [−0.5, −0.3]，F_norm > 2/3) 在 Panel B 的 TR 象限自然现身（`epi_1150, 1146, 583, hanyuxuan, 1077, liyouran, 590, chengshuai`，n=8），无需任何 binary 操作就能可视化"PR-2.5 硬阈值漏掉了哪些 continuous high-reversal subject"。

更细致的 borderline 分析（continuous metric vs continuous outcome 回归、bootstrap CI、per-subject case series 等）属于另一个 PR 的范围，本 supplementary 不做。

### 5.3 可以说 / 不可以说

可以说：
- "Continuous-version footrule + Kendall τ 与 PR-6 离散 swap_node 同向，6/6 fwd/rev-reproduced subject 一致"
- "Geometry reversal 在 cohort 上呈现连续谱（F_norm 0.39 → 1.00），上至 1.00（完全反向边界）下至 0.39（低于 D-G 渐近随机参考 2/3）"
- "PR-2.5 候选门槛 ρ_inter < −0.5 把谱一刀切为 7 vs 16，**漏掉 8 个 F_norm 仍高于 D-G 渐近参考的 borderline subject**（`epi_1150, 1146, 583, hanyuxuan, 1077, liyouran, 590, chengshuai`）"
- "高反向 subject 与 SOZ enrichment 之间无强关联（Spearman ρ(F_norm, soz_contribution_excess) = 0.19, p = 0.38, n=23）；在 lagPat 选定通道集内，rank reversal 程度不能由 SOZ-vs-nonSOZ 通道占比解释"

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
