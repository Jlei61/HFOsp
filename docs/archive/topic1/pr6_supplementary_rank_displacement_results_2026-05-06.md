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

## 3. 主结果（paper-level，单图叙事）

本节按 paper-level supplementary figure 的口径写：只讲科学对象（template-pair rank reversal、SOZ contribution），**不**写 PR-2.5 内部分类系统（reproduced / candidate-fail / non-candidate / candidate gate）。PR-2.5 与本 supplementary 的方法学比对放在 §6。

**最终 paper-level deliverable 是单张 composite figure**：`figures/cohort_displacement_heatmap.{png,pdf}` —— 主热图 + 三条共享 y 轴的连续 summary track。每一行 = 一个 subject，把通道级图案、整体反向程度、SOZ contribution 集中到同一坐标系。前期版本曾把 cohort heatmap 与 reversal spectrum / SOZ scatter 拆成 Panel A 与 Panel B/C 两图，已弃用（详见 §6.3）。

### 3.1 Composite figure layout

| 元素 | 内容 |
|---|---|
| 主热图（左）| 23 subject × 通道 Δr 热图，行按 **F_norm 降序**（最反向在最上），列按 rank_T_a_dense（T_a source → sink），SOZ 通道黑框。**x-axis ticks 与 xlabel 在上方**（下方留给 colorbar）|
| F_norm track（右）| 水平 mini-bar，范围 [0, 1]，虚线参考 **2/3 (Diaconis-Graham 渐近随机期望)**，x-axis 也在上方 |
| Colorbar（主热图正下方，水平）| Signed Δr (= rank_T_b − rank_T_a) 量级 |
| SOZ legend（colorbar 右侧）| 白底黑框小方块 + "SOZ channel" 文字 |

**Kendall τ 不画在主图上**：τ 与 F_norm 在本 cohort Spearman ρ = −0.92，τ bars 与 F_norm bars 视觉镜像一致，信息冗余。τ 数值仍保留在 per-subject JSON、cohort_summary.json 与 archive 文档作 cross-check 用，但不进 paper figure。

**SOZ contribution_excess 不画在主图上**：lagPat 通道集对 SOZ 的覆盖与 SOZ 标注本身（i/l/e 边界）在本 cohort 上还没稳定到能进 paper 图的程度，§3.3 / §5.1 详述。SOZ 统计存在 archive，但不进 paper-level supplementary。

### 3.2 两条线性叙事（每一行同时讲两件事）

1. **Cohort 是连续谱（不是离散二分）**：F_norm Track 从顶部 1.00（`epi_1073, epi_139`）单调递减到底部 0.39（`epi_442`），中间没有自然分界。
2. **真反向看主热图梯度**：最上几行（F_norm > 0.92）呈"红→蓝单调梯度"——因为列轴严格按 rank_T_a_dense 排序，梯度只来自数据本身、不是排序伪影。中段 F_norm ≈ 2/3 的几行颜色散乱、无单调梯度；底部 F_norm < 0.5 的几行 pale，反映两个 template 几乎一致或弱差异。

### 3.3 关键数字（cohort-level）

| 指标 | 范围 | 中位数 | 注 |
|---|---|---|---|
| F_norm | [0.393, 1.000] | ≈ 0.78 | Diaconis-Graham 渐近随机参考点 = 2/3 |
| Kendall τ | [−0.767, +0.429] | ≈ −0.20 | τ = 0 为零相关参考 |

**SOZ 相关数字（不进 paper figure，仅记录）**：
- soz_contribution_excess 范围 [−0.357, +0.190]，median ≈ 0
- soz_channel_fraction median ≈ 0.55（cohort 通道集大约 55% 是 SOZ）
- Spearman ρ(F_norm, soz_contribution_excess) = 0.193, p = 0.376, n=23 — 几乎不相关

但 lagPat 通道选择本身 + SOZ 标注 i/l/e 边界都还没稳定（§5.1, §6）；这些数字是 descriptive only，**不**进 paper-level conclusion。

per-subject Δr 痕迹（按 Kendall τ 升序，沿 T_a source→sink 排序）：

```
epilepsiae_958     τ=-0.767  Δr = [+14, +14, +11, +9, +4, +5, +3, -1, -1, +2, -6, -10, -7, -11, -11, -15]
epilepsiae_635     τ=-0.644  Δr = [+8, +6, +4, +1, +1, -2, +3, -6, -6, -9]
epilepsiae_139     τ=-0.524  Δr = [+4, +4, +4, 0, -4, -3, -5]
epilepsiae_1073    τ=-0.467  Δr = [+5, +2, +2, -3, -3, -3]
yuquan_chenziyang  τ=-0.467  Δr = [+6, +7, +5, +6, -3, -2, -2, -2, -6, -9]
epilepsiae_548     τ=-0.394  Δr = [+11, +7, +8, +6, -1, -4, -4, -3, -2, -4, -3, -11]
```

每行单调梯度都来自数据本身（不是 sorting bias）；τ ≈ 0 的 subject（如 `yuquan_chengshuai` τ=0.000, `epilepsiae_922` τ=+0.429）在同一排序规则下散乱无梯度。

### 3.4 通道选择 caveat（必带）

lagPat 通道集来自 legacy high-HI gate + 高 HFO rate；F_norm 与 Kendall τ 度量的是该选定通道集**之内**的 ranking，不是真实全脑反向程度。SOZ contribution_excess 同理——`soz_channel_fraction ≈ 0.55` 已经反映这种采样偏差，所以 SOZ vs nonSOZ 的对比 **没有进入 paper figure**。详见 §5.1。

## 4. 图

**Paper-level supplementary figure**（单张）：

- `results/interictal_propagation/rank_displacement/figures/cohort_displacement_heatmap.{png,pdf}` — composite figure：23 subject × 通道 Δr 主热图（行按 F_norm 降序，列按 rank_T_a_dense）+ 右侧三条共享 y 轴的连续 summary track（F_norm + 2/3 ref, Kendall τ + 0 ref, SOZ contribution_excess + 0 ref）+ Δr colorbar

**Debug / supplement only**（不进 paper）：

- `results/interictal_propagation/rank_displacement/figures/per_subject/<subject>_displacement.png` — 每 subject zoom-in 详图（共 23 张）
- `results/interictal_propagation/rank_displacement/figures/README.md` — 中文图说明

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

### 5.2 可以说 / 不可以说

**可以说**：
- "Template-pair rank reversal 在本 cohort 上呈连续谱（F_norm 0.39 → 1.00），不是离散二分"
- "高反向 subject 与 SOZ enrichment 无强关联（Spearman ρ(F_norm, soz_contribution_excess) = 0.19, p = 0.38, n=23）；在 lagPat 选定通道集内，rank reversal 程度不能由 SOZ-vs-nonSOZ 通道占比解释"
- "F_norm 范围 [0.39, 1.00] 中位数 ~0.78；Kendall τ 范围 [−0.77, +0.43] 中位数 ~−0.20"

**不**可以说：
- ~~"反向 template 是抑制墙的反弹"~~ — HFO 80–250 Hz 不区分 E/I
- ~~"F_norm 高 ⇒ 致痫"~~ — 没做任何疾病侧 outcome 检验
- ~~"forward template SOZ-leading"~~ — 这是 PR-8 v1（DEFERRED）的范围，本 supplementary 不做 SOZ 极性方向判读
- ~~"SOZ contribution_fraction = X% ⇒ SOZ 主导"~~ — 必须用 `contribution_excess` 或 `abs_mean` 比较，裸 fraction 受通道数 confound
- ~~"F_norm > 2/3 = 真实反向"~~ — 2/3 是 n→∞ 渐近期望，不是显著性阈值
- ~~"signed Δr 在 SOZ 通道为正 / 负"~~ — signed Δr 只在 subject 内部有效，跨 subject 聚合方向无意义

## 6. 方法学背景：与 PR-2.5 / PR-6 离散结果的对比

> 本节是开发日志，**不**进 paper-level supplementary figure caption。Paper 图只讲连续谱（§3），PR-2.5 / PR-6 的内部分类系统在这里独立记录。

### 6.1 与 PR-6 离散 swap_node 的 cross-check

PR-6 Step 4b sign-test n=6, p=0.031（fwd/rev-reproduced subset）。本 supplementary 在同一 6 个 subject 上的 Kendall τ：

- 6/6 subject Kendall τ < 0（100% direction consistency）
- median τ = −0.495；range = [−0.767, −0.394]
- 所有 6 subject 的 PR-6 `pr6_swap_score` 都 ≥ 0.35（H2 swap 方向已建）

→ **continuous metric 与 PR-6 离散 swap_node 同向，6/6 subject 一致**。这不是新的独立检验（同一 cohort），是对 PR-6 离散结果的 continuous-axis 补充。

**Independent cohort sanity**：本 supplementary 的 reproduced n=6 是从 PR-2 / PR-2.5 `time_split_reproducibility.splits` (OR rule) 与 PR-6 `template_anchoring/per_subject/*.json` 的 `valid_mask` 交集独立 derive 出来的，**与 PR-6 plan §15 H2 cohort 定义** (`endpoint_defined ∩ forward_reverse_reproduced`) **完全相同**（PR-6 plan 中 "forward/reverse-reproduced subset n=6" 在 §15 Step 4b）。这是对 cluster_id × valid_mask alignment 合同正确性的 cross-check —— 两个独立 runner 在不同字段路径上聚到同一个 6 subject cohort。

### 6.2 与 PR-2.5 二步嵌套 binary 的对比

PR-2.5 `forward_reverse_reproduced` flag 的逻辑是两步嵌套：

1. 候选门槛：`inter_cluster_corr_matrix` Spearman ρ_inter < −0.5 → 候选
2. 候选 subject 跑 reproducibility 测试（split-half + odd-even）→ TRUE / FALSE
3. 非候选 subject 未跑测试 → flag = None

把 PR-2.5 的二步嵌套 binary 投影到本 supplementary 的连续 F_norm 谱上：

- **TRUE (n=6)**：候选 + reproducibility 通过。F_norm 全部 ≥ 0.92
- **FALSE (n=1, `huanghanwen`)**：候选但 reproducibility 失败。F_norm = 0.880
- **None (n=16)**：非候选（ρ_inter ≥ −0.5），未跑测试

**关键观察**：在 None 组中存在 borderline subject（ρ_inter ∈ [−0.5, −0.3]，但 F_norm 仍 > 2/3）：

| subject | ρ_inter | F_norm | n_v |
|---|---|---|---|
| `epilepsiae_1146` | −0.464 | 0.857 | 15 |
| `epilepsiae_583` | −0.286 | 0.833 | 7 |
| `yuquan_hanyuxuan` | −0.380 | 0.810 | 22 |
| `epilepsiae_1077` | −0.371 | 0.778 | 6 |
| `yuquan_liyouran` | −0.404 | 0.764 | 17 |

PR-2.5 候选门槛 ρ_inter < −0.5 在这些 subject 上一刀切，把它们划到 None 组，**忽略**他们仍带明显反向几何。这说明：PR-2.5 的二分门槛是为 reproducibility 测试设计的 gate，**不**等价于 "reversal 是否存在" 的判定。

### 6.3 figure 设计演化（开发日志）

本 supplementary figure 经过六轮 user review 才收敛到 paper-level 单图设计：

1. v1：reproduced/not-reproduced binary violin → user 指出"未测候选 + 测了不过"被混在一起
2. v2：candidate vs non-candidate (ρ < −0.5) violin → user 指出 0.5 也是 PR-2.5 的硬阈值
3. v3：F_norm > 2/3 binary violin → user 指出按 F_norm 分组再画 F_norm violin 是循环展示
4. v4：ρ_inter vs F_norm + F_norm vs SOZ excess 双 scatter → user 指出 Panel B 的 PR-2.5 内部 workflow 不该出现在 paper 图
5. v5：ranked F_norm spectrum + F_norm vs SOZ excess scatter，去 PR-2.5 → user 指出仍把 cohort heatmap 与 reversal/SOZ 拆成两图，信息散在三个坐标系，不够干净
6. v6：单张 composite figure —— cohort heatmap + 三条共享 y 轴 summary track（F_norm, Kendall τ, SOZ excess）+ 右侧 vertical colorbar → user 指出 SOZ 列在本 cohort 上还没稳定，不该进 paper 图；同时建议 colorbar 移到主热图正下方
7. v7：单张 composite figure，保留两条 summary track（F_norm + Kendall τ），SOZ 列移除；colorbar 水平放置在主热图下方 → user 进一步指出 τ 与 F_norm 共线、track 冗余；同时 SOZ legend 与 τ 列重叠；colorbar 在下方了所以 heatmap x-axis 应移到上方
8. **v8（当前）**：单张 composite figure，**只保留 F_norm 一条 track**（τ 列也去掉）；heatmap x-axis ticks + xlabel **移到主热图上方**；SOZ legend 移到 colorbar 右侧（不再与 summary track 重叠）。τ 数值仍在 archive 文档作 cross-check 保留。

每一轮的具体批评见 worktree git log；当前归档文档与 figure 已收敛到 v8。

## 7. 历史链接

- `docs/topic1_within_event_dynamics.md` §7 — Topic 1 主文档
- `docs/archive/topic1/pr6_template_endpoint_anchoring_plan_2026-04-25.md` — PR-6 plan（上游 swap_node 离散合同）
- `docs/archive/topic1/ping_pong_hypothesis_review_2026-04-28.md` — PR-8 candidate 来源
- `docs/archive/topic1/pr6_supplementary_rank_displacement_plan_2026-05-06.md` — 本 supplementary 的 plan
