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

## 2. Cohort（**完整 stable_k=2 cohort，n=35**）

> **Cohort 演化**：
> - v1–v12（2026-05-06 最初~第十二轮）使用 PR-6 endpoint-defined cohort（n=23），因 PR-6 anchoring 缺 4 个 SOZ-空 subject 而被裁
> - v13（2026-05-06 末）改为**直接从 PR-2 `template_rank` sentinel 派生 valid_mask**（rank ≠ −1），覆盖**全部当时 27 个 stable_k=2 subject**
> - **v14（2026-05-07，当前）**：上游 PR-2 流水线补全 yuquan，新加 10 个 subject（gaolan, pengzihang, songzishuo, wangyiyang, zhangbichen, zhangjiaqi, zhangkexuan, zhaochenxi 共 8 个 stable_k=2，外加 zhaojinrui/zhourongxuan 是 stable_k=5 不进 cohort）。PR-2 cohort 从 30 → 40，stable_k=2 cohort 从 27 → **35**。PR-6 anchoring 同步重跑覆盖 30/35（新加 7 个 yuquan stable_k=2 都进了 PR-6）；剩 5 个仍走 PR-2 sentinel fallback（SOZ JSON 空）

PR-2 cohort 全景（40 subject）：

| stable_k | n | subject 列表 |
|---|---|---|
| 2 | **35** | `epilepsiae_{1073, 1077, 1084, 1096, 1125, 1146, 1150, 139, 253, 384, 442, 548, 583, 590, 620, 635, 916, 922, 958}`（19 个）+ `yuquan_{chengshuai, chenziyang, gaolan, hanyuxuan, huanghanwen, litengsheng, liyouran, pengzihang, songzishuo, sunyuanxin, wangyiyang, xuxinyi, zhangbichen, zhangjiaqi, zhangkexuan, zhaochenxi}`（16 个）|
| 4 | 2 | `epilepsiae_818`, `yuquan_huangwanling` |
| 5 | 2 | `yuquan_zhaojinrui`, `yuquan_zhourongxuan` |
| 6 | 1 | `yuquan_zhangjinhan` |

本 supplementary cohort = **全部 35 个 stable_k=2 subject**。5 个 stable_k≠2 不在 paper-level 主热图范围（cohort_summary.json 里仍写入 pairs 数据，但不进 figure）。

**valid_mask 来源** (`valid_mask_provenance` 字段 in per-subject JSON)：
- 30 subject 从 PR-6 `per_template[k].valid_mask` 取（`pr6` provenance）
- 5 subject 从 PR-2 `template_rank` sentinel 派生 = `(rank ≠ −1)`（`pr2_sentinel` provenance）：`epilepsiae_{1125, 384, 620, 916}` + `yuquan_gaolan`。这 5 个在 PR-6 anchoring 阶段因 SOZ JSON 条目为空被裁掉

**两种 provenance 等价性已 cross-check**：在 PR-6 与 PR-2 都覆盖的 subject 上（如 `epi_1073`），`(template_rank ≠ −1)` 与 PR-6 `valid_mask` 完全一致——都基于"该 cluster 的事件参与门槛"。所以从 PR-2 派生 valid_mask 不是放宽合同，是去掉对 PR-6 SOZ 流程的不必要依赖。

forward/reverse-reproduced 在 PR-2.5 内有三种 outcome（不是简单二分）：

| PR-2.5 outcome | 含义 | n（n=35 cohort） |
|---|---|---|
| **TRUE** | `inter_cluster_corr_matrix` Spearman ρ < −0.5（候选）AND split-half OR odd-even reproducibility 通过 | **11** |
| **FALSE** | 候选（ρ < −0.5）但 reproducibility 测试失败 | 1（`yuquan_huanghanwen`，ρ = −0.527）|
| **None** | 非候选（ρ ≥ −0.5），未跑 reproducibility 测试 | 23 |

**关键**：本 supplementary 的 cohort 划分按 **PR-2.5 候选门槛**（fwd/rev cohort = TRUE+FALSE = n=12）vs **非候选**（None, n=23），**不**按裸的 reproduced/not-reproduced binary 划分（plan 早期版本采用过的错误划分会让 borderline ρ ≈ −0.4 但 F_norm ≈ 0.8 的 subject 错误归到"未反向"组）。

reproduced 11 个 subject（v14 cohort）：原 v13 的 6 个（`epi_1073, 139, 548, 635, 958, yuquan_chenziyang`）+ v13 新增的 2 个 PR-2-sentinel（`epi_620, 1125`）+ v14 新增的 yuquan（`zhangjiaqi, wangyiyang, zhaochenxi`）。candidate-fail 1 个：`yuquan_huanghanwen`。

## 3. 主结果（paper-level，单图叙事）

本节按 paper-level supplementary figure 的口径写：只讲科学对象（template-pair rank reversal、SOZ contribution），**不**写 PR-2.5 内部分类系统（reproduced / candidate-fail / non-candidate / candidate gate）。PR-2.5 与本 supplementary 的方法学比对放在 §6。

**最终 paper-level deliverable 是单张 composite figure**：`figures/cohort_displacement_heatmap.{png,pdf}` —— 主热图 + 三条共享 y 轴的连续 summary track。每一行 = 一个 subject，把通道级图案、整体反向程度、SOZ contribution 集中到同一坐标系。前期版本曾把 cohort heatmap 与 reversal spectrum / SOZ scatter 拆成 Panel A 与 Panel B/C 两图，已弃用（详见 §6.3）。

### 3.1 Composite figure layout

| 元素 | 内容 |
|---|---|
| 主热图（左）| 35 subject × 通道 Δr 热图，行按 **F_norm 降序**（最反向在最上），列按 rank_T_a_dense（T_a source → sink）。**x-axis ticks 与 xlabel 在上方**（下方留给 colorbar）|
| F_norm track（右）| 水平 mini-bar，范围 [0, 1]，虚线参考 **2/3 (Diaconis-Graham 渐近随机期望)**，x-axis 也在上方 |
| Colorbar（主热图正下方，水平）| Signed Δr (= rank_T_b − rank_T_a) 量级 |

**Kendall τ 不画在主图上**：τ 与 F_norm 在本 cohort Spearman ρ = **−0.916（n=35, p ≈ 1e-14）**，τ bars 与 F_norm bars 视觉镜像一致，信息冗余。τ 数值仍保留在 per-subject JSON、cohort_summary.json 与 archive 文档作 cross-check 用，但不进 paper figure。

**SOZ contribution_excess 不画在主图上**：lagPat 通道集对 SOZ 的覆盖与 SOZ 标注本身（i/l/e 边界）在本 cohort 上还没稳定到能进 paper 图的程度，§3.3 / §5.1 详述。SOZ 统计存在 archive，但不进 paper-level supplementary。

### 3.2 两条线性叙事（每一行同时讲两件事）

1. **Cohort 是连续谱（不是离散二分）**：F_norm Track 从顶部 1.00（`epi_1073, epi_139, epi_620, yuq_zhangjiaqi` 4 个 perfect reversal）单调递减到底部 0.39（`yuq_pengzihang`），中间没有自然分界。**23/35 subject above 2/3** truly reversed；**12/35 ≤ 2/3** 落在随机 null 区。
2. **真反向看主热图梯度**：最上几行（F_norm > 0.92）呈"红→蓝单调梯度"——因为列轴严格按 rank_T_a_dense 排序，梯度只来自数据本身、不是排序伪影。中段 F_norm ≈ 2/3 的几行颜色散乱、无单调梯度；底部 F_norm < 0.5 的几行 pale，反映两个 template 几乎一致或弱差异。

### 3.3 关键数字（cohort-level，n=35）

| 指标 | 范围 | 中位数 | 注 |
|---|---|---|---|
| F_norm | [0.389, 1.000] | **0.800** | Diaconis-Graham 渐近随机参考点 = 2/3；23/35 above |
| Kendall τ | [−0.810, +0.429] | **−0.203** | τ = 0 为零相关参考 |

cohort-level 关联：**Spearman ρ(F_norm, Kendall τ) = −0.916, p = 1.17e-14, n=35**（continuous metric 与 τ 高度共线，τ 不进 paper figure）。

PR-2.5 forward/reverse-reproduced flag 在 v14 n=35 cohort 上：
- **11 reproduced**：原 v13 的 6 个（`epi_1073, 139, 548, 635, 958, chenziyang`）+ v13 新增 PR-2-sentinel 2 个（`epi_620, 1125`，全都集中在 F_norm 高端）+ v14 yuquan 新增 3 个（`zhangjiaqi, wangyiyang, zhaochenxi`）
- 1 candidate-fail：`yuq_huanghanwen`
- 23 None（非候选或不可测）

**SOZ 相关数字（不进 paper figure，仅记录）**：
- soz_contribution_excess 中位数 ≈ 0.035（n=30，5 个 PR-2-sentinel subject 因 SOZ JSON 空无法计算 → NaN，从该统计中剔除）
- Spearman ρ(F_norm, soz_contribution_excess) = **0.239, p = 0.203, n=30** — 几乎不相关

但 lagPat 通道选择本身 + SOZ 标注 i/l/e 边界都还没稳定（§5.1, §6）；这些数字是 descriptive only，**不**进 paper-level conclusion。

per-subject Δr 痕迹（按 Kendall τ 升序，沿 T_a source→sink 排序，前 6 个最反向 subject）：

```
epilepsiae_958     τ=-0.767  Δr = [+14, +14, +11, +9, +4, +5, +3, -1, -1, +2, -6, -10, -7, -11, -11, -15]
epilepsiae_635     τ=-0.644  Δr = [+8, +6, +4, +1, +1, -2, +3, -6, -6, -9]
epilepsiae_139     τ=-0.524  Δr = [+4, +4, +4, 0, -4, -3, -5]
epilepsiae_1073    τ=-0.467  Δr = [+5, +2, +2, -3, -3, -3]
yuquan_chenziyang  τ=-0.467  Δr = [+6, +7, +5, +6, -3, -2, -2, -2, -6, -9]
epilepsiae_548     τ=-0.394  Δr = [+11, +7, +8, +6, -1, -4, -4, -3, -2, -4, -3, -11]
```

每行单调梯度都来自数据本身（不是 sorting bias）；τ ≈ 0 的 subject（如 `yuquan_chengshuai` τ=0.000, `epilepsiae_922` τ=+0.429）在同一排序规则下散乱无梯度。新加 yuquan subject 的 Δr 数值见 per-subject JSON。

### 3.4 通道选择 caveat（必带）

lagPat 通道集来自 legacy high-HI gate + 高 HFO rate；F_norm 与 Kendall τ 度量的是该选定通道集**之内**的 ranking，不是真实全脑反向程度。SOZ contribution_excess 同理——`soz_channel_fraction ≈ 0.55` 已经反映这种采样偏差，所以 SOZ vs nonSOZ 的对比 **没有进入 paper figure**。详见 §5.1。

## 4. 图

**Paper-level supplementary figure**（单张）：

- `results/interictal_propagation/rank_displacement/figures/cohort_displacement_heatmap.{png,pdf}` — composite figure：35 subject × 通道 Δr 主热图（行按 F_norm 降序，列按 rank_T_a_dense）+ 右侧 F_norm summary track（共享 y 轴，2/3 random null reference）+ 主热图正下方水平 Δr colorbar

**Debug / supplement only**（不进 paper）：

- `results/interictal_propagation/rank_displacement/figures/per_subject/<subject>_displacement.png` — 每 subject zoom-in 详图（共 35 张）
- `results/interictal_propagation/rank_displacement/figures/swap_cardinality_heatmap.{png,pdf}` — Variable-k swap classifier supplementary figure（2026-05-07 增量 v2，FW max-null；详见 §8）。**v1 的 `swap_classification.{png,pdf}`（per-k null + scatter+curves 双面板）已弃用并删除**
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
- "高反向 subject 与 SOZ enrichment 无强关联（Spearman ρ(F_norm, soz_contribution_excess) = 0.239, p = 0.203, n=30）；在 lagPat 选定通道集内，rank reversal 程度不能由 SOZ-vs-nonSOZ 通道占比解释"
- "F_norm 范围 [0.39, 1.00] 中位数 ~0.80；Kendall τ 范围 [−0.81, +0.43] 中位数 ~−0.20（n=35）"

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

PR-6 Step 4b sign-test n=6, p=0.031（fwd/rev-reproduced subset）。本 supplementary 在 v14 reproduced cohort（n=11）上的 Kendall τ：

- **11/11 subject Kendall τ < 0**（100% direction consistency）
- median τ = **−0.524**；range = [−0.810, −0.394]
- 9/11 有 PR-6 `pr6_swap_score`（2 个 PR-2-sentinel subject `epi_620, 1125` 无 PR-6 anchoring 数据，swap_score=None）；其中 9/9 swap_score > 0
- sign-test (τ<0)：n=11, p = 0.0005 — 比 v13/PR-6 的 p=0.031 更强

→ **continuous metric 与 PR-6 离散 swap_node 同向，11/11 subject 一致**。这不是独立检验（PR-2.5 `forward_reverse_reproduced` flag 本身已经 encode 了"两 template 反 correlated 且 reproducible"，11/11 τ<0 部分由构造保证），是对 PR-6 离散结果的 continuous-axis 补充。

**Independent cohort sanity**：本 supplementary 的 reproduced n=11 是从 PR-2 / PR-2.5 `time_split_reproducibility.splits` (OR rule) 与 PR-6 `template_anchoring/per_subject/*.json` 的 `valid_mask`（缺失时 PR-2 sentinel）交集独立 derive 出来的。**v14 比 v13 多 5 个**（`epi_620, epi_1125` 因 PR-2 sentinel 入选；`yuq_zhangjiaqi, wangyiyang, zhaochenxi` 是 v14 上游补全 yuquan 后新出现的 reproduced subject）。这是对 cluster_id × valid_mask alignment 合同在 cohort 扩展后仍然 self-consistent 的 cross-check。

### 6.2 与 PR-2.5 二步嵌套 binary 的对比

PR-2.5 `forward_reverse_reproduced` flag 的逻辑是两步嵌套：

1. 候选门槛：`inter_cluster_corr_matrix` Spearman ρ_inter < −0.5 → 候选
2. 候选 subject 跑 reproducibility 测试（split-half + odd-even）→ TRUE / FALSE
3. 非候选 subject 未跑测试 → flag = None

把 PR-2.5 的二步嵌套 binary 投影到本 supplementary 的连续 F_norm 谱上（v14 cohort）：

- **TRUE (n=11)**：候选 + reproducibility 通过。F_norm 全部 ≥ 0.84
- **FALSE (n=1, `huanghanwen`)**：候选但 reproducibility 失败。F_norm = 0.880
- **None (n=23)**：非候选（ρ_inter ≥ −0.5），未跑测试

**关键观察**：在 None 组中存在 borderline subject（ρ_inter ∈ [−0.5, −0.3]，但 F_norm 仍 > 2/3）：

| subject | ρ_inter | F_norm | n_v |
|---|---|---|---|
| `epilepsiae_1150` | −0.483 | 0.900 | 9 |
| `epilepsiae_1146` | −0.464 | 0.857 | 15 |
| `yuquan_hanyuxuan` | −0.380 | 0.810 | 22 |
| `epilepsiae_916` | −0.371 | 0.889 | 6 |
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
8. v8：单张 composite figure，只保留 F_norm 一条 track（τ 列也去掉）；heatmap x-axis ticks + xlabel 移到主热图上方；SOZ legend 移到 colorbar 右侧（不再与 summary track 重叠）。
9. v9：整体字号放大；colorbar 与主图距离收紧；F_norm 列与主热图轻微分隔
10. v10：NaN 填充由灰色改为白色（`cmap.set_bad("white")`），更干净
11. v11：source/sink 字号继续放大；F_norm track 加 random null zone shading + rust 2/3 ref，让 2/3 显著
12. v12：去掉 heatmap top frame、SOZ cell outlines、SOZ channel legend → 只剩纯色格子的视觉流
13. v13：**Cohort 从 23 (PR-6 endpoint-defined) 扩到全部 27 个 stable_k=2 subject**。runner 改为优先用 PR-6 `valid_mask`，缺失时 fallback 到 PR-2 `template_rank` sentinel `(rank ≠ −1)`。验证 PR-2 sentinel 与 PR-6 `valid_mask` 在共有 subject 上完全一致。新加入的 4 个 subject 中 `epi_620` (F=1.000, τ=−0.778) 与 `epi_1125` (F=0.875, τ=−0.714) 都是 PR-2.5 forward/reverse-reproduced，本应在主热图最上几行——之前因 PR-6 SOZ-空过滤错失。
14. **v14（当前，2026-05-07）**：**上游 PR-2 流水线补全 yuquan**（新加 10 个 subject：gaolan, pengzihang, songzishuo, wangyiyang, zhangbichen, zhangjiaqi, zhangkexuan, zhaochenxi 共 8 个 stable_k=2，+ zhaojinrui/zhourongxuan 是 stable_k=5 不进 cohort）。**PR-6 anchoring 同步重跑**覆盖 30/35（新加 7 个 yuquan stable_k=2 都进了 PR-6；剩 5 个仍走 PR-2 sentinel fallback）。**Cohort 从 27 → 35**。新加入 reproduced subject：`yuq_zhangjiaqi` (F=1.000)、`yuq_wangyiyang` (F=0.992)、`yuq_zhaochenxi` (F=0.952)。**关键统计在 n=35 上保持稳定**：ρ(F_norm, τ) = −0.916 (n=27 时 −0.92)；ρ(F_norm, SOZ excess) = 0.239, p = 0.203 (n=30 有效 SOZ overlap)。**结论无任何变化**：连续谱、23/35 above 2/3、SOZ 弱关联——都与 v13 同向。

每一轮的具体批评见 worktree git log；当前归档文档与 figure 已收敛到 v14。

## 8. Variable-k swap classifier（2026-05-07 增量；FW max-null 修订）

> **状态：supplementary 内的二级 deliverable，不开新的 cohort claim。** 复用 PR-6
> H2 swap_score 公式但把 endpoint cardinality 从 hard-coded `k=3` 改成数据驱动
> 扫描 `k ∈ {2 .. ⌊n_valid/2⌋}`。motivation：PR-6 top-3 固定阈值在描述层面无法
> 表达"只有 2 通道或 5+ 通道反转"的 subject。
>
> **Hypothesis tier 不变**：PR-6 plan §3.3 把 H2 forward/reverse swap
> registered as *directional mechanism sanity, not cohort claim*。本 classifier
> 输出仍然是描述性的"swap-like geometry detected at the data-driven k"，不是
> SOZ-anchoring 或 disease-side outcome。**PR-2.5 fwd/rev-reproduced flag 不是
> ground truth**——它来自 inter-cluster Spearman ρ < −0.5 + 时间分半 reproducibility，
> 与 swap-overlap 测度的是不同 construct。本 classifier 与 PR-2.5 的吻合或不吻合
> 都仅作 cross-check，不构成对任一方的"validation"。
>
> **本节修订**：旧版本（v1，2026-05-07 早些时间）使用 per-k null 95th 分位 +
> "存在某 k 同时通过 0.5 floor 与 per-k null"作为判定，**不做跨 k family-wise
> 校正**，假阳性会随 k_max 膨胀。当前 v2（本节）改为 family-wise max-null：
> 1000 次 permutation 内每次取 `T_perm = max_k swap_score_perm(k)`，与观测
> `T_obs = max_k swap_score(k)` 比较，得 `p_fw`。这把 argmax_k 与 decision_k
> 合并成单一统计量。

### 8.1 度量定义

对每对 cluster template (T_a, T_b)（T_a = 较小 cluster_id），逐 k 算 PR-6 风格的
swap_score：

```
swap_score(k) = ½ · [ Jaccard(top_k(T_a), bottom_k(T_b))
                    + Jaccard(bottom_k(T_a), top_k(T_b)) ]
```

family-wise null（避免跨 k 多重比较）：

```
对每次 permutation i ∈ [1, n_perm]:
    rb_perm[i] = shuffle(rank_b on joint_valid)
    T_perm[i]  = max_k swap_score(rank_a, rb_perm[i], k)

T_obs       = max_k swap_score(rank_a, rank_b, k)
decision_k  = argmin_k k subject to swap_score(k) == T_obs   # 同分取最小 k
p_fw        = (1 + sum_i [T_perm[i] >= T_obs]) / (n_perm + 1)   # 1-sided + smoothing
```

**Decision rule（双轨，user-locked 2026-05-07 v3）**：

```
swap_class = "strict"    if T_obs >= score_floor=0.5 AND p_fw < alpha_fw=0.05
           = "candidate" if T_obs >= score_floor=0.5 AND alpha_fw <= p_fw < alpha_candidate=0.20
           = "none"      otherwise
has_swap   = (swap_class == "strict")   # backward-compat alias
```

**双轨语义**：
- **Tier 1 strict**（FW α=0.05）：通过 family-wise gate，可作 subject-level binary
  label，可作 channel-level `swap_endpoint_candidate` label（**channel label 后续
  若进入富集 / 模型必须 split-half 验证**：一半数据选 k 与通道，另一半验证；否则
  在同一 cohort 上既挑 label 又证明 label 是循环）
- **Tier 2 candidate**（FW 0.05 ≤ α < 0.20）：几何上有 swap 形状但 FW 力不足，
  仅作 descriptive / exploratory，**不**得用作 channel label 或下游统计
- **none**：T_obs < 0.5 或 p_fw ≥ 0.20，不报 swap label

α=0.20 是 mechanism-sanity tier 上 exploratory 通用 cutoff。再宽就和 T_obs ≥ 0.5
floor 重叠不再有 null 信息；α=0.10 (n=14/35) 不覆盖"大部分 subject"。这是双轨
覆盖与统计严格性之间的工程折衷。

> **0.5 floor 的真实语义**：`swap_score = 0.5 = ½(J₁ + J₂)` 只表示两个端点-集合
> Jaccard **的平均**达到一半，**不**严格等价于"至少一半节点完成完整 swap"。
> 例如 (J₁=1, J₂=0)、(J₁=0.5, J₂=0.5) 都给 swap_score = 0.5 但节点-级别图景
> 完全不同。0.5 是"endpoint-set overlap 平均达到半数"的分布参考，不是节点级
> 完整反转的下界。这条 floor 与 max-null FW gate 联合使用，意在排除
> "极小 score 但偶然显著"的 trivially-small-overlap 通过情形。

### 8.2 决定性 / 复现性

`compute_swap_score_sweep` 用 `numpy.default_rng(seed)` 固定 RNG。每个 pair
JSON 的 `swap_sweep` 子结构写入 `seed`、`n_perm`、`alpha_fw`、`score_floor` 四
个字段；同 (seed, n_perm, alpha_fw, score_floor) 输入下 (T_obs, p_fw, decision_k,
has_swap) bit-reproducible。当前 cohort run：seed=0, n_perm=1000, alpha_fw=0.05,
score_floor=0.5。

### 8.3 cohort-level 结果（FW max-null 双轨, n=35）

| swap_class | n | 占比 | 与 PR-2.5 fwd/rev 关系 |
|---|---|---|---|
| **strict** | **10** | 29% | 8 reproduced + 0 candidate-fail + 2 non-candidate |
| **candidate** | **8** | 23% | 3 reproduced + 1 candidate-fail + 4 non-candidate |
| none | 17 | 49% | 0 reproduced + 0 candidate-fail + 17 non-candidate |

**strict ∪ candidate = 18/35**（≈ 51%，覆盖到"大部分"）。

**叙述（避免把 PR-2.5 当 ground truth）**：

Tier 1 strict（n=10）：`epi_1073, epi_139, epi_620, epi_635, epi_958, epi_1146,
yuq_hanyuxuan, yuq_wangyiyang, yuq_zhangjiaqi, yuq_zhaochenxi`。其中 8 个落在
PR-2.5 fwd/rev-reproduced 子集；2 个 PR-2.5 非候选（`epi_1146 dk=7,
yuq_hanyuxuan dk=6`）通过 FW gate——这是 swap-overlap construct 给出而 PR-2.5
inter-cluster Spearman ρ < −0.5 + reproducibility construct 给不出的信息。

Tier 2 candidate（n=8）：`epi_1125 (dk=2, p_fw=0.067), epi_548 (dk=4, 0.057),
yuq_zhangkexuan (dk=13, 0.056), yuq_liyouran (dk=3, 0.093), yuq_chenziyang
(dk=4, 0.112), epi_1150 (dk=3, 0.117), yuq_huanghanwen (dk=4, 0.135),
epi_384 (dk=2, 0.174)`。其中 3 个是 PR-2.5 fwd/rev-reproduced（`epi_1125, epi_548,
yuq_chenziyang`）但 FW 力不足。**`epi_1125` 与 `epi_384` 都落在 dk=2**——user
motivation 中"2 节点 swap"在 candidate tier 上能被捕获（在 strict tier 上不
通过 α=0.05）。

PR-2.5 fwd/rev-reproduced（n=11）在双轨上的分布：8 strict + 3 candidate + 0 none。
**3 reproduced 落在 candidate 而非 strict**，这是 swap-overlap 与 PR-2.5
inter-cluster ρ + reproducibility 在 borderline subject 上的分歧——不是 classifier
失效，是测量不同 construct。

none（n=17）里 4 个 F_norm > 2/3（`epi_1077=0.778, 916=0.889, 583=0.833`，加 1
不到 2/3 的）—— 整体 rank 部分反转但没有 concentrated endpoint swap。F_norm + 
swap_class 联合给出比单独 F_norm 更精细的几何 typology。

→ swap_class 与 PR-2.5 fwd/rev-reproduced 是测两个不同 construct 的 flag，**互不验证**；
分歧本身（3 reproduced 在 candidate；2 non-candidate 在 strict）就是科学信息。
**不**得用 "recall / specificity" 表述。

### 8.4 decision_k 分布（strict + candidate, n=18）

| decision_k | strict | candidate | subjects |
|---|---|---|---|
| 2 | 0 | 2 | candidate: `epi_1125, epi_384` |
| 3 | 4 | 2 | strict: `epi_1073, epi_139, epi_635, yuq_zhangjiaqi`; candidate: `epi_1150, yuq_liyouran` |
| 4 | 1 | 3 | strict: `epi_620`; candidate: `epi_548, yuq_chenziyang, yuq_huanghanwen` |
| 6 | 2 | 0 | strict: `epi_958, yuq_hanyuxuan` |
| 7 | 1 | 0 | strict: `epi_1146` |
| 11 | 1 | 0 | strict: `yuq_wangyiyang` |
| 12 | 1 | 0 | strict: `yuq_zhaochenxi` |
| 13 | 0 | 1 | candidate: `yuq_zhangkexuan` |

**strict ∪ candidate 中 12/18 (67%) decision_k > 3**——PR-6 hard-coded top-3 在
描述层面无法表达这些 subject 的 swap geometry。这是变 k classifier 对 PR-6 离散
swap_node 的描述补充价值。

**user motivation "2 节点 swap"**：strict tier 在 FW α=0.05 下没有 subject 通过；
candidate tier 上 `epi_1125 (T_obs=0.667, p_fw=0.067)` 与 `epi_384 (T_obs=0.500,
p_fw=0.174)` 落在 dk=2。**所以"2 节点 swap" 在 cohort 层面**：在 strict 标准下
无显著支持；在 candidate 标准下有 2 例 exploratory 信号——前者写进结论，后者只
写进 archive 描述。

> **重要 caveat**：decision_k 不能直接解读为"swap 涉及多少节点"。这是 argmax
> 位置，且当多个 k 同时给 T_obs 时取最小 k（如完美反向 F_norm=1.0 的几个 subject
> 在所有合法 k 上都 saturate，decision_k=3 仅是 tie-break 结果，不代表"只有 3
> 节点反向"）。把 decision_k 读成 swap cardinality 估计是错误的；它只是回答
> "在哪个 k 上 swap_score 最大"。large decision_k（如 11/12）表示 swap_score
> 在小 k 上没有 saturate，几何更分散——这一类用法是合理的。

### 8.5 与 F_norm 的关系（不重复 §3）



F_norm > 2/3 是 has_swap=True 的**必要不充分**条件：
- has_swap=True 全部 F_norm ≥ 0.81（最小 `yuq_hanyuxuan` 0.810）
- has_swap=False 中存在多个 F_norm > 2/3 但 FW gate 不通过的 subject（如
  `epi_916=0.889, epi_583=0.833, epi_1125=0.875, epi_548=0.889, yuq_chenziyang=0.960`）
  —— 整体 rank 不一致 / 部分反转，但 endpoint-set overlap 集中度不足以通过 FW gate

→ **F_norm 是连续 reversal magnitude；has_swap 是 endpoint-overlap 集中度的离散判定**。
两者互补，不能彼此取代。F_norm + has_swap 联合给出比单独 F_norm 更精细的几何 typology。

### 8.6 文件

- `src/rank_displacement.py::compute_swap_score_at_k`，`::compute_swap_score_sweep`（v3 with FW max-null + dual-tier strict/candidate）
- `scripts/run_rank_displacement.py`：每 pair 写入 `swap_sweep` 子结构（含 seed/n_perm/alpha_fw/alpha_candidate/score_floor 全合同字段；`swap_class` ∈ {strict, candidate, none}；`has_swap` 为 `swap_class == "strict"` 的 alias）
- `scripts/plot_rank_displacement.py`：
  - `plot_cohort_heatmap`（`--what cohort`）—— **paper-level 主图**，每行在 leftmost decision_k 与 rightmost decision_k 个 cell 上画圈：strict 实线黑圈，candidate 虚线灰圈
  - `plot_swap_cardinality_heatmap`（`--what swap`）—— supplementary debug 图，subject × k swap_score 热图
- `tests/test_rank_displacement.py`：12 个 swap 相关 unit test（含 dual-tier strict/candidate 检验、perfect reversal FW p_fw、only_top_two_swap、determinism、score-floor gate、none-when-floor-unmet）
- 输出：
  - `results/interictal_propagation/rank_displacement/figures/cohort_displacement_heatmap.{png,pdf}` — 主图（带 swap markers）
  - `results/interictal_propagation/rank_displacement/figures/swap_cardinality_heatmap.{png,pdf}` — supplementary subject × k 热图

### 8.7 Channel-label 合同（关键安全合同）

如果有人想把 swap markers 标识的 endpoint cell 转写成 channel-level label
（例如做 SOZ 富集 / 后续模型 / paper-level 通道贴标），**必须遵守这套合同**：

1. **只 strict tier 才允许做 channel label**。candidate tier 完全禁止——FW
   evidence 没到 α=0.05，不能从 candidate subject 把通道贴成 endpoint。
2. **Label 名称只能是 `swap_endpoint_candidate`**，**不**得叫 SOZ / 致痫 /
   ictal-leading / "真实反转节点"。本 classifier 不做疾病侧 / 极性判读。
3. **Split-half 验证**：channel label 后续若进入富集统计或预测模型，**必须**
   把数据切成两半——一半选 (decision_k, 通道集)，另一半验证这些通道在两个
   template 上仍然形成 endpoint swap 几何。否则在同一批数据上既挑 label 又
   证明 label 是 circular。
4. **Split-half 失败的 subject 必须从 channel-label cohort 剔除**，不能"prefer
   higher T_obs"二次挑。
5. 任何对 strict tier 的 cohort claim **必须把 channel selection caveat（§5.1）
   重述在结果数字旁**——lagPat 高-HI gate 是上游已选择的子集，"通过 FW gate"
   ≠ "在全脑反向"。

### 8.8 不可以说

- ~~"swap_class=strict ⇒ 致痫 / SOZ-leading"~~ — 与 §5.2 同：本 classifier 不做疾病侧或 SOZ 极性判读
- ~~"decision_k = N ⇒ swap 涉及 N 节点"~~ — 见 §8.4 caveat；decision_k 是 argmax 位置 + smallest-k tie-break
- ~~"swap classifier 的 PR-2.5 recall / specificity = X%"~~ — PR-2.5 fwd/rev-reproduced 不是 ground truth；分歧（3 reproduced 在 candidate、2 non-candidate 在 strict）正是分歧本身的科学信息
- ~~"per-k null 通过即足够 evidence"~~ — v1 旧版本的 per-k OR 规则没做 family-wise 校正，已弃用；当前合同必须用 FW max-null
- ~~"swap_score = 0.5 等价于一半节点完整 swap"~~ — 见 §8.1 floor 注；0.5 仅是端点-集合 Jaccard 的平均
- ~~"candidate tier 与 strict tier 等价"~~ — candidate 是 exploratory；FW evidence 不到 α=0.05，channel label 与 cohort claim 都不允许从 candidate 拿
- ~~"主图上画了圈 ⇒ 这些通道是该 subject 的反转端点"~~ — 圈只表示"在 lagPat 选定通道集内、用 FW max-null 与 decision_k 切出的 endpoint 候选"；通道选择 caveat（§5.1）必须紧邻数字一起出现

### 8.9 channel-selection caveat 仍生效

§5.1 的所有 caveat 全文保留：F_norm 与 swap_score 都在 lagPat 既定通道集内
计算，decision_k 也只反映该子集内的最佳 swap cardinality。换通道选择策略（PR-1
ER-leading / Topic 3 SOZ audit / 更严 high-HI 阈值）后 decision_k、swap_class、
"strict ∪ candidate" cohort 比例都会变化。任何 cohort-level 表述必须带通道选择
caveat，**且 figure caption / 正文紧邻 decision_k 数字处必须重述这条 caveat**
（不只是放在远处的方法学段）。

## 7. 历史链接

- `docs/topic1_within_event_dynamics.md` §7 — Topic 1 主文档
- `docs/archive/topic1/pr6_template_endpoint_anchoring_plan_2026-04-25.md` — PR-6 plan（上游 swap_node 离散合同）
- `docs/archive/topic1/ping_pong_hypothesis_review_2026-04-28.md` — PR-8 candidate 来源
- `docs/archive/topic1/pr6_supplementary_rank_displacement_plan_2026-05-06.md` — 本 supplementary 的 plan
