# PR-6 Supplementary §9：Swap × Clinical SOZ Set-Relationship Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **状态：planned (2026-05-08)。** 本计划是 PR-6 supplementary rank_displacement 文档的 §9 扩展，**不**立独立 PR、**不**开新 cohort gate、**不**升级 main doc 结论。
>
> **上游合同**：PR-6 supplementary §8 dual-tier swap_class (`strict / candidate / none`，FW max-null α=0.05 / 0.20，决定 `decision_k`) + Yuquan/Epilepsiae clinical SOZ JSON。
>
> **下游归属**：写入 `docs/archive/topic1/pr6_supplementary_rank_displacement_results_2026-05-06.md` §9（`8.x` 之后），同步在 main doc `docs/topic1_within_event_dynamics.md` PR-6 小节加一行回链。

**Goal:** 在 lagPat universe 内对 strict / candidate swap_endpoint × clinical SOZ 做 set-relationship 描述，给"间期刻板时序是癫痫病理网络指示器"这条 Topic 1 论文第一部分主线落最后一块**空间证据**——而**不**做硬 cohort p-value（避开 advisor 抓到的 paired Δ n_middle 结构性退化），**不**引入 outcome × partition 检验（HFO rate 是 lagPat 选择回声、detrend_fraction 属于 Topic 1 Part 2 演化故事、ER-rank 未 cohort-stable）。

**Scientific boundary（写死）**：

- 是 PR-6 H1（hard-coded `k=3` swap_node × template-specific SOZ frac，paired Wilcoxon `p=0.19` NULL prior）的方法学改进 retest，但**主结论形态从"cohort p-value"改为"cohort typology distribution + enrichment_over_lagPat sign test"**。
- 是 paper1_framework_sba H1 三源（clinical / M1=HFO-onset rate / M2=ER-ratio）协同预言中的 **clinical 单源首发**。M1/M2 sensitivity 等 PR-T3-1 producer 重设计后单独补 §9.5。
- **circular caveat 写在每个 cohort 数字旁**：lagPat 通道由 high-HI / high-HFO-rate 阈值选出，universe 本身已对 SOZ 富集；本 §9 量化的是"在 high-HI 区内 swap_endpoint 是否进一步富集 SOZ"，不是全脑 swap-vs-SOZ 关系。
- **outcome × partition 检验（HFO rate / detrend_fraction / ER-rank / cluster participation × 4-cell partition）属于下一 PR Part 2 演化故事范围**，不在本 §9。

**Architecture:** 复用 `src/rank_displacement.py`（与 §8 swap_class 同模块）、`scripts/run_rank_displacement.py`（add per-pair `clinical_soz_set_relation` 字段）、`scripts/plot_rank_displacement.py`（add `--what clinical-soz-set-relation` 3-panel figure）；测试加 `tests/test_rank_displacement.py`；不新增 module。

**Tech Stack:** numpy, scipy.stats (binomtest)，bootstrap by hand（reproducible seed=0, 2000 reps）。matplotlib + 现有 `src/plot_style.py`。**禁止**新增 sklearn / pandas-stats 依赖。

---

## 0. 范围与禁区（pre-registered, 写死）

**做**：

1. 在 §8 swap_class 写入的 `swap_sweep` 旁，加一个新的 `clinical_soz_set_relation` 字段（每 pair 一份）
2. per-pair 计算 `precision`、`recall_within_lagPat`、`coverage`、`lagpat_baseline`、`enrichment_over_lagPat`、`typology`
3. cohort-level 在 strict ∩ informative 上跑 `enrichment_over_lagPat > 0` 的 binomial sign test + median enrichment + bootstrap 95% CI
4. candidate ∩ informative 跑同样 stat 作 sensitivity（**不**与 strict 合并入 primary）
5. 3-panel paper-level supplementary figure（precision×recall scatter / enrichment vs coverage / typology stacked bar）
6. 写 §9 进 supplementary results doc，加 main doc 一行回链

**不做**（违反则任务失败）：

1. **不**用 `endpoint_soz_frac − middle_soz_frac` paired Δ（advisor 2026-05-08 抓到 strict tier `n_middle ∈ {0,1}` 占多数，结构性退化；本 §9 完全弃用 paired Δ 框架）
2. **不**做"swap × SOZ × outcome 4-cell partition"检验（outcome 须等 Topic 1 Part 2 PR；本 §9 是纯 spatial set-relationship）
3. **不**用 HFO event rate 当 outcome（lagPat 选择回声，循环论证）
4. **不**用 PR-2 cluster participation 当 outcome（与 swap_class 同源，循环论证）
5. **不**升级 hypothesis tier。本 §9 是 descriptive primary + mechanism-sanity tier；**不**写"通过 §9 ⇒ swap 可替代 clinical SOZ"
6. **不**重写 main doc §7 PR-6 结论；只加一行回链到本 §9
7. **不**对 candidate tier 做 channel label（§8.7 channel-label 安全合同：candidate **禁止**做 channel label）
8. **不**预承诺 informative cohort 数。informative gate 是 `0 < |S| < |L| AND |E| < |L|`；最终 n_informative 由 Task 4 跑数后报告
9. **不**做 split-half 验证。本 §9 不依赖 swap label generalization（descriptive set-relationship within observed cohort），split-half 留给 Part 2 PR 用 swap label 做 outcome 检验时再做
10. **不**把 informative cohort sign test 通过包装成 "PR-6 H1 显著 / refined classifier 有效"。多源（M1/M2）未协同前不可声明；paper1_framework_sba H1 三源条件未满足

---

## 1. 数据合同（必须验证后才能跑）

**输入文件**：

| 文件 | 字段 | 用途 |
|---|---|---|
| `results/interictal_propagation/rank_displacement/per_subject/<dataset>_<subject>.json` | `pairs[*].swap_sweep.{decision_k, swap_class, T_obs, p_fw}`、`pairs[*].n_valid`、`pairs[*].rank_T_a_dense`（or 重算）、`channel_names_valid` | swap_endpoint 派生（top decision_k ∪ bottom decision_k channels in T_a） |
| `results/yuquan_soz_core_channels.json` | `<subject>: [ch_name, ...]` | Yuquan clinical SOZ |
| `results/epilepsiae_soz_core_channels.json` | `<subject>: [ch_name, ...]` | Epilepsiae clinical SOZ（focus_rel == "i"） |

**通道顺序 / valid_mask 合同**（**写死，复用 §8**）：

- 通道顺序由 PR-2 JSON `channel_names` 决定；rank_displacement 已在 §3 / §8 校验对齐
- valid_mask 来源同 §3：PR-6 `per_template[k].valid_mask`（23 subject）或 PR-2 `(template_rank ≠ −1)` sentinel fallback（4 subject `epilepsiae_1125, 384, 620, 916` + 1 yuquan_gaolan + 7 个 yuquan v14 backfill 等）
- `swap_endpoint` 通道 = `top decision_k` ∪ `bottom decision_k`（按 `rank_T_a_dense` 排序，与 §8 主图圈位一一对应）
- `swap_endpoint` 永远 ⊆ `valid_chs`（PR-6 `extract_endpoint_middle` 已保证）

**SOZ 通道映射合同**（**写死**）：

- Yuquan：JSON `<subject>: [ch_name, ...]` 直接给 channel name 列表，与 PR-2 `channel_names` 字面字符串 match。不一致 → `raise ValueError("yuquan SOZ channel name mismatch")`
- Epilepsiae：JSON 已是 SOZ-only（focus_rel=="i"）。同样 channel name 字面 match。不一致 → `raise ValueError`
- Bipolar pair name 形态（如 `LHa1-LHa2`）必须与 PR-2 `channel_names` 完全一致。lagPat 既已用同一 channel_names 跑下来，这条合同自动满足
- SOZ JSON 缺 subject → 该 pair 入 `exit_reason="no_clinical_soz"`，跳过 set_relation 计算（**不**进 cohort 任何分支）

**informative gate**（**写死，pre-registered**）：

```
informative := (0 < |S_in_lagpat| < |L|) AND (|E| < |L|)
```

- `|S_in_lagpat| = 0` → universe 无 SOZ → typology = `degenerate`
- `|S_in_lagpat| = |L|` → universe 全 SOZ → typology = `degenerate`
- `|E| = |L|` → endpoint 等于 universe → typology = `degenerate`（**注**：endpoint 几乎=universe 时 precision = lagpat_baseline，enrichment ≡ 0）
- 三者只要任一退化，`enrichment_over_lagPat` 仍计算并存档（值 ≈ 0），但 cohort sign test **排除**这些 pair；descriptive 图保留这些点

---

## 2. Per-pair JSON Schema（写死）

```json
"clinical_soz_set_relation": {
  "soz_source": "clinical",
  "n_E": int,
  "n_S": int,
  "n_L": int,
  "n_E_inter_S": int,
  "precision": float | null,
  "recall_within_lagPat": float | null,
  "coverage": float | null,
  "lagpat_baseline": float | null,
  "enrichment_over_lagPat": float | null,
  "typology": "E_subset_S" | "S_subset_E" | "partial" | "disjoint" | "degenerate",
  "informative": bool,
  "exit_reason": null | "no_clinical_soz"
}
```

**字段语义**：

- `n_E = |swap_endpoint|`，`n_S = |SOZ ∩ valid_chs|`，`n_L = |valid_chs|`
- `precision = n_E_inter_S / n_E`（`null` iff `n_E == 0`，本场景下 dk ≥ 2 所以不会发生）
- `recall_within_lagPat = n_E_inter_S / n_S`（`null` iff `n_S == 0`）
- `coverage = n_E / n_L`
- `lagpat_baseline = n_S / n_L`
- `enrichment_over_lagPat = precision − lagpat_baseline`（baseline-corrected enrichment；`null` iff `n_S == 0` or `n_L == 0`）
- `typology` 决策树（按顺序判断，命中即停）：
  1. `degenerate` if `n_S == 0 OR n_S == n_L OR n_E == n_L`
  2. `disjoint` if `n_E_inter_S == 0`
  3. `E_subset_S` if `n_E_inter_S == n_E AND n_E_inter_S < n_S`
  4. `S_subset_E` if `n_E_inter_S == n_S AND n_E_inter_S < n_E`
  5. `partial` else
- `informative` = NOT(`degenerate`)
- `exit_reason = "no_clinical_soz"` iff SOZ JSON 没这个 subject；其他字段全 null

---

## 3. Cohort 划分（pre-registered，与 §8 lock 一致）

| Tier | 来源 | §9 用途 |
|---|---|---|
| **Primary** | `swap_class == "strict"` ∩ `informative == true` | binomial sign test on `enrichment_over_lagPat > 0` + median + bootstrap CI |
| **Sensitivity** | `swap_class == "candidate"` ∩ `informative == true` | 同样 stat（**不**合并入 primary） |
| **Descriptive only** | degenerate (strict / candidate / none) + `none ∩ informative` | typology 分布表 + scatter 上画但**不**进 sign test |

`none ∩ informative` 也算入 typology 分布展示（exhaustive cohort coverage），但不算入 swap_class 假设的 sign test。

---

## 4. Cohort 统计（写死）

**主统计** (Primary cohort = strict ∩ informative)：

- **Binomial sign test**：H_0: `P(enrichment > 0) == 0.5`；H_alt: `> 0.5`；one-sided exact binomial（`scipy.stats.binomtest(k, n, p=0.5, alternative='greater')`）
- **Median enrichment** + **bootstrap 95% CI**（reproducible：seed=0, n_boot=2000, percentile method）
- **Effect size threshold（descriptive）**：`median_enrichment >= 0.10` 视为"实质性方向偏离 lagpat baseline"，`< 0.10` 视为"接近 baseline"

**Sensitivity** (Sensitivity cohort = candidate ∩ informative)：

- 跑同样 sign test + median + bootstrap CI
- 报告 strict / candidate 两 cohort 的 effect size 一致性（**不**做合并 stat、**不**做 strict-vs-candidate 对比检验）

**报告口径**：

- "strict ∩ informative cohort 上 enrichment_over_lagPat 方向 sign test p=X，median enrichment=Y (95% CI [a, b])"
- "candidate ∩ informative cohort sensitivity：sign test p=X', median enrichment=Y' (95% CI [a', b'])"
- "circular caveat：lagPat universe 已偏置 SOZ；本 §9 量化的是 high-HI 区内 swap 进一步富集，不是全脑关系"

**禁止**：

- ~~跨 strict/candidate 合并 cohort 跑单个 p-value~~
- ~~把 sign test p<0.05 包装成 "swap_endpoint 是 SOZ 子集 / 替代"~~
- ~~把 median enrichment 数字放进 main doc 当 cohort 主结论~~

---

## 5. Figure 规范（pre-registered，写死）

**输出**：`results/interictal_propagation/rank_displacement/figures/swap_clinical_soz_set_relation.{png,pdf}`

**3-panel layout**（横向并排，DPI_PUB=300）：

**Panel A — precision × recall scatter**：
- x = `recall_within_lagPat`，y = `precision`
- 每 informative subject 一个点（degenerate 不画）
- 标记：strict = 实心黑圆；candidate = 空心灰圆；none = 浅灰小三角
- Reference lines：`y = 1` (precision=1)、`x = 1` (recall=1)、`y = lagpat_baseline_subject` 用每点的 baseline 画淡色短线段（视觉上提醒 baseline 因 subject 不同）
- 每 strict subject 标 subject name（短码）

**Panel B — enrichment_over_lagPat × coverage scatter**：
- x = `coverage`，y = `enrichment_over_lagPat`
- 同 marker 规则
- Reference lines：`y = 0` 横虚线，`y_max(coverage) = 1 − coverage` 上界曲线（淡灰）—— 提醒 coverage→1 时 enrichment 上界压缩到 0（结构性 saturation 边界）
- 每 strict subject 标 subject name

**Panel C — typology stacked bar**：
- x = tier（strict / candidate / none）
- y = subject count
- 颜色按 typology：`E_subset_S` / `S_subset_E` / `partial` / `disjoint` / `degenerate`
- 顶部数字标 informative count（degenerate 不算）

**Caption（写死）**：

> **(A)** Precision × recall_within_lagPat scatter for swap_endpoint vs clinical SOZ within lagPat universe (strict swap_class = filled black, candidate = open grey, none = light triangle, degenerate subjects excluded). **(B)** Enrichment_over_lagPat (precision − lagpat_baseline) vs coverage (|E|/|L|); dashed line at enrichment=0; grey envelope = 1 − coverage upper bound (structural ceiling at coverage→1). **(C)** Typology distribution by swap_class tier. **All numbers within lagPat (high-HI) universe only**; lagPat selection is biased toward SOZ a priori, so "swap enrichment" is measured relative to lagpat_baseline, not full electrode set.

**禁止**：

- ~~P-value annotation 放图上~~（cohort sign test 数字写 caption / archive doc 文字，不写 panel 内）
- ~~把 precision = 1 / recall = 1 reference line 标"perfect"~~（lagPat universe 的 perfect 不是全脑 perfect，避免误导）

---

## 6. TDD 合同（pre-registered，写死）

新增到 `tests/test_rank_displacement.py`，复用 §8 的 fixture 风格：

```python
def test_set_relation_intermediate_universe():
    """n_S ∈ (0, n_L), n_E < n_L → all fields well-defined, typology ∈ {partial, E_subset_S, S_subset_E, disjoint}"""

def test_set_relation_saturated_universe():
    """n_S == n_L → typology == 'degenerate', informative == False, enrichment ≈ 0"""

def test_set_relation_empty_soz_universe():
    """n_S == 0 → typology == 'degenerate', informative == False, recall_within_lagPat is None"""

def test_set_relation_full_coverage():
    """n_E == n_L → typology == 'degenerate', informative == False; precision == lagpat_baseline → enrichment ≡ 0"""

def test_set_relation_E_subset_S():
    """E ⊂ S strictly → typology == 'E_subset_S', precision == 1, recall < 1, enrichment > 0"""

def test_set_relation_S_subset_E():
    """S ⊂ E strictly → typology == 'S_subset_E', recall == 1, precision < 1, enrichment depends"""

def test_set_relation_disjoint():
    """E ∩ S == ∅ → typology == 'disjoint', precision == recall == 0, enrichment == -lagpat_baseline"""

def test_set_relation_partial():
    """proper partial overlap → typology == 'partial'"""

def test_enrichment_zero_when_E_is_uniform_sample_of_L():
    """E uniformly samples L (E∩S/E ≈ S/L) → enrichment ≈ 0 within rounding"""

def test_cohort_sign_test_excludes_degenerate():
    """mixed cohort with 3 degenerate + 5 informative → sign test n=5, degenerate not counted"""

def test_cohort_bootstrap_reproducible():
    """seed=0, n_boot=2000 → CI deterministic across runs"""
```

测试不依赖真实 SOZ JSON / cohort 数据：纯合成 channel set 输入。

---

## 7. 实施步骤（checkbox）

### Phase A — TDD core helper

- [ ] **A.1** 写测试 1–11 (`tests/test_rank_displacement.py`，复用 §8 fixture 风格；测试函数 + 合成 fixture，**不**碰真实 JSON)
- [ ] **A.2** 跑测试，确认 11 条全 fail（`compute_clinical_soz_set_relation` 还没实现）
- [ ] **A.3** 实现 `compute_clinical_soz_set_relation(valid_chs: list[str], endpoint_chs: list[str], soz_chs: list[str]) -> dict`（在 `src/rank_displacement.py` §8 swap helpers 之后）。decision tree 严格按 §2 schema typology 顺序。
- [ ] **A.4** 跑测试，确认 11 条全 pass
- [ ] **A.5** Commit：`feat(pr6 §9): compute_clinical_soz_set_relation + TDD`

### Phase B — wire into runner

- [ ] **B.1** 在 `scripts/run_rank_displacement.py` 加 SOZ JSON 加载 helper（`_load_clinical_soz_map(dataset, results_root)`，返回 `{subject: set(ch_names)}`）
- [ ] **B.2** per-pair 计算 endpoint_chs（top decision_k ∪ bottom decision_k by `rank_T_a_dense`），写 `clinical_soz_set_relation`；缺 SOZ → `exit_reason="no_clinical_soz"`
- [ ] **B.3** cohort aggregator 加 `clinical_soz_set_relation_summary` 子结构：strict / candidate / none × informative / degenerate × cell counts；strict ∩ informative + candidate ∩ informative 上跑 sign test + median + bootstrap CI（seed=0, n_boot=2000）
- [ ] **B.4** 跑全 cohort（40 subject），确认 `cohort_summary.json` 多了新字段；对照 §3 exit_reason 数核对（如果 SOZ JSON 缺某 subject，应该 `no_clinical_soz`）
- [ ] **B.5** Commit：`feat(pr6 §9): runner + cohort aggregator for clinical SOZ set-relation`

### Phase C — figure

- [ ] **C.1** 在 `scripts/plot_rank_displacement.py` 加 `plot_clinical_soz_set_relation(cohort_summary_path, output_dir)`（3 panel layout per §5）
- [ ] **C.2** CLI 加 `--what clinical-soz-set-relation` 选项；调用上面 plot fn
- [ ] **C.3** 跑 plot，确认 PNG + PDF 落在 `results/interictal_propagation/rank_displacement/figures/`
- [ ] **C.4** 视觉检查：strict 标黑、candidate 空心、reference line 不混淆，panel 内 NO p-value
- [ ] **C.5** Commit：`feat(pr6 §9): 3-panel figure for clinical SOZ set-relation`

### Phase D — write up

- [ ] **D.1** 在 `docs/archive/topic1/pr6_supplementary_rank_displacement_results_2026-05-06.md` §8 之后加 §9，按下面口径：
  - §9.0 framing（与 §8 衔接、PR-6 H1 prior NULL、circular caveat）
  - §9.1 schema + cohort 划分
  - §9.2 cohort 主结果（strict ∩ informative 数字 + median + CI）
  - §9.3 candidate sensitivity
  - §9.4 typology 分布 + per-subject 数字表
  - §9.5 不可以说 list（与 §8.7 / §8.8 接续）
- [ ] **D.2** 更新 `results/interictal_propagation/rank_displacement/figures/README.md` 加新 figure 中文说明（按 AGENTS.md "results 目录规范" 要求）
- [ ] **D.3** 在 `docs/topic1_within_event_dynamics.md` PR-6 supplementary 段落末尾加一行回链：`Topic 1 论文第一部分 spatial 收束证据见 §9（…）`
- [ ] **D.4** Commit：`docs(pr6 §9): write supplementary §9 + figures README + main doc backlink`

---

## 8. 不在范围 / 后续

- **multi-source SOZ 协同 (paper1_framework_sba H1)**：clinical + M1 + M2 三源同向才支持 strong claim。M1 = HFO-onset rate 已在 PR-T3-1 v1.1 obsolete archive；M2 = ER-rank 当前 cohort_summary 显示 16 subject 中 broad_ER 6/16 concordant，cohort 未稳定。等 PR-T3-1 producer 重设计后单独以 §9.5 sensitivity 形式补，不 block 本 §9
- **outcome × partition 检验（HFO rate / detrend_fraction / cluster participation × 4-cell partition）**：detrend_fraction n=31 (yuquan 11 + epilepsiae 20) 是 Topic 1 Part 2 演化故事 PR 的主力 outcome，不在本 §9 空间收束范围
- **swap label 的 split-half generalization**：本 §9 不依赖 swap label generalize 到 held-out events（descriptive within-observed）；§8.7 channel-label 合同要求的 split-half 留给 Part 2 PR 把 swap label 当 input 跑 outcome 检验时再做
- **swap source / sink 拆分 enrichment**：Δ_source / Δ_sink 拆分需要每 cell n_endpoint = decision_k ≥ 2，多数 strict subject 满足，但本 §9 主图保持"endpoint = top dk ∪ bottom dk 一起"以与 §8 主图圈位一致；source / sink 拆分作为 Part 2 PR 的 mechanism descriptive 候选
- **PR-6 H1 NULL prior 的方法学回写**：§9 引用 PR-6 H1 NULL 作对比，但**不**重写 PR-6 archive doc。PR-6 H1 的最终判读仍是 hard-coded `k=3` 下 Wilcoxon `p=0.19` NULL；本 §9 的 sign test 结果**不**作为推翻 PR-6 H1 NULL 的 evidence，只是同问题在更精细 swap label + 不同统计形态下的描述性补充
- **multi-source SOZ source 含义辨析（i / l / e Epilepsiae focus_rel）**：本 §9 用 `epilepsiae_soz_core_channels.json`（focus_rel=="i" only）。i ∪ l 三源协同分析等 paper1_framework H1 完整 sensitivity round，不进 §9

---

## 9. Cross-PR 合同 lookup（与 §8 同）

- §8 dual-tier swap_class：写在 `pr6_supplementary_rank_displacement_results_2026-05-06.md` §8（v3 final, 2026-05-07）
- §8.7 channel-label 安全合同：candidate tier **禁止**做 channel label；strict tier label 后续如进入下游统计 / 模型必须 split-half。本 §9 不进入这一类（descriptive set-relationship 不算"下游统计 / 模型"）
- PR-6 H1 NULL prior：写在 `docs/archive/topic1/pr6_template_anchoring/` 下；本 §9 引用而**不**重写
- paper1_framework_sba H1 三源协同预言：写在 `docs/paper1_framework_sba.md`；本 §9 是该预言的 clinical 单源首发

---

**Plan version**：v1.0 (2026-05-08)
**Author**：Topic 1 §9 supplementary
**Approval**：implementation-only plan; cohort gate 在 §0 写死；§4 sign test + median + bootstrap CI pre-registered；TDD 合同 §6 pre-registered（11 测试）。
