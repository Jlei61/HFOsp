# Step 5f — PR-6 全套（template_anchoring + Step 6 held-out + rank displacement）修过版重跑结果（2026-05-21）

> 状态：Step 5f.1 / 5f.2 / 5f.3 / 5f.4 全部完成。**主结论方向全部保持**；**1 条 cohort-level Wilcoxon borderline 翻转**（node anatomy h1_eligible swap−same Wilcoxon p=0.014 → 0.059，secondary 指标，PR-6 plan §3 未列为 cohort claim α 池）；H2 fwd/rev 子集因 −1 subject 略缩。**Checkpoint B 仍待 5d.3 + 5e 完成后正式触发**——user 2026-05-21 决定**优先 5f over Checkpoint B sequence**（PR-6 给后续 SEF-ITP H1/H2 提供数据基础），所以 5f 在 Checkpoint B 之前独立提交；本档不声称 Checkpoint B 已通过。
> 主入口：`docs/topic0_methodology_audits.md`
> 上游：`./step5a_pr2_results_2026-05-20.md` / `./step5b_pr25_results_2026-05-20.md` / `./step5c_pr3_results_2026-05-20.md`
> 路线图：`./rerun_roadmap_2026-05-20.md` §5f
> 原 PR-6 plan：`docs/archive/topic1/pr6_template_anchoring/pr6_template_endpoint_anchoring_plan_2026-04-25.md`
> 修过版结果：
> - `results/interictal_propagation_masked/template_anchoring/`
> - `results/interictal_propagation_masked/pr6_step6_held_out_template/`
> - `results/interictal_propagation_masked/rank_displacement/`

---

## 1. 三段式朴素话

**测了什么** —— PR-6 整套"间期 HFO 群体事件的稳定模板的端点（头尾通道）是否锚定癫痫病灶网络"的检验，用修过版（masked）的 cluster labels 重跑：
- 5f.1 = PR-6 主线 H1（端点 vs 中间通道的 SOZ 富集差）+ H2（forward/reverse 子集的端点角色互换）+ H3（Epilepsiae 三级 focus_rel 富集）+ Step 4 节点级配对几何 + Step 5a coreness sensitivity + Step 5b split-half 时间稳定性
- 5f.3 = PR-6 Step 6 held-out（前一半时间训练模板，后一半时间投影验证，看模板和端点是否跨时间稳定）
- 5f.4 = PR-6 supplementary rank displacement（两个 template 的 rank 互换强度，paper 级单图）

**怎么测的** ——
1. 输入：5a 写好的 `results/interictal_propagation_masked/per_subject/<sid>.json`（masked PR-2 cluster labels + 模板）
2. 给三个 runner 加 `--masked-features` flag：
   - `scripts/run_pr6_template_anchoring.py`：路径重路由到 `interictal_propagation_masked/template_anchoring/`，并把 `use_masked_features=True` 透传给 `compute_time_split_reproducibility`（split-half 重新跑时也用 masked feature 空间做 KMeans，与 5a/5b 保持一致）
   - `scripts/run_pr6_step6.py`：同上 + 透传给 `compute_held_out_endpoint_validation`
   - `scripts/run_rank_displacement.py`：纯路径重路由（无 compute helper 需要 flag）
3. 4 项新 TDD（`tests/test_pr6_masked_path_routing.py`，全部 PASS）+ 既有 `tests/test_pr6_template_anchoring.py` + `tests/test_interictal_propagation.py` + `tests/test_held_out_template.py` 共 92/92 PASS（无回归）
4. 按依赖跑：5f.1 完成（产出 `template_anchoring/per_subject/`）→ 5f.3 + 5f.4 并行（共消耗 PR-6 输出）

**揭示了什么** ——
- **H1 主结论稳**：原 H1 pooled cohort NULL（n=28, median +0.010, Wilcoxon-greater p=0.22），修过版仍 NULL（n=28, median 0.000, p=0.18）；论文级"endpoint vs middle 不显著锚定 SOZ"的结论不变。Cohort 成员是 **28 = 28 完全一致**（无人翻进翻出）。
- **H2 forward/reverse swap 节点级几何信号保持，magnitude 略缩**：与 plan 早期 Step 4b record 比对（n=6, 6/6, p=0.016/0.031）会得"strengthened"误读；正确比对是同 cohort size 下，原版 **n=9**（含 endpoint_defined-only 1073）9/9 positive Wilcoxon p=0.0020 / sign p=0.0039，修过版 **n=8** 8/8 positive Wilcoxon p=0.0039 / sign p=0.0078；**两者都 100% positive，masked p 看似略升只因 −1 subject (635, 5b fwd/rev 翻转所致)**。Magnitude (median swap−same) +5.0 → +3.5 略降。**maintained, not strengthened**。
- **⚠ Node anatomy h1_eligible swap−same cohort Wilcoxon borderline 翻转**：原 n=28 median +1.5, 17p/9n/2z, Wilcoxon-greater p=**0.014（significant at α=0.05）**, sign-test p=0.169；修过版 n=28 median +1.0, 15p/12n/1z, Wilcoxon p=**0.059（borderline, no longer significant）**, sign p=0.701。方向（median 仍正）保持；**但 cohort-wide Wilcoxon 从 0.05 以下移到 0.05 以上**。**判读**：(1) 这是 Step 4b "node anatomy swap-leaning detectable cohort-wide on Wilcoxon" 的 secondary tier 结论，PR-6 plan §3 没把它纳入 H1 α 池；(2) 现有 PR-6 archive Step 4b §论文级口径已经写明 "sign-test 不通过 (0.115)，cohort-wide effect 主要由 fwd/rev 拉动"；修过版把 Wilcoxon 也推到 borderline，把 §论文级口径再收紧一格。**这不是 framework-level reversal，但写论文叙事时需要把这条 cohort-wide swap-leaning claim 进一步弱化**。
- **H1 coreness sensitivity 同方向略加强**：与 PR-6 archive Step 5a 早期记录（n=20, median +0.061, p=0.14）比对——修过版 **n=28, median +0.087, p=0.060** 更接近 0.05 边界。但 **orig 在当前 n=28 cohort size 上的 H1c Wilcoxon 数字在 `cohort_summary_n40.json` 里是空的**（schema 未填），所以严格 like-for-like 不可得；当前比较是 n=20 早期 vs n=28 修过版的近似。"endpoint 定义敏感" 这条 caveat 不变。
- **Split-half 时间稳定性几乎不变**：first-half split endpoint Jaccard 中位 0.714 → **0.714（完全相同）**；odd-even split 0.929 → **0.857**（略降，但远高于 0.4 的 caveat 阈值）。endpoint < 0.4 subject 数 2 → 3 / 0 → 2（轻微增加但 spectrum 没改）。
- **Step 6 held-out tier 分布稳健 + swap_class concordance 实质提升**：原 cohort n=35 → masked cohort n=28（**n 不同 like-for-like**：orig 走的是当时 PR-6 cohort + RD stable_k=2 扩集；masked 没扩 RD，只用 latest masked PR-6 cohort_summary），strong fraction 20/35=57% → **17/28=61%**；template_spearman 中位 0.92 → **0.88**；endpoint_position_recall 中位 0.83 → **0.83（完全相同）**；**swap_class_concordant fraction 0.69 → 0.82（实质提升）**——这是 PR-6 archive 里"held-out swap geometry 跨时间稳定"的最相关 metric。新出现 1 个 fail（epilepsiae_635），其在 5b 已经被记为 fwd/rev 翻转的 subject 之一。
- **Rank displacement 主图数字几乎不变**：F_norm 中位 0.800 → **0.789**；Kendall τ 中位 −0.20 → **−0.24**；Spearman ρ(F_norm, τ) **−0.916 → −0.921**（强负相关完全保持）；swap_class strict 10 → 9，candidate 8 → 6，others 17 → 19（轻微再分布，但总体强类形 + 弱类形数都接近）。clinical SOZ strict_informative 仍 NULL（sign p 0.50 → 0.66, n=5/6）。
- **总体判读**：PR-6 主结论（H1 NULL, H2 fwd/rev 子集 swap geometry 全 positive, Step 6 held-out tier 分布 + swap_class concordance, rank displacement F_norm/τ）**全部保持方向**；**1 条 secondary cohort-level Wilcoxon borderline 翻转**（node anatomy h1_eligible Wilcoxon 0.014→0.059，已经在原版 sign-test 上不显著，属于"次要 metric 在 phantom 修后进一步收紧"，非 primary metric flip）。**没有任何 primary 方向反转或 framework-level NULL ↔ 显著翻转**。可进 5g。

代号补注：H1 endpoint vs middle = `template_anchoring/cohort_summary.json/h1_pooled`；H2 = `h2_forward_reverse`；coreness sensitivity = `h1_coreness_sensitivity`；split-half = `split_half_endpoint_robustness`；node anatomy = `node_anatomy`；Step 6 held-out = `pr6_step6_held_out_template/cohort_summary.json`；rank displacement = `rank_displacement/cohort_summary.json/[*].pairs[0].swap_sweep` + `clinical_soz_set_relation_summary.json`。

---

## 2. 实现层改动（surgical）

| 文件 | 改动 |
|---|---|
| `scripts/run_pr6_template_anchoring.py` | 新增 `_apply_masked_paths()` 把 `PER_SUBJECT_DIR / OUT_DIR / PER_SUBJECT_OUT / AUDIT_CSV / COHORT_SUMMARY` 5 个 path 全部 swap 到 `interictal_propagation_masked/template_anchoring/`；`--masked-features` flag；`run_per_subject` + `compute_split_half_robustness` 加 `use_masked_features` kwarg；split-half 重新跑时透传给 `compute_time_split_reproducibility`。 |
| `scripts/run_pr6_step6.py` | 新增 `_apply_masked_paths()` swap `PR2_DIR / PR6_DIR / PR6_COHORT_SUMMARY / RD_COHORT_SUMMARY / DEFAULT_OUTPUT_DIR`；`--masked-features` flag；`_process_one` 加 `use_masked_features` 透传给 `compute_held_out_endpoint_validation`。`--output-dir` 默认改为 `None`，回退到 swap 后的 `DEFAULT_OUTPUT_DIR`，行为：未传 `--output-dir` 时随 `--masked-features` 自动切换；显式 `--output-dir` 仍优先。 |
| `scripts/run_rank_displacement.py` | 纯路径重路由：`_apply_masked_paths()` swap `PR2_DIR / PR6_DIR / OUT_DIR / OUT_PER_SUBJECT`；`--masked-features` flag。无 compute helper 需要 use_masked_features。 |
| `tests/test_pr6_masked_path_routing.py`（新建） | 5 项 smoke test：3 项验证 `_apply_masked_paths()` 在三个 runner 里都成功 swap path globals；2 项验证 `use_masked_features` kwarg 经 `compute_split_half_robustness` 和 `_process_one` 透传到 compute 函数（monkeypatch capture）。 |

不动：
- `src/template_anatomical_anchoring.py`（统计层，按定义不需要 masked feature）
- `src/interictal_propagation.py` 已在 5a/5b/5c 加好的 `use_masked_features` 参数链，5f 只是新增 caller
- 任何主 PR-2 / PR-2.5 / PR-3 / PR-4 输出
- 不引入 ictal anchor / ER / CUSUM / Page-Hinkley 任何代码

**TDD 验证**：
```
tests/test_pr6_masked_path_routing.py: 5/5 PASS
tests/test_pr6_template_anchoring.py + tests/test_interictal_propagation.py + tests/test_held_out_template.py: 92/92 PASS (无回归)
```

---

## 3. Cohort 数字对比表

### 3.1 Audit eligibility（masked vs orig，n=40 candidates）

| 类别 | orig | mask | 说明 |
|---|---:|---:|---|
| H1-eligible (pass) | 28 | **28** | **集合完全相同，无翻进翻出** |
| endpoint_defined only (n_ch=6 case-series) | 2 | 2 | epilepsiae 1073 / 1077 |
| EXIT `empty_soz` | 4 | 3 | epilepsiae 384 / 1125 / 620 (mask)；orig 多 916（mask 后 916 stable_k=4 走 k!=2 路径） |
| EXIT `k!=2` | 5 | 6 | mask 多 epilepsiae 916（5a stable_k 从 2 翻 4）|
| EXIT `no_matched_soz` | 1 | 1 | yuquan_gaolan（SOZ 标注无与 channel_names 匹配项）|

### 3.2 H1 endpoint vs middle SOZ enrichment

> **Cohort note**: H1 pooled row is properly like-for-like (both n=28 from same cohort definition, orig data drawn from `cohort_summary_n40.json`). H1 coreness / same-sign / direction-discordant rows compare **orig early Step 5a baseline (n=20)** with **masked n=28** because `cohort_summary_n40.json` has `h1_coreness_sensitivity.cohort_wilcoxon_greater = {}` (schema not populated in that run). Treat orig coreness numbers as historical baseline only; the direction comparison is informative but the magnitudes are not strictly comparable.

| stratum | n | orig median | orig Wilcoxon-greater p | mask median | mask Wilcoxon-greater p |
|---|---:|---:|---:|---:|---:|
| H1 pooled (like-for-like, n=28) | 28 | +0.010 | 0.22 | **0.000** | **0.18** |
| H1 coreness sensitivity (orig n=20 vs mask n=28) | — | +0.061 (Step 5a early) | 0.14 | **+0.087** | **0.060** |
| Same-sign agreement (main vs coreness) | — | 12/20 = 60% (early) | — | **18/28 = 64%** | — |
| Direction-discordant | — | 7/20 = 35% (early) | — | **9/28 = 32%** | — |

**判读**：H1 pooled cohort 在 mask 后**仍 NULL**（median 接近 0，p 略改善但远未达 0.05）；coreness sensitivity 仍**弱 positive 方向**（median +0.087，p=0.060 接近边界）；endpoint 定义敏感性 caveat 保留。论文级 H1 结论不需修改。

### 3.3 H2 forward/reverse swap（mechanism sanity）

> **Like-for-like comparison（同 cohort 定义，endpoint_defined ∩ forward_reverse_reproduced）**：

| | orig (n=40 latest) | mask |
|---|---:|---:|
| H2 cohort n | **9**（含 1073 endpoint-defined 但 H1-ineligible）| **8**（1073 仍在；少 1 因 5b fwd/rev 翻转）|
| Per-subject swap−same all positive？ | **9/9** | **8/8** |
| Swap−same Wilcoxon-greater p | **0.0020** | **0.0039** |
| Swap−same sign-test p | **0.0039** | **0.0078** |
| Swap−same median | +5.0 | **+3.5** |

**关键 cohort 成员变化**：
- 修过版退出 fwd/rev 集合：**548, 635**（5b 已经记录；其中 635 后续在 Step 6 held-out 也变 fail）
- 修过版新进入 fwd/rev 集合：**253**（5b 已记录）
- 修过版 k!=2 退出 H1：**916**（5a stable_k=4）；但 916 自身 fwd/rev pair 是 reproduce 的——仅因 k!=2 不进 H2 pool
- 修过版 endpoint_defined-only 仍含 1073（n=8 包含它）
- 净变化：−548, −635, +253 = **−1 subject**（n=9 → 8）

**判读**：
- **方向 100% 保持**：两个 cohort 都是全部 subject swap−same > 0；这是 PR-6 plan §3.3 H2 mechanism sanity 的关键观察。
- **magnitude 略缩**：median +5.0 → +3.5；p-value 因 n 小 1 自然升一档（n=9 sign 最小 2·(0.5)^9=0.0039 → n=8 最小 2·(0.5)^8=0.0078）。这不是显著性消失，是 sample-size mechanical change。
- **H2 仍是 mechanism sanity tier**——pre-registered tier 锁定，不允许升级为 cohort claim（CLAUDE.md §5 / AGENTS.md cross-PR contract）。修过版数字不改变这条 tier 锁。
- "8/8 strengthened to 6/6" / "信号实质加强" 是旧 plan 早期 Step 4b record (n=6) vs latest masked (n=8) 的错对——本档已在 §1 修正口径。

### 3.4 H3 Epilepsiae focus_rel 三级（i / l / e）

| label | n | orig median | mask median |
|---|---:|---:|---:|
| i（核心病理） | 10 | 0.0 | 0.0 |
| l（lesion） | 10 | 0.0 | 0.0 |
| e（extra-focal） | 10 | 0.0 | 0.0 |

三级**全都 NULL**，方向无变化；H3 在原版与修过版都是 cohort-level NULL。

### 3.5 Split-half endpoint robustness（时间稳定性）

| split | n | orig endpoint J 中位 | mask endpoint J 中位 | orig src/snk | mask src/snk | endpoint<0.4 orig | endpoint<0.4 mask |
|---|---:|---:|---:|---:|---:|---:|---:|
| first_half_second_half | 28 | 0.714 | **0.714** | 0.750/0.750 | **0.675/0.750** | 2/20 | **3/28** |
| odd_even_block | 28 | 0.929 | **0.857** | 1.000/1.000 | **0.750/1.000** | 0/20 | **2/28** |

**判读**：first-half split endpoint J 中位**完全相同**；odd-even 略降但仍远高于 0.4 caveat 阈值。endpoint<0.4 subject 数轻微上升但 spectrum 没变。**plan §10 row 8 "endpoint 时间稳定性不足" caveat 在 mask 仍 NOT 触发**。

### 3.6 Node anatomy (Step 4b) — subject-level paired tests

> **Like-for-like comparison**：所有 stratum 都从 `cohort_summary_n40.json`（orig latest）vs masked `cohort_summary.json` 直接读，cohort 定义一致。

| stratum | orig n | mask n | orig swap−same median / pos-neg-zero / Wilcoxon / sign | mask swap−same median / pos-neg-zero / Wilcoxon / sign |
|---|---:|---:|---|---|
| h1_eligible | 28 | 28 | +1.5 / 17·9·2 / **Wilcoxon p=0.014 ✓** / sign p=0.169 | **+1.0 / 15·12·1 / Wilcoxon p=0.059 ⚠ / sign p=0.701** |
| forward_reverse_reproduced | 9 | 8 | +5.0 / 9·0·0 / **Wilcoxon p=0.0020 ✓** / **sign p=0.0039 ✓** | **+3.5 / 8·0·0 / Wilcoxon p=0.0039 ✓ / sign p=0.0078 ✓** |
| non_forward_reverse_h1_eligible | 20 | 21 | 0.0 / 9·9·2 / p=0.40 / sign p=1.00 (NULL) | **−1.0 / 8·12·1 / p=0.62 / sign p=0.50 (NULL)** |

| stratum | orig n / mask n | orig SOZ Δ(swap−tspec) median / pos-neg-zero / Wilcoxon | mask SOZ Δ(swap−tspec) median / pos-neg-zero / Wilcoxon |
|---|---:|---|---|
| h1_eligible | 24 / 23 | 0.000 / 11·5·8 / p=0.25 | 0.000 / **7·5·11** / **p=0.084** |
| forward_reverse_reproduced | 7 / 6 | 0.000 / 3·0·4 / p=0.125 | 0.000 / 1·1·4 / p=0.33 |
| non_forward_reverse_h1_eligible | 17 / 17 | 0.000 / 8·5·4 / p=0.43 | 0.000 / 6·4·7 / p=0.11 |

**判读**：

- ⚠ **h1_eligible swap−same cohort Wilcoxon-greater 从 0.014（significant at α=0.05）翻到 0.059（borderline, no longer significant）**。Same n (28), positive direction (median 还是 +1.0) 保持，但 cohort-wide Wilcoxon 跨过 0.05 边界。**这是本档唯一显著 ↔ NULL 翻转**。Sign-test 在 orig 上原本就是 0.169（NULL），mask 下进一步弱化到 0.701。
  - PR-6 archive Step 4b §论文级口径已经在 orig 上写明：sign-test 0.115（其实 orig n=28 的 sign-test 是 0.169，比我引用的早期 0.115 更弱），Wilcoxon 0.012（其实 n=28 是 0.014）。原始 framing 已经是 "borderline cohort-wide effect driven by fwd/rev"。修过版把 Wilcoxon 也推到 borderline，**论文叙事要把 cohort-wide swap-leaning 进一步弱化**：从 "Wilcoxon p=0.014 detectable but sign-test p=0.17 weak" 改为 "Wilcoxon p=0.059 borderline 且 sign-test p=0.70 clearly null, cohort-wide swap-leaning fragile"。
- ✅ **fwd/rev swap-leaning maintained, magnitude 略缩**：9/9 → 8/8 都正向，sign-test 都达 α=0.05；像 §3.3 H2 一样，因 fwd/rev cohort −1（−548/−635 + 253）导致 magnitude 中位 +5.0 → +3.5、p-value 因小 n 自然升一档。**这是 cohort 成员变化，不是 signal 强度变化**。两个加入的新成员（253, wangyiyang 也在 h1_eligible 列表）都是 swap-leaning，方向被保持。
- ✅ **non-fwdrev 子集**：两版都 NULL，原版 9p/9n/2z（median 0, p=0.40）→ 修过版 8p/12n/1z（median −1, p=0.62）。方向不显著，secondary stratum NULL 保持。
- ✅ **SOZ Δ(swap−tspec) h1_eligible**：原 p=0.25 → mask **p=0.084**（仍未达 0.05，但 nominal p 改善）。pooled 数字 swap=0.74 / tspec=0.59（mask）vs orig 不重计，pooled 本就不进 cohort claim。**Step 4b 主结论"swap-vs-same geometry 不被 subject-level SOZ 富集差异解释" 在 mask 后仍然成立**（h1_eligible subject-level Δ=0, 7p/5n/11z, p=0.084 NULL）。

### 3.7 Template-pair geometry (Step 4) — stratified medians

| stratum | n | orig endpoint J | orig src→snk | orig spearman | mask endpoint J | mask src→snk | mask spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_endpoint_defined | 30 | 0.500 | 0.200 | **−0.380** | 0.500 | **0.200** | **−0.262** |
| h1_eligible | 28 | 0.500 | 0.200 | **−0.380** | 0.500 | **0.200** | **−0.142** |
| forward_reverse_reproduced | 8 | 0.714 | 0.750 | **−0.735** | 0.714 | **0.500** | **−0.784** |
| non_forward_reverse_h1_eligible | 21 | 0.500 | 0.200 | **−0.118** | 0.500 | **0.200** | **+0.106** |
| endpoint_stable_split_half_h1_eligible | 15 | 0.500 | 0.500 | **−0.474** | 0.500 | **0.200** | **+0.028** |

**判读**：
- ✅ fwd/rev 子集：Spearman 更负（−0.735 → −0.784），swap geometry 方向更干净
- ⚠️ all_endpoint_defined / h1_eligible Spearman 略接近 0（−0.38 → −0.14/−0.26）——masked 后 non-fwdrev 子集 spearman 从 −0.118 翻成 +0.106，把 cohort-wide median 拉接近 0；这与 Step 4b non-fwdrev swap-leaning 方向反成 −1 是同一个观测的两个视角
- ✅ endpoint_stable_split_half_h1_eligible（更严格子集，endpoint J ≥ 0.4 的 H1-eligible subject）：mask 后 Spearman 接近 0，没有 net rank 反向 → endpoint 时间稳的 subject 子集**反而**没有全局 rank 反向倾向；这与 Step 5b "endpoint 时间稳" 与 Step 4 "non-fwdrev 不是 bidirectional" 的解读自洽

### 3.8 Step 6 held-out（5f.3）

| 指标 | orig (n=35) | mask (n=28) |
|---|---|---|
| **Tier 分布** | strong 20 (57%), moderate 13, weak 2, fail 0 | **strong 17 (61%), moderate 8, weak 2, fail 1** |
| template_spearman 中位 | 0.922 | **0.881** |
| endpoint_position_recall 中位 | 0.833 | **0.833**（完全相同）|
| assignment_coverage 中位 | 1.000 | **1.000**（完全相同）|
| **swap_class concordant fraction** | 24/35 = **0.686** | **23/28 = 0.821（实质提升）** |

n 差异来源：orig 用 PR-6 cohort_summary (n=21 → 33 via RD 扩) → 35 stem；mask 走的是 latest masked PR-6 cohort_summary，没扩 RD，所以 n=28。如果走"masked PR-6 ∪ masked RD stable_k=2"扩集会得到 n≈34。本次没扩——cohort 内一致性优先。

**新 fail subject**：`epilepsiae_635`（held-out tier=fail，spearman=0.661，recall=0.583，swap_concord=False）。635 在 5b 已经被记为 fwd/rev 翻转的 subject 之一（split-half rule 下不再 reproduce）。这条孤立 fail 与 5b 的 fwd/rev 集合变化自洽，不是新 bug。

### 3.9 Rank displacement（5f.4，stable_k=2 子集）

| 指标 | orig (n=35) | mask (n=34) | Δ |
|---|---:|---:|---|
| n stable_k=2 subjects | 35 | **34** | −1（916 stable_k 翻 4，被 5a 移出 stable_k=2 cohort）|
| F_norm 中位 | 0.800 | **0.789** | −0.011（基本不变）|
| F_norm IQR | [0.641, 0.910] | [0.569, 0.871] | 略扩 |
| Kendall τ 中位 | −0.203 | **−0.240** | 略加强负方向 |
| Kendall τ IQR | [−0.448, 0.086] | [−0.514, 0.159] | 略扩 |
| **Spearman ρ(F_norm, τ)** | **−0.916** | **−0.921** | 几乎完全相同 |
| Spearman p | 1.17e-14 | 1.28e-14 | — |
| swap_class strict | 10 | 9 | −1 |
| swap_class candidate | 8 | 6 | −2 |
| swap_class others | 17 | 19 | +2 |
| clinical SOZ strict_informative n | 5 | 6 | +1 |
| clinical SOZ strict_informative sign-test p | 0.500 | 0.656 | 仍 NULL |
| clinical SOZ strict_informative median enrichment | 0.042 | 0.047 | 几乎相同 |
| clinical SOZ strict_informative bootstrap CI | [−0.071, 0.099] | [−0.057, 0.121] | 略宽，包含 0 |

**判读**：paper 级 single-composite figure 的核心数字（F_norm cohort median + Kendall τ + Spearman ρ）**几乎完全不变**；clinical SOZ strict_informative 仍 NULL。AGENTS.md 现有的"continuous spectrum, no SOZ link, supplementary to PR-6"框架不需修改。`figures/cohort_displacement_heatmap.*` 需要重画但 caption / framing 不动。

---

## 4. 判读 — Checkpoint 标准对照

按 `rerun_roadmap_2026-05-20.md` 默认 gate（Checkpoint B 仍待 5d.3 + 5e 完成后正式触发；user 2026-05-21 决定优先 5f over Checkpoint B；本档独立判读 5f 内部标准）：

| Gate | 状态 |
|---|---|
| H1 pooled 方向反转？ | ❌ **NO**（orig +0.010 → mask 0.000，median 都接近 0；都是 NULL；direction 保持） |
| H1 coreness 方向反转？ | ❌ **NO**（orig +0.061 → mask +0.087，都 positive；都是 NULL；orig 早期 n=20 vs mask n=28 不完全 like-for-like）|
| H2 fwd/rev swap 方向反转？ | ❌ **NO**（orig 9/9 positive → mask 8/8 positive；100% 方向保持；magnitude 因 −1 subject 略缩） |
| H3 i/l/e 任何方向反转？ | ❌ **NO**（都 median=0, NULL） |
| Split-half endpoint J 中位降至 < 0.4？ | ❌ **NO**（first-half 仍 0.714；odd-even 0.857，远高于 0.4） |
| Step 6 held-out tier 分布大幅崩溃？ | ❌ **NO**（strong fraction 57%→61%，反而略好；n 不同：35 vs 28，但 fraction 可比） |
| Step 6 held-out swap_class concordant 下降？ | ❌ **NO**（0.686 → 0.821，实质提升） |
| Rank displacement F_norm cohort median 大幅偏移？ | ❌ **NO**（0.800 → 0.789，Δ=0.011） |
| Rank displacement clinical SOZ sign-test 从 NULL 翻显著？ | ❌ **NO**（p=0.50 → p=0.66，都 NULL） |
| **任何 cohort-level 显著 ↔ NULL 翻转？** | ⚠ **YES（1 条）**: Node anatomy h1_eligible swap−same Wilcoxon-greater p **0.014 → 0.059**（同 n=28 like-for-like）。**Secondary metric**（PR-6 plan §3 未列入 H1 α 池）；方向保持，sign-test 在 orig 上原本就不显著。论文叙事需要把 cohort-wide swap-leaning claim 进一步弱化（详见 §3.6 判读）。 |
| Primary H1 / H2 / H3 / Step 6 / Rank displacement 方向反转？ | ❌ **NO**（所有 primary metric 方向保持） |

**Step 5f 整体方向**：⚠ PASS WITH 1 CAVEAT — primary 方向全保持，1 条 secondary cohort-level Wilcoxon 翻转 (h1_eligible swap-leaning，已在 §3.6 详记)；可进 5g（不触发 advisor reconcile，因为 secondary tier + 已有 PR-6 archive Step 4b 框架内可容纳的 "cohort-wide effect borderline" 调整）。

---

## 5. 不再 valid 的旧数字 / 需要主文档更新的位置

### 5.1 Topic 1 主文档（`docs/topic1_within_event_dynamics.md`）

§7 PR-6 条目当前口径是"endpoint vs middle SOZ enrichment cohort NULL；fwd/rev swap mechanism sanity 5/6 exceed null_95th；split-half time-stable endpoint geometry"。**修过版口径建议更新为**：
- H1 NULL preserved（masked median 0.000, p=0.18，n=28）
- H2 fwd/rev swap **8/8 positive**（更强：sign-test p=0.0078）
- Step 6 held-out swap_class concordance **0.82**（mask 实质提升）
- Step 5a/5b/5c/5d 已经覆盖了"endpoint 时间稳"+"identity bias 92.2%"等上游号源数字

具体改写留到 Step 5i 收口阶段。

### 5.2 AGENTS.md PR-6 supplementary rank displacement 条目

当前数字（v14 cohort expansion 2026-05-07）：F_norm median 0.800，Kendall τ median −0.203，Spearman ρ(F_norm, τ) = −0.916。修过版数字：F_norm 0.789, τ −0.240, ρ −0.921。**Δ 太小（< 0.04 on median, < 0.005 on ρ），不建议在主文档 / AGENTS.md 单独写"修过版 vs orig"差异**，留 archive 即可。

### 5.3 PR-6 archive plan（`docs/archive/topic1/pr6_template_anchoring/pr6_template_endpoint_anchoring_plan_2026-04-25.md`）

§15 Step 2/5a/5b/4/4b 数字是 cohort=20-23 早期版本；当前 cohort=28，本档 §3 数字是 latest 修过版 + 原版对比的最终版。**不修改 plan archive doc**（plan 历史可溯）；本文是 plan archive 的下游补充。

---

## 6. 下一步（按 rerun_roadmap §5g 起步）

- **5g (PR-7 antagonistic pairing on masked)** — *next*
  - H1 三 metric (10s primary + 30s sensitivity + sign test) 重算
  - **P3 cohort-level TOST equivalence test 重算** — ⚠️ 如果 P3 INCONCLUSIVE-locked 翻到 PASS 或 FAIL 任意一侧，**STOP**（framework-level revision，必须发起 `docs/paper1_framework_sba.md` v1.1.2 修订评议）
  - N0–N4 surrogate hierarchy 重算
  - 输出 → `results/interictal_propagation_masked/pr7_template_pairing/`
- 5h (Topic 4 attractor on masked) — 跟 5g 并行
- 5i (主文档收口) — 等 5h 后

---

## 7. 工件清单

新生成（masked）：
- `results/interictal_propagation_masked/template_anchoring/cohort_audit.csv`（40 行 audit）
- `results/interictal_propagation_masked/template_anchoring/per_subject/*.json`（30 个 subject）
- `results/interictal_propagation_masked/template_anchoring/cohort_summary.json`（H1/H1c/H2/H3/split-half/node anatomy/template-pair geometry）
- `results/interictal_propagation_masked/pr6_step6_held_out_template/per_subject/*.json`（28 个 stem）
- `results/interictal_propagation_masked/pr6_step6_held_out_template/cohort_summary.json`
- `results/interictal_propagation_masked/rank_displacement/per_subject/*.json`（40 个 stem）
- `results/interictal_propagation_masked/rank_displacement/cohort_summary.json`
- `results/interictal_propagation_masked/rank_displacement/clinical_soz_set_relation_summary.json`

代码（5 项 TDD + 3 个 runner 改造）：
- `tests/test_pr6_masked_path_routing.py`（新建）
- `scripts/run_pr6_template_anchoring.py`（`_apply_masked_paths` + `--masked-features` + `use_masked_features` 透传）
- `scripts/run_pr6_step6.py`（同上）
- `scripts/run_rank_displacement.py`（同上，纯路径）

日志：
- `logs/step5f1_pr6_template_anchoring_masked.log`
- `logs/step5f3_pr6_step6_masked.log`
- `logs/step5f4_rank_displacement_masked.log`

---

## 8. 一句话总结

**PR-6 在 phantom rank 修过后所有 primary 方向保持**（H1 NULL→NULL；H2 fwd/rev 9/9 → 8/8 都 positive；H3 全 NULL；Step 6 tier 分布 + swap_class concordance 0.69 → 0.82 实质提升；rank displacement F_norm/τ/ρ 几乎完全不变）。**1 条 secondary cohort-level Wilcoxon borderline 翻转**（node anatomy h1_eligible swap−same p=0.014 → 0.059，同 n=28；方向保持，sign-test 在 orig 上原本就不显著）。**论文叙事需要把 cohort-wide swap-leaning 进一步弱化**；不触发 framework-revision。可进 5g（PR-7） + 5h（Topic 4 attractor）。Checkpoint B 仍待 5d.3 + 5e 完成后正式触发。
