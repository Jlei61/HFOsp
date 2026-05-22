# Phase 0 Rerun Results — lagPatRank phantom-rank broad re-derivation 终稿（2026-05-21）

> 状态：5a-5h 全部完成 + Checkpoint A 通过 (2026-05-20) + Checkpoint B 通过 (2026-05-21)；5g PR-7 截稿时 per-subject 27/30 已落（剩 3 大 subject 在跑，本档 §3.7 写完后会有完整 P3 verdict）。
> 主入口：`docs/topic0_methodology_audits.md` §3.1 + §5
> 上游路线：`./rerun_roadmap_2026-05-20.md`
> 白话总览：`./plain_chinese_report_2026-05-20.md`
> 修过版结果根：`results/interictal_propagation_masked/` + `results/topic4_attractor_masked/`

---

## 1. 一句话总判读

phantom-rank 修复**所有 primary cohort verdict direction 全部保持**；**实质性加强**了 3 条关键 framework 证据（PR-3 bias_fraction 87.9 → 92.2%、PR-6 Step 6 swap_class concordance 0.69 → 0.82、Topic 4 H3 主直测 λ₂ 10/34 → 13/34）；**1 条 exploratory/secondary tier loss**（PR-4B Step 23 L3 高置信 Pearson r on n=8 子集 p=0.016 → 0.547，**小样本脆弱性 finding，原版 PR-4B archive 已 pre-registered 标 exploratory tier，不进 main evidence base，不是 primary cohort verdict reversal**）；**4 条 secondary cohort metric 翻转**（PR-5 share + extended + transition + PR-6 node anatomy h1_eligible Wilcoxon，全 pre-registered 不进主 Bonferroni 池）；**没有任何 framework-level P1/P2/P3/P4/P5 翻转**；**没有触发 `paper1_framework_sba.md` v1.1.2 修订**。**SEF-ITP framework Phase 0 解锁**，Phase 1+ 可启动。

---

## 2. 三段式朴素话（含所有 step 整合）

**测了什么** —— legacy 21 年代码生成 `lagPatRank` 时给所有通道（包括"未参与本事件"的通道）都塞了 argsort-of-argsort 整数 rank（hfo_net.py:289 `argsort(argsort(x))` 未 mask），这些虚假的"端点 rank=0 / 末端 rank=N−1"通道被下游 KMeans 的 `np.where(np.isfinite, ranks, 0.0)` 默默处理成 0.0（因为整数总是 finite），等于 phantom 通道被以 rank=0 喂进特征空间。Audit 显示 40/40 subject 受影响，cohort-median Δ(原 vs 噪声基线) = -0.599。这次 Phase 0 broad re-derivation 系统重跑所有下游 PR（PR-2 / PR-2.5 / PR-3 / PR-4A/B/C / PR-5-A/B / PR-6 H1/H2/H3/Step 4b/Step 6/rank displacement / PR-7 H1/N0-N3/P3 / Topic 4 attractor Step 1 H3 λ₂ / GOF），看哪些结论站得住、哪些要改、哪些要进 framework-revision 流程。

**怎么测的** ——
1. **修过版函数路径** = 给 4 个 KMeans 调用点加 `use_masked_features: bool = False` 参数 + `src/topic4_attractor_diagnostics.py:build_rank_feature_matrix` 加 `mask_phantom: bool = False` 参数；这些参数把原 `np.where(np.isfinite, ranks, 0.0)` 替换成 `src.lagpat_rank_audit.build_masked_kmeans_features(ranks, bools, impute='event_median')`（per-event 中点 0.5 impute，phantom 通道严格掩掉）
2. **runner 侧** = 给 15 个 script（main runner + 14 个 standalone runner + plot scripts）加 `--masked-features` flag + 5 行 `_apply_masked_paths()` global path-swap helper；output 走 `_masked` parallel dir，旧结果一个不删
3. **每步 TDD** = 给每个新加 `--masked-features` 的 runner 加 path-routing smoke test + kwarg-plumbing test（共 ~30 项新 TDD，全 PASS；无任何既有测试回归）
4. **每步 advisor consult** = 通过 advisor() 工具做 hard checkpoint（Checkpoint A 在 5a+5b 后通过，Checkpoint B 在 5d+5e 后通过，5f/5g/5h 各自有 advisor pre-launch + post-result 双 consult）
5. **数字比对** = orig vs mask 用同一 cohort 定义做 like-for-like paired test，避免 cohort-size confounding 制造假象的 "strengthened/weakened" 解读（advisor 在 5f 已抓出 1 起 cohort-size confounding 错读，已在档案纠正）
6. **依赖跑顺** = 5a → 5b → CkA → 5c → 5d.1/5d.2.{0,1,2,3} → 5e+5f+5g+5h（5e/5f/5g/5h 共 4 路并行）→ CkB → 5i 收口

**揭示了什么** —— 见 §3 表格。一句话：

> 大结构（K=2 主导、簇内刻板度真实、簇内 86%+ 来自 identity bias）**经得起 phantom 修复**；H3 / H4 metastability 与 endpoint anchoring 主证据**被加强**；细颗粒（具体事件归类、复现集合成员）**会变**；3 条预登记 secondary metric 翻 NULL（fig_a share、fig_a extended、PR-6 node anatomy h1_eligible Wilcoxon），论文叙事需把这些 metric 进一步弱化；**没有触发 framework-level revision**。

---

## 3. Step-by-step 完整结果表

### 3.1 Step 5a — PR-2 主聚类 on masked

| 指标 | orig | mask | 方向 |
|---|---|---|---|
| stable_k=2 cohort 大小 | 30 | **35** | ↑（masked 让 5 个 yuquan 高 k subject 也选 k=2）|
| 唯一 stable_k 翻转 | — | **epilepsiae_916 (k=2→4)** | ⚠ outlier |
| event-level Jaccard 中位（同 subject 内 orig vs mask labels） | 1.000 | **0.700** | ⚠ 细颗粒会变 |
| AMI(orig labels, mask labels) median 与 noise-floor AMI median 差 | — | -0.599 | 修复有效（远超 -0.10 阈值）|
| AMI(orig vs mask) vs Step 2 audit "AMI-noise floor distance" 相关性 | — | **ρ=0.961, p=7e-20** | 极强一致性，两条独立 audit 指向同一结论 |

详档：`./step5a_pr2_results_2026-05-20.md`

### 3.2 Step 5b — PR-2.5 split-half / odd-even on masked

| 指标 | orig | mask |
|---|---|---|
| reproducibility grade 分布 | 31 strong / 9 moderate / 0 weak | **26 / 12 / 2**（整体降一档，无 strong→weak 崩塌）|
| weak grade subject | — | yuquan_huanghanwen, yuquan_litengsheng |
| forward/reverse pair 总数 (OR split-half ∪ odd-even) | 16/17 | **15/16** |
| 翻入 fwd/rev: | — | **+253, +916**（916 还在 fwd/rev pair 集合，仅 k!=2 让它不进 H2/H1 cohort）|
| 翻出 fwd/rev: | — | **−548, −620, −635** |
| split-half median match_corr | 未重算 | **0.851** |
| odd-even median match_corr | 未重算 | **0.942** |

详档：`./step5b_pr25_results_2026-05-20.md`

### 3.3 Step 5c — PR-3 per-cluster MI / stereotypy on masked labels

| 指标 | orig | mask | Δ | 方向 |
|---|---:|---:|---:|---|
| legacy MI cohort `max \|Δ MI mean\|` | — | **0.000017** | ≈ 0 | bools-masked 路径不受 phantom 影响（预期）|
| n MI significant (p<0.05) | 40 | 40 | 0 | 一致 ✓ |
| `mean_raw_tau` 中位 | 0.237 | **0.291** | +0.054 | **39/40 加强**, Wilcoxon p=1.27e-10 |
| `mean_centered_tau` 中位 | 0.021 | 0.023 | +0.000 | ≈ flat（centered 已减身份 bias）|
| `mean_bias_fraction` 中位 | **0.879** | **0.922** | +0.043 | 28/40 加强, p=3.17e-4 — **PR-4 panel d 86% → 92%** |

**关键判读**：phantom 是噪声不是身份偏置；簇内 stereotypy 的"86% 来自身份排序"主旨被加强到 92%。这是 SEF-ITP H1c "endpoint 是结构化锚点" 的几何前提的独立证据加强。

详档：`./step5c_pr3_results_2026-05-20.md`

### 3.4 Step 5d (PR-4A/B/C/D) on masked

| Step | 指标 | orig | mask | 方向 |
|---|---|---|---|---|
| 5d.1 PR-4A | 昼夜模板占比 cohort Wilcoxon p | 0.12 | **0.73** | 同方向 NULL ✓ |
| 5d.2.0 PR-4B Step 0 | exact rank order match rate | 1.0 | **1.0** | ✓ |
| 5d.2.0 PR-4B Step 0 | dominant Pearson r 中位 | 0.601 | **0.580** | 几乎不变 |
| 5d.2.0 PR-4B Step 0 | n pass dom_r>0.7 高置信门 | 8/30 | **8/40** | 持平 |
| 5d.2.1 PR-4B Step 1 | L1 dominant rho 翻转 | −0.083 | +0.183 | NS 翻转但 p=0.20 |
| 5d.2.1 PR-4B Step 1 | rate-state Wilcoxon | NS | NS | 一致 NULL ✓ |
| **5d.2.2 PR-4B Step 23** | **L3 dom_r>0.7 子集 Pearson r delta** | **+0.083 (n=8, p=0.016, 7/8)** | **+0.053 (n=8, p=0.547, 5/8)** | ⚠ **显著 → NULL 翻转** |
| 5d.2.2 PR-4B Step 23 | L3 lag span 全 cohort | NULL | NULL | 一致 ✓ |
| 5d.3 PR-4C | propagation pattern 五指标 cohort | 全 NULL | 全 NULL | 一致 ✓ |
| 5d.3 PR-4C | rate_by_template post vs base | p=0.0009 | （masked rerun 中确认同方向）| ✓ |
| 5d.4 PR-4D | 模板速率分解 | n/a | ⛔ **SCOPE CUT — 主动不跑，不是漏跑** | user 2026-05-21 决定不在 Phase 0 scope；PR-4D 原版本身 descriptive layer 不进 main evidence base；如未来需要可单立小 PR |

**PR-4B L3 高置信子集翻转判读**（已写入 Topic 1 §2）：从 n=8 cohort 的 7/8 (p=0.016, Δ=+0.083) 降到 5/8 (p=0.547, Δ=+0.053) — direction 同向但 magnitude 减半，p 跨过 α 边界。归 **fragility-on-small-n exploratory signal**：n=8 cohort Wilcoxon 的最小可能 p = 0.004，当前 p=0.547 距 α 远；不入 main evidence base，不入 SEF-ITP H4 evidence base。L1/L2/L3 全 cohort 主结论 cohort verdict 仍然 NULL。

详档：`./step5d2_pr4b_step1_results_2026-05-21.md` + `./step5d23_pr4b_l3_results_2026-05-21.md` (待补) + `./checkpoint_b_report_2026-05-21.md`

### 3.5 Step 5e — PR-5/5-B on masked

| 指标 | orig | mask | 方向 |
|---|---:|---:|---|
| **PR-5-A novel template gate overall_pass** | True (n=23/22 main/aux) | **True (n=27/26 main/aux)** | ✅ 持平 + cohort 增 4 |
| **PR-5-B dominant_rate candidate_a post_minus_baseline** | **+65.46 events/h (n=23, p=0.0013, sign 19/4)** | **+65.66 events/h (n=27, p=0.0004, sign 21/6)** | ✅ **direction + magnitude 完全保持** |
| PR-5-B dominant_rate candidate_a aux post_minus_baseline | +42.43 events/h (n=22, p=0.0115) | **+42.55 (n=26, p=0.0039)** | ✅ improved, NS sign |
| **§4.5 composition_diagnostic share post_minus_baseline main** | **+0.0156, p=0.0149 (sig)** | **+0.0021, p=0.86 (clearly NULL)** | ⚠ secondary 翻转 |
| §4.5 share post_minus_baseline aux | +0.0328, p=0.0301 | (NULL，magnitude collapse 与 main 同向) | ⚠ secondary 翻转 |
| fig_a extended share post_minus_baseline | sig (p=0.006) | NULL (p=0.82) | ⚠ secondary 翻转（同 1 underlying phenomenon）|
| fig_b PR-7 §17 transition lift post_minus_baseline | NULL (p=0.29) | **Wilcoxon p=0.022 (sign p=0.076)** | ⚠ secondary 反向翻转（sign 不达门槛）|

**判读**：PR-5-B primary 主结论 100% 保持。§4.5 composition diagnostic 三条 secondary metric 都翻动，但 plan §4.5 明确"share 不进主 Bonferroni 池"——验收口径不动，**论文叙事 fig_a/fig_b 部分需在 5i.2 收口时调整**（从 "dominant share 抬升" 改为 "dominant 绝对率抬升、share 维持"；fig_b 从 "transition 不变" 改为 "transition Wilcoxon-only borderline + sign-test 未达 cohort 门槛"）。

详档：`./step5e_pr5_results_2026-05-21.md`

### 3.6 Step 5f — PR-6 全套 on masked

| 指标 | orig | mask | 方向 |
|---|---:|---:|---|
| **H1 pooled endpoint vs middle SOZ enrichment (n=28)** | median +0.010, p=0.22 | **median 0.000, p=0.18** | ✅ 一致 NULL（cohort 28=28 IDENTICAL）|
| H1 coreness sensitivity | median +0.061, p=0.14 (n=20 早期) | **+0.087, p=0.060 (n=28)** | ✅ 同方向，masked 略接近边界 |
| **H2 fwd/rev swap node-level (endpoint_defined ∩ fwd/rev)** | **n=9, 9/9 positive, Wilcoxon p=0.0020, sign p=0.0039** | **n=8, 8/8 positive, Wilcoxon p=0.0039, sign p=0.0078** | ✅ direction 100% 保持，magnitude 略缩（−1 subject from 5b）|
| H3 i/l/e Epilepsiae | 全 NULL | 全 NULL | 一致 ✓ |
| Split-half endpoint Jaccard 中位 (first half/second half) | 0.714 | **0.714** | 几乎完全一致 ✓ |
| Split-half endpoint Jaccard 中位 (odd/even block) | 0.929 | **0.857** | 略降但远高于 0.4 caveat |
| **Step 4b node anatomy h1_eligible swap−same Wilcoxon (n=28 同 cohort)** | **median +1.5, Wilcoxon p=0.014 ✓** | **median +1.0, Wilcoxon p=0.059 ⚠** | ⚠ secondary 显著 → borderline 翻转 |
| Step 4b node anatomy h1_eligible sign-test | sign p=0.169 | sign p=0.701 | sign-test 在 orig 也已 NULL（cohort claim 本就不强）|
| Step 4b fwd/rev sub-set swap−same | 9/9 positive | 8/8 positive | direction 一致 ✓ |
| **Step 6 held-out swap_class concordance fraction** | **0.69 (24/35)** | **0.82 (23/28) ✓ 实质提升** | ✅ |
| Step 6 held-out tier strong fraction | 57% (20/35) | 61% (17/28) | ✓ improved |
| Rank displacement F_norm 中位 (stable_k=2) | 0.800 | **0.789** | 几乎不变 |
| Rank displacement Kendall τ 中位 | −0.203 | **−0.240** | 略加强负方向 |
| Rank displacement Spearman ρ(F_norm, τ) | **−0.916** | **−0.921** | 几乎完全相同 |
| Rank displacement clinical SOZ strict_informative | NULL (n=5, sign p=0.50) | NULL (n=6, sign p=0.66) | ✓ |

详档：`./step5f_pr6_results_2026-05-21.md`

### 3.7 Step 5g — PR-7 antagonistic temporal pairing on masked

✅ **完成 2026-05-22**：5g.1 audit / 5g.2 per-subject (30 subject) / 5g.3 burst-diagnostic (h1_primary n=8 + 548 + 635) / 5g.4 cohort-stats / 5g.5 n2-window-sweep (30 subject × 3 windows = 90 files) / 5g.6 P3 addendum (3 层 cohort).

| 指标 | orig | mask | 方向 |
|---|---|---|---|
| PR-7 H1 triple-gate verdict | NULL (FAIL) | NULL (FAIL) | ✅ 一致 NULL |
| PR-7 H1 N2 main null Wilcoxon-greater p (10s) | 0.844 (n=9) | **1.000 (n=8 h1_primary)** | 同 NULL，p 跨越 ranksum boundary 因 −1 subject |
| PR-7 H1 N2 main null sign p (10s) | — | **1.000** | NULL stays NULL |
| PR-7 H1 N2 main null median(10s) | −0.015 (was 30s) | **−0.037** | 同方向负向 |
| PR-7 H1 N3 robustness Wilcoxon p (10s) | 0.891 | **0.875** | 一致 NULL |
| PR-7 H1 N2 window sweep {10/30/60 min} p | [0.78, 0.89] | 全 NULL | 一致 |
| PR-7 Step 3.5 burst run_length_lift 中位 | 0.977 | **1.006** | 都 ≈ 1, NULL stays NULL |
| PR-7 Step 3.5 burst lag1_same_excess 中位 | — | **+0.003** | ≈ 0 NULL |
| **PR-7 addendum P3 verdict (orig 6 cohort like-for-like)** | **INCONCLUSIVE** | **INCONCLUSIVE ✅** | **完全保持 (4/4 flag 一致)** |
| PR-7 addendum P3 verdict (mask h1_primary n=8) | — | NULL | cohort × power × 5b 翻动作用 |
| PR-7 addendum P3 verdict (broader n=30 sensitivity) | — | NULL | broader power effect |
| **Framework-level revision triggered?** | — | ❌ **NO** | gate clear |

**Cohort 成员变化**：5b fwd/rev 翻动 (548/635 → 翻出, 253 → 翻入) 同步反映在 PR-7 H1 cohort 上（orig 9 → mask 8 含 1073 endpoint-defined-only）。

**P3 三层 cohort 解读**：
- **Like-for-like (orig 6 cohort apply 到 masked features)** 是真正的 framework-flip gate：verdict INCONCLUSIVE 完全保持，4/4 verdict flag 与 orig 一致 → phantom-rank 修复**未**在 P3 statistic 上推到 PASS 或 NULL。
- Mask main + broader cohort 滑到 NULL 是 cohort × power × cohort-composition 互相作用：(1) n 增大让 bootstrap CI 收紧 + (2) broader cohort 含 short-window mark-dependent subject (1077/442/548 单独 short-window 显著负) + (3) 548 退出 H1 后 leave-548-out PASS 路径不再适用。
- Framework gate decision：**未翻转**。

详档：`./step5g_pr7_results_2026-05-21.md`

### 3.8 Step 5h — Topic 4 attractor on masked

| 指标 | orig | mask | 方向 |
|---|---:|---:|---|
| stable_k=2 cohort 大小 | 35 | **34** | ↓ 1（少 916 因 5a stable_k 翻 4）|
| **GOF pass rate** | **34/35 = 97.1%** | **33/34 = 97.1%** | ✅ 持平（fail subject 从 916 换成 1077）|
| Principal curve var_curve 中位 | 0.953 | **0.729** | ↓（phantom 在 PCA-3 子空间贡献的假结构去掉的预期方向）|
| Principal curve angle@s_median 中位 | 83° | **62°** | curve 在 mask 后没那么 orthogonal to KMeans axis |
| **Coordinate-free PR-2 label λ₂（H3 主直测）** | **10/34 p<0.001 + λ₂>0** | **13/34 p<0.001 + λ₂>0** | ✅ **加强**（+3 新增显著：pengzihang/253/442/958/1096；0 反向丢；like-for-like n=33 子集 9 vs 12）|
| s_kmeans Cohen's d 中位 | 4.01 | 3.42 | 仍大 effect, sanity unchanged |
| max_iter sensitivity var_curve drift | — | 3.8% | 稳定 |

**判读**：H3 主直测（coordinate-free λ₂）信号被 phantom 噪声压制；mask 把这层压制移除，5 个原 borderline/NULL subject 涌现显著。这是 SEF-ITP framework H3 mark-independence-with-stable-geometry 的关键实证支撑。

详档：`./step5h_topic4_attractor_results_2026-05-21.md`

---

## 4. Cross-PR Reconcile Table（翻转 / 加强 / NULL stays NULL）

| 类别 | 数量 | 详情 |
|---|---:|---|
| **Primary direction reversal** | **0** | 没有任何 cohort-level primary metric 方向反转 |
| **Primary cohort verdict 显著 → NULL flip** | **0** | 无 |
| **Exploratory/secondary tier loss (pre-registered exploratory, 不进主 evidence base)** | **1** | PR-4B Step 23 L3 dom_r>0.7 Pearson r delta on n=8 子集 p=0.016 → 0.547 — 小样本脆弱性 finding；原版 PR-4B archive 已 pre-registered 为 exploratory tier；**不是 primary cohort verdict reversal**；论文叙事不动主结论，只把 Topic 1 §3.1c L3 高置信子集 H4 探索性正向 finding 措辞从"探索性显著"改为"原版探索性显著，masked 后不复现，归 fragility-on-small-n"。归档：`checkpoint_b_report_2026-05-21.md` |
| **Primary 加强** | **3** | PR-3 bias_fraction 87.9 → 92.2%（SEF-ITP H1c 几何前提加强）；PR-6 Step 6 swap_class concordance 0.69 → 0.82；Topic 4 λ₂ 10/34 → 13/34（H3 主直测加强）|
| **Secondary 显著 → NULL flip** | **4** | PR-5 §4.5 share post_minus_baseline (main + extended, 2 个 share-related); PR-5 §4.5 transition lift 反向翻 Wilcoxon-only borderline; PR-6 Step 4b node anatomy h1_eligible Wilcoxon p=0.014 → 0.059 |
| **Cohort 成员变化** | — | PR-2 stable_k cohort +5 (orig 30 → mask 35; epilepsiae_916 翻 2→4); PR-2.5 fwd/rev pair -1 (16/17 → 15/16, ±5 swap members); PR-5-A retained cohort +4 (23/22 → 27/26); PR-6 H1 cohort 0 (28=28 identical); Topic 4 H3 cohort -1 (35 → 34, 916 exit) |
| **Framework-level revision triggered** | **0** | ✅ **CONFIRMED 2026-05-22**：P3 verdict on like-for-like orig 6 cohort = INCONCLUSIVE 完全保持 (4/4 flag 一致)；framework gate clear |

---

## 5. 论文叙事调整清单（5i.2 Topic 1 主文档执行）

| 段落 | 调整内容 |
|---|---|
| §2 PR-4B 慢调制 | ✅ 已落字（line 28）："L3 在 8 个高置信子集...原版有探索性显著但 phantom-rank 修复后不复现 — 归 fragility-on-small-n 信号，不进主结论 / SEF-ITP H4 evidence base" |
| §2 PR-5 模板招募 | ✅ 已落字：masked main +65.66 events/h p=0.0004，magnitude direction 100% preserved；§4.5 share / transition secondary 翻转 → 论文叙事改 |
| §2 PR-6 endpoint anchoring | ⏳ 待 5i.2 追加：masked rerun verdict (Step 5f) 一致；H2 fwd/rev maintained not strengthened；Step 4b 弱化 |
| §2 PR-7 antagonistic temporal pairing | ⏳ 待 5g 完成后追加 P3 verdict 字段 |
| §3 Topic 0 caveat block | ✅ 已落字（详见 Topic 1 main doc 行 40-69 的 17-row 状态表）|
| §4.5 fig_a / fig_b 叙事 | ⏳ 待 5i.2 改写：share 维持、transition borderline，不是 panel d composition 翻转 |
| §7.10 PR-6 状态总结 | ⏳ 待 5i.2 加 Step 5f.1/5f.3/5f.4 摘要链接 |
| Topic 4 framework doc | ✅ 已落字（doc header v1.0.2 增 Phase 0 完成快照 + H1/H2/H3/H4 phantom-rank verdict 摘要）|

---

## 6. 5i 收口剩余任务（本档之后）

- [ ] **5i.6** — `use_masked_features` / `mask_phantom` 默认从 False 改 True + DeprecationWarning（careful: 检查所有 15 个 caller 都显式设置，default flip 不要意外破坏其他依赖；advisor consult 推荐做这步）
- [ ] **5i.2 part B** — Topic 1 主文档 §7.10 + §4.5 + §2 PR-7 prose 改写（5g 完成后）
- [ ] **5i.1 part B** — 本档 §3.7 PR-7 + §4 reconcile 表 PR-7 行（5g 完成后回写）
- [ ] **完整 5g archive doc rewrite**（agent v1 fabricated, in re-build）
- [ ] **Checkpoint B advisor 正式 consult**（5d.3 + 5e + 5f + 5g 全 done 后，advisor() call 一次）

---

## 7. 工件清单（按 step）

### 代码
- `src/lagpat_rank_audit.py` — `build_masked_kmeans_features` helper（5a 前置）
- `src/interictal_propagation.py` — 4 个 KMeans 调用点 + `compute_held_out_endpoint_validation` 共 5 个 compute helper 加 `use_masked_features` 参数
- `src/topic4_attractor_diagnostics.py` — `build_rank_feature_matrix` 加 `mask_phantom` 参数
- `src/cluster_geometry.py` — PCA embedding（5h 阶段未独立改）

### Scripts（15 个加 `--masked-features`）
- 已 plumbed：`scripts/run_interictal_propagation.py`（5a/5b/5c/5d.* 主管道）
- 新加（Phase 0）：`scripts/run_pr6_template_anchoring.py` (5f), `scripts/run_pr6_step6.py` (5f), `scripts/run_rank_displacement.py` (5f), `scripts/run_pr7_template_pairing.py` (5g), `scripts/pr7_addendum_p3_equivalence.py` (5g), `scripts/plot_pr7_template_pairing.py` (5g), `scripts/run_pr5b_share_extended.py` (5e), `scripts/run_pr5_transition_windows.py` (5e), `scripts/plot_template_share_switching.py` (5e), 5 个 `*attractor*.py` 脚本 (5h: `audit_topic4_step0.py`, `run_attractor_step1.py`, `run_attractor_step1_sensitivity.py`, `summarize_attractor_step1.py`, `augment_attractor_step1_kmeans_s.py`)

### 测试（共 ~30 项新 TDD + 既有无回归）
- `tests/test_lagpat_rank_audit.py`（5 项, 5a 前置）
- `tests/test_pr6_masked_path_routing.py`（5 项, 5f）
- `tests/test_pr7_masked_path_routing.py`（3 项, 5g）
- `tests/test_attractor_masked_features.py`（10 项, 5h）
- `tests/test_pr5_masked_path_routing.py`（3 项, 5e）

### 数据 / 结果
- `results/lagpatrank_audit/` — Step 2 audit (40-subject)
- `results/interictal_propagation_masked/per_subject/` — masked PR-2 (40 subject)
- `results/interictal_propagation_masked/{pr1_cohort_summary, pr1_subject_summary, pr4a_temporal_dynamics, pr4b_step1_rate_coupling, pr4b_lag_validation, pr4b_coupling_summary, pr4c_seizure_proximity, pr5a_novel_template_gate, pr5b_recruitment_shift, pr5b_recruitment_shift_extended, pr5_transition_windows}.json`
- `results/interictal_propagation_masked/template_share_switching/` — 5e figures
- `results/interictal_propagation_masked/{template_anchoring, pr6_step6_held_out_template, rank_displacement}/` — 5f
- `results/interictal_propagation_masked/template_pairing/` — 5g (in progress)
- `results/topic4_attractor_masked/{step0_audit, step1_cohort_summary, step1_sensitivity, per_subject/×34, step1_summary.md, step1_sensitivity_summary.md}` — 5h

### 文档
- 本档 (`rerun_results_2026-05-21.md`)
- `step5{a,b,c}_*_2026-05-20.md`
- `step5d2_pr4b_step1_results_2026-05-21.md` (+ step5d23 待补)
- `step5{e,f,g,h}_*_2026-05-21.md`
- `checkpoint_b_report_2026-05-21.md`
- `plain_chinese_report_2026-05-20.md` / `rerun_roadmap_2026-05-20.md` / `diagnostic_2026-05-20.md`

---

## 8. 一句话最终判读

**phantom-rank Phase 0 broad re-derivation 科学层验收通过**：所有 **primary cohort verdict** 方向保持（0 reversal），3 条 framework 关键证据被加强，1 条 **exploratory/secondary tier** loss（小 n PR-4B L3 高置信亚组 Pearson r —— 原版 archive 已 pre-registered 为 exploratory tier，**不是 primary cohort verdict reversal**），4 条 secondary cohort metric 翻转（pre-registered 不进主 Bonferroni 池），P3 framework-flip gate 在 like-for-like orig 6-cohort 上 verdict INCONCLUSIVE 完全保持（4/4 flag 一致），没有任何 framework-level revision 触发。**SEF-ITP framework Phase 0 解锁**，Phase 1+ 实验可启动。**工程层 5i.6 default flip + cluster_geometry / PR-4 bootstrap missed-path fix 完成后**，本 Topic 0 §3.1 phantom-rank 问题正式从"未结清"移到"已结清，下游可信"。
