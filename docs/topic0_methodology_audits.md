# Topic 0：方法学审计与数据合同

> 状态：当前正式入口
> 范围：跨 topic 的方法学问题、数据合同 bug、需要回溯重跑的污染源。
> 不属于：单个 topic 内部的科学结论（→ Topic 1–5 主文档）。
> **优先级**：本 topic 的问题一旦未解决，下游 topic 的结论强度受限。`docs/paper_overview.md` 的 "一句话总论" 必须明示当前未结清的 Topic 0 问题。

---

## 1. 这个 topic 只回答什么

本 topic 回答 **"我们用的数据本身是否如其所言"**——而不是"数据告诉我们生物学是什么"。

它**只**记录：

1. 已发现的方法学 bug / 数据合同问题（legacy 继承的、本仓库引入的、未来发现的）
2. 每个问题的实证证据 + 影响范围
3. 修复合同 + 重跑路线图
4. 修复版结果与旧结果的并行存放约定（这样能"看出问题"）

它**不**记录：

- 任何 topic 内部的科学结论（→ 各 topic 主文档）
- 工程基础设施（→ AGENTS.md / CLAUDE.md）
- 文献综述 / framework discussion（→ `docs/paper1_framework_sba.md`）

---

## 2. 一句话当前状态

- **未结清问题 1（已结清）**：`lagPatRank` phantom pseudo-rank（2026-05-20 确诊，**broad re-derivation 全部完成 2026-05-22；5a-5h 全部跑完 + Checkpoint A/B 通过 + P3 framework-flip gate clear (like-for-like cohort verdict INCONCLUSIVE 保持)；剩 5i.6 default flip 单独执行**；§3.1 + §5 + 终稿 `docs/archive/topic0/lagpat_phantom_rank/rerun_results_2026-05-21.md`）
- **未结清问题 2**：SEEG 3D coord loader（**v3 落地 2026-05-21**；§3.2）—— `src/seeg_coord_loader.py` 实现完成；Yuquan 输出 `fs_native_ras_mm`，**Epilepsiae 自动发现 MRI + 应用 MNI152 affine 输出 `mni152_1mm`**（37 unit tests + real-data smoke pass）；当前 stable_k=2 cohort 约 22/34 已可直接进 H1/H2 主分析（Yuquan 8 subject-native + Epilepsiae 14 MNI mm）；剩下待办 = 把 loader 接进 Phase 1 runner（`load_subject_for_phase1()` integration PR）
- **当前进行中的重跑**：见 §5 重跑路线图。**Topic 1 / Topic 4 / PR-5 / PR-6 / PR-7 的所有数字背后仍挂方法学 caveat**，直到 §5 走完
- **当前 Phase 0 数字方向（2026-05-22 更新，全部 5a–5h + 5g 完成 + Checkpoint A/B 通过 + P3 framework-flip gate clear）**：**未发现任何 primary cohort verdict 翻转**。大结构（K=2 主导、簇内刻板度真实、簇内 86% → 92% identity bias **加强**）经得起 phantom 修复；细颗粒（具体事件归类、复现集合成员）会变。**统计层变化**：(1) **1 条 exploratory/secondary tier loss**（PR-4B Step 23 L3 高置信子集 n=8 Pearson r delta 原版 p=0.016 → 修过版 p=0.547 — 小样本脆弱性，原版 PR-4B archive 已 pre-registered 为 exploratory tier，**不进 main evidence base，不是 primary cohort verdict reversal**）；(2) 5f node anatomy h1_eligible swap−same Wilcoxon p=0.014 → 0.059 — secondary tier borderline 弱化；(3) PR-5 fig_a / fig_b 3 条 secondary（pre-registered 不进主 Bonferroni 池）；(4) PR-7 short-window mark-dependent 偏离方向上加强但 cohort-level verdict INCONCLUSIVE 保持。所有 primary cohort verdict 方向保持。Checkpoint B verdict = **trigger met soft form, 不影响主结论**, broad re-derivation 按 plan 继续。
- **下一步**：5e PR-5 / 5g PR-7 / 5h Topic 4 attractor 在用户并行 session 处理中；5i 收口阶段需小幅修订 Topic 1 §3.1c L3 exploratory tier 措辞（不动主 verdict），详见 `docs/archive/topic0/lagpat_phantom_rank/checkpoint_b_report_2026-05-21.md`

---

## 3. 已记录的方法学问题

### 3.1 `lagPatRank` phantom pseudo-rank（已确诊 2026-05-20，broad re-derivation 已完成 2026-05-22）

**问题**：legacy 数据生成代码（`ReplayIED/.../hfo_net.py:289`）对**所有通道**排名次，包括"不参与"的通道——这些通道获得来自谱图噪声的虚假整数名次。HFOsp 把这些假名次直接喂给 KMeans，造成事件归类被假名次驱动。

**实证证据**（chengshuai/FC10477Y, n_ch=8, n_ev=3513）：
- 9064 个本该空缺的格子全部有 0–7 的整数名次
- 分布 U 形：rank=7 占 19.1%、rank=0 占 17.8%、中段 ~10%
- 修过版 vs 原版 KMeans 分类的事件归属一致程度 = 0.001（噪声基线 0.95）

**40-subject cohort 实测**（`results/lagpatrank_audit/cohort_summary.csv`）：
- cohort-median Δ（修过版 vs 原版） = **-0.599**
- **40/40** subject 都越 broad-rederivation 阈值（Δ < -0.10）
- 主线 stable_k=2 cohort 唯一最佳类别数翻转：epilepsiae_916（2→4）
- 3 个高 k 翻转都在 stable_k>2 cohort（这部分已被 Topic 4 H3 主分析独立排除）

**方法学定性**（不是 noise）：
phantom-driven clustering 在 seed 间**完全可重现**（noise floor median 0.997）——bug 制造了一组与真实激活顺序无关、但稳定、可重现的 cluster。性质上与 PR-4 panel d "簇内 86% identity bias" 同类。

**受影响 PR / 重跑状态表（2026-05-21 更新）**：

| PR / 单元 | 状态 | 方向 | 关键发现 | 归档 |
|---|---|---|---|---|
| PR-2 主聚类 | ✅ 重跑完 | 一致 | 35 stable_k=2 (orig 30 → mask 35)；唯一翻转 epilepsiae_916 stable_k=2→4；event-level Jaccard 中位 0.70；AMI 与噪声基线相关 0.961 | 5a |
| PR-2.5 split-half/odd-even | ✅ 重跑完 | 一致 | grade 31/9/0 → 26/12/2（整体降一档无崩塌）；fwd/rev 16/17 → 15/16；含 5 翻动 4 个 non-degenerate 与 5a 自洽 | 5b |
| PR-3 簇内 MI / stereotypy | ✅ 重跑完 | **加强** | legacy MI 不动 (40/40 仍 p<0.05)；raw τ 中位 0.237 → 0.291（+0.054，39/40 同向 p=1e-10）；bias_fraction 87.9% → 92.2% — PR-4 panel d "86% identity bias" 被加强到 92% | 5c |
| PR-4A 昼夜模板占比 | ✅ 重跑完 | 一致 NULL | masked p=0.73 / orig p=0.12，同方向 NULL | 5d.1 |
| PR-4B Step 0 lag validation | ✅ 重跑完 | 一致 | exact order match=1.0；dominant r 0.580 vs orig 0.601；8/40 pass dom_r>0.7 高置信门 | 5d.2.0 |
| PR-4B Step 1 rate-state coupling | ✅ 重跑完 | 一致 NULL | L1 dominant rho 翻 −0.083 → +0.183, 25/40 同向 p=0.20 NS | 5d.2.1 |
| **PR-4B Step 23 (L3 high-confidence Pearson r, n=8)** | ✅ 重跑完 | ⚠ **exploratory/secondary tier loss**（**不是 primary reversal**） | orig "唯一显著" finding (n=8, dom_r>0.7, p=0.016, 7/8, Δ=+0.083) 修过版降到 (p=0.547, 5/8, Δ=+0.053) — 小样本脆弱性 finding，**原版 PR-4B archive 已 pre-registered 为 exploratory tier，不进 main evidence base，本身不是 primary cohort verdict reversal**；Topic 1 §3.1c 措辞从"探索性显著"改为"原版探索性显著，masked 后不复现，归 fragility-on-small-n" | 5d.2.2 + CkB |
| PR-4C 发作邻近 | ✅ 重跑完 | 一致 NULL | 40 subject 全跑，cohort verdict 不变 | 5d.3 |
| PR-4D 模板速率分解 | ⛔ **SCOPE CUT, not missed run** | n/a — 主动不跑 | user 2026-05-21 决定优先 5e/5f/5g/5h（PR-4D 不在论文 main evidence base 内，原版本身是 descriptive layer），phantom-rank 修过版重跑明确**不在 Phase 0 scope 内**；如未来需要 PR-4D 数字可单立小 PR 重跑 | — |
| PR-5-A novel template gate | ✅ 重跑完 | 一致 | overall_pass=True 两版本一致；cohort grew main 23→27 / aux 22→26 | 5e |
| PR-5-B template recruitment shift | ✅ 重跑完 | **主结论保持 + magnitude ≈ 同** | `dominant_rate.candidate_a_global.post_minus_baseline`: orig +65.46 events/h (p=0.0013, sign 19/4) → mask +65.66 (p=0.0004, sign 21/6)；direction + magnitude 100% preserved | 5e |
| PR-5-B composition diagnostic (share) | ✅ 重跑完 | ⚠ **secondary 显著 → NULL 翻转** | fig_a `share post_minus_baseline` main p=0.015 → 0.86（median 0.016 → 0.002）；fig_a extended 同向；fig_b transition lift NULL → Wilcoxon-only borderline；**plan §4.5 明确不进主 Bonferroni 池**，论文 fig_a/fig_b 叙事需在 5i.2 调整 | 5e |
| PR-6 H1 endpoint vs middle SOZ enrichment | ✅ 重跑完 | 一致 NULL | n=28=28 cohort identical；median +0.010 → 0.000；Wilcoxon-greater p 0.22 → 0.18 | 5f.1 |
| PR-6 H2 fwd/rev swap (mechanism sanity) | ✅ 重跑完 | 一致 | orig 9/9 → mask 8/8 都 positive；net −1 subject (5b fwd/rev 翻动)；magnitude 略缩；**maintained, not strengthened** | 5f.1 |
| PR-6 H3 focus_rel (i/l/e) | ✅ 重跑完 | 一致 NULL | i/l/e 全 median=0 都 NULL | 5f.1 |
| PR-6 Step 4b node anatomy (h1_eligible swap−same) | ✅ 重跑完 | ⚠ **secondary cohort Wilcoxon borderline 翻转** | 同 n=28 like-for-like：Wilcoxon p=0.014 → 0.059；sign-test p=0.17 → 0.70（orig 也未达 α=0.05）；方向保持；论文叙事进一步收紧 | 5f.1 |
| PR-6 Step 6 held-out | ✅ 重跑完 | **加强** | tier 分布稳健；**swap_class concordance 0.69 → 0.82 实质提升**；新 fail epilepsiae_635（与 5b fwd/rev 翻转自洽） | 5f.3 |
| PR-6 supplementary rank displacement | ✅ 重跑完 | 一致 | F_norm 0.800 → 0.789；Kendall τ −0.20 → −0.24；Spearman ρ(F_norm,τ) −0.916 → −0.921 几乎完全相同；clinical SOZ strict_informative 仍 NULL | 5f.4 |
| PR-7 antagonistic temporal pairing | ✅ 重跑完 | 一致（exploratory directional 信号 logged） | H1 triple-gate NULL stays NULL (orig p=0.844 → mask p=1.000)；burst run_length_lift 0.977 → 1.006 都 ≈ 1；**P3 framework-flip gate CLEAR**：on orig 6-cohort 用 mask features verdict = INCONCLUSIVE 完全保持 (4/4 flag 一致)。broader cohort (n=8 or n=30) NULL 是 cohort × power × 5b 翻动效应非 phantom-rank statistic effect。phantom-rank 在 short-window 反而 mask 真实 mark-dependent 偏离（orig 10s median −0.018 → mask −0.045）—— directional finding，CI 仍宽 verdict 不变 INCONCLUSIVE，作 PR-7 v2 power-analysis 跟踪 | 5g (2026-05-22) |
| Topic 4 attractor Step 1 | ✅ 重跑完 | **加强** | cohort 34（少 916 stable_k 翻转）；GOF pass 33/34 = 97% 持平；**coordinate-free λ₂（H3 主直测）orig 10/34 → mask 13/34**，5 个原 NULL/borderline 翻到 p≤0.001（pengzihang/253/442/958/1096），0 个反向丢；var_curve 中位 0.953 → 0.729（phantom 在 PCA-3 子空间假结构去掉的预期方向） | 5h |
| cluster_geometry PCA embedding | ⛔ 未独立重跑 | — | 该图旧 framework lock 时已归 Tier 0 archive，未在 phase 0 re-derivation list；如需补可单立小 PR | — |

**修复方式**：consumer 侧 masked re-rank + per-event 中点 (0.5) impute。不重跑 legacy producer（破坏 paper 字节级 reproducibility，且代价过高）。详见：
- 计划：`docs/superpowers/plans/2026-05-20-lagpatrank-phantom-rank-audit-and-fix.md`
- 诊断归档：`docs/archive/topic0/lagpat_phantom_rank/diagnostic_2026-05-20.md`
- 白话报告：`docs/archive/topic0/lagpat_phantom_rank/plain_chinese_report_2026-05-20.md`
- 重跑路线图：`docs/archive/topic0/lagpat_phantom_rank/rerun_roadmap_2026-05-20.md`

**重要 disclaimer**：修过版只去除了"端点名次假"的偏置，**不**解开"事件归类是被排序驱动还是被参与模式驱动"这道更深层的问题。后者由 cluster geometry / participation-pattern 框架的后续工作回答。

**2026-05-21 重跑期间科学发现（更新 PR-4 panel d 解读）**：

Step 5c PR-3 在 masked 上重跑簇内刻板度，发现：
- 簇内 raw τ：原版 0.237 → 修过版 0.291（+0.054，39/40 同向，p=1.27e-10）
- 簇内 centered τ：基本不动（p=0.69 NS）
- bias_fraction：87.9% → 92.2%（p=3.17e-4）

**朴素翻译**：phantom 之前在做的是给数据**加噪声**（稀释"通道身份"信号），**不是伪造身份偏置**。把 phantom 去掉，"簇内一致性大部分来自通道身份排序"这条结论**被加强**——PR-4 panel d 的 86% 在干净数据上其实是 92%。

**对 SEF-ITP framework 的科学意义**：endpoint 作为"结构化空间锚点"的几何前提（即模板由稳定的通道身份排序驱动，而非事件特异性传播）**被独立证据加强**，不被削弱。详见 `docs/archive/topic0/lagpat_phantom_rank/step5c_pr3_results_2026-05-20.md`。

### 3.2 SEEG 3D 坐标：v3.1 loader 已落地（2026-05-21）

**当前状态**：两套数据源都在挂载里，loader 已实现 + 27-subject smoke + 49 unit tests 全 GREEN（2026-05-21）。

| 数据集 | 坐标源 | `coord_space` | Phase 1 适用 | cohort 覆盖 (stable_k=2) |
|---|---|---|---|---|
| Yuquan | `/mnt/yuquan_data/yuquan_images/.../patients_elecs_reGen/<sid>/chnXyzDict.npy` | `fs_native_ras_mm` (subject T1 RAS, mm) | ✅ subject-level H1/H2/H6 主分析 | 8 / 16 |
| Epilepsiae | SQL `electrode` + MRI `pat_*/MRI/mri_*.img` (FSL MNI152 1mm grid，27 病人 affine 一致) | `mni152_1mm` (auto-discover, mm) | ✅ subject + cohort-aggregate；cohort-pooled brain map 需带 `normalization_certainty="grid_confirmed_warp_type_unverified"` caveat | 14 / 18 |

**Phase 1 启动 cohort**：≈ 22 subject。两数据集**不能** pool 做点云（坐标空间不同）。

**实现**：`src/seeg_coord_loader.py::load_subject_coords` v3.1。完整 schema + 10 条 strict invariants（NULL→missing 不填 0、bipolar midpoint、顺序锚点、空间合同 lock、canonical_subject_id 防 cross-patient、MRI affine byte 匹配 MNI152、默认 raise 而非 voxel fallback）+ MNI152 27-subject 验证证据（shape / affine / brain mask 质心 SD / 卷相关 全表）+ Yuquan `fs_native_ras_mm` 判定证据（`reg.sh` rigid-only、`useRealRAS 1` header、3.501 mm contact 间距、负值范围排除 MNI）+ Phase 1 消费侧 `assert_coord_result_is_mm_for_main_analysis()` 合同，**全部归档在** [`docs/archive/topic0/seeg_coord_loader/v3_1_lock_2026-05-21.md`](archive/topic0/seeg_coord_loader/v3_1_lock_2026-05-21.md)。

**仍待办（不阻塞 H1/H2 启动）**：找原始 Epilepsiae processing 文档 / warp field → `normalization_certainty` 升级到 `mni_normalized_verified`（cohort MNI brain map 声明时 nice-to-have）。

---

## 4. 命名约定 — 修复版结果如何与旧结果区分

为了让"修过版 vs 旧版"能并排对比，约定以下结构：

### 4.1 结果目录（默认 parallel dir）

修复版结果 = **旧结果路径 + `_masked` 后缀** 的并行目录：

```
results/interictal_propagation/             ← 旧（phantom-contaminated），不动
results/interictal_propagation_masked/      ← 新（masked re-rank）

results/topic4_attractor/                   ← 旧
results/topic4_attractor_masked/            ← 新

results/interictal_propagation/pr6_template_anchoring/   ← 旧
results/interictal_propagation_masked/pr6_template_anchoring/   ← 新
```

**理由**：
- 旧结果对旧归档 doc 是 evidence，不可删
- 修复版与旧版并存便于 diff、可视化对比、独立审计
- git ls-files 上 path 不重叠，blame / log 干净

### 4.2 图文件名（不加后缀）

修复版图直接放在 `_masked/` 子目录里，**文件名不重复加 `_masked`**——目录已经区分了，加后缀只会冗余：

```
results/interictal_propagation_masked/figures/cohort_propagation_summary.png   ✓
results/interictal_propagation_masked/figures/cohort_propagation_summary_masked.png   ✗
```

例外：当某张对比图直接放在原 dir 里（比如 PPT 文件夹），文件名加 `_v2masked` 后缀。

### 4.3 对比图（专门的 dir）

如果做 "before-vs-after" 并排对比图（典型场景：让评委一眼看出 phantom 把哪些事件挪到了哪一类），放在专门的 `_vs_masked/` dir：

```
results/interictal_propagation_vs_masked/figures/
  per_subject/yuquan_chengshuai_cluster_relabel.png   ← 显示 phantom 把哪些事件挪了类
  cohort_stable_k_flip.png
```

### 4.4 per-subject JSON 字段

per-subject JSON 顶层不动现有 schema；如果同时要在一个文件里携带旧+新结果，约定 key：

```json
{
  "adaptive_cluster": {...},                ← 旧（phantom-contaminated）
  "adaptive_cluster_masked": {...},         ← 新（masked）
  "audit": {                                ← 跨修复的对比汇总
    "ami_audit_minus_floor": -0.599,
    "stable_k_changed": false,
    "labels_relabel_jaccard": 0.34
  }
}
```

但**默认不混存**——修过版结果就完全独立写到 `_masked/` 目录的 JSON，避免老 reader 不小心读到混合 schema。

---

## 5. 重跑路线图

详见：`docs/archive/topic0/lagpat_phantom_rank/rerun_roadmap_2026-05-20.md`

按依赖顺序：

| Step | 内容 | 输出目录 | 状态 |
|---|---|---|---|
| 5a | PR-2 re-cluster on masked | `results/interictal_propagation_masked/` per_subject | **已完成 2026-05-20**（35 主线 / 34 仍 K=2；只 epilepsiae_916 翻 2→4 已知 outlier；event-level Jaccard median 0.70；AMI 与噪声基线距离相关 0.961）；详见 `docs/archive/topic0/lagpat_phantom_rank/step5a_pr2_results_2026-05-20.md` |
| 5b | PR-2.5 split-half / odd-even on masked | 同上 | **已完成 2026-05-20**（复现 grade 31/9/0 → 26/12/2；fwd/rev cohort 16/17 → 15/16；5 例位置换位 4 例 non-degenerate 可追溯到 5a 事件归类变化，非新伪影）；详见 `step5b_pr25_results_2026-05-20.md` |
| **Checkpoint A** | advisor consult: stable_k flip + forward/reverse 复现集合变化 | — | **PASSED 2026-05-20** |
| 5c | PR-3 per-cluster MI on masked labels | 同上 | **已完成 2026-05-20**（不需新跑，5a 主管道已含）；簇内 raw τ +0.054 增强 39/40；centered τ 不动；bias_fraction 87.9% → 92.2% **加强 PR-4 panel d**；详见 `step5c_pr3_results_2026-05-20.md` |
| 5d.1 | PR-4A 昼夜模板占比 on masked | 同上 | **已完成 2026-05-20**（masked p=0.73 / 原版 p=0.12 同方向 NULL，结论不动） |
| 5d.2.0 | PR-4B Step 0 lag validation | 同上 | **已完成 2026-05-21**（exact order match=1.0，与原版一致；dominant r 0.580 vs 原 0.601；8/40 pass dom_r>0.7 高置信门，与原版数量同） |
| 5d.2.1 | PR-4B Step 1 rate-state coupling | 同上 | **已完成 2026-05-21**（全 NULL；L1 dominant rho 翻 −0.083 → +0.183, 25/40 同向 p=0.20 NS；cohort verdict 不动） |
| 5d.2.2 | PR-4B Step 23 (lag span + L3 Pearson) | 同上 | **已完成 2026-05-21** ⚠ 原版 exploratory finding (L3 dom_r>0.7 子集 n=8 Pearson r delta, p=0.016, 7/8) 在修过版降到 p=0.547, 5/8 — **exploratory/secondary tier loss**（原版 PR-4B archive 已 pre-registered 为 exploratory tier，**不进 main evidence base**，本身**不是 primary cohort verdict reversal**），Checkpoint B 2026-05-21 verdict: trigger met soft form, broad re-derivation 继续 |
| 5d.3 | PR-4C 发作邻近 cohort | 同上 | **已完成 2026-05-21**（n=30 usable subjects；5 propagation 指标 × 3 windows 全 NULL，与原版"5 indicators all NULL" 历史封板同向；rate_by_template 终点由 5e PR-5-B 重做不在本 step 范围）；详见 `step5d3_pr4c_results_2026-05-21.md` |
| 5d.4 | PR-4D 模板速率分解 | n/a | ⛔ **SCOPE CUT — 主动不跑，不是漏跑**。user 2026-05-21 决定优先 5e/5f/5g/5h；PR-4D 原版本身是 descriptive layer，不在论文 main evidence base 内；phantom-rank 修过版重跑明确不在 Phase 0 scope 内。如未来需要 PR-4D 数字，可单立小 PR 重跑 |
| **Checkpoint B** | advisor consult: PR-4B/D 方向 + L3 显著消失 reconcile | — | **VERDICT 2026-05-21: trigger met (soft form), 不影响主结论, broad re-derivation 继续**。仅 1 条 secondary tier "显著 → NULL" 翻转 = PR-4B L3 高置信 (n=8) Pearson r delta 同方向 (p 0.016 → 0.547)，原版 archive 已标 exploratory，归 fragility-on-small-n；L1 sign flip 但 NS 两侧；其他 NULL stays NULL。5i 收口仅需在 Topic 1 §3.1c 把 L3 高置信 finding 措辞从"探索性显著"改为"原版探索性显著，masked 后不复现，归 fragility-on-small-n"；不动 cohort-level 主结论。详见 `checkpoint_b_report_2026-05-21.md` |
| 5e | PR-5 / PR-5-B on masked（核心：post-ictal +65.46 events/h, p=0.00128） | `results/interictal_propagation_masked/template_share_switching/` | **已完成 2026-05-21**（3 个 aux script 加 `--masked-features` + filename trap fix；3 项新 TDD path-routing PASS；既有 54/54 PASS 无回归）：**plan §4 pre-registered primary 主结论保持**——PR-5-A gate overall_pass=True (6/6 axis × 2 configs, mask cohort main 27/aux 26)；PR-5-B retained `dominant_rate.candidate_a_global.post_minus_baseline = +65.66 events/h, Wilcoxon p=0.0004, sign 21+/6−, bonferroni PASS`（orig +65.46/p=0.0013/19+/4−，方向 100% 保持 magnitude 几乎相同）。**3 条 plan §4.5 + PR-7 §17 pre-registered secondary diagnostic 翻转**（intersection like-for-like 复核确认非 cohort 替换造成；plan §4.5 明确"share 不进主 Bonferroni 池"）：(a-b) fig_a `composition_diagnostic.share post−base` retained + extended 双 cohort 同向显著→NULL（p=0.015→0.86 / p=0.006→0.82；median +0.016→+0.002 magnitude collapse；原版本身已是 panel d 反方向 direction_consistent 6/23，本身不算 cohort claim）；(c) fig_b `transition lift post−base` NULL→Wilcoxon-only borderline sig（p=0.29→0.022, sign p=0.076 不达 cohort 门槛，反方向翻转）。**Checkpoint B 不重新打开**（CkB 报告 §3.3 "5e 后单独评估"的触发条件指向 PR-5-B 主信号翻转，本档主信号未翻）。**论文 fig_a 叙事需调整**（5i 收口时执行）：从"dominant share 抬升"改为"dominant 绝对率抬升、share 维持"；fig_b 从"transition 不变"改为"transition Wilcoxon-only borderline 抬升 + sign-test 未达 cohort 门槛"。不动 PR-5 plan §4 核心 cohort claim。详见 `step5e_pr5_results_2026-05-21.md` |
| 5f | PR-6 endpoint anchoring / swap / Step 6 / rank displacement on masked | `results/interictal_propagation_masked/{template_anchoring,pr6_step6_held_out_template,rank_displacement}/` | **已完成 2026-05-21（user 优先 5f over Checkpoint B；后者仍待 5d.3 + 5e 完成正式触发）**（5 项 TDD + 3 个 runner 加 `--masked-features`；92/92 既有测试无回归）：所有 primary 方向保持——H1 pooled NULL 不变（n=28, p 0.22→0.18）；H2 fwd/rev swap 节点级 9/9 → 8/8 都 positive（mask sign p=0.0078, 因 −1 subject magnitude 略缩）；H3 全 NULL；Step 6 held-out swap_class concordance **0.69 → 0.82 实质提升**；rank displacement F_norm 0.800 → 0.789，Spearman ρ(F_norm,τ) −0.916 → −0.921 几乎不变。**1 条 secondary cohort Wilcoxon borderline 翻转**：node anatomy h1_eligible swap−same p=0.014 → 0.059（同 n=28；方向保持，sign-test 在 orig 上原本就 NULL，论文叙事需进一步弱化 cohort-wide swap-leaning claim）。详见 `step5f_pr6_results_2026-05-21.md` |
| 5g | PR-7 antagonistic pairing on masked | `results/interictal_propagation_masked/template_pairing/` | **已完成 2026-05-22**（3 个 runner 加 `--masked-features`；3 项新 TDD + 既有 30 项 = 33/33 PASS 无回归）：H1 triple-gate cohort-level NULL 与 orig 同方向；burst diagnostic run_length_lift 0.977 → 1.006 都 ≈ 1 NULL；N2 window sweep 10/30/60 min 全 NULL。**P3 framework-flip gate CLEAR**：on like-for-like orig 6-subject cohort (用 masked features) verdict = INCONCLUSIVE 完全保持（4/4 flag 与 orig 一致）；broader cohort (n=8 或 n=30) verdict 形式上 NULL 但归 cohort × power 互相作用（5b 把 548/635 翻 out fwd/rev 集合 + n 增大让 CI 收紧），**不**是 phantom-rank 在 P3 statistic 上的直接效应。`docs/paper1_framework_sba.md` v1.1.2 **不修订**。3 层 cohort P3 verdict 全部归档（`pr7_addendum_p3{,_orig6_cohort,_all30_eligible}.json`）。详见 `step5g_pr7_results_2026-05-21.md` |
| 5h | Topic 4 attractor Step 1 on masked | `results/topic4_attractor_masked/` | **已完成 2026-05-21**（10 项 TDD + 5 个 script 加 `--masked-features`+ `src/topic4_attractor_diagnostics.py:build_rank_feature_matrix` 加 `mask_phantom` 参数；92/92 既有测试无回归）：cohort 34（orig 35，少 916 因 5a stable_k 翻 4）；GOF pass 持平 33/34（fail subject 从 916 换成 1077 var 0.658→0.515）；**coordinate-free λ₂（H3 主直测）显著加强**：orig 10/34 → mask **13/34**，5 个原 NULL/borderline 翻到 p≤0.001（pengzihang/253/442/958，加 1096 原 drift 排除现进 cohort），2 个失去显著都无害（916 cohort exit + liyouran p=0.001→0.002 borderline，λ₂ 实际值几乎不动）。Principal curve var_curve 中位 0.953 → 0.729（phantom 在 PCA-3 子空间贡献的假结构被去掉的预期方向，GOF 闸门 0.6 仍稳）。**没有任何 H3 primary 方向反转**。详见 `step5h_topic4_attractor_results_2026-05-21.md` |
| 5i | 主文档更新 + AGENTS.md cross-PR 合同更新 + default flip + missed-paths fix | `docs/topic{0,1,4}_*.md` / AGENTS.md / CLAUDE.md / `src/cluster_geometry.py` / `scripts/run_interictal_propagation.py` PR-4 bootstrap branches | **5i.1–5i.5 + 5i.7 已完成 2026-05-22**：Topic 0 二级 archive `rerun_results_2026-05-21.md` 已落字；Topic 1 §2 PR-5/6/7 + §3 caveat 表更新；Topic 4 framework doc header 加 Phase 0 解锁 + H1–H6 verdict 摘要；AGENTS.md Cross-PR 条目改"完成"；CLAUDE.md §5 phantom worked example 落字；§3.1 受影响 PR 表 21 行状态化。**5i.6 default flip 进行中 2026-05-22**：use_masked_features / mask_phantom 默认 False → True + DeprecationWarning + cluster_geometry & PR-4 bootstrap branches missed-path fix（review-issue P0）。**PR-4D 显式 scope cut 2026-05-21**：user 决定优先 5e/5f/5g/5h，不补跑。 |

**Phase 0 整体进度（2026-05-22 更新）**：14 / 14 main step 完成（5a/5b/5c/5d.1/5d.2.0/5d.2.1/5d.2.2/5d.3 + Checkpoint A + **5e** + 5f + Checkpoint B + **5g** + **5h**），**5i 收口剩 default flip + missed-path fix**。**所有 plan-pre-registered primary cohort verdict 方向保持，无 framework-level 翻转**（P3 framework-flip gate 在 like-for-like orig 6-cohort 上 verdict INCONCLUSIVE 完全保持 2026-05-22 verified）。**Phase 0 验收口径**（per 2026-05-22 review）：

- **科学层**：✅ 5a–5h + 5g 主结论方向全部保持，3 条加强（PR-3 bias_fraction、PR-6 Step 6 concordance、Topic 4 λ₂），1 条 **exploratory/secondary loss**（PR-4B L3 高置信 n=8 Pearson r delta p=0.016→0.547 — **小样本脆弱性 finding, 原版 archive 已标 exploratory, 不进 main evidence base，本身不是 primary cohort reversal**），4 条 secondary cohort flip（PR-5 share×2 + transition + PR-6 node anatomy h1_eligible Wilcoxon）。framework-level revision 0 触发。
- **工程层**：⏳ 收口未封板——`use_masked_features` / `mask_phantom` 默认仍是 False（5i.6 进行中），`src/cluster_geometry.py` PCA 嵌入路径还未走 masked features（5i.6 一起修），`scripts/run_interictal_propagation.py` PR-4* bootstrap 7 个 callsites 漏传 `use_masked_features`（5i.6 一起修）。
- **文档层**：✅ 主文档 / AGENTS.md / CLAUDE.md / Topic 4 framework doc / Topic 0 二级 archive / INDEX.md 全部已更新到 Phase 0 closure 状态（2026-05-22）。

详见 Topic 0 二级 archive `rerun_results_2026-05-21.md` + Checkpoint B 报告 `checkpoint_b_report_2026-05-21.md` + 5e/5f/5g/5h 各 step archive。

每个 Step 跑完，更新 §3.1 的 "受影响 PR" 表，把对应行从"待重跑"挪到"重跑完，方向 = 一致 / 翻转 / NULL flip"。

**Checkpoint A / B 是 hard stops**：未通过 advisor consult 不进入下一阶段。

**PR-7 P3 翻转特别 flag** *（2026-05-22 status: gate CLEAR）*：5g P3 verdict on like-for-like orig 6-cohort 用 mask features 保持 INCONCLUSIVE（4/4 flag 一致），未触发 framework revision。broader cohort verdict 形式上滑到 NULL 归 cohort × power 互相作用非 phantom-rank 在 statistic 上的直接效应，作 PR-7 v2 power-analysis 跟踪条目。

---

## 6. 文件清单

### 代码

- `src/lagpat_rank_audit.py` — masked re-rank helpers（`build_masked_kmeans_features` canonical entry）
- ✅ `src/interictal_propagation.py` — 5 个 compute helper 加 `use_masked_features` 参数（`compute_adaptive_cluster_stereotypy` / `compute_cluster_stereotypy` / `compute_time_split_reproducibility` / `compute_held_out_endpoint_validation` / `run_subject_interictal_propagation_pr1`）；5i.6 进行中 = 默认 False → True + DeprecationWarning + 7 个 PR-4* bootstrap callsites pass-through
- ⏳ `src/cluster_geometry.py:compute_pca_embedding` — 同参数加（5i.6 一起做）；当前仍是 phantom-vulnerable 路径
- ✅ `src/topic4_attractor_diagnostics.py:build_rank_feature_matrix` — `mask_phantom` 参数（5h Step 5h 加；5i.6 一起 default flip）

### 测试

- `tests/test_lagpat_rank_audit.py`（5 tests pass 2026-05-20）

### 脚本

- `scripts/audit_kmeans_phantom_rank.py` — 40-subject AMI audit driver
- `scripts/augment_lagpat_audit_masked_stable_k.py` — masked stable_k 重选
- `scripts/plot_lagpat_phantom_audit.py` — 3 张诊断图

### 数据 / 图（已落地）

- `results/lagpatrank_audit/cohort_summary.csv`（40 行）
- `results/lagpatrank_audit/<sid>.json` × 40
- `results/lagpatrank_audit/figures/`：
  - `ami_vs_noise_floor.{png,pdf}`
  - `phantom_fraction_vs_delta.{png,pdf}`
  - `stable_k_confusion.{png,pdf}`
  - `README.md`

---

## 7. 与其他 topic 的边界

- Topic 1–5 主文档的科学结论引用本 topic 时，必须明示 "假设 §3.1 phantom-rank 重跑完成且方向一致"
- 任何下游 PR plan 在 trigger 前必须检查 §3 是否有未结清的 audit 直接影响该 PR
- AGENTS.md "Cross-PR Contract Lookups" 会增加本 topic 的 lookup 条目（待 §5i 更新）
- **Topic 4 SEF-ITP framework**（`docs/topic4_sef_itp_framework.md` v1 lock 2026-05-20）将本 topic §3.1 phantom-rank 修复 + §5 broad re-derivation 列为 **Phase 1+ 的硬前置**；Phase 0 完成前 SEF-ITP 处于"等待修复完成的占位文档"状态。§5a–§5h 的每一 step 同时是 SEF-ITP Phase 0 子任务。

---

## 8. 历史文档索引

- `docs/superpowers/plans/2026-05-20-lagpatrank-phantom-rank-audit-and-fix.md` — plan-of-record（含 noise-floor anchor gate 设计 + advisor checkpoint A/B + PR-7 framework flag）
- `docs/archive/topic0/lagpat_phantom_rank/diagnostic_2026-05-20.md` — 技术诊断归档（cohort 数字 + gate verdict）
- `docs/archive/topic0/lagpat_phantom_rank/plain_chinese_report_2026-05-20.md` — 白话报告（给非技术读者 / 自己回看用）
- `docs/archive/topic0/lagpat_phantom_rank/rerun_roadmap_2026-05-20.md` — Step 5a-5i 重跑路线图
- `docs/archive/topic0/lagpat_phantom_rank/step5a_pr2_results_2026-05-20.md` — Step 5a PR-2 修过版结果
- `docs/archive/topic0/lagpat_phantom_rank/step5b_pr25_results_2026-05-20.md` — Step 5b PR-2.5 修过版结果
- `docs/archive/topic0/lagpat_phantom_rank/step5c_pr3_results_2026-05-20.md` — Step 5c PR-3 修过版结果
- `docs/archive/topic0/lagpat_phantom_rank/step5e_pr5_results_2026-05-21.md` — Step 5e PR-5 / PR-5-B（novel-template gate + template recruitment shift + share extended + transition windows + fig_a/b 修过版图）修过版结果
- `docs/archive/topic0/lagpat_phantom_rank/step5f_pr6_results_2026-05-21.md` — Step 5f PR-6 全套（template_anchoring + Step 6 held-out + rank displacement）修过版结果
- `docs/archive/topic0/lagpat_phantom_rank/step5h_topic4_attractor_results_2026-05-21.md` — Step 5h Topic 4 attractor diagnostics Step 1（principal curve + GOF + coordinate-free λ₂）修过版结果
- `docs/archive/topic0/lagpat_phantom_rank/step5g_pr7_results_2026-05-21.md` — Step 5g PR-7 antagonistic temporal pairing 修过版结果（**真版 2026-05-22**：H1 triple-gate NULL stays NULL；P3 like-for-like orig 6-cohort verdict INCONCLUSIVE 完全保持；新 directional finding "phantom-rank 在 short-window mask 真实 mark-dependent 偏离"作 PR-7 v2 跟踪条目）
- `docs/archive/topic0/lagpat_phantom_rank/rerun_results_2026-05-21.md` — **Phase 0 终稿**（5a-5h 整合 + Cross-PR reconcile 表 + 论文叙事调整清单 + 5i 收口剩余任务）
- `docs/archive/topic0/lagpat_phantom_rank/checkpoint_b_report_2026-05-21.md` — Checkpoint B advisor consult report (5d 完成后)
- `docs/archive/topic0/lagpat_phantom_rank/phase0_progress_report_2026-05-21.md` — Phase 0 进度跟踪报告
- `docs/archive/topic1/propagation/lagpatrank_phantom_audit_diagnostic_2026-05-20.md` — **MOVED 2026-05-20** 到 `docs/archive/topic0/lagpat_phantom_rank/diagnostic_2026-05-20.md`（原 path 仅留 redirect stub）
