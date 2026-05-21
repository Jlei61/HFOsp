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

- **未结清问题 1（已基本结清）**：`lagPatRank` phantom pseudo-rank（2026-05-20 确诊，**broad re-derivation 2026-05-21 完成 5a-5h；5g PR-7 per-subject 28/30 已落剩 2 大 subject 还在跑，预计 16:00 前后完，P3 verdict 出后正式收口**；§3.1 + §5 + 终稿 `docs/archive/topic0/lagpat_phantom_rank/rerun_results_2026-05-21.md`）
- **未结清问题 2**：SEEG 3D coord loader（**v3 落地 2026-05-21**；§3.2）—— `src/seeg_coord_loader.py` 实现完成；Yuquan 输出 `fs_native_ras_mm`，**Epilepsiae 自动发现 MRI + 应用 MNI152 affine 输出 `mni152_1mm`**（37 unit tests + real-data smoke pass）；当前 stable_k=2 cohort 约 22/34 已可直接进 H1/H2 主分析（Yuquan 8 subject-native + Epilepsiae 14 MNI mm）；剩下待办 = 把 loader 接进 Phase 1 runner（`load_subject_for_phase1()` integration PR）
- **当前进行中的重跑**：见 §5 重跑路线图。**Topic 1 / Topic 4 / PR-5 / PR-6 / PR-7 的所有数字背后仍挂方法学 caveat**，直到 §5 走完
- **当前 Phase 0 数字方向（2026-05-21 14:00 更新，Checkpoint B 之前所有 step + 5f 完成）**：**未发现任何主结论 cohort verdict 翻转**。大结构（K=2 主导、簇内刻板度真实、簇内 86% → 92% identity bias **加强**）经得起 phantom 修复；细颗粒（具体事件归类、复现集合成员）会变。**统计层有 2 条 secondary tier 显著性变化**：(1) PR-4B Step 23 L3 高置信子集 Pearson r delta 原版 p=0.016 (7/8, n=8) → 修过版 p=0.547 (5/8) — **同方向 (+), 幅度减半**，归 fragility-on-small-n（原版 archive 已标 exploratory tier，不属于 cohort 主结论）；(2) 5f node anatomy h1_eligible swap−same Wilcoxon p=0.014 → 0.059 — 同方向 borderline 弱化（secondary tier，不动主 H1/H2/H3 verdict）。所有 primary cohort verdict 方向保持。Checkpoint B verdict = **trigger met soft form, 不影响主结论**, broad re-derivation 按 plan 继续。
- **下一步**：5e PR-5 / 5g PR-7 / 5h Topic 4 attractor 在用户并行 session 处理中；5i 收口阶段需小幅修订 Topic 1 §3.1c L3 exploratory tier 措辞（不动主 verdict），详见 `docs/archive/topic0/lagpat_phantom_rank/checkpoint_b_report_2026-05-21.md`

---

## 3. 已记录的方法学问题

### 3.1 `lagPatRank` phantom pseudo-rank（已确诊 2026-05-20，重跑中）

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
| **PR-4B Step 23 (L3 high-confidence Pearson r)** | ✅ 重跑完 | ⚠ **显著 → NULL 翻转** | orig 唯一显著 finding (n=8, dom_r>0.7, p=0.016, 7/8, Δ=+0.083) 修过版降到 (p=0.547, 5/8, Δ=+0.053) — 归 fragility-on-small-n，**不**进主结论；**已写入 Topic 1 §2** | 5d.2.2 + CkB |
| PR-4C 发作邻近 | ✅ 重跑完 | 一致 NULL | 40 subject 全跑，cohort verdict 不变 | 5d.3 |
| PR-4D 模板速率分解 | ⛔ SKIPPED | — | user 决定优先 5e/5f/5g/5h，留待 5i 收口判定是否补 | — |
| PR-5-A novel template gate | ✅ 重跑完 | 一致 | overall_pass=True 两版本一致；cohort grew main 23→27 / aux 22→26 | 5e |
| PR-5-B template recruitment shift | ✅ 重跑完 | **主结论保持 + magnitude ≈ 同** | `dominant_rate.candidate_a_global.post_minus_baseline`: orig +65.46 events/h (p=0.0013, sign 19/4) → mask +65.66 (p=0.0004, sign 21/6)；direction + magnitude 100% preserved | 5e |
| PR-5-B composition diagnostic (share) | ✅ 重跑完 | ⚠ **secondary 显著 → NULL 翻转** | fig_a `share post_minus_baseline` main p=0.015 → 0.86（median 0.016 → 0.002）；fig_a extended 同向；fig_b transition lift NULL → Wilcoxon-only borderline；**plan §4.5 明确不进主 Bonferroni 池**，论文 fig_a/fig_b 叙事需在 5i.2 调整 | 5e |
| PR-6 H1 endpoint vs middle SOZ enrichment | ✅ 重跑完 | 一致 NULL | n=28=28 cohort identical；median +0.010 → 0.000；Wilcoxon-greater p 0.22 → 0.18 | 5f.1 |
| PR-6 H2 fwd/rev swap (mechanism sanity) | ✅ 重跑完 | 一致 | orig 9/9 → mask 8/8 都 positive；net −1 subject (5b fwd/rev 翻动)；magnitude 略缩；**maintained, not strengthened** | 5f.1 |
| PR-6 H3 focus_rel (i/l/e) | ✅ 重跑完 | 一致 NULL | i/l/e 全 median=0 都 NULL | 5f.1 |
| PR-6 Step 4b node anatomy (h1_eligible swap−same) | ✅ 重跑完 | ⚠ **secondary cohort Wilcoxon borderline 翻转** | 同 n=28 like-for-like：Wilcoxon p=0.014 → 0.059；sign-test p=0.17 → 0.70（orig 也未达 α=0.05）；方向保持；论文叙事进一步收紧 | 5f.1 |
| PR-6 Step 6 held-out | ✅ 重跑完 | **加强** | tier 分布稳健；**swap_class concordance 0.69 → 0.82 实质提升**；新 fail epilepsiae_635（与 5b fwd/rev 翻转自洽） | 5f.3 |
| PR-6 supplementary rank displacement | ✅ 重跑完 | 一致 | F_norm 0.800 → 0.789；Kendall τ −0.20 → −0.24；Spearman ρ(F_norm,τ) −0.916 → −0.921 几乎完全相同；clinical SOZ strict_informative 仍 NULL | 5f.4 |
| PR-7 antagonistic temporal pairing | 🟡 **重跑中** | (待 5g 完) | code+audit 完成 (audit cohort H1=8 / H2=22 / 全 eligible=30)；per-subject 还在并行跑（29 subject × ~50 min 不等）；P3 verdict 待出 | 5g (in-flight 2026-05-21) |
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

### 3.2 SEEG 3D 坐标：数据源已确认，但当前代码缺 loader（2026-05-21 修订）

**修订 note**：本节首版（2026-05-21 早）写"`/mnt` 里没有坐标"——**这一结论错了**。坏味道：只查了 EDF/SQL 主数据根（`/mnt/yuquan_data/yuquan_24h_edf/`、`/mnt/epilepsia_data/all_data_sqls/` 第一层），漏掉了影像 / 电极坐标侧的挂载路径。用户回归后立刻补查，找到两套现成数据。

**真实情况**：两套独立坐标源都在挂载里。当前缺的是 **loader 代码**，不是数据本身。

#### 3.2.1 Yuquan 坐标源（已确认 + 2026-05-21 坐标空间已查清）

- **位置**：`/mnt/yuquan_data/yuquan_images/nii格式及点电极坐标/caseAndMRI/yuquan_24h_mriCT/patients_elecs_reGen/<subject>/`
- **文件格式**：
  - `chnXyzDict.npy`：Python pickle dict，key = shaft 前缀（A / B / ...），value = (n_contacts, 3) numpy 数组
  - `<shaft>.dat`：ASCII 文本备份，每行 `x y z`
- **坐标空间**：**`fs_native_ras_mm`**（subject-native FreeSurfer / T1 RAS，单位 mm）—— **不是 MNI**
- **空间判定证据**（用户 2026-05-21 实测）：
  - `reg.sh` 只有 CT→T1 的 `flirt -dof 6`（rigid-body），**无** MNI normalize 步
  - 原 `patients_elecs/*.dat` 文件 header 显式声明 `useRealRAS 1` → RAS / mm
  - `patients_elecs_reGen` 相邻 contact 间距全局中位数 **3.501 mm** → 按电极规格重采样 / 补点（非 voxel）
  - legacy 脚本对 `chnXyzDict.npy` 应用 `inv(mri_affine)` 转 voxel → 反推输入是 RAS / mm
  - 坐标范围含**负值** → 与 MNI cohort space 形态不符
- **Phase 1 主分析适用性**：✅ 可直接作 3D Euclidean 主距离，单位 mm
- **当前覆盖**：20 个 Yuquan subject 中 11 个有坐标目录；stable_k=2 子集 16 个里 **8 个当前通道可全量映射**

#### 3.2.2 Epilepsiae 坐标源（**v3 大修订 2026-05-21**：MNI152 grid 已确认）

> **修订 note**：本节中段版（2026-05-21）写"subject-native MRI voxel ijk，必须 per-subject affine"——**这一结论错了**。用户独立调查 27 个 Epilepsiae MRI 后发现实际是 MNI152 标准 grid。下面 v3 修订版。

**位置（SQL 坐标）**：`/mnt/epilepsia_data/all_data_sqls/pat_*.sql` 表 `electrode`

**位置（MRI 文件 — 新发现）**：每个病人都有 `.img/.hdr` Analyze 对：

```
/mnt/epilepsia_data/inv/pat_<sql_id>/adm_<adm_id>/MRI/mri_<adm_id>.img
/mnt/epilepsia_data/inv_1_part/pat_<sql_id>/...
/mnt/epilepsia_data/inv2/pat_<sql_id>/...
```

共 **27 个**病人 MRI 都在挂载里。

**坐标空间（v3 修订）**：

- **SQL coord 类型**：`sql_voxel_ijk`（MRI voxel index）
- **应用 MRI affine 后**：**`mni152_1mm`**（FSL MNI152 1mm 标准 grid，单位 mm，cohort-comparable）
- **不是** subject-native voxel；**是** MNI152 grid 上的 voxel index

**空间判定证据（2026-05-21 跨 27 病人完整验证）**：

| 证据 | 数字 | 解读 |
|---|---|---|
| 全部 27 MRI shape | (182, 218, 182, 1) | FSL MNI152 1mm 标准 shape |
| 全部 voxel size | 1.0 × 1.0 × 1.0 mm | MNI152 1mm 标准 |
| 全部 affine | `[[-1,0,0,90],[0,1,0,-126],[0,0,1,-72],[0,0,0,1]]` | FSL MNI152 1mm 标准 affine |
| Orientation | LAS | MNI 标准 |
| 27 个 .img md5 | 全部不同 | **不是**共享 template — 每个 subject 自己的 brain 被 warp / reslice 到 MNI152 grid |
| brain mask 质心 SD（27 病人）| [0.42, 2.23, 2.47] mm | < 2.5 mm 全方向 — 跨 subject 高度对齐 |
| 12-sample 全卷相关中位数 | 0.911 | 跨 subject 体素值相关高 |
| legacy `plotting_fig4_extractAllElecs.py` | `xyz = aff_matrix @ [coord_x, coord_y, coord_z, 1]` | 把 SQL coord 当 voxel index，乘 MRI affine 转 mm |

**转换公式**（对所有 27 病人都一样）：

```
x_mm = 90 − coord_x
y_mm = coord_y − 126
z_mm = coord_z − 72

例: SQL (131, 139, 119) → MNI mm (−41, 13, 47)
```

**A vs B 判定**（normalization 程度）：

- **A**：完整空间正规化到 MNI152（linear + nonlinear warp）→ cohort-comparable MNI
- **B**：只重采样到 MNI grid 维度，不可 cohort-comparable

**当前判断：A 更可能，置信度中高**：
- 相同 affine + shape + brain mask 质心 SD < 2.5 mm + 全卷相关 0.91 → A
- 但**未找到** transform log / warp field 文件 → 不能 100% 证明 nonlinear warp
- **honest tag**: `normalization_certainty = "grid_confirmed_warp_type_unverified"`
- 对 SEF-ITP **subject-level** H1/H2 几何分析**足够**；paper-level "cohort anatomical MNI localization" 声明**还需找原始 processing 文档**

**Phase 1 主分析适用性（v3 修订）**：

- ✅ Epilepsiae 可作 H1 三层 / H2 主分析的**主坐标**（不再只是 sensitivity）
- ✅ 跨 subject cohort summary 可作
- ⚠️ "cohort 空间图 / endpoint 集中在 MNI 某脑区" 这种 cohort-pooled brain map 声明需带 normalization_certainty caveat
- ⚠️ Yuquan 仍 subject-native T1 RAS mm，**不能**和 Epilepsiae pool 做点云

**Coord loader 实现**：`src/seeg_coord_loader.py` v3 默认对 Epilepsiae 自动发现 MRI、断言 affine 匹配 MNI152、应用 → 输出 `coord_space="mni152_1mm"`。若 subject MRI 的 affine 不匹配 canonical MNI152 → **raise**（防止 cohort 假设静默 violated）。

**当前覆盖**：27 病人 MRI 全在挂载里；stable_k=2 子集 18 个 SQL 全有，**14 个当前通道可全量映射 + MNI 转换可用**

#### 3.2.3 合并覆盖（**v3 修订 2026-05-21**：Epilepsiae 入主分析）

- 当前 stable_k=2 cohort 约 22/34 有完整通道坐标覆盖（Yuquan 8 + Epilepsiae 14）
- **v3 修订**：Epilepsiae 那 14 通过 coord loader 自动应用 MNI152 affine 后输出 `coord_space="mni152_1mm"`（mm 单位）→ **可直接进 H1/H2 主分析**；不再是 "sensitivity only"
- 两 cohort 坐标空间不同：
  - **Yuquan**：subject-native T1 RAS mm
  - **Epilepsiae**：MNI152 1mm（subject 自己的 brain warp 到 MNI grid）
- 因此 SEF-ITP H1/H2 **subject-level**统计 + cohort-aggregate **两边都可作**；但 **cohort-pooled 点云 / brain map** 暂不可作（Yuquan 没有 MNI 配准）
- 部分 subject 仅缺少数通道坐标 —— 如果端点没落在缺失通道上仍可用
- **重要修订（用户审阅 2026-05-21）**：原 v1 描述说"shaft_ordinal 可顶 H1 描述层用"——**错**。shaft_ordinal 距离 = 同一根电极杆内的 contact 编号差，**跨杆 = inf 并被过滤**。这**不等价**"绝对 3D 距离 / 直径"。H1 描述层的"绝对距离 / 直径 / 回转半径"语义**必须有 3D 坐标**。没有 coord loader 时：
  - H1 三层全部 gated（描述 / 严格 / envelope 都需要 3D）
  - H6 仍可用 `distance_metric="shaft_ordinal"` 跑（H6 测的是 participation rate 的空间组织，shaft-ordinal 在 Moran's I 权重里把跨杆 inf 自动当 0，可接受）
  - H2 spatial reversal 也 gated（需 3D 坐标算 centroid 距离）

#### 3.2.4 Coord loader 实现现状（v3.1，2026-05-21 落地）

> 本节是**当前已实现的 loader** 的 schema + 严格 invariants 文档。v1 / v2 计划性内容已合并进 v3.1，**不**保留矛盾的旧合同。
>
> **实现位置**：`src/seeg_coord_loader.py` + `tests/test_seeg_coord_loader.py` —— **49 unit tests + 27-subject real-data smoke 全 GREEN**

**实际工程量**：v3 + v3.1 合计 **< 1 天**（已完成 2026-05-21）。

**输入 / 输出合同**（v3.1，单一权威版本）：

```python
# 模块: src/seeg_coord_loader.py
# 函数: load_subject_coords(dataset, subject_id, channel_names_requested,
#                            mri_affine=None,
#                            epilepsiae_mri_search_roots=None,
#                            allow_voxel_fallback=False) -> CoordResult

CoordResult schema (v3.1 — 单一权威版本):
{
  "schema_version": "coord_loader_v3",
  "dataset": "yuquan" | "epilepsiae",
  "subject_id": str,                                    # canonical form for Epilepsiae
  "channel_names_requested": List[str],                 # 输入原样回放 — 顺序锚点
  "coords_array_in_requested_order": np.ndarray,        # (n_requested, 3); NaN if missing
  "mapped_mask_in_requested_order": np.ndarray,         # (n_requested,) bool
  "coord_space": "fs_native_ras_mm" |                   # Yuquan
                 "mni152_1mm" |                         # Epilepsiae auto-discovery (默认)
                 "ras_mm_via_affine" |                  # Epilepsiae 显式 mri_affine override
                 "mri_native_voxel_ijk",                # Epilepsiae voxel sensitivity (opt-in)
  "coord_units": "mm" | "voxel",
  "source_coord_type": "direct_ras_mm" | "sql_voxel_ijk",
  "normalization_certainty": "subject_native" |
                             "grid_confirmed_warp_type_unverified" |
                             "external_affine_provided" |
                             "subject_native_voxel_no_affine",
  "provenance": {
    "source_path": str,                                  # 绝对路径 SQL / chnXyzDict.npy
    "affine_path": Optional[str],                        # MRI .img path for mni152_1mm
    "loader_version": "coord_loader_v3_2026-05-21",
    "canonical_subject_id": str,                         # Epilepsiae only (v3.1)
    "input_subject_id": str,                             # what caller passed
  },
  "missing": [
    {"channel": str,
     "reason": "sql_null" | "name_not_found" |
               "bipolar_partial_endpoint" |
               "commentary_<text>",
     "index_in_requested": int,
     "commentary": Optional[str]},
    ...
  ],
  "bipolar_resolution": {                                # 仅 bipolar 通道
    "<bipolar_name>": {
      "left_endpoint": str, "right_endpoint": str,
      "left_coord": Optional[Tuple[float, float, float]],
      "right_coord": Optional[Tuple[float, float, float]],
      "midpoint_strategy": "mean_both_required"
    },
    ...
  }
}
```

**严格 invariants（v3.1 lock，每条违反 = 静默科学污染）**：

1. **NULL 坐标显式标 missing**：列入 `missing[]` + reason；该行 NaN + mask=False。**禁止** 0 / centroid / 邻居平均填
2. **Bipolar midpoint**：`HLA1-HLA2` = `mean(HLA1, HLA2)`；**任一端缺 → 整对 missing**
3. **通道名匹配三态**：found / not_found / `bipolar_partial_endpoint`
4. **Provenance**：绝对路径 + loader 版本号 + canonical_subject_id + input_subject_id
5. **顺序锚点**：output array 严格按 `channel_names_requested` index 对齐；任何 sort / dict→array 必须以输入顺序为唯一来源。**违反 = 端点距离静默错误**
6. **空间合同 lock**：
   - **Yuquan**：`coord_space="fs_native_ras_mm"`, `coord_units="mm"`
   - **Epilepsiae 默认（auto-discover MRI）**：`coord_space="mni152_1mm"`, `coord_units="mm"`（**v3 升级；旧 v2 误称 voxel 默认**）
   - **Epilepsiae 显式 mri_affine override**：`coord_space="ras_mm_via_affine"`, `coord_units="mm"`
   - **Epilepsiae voxel 模式（allow_voxel_fallback=True opt-in）**：`coord_space="mri_native_voxel_ijk"`, `coord_units="voxel"`
   - **禁止**把 voxel 当 mm 用；**禁止**默默跨 subject 配准
7. **不跨 subject 配准** —— 每 subject 在自己的影像空间
8. **v3.1 — Epilepsiae subject_id 必须 canonicalize**：
   - 规则：`<id>02` if not ending in `02`；else as-is（例：`'115' → '11502'`, `'1150' → '115002'`, `'108402' → '108402'`）
   - SQL glob **精确** `pat_<canonical>_*.sql`；0 或 >1 → raise
   - MRI 搜 **精确** `pat_<canonical>/adm_*/MRI/mri_*.img`
   - SQL 与 MRI 必须同一 `pat_<canonical>` provenance；mismatch → raise
   - **禁止** fuzzy glob `*<id>*.sql`（曾导致 `'115'` 静默匹配 pat_115002 的 cross-patient 污染）
9. **v3.1 — Epilepsiae MRI affine 必须 byte-精确匹配 MNI152 1mm + shape (182,218,182)**；不匹配 → raise（cohort 假设 broken）
10. **v3.1 — 默认 MRI miss → raise**（不再静默降级到 voxel）；显式 `allow_voxel_fallback=True` 才进 voxel sensitivity

**Phase 1 消费侧合同（强制断言）**：

- `compute_h1_compactness` / `compute_h1_descriptive` / `compute_h2_spatial_reversal` 接 `coords` 时**必须**调用 `assert_coord_result_is_mm_for_main_analysis()` 断言 `coord_units == "mm"`
- **Yuquan + Epilepsiae 都进主分析**（v3 起 Epilepsiae 默认 mni152_1mm = mm）
- Epilepsiae voxel 模式（allow_voxel_fallback=True）**只能作 sensitivity**，下游断言会 raise

**当前 cohort 覆盖**：

- Yuquan: 8 / 16 stable_k=2 全量映射到 fs_native_ras_mm
- Epilepsiae: 14 / 18 stable_k=2 全量映射到 mni152_1mm（27 个 subject MRI 全有 + affine 全一致 + canonical ID 防 cross-patient 污染）
- **SEF-ITP Phase 1 H1/H2 主分析可启动 cohort ≈ 22 subject**

**仍待办（不阻塞 H1/H2 启动）**：

- 找原始 Epilepsiae processing 文档 / warp field → normalization_certainty 从 `grid_confirmed_warp_type_unverified` 升级到 `mni_normalized_verified`（subject-level 几何分析不阻塞；paper-level cohort MNI brain map 声明时 nice-to-have）
- `load_subject_for_phase1()` integration PR（把 loader 接进 SEF-ITP Phase 1 runner）

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
| 5d.2.2 | PR-4B Step 23 (lag span + L3 Pearson) | 同上 | **已完成 2026-05-21** ⚠ **原版唯一显著 finding (L3 dom_r>0.7 子集 Pearson r delta, p=0.016, 7/8) 在修过版降到 p=0.547, 5/8**——"显著 → NULL"翻转，待 Checkpoint B 正式判定 |
| 5d.3 | PR-4C 发作邻近 cohort | 同上 | **已完成 2026-05-21**（n=30 usable subjects；5 propagation 指标 × 3 windows 全 NULL，与原版"5 indicators all NULL" 历史封板同向；rate_by_template 终点由 5e PR-5-B 重做不在本 step 范围）；详见 `step5d3_pr4c_results_2026-05-21.md` |
| 5d.4 | PR-4D 模板速率分解 | 同上 | ⛔ **SKIPPED**（user 2026-05-21 决定优先 5e/5f/5g/5h，留待 5i 收口 phase 判定是否补） |
| **Checkpoint B** | advisor consult: PR-4B/D 方向 + L3 显著消失 reconcile | — | **VERDICT 2026-05-21: trigger met (soft form), 不影响主结论, broad re-derivation 继续**。仅 1 条 secondary tier "显著 → NULL" 翻转 = PR-4B L3 高置信 (n=8) Pearson r delta 同方向 (p 0.016 → 0.547)，原版 archive 已标 exploratory，归 fragility-on-small-n；L1 sign flip 但 NS 两侧；其他 NULL stays NULL。5i 收口仅需在 Topic 1 §3.1c 把 L3 高置信 finding 措辞从"探索性显著"改为"原版探索性显著，masked 后不复现，归 fragility-on-small-n"；不动 cohort-level 主结论。详见 `checkpoint_b_report_2026-05-21.md` |
| 5e | PR-5 / PR-5-B on masked（核心：post-ictal +65.46 events/h, p=0.00128） | `results/interictal_propagation_masked/template_share_switching/` | **已完成 2026-05-21**（3 个 aux script 加 `--masked-features` + filename trap fix；3 项新 TDD path-routing PASS；既有 54/54 PASS 无回归）：**plan §4 pre-registered primary 主结论保持**——PR-5-A gate overall_pass=True (6/6 axis × 2 configs, mask cohort main 27/aux 26)；PR-5-B retained `dominant_rate.candidate_a_global.post_minus_baseline = +65.66 events/h, Wilcoxon p=0.0004, sign 21+/6−, bonferroni PASS`（orig +65.46/p=0.0013/19+/4−，方向 100% 保持 magnitude 几乎相同）。**3 条 plan §4.5 + PR-7 §17 pre-registered secondary diagnostic 翻转**（intersection like-for-like 复核确认非 cohort 替换造成；plan §4.5 明确"share 不进主 Bonferroni 池"）：(a-b) fig_a `composition_diagnostic.share post−base` retained + extended 双 cohort 同向显著→NULL（p=0.015→0.86 / p=0.006→0.82；median +0.016→+0.002 magnitude collapse；原版本身已是 panel d 反方向 direction_consistent 6/23，本身不算 cohort claim）；(c) fig_b `transition lift post−base` NULL→Wilcoxon-only borderline sig（p=0.29→0.022, sign p=0.076 不达 cohort 门槛，反方向翻转）。**Checkpoint B 不重新打开**（CkB 报告 §3.3 "5e 后单独评估"的触发条件指向 PR-5-B 主信号翻转，本档主信号未翻）。**论文 fig_a 叙事需调整**（5i 收口时执行）：从"dominant share 抬升"改为"dominant 绝对率抬升、share 维持"；fig_b 从"transition 不变"改为"transition Wilcoxon-only borderline 抬升 + sign-test 未达 cohort 门槛"。不动 PR-5 plan §4 核心 cohort claim。详见 `step5e_pr5_results_2026-05-21.md` |
| 5f | PR-6 endpoint anchoring / swap / Step 6 / rank displacement on masked | `results/interictal_propagation_masked/{template_anchoring,pr6_step6_held_out_template,rank_displacement}/` | **已完成 2026-05-21（user 优先 5f over Checkpoint B；后者仍待 5d.3 + 5e 完成正式触发）**（5 项 TDD + 3 个 runner 加 `--masked-features`；92/92 既有测试无回归）：所有 primary 方向保持——H1 pooled NULL 不变（n=28, p 0.22→0.18）；H2 fwd/rev swap 节点级 9/9 → 8/8 都 positive（mask sign p=0.0078, 因 −1 subject magnitude 略缩）；H3 全 NULL；Step 6 held-out swap_class concordance **0.69 → 0.82 实质提升**；rank displacement F_norm 0.800 → 0.789，Spearman ρ(F_norm,τ) −0.916 → −0.921 几乎不变。**1 条 secondary cohort Wilcoxon borderline 翻转**：node anatomy h1_eligible swap−same p=0.014 → 0.059（同 n=28；方向保持，sign-test 在 orig 上原本就 NULL，论文叙事需进一步弱化 cohort-wide swap-leaning claim）。详见 `step5f_pr6_results_2026-05-21.md` |
| 5g | PR-7 antagonistic pairing on masked | `results/interictal_propagation_masked/pr7_*/` | 待启动（先 patch runner；~1-2h 并行）—— **PR-7 P3 翻转 = framework-level revision** |
| 5h | Topic 4 attractor Step 1 on masked | `results/topic4_attractor_masked/` | **已完成 2026-05-21**（10 项 TDD + 5 个 script 加 `--masked-features`+ `src/topic4_attractor_diagnostics.py:build_rank_feature_matrix` 加 `mask_phantom` 参数；92/92 既有测试无回归）：cohort 34（orig 35，少 916 因 5a stable_k 翻 4）；GOF pass 持平 33/34（fail subject 从 916 换成 1077 var 0.658→0.515）；**coordinate-free λ₂（H3 主直测）显著加强**：orig 10/34 → mask **13/34**，5 个原 NULL/borderline 翻到 p≤0.001（pengzihang/253/442/958，加 1096 原 drift 排除现进 cohort），2 个失去显著都无害（916 cohort exit + liyouran p=0.001→0.002 borderline，λ₂ 实际值几乎不动）。Principal curve var_curve 中位 0.953 → 0.729（phantom 在 PCA-3 子空间贡献的假结构被去掉的预期方向，GOF 闸门 0.6 仍稳）。**没有任何 H3 primary 方向反转**。详见 `step5h_topic4_attractor_results_2026-05-21.md` |
| 5i | 主文档更新 + AGENTS.md cross-PR 合同更新 | `docs/topic1_*.md` / `docs/topic4_sef_itp_framework.md` cohort tier 数字 lock | 待启动（半天） |

**Phase 0 整体进度（2026-05-21 15:30 更新）**：13 / 14 main step 完成（5a/5b/5c/5d.1/5d.2.0/5d.2.1/5d.2.2/5d.3 + Checkpoint A + **5e** + 5f + Checkpoint B + **5h**）；剩 5g（user 并行 session 处理）+ 5i 收口。**所有 plan-pre-registered primary cohort verdict 方向保持，无 framework-level 翻转**。已记录的 secondary tier 显著性变化：(1) 5d.2.2 PR-4B L3 高置信 (n=8) Pearson r 从 p=0.016 (7/8) 降到 p=0.547 (5/8) — 同方向 + fragility-on-small-n，原版 archive 已标 exploratory tier; (2) 5f node anatomy h1_eligible swap−same Wilcoxon p=0.014 → 0.059 — 同方向 borderline 弱化，secondary tier; (3) 5f Step 6 held-out swap_class concordance 0.69 → 0.82 — 实质提升; (4) 5c PR-4 panel d identity bias fraction 87.9 → 92.2% — 加强; (5) 5h Topic 4 coordinate-free λ₂ 显著 +3 (10/34 → 13/34) — 加强（H3 主直测）; (6) **5e fig_a `composition_diagnostic.share post−base` (PR-5 plan §4.5 pre-registered secondary, retained + extended 双 cohort 同向同翻) 显著→NULL (p=0.015→0.86 retained, p=0.006→0.82 extended; intersection 复核同向)** — secondary tier, 论文 fig_a 叙事需调整；(7) **5e fig_b `transition lift post−base` (PR-7 §17 补丁 secondary diagnostic) NULL→Wilcoxon-only borderline sig (p=0.29→0.022, sign p=0.076 不达 cohort 门槛)** — secondary tier 反方向翻转；其他 step 方向一致或 NULL stays NULL。**Checkpoint B 不重新打开**：CkB 报告 §3.3 触发条件"5e 后单独评估"指向 PR-5-B 主信号翻转，本档 primary 主信号未翻。N=10 并行工具：`scripts/run_pr_parallel.py` + `scripts/consolidate_pr_cohort_masked.py`（commit `45eb221`）。详见进度报告 `docs/archive/topic0/lagpat_phantom_rank/phase0_progress_report_2026-05-21.md` + Checkpoint B 报告 `checkpoint_b_report_2026-05-21.md` + 5e 结果 `step5e_pr5_results_2026-05-21.md` + 5h 结果 `step5h_topic4_attractor_results_2026-05-21.md`。

每个 Step 跑完，更新 §3.1 的 "受影响 PR" 表，把对应行从"待重跑"挪到"重跑完，方向 = 一致 / 翻转 / NULL flip"。

**Checkpoint A / B 是 hard stops**：未通过 advisor consult 不进入下一阶段。

**PR-7 P3 翻转特别 flag**：如果 §5g 把 P3 从 INCONCLUSIVE 推到 PASS 或 FAIL，这是 framework-level revision，必须先停下来发起 `docs/paper1_framework_sba.md` 修订 review，**不允许**默默改写。

---

## 6. 文件清单

### 代码

- `src/lagpat_rank_audit.py` — masked re-rank helpers
- 修改：`src/interictal_propagation.py` — 4 个 KMeans 调用点加 `use_masked_features` 参数（待启动）
- 修改：`src/cluster_geometry.py:compute_pca_embedding` — 同参数（待启动）
- 修改：`src/topic4_attractor_diagnostics.py:build_rank_feature_matrix` — `mask_phantom` 参数（待启动）

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
- `docs/archive/topic0/lagpat_phantom_rank/step5g_pr7_results_2026-05-21.md` — Step 5g PR-7 antagonistic temporal pairing 修过版结果（agent v1 下游 cohort 段 fabricated；in-flight 重跑 per-subject 28/30，剩 2 大 subject 待完；P3 verdict 待出）
- `docs/archive/topic0/lagpat_phantom_rank/rerun_results_2026-05-21.md` — **Phase 0 终稿**（5a-5h 整合 + Cross-PR reconcile 表 + 论文叙事调整清单 + 5i 收口剩余任务）
- `docs/archive/topic0/lagpat_phantom_rank/checkpoint_b_report_2026-05-21.md` — Checkpoint B advisor consult report (5d 完成后)
- `docs/archive/topic0/lagpat_phantom_rank/phase0_progress_report_2026-05-21.md` — Phase 0 进度跟踪报告
- `docs/archive/topic1/propagation/lagpatrank_phantom_audit_diagnostic_2026-05-20.md` — **MOVED 2026-05-20** 到 `docs/archive/topic0/lagpat_phantom_rank/diagnostic_2026-05-20.md`（原 path 仅留 redirect stub）
