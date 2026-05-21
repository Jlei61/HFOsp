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

- **未结清问题 1**：`lagPatRank` phantom pseudo-rank（2026-05-20 确诊，broad re-derivation 进行中；§3.1 + §5）
- **未结清问题 2**：SEEG 3D coord loader 缺失（2026-05-21 修订；§3.2）—— **数据源已确认在 `/mnt/yuquan_data/yuquan_images/...` + Epilepsiae SQL `electrode` 表**；当前 stable_k=2 cohort 约 22/34 已有完整通道覆盖；缺的是 loader 代码，不是数据本身（小焦点 PR 1–2 周）
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

**受影响 PR**：PR-2 / PR-2.5 / PR-3 / PR-4A / PR-4B / PR-4C / PR-4D / PR-5 / PR-5-B / PR-6 H1 / PR-6 H2 / PR-6 Step 6 / PR-6 supplementary rank displacement / PR-7 / Topic 4 attractor Step 1 / cluster_geometry PCA embedding。

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

#### 3.2.2 Epilepsiae 坐标源（已确认 + 2026-05-21 坐标空间已查清）

- **位置**：`/mnt/epilepsia_data/all_data_sqls/pat_*.sql`
- **表**：`electrode`，字段 `id, array, name, moniker, focus_rel, invasive, supplier, coord_x, coord_y, coord_z, commentary`
- **坐标空间**：**`mri_native_voxel_ijk`**（subject-native MRI voxel index，单位 voxel）—— **不是 mm**，**不是 MNI**
- **空间判定证据**（用户 2026-05-21 实测）：
  - SQL 坐标全是整数 `.0`，范围 x≈12–161 / y≈19–200 / z≈14–151 → MRI voxel index 形态
  - legacy readme 明写 "电极坐标在 ijk 空间"
  - legacy 脚本用 `aff_matrix @ [coord_x, coord_y, coord_z, 1]` 转 native mm
  - NULL 通道带 `Mikrokontakt / nicht abgrenzbar` 等定位失败注释
- **Phase 1 主分析适用性**：
  - **原始 voxel 坐标**：仅可作 sensitivity，**不能**当 3D Euclidean 主距离（不同 subject 的 voxel 物理 size 不同，距离比较无物理意义；voxel 各方向 size 也不一定相等）
  - **要进主分析**：必须先拿到 per-subject MRI affine，把 voxel → ras_mm 转完才行
  - **如果 MRI affine 拿不到**：Epilepsiae cohort 在 H1 严格层 / H2 主分析里**只能作 sensitivity**，不进主结论
- **数据样式（示例 pat_108402）**：
  ```sql
  INSERT INTO electrode (..., name, ..., coord_x, coord_y, coord_z, commentary) VALUES
    (..., 'GC4', ..., 131.0, 139.0, 119.0, NULL);
    (..., 'HLA5', ..., NULL, NULL, NULL, 'nicht abgrenzbar!');
  ```
- **当前覆盖**：20 个 Epilepsiae subject 全有 SQL；stable_k=2 子集 18 个里 **14 个当前通道可全量映射**

#### 3.2.3 合并覆盖（重要修订 2026-05-21）

- 当前 stable_k=2 cohort 约 22/34 有完整通道坐标覆盖（Yuquan 8 + Epilepsiae 14）
- **但是**：Epilepsiae 那 14 是 **voxel 形态**，进 H1 严格层 / H2 主分析前**还需 MRI affine 转 mm**；如果 affine 不可得，Epilepsiae cohort 在主分析里只是 sensitivity
- 部分 subject 仅缺少数通道坐标 —— 如果端点没落在缺失通道上仍可用
- **重要修订（用户审阅 2026-05-21）**：原 v1 描述说"shaft_ordinal 可顶 H1 描述层用"——**错**。shaft_ordinal 距离 = 同一根电极杆内的 contact 编号差，**跨杆 = inf 并被过滤**。这**不等价**"绝对 3D 距离 / 直径"。H1 描述层的"绝对距离 / 直径 / 回转半径"语义**必须有 3D 坐标**。没有 coord loader 时：
  - H1 三层全部 gated（描述 / 严格 / envelope 都需要 3D）
  - H6 仍可用 `distance_metric="shaft_ordinal"` 跑（H6 测的是 participation rate 的空间组织，shaft-ordinal 在 Moran's I 权重里把跨杆 inf 自动当 0，可接受）
  - H2 spatial reversal 也 gated（需 3D 坐标算 centroid 距离）

#### 3.2.4 Coord loader PR 设计合同（v2 严格化 2026-05-21）

**真实工程量**：1–2 周小焦点 PR，不是数月 SEEG re-localization。

**输入 / 输出合同**（v2 必须强制顺序对齐 + 显式空间标识）：

```python
# 模块: src/seeg_coord_loader.py
# 函数: load_subject_coords(dataset, subject_id, channel_names_requested,
#                            mri_affine=None) -> CoordResult

CoordResult schema (v2 strict):
{
  "schema_version": "coord_loader_v2",
  "dataset": "yuquan" | "epilepsiae",
  "subject_id": str,
  "channel_names_requested": List[str],                 # 输入原样回放 — 顺序锚点
  "coords_array_in_requested_order": np.ndarray,        # shape (n_requested, 3);
                                                        # 未映射通道行 = [NaN, NaN, NaN]
  "mapped_mask_in_requested_order": np.ndarray,         # shape (n_requested,) bool
  "coord_space": "fs_native_ras_mm" |
                 "mri_native_voxel_ijk" |
                 "ras_mm_via_affine",                   # affine 转换后的 Epilepsiae
  "coord_units": "mm" | "voxel",
  "provenance": {
    "source_path": str,                                  # 绝对路径
    "affine_path": Optional[str],                        # voxel→mm 转换时填
    "loader_version": str,
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

**严格 invariants（v2 lock，每条违反 = 静默科学污染）**：

1. **NULL 坐标显式标 missing**：必须列入 `missing[]` + 给 reason；`coords_array_in_requested_order` 该行用 `[NaN, NaN, NaN]` 占位但 `mapped_mask_in_requested_order` 标 False。**禁止**用 0 / shaft centroid / 邻居平均填
2. **Bipolar midpoint**：`HLA1-HLA2` 的坐标 = `mean(monopolar HLA1, monopolar HLA2)`；**任一端 NULL / 缺失 → 整对 missing**，**不允许**"只用一端"
3. **通道名匹配三态**：found / not_found / `bipolar_partial_endpoint`
4. **Provenance**：绝对路径 + loader 版本号
5. **顺序锚点**：`coords_array_in_requested_order` 必须与 `channel_names_requested` 严格按 index 对齐；任何 sort / dict→array 转换必须以输入顺序为唯一来源。**这条 invariant 失败 = 端点距离静默错误**，是最大科学污染口
6. **空间合同 lock**：
   - Yuquan：必须输出 `coord_space="fs_native_ras_mm"`, `coord_units="mm"`
   - Epilepsiae 默认（无 affine）：`coord_space="mri_native_voxel_ijk"`, `coord_units="voxel"`
   - Epilepsiae 传入 affine：`coord_space="ras_mm_via_affine"`, `coord_units="mm"`，且 `provenance.affine_path` 必填
   - **禁止**把 voxel 当 mm 用；**禁止**默默跨 subject 配准
7. **不跨 subject 配准 / 标准化** —— 每 subject 在自己的影像空间里

**Phase 1 消费侧合同（强制断言）**：

- `compute_h1_compactness` / `compute_h1_descriptive` / `compute_h2_spatial_reversal` 接受 `coords` 时**必须断言** `coord_units == "mm"`
- Yuquan 默认进主分析（fs_native_ras_mm = mm）
- Epilepsiae 默认进 sensitivity；只有 `coord_space == "ras_mm_via_affine"` 才进主分析
- 接受 voxel 坐标的代码必须 raise，**不**允许"假装 voxel 是 mm"

**待补查（不阻塞 loader 主 PR，可后续）**：

- Epilepsiae per-subject MRI affine 文件是否可得（SQL `files` 表已知有 `mri_*.img/.hdr`；是否可推 affine 需核实）
- Epilepsiae voxel size：各 subject 是否一致；任何一致性假设都需显式核实，**不应**默认 1×1×1 mm

**新硬前置正确版**：

- SEF-ITP **H1 三层（描述 + 严格 + envelope）+ H2 spatial reversal** 在 coord loader v2 完成前**不启动**
- **H6 with `distance_metric="shaft_ordinal"`**：真正不需 coord loader 的一项；shaft-ordinal 在 Moran's I 权重里把跨杆 inf 自动当 0，可接受为 H6 主分析的合法形态

**Framework 文档级修订建议**：SEF-ITP framework v1.0.2 §3.1 H1 应在用户回归后加 caveat："3D 距离需 coord loader v2 (数据源 §3.2 已确认；Yuquan 直接可用，Epilepsiae 需 affine 转换才进主分析)；shaft-ordinal **不**等价绝对距离，**不**能作 H1 描述层 fallback"。这是 plan-level 操作细节修订，不升 framework erratum。

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
| 5e | PR-5 / PR-5-B on masked（核心：post-ictal +65.46 events/h, p=0.00128） | `results/interictal_propagation_masked/template_share_switching/` | 🟡 **即将 N=10 并行启动**（PR-5-A gate → consolidate → PR-5-B 串行；ETA ~1h） |
| 5f | PR-6 endpoint anchoring / swap / Step 6 / rank displacement on masked | `results/interictal_propagation_masked/{template_anchoring,pr6_step6_held_out_template,rank_displacement}/` | **已完成 2026-05-21（user 优先 5f over Checkpoint B；后者仍待 5d.3 + 5e 完成正式触发）**（5 项 TDD + 3 个 runner 加 `--masked-features`；92/92 既有测试无回归）：所有 primary 方向保持——H1 pooled NULL 不变（n=28, p 0.22→0.18）；H2 fwd/rev swap 节点级 9/9 → 8/8 都 positive（mask sign p=0.0078, 因 −1 subject magnitude 略缩）；H3 全 NULL；Step 6 held-out swap_class concordance **0.69 → 0.82 实质提升**；rank displacement F_norm 0.800 → 0.789，Spearman ρ(F_norm,τ) −0.916 → −0.921 几乎不变。**1 条 secondary cohort Wilcoxon borderline 翻转**：node anatomy h1_eligible swap−same p=0.014 → 0.059（同 n=28；方向保持，sign-test 在 orig 上原本就 NULL，论文叙事需进一步弱化 cohort-wide swap-leaning claim）。详见 `step5f_pr6_results_2026-05-21.md` |
| 5g | PR-7 antagonistic pairing on masked | `results/interictal_propagation_masked/pr7_*/` | 待启动（先 patch runner；~1-2h 并行）—— **PR-7 P3 翻转 = framework-level revision** |
| 5h | Topic 4 attractor Step 1 on masked | `results/topic4_attractor_masked/` | 待启动（先 patch runner 加 `mask_phantom` 参数；~30min 并行） |
| 5i | 主文档更新 + AGENTS.md cross-PR 合同更新 | `docs/topic1_*.md` / `docs/topic4_sef_itp_framework.md` cohort tier 数字 lock | 待启动（半天） |

**Phase 0 整体进度（2026-05-21 14:00 更新）**：11 / 14 main step 完成（5a/5b/5c/5d.1/5d.2.0/5d.2.1/5d.2.2/5d.3 + Checkpoint A + 5f + Checkpoint B）；剩 5e / 5g / 5h（user 并行 session 处理）+ 5i 收口。**所有 primary cohort verdict 方向保持，无主结论翻转**。已记录的 secondary tier 显著性变化：(1) 5d.2.2 PR-4B L3 高置信 (n=8) Pearson r 从 p=0.016 (7/8) 降到 p=0.547 (5/8) — 同方向 + fragility-on-small-n，原版 archive 已标 exploratory tier; (2) 5f node anatomy h1_eligible swap−same Wilcoxon p=0.014 → 0.059 — 同方向 borderline 弱化，secondary tier; (3) 5f Step 6 held-out swap_class concordance 0.69 → 0.82 — 实质提升; (4) 5c PR-4 panel d identity bias fraction 87.9 → 92.2% — 加强; 其他 step 方向一致或 NULL stays NULL。N=10 并行工具：`scripts/run_pr_parallel.py` + `scripts/consolidate_pr_cohort_masked.py`（commit `45eb221`）。详见进度报告 `docs/archive/topic0/lagpat_phantom_rank/phase0_progress_report_2026-05-21.md` + Checkpoint B 报告 `checkpoint_b_report_2026-05-21.md`。

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
- `docs/archive/topic0/lagpat_phantom_rank/step5f_pr6_results_2026-05-21.md` — Step 5f PR-6 全套（template_anchoring + Step 6 held-out + rank displacement）修过版结果
- `docs/archive/topic0/lagpat_phantom_rank/phase0_progress_report_2026-05-21.md` — Phase 0 进度跟踪报告
- `docs/archive/topic1/propagation/lagpatrank_phantom_audit_diagnostic_2026-05-20.md` — **MOVED 2026-05-20** 到 `docs/archive/topic0/lagpat_phantom_rank/diagnostic_2026-05-20.md`（原 path 仅留 redirect stub）
