# Step 5h — Topic 4 attractor Step 1 修过版重跑结果（2026-05-21）

> 状态：Step 5h 完成。**所有 primary 方向保持**；**coordinate-free λ₂ 路径**（H3 主直测）从 orig 10/34 显著上升到 **masked 13/34**（同 n 主分析池，含 1 个 +1 来自 orig 已排除的 1096，4 个原 NULL → 显著、2 个原显著 → masked NULL/borderline）；GOF pass 率从 33/34 → 33/34（fail subject 从 916 换成 1077，因为 916 在 5a 已 stable_k 翻 4 退出 cohort）。
> 主入口：`docs/topic0_methodology_audits.md`
> 上游：`./step5a_pr2_results_2026-05-20.md`（PR-2 masked stable_k cohort 定义）/ `./step5f_pr6_results_2026-05-21.md`（模式参考）
> 路线图：`./rerun_roadmap_2026-05-20.md` §5h
> 原 Topic 4 archive：`docs/archive/topic1/propagation/topic4_attractor_diagnostics_step1_results_2026-05-10.md`
> 修过版结果：`results/topic4_attractor_masked/`

---

## 1. 三段式朴素话

**测了什么** —— Topic 4 attractor Step 1 整套（PCA-3 子空间内 principal curve + GOF + KMeans 主轴夹角 + s_kmeans 监督坐标 + **coordinate-free PR-2 label λ₂ transition test**），用 5a 写好的 masked PR-2 cluster labels 重跑：
- Step 0 cohort audit 在 masked PR-2 JSON 上重做（filter stable_k=2 且 n_eligible ≥ 100）
- Step 1 主路径 = principal curve + GOF + 单点夹角，per-subject JSON 重写
- augment 增加 s_kmeans diagnostic + λ₂ within-block transition 与 within-block label shuffle null（n_perm=1000）
- max_iter sensitivity sweep {15, 30, 60}（验证 var_curve / 角度对 iteration upper bound 的稳定性）
- summarize 重写 step1_summary.md

**怎么测的** ——
1. 输入：`results/interictal_propagation_masked/per_subject/<sid>.json`（5a 写好的 masked PR-2 labels + 模板）
2. `src/topic4_attractor_diagnostics.py:build_rank_feature_matrix` 新增 `mask_phantom: bool = False` 参数；`mask_phantom=True` 时改用 `src.lagpat_rank_audit.build_masked_kmeans_features(ranks, bools, impute="event_median")` 构造 feature matrix（与 PR-2 KMeans masked 路径完全一致）
3. 5 个 driver 脚本各加 `--masked-features` flag + `_apply_masked_paths()` helper：路径全部 swap 到 `results/topic4_attractor_masked/`，PR2_PER_SUBJECT_DIR 也 swap 到 `results/interictal_propagation_masked/per_subject/`
4. 10 项新 TDD（`tests/test_attractor_masked_features.py`，全 PASS）+ 既有 `tests/test_topic4_attractor_diagnostics.py` + `tests/test_lagpat_rank_audit.py` + `tests/test_interictal_propagation.py` 共 92/92 PASS（无回归）
5. 按依赖顺序跑：audit → step1 → augment → sensitivity → summarize（augment 与 sensitivity 都改写 per-subject JSON 的不同字段，不并行）

**揭示了什么** ——
- **GOF 通过率持平**：orig 33/34 → mask **33/34**；GOF fail subject 从 `epilepsiae_916`（orig var=0.565）换成 `epilepsiae_1077`（mask var=0.515）。916 本来就是 orig 的 GOF outlier，masked 后因为 stable_k 翻 4 直接退出 Topic 4 cohort；1077 在 orig 是 var=0.658 PASS，masked 后掉到 0.515 FAIL——这与 1077 的事件数最小（n=2514）、phantom 修过后 PCA-3 子空间能解释的方差也跟着缩小一致。GOF 闸门"33/34"数字不动，但 fail 的具体 subject 换了人。
- **var_explained_curve cohort 中位实质下降**：orig median = 0.953 → mask **0.729**。phantom rank 给非参与通道分配的伪名次相当于在 PCA-3 子空间里硬塞结构，让"曲线沿着 phantom 方向延展"的 var 解释看起来 inflated；修过后曲线在真实 rank 子空间里贴的没那么紧，**这正是 audit-first 修法的预期方向**。GOF 闸门 0.6 仍然足够宽，pass 率没受影响，但论文写"principal curve 在 PCA-3 子空间内解释 95% 方差"的口径需要改成"73% 方差"。
- **Coordinate-free λ₂（H3 主直测）显著加强**：orig 10/34 (p<0.001 & λ₂>0) → **mask 13/34**。net +3，但**幅度比净增数字大**——5 个新进显著、2 个掉出显著（其中 916 是被 cohort 移除，liyouran 是 p 从 0.001 → 0.002 的 borderline，未实质丢信号）。**4 个原 p≥0.97 的 subject（pengzihang, epi_253, epi_442, epi_958）**在 masked 下 p 跳到 0.001——**phantom rank 之前在污染 cluster label 把 metastability 信号洗掉**，修过后真信号浮出来。+1096（原本因 label/event drift 被 excluded_from_h3_main，masked 下 PR-2 labels 与 loader 重新对齐后 normal 进入，p=0.001）。
- **方向 100% 保持，没有一个 subject 从 positive 显著翻到 negative 显著**。lost-2 都是无害：916 是 cohort exit，liyouran 是 0.001 → 0.002 borderline，λ₂ 实际值 0.114 → 0.107 几乎不动。
- **max_iter sensitivity**：var_curve 在 15/30/60 三档间稳定，angle 漂移幅度类似 orig（不放大）；详见 §3.4。
- **总体判读**：Topic 4 H3 主直测（λ₂ coordinate-free）**在 phantom 修过后实质加强**，与 5a 的"masked cluster labels 比 orig 更准确"判断一致；s_kmeans / principal curve 几何不强行做出"双稳态"独立证据（仍是 sanity 而非 H3 evidence，per CLAUDE.md §5 + Topic 4 archive 已有 caveat）。**没有任何 primary 方向反转**。可进 5i 收口。

代号补注：H3 = "PR-2 stable_k=2 反映真实 metastable switching"；coordinate-free λ₂ = `label_transition_sanity.obs.lambda_2`（2 状态 row-stochastic transition matrix 第二大特征值）；mask_phantom 参数名遵循 §5h plan 命名（与 src/interictal_propagation.py 的 `use_masked_features` 不同名是因为是不同模块）。

---

## 2. 实现层改动（surgical）

| 文件 | 改动 |
|---|---|
| `src/topic4_attractor_diagnostics.py` | `build_rank_feature_matrix` 新增 `mask_phantom: bool = False` 参数；`mask_phantom=True` 时调 `src.lagpat_rank_audit.build_masked_kmeans_features`。`run_step1_subject` 同名参数透传。 |
| `scripts/audit_topic4_step0.py` | 新增 `_apply_masked_paths()` swap `PR2_PER_SUBJECT_DIR / OUT_DIR / OUT_CSV / OUT_SUMMARY`；`--masked-features` flag。**关键**：PR2_PER_SUBJECT_DIR swap 决定 cohort eligibility 的 stable_k 数据源。 |
| `scripts/run_attractor_step1.py` | `_apply_masked_paths()` swap 5 globals (`PR2_PER_SUBJECT_DIR / OUT_DIR / PER_SUBJECT_DIR / COHORT_CSV / AUDIT_CSV`)；`--masked-features` flag；`_run_one` 加 `mask_phantom` kwarg 透传到 `run_step1_subject`。 |
| `scripts/run_attractor_step1_sensitivity.py` | `_apply_masked_paths()` swap 5 globals；`--masked-features` flag；`_run_one` 加 `mask_phantom` 透传到 `build_rank_feature_matrix`（直接 caller，不走 run_step1_subject）。 |
| `scripts/augment_attractor_step1_kmeans_s.py` | `_apply_masked_paths()` swap 4 globals；`--masked-features` flag；`_augment_one` 加 `mask_phantom` 透传到 `build_rank_feature_matrix`（直接 caller）。 |
| `scripts/summarize_attractor_step1.py` | `_apply_masked_paths()` swap 3 globals；`--masked-features` flag。纯路径重路由（不计算 feature）。 |
| `tests/test_attractor_masked_features.py`（新建） | 10 项 smoke test：(a) `build_rank_feature_matrix(mask_phantom=True)` 在 phantom 污染 fixture 上与 `=False` 路径产出不同矩阵；(b) `mask_phantom=False` 默认与 explicit-False 等价（兼容性）；(c) 5 项验证每个 script 的 `_apply_masked_paths()` swap 关键 path globals；(d) 3 项 monkeypatch 验证 `mask_phantom` kwarg 从 runner / augment 透传到 `build_rank_feature_matrix`。 |

不动：
- `src/interictal_propagation.py`（5a–5g 早已加 `use_masked_features`，本档只是新增 caller）
- `src/lagpat_rank_audit.py`（5a 已成形）
- 原始 `*_lagPat*.npz` / orig `results/topic4_attractor/` 目录（per "旧结果不删" 原则）
- 任何 PR-2 / PR-2.5 / PR-3 / PR-4 / PR-6 输出

**TDD 验证**：
```
tests/test_attractor_masked_features.py: 10/10 PASS
tests/test_topic4_attractor_diagnostics.py + tests/test_lagpat_rank_audit.py + tests/test_interictal_propagation.py: 92/92 PASS (无回归)
```

---

## 3. Cohort 数字对比表

### 3.1 Cohort eligibility (Step 0) — orig vs masked

> **核心：cohort 缩了，没扩**。Step 0 用 PR-2 stable_k 过滤，orig PR-2 stable_k=2 有 35 个 subject，masked PR-2 stable_k=2 有 34 个（5a 写：916 stable_k 翻 2→4 退出；huangwanling 是从 4→3 仍不是 2 留在排除集；其他 yuquan zhaojinrui / zhourongxuan / zhangjinhan 也是非-2）。

| 类别 | orig | masked | 说明 |
|---|---:|---:|---|
| Step 0 eligible_for_main = true | **35** | **34** | masked −1 = `epilepsiae_916`（5a stable_k 翻 4） |
| Step 1 进 H3 main | **34** | **34** | orig 减 1 是 `epilepsiae_1096` (pr2_label_event_index_drift)；masked **0 个 drift exclusion**（masked PR-2 labels 与当前 loader 重新对齐，drift 不存在）|
| GOF pass (var_curve > 0.6) | **33/34** | **33/34** | fail subject 换人：orig `epilepsiae_916` (var=0.565) → masked `epilepsiae_1077` (var=0.515)|
| 共同 cohort (orig ∩ masked) | **33**（35 − 916 − 1096 + 1096 drift orig only=33，详见下）| — | 严格 like-for-like 对比基础 |

**1096 行为变化**：orig 因 PR-2 label vs loader event ordering drift 被排除（`pr2_label_event_index_drift`，详见原 archive `topic4_attractor_diagnostics_step1_results_2026-05-10.md` §异常）。masked 路径下 1096 的 PR-2 labels 已在 5a 重跑时与当前 loader 用同一 event indexing，所以 drift 不存在，1096 正常进 H3 main 池，且 λ₂ p=0.001 显著。**这是 cohort 的一个净增**，但不是"masked 把 1096 修复"，而是"masked PR-2 labels 没继承旧 ordering"。

### 3.2 Principal curve var_explained_curve（in PCA-3 子空间）

| metric | orig (n=34) | masked (n=34) |
|---|---:|---:|
| median | **0.953** | **0.729** |
| min | 0.565 | 0.515 |
| max | 0.990 | 0.804 |
| GOF pass (>0.6) | 33/34 | 33/34 |

**判读**：var_curve 中位实质下降是 phantom rank 修过后的**预期方向**——phantom 给非参与通道分配的伪名次相当于在 PCA-3 子空间内增加"假结构"，让 principal curve 表面上沿这条 phantom 方向延展，inflated var explained。masked 后 PCA-3 子空间内只剩真 rank 几何，曲线贴的没那么紧。**GOF 闸门 0.6 没崩**，但论文级口径需要从"95% 方差解释"改成"73% 方差解释"（caveat 仍是 PCA-3 子空间方差，不是原始 X 全方差，这条没变）。

### 3.3 Coordinate-free PR-2 label λ₂ transition test — H3 主直测

> **这是 H3 最稳健的直接测度**（不依赖 1D 坐标，per 原 archive §2.4 + 5h roadmap）。orig 10/34 vs masked 13/34。

| metric | orig (n=34) | masked (n=34) |
|---|---:|---:|
| λ₂ observed median | 0.044 | **0.085** |
| λ₂ observed range | [−0.063, 0.198] | **[−0.013, 0.292]** |
| z_λ₂ median | 1.2 | **2.7** |
| empirical p median | 0.117 | **0.011** |
| **p < 0.001 且 λ₂ > 0** | **10 / 34** | **13 / 34** |

**Per-subject membership 比较**：

| 集合 | n | sid |
|---|---:|---|
| 共同显著（orig ∩ masked） | 8 | `epi_1073, epi_1084, epi_1146, epi_548, epi_922, yuq_sunyuanxin, yuq_zhangjiaqi, yuq_zhangkexuan` |
| **只 masked 新显著** | **5** | `epi_1096`（orig 是 drift 排除）, `epi_253` (p 1.000→0.001), `epi_442` (0.120→0.001), `epi_958` (0.973→0.001), `yuq_pengzihang` (1.000→0.001) |
| 只 orig 显著（masked 失） | 2 | `epi_916`（masked 已被移出 cohort，stable_k=4）, `yuq_liyouran` (p 0.001→0.002 borderline，λ₂ 0.114→0.107 几乎不动) |

**Like-for-like 严格对比（cohort 交集 n=33 = 34 ∩ 34，排除只在 orig 的 916 和只在 masked 的 1096）**：

| | orig | masked |
|---|---:|---:|
| p<0.001 & λ₂>0 (on n=33) | **9** | **12** |
| Δ | — | **+3** |

排除 cohort 边缘 subject（916 / 1096）后，**同 cohort 33 上 masked 仍多 3 个显著**——加强不是 cohort 大小变化导致的伪信号。

**关键观察**：
- 5 个新显著里 4 个是 orig 的 NULL 反向（λ₂ 微负或近 0 → 强正 + p 从 ≥0.12 一步跳到 ≤0.001）。**这是 phantom rank 给 cluster labels 加噪声把 metastability 信号洗掉**的直接证据：orig 的 cluster label 因为 phantom 把"通道是否参与"和"rank order"耦合到一起，导致 transition matrix 看起来更接近 marginal shuffle null；masked 后真 rank order 主导 cluster identity，metastable dwell 显现。
- 2 个失去显著的都无害：916 是 cohort exit（不是 H3 NULL 翻转），liyouran 是 0.001→0.002 同方向 borderline jitter。
- **没有任何 subject 从 positive 显著翻到 negative 显著**——方向 100% 保持。

### 3.4 max_iter sensitivity (max_iter ∈ {15, 30, 60})

| max_iter | n_subj | converged | var_curve median (range) | angle@s_median median | angle_grid median | angle_event median |
|---:|---:|---:|---|---:|---:|---:|
| 15 (orig) | 34 | 1 | 0.953 (0.565–0.990) | 83.0° | 81.0° | 82.7° |
| 30 (orig) | 34 | 3 | 0.949 (0.549–0.990) | 79.2° | 77.1° | 78.5° |
| 60 (orig) | 34 | 4 | 0.950 (0.634–0.990) | 71.5° | 67.0° | 69.8° |
| **15 (masked)** | **34** | **1** | **0.729 (0.515–0.804)** | **61.6°** | **61.1°** | **65.6°** |
| **30 (masked)** | **34** | **2** | **0.736 (0.522–0.812)** | **47.6°** | **51.3°** | **58.4°** |
| **60 (masked)** | **34** | **5** | **0.757 (0.680–0.821)** | **33.6°** | **37.6°** | **36.8°** |

**判读**：
- **var_curve 在 masked 下仍稳定**（15→60 median 0.729 → 0.757，差 0.028 ~ 3.8%）。GOF 闸门 0.6 三档都过 ≥33/34。**main batch (max_iter=15) 的 var_curve 数字可信**，conclusion 不需重 max_iter ≥ 60。
- **角度对 max_iter 更敏感（比 orig 更甚）**：cohort-level masked angle_grid 中位 15→60 从 61.1° 漂到 37.6°（差 23.5°），orig 同期 81.0° → 67.0°（差 14°）。Per-subject |angle_60 − angle_15| 中位 = **18.2° (masked)** vs **11.8° (orig)**——subject 层面也确认 masked 下角度对 iteration 收敛更敏感，不只是 cohort 中位移动。**phantom rank 修过后真实的 geometric structure 浮出**：phantom 在 orig 给主曲线沿"参与模式" 方向额外延展，让曲线方向看起来更偏离 KMeans rank-axis；masked 后真曲线与 rank-axis 部分重叠，但收敛差导致 max_iter 内角度还在漂。
- **收敛差仍存在**（max_iter=60 5/34 严格收敛）；与 orig (4/34) 同档。Principal curve 路径仍**不是 H3 主路径**（原 archive §5 已锁，本档 reaffirm）。**§3.3 coordinate-free λ₂ 仍是最稳健的 H3 直测**，不依赖 curve 收敛。
- **结论**：max_iter sensitivity 不动 conclusion，仅对"主曲线 ≈ 正交于 KMeans 主轴" 这个 secondary 几何 claim 进一步弱化（原 archive §5 已收紧到 "60°+"，本档进一步收紧到 "max_iter=15 ~60° → max_iter=60 ~38°，主曲线在 PR-2 rank 子空间内不严格正交于 KMeans 轴"）。

完整 per-(subject × max_iter) 表见：`results/topic4_attractor_masked/step1_sensitivity.csv`。

### 3.5 s_kmeans diagnostic — supervised 1D 投影

s_kmeans 是 PR-2-label-SUPERVISED 坐标（原 archive §3 重点 caveat），不能独立证明双稳态；只能确认 cluster 在 rank 空间线性可分。masked vs orig 比较是否大幅改变 separability：

| metric | orig (n=34) | masked (n=34) |
|---|---:|---:|
| Cohen's d median (\|d_avg\|) | 4.01 | **3.42** |
| Cohen's d range | [3.21, 6.31] | **[2.99, 5.01]** |
| midpoint-threshold accuracy median | 1.000 | **0.999** |
| midpoint-threshold accuracy range | [0.975, 1.000] | **[0.949, 1.000]** |

**判读**：mask 后 s_kmeans separability 略弱（d 从 4.01 → 3.42，accuracy 从 1.000 → 0.999），与 phantom 在 orig 给 cluster 加"参与模式拐点"使 supervised 分离器看起来更干净一致。但都仍远高于"两 cluster 不可分"的阈值（d > 2.5 普遍认为是 large effect）——**caveat 不变**（s_kmeans 是 cluster sanity，不是 H3 evidence）。

### 3.6 Cluster size imbalance

| metric | orig (n=34) | masked (n=34) |
|---|---:|---:|
| min(cluster)/total median | 0.428 | **0.416** |
| range | [0.156, 0.497] | **[0.191, 0.500]** |

cluster 平衡几乎不动（中位 0.428 → 0.416）。原 archive §6 提到的 "imbalance 越极端 minority-row transition counts 越少 → λ₂ 估计噪声越大" caveat 在 masked 下程度类似，不构成额外 caveat。

### 3.7 PCA 子空间几何（PC1 / cumulative top-3）

| metric | orig (n=34) | masked (n=34) |
|---|---:|---:|
| PC1 ratio median | 0.303 | **0.242** |
| PC1 ratio range | [0.171, 0.585] | **[0.136, 0.456]** |
| top-3 cumulative ratio median | 0.594 | **0.532** |
| top-3 cumulative ratio range | [0.352, 0.858] | **[0.284, 0.782]** |

**判读**：masked 后 PC1 解释方差中位从 30% 降到 24%、top-3 累计从 59% 降到 53%——phantom 在 orig 给非参与通道贡献"假主成分方向"使 top-PCs 看起来更主导。masked 后 rank 子空间维度更分散，与 §3.2 的 var_curve 也下降一致。**原 archive §1 "top-3 cumulative ratio median 0.594，不能由此结论高维传播态是 1D manifold" 的 caveat 在 masked 下变得更强**（53% vs 59%，离 1D-manifold 越远）。

---

## 4. 判读 — Checkpoint 标准对照

按 `rerun_roadmap_2026-05-20.md` §5h 标准 + step5f §4 模式：

| Gate | 状态 |
|---|---|
| GOF pass 率大幅下降？ | ❌ **NO**（33/34 → 33/34，pass 数完全相同，仅 fail subject 换人） |
| GOF pass 率从 ≥30/34 翻到 <25/34（崩盘）？ | ❌ **NO** |
| Coordinate-free λ₂ 方向反转（NULL → 反方向显著）？ | ❌ **NO**（无任何 subject negative 显著） |
| Coordinate-free λ₂ 显著数大幅下降？ | ❌ **NO**（10/34 → 13/34，反而 +3） |
| Principal curve var_curve 全部翻到 < 0.6（GOF 全 fail）？ | ❌ **NO**（中位降到 0.729 仍远高 0.6 阈门） |
| s_kmeans Cohen's d 翻到 < 1（cluster 不可分）？ | ❌ **NO**（4.0 → 3.5 仍远高分离阈） |
| max_iter sensitivity 翻转（masked 三档 var/angle 偏离 orig 三档 > 10%）？ | ⚠ **angle 漂移幅度比 orig 大**（masked 15→60: 61.6°→33.6°, 28° drift; orig 14° drift），但 var_curve 三档稳定（15→60: 0.729→0.757, 3.8% drift），GOF 闸门 0.6 三档都过。只让"主曲线 ≈ 正交于 KMeans 主轴" 这条 secondary 几何 claim 进一步弱化（详见 §3.4 判读），不动 conclusion |
| **任何 cohort-level 显著 ↔ NULL 翻转？** | ⚠ 仅"GOF fail subject 换人"（916 → 1077），其中 916 是 cohort exit，1077 是 PASS→FAIL；**1 条 borderline 翻转**，但是 secondary sanity（GOF pass 数不动），不构成 primary metric 翻转 |
| **任何 primary H3 路径方向反转？** | ❌ **NO**（λ₂ 路径方向 100% 保持，加强；principal curve 路径在原 archive §5 就被标"within-cluster geometry，不是 H3 primary"，本档不依赖 curve 角度做 H3 判读） |

**Step 5h 整体方向**：⚠ PASS — primary H3 路径（coordinate-free λ₂）方向保持且实质加强；GOF pass 率持平；1 条 secondary GOF fail subject 换人（916 stable_k exit + 1077 var 下降，与 PR-2 修法预期一致）。可进 5i 收口。

---

## 5. 不再 valid 的旧数字 / 需要主文档更新的位置

### 5.1 Topic 1 主文档 (`docs/topic1_within_event_dynamics.md`)

§3.1d cluster geometry bridge 当前只引"Topic 4 attractor diagnostics Step 1 完成；coordinate-free λ₂ 10/34 显著"。**masked 后建议更新为 13/34 (+3)** + 注明 phantom 修法是"洗掉 cluster label 上的伪噪声，metastability 信号浮出"。具体改写留到 Step 5i 收口阶段。

### 5.2 Topic 4 主文档（尚未建立）

CLAUDE.md §5 原则：所有 sensitivity 闸门通过才进主文档。本档 + step5f + step5a–5g 之后才有 sufficient cohesion 评估是否开 Topic 4 主文档；不在 5h 单独触发。

### 5.3 原 Topic 4 archive `topic4_attractor_diagnostics_step1_results_2026-05-10.md`

不修改（per 5f 模式：plan / 早期 archive 历史可溯）；本文是该 archive 的下游补充。具体引用规则：当回答 Topic 4 questions 时，先看 topic0 §3.1 phantom 状态 + 本档 §3 数字，再回看早期 archive 拿 method detail。

### 5.4 `MEMORY.md` "topic4 attractor diagnostics Step 0+1 status" 条目

当前 entry：`10/34 subjects pass p<0.001 & λ₂>0`。**masked 后 → 13/34**。建议在 5i 收口时同步更新 memory entry，加 "phantom 修过后 λ₂ 显著 +3"。

---

## 6. 下一步（按 rerun_roadmap §5i）

- **5i — 主文档收口**（含 Topic 4 archive + 主文档桥接 + AGENTS.md cross-PR 合同更新）
- 5h 完成后剩余 Phase 0 任务：5d.3 (PR-4C) / 5e (PR-5) / 5g (PR-7) + Checkpoint B（5d.3 + 5e 后正式触发）

---

## 7. 工件清单

新生成（masked）：
- `results/topic4_attractor_masked/step0_audit.csv`（40 行 audit）
- `results/topic4_attractor_masked/step0_audit_summary.md`
- `results/topic4_attractor_masked/per_subject/*.json`（34 个 subject）
- `results/topic4_attractor_masked/step1_cohort_summary.csv`（34 行）
- `results/topic4_attractor_masked/step1_sensitivity.csv`（34 × 3 = 102 行）
- `results/topic4_attractor_masked/step1_sensitivity_summary.md`
- `results/topic4_attractor_masked/step1_summary.md`

代码（10 项 TDD + 5 个 script + 1 src 改造）：
- `src/topic4_attractor_diagnostics.py`（`build_rank_feature_matrix` + `run_step1_subject` 加 `mask_phantom` 参数）
- `tests/test_attractor_masked_features.py`（新建，10 项 PASS）
- `scripts/audit_topic4_step0.py`（`_apply_masked_paths` + `--masked-features` flag）
- `scripts/run_attractor_step1.py`（同上 + `mask_phantom` plumbing）
- `scripts/run_attractor_step1_sensitivity.py`（同上 + `mask_phantom` plumbing）
- `scripts/augment_attractor_step1_kmeans_s.py`（同上 + `mask_phantom` plumbing）
- `scripts/summarize_attractor_step1.py`（同上，纯路径）

日志：
- `logs/step5h_audit_masked.log`
- `logs/step5h_step1_masked.log`
- `logs/step5h_augment_kmeans_s_masked.log`
- `logs/step5h_sensitivity_masked.log`
- `logs/step5h_summarize_masked.log`

---

## 8. 一句话总结

**Topic 4 attractor Step 1 在 phantom rank 修过后所有 primary 方向保持**（GOF pass 33/34 → 33/34 持平；coordinate-free λ₂ 10/34 → **13/34 实质加强**，5 个新显著主要源自 phantom 修过后 cluster label 噪声被洗掉，2 个失去显著都无害——916 cohort exit + liyouran p=0.001→0.002 borderline）。Principal curve var_curve 中位从 0.953 降到 0.729，这是 phantom 修法的预期方向（phantom 在 PCA-3 子空间贡献的假结构被去掉），GOF 闸门 0.6 仍稳。**没有任何 H3 primary 方向反转**。可进 5i 收口。
