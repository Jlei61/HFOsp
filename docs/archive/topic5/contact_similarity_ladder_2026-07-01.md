# Topic 5 A 线 · 触点相似性几何阶梯（sensitivity/robustness 复核）

> 日期：2026-07-01 · 上游：`docs/archive/topic5/axis_alignment_AB_result_2026-06-14.md`（A 线主统计 = R3 场相关）
> 代码：`scripts/run_topic5_contact_similarity.py` + `scripts/plot_topic5_contact_similarity.py`
>       （核心复用 `src/topic5_contact_similarity.py` + `src/topic5_axis_alignment.py` + `src/propagation_contact_plane_readout.py`）
> 结果：`results/topic5_ictal_recruitment/contact_similarity/`（gitignored；见下方复现命令）
> 计划：`docs/superpowers/plans/2026-06-30-topic5-contact-similarity-ladder.md`

---

## 0. 目的

A 线主结论说"间期传播轴与发作期激活在空间上相似"，这个相似统计量（R3）本身是"把触点值铺到
81×81 网格再算场相关"算出来的——铺网格、平滑这些几何步骤有没有可能是相似数字的主要来源、而不是
真正的对齐信号？这份复核把同一套触点、同一个零假设检验，沿着"几何复杂度"搭一个三级阶梯重新算一遍：
**R1**（不做任何几何处理的逐触点 Pearson）→ **R2**（在触点位置上做与 R3 完全同一带宽的高斯平滑，
但不铺网格）→ **R3**（现有发表统计量：铺网格 + mirror-invariant 场相关）。三档共享同一批匹配触点、
同一冻结带宽 σ、同一 per-seizure 配对零假设harness（within_shaft / channel / anchor_matched），
唯一变量是几何处理的复杂度。这是**灵敏度/稳健性**复核，不产生新的队列级主张（见 §4 解读）。

## 1. 队列数字（已核验）

队列发现：20 个候选被试 → 19 个有 T0 特征缓存（`yuquan_xuxinyi` 无 t0-cache，未进候选）→
**n_ok = 18**（两个激活量宽频 broadband 和高频 hfa 都是 18；唯一被排除的 `epilepsiae_139` 是单杆被试，
within-shaft 零假设在单杆上退化，两个激活量下都排除）。

### 宽频（broadband，80–500 Hz 激活；n_ok=18，B=1000）

| 量 | 数值 |
|---|---|
| smooth_delta 中位（R1→R2，平面内平滑贡献） | **+0.176** |
| grid_delta 中位（R2→R3，网格贡献） | **−0.018**，bootstrap CI [−0.104, +0.016] |
| 网格等价检验（SESOI=±0.05） | **未通过**（CI 跨出 −0.05 之外） |
| within-shaft 超零假设通过数 R1 / R2 / R3 | **6 / 5 / 4**（满分 18） |
| 无几何顺序基准 Spearman / Kendall 超零假设 | **7/18 / 8/18** |
| R3 场数值与 A 线主统计逐位对照 | max\|Δ\| = **0.0000**，0/18 偏差（复刻无误） |

### 高频（hfa，60–100 Hz 激活；n_ok=18，B=1000）

| 量 | 数值 |
|---|---|
| smooth_delta 中位 | **+0.201** |
| grid_delta 中位 | **−0.024**，bootstrap CI [−0.081, +0.003] |
| 网格等价检验（SESOI=±0.05） | **未通过** |
| within-shaft 超零假设通过数 R1 / R2 / R3 | **9 / 7 / 5**（满分 18） |
| Spearman / Kendall 超零假设 | **7/18 / 8/18** |
| R3 场数值与 A 线主统计逐位对照 | max\|Δ\| = **0.0000**，0/18 偏差 |

## 2. 复现命令（逐字）

输入根 `--input-results-root` 指向持有 gitignored T0 特征缓存 + axis 记录的主树；
输出统一写入同一个命名空间目录 `results/topic5_ictal_recruitment/contact_similarity/`
（文件名带激活量后缀，不再用 `contact_similarity_hfa/` 这样的独立子目录——与
`axis_alignment/axis_alignment_{band}_max_ab_B1000.json` 的单目录命名约定保持一致）：

```bash
python scripts/run_topic5_contact_similarity.py --activation broadband --B 1000 --input-results-root /home/honglab/leijiaxin/HFOsp/results --out-dir results/topic5_ictal_recruitment/contact_similarity
python scripts/run_topic5_contact_similarity.py --activation hfa --B 1000 --input-results-root /home/honglab/leijiaxin/HFOsp/results --out-dir results/topic5_ictal_recruitment/contact_similarity
python scripts/plot_topic5_contact_similarity.py --activation broadband --out-dir results/topic5_ictal_recruitment/contact_similarity
python scripts/plot_topic5_contact_similarity.py --activation hfa     --out-dir results/topic5_ictal_recruitment/contact_similarity
```

标定耗时 ≈187 s/被试（单核，B=1000）。产出的 `cohort_summary_{activation}.{json,csv}`、
`per_subject/*.json`、`figures/*.png` 均按项目惯例被 `results/` 顶层 `.gitignore` 规则忽略，
不会随分支流转——本文档 + `figures/README.md` 是唯一随分支走的记录，因此上表数字和复现命令
必须能独立重建同一结果。

## 3. 纠正后的解读（逐字，禁改写/禁强化）

以下文字与 `results/topic5_ictal_recruitment/contact_similarity/figures/README.md` 的
`**关注点**` 段落逐字一致（宽频档；hfa 档把数字替换为 §1 高频表的对应值，文字结构不变）：

> 平面内平滑 (R1→R2) 把观测相似度中位数抬高 +0.176(宽频)/+0.201(高频),CI 不含 0 —— 即用触点的平面位置做平滑确实抬高了观测到的相似度数值。**但这不等于'几何让更多被试超过随机对照'**:超过'同电极杆内打乱'零假设的被试数从 R1→R2→R3 是**下降**的(宽频 6→5→4,高频 9→7→5),因为平滑同时也抬高了零假设(打乱后的数据平滑后也更像)。准确说法:平滑抬高的是**原始相似度数值**,不是**超随机对齐的证据量**。网格步 (R2→R3) 中位数几乎不动、略降(−0.018/−0.024),等价检验在 ±0.05 内未通过 → 只能说'未见可分辨的网格增益',不能说'严格为零'。不带几何的纯触点序列 Spearman/Kendall 分别 7/18、8/18 超零假设。R3 场数值与 A 线主统计逐位一致(0/18 偏差),证明复刻无误。结论(稳健性层,非新主张):场更高的相似度数值主要来自平面几何平滑(它同时抬高信号与零假设);不铺网格的触点度量已复现场;网格无可分辨增益。

## 4. 口径

- **允许**：把上面第 3 节的解读原文用于论文/主文档的稳健性小节。
- **不允许**：把"smooth_delta 为正"简化成"几何平滑证明了对齐更强"——它同时抬高了零假设，
  超零假设的被试数反而下降。
- **不改变主线**：A 线主结论（间期传播轴 = 患者内共享粗骨架 readout）不受影响；本复核只回答
  "R3 场统计量的数值大小依赖几何处理的哪一步"，是灵敏度/稳健性附注，不是新的队列级主张。
- **Tier**：sensitivity/robustness（非 primary cohort claim），与 `docs/superpowers/plans/2026-06-30-topic5-contact-similarity-ladder.md` Global Constraints 一致。

### Scope block（英文原文逐字保留，供论文/跨 topic 引用）

- **Supports ONLY**: "Spatially weighted contact-level similarity captures the same coarse interictal–ictal spatial scaffold as the gridded field readout, indicating the field result is driven mainly by local spatial smoothing rather than grid interpolation." — a useful spatial readout / sensitivity metric.
- **Does NOT support**: "effectively characterizes the epileptic pathological network." Evidence: within-shaft pass counts DROP R1→R3 (6/5/4 broadband, 9/7/5 hfa) — no increased cohort-positive evidence.
- **Upgrade** to a pathological-network claim requires clinical validation (SOZ/resection/outcome, propagation endpoint, cross-window stability). The R2b native-3D sensitivity (§5 below, now complete) is a defensive check against a 2D-projection artifact, NOT such a clinical upgrade.

---

## 5. R2b native-3D 灵敏度复核（B=1000 全队列，已核验）

> 代码：`scripts/augment_topic5_r2b_3d.py` + `scripts/plot_topic5_r2b_sensitivity.py`
>       （核心复用 `src/topic5_contact_similarity.py` 的 n-D 触点核 + `src/seeg_coord_loader.py` + `src/propagation_skeleton_geometry.parse_shaft`）
> 结果：`results/topic5_ictal_recruitment/contact_similarity/r2b_summary_{broadband,hfa}.json`、`r2b_coverage_{band}.csv`、`figures/r2b_sensitivity_{band}.png`（均 gitignored；见 §5.4 复现命令）
> 计划：`docs/superpowers/plans/2026-07-01-topic5-r2b-3d-sensitivity.md`
> Tier：sensitivity/robustness（非 primary cohort claim）

### 5.0 这一档在问什么（朴素话）

R1/R2/R3 里所有几何处理都发生在一个 **2D「触点平面」**上——先对每根电极杆做横切 PCA 得到一个平面，再把触点投到这个平面上算相似度。审稿人自然会问：会不会正是「把 3D 排布压成一个 2D 平面」这一步、而不是触点真实的解剖排布，制造了那些相似的数字？

R2b 就是把这个 2D 平面换成触点的**原生 3D 毫米坐标**（直接算三维欧氏距离），其它一切都不动：同一批触点、同一个「同电极杆内打乱」零假设、同一冻结带宽逻辑、mirror 都关掉、同一 B=1000 / seed=20260614。为了公平对照，2D 那一级也在**完全相同的公共触点子集上、同样关掉 mirror** 重算一遍，记作 R2_nm。主比较 = **R2b − R2_nm**（两者都 no-mirror、同一公共子集）——如果这个差落在 ±0.05（SESOI 最小关注差）以内，就说明「换成真三维坐标」看不出可分辨的差别，2D 平面读出已经够用。

「公共子集」= 既在 2D 轴记录里、又有合法 mm 3D 坐标的那批触点；两级（R2_nm/R2b）都只用这批，绝不拿全通道的旧 R2 去比缩水通道的 R2b。坐标必须是毫米（mm 硬门 P1-3）；每个被试各用各的坐标系（Epilepsiae=MNI152-mm、Yuquan=fs-native-RAS-mm），**从不跨被试合并点云**。

### 5.1 数字（主比较 = R2b − R2_nm，仅 r2b_status=ok 被试）

| 量 | 宽频 broadband | 高频 hfa |
|---|---|---|
| n_ok | **18** / 19 | **18** / 19 |
| R2b − R2_nm 队列中位 | **+0.0001** | **+0.0012** |
| bootstrap CI | **[−0.0099, +0.0077]** | **[−0.0051, +0.0107]** |
| SESOI=±0.05 等价检验 | **通过**（CI 严格落在 ±0.05 内） | **通过** |
| n_ok_insufficient_null（两级零假设都不欠功率） | **0** | **0** |
| R3 provenance 逐位对照（augment 记录的 stored-R3 == `cohort_summary_{band}.json` R3 within_shaft obs） | **18/18 一致** | **18/18 一致** |

> stored-R3 一致性说明 augment 没有把基线 `cohort_summary_{band}.json` 漂移/覆盖（该文件 mtime 保持在 augment 运行前，R3 的 81×81 网格未重算）。R2_nm 与 R2b 的 `obs_subject` 对所有 ok 被试均为有限值。

### 5.2 覆盖（coverage）

- **n_ok = 18 / 19（两个激活量都一样）**；唯一非 ok 的被试是 `epilepsiae_139`，落入 `NA_insufficient`——它只有 1 根电极杆（`n_shafts_common=1 < 2`），within-shaft 零假设在单杆上退化，与 §1 R1/R2/R3 主表里被排除的原因一致。
- **NA 分解（两个激活量都相同）**：`{NA_ineligible:0, NA_coords:0, NA_units:0, NA_insufficient:1, NA_degenerate:0, NA_no_null:0}`。
- **因缺坐标丢弃的触点数 = 0**：每个 ok 被试的匹配 2D 触点数 = 公共 3D 触点数（`n_common == n_matched_2d`，逐被试见 `r2b_coverage_{band}.csv`），即所有匹配触点都拿到了合法 mm 3D 坐标，没有任何触点因坐标缺失被丢。
- **坐标单位**：全部 mm（Epilepsiae 18 例 `mni152_1mm`，Yuquan 1 例 `fs_native_ras_mm`），无 voxel、无静默回退。

### 5.3 三选一判读（基于实测；守窄口径）

实测：宽频与高频两个激活量下，**R2b − R2_nm 的 bootstrap CI 都严格落在 ±SESOI(0.05) 之内**（宽频 [−0.0099,+0.0077]、高频 [−0.0051,+0.0107]），等价检验双双通过 → 命中判读**第一支**：

> **原生三维几何相对 2D 触点平面没有带来可分辨的额外信息；2D 平面读出已经够用，当前结论稳定。** 换句话说，§3 那个「场的高相似度主要来自平面几何平滑」的结论不是「压成 2D 平面」这一步造出来的伪影——把平面换成真三维坐标，观测相似度实质不变。

未命中的另两支（仅供对照，本次数据不适用）：
- 若 CI 整体高于 +SESOI（`R2b > R2_nm`）→ 「原生三维携带超出 2D 平面的额外信息（补充性）」；
- 若 CI 整体低于 −SESOI（`R2b < R2_nm`）→ 「把结论收窄到 2D 平面读出」。

**口径边界（严格守窄）**：本节只回答「2D 平面几何 vs 原生三维几何」这一个技术问题，是灵敏度/稳健性附注。**绝不**据此写「刻画了病理网络 / characterizes the pathological network」——那需要 §4 Scope block 列的临床验证（SOZ/切除/预后、传播端点、跨窗稳定性），R2b 不是这种升级。A 线主结论（间期传播轴 = 患者内共享粗骨架 readout）不受本节影响。

### 5.4 复现命令（逐字）

```bash
python scripts/augment_topic5_r2b_3d.py --activation broadband --B 1000 --seed 20260614 --input-results-root /home/honglab/leijiaxin/HFOsp/results --out-dir /home/honglab/leijiaxin/HFOsp/results/topic5_ictal_recruitment/contact_similarity
python scripts/augment_topic5_r2b_3d.py --activation hfa       --B 1000 --seed 20260614 --input-results-root /home/honglab/leijiaxin/HFOsp/results --out-dir /home/honglab/leijiaxin/HFOsp/results/topic5_ictal_recruitment/contact_similarity
python scripts/plot_topic5_r2b_sensitivity.py --out-dir /home/honglab/leijiaxin/HFOsp/results/topic5_ictal_recruitment/contact_similarity
```

标定耗时 ≈ 25–26 min/band（单核，B=1000，两级零假设 R2_nm+R2b 各跑一遍）。`r2b_summary_{band}.json` / `r2b_coverage_{band}.csv` / `figures/r2b_sensitivity_{band}.png` 均被 `results/` 顶层 `.gitignore` 忽略；本文档 + `figures/README.md` 是唯一随分支走的记录。（注：两个激活量并发跑时共享同一 `per_subject_r2b/` 中间目录、后写覆盖先写——该目录只是 provenance，不被图消费；band-specific 的 `r2b_summary_{band}.json` 才是权威产出，交叉核对由重放 augment 的确定性 stored-R3 读取完成。）

### 5.5 六个 NA 代码朴素话（figures README 同步）

augment 对每个被试给出的 `r2b_status` 只有 `ok` 或以下六种 NA 之一，含义：

- **NA_ineligible** — 这个被试连触点相似性阶梯的**基础上下文**都没建起来（`_ctx` 返回空：没匹配到发作触点 / 缺 T0 缓存），根本没进 R2b 这一级。
- **NA_coords** — 拿不到该被试触点的 **3D 毫米坐标**：坐标文件缺失、加载器报错、或坐标的通道顺序与匹配通道顺序对不上（不敢乱对齐索引）。
- **NA_units** — 拿到了坐标但**不是毫米单位**（例如体素 voxel）。mm 硬门（P1-3）直接拒绝，**不做静默回退**。
- **NA_insufficient** — 「既有 2D 平面记录、又有合法 mm 3D 坐标」的**公共触点子集太小**：公共触点 < 6，或跨的电极杆 < 2，或没有任何一次发作在公共子集上凑够 ≥6 个有限值。（本次唯一一例 `epilepsiae_139` 就是单杆 → 跨杆数 < 2。）
- **NA_degenerate** — 公共子集的 **3D 点云退化**：所有触点坐标几乎重合 → 3D 最近邻间距中位数 = 0 → 高斯核带宽 `sigma_3d ≤ 0` 无法定义。
- **NA_no_null** — **零假设 harness 没给出观测统计量**：R2_nm / R2b 两级里至少一级没能产出 `obs_subject`。

本次全队列（两个激活量）只出现 `NA_insufficient` 一例，其余五种代码计数均为 0。
