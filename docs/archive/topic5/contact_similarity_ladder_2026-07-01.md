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
