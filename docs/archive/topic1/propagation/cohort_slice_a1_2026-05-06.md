# Slice A1 cohort 扩容：3 个 Yuquan subject 加入 PR-2 cohort

> 日期：2026-05-06
> 范围：**仅 PR-1 / PR-2 cluster + PR-3 viz**。`zhangjiaqi`、`gaolan`、`wangyiyang` 三个 Yuquan subject 落 per-subject JSON、cohort summary 重建到 n=33、PR-3 viz 重画。
> **不在本次范围**：PR-2.5 reproducibility、PR-4A occupancy、PR-4B (Step 0/1/2-3) 耦合、PR-5、PR-6 anchoring 的 **cohort-level p 值 / Wilcoxon / Spearman** 全都没重算。任何引用 `n_subjects=33` 做 PR-2.5/PR-4*/PR-5/PR-6 cohort 主张的下游必须先单独发 PR 重算（详见 §4）。
> 状态：lineage-adjacent（**不是 bit-replicate**）—— 详见 §3
> 上游主文档：`docs/topic1_within_event_dynamics.md`，`docs/archive/topic1/propagation/interictal_group_event_internal_propagation.md`

## 1. 背景

之前 Yuquan PR-2 cohort 只有 10 个 subject（`chengshuai, chenziyang, hanyuxuan, huanghanwen, huangwanling, litengsheng, liyouran, sunyuanxin, xuxinyi, zhangjinhan`）。`scripts/run_interictal_propagation.py::YUQUAN_SUBJECTS` 列了 18 个 Yuquan subject，但只有 10 个被实际跑出 per-subject JSON。Audit（2026-05-06）发现：

- **8 个 silent-failure subject**（`zhangkexuan, pengzihang, songzishuo, zhangbichen, zhaochenxi, zhaojinrui, zhourongxuan, zhangjiaqi`）：`subject_dir.glob("*_lagPat.npz")` 看上去通过，但 `load_subject_propagation_events` 实际需要 `_lagPat_withFreqCent.npz`（10ch full set，cross-PR 合同要求），这些 subject 多数没有 withFreqCent 文件。
- **`gaolan`、`wangyiyang`、`dongyiming` 不在 inclusion list**：`gaolan` 双重缺失（`withFreqCent` 没生成 + 不在 list）；`wangyiyang` 有 9/12 块的 `withFreqCent`，只是名字不在 list；`dongyiming` 有 7/12 withFreqCent，也不在 list。

Slice A1 修补 3 个：`zhangjiaqi`、`gaolan`、`wangyiyang`。前两个用 legacy pack 重新生成 `_lagPat_withFreqCent.npz`，第三个直接加 inclusion list 用其 2021-11 已有的 9/12 块。**`dongyiming` 暂不加**（用户没明确点名；7 个 silent-failure 中除了 zhangjiaqi 也都暂不加，因为它们缺 `_gpu.npz`，需要从 raw EDF 重新做 detection，触发更大规模的谱系问题）。

## 2. 操作

### 2.1 Loader 修复（强制前提）

发现一个跟 task 直接相关的 bug：`src/interictal_propagation.py::load_subject_propagation_events` 的 glob 是 `*_lagPat.npz`，但 cohort 缓存 JSON（chengshuai n_events_total=27632, n_channels=8）实际是从 `*_lagPat_withFreqCent.npz` 出来的（同 dir `_lagPat.npz` 总事件数 30050，6ch）。当前代码在源头就违反了 `docs/archive/topic1/propagation/interictal_group_event_internal_propagation.md` cross-PR 合同：

> Use `_load_bools_and_channels` (or `load_subject_propagation_events`) on the `*_lagPat_withFreqCent.npz` files (10ch full set), not `*_lagPat.npz` (older 7ch legacy slice).

不修这个 bug，新加的 3 个 subject 会读 6/7ch `_lagPat.npz`，跟旧 27 个 cohort（8/10ch withFreqCent）通道集不一致；cohort summary 会被污染。

修法（commit 待补）：

- `load_subject_propagation_events` 先 glob `*_lagPat_withFreqCent.npz`，没有再 fallback 到 `*_lagPat.npz`。Epilepsiae 两个文件通常等价（同 chns、同事件数），不受影响；Yuquan 切回 withFreqCent 的 8/10ch 集。
- `_record_name_to_packed_paths` 按 lagPat 变体配 packedTimes：withFreqCent → `_packedTimes_withFreqCent.npy`，否则 `_packedTimes.npy`。
- `_record_name_from_lagpat_path` 处理 `_lagPat_withFreqCent.npz` 后缀。

验证（2026-05-06）：3 个 Yuquan subject + 3 个 Epilepsiae subject loader 输出全部跟缓存 JSON 完全匹配（chengshuai 8ch/27632, huangwanling 4ch/107062, litengsheng 24ch/2070, 253 8ch/75062, 548 12ch/25282, 139 7ch/14439）。56 个 propagation 单元测试全过。

### 2.2 上游 pack：`zhangjiaqi`、`gaolan`

driver：`scripts/legacy_pack_lagpat_withfreqcent.py`（不改 legacy 模块本体，import + monkey-patch）。

- legacy 脚本路径：`ReplayIED/inter_events/yuquan_24h_perPatientAnalysis_dropRef/p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter.py`
- 从 legacy `__main__` 摘出 per-subject 参数：`zhangjiaqi: pickChn_thresh=1.7, packWinLen=150e-3` / `gaolan: pickChn_thresh=1.9, packWinLen=300e-3`
- monkey-patch：`mne.io.read_raw_edf` 默认加 `encoding='latin1'`（Yuquan EDF annotation 通道有非 ASCII 字节）；`plot_perSeg_specCenter` no-op 掉（避免交互绘图阻塞）
- 输入：原始 `.edf` + 已有 2021-06 `_gpu.npz` + 2021 `_refineGpu.npz`
- 输出：`<stem>_lagPat_withFreqCent.npz`、`<stem>_packedTimes_withFreqCent.npy`，落到 artifact 根（`/mnt/yuquan_data/yuquan_24h_edf/<subject>/`）

### 2.3 Inclusion list

`scripts/run_interictal_propagation.py::YUQUAN_SUBJECTS` 增补：`gaolan`、`wangyiyang`。`zhangjiaqi` 之前已经在 list 内，只是缺 withFreqCent 输入；这次补上。

cohort：18 → 20 名义上；实际 PR-2 跑得通的 Yuquan subject 数 **10 → 13**（旧 cohort 10 + 新加 3）。

## 3. Caveat: lineage-adjacent ≠ bit-replicate

**这点必须保留在主文档和 archive 双向回链。**

- 旧 cohort 的 `_lagPat_withFreqCent.npz` 是 2021-11 在 `niking314` 机器上跑出来的，依赖 2021 vintage cusignal（`p16_packGroupEvents...withFreqCenter.py` 顶部 `import cupy as cp; import cusignal`）。
- 当前机器的 `cuda_env` 上 `cusignal=23.08.00` / `cupy=13.6.0`，**不是 2021 vintage**。
- 即便 legacy 脚本本体未改、输入 `.edf` 是同一份原始数据、输入 `_gpu.npz` / `_refineGpu.npz` 是 2021-06 的同一批 artifact，输出的 `_lagPat_withFreqCent.npz` 也**不会 bit-equivalent** 于 2021-11 在原机器的同一脚本输出。
- 作为旁证：`docs/archive/hfo_detector_v2/v2_cohort_rebuild_plan_2026-05-05.md` §3 已声明 "21 年 cusignal vintage cannot be bit-reproduced on modern stacks"。

视为 **lineage-adjacent**（同代码路径、同输入、同 subject 参数；CPU 部分纯 numpy/scipy，唯一漂移点是 `return_massCenterPat` / spectrogram-related code 是否走 cupy/cusignal，本脚本中是纯 scipy 路径），但**不能等同于 2021 cohort bit-replicate**。

未来如果做 v2 detector cohort 全量重建（含 Yuquan），整批 cohort 都会 reset 到 v2 谱系，那时这条 caveat 失效。

## 4. PR-2 / PR-3 / PR-2.5 / PR-4 重算范围

加入 cohort 后必须重算的下游：

| PR | 重算范围 | 为什么 |
|---|---|---|
| PR-1 / PR-2 cluster | 仅 3 个新 subject | 旧 27 个 JSON 缓存仍然有效（loader 修复后跟缓存数值一致） |
| PR-2.5 reproducibility | 仅 3 个新 subject | 同上；cohort summary（`stable_k` 分布、forward/reverse 数）会变 |
| PR-3 viz / cohort fig | 重画 | per-subject heatmap/MI 直接生成；6-panel cohort 图依赖全量 |
| PR-4A occupancy | 仅 3 个新 subject + 重新 day/night Wilcoxon | day/night 检验在 cohort 层 |
| PR-4B Steps 0-3 | 仅 3 个新 subject + cohort 层 Wilcoxon / Spearman 重算 | L1/L2/L3 全部 cohort-level p 值 |
| PR-6 anchoring | 仅 3 个新 subject + cohort 重新汇总 | h1_primary_eligible 计数 + endpoint 统计 |

**本次 Slice A1 范围只覆盖 PR-1 / PR-2 cluster / PR-3 viz**。PR-2.5 reproducibility / PR-4A occupancy / PR-4B Step 0–3 / PR-5 / PR-6 的 cohort-level p 值（Wilcoxon、Spearman、占比、deltas 中位数）**全部没有重算**。`pr1_cohort_summary.json` 里仍然带这些字段，是 aggregator 把每个 subject 的 per-subject JSON 中已存在的字段简单合并得到的，**不是用统一 PR-2.5+/PR-4+/PR-5/PR-6 流程重新跑出来的**。下游引用任何 `n=33` 做 PR-2.5+ / PR-4+ / PR-5 / PR-6 cohort 主张前必须单独发 PR 重算。

按价值优先级，follow-up PR 推荐顺序见本文末"§8 后续重跑优先级"。

## 5. 7 个 known-gap subject + runner gate

- `zhangkexuan, pengzihang, songzishuo, zhangbichen, zhaochenxi, zhaojinrui, zhourongxuan`：
  - 没有 2021 `_gpu.npz`（zhangjiaqi 是例外，那个有完整 `_gpu.npz`）
  - 上游 detect 必须用 `scripts/run_hfo_detection.py` 在 v1 / v2 detector 重做，会产生 detect 谱系混杂 cohort（v2-flavored detect × legacy pack）
  - 撞期：v2 cohort rebuild 正在跑（Phase 3.4 Epilepsiae GPU run，~33h）
  - 决议：暂不补，等 v2 cohort rebuild 完成后整体重建

**Runner gate（2026-05-06 加固）**：因为 loader 修复后会回退到 `_lagPat.npz`，如果不在 runner 层加 gate，这 7 个 silent-failure subject 仍会被默认 `--dataset yuquan` 跑（fallback 到 6/7ch lagPat），产生异质 cohort，跟 §5 known-gap 自相矛盾。

补丁：`scripts/run_interictal_propagation.py` 加 `_has_propagation_inputs(dataset, subject_dir)`：

- Yuquan：必须存在 `*_lagPat_withFreqCent.npz`（cross-PR 合同要求 10ch full set）
- Epilepsiae：保留宽松（`*_lagPat.npz` 即可，因为 cohort 路径下两个文件等价）

替换了 11 处旧 gate `subject_dir.glob("*_lagPat.npz")`。验证 7 个 silent-failure subject 全部 SKIP；zhangjiaqi / gaolan / wangyiyang / chengshuai 全部 PASS。

## 6. 文件清单

新生成（artifact 根 `/mnt/yuquan_data/yuquan_24h_edf/`）：

- `zhangjiaqi/FC1047T*_lagPat_withFreqCent.npz` × 13
- `zhangjiaqi/FC1047T*_packedTimes_withFreqCent.npy` × 13
- `gaolan/FA0013*_lagPat_withFreqCent.npz` × 12
- `gaolan/FA0013*_packedTimes_withFreqCent.npy` × 12

代码改动：

- `src/interictal_propagation.py`：loader 修复（withFreqCent prefer + fallback；packedTimes 按变体分支）
- `scripts/run_interictal_propagation.py`：(a) `YUQUAN_SUBJECTS` 加 `gaolan`、`wangyiyang`；(b) 新增 `_has_propagation_inputs` helper，对 Yuquan 强制要求 `*_lagPat_withFreqCent.npz`；替换 11 处旧 gate
- `scripts/legacy_pack_lagpat_withfreqcent.py`：新建 driver
- `scripts/aggregate_propagation_cohort.py`：新建 cohort summary 重建工具

PR-2 输出（待补）：

- `results/interictal_propagation/per_subject/yuquan_zhangjiaqi.json`
- `results/interictal_propagation/per_subject/yuquan_gaolan.json`
- `results/interictal_propagation/per_subject/yuquan_wangyiyang.json`

PR-3 viz（待补）：

- `results/interictal_propagation/figures/yuquan/{zhangjiaqi,gaolan,wangyiyang}_propagation_heatmap.png`
- `results/interictal_propagation/figures/yuquan/{zhangjiaqi,gaolan,wangyiyang}_mi_distribution.png`

跑日志：`results/run_logs/legacy_pack_lagpat_zhangjiaqi_gaolan.log`

## 7. PR-2 cohort 数值

### 3 个新 subject 的 PR-1 / PR-2 结果

| Subject | n_ch | n_events | n_blocks | mixture (strict / possible) | bias_fraction | mean_tau (all) | stable_k |
|---|---|---|---|---|---|---|---|
| zhangjiaqi | 7 | 48,494 | 13 | False / True | 0.000 | 0.0072 | 2 |
| gaolan | 12 | 7,451 | 12 | False / True | 0.735 | 0.097 | 2 |
| wangyiyang | 22 | 1,919 | 9 | False / True | 0.742 | 0.023 | 2 |

**3 个全部是 `is_mixture=False`、`possible_mixture=True`，不是 strict mixture。** 这意味着 cohort 从 `30 strict / 0 possible` 变成 `30 strict / 3 possible`，**不能写成"33/33 strict multimodal"**。`stable_k=2` 仍然落在 cohort 主流。

**zhangjiaqi anomaly 验证（2026-05-06）**：`bias_fraction=0` 与 `mean_tau=0.007` 是异常低值（cohort median 0.66 / 0.088）。直接对比同 subject 两份 lagPat 输出：

- 旧 6ch `_lagPat.npz`（2026-04-23 vintage、不同打包参数）：raw mean_tau=0.0068，centered=0.0068，bias=0
- 新 7ch `_lagPat_withFreqCent.npz`（本次 2026-05-06 pack）：raw=0.0072，centered=0.0072，bias=0

两个独立打包路径数值几乎一致 → **data-property，不是新 pack bug**。底因推测：H1–H7 是同一个 H 电极相邻 7 个触点（深度电极），相邻通道时间差落到 spectrogram 时间分辨率以下，传播 rank 在事件之间几乎随机。chengshuai 8ch（多电极混合）作为 sanity 仍然给 mean_tau=0.028 / bias=0.24 → 正常 cohort 行为。zhangjiaqi 作为 known low-stereotypy outlier 入 cohort，不阻塞结论。

### Cohort summary 对比（n=30 → n=33）

| 指标 | n=30（原） | n=33（新加 3） | 是否 PR-1/2 范围内 |
|---|---|---|---|
| `n_strict_mixture` | 30 | **30**（不是 33） | 是 |
| `n_possible_mixture` | 0 | **3** | 是 |
| `mean_tau_median` | 0.0885 | 0.0884 | 是 |
| `bias_fraction_median` | 0.6516 | 0.6568 | 是 |
| `stable_k_distribution` | `{2:27, 4:2, 6:1}` | `{2:30, 4:2, 6:1}` | 是 |
| `within_cluster_tau_median` | 0.252 | 0.232 | 是 |
| `n_subjects_with_forward_reverse` | 12 | 14 | **当时否，2026-05-07 已修** —— Slice A2 (cohort_slice_a2_legacy_variant_2026-05-07.md) 用 `--pr25` 在 3 个 Slice A1 subject 上跑了统一 PR-2.5 split-half/odd-even；当前 n=33 cohort summary `reproducibility_analysis.forward_reverse.n_subjects_with_pairs=14, n_reproduced=13` 已是统一流程结果 |
| `total_forward_reverse_pairs` | 17 | 19 | **当时否，2026-05-07 已修** —— 同上，详见 Slice A2 §5.3 |

新 3 subject 全部纳入 stable_k=2 的主流分布，PR-1/PR-2 cohort-level 结论稳定（mean_tau / bias_fraction 几乎不变，within_cluster_tau 略降）。**Mixture screen 的 `n_strict_mixture` 不变（仍 30）**：3 个新 subject 是 possible mixture，不是 strict。原 cohort 备份保存在 `results/interictal_propagation/pr1_cohort_summary.backup_2026-05-06.json` 与 `pr1_subject_summary.backup_2026-05-06.json`。

**PR-2.5 / PR-4* / PR-5 / PR-6 任何 cohort-level p 值（Wilcoxon、Spearman、deltas 中位数、forward/reverse 占比）此次没有重算。** 当前 `pr1_cohort_summary.json` 里仍带有 PR-4 / PR-5 类字段（`rate_state_coupling_analysis`、`temporal_dynamics_analysis`、`seizure_proximity_analysis`、`absolute_lag_validation_analysis` 等），这些值是 aggregator 把每个 subject 的 per-subject 字段简单聚合的结果，**不是用统一 PR-4* 流程重新跑出来的**。下游引用任何 PR-2.5+/PR-4+/PR-5/PR-6 cohort 主张前必须单独发 PR 重算。

### Cohort summary 重建方式

由于 `scripts/run_interictal_propagation.py` 默认主循环跑完后会用本次 RUN 的 `subject_results` 重写 `pr1_cohort_summary.json`，本次 `--subjects zhangjiaqi gaolan wangyiyang` 跑出的 cohort summary 只含 3 subject。完整 cohort summary 用 `scripts/aggregate_propagation_cohort.py` 从 `per_subject/*.json` 重建。

**Manifest 是 cohort 唯一真相源，不是目录 glob。** 落盘的 manifest：

- `results/interictal_propagation/cohort_manifest_n33_2026-05-06.txt`（n=33，Yuquan 13 + Epilepsiae 20）

```bash
conda run -n cuda_env --no-capture-output python scripts/aggregate_propagation_cohort.py \
    --manifest results/interictal_propagation/cohort_manifest_n33_2026-05-06.txt
# 默认输出：results/interictal_propagation/pr1_subject_summary.json + pr1_cohort_summary.json
# 缺少 manifest 中任何一条 per-subject JSON 会直接 SystemExit；多余的 stale JSON 会被打印为 "ignored (not in manifest)"。
```

`--manifest` 是当前推荐用法。**不要直接跑无 `--manifest` 的 aggregator** —— 默认 discovery 模式会把目录里所有 `<dataset>_<subject>.json` 都吸进 cohort，等于把任何遗忘的 stale JSON 默默算进结论。Discovery 模式只在初始化新 cohort 时一次性使用，使用后必须立刻把当时的 inclusion list 钉成 manifest。

后续 cohort 扩缩容流程：(1) 跑新 subject 的 PR-1/PR-2 → (2) 把 `dataset/subject` 加进新版 manifest 文件 → (3) `--manifest` 重建 cohort summary → (4) 按 §4 重算 PR-2.5 / PR-4 / PR-5 / PR-6 cohort-level 字段（需要单独发 PR）。

### PR-3 viz

per-subject 图（默认目录 + 各自 README 已存在）：

```
results/interictal_propagation/figures/per_subject/yuquan_{zhangjiaqi,gaolan,wangyiyang}_propagation.png
results/interictal_propagation/figures/per_subject_mi/yuquan_{zhangjiaqi,gaolan,wangyiyang}_mi_distribution.png
```

Cohort 6-panel 图重画（`--dataset both` 含全 33 subject）：

```
results/interictal_propagation/figures/cohort_propagation_summary.png
```

## 8. 后续重跑优先级（follow-up PR 推荐顺序）

按"科学价值 / 工程成本"评分。每条都需要单独发 PR；不要在本 archive 内偷偷做。

| 排名 | PR | 重算内容 | 输入是否就绪 | 推荐理由 |
|---|---|---|---|---|
| **1** | PR-2.5 reproducibility | split-half / odd-even / forward-reverse 的统一 cohort 重算 | ✅ 三个 subject 完整 `_lagPat_withFreqCent.npz` 已就绪 | 当前 cohort summary 里 `n_subjects_with_forward_reverse=14`、`total_forward_reverse_pairs=19` 是 aggregator 拼出来的，没有走 PR-2.5 split 逻辑。8/9 → 9-12/X 这条主线结论会变，且 PR-6 anchoring 直接消费这个字段。**必须先做 PR-2.5 才能动 PR-6。** |
| **2** | PR-4A occupancy day/night | 加入 3 subject 后 dominant_fraction / normalized_entropy / TV distance 的 day/night Wilcoxon | ✅ | 当前主线结论是"day/night 漂移弱"（Wilcoxon p=0.124），cohort 增加到 33 后 p 值方向稳定与否值得验证。zhangjiaqi 的 `bias_fraction=0` + 单 H 电极性质会让 template projection agreement 重新分布。 |
| **3** | PR-3 cohort 6-panel 图（已重画但需 audit） | 当前 §7 PR-3 已经把 6-panel 重画到 n=33 | 已落盘 | 注意 panel 里"within-cluster vs between-cluster τ"和 "inter-cluster r"两个面板会被新 3 subject 拉低 within-cluster median；要确认 figures/README 描述是否仍然对得上图面。 |
| **4** | PR-4B Step 0 + Step 1 + Step 2-3 (rate-state coupling) | dominant cluster Pearson r、L2 raw τ delta、L3 lag-span / Pearson r、Spearman 一致性 | ✅ | 现有 high-confidence 子集 (n=8, dom_r>0.7) 是 cohort 唯一显著结果（7/8, p=0.016）。3 个新 subject 中 zhangjiaqi 因 bias=0 几乎确定不会进 HC subset；gaolan/wangyiyang 视 dom_r 而定。HC subset 大小变 8 → 9 或 10，p 值方向会调整但量级应稳定。 |
| **5** | PR-6 template anchoring | 仅对 3 个新 subject 重新跑 anchoring + cohort 汇总 + h1_primary_eligible 计数 | ✅（但依赖 PR-2.5 先做） | PR-6 H1/H2 都消费 PR-2.5 forward/reverse 字段。先 PR-2.5、再 PR-6，否则 PR-6 cohort summary 里仍混合 aggregator 简单拼接。 |
| 低 | PR-5（recruitment shift / novel template gate） | 现有 `pr5a/`、`pr5b/` 子目录文件仅含旧 30 subject | ✅ | 价值取决于 pr5 各自当时 hypothesis 的状态；如果 pr5 当时已经 null，新 3 subject 大概率不会改变结论，可缓做。 |
| 低 | 7 个 silent-failure subject + dongyiming 补回 | 需要 v2 detector 全量 rebuild 完成 | ❌ | v2 cohort rebuild 在跑（Phase 3.4 ~33h Epilepsiae GPU run）。等 v2 收尾后整批 cohort reset，不要在当前谱系下零敲碎打。 |

总结：**PR-2.5 是优先做的 1 项**——它价值最高，因为：(1) 它是 cohort summary 当前唯一被语义污染的 cohort-level 字段；(2) PR-6 anchoring 等下游 PR 都依赖它；(3) 输入已经就绪，工程成本低。其他都可以等 PR-2.5 落地后再排。

## 9. 三个新 subject 的 SOZ 标注

数据源：`results/yuquan_soz_core_channels.json`（来自 `p16_subs_info.py` 手工标注）。三个 subject 全部有 SOZ 标注（**非 silent-failure，不是从 SOZ 角度补的盲点**）。

| Subject | n_SOZ | 电极 | 备注 |
|---|---|---|---|
| zhangjiaqi | 10 | 单 H 电极（H1–H10，深度电极相邻 10 触点） | **SOZ 全在 H 电极上**。这与 §7 异常验证一致 —— pack 阶段 lagPat 出 7ch 大概率是 H 电极的子集，所以传播 rank 之间几乎不可分。从 SOZ vs non-SOZ 比较的角度，这个 subject 极端：所有传播参与通道都在 SOZ 内，缺少 non-SOZ 对照样本。**进 cohort 但 PR-4 SOZ-stratified 分析时应特别标记**。 |
| gaolan | 20 | A、B'、E、F 四电极（A6-A10、B'9-B'13、E6-E10、F10-F14） | 多电极混合 SOZ；lagPat 12ch，足以提供 SOZ vs non-SOZ 对照。 |
| wangyiyang | 38 | A、B、D、E、G 五电极（覆盖广） | 大范围 SOZ；lagPat 22ch；SOZ 占比偏高。 |

**对 PR-4* / PR-6 follow-up 的提醒**：zhangjiaqi 的"SOZ = 单电极相邻触点"模式可能让任何 SOZ-vs-nonSOZ 对照在该 subject 失效（缺 non-SOZ 通道参与传播）。下游 PR 在做 SOZ-stratified 分析前需明确该 subject 的处理方式（排除 / 标记 / 单独 case-study）。
