# HFOsp 项目指南

本文件只保留项目入口与影响科学正确性的约定。协作、澄清、批准和科学审阅遵循全局 AGENTS.md；已授权的实施任务须完成验证和交付。不要把旧图号、目录名或历史统计摘要当成当前科学结论。

## 按任务读取

- 科学状态与 Topic 1–5 数值：先读 `docs/topic0_methodology_audits.md` 中相关审计，再读对应 topic 文档及其引用的已接受方案；总入口为 `docs/paper_overview.md`。
- Topic 1：`docs/topic1_within_event_dynamics.md`；Topic 2：`docs/topic2_between_event_dynamics.md`；Topic 3：`docs/topic3_spatial_soz_modulation.md`；Topic 4：`docs/topic4_sef_hfo.md`；Topic 5：从总入口及 `docs/topic5_seizure_subtyping.md` 定位当前分支。
- Epilepsiae 数据或临床标签：读 `docs/epilepsiae_dataset_structure.md`。
- 论文图：先用 `docs/paper_figure_registry.md` 定位当前身份，再核对 producer、输入和说明；按 `docs/figure_style_guide.md` 的适用小节绘制。
- 仅在追溯 Yuquan legacy 结果时，依次读 `docs/LEGACY_YUQUAN_CODEBASE_MAP.md`、`docs/LEGACY_YUQUAN_FIGURE_ASSET_MAP.md`、`docs/LEGACY_PAPER_TIFF_CHAIN.md`、`docs/OLD_vs_NEW_algorithm_comparison.md`、`docs/yuquan_24h_dataset_structure.md`、`config/default.yaml`。用户明确引用 `ReplayIED/tiffs` 时先读 TIFF chain。
- 按当前依赖读取；已读且未变化的内容不重复加载。非 legacy 任务不因上述历史材料缺失而停止。

## 数据来源与局部阻塞

- 当前代码：`/home/honglab/leijiaxin/HFOsp`。
- Yuquan 原始数据及 canonical artifacts：`/mnt/yuquan_data/yuquan_24h_edf`。
- Epilepsiae raw、SQL 与 artifacts：`/mnt/epilepsia_data`。
- 历史代码：`/home/honglab/leijiaxin/HFOsp/ReplayIED`；Yuquan 主线在 `inter_events/yuquan_24h_perPatientAnalysis_dropRef/`。

追溯历史来源时，从实际 artifact 找 producer，再找绘图脚本；不能只按 Fig7 / fig7 / 7b 等名字判断。历史事实优先由原始文件和当时 producer 证明。当前方法应遵循适用的已接受合同及明确替代关系；实际实现和结果仍须核查，不能因文档声称正确就忽略矛盾，也不能把已修复的 legacy 行为恢复为标准。

必需的 legacy 目录、绘图脚本或 artifact 缺失时，先查配置、文档和已知位置；仍找不到则请求真实来源，只暂停依赖它的步骤，继续独立工作。需要而无法支持的临床标签同样局部暂停；不得猜造路径、证据或标签。

## 实现入口

入口为 `scripts/run_pipeline.py`；检测批处理为 `scripts/run_hfo_detection.py`，被试参数在 `config/subject_params.json`。用 `rg` 查实际函数和调用者，不依赖易漂移的函数清单。

Legacy 主链：检测 → `_gpu.npz`；refine → `_refineGpu.npz`；pack → `_packedTimes*.npy`；lag/freq → `_lagPat*.npz`；24h 汇总 → `hist_meanX.npz`。`_withFreqCenter` packer 写出 `_lagPat_withFreqCent.npz` 和 `_packedTimes_withFreqCent.npy`，旧 plotter 仍可能读取旧名，需核对真实输入。

HFO Detector v2 的 canonical 输出在 `results/hfo_detector_v2/`；规范见 `docs/archive/hfo_detector_v2/` 的 specification、validation contract 与 cohort rebuild plan。不得将 `results/_legacy_2021_readonly/` 当作 v2 逐事件复现目标。

## 影响科学正确性的约定

以下是检查入口；具体定义、适用队列和假设层级以对应已接受方案为准，不在此复制阶段性数值表。

- **参与通道与 rank**：legacy `lagPatRank` 给未参与通道也填了有限整数，`isfinite` 无法排除它们。KMeans 输入须用 `src.lagpat_rank_audit.build_masked_kmeans_features(ranks, bools, impute='event_median')`，或对应 helper 的 `use_masked_features=True` / attractor 的 `mask_phantom=True`。新 PR-2 标签消费者沿用已有 runner 的 `--masked-features` 与 `_apply_masked_paths()`，保证标签和输出都走正确路径。见 Topic 0 和 `scripts/run_interictal_propagation.py`。
- **端点选择**：`template_rank` 的非参与通道也可能有 rank。全数据消费者必须从每个 cluster 的原始 bools 计算并传入 `valid_mask`，不能使用默认“全部有效”；split-half 的 `-1` sentinel 模式与此区分。用 `*_lagPat_withFreqCent.npz` 的完整通道集，不用旧 `*_lagPat.npz` 切片。
- **通道对齐**：raw NPZ 各 block 的顺序可能不同；重建 union 顺序并核对 JSON `channel_names` 后，再索引 `template_rank` 或 mask。
- **复现定义**：PR-2.5 的 `forward_reverse_reproduced` 是 split-half **OR** odd-even，不能只取前者。见 `docs/archive/topic1/propagation/interictal_group_event_internal_propagation.md`。
- **Topic 4 H2**：PR-2/2.5 是模板发现层，PR-6 top-3 是端点摘要；建模通道标签优先用 masked rank-displacement 的 `primary_pair.swap_sweep`，由 `joint_valid`、`rank_a_dense_full` 和 `src.rank_displacement.derive_swap_endpoint` 推导。来源：`results/interictal_propagation_masked/rank_displacement/per_subject/`。标签分布是描述性/机制检查，swap-k 节点的 source/sink 空间紧凑性才是该方案的 primary cohort 层；见 `docs/topic4_sef_hfo.md`。
- **假设层级**：从预注册方案查 primary、secondary、mechanism sanity、sensitivity，不能按结果强弱改层级。PR-6 自身的 forward/reverse swap 检查属于 mechanism sanity；不要将其层级套到不同队列的 Topic 4 H2 空间检验。见 `docs/archive/topic1/pr6_template_anchoring/pr6_template_endpoint_anchoring_plan_2026-04-25.md`。
- **资格定义**：PR-6 的 `endpoint_defined` 为 n_ch ≥ 6，`h1_primary_eligible` 为 n_ch ≥ 7，`pass = h1_primary_eligible`；不能合并而丢失 n_ch=6 的 case series。
- **Epilepsiae**：raw 是 `*.data + *.head`，interictal 输出在 `interilca_inter_results/all_data_lns/<subject>/all_recs`。临床元数据遵循 SQL `recording/block/seizure` > `.head.start_ts`（仅 block 级校验）> legacy 脚本提示。不得用 vigilance 推断昼夜；当前 UKLFR 使用 Europe/Berlin，08:00–20:00 为 day。1h lagPat parent block 跨 seizure、post-ictal、昼夜或非平凡 gap 边界时，排除相关事件，不强行归类。

## 文档、结果与图

- Topic 1/2/3 主文档用分层中文 Markdown，只保留当前结论、关键缺口、适用合同和下一步；阶段报告与全量表放 `docs/archive/<topic>/<descriptive>_<YYYY-MM-DD>.md` 并回链。用户只要求对话审阅或禁止写文件时，不自动归档。除非用户明确要求 canvas，不改为 React 面板。
- 新结果目录按 topic 和分析含义命名，不用 PR 编号作目录名。新输出路径遵循此规则；引用历史证据保留真实原路径，不为命名规范搬动 archive。
- 审计修复走 parallel dir，旧结果保留：phantom-rank 修复使用 `interictal_propagation_masked/`、`topic4_attractor_masked/`。图文件名不重复加 `_masked`；旧文档链接保持有效。
- 图放 `figures/`；聚合 CSV/JSON 与其同级；中间结果进 `per_subject/` 或相应阶段/数据集子目录。
- 每次生成图后，用户仍需亲自目视检查。等待人工检查不阻塞 Agent 自查、修复及完整候选交付：完成本次要求的图、相关验证和配套说明，再明确列出待人工检查的版本；未经用户检查不得宣称通过人工验收。已有正式图按原验收/替换要求处理。
- 新图目录在图实际生成后写 `figures/README.md`：每图以 `### filename` 开头，中文 2–4 句说明展示内容，末尾 `**关注点**：`。不提前放空 README；含义未变时不必重写。
- Topic 4/SNN 主图或 paper-ready 图先读 figure style guide 的 Topic 4 小节，默认 `mechanism + tempA source + tempB source + electrode readout` 单行布局；诊断图、参数扫描或用户明确要求的其他布局按任务处理。
- 保留其他工作的未提交更改、冲突和运行任务；它们不自动阻塞独立工作。必要时用隔离 worktree 完成本次任务，不顺手清理他人改动。
