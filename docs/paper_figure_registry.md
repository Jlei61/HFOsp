# Paper-ready figure 唯一登记表

> 状态：v4，2026-08-31。本文是 `results/paper-ready-figure/` 的唯一指代入口。
> 论文 panel 编号、资产语义、物理路径和科学状态必须分开记录；不能再仅凭目录名里的
> `fig2` / `fig3b` / `fig6` 判断它现在属于哪张论文图。

## 1. 指代规则

每个可引用图包使用四个字段：

1. `asset_id`：稳定语义 ID。只描述图在展示什么，不包含可能变化的 panel 编号。
2. `paper_slot`：当前论文位置，例如 `Fig3-A`。位置未锁定时写 `TBD`。
3. `status`：科学和出版状态。
4. `canonical_path`：当前唯一可消费路径。

状态只允许以下六类：

- `LOCKED`：科学合同和可视版本均已锁定，可进入当前稿件。
- `CANDIDATE`：可作为候选，但总拼版或 panel 位置尚未最终锁定。
- `SUPPLEMENT`：只进入补图或补充计算材料。
- `SOURCE`：正式 panel 的源素材或公共 renderer 输出，不能单独当论文结论引用。
- `HISTORICAL`：已撤回、被替代或仅保留模型谱系；只能从 `archive/` 引用。
- `LAYOUT_INCOMPLETE_RESERVED_PANEL_B`：作者已锁定现有 panel 与排版，但 B 明确预留，尚不能作为完整投稿图。

引用时优先写 `asset_id (paper_slot)`，例如
`ictal_field_scaffold (Fig3-A–F)`，不要只写“fig3 图”或直接沿用脚本内部的 `fig3c_*` 名称。

## 2. 当前主文资产映射

| asset_id | paper_slot | status | canonical_path | producer / 说明 |
|---|---|---|---|---|
| `interictal_hfo_temporal_scaffold` | Fig1-B–F | `LOCKED` | `results/paper-ready-figure/fig1/figures/` | 无角标的 B1/B2、C、D、E、F 独立输出 + 带 B–F 角标的 `fig1-complete-layout`；Fig1-A 为作者手绘。Fig1-F 自 2026-09-02 锁定为 v5：主散点右下必须是 single-template vs multi-cluster MI 配对 inset（患者连线、均值柱、paired Wilcoxon 括号），不得退回灰色 median ΔMI 小字 |
| `interictal_spatial_scaffold` | Fig2-A–F | `CANDIDATE` | `results/paper-ready-figure/fig2/figures/` | B/E/F 默认读取 `all_event_timing_plus_space_no_hard_qc_v1`：B 为 26 人 held-out 方向比较，E 为 4 例显示，F 为完整 18 人 shared-plane 队列；A/C/D 保持原 canonical 版本 |
| `ictal_field_scaffold` | Fig3-A–F | `LOCKED` | `results/paper-ready-figure/fig3/figures/` | A/B 保持原 raw/TFR 与四频带 signal context；C–F 默认读取 `all_event_timing_plus_space_no_hard_qc_v1` 重算源，D 为 n=17/16/11，F 为 17-subject A/B heatmap |
| `data_driven_interictal_snn_fig4` | Fig4-A–G | `LAYOUT_INCOMPLETE_RESERVED_PANEL_B` | `results/paper-ready-figure/fig4/figures/` | A、C–G 有无角标独立 PNG/PDF，完整拼板带 A–G 角标。A 为 local E/I circuit + patient-specific substrate；B 使用右上现有留白并明确预留，当前无独立文件，后续用于 data-driven 参数对患者间期事件复现的影响；C 为 Node field + Model TA/MTB；D 为 rank profiles；E 为 cross-fit；F 为 recruitment-onset-span readout；G 为 34 人 cohort。原 KMeans heatmap/rank distribution 保留在 FigS7-E。 |
| `data_driven_snn_nlc_pathway_confirmation` | FigS7-A–E | `SUPPLEMENT` | `results/paper-ready-figure/supp_fig7_nlc_pathway_confirmation/figures/` | A–D 为 1581--1592 共 12 张全新配对网络的冻结 local-connectivity ablation；E 为从主 Fig4 移入的 627-event masked-rank KMeans heatmap、逐触点 rank distribution 与共享色条。该补图只支持 development-case 模型内部 pathway effect 与事件结构。 |
| `data_driven_snn_dual_mode_validation` | Fig4-C historical candidate | `DIAGNOSTIC_ONLY` | `results/topic4_sef_hfo/data_driven_core_field_rev10_d/spatial_ou_accessibility_d5_2_confirmation/figures/` | 连续场 + MTA/MTB same-network readout 和 KMeans 核验图；natural KMeans 未复现患者 TA/TB（direction purity `0.674` < patient-matched q05 `0.884`，pooled MTB↔TB `−0.60`），已被独立冻结终点的 NLC pathway panel 替代，不得再作为当前 Fig.4C source |
| `data_driven_snn_d6_3_replication_diagnostic` | Fig4-C source（诊断，不可替换主文） | `DIAGNOSTIC_ONLY` | `results/topic4_sef_hfo/data_driven_core_field_rev10_d/continuous_field_kmeans_d6_3_fresh_replication/figures/` | 同规格第二套图，冻结连续场候选 `d62_a0p5_b0p5` 在 12 张全新网络上的复制臂；verdict `REV10D6_3_JOINT_CONTINUOUS_FIELD_NOT_REPLICATED`（patient cross-fit paired delta `−0.107`、K=2 支持仅 `4/12`）。图为 pooled 展示，不覆盖网络级复制失败，**不得替换主文 Fig.4** |
| `core_model_s3_brakeoff` | Fig5-A working slot | `CANDIDATE` | `results/paper-ready-figure/archive/2026-08-13_non_main_figure_packages/fig5_core_model_s3_brakeoff/figures/` | 尚未形成 canonical `fig5/`，作为工作材料归档；Fig5/6 机制线尚未整体收口 |

## 3. 当前补图 / 计算材料

| asset_id | paper_slot | status | canonical_path | 说明 |
|---|---|---|---|---|
| `interictal_event_phenotypes` | FigS1 | `SUPPLEMENT` | `results/paper-ready-figure/archive/2026-08-13_non_main_figure_packages/supp_fig1_interictal_event_phenotypes/figures/` | legacy 人工标注事件验证集 |
| `soz_auc` | FigS2 | `SUPPLEMENT` | `results/paper-ready-figure/archive/2026-08-13_non_main_figure_packages/supp_fig2_soz_auc/figures/` | raw vs synchronized SOZ AUC |
| `k_scan_templates` | FigS3 | `SUPPLEMENT` | `results/paper-ready-figure/archive/2026-08-13_non_main_figure_packages/supp_fig3_k_scan_templates/figures/` | masked K=2–10 扫描与高阶示例 |
| `axis_geometry` | FigS4 | `SUPPLEMENT` | `results/paper-ready-figure/archive/2026-08-13_non_main_figure_packages/supp_fig4_axis_geometry/figures/` | axis direction / held-out readback |
| `early_seizure_phenotypes` | FigS5 | `SUPPLEMENT` | `results/paper-ready-figure/archive/2026-08-13_non_main_figure_packages/supp_fig5_early_seizure_phenotypes/figures/` | early-ictal spectral phenotype + gamma example |
| `multiband_field_concordance` | FigS6 empirical | `SUPPLEMENT` | `results/paper-ready-figure/archive/2026-08-13_non_main_figure_packages/supp_fig6_multiband_field_concordance/figures/` | 与 dense-grid method-sensitivity 输出同源；只保留此投稿命名入口 |
| `static_contact_topography` | FigS6 computational | `SUPPLEMENT` | `results/paper-ready-figure/archive/2026-08-13_non_main_figure_packages/fig6_static_contact_topography/figures/` | manuscript-facing RNN 计算材料之一，当前不与主图并列 |
| `ordered_history_architecture_audit` | FigS6 computational companion | `SUPPLEMENT` | `results/paper-ready-figure/archive/2026-08-13_non_main_figure_packages/fig6_ordered_history_architecture_audit/figures/` | bounded-history / architecture audit companion |

注意：当前草稿同时把 empirical multiband figure 和 computational RNN account 称为
“FigS6”。这是仍待作者层面解决的编号冲突；本登记表保留两个稳定 `asset_id`，不伪造一个已经锁定的
补图编号。

## 4. 不能由目录名直接升级的图

- `fig6_*` 的多数目录是模型迭代谱系，不等于当前主文 Figure 6。
  `fig6_structured_rank_rnn`、`fig6_persistent_path_mode_bounded_negative`、
  `fig6_symmetric_axis_propagation_state_v2_2` 等只能按各自 bounded / historical 合同引用。
- `fig_m3a_*`、`fig_m4_dynamic_qi`、`fig_stage4_*` 是机制筛选或 visual diagnostic。
  即使目录位于 `paper-ready-figure/`，也不能自动升级成 paper-facing panel。
- `fig3a_raw_spectral_context/`、`fig3b_interictal_ictal_shared_field/`、
  `fig3-sup-tspectral-field-concordance/`、`fig3_peri_onset_field_similarity/` 和
  `fig3f_ab_dominance_cohort/` 现在都是 Fig3 canonical panel 的 producer/source 包；论文组装只消费
  `fig3/figures/`。
- `fig3_ictal_field_concordance_grid_rebuild/` 与
  `fig3_ictal_field_concordance_grid_method_sensitivity/` 是计算与审计包；正式图应引用登记表中的
  panel 资产，而不是把整个 staging 目录叫作“Figure 3”。
- `fig4_subject_snn_e1146/`、`fig_subject_snn_epilepsiae_1146*/` 和其余
  `fig4_data_driven_core_field_*` 是 Figure 4 的 source/迭代包；论文组装只消费
  `fig4/figures/`。当前 A–G 中 B 预留为空，A、C–F 来自冻结 E10 development case，G 来自
  34 人 cohort formal result；该视觉合同不扩大 scientific claim。

## 5. 2026-08-09 第一批归档

以下顶层入口已移动到
`results/paper-ready-figure/archive/2026-08-09_stale_aliases/`，未删除：

| 原目录 | 原因 | 当前替代 |
|---|---|---|
| `fig2b_interictal_ictal_shared_field/` | 旧 Fig2-B 试制，语义已迁移并重新锁定 | `ictal_field_scaffold (Fig3-C)` |
| `fig3_raw_spectral_context/` | 未登记的过渡输出，含多个 seizure 版本 | `ictal_field_scaffold (Fig3-A/B)` |
| `fig3_sup2_raw_spectral_context/` | 与正式 raw spectral context 存在逐字节重复的历史别名 | `ictal_field_scaffold (Fig3-A/B)` |
| `fig3_sup_peri_onset_field_similarity/` | 旧单病例 supplement 路径 | `ictal_field_scaffold (Fig3-E)` |
| `fig3_sup_preictal_field_similarity/` | 旧 preictal-only 路径，已被统一窗口合同替代 | `ictal_field_scaffold (Fig3-E)` |
| `fig3f_shared_ab_dominance/` | 旧 shared-only 1/3 分母图，不是当前 17 人 Fig3-F 包 | `ictal_field_scaffold (Fig3-F)` |

另有三套无任何仓库引用的 E1146 2026-07-06 参数重跑输出移至
`results/paper-ready-figure/archive/2026-08-09_unused_subject_snn_reruns/`。有 sensitivity 合同或
archive 证据引用的 E1146 变体仍保留在顶层，未按文件名相似性一并移动。

归档目录只保留 provenance。任何 producer 若重新生成这些旧顶层路径，应视为路径回归错误，而不是恢复
canonical 输出。

## 6. 2026-08-09 Figure 1/2 panel 合同归档

- Figure 1 的旧 `panela*`、SOZ ROC 误占 `panelb*`、C/E 联合文件以及 `d1/d2` 命名，已整体移到
  `archive/2026-08-09_fig1_pre_panel_contract/`；B 的原始 HFO 素材移到
  `archive/2026-08-09_fig1_source_material/`。Fig1-A 是手绘，不在代码输出中保留。
- Figure 2 的 projection、direction、event-envelope、template-field 与 E+F 联合行旧目录，已移到
  `archive/2026-08-09_fig2_pre_panel_contract/`。当前唯一 panel 入口为 `fig2/figures/fig2-panel{a..f}`；
  Fig2-A 固定使用局部双电极杆三维几何、shared-plane 投影和 TA/TB 场的 composite producer，不使用 MRI 三步示意图。
- 独立 panel 文件内部不画左上角字母；每张主图另输出 `fig1-complete-layout` / `fig2-complete-layout`，
  panel 字母只出现在完整排版版。
- 统一复现入口：`scripts/paper_figures/build_main_figures_1_2.py`。Fig2-C v10 为 4 个严格等间距
  frame（`0/+16/+32/+48 ms`），要求最终二维场使用的全部参与触点质心和 top-3 热点每一步均相反移动；静态中间场逐帧按 participant top-3 mean 显示相对包络并固定使用 `PowerNorm(gamma=0.5)`，GIF 则冻结 per-event complete-window q99，默认从已验收矢量 PDF
  重新导出 600 dpi PNG；只有显式 `--recompute-fig2c` 才重新读取原始 EEG。

## 7. 2026-08-13 非主图顶层归档

原顶层 60 个 supplementary、source、候选、诊断和历史模型目录已完整移动到
`results/paper-ready-figure/archive/2026-08-13_non_main_figure_packages/`。顶层现在只保留 canonical
`fig1/`–`fig4/`；尚未锁定的 Fig5/Fig6 不以含混目录名占据主图入口。

Figure 3/4 builder 已同步读取归档 source，因此归档不影响当前主图重建。后续 producer 若再次向顶层生成
`fig3_*`、`fig4_*`、`fig6_*` 或 `supp_fig*`，应视为输出路径回归；正式锁定新主图时统一建立
`fig5/` 或 `fig6/`，而不是恢复旧候选目录。

## 8. Figure 2/3 当前数据注册

`config/paper_figure_source_registry.json` 是 Fig2-B/E/F 与 Fig3-C/D/E/F 的唯一 tracked 数据入口。当前 active contract 为 `all_event_timing_plus_space_no_hard_qc_v1`；旧 hard-QC 与 timing-only 路径均登记为 historical，禁止默认 fallback。任何对话或脚本在重画前应先运行：

```bash
python scripts/paper_figures/paper_figure_source_registry.py --figure all
```

若 active artifact 缺失或 SHA-256 不匹配，命令直接失败，不会静默使用旧结果。
