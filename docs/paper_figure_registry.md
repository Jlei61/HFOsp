> **2026-10-27 当前新 Fig5：** 作者指定 `vertical_legend_20261027` 空间 SNN 传播与 core 对照，见 [当前 Fig5](current_figure5.md)；旧 Z/M Figure 5 改登记为 [Fig6](current_figure6.md)。新 Fig4 由独立任务维护，本次不替换 Fig4。

> **2026-09-30旧图整合入口（现为 Figure 6）**：[原A–F版](current_figure6.md)，已纳入70点及19条长随访。下面任何旧seed1801 v5入口仅为历史。空间与六群体分岔分别见[模型版本表](topic4_model_versions.md)。其他图的路由不受本次整合影响。

> **当前 Fig4：作者确认定稿（2026-10-11）。** `core_search_final_20261011`、`AUTHOR_ACCEPTED_FINAL`；正式完整图及A–I单图统一在`fig4/figures/`。见[当前说明](current_figure4.md)与[机器入口](../results/paper-ready-figure/fig4/current_version.json)。A为已确认v15：三个问号区域、两个core、上方两条OPTM箭头，浅绿底纹仅在中央取样框；右侧保留局部高斯采样权重。其余面板及三行紧凑布局保持。正式入口`build_fig4_current.py`验证并恢复冻结导出；旧正式版和58个候选已归档，见[历史登记](archive/figures/figure4_revision_history_20261011.md)。

> **Figure 1已定稿（2026-10-10，`y1_final_20261010`）。** [当前版本](current_figure1.md)和`fig1/current_revision.json`为唯一入口；完整图、A–F独立PNG/PDF及7页PPTX位于`fig1/figures/`。当前producer为`build_fig1_current.py`；冻结代码、约13 MB必要科学输入及原始来源随包保存。旧正式图、候选及修订全部归档至`archive/2026-10-10_fig1_finalization/fig1/`。定稿完整PNG/PDF已逐字节复现。


# Paper-ready figure 唯一登记表

> **当前Fig3：作者已确认的A–E正式版（2026-10-10）。** [完整图及独立面板](../results/paper-ready-figure/fig3/figures/README.md)为`visual_alignment_compact_rows_20261010`、`AUTHOR_ACCEPTED_FINAL`；[当前说明](current_figure3.md)与[机器入口](../results/paper-ready-figure/fig3/current_revision.json)为统一入口。A为E10/SZ3，B为Y1/SZ6固定0–10 s、30–80 Hz；C/D/E承接原D、原E右半、原F。完整可见列宽包含色条及标签，两处行距均4.064 mm，画布308.11×254.24 mm。正式代码`build_fig3_current.py`从随包冻结原生图层重建，无需原始数据盘；旧A–F及原组装代码已归档，不能覆盖本版。

> 状态：v9，2026-08-31。本文是 `results/paper-ready-figure/` 的唯一指代入口。
> 论文 panel 编号、资产语义、物理路径和科学状态必须分开记录；不能再仅凭目录名里的
> `fig2` / `fig3b` / `fig6` 判断它现在属于哪张论文图。

## 1. 指代规则

每个可引用图包使用四个字段：

1. `asset_id`：稳定语义 ID。只描述图在展示什么，不包含可能变化的 panel 编号。
2. `paper_slot`：当前论文位置，例如 `Fig3-A`。位置未锁定时写 `TBD`。
3. `status`：科学和出版状态。
4. `canonical_path`：当前唯一可消费路径。

状态按以下类别记录：

- `AUTHOR_ACCEPTED_FINAL`：作者明确确认并指定为当前正式版本。
- `LOCKED`：科学合同和可视版本均已锁定，可进入当前稿件。
- `CANDIDATE`：可作为候选，但总拼版或 panel 位置尚未最终锁定。
- `SUPPLEMENT`：只进入补图或补充计算材料。
- `SOURCE`：正式 panel 的源素材或公共 renderer 输出，不能单独当论文结论引用。
- `HISTORICAL`：已撤回、被替代或仅保留模型谱系；只能从 `archive/` 引用。

引用时优先写 `asset_id (paper_slot)`，例如
`ictal_phenotype_to_field_layout (Fig3-A–E)`，不要只写“fig3 图”或直接沿用脚本内部的 `fig3c_*` 名称。

## 2. 当前主文资产映射

| asset_id | paper_slot | status | canonical_path | producer / 说明 |
|---|---|---|---|---|
| `interictal_hfo_temporal_scaffold` | Fig1-A–F | `AUTHOR_ACCEPTED_FINAL` | `results/paper-ready-figure/fig1/figures/` | Y1完整定稿，2026-10-10作者接受；代码、数据部分备份及旧版归档见`docs/current_figure1.md`。 |
| `interictal_spatial_scaffold` | Fig2-A–F | `CANDIDATE` | `results/paper-ready-figure/fig2/figures/` | 无角标 A–F 独立输出 + 带 A–F 角标的 `fig2-complete-layout`。2026-09-02 保持原拼板位置，仅收紧 A 的 2×2 内部留白、放大 B 主体文字、将 C 行标题改为 `TA samples` / `TB samples`；D 保持左侧 `TA field` / `TB field`。B/E/F 仍使用 all-event Timing+Space 合同 |
| `interictal_projection_typography` | **Fig2当前完整A–E排版候选** | `PENDING_AUTHOR_VISUAL_REVIEW` | `results/paper-ready-figure/fig2/candidates/projection_typography_20261023/figures/` | 2026-10-23按作者要求将A改为2D projection并把箭头放入虚线投影区域；C的rose显示直径108→124 pt，圆内图例与原灰柱/红蓝实线保留。A/D/E的坐标标签及数字刻度按B的13.6/11.05 pt统一，A密集通道名称保留行距对应字号。D/E等高且xlabel基线一致，底部为增大字体留出间距；保留收窄画布宽度1000.7 pt。B导出不变，E统计逐项核对，A/D科学图层保持。producer为`build_fig2_projection_typography.py`，输出同态PDF/600dpi PNG及单图，待作者目视验收。 |
| `ictal_phenotype_to_field_layout` | Fig3-A–E | `AUTHOR_ACCEPTED_FINAL` | `results/paper-ready-figure/fig3/figures/` | 2026-10-10作者正式接受；E10/SZ3 broadband与Y1/SZ6 gamma两行均为raw/TFR、早期能量场、间期TA场。Y1固定0–10 s、30–80 Hz、自身TA平面、Y=−20～20 mm；C/D/E为原D、原E右半、原F。含色条的完整列宽对齐，行距4.064 mm。代码、来源与远端重建见`docs/current_figure3.md`。 |
| `ictal_field_scaffold` | 历史Fig3-A–F | `HISTORICAL` | `results/paper-ready-figure/archive/2026-10-10_pre_current_fig3_ae/fig3/figures/` | 2026-10-10被当前A–E正式版替代；旧独立面板、整图、metadata和组装代码保留供追溯。旧图号不是当前panel身份。 |
| `data_driven_interictal_snn_fig4` | Fig4-A–G | `LAYOUT_INCOMPLETE_RESERVED_PANEL_B` | `results/paper-ready-figure/fig4/figures/` | 无角标 A、C–G 独立 PNG/PDF + 带 A–G 角标的 `fig4-complete-layout`；B 使用右上现有留白并明确预留，当前无独立文件，后续用于 data-driven 参数对患者间期事件复现的影响。A 为已确认的 local E/I circuit + patient-specific substrate；C 为 Node field + Model TA/MTB；D 为模型/患者 rank profile；E 为 cross-fit matrix；F 为 recruitment-onset-span readout；G 为 34 人 formal cohort。原 KMeans heatmap/rank distribution 保留在 FigS7-E。科学口径不扩大 |
| `patient_geometry_prior_snn_compact_a_i` | **Fig4-A–I 正式定稿** | `AUTHOR_ACCEPTED_FINAL` | `results/paper-ready-figure/fig4/figures/` | 2026-10-11作者确认；core_search_final_20261011，A采用v15，B–I冻结数据及导出保持。正式入口build_fig4_current.py；旧版均登记为历史，见docs/current_figure4.md。 |
| `data_driven_interictal_snn_fig4` | Fig4-A–G historical | `HISTORICAL_SOURCE_ONLY` | `results/paper-ready-figure/archive/2026-10-09_pre_remote_compact_ai/fig4/figures/` | 无角标 A、C–G 独立 PNG/PDF + 带 A–G 角标的 `fig4-complete-layout`；B 使用右上现有留白并明确预留，当前无独立文件，后续用于 data-driven 参数对患者间期事件复现的影响。A 为已确认的 local E/I circuit + patient-specific substrate；C 为 Node field + Model TA/MTB；D 为模型/患者 rank profile；E 为 cross-fit matrix；F 为 recruitment-onset-span readout；G 为 34 人 formal cohort。原 KMeans heatmap/rank distribution 保留在 FigS7-E。科学口径不扩大 |
| `data_driven_snn_dual_mode_validation` | Fig4-C historical candidate | `DIAGNOSTIC_ONLY` | `results/topic4_sef_hfo/data_driven_core_field_rev10_d/spatial_ou_accessibility_d5_2_confirmation/figures/` | 连续场 + MTA/MTB same-network readout 和 KMeans 核验图；natural KMeans 未复现患者 TA/TB（direction purity `0.674` < patient-matched q05 `0.884`，pooled MTB↔TB `−0.60`），已被独立冻结终点的 NLC pathway panel 替代，不得再作为当前 Fig.4C source |
| `data_driven_snn_d6_3_replication_diagnostic` | Fig4-C source（诊断，不可替换主文） | `DIAGNOSTIC_ONLY` | `results/topic4_sef_hfo/data_driven_core_field_rev10_d/continuous_field_kmeans_d6_3_fresh_replication/figures/` | 同规格第二套图，冻结连续场候选 `d62_a0p5_b0p5` 在 12 张全新网络上的复制臂；verdict `REV10D6_3_JOINT_CONTINUOUS_FIELD_NOT_REPLICATED`（patient cross-fit paired delta `−0.107`、K=2 支持仅 `4/12`）。图为 pooled 展示，不覆盖网络级复制失败，**不得替换主文 Fig.4** |
| `spatial_snn_propagation_and_core_count` | Fig5-A–F（旧 Fig4 拆分） | `PENDING_AUTHOR_VISUAL_REVIEW` | `results/paper-ready-figure/fig5/figures/` | 2026-10-27：vertical_legend_20261027；A–E 保留 xy_left_20、11 pt、等高两排与 D 共同 0/16/32/64/96 ms。F 保留真实 TA/TB 红蓝身份：17 位 shared axis 患者、前 16/32 次搜索，单 core 在 ρTA>ρTB 约束下取最低 J（7 人换点），双 core 保持原最低 J；探索性两项 Holm P=0.051/0.019。F 仅对显著比较显示星号及括号（TB *），不标 n.s.，横轴 TA–MTA/TB–MTB 各组居中，1 core/2 cores 图例在右上方竖排，星号下移且无重叠，样本量与中期状态只写说明；本版为非对称选点策略对照，新点未独立复测。见 docs/current_figure5.md。 |
| `single_seed_zm_entry_and_onset_field` | Fig6-A–F（原 Fig5） | `CANDIDATE` | `results/paper-ready-figure/fig6/figures/` | 2026-09-30更新：A为15触点未滤波原始电流读出，保留持续电平，按进入前原始信号P95固定定标；B为蓝色Z、橙色adaptation，All E实线/Core A虚线，A/B/E共用8–11秒；C为8.370–8.670和9.370–9.670秒E-only raster；D为9.420/9.480/9.600/10.370秒的原生50 ms空间图；E状态标记同步；F仍为原模型/患者能量场。本图为历史手放核场，Fig4 F2+Z/M另行单种子运行。原70点参数图留于parameter_scan。当前编号校验入口为build_fig6_current.py，历史producer为build_fig5_transition_focus.py；图注、读出诊断、PNG/PDF/SVG及单panel包已自查，待作者目视检查 |
| `data_driven_zm_transition_fig5` | Fig5-A–D historical | `HISTORICAL` | `results/paper-ready-figure/archive/2026-09-30_pre_integration_fig5/fig5/figures/` | 原seed1801 v5原样归档，已由当前A–F替代；其 `NOT_SUSTAINED_ICTAL_MORPHOLOGY` 标签只属于这个历史版本，不移植给当前图 |

## 3. 当前补图 / 计算材料

| asset_id | paper_slot | status | canonical_path | 说明 |
|---|---|---|---|---|
| `interictal_single_event_propagation_video` | Supplementary Video 1 | `SUPPLEMENT` | `results/paper-ready-figure/supplementary-video-1.gif` | 2026-08-19 作者锁定，2026-08-30 v2 将最右 template-rank colorbar 统一为各模板内归一化 `0–1`。E10 单次 TA/TB 代表事件，30 帧、2 ms biological step、12.5 fps（80 ms/frame）；soft teal-to-navy HFO amplitude envelope + 冻结 template-rank field。只支持 frozen-axis 上的 representative raw-envelope timing cross-check，不支持 template-free、cohort、continuous-tissue traveling wave 或机制结论 |
| `model_interictal_propagation_fit_comparison_video` | Supplementary Video 2 | `AUTHOR_ACCEPTED_FINAL` | `results/paper-ready-figure/supplementary-video-2.gif` | 2026-10-15 作者确认，`paired_rows_20261015`。高误差／较差拟合在上、低误差／较好拟合在下；每行依次为 MTA、MTB 包络及各自动态场，上下对应，醒目条件标签替代内部编号。沿用原事件、15 触点顺序、模型几何、0–1 包络和 0–75 原生场色标；各自完整 250 ms 窗、2 ms 步长、125 帧、20 fps。时间各自相对最早有效质心。正式包和版本指针见 [当前说明](current_supplementary_video2.md)；静态配套不编号为 FigS2 |
| `fig3c_peri_onset_field_evolution_video` | 原 Supplementary Video 2（历史，当前无编号） | `HISTORICAL_SUPERSEDED` | `results/paper-ready-figure/archive/2026-10-15_supplementary_video2_replacement/supplementary-video-2.gif` | 原 E10/SZ3 围发作场动画，2026-10-15 由模型传播对比动画接替 Video 2。旧视频及元数据逐字节归档，不自动改为 Video 3 |
| `interictal_event_phenotypes` | FigS1 | `SUPPLEMENT` | `results/paper-ready-figure/supp_fig1_interictal_event_phenotypes/figures/` | legacy 人工标注事件验证集；作者已锁定为正式补图入口 |
| `soz_auc` | FigS2 | `SUPPLEMENT` | `results/paper-ready-figure/supp_fig2_soz_auc/figures/` | raw vs synchronized SOZ AUC；作者已锁定为正式补图入口 |
| `k_scan_templates` | FigS3 | `SUPPLEMENT` | `results/paper-ready-figure/supp_fig3_k_scan_templates/figures/` | masked K=2–10 扫描与高阶示例；作者已锁定为正式补图入口 |
| `axis_geometry` | FigS4-A–D | `SUPPLEMENT` | `results/paper-ready-figure/supp_fig4_axis_geometry/figures/` | paired timing-versus-timing+space direction score / axis direction / held-out readback / clinical SOZ all-contact compactness；正式补图入口 |
| `early_seizure_phenotypes` | absorbed into Fig3-A/B | `OMITTED_FROM_SUPPLEMENT` | `results/paper-ready-figure/supp_fig5_early_seizure_phenotypes/figures/` | early-ictal spectral phenotype + gamma example 已进入主图 Fig3，不再作为独立补图；旧产物仅保留追溯 |
| `multiband_field_concordance` | FigS5 | `SUPPLEMENT` | `results/paper-ready-figure/supp_fig5_multiband_field_concordance/figures/` | 与 dense-grid method-sensitivity 输出同源；正式补图入口 |
| `data_driven_snn_nlc_pathway_confirmation` | FigS7-A–E | `SUPPLEMENT` | `results/paper-ready-figure/supp_fig7_nlc_pathway_confirmation/figures/` | A–D 为 1581--1592 共 12 张全新配对网络的冻结 local-connectivity ablation；E 为从主 Fig4 移入的 627-event masked-rank KMeans heatmap、逐触点 rank distribution 与共享色条。独立 panel 无角标，完整拼板带 A–E。定位仍是 development-only 的模型内部 pathway effect 与事件结构，不是患者因果连接机制或患者盲泛化 |
| `static_contact_topography` | FigS6 computational | `SUPPLEMENT` | `results/paper-ready-figure/archive/2026-08-13_non_main_figure_packages/fig6_static_contact_topography/figures/` | manuscript-facing RNN 计算材料之一，当前不与主图并列 |
| `ordered_history_architecture_audit` | FigS6 computational companion | `SUPPLEMENT` | `results/paper-ready-figure/archive/2026-08-13_non_main_figure_packages/fig6_ordered_history_architecture_audit/figures/` | bounded-history / architecture audit companion |

编号说明：empirical multiband figure 已前移为 FigS5；computational RNN account 保留 FigS6，原编号冲突已解除。

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
  `fig4_data_driven_core_field_*` 是 Figure 4 的 source/迭代包；论文组装只消费 `fig4/figures/`。
  当前 A–G 已按作者 2026-09-03 最新顺序重排并有完整 producer；B 预留为空，A、C–F 来自
  `frozen_substrate_confirmation`，G 来自 34 人 cohort formal result。2026-08-19 的误改 A–G 版本已可恢复地归档到
  `results/paper-ready-figure/archive/2026-08-19_rejected_a_g_fig4/fig4/`；改动前 A–H 快照仍保存在
  `results/paper-ready-figure/archive/2026-08-19_pre_a_g_fig4/fig4/`。本次合并 A/B 前的完整 A–H 素材包保存在
  `results/paper-ready-figure/archive/2026-09-03_pre_combined_fig4_ab/fig4/`；改成本次 A–E 顺序前的 A–G 快照在
  `results/paper-ready-figure/archive/2026-09-03_pre_fig4_a_e_reorder/fig4/`；预留 B 之前的 A–F 快照位于
  `results/paper-ready-figure/archive/2026-09-03_pre_fig4_reserved_panel_b/fig4/`。pathway confirmation 与 KMeans panel 已整理为
  `data_driven_snn_nlc_pathway_confirmation (FigS7-A–E)`；其冻结上游结果仍保留在 Topic 4
  scientific root 中作为 provenance。`data_driven_snn_dual_mode_validation` 仍是历史诊断，不能替换当前主图。

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
  Fig2-A 为 2×2：作者提供的 Y9 植入概览、E10 subject-specific T1 局部 cutaway、
  真实局部电极几何和含 6 mm Gaussian support 范围的二维投影；四格等宽等高、标题居中且不画流程箭头。Y9 overview
  内的红/蓝方向 glyph 不延伸到 E10 三块；Y9 与 E10 不得写成同一病例连续 zoom。
- 独立 panel 文件内部不画左上角字母；每张主图另输出 `fig1-complete-layout` / `fig2-complete-layout`，
  panel 字母只出现在完整排版版。
- 统一复现入口：`scripts/paper_figures/build_main_figures_1_2.py`。Fig2-C v12 为 4 个严格等间距
  frame（`0/+16/+32/+48 ms`），要求最终二维场使用的全部参与触点质心和 top-3 热点每一步均相反移动；静态中间场逐帧按 participant top-3 mean 显示相对包络并固定使用 `PowerNorm(gamma=0.5)`，GIF 则冻结 per-event complete-window q99，默认从已验收矢量 PDF
  重新导出 600 dpi PNG；只有显式 `--recompute-fig2c` 才重新读取原始 EEG。
- Fig2-C GIF 已锁定为 `interictal_single_event_propagation_video (Supplementary Video 1)`；正式入口
  直接位于 paper-ready 根目录，`fig2/figures/fig2-panelc.gif` 只作同字节 source sidecar。

## 7. 2026-08-13 非主图顶层归档

原顶层 60 个目录曾完整移动到
`results/paper-ready-figure/archive/2026-08-13_non_main_figure_packages/`。2026-08-19 根据作者纠正，已将
正式 FigS1–FigS7 七个 `supp_fig*` 包已恢复为顶层 canonical supplementary 入口。2026-09-02 新增明确的
`fig5/` 最新候选入口，并把其余 producer staging、source、comparison 与 visual-QA 包移入日期归档。

Figure 3/4 builder 已同步读取归档 source，因此归档不影响当前主图重建。后续 producer 若再次向顶层生成
`fig3_*`、`fig4_*`、`fig6_*` 或 `supp_fig*`，应视为输出路径回归；正式锁定新主图时统一建立
`fig5/` 或 `fig6/`，而不是恢复旧候选目录。

## 8. 2026-08-31 all-event Timing+Space 主图更新

Fig2-B/E/F 与 Fig3-C/D/E/F 已写入各自 canonical `figures/` 目录，并在同一状态下重建两张 complete layout。更新前的 Fig2/Fig3 完整图包保存在 `results/paper-ready-figure/archive/2026-08-31_pre_all_event_timing_plus_space/`；归档包含逐 panel、完整拼版、README、metadata、registry 与随图 null 资产，可整体恢复。
