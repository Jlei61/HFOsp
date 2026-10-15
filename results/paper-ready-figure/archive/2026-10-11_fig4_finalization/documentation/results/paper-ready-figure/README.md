# Paper-ready figure outputs

> **当前Fig3已定稿（2026-10-10）：** 正式[完整图和A–E单panel](fig3/figures/README.md)为`visual_alignment_compact_rows_20261010`、`AUTHOR_ACCEPTED_FINAL`。统一入口：[版本说明](../../docs/current_figure3.md)、[机器指针](fig3/current_revision.json)；`build_fig3_current.py`可在远端从随包原生图层重建。旧A–F及旧代码已归档。

> **当前 Fig1：完整A–F已定稿（2026-10-10）。** [完整图、各面板及PPTX](fig1/figures/README.md)为`y1_final_20261010`，入口为[版本指针](fig1/current_revision.json)和[说明](../../docs/current_figure1.md)。来源代码和约13 MB数据部分备份随包保存，旧正式图、全部候选及修订已整体归档。

> **当前 Fig4：三行A–I（作者草图，2026-10-09）。** 读取[机器入口](fig4/current_version.json)或[版本说明](../../docs/current_figure4.md)。B/C/角度响应D/位置密度E组成中排，底排旧G–J改为F–I；A框移至(-8.5,0) mm并避开电极；B/C/D/E/H/I扩大到50 mm正方形，相邻横向留白收紧，B去除两端横轴留白、色条贴近，上下对应列对齐；右侧机制图待补充，整图待作者检查。

这里的顶层只保存当前主图组装包 `fig1/`–`fig5/`、正式补图包 `supp_fig1_*`–`supp_fig7_*`，
以及投稿级补充视频文件。新主图候选放在对应 `figN/candidates/`；其中作者已指定的Fig4 A–I正式发布于`fig4/figures/`，登记为 `CURRENT_AUTHOR_DESIGNATED`；其余 source、诊断和历史模型包放入 `archive/`。

唯一登记入口：[`docs/paper_figure_registry.md`](../../docs/paper_figure_registry.md)。

## 使用规则

1. 引用格式统一为 `asset_id (paper_slot)`，例如
   `ictal_phenotype_to_field_layout (Fig3-A–E)`。
2. `LOCKED` 或作者明确指定的 `CURRENT_AUTHOR_DESIGNATED` 是当前稿件入口；`CANDIDATE` 仍需总拼版或作者指定。
3. `SOURCE` 只提供正式 panel 的素材或 renderer，不能作为独立结论引用。
4. 正式 supplementary 保留在顶层；source、诊断和模型谱系只从 `archive/` 查找。
5. 旧版不删除，统一移到 `archive/<date>_<reason>/`；顶层不再保留多个同义入口。Fig1全部历史在`archive/2026-10-10_fig1_finalization/`；三个未登记的Fig5历史包在`archive/2026-10-10_top_level_historical_packages/`。其他图的当前登记版本及仍在研候选沿用各自状态，不能据本次Figure 1验收升级。
6. 每个实际含图的 `figures/` 目录必须有中文 `README.md`，并指出接受版本和科学边界。
7. 独立 panel 内不画左上角字母；每张主图同时保留一个带字母的 `figN-complete-layout.{png,pdf}`。

## 当前主文入口

| paper slot | asset_id | status | canonical path |
|---|---|---|---|
| Fig1-A–F | `interictal_hfo_temporal_scaffold` | `AUTHOR_ACCEPTED_FINAL` | `fig1/figures/` |
| Fig2-A–F | `interictal_spatial_scaffold` | `CANDIDATE` | `fig2/figures/` |
| Fig3-A–E | `ictal_phenotype_to_field_layout` | `AUTHOR_ACCEPTED_FINAL` | `fig3/figures/` |
| Fig4-A–I | `patient_geometry_prior_snn_compact_a_i` | `CURRENT_AUTHOR_DESIGNATED` | `fig4/figures/` |
| Fig5-A–F | `single_seed_zm_entry_and_onset_field` | `CANDIDATE` | `fig5/figures/` |
Fig1完整A–F已获作者确认，当前采用Y1、18通道显示及紧凑行距。A由真实脑表面、电极坐标及EDF波形生成。Fig1/2/3 的独立 panel 均不含左上角字母；完整排版分别为
`fig1-complete-layout`、`fig2-complete-layout` 和 `fig3-complete-layout`，仅完整排版带 panel 字母。
旧素材与 producer/source 包不作为论文组装入口，正式引用只回到 `fig1/current_revision.json` 指向的目录、`fig2/figures/`、
`fig3/figures/`、`fig4/current_version.json` 指向的目录或 `fig5/figures/`。

Fig1-F 保留自2026-09-02起采用的 v5 配对 inset 统计与表现，当前输出随 `fig1/current_revision.json` 路由：主散点右下比较 single-template 与 multi-cluster MI，显示患者配对连线、均值柱和
paired Wilcoxon 括号；旧灰色 `median ΔMI` 小字版已被替代，不得作为后续拼版或重建来源。

Fig4 当前为三行A–I，独立panel无角标、完整拼图有字母，详见[编号说明](fig4/panel_map.md)。producer为`scripts/paper_figures/build_fig4_compact_ai.py`；B保留第6–16阶段194点，C为四core均值±样本标准差，D为角度三误差响应，E为低损失83/413配置的位置密度；底排沿用旧G/H/I/J并改为F/G/H/I。A框位于(-8.5,0) mm、虚线避开电极；B/C/D/E/H/I扩大到50 mm正方形，相邻横向留白收紧，D/H与E/I逐列对齐，B/C整体与F/G组合对齐；B两端无空白延伸，紧凑行距保持，右侧机制图待补充；科学合同不扩大。旧完整A–J包保留在`archive/2026-10-09_pre_compact_ai_fig4/fig4/`。

## 当前补充视频入口

| paper slot | asset_id | status | canonical path |
|---|---|---|---|
| Supplementary Video 1 | `interictal_single_event_propagation_video` | author-locked | `supplementary-video-1.gif` |
| Supplementary Video 2 | `fig3c_peri_onset_field_evolution_video` | author-locked | `supplementary-video-2.gif` |

Supplementary Video 1 展示 E10 一次 TA 和一次 TB 代表事件在冻结 shared axis/plane 上的 HFO
amplitude-envelope 演化。共 30 帧，生物学步长 2 ms，播放 12.5 fps（80 ms/frame）；中间场使用色盲友好的 soft
teal-to-navy 顺序色图、固定 `PowerNorm(gamma=0.5)` 和每事件完整显示窗 participant-only q99。
最右 TA/TB template field 保持冻结 `viridis` 场，但各自的 rank colorbar 线性归一化为 `0–1`，
端点分别表示 early / late。
它只支持 representative raw-envelope timing cross-check，不是 template-free、cohort、连续组织
traveling-wave 或机制证据。完整合同和 SHA-256 见 `supplementary-video-1_metadata.json`。

Supplementary Video 2 的正式投稿入口为 `supplementary-video-2.gif`，生成与海报 sidecar 位于
`supplementary-video-2-fig3c-peri-onset-field/figures/`。该视频把主图
E10 | SZ3 的发作场（旧Fig3C，现Fig3A场部分）扩展到临床起始前 120 s 至起始后 20 s。动画固定 shared TA plane、
15 个触点、support、6 mm 显示核和 power-z 色标，仅更新 10 s 滑窗内的 1–150 Hz robust-z 场；
下方幅度感知模板表达量 `Q=max(|q_A|,|q_B|)` 用红蓝点标出当时由 TA/TB 主导，并以游标同步当前帧。该视频是单病例动态
配套，不是 onset-emergent alignment、template-free replay 或 cohort 证据。完整合同和 SHA-256 见
`supplementary-video-2_metadata.json`。

## 补图与计算材料

正式 empirical FigS1–FigS5 位于本目录顶层，分别为 `supp_fig1_interictal_event_phenotypes/`、
`supp_fig2_soz_auc/`、`supp_fig3_k_scan_templates/`、`supp_fig4_axis_geometry/` 和
`supp_fig5_multiband_field_concordance/`。原 `supp_fig5_early_seizure_phenotypes/` 已并入主图 Fig3，
`supp_fig6_multiband_field_concordance/` 为编号调整前的历史导出，二者均不再是正式补图入口。
FigS7 位于 `supp_fig7_nlc_pathway_confirmation/`；其 A–D 是从旧 Fig4 supporting source 标准化出来的
连接路径消融图，E 是从主 Fig4 移入的 KMeans heatmap/rank-distribution panel；上游冻结仿真和统计结果不迁移。

尚有编号冲突或未锁定的计算材料继续保存在归档中。当前[Fig5](fig5/README.md)为2026-09-30按作者要求重排的seed9108401空间Z/M A–F图：A为15触点未滤波原始电流读出，保留持续电平，按进入前原始信号P95固定定标，B为蓝色Z/橙色adaptation两行，All E实线/Core A虚线，C为两个E-only raster放大窗，D为四张进入前后原生空间快照，E为同步状态轨迹，F为其下方原模型/患者能量场。A/B/E聚焦8–11秒，保留间期时刻竖线；Rest及Interictal空间图已移除，原70点参数图和19条长时证据单独放在[parameter_scan](fig5/parameter_scan/README.md)。提供无角标单panel PNG/PDF/SVG及带字母完整拼版，[详细图注](fig5/fig5_caption_detailed_zh.md)已同步；自查通过，待作者目视检查。

## 归档

- `archive/2026-09-30_pre_unfiltered_readout_fig5/`：未滤波修改前的30–80 Hz、进入前P95固定定标版及producer快照。
- `archive/2026-09-30_pre_p95_readout_fig5/`：恢复旧图P95定标、撤去新增均值轴之前的Fig5整包。
- `archive/2026-09-30_pre_8to11_readout_fig5/`：缩短到8–11秒、修正变量颜色及新增原始电流窄轴之前的Fig5整包。
- `archive/2026-09-30_pre_transition_focus_fig5/`：SEEG/慢变量/栅格/空间序列系统重排前的完整Fig5，保留70点参数图、19条长随访数据与旧图注。
- `archive/2026-09-28_pre_long_followup_fig5/`：并入19条长时补充之前的59点连续色面版Fig5、单panel及图注整包。
- `archive/2026-09-23_pre_surface_only_fig5/`：E去掉采样点、边界线、斜线与图例之前的59点完整连续图，原文件完整归档。
- `archive/2026-09-23_pre_continuous_boundary_fig5/`：E改为59点完整连续边界之前的paper-ready A–F整包，保留11/24加密点的历史快照与全部原文件。
- `archive/2026-09-18_pre_single_seed_fig5/`：作者指定新A–F Fig5之前的seed1801 A–D v5整包，5个原文件字节保持不变。
- `archive/2026-08-09_stale_aliases/`：第一批已撤回、被替代或重复的顶层入口。
- `archive/2026-08-09_unused_subject_snn_reruns/`：三套无仓库引用的 E1146 参数重跑输出。
- `archive/2026-08-09_fig1_pre_panel_contract/`：Figure 1 旧 panel 字母与联合输出。
- `archive/2026-08-09_fig1_source_material/`：Figure 1B 的原始 HFO 素材。
- `archive/2026-08-09_fig2_pre_panel_contract/`：Figure 2 旧分散目录与 E+F 联合输出。
- `archive/2026-08-13_non_main_figure_packages/`：现保留 54 个 Fig3/Fig4 source、Fig5/Fig6 候选、
  Topic 4 诊断图和历史模型谱系；六个正式 `supp_fig*` 已于 2026-08-19 恢复至顶层。
- `archive/2026-08-19_pre_a_g_fig4/`：Fig4 从旧 A–H 改为 data-driven A–G 前的完整可恢复快照。
- `archive/2026-08-19_rejected_a_g_fig4/`：按错误 A–G 语义生成、随后撤回的版本；仅保留用于恢复与 provenance。
- `archive/2026-09-03_pre_combined_fig4_ab/`：本次将原 Fig4A/B 合并为新 Panel A 前的完整 A–H 素材包；包含原始独立 panel、旧完整拼板和过渡候选，仅用于恢复与 provenance。
- `archive/2026-09-03_pre_fig4_a_e_reorder/`：Fig4 从 A–G 精简重排前的完整快照；其中 masked-rank KMeans panel 已移入 FigS7-E，cross-fit 保留在主图。
- `archive/2026-09-03_pre_fig4_reserved_panel_b/`：B 预留为空之前的 A–F canonical 快照；原 B–F 内容在当前主图中原样顺延为 C–G。
- `archive/2026-09-02_noncanonical_staging_qa_cleanup/`：本轮从顶层移走的 Fig2–4 producer staging、comparison、source 和 typography visual-QA 包。
- 归档是路径迁移，不是删除；旧图只能用于 provenance，不再参与当前稿件组装。
