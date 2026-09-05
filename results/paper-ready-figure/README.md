# Paper-ready figure outputs

这里的顶层只保存当前主图组装包 `fig1/`–`fig5/`、正式补图包 `supp_fig1_*`–`supp_fig7_*`，
以及投稿级补充视频文件。source、候选、诊断和历史模型包放入 `archive/`，不与正式稿件入口并列。

唯一登记入口：[`docs/paper_figure_registry.md`](../../docs/paper_figure_registry.md)。

## 使用规则

1. 引用格式统一为 `asset_id (paper_slot)`，例如
   `ictal_field_scaffold (Fig3-A–F)`。
2. 只有登记为 `LOCKED` 的资产可直接进入当前稿件；`CANDIDATE` 仍需总拼版或作者锁图。
3. `SOURCE` 只提供正式 panel 的素材或 renderer，不能作为独立结论引用。
4. 正式 supplementary 保留在顶层；source、诊断和模型谱系只从 `archive/` 查找。
5. 旧版不删除，统一移到 `archive/<date>_<reason>/`；顶层不再保留多个同义入口。
6. 每个实际含图的 `figures/` 目录必须有中文 `README.md`，并指出接受版本和科学边界。
7. 独立 panel 内不画左上角字母；每张主图同时保留一个带字母的 `figN-complete-layout.{png,pdf}`。

## 当前主文入口

| paper slot | asset_id | status | canonical path |
|---|---|---|---|
| Fig1-B–F | `interictal_hfo_temporal_scaffold` | `LOCKED` | `fig1/figures/` |
| Fig2-A–F | `interictal_spatial_scaffold` | `CANDIDATE` | `fig2/figures/` |
| Fig3-A–F | `ictal_field_scaffold` | `LOCKED` | `fig3/figures/` |
| Fig4-A–G | `data_driven_interictal_snn_fig4` | `LAYOUT_INCOMPLETE_RESERVED_PANEL_B` | `fig4/figures/` |
| Fig5-A–D | `data_driven_zm_transition_fig5` | `CANDIDATE` | `fig5/figures/` |
Fig1-A 是从登记的 legacy supplementary TIFF 固定裁剪得到的代表性植入脑图；裁剪后的 source asset 随 producer 入库，不重新绘制科学内容。Fig1/2/3 的独立 panel 均不含左上角字母；完整排版分别为
`fig1-complete-layout`、`fig2-complete-layout` 和 `fig3-complete-layout`，仅完整排版带 panel 字母。
旧素材与 producer/source 包不作为论文组装入口，正式引用只回到 `fig1/figures/`、`fig2/figures/`、
`fig3/figures/`、`fig4/figures/` 或 `fig5/figures/`。

Fig1-F 自 2026-09-02 起以 `fig1/figures/fig1-panelf.{png,pdf}` 的 v5 配对 inset 版为唯一
canonical 输出：主散点右下比较 single-template 与 multi-cluster MI，显示患者配对连线、均值柱和
paired Wilcoxon 括号；旧灰色 `median ΔMI` 小字版已被替代，不得作为后续拼版或重建来源。

Fig4 同样遵守无角标独立 panel + 带字母完整排版规则。当前 A–G 按 2026-09-03 最新作者顺序重排：A 为已确认的 local E/I circuit + patient-specific substrate；B 使用 A 右侧现有留白明确预留，当前不放图和占位文字；C 为 Node field + Model TA/MTB；D 为模型/患者 rank profile；E 为 cross-fit matrix；F 为 recruitment-onset-span readout；G 为 34 人 cohort。原 KMeans heatmap/rank distribution 保留在 FigS7-E。当前状态为 `LAYOUT_INCOMPLETE_RESERVED_PANEL_B`；该视觉状态不扩大 development-level scientific claim。

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
Fig3C 的 E10 | SZ3 右侧发作场扩展到临床起始前 120 s 至起始后 20 s。动画固定 shared TA plane、
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

尚有编号冲突或未锁定的计算材料继续保存在归档中。当前 `fig5/` 只保留 v5 最新完整候选；其 metadata 的科学边界仍是单 seed operational model transition，且形态闸门为 `NOT_SUSTAINED_ICTAL_MORPHOLOGY`，不能写成临床发作起始或多 seed 确认。

## 归档

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
