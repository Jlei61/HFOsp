# 主图计划

本文主图围绕两个核心论点组织：

1. 间期 HFO 群体事件是癫痫病理网络的指示器。
2. 间期活动可能是病理网络动态的推动者；这部分主要通过模型和病例场景说明可行机制。

## Fig1: 间期 HFO 群体事件与病理网络读出

### Fig1-A: 原始群体事件示例

**目的**：用最直观的原始信号说明，间期 HFO 不是孤立单通道尖峰，而是跨通道共同出现的群体事件，并且群体内部存在稳定的早晚关系。

**当前验收版本**：

- 输出目录：`results/paper-ready-figure/fig1_hfo_group_event_demo/figures/`
- 正式文件：`yuquan_y1_hfo_group_event_demo.png` / `yuquan_y1_hfo_group_event_demo.pdf`
- 复现入口：`scripts/paper_figures/plot_fig1_hfo_group_event_legacy_style.py`
- 数据来源：Yuquan Y1, `FC10477Q`
- 固定示例事件：packed event indices `22,237,1458`
- 图形合同：左侧为 80-250 Hz stacked bipolar traces；右侧为 legacy-style normalized spectrogram，并用 spec-center 点/线显示群体事件内部时序。

**当前口径**：

这张图只承担现象入口作用，不单独证明 cohort-level 传播模板或机制结论。它应该把读者带到后续 Fig1-B/C/D 的定量结果：群体事件可被定义、可排序、可汇总到病理网络轴。

### Fig1-B: 群体事件定义与传播 rank

**计划内容**：展示从 HFO detections 到 packed group event，再到 channel-level event rank / template 的分析流程。

**需要补齐**：

- 明确使用 masked `lagPatRank` 后的正式 pipeline 输出。
- 选一个 subject-level schematic，而不是堆 cohort 数值。
- 避免把示意图画成方法 supplement；主图只保留读者理解传播 rank 所需的最小链条。

### Fig1-C: 病理网络指示器的 cohort-level 证据

**计划内容**：展示间期 HFO 群体事件的空间组织、SOZ/病灶相关性或网络轴 readout。

**需要补齐**：

- 从 Topic 1/3 当前验收结论里选择最稳的 cohort-level readout。
- 区分“事件存在时序结构”和“该结构指向病理网络”的证据层级。
- 主图只放一个核心统计面板，完整分层表放 supplement。

### Fig1-D: 从指示器到动力学 scaffold

**计划内容**：把间期传播模板和病理网络 scaffold 连接起来，作为后续建模主张的入口。

**需要补齐**：

- 明确哪些内容来自真实数据，哪些只是模型 bridge。
- 不在 Fig1 里提前声称“推动者”机制已经被证明；只说明 Fig1 给出可被模型解释的病理网络读出。

## Fig2-Fig6 暂定分工

### Fig2-Fig3: 间期事件作为病理网络指示器

优先承载真实数据主结果：传播模板、网络轴、SOZ/临床相关性、跨事件稳定性。这里应该是第一核心论点的主要证据区。

### Fig3-A: 间期传播场与发作早期激活场的 cohort 级一致性

**目的**：用一个紧凑 Data-vs-Null 统计面板说明，间期 HFO 传播场和发作早期激活场在 maxAB 可评估 subject 层面整体高于 channel-shuffle null；不再展示 per-subject board。

**当前验收版本**：

- 输出目录：`results/paper-ready-figure/fig3_field_concordance_cohort_stat/figures/`
- 正式文件：`field_concordance_cohort_stat.png` / `field_concordance_cohort_stat.pdf`
- 复现入口：`scripts/paper_figures/plot_fig3_field_concordance_cohort_stat.py`
- 数据来源：`results/topic5_ictal_recruitment/axis_alignment/axis_alignment_{broadband,hfa}_max_ab_B1000.json`
- 图形合同：按参考图风格画两组 `Data` vs `Null` 的 violin + box + subject 点；左侧为 `Broadband maxAB`，右侧为 `HFA maxAB`；不写 `All candidates`，不画背景网格线，不混入 broad fallback。严格 maxAB 可评估 subject 为 19 个，因为 `yuquan_xuxinyi` 只有 broad 几何、没有 maxAB。

**当前口径**：

这个 panel 支持“共享粗网络轴 / field concordance”，不表示发作沿间期方向逐点重放，也不替代 Topic 5 A-line primary FDR 定稿表。这里的 Null 是所选候选的 channel-shuffle median，用于展示 cohort-level shift above null；formal pass 仍以 selection-corrected p95/p-value 表为准。

### Fig4: 被试特异性 SNN + KMeans readout 核验（E1146）

**目的**：把同一 cm-SNN 标准底物按**病人真实电极平面**摆放，两个低阈值核放在**两类间期模板各自最早的电极区**（=两类模板的 source，轴两端），看同一虚拟 SEEG（=病人真实触点）能否读出正/反间期传播，并用无监督 KMeans 验证 readout 事件是否自然分成两类。

**Fig4A：subject-specific SNN readout**

- 输出目录：`results/paper-ready-figure/fig_subject_snn_epilepsiae_1146/figures/`
- 正式文件：`fig_subject_snn_epilepsiae_1146.png` / `.pdf`
- 复现入口：`scripts/paper_figures/plot_fig_subject_snn.py`（消费 `scripts/run_sef_hfo_subject_snn.py --placement template_source` 的产物）
- 图形合同：四列结构 `mechanism | tempA source | tempB source | electrode readout`；mechanism 显示两核与电极最早区 overlap + E->E 长轴带；readout 用 spontaneous twoend，暖/冷阴影分 tempA/tempB 事件。

**Fig4B：KMeans k=2 readout verification**

- 正式文件：`fig_subject_snn_epilepsiae_1146_kmeans2.png` / `.pdf`
- 复现入口：`scripts/paper_figures/plot_fig_subject_snn_kmeans2.py`
- 图形合同：**四块** `clustered event heatmap | per-channel rank distribution | cluster rank distributions | model-vs-real 2×2 相似性矩阵`。前三块**复用仓库成熟 canonical 画图函数**（`scripts/plot_interictal_propagation.py` 的 `_plot_rank_histogram` / `_plot_rank_heatmap` / `_plot_cluster_boundaries` / `_plot_cluster_rank_fig4`，与 Topic-1a per_subject 图同一套），不手搓；heatmap rank colorbar 横放 x-label 下。第四块 = 模型 fwd/rev × 真实 t_a/t_b Spearman 矩阵，只用 star 显示方向性置换 p（不写数值）、aspect=equal。
- **LOCKED 模式**：每个 subject-SNN 案例固定出 Fig4A（readout 四列）+ Fig4B（KMeans 四块）两张主图；Fig4C（real-vs-model profile）/ Fig4D（组合 S 置换 null）为可选 supplement。
- 当前结果：同一个 seed3 spontaneous twoend readout 的 14 个 clean directional events 被 `KMeans k=2` 分成 C0/C1 = 6/8；方向 purity=1.00；`within_cluster_tau_mean=0.939`；更干净的 shared-overlap corr = -0.69。

**Fig4C：模型模板 vs 真实间期模板一致性**

- 正式文件：`fig_subject_snn_epilepsiae_1146_realvsmodel.png` / `.pdf`
- 复现入口：`scripts/paper_figures/plot_fig_subject_snn_realvsmodel.py`
- 图形合同：A=真实 t_a/t_b 逐通道 `typical_rank`，B=模型 forward/reverse 逐通道平均 rank；一致性以逐通道 Spearman 判。
- 当前结果：**model-forward vs real-t_a ρ=+0.87（n=7）、model-reverse vs real-t_b ρ=+0.62（n=11）**，交叉项为负 → 模型在 ICL readout 通道上复现了真实间期模板的传播顺序与 swap 反向结构。**结论=一致**，故未触发"不一致则重做 1146 仿真"。

**当前口径（诚实）**：E1146（ICL 密杆，能采到完整传播）成立，但不是机制证明。模型 readout 顺序与真实间期模板一致（Fig4C），但属单被试、读出级一致性，非因果/cohort。自发双向**与 seed 有关**（seed3 6 正/8 反；seed1/2 偏反向），分开驱动 source 5/0、sink 0/9。读出依赖 `k_dir=2`（病人电极稀疏放宽，载重参数）+ 真实几何 plane-fit。E958（稀疏栅格）阴性。不声称"真实病人机制被证明"；这是机制/读出可行性示意。

### Fig5-Fig6: 间期活动作为推动者的模型与病例场景

当前机制证据还没有收口，因此 Fig5/6 先按建模工作组织。允许呈现几类可能病例场景，但必须清楚区分：

- 真实数据已经支持的 readout；
- 模型能够复现或解释的 dynamics；
- 仍然是假设、需要后续验证的机制。

### Fig5-A: cm-SNN 自发双向 readout 机制示意

**目的**：用最少 panel 展示同一个 stage-3 brake-off cm-SNN 底物如何产生正向和反向间期传播事件，并被同一虚拟 SEEG montage 读出。

**当前验收版本**：

- 输出目录：`results/paper-ready-figure/fig5_core_model_s3_brakeoff/figures/`
- 正式文件：`core_model_s3_brakeoff.png` / `core_model_s3_brakeoff.pdf`
- 复现入口：`scripts/paper_figures/plot_fig5_core_model_s3_brakeoff.py`
- 兼容输出：`results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/figures/core_model_s3_brakeoff.png`
- 图形合同：按 SNN 仿真标准画法组织为 `mechanism + tempA source + tempB source + electrode readout`；左侧机制 panel 显式画出 E->E 长轴作用范围；中间两个方形 panel 分别展示两种特异性组合的代表传播；右侧 readout 用不同颜色阴影区分 forward / reverse clean propagation events。

**当前口径**：

这张图只支持“模型底物可产生自发双向传播，并且虚拟 SEEG 可读出方向”。它不单独证明真实病人的机制，也不声称 M2 brake-off 已解决沿轴空间自限。

## 当前执行原则

- 主图脚本统一放在 `scripts/paper_figures/`。
- 主图输出统一放在 `results/paper-ready-figure/`，每个 figure/panel group 单独建目录。
- 每个 figure 输出目录必须有 `figures/README.md`，说明展示目的、正式文件、关注点。
- 主图计划文档只记录正式口径和待补齐内容；详细审阅和数值表继续放 `docs/archive/`。
