# Paper-Ready Figure Outputs — 总索引

本目录收集供论文组装用的图输出。约定：每个主图 / panel 组用自己的子目录，子目录内有
`figures/README.md` 说明这张图展示什么、哪个文件是接受版本。**本目录被 `.gitignore` 忽略**
（图是产物，由脚本再生），所以这里的结构是组织约定、不进版本控制。

> **2026-06-30 整理记录**：① 删除 scratch 子目录 `fig_subject_snn_epilepsiae_1146_style_preview`
> （仅样式预览）、`fig_m3a_v2_1_qigk_trajectory`（单文件孤儿预览）。② 把 24 个 per-subject
> subject-SNN 读出图按主题打包进 `fig4_subject_snn_cohort/`（E1146 headline + 其变体仍留顶层，
> 因 `FIGURE_INDEX.md` 直接引用 E1146 路径）。③ 本 README 重写为完整索引。
> **Caveat**：subject-SNN 生成脚本（`plot_fig_subject_snn*.py`）默认把 `fig_subject_snn_<subject>/`
> 写到**顶层**；若整批重跑，新输出会重新散到顶层，需再跑一次本打包步骤（或改脚本输出根目录）。

---

## Fig1 — 间期 HFO 群体事件素材

- `fig1_hfo_group_event_demo/` — **Fig1-a1/a2 源素材**：legacy 人工标注 HFO n=178 形态图 + Yuquan Y3 群体事件示例。
  脚本为 `scripts/paper_figures/plot_fig1_single_hfo_schematic.py` 与 `plot_fig1_hfo_group_event_legacy_style.py`；正式 panel 文件位于 `fig1_interictal_hfo_temporal_scaffold/figures/`。

> Fig2（单 subject 间期传播时序素材）不在本目录：它由 Topic 1 propagation 主绘图器生成，
> 见 `results/interictal_propagation_masked/figures/per_subject/` + `docs/fig2_temporal_propagation_panel_spec.md`。
> Fig2-B 模板匹配预览脚本 `scripts/paper_figures/plot_fig2b_template_matching_preview.py` 仍只作历史 preview；间期-vs-ictal paired field 已改列 Fig3-B。
- `fig2c_interictal_event_envelope_field/` — **Fig2-C paper-ready candidate**：E1146 两次真实 TA/TB
  代表事件的 Fig1a 质心 readout + 冻结 shared-plane 包络 frame；同时输出同一合同的 2 ms 生物学步长
  GIF。canonical producer 为 `scripts/paper_figures/plot_fig2c_interictal_event_envelope_field.py`，唯一
  frame/GIF 规范为 `docs/fig2c_interictal_event_envelope_field_spec.md`。该合同只管 TA/TB 每行一个
  exemplar 的单事件版本；未来多事件 GIF 单独建规范。定位是 representative
  raw-envelope timing cross-check，不是 template-free/cohort/traveling-wave 证明。
- `fig2e_interictal_template_fields/` — **Fig2-E paper-ready candidate**：患者特异 TA/TB 静态 rank
  field；与 Fig2-C 共用冻结轴/平面，但不使用事件包络时间轴。
- `fig2_shared_field_reversal/` — **Fig2-F 最后一行 paper-ready candidate**：12名 shared-axis +
  有效二维几何患者的 TA–TB contact-field signed `r`；左侧复用统一 renderer 展示匿名投稿代号 E15/E14/E13/Y9
  四个几何易读的负相关 TA/TB Viridis 场，右上为完整12人分布，右下为全触点 channel-shuffle
  cohort-median null。
  当前8/12为负、中位 `r=-0.353`，`Δmedian=-0.339`，`P_perm=0.01840`；不做关系分组。
- `fig_interictal_ab_direction_axis/` — **间期空间方向候选图**：A/B 独立 earliness gradient、过门后的共享线、
  subject-native 脑表面 early→late 箭头与描述性解剖 overlay。主触点池与 expanded-contact sensitivity 并存；
  脚本 `scripts/paper_figures/plot_interictal_ab_direction_axis.py`。

## Fig3 — 间期传播场 vs 发作早期激活场一致性

- `fig3a_raw_spectral_context/` — **Fig3-A 正式版**：E1146 seizure 7 的 raw SEEG、严格对齐的 SCL9 TFR 与
  low bands / gamma / high-gamma / broadband 2×2 band-power context。右侧按行共享 y 轴；只标 baseline 与
  clinical-onset `[0,10) s` 阴影。脚本 `scripts/paper_figures/plot_fig3_raw_spectral_context.py`；完整规范见
  `docs/figure_style_guide.md` §5a。
- `fig3b_interictal_ictal_shared_field/` — **Fig3-B paper-ready candidate**：E1146 的冻结 TA timing
  field 与当前 25 次 complete / exact 1–150 Hz 发作中 `shared_a_signed` 最高的 seizure 15 broadband power 场。
  左右共用 shared plane、TA support、extent 和 6 mm display sigma；`TA fields` 标题固定红色，右图为 `magma_r`
  连续 min–max 插值、无 rank/sign flip，两个 panel 各自写 xlabel。colorbar 直接报告 propagation rank / robust-z，
  深色统一表示最早传播或最高 broadband power。
  producer 为 `scripts/paper_figures/plot_fig3b_interictal_ictal_shared_field.py`，合同为
  `docs/fig3b_interictal_ictal_shared_field_spec.md`。
- `fig3_field_concordance_cohort_stat/` — **Fig3 field-concordance cohort statistic（panel 编号待总拼版）**：cohort 级 Data-vs-Null 统计面板
  （maxAB 可评估 subject 上间期传播场与发作早期激活场整体高于 channel-shuffle null）。正式图现在包含
  `BB 1-45 maxAB`、line-noise-masked `BB 1-150 maxAB` 和 `HFA 60-100 maxAB` 三组配对统计。
  脚本 `scripts/paper_figures/plot_fig3_field_concordance_cohort_stat.py`。
- `fig3_sup1_multiband_field_alignment/` — **Fig3-Sup1**（V2 Phase-1，multi-band supplement）：3 panel
  A 观测 subject×band maxAB 热图（narrow>broad、band-generic）/ B 每 primary 带 per-subject Δ vs 弱空间
  null violin（6/7 过 family-wise、ripple_high n.s.=NOT ripple-specific）/ C per-subject 稳定性（cohort
  6/7 是聚合、narrow 中位 2/7 = 承重 caveat）。**tier=exploratory candidate scaffold（cohort 层，非
  formal/机制）**；formal within-shaft Gate A 未评估(2/20)、Gate B/C 未跑。脚本
  `scripts/paper_figures/plot_fig3_sup1_multiband_field_alignment.py`；归档
  `docs/archive/topic5/v2_phase1_band_scan_backbone_2026-07-02.md`。
- `fig3_sup2_raw_spectral_context/` — Fig3-A 定稿前的历史输出路径；保留溯源，不再作为 canonical panel 引用。
- `fig_topic5_field_extrapolation_energy/` — **Topic 5 energy-field paper-ready 主图**：A E1146 真实电极布局上的测试设计
  （core-field vs own-order 预测 hidden seizure energy）/ B cohort Δ 直接裁决 / C 证据阶梯。
  结论边界：network extension supported，但 core-field 不系统性优于 hidden 电极自身间期顺序。
  脚本 `scripts/paper_figures/plot_fig_topic5_field_extrapolation_energy.py`。
- `fig_topic5_network_extension_null/` — **Topic 5 network-extension three-way 独立统计图**：每个频段同图放
  core-field prediction / hidden own-order C1 / channel-shuffle null，三条 bracket 分别裁决 core>null、
  own>null、core>own。脚本 `scripts/paper_figures/plot_fig_topic5_network_extension_null.py`。

## Fig4 — Subject-specific SNN 读出（病人真实电极几何）

**Headline（E1146，顶层；被 `FIGURE_INDEX.md` 引用）：**

- `fig_subject_snn_epilepsiae_1146/` — **Fig4A/B/C**：E1146 真实电极布局 subject-specific SNN
  读出（A）+ KMeans k=2 核验（B）+ 模型 vs 真实间期模板一致性（C）。
  脚本 `plot_fig_subject_snn.py` / `plot_fig_subject_snn_kmeans2.py` / `..._realvsmodel.py`。
- `fig_subject_snn_epilepsiae_1146_mechanism/` — E1146 独立高清**机制面板**（E/I 神经元 + source
  core + AR=2 的 E→E 长轴连接 lobe），脚本 `plot_fig_subject_snn_mechanism.py`。
- `fig_subject_snn_epilepsiae_1146_stimulation/` — E1146 真实几何上的**刺激示意图**
  （刺激前传播事件 vs 刺激打开后局部事件 + readout 刺激窗），脚本 `plot_fig_subject_snn_stimulation.py`。
- `fig_subject_snn_epilepsiae_1146_COVERAGE_VARIANT/` — **覆盖优先对照样张**，被 archive
  `docs/archive/topic4/sef_hfo/cohort_field_swap_snn_coverage_tradeoff_2026-06-27.md` 引用为证据；
  保留（非 scratch）。
- `fig_subject_snn_epilepsiae_1146_CR1p5_S4_T20000_20260706/` — **E1146 小核长时重跑版**：
  `core_r=1.5, seed=4, T=20000 ms`，Fig4A 使用椭圆 E→E 长轴 footprint + 完整 20 秒 readout；
  Fig4B 使用 active-contact KMeans（>=30% clean-event participation）减少灰格。双向仍存在但偏
  reverse（5/11），KMeans purity=0.8125；用于长时 sensitivity，不替代 headline seed3 正式图。

**Cohort（按主题打包）：**

- `fig4_subject_snn_cohort/` — 24 个 per-subject subject-SNN 读出图（14 epilepsiae + 10 yuquan），
  与 E1146 同一套四列标准、按各自真实电极布局。每个子目录见其内文件 + metadata。
- `_cohort_field_swap_snn/` — **Fig4 cohort 接触片**（contact sheet 4A/4B）+ cohort 统计/索引。
  脚本 `run_cohort_field_swap_snn.py` + `cohort_field_swap_summary.py`。

## Fig5 — 核心模型 S3 brake-off

- `fig5_core_model_s3_brakeoff/` — **Fig5**：模型底物可产生自发双向传播、虚拟 SEEG 可读出方向
  （只支持"能产生 + 可读方向"，不单独证明真实病人机制）。脚本 `plot_fig5_core_model_s3_brakeoff.py`。

## Fig6 — RNN / contact-field computational supplements

- `fig_topic5_minimal_sequence_kernel_closeout/` — **当前最小序列结构收口图**：主六联图把
  where（稳定 contact scaffold）、how（contact choice 只需当前和前一 rank；第三 rank
  主要改善 STOP）和 when（现有 event-reset 模型不检验真实时间）明确分开；辅助三联图
  报告 0–2 ms timing sensitivity、early-ictal target reliability gate 与 IEI-aware
  跨事件 feasibility。结论为 supplementary bounded result；不支持脑流形、真实时间慢状态
  或 ordered residual 的 early-ictal 增量。producer 为
  `scripts/paper_figures/plot_topic5_minimal_sequence_kernel_closeout_v0_2.py`。
- `fig6_static_contact_topography/` — 六块固定验收图：target-sealed contact-field 合同、
  formal heldout order gain 与 matched perturbation 的区别、signed primary、sign-free spatial
  morphology、full GRU 相对静态/一阶/打乱对照的增量，以及 contact-confound sensitivity。
  当前只支持患者特异 interictal contact topography 与 early-ictal energy 的
  orientation-free static correspondence；不支持 GRU-specific recurrent-order increment、
  自动 physical-axis 恢复或 seizure trajectory prediction。脚本
  `scripts/paper_figures/plot_fig6_static_scaffold_fixed_readout.py`。
- `fig6_ordered_history_architecture_audit/` — 六块顺序信息与架构控制图：精确定义事件内
  rank-step state、比较 static/unordered/linear/rate/GRU/low-rank、匹配 rank-shuffle、
  reverse/drop/reset 干预、clinical-onset early-ictal 条件增量和预先固定的 E1146 contact
  field。target-blind linear-state 有 heldout 顺序增量，但 7 个预注册递归家族仅 1 个通过
  family-wise inference；early-ictal 条件增量未建立。producer 为
  `scripts/paper_figures/plot_topic5_ordered_history_architecture_audit_v0_1.py`。

## 模型机制诊断图（M3A-v2.x，visual diagnostic，非主图 claim）

> 这些是 M3A-v2 慢变量场闭环 screen（一致 NEGATIVE）的目视诊断图。统计判读以 sweep + 归档
> doc 为准，不在这些图。详见 `docs/topic4_m3_stage.md` §2 + `docs/archive/topic4/m3a_v2_2_carrier_exploration_2026-06-29.md`。

**M3A-v2.2（sustained 协议 + `h_G` 载体，主收口图）：**

- `fig_m3a_v2_2_explore_summary/` — 读 3184-sim sweep 的三联结果汇总图。
- `fig_m3a_v2_2_dynamics/` — 代表性四列动力学示意（单 seed 重跑）。
- `fig_m3a_v2_2_hG_runaway_transition/` — 全局恢复 `h_G` 单轨迹 GIF（减法式刹车拉不回 runaway）。
- `fig_m3a_v2_2_qI_runaway_transition_epilepsiae_1146/` — `q_I` 载体 + 轴向 `g_K` 疲劳 runaway GIF（E1146 几何）。
- `fig_m3a_v2_2_qI_stim_runaway_epilepsiae_1146/` — 刺激 vs 不刺激对照 GIF（中段 `V_th` clamp 把
  runaway 推后 +834 ms；外部预防式压制示意，非治疗主张）。

**M3A-v2.1 / Step4（诊断 + h_G-OFF 对照基线）：**

- `fig_m3a_v2_1_qigk_runaway_transition/` — v2.1 `q_I/g_K` runaway GIF；是 v2.2 `h_G` 图的 **h_G-OFF 对照基线**。
- `fig_m3a_v2_1_qigk_runaway_transition_epilepsiae_1146/` — 同上 E1146 几何变体。
- `fig_m3a_v2_1_qigk_gap_dynamics/`、`..._L20/` — v2.1 gap-sweep 代表状态四列诊断（L=默认 / L=20）。
- `fig_m3a_v2_step4_dynamics/` — M3A-v2 Step4 低-q / g_K 闭环负结果代表动力学（visual diagnostic）。
