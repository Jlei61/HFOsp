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

- `fig1_hfo_group_event_demo/` — **Fig1-A**：Yuquan Y1 单段间期 HFO 群体事件紧凑示例。
  脚本 `scripts/paper_figures/plot_fig1_hfo_group_event_legacy_style.py`（+ `..._prototype.py`）。

> Fig2（单 subject 间期传播时序素材）不在本目录：它由 Topic 1 propagation 主绘图器生成，
> 见 `results/interictal_propagation_masked/figures/per_subject/` + `docs/fig2_temporal_propagation_panel_spec.md`。
> Fig2-B 模板匹配预览脚本 `scripts/paper_figures/plot_fig2b_template_matching_preview.py`。

## Fig3 — 间期传播场 vs 发作早期激活场一致性

- `fig3_field_concordance_cohort_stat/` — **Fig3-A**：cohort 级 Data-vs-Null 统计面板
  （maxAB 可评估 subject 上间期传播场与发作早期激活场整体高于 channel-shuffle null）。
  脚本 `scripts/paper_figures/plot_fig3_field_concordance_cohort_stat.py`。
- `fig3_sup1_multiband_field_alignment/` — **Fig3-Sup1**（V2 Phase-1，multi-band supplement）：3 panel
  A 观测 subject×band maxAB 热图（narrow>broad、band-generic）/ B 每 primary 带 per-subject Δ vs 弱空间
  null violin（6/7 过 family-wise、ripple_high n.s.=NOT ripple-specific）/ C per-subject 稳定性（cohort
  6/7 是聚合、narrow 中位 2/7 = 承重 caveat）。**tier=exploratory candidate scaffold（cohort 层，非
  formal/机制）**；formal within-shaft Gate A 未评估(2/20)、Gate B/C 未跑。脚本
  `scripts/paper_figures/plot_fig3_sup1_multiband_field_alignment.py`；归档
  `docs/archive/topic5/v2_phase1_band_scan_backbone_2026-07-02.md`。

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
