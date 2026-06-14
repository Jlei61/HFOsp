# results/ 图总索引（FIGURE INDEX）

> 解决"重要的图埋太深、容易错过"的问题。本文件是 `results/` 下所有**结论级图**的导航入口。
> 每个目录里仍有自己的中文 `figures/README.md`（逐图说明），本文件只负责"按 topic 快速定位 + 指路"。
>
> 用法：先看下面「只看这几张」挑代表图；要细节就点到对应目录的 `README.md`。
> per-subject / per-seizure 的诊断图（占全部图的 ~75%）不在本索引里逐一列出——它们藏在各目录的
> `per_subject/` `per_seizure/` `subjects/` 子目录中，是单被试核对用，不是结论图。
>
> 最近更新：2026-06-14。新增结论图目录时，请在对应 topic 表里补一行。
>
> **画新图前先看可视化标准** → [`docs/figure_style_guide.md`](../docs/figure_style_guide.md)：
> 每类反复出现的图（时序模板 / swap 节点 / 几何传播 / 事件时序 / 机制模型）的固定布局 + 配色 + 轴约定。

---

## 只看这几张（每个 topic 的代表图）

| Topic | 代表图 | 一句话 |
|---|---|---|
| 0 方法学 | [lagpatrank_audit/figures/ami_vs_noise_floor.png](lagpatrank_audit/figures/) | lagPatRank phantom-rank 审计：旧聚类特征被非参与通道污染的程度 |
| 1 同步 | [interictal_synchrony/analysis/combined/figures/figure_b_trajectory_all.png](interictal_synchrony/analysis/combined/figures/) | 合并队列：事件级同步性随时间的轨迹 |
| 1 传播 | [interictal_propagation_masked/figures/cohort_propagation_summary.png](interictal_propagation_masked/figures/) | 队列间期传播汇总（masked = 当前 canonical） |
| 2 周期性 | [event_periodicity/figures/yuquan_cohort_psd_stack.png](event_periodicity/figures/) | 群体事件脉冲序列的功率谱（是否有周期峰） |
| 3 空间/SOZ | [spatial_modulation/soz_comparison/figures/soz_vs_nonsoz_lag1r_paired.png](spatial_modulation/soz_comparison/figures/) | SOZ vs 非 SOZ 通道的配对差异 |
| 3 几何骨架 | [spatial_modulation/propagation_geometry/components/path_axis/figures/along_axis_stereotypy_profile.png](spatial_modulation/propagation_geometry/components/path_axis/figures/) | 传播是否沿一条稳定空间轴 |
| 4 模型 | [topic4_sef_hfo/snn_heterogeneity/figures/mean_scan.png](topic4_sef_hfo/snn_heterogeneity/figures/) | SNN 阈值异质核：点火边界的参数扫描 |
| 4 观测层 | [topic4_sef_hfo/observation_layer/snn_cm_spontaneous/figures/stage2_summary.png](topic4_sef_hfo/observation_layer/snn_cm_spontaneous/figures/) | 虚拟 SEEG 把模型读回成方向/模板 |
| 5 ictal 回响 | [topic5_ictal_template_echo/figures/echo_anchor_not_path.png](topic5_ictal_template_echo/figures/) | 发作期通道顺序是否回响间期模板（共享粗锚 vs 具体路径） |
| 5 桥接 | [topic1_topic5_bridge/figures/q1prime_cohort_effect.png](topic1_topic5_bridge/figures/) | Topic 1 模板 × Topic 5 亚型 的队列效应 |

---

## Topic 0 — 方法学审计（任何数字前先读）

| 目录 | 内容 |
|---|---|
| [lagpatrank_audit/figures/](lagpatrank_audit/figures/) | lagPatRank phantom-rank 诊断（ami_vs_noise_floor / phantom_fraction_vs_delta / stable_k_confusion） |
| [interictal_propagation_vs_masked/figures/](interictal_propagation_vs_masked/figures/) | phantom vs masked PR-2 的 before/after 对比 |

## Topic 1 — 间期事件动态（传播 + 同步）

**同步性**
| 目录 | 内容 |
|---|---|
| [interictal_synchrony/analysis/combined/figures/](interictal_synchrony/analysis/combined/figures/) | 合并队列（Epilepsiae+Yuquan）：trajectory / fixed_window / robustness / coverage / event_rate |
| [interictal_synchrony/analysis/yuquan/figures/](interictal_synchrony/analysis/yuquan/figures/) | Yuquan 独立队列（含 per-subject timeline） |

**传播（masked = 当前 canonical）**
| 目录 | 内容 |
|---|---|
| [interictal_propagation_masked/figures/](interictal_propagation_masked/figures/) | PR-2/PR-3 队列传播汇总 |
| [interictal_propagation_masked/rank_displacement/figures/](interictal_propagation_masked/rank_displacement/figures/) | 连续 swap 几何（displacement / cardinality / SOZ overlap） |
| [interictal_propagation_masked/template_anchoring/figures/](interictal_propagation_masked/template_anchoring/figures/) | endpoint 几何 + 模板对几何 |
| [interictal_propagation_masked/template_share_switching/figures/](interictal_propagation_masked/template_share_switching/figures/) | 发作前后窗口的模板占比 + 切换 |

**传播（旧版 phantom-contaminated，部分未重跑——引用前确认是否已有 masked 对应）**
| 目录 | 内容 |
|---|---|
| [interictal_propagation/pr6_step6_held_out_template/figures/](interictal_propagation/pr6_step6_held_out_template/figures/) | 留出时间窗的模板稳定性 |
| [interictal_propagation/pr6_sup1_rank_entropy/figures/](interictal_propagation/pr6_sup1_rank_entropy/figures/) | first-rank entropy 补充 |
| [interictal_propagation/template_share_switching/figures/](interictal_propagation/template_share_switching/figures/) | 模板占比/切换（旧版，masked 版见上） |

**broad channel-pool 扩展**
| 目录 | 内容 |
|---|---|
| [lagpat_broad/figures/](lagpat_broad/figures/) | broad lagPat 通道池扩展（SOZ 内外覆盖 + 更大 KMeans） |

## Topic 2 — 事件周期性

| 目录 | 内容 |
|---|---|
| [event_periodicity/figures/](event_periodicity/figures/) | cohort PSD stack + IEI summary（Epilepsiae & Yuquan） |
| [event_periodicity/phase2/figures/](event_periodicity/phase2/figures/) | Phase 2 五个实验图 |

## Topic 3 — 空间 / SOZ 调制

| 目录 | 内容 |
|---|---|
| [spatial_modulation/soz_comparison/figures/](spatial_modulation/soz_comparison/figures/) | SOZ vs 非 SOZ 配对（lag1r / IEI / deadtime / detrend） |
| [spatial_modulation/propagation_geometry/figures/](spatial_modulation/propagation_geometry/figures/) | 传播几何总图 |
| [spatial_modulation/propagation_geometry/components/path_axis/figures/](spatial_modulation/propagation_geometry/components/path_axis/figures/) | 路径轴骨架（沿轴刻板性 + 轴框示例） |
| [spatial_modulation/propagation_geometry/components/entry_variability/figures/](spatial_modulation/propagation_geometry/components/entry_variability/figures/) | 入口分散度（含 3D overlap，per-subject 在子目录） |
| [spatial_modulation/propagation_geometry/observation_readout/figures/](spatial_modulation/propagation_geometry/observation_readout/figures/) | 触点平面读出（static_maps 子目录 52 张 per-subject） |
| [refine_soz_validation/figures/](refine_soz_validation/figures/) · [.../epilepsiae/figures/](refine_soz_validation/epilepsiae/figures/) | refine-SOZ 验证（cohort + per-subject） |
| [propagation_entry_dispersion/figures/](propagation_entry_dispersion/figures/) | 入口分散度（独立目录，3D overlap 在子目录） |

## Topic 4 — SEF-HFO / SEF-ITP 机制模型

**SEF-HFO（rate field + spiking network）**
| 目录 | 内容 |
|---|---|
| [topic4_sef_hfo/schematic/figures/](topic4_sef_hfo/schematic/figures/) | 机制示意图 |
| [topic4_sef_hfo/linear_stability/figures/](topic4_sef_hfo/linear_stability/figures/) | Step 0a：LIF 自洽工作点 |
| [topic4_sef_hfo/finite_pulse/figures/](topic4_sef_hfo/finite_pulse/figures/) | Step 0b/0d：LIF rate field 真实场 |
| [topic4_sef_hfo/step1_noise/figures/](topic4_sef_hfo/step1_noise/figures/) | Step 1：drive × σ 联合分析 |
| [topic4_sef_hfo/lif_snn/figures/](topic4_sef_hfo/lif_snn/figures/) | LIF ↔ spiking-network 验证 |
| [topic4_sef_hfo/low_rate_template_stability/figures/](topic4_sef_hfo/low_rate_template_stability/figures/) | 低事件率：传播模板 vs 发放计数复现度 |
| [topic4_sef_hfo/snn_heterogeneity/figures/](topic4_sef_hfo/snn_heterogeneity/figures/) | **SNN 阈值异质核 sweep**（headline: mean_scan / sweep_ignition；mechanism_* 是各 kick×core 组合） |
| [topic4_sef_hfo/skeleton_geometry/figures/](topic4_sef_hfo/skeleton_geometry/figures/) | 几何骨架（per-subject 在子目录） |
| [topic4_sef_hfo/observation_layer/figures/](topic4_sef_hfo/observation_layer/figures/) | 虚拟 SEEG 观测层 |
| [topic4_sef_hfo/observation_layer/increment3a_rate_parity/figures/](topic4_sef_hfo/observation_layer/increment3a_rate_parity/figures/) | rate parity 增量 |
| [topic4_sef_hfo/observation_layer/snn_cm_spontaneous/figures/](topic4_sef_hfo/observation_layer/snn_cm_spontaneous/figures/) | cm-SNN 自发（headline: stage2_summary / stage3_regime_compare） |
| [topic4_sef_hfo/observation_layer/snn_cm_spontaneous/candidate_confirm/figures/](topic4_sef_hfo/observation_layer/snn_cm_spontaneous/candidate_confirm/figures/) | 候选格电极读出 train |
| [topic4_sef_hfo/observation_layer/snn_cm_spontaneous/a1_formal/figures/](topic4_sef_hfo/observation_layer/snn_cm_spontaneous/a1_formal/figures/) | **axis-A A1** 阈值离散→指纹 = NULL（只改点火率不改指纹） |
| [topic4_sef_hfo/observation_layer/snn_cm_spontaneous/a3_0a_scan/figures/](topic4_sef_hfo/observation_layer/snn_cm_spontaneous/a3_0a_scan/figures/) | **axis-A A3** 局部 E/I 病灶 screen = NULL（不复现 V_th↓ 方向模板） |

**SEF-ITP**
| 目录 | 内容 |
|---|---|
| [topic4_sef_itp/phase1_spatial_geometry/figures/](topic4_sef_itp/phase1_spatial_geometry/figures/) | Phase 1 cohort 空间几何 |
| [topic4_sef_itp/phase3_ictal_adjacent/v2_trajectory/figures/](topic4_sef_itp/phase3_ictal_adjacent/v2_trajectory/figures/) | Phase 3 v2.2：per-event timeline + RMS-vs-k |
| [topic4_sef_itp/direction_axis/figures/](topic4_sef_itp/direction_axis/figures/) | H2b 方向轴诊断（per-event 多） |
| ⚠️ [topic4_sef_itp/phase4_hr_route_SUPERSEDED/...](topic4_sef_itp/phase4_hr_route_SUPERSEDED/) | **已废弃**（目录名标 SUPERSEDED），只作历史 |

**attractor（无 figures/README，旧诊断）**：`topic4_attractor/`、`topic4_attractor_masked/`

## Topic 5 — 亚型 / ictal 回响 / network axis / 临床结局

| 目录 | 内容 |
|---|---|
| [data_driven_soz/layer_a_ictal_er_rank/atlas_v2_3/figures/](data_driven_soz/layer_a_ictal_er_rank/atlas_v2_3/figures/) | ictal ER-onset timing atlas（per_seizure 子目录 339 张） |
| [data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/figures/](data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/figures/) | PR-1 z-ER 子型聚类（per-subject 在子目录） |
| [topic5_ictal_template_echo/figures/](topic5_ictal_template_echo/figures/) | Stage 1 ictal-template-echo（anchor vs path） |
| [topic5_dynamic_echo/sentinel/figures/](topic5_dynamic_echo/sentinel/figures/) | Stage 2b 动态回响哨兵 |
| [topic5_ictal_recruitment/axis_alignment/figures/](topic5_ictal_recruitment/axis_alignment/figures/) | A 线 axis alignment 可视化 |
| [topic1_topic5_bridge/figures/](topic1_topic5_bridge/figures/) | Topic 1 模板 × Topic 5 亚型 桥接（q1 / q1prime 系列） |
| [template_resection_outcome/figures/](template_resection_outcome/figures/) | Track E1 切除结局预测变量（覆盖景观/对比） |
| [template_ablation_coverage/figures/](template_ablation_coverage/figures/) | 模板消融覆盖 |

---

## 其他（无 curated README 的图目录）

以下目录有图但没写 `figures/README.md`，含义需看对应 archive doc：
`spatial_modulation/figures/`、`refine_soz_validation/yuquan/figures/`、`topic4_sef_hfo/figures/`、
`topic4_sef_hfo/observation_layer/snn_cm_spontaneous/regime_screen/figures/`、`topic4_sef_itp/swap_mechanics/figures/`。

## 保留不动的大目录（非图，供参考）

`_legacy_2021_readonly/`（1.6G 引用基线，只读）、`hfo_detection/` `hfo_detector_v2/`（检测产物/输入）、
`interictal_synchrony/epilepsiae_ready_full_artifacts/`（2.9G 同步输入）、`_cold_archive/`（phantom-superseded 打包冷藏）。
