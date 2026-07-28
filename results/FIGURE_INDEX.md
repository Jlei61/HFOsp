# results/ 图总索引（FIGURE INDEX）

> 解决"重要的图埋太深、容易错过"的问题。本文件是 `results/` 下所有**结论级图**的导航入口。
> 每个目录里仍有自己的中文 `figures/README.md`（逐图说明），本文件只负责"按 topic 快速定位 + 指路"。
>
> 用法：先看下面「只看这几张」挑代表图；要细节就点到对应目录的 `README.md`。
> per-subject / per-seizure 的诊断图（占全部图的 ~75%）不在本索引里逐一列出——它们藏在各目录的
> `per_subject/` `per_seizure/` `subjects/` 子目录中，是单被试核对用，不是结论图。
>
> 最近更新：2026-07-28。新增结论图目录时，请在对应 topic 表里补一行。
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
| 5 V2 state 层 | [topic5_ictal_recruitment/v2_criticality/figures/phase2_state_layer_alignment.png](topic5_ictal_recruitment/v2_criticality/figures/) | 发作前"变脆状态"是否沿间期 HFO 几何排布（EXPLORATORY，偏阴性；动力学 leg 0/16 显著）|
| 5 子型×方向 | [topic5_ictal_recruitment/subtype_direction/figures/cohort_C_to_A_connection_broadband.png](topic5_ictal_recruitment/subtype_direction/figures/) | C 线：发作子型是否决定激活方向 + 与 A 线奇偶不稳的连接（队列不可行/无信号） |
| 5 方向两类 | [topic5_ictal_recruitment/directional_clustering/figures/epilepsiae_442__classes_vs_interictal_hist_broadband.png](topic5_ictal_recruitment/directional_clustering/figures/) | 发作方向无监督两类是否对应间期 A/B（6 ECoG 全无 two_class_mapped；442 真两堆但对不齐；防自欺 null） |

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
| [paper-ready-figure/fig2c_interictal_event_envelope_field/figures/](paper-ready-figure/fig2c_interictal_event_envelope_field/figures/) | **Fig2-C candidate**：E1146 两次真实 TA/TB 代表事件的 Fig1a 质心 readout + 冻结 shared-plane 包络 frame；同目录含 2 ms 生物学步长 TA/TB GIF。代表性 timing cross-check，非 template-free/cohort/traveling-wave 证明；规范 `docs/fig2c_interictal_event_envelope_field_spec.md` |
| [paper-ready-figure/fig2e_interictal_template_fields/figures/](paper-ready-figure/fig2e_interictal_template_fields/figures/) | **Fig2-E candidate**：冻结 TA/TB 模板本身的患者特异静态 rank field；与 Fig2-C 分工不同 |

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
| [paper-ready-figure/fig_subject_snn_epilepsiae_1146/figures/](paper-ready-figure/fig_subject_snn_epilepsiae_1146/figures/) | **Fig4A/B/C**：E1146 真实电极布局 subject-specific SNN readout（A）+ KMeans k=2 核验（B）+ 模型 vs 真实间期模板一致性（C，forward~t_a ρ+0.87 / reverse~t_b ρ+0.62） |
| [paper-ready-figure/fig_m3a_v2_2_hG_runaway_transition/figures/](paper-ready-figure/fig_m3a_v2_2_hG_runaway_transition/figures/) | **M3A-v2.2 全局恢复 h_G 单轨迹 GIF**（visual diagnostic，非 sweep）：间期 axis-like→runaway 转变，把全局抑制恢复标量 `h_G` 打开。`h_G` 只在 runaway 时升起（局部事件期 χ_G≈0），但 runaway **未被逆转**；`eta_G` 阶梯 0→80（>10× 膜跨度）结构性无效——减法式全局刹车拉不回饱和 recurrent 雪崩，瓶颈在 E→E 衬底。配 v2.1 `fig_m3a_v2_1_qigk_runaway_transition`（同轨迹 h_G OFF） |
| [topic4_criticality/figures/](topic4_criticality/figures/) | **M3A-v2.2 criticality M1**：frozen-Jacobian 轨迹判读；采样快照跨过失稳边界但没有直接命中 crossing，故 `csd_verdict=unresolved_operating_point`，作为历史诊断保留 |
| [topic4_criticality_m2/figures/](topic4_criticality_m2/figures/) | **M3A-v2.2 criticality M2（2026-07-12 有界验收）**：线性起燃模式为 `core_localized`；非线性铺开对扰动强度/极性敏感，维持 `undetermined`。这是 M4 的上游诊断，不替代 M4 的恢复/终止设计 |
| [paper-ready-figure/fig_m4_dynamic_qi/figures/](paper-ready-figure/fig_m4_dynamic_qi/figures/) | **M4 pass-1 除法共享抑制池 = bounded 第三态机制筛选通过，非完整 seizure cycle（2026-07-09 验收）**：活动依赖的除法池 `S_G` 打开窄窗口 bounded sustained attractor；40s 多 seed 稳健锚点 `(k_q=0.10,aG16)` 为 3/4 seed。限制：空间宽、marginal、不可撤回/不自终止，故结论是池只 bound 不 terminate；matched-subtractive 仍失控、clamped-SG 压死活动。连续性/经验本征模只支持 Hopf-like 迹象，不能写成严格 Hopf。验收文档 `docs/archive/topic4/sef_hfo/m4_pass1_divisive_shared_pool_acceptance_2026-07-09.md`；已于 2026-07-12 合入 `main` |
| [topic4_m4_dynamic_p1_sweep/figures/](topic4_m4_dynamic_p1_sweep/figures/) | **M4-2 STD 终止器 P1 sweep = 3-seed scoped clean no-go（2026-07-08）**：同一有界工作点扫 `ee_std_u × ee_std_tau_ms`，全网格 0 `terminate_clean`；弱 STD 碎裂，强/慢 STD 压死，中间无干净终止，且 seed4 Arm0 本身 fragment。只限此衬底/工作点/网格/3 seed，不外推为普适失败。archive `docs/archive/topic4/m4_2_std_termination_p1_sweep_2026-07-08.md`；已于 2026-07-12 合入 `main` |
| [topic4_sef_hfo/m4_snn_native_exit/figures/](topic4_sef_hfo/m4_snn_native_exit/figures/) | **M4 SNN-native containment→exit = 开环+对称退不出；不对称 slow-release=候选（2026-07-21）**：给 M4 有界态加持续时间门控局部恢复电流 `p(x,t)`。开环抬阈值（seed1、5 hold、`q_I` 到 0.87）+ 对称闭环电流（2 seed）都退不出（rebound / lower-rate-persistent / runaway）；**但真·不对称 slow-release（快充 τ_p3000/慢放 τ_p_down12000）质变=有希望候选**：`no_runaway`、活动被压到 0 → `q_I` 回灌 0.6–0.8 → 之后离散短促自终止 burst、不再回宽持续态。**未确证**（快充可能 prevention 非 terminate、burst 未验、seed1、退出 basin 未映射）。`fig5_exit_attempts_diagnostic.png`（⚠️ 旧 `fig5_no_go_diagnostic.png` 第四列因 `d_sweep` 漏传 `tau_p_down` 的 P0 bug 实为对称、已作废删除）。archive `docs/archive/topic4/sef_hfo/m4_snn_native_exit_execution_2026-07-21.md`；分支 `codex/topic4-m4-snn-native-exit`（未合 main）|
| [topic4_sef_hfo/zm_ictal_carrier_gate/figures/](topic4_sef_hfo/zm_ictal_carrier_gate/figures/) | **Z/M ictal-carrier 门 seed-1 = HFO 样爆发串，非持续 carrier（2026-07-24 NO-GO）**：在原始各向异性 Z/M(+S_G) SNN 上，用**跑前锁死**的两层门判 sg 态是否为持续发作载体——门 A 源空间核心率 macroepisode（≥2 s 且 occupancy ≥80%），门 B 电极 30–80/80–150/1–150 Hz 能量包络 occupancy（虚拟 SEEG 存 2 kHz、Nyquist 1 kHz > 150 Hz）。**三臂全否**：sg=`fail_hfo_like_train`（源 occupancy **0.17**、核心峰 455 Hz 但爆发间掉回基线；汇侧 ICL8–11 有 14–32 dB 的 30–80 Hz 凸起但最优触点 occupancy 仅 0.55<0.80→0 持续触点）、bare=`fail_runaway`（2871.8 ms 截断）、interictal_ctrl=`fail_hfo_like_train`。Section-8 慢-快=`transient_burst_train`（IBI≈300 ms 平稳、幅度随 z 耗竭 escalate 0.55、S_G 滞后核心 65 ms＝松弛泵），**非** limit cycle。机制瓶颈=单一**全局** S_G 同步 reset 整个核心→窄爆发串；下一步 **Path B**（patchwise 抑制造 carrier），**停** H 扫描。**⚠️ v1 门实现偏离预注册 spec（onset/baseline/B2/A7/A8），已用 revised-protocol v2.1（`carrier_gate_v2`）离线重判修订（archive §10）：源根本无持续 onset（更干净爆发串）；电极有触点 occ 达 0.8-1.0 但持续簇只 ~0.6s、**0 个过完整持续门（occ≥0.8 且 dur≥2s）**（占空非稀疏，缺的是连成 ≥2s macroepisode）；定性 NO-GO 稳。** archive `docs/archive/topic4/sef_hfo/zm_ictal_carrier_gate_2026-07-24.md`、spec `docs/superpowers/specs/2026-07-24-topic4-zm-ictal-carrier-gate-design.md`；分支 `codex/topic4-m4-snn-native-exit`（未合 main）|
| [topic4_sef_hfo/zm_patch_screen/figures/](topic4_sef_hfo/zm_patch_screen/figures/) | **Path-B cheap-first screen：去同步局部抑制假设通过 reduced-model plausibility 筛选（2026-07-24，非验证）**：carrier 门 NO-GO 后，不改 E→E、只在抑制侧把单一全局标量 S_G 拆成空间分辨结构，在**高度简化的 K-patch rate 模型**（非 SNN、无 Z/M、无二维 E/I、无各向异性、patch index 是环非物理空间、`w_rec` 人为放入制造振荡）筛哪种结构能填平群体波谷。四种结构（K=16、异质、4 seeds、OFF 态为基线的群体 occupancy + 跨 patch 同步度）：global 标量（homogeneous）=同步爆发串（occ 0.52、sync +1.0、**0/4**，SNN 复现）、global（heterogeneous）=死不动点（occ 1.0 但不振荡、0/4）、**patchwise 独立局部池=去同步+仍振荡+群体不塌到 OFF（occ≈1.0、sync +1.0→+0.04、4/4）**、patchwise+平滑/local+weak-global 也 4/4。**结论=假设值得移植（非"正确杠杆已验证"）**：下一步在原始 SNN 上实测（仅抑制侧），仍须过预注册 A+B 门。⚠️rate 代理非 LFP carrier、群体仍 ~1.3× 调制、图里"斜纹"只是 phase-staggered patch activity 非空间波。图 `patch_screen.png`；模型 `src/topic4_zm_patch_screen.py`（`carrier_proxy` 已把去同步写进 pass 条件）、测试（8 green）。archive §9-§10；分支 `codex/topic4-m4-snn-native-exit`（未合 main）|
| [topic4_sef_hfo/zm_field_screen/figures/](topic4_sef_hfo/zm_field_screen/figures/) | **简化 2-D 场：局部 vs 全局抑制线性稳定性筛查（2026-07-25，`both_stable`）**：Path-B patch screen 之后，在**简化 2-D 场模型**（非 SNN；**新构造的分裂+减法双成分抑制池振子**，**不是** Z/M 慢变量的 mean-field 等价物——当前 Z/M `sg` arm 是 β=0 纯分裂式、**根本不振荡**，故本轮**无法**回答"把现有 Z/M 池局部化能否造载体"；各向异性椭圆 E→E 核 AR=2 + 各向同性高斯 S 池，轴向固定 0 rad、**无病人电极几何**）上直接问局部化假设——把抑制反馈从全局改成空间局部，是否会让均匀振荡失稳、裂成相位交错的空间斑图？四面板：(A) 锁定工作点上的均匀 mean-field 轨道确实在振荡（周期≈170 ms，r/μ/S 三条轨迹）；(B) `dual_global` vs `dual_local` 在整数空间模态 `mx,my`（取遍 -4..4）上的 transverse 增长率热图（DC 模态除外，共用以 0 为中心的对称色标）——两者全负；(C) 5 档兴奋性 `I0` × 4 种抑制拓扑（`div_global`/`dual_global`/`dual_local`/`dual_mixed`）的最大增长率折线，全部落在零线以下、离 ±0.002 判断地板还有余量——但 `div_global`（beta=0 消融）在这个工作点本身并不振荡（已退化到不动点，见 `test_divisive_only_beta0_has_no_orbit`），这条线不是一个独立的稳定性结果，只用来标注"减法抑制项才是振荡的成因"这一单独事实，图上已调浅调细并改图例文字标注，不可与另外三条同台比稳；(D) 3000 ms 短诊断跑的 r(x,y) 快照 + 群体均值 r(t)（`dual_global` vs `dual_local`，seed 0），肉眼确认场保持空间均匀、两条均值曲线几乎重合。**结论=`both_stable`**：**收尾复核已把扫描从图上的 81 个模态（|m|≤4）扩到 n=32 网格的全部 513 个独立非 DC 模态——每档每臂 513/513 增长率严格为负**（全场最大 −0.0033，`any_mode_above_floor=False`；5 档全过 dt/dt-半 符号检查，见 `floquet_full_spectrum.json`），没有造出局部抑制诱发的失稳分岔。⚠️**早稿"局部离零近 ~15 倍＝方向支持假设"已撤回**：最不稳模态恒是盒子能装下的最长波长，那里增长率必然趋近均匀态自身的中性方向（构造必然、非到分岔的距离），而真正的图案模态还更稳 3–5 倍；(D) 的快照只是单 seed 短程序跑的直观核验，不是 (B)/(C) 的承重判据。脚本 `scripts/plot_topic4_zm_field_screen.py`；数据 `results/topic4_sef_hfo/zm_field_screen/{phaseA_lock,floquet_map,floquet_full_spectrum,field_screen_summary}.json`；分支 `codex/topic4-m4-snn-native-exit`（未合 main）|
| [topic4_sef_hfo/zm_branch_decision/figures/](topic4_sef_hfo/zm_branch_decision/figures/) | **Z/M minimal-carrier Rev3.1 branch decision（2026-07-28，诊断性收口）**：三个 primary seed 的自然 slow-state forks 确认 visited frozen states 上存在可持续的 **source-space tonic carrier**，代表 continuation 约 151 Hz、CV≈0.001；但跨 seed fine-source rhythm class 不一致，真实 observation reference 被阻断，Z-entry 未括住，functional rank 无证据。existing-coordinate offset 按 fail-closed 判为 `no_evidence`：M 与 M+S_G 不退出，static M+Z-recovery 非单调且无 low-basin coexistence，真实 dynamic Z/M 为 **0/9 offset、9/9 runaway**。顶层 verdict=`carrier_at_visited_states`；不是 bounded ictal oscillation 或 recoverable lifecycle，不授权 Phase 3/actuator。archive `docs/archive/topic4/sef_hfo/zm_minimal_carrier_branch_decision_2026-07-28.md`；本目录仅作诊断/负结果证据，不是 Figure 5 候选。 |
| [paper-ready-figure/fig_m3a_v2_2_qI_runaway_transition_epilepsiae_1146/figures/](paper-ready-figure/fig_m3a_v2_2_qI_runaway_transition_epilepsiae_1146/figures/) | **M3A-v2.2 q_I 载体 + 轴向 g_K 疲劳 runaway GIF（E1146 真实电极几何）**（visual diagnostic）：顶部画 **抑制资源 `q_I(t)`（mean+min）+ 轴向区域 `g_K` 疲劳**（h_G 关、左图 E→E 梯度画成椭圆）。看 `q_I`（min=轴向走廊先耗竭）掉到地板 + `g_K` 累积→局部小事件铺成 runaway（757 ms）。`g_K` 膜耦合关（eta_K=0，只可视化不改轨迹）；**真耦合 eta_K=1 反而在点火前就压住核、阻止 runaway**（=limit 角色，另一张图）。电极=ICL/SCL 真实触点 |
| [paper-ready-figure/fig_m3a_v2_2_qI_stim_runaway_epilepsiae_1146/figures/](paper-ready-figure/fig_m3a_v2_2_qI_stim_runaway_epilepsiae_1146/figures/) | **M3A-v2.2 刺激 vs 不刺激 对照 GIF（E1146 真实电极几何）**（visual diagnostic，外部预防式压制示意，非治疗/recovery 主张）：与 `fig_m3a_v2_2_qI_runaway_transition` 同轨迹，两臂唯一区别=刺激臂在 500–1400 ms 把中段 4 触点 `ICL4–7` 附近 E 细胞 V_th clamp。两臂在刺激开前**逐比特一致**。不刺激 runaway 758 ms；刺激把间期事件压掉→`q_I` 少耗竭（窗内基本不掉，叠 no-stim 灰虚线对照）→runaway 推后到 **1592 ms（+834 ms）**，且**关刺激后才反弹**（窗内不发生）。布局=2 行（上不刺激/下刺激）× `permissivity` \| `2D 活动` \| (`q_I/g_K 轨迹` \| `SEEG readout`)。脚本 `plot_fig_m3a_v2_2_qI_stim_runaway_gif.py` |
| [paper-ready-figure/fig_m3a_v2_2_qI_stim_site_compare_epilepsiae_1146/figures/](paper-ready-figure/fig_m3a_v2_2_qI_stim_site_compare_epilepsiae_1146/figures/) | **M3A-v2.2 刺激位点对照：最早端点 vs 中段（E1146 真实电极几何）**（visual diagnostic，非治疗主张）：同轨迹、两条刺激臂比哪个把 runaway 推得更后。上行=刺激最先点火的灶端点 `ICL8–11`、下行=刺激中段 `ICL4–7`，都在 500–1400 ms。不刺激 runaway 758 ms；**端点只推到 1171 ms（+414，窗内就击穿）、中段推到 1592 ms（+834，关刺激后才反弹）→中段更狠**。机制=打端点只掐掉一个灶（另一灶+中段走廊仍在磨 `q_I`）、打中段堵走廊让两灶事件都传不过去。脚本 `plot_fig_m3a_v2_2_qI_stim_runaway_gif.py --mode endpoint_vs_middle` |
| [paper-ready-figure/fig_m3a_v2_2_qI_stim_both_foci_epilepsiae_1146/figures/](paper-ready-figure/fig_m3a_v2_2_qI_stim_both_foci_epilepsiae_1146/figures/) | **M3A-v2.2 刺激两个灶都打 vs 不刺激（E1146 真实电极几何）**（visual diagnostic，非治疗主张）：刺激臂在 500–1400 ms 把**两个灶端点** `ICL1–4`+`ICL8–11`（8 触点）都 clamp。不刺激 runaway 758 ms；两灶都打推到 **1606 ms（+848）、关刺激后才反弹**——和只打中段（+834）**几乎打平**，虽然触点数/覆盖 E 细胞翻倍（3102 vs 1749）。说明"堵中段走廊"≈"掐两个灶"，都靠窗内保住 `q_I`；延迟主要由刺激窗时长决定。脚本 `plot_fig_m3a_v2_2_qI_stim_runaway_gif.py --mode no_stim_vs_both_foci` |
| [paper-ready-figure/fig_stage4_axis_vs_core_difficulty/figures/](paper-ready-figure/fig_stage4_axis_vs_core_difficulty/figures/) | **挡轴 vs 打灶 + 自发难点图**（visual diagnostic，非临床证明；runaway 非 ictal）：两张图。**难点图** `difficulty_3row`（3 行 big/small/kick）=为什么自发单灶出不了事件串——大核 23 ms 整片同步爆（1 事件、铺满 1.0）、小核 42.8 ms 前锋铺满整片（1 事件、1.0）、只有外部戳的两灶先出 **3** 次分开事件（`q_I` 3 档台阶、col1 contained frac_ever=0.22）再 757 ms 失控。**轴 vs 灶图** `axis_vs_core`（固定 footprint=4）=两灶+中段走廊（有咽喉）挡轴 **+834** ≥ 打灶 **+414**（引用 E1146）；单中心核（无咽喉、径向漏）打灶 **+37** > 挡轴 **+8**＝诚实压力测试不成立。脚本 `plot_fig_stage4_axis_vs_core_difficulty.py` + `run_stage4_axis_vs_core_stim.py`；归档 `docs/archive/topic4/axis_vs_core_stim_2026-07-02.md` |
| [topic4_sef_hfo/linear_stability/figures/](topic4_sef_hfo/linear_stability/figures/) | Step 0a：LIF 自洽工作点 |
| [topic4_sef_hfo/finite_pulse/figures/](topic4_sef_hfo/finite_pulse/figures/) | Step 0b/0d：LIF rate field 真实场 |
| [topic4_sef_hfo/step1_noise/figures/](topic4_sef_hfo/step1_noise/figures/) | Step 1：drive × σ 联合分析 |
| [topic4_sef_hfo/lif_snn/figures/](topic4_sef_hfo/lif_snn/figures/) | LIF ↔ spiking-network 验证 |
| [topic4_sef_hfo/low_rate_template_stability/figures/](topic4_sef_hfo/low_rate_template_stability/figures/) | 低事件率：传播模板 vs 发放计数复现度 |
| [topic4_sef_hfo/snn_heterogeneity/figures/](topic4_sef_hfo/snn_heterogeneity/figures/) | **SNN 阈值异质核 sweep**（headline: mean_scan / sweep_ignition；mechanism_* 是各 kick×core 组合） |
| [topic4_sef_hfo/skeleton_geometry/figures/](topic4_sef_hfo/skeleton_geometry/figures/) | 几何骨架（per-subject 在子目录） |
| [topic4_sef_hfo/observation_layer/figures/](topic4_sef_hfo/observation_layer/figures/) | 虚拟 SEEG 观测层 |
| [topic4_sef_hfo/observation_layer/increment3a_rate_parity/figures/](topic4_sef_hfo/observation_layer/increment3a_rate_parity/figures/) | rate parity 增量 |
| [topic4_sef_hfo/observation_layer/snn_cm_spontaneous/figures/](topic4_sef_hfo/observation_layer/snn_cm_spontaneous/figures/) | cm-SNN 自发（headline: stage2_summary / stage3_regime_compare；**M2 刹车 core_model 三行图: s3_brakeoff/brakeon + s4_brakeoff/brakeon**，刹车不复现沿轴空间自限=诚实 NULL） |
| [topic4_sef_hfo/observation_layer/snn_cm_spontaneous/candidate_confirm/figures/](topic4_sef_hfo/observation_layer/snn_cm_spontaneous/candidate_confirm/figures/) | 候选格电极读出 train |
| [topic4_sef_hfo/observation_layer/snn_cm_spontaneous/a1_formal/figures/](topic4_sef_hfo/observation_layer/snn_cm_spontaneous/a1_formal/figures/) | **axis-A A1** 阈值离散→指纹 = NULL（只改点火率不改指纹） |
| [topic4_sef_hfo/observation_layer/snn_cm_spontaneous/a3_0a_scan/figures/](topic4_sef_hfo/observation_layer/snn_cm_spontaneous/a3_0a_scan/figures/) | **axis-A A3** 局部 E/I 病灶 screen = NULL（不复现 V_th↓ 方向模板） |
| [topic4_sef_hfo/observation_layer/snn_cm_spontaneous/ei_param_scan/figures/](topic4_sef_hfo/observation_layer/snn_cm_spontaneous/ei_param_scan/figures/) | **axis-A E/I 参数扫描** broad-basis NULL（无"既安静又出模板"甜区） |

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
| [data_driven_soz/layer_a_ictal_er_rank/atlas_v2_3/figures/](data_driven_soz/layer_a_ictal_er_rank/atlas_v2_3/figures/) | ictal ER-onset timing atlas（per_seizure：全 371 张横向布局[上gamma/下broad，左raw/右heatmap，y轴对齐]，Epilepsiae 339 + Yuquan 32，EEG-zoom ±90s，z-ER bin=0.1s 与 field 一致；见 figures/README.md） |
| [data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/figures/](data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/figures/) | PR-1 z-ER 子型聚类（per-subject 在子目录） |
| [topic5_ictal_template_echo/figures/](topic5_ictal_template_echo/figures/) | Stage 1 ictal-template-echo（anchor vs path） |
| [topic5_dynamic_echo/sentinel/figures/](topic5_dynamic_echo/sentinel/figures/) | Stage 2b 动态回响哨兵 |
| [topic5_ictal_recruitment/axis_alignment/figures/](topic5_ictal_recruitment/axis_alignment/figures/) | A 线 axis alignment 可视化 |
| [topic5_ictal_recruitment/contact_similarity/figures/](topic5_ictal_recruitment/contact_similarity/figures/) | 触点相似性几何阶梯（R1 无几何 / R2 同平面触点核 / R3 场）+ R2b native-3D 灵敏度：相似度数值抬高主要来自平面几何平滑，但平滑也抬高轴内 null；网格无可分辨增益，native-3D vs 2D-plane 等价通过。属于 sensitivity，不升级主张；producer `scripts/plot_topic5_cs_paper.py` |
| [paper-ready-figure/fig3a_raw_spectral_context/figures/](paper-ready-figure/fig3a_raw_spectral_context/figures/) | **Fig3-A 正式版 raw spectral context**：E1146 seizure 7；15 条 lagPat joint-valid raw SEEG 与 SCL9 baseline-normalized TFR 严格共用 `[-120,+20] s` 时间轴，右侧固定 low bands (1–30 Hz) / gamma (30–80 Hz) / high-gamma (80–150 Hz) / broadband (1–150 Hz) 2×2。只标 baseline 和 clinical-onset `[0,10) s` 阴影；无 EEG onset / onset 竖线 / 内部 a-b 编号。右侧同行共享 y limits，ticks 只在左图。复现 `scripts/paper_figures/plot_fig3_raw_spectral_context.py --subject epilepsiae_1146 --seizure-idx 7`；规范 `docs/figure_style_guide.md` §5a。 |
| [paper-ready-figure/fig3b_interictal_ictal_shared_field/figures/](paper-ready-figure/fig3b_interictal_ictal_shared_field/figures/) | **Fig3-B candidate shared-field pair**：左为冻结 TA timing field（红色 `TA fields` 标题），右为 E1146 25 次 complete/exact 1–150 Hz 发作中 `shared_a_signed` 最高的 seizure 15 clinical 0–10 s robust-z broadband power 场。15/15 exact-name join；左右同 shared plane / TA support / extent / 6 mm display sigma；右图 `magma_r`、无 rank/sign flip、保留完整边框，双 xlabel；colorbar 显示真实 propagation rank / robust-z，深色统一表示最早传播或最高 power。best-case representative，非独立 replay/cohort/机制证据；规范 `docs/fig3b_interictal_ictal_shared_field_spec.md`。 |
| [paper-ready-figure/fig3_field_concordance_cohort_stat/figures/field_concordance_cohort_stat.png](paper-ready-figure/fig3_field_concordance_cohort_stat/figures/) | **Fig3 field-concordance cohort statistic（panel 编号待总拼版）**：Data vs channel-null paired subject-level 面板；`BB 1-45 maxAB`、line-noise-masked `BB 1-150 maxAB`、`HFA 60-100 maxAB` 三组均显示 cohort-level shift above null；formal pass 仍看 selection-corrected 表。 |
| [paper-ready-figure/fig3_sup2_raw_spectral_context/figures/](paper-ready-figure/fig3_sup2_raw_spectral_context/figures/) | **历史路径**：Fig3-A 定稿前的 raw spectral context 输出；保留溯源，不再作为 canonical panel 引用。 |
| [paper-ready-figure/fig3_peri_onset_field_similarity/fig3_peri_onset_run_manifest.json](paper-ready-figure/fig3_peri_onset_field_similarity/fig3_peri_onset_run_manifest.json) | **Fig3-C 二维 shared-gradient peri-onset trajectory**：只消费 fingerprint-valid `shared_a/shared_b`，不 fallback own，并要求 `geometry_2d_supported=true`。denominator flow：40 frozen → 14 shared/fingerprint → 12 二维 → 10 有 inventory → 7 有 derived cache → **7 出图**；coverage=`complete_ok 3 / partial_ok 3 / severely_partial 1`，E583 仅 3/22。当前 canonical run=`20260718T071020Z_d99c96ec`，manifest 指向 immutable run artifacts；subset/中断不改 canonical。E139 单杆仅在 `sensitivity_1d/`；raw 个体级描述，非 cohort gate / onset-emergent alignment。 |
| [paper-ready-figure/fig3_peri_onset_field_similarity/spatial_null/figures/](paper-ready-figure/fig3_peri_onset_field_similarity/spatial_null/figures/) | **Fig3-C shared-matched maxAB 空间 null（fixed-time-mapping v2）**：同 7 人、同 fingerprint、同成功 seizure 集、同 `shared_a/shared_b` scorer；每个 seizure×replicate 的空间映射固定贯穿 66 窗，R=1000。vectorized/exact 与 source observed 最大误差 ≤`5.6e-14`；3/7 有 within-shaft cluster、2/7 有 maxT 窗，只作 per-subject 时间分辨描述。旧逐窗置换 `5/7` 已撤回；不能替代 frozen archive 的 cohort n=7 shared-field null（p=0.346）。manifest 含 summary，共 35 个 null artifacts。 |
| [topic5_ictal_recruitment/v2_band_timecourse/figures/](topic5_ictal_recruitment/v2_band_timecourse/figures/) | **多频带 peri-onset field-similarity 时程图（Fig3-Sup1 的时间分辨扩展，2026-07-08，EXPLORATORY 候选骨架）**：读 `v2_band_scan/cache` 的 masked band-power robust-z，`[-120,+20]s`/10s/2s 滑窗，用与 Fig3-C 相同的 formal-plane mirror-invariant signed-corr，逐频带（primary 7 + composite 4）算 maxAB\|r\|+signed A/B。每被试 band×time 热图（maxAB\|r\| + signed A/B sidecar）+ primary-7 三组线图（median±IQR + 1–150 Fig3-C 参考）；cohort band×time 热图 broad(17)/narrow(20) **各一张永不 pool** + `cohort_pre_vs_early_delta_{broad,narrow}.csv`。20/20 cached subjects ok。**描述性结论**：pre-onset 全频段 cohort 中位 \|r\|~0.70–0.76（**band-generic、preictal-present**）、early−pre 增量绝大多数 ≈±0.02、**只 δ/低频小上抬**（δ narrow +0.071/13-of-20、broad +0.080/10-of-17）→ 偏 band-generic-preictal-present + δ 小增量、非 onset 频带分化。**禁** formal Gate/HFO-/ripple-/FR-specific/机制。脚本 `scripts/plot_topic5_multiband_field_similarity_timecourse.py`，索引 `../multiband_timecourse_subject_index.{csv,json}` |
| [topic5_ictal_recruitment/subtype_direction/figures/](topic5_ictal_recruitment/subtype_direction/figures/) | C 线 子型×激活方向（玫瑰 + C↔A 连接） |
| [topic5_ictal_recruitment/directional_clustering/figures/](topic5_ictal_recruitment/directional_clustering/figures/) | 发作方向无监督两类 ↔ 间期 A/B（每被试玫瑰 + 442 间期事件 hist×两色发作方向类；exploratory negative，6 ECoG 全无 two_class_mapped） |
| [topic5_ictal_recruitment/event_resolved_alignment/figures/](topic5_ictal_recruitment/event_resolved_alignment/figures/) | A 线 event-resolved 二级：逐间期事件(A/B 两类)对发作场的对齐分布 + 事件级离散度（PILOT 3 epi，exploratory，broad 底物） |
| [topic5_ictal_recruitment/field_dynamics/figures/](topic5_ictal_recruitment/field_dynamics/figures/) | **发作内 field 动力学 pilot — broad 9（8 swap + E916 非swap）**；field_vs_ictal 形式锚发作前：间期 A\|B 锚 + 平均早期场 + progress/offset summary + 每 subject 1 个 **field 演化 GIF**（onset→offset 直观看场传播）；方向(轴向降/非轴向升)broad 有暗示但 narrow 证否=不稳健；z-ER 中后期示意 |
| [topic5_ictal_recruitment/field_dynamics_narrow/figures/](topic5_ictal_recruitment/field_dynamics_narrow/figures/) | 同上 **narrow 平行批 7**（用间期模板端点 compact core 构轴，证明**不必 swap**）；narrow 多数被试方向反向 → broad 的方向暗示不复现（扩队列证否），exploratory |
| [topic5_ictal_recruitment/event_resolved_alignment/figures/fields/](topic5_ictal_recruitment/event_resolved_alignment/figures/fields/) | 同上的**场图版**（用户 pivot）：A 类全事件场 \| B 类全事件场 \| 发作场，同一物理平面、左图形式 |
| [topic5_ictal_recruitment/event_resolved_alignment/figures/class_vs_template_field_similarity_cohort.png](topic5_ictal_recruitment/event_resolved_alignment/figures/) | cohort 级：用 class 事件造的间期场 ≈ 用模板投影造的间期场（N=23 中位 \|r\|=0.985；构念等价） |
| [topic5_ictal_recruitment/event_resolved_alignment/class_vs_template/figures/](topic5_ictal_recruitment/event_resolved_alignment/class_vs_template/figures/) | **6 面板/被试**（全 12 epi）：模板A\|模板B\|发作前−10..0s / 类A场\|类B场\|发作前−120..−90s；配 max-AB 统计（class 场 ≈ template 场） |
| [topic5_ictal_recruitment/field_extrapolation/figures/](topic5_ictal_recruitment/field_extrapolation/figures/) | **间期场外推到发作隐身电极 pilot**（2026-06-30，exploratory-negative）：间期 broad 顺序场预测 broad∖narrow 隐身电极发作 z-ER 招募序，比"场 F vs 逐通道 C vs 半径"；**被数据现实阻塞**——队列 ~15/16 发作排序本身不稳(s_sz<0.3)，唯一干净被试 583 场无增益(F≈C≈radial≈0.22)；印证"粗骨架共享/细发作招募不稳" |
| [topic5_ictal_recruitment/field_extrapolation/figures/cohort_energy_F_core_vs_baselines_{bb,hfa}.png](topic5_ictal_recruitment/field_extrapolation/figures/) | **能量基础 cohort（2026-07-01）**：F_core_only(核心场留一预测隐身电极发作能量) vs C1/C2 baselines，16-subject；延伸成立(过 null 两频段)、但场必要性不立(never beats C1)；FDR 表 `energy_field_extrapolation_FINAL.md` |
| [topic5_ictal_recruitment/field_extrapolation/figures/triptych_preview/](topic5_ictal_recruitment/field_extrapolation/figures/triptych_preview/) | **1-3 vs 2-3 单被试图（16×2 频段=32 张，2026-07-01）**：单行 5 格=发作能量场\|1-3 core外推场\|2-3 自身order场\|colorbar\|(D)逐发作\|corr\|箱线；隐身=红方块三场对齐；问"core 外推能否比电极自身间期顺序更好预测发作能量" |
| [topic5_ictal_recruitment/field_extrapolation/figures/cohort_1v3_vs_2v3_summary.png](topic5_ictal_recruitment/field_extrapolation/figures/) | **1-3 vs 2-3 cohort 汇总**：每频段配对图(F_core_only→C1) + 计数；**大致打平** broadband 7:5:4、HFA 5:8:3(16-subject) → 外推不系统性赢过自身间期顺序；590/1146 nhid=4 低功率；报告 `docs/archive/topic5/field_extrapolation_1v3_vs_2v3_report_2026-07-01.md` |
| [paper-ready-figure/fig_topic5_field_extrapolation_energy/figures/topic5_field_extrapolation_energy_main.png](paper-ready-figure/fig_topic5_field_extrapolation_energy/figures/) | **Topic 5 energy-field paper-ready 主图**：A E1146 真实电极布局上的测试设计（core-field vs own-order 预测 hidden seizure energy）/ B cohort Δ(F_core−C1) 直接裁决 / C 证据阶梯；结论=network extension supported，但 added advantage over own order 不支持 |
| [paper-ready-figure/fig_topic5_network_extension_null/figures/topic5_network_extension_three_way_comparison.png](paper-ready-figure/fig_topic5_network_extension_null/figures/) | **Topic 5 network-extension three-way 统计图**：每频段同图比较 core-field prediction / hidden own-order C1 / channel-shuffle null。Core 和 Own 都显著高于 null（network signal exists）；Core 不系统性赢 Own（added advantage 不成立；BB core>own/own>core/tie=7/5/4, p=0.297；HFA 5/8/3, p=0.872） |
| [paper-ready-figure/fig_axis_representativeness/figures/axis_representativeness_cohort.png](paper-ready-figure/fig_axis_representativeness/figures/) | **间期 TA/TB gradient 轴的单事件方向代表性**：subject 内等权折叠 TA/TB，在所有 axis 可拟合且两类均有足量二维 QC-clean 事件的患者中（主分析 n=26），检验真实轴的 mean signed cosine 是否高于同 montage template-rank-shuffle null；strict-stability n=13 仅作 sensitivity，不比较 endpoint，不含 ictal 输入 |
| [topic1_topic5_bridge/figures/](topic1_topic5_bridge/figures/) | Topic 1 模板 × Topic 5 亚型 桥接（q1 / q1prime 系列） |
| [template_resection_outcome/figures/](template_resection_outcome/figures/) | Track E1 切除结局预测变量（覆盖景观/对比） |
| [template_ablation_coverage/figures/](template_ablation_coverage/figures/) | 模板消融覆盖 |
| [topic4_sef_hfo/event_extent_audit/figures/](topic4_sef_hfo/event_extent_audit/figures/) | M2 Task 0：真实事件轴向铺满 vs 收一段（实测≈随机对照→非自限） |
| [topic5_ictal_recruitment/v3_mode_transition/figures/](topic5_ictal_recruitment/v3_mode_transition/figures/) | V3a：发作开始前后，活动是否从间期固定电极通路搬到通路外（**每队列一张、真实秒数横轴**：主图 `v3_axis_vs_offaxis_{narrow,broad}` 沿轴组织度 vs 离轴流随时间、扣发作前基线，看 onset 前后是否反向张开；附图 `v3_mode_direction_{narrow,broad}` 模态方向阴性）；EXPLORATORY **脆弱阳性**——配对后非轴向净流增量在主(Holm 0.031)+复制(0.008)队列都队列级显著(tier4 机械/supported)，但 null-relative(原始流大多在降)+同时共激活为主(lag1≈lag0)+个体稳健性弱(流腿 0/7)+模态腿全阴 → 数据侧候选信号非确立的模态转移，pending sensitivity/V3b。详见 docs/archive/topic5/v3a_mode_transition_2026-07-04.md |
| [topic5_ictal_recruitment/v3p_preictal_trajectory/{narrow,broad}/figures/](topic5_ictal_recruitment/v3p_preictal_trajectory/narrow/figures/) | V3p：发作真正开始前最后两分钟（横轴=相对脑电起始的真实秒数 P0..P3＝−105/−75/−45/−20s，只画到 −10 秒边界，onset 本身及之后完全不碰）连锁活动是否逐渐搬到间期固定高频通路之外——real-time 轴重绘（2026-07-05，替换旧 6 面板栅格图），narrow/broad 各自独立出图，从不共享一根轴。每队列 2 张：**主图** `v3p_axis_vs_offaxis_{narrow,broad}.png`＝沿轴组织度（橙）vs 离轴流（青）migration pair，两线各自按发作前最远基线（−105/−75s）归一为基线标准差单位；**附图** `v3p_mode_direction_{narrow,broad}.png`＝最易放大模态的离轴方向（紫）。图上只留一行斜体角注（Holm p，实时读自 tier JSON），文字说明在各队列 `figures/README.md`。**结果沿用真实 n_perm=1000 大跑（80 核/20 job/7h35m）数字，未变：完整硬门阴性 tier 0**（narrow 0/7 Holm p=1.000/1.000、broad_expanded 0/13 Holm p=0.685/0.685、broad_core 0/9 Holm p=0.652；broad 有 3 人非轴向流 + 2 人模态方向零散单-null 命中，全被 rate/lag1/phase/block/双 span 硬门筛掉，不算潜在阳性） |
| [paper-ready-figure/fig_topic5_scaffold_ab/figures/](paper-ready-figure/fig_topic5_scaffold_ab/figures/) | V3d：间期传播轴的 A/B 两态占据 + 发作按类型分。**cohort 3 张**：`cohort_two_state_vs_geometry`（两态是否几何伪影 —— ρ≈0/>0 时中点仍空 ≤10% → 两态 largely real）、`cohort_ab_typing`（发作分型，cross-seizure-shuffle null **11/20 显著、9/20 可区分两型**）、`cohort_h1_nearonset_forest`（近-onset 收敛 **1/3 合格锁定、binom p=0.14 → 非队列效应**）；**per_subject/ 14 张**：per-seizure C_AB 分布图**全部 12 个被试**（发作分成 A/B 主导，全被试交付物）+ 时程图**只 E442、E1146 两个被试**（轴存在窗口密度够连续读出趋势才出图；其余 10 个被试窗口太稀疏、中位线会碎成孤立短线段读不出趋势，已不再渲染这张图，避免误导）。口径 = 间期两模板投到发作能量：能量落两个离散态（强 A/强 B，几乎不落中点）、发作按 A/B 分型、近-onset 收敛只 442 单被试锁定（E1146 数据量够但未锁定，作为诚实对照一并出图）。详见 figures/README.md + spec `docs/superpowers/specs/2026-07-09-topic5-v3d-*` |

---

## 其他（无 curated README 的图目录）

以下目录有图但没写 `figures/README.md`，含义需看对应 archive doc：
`spatial_modulation/figures/`、`refine_soz_validation/yuquan/figures/`、`topic4_sef_hfo/figures/`、
`topic4_sef_hfo/observation_layer/snn_cm_spontaneous/regime_screen/figures/`、`topic4_sef_itp/swap_mechanics/figures/`。

## 保留不动的大目录（非图，供参考）

`_legacy_2021_readonly/`（1.6G 引用基线，只读）、`hfo_detection/` `hfo_detector_v2/`（检测产物/输入）、
`interictal_synchrony/epilepsiae_ready_full_artifacts/`（2.9G 同步输入）、`_cold_archive/`（phantom-superseded 打包冷藏）。
