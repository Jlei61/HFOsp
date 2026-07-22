# 主图计划

本文主图围绕两个核心论点组织：

1. 间期 HFO 群体事件是癫痫病理网络的指示器。
2. 间期活动可能是病理网络动态的推动者；这部分主要通过模型和病例场景说明可行机制。

## Fig1: 间期 HFO 群体事件与病理网络读出

### Fig1-A: 单 HFO 形态与原始群体事件示例

**目的**：先用人工标注 HFO 集合说明单个 HFO 的波形与时频形态，再用最直观的原始信号说明，间期 HFO 可跨通道共同出现，并且群体内部存在可量化的早晚关系。

**当前验收版本**：

- 输出目录：`results/paper-ready-figure/fig1_hfo_group_event_demo/figures/`
- 正式文件：`fig1-panela1.{png,pdf}` / `fig1-panela2.{png,pdf}`
- 复现入口：`scripts/paper_figures/plot_fig1_single_hfo_schematic.py` + `plot_fig1_hfo_group_event_legacy_style.py`，由 `plot_fig1_interictal_hfo_temporal_scaffold.py` 收口到正式文件名
- a1 数据来源：legacy `zhangkexuan_pickSigs.npz` + `zhangkexuan_annot_v4.pik`，人工标注 HFO n=178
- a2 数据来源：Yuquan Y3, `FC10477Q`
- 固定示例事件：packed event indices `22,237,1458`
- 图形合同：a1 为 178 段 HFO 叠加波形 + raw/normalized mean spectrum；a2 左侧为 80–250 Hz stacked bipolar traces，右侧为 normalized magnitude spectrogram，并用主高频增强连通区的加权质心点/线显示群体事件内部时序。两者统一使用 magnitude + Gaussian σ=1.5；A1/A2 分别保留 180 ms / 50 ms 窗以匹配各自时间尺度。

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

### Fig2-A: 单 subject 间期传播时序素材图

**目的**：用真实数据单 subject 的长时序图展示，间期 HFO group events 不是随机单通道事件，而是反复落在稳定的 channel-level first-to-last rank 结构上；无监督聚类后，主素材优先呈现两类模板 TA/TB。

**当前素材版本**：

- 输出目录：`results/interictal_propagation_masked/figures/per_subject/`
- 示例文件：`epilepsiae_958_propagation.png`
- 复现入口：`scripts/plot_interictal_propagation.py --masked-features --pr3 --paper-style --max-events 2000`
- 视觉合同：`docs/fig2_temporal_propagation_panel_spec.md`

**当前口径**：

Fig2 时序图是 subject-level 真实数据素材，不是 cohort-level 统计，也不是模型图。它支持“事件反复进入稳定传播模板”的可视化叙事；cohort-level 数字仍要由 masked propagation summary / matching-index 统计面板承载。TA/TB 只是同一 subject 内两类模板的图上别名，不跨 subject 合并。

### Fig2-C: 间期单事件包络传播场（paper-ready 候选）

**目的**：在已经冻结的患者特异 shared axis/plane 上，用两次真实 TA/TB 代表事件展示 HFO 包络在几十毫秒内的相反时序演化，并用 Fig1a 同源 spectrogram/质心 readout 直接核对每个 field frame 的时间含义。

**当前候选版本**：

- 输出目录：`results/paper-ready-figure/fig2c_interictal_event_envelope_field/figures/`
- 正式 candidate：`fig2c_candidate_E1146_interictal_event_envelope_field.{png,pdf}`
- 动态 sidecar：`fig2c_candidate_E1146_interictal_event_envelope_field.gif`
- 复现入口：`scripts/paper_figures/plot_fig2c_interictal_event_envelope_field.py --subject epilepsiae_1146`
- 固定视觉与科学合同：`docs/fig2c_interictal_event_envelope_field_spec.md`；以后所有间期传播场 frame/GIF 必须先读该文件并复用 canonical renderer。
- 图形合同：两行 TA/TB；每行按 `readout | readout colorbar | gap | 6 square field frames | field colorbar` 排列。E1146 静态帧为 `−8, +4, +15, +27, +38, +50 ms`；GIF 为同一窗口、2 ms 生物学帧间隔。包络场使用 `magma`、participant-only support、固定 6 mm display kernel 和 TA/TB 共同 `vmax`。

**当前口径**：

Fig2-C 是 raw-EEG-derived envelope timing 在既有冻结间期轴上的 representative cross-check。当前合同只覆盖 TA/TB 各一次 exemplar 的单事件版本，不覆盖多事件 event train；后续多事件 GIF 另立事件边界、逐事件 t0 与抽样规范。exemplar 分组、参与触点和显示几何仍来自模板管线，因此不是 template-free 或独立验证；单被试两次事件不能升级为 cohort-level traveling-wave 或机制证据。E1146 当前仍是 candidate，最终是否进入主图需与 Fig2 其余 panel 的信息增量共同裁决。

### Fig2-E: 患者特异 TA/TB 间期传播场（paper-ready 候选）

**目的**：把 channel-level TA/TB propagation rank 放回患者自己的电极几何中，展示间期群体事件模板所定义的连续空间传播结构。该 panel 只使用冻结的间期轴与 field，不读取任何发作、onset、subtype 或能量数据。

**当前候选版本**：

- 候选目录：`results/paper-ready-figure/fig2e_interictal_template_fields/figures/`
- 当前代表患者：`E1146`；TA/TB 两轴宽泛共线且传播方向相反，因此两幅图使用同一个 shared plane。代表患者尚未最终锁定。
- 全部素材：`results/interictal_propagation_masked/template_gradient_fields/figures/`，包含 28 名可建双轴患者的单患者图和 atlas；26 人具有有效二维几何，单杆患者只保留作方向审阅，不作为最终二维示例候选。
- 冻结输入：`results/interictal_propagation_masked/template_gradient_fields/per_subject/<subject>.json`。
- 复现入口：`scripts/plot_topic5_interictal_template_ab_fields.py`；paper 候选同时导出 PNG/PDF。
- 固定画图规范：`docs/topic5_interictal_field_figure_spec.md`；之后所有间期 TA/TB 场图必须复用其中的公共 payload/panel/subject/atlas 函数。
- 图形合同：传播主轴正方向固定为 early→late；宽泛共线者使用 shared plane，其余使用 TA/TB 各自平面；transverse 正负只按电极几何固定，不根据 field 颜色调向；`sigma_display=6 mm` 仅用于显示，不替换评分时冻结的患者特异 kernel。TA/TB 使用红/蓝语义色、一个共享横轴标签和一个与 field 等高的共享 colorbar。

**当前口径**：

Fig2-E 候选支持“间期 TA/TB 模板在患者特异电极几何中形成可视化的空间传播场”。它是 representative subject visualization，不是 cohort 统计，也不证明与发作早期能量一致；后者由 Fig3 的 field-concordance 分析独立回答。

### Fig2-F: shared-axis 队列的 TA–TB field 反向性（最后一行候选）

**目的**：在 Fig2 前面已经展示定轴、单事件传播和代表患者 TA/TB 场之后，用一个不做关系分层的
cohort 行回答：已有 shared axis 的患者中，TA 与 TB 连续 field 是否呈系统性的反向组织，并且 cohort
中位反向程度是否比全触点随机打乱更极端。

**当前候选版本**：

- 输出目录：`results/paper-ready-figure/fig2_shared_field_reversal/figures/`。
- 正式候选：`fig2_shared_field_reversal_last_row.{png,pdf}`，尺寸固定为 7.15 × 3.05 inch，供 Figure 2
  最后一行横跨整版使用。
- 复现入口：`scripts/paper_figures/plot_fig2_shared_field_reversal_row.py`。
- 主分母：已有 `shared_a/shared_b` 且 `geometry_2d_supported=true` 的12名患者；不按 signed axis cosine、
  same/reversed 标签或 strict-stability 分组。
- 左侧锁定匿名投稿代号 E15、E14、E13、Y9 四个负相关且二维几何易读的案例，复用统一间期场 renderer 成对
  绘制 TA/TB Viridis rank field，只作形态例子。E958 因触点过密、图形瘦长而排除，E1146 因已在
  Figure 2 前文出现而不重复。右上显示全部12名患者的 signed `r`、零线、中位数和 IQR；当前8/12
  为负，中位 `r=-0.353`。
- `r` 直接由冻结 artifact 的 contact-evaluated `shared_a.template_field` 与
  `shared_b.template_field` 计算；左侧 6 mm field map 是统一 display renderer，不能用图像 pixel 重算 `r`。
  四个示例统一裁到以各自触点范围为中心的 `50 × 60 mm` display-only 窗口，裁切不改变轴、触点、rank、
  kernel 或统计；白色触点圈使用细线，避免遮盖连续 field。
- 右下固定显示 primary channel shuffle 的层级 cohort-median-shift null：TB earliness 与 support 在全部
  触点间联合打乱、在冻结 shared plane 上重建 TB field。当前 `Δmedian=-0.339`，lower-tail
  `P_perm=0.01840`。

**当前口径**：

该行支持“shared-axis cohort 的 TA/TB field 中位相关比全触点随机化更负”。左侧显示案例是
outcome-selected morphology，不能冒充独立验证；右侧全12人分布才是 cohort 视图。逐患者 observed 与
各自 channel-null 中位数的配对 Wilcoxon 为 `P=0.08813`，within-shaft cohort sensitivity 为
`P=0.87836`，因此不能泛化成所有空间 null 均显著或多数单患者均显著。

### Fig3-A: 发作原始波形与标准频带谱（正式版，LOCKED 2026-07-18）

**目的**：作为 Fig3 的正式起始 panel，用一个真实 seizure 建立读者可见的 signal context：clinical onset 前后的 raw SEEG、同一代表通道的 baseline-normalized TFR，以及四档 band-power trajectory。它解释后续 field readout 消费的原始信号与 baseline 关系，但本身不是 cohort 统计或机制证据。

**正式版本**：

- canonical 输出目录：`results/paper-ready-figure/fig3a_raw_spectral_context/figures/`；旧 `fig3_sup2_raw_spectral_context/` 只保留历史副本。
- 正式文件：`epilepsiae_1146_seizure_07_raw_spectral_context.png` / `.pdf`，同目录含 summary JSON 和中文 README。
- 复现入口：`scripts/paper_figures/plot_fig3_raw_spectral_context.py --subject epilepsiae_1146 --seizure-idx 7`
- 冻结案例：`E1146` seizure `7`，CAR，lagPat joint-valid 15 contacts，代表通道 `SCL9`；显示 `[-120,+20] s`，baseline `[-120,-90) s`，clinical-onset shading `[0,10) s`。
- 图形合同：左侧 raw SEEG / TFR 上下同宽并严格时间对齐，TFR colorbar 占独立窄列；右侧 2×2 为 low bands (1–30 Hz)、gamma (30–80 Hz)、high-gamma (80–150 Hz)、broadband (1–150 Hz)。同一行共享 y limits，y ticks 只放左图。无内部 a/b、无 EEG onset、无 onset 竖线；标题只写 `E1146` / `TFR on SCL9`。
- 完整视觉合同：`docs/figure_style_guide.md` §5a；验收记录：`docs/archive/topic5/fig3a_raw_spectral_context_acceptance_2026-07-18.md`。

**当前口径**：

Fig3-A 是正式的 representative signal-context panel。它可支持“clinical onset 附近出现宽频能量增强，并可在 raw SEEG/TFR/band-power 三层直接核对”；不能单独支持 timing-order replay、direction replay、onset-emergent alignment、cohort superiority 或机制。

### Fig3-B: 间期 TA 时序场与发作早期能量场（paper-ready 候选）

**目的**：在同一个冻结 shared plane 上，把 E1146 的间期 TA timing field 与一例和 TA 最一致的真实发作 early broadband power 并排展示，建立 Fig3-A signal context 与下游 field-concordance 统计之间的空间读出桥。

**当前候选版本**：

- 输出目录：`results/paper-ready-figure/fig3b_interictal_ictal_shared_field/figures/`。
- 当前实例：E1146 seizure 15；在 25 次 complete / exact `1–150 Hz` 发作中 `shared_a_signed` 最大（`0.869905`）。Fig3-A 仍为 seizure 7，两者不是同一次发作。
- 左图：冻结 TA timing field，`viridis` early→late；`TA fields` 使用红色固定语义色。
- 右图：精确 `1–150 Hz`、clinical `[0,10] s`、远端 EEG baseline-normalized log-band power；连续 min–max 插值、无 rank、无 sign flip，使用 `magma_r`，高 power 为深色。
- 左右严格共用 TA support、shared plane、extent 和同一个 6 mm display sigma；两个 panel 分别写 xlabel，右图保留完整边框。colorbar 分别显示真实 propagation rank 与 robust-z，不再显示无量纲 `0/1`。
- 复现入口：`scripts/paper_figures/plot_fig3b_interictal_ictal_shared_field.py`；默认自动按 `shared_a_signed` 选 seizure 15。
- 完整合同：`docs/fig3b_interictal_ictal_shared_field_spec.md`。

**当前口径**：

Fig3-B 是 intentionally selected best-TA representative。它说明一例发作早期能量分布可在冻结间期 TA 坐标中被直观看到；由于 seizure 按 TA 一致性选择，不能写成独立 replay 证据、cohort 结论或机制证明。

### Fig3 field-concordance cohort statistic（panel 编号待总拼版）

**目的**：用一个紧凑 Data-vs-Null 统计面板说明，间期 HFO 传播场和发作早期激活场在 maxAB 可评估 subject 层面整体高于 channel-shuffle null；不再展示 per-subject board。

**当前验收版本**：

- 输出目录：`results/paper-ready-figure/fig3_field_concordance_cohort_stat/figures/`
- 正式文件：`field_concordance_cohort_stat.png` / `field_concordance_cohort_stat.pdf`
- 复现入口：`scripts/paper_figures/plot_fig3_field_concordance_cohort_stat.py`
- 数据来源：`results/topic5_ictal_recruitment/axis_alignment/axis_alignment_{broadband,broadband150,hfa}_max_ab_B1000.json`
- 图形合同：按参考图风格画三组 `Data` vs `Null` 的 violin + box + subject 点，并用浅灰线连接同一 subject 的 paired Data/Null 值；三组为 `BB 1-45 maxAB`、`BB 1-150 maxAB`、`HFA 60-100 maxAB`。不写 `All candidates`，不画背景网格线，不混入 broad fallback。当前 maxAB 可评估 subject 为 20 个；`BB 1-150` 为 line-noise-masked sensitivity，原 `bb_auc` 仍是 legacy 1-45 Hz。

**当前口径**：

这个 panel 支持“共享粗网络轴 / field concordance”，不表示发作沿间期方向逐点重放，也不替代 Topic 5 A-line primary FDR 定稿表。这里的 Null 是所选候选的 channel-shuffle median，用于展示 cohort-level shift above null；formal pass 仍以 selection-corrected p95/p-value 表为准。

**Field-concordance supplement（发作内 field 动力学，exploratory，2026-06-28）**：把早期单窗 field concordance 扩到**整段发作**——同一 field 渲染（间期 **A|B 锚** + 发作内各时刻激活场，锚到发作前布局，红/蓝=source 端点集合），外加每 subject 一个 **field 演化 GIF**（onset→offset，直观看发作场的传播变化）。配套统计（走廊轴向 vs 非轴向占比随进程）：**broad 队列有暗示（轴向降 5/8、非轴向升 8/8 by sign）但 narrow 扩队列不复现甚至反向（3/7、2/7）→ 方向减弱假设不稳健、依队列/substrate**。**定位 = supplementary**：主推**可视化**，方向统计**不进 claim**；z-ER 中后期偏示意。复现：`scripts/{run,plot,analyze}_topic5_*field_dynamics*.py --substrate {broad,narrow}`；归档 `docs/archive/topic5/ictal_field_dynamics_pilot_2026-06-28.md`。

**Fig3-Sup1（间期 HFO 几何 ↔ 发作早期多频带能量场 alignment，V2 Phase-1，exploratory，2026-07-04，已验收）**：把单频段 field concordance 扩到**全 12 频带扫描**（δ→ripple）+ 诚实 null / per-subject caveat。3 panel：**A** subject×band maxAB 热图（红蓝 diverging、narrow>broad、band-generic）；**B** 每 primary 带 per-subject Δ vs 弱空间 null violin+点（两池 **6/7** 过 family-wise、唯 ripple_high n.s.=**NOT ripple-specific**）；**C** per-subject 稳定性（cohort 6/7 是**聚合**、narrow 中位仅 **2/7**、≥5/7 仅 **3/20** = **承重 caveat**）。**tier = exploratory candidate early-ictal spatial recruitment scaffold（cohort 层，非 formal/机制）**；**formal within-shaft Gate A 未评估**（2/20 within_shaft_strong、弱 null likely inflated）、**Gate B/C 未跑**、仅 onset+0–20s。**禁** HFO-/LVFA-/ripple-specific / timing-order / formal Gate A passed。复现 `scripts/paper_figures/plot_fig3_sup1_multiband_field_alignment.py`；归档 `docs/archive/topic5/v2_phase1_band_scan_backbone_2026-07-02.md`。

### Fig3-C: peri-onset field similarity trajectory（E1146 示范 + 全 subject material pool）

**目的**：作为 field-concordance cohort statistic 的 per-subject dynamic material，展示二维 shared-gradient plane 上 `[-120,+20]s`、1-150 Hz signed robust-z 能量场的 raw similarity trajectory。它是描述性素材，不预设相似度高于 shaft geometry，也不解释为 onset 时新出现的 alignment。

**当前验收版本**：

- 输出目录：`results/paper-ready-figure/fig3_peri_onset_field_similarity/figures/`
- 正式文件：`epilepsiae_1146_peri_onset_field_similarity_paper_ready.png` / `.pdf`
- 复现入口：`scripts/paper_figures/plot_fig3_peri_onset_field_similarity.py --subject epilepsiae_1146`
- 数据来源：`results/topic5_ictal_recruitment/field_dynamics_signed/epilepsiae_1146_signed_broadband_1_150Hz_similarity_timecourse_m120_p20_10s_step2s_per_seizure.csv`
- 上游生成：`scripts/plot_topic5_signed_broadband_similarity_timecourse.py --subject epilepsiae_1146 --start-sec -120 --stop-sec 20 --band-lo 1 --band-hi 150 --window-sec 10 --step-sec 2`
- 输入合同：只消费 fingerprint-valid frozen `shared_a/shared_b`，且必须 `geometry_2d_supported=true`、两轴均至少两根 shaft 和二维有效秩；不回退 own A/B。图形合同：双面板，A=`max(|r_A|,|r_B|)` raw shared similarity，B=signed A/B polarity sidecar；10 s sliding window、2 s step、每 seizure 固定 66 窗。

**当前口径**：

Panel A 只展示 raw shared-plane similarity trajectory；Panel B 只在 seizure coverage 足够时作为 polarity sidecar。它不是 cohort 统计、不是 onset-emergent alignment、timing-order replay 或机制证据；超过 `+20s` 的比较仍需 duration warping 或阶段对齐。

**二维 shared-only 扩展（2026-07-18）**：正式 denominator flow 为 40 frozen records → 14 shared-pair 且 fingerprint-valid → 12 二维 shared candidates → 10 有 seizure inventory → 7 有 eligible derived cache → **7/12 出图**。生成病例为 E1084、E1146、E384、E548、E583、E590、E958；coverage 分层为 `complete_ok=3`、`partial_ok=3`、`severely_partial=1`。E384 仅 6/12，E583 仅 3/22；E583 不承担 polarity 稳定叙述。5 个 Yuquan 二维 candidates 中，3 人有 inventory 但缺 derived eligibility cache，2 人缺 inventory，均记为 `blocked_input`。E139 与 `yuquan_zhangjiaqi` 为单杆 `geometry_2d_supported=false`，不进入二维分母；E139 仅保留在 `sensitivity_1d/`。当前 canonical run=`20260718T071020Z_d99c96ec`：producer/renderer 只写 `runs/<run_id>/artifacts/`，完整默认 batch 验收后才替换顶层 index，并最后原子替换 manifest 作为 completion pointer；explicit subset 和中断 run 不改 canonical artifact。

**shared-matched maxAB 空间置换 null（2026-07-18，fixed-time-mapping v2）**：新版 null 对上述 7 人使用相同 frozen `shared_a/shared_b`、fingerprint、二维 geometry 和成功 seizure 集；每个 `seizure×replicate` 固定一次空间映射贯穿全部 66 窗，每次 shuffle 都重算 A/B、mirror choice 与 maxAB。R=1000，vectorized-vs-exact 及 observed-vs-source 最大误差均 ≤`5.6e-14`。within-shaft 结果为 **3/7** 至少一个 cluster（E1084、E1146、E590），**2/7** 有 maxT 窗（E1084、E1146）；这是 per-subject 时间分辨描述，不是 formal cohort gate。旧逐窗置换的 `5/7` 已撤回；旧 own-plane null 的 E922/E1146、`13/20`、`7/20`、`2/20` 等数字也不得接到新版图。canonical 输出为 `spatial_null/`，manifest 含 7 个 summary，共 35 个 null artifacts。

**科学边界**：冻结 archive 的 early-ictal shared-field cohort null（二维共线 n=7，within-shaft p=0.346；shared-vs-own p=0.938）仍是 cohort 级主参照。这里的 sliding-window null 回答逐人哪些时间段偏离杆内置换，不能把 3/7 cluster 写成 cohort superiority，也不能因为 pre-onset cluster 而写 onset-emergent alignment。within-shaft power 仍依赖 shaft sizes；E583 的 3/22 coverage 尤其需要降级解释。

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
- 图形合同：**四块** `clustered event heatmap | per-channel rank distribution | cluster rank distributions | model-vs-real 2×2 相似性矩阵`。前三块**复用仓库成熟 canonical 画图函数**（`scripts/plot_interictal_propagation.py` 的 `_plot_rank_histogram` / `_plot_rank_heatmap` / `_plot_cluster_boundaries` / `_plot_cluster_rank_fig4`，与 Topic-1a per_subject 图同一套），不手搓；heatmap rank colorbar 竖放在 heatmap 右侧；KMeans 显示标签不用 C0/C1，而用模板名（`t_a` 红、`t_b` 蓝）；第三块 legend 放在 panel 内右上角。第四块 = 模型 fwd/rev × 真实 t_a/t_b Spearman 矩阵，只用 star 显示方向性置换 p（不写数值）、aspect=equal，matrix colorbar 与矩阵等高。
- **LOCKED 模式**：每个 subject-SNN 案例固定出 Fig4A（readout 四列）+ Fig4B（KMeans 四块）两张主图；Fig4C（real-vs-model profile）/ Fig4D（组合 S 置换 null）为可选 supplement。
- 当前结果：同一个 seed3 spontaneous twoend readout 的 14 个 clean directional events 被 `KMeans k=2` 分成 `t_a/t_b` = 6/8；方向 purity=1.00；`within_cluster_tau_mean=0.939`；更干净的 shared-overlap corr = -0.69。

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

### M3A-v2 Step4 诊断图：低-q / gK closed-loop 负结果目视审阅

**目的**：把 M3A-v2 Step4 的代表性动力学现象画成同一套 SNN 四列图，供目视检查：baseline 轴向事件、shallow low-q 无明显扩张、shallow low-q + gK suppress、deep low-q runaway、deep low-q + gK 仍不 rescue。

**当前输出**：

- 输出目录：`results/paper-ready-figure/fig_m3a_v2_step4_dynamics/figures/`
- 文件：`baseline_axial.png/pdf`、`lowq_shallow_qonly.png/pdf`、`lowq_shallow_braked.png/pdf`、`lowq_deep_qonly.png/pdf`、`lowq_deep_braked.png/pdf`
- 复现入口：`scripts/paper_figures/plot_fig_m3a_v2_step4_dynamics.py`

**当前口径**：

这是 visual diagnostic，不是新的统计 sweep，也不是主图 claim。Step4 统计源仍是 `results/topic4_m3a_v2_step4_lowq/step4_lowq_{small,finer}.json`。由于原 Step4 是单核 kick 协议，诊断图把 source 镜像到 scaffold 两端以保留 `mechanism | tempA source | tempB source | electrode readout` 标准；readout shading 表示 source identity，不表示发作方向。

### M3A-v2.1 qI/gK gap 诊断图：代表性负结果目视审阅

**目的**：把 q_I/g_K gap sweep 的代表性动力学现象画成同一套 SNN 四列图，供目视检查：baseline 轴向事件、returned 但仍轴向的 closest clean case、小范围 metric edge、动态 g_K suppress、off-axis/global runaway。

**当前输出**：

- 输出目录：`results/paper-ready-figure/fig_m3a_v2_1_qigk_gap_dynamics/figures/`
- 文件：`baseline_axial.png/pdf`、`returned_axis_only_clean.png/pdf`、`metric_edge_small_suppress.png/pdf`、`dynamic_gk_suppress.png/pdf`、`dynamic_gk_runaway.png/pdf`
- 复现入口：`scripts/paper_figures/plot_fig_m3a_v2_1_qigk_gap_dynamics.py`

**当前口径**：

这是 visual diagnostic，不是新的统计 sweep，也不是主图 claim。统计源仍是 q_I/g_K gap sweep 的 `per_run.jsonl` 和报告。由于原 sweep 是单核 probe，诊断图把 source 镜像到 scaffold 两端以保留 `mechanism | tempA source | tempB source | electrode readout` 标准；readout shading 表示 source identity，不表示传播方向或发作方向。

### M3A-v2.1 qI build-up GIF：连续轨迹到 runaway

**目的**：用一条连续仿真轨迹目视检查“局部轴向事件反复发生 → q_I/permissivity 累积 → delayed runaway”的动力学。该图不包含 h_G/recovery，只展示 build-up-to-runaway leg。

**当前输出**：

- 输出目录：`results/paper-ready-figure/fig_m3a_v2_1_qigk_runaway_transition/figures/`
- E1146 几何变体：`results/paper-ready-figure/fig_m3a_v2_1_qigk_runaway_transition_epilepsiae_1146/figures/`
- 文件：`qigk_runaway_transition.gif`、`qigk_runaway_transition_final.png`、`qigk_runaway_transition_metadata.json`
- 复现入口：`scripts/paper_figures/plot_fig_m3a_v2_1_qigk_runaway_transition_gif.py --k-q 0.10 --q-min 0.05 --kick-boost 5.0 --r-kick 0.6 --T 1500`
- E1146 几何复现入口：`scripts/paper_figures/plot_fig_m3a_v2_1_qigk_runaway_transition_gif.py --layout subject1146 --fig-name fig_m3a_v2_1_qigk_runaway_transition_epilepsiae_1146 --k-q 0.10 --q-min 0.05 --kick-boost 5.0 --r-kick 0.6 --T 1500`

**当前口径**：

这是 visual diagnostic，不是新的统计 sweep，也不是主图 claim。右侧 SEEG readout 是同一条连续 trace，不再拼接 tempA/tempB 片段；电极摆放沿用 Fig5/Stage5 的 A0-A5/B0-B5 contact placement，并按当前 M3A sheet 等比例缩放。metadata 记录 transition gate、runaway onset、pre-runaway local event 计数和 shading 规则；runaway onset 之后不再用 KMeans 或事件检测继续画 shading。

当前动态图恢复为一行三列布局：`permissivity | 2D SNN activity | continuous SEEG readout`。右侧 readout 显示完整 0-T trace，runaway 前只画 response shading；机制 overlay 用沿轴椭圆，峰值连线按 contact y 轴从上到下连接，不按 temporal rank order 连线。它只用于目视核对 build-up-to-runaway timing，不把这个单轨迹写成方向性传播或 recovery 结论。

E1146 几何变体只替换 contact/foci placement：读取 `fig_subject_snn_epilepsiae_1146` 对应 figdata，把 subject 的电极排布和两个灶位置等比例缩放到当前 M3A L=10 sheet；动力学参数和判读边界保持同一套 visual diagnostic 口径。

## 当前执行原则

- 主图脚本统一放在 `scripts/paper_figures/`。
- 主图输出统一放在 `results/paper-ready-figure/`，每个 figure/panel group 单独建目录。
- 每个 figure 输出目录必须有 `figures/README.md`，说明展示目的、正式文件、关注点。
- 主图计划文档只记录正式口径和待补齐内容；详细审阅和数值表继续放 `docs/archive/`。
