# 主图计划

本文主图围绕两个核心论点组织：

1. 间期 HFO 群体事件是癫痫病理网络的指示器。
2. 间期活动可能是病理网络动态的推动者；这部分主要通过模型和病例场景说明可行机制。

## Fig1: 间期 HFO 群体事件的患者特异性时序组织

**唯一输出目录**：`results/paper-ready-figure/fig1/figures/`。Figure 1A 是作者手绘示意图，不由代码生成，也不放入 `paper-ready-figure`。

| panel | 当前内容 | 文件 |
|---|---|---|
| B1 | 人工标注 HFO n=178 的波形与 raw/normalized spectrum | `fig1-panelb1.{png,pdf}` |
| B2 | Yuquan Y3 群体事件原始波形、normalized spectrogram 与质心轨迹 | `fig1-panelb2.{png,pdf}` |
| C | E7 时间顺序 masked rank heatmap、day/night strip 与 rank distribution | `fig1-panelc.{png,pdf}` |
| D | 40 人 masked shared-participant MI data vs permutation null | `fig1-paneld.{png,pdf}` |
| E | 同一 E7 的全量 TA/TB 聚类重排与 mean-rank profiles | `fig1-panele.{png,pdf}` |
| F | overall vs within-template MI uplift；右下配对小 panel 比较 single-template 与 multi-cluster MI | `fig1-panelf.{png,pdf}` |

复现入口为 `scripts/paper_figures/build_main_figures_1_2.py --figure 1`；底层仍调用已验收的 HFO、heatmap、rank distribution 与统计 painter。独立文件均不带 panel 字母；`fig1-complete-layout.{png,pdf}` 是带 B–F 字母的完整排版。C 与 E 必须来自同一患者、同一 6,556 个有效事件；E 中 TA/TB 计数之和必须守恒。SOZ ROC 不属于当前 Figure 1，保留在 supplementary SOZ 包。

**当前口径**：Figure 1 支持群体 HFO 事件存在患者特异、可重复出现的时序模板，并显示分模板后的 rank-concordance 提升；它不单独证明共享三维空间轴、发作传播或机制因果。

## Fig2-Fig6 暂定分工

### Fig2-Fig3: 间期事件作为病理网络指示器

优先承载真实数据主结果：传播模板、网络轴、SOZ/临床相关性、跨事件稳定性。这里应该是第一核心论点的主要证据区。

### Fig2-A/B: 空间投影方法与方向代表性

- **唯一输出目录**：`results/paper-ready-figure/fig2/figures/`。
- **Fig2-A**：`fig2-panela.{png,pdf,svg}`，近方形 2×2 展示作者提供的 Y9 植入概览、E1146 真实 skull-stripped T1 局部 cutaway、电极到冻结平面的三维几何，以及含 `σ=6 mm` Gaussian 显示范围的二维触点覆盖。E1146 三块均标明 `ICL` / `SCL` 电极杆。上排不写标题；下排只保留居中的 `Electrodes projection` 与 `2D local field`。不画流程箭头或 legend；右下 viewport 固定为单元格的 72%，左下完整显示投影平面。Y9 overview 的既有红/蓝方向 glyph 只属于 overview，E1146 三块不重复 TA/TB rank field、方向、rank 色彩或 early/late。
- **Fig2-B**：`fig2-panelb.{png,pdf}`，左侧以 E1146/E548 的同一组留出事件方向对比仅时序模板轴和时序--空间模板轴，右侧展示 25 名可评估患者的留出方向增益及记录块内方向置换零模型。配对小提琴比较移至 FigS4-B。
- **口径**：A 只解释代表性植入概览到 subject-T1 空间注册、冻结坐标投影和 display-support 的方法 pipeline；Y9 与 E1146 不是同一病例，不能写成连续 patient-specific zoom。Gaussian 不是组织活动或 scoring kernel，且 E1146 历史 MNI warp 类型不可重建。B 支持真实三维电极信息提高患者内跨记录块的方向一致性；该留出设计不是未见患者预测，也不证明连续组织传播轨迹、速度或机制因果。
- **排版合同**：A–F 独立文件均不带左上角字母；`fig2-complete-layout.{png,pdf}` 是带 A–F 字母的完整排版。

### Fig2-C: 间期单事件包络传播场（paper-ready 候选）

**目的**：在已经冻结的患者特异 shared axis/plane 上，用两次真实 TA/TB 代表事件展示 HFO 包络在几十毫秒内的相反时序演化，并用 Fig1a 同源 spectrogram/质心 readout 直接核对每个 field frame 的时间含义。

**当前候选版本**：

- 输出目录：`results/paper-ready-figure/fig2/figures/`
- 正式 candidate：`fig2-panelc.{png,pdf}`
- 统一复现入口：`scripts/paper_figures/build_main_figures_1_2.py --figure 2`；默认从已验收 v7 矢量 PDF 导出 600 dpi PNG，显式 `--recompute-fig2c` 才重新读取原始 EEG。
- 固定视觉与科学合同：`docs/fig2c_interictal_event_envelope_field_spec.md`；以后所有间期传播场 frame/GIF 必须先读该文件并复用 canonical renderer。
- 图形合同：两行 TA/TB；每行按 `readout | 4 envelope frames | frozen template field` 排列。E1146 v7 静态帧为严格等间距的 `0, +14, +28, +42 ms`，由 contact-level selector 自动选择；包络场使用低饱和蓝灰、participant-only support、固定 6 mm display kernel，并按每次事件完整窗 q99 分别归一化。

**当前口径**：

Fig2-C 是 raw-EEG-derived envelope timing 在既有冻结间期轴上的 representative cross-check。当前合同只覆盖 TA/TB 各一次 exemplar 的单事件版本，不覆盖多事件 event train；后续多事件 GIF 另立事件边界、逐事件 t0 与抽样规范。exemplar 分组、参与触点和显示几何仍来自模板管线，因此不是 template-free 或独立验证；单被试两次事件不能升级为 cohort-level traveling-wave 或机制证据。E1146 当前仍是 candidate，最终是否进入主图需与 Fig2 其余 panel 的信息增量共同裁决。

### Fig2-D: E1146 冻结 TA/TB shared-plane template fields

**目的**：把 channel-level TA/TB propagation rank 放回患者自己的电极几何中，展示间期群体事件模板所定义的连续空间传播结构。该 panel 只使用冻结的间期轴与 field，不读取任何发作、onset、subtype 或能量数据。

**当前候选版本**：

- 正式文件：`results/paper-ready-figure/fig2/figures/fig2-paneld.{png,pdf}`
- 当前代表患者：`E1146`；TA/TB 两轴宽泛共线且传播方向相反，因此两幅图使用同一个 shared plane。代表患者尚未最终锁定。
- 全部素材：`results/interictal_propagation_masked/template_gradient_fields/figures/`，包含 28 名可建双轴患者的单患者图和 atlas；26 人具有有效二维几何，单杆患者只保留作方向审阅，不作为最终二维示例候选。
- 冻结输入：`results/interictal_propagation_masked/template_gradient_fields/per_subject/<subject>.json`。
- 复现入口：`scripts/plot_topic5_interictal_template_ab_fields.py`；paper 候选同时导出 PNG/PDF。
- 固定画图规范：`docs/topic5_interictal_field_figure_spec.md`；之后所有间期 TA/TB 场图必须复用其中的公共 payload/panel/subject/atlas 函数。
- 图形合同：传播主轴正方向固定为 early→late；宽泛共线者使用 shared plane，其余使用 TA/TB 各自平面；transverse 正负只按电极几何固定，不根据 field 颜色调向；`sigma_display=6 mm` 仅用于显示，不替换评分时冻结的患者特异 kernel。TA/TB 使用红/蓝语义色、一个共享横轴标签和一个与 field 等高的共享 colorbar。

**当前状态**：

该 panel 只展示代表患者的冻结静态模板场。Figure 2E 扩展为下述4例 shared-axis TA/TB 场；发作早期能量关系仍由 Fig3 的 field-concordance 分析独立回答。

### Fig2-E/F: shared-axis TA–TB field 形态与队列反向性（独立 panel 候选）

**目的**：在 Fig2 前面已经展示定轴、单事件传播和代表患者 TA/TB 场之后，用一个不做关系分层的
cohort 行回答：已有 shared axis 的患者中，TA 与 TB 连续 field 是否呈系统性的反向组织，并且 cohort
中位反向程度是否比全触点随机打乱更极端。

**当前候选版本**：

- 输出目录：`results/paper-ready-figure/fig2/figures/`。
- 正式候选：`fig2-panele.{png,pdf}` 与 `fig2-panelf.{png,pdf}`；不再输出 E+F 联合文件。
- 复现入口：`scripts/paper_figures/plot_fig2_shared_field_reversal_row.py`。
- 主分母：已有 `shared_a/shared_b` 且 `geometry_2d_supported=true` 的12名患者；不按 signed axis cosine、
  same/reversed 标签或 strict-stability 分组。
- **Figure 2E（左侧）**：锁定匿名投稿代号 E15、E14、E13、Y9 四个负相关且二维几何易读的案例，复用统一间期场 renderer 成对
  绘制 TA/TB Viridis rank field，只作形态例子。E958 因触点过密、图形瘦长而排除，E1146 因已在
  Figure 2 前文出现而不重复。
- **Figure 2F（右侧）**：右上显示全部12名患者的 signed `r`、零线、中位数和 IQR；当前8/12为负，中位 `r=-0.353`；右下显示对应的全触点空间 null。
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

### Fig3-A/B: 断轴发作信号与代表性频谱表型对照（UPDATED 2026-09-03）

**目的**：A 并列展示一次 broadband-type 与一次典型 gamma-type 发作的 raw SEEG + TFR；B 用未经截断的连续时间轴比较二者的四档 band-power trajectory。颜色表示发作表型，不表示频带。

**正式版本**：

- canonical 输出目录：`results/paper-ready-figure/fig3/figures/`。
- 正式文件：`fig3-panela.{png,pdf}`、`fig3-panelb.{png,pdf}`；完整拼版为 `fig3-complete-layout.{png,pdf}`。
- 统一复现入口：`scripts/paper_figures/build_main_figure_3.py`；A/B producer 为 `plot_fig3_raw_spectral_context.py --independent-only`。
- 冻结案例：`E10 | SZ8`（`broadband_1_150`，`SCL9`）与主图 Fig3 已接受的 `E20 | SZ8`（source zero-based seizure index 7；`gamma_nonbroadband`，`HRB1`），均为 CAR。A 的两个内部示例只显示 20 s baseline `[-110,-90] s` 与 `[-10,+20] s`，省略 `[-90,-10] s`；baseline 归一化仍使用 `[-120,-90) s`，clinical-onset shading 仍为 `[0,10) s`。
- 图形合同：A 内横向排列两次发作，各自 raw SEEG / TFR 上下同宽并共享同一断轴映射，两个 TFR 共用 colorbar；B 为四频带 2×2，每格叠加两次发作并连续显示 `[-120,+20] s`，不得断轴。浅蓝紫 `#8D9FCD` 表示 broadband-type，青绿 `#62BE9F` 表示 gamma-type；A 的类型标签与 B 的曲线/legend 共用这套语义色。B 的 legend 放在 low-bands 图左上角的无曲线区、两项纵向排列，只写 `Broadband` / `Gamma`；患者、SZ 与通道身份由 A 标题承担。四图在 0 s 统一画黑色竖直虚线；同一行共享 y limits，y ticks 只放左图，ylabel 简写为 `dB`。独立文件不写 A/B；完整拼版才加字母。
- 完整视觉合同：`docs/figure_style_guide.md` §5a；验收记录：`docs/archive/topic5/fig3ab_representative_phenotype_context_acceptance_2026-09-03.md`。

**当前口径**：

Fig3-A/B 是 representative signal-context。它们可支持“E10/SZ8/SCL9 呈 broadband 增强，而 E20/SZ8/HRB1 呈典型 gamma-dominant 快活动增强”。两例患者和代表通道均不同，因此只能说明两类可见形态，不能把差异归因于发作模式本身；也不能单独支持表型 prevalence、cohort superiority、timing-order replay、direction replay、onset-emergent alignment 或机制。

### Fig3-C: 间期 TA 时序场与发作早期能量场（paper-ready locked）

**目的**：在同一个冻结 shared plane 上，把 E1146 的间期 TA timing field 与一例和 TA 最一致的真实发作 early broadband power 并排展示，建立 Fig3-A/B signal context 与下游 field-concordance 统计之间的空间读出桥。

**当前正式版本**：

- canonical 输出：`results/paper-ready-figure/fig3/figures/fig3-panelc.{png,pdf}`；原目录保留为 producer/source。
- 当前实例锁定 E1146 seizure 2。仅按整体 TA correlation 选出的 seizure 9 会出现中段先亮；加入全触点正 robust-z、TA winner、左右最早端点覆盖和 early-to-late 直接梯度后得到 seizure 2 / 10 / 23 / 1 四个候选，最终目视选择 seizure 2。Fig3-A 仍为 seizure 7。
- 左图：冻结 TA timing field，`viridis` early→late；`TA fields` 使用红色固定语义色。
- C 不写整体标题。右图标题分两行：上行为匿名病例 `E10 | SZ3`，下行为 `Early ictal field`；精确 `1–150 Hz`、clinical `[0,10] s`、远端 EEG baseline-normalized log-band power；连续 min–max 插值、无 rank、无 sign flip，使用 `Blues`，高 power 为深色。右 colorbar 的可见标题简写为 `power` / `z`。
- 左右严格共用 TA support、shared plane、extent 和同一个 6 mm display sigma；两个 panel 分别写 xlabel，空间 y label 统一为 `Y (mm)`，右图不重复 shared y ticks。左 colorbar 与 Fig2 共用 `viridis` normalized-rank 语法，标题 `ranks`、ticks 为 `0 early / 0.5 / 1 late`；原始 propagation rank 范围只留在 metadata。右 colorbar 继续显示 robust-z。
- seizure 2 的 15/15 early-ictal robust-z 均为正（`+1.03–+3.68`）；TA 一致性仍是触点间空间模式相关，不等于 cohort 级能量结论。
- 正式复现入口：`scripts/paper_figures/plot_fig3b_interictal_ictal_shared_field.py`，默认输出 seizure 2；候选 provenance 由 `plot_fig3b_positive_ta_candidates.py` 保留。
- 完整合同：`docs/fig3b_interictal_ictal_shared_field_spec.md`。

**当前口径**：

Fig3-C 是 intentionally selected morphology-aware TA representative。它说明一例正向发作早期能量分布可在冻结间期 TA 坐标中被直观看到；由于 seizure 经过形态 gate 与目视选择，不能写成独立 replay 证据、cohort 结论或机制证明。

### Fig3-D: clinical-onset gradient-field cohort statistic（LOCKED）

**目的**：用一个紧凑 Data-vs-Null 面板比较 clinical onset 后 0–10 s 的 gradient-field concordance 与 channel-shuffle null。

**当前验收版本**：

- canonical 输出：`results/paper-ready-figure/fig3/figures/fig3-paneld.{png,pdf}`。
- source：`results/paper-ready-figure/fig3-sup-tspectral-field-concordance/figures/clinical_onset_gradient_field_cohort_stat.pdf`。
- 复现入口：`scripts/paper_figures/plot_fig3_clinical_onset_gradient_field_cohort.py`，总拼版由 `build_main_figure_3.py` 复制矢量源并以 600 dpi 输出 PNG。
- 图形合同：Pooled `n=17`、Broadband `n=16`、Gamma `n=11`；Pooled/Broadband 标 `*`，Gamma 标 `n.s.`。旧 endpoint `n=20` 三组全显著版本不得替换本 panel。

**当前口径**：

这个 panel 支持 clinical-onset gradient-field concordance 高于 channel-shuffle null 的 pooled/broadband cohort 读出；Gamma 不显著。它不表示发作沿间期方向逐点重放，也不支持频段特异或机制结论。

**Field-concordance supplement（发作内 field 动力学，exploratory，2026-06-28）**：把早期单窗 field concordance 扩到**整段发作**——同一 field 渲染（间期 **A|B 锚** + 发作内各时刻激活场，锚到发作前布局，红/蓝=source 端点集合），外加每 subject 一个 **field 演化 GIF**（onset→offset，直观看发作场的传播变化）。配套统计（走廊轴向 vs 非轴向占比随进程）：**broad 队列有暗示（轴向降 5/8、非轴向升 8/8 by sign）但 narrow 扩队列不复现甚至反向（3/7、2/7）→ 方向减弱假设不稳健、依队列/substrate**。**定位 = supplementary**：主推**可视化**，方向统计**不进 claim**；z-ER 中后期偏示意。复现：`scripts/{run,plot,analyze}_topic5_*field_dynamics*.py --substrate {broad,narrow}`；归档 `docs/archive/topic5/ictal_field_dynamics_pilot_2026-06-28.md`。

**Fig3-Sup1（间期 HFO 几何 ↔ 发作早期多频带能量场 alignment，V2 Phase-1，exploratory，2026-07-04，已验收）**：把单频段 field concordance 扩到**全 12 频带扫描**（δ→ripple）+ 诚实 null / per-subject caveat。3 panel：**A** subject×band maxAB 热图（红蓝 diverging、narrow>broad、band-generic）；**B** 每 primary 带 per-subject Δ vs 弱空间 null violin+点（两池 **6/7** 过 family-wise、唯 ripple_high n.s.=**NOT ripple-specific**）；**C** per-subject 稳定性（cohort 6/7 是**聚合**、narrow 中位仅 **2/7**、≥5/7 仅 **3/20** = **承重 caveat**）。**tier = exploratory candidate early-ictal spatial recruitment scaffold（cohort 层，非 formal/机制）**；**formal within-shaft Gate A 未评估**（2/20 within_shaft_strong、弱 null likely inflated）、**Gate B/C 未跑**、仅 onset+0–20s。**禁** HFO-/LVFA-/ripple-specific / timing-order / formal Gate A passed。复现 `scripts/paper_figures/plot_fig3_sup1_multiband_field_alignment.py`；归档 `docs/archive/topic5/v2_phase1_band_scan_backbone_2026-07-02.md`。

### Fig3-E: peri-onset field similarity trajectory（visual template locked；E1146 示范）

**目的**：作为 field-concordance cohort statistic 的 per-subject dynamic material，展示二维 shared-gradient plane 上 `[-120,+20]s`、1-150 Hz signed robust-z 能量场的 raw similarity trajectory。它是描述性素材，不预设相似度高于 shaft geometry，也不解释为 onset 时新出现的 alignment。

**当前验收版本**：

- paper-facing 输出目录：`results/paper-ready-figure/fig3_peri_onset_field_similarity/design_variants/figures/`
- Fig3-E visual template：`epilepsiae_1146_peri_onset_field_similarity_paper_ready_journal_clean.png` / `.pdf`
- 复现入口：`scripts/paper_figures/plot_fig3_peri_onset_field_similarity.py --subject epilepsiae_1146 --source-csv <下述 immutable source> --design-variant journal_clean --out-dir results/paper-ready-figure/fig3_peri_onset_field_similarity/design_variants/figures/`
- 数据来源：`results/paper-ready-figure/fig3_peri_onset_field_similarity/runs/20260718T071020Z_d99c96ec/artifacts/field_dynamics_signed/epilepsiae_1146_signed_broadband_1_150Hz_similarity_timecourse_m120_p20_10s_step2s_per_seizure.csv`
- 上游生成：`scripts/plot_topic5_signed_broadband_similarity_timecourse.py --subject epilepsiae_1146 --start-sec -120 --stop-sec 20 --band-lo 1 --band-hi 150 --window-sec 10 --step-sec 2`
- 输入合同：只消费 fingerprint-valid frozen `shared_a/shared_b`，且必须 `geometry_2d_supported=true`、两轴均至少两根 shaft 和二维有效秩；不回退 own A/B。图形合同：双面板，左=`max(|r_A|,|r_B|)` raw shared similarity，右=signed TA/TB polarity sidecar；10 s sliding window、2 s step、每 seizure 固定 66 窗。
- 编号合同：投稿图中登记为 **Fig3-E**；既有 R3 计算 artifact 的 `fig3c_*` contract/name 只保留历史 provenance，不再代表 paper-facing panel 编号。该登记锁定画图类型，不改变 R3 quantitative package 的主次关系。

**当前口径**：

左图只展示 raw shared-plane similarity trajectory；右图只在 seizure coverage 足够时作为 TA/TB polarity sidecar。它不是 cohort 统计、不是 onset-emergent alignment、timing-order replay 或机制证据；超过 `+20s` 的比较仍需 duration warping 或阶段对齐。

**二维 shared-only 扩展（2026-07-18）**：正式 denominator flow 为 40 frozen records → 14 shared-pair 且 fingerprint-valid → 12 二维 shared candidates → 10 有 seizure inventory → 7 有 eligible derived cache → **7/12 出图**。生成病例为 E1084、E1146、E384、E548、E583、E590、E958；coverage 分层为 `complete_ok=3`、`partial_ok=3`、`severely_partial=1`。E384 仅 6/12，E583 仅 3/22；E583 不承担 polarity 稳定叙述。5 个 Yuquan 二维 candidates 中，3 人有 inventory 但缺 derived eligibility cache，2 人缺 inventory，均记为 `blocked_input`。E139 与 `yuquan_zhangjiaqi` 为单杆 `geometry_2d_supported=false`，不进入二维分母；E139 仅保留在 `sensitivity_1d/`。当前 canonical run=`20260718T071020Z_d99c96ec`：producer/renderer 只写 `runs/<run_id>/artifacts/`，完整默认 batch 验收后才替换顶层 index，并最后原子替换 manifest 作为 completion pointer；explicit subset 和中断 run 不改 canonical artifact。

**shared-matched maxAB 空间置换 null（2026-07-18，fixed-time-mapping v2）**：新版 null 对上述 7 人使用相同 frozen `shared_a/shared_b`、fingerprint、二维 geometry 和成功 seizure 集；每个 `seizure×replicate` 固定一次空间映射贯穿全部 66 窗，每次 shuffle 都重算 A/B、mirror choice 与 maxAB。R=1000，vectorized-vs-exact 及 observed-vs-source 最大误差均 ≤`5.6e-14`。within-shaft 结果为 **3/7** 至少一个 cluster（E1084、E1146、E590），**2/7** 有 maxT 窗（E1084、E1146）；这是 per-subject 时间分辨描述，不是 formal cohort gate。旧逐窗置换的 `5/7` 已撤回；旧 own-plane null 的 E922/E1146、`13/20`、`7/20`、`2/20` 等数字也不得接到新版图。canonical 输出为 `spatial_null/`，manifest 含 7 个 summary，共 35 个 null artifacts。

**科学边界**：冻结 archive 的 early-ictal shared-field cohort null（二维共线 n=7，within-shaft p=0.346；shared-vs-own p=0.938）仍是 cohort 级主参照。这里的 sliding-window null 回答逐人哪些时间段偏离杆内置换，不能把 3/7 cluster 写成 cohort superiority，也不能因为 pre-onset cluster 而写 onset-emergent alignment。within-shaft power 仍依赖 shaft sizes；E583 的 3/22 coverage 尤其需要降级解释。

### Supplementary Video 2: Figure 3C peri-onset field evolution

E10 | SZ3 的静态 Figure 3C 被扩展为 `[-120,+20] s` 动态场。上排固定冻结 TA 间期场，仅更新右侧
1–150 Hz baseline-robust-z 发作场；下排显示幅度感知模板表达量 `Q=max(|q_A|,|q_B|)`，并以红蓝点标记 TA/TB 主导。每帧为
10 s 滑窗，步长 2 s，右侧 power-z 色标跨帧固定，避免逐帧归一化制造虚假的场强变化；`[0,10] s`
帧必须通过正式 Fig3C 的逐触点数值一致性闸门。正式投稿入口为
`results/paper-ready-figure/supplementary-video-2.gif`，生成 sidecar 位于
`results/paper-ready-figure/supplementary-video-2-fig3c-peri-onset-field/figures/`。该视频只补足代表病例的
动态可视化，不新增 cohort、onset-emergent alignment、template-free replay 或机制结论。

### Fig4: Data-driven interictal SNN（A–G 修订版，B 预留）

**当前状态**：A–G 已按作者 2026-09-03 最新顺序重排，B 明确预留为空；科学口径仍受冻结结果边界约束。唯一完整布局为
`results/paper-ready-figure/fig4/figures/fig4-complete-layout.png`；同目录保存无角标 A、C–G 独立
PNG/PDF，B 当前没有独立文件。完整 producer 为 `scripts/paper_figures/build_main_figure_4.py`，只消费冻结产物，不重跑仿真。

**作者布局叙事**：

- **A**：将 local E/I microcircuit 与 patient-specific E/I substrate 拼为同一 panel。左侧机制图置于虚线框内、标题在框外；右侧保留触点几何和 E/I substrate，不显示各向异性 E→E corridor 或 possible-core 覆盖。左右局部框按同一视野尺度匹配；左侧 scale bar 为 0.5 mm，右侧显示 −10–10 mm 坐标，触点 sampling footprint 以低透明度深绿色表示。
- **B**：使用 A 右侧现有留白，完整拼板中只保留 B 角标；后续补入 data-driven 不同参数对患者间期事件复现的影响，当前不放占位文字或临时数据。
- **C**：展示冻结 data-driven Node field 与 Model TA/Model TB 两种空间模式。Node-field 色条不得与 3D 图或 Model TA 的 y-label 重叠；两个模型方图缩至网格单元的 74%。
- **D**：精简后的模型/患者 rank profile；图例置于绘图区下方，不压低主图。
- **E**：模型 MTA/MTB 与患者 TA/TB 的 contact-split cross-fit matrix；两个命名匹配单元格分别显示 matched within-shaft contact-permutation 结果。
- **F**：同一冻结网络的 30–80 Hz virtual-contact readout；MTA/MTB 阴影按各自实际 recruitment-onset span 加 12 ms 边界显示，不使用完整 detector-event 时长；不显示 a.u. 比例文字或逐通道黑色 onset 点。
- **G**：34 人 canonical-layout held-out recovery 与 matched within-shaft null；完整显示 34/23/15/11 四层人数。

**当前口径**：图支持 patient-constrained Node/local-connectivity substrate 能产生自发、可返回、双簇
组织的间期 event-like activity，并给出同一冻结 E10 development case 的 direct readout、自然聚类与
patient-template rank 对照。34 人 canonical layout 上存在弱 held-out 优势，但 same-network K=2 门槛未过，
真实几何优势归零。因此不能写“复现了完整患者间期活动”“恢复了解剖 core”或“证明了 EE/E→I 临床
因果机制”。原 KMeans heatmap/rank-distribution panel 移至 FigS7-E；冻结 pathway ablation 作为 FigS7-A–D 展示。详见
`docs/archive/topic4/sef_hfo/data_driven_interictal_snn_closeout_2026-08-17.md`。

### FigS7: 冻结局部连接路径消融与 KMeans 结构

12 张全新配对网络分别运行 Node、+EE、+E-to-I 和 +EE+EI 四个冻结连接臂，以 network seed
为独立单位展示 Mode 1/2 事件占比、de novo KMeans K=2 与冻结标签的 balanced match，以及
返回事件 OOD 比例。E 为从主图移入的 627 个 formal clean model events 的 masked-rank heatmap、
逐触点 rank distribution 和唯一共享色条。独立 A–E panel 不带角标，`supp_fig7-complete-layout` 才带 A–E；
A–D 星号表示相对 Node 的配对 90% network-bootstrap CI 不跨 0（4,096 draws，未做多重比较校正）。
该补图只支持 development case 中的模型内部 pathway effect pattern 与事件结构，不支持患者因果连接、
解剖 core 恢复或真实几何泛化。

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
