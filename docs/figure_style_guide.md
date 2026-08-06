# 图可视化标准（Figure Style Guide）

> 本仓库**反复出现的图类型**的固定画法。新图必须按本文件的「示范图 + viz 机制 + 配色/轴约定」来画，
> 不要每次重新发明布局或配色。目的：跨 topic 的同类图长一个样，读者一眼就知道在看什么。
>
> 用法：要画某类图前，先翻到对应 topic 小节，照「示范图」的布局和「配色/轴」复刻。
> 导航见 [`results/FIGURE_INDEX.md`](../results/FIGURE_INDEX.md)。
>
> 锁定日期 2026-06-14。`epilepsiae:139` 是跨「模板 / 几何 / swap」共用的 canonical 示例被试（7 触点单杆 HL2–HL8，方便对照）。

---

## 0. 贯穿全局的硬规则（先于一切单图约定）

**0.1 配色锁定（同一物理量在所有 topic 用同一配色）**

| 物理量 | 配色 | 含义 |
|---|---|---|
| 传播顺序 / 时序 rank | **viridis**（顺序色） | 深紫=最早(First)，黄=最晚(Last)，统一写 `First → Last` 或 `0=early,1=late` |
| rank 位移 / swap（Δr） | **diverging 红-蓝**（0 居中） | 红=源变汇（rank 变大），蓝=汇变源（rank 变小），白≈不动 |
| SOZ | **黑环 overlay** | 永远只是叠加标注，**绝不作为度量输入**；图里必须写明 "SOZ overlay only, not metric input" |

不要用 jet / rainbow。顺序量永远 viridis，带正负的差值永远 diverging。

**0.2 paper-grade 自洽**（沿用既有规矩，见记忆 `feedback_figure_self_contained_paper_grade`）

- 不出现内部术语：`§X`、`cluster_id`、`PR-6`、`stable_k=2` 等不进坐标轴/图例/标题面向读者的位置（标题里的 `k=2`/`τ`/`MI` 这类是统计量，可保留）。
- 坐标轴贴紧数据（tight axes），不留大片空白。
- 一张图一套共享图例 / colorbar，不每个子图各放一个重复的。
- 流程：render → 亲自目视 → 改 → 再 render，确认无误才提交。

**0.3 多面板纪律（CLAUDE.md §7）**

一个面板答一个独立科学问题。同一构造的两种角度 = 冗余，删一个。X-vs-Y 联合散点只在「边际 X、边际 Y 各自看不出耦合」时才用。

**0.4 论文主图定稿纪律（Fig. 1 对话锁定，2026-07-15）**

- **复用原 producer，不在拼版脚本里自由发挥**：已有统计图、rank distribution、heatmap 或显著性标记函数时，正式 panel 必须直接调用原函数或做向后兼容的小参数扩展。若视觉结果与旧图不同，先核数据合同、channel order、坐标映射和 helper 调用，不先重写一种“相似画法”。
- **同类谱图共用算法**：同一 figure 中承担对照关系的 spectrogram 必须使用同一显示量、平滑和归一化定义。质心必须从画布实际显示的高频增强量计算，并落在主增强连通区，不能用另一套权重或全局质心落到能量谷中；STFT cell、marker 与 x 轴必须共用真实坐标，左右边界不得出现自动白边。
- **坐标起点与留白**：当量的自然下界是 0（ROC、MI、matching-uplift 等）时，正式主图 x/y 轴从 0 起。time heatmap、spectrogram 和 waveform 用显式 `xlim`/cell edges，并关闭自动 x margin。紧凑不是压缩信息：rank ridgeline 要保留原始可读宽度和形状。
- **跨子图数据必须可核对**：同一案例的原序图和聚类重排图必须来自同一 subject、同一有效 event 全集；各 cluster 的 `n` 之和必须等于原序图的 `n`。heatmap、rank distribution 和 centroid/profile 必须共享 channel order、`ylim` 和通道中心，禁止各自排序后视觉错位。
- **布局先对齐再装饰**：需要联合阅读的 c1/c2 放在同一文件，上下等长、列宽对齐、间距收紧；共享 colorbar 紧贴主图并放在读图顺序合理的位置。cluster 分割使用清楚的白色断带/断轴语法，必要时加灰色斜线表示省略区，不用红线冒充数据边界。
- **主图画布只留读者需要的信息**：不要在主图标题写 `40/40`、事件总数句、内部 MI/inter-template 状态或 `events over time` 一类工程说明；这些放 caption、README 或 metadata。统计量统一用论文缩写（如 `MI`），dataset label 使用常规字重并贴近轴；标题、图例和注释不得遮挡数据点。
- **统计图保留完整统计语法**：data-vs-null 图沿用已验证的 violin/box、whisker、subject points、error representation 和显著性括号；不得为了“简洁”删掉 p 值显著性层或不确定性。散点比较图保留零起点、参考对角线、具有含义的灰色区域和颜色图例，annotation 放在无数据区。
- **最终检查顺序固定**：render 后同时目视 PNG 与核对 metadata；逐项检查算法一致、事件数守恒、channel 对齐、marker 注册、坐标零点、白边、标题/图例遮挡和字体层级，再生成 PDF 并提交。

---

## Topic 1 · 间期事件「按什么顺序传」

### 1a. 刻板时序模板（rank template）

- **示范图**：[`results/interictal_propagation_masked/figures/per_subject/epilepsiae_139_propagation.png`](../results/interictal_propagation_masked/figures/per_subject/epilepsiae_139_propagation.png)
- **回答**：这个被试的间期群体事件，通道点火先后顺序是否刻板？是否存在两种（正/反向）模板？
- **布局（双行 + 右侧分布）**：
  - 上行：原始 rank 热图（行=通道，列=pop events **按时间排列**，色=`First→Last` viridis）＋右侧 per-channel rank 分布**堆叠条**。
  - 下行：KMeans **k=2** 重排后的热图（列按 cluster 分组）＋右侧 cluster rank 分布**折线**（C0 vs C1 两条）。
- **标题约定**：`<dataset>:<subject> | repro=<strong/…>`，子标题带 `n=<事件数> | τ=<within-τ> | MI=<…> | p=<…>`。
- **配色**：viridis `First → Last`（0=早 1=晚）。

### 1b. swap 节点（rank displacement，三联：个体 → 队列 → 临床）

- **回答**：两个模板之间，哪些通道把「源/汇」角色对调了（= swap 节点），这些节点落不落在临床 SOZ 上。
- **示范图（三张一组，固定这三层）**：
  1. **个体** [`.../rank_displacement/figures/per_subject/epilepsiae_139_displacement.png`](../results/interictal_propagation_masked/rank_displacement/figures/per_subject/epilepsiae_139_displacement.png)
     单行热图，通道按 `rank_T_a`（**source → sink**）排列，色=Δr（模板A→B 的 rank 位移），**diverging 红-蓝**。|Δr| 大的格=swap 节点。标题 `<dataset> <subject> | k=2 | fwd/rev✓ | τ=… | F_norm=…`。
  2. **队列** [`.../rank_displacement/figures/cohort_displacement_heatmap.png`](../results/interictal_propagation_masked/rank_displacement/figures/cohort_displacement_heatmap.png)
     行=被试（按 F_norm 排序），列=沿 T_a 的通道位置（source→sink），色=`signed Δr (= rank_T_b − rank_T_a)` diverging 红蓝。strict 用实心 `►◄`、candidate 用空心 `▷◁` 标记；右侧附 `F_norm` 横条（2/3、1 参考线）。
  3. **临床** [`.../rank_displacement/figures/swap_clinical_soz_overlap.png`](../results/interictal_propagation_masked/rank_displacement/figures/swap_clinical_soz_overlap.png)
     (A) Precision、(B) Recall 两面板，x=`k`（swap=T_a 中 top-k ∪ bottom-k），y=swap 端点 vs 临床 SOZ 的 precision/recall。cohort median 用**红粗线**，random baseline 用**灰虚线**，附 AUC（median over k）。
- **配色**：Δr 一律 diverging 红-蓝，0 居中。

---

## Topic 2 · 间期事件「什么时候发生」（组合：周期性 + 间隔分布，两张并列）

- **回答**：间期事件在时间上有没有节律（周期性）；事件间隔的分布是什么形状。
- **示范图（两张各管一面，成对出现）**：
  1. **周期性** [`results/event_periodicity/figures/yuquan_cohort_psd_stack.png`](../results/event_periodicity/figures/yuquan_cohort_psd_stack.png)（Epilepsiae 同名）
     左=各被试归一化功率谱**堆叠**（x=频率 0–8 Hz，一被试一色，纵向偏移堆叠）；右=峰频**直方图**+中位数虚线。
  2. **间隔分布** [`results/event_periodicity/figures/yuquan_iei_summary.png`](../results/event_periodicity/figures/yuquan_iei_summary.png)（Epilepsiae 同名）
     左=各被试 IEI 幂律指数 α **横条** + `α=2.0` 参考虚线；右=幂律 vs 对数正态 **似然比横条**（R>0=幂律占优）。
- **轴约定**：横条图被试名左侧纵排；参考线（α=2.0 / median）用红虚线。

---

## Topic 3 · 间期事件「空间几何 / 传播方向」

- **示范图**：[`results/spatial_modulation/propagation_geometry/observation_readout/figures/static_maps/epilepsiae_139.png`](../results/spatial_modulation/propagation_geometry/observation_readout/figures/static_maps/epilepsiae_139.png)
- **回答**：把时序模板摊到真实触点平面上，传播是不是沿一条稳定的空间轴；两个模板是不是方向反转。
- **布局（2×2）**：
  - 左列：触点**散点**（subject-fixed **mm 坐标**，色=typical order `0=early,1=late` viridis，SOZ=**黑环**、SOZ 触点画大圈）；上=t_a，下=t_b。子标题带 `rho_x_rank=…`。
  - 右列：**高斯平滑 order field**（σ=6 mm）连续梯度（同 viridis）；上=t_a，下=t_b。
  - 两模板上下堆叠，直接看出 t_a 与 t_b 的梯度方向相反。
- **标题约定**：`<dataset>:<subject> | t_a top, t_b bottom | SOZ overlay only, not metric input`。

### 3b. Template A/B 主方向 + 解剖读出

- **示范图**：[`results/paper-ready-figure/fig_interictal_ab_direction_axis/figures/yuquan_example_interictal_ab_direction_axis.png`](../results/paper-ready-figure/fig_interictal_ab_direction_axis/figures/yuquan_example_interictal_ab_direction_axis.png)
- **复现入口**：`scripts/paper_figures/plot_interictal_ab_direction_axis.py`。
- **回答**：两类模板是否各自形成稳定的三维 earliness gradient；两条轴是否近似共线；在共线时，A/B 真正的 `early → late` 传播方向是否同向或反向；这些方向覆盖哪些脑区。
- **方法锁**：A/B 先分别拟合 `e_T=-z(rank_T)` 的三维梯度 `g_T`，正式传播向量固定为 `u_T=-g_T`。producer 的 `u` 与所有箭头都必须表示 early→late；原始晚→早梯度只能写作 `earliness_gradient_u`，绘图层不得再次取负。只有两轴都过冻结 QC 且 `|cos(u_A,u_B)|>=0.5` 才画共享直线。`D_AB` 只作触点相对早晚对比着色，不直接当传播箭头。
- **布局**：左上 A/B 两个 contact-scale rank-vs-axis 拟合；左下为区域级 temporal-order overlay；右侧为 subject-native 透明脑表面、真实电极杆与红/蓝 early-to-late 箭头。SOZ 仍用黑环，并写明 overlay only。
- **解剖边界**：区域名称不参与轴拟合或定向。主触点池不足以解析 region route 时必须写 `not resolved`；expanded-contact 版本只能叫 sensitivity。要写“复用了自然解剖梯度”，还需 cohort-level region transition，并用杆内触点 shuffle + 整杆 profile 重分配两级几何 null 分开检验杆内深度梯度与跨杆区域路线。

### 3c. 患者特异 TA/TB 间期传播场（Fig2-E 候选）

- **完整规范**：[`docs/topic5_interictal_field_figure_spec.md`](topic5_interictal_field_figure_spec.md)。后续所有间期 TA/TB 场图均以该文件为唯一视觉合同。
- **示范图**：[`results/paper-ready-figure/fig2e_interictal_template_fields/figures/epilepsiae_1146_interictal_AB.png`](../results/paper-ready-figure/fig2e_interictal_template_fields/figures/epilepsiae_1146_interictal_AB.png)。
- **复现入口**：`scripts/plot_topic5_interictal_template_ab_fields.py`；完整图复用 `plot_interictal_ab_subject()` / `plot_interictal_ab_atlas()`，拼版复用 `build_interictal_ab_panel_payloads()` / `draw_interictal_rank_field_panel()`，底层场只调用 `draw_topic5_field_panel()`。
- **硬锁**：正式轴只读冻结 artifact 的 early→late `u`；共线才用 shared plane，否则用各自 own plane；transverse 只按电极几何定号；显示核固定 6 mm 且不得冒充评分 kernel；viridis 0=early/1=late；单患者只用一个共享 xlabel、一个 y 轴和一个与 field 等高的 colorbar。
- **禁止**：不复制 renderer，不按颜色翻轴，不强行让 TA/TB 共面，不把单杆图解释为二维 field，不把这张间期图写成发作场一致性证据。

### 3d. 间期单事件包络传播 frame / GIF（Fig2-C 候选）

- **唯一规范**：[`docs/fig2c_interictal_event_envelope_field_spec.md`](fig2c_interictal_event_envelope_field_spec.md)。之后所有“间期传播场 frame / event-envelope GIF”先读该文件；不要套用静态模板场 Fig2-E 的 viridis/rank 语法。
- **示范图**：[`results/paper-ready-figure/fig2c_interictal_event_envelope_field/figures/fig2c_candidate_E1146_interictal_event_envelope_field.png`](../results/paper-ready-figure/fig2c_interictal_event_envelope_field/figures/fig2c_candidate_E1146_interictal_event_envelope_field.png)。
- **动态 sidecar**：同目录 `fig2c_candidate_E1146_interictal_event_envelope_field.gif`。
- **复现入口**：`scripts/paper_figures/plot_fig2c_interictal_event_envelope_field.py --subject epilepsiae_1146`；core renderer 为 `scripts/plot_topic5_interictal_event_envelope_field.py`。
- **视觉硬锁**：两行 TA/TB；左侧为方形 Fig1a readout，右侧为 6 个等高方形 frame；静态时间只到 +50 ms；包络用 `magma`，TA/TB 共享 `vmax`、各有等高 colorbar；TA 红/TB 蓝；标题只写匿名患者号；所有 frame 保留 x ticks。
- **数据硬锁**：frozen fingerprint/contact order/shared plane 不重拟合；单带 `return_hil_enve`；participant-only support；6 mm 只作 display kernel；GIF 与静态图使用同一 exemplar/几何/色标，2 ms biological step 与 playback fps 分开记录。
- **适用范围硬锁**：这是单事件规范——TA/TB 每行各一个 exemplar。未来多事件 GIF 必须另立事件边界、事件间隔、逐事件 t0 和抽样合同，不得把 event train 塞进本 renderer。
- **禁止**：不称 template-free，不把 Hilbert amplitude 写成 power，不把单被试两次事件写成 cohort 传播定律、跨未采样组织的 traveling wave 或机制证明。

---

## Topic 4 · 机制模型（SEF-HFO / cm-SNN）

### Figure 5 上半部分定稿布局（MZ early-field bridge；2026-07-22 LOCKED）

- **正式图与 producer**：[`results/paper-ready-figure/fig_mz_early_bridge_v2_zm_tau500/figures/fig_mz_early_bridge_v2_zm_tau500.png`](../results/paper-ready-figure/fig_mz_early_bridge_v2_zm_tau500/figures/fig_mz_early_bridge_v2_zm_tau500.png)，由 `scripts/paper_figures/plot_fig_mz_early_bridge_v2.py` 生成。
- **Figure assignment**：当前整张复合图作为 **Figure 5 上半部分**；下半部分另行设计与拼接。最终组版可在 block 外加 panel letter，但不得拆换内部 panel。
- **上排锁定**：左侧连续 z+m native Virtual-SEEG（30–80 Hz），红虚线为统一的 early onset = operational `t120−120 ms`，只保留 `early onset` 与 `TB sample event` legend；右侧为单条 z–m 慢状态轨迹和 operational boundary 示意。
- **下排锁定**：四张等宽空间图依次为 `TB event order | Early-onset energy | Baseline mode | Early-onset mode −120 ms`；统一 `TA shared axis (mm)`，最左图保留 `y (mm)`；前两张各自使用窄 colorbar，后两张共用 mode colorbar。
- **空间 probe**：Baseline mode 左上角保留灰度 localized E-rate spatial-probe glyph 和输入箭头。它只解释“小空间扰动如何进入冻结系统”，不是额外仿真；mode panel 是 frozen Jacobian leading-mode loading，不得改写为 fixed-kick response。
- **冻结边界**：未获用户明确解锁，不改 panel 数、相对宽度、时间锚点、标题、legend、colormap、colorbar、probe glyph、电极布局或坐标标签。Figure 5 下半部分的新增内容不得反向挤压或重排这一上半部 block。

- **示范图**：[`results/paper-ready-figure/fig5_core_model_s3_brakeoff/figures/core_model_s3_brakeoff.png`](../results/paper-ready-figure/fig5_core_model_s3_brakeoff/figures/core_model_s3_brakeoff.png)
- **默认标准（SNN 仿真图都按这个画）**：`mechanism + tempA source + tempB source + electrode readout`。上面已单独锁定的 Figure 5 上半部分是明确例外；除此之外，除非用户明确要求做诊断图、参数扫描图或 pipeline/KMeans 结果图，任何 SNN 相关主图 / paper-ready 图都不得回到旧的三行 Forward/Reverse/C 行堆叠布局。
- **回答**：同一个 SNN 基底里，机制变量在哪里、两种特异性组合如何产生相反传播、同一虚拟 SEEG montage 是否能在电极 readout 中读出正/反事件。
- **布局（单行 4 列）**：
  - **mechanism**：左 1 格，画底物和机制变量。必须标出关键连接 / 病理范围，例如 E->E 长轴作用范围、病灶核、虚拟电极位置。这个 panel 只解释“机制是什么”，不堆长说明文字。
  - **tempA source**：中间第 1 个方形 panel，画 tempA / source-A / 组合 A 的代表传播事件。点云颜色为传播起始相对时间，沿用 viridis `early → late`；红圈标病灶 / source 区，星号只标该事件的实际 source。
  - **tempB source**：中间第 2 个方形 panel，画 tempB / source-B / 组合 B 的代表传播事件。坐标轴、colormap、contact 标注必须与 tempA panel 完全一致，便于直接比较两种组合是否反向。
  - **electrode readout**：右侧宽 panel，画同一 montage 的多事件虚拟 SEEG train。只画 active contacts，y 轴为 contact，x 轴为 time；用不同颜色 shading 区分 forward / reverse clean propagation events，黑点/线标每个事件的 peak order。
- **标题约定**：子图标题只用短名：`mechanism`、`tempA source`、`tempB source`；readout 不加长标题。必要统计写进 metadata/README，不压到图上。
- **配色 / 渲染锁定**：
  - mechanism 底物用原 SNN 机制图风格：`plasma`，点大小和透明度沿用原机制脚本的粗点风格。
  - 传播事件用 viridis，早=紫、晚=黄；不得换成低饱和替代色。
  - readout 事件 shading：forward 用暖色，reverse 用浅蓝；同一图只保留一个共享 legend。
  - 电极颜色固定：沿轴 / A shaft 为橙色，横轴 / B shaft 为青色。
- **输出纪律**：正式 SNN 仿真图脚本放 `scripts/paper_figures/`，输出放 `results/paper-ready-figure/<figure_name>/figures/`，同时可写一份兼容旧 Topic 4 结果目录；`figures/README.md` 必须说明四列各自回答什么。
- **科学边界**：这类图只能写成“模型底物 + 两种特异性组合 + 虚拟 SEEG readout”的机制/读出示意。不能因为图上有正反事件，就直接声称真实病人机制被证明；pipeline/KMeans 验证若需要，另作补图或下游结果图。
- **建模图 KMeans 核验图（modeling-KMeans companion）**：用于回答“模型 readout 里的多事件是否自然分成两类，以及这两类是否对应 tempA/tempB 或真实模板”。它是 SNN 四列 readout 图的**配套核验图**，不是新的机制图，也不是 cohort 统计图。
  - **输入合同**：只消费同一模型 readout 的 clean directional events，不重跑仿真；必须在 metadata/README 写清楚 event filter、`k_dir`、seed/tag、n_events、每类 event 数。若 readout 没有两个方向或每方向事件数不足，模型×真实模板矩阵必须显示 N/A，不能用两个 KMeans 簇硬冒充 forward/reverse。
  - **布局（单行四块，左→右）**：`clustered event heatmap | per-channel rank distribution | cluster rank profile | model-vs-real 2×2 similarity matrix`。最左 heatmap 占主导宽度；中间两块紧凑，不挤占最右矩阵空间。
  - **heatmap**：列按 KMeans 簇分组，红色边界标簇切分；灰格表示该事件未招募该触点；rank colorbar 竖放在 heatmap 右侧，标签为 `First → Last`。
  - **左三块 y 轴**：必须 channel-for-channel 对齐。后两块不得各自重算 y 轴；应复制 heatmap 的 `ylim / yticks / yticklabels`，同一通道在三块里必须同一高度。
  - **cluster 命名 / 配色**：显示层不用 `C0/C1`，改用模板语义。`t_a` 固定红色，`t_b` 固定蓝色；原始 KMeans id 只保留在 metadata。若某个模型不是 t_a/t_b 语境，则用对应模板名，但必须固定“模板名→颜色”映射。
  - **cluster profile**：画每个模板簇的 mean±std rank profile；legend 放在本 panel 内右上角，不单独占一行。
  - **model-vs-real matrix**：右侧矩阵为模型模板 × 数据模板的 Spearman 相似性矩阵；只用 star 显示方向性 channel-shuffle permutation p，不在格子里堆数值；矩阵 cell 必须 `aspect=equal`，右侧 colorbar 与矩阵本体等高。
  - **报告口径**：图上/README 至少报告 cluster size、direction purity、within-cluster tau、shared-overlap corr、矩阵是否 valid。结论只能写“readout 聚类/模板一致性核验通过/不通过”；不能单独写成机制因果证明。
- **M3A-v2 诊断变体（closed-loop negative screen）**：若要目视审阅慢变量动力学，可沿用同一四列视觉语法，但必须在 README/metadata 中注明它是 visual diagnostic，不是主 claim 图。Step4 的单核 kick 结果若镜像成 tempA/tempB 两端 probe，legend 必须写 source identity（tempA-source / tempB-source），不得把 source identity 写成传播方向或发作方向。示范输出：`results/paper-ready-figure/fig_m3a_v2_step4_dynamics/figures/`，脚本：`scripts/paper_figures/plot_fig_m3a_v2_step4_dynamics.py`。q_I/g_K gap sweep 的代表状态同样使用这个规则，输出 `results/paper-ready-figure/fig_m3a_v2_1_qigk_gap_dynamics/figures/`，脚本 `scripts/paper_figures/plot_fig_m3a_v2_1_qigk_gap_dynamics.py`。
- **M3A-v2.2（sustained 协议 + h_G 载体）两类图（2026-06-29）**：(1) **结果汇总图**（不是四列、不是 SNN 重跑）——读自主探索的 `per_run.jsonl`（3184 sim），三联面板各答一个独立问题：slow-off 失败模式 vs r_hold / `q_I+g_K` 表型组成+候选数 / 干净事件 vs partial-fill 目标框；脚本 `scripts/paper_figures/plot_fig_m3a_v2_2_explore_summary.py`，输出 `results/paper-ready-figure/fig_m3a_v2_2_explore_summary/figures/`。**统计主张只在此图 + 归档 doc，不在四列动力学图。** (2) **代表性动力学四列图**——沿用上面的 M3A-v2 诊断变体规则（visual diagnostic、非主 claim、mechanism 轴线**无箭头**、legend 标 source identity 非方向），代表 case = fail-closed tonic（slow-off / `q_I+g_K`）+ 唯一干净事件（backup r=0.85），脚本 `scripts/paper_figures/plot_fig_m3a_v2_2_dynamics.py`，输出 `results/paper-ready-figure/fig_m3a_v2_2_dynamics/figures/`。四列图是**单 seed 重跑示意**，metadata 注明判读以 sweep + 汇总图为准。
- **被试特异性变体（subject-specific SNN，Fig4A/B 起）**：Fig4A 用同一四列标准，但底物按**病人真实电极布局**摆放（示范 `results/paper-ready-figure/fig_subject_snn_epilepsiae_1146/`，脚本 `scripts/paper_figures/plot_fig_subject_snn.py`）。Fig4B 是同一 readout 的 KMeans=2 核验图，脚本 `scripts/paper_figures/plot_fig_subject_snn_kmeans2.py`。锁定约定：
  - 两个低阈值核 = **两类间期模板各自最早的 k 个电极**（`template_source` 摆放，`src/sef_hfo_subject_placement.py`），即两类模板的 source 区，落在传播轴两端；不要用 swap decision_k 的宽 strip 质心（会被拉向中间）。
  - 用**真实几何 plane-fit**（核间距自然 ≈ blessed sep0.7），不得人为 core-anchor 到固定间距。
  - mechanism panel 必须显示**核与电极间期最早区 overlap**（高亮核成员触点）+ E→E 长轴带。
  - readout 用 `k_dir=2`（病人电极比模型密杆稀疏的放宽，**载重参数**，必须在 metadata/README 注明 k_dir=3 的退化情况）。
  - 诚实口径：readout 若用 spontaneous twoend，需注明自发双向**与 seed 有关**；separate-then-pool 只能写"仪器对齐"不能写"自发机制"。
  - **LOCKED 模式（2026-06-26）= 每个 subject-SNN 案例固定出两张主图：Fig4A（readout 四列）+ Fig4B（KMeans 核验四块）。** Fig4C（real-vs-model profile）、Fig4D（组合 S 置换 null）是可选 supplement。
  - Fig4B 必须遵守上面的 **建模图 KMeans 核验图** 规范；E1146 当前示范脚本为 `scripts/paper_figures/plot_fig_subject_snn_kmeans2.py`。

---

## Topic 5 · ictal field readout / peri-onset trajectory

Topic 5 仍有多条探索线，只有已经进 paper-ready Fig3 的 field readout 图型先锁定。其它候选见
`results/FIGURE_INDEX.md` 的 Topic 5 段，暂按个案处理，不强制统一布局。

### 5a. Fig3-B：peri-onset field similarity trajectory

- **示范图**：[`results/paper-ready-figure/fig3_peri_onset_field_similarity/figures/epilepsiae_1146_peri_onset_field_similarity_paper_ready.png`](../results/paper-ready-figure/fig3_peri_onset_field_similarity/figures/epilepsiae_1146_peri_onset_field_similarity_paper_ready.png)
- **复现入口**：`scripts/paper_figures/plot_fig3_peri_onset_field_similarity.py --subject epilepsiae_1146`
- **上游数据**：`results/topic5_ictal_recruitment/field_dynamics_signed/epilepsiae_1146_signed_broadband_1_150Hz_similarity_timecourse_m120_p20_10s_step2s_per_seizure.csv`
- **上游生成命令**：`python scripts/plot_topic5_signed_broadband_similarity_timecourse.py --subject epilepsiae_1146 --start-sec -120 --stop-sec 20 --band-lo 1 --band-hi 150 --window-sec 10 --step-sec 2`
- **全 subject 批处理**：`scripts/paper_figures/run_fig3_peri_onset_all_subjects.py`（fail-closed 逐 subject 跑上面两步，一个 subject 失败不中断整批）；主索引 `results/paper-ready-figure/fig3_peri_onset_field_similarity/fig3_peri_onset_subject_index.{csv,json}` 汇总每 subject 的 status / drop_reason / n_seizures / n_windows / maxAB + signed A/B 摘要 / 输出路径。当前 20/35 出图，15 因缺上游 T0 eligibility 缓存 drop。每 subject 同一 locked 布局；这是 per-subject material pool，非 formal cohort gate。
- **回答**：在同一 subject 的多次 seizure 中，onset 前后 1-150 Hz signed robust-z 能量场是否持续接近间期 propagation field scaffold；以及这种接近是否有稳定的 signed A/B polarity。
- **布局（单行双面板）**：
  - Panel a：`max(|r_A|, |r_B|)`，即 sign-free maxAB scaffold similarity。
  - Panel b：signed `r_A` 与 signed `r_B`，分别对应 template A/B。
  - 两个 panel 都使用同一时间范围、同一窗口定义、同一 seizure 集合。
- **时间轴合同**：
  - 数据窗口固定为 `[-120,+20]s`，10 s sliding window，2 s step。
  - x 轴画 **window center**，因此当前显示中心范围是 `[-115,+15]s`。
  - `xlim` 必须贴第一个/最后一个 window center，不留 Matplotlib 自动白边。
  - 0 s 用灰色虚线标 clinical onset。
  - 不把 `+20s` 之后的发作中轨迹直接接上；完整发作期比较必须先做 duration warping 或阶段对齐。
- **线型 / 统计显示**：
  - 浅细线：单次 seizure trajectory，低 alpha，只作为异质性背景。
  - 粗线：跨 seizure median，是主视觉读出。
  - 阴影：跨 seizure IQR；不要用 mean±SD 作主图阴影。
  - 不画诊断下排。跨 seizure variance、`n_seizures`、drop 信息写入 summary JSON / README。
- **配色**：
  - maxAB：Morandi rust `#A35E48`。
  - template A：红 `#B2182B`。
  - template B：蓝 `#2166AC`。
  - 单次 seizure：浅灰；不要让个体线压过 median/IQR。
- **禁止事项**：
  - 不用 step 图作为 paper-ready 主版；step 只可用于检查窗口边界。
  - 不把 `maxAB |r|` 和 signed A/B 混成一个指标。
  - 不用 1-45 Hz cache 顶替 1-150 Hz；Fig3-B 的 1-150 Hz 特征 = notch 滤波输入（50/100/150/200Hz）后对 `[1,150]` 全 bin 求和、**无额外 FFT-bin line mask**（区别于 Fig3-A / v2 的额外 bin-mask 版本，谐波处理不同——别把两者当同一合同）；用别的频段要明确标注。
  - 不把 signed A/B sidecar 写成 formal gate；当前 formal-ish scaffold 读出仍是 sign-free / maxAB 语义。
  - 不写 replay、direction replay、timing-order replay、mechanism proof。

**当前口径**：这类图是 Fig3 field concordance 的 subject-level 动态素材。Panel a 支持 coarse scaffold similarity 在 onset-near 时间轴上持续偏高；Panel b 说明 signed polarity 在 seizure 间是否稳定，只能作为 polarity sidecar。

### 5b. Fig3-B maxAB 空间置换 null（两档）+ 时间维校正

- **示范图**：`.../spatial_null/figures/epilepsiae_548_maxab_spatial_null.png`（观测远高 all-contact null 但**整段贴 within-shaft null**＝相似几乎全是杆几何，反例）与 `epilepsiae_1146_...png` / `epilepsiae_922_...png`（扛过 within-shaft + maxT 的稳健正例）。
- **复现入口**：`python scripts/run_topic5_fig3b_maxab_spatial_null.py --all-ok`（`--skip-existing` 断点续；`--rebuild-from-stats` 从 `.npz` 只重算校正+重画不重跑；`--verify` 校验向量化读出与 exact `score()` 一致到机器精度）。
- **回答**：某 subject 发作前后 maxAB scaffold similarity 是否高于**保留植入几何**的空间置换——即高相似是不是电极摆放（尤其杆级）自带的。**只检验 maxAB scaffold**，不做 onset increment / signed A/B / 多频带。
- **两个 null（承重合同；都：同批 seizure / 时间窗 / A|B 模板 / 场平滑 sigma / maxAB 逻辑，只打乱每窗 per-channel 能量值[值换位、support 随位置不动]、完整重跑 值→`make_field_record`→support 加权平滑→镜像不变相关→`max(|r_A|,|r_B|)`、对 seizure 取中位、每次 seizure 独立置换、R=1000；禁止只洗已算好的 maxAB）**：
  - **all-contact**（弱，`channel_shuffle`）：值在**全部触点**间打乱。
  - **within-shaft**（强，主，`within_shaft_shuffle`）：值只在**每根杆内**打乱，保留"哪根杆热"的植入几何。
- **三档显著性（stats CSV 里两 null 各一套，都单侧上尾）**：pointwise（逐窗 `(1+#{null≥obs})/(R+1)`，未校正）< maxT（逐窗 FWER，Nichols-Holmes 标准化 z 的窗间 max）< cluster（Maris-Oostenveld，cluster-forming=pointwise p<0.05、mass=Σz、null=每 perm 最大 cluster mass；时间维、对持续抬升敏感＝paper-facing"显著区间"）。
- **布局（单面板）**：粗 rust=观测中位、浅 rust 带=观测 IQR；蓝虚线+蓝带=within-shaft null 中位+95%；灰点线=all-contact null 中位（仅参考）；浅 rust 竖带=within-shaft **cluster 显著区间**；蓝三角=within-shaft **maxT 显著窗**；0 s 灰虚线；图例左下不遮数据。
- **读法（三条承重边界）**：观测 rust 是否**离开蓝色 within-shaft null 带**并成 cluster。⚠️(1) **高于 all-contact ≠ 高于 within-shaft**——E548 all-contact pointwise 64/66 但 within-shaft cluster 0（相似几乎全是杆几何）；(2) **within-shaft null 分辨力依赖每根杆触点数**（见 `summary.shaft_structure` / index `n_singleton_shafts`）——单触点杆多则 within-shaft 偏弱、两 null 可能非严格嵌套（如 E1150 3/4 杆单触点）；(3) **maxT 很严苛**（只逐窗强峰过）、**cluster 对持续抬升敏感**，报告时说清是哪一档。
- **tier**：per-subject 素材，非 formal cohort spatial gate；不写 replay / timing-order / mechanism。

---

## 维护

新增一类反复出现的图，或某类图的画法发生根本改变时，更新本文件对应小节（示范图路径 + viz 机制 + 配色/轴）。
单个被试/单次实验的一次性图不进本文件。
