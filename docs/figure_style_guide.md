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

---

## Topic 4 · 机制模型（SEF-HFO / cm-SNN）

- **示范图**：[`results/paper-ready-figure/fig5_core_model_s3_brakeoff/figures/core_model_s3_brakeoff.png`](../results/paper-ready-figure/fig5_core_model_s3_brakeoff/figures/core_model_s3_brakeoff.png)
- **默认标准（SNN 仿真图都按这个画）**：`mechanism + tempA source + tempB source + electrode readout`。除非用户明确要求做诊断图、参数扫描图或 pipeline/KMeans 结果图，任何 SNN 相关主图 / paper-ready 图都不得回到旧的三行 Forward/Reverse/C 行堆叠布局。
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
- **被试特异性变体（subject-specific SNN，Fig4A/B 起）**：Fig4A 用同一四列标准，但底物按**病人真实电极布局**摆放（示范 `results/paper-ready-figure/fig_subject_snn_epilepsiae_1146/`，脚本 `scripts/paper_figures/plot_fig_subject_snn.py`）。Fig4B 是同一 readout 的 KMeans=2 核验图，脚本 `scripts/paper_figures/plot_fig_subject_snn_kmeans2.py`。锁定约定：
  - 两个低阈值核 = **两类间期模板各自最早的 k 个电极**（`template_source` 摆放，`src/sef_hfo_subject_placement.py`），即两类模板的 source 区，落在传播轴两端；不要用 swap decision_k 的宽 strip 质心（会被拉向中间）。
  - 用**真实几何 plane-fit**（核间距自然 ≈ blessed sep0.7），不得人为 core-anchor 到固定间距。
  - mechanism panel 必须显示**核与电极间期最早区 overlap**（高亮核成员触点）+ E→E 长轴带。
  - readout 用 `k_dir=2`（病人电极比模型密杆稀疏的放宽，**载重参数**，必须在 metadata/README 注明 k_dir=3 的退化情况）。
  - 诚实口径：readout 若用 spontaneous twoend，需注明自发双向**与 seed 有关**；separate-then-pool 只能写"仪器对齐"不能写"自发机制"。
  - **LOCKED 模式（2026-06-26）= 每个 subject-SNN 案例固定出两张主图：Fig4A（readout 四列）+ Fig4B（KMeans 核验四块）。** Fig4C（real-vs-model profile）、Fig4D（组合 S 置换 null）是可选 supplement。
  - **Fig4B 四块（左→右）**：`clustered event heatmap | per-channel rank distribution | cluster rank distributions | model-vs-real 2×2 相似性矩阵`。最左 heatmap **占主导宽度**，后三块紧凑。
    - **左三块 y 轴必须 channel-for-channel 对齐**（同一通道在三块里同一高度，`sharey`）。heatmap 用 canonical `_plot_rank_heatmap` + `_plot_cluster_boundaries`；rank distribution + cluster profile **沿用 canonical 视觉风格但画在 heatmap 的 1-unit 通道坐标上**（`_hist_aligned` / `_cluster_aligned`，**不用** canonical 的 ridge `_plot_rank_histogram`——它 0.15 spacing 无法与 heatmap 对齐）。heatmap rank colorbar 横放 x-label 下方。
    - 第四块 = 模型(fwd/rev)×真实(t_a/t_b) Spearman 矩阵，**只用 star 显示方向性 channel-shuffle 置换 p**（不写数值），颜色=ρ、`aspect=equal`（cells 方形不压扁）。
    - 只消费 Fig4A 同一 readout 的 clean directional events，不重跑仿真；标题/README 报告 cluster-size、direction purity、within-cluster tau、shared-overlap corr + 模型↔真实模板一致性，不是 cohort 统计。

---

## Topic 5 · 亚型 / ictal 回响（暂不锁定）

Topic 5 仍在探索期，canonical 图型待结论稳定后再补。现有候选见 `results/FIGURE_INDEX.md` 的 Topic 5 段
（`topic5_ictal_template_echo/`、`topic1_topic5_bridge/` 等），暂按个案处理，不强制统一布局。

---

## 维护

新增一类反复出现的图，或某类图的画法发生根本改变时，更新本文件对应小节（示范图路径 + viz 机制 + 配色/轴）。
单个被试/单次实验的一次性图不进本文件。
