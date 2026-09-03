# Figure 4：患者约束的局部 E/I 网络与间期传播读出

> 状态：2026-09-03 作者指定的 A–G 布局版本。Panel A 的组合设计已确认；Panel B 暂时空置，等待 data-driven 参数敏感性结果；原 B–F 顺延为 C–G。当前代表病例统一显示为 E10。

## 一句话论证

在患者触点几何和传播模板约束下，冻结的局部 E/I 网络能够产生两种方向相关的间期样读出，但队列闸门与真实几何敏感性限制了该结果的泛化和机制解释。

## Panel 合同

### A｜局部 E/I circuit 与患者特异性底物

左侧机制示意与右侧 patient-specific E/I substrate 拼为一个 panel。虚线框表示具有相同视野尺度的代表性局部邻域，不表示从患者空间中恢复出的解剖核心。左侧 scale bar 为 0.5 mm；右侧坐标范围为 −10 至 10 mm。触点周围的低透明度深绿色区域表示 virtual-contact firing-density Gaussian sampling footprint（σ = 0.25 mm），不是解剖边界。右侧不显示 anisotropic E→E corridor 或 possible-core 覆盖层。

### B｜预留：data-driven 参数敏感性

该 panel 暂时空置，后续用于展示不同 data-driven 参数对患者间期事件复现的影响。当前拼板只保留 B 角标，不加入占位文字、示意数据或未经确认的结果。

### C｜Node field 与模型空间模式

左侧展示冻结 data-driven Node field，右侧展示 12 张网络 pooled clean events 的 Model TA 与 Model TB 空间模式。Node-field 色条与空间图分离；模型图使用 x (mm) 和 y (mm) 坐标。MTA/MTB 是 development-case 模型模式，不是恢复出的离散解剖核心。

### D｜模型与患者 rank profiles

该 panel 比较模型 MTA/MTB 与患者 TA/TB 的 mean-rank profiles。患者模板使用 all-event Timing+Space 标签，模型事件与 KMeans 保持冻结。该图显示同一 development case 中的模板对应，不构成患者盲或队列级模板恢复。

### E｜模型与患者 cross-fit

该 panel 展示模型 MTA/MTB 与患者 TA/TB 的 contact-split Spearman cross-fit matrix。两个命名匹配单元格中的星号分别表示 MTA–TA 与 MTB–TB 相似度高于各自的 within-shaft contact-permutation null；不显示统一 diagonal-margin 检验。

### F｜同网络 virtual-contact readout

该 panel 展示同一冻结网络窗口内的 30–80 Hz virtual-contact firing-density readout。MTA/MTB 阴影按参与触点的 recruitment-onset span 绘制，两侧各增加 12 ms 显示边界。该信号不是 current、LFP 或临床 SEEG。

### G｜队列闸门与 matched null

该 panel 展示 34 名患者的 canonical-layout held-out recovery：34 名患者进入评估，23 名的 held-out loss 低于 matched null，15 名满足 same-network two-mode gate，11 名同时满足两项。右侧为患者内 paired held-out loss；图中显著性只对应连续 loss 的预注册配对检验，11/34 为描述性交集。

## Figure legend

**Figure 4 | A patient-constrained local E/I network generates two interictal-like propagation readouts.** **A,** Local E/I circuit schematic and patient-specific substrate. The dashed callout links scale-matched local fields of view without implying an anatomically recovered core. Dark-green halos show the virtual-contact firing-density sampling footprint. **B,** [Reserved for the effects of data-driven parameter variation on reproducing patient interictal events.] **C,** Frozen data-driven Node field and pooled Model TA and Model TB spatial patterns. **D,** Mean recruitment-rank profiles for model MTA/MTB and patient TA/TB templates defined using all-event Timing+Space labels. **E,** Equal-network contact-split cross-fit between the frozen model and patient templates. **F,** Representative 30–80 Hz virtual-contact firing-density readout from one frozen network window; shading denotes recruitment-onset spans. **G,** Cohort gates and paired canonical-layout held-out loss relative to the matched within-shaft null in 34 patients. Panels A and C–G support development-case model-to-readout correspondence and a weak canonical-layout cohort advantage, but do not establish anatomical-core recovery, patient causal connectivity or patient-blind real-geometry generalization. Panel B remains a layout placeholder and contributes no evidence in the current version.

## Supplementary migration

原主图 masked-rank KMeans heatmap/rank distribution 已移至 Supplementary Fig. 7E。cross-fit similarity matrix 保留在主图 Fig4-E，并继续显示两项命名匹配相似度各自相对于 within-shaft contact-permutation null 的检验。

## 输出与复现

- 当前完整拼板：`results/paper-ready-figure/fig4/figures/fig4-complete-layout.{png,pdf}`
- 独立 panel：`results/paper-ready-figure/fig4/figures/fig4-panela.{png,pdf}` 与 `fig4-panel{c..g}.{png,pdf}`；B 当前无独立文件
- 主图 producer：`scripts/paper_figures/build_main_figure_4.py`
- Supplementary Fig. 7 producer：`scripts/paper_figures/build_supplementary_figure_7.py`
- 旧草稿归档：`docs/archive/topic4/figure4_subject_specific_snn_pre_2026-09-03.md`

## 当前解释边界

- 允许：冻结的患者约束底物可产生并读出两种间期样传播模式；模型与患者模板在 E10 development case 中具有方向相关的 rank 对应。
- 有限支持：34 人 canonical-layout held-out loss 相对 matched null 存在弱优势，但 same-network K = 2 与 observation-layout sensitivity 闸门未全部通过。
- 不允许：将图解释为恢复了解剖核心、证明了患者因果连接，或实现了患者盲、真实几何条件下的泛化。
