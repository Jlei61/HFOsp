# Fig2-E 候选：患者特异 TA/TB 间期传播场

### epilepsiae_1146_interictal_AB.png / .pdf

当前 Fig2-E representative-subject 候选。E1146 的 TA/TB 两条 early→late 传播轴宽泛共线且方向相反，因此两幅 rank field 使用同一个冻结 shared plane；横轴正方向由 TA 的 early→late 方向固定。背景与触点颜色均表示间期模板 rank，0 为 early、1 为 late。

该图只读取 `topic5_interictal_template_fields_v1` 冻结 artifact，不使用发作、onset、subtype 或能量数据。`sigma_display=6 mm` 仅控制连续场的显示覆盖，不改变下游 field correlation 使用的患者特异冻结 kernel。transverse 轴符号只由电极几何决定，不按场颜色翻转；colorbar 与 field 坐标框等高。

当前状态是 **paper-ready candidate**，不是已锁定的最终 Fig2-E。E1146 用于延续全文的代表患者叙事；最终是否保留，应与 Fig2 其余 panel 的版面、subject 复用和信息增量一起决定。完整 28 人素材位于 `results/interictal_propagation_masked/template_gradient_fields/figures/`。

固定画图规范为 `docs/topic5_interictal_field_figure_spec.md`。之后所有间期 TA/TB 场图必须复用 `plot_interictal_ab_subject()` / `plot_interictal_ab_atlas()`；需要拼版时复用 `build_interictal_ab_panel_payloads()` / `draw_interictal_rank_field_panel()`，不得复制 renderer 另写一套。

**关注点**：这张图展示患者特异的间期空间传播结构；不要把它解读为发作场一致性、逐点传播重放或 cohort-level 统计证据。
