# 当前 Figure 5：空间 SNN 传播与偏 TA 单 core 对照

**本次显示调整：** F 改为显著性符号：TA 不标注显著性，TB 保留 *；按未舍入的 Holm 校正 P 判断，* P<0.05、** P<0.01、*** P<0.001，不显著时不标文字或括号。精确 P=0.050537/0.018677 仍保留于统计文件和图注。F 横轴刻度标签 TA–MTA、TB–MTB 分别居中对齐两组点，右上方使用“空心点 1 core／实心点 2 cores”图例；显著性星号及括号下移，与图例留开间距；其余数据、统计和 A–E 均不变。

版本 `template_ticks_20261027`，状态 `PENDING_AUTHOR_VISUAL_REVIEW`；原 Figure 5 已登记为 [Figure 6](current_figure6.md)。

- [完整 PDF](../results/paper-ready-figure/fig5/figures/fig5-complete-layout.pdf) · [PNG](../results/paper-ready-figure/fig5/figures/fig5-complete-layout.png)
- [各图说明](../results/paper-ready-figure/fig5/figures/README.md) · [版本入口](../results/paper-ready-figure/fig5/current_version.json)
- [画点数据](../results/paper-ready-figure/fig5/candidates/template_ticks_20261027/core_count_subjects.csv) · [统计](../results/paper-ready-figure/fig5/candidates/template_ticks_20261027/core_count_statistics.json) · [选点代价](../results/paper-ready-figure/fig5/candidates/template_ticks_20261027/selection_costs.csv) · [核验](../results/paper-ready-figure/fig5/candidates/template_ticks_20261027/layout_validation.json)

| Panel | 内容 |
|---|---|
| A | xy_left_20 两核边界与 EE 轴示意 |
| B | 旧正式 Fig4F 连续 30–80 Hz 活动 |
| C | 旧正式 Fig4G 平均 rank |
| D | 同一工作点 MTA/MTB，共同 0/16/32/64/96 ms 五帧 |
| E | 旧正式 Fig4H 触点匹配矩阵 |
| F | shared axis 患者的偏 TA 单 core / 无约束双 core，真实 TA–MTA 与 TB–MTB |

**2026-10-27 作者修订：** 保留患者真实 TA/TB 身份，单 core 在两项有符号相关均有效且 ρTA>ρTB 的候选中选择原训练损失 J 最低的工作点（并列时取更早 epoch）；双 core 保持原无约束最低 J 工作点。每位患者每组的 TA–MTA、TB–MTB 来自同一个工作点，不互换模板名，不取绝对相关，不增加 TB 上限或其他相关阈值。

继续使用 2026-10-27 09:42 冻结快照，每人每组前 16/32 次搜索；同一 17 位 shared axis 患者均有合格单 core 候选。7 位单 core 换点，10 位不变，全部双 core 不变。TA 中位数：单 core 0.725、双 core 0.557；TB：单 core −0.150、双 core 0.282，13/17 位患者的 TB 在双 core 中更高。

统计单位为患者，采用配对双侧 Wilcoxon 符号秩检验；差值舍入至 12 位小数、零差值不计秩，以平均秩的精确符号置换分布计算 P，跨本版 TA、TB 两项作 Holm 校正。TA：原始 P=0.050537，校正 P=0.050537；TB：原始 P=0.009338，校正 P=0.018677。此处 P 值已重新计算，不沿用原无约束 TB 的 P=0.380 或 Stronger/Weaker 的 P=0.033；这是另一次探索性分析的两项校正，并非覆盖此前所有分析及多次中期查看的全局校正。

本次是在查看中期结果后提出的非对称选点策略对照，模板相关同时参与单 core 的入选条件和评分；TB 相对较弱部分由约束决定。因此 P 值描述的是这两种选点策略的配对差异，不是独立于选点的纯 core 数量检验，也不能证明单 core 无法同时匹配 TA/TB。新工作点尚未使用独立噪声复测，原工作点的复测不能代替新选点。

约束带来训练拟合代价：例如 E1（epilepsiae_1096）的 J 从 2.885 升至 13.862；原单 core 点的 TA/TB 相关为 0.643/0.893，新点为 0.929/−0.607。故不能把被排除的原点描述为不能拟合两种模板。逐患者选点与 J 变化保留在 selection_costs.csv；无合格候选时保留该患者并将单 core 读出记为缺失，不因模型结果排除患者。

患者纳入标准保持 all-event Timing+Space 记录状态 ok、二维几何支持、保存 shared_a/shared_b 场；不加严格稳定性或方向符号筛选。18 位符合者与原 25 人模型队列交集为 17 位，另 1 位无本批模型结果；不按拟合结果筛人。作者已确认统一 shared axis 标准、先更新 Fig5；Fig3 实际仍含自身平面回退及非共轴患者，本次未修改 Fig3，也不声称两图使用相同名单。

F 空心/实心点分别为单/双 core，细线连接同一患者，横线及竖段为中位数和四分位区间；TA 红色、TB 蓝色，横轴刻度标签为居中的 TA–MTA、TB–MTB，右上方图例为空心点 1 core、实心点 2 cores，无额外轴标题。样本量和中期状态只写于说明及元数据，图内不标 n=17 或 Interim。

作者已明确选择 A/D 与 B/C/E 沿用同一 xy_left_20 示例工作点，不改用后续 J_joint 最低点。A–E 与上一版保持一致：B/C/E 复用冻结数据和原生绘图函数，波形数值、事件阴影、rank 曲线和匹配矩阵不变；D 为同一原生轨迹的共同经过时间五帧。F 按本次要求改为保留 TA/TB 的约束单 core 对照。

2026-10-26 修订：D 上下两行均为 0/16/32/64/96 ms，各自起始仍是全场活动首次达到各自峰值的 1%。按作者要求移除 128 ms 末帧，补入 16 ms 的早期局部活动；帧间隔不等、逐帧标明，不拉伸任一行的时间。16 ms 主要展示各自起始 core 附近的活动，不要求全部活动严格位于核圆内；最后一帧不再表示完全消退。各行分别除以完整冻结事件时间窗内的时空最大值（MTA 82、MTB 87），线性 0–1 色标，标题为 Norm. Firing rate，不逐帧拉满，因此保留相对强弱随时间变化；两行颜色不用于比较绝对发放幅度。绝对活动单位标签已移除，空间标尺仍为 5 mm。

正文、轴 label、刻度、触点名、图例及帧内时间统一为 11 pt，面板字母为 15 pt。A 的 E→E axis 文字位于红色椭圆正下方。上下两排绘图区仍均高 50 mm，B 宽 133 mm，C/F 同宽 50 mm；D 为五列 22.5 mm 正方帧，上下逐列对齐，排间距 5 mm，无额外时间轴、D 标题和灰色实验进度说明。历史候选保留。

原后台实验仍为 25 人、每组 32 条 60 s 仿真及各原提名工作点两条新噪声复测，共 1600 条训练和 100 条原点复测。原训练目标不变；本次改变的是在已有搜索候选中选择单 core 工作点的规则。固定每核大小，不能分离 core 数量与总兴奋面积；固定模型 K=2 分型也不能单独证明存在两种自然传播模式。

[原实验合同](archive/topic4/cohort_core_count_2026-10-24.md)保持为历史依据，本页明确记录本次作者追加的单 core 选点修订。完成后 `scripts/paper_figures/finalize_fig5_core_count.py --watch` 和 `fig5/source/template_ticks_completion_pipeline_20261027/` 在完整 32 次搜索内沿用相同约束与 TA/TB 两项 Holm，输出 `ta_preferred_search_complete_template_ticks_20261027` 候选；新选点仍需新噪声复测，cohort_comparison_complete 保持 false，不自动覆盖当前入口。原无约束、Stronger/Weaker 候选均保留；并行 Fig4、Fig3 和冻结论文未修改。
