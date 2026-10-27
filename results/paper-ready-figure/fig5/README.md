# Figure 5：偏 TA 的单 core 与双 core 模板匹配

**本次显示调整：** F 改为显著性符号：TA 不标注显著性，TB 保留 *；按未舍入的 Holm 校正 P 判断，* P<0.05、** P<0.01、*** P<0.001，不显著时不标文字或括号。精确 P=0.050537/0.018677 仍保留于统计文件和图注。F 横轴刻度标签 TA–MTA、TB–MTB 分别居中对齐两组点，右上方竖排显示“空心点 1 core／实心点 2 cores”图例；显著性星号及括号下移，与图例留开间距；其余数据、统计和 A–E 均不变。

[完整 PDF](figures/fig5-complete-layout.pdf) · [PNG](figures/fig5-complete-layout.png) · [各图说明](figures/README.md)

当前版本 `vertical_legend_20261027`，状态 `PENDING_AUTHOR_VISUAL_REVIEW`；旧 Figure 5 仍登记为 Figure 6。

保留患者真实 TA/TB 身份，单 core 在两项有符号相关均有效且 ρTA>ρTB 的候选中选择原训练损失 J 最低的工作点（并列时取更早 epoch）；双 core 保持原无约束最低 J 工作点。每位患者每组的 TA–MTA、TB–MTB 来自同一个工作点，不互换模板名，不取绝对相关，不增加 TB 上限或其他相关阈值。

继续使用 2026-10-27 09:42 冻结快照，每人每组前 16/32 次搜索；同一 17 位 shared axis 患者均有合格单 core 候选。7 位单 core 换点，10 位不变，全部双 core 不变。TA 中位数：单 core 0.725、双 core 0.557；TB：单 core −0.150、双 core 0.282，13/17 位患者的 TB 在双 core 中更高。

统计单位为患者，采用配对双侧 Wilcoxon 符号秩检验；差值舍入至 12 位小数、零差值不计秩，以平均秩的精确符号置换分布计算 P，跨本版 TA、TB 两项作 Holm 校正。TA：原始 P=0.050537，校正 P=0.050537；TB：原始 P=0.009338，校正 P=0.018677。此处 P 值已重新计算，不沿用原无约束 TB 的 P=0.380 或 Stronger/Weaker 的 P=0.033；这是另一次探索性分析的两项校正，并非覆盖此前所有分析及多次中期查看的全局校正。

本次是在查看中期结果后提出的非对称选点策略对照，模板相关同时参与单 core 的入选条件和评分；TB 相对较弱部分由约束决定。因此 P 值描述的是这两种选点策略的配对差异，不是独立于选点的纯 core 数量检验，也不能证明单 core 无法同时匹配 TA/TB。新工作点尚未使用独立噪声复测，原工作点的复测不能代替新选点。

约束带来训练拟合代价：例如 E1（epilepsiae_1096）的 J 从 2.885 升至 13.862；原单 core 点的 TA/TB 相关为 0.643/0.893，新点为 0.929/−0.607。故不能把被排除的原点描述为不能拟合两种模板。逐患者选点与 J 变化保留在 candidates/vertical_legend_20261027/selection_costs.csv；无合格候选时保留该患者并将单 core 读出记为缺失，不因模型结果排除患者。

当前配对数据、统计、核验及选点审计位于 `candidates/vertical_legend_20261027/`。图内不显示 n=17 或 Interim；两者仍保留于说明及元数据。

校验：`python scripts/paper_figures/build_fig5_current.py`；重绘：`python scripts/paper_figures/build_fig5_aligned_layout.py`。后台完成管线沿用同一约束在完整 32 次预算内重新选点，另存候选，不自动覆盖当前入口，也不把旧工作点复测记作新点验证。
