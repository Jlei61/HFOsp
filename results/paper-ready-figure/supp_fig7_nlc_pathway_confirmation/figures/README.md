### supp_fig7-complete-layout.png / .pdf

**Supplementary Fig. 7 | Frozen local-connectivity ablation and model-event KMeans structure.**

**A,** Mode 1 share among classified formal events, defined as Mode 1/(Mode 1 + Mode 2). **B,** Complementary Mode 2 share. **C,** Balanced alignment between de novo KMeans K = 2 clusters and the frozen Mode 1/2 labels. The natural clusters were not relabelled as patient modes. **D,** Fraction of returned events outside the frozen patient-distribution support (out of distribution, OOD). Across A–D, Node, +EE, +E-to-I and +EE+EI were evaluated for 20 s in each of 12 paired network seeds (1581–1592) under frozen topology, delays and separately conserved incoming-weight budgets. Open circles denote individual networks, grey lines connect the same network seed across arms, filled circles show equal-network means and error bars show 90% network-bootstrap confidence intervals (CIs). Stars indicate that the paired arm-minus-Node 90% network-bootstrap CI excluded zero (4,096 resamples; no multiplicity correction): +E-to-I in A and B, +EE+EI in C, and +E-to-I and +EE+EI in D. **E,** Masked recruitment-rank heatmap for 627 formal clean model events, grouped by frozen MTA and MTB KMeans labels, with aligned per-contact rank distributions and one shared first-to-last color scale. Mode 1/2 and MTA/MTB are frozen model labels and have no independent pathological interpretation. These analyses estimate development-case model-internal pathway effects and event structure; they do not establish patient causal connectivity, anatomical-core recovery or patient-blind/real-geometry generalization.

**关注点**：A–D 的统计单位是 network seed（n = 12），不是事件；E 的 627 列是 pooled model events，不是独立患者或网络。图只支持 development-case 模型内部 pathway effect 与事件结构。

### supp_fig7-panela.png / .pdf

Supplementary Fig. 7A 的无角标独立导出，展示 Mode 1 事件占比；图形元素与统计定义以上方完整图注为准。

**关注点**：完整投稿图请使用带 A–E 角标的 `supp_fig7-complete-layout`。

### supp_fig7-panelb.png / .pdf

Supplementary Fig. 7B 的无角标独立导出，展示与 A 互补的 Mode 2 事件占比；图形元素与统计定义以上方完整图注为准。

**关注点**：Mode 1/2 是冻结分类器标签，不是临床病理亚型。

### supp_fig7-panelc.png / .pdf

Supplementary Fig. 7C 的无角标独立导出，展示 de novo KMeans K = 2 与冻结 Mode 1/2 标签的 balanced match。

**关注点**：自然簇未被重命名为患者模式；联合臂 match 的下降不能写成患者模式几何恢复。

### supp_fig7-paneld.png / .pdf

Supplementary Fig. 7D 的无角标独立导出，展示返回事件的 OOD 比例；图形元素与统计定义以上方完整图注为准。

**关注点**：较低 OOD 不代表真实几何或患者外泛化。

### supp_fig7-panele.png / .pdf

原主图 KMeans panel 的无角标独立导出。热图展示 627 个 formal clean model events 的 masked recruitment ranks，并按冻结 MTA/MTB 标签分组；右侧为对齐的逐触点 rank distribution 与唯一共享色条。

**关注点**：MTA `n=437`、MTB `n=190` 是 pooled model events，不是患者数或独立网络数。
