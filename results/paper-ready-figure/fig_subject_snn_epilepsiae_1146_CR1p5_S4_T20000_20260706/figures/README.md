# fig_subject_snn_epilepsiae_1146_CR1p5_S4_T20000_20260706

### fig_subject_snn_epilepsiae_1146_CR1p5_S4_T20000_20260706.png / .pdf

E1146 小核长时重跑版 Fig4A。输入 readout tag 为 `epilepsiae_1146_tsrc_cr1p5_s4_T20000_20260706`，同一 template-source / plane-fit 布局，`core_r=1.5`，seed=4，仿真时长从 8000 ms 拉到 20000 ms。

机制面板中的淡橙色区域是按两灶连线方向旋转的 E->E 长轴椭圆，不再使用矩形 band 或方向箭头。右侧 readout 面板显示完整 20 秒窗口，clean directional events 为 5 forward / 11 reverse。长时累积增加了事件数，但也让 readout 更偏 reverse，不能写成比短时 seed4 更平衡。

**关注点**：看同一小核布局在更长时间里是否持续产生两类事件。结论是双向仍存在，但比例偏 reverse。

### fig_subject_snn_epilepsiae_1146_CR1p5_S4_T20000_20260706_kmeans2.png / .pdf

同一长时 readout 的 active-contact KMeans k=2 图。KMeans 使用 16 个 clean directional events；图面只保留参与率 >=30% 的 8 个 active contacts，以减少低参与触点造成的灰格。

KMeans 得到 `t_a n=8`、`t_b n=8`，shared-overlap corr=-0.939，model-vs-real 2x2 矩阵保持有效。方向 purity=0.8125：`t_b` 簇全 reverse，但 `t_a` 簇里混入 3 个 reverse，因此它是长时稳定聚类核验，不是完美方向标签复现。

**关注点**：这张图是推荐的长时低灰格 KMeans 图。写作时要同时说明 active-contact display filter 和 direction purity caveat。
