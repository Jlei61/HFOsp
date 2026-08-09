# Fig. 4 data-driven core-field rev8.1

### fig4a_data_driven_core_field_waveforms

这张图使用最终冻结候选的同一代表网络：左侧显示学得的病理场，中间显示由全体最终事件 KMeans 后选出的两个模式代表传播，右侧显示未经模板平均的 30–80 Hz virtual-SEEG 直接波形。阴影只表示无监督模式身份，不预先当作 forward/reverse 标签。

**关注点**：两个模式是否都在同一网络中出现、空间传播是否不同，以及直接电极波形是否支持而不是掩盖 KMeans 结论。

### fig4b_data_driven_core_field_kmeans

这张图使用与 Fig4A 完全相同的最终 unseen-network 事件池。事件先在冻结的 patient-training embedding 中独立做 KMeans=2，再与两个病人训练模式做 Hungarian 匹配；最右矩阵是模型模式与病人模式的 Spearman 一致性，红框表示至少一项正式门未通过。

**关注点**：两簇是否都有足够事件、两条逐触点 profile 是否真正不同，以及 2×2 矩阵是否呈现正对角、负交叉并达到 rigid-control 基准。
