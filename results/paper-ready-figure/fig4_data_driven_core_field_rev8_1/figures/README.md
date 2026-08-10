# Fig. 4 data-driven core-field rev8.1

### fig4a_data_driven_core_field_waveforms

这张图使用最终冻结候选的同一代表网络：左侧同时显示优化得到的 h envelope 和神经元实际承受的 signed Delta Vtheta；中间的代表传播事件叠加全部 50 个 final events 的 event-equal earliest-activation density 与 h contours；右侧是未经模板平均、所有 contact 共用一个幅度标尺的 30–80 Hz model-current readout。阴影只表示无监督模式身份，不预先当作 forward/reverse 标签。

**关注点**：两个模式是否都在同一网络中出现、空间传播是否不同，以及直接电极波形是否支持而不是掩盖 KMeans 结论。

### fig4b_data_driven_core_field_kmeans

这张图使用与 Fig4A 完全相同的最终 unseen-network 事件池。模型 profile 用实线，冻结 patient-training prototype 用虚线，浅色带表示 recording-block 之间的 patient profile 变异；2x2 matrix 给出条件于冻结 KMeans 标签的 network-to-event / block-to-event hierarchical bootstrap 95% CI。最右 benchmark 同时展示 global curve distance 与 worst-mode correlation；control 的纵轴仍是描述性点估计。

**关注点**：两簇是否都有足够事件、两条逐触点 profile 是否真正不同，以及 2×2 矩阵是否呈现正对角、负交叉并达到 rigid-control 基准。
