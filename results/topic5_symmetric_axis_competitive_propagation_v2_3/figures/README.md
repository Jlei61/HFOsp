### competitive_propagation_rnn_formal.png

A 给出唯一允许的对称 scaffold、source 与 propagation/competition 状态；B–D 依次检验模型是否可预测、历史状态是否必要，以及 axis/source 项是否提供匹配增益。E 比较可解释模型恢复了多少 empirical ordered-history Markov 信号；F 仅在 A–C 与 matched-axis safeguard 通过时显示状态参数，否则如实显示停止门。

**关注点**：所有点均为 patient-first heldout20 结果；模型不读取 A/B、SOZ 或发作期 target，图中正向 benefit 表示前一个模型的 NLL 更低。
