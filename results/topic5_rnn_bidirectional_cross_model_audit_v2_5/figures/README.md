### cohort_bidirectional_static_transfer_summary.png

四个 panel 依次展示同一冻结方向在两侧 source 事件中的位移、该方向相对候选方向的特异性、共同 11 人中经验分布/普通 GRU/结构化 RNN 的发作早期静态相似度，以及主要模型和对照相对全通道打乱 null 的 margin。两侧位移为正不等于方向特异；第二个 panel 专门检验这一替代解释。

**关注点**：普通和结构化模型都保留静态 contact scaffold，但结构化轴和 source 项没有显示独立增量。

### representative_rank_distribution_comparison.png

三名预先固定患者逐行展示真实 held-out 间期 rank distribution、普通 full-history GRU、结构化 RNN，以及 strict clinical-onset 后 `[0,10] s` 的 1–150 Hz 静态能量。每行 contact ordering 只由真实间期平均 rank 冻结，不读取发作 target 排序。

**关注点**：比较模型是否复现每个 contact 的完整 rank 分布，并观察这种分布与发作早期静态能量的形态关系；该图不是逐触点 replay 证据。
