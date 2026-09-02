# Group-Event State core evidence

### group_event_state_h1_future_blocks.png

这张图回答 H1：在显式多尺度历史 H 已经进入每个模型之后，动态状态 S 是否还对未来 5、30、120 分钟的事件块提供增量。B 比较 H+S 与 H，C 比较正确时刻与 block-shifted S，D 比较动态 S 与 TRAIN 均值 S；纵轴均为正值支持 residual state。

**关注点**：实心点为事前数据资格合格，空心点仅为 development 诊断；中位数只汇总实心点。当前阳性合成定标未通过，因此图中人体数值只能说明这版模型的观测表现，不能作为慢状态成立或不存在的证据。

### group_event_state_h2a_repertoire.png

这张图回答 H2a：给定相同或相近的事件开头，H+S_correct 是否同时胜过 H、H+S_shifted 和 H+S_mean，进而改变事件继续/停止、继续时的招募规模以及具体触点集合。三个统计 panel 共用 y 轴。

**关注点**：状态来自 30 分钟 count 任务并已冻结；grammar 只训练低容量 residual adapter。当前阳性合成定标未通过，且任何选在训练预算末端的 arm 都必须标为优化未收口。

### group_event_state_h2b_h3_transfer_feedback.png

这张图预先固定跨任务与反馈机制的最终接口。A/B 分别放冻结间期状态对发作风险和发作早期空间场的增量；C 比较 no-feedback、count/rate feedback 与 mark-specific feedback；D 显示不同 IED 类型的有符号状态冲击。当前这些实验尚未运行，因此只显示坐标、对照方向和 not yet run，不填模拟数据。

**关注点**：H2b 必须以 held-out seizure 为分母；H3 必须先控制共同 pre-event state，且冲击允许正负方向。
