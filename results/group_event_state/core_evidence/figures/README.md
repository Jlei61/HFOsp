# Group-Event State core evidence

### group_event_state_h1_future_blocks.png

这张图回答 H1：在显式多尺度历史 H 已经进入每个模型之后，动态状态 S 是否还对未来 5、30、120 分钟的事件块提供增量。B 比较 H+S 与 H，C 比较正确时刻与 block-shifted S，D 比较动态 S 与 TRAIN 均值 S；纵轴均为正值支持 residual state。

**关注点**：只绘制事前资格合格的患者；不同 horizon 不连线，n=1 不画 cohort median，n=0 留空。当前 positive-recovery power 尚未定标，因此人体数值只能说明这版 count-trained representation 的观测表现。

### group_event_state_h2a_repertoire.png

这张图回答 H2a：给定相同或相近的事件开头，H+S_correct 是否胜过 H，并处于 block-shifted state 的有利方向，进而改变事件继续/停止、继续时的招募规模以及具体触点集合。三个统计 panel 共用 y 轴。

**关注点**：状态来自 30 分钟 count 任务并已冻结；grammar 只训练低容量 residual adapter。主图分别显示相对 H 与相对五个 shift 均值的增量；test-best-control 只保留在机器结果中作为敏感性，不再承担主结论。

### group_event_state_h2b_h3_transfer_feedback.png

这张图预先固定跨任务与反馈机制的最终接口。A/B 分别放冻结间期状态对发作风险和发作早期空间场的增量；C 比较 no-feedback、count/rate feedback 与 mark-specific feedback；D 显示不同 IED 类型的有符号状态冲击。当前这些实验尚未运行，因此只显示坐标、对照方向和 not yet run，不填模拟数据。

**关注点**：H2b 必须以 held-out seizure 为分母；H3 必须先控制共同 pre-event state，且冲击允许正负方向。
