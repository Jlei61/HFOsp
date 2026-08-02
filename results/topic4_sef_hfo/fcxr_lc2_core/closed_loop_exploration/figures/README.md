### r1_sensor_characterization.png

这张图把 HEO2 旧参考轨迹拆成两个活动段和中间的 rest-like gap，并同时给出虚拟 SEEG 与局部 recurrent-drive support。它说明旧的完整窗口不能被当成一段连续高态，但活动段仍提供了闭环 H 的候选传感范围。

**关注点**：中间 gap 的低率与 15/15 触点回到 ±3 dB，而 recruited support 只占采样 E 细胞的一部分。

### r1_sensor_pareto.png

这张图展示 baseline false latch、HEO1 持续支持、HEO2 活动段支持和长间隙残留之间的折中。红星是按预注册角色选择的六个候选；这些点只进入闭环筛查，不代表已经存在高态 basin。

**关注点**：不存在一个同时桥接完整长 gap 且对所有间期事件零误锁的单一时间常数。

### h_loop_screen.png

六个 H 传感候选各自扫描平滑宽度与反馈强度。点的颜色是 1 s 开发标签，大小近似尾段活动率；`screen_survivor` 只表示值得延长，不能解释成双稳态或发作 carrier。

**关注点**：90 格里有 52 个 survivor、38 个 saturated-tonic；弱反馈高初值已约 101 Hz，反馈增强后升到 275 Hz，说明这一步主要看见有限限幅，不是选择性迟滞。

### frozen_fork_map.png

这张图并列 healthy low/high 初值、susceptible low/high 初值与两个冻结 X 负荷，并把经验工作点标签、尾段率和代表性 300 ms 平滑轨迹放在一起。两个正式候选的 A-low 都离开间期工作点；C 是有限高态，但 D1/D2 仍停在高态。

**关注点**：H 形成有界高态是正结果；失败在于 Z 没取得选择性 onset control，两档 X 只有 amplitude control、没有 offset state-transition authority。

### failure_taxonomy.png

这张图按真实执行顺序汇总 R1 传感器表征、E3 高初值 screen、E4 frozen geometry 和未解锁的 E5。它把开发性 survivor 与真正的 basin/termination 判据分开，避免把开环分类写成机制终局。

**关注点**：正式口径是 bounded high-state generation positive；susceptibility-selective onset 与所测 LC1 负荷下的 X-controlled offset negative。dynamic Z/H/X lifecycle 没有被测试。
