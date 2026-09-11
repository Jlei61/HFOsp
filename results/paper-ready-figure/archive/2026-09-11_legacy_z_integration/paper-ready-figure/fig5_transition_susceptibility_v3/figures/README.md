### fig5-transition-susceptibility-v3.png
A/B 是同一条 two-core、连续 OU、Z/M 开启的 40,000-cell 轨迹，保留间期事件、转变前和 early runaway。A 是未经带通的电流型 LFP proxy，8 个固定触点；B 的 Z/M 为双 core 合并均值，不能解释为两核始终相等。C 修复旧版 y/x 误置后展示同轨迹的空间招募；D 使用同 16 个位置、同 16-cell 脉冲与精确 sham，比的是刺激位置的额外后继放电，不是自然活动热图。E/F 使用 realized delays 和 0.1 ms 步长，展示 10 s 有限初值响应，不能把区域边界称为已定型的局部分叉。

**关注点**：D 的 pre-onset 在 2015.4 ms，整个 200 ms 响应窗都早于 onset；E/F 同时保留未定与初值依赖。此版继承 Joint=1.25，不是 rev22 新最优点，尚未作者接受。

### fig5-transition-susceptibility-diagnostics.png
左图展示 s=0.428 状态完整传到 0.429 后的群体率。右图展示 pre-onset 扰动减 sham 的逐时间响应，每条浅线是一个刺激位置。

**关注点**：位置是同一网络内重复测量，不作为独立患者或多 seed 检验；短窗响应不能替代长时转变概率。

### fig5-perturbation-response-fields.png
每行比较相同窗口中 low 与 pre-onset 的 probe-minus-sham 响应场，分别为 0–50 ms 与 0–200 ms；每张图平均全部 16 个刺激位置。每个格点是当地 E 神经元的额外后继放电数，同一行共用 signed 线性色标。

**关注点**：这里画响应放电发生在哪里；主图 D 画从哪里施加刺激更易产生响应，两者不能混为一谈。负值和所有位置均被保留。
