### fig5-spatial-zm-ou-tonic-main-v2.png / .pdf / .svg

严格按作者提供的 paper-ready Figure 5 A–D 布局生成的静态候选图。A 是同一条连续轨迹的全局招募与 15 个虚拟触点 tonic-level readout；B 把 q_core/q_mean、M 和 rE 投到 1 mm patient-matched reduced critical manifold，并标出 saddle-node；C 左是按群体率规则选出的低态事件逐神经元首放电次序，C 右是固定 onset 后 100 ms 的活动能量；D 是同一seed、同一 16-cell 弱 probe、同一组 16 个分层随机位点在 200 ms 低态和 600 ms early-runaway 状态的 probe-minus-sham 平均响应。

**关注点**：A–D 没有混入旧 seed 1801 或其他工作点，C/D 来自 seed 1842 的 bit-identical replay 与 exact-resume checkpoint。B 的高支明确是delay-unstable skeleton，不是稳定高固定点；折点率也尚未通过多网格收敛。代表轨迹属于固定参数 3/3 confirmation family；三种子统计只写入 metadata，不替换原 D 的状态微扰语义。该图展示模型中的 tonic global runaway，不要求30–80 Hz 深调制，也不能表述为临床发作或患者机制证明。

### fig5-panel-b-zm-critical-manifold.png / .pdf / .svg

从同一 payload 单独输出、且不带 panel 字母的 Fig.5B 放大版。紫/橙/蓝线分别为 1 mm reduced high/returned/near-silent branches；星号是 generic saddle-node，深蓝实线与浅蓝虚线分别是 seed 1842 的q_core 和 q_mean 轨迹，点颜色编码 20 ms 平滑 rE。

**关注点**：高支是包含真实延迟后线性不稳定的 skeleton；这张图不把reduced fold 写成有限 SNN 的精确 phase-transition threshold。
