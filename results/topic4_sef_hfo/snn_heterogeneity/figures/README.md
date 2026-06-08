# results/topic4_sef_hfo/snn_heterogeneity/figures

放电网络（SNN）异质性病理核机制实验（局部化设计 + 2×2 轴，spec `2026-06-08-sef-hfo-snn-heterogeneity-mechanism-design.md`）。外围标量安静、异质性只放病理核；核做成 2×2（平均门槛 {18,16} × 参差 {宽,窄}）：baseline / matched(纯方差) / mean_only(纯均值) / unmatched(合并)。读出是 firing-density / current-LFP proxy（**不是 LFP**），只验方向顺序。

### grid_overview.png
2×2 网格每格病理核对"核内同时活跃比例"的影响（核 − 基线）。**蓝=matched(纯方差)、橙=mean_only(纯均值)、红=unmatched(合并)**。**hatch 斜纹 = 核戳前自发点着**（其 d 值是状态切换粗指标、**非诱发同步**，review 1B）；黑边 = 戳在核上（非机制证据）。一眼看：**蓝（方差）全小且无 hatch（不点着）；橙/红（mean-down）全 hatch（点着）** = 点不点着由均值定、参差零影响。

### baseline_compare.png
四个代表的整体网络状态（**源空间群体率**，竖虚线 = 戳 @150ms）：matched 与基线几乎重合、单个干净**诱发**事件（戳后 ~185ms）；mean_only / unmatched 在**戳之前** ~55–80ms 就有早峰 = **自点火灶**（标题标 IGNITES pre-kick）。直接展示"mean-down 核戳前自己烧起来"。

### propagation_<tag>.png（恢复传播顺序视图，复用参考图画法）
左 a：神经元面按**事件起始时刻**上色（viridis），**电极触点按到达时刻上色**（传播顺序在电极上也可见），叠病理核虚线 + 戳点 + 长轴箭头。中 b：∥ 杆 per-contact 峰值依次扫（**斜的峰轨迹 = 读出方向**）；右 c：⊥ 杆峰值对齐（竖直 = 无方向）。tag = mid_matched（最干净诱发波）/ mid_mean_only / mid_unmatched。

**关注点**：b 面板峰轨迹斜率 = 传播方向；matched 是干净诱发波，自点火核的 onset 图被戳前早爆复杂化。

### heterogeneity_<tag>.png（异质性地图，**不同 colorbar** plasma）
左 a：神经元面按**邻域门槛离散度**上色（plasma，与 onset viridis 区分）+ 病理核虚线轮廓。展示"引入的异质性是什么、核在哪、多大"。**注意**：mean-shifted 核（mean_only/unmatched）边界有一圈高离散度，是核内外**平均门槛跳变**（16 vs 18）的伪环，不是核内真实参差——看**核内部**。

**关注点**：异质性的空间位置/范围；边界环是均值跳变伪迹。

### spontaneous_probe.png（B 侧旁支 = tail-driven nucleation，独立机制）
不戳：均匀标量安静（2 Hz）；宽参差铺满整片自发爆发（328 Hz @65ms）。**宽**门槛分布的低门槛尾细胞自点着造有限尺寸成核点。**方向纪律**：这是"变宽也能造起火点"，跟主线"变窄危险"不是同一命题，**不并进**"失去异质性"链。放电版看得见、率均值场看不见。

**关注点**：宽尾 → 自发成核（独立机制），是局部化设计的动机。
