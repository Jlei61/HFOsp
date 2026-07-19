# fig_mz_early_bridge_v2_zm_tau500 — 图说明（中文）

本目录是 MZ early-field bridge **V2（z+m，τ_adp=500 ms）** 的 paper-ready 图。它和已冻结的 V1（只有去抑制 z）用**完全相同的图语**，只把中间那段自然轨迹从 z-only 换成 z+m，用来看"加了快速适应变量 m 之后，间期时序轴还能不能预测失控前的早期能量场"。

### fig_mz_early_bridge_v2_zm_tau500.png

上排是一条**连续的 z+m native Virtual-SEEG 轨迹**（seed1，非拼接）：橙色是 ICL 杆的 11 个触点、青色是 SCL 杆的 4 个触点。蓝窗是按固定规则选出的**一个间期样返回事件**（规则：t_recruit 之前、多数 slow-off 方向里最后一个合格事件，选取不看目标能量）；粉窗是 `t_recruit` 后 0–50 ms 的 **pre-t120 早期能量窗**；红虚线是 operational-runaway `t120`（此图显示时间标注为 1106 ms，对应绝对时间约 12956 ms）。下排只有两张与上方窗一一对应的 contact 场：左 = 蓝窗事件的**触点发放次序**（viridis，1 早→晚），右 = 粉窗的 **pre-t120 早期能量**（Blues）。两张场沿 E1146 长轴**同向抬升**（长轴右侧既是事件里更早、又是失控前能量更高），这就是"同一支架、状态依赖读出"的桥在 z+m 下仍然成立的直观证据。灰点只表示固定 E-neuron 几何背景，**不表示局部招募**；operational runaway 是模型代号、不是临床发作，virtual-LFP 30–80 Hz 能量不是临床宽带功率。正式统计仍来自 slow-off held-out 双向模板的 maxAB（不是图里这一例的描述性 earliness）。

**关注点**：下排两张场的长轴梯度是否同向（间期时序轴是否预测 pre-runaway 能量），以及与 V1 z-only 主图对照——z+m 把 `t120` 从约 9.3 s 推后到约 13.0 s，但这条轴是否仍被保留。
