# Current MZ-conductance dynamics

定位：**M4/MZ conductance 阶段验收机制图**，不是最终 Figure 4/5，也不是可恢复 seizure-state 的证据。数值来源、窗口与选择规则锁在同目录 metadata；producer 为 `scripts/paper_figures/plot_fig_mz_conductance_current_dynamics.py`。

### mz_conductance_current_dynamics.png

这张图使用同一条 L=20、seed 1 自发轨迹：左侧给出当前 conductance + local-Z + protected additive-global GABA 结构；中间分别显示一个按固定规则选出的 returning event 和 terminal early-runaway 的空间招募顺序；右侧把群体放电率、core Z 消耗以及同一真实 E1146 montage 的 15 触点 readout 放在连续时间轴上。returning event 的 onset-axis Spearman 为 0.96，early runaway 降为 -0.12；该轨迹在 7180.1 ms 进入 runaway，图中没有把它标成可恢复发作态。

**关注点**：先看 returning event 是否仍有局部时序结构，再看 runaway 是否表现为空间招募扩大；最后核对 Z 阶梯只把系统推向 runaway，而没有生成高活动后的回落段。
