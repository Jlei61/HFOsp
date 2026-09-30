# Core空间群体动力学：第二阶段探索

用户明确上一轮完全不能验收，要求继续。本轮完成7条自主空间群体粒子轨迹、独立数值验证和6张波形/raster/SEEG/2D传播/三项评分诊断图。**所有候选仍未通过传播对应，不进入分岔。**

同一1mm连接投影下，保留膜电位、reset、不应期和原输入抽样使核心由旧rate约310/340Hz持续高背景恢复到约35Hz平均率、约82%低活动时间及反复burst。这个中间模型仍保留40,000细胞状态，尚非低维rate方程。网格细化改善部分二维招募指标，但恢复原图E/I总入强度或六类来源组成仍未解决触点参与失配：来源组成候选SCL9参与100%，原生三种子41–64%；ICL10候选77%，原生93–100%。三项误差中的参与分量超出所有原生配对差异，不能以平均rank接近或snapshot相似来接受。

- [完整执行报告、定量表与六张图](../../../results/topic4_sef_hfo/spatial_population_dynamics_20260916/scientific_report.md)
- [实际方程及原SNN映射](../../../results/topic4_sef_hfo/spatial_population_dynamics_20260916/equations.md)
- [波形与SEEG](../../../results/topic4_sef_hfo/spatial_population_dynamics_20260916/figures/01_dynamics_and_seeg.png)
- [三项传播观测](../../../results/topic4_sef_hfo/spatial_population_dynamics_20260916/figures/03_propagation_observables.png)
- 代码：`scripts/topic4_spatial_population_dynamics/`。

下一项具体区分检验为仅压缩发送端源群体、保留每个接收细胞原空间和延迟输入分布；尚未运行。所有核验仍需原几何、原输入及同一SEEG observer；通过空间/时间读出后再考虑密度/历史变量降维、临界邻域及扰动验证。旧六群体LP/PD与这条验证链没有继承关系。图已完成Agent自查，用户尚未人工验收。
