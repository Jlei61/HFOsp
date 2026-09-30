# Core空间群体动力学：第三阶段，保留接收端结构

本轮完成两条12秒自主轨迹：保留每个接收神经元的原连接与延迟，只平均发送端群体；随后把源格宽再减半。全部原输入、阈值、几何及SEEG检测规则固定。**两条候选仍不接受，不进入分岔分析。**

保留接收端结构后，二维招募面积与启动跨度接近原生范围。源格进一步细化使ICL10参与由81.25%改善到96.67%，但SCL9仍为93.33%，原生三噪声种子仅40.7%–64.3%。同种子参与误差从0.05317降至0.03782，仍超过原生两两最大差异0.03280；后者仅为描述性参照。网格细化没有让所有场指标单调接近，尚未建立完整时空对应。

固定检测窗的审计定位到局部包络分布错误：原生SCL9有大量弱事件，候选明显过度招募。另外，同一细源轨迹用粗群体SEEG读出会把两个不满足持续时间条件的SCL9事件判为参与。传播方程和观测方程需分别验证。模型仍保留40,000个细胞状态，是定位闭合误差的中间模型，尚非低维rate方程。

- [完整报告、数值表和8张主要诊断图](../../../results/topic4_sef_hfo/spatial_source_projection_20260916/scientific_report.md)
- [实际方程和原SNN映射](../../../results/topic4_sef_hfo/spatial_source_projection_20260916/equations.md)
- [三项传播观测](../../../results/topic4_sef_hfo/spatial_source_projection_20260916/figures/refined_sources/03_three_propagation_observables.png)
- [二维传播及SEEG包络](../../../results/topic4_sef_hfo/spatial_source_projection_20260916/figures/refined_sources/04_spatial_b_leads.png)
- [局部包络幅度审计](../../../results/topic4_sef_hfo/spatial_source_projection_20260916/figures/refined_sources/06_contact_amplitude_audit.png)
- [执行和验收状态](../../../results/topic4_sef_hfo/spatial_source_projection_20260916/delivery_status.json)
- 代码：`scripts/topic4_spatial_source_projection/`。

原图投影、独立小网络动态、单细胞源极限与分段延续均已核验；真实网络前2秒CPU/GPU的spike、二维场和群体状态完全一致。两条GPU轨迹均完成12秒，CPU参照在2秒核验后停止，不计第三条完整科学轨迹。8张主要图已由Agent目视检查，尚未经用户人工验收。

下一项有区分力的对照为：固定当前粗源网格和每步群体spike数，无放回抽样原发送身份，再沿原边与延迟传递；它保留当前均值而恢复由原图决定的有限群体波动。**这项对照尚未实现或运行**，且逐步身份重采样不保持原发送身份的不应期历史。完整条件与不同结果的解释见报告；不继承旧六群体模型的LP/PD。
