### stage0e_poincare_floquet_audit.png

这张诊断图按同一锁定尺度展示两个固定参数点的高分辨率轨迹、Poincaré shooting 残差、全部八个横向 Floquet 乘子，以及 fast/pool 非共线扰动的逐圈回归距离。若某一后续 panel 留空，表示该参数点在 cheap-first 的更早数值门已经停止，不能用频谱结果补救。

**关注点**：只有 shooting 闭合、三档 epsilon 与 dt/2 均稳健、全部横向乘子远离单位圆边界，并且两类扰动都逐圈回归时，才能称为稳定周期轨道。

### stage0e_poincare_floquet_audit.pdf

与 PNG 内容相同的矢量版本，供文档审阅和局部放大检查复乘子位置。

**关注点**：单位圆内并不自动等于通过；还必须满足预注册的数值不确定性裕量。
