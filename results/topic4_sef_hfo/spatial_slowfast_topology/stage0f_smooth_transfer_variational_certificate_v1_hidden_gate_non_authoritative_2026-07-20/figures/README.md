### stage0f_smooth_transfer_variational_certificate.png

这张诊断图展示锁定的两个参数点在平滑 exact-table transfer 下重建的单周期轨道、chain-rule 与 centered-RHS 两套离散变分导数得到的谱半径、全部横向乘子，以及 spline 对 direct exact Siegert 值和导数的误差。它只修复 Stage 0E 的导数证书缺口，不是新的参数扫描，也不是空间发作图。

**关注点**：先看 exact transfer parity 和 smooth-vs-LUT orbit parity，再看两套导数及 dt/2 是否给出一致且远离单位圆的乘子；任一前置门失败都不能称为稳定轨道证书。

### stage0f_smooth_transfer_variational_certificate.pdf

与 PNG 内容相同的矢量版本，便于放大检查靠近原点的乘子和两种导数之间的细小差异。

**关注点**：即使证书通过，Stage 1 和空间模拟仍保持关闭；该结果只说明 frozen homogeneous fast system 的局部横向稳定性。
