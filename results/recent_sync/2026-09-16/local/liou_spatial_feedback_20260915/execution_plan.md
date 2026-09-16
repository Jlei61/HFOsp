# Liou 原始全局空间反馈对照

用户2026-09-15要求在当前模型上尝试原文全局抑制的空间反馈。继承历史手放双core、原生Z/M及噪声，不消费另一任务中的新训练底物。

原文空间投射为 `P_I2.W = @(x) sum(x(:))/prod(O.n)`，`WPost=50`；局部投射权重250，对应γ=1/6。兴奋性细胞输出经15ms抑制突触；没有10秒额外招募状态或50Hz开启阈值。本轮复现此均匀全局空间算子、快速突触脉冲响应及γ的相对权重定义；原有显式I局部电路、LIF电流模型和逐细胞Z/M保留，因此不是全套LAS电导/氯模型的原样复现。

## 实际方程

令s_E(t)为全E脉冲总数除以NE后的脉冲列。`15ms*dR_syn/dt = -R_syn+s_E(t)`（相应单位换成Hz）。数值更新为`R_next=exp(-dt/15ms)*R+n_E_spikes/NE/(.015s)`，当步放电在下一步驱动电流，与原生引擎的因果时序一致。

`J_i=(1-γ) I_local,i + γ C R_syn`，`I_applied,i=Z_i*J_i`。原生Z仍为`5s*dZ_i/dt=1[J_i<I_th]-Z_i`；M为每次E脉冲加1、2s衰减、有效电流`.005*M`。I细胞输入不改、Z=1/M=0。

C=6.5338848042521835 mV-equivalent/Hz。它来自已存在的两条原噪声轨迹0.5–8s原局部GABA均值/E平均率的平均值，该段旧额外全局池实际输出为0。C只做固定参考强度匹配，未拟合终止结果；γ=1/6的5:1是归一化连接权重比，实际各时刻电流不要求5:1。此换算不是将原文nS数值直接当作等效电流，也没有声称两种膜模型生物物理等同。

## 有界实验及验收

γ={0,1/6,1/2} × 噪声{9108401,9108402}，共6条，每条完整60s。固定拓扑6101及原生连续快慢状态/OU/Poisson；M=.005/2s，τZ=5s，I_th=95.1985。γ=0与原生模型逐位一致性检查，非零γ进行原生1s分段续跑与未分段完整状态比较；空间置换、人口数归一化、E-only、15ms脉冲衰减、无50Hz门及实际Z驱动同步核查。

保留原高态/返回门，但不在首次阳性后停止；完整raster、双core/全体E率、原生二维招募、Z实际回补条件和M同时检查。无进入仅代表60秒未过门；core局部持续、较低平台或全场宽爆发均不算恢复原间期。上轮ρ=.25不加入此次候选。

调度最多6worker，同时计入其他Topic4 worker，保留80GiB内存和40GiB磁盘。新的4小时执行上限仅用于该6条件及收尾，不复用已结束夜间预算；不自动扩大参数网格。结果需Agent与用户目视，不能自动冻结模型。

原始来源：[Liou2020 eLife](https://elifesciences.org/articles/50927)、[StandardRecurrentConnection.m](https://github.com/elifesciences-publications/LAS-Model/blob/master/StandardRecurrentConnection.m)、[SpikingModel](https://github.com/elifesciences-publications/LAS-Model/blob/master/%40SpikingModel/SpikingModel.m)。本地原文件哈希和校准来源保存在protocol.json。
