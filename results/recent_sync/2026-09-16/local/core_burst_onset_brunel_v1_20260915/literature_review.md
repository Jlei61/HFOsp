# Brunel / Hakim 等原始研究：如何分析 core burst 的出现

## 问题与检索范围

本任务仅研究既有 SNN 的低活动→群体 burst 起始，以及不规则、中间、规则表型的关系。已检索并读取下列六篇文章的模型定义、相关稳定性推导及模拟比较；公开作者版PDF和检索来源保存在 `literature/`。不把经典模型的边界、频率或标签移植到当前空间双核网络。

题名核对自作者版正文：

1. Brunel & Hakim (1999), *Fast Global Oscillations in Networks of Integrate-and-Fire Neurons with Low Firing Rates*.
2. Brunel (2000), *Dynamics of Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons*.
3. Brunel & Wang (2003), *What Determines the Frequency of Fast Network Oscillations With Irregular Neural Discharges? I. Synaptic Dynamics and Excitation-Inhibition Balance*.
4. Brunel & Hansel (2006), *How Noise Affects the Synchronization Properties of Recurrent Networks of Inhibitory Neurons*.
5. Brunel & Hakim (2008), *Sparsely synchronized neuronal oscillations*.
6. Ledoux & Brunel (2011), *Dynamics of networks of excitatory and inhibitory neurons in response to time-dependent inputs*.

## 最直接的研究

| 文章 | 直接相关内容 | 本版吸取的方法与适用限制 |
|---|---|---|
| [Brunel & Hakim 1999, Neural Computation](https://webhome.phy.duke.edu/~nb170/pdfs/brunel99.pdf) | 稀疏抑制LIF网络从稳定群体率到振荡的起始；由线性稳定性和三阶展开确认超临界Hopf。有限规模下，边界两侧的群体自相关均可表现为衰减振荡。 | 同时观察幅度、频率及相干性；不能用有谱峰或自相关衰减判断处在哪一侧。其推导涉及稀疏、低率、弱相关极限，不能直接用于当前强同步core。重点读取§3.3–3.6、式3.20、3.27–3.31及§4。 |
| [Brunel 2000, Journal of Computational Neuroscience](https://webhome.phy.duke.edu/~nb170/pdfs/brunel00JCNS.pdf) | E/I网络的参数相图、固定点分支、多个振荡失稳边界，以及对应四类群体率+raster。阈值附近输入可产生活动—静息交替的较慢群体振荡。 | 复用“参数图→实际波形/raster→稳定性验证”的组织；保留输入均值和方差的自洽作用。文中SI的irregular指单细胞ISI，不是群体burst IEI；其较慢振荡仍主要是约10–60Hz，不能解释成已复现我们约3–4Hz重复事件。重点读取§5.1、§5.2、§6与图2、6–8。 |
| [Brunel & Wang 2003, Journal of Neurophysiology](https://webhome.phy.duke.edu/~nb170/pdfs/brunel03JNP.pdf) | 真实突触时程下的群体振荡频率；区分I–I与E–I反馈，计算神经元响应和突触滤波的幅度及相位。 | 当前模型必须保留AMPA/GABA上升、衰减和分布时延；不把18ms GABA直接换算成burst周期。本版增加core局部I放电及E细胞接收的两种电流记录，为下一步动态响应验证保留原生信息。 |
| [Brunel & Hansel 2006, Neural Computation](https://webhome.phy.duke.edu/~nb170/pdfs/brunel06.pdf) | 随噪声改变的聚簇失稳与群体率振荡失稳；两者对噪声的依赖并不相同。 | 检验部分招募和强同步，不将所有规则波形都称为同一种极限环；噪声改变可能改变工作点及失稳机制。本文为均匀全连接抑制网络，其结论不直接指定双核中间态的类型。重点读取摘要、§2与噪声依赖的稳定性结果。 |
| [Brunel & Hakim 2008, Chaos](https://www.phys.ens.psl.eu/~hakim/08chaosnbvh.pdf) | 从时延rate模型解释临界反馈增益，再用动态放电响应R(ω)和突触响应S(ω)推广到脉冲网络；静态f–I斜率不能替代一般频率下的响应。 | 本版独立复算图2的时延rate例子，验证特征根跨界和临界两侧轨道；图中明确标记为文献基准。不能把该基准当作当前SNN降阶模型。重点读取§II–III及式12、18、20–23。 |
| [Ledoux & Brunel 2011, Frontiers in Computational Neuroscience](https://webhome.phy.duke.edu/~nb170/pdfs/ledoux11.pdf) | E/I网络对时间变化输入的响应；EE率失稳、II反馈及EI反馈失稳的区分，及rate/LIF动态响应的比较。 | 下一步应比较同一工作点的响应幅度、相位及特征根；仅拟合平均发放率不构成匹配。式34–35把三条反馈路径放在同一特征方程中，适合指导当前模型的分区动态响应验证。 |

## 对这次分析真正改变了什么

1. **先拆开两种噪声。** 当前每核共享OU具有150ms相关时间，叠加每细胞Poisson输入。先去掉OU、保留固定强度Poisson，可以检验群体burst是否必须跟随慢外部波动。一次性移除两者，会连维持亚阈值细胞放电的涨落一起改变，不足以区分外部驱动和群体失稳。
2. **改为连续状态干预。** 已有noise-off从低初态启动。本版新增在6秒活动中去噪、12秒单次有限脉冲的连续轨迹，检验所访问状态是否可维持及一次点火后是否自限；不清空突触/延迟历史。
3. **用连续指标解释中间态。** 展示真实事件峰率、IEI、峰值参与比例、群体自相关及谱，不把CV阈值之间的剩余区域当成一个已经证明的吸引子。
4. **严格区分三种时间尺度。** 神经元spike ISI、群体burst IEI，以及burst内部潜在振荡都不同。新增精确spike时刻可测前者；既有2ms占用raster不足以恢复同格多次spike。此次未把模型放电率当作患者HFO波形。
5. **保留本模型的空间反馈。** 观测在core内，仿真对象仍是完整40,000细胞网络；两个core不是独立网络样本。简化成闭合的单个E/I核需要另行验证其与周边、另一core的反馈。

## 独立文献基准的复算

使用Brunel–Hakim 2008图2的方程：

\[
10\dot r(t)=-r(t)+1+\tanh\{I_{ext}-Jr(t-2)\},
\]

时间以ms计。令工作点输入为1，得到 \(r_0=1+\tanh(1)\)，并随J调整 \(I_{ext}=1+Jr_0\)。有效增益 \(K=J\operatorname{sech}^2(1)\)，特征方程为

\[
10\lambda+1+K e^{-2\lambda}=0.
\]

由虚根条件得到 \(K_c=8.50242499\)、\(f_c=134.38110\)Hz。使用Lambert W支计算特征根，并以二阶Heun时延积分检查K=8.4的扰动衰减和K=8.8的持续振荡；0.01与0.005ms步长的晚期幅度相差约0.008%。这个K扫描同时改变耦合和外部均值以保持工作点，和我们只改核内EE的扫描不同。

该计算验证文献分析套路及实现；**没有建立当前空间SNN的临界EE、Hopf类型或周期分支。** 当前SNN首先交付原生状态图及有针对性的噪声/有限扰动诊断，严格分岔命名需补齐匹配工作点的动态响应和稳定性证据。

## 下一步何时值得进入正式分岔分析

只有对应当前阈值、输入统计、空间连接和突触时程的群体近似，同时匹配原生低率工作点与小扰动响应，才使用其特征根命名当前网络的分岔。去OU仍重复burst将支持继续研究内源群体组织；若去OU后消失，则需优先检验慢输入如何访问可激区，而不能直接画一个自主Hopf边界。完全去噪后的有限脉冲若产生持续活动，可进一步检验共存和周期分支；若返回，只约束本次有限初态/刺激，不证明所有分岔不存在。
