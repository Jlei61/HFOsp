# Topic 4 模型版本与分岔结果入口

2026-09-30 整理。用户指定：日常所说的“分岔分析”优先指保留空间的网格 rate 模型。不同闭合方程的阈值、Hopf、周期分支及稳定性不能互相替用。此次验收是既有工作的整合与来源核验；各结果原有的未完成科学问题继续保留。

| 版本身份 | 保留什么、简化什么 | 代码入口 | 结果入口与用途 |
|---|---|---|---|
| **SPATIAL_RATE_INTERICTAL，默认** | 原空间 SNN 的 20×20 格、935 个空间×E/I×区域×阈值群体；每群体9个局部状态，共8415个状态，另有物理延迟历史。Z固定为1、M保留动态反馈；私有输入方差保留，共享OU取均值。不是逐细胞spiking仿真。 | [`rate_field.py`](../scripts/topic4_brunel_spatial_bifurcation/rate_field.py)、[`model.py`](../scripts/topic4_brunel_spatial_bifurcation/model.py) | [`interictal_rate_summary_20260923`](../results/topic4_sef_hfo/interictal_rate_summary_20260923/)、[总图PDF](../results/topic4_sef_hfo/interictal_rate_summary_20260923/figures/spatial_rate_focused_composite.pdf)。用于连接强度、动力学分支、二维传播及触点率读出。 |
| **REGIONAL_SIX_RATE，区域简化版** | 将 Core A、Core B、Surround 各压成E/I，共六群体；保留区域连接矩、阈值分布、突触滤波和延迟，丢失区域内部二维传播。没有Z/M适应状态，也不是严格六维ODE。 | [`model.py`](../scripts/topic4_core_bifurcation_v2/model.py)、[完整说明](topic4_six_population_rate_model.md) | [`core_burst_bifurcation_v2_20260915`](../results/topic4_sef_hfo/core_burst_bifurcation_v2_20260915/)、[`core_bifurcation_composite_v9_20260916`](../results/topic4_sef_hfo/core_bifurcation_composite_v9_20260916/)。口语中的“三层”在此应明确写成**三个区域、六个E/I群体**，不是三层前馈网络。 |
| **REGIONAL_SIX_RATE_HETEROGENEITY** | 上一六群体DDE的阈值异质性/均值实验。改变Core A阈值分布，不能据此声称空间网格上的传播范围已经验证。 | [`topic4_core_heterogeneity_v11`](../scripts/topic4_core_heterogeneity_v11/) | [`core_heterogeneity_bifurcation_20260918`](../results/topic4_sef_hfo/core_heterogeneity_bifurcation_20260918/)、[Core A参考样式图](../results/topic4_sef_hfo/core_heterogeneity_bifurcation_20260918/core_a_focus/figures/core_A_reference_style.pdf)。 |
| **SPATIAL_RATE_ZM_V3，Fig5状态路线** | 同空间算子族，局部转移表、响应滤波及Z/M闭合不同；实际 `NS=14` 个状态/群体。与默认间期版9状态闭合不同。后续 frozen_v3 和原生Z切片另有各自证据边界。 | [`dynamics_v3.py`](../scripts/topic4_zm_onset_rate_v3/dynamics_v3.py)、[`topic4_zm_runaway_mechanism`](../scripts/topic4_zm_runaway_mechanism/) | [`fig5_zm_rate_v3_20260918`](../results/topic4_sef_hfo/fig5_zm_rate_v3_20260918/)、[`fig5_zm_runaway_mechanism_20260918`](../results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918/)。不能把该路线的条件Z分岔数值移给默认间期版。 |
| **SPATIAL_RATE_TRANSFER_RESTART，早期空间闭合** | 空间格与延迟保留，以局部响应表构成另一种率方程。`topic4_spatial_rate_model` 这个名称不代表它就是当前默认的Brunel两滤波器版本。 | [`topic4_spatial_rate_model/model.py`](../scripts/topic4_spatial_rate_model/model.py) | [`fig5_spatial_rate_model_20260917`](../results/topic4_sef_hfo/fig5_spatial_rate_model_20260917/)。作为历史闭合及局部响应验证保留。 |
| **NATIVE_SPATIAL_ZGK_LOOP，原生闭环/条件转换** | 原生空间spiking网络以及其复制/密度近似；Z、G、K、M有各自定义。固定Z/K的条件响应与自主轨迹分开。 | [`topic4_loop_bifurcation`](../scripts/topic4_loop_bifurcation/) | [`fig5_loop_bifurcation_20260927`](../results/topic4_sef_hfo/fig5_loop_bifurcation_20260927/)、[`mechanism_synthesis_current.md`](../results/topic4_sef_hfo/fig5_loop_bifurcation_20260927/mechanism_synthesis_current.md)。正式分岔仍为 `NOT_ESTABLISHED`。 |

## 默认空间版的证据边界

冻结总览的统计单位是某一确定参数、历史和分支的确定性解；重复周期、不同分箱原点不是独立样本。二维活动和触点加权率来自同一空间模型，触点率不是电压SEEG。`J_EE,core`只放大两核各自核内E→E：一阶矩乘J、平方权重矩乘J²。

当前总览支持多个稳定状态及不同的周期burst传播模式。保存的48个周期折点中41个完成局部验证，7个待核验；整段稳定性、完整分支连接、稳定irregular burst及原生SNN完整动力学等价仍未建立。不能因为图上有二维传播，就把这些缺口视为已经通过。

静态 `SpatialBrunel`、早期线性响应谱和 `RateField` 的动力学闭合也要区分：相同固定点不意味着相同时间谱。读取总图时使用 `RateField` 的方程、同版本谱及相应轨道。

## Figure 5 与复现范围

[Figure 5当前入口](current_figure5.md)是单种子70点进入时间图；它与上述默认间期分岔总图、Z/G/K退出候选均为独立产物。不能按文件名中的 `fig5`、`rate` 或 `spatial` 合并科学结论。

[机器可读版本表](../config/topic4_model_versions.json)与每个代码/结果目录中的 `MODEL_IDENTITY.md` 同步。切换版本时必须同时切换算子、状态方程、慢变量、参数含义和结果目录。

远端包含默认空间版的g20算子、平衡分支和两个Hopf状态，可独立进行有界方程核验；六群体冻结输入与异质性结果已打包。完整空间轨道、原生spikes、检查点及数据缓存保留原位置，见[外部工件清单](archive/workspace/integration_2026-09-30/external_artifacts.json)。该清单中标为 `existence_and_size_only` 的文件只核对存在性/大小，不冒称内容哈希验证。重跑完整实验仍需要原数据挂载；本次没有重新启动科学扫描。
