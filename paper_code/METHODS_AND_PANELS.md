# 计算方法与面板接口

本文件描述可公开整理的计算流程。正文终稿保持本机保存；这里不复制 `.tex`。面板调用论文 producer 的本地绘图副本，接收本次原始信号分析与仿真产生的数组。E1146 demo 的数据规模、搜索预算与历史论文批次不同，因此不预设相同输出数值。

## 数据接口

默认E1146 demo为4×200秒电压片段，15个分析触点另加一行 `CAR_REFERENCE`。该行是原114个颅内触点的均值；使用 `reference="channel"` 减去这一行并移除参考行，再执行后续处理。保存的是电压与参考，未保存检测、rank或聚类结果。

自己的原始 SEEG 可直接使用 EDF，或不含检测结果的 NPZ：

```python
np.savez_compressed('record.npz', samples_V=raw_volts,
                    sfreq=native_sampling_rate, channel_names=channel_names)
```

`raw_volts` 为 channel×sample、单位V、未参考的有限数组；通道名唯一。在 `raw_example.json` 设置 `records[].file/start_sec/duration_sec/block_id`、参考、双极配对、坏道/保留通道、滤波/检测/打包参数。相同原记录的片段必须同一 `block_id`。本示例仅演示选定片段，正式interictal队列应在输入清单中完成临床seizure、postictal、gap排除，不能把所有raw窗自动当间期。

Epilepsiae 原始 `.data/.head` 用包内 `load_epilepsiae_block` 读取；其参考、采样率和起始时刻随 `PreprocessingResult` 返回。该输出已经参考并按 `.head` 的 conversion_factor 转换：先核对该批次的物理单位再换算为V，继续处理时设 reference="none"，已陷波则设 notch_freqs=[]；不要再次双极。不能把未核对单位的int16或微伏当V。Yuquan示例的几何采用左触点别名；其他研究若用双极中点，要显式更换坐标合同。

可用 `python tools/export_raw_excerpt.py recording.edf --start-sec 0 --duration-sec 200 --output ../paper_inputs/computational/example.npz` 导出未参考的原始片段及来源记录；也可以直接把EDF路径放进records，不需要预生成任何事件表。

每个患者的坐标JSON为 `{channel_name: [x_mm,y_mm,z_mm]}`。缺失映射保留NaN；不把坐标缺失的事件删出timing训练。

模型输入 `E1146_training_observations.npz` 包含 `fit_centroid_ms[event,contact]`、独立 `cal_centroid_ms`、`cal_draw_indices[256,16]`、`contact_names`、`contact_xy_mm`。缺席触点NaN，时间毫秒；不含模型结果、旧loss、旧动画。`E1146_observer.json` 是独立参考冻结的检测合同，`E1146_contact_contract.json` 固定触点拆分与几何。更换患者须一起更换这些输入；多杆目标用 `cohort_objective.py`。

## Notebook1：Fig1

| 面板 | 本次从原始电压计算的内容 | 论文绘图口径 |
|---|---|---|
| A | 参考、滤波、HFO检测及群窗，三个160ms片段 | 黑色波形；demo不含MRI |
| B | HFO片段平均谱；三个群事件的完整S³质心 | 原HFO均谱函数和群事件谱函数；频率在每个触点内向上递增，白点黑线连接质心 |
| C | 参与触点的dense rank，按名字合并记录 | 原rank热图；右侧离散直方图按每行峰高缩放 |
| E | masked normalized rank → K-means | 相同通道顺序，按簇排列；原均值±总体SD绘图函数 |
| D | legacy MI与置换基线 | 每患者一对数据；一个demo患者不画队列分布或显著性 |
| F | 事件对之间的Kendall一致性，整体与模板内对比 | 原散点/对角线/配对inset结构；两模板等权平均 |

预处理在完整片段上执行polyphase重采样、Q=30陷波和三阶Butterworth带通。按患者mean+k·SD通道事件数规则选择pack触点，±30ms扩展、500Hz栅格、50%通道门槛和固定群窗合并；保留原重叠处理。拼接群窗后计算50ms Hamming、40ms overlap幅度谱，先做σ=1.5 Gaussian平滑，再截50<f<300Hz，以完整事件S³加权求质心。只有真实参与触点获得rank；K-means缺失位置以event-median=0.5插补，标签按事件数排序。

B左侧HFO均谱沿用单独的180ms Hann、160ms overlap及早期基线归一化方法。demo使用新检测片段，论文使用Y1人工标记的178段，不能把两个计数等同。C/E使用E1146新检测事件，不读取Y1的18190事件。K=2是模板设定，不是两个生理状态的证据。

`patient_statistics`同时保存D的legacy MI和F的pairwise统计；`cohort_statistics`以患者为单位，D按队列做Mann–Whitney U单侧检验（data > null），F做整体/模板内的双侧配对Wilcoxon检验。批量原始处理入口为`workflows/cohort.py`。

## Notebook2：Fig2

| 面板 | 计算 | 绘图 |
|---|---|---|
| A | 名称对齐的3D触点 → rank梯度 → 固定TA平面 | 实际坐标、投影平面、6mm Gaussian示意 |
| B | recording-block cross-fit，Timing与Timing+Space用同一训练/测试事件 | 原方向玫瑰图；Timing虚线、Timing+Space实线；单患者held-out配对点 |
| C | 从原始电压重算每个事件的Hilbert包络和四帧空间分布 | 原时频读出/动态场函数；6mm平滑、每帧前三参与触点定标、PowerNorm γ=0.5 |
| D | 全事件Timing+Space拟合后的两模板rank场 | 原物理mm平滑；viridis 0=early、1=late |
| 场比较 | 固定TA平面内两模板场相关及完整触点置换 | 单患者置换分布；不代替论文队列统计 |

B从rank与3D位置做OLS梯度，方向正向early→late。默认1000次held-out block内方向置换和50次training-space shuffle重拟合；两方法共享可评分分母。训练缺少空间坐标的事件保留时序信息。方向玫瑰图使用相同held-out事件和共同投影基底。

C的显示质心沿用其producer的70%主增强成分规则，Fig1的事件rank仍由完整S³计算，两者不混用。D的展示采用物理6mm平滑；统计场保留原方向轴、尺度和支持度的定义：$F(x)=\sum_i K(x,x_i)s_i e_i/\sum_i K(x,x_i)s_i$，$e_i=-z(rank_i)$。方向余弦、展示rank场和统计earliness场是不同读出。`field_record`保存资格与稳定性；`status=ok`不代表通过作者验收。

## Notebook3：Fig4、Fig5、Video2

原生有延迟E/I LIF网络：20×20mm、40,000神经元、E:I=4:1、dt=0.1ms。AMPA/GABA双指数电流；背景阈值18mV、reset 11mV；core内用固定逐细胞分位数降低阈值。独立core OU及逐细胞Poisson输入，无事件定时kick。按原独立RNG重新采样E→E边：垂直连接宽度×1.5，可旋转方向；E→I及GABA拓扑保留。

| 面板 | 新计算及对应绘图 |
|---|---|
| 4A | 本地E/I机制图、新网络/core、Gaussian采样和新事件包络 |
| 4B | 每次候选的三类可观察误差横向连线，颜色为迭代顺序；与优化目标J分开 |
| 4C | 1–4 core best-so-far曲线；原均值±样本SD函数 |
| 4D | −20°至20°共13个E→E方向重新构图、仿真、评分；原三误差响应绘图函数 |
| 4E | 两core候选中最低20% J的位置，σ=0.35mm密度、90%质量轮廓和原relief绘图函数；新峰超出旧裁框时仅扩展画布 |
| 4F | 全轨迹30–80Hz四阶零相位滤波，再截窗；统一P99缩放、SCL/ICL固定顺序及类别底色 |
| 4G/H | 新模型与患者模板mean-rank、触点交叉拟合矩阵；原rank/matrix绘图函数 |
| 4I | E1146的TA–MTA、TB–MTB signed Spearman相似度；原按mode绘图函数 |
| Video2 | 上差/下好工作点，每行MTA/MTB包络及两动态场；固定15触点顺序、250ms、2ms帧、20fps |
| 5A | 新Z/M轨迹10kHz原生电流；原80–1000Hz滤波、募集前P99.5及15% median floor定标 |
| 5B | 同次积分的All E/Core A的Z和ηM，原慢变量绘图函数 |
| 5C | 1ms真实spike计数 → 20ms各格E发放率；原四帧magma/PowerNorm(0.6)绘图函数 |
| 5D | 同轨迹Z、ηM、5ms发放率，原三维绘图函数，颜色t/T |
| 5E | 新模型进入后1s场与原始临床0–10s场：各自基线robust-z；原Blues/min-max绘图，另附真实模型z色条 |
| 5F | 原始发作六带阶段 + 新模型最终募集窗口 → dB场；平滑前完整触点置换，每次重选最佳发作；原患者配对统计绘图函数 |

患者FIT重新建立原三项Fourier分布目标；独立CAL中连续16事件的256组索引用于重算尺度。少于16合格模型事件、无事件或runaway不提供伪造的惩罚loss。患者模板使用FIT的Timing+Space；模型事件沿用FIT训练的rank K-means分类器，语义按FIT与空间模板的一致性固定映射。F/G/Video2共用这些模型标签，H单独执行触点交叉分组。G/H使用新搜索选出的两core模型；Video2比较基准与选优模型，按本次J排列，按每个模板最早合格事件选片，缺任一类别则不生成四事件比较。

优化保留原Sobol起点、Matérn-5/2 ARD GP、固定数值噪声、局部±0.35归一化边界、qLogEI和全局后验不确定性候选。8候选批次为6个qLogEI+2个不确定性点；末批只取剩余预算。不足8个可评分观测时空间填充。可复制入口使用同步批次，原分布式delayed调度未打包；同算法不保证相同历史候选次序。默认10评估×2重启×4core、12s/次；论文规模配置为31×4×4、60s/次。C的样本SD来自独立重启，不把候选当重复实验。

Fig5使用2026-10-11六面板候选的定义。τz=5000ms、τm=500ms、ηm=0.007451594355587098、抑制电流阈值95.19851312666987；从Fig4选优几何重新积分，启用Z/M不重拟合拓扑。t120显示进入与200Hz/200ms保护分别保存。10ms非重叠全E率连续200ms达到200Hz后，按max(15s,进入+3s)停止；为容纳新拟合网络的较晚进入，未进入则最多60s。新进入时刻决定显示窗口；不写死历史14.3–15.4s，也不承诺重现历史进入时刻。显示时刻沿用该版本相对t120的偏移；数值编号不预先判定新轨迹的状态名称。滤波80–1000Hz、快照20ms、显示系数0.65及Z/M颜色均取冻结版本的运行参数，不取旧脚本默认值。每个空间快照的总spike数与全网记录相等。

Fig5临床demo包含E1146进入原论文阶段审阅的23次候选发作。打包的是1024Hz未滤波电压、完整颅内CAR参考、SQL时间和先前独立选择的静息基线标注；不含旧频谱、阶段结果或场。`clinical_candidates`先重新参考和50/100/150/200Hz、Q=30零相位陷波，再计算1s Tukey谱（0.5s hop）。Fig5F重算六带共同增强及阶段，在线噪声±2Hz排除后要求≥5/6带、≥2低频带、≥2快频带，且阶段内≥2s共同支持。静息基线标注是输入；本demo不重做整队列的基线人工筛选。

Fig5E独立保留EEG前−120至−90s基线和clinical0–10s观察窗，选择正平均增强、最高signed场相关的示例。Fig5F使用另一套读出：模型100ms谱窗、50ms hop，1Hz零填充网格的真实分辨率为10Hz；静息窗由低率规则确定，最终20%率跨越作为起点，在最后率上升前100ms至200Hz进入后600ms内寻找持续200ms功率峰，峰窗结束后再保留200ms。患者及模型的dB均不除以MAD。完整触点置换先于原空间平滑，每次在同一合格发作池重选最大绝对相关；99999次，各患者独立随机流。`cohort_statistic`在每次置换内取患者中位数；E1146例图只含一个患者，不声称得到13人队列结论。新模型若缺少进入、完整谱窗或稳定平台，则明确返回不可估计。

## 复现证据与公开边界

自动检查证明代码可独立运行、观测定义一致、输入与输出可追溯；不能替代作者验图或科学验收。每次运行输出独立时间戳目录，保留参数、事件表、仿真轨迹、评分及PNG/PDF/GIF。新的输出不覆盖正式paper-ready图。

完整临床队列与原历史优化的逐次数值未在短示例中重跑。代码目录可以独立复制；运行仍需合规取得的原始数据/患者观测输入。原始临床数据包留在本机且不进入Git。

多患者SNN入口：`python -m modeling.cohort manifest.json --output ../paper_outputs/model_cohort`。manifest每项提供subject、observations、configuration、contact_contract；观测NPZ另需cal_block_ids以保留校准的记录内连续抽样，以及按contact_names排列的coords_mm三维坐标。逐患者保存训练选优、触点交叉匹配和独立噪声确认；Fig4I汇总单位为患者。此完整队列入口尚未用整队列执行。

更换临床输入时，在`clinical_examples.json`逐条提供原始NPZ、相对clinical onset的记录起点、EEG起止时间、静息基线区间、分析/资格触点名和参考通道。参考信号须来自原完整颅内montage，不能用打包的15个分析触点重新求CAR。Notebook 03直接从这些采样重算E/F；代码不读取旧临床结果。
