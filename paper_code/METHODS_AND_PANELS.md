# 计算方法与面板接口

本文件描述可公开整理的计算流程。正文终稿保持本机保存；这里不复制 `.tex`。论文定稿面板与本次示例的相同点是计算任务，不预设相同数据规模、随机网络身份或输出数值。

## 数据接口

原始 SEEG 可直接使用 EDF，或不含检测结果的 NPZ：

```python
np.savez_compressed('record.npz', samples_V=raw_volts,
                    sfreq=native_sampling_rate, channel_names=channel_names)
```

`raw_volts` 为 channel×sample、单位V、未参考的有限数组；通道名唯一。在 `raw_example.json` 设置 `records[].file/start_sec/duration_sec/block_id`、参考、双极配对、坏道/保留通道、滤波/检测/打包参数。相同原记录的片段必须同一 `block_id`。本示例仅演示选定片段，正式interictal队列应在输入清单中完成临床seizure、postictal、gap排除，不能把所有raw窗自动当间期。

Epilepsiae 原始 `.data/.head` 用包内 `load_epilepsiae_block` 读取；其参考、采样率和起始时刻随 `PreprocessingResult` 返回。该输出已经参考并按 `.head` 的 conversion_factor 转换：先核对该批次的物理单位再换算为V，继续处理时设 reference="none"，已陷波则设 notch_freqs=[]；不要再次双极。不能把未核对单位的int16或微伏当V。Yuquan示例的几何采用左触点别名；其他研究若用双极中点，要显式更换坐标合同。

可用 `python tools/export_raw_excerpt.py recording.edf --config config/raw_example.json --start-sec 0 --duration-sec 200 --output ../paper_inputs/computational/example.npz` 导出未参考的原始片段及来源记录；也可以直接把EDF路径放进records，不需要预生成任何事件表。

每个患者的坐标JSON为 `{channel_name: [x_mm,y_mm,z_mm]}`。缺失映射保留NaN；不把坐标缺失的事件删出timing训练。

模型输入 `E1146_training_observations.npz` 包含 `fit_centroid_ms[event,contact]`、独立 `cal_centroid_ms`、`cal_draw_indices[256,16]`、`contact_names`、`contact_xy_mm`。缺席触点NaN，时间毫秒；不含模型结果、旧loss、旧动画。`E1146_observer.json` 是独立参考冻结的检测合同，`E1146_contact_contract.json` 固定触点拆分与几何。更换患者须一起更换这些输入；多杆目标用 `cohort_objective.py`。

## Notebook1：Fig1

| 面板 | 输入 → 计算 → 输出 |
|---|---|
| A | 原始采样 → 显式参考、重采样、陷波、80–250 Hz带通 → 事件波形 |
| B | BQK通道检测 → 患者规则选择打包通道 → 扩展/阈值/固定群窗 → 拼接频谱S³时间质心 |
| C | 新质心 + 真实参与mask → 每事件内dense rank → 按通道名拼接记录 |
| E | masked normalized rank、缺失插补0.5 → K-means → 新模板/标签 |
| D/F | 共同参与通道对先后的一致性 → 每患者置换基线与模板内统计 → 患者单位配对检验 |

当前小例不等于原18190事件，也不提供队列显著性。K=2是模板模型设定，不证明两个生理状态。群事件pack保留legacy重叠处理；不同pack版本不能悄悄混用。

## Notebook2：Fig2及临床能量场接口

| 面板/任务 | 计算 |
|---|---|
| A | 三维触点名称对齐；由时序计算空间方向 |
| B方向 | OLS rank梯度归一化，正向early→late；数学不可估保留缺失 |
| B对照 | recording-block cross-fit；Timing与Timing+Space共享训练事件；测试rank-only分配；共同可评分分母；block内方向置换、训练空间视图置换 |
| C投影 | 评分完成后单独拟合全事件Timing+Space模板；固定轴、横向方向、同一个长度尺度；估计核宽、支持域和Gaussian earliness场 |
| D | 共同平面触点处TA/TB场Pearson r；打乱TB触点身份并重建场；每患者一个统计单位 |
| Fig3/Fig5临床扩展 | 参考后真实SEEG → 频带功率 → 真实远端基线robust-z → 固定interictal场评分 |

方向轴余弦、平滑场相关、能量幅度投影是不同量。图的显示平面与统计共享平面也要区分。`field_record`保留几何、bootstrap、leave-one-shaft-out资格；`status=ok`表示计算成功，不替代其中的稳定性字段。

## Notebook3：Fig4、Fig5及Video2

神经元方程是原生有延迟E/I LIF；突触电流通过AMPA/GABA双指数滤波。背景阈值18mV、reset11mV；core内使用固定每神经元分位数的lowering-only阈值。40,000神经元中32,000为E。独立core OU控制输入率，Poisson创新逐细胞不同；core外E及I采用确定性期望输入。无事件指定的kick。

| 面板 | 从新仿真计算的读出 |
|---|---|
| 4A | 位置、无自连接空间图、core阈值、采样位置及E/I发放率 |
| 4B | 新候选的平均rank、杆内先后概率和participation误差；与优化目标J分开；曲线要求完整触点/通道对支持，部分支持保留JSON单列 |
| 4C | K=1..4，相同候选预算的best-so-far训练曲线；不可评分尝试计入预算 |
| 4D | 改变连接核角度，重新构图和仿真，再计算三项误差 |
| 4E | 本次有限J配置的最低20%core位置；不是读取原83/413点 |
| 4F | 新spike Gaussian读出、30–80 Hz滤波波形和新检测事件窗 |
| 4G | 模型/患者平均归一化传播rank，缺席触点不填值 |
| 4H | 一半触点分配、另一半评分，交换后汇总；within-shaft置换重复完整流程 |
| 4I | 单患者独立噪声确认；完整队列另走逐患者输入/模型，不能把种子当患者 |
| 5A | 新Z/M轨迹的原始电流波形、固定P95定标；t120与200Hz/200ms分开 |
| 5B | 同一新积分的平均E细胞Z及ηM有效适应电流 |
| 5C | 新2ms活动网格在预定时刻的50ms平均空间快照 |
| 5D | Z、ηM、5ms全体E速率的三维轨迹；不是分岔图 |
| 5E | CAR电流 → 1–150Hz、1s Tukey PSD → 0.5–3.5s基线z → 完整进入后1s场；患者另需原始发作信号 |
| 5F | 13人/140次候选的dB统计需要完整临床队列及其phase定义，默认未执行 |
| Video2 | 两个新工作点×两个新事件，250ms、2ms帧、20fps；每事件以自身最早质心对齐 |

模型能量场先由interictal rank拟合轴并中心化、归一化投影；镜像围绕固定模板轴，不能围绕20mm仿真片的边界。平移坐标原点后相关和投影分数应不变，科学检查包含这一不变量。

Z/M使用当前Fig5历史版本：τz=5000ms、τm=500ms、ηm=0.007451594355587098、抑制电流阈值95.19851312666987。该版本与现代同名模块的运算顺序分别保留。保护触发后保留1.5s记录，确保进入后1s谱窗完整。基本例电流采样2000Hz并展示未滤波波形；与当前候选10kHz采样的30–1000Hz显示区别明确。外部输入、图和core不随Z/M启用重拟合；新图与历史缓存图身份不同，不能承诺相同进入时刻。

原三项分布目标用患者FIT重新建立Fourier映射与目标矩，按原CAL记录内连续16事件的256组索引重算尺度。E1146尺度核对值为rank 0.04485866893176316、timing 0.043128270322858264、participation 0.04559242266535053。它们是实现/输入口径核对，不证明患者机制识别。少于16个合格模型事件、runaway、无事件均保留原状态，不伪造惩罚loss。

默认预算的core比较使用相同独立Sobol初始设计；后半段若已有可评分点则作局部精修。完整参数文件可扩大评估数，但该透明优化器仍不同于论文历史delayed Matern-GP/qLogEI。历史调度器副本在 `modeling/scripts/topic4_core_count_optimization`，依赖自己的冻结实验输入；不自动指向原工作区启动旧实验。

## 复现证据与公开边界

自动检查证明代码可独立运行、观测定义一致、输入与输出可追溯；不能替代作者验图或科学验收。每次运行输出独立时间戳目录，保留参数、输入身份、事件表、仿真轨迹、评分及PNG/PDF/GIF。新的输出不覆盖正式paper-ready图。

完整临床队列与原历史优化的逐次数值未在短示例中重跑。代码目录可以独立复制；运行仍需合规取得的原始数据/患者观测输入。原始临床数据包留在本机且不进入Git。

多患者SNN入口：`python -m modeling.cohort manifest.json --output ../paper_outputs/model_cohort`。manifest每项提供subject、observations、configuration、contact_contract；观测NPZ另需cal_block_ids以保留校准的记录内连续抽样。逐患者保存训练选优、触点交叉匹配和独立噪声确认；Fig4I汇总单位为患者。此完整队列入口尚未用整队列执行。

Fig5E临床例图可提供 `paper_inputs/computational/clinical_examples.json`：每项为file、onset_sec、baseline_sec、seizure_index；file指向含samples_V、sfreq、channel_names的完整CAR montage原始NPZ。baseline_sec为该片段内真实远端基线（通常对应发作前−120至−90s）。`compare_raw_patient_fields`从原始采样逐次重算10s场，再执行正平均增强门槛与最大signed identity-field相关的描述性选择；不输出独立验证p值。默认未附这组原始发作输入。
