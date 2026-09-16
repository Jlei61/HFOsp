# SNN 当前协作者报告

**2026-09-15 对话展示补充**：三张参数PNG直接显示在对话。患者平均模板已接入Fig2C左侧的触点分层、时间轴与配色，仍按SCL/ICL固定分组；当前用预先冻结的每类32个真实HFO包络求均值，并明确标注“非频谱”。原始EEG旧挂载路径为空，新磁盘位置不可读，真正平均频谱尚未完成；不能用单事件频谱或质心合成图替代。该更新只涉及显示，优化仍使用全量冻结训练目标和无标签评分。

**2026-09-15 交付格式更新（用户明确要求）**：每次只给最佳CID多事件GIF、患者平均模板与同一模型事件的时序/原生场快照，以及三张参数比较PNG。每观测一张，列为位置、向外EE、方向；总体/TA/TB作三行诊断，不把模式标签反馈入优化。不再每轮自动生成PDF或图册；后台数据保留。[当前关键结果](/data/hfosp/topic4_sef_hfo/label_free_dense_parameter_search_20260915/key_results/README.md)。

**2026-09-15 当前追加授权：下一版改为无标签事件分布拟合，固定单拓扑2511和单噪声847401。** 已实际启动[加密参数与连续贝叶斯优化](/data/hfosp/topic4_sef_hfo/label_free_dense_parameter_search_20260915/README.md)：继承固定背景，只开放两核XY、核向外EE、全局EE方向；每标量13点，加4批×8连续优化组合，新增上限102条。TA/TB退出评分和优化反馈，仅保留分组诊断；本阶段不跑额外seed。CPU积分与GPU提案梯度并行，实际派发以[状态](/data/hfosp/topic4_sef_hfo/label_free_dense_parameter_search_20260915/status.json)为准。该更新替代本阶段旧多seed确认安排，不改既往结果；当前仍不接受患者双模式恢复。

**2026-09-15 11:58：本轮优化、确认和Agent科学审阅已完整收尾。** 累计72/72个完整可评分槽位（64新仿真＋8严格相容复用），10,847事件，0工程失败、0runaway。新拓扑确认8/8、1,261事件已完成；联合1-1的四运行平均J为2.178，参考为5.199，3/4配对降低，但TB杆间中位差仍34.38–35.50 ms（患者1.19 ms）。去除模式频率因素后的三组条件距离，TB均值改善、TA均值变差；TA杆内顺序误差四配对均增加。因此接受部分分布拟合改善，尚不接受完整患者双模式或Fig5基底。

交付：[最终30页报告](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/final_optimization_report.pdf)、[中文科学审阅](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/analysis/confirmation_review/scientific_review.md)、[16页患者/原生图附册](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/analysis/confirmation_review/native_review_appendix.pdf)、[新候选新拓扑多事件GIF](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/analysis/native_review/g2_b01_p01/3711_849401/patient_mean_native_multievent.gif)、[TB单参数响应图](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/analysis/figures/raw_paired_response_TB.png)、[完成核验](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/analysis/confirmation_review/completion_audit.json)。实际审阅27张最终PNG，其中包括48事件的336个原生抽帧和8张频谱/包络对照；8个GIF共3,024帧解码；抽帧不冒充逐帧观看全部60秒记录。用户人工科学验图仍待进行。

09:51关闭新增派发，之前派发的确认约11:09全部收尾。数值G4触发标准通过，但TA局部传播退步未获共同改善的科学接受；窗口本身也独立禁止追加，G4新增0。既有G1单参数响应已完整出图；未冻结模型、未进入Fig5。以下记录保留为历史快照。

**2026-09-15 09:51：十小时探索窗口结束，8条已派发确认继续收尾。** 四批优化32条件/64训练槽位、9,586事件已完成；完整新网络确认及其原生审阅尚未完成，不能将本夜写为整个优化目标已达成。已派发8条确认均实际存活，G4派发0，窗口后不再新增物理参数响应或搜索。见[窗口交付](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/window_closeout.md)及[实际进程记录](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/window_closeout.json)。保留完整轨迹完成既定确认分析；患者双模式与Fig5基底仍未接受。下方为历史快照。

**2026-09-15 09:10：四批自适应优化全部完成，新网络确认已于08:39启动。** 训练共32条件、64个完整槽位（56新+8复用）、9,586个合格事件，0工程失败或runaway。最低J仍为联合1-1的0.699；后三批没有超过它，不能据此声称全局最优。联合最低与时序分项最低指向同一候选，按冻结规则去重，现为参考与该候选各2张新拓扑×2条新噪声，共8条确认，均实际在跑。

新[第四批及训练最终审阅](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/batch04_review/scientific_review.md)、[参数—观测图](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/batch04_review/figures/batch04_mode_tradeoffs.png)及[患者TA/TB模板对照](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/batch04_review/figures/batch04_patient_model_rank_templates.png)已实际目视检查。联合4-4虽使TB平均rank相关达到0.771/0.600，TB杆间差仍约38ms、TA偏至约−31ms；最低J候选的TB仍32–33ms，因此尚未恢复完整患者双模式。确认完成后另做实际原生场审阅，不自动冻结或进入Fig5；09:51停止新增条件性扩展，已派发完整收尾。以下为历史快照。

**2026-09-15 06:42：第三批8/8完成，第四批8条已于06:30运行。** 累计56个完整槽位（48新+8复用）、8,302个合格事件，0工程失败或runaway。最低J仍为联合1-1的0.699；本批未改进后，优化器按规则将局部搜索半宽由0.25缩至0.125，第四批已读取28条件的真实反馈。新网络确认尚未开始。

新[第三批科学审阅](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/batch03_review/scientific_review.md)进一步说明了噪声敏感性：联合3-3的TB平均rank相关约0.64–0.65，但杆间差在近0与约60–70ms之间变化，两噪声的中位数为61.51/41.96ms。同窗两核峰差与杆间差高度相关（两个噪声r约−0.986/−0.966），[实际原生帧](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/batch03_review/figures/g2_b03_p03_TB_native_frames.png)也显示先B后A与两核近同时活跃的不同过程；这是模型内关联，不能称双稳态、核间因果或患者机制恢复。见[逐事件时间分布](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/batch03_review/figures/g2_b03_p03_rod_lag_timecourse.png)、[参数—观测图](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/batch03_review/figures/batch03_mode_tradeoffs.png)与[患者平均模板对照](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/batch03_review/figures/batch03_patient_model_rank_templates.png)。它们已加入同一持续更新报告；本轮目标、范围、预算不变。以下为历史快照。

**2026-09-15 04:39：第二批BO的8/8完成，第三批8条已于04:31接上。** 累计48个完整槽位（40条新增、8条复用）、7,038个合格事件，0工程失败或runaway。最低J仍为联合1-1的0.699；本批三个局部探索点未超过它，TB杆间延迟部分增至约48–61ms。全范围点联合2-4将TB时差变为约−4.7/−4.8ms，且TB两杆参与保留，却使TA两杆参与降到58–59%、TA的SCL晚约109–111ms；不能把单模式改善写成双模式恢复。

补充[单参数作用的具体解释](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/initial_response_interpretation.md)：同一参考点附近，核向外EE从1.25增至1.375，使TA的SCL内部顺序误差增加约0.139、TB对应误差降低约0.139，两噪声方向一致；整体rank、杆内顺序和杆间时差存在取舍。这里是共用拓扑种子的配对参数响应，角度改变会重建EE边与时延，不能称完全相同的连接图；新拓扑确认尚未完成。

已新增[第二批参数—观测图](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/batch02_review/figures/batch02_mode_tradeoffs.png)、[患者TA/TB与各条件平均模板](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/batch02_review/figures/batch02_patient_model_rank_templates.png)和[本批科学审阅](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/batch02_review/scientific_review.md)，并纳入同一[持续更新报告PDF](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/figures/night_progress_report.pdf)。实际查看的联合2-4频谱/密度对照也显示TA两杆脱节及TB局部顺序残差。第三批已读取完整24条件的训练反馈，中心保留联合1-1，目标和范围不变；G3新拓扑确认仍未开始。下方为历史快照。

**2026-09-15 02:04：第一批BO的8/8完成，第二批8条已运行。** 目前共40个完整槽位（32条新增、8条复用），5,830个合格事件，无工程失败或runaway。联合1-1的两噪声平均J为0.699，低于固定参考1.069；三组加权误差均降低，TB两杆参与保留，但TB杆间差仍32–33ms，TB平均rank相关未提高。逐模式条件距离还显示TA时序变差，不能把总分改善写成全部观测恢复。

第二批读取完整20条件后，搜索中心实际移到联合1-1，确认了逐批自适应更新。[本批科学审阅](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/batch01_review/scientific_review.md)、[10页报告](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/figures/night_progress_report.pdf)、[新候选多事件GIF](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/analysis/native_review/g2_b01_p01/2511_847401/patient_mean_native_multievent.gif)已生成。另一个联合点的TB时差虽降到5.5–8.3ms，却有约35–42%的TB事件缺少SCL参与，相关[两核时序分解](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/core_timing_tradeoff/scientific_note.md)仅作诊断，不改变loss或提名。以下完成量保留为历史快照。

**2026-09-14 23:51：首批32/32完成，真正的自适应选点已经运行；今晚继续8–10小时。** 24条新仿真及8条兼容复用均完整、可评分，无工程失败或runaway。右核左移0.75mm的三组联合J由参考1.069降至0.850；左核左移更明显改善TA平均rank，但TB顺序和约35ms杆间差仍有缺口。当前8条来自首批BO真实提案，不是预先扫参点改名。新拓扑确认尚未开始，暂不能称跨网络稳定。

新[夜间报告PDF](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/figures/night_progress_report.pdf)、[科学说明](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/scientific_progress.md)和[实时状态](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/status.json)作为当前入口。已设置本夜自主goal，至次日07:51–09:51复审交付；按冻结方案完成优化、确认、真实多事件原生图审阅及符合条件的参数响应，不自动增加生物参数或进入Fig5。下方18点及更早完成量保留为历史快照。

2026-09-15 00:15补充[按实际事件数匹配的患者块参照](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/analysis/patient_count_block_reference/figures/patient_actual_N_reference.png)：首批最低分候选N=144/145的TB杆间差仍为35.5–35.9ms，患者连续N事件窗中位数的5–95%范围约−8至7ms。TA的时差中心更接近，但事件间散布仍偏窄。该参照保留CAL原始块内顺序，是开发数据的描述性范围；不匹配记录时长、不作为独立验证或新loss。报告PDF已纳入此图。

**2026-09-14 17:56 新优化已启动：** [三组参数×三组观测执行包](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/README.md)已完成新目标校准、缺失/负分/时间单位检查和GPU批量贝叶斯优化器验证。首批24条新轨迹中8条在跑、16条排队，另有8条严格相容的旧轨迹复用并重评分，暂无工程失败。后续四批选点由前批真实loss更新；2张新拓扑×2条新噪声用于确认。参考物理保持固定，旧分数与新三组目标分别保存。

本轮实时交付包括[患者TA/TB模板](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/analysis/figures/patient_model_rank_templates.png)、[原量分布对照](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/analysis/figures/patient_model_raw_observables.png)、[参数—观测图说明](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/analysis/figures/README.md)及[患者平均模板/原生场/多事件读出GIF](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/analysis/native_review/bridge_circle_out125_xminus075/2511_847401/patient_mean_native_multievent.gif)。此时均来自兼容旧轨迹，不能称为新优化已恢复TB。图件逐批更新；G3实际原生图审阅与用户人工验图分别记录，不自动升级为Fig5基底。

**2026-09-14 当前参考点：其他物理参数已固定，准备收敛到局部优化。** 用户指定先固定此前最佳可用工作点，现选用`bridge_circle_out125_xminus075`（左移圆核＋向外EE增强）。三张拓扑×两条噪声六次60秒结果已核对，正式分析共924事件（TA470/TB454）；其既有综合分数在当前六个完整复测条件中最低。完整参数及保留的三组变量见[冻结参考点说明](/data/hfosp/topic4_sef_hfo/local_geometry_ee_axis_reference_20260914/README.md)和[主目录配置](../config/topic4_local_geometry_ee_axis_reference_v1.json)。冻结的是搜索背景：未来只留两核XY、核向外EE强度和全局EE方向；本次未启动新搜索。TB杆间差与局部顺序尚未恢复，仍未接受完整患者双模式或Fig5基底。以下9月13日完成量属于历史时点快照。

**2026-09-13：01:32–09:32的8小时自主探索及审阅完成；既定物理批次仍在运行。** 当前接受部分参数—观测响应，尚不接受患者TA/TB完整传播恢复，未冻结模型或进入Fig5。

先看[本次科学判断](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/review_window_close_0932/interpretation_addendum.md)、[统一图集PDF](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/review_window_close_0932/overnight_review.pdf)和[新噪声多事件GIF](/data/hfosp/topic4_sef_hfo/core_multiseed_response_curves_20260913/analysis/first_noise_review/README.md)。图件已由Agent核查，仍待用户人工目视验图。

## 完成量与继续执行

|队列|完整运行|正在运行|尚未派发|
|---|---:|---:|---:|
|续跑参数响应|60/60|0|0|
|续跑确认|0/16|8|8|
|多网络响应|4/108|8|96|
|全局EE方向|4/6|1|1|

以上为2026-09-13T09:34:12.921484+08:00快照；续跑响应含48条该批新运行及12条历史复用，原140条已完成，不能重复计算。交付时状态与存活执行单元见[窗口交付记录](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/window_closeout.json)。已授权的确认与108条多网络响应继续至各自预算结束；方向探针09:32停止新增派发，已派发轨迹完整收尾。完成后停在审阅点。

## 当前科学判断

- 左核左移0.75mm在开发网络的两条开发噪声及一条新噪声下，改善TA上部SCL参与与杆间时差；新网络2711上，左移使TA时差和参与分布误差变差，整体收益没有跨网络保留。
- 扩大core、拉长／上移core或改变EE方向会在参与、时序和模式数量间产生取舍。扩大且匹配降阈值总量时，左核总发放增加，TA事件数量反而减少；不能只以覆盖电极更多为优。
- TA的时差中心改善后，完整事件散布仍不相符；TB多数事件仍在约33–35ms的跨杆延迟附近聚集，固定ICL局部顺序仍偏离患者。患者参考跨杆中位差约1.19ms；完整时差分布、患者STFT与原生GIF共同判断，不以两个标签或低分替代传播恢复。
- 核间先后与电极路径并非一一对应。新网络原位置的晚SCL尾部个例显示后段沿上缘传播；这是诊断个例，不证明全部事件同机制或边界反射。约半圈旋转候选也尚不能接受为稳定螺旋。

最新新网络配对见[逐项结果](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/position_new_network_replay/scientific_note.md)：参与增加和顺序误差降低部分同向，但患者相似度并非普遍改善；不能冻结该位置。

## 模型、观测与解释口径

见[共享设计口径](/home/honglab/leijiaxin/HFOsp/docs/topic4_patient_geometry_prior_snn.md)。两核E阈值只降或保持背景，随机OU/Poisson输入限核；核外保留确定期望输入与递归传播。Z/M、空间OU、慢I、定向刺激关闭，GABA18ms固定。当前9月15日优化采用rank、杆内/杆间时序、参与三组去自配对统计量，各自除以冻结正尺度后等权平均；仅全事件无标签特征参与拟合，不读取TA/TB分类器、模式比例或条件目标。9月14日版本包含模式加权特征，已作为历史结果保留，不再作为当前目标。此前9月13日的0.25/0.25/0.50权重属于旧损失，现仅离线保留作评分比较，不能与当前J混用。

患者FIT：TA13,165、TB6,605。运行是实验单位，事件是运行内样本；患者分布、逐网络重复性与机制解释分开。上部SCL参与为SCL9/8概率平均；杆间差为每事件两杆参与触点质心的中位数之差，再汇总事件。部分旧图注误写均值，已更正，数值不变；见[勘误](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/rod_definition_caption_correction/correction.json)。患者STFT与模型发放包络不等价，固定分杆15行、缺失保留、实际毫秒轴不拉伸。

## 图与历史入口

- [全事件时差分布及均值／中位数／方差／范围](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/bridge_event_distributions/scientific_note.md)。
- [全部14种既有局部参数的配对作用](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/completed_parameter_effects/scientific_note.md)。
- [范围三点曲线与实际阈值剂量](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/radius_tradeoff_review/scientific_note.md)。
- [旧32条几何确认的网络依赖](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/geometry_confirmation_effects/scientific_note.md)。
- [窗口内过程及之前各版本报告](/home/honglab/leijiaxin/HFOsp/docs/archive/topic4/snn_overnight_progress_2026-09-13.md)；[逐轨迹来源](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/review_window_close_0932/completed_run_summary.csv)。
