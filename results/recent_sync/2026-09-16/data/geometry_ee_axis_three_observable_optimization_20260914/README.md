# 三组参数 × 三组观测：实际执行

**2026-09-15 11:58：本轮优化、确认和Agent科学审阅已完整收尾。** 累计72/72个完整可评分槽位（64新仿真＋8严格相容复用），10,847事件，0工程失败、0runaway。新拓扑确认8/8、1,261事件已完成；联合1-1的四运行平均J为2.178，参考为5.199，3/4配对降低，但TB杆间中位差仍34.38–35.50 ms（患者1.19 ms）。去除模式频率因素后的三组条件距离，TB均值改善、TA均值变差；TA杆内顺序误差四配对均增加。因此接受部分分布拟合改善，尚不接受完整患者双模式或Fig5基底。

交付：[最终30页报告](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/final_optimization_report.pdf)、[中文科学审阅](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/analysis/confirmation_review/scientific_review.md)、[16页患者/原生图附册](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/analysis/confirmation_review/native_review_appendix.pdf)、[新候选新拓扑多事件GIF](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/analysis/native_review/g2_b01_p01/3711_849401/patient_mean_native_multievent.gif)、[TB单参数响应图](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/analysis/figures/raw_paired_response_TB.png)、[完成核验](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/analysis/confirmation_review/completion_audit.json)。实际审阅27张最终PNG，其中包括48事件的336个原生抽帧和8张频谱/包络对照；8个GIF共3,024帧解码；抽帧不冒充逐帧观看全部60秒记录。用户人工科学验图仍待进行。

09:51关闭新增派发，之前派发的确认约11:09全部收尾。数值G4触发标准通过，但TA局部传播退步未获共同改善的科学接受；窗口本身也独立禁止追加，G4新增0。既有G1单参数响应已完整出图；未冻结模型、未进入Fig5。以下记录保留为历史快照。

**2026-09-15 09:51：十小时探索窗口结束，8条已派发确认继续收尾。** 四批优化32条件/64训练槽位、9,586事件已完成；完整新网络确认及其原生审阅尚未完成，不能将本夜写为整个优化目标已达成。已派发8条确认均实际存活，G4派发0，窗口后不再新增物理参数响应或搜索。见[窗口交付](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/window_closeout.md)及[实际进程记录](/data/hfosp/topic4_sef_hfo/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/window_closeout.json)。保留完整轨迹完成既定确认分析；患者双模式与Fig5基底仍未接受。下方为历史快照。

**2026-09-15 09:10更新：四批优化已完成，8条新网络确认正在运行。** 训练32条件/64完整槽位（56新+8复用）、9,586合格事件、0失败/runaway；最低J仍0.699。参考与联合1-1于08:39进入2张新拓扑×2噪声确认，提名已冻结；最低J与时序最低同一候选，按规则去重。新[第四批及训练最终科学审阅](overnight_20260914/batch04_review/scientific_review.md)、[参数—观测图](overnight_20260914/batch04_review/figures/batch04_mode_tradeoffs.png)、[患者模板对照](overnight_20260914/batch04_review/figures/batch04_patient_model_rank_templates.png)已生成且Agent实际检查。完整患者TB尚未恢复；后续只完成既定确认及符合合同的窗口内扩展。以下为历史快照。

**2026-09-15 06:42更新：第三批8/8完成，第四批8条已运行。** 共56完整槽位、8302合格事件、0失败/runaway；最低J仍0.699。第四批使用28个完整条件反馈，局部搜索半宽按规则缩至0.125。新的[第三批科学审阅](overnight_20260914/batch03_review/scientific_review.md)包含TA/TB参数响应、患者模板、全事件时间分布和两核相位—读出关联；没有据此改变loss或提案。

**2026-09-15 04:39快照：第二批8/8完成，第三批8条已运行。** 累计48个完整槽位（40新+8复用）、7,038个合格事件，0工程失败/runaway；最低J仍为联合1-1的0.699。第二批呈现更明显的TA/TB时序取舍，没有恢复两类；[本批科学审阅](overnight_20260914/batch02_review/scientific_review.md)、[参数—观测图](overnight_20260914/batch02_review/figures/batch02_mode_tradeoffs.png)与[患者平均模板对照](overnight_20260914/batch02_review/figures/batch02_patient_model_rank_templates.png)已补齐。第三批实际使用24个完整条件更新代理模型，固定物理和目标不变。

**2026-09-15 02:04快照：第一批BO的8/8完成，第二批8条运行中。** 总计40个完整槽位（32新+8复用），5,830个合格事件，0失败/runaway。新最低分联合1-1的J=0.699，但TB杆间差仍32–33ms，患者TB传播尚未恢复；逐模式条件误差的取舍见[第一批审阅](overnight_20260914/batch01_review/scientific_review.md)。下一批已读取20个完整条件并实际更新搜索中心；运行与目标不变。

**2026-09-14 23:51快照：首批32/32完成，BO第一批8条在跑。** 首批含24条新仿真和8条复用，0工程失败/runaway。用户已追加今晚8–10小时自主探索，窗口及目标见[window.json](overnight_20260914/window.json)，动态进度见[状态](overnight_20260914/status.json)，科学结果和参数响应见[报告](overnight_20260914/scientific_progress.md)及[PDF](overnight_20260914/figures/night_progress_report.pdf)。以下17:56记录是启动快照；本夜不改变冻结物理、三组loss或硬范围。条件性G4到窗口终点停止新派发并保留已派发轨迹完整收尾，基础优化/确认依既有合同完成。

用户2026-09-14授权“开始，不要让CPU和GPU空置”。物理背景、变量范围和预算继承[已接受设计](/home/honglab/leijiaxin/HFOsp/docs/archive/topic4/geometry_ee_axis_three_observable_optimization_2026-09-14.md)，原设计快照不回写。运行参数和复用来源见[plan.json](plan.json)。

## 当前状态与自动衔接

2026-09-14 17:56快照：首批32个槽位中8个是旧60秒轨迹复用，24个是新运行；8条新轨迹正在积分，16条排队，尚无新轨迹完成和工程失败。每个条件只开放两核X/Y、核向外EE倍率、全局EE角度六个标量。

- [物理队列实时状态](status.json)：8个worker上限，监测进程树实际RSS和可用内存，每树18GiB保护，预留现有树增长和40GiB系统余量。
- [优化器实时状态](optimizer_status.json)：首批完成并评分后，四批×4条件×2噪声；每批重新拟合Matérn-5/2 GP，用3个qLogNEI点＋1个全范围不确定性点提案，实际结果影响下一批。
- [目标冻结记录](analysis/objective_frozen.json)、[针对性检查](analysis/objective_sanity.json)、[实现验证](implementation_tests.json)、[实际应用值核对](analysis/applied_values_audit.json)。
- G3为参考＋至多2个训练提名条件，拓扑3711/3712×噪声849401/849402，至多12条。确认不反哺本轮BO。
- G3结束自动形成数值触发判断和原生审阅包；G4最多40条，需实际Agent原生图审阅支持有用传播改善。`g3_agent_native_review.json`尚不存在时，停在原生审阅点，不伪造目视通过，也不向用户重复索取已给出的实验授权。

GPU 0负责float64 Fourier特征、GP与采集函数；GPU 1负责Hilbert相位及原生旋转候选筛查。SNN沿用CPU积分器，未为占满显卡更改物理数值实现。分析队列无可处理任务时等待下一份真实轨迹；不以设备空闲为理由增加预算或制造无意义任务。

## 当前图与原始量

[图件说明](analysis/figures/README.md)包含患者TA/TB平均rank与范围、SCL/ICL内部成对时序、杆间毫秒分布、触点参与和真实单标量配对响应。启动时的8条重评分属于旧数据复用；首批完成后已纳入24条新物理轨迹。新结果自动进入同一报告，患者黑色参考线与固定两杆布局保留。

- [患者模板](analysis/figures/patient_model_rank_templates.png)
- [患者—模型原量分布](analysis/figures/patient_model_raw_observables.png)
- [全部ICL触点对](analysis/figures/ICL_all_pair_order.png)
- [TB参数—观测响应](analysis/figures/raw_paired_response_TB.png)
- [原量统计](analysis/raw/observables.json)、[杆间均值/中位数/方差/区间](analysis/raw/rod_lag_summary.csv)、[杆内时差与顺序](analysis/raw/within_rod_pairs.csv)
- [按模型实际N匹配的患者连续块参照](analysis/patient_count_block_reference/figures/patient_actual_N_reference.png)：保留自然标签比例与块内顺序，分别比较中心、散布及参与；不是独立验证、置信区间或相同时长比较。抽样规则及逐运行数据保存在同目录。
- [参考：患者平均模板＋原生场＋多事件读出](analysis/native_review/bridge_circle_out125_xminus075/2511_847401/patient_mean_native_multievent.gif)
- [参考图及Fig2C真实STFT对照说明](analysis/native_review/bridge_circle_out125_xminus075/2511_847401/README.md)

患者平均质心模板是统计量，不能称为平均HFO频谱；真实Fig2C示例单独显示。每模式最早三个合格事件按时间排列，未按患者距离挑选；全部原生E活动始终可见。文件解码与Agent/用户目视审阅各自记录。初始参考图已目视检查显示布局；这不构成未来G3提名候选的科学验收。

原损失和局部宽度、热点等诊断保留在`analysis/legacy_diagnostics/`，旋转候选在`rotation/`；均不进入本轮三组目标。旧损失重排只能比较评分如何改变选择，不能证明优化算法的独立贡献。所有时间原量是ms，方差是ms²；跨维度不压成未经定义的总体方差百分比。

## 控制器与恢复

代码位于`/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-continuous-core-state-r1`。物理环境不变；优化器隔离环境`/data/hfosp/envs/topic4_bo_v1`包含botorch0.13.0、gpytorch1.14，Torch2.5.1沿用系统环境。进程身份保存在本目录`*_process.json`，不可只看PID文件判断存活。

新文件：`scripts/run_topic4_three_observable_bo.py`（准备/物理队列）、`scripts/control_topic4_three_observable_bo.py`（四批BO和G3）、`scripts/analyze_topic4_three_observable_bo.py`（冻结评分和观测器）、`scripts/media_topic4_three_observable_bo.py`（GIF/旋转/旧诊断）、`scripts/continue_topic4_three_observable_response.py`（G4）。恢复时先检查进程身份；已有提案从落盘文件恢复，完整物理单元复用，存活孤立worker按精确执行身份接管，不重复抽参数。

G3后Agent必须读数值触发结果、逐网络模式支持、患者时序和固定规则多事件原生图，实际查看后写`analysis/g3_agent_native_review.json`。其字段须包括`review_type=ACTUAL_AGENT_VISUAL_AND_SCIENTIFIC_REVIEW`、被查看文件及SHA256、科学理由、各候选`no_new_contradictory_propagation`判断。随后运行`continue_topic4_three_observable_response.py run`即可执行已授权条件性G4，不需要新的用户许可；若无候选触发则明确结束本轮。

没有自动无限扩围、冻结患者完整双模式、进入Fig5或提交Git的授权变化。
