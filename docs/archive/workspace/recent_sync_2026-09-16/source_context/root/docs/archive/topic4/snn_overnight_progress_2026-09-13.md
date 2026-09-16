# 9月13日窗口结束前的SNN协作者报告

原位置：/home/honglab/leijiaxin/HFOsp/docs/snn_model_report_current.md；下文保留当时的过程与历史结论，最新状态见当前报告。

# SNN 当前协作者报告

**自主探索窗口已开启：2026-09-13 01:32–09:32（北京时间）。** 用户已明确要求设置goal并自主推进8小时；保留当前续跑和108条多网络响应预算，继续分析、检查资源与更新图件。09:32交付实际完成量、仍运行任务与科学判断，不把到点写成实验全部完成；窗口记录见[window.json](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/window.json)。

当前已整理[二十二页统一图集](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/review_0835/overnight_review.pdf)及[本次结果解读](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/review_0835/interpretation_addendum.md)。这是08:34的数据快照：续跑响应60/60、确认0/16、多网络3/108、方向探针3/6；并非8小时窗口结束或全部批次完成。新增完整三点半径响应、全局EE方向±15°对照和首组新噪声位置响应。图件经Agent核查，仍待用户人工验图；后续完整轨迹另存新快照。

**2026-09-13 更新：原140/140条已全部完成，15,049个合格事件，0工程失败/物理runaway。** 形状/范围52、局部参数56、新拓扑确认32均完成。续跑在本次01:17快照中完成29条新运行，另复用12条直接对照；其几何剩余条件和最多16条新噪声确认继续执行。复用事件不与原140条重复计为新样本。见[本次科学报告](/data/hfosp/topic4_sef_hfo/core_response_review_20260913/scientific_report.md)、[更新的关键图PDF](/data/hfosp/topic4_sef_hfo/core_response_review_20260913/result_report.pdf)、[图与六事件原生场GIF](/data/hfosp/topic4_sef_hfo/core_response_review_20260913/figures/README.md)。这是固定数据快照，后续状态以[续跑实时文件](/data/hfosp/topic4_sef_hfo/core_recruitment_tradeoff_followup_20260912/status.json)为准。

**当前接受参数—观测响应证据，仍不接受患者双模式已恢复。** 向外EE增强25%的圆核条件下，同一网络2511、两次噪声逐条配对，左核x−0.75mm使TA上部SCL平均参与由33.4%升至62.0%，两运行时差中位数的平均由−16.07变为−12.55ms，顺序概率误差由0.184降至0.149，冻结分数由3.480降至2.630。患者TA参考为84.8%、−9.28ms；TB仍约+35.1ms，患者约+1.19ms。实际患者—模型图仍见TA触点缺失和TB左端折返不符；低分不是传播机制验收。

**07:40首组新噪声的配对已完成，左移对TA的改善方向保留。** 固定基础网络2511、60秒、噪声847401，圆核原位置与左移各139个合格事件。TA上部SCL参与39.6%→61.7%，成对顺序概率误差0.167→0.156，杆间时差−15.62→−12.48ms；与两次开发噪声方向一致。TB杆间差仍35.15→34.94ms；ICL11先于ICL9仅10/72→12/49，患者约69.4%。TA标签比例48.2%→64.7%和总分3.200→1.952改善，不代表TB完整路径恢复。本次是同网络换噪声，不是跨网络确认；静态数组已按位置核对一致。见[开发与新噪声配对图](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/position_new_noise_replay/figures/position_effect_old_and_new_noise.png)、[定义和逐运行结果](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/position_new_noise_replay/scientific_note.md)、[首条新噪声患者对照及六事件GIF](/data/hfosp/topic4_sef_hfo/core_multiseed_response_curves_20260913/analysis/first_noise_review/bridge_circle_out125_xminus075_topology2511/README.md)。第二新噪声和新基础网络仍在运行。

**补充完整时差分布：TA摘要改善并未恢复患者散布。** 网络2511首条新噪声的左移条件，TA杆间时差均值−13.28ms接近患者−12.52ms，但方差23.97ms²远小于患者268.52ms²；5–95%范围−21.73至−5.57ms，患者−42.63至5.42ms。TB仍大部分集中在约35ms，固定ICL局部顺序仍不符。见[ALL／TA／TB完整累计分布](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/bridge_event_distributions/figures/new_noise_event_delay_distributions.png)和[有效样本数、单位及解释](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/bridge_event_distributions/scientific_note.md)。杆间量实际为每杆参与触点质心中位数之差；部分源图注误写均值，已[修正](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/rod_definition_caption_correction/correction.json)，原代码与数值未变。

**08:26续跑响应60/60及全部12条范围对照完成，16条确认已接上。** 固定原损失、每条件两噪声等权后，提名圆核向外EE×1.25＋左移0.75mm，以及圆核向外EE×1.125／核内EI×0.875，各带直接对照；四条件×两张已用过的网络2611/2612×新噪声847301/847302。已独立核对全部24个新条件的排序和48个新响应来源，见[提名核查](/data/hfosp/topic4_sef_hfo/core_recruitment_tradeoff_followup_20260912/replication_selection_audit.json)。这不是新拓扑确认，尚无确认结果，不能因提名就冻结模型。

范围从1.75增到2.35mm时，匹配左核降阈值总量，实际E成员720→1358、平均降幅0.736→0.390mV；左核群体总发放增加。圆核TA数71/53→15/15，椭圆39/31→11/8，TB仍约35ms且局部顺序未恢复。SCL参与与模式比例存在取舍，椭圆的参与率随范围也并非每条噪声单调上升。见[完整三点范围曲线](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/radius_tradeoff_review/figures/radius_participation_and_activity.png)和[实际物理量、支持数及解释边界](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/radius_tradeoff_review/scientific_note.md)。

**07:00两噪声完成：更多SCL招募不一定更接近患者。** 椭圆核内EI×0.75工作点再上移1mm，两噪声的TA上部SCL参与分别由84.6%/80.6%升至98.2%/100%，患者约84.8%；TA杆间时差由−21.80/−21.37变为−29.74/−30.52ms，离患者−9.28ms更远。总分7.488/8.827降至6.430/7.627，但参与组合项反而变差。TB的ICL11先于ICL9仍为0/108、0/125，患者约69.4%。接受本网络的招募—时序取舍，不接受双模式恢复；见[四条配对轨迹的定义和结果](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/ellipse_yplus1_two_noise/scientific_note.md)、[两噪声固定分杆患者对照与新增第二噪声GIF](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/ellipse_yplus1_two_noise/figures/README.md)。

**02:08冻结输出补充：核间活动先后已与标签相关，尚未转化为正确接触路径。** 四个固定工作点、2511两条噪声中，TA均伴左核窗口内累计10%发放较早；TB中右核较早占70%–91%。这不能套用旧混合阈值版本的近五五开解释，也不证明核间因果驱动。即使核先后已相反，TB时差和完整接触对顺序仍有残差；不分TA/TB的全部事件也在约+35ms形成患者不具有的窄峰。见[核时序与路径审计](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/core_timing_route_audit_0212/scientific_note.md)和[四页图件](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/core_timing_route_audit_0212/core_timing_and_tb_routes.pdf)。杆间质心差还会受参与触点组成影响，正在用固定接触对及共同接触集合单独拆解；当前不能把全部差值都归因为速度或真实跨核间隔。

**02:13接触集合分解已完成：TB主要残差并未因统一触点而消失。** 左移候选两噪声中，模型减患者的杆间均值差由23.38/21.53ms变为23.35/22.08ms；统一触点后的模型中位数31.45/28.59ms，患者1.19/1.10ms。这里均值差与上文约35ms的模型中位数不同，不能混用。固定触点对也有反向残差：TB的ICL9−ICL11质心差患者中位数+5.97ms，模型−14.52/−14.54ms。见[完整前提与结果](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/rod_lag_participation_audit_0206/scientific_note.md)、[图件](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/rod_lag_participation_audit_0206/rod_lag_participation_audit.pdf)。使用所有模型—患者事件交叉计算，无最相似配对、不改loss；交叉对不是独立样本。该审计定位观测残差，尚未唯一识别应改哪类连接。

形状×EE×EI网格已完整，显示固定形状下连接参数会相互影响：减弱核内EI并增强向外EE，可能由SCL漏招募变成近乎全部参与。只有一个拓扑的组合结果，不能直接称为全局效应。完整半径确认中，原圆核/半径2.5mm/扩大且匹配降阈值总量的平均分数为7.867/5.927/13.371；后者各确认运行仅6、1、2、6个TA事件。扩大core改变阈值总量、随机输入支持和连接分块，不能据此唯一归因或宣布两模式恢复。

**新定位：同一TB标签内部仍混合了不同的核活动先后与接触路径。** 左移圆核中，左核窗口累计10%活动较早的13/16个TB事件，其ICL11早于ICL9的比例为84.6%/87.5%；右核较早的59/50个事件仅6.8%/12.0%，且后者杆间时差维持+35.49ms。前者完整接触对顺序误差也较低，但仍未完整恢复患者分布。这里的窗口内t10/t50不是因果起源或事件前状态，原生场可见双前沿叠加；不能筛掉较差亚组后声称恢复，也不能据此直接要求“让左核总先发”。见[定义、五工作点数据与边界](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/core_phase_tb_fold/scientific_note.md)及[四页图集](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/core_phase_tb_fold/core_phase_tb_paths.pdf)。对377个TB事件，仅将核心时序诊断窗前后平移25ms，+25ms全部保持先后、−25ms仅1例改变；这减弱了窗口任意位置的解释，仍不是因果验证。本夜物理、loss和提名不变，新结果将同时检查路径变化与亚组组成变化。

**04:12补充：核心先后与电极先后之间还存在采样不对称。** 两个左移组合的左核几乎不直接进入任何触点的0.25mm空间读出核；右核与ICL1–3明显重叠。这是几何采样权重，不能写成事件信号或解释方差的百分比。左核可先活动、再通过核外组织被电极观察，因此不能直接用最早触点判断哪一核先发。见[实际core成员与固定SEEG布局图](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/core_contact_sampling/figures/core_contact_sampling.png)和[定义及量化结果](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/core_contact_sampling/scientific_note.md)。已补[两类TB过程的同步原生场、核内发放及患者对照GIF](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/tb_core_timing_native_review/figures/README.md)：同条记录中按各时序亚组自身均值选例，较好接触顺序仍可伴双前沿相遇，不能据此接受完整TB机制恢复。

**两个组合的收益已经拆开比较，固定分杆图和多事件GIF已补齐。** 圆核向外EE增强＋左移组合的TA占比为48.6%/53.5%、上杆参与65.4%/58.6%；椭圆核内EI减弱＋左移组合的TA占比30.5%/26.8%、上杆参与85.3%/83.3%。患者对应66.6%和84.8%。两者TB跨杆中位时差均约+35ms，患者+1.19ms；ICL11早于ICL9的概率也仍明显偏低。这是多参数组合间的权衡，不是单独形状作用，不能只按一个参与指标判定优胜。见[逐运行比较及定义](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/workpoint_tradeoffs/scientific_note.md)、[四页比较图](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/workpoint_tradeoffs/workpoint_comparison.pdf)及其中每类前三例原生场GIF。

**已补齐全部14种局部干预的直接作用图。** 原140条中的56条局部干预与4条基底按同形状、同基础网络、同噪声相减，ALL/TA/TB分别展示参与、时序、模式比例、宽度和原生空间观测，颜色明确区分两种形状而非“不同网络”。两核OU相关系数提高至1主要增加TB标签比例，TB杆间时差却仅变化约−0.1至−0.3ms；核内EI增强25%的四条运行未检测到事件，传播指标明确不可估计。见[图集PDF](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/completed_parameter_effects/parameter_effect_atlas.pdf)及[完整定义与配对数值](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/completed_parameter_effects/scientific_note.md)。这仍是单个开发拓扑的结果，多网络响应继续执行。

**05:25补充：几何效应不能只看跨网络平均值。** 原32条几何确认已按两张网络、两次噪声分别配对作图。上移后的左圆核改为等面积椭圆时，网络2611的TA杆间中位时差由约88ms变为−26/−17ms；网络2612却由−4.5/+1.0ms变为+4.2/+10.2ms，跨网络变化方向相反。两噪声在各网络内方向较一致，但这里只能说这两张网络上的条件依赖，不能估计“总体有多少方差由网络决定”。TA标签内事件组成也会随参数变化。全部32条的TB中位时差仍约36–38ms，ICL11先于ICL9的概率仍不超过约11.5%。见[四页确认图集](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/geometry_confirmation_effects/geometry_confirmation_report.pdf)及[逐运行定义与边界](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/geometry_confirmation_effects/scientific_note.md)。这些几何条件均为核内EI=1、向外EE=1，不能替代正在跑的EE/EI或左移候选复测；没有新增或重复计算物理运行。

相同噪声种子下，以上32条运行存下的两核实际OU调制率曲线均逐值相同，因而不能把响应差异归因于这两条曲线改变。逐细胞Poisson实现尚未核对，仍不宣称所有随机输入完全配对或网络结构单独解释了全部差异；见[输入核查及未启用全局日志的区别](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/geometry_confirmation_effects/ou_input_interpretation.md)。

其中圆核／椭圆的8条确认已进一步拆开核心时序：160个TA事件全部是左核窗口累计10%活动较早，154个可测杆间时差。网络2611核间中位差仅由约65ms变为69ms，杆间差却由约88ms变为−26/−17ms；不能用领先core交换解释。图见[核心先后与电极先后](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/geometry_confirmation_effects/core_contact_timing/figures/core_phase_and_rod_timing.png)，[定义与解释边界](/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/geometry_confirmation_effects/core_contact_timing/scientific_note.md)。窗口t10不是因果起源，差异仍须由实际传播、接触参与和采样共同解释。

**本夜新增有边界的连接方向探针；三条500ms实际应用检查已通过，20秒正式运行已开始。** 对现有左移候选，仅将全局EE连接轴相对原轴改为0、−15、+15度，每条件2个既有开发噪声、20秒，共最多6条。实际加权EE边位移主轴为−22.05／−37.37／−7.40度，原始位置、阈值、GABA图和500ms OU输入前缀保持；[实际图检查](/data/hfosp/topic4_sef_hfo/global_axis_residual_probe_20260913/applied_geometry_audit/scientific_note.md)已完成。但改变方向会重采样EE边和距离时延，也使局部总输入略变，不能称为同一张连接图或逐目标剂量严格匹配。该小批检查TB时序残差对连接方向是否敏感，目前非零角度尚未完整输出；不修改108条、固定loss或已有物理，09:32后不再派发新探针。见[具体合同](/data/hfosp/topic4_sef_hfo/global_axis_residual_probe_20260913/execution_plan.md)与[真实状态](/data/hfosp/topic4_sef_hfo/global_axis_residual_probe_20260913/status.json)。

06:00已完成零角度首条20秒基线：53个合格事件（TA22、TB31）；全20秒原始场、两核发放与OU输入和旧60秒轨迹前缀逐值一致，见[前缀检查](/data/hfosp/topic4_sef_hfo/global_axis_residual_probe_20260913/analysis/baseline_prefix_replay_847101.json)。它验证执行链，不是新增独立重复，也不能据此判断改变角度的传播效应。首条基线的[患者对照与GIF](/data/hfosp/topic4_sef_hfo/global_axis_residual_probe_20260913/analysis/figures/global_axis_+0/README.md)已生成，第二噪声列明确待完成。

**07:00首个非零角度完成，尚未改善TB路径。** −15度相对同噪声、同20秒基线，TB杆间差35.31→31.94ms，但ICL11先于ICL9由8/31变为0/23；TA上部SCL参与54.5%→37.0%，两类成对顺序误差均升高。总分3.416→2.867仍不构成传播验收。方向改变了实际EE图与距离时延，第二噪声及+15度待完成；见[逐观测配对结果](/data/hfosp/topic4_sef_hfo/global_axis_residual_probe_20260913/analysis/first_angle_review/scientific_note.md)。旋转筛查新增4个约半圈、58–64ms候选，频段匹配很弱；[最大候选原生GIF](/data/hfosp/topic4_sef_hfo/global_axis_residual_probe_20260913/analysis/first_angle_review/figures/global_axis_-15_847101_rotation_candidate.gif)目视为弯曲前沿向右移出固定环，片段内未见反复重入，不接受稳定螺旋解释。

**08:00三个角度的首条噪声齐全：+15°改善部分TB顺序，仍未恢复完整路径。** 同20秒、同基础拓扑种子及噪声，+15°使TA上部SCL由54.5%升至89.6%，TB成对顺序误差由0.321降至0.258；TA杆间差却从−12.39变为−18.96ms，TB仍33.15ms。TB局部顺序是8/31→8/26，分子没有增加。−15°虽总分更低，TB顺序误差反而更高，不能据此提名最佳传播机制。见[三角度完整对照与定义](/data/hfosp/topic4_sef_hfo/global_axis_residual_probe_20260913/analysis/three_angle_first_noise/scientific_note.md)。第二噪声继续，角度重采样实际EE边和距离时延，不能宣称同一物理图的纯权重效应。

**2026-09-13 用户再次授权继续；新批次已正式运行。** [多网络参数曲线合同](/data/hfosp/topic4_sef_hfo/core_multiseed_response_curves_20260913/execution_plan.md)固定18条件×3网络(2511/2711/2712)×2新噪声(847401/847402)=108条新60秒运行。两种形状下核内EI补5点，圆核固定EI=0.875下向外EE补5点，共享条件后14个唯一条件；另复测左移及直接对照、向外增边与名义剂量接近的增权。两张新网络与3条500ms实际应用检查已通过；根据本夜8小时授权，[调度修订](/data/hfosp/topic4_sef_hfo/core_multiseed_response_curves_20260913/network_order_amendment.json)允许在旧批收尾时按资源最多提前运行6条，并逐进程树预留旧批及本批内存增长。旧批物理结束后上限18条，实际并发仍由内存决定。条件、种子、预算和物理不变。自动逐运行分析、参数曲线、固定分杆患者比较、多事件原生场及GPU旋转诊断接上；[实际进度](/data/hfosp/topic4_sef_hfo/core_multiseed_response_curves_20260913/status.json)不等同于已产生新科学结果。

新批次继承降阈值双核、限核随机输入、Z/M及空间OU关闭、GABA18ms与固定loss；不加入指定TB路线。参数效应以同图同噪声差及逐图方向/范围展示，不把事件混池成网络重复。108条完成后停在科学审阅点，不自动加下一轮或冻结Fig5。此次明确授权替代9月12日“续跑之后不再自动加第三轮”的旧限制。

当前资源调度采用[阶段内存修订](/data/hfosp/topic4_sef_hfo/core_multiseed_response_curves_20260913/phase_memory_resource_amendment.json)及[07:26并发补充](/data/hfosp/topic4_sef_hfo/core_multiseed_response_curves_20260913/early_ten_resource_amendment.json)：建图按9GiB估计，身份核对且进入1秒积分后按6GiB估计；保留18GiB进程树硬保护、40GiB派发余量和30GiB主机保护。旧续跑降至3条后，108条队列上限由8增至10，旧批结束且方向探针优先时上限16，此后18，实际由共享余量决定。原8条多网络轨迹全部保留，新增网络2711、噪声847402的原位置／左移配对；没有改参数、种子、顺序、时长或预算，也没有重启物理轨迹。

9月12日的[历史133条审阅](/data/hfosp/topic4_sef_hfo/core_shape_output_response_20260911/scientific_review_20260912/scientific_review.md)与[全部56条单参数响应图](/data/hfosp/topic4_sef_hfo/core_shape_output_response_20260911/scientific_review_20260912/figures/all_local_parameters.png)保留来源。圆核减弱核内E→I改善TA时差却丢失SCL；椭圆加同样干预可保留约81–85%的TA上部SCL参与；向外增边约50%也改善TA时序，但参与接近全覆盖。该批TB中ICL11先于ICL9的条件概率患者约69%，几个主要模型仅0–21%。不得借用形状/范围条件的确认结果替代局部参数候选自身的多网络复测。

增边条件另显示：两噪声的TA、TB各自时差中位数几乎相同，TA占比却从52%变到39%，使ALL时差中位数由负变正。需同时看模式比例和模式内分布，不能把总体变号都解释为传播机制受噪声颠倒。增大输入均值后“合格事件少”则主要由窗口重叠排除造成；全部检测仍很多。新报告保留这些替代解释、完整局部参数表、固定分杆患者对照及原生场，仍待用户人工验图。

**9月12日续跑已实际接上，合同和预算保持。** 最多64条新60秒运行：形状×离核EE×局部EI组合新增24条，两个有时序响应工作点附近的中心/范围探针24条，连同直接对照的新噪声重演最多16条；另复用12条历史直接对照。[续跑合同](/data/hfosp/topic4_sef_hfo/core_recruitment_tradeoff_followup_20260912/execution_plan.md)与[实时状态](/data/hfosp/topic4_sef_hfo/core_recruitment_tradeoff_followup_20260912/status.json)保留独立入口；新108条的提前并发按上方9月13日调度修订执行，不修改续跑提名规则或任何已有轨迹。

**2026-09-11 新增授权批次已启动：core形状与离核输出响应。** 此前168条完成结论保留；本次新批次单独记录，正式上限140条60秒轨迹：52条形状/输出范围、56条圆核与椭圆核下的局部连接/阈值/输入响应、32条两个新拓扑×两个新噪声确认。4条500ms应用检查已通过，圆核的静态数组和动态前缀与历史基线一致。当前实际进度读[实时状态](/data/hfosp/topic4_sef_hfo/core_shape_output_response_20260911/status.json)，不能把“已启动”理解为已得到双模式恢复结果。

本次比较原端点位置与左核上移3mm，等E成员数且匹配降阈值总量的4:1/9:1椭圆、正交形状对照、2.5mm半径及剂量匹配对照，以及仅左核E→核外E的横/纵范围和方向。局部重采样保留各源出度与总权重，与同规则原范围重采样作直接对照；目标入度与距离时延允许变化。两核E只降阈值、随机输入限核、Z/M及空间OU关闭的定义保持。扩大core也扩大随机输入支持，阈值剂量匹配不代表所有输入总量匹配。

自动分析与图件在[新批次报告](/data/hfosp/topic4_sef_hfo/core_shape_output_response_20260911/analysis/scientific_note.md)、[执行合同](/data/hfosp/topic4_sef_hfo/core_shape_output_response_20260911/execution_plan.md)。参数效果逐运行配对，分别显示参与、顺序、毫秒时差、局部宽度、原生活动范围与并行连通域；另加入旋转候选的占用时间、转角和轨迹数，不进入患者拟合loss。相位与环检也可能被非旋转波源叠加触发，不能自动认定为稳定螺旋。患者Fig2C真实STFT、模型发放包络和原生场多事件GIF保持固定分杆15行；完成后仍须科学审阅，不自动冻结或进入Fig5。

**2026-09-11 显示修订**：患者—模型对照、三个 TB 同步 GIF 及报告内对应旧图已统一为固定按杆分组的Y轴：上方 `SCL9→SCL6`，下方 `ICL11→ICL1`。全部15行始终保留，未参与触点用深灰标注，杆间/缺失处不连接质心线；相同通道在各例处于同一高度。新版图见[末批对照](/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911/main_review_final_B_primary_shaft_fixed/figures/README.md)和[同步GIF](/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911/tb_fixed_shaft_review/figures/README.md)。只改变显示，原事件、包络、STFT、参与定义及结论不变；历史交错行序图保留追溯，不再用作当前对照。

**2026-09-11：本轮实验与分析完成，患者双模式恢复尚未通过。** 既有120条加本夜48条正式运行全部完整，0工程失败/物理runaway；新增只涉及一个拓扑身份与两条噪声，没有新拓扑确认。不冻结工作点，不进入Fig5。

先看[11页关键图审阅包](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/overnight_key_results_20260911/overnight_key_results.pdf)，再按需查[完整协作者报告v8](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/model_collaborator_report_v8_2026-09-11/model_collaborator_report_v8.pdf)与[完整科学审阅](/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911/scientific_review.md)。每次PDF构建保留独立快照；[关键图版本信息](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/overnight_key_results_20260911/current.json)和[v8版本信息](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/model_collaborator_report_v8_2026-09-11/current.json)注明实际生成时间。Agent已检查关键源图及图件可读性，仍待用户人工目视审阅。

## 当前可接受的结果

- **局部参数确实影响网络。** 近SCL布局增强左核易激性，可改变右核完整活动并提高TA标签比例，但伴随ICL参与下降。核内EE的作用随布局改变；全检测和原合格集合的证据强度不同。
- **输入相关性影响出现机会。** 上移3mm、EE0.85、输入均值0.95背景，两核OU由共享变独立后，TA从5/103、4/107增至22/127、22/127；但上部SCL仍晚约91–93ms，患者TA中位约提前9ms。TB左端顺序概率也仍有明显残差。
- **最低分不代表恢复。** 中点布局、EE0.85、均值0.95、左核降幅1.15的联合训练分数6.563/7.287最低，但TA仍缺左端ICL与上部SCL。分数未接近零，也未将这些结果评为完美匹配。
- **少见相容个例存在。** 它们保留为能力线索，不能替换正常选例或解释成正确分布已恢复。下一版优先区分core范围/形状与向外连接的扩散范围，不继续只移动相同小圆或加密同一全局参数族。

## 当前模型与比较定义

当前物理见[共享科学口径](/home/honglab/leijiaxin/HFOsp/docs/topic4_patient_geometry_prior_snn.md)：患者端点几何提供先验；E-core阈值只降低或保持背景，随机输入仅在两核E，核外E及全部I使用确定期望到达，保留核外递归活动。空间OU、慢I、Z/M、定向刺激关闭，GABA18ms固定。核内EE/向外EE、EI/IE/II权重、边数、空间范围等分别解释，不强制总输入守恒。

**损失确实使用冻结的患者模式信息；不能称为“完全不看TA/TB标签”。** 实际 `L_search` 展开为0.25×全体事件联合特征统计／其冻结正尺度，＋0.25×模式条件特征统计／其冻结正尺度，＋0.50×参与组合统计／其冻结正尺度，三项均使用去自配对形式。模式项用患者FIT标签建立目标，再由同一个冻结rank聚类器分配模型事件；它同时涉及类内联合特征和模式占比。这些是设计系数，不是各项实际损失贡献占比，也不是恢复或解释方差百分比。神经元输入不读取TA/TB、细路线或指定事件时间，与参数筛选使用患者模式信息应明确区分。相同标签下的接触统计是拟合诊断；新拓扑／新噪声检验重复性，原生场与患者时间过程用于检验另一层观测，均不能仅凭两个标签升级为机制恢复。代码见[冻结模式目标](/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-continuous-core-state-r1/src/topic4_multievent_distribution_objective_v2_1.py)和[参与组合损失](/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-continuous-core-state-r1/src/topic4_joint_participation_mask.py)。

患者FIT为TA13,165、TB6,605事件；运行是实验单位，事件是运行内样本。参与、顺序、毫秒时差与空间过程分别展示均值/中位数、散布和联合结构，不先压为总体方差恢复率。原合格孤立250ms窗口规则未改变；全检测只是开发诊断，短长前缀重演不充当独立复制。

Fig2C主图使用实际参与触点和真实STFT，模型显示发放密度包络。全触点患者QC中的亮信号不能直接当作该群体事件起点：TB937的SCL8/9就是未参与的例子。两种信号的质心/宽度含义仍有差别，保持真实毫秒轴不等于频谱等价。

## 关键原始图与动画

1. [固定按杆分组的患者真实频谱与正常模型选例](/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911/main_review_final_B_primary_shaft_fixed/figures/README.md)；[同网四列布局](/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911/main_review_final_B_primary/figures/README.md)的连续读出原已按杆分组。
2. [配对参数响应概要](/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911/parameter_response_summary/figures/README.md)、[输入相关性到三项传播观测](/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911/core_ou_propagation_response/figures/README.md)。
3. [逐触点参与、顺序、时差和顺序概率矩阵](/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911/distribution_review_final_B/figures/README.md)。
4. [未经事件挑选的连续六秒原生GIF/MP4](/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911/continuous_native_final_B/figures/README.md)、[每类最早三个事件组织的多事件GIF](/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911/analysis_final_B/figures/README.md)。后者有窗口拼接，不代表自然事件切换序列。
5. [端点范围与core几何审计](/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911/core_prior_extent_audit/figures/README.md)、[核心活动与同窗接触读出](/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911/core_window_alias_final_B/coreOU_mid_EE085_mean095_rho000/figures/README.md)。

## 版本与维护

[本夜执行修订](/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911/execution_amendment.md)、[当前状态](/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911/status.json)、[物理与分析验收](/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911/physical_analysis_closeout.json)。夜间逐时过程已[归档](/home/honglab/leijiaxin/HFOsp/docs/archive/topic4/snn_overnight_progress_2026-09-11.md)，不再用旧“运行中”状态代替最终结果。

生成器为[完整v8](/home/honglab/leijiaxin/HFOsp/scripts/build_topic4_model_collaborator_report_v8.py)和[关键图审阅包](/home/honglab/leijiaxin/HFOsp/scripts/build_topic4_overnight_key_results_20260911.py)，均只读已完成产物，不派发实验。[系统搜索设计](/home/honglab/leijiaxin/HFOsp/docs/archive/topic4/core_connectivity_search_design_2026-09-10.md)保留提出时的预算与参数表，实际执行修订以本夜记录为准。

历史[v7](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/model_collaborator_report_v7_2026-09-10/model_collaborator_report_v7.pdf)和[v6](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/model_collaborator_report_v6_2026-09-08/model_collaborator_report_v6.pdf)保留手放/混合阈值、全场输入、C/Z及慢状态的各自身份。它们不自动继承新版物理；旧全降阈值对照和相容传播也不能被说成从未存在。
