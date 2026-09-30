# 最新：十秒两历史和R1通过；唯一新科学批次为K9.5原生/模型配对

history_pair两个worker及analysis全部EXIT0，sessions43606/66945/68976已收齐。后窗fieldRMS高0.08605、不对称0.06383Hz，双core误差最大0.142Hz；前后窗所有5门PASS，两主/外源RNG末态配对。图v1实际打开发现模型1ms/native5ms显示口径不同及G1e-26放大，已平行v2统一20ms核心bin/G0-.1轴/外置图例，v2PNG真正目视PASS，humanPENDING；图session80701已收齐。科学readouts未改。

R1核验权威mean_single_replica_identity_v2/result.json PASS：40k×1000spike零差、refexact、八状态maxM1.38e-11、全局exact；会计原生方程/延迟/fullpending/外源共同counts，递归全自由。首版native100ms完整但正常终止异常跳过fast返回后的保存，native_record缺失；replay session61839在load前失败无任何模型步。原结果及failure记录保留。repair_single_replica_record.py新parallel_v2用finally落盘、核验原生完整末态与首版逐位同；observe98503/replay49313都EXIT0收齐。不是新独立seed。

当前运行 mean_boundary_correspondence.py：native --device0 PID64035/session11396；model --device1 PID64037/session45665。OUT=ROOT/mean_boundary_correspondence。仅一个K9.5条件，各10s，同完整72s高史，仅Kfield9.5/9.35，Zheld/G/Mfree/nuconst与9.35旧reference配对。model输出model/upper；native输出native/runs/high_history_K9p5_constant_background。检查progress再行动，别重复派发。分析器analyze_mean_boundary.py --wait/session70456已启动，等待native/result及model/upper/result带both_RNGs_paired_with_K9p35=true才读。末5s空间/core/drift/samepositive/quiet，以及first100msR<=5相差<=.25s门已冻结。完成需收结果、真正打开figures/mean_boundary_correspondence.png，不得先称fold。

解释文件replicated_network_definition.md定义有限R复制图w/R及R1端点；没有另外的Gaussian递归残余不等于有限R无波动。R64也不是原生64seed或证明分布极限。进入/返回/实际退出仍按各自native证据，GoalACTIVE/formalNOT_ESTABLISHED。代码仅scripts/topic4_loop_bifurcation，无subagent/push/更改正式图/原引擎。

下方为旧进度，以本段及实际result为准。

# 当前优先：一秒leading-mean通过，两个十秒历史检验和R1实现核验在跑

一秒结果dynamic_mean_input_pilot/analysis/result.json：全部4开发门PASS，尾fieldRMS0.175Hz（原19Hz），R约197不跨200，G保持原生近零，coreZdot约-.03483/-.03307。PNG已经目视PASS/humanPENDING。不是formal候选自动验收。

当前两个worker：dynamic_mean_history_pair.py run，high/device0/PID62875/session43606；asymmetric/device1/PID62874/session66945。恰好各10秒、40000×64、同一固定Z/K、逐cell固定nu/Poisson原law，完整72/42秒原生历史。0–5/5–10s同门固定。每100ms独立chunks，未完成勿将其作为最终结果。分析器analyze_mean_history_pair.py --wait/session68976已启动；等两条result后自动写analysis/result/readouts及figures/dynamic_mean_history_pair，需真正目视。

另外verify_mean_single_replica.py observe --device0/session93850在做100ms原生固定外源counts/spikes记录；新目录mean_single_replica_identity。observe后必须运行同脚本replay --device0（尚未派发），模型R=1只回放外源实际counts，递归自由，比较40k×1000spike和全部末状态。检查native_progress/result再派发，失败不重复盲重启。两gpu约4GB每个主worker，短观察共享device0。

GoalACTIVE/formalNOT_ESTABLISHED。代码仅scripts/topic4_loop_bifurcation，无subagent、push、改原引擎/正式Fig5。原恒定tau56.8秒完整、13brief跨度2.97s未达持续门；空间非全core起燃。下方旧运行语句均为历史，以本段和实际progress为准。

# 当前唯一新运行：leading mean-input单因素诊断

dynamic_mean_input_pilot.py run --device1已启动，当前工具session请查本轮最新结果。OUT=ROOT/dynamic_mean_input_pilot，PID从supervisor.json读。严格一条1s×40000×64，同72s完整native初态、原graph/threshold/delay、Z/Kheld/M/G动态、逐cell Poisson外源。沿用刚完成Poisson-external实现，只把添加的递归残余variance设0；源mean仍来自自己的逐cellspike概率、不规定period、不teacherforce，不改生物参数。主Gaussian与外部Poisson数值RNG末态需和前条逐位同。

动机是分离自由递归mean与已被证明失真的white residual，而非认定原生网络无残余波动。哪怕吻合也只是leading mean诊断；须独立状态/长程/有限noise对应后才能作为formal候选。先等完整result，不读在写npz，不自动延长/扫参/求根。分析器尚未启动，待写，复用固定1s native字段及先前同门即可。

下方“无worker”已被本段替代；全部旧native及其它tests仍结束。GoalACTIVE，formalNOT_ESTABLISHED。

# 再次更新：本轮所有任务已收齐，无worker

AR64准备session27948完整结束，但两路径representation_guard均FAIL；首次session94361在任何fit前因13677个AMPA目标谱恰为0而中止。零谱现在正确编码为零过程，旧失败/producer/6.4GB空谱缓存保留failed_zero_spectrum_preflight，门只对非零谱评且未放松阈值。最终AMPA谱L1median.924,p95 1.366；GABA .184/.306，均不能用于新的physicalmodel。AMPA少数近单位圆窄峰使离散频格方差评估极大，不能简单称真实variance发散；CPU Yule-Walker核验通过。这只是固定阶表示失败，不是原生机制失败。不要自动加阶或把AR64模型投入动态。

constant_tau_return_spatial_sequence_v2.png/svg已生成且真正目视：共同200ms/8个5ms帧，涵盖180ms事件；检测事件以外的trace灰色、frame标Context，解决弱事件之后新活动误归属和前版截断问题。v2显示不改变任何指标；图agentPASS/humanPENDING。session30552收齐。源第一次reference固定.57窗口，严格事件.58-.66；其Context现在明确。返回2/3有早期核外活动随后核招募，不能写成全都core起燃。

所有native、动态、分析、只读和AR准备当前都已完成/收齐。Goal仍ACTIVE，formal NOT_ESTABLISHED，下一步尚未派发新仿真。下方‘AR在跑’已过时，以此段为准。

# 最新优先信息：恒定tau原生延长结束；四条物理闭合仍失败；仅AR64表示准备在跑

Goal ACTIVE，formal bifurcation NOT_ESTABLISHED；不complete/blocked。本轮native/动态pilot都已完整结束，没有原生worker。当前唯一新GPU工作为prepare_colored_residual_memory.py run --device1，ROOT/dynamic_colored_memory_preparation；从progress.json查PID。只是输入残余的AR64数值表示准备，不是新的physicalmodel、根或正式分岔。工具session请在当前对话最新启动结果找；不重复派发。固定order64，不自动加order/参数/仿真；先审representation_guard。

## 已完成的原生延长
constant_tau_return_followup恰好56.8s完成，supervisor57343/worker57353/分析59106/画图60088都结束；sessions10034/77596/12899已收齐EXIT0。分析与raster/图全部完成，相关脚本review_constant_tau_return.py最后完成session71405也收齐。恒定tau0.5替代原式低率5s；前0–16.8原生前缀不受改动影响证明仍有效。统计是一条配对参数轨迹，不是新seed、不加入原式confirmedloops。568条外源记录逐位、8秒原始计数/raster、Z预算/计数/实际Ktau核验PASS。

两核reference25.48s，首brief26.47s；13brief/14finite、span2.97s、brief比例.9286。duration/IEI/globalpeak比1.2/1.1739/1.626全在旧.5–2门内，但span不足5s，29.87再次进入，sustained_return FALSE。后两次低活动后再次进入前均无双核reference；entries9.94/29.87/48.45/56.22，完整观测到56.8。不是完全无间期事件，也不是持续返回通过。

原式首次49.31返回前coreZ.99851/.99850、K.016873；恒定tau26.47前coreZ.78754/.78595、K.003426，29.87再进入前coreZ.70263/.68737。支持达到参考与恢复余量不同；不能说Z余量是唯一时长中介，因为K/G/M与对应时间外源也不同。见scientific_review/review.md、result.json。

原样raster/率/Z zoom和全56.8trace已真正打开。figure_review/result.json agent布局PASS，humanPENDING。空间序列constant_tau_return_spatial_sequence.png也打开：第1B→A，第2/3早期可见核外活动再招募核心；weak远处活动。不能将11双核参与说成11核起燃。原storyboard固定150ms窗口：第2总180ms只显示前150ms；weak总90ms，后段可见下一事件context，未计入weak指标。此显示限制已写review/JSON，可进一步做带事件边界标注的v2，保留原图。没有替换正式Fig5。

## 四条物理时间闭合，均不准进入正式分岔
1.dynamic_individual_source_pilot两个1s×40000×64，group_source与individual_source：相同完整72s高态、原图完整delay、精确threshold/逐cell固定nu、Z/Kheld、M/G自由。源不同之外全部配对。尾fieldRMS19.03/18.84Hz，R约201.4/201.3，native约197且G0，两个模型错误启动G；core均率看似近不够。二者COMPLETE。原先外源J单位assert与实际jump混比在仿真前fail，已存failed_preflight_external_units；正确换算J*tm/tr，物理数值没改。分析首次40×40/operator与20×20/native显示索引误配，改用已核验ADAPTED显示mapping，并assertcellgroupsize身份。无native/model重跑，修复日志保留。
2.dynamic_bernoulli_source_pilot：只把递归variance r改为r*(1-.1r)，其余上一fullsource一样。1s完成，主RNG末态逐位配对；起始更近但尾fieldRMS18.899，R仍>200/G仍启动。不是闭合修复。
3.dynamic_poisson_external_pilot：在Bernoulli递归variance基础上，仅把外源Gaussian改为实际Poisson计数，主GaussianRNG相同、独立external RNG。原native_cell不改，通过等效nx令实际rise增量=Gaussian recurrent+原jump*Poisson，CPU逐步误差1.42e-14/spikesrefs一致。1s完成，尾fieldRMS18.895，几乎无效。三张对应图PNG都真正打开，agentPASS/humanPENDING；native gate科学均FAIL。所有sessions都收齐EXIT0。

## 时间记忆：两项只读审计已完成
source_refractory_memory_audit/result.json：native原72–74s40000源spike，最小ISI核验；p(t)边际概率意味着E源lag1..19严格不能jointspike，所以输入残余必须有−p(t)p(t+lag)协方差。独立白残余丢失它。不是说density细胞自身违反refractory，也不是单凭该恒等式证明全部rate偏差源于它。native22step相位分解只作诊断，不给未来新模型规定period。E残余短lag相关约−.045至−.055，I类似；2.2ms仍有余相关。
filtered_residual_variance_audit/result.json：已验证native phase残余PSD经过原graph W²及同discretefilters，全E目标独立colored IE/II residualvar3.582/1.518mV²，保持同一步var却white化变185.60/220.25；finitezeroDC调整仍184.83/215.50，数量级不变。原59误差诊断target实测残余8.651/5.128，colored independent5.899/2.677，white269.94/335.74。所以source residual crosscorr仍缺，不能只修时间相关就宣布通过。两个只读脚本session90918/60642收齐EXIT0。

## 当前AR64准备与后续科学方向
prepare_colored_residual_memory.py新脚本在跑：缓存2×40000×10001 projected_current_residual_PSD.npy，原native phase源PSD+原jumpW²+原filter，无参数拟合rate。每target/pathway固定order64 Yule-Walker/Levinson，0.1ms步；CPU Toeplitz独立核验前6targets，稳定reflection<1与positiveinnovation，记录谱L1/variance/shortcov errors。两路径谱L1median<=.05且p95<=.15才retained表示，不是native correspondence。不会自动动态模拟；order不自动增。核验是否完成/失败，不要在输出NPZ正在写时读，等result.json。

若表示本身过不了，先按误差结构判断是否适合；不要为过门盲加order。若过，只允许作为下一物理时间输入统计假设：均值由逐cell源反馈自由产生，不固定22period；残余cov暂用native输入仅是development anchor，后续必须改用自己的输出统计验证self-consistency。必须明确新噪声memory初态如何与native膜/突触/排队完整状态相容；不能把stationarycurrent直接叠到已有current造成初始重复噪声。也不能把固定谱近似当已闭合，更不能直接认证稳定性。原生候选/正式Fig5及density_spatial.py不变。

## 文档与权限
当前主说明physical_time_source_review.md，mechanism_model等5文件顶部已更新本次最新状态；旧阶段记录保留，下方RUNNING为历史。ROOT=/data/hfosp/topic4_sef_hfo/fig5_loop_bifurcation_20260927。PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python。代码仅scripts/topic4_loop_bifurcation，无subagents、push、清理、改正式Fig5/原生engine/density_spatial。Memory324–325和两rolloutcitation要求见旧handoff或上下文。

--- 以下为上一阶段交接，均以本页顶部新状态为准 ---

# 最新优先信息：中介对照全部完成，仅恒定tau返回延长在跑

Goal ACTIVE，formal bifurcation NOT_ESTABLISHED；不complete/blocked。以下旧LIVE段是本次早期记录，**当前唯一原生worker**为constant_tau_return_followup：supervisor57343、worker57353/device0，tool session10034。脚本continue_constant_tau_return.py。恰好一条26.8→56.8s（30s），从已完成fastK完整状态续演；方程是两段Ktau均.5s，原生Z/M/G/R/输入/延迟自由演化。不重复派发，不自动新种子/再延长。prepare session17254完成收齐。无分析等待器，下一步应为此续演写收集器，将16.8–26.8旧fastK段与26.8–56.8新段按原时钟连续拼接，沿用原strict事件及原8s参考标准，重点>=10brief/跨度>=5s/brief比例>=.8/持续时间-IEI-global峰值比.5–2，空间传播另验；不把单个事件当通过。

## 最新等待器与后续闭合入口（补充，优先于下面旧信息）

analyze_constant_tau_return.py --wait 已实现并启动，tool session77596，PID从return_analysis/progress.json查。它等唯一56.8s续演完成，拼原0–16.8s不变前缀、fastK16.8–26.8s及新26.8–56.8s，全时钟/区域计数/field/Z预算/568条外源输入逐位核验；调用原temporal_audit和strict_events、原core参考及原reference_features评每个恢复窗，输出constant_tau_return_followup/return_analysis/result.json/readouts.npz。需实际检查是否运行完成及有无错误。结果中的空间审阅仍PENDING，后续需选前后原生场和raster，不仅看均率。上文“无分析等待器”是更早记录，现已被本段替代。

已有native reference_features为duration100ms、IEI230ms、allEpeak69.49375Hz、coreA/Bpeak360.08/367.94。同一示例的视觉范围不同不自动意味着返回整体不合格，仍按预定事件分布及原生容差判断。

下一物理动态闭合可评估：保留原TargetNetwork逐细胞V/ref/M/突触/pending和精确threshold，源由3479组平均改为40000个原物理cell的逐步复制平均spike，再用完整delay operators自由递归；G保持动态。不能再固定22步输入周期来判稳定性。原target_density_exit.py已经有完整初态（别误认它重置了V）；它丢的是源细胞身份/相关时间结构。新算子可复用target_delayed内核，flattenP改40000；观测/全局率仍可按3479组汇总。物理kernel还含独立whiteGaussian源近似，算子精确不代表这个噪声近似通过；需配对局部/动态对应，不能直接宣称修复。现**没有**新dynamicmodel仿真启动，勿虚报。

high_history_spectral_value.py中MATCHED为native_K9p35_constant_background_pair_v2，NAME high_history_constant_background，已有72–82s原生；parameters.npz包含原精确theta、nu及V/ref/M。observe_high_history_spectra.INITIAL是原72s完整高态checkpoint。若选择条件动态对应，应沿用同实际Z/K场、完整初始物理状态及逐细胞固定外源，与原同条件原生比较，再评估近边界。不要把任何开发anchor响应直接升格正式分岔。

**中介全部完成**：修复session30758、分析59885、原绘图76738完成退出0；原supervisor31787因旧failedchild退出1已收齐，修复supervisor权威COMPLETE。baseline/G一次移除/fastK首次低率16.868/17.921/16.868；两核reference23.9/24.28/25.48。fastK前几次高活动期间Z不足，但最后仍恢复，否定‘5s保留是充分Z恢复必要条件’。不能只沿用中途前缀推断成fastK完全无法恢复。

fastK19.939s q再开，coreZ.469/.458；26.47s首个恢复后完整120msbrief。event_metrics为B先5ms、A35ms；coreB/A峰419.85/402.39Hz，other109.84，core/other3.82；25/50/100Hz三阈值均有core先于other。PNG显示B到A及向外传播，较原0.57s固定示例更宽。仍须持续返回与条件形态检验。sourceprefix只读证明：K首次非零9.874s，此前记录K全0，之后R连续下界13.864>5直到16.8；故改低率tau从t0不会改已有前缀。文件natural_exit_mediator_probes/constant_tau_prefix_diagnostic.json；新参数轨迹不计新增独立seed、不替换原已认可源。

**新图已真正打开**：native_exit_upper_history_v2（同K9高史的K9.35活动/K9.5静默）；natural_exit_mediators_v2（5行×3列，独立最初.3秒对数R放大，避免v1inset挡住后续burst）；constant_tau_first_return_spatial（旧布局2行原间期/新事件，6帧5ms空间）。agent PASS，human PENDING。原v1图和producer保留。新分析natural_exit_capture_review.md、capture_review.json；review_natural_exit_capture.py、plot_constant_tau_first_event.py。

**精确动态源算子准备完成**：prepare_dynamic_source_operators.py，session95980已齐；dynamic_individual_source_operators/result.json。40000源/target、D358、AMPA32M/GABA8M边，第一/平方权重聚合回旧源群4项误差全0。完整时延，不是mod22。仅算子，没有自由动态模型。下一动态闭合若使用此算子，需让源相位/频率在物理时间自主产生，并验证残余噪声假设；不要把固定22步统计map当物理稳定性，也不能直接将W²当已验证白Gaussian闭合。

五份当前科学文件头、mechanism_model末节和snapshot已同步。正式分支目标仍欠物理动态对应、稳定性和相关分岔认证；不因有新条件图或这组对照结束就goal完成。下面保留阶段内详细排错和旧资料索引。

---

# 当前交接：原生上端点完成，实际退出的两项中介对照正在收尾

Goal ACTIVE；不得把有限条件图或本批运行完成当作完整分岔目标完成。正式分岔 NOT_ESTABLISHED。代码仅 scripts/topic4_loop_bifurcation；cuda_env 解释器。原生引擎、density_spatial.py、正式 Fig5、已认可状态空间和他人工作不变；无 subagent/push/清理。完整前阶段信息见 HANDOFF_before_natural_mediator_controls.md。

## 活动任务，先查状态，不得重复启动

- natural_exit_mediator_probes：原 supervisor PID55189/tool session31787；G一次移除 worker55728(device0)，10s从16.8至26.8。原生 complete20s完整续演与观测逐位PASS，门为 full_state_gate.json；qoff实际 R194.911/G11.9675/K12.6894/meanZ.20918。不是旧20s回溯exitcheckpoint。
- 同批 fastK worker55729在任何仿真步之前因 checkpoint class kind ConditionalSlow vs NoRetention 不匹配失败。原代码未编辑。retry_low_rate_mediator.py 将此条checkpoint的类名元数据改为NoRetention，全物理状态逐位不变，原失败log/progress/runtime/checkpoint保存在 failed_low_rate_before_first_step/。修复supervisor PID56037/tool session30758，retry worker56083(device1)。仍同一条10s对照，只把 R<=5 的Ktau5s改为.5s，初态K和G不变。early_retry_kinetics_qa.json已在真实前1200个1ms记录中验证1131个整段low区间的K指数式，误差3.55e-15。
- 原supervisor会因旧失败子进程返回1而最终FAILED，这是预期的保留记录；实际修复完成状态以 repair_supervisor.json为准。修复supervisor等待两条完整且wrapper干预metadata完成。不要把原supervisorFAILED当成需要重派已完成G任务。
- analyze_natural_exit_mediators.py --wait：新tool session59885，旧54929/PID55661只读等待器已明确SIGTERM退出143，用新版本识别repair_supervisor。最终输出 natural_exit_mediator_probes/mediator_analysis/result.json 和3个npz。读取baseline16.8–26.8做过真实验证：原firstR5=16.868，bothcore reference23.9，budget和K式PASS。spike计数是1ms中心，mechanism是左端，已明确半ms对齐。
- plot_natural_exit_mediators.py --wait：tool session76738，等分析完成生成 figures/natural_exit_mediators.png/svg及README。尚未产生/目视，必须真正打开。该图有3条件×4行，含最初0.3s退出放大。人工PENDING。

## 本轮已完成

1. native_K9p5_held_history唯一30s任务和自动比较完成，sessions49668/8490全部收齐。K9.5同heldK9初态确实静默；同初态K9.35持续双核约473/474Hz。配对300条futureinputs完全一致。firstR5(K9.5)为分支1.253s；K9.35最低约169.4Hz，未进入5Hz段。只支持有限条件变化，尚非fold。
2. native_exit_upper_history.png初版实际打开，图例压到了K9数据点。plot_native_upper_history_point.py现在写平行_v2输出，原版producer/图/结果保留。figures/native_exit_upper_history_v2.png/svg已生成且PNG打开自查PASS，SVG XML PASS，人工PENDING；result和figure_qa位于native_exit_upper_history_figure_v2。有零值重合标注，不画未认证stable/unstable曲线。
3. phasecarry三轮完成（全部40k×64）。末轮空间fieldRMS.1321Hz，E phasevar.0062545/native.0068747；E/I率更新RMS.2419/5.1987Hz，非root。修复了大部分reset引起相位损失，但不作物理稳定性解释。phase_history_carry_review.md已写。不要自动追加静态根或damping；先看与真实退出的关系。
4. feedback_build_up_review_v2 两pairedseeds原生既有数据审计完成；old v1失败保留。prefix QA在连续qoff证明区逐位核对，第一处q>0的1ms样本可能已含反馈效果，不能纳入未反馈前缀。参数差只有name/global_tau_s，20s外源状态/记录inputs逐位同。延迟G首半秒允许更高率/K峰，但两秒平均与末K并非总更高。8403在14.236s即时R266.48,K4.7887；延迟R4.8788,K4.8166。因此单时刻meanK不能解释退出，仍需时间与空间历史。feedback_build_up_review.md已写。

## 关键新前缀观察（待完整对照确认，不作全窗结论）

去掉16.8s已有G后，第一段qoff最低R=9.18356Hz在16.941s，K9.57126；因此没跨5Hz保留段，K继续.5s快速衰减。17.336s q重新开启，K仅4.34391；G再积累后17.921s首次R<=5并持续，随后继续Z恢复。原baseline16.868s即跨R5。说明G尾迹帮助第一次直接捕获低率段；不能说一次移除G就永远无法退出，因为允许G重新生成。

fastK初次R5与baseline相同16.868；当前约20–21s已有活动/新的K积累而meanZ仍约.49，核心是否回到参考等待完整读出。不要把提前回来的一点活动当成间期恢复通过。干预不计自主闭环。

## 后续完成步骤

等两条各26.8s原生完整及修复supervisor；收分析与图并查任何错误，不重复native。核对配对外源/实际tau/整段budget、coreZ参考、反弹/事件及截尾。添加 firstq-reopen前minimumR及连续R>5下界，明确G一次移除仍能再生成。更新机制解释和当前文档，不能只停在旧静态K阈值；同时不要宣称两条机制对照等于正式分岔。下一阶段只推进与真实轨迹相关且物理对应的分支或动态闭合，固定22步统计map不能决定自由频率或稳定性。

现有自主原式、15片段/3种子恢复界、空间条件场/轴控制限制等见 mechanism_model.md 与前handoff；旧章节带有早期RUNNING语句，顶端最新阶段及实际result权威。当前图都候选；未替换正式Fig5。

Memory used MEMORY.md324–325；final最后按规定citation，rollouts01a09eae-c163-7cf2-8f2d-f11d43bdeaaf与01a0add1-6bb8-78f1-8af3-98dc7b0724b2。无memory写授权。
