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
