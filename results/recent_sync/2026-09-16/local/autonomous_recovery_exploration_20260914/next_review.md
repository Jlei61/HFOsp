# 下一次goal延续入口

本轮goal仍active，截止epoch1789433430（北京时间9月15日08:50），不要因为第一批已派发而标记complete。用户要求9小时自主探索；尚待实际结果、逐轮反思和确认。

1. 先读status.json、runs/*/progress.json、analysis.json、monitor_failure.json（如存在），检查supervisor/monitor进程。初始supervisor PID81113，monitor81114；进程可能变化，以实时查验为准。第一轮12条件最多12worker，与旧log-M扫描合计24，不能影响另一个Fig4工作树的任务。
2. 零效应及断点续跑检查qa.json已经PASS，两个噪声1秒raster/计数/场/Z/M/输入/接触电流逐位相等。不要重复跑原生QA；只有新物理修改或实际错误才追加针对性检查。
3. 当前runner被冻结producer_sha；不要修改仍运行的runner及原生源文件。分析器可修正。新机制需要独立v2脚本/模块和相应检查。主模型只包含native/mix/add，无外部干预；I细胞原生且M=0/Z=1。
4. 第一批结果到达后，目视对应figures/fig5_dynamics.png并审查：Z是否耗到0；全局/核心是否仍高率；是否真实反复爆发而非500Hz饱和；mix相对add是否因总量或Z消耗改变。300s旧扫描将陆续完成，补齐原E图并审阅，不把未进入称为永久稳定。
5. 新发现已保存saturation_current_audit.json：旧native弱M及η.02τ2进入后约60s，全E接近500Hz，Z几乎0，IE约1580、rawII约1775但effectiveII约.03，M约2.5/20。原生M加强又会在进入前压制间期。这是选后续机制的关键冲突，而非简单延长时间就能解决的保证。
6. 本轮mix=(1−gamma)II+gamma meanII，总原始抑制匹配；add=II+gamma meanII，剂量增加。Z更新用实际deliveredII，全局项也受Z抑制耗竭。Liou源代码有全局平均投射、同一Z作用总I、独立sAHP；其电导/瞬时I与本电流显式I模型不同。旧M4除法池只bound并未terminate，不可直接宣布成功。
7. 若候选有返回再入，先第二噪声9108402验证，再新噪声和机制消融。若所有候选都失败，应基于此次真实失败设计有限第二轮（总新增上限48条；留至少90分钟收尾），不能盲目重复同一网格。新受保护全局抑制/非线性池/电导改变必须作为明确新增机制，不能声称是原生ZM或Liou原样实现。
8. 当前自动图是diagnostic版本，3D慢量每20ms采样较粗，不能作为最终清晰轨迹图；若有候选，应在同一seed同一物理重演较密读出或用完整候选作图。末段当前可多保存最多一个10s检查点，display_stop已在secondconfirmation+2s，最终图需按它裁切。真正交付仍参考用户既有Fig5格式与raster放大，必要时配多次事件/局部率及原生空间图。
9. 到时完成科学审阅和具体图件，诚实报告成功/失败与有限时间边界，goal才能complete。未成功不等于必须无限调参。未经用户目视不能称人工验收、冻结模型或替换正式主图。

## 最新执行更新（北京时间约01:15）

- 首轮12生产任务全部仍运行；10–20秒完整前缀已分析，未看到自主返回。mix gamma=.5/.75在启动后很早持续招募；native_midM/fastZ及add分支当前未进入；不要只看mean-rate门，应检查前期是否有多个有限事件。
- 第二轮脚本run_topic4_fast_threshold_recovery.py已经冻结，QA PASS；supervise_topic4_fast_threshold_recovery.py PID82292在按空闲名额派发8条。tau_phi100ms、jump.25/1/2.5mV，gamma0/.25/.5。部分任务仅有10秒数据；先看实际读出再决定第三轮。
- 新脚本run_topic4_activity_global_pool.py与activity_global_pool_round3目录已准备、QA PASS，没有生产派发器。它新增因果群体率R_G和J_G=k*[R_G−r0]_+，Z仍作用在全部局部+全局输入，局部M保持弱M基线。待审8条（r0=0/50Hz，k=5/10mV-equivalent per Hz，tauG=.5/2s）。正式启动前写实际证据审阅；应与R2协调共享24worker上限，避免两个派发器同时抢剩余名额。
- 原log-M supervisor70996已正常移交给82494（supervise_topic4_quiet_backend_resume.py）。已有4/6强M低活动任务从40秒完整检查点转CPU，另两个等待40秒。新worker为run_topic4_quiet_cpu_resume.py，argv含原producer路径，现有资源计数识别它们。CPU/GPU 200ms实际低态续跑全脉冲及整份checkpoint（含空间OU）逐位一致，CPU约快2.4倍。勿误报旧PID停止为仿真失败；看backend_migration.json和实时状态。
- 当前分析器修复了：第二次确认+2s裁图；没有前后安静段的峰不标Interictal；entry空间采样改为首次细胞持续招募附近，避免整体率门已经很晚；新机制单独added_feedback图。3D仍是20ms采样诊断，正式Fig5应有清晰core raster放大和更密观测。
- 条件性的AHP电导方案仅在mechanism_review_pending_round3.md设计备忘里，没有实现/派发。当前优先审查全局活动池，因为它能直接检验全局反馈是否可兼顾前期事件和终止；不能把备忘草案当成结果。

### 后续更新（约01:30）

第三轮QA已通过，现已正式派发单个sentinel pool_k10_r50_tau2_s9108401；控制器supervise_topic4_activity_global_pool.py PID83756。其余7条不自动派发。dispatch_review.md/dispatch_authorization.json记录依据和单个额外进程的资源修订（最多25；该1条新启动需120GiB可用，原80GiB保留；机器80CPU且剩179GiB，GPU各剩13GiB）。以实时PID为准。若该条有价值，再审阅其他条件和第二噪声；若修改approved_names并重启第三轮控制器，额外实验必须等第二轮pending为空，并按总24worker计数，避免和旧第二轮控制器抢名额。

新增analyze_topic4_autonomous_events.py：在高态门之外，检查有限事件是否真正回来了，并单独报告Z实际向上变化，不要求Z必须回升人为设定的幅度。首轮gamma=.5/.75的四条在首次进入前的有限事件数=0，不能作为有积累过程的理想候选；其他原生或add分支保留多次有限事件。该审阅不把患者TA/TB标签套到模型事件，也不改变任何物理或原高态门。

## 当前执行修订（北京时间02:05，替代上述早期资源/PID快照）

第三轮现由 `supervise_topic4_reviewed_global_batch.py` PID84002接管，保留原sentinel状态，已批准4条κ=10的r0=0/50Hz × τG=.5/2s对照；κ=5四条仍未派发。总生产上限28、GPU上限24、新增派发需主存余量120GiB；现28生产中含6条已完整迁移CPU的旧log-M任务，GPU22条。原物理执行器及在跑任务不变，总新增科学条件仍为24/48。实时状态和activity_global_pool_round3/dispatch_authorization.json优先于本文件旧快照。

旧log-M的6条强M低活动任务均已从完整40s检查点迁移到经整份状态逐位验证的CPU实现，现继续约100–130s；36/42新条已完成，另6条复用。原第二轮6条运行、2条等待旧任务名额。当前尚未确认自主恢复。

新增观测版 `record_topic4_global_candidate_native_fields.py` 已通过1秒全脉冲、全部旧观测与整份末状态逐位一致检查。它在选定时间窗以原生10kHz保存电极电流幅值代理和直接1mm细胞场（不经电极投影），并以1ms保存Z/M/实际抑制与全局反馈。仅QA已执行，正式候选重演尚未派发。原legacy readout是|IE|+|II|，且II在乘Z之前；不能把它的未滤波二阶矩称为1–150Hz能量，也不能把它的全局抑制亮度误当作神经元招募。最终候选需先低通/带通再降采样，展示实际乘Z后的代理并明确单位。

有限事件审阅另保存真实持续高段和双核/全局安静区间，检查滚动低窗确认点是否落入下一次高态。它不改变原操作性检测门；若时间重叠，必须审阅实际轨迹，不能只凭标签宣称高—恢复—高。

## 20s完整窗口审阅后（北京时间约02:40）的最新状态

第三轮控制器现为86884，已无重启地接管原4条。`joint4_review.json`批准了4个新参数组合，已加入protocol/jobs并在资源门内排队：κ50/τG2/中等M、κ50/τG2/弱M、κ10/τG2/中等M、κ50/τG.5/中等M。中等Mη=.005/τM2s，弱Mη=.0005/τM1s。由已有κ10/弱M/τG2和新增前三条形成κ×M的2×2；最后一条检验τG效应。这取代`global_pool_joint_review_pending.md`中更早的暂拟τG×M布局。新物理状态或耦合没有变化，仍Z作用全部GABA、同一20mm手放双核、同一9108401噪声，60s上限；总新科学条件28/48。当前控制器8approved、12prepared（另κ5四条仍不批准）。

已目视`activity_global_pool_round3/figures/global_feedback_comparison.png`：r0=0主要抑制进入；r0=50的两组在约10s后双核持续、随后全场高率。`global_depletion_coupling.json`用5msRg的严格因果下界，验证了496/499个20ms区间内全局输入单独使每个E的Z按原生Euler规律下降，平均/双核误差约3e-15。当前不是恢复，只是推迟高态。原4条仍跑满60s观察上限。

旧η.005/τM2s的配对噪声原生控制已经找到：`m_parameter_modes_fig5_20260913/runs/e0_t1_s9108401`与seed2。底物identity和全部冻结依赖hash与本轮一致，首次进入11.72/13.21s；人工Z恢复分别在71.93/73.42s，故0–60s可作原生负对照。记录为moderate_M_prior_control.json。新中等M条件前10s保存后，应对照该旧控制的完整spikes_1ms/raster及抽样到20ms的Z/M，确认新增G起效之前轨迹确实一致。不要使用外部Z恢复后的段落作为自主返回。

新增分析：`analyze_topic4_autonomous_events.py`现在另列高段/严格双核及全E安静区间并标出滚动确认重叠；合成边界检查PASS。`analyze_topic4_global_depletion_coupling.py`已经在实际20s数据验证上述资源因果链，且新控制器每次分析会自动调用。`analyze_topic4_native_band_energy.py`从已通过源轨迹相等检查的10kHz原生字段计算相同长度Hann谱能量（去DC、1–150Hz），合成带内/带外/DC及缺窗检查PASS；尚无正式候选重演与能量图。它是模型观测诊断，不替代临床Fig3算法。

首轮addγ.5已完整60s完成：没有进入，末10s全E2.43Hz、双核8.42/8.69Hz、Zmean.949；完整60s检出180个有限模型事件。图已目视，保留核内短事件但不是自主终止候选。所有旧/新正式任务仍需各自完成，尚没有自主恢复确认，不能标goal完成。

## 北京时间03:30后更新：资源门与第五轮候选

第三轮86884当前总生产门30、GPU24、新增需120GiB可用RAM（resource_amendment_30_total.json）。4条joint任务已经全部派发；第四轮87182继续等总数<28，优先派发原机制gamma.25/etaM.05或.1/tauM2s两条，不改变原方程。总已批准科学条件30/48，尚无自主恢复。旧log-M六条CPU任务进至约200–230s，其余36新条+6复用已完成。以上数值是快照，须读实时status。

新第五轮脚本run_topic4_continuous_resource_recovery.py只准备3条且正在逐条QA，未批准生产。它明确改变Z方程：tauZ dz=b-z+rho(1-b)(1-z)，b=1[J<Ith]。rho=0严格回到原生；rho.25/1的高输入平衡分别.2/.5，弱输入恢复仍相同。条件为kappa50×rho.25/1和kappa0×rho1，均中等M.005/2s、tauG2s、r0=50、60s。已有原生中等M和kappa50中等M是rho0对照。没有外部复位或按事件触发的项，没有保护全局GABA不受Z耗竭。

这是检验高输入时仍有资源补充的新现象学假设；Liou原始氯方程3有持续清除，但其通用方程8与原生阈值Z相似，故不能说本修订是Liou原方程、氯模型推导、原代码错误或恢复所必需。修改同时改变高输入平衡和松弛时长，须保留这一解释边界。先完成零rho完整状态相等与非零rho续跑QA，再按实际证据决定小批派发；不得抢第四轮等待名额。

native_event_extent分析显示：基线0.2–8s的34个有限事件，在100Hz局部阈值下，中位事件总招募面积65.75%，最大同时活动面积26.625%；传播本来可遍历广域。不能只以面积宽窄替代持续性/安静间隔来判发作。native_weak的0.5–2s原生GIF及12帧contact sheet已生成且目视，包含多次事件且未经电极投影；不是患者匹配验收。

已实际完成中等M来源交叉验证：audit_topic4_moderate_pool_prefix.py 对 kappa50/r0=50/tauG2 的前10s，所有global/regional/400-cell counts、固定raster、Z/M、raw/effective currents、legacy contact proxy和noise summaries均与旧原生中等M逐位一致；前10s保存的全局反馈均0。其余中等M条件待首10s块保存后同样运行该脚本。

第五轮只准备+QA，若当前kappa50原生Z条件已达到自主返回/再入，应优先确认原生候选，不必为凑条件派发新Z方程。若其20–30s实际数据仍只有预防进入/持续平台/资源耗竭，再审阅是否派发三条机制诊断。已写controller但未启动，且它要求qa.json PASS+dispatch_authorization.json，等待第3/4轮pending为空、总28/GPU24和120GiB余量。

第五轮QA已完成PASS：rho0的所有原观测、全部数值状态及RNG与原生全局池逐位相同，唯一差异是slow.kind类名；该差异已显式核对，不属于物理差异。rho1的分段续跑与全程末checkpoint及所有观测逐位相等。常输入解析Euler松弛、I-cell Z1/M0也通过。验证器修订只处理预期类名元数据，物理class/worker/prepare/qa的AST均未改变，保留前版和修订记录。未派发生产，等待kappa50原Z实际20–30s审阅。

注意：现有record_topic4_global_candidate_native_fields.py只针对R3 GlobalPoolSlow。若R5新Z条件有候选，不可直接调用它丢掉rho；需用R5物理类作独立记录适配，并验证全部原始观测/数值状态相等。R5尚无生产。

## 约03:50之后：第五轮正式三条已批准

已目视kappa50弱M20s的fig5_dynamics与added_feedback：9.9s后双核持续放电，全局均率暂被压低但不断上升；首次高态门约20s达到，Zmean20s=.09829，无自主返回。全局项单独驱动耗竭9.935–19.995s，502个20ms区间解析Euler衰减匹配误差3e-15。仍不能宣称原Z中等M失败，它的12–15s前缀在继续。

基于该路径证据及全部QA通过，第五轮三条预设rho新方程测试现已写dispatch_authorization.json并启动controller PID89173。总新增科学条件33/48，生产上限总28/GPU24、RAM120GiB，之前第3/4轮pending为空。三条rho1/kappa50、rho.25/kappa50、rho1/kappa0，均中等M.005/2s、tauG2、r0=50、tauZ5、60s。仍优先验证任何先出现的原Z自主候选。不要把新rho成功叫原生ZM成功。

新增图审查修复：Rest不再可能位于所选Interictal事件之后；High取首次操作性高段内部；Recovered优先取实际恢复窗口中的低活动点，避免默认放到下一次onset之后。figure_metadata记录是否时间严格递增。已经完成旧图不会自动全部重画，最后交付前按所选条件重画并目视。

新版空间Entry取首次原生局部持续招募邻近窗口，可能显著早于全E200Hz/200ms门；这是空间发展与全局高态的不同读出，不要混为同一个onset。当前20ms慢量3D仍为诊断采样，最终候选需密记录。

## 北京时间04:15：当前已核查的进展

R5密记录适配器 `record_topic4_resource_candidate_native_fields.py` 的1秒QA已PASS：实际ResourceRecoverySlow类保留，全部旧观测、额外resource/pool轨迹和整份状态/RNG与源仿真逐位相同。结果在native_field_candidates_resource/qa_full/observation_qa.json；正式科学候选尚未重演。后续R5必须用该适配器，不能用R3母版丢掉rho。

R3中等M三条前10秒与旧原生中等M控制的全部已保存观测逐位一致，新增全局项当时为0。此后到约17–24s，三条都已达到原高态门，目前仍未返回：k10/tau2进入15.29s，k50/tau2进入21.03s，k50/tau.5进入22.89s。k50弱M进入19.58s。必须继续各自60秒，不将暂未返回称永远不能返回。

`regional_current_balance.png`已经目视，重要现象是全网均值掩盖核内驱动差异。k50弱M14–15秒全E平均IE/有效II约315/525，但Core A约1251/730、439.8Hz；所以全E平均抑制大于兴奋并不保证核内已受控。区域均值仍不是逐细胞阈值条件或Jacobian证据。

`fast_threshold_round2/fast_threshold_drive_review.json`记录实际效应大小：phi1/gamma0完整60秒维持有限事件、末5秒全E14.4Hz；phi2.5/gamma.5在15–20秒已有平均phi约63.8mV，而IE约812mV-equivalent、有效II约57.8、全E255Hz，未终止。Liou电导模型的阈值跳幅数值不能直接视为本电流LIF的等效强度。没有由这些均值推导分岔。

第四轮原方程更强M的eta.05/.1两条也均已进入（4.52/12.87s），当前前缀尚未返回，仍继续60s。旧对数扫描已37/42新条完成+6复用，5条CPU继续，需完成整表和原Fig5交付。

## 北京时间04:27：旧对数M整轮完成

42/42新条+6精确复用=48/48完成、无失败；endpoint_audit全部42新条从实际10ms全/分区计数核对PASS。24格每格两个种子，其中13格两条都进入、11格两条均300s删失，总26进入/22未进入。完整grid及两个seed的原A–F Fig5已重绘并目视，metadata标agent PASS、人审仍PENDING。路径仍clean_panels_v2/eta0.0005_s9108401和seed2。新scientific_review.json指出该网格主要是约10–13秒进入或300秒不进入，不是平滑时延梯度，也不测进入后的终止，更非严格分岔；tauM同时改变时间常数和持续M电流强度。F为明确标注的未滤波proxy二阶矩增量，不冒充1–150Hz临床能量。

R5resource_balance新诊断图已在rho.25/k50前10秒生成并目视；实际平均native drift与(1−超过阈值E比例−meanZ)/tauZ恒等式误差<1e-12，额外项使用逐细胞Z/J联合值，不以平均量乘积替代。0–10s仍是有限间期事件，无恢复证据。

新增CPU/GPU配对后端检查：rho.25/k50实际10秒checkpoint后200ms，所有4万细胞脉冲、整份快/慢/噪声状态及Rg/resource flux逐位相同，CPU中间160ms约快2.07倍。正在延长为1秒（rho.25和rho1/k50各一条）以覆盖实际多个事件，决定是否安全迁移当前低态以加快探索。暂未改动任何生产worker或监督器。脚本benchmark_topic4_resource_backend.py，注意Rg必须恢复。

## 北京时间04:45：R6已启动，R5出现局部持续态

新增R6 `adaptation_capacity_round6` 3条参数条件，supervisor PID91262，均已派发：gamma.25/eta1/tauM2、gamma.25/eta2/tauM2、gamma.25/eta.4/tauM10，seed9108401、tauZ5、60s。总科学条件36/48。完全复用冻结R1原方程，不加phi、Rg或rho；eta2/tau2与eta.4/tau10有同样eta*tau=4，比较容量一致但积累/恢复速度不同。依据是R4实际强M.1在15–20秒的IE1497、有效II495、M80，远未覆盖可竞争的M容量；eta1/2对应500Hz持续率的容量1000/2000，这只是容量代数而非稳态解或必然终止。见dispatch_review与实际数值授权文件。要先区分点火被抑制、低平台与真正返回，任何新候选先配对种子确认。

R5 rho.25/k50完成20秒前缀，无200Hz/200ms全E高态门；15–20秒Core A/B/外围均率283/382/57.5Hz，全E约60–80Hz，持续窄带而非恢复。已目视新版fig5_dynamics和event_extent：此分支仍无自主高—返回—高。R5 rho1/k50与rho1/k0约18秒仍有限事件，继续观察。

观测绘图修复：Rest不再只凭某一点低率；要求整个50ms空间窗口落在全E与双核至少50ms的<5Hz区间内，而且该原生空间窗全E均率<1Hz。实际rho.25图Rest50ms为0.0681Hz。Finite event现在复用独立audit的<=300ms且两侧>=30ms安静定义；没有全E高态门的条件仍保留Tail activity空间图，避免漏掉低全E均值下的持续core。状态依然按真实时间、无物理变化。该图已目视，chronological PASS；其他最终交付图需用修复后的观察器重画。

R5后端：rho.25和rho1/k50实际10s状态后1秒CPU/GPU续跑，包含全E平均23.3/21.1Hz和峰20ms率160.6/151.9Hz的事件，全部4万脉冲、完整状态/RNG与新增资源/全局变量逐位相等；CPU分别快1.44/1.70倍。rho.25现在已进入核内持续态，故不据静态前缀迁移它。仅rho1/k50在下一20s完整检查点切CPU，其余两条留GPU。新supervisor PID90968已安全接管旧89173，三个原worker保持；migration_pending=resource_rho1_k50_s9108401:200000，检查20s后backend_resume.json/新PID。新脚本run_topic4_resource_cpu_resume.py与supervise_topic4_resource_backend_resume.py；运行原物理/观察器不变。资源继承计数器会将这个新CPU wrapper保守地计入GPU数，不会因此超发。

截至最新快照，R1完成11/12、R2完成6/8、R3完成2/8、R4完成0/2、R5完成0/3、R6完成0/3；全部无失败，0已确认自主恢复。旧48条log-M已完全完成并交付两个seed完整原Fig5。goal仍active至08:50；07:20停止新派发，保留候选确认和最后审阅时间，不因本轮已经有许多负结果而提前标完成。

## 北京时间05:05：第七轮参数对照与高态结构实测

R7 preserved_global_gain_round7 已于04:52左右派发三条，supervisor92014：rho.25/k200/tauG2、rho.25/k50/tauG10、rho.25/k200/tauG10，均etaM.005/tauM2、tauZ5、r0=50、60秒、seed9108401。与R5的rho.25/k50/tauG2构成保留Z时的增益50/200 × 建立时间2/10秒的2×2。只改变已验证R5方程的参数；全部全局抑制仍受Z影响，也参与其耗竭，没有引入受保护抑制池。总科学条件39/48，剩9条优先留给独立噪声确认；不能自动扩展新机制。依据是R5 rho.25在15–20秒已形成Core A/B 283/382Hz的持续放电带，尚无全E200Hz高态；需分辨加大实际抑制能否终止核，而较慢建立是否保留进入过程。

新高态结构图 high_state_structure/figures/high_state_structure.png/pdf 已生成并目视。实际1ms全群体计数与原生0.1ms采样E脉冲显示：原生55–60秒全E与双核均500Hz，20ms率CV=0、采样ISI固定2ms；快速阈值phi2.5/gamma.5在35–40秒全E263.29Hz、20ms CV约.0006、ISI固定3.8ms；保留Z rho.25/k50在15–20秒全E70.77Hz而双核283/382Hz，20/60采样E完全静默，活跃样本ISI主要2.2–2.7ms。原生有限事件窗口则有20ms率CV1.36和很长的事件间隔。快谱峰或规则脉冲不能代替持续振荡包络、自限事件或自主恢复，更不是Hopf证据。具体窗口、谱分段、均值归一化、采样条件与全部数值见observed_structure.json。

R5 rho1/k50在20秒整份检查点迁CPU已完成，原PID89180替换为91819，检查点SHA daa110ba2a339c6e66d25e3c3618756dc6208eee5c4d9c549f02ffce3c954441；所有快慢状态/延迟/噪声/Rg保留，后续已到29秒。rho1/k0 GPU到28秒；比较相同30秒实际chunks后再判断全局项是否仍为0。R6三条原M容量测试已到13–15秒，均未进入、Z约.987–.998，需从实际有限事件/局部率审查是否主要阻止进入，仍按60秒上限观察。

当前R1完成11/12，R2 6/8，R3 3/8，R4/R5/R6/R7均未完成；所有无失败，无已确认自主恢复。deadline及07:20新派发截止保持不变。

## 北京时间05:18：局部终止与全网恢复分离

已目视R3 global_M_joint_comparison以及k50/tauG2/中等M的regional_current_balance图。新可复现观察脚本analyze_topic4_core_quenching.py：14.17–18.87秒Core A全区域平均0.032Hz，每个10ms都<5Hz；Core B平均465.50Hz且每个10ms>=362.7Hz，外围88.38Hz。A的平均IE/实际I/M约37.8/358.8/.82，B约1439.3/739.6/3.91。与此同时全E meanZ .4564→.1790。5ms Rg指数衰减的严格因果下界保证该区间内全局输入至少593.90、超过Ith95.20，因此每个E的b=0：A安静并没有带来其Z恢复。它是局部暂时终止，不能称为网络自主恢复。弱M/k50同样诊断没有这种>=200ms的一静一高区间；仍只有一个开发噪声，不声称跨噪声稳健。脚本对rho>0只说明b=0，不误套原生dZ=-Z规律。

R5 rho1/k50与rho1/k0的实际0–30秒、全部原始/pool/resource chunks逐位相同，额外全局电流始终为0；其中20–30秒包括k50由完整检查点转CPU后的生产数据。这说明该前缀的抑制进入来自rho修订本身，不能归功于全局池；也实测确认了此次CPU续跑的原轨迹一致。此对照不是独立噪声重复，记录rho1_pool_zero_production_audit.json。

R6 eta.4/tauM10现在已保存20秒并重画fig5_dynamics且目视：15.2秒和19.03秒仍有稀疏有限传播事件，Z接近1、全局无高态，不能从早期安静宣称永久沉默。观察图修订为按真实状态选色（未进入的Tail灰色，有限事件橙色），末段zoom向前平移以保留完整300ms，不再截成75ms。数据、原操作性判据和仿真物理未变化。


已准备新的跨R5/R7绘图器plot_topic4_preserved_global_comparison.py，只有四条都保存>=20秒才画共同窗口的2×2。当前尚WAITING，不要把脚本准备当作已出图。它核对不变参数/几何，实际全/核率、Z、Rg、乘Z前后反馈和同一末50ms原生场；无阈值/方程改变。下一轮延续应在R7到20秒后执行并目视，再决定第二噪声确认或有限参数补点。若全E始终低于200Hz，仍应查event_extent/真实双核率与安静间隔，可能存在局部持续态的终止；应单独报告而非偷偷修改原全局high门。

## 北京时间05:33：配对原生控制的实际60秒高态复核

analyze_topic4_paired_native_controls.py已重读旧etaM.005/tauM2两噪声的真实chunks，并各裁到首次高态确认后60秒、外部恢复之前：seed1到71.92s（外部71.93s），seed2到73.41s（外部73.42s）。当前全E/双核恢复门重新逐10ms应用，两条都无恢复，末5秒全E、Core A、Core B和外围均500Hz，原始前10秒两噪声确实不同；这是两条已存在的噪声复用，不增加39个新条件数，也不算不同网络重复。paired_native_controls/figures/paired_native_prefixes.png/pdf已目视PASS，human PENDING。绘图时发现旧chunks的regions是uint16，已在本新绘图器里先转float再乘1000，且加入图中core均值与10ms汇总相等的实际数据检查；当前夜间runner为uint32，不受这个旧dtype乘法问题影响。

R7三条的前10秒全部原始观测（含所有spikes/分区/原生场/raster/输入/电流/ZM）及resource flux均与R5 rho.25 reference逐位一致，额外全局输入为0；tauG不同所以Rg轨迹本来不同。记录pre_feedback_prefix_audit.json。R7两个tauG10条件已在13.66s进入、13.86s确认，tauG2/k200当时仍无全局高态；尚无返回，等完整20秒及以后实际字段审阅。

R5 rho.25实际30秒局部持续状态后200ms CPU/GPU对照，全4万spikes、整份状态/RNG、pool/resource flux逐位一致；但meanE93.31Hz时GPU更快，GPU/CPU中段时间比.704（CPU41.27s、GPU29.05s对应160ms）。结果另存backend_benchmarks/..._persistent_state_200ms.json，未覆盖早期低态证据。故rho.25及R7不据低态结果迁CPU；仅既有rho1低态CPU继续。没有新增科学条件或物理修改。

## 北京时间06:00：两种自主返回候选与第八轮确认

R7的rho.25、tauG10、etaM.005/tauM2、tauZ5两条已完整保存30秒，并按再次进入后停止的规则结束。k50首次进入13.66s，实际全E与双核共同<5Hz为16.97–19.11s，共2.14s，19.14s再次进入；该返回段没有有限间期事件，只能称高—安静—高。k200首次13.66s进入，第一次高活动后已有多次有限爆发；后续低态门确认22.16s、再次进入25.29s，需用完整30秒finite/event_extent审计实际间隔及返回事件分布。此前20秒的k200“尚未通过低态门”已被新数据更新；不以旧前缀断言永不返回。两条均没有Z/M reset、刺激或clamp；新增rho并行恢复项仍是新方程，不归于原生Z已成功，也不声称Hopf/稳定周期吸引子或患者传播已恢复。

R8已派发6条（supervisor94019）：rho.25/k50与k200/tauG10的第二噪声9108402；rho0/k50与k200/tauG10的seed1原Z方程对照；rho.25/k0/tauG10的两个噪声。总45/48新科学条件，3条保留给决定性确认，不自动新机制扩展。tauG10在原生Z上此前未测试，故不能把当前成功归因于rho必要；R8直接区分。

增益50seed1的精确dense replay已启动：supervisor94198、worker94203，native_field_candidates_recurrence目录。完整1ms状态和所选10kHz原生电流/电极窗口用于完整Fig5，需通过全部原始观测、完整checkpoint与pool/resource记录一致性之后才能绘图。实际源30秒含检查点粒度的额外循环，主图只到第二次确认后2秒=21.34s。既有log-M48格属于原生Z基底，不能作为rho候选的同模型参数图，候选使用匹配gain×tauG对照。

## 北京时间06:35：原生Z也出现返回，优先验证原方程

R8 rho0/k200/tauG10的原生Z条件已完整30秒并目视：首个200Hz高段11.72–12.93秒，后有较短爆发；全E与双核共同安静20.54–22.67秒（2.13秒），22.71–23.20秒再次高率，26.45–27.41秒又一次。滚动低窗确认22.41/25.16未落入高段，返回间隔没有有限间期事件。Z仍总体下降；此结果否定了“当前自主终止必须增加rho恢复项”的说法，但不是已恢复患者间期分布或已建立稳定周期。R7 rho.25/k200第2次越门仅210ms，属于较宽短爆发，不能直接叫第二次持续发作。

R8 rho.25/k200的第二噪声也已30秒完成：13.86秒进入，22.33秒低窗确认，23.89秒再次越门；finite-return审阅可用。完整paired-noise/native-Z/no-pool矩阵继续收尾。R9只新增原生rho0/k200/tauG10的seed9108402；supervisor95850、worker95856，46个参数/噪声条件。原方程无需新QA，rho0早已验证为逐位原生分支。

另启动原生候选的同一轨迹30–60秒续跑，监督器96088。目的是检验Z继续下降后是否最终转为高平台；只延长观测，所有engine/RNG/Z/M/Rg/延迟和全部参数保留，源30秒文件不改。准备时engine哈希2968c937285c8877033f335874c52c17a1d81c7baf26de6710e0012304adbe53一致；中途观测Stop只在完整检查点后被忽略，最终60秒、墙钟deadline和其他异常仍有效。保守把这一续跑计为额外1名额，46+1=47/48，剩1名额不自动新扩展。

三个完整Fig5精确dense replay：rho.25/k50监督94198/worker94203；rho.25/k200监督94423/worker94433；原生rho0/k200监督95851/worker95859。前两条由图监督94898在QA和native10kHz能量后自动拼A–F，原生条自身同样接图。新plot_topic4_autonomous_recurrence_fig5.py要求整份源重演一致；D为实际1ms Z/H_E/rE，F为B电极顺序与原生细胞1–150Hz功率、电极独立读出，E不借用旧log-M图。原生图E用rho0/.25×gain50/200同tauG10/同30秒；其余图E用rho.25下tauG2/10×gain50/200。仍需目视修复，生成不是人审通过。

已生成并目视两段原生GIF及contact sheets：k50的16.7–20.1秒安静后全场再入；k200的15.25–18.45秒多次广泛短爆发。返回事件比较55个进入前事件与9个两次越门间有限事件：时长中位100→230ms，峰率85.36→421.68Hz，原生union面积.645→.97。它们显著在描述上不同，不能称原间期分布已恢复；没有把事件当独立网络做p值。core时差仅42/55进入前事件可估计，已把图D样本数修正为42。event_field_sequences显示进入前局部波与返回后近全场招募，待用户人审。

最新状态统一用scripts/snapshot_topic4_autonomous_exploration.py写latest_review_snapshot.json；旧静态39条件/0返回字段已移除。07:20不再新派发；08:50按原goal截止审阅，未满时长明确删失，不提前宣称完成。

## 北京时间06:58：短循环的反证与最后同状态对照

原生Z k200/tauG10同状态续跑已保存40秒：33.02秒再次进入后到40秒仍连续高活动，meanZ=.01514，broad sustained段33.00–40.00秒。已目视完整40秒raster/局部率及added_feedback图，初始几次返回不代表长期稳定循环；全局原始反馈不断增加但乘Z后的反馈下降。继续原定60秒，不自动把短图判阳性。

最后1预算用于rho.25/k200/tauG10的同状态30–60秒续跑，revisedZ_continuation_to60，监督97182。source engine hash0380bec86be1bd5bb88a9110d7ec8d72a5fbbcefea7cc7abd5289b4bd1cdf3aa，所有状态/噪声/延迟/参数完整继承，原图源保持不变。QA和截断规则已通过；和原生续跑构成资源项是否影响晚期失败的直接对照。总46参数噪声条件+2条同状态续跑=48/48，不再新增任何科学条件。

rho.25/k50的F中B顺序与原生1–150Hz早期能量为负相关，七个固定非重叠基线窗口系数−.614至−.843、合并基线−.75；不是选窗偶然。另计算A只有−.108（原.5–1.5s基线），不能说其实A已有明确正对应。正在把两family读数共同写进完整图metadata；图保留B负结果，不做有利选窗。光谱场来自原生细胞、电极独立计算；正负变化保留。独立谱图colorbar排版已修复并目视；返回前后事件比较图已修正A/B时差可估计样本42对9并目视。

scripts/summarize_topic4_autonomous_overnight.py已从实际原始计数重建各条高/低标记、检查时间bins和分区计数恒等关系，已完成初次47条在途ledger；最终需要在48条有端点后用--final重跑，不以缓存进度冒充完整时长。输出overnight_review/trajectory_ledger.json/csv，仍待最终报告和人审。

07:05更新：R9原Z第二噪声完成30秒并目视，首13.21/13.41s，低窗确认23.25s，再入23.32/23.52s；实际第二高段430ms、第三720ms。最长首返回严格双核/全局安静1.90s，后2.74s；确认未与高段重叠，返回窗没有有限间期事件。原Z短程返回已两噪声重复，但seed1同状态40秒晚期持续高活动的反证仍成立。

## 北京时间07:25附近：交付收尾状态

48/48预算已用：46参数噪声条件+原Z/rho.25各一条同状态30–60s续跑；新派发已停止。新rho续跑监督97182/worker97186，所有原状态哈希保持、物理未变。统一状态脚本及ledger已改为48；ledger明确同seed条件是配对反事实，续跑不是独立重复。

原Z续跑已60秒完成，source unchanged PASS，中途仅40/50秒完整检查点后的观测停止被忽略。33.0–60.0秒全场持续高，末10秒全E497.364Hz/coreA499.099/coreB498.939，meanZ=.00027727。新的long_recovery_contrast脚本对两条长跑实际5ms全局反馈作严格下界，验证原Z条件指数衰减、rho条件趋向rho/(1+rho)的20ms Euler解；截至50/40秒已匹配到<5e−15，并目视初图。最终必须用--require-complete在两条端点就绪后重画。

F语义更新：两个rho.25候选A原窗−.108，B−.743，未建立正对应；原Z候选A原窗+.413，七固定基线+.406至+.811，合并基线+.776，B则−.893。为遵循展示较相近模型类别，main full_fig5的F现在默认auto选固定原窗中较大有符号系数；两个family全部系数和选择规则存metadata，无选窗，不能当独立验证。三张旧B-only图及metadata已完整复制为full_fig5_B_diagnostic。三张auto版本已经目视，需最终刷新visual_review中的新图hash（如尚未写）；人审始终PENDING。baseline sensitivity helper已区分displayed family和B，最终自动收尾会重算三条。

新增supervise_topic4_overnight_delivery.py，监督100514，只等待48个实际端点，不开仿真。就绪后自动运行summarize_topic4_autonomous_overnight.py --final、long_recovery_contrast --require-complete、三条能量基线复核，写overnight_review/delivery_status.json=READY_FOR_FINAL_SCIENTIFIC_REVIEW；仍须agent审读、目视最终长程图、更新scientific_review.md和README、交付及goal收尾。报告草稿和3套完整图链接已在overnight_review中。当前主图不能证明患者传播恢复；返回事件近全场、峰率过大，M必要性未测，无Hopf证明。

额外双噪声κ0 vsκ200反事实前缀已核对PASS：两种子0–10s全部原始观测、pool与resource字段逐位相同，新全局电流当时为0。结果paired_feedback_counterfactual_prefix.json。Liou原文再次核查：其全局连接是空间累积反馈，不等于本轮新增10s时间滤波；强全局尾在安静期仍耗竭Z是当前模型特定问题，不归于原文已证机制。
