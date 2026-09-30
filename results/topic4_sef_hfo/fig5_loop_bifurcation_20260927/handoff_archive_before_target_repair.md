最新中途观察：native_count_noise两条在13.7–14秒均已降到近零活动，接近候选早退时间；原来8405参考保留核外活动。**尚未终点外源QA和完整分析**，但不能继续把与唯一原参考的时间差当作已确定的系统闭合失败。928701最新17秒、92870215秒，继续到20秒；待paired_input_audit PASS再判定。新增计数worker不在旧live_native的文件名白名单内，任何新派发须额外计入8848/8849，当前旧控制器无pending。bifurcation_scope已补充确定性条件分支与随机转换分开检验，完整目标未降低。

# 最新交接：epoch 1790528041（以下为当前状态，覆盖后文较早运行描述）

本轮是progress：**两条完整反馈分布候选已结束**，三张新诊断图实际打开并检查，6/6长窗及8/8退出/初始G/返回探针全部完成；新启动两条只改外部Poisson计数的原生续演。Goal保持ACTIVE，完整目标未缩减，正式分岔NOT_ESTABLISHED。无子agent/push/清理，原生kernel和已接受Fig5未改。

- coupled_density_exit：数值流927671/927672各10–20秒，3479组×2048粒子。只给配对外源，递归/R/G/Z/M/K自主。外源重建和接口QA PASS。原生R<=5首个100ms为16.868s，候选13.750/13.764s；候选18–20s再次持续活动，均率168/180Hz，原生零。自主对应仍未通过。PID7335及分析7827已经正常结束。
- coupled_first_dip：原生13.7秒右上核外活动继续，两核已近静默；原生R仍>5，K实际0.5s消退。两候选核外残余消失，K实际5s消退。定位误差放大过程，尚未识别最初差异原因。图已看、README已写。
- history_noise_review：120秒K6/K9历史差异保留；只改初始G未改变共同K9场的末窗分离；三条未来输入在K.01为31/27/28个短事件，K.02为3/3/6，K.03为1/1/0。新plot_history_noise_cuts已结束并修正JSON漂移变量名（实际是Z/K两列），PNG与已看的图逐位相同。
- native_count_noise：**两条在跑**，种子928701/928702只指新外部Poisson计数流，原生完整初态与全部OU路径相同。原rng仍消耗原Poisson抽样，独立rng替换返回计数；无膜/递归/慢状态改动。PID8848/GPU1和8849/GPU0。严格10–20s，不能自动增种子或延长；不要按等待超时重启。脚本native_count_noise.py，contract/rng_routing_qa/launch在同名目录。每个checkpoint增加paired_external_count_noise字典保存独立rng和计数；结束核对100输入记录、20秒完整外源/RNG/时钟，paired_input_audit.json PASS才允许分析。
- 对应collector：PID8916，analyze_native_count_noise.py --wait；最终生成native_count_noise/comparison.json和figures/native_count_noise_exit.png/svg。该新图尚未生成/目检，不能提前写README或称验收。原生新结果未完成；从进度继续。
- independent_noise_probes仍有两条K9未来噪声在跑，控制4192984及collector4177916正常；其他8/10结束。其余入口以current_stage_snapshot和逐条实时结果为准。不要修改正在运行的worker脚本。

后续：先收取两条原生计数续演和余下K9条件噪声，检查paired_input_audit及实际图。若原生同OU也早退，有限计数敏感性可解释部分时间差，但两条不能估准分布/自动认证候选；若仍都保留核外活动，空间耦合/波动近似更可疑。正式分支对象仍优先实际退出场K9–12，并需本地动态/空间对应和稳定性；不要重新追共同场无关静态根或拟合Z/onset。新科学说明见coupled_density_review.md、native_history_noise_review.md；completion_audit_live和mechanism_model已更新。此次是推进而非等待或blocked。

---

最新补充：entrynoise两条均已完整结束（2/2）；collector4184623会统一最终分析。120秒entryZ.74高史尾窗全E169.925、核A169.518、核B295.124，0brief。以下较早的1/2等待描述以本句和current_stage_snapshot为准。

# 最新交接：epoch 1790525629（本节覆盖下方旧运行描述）

本轮是 **progress**：8项早期局部条件 + 1条3.3秒原生只读退出记录 + 4项实际G/K局部条件均已完成，2张图实际打开两次检查，独立输入/全引擎审计通过。Goal仍ACTIVE，全部完整范围保留，正式分岔NOT_ESTABLISHED；不是等待/blocked。没有子agent、push、清理，也没有改旧原生kernel或已接受Figure5。

## 新完成工作与结论

1. `conditional_density_inputs` / producer `conditional_density_inputs.py`、`analyze_conditional_density_inputs.py`。复用20260918/native_early_surround_inputs 已逐位审计的0–3秒记录、g40=3479原图、原先16个几何群(210物理细胞)。4arms×2numericalstreams(927641/42),8192particles：projected/measured means × full/private_sensitivity variance。用冻结density_spatial.native_cell，自己的V/ref/current/Z/M，G/Koff。不是新native。
   - 固定0.5–3秒50ms窗，full投影均值计数/原生：surround1.0080/core1.0017/I1.0036；实测均值1.0038/1.0006/1.0010。波形relativeL2约.008–.020。2990msE平均Z误差投影−.00039/−.00011，实测−.00018/−.00003。
   - 原先fixedrate约13%少放电在当前jointdensity局部检验中未重现。不是空间或闭环通过。私有分拆只是旧stationaryPoisson敏感性，不是精确非平稳条件cov；不修改主方差。
   - 原生输入延迟GPU/CPU，旧独立discrete均值重建，逆滤波均值<1e−9，37+63与100完整state/ref/RNG/count逐位。新图`conditional_density_response.png/svg`已viewPASS、人审PENDING。
2. `native_exit_input_observation` / `observe_native_exit_inputs.py`：一条完全原样16.7→20s记录，从已核验完整state继续。记录all3479counts、16target IE/II/V/Z/M/K/eligibilitymoments、每步drive、globalprestepR/s，保存210cell完整初态和359格pendingring。**92旧数组及20s整个physicalengine逐位一致**；只normalize slow.kind的classmetadata，clampFalse。新增counts汇总等原全E1ms。producer正常Stop通过finally保存，新观察器未改物理。
3. `conditional_exit_density` / `conditional_exit_density.py`、`analyze_conditional_exit_density.py`：2means×2numericalstreams927651/52,8192particles，16.7→20完整。原jointstates和每cellpendingring共同均衡复制；新Gaussian输入只来自t0之后真实groupcounts。实际globalR/G提供，Z/M/K自行演化，平均threshold仍原候选。不能当自主反馈。
   - 抑制16.7–16.9：native selectedE counts5342/766，投影预测比.9948/.9946，实测.9975/.9962。后续E都是0，Z先降后约17.55回升；区域Zmaxerr<9e−5,K/gLmaxerr<.014。20sZ约.47–.48，**尚未充分恢复，更无间期返回**。
   - 反证必须带：I在Gtail/recovery为native6/21counts，投影3/11、实测6/23；输入均值在低态仍重要，这些少量脉冲不是独立统计重复。后半静默段Z/K受提供G/R强约束，不能把接近重合说成自主闭合或线性增益已验证。
   - 359格pendingqueue按原绝对clock重排、paired同cell不打乱联合状态；CPU原图delaycheck<8e−13、meaninverse<2.3e−13；400 vs357+43全部state/ref/RNG/count逐位。图`conditional_exit_density.png/svg`已实际view，并把1msrate标签写清后再次viewPASS；I限制在review/analysis完整保留。人审PENDING。
4. 完整科学review新增 `conditional_density_review.md` / `conditional_exit_density_review.md`；mechanism_model、scientific_review_live、completion_audit_live、figures/README已更新。**图与所有原始数据都仍候选**。

## 活体/新原生结果

以最新current_stage_snapshot.json为准：native_extensions4/6complete，剩K6high(终132s)/entryZ.74interictal(终170s)。K9high120秒已分析：84.997/core54.132/coreB0，与30s相同，recovery仍0；这是finitehistory持续，不是认证bistability。entryZ.74high120秒已结束，collector会分析，不要重复launch。
entry_spatial4/4；exit_return4/8，剩3初始G+K.02仍跑(最近绝对30/30/53/54s，终42/42/60/60)。noise4/10，K.03两future noise接近60终点，K6high/recovery在28/54s，K9两history待派；entrynoise1/2complete，高史实际Z.78 noise8402由incremental_review单独读出125.436/core209.071/350.828，无brief、无quiet，是局部持续活动复现，**不是全E>200进入**。另一interictal接近80s终点。
已结束新PID2899(earlylocal),4166(observation),4690(exitlocal),5099(analysis)不再占GPU；启动记录在对应*_launch.json。现有supervisors4172089/4192983/4192984/4189148、collectors4174808/4177916/4184623保留。所有新最近active PID identity/live均真、failed空，没有重启/更改nativejobs。

## 下一步：不要重新铺宽域拟合

先收齐近完成的entrynoise、120s和K.03及G历史结果，对实际fieldfamily作新转换图。新局部条件验证已改变定位：早期少放电不再是当前主故障，但自主网络空间招募/输入coupling与数值收敛尚缺；实际G/K局部瞬态也有对应，但G/R是提供的，不能据此准许正式延拓。
下一对象优先真正退出相关的actual16.7field K9–12/携带G；如果做完整density条件延续，可复用本次jointinitial/pendingcellqueues初始化原则，保持Z/K场族、G/M动态与外源/噪声定义，先证明对应，不要直接套旧共同场K9失败根。还缺实际相关工作点的独立dynamicgain/linearstability和相关稳定/不稳定支认证。原有formal目标不能降为仅native转移图，也不要为了找可求根而离开真实转换。
新的conditional脚本没有自动派发任何network/fit/grid。数学/科学范围见bifurcation_scope.md。所有脚本在scripts/topic4_loop_bifurcation，用cuda_env。Memory只用MEMORY.md324–325，最终保留原citationblock两UUID。不要update_goal complete/blocked。

---

# 最新交接：epoch 1790523516（覆盖下方旧运行状态）

本轮 **progress**：一条细输入配对修复完整、4条真实退出场完整、4条未来噪声返回完整、2条120秒延长完整；新增两张已实际打开的图和两个原生方程机制核对。Goal仍ACTIVE，正式分岔NOT_ESTABLISHED。没有子agent、push/清理或改原正式图。

## 本轮最重要的科学结果

1. **实际退出场4/4**：Z.21/K9高态史，全E242.58Hz/核A464.53/核B470.77，恢复史静默；K12两史静默。共同场K9高史全E85/核A54/核B0。自然全E Z漂移共同场+0.10710/s，实际场−.042/s。实际场Graw均4.3386，20ms采样间保守下界4.1546仍高于2.6694，所以全E恢复目标在完整末10秒必为0，漂移恰−Z/5。**exit_field_feedback_audit/analysis.json** 与 producer analyze_exit_field_feedback.py。Z/K两个场一起换，不能单归Z。
2. **恢复时间下界**：analyze_recovery_time_bound.py→recovery_time_bound/analysis.json。8402/3/5既有15退出段嵌套3seed；5段在最快允许的Z参考恢复之前重新进入，8完整返回，2在观察末尾仅Z恢复/返回截尾。8403首次可用2.89s却最低5.2116s，8405第二次2.46vs5.4562。原Euler严格上界逐采样核对，maxexcess5.6e−17；10段已达Z参考的到达时间均距离散最快界<20ms采样步。这不是新release阈值或正式分岔。
3. **G尾部核对**沿用本轮前半完成的feedback_tail_mechanism。新图feedback_tail_recovery PNG/SVG，两配对seed8402/3，按延迟G首次R<=5对齐但对照同绝对时间；R/G/K/两核netZ较小值。8403在+2.9s又有活动，保留反证，首次低活动不等于充分恢复。重绘拆分ylabel后再次实际打开，agentvisualPASS/humanPENDING。
4. **细空间输入修复完整**：density_fine_forcing，repair_density_forcing.py，3479×2048、数值927611、原G/Koff、12.5s。只换原成员细分组外源，原kernel冻结；旧100ms8观测逐位，完整final RNG/clock/R/seed逐位，paired_review.json全部PASS。原OU重建来自20260918/native_fine_external_drive，7检查点，1ms保持和float32global仍不变。
   - 粗输入→细输入：onset10.495→10.745s；D9870 .205705→.236700（原生.25634）；42vs39事件、median92vs90ms；area.5898vs.5705（原生.676/.746）；quiet.4945vs.5909（原生.492/.518）。A4从5/6到6/6，仍不是整体验收。
   - firingcontacts19vs23（原生14/17）；原0.3方向6/5/8vs4/9/10（A→B/B→A/弱）。总体rank/order错误增大，但B→A类内order基本不变；正向native仅3/1，不能说全部类都更差。无rawcurrentLFP新增、无数值收敛、无真实G/K空间认证。**density_forcing_review.md**完整判断。当前没有新的density/MC/求根进程在跑，不自动再加网格、粒子或拟合onset。
5. **独立未来噪声4/10返回已完整且已手动增量analyze一次**：K.01末10秒27/28brief、22/24强核，quiet.718/.720；K.02为3/6brief且均强核，quiet.959/.941。一个K.02粗标签quiet但有3真实事件，绝不可读成0。这是相同内源8405下的未来8402/3，不是完整自主新seed。原collector等待全部10条后会最终分析。

## 新图/文档

- **exit_spatial_response.png/svg**：两列高史/恢复史，两行rates/Z自然漂移，共同场vs实际16.7场，K6/9/12附近。实际打开PASS，humanPENDING。没有改native旧plotter（其会覆盖README）。
- 上述feedback_tail_recovery，README已真实追加。两图metadata均PASS/PENDING。
- **bifurcation_scope.md**重要：Z/K作参数平面没有被否定；固定完整空间场族仍能定义参数化条件系统，G/M可以留作动态状态。同点多历史本身可能多稳态，不能当二维参数图不合法。被否定的是所有空间场共享单一均值边界。自然K高率消退.5s与G同尺度，不能未经检查当准静态参数。正式相关分支优先实际退出场K9–12；旧共同场K9数值根不要自动再追。
- mechanism_model.md加入p>meanZ、G尾部、严格Z恢复时限、实际场G阻断、Liou2020来源边界。scientific_review_live.md / completion_audit_live.md均更新。

## 运行/资源（以新current_stage_snapshot.json和活体为准）

新 **execution_resources_v5.json** nativeglobal16/perGPU8/8，保留3GiB每worker预留和host70GiB门；只并发已有冻结清单。新supervise_probes_v4.py仅v3的资源文件路径改变。持有native_dispatch.lock并核对PID/root/cmd后，仅停止旧exitcontroller4174740/noisecontroller4177418，原nativeworkers未动，新controller接管：
- exit_return controller **4192983**，exit_return_probes_launch_v4.json，日志exit_return_probes_controller_v4.log。4/8complete；剩3个初始G+K.02已全部运行，无pending。
- independent_noise controller **4192984**，independent_noise_probes_launch_v4.json；4/10complete；K.03两未来噪声+K6两history noise8402在跑，K9两history noise8402仍pending。collector4177916保留。
- entrynoise仍controller4189148 v3（13/7/6 cap但无pending，仅监控2活体），workers4189159/4189688，collector4184623。终点42/80，最近28/64绝对秒。
- extensions仍controller4172089，2/6complete：K6recovery4172107和K9recovery4172109正常结束并分析，末窗全0/quiet1。活体K6high4172106到112s(终132)、K9high4172108到120(终132)、entryZ.74high4172110到117(终132)、entryinterictal4172111到149(终170)。这是120秒总观察，不是绝对终点120。
- followup collector4174808仍在，自动分析前三root完成条目，identity/prefix/check输入保留。
- 最新检查14nativeworker活体/命令identity全真，failed空。No duplicate dispatch。细输入worker4191663/audit4191664/review4193795均已完成，不要重复派发。

## 下一步

先继续收集余下长窗、初始G及独立噪声，形成明确场族/历史下的新转换图及漂移证据。实际场K9高态已经与旧局部残留分支不同，是相关对象选择的新依据。细输入修复只支持部分闭合改善，不能直接认证G/K分岔或又自动扩展全域拟合。需要决定的是真实退出局部动态对应/稳定性对象，而不是把整个混合噪声闭环强叫Hopf/稳定周期。已接受状态空间仍不改。

本轮浏览并读了Liou/Abbott2020 eLife https://elifesciences.org/articles/50927 ：globalinhibition/adaptation和更长静默允许抑制恢复是文献方向；当前0.5sG/high-rateK/5Hz保留仍是本项目假设，不称原文逐式。最终若引用文献主张用原文链接。

Memory已读MEMORY.md324–325；普通最终保留原约定citationblock和两UUID01a09eae-c163-7cf2-8f2d-f11d43bdeaaf、01a0add1-6bb8-78f1-8af3-98dc7b0724b2。Goal未完成，不调update_goal。

---

# 最新交接：epoch 1790520438（本节覆盖下方旧运行描述）

本轮是**progress**：8192完整结果、原始电流只读重放、冻结触点观察器及方向内比较均完成，两张新图实际打开并修复后agentvisualPASS、人审PENDING。Goal仍ACTIVE，不能按A4六项通过宣布分岔或闭环近似已验收。

## 新结论与产物

- density_spatial_resolution **COMPLETE**：R8192进入10.610s，D9870=.212095，39事件/86ms/area.60084/quiet.612。原生9.8685/9.8695s、D.25634/.25646、面积.746/.676；原2048进入11.153/11.924s。同流4倍粒子进入变化−543ms，数值收敛尚未证明。
- compare_density_resolution.py 生成 comparison.json及 figures/density_baseline_correspondence.png/svg。原A4开始时间纳入会包含原生onset相连长事件；图中空心三角标越过9.42s，C轴log，另存complete_quiet_bounded_windows。不可将这些长事件说成完整间期事件。最终图已真正再次打开，metadata agentvisualPASS，humanPENDING。
- replay_density_contacts.py 的 **density_contact_replay COMPLETE**，原2048num927612整段重放，所有群体float32观测与finalstate/ref/history/RNG/clock/global/accumulator逐位。新只读abs(rawIE)+abs(rawII)每0.5ms，原Eq9–11核在各群内聚合，独立核代数误差7.1e-15。没有修改density_spatial.py。原native时间标签为步前，density采样按end记录，转换到原标签有.4ms相位差；分布比较不要求迹线相同。
- compare_density_contacts.py **COMPLETE**，沿用旧interictal_surrogate_6101冻结观察器，0.5–8s，未重校准。缓存native rawLFP与当前Fig5原始chunk逐位核对。firing合格14/17 vs20/24/20；currentHFO14/14 vs重放24。总体顺序多在原生间差异量级，数量差异仍在。
- analyze_density_directions.py **COMPLETE**，复用旧50Hz持续5ms空间arrival与rho分类0.2/.3/.4。主.3 native A→B仅3/1，B→A7/10；8192为6/11/3弱复杂。B→A空间与触点较接近；A→B原生样本少，弱复杂仍差，不能全方向接受。所有事件嵌套描述；与A4质心方向不要混为同一个分类。
- plot_density_spatial_contacts.py 已出 figures/density_spatial_contacts.png/svg。三行native8401/2048stream2/8192stream1，列Z8s、按自身类中位选例的正反向场、固定3–4.5s发放触点；原生q99.5逐触点共用标度，固定15槽。已目视PASS，人审PENDING。这里是firing envelope，不是rawcurrent图。
- 新完整科学审阅 **density_correspondence_review.md**。mechanism_model.md只更新了四条实际entry已完成的证据，其他物理不改。

## 唯一新运行的近似诊断

**density_spatial_grouping**：PID4188453，density_grouping_launch.json，refine_density_space.py --device1。最近750ms/64.96s，活体核实。只一条3479g40×2048粒子、12.5s，原927611数值标签、原8401记录外源、G/K关，无自动后续。

原density_spatial.py冻结未改；新wrapper在构造时临时换OPERATORS，随后恢复moduleglobal。g40原四套物理矩阵和prepared均symlink，只geometry.group_cell按细cell整除2改为原1mm强迫/输出索引，保留group_cell_native_fine。全部finegroups属于唯一coarsegroup；原阈值加权均值同一，四套含delay矩阵在coarseconstant rate场聚合误差<3e-13。operator_check.json PASS。约712万粒子 vscoarse8192的766万；均匀独立Bernoulli整体数值方差因子3.093e-7 vs2.957e-7，不等于整网噪声精确配对。核/物理、子ms与群内外源缺失、Gaussian独立输入假设均未改。

结束后需分析新的trajectory.npz/finalstate：同窗A4/complete事件、area/Z、firing contacts/directions。现有compare_density_resolution.sources()只列旧5轨迹，不能直接把新3479数据喂旧935权重；新operator geometry有3479 contact_rate_weights，输出field_E_Hz仍400格。当前没挂自动分析者，也没补rawLFP观察器；不要冒充有。判断依据见新contract和review，勿自动新网格/调参。

## 原生队列

以current_stage_snapshot.json epoch1790520438为准；12worker活体和PID创建/命令已核对、failed全空。native_extensions0/6：高K6绝对94s，恢复133；高K9 100，恢复132；entryZ.74高98/间期134（各有不同原start，终点132/150/170，非统一120绝对秒）。entry_spatial4/4已分析。exit_return2/8完整，实际K9高绝对29/恢复55s在跑。noise0/10、4active当前50/45/46/45；entrynoise2排队。所有控制器/collector沿用旧启动记录，无重启。最终两entrynoise已用完预算，全部12noise槽无剩余。

旧density_resolution4184622、replay4185991、两个audit4186504/4186827均已结束；不是当前活体。原生workers/collectors和新fine4188453继续。无子agent、push/清理。cuda_env。Memory引用仍MEMORY.md324–325及原两个UUID。最终必须保留goalactive，不调用update_goal。

---

# 当前交接更新：2026-09-27，epoch1790518832

Goal仍ACTIVE；本次完成了有信息量的新阶段，绝不能把以下候选和A4门当作整个goal完成。下方旧交接保留科学来源；**运行与最新结论以本节及current_stage_snapshot.json为准**。解释器cuda_env，无子agent，无push/清理。原生方程、旧正式图和已认可状态空间均未改。

## 当前运行 / 已完成状态

- 12个native worker仍活体、PID创建时间和命令核实，failed为空。native_extensions6条120s继续中，当前绝对84/121/88/120/87/125s，原起点12/30/12/30/12/50s；不是都同一个绝对终点。旧6worker与controller未改。
- **entry_spatial_probes4/4完整，collector已分析，controller已正常完成**。entry_spatial_response PNG/SVG已生成、真正看图并记录agentvisualPASS；humanPENDING。此图已准备在本次对话展示。新producer plot_entry_spatial_response.py；不会覆盖正式Fig5。
- exit_return_probes2/8完整：实际16.7场K12两历史均静默。现在实际场K9两历史worker4181662/4181810，绝对19/40s；后面3个G覆盖和返回K.02仍排队。
- independent_noise_probes10条：4active（返回K.01两外源，K.02两外源），0complete；其他排队。旧collector4177916还活体。
- **最后2条独立未来噪声名额已冻结并准备**：entry_noise_probes，实际t10Z场、Z.78/K.0002，两内源历史，未来noise8402。来源entry_noise_probe_spec.json；12noise预算已用完，无剩余。controller4184500，entry_noise_launch.json，当前WAITING_RESOURCES，全局native12上限和dispatch锁保持。collector4184623，collect_native_root.py，entry_noise_collector_launch.json。
- followup_collector4174808还活体，native3roots按完整条目读取。entryplot4183263已正常生成结束；densityonsetaudit4183261已完成。
- **新增density_spatial_resolution**：单个8192粒子/群12.5s分辨率诊断，PID4184622，density_resolution_launch.json，GPU1，当前250ms。不是新的native仿真/物理seed。原native队列不改。预计10–20min量级，按真实进度，不重启。
- 目前唯一近似模型运行是上述8192；所有MLP、局部MC、数值求根、新局部kernelQA、两条2048全窗density都已结束。不要重复派发。

## 新原生科学结论

两种内源历史均一致：平均Z=.78、K=.0002时，共同t20场有完整短事件（末10s44/40个、全E约31/29Hz）；改成实际进入附近t10Z场后，末10s0个短事件、jointquiet0，持续局部活动全E125.8/124.7Hz，核A211/204、核B351/351Hz。全E未持续超过200Hz，所以不能把这一步直接叫完整发作进入。

Z=.74实际场两历史全E203.27/203.34Hz、核A410/B424、0短事件；共同场全E约173/175，也无短事件。same futureinput配对、同一history比较只有Z空间形状改变。t10是操作性进入后60ms，不是精确分岔快照。

完整heldcoreZ示例（1.75mm原observer）：共同Z.74核A.72523/B.72205，Z.78核A.76685/B.76399；实际t10场变换到Z.74核A.68153/B.64771，Z.78核A.72804/B.69697。支持关注双核局部资源；还不能断言用min(coreZ)即可得到普适单值边界。

新完整解释写在mechanism_model.md：原式符号/单位、原native离散更新、G一方面抑制膜一方面暂时阻断Z恢复、低活动K长尾、Z参考恢复与短事件返回分离、光滑/分段/噪声分岔边界。科学状态未冒充完成。

## 分布保留路线（替代失败历史MLP，已实施而非只提案）

新代码都在scripts/topic4_loop_bifurcation/，不改旧模型或他人文件。

1. colored_population.py/check_colored_population.py：每个数值粒子保留V,qA,IA,qG,IG和ref，rawcurrent在g改变时不被错误重缩放。colored_population_local：独立CPU时变g检查最大误差7.55e-14、spike/ref一致；随意分块继续全state/RNG/count逐位一致；18个恒定g条件×8192副本，所有逐副本计数与原lif_mc相同。**仅实现QA**。
2. density_spatial.py：935g20原图一二阶矩/原delay，各粒子V,sA,IA,sI,II,M,Z,K+ref，Z取本粒子的rawII阈值、M取其自己的spike。没有learnedhazard。使用原dt.1ms decay-arrival-current顺序的离散Gaussian输入；这与原lif_mc连续扩散精确covariance不同，另行QA。群均值/方差input近似、独立Gaussian到达、缺跨细胞关联与群内细位置、阈值群均值、外源1ms群均值仍都是局限。数值粒子数不是物理cellN。
3. check_density_spatial.py：CPU延迟矩阵3个wrapclock误差<7e-14；G0/G30高/低R的完整局部状态/spike/ref对照通过；CUDAgraph/非graph全state/RNG逐位。baselinecontract中的Gaussianmoment algebra是方程定义，尚没有独立原生inputdistribution验收，不能说Gaussian假设通过。
4. run_density_spatial.py：density_spatial_baseline，R512/R2048各3s，原seed8401记录externalforcing，同一nested数值流927611，G/K关，只检验原有ZM底物。0.5–3s12/11事件、中位82/81ms、均9双核、方向1正11/10反；原生8401/8402为12/15、中位95.5/86ms。候选quiet.634/.639高于原生.552/.504。analyze_density_spatial.py/comparison.json留全表。
5. extend_density_spatial.py：density_spatial_onset，R2048_num927611从原3s完整state/RNG继续至12.5s；R2048_num927612从零到12.5s。同一原生8401external记录，**两个数值流不是两个原生种子**。两条已完成；prefix所有存储group观测逐位保留（field重算可能有float32投影roundoff，不声称所有输出逐位）。audit_density_onset.py已完成原A4六项诊断，两条6/6。
   - num927611进入11153ms，1–9.42s38事件、中位83.5ms、30双核、12正24反2无向、area.5947；D9870=.207767（原生.256344，差.04858近.05容差边缘）。
   - num927612进入11924ms，39事件、中位81ms、30双核、20正19反、area.55547；D9870=.212302。
   - 数值流之间entry差.771s，尚无fullwindow粒子数收敛；不能因A4门通过就闭环/正式分岔认证。Z消耗偏慢仍是真差异。
6. refine_density_resolution.py：据上述证据冻结**只一条8192×935粒子、12.5s、927611同nestedstream**，同方程同dt同外源，正在跑。OUT density_spatial_resolution。结束后比较同窗口事件/传播/D/entry与两条2048，以及原生。没有自动进一步分辨率扫描或方程调参。

## 接下来要做

- 继续收集native120s/实际K9/G/返回/12noise，全部已冻结，勿擅自新铺面。
- 看density8192是否保持进入/晚期耗竭及传播。完整原A4门通过只是数值诊断；需要具体2D空间/触点包络/事件variation对应与原生噪声差异。尚未画新的density代表图，也没做人审。
- **触点观测特别注意**：旧原生lfp_raw是LFPRecorder的abs(rawIE)+abs(rawII)、E细胞、原Eq9–11近场核；density现已存的group_abs_current是abs(IE)+abs(ZII)，不能直接当同一个LFP比较！geo.contact_rate_weights是Gaussianσ.25的rateobserver而非原LFP核。可做明确同定义ratefieldobserver比较，但不能据此说currentLFP通过。若需rawLFP，新增只读observer/replay先核验physicalstate逐位；不改现有冻结worker。
- 发现既有scripts/topic4_kinetic_bifurcation与topic4_interictal_surrogate旧density工作，只读过autonomous_density.py、prepare_resource_strata.py、interictal_common.py。旧density是外源coloredPoisson多项式基、meanrecurrent、Zclamp/Mgroupmean，不能冒充本次jointstateGaussiancandidate；可复用确切contactweights()（Gaussian与LFP两种分开），不修改旧分支/不假设其通过。没有新读memoryrollout。
- 只有关键原生对应和数值近似清楚后，才选择相关条件density/直接MC响应分支，不自动又做genericMLPv5。正式分岔仍NOT_ESTABLISHED。

Memory引用仍MEMORY.md324–325；普通最终回复末尾带原约定block和两rolloutUUID。Goal未结束，不reportcomplete。

---

## 此前交接（历史运行描述已被上文覆盖；科学证据与路径仍保留）

# 最新执行交接（先读本页，旧阶段无需重做）

Goal ACTIVE，用户授权做完新的Fig5分岔/转换图与完整机制，持续反思。不要因启动、失败门或已给候选图标完成。主脚本 scripts/topic4_loop_bifurcation；解释器 /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python。保留所有旧正式图、用户已认可状态空间及他人工作，不提交、不清理。没有子agent授权。

## 当前运行（必须核实活体，时间可能跨工具跳变）

- native_extensions：6条既有30秒前缀继续到120秒，控制器启动记录extensions_launch.json（PID4172089），原6worker4172106–4172111。高态K6/K9、恢复态K6/K9、进入Z.74两历史。prefix硬链接只读；collect_followups会再核验新旧prefix哈希。
- entry_spatial_probes：4条30秒，真实t10Z场（Z均值.74/.78，两历史），K仍t20场。控制器entry_probe_launch_v2.json PID4174474，接管旧worker并并行。新资源控制器supervise_probes_v2.py。
- exit_return_probes：8条30秒，控制器exit_probe_launch_v2.json PID4174740。4条真实16.7秒Z/K场（K9/12，两历史）；3条只改初始Graw（共同场K9，高态改0/11.495，恢复态改11.495）；1条返回K.02恢复历史。12条局部名额全部使用，清单exit_return_probe_spec.json。
- independent_noise_probes：10条30秒已准备并排队，控制器noise_probe_launch.json PID4177418。返回K.01/.02/.03各噪声源8402/8403；退出K6/K9各两历史用噪声源8402。完整外源OU+RNG从同图G30_response0.5_s9108402/3的t50s快照复制，内源历史仍8405。是条件未来噪声重复，不是完整自主新种子。最多12噪声名额，尚余2条未分配，须根据进入实际空间场结果决定。
- collect_followups.py PID4174808（followup_collector_launch.json）：只分析前3个root的已完成条目；包括120秒prefix核验、相同噪声/时长组内输入逐位一致、扩展事件/截尾/漂移/场读出。
- collect_noise_probes.py PID4177916（noise_collector_launch.json）：独立噪声10条全部结束后分析，按噪声来源分组核对，不能要求不同噪声逐位相同。
- stationary_trust_region.py PID4178115（stationary_trust_region_launch.json）：仅同一Z.21/K9的数值求根fallback，见下文，可能很快完成。
- 所有静态/动态v2/v3/v4局部MC和拟合都已结束。不要误派发第二份。

## 资源

execution_resources_v3.json最新：全局原生12worker，每GPU6，每条预留3GiB显存、主机70GiB准入；跨probe控制器使用native_dispatch.lock。原6延长已在跑，没有新派发，仍由旧supervisor管理。实测每worker约1.1GiB GPU/2.4GiBhost，8worker时GPU约39/67%、host可用175GiB。资源增加只并行既定科学清单。旧supervise_probes.py已停止，原生worker未重启；旧exit_probe_launch.json记录了一次因锁退出的controller，当前用_v2启动记录和活体。

## 已完成的原生证据

首批30/30完整。native_slices/extended_analysis_summary.json与figures/native_transition_cuts、native_recovery_drift_cuts（PNG/SVG）已目视自查且在对话展示，人工待审；是有限窗条件响应，非正式分岔。

共同t20场：进入K.0002时Z.65/.70持续高，.74持续混合，.78/.85有大量短事件。退出Z.21时K6高态史455Hz、恢复史静默；K9高态史约85Hz且CoreA54/CoreB0、恢复史静默；K>=12两历史静默。返回Z.995时K.003/.01频繁短事件、.03仅1个、>=.1静默。高态Z消耗/K积累，强K压低活动后Z恢复/K消退，短事件回来后Z又消耗。不能仅凭30/120秒历史差异称双稳态。

analyze_native.py已改为任意job的horizon-branchstart判断完整时长，场模板取job显式字段；未来输入按噪声来源+时长分组。核心事件观察器未改变。collector主缓存JSON中的解释才含这些修订，旧original.analyze输出仍可能写Finite30s，不能直接拿旧analysis字段作120秒解释。

## 真实空间场与退出重建

t10实际全E Z.736，核A/B .677/.643；同均值t20模板核.721/.718。共同均值不能代表原进入局部疲劳。t10晚于操作性进入9.94s共60ms。

exit_state_reconstruction/runs/source10_to16p70/checkpoint.pkl已有效：10→16.7s原观测168数组逐位一致（observation_gate.json）；继续16.7→20s完整物理engine与原t20逐位一致（exit_state_continuation_qa/gate.json）。唯一归一化slow.kind=ConditionalSlow/GlobalResponseSlow；clampFalse明确委托原方程，无数值状态删除。快照Z.213404,K12.660267,Graw11.495041；核Z.177/.168，模板.200/.198。两个重放已结束，不算独立闭环。

figures/spatial_Z_conditioning.png/.svg已生成、看图修复并在对话展示。原生40×40显示格均值，不平滑；实圈物理1.5mm、虚圈原观测1.75mm，数字采用754/786个E细胞，与其他率/资源读出一致。agent visual PASS、人审PENDING。该图只显示空间差异，不证明转换影响；待配对探针。

## 局部率闭合：已停止通用扩大拟合

v2静态FAIL根因旧父表σE最多6.5，g0固定父模型无法修正表外误差。parent_domain_diagnostic已确认，不是MC噪声。
v3取消g0继承约束；全新1024静态点1021严格/1024宽上限，通过。v3线性81点只46通过；g0狭窄σ2/3下27全通过，g>0动态失败。纯历史时间压缩τ/(1+g)在已知诊断改善为68/81，未认证。

conductance_dynamic_v4：新增电导依赖history权重，并在同一静态函数拟合DC导数。256新+9已知训练工作点×3输入×5频率，3975条件，8192副本/2秒；18k固定步后冻结。全新64点×3输入×4频率=768主响应，660可估计，仅322通过，108不可估计；96/96半振幅检查通过。独立静态512点508严格/512宽上限，仍PASS。新DC165点仅83过；有符号错误，不能只说时间常数未修好。g0只48/88过，g>0 274/572过。

v4初始实现与已知time-scaled v3 gain相差<5e-15，192个DC自身有限差分导数检查通过。故不把拟合失败归于已排除的公式实现错误，也不能因为均值通过就正式延拓。training_diagnostic.json与result.json分别开发和独立证据；source/权重均冻结。**不要自动v5、加容量或放宽容差。** 下一步围绕实际原生工作点检查空间/局部响应，而不是继续全域MLP赛跑。时变g与非线性波形尚未验证。

## 当前有界空间定态诊断（不是分岔）

stationary_native_diagnostic.py已完成6次求根，使用静态已过的v3 E函数、原I父函数，原g20=935群，Z.21/K6/9/12共同空间场，M稳态、Graw=30clip((R_E-200)/300)，外源取原参数平均nu=1.317463/ms。原生graph8项身份和400格cellcounts精确一致，同方程JVP相对误差检查通过。它是均值驱动近似与有噪声原生末窗的描述性比较，不能当动力学或传播验收。

K6高根收敛，候选462.71Hz vs原生454.95，核466.50/465.52 vs463.48/462.59，400格加权RMS7.90Hz。K6/9/12低根收敛，预测E约.0012–.0018Hz vs原生0；并非严格静默，不能把低小数率当精确零。K9高初值Newton未收敛，残差733Hz；虽其平均场很近，这只是接近测量初始化，**不能叫匹配成功**。失败主要I局部率未由原生观察提供（只给全I均值），positive line-search卡在接近0的I分量。

diagnose_stationary_I_seed.py一次40步I-only+原80步仍未前进，并非物理无平衡证明。当前stationary_trust_region.py仅对同一方程/同一点作一份200函数评估上限的有界TRF numerical fallback，无参数变更。读取trust_region_result.json：仅max残差<1e-6Hz才是数值根；solver success/field接近均不足。即使收敛也无稳定性或原生对应认证。不能把本诊断的6+求根写成6条新SNN。

## 后续要完成

1. 核实上述运行、收集所有局部探针与120秒持续性；检查collector错误与噪声分组。
2. 给进入实际Z场效应、退出实际场/G记忆效应、返回K.02与新噪声作清晰新图，沿用原Fig5语义，不造平衡/周期分支。所有图自查并对话内展示，人审待定。
3. 结合空间定态诊断决定真正相关分支的局部响应校准/更合适原生分析。高根近似不等于全局验证；K9根失败首先区分数值问题与不存在/非定态。保留全部负证据。
4. 用本轮原生结论与已证实Z/G/K恢复方程整合三段机制，说明均值场/空间场/G/未来噪声边界。最后完整候选和复现入口；goal尚未完成。

原自主源 /data/hfosp/topic4_sef_hfo/fig5_global_feedback_response_20260923/runs/G30_response0.5_s9108405；240s四次完整返回是同一seed，另8402/3反馈时延配对支持时序机制。此前mechanism与axiscontrols在fig5_autonomous_loop_zk_20260924；原axis比较并未严格匹配输出度/低阈值源输出，不能声称纯轴角度因果。

Memory已使用MEMORY.md行324–325；最终末尾按记忆引用要求写。所有当下科学结果均已从真实产物核对，不只凭memory。


最新数值fallback已结束：trust_region_result.json max残差17.678Hz，200评估到上限，numerical_root=False，不能把globalE86Hz/field接近说成稳态对应。暂不继续追逐K9根；先看真实退出场/G探针是否表明该条件分支对自然退出必需。当前没有在跑的rate/MC/求根任务，只有12个原生worker、既定探针控制器和读出collector。新噪声控制器会在这些任务释放槽位后派发，不超全局12。
