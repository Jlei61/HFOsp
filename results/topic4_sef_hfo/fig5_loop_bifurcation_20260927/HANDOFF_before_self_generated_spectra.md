# 最新交接：完整原生历史对照、输入时间结构诊断、实际阈值配对均完成

**Goal ACTIVE，完整目标未缩减，正式分岔NOT_ESTABLISHED。本turn有实质progress；不能complete/blocked。** 当前本批所有worker/collector已结束，下面历史段落的LIVE信息已经过时。先读`history_and_input_closure_review.md`和该段；原生引擎、density_spatial.py、正式Fig5、已认可状态空间未改。新code仍仅scripts/topic4_loop_bifurcation，无子agent/push/清理。

## 新的原生结果与图

- native_K9p35_held_history完整30s（42→72）完成，33768/33778/33780/33817/34170等退出。history_comparison/result.json：原12s史末20–30s全E/A/B=98.407/392.366/132.067；heldK9完整42s史=196.620/473.160/474.098，Graw皆近零，场RMS差198.401。Z/K相同、300futureinputs逐位。两种有限条件状态，不是吸引子/稳定性证明，无末窗短事件。
- density_history_selection_K9p35两条10s完整、最终RNG逐位：原12s史108.770/393.509/137.072、held42s史200.457/472.309/474.995，源数值流和时变期望外源相同。density_K9p35_held_history/comparison/result.json：同5–10s原生causalR197.567，density201.125，G激活分段不同，不能因核率接近晋级formalbranch。
- figures/native_exit_branch_candidates.png/svg：9条完整原生条件点，rate/counterfactualZdrift/4空间场，无虚构稳定线或临界点；figures/native_density_history_pair.png/svg：两历史原生/密度四列。均真正打开自查，SVG_XML PASS，humanPENDING。前者最适合对话展示当前条件分支；不替换正式Fig5。1.5mm底物圆圈和原1.75mm统计分组有明确说明。

## 输入诊断完成，不要重派

- native_K9p35_source_statistics：42–44s unchanged原生2秒，40000targets counts+IE/II二阶矩，20k样本；观察QA、I均值预算1.1e−15、E计数整数残差1.1e−9，E由预算反解不能当独立均值验证。
- 误差边缘785E：群源均值的净偏差2.664/RMS5.543mV；实测IE/IIvar28.5966/6.84281，white290.5187/236.6101。
- native_K9p35_temporal_inputs_v2：同2秒全源spikes0.1ms、39target IE/II/V/M；最终完整engine与原44s参考逐位、counts/globaltrace/矩QA PASS。初次目录native_K9p35_temporal_inputs因终点Stop异常跳过save而失败，已保留；v2只是finally保存修复，无物理变化。
- v2/spectral_analysis/result.json：保留各源自己实测时间自相关、忽略distinctsourcecross谱，边缘预测var22.54966/3.84669。全E至少20ISI的7364sources medianCV0，0.1–0.9分位0/.01753；不是Poisson。有限2秒periodicPSD残差含相关/边界/非平稳，不能纯归因crosscov。whiteprojection4.44e−16，ParsevalPASS，重复跨延迟边0。
- local_response_factorial：39target×4conditions×1024MC×16s，phaseobserver。30edge RMS errors11.947/6.189/5.405/.180Hz。分开native/group mean和white/measuredvar，实测moments不是autonomousclosure。
- threshold_audit.json：发现density个体targetweights已经保留但threshold仍groupmean。1193targets不同，max.455mV。30edges全18mV不受影响。原阈值由native_target_thresholds.py读取实际substrate.npz并核nativeidentity+positions。
- local_response_factorial_exact_thresholds重测完，edge结果逐位相同；controls最终RMS2.0835Hz（旧4.7982）。Core5148新测463.199vsnative457，旧群阈值471.060。不要再说全部目标用原个体阈值的旧groupdensity版本已精确。
- v2/selected_current_replay_exact_thresholds：实际theta+IE/II/M/s->每步V误差0、每个spikeexact。固定M均值后39cells总rate全部不变。旧groupthreshold失败保存。result中的Gaussian比较引用已更新为exactthresholdMC，原result/producer保留。
- s/G单位核对：original localassay把s当G（此时s4.24e−28）。修正30*s后156个真实inputpars逐位不变，raw_G_units_qa PASS；不要混淆temporal文件causal_R_G[:,1]实际上s，用30*s。
- v2/prescribed_spectrum_response：六target×1024 GaussianFFT实测NET谱，2speriodic、1–2sburn、16srecord(8重复不是8独立样本)，kernel每个native suppliedspike核验PASS。5148=457.741±.068MCSEM vsnative457、低通方差463.199。total_variance_control排除EIcov改变总方差：匹配相同netvar的低通仍463.201。其他目标不是一律改善。仍是dataprescribed，不可称闭合通过。
- figures/input_closure_diagnosis_exact_thresholds.png/svg已实际目视+XML，人工PENDING；旧无suffix图仅作旧groupthreshold记录。

## 刚完成的两条自由阈值修复实验

- density_K9p35_exact_thresholds：workers39061/GPU0和39067/GPU1、collector39082已结束，结果全收齐。代码density_exact_thresholds.py只wrapper修改e.pars[:,2]为native原theta，其他pars bits、初态、Z/K、时变外源、Gaussianseed928751不变，末RNG逐位。
- 原12s史fieldRMS42.2346→42.3128，rates108.783/393.481/137.017；held高史18.5216→18.5271，rates200.452/472.114/474.815，Graw.112315，causalR范围200.646–201.610仍开G。这排除阈值平均是主要全网偏差；以后保留actualtheta但不要反复微调threshold。
- held_K9p35_G_increment_response/result.json：完整/半幅增量G负反馈η≈−1.038/−1.054，使staticdR/dK−83.99→−41.21。背景G保留，不是G=0支，也不证明动态稳定，更不能用于原生另一激活分段。

## 下一步的边界和目的

1. 正式条件分支仍只围绕actualexitfield和原生真实可达状态，完整natural loop仍为目标。不要再花一轮精修旧white/groupthreshold根的数值残差；这个根属于原近似模型，当前native相应Gsegment不同。
2. 现在有理由检验**模型自己产生的源时间相关**，不只是调均值/总noisegain。先设计一个有界源输入/输出谱自洽或等价历史结构候选；实际原theta、individualtarget/source异质性、原synapticfilter/延迟、G/M必须保留。此次实测谱只能初始化/诊断/验证，不能成为未来神经驱动。文献Dummer2014和VellmerLindner2019已经读过并引在新review；弱空间相关/异步假设不可直接默认。
3. 在这两个native K9.35历史锚点先检查空间/Gsegment/核心Z漂移。若新的自主近似不能保持这些联系，换合适的保历史分析，不能放宽门或漫扫无关根。谱迭代收敛不等于物理时间稳定。
4. 自然exit携带大的Graw，K/G同.5s，静态Z/K条件支不能单独解释退出。进入和间期返回的原生传播验收不能被本固定高态诊断替代。目标未完成，不标complete/blocked；本turn是明确progress。

---

# 最新交接：原生K边界与同协议密度已完成；高态可达性对照在跑

**Goal ACTIVE，全目标未缩减，正式分岔NOT_ESTABLISHED。本turn有实质progress，不能complete/blocked。** 原18批之后持续目标见get_goal；现阶段关键不是继续减小静态残差，而是确定分支是否对应原生真正到达的状态。先读native_exit_bracket_review.md；下方旧handoff的live/待完成描述已被本段替代。

## 完成的新证据，勿重跑

- native_exit_K_bracket两条均完整30s，supervisor27841/workers27851/27852/collectors27854/29014均已结束。density_correspondence/result.json：K9.35原生末20–30s全E98.407、A392.366、B132.067，G≈0；旧carried density200.450/472.310/475.001，400格RMS203.416。K9.5双方安静。300futureinput逐位配对。直接钳制原12s史不等于density自身K9高态慢斜坡，禁止把此区间称formal fold。
- density_exit_bracket_protocol两条相同原生12s初态、即时K、相同50–60s期望外源率的密度各10s完成；supervisor32517/workers32525/33401/comparison33164均结束。原phys/kernel未改。K9.35匹配5–10s全E108.770/A393.509/B137.072，原生98.444/392.482/132.215；场RMS42.2346，核心/G/Z漂移顺序接近但外围偏多，未认证全对应。最后4个1s全E≈109已近持平；原生5–10对20–30场RMS.1304只是轨迹内变化，不是noise acceptance band。K9.5前5s场RMS.941，5–10s静默完全一致。
- 同协议改善不能单归因历史：此前density还换了常值外源和K斜坡。本轮residual定位11/400格贡献80%平方误差，主要活动带下边缘被多招募；spatial_residual.json有top10及坐标。
- G_activation_rectification/result.json完成并已刷新全30s：两边尾窗无跨200门槛波动，meanq与q(meanR)无实质差异。1ms采样，不是精确0.1ms预算。
- native_exit_K_bracket/unclamped_flow/result.json：20–29s有两端R，非负spike+15ms衰减给采样间界。K9.35全程5<R<200，q0/Ktau.5s，counterKdot−18.6981/s；全EZdot+.08064，cores−.03483/−.01500，恢复资格p约.613/0/.090。K9.5全程R<5，Ktau5s、Kdot−1.89998，全部恢复资格1。逐步预算对解析差≤1.2e−11。K/Z仍钳制，不是释放试验或新自主闭环。

## 新图实际已验

figures/native_density_K9p35_protocol.png/.svg：三列原生12s即时K、同初态density、旧density高态斜坡；率颜色仍AllE紫/A粉/B青，G，400格空间场。agent已实际打开修正图例遮挡后再次验PNG，SVG_XML PASS，human PENDING，未替换正式Fig5。README/metadata已写。圆圈1.5mm是原底物标志，率口径沿用原1.75mm近核分组（metadata/README明确）。冻结analysis_producer.py哈希匹配result；最新figure_producer.py匹配修正renderer。producer compare_density_bracket_protocol.py。

## 真正LIVE：只这组新历史对照，先核对勿重派

- 原生native_K9p35_held_history：supervisor33768/session56662，worker33778 GPU0；genericcollector33780/session42872；具体historycompare33817/session83749。当前snapshot [{'name': 'exit_z0.21_k9.35_fields16p7_held_K9_history', 'pid': 33778, 'device': 0, 'created_epoch': 1790548351.41, 'time_s': 46.0, 'status': 'RUNNING'}]。从现有原生K9完整42s终点出发，仅K设9.35/未来外源配对原t50，Z/完整内源态保持；30simsec绝对42→72。不是K慢斜坡；一开始小步钳制。prepare_native_held_history.py完整引擎逐位QA PASS。仍一共享seed，不是新自主样本。
- density_K9p35_held_history：worker34166 GPU1/session45376，collector34170/session2916。当前 {'status': 'RUNNING', 'pid': 34166, 'device': 1, 'simulation_s': 7.0, 'elapsed_wall_s': 277.4808325767517, 'updated_epoch': 1790548864.5774808}。恰好一条10s40000x128，起始物理态来自同一原生42s高史，期望外源仍50–60s，局部Gaussianseed928751。复用density_bracket_protocol.worker（wrapper仅换OUT/SOURCE/contract，不改物理）。新脚本density_held_history.py。没有常值均值替换。
- 两条均有固定预算；native对比在history_comparison/result.json，density对比在comparison/result.json（都等完整native30s）。还未有结果，不把早期prefix当末态。

## 下一步按证据决定

1. 先收这组原生高史和同协议density；完整未来input300逐位QA由collector做。若原生也达到两核高态，则有限支持原高支相关性，再决定有界稳定性/延拓；不自动认证吸引子。若仍不达到，不继续精修无关的200Hz高支，转向实际不对称状态及外围过招募。
2. 不重复已通过的K9.35phase-aware非线性校正；它仍只是density特定高态的有限精度近自洽，不能解决历史/空间对应。
3. 原生source、density_spatial.py和正式Fig5均未变。新code仅scripts/topic4_loop_bifurcation/；cuda_env解释器；无子agent/push/cleanup。不要为一个formal名称放宽门或混淆conditional/autonomous。

---

# 最后更新：独立非线性校正已通过有限精度规则

**held_exit_phase_correction_validation/result.json 已COMPLETE：NEAR_SELFCONSISTENT_AT_RECORDED_SAMPLING_PRECISION。** worker30750/30756及finish30760已全部退出；不要重派。当前仅剩native_exit_K_bracket的原生两条和其自动分析仍活跃，具体PID见下并现场核对。

- E sourceRMS .0169279→.00313429Hz，combinedSEM .00325267，0群超6SEM；I .213939→.0660803，combinedSEM .0648059，0群超6SEM。
- E个体targetRMS .0143112，combinedSEM .0141605。源群和localM同时满足原预设有限精度标准。
- 这只是独立数值流支持的近自洽点：root_certified/stability/formal_bifurcation仍false；原生K9.35空间对应未收齐。没有改标准、没有改网络、没有自动新K点。
- 读只读预测bounds（未派发）：若从新候选用当前切线前移K .01，最大target/sourcechange39.35Hz、target负投影1.202Hz；.005对应19.68Hz/负投影.00512Hz；.0025对应9.84Hz/.000890Hz。这些不是已执行的参数点，也不是允许随意放宽当前零K校正的.001Hz投影界；新参数步若需要应单独冻结适用bounds和独立非线性验证，不能把线性预测画成branch。
- 优先等当前原生两侧完整结果核对空间/核心与历史；不要在尚未对应的新邻域过早继续很多细步。接近R200时特别检查真实R波动与meanG关系（mean clip(R)不等于clip(meanR)），再决定是否需要修复反馈闭合；采样1ms的Jensen诊断不能冒充原生0.1ms精确预算。

---

# 最新进展：修正值、当前点校正和K切线已完成；独立非线性验证在跑

**Goal ACTIVE，完整目标不变，正式分岔NOT_ESTABLISHED。当前turn是progress，不可complete/blocked。** 先读本段；下方上一阶段细节保留但“phase值/DC待完成”已被本段替代。最新机制说明`held_K9p35_current_point_review.md`。无新正式图，前轮两张候选仍有效。

## 真实live任务（先核对，勿重派）

- 原生native_exit_K_bracket仍supervisor27841/session10553、workers27851(K9.35/GPU0)、27852(K9.5/GPU1)、collector27854/session32227，完整对比29014/session53747。最近绝对clock27/36秒，起点12，所以相对完成15/24秒；终点42，必须跑全30秒。完整20–30秒末窗尚未齐，不当最终空间对应。
- **held_exit_phase_correction_validation**：worker30750/session86837 GPU0和30756/session28287 GPU1；finish/session86426。命令`validate_phase_held_correction.py worker/finish --wait`。每流40000目标×256复制、16秒记录、phase随机额外burn0–1秒，freshseed929391/929392。最近约27904/28928目标完成。独立验证当前点的一个校正，没有自动下一轮。
- 已完成退出不要重派：phase全目标值workers30034/30041、collector30047；phaseoperator30204；same-equationauditor30283；校正prepare30609；局部K tangent也完整结束。
- 活跃时不要改validate_phase_held_correction.py、其复用sampler measure_held_phase_stationarity.py、phase_lif_mc.py、held_direct_moments.py。新改动另建文件。

## 本turn实质完成结果

### 1. phase修复全40000目标结果
`held_exit_phase_stationarity_K9p35/result.json` COMPLETE。allE200.449413对原密度200.449563；A472.307932对472.309527；B474.997306对475.000667。400格RMS **.006050Hz是对密度，不是对原生**。相比旧fixed4s值有改善，但源群RMS E.01693/I.21394、个体E_M_RMS.21272仍非zero；更精细采样后E432/1885与I272/1594群超6countSEM，不能把均值接近当精确根。

### 2. 相位修复后的当前点校正与独立QA
`held_exit_phase_dc_operator_K9p35/analysis.json`及`joint_operator_qa.json` COMPLETE/PASS。使用K9.35测得DC（保留幅度失败/weak），不是K9导数。GMRES60步info0，最大群变化1.57289Hz，source投影5.68e−14，预测linearres1.86e−7。同方程JVP相对误差2.11e−10/2.28e−9，sourceconstraint重建3.08e−13Hz。
5个target预测轻微负值，最小−1.6789e−5Hz；明确project到0后才准备独立response（最大允许project.001Hz、sourcechange5Hz，均满足）。投影后不当自洽，依靠正在进行的freshphase counts。
当前点新helper`held_direct_moments.py`仅从现K9.35inputs读Z/K/M/r/external，与不变W矩阵构建同一输入公式；没有加载K9响应斜率。校正prepare当前input重建PASS，G.1122717/R201.122717，新source/M为唯一区别。

### 3. 局部K方向与机制路径
`held_exit_K9p35_tangent/result.json` COMPLETE，producer`analyze_held_K_tangent.py`。K参数独立有限差分链式误差1.14e−10（包括K/G反转电位差）；GMRES68步info0。静态局部dRate/dK：allE−41.073，A−11.037，B−3.887，surround−42.776，I−51.118Hz/K；dGraw/dK−4.121。当前R201.118，线性外推约deltaK.02714到R200的G激活分段边缘。**不是已测criticalK/fold，过分段边缘导数公式会变。**
上一个完整9.2→9.35保持有限差分allE−70.137，不能把.15步长/历史差当微小导数验证。最大8block矩阵actionSEM14.098Hz/K，不是整个tangent/closure误差界，更不能据此做稳定性认证。
`decompose_held_K_sensitivity.py`及`sensitivity_components.json/.npz`复现输入路径分解，所有项加和误差<1e−10：allE递归E均值−45.34、directK−6.90、localI缓解+7.80、G缓解+3.39Hz/K。A相应−10.94/−6.20/+3.16/+2.92；B−2.61/−3.66/+.74/+1.63。包含implicitM，依赖完整已解递归响应，是线性路径分解，不是独立因果消融。提示外围递归招募和G反馈分段与下一步核心退出相关，不能直接说naturalexit就是这个分段点。

## 下一步操作

1. 收freshphase校正结果和native两条完整30秒；不要无谓重测已结束的DC/phase均值/QA。
2. 当前非线性验证仍沿用原有限精度判据：E/I sourceRMS<=2combinedSEM、no group>6SEM+1e−7，EtargetRMS<=2combinedSEM；SEM明确不包括closure/systematicderivativebias。通过也只叫finiteprecision近自洽，不是精确root/稳定支。
3. 若通过且原生空间/核心对应支持，在这个相关点附近探查K对R200分段边缘及核心退出的连接；真实分支需当前点响应，不能越过G分段后沿用旧dG。若失败先看具体残差是否估计/闭合/非线性，保持有界，不为求收敛改标准或漫扫频率。
4. 保持原始闭环及进入/返回两段目标。固定K图不等于原自主平衡；高态K/G同.5s，G历史/空间场和Z/K自然漂移要参与解释。未新增releaseK/fixedZ-only实验。其余原生机制和旧失败保留见下及归档。

---

# 当前交接：发现固定计数起点偏差；在同一K9.35输入上修复定常值

**Goal ACTIVE，完整目标未縮减，不可complete/blocked。** 本轮有实质进展：完成21目标相位/窗口审计，验证观察器修复及DC协议，K9.35全目标DC收齐，并实际启动相同输入下全部目标16秒相位随机化响应。正式分岔NOT_ESTABLISHED。保留原生/耦合密度物理、density_spatial.py、正式Fig5与已认可状态空间；无子agent/push/清理。新文件限scripts/topic4_loop_bifurcation，cuda_env解释器。

上一交接完整归档`HANDOFF_before_phase_count_repair.md`，其中完整四保持、G30秒、当前图和既往机制数字仍有效。当前科学说明另见`phase_observer_review.md`和已刷新的`scientific_review_live.md`。**当前关键变化：不能直接使用旧固定起点4秒定常值构造的Newton建议作为已验证平衡；先用新phase-aware值。**

## LIVE：逐个核对PID和result，不重派

- **native_exit_K_bracket** supervisor27841/session10553，原生workers27851 GPU0(K9.35)和27852 GPU1(K9.5)，genericcollector27854/session32227。原生绝对clock12→42即30simsec；最近绝对20/26秒，分别只完成约8/14秒，不是20/26秒的相对历时。后续以liveJSON为准。只有K固定值不同，其余完整12秒高史/实际16.7秒ZK场族/未来t50输入同旧控制。`compare_native_exit_bracket.py --wait` PID29014/session53747自动收完整30秒后比较densitytail；不能把前缀当最终状态。
- **held_exit_phase_stationarity_K9p35**：新全目标phase-aware静态响应。worker30034/session42379 GPU0，worker30041/session85137 GPU1；collector30047/session87064。每流40000目标×256replica、16s记录、1sburn+均匀0–1s额外burn，seed929381/929382按batch分开。输入npz与旧K9.35值检查逐字节一致，仅估计器变。最近仍采样，40000目标，预计总数值壁时数分钟，不预设完成。
- **held_exit_phase_dc_operator_K9p35** collector30204/session58612，脚本`collect_phase_held_dc.py --wait`。等新phase值COMPLETE后，链接已经完成的旧K9.35part0/1只读数据（同一点输入），用新值重算一个未验证校正建议；从新inputs读实际Z/K/r/M/g，不用旧K9导数。运行中不要改helper`collect_held_exit_dc.py`。没有自动采样或root接受。
- `audit_phase_held_dc_operator.py --wait` PID30283/session11230等上一步analysis后做当前点same-equationJVP与implicitM/G一致性，写`joint_operator_qa.json`及`joint_newton_proposal.npz`。由既有audit复制、全部工作点输入改成新K9.35，IE/II从当前r和不变W重建。它尚未实际完成QA，不可先称PASS。亦不启动新GPU计数。
- 所有当前活跃新脚本/phase_lif_mc不要中途修改。新步骤另建文件/目录。没有自动K扫描/动态频率扩张/native新job。

## 本轮已完成且不要重复运行

### 全目标K9.35DC：完成，矩阵可供检验；旧值下的proposal不晋级

`held_exit_dc_K9p35/analysis.json` COMPLETE_DC_OPERATOR_AND_UNVALIDATED_STEP。两个workers28860/28867、collector28873已退出。GMRES62步info0，旧值最大groupchange1.62732Hz，clipping1.84e−5，最大matrixactionSEM.011716Hz；这只是一个未验证建议，尚未独立测量，更因下面值偏差不能直接使用它。

幅度失败必须保留：E mean107、varianceE471、physicalG395；E varianceI没有可估计单细胞组件（32000均nonestimable）；I mean17、varianceE97、varianceI84个estimable幅度失败。不能用21个代表细胞的协议检验给整个矩阵盖章，更不能把DC eigen/gain当时间增长率。

### 固定短窗相位偏差：已被直接证实

`held_exit_response_window_audit/result.json` COMPLETE。producer`audit_held_response_window.py`；21目标，覆盖核心/核外/I率区间与最大残差；2048复制，原核未改，同噪声路径，4/16秒×起点偏移0/.5/1/1.5ms。
四核心cell2822/16370/16813/4558：原4秒起点476.0Hz，偏移.5ms后476.25Hz，MCSEM全0；四个16秒起点均476.1875Hz，M约476.1905。整个21目标最大4s起点spread.25Hz，16s约.0079Hz。其他随机细胞有独立噪声差别，全部保留。**这不是物理root转弯，零SEM不代表对定常率无限精确。**

### 观察器修复有界验证：完成

新`phase_lif_mc.py`修改局部MC观察器：原burn再加按复制hash生成的0–1000ms离散额外burn；不消耗额外物理RNG；计数仍恰好T/.1ms步。可选DC在burn中已施加以比较真正定常增益。原生网络与density网络均不改。streammode0独立target，1全conditions共享replica，2相邻full/half共享、不同target/channel独立。

`held_exit_phase_response_validation/result.json`和`review.json` COMPLETE。producer`validate_phase_response.py`、`review_phase_response.py`，原21目标×2048复制。QA：零extra-burn/原调制时序下，静态mode0/1、原physical四通道、原全目标mode2配对均逐位一致。
- 随机相位4s vs16s均值21/21在3pairedSEM内一致。
- 四相位锁定核心16s约476.190Hz，4s约476.186–476.195Hz，不再零SEM假精度。
- 原4s→phase4s记录开始调制47/47estimable一致；phase4s→先burn调制47/47；phase4s warm→phase16s warm51/51；原4s→phase16swarm51/51。比较沿用10%或2pairedSEM，不降标准。
- 原/phase4s各47estimable full/half47/47，phase16s51/51。其余26（16s）或30（4s）弱通道仍nonestimable，**不计PASS**。
这支持重新测量定常值，不是修改模型或精确root证明。phase均匀有限burn窗口与16秒也非无限时间真值；任何新根仍需独立验证。

### 先前已完成证据与当前候选图

四K保持：9.2/9.35最后3秒两核~474Hz仍活跃；9.5/9.65静默。9.35全E200.4496/Graw.112035，allEZdot+.03817但核心仍−.03483/−.03307。有限8秒持续性不是稳定根。
原生初始G30秒对照完整结束：baseline initialG4.51、G0、G11.495均未两核退出；强初G使allEtail约209.65、空间场改变，但两核约471.56/470.09；allEZdot+.04449而核心仍负。完整300记录输入逐位相同，共享历史噪声不代表独立seed。文件是`exit_actual_G_history_probes/G_history_comparison/analysis.json`。
当前新图仍是`figures/carried_K_transition_bracket.png/.svg`和`figures/native_initial_G_history.png/.svg`，均已实际view最终版，agentPASS/humanPENDING，未替换正式图。上一turn已在final向用户显示前者。当前turn没有需要额外画的科学图，不必重复图。

## 下一步实际操作（完整目标保持）

1. 先核对phase全目标计数、重算operator、sameequationQA、原生bracket的真实结果/PID。完成的不要重派。
2. 新operator若info0、有限物理邻域/投影和QA通过，准备一次独立phase-aware非线性响应，不能使用旧fixedstart4s的validate脚本原样采样；需同观察器、独立seed，明确当前输入identity与M更新。用新数据的残差和MC误差判定，保留未通过分量，不把数值停止叫fold。
3. 原生K9.35/9.5的最终rate/空间必须核对；即时clamp起态不同于密度ramp，若有差别先分辨历史/输入/闭合。不可跳到正式相关稳定性分支结论。
4. 原生自然K/G都是高态.5秒，条件固定K不是全自主平衡；G历史影响空间场。完整目标包括进入和返回，不因本轮局部numerics进展缩成退出的一张图。继续检查是否需要包含动态K/G的更合适条件系统；**本轮没有新增releaseK或fixedZ-only实验**。
5. 旧K9.01875数值残差不是fold，旧代理22roots已否决，0/1/5Hzfeedbackgains不是时间特征值。不得为了一个形式名称继续无关分支。

Memory使用仍为MEMORY.md324–325；final最后一个citation块，rolloutids01a09eae-c163-7cf2-8f2d-f11d43bdeaaf、01a0add1-6bb8-78f1-8af3-98dc7b0724b2。无memory写授权。当前turn分类progress，不能blocked。
