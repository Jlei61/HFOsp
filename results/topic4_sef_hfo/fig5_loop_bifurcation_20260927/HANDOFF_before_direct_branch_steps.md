# 当前交接：完整直接DC及一次独立自洽修正已完成，继续相关分支

**Goal ACTIVE，完整目标未缩减。不要mark complete/blocked。正式分岔仍NOT_ESTABLISHED。** 用户要Figure5自主闭环、相关稳定/不稳定分支、完整恢复/间期返回机制，并要求持续反思。无子agent/push/清理授权；保留原生内核、density_spatial.py、正式Fig5及其他脏工作。新代码在scripts/topic4_loop_bifurcation，解释器cuda_env。旧完整历史见HANDOFF_before_direct_response.md及handoff_archive_before_target_repair.md。最新推导与判断见direct_response_operator.md。

## 立即继续的任务（先查实时文件，不重复启动）

**当前所有上述作业均已完成，无需再等待旧PID。** `all_target_dc_direct`两个分区、collector18007、QA18238及`direct_newton_validation` workers18813/18814/collector18815全完成。每个fresh流40000目标×256replica、4秒记录/1秒预热、seed928901/928902；仍是直接局部数值响应，不是新原生seed。新结果已经独立验证一次修正有效，当前应继续实质分析，不要重复启动旧作业或空等。

**最新独立结果**：`direct_newton_validation/result.json`为`DIRECT_NEWTON_STEP_REDUCES_RESIDUAL`。E源群RMS从.03404降至.006972Hz，I从.20264降至.141964Hz；fresh MC SEM RMS分别.004781/.100725Hz。加入起始直接响应误差与8块矩阵作用误差后的合并SEM RMS为.006763/.142398Hz，与剩余残差相当；E/I超3合并SEM+1e−7数值底线为0/2群。见`sampling_precision.json`及producer `score_direct_newton_precision.py`。这是采样精度诊断，不是事后改验收或证明精确root；没有事件的群SEM=0也不证明真实率=0。不要继续迭代同一噪声映射追逐1e−6Hz。

**立即下一步建议**：从已独立验证的K9近自洽候选准备一个小的相关K步（例如同实际场形状到K9.05），用保存的直接DC算子求切线/预测，再fresh直接非线性计数验证/有界校正；先完成第一个物理分支点，不重新拟合NN。当前静态Jacobian和参数导数公式见direct_response_operator.md。完整动态稳定性仍缺原时延、G/M与实际局部频率响应组合，DC矩阵特征值不能直接标stable/unstable。若新的直接分支在原生K10.5仍存在，应检验其稳定性/吸引域，不能凭先前原生两个历史安静就断言该平衡不存在。高史原12s初态Graw4.5113，与K9稳态4.34接近；此前初始G消融仅共同场，不能移用于本实际场。

完整DC数据：`all_target_dc_direct/analysis.json`和`measured_dc.npz`。一次GMRES59iterations/info0；最大source修正2.19375Hz，预测线性残差5.05e−6Hz，矩阵作用MC SEM最大.00705Hz。**后者预测不是独立验收**。E mean/varianceE/physicalG可估计15378/12298/14787，其中幅度失败108/578/440；E varianceI全部32000在本256样本量下不可估计，保留完整数值及不确定性；I各通道也有失败，不能写全目标导数全通过。

`joint_operator_qa.json`：联合source/目标rate有限差分误差2.18e−10及2.31e−9，source约束一致误差2.71e−13。`joint_newton_proposal.npz`包含source改变后的完整target/M提案。15个目标小于0，最小−5.66e−5Hz；fresh nonlinear验证前将这些M/target预测投影到0并记录，不改源群候选，不把约束当精确满足。没有目标超ratecap。

新验证准备首次在任何计数前因参考per_ms float32→Hz→per_ms重构的3.81e−8mV误差触发1e−9检查；提升参考到float64后误差1.36e−12，G差1.78e−15，**未放宽容差或改物理/提案**。记录`direct_newton_validation/preparation_precision_fix.json`。新的G候选4.49165287，非上一轮NN根G4.71；不可混淆两条路线。

这次验证已实际降低残差至采样量级，但尚非精确root/动态稳定性。后续基于已验证提案推进相关条件分支，保留原时延、G/M及波动。对有限记录/reset相位可能的系统误差保持区分；必要时作有针对性的时长/初相检验，不凭有限样本SEM宣称绝对精确。下一步不能再回旧NN代理。

## 前一阶段进程记录（以下原生、DC测量、收集均已结束）

- **all_target_dc_direct已完成**：固定40000目标×均值/varianceE/varianceI/物理appliedG（I无G）×全/半幅度，256replica，4秒记录+1秒预热。两个目标分区[0,20000]/[20000,40000]，GPU0/1，旧workers17774/17775。四通道kernel逐位检查都PASS。保存pairedcounts、8个replicate块、幅度误差、SNR，不可估计不置零。
- **collector18007** `collect_all_target_dc.py --wait`：等两分区完整后收全数组、同一图DC算子和一次GMRES修正建议，写`measured_dc.npz`、`newton_proposal.npz`、`analysis.json`。不是正式根/特征值/稳定性；必须看所有幅度失败和矩阵作用采样误差。完整算子恢复局部M隐式项和原离散R→G的DC项，率单位Hz、输入除1000。候选新点还须fresh direct nonlinear response验证才能接受。不要看到低预测线性残差就叫root。
- **后置QA18238** `audit_direct_dc_operator.py --wait`：等analysis.json后，用同一局部affine响应的联合source/target有限差分检查1000单位、physicalG和implicitM消元，生成`joint_operator_qa.json`及`joint_newton_proposal.npz`。后者才包含source变动后的全部目标/M修正；前一个proposal的`local_adaptation_corrected_rate_Hz`只在旧source输入下修正M。检查目标非负/上限计数及gmres_info后，才能考虑fresh nonlinear验证。该QA是实现自查，不是独立物理验收。
- **所有原生运行已完成**，包括本轮新增exit_midpoint_probes两条各30秒。此前supervisor13842/collector13843/workers13858/13859均已退出。target_density_field_family五条10秒以及collector14261也全结束。没有这些旧worker需要等待。
- 最后一次全目标direct stationary、rawG response、18目标directfrequency均已完成；不是还在跑。fullDC才是当前依赖。

## 本轮关键科学结果

1. **原生中点与邻域对应全结束。** 实际16.7s Z/K场，Zmean.21：K9高史维持原生约243Hz，恢复史静默；K10.5两史30秒尾窗均静默，K12两史亦静默。五个追加target-density条件5–10秒场全静默、正Zcounterfactualdrift与原生对应；K10.5高史0–5s全E原生7.105/候选7.007Hz，场RMS.166Hz。高/静默历史差异仍只是有限条件响应，不能称认证双稳态。Z/K被钳住，不算自主恢复。

2. **先求到实际域高根，再由独立证据否决其分岔资格。**
   - `target_stationary_response_audit`：保留40000目标输入/heldZK/M，DC算子求和延迟后仍严格一致；使用精确因果R的DC因子 b_R=1.003337037。旧冻结v3局部mean预测接近，E群RMS3.216Hz，全部E仍在训练domain。
   - `target_stationary_root.py`求到actualK9高根，maxresidual3.42e-8Hz；E246.293、A467.330/B468.348、G4.7115。JVP相对误差<6e-9，静态代理数值正确不等于机制通过。
   - `target_exit_equilibrium_branch`v1因负率预测guard频繁缩步，保留14点；只修正数值predictor非负投影后`continue_target_exit_equilibria_v2.py`从point013继续。共**22个不重复收敛数值根**，K最大采样9.20043；v2point8为K9.19286、E237.396、A466.371/B468.058，已经转弯。
   - v2 PID15604已经在新的独立响应失败后SIGINT停止，状态**STOPPED_AFTER_INDEPENDENT_LOCAL_DC_FAILURE**。不要重启v1/v2或继续从这套NN响应认证fold。
   - 跨v2point006→008的目标率变化99.6286%绝对质量在surround，90%质量仅1602E细胞(5.006%)；两核仍>466Hz。这是有限相邻解差，不是Jacobian零模态。即便它是代理方程的转弯，也不是自动的发作终止。

3. **实际工作点静态斜率失败明确，不再泛化训练旧MLP。**
   - `audit_target_root_response.py` / `target_root_response_audit`：18个目标，由E三region和I的固定率strata中位及最大mean导数选出；4通道（I3）、全/半幅度，2种电流kernel；8192replica，4s+1s，seed928811。520非调制±条件全部完成。
   - exact-colored E44primary/31estimable/**2pass**；density-discrete E44/32/**2pass**。I21/19，14或15pass。两kernel结论相同，不能归咎电流离散方式。
   - E均率大体接近但梯度可差数倍；fringecell5059预测1.11Hz而direct12.03Hz，mu导数4.28而15.63。高核cell23177预测465.75/direct468.84Hz，但mu导数.584/direct3.38。
   - 当前数值曲线正式稳定性/分岔资格被否决，原生轨迹不变，target-density也未被否决。

4. **直接响应替代入口已取得实质对应。**
   - `measure_target_direct_response.py` / `target_direct_response`：同18目标，650个paired调制条件，0/1/5/20/80Hz、全/半幅度，seed928821，4s+1s、8192replicas；原mean/variance kernel及constantg逐位PASS。direct零频独立复现65/65，50个DC可估计组件的250个幅度检查250/250。全部325幅度比较320pass；5fail都属于15个DC不可估计弱组件（主要varianceI与静默cell），不可删掉。尚非全空间动态稳定性。
   - `measure_target_raw_G_response.py` / `target_raw_G_response`：11E×5freq×2amplitudes=110conditions，seed928841；真实新增分流的vinf=(h0*current+delta_g*EG)/(h0+delta_g)，同时改变membrane decay，原始随机电流保持不变。比“只改decay且有效输入固定”更符合G。独立零频链式关系11/11；全/半55/55。网络G_raw仍需乘Z；不重复加噪声重缩放项。producer原核未改。
   - **`audit_all_target_stationary.py` / `all_target_stationary_direct`**：全部40000目标、两独立流928831/32，每256replicas，4s+1s，density单创新filter、fixed observedM，输入来自已对应网络5–10s均值。无NN。全E direct244.10617 vsdensity244.11083；A469.172 vs469.231，B474.232 vs474.302，I309.818 vs309.803。400格场RMS **.019429Hz**（对density，不是native）；E/I群RMS .03407/.20264Hz，maxI1.89093Hz。G4.49208。两流target差RMS .20619，与合并SEM RMS .20556一致。此为近自洽固定输入，不是精确root；固定M/upstreammean近似和小残差保留。两流是数值样本，不是新原生种子。

## 已完成新图与审阅

- `figures/target_branch_response_audit.png/svg`，producer `review_target_branch_response.py`，readouts和metadata在target_branch_response_review。实际看过PNG，SVG XML通过，agentPASS/humanPENDING，README已写。A数值高支、B转弯主要是核外边缘变化、C均率、D斜率失败；不标stable/unstable或fold。
- `figures/target_density_field_family.png/svg`，5追加条件全部quiet。已把rate轴统一0–300，重新运行collector产图并**重新看过**PNG，SVG XML通过，agentPASS/humanPENDING；README已写。图需与target_density_exit K9high对照理解，不能将全黑图说成完整闭环。只有显示轴改动，仿真未改。
- 无新正式Fig5替换、无人类验收；原accepted state-space和原生闭环轨迹不改。

## 下步核心，避免再走偏

读取all_target_dc_direct两个worker及collector状态。完成后先审幅度/SNR、implicitM分母、source矩阵的采样误差，再审一次Newton proposal。source单位Hz、W输入除1000；G项是chi_physicalG*Z*0.1*b_R*w，不能再加phi_mu/variance电导转换项。详见direct_response_operator.md。

如果proposal小且方程残差预期能下降，按同直接局部方程生成fresh nonlinear response（局部M按自己的率自洽）核实；成功后才继续相关直接响应平衡族。完整动态稳定性还须保留原连接时延、AMPA/GABA、M和R/G更新顺序，不能用DC矩阵特征值代替。不要把全mean响应接近当全native已对应；要继续检验失稳/场族/自然轨迹携带G的关系。不要为好看的分岔名继续旧代理局部转弯或追共同场K9无关根。

物理机制不变：G/K先压低活动，G再降到恢复阻断阈值2.6694以下后Z才有机会净恢复；K在R<=5Hz保留5秒给时间，而高活动K为.5秒，和G同尺度。Z参考恢复与core短事件返回不是同一时刻；未充分恢复的早reentry中已有5例按Z离散方程上界就来不及恢复。原8405首轮23.90s到Z参考、49.31s才短事件返回。

Memory已使用MEMORY.md324–325，最终citation对应01a09eae-c163-7cf2-8f2d-f11d43bdeaaf和01a0add1-6bb8-78f1-8af3-98dc7b0724b2。无memory写入。
