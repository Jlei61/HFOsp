# 当前交接：较低K保持与原生G30秒已完成；正在新原生K夹逼与该点DC

**Goal ACTIVE，完整目标未缩减，正式分岔NOT_ESTABLISHED。不可mark complete/blocked。** 本轮有实质进展，见goal_progress_current.json。用户要求同一原生间期传播→进入→自主退出→两核Z充分恢复→间期返回与对应机制/分支；原生状态空间已认可。保留原引擎、density_spatial.py、正式图、其他脏工作，无子agent/push/清理。新代码限scripts/topic4_loop_bifurcation。用cuda_env Python。

上一版完整交接保留`HANDOFF_before_lower_hold_and_G_completion.md`，较早证据、公式和频率限制仍有效。当前科学正文`scientific_review_live.md`已刷新。最终应向用户展示新图；本轮尚未发final。

## LIVE：先核对进程和result，绝不重派已完成任务

1. **native_exit_K_bracket**：supervisor27841，session10553；worker27851 GPU0(K9.35)、27852 GPU1(K9.5)，collector27854/session32227。原生绝对12→42秒，共30秒，最近到绝对14–15秒，仍早期；后续以实时JSON为准。与既有K9/10.5控制同12秒完整高史、同t50未来输入、同实际16.7秒Z/K场族，平均Z.21，只改变固定K。`prepare_native_exit_bracket.py`完整初态逐位QA PASS。没有新物理、没有新独立seed。原生是即时设K，密度是自己高态缓慢接近K，两者历史/外源差别明确保留。
2. `compare_native_exit_bracket.py --wait` PID29014/session53747，等待以上完整genericextendedanalysis后自动比较原生末20–30秒与密度末3秒rates/G/Zdrift/400cell场，300外源记录逐位对旧K9输入。输出`native_exit_K_bracket/density_correspondence/result.json`和每K数组。新代码只是完成后分析，未产生科学结论。若worker失败只修分析/局部失败，不重复所有仿真。
3. **held_exit_dc_K9p35**：当前点40000目标DC；part0 PID28860/session68990 GPU0，part1 PID28867/session88194 GPU1；collector28873/session75094。调用`measure_held_exit_dc.py worker`，该wrapper复用未改的`measure_all_target_dc.py`kernel/worker，但OUT/SOURCE指新K9.35。两个worker在当前点四通道bitwiseQA PASS，均在采样。每目标全/半幅、256粒子、4秒记录1秒burn、八复制block；同旧固定种子方案，与K9数值配对，不能称独立native种子。只测这一个点，无自动扫描。collector`collect_held_exit_dc.py --wait`会收齐并产生一个待验证Newton建议；它从当前inputs读取Z/K/r/M/g，**不是旧K9的Z/K或r**。正式稳定性不由DC本身成立。
4. 活跃时别改上述wrapper/helper/collector及其依赖。若需后续独立检验，新文件/新目录。不得静默重用K9Jacobian作为K9.35导数。

## 本轮新完成结果

### 较低K保持：9.35仍有高活动，9.5保持后静默

`carried_exit_lower_holds/analysis/result.json` COMPLETE。
- K9.2最后3s：allE210.970，A473.895，B475.228，surround197.643；Graw1.16749。全EZdot+.031109，A/B−.034834/−.033074。
- K9.35：allE200.4496，A472.3095，B475.0007，surround186.6354；Graw.112035。全EZdot+.038173，A/B仍同负值。
- 两条各保持8s，末三段1s均率相近，最后两1s空间RMS.03518/.05587Hz。未退出记右截尾；有限持续性不是稳定根认证。
- 旧K9.5/9.65两保持已完整静默。K9.5持定后约1.83s才退出，所以原慢轨迹经过K9.5时的高率为暂态。这里只夹住这组历史协议下转换，不证明所有高态解消失。
- 所有已完成lower/gcontrols/oldramps旧PID已退出，勿重派。

### 原生初始G完整30秒：空间历史效应，没有核心退出

`exit_actual_G_history_probes/G_history_comparison/analysis.json` COMPLETE（注意文件名analysis.json不是result.json）。两条新干预+复用基线，300外源100ms记录逐位相同；初态只有G不同。
- baseline initialG4.5113：tailallE242.578/A464.533/B470.765，Graw4.33858，allEZdot−.042、两核负。
- initialG0：243.824/A444.810/B470.143，Graw4.46307；空间场与基线RMS76.715Hz，allE和两核Z方向仍负。
- initialG11.495：209.647/A471.559/B470.087，Graw1.0354；空间场对基线RMS267.213Hz；allEZdot+.044487但两核−.034834/−.033074。
- 三条30秒均sustained_high、无末窗brief events、无R<=5。强初G在这个固定Z/K和起态下不足以自主/条件退出；改了空间招募，但不能独立宣称多个吸引子/盆边界。不能把反事实Zdot当实际Z恢复。

### K9.35新独立静态响应，未认证精确根

`held_exit_stationarity_K9p35/result.json` COMPLETE，两个数值流已退出。producer`audit_held_exit_stationarity.py`。
- 每流40000物理目标×256粒子、4srecord/1sburn，新seed929351/929352；原局部密度细胞方程，供给保持末3s实际群率、逐目标平均M、保存Z/K和相同期望外源。群率重新float64累积。
- direct allE200.4450对密度200.4496；A472.2432对472.3095，B474.9321对475.0007；fieldRMS **.017526Hz是vs密度，不是vs原生**。
- 源群残差weightedRMS：E.03076、I.22367、A.12395/B.13136Hz；各群计数SEM .0046/.0915/.00395/.00304。E有297/1885群超6countSEM，I29/1594。individual E M residualRMS.21517Hz。没有精确root/stability pass。
- 误差不能全归于MC：3秒末窗与4秒coldstart记录的phase/count边界、固定M、输入均值和densityclosure独立存在。核心最大群残差约.33334，很像有限窗计数效应，但仅此数字不是因果证明，**不可直接放宽门**。新DC用于当前点校正；不要反复追逐同一个有噪声映射到任意1e−6Hz。
- prepare首次在任何新计数前因Z逐位检查中止。旧audit对128个相同Z取mean造成~1e−15roundoff，external重聚合同量级；已验证，重建原参考input误差<1e−9，再用当前实际Z/K/external。无科学阈值或physics修改。DirectDC在prepare只用exactmoments、S/field，旧J从未使用。此执行细节已记录QA/科学说明。

### 两张新图已实际打开和修好

1. **figures/carried_K_transition_bracket.png/.svg**：在原ramps+保持图上加入全部4个保持端点，三角显示rates/G/counterfactualZdot；下排仍是慢ramp空间带收缩，不是4个endpoint的场。原生方块使用旧K9/10.5不同初史/外源，不是同协议roots。虚线=快2sramp，绝非不稳定支。producer`plot_carried_transition_zoom.py --include-lower-holds`，metadata和精确producer快照在`target_high_state_continuation/transition_bracket/`。
2. **figures/native_initial_G_history.png/.svg**：三列不同initialG，30srate、G、20–30s400cell场；第一幅legend已移到下左避免叠线，再次view最终PNG。producer`plot_native_G_history.py`，metadata/producer在`exit_actual_G_history_probes/G_history_comparison/`。
两者agent_visual PASS、SVG_XML PASS、human PENDING，README已写，未替换正式Figure5或已认可状态空间。现有旧图保留。

## 本轮机制反思与继续条件

- 全网平均Z恢复可掩盖两核耗竭，必须各核逐细胞恢复资格预算。原生准确恢复条件是rawII+(18−EG)Graw<95.1985，K不直接进入。Graw>2.6694时原生II>=0保证所有E不能恢复；低于仅必要，不充分。
- 同Z/K均值及同场族，初始G能改变有限空间状态；不否定Z/K作为条件参数轴，但否定二均值足以决定自主状态/曲线。G可继续是条件系统动态状态，不必未经检验另设参数。
- 慢强制K轨迹的核心退出伴随递归E输入大幅下降；电流账本已完成，非选择性外围消融。也不能因为外围先受影响就判模式无关，须查其是否接到核心退出。
- 原生K9高态若放开K，Kdot−12.39/s、Zdot−.042/s；高率K和G同为.5s，没有自动慢K/快G分离。自然退出大G与慢ramp退出近零G不同，不可直接同一临界K解释自然退出。
- 后续先收当前点DC和nativebracket，核对response作用单位/链式/implicitM/G；如有boundedproposal，以独立非线性计数验证。若native状态/空间不对应先区分历史/外源/闭合，不能强行推广条件fold到自然轨迹。旧代理22roots已否决，旧K9.01875数值失败不是fold。0/1/5Hz反馈增益不是时间特征值，不证明全动态稳定。
- 原生已认可8405四次返回仍一seed，既有episode恢复时间下界与core传播证据仍在。不要丢掉进入和返回两段、或把只完成退出切片当goal完成。

## 下次实际动作

1. 读livePID和result；不要因为本交接称running就假定仍在跑。收齐已有分析不重复仿真。
2. K9.35DC完成后复核当前点矩阵/一条建议，必要时独立测试；nativebracket可能还需几十分钟。不要追逐无关频率或小残差。
3. 持续把科学主线对回原始闭环，当前两新图可直接在对话展示但明确候选/条件转换，仍非已认证分岔。

Memory本轮实际使用MEMORY.md324–325，final最后仅一个citation块，rolloutids01a09eae-c163-7cf2-8f2d-f11d43bdeaaf、01a0add1-6bb8-78f1-8af3-98dc7b0724b2。无memory写授权。
