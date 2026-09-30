# 当前交接：全空间频率响应已结束，完整高态K续接与原生初始G对照在跑

**Goal ACTIVE，完整目标未缩减，正式分岔NOT_ESTABLISHED。不能mark complete/blocked。** 用户要原生间期→进入→自主退出→两核Z充分恢复→间期返回，以及真正对应的分岔/机制。不要用局部数值成功交差。无子agent、push、清理授权；保留原生引擎、density_spatial.py、已认可状态空间/正式Fig5、其他脏工作。新脚本限scripts/topic4_loop_bifurcation；cuda_env解释器。上阶段详细事实见HANDOFF_before_frequency_and_carried_K.md / HANDOFF_before_direct_branch_steps.md / DIRECT_RESPONSE_HANDOFF.md。

## 首先核查实时任务，不重复派发

1. **actual_G_history**：`exit_actual_G_history_probes` supervisor22024/session13292，原生worker22033 GPU0（G0）/22034 GPU1（G11.495）。各30秒，从原12秒完整高史开始，绝对clock12→42；同真实16.7秒Z/K场缩放到mean.21/9、同t50未来外源。未改基线G4.511复用exit_return_probes。完整初始状态只有Graw一项不同，QA PASS。两条仍运行，勿因慢而重复。collector22037/session54377，comparison22229/session2413，结束自动写完整分析。

2. **carried_K**：`target_high_state_continuation` supervisor23467/session23370。K9恒定5秒对照已COMPLETE并保留高态；现worker23983 GPU0 `ramp2s_K10p5`（总7秒）、23984 GPU1 `ramp10s_K10p5`（总15秒）。实际进度从各progress读。两条从同一个控制末状态出发，K9→10.5分别用2/10秒，末端各保持5秒；没有其他自动扩张。producer`continue_target_high_state.py`，工作时不可修改。

3. **carried_K analyzer**：23567/session26324，`analyze_high_state_continuation.py --wait`。完整一条即写analysis/<name>.json/npz；三条完成后写analysis/result.json和figures/carried_high_state_K_ramps.png/svg并追加README。**生成后必须实际view_image**，当前未生成、agent/human均PENDING；不要把强制K轨迹称为分岔图或自主退出。若collector失败只修分析，不重跑仿真。

4. **频率任务全部结束**：20790/21019及1/5Hzworkers均已完成。`all_target_frequency_direct/progress.json` TWO_DIRECT_FREQUENCY_PROBES_COMPLETE；`direct_frequency_operator/analysis_progress.json` TWO_LOOP_FREQUENCIES_ANALYZED。`review_frequency_modes.py --frequency 0/1/5`全完成。不要再次启动这些任务。

## 最新实质结果

### 完整空间频率与误差分解

`frequency_mode_review/summary.json`按8个计算特征值中离1最近者选择，不是最大实部。
- 0Hz λ=.96031976，距离.039680；MC投影SEM.000380，全/半幅差.000093；两核质量合计.247%，核外70.94%、I28.82%。
- 1Hz λ=.96472654−.04023455i，距离.053507；SEM.000620、半幅差.000985；两核.290%、核外70.79%、I28.92%。
- 5Hz最近是 **.91720653−.20671736i**，距离.222681；SEM.000443、半幅差.001034；两核.0690%、核外68.30%、I31.63%。最大实部.9382747−.2156376i距离反而.2243，勿混用。

区域质量为细胞数加权的源群特征向量绝对幅值份额，不是率或细胞比例。双正交左右模式用于8replica块的一阶MC特征值误差，排除了幅度/有限窗口/源群/fixedM近似误差；所有弱通道和幅度失败保留。L(e^iw)特征值是回路增益，**不是时间增长率**。这三个频率既不认证全动态稳定，也不能排除未测频率失稳或命名Hopf。

L=A+u wT，A保留相同基线G但固定G扰动。eta=wT(I−A)^−1u在0/1/5Hz为−1.19529、.108445+.230737i、.0167224+.0054706i；1−eta只是全局秩一反馈行列式因子，不独立认证稳定性。当前最接近临界的是核外/I招募模式，尚未连到两核终止。因此**不要自动扩展全频扫描或追微小无关尾部残差**。

### 实际场初始G：完整5秒前缀，不是30秒最终结论

`exit_actual_G_history_probes/G_history_comparison/prefix1s.json`及prefix5s.json。三组50个已记录外源100ms样本逐位一致。
- 基线G4.511：1–5秒平均R243.43Hz，G4.289，全E dZ≈−.042/s。
- G0：1–5秒R241.53，G4.379，全E dZ≈−.042/s。
- G11.495：首次1秒R最低107.16；1–5秒平均R199.32、G1.011，全E反事实dZ **+.04680/s**，但coreA/B仍 **−.03483/−.03307/s**。三条0–5秒均没有R≤5。

Z/K是固定的，dZ仅为原方程在固定场处的反事实方向，**不是已经恢复**。新信息是“全网平均可转正而两核仍完全消耗”，全局必要恢复阻断解除并不替代局部I/核心资格。不能根据前缀报最终吸引态或自主闭环；完整30秒尾窗未到。

### 完整高态续接：基线已验证，两个强制K轨迹在跑

先前K10.5/12的native与density均从原12秒高史突然钳制；该原史Zmean.5463、Kmean3.294，故其静默不能独立证明邻近高态不存在。当前先从`target_density_exit/individual/final_state.npz`完整density高态续接，所有40000x128细胞/ref/RNG/延迟环/clock/M/R/G逐位保存。原clock100000保留，过期pending不再注入。

常数外源期望使用DirectDC同一个mean_external_rate_per_ms；这是代替旧timevaryingOUmean，并非native同一OU未来，必须注明。恒K9五秒控制已完成，末2秒全E244.117/core469.222/474.297、G4.4933；与旧高态末2秒空间400bin RMS **.07250Hz**。见control_K9/constant_mean_comparison.json。没有以这一检查认证全面native动态。

两条从control完整末状态clock150000出发，同一个未来数值RNG，采用原held_cell，仅每步前按初始K场比例线性改K；Z不改，R/G/M/递归输入自由。CPU原局部方程原有QA沿用；本轮原kernel恒定输入单步逐位、100步graph vs逐步逐位、K开始/中段/末端日程和Z保持核验PASS。条件是强制参数变化，不能计入自主闭环。

分析20ms非重叠core率，两核同时<5Hz持续1秒才记录有限转换；原1ms率保留。对比末3秒、两个变化速度，以及慢变化固定0–.25/4.75–5.25/9.5–10/14–15秒空间图。若高态继续维持，之前突然钳制静默不能说高态不存在；若失去两核，需要在真实相关位置进一步验证分支及速度/历史作用。

### 局部分支已完成的影响与电流分解

K9/9.00625/9.0125直接有限精度检查通过；9.01875仍三低率群超6SEM，未接受且没有改gate。`direct_exit_K_pair/point_01/residual_impact/analysis.json`：三个残差经过冻结K9线性逆，预测均E变化.000754Hz、Graw.0000757。局部诊断不是严格界，不恢复失败点。

`analyze_direct_branch_input_balance.py`已完成，数据`direct_branch_step_review/input_balance`：以各目标组阈值作参考，逐项分解9→9.0125的平均电压漂移，代数重构误差3.65e−12mV-equiv。CoreA的核外兴奋减少约−1.099，K项−.657，被localI+ .324和Graw+ .338部分抵消；总−1.342，而原平均阈上漂移约653。CoreB核外兴奋−.294、K−.666，localI+.073、Graw+.321，总−.666，原约690。说明该微小步已减少外围提供的兴奋，但两核仍远处于高驱动区。它是近自洽输入账本，不是原生电流/消融或终止证明。原内联首次bool负号代码错误在任何结果前失败，已修为float乘法并独立输出；无仿真受影响。

## 物理与目标边界

E: tau_m dV=−V+IE−ZII−etaM M−ZGraw(V−EG)−K(V−EK)。EG−17.66285/EK−30，etaM.0005，tauM1s。因果R15ms，qclip((R−200)/300)，Graw目标30q/tauG.5s；每spike K增加.16q；K消退在preR≤5为5s，否则.5s。Z资格原式rawII+(18−EG)Graw<95.1985，tauZ5s，K不直接进入Z资格。平均dZ=(资格比例−平均Z)/5。原生II非负，因此Graw>2.6694时所有E不能恢复；低于只是必要条件。

原生16.7s参考 **Zmean.213404、Kmean12.660267、Graw11.495041、R428.306Hz**；不是条件K9平衡。G/K相同.5s高活动尺度，不能先验把K作G的慢参数。原生8405首次充分核Z参考23.90秒，首次完整短事件49.31秒，二者分开；240秒4回归同一seed，不算4独立seed。恢复链/G尾部、15episode/3seed的最短Z恢复下界等保留早期HANDOFF与mechanism_model.md。

## 文档/下一步

scientific_review_live.md已更新，旧版本scientific_review_before_frequency_and_carried_K.md保留；mechanism_model/bifurcation_scope/completion_audit_live顶部已刷新，direct_response_operator追加完整频率解释。current_stage_snapshot含新结果。完整旧原始/近似失败结果全保留。

下一次先读现有任务result/progress和实际PID：收完2条K变化（分析会出PNG，必须目视）及2条原生初始G30秒尾窗。依据两核是否终止、核心dZ以及速度差异选择真正相关分支位置。无需重新测0/1/5Hz，无需用旧NN22根。不要仅因还未正式分岔就继续大面积扫参数。

Memory使用MEMORY.md324–325，最终追加一个citation block，rolloutids01a09eae-c163-7cf2-8f2d-f11d43bdeaaf、01a0add1-6bb8-78f1-8af3-98dc7b0724b2；无memory写入授权。

## 较快K变化已完成（当前最新）

`ramp2s_K10p5`全7秒已完成，末3秒全部E及两核率为0，G约0.00055；不是自主闭环，因为K由外部指定、Z保持。两核20ms率同时<5Hz的首个连续1秒区间为1.28–2.28秒，开始时K约9.961，确认时K10.5；不能把这一强制轨迹位置直接标为fold。

`analysis/fast_ramp_sequence.json`给出事后描述性20ms窗口：K约9.37时全E209Hz、两核460/468Hz；K约9.56时全E165Hz、两核399/466Hz，此时全E反事实Z漂移已正但两核仍负；K约9.89时核A约2.8Hz而B仍179Hz；K约9.97时两核接近静默，随后核心恢复资格出现。它建立了核外收缩→核心A/B先后停止的条件轨迹联系，但较慢10秒变化尚在跑，速度依赖/相关平衡边界未定。以慢变化完成后的双路径比较决定后续，不直接加新参数批次。
