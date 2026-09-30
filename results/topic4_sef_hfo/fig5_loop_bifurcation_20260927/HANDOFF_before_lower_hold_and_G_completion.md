# 当前交接：K变化与高K保持已收齐；正在较低K保持和原生G最后窗口

**Goal ACTIVE，完整目标未缩减，正式分岔NOT_ESTABLISHED。不能mark complete/blocked。** 用户需要同一原生网络的间期传播→进入→自主退出→两核Z充分恢复→间期返回及对应分岔/机制，原生状态空间已认可。新有限条件图不是最终验收。无子agent/push/清理授权；保留原生引擎、density_spatial.py、已认可图及其他脏工作。新脚本限scripts/topic4_loop_bifurcation，cuda_env解释器。上一交接完整保留HANDOFF_before_completed_K_holds.md及更早HANDOFF_before_frequency_and_carried_K.md，公式见mechanism_model.md/direct_response_operator.md。

## 当前实际任务：先核查PID与result，不重派

- **carried_exit_lower_holds**：wrapper`hold_lower_exit_states.py` supervisor26714/session55348，worker26724 GPU0 `held_K9p2`、26725 GPU1 `held_K9p35`。从同一个K9完整density控制末状态，以0.15K/s到9.2/9.35，各保持8秒，总9.34/10.34秒。最新时刻见progress。与之前保持一样40000x128个目标粒子、同未来数值RNG、同恒定外源期望，Z保持实际退出场mean.21，R/G/M及递归输入自由。仅此两条，没有自动下一批。
- lower analyzer26720/session46380，命令`hold_lower_exit_states.py analyze`。复用`analyze_carried_exit_holds.py`，完成一条即写lower/analysis/<name>.json/npz，两条后result.json。**wrapper会设置runner/collector OUT和JOBS，不能用旧默认命令把输出跑回旧目录。** 活跃时不要改wrapper、hold_carried_exit_states.py、analyze_carried_exit_holds.py、continue_target_high_state.py。
- **exit_actual_G_history_probes**：原生worker22033/22034，supervisor22024/session13292。source绝对clock12→42，共30秒；最近time_s39，即已跑27秒，末段仍在跑，不是失败。完整内源高史+实际16.7s场缩放meanZ.21/K9；初始Graw0或11.495，其他全部相同；未改基线Graw4.511复用。原生collector22037/session54377，comparison22229/session2413自动收完整尾窗。先核查实时结果。绝不可用前缀宣称30秒最终态。
- 已完成且不应再派发：target_high_state_continuation全部3条；carried_exit_fixed_holds全部2条；all_target_frequency_direct全部1/5Hz；direct_frequency_operator全部；review_frequency_modes0/1/5Hz全部。相关旧PID23467/23567/23984/25693/25709/25710/25754均应终止（仍以live核对为准）。

## 本轮已完成的实质进展

### 两种K变化和两个保持：不能把斜坡当平衡分支

`target_high_state_continuation/analysis/result.json` COMPLETE。先完整续接K9高态5秒、改成DirectDC同一恒定外源均值，控制保留allE244.117/core469.22/474.30/G4.493，较旧高态最后2秒400格场RMS.07250Hz。改外源期望由此控制暴露，不冒充原生同一OU未来。完整cell/ref/RNG/延迟环/clock/M/R/G逐位续接，原核常量输入单步和graph100步逐位QA PASS。

K9→10.5快2秒/慢10秒变化，各再保持5秒；最终3秒两条均全E/两核率0。两核20ms率同时<5Hz持续1秒起点：快1.28秒/K9.960675，慢4.58秒/K9.687135。快的核心A先早跌，慢的B先降到350Hz；不要把哪个核先失活动说成普遍定律。慢变化的全E降到200在K9.2716，B降到350在9.63756，A在9.65856。

`carried_exit_fixed_holds/analysis/result.json` COMPLETE。随后从同一K9高态以同慢速接近K9.5和9.65，各保持8秒，**两条也均静默**。K9.5两核持续低率起点5.16秒，约在结束K变化后1.83秒；9.65为4.60秒，约保持.267秒后。末3秒率全0，Graw约8.01e−7/1.08e−7。M仍在消退（核心均M约4.79/4.17及1.07/1.02），不能说精确全状态平衡。

这改变了下一步：原K9.5经过轨迹的高率段是暂态，不是可直接标稳定支；也不能因两条历史静默就证明所有高态解不存在。故当前把有界保持向下收紧至9.2/9.35。若找到持久活跃状态，再用保存的stationary_candidate_observations和独立直接细胞响应检查自洽，更新相应点导数，不能直接沿用K9冻结Jacobian当真导数。**没有新的正式root/稳定/分岔类型被接受。**

### 已交付并实际目视的新图

当前推荐 `figures/carried_K_ramps_and_holds.png/.svg`，producer`plot_carried_transition_zoom.py --include-holds`。PNG实际view，标题重叠已修；SVG XML PASS，human PENDING。数据/metadata/精确producer快照在target_high_state_continuation/transition_zoom_with_holds。README已写。
- A两种速度的rate–K轨迹，黑三角为K9.5/9.65保持后的0Hz；开方块为旧原生固定参数最后20–30秒均率（不同初态/外源路径，不能叫同协议的根）。
- B G随K下降；水平线只标原生Z恢复必要条件。密度GaussianI可为负，不把原生严格界无条件移植到所有近似细胞。
- C固定Z处的反事实恢复方向；全网先转正，两核后转正，不是观察到Z恢复。
- 下排展示慢变化0–.2、1.5–1.7、3.5–3.7、4.35–4.45、4.7–4.9秒活动带收缩，窗口事后为展示选取，原预定窗口图`carried_high_state_K_ramps`保留。虚线仅表示快速度，**不是不稳定支**。
- 较早`carried_high_state_K_transition_zoom`不含保持点，当前图优先。正式Fig5未替换。

### 电流与自然流预算，防止解释走偏

`analyze_ramp_current_budget.py`完成，目录target_high_state_continuation/current_budget。用自由密度轨迹实际突触电流作各目标群阈值处的输入账本。G输出在最后0.1ms更新后、原膜用更新前；0<=q<=1给出对齐误差界，连同M一步误差每核<.04mV-equiv。慢变化接近退出时，A/B兴奋输入较初始下降约662/1007，而K直接额外项仅约35；localI和G变小反而部分抵消。该结果显示递归兴奋衰减参与失衡，但**不是外围输入必要性的单独消融，也不是native电流或分岔证明**。

`native_unclamped_flow_check.json`直接取已完成原生逐步预算：K9高态若放开K，meanKdot约−12.3869/s，Zdot−.042/s；K9静默则Kdot−1.79998/s，Zdot+.158/s。说明条件长期状态不是全自主系统平衡。高活动K与G同为.5秒尺度，不能假设K比G慢；自然退出R首次<5时Graw仍约10.57，慢强制K变化退出Graw约.009，不能用同一临界K直接解释自然退出。未来相关条件分支仍需叠加实际G记忆、K/Z流和空间场变化。是否需要把G作为第三个慢状态或分析完整K/G动态，必须按证据判断；此处没有新增releaseK/改方程试验。

局部K9→9.0125静态输入分解和K9基线源分解保存在direct_branch_step_review/input_balance。基线两核自核EE约722/762、核外EE约717/717mV-equiv，两核间直接EE项为0。是原图模型的输入账本，不是轴方向纯因果证据。不要从此推出一个孤立核的稳定性。

### 原生初始G前缀：完整30秒仍待收

`G_history_comparison/prefix10s.json`为预定5–10秒完整窗口：基线/G0平均R243.39/244.67、Graw4.34/4.47，全E和核心Z漂移均负；G11.495初史R210.53/G1.053，全E漂移+.04438，但coreA/B仍−.03483/−.03307。100个外源100ms记录逐位相同。0–10秒都没有R<=5。意味着全网恢复方向转正可以掩盖持续核心消耗；尚不能据前缀定义最终吸引态。完整30秒分析器会比较预定0–1/1–5/5–10/20–30秒窗口与空间场。

## 以前结果必须保留的界限

全目标直接K9/9.00625/9.0125有限精度自洽通过；9.01875三低率群未达6SEM，未接受。其冻结线性残差影响全E约.000754Hz、G约.0000757，不能把数值停止当fold。但新的活动带收缩提醒：不能仅因某模式先在核外出现，就认定它与最终核心退出无关；需要实际传播/分支联系。旧NN22根因真实E导数2/32通过被否决，勿复活。

0/1/5Hz的8近单位模式已全完成，最小距离1分别.03968/.05351/.22268，主要核外/I。它们是回路频率增益，不是时间增长率；既不证明全动态稳定，也不排除未采频率。不要自动大扫频率或继续无关微小残差循环。

原生完整自主闭环、恢复预算/时间下界和间期返回仍按原接受证据。Z恢复资格是rawII+(18−EG)Graw<95.1985，不含K；平均Zdot=(资格比例−平均Z)/5。Graw>2.6694时原生所有E不能恢复；降到此线以下仅是必要条件。G/K先降活动，G再消退解除阻断，低率K保留5秒给恢复时间。两核Z达参考和短群体传播返回分开，不要求每seed都成功退出。8405的240秒4次返回是同一seed不是4独立样本。

## 下一次实际动作

1. 先核查lower两条result与nativeG两条30秒结果及实际PID。分析器已挂，失败只修分析，不重跑已完成仿真。
2. 根据9.2/9.35是否保留活动，选择真实附近活跃状态作独立静态/响应核对和native空间对应；有限持续性不能直接认证稳定根。若又都退，先检查场收缩时间过程和自洽，而非盲目继续许多K点。
3. 保持完整分岔/机制目标，但不把这张转换图标为完成。当前所有文档顶部及scientific_review_live.md已更新。无需改正式Figure5或启动更多基础/患者工作。

Memory使用MEMORY.md324–325，最终追加一个citation block，rolloutids01a09eae-c163-7cf2-8f2d-f11d43bdeaaf、01a0add1-6bb8-78f1-8af3-98dc7b0724b2。无memory写入授权。
