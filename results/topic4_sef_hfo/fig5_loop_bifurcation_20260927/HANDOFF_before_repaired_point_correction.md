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
