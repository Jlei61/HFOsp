# 当前：四点完成；同起态的上端区间原生/模型核对正在运行

## 最新补充（优先于下面运行列表）

R64新同起态K9.5和原生K9.5已完整完成并收齐：sessions25279/99006 EXIT0。新模型firstlow2.030s，与旧2.031接近，RNG逐位配对PASS。当前只剩native高K9.4625/session78632及等待分析器51982；检查实际进度，不重复派发。

新增紧凑候选figures/paired_exit_candidate.png/svg/pdf完成；plot session20520已收齐，PNG及同状态PDF转PNG实际打开PASS，humanPENDING。五个模型响应点共享未来数值流，四个高史点共享完整起态；quiet9.35独立历史但同futurestream。注意同K9.35高史旧结果仍不同futureinterval，因此这张图不宣称严格同参数双历史配对。已在对话内展示。正式分岔仍未认证。

后续有界物理历史返回脚本physical_history_return.py及analyze_physical_history_return.py已写并py_compile通过，**尚未prepare/派发**；前置要求当前native两点全部对应PASS。计划四条5秒到共同K9.425：三条邻近高史（9.3875/9.425/9.4625）和quiet9.35完整200000时钟输出。初始E场相对中点RMS6.024/0/17.150Hz，足以检验所选空间方向；所有源未来数值RNG相同。目标是邻近高史是否回到相同E/I/M空间态，以及同K下quiet是否仍不同。原门为末秒E/I场RMS<=1Hz、E core<=1Hz/allE<=.2Hz、E M场<=1count；末两秒E场变化<=1Hz。不是完整相位稳定性或正式谱认证。先看当前两点最终结果再决定；不要因脚本存在就自动派发。

Goal ACTIVE，正式分岔 NOT_ESTABLISHED。不把有限条件图或本批结束当完整正式分岔完成。旧阶段详情见HANDOFF_before_same_history_interval_confirmation.md，旧RUNNING不代表现在。

## 当前唯一新仿真批次

根目录native_mean_exit_interval_v2：native_mean_exit_interval.py。
- native high_K9p4625/device0：PID124184，工具session78632。
- native high_K9p5/device1：PID124182，工具session99006。
- R64model high_K9p5/device0：PID124181，工具session25279。
- analyze_native_mean_exit_interval.py --wait：PID124190，工具session51982。

原生两条10s从固定nu旧高态82s完整checkpoint出发到92s，只变heldK，Z完整场固定0.21、G/M自由。候选高9.4625复用刚完成结果；新增模型9.5从同一已完成100000时钟高态续演，恰好10s到200000，与前4探针两RNG未来流配对。此前9.5退出从72s起态，不能直接冒充同一起态边界；这批修正这一可改变解释的差异。开始前整状态除K逐位核验PASS。OUT/model/runs/high_K9p5进度独立；原生进度在各runs中，fixed_background_result写出才算wrapper完整。

初版prepare在任何job/模拟之前因观察器helper假定源旁存在held_fields.npz失败。源实际持有字段由job引用；v2直接配置已指定字段，原状态/参数不改。失败目录native_mean_exit_interval与failed_prepare_producer/prepare_failure保留。preparev2/session83813已EXIT0。别重跑已在运行的三条。

分析器会写analysis/result.json及两个配对readouts、figures/native_mean_exit_interval.png/svg、exit_conditional_branch_candidate.png/svg/pdf。原native同futureinputs及末RNG必须配对；前后5s spatial<=10Hz、core<=10Hz、drift<=.01/s以及R/G分段和退出时间<=.25s门不变。仅在两点同历史native/model都通过且低K高率/高K静默时画9.4625–9.5灰色有限转换区间，不画未经认证unstable支或Fold/Hopf。图未生成/未目视，完成后须真正打开。

## 本次已经完成

mean_exit_interval四条10s全部COMPLETE，lanes83326/97980、分析40802均EXIT0收齐。结果analysis/result.json：高史K9.3875/9.425/9.4625末窗AllE192.662/190.917/186.572Hz，core仍464–471Hz；三个点coreZdot全−.03483/−.03307。静默史K退回9.35十秒全静默，coreZdot+.16517/+.16693。4条完整ZK/时钟/两RNG逐位配对。可称有限历史依赖，不能称3稳定吸引子或正式fold。figures/mean_exit_interval与_spatial都真正打开PASS/humanPENDING。顶部零曲线与下轴重合是显示限制，数据及点图完整，不隐去静默。

autonomous_quiet_duration_review/result.json也完成，最终session51508已收齐：原生16.868–49.31低活动32.442s，采样间R严格上界4.683<5；K11.0938→.0168734整段满足tau5s指数，最大relative3.53e−11。5ln(K0/K返回)=32.442，K返回是事后读数不是普适释放门。G解析阻断解除17.55017、观测17.551。CoreA还有短暂局部不合格，17.58起两核消耗预算为0，随后至返回前原离散最快恢复式误差<4e−13；两核达原参考至少还需6.313s，实际23.90。最初从17.56立即全恢复的猜测被5.45e−5偏差否决，保留first_global_clear_prediction.py/global_clear_prediction_review.json，未改native。机制全文mechanism_synthesis_current.md已加这一段。

补充只读观察尚未产单独artifact：原/移除G/fastK经过K9.5参考坐标时分别17.644(R近0,G2.213)、16.945(R9.297,G0)、16.945(R.0276,G8.955)。不能称自然临界跨越，因当时完整Z场/历史不同；仅说明G携带状态与K消退先后不同。若使用必须保存正式来源和限制，避免按冻结场阈值解释每次自然退出。

## 之前已完成的有效对应

两个十秒历史native/leadingmean空间RMS.086/.064Hz；R1共同Poissoncounts全40k×1000spike同、末状态误差<1.4e−11；72s高史K9.5native/modelfirstlow2.016/2.031，静默和coreZdot正；R64→256K9.5firstlow均2.031、100ms空间差max.439Hz。图v2全部真看PASS/humanPENDING。空间分解高率面积先缩，局部率仍高；不是逐细胞active比例或axis因果。原式/constanttau完整56.8s对照、真实G/K中介见旧handoff和mechanism_synthesis_current.md。

## 下一步保持聚焦

先等同历史区间完整结果并看新图。不要自动全域扫Z/K、复制数、原22步统计根或随意类型命名。图和公式的联系必须同时包括G尾迹捕获低率、K慢消退给恢复余量、G消退解除恢复阻断、核心资源回升和持续间期返回。自然高率时K和G都是0.5s，冻结ZK条件图不等同准静态自然阈值。需要决定如何认证相关分支时，从当前物理模型及实际转换证据出发，不追数学上容易但不相关的根。

只增scripts/topic4_loop_bifurcation；原引擎/density_spatial.py/正式Fig5/已认可状态空间不改。无subagents/push/cleanup。PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python。Memory引用MEMORY.md324–325，rollout01a09eae-c163-7cf2-8f2d-f11d43bdeaaf及01a0add1-6bb8-78f1-8af3-98dc7b0724b2。Goal不得误标完成。
