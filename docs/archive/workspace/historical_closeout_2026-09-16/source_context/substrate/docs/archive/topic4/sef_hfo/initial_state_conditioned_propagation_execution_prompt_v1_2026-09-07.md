# 初态条件传播 v1：可直接交给执行 Agent 的 prompt

```text
请实施并完成 Topic 4「同一结构下的初态条件传播 v1」实验。

先在以下源工作树读取：
/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix

执行合同：
docs/archive/topic4/sef_hfo/initial_state_conditioned_propagation_plan_v1_2026-09-07.md
机器设计：
config/topic4_initial_state_conditioned_propagation_v1.json
输入复核：
docs/archive/topic4/sef_hfo/initial_state_conditioned_propagation_input_audit_v1_2026-09-07.json

这是独立机制支线。完成实现、资格检查、I1物理实验、按预定条件触发的I2、统计、候选图和科学报告。常规实现与恢复自行处理；结果阴性照常交付。不要只给计划或写完代码便结束。不要改主线G1/G2/G3排名、参数、正在运行的源码或患者验证释放条件。不要启动第9节的慢状态/双稳态后续项目。

1. 保存并隔离可复现的当前源码快照：此工作树有已修改及未跟踪的依赖，不能仅checkout旧HEAD冒充当前worker。使用codex/topic4-initial-state-propagation-v1独立工作树或等效隔离快照；保护原脏改动、冲突和进程。大图缓存只读引用。核对必要import/配置依赖和数组来源后再执行。

2. 按设计冻结唯一候选v2_anchor_old_joint__baseline完整参数。I1使用topology2511、dynamics820101–820112。I2仅在I1达到已定持续效应门槛时使用新图6251、dynamics820201–820212。seed若与已用实验冲突，只能在任何新结果出现前统一重新分配一次并说明。每个seed配对B0/B1/B2三臂，每次24秒；I1=36次，I2最多36次，三次正式canary包含在I1内。

3. 新增明确的initial_voltage接口及初态构造器，按完整精度h/core数组取两组等量E细胞，分别仅在t=0加1 mV。其余状态、结构、权重、阈值、噪声规律、观察器全相同，Z/M关闭。初态None与resume_state互斥；验证阈值余量，不按患者TA/TB路径选择细胞、幅度或工作点。不要用kick、forced spikes、阈值持续偏移代替初始V。

4. I0检查默认/B0轨迹一致、初态确实生效、静态数组身份相同、其他初始状态相同、实际global/spatial OU创新和Poisson计数在三臂逐段相同。记录器和初态不得消耗模拟RNG。外源输入用流式摘要校验，禁止只凭相同整数seed声称严格配对。不满足因果隔离先修复，不启动整批。

5. 从0秒记录原生场和状态摘要，冻结观察器仍用0.5秒burn-in。主对比为12–24秒B1−B2的M0比例差，以12个噪声配对运行作统计单位。不要池化事件增加n，不要求每条短轨迹含两模式，不把主线N>=16排名门槛套到本实验。任何零事件/缺失运行按设计处理null、NOT_ESTIMABLE和全12对界限，不能删去失败pair作幸存者分析。

6. 完成整批后计算配对效应、10000次配对bootstrap 95%区间、全部4096种双侧配对交换参考。只有主要p<=0.05、CI排除0、绝对效应>=10个百分点且12对全部可估计，才转I2。检查单pair驱动和统计不稳；其余按设计给小效应界限、瞬态或不确定结论，不反复加seed/换窗口找阳性。

7. 单独交付模式条件传播质量：所有可分类事件和支持/OOD数量、参与、signed timing、模式散布、冻结phi的条件D_off及其均值项/减项，模型run级和患者FIT块级参考。禁止为改善比例丢弃不支持事件，禁止重聚类/换标签。不得调用会自动打开PROBE的evaluator.metrics()；不要解封主线patient_time_packet。具体路径仅作已有FIT开发诊断，不反馈到初态设计或继续门槛。

8. 默认最多2物理worker，与主线合计不超过8且满足实测内存/磁盘准入。主线占满时等待资源并继续独立实现，不杀任务。I0估算峰值、worker小时和输出量，主动同步有用进展。遇工程故障恢复原manifest单元；runaway与工程故障分别记录。

9. 输出results/topic4_sef_hfo/initial_state_conditioned_propagation_v1/，包括manifest、初态数组、资格结果、完整逐事件/逐运行表、主要效应、条件传播质量、三臂全程状态图、按时间顺序全部事件展示及中文说明。实际检查PNG/PDF/GIF和事件覆盖，保留CANDIDATE/待用户人工审阅状态。遵循figure guide适用样式；这是比较诊断图，不强制套机制主图布局。

最后先回答：初态效应是瞬态、持续、拓扑依赖还是不确定？相应的患者传播结构有没有改善？再说明完成数、统计区间、必要限制及下一步。两张图的显著效应也不等于患者双稳态；不得把人工两个初态按患者比例拼接当成状态访问机制。终态按设计写INITIAL_STATE_ROUND_COMPLETE_PENDING_SCIENTIFIC_REVIEW，工程未完成则如实报告未完成项。
```
