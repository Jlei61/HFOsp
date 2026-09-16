# 05:14阶段记录

前一轮progress，本轮新增配对科学分析与实际图，亦确认具体物理PID仍live。Goal仍active，09:40前尚有约4小时25分钟；不是blocked。

## 新结果：同状态、同后续噪声三臂50秒对照
新增scripts/analyze_topic4_reset_matched_prefix.py，输出reset_matched_50s/analysis.json、scientific_review.md、figures/matched_reset_prefix.png/pdf、figure_qa.json。
三臂Z-only、76.5秒M清零、额外V/ref/synapse/delay清零，固定η=.02 τM=2 τZ=5同seed。统一76.5–126.5实际闭合数据，无新仿真，无新的统计样本，不截短原随访。
- 输入摘要三臂逐位相同；Mclear/fullfast每100ms完整输入向量digest逐块一致。
- 每臂500000×80原始raster、50000个原生1ms E/I计数、10000个5ms Z/M点；连续无缺口，E/I区域计数一致。
- 已view_image。连续raster、4行共轴、M原有7.925电流衰减与Mclear从0增长清楚。正式Fig5 C依然Z/M双轴，这只是附加机制诊断，不冒充完整Fig5。

前2秒Z-only E=.894Hz、Mclear11.807、fullfast13.547，说明残留M确实压低最初的有限活动。10–50秒E分别13.854、13.704、13.870Hz；Z .844945/.843079/.840321；M电流 .549139/.545118/.554222。粗平均接近并不证明完整状态或再入概率相同。50秒均无全E200Hz持续200ms。

实际mz_slow_vars.py M使用Euler而非精确指数：m[k+1]=(1-.1/2000)m[k]+spike。用这条方程分解Z-only释放时旧M的齐次分量，20秒后只剩.000359726电流（该时刻实际M的0.07289%），49.995秒约1.10e-10。给定真实轨迹的线性M组成分解，不是无M历史反事实；旧M经早期放电影响Z/突触的间接状态效应仍不能排除。新增该分解重跑后PNG逐字节相同，延续已目视通过；未改物理源码。
因此当前可解释的是早期M遗留抑制，不能把它直接当成1000秒未再次runaway的完整解释。

## 当前运行（05:12实时）
- Z117/147结果、30运行。M40 0完整/13运行；新lookup e0_t1_s9108402 PID249537已正常到5s,Z=.8684,Mcurrent=.1458，无进入，非仍building。
- dense211021已11s及与原11s progress parity PASS；尚无11.5保存。CPU2392s且live，不因10–20s原chunk缺失重启。prefixwatch211022仍0新图，需实际≥11.59s，通常12s闭合块才能满足完整1s E2目标。
- 原early2臂live，source1最近12s，均尚无restore已保存。η=.2高态gain最近77s，live10159+CPU秒；原5s终点80.5不改。
- Mclear3202691、allfast4073743均live，CPU48140/12436s；未完成长程，不宣称永久阴性。
- comprehensive_inventory/latest.json/md已刷新，所有追加分支也纳入。

## 科学约束与接续
不调整正式E2窗口来追逐相关/正能量：现有早期定位audit已显示局部起点/全局高率起点不同，正式Fig5仍如实呈现固定定义及未匹配，待用户审阅科学选择。不要把1–150Hz滤波波包或高平台谱峰当作持续振荡/Hopf。
下一步优先看dense实际12s新完整布局（目前没有），检查它同条件早期能量和原生场；等待提前Zrefill后的真实返回和再入。新图需Agentview；①–⑤与完整布局分别验收。原长窗与40条网格保持既定终点，未返回绝不能用普通事件填⑤。
