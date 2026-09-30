# 当前交接：目标输入异质性修复完成，邻域对应正在运行

Goal ACTIVE，完整目标未缩减，正式分岔 NOT_ESTABLISHED。用户要求完成Figure5自主闭环、对应的分岔和完整机制，持续防止走偏。不要mark complete/blocked。无子agent/push/清理；保留原生内核、density_spatial.py和原正式Fig5。所有新代码scripts/topic4_loop_bifurcation，用cuda_env。旧详情在handoff_archive_before_target_repair.md；实时状态current_stage_snapshot.json。

## 最新突破（已核对真实输出并看图）

1. native_count_noise两条10–20s全完成，paired_input_audit PASS。原global/spatialOU全路径及终点RNG逐位，唯Poisson计数流928701/2改变。两条均13.744–13.748s退出、17s再激活，density13.75也早退；原8405为16.868。故不能把3s时间差单独断言系统闭合失败。原生再激活场仍更广、两核更强；不是间期返回。见native_count_noise_review，相关exit/space图已看。
2. 旧原生所有清单全完成：native_slices30/30、extensions6/6、entryspatial4/4、entrynoise2/2、exitreturn8/8、independentnoise10/10。没有这些旧worker。原有共同场K6/K9两史差异持续120s；有限历史不是稳定性认证。
3. exit_branch_density四条10s全完成：实际16.7s Z/K场，Zmean.21，K9/12×高/恢复史。K9high原生全E242.59、G4.34、Zdrift−.042；groupdensity全E216.64、G1.73，Zdrift端点+.0263。K9recovery和K12两史均静默。不能把相同高/静默类别当机制对应。图exit_branch_density已看。
4. native_exit_branch_inputs：原生K9high42–44s只读续演20000步完成。初态完整42s逐位、heldZ/K和counts汇总PASS；无预存44s参考，不能称终点重放逐位。全部3479源群count及16几何群(210cells)局部moments/初态/队列/drive/RG。
5. conditional_exit_branch_density4项完成：projected/measured群均输入×数值928731/32，8192粒子/群。区域均值表面接近，但edge531(13cells)原生55.54Hz，两均值arm都是0；选中I低约2–2.5%。图加第四列明确反例，已看并修复标签。
6. conditional_target_heterogeneity6项完成：同210cells、原始目标cell入度/权重；homogeneous、individual_mean、individual_full×数值928741/42，1024/cell。edge原生55.54，hom0，mean55.75–55.77，full55.36–55.39。E/I群内currentSD原生121.93/103.75，hom16.56/14.85，mean121.41/102.08。群电流均值误差相同，修复来自目标输入异质性，不是整体加兴奋。算子重新groupavg严格恢复旧mean/variance；全部原始平方权重保留。图target_input_heterogeneity已看。
7. **target_density_exit配对自由网络两条10s全完成，决定性支持修复**：40000targets×128，seed928751，originalhigh12s初态，实际Z.21/K9，same exactdrive50–60，groupmeanθ/external保留，only targetinputprojection改变。hom全E216.640/G1.7366/Zdrift+.02617，individual244.111/G4.493/Zdrift−.042，native242.589/G4.340/−.042。400格weightedRMS145.03→6.357，MAE64.17→3.751。两核individual469.23/474.30 vsnative464.55/470.68，仍略高。图target_density_exit已实际打开，agentPASS/humanPENDING；这是固定Z/K高态一个工作点，非完整闭环/分岔。

## 当前运行（按实时status核对，不重复launch）

- exit_midpoint_probes：**仅新增原生K10.5、Z.21、实际16.7s场×高/恢复史各30s**，同未来source50s输入。spec=exit_midpoint_spec.json，用原spatial_probes.py prepare/worker。supervisor13842（v4）、collector13843。首次worker13858/13859均GPU1，已经真实派发；futurestatus为准。两native是新参数点的条件历史，不是自主新seed。collector完成后analyze_native生成extended_analysis_summary/readouts。
- target_density_field_family：5条×10s：K9recovery、K12high/recovery、K10.5high/recovery。sourcefolds分别exit_return_probes/exit_midpoint_probes。**supervisor14178** 每GPU一个密度worker，原两native也占GPU1但总显存够；禁止同一job重启。第一次worker在该root/status.json。producer target_density_field_family.py，参数/物理不调，全部individualtargets40000×128、seed928751。同50–60精确外源。
- **collector14261** analyze_target_field_family.py --wait：等待候选结果及native完整readouts逐条收集，最终figures/target_density_field_family.png/svg。这张图尚未生成/目检，不能提前README或visualPASS。
- 两条旧target_density_exit workers13495/13496已正常结束。旧collector13567曾因40格geometry1600cell用于400cellplot报错，已停；改collector仅用ADAPTED20×20显示mapping，v2=13841正常收尾，target_density_analysis_fix.json/原log保留，仿真未变。

## 本轮源文件与实现边界

- target_density_exit.py已完成/被family继续import，**不要修改该父producer**：family合同锁sha。它复用冻结physical.native_cell，exacttargetoperators mean/variance from原graph，source仍3479g40groups。保留原cell jointstate与pendingqueue；M/R/G自由、Z/Kheld。新CPU/GPUlocal在ticks0/358/359误差<2e−11、GPU/CPUdelay<1e−8、100stepcapture全状态/ref/RNG/history/global/output逐位PASS。每1ms输出group_output9channels末项Zeligiblefraction，是逐粒子predicate；counterfactualZdrift=(eligible−Z)/5。Native比较为20ms步预算，不混淆统计。
- family构造先用父类加载相同图/drive，再在任何step前完整替换sourcehistory、heldZ/K、ref、pendingrings、global，reset。只把剩余条件具体化，没有改父kernel。I groupθ仍groupmean、external仍成员groupmean、源群内各cell firing identity仍平均，这是保留的近似。
- prepare_target_density.py算子合计ampa22.47M/gaba7.16M nnz，CPU/重聚合核验PASS。不是新拟合。

## 后续真正需要完成

先收固定2native+5density，核对target_field_family动态/空间/G/Z方向，直接看新图并做README/metadata。这会决定是否能用修复后的条件系统在实际退出K9–12（中点10.5）继续相关稳定/不稳定支。不要重启共同场K9失败静态根，不要再全域MLP拟合，不把30s/history差异称双稳态。仍缺正式相关分支/稳定性和自然轨迹对应；K高态tau=.5s同G，不能强做slowK quasistatic假设。

科学解释：Z恢复资格比例必须超过meanZ；G先压低活动也暂时阻止Z恢复，须衰减后释放；K在causalR<=5时的5s长尾提供恢复时间，R>5时.5s消退。不是手动30s窗口。G>2.6694在nativeII>=0时全局阻断Z恢复；如果G处于稳态G=.1(R−200)，这对应R226.69，但不是新开关或普适发作阈值。Z恢复与core主导短事件返回分开。先前5次过早reentry时间不足以按Z方程达到参考水平，不能通过“短暂安静”宣布闭环。

文档：target_heterogeneity_review.md、exit_branch_density_review.md、mechanism_model.md、scientific_review_live.md、bifurcation_scope.md、completion_audit_live.md已更新；新看过图README与visualPASS/humanPENDING完整。Goal仍ACTIVE，本轮是实质progress。

Memory已使用MEMORY.md324–325，最终citation对应01a09eae-c163-7cf2-8f2d-f11d43bdeaaf和01a0add1-6bb8-78f1-8af3-98dc7b0724b2。无其他新memory变更。
