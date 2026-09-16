# 自主恢复与再入探索

**目标处于执行中，尚未获得自主恢复结论。** 9小时窗口：北京时间2026-09-14 23:50—2026-09-15 08:50。

- [执行设计与历史教训](execution_plan.md)
- [实时派发状态](status.json)
- [基础物理及断点续跑检查](qa.json)：两条噪声的1秒原生前缀通过逐位一致性检查；局部/全局抑制公式及双核返回检测通过。
- [旧40条完整运行复核](prior_40_run_audit.json)：25条未进入，15条经外部Z补充返回，原生返回0条。
- [实际饱和态电流复核](saturation_current_audit.json)：外部补充前的10秒窗口，区分兴奋驱动、原始/有效抑制和M。
- [随完整检查点更新的科学审阅](scientific_progress.md)
- [汇总图](figures/exploration_summary.png)

12条初始轨迹比较原生M时间尺度与抑制全局占比；每条保存真实连续SNN动态。后续必须依据实际电流、局部活动及空间场逐轮判断，再做配对确认。原log-M扫描仍只测首次进入，不把其停止后的空白当作自主恢复证据。

主执行器：`scripts/run_topic4_autonomous_recovery.py`；分析器：`scripts/analyze_topic4_autonomous_recovery.py`；自动分析与作图监听：`scripts/monitor_topic4_autonomous_recovery.py`。本目录的`qa_s*`是1秒程序检查，不是科学候选。

## 当前扩展与边界

- [第二轮：快速阈值phi](fast_threshold_round2/round1_prefix_review.md)，[状态](fast_threshold_round2/status.json)，[原生等价与续跑检查](fast_threshold_round2/qa.json)。8个有界条件，保持原生Z/M；phi是新增快状态。
- [第三轮后备：活动驱动的全局池](activity_global_pool_round3/design_review.md)，[QA](activity_global_pool_round3/qa.json)。已完成QA，现只先派发1条全局反馈候选（k=10、r0=50Hz、tauG=2s）；其余7条仍未批准生产。新增群体滤波状态，所有额外抑制仍受Z耗竭。
- [持续招募的空间位置](recruitment_origin/figures/first_persistent_activity.png)：持续锁定位置不等于事件初始起源；核外先持续不应直接理解成异常事件起源。
- [M反馈容量诊断](M_feedback_capacity.json)，[后续机制审阅备忘](mechanism_review_pending_round3.md)。
- [静息条件CPU/GPU逐位等价及时效比较](quiet_backend_benchmark.json)：旧log-M的6个低活动长跑从完整40秒检查点切换原始有序CPU累加，模型和终点不变。迁移记录位于旧扫描目录backend_migration.json。

新增机制的候选另输出added_feedback图，分别标出真实phi阈值偏移或全局反馈的原始/有效电流，不能将其作用误写成原生M的作用。所有当前图仍为开发诊断，未通过人工验收。

第三轮先行条件的[证据审阅及资源修订](activity_global_pool_round3/dispatch_review.md)：只允许该1条使生产上限暂时达到25，其余扩展仍按24。实测约179GiB可用主存、80逻辑CPU、每GPU约13GiB可用；所有旧轨迹保留。

## 当前执行修订（北京时间02:05，替代上述早期资源/PID快照）

第三轮现由 `supervise_topic4_reviewed_global_batch.py` PID84002接管，保留原sentinel状态，已批准4条κ=10的r0=0/50Hz × τG=.5/2s对照；κ=5四条仍未派发。总生产上限28、GPU上限24、新增派发需主存余量120GiB；现28生产中含6条已完整迁移CPU的旧log-M任务，GPU22条。原物理执行器及在跑任务不变，总新增科学条件仍为24/48。实时状态和activity_global_pool_round3/dispatch_authorization.json优先于本文件旧快照。

旧log-M的6条强M低活动任务均已从完整40s检查点迁移到经整份状态逐位验证的CPU实现，现继续约100–130s；36/42新条已完成，另6条复用。原第二轮6条运行、2条等待旧任务名额。当前尚未确认自主恢复。

新增观测版 `record_topic4_global_candidate_native_fields.py` 已通过1秒全脉冲、全部旧观测与整份末状态逐位一致检查。它在选定时间窗以原生10kHz保存电极电流幅值代理和直接1mm细胞场（不经电极投影），并以1ms保存Z/M/实际抑制与全局反馈。仅QA已执行，正式候选重演尚未派发。原legacy readout是|IE|+|II|，且II在乘Z之前；不能把它的未滤波二阶矩称为1–150Hz能量，也不能把它的全局抑制亮度误当作神经元招募。最终候选需先低通/带通再降采样，展示实际乘Z后的代理并明确单位。

有限事件审阅另保存真实持续高段和双核/全局安静区间，检查滚动低窗确认点是否落入下一次高态。它不改变原操作性检测门；若时间重叠，必须审阅实际轨迹，不能只凭标签宣称高—恢复—高。
