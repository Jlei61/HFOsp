# Z/M minimal-carrier branch-decision 图说明

本目录是动力学诊断图，不是 paper-ready lifecycle figure。当前阶段未运行或被阻断的证据不得由空白面板代替。

### phase0_state_and_resume_parity.png

展示 canonical Z/M 状态清单、checkpoint round-trip 与 exact-resume 门。这是工程/状态语义门，不是 carrier 阳性证据。

**关注点**：状态字段完整且 split/resume 与连续运行一致。

### anchor_trajectory_bins.png

展示三个 primary seed 的 Z/M+S_G 自然轨迹，以及被选中的 slow-state bin 和 fast-phase snapshot。三个 seed 都出现随 z 耗竭而加密的间歇事件，为 frozen-state 测试提供自然访问状态。

**关注点**：这是 fork 来源，不证明自然轨迹已经进入并退出有界 ictal state。

### carrier_subsystem_matrix.png

展示 completed fork cell 的 Jeffreys carrier posterior 和失败类型；空白表示未运行。freeze_all 在相邻 slow bins、多个 fast phases 和多个 seed 上形成兼容 carrier window，说明冻结慢状态后 fast network 可维持高活动支。

**关注点**：这是 visited frozen states 上的 source-space carrier，不证明动态 Z/M 会自然到达或维持它。

### carrier_continuation_dynamics.png

代表性 freeze_all continuation 在 burn-in 后维持约 151 Hz、CV 约 0.001 的高率状态，并持续远离间期 rest basin。该结果确认了有界、持续的 tonic carrier branch。

**关注点**：宏观率近乎平坦，不能称为 ictal oscillation、bursting orbit 或 limit cycle。

### native_confirmation_spatiotemporal_morphology.png

并列展示 seed1/3 的原生 dt/2 8 秒与标准 dt 20 秒 confirmation，包括 population rate、轴向 kymograph 和多触点 readout。长时轨迹保持约 136–160 Hz，宏观 source rate 与空间分布近乎静止。

**关注点**：确认的是持续 tonic source-space carrier，不是传播性 ictal pattern；真实 observation reference 仍被阻断。

### fine_source_rhythm_and_phase_map.png

用细时间分辨率检查同一 carrier 的 E/I source rate、局部主频振幅和相位。seed1/3 是 asynchronous/irregular candidate，seed4 是 phase-staggered periodic candidate，跨 seed 分类不一致。

**关注点**：不能从单 seed 相位图外推统一 periodic attractor；class disagreement 使 modal audit 按合同跳过。

### readout_impostor_discrimination.png

展示合成 broadband carrier、尖锐谐波 pulse train 与全局固定振荡器在 readout 指标上的可分性。这是 observation gate 的 synthetic sanity check。

**关注点**：真实 observation claim 仍需被锁定的 returning-event/early-ictal 参考窗。

### slow_coordinate_effective_rank.png

展示真实 early-to-late Z、M、S_G 场方向的中心配对覆盖。当前 M 方向在三个 seed 均缺少满足物理边界的中央位移，因此没有 seed 可进入奇异值谱判决。

**关注点**：正式结果是 no_evidence_incomplete_central_pairs；这既不是 rank-1，也不是 rank-2/3 证据。

### conditional_z_entry_boundary.png

展示从 matched pre-entry fast state 出发，沿真实 pre-entry→carrier Z 场方向插值时的 P_enter；M 和完整 S_G family 固定在 onset-adjacent 值。误差条是点级 Jeffreys 区间；本轮没有得到有效 boundary bootstrap。

**关注点**：所有采样点都位于高进入概率侧，P=0.5 未被包围，因此 entry boundary unresolved；不能写成 Z 的全局充分性或已定位的 entry bifurcation。

### existing_slow_coordinate_offset.png

比较 M、M+S_G 和 M+Z-recovery 三条真实慢场方向的 P_remain，并显示 matched-low basin 与动态 Z+M 实现。点旁 k/n 明示自适应扩增造成的覆盖不等；当前正式 offset 结果为 no_evidence，dynamic Z/M 为 9/9 runaway。

**关注点**：static M+Z 的非单调曲线不能解释成已定位分岔；离开 carrier 也不等于返回原有间期事件或建立完整 lifecycle。

### phase_completion_status.png

展示 Rev3.1 各阶段完成度、阻断项和当前 fail-closed verdict。carrier window 已在 source space 确认，但 observation、entry、offset 与 lifecycle 分层显示。

**关注点**：确认对象是 frozen-fast source-space carrier；offset no_evidence，Phase 3 与 actuator 均未授权。
