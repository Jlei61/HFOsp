### mz_inhibitory_reserve_recovery_corridor.png

这张 2×3 图检验不改方程、只延长 inhibitory-reserve recovery timescale，能否同时修复第六事件首次 entry、维持 bounded CCO，并通过真实 latch 语义下的两阶段 postictal handoff。A 显示每个 q_hold 的 q_res/tau_D 重映射；B 显示 pre-last 与 post-last q；C 给出 30 个 cell 的联合验收；D 汇总完整 phase×source-dt 周期范围；E 展示 A 在 reset 前冻结、reset 后才衰减；F 锁定机制级结论。

当前状态为 `R2_RECOVERY_TIMESCALE_CORRIDOR_CLEAN_NO_GO_REGISTERED_GATES`，全 q_hold 联合通过的 tau_r 节点为 [80.0]，80 s 所在连续 component 为 [80.0]。

即使通过，本节点也只允许 [60,70,80] s 的短 P3 regional state-fork；fixed bath mask 仍是非涌现的，不能写成 autonomous lifecycle、continuous spatial containment 或 full SNN seizure。

**关注点**：必须同时查看 entry ordering、完整 periodic oracle、locked schedule probes、72/88 s fixed-parameter sensitivity，以及 reset 前 A 冻结的 hybrid handoff，而不能只看某个 tau_r 点。
