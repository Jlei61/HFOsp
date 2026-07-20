### mz_inhibitory_reserve_corridor_r0a.png

这张图把原先的一维 Z-floor 假设改成二维 `(q,A)` frozen geometry。A 显示 autonomous failure 与 low-root fold 的对齐；B 验证 A=0 established CCO；C–D 给出四 phase 的瞬时 event-locked A-step fiber；E 展示一个代表性 matched-cycle 与 fold step；F 锁定 R0a 只解锁 R0b。

所有 q 的 step exit 都几乎从 `A_SN(q)` 开始，这说明瞬时跳变会绕过原 autonomous arm 的 simultaneous slow q-A passage。当前图不含 fixed-q smooth M ramp、reserve dynamics、half-dt 或 recovery/retrigger；bath q 被固定用于 oracle parity，不能把 bath-low 写成 emergent containment。

**关注点**：R0b 必须证明原 225-ms occupancy-gated M 在固定 q 下平滑穿越 fold 仍安全，再确认连续 q strip 的双 dt 与 recovery；step fiber 单独不能解锁 q_res。
