### mz_m_gated_reserve_recovery.png

这张 2×3 图检验已有的无量纲 M 状态能否只通过切换 q 的恢复速率，化解 R2 中“发作前需要慢恢复、退出后需要快恢复”的冲突。A 显示 M 只上移唯一稳定的 q-nullcline；B 保留 R2 的 entry/schedule 边界；C 使用重新生成的 24-cell fixed-q M-ramp，而不是把 M 跳到终点；D 给出 primary arm 的完整 path gate；E 显示 latch reset 前 M 冻结、reset 后才以 12 s 释放；F 列出 fail-closed verdict。

当前状态为 `R3_M_GATED_RESERVE_RECOVERY_PATH_SUPPORTED_SHORT_P3_FORK_UNLOCKED`，tau_fast=20/15/25 s 对应的通过节点分别为 [80.0, 90.0, 100.0]、[80.0, 90.0, 100.0]、[80.0, 90.0, 100.0]。

这仍不是 coupled lifecycle：q 没有反馈到 fast regional dynamics，fixed bath mask 也仍是 imposed boundary。即使通过，也只解锁 tau_slow=80/90/100 s 的短 P3 state-fork。

**关注点**：必须同时看 frozen-sensor q excursion、动态 fold margin、120 s reset、gate-off R2 parity 和 gate-only nullcline control；不能把 recovery gate 单独写成 termination。
