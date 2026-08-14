# FCXR-LC5v2.1 dose-boundary patch — implementation plan

对应 spec：
`docs/superpowers/specs/2026-08-13-topic4-fcxr-lc5v2p1-boundary-patch-design.md`

状态：**EXECUTED — CLOSED（2026-08-14）**

> 最终联合判决见
> `docs/archive/topic4/sef_hfo/fcxr_lc5v2p1_timescale_dose_map_closeout_2026-08-14.md`。

1. 锁定 11 格 manifest，并验证与主实验的 model/source/observation 合同一致。
2. 对 manifest-only runner 增加 irregular-cell 支持；基础 3x3 与边界块都必须由硬编码 profile 防漂移。
3. 新 Gamma 只解析缩放 `Imax=Gamma*force/excess`，不重算 `a_U/p0_i`。
4. 使用独立 block lock、PID、日志、RUNNING/DONE/FAILED sentinel；4 个 arm 并行、顺序补满 11 格。
5. 全块完成后与基础 3x3 联合生成 outcome、onset latency、saturation、achieved-dose 相图。
6. 若出现 contained/finite，只延长最佳一格；若仍是两类直接相邻，再决定是否需要最后一个中点。

完成定义：11 格均有 terminal sentinel；无未解释 hash/resource failure；整块统一聚合后再下机制结论。
