### mz_actual_entry_lifecycle_closure.png

这张 2x3 图按 R4 锁定合同展示 actual-entry-aligned center closure。A 从 hash-lock 的 R3 trace 重算 event-5 entry、四次 core/annulus 配对回返和 event-6 suppression；B–C 展示从 20 s double checkpoint 出发的 protected q recovery、真实 latch reset、M 自然衰减和同一低态回归；D 对照 base/half-dt gate label；E 用完全相同的六事件 classifier 比较 early 与 late fork；F 给出边界化 verdict。

图中的长时间空档是经过 500 ms full sentinel 认证的 zero-use analytic bridge，不是逐点快变量积分，也没有人工把 q/M 赋回根。protected fork 来自原始 20 s checkpoint，recovered fork 来自自然恢复 checkpoint。protected fork 仍保留六次事件诱发的 section crossings；它通过的是 response-excluded window 内没有四次自主配对回返、且末段回到持续低态，而不是 electrical silence。

**关注点**：真实 state machine 是否只 reset 一次、最终 q/A/p 与 fast vector field 是否回到 LLL basin、protected challenge 是否仅有 evoked responses 而没有 autonomous lifecycle、recovered challenge 是否重新出现 event-5 entry 和至少四次配对 burst。该结果只属于 fixed-bath regional hybrid center，不代表零输入自发 onset、连续空间 wavefront 或 full SNN。
