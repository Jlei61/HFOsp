### lc4c_entry_offset_alignment_diagnostic.png

这是 LC4c 已执行阶段的诊断图，不是完整生命周期或论文主张图。A 显示新的 H 阈值仍允许网络在无 kick、无 reset、无参数 step 的条件下于 11 秒进入，进入前有 29 次 returning IED，且前 4 秒终止器电流严格为零；B 显示同一锁定候选在 70 秒 nominal 轨迹中于 11 秒进入、66 秒自主退出，但高态持续 55 秒，远超预注册的 1–5 秒。C 显示终止电流、去抑制程度 D 和 relay 耗竭随时间的变化；D 显示慢变量路径确实折返，但 offset 后只观察了 4 秒且没有 returning event，因此不能判定统计恢复或低态稳定性。

**关注点**：本轮第一次在一条无外部干预轨迹中同时观察到 cumulative entry 和 autonomous offset；但 offset 太晚，既不满足 ictal bout 时长，也没有留下 8 秒 returning-IED 验收窗，故完整 lifecycle 仍未成立。
