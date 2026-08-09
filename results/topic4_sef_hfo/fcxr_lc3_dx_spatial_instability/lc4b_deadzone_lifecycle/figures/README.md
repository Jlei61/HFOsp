### lc4b_deadzone_lifecycle_diagnostic.png

这张四联诊断图只画实际完成的 D1-D3 nominal。A 表明 exact dead zone 在间期严格不出手，候选与 actuator-off 的群体率和 active fraction 逐位相同；B 表明这个改动没有抹掉 D/Z 进入面，固定 `D10` 时仍会进入高密事件态；C 是 70 秒无 kick 连续轨迹，5 秒进入后直到记录结束都没有自主 offset；D 把同一轨迹画到 `D`—执行电流平面，路径没有闭合回低态。

**关注点**：这不是完整 lifecycle 图。终止器实际最高只达到既往配平目标的 58.6%，末 8 秒仍是 9 次/秒的密事件串；exact-D recovery continuation 因 nominal 不合格而没有运行。
