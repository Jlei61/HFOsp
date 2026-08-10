### lc4e_shared_executor_screen.png / .pdf

这是 LC4e 实际执行到 E1 后的四联诊断图。A 比较 cell-local 与 spatially shared 两条轨迹：二者在终止电流第一次出现以前完全一致、均在 11 s 自主进入，但到 18 s 都没有退出；B 显示 shared 臂因闭环负反馈只实际送出约 20.3 的峰值电流，低于 local 臂的约 51.4；C、D 显示 shared 分配消除了“核心被压住、轴外继续承载”的局部逃逸形态，却留下了全空间仍存活的高态。该图只支持单 seed 架构阴性，不是完整 lifecycle 或论文主结论图。

**关注点**：空间均摊改变了失败的空间形态，但没有产生自主 offset；因此空间分配不是 LC4 cell-load actuator 的唯一阻断，不能继续只加大同一执行器。
