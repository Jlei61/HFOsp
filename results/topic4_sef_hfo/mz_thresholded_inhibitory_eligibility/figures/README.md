### mz_thresholded_inhibitory_eligibility.png

这张图检验慢 eligibility trace 是否能在不改变 E→E 或膜电导的前提下，把区域 q 的耗竭限制到近期 inhibitory-use 足够密集的事件。A–B 展示注册中心点的 U、H、门函数与 locked 六事件 q 顺序；C 同时显示 3×3 cell verdict、mapped tau_D 与逐格失败原因；D 检查全部 q_hold=.8425 周期 sensor 的安全范围；E 对比 isolated、dense 与 sparse equal-dose probes；F 给出 theta sensitivity、schedule、recovery 与 root-resolution 合同。

当前结果为 `THRESHOLDED_INHIBITORY_ELIGIBILITY_SCALAR_CLEAN_NO_GO_REGISTERED_ROBUSTNESS_GATES`：8/9 cells 有唯一单调 root，注册域内 no-root 是 resolved failed cell；有 root 的 cells 中 theta sensitivity 通过 0/8，因此没有 discovery-safe cell。

即使全部门通过，本节点仍只是 pilot-informed scalar mechanism discovery，只允许中心点及两个 theta 邻点进入短 coupled arm；不允许写成 autonomous lifecycle、空间 containment 或 seizure mechanism proof。

**关注点**：承重结果是 edge-adjacent safe cells、held-out schedule 混合结果、theta sensitivity 与 U=0 recovery 同时成立，而不是某个单独参数点成功。
