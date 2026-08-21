### dual_core_vs_free_field_explanatory_power.png
这张图在完全相同的 Node 预算、12 张网络、背景噪声和事件读出下，直接比较手放双 core 与连续自由场。上排给出场本身及两个患者方向的 rank profile；下排以网络为独立单位展示 shaft-aware 分布误差和配对差。

**关注点**：先看连续场相对手放双 core 的配对区间是否跨 0，再看改善来自 recruitment、precedence、profile 还是 event cloud，不能只看一个总分。

### dual_core_vs_free_field_kmeans.png
这张图沿用 Fig.4 的 KMeans 语法：每个场只使用 returned、双杆、patient-support 内事件，无监督分成两簇，再与冻结的患者方向比较。虚线是患者 held-out prototype，实线是模型簇 profile。

**关注点**：K=2 是否稳定存在与其是否贴近患者是两件事；同时比较簇纯度、balanced alignment 和逐触点 profile。
