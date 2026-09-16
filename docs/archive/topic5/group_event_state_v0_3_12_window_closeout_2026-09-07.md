# v0.3.12 可读丰富事件状态：自主窗口收口

状态：**开发工作包通过工程验收；原始核心科学闭环未关闭。** 67/67 个主计划任务完成；合成扩展 56 个完成、2 个数值失败、2 个后继因前置失败而不可运行。失败 realization 保留在分母。

机器交付位于 `/data/hfosp_group_event_state_observable_state_validation_v0312/final_reports/`；正式科学卡源码摘要为 `bdf901caf0ff7dbddbd028095cfbd6f7365b6bc148b6699b4e6921b303264137`。窗口后数值修复使用新摘要 `cb20be5f3481b3071515d1a1538e1d179e56119493528e9e8a0be60e8bbec2de`，没有用新代码覆盖或重算本轮人体结论。

## 科学验收

E1125 的 S-E 查询跨 66.5 小时、3 个日历日和全部 24 个小时层，信息年龄中位 29 分钟。因此 v0.3.10 的“留出窗短于主导节律”缺陷已实质修复。本轮仍没有建立丰富 IED 历史形成可向前泛化的共同状态：`P_marks-state` 相对 `P_stats-state` 的数量、粗空间和负荷在四个时距均为 0/3 种子胜出；条件形态只在一个种子的 1/5/30 分钟为正，120 分钟为 0/3。相对可学习 marked-history 和参考封底的四类端点也均为 0/3。

回顾性 S-ID 中，EVOLVE 相对 RESET 对数量和粗空间为正，且状态相对 marked-history 有小的点估计增量；按真实小时块重采样的描述区间全部跨零。它只构成局部粗时间状态线索，不能替代真正前推状态，也没有证明丰富事件信息有增量。

冻结细触点读出没有状态增量：S-E 的 48 个相关适配器全部选择第 0 步，S-ID 的 state 和 state+history 也选择第 0 步。发作主窗 11 次病例的未匹配中位状态增量为正，但在可读近期率匹配后的 7 次病例中反向；临床起点 0–10 秒空间读出中 trait、history、state 和 state+history 逐病例相同。S-C 缺完整前瞻人时分母，仍为 `NOT_ESTIMABLE`。

所以当前结论为：

| 科学层 | 判定 |
|---|---|
| 丰富 IED 的前推状态 | `NOT_ESTABLISHED` |
| 回顾性局部粗状态 | `LOCAL_SUPPORT_COARSE_ONLY` |
| 数量、形态和触点的共同状态 | `NOT_ESTABLISHED` |
| 发作前特异状态 | `NOT_ESTABLISHED`，不是临床风险阴性 |
| 非线性／多时间尺度动力学 | `NOT_TESTED` |
| 每次 IED 对生理状态的反馈 | `NOT_TESTED` |

`POWER_CALIBRATED=false`，`CONFIRMATORY=false`。本轮不支持扩大患者、latent 容量或动力学结构矩阵。

## 合成仪器边界

形态联系世界在 1/5/30 分钟为 4/5 realization 正向，120 分钟为 3/5；identity 世界可用的 4 个 realization 在前三个时距均为 4/4 正向。但效应跨 realization 出现极大的双向爆炸，zero 世界也在 5/30/120 分钟出现 4/5 正向。identity realization 905 的两个输入臂均因数值失败不可估。这个仪器能检出部分强联系，却没有校准小形态效应的特异性；因此人体非正结果不能升级为科学阴性。

## 训练与数值修复

34 张正式训练卡中，19 张经历两次降率后的耐心停止，15 张按冻结 INNER 步数重拟合；这里的耐心停止只记为局部优化证据。新合同额外要求两个 temporal INNER 都选择非初始 checkpoint，OUTER 固定步数不再被写成收敛证据。

同一卡上并发两个训练进程是两次异步 CUDA 失败的共同运行条件。最终修复保持 GPU 上的 FP64 Lyapunov 求解，并把下一次调度锁为每张 GPU 一个训练进程：两张 RTX 3090 各完成 500 次前向／反向，最大残差 `1.86e-9`；人为中断后模型、优化器、采样器和调度器逐位相同，权重最大差为 `0.0`。将求解移到 CPU 的候选方案虽然通过并发压力，却破坏逐位恢复合同，已撤回并保留为 superseded 证据。

## 下一窗口边界

下一次长跑只准解决两个当前决定性缺口：

1. 在不读人体 OUTER 的合成世界中，校准 morphology 联系相对 zero 的特异性、效应尺度和数值稳定性；任何失败 realization 留在分母。
2. 只用两个 temporal INNER 修复跨优化种子的训练选择，使强历史、RESET 和丰富输入比较能够稳定复现。

只有 morphology 正对照稳定超过 zero、两个 INNER 均学习且无预算边缘、强历史和 RESET 合格后，才解锁严格 H=0.5/2/8 小时和第二位患者。否则转向测量覆盖与可检出效应上界，不继续增加模型容量。

主要交付：

- [白话报告](/data/hfosp_group_event_state_observable_state_validation_v0312/final_reports/group_event_state_v0312_closeout_plain.md)
- [技术报告](/data/hfosp_group_event_state_observable_state_validation_v0312/final_reports/group_event_state_v0312_closeout_technical.md)
- [科学验收](/data/hfosp_group_event_state_observable_state_validation_v0312/final_reports/acceptance_decision.json)
- [机器收口](/data/hfosp_group_event_state_observable_state_validation_v0312/final_reports/machine_closeout.json)
- [合成复现汇总](/data/hfosp_group_event_state_observable_state_validation_v0312/final_reports/synthetic_replication_summary.json)
- [窗口后数值修复](/data/hfosp_group_event_state_observable_state_validation_v0312/final_reports/postrun_code_patch.json)

