# v0.3.3 S_G 合成恢复实验：白话收口

## 一句话

这轮没有把 grammar state encoder 救回来。真状态直接交给读出头或固定时间库时可以被找回；一旦把可见的合成 mark 和冻结的 scaffold 输入一起交给 encoder，调学习率、训练时间和网络宽度仍然在 3/3 个独立种子上失败。因此当前问题应定在 **encoder 与 grammar 目标在杂讯输入下的错配**，不能再用“可能只是没训够”解释，也不应继续盲目扫参。

## 先修掉的两个承重问题

1. 旧 Level-2 state 臂不是严格接在已经校准好的 H-only grammar 上，而是重新从原始参与率拟合截距。state 臂表现差时，无法区分“状态没学到”和“基线自己没校准好”。现在 T1 将 H-only 截距冻结，只允许 state 提供增量。
2. 把训练预算从 120 增到 600 后，旧 H-only helper 直接取最后一次迭代，没有用 inner-validation 选步数，反而会让 H 基线变差。现在 H-only 只用 TRAIN 拟合，用 inner-validation 选 checkpoint，再冻结。

第一次资源 sentinel 成功完成，但在发现第 2 个问题后已移到 `superseded_resource_sentinel/`，不参与科学判断。

## 实验结果

- full-input 小网格包括：较低学习率、余弦退火、延长到最多 600 步、冻结 H 截距、以及把 hidden/write 从 16/2 扩到 64/4。没有一个 tuning 配方得到正的 held-out gain。
- 按 inner-validation 事先锁定的 full-input 配方，在 3 个独立 D3 种子上的 gain 全为负：`-0.432, -1.143, -0.229`，`0/3` 恢复。
- marks-only 在 tuning seed 上曾得到 `+0.128`，95% CI `[+0.055, +0.207]`；但独立 3 seeds 只有 `1/3` 恢复，另外两次反向，不能收口成成功。
- D0 假阳性检查：full-input `0/3`，marks-only `0/3`。仪器没有稳定编造阳性。

## 允许结论

> 当前 D3 的 grammar 真值能够通过 Level 0/1，但不能稳定穿过 Level 2 encoder。有限的优化器和容量调整已经排除“只差多训一点”这一简单解释；marks-only 也未通过 3-seed 复核。下一版应重新检查 encoder 的目标对齐和 nuisance 处理，而不是继续扩大同一网格。

这只是合成诊断，不包含人体 target、发作、sealed partition、H2a/H2b/H3，也不支持任何生物学阴性。

机器报告：`/data/hfosp_group_event_state_v0_3_3/training_lab/sg_synthetic_recovery/reports/final_report.json`

