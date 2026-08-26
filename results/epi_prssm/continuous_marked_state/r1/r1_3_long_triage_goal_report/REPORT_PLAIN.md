# 长序列 T1 分诊与最小 H3：白话报告

## 一句话

本轮先检验长记录患者能否在最新 R1.3 目标训练下形成可用状态，再决定是否运行长尺度 H3；最终机器判定为 `H3_NOT_RUN_NO_PATIENT_MET_STATE_AND_INDEPENDENT_SUPPORT`。这不是正式队列结论，普通阴性也不否定更长或不同形式的 IED 累积作用。

## 为什么要做这轮

前一轮长患者实际上只跑过 R1.2，而且 R1.3 命令入口把扩展患者排除在外。现在第一次让固定的三类长患者进入相同的 timing + sequential mark target alignment：韩宇轩代表已有预测记忆，陈子阳代表训练启动但外层失败，程帅代表万次支持但旧模型 epoch 0。

## T1 结果

- yuquan_hanyuxuan: target alignment 3/3；persistent 胜 memoryless 1/3；correct-time 胜 wrong-time 3/3；persistent joint 中位 +0.00023321 （timing +0.0056592，mark -0.0049346）；correct−wrong 中位 -6.4442e-05；端点中位 subset -0.0013015，continuation -0.0037134，STOP +0.012162，size -0.01222；起点/终点 distinct payload 3/3。
- yuquan_chenziyang: target alignment 0/3；persistent 胜 memoryless 0/3；correct-time 胜 wrong-time 1/3；persistent joint 中位 +3.3924e-05 （timing -0.00020469，mark +0.00022413）；correct−wrong 中位 +1.2063e-05；端点中位 subset +5.4698e-05，continuation +0.0010528，STOP +0.0062059，size -0.0071182；起点/终点 distinct payload 3/3。
- yuquan_chengshuai: target alignment 1/3；persistent 胜 memoryless 1/3；correct-time 胜 wrong-time 1/3；persistent joint 中位 +0 （timing +0，mark +0）；correct−wrong 中位 +0；端点中位 subset +0，continuation +0，STOP +0，size +0；起点/终点 distinct payload 3/3。

seed 是优化起点，不是患者数。persistent 胜 memoryless 才说明跨窗口携带额外信息；correct-time 胜 wrong-time 决定能否进一步称为时刻专属状态。

## 独立长窗口与 H3

运行前已按真实事件时刻计算不重叠整窗。自动规则只选择两侧至少各 3 个独立长窗口的最大 N，并只在至少 2/3 T1 起点同时训练启动且 persistent 有利时运行 H3。

- yuquan_chengshuai: 选择 N=1000；完整对比实际需要 2000 events；TRAIN/validation 不重叠窗 8/3；validation 名义/完整支持时长中位 0.61/1.15 h。
- yuquan_chenziyang: 没有任何 N 同时达到 TRAIN/validation 各至少 3 个不重叠完整对比支持窗。
- yuquan_hanyuxuan: 没有任何 N 同时达到 TRAIN/validation 各至少 3 个不重叠完整对比支持窗。

- 没有患者同时满足可用跨窗口状态和 TRAIN/validation 各至少 3 个不重叠长窗口，因此本轮没有为了凑结果而运行新的人体 H3。

若出现合格患者，H3 才会同时比较 load 和 participation/repertoire composition。只有真实 exposure 同时胜过同容量截距与因果延迟对照，才保留为探索性候选；无边对照不再作为暴露证据。本轮这两类人体臂均未调度，不能引用任何新 H3 效应量。

## 当前边界

- 正式检验分区和 seizure probe 均未打开；
- 三位患者只是事前固定的开发分诊，不是队列推断；
- 多 seed 只检查优化稳定性；
- 即使出现探索性候选，也必须在更多患者与更多独立长记录中复现。
