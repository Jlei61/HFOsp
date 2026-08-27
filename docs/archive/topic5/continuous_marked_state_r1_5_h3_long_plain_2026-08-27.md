# R1.5 / H3-long 阶段报告：白话版

## 一句话

R1.5 在 3 位真正新增患者中有 **1/3** 位找到跨窗口且正确时刻特异的候选状态；3 位旧长记录校准患者中有 **0/3** 位满足同一条件。H3-long 的 **0/16** 个完整对照患者-source-尺度组合达到患者级阳性，**0/10** 个边界组合达到支持标准；本轮没有证据表明 1,000--10,000 次 IED 历史稳定增加下一事件或未来状态预测。

## H1 / H2a：连续背景是否形成有用状态

| 患者 | 身份 | 已更新 seed | persistent 有利 | correct-time 有利 | first subset 有利 | continuation 有利 | 联合稳定 |
|---|---|---:|---:|---:|---:|---:|---:|
| epilepsiae_1096 | independent_extension | 0/5 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 distinct |
| epilepsiae_384 | independent_extension | 5/5 | 2/5 | 5/5 | 0/5 | 2/5 | 2/2 distinct |
| yuquan_zhangkexuan | independent_extension | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 5/5 distinct |
| yuquan_chengshuai | previously_seen_long_record_calibration | 1/5 | 1/1 | 1/1 | 0/1 | 1/1 | 1/1 distinct |
| yuquan_chenziyang | previously_seen_long_record_calibration | 0/5 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 distinct |
| yuquan_zhangjiaqi | previously_seen_long_record_calibration | 0/5 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 distinct |

这里 5 seeds 是优化稳定性检查，不是 5 个独立患者。`0/0` 表示没有 seed 真正更新，不能读成生物学阴性。只有同一 seed 同时满足模型确实更新、persistent 胜 memoryless、正确时刻胜 matched wrong-time，且患者至少有 3 个不同 checkpoint，才允许进入后面的状态持续性分析。

具体来说，张克轩是唯一达到联合稳定标准的新增患者：5/5 seeds 均支持跨窗口持续和正确时刻特异，first subset 也为 5/5，但 continuation 为 0/5。最安全的解释是候选状态帮助预测下一事件最先募集哪些触点，尚未证明它控制整个后续传播路径。E384 的正确时刻为 5/5，但 persistent 只有 2/5，更像时刻相关 observation code；E1096 没有任何 seed 更新，不能作为阴性。

## H3：很长一段 IED 历史是否还有增量

| 患者 | source | N | 支持层 | 边可估 seed | 独立 validation 单元 | full 阳性 | boundary 支持 | H5 | H10 | real-state | real-time-trend |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| epilepsiae_1096 | load | 1000 | full_control | 0/5 | 10 | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| epilepsiae_1096 | participation | 1000 | full_control | 0/5 | 10 | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| epilepsiae_1096 | load | 3000 | boundary_incomplete_control | 0/5 | 7 | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| epilepsiae_1096 | participation | 3000 | boundary_incomplete_control | 0/5 | 7 | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| epilepsiae_384 | load | 1000 | full_control | 5/5 | 2 | 0/5 | 0/5 | 0/5 | 0/5 | +0.0453 | +0.1251 |
| epilepsiae_384 | participation | 1000 | full_control | 5/5 | 2 | 0/5 | 0/5 | 0/5 | 0/5 | +0.0527 | +0.1251 |
| yuquan_chengshuai | load | 1000 | full_control | 0/5 | 3 | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_chengshuai | participation | 1000 | full_control | 1/5 | 3 | 0/5 | 0/5 | 0/5 | 0/5 | -0.0022 | -0.0022 |
| yuquan_chengshuai | load | 3000 | full_control | 0/5 | 1 | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_chengshuai | participation | 3000 | full_control | 0/5 | 1 | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_chengshuai | load | 10000 | boundary_incomplete_control | 0/5 | NA | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_chengshuai | participation | 10000 | boundary_incomplete_control | 0/5 | NA | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_chenziyang | load | 1000 | full_control | 2/5 | 1 | 0/5 | 0/5 | 0/5 | 0/5 | +0.0005 | +0.0036 |
| yuquan_chenziyang | participation | 1000 | full_control | 5/5 | 1 | 0/5 | 0/5 | 0/5 | 0/5 | +0.0001 | +0.0006 |
| yuquan_chenziyang | load | 3000 | boundary_incomplete_control | 0/5 | NA | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_chenziyang | participation | 3000 | boundary_incomplete_control | 0/5 | NA | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_zhangjiaqi | load | 1000 | full_control | 1/5 | 5 | 0/5 | 0/5 | 0/5 | 0/5 | -0.0007 | -0.0007 |
| yuquan_zhangjiaqi | participation | 1000 | full_control | 0/5 | 5 | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_zhangjiaqi | load | 3000 | full_control | 0/5 | 2 | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_zhangjiaqi | participation | 3000 | full_control | 0/5 | 2 | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_zhangjiaqi | load | 10000 | boundary_incomplete_control | 1/5 | 1 | 0/5 | 0/5 | 0/5 | 0/5 | -0.0128 | +0.0028 |
| yuquan_zhangjiaqi | participation | 10000 | boundary_incomplete_control | 1/5 | 1 | 0/5 | 0/5 | 0/5 | 0/5 | +0.0031 | +0.0114 |
| yuquan_zhangkexuan | load | 1000 | full_control | 3/5 | 1 | 0/5 | 0/5 | 0/5 | 0/5 | -0.0006 | -0.0069 |
| yuquan_zhangkexuan | participation | 1000 | full_control | 5/5 | 1 | 0/5 | 0/5 | 0/5 | 0/5 | +0.0361 | +0.0243 |
| yuquan_zhangkexuan | load | 3000 | boundary_incomplete_control | 0/5 | NA | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_zhangkexuan | participation | 3000 | boundary_incomplete_control | 0/5 | NA | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |

读表时负数表示真实累积历史比对照预测误差更低。只有真实历史同时胜过 state-matched、当前单次事件、时间趋势、拟合截距和可用的前一整块，并且最终共同支持至少有 3 个互不重叠 validation 单元，才记作患者级 development 阳性。其余普通阴性、epoch-0、匹配失败和支持不足分别保留。

130 个预定 seed-cells 中，只有 **29** 个学到了可估的非零边；**60** 个是零梯度，**11** 个选择了零边，**30** 个在最终共同支持上不可估。唯一稳定 T1 患者张克轩在 N=1,000 只有 1 个独立 validation 单元，N=3,000 又没有完整 TRAIN 支持。因此 H3 的 0 个阳性既不是仪器全面看清后的生物学否定，也没有留下可升级的稳定正信号。

## 能说与不能说

- 能说：development 数据中有 1 位新增患者形成了稳定、正确时刻特异的候选状态，主要增量落在下一事件的 first subset，而不是 continuation。
- 能说：本轮 exact-N H3 没有患者级完整对照支持；个别 seed 的有利点估计没有跨 seeds、完整对照和独立单元共同成立。
- H5/H10 没有任何组合通过。即使未来通过，也只能称 exposure-conditioned latent correction 在使用真实未来事件历史时仍有预测持续性。
- 不能说：IED 已因果塑造了真实慢状态、生成器或癫痫网络。当前并非逐事件递推的自主生成模型。
- 也不能说：长尺度 IED 对状态没有作用。唯一稳定 T1 患者的最终独立支持不足以承担这种否定。
- formal/sealed 分区、seizure probe 与 paper-ready 图均未打开或修改。
