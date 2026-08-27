# R1.5 / H3-long 退役说明：原报告保留，但不再作为当前结论

## 一句话

R1.5 / H3-long 完成了当时登记的运行，但后续复审证明其承重比较同时受到选择数据复用、零梯度、
免费截距、名义尺度失真、独立窗口不足和部分拟合发散影响；因此原白话版和技术版只保留为过程记录，
当前科学结论一律以 R1.6 更正和 R1.7A / T2-R2.0 合同为准。

## 1. 哪些旧结果被撤回

### 张克轩不是已证实的 persistent state

R1.5 曾把张克轩列为唯一稳定、正确时刻特异的 persistent-state 患者。R1.6 用选择安全的 TRAIN
三段设计和冻结公共优化配置重跑后，张克轩正确时刻仍 5/5 seeds 优于匹配错误时刻，中位 joint
NLL/event 差为 `−0.01278`；但 persistent 5/5 输给 memoryless，中位差为 `+0.02507`。
因此保留的是 time-specific observation-conditioned code，撤回跨窗口持续状态结论。

### epoch 0 和 no-update 不是人体阴性

R1.5 的 epoch 0 来自在完整 TRAIN 上已经 refit 的旧 checkpoint，随后却使用 TRAIN 尾部做新的
target-alignment 选择。起点已经见过选择数据，更新臂与 epoch 0 的比较不公平。旧 `0/0`、epoch 0
和 no-update 只能说明当时的选择与训练设计没有找到可接受更新，不能说明患者没有状态。

### H3 的零边不是 IED 无作用

旧张家齐 T1 的 state-to-timing、state-to-contact 和 state-to-size 读出停在零；H3 edge 在零点的
梯度也精确为零。这个结构下边不可能通过调学习率学起来。旧 `ZERO_GRADIENT` 全部改记为
`NOT_ESTIMABLE`，不进入有利/不利患者分母。

## 2. 为什么 N=1,000–10,000 和六小时 boxcar 退役

旧长尺度路线不能继续承担“更长 IED 历史是否塑造状态”的结论，原因不是结果难看，而是测量对象不稳：

1. real cumulative 相对 no-edge 的主要差异可由 exposure 自带的常数平移解释；旧 no-edge 没有匹配
   这项自由度。
2. 多个 real 与 placebo checkpoint 选择 epoch 0，edge 停在零初始化；这类恒零比较是结构零，
   正确分母是 `0/0`，不是 `0/2` 或 `0/5`。
3. 名义一万次和约六小时在默认约 54 分钟生成器下，90% 权重实际来自最近约 1.6 小时；两档不是两台
   独立长尺度仪器。
4. 逐事件滑动窗口大量重叠，窗口行数远大于独立信息量。部分患者 validation 只有约 1–2 个有效独立
   窗口，不能靠数千行窗口制造效力。
5. 六小时 boxcar 的 ridge 正则随 Gram 尺度失配，归档超长臂中出现远高于拟合截距对照的外推发散；
   这类结果是不可估计，不是普通阴性。
6. 延迟对照与真实暴露在大 N 下高度重叠，零差不能解释为暴露时序不重要。

因此 N=1,000/3,000/10,000、六小时 boxcar 以及旧 `real−no_edge` 主比较全部退出当前实验主线。
旧文件不删除，只用于复现“为什么退役”。

## 3. R1.6 后仍可保留的证据

- synthetic 正真值、反向真值和短段过拟合说明模型及优化器可以学习；这证明仪器可训练，不证明人体
  H1–H3 成立。
- `epilepsiae_384` 在冻结公共配置下有 3/5 stable checkpoints，两个未参与调参的确认 seeds 均稳定；
  患者内中位 persistent−memoryless 为 `−0.00168`，correct−wrong 为 `−0.00890`。这是目前唯一
  optimization-robust 的单患者 development 支持，不是队列阳性率。
- 程帅和陈子阳各只有 1/5 stable，独立确认 0/2，只能称 optimizer-sensitive signal。
- E1096 与张家齐均能降低训练损失并通过短段过拟合，但未见后段没有稳定增量；安全定级为当前模型下
  泛化失败或不可识别。
- E384 的最小 H3 六个单元均未通过完整控制，且每个 seed 只有两个独立 validation 单元；结论仍是
  H3 未决，而不是 IED 不塑造状态。

## 4. 当前替代方案

R1.7A 使用完全未参与旧架构、优化器、预算或阈值选择的 10 位 development 患者，冻结同一个 R1.6
配置和五 seeds。development validation 按真实记录时长预先拆为 D_state 前 60% 和 D_mechanism 后
40%：H1/H2a 只读 D_state，T2 只读 D_mechanism。

T2 回到事前最有支持的 N=100，只保留 no-edge、real cumulative、state-matched non-overlap 和
current-event-only 四臂，不给 exposure 自由截距。只有 D_state 同时支持 persistent、correct-time
并且 T1 读出健康的患者才运行；D_mechanism 少于五个非重叠 100-event blocks 时只作个案。

## 5. 合作者讨论时的安全版本

可以说：

> R1.5 提示训练与状态识别可能存在信号，但其关键比较后来被证明不够公平，长尺度 H3 仪器也没有真正
> 测到登记的数千次至数小时效应。R1.6 修复后，唯一稳健的人体信号来自 E384 单例；H3 仍然未决。
> 因此我们正在未见患者上前瞻复现状态，再用独立后段数据检验 N=100 的 IED→state 增量。

不能说：

- R1.5 已证明张克轩存在 persistent state；
- 长尺度 H3 为 0，因此数千次 IED 不会改变状态；
- E384 单例等于队列支持；
- seeds 是独立患者重复；
- formal/sealed 分区已经验证这些结论。

## 6. 权威顺序

发生冲突时按以下顺序读取：

1. `continuous_marked_state_optimizer_identifiability_r1_6_correction_boundary_2026-08-27.md`；
2. `continuous_marked_state_r1_5_retirement_2026-08-27.md`（本文）；
3. `continuous_marked_state_r1_7a_r2_0_contract_2026-08-27.md`；
4. R1.5 原白话版、技术版和机器审计，仅作历史 provenance。

formal/sealed 分区、seizure probe 和 paper-ready figures 均未打开或修改。
