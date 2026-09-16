# Group-Event State v0.3.5：共享长状态执行合同

## 1. 核心问题

本阶段不再把“某个长窗的事件数更容易预测”当成状态已经建立。它只把既往群体间期事件压成两条候选轨迹：

- `S_N(t)`：预测未来一段真实时间内的事件负荷；
- `S_G(t)`：预测未来事件如何在患者自己的 contact 网络中表达。

二者都只由已经发生的间期群体事件和真实的无事件时段更新。它们首先只能叫 **marked-history predictive state**；除非跨任务预测发作成立，不能叫癫痫易感状态，更不能叫独立的生理潜状态。

## 2. 为什么不是四个独立长窗模型

2、6、8 小时使用同一组按最大 horizon 划出的 FIT、INNER 和 SELECTION 边界。每个 `S_N` 或 `S_G` 只有一个 producer checkpoint 和一条因果轨迹。各 horizon 可以单独拟合低容量读出，但不能各训一套 state producer 后再拼成“同一状态跨尺度存在”。

已完成的 per-horizon rate 模型只登记为 `H_N dynamic baseline`，用于判断难度和可估性，不承担 H1。

## 3. S_G 预测什么

`S_G` 不假设每次 IED 的局部传播规则会整体翻转。它同时预测：

1. 冻结 contact-sequence decoder 中，已知首个 tied group 后的 continue/STOP、后续招募和 contact identity；
2. 未来块中各 contact community 的占用比例；
3. community 之间的转移/耦合比例；
4. 患者内 event repertoire 的混合比例；
5. 连续 repertoire embedding 的均值。

community、PCA 和 repertoire 字典只用 CALIBRATION/FIT 建立并冻结。块内 grammar 只在未来确实有事件时评分，避免单纯 event rate 上升伪装成传播语法变化。

## 4. 嵌套时间合同

- base contact decoder 沿用既有患者内网络，但所有会随时段改变的 static/dynamic adapter 必须在共享长窗的 FIT/INNER 上重训；旧短窗 adapter 不得直接用于正式比较。
- producer 只用 FIT 训练、只用 INNER 选 checkpoint；SELECTION 不参与选择。
- horizon-specific evaluator 在 producer 完全冻结后拟合；其梯度不能回到 producer。
- 长窗可以跨越未观测区间，但未观测秒数不提供“没有事件”的证据；count likelihood 使用有效 exposure offset。
- 主层未来块不得跨过已知发作。跨发作版本只作敏感性。
- development、sealed/test 均保持关闭。

## 5. 承重比较

每个 horizon、每个 endpoint 都在完全相同的 SELECTION anchors 上比较：

1. FIT 期常数；
2. 可解释动态基线 `q(t)`；
3. `q(t) + correct-time state`；
4. `q(t) + distant block-shift state`。

只有状态超过动态基线，且 correct-time 优于保留真实状态分布和自相关的 distant shift，才支持时刻特异的持续预测信息。常数或动态慢水平本身是真实信号，但只属于证据阶梯的较低层。

另做交叉读取：`S_N -> grammar`、`S_G -> burden`、`S_N+S_G -> 两类目标`。这用于判断两个名称是否真对应不同信息，而不是预设它们已经解耦。

## 6. H2a 与 H2b

- H2a：冻结 contact decoder 已看到目标事件的首个 tied group；比较正确状态与错时状态对 later recruitment、STOP、contact identity 的预测。它是 conditional continuation，不是无条件 next-event accuracy。
- H2b：冻结 producer 后，单独拟合发作距离 survival head 和早期 ictal spatial-field head；不允许 seizure loss 回传到 state。缺少 held-out seizure 的患者记为不可估，不记为阴性。

## 7. 首轮执行范围

可估性与已有 decoder 共同支持的首轮患者为 `E253`、`E1096`、`E1125`；每位 3 seed，分别训练 `S_N` 和 `S_G`。这是 development pilot，不是队列确认。E1077、E958 等扩展患者需先完成同一共享分区下的 decoder adapter，不能为了增加人数复用不相容 checkpoint。

## 8. 自动执行与验收

执行根目录：`/data/hfosp_group_event_state_v0_3_5_shared/`。

顺序为：共享 rate prerequisite → 共享分区 contact adapter → `S_N/S_G` producer → 冻结 horizon evaluator → H2a conditional continuation → 冻结 H2b → 交叉读取 → 白话/技术报告。

工程验收要求：原子进度、可恢复跳过已完成单元、两 GPU 并行、OOM 只允许降低 chunk size 后重试一次、所有来源 checkpoint/trajectory 写入 registry。科学验收不设“必须显著”的 gate；阴性、异质或不可估都照实进入报告。

训练充分性按单元判断：若一个 producer 的最佳 checkpoint 落在最后一个已训练 epoch，则该单元不能直接进入科学裁决，必须用同一数据、同一 seed 和同一配方把上限从 20 延长到 60 epoch；只有扩展训练完成后才允许重做下游读出。若最佳点在第一次完整 FIT→INNER 评估，且随后至少经历完整 patience 仍不改善，则记为“首轮即最优”，不把它误写成 epoch-0 未训练，也不靠无休止延长预算制造结果。

## 9. 运行中冻结评价修正

首个 `S_N` producer 落盘后，冻结 ridge 读出暴露出两个会放大长窗计数差异的问题，均在读取其余结果前修正：

- ridge 惩罚改为定义在平均损失尺度，因此把同一批 anchor 重复若干遍不会改变解；
- 事件负荷的评分改为与训练合同一致的负二项 likelihood，dispersion 只在 q-only INNER 上选择，随后对 constant、q-only、q+state 和 shifted-state 四臂共同冻结。

旧的绝对 ridge 与 Poisson 临时读数已覆盖，不进入报告。回归测试包含逐行复制不变性和负二项的 Poisson 极限。

## 10. 独立长窗口径

固定网格的锚点高度重叠，锚点行数不是样本量。每个 horizon 的读出卡必须同时报告 FIT、INNER、SELECTION 中用贪心法得到的互不重叠墙钟窗数；错时对照也必须报告其独立窗支持。少于 3 个独立 SELECTION 窗的结果只能作为方向性探索，0 个错时独立窗时不能裁定时刻特异性。该计数只修正证据强度，不改变训练损失或逐锚点评分。
