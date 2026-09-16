# Group-Event State v0.3.6：全信用分配共享状态 Spec 与 Plan

> **状态：`SUPERSEDED_BEFORE_HUMAN_EXECUTION`。** 该稿正确提出了完整信用分配，但仍沿用旧门控递归；整链探针证明旧递归即使完全不 `detach`，99% 梯度也只到 1.65 小时。因此 v0.3.6 不再执行，已由 v0.3.7 的双流连续时间 observer + 独立 H3 生成模型取代。其 smoke 只作工程诊断，不进入人体科学汇总。

## 1. 为什么暂停 v0.3.5

v0.3.5 的状态数值可以跨很多小时传递，但训练图每 30 分钟 `detach`。因此模型能把旧信息带到后面，却不能从后面的预测误差中学会几小时前应该保留什么。把截断从 30 分钟拉到 2 小时后，改善主要落在 2/6/8 小时的 block grammar，而短程 contact field 不变，说明这是长程信用分配问题，不是普遍噪声。

2 小时只是一条诊断臂，不是新合同。v0.3.5 的 13/18 个 producer 与已有 evaluator 全部降级为旧仪器诊断，不承担 H1/H2 结论。

## 2. 核心科学问题

同一个只读取既往群体间期事件的状态，能否同时保留：

1. 未来真实时间块内的事件负荷；
2. 稳定局部 grammar 上的 community occupancy、跨 community coupling 与 repertoire mixture；
3. 下一场事件在相同开头之后的 recruitment/STOP/contact path；
4. 冻结后与距发作时间及发作早期空间场有关的信息。

状态先称 `marked-history predictive state`。只有跨任务发作读出成立，才接近 susceptibility state；预测增量本身不等于生理因果状态。

## 3. 状态输入与读出

每次群体事件仍是一个真实输入单位。event encoder 读取完整 participation、tied group、精确相对延迟、bipolar/CAR 波形摘要与多频带能量/峰时；真实 `dt` 决定事件间演化。动态 rate/history 特征 `q(t)` 只作为嵌套基线和读出协变量，不写进 recurrent state。

`S_N` 与 `S_G` 各有一个跨 horizon 共享 producer。2/6/8 小时只是一组首轮输出任务，各自有冻结的小读出头；不能各训一条状态再拼成“共享状态”。可估时 12 小时只作探索，不进入首轮 Core。

时间尺度库与输出 horizon 解耦。第一版使用从分钟到数日的固定宽基函数库（5 min、30 min、2 h、8 h、24 h、72 h），由网络学习如何混合；这些是记忆基函数，不解释为数据识别出的生理时间常数。后续用整体平移一档的时间库做敏感性，而不为每位患者挑最佳常数。

## 4. 全信用分配合同

### 4.1 训练样本

一个训练样本是两个真实因果边界之间的完整事件链。真实边界包括发作以及显著长于人工切口的断录；十分钟以内人工切口合并。缺失时间不提供“没有事件”的证据，但真实 `dt` 仍用于状态演化。发作后不把发作前状态静默接到发作后。

### 4.2 梯度

样本内部禁止 `detach`。训练 loss 对该样本中所有更早事件保持可微路径。512/1024 事件 microchunk 仅用于 `torch.utils.checkpoint` 的前向重算和显存控制，不改变状态、不重置状态、不截断梯度，也不是科学时间尺度。

只有在完整样本 backward 后才允许 optimizer step。不同真实事件链可作为独立 batch/梯度累积单元；同一条链不能为了吞吐随机切开重置。

### 4.3 梯度到达审计

每个训练单元必须输出“目标 loss 对历史 event innovation 的梯度范数—真实时间滞后”曲线，至少分 0–0.5、0.5–2、2–8、8–24、24 h 以上五档。合同要求：

- 早期事件与最终 loss 在 autograd 图上连通；
- 0.5 小时之后不能因 `detach` 精确为零；
- 数值衰减可以存在，但若某一整档全部触底，该时间尺度记为不可训练，而不是生物学阴性；
- 同时运行显式 detach 反证，证明审计能抓到结构零。

## 5. 训练目标

- `S_N`：对 observed-support future count 使用带 exposure offset 的负二项 likelihood；silence 通过有效观测时间进入目标。
- `S_G`：局部 frozen contact-sequence likelihood，加未来块的 community occupancy、cross-community coupling、repertoire mixture 与连续 repertoire embedding。
- 多 horizon 在 producer 中显式等权；缺少该 horizon 训练支持时该项不参与，并在卡片中报告，不能让样本多的 horizon 隐式支配。
- contact decoder 骨架冻结，状态通过低秩调制进入每一个生成步骤，而不只进入初始触点。

## 6. 优化与训练充分性

全信用分配改变了每个 optimizer step 的样本量，不能照搬 v0.3.5 学习率。先在 E253 的 FIT 切片做三档搜索：encoder LR `3e-5 / 3e-4 / 1e-3`，state LR `3e-4 / 1e-3`；其余设置相同。比较完整训练目标、各组成项、梯度 lag profile、参数相对更新量和 INNER 外推，不按 FIT 降幅单独选配方。

选定配方后，producer 若最佳点仍贴预算末端，统一延长；首轮即最优必须有完整 patience 和非零梯度审计。模型容量只在全信用分配与优化器均合格后做一小一大两档，不先用更大网络掩盖训练合同问题。

## 7. 冻结评价

每个 endpoint 同时比较 FIT 常数、动态 `q(t)`、`q+correct state`、`q+block-shift state`。主结论要求状态超过动态基线且正确时刻优于错时。所有长窗同时报告重叠锚点数与互不重叠墙钟窗数；少于 3 个独立 SELECTION 窗只作探索。

H2a 同时报状态相对 rate、FIT 常数与 block shift 的增量；H2b 只在 producer 冻结后拟合统一 survival head 和 early ictal field head，发作梯度不得回传。缺发作是不可估，不是阴性。

## 8. 执行顺序

1. 实现 checkpointed full-credit event chain 和梯度 lag 审计；
2. 用合成长依赖与显式 detach 反证证明图连通，但不把合成通过当人体可训练；
3. E253 同切片比较 0.5 h detach、2 h detach、full-credit 三臂；
4. 完成学习率与小/大容量训练性诊断；
5. 只有 full-credit 在 FIT 与 INNER 均可训练后，才在独立根目录启动 E253、E1096、E1125 × 3 seed × `S_N/S_G`；
6. producer 全部冻结后再运行 H1/H2a/H2b 与交叉读取；
7. 输出机器审计、白话报告和技术报告。

新结果根：`/data/hfosp_group_event_state_v0_3_6_full_credit/`。不得写回 v0.3.5 旧根。development、sealed/test 均保持关闭。

## 9. 允许与禁止的结论

- 0.5 h→2 h 训练改善：支持旧仪器信用分配不足；不支持 2 h 生理状态。
- full-credit 能拟合人体 FIT：只说明网络可训练；INNER/SELECTION 与对照决定预测信息。
- 只胜常数：阶段或慢背景水平。
- 胜动态基线但不胜错时：额外预测表示，但时刻特异性未建立。
- 两者都胜且跨多个 horizon：共享的时刻特异 predictive state 候选。
- S_G 在 conditional grammar 上成立：状态调制事件表达；若只在 count 上成立，不得外推为空间网络状态。
- 冻结 H2b 成立：间期学习状态具有跨任务发作相关信息；仍不是因果机制。
