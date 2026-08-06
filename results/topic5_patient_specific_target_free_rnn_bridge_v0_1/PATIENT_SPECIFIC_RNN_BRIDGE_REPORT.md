# 患者特异 target-free RNN 跨状态桥梁 v0.1

## 一句话结论

本轮把跨患者读出彻底拿掉后，患者自己的 RNN 可以从自己的间期 contact-rank events
中学到可泛化到 heldout events 的传播顺序。模型生成的 contact field 随后在完全不重训的
情况下，与同一患者发作早期 1--150 Hz 能量场进行比较；这一跨状态对应及其相对完整静态
participation/rank scaffold 的增量见下表，不能再归因于其他患者或经验 A/B 被注入模型。

## 1. 到底做了什么

- 每名患者独立训练，不共享任何模型权重或其他患者事件。
- 输入是 masked contact-rank events；每场事件内逐 rank set 预测下一 contact/STOP。
- 时间切分为 fit60 / validation20 / untouched test20。
- 主模型为 hidden-32 GRU；相同任务下另跑 linear state；rank-shuffle GRU 只破坏事件内顺序。
- 三个 seed，固定 7 次完整数据覆盖、每次 32 次更新、lr 3e-4。
- checkpoint 冻结后自由生成 5000 个完整事件，汇总 participation、early/late rank mass、
  endpoint mass 和 weighted earliness。
- 最后才读取同一患者共 33 次 clinical onset 后 0--10 s 的 early-ictal target。主频段 1--150 Hz，
  1--45 Hz 为 sensitivity。所有置换都重新执行候选 field 最大化。

## 2. RNN 是否学到了间期传播结构

完整 16 人中，真实顺序 GRU 相对 rank-shuffle 的 heldout NLL 改善中位数为
**0.0603 nats/event**，15/16
患者方向一致；排除 development patient 后为
14/15，
精确配对 `P=0.0001221`。
GRU 自由 rollout 与真实 test20 的 pairwise contact precedence 相关中位数为
**0.775**；rank-shuffle 为
**0.052**。这直接支持模型学到事件内部“谁之后更可能到谁”的
患者特异结构，而不只是 contact 出现频率。

线性状态模型的 precedence 相关中位数为 **0.800**。
因此信息并不只依赖 GRU 门控；GRU 与更简单状态模型都能利用，但二者的 heldout NLL 和
跨状态 readout 需分别报告。

## 3. 模型场是否联系到发作早期

Primary 15 人（E1146 单列 supportive）的 1--150 Hz 结果：

| readout | 患者中位绝对相似度 | 中位 all-contact margin | margin>0 |
|---|---:|---:|---:|
| patient-only GRU | 0.584 | 0.167 | 13/15 |
| rank-shuffle GRU | 0.442 | 0.155 | 13/15 |
| static fit60 participation + rank distribution | 0.607 | 0.167 | 12/15 |

GRU 相对完整 static fit60 participation + rank distribution 的患者级 margin 增量中位数为
**0.025**；相对 rank-shuffle GRU 为
**-0.008**。精确配对检验分别为
`P=0.3054` 和
`P=0.5719`。

杆内打乱是更严格的几何敏感性：GRU 的中位 margin 为
**0.071**，
10 正 /
4 负 /
1 并列，
`P=0.149`。因此当前跨状态对应在论文既有的
全通道打乱口径下成立，但不能声称已经排除了全部电极杆几何贡献。

这里最重要的不是要求 15/15 阳性，而是看两层证据是否同时存在：

1. 真实顺序模型在患者自己的 heldout 间期事件上确实学到传播结构；
2. 同一模型生成的患者 contact field 在发作 target 完全未参与训练时仍有 above-null 对应，
   并评估它相对完整静态 participation/rank scaffold 和 rank-shuffle 的增量。

## 4. 这能支持什么

若 above-null 对应成立，安全结论是：

> 仅用患者自身间期 contact-rank sequences 自监督训练的 recurrent model，恢复了患者特异
> contact recruitment/rank structure；该模型恢复出的患者空间场与同一患者发作早期 broadband
> energy field 在全通道打乱零分布下存在跨状态空间对应。

它比上一轮强在：没有跨患者 readout、没有经验 A/B 输入、没有用 ictal target 训练残差支路。

不能写成：有序 RNN 动力学显著优于静态间期场、RNN 自动恢复了唯一物理 A/B 轴、逐次预测了
发作传播路径、或所有患者共享同一个 RNN 机制。当前 readout 是患者级空间场，不是逐发作动态 replay。

## 5. 工程验收

- target-free units：144；失败 0。
- 其他患者事件进入模型：否。
- empirical A/B 进入模型：否。
- ictal target 进入训练：否。
- checkpoint、training log、heldout metrics、free rollout 和 contact distribution 均逐 unit 保存。
- 运行可由 launcher state 和 `DONE.json` 断点续跑。
