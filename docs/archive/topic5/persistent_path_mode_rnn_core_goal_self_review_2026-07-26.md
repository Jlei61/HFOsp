# 审阅结论

> **历史状态**：这是正式 510-run 完成前的执行审阅。最终结果见
> `persistent_path_mode_rnn_formal_result_2026-07-26.md`。

## 1. 一句话判断

v0.9 三人试跑在工程上完成了，但它把“不稳定的完整传播顺序”设成主门，不能回答论文真正
需要的触点级 rank 分布问题；v1.0 已纠正主终点，并启动 34 人 × 3 seeds 的正式 LOSO
仿真，目前尚不能宣称模型已复现间期传播或支持间期—发作期桥接。

## 2. 完成程度

> **完成度：70/100**

已经完成：

- 使用 masked interictal contact-rank events，未使用 phantom rank。
- chronological train80 / heldout20 与 LOSO 被试隔离。
- path prior 只从 train80 构造；A/B、IEI 和发作期数据没有进入训练。
- 模型能自由生成事件，并可汇总每个触点的参与概率、完整 rank 分布及
  early/middle/late 概率。
- 固定 `K=2`，加入 no-history、merged-path、weight-shuffle、mode-shuffle 及结构损伤。
- 34 人 × 3 seeds × 5 条件的 510 个正式任务已经启动，并带有精确覆盖、日志、checkpoint、
  显存限制和 target seal。

尚未完成：

- 510 个任务尚未全部结束，因此主门和结构必要性还没有 cohort-level 结论。
- 发作期静态能量场 readout 尚未开始；这一步必须等待纯间期主门。
- paper-ready 主图必须使用正式结果重画，旧 v0.9 图只能保留为 pilot 诊断。
- 当前实现应称为 structured graph RNN；其路径混合是低维的，但循环权重没有受到严格的
  矩阵低秩分解约束，不能直接写成 algebraic low-rank RNN。

## 3. P0 / P1 关键问题

### P0：旧主终点与核心科学目标不一致

**问题是什么**：v0.9 用 pairwise precedence 和 whole-path distance 决定是否推进。

**为什么严重**：已有数据并不支持逐次完整顺序稳定；用它否决模型会把“每个触点稳定的
rank 分布”与“每次事件的精确重放”混为一谈。

**怎么改**：v1.0 已把 participation MAE 和 per-contact rank-distribution
Wasserstein distance 设为两个共同主终点；precedence 和 whole-path 只保留为次要诊断。

### P0：三人 pilot 不能代表 34 人队列

**问题是什么**：旧结论来自 3 位患者 × 3 seeds。

**为什么严重**：患者内事件很多并不能替代患者数；论文级推断必须以患者为统计单位。

**怎么改**：正式任务固定 34 个 heldout folds × 3 seeds，先在患者内取 seed 中位数，
再做 patient-level directional Wilcoxon 和 BH-FDR。

### P1：模型的“结构性”仍需用损伤实验验证

**问题是什么**：即使 intact 生成的边缘分布正确，也可能只来自静态触点偏置，而不是
路径图或多路径结构。

**为什么严重**：没有结构损伤，预测性能不能支持内部动力学解释。

**怎么改**：正式门要求 graph lesion 或 mode-collapse lesion 至少有一个同时破坏两个
主终点；方向和 inhibition 损伤用于解释贡献，不单独决定主门。

### P1：全 34 人中包含三位开发病例

**问题是什么**：`epilepsiae_1073`、`epilepsiae_1146` 和
`yuquan_chenziyang` 曾用于 v0.9 pilot，模型结构的修订发生在查看这些病例之后。

**为什么严重**：即使 v1.0 的 `K=2` 和新主门在正式运行前冻结，包含这三人的 34 人结果
也不能称为完全独立外部验证。

**怎么改**：保留用户要求的 34 人主分析，同时固定报告排除三位开发病例的 31 人
sensitivity；两者不一致时，以“开发病例依赖”作为主要限制，不隐藏差异。

### P1：发作期任务必须保持静态、冻结和后验

**问题是什么**：若用逐秒发作序列反向训练 RNN，会把已有的静态早期能量重用结果改写成
未被数据支持的发作传播重放。

**为什么严重**：这会偏离论文主线，并造成 target leakage。

**怎么改**：只有纯间期正式门通过后，才读取 clinical onset `[0,10] s`、`1–150 Hz`
baseline-robust-z 静态能量场；RNN、`K` 和 path prior 全部冻结。

## 4. 科学性问题

当前 v1.0 的逻辑链与论文主线基本一致：

1. 用间期 rank 事件做自监督 next-set/STOP 学习；
2. 不把 A/B 当标签，而从大量自由生成事件汇总患者内触点级分布；
3. 检验患者特异路径图和多路径结构是否是复现这些分布的必要条件；
4. 只有间期生成模型成立后，才测试其冻结 readout 与发作早期静态能量场是否一致。

这个设计最多支持“受结构约束的间期生成动力学可读出跨状态共享场”。预测更准本身不等于
机制成立；若结构损伤不破坏结果，安全结论只能是模型恢复了边缘分布，不能说路径动力学被
识别。即使发作期 readout 阳性，也只能支持 shared scaffold/readout，不能证明逐触点发作
重放或因果机制。当前 train80 也不是逐次目标发作前的 prefix，因此这不是 prospective
seizure predictor，不能给出提前量或报警性能。

## 5. 工程性问题

- 正式 runner 为每个任务保存 `run_state.json`、`summary.json`、训练 log、checkpoint、
  heldout metrics 和自由生成事件。
- 每个 fold 的 shared training 完整覆盖其余 33 人的 train80 两轮；heldout local offset
  完整覆盖本人的 train80 四轮，避免随机 update 数导致大患者或小患者覆盖不透明。
- 每进程显存上限为 12%，最多并行 8 个进程；启动前要求至少 64 GiB 可用内存。
- analyzer 会拒绝任何缺 run、缺 checkpoint、覆盖不足或读取过发作 target 的结果。
- 当前风险是 launcher 异常退出后，个别 `RUNNING` 状态不会自行改成 `FAILED`；monitor
  会暴露这种不一致，恢复时 runner 会先归档不完整目录再重跑。

## 6. 最小修改路线

1. 完成并核对全部 510 个正式任务，任何失败任务按相同 config 原位恢复。
2. 运行 patient-first 主统计和结构损伤门，输出触点级 observed-versus-generated 分布。
3. 若纯间期主门通过，运行冻结的发作早期静态场 readout；若失败，到此收口为 34 人
   bounded-negative，不用发作结果挽救模型。
4. 只用正式结果生成 paper-ready Figure 6，并把 v0.9 图明确标成 pilot/sensitivity。

## 7. 下一步的建议

核心目标不是证明 RNN “有用”，而是判断一个患者特异、带传播方向和抑制状态的结构化
循环系统，能否从间期 rank 事件中生成真实的触点级分布，并在不重新训练的情况下读出发作
早期静态能量场。当前不应改超参数或再筛 `K`；先让冻结的 v1.0 合同跑完，再按预注册门
决定是否进入发作期。
