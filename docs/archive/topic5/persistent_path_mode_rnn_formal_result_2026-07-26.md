# 审阅结论

## 1. 一句话判断

510 个正式仿真已经完整跑完，但当前 structured path-mode graph RNN **不符合论文核心
科学目标的阳性要求**：它能改善真实 prefix 条件下的下一步预测，却不能在自由生成时稳定
恢复 heldout 触点参与概率和完整 rank distribution，也没有证据表明路径图或多路径结构
是必要的；因此不能进入发作期桥接。

## 2. 完成程度

> **执行完成度：100/100；科学目标达成度：35/100**

已完成：

- 34 位患者 × 3 seeds × 5 conditions，共 510 个 LOSO runs，0 个训练失败。
- 每个 run 均保存 config、log、checkpoint、heldout metrics、5,000 条自由生成事件和
  completion state。
- 主统计以患者为单位，先取 seed 中位数；另有固定 31 人 development-excluded
  sensitivity。
- 完成 graph、inhibition、forward、reverse、mode-collapse 和 dominant-mode lesions。
- 完成 target-sealed 内部动力学分析与 paper-ready 六面板图。
- 纯间期主门失败后，clinical-onset target 保持未读。

扣分原因：

- 两个自由生成主终点没有同时优于四个冻结对照。
- graph lesion 和 mode-collapse lesion 均未证明结构必要性。
- path-direction posterior 接近均匀，模型内部没有形成可辨识的持续路径状态。
- 因间期门失败，间期—发作早期静态场桥接没有被测试。

## 3. P0 / P1 关键问题

### P0：训练目标与最终科学终点失配

**问题是什么**：模型在真实 prefix 下优化 next-set / STOP likelihood，但论文要回答的是
自由生成大量事件后能否恢复患者级触点分布。

**为什么严重**：held-out next-set NLL 对 no-history、single-path、weight-shuffle 和
mode-shuffle 均有显著改善，但这些改善没有转化为 participation MAE 与
rank-distribution Wasserstein 的联合改善。这是 teacher forcing 下局部预测与自由生成
分布之间的直接断裂。

**怎么改**：若重开，必须新建合同，把 participation 和完整 rank distribution 的
sequence-level distribution matching 直接纳入训练，而不是继续调当前 NLL 模型的
hidden size、学习率或 seed。

### P0：结构必要性未成立

**问题是什么**：graph lesion 和 mode collapse 对两个主终点均无一致损害。

**为什么严重**：即使完整模型画出的节点分布看起来合理，也不能据此声称患者路径图或
多路径动力学被模型识别。

**怎么改**：在新模型中必须让结构先验以可检验方式约束生成分布，并继续保留同一类
lesion。若 lesion 仍不影响输出，应接受静态触点偏置已足以解释结果。

### P1：path-direction latent state 不可辨识

**问题是什么**：normalized posterior entropy 从事件开始的 0.999 仅降到结束时的
0.975；最大 component probability 仅从 0.264 升到 0.332，forward probability 始终
约为 0.50。

**为什么严重**：模型虽然定义了两个路径和两个方向，但事件 prefix 几乎不能决定当前
component；内部“路径状态”主要停留在结构设定，而不是数据驱动的动态状态。

**怎么改**：若重开，应比较连续 mode mixture、有限状态切换或显式可辨识约束；不得把
当前高熵 posterior 解释成 A/B 自然涌现。

### P1：当前结果不能支撑发作预测或跨状态桥接

**问题是什么**：正式间期门失败，clinical-onset `[0,10] s`、`1–150 Hz` 静态能量
target 未被读取。

**为什么严重**：没有发作期结果，就不能写间期模型预测了发作早期能量场；反过来，封存也
意味着本结果不能否定真实数据中已经观察到的 shared scaffold。

**怎么改**：维持封存。只有新的纯间期模型重新通过预注册门后，才允许进行冻结 readout。

## 4. 科学性问题

### 主比较

完整模型只在 participation MAE 相对 weight shuffle 时通过单项门（median benefit
0.00210，28/34，BH \(q=8.6\times10^{-4}\)），但对应 rank-distribution benefit
未通过（0.00013，20/34，\(q=0.305\)）。对 no-history、single-path 和 mode-shuffle，
两个主终点均未联合改善，因此 comparison gate 为 false。

### 结构损伤

graph lesion 的 participation / rank benefit 分别为 0.00095 和 0.00007，均
\(q=0.790\)；mode collapse 分别为 −0.00030 和 0.00001，均未通过。方向、抑制和
dominant-mode lesions 也没有一致损害。31 人敏感性分析给出同一结论。

### 可保留的次要结果

在真实 prefix 条件下，next-set NLL 相对 no-history（34/34）、weight shuffle
（34/34）、single path（29/34）和 mode shuffle（26/34）均改善。这说明间期事件含有
局部历史依赖，模型也确实学到了它；但该结果只能解释为局部条件统计，不能升级为完整传播
生成或机制证据。

### 与论文主线的关系

本结果不否定“间期群体事件刻画病理网络、并在发作早期静态能量场中被重用”的真实数据
发现。它否定的是更窄的模型命题：**当前 event-persistent、teacher-forced
path-mode RNN 不能作为该 shared scaffold 的充分生成机制。**

## 5. 工程性问题

- 510/510 正式 run 的 `run_state.json` 均为 `COMPLETE`，无 OOM、NaN 或训练失败。
- 峰值单进程 GPU memory 约 654 MB；所有 run 累计 21.92 process-hours，8 路并行的实际
  墙钟约 2 小时 52 分。
- analyzer 校验了 run 数、患者集合、seed、condition、checkpoint、精确训练覆盖、
  config fingerprint 和 target seal。
- 正式 launcher 暴露一个 conda 输出尾部空行问题：shell 把空字符串当成第 35 个患者，
  在所有有效任务完成后尝试移动 condition 根目录并退出。有效的 510 个结果未受损；
  runner 已增加空行过滤、34 人计数断言和 empty-subject hard gate。
- postprocess watcher 曾在 launcher 退出与 analyzer 写文件之间误报失败；现已改为先核对
  510 个 completion states，并可自动恢复 analyzer。

## 6. 最小修改路线

1. 本 v1.0 到此冻结为 34 人 bounded-negative，不再对同一结果调参追阳性。
2. 正文若需要保留 Figure 6，应把它定位为“局部可学但结构化自由生成失败”的模型边界，
   不能作为 shared scaffold 的正向机制图。
3. 若还要完成正向计算桥接，新版本必须先改训练目标：在真实 prefix NLL 之外，直接约束
   自由生成的 participation 与完整 rank distributions。
4. 新模型仍需独立的 graph/mode lesions 和纯间期硬门；通过后才打开发作期静态场。

## 7. 下一步的建议

这条线**可以继续，但不能沿当前模型做常规超参数优化**。最有信息量的下一步不是换 GRU
或把 RNN 做得更大，而是解决“局部 teacher-forced 预测阳性、自由生成分布阴性”的目标
失配，并让 path modes 在事件内真正可辨识。若不愿意重新定义训练目标，本次结果应作为
完整的模型 falsification 收口，论文的 shared-scaffold 结论继续由已有真实数据与 SNN
线承担。

## 关键产物

- 正式门：`results/topic5_structured_axis_graph/formal_persistent_path_mode_v1_0/analysis/formal_gate_summary.json`
- 主比较：`results/topic5_structured_axis_graph/formal_persistent_path_mode_v1_0/analysis/comparison_primary_statistics.csv`
- 结构损伤：`results/topic5_structured_axis_graph/formal_persistent_path_mode_v1_0/analysis/lesion_primary_statistics.csv`
- 内部状态：`results/topic5_structured_axis_graph/formal_persistent_path_mode_v1_0/analysis/intact_k2_internal_dynamics_cohort.csv`
- Figure 6：`results/paper-ready-figure/fig6_structured_rank_rnn/figures/fig6_structured_rank_rnn.png`
