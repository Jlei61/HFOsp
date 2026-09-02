# Topic 5.2 ECoG 物理近邻 RNN v0.1 设计冻结

状态：`FROZEN_FOR_EXECUTION`

日期：2026-08-16

## 0. 要回答的唯一核心问题

在高密度 ECoG 网格上，患者真实的物理近邻关系是否：

1. 在训练前作为正确结构约束，帮助 RNN 更好地预测未见过 block 中的后续触点；
2. 在模型训练完成后，仍被模型在线使用，因而削弱一块连续网格中的局部 recurrent 连接会选择性损害该区域的后续预测。

这是两条独立证据。第一条回答“正确几何是否帮助学习”，第二条回答“训练好的模型是否实际依赖这些局部连接”。只通过其中一条时，不把另一条写成已成立。

## 1. 患者与数据范围

### 1.1 预设患者顺序

- 主患者：Epilepsiae 958，8×8 网格，`GA1`–`GH8`，64/64 个触点有 v2 单触点 HFO 检测。
- 预设复制：Epilepsiae 1084，8×8 网格；`GC1` 在患者配置中被明确列为坏触点，因此使用其余 63 个触点，不插值、不补造。

患者选择只基于网格几何、原始 block、v2 检测与事件数量的可行性，不查看任何 RNN 结果。

### 1.2 信号与事件

- 频段：80–250 Hz ripple。
- 原始信号：Epilepsiae `.data + .head`，CAR，原采样率；只纳入 Nyquist 合格 block。
- E1084 的坏触点 `GC1` 在计算 CAR 之前即删除，而不只是从模型节点中删除。
- 单触点事件：复用 `results/hfo_detection/<subject>/*_gpu.npz`，不改变 HFO 检测定义。
- 群体事件窗口：重新从 v2 检测构建，不直接把旧的少触点 `lagPat` 当完整网格事件。
- 间期排除：使用 SQL canonical seizure inventory，删除与完整 EEG onset–offset 区间重叠的窗口；seizure field、能量和临床结局不进入训练或选模。
- 窗口参数：`ext=30 ms`、参与触发触点比例 `0.5`、患者既有 `pack_win_sec`（958: 0.25 s；1084: 0.18 s）、全网格同步比例达到 0.7 的窗口剔除。
- 窗口内每个网格触点是否参与，只由该触点的 v2 检测是否与窗口重叠决定。
- 触点时间值使用 80–250 Hz 时频质心；非参与触点必须为缺失值，绝不读取 legacy phantom rank。
- 相差不超过 5 ms 的质心合并为同一个 rank set；并列后统一重编号成连续的 `0..K-1` rank-set 标签，不保留 competition-rank 的空号；`0/2/10 ms` 作为敏感性分析。
- 为避免反复读取整小时原始信号，正式回填只读取每个事件窗口前后各 1 s 的合并区间，再执行同一 CAR、notch、80–250 Hz band-pass 与时频质心计算。该优化在模型训练前用 E958 939 个事件及 E1084 稀疏/高密度两个 block（21/450 个事件）对整块读取逐 event 验证，三组 rank-set 均 100% 一致。
- 事件绝对时间使用 SQL `block_start_epoch + window midpoint`；`.head.start_ts` 只作 block 级核对，不进入正式时间戳，因为抽查记录存在 8 h 时区偏移。

### 1.3 训练、验证、测试切分

先按 recording 内连续 block 分成最多 6 个相邻 block 的组；遇到非平凡 gap 立即断组。使用固定 hash 将整组分到 train/validation/test，比例目标为 70/15/15。一个 block 及其事件只能属于一个 split。

用于触发群体事件的触点集合只由 train block 的网格触点事件计数决定：

- 主规则：`count > mean + 1×SD`；
- 若少于 4 个触点，固定使用 train block 计数最高的 4 个触点；
- 该集合随后冻结并原样用于 validation/test block。

模型、超参数、patch 资格和任何标准化参数都不能读取 test suffix。

## 2. 完整网格模型

每个可用 ECoG 触点就是一个 RNN 节点，输入与输出都是同一组触点，因此 `H=I`。本实验不使用 latent tissue 插值，避免把“真实网格邻接”与观察算子混在一起。

每次事件开始时 hidden state 清零。第 k 个 rank set 作为多热输入，主 RNN 预测：

- 第 k+1 个 rank set 中的触点；
- STOP。

主 RNN 冻结后，再只用 train decisions 拟合相同的低容量 next-set size decoder，并由 validation 选择 epoch；它只服务于无未来信息的自由生成，不进入 contact NLL primary，也不读取 test 下一集合大小。

每个观测 rank set 固定做两次内部更新：第一次注入当前 rank set，第二次用零输入做一次 recurrent relaxation，再读出下一步。这样当前触点可以在本次决策中通过一条局部边影响邻居，避免单次更新造成“当前输入要到下一 rank 才能离开原节点”的人为一步延迟。一次更新作为架构敏感性，不替代两次更新的 primary。

正式优化配置在读取 test 结果前固定为：每个触点 1 个 leaky-RNN state、Adam `lr=0.006`、batch 512、E958 每 epoch 固定抽取最多 32,768 个 train events（E1084 使用全部 train events）、最少 15 epoch、最多 100 epoch、validation contact NLL early stopping patience 10、gradient clipping 5.0。三个模型 seed 为 `2026081611/12/13`。所有比较臂使用完全相同配置。

四种网络使用相同节点、输入、输出、损失、优化器、训练轮数、early stopping 和 3 个配对 seed。区别只在训练开始前固定的 recurrent mask 或训练标签。

## 3. 训练前物理几何检验

### 3.1 四种正式网络

1. `TRUE_GRID`：按 8×8 实际位置连接上下左右物理近邻；1084 跳过缺失 `GC1`，不跨空位补边。
2. `WRONG_GRID`：在 corner/edge/interior 度类别内置换触点身份，再连接同一张 8×8 格子。每个触点度数、总边数、互易性、连通性和整张图的谱保持，但“谁和谁是真实邻居”被破坏。
3. `DEGREE_RANDOM`：保持每个节点 in/out degree、总边数、互易性和强连通的随机图；不保留格子 motif。
4. `SUFFIX_SHUFFLED`：使用真实格子，保留每个事件最初三个 rank sets，在 train/validation 各自内部把后续 rank sets 与另一事件的 prefix 重新配对；匹配 suffix rank-set 数并禁止 donor suffix 与 recipient prefix 重叠。test 标签不打乱。

`WRONG_GRID` 使用 31 个预先生成且通过审计的置换，每个置换 3 个配对 seed。`DEGREE_RANDOM` 使用相同数量的图。不得根据结果删除“不好看”的图。

### 3.2 主要终点

主要终点是 held-out test 中每个真实下一触点决策的 contact NLL：

`TRUE_GRID - median(WRONG_GRID)`，负值表示真实物理邻接预测更好。

统计单位是图置换与训练 seed 的配对结果；报告完整置换分布和 exact one-sided p。不得把事件行当独立患者。

### 3.3 平行终点

- next-set top-k 命中率；
- STOP / remaining-length 轨迹；
- 只给第一 rank set 后自由生成的 full-grid 场与 held-out 经验场一致性；
- 去掉第一 rank set 后的生成场一致性；
- 按实际网格距离分层的下一触点 NLL；
- 4-neighbour 主分析与 8-neighbour 敏感性。

这些终点都执行，不由主要终点的显著性决定是否运行。

为控制自由生成的额外计算量，四类结构的 held-out 生成场使用在任何模型结果产生前固定的代表：`TRUE_GRID`、`SUFFIX_SHUFFLED`、`WRONG_GRID_00`、`DEGREE_RANDOM_00`，各 3 个 seed。31 张错位/随机图的完整分布只承担 contact-NLL primary，不根据 test 表现挑选“中位图”或“最好图”。

## 4. 训练后局部连接必要性检验

只在冻结的 `TRUE_GRID` checkpoint 上执行；参数 hash 在干预前后必须不变。

### 4.1 连续 patch

- primary：所有 train-eligible 2×2 连续 patch；
- sensitivity：有足够 train 覆盖的 3×3 patch；
- patch 资格只看 train 事件：至少 200 个事件中有 patch 触点参与，且至少 50 个训练决策的下一 rank set 进入 patch。
- 若某个边界 patch 使用了网格中数量稀少的 endpoint-degree 组合，导致排除 patch 自身后无法构造同度数的分散边集合，该 patch 记为 `MATCHING_INELIGIBLE`，不放宽匹配后混入 primary；逐患者同时报告 train-eligible 与 matching-eligible 的实际分母。

### 4.2 干预

对与 patch 相交的物理近邻 recurrent 边乘以剂量系数：1.0、0.75、0.5、0.0。相交指边的 source 或 target 至少一个位于 patch 内，包括 patch 内部边和跨 patch 边界的局部边。

每个 patch 生成 32 个分散边对照。每个对照与 patch 边集合匹配：

- 边数；
- 物理边长；
- source/target degree；
- 干预前权重绝对值分位；
- 不形成同样大小的连续空间块。

### 4.3 主要必要性终点

对每个 test 决策计算干预相对未干预的 NLL 增量，并形成差中差：

`(patch lesion 对“下一步进入 patch”的损害 - 对“下一步不进入 patch”的损害)`

减去

`(matched dispersed lesion 的相同差值)`。

正值且随削弱剂量单调增大，才支持“该连续局部连接块被模型在线使用”。

### 4.4 次要终点

- 连续 future-contact logits 的空间响应；
- 生成 suffix 场在 patch 内与 patch 外的改变；
- STOP probability 与离散生成长度；
- 当前 prefix 是否邻近 patch 的分层；
- 2×2 与 3×3 的一致性。

## 5. 解释矩阵

| 训练前真实几何更好 | 训练后局部削弱有选择性损害 | 允许结论 |
|---|---|---|
| 是 | 是 | 真实物理近邻既帮助学习，也被训练后模型在线使用 |
| 是 | 否 | 真实几何是有效训练约束，但未证明最终计算必须走这些边 |
| 否 | 是 | 多种图可学会任务，但真实网格模型内部依赖自己的局部实现 |
| 否 | 否 | 当前结果更符合一般共现/递归容量，未发现物理局部计算证据 |

单患者阳性只称为“E958 内的高密度网格机制证据”；E1084 方向一致后才称“跨两个 ECoG 患者复制”。不外推为队列结论。

## 6. 仅保留的硬 gate

### 6.1 工程 gate

- 原始 block、head、SQL、v2 GPU artifact 一一对应；
- grid channel 顺序固定且可回溯；
- train/validation/test block 无重叠；
- 所有非参与触点在 rank 矩阵中为 `-1`；
- checkpoint 参数 hash、图 hash、事件 cache hash 完整；
- 无 NaN/Inf，所有运行单元可恢复。

### 6.2 数值有效性 gate

- E958 至少 48/64 个触点在 train 事件中出现；
- train 至少 5,000 个含两个以上网格触点的事件；
- test 至少 1,000 个有效下一触点决策；
- patch 结果按实际 eligible denominator 报告，不因某个 patch 不够样本而停止其他 patch。

除以上两类，不设置按 p 值停止后续实验的科学 gate。

## 6.3 结果解封前冻结的 claim 判定

- `PHYSICAL_GRID_HELPS_LEARNING`：患者内 `TRUE_GRID - median(WRONG_GRID) < 0`，且 31 张错位网格的单侧 exact permutation `p <= 0.05`。
- `LOCAL_EDGES_USED_ONLINE`：2×2 primary patch 的完整削弱差中差在患者内同时满足：patch 中位数大于 0、分层随机化单侧 `p <= 0.05`，并且患者级 `1.0 -> 0.75 -> 0.5 -> 0.0` 剂量中位曲线随连接削弱而不下降。随机化不能把重叠 patch 当独立样本；在每个 `patch × seed` 层内，将 1 个连续 patch 边集合与 32 个匹配分散边集合交换“被检验集合”标签，随后按 `seed 中位数 -> patch 中位数` 聚合，固定 20,000 次。普通 patch bootstrap/sign-test 只作描述，不承担 claim。
- E958 与 E1084 分开判定；只有两位患者方向和判定均一致时才写“在两个 ECoG 患者中复制”。
- 8-neighbour、3×3、距离分层、自由生成场和 suffix-shuffled 均为平行/敏感性证据，不替代上述两条 primary，也不因 primary 阴性而停止。

## 7. 禁止措辞

- 不把 recurrent mask 写成白质连接或解剖通路；
- 不把某次 lesion 后性能下降写成自然组织损伤；
- 不把正确网格优于错误网格写成唯一 topology 被识别；
- 不把 E958 单患者结果写成 cohort-level ECoG 机制；
- 不把 core-triggered 群体事件误写成所有 64 个触点无条件发现的全脑事件。

## 8. 结果解封后的 P0 方向性修复（2026-08-16）

原 §4 干预同时削弱与 patch 相交的两个方向，因此既切断 `patch 外 -> patch 内`，也切断 `patch 内 -> patch 外`。在 E1084 解封后观察到 patch 外预测损害更大，确认原差中差混合了流入与流出；原结果必须保留为 `SYMMETRIC_ISOLATION`，不得再把其负号解释为“局部连接有保护作用”或“进入 patch 的必要性阴性”。

修复实验在运行任何有向结果前冻结为：

- `INBOUND_FIRST_ENTRY` 只削弱 recurrent matrix `[target, source]` 中 `target 在 patch、source 在 patch 外` 的有向边；不削弱反向出边或 patch 内部边。
- 只评价“截至当前 rank-set 从未招募过 patch 触点、下一 rank-set 至少一个触点第一次进入 patch”的决策；实现必须检查累计 recruited mask，不能只检查当前 rank set。outside comparator 同样要求截至当前都没有 patch 触点。
- 每个对照是完全避开该 patch 的分散有向边，精确匹配有向边数及 source-degree/target-degree 类别，并近似匹配训练后权重绝对值分位；仍为 32 个对照、4 个剂量、2×2 patch、3 个 seed。
- 统计仍用 `patch × seed` 内 33 候选集合的 20,000 次 focal-label randomization，不把重叠 patch 当独立样本。
- 因为该修复由 E1084 原干预语义触发，E1084 有向结果标为 development；E958 在有向结果未查看前锁定为 independent confirmation。无论结果如何，不能把该修复回写成原 §4 primary，也不能删除 `SYMMETRIC_ISOLATION` 结果。

E1084 有向差中差运行后进一步确认：两次内部更新使“outside”也受 `outside -> patch -> outside` 状态路径影响，因此 outside 不是干净的阴性终点。最终的直接必要性 estimand 在查看任何该 estimand 数值前冻结为：

`INBOUND_ENTRY_DAMAGE = patch 入边削弱在 first-entry decisions 上的 NLL 增量 - 32 个匹配分散有向边削弱在同一批 first-entry decisions 上的 NLL 增量中位数`。

它不再减 outside endpoint；剂量、patch、seed、匹配和分层随机化规则不变。E1084 仍是 development，E958 仍是 independent confirmation。原 symmetric DID 与 inbound DID 均原样保留为阴性，不得用这个修复覆盖或删除。
