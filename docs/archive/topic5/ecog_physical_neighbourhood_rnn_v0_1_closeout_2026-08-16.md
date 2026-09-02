# Topic 5.2 高密度 ECoG 物理近邻 RNN v0.1 —— 细节版收口报告

日期：2026-08-16
冻结 spec：`docs/superpowers/specs/2026-08-16-topic5-ecog-physical-neighborhood-rnn-v0-1-design.md`
执行计划：`docs/superpowers/plans/2026-08-16-topic5-ecog-physical-neighborhood-rnn-v0-1.md`
白话版报告：`docs/archive/topic5/ecog_physical_neighbourhood_rnn_v0_1_plain_report_2026-08-16.md`

> 面向读者：需要复核数值、路径、合同与审计链的人。整段口语化叙述见白话版。

---

## 0. 一句话结论

真实的 ECoG 上下左右物理近邻，在 E958 中是**可辨识的训练归纳偏置**（held-out 下一触点 contact NLL 相对 31 张 degree-matched 错位网格 −0.13345 nats，31/31，exact one-sided p=0.03125）；在 E1084 方向一致但未通过严格图零模型（−0.02588，28/31，p=0.125）。**两位患者的优势在空间上完全局限于"下一触点就在上下左右紧邻"这一档**（E958 −0.5273、E1084 −0.1901，都是 31/31，raw p=0.03125；四档 Holm 后 0.125），更远的三档反而略差。训练完成后，**削弱进入连续 2×2 区域的真实局部入边，相对同数量、同端点度数、同权重分位的分散有向边，没有产生选择性损害**（E958 −0.00364，p=0.718；E1084 −0.09012，p=1.0），两位患者均不支持在线必要性。

**该训练优势是架构条件性的**：把每个 rank set 的内部更新从预注册的 2 次改成 1 次后，E958 的优势**完全消失**（+0.00048，15/31，p=0.531）；八邻接版本则不改变任何判定（E958 −0.12360，31/31，p=0.03125；E1084 −0.03675，28/31，p=0.125）。因此主结论必须写成"在两次内部更新的架构下"，不得表述为与架构无关的普遍结论。

---

## 1. 数据来源与两患者选择理由

- 数据集：Epilepsiae，raw `*.data + *.head`，元数据真值 `all_data_sqls/*.sql`（口径见 `docs/epilepsiae_dataset_structure.md`）。
- 单触点 HFO 事件复用既有 v2 检测 `results/hfo_detection/<subject>/*_gpu.npz`，本实验**不改动 HFO 检测定义**。
- 频段 80–250 Hz ripple；CAR；原采样率；只纳入 Nyquist 合格 block。
- **E958**：8×8 网格 `GA1`–`GH8`，64/64 触点全部有 v2 单触点检测 → 主患者。
- **E1084**：8×8 网格，患者配置明确把 `GC1` 列为坏触点，**在计算 CAR 之前即删除**，使用其余 63 个触点，不插值、不补造、不跨空位补边 → 预设复制患者。
- 患者选择只依据网格几何、原始 block 可用性、v2 检测覆盖与事件数量的可行性，**未查看任何 RNN 结果**（spec §1.1）。
- 间期口径：使用 SQL canonical seizure inventory 删除与完整 EEG onset–offset 区间重叠的窗口；seizure field / 能量 / 临床结局**不进入训练或选模**。

## 2. 事件构建、cache 与切分

### 2.1 群体事件窗口

- 参数：`ext=30 ms`，参与触发触点比例 `0.5`，患者既有 `pack_win_sec`（958: 0.25 s；1084: 0.18 s），全网格同步比例达 0.7 的窗口剔除。
- 触发触点集合**只由 train block 的网格触点事件计数决定**（`count > mean + 1×SD`；不足 4 个则取 train 计数最高的 4 个），随后冻结并原样用于 validation/test block。
- 窗口内每个网格触点是否参与，只由该触点的 v2 检测是否与窗口重叠决定；触点时间值使用 80–250 Hz 时频质心；**非参与触点在 rank 矩阵中必须为 `-1`**（决不读取 legacy phantom rank，见 AGENTS.md `lagPatRank` 条目）。
- 相差 ≤ 5 ms 的质心合并为同一 rank set，并列后统一重编号为连续 `0..K-1`（不保留 competition-rank 空号）。
- 事件绝对时间使用 SQL `block_start_epoch + window midpoint`；`.head.start_ts` 只作 block 级核对（抽查存在 8 h 时区偏移，不进入正式时间戳）。

### 2.2 稀疏读取等价性

正式回填只读取每个事件窗口前后各 1 s 的合并区间，再跑同一 CAR / notch / 80–250 Hz band-pass / 时频质心。该优化在训练前用整块读取逐 event 验证：

| block | 患者 | 事件数 | 参与数 | participation_exact | rank_matrix_exact | 质心最大偏差 |
|---|---|---|---|---|---|---|
| `95800102_0000` | E958 | 939 | 22,091 | ✅ | ✅ | 1.56e-3 ms |
| `108400102_0005` | E1084 | 21 | 283 | ✅ | ✅ | 8.57e-5 ms |
| `108401102_0026` | E1084 | 450 | 4,136 | ✅ | ✅ | 2.24e-4 ms |

`event_rows_exact_fraction = 1.0` 三块全中。审计文件：`results/topic5_ecog_physical_neighborhood_rnn_v0_1/sparse_validation/SPARSE_READ_EQUIVALENCE_AUDIT.json`。

### 2.3 最终 cache 与 split

先按 recording 内连续 block 分成最多 6 个相邻 block 的组，遇非平凡 gap 立即断组；固定 hash 分到 train/validation/test，目标 70/15/15，一个 block 及其事件只属于一个 split。

| 患者 | 触点 | 事件总数 | train | validation | test | test 连续决策数 | 最少触点出现事件数 | `events_sha256` |
|---|---|---|---|---|---|---|---|---|
| E958 | 64 | 163,438 | 108,061 | 32,463 | 22,914 | 211,670 | 109 | `8db43bca…` |
| E1084 | 63 | 7,918 | 5,728 | 650 | 1,540 | 6,059 | 63 | `8d0623e9…` |

数值有效性 gate（spec §6.2）全部满足：E958 ≥48/64 触点出现在 train 事件（实际 64/64，最少出现 109 次）；train ≥5,000 个含 ≥2 触点的事件；test ≥1,000 个有效下一触点决策（E958 211,670；E1084 6,059）。

**全部 384 个正式四邻接训练单元、154+ 个敏感性单元的 `events_sha256` 唯一且与 `INPUT_AUDIT.json` 完全一致**（逐单元核对）。

## 3. 四种训练条件的精确定义

每个可用 ECoG 触点 = 一个 RNN 节点，输入输出同一组触点，`H=I`（不使用 latent tissue 插值，避免把"真实网格邻接"与观察算子混在一起）。每个事件开始时 hidden state 清零；第 k 个 rank set 作为多热输入，主 RNN 预测第 k+1 个 rank set 的触点 + STOP。每个观测 rank set 固定做 **2 次内部更新**（先注入当前 rank set，再用零输入做一次 recurrent relaxation 后读出）。

| 条件 | 定义 | 图数 × seed |
|---|---|---|
| `TRUE_GRID` | 按 8×8 实际位置连接上下左右物理近邻；E1084 跳过缺失 `GC1`，不跨空位补边 | 1 × 3 |
| `WRONG_GRID` | 在 corner/edge/interior 度类别内置换触点身份，再连同一张 8×8 格子；每触点度数、总边数、互易性、连通性与整图谱保持不变 | 31 × 3 |
| `DEGREE_RANDOM` | 保持每节点 in/out degree、总边数、互易性与强连通的随机图；不保留格子 motif | 31 × 3 |
| `SUFFIX_SHUFFLED` | 使用真实格子，保留每个事件最初 3 个 rank set，在 train/validation 各自内部把后续 rank set 与另一事件的 prefix 重新配对（匹配 suffix rank-set 数、禁止 donor suffix 与 recipient prefix 重叠）；**test 标签不打乱** | 1 × 3 |

图审计（`INPUT_AUDIT.json::graph_records`，63 张/患者，全部 `same_directed_edge_count` / `same_per_node_degree` / `symmetric` 通过，63 张 mask 互不相同）：

| 患者 | 触点 | 有向边 | 度集合 | 错位图与真边重合率（中位 / 范围） | 随机图与真边重合率 |
|---|---|---|---|---|---|
| E958 | 64 | 224 | {2,3,4} | 0.071 (0.018–0.134) | 0.062 (0.027–0.107) |
| E1084 | 63 | 218 | {2,3,4} | 0.073 (0.028–0.110) | 0.064 (0.009–0.101) |

`SUFFIX_SHUFFLED` 标签审计：`development_first_three_exact=true`、`test_exact=true`（3 个 null 全中）；E958 development 事件改动比例 0.9963，E1084 为 0.713–0.719。

**四种网络使用完全相同的节点、输入、输出、损失、优化器、训练轮数、early stopping 与 3 个配对 seed；差别只在训练开始前固定的 recurrent mask 或训练标签。**

## 4. 训练配置与 unit 数

正式配置在读取任何 test 结果前冻结：

- 每触点 1 个 leaky-RNN state（`state_dim=1`），`microsteps=2`
- Adam `lr=0.006`，`batch_size=512`，`gradient_clip=5.0`
- 每 epoch 最多抽取 32,768 个 train event（E1084 train 只有 5,728 → 全用）
- `min_epochs=15`，`max_epochs=100`，validation contact NLL early stopping `patience=10`，`min_relative_improvement=1e-4`
- 3 个模型 seed：`2026081611 / 12 / 13`
- 正式训练全部在 **CPU** 上执行（`training_device_type=cpu`）

正式 unit 数：

| 矩阵 | 单元数 | 状态 |
|---|---|---|
| 四邻接主矩阵（2 患者 × 64 图 × 3 seed，其中 SUFFIX_SHUFFLED 复用 TRUE_GRID 图） | **384/384** | ✅ |
| held-out 扩展指标 | **384/384** | ✅ |
| 自由生成场（2 患者 × 4 代表 × 3 seed） | **24/24** | ✅ |
| 八邻接敏感性（2 患者 × 32 图 × 3 seed） | **192/192** | ✅ |
| 一次内部更新敏感性（2 患者 × 32 图 × 3 seed） | **192/192** | ✅ |
| symmetric patch（2 患者 × 3 seed × {2×2, 3×3}） | **12/12** | ✅ |
| inbound first-entry patch（2 患者 × 3 seed × 2×2） | **6/6** | ✅ |

**收敛状态（诚实限制）**：E958 的 192 个单元中 **103 个跑满 100 epoch 上限**（best epoch 中位 93），即多数 E958 模型在训练预算耗尽时验证损失仍在下降；E1084 只有 11/192 跑满上限（best epoch 中位 56 / 完成 68）。四种条件预算完全相同，**比较是公平的，但 E958 的绝对 NLL 不是收敛值**。单元训练机时中位数：E958 14.1 min、E1084 0.9 min；四邻接矩阵总机时 E958 41.2 h + E1084 2.7 h。

**配对初始化**：`initial_parameter_sha256` 在每个 `subject × seed` 内唯一（`initial_trainable_parameters_identical_within_seed=true`，两患者均通过），即所有比较臂从完全相同的初始参数出发。

## 5. 主要终点：held-out 下一触点 contact NLL

统计单位是**图置换 × 训练 seed 的配对结果**（不把事件行当独立样本）。逐图 effect = `median_seed(TRUE_GRID) − median_seed(control graph)`，负值表示真实物理邻接更好；患者级 effect 取 31 个逐图 effect 的中位数。exact one-sided p = 真值在 31 张对照图统计量中的 plus-one 排位 `(1 + #{control ≤ true}) / 32`（脚本 `exact_lower_tail`），因此该检验的**最小可能 p 值就是 1/32 = 0.03125**。95% CI 是对 31 个逐图 effect 的中位数做固定种子 bootstrap（`bootstrap_median_ci`），只描述图间离散度，不承担 claim。

| 患者 | TRUE_GRID NLL | vs 31 张错位网格 | 95% CI | 方向计数 | exact p | vs 31 张度数随机图 | 方向计数 | exact p | vs 事件结尾打乱 |
|---|---|---|---|---|---|---|---|---|---|
| **E958** | 3.27045 | **−0.13345** | [−0.14319, −0.12877] | **31/31** | **0.03125** | **−0.13596** | **31/31** | **0.03125** | −0.04935（3/3 seed 同向） |
| **E1084** | 2.99772 | −0.02588 | [−0.03596, −0.01747] | 28/31 | 0.125 | −0.03117 | 27/31 | 0.15625 | −0.06069（2/3 seed 同向） |

对照臂中位 NLL：E958 错位 3.40390 / 随机 3.40641；E1084 错位 3.02360 / 随机 3.02889。

**冻结 claim 判定 `PHYSICAL_GRID_HELPS_LEARNING`（spec §6.3）**：E958 满足（effect<0 且 p≤0.05）；E1084 不满足。**不得写"在两个 ECoG 患者中复制"**。

## 6. 平行终点：top-1、观测基数召回、STOP、距离分层

`top1` 的合同是 `top_prediction_is_any_member_of_tied_next_rank_set_v0.1`（预测的最高分触点属于并列的下一 rank set 即算命中）；`top_observed_cardinality_recall` = 取与真实下一 rank set 同样多的最高分触点后的召回率。

| 患者 | TRUE_GRID top1 | recall | STOP Brier | STOP BCE |
|---|---|---|---|---|
| E958 | 0.2347 | 0.2118 | 0.0635 | 0.2044 |
| E1084 | 0.3945 | 0.3513 | 0.0884 | 0.2713 |

相对 31 张错位网格 / 31 张度数随机图。**下表 effect 一律是原始的 `median_seed(TRUE_GRID) − median_graph(control)`，不做符号归一**：`top1` / `recall` 是**越大越好**（正值 = 真格子更好），`STOP Brier` 是**越小越好**（负值 = 真格子更好）；`better_count` 已按各自的"更好"方向计数。

| 患者 | 指标 | vs 错位网格 | vs 随机图 |
|---|---|---|---|
| E958 | top1 | **+0.05836, 31/31, p=0.0312** | **+0.05566, 31/31, p=0.0312** |
| E958 | recall | **+0.05021, 31/31, p=0.0312** | **+0.04899, 31/31, p=0.0312** |
| E958 | STOP Brier | −0.00058, 23/31, p=0.281 | −0.00085, 20/31, p=0.375 |
| E1084 | top1 | −0.00875, 11/31, p=0.656 | −0.00792, 8/31, p=0.750 |
| E1084 | recall | −0.00419, 10/31, p=0.688 | −0.00355, 10/31, p=0.688 |
| E1084 | STOP Brier | −0.00744, 29/31, p=0.0938 | −0.00551, 29/31, p=0.0938 |

**距离分层（每个真实下一触点按其到当前 rank set 最近触点的网格欧氏距离分箱，箱内对真实触点求平均损失）**：

| 患者 | 对照族 | 紧邻（上下左右） | 斜对角 | 隔两格 | 更远 |
|---|---|---|---|---|---|
| E958 | 错位网格 | **−0.5273, 31/31, p=0.0312, Holm 0.125** | +0.1362, 0/31, p=1.0 | +0.1063, 0/31, p=1.0 | +0.1143, 0/31, p=1.0 |
| E958 | 随机图 | **−0.5335, 31/31, p=0.0312, Holm 0.125** | +0.1341, 0/31 | +0.1000, 0/31 | +0.1048, 0/31 |
| E1084 | 错位网格 | **−0.1901, 31/31, p=0.0312, Holm 0.125** | +0.1490, 1/31 | +0.2237, 0/31 | +0.0701, 1/31 |
| E1084 | 随机图 | **−0.2043, 31/31, p=0.0312, Holm 0.125** | +0.1411, 0/31 | +0.2337, 0/31 | +0.0761, 1/31 |

E1084 距离分箱样本量（TRUE_GRID seed0）：紧邻 7,448 / 斜对角 2,987 / 隔两格 1,619 / 更远 5,274。

**读法**：真实物理邻接买到的准确度严格长在它自己连出来的那些边上，代价是所有更远跳跃都略差；两位患者、两族零模型、四种组合方向完全一致（8/8 个"紧邻"格子都是 31/31）。但四个距离箱一起做 Holm 校正后，紧邻箱的调整 p 均为 0.125，**未达严格显著**。因此这条只能写成"方向一致的局部优势（secondary，directional）"，不能写成校正后成立的独立主结论。

真实 − 事件结尾打乱的 contact NLL（中位 seed）：E958 −0.03149、E1084 −0.04826（两患者都表明真实的事件后续顺序本身携带可学信息）。

## 7. 自由生成场

主 RNN 冻结后，**只用 train decisions** 另拟一个同样低容量的 next-set size decoder（validation 选 epoch，上限 200 epoch / patience 20，最多 200,000 个 train 决策与 100,000 个 validation 决策）。它只服务于无未来信息的自由生成，**不进入 contact NLL primary，也不读取 test 下一集合大小**。

为控制额外计算量，四类结构的 held-out 生成场使用**在任何模型结果产生前就固定的代表**：`TRUE_GRID`、`SUFFIX_SHUFFLED`、`WRONG_GRID_00`、`DEGREE_RANDOM_00`，各 3 seed，共 24 个单元（**未按 test 表现挑代表图**，spec §3.3）。31 张错位/随机图的完整分布只承担 contact-NLL primary。

- `full_field_spearman`：只给第一个 rank set 后自由生成整段事件，把"每个触点参与得多不多"这张分布图与 held-out 经验场做 Spearman。
- `start_removed_field_spearman`：从生成场中去掉被白送的第一个 rank set 后再比。

| 患者 | 条件 | 代表图 | 完整生成场 ρ（3 seed 中位） | 去掉起点后 ρ | 生成事件参与数中位 | 实测参与数中位 |
|---|---|---|---|---|---|---|
| E958 | 真实上下左右邻接 | `TRUE_GRID` | **0.9244** | **0.8799** | 11 | 19 |
| E958 | 触点位置打乱 | `WRONG_GRID_00` | 0.9186 | 0.8796 | 11 | 19 |
| E958 | 度数保持随机 | `DEGREE_RANDOM_00` | 0.9184 | 0.8698 | 12 | 19 |
| E958 | 事件结尾打乱 | `TRUE_GRID` | 0.9013 | 0.8297 | 14 | 19 |
| E1084 | 真实上下左右邻接 | `TRUE_GRID` | 0.9127 | **0.7944** | 5 | 10 |
| E1084 | 触点位置打乱 | `WRONG_GRID_00` | **0.9184** | 0.7713 | 5 | 10 |
| E1084 | 度数保持随机 | `DEGREE_RANDOM_00` | 0.9130 | 0.6751 | 5 | 10 |
| E1084 | 事件结尾打乱 | `TRUE_GRID` | 0.9073 | 0.7167 | 5 | 10 |

**读法**：四种网络都能把 held-out 的粗空间分布场重建到 ρ≈0.90–0.93，家族之间的差距（E958 跨度 0.901–0.924；E1084 0.907–0.918，且 E1084 的**名义最高值出现在触点位置打乱那一档**）远小于逐决策 contact NLL 上的差距。**这条平行终点对"哪张网络更对"是不敏感的**，判别力全部落在逐步下一触点预测上。去掉白送起点后差距略微拉开（E1084 真格子 0.794 vs 随机 0.675），但每族只有 1 张代表图 × 3 seed，按 spec 不做正式统计检验，只作描述。

**共同缺陷**：四种网络都把事件生成得偏短（E958 生成参与数中位 11–14 vs 实测 19；E1084 全部 5 vs 实测 10）。这是低容量 size decoder + 闭环 rollout 的共同限制，不是某一种网络独有的问题。

## 8. 训练后局部连接必要性

### 8.1 原始 symmetric 干预（`SYMMETRIC_ISOLATION`，保留为阴性证据）

原 spec §4 的干预对**与区域相交的所有物理近邻 recurrent 边**（source 或 target 任一在区域内，含区域内部边与跨界边）乘剂量系数 1.0 / 0.75 / 0.5 / 0.0，主要终点是差中差：

```
(区域削弱对"下一步进入区域"的损害 − 对"下一步不进入区域"的损害)
  − (匹配分散边削弱的相同差值)
```

| 患者 | 区域 | 合格区域数 | 完整削弱 DID | 95% CI | 正/负区域 | 患者中位剂量曲线单调 | 单调区域数 | 分层随机化 p |
|---|---|---|---|---|---|---|---|---|
| E958 | 2×2（primary） | 49 | **+0.01395** | [−0.0362, +0.0864] | 26 / 23 | ❌ | 20/49 | **0.124**（20,000 次） |
| E958 | 3×3（sensitivity） | 20 | +0.04761 | [−0.0979, +0.1772] | 11 / 9 | ❌ | 8/20 | 未做（按 spec 只对 2×2 primary 做随机化） |
| E1084 | 2×2（primary） | 46 | **−0.24742** | [−0.2966, −0.1999] | 3 / 43 | ❌ | 1/46 | **1.0**（20,000 次） |
| E1084 | 3×3（sensitivity） | 16 | −0.26428 | [−0.3469, −0.1798] | 0 / 16 | ❌ | 0/16 | 未做 |

中间剂量（0.75 / 0.5 档）：E958 2×2 = +0.00758 / +0.01755，3×3 = +0.04248 / +0.05499；E1084 2×2 = −0.05762 / −0.11604，3×3 = −0.04723 / −0.10295。

**判定**：`LOCAL_EDGES_USED_ONLINE`（定义在 2×2 primary 上）两位患者均不满足。E958 未达显著且剂量曲线不单调；E1084 强负。

**E1084 的强负号正是触发 P0 修复的信号**：它意味着切断与区域相交的边之后，**区域外**的预测反而比区域内受损更多，说明这个差中差同时混进了流入与流出两条通路，不能读成"局部连接有保护作用"，也不能读成"进入区域的必要性阴性"。该结果原样保留为 `SYMMETRIC_ISOLATION`（§11 时间线第 2 步）。

### 8.2 最终直接检验：`INBOUND_FIRST_ENTRY`

**Estimand（在查看任何该 estimand 数值前冻结，spec §8）**：

```
INBOUND_ENTRY_DAMAGE
  = 削弱"区域外 → 连续 2×2 区域内"有向入边在 first-entry 决策上的 NLL 增量
  − 32 个匹配分散有向边削弱在同一批 first-entry 决策上的 NLL 增量中位数
```

不再减去 outside endpoint（因为两次内部更新使 outside 也受 `outside → patch → outside` 状态路径影响，outside 不是干净的阴性终点）。

**first-entry 合同**：`no_patch_contact_recruited_before_next_rank_v0.1` —— 只评价"截至当前 rank set **从未招募过**该区域任何触点、且下一 rank set 至少一个触点第一次进入该区域"的决策。实现使用累计 `recruited` mask（`scripts/run_topic5_ecog_patch_necessity_v0_1.py` 中 `batch["recruited"]`），不是只检查当前 rank set。

**干预**：只对 recurrent matrix `[target, source]` 中 `target ∈ patch 且 source ∉ patch` 的有向边乘以剂量系数 1.0 / 0.75 / 0.5 / 0.0；不削弱反向出边，不削弱区域内部边。

**结果**：

| 患者 | 角色 | 配对区域数 | dose 0.75 | dose 0.5 | dose 0（完整削弱） | 正/负区域 | 剂量曲线单调 | 分层随机化 p | null 2.5–97.5% |
|---|---|---|---|---|---|---|---|---|---|
| **E958** | independent confirmation | 49 | +0.00088 | +0.00125 | **−0.00364** | 24 / 25 | ❌ | **0.718** | [−0.01194, +0.01290] |
| **E1084** | development | 47 | −0.01492 | −0.03076 | **−0.09012** | 10 / 37 | ❌ | **1.0** | [−0.00634, +0.00716] |

**冻结 claim 判定 `LOCAL_EDGES_USED_ONLINE`：两位患者均不满足**（要求中位数 > 0、单侧 p ≤ 0.05、剂量曲线不下降；三条全部不满足）。E958 的观测值落在自身零分布正中间，E1084 明显在负侧。

## 9. 匹配对照与患者内随机化

每个区域生成 **32 个分散有向边对照**，精确匹配：有向边数、source-degree / target-degree 类别；近似匹配干预前权重绝对值分位；且**完全避开该区域的所有触点**（`control[patch, :]` 与 `control[:, patch]` 均为空）；不形成同样大小的连续空间块。

统计不把重叠区域当独立样本：在每个 `patch × seed` 层内，把 1 个连续区域边集合与 32 个匹配分散边集合交换"被检验集合"标签，随后按 `seed 中位数 → patch 中位数` 聚合，**固定 20,000 次 focal-label randomization**。普通 patch bootstrap / sign-test 只作描述，不承担 claim。

区域资格分母（`INBOUND_ENTRY_UNIT_AUDIT.csv`，逐 seed）：

| 患者 | seed | train-eligible | matching-eligible | matching-ineligible | 参数 hash 不变 |
|---|---|---|---|---|---|
| E958 | 0 / 1 / 2 | 49 / 49 / 49 | 49 / 49 / 49 | 0 / 0 / 0 | ✅ ✅ ✅ |
| E1084 | 0 / 1 / 2 | 47 / 47 / 47 | 47 / 47 / 47 | 0 / 0 / 0 | ✅ ✅ ✅ |

（区域资格只看 train 事件：至少 200 个事件中有区域触点参与，且至少 50 个训练决策的下一 rank set 进入该区域。E958 8×8 共 49 个 2×2 区域全部合格；E1084 因 `GC1` 缺失只有 47 个完整 2×2 区域，全部合格。）

## 10. 敏感性

### 10.1 八邻接（把斜对角也算成邻居）

同一套流程重跑，只把 `TRUE_GRID` 从上下左右四邻接换成含斜对角的八邻接，对应的 31 张错位网格也在八邻接图上生成（`graphs/<subject>/eight_neighbour/`）。不含 `DEGREE_RANDOM` 与 `SUFFIX_SHUFFLED`，故每患者 32 图 × 3 seed = 96 单元。

| 患者 | 真格子 NLL | 错位网格中位 | 真 − 中位（图级） | 方向计数 | exact p |
|---|---|---|---|---|---|
| E958 | 3.26965 | 3.39324 | **−0.12360** | **31/31** | **0.03125** |
| E1084 | 3.00751 | 3.04426 | −0.03675 | 28/31 | 0.125 |

**结论不变**：E958 仍然是 31/31 全胜、达到该检验的 p 下限；E1084 仍然方向一致但未跨门槛（28/31，与四邻接主分析的 28/31 完全一致）。八邻接版本的效应量比四邻接主分析略小（E958 −0.1236 vs −0.1335），但两患者的判定都没有翻转。

### 10.2 一次内部更新（**改变了 E958 主结论的适用边界，不是中性稳健性检查**）

把每个观测 rank set 的内部更新从 2 次改成 1 次，其余全部不变（同样的图、同样的事件 cache、同样的优化器与预算、同样 3 个配对 seed）。

| 患者 | 真格子 NLL | 错位网格中位 | 真 − 中位（图级） | 方向计数 | exact p |
|---|---|---|---|---|---|
| **E958** | 3.38921 | 3.38873 | **+0.00048** | **15/31** | **0.53125** |
| E1084 | 3.00529 | 3.05105 | −0.04576 | 30/31 | 0.0625 |

**E958 的训练优势在一次内部更新下完全消失**：真格子只赢 15/31 张错位网格（随机排位就是 ~15.5/31），效应量 +0.0005 nats（方向甚至为正，即略差），p=0.531。作为对照，两次内部更新的主分析是 −0.13345、31/31、p=0.031。

这与距离分层的结果**互相印证，且指向同一个架构事实**：模型的两次内部更新中，第一次注入当前 rank set，第二次做一次零输入的 recurrent relaxation 后读出。回归测试 `test_two_microsteps_allow_current_input_to_reach_a_neighbour` 证明了——在 `microsteps=1` 下，当前 rank set 输入对**邻居触点** logit 的梯度**精确为 0**（`< 1e-12`），而在 `microsteps=2` 下不为 0（`> 1e-8`）。也就是说，一次更新的模型在结构上**无法**表达"刚刚放电的触点在本次决策内推动它的物理邻居"，而 E958 的全部优势恰好只长在"下一触点就在上下左右紧邻"这一档（−0.5273，31/31）。

绝对损失也一致：E958 从两次更新的 3.2705 涨到一次更新的 3.3892（差 0.119 nats），**这个由第二次更新本身买到的量，与真格子相对错位网格买到的量（0.133 nats）是同一个数量级**。

**必须按预注册档位表述**：spec §2 明确"一次更新作为架构敏感性，不替代两次更新的 primary"，spec §6.3 也明确敏感性不替代 primary。因此：

- 主结论仍然是**两次内部更新架构下**的 `PHYSICAL_GRID_HELPS_LEARNING`（E958 成立）。
- 但必须同时写明**该结论是架构条件性的**：在只做一次内部更新的变体中，E958 的优势不存在。不得把主结论表述成"与架构无关的、真实物理邻接普遍有助于学习"。
- 上面那段机制解读（第二次更新提供了"当前触点在本步内到达邻居"的通路）是**对两条已观测结果 + 一条已验证架构性质的一致性解读**，不是独立检验，不得写成"已证明该机制"。

E1084 的（本来就未跨门槛的）优势**不**依赖两次更新（一次更新下 −0.0458、30/31、p=0.0625，比其两次更新版本 −0.0259、28/31、p=0.125 略强）。两位患者在这一点上的方向不同；鉴于只有 31 张对照图、p 在 0.125↔0.0625 之间移动，这个差异本身在噪声范围内，**不作为独立发现报告**。

### 10.3 tie tolerance（仅数据构建敏感性，不重训模型）

| 患者 | 容差 | 事件数 | train/val/test | 参与数中位 | rank-set 数中位 | 相对 5 ms 的 held-out 生成场 Spearman |
|---|---|---|---|---|---|---|
| E958 | 0 ms | 163,441 | 108,063 / 32,463 / 22,915 | 22 | 22 | 0.99945 |
| E958 | 2 ms | 163,439 | 108,062 / 32,463 / 22,914 | 22 | 15 | 0.99968 |
| E958 | 10 ms | 163,428 | 108,057 / 32,462 / 22,909 | 22 | 7 | 0.99954 |
| E1084 | 0 ms | 7,955 | 5,753 / 660 / 1,542 | 9 | 9 | 0.99707 |
| E1084 | 2 ms | 7,943 | 5,747 / 654 / 1,542 | 9 | 6 | 0.99885 |
| E1084 | 10 ms | 7,732 | 5,589 / 623 / 1,520 | 9 | 3 | 0.99914 |

合并阈值把"一个事件被切成几步"改变了 3 倍以上（E958 rank-set 中位数 22 → 7），但 held-out 空间分布场几乎不变（全部 ≥0.997）。

## 11. P0 修复时间线（必须完整保留）

1. **原始设计（spec §4）**：干预同时削弱与区域相交的**两个方向**（source 或 target 任一在区域内），因此既切断 `区域外 → 区域内` 也切断 `区域内 → 区域外`。
2. **E1084 解封后发现**：区域外预测的损害更大，确认原差中差混合了流入与流出两个方向。原结果保留为 `SYMMETRIC_ISOLATION`，**其负号不得再解释为"局部连接有保护作用"或"进入区域的必要性阴性"**。
3. **修复一（`INBOUND` 有向化）**：只削弱 `target ∈ patch 且 source ∉ patch` 的有向入边。
4. **发现二**：两次内部更新使 outside 也受 `outside → patch → outside` 状态路径影响，outside 不是干净阴性终点。
5. **修复二（最终 estimand）**：不再做差中差，直接在**同一批 first-entry 决策**上比较"区域入边削弱"与"32 组匹配分散有向边削弱"的 NLL 增量（§8.2）。
6. **实现错误修复**：旧代码只检查当前 rank set 是否在区域内；正式实现必须检查**累计 recruited mask**，确保此前从未进入过该区域。合同名 `no_patch_contact_recruited_before_next_rank_v0.1`，回归测试 `test_first_entry_coverage_excludes_later_reentry`。
7. **角色分配**：因为修复由 E1084 原干预语义触发，**E1084 有向结果标为 development**；**E958 在有向结果未查看前锁定为 independent confirmation**。修复后的检验**不得回写成原 §4 primary**，`SYMMETRIC_ISOLATION` 结果**不得删除**。

## 12. 工程验收

| 检查 | 结果 |
|---|---|
| 输入审计 `INPUT_AUDIT.json::pass` | ✅ true（11 项 gate 全过） |
| 稀疏 vs 整块读取 rank 矩阵逐位一致 | ✅ `all_participation_exact` + `all_rank_matrices_exact` |
| 四张图 manifest（2 患者 × 四/八邻接）各 31 wrong + 31 random | ✅ |
| 四邻接正式单元 384/384，合同字段（cpu / batch 512 / microsteps 2 / state_dim 1 / tied-set top1） | ✅ 零违规 |
| checkpoint SHA-256 与 summary 记录一致 | ✅ 384/384 |
| train/validation/test 输出无 NaN/Inf | ✅ |
| 每个 `subject × seed` 内初始可训练参数 hash 唯一 | ✅ |
| 所有训练 worker 状态 JSON `failed` 为空 | ✅ 70 份日志，0 个失败 |
| symmetric patch 12/12 参数 hash 干预前后不变 + 每区域 32 对照 | ✅ |
| inbound patch 6/6 参数 hash 不变 + 32 对照 + first-entry 合同名 | ✅ |
| tied-set top-1 合同统一 | ✅ 384/384；第一轮修复 106 个 E1084 单元、最大 held-out NLL 复现误差 **4.960e-7**（≤1e-6）、`model_parameters_changed=false`；本轮全量复扫 `n_repaired=0, n_already_current=384` |
| 全部单元 `events_sha256` 与 `INPUT_AUDIT` 一致 | ✅ 逐单元核对 |
| OOM / 非有限值 / 运行失败 | 无 |
| 目标测试 `tests/test_topic5_ecog_physical_neighborhood_v0_1.py` | ✅ 14/14 |
| `py_compile`（31 个 `*topic5_ecog*` 脚本 + 2 个 src 模块 + 测试） | ✅ |
| 最终 closeout 审计 | ✅ **PASS，22/22 项**（`CLOSEOUT_AUDIT.json`，exit 0） |

### 12.1 本轮接手期间的两处工程修复

1. **E1084 symmetric patch 缺 `lesion_mode` 标签**：那 6 个单元产出于该字段引入之前，summarizer 会硬失败。已用当前代码原样重跑并逐值核对：**63 个区域 × 22 个原有列共 3,440+ 个数值全部 bit-identical（max |diff| = 0）**，`MATCHED_CONTROL_RESULTS.csv` 逐字节相同，`SUMMARY.json` 只有 `lesion_mode` 与 `runtime_sec` 变化。原始阴性数值未被任何修改。
2. **绘图脚本字体**：`scripts/plot_topic5_ecog_physical_neighbourhood_v0_1.py` 原写死 `font.family="Arial"`，本机未安装 Arial（其余 10 个 `plot_topic5_*.py` 均用 `DejaVu Sans`），会静默回退并撑爆版式。已改为显式 `DejaVu Sans` 并重排 panel 网格（每个 panel 拆成"上排示意图 / 下排结果"两行、加大左边距）。**未改动任何文字内容或统计标注。**

## 13. 允许与禁止的措辞

**允许**：

- "在 E958 这一位高密度 ECoG 患者中，训练前固定真实上下左右物理近邻，比 31 张度数配平的位置打乱网格更利于从未见 recording block 中学习下一触点（−0.133 nats，31/31，exact p=0.031）。"
- "E1084 方向一致但未通过严格图零模型；因此不能称为跨两位患者的复制。"
- "两位患者中，真实物理近邻的预测优势都严格集中在下一触点位于上下左右紧邻的转移上（各 31/31 张对照图），更远距离反而略差；四距离箱 Holm 校正后未达严格显著，属方向一致的次要证据。"
- "训练完成后，削弱进入连续 2×2 区域的真实局部入边，相对同数量、同端点度数、同训练后权重分位的分散有向边，未产生选择性损害。"
- "真实局部几何更像有用的训练归纳偏置，而不是当前证据下唯一且必要的在线计算通路。"
- "该训练优势依赖于预注册的两次内部更新架构：把内部更新减到一次后，E958 的优势消失（+0.0005，15/31，p=0.53）。把斜对角也算成邻居则不改变任何判定。"

**禁止**（spec §7 + 本轮补充）：

- 不把 recurrent mask 写成白质连接、解剖通路或"组织骨架"。
- 不把某次 lesion 后性能下降写成自然组织损伤。
- 不把正确网格优于错误网格写成"唯一 topology 被识别"。
- 不把 E958 单患者结果写成 cohort-level ECoG 机制结论。
- 不把 core-triggered 群体事件误写成所有 64 个触点无条件发现的全脑事件。
- 不因删边阴性就说"模型只学了共现"。
- 不因训练网格阳性就说在线必要性也成立。
- 不把修复后的 first-entry 检验回写成原 §4 预注册 primary。
- **不把训练优势写成与模型架构无关的普遍结论**（一次内部更新下 E958 优势消失）。
- **不把一次内部更新的结果升格成 primary 或独立发现** —— spec §2/§6.3 把它锁定为架构敏感性；它的作用是给主结论加边界条件，不是替代主结论，也不是"证明了某个机制"。

## 14. 图的逐 panel 解释

生产者：`scripts/plot_topic5_ecog_physical_neighbourhood_v0_1.py`。输出 PNG（400 dpi）/ PDF（矢量、Type-42 内嵌字体）/ SVG 三份 + `FIGURE_METADATA.json` + 中文 `README.md`。版面为 2 个 panel，每个 panel 上排画"干预是什么"，下排画"结果是什么"，两位患者并排且共用纵轴范围。

### Panel A —— 训练前固定网络约束，对未见 block 的下一触点预测

- **上排 4 张 8×8 示意图**：以 `GE5` 为中心触点（红点），画出四种条件下该触点在训练开始前被连到哪里。淡灰细线是真实网格骨架（作为空间参照，四张图相同）。
  - 蓝：真实上下左右近邻（`TRUE_GRID`）。
  - 橙：触点位置打乱、但每个触点度数/总边数/互易性/整图谱保持（`WRONG_GRID_00`，31 张中的第 1 张，**在任何模型结果产生前就固定为示意代表**）。
  - 紫：度数保持的随机图（`DEGREE_RANDOM_00`，同上）。
  - 灰：真实网格但训练标签的事件后续被重新配对（`SUFFIX_SHUFFLED`；因为图本身就是真格子，所以画出来与蓝图形状一致，**这是设计本身，不是错画**）。
- **下排 2 张点图（E958 左 / E1084 右）**：纵轴 = 对照条件相对真实网格**多出来的** held-out 下一触点损失（nats/真实下一触点），**正值表示真实上下左右近邻更好**；虚线 0 = 打平。
  - 第 1、2 列每个点是一张**预先冻结**的对照图（31 张错位 / 31 张随机），值取该图 3 个 seed 的中位数再减真实网格的 3-seed 中位数；小提琴是这 31 个点的密度；黑短横是中位数。
  - 第 3 列只有 3 个点（事件结尾打乱只有 1 张图 × 3 seed），逐 seed 与同 seed 的真实网格配对相减，故不画小提琴。
  - `*` 只在**真实网格 vs 31 张错位网格的单侧 exact 图尾检验 p ≤ 0.05** 时出现 —— 仅 E958 有。
- **读者应看到**：E958 的 31 张错位图和 31 张随机图**全部落在 0 以上**（真格子全胜）；E1084 的两团明显更靠近 0 且有点落到 0 以下。

### Panel B —— 训练完成后削弱进入连续 2×2 区域的真实局部入边

- **上排 2 张示意图**：
  - 红：真实干预 —— 只画 `区域外 → 区域内` 的**有向**入边（箭头指向粉色的 2×2 区域）。方向按 recurrent 矩阵 `[target, source]` 索引读出，与正式 estimand 完全一致；不含反向出边与区域内部边。
  - 灰：同数量的分散有向边示意。**这是示意，不是某一次正式对照** —— 正式对照对每个区域、每个 seed 重新生成 32 组，并按有向边数、source/target degree 类别与训练后权重绝对值分位匹配（脚本内注释已注明）。
- **下排 2 张剂量曲线（E958 左 / E1084 右）**：横轴 = 保留的入边强度（100% → 75% → 50% → 0%，**向右递减**）；纵轴 = 在同一批 first-entry 决策上，"削弱区域入边"比"削弱 32 组匹配分散边的中位数"**多出来的**损失。**正值才表示真实入边具有选择性必要性。**
  - 每条浅线是一个合格区域在 3 个 seed 上的中位响应（E958 49 条 / E1084 47 条）；色带是区域间四分位范围；粗线 + 空心圆是患者内中位数。
  - `*` 只在分层 focal-label 随机化单侧 p ≤ 0.05 时出现 —— **两位患者都没有**。
- **读者应看到**：E958 的粗线**贴着 0 平走**（浅线向正负两侧对称散开）；E1084 的粗线**随削弱加深单调下探到 −0.09**。图面不含任何暗示删边阳性的元素。

### 版面与字体核对（本轮实际执行）

首次渲染后逐帧目视检查，发现四处版式问题（条件标题横向重叠、纵轴标签压到相邻子图、横轴刻度标签互相重叠、最右子图标签被裁切），根因是脚本写死 `font.family="Arial"` 而本机未安装 Arial，静默回退到更宽的 DejaVu Sans 后版面溢出。修复方式为把字体显式设为仓库通用的 `DejaVu Sans` 并把每个 panel 从"一行 6 格"改为"上排示意 / 下排结果"两行、加大左边距；**未增删任何文字、未改动任何统计标注或数据变换**。重渲染后无重叠无裁切；PDF 首页栅格化后与 PNG 逐元素一致。

## 15. 正式结果与审计文件路径

工作根：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-lbss-rnn-v0-1`
结果根（下称 `<R>`）：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-lbss-rnn-v0-1/results/topic5_ecog_physical_neighborhood_rnn_v0_1`

### 15.1 冻结合同与审计

| 内容 | 绝对路径 |
|---|---|
| 冻结 spec | `/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-lbss-rnn-v0-1/docs/superpowers/specs/2026-08-16-topic5-ecog-physical-neighborhood-rnn-v0-1-design.md` |
| 执行计划 | `/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-lbss-rnn-v0-1/docs/superpowers/plans/2026-08-16-topic5-ecog-physical-neighborhood-rnn-v0-1.md` |
| 输入审计 | `<R>/INPUT_AUDIT.json` |
| 稀疏读取等价性审计 | `<R>/sparse_validation/SPARSE_READ_EQUIVALENCE_AUDIT.json` |
| 最终工程 closeout 审计 | `<R>/CLOSEOUT_AUDIT.json` |
| 最终 claim 裁决 | `<R>/FINAL_CLAIM_ADJUDICATION.json` |
| tied-set top-1 修复记录（本轮复扫） | `<R>/training/TIED_TOP1_REPAIR.json` |
| tied-set top-1 修复记录（第一轮 106 个 E1084 单元） | `<R>/training/TIED_TOP1_REPAIR_PASS1_1084_2026-08-16.json` |

### 15.2 数据与图合同

| 内容 | 绝对路径 |
|---|---|
| block split / 网格触点 / 事件可行性 / 区域可行性 | `<R>/feasibility/{BLOCK_SPLIT.csv, GRID_CHANNELS.csv, EVENT_FEASIBILITY.json, PATCH_FEASIBILITY.csv}` |
| 患者事件 cache + provenance | `<R>/cache/{958,1084}/{events.npz, events_suffix_null_seed{0,1,2}.npz, provenance.json, per_block/}` |
| 冻结图（四邻接主 / 八邻接敏感性） | `<R>/graphs/{958,1084}/{four_neighbour,eight_neighbour}/{TRUE_GRID,WRONG_GRID_00..30,DEGREE_RANDOM_00..30}.npz` + `GRAPH_MANIFEST.json` |

### 15.3 训练单元与逐单元产物

| 内容 | 绝对路径 |
|---|---|
| 四邻接正式矩阵（384 单元） | `<R>/training/{958,1084}/<FAMILY>__<GRAPH_ID>__seed<K>/{summary.json, checkpoint.pt, heldout_extended_metrics.json}` |
| 自由生成场（24 单元） | `<R>/training/{958,1084}/<UNIT>/{field_metrics.json, heldout_free_fields.npz}` |
| 八邻接敏感性 | `<R>/training_eight_neighbour/{958,1084}/<UNIT>/summary.json` |
| 一次内部更新敏感性 | `<R>/training_one_microstep/{958,1084}/<UNIT>/summary.json` |
| 单元清单 | `<R>/{training,training_eight_neighbour,training_one_microstep}/TRAINING_UNIT_MANIFEST.csv` |
| worker 状态 JSON | `<R>/{training,training_eight_neighbour,training_one_microstep}/worker_logs/*.json` |

### 15.4 汇总统计

| 内容 | 绝对路径 |
|---|---|
| 主 NLL（患者 / 图 / 单元三级） | `<R>/summary/{GRAPH_TRAINING_SUMMARY.json, PATIENT_RESULTS.csv, GRAPH_LEVEL_EFFECTS.csv, TRAINING_UNIT_RESULTS.csv}` |
| held-out 扩展指标（top1 / recall / STOP / 距离分层） | `<R>/summary/{HELDOUT_EXTENDED_SUMMARY.json, HELDOUT_EXTENDED_PATIENT_RESULTS.csv, HELDOUT_EXTENDED_GRAPH_EFFECTS.csv, HELDOUT_EXTENDED_UNIT_RESULTS.csv}` |
| 自由生成场 | `<R>/summary/{FREE_FIELD_SUMMARY.json, FREE_FIELD_FAMILY_RESULTS.csv, FREE_FIELD_UNIT_RESULTS.csv}` |
| 原始 symmetric 区域删边 | `<R>/summary/{PATCH_NECESSITY_SUMMARY.json, PATCH_PATIENT_RESULTS.csv, PATCH_SEED_AGGREGATED_RESULTS.csv, PATCH_UNIT_AUDIT.csv}` + 逐单元 `<R>/patch_necessity/{958,1084}/seed{0,1,2}/patch_{2x2,3x3}/{SUMMARY.json, PATCH_RESULTS.csv, MATCHED_CONTROL_RESULTS.csv}` |
| 最终直接入边检验 | `<R>/summary_inbound/{INBOUND_ENTRY_SUMMARY.json, INBOUND_ENTRY_PATIENT_RESULTS.csv, INBOUND_ENTRY_PATCH_RESULTS.csv, INBOUND_ENTRY_UNIT_AUDIT.csv}` + 逐单元 `<R>/patch_necessity_inbound/{958,1084}/seed{0,1,2}/patch_2x2/{SUMMARY.json, PATCH_RESULTS.csv, MATCHED_CONTROL_RESULTS.csv}` |
| 八邻接 / 一次内部更新 / tie tolerance | `<R>/summary/{EIGHT_NEIGHBOUR_SUMMARY.json, EIGHT_NEIGHBOUR_PATIENT_RESULTS.csv, ONE_MICROSTEP_SUMMARY.json, ONE_MICROSTEP_PATIENT_RESULTS.csv, TIE_TOLERANCE_SENSITIVITY.json}` |

### 15.5 图

`<R>/figures/{topic5_ecog_physical_neighbourhood_v0_1.png, .pdf, .svg, FIGURE_METADATA.json, README.md}`

### 15.6 生产者脚本

`scripts/` 下 31 个 `*topic5_ecog*` 脚本；核心模块 `src/topic5_ecog_physical_neighborhood_v0_1.py`、`src/topic5_wiring_economy_rnn.py`；目标测试 `tests/test_topic5_ecog_physical_neighborhood_v0_1.py`。
