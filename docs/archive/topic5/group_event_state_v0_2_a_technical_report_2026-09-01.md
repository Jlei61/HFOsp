# Group-Event State v0.2-A — 技术报告（H1/H2a predictive-state identification）

状态：**development；工程与仪器层已完成，科学层待队列跑完后填入**
代码线：`codex/topic5-group-event-state-v02-a`（base `codex/topic5-group-event-state-v0-2` @ f0c9e075）

配套：[共同科学合同](group_event_state_v0_2_common_contract_2026-09-01.md) ·
[工程不变量](group_event_state_v0_2_engineering_invariants_2026-09-01.md) ·
[A 线 spec](group_event_state_v0_2_h1_h2a_spec_plan_2026-09-01.md) ·
[合同条款逐条核验表](group_event_state_v0_2_agent_a_contract_clauses_2026-09-01.md) ·
[v0.1 数据合同](group_event_state_v0_1_data_contract_2026-08-31.md)

机器可读产物：`results/epi_prssm/group_event_state/v0_2/h1_h2a/`
大产物：`/data/hfosp_group_event_state_v0_2/agent_a/`

---

## 1. 相对 v0.1 改了什么，为什么每一处都是承重的

| v0.1 的做法 | 为什么对慢状态问题是错的 | v0.2-A 的做法 |
|---|---|---|
| 按**事件数** 70/10/20 切分 | 前几小时发放快十倍的患者，TRAIN 会变成几小时而 TEST 变成几天（或反过来），horizon 轴在患者间不再是同一个量 | 按**累计 recorded physical time** 切（CC §7.1），`v02/timeline.py::physical_time_split` |
| `n_streams=8` 把 TRAIN 区间等分成 8 段并行 | 每段从记录中途以初始化状态起步；对 next-event 模型只是轻微削弱，对"小时级状态"这个被测对象是直接破坏 | batch 的并行维是**真实 carry segment**，每个 slot 顺序流一整段、只 detach 不 reset（CC §7.2、EI §3） |
| 状态只在 recorded-session 变化处 reset | 发作与发作后一小时被静默桥接成一条状态链 | segment 在发作 onset 处断开，下一段从 offset + 60 min 起（CC §7.4–7.6） |
| 没有固定时间锚点，所有量按事件 | 一个吵闹小时对"关于时间"的结论贡献十倍于同长度的安静小时 | 每 5 min 一个固定物理 anchor，状态按真实 `dt` 传播到 grid（CC §5.2） |
| 只有 next-event 目标 | 无法回答"未来一段时间会怎样" | `P_slow` 直接训练 5/30/120 min future-block 头（CC §3.3） |

## 2. 一个未来块的目标是怎么表示的（clause C5）

dense 表示会是 `(anchor × horizon × 触点 × 频带)`，而且**语义上仍然是错的**——一个窗口里的事件数是可变的。

关键观察：spec 要求的每一个端点都是**逐事件的 proper score**，而每一个这样的分数只通过
计数与一阶/二阶矩依赖窗口。于是：

```
count            -> 事件数前缀和
participation    -> participation 布尔矩阵的前缀和   (k_c, N) 即充分统计量
continuous mark  -> x 与 x^2 的前缀和（仅 mark 有效的事件）
```

窗口统计量是一次减法。这不是 dense 计算的近似——
`test_window_statistics_match_brute_force` 逐窗口对着字面循环核对过。

由于 anchor 的合格条件要求整个窗口落在同一个 coverage segment 内，
**每个 anchor 的 exposure 恰好等于 horizon**：没有 exposure offset 可以被拟合变成免费截距。
这是本线 2026-08-26 复审记录过的失效模式，这里是结构性排除，不是事后检查。

## 3. 四类 mark 与 K-free repertoire 坐标

`v02/marks.py` 输出四族，分开报告：

| 族 | 内容 | 单位 |
|---|---|---|
| `participation` | 每触点是否参与 | nats / (事件 × 触点) |
| `size` + `span` | 招募到多少触点、持续多长才停（= extent/STOP） | nats / (事件 × 维) |
| `band_energy` + `band_peak` | 各**可用**频带上参与触点的平均能量与峰时 | 同上 |
| `embedding` | TRAIN 冻结 PCA 的 8 维连续 repertoire 坐标 | 同上 |

`embedding` 的存在理由是 CC §8：KMeans cluster 身份会让每一个下游结论依赖 K 与初始化，
连续坐标没有这个旋钮。cluster 只作可解释 secondary。

不支持的频带（`epilepsiae_253` / `epilepsiae_139` 的 150–250 Hz）**整列去掉**，不填 0。
mark 非有限的事件（cohort 约 0.03–0.05%）打 `valid=False` 标记并退出 conditional-mark 评分，
**不做插补**；它仍然计入 `p(N)`。

## 4. 评分与嵌套增量（clause C9）

唯一评分入口 `v02/scoring.py`：

- count：负二项（NB2）。Poisson 不可用——间期事件数在 5–120 min 上强过散，
  Poisson 分数会奖励任何把均值缩小的臂。
- participation：由 `(k_c, N)` 充分统计量还原的伯努利。
- 连续 mark：由一阶/二阶矩还原的高斯。

承重比较是同一批 anchor 上的嵌套增量 `B` vs `B + S`（CC §6），
读出用同一族 GLM 重新拟合。**为什么不用各自的头对打**：`P_local` 根本没有 future-block 头，
直接对打等于拿"被训练过这个问题的模型"和"从没被问过这个问题的模型"比。
冻结状态 + 共同读出让四个臂只差"`X` 里多了哪几列"。`P_slow` 自己的头作为 secondary 并报。

### 4.1 ridge 纪律（clause C13）

- 特征先标准化，数据项按单位数归一化 → 同一个 λ 网格在所有患者上含义一致。
- **逐 endpoint family 独立拟合并各自选 λ**。第一次 smoke 就撞到了共用 λ 的失败：
  `yuquan_chengshuai` 195 个训练 anchor 对 112 个特征，count 的 NLL 拟到 **27.5**
  而截距地板是 **6.85**；同一个 λ 下 participation 却正常——因为 participation 的观测数是
  `事件 × 触点`，比 count 多三个数量级。
- λ 由 **TRAIN 内部按时间分块的交叉验证**选。先前用 10% inner-validation 选，
  在 2 小时 horizon 上那一片只有 1 个独立窗口，选择等于抛硬币；
  后来试过"继承 5 分钟档的 λ"，**更糟**——在 200 个 anchor、目标容易的档位调出来的 λ
  远不足以约束 2 小时的 count，把一次拟合推到 NLL 892（截距 8.19）。
- λ 网格 `1e-4 … 1e5`。顶端等价于截距模型，所以顶格是"这些特征在这个 horizon 上没有增量"
  这一诚实答案，不是网格被截断。每一次拟合都记录选中的 λ 与是否顶格。
- 每个臂都对 **TRAIN 边缘分布截距模型**报分；比它差 0.5 nats 以上的标 `not_estimable`，
  **不当作弱阴性**。

## 5. Session-preserving 训练（clause C2）

`B` 个 slot，每个 slot 依次流一整个 carry segment；segment 之间打乱顺序，segment 内部块顺序不打乱；
只有 slot 换段时才 reset。EI §3 点名的三条最小测试全部落地并通过：

| 测试 | 断言 |
|---|---|
| `test_one_uninterrupted_forward_equals_the_chunked_carry` | 整段一次 forward 与 16 事件分块 carry 的每个 anchor 状态一致（1e-4） |
| `test_state_is_reset_across_a_recording_gap` | 改第一段的数据，第二段 anchor 状态逐位不变；第一段则变 |
| `test_shuffling_the_chunk_order_inside_a_segment_changes_the_answer` | 段内乱序必须改变结果 |

另外：`test_slot_count_does_not_change_the_state_of_any_anchor`、
`test_padded_slots_contribute_no_loss`、
`test_every_module_including_the_future_heads_actually_updates`（encoder / state / heads / future 四组
参数更新幅度均 > 0）、`test_future_loss_weights_are_frozen_after_the_initial_balance`。

### 5.1 `P_slow` 的 future 头读什么

future-block 头在**固定 5 min anchor** 上读，不在每个事件上读——
按事件训练会让吵闹小时的权重是同长度安静小时的十倍（CC §5.2 明确禁止），
而且训练对象会与评分对象不同。

长 horizon 头只读 `z_slow`（SP §2）。这只决定目标把信息推进哪个 latent，**不定义科学**：
承重评估冻结**整个**状态并在其上重拟合共同读出，所以没有 producer 能靠给 latent 起名叫 slow 取胜。

权重 `λ_5 / λ_30 / λ_120` 按初始化时各项对共享参数的梯度范数配平后**冻结**
（例：`epilepsiae_1073` 得到 9.15 / 7.72 / 3.42；`yuquan_zhangjinhan` 41.39 / 5.86 / 4.56），
不因 development 结果调整。

## 6. 分母（必须并报，EI §2）

| horizon | 每患者 test 段独立（不重叠）窗口数 |
|---|---|
| 5 min | 45–865 |
| 30 min | 6–138 |
| 120 min | **2–31** |

anchor 每 5 min 一个，所以 2 小时档相邻窗口共享 96% 的内容；
**anchor 数不是样本量**。全部逐患者逐 horizon 落在 `denominators.csv`。

4 位患者（`epilepsiae_922` / `yuquan_gaolan` / `yuquan_pengzihang` / `yuquan_sunyuanxin`）
在 120 min 档没有合格 anchor，记 `insufficient_coverage`——这是覆盖长度问题，**不是阴性**。

## 7. 发作与发作后排除的代价

按 CC §7.5，segment 在发作 onset 断开、下一段从 offset + 60 min 起。
代价是从间期流里移除了 **中位 10.3%、最多 33.1%** 的事件
（`epilepsiae_1146` 26 次发作 → 14,655/44,283）。逐患者数字在 per-subject JSON 的
`timeline.excluded`。60 min 是首轮 primary，参数化在 `SubjectTimelineConfig.postictal_exclusion_seconds`，
其他长度只作敏感性。

**判断说明**：CC §7.5 那一条以 "H2b 只读取 seizure 前 trajectory" 开头，但
"发作后不静默桥接：从 seizure offset 后 60 min 起新 segment" 位于面向三线的 §7 且措辞是一般性的。
本线按一般规则执行。若认为 A 线不应排除 postictal，改一个参数即可重跑。

## 8. `B_multiscale` 自身的性质（在读任何状态增量之前必须知道）

全 27 位患者、纯 CPU、314 秒跑完。相对截距地板：

| horizon | count | participation | continuous |
|---|---|---|---|
| 5 min | 中位 +0.118（22/27 更好） | +0.0228（26/27） | +0.0376（27/27） |
| 30 min | +0.088（19/27） | +0.0060（22/27） | +0.0230（25/27） |
| 120 min | +0.020（16/23） | +0.0005（15/23） | +0.0068（16/23） |

两条必须照实写的性质：

1. **120 min 的 count 上，23 位患者里 14 位把 ridge 推到网格顶端**，即多尺度特征
   对"两小时后会来多少事件"没有增量。这是基线自己的性质。
2. 4 个 (患者 × horizon × 端点) 单元里 `B_multiscale` 比截距还差，已标 `not_estimable`
   并从队列聚合中剔除且计数（`epilepsiae_1096` 5 min count、`yuquan_pengzihang` 5 min count、
   `epilepsiae_583` 三个 horizon 的 band energy、`yuquan_gaolan` 30 min band energy）。

小容量 MLP 与线性 GLM 在同一批 anchor 上互有胜负，所以线性 GLM 作主、MLP 作容量敏感性是公平的。

## 9. 仪器灵敏度（不是工程测试，是"这套装置能不能看见"）

`tests/test_topic5_group_event_state_v02_readout.py` 里有一对正/负对照：
在计数与 mark 都由一个已知慢驱动生成的合成数据上，把该驱动作为状态列加入，
count 增益 > 0.5 nats/window、participation > 0.02、continuous > 0.05；
把同一个驱动**打乱时间顺序**后，三个端点的增益都 < 0.05。
没有这一对，阴性结果无法与"估计器根本看不见"区分开。

## 10. 工程事故记录（都已修，都留了防线）

| 事故 | 后果 | 修法 |
|---|---|---|
| 三个 endpoint family 共用一个 ridge λ | count 拟合到 NLL 27.5（截距 6.85），participation 却正常 | 逐 family 独立拟合与选 λ |
| λ 由 10% inner-validation 选 | 2 小时档该片只有 1 个独立窗口，选择等于抛硬币 | 改 TRAIN 内部按时间分块 CV |
| 试过"长 horizon 继承短 horizon 的 λ" | 更糟：一次拟合到 NLL 892（截距 8.19） | 已撤回；不同 horizon 的信噪比不同，不可继承 |
| `torch.cuda.reset_peak_memory_stats(device)` 在 CUDA 未初始化时 | `RuntimeError: Invalid device argument`，两个 job 直接失败 | 在 import torch **之前**用 `CUDA_VISIBLE_DEVICES` 钉卡 |
| 验证 pass 重放全部 segment | 每个 epoch 多跑一遍整条记录 | 只重放**含验证事件**的 segment（段仍从自身起点热身，warm-up 不变） |
| `estimate_stats` 只接受连续区间 | v0.2 的 TRAIN 不是连续区间 | 加可选 `positions=`，默认行为不变 |

## 11. 结果

（待 162 个训练 job 跑完后填入：嵌套增量、block-shift 零假设、same-prefix continuation、
缩减 reset 阶梯、fast/slow-only、memoryless、粗匹配错时 donor。）

## 12. 未触碰

`/tmp/hfosp_group_event_state_v01`（v0.1 队列与结果树，只读）；Agent B 的 `h2b/`、
Agent C 的 `h3/`；formal / sealed 分区；paper-ready Fig1–Fig4；
`/data/hfosp_group_event_state_v0_1/dataset`（只读复用，未写入）。
