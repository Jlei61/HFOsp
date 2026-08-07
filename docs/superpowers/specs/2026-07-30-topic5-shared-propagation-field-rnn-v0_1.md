# Topic 5：Shared Propagation Field RNN v0.1 科学与数据合同

## 0. 当前状态

- 合同名：`topic5_shared_propagation_field_v0_1`
- 层级：exploratory system-identification development
- Closure note（2026-07-31）：**v0.1 已按其原问题 100% 完成并阴性收口。**
  后续 SIG-RNN 改变了生成因子分解、feedback 和科学对象，属于新合同，不是
  v0.1 调参或续跑。
- 工程状态：v0.4 六患者 × 3 seed × 8 模型全部完成，144/144 个拟合可用
- Post-stop 诊断：learning curve 432 fits、dimension sensitivity 108 fits、
  SNN 只读辨识 18 fits，训练充分性修复后全部可用
- 人类开发结论：**输出不反馈的确定性 autonomous latent-trajectory model
  未被数据选择，按 bounded negative 收口**
- SNN 状态：既有 Topic 4 artifact 的 Round 5 仅保留为 exploratory
  compatibility check。它不满足同条件 pooling 与嵌套事件数 `N_min` 合同，
  且存在 first-rank lookup 捷径，**不能判 G0，现已移出所有 RNN Gate**
- 多轮最终审阅：
  `docs/archive/topic5/shared_propagation_field_rnn_multiround_review_2026-07-31.md`
- 初始阶梯结果：
  `docs/archive/topic5/shared_propagation_field_rnn_ladder_pilot_2026-07-30.md`

当前安全结论是：

> 完整事件中存在超出静态 participation 和普通 stationary first-order
> Markov 的可预测组织；但在控制事件进度和离散路径后，数据没有选择由第一
> rank 初始化、输出不反馈、确定性演化的低维 autonomous latent trajectory。
> 该结果不能排除可辨识的稳定 contact interaction，因为 v0.1 没有显式
> contact-level interaction，也没有让生成分支改变后续状态。

不得写成“已辨识稳定传播场”“一个场产生患者内多样轨迹”“已恢复有效连接”，
也不得写成“G0 未通过”或“数据不支持任何稳定传播结构”。

## 1. 冻结的科学问题

目标条件分布为：

`p(X_{2:T} | X_1, k_{2:T}, T)`。

模型可读取第一 rank set、事件长度和每一步 cardinality；禁止读取旧
`heldout20`、A/B、pathological axis、SOZ、物理坐标、发作数据、IEI 和
event identity。自主 field 不读取 rank index、真实 suffix 或已生成 contact。

这检验的是：给定起点与同步规模，一个跨事件共享的自主 latent system 能否
自由生成完整 contact 顺序；不是 next-rank teacher forcing。

## 2. 数据与划分

唯一人类输入是：

`results/topic5_interictal_rank_distribution/dataset_v0_4`

旧 train80 内按时间顺序再分为：

- inner train：前 70%
- monitor validation：随后 15%，只用于选 checkpoint
- development test：最后 15%，所有模型 checkpoint 锁定后才评分

旧 outer heldout20 完全排除于训练、选模和评分。冻结 NPZ 共置两个 split，
因此这不是“字节级从未读取”的声明。六名 pilot 是 target-blind 的工程覆盖样本，
不是生物亚型，也不是独立确认 cohort。

加载时必须验证 masked、连续 rank；非参与 contact 为 `-1`；同一 contact
每场最多出现一次；生成时永久排除已招募 contact。`T` 和 `k_t` 只规定 rollout
长度和集合大小，不得被当作命中指标。

## 3. 八模型公平阶梯

全部模型共享相同：

- inner train / validation / development-test 事件
- 训练集极大似然 static scaffold
- candidate mask 与 without-replacement 支持集
- exact conditional `k`-subset observation likelihood
- 第一 rank、长度和 cardinality 条件
- 自由生成器与评价事件

比较项为：

| 代号 | 模型 | 科学作用 |
| --- | --- | --- |
| M0 | static scaffold | 静态 participation |
| M1 | stationary first-order Markov | 普通局部转移 |
| M1-phase | phase-conditioned first-order Markov | 控制事件进度 |
| M2 | mixture first-order Markov | 少量离散路径 |
| M2-phase | phase-conditioned mixture Markov | 同时控制离散路径和进度 |
| M3 | latent time-indexed template | 低维时间形状，无自主 recurrence |
| M4 | autonomous latent-trajectory null | v0.1 主检验；输出不反馈 |
| M4-phase | phase-conditioned field | 非自主时钟诊断，不支持 autonomous-field claim |

M3 与 M4 共用 encoder、prior/posterior、loading 和 observation family；M3
以 `t/(T-1)` 生成状态，M4 仅由自身 recurrent transition 推进。M4-phase
用于检查 M3–M4 差异是否只来自进度信息。

## 4. 拟合与评分合同

### 4.1 拟合

- M0 scaffold 在 exact subset likelihood 下做极大似然拟合后冻结。
- M1/M2 直接最大化相同 observation likelihood。
- M3/M4 优化该 likelihood 的 ELBO；因此“同一 likelihood family”不等于
  “同一优化目标”，报告时不得隐藏这一点。
- mixture 分量用非同值初始化，避免 K>1 数学上塌回 K=1。
- 每个模型使用与阶梯顺序无关的稳定 model seed。

### 4.2 训练充分性

每次拟合保存 epoch 曲线、最佳模型、最佳点 optimizer state 和完整训练尝试。
以下 run 不得进入模型比较：

- `NO_LEARNING_PROGRESS`
- `EARLY_OPTIMUM_UNVERIFIED`
- `INSUFFICIENT_EPOCHS`
- `NOT_CONVERGED`

若最佳点过早，必须从同一初始化自动降低学习率复核。patience 只被达到预设
相对量级的改善重置；末位小数抖动不能无限延长训练。M0 标为
`NO_FREE_PARAMETERS`。

### 4.3 评分

主表用 NLL / suffix decision，避免不同事件长度污染 NLL / event。

- M0–M2：解析 exact likelihood。
- M3/M4：重复 importance-weighted marginal-likelihood estimate；full-event
  posterior 只作 importance proposal，不用于生成。
- 同时报告重复的 future-blind prior-predictive NLL；所有正式 rollout 均从
  `p(z0 | X1)` 开始。
- 报告 Monte Carlo SD、逐决策位置 NLL、4 次独立 rollout 的均值和 SD。

正式比较以患者为单位；3 seeds 先在患者内折叠。

## 5. 可解释结构边界

自主模型的 emitted contact 不回写 latent state，因此一般
`do(contact i active)` 或 contact lesion 在 v0.1 中没有定义。raw `A`、raw
`Q` 和 `Q Q^T` 均不能称为突触连接或已验证的有效连接。

若未来重开 G2，主对象只能是 empirically supported
`p(generated suffix | first rank set)` 的可观测等价性；latent Gram matrix
仅作 secondary diagnostic。

## 6. SNN 既有资产的历史兼容性检查

本轮不运行 SNN simulator。只读审计入口：

`scripts/audit_topic5_spf_existing_snn_controls.py`

冻结输出：

`results/topic5_shared_propagation_field/snn_positive_control/existing_artifact_reuse/`

现有 E1146 family：

| family | seeds | raw events | model-ready | 方向 |
| --- | ---: | ---: | ---: | --- |
| source-only | 20 | 246 | 180 | 158 forward / 0 reverse / 22 unreadable |
| sink-only | 20 | 237 | 230 | 0 forward / 217 reverse / 13 unreadable |
| paired source/sink | 21 | 222 | 222 | 103 forward / 119 reverse |

前两行是从保存 ranks/coordinates 以显式 `k_dir=2, eps=2 mm` 重算的方向；
历史 reported labels 作为单独字段保留。方向来自低阈值 pathological
kernel/core 位于哪一端；E→E 各向异性塑造传播
通道，但 `AR=1` 本身不应抹去方向。因此旧 isotropic yield probe 只能作
诊断，不能作“方向应消失”的负对照。

这些文件跨条件和 seed 的 pool 不具备冻结的同条件观测合同；`N_min` 也没有
通过同一条件下的 nested event-count curve 建立。后续将 21 个 paired 文件
合并训练的 Round 5 因此不能升级为 G0。

在 paired seeds 1–15 训练、16–18 monitor、19–21 test 的探索性评分中，M4
的 NLL/decision 为 `0.528`，M1-phase 为 `0.113`、M3 为 `0.244`；但 M4-phase
和 M4 的方向 Brier 分别为 `0.087/0.090`，优于列入比较的其余模型。第一 rank
查表在 source-only、sink-only 和 paired 上已分别达到 `100%/100%/78.4%`
方向准确率，因此 NLL 和方向迁移给出冲突排序，且两者都不能证明或否定结构
恢复。Round 5 的正式状态是：

`EXPLORATORY_COMPATIBILITY_CHECK_ONLY / G0_NOT_EVALUABLE`

RNN 与 SNN 从此独立开发；SNN 不再承担 RNN 的进入门或正对照。

## 7. v0.1 Gate 与停止规则

- 历史 SNN G0：`REMOVED_FROM_RNN_GATE_NOT_EVALUABLE_FROM_ROUND5`
- v0.1 主停止门：`STOP_RULE_REACHED_NOT_SELECTED`；M4 未超过公平的
  M2-phase 或 M3 对照
- M4-phase vs M3：`NON_SELECTION_TIE`；2/6、患者中位差约
  `+0.010 NLL/decision`，不能写成方向性失败
- 稳定 contact structure：v0.1 没有 emitted-contact feedback 或可执行的
  contact intervention object，因此未开放该 Gate，而不是判负
- one structure, many trajectories：超出 v0.1 模型对象，不启动
- 34 人全队列：不启动
- mixture of fields、更多 GRU、低秩约束和发作 readout：不得用于挽救该主张

小样本 learning curve 只能回答“归纳偏置”的方法学问题，不能把 full-data
G1 的失败改写成 shared-field 阳性。2026-07-31 的 post-stop nested learning
curve 已按此边界执行：10%–100% 六档
中，M4 相对 M3 每档均为 `0/6`，没有小样本归纳偏置优势。`d={2,4,6}` 中
autonomous M4 同样均为 `0/6`。这些诊断只收窄 v0.1 null 的解释边界；
它们复用 development test，不是独立复制，也不评价显式反馈 graph。

## 8. 当前实现入口

- 核心模型：`src/topic5_shared_propagation_field.py`
- canonical runner：`scripts/run_topic5_spf_model_ladder.py`
- fail-closed launcher：`scripts/launch_topic5_spf_ladder_pilot.sh`
- patient-first 聚合：`scripts/aggregate_topic5_spf_ladder.py`
- SNN artifact-only 审计：`scripts/audit_topic5_spf_existing_snn_controls.py`
- 多轮 checkpoint 诊断：`scripts/analyze_topic5_spf_multiround.py`
- nested learning curve：`scripts/run_topic5_spf_nested_learning_curve.py`
- SNN 只读辨识：`scripts/run_topic5_spf_existing_snn_identifiability.py`
- latent-d sensitivity：`scripts/run_topic5_spf_latent_dimension_sensitivity.py`
- dynamic-residual stability：
  `scripts/analyze_topic5_spf_dynamic_residual_stability.py`
- field utilization：`scripts/analyze_topic5_spf_field_utilization.py`
- fail-closed 总判决：`scripts/build_topic5_spf_multiround_verdict.py`
- 配置：`config/topic5_shared_propagation_field_v0_1.yaml`
- 测试：`tests/test_topic5_shared_propagation_field.py`、
  `tests/test_topic5_spf_model_ladder.py`

每个 shard 必须保存 config SHA、source SHA、输入 SHA、split SHA、checkpoint
SHA 和四个 leakage flags。聚合器拒绝缺失笛卡尔积、未完成 shard、混合配置、
混合 source 版本或任一泄漏标志。
