# Continuous Marked State R1.6：优化器与可识别性诊断合同

## 1. 核心问题

本阶段不直接再次裁定 H1--H3，而是先回答：

> 当前的 epoch-0、零梯度和零边，到底来自训练/选择失败，还是在当前输入、模型和数据支持下确实没有可识别增量？

工程上能完成训练不等于科学假设被有效检验。只有先证明模型在同一目标上可学习、优化器能稳定到达可泛化解，人体阴性才有解释力。

## 2. 已确认的旧设计问题

R1.5 的 target-alignment 从已在完整 TRAIN 上 refit 的 R1.2 checkpoint 出发，却用 TRAIN 尾部 20% 作为新的 inner-validation，并把该 checkpoint 当作 epoch 0。这个 epoch 0 已经见过 inner-validation，因此 target-alignment 更新与 epoch 0 的比较不公平。旧 R1.5 的 no-update 只能说明“更新没有胜过一个见过选择集的起点”，不能区分优化失败和科学阴性。

同时，旧结果只保存 inner-validation NLL、最大梯度和最终更新量，没有保存训练 NLL、optimizer steps、裁剪频率、各参数组实际 learning rate 或每轮更新/参数比；8 个 epoch 在不同患者上对应约 872--7,304 个 optimizer steps。旧产物不足以诊断收敛。

## 3. 冻结范围

- 仅使用 development TRAIN 与 development validation；formal/sealed 始终关闭。
- 不改变输入、IED timing+mark likelihood、state dimension、患者身份或 H1/H2a 端点。
- 主实验只优化 explicit persistent T1；raw observer、seizure probe 和 paper-ready 图不进入本阶段。
- 固定六位患者：
  - `yuquan_zhangkexuan`：旧 R1.5 stable positive；
  - `epilepsiae_384`：旧 R1.5 partial update；
  - `epilepsiae_1096`：大样本 no-update；
  - `yuquan_zhangjiaqi`：长记录 no-update；
  - `yuquan_chengshuai`、`yuquan_chenziyang`：旧长记录校准层。
- seed 是优化稳定性重复，不当作患者重复。

## 4. 选择安全的三段时间设计

所有界限在每位患者 TRAIN 内按时间冻结：

1. `base_train`：TRAIN 前 60%；
2. `base_select`：TRAIN 的 60--80%；
3. `alignment_select`：TRAIN 的 80--100%。

R1.2 core 只能在 `base_train` 拟合、在 `base_select` 选 epoch，再按选定预算 refit 到 TRAIN 前 80%。R1.6 target-alignment 只能在 TRAIN 前 80% 拟合、在 `alignment_select` 选配置和 epoch。这样 epoch 0 与更新臂都没有见过 `alignment_select`。

配置冻结后，最终模型才允许从同一初始构造出发，按选定预算在完整 TRAIN 上 refit；development validation 只评分，不参与配置或 epoch 选择。

## 5. 先做可学习性，不先扫人体网格

### 5.1 Synthetic

至少包含：

- positive truth：当前模型族内存在 observation→state→mark/timing 信号；
- zero truth：没有状态增量；
- reversed truth：方向相反仍可恢复；
- scale stress：不同事件数和梯度尺度下恢复方向不变。

positive 必须在至少 4/5 seeds 降低 unseen inner NLL，并恢复非零参数；zero 不应系统选择非零更新。

### 5.2 小样本过拟合

对张克轩、E384、E1096、张家齐各取事前固定的短连续 `base_train` 子段。若模型连训练子段都不能明显降低 exact joint NLL，先判为训练/实现失败，不解释人体假设。

## 6. 优化配置搜索

不做无界超参数挖掘。先在 synthetic 与小样本过拟合上筛掉明显不可训练配置，再在人体验证以下小矩阵：

搜索必须分两层，不能只调 target-alignment：

1. 先用 `base_train`→`base_select` 选择一个全患者公共的 prefix/core 配置，比较 learning rate、4/8/12 个完整时间 pass、64/128/256 anchors 的更新粒度、weight decay、warm-up、clip 以及 Adam 诊断；
2. 冻结该 prefix 配置并 refit 到 TRAIN 前 80%，再在 `alignment_select` 上选择 observer/readout 配置。

若最高学习率的 8-pass prefix 在全部稳定 seeds 都选择预算末端，允许在冻结前只补一个同设置 12-pass 边界延伸；不据此展开全因子矩阵。该条件已在 3/3 张克轩 seeds 触发，因此 prefix 预期单元从 96 增至 108。

初始试跑中，固定旧 prefix 配置在 E1096、E384 和张家齐均 3/3 选择 epoch 0，只有张克轩更新；因此任何在旧 prefix 上展开的后半段大矩阵只算诊断，不承担最终结论。

- optimizer：`AdamW` 主线；`Adam` 仅作无 weight-decay 等价诊断；
- state/readout LR：`3e-5, 1e-4, 3e-4, 1e-3`；
- observer LR 固定为 state LR 的 `0.1`，另测 `0.03`；
- weight decay：`0` 与 `1e-3`；
- warm-up：`0` 与前 10% optimizer steps；
- global clip：`1.0` 与 `5.0`；
- target-alignment 预算：常规 `4+4` 个 observer/joint pass，并加一个 `8+8` 的固定公共配置检验预算不足；
- checkpoint selection：旧的“任意微小改善即更新”与 `min_delta=1e-4`、stage 内 patience=3 的诊断配置；synthetic 另用较长 patience 校准，要求同时保留正真值恢复并压低零真值的伪更新；
- budget 以 optimizer steps 和完整时间 pass 同时报，不再只报 epoch。

先做逐因素/小型覆盖，不做全因子组合。配置由 tuning seeds `0,1,2` 的 patient-first inner 指标选择；排序先最大化至少 2/3 seeds 同向改善的患者数，若并列，再比较这些稳定患者的中位改善量，随后才使用全体患者中位数和事前配置顺序打破剩余并列。固定一个全患者公共配置，不给每位患者单独挑最好参数。seeds `3,4` 用于确认训练轨迹，development validation 只对冻结配置评分。

## 7. 每次拟合必须保存

- 完整 optimizer 参数组、实际 LR schedule、weight decay、clip 和 warm-up；
- 每轮/固定 steps 的 TRAIN 与 inner NLL（joint、timing、mark、STOP/size、subset）；
- 每组梯度 norm、clip 前后 norm、裁剪比例、非有限梯度计数；
- 每组参数 norm、update norm、update/parameter ratio；
- 每阶段 optimizer steps、有效事件数、anchors、session 数；
- epoch 0 是否见过 inner-selection 数据；
- 最终 checkpoint、配置、代码、数据与 split hash。

## 8. 结论分类

- **OPTIMIZATION_FAILURE**：positive synthetic 或短段过拟合失败，或梯度/更新链断裂。
- **GENERALIZATION_FAILURE**：训练 NLL 明显下降，但选择安全的 inner NLL 不改善。
- **CURRENT_MODEL_NONIDENTIFIABLE**：仪器和过拟合通过，多种合理配置均无稳定 unseen-inner 增量。
- **OPTIMIZER_SENSITIVE_SUPPORT**：仅少数窄配置阳性，跨 seeds/患者不稳；只作探索信号。
- **OPTIMIZATION_ROBUST_SUPPORT**：冻结公共配置在确认 seeds 和患者层复现 persistent、correct-time 及具体 H2a endpoint。

这些分类不是对生物学真假的总裁决，只限定当前模型、输入和 development 数据。

## 9. H3 边界

只有新配置产生至少 3 个 distinct、选择安全且稳定的 T1 checkpoints 时，才重跑该患者已冻结合同下最小 H3 单元。不得新增患者特异 N、source 或时间尺度。否则 H3 保持“前置状态不足，未决”。

## 10. 执行与交付

- 结果根：`results/epi_prssm/continuous_marked_state/r1/optimizer_identifiability_r1_6/`。
- 后台队列使用 OOM-safe、原子结果、可恢复 manifest；普通阴性继续运行。
- 最终交付白话版、技术版、机器审计、推荐默认优化配置以及对 R1.5/H3-long 旧结论的更正边界。
- 主工作区既有修改不纳入本 goal；仅在隔离 worktree 窄范围提交并推送。
