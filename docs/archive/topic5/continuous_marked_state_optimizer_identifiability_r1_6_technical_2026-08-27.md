# Continuous Marked State R1.6：优化器、可训练性与可识别性技术报告

## 0. 验收结论

本阶段完成了优化器/训练充分性诊断，但没有完成队列级 H1–H3 验收。

核心结论有三条：

1. 旧 R1.5 的 epoch-0 选择不具备科学阴性解释力，因为 epoch 0 已见过 target-alignment 的选择尾段；旧 H3 的 `ZERO_GRADIENT` 是零 state readout 导致的结构性零导数。
2. 修复时间选择并扩大合理训练预算后，当前模型在 positive/reversed synthetic 和六位患者短段过拟合上可学习；因此不能再把所有 no-update 归为全局 optimization failure。
3. 冻结公共配置的六患者×五 seed development confirmation 中，只有 E384 达到跨 seed 的 persistent + correct-time 双重标准。随后限定于 E384 的最小 H3 没有通过完整控制，但独立 validation 单元只有 2 个，结论为“无支持、低效力、仍未决”。

正式 test 与 sealed partition 全程关闭；development validation 不参与配置或 epoch 选择。

## 1. 术语与判定锁定

| 术语 | 本报告中的固定含义 |
|---|---|
| prefix/core | R1.2 基础 timing+mark 模型及其基础训练阶段 |
| target alignment | observation correction、spatial fusion 与 state readout 面向同一 exact IED likelihood 的更新 |
| memoryless | 每个 observation anchor 从固定均值状态重新更新，不携带上一窗口状态 |
| persistent | 状态在 anchor 间传播，并由当前 observation correction 更新 |
| correct-time | 当前 anchor 的状态；与同 session、混杂匹配的错误时刻状态比较 |
| stable checkpoint | selected epoch > 0，且 development validation 上 persistent−memoryless joint NLL < 0、correct−matched-wrong joint NLL < 0 |
| seed | 优化稳定性重复，不是患者重复 |
| H3 full-control | real cumulative exposure 必须同时优于拟合截距、state-matched non-overlap、causal previous block、current-event-only 与 chronological trend，并满足独立 validation 单元数 |

## 2. R1.5 为什么不能区分训练失败与科学阴性

### 2.1 选择集被 epoch 0 预先见过

R1.5 从已在完整 TRAIN 上 refit 的 R1.2 checkpoint 出发，又以 TRAIN 尾部 20% 选择 target-alignment epoch。epoch 0 已在 refit 中使用过该尾段，而更新后的 checkpoint 必须在同一尾段证明自己更好。该比较系统性偏向 epoch 0。

R1.6 在每位患者 TRAIN 内冻结三个连续区间：

1. `base_train = 0–60%`；
2. `base_select = 60–80%`；
3. `alignment_select = 80–100%`。

prefix/core 只在 `base_train` 拟合、由 `base_select` 选预算，再按预算 refit TRAIN 前 80%；target alignment 只在 TRAIN 前 80% 拟合、由 `alignment_select` 选 epoch。冻结公共配置后才在完整 TRAIN refit，并在 development validation 上评分一次。

### 2.2 `ZERO_SELECTED` 不等于 `ZERO_GRADIENT`

旧张家齐 R1.5 选择 epoch 0，但 selection block 上梯度并非零：

- observation correction 最大梯度 `0.0613`；
- spatial fusion `0.0954`；
- state readout `0.6261`。

这属于 `ZERO_SELECTED`：参数能动、训练目标能降，但所用选择规则没有选择更新。

相反，旧张家齐 H3 的 T1 checkpoint 中 `state_timing.weight`、`state_contact.weight`、`state_size.weight` 的 norm 均为 0；H3 real edge 在零点的 matrix gradient 和 intercept gradient 也均为 `0.0`。这是 `ZERO_GRADIENT`：下游 loss 对边没有导数，任何 H3 learning rate 都无法修复。

## 3. 可学习性校准

### 3.1 Synthetic truth

共保存 12 个 synthetic packages。承担主要解释的四组如下：

| 设置 | 真值 | 恢复 | 未见数据胜基线 | correct-time 胜 wrong-time | 解释 |
|---|---|---:|---:|---:|---|
| n=600、LR 3e-3、80 epochs | positive | 5/5 | 5/5 | 5/5 | 模型族和优化器在充足信息下可恢复正信号 |
| LR 1e-2、80 epochs | positive | 4/5 | 5/5 | 4/5 | 更高 LR 可学，但时刻判别略不稳 |
| LR 1e-2、80 epochs | reversed | 5/5 | 5/5 | 5/5 | 可恢复相反方向，不依赖预设符号 |
| LR 1e-2、80 epochs | zero | 合同容差 5/5 | 1/5 | 1/5 | 3/5 仍选非零；必须靠 held-out 与 wrong-time 抑制伪更新 |
| 短 patience=10 | positive | 1/5 | 3/5 | 2/5 | 短 patience 会漏掉真信号 |

zero truth 的“合同容差 5/5”只表示 test 不显著恶化，不表示五个 seed 都正确选择零；实际有 3/5 选择了非零 epoch。因此 synthetic 证明“能学”，没有证明高 LR 的 checkpoint selection 完美无假阳性。

另有一条 positive seed 约到 epoch 105 才首次改善。这直接否定把短 patience 当作全局停止规则。

### 3.2 人体短段过拟合

固定六位患者，每位三个 seeds，在同一所选 prefix 和 target-alignment 配置下过拟合事前固定短连续片段。18/18 均降低 exact joint NLL。由此排除：

- 当前实现普遍断梯度；
- 当前模型完全没有容量拟合该 joint objective；
- 六位患者的阴性都由同一个全局 optimizer failure 解释。

过拟合通过不等于具有泛化状态，只把“连训练数据都学不会”的解释移出主桌面。

## 4. 公共优化器搜索

### 4.1 Prefix/core

正式 prefix tuning 为 4 患者×3 seeds×9 配置，加事前边界延伸，共 108/108 单元。探索和后续确认产物总数为 138，但不进入 108 的 tuning 分母。

排序第一为 `prefix_high_lr_e12_c128`：

```yaml
optimizer: AdamW
lr: 1.0e-3
weight_decay: 0
warmup_fraction: 0.10
gradient_clip: 5.0
epochs_or_full_time_passes: 12
anchor_chunk: 128
min_delta: 0
patience: disabled
```

张克轩在该配置上 3/3 seeds 改善，`base_select` 中位 improvement `0.06522`；8-pass 同设置为 `0.05613`，说明追加预算有实际收益。Adam 与无 decay AdamW 的表现近似，未发现 optimizer family 本身是决定因素。其他三位 tuning 患者的 prefix 后段仍主要选 epoch 0，提示其难点不只在 prefix LR。

### 4.2 Target alignment

alignment tuning 共 120 个预期单元：117 个有效结果和 3 个预期记录的 `NONFINITE_GRADIENT`。旧形态 `nested_current` 使用 chunk 8、clip 1、weight decay 1e-3，在 E1096 3/3 seeds 非有限，因此该配置判 inadmissible，而不是把失败单元静默删除。

冻结配置为 `nested_extended_budget`：

```yaml
optimizer: AdamW
state_and_readout_lr: 3.0e-4
observer_lr_ratio: 0.1
observer_lr: 3.0e-5
weight_decay: 0
warmup_fraction: 0.10
gradient_clip: 5.0
observer_passes: 8
joint_passes: 8
anchor_chunk: 32
min_delta: 0
patience: disabled
```

在 tuning seeds 0–2 中，该配置令张克轩 3/3、E384 2/3、张家齐 2/3 在选择安全的 TRAIN 尾段改善，E1096 0/3；按 patient-first 规则得到 3 个稳定 tuning 患者，优于其余 admissible 配置的 1 个。该排序没有读取 development validation。

## 5. 冻结配置的六患者×五 seed 确认

30/30 单元完成。seeds 0–2 是调参 seed 在配置冻结后的重新评分；seeds 3–4 是独立 optimizer confirmation。每位患者的 development validation 只在最终配置冻结后评分一次。

| 患者 | selected epoch >0 | persistent 与 correct-time 同时过 | 独立 seed 过 | 训练改善 | 短段过拟合 | 分类 |
|---|---:|---:|---:|---:|---:|---|
| E384 | 4/5 | 3/5 | 2/2 | 5/5 | 3/3 | `OPTIMIZATION_ROBUST_SUPPORT` |
| 程帅 | 1/5 | 1/5 | 0/2 | 5/5 | 3/3 | `OPTIMIZER_SENSITIVE_SUPPORT` |
| 陈子阳 | 5/5 | 1/5 | 0/2 | 5/5 | 3/3 | `OPTIMIZER_SENSITIVE_SUPPORT` |
| 张克轩 | 5/5 | 0/5 | 0/2 | 5/5 | 3/3 | `GENERALISATION_FAILURE_OR_CURRENT_MODEL_NONIDENTIFIABLE` |
| E1096 | 0/5 | 0/5 | 0/2 | 5/5 | 3/3 | `GENERALISATION_FAILURE_OR_CURRENT_MODEL_NONIDENTIFIABLE` |
| 张家齐 | 3/5 | 0/5 | 0/2 | 5/5 | 3/3 | `GENERALISATION_FAILURE_OR_CURRENT_MODEL_NONIDENTIFIABLE` |

### 5.1 E384 的 H1/H2a 分解

E384 的 stable seeds 为 1、3、4；两个独立 seeds 3、4 都通过。五 seed patient-first 中位数：

| contrast，负值有利 | joint | timing | mark | STOP | first subset | continuation |
|---|---:|---:|---:|---:|---:|---:|
| persistent − memoryless | −0.001678 | −0.002220 | 0.000000 | +0.013871 | −0.002038 | −0.002009 |
| correct − matched wrong | −0.008900 | −0.002568 | −0.005238 | −0.003585 | +0.001119 | −0.007684 |

persistent 相对 memoryless 的 joint 优势为 3/5 seeds，correct-time 优势为 4/5。该结果支持 E384 development 内的 time-specific persistent estimate，但 endpoint 并非单一方向：persistent 对 STOP 不利，而对 timing、first subset 和 continuation 有利；correct-time 对 continuation、STOP 和 timing 更一致，对 first subset 不稳。

### 5.2 张克轩的区分性反例

张克轩 correct−wrong joint 为 5/5 有利，中位 `−0.012781`；persistent−memoryless 为 0/5 有利，中位 `+0.025073`。该模式证明 correct-time test 与 persistence test 测量不同对象：前者可由当前窗口的 time-specific observation code 通过，后者要求跨窗口 carry 提供额外信息。

### 5.3 其余患者的边界

- E1096：5/5 训练改善、3/3 过拟合通过，但 5/5 选择 epoch 0；修复数值故障后仍不泛化。
- 张家齐：3/5 选择非零，但 persistent/correct-time 双标准 0/5。
- 程帅：虽然 patient-first contrasts 多数方向有利，只有 seed 1 实际选择非零并满足双标准；独立 seeds 均不通过。
- 陈子阳：5/5 选择非零，但双标准仅 1/5，效应接近数值地板。

这些结果不能转成 1/6、3/6 患者阳性率，因为本阶段是事前固定的 optimizer diagnosis panel，而非队列抽样或正式 patient-level inference。

## 6. 最小 H3 重跑

### 6.1 资格与设计

只有 E384 达到至少 3 个 stable T1 checkpoints，故只运行 seeds 1、3、4。每个 checkpoint 固定：

- scale：`N=1000 events`；
- exposure：`load` 和 `participation` 分别运行；
- endpoint：应用 exposure edge 后预测下一事件；
- controls：fitted intercept、state-matched non-overlap、causal previous block、current-event-only、chronological trend；
- T1 checkpoint、split、输入和 endpoint 全冻结，不新增患者特异尺度。

6/6 real edges 数值可估，但每个 seed 的共同支持只有 2 个独立 validation 单元，低于事前要求 3 个。

### 6.2 结果

`load` 的 3/3 seeds 在 next-event average 和 independent-block median 上均输给五个对照。next-event joint NLL contrasts 的范围为：

- real−intercept：`+0.0491` 到 `+0.0706`；
- real−state-matched：`+0.0411` 到 `+0.0625`；
- real−causal-delayed：`+0.0920` 到 `+0.0979`；
- real−current-event-only：`+0.0348` 到 `+0.0583`；
- real−trend：`+0.1266` 到 `+0.1291`。

`participation` 的 seed 1 在 next-event average 上胜 intercept、state-matched 和 current-event-only，但输 causal-delayed `+0.00304` 和 trend `+0.03242`；seeds 3、4 对所有控制均不利。三个 seed 的 independent-block medians 对五个控制全为正。

因此 `primary_full_control_increment = false` 为 6/6。安全判读为：E384、N=1000 的最小 H3 未支持 exposure edge；受限于单患者和 2 个独立 validation 单元，不构成长尺度 H3 生物学阴性。

## 7. 推荐默认配置与使用边界

机器可读配置位于：

`results/epi_prssm/continuous_marked_state/r1/optimizer_identifiability_r1_6/reports/recommended_optimizer_config.json`

建议后续将该配置作为当前 exact event model 的默认开发配置，而不是继续为每位患者搜索专属超参数。使用时必须保留：

1. TRAIN 内 chronological inner selection；
2. development validation 不参与配置或 epoch 选择；
3. persistent−memoryless 与 correct−matched-wrong 双测试；
4. 短段过拟合、梯度/更新量和 non-finite 记录；
5. H3 只接受非零、选择安全且稳定的 T1 checkpoint。

该配置是当前六患者 development panel 的经验默认值，不是全队列最优超参数，也不是正式 test 前可继续无限调参的许可。

## 8. 科学结论分层

### H1

- 工程层：当前架构可学习，optimization failure 不是所有阴性的共同解释。
- development 人体层：E384 为稳健支持；程帅/陈子阳为 optimizer-sensitive；张克轩只有 time-specific code 而无 persistence；E1096/张家齐为当前模型泛化失败或不可识别。
- 未建立：队列级 persistent state、raw-informed state、controlled/autonomous generative state、正式 test 复现。

### H2a

E384 显示状态对 timing、continuation 与部分 subset endpoint 有增量，但 STOP 与 first-subset 的方向依 contrast 不同。允许写为单患者 development predictive support，不允许写成统一的 recruitment-route mechanism。

### H2b

本轮未运行 seizure probe；既有探索性发作前结果不由本轮升级或否定。

### H3

旧 `ZERO_GRADIENT` 作废为生物学结果；新 E384 最小 H3 无支持但低效力。未检验更长的几千至上万次 IED 累积尺度。

## 9. 产物与复现

结果根：

`results/epi_prssm/continuous_marked_state/r1/optimizer_identifiability_r1_6/`

关键产物：

- `PREFIX_TUNING_STATUS.json`
- `ALIGNMENT_TUNING_STATUS.json`
- `CONFIRMATION_STATUS.json`
- `MINIMAL_H3_STATUS.json`
- `reports/prefix_tuning_summary.json`
- `reports/tuning_summary.json`
- `reports/optimizer_confirmation_summary.json`
- `reports/optimizer_confirmation_seed_rows.csv`
- `reports/recommended_optimizer_config.json`
- `reports/optimizer_identifiability_machine_audit.json`

供 git/远端长期审阅的逐位快照：

- `docs/archive/topic5/continuous_marked_state_optimizer_identifiability_r1_6_machine_audit_2026-08-27.json`，SHA256 `a99712fb6875157b0869d8d346e31344b26a3784e66a64f73b08f492e839115a`；
- `docs/archive/topic5/continuous_marked_state_optimizer_identifiability_r1_6_recommended_config_2026-08-27.json`，SHA256 `f66f1758deec641558ee48c20d4b28e8d7672c4ea6058842e23ed49bcc79fabc`。

最终机器审计的主计数：

- prefix tuning `108/108`；
- alignment `120/120`，其中有效结果 117、显式 non-finite failures 3；
- 所选配置短段过拟合 `18/18`；
- confirmation `30/30`；
- minimal H3 `6/6`；
- synthetic packages `12`。

复现 closeout：

```bash
cd /home/honglab/leijiaxin/HFOsp
PYTHONPATH=. /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/topic5_continuous_marked_state_r1/finalize_r1_6_optimizer_identifiability.py
```

机器审计为每个输入结果、顶层状态、关键源码、split manifest 和最终配置保存 SHA256；所有顶层状态均为 `COMPLETE`，`development_validation_used_for_configuration_selection=false`，`formal_test_partition_opened=false`，`sealed_opened=false`。
