# Topic 5 RNN 全 cohort 间期学习与 Figure 3D 跨状态检验：分母修复合同 v0.1

> 状态：2026-08-11 锁定执行。本合同只修正 cohort、target 路由与图表，不根据 early-ictal
> 结果改动或重训模型。

## 1. 科学问题

检验两个相互分开的量：

1. 34 位 K=2 患者中，患者内自监督 recurrent model 是否能从间期 contact-rank sequences
   学到可在 held-out 事件中复现的传播结构；
2. 只由间期数据训练并冻结的患者特异 model field，是否与 Figure 3D 同一患者的发作早期能量场
   存在高于 all-contact channel shuffle 的空间对应。

这不是跨患者学习，也不是发作预测训练。统计单位始终是患者。

## 2. 间期 cohort

- source：`results/topic5_interictal_rank_distribution/dataset_v0_4`；
- denominator：34/34，均为 masked-rank 修复后的 K=2 患者；
- model：已冻结的 converged contact-space linear-state RNN，3 seeds；
- task：患者内 next-contact/STOP 自监督学习；
- field：从 held-out source-conditioned native rollout 中做 train-only K=2 read-back，生成两个
  model propagation fields；A/B 名称只按冻结间期模板 post-hoc 匹配，early-ictal target 不参与。

LBSS 21 人物理几何模型继续用于 connectivity motif 子分析，但不能替代 34 人间期 cohort。

## 3. Early-ictal 外部测试 cohort

唯一母清单逐位复用 Figure 3D：

- primary pooled phenotype-matched：17 人、167 seizures；
- strict broadband：16 人、106 seizures，clinical onset 后 0–10 s，1–150 Hz；
- gamma non-broadband：11 人、61 seizures，30–80 Hz；
- Epilepsiae 用 clinical onset；Yuquan Xuxinyi 保留 EEG onset。

所有合格 seizures 都进入；event→patient 先取中位数，再做 cohort inference。不得从旧的
history-RNN `outer_*` cache 重新定义分母。

## 4. 模型场与评分

每位患者、每个 seed：

1. 只用间期 train events 冻结两类传播 read-back；
2. 对 held-out native rollouts 分配 mode；
3. 计算每个 mode 的 contact participation support 与平均 normalized rank；
4. 三 seed 在 contact 层聚合；
5. 复用 Figure 3 的 patient plane、sigma、mirror 与 maxAB field engine，生成两个 model fields；
6. target 读取前写 `MODEL_FIELD_MANIFEST.json`，必须覆盖 17/17。

Early-ictal scorer 对每次 seizure 使用与 Figure 3 完全相同的 phenotype-matched activation、
all-contact permutation、mirror reselection、A/B max reselection 和 patient-first fold。

## 5. 主要量

### Interictal

- primary：native rollout transition correlation；
- comparison：同一患者 static-only generator；
- supporting：next-contact NLL、rank Wasserstein、participation error。

### Cross-state

- primary：RNN model-field maxAB `|r|` 与 synchronized all-contact shuffle null 的患者配对差；
- reference：冻结 empirical interictal field 的 Figure 3D 结果；
- 不把 RNN 与 empirical field 的差异作为模型选择依据。

## 6. 强制断言

```text
interictal_subjects == 34
model_seeds_per_subject == 3
primary_ictal_subjects == 17
primary_ictal_seizures == 167
model_field_subjects_before_target_read == 17
legacy_outer_cache_reads == 0
target_used_for_training_or_selection == false
```

任一断言失败即停止，不允许自动缩小分母。

## 7. 图

1. `topic5_rnn_e1146_field_transfer`：真实 E1146 SEEG tissue-plane layout 上并列 empirical
   TA/TB、RNN-generated 两个 mode 与患者内发作早期能量场；
2. `topic5_rnn_full_cohort_interictal_ictal`：左侧 34 人间期生成一致性；右侧 17 人、全部合格
   seizures 的冻结 RNN field vs channel-shuffle null。

图内只保留必要的 axis label、ticks、legend 与显著性星号；精确数值写入 CSV/JSON/README。
