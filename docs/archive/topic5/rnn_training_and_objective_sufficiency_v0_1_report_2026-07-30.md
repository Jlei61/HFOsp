# Topic 5 RNN 训练充分性与目标充分性审计 v0.1

日期：2026-07-30
冻结 spec：`docs/superpowers/specs/2026-07-30-topic5-rnn-training-sufficiency-v0_1.md`
上游：`docs/archive/topic5/rnn_stage_acceptance_and_training_sufficiency_2026-07-30.md` §7

状态：`ACCEPTED_TRAINING_SUFFICIENCY_CLOSED_GENERATION_BAR_STILL_NOT_MET`

## 0. 这一轮只问两件事

上一轮的结论是：冻结的 teacher-forced `LinearStateSequenceRNN` 能改善局部一阶转移
统计，但不能自由生成真实的完整间期传播事件。本轮**只**检验这个阴性的两个平凡解释：

1. **optimization sufficiency**：同一模型、同一 one-step loss，训练轮数与优化分辨率
   是否已经足够；
2. **objective sufficiency**：one-step teacher forcing 这个目标本身，是否足以支持
   free-running generation。

没有寻找"更强的 RNN"，没有重开 axis / path / low-rank architecture zoo，没有新增
contact-mixing 层或 path/axis 参数，没有读 IEI / 发作倒计时 / early-ictal target。

**用词边界**：本轮任何结论都只关于"这个最小 linear-state 模型在这个训练合同下能做
什么"，不能写成"RNN 证明/否证了癫痫机制"。

## 1. 数据与封存审计（只读，通过）

| 项 | 值 |
|---|---|
| 数据集 | `results/topic5_interictal_rank_distribution/dataset_v0_4` |
| 队列 | 34 人（Epilepsiae 18 / Yuquan 16） |
| 事件 | 864,163（train80 691,314 / heldout20 172,849） |
| 决策数 | train 5,448,027 / heldout 1,318,682 |
| 分层 | 逐患者 chronological 80/20，逐人验证时间单调且 split0 最大时刻 ≤ split1 最小时刻 |
| masked 来源 | `results/interictal_propagation_masked/per_subject`，非参与触点 `group_id = -1` |
| 封存 | 每人 `forbidden_inputs_present` 全 false（IEI / event_rate / time_to_seizure / seizure_seed / ictal_target / 字符串 ID） |
| 特征 | 8 维静态几何 + prefix participation；无 A/B、无物理轴、无 SOZ |

与既有合同一致，未触发停止条件。

## 2. Phase A：正式训练的真实优化语义

产物：`results/topic5_rnn_training_sufficiency_v0_1/input_audit/TRAINING_SEMANTICS_AUDIT.json`

审计 204 个既有正式 fold（其中 `linear_state` 102 个 = 34 人 × 3 seeds）：

| 量 | 值 |
|---|---|
| 每个 fold 的 outer training patients | 33 |
| coverage cycles | 1 |
| updates per patient | 8 |
| **每个 fold 的 shared optimizer step 数** | **264**（= 33 × 8） |
| 每个 fold 的 backward chunk 数 | 816（中位） |
| 每次 update 覆盖的事件数 | 中位 816，最大 14,034 |
| gradient clipping 触发比例（shared） | 中位 **0.277**，最大 0.439 |
| heldout local-offset 更新次数 | 32（4 cycles × 8） |
| heldout offset clipping 比例 | 0.000 |
| teacher-forced unroll rank steps | 中位 7，最大 52 |

### 2.1 `batch_size=1024` 是显存 chunk，不是 optimizer minibatch

三重证据，结论一致：

1. **静态代码结构**：`backward()` 位于 `batch_start` 分块循环内（循环深度 4），
   `optimizer.step()` 与 `optimizer.zero_grad()` 位于 segment 循环（深度 3）。
2. **逐 fold 计数**：102/102 个 `linear_state` fold 的 backward chunk 数（816）
   严格大于 optimizer step 数（264）。
3. **数值 parity**（Phase B3，见 §3.4）。

**归一化正确性**：`next_set_stop_loss` 返回 event-mean，分块按 `len(chunk)/len(segment)`
加权累积，因此加权和精确等于整段的 event-mean。这是 B3 parity 成立的原因。

### 2.2 Phase A 的判读

真正决定优化分辨率的是 `updates_per_patient` 与 coverage cycles，而不是 1024。
264 步、每步平均吃 816 个事件、其中 27.7% 被梯度裁剪——这是一个分辨率非常低的
优化过程，先验上就不该假定它已经收敛。

## 3. Phase B：纯收敛审计

Development 结构：全部 34 人的 train80 再切成时间序的前 90%（inner training，
622,182 事件）与后 10%（inner validation，69,132 事件）；outer heldout20 全程封存
（`event_split` 标为 2）。

Primary endpoint：validation **contact-choice NLL**（只在非终止决策上平均，
单位 nats/decision），先在患者内合并 seeds，再取患者中位数。

### 3.1 B1 + B1x：训练预算

seeds {20260725, 20260726, 20260727}，learning rate 1e-3，AdamW，weight decay 1e-4。
coverage cycles 是嵌套读出：一次训练到 8 轮，每轮末尾评估一次。
**嵌套读出的精确性已验证**：8 轮运行的第 1–4 轮与独立 4 轮运行在 816/816 个
（患者 × 轮次 × seed）行上逐位相同（`development/reproducibility/NESTED_CYCLE_READOUT.json`）。

hidden 32、每人 32 次更新的主线（患者中位 nats/decision）：

| 完整过数据的遍数 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| validation contact-choice NLL | 1.782 | 1.807 | 1.734 | 1.724 | 1.719 | 1.718 | **1.707** | 1.706 |

每人 8 次更新（即已发表的更新分辨率）在第 1 轮为 **1.852**。

**承重对比：已发表训练预算 vs 收敛预算**

| 量 | 值 |
|---|---|
| 已发表（1 遍 × 每人 8 次更新），患者中位 | 1.852 |
| 收敛（7 遍 × 每人 32 次更新），患者中位 | 1.707 |
| **配对差（患者中位）** | **0.124 nats/decision** |
| bootstrap 95% CI | [0.081, 0.166] |
| 收敛更优的患者数 | **33/34** |
| Wilcoxon P | 3.5 × 10⁻¹⁰ |
| 与既有承重效应（ordered vs unordered = 0.0257）之比 | **4.8×** |

### 3.2 收敛判据未满足

预注册判据：连续两遍的患者中位改善都 < 0.002 nats/decision。

hidden 32 / 每人 32 次更新的逐遍改善：
0.035、0.044、−0.001、0.009、0.011、0.013、−0.005。

改善不是平滑衰减，而是在 −0.005 到 +0.044 之间抖动；**跑到预注册上限 8 遍仍未出现
连续两遍落入判据带**。因此本轮只能写"比已发表配置接近收敛得多"，不能写"已收敛"。

按预注册的配对一倍标准误 + 最省预算规则，冻结在 **7 遍 × 每人 32 次更新 × hidden 32**
（相对 8 遍的配对差 −0.0006 ± 0.0039，二者不可区分且更省）。

### 3.3 容量、学习率与优化器敏感性

hidden 64 在所有遍数上都不优于 hidden 32（第 7 遍配对差 +0.007 ± 0.004，方向为 64 更差），
因此**容量不是瓶颈**。

B2 在冻结预算（7 遍 × 每人 32 次更新 × hidden 32）的最终轮上比较学习率与优化器，
3 seeds，选择限制在最终轮且 Adam 只作敏感性臂（不参与选择）：

| 配置 | 患者中位 NLL | 相对最优的配对差 ± SE |
|---|---:|---:|
| **lr 3e-4, AdamW, wd 0（选中）** | **1.6911** | — |
| lr 3e-4, AdamW, wd 1e-4 | 1.6911 | 0.00001 ± 0.00000 |
| lr 1e-3, AdamW, wd 0 | 1.7071 | 0.0197 ± 0.0036 |
| lr 1e-3, AdamW, wd 1e-4（**已发表所用**） | 1.7071 | 0.0197 ± 0.0036 |
| lr 3e-3, AdamW, wd 0 | 1.7082 | 0.0285 ± 0.0042 |
| lr 3e-3, AdamW, wd 1e-4 | 1.7086 | 0.0301 ± 0.0043 |

Adam 敏感性臂（不参与选择）：lr 3e-4 → 1.6911，lr 1e-3 → 1.7071，lr 3e-3 → 1.7082。

**判读**：

- **学习率有真实影响**：3e-4 比已发表所用的 1e-3 好 0.0197 nats/decision（配对，SE 0.0036）。
- **优化器家族和权重衰减都不重要**：Adam 与 AdamW 在 weight decay 0 时数值到小数点后
  5 位一致（这是实现正确性的旁证，因为 AdamW 在 wd=0 时就是 Adam）；wd 从 0 改到 1e-4
  只改变第 5 位小数。因此"没有比较过 optimizer 家族"这一未关闭项现在关闭了，答案是
  **它不是限制因素**。
- **选中的学习率位于预注册网格的下边缘**。按 spec 不做网格外扩展，这是本轮的明确限制。
  方向上它是**保守的**：真实最优学习率若更小，只会让"已发表实现训练不足"的差距更大。

### 3.4 B3：分块等价性（工程 parity）

同一 seed、冻结预算（7 遍 × 每人 32 次更新 × lr 3e-4 × AdamW wd 0），只改显存 chunk：

| 量 | chunk 1024 | chunk 512 |
|---|---:|---:|
| optimizer 更新次数 | 7,616 | **7,616**（逐字相同） |
| backward 调用次数 | 10,080 | **13,664**（多 35.6%） |
| 患者中位 validation contact-choice NLL | 1.69106758 | 1.69106740 |

- 最大绝对参数差：**5.22 × 10⁻⁷**
- 最大绝对 local-offset 差：**6.33 × 10⁻⁷**
- validation NLL 差：**1.79 × 10⁻⁷**
- 预注册容差：10⁻⁴ → **PASS**

差异只有 float32 加法非结合性的量级。**这条与 §2.1 的静态结构证据和逐 fold 计数
共同闭合了 Phase A 的核心命题：`batch_size` 是显存分块，不是 optimizer minibatch。**

适用范围声明：parity 只对 `teacher_forced_one_step` 成立。self-fed 目标在采样时消耗
随机数，chunk 边界会改变随机数消耗序列——这是设计使然，不是缺陷。

### 3.5 训练–验证差距

留出代价在所有配置、所有遍数上都**低于**训练代价（患者中位 −0.010 至 −0.023
nats/decision），即没有过拟合迹象；§3.1 的改善是真泛化，不是记忆训练事件。

### 3.6 B1c：LOSO 结构确认与超参冻结

§3.1–§3.5 的筛选用的是廉价的非-LOSO development（单一共享模型 + 每位患者自己的
inner-validation）。B1c 把结论放回**与正式协议结构一致**的 LOSO 上复核：33 人训
shared core → 第 34 人在自己的 inner-train 上校准 local offset（core 冻结）→ 在第 34 人的
inner-validation 上评估。三个臂 × 3 seeds × 34 folds = **306 单元，零失败**；同一次校准
运行在第 4 与第 8 轮各取一次快照，因此两个 offset 预算不额外花训练成本。

| 臂（患者中位 contact-choice NLL） | 值 |
|---|---:|
| 已发表预算：1 遍 × 每人 8 次更新，offset 4 轮 | 1.86783 |
| 已发表预算，offset 8 轮 | 1.84499 |
| 4 遍 × 32 次更新，offset 4 轮 | 1.71625 |
| 4 遍 × 32 次更新，offset 8 轮 | 1.70689 |
| 7 遍 × 32 次更新，offset 4 轮 | 1.70061 |
| **7 遍 × 32 次更新，offset 8 轮（选中）** | **1.69211** |

逐患者配对（n=34）：

| 对比 | 中位增益 | bootstrap 95% CI | 改善患者 | Wilcoxon P |
|---|---:|---:|---:|---:|
| 选中 vs **已发表预算** | **+0.13404** | [+0.0862, +0.1991] | **34/34** | 1.16 × 10⁻¹⁰ |
| 选中 vs 4 遍（同 offset） | +0.00848 | [+0.0042, +0.0165] | 30/34 | 2.37 × 10⁻⁷ |
| offset 8 vs offset 4（同 shared 预算） | +0.00653 | [+0.0036, +0.0125] | 30/34 | 1.01 × 10⁻⁶ |

**判读**：

1. 非-LOSO 筛选给出的已发表-vs-收敛差距是 0.124，LOSO 结构下是 **0.134**——两段式
   development 设计的代理误差很小，方向与量级都稳。34/34 与 P=1.16×10⁻¹⁰ 是 n=34
   Wilcoxon 的下限，即**所有患者一致**。
2. 7 遍在 LOSO 下仍优于 4 遍（30/34），所以选中的预算不是在过拟合那 33 位训练患者。
3. heldout local-offset 校准从 4 轮加到 8 轮也有独立收益（30/34），但量级只有 shared
   预算效应的 1/20——**shared model 的训练预算才是主导项**，与 spec 的优先级一致。

**冻结配置**（`development/HYPERPARAMETER_FREEZE.json`）：

| 项 | 值 |
|---|---|
| 模型 | `LinearStateSequenceRNN`，hidden 32，local offset dim 4 |
| shared coverage cycles | **7** |
| updates per patient | **32** |
| heldout offset calibration cycles | **8** |
| learning rate / optimizer / weight decay | **3e-4 / AdamW / 0** |
| gradient clip / 显存分块 | 1.0 / 1024 |

封存审计：**354 个已完成单元逐个检查，全部 `ictal_target_read=false` 且
`outer_heldout_read=false`**。选择过程读取的数据范围只有 train80 的时间序前 90%
（inner training）与后 10%（inner validation）。

## 4. Phase C：目标充分性

冻结预算下只比较四个训练目标（模型参数量、训练事件、患者权重、优化器预算全部相同，
只有喂进递归状态的那一步历史不同；监督目标、候选集合与分母逐字一致）。
LOSO-development 结构，4 目标 × 3 seeds × 34 folds = **408 单元，零失败**。

### 4.1 三个 rollout-aware 目标全部劣于 teacher forcing

相对 `teacher_forced_one_step` 的逐患者配对（n=34，constructive 发生器）：

| 目标 | 相邻步顺序 r | 一步预测 NLL（护栏） |
|---|---:|---:|
| `scheduled_sampling` | −0.0399（2/34，P=1.3×10⁻⁶） | −0.0130（2/34，P=1.3×10⁻⁸） |
| `self_fed_2step` | −0.0404（2/34，P=4.3×10⁻⁷） | −0.0213（4/34，P=3.6×10⁻⁸） |
| `self_fed_3step` | −0.0720（3/34，P=7.5×10⁻⁸） | −0.0333（1/34，P=2.3×10⁻¹⁰） |

整场事件端点（名次分布、成对先后、参与、事件长度）没有任何一项改善。
模型自身发生器下结论同向。

**单调剂量反应**：自喂越多越差（3 步 > 2 步 > 渐增 schedule）。这不是噪声，是
"用模型自己的输出当历史会污染训练信号"的直接证据。

**预注册护栏全部失守**：`any_objective_passed_the_one_step_guard = false`。按预注册
回退规则，冻结时仍选出退化最小的 `scheduled_sampling` 带入 Phase D，以便在外层留出
数据上给 rollout-aware 路线一次公平的正式机会。

### 4.2 发现：上一轮用来评价生成的复合发生器本身有问题

这是本轮 spec §5.4 新增的次要诊断给出的结果。同一训练条件内、同一批冻结随机数配对，
比较**模型自身的联合分布**与**上一轮的复合发生器**（静态骨架 + 冻结顺序残差 + 经验终止 hazard）：

| 端点（teacher-forced 条件） | 模型自身 − 复合 | 改善患者 | P |
|---|---:|---:|---:|
| 相邻步顺序 r | +0.0446 | 29/34 | 1.7×10⁻⁷ |
| 名次分布 W1 | +0.0375 | 33/34 | 6.4×10⁻⁹ |
| **成对先后 r** | **+0.5287** | 29/34 | 4.3×10⁻⁸ |
| 参与误差 | +0.0426 | 31/34 | 8.2×10⁻⁹ |

四个训练条件下结论一致（28–33/34，P ≤ 5.8×10⁻⁷）。


## 5. Phase D：正式冻结确认（外层留出 20%，只读一次）

34 人 × 3 seeds × 3 个训练条件 = **306 单元，零失败**。四个条件共享同一个真实首触点、
同一批冻结的均匀随机数、同样的 rollout 次数、同样的结束规则与同样的候选集合。

**参照条件就是已发表的那个模型本身**：直接加载归档 checkpoint，不重训数学等价的复制品。
102/102 个参照单元复算出的留出 NLL 与归档值一致，最大偏差 3.0×10⁻⁷、中位 6.2×10⁻⁸。

**工程校验**：`static_only` rollout 在三个训练条件下逐位相同（所有端点差值恰为 0），
证明跨条件配对随机数精确对齐。

### 5.1 结论完全取决于用哪台发生器把模型读出来

绝对水平（患者中位，外层留出）：

| 训练条件 | 发生器 | 相邻步 r ↑ | 名次 W1 ↓ | **成对先后 r ↑** | 参与误差 ↓ | 长度 W1 ↓ | 终止误差 ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| （任一） | 只用静态骨架 | 0.445 | 0.0916 | 0.184 | 0.0834 | 0.0167 | 0.0300 |
| 已发表 | 复合 | 0.635 | 0.0943 | 0.201 | 0.0850 | 0.0167 | 0.0300 |
| 收敛 | 复合 | 0.815 | 0.1084 | **0.014** | 0.1255 | 0.0167 | 0.0300 |
| rollout-aware | 复合 | 0.802 | 0.1061 | 0.067 | 0.1283 | 0.0167 | 0.0300 |
| 已发表 | 模型自身 | 0.621 | 0.0874 | 0.459 | 0.0991 | 0.0874 | 0.0819 |
| **收敛** | **模型自身** | **0.867** | **0.0497** | **0.804** | **0.0681** | 0.0477 | 0.0714 |
| rollout-aware | 模型自身 | 0.876 | 0.0575 | 0.807 | 0.0772 | 0.0533 | 0.0797 |

**复合发生器下：模型训练得越好，整场事件反而越差。** 成对先后相关从 0.184（完全不用
历史）→ 0.201（已发表）→ **0.014**（收敛）；参与误差从 0.0834 → 0.0850 → **0.1255**。
收敛模型经复合发生器读出后，整场事件的成对顺序与真实数据**几乎不相关**。

**模型自身发生器下：全部单调改善。**

### 5.2 收敛 vs 已发表（模型自身发生器，逐患者配对，n=34）

| 端点 | 中位增益 | bootstrap 95% CI | 改善患者 | rank-biserial | Wilcoxon P | Epilepsiae | Yuquan |
|---|---:|---:|---:|---:|---:|---:|---:|
| 相邻步顺序 r | **+0.1880** | [+0.112, +0.264] | **33/34** | +0.993 | 3.5×10⁻¹⁰ | 17/18 | 16/16 |
| 名次分布 W1 | +0.0202 | [+0.012, +0.040] | 31/34 | +0.933 | 4.3×10⁻⁸ | 17/18 | 14/16 |
| 成对先后 r | **+0.2494** | [+0.140, +0.380] | 30/34 | +0.946 | 2.0×10⁻⁸ | 17/18 | 13/16 |
| 参与误差 | +0.0199 | [+0.013, +0.035] | 32/34 | +0.943 | 2.4×10⁻⁸ | 17/18 | 15/16 |
| 事件长度 W1 | +0.0402 | [+0.027, +0.046] | 32/34 | +0.956 | 1.0×10⁻⁸ | 16/18 | 16/16 |
| 一步预测 NLL | **+0.1389** | [+0.094, +0.185] | **34/34** | +1.000 | 1.2×10⁻¹⁰ | 18/18 | 16/16 |

**两个队列方向完全一致，没有任何一项由单一数据集驱动。**

### 5.3 rollout-aware 在正式数据上也没有优势

相对收敛 teacher-forced（模型自身发生器）：相邻步顺序 −0.0172（9/34，P=8.4×10⁻⁴）、
一步预测 −0.0193（4/34，P=3.0×10⁻⁸），其余端点无显著差异。Phase C 的结论在外层留出
数据上原样成立。

### 5.4 用上一轮自己的预注册验收标准重新评分

判据（上一轮原文）：一位患者若在**参与误差、名次分布、成对先后误差**三项中至少两项落入
"真实留出事件前后两半互比"误差的 **+10%** 以内，即算通过；队列门槛 **17/34**。

本轮从冻结数据重算了那条经验基准，与归档值一致到 **9.7×10⁻¹⁷**（float64 舍入量级）。

| 训练条件 | 发生器 | 通过患者数 | 门槛 | 达标 |
|---|---|---:|---:|---|
| 已发表 | 复合 | **9/34** ← 精确复现上一轮 | 17 | 否 |
| 收敛 | 复合 | 7/34 | 17 | 否 |
| rollout-aware | 复合 | 7/34 | 17 | 否 |
| （任一） | 只用静态骨架 | 10/34 | 17 | 否 |
| 已发表 | 模型自身 | 10/34 | 17 | 否 |
| **收敛** | **模型自身** | **13/34** | 17 | **否** |
| **rollout-aware** | **模型自身** | **14/34** | 17 | **否** |

**没有任何条件达到预注册门槛。** 但两个事实必须同时记录：

1. 上一轮的 9/34 **低于完全不用历史的静态对照 10/34**——那个数字里有一部分是读出方式的
   人为损失，不是模型的性质。
2. 用正确的读出方式加收敛训练，从 9–10/34 提高到 13–14/34，是真实进步，但**仍低于 17/34**。


## 6. Go / no-go 判读

预注册的四个结果与实际观测的对应：

| 预注册结果 | 判定 |
|---|---|
| **结果 1**：更多训练只改善一步预测，不改善自由生成 | **仅在上一轮的复合读出方式下成立**；用模型自身的读出方式，自由生成的每一个端点都显著改善 |
| **结果 2**：rollout-aware 改善完整事件且不伤一步预测 | **否决**。三个目标全部同时伤害两者，且呈单调剂量反应 |
| **结果 3**：增加覆盖轮数后一步预测与自由生成都显著改善 | **成立**（模型自身读出方式；6 个端点全部显著，两队列同向） |
| **结果 4**：所有条件仍只改善局部转移 | **否**——整场事件的名次分布、成对先后、参与、长度全部改善 |

### 一句话科学结论

> 上一版"完整事件生成阴性"里，有两个可分离的成分：一部分来自**训练不足**（已发表配置
> 比收敛配置差 0.134 nats/decision，34/34 患者），另一部分来自**读出方式**（用于评价的
> 静态骨架＋顺序残差复合发生器，会随模型变好而系统性变差）。两者都修正之后，自由生成
> 的每一项整场指标都显著改善，但**仍未达到"与真实数据自身前后半一样接近"的预注册门槛
> （13–14/34，门槛 17/34）**。曝光偏差不是原阴性的原因：三种 rollout-aware 目标同时
> 损害局部预测与整场生成。

### 训练充分性是否关闭

**关闭。** 一步预测方向上：已发表配置差 0.134 nats/decision（34/34 患者，
P=1.16×10⁻¹⁰，LOSO 结构确认）。容量（hidden 64）不是限制、优化器家族与权重衰减不是
限制、显存分块不影响优化边界（数值等价到 5×10⁻⁷）。学习率仍是限制之一，且选中值位于
预注册网格下边缘——方向保守。

收敛判据本身**未满足**：跑到预注册上限 8 遍仍未出现连续两遍改善小于 0.002 的情形。
因此允许写"比已发表配置接近收敛得多"，**不允许写"已收敛"**。

### 明确不能写的

- 不能写"RNN 证明/否证了癫痫机制"；
- 不能写"RNN 自由生成了真实的完整双向传播事件"——预注册门槛未达到；
- 不能写"已收敛"；
- 不能把 §4.2/§5.1 的读出方式问题写成新的科学发现，它是工程构造的性质。


## 7. 对论文当前 RNN 措辞的修订建议

### 7.1 由 Phase A / B 直接触发（与 C/D 结果无关，必须改）

**R1｜任何"当前模型不足以生成完整事件"的句子都必须同时给出训练预算。**

- 触发位置：
  `rnn_stage_acceptance_and_training_sufficiency_2026-07-30.md` §1、§3.3、§6 第 4 条；
  `constructive_event_generation_sufficiency_v0_1_report_2026-07-30.md` §5「不能写」第 1 条与
  §6；`docs/paper-draft/figure6_persistent_path_mode_rnn_bounded_negative.md`
  「the present teacher-forced, event-persistent path-mode architecture is not a
  sufficient generative bridge」。
- 理由：那些句子背后的模型每个 fold 只有 264 次共享参数更新、只完整过一遍训练事件，
  在留出一步预测上比收敛配置差 0.124 nats/decision（34 人中 33 人，p=3.5×10⁻¹⁰），
  约为该论文顺序增量（0.0257）的 4.8 倍。
- 修订形式：把"当前模型/当前架构不足"改成"**在该训练预算下**的这一实现不足"，并在
  Methods 或图注写明预算（1 遍覆盖、每位患者 8 次更新、每 fold 264 次共享更新）。

**R2｜超参数来源必须写清楚不是同一个模型选的。**

- 触发位置：`rnn_stage_acceptance_and_training_sufficiency_2026-07-30.md` §4.1。
- 事实：那 8 格 target-blind tuning 用的是 `FullHistorySequenceGRU` + step-based
  `train_shared`（512 步 × 128 事件），而最终获胜的 `LinearStateSequenceRNN` 用的是
  coverage-based 训练，二者既不是同一模型也不是同一优化语义。
- 修订形式：明确写"hyperparameters were selected on a gated recurrent surrogate under
  a different optimizer schedule and transferred to the linear-state model"，并补上本轮
  在 linear-state 上的直接扫描结果（见 §3.3）。

**R3｜`batch_size=1024` 的性质从推断升级为已验证。**

- 触发位置：同上 §4.2 第 3 条（原文用"因此…不是普通意义上的有效 minibatch size"）。
- 修订形式：改为已验证陈述，并给三条证据（静态循环结构；102/102 fold 的 816 次 backward
  vs 264 次参数更新；数值 parity 实验）。

**R4｜"尚未比较 coverage cycles"改为"已比较到 8 遍且仍未达到预注册收敛判据"。**

- 触发位置：同上 §4.2 第 2 条。
- 措辞纪律：**不得写"已收敛"**。允许写"marginal per-pass gains fell from 0.034/0.044
  early to about 0.01, and the two best budgets were statistically indistinguishable,
  but the preregistered plateau criterion was not met within the preregistered
  eight-pass ceiling"。

**R5｜容量不是瓶颈这一点可以正面写。**

- 事实：hidden 64 在 1–8 遍的每一档都不优于 hidden 32。
- 修订形式：把原来的"参数量匹配的 h64 sensitivity 保留了主要 one-step 结果"升级为
  "increasing the hidden size did not improve held-out prediction at any training
  budget, so model capacity was not the limiting factor"。

**R6｜学习率同样不是已经调到位的。**

- 事实：在收敛预算上，`3e-4` 优于已发表所用的 `1e-3`；选中的学习率位于预注册网格的边缘。
- 修订形式：报告这一点，并说明它的方向是**保守的**——它只会让"已发表实现训练不足"
  这一判断更强，不会削弱它。同时声明网格边缘是本轮的限制，未做网格外扩展。

### 7.2 由 Phase C / D 触发

**R7｜"完整事件生成阴性"必须拆成两个可分离的成分。**

- 触发位置：`constructive_event_generation_sufficiency_v0_1_report_2026-07-30.md` §4.2、
  §5「不能写」第 1 条、§6；`rnn_stage_acceptance_and_training_sufficiency_2026-07-30.md`
  §3.3；`docs/paper-draft/figure6_persistent_path_mode_rnn_bounded_negative.md` 摘要与
  「Claim boundary」。
- 事实：其中一部分来自训练不足（见 R1），另一部分来自**评价用的复合发生器**——它把
  模型对事件起点的偏好当作参照减掉，再加回训练集的静态参与先验；模型越准，这个组合
  注入的系统性偏差越大。外层留出数据上，收敛模型经复合发生器读出后成对先后相关塌到
  **0.014**（不用历史的静态对照是 0.184），参与误差反而升到 0.1255（静态对照 0.0834）。
- 修订形式：凡是引用那个阴性的地方，都必须写明它是"在该读出方式与该训练预算下"的结论；
  并报告用模型自身联合分布读出时六个整场端点全部显著改善（30–33/34，两队列同向）。

**R8｜整场生成的预注册门槛仍未达到，这一句要保留但要更新数字。**

- 事实：上一轮 9/34（门槛 17/34）。本轮精确复现 9/34；用正确读出方式加收敛训练是
  13–14/34，仍低于 17/34；完全不用历史的静态对照是 10/34。
- 修订形式：保留"未达到"的结论，但把 9/34 改写为"9/34（该读出方式）／13–14/34（模型
  自身读出方式，收敛训练）"，并点明静态对照是 10/34——否则读者会把 9/34 误读成
  "历史信息毫无价值"。

**R9｜曝光偏差可以正式排除。**

- 事实：三个 rollout-aware 目标（每 2 步自喂、每 3 步自喂、渐增 schedule）在
  development 与外层留出上都同时**损害**局部预测与整场生成，且呈单调剂量反应。
- 修订形式：可以写"we tested and excluded exposure bias as an explanation"，并给出
  一步预测护栏全部失守的数字。这是一个**阳性的排除**，不是未做的检查。

**R10｜禁止的新写法。**

- 不得写"RNN 自由生成了真实的完整双向传播事件"——预注册门槛未达到。
- 不得写"模型已收敛"——8 遍仍未满足连续两遍改善 < 0.002 的判据。
- 不得把 R7 的读出方式问题包装成新的科学发现；它是工程构造的性质，属于方法学更正。
- 不得写"RNN 证明/否证了癫痫机制"。


## 8. 工程与复现

### 8.1 并发缩放（实测，供后续同类审计参考）

模型只有 5,870 个参数，单元的算子都很小，因此**串行点是 GPU 的核函数队列，不是显存也
不是 CPU**。每个 worker 进程都显示 ~100% CPU，但其中大部分是 CUDA 同步的自旋等待。

| worker 数 | 每单元耗时 | 吞吐（7 遍等效单元/小时） |
|---:|---:|---:|
| 6 | ~950 s | 23.2 |
| 14 | ≥1220 s（未跑完即停，故为上界） | ≤41.3 |
| 24 | 2194 s（中位，极差 2127–2226） | 39.4 |

结论：**6 → 14 有约 1.7 倍增益；14 → 24 没有增益**，GPU 队列在 ~14 并发处饱和。
单元峰值资源：GPU allocated ≤ 201 MB、reserved ≈ 460 MB（含 CUDA context）、RSS ≤ 1.36 GB。
因此 OOM 从来不是本轮的风险；防护仍按 `workers × per_process_fraction < 0.95` 的硬闸设置。

未采用的两个提速方案及理由：

- **CPU/GPU 混合 worker**：CPU 与 GPU 的数值结果不同，若同一患者在不同条件下落到不同
  设备，配对比较会混入设备伪差，违反同分母合同。
- **CUDA MPS**：可能有效，但它是全卡级设置、会影响同机其他使用者，未擅自开启。

### 8.2 复现与断点续跑

- **断点续跑**：每个单元完成后写 `DONE.json`；重跑同一 manifest 只跳过已完成单元。
  存在但未完成的目录会**响亮地阻塞**恢复（`BLOCKED_PARTIAL_CELLS`），不会被静默覆盖。
  本轮实际发生过一次会话中断与一次人为重启（并发从 14 调到 24），两次都靠这套机制
  无损恢复；重启前清理了 14 个未完成单元目录，没有任何已完成结果丢失。
- **逐位复现**：同一 seed 重跑一个已完成单元，4 个 coverage cycle 的验证 NLL 逐位相同
  （`development/reproducibility/REPRODUCIBILITY.json`）。
- **嵌套读出精确**：8 轮运行的第 1–4 轮与独立 4 轮运行在 816/816 行上逐位相同
  （`development/reproducibility/NESTED_CYCLE_READOUT.json`）。
- **已发表模型复现**：102/102 个参照单元从归档 checkpoint 复算的留出 NLL 与归档值最大
  偏差 3.0×10⁻⁷、中位 6.2×10⁻⁸。
- **配对随机数对齐**：`static_only` rollout 在所有训练条件下逐位相同（差值恰为 0）。
- **失败 / OOM / NaN**：**全程 0**。1,062 个训练/评估单元，无一失败、无一 OOM、无 NaN。
  峰值资源：GPU allocated ≤ 0.204 GB、RSS ≤ 1.45 GB（单进程）。
- **测试**：新增 24 项定向测试全部通过；Topic 5 RNN 相关 20 个测试文件共 **172 项全部通过**。


## 9. 产物清单

### 9.1 冻结与验收

- `results/topic5_rnn_training_sufficiency_v0_1/FINAL_ACCEPTANCE.json` — 状态 `ACCEPTED`
- `.../input_audit/TRAINING_SEMANTICS_AUDIT.json` — Phase A 优化语义审计
- `.../development/HYPERPARAMETER_FREEZE.json` — 训练预算冻结（含选择所读数据范围）
- `.../development/OBJECTIVE_FREEZE.json` — 训练目标冻结
- `.../development/reproducibility/{REPRODUCIBILITY,NESTED_CYCLE_READOUT}.json`

### 9.2 统计

- `.../analysis/b1_selection.json`、`b1x_selection.json`、`b2_selection.json` — 逐阶段选择
- `.../analysis/b3_chunk_parity.json` — 分块等价
- `.../analysis/b1c_paired_tests.json` — LOSO 结构确认
- `.../analysis/c_paired_tests.json`、`d_paired_tests.json` — 目标与正式确认的全部配对检验
- `.../analysis/d_empirical_variability_rescore.{json,csv}`、`d_empirical_variability_summary.csv`
  — 用上一轮预注册验收标准的重新评分
- `.../analysis/empirical_variability_reference.csv` — 本轮重算的经验基准（与归档一致到 9.7×10⁻¹⁷）

### 9.3 图（说明见 `figures/README.md`）

- `topic5_rnn_training_sufficiency_convergence` — 收敛审计四联
- `topic5_rnn_objective_sufficiency_generation{,_native}` — Phase C 两种发生器
- `topic5_rnn_formal_sufficiency_generation{,_native}` — Phase D 两种发生器
- `topic5_rnn_{objective,formal}_sufficiency_cohort` — 两队列分层
- `topic5_rnn_representative_patient_formal{,_native}` — 代表患者观测 vs 生成

### 9.4 可复现 manifest

`.../development/manifests/{b1,b1x,b2,b3,b1c,c,d}.json` — 每个阶段的完整单元清单
（逐单元参数、seed、路径）。重跑命令：
`bash scripts/run_topic5_training_sufficiency_pipeline_v0_1.sh <stage> <workers>`。

