# Topic 5 可训练 event-level RNN v2.5.4：五轮正式验收

**日期：** 2026-08-02  
**主结果目录：** `results/topic5_stable_repertoire_event_rnn/v2_5_4/`  
**验收状态：** `COMPLETE_NARROW_WINDOW_RESET_RESULT`  
**后续修订：** v2.5.4 只验收固定窗口、每个样本重置状态的 residual RNN；它不是连续 stateful event-sequence RNN 的终局判决。核心问题转入 v2.6。

# 审阅结论

## 1. 一句话判断

这次已经真正训练并调过一个以完整事件为时间步的 RNN，也完成了 34/34 患者、三种历史长度和两类 coherent chronology null；结果不支持队列层面的 recurrent increment，但这个阴性只约束当前已充分校准的 event-level RNN family，不否定此前已经成立的患者内稳定传播 repertoire。

## 2. 完成程度

> **完成度：97/100**

已完成：

- 34/34 患者主分析，0 个运行失败；
- 六位 development 患者调参，剩余 28 位 locked extension 作主统计；
- RNN、GRU、LSTM、hidden size、层数、event step、normalization、LayerNorm、optimizer、learning rate、batch size、weight decay 和 gradient clipping 的有界搜索；
- 3 个冻结 seed；
- source-disjoint chronology split、旧 heldout20 排除、正式 target 不重叠；
- source-coherent block shuffle 和 safe circular pairing；
- `L=20/40/80` 历史长度；
- synthetic order calibration、33 个相关测试和 102 个真实 checkpoint strict reload。

扣分来自三个工程或外部确认缺口：两个低支持患者无法构造 safe circular pairing；checkpoint 未把拟合后的 ridge baseline 封装为单文件；六个探索性阳性患者没有独立数据确认。

## 3. P0 / P1 关键问题

### P0

没有剩余 P0。主数据合同、训练、冻结哈希、null provenance 和 34 人分母均通过。

### P1-1：checkpoint 不是完全独立的单文件包

102 个 RNN `state_dict` 均能 strict reload，单元测试也验证了 reload 前后 prediction parity。但 checkpoint 只记录 baseline 的类型和参数，没有保存拟合后的 ridge coefficients 和 template encoder。当前结果可以依靠冻结代码、输入 SHA 和患者预测 artifact 重放；若未来要把模型交给外部使用，应补一个完整 inference bundle。

### P1-2：safe circular 对两位低支持患者不可估计

`yuquan_huanghanwen` 和 `yuquan_songzishuo` 找不到同时满足不重叠和最小间隔的 donor target，因此 circular 比较是 26 位 extension 中的 26 位，而不是 28 位。代码没有强行放宽合同，也没有静默填补。

### P1-3：冻结 state 的 sign test 把平局放进了分母

原 state 中的 sign test 过于保守。正式验收保留原 state，并另报剔除零差值后的 exact sign test。修正后主要结论不变：真实增量 `P=0.760`，block-null `P=0.084`，circular-null `P=0.202`。

### P1-4：六位联合阳性患者只能作为异质性线索

六位 high-support 患者同时满足真实 RNN 改善、优于 block null、优于 circular null，但这是同一批结果中事后取交集，不能命名为患者 subtype，也不能进入主结论。

## 4. 科学性审阅

### 4.1 这一次实际检验了什么

一个时间步是一场完整间期事件：

\[
E_{e-L+1:e}\rightarrow \mathcal D(E_{e+1:e+20}).
\]

目标是未来 20 场事件的 mode occupancy、contact mean rank 和 participation。主分数只取 occupancy 与 rank，participation 单独作为 secondary endpoint。

前面的 split-half 和 odd/even 结果已经说明患者特异 repertoire 跨时间稳定。这里不重新发现或否定这个 backbone，而是问：在稳定 repertoire 之上，过去事件的有序历史是否还提供可学习的 recurrent state。

### 4.2 参数确实进行了系统调整

development screen 比较了：

- cell：tanh-RNN、GRU、LSTM；
- hidden size：4、8、16、32；
- recurrent layers：1、2；
- RNN step：逐事件、每 5 场平均；
- normalization：none、z-score、robust；
- input / hidden LayerNorm；
- optimizer：Adam、AdamW、RMSprop；
- learning rate：`3e-4` 到 `1e-2`；
- batch size：32、64、128、256；
- gradient clipping：0.5、1、5；
- weight decay：0、`1e-4`。

冻结模型为：

| 项目 | 冻结值 |
|---|---:|
| matched baseline | descriptor EWMA |
| EWMA decay | 0.95 |
| ridge alpha | 100 |
| recurrent cell | GRU |
| hidden size / layers | 16 / 1 |
| recurrent step | 1 个完整事件 |
| normalization | train-only z-score |
| LayerNorm | input only |
| optimizer | RMSprop |
| learning rate | 0.001 |
| batch size | 128 |
| gradient clip | 1.0 |
| weight decay | 0 |

该 profile 在六位 development validation 上 5/6 增益为正，中位 baseline-minus-RNN gain 为 `0.01093`。冻结后没有再根据 34 人 test 结果改参数。

### 4.3 34 人主结果

分数越低越好；表中 delta 为 `RNN - baseline`。

| 人群 | n | RNN 更好 | median delta | Wilcoxon 单侧 P |
|---|---:|---:|---:|---:|
| 全 34 人描述 | 34 | 11 | 0 | 0.442 |
| locked extension | 28 | 8 | 0 | 0.526 |
| extension recruitment | 28 | 7 | 0 | 0.901 |

因此，当前 GRU 没有稳定超过 matched descriptor-EWMA。很多患者由 validation 正确选择 epoch `-1`，即保留 exact baseline；这不是训练失败，而是嵌套模型在没有可靠增量时避免向 test 注入有害 correction。

### 4.4 chronology null

以下 delta 使用“真实 chronology 的 gain 减去 null chronology 的 gain”，正值有利于真实顺序。

| 比较 | n | median | Wilcoxon 单侧 P | tie-excluded sign P |
|---|---:|---:|---:|---:|
| true gain 本身 | 28 | 0 | 0.526 | 0.760 |
| true minus block shuffle | 28 | 0.00782 | 0.0149 | 0.0843 |
| true minus safe circular | 26 | 0.000894 | 0.1368 | 0.202 |

block shuffle 有一个幅度上的弱信号，但有三点不能升级：

1. 真实 RNN 本身没有超过 matched baseline；
2. 更严格的 safe circular 没有通过；
3. 仅保留 high-support 患者后，block-null Wilcoxon 变为 `P=0.153`。

所以联合 chronology gate 明确未通过。

### 4.5 历史长度敏感性

`L=40/80` 完全复用冻结模型、基线、checkpoint rule 和三 seed，没有重新调参。

| 历史长度 | 完成患者 | extension n | RNN 更好 | median delta | Wilcoxon P |
|---|---:|---:|---:|---:|---:|
| 20 | 34 | 28 | 8 | 0 | 0.526 |
| 40 | 32 | 26 | 7 | 0 | 0.655 |
| 80 | 31 | 25 | 7 | 0 | 0.833 |

三种时间跨度结论一致，因此主阴性不能解释为“RNN 只看了 20 场，step 太短”。

### 4.6 synthetic calibration

在每个 history 都含相同数量 0/1 事件、只有最后事件顺序决定 target 的任务中：

- unordered baseline propagation score：`0.50125`；
- GRU propagation score：`0`；
- best epoch：7；
- 全部训练有限。

所以当前实现确实能在顺序含有信号时利用顺序。

### 4.7 当前安全结论

可以写：

> 在已由 split-half 和 odd/even 证明稳定的患者特异传播 repertoire 之上，一个经过实际架构和训练参数校准的 event-level GRU，没有在 28 位 locked extension 患者中稳定提高未来 repertoire prediction；该结果在 20、40 和 80 场历史下保持一致，并且没有通过联合 coherent-null chronology gate。

不能写：

- 稳定患者特异 repertoire 不存在；
- 所有可能的 RNN 都不可能学习事件历史；
- 间期事件不会塑造或一定会塑造病理网络；
- 六位探索性患者构成已确认 subtype；
- RNN 已经恢复了生物连接图。

## 5. 工程性审阅

### Round 1：数据与泄漏

**PASS。** 34/34 主分析完成；所有事件逐项验证属于旧 train80；history 严格早于 target；正式 target 不重叠；source 保持原子性，只有两 source 患者使用早 source 内部时间切分；旧 heldout20、A/B 轴、几何、SOZ、ictal 和 SNN 均未输入。

### Round 2：估计量与 checkpoint

**PASS。** RNN 是 matched baseline 上的 residual nested model；epoch `-1` 是 exact baseline；按患者 validation propagation 选 checkpoint，不再执行错误的固定 epoch refit。34 人中 31 人至少一个 seed 选择了已训练 checkpoint，15 人三个 seed 都选择训练 checkpoint，只有 3 人三个 seed 全部回退 baseline。

### Round 3：优化与结构

**PASS。** 所有 102 个主 run 有限，best epoch 范围 `-1` 到 15，没有撞到 120 epoch 上界；患者内 median clipping fraction 为 0.1125；模型 family、optimizer、LR、batch、normalization、step 和结构均真实进入 development validation 比较。

### Round 4：null 与科学 endpoint

**PASS engineering / NOT PASSED scientific gate。** 共 1,188 个 null contract checks 全部通过；block shuffle 是 source-coherent 的整序列重建；safe circular 同步移动 target 数值和 indices，并强制 history-target 不重叠。科学上 circular gate 未通过。

### Round 5：分母与解释

**PASS，bounded negative。** 主分母保留 34 人；六位 development 不进入 extension P 值；低支持患者没有被静默删除；L=40/80 失败名单显式保存；Epilepsiae 和 Yuquan 的主增量中位数都为 0。

## 6. 最小修改路线

本合同作为 window-reset residual family 已收口，但不能据此停止核心 stateful RNN。后续路线是：

1. 将 v2.5.4 归档为 window-reset residual RNN 的窄阴性；
2. 进入 v2.6，让 hidden state 沿完整 source chronology 连续传递；
3. 在所有患者 validation 上充分调整 recurrent cell、state dimension、TBPTT、optimizer、LR、normalization 和更新批量；
4. 之后才重新评价 cohort recurrent state；
5. 若未来需要模型外发，再补完整 baseline + encoder + RNN inference bundle。

## 7. 下一步建议

v2.5.4 只能说明固定窗口、状态重置的 residual GRU 没有稳定增益。它没有完成“状态沿完整事件序列持续演化”的核心检验，因此不能被用作整个 RNN 方向的停止门，也不能被用来否定已经成立的 split-half / odd-even 稳定性。

## 8. 关键产物

- 主状态：`results/topic5_stable_repertoire_event_rnn/v2_5_4/TRUE_CHRONOLOGY_STATE.json`
- 34 人表：`results/topic5_stable_repertoire_event_rnn/v2_5_4/patient_summary.csv`
- chronology null：`results/topic5_stable_repertoire_event_rnn/v2_5_4/chronology_nulls/CHRONOLOGY_NULL_STATE.json`
- L=40：`results/topic5_stable_repertoire_event_rnn/v2_5_4/history_length_sensitivity/l40/HISTORY_LENGTH_STATE.json`
- L=80：`results/topic5_stable_repertoire_event_rnn/v2_5_4/history_length_sensitivity/l80/HISTORY_LENGTH_STATE.json`
- 五轮验收状态：`results/topic5_stable_repertoire_event_rnn/v2_5_4/FIVE_ROUND_ACCEPTANCE_STATE.json`
- 冻结 profile：`results/topic5_stable_repertoire_event_rnn/v2_5_4/development_screen/FROZEN_PROFILE.json`
- spec：`docs/superpowers/specs/2026-08-02-topic5-trainable-event-rnn-v2_5.md`
- plan：`docs/superpowers/plans/2026-08-02-topic5-trainable-event-rnn-v2_5.md`
