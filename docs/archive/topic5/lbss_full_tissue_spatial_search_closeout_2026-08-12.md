# Topic 5.1 Full-tissue LBSS-RNN：空间连接搜索与收口报告

日期：2026-08-12
状态：**已按冻结合同完成并通过 closeout audit**；间期、full-tissue 几何、target-free 空间搜索、attenuation、early-ictal 外部 benchmark 与 Figure 6 候选均已收口

## 1. 这轮到底问什么

这轮不再把 SEEG contact 当成 RNN node。模型状态铺在患者自己的二维组织平面上，真实 contacts 只通过局部的 `H^T/H` 将 rank-set 输入组织、再读出组织活动。科学问题固定为：

> 患者自己的有序间期传播，能否由一个覆盖完整组织面的局部 recurrent backbone 学会；在这个 backbone 上，少量由任务选择的非局部 shortcuts 是否比等容量局部边或随机非局部边更有用？

发作数据不参与训练、连接选择或超参数选择。所有模型场及 attenuation 场冻结以后，才统一读取 Figure 3 的 early-ictal 目标。

## 2. 为什么必须重跑 full-tissue 版本

旧 v0.2 的 latent nodes 全部位于任一 contact 的 `3 sigma` 读出带内：31/31 fits 均没有真正不被 SEEG 直接读出的节点。那一版检验的是 contact 周围扩张出来的 latent domain，局部 kNN 可以直接跨过没有被建模的组织空白，不能据此关闭 selective nonlocal shortcut 假设。

v0.3 改为对整个 offset contact-cloud envelope 做近均匀铺点，再为 contacts 补最低限度的局部读出节点。正式几何审计如下：

| 指标 | v0.2 | full-tissue v0.3 |
|---|---:|---:|
| latent nodes / fit，中位（范围） | 60（32–192） | 104（64–346） |
| zero-H nodes，中位（范围） | 0（0–0） | 53（16–318） |
| zero-H fraction，中位（范围） | 0（0–0） | 0.578（0.250–0.919） |
| local edge 穿过直接读出区外组织，中位 | 0.131 | 0.702 |
| 所有 nodes 单一强连通分量 | 31/31 | 31/31 |

E1146 有 15 个 contacts、104 个 tissue nodes，其中 53 个完全不被任何 contact 直接读出。

## 3. 未观测组织节点不是摆设

对正式 L3 模型，在每个 rank step 后把全部 zero-H tissue states 钳为零，不重新训练：

- 21/21 患者留出 next-contact NLL 变差；
- 中位增加 `+0.26837 nats/decision`；
- zero-H / directly supported state amplitude 中位比值为 `1.159`；
- zero-H engaged fraction 中位为 `1.000`。

因此 v0.3 的非观测组织状态不仅存在，而且确实参与间期传播计算。这个 clamp 是诊断性干预，不是匹配 lesion，也不用于选择模型。

## 4. 模型与严格匹配对照

所有 arms 使用完全相同的 tissue nodes、`H^T/H`、leaky RNN、rank/STOP 目标、训练预算、free-rollout decoder 和 seeds；只改变局部 backbone 之外的新增边：

- `L0_LOCAL_ONLY`：强连通、双向 mask、方向权重独立学习的局部 backbone；
- `L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL`：同一 backbone，加等数量、仍偏局部的任务选择边；
- `L2_LOCAL_PLUS_RANDOM_LR`：同一 backbone，加等数量固定随机 nonlocal 边；
- `L3_LOCAL_PLUS_LEARNED_LR`：同一 backbone，加等数量任务选择 nonlocal 边；
- `C_L3_ORDER_SHUFFLED`：同一 L3 结构，在保持 rank 1 后破坏后续训练顺序。

`nonlocal` 只表示相对局部 backbone 更远的一步有效通信，不是白质束，也没有传导速度含义。

## 5. 全队列间期结果

正式矩阵为 21 位患者、31 fits、5 arms、3 seeds，共 465/465 单元；0 失败、0 unresolved OOM、0 非有限值。

### 5.1 局部 recurrence 已经足够

L0 相对 matched no-recurrence：

- held-out contact NLL 改善中位数 `+0.14083`；
- 20/21 患者同向；
- 双侧 Wilcoxon `P=1.91e-6`。

### 5.2 真实顺序是稳定信息

L3 相对 order-shuffle：

- 全部 transitions 改善中位数 `+0.12967`，21/21 同向，双侧 `P=9.54e-7`；
- distal transitions 改善中位数 `+0.06073`，14/21 同向，双侧 `P=0.0290`。

因此模型学到的不只是哪些 contacts 常一起出现，真实 rank order 对 recurrent computation 有稳定贡献。

### 5.3 task-selected nonlocal 没有选择性优势

在全部 transitions 上，L3 相对 L0/L1/L2 的中位增益分别为：

- L3−L0：`+0.00245`，Holm 单侧 `q=0.0753`；
- L3−L1：`+0.00149`，`q=0.6578`；
- L3−L2：`+0.00055`，`q=0.6578`。

在 distal transitions 上分别为：

- L3−L0：`+0.00222`，Holm `q=1`；
- L3−L1：`−0.00291`，`q=1`；
- L3−L2：`−0.00136`，`q=1`。

所以当前全队列最稳的空间结论是：**full-tissue local recurrence 足以完成患者间期传播；额外 task-selected nonlocal shortcuts 没有超过等容量局部或随机非局部连接。**

### 5.4 recurrent rollout 能恢复患者特异间期场

将每个模型只给第一 rank 后的自由生成事件，按与经验数据相同的 contact-rank field 合同汇总。四种真实顺序 recurrent arms 的 canonical model field 与经验间期场在 **21/21** 患者中均为正相关：

| arm | canonical field 相关中位数 | 正相关患者 | 相对 order-shuffle 的配对中位增量 | 双侧 Wilcoxon P |
|---|---:|---:|---:|---:|
| L0 local-only | 0.417 | 21/21 | +0.241 | 1.68e-4 |
| L1 extra-local | 0.454 | 21/21 | +0.242 | 2.67e-5 |
| L2 random nonlocal | 0.389 | 21/21 | +0.242 | 1.31e-4 |
| L3 selected nonlocal | 0.417 | 21/21 | +0.242 | 6.68e-5 |

去掉模型被直接给定的第一 rank 后，field 相关中位数仍为 `0.295–0.328`，各 arm 有 `17–19/21` 患者为正；相对 order-shuffle 的配对增量均为正且双侧 `P<=8.40e-5`。TA/TB contrast field 的经验对应同样稳定，canonical contrast 相关中位数为 `0.749–0.802`。

因此本轮可以把“full-tissue recurrent model 能从患者自己的有序间期序列生成患者特异传播场”作为正结果。该结果对具体 topology 不具选择性：四种真实顺序连接结构均成立，不能用最高的一个描述性中位数挑选所谓最佳连接图。上述 field 对照是预先冻结模型场的患者级补充统计，不反向参与训练、空间配置搜索或 early-ictal model selection。

## 6. 连接和超参数搜索

搜索完全基于三位冻结 development fits 的间期留出数据，不读取 early-ictal values。screen 系统改变：

- local density：0.06 / 0.10 / 0.15；
- added fraction：0.05 / 0.10 / 0.20；
- nonlocal cutoff：1.5 / 2 / 3 倍 local-edge 中位长度；
- rewiring fraction：0.10 / 0.20 / 0.35；
- learning rate：0.003 / 0.006 / 0.010；
- state dimension：1 / 2 / 4。

共完成 117 个单因素 screen 单元，另完成 9 个组合设置检查。screen 最好的两个单因素设置是较低 rewiring fraction `0.10` 和较低学习率 `0.003`，随后各自用 L0/L1/L2/L3/shuffle、3 fits、3 seeds 做 90 个 matched confirmation 单元。三阶段合计 216/216 单元，0 失败、0 OOM、0 非有限值。

确认结果：

| 设置 | L3−L0 distal | L3−L1 distal | L3−L2 distal | L3−shuffle distal | 判决 |
|---|---:|---:|---:|---:|---|
| rewiring 0.10 | +0.00007 | +0.00977 | −0.00068 | −0.01690 | 未确认 |
| learning rate 0.003 | +0.00137 | +0.00420 | −0.00590 | −0.00978 | 未确认 |

两个设置都没有同时超过三个 matched topology controls 和 order-shuffle，因此没有候选进入 21 人正式替换。冻结判决为：

```text
NO_SELECTIVE_NONLOCAL_CONFIGURATION_IN_FROZEN_SEARCH
```

这不是说任何其他超参数都不可能更好；它说明在预先限定、target-free、覆盖主要结构与训练尺度的搜索中，没有找到可复现的 selective-nonlocal 优势。继续扩大搜索直到出现阳性会变成结果追逐，不进入本论文主线。

## 7. Early-ictal 外部 benchmark

全部 intact/attenuated model fields 先冻结并写入 manifest，之后才读取 early-ictal values。固定分母为：

- contact-space RNN 总体参照：34 人间期；17 人/167 seizures early-ictal；
- full-tissue spatial mechanism：21 人间期；与 Figure 3 精确相交为 12 人/141 seizures；
- strict broadband sensitivity：11 人/92 seizures；
- E1146：15 次 strict-broadband seizures，clinical onset 后 0–10 s、1–150 Hz。

### 7.1 冻结 L3 场与 early-ictal field 存在正向对应

在 12 位 spatial exact-join 患者、141 次 seizures 中，L3 canonical-full field 相对 synchronized all-contact channel-shuffle null 的患者级 margin 为：

- 中位 `+0.14204`；
- bootstrap 95% CI `[+0.00770, +0.24333]`；
- 9/12 患者为正；
- 双侧 Wilcoxon `P=0.0122`，patient sign-permutation `P=0.0140`。

去掉模型被直接给定的第一 rank 后，seed-removed field 的中位 margin 仍为 `+0.13432`，9/12 为正，双侧 Wilcoxon `P=0.0161`，但 bootstrap 95% CI 为 `[-0.00845, +0.22264]`。strict-broadband 11 人/92 seizures 中，canonical 与 seed-removed 的中位 margin 分别为 `+0.10104` 和 `+0.10485`，方向仍为正，但双侧 `P=0.0830/0.1016`。

这支持的对象是：**完全由间期序列训练并冻结的 model-generated field，与同患者 early-ictal broadband energy 存在正向空间对应。**它不是独立外部数据集验证，也不是对某次发作 recruitment path 的预测。

### 7.2 跨状态对应不具有 selective-nonlocal 特异性

在 seed-removed endpoint 上，L3 相对三种 matched topology controls 的患者级中位增量为：

- L3−L0 local-only：`+0.00885`，双侧 `P=0.151`；
- L3−L1 extra-local：`+0.00291`，`P=0.470`；
- L3−L2 random nonlocal：`+0.00145`，`P=0.206`。

Claim D 的 7 项冻结 family 内 Holm 校正后，上述比较均不显著（`q>=0.908`）；canonical L3-vs-null 的 raw `P=0.0122` 在同一 family 内为 `q=0.0854`。控制 interictal fidelity 后，L3 相对其他 recurrent arms 的 cross-state effect 区间也均包含零。

因此不能写某一种 nonlocal topology 被发作早期特异复用。当前结果更像是：多种 recurrent solutions 都能生成患者特异间期场，并共享一部分能够跨状态对应的粗空间 scaffold。

### 7.3 Attenuation 没有闭合 shortcut-specific 因果链

L3 selected-nonlocal attenuation 的 early-ictal loss 随剂量数值上升；canonical 与 seed-removed 的患者级 dose-AUC 中位分别为 `+0.01254` 与 `+0.00775`，均有 9/12 患者为正，但双侧 `P=0.0923/0.233`。L3 added edges 相对 L1 added、L2 added 或 L3 matched-local 的 attenuation-AUC 对比均未显著；间期 distal-vs-local double-dissociation 同样未成立。

所以 attenuation 只提供方向一致的探索性趋势，不能支撑“selected shortcuts 对 distal propagation 或 early-ictal correspondence 具有特异必要性”。

### 7.4 评分修复与可复现记录

第一次 early-ictal scorer 在 E590 处停止，因为 target-free matching 合法地产生了 17 个而不是 21 个 attenuation fields；旧 scorer 错把可选的 `L3_MATCHED_LOCAL` 当成必需项。该次失败没有生成完成标记，原日志与 snapshot 均保留。修复后 scorer 在读取 target 前先完成全部 inventory preflight：5 个 intact arms 与 L1/L2/L3 added-edge 四档 attenuation 必须齐全；matched-local 允许“四档齐全或整组缺失”，禁止部分缺失。修复授权、restart record、旧/新 snapshot hash 均已归档，重启后流水线完整通过。

最终工程状态：正式训练 465/465；target-free search 216/216；未解决失败/OOM/非有限值均为 0；closeout audit `PASS`；Figure 6 metadata 不含 NaN/Inf。

## 8. 允许和不允许的科学表述

### 可以写

> Patient-specific ordered interictal rank sequences were learned by recurrent networks operating on a full-tissue latent plane. A strongly connected local recurrent backbone was sufficient for held-out propagation, and disrupting the true rank order consistently impaired prediction.

> Explicitly unobserved tissue states were dynamically engaged, showing that the model did not collapse to a contact-only recurrent graph.

> Model-generated fields learned exclusively from interictal events showed a positive cross-state spatial correspondence with early-ictal broadband energy. This correspondence was not specific to the task-selected nonlocal topology.

### 目前不可以写

- “RNN 恢复了患者真实白质连接”；
- “task-selected nonlocal shortcuts 优于局部或随机连接”；
- “搜索已经证明不存在 nonlocal pathway”；
- “精确 learned edges 是患者 connectome”；
- “某种 spatial topology 在 early-ictal 中被特异复用”；
- “L3 恢复了患者 anatomical/white-matter connectivity”；
- “early-ictal broadband energy 等同于组织已被招募进入 ictal core”；
- 用 L3-vs-null 的 raw `P=0.0122` 隐去预冻结 Claim-D family 的 `q=0.0854`，或反过来用 family 校正抹去其正向效应量与区间。

## 9. 当前判决

```text
FULL_TISSUE_LATENT_DOMAIN:                         PASS
ZERO_H_TISSUE_STATE_ENGAGEMENT:                   PASS
LOCAL_RECURRENT_SUFFICIENCY:                      SUPPORTED
ORDER_SPECIFIC_RECURRENT_INFORMATION:             SUPPORTED
SELECTIVE_NONLOCAL_ADVANTAGE:                     NOT SUPPORTED
BETTER_TARGET_FREE_SPATIAL_CONFIG_IN_FROZEN_SEARCH: NOT FOUND
EARLY_ICTAL_CROSS_STATE_BENCHMARK:                 POSITIVE RAW CORRESPONDENCE; FAMILY-LEVEL NONCONFIRMATORY
SELECTIVE_NONLOCAL_CROSS_STATE_CONTRIBUTION:        NOT SUPPORTED
ATTENUATION_DOUBLE_DISSOCIATION:                    NOT SUPPORTED
CLOSEOUT_AUDIT:                                     PASS
```

## 10. 交付物

- 正式结果根：`results/topic5_lbss_full_tissue_rnn_v0_3/`；
- claim summary：`LBSS_CLAIM_SUMMARY_V0_3.json`；
- early-ictal patient/seizure tables：`early_ictal/`；
- scorer repair authorization：`TARGET_SCORER_REPAIR_AUTHORIZATION.json`；
- restart provenance：`early_ictal/TARGET_SCORING_RESTART_RECORD.json`；
- closeout audit：`CLOSEOUT_AUDIT.json`；
- Figure 6 candidate：`results/paper-ready-figure/fig6_lbss_full_tissue_rnn/figures/topic5_figure6_lbss_full_tissue_rnn.{png,pdf,svg}`；
- Figure source metadata：同目录 `FIGURE6_METADATA.json` 与 `README.md`。
