# Group-Event State v0.3.2 阶段收口（白话版，复审更正）

**日期：** 2026-09-02

**状态：** `V0_3_2_PIPELINE_ACCEPTED_ASSAY_POWER_UNCALIBRATED_CLOSEOUT`

**范围：** development-only；正式封存分区未打开。

## 一句话结论

v0.3.2 已经把数据、显式历史、候选状态、错时对照和冻结 grammar probe 接成一条可复现管线；但它还没有把“多大真效应能够被稳定找回”定标清楚，而且只用未来事件数训练状态。因此，这一轮既没有发现共享慢状态，也没有排除它。

## 这轮正式验收什么

本轮验收的是工程和测量地基：27 位患者的记录支持与资格审计完成，三位开发患者各三个 seed 的 12 维候选状态、H1 count 评价和 H2a grammar 迁移均已跑齐，相关 65 项测试通过，H2b、H3 和 sealed partition 均未运行。

当前状态应拆成六句：

```text
pipeline：通过
assay power：未定标
H1：未决，30 min 合格分母只有 1 人
H2a：未决，训练目标与迁移目标不一致
H2b：未运行
H3：未运行
```

## 为什么不再叫“仪器不稳定”

空真值 6 次中没有观察到假阳性，但 6 次只够作 sanity check，不能证明特异性已经可靠建立。

阳性合成中，连续的 held-out gain 随人工效应增强而增加：

| 人工效应 β | median gain（nats/anchor） | CI 规则通过数 |
|---:|---:|---:|
| 0.35，新 seed | +0.0227 | 2/3 |
| 0.70 | +0.1931 | 3/3 |
| 1.40 | +0.2738 | 1/3 |

真正不稳定的是“3 次重复里有几次 CI 下界过零”这个二值判据，而不是连续 gain。每档只有三个 replicate，2/3、3/3 和 1/3 的波动不能证明更强效应更难恢复。

所以正确口径是：**positive-recovery power 未定标**。下一版必须用更多 replicate、oracle effect 和分层 recovery 定位问题。

## H1：过去群体 IED 是否形成了超出近期统计的预测状态

主任务预测未来 30 分钟事件数，并比较：

1. `H+正确时刻状态` 对显式多尺度历史 `H`；
2. 正确时刻状态对 block-shifted 状态；
3. 动态状态对训练期平均状态。

三位开发患者中，只有 `epilepsiae_1146` 对 30 分钟主任务具备足够独立时间块。其三项增量分别为 −0.329、−0.443、−0.329，均不支持当前模型。

这只能写成：

> 在唯一合格患者的 development block 上，当前 count-trained leaky representation 没有在显式历史之外提供正确时刻的 30 分钟 count 增量。

不能写成“三位患者阴性”，更不能写成“脑内不存在慢状态”。另外，模型内部 dev-test 曾给 E1146 一个方向相反的 `H−correct=+0.1277`；外部统一评价为 −0.3291。两条评分路径尚未数值对齐，这将是 v0.3.3 的第一项修复。

## H2a：状态是否改变下一次群体事件的空间走法

三位患者都具备足够的事件 prefix 和触点支持。当前结果没有显示 count-trained 状态能够稳定改善继续/停止、招募大小、具体触点集合或后续继续传播。

但这个实验存在明确的目标错位：状态只用未来 30 分钟事件数训练，再被冻结去预测空间 grammar。一个 scalar count readout 最容易训练出 count-relevant 方向，不能据此排除另一种专门承载 propagation grammar 的状态。

此外，“相对测试集中最强对照”的数值受极值选择影响，已降为敏感性。主解释只看 `vs H` 和完整 shift-null 分布。

**H2a 结论：未决，不是空间状态阴性。**

## “是不是网络根本没学会”的当前证据

这个担心合理，而且 v0.3.2 还没有系统排除。

当前 event encoder 只有一套超参数：两层 MLP，hidden=32，输出 4 维 event write；AdamW 的 encoder LR 为 1e−3、adapter LR 为 3e−3、weight decay 为 1e−4，最多 600 steps。gate `alpha` 初始化为 0.03，并冻结前 50 steps。

九个 learned run 的最佳 checkpoint 全在 step 20–50；也就是说，被选中的模型里 gate 仍然停在初始化值 0.03。encoder 有梯度，训练损失也下降，因此不是“完全没训练”；但只试过一套学习率、容量、gate schedule 和 early-stopping 规则，不能排除训练配方限制。

还要特别澄清：v0.3.2 **没有学习原始脑电波形**。它读取的是已经提取出的群体事件参与、tied group、精确延迟、空间离散度、多频带汇总和 cross-band lag。当前结果最多约束“这套提取后事件特征 + 两层 MLP + count objective”，不能说 raw SEEG 没有信息，也不能说 waveform encoder 没学会。

## H2b 与 H3

- H2b 未运行，不是发作迁移阴性。下一版在 interictal objective 锁定状态后，可以运行 development-only 的冻结发作迁移诊断；发作结果不得反向选择或训练 state。
- H3 未运行，不是 IED feedback 阴性。只有先得到有 correct-time 预测价值的状态，才比较 common-drive、count feedback 和 mark-specific feedback。

## 下一版的核心变化

下一版不先换大网络，而按顺序回答三个问题：

1. **量尺是否能测到：** 用 oracle evaluator、oracle memory、encoder recovery 和完整恢复四层合成实验定标。
2. **网络是否学会：** 固定架构，系统检查学习率、gate schedule、容量、正则、训练步数和 checkpoint 选择；先做小样本过拟合与 blocked inner-validation，再谈架构。
3. **学的目标是否正确：** 分别训练 count-view state 与 grammar-view state，再用 cross-transfer 判断它们是否共享；不再预设一个 count-state 自动代表病理网络状态。

只有上述三层分开后，真实人体阴性才知道是在说量尺、训练、目标、架构，还是数据本身。

## 产物

- 机器收口：`results/group_event_state/v0_3_2/v0_3_2_closeout_summary.json`
- 技术报告：`docs/archive/topic5/group_event_state_v0_3_2_closeout_technical_2026-09-02.md`
- 下一版 spec：`docs/archive/topic5/group_event_state_v0_3_3_dual_view_state_spec_2026-09-02.md`
- 下一版 plan：`docs/archive/topic5/group_event_state_v0_3_3_dual_view_state_plan_2026-09-02.md`
- H1/H2a 图：`results/group_event_state/core_evidence/figures/`
