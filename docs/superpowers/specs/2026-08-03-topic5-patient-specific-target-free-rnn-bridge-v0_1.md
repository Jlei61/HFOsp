# Topic 5 patient-specific target-free RNN bridge v0.1

## 科学问题

本合同只问一件事：

\[
\text{同一患者的间期 contact-rank events}
\rightarrow
\text{患者内自监督 RNN}
\rightarrow
\text{RNN 自己生成的 contact field}
\]

是否与该患者 clinical onset 后 0--10 s 的 early-ictal contact-energy field 对应。

这不是跨患者泛化任务，也不要求每名患者都有同样强的效应。患者级结果先独立计算，
cohort 统计只总结效应分布。

## 数据和防泄漏合同

- 输入仅为 masked contact-rank events；未参与 contact 必须保持 `-1`。
- 每名患者单独按时间切成 60% fit、20% validation、20% test。
- 任何其他患者的数据都不得进入该患者模型。
- empirical A/B field、A/B label、SOZ、geometry 和 ictal target 都不得进入训练。
- early-ictal target 只在三个 seed 的间期 checkpoint 和 rollout 冻结后读取。
- 主 target 为 clinical-onset `[0,10] s`、1--150 Hz contact energy；1--45 Hz 为
  sensitivity。

## 模型与任务

主模型是 hidden-32 的 within-event full-history GRU。一次群体事件开始时状态清零；
模型逐 rank set 观察已经出现的 contact，并自监督预测下一 rank set 与 STOP。

线性状态模型在完全相同的数据、训练预算和 readout 下作为跨架构 sensitivity。它不是
新的科学分支。

每名患者的 contact identity 通过只在该患者 fit60 中学习的 local embedding 表示；
固定 contact features 只含由 fit60 rank events 计算的 participation、conditional
mean-rank 和常数项。没有经验 A/B 或发作信息。

训练配置冻结为：7 次完整覆盖、每次 32 次参数更新、batch 256、AdamW、lr `3e-4`、
weight decay 0、gradient clip 1。三个 seed 为 11、29、47。

## 患者内间期验收

validation20 和 test20 只用于报告：

- next-set/STOP event NLL；
- top-1 next-contact accuracy；
- rollout participation error；
- rollout rank-distribution Wasserstein distance；
- pairwise precedence correlation。

within-event rank-shuffle 使用相同架构和训练预算。它破坏事件内传播顺序，但保留每个
事件的参与 contact 集合。静态 participation/rank distribution 由 fit60 直接估计。

这些比较是信息拆解，不设置阻止 early-ictal readout 的复杂 gate。只要训练有限、无
泄漏、无 NaN/OOM 且 checkpoint 完整，就进入跨状态评分。

## RNN-derived contact fields

每个 seed 从冻结模型自由生成 5000 个完整事件，汇总：

1. participation probability；
2. early rank mass；
3. late rank mass；
4. early+late endpoint mass；
5. participation-weighted earliness。

这些 field 全部从模型 rollout 得到，不注入 empirical A/B。empirical fit60/test20
distribution 只作为数据参照；participation-only 为静态 baseline。

## Early-ictal 评分

在每次发作的 exact joined contacts 上，对候选 model fields 与 target 分别计算
Spearman correlation，取绝对值后取最大候选值。每次 all-contact permutation 和
within-shaft permutation 都必须重新执行同样的候选最大化，从而支付 readout 选择成本。

先在 seizure 内评分，再在 seed 内折叠，最后得到患者级分数。正式报告包括：

- 每名患者绝对相似度和 null margin；
- GRU、linear-state、rank-shuffle、static participation 和 empirical test reference；
- development subject `epilepsiae_1146` 单列 supportive；其余患者为 primary summary；
- 不以“所有患者阳性”为要求。

## 可写与不可写

若结果成立，可写：患者自身间期 rank events 训练出的自监督 RNN 恢复了患者特异
contact distribution，该模型场与同一患者 early-ictal broadband field 存在空间对应。

不能仅据此写：RNN 恢复了唯一物理 A/B 轴、预测了发作传播顺序、hidden unit 是真实
E/I 神经元、或每名患者都有相同机制。
