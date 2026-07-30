# Topic 5 RNN bounded closeout and replication readiness v0.1

> **Superseded as an execution contract.** 本文件把当前论文收口与未来外部复制混在一起。
> 现已拆分为：
>
> 1. `2026-07-28-topic5-static-contact-topography-supplementary-closeout-v1_0.md`
> 2. `2026-07-28-topic5-external-clinical-onset-replication-protocol-v1_0.md`
>
> 本文件只保留设计谱系，不再继续执行。

## 1. 目的

本合同不再调 GRU、low-rank、axis、source、loss 或 rollout 参数。它把当前计算线收束成两个
互不替代的结果：

1. full GRU 对间期事件内部顺序扰动敏感，但没有稳定优于 rank-shuffle 的 heldout NLL；
2. 间期 contact participation 与 clinical-onset 后早期能量共享 sign-free 静态空间形态，
   但该形态同样存在于正则化非递归场，不能归因于 RNN 特有动力学。

下一阶段只完成论文整合和独立复现准备，不在已经打开的 16 人/106 seizure target 上继续
搜索 readout、polarity、模型或患者亚组。

## 2. 冻结结果

- formal interictal cohort：34 人，3 seeds；
- strict early-ictal cohort：Epilepsiae 16 人、106 seizures；
- target：clinical onset 后 `[0,10] s`、`1–150 Hz` baseline-normalized energy；
- primary signed field：interictal participation，固定正方向；
- sensitivity：`abs(rho)` morphology；
- spatial null：all-contact、within-shaft circular/dihedral、geometry-smooth；
- comparator：raw train80、target-free best regularized、static hazard、first-order、
  rank-shuffle、teacher-forced；
- 所有 target-free baseline 和 teacher-forced 输出保持冻结。

## 3. 允许写入论文的结论

### 3.1 允许

> The full-history GRU encoded sensitivity to within-event order, but did not
> reliably improve held-out next-contact likelihood over a rank-shuffled GRU.

> Interictal contact participation captured a patient-specific sign-free spatial
> morphology shared with early-ictal broadband energy. This correspondence was
> retained by regularized non-recurrent fields and therefore did not require a
> recurrent dynamical explanation.

### 3.2 只作 sensitivity

- free rollout 相对 teacher-forced 的差异；
- partial-rank residual analyses；
- participation-residualized hidden-state PCs；
- baseline-power、SOZ、geometry 单混杂控制。

### 3.3 禁止

- RNN 预测了发作传播；
- RNN 自动恢复了患者病理轴或 source；
- low-dimensional PCA state 等同于生物 E/I state；
- sign-free morphology 等同于固定正方向 replay；
- 当前结果是独立确认；
- 依据当前 target 再选择 polarity、field、模型或患者亚组。

## 4. 论文位置

- 主文：最多一句 bounded computational result，不能承担主机制结论；
- Supplementary Results：完整 34 人 interictal 诊断和 16 人 static bridge；
- Supplementary Figure：固定六块 Figure 6 candidate；
- Methods：数据切分、target sealing、teacher/free 定义、空间 null、patient-first 统计；
- Discussion：静态 scaffold 与动态 replay 的区分，以及当前没有独立复制队列。

## 5. 独立复现门

只有存在未参与当前设计和读取的 patient-level target cohort，才能启动 replication：

1. 在读取结果前用预期效应和 precision/power 明确 patient-level 样本量，不设事后便利阈值；
2. exact clinical-onset anchor 和 contact join；
3. 同一 `[0,10] s`、`1–150 Hz` target；
4. 不重新训练或选择 field，直接应用冻结的 raw participation、best regularized、
   rank-shuffle、full free-rollout；
5. primary 仍为 signed Spearman；`abs(rho)` 只作 sensitivity；
6. 复制失败不能用重新选择 polarity 或 patient subgroup 补救。

若没有独立 cohort，本合同停在 replication-ready handoff，不启动新的 early-ictal RNN。

## 6. 新动力学模型的前置条件

新的 matched-prefix state-swap、on-manifold dynamics 或 structured RNN 只有在独立数据中同时
满足以下条件后才可启动：

1. full GRU 对 rank-shuffle/first-order 有稳定 heldout NLL 增益；
2. full free-rollout 对 strongest target-free static baseline 有 early-ictal 增量；
3. 该增量在 within-shaft 和 geometry-aware null 下保留。

当前 v0.1 不满足这些条件，因此该路线标记为 `DEFERRED_NOT_STARTED`，不是全局 no-go。
