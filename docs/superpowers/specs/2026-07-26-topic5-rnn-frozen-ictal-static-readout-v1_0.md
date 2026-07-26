# Topic 5 / Figure 6：冻结 RNN 表征到发作早期静态场的条件合同

**日期**：2026-07-26
**状态**：预注册但封存；只有 interictal v1.0 formal gate 通过才执行

## 1. 科学问题

间期 RNN 的训练是自监督的；发作期不是第二段序列训练。第二阶段只问：

> 纯间期模型自由生成的触点级 rank distribution，能否在模型和路径结构完全冻结后，
> 读出同一患者 clinical onset 后的静态 1–150 Hz 能量场？

阳性结果支持 shared scaffold/readout，不支持逐秒发作传播重放，也不证明因果机制。
由于间期 train80 不是按每次目标发作之前截断，本任务是患者特异的跨状态场读出，
不是 prospective seizure forecasting，也不能报告提前量、灵敏度或报警性能。

## 2. 硬启动门

只在
`formal_persistent_path_mode_v1_0/analysis/formal_gate_summary.json::
formal_interictal_gate_pass=true`
时读取任何发作期数值。

若该字段为 false，本阶段不得执行；interictal 结果按 34 人 bounded-negative 收口。

## 3. 输入和目标

### 冻结输入

- 使用 34 人正式任务中 `K=2 intact` 的 5000-event free rollouts。
- 三个 seed 先在每个患者、每个触点内取中位数，不能把 seed 当独立样本。
- 每个触点表示成一个 11 维概率向量：
  - 第 1 维：不参与事件的概率；
  - 后 10 维：参与且位于相应 normalized-rank bin 的联合概率。
- 这 11 维之和为 1；不输入 A/B label、contact name、坐标、SOZ 或发作信息。

### 冻结目标

- 只使用有 accepted clinical-onset target 的患者；与 34 人间期队列取交集。
- 排除只有 EEG-onset 的病例，不用 EEG onset 代替 clinical onset。
- 每次发作的目标固定为 clinical onset `[0,10] s`、1–150 Hz、
  baseline-robust-z contact energy。
- 先在患者内跨 seizure 取 contact-wise median，形成每位患者一个静态场；seizure
  不能作为独立训练样本。
- exact contact-name join，少于 6 个共同触点则该患者不进入 readout。

## 4. 冻结 readout

- 外层按患者 LOSO。
- 在其余 eligible 患者的触点上拟合一个共享 ridge readout，`alpha=1.0` 固定，不调参。
- 每位训练患者总权重相等，避免触点多的患者主导拟合。
- 输入标准化只用外层训练患者。
- 目标只做患者内 median/MAD 标准化，readout 预测相对 contact-field shape。
- heldout 患者不校准截距、不重训 RNN、不改 `K`、不改 path prior。

同一流程并行评估：

1. `K=2 intact` 自由生成分布；
2. train80 empirical rank distribution（数据上限参照）；
3. no-history 自由生成分布；
4. graph lesion 与 mode-collapse lesion 分布。

## 5. 主统计

每位 heldout 患者得到一个 contact-level Spearman `rho`，比较预测场与真实 clinical-onset
BB150 场。主 null 为 heldout 患者内 coherent all-contact channel-label shuffle：

- readout 和输入保持不变；
- 每次 draw 只重排真实能量场的 contact labels；
- `n_perm=5000`；
- 患者统计量为 `rho_data - median(rho_shuffle)`。

队列以患者为单位，报告 median margin、正 margin 患者数和 one-sided Wilcoxon；
all-contact channel shuffle 是主基准。within-shaft shuffle 只作解剖敏感性，不是硬门。

主 readout 只有同时满足以下条件才称为 cross-state positive：

1. 至少 8 位 clinical-onset eligible 患者；
2. intact cohort median margin > 0；
3. intact 正 margin 患者超过一半；
4. intact patient-level one-sided Wilcoxon `p<0.05`；
5. intact 优于 no-history，且 graph lesion 或 mode-collapse 至少一个使患者级 `rho`
   显著下降。

empirical readout 用于说明可达到的上限，不要求 intact 超过 empirical。

## 6. 允许与禁止

允许写：

- 自监督间期 RNN 产生的冻结触点分布保留/未保留 clinical-onset 静态能量场信息；
- 路径结构损伤是否削弱这一跨状态 readout；
- 34 人 interictal formal cohort 与较小 clinical-onset target subset 是两个不同分母。

禁止写：

- RNN 学到了逐秒发作传播序列；
- 当前结果构成了前瞻性发作预测器或能预测某次发作何时发生；
- EEG onset 阴性结果被 clinical-onset 任务覆盖；
- seed、seizure 或 contact 是独立 cohort 样本；
- 预测相关性本身证明真实病理机制或因果关系。
