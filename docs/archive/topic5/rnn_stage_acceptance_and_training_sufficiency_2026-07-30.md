# Topic 5 RNN 阶段性验收与训练充分性边界

日期：2026-07-30

状态：`STAGE_ACCEPTED_WITH_TRAINING_SUFFICIENCY_OPEN`

## 1. 一句话判断

当前 RNN 线已足以支持：

\[
\boxed{
\text{稳定的患者特异 contact scaffold}
+
\text{短程、事件内的有序转移信息}
}
\]

但完整事件自由生成失败目前只能解释为：

> 冻结的 teacher-forced linear-state 模型没有把局部 next-contact 信息稳定组合成完整传播事件。

在完成训练覆盖轮数和 self-fed rollout 目标的冻结审计前，不能写成：

> RNN 原理上学不到完整间期传播。

## 2. 阶段性验收

### 2.1 科学合同

**执行至冻结停止点：100/100。**

- 34 人、3 seeds 的 chronological train80 / heldout20 分析已完成；
- static、unordered、first-order、linear-state、vanilla RNN、GRU 和
  low-rank families 已在同一 next-set / STOP 合同下比较；
- H1、H2、H3 和 full history 已完成同分母重评分；
- contact-choice 与 STOP 已分解；
- linear-state 已展开为输入—输出 lag kernel \(K_k=CA^kB\)；
- source-conditioned free rollout 已完成 102/102 单元；
- patient-mean early-ictal reused target 已在模型冻结后读取；
- 没有发现 target leakage、事件 mask 不一致、NaN、OOM 或未完成单元。

### 2.2 训练充分性

**完成度：65/100。**

当前训练足以证明模型学到了局部顺序信息，但不足以把完整事件生成阴性归因于模型类本身。

## 3. 已经成立的结果

### 3.1 Where：静态 contact scaffold 稳定

- train80–heldout20 participation Spearman 中位数为 0.893；
- 34/34 患者方向为正；
- patient-mean early-ictal field 的 orientation-free correspondence
  主要由这一稳定空间 scaffold 解释。

这支持“患者特异的病理招募骨架在间期和发作早期具有空间对应”，但不支持逐触点动态 replay。

### 3.2 How：有序信息集中在最近的 rank steps

- linear-state 相对 unordered-prefix 的患者中位 NLL 增益为 0.0257，
  26/34 患者为正；
- matched rank shuffle 显示真实顺序增量；
- contact-choice 的可识别贡献主要来自 \(K_0\) 和 \(K_1\)；
- 第三个 rank 的主要作用是 STOP，而不是下一个 contact 身份；
- full history 不优于 H3。

因此当前支持的是 bounded within-event predictive memory，不是真实时间的慢变量或发作倒计时。

### 3.3 自由生成：局部 transition 可恢复，完整事件不可恢复

冻结 linear-state 的 source-conditioned rollout：

- 改善 first-order transition correlation：30/34；
- 改善 first-order transition MAE：28/34；
- 不改善完整 suffix rank 或 precedence；
- 只有 9/34 患者在 participation、rank、precedence 中至少两项达到
  heldout-half 经验变异范围；
- template 和 signed physical-axis fidelity 没有改善。

因此模型不是“完全没有学到”，失败发生在局部转移组合为完整事件这一层。

## 4. 当前超参数和训练合同审计

### 4.1 已经调整过的参数

target-blind tuning 比较了 8 个配置：

- hidden size：32、64；
- learning rate：\(5\times10^{-4}\)、\(10^{-3}\)；
- local offset dimension：4、8；
- batch events：128；
- shared optimizer steps：512；
- optimizer：AdamW；
- weight decay：\(10^{-4}\)；
- gradient clipping：1.0。

one-standard-error 规则选择 h32、learning rate \(10^{-3}\)、offset 4；均值最优配置为 h64、learning rate \(10^{-3}\)、offset 8。参数量匹配的 linear-state h64 sensitivity 保留了主要 one-step 结果。

### 4.2 尚未关闭的问题

1. **调参模型不完全匹配。**
   上述 8-cell tuning 使用 `FullHistorySequenceGRU`，最终获胜的
   `LinearStateSequenceRNN` 没有单独进行 learning-rate / training-budget tuning。

2. **正式 shared training 只覆盖训练事件一轮。**
   正式架构审计使用 `shared_cycles=1`、每位患者 8 次更新；LOSO 外层每个模型约
   264 次 shared optimizer updates。尚未比较 1、2、4 个 coverage cycles
   是否达到 validation plateau。

3. **batch size 主要是显存 chunk。**
   正式训练把患者事件段切成 1024-event chunks，但在整个事件段累计梯度后只执行
   一次 `optimizer.step()`。因此 1024 不是普通意义上的有效 minibatch size；
   真正改变优化分辨率的是 `updates_per_patient`。

4. **优化器家族基本固定为 AdamW。**
   没有针对最终 linear-state 系统比较 Adam、weight decay、scheduler 或 warmup。

5. **训练和自由生成的输入分布不同。**
   训练只使用真实 prefix 做 one-step teacher forcing；自由生成使用模型自己的历史。
   没有 scheduled sampling、multi-step self-fed loss 或 rollout-aware objective。

## 5. 当前允许和禁止的论文措辞

### 允许

> Ordered interictal rank sequences contained short-range predictive information beyond stable contact recruitment preferences. A simple linear recurrent state captured this local dependence, whereas its frozen teacher-forced realization was insufficient to generate the full event-level rank and precedence distributions.

### 禁止

- RNN 自动恢复了患者病理轴；
- hidden PC1/PC2 是兴奋/抑制状态；
- low-rank mode 等同于生物低维流形；
- RNN 能预测 clinical onset 或 seizure-specific early-ictal dynamics；
- 完整事件生成阴性证明任何 RNN 都学不到传播。

## 6. 阶段性论文定位

当前结果可以进入 Extended Data / Supplementary，回答：

> 在稳定 contact 招募先验之外，间期事件内部是否存在可学习的有序信息，以及这种信息能否形成可生成完整事件或跨状态复用的 recurrent state？

当前答案为：

1. 短程有序信息存在；
2. 简单 linear-state 可以利用；
3. 信息量有限且主要集中在最近两个 rank steps；
4. 当前训练合同不能把它组合成完整双向传播事件；
5. early-ictal patient-mean correspondence 主要来自 static scaffold。

## 7. 下一步冻结任务

下一轮只允许进行：

1. linear-state 的训练收敛审计；
2. teacher-forced 与 rollout-aware objective 的分离实验；
3. 最优训练合同下的 34 人 × 3 seeds 冻结确认；
4. 更新完整事件生成和局部 transition 两层结论。

不得重新开启 axis/path/low-rank architecture zoo，也不得用 outer heldout、
A/B、physical axis 或 early-ictal target 选择训练超参数。
