# Continuous Marked State H2b Cross-task Transfer v0.4：白话报告

## 一句话结论

允许同一患者的不同发作通过不同的 seizure-entry route 后，冻结状态相对“当前 observation”出现很小的有利方向，但它没有胜过最简单的 recent IED history，也没有稳定胜过 memoryless、wrong-time 或单一慢轴。因此，本轮没有建立 H2b；同时由于真实可评分发作少、严格 assay power 未过，也不能把结果解释成生物学阴性。

这里“不同 route”只表示不同发作可能对应不同的低容量读出路径，不表示不同病因。

## 十个问题的直接回答

1. **是否真的只从间期任务学状态？** 是。46 个 state-cache cells 全部来自已机器验收的 v0.3 frozen source；checkpoint 与 cache SHA256 逐个复算一致。seizure label 只进入低容量 probe，不能更新 observer、state update、generator 或 IED decoder。所有 anchor 只使用 `t <= onset - lead` 的信息，跨 gap 不传播。

2. **多少患者和发作可分析？** 来源是 10 位患者、46 个 patient×seed cells。30 min 时 6/10 位可估计：1 位达到至少 10 次支持发作的 primary chronological split，4 位属于 5–9 次 rolling sensitivity，1 位仅为 3–4 次 descriptive；另 4 位少于 3 次支持发作而不可估计。26/46 cells 完成 development 评分，20/46 明确记为支持不足。

3. **状态是否胜过 recent IED history？** 没有。30 min 的 `B_route_state - B_history` 患者中位数为 `+0.2633`，仅 2/6 患者方向有利；正值表示加入 observation 和 state 后反而更差。这个直接比较避免了把两个分别聚合的中位数相加。

4. **状态是否胜过当前 observation？** 只有弱而不稳定的方向。主比较 `B_route_state - B_observation` 中位数为 `-0.0133`，4/6 有利，95% patient bootstrap 区间为 `[-0.1201, +0.0034]`，exact sign `p=0.6875`。而 `B_observation - B_history` 本身为 `+0.1748`、仅 1/6 有利，说明这点小增量发生在一个可能过拟合的 observation baseline 之上，不能升级为状态迁移证据。

5. **persistent 是否胜过 memoryless？** 不稳定。`B_route_state - B_route_memoryless` 中位数 `-0.0191`，3/6 有利，区间 `[-0.2564, +0.0050]`。

6. **correct-time 是否胜过 wrong-time？** 不稳定。matched wrong-time 对比中位数 `-0.00284`，3/6 有利；circular-shift 对比中位数 `-0.00277`，同样 3/6 有利。没有一致的正确时刻特异性。

7. **多早能看到信息？** 固定 sensitivity 中，state 相对 observation 的患者中位数分别为：5 min `+0.0024`（3/6 有利）、15 min `+0.0039`（3/6）、30 min `-0.0133`（4/6）、60 min `-0.0069`（2/3）；120 min 无患者可估计。不能事后把 60 min 或其他 lead 选成新的主结果。相对 history，5/15/30 min 均不利，只有 60 min 的 3 人描述层为负。

8. **是否预测发作表现？** 没有可靠证据。预先存在、未按状态重新聚类的两个连续 recruitment targets 共形成 12 个可估计 patient-target rows，只有 1 个 state-vs-observation 方向有利；患者—目标中位 loss difference 为 `+0.00086`。route-specific phenotype 因 target 支持过稀而不可估计。

9. **H2b 的证据等级？** 工程验收通过，科学证据未建立。真实 coverage 半合成 assay 能方向性恢复 two-route positive world：state-vs-observation 82%、persistent-vs-memoryless 75%、two-route-vs-single-axis 88%、correct-vs-wrong-time 89%；三个负对照也受控。但 single-route time-specificity 只有 66%，严格单次检出力的五项检查全部未过，所以不能据此宣布“没有跨任务状态”。

10. **是否值得现在进入正式分区？** 不值得。当前只有 1 位 primary chronological 患者；实际双 route 比单轴仅 2 位可估计，而且 2/2 都更差（中位 `+0.1134`）。下一步应先扩大每位患者的完整发作支持，并在不打开 formal/sealed 的前提下把 assay 的 correct-time 和单次检出力做到可接受，再决定是否正式验证。

## 流形—流场结果怎么理解

四位可评分患者都出现了描述性的 abrupt seizure-entry transition（4/4，患者中位 1.0），但 route-specific basin gating 只有 3/4，directed approach 只有 1/3 非零患者方向有利。双 route 相对单 route 的几何增量基本为零。这更像“发作进入时有一般性突变”，不足以证明存在可迁移的、route-specific persistent state。

## 安全结论

本轮回答了“过严的共同原因 gate 是否掩盖信号”：不是。即使允许患者内异质 entry route，冻结间期状态仍没有在 history、memoryless、正确时刻和 route-specific 对照上形成一致证据。允许写的是“异质 route 模型已完成 development 检验，H2b 未建立且仍未决”；不能写机制、因果、attractor、临床预测或 formal confirmation。
