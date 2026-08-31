# Continuous Marked State H2b Cross-task Transfer v0.3：白话验收

## 一句话结论

这轮把问题问得更严格之后，答案不是“状态不能预测发作”，而是：**当前 R1.7B checkpoint 还没有先证明自己是一个合格的、多维、跨窗口 persistent state，因此本轮不能进入有科学主张资格的发作风险和流形—流场检验。H2b 仍未建立。**

v0.2 仍按既有验收收口为工程通过的 negative pilot。v0.3 修复了分母、状态资格和 assay 的关键实现问题，并在完整的 75 个可读 checkpoint cells、16 位患者上完成了 outcome-blind 状态体检。工程链闭合，但科学路线在 A1/A2 正确停下。

本次指定验收意见优先于较早的“少 gate、多探索”执行附录：A1 没有足够 `state_qualified` checkpoint 时必须停止 downstream，A2 power 不足时不得解释真实阴性。旧附录保留为历史记录，但不再授权 diagnostic override。

## 1. 我们是否真的只从间期任务学状态

是。

- 75 个 checkpoint 均来自连续背景与 IED timing/mark 任务；
- 每个 checkpoint 的 SHA256 均重新计算并与 manifest 一致；
- observer、state update、generator 和 IED decoder 在 H2b 前冻结，所有参数 `requires_grad=false`；
- A1 不读取未来 seizure-risk outcome；已验证的过去发作时间只在 Q6 中作为严格因果 nuisance；
- seizure loss 没有回到状态模型；
- formal、sealed、H3、T2、physical clock 和 paper-ready figures 均未打开或运行。

因此，这轮没有“用发作标签把状态训练成会预测发作”的反向泄漏。

## 2. 实际分母是多少

R1.7B inventory 共有 85 个候选 cells：

- 75 个 checkpoint 可读，来自 16 位患者；
- 10 个 cells 因上游 `NONFINITE_GRADIENT` 无 checkpoint；
- v0.2 已有 46 个 seizure state-cache cells；
- 24 个可读 checkpoint cells 当时没有冻结的 seizure support；
- 5 个 cells 没有完整 30 min coverage。

v0.3 的 A1 没再用 seizure support 决定谁能做状态体检，而是把全部 75 个可读 checkpoint cells 都纳入。E1084 和 E583 缺失的 interictal design 也已重建，并同时通过 design hash、normalized explicit hash 和所有 seed baseline tensor 的位级等价检查。

## 3. 状态资格结果

最终 `state_qualified = 0/16`。

资格要求不是单项好看，而是同一患者至少 3 个 seeds 联合通过：

- Q1：decoder-output state 是非坍缩的多维状态；
- Q2：persistent state 比 memoryless state 更能预测 held-out future IED；
- Q3：窗口间 generator 确实保留信息，而不是 observer 每窗重写；
- Q4：decoder metric 中的 `tau_z` 可辨识；
- Q5：跨 seed 的 decoder geometry 稳定；
- Q6：在 time of day、recording position、近期 IED history、segment 和严格过去发作 nuisance 后，增量仍存在。

很多患者只通过其中一部分。例如：

- E1125 的 Q2、Q3、Q4、Q5、Q6 多数有利，但 Q1 为 0/5；它更像一个稳定的 scalar slow-axis candidate，不是合格的多维 persistent state；
- E583 的 Q2–Q5 较好，但 Q1 和 Q6 为 0/5；
- E548 和 Yuquan zhangbichen 各有 3/5 seeds 通过 Q1，但没有联合通过 Q2–Q6；
- E1073 的 decoder state 基本坍缩。

这说明“seed 间看起来稳定”不等于“状态合格”。单一 decoder 轴、稳定 raw latent 对齐或长 generator time constant 都不能单独承担 H2b。

## 4. 半合成 assay 告诉了什么

旧 assay 草稿有三个会误导结论的问题：把 T 和 M 写成同一个量、用 raw latent 做 geometry、用同一批 null 同时定阈值和验收。旧 smoke 的 null 假阳性约 20%，却仍会选择 K=5，不能使用。

修复后：

- geometry 全部改在 frozen decoder-output metric 中计算；
- T 检验 state beyond current observation；
- M 用 outer-training fold 内拟合的 persistent-history residual，独立于 T；
- lag degradation 单独计算；
- null calibration 与 null evaluation 分离；
- 使用真实 interictal coverage、时钟支持、固定 seizure count 和状态自相关；
- 因 A1 为空，整个 assay 明确标记为 diagnostic-only。

7 个 worlds × 100 次 smoke 选择 K=2。独立 null 中，T 假阳性约 4–5%，M 约 6%；三种 geometry world 的识别率约 99–100%。但是 persistent-state world 的联合 `T+M+lag` 恢复率为 **0/100**。

所以这个 assay 能识别被显式注入的 basin、approach 和 abrupt geometry，却不能在当前只有 8 次模拟发作的真实支持结构中可靠恢复 transfer estimands。它是 `ASSAY_NOT_SENSITIVE` 的诊断结果，不是生物学阴性。

## 5. 状态是否胜过 recent IED history、当前 observation 或 memoryless

本轮不能给出 claim-bearing 答案。

原因不是模型比较漏跑，而是状态在接触真实 seizure outcome 前就没有通过 A1，assay 又没有恢复 T、M 和 lag 的能力。继续报告一个 hazard 数字会把“不合格 instrument + 无 power”误写成 seizure transfer。

曾有并发进程越过 gate：先生成三批 support-conditioned hazard/geometry 探索，之后又两次运行 full-grid hazard、geometry 和 phenotype。它们共放入 9 个独立 `quarantine/` 目录，均不进入结论、聚合或机器验收。保留这些文件只是为了审计发生过什么，不表示接受其科学路线。

full-grid 基础设施中的 anchor 和 persistent state 本身不读取 seizure outcome；但同一个序列化文件里的 wrong-time donor 索引会使用既定 ictal/postictal exclusion。这个 donor 部分只作未释放 probe 的准备，不可把整份 cache 笼统称为“完全不接触 seizure metadata”。

full-grid 状态提取本身已完成 46/46 cells、10 位有既定 seizure support 的患者、10,597 个唯一 anchors（跨 seed 为 45,841 行）。流式预处理后按内存预算使用 8 workers，36 份 RSS receipt 的最大单任务峰值约 1.48 GiB，最终运行没有 retry 或 OOM。它是可复用基础设施，不是 H2b 结果。

代码审查还发现，早期 hazard scaffold 把 T 和 M 写成了几乎同一个特征空间，且可能让 horizon 尚未完整观察的 row 进入 fold；早期 geometry scaffold 也把训练发作前窗口混进了 manifold fit。两处都已修复并加测试，但因为 A1/A2 没过，修复后的真实 seizure probe 仍不运行。这里的“代码修好”不能替代“科学问题可估计”。

## 6. correct-time、wrong-time 与提前量

不能作科学判断。

当前没有通过 A1/A2 的主路线，因此 correct-time vs wrong-time、`tau_z` lag-response 和 5/15/30/60 min risk curve 均未释放。旧 support-conditioned 数值只保留作代码诊断，不能证明正确时刻状态更接近发作。

## 7. 是否预测发作表现

没有运行 claim-bearing phenotype bridge，也没有重新聚类 seizure subtype。

在风险主路线尚未获得合格 state instrument 和敏感 assay 前，继续增加 recruitment、subtype 或 IED–ictal reuse heads 只会增加自由度，不能提升证据等级。

## 8. 对 H2b 的证据等级

当前等级是：

> **development-only gated negative closeout：H2b not established；当前 R1.7B checkpoint 的 transfer utility 不可估计。**

可以安全地说：

- v0.2 是工程合格但科学未建立的 negative pilot；
- v0.3 outcome-blind state qualification 在完整可读分母上为 0/16；
- 当前 assay 对 transfer estimand 不敏感；
- 现有结果不足以支持 transferable persistent interictal state。

不能说：

- 不存在共享 interictal–ictal state；
- state 不预测 seizure；
- 找到了 seizure mechanism、attractor 或 causal transition；
- 已有临床 seizure predictor；
- 已完成 formal held-out confirmation。

## 9. 下一步是否值得进入正式分区

不值得。formal 和 sealed 应继续关闭。

下一步不是给 R1.7B 增加更复杂的 seizure head，而是回到纯间期上游做 R1.8 slow/fast instrument redesign：先让 decoder-output state 在多个 seeds 中同时具有多维非坍缩、跨窗口 IED 增量、generator retention 和 nuisance-robustness，再重新校准能在真实 seizure support 下恢复至少 5% held-out log-loss gain 的 assay。只有这两步通过，才重新释放 nested hazard 和 OOS seizure-entry geometry。

## 10. 最终验收

- A0 attrition：完成；
- A1 75 cells / 16 patients：完成，0 state-qualified；
- A2 corrected diagnostic smoke：完成，未通过 power；
- A3–A8 claim-bearing route：按 gate 未释放；
- 九个目录中的越 gate 探索结果：已隔离；
- full-grid 状态基础设施：46/46 cells，8 workers，零 OOM；
- scoped tests：168 passed，5 个 PyTorch warning；
- machine audit：`PASS_GATED_NEGATIVE_CLOSEOUT_H2B_NOT_ESTABLISHED`。

这轮最重要的结果不是一个预测分数，而是把“当前 checkpoint 不够像状态”“assay 不够敏感”和“真实 H2b 为阴性”三件事重新分开了。
