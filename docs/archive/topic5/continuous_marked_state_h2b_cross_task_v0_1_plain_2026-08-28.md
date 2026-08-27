# H2b Cross-task Transfer v0.1：白话报告

## 一句话结论

这轮已经证明“只用间期任务训练、冻结后再读出发作风险”的整套仪器可以严格运行，但 **E384 的人体 development pilot 不支持 H2b**。在预先固定的 30 min 主提前量上，加入 persistent state 后的 conditional log loss 比当前 observation 高 `+0.1560`；persistent state 也比 memoryless code 高 `+0.1876`。两者都是正值，表示更差而不是更好。R1.7 已完成机器审计，但尚未形成 clean committed release，所以本轮没有队列结论。

## 1. 我们是否真的只从间期任务学状态

是。状态来自 R1.6 的 `epilepsiae_384` 稳定 seeds 1、3、4，三个 checkpoint 的 SHA256 均已复算一致。observer、state update、generator 和 IED decoder 在 seizure probe 前全部冻结；seizure label 只用于构造 query、排除 ictal/postictal 时间和评分，不进入状态更新，也没有 seizure gradient 回到状态模型。

因果提取只使用 `t <= s-h` 的窗口。状态在每个真实 coverage segment 起点重置，绝对时间用 float64。真实扰动检查把 anchor 后 10 个 observation 窗全部替换成极端值，anchor 前 13 个输出字段逐位不变。

## 2. 多少患者和发作可分析

当前只有一个 instrument patient：E384，不能承担 cohort claim。

- SQL 中 15 次发作全部唯一 crosswalk，onset 时间差均为 0 秒。
- development partition 中有 5 次发作。
- 5/15/30/60/120 min 有完整记录覆盖的发作数为 `5/5/4/2/1`。
- 同期有新鲜 frozen observation、最终可提取 state 的发作数为 `5/5/4/2/0`。
- 30 min 主提前量只有 4 次，按冻结规则只能是 descriptive case series。
- 120 min 虽有 1 次完整记录覆盖，但没有合格的当前 observation，因此不可估计。

旧训练 cache 因 seizure guard 在 5/15/30/60/120 min 只剩 `2/2/2/2/0`。本轮没有把这个训练 guard 错当成冻结后推理分母；推理使用 hash 验证的 0.70 有效接触阈值，30 min 四次发作的 observation age 为 `10.383/2.875/4.626/1.991 s`。

## 3. 状态是否胜过 recent IED history

没有。在 30 min，`B_history` 的 log loss 是 `1.6191`，`B_state` 是 `1.7637`；加入 observation 和 persistent state 后反而高 `+0.1446`。这不是“recent IED history 已经正式解释了一切”的队列证据，只是 E384 的描述性结果不支持 latent-state 增量。

## 4. 状态是否胜过当前 observation

没有。主比较 `B_state − B_observation = +0.1560`，方向与 H2b 所需的负值相反。当前 observation 相对 history 只改善 `−0.0115`，在 4 次发作上也不能单独宣称稳定的 preictal observation signal。

## 5. persistent 是否胜过 memoryless

没有。30 min 的 `persistent − memoryless = +0.1876`，说明这次 pilot 中持续状态没有超过只看当前 observation 的 code。

## 6. correct-time 是否胜过 wrong-time

没有。严格 wrong-time donor 必须同患者、同 coverage segment，并避开目标发作及 ictal/postictal 排除窗。30 min 只有 2 次发作满足该子合同；`correct − wrong-time = +0.8629`，正确时刻状态更差。这个 n=2 子集只能作描述，不能独立裁决时间特异性。

## 7. 多早能看到信息

`B_state − B_observation` 在 5/15/30/60 min 分别为 `−0.0508/−0.0727/+0.1560/+0.0546`。短提前量方向为负，30 和 60 min 为正，且 5/15 min 是 sensitivity、不能用来事后替换 30 min 主端点。persistent 对 memoryless 在 5 min 也为正，仅 15 min 为负。因此当前没有一致的“信息提前出现时间”。

## 8. 是否预测发作表现

不可估计。E384 没有预先冻结的 seizure subtype；既有盲法 onset-contact 登记为 `0/71`。现有 ictal cache 只覆盖 4 次主发作中的 2 次，而且保存的是逐通道信号数组，不是预先定义的 seizure-level recruitment extent。为了避免看过 state 后再造 target，本轮没有重新聚类，也没有用 SOZ、患者 focus、模板端点或最高能量触点顶替。

## 9. 对 H2b 的证据等级

- 工程层：`COMPLETE`，所有机器检查通过。
- E384 科学层：development descriptive，方向不支持 30 min H2b。
- 队列层：未运行。R1.7 的 `machine_audit.json` 已为 `COMPLETE`，50 个 result、审计列出的 45 个 checkpoint 及其 source/checkpoint hashes 均通过只读复核，formal/sealed 也均为 false；但其工作树仍有未提交文件，尚未形成 clean committed release，因此 H2b 没有加载这些模型或把它们作为分析输入。
- 正式层：未开启，`formal=false`、`sealed=false`。

因此当前最安全的结论是：**冻结间期状态的跨任务读取仪器已建立；E384 没有显示相对 recent history、当前 observation、memoryless 或 wrong-time 的 30 min 增量；是否存在队列级 transfer 仍未检验。**

## 10. 下一步是否值得进入正式分区

不值得直接进入正式分区。下一步只应在 R1.7 同时具备最终 COMPLETE machine audit、50/50 fits、可复算 source/checkpoint hashes、clean commit 和 push 后，接入全部 checkpoint-available 患者；不能只挑 H1 阳性患者。先完成 development cohort 的 patient-first 估计，若 30 min 的 state-vs-observation、persistent-vs-memoryless 和 correct-vs-wrong-time 三条方向一致，再讨论预注册正式分区。

## 机器证据

- 结果根：`results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_1/`
- 最终审计：`reports/machine_audit.json`
- E384 主结果：`fits/e384_instrument/patient_median_probe_metrics.csv`
- wrong-time 子集：`fits/e384_wrong_time_instrument/patient_median_probe_metrics.csv`
- phenotype 边界：`reports/e384_phenotype_availability.json`
- R1.7 边界：`reports/r1_7_availability.json`

禁止把本轮写成 seizure mechanism、latent attractor、state causes seizures、clinical predictor 或 formal held-out confirmation。
