# Topic 5 SIG-RNN v2：合同修订、合成校准与六患者 development 收口（2026-07-31）

> **Closure note（2026-07-31）**：本文件保留 v2 predictive screen 历史；其
> structure Gate 已由 v2.1 撤销并完成重判。当前结论见
> `stable_interaction_identifiability_v2_1_multiround_2026-07-31.md`。

## 1. 一句话结论

旧 v0.1 只能否定“输出不反馈、由第一 rank 初始化的确定性 autonomous latent
trajectory”，不能否定稳定 contact interaction。新 SIG-RNN 让生成 rank
通过共享 contact graph 反馈到下一步后，在六患者上相对完全匹配的 noGraph
模型同时改善 likelihood 和自由生成 precedence（6/6）；但相对在 development
test 上逐端点事后选择的 phase-matched Markov mixture / latent time template
oracle envelope，两个端点同时改善只有 1/6。因此：

> contact feedback 有可重复的预测增量；current single fixed graph 未取得已见
> 分布 predictive dominance。这个结果不能裁决 stable shared interaction
> structure，因为跨时间稳定性和可组合泛化尚未运行。

SNN 与 RNN 独立，本轮没有运行或读取 SNN。

## 2. v0.1 判决纠正

原 Round 5 不能判 G0：

1. pooled legacy SNN files 不满足同条件 nested event-count / `N_min` 合同；
2. 原产物明确写着 `full G0 remains open`；
3. first-rank lookup 对 source-only/sink-only/paired 的方向准确率已为
   `100%/100%/78.4%`；
4. NLL 选择简单模型，扰动方向 Brier 却选择 M4/M4-phase，两端点互相冲突。

机器判决现已改为：

- `G0 = REMOVED_FROM_RNN_GATE_NOT_EVALUABLE_FROM_ROUND5`；
- v0.1 M4 = `autonomous latent-trajectory null`；
- M4-phase vs M3 的 2/6、约 `0.010 nats/decision` = non-selection tie；
- v0.1 按其窄合同 100% bounded closure，不参与 human-to-SNN 映射。

## 3. 数据可辨识性审计

冻结输入仍为：

`results/topic5_interictal_rank_distribution/dataset_v0_4`

34 人结果：

- 34/34 有足够 generation 评分 decisions；
- 33/34 满足预注册 unseen-start support；
- event 数 447–140,337，中位 8,159.5；
- contact 数 6–52，中位 11.5；
- inner-train suffix decisions 中位 32,691.5；
- within-start suffix distance 中位 0.327；
- 34/34 保存对齐的连续 `event_lag_raw`。

`event_lag_raw` 是 legacy spectrogram-centroid time，不是 certified contact
peak time，因此 primary 仍使用 rank step。审计未读取 A/B、axis、SOZ、
geometry、ictal、IEI 或 SNN。

另一个模型合同修正是删除局部 refractory vector：在 contact 每场只招募一次
且永久从 candidate set 排除时，它只改变已排除 logits，数学上不可辨识。

## 4. 通用 synthetic feedback-graph 校准

### 4.1 首轮 G0-A

12-contact、2,400-event matched-family benchmark：

- SIG1 vs SIG0 NLL gain：中位 0.143 nats/decision；
- top-positive influence overlap：0.89–0.93；
- shuffle/lesion：3/3 变差；
- unseen-start NLL/precedence：3/3 改善；
- 全 132 条 influence Spearman：中位 0.684，低于冻结 0.75。

因此首轮保持 `FAIL_CLOSED`，没有追改阈值。

### 4.2 失败分解

- 每个 sender 有 706–1,542 次非终止暴露；
- fitted influence 跨 seed stability 为 0.93–0.96；
- 按 occupied empirical prefix states 平均后 recovery 提高到 0.734–0.747，
  仍未越过；
- top 50% / 25% effect 排序明显更好，缺口主要来自全 pair 的弱效应精细次序。

### 4.3 nested event-count curve 与独立确认

同一首轮图的 nested curve 显示，9,600 events 才首次使 3/3 seeds 越过原
threshold；这只用于设计新确认，不能回改 G0-A。

随后在运行前冻结全新 `seed=20260801` ring-plus-branch graph、全新 event
seeds、9,600 train events 和完全不变的阈值。独立 G0-A2 全项通过：

- influence Spearman 中位 0.807；
- SIG1 vs SIG0 NLL gain 中位 0.128；
- top-positive overlap 0.889；
- shuffle/lesion、unseen-start NLL 和 precedence 均 3/3 正确方向。

训练充分性终审后，G0-A2 以记录 pre-update validation、统一
`training_adequacy_verdict` 和保存 best optimizer state 的 v0.2 runner
重新执行；6/6 SIG0/SIG1 fits 均收敛，全部数值门和 PASS 标签不变。

这只证明最小反馈实现具有理想工程可辨识性，不是人类或 SNN 机制证据。

## 5. 六患者 graph-increment screen

六位 target-blind pilot、每人 3 seeds；old train80 内 train/monitor/test，
最多 9,600 train events，outer heldout20 未评分。

SIG0 与 SIG1 使用完全相同：

- train-only exact-ML static scaffold；
- phase basis `[φ, φ², sin(πφ)]`；
- exact conditional k-subset likelihood；
- candidate mask、cardinality schedule 和 checkpoint rule。

结果：

- SIG1 NLL 优于 SIG0：6/6；
- SIG1 free-rollout precedence MAE 优于 SIG0：6/6；
- 两项同一患者改善：6/6。

患者 NLL gain 为 0.056–0.409 nats/decision；precedence MAE gain 为
0.0029–0.0321。该结果只说明 contact feedback 相对 phase-only noGraph 有
增量，不是 G1。

终审发现旧 v0.1 SIG runner 只用 `best_epoch≥5` 代替完整训练充分性判据，
其中两次 52-contact noGraph run 的最佳点落在 epoch 292/297，不能称为已
平台化。修复后的 v0.2 从零重跑、记录 pre-update validation、在预算末仍下降
时自动扩展预算，并保存 best optimizer state；36/36 fits 全部通过统一判据。
上述 6/6 结果在修复后保持，因此不再依赖假收敛 baseline。

## 6. 同事件、同 phase 强基线

随后在相同事件和划分上拟合：

- M1 matched-phase first-order Markov；
- M2 matched-phase 3-component Markov mixture；
- M3 latent time-indexed template。

M1/M2 使用与 SIG 完全相同的 phase basis；M3 是更灵活的 event-latent smooth
time-template control。seeds 先在患者内折叠；旧聚合再分别在 development test
的 NLL 和 precedence 上取三个 family 的最小值。后一步形成 endpoint-specific
test oracle，只能保留为保守 stress test，不能代表可部署 baseline selection。

结果：

| 判据 | SIG1 更好 |
| --- | ---: |
| NLL vs 每患者最强 baseline | 3/6 |
| precedence MAE vs 每患者最强 baseline | 3/6 |
| 两项在同一患者都更好 | 1/6 |

旧 continue threshold 为两项同一患者至少 4/6，因此旧机器记录写成：

`SEEN_DISTRIBUTION_PREDICTIVE_DOMINANCE_NOT_ESTABLISHED`

这不是 stable-structure Gate。v2.1 撤销其对结构稳定性和 unseen-start 的锁定。

### 6.1 实际执行的判据与 spec §8 的差别（2026-07-31 复核补记）

实际跑的规则是：**每位患者 seed 折叠后，SIG1 的 held-out NLL/decision 与
free-rollout precedence MAE 都必须严格低于 M1-phase/M2-phase/M3 的逐指标最小
值；至少 4/6 患者同时满足。** 这条规则在三处比 spec §8 写的 G1 更严：

1. §8 写的是 conditional NLL **非劣**，执行时用的是**严格更优**；
2. §8 的 rollout 端比较对象是 **M2-phase 和 M3**，执行时用的是 M1/M2/M3 三者
   的逐指标最小值；
3. §8 没有患者计数门；4/6 是 §6.1 的 SIG0-vs-SIG1 **screen** continue rule，
   被沿用为 ladder 的 stop rule（现已写进
   `MATCHED_BASELINE_LADDER.json::decision_rule_provenance`）。

因为只往更严的方向偏，这条规则可作旧 predictive stress test，不能停止结构
辨识。为确认 predictive ranking 不是这个偏严读法造成的，按 §8 字面重算：

| §8 字面读法 | SIG1 满足 |
| --- | ---: |
| NLL 非劣（margin 0.01 nats/decision） | 4/6 |
| NLL 非劣（margin 0.03） | 6/6 |
| rollout 同时优于 M2-phase 与 M3 | 3/6 |
| 两端点在同一患者同时满足（margin 0.01） | 2/6 |

即使放宽到非劣读法，同一患者两端点同时满足最多 2/6，仍低于旧 4/6 门。这只
说明 current SIG 没有取得普遍 interpolation dominance。

另外，§8 的 G1 还有一句 **“无 mode collapse 或明显 over-dispersion”**：本轮
**没有计算任何 within-start 离散度诊断**，plan Task 3 把“过/欠分散诊断”列为
主输出也未产出。因此准确表述是 **G1 在阶梯前置条件处停住**，不是“G1 被完整
评估后未通过”；`g1_clauses_not_evaluated` 已写入机器产物。

### 6.2 逐个 baseline 的分解

主表只报“对最强 baseline”，掩盖了增量的来源。逐个对手的患者计数为：

| 对手 | NLL 更优 | precedence 更优 | 两项同患者 |
| --- | ---: | ---: | ---: |
| M1-phase（同量级 first-order Markov） | 5/6 | 5/6 | 4/6 |
| M2-phase（3 分量 mixture） | 4/6 | 3/6 | 2/6 |
| M3（latent time template） | 3/6 | 4/6 | 2/6 |

逐患者 NLL 最优模型：SIG1 3 人、M2-phase 2 人、M3 1 人、M1-phase 0 人。

因此更准确的机制陈述是：**把已发出的 contact 通过共享 graph 反馈回去，稳定
胜过同量级的无记忆 first-order Markov（5/6、5/6、两项同患者 4/6）；输的是更
灵活的离散 mixture 与事件级 latent 模板。** 这仍然不构成 shared stable graph
的结构辨识证据，因为 mixture/template 用完全不同的机制取得了同等或更好的
已见事件拟合，而真正区分 structure 的 stability / compositional tests 尚未做。

顺带说明 SIG0 的量级：`learn_graph=False` 时 `W≡0`，状态从零出发恒为零，模型
退化为 **static scaffold + phase head**，比 M1-phase 还弱。所以 §5 的 6/6
“增量”对照的是无任何转移项的模型，不能读成“反馈胜过 Markov”。

强基线 v0.2 的 54/54 fits 均通过同一训练充分性聚合门。
M2 的三分量在 18/18 runs 中均保持分离（最小 component-parameter distance
3.03），排除了“同值初始化后永不分开”的旧 bug。M1/M2 正式 NLL 为 exact；
M3 为有限样本 IWAE marginal-likelihood estimate，free rollout 只用
future-blind prior，因此 M3 没有因偷看未来而获得生成优势。
precedence 的个别差值很小，但 NLL 端最多只有 3/6 患者超过全部强基线；
因此即使额外 rollout 令所有 precedence 边缘值都向 SIG 翻转，同一患者两项
同时改善也最多 3/6，仍不可能达到旧 4/6 predictive screen。这不影响 v2.1
重开结构特异实验。

这不是“反馈无用”：SIG1 对 noGraph 的 6/6 增量仍成立。它只说明 current
single graph 没有在 seen-distribution scoring 上普遍支配 mixture/template。

## 7. Gates 与安全口径

| Gate | 状态 |
| --- | --- |
| generic synthetic engineering | PASS at 9,600 events on independent graph |
| human graph increment | PASS 6/6 |
| seen-distribution predictive stress test | predictive dominance not established |
| D0 patient-matched identifiability | OPEN / NOT RUN |
| D1 baseline/envelope/diversity audit | OPEN / NOT RUN |
| D2 M2 observable-operator audit | OPEN / NOT RUN |
| D3 structure stability | OPEN / NOT RUN |
| D4 human unseen-start | OPEN / NOT RUN |
| D5 shared-backbone modulation | LOCKED pending D0 and D3/D4 signal |
| G5 full cohort / replication | LOCKED / NOT RUN |
| SNN Gate | ABSENT BY CONTRACT |

当前允许写：

> 在六患者 development 数据中，生成 contact 通过共享 contact-space graph
> 反馈到后续状态，相对匹配 phase-only noGraph 改善完整 suffix likelihood 和
> 自由生成 precedence；但仅依靠 sampling 表达多样性的单一固定 graph 未在
> 已见分布上稳定超过 phase-matched mixture 或 latent time template。该结果
> 不裁决 stable shared backbone，因为跨时间结构稳定性和可组合泛化尚未检验。

不得写：

- 已辨识患者特异稳定 interaction graph；
- one structure, many trajectories 已成立；
- 已恢复 effective connectivity；
- SNN 验证、否定或 gating 了 RNN。

## 8. 论文地位与下一步

v0.1 是窄的 autonomous-trajectory bounded null；v2 当前是
`feedback increment without predictive dominance` 的 development result。
v2.1 已重开 patient-matched identifiability、structure stability 与
human unseen-start；它们使用当前模型，不增加 event drive、process noise、
multiple fields 或 hidden width。

只有患者尺度 synthetic 证明可辨识，且 stability 或 compositional
generalization 出现 real-over-null 信号，才允许另立 shared-backbone modulation
合同。否则停止 RNN structure interpretation。

## 9. 产物

- v2 spec：
  `docs/superpowers/specs/2026-07-31-topic5-stable-interaction-graph-rnn-v2.md`
- v2 plan：
  `docs/superpowers/plans/2026-07-31-topic5-stable-interaction-graph-rnn-v2.md`
- data audit：
  `results/topic5_stable_interaction_graph/development/identifiability_audit/`
- synthetic rounds：
  `results/topic5_stable_interaction_graph/development/synthetic_g0a*/`
- graph increment：
  `results/topic5_stable_interaction_graph/development/human_graph_increment_pilot_v0_2_training_adequacy/`
- matched ladder：
  `results/topic5_stable_interaction_graph/development/human_matched_baseline_ladder_v0_2_training_adequacy/`
- machine verdict：
  `results/topic5_stable_interaction_graph/development/SIG_V2_DEVELOPMENT_STATE.json`
