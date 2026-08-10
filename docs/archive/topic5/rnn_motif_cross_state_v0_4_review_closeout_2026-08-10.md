# Topic 5 RNN connectivity motif / cross-state v0.4 审阅后收口报告

> 状态：代码与统计已按审阅意见修复并重算；冻结模型未重训；等待用户终审后再 commit。

## 一句话结论

这轮已经可靠证明：患者内间期 contact-rank 序列含有可被 recurrent network 学会并自由生成的有序传播信息，多种 dense、sparse、local 和 spatial recurrent topology 都足以完成该计算；Spatial + cost 只用约 10% 的边、约 4.9% 的总几何布线长度，仍保持相近的传播表现。

但冻结模型场到发作早期场的证据仍未闭合。当前 10 人交集中，第一 rank/source 对跨状态一致性有稳定贡献；recurrence 是正向但未确认；真实顺序和 wiring cost 没有显示额外跨状态增量。内部网络只支持“局部有效影响富集”，不支持“关键长程 connector 已被辨识并经特异干预验证”。

## 1. 本次审阅后到底修了什么

本次没有改训练目标、网络权重、模型选择或 early-ictal endpoint，因此不需要重跑 1,426 个训练单元。修复集中在五类会影响解释的地方：

1. 重建 strict target 16 人 → primary 15 人 → exact model–target intersection 10 人的逐患者排除链；
2. 明确 `FIELD_CANONICAL_FULL` 才是冻结 spec 的跨 Human–RNN–SNN primary，`FIELD_SEED_REMOVED` 是 recurrence-specific key secondary；
3. 将 early-ictal 对应拆成 source、static scaffold、recurrence、true order 和 wiring cost；
4. 将训练中的平均 active-edge cost 与总几何长度、总强度加权布线量分开；
5. 用 Kendall τb、归一化 rank 误差和参与集合 Jaccard 补查 free-rollout Spearman 的 0.5 平台是否掩盖模型差异。

新增 7 个回归测试，覆盖 field 分解、fit-first aggregation、布线口径与 frozen-table parity、smoke 排除、rollout seed removal、target LOO reliability 和 README 持久化；`tests/test_topic5_rnn_motif_v0_4.py` 当前 30/30，通过两份相关测试共 50/50。

队列排除审计所用的上游几何清单已复制为
`results/topic5_rnn_motif_cross_state_benchmark_v0_4/REVIEW_SOURCE_GEOMETRY_MANIFEST.json`，
其 SHA256 同时写入 `REVIEW_ATTRITION_AUDIT.json` 和
`REVIEW_CLOSEOUT_AUDIT_COMPLETE.json`。后续重跑不依赖产生该清单的旧 worktree。

最终状态写入也已修正为互斥：工程验收成功时删除旧 `PIPELINE_FAILED.json`，失败时删除旧 `PIPELINE_COMPLETE.json`，避免历史失败标记与当前成功标记同时存在。

## 2. 队列为什么是 10 人，不是 15 人

冻结的 strict early-ictal target 有 16 人；E1146 在设计阶段已固定为 development/supportive，因此 primary 预期为 15 人。实际进入模型—target 比较的是 10 人、24 次 seizures、每患者 8–16 个精确评分触点。

缺失的 5 人不是评分脚本静默漏掉，也不是 contact 名称临时 join 失败：

| 患者 | event contacts | 冻结几何 exact joint contacts | 排除原因 |
|---|---:|---:|---|
| E1077 | 6 | 6 | 少于冻结门槛 8 |
| E1096 | 7 | 7 | 少于冻结门槛 8 |
| E1125 | 8 | 7 | HR11 无几何，joint 少于 8 |
| E139 | 7 | 7 | joint 少于 8；同时平面质量较差 |
| E635 | 10 | 7 | HL2/HL8/HL11 无几何，joint 少于 8 |

所以，在不改变当前模型 cohort 合同的前提下，这 5 人不能“补回”。现在事后把门槛从 8 降到 6/7，会改变物理坐标模型的输入定义，应另作预先声明的 sensitivity，而不能并入 v0.4 primary。

## 3. Q1：RNN 是否学会了间期传播

### 3.1 已冻结的主结果

Spatial + cost 相对 no-recurrence：

- next-contact NLL 改善 0.1493，21/21 患者同向；
- 删除白送第一 rank 后的自由推演 rank correlation 提高 0.3286；
- true order 相对 order-shuffle 的 NLL 改善 0.1460，21/21 同向。

因此可以写：

> 患者内自监督 recurrent models 学会了可从第一 rank 自由生成的间期传播规律，而不只是触点参与频率。

不能写成恢复了真实脑连接组；多种网络拓扑都能产生相似可观察传播。

### 3.2 0.500 rollout 平台是不是指标假象

审阅后从 2,349,312 条冻结 rollout 记录中，对每个 fit/seed 均匀抽取最多 128 条，共重评 164,726 条，registered Spearman 主指标不变，只增加诊断：

| 模型 | 患者中位 Kendall τb |
|---|---:|
| No recurrence | 0.087 |
| Dense | 0.333 |
| Local | 0.351 |
| Spatial + cost | 0.341 |
| Order shuffle | 0.128 |

Spatial + cost 相对 no-recurrence 的 τb 增量中位为 +0.284，18/21 为正；相对 order-shuffle 为 +0.200，18/21 为正、0 负。说明原来的 0.500 平台确实较离散，但“真实 recurrent computation 能生成传播、shuffle 不能”的结论不依赖 Spearman 量化。

## 4. Wiring economy 应该写到什么程度

训练日志中的 `c_wiring` 精确定义是：

\[
C_{\mathrm{mean-edge}}
=
\frac{1}{|E|}
\sum_{ij\in E}
|w_{ij}|\frac{d_{ij}}{10\,\mathrm{mm}}.
\]

它是 active-edge 平均代价，不是总布线量。重算 1,246 份 `graph.npz` 后，与训练日志最大绝对误差为 6.04×10⁻⁷。

相对 dense，Spatial + cost 的患者内中位比例为：

| 资源量 | Spatial + cost / Dense |
|---|---:|
| active edge count | 10.0% |
| total geometric length | 4.89% |
| total strength-weighted length | 3.38% |
| mean-edge normalized cost | 33.7% |

因此“约 5% total wiring”只允许指纯几何总长度；Panel D 现在明确标为 mean active-edge strength × distance / 10 mm。wiring economy 的安全结论是：它能用显著更少的连接资源保持任务表现，但它不是癫痫网络形成的病理哲学，也没有显示独立的 early-ictal 优势。

审阅建议的 weight-only regularization 和 distance-permuted cost 两个新训练对照本次没有追加。原因不是忽略替代解释，而是当前收口已经主动撤回“患者真实几何带来特异优势”的主张，只保留资源—性能 trade-off；这两个对照只会继续拆解 generic wiring regularization，不会闭合 Q2 或 Q3。若以后仍要主张 geometry-specific wiring benefit，它们必须补做；下一阶段 LBSS 则直接取消长边距离惩罚，改测少量任务选择长程 pathway。

## 5. Q2：冻结间期 RNN field 是否跨状态对应

### 5.1 先纠正 endpoint 口径

冻结 spec 从一开始规定：

- `FIELD_CANONICAL_FULL`：跨 Human–RNN–SNN primary；
- `FIELD_SEED_REMOVED`：判断后续生成是否超越白送起点的 key secondary。

本次没有交换 primary。审阅报告中“当前 RNN primary 更强调 seed-removed”的表述不符合冻结合同。

### 5.2 early-ictal target 自己是否可靠

在 10 位 primary 中，8 人有至少 2 次 seizures，可做 leave-one-seizure-out：

- 患者级中位 Spearman ρ=0.351；
- 8/8 为正；
- Wilcoxon P=0.0078；
- bootstrap 95% CI=[0.200, 0.625]；
- 另外 2 人各只有 1 次 seizure，无法估计患者内可靠性。

所以当前 early-ictal target 有可重复空间成分，不是纯测量噪声；但可靠性中等、每患者只有 8–16 个触点，仍明显限制模型间细微差异的功效。

### 5.3 经验场、静态场和 RNN 场

canonical-full、同步 all-contact null 下：

| 场 | null-relative margin 中位 | 正向患者 | P |
|---|---:|---:|---:|
| Empirical interictal field | +0.0675 | 7/10 | 0.0840 |
| Static/no-recurrence | +0.0108 | 7/10 | 0.2754 |
| Order-shuffled recurrent | +0.0744 | 8/10 | 0.0371（未作该行独立主张） |
| True-order Spatial + cost | +0.1104 | 6/10 | 0.1602 |

这里最重要的不是某一行 P<0.05，而是 paired decomposition：

| 冻结场增量 | 中位 | 正/负 | Holm q（本次四项 review family） |
|---|---:|---:|---:|
| Source：full − seed-removed | +0.0197 | 9/1 | 0.0234 |
| Recurrence：M6 − no-rec | +0.0713 | 7/3 | 0.3164 |
| True order：M6 − order-shuffle | +0.0131 | 6/4 | 1.000 |
| Wiring cost：M6 − spatial-no-cost | −0.00047 | 4/6 | 1.000 |

因此当前跨状态对应的可靠解释是：

> 第一 rank/source 对 early-ictal 空间对应有稳定贡献；recurrence 的数值方向为正，但当前 n=10 未确认；真实 rank order 和 wiring cost 没有显示额外跨状态增量。

这与 Q1 不矛盾。真实顺序对生成间期传播非常重要，但当前 early-ictal benchmark 主要读取的是更宽尺度的患者空间 scaffold，而不是精细 rank-to-rank recurrence。

已有的患者固定效应模型也给出同一边界：控制 interictal field fidelity 后，Spatial + cost 相对 no-recurrence 的模型效应为 +0.0504，patient-cluster bootstrap 95% CI=[−0.0342, 0.1519]，置换 P=0.0573；相对 dense 的估计反而为 −0.0143，区间跨零。它是未解决的正向趋势，不是“接近显著”或 connectivity-specific transfer。

### 5.4 GRU 的边界

GRU 也进入了冻结 early-ictal scorer。其 Spatial + cost canonical-full margin 中位为 +0.1022，7/10 为正，P=0.0645；相对 GRU no-recurrence 的预先汇总差为 +0.0969，Holm q=0.328。方向与 leaky RNN 一致，但同样未确认，也没有证明某种 connectivity motif 跨 cell family 特异胜出。

可以写“间期 autoregressive learnability 可从 leaky RNN 复现到 GRU”；不能写“early-ictal cross-state transfer 已 architecture-general”。

## 6. Q3：内部网络到底支持什么 motif

Spatial + cost 的患者级结果：

- local effective influence enrichment 中位 +0.0461，18/21 为正，P=1.34×10⁻⁵；
- 同一冻结模型在 train split-halves 上的 effective operator 稳定性中位 ρ=0.980；
- 不同随机 seed 间完整 operator 稳定性只有中位 ρ=0.142；
- long-range high-influence enrichment 不成立；
- motif score 与任务表现不稳定相关。

所以“局部影响的统计组织”跨患者稳定，但精确 edge-weight operator 并没有跨 seed 收敛到唯一答案。这再次说明网络级充分性强、逐边可辨识性弱。

Matched lesion 的真正状态是：

- local backbone：5 位患者满足完整匹配合同，4 正/1 负，Holm q=0.9375；
- connector nodes：7 位，4 正/3 负，Holm q=0.9375。

因此状态已改为：

```text
MATCHED_LESION: INCONCLUSIVE_DUE_TO_MATCHING_ELIGIBILITY
```

不是 lesion 失败，更不是已经建立了 causal connector motif。主图 Panel F 现在只展示承重更强的 local enrichment 与 operator stability；n=5/7 的 lesion 移到探索图并直接标分母。

## 7. 修订后的图

- 主六联图：`results/topic5_rnn_motif_cross_state_benchmark_v0_4/figures/topic5_figure6_rnn_connectivity_motifs.png/.pdf/.svg`
- 审阅收口诊断：`.../figures/topic5_rnn_motif_review_closeout.png/.pdf`
- 探索性 matched lesion：`.../figures/topic5_matched_lesion_exploratory.png/.pdf`

主图的变化：

1. Panel D 轴名不再把 mean-edge cost 混写成 total wiring；
2. Panel E 加入 empirical field、static、order-shuffle 和 true-order RNN 的同尺度比较；
3. Panel F 改为局部有效影响及跨 seed/train-half 稳定性；
4. matched lesion 从主图移出，并显式标出 n=5/7。

## 8. 当前最终允许写 / 不允许写

可以写：

> Patient-specific interictal contact-rank sequences contained robust ordered information that recurrent networks could learn and freely generate. Multiple recurrent topologies were computationally sufficient, while wiring-cost constraints preserved propagation fidelity with markedly reduced wiring resources. Task-trained networks showed stable local effective organization.

跨状态部分只能写：

> Frozen recurrent fields showed a positive but statistically unresolved correspondence with early-ictal broadband energy. The supplied first rank contributed reliably to this correspondence, whereas order-specific recurrence and wiring cost did not show confirmed incremental effects in the current 10-patient intersection.

不能写：

- RNN 恢复了患者真实连接组；
- Spatial + cost 是癫痫特异的最佳 topology；
- 精确 interictal rank order 已被证明在发作早期复用；
- local backbone + long-range connector 已经建立；
- 非显著等于达到数据天花板或证明等价。

## 9. 收口判决

```text
Q1_INTERICTAL_GENERATIVE_SUFFICIENCY:       CLOSED_POSITIVE
Q2_EARLY_ICTAL_CROSS_STATE_TRANSFER:        POSITIVE_TREND_NOT_CONFIRMED
Q3_LOCAL_EFFECTIVE_ORGANIZATION:            SUPPORTED
Q3_LONG_RANGE_CONNECTOR_MOTIF:              NOT_ESTABLISHED
MATCHED_LESION:                              INCONCLUSIVE_LOW_ELIGIBILITY
V0_4_ENGINEERING_CLOSEOUT:                   COMPLETE
READY_FOR_USER_FINAL_REVIEW:                 YES
COMMITTED:                                   NO
```

下一阶段不应继续围绕 generic wiring economy 加模型，而应单独检验“固定局部 recurrent backbone + 少量任务选择的长程 pathway”。对应 LBSS-RNN spec/plan 与本报告同时提供，但在用户审阅前不启动。
