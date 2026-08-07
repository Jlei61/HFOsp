# Topic 5 v0.1：多轮判别审阅、判决纠正与 bounded closure（2026-07-31）

原始合同：
`docs/superpowers/specs/2026-07-30-topic5-shared-propagation-field-rnn-v0_1.md`

机器可读总判决：
`results/topic5_shared_propagation_field/development/multiround_review_2026-07-31/MULTIROUND_VERDICT.json`

SNN 只读兼容性检查：
`results/topic5_shared_propagation_field/snn_positive_control/existing_artifact_system_identification/`

## 1. 一句话结论

给定第一 rank、事件长度和逐 rank cardinality，完整 suffix 中确实有超出静态
participation 和 stationary first-order Markov 的组织；但六患者 development
数据没有选择出输出不反馈的 deterministic autonomous latent trajectory。M4 在完整条件似然和
precedence 两个 endpoint 上均不超过带进度的 Markov mixture 或时间模板，
10%–100% 事件数学习曲线、`d={2,4,6}`、估计器校准和非塌缩诊断都没有改变
这个排序。该结果不评价显式 contact interaction graph。

因此：

- 历史 SNN G0：Round 5 不可判，移出 RNN Gate；
- v0.1 主门：stop rule reached / autonomous null 未被选择；
- contact-structure Gate：v0.1 未定义该科学对象，因此未开放；
- one-structure-many-trajectories：不属于 v0.1 判决；
- 34 人扩展：停止。

这是对**由第一 rank 初始化、输出不反馈的 deterministic-in-state M4**
的 bounded negative，不是“重复事件中没有可辨识稳定结构”或“患者/SNN
没有稳定网络”的结论。

## 2. 审阅边界

- 人类数据仍限六名 target-blind development patients、三个 fit seeds。
- 旧 outer heldout20 不进入训练、checkpoint 选择或评分。冻结 NPZ 物理上把
  train80 与旧 heldout20 共置，因此这里只能作“未用于分析”的保证，不能再写
  “字节级从未读取”。
- A/B、病理轴、SOZ、几何、发作数据和 IEI 均未进入人类模型。
- 本轮没有调用 SNN simulator；只复用既有 event/readout/figdata artifacts。
- 人类 Round 1–4、6–7 复用同一 development pool/test partition。它们排除
  不同替代解释，但不是七次独立统计复制。

## 3. 本轮修复

### 3.1 SNN 方向读出

历史 subject-SNN runner 通过修改 imported module global 试图改变
`read_event` 的默认 `k_dir/part_min`，但 Python 默认参数在函数定义时已经
绑定。现已：

1. 在调用 `read_event` 时显式传入 `k_dir` 和 `part_min`；
2. 将 participant floor 统一为估计器真实合同 `2*k_dir+1`；
3. 从保存的 ranks 和 coordinates 以显式 `k_dir=2, eps=2 mm` 重算方向；
4. 同时保留历史 `event_direction_sign_reported`，不再混用两种口径。

显式重算后的既有数据为：

| family | model-ready | forward | reverse | unreadable | 历史 reported |
| --- | ---: | ---: | ---: | ---: | ---: |
| source-only | 180 | 158 | 0 | 22 | 98 / 0 / 82 |
| sink-only | 230 | 0 | 217 | 13 | 0 / 110 / 120 |
| paired source/sink | 222 | 103 | 119 | 0 | 103 / 119 / 0 |

这不改变 SNN 机制口径：方向由低阈值 pathological kernel/core 的位置和身份
产生；E→E 各向异性塑造传播通道，`AR=1` 本身不是方向消失 null。

### 3.2 训练充分性与 provenance

Round 3 共 `108 shards × 4 models = 432 fits`，Round 6 共
`36 shards × 3 models = 108 fits`。初次运行有 4 个 fit 未通过充分性 gate；
修复器只从原 deterministic initialization 以预声明 lower-LR ladder 重训这
四个 fit。最终全部为 `CONVERGED`。

机器汇总在写总判决前强制检查：

- 108/108 learning-curve shards；
- 36/36 dimension shards；
- 所有 free-parameter fits 训练充分；
- 每轮 source SHA 与当前脚本一致；
- v0.4 config SHA 与 frozen state 一致。

## 4. 七轮判别结果

### Round 1：likelihood estimator 不是排序来源

把 latent likelihood 的样本数提高到 256 后：

- M4 − M3 importance NLL/decision：中位 `+0.0321`，M4 胜 `0/6`；
- M4 − M3 pure-prior：中位 `+0.0320`，M4 胜 `0/6`；
- M4-phase − M3 importance：中位 `+0.0102`，胜 `2/6`；
- importance ESS fraction 中位 `0.880`；
- 128→256 samples 的绝对变化中位 `0.00031`，最大 `0.00476`。

posterior 只作 importance proposal；正式生成仍来自 future-blind prior。
排序不是单次 Monte Carlo 抖动造成的。

### Round 2：旧的“长事件时钟混淆”不是患者内规律

旧跨患者诊断中，M3–M4 差距与患者平均事件长度相关；但在患者内逐事件重算：

- M4 − M3 prior gap 与 rank count 的 Spearman 中位 `−0.080`；
- M4-phase − M3 prior 的 Spearman 中位 `+0.0069`。

进度时钟解释了部分差异，但没有解释全部差异；旧 `rho≈−0.62` 是跨患者
ecological association，不能作为 M4 失败的充分解释。

### Round 3：小样本归纳偏置没有救回 M4

使用同一个 target-blind random ordering，所有 fraction 都是严格 nested
prefix；monitor 和 development test 固定。M4 − M3 的结果为：

| 训练 fraction | 中位 ΔNLL/decision | M4 胜 |
| ---: | ---: | ---: |
| 10% | +0.0143 | 0/6 |
| 20% | +0.0255 | 0/6 |
| 40% | +0.0214 | 0/6 |
| 60% | +0.0288 | 0/6 |
| 80% | +0.0303 | 0/6 |
| 100% | +0.0346 | 0/6 |

相对 M2-phase，M4 在六档中分别只胜 `0,1,1,1,1,1/6`。因此没有证据支持
“field 只在小样本端体现更好的归纳偏置”。这一 learning curve 只回答样本
效率，不能反转 full-data G1。

### Round 4：高 seed stability 主要不是 field-specific

完整 observable response 的跨 seed correlation 都很高：

- M2-phase `0.988`；
- M3 `0.980`；
- M4 `0.981`；
- M4-phase `0.978`。

扣除每个 run 自己的 M0 static response 后：

- M2-phase `0.912`；
- M3 `0.918`；
- M4 `0.832`；
- M4-phase `0.882`。

M4 的 residual fidelity 也低于 M2/M3（`0.646` vs `0.656/0.701`）。
生成 entropy ratio 为 `1.165`，说明 M4 偏过度分散，不是 mode collapse。
因此“不同 seed 都稳定”主要受 static scaffold 支撑，不能作为 stable field
证据。

### Round 5：既有 SNN 兼容性检查不能判 G0

只用 paired family 的 SNN seeds 1–15 训练、16–18 monitor、19–21 test；
方向标签和几何不参与拟合或 checkpoint 选择。paired test NLL/decision：

| model | NLL/decision |
| --- | ---: |
| M0 | 1.536 |
| M1-phase | **0.113** |
| M2-phase | 0.128 |
| M3 | 0.244 |
| M4 | 0.528 |
| M4-phase | 0.512 |

方向/扰动迁移端点的实际排序为：

| model | direction Brier | sink-only forward fraction（真值约 0） |
| --- | ---: | ---: |
| M4-phase | **0.087** | **0.105** |
| M4 | 0.090 | 0.110 |
| M1-phase | 0.117 | 0.216 |
| M3 | 0.128 | 0.158 |
| M2-phase | 0.139 | 0.269 |

自主模型在原 G0 定义的扰动响应端点上最好，却在 NLL 上最差。更重要的是，
一个只根据第一 rank set 查表的诊断在 paired test 上已有 `78.4%`
direction accuracy；source-only/sink-only 为 `100%`。而 source/sink rollout
又直接获得被扰动 family 的第一 rank、长度和 cardinality schedule，所以
这里只是 OOD conditional reuse，不是 blind lesion prediction。

此外，这 21 个 legacy files 跨 seed/条件 pooling，没有同条件 nested
event-count curve，也没有建立 `N_min`。这违反 Phase 0 冻结的数据准备要求。

因此 Round 5 对 G0 的两个方向都不可判：

> NLL 不能宣布结构未恢复；较好的 Brier 也不能宣布结构已恢复。Round 5 仅是
> exploratory compatibility check，SNN 从 RNN Gate 中删除。

### Round 6：`d={2,4,6}` 不改变 autonomous M4 排序

M4 − M3：

| latent d | 中位 ΔNLL/decision | M4 胜 |
| ---: | ---: | ---: |
| 2 | +0.0471 | 0/6 |
| 4 | +0.0319 | 0/6 |
| 6 | +0.0298 | 0/6 |

M4-phase 在 `d=6` 有很小的中位优势（`−0.0026`, 4/6），但它读取显式 phase。
主配置中 M4-phase vs M3 为 2/6、中位 `+0.010`，正式记为未被选择的平局，
不能写成统计方向性失败。

### Round 7：M4 不是因为完全没动而输

冻结 checkpoint 的 future-blind prior-mean 诊断显示，M4：

- best-epoch raw KL/event 中位 `0.697`；
- 总 latent state displacement 中位 `4.44`；
- temporal logit SD 中位 `0.395`；
- `alpha` 中位 `0.356`。

所以 latent code 和 recurrent trajectory 都被实际使用。该结果只排除
“posterior/trajectory 完全塌缩”的工程解释；非零 dynamics 本身不是
identifiable field 的证据。

## 5. 原始 co-primary endpoint 复核

v0.4 主表中：

- M4 − M2-phase NLL：`+0.0347`，M4 胜 `0/6`；
- M4 − M3 NLL：`+0.0319`，M4 胜 `0/6`；
- M4 − M2-phase precedence correlation：中位 `−0.0457`，M4 胜 `0/6`；
- M4 − M3 precedence correlation：中位 `−0.0394`，M4 胜 `1/6`。

M3 的参数较 M4 多（患者中位 `5206` vs `3847`），所以不能把 M3 的胜出写成
机制证明；但参数远少的 M2-phase（中位 `297`）也在 NLL 和 precedence 上
6/6 超过 M4，容量差异不能解释主 gate 的失败。

## 6. Gate 与论文地位

| Gate | 判决 | 含义 |
| --- | --- | --- |
| 历史 SNN G0 | REMOVED / NOT EVALUABLE | Round 5 违反同条件 `N_min` 要求且被 first-rank lookup 混淆 |
| v0.1 主停止门 | STOP RULE REACHED | M4 不超过 M2-phase/M3，停止扩展这一 null |
| contact structure | NOT OPENED | v0.1 没有输出反馈或 contact intervention object |
| one structure, many trajectories | OUT OF SCOPE | 需由新 SIG 合同检验 |
| 34 人扩展 | STOP | 不运行正式队列，不让旧 outer heldout20 进入分析 |

当前只允许写：

> 给定第一 rank、长度和 rank cardinality，完整 suffix 含有超出 static
> scaffold 和 stationary first-order Markov 的组织；但六患者 development
> 数据没有选择由第一 rank 初始化、输出不反馈的 deterministic autonomous
> latent-trajectory model。该结果不检验或排除稳定 contact interaction graph。

不得写：

- 已恢复患者特异传播场或有效连接；
- one structure, many trajectories 已成立；
- M3 是脑内真实时间模板；
- 当前结果否定 SNN 的 kernel heterogeneity 机制；
- 七轮诊断等于七个独立 cohort 复制。

论文地位固定为窄的 bounded null / Extended Data 或方法学审计。RNN 与 SNN
独立开发，v0.1 不参与二者机制映射。

## 7. 后续决策

不再通过更宽 hidden state、更多 GRU、mixture-of-fields 或发作 readout 挽救
当前 M4。若未来重开逆问题，应视为新模型合同，至少需要：

1. 定义可观测、可执行的 contact/network intervention operator；
2. 优先考虑连续 latency/field，而不是只保留 rank sets；
3. 若加入 process noise 或多 field，重新预注册其必要对照；
4. 以与 SNN 无关的通用 synthetic feedback graph 做代码和可辨识性 sanity
   check；SNN 作为独立生物物理分析线，不再是 RNN 进入门。

当前 v0.1 没有过程噪声、没有 mixture-of-fields、也没有 contact lesion
operator；本报告没有检验这些更宽的模型族。
