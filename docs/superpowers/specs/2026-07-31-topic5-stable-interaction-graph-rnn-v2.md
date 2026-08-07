# Topic 5：Stable Interaction Graph RNN v2 科学与数据合同

> **Gate correction（2026-07-31）**：本合同的模型实现和既有运行保持有效，
> 但 §8–§11 中把 seen-distribution NLL/precedence dominance 作为 G2–G4
> 前置停止门的顺序已被撤销。逐患者、逐 endpoint 在 development test 上取
> M1/M2/M3 最小值形成的是事后 oracle stress test，只能判定当前 single-graph
> 形式没有 predictive dominance，不能裁决 stable interaction structure。
> 后续结构判别以 v2.1 合同为准：
> `2026-07-31-topic5-stable-interaction-identifiability-v2_1.md`。

## 0. 科学对象与独立性

合同名：`topic5_stable_interaction_graph_rnn_v2`

本合同只回答：

> 同一患者的大量间期 rank events，是否足以自监督辨识一个跨事件稳定的
> contact-space effective interaction graph；固定该 graph，仅改变起点、
> 逐步传播采样和必要时的低维事件驱动，是否能够生成患者内多样 repertoire？

RNN 与 SNN 独立成立。SNN 不提供训练数据、不提供先验、不承担正对照，也不
出现在任何 RNN Gate。未来二者只能在各自冻结后作 secondary comparison。

v0.1 的 M4 统一改称 `ALT-null`（autonomous latent-trajectory null）。它的
emitted contacts 不反馈 latent state；其阴性结果不用于判决本合同。

## 1. 数据与禁止输入

主任务仍为：

`p(x_{2:T} | x_1, k_{2:T}, T)`。

第一阶段条件化真实第一 rank、事件长度和逐步 cardinality，只辨识“经过哪些
contacts、以什么顺序”。训练和正式评分禁止读取：

- A/B、KMeans/pathological-axis、SOZ、发作数据、IEI；
- 物理坐标、电极距离、SNN artifact；
- event identity；
- 当前时刻之后的 contact。

likelihood 训练可以读取真实过去 `x_1,...,x_t`，因为这是自回归联合概率的
精确分解，不是未来泄漏。正式 free rollout 从 `x_1` 开始，此后只反馈模型
自己采样的 rank sets。

旧 outer heldout20 已参与历史分析，v2 development 不将其称为独立确认集。
所有开发选择只在旧 train80 内的 chronological train/monitor/test 完成。

## 2. 共享 interaction 与状态更新

对患者 `p`：

- `b_p ∈ R^C`：只从 inner train 极大似然估计并冻结的 static scaffold；
- `W_p ∈ R^{C×C}`：跨所有事件共享的 directed effective interaction；
- `h_t ∈ R^C`：contact-space propagation state；
- `u_e ∈ R^q`：仅在数据要求时开放的低维 event drive。

方向合同固定为：

`W[i,j] = contact j 在当前被招募后，对 contact i 后续招募倾向的影响`。

最小更新：

```text
h_{t+1} = lambda * h_t + tanh(W @ x_t + B @ u_e)
eta_{t+1} = b + h_{t+1} + P @ psi(phi_t)
x_{t+1} ~ exact k-subset softmax(eta_{t+1}, A_{t+1}, k_{t+1})
```

其中：

- `diag(W)=0`；
- `lambda ∈ (0,1)`；
- 已招募 contact 永久移出 candidate set；
- `P @ psi(phi)` 是所有相关模型共享的低秩 phase nuisance；
- `W` 使用 row/spectral control 与稀疏 shrinkage，但不强制低秩、对称或空间局部；
- raw `W` 是 secondary representation，主结构对象是可执行干预得到的
  observable influence。

主模型不含局部 refractory vector：在“每个 contact 每场最多出现一次、已
招募 contact 永久排除、cardinality 已条件化”的合同下，`-gamma*r_t` 只改变
已排除 contact 的 logit，因而数学上不可辨识。只有未来开放重复招募或
cardinality/STOP dynamics 时才重新定义 adaptation。

## 3. 分阶段模型

| 代号 | 模型 | 科学作用 |
| --- | --- | --- |
| M0 | static | participation scaffold |
| M1-phase | phase-conditioned first-order | 非平稳局部转移 |
| M2-phase | phase-conditioned mixture | 少量离散路径 |
| M3 | latent time template | event template，无 recurrent feedback |
| SIG0 | noGraph | 与 SIG1 相同 phase/state，固定 `W=0` |
| SIG1 | feedback graph | shared `W` + emitted-contact feedback |
| SIG2 | feedback graph + event drive | 固定 `W`，增加低维连续事件驱动 |
| ALT-null | v0.1 autonomous trajectory | 历史 bounded null，不参与 v2 主 Gate |

开放顺序：

1. 默认只训练 SIG0/SIG1；
2. 仅在同一 first-rank 下 SIG1 明显 under-dispersed 时开放 `q=1,2,3` 的 SIG2；
3. 仅在固定 `x_1,u_e` 后仍 under-dispersed 时开放 process noise；
4. 不通过多 fields、更宽 hidden state 或 architecture zoo 挽救主张。

## 4. 观测概率与 phase 公平性

沿用 v0.1 已验证的 exact conditional `k`-subset likelihood、candidate mask
和 without-replacement sampler。所有包含 phase 的模型使用同一冻结
`psi(phi)` basis、相同信息集和同一评分 estimator。

graph 的增量只能定义为：

`Score(SIG1) - Score(SIG0)`，

以及 SIG1 相对 M1-phase/M2-phase/M3 的完整事件表现。phase head 的优势不得
写成 graph 证据。

## 5. 可观测结构对象

主对象为：

```text
I_ij^(tau) =
P(j at t+tau | do(i active at t))
- P(j at t+tau | matched state, do(i inactive at t))

I_eff = sum_tau discount^(tau-1) I^(tau)
```

干预必须：

- 在相同 state、candidate set、phase、cardinality schedule 和随机数下配对；
- 只改变当前 contact input；
- 报告可达/不可达 pair mask；
- 同时保存 horizon-specific 和 integrated influence。

正式稳定性比较使用 `I_eff`，不使用 raw latent coordinates。raw `W` 仅用于
工程恢复检查和 secondary interpretation。

## 6. 与 SNN 无关的 G0 合成校准

G0 使用固定的通用 directed stochastic graph process，不预设 A/B：

- 12 个 contacts；
- 多个起点、重叠通道和至少三个随机分支；
- 留出一个起点，但其 contact 在训练中作为中间节点出现；
- phase nuisance 同时存在；
- 训练只见 rank events，不见真 `W`。

在查看结果前冻结 G0-A（matched-family engineering benchmark）通过条件：

1. 3/3 fit seeds 完成且无 NaN、mask/gradient/provenance 失败；
2. SIG1 held-out NLL/decision 每 seed 比 SIG0 至少低 `0.02`；
3. 真值与估计 `I_eff` 的 off-diagonal Spearman：每 seed `≥0.60`，中位
   `≥0.75`；
4. top-20% 正向 influence pair 的 overlap coefficient 每 seed `≥0.50`；
5. learned graph 的 paired shuffle/lesion 使 NLL 或 precedence fidelity
   在 3/3 seeds 变差；
6. 未见起点上 SIG1 的 NLL 与 precedence fidelity 均优于 SIG0。

这些阈值只证明代码和最小模型在理想已知系统中可工作，不是人类机制证据。
G0-A 将 `I_eff` 的 horizon 冻结为 `L=1`，避免用额外 rollout Monte Carlo
决定工程 Gate；人类结构分析再按独立合同扩展到 `L=1,2,3`。
若 G0-A 失败，停止人类 pilot，先定位实现/可辨识性问题。G0-A 通过只允许进入
六患者 development pilot；任何人类结构 claim 前仍需多 graph/mild-
misspecification 的 G0-B。

### 6.1 G0-A 首轮失败后的冻结补充（运行独立确认前写入）

首轮 2,400-event G0-A 保持 `FAIL_CLOSED`：唯一失败项为全 132 条
off-diagonal `I_eff` 排序中位 `0.684 < 0.75`。post-failure 诊断显示 sender
支持充足、seed stability `0.93–0.96`、top-positive overlap `0.89–0.93`；
嵌套事件数曲线在 9,600 events 首次使同一阈值 3/3 seeds 越过。

因此允许一次版本化的 G0-A2 工程确认，且必须在运行前冻结：

- 使用由独立 seed `20260801` 产生的 ring-plus-branch graph；
- 使用全新 train/validation/test event seeds；
- train events 固定为 9,600，不再继续增量追阈值；
- 三个 fit seeds 保持不变；
- G0-A 的全部数值阈值原样保持；
- G0-A 原始失败 artifact 不覆盖、不改标签。

只有 G0-A2 全部通过，才把结论写成
`ENGINEERING_CALIBRATION_PASS_AT_N_MIN_9600` 并允许六患者工程 pilot。该
`N_min` 只属于 12-contact generic synthetic graph，不外推为人类结构样本门。
G0-A2 失败则继续 `BLOCK_HUMAN_PILOT`，不得再开第三次同类确认。

G0-A2 已按上述冻结合同全项通过。六患者首先只做 SIG0-vs-SIG1 的
graph-increment screen；它不是 G1。进入完整 M1-phase/M2-phase/M3 阶梯前，
患者内 3 seeds 折叠后必须同时满足：

- SIG1 NLL/decision 优于 SIG0：至少 4/6；
- SIG1 free-rollout precedence MAE 优于 SIG0：至少 4/6；
- 两项在同一患者同时改善：至少 4/6。

该 4/6 是 development continue/stop rule，不作小样本显著性陈述。未达到则
停止 SIG human line；达到只允许补齐强 baseline ladder。

## 7. 人类数据可辨识性审计

每位患者必须输出：

- contact/event/transition-decision 数；
- first-rank 类型和重复次数；
- 相同 first-rank 下的 suffix unique fraction、entropy 与 sampled distance；
- contact 作为起点、中间节点和非终止 sender 的覆盖；
- chronological halves 的 transition support；
- leave-one-start-group-out feasibility；
- rank/cardinality 分布；
- `event_lag_raw` 的存在、单位、有限性和语义。

unseen-start 资格预先定义为：

- 至少 4 个 start groups 各有 `≥20` 个 inner-train events；
- 至少一个可留出 start group 的全部起始 contacts，在剩余训练事件中各作为
  中间节点出现 `≥20` 次；
- 剩余训练集每个可估 sender contact 有 `≥20` 个非终止暴露；
- validation/test 各至少 100 个 suffix decisions。

不合格患者仍可参与 generation adequacy，但不得进入 unseen-start 或
structure-stability cohort claim。

冻结 dataset 中的 `event_lag_raw` 是事件内 spectrogram-centroid time，不是
certified contact peak time。v2 primary 使用 rank step；连续时间只能作另行
预注册的 sensitivity。

## 8. 人类 Gates

> 本节原版本由 v2.1 supersede。G2/G4 是结构假设的直接检验，不再从属于
> seen-distribution predictive superiority。

- G1 完整事件充分性：SIG1 对最强 baseline 的 conditional NLL 非劣，且
  free-rollout repertoire distance 同时优于 M2-phase 和 M3；无 mode collapse
  或明显 over-dispersion。
- G2 结构稳定性：independent chronological halves 和 seeds 的 `I_eff`
  稳定性高于 phase-conditioned Markov、rank-wise phase shuffle 和 static
  surrogate。
- G3 one structure, many trajectories：固定 `W`，仅改变 `x_1`、subset
  sampling 和必要时低维 `u_e` 即重现 within-start diversity。
- G4 unseen-start：在审计合格患者中，SIG1/SIG2 在完全留出的 start group
  上优于 mixture/template lookup。
- G5 独立确认：G1–G4 冻结后，才在第二数据集或全新 cohort 确认。

开发 pilot 统计单位是患者；seeds 先在患者内折叠。六患者的 2/6、3/6 等小差
不得写成统计失败或成功，只用于 stop/continue。

## 9. 停止规则与安全结论

- G0-A 不通过：不运行人类 pilot；
- SIG1 未超过 SIG0：不能声称 graph 有增量；
- SIG1 只赢 likelihood、不赢 free rollout：只能写局部自回归拟合改善；
- 只有 event-specific `W_e` 才能拟合：不能声称 shared structure；
- G1–G4 未全过：不扩全队列、不使用 outer heldout20 做确认；
- 任何阶段都不调用 SNN 作为救援或解释依据。

成功时允许写“重复自发事件足以辨识稳定 effective interaction constraints”；
不得写“恢复了突触连接”或“患者脑内通过学习形成了该网络”。

## 10. 可复现性

每次运行保存 config/source/input/split/checkpoint SHA、完整学习曲线、best
optimizer state、训练充分性判决、四个泄漏标志和 free-rollout artifact。
聚合器必须 fail closed：拒绝缺 run、混 source/config、未收敛、未来数据读取
或模型间不一致的信息集。

「混 source/config」包含三种形态，缺一不可：per-run artifact 根本没有
provenance 字段；各 run 之间互相不一致；各 run 一致但与**正在聚合的这份
源码**不一致（编辑 runner 后只重跑聚合，会把新 hash 盖到旧 fits 上）。
实现见 `src.topic5_stable_interaction_graph.uniform_provenance`；聚合产物必须
分别记录 `fit_time_source_sha256` 与 `aggregation_source_sha256`，不得用单一
字段代表两者。

## 11. Development 执行结论（2026-07-31）

合同已执行到冻结停止点：

- generic synthetic calibration：独立 graph/event seeds、9,600 train events、
  原阈值不变时通过；首轮 2,400-event G0-A 的失败标签原样保留；统一训练充分性
  v0.2 重跑的 6/6 fits 均收敛；
- human SIG1-vs-SIG0 screen：36/36 fits 训练充分，NLL 与 free-rollout
  precedence 同时改善 6/6；
- matched strong-baseline ladder：SIG1 相对每位患者最强
  M1-phase/M2-phase/M3，NLL 改善 3/6、precedence 改善 3/6、两者同一患者改善
  1/6；54/54 baseline fits 训练充分；
- 因预先冻结的 continue threshold 为两端点同一患者至少 4/6，
  `G1=NOT_PASSED_DEVELOPMENT`；该 4/6 是 §6.1 的 SIG0-vs-SIG1 screen
  continue rule 被沿用为 ladder stop rule，且执行时把 §8 的“NLL 非劣”收紧为
  “严格更优”、把 rollout 对手从“M2-phase 和 M3”扩为 M1/M2/M3 逐指标最小值。
  三处都只往更严偏，所以这条规则**能停线不能放行**；按 §8 字面的非劣读法
  （margin 0.01）重算，两端点同患者同时满足也只有 2/6，停止结论不依赖读法。
  §8 中“无 mode collapse 或明显 over-dispersion”一句**本轮未评估**（没有产出
  任何 within-start 离散度诊断），因此准确说法是 G1 停在阶梯前置条件，不是
  被完整评估后未通过；机器产物以 `g1_clauses_not_evaluated` 记录该边界；
- 原执行把 G2–G4 写成 `LOCKED_NOT_RUN`；该锁在 v2.1 中撤销。当前准确状态是
  `STRUCTURE_NOT_ADJUDICATED`：允许在同一六患者 development 数据上运行
  patient-matched calibration、split stability 和 unseen-start，但仍禁止全队列、
  outer heldout20、SIG2、process noise 或 multiple fields；
- `snn_gate=ABSENT_BY_CONTRACT`，本合同没有运行或读取 SNN。

这不是 shared graph 不存在的证据。它只说明在当前六患者 development 的已见
分布任务中，single fixed graph 虽比无 graph 的匹配模型好，却没有取得相对
phase-conditioned mixture/template 的 predictive dominance。是否存在稳定共享
backbone 尚未由跨时间稳定性、matched-null specificity 或可组合泛化裁决。
