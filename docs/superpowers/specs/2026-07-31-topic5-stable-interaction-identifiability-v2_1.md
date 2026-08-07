# Topic 5：Stable Interaction Identifiability v2.1 科学合同

> **2026-08-01 closure**：本合同已按
> `SIG_V2_1_IDENTIFIABILITY_STATE.json` 正式验收并冻结。v2.2 改用 event index
> 作为长期时间轴，属于新科学对象，不是本模型的调参或容量扩展。

## 0. 修订原因

v2 正确加入 emitted-contact feedback，却再次用非结构特异的
seen-distribution NLL/precedence 排序裁决 stable structure，并把真正具有机制
区分力的 split stability 与 unseen-start 锁在其后。另一个错误是分别在
development test 上为 NLL 和 precedence 选择 M1/M2/M3 最小值，形成不可部署的
endpoint-specific oracle envelope。

v2.1 撤销这两个 Gate。既有运行不作废，但只支持：

1. feedback 相对 matched noGraph 有增量；
2. current single fixed graph 未取得 seen-distribution predictive dominance；
3. stable shared interaction structure 仍为 `NOT_ADJUDICATED`。

## 1. 冻结科学问题

> 重复间期事件是否约束出跨时间稳定、超越 static/phase/template null、并能在
> 未见起点或未见组合上复用的 contact-level effective interaction structure？

本轮不新增 event drive、process noise、multiple fields 或更宽网络。只有当前
SIG1、M1-phase、M2-phase、M3 与匹配 surrogate 通过结构特异诊断后，才允许另立
`shared backbone + low-dimensional modulation` 合同。

## 2. 既有 predictive screen 的正确地位

- SIG1 vs SIG0 的 6/6 NLL 与 precedence 改善：`FEEDBACK_INCREMENT_PRESENT`；
- test endpoint-specific oracle envelope 的 1/6：
  `SEEN_DISTRIBUTION_PREDICTIVE_DOMINANCE_NOT_ESTABLISHED`；
- oracle envelope 只作保守 stress test，不是 G1/G2/G3/G4；
- baseline family 必须由 inner validation 选择一次，并用同一 family 同时评分
  untouched development test 的 NLL 与 repertoire；同时逐模型报告 SIG-vs-M1、
  SIG-vs-M2、SIG-vs-M3，不再拼接 endpoint-specific oracle。

不根据已见 test 差值事后定义 non-inferiority margin。v2.1 先报告连续效应量、
seed/rollout uncertainty 和 empirical half-to-half scale，再冻结后续 margin。

## 3. D0：patient-matched identifiability

通用 12-contact、9,600-event synthetic pass 只证明实现可工作。对六位 pilot
分别匹配：

- contact 数、训练事件数、rank-count 与 cardinality schedule；
- first-rank 分布、participation imbalance、candidate support；
- 与真实数据一致的 chronological split 和最大训练上限。

必须同时评估：

1. shared-graph positive：恢复 observable `I_eff`；
2. phase/template-only negative：不得产生超越 null 的稳定 graph；
3. mixture-Markov negative：不得把多个无共享 backbone 的 operator 误称为
   single stable graph；
4. event-specific random-graph negative：不得产生假 shared structure。

D0 是 sensitivity + specificity 校准，不以拟合 NLL 代替 structure recovery。
在某患者匹配规模下不可分辨时，该患者不能进入 human structure interpretation。

## 4. D1：未来 event-envelope 与 baseline-selection 审计

在不重训模型前完成：

1. 逐患者逐模型 SIG1−M1/M2/M3 NLL 和 repertoire 效应量；
2. inner-validation-selected 单一 baseline family 的 untouched-test 结果；
3. 标记旧 endpoint-specific test oracle 仅为 stress test；
4. 比较 `X1+T` 与 `X1+T+k_{2:T}` 对 M2 posterior route identity 的预测；
5. 计算真实及生成事件的 within-start dispersion、entropy、unique fraction；
6. 审计 full schedule shuffle 与 current-`k_t`/standardized schedule 的可行性。

若未来 schedule 单独高度预测 route，当前任务只能称为
`conditional path reconstruction given future envelope`。

## 5. D2：M2-phase 结构审计

M2 不只作 nuisance baseline。对现有三个 seeds：

- 在 observable contact influence 空间匹配 component permutation；
- 报告 component 与 component-mean backbone 的跨 seed 稳定性；
- 报告 posterior occupancy、occupancy entropy 与 first-rank/schedule 可预测性；
- 不比较 raw transition matrix 的直接相关作为主结果。

随后只在 D1/D2 有信息时运行 chronological early/late 独立重训。若 component
operator 跨 seed/split 稳定且共享 backbone，再考虑 modulated graph；参数距离
非零本身不算稳定性证据。

## 6. D3：结构稳定性主 Gate

对当前 SIG1 独立拟合 chronological early/late halves，主对象为 supported-state
observable `I_eff`，不是 raw `W`。比较：

- early vs late；
- seed vs seed；
- real events vs phase-conditioned Markov surrogate；
- real events vs phase/template surrogate；
- contact-label permutation 与 static/phase-only response。

必须先扣除 static scaffold 和 phase-only response。只有 real-minus-null stability
在患者内方向一致，才称为 development structural signal；六患者计数只作内部
方向性 triage，不作 cohort 显著性结论。

## 7. D4：可组合泛化主 Gate

对审计合格患者留出一个完整 first-rank group；被留出的 contacts 必须在其余
训练事件中作为 intermediate sender 有预注册支持。使用相同 inner train/monitor
规则，从头拟合 SIG1、M1-phase、M2-phase、M3。

主要比较：

- unseen-start suffix NLL；
- free-rollout precedence 与 distributional distance；
- start lookup / `X1+T+k` envelope-only control；
- seen-start performance 作为校准，不作主 Gate。

若可行，再留出 start × terminal/prefix-transition 组合。该实验用于区分可组合
interaction 与 template lookup，不要求 SIG 在普通 seen-start interpolation 上
全面获胜。

## 8. D5：多样性组织与模型升级条件

只有 D0 证明患者尺度可辨识，且 D3 或 D4 至少一项出现 real-over-null 信号，才
允许新建：

`W_e = W_0 + sum_r a[e,r] * DeltaW_r`。

届时必须嵌套比较 `R=0`、continuous low-rank modulation、one-hot stable regimes、
`W0=0` route-specific graphs 和 time template。当前不实现这些模型。

## 9. 决策矩阵

| 结果 | 决策 |
| --- | --- |
| patient-matched D0 不可分辨 | 停止该患者 structure interpretation |
| D0 可分辨，但 D3/D4 均不超过 null | 停止 single-graph structure line |
| SIG seen interpolation 略弱，但 D3/D4 明确 | 保留 shared graph，先冻结结构指标 |
| M2 regimes 稳定且存在共享 backbone | 另立 modulated-backbone 合同 |
| 仅 route/template 稳定，无共享 backbone | 结论为少量稳定 regimes，不声称 one structure |

完整 34 人、outer heldout20、SNN comparison 与独立确认在上述 development
结构 Gate 冻结前均禁止。SNN 不参与任何 RNN Gate。

## 10. 安全表述

当前允许写：

> Contact feedback improved conditional suffix generation relative to a
> matched no-feedback model. A single fixed graph did not achieve predictive
> dominance over flexible route/phase controls on the seen distribution;
> stable shared interaction structure remains unadjudicated pending matched
> identifiability, temporal stability, and compositional generalization tests.

不得写“stable graph 已失败”“shared structure 不必要”或“mixture/template 证明
没有 network structure”。

## 11. 防止再次走错分支的硬规则

本轮错误重复的根因不是某个阈值写错，而是把“模型在一个任务上赢”继续当成
“结构对象存在”的替代证据。后续所有 RNN 合同必须逐项通过以下审查：

1. **claim–endpoint 必要关系**：先写清楚某 endpoint 为什么能区分 stable
   structure 与 phase/template/local-transition；不能区分的只叫 predictive
   diagnostic，不能成为 structure Gate。
2. **主证据不得被前置代理锁死**：temporal stability、matched-null specificity
   和 compositional generalization 本身是主假设，不能因普通 NLL 未胜而不运行。
3. **test 不参与 comparator 选择**：baseline family、margin、threshold 和停止规则
   只由 inner train/validation 或 synthetic calibration 冻结；test 只评分一次。
4. **正负校准同时存在**：任何“恢复了结构”的指标必须同时在患者匹配 positive
   中有 sensitivity、在 template/mixture/event-random negatives 中有 specificity。
5. **增加容量需要结构特异前因**：只有 D0 通过且 D3/D4 至少一项 real-over-null
   时，才允许 event drive、process noise、multiple fields 或 modulation；否则
   增加容量属于拟合救援，不属于科学假设推进。
6. **稳定参数不是稳定结构**：raw weight、seed reproducibility 或 absolute
   split correlation 均不足；主结果必须是 observable operator 的
   real-minus-matched-null stability。

## 12. 执行收口（结果后追加，不回改冻结阈值）

- D0 patient-matched calibration：4/6 PASS；另外两位分别为 sensitivity 不足和
  fixed-vs-mixture specificity 不足；
- D1：validation-selected baseline 下两端点同时 2/6；future schedule route
  increment 中位接近 0；无 collapse、主要为轻度 over-dispersion；
- D2：M2 component/backbone seed stability 高，但仅作描述；
- D3：real-minus-strongest-null 为 0/6 正方向，在 D0 PASS 患者中为 0/4；
- D3 sensitivity：改用未参与 saved checkpoint 选择的 inner-test probe 后仍为
  0/6 正方向，中位 real-minus-null 从 `−0.079` 变为 `−0.072`，判决不变；
- D4：unseen-start NLL 5/6 正方向，precedence 2/6，两项同时 2/6；
- D5：`NOT_AUTHORIZED`；full cohort / replication 保持锁定。

因此 current single fixed graph 仅在 4 位可辨识患者中形成 bounded negative；两位
D0 未通过患者保持未裁决。该结果不外推到所有 stable/modulated structures。
