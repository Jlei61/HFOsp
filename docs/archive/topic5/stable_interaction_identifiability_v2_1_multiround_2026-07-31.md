# Topic 5 RNNv2.1：稳定 interaction 可辨识性五轮修订与收口（2026-07-31）

## 1. 三段式白话结论

**测了什么。** 同一患者反复出现的 rank events，是否真的能选择出一个跨时间
稳定、可跨起点复用、并且高于 static participation、phase、局部一阶转移和
time template 的 contact interaction graph。这里不再把普通 test NLL 排名当成
结构存在与否的裁判。

**怎么测的。** 先修正旧 comparator：旧结果是在 development test 上分别为 NLL
和 precedence 挑最小 baseline，是 endpoint-specific oracle，只保留为 stress
test。随后不增加模型容量，依次完成：(1) validation 只选一个 baseline、未来
cardinality schedule 与生成多样性审计；(2) M2 observable operator seed stability；
(3) train-only 留出完整 first-rank group 的 unseen-start 重训；(4) early/late
独立 SIG influence 与 phase-Markov / latent-template surrogate 的 real-minus-null
比较；(5) 按每位患者真实 contacts、事件数、T、cardinality 和起点支持构造
fixed-graph positive 及三类 negative synthetic，校准 sensitivity 和 specificity。

**揭示了什么。** Contact feedback 相对 matched noGraph 仍有 6/6 增量；但它没有
形成稳定的结构特异证据。patient-matched calibration 只有 4/6 患者能同时区分
fixed graph 与 phase、stable mixture、event-random negatives；在这 4 位可辨识
患者中，真实 early/late influence stability 0/4 超过 matched null。全部六位也是
0/6 real-minus-null 为正。Unseen-start NLL 对 validation-selected baseline 为
5/6 改善，但 precedence 只有 2/6，两端点同时 2/6。因此当前 single fixed graph
分支按 bounded negative 收口；另外两位因 calibration 不足保持未裁决。没有证据
授权 event drive、process noise、multiple fields 或 shared-backbone modulation。

## 2. 为什么会重复 v0.1 的错误

这次模型对象比 v0.1 更正确：生成 contact 会反馈进 contact-space state，SNN 也已
从 RNN Gate 删除。但证据逻辑仍继承了旧 ladder：

1. 把“在已见分布上赢过灵活 baseline”当成“稳定结构存在”的代理；
2. 用 development test 分别选择 NLL 和 precedence 的最强 baseline，形成不可部署
   的 oracle envelope；
3. 把真正区分 structure 的 temporal stability 与 unseen-start 锁在预测 Gate 后；
4. 没有先做患者尺度 positive + negative calibration，就用绝对稳定性解释内部图；
5. 把模型设计更接近科学问题，误当成评价指标也自动具有机制区分力。

所以重复错误的根因不是网络结构写错，而是 **claim–endpoint 逻辑没有重写**。
v2.1 已把该错误写成合同级禁令：test 不选 comparator；预测指标不能锁死结构主证据；
所有结构指标必须同时通过 patient-matched sensitivity 和 specificity；没有 D0 与
D3/D4 信号就禁止增加容量。

## 3. 第一轮：既有产物与未来 event-envelope 审计（D1）

### 3.1 comparator 修正

逐模型结果保持不变：

| 对手 | SIG NLL 更好 | SIG precedence 更好 | 两项同时 |
| --- | ---: | ---: | ---: |
| M1-phase | 5/6 | 5/6 | 4/6 |
| M2-phase | 4/6 | 3/6 | 2/6 |
| M3 latent template | 3/6 | 4/6 | 2/6 |

只用 inner-validation seed-median NLL 选择一个 baseline family，并用同一 family
评分两个 untouched test endpoints 后，六位都选择 M2-phase；SIG 为 NLL 4/6、
precedence 3/6、两项同时 2/6。旧 1/6 只能称 endpoint-specific test-oracle stress
test，不能称 Gate。

### 3.2 future schedule 与 diversity

使用 `X1+T` 与 `X1+T+k_{2:T}` 预测 M2 posterior route identity；完整 schedule
增加的 balanced accuracy 中位为 `−0.0004`，仅 2/6 为正，没有发现未来 schedule
直接携带 route identity 的证据。

现有 free rollout 不是 mode collapse。SIG within-start pair-distance 相对真实事件
为 0.97–1.12，5/6 略高于 1，主要是轻度 over-dispersion。这个结果限制的是当前
sampling law，不支持用增加 process noise 作为下一步。

## 4. 第二轮：M2 observable operator（D2）

对 occupied prefixes 移除一个 sender，同时固定 phase、candidate set 和其他同 rank
senders；比较下一 contact 概率，得到 contact-space observable operator。经 component
permutation matching 后：

- component seed-stability cohort median：`0.974`；
- component-mean backbone seed-stability median：`0.990`；
- shared-backbone energy fraction median：`0.662`。

这是值得记录的描述性线索，但不是稳定 regime 证据。后续 D3 证明 matched local /
phase null 也能产生同等或更高的稳定性，因此不能依据 seed reproducibility 启动
modulated-backbone 模型。

## 5. 第三轮：unseen-start compositional generalization（D4）

每位患者只用 inner train 选择事件最多且在其余训练事件中仍有至少 20 次
intermediate-sender 支持的 start group。该 group 从 train 和 monitor 全部剔除；
若 untouched test 支持不足则 fail closed，不会改选 test 更有利的起点。

五个模型均从零重训，90/90 fits 充分，源码、配置和 split provenance 一致。

| patient | held-out start | SIG NLL gain | SIG precedence gain |
| --- | --- | ---: | ---: |
| epilepsiae_922 | GC2 | 0.021 | −0.008 |
| epilepsiae_620 | HR5 | 0.013 | 0.025 |
| epilepsiae_1096 | HL3 | 0.005 | −0.002 |
| yuquan_zhangkexuan | E1 | 0.029 | −0.014 |
| yuquan_chenziyang | G1 | −0.022 | −0.024 |
| yuquan_zhangjiaqi | H7 | 0.102 | 0.049 |

OOD NLL 有 5/6 正方向，但完整 precedence 只有 2/6；这是一条混合信号，不支持
shared graph 的完整 repertoire 可组合优势。

## 6. 第四轮：chronological real-minus-null stability（D3）

分别在 inner-train early/late halves 独立拟合 SIG；所有模型在同一 common probe
上计算 supported marginal sender response。static scaffold 和 phase head 在干预
两臂相同，因此被算子差分消除。Matched null 使用冻结 M1-phase 与 M3 生成器，
保留 first rank、T 和每步 cardinality。

108/108 fits 充分。结果：

| patient | real stability | strongest matched-null | real−null |
| --- | ---: | ---: | ---: |
| epilepsiae_922 | 0.575 | 0.924 | −0.349 |
| epilepsiae_620 | 0.496 | 0.970 | −0.474 |
| epilepsiae_1096 | 0.839 | 0.951 | −0.111 |
| yuquan_zhangkexuan | 0.818 | 0.866 | −0.047 |
| yuquan_chenziyang | 0.978 | 0.987 | −0.009 |
| yuquan_zhangjiaqi | 0.966 | 0.971 | −0.005 |

真实 absolute stability 常常很高，也全部高于 contact-label permutation；但 0/6
超过 matched local/phase null，中位 real−null `−0.079`。因此 absolute stability
不能解释为高于局部统计的 shared structure。

原 common probe 来自 inner-validation，而既有 M1/M3 generator 的 checkpoint 也用
inner-validation 选择，存在对 null 略有利的可能。为排除这一点，保存的 early/late
SIG checkpoint 不重训，改用未参与这些 checkpoint 选择的 inner-test 事件作为共同
probe，并重新从冻结 M1/M3 generator 产生匹配 null。结果仍为 0/6 正方向，中位
real−null `−0.072`（逐患者：`−0.409, −0.550, −0.110, −0.033, −0.004,
−0.004`）。因此 D3 阴性不依赖原 probe 与 checkpoint-selection 的重叠。由于该
inner-test 已在此前 development 中被查看，这仍是 sensitivity，不是独立确认。

## 7. 第五轮：patient-matched sensitivity + specificity（D0）

每位患者使用真实 contact 数、inner-train 事件数、rank-count、cardinality schedule、
first-rank 分布和 train-only scaffold。四种 synthetic 条件为：

1. fixed shared graph positive；
2. phase/template-only negative；
3. 三个 component 平均 graph 严格为零的 stable mixture negative；
4. 每事件随机 contact permutation graph negative。

所有阈值在运行前冻结：positive truth recovery ≥0.50、positive split stability
≥0.60、positive 与最强 negative stability margin ≥0.10。144/144 fits 充分。

| patient | D0 | truth recovery | specificity margin | human real−null |
| --- | --- | ---: | ---: | ---: |
| epilepsiae_922 | PASS | 0.810 | 0.222 | −0.349 |
| epilepsiae_620 | PASS | 0.833 | 0.117 | −0.474 |
| epilepsiae_1096 | PASS | 0.944 | 0.340 | −0.111 |
| yuquan_zhangkexuan | NOT_PASSED | 0.340 | 0.358 | −0.047 |
| yuquan_chenziyang | PASS | 0.863 | 0.245 | −0.009 |
| yuquan_zhangjiaqi | NOT_PASSED | 0.921 | 0.028 | −0.005 |

`yuquan_zhangkexuan` 是 sensitivity 不足；`yuquan_zhangjiaqi` 能恢复 positive graph，
但不能稳定区分 fixed graph 与 stable mixture。两位均不能用于 single-graph human
structure interpretation。其余 4 位方法学可辨识，但 human real−null 仍为 0/4。

## 8. 最终科学判断

### 支持

- emitted-contact feedback 相对 matched noGraph 有稳定预测增量；
- 当前方法在 4/6 pilot 的真实观测规模下可辨识 fixed graph，并具有预注册
  negative specificity；
- unseen-start NLL 有局部可组合增量；
- M2 operators 在同一数据重复拟合中高度可复现。

### 不支持

- 当前 single fixed graph 高于 matched local/phase/template 的结构稳定性；
- single graph 对 unseen-start 完整 repertoire 的一致优势；
- one shared graph + sampling-only diversity；
- 依据现有结果启动 shared-backbone modulation、event drive、process noise 或多个
  fields。

### 未裁决

- 两位 D0 未通过患者中的 human structure；
- 任何超出当前 fixed-graph family 的稳定结构；
- 稳定但 route-modulated regimes 是否存在。M2 seed stability 只能保留为未来独立
  假设，不能由本轮结果直接升级模型。

安全中文口径：

> Contact feedback 改善了条件 suffix 生成，但在患者匹配校准可辨识的四位 pilot
> 中，当前 single fixed graph 的跨时间 observable influence stability 均未超过
> phase/local-transition/template matched null。Unseen-start 只出现 NLL 而非完整
> repertoire 的一致优势。因此当前 single-graph 分支阴性收口；该结果不排除其他
> route-modulated 或更高阶结构，两位校准不足患者保持未裁决。

## 9. 决策与产物

- current single fixed graph：`BOUNDED_NEGATIVE_IN_4_CALIBRATED_PATIENTS`；
- two D0-ineligible patients：`UNADJUDICATED`；
- modulated-backbone：`NOT_AUTHORIZED`；
- full 34-patient cohort / independent replication：`LOCKED_NOT_RUN`；
- SNN Gate：`ABSENT_BY_CONTRACT`。

机器状态：

`results/topic5_stable_interaction_graph/development/SIG_V2_1_IDENTIFIABILITY_STATE.json`

主要产物：

- `v2_1_existing_artifact_audit/`；
- `v2_1_m2_operator_audit/`；
- `v2_1_unseen_start/`；
- `v2_1_split_stability/`；
- `v2_1_split_stability_test_probe_rescore/`；
- `v2_1_patient_matched_identifiability/`。

第一次 unseen-start 运行因训练期间共享 source/config 文件被扩充而主动终止，产物
移动至 `v2_1_unseen_start_invalidated_provenance_race/` 并标记
`INVALIDATED_NOT_EVIDENCE`；正式结果来自冻结后从零重跑。

## 10. 最终工程验收

- D0、D3、D3 test-probe sensitivity 与 D4 汇总中的 source hash 均与当前 runner
  一致；所有正式产物均记录 `old_heldout20_scored=false`、`snn_inputs_read=false`；
- D0 144/144、D3 108/108、D4 90/90 fits 通过各自预先定义的 training adequacy；
- 相关单元与合同测试共 75 项通过；全部 v2.1 runner 可编译，7 个最终 JSON 可解析；
- 当前分支存在大量与本分析无关的 manuscript / figure 脏改动，本轮未自动 stage 或
  commit，避免把其他工作混入 RNNv2.1 证据栈。
