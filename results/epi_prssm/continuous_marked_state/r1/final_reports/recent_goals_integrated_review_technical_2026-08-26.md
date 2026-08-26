# 近期 Continuous marked-state 目标综合复审：技术版

> **修订 3（2026-08-27）**：本版已把第二轮代码复审与最终路线收口直接吸收到正文。审计历史见
> `docs/archive/topic5/recent_goals_integrated_review_post_review_corrections_2026-08-26.md`，
> 机器重算见 `final_reports/recent_goals_post_review_audit.json`。H1/H2a 主判定不变；
> H2b 数字改用产生表重算值但仍待修复后重跑；六小时 boxcar 改判为不可估计；下一 H3
> 主实验固定为稳定 T1 上的 N=100 一步 generator-edge，超长 boxcar 退出主线。

**复审日期：** 2026-08-27
**范围：** 2026-08-21 至 2026-08-26 的 Raw-SEEG R0.1/R0.2、Continuous marked-state R0.1/R1/R1.2/R1.2b/R1.3、T2-S1、long-total、very-long boxcar 与 long-patient R1.3 triage。
**优先级规则：** 机器审计和后出的 corrections 高于早期 plain/technical report；development 与 formal/sealed partition 严格分开。
**当前总判定：** 工程完成度 **84/100**；科学路线对齐度 **80/100**；H1–H3 总体科学闭合度 **55/100**。工程分下调是因为新复审又发现资格闸门分段、ridge 尺度和 H2b 零分布三类承重问题；提交前覆盖 Raw-SEEG、Epi-PRSSM 与 Continuous marked-state 的 295 项测试证明修复后的实现满足当前合同，不把既有修复前人体产物自动变成合格证据。

## 1. 一句话审阅结论

8 月 24 日后的主线已回到原始科学目标，并建立了目前最合适的 joint timing + exact sequential mark 测量框架。formal R1.3 在三位固定 development 患者上支持跨窗口 predictive memory，并在 first subset 和 continuation 上给出 H2a 的精确 development 证据；raw waveform 尚无 explicit-feature 之外的稳定增量。H3 的人体总效应尚未被合格估计：短尺度为结构零，旧 long-total 对照有免费截距且名义长窗口被约 54 分钟生成器压缩，very-long boxcar 又缺少有效 T1 与多个独立验证窗的共同支持。最新 goal 正确地以 0 个 H3 作业收口。

## 2. 核心科学 estimand

### H1：持续、时刻相关的预测状态

主比较必须分层：

1. `filtered < no-state`：当前 observation 是否有用；
2. `persistent < memoryless`：跨 observation window carry 是否有用；
3. `correct-time < matched wrong-time`：该 carry 是否属于正确时刻；
4. correction-off/open-loop：在停止新 observation 后是否仍保留预测力；
5. raw < explicit：raw waveform 是否增加显式 spectral/variance/autocorrelation 之外的信息。

只有第 2 项可称 persistent predictive memory；第 2+3 项才接近 time-specific persistent state estimate；当前没有 autonomous physiological state 的验收。

### H2：state 对 IED repertoire 与 seizure transition 的关系

- H2a 主端点：first tied-group subset、later-group continuation、same-prefix continuation；STOP/size 单列为 termination/extent，不代替 repertoire。
- H2b 主问题：冻结 state 在发作前是否有连续变化，并与 early recruitment 或 seizure subtype 相连。患者为统计单位，发作数只作患者内分母。

### H3：IED exposure 对未来 state 的增量更新

H3a 要求从共同 pre-exposure state 出发，real exposure 至少胜过：

1. `no_edge_plus_fitted_intercept`；
2. 一个与真实暴露低重叠或不重叠的反事实 exposure。

固定延迟 1,000 次的 `causal_delayed` 在 N=2,000–10,000 时与真实窗共享 50%–90% 暴露，只能在报告重叠比例后作支持性对照，不能单独承担“时序不重要”或“无暴露效应”的判决。上述比较还必须在有效 T1、非退化 state readout、可估计拟合和足够独立长窗下成立。`real - no_edge` 已降级为 free-intercept artefact 诊断。H3b 只有在 T2 冻结后才允许接 seizure probe。

## 3. 阶段审阅总表

| 阶段 | 实际科学问题 | 主要结果 | 复审状态 |
|---|---|---|---|
| Raw-SEEG R0.1/R0.2 | 10 min raw context 能否预测 1/5/10/100 min 频谱 | wide Transformer 3/3 患者、4/4 horizon 胜等容量 Conformer；短 horizon 仍输 feature AR | 技术 closeout；不直接检验 H1–H3 |
| Continuous R0.1 | 汇总旧 Epi-PRSSM、Bridge、T1、H2b、H3-S0 | H2a 最强；H1 仅 predictive filter；H2b 探索性；H3-S0 为 25–200 event STOP/extent screen | 限定验收 |
| R1 | exact recorded-time timing + tied-group mark 仪器 | synthetic 3/3 恢复；旧 coverage 后被发现不完整 | mark 工具保留；旧 timing support 作废 |
| R1.2 | full raw-block support、全 anchor 六人 T1 | history timing 6/6；persistent 三主比较仅 2/6、中位 0 | 验收为 H1 阴性 pilot，不是队列结论 |
| R1.2b | spatial-tail target alignment | filtered/off 2/3；correct/wrong 0/3；raw upstream 未训练 | limited diagnostic |
| formal R1.3 | full target-trained observer + persistent/memoryless + strict swap | persistent 3/3；correct-time 2/3；subset/continuation 3/3；raw joint 1/3 | 当前 H1/H2a 主 development 证据 |
| T2-S1 N=100/1000 | condition-on-current-state one-step exposure edge | real/placebo edges 全部 epoch 0，差值恒 0 | 结构零；正确分母 0/0 |
| long-total | 10,000-event / ~6 h generator-weighted total effect | 张家齐 T1 全初始化；免费截距；有效核约 54 min；1.8/2.4 独立窗 | 人体结果不可验收 |
| very-long boxcar | true whole-window thousands-event exposure | 35 T1 + 70 H3 artifacts；韩宇轩 N=2,000 可估计但未胜拟合截距，6 h 为 7/7 发散；boxcar 有效独立 validation 窗仅 0.48–0.93 | 工程完成；H3 unresolved，旧人体值不能形成合格阴性 |
| long-patient R1.3 triage | 先验 T1 和完整支持联合分诊 | 9/9 T1 完成；无人同时满足稳定 T1 与完整独立长窗；H3 jobs=0 | 正确收口，非 H3 阴性 |

## 4. Raw-SEEG R0.1/R0.2 的正式边界

### 4.1 数据与模型

- 34 位患者，3,547 个记录块，3,182 个 bipolar contacts；development recorded support 2,393 h。
- 输入为 10 min raw SEEG，32 维 candidate latent code，固定 damped-rotation dynamics，预测 1/5/10/100 min 的 12-band spectral field。
- target-shuffle 与 mean baseline 重合，说明未见未来泄漏；identity dynamics 与 full 差异很小；latent consistency ratio 大于 1，学到的 dynamics 不优于保持状态不动。

### 4.2 更正后的解释

- “21 参数 ridge”更正为 4 horizon × 12 band × 21 = 1,008 个系数；网络为 1,492,496 参数。
- 1 min–48 h 的 `tau` 是 log-uniform 初始化覆盖，训练后相对变化中位 11%、最大 40%；只能说模型可表示，不是数据识别。
- 620 的 100 min persistence 胜出未在 958 复现。
- 60 epoch 只排除同配置继续加 epoch，不排除表示、目标和优化结构失配。

**验收：** 该阶段反驳的是“10 min raw encoder + 32D global latent + fixed damped rotation + spectral forecast”这一组合，不反驳 H1–H3。

## 5. Exact event-model 路线

### 5.1 R1/R1.2 仪器修复

模型联合计算 recorded-support point-process timing likelihood 与 exact tied-group unordered-without-replacement sequential mark likelihood。显式 event history 固定计算，不允许 free history RNN 成为第二个 latent state。

R1 的 P0 coverage 问题是：记录轴由“至少含一个 definite-interictal IED 的块”反推，遗漏有记录但无入选事件的区间，使 survival integral 被事件密度反向定义。R1.2 改为 raw-SEEG block inventory 减 ictal/postictal support，并在禁用区后重开 latent/history session。六人合计 60,930 个 admissible readable anchors。

R1.2 patient-first 结果：

- history timing−static：6/6 负值；
- history mark−static：3/6 有利；
- history mark−shuffle：4/6 有利；
- explicit filtered−no-state、filtered−off、filtered−wrong：均 2/6 有利，patient median 0；
- raw−explicit filtered：1/6 有利，量级约 5e-6。

结论仅为 history timing information 稳定存在；背景 persistent state 未建立。

### 5.2 R1.2b 收口

固定 3 patients × 2 arms × 3 seeds = 18 fits。explicit `filtered-no-state` patient median −0.01786，2/3 有利；`mark filtered-off` median −0.01527，2/3；strict `filtered-wrong` median +1.66e-5，0/3。620 为三 seed epoch 0 no-update。

raw tokenizer 和 temporal blocks 未接受 target gradient，因此 raw near-zero 不可解释为 raw 阴性。后处理的 persistent-memoryless 在有更新患者上提示跨窗 carry，但 strict swap 不支持时刻专属性。该阶段不扩队列。

### 5.3 formal R1.3 主结果

固定三人，每人 explicit/raw 两臂、三 seed，共 18/18 fits。paired raw 从同 seed explicit checkpoint 出发，只更新 raw tokenizer、两层 temporal Transformer、projection/norm/gate；共同 spatial/state/readout 更新量严格为 0，raw selection gradients 全部非零。

所有差值为 left−right，负值有利：

| 患者 | persistent−memoryless joint | timing | mark | first subset | continuation | correct−wrong joint |
|---|---:|---:|---:|---:|---:|---:|
| epilepsiae_620 | −0.01931 | −0.000927 | −0.01905 | −0.01097 | −0.007922 | −0.03654 |
| epilepsiae_958 | −0.04900 | −0.01411 | −0.03666 | −0.01721 | −0.01181 | −0.005010 |
| yuquan_huanghanwen | −0.21092 | +0.23484 | −0.44575 | −0.19718 | −0.23633 | +0.01558 |

因此 persistent-memoryless joint、first subset、continuation 均 3/3 有利；correct-time joint 2/3 有利。黄瀚文 validation 仅 107 events，其 joint 阳性由 mark 驱动而 timing 明显反向，必须单列。

paired raw increment：

| 患者 | raw−explicit joint | timing | mark | group size | subset |
|---|---:|---:|---:|---:|---:|
| epilepsiae_620 | +0.003206 | −0.001587 | +0.004421 | +9.88e-5 | +0.004319 |
| epilepsiae_958 | +0.001804 | +0.001892 | −7.59e-6 | −0.000406 | +0.000399 |
| yuquan_huanghanwen | −0.000162 | −0.000358 | +0.000196 | +6.56e-6 | +0.000191 |

raw joint 仅 1/3 有利，patient median +0.001804；6/9 raw fits 在 4-epoch 当前预算末端被选中。因此可写“full raw stack 已接受 target gradient，但当前 pilot 未见稳定 independent increment”，不能写 raw 无信息。

## 6. H2a 与 H2b 的证据边界

### H2a

旧 development 证据包括 state swap、same-prefix continuation、患者特异图接线和开放环 event-history state。formal R1.3 又在 exact mark 下得到 persistent-memoryless first subset 与 continuation 3/3 有利。这使 H2a 成为当前最强假设。

但其限制是：

- fixed development n=3，不是队列估计；
- 黄瀚文 validation events=107；
- correct-time joint 仅 2/3；
- prediction increment 不等于 state 对 IED 的生物因果控制；
- raw independent increment 未成立。

### H2b

原报告的 "+0.446 SD，20/27，p=0.019，361 次"来自已被覆盖的 2026-08-20 markdown。
当前磁盘产物是 2026-08-21 01:50 重跑的版本；从逐发作产生表
`preictal_effects__linear_graph_recurrent__lead30m.csv` 重算得到：

| 层 | 中位偏移 | 同向 | sign p | 进入读数的发作 | 表内行数 |
|---|---:|---:|---:|---:|---:|
| all_eligible / open_loop | **+0.4582 SD** | **21/27** | **0.00592** | **339** | 361 eligible |
| high_observability / open_loop | **+0.2667 SD** | **17/27** | **0.2478** | **201** | 203 premise-met |

361 / 203 是资格行数不是读数分母：22 次发作的 pseudo-onset null 退化、算不出 z。
方向上主层比原文更强、高可观测层比原文更弱，**定性结论不变**。
leave-one-patient / leave-one-seizure 符号仍稳定（LOPO +0.4600 [0.4563, 0.4620]，
LOSO +0.4582 [0.4544, 0.4657]）。连续 observability gradient 仍接近零，subtype
interaction 仍未胜自身 permutation null。R1.3 后未重跑 seizure probe。

但 `run_goal3b_preictal.py` 有两处会影响这组数字——peri-ictal 排除
阶梯只用了落在 admissible span 内的发作（span 外的发作不排除附近 pseudo cut-off，
把效应压向零），干扰变量基准 z 用 `pseudo_times[:60]` 而端点 z 用全部 200 个
（伪起点按匹配代价排序，基准因此拿到更紧的零分布）。两处已改代码；12 个产物是
修复前跑的。因此上表是当前磁盘上最好的重算值，但**不是冻结论文数字；重跑后必须再次更新**。

因此 H2b 只能保留为 exploratory preictal association，不能写 latent transition 或 subtype mechanism。

## 7. H3 审阅

### 7.1 H3-S0 antecedent screen

34 人 count-memory screen 显示约 25–200 recent events 对下一 mark/STOP 有约 0.001–0.003 NLL/event 的小增量；current-event/1 s 近零，timing 与 contact order 无稳定方向，physical clock 未胜 event-count clock。该结果支持多事件累积候选，但仍可由共同未观测 state 解释。

### 7.2 T2-S1 structural zero

两人 × N{100,1000} × 3 seeds 共 12 fits 中，`real_cumulative` 和 `state_matched_placebo` 全部 selected epoch=0，edge vector 保持零初始化，故 real-placebo 和 real-no-edge 恒为 0。正确统计为：

- `n_structural_zero=2` patients；
- `n_estimated=0`；
- favourable denominator=0/0。

同轮 `current_event_only` 选到 epoch 5–15 且对比非零，排除 runner 全局失效。既有人体 placebo 结果又早于 donor exclusion 覆盖 validation target 的修复，重跑前不得引用。

### 7.3 long-total P0/P1

1. **P0 free intercept：** exposure arms 的固定 event jump 经衰减累加后近似常数，real/delayed 各自拥有 no-edge 没有的 state offset。无 exposure 的 constant-offset target 上 `real-no_edge=-445`，linear drift 为 −31。主对比已改为 real−fitted-intercept；real−causal-delayed 只作支持性比较，并必须同时报告暴露重叠率。
2. **P0 T1 degeneracy：** 张家齐 generator、observation correction、state readout 全部停在初始化；decoder rank=0；三个 seed payload 实际等同。人体差值为结构零。
3. **P1 nominal scale mismatch：** frozen generator `tau=54.06 min`。名义 10,000 events/约 6 h 的 effective weighted events 为 2,409/2,285；50% 权重来自最近约 0.5 h，90% 来自约 1.6 h。
4. **P1 pseudo-replication：** 10,000-event validation 4,715 sliding windows 只覆盖 1.63 h≈1.8 effective independent windows；6 h 5,991 windows 只覆盖 2.16 h≈2.4 independent windows。

合成仪器补入 intercept-matched null 后 10/10 acceptance 通过；人体 long-total 不可采信。

### 7.4 very-long boxcar

5 名按长序列支持事前选择患者、7 seeds：35 T1；generator-weighted 与 boxcar 共 70 H3 artifacts。boxcar 等权整窗，技术上真正覆盖数千次事件。

旧 very-long 产物按当时的 R1.2 T1 外层规则重聚合后，只有韩宇轩 7/7 达到 predictive+persistent；correct-time 仅 2/7。注意这不是 formal R1.3 的闸门语义：R1.3 没有独立的 filtered-vs-no-state 臂，不能把同一个 persistent-memoryless 表达式重复记作 predictive 和 persistent 两项。韩宇轩归档主读数应按可估计性重写为：

| window | estimable seeds | real−fitted-intercept | favourable seeds | real−delayed | 有效独立 VAL 窗 |
|---|---:|---:|---:|---:|---:|
| N=2,000 boxcar | 7/7 | +4.1003 | 1/7 | −1.6741（窗重叠 50%） | **0.48** |
| 6 h boxcar | **0/7** | 不可解释（归档值 +94.9233） | n/a | 不可解释 | **0.79** |

N=2,000 虽胜 delayed 6/7，但未胜 fitted intercept；这只说明高度重叠的 delayed arm 更差，不构成 exposure increment。6 h 的七个拟合臂全部超过 intercept 误差四倍，实际为 20–149 倍，属于外推发散而不是“0/7 阴性”。全 70 条归档超长臂中 26 条不可估计，最大 arm/intercept 比为 6,809；76 条 ridge 拟合中 47 条把惩罚选在旧网格上界。当前实现已改为按 Gram 尺度归一的 ridge，并加入 `estimability_guard` 和 TRAIN→validation `target_shift_audit`，但这些旧人体臂尚未因此自动变成新版结果。

陈子阳的 boxcar 拟合本身可估计，旧宽松闸门下 N=4,000 为 real−intercept −4.31（5/7），6 h 为 −33.70（6/7）；相应有效独立 validation 窗仅 0.91/0.93。由于该患者 T1 外层验证 0/7 合格，这些数值只能保留为“若未来 T1 修复后值得复查”的候选，不能进入 H3 证据。

延迟对照与真实窗的共享暴露比例在归档超长臂中位为 0.667；N=10,000 时为 0.9。因此大 N 下 `real−delayed≈0` 不等于 load timing 不重要。未来应使用延迟至少一个完整窗口或匹配的不重叠 donor。

### 7.5 最新 long-patient R1.3 triage

固定韩宇轩、陈子阳、程帅，各 3 seeds，9/9 无 OOM 完成：

| 患者 | target-aligned | persistent<memless | correct<wrong | patient-median persistent joint | 完整 H3 独立窗 |
|---|---:|---:|---:|---:|---|
| 韩宇轩 | 3/3 | 1/3 | 3/3 | +0.000233 | N=1000 时 TRAIN/VAL 1/1 |
| 陈子阳 | 0/3 | 0/3 | 1/3 | +3.39e-5 | N=1000 时 2/1 |
| 程帅 | 1/3 | 1/3 | 1/3 | 0（两 seed epoch 0） | N=1000 full instrument 8/3 |

旧支持审计只按 real N-event window 计数，遗漏 causal-delayed arm 额外 1,000 events。修正后程帅 N=2,000 从表面 7/3 降为 full-union 5/2；最大合格支持为 N=1,000，full instrument 需 2,000 events，TRAIN/VAL 8/3。但程帅 T1 仅 1/3 稳定，故不运行 H3。

最终 verdict：`H3_NOT_RUN_NO_PATIENT_MET_STATE_AND_INDEPENDENT_SUPPORT`。这是 eligibility 结论，不是 biological null。

### 7.6 prospective H3 支持度的覆盖段复算

原资格函数按 `event_session` 分组，而真正的 `build_long_window_design` 按记录覆盖段建窗。
对 Epilepsiae，session 可跨普通记录间隙，因而旧闸门系统性虚高。使用与建窗程序一致的
覆盖段口径后：

| 患者 | 覆盖段 / session | 仍合格的最大尺度 | TRAIN/VAL 完整不重叠窗 |
|---|---:|---:|---:|
| epilepsiae_922 | 49 / 11 | N=2,000 | 10/3（N=1,000 为 21/7） |
| yuquan_pengzihang | 4 / 4 | N=1,000 | 9/3 |
| epilepsiae_620 | 64 / 9 | 无 | 0/0 |
| epilepsiae_958 | 135 / 9 | 无 | N=1,000 仅 2/0；更大尺度无窗口 |

旧闸门会把 E922 选到 N=5,000，但真实设计只有 204 个窗口且 TRAIN 为 0 个完整独立窗。
代码已改为记录覆盖段分区，并同时持久化段数与 session 数。本轮三位 Yuquan 恰好都是
1 segment=1 session，因此既有 `H3_NOT_RUN` 判定不变；下一轮患者选择则必须使用新口径。

## 8. P0/P1 问题清单与处置

| 优先级 | 问题 | 科学后果 | 当前处置 |
|---|---|---|---|
| P0 | R1 coverage 由事件块反推 | survival integral 被事件密度污染 | R1.2 raw-block support 替代；旧 timing 作废 |
| P0 | R1.2b raw upstream 未接受 target gradient | raw near-zero 被误作 raw 阴性 | R1.3 full target training |
| P0 | long-total real arm 免费截距 | 无 exposure 也可产生巨大阳性 | fitted-intercept 主对照；real-no-edge 降级 |
| P0 | short T2 zero edge 被写成 0/2 | 构造零被误作人体阴性 | n_structural_zero/n_estimated 分母 |
| P0 | 张家齐 state/readout 全初始化 | 所有 H3 对比结构相同 | 标记 T1 degenerate，不引用人体值 |
| P1 | nominal long window 被 54 min K 压缩 | 万次/6 h 命名失真 | 报 effective kernel；boxcar 才作长记忆 |
| P1 | sliding windows 当样本量 | 严重伪重复 | 报 endpoint span 与 non-overlap windows |
| P1 | 旧 R1.2 very-long scheduler 只检查 epoch>0 | 训练动了被当成状态有效 | 增加 persistent、time-specific 与 readout 条件；R1.3 不虚构不存在的 filtered-vs-no-state 检验 |
| P1 | support audit 忽略 delayed extra history | 完整对比支持虚高 | 按 arm union 计 full instrument window |
| P1 | seed payload 重复仍报 x/3 | 优化重复数虚高 | distinct payload audit；seed 不作患者分母 |
| **P0（新）** | H3 资格闸门按 `event_session` 分组，设计按记录覆盖段分组 | 每位 Epilepsiae 患者的独立窗口预算虚高；922 会被安排在 N=5000（真实设计 204 窗、TRAIN 0 个完整不重叠窗），958 全 N 假合格 | `support_for` 改按覆盖段分区并记录两种计数 |
| **P0（新）** | 绝对 ridge 惩罚使网格等于不做正则化 | 47/76 拟合选在网格上界；26/70 超长臂外推到 intercept 的 4–6809 倍，被写成"检验失败" | 尺度无关惩罚 + 扩网格 + `estimability_guard` + `target_shift_audit` |
| **P0（新）** | H2b 承重数字取自 2026-08-20 旧 markdown | 主层 p 0.019 vs 实际 0.0059、分母 361 vs 实际 339 | 从产生表重算并写入 `recent_goals_post_review_audit.json` |
| **P1（新）** | `--t1-source r1_3` 的两个闸门条件是同一个表达式 | 两条件闸门被写成三条件；R1.3 根本没有 filtered-vs-no-state 臂 | R1.3 分支下该字段置 `null` + `t1_predictive_check_available` |
| **P1（新）** | 后续 H3 闸门只实现三条准则里的两条 | 实现比计划宽松（本轮不改变结论） | 补上 `time_specific_supported` |
| **P1（新）** | boxcar 用生成器 tau 当去相关长度 | 独立窗口预算高估 6–11 倍 | 审计接受 `exposure_memory`，boxcar 用窗口长度 |
| **P1（新）** | 延迟对照与真实臂共享 (N−1000)/N 的暴露 | 大 N 下 `real − delayed` 近乎无分辨力 | 新增 `delayed_control_overlap` |
| **P1（新）** | H2b peri-ictal 排除只用 span 内发作；干扰基准零分布用 60/200 | 效应被压向零；基准 \|z\| 被抬高 | 两处已改；12 个产物需重跑后才可再次引用 |
| **P1（新）** | “同容量截距”命名错误 | fitted intercept 只有 dim 个参数，真实臂为 2×dim，repertoire 为 3×dim | 统一改称“拟合截距对照”或“截距匹配”，不再称同容量 |

## 9. 对执行策略的判断

### 科学路线是否偏离

- Raw spectral R0.1/R0.2 确实偏离原始 H1–H3，现已正确永久收口。
- Continuous marked-state R1 之后没有偏离：timing、mark、persistent state、IED→state edge 均直接对应原始问题。
- H3 并未被设成 H1/H2 的总 gate；当前 H1/H2a 证据可独立保留。

### agent 是否过度保守

结论边界本身不算过度保守。上述 P0/P1 每一项都会把不可解释结果包装成阳性或阴性，必须修。真正的问题是运行顺序过于激进：在 T1 可用性、effective scale、intercept matching 和 independent support 未预审完前就发起大量 H3 作业，之后才靠防御性审计撤回。

最新 goal 已把流程改成正确顺序：先 T1、再 full-arm support、最后才调度 H3。后续应保留这一顺序，但不应让 H3 eligibility 阻断 H1/H2a 的探索。

第二轮复审进一步表明，仅调整执行顺序还不够：**资格闸门本身必须与真实 producer 使用同一分段、同一完整区间，并先用会翻转资格的边界病例验证。** 用未验证的 gate 选择患者，与用未验证的 estimator 产生效应量属于同一类科学风险。

## 10. 当前假设验收

| 假设 | 科学完成度 | 当前允许结论 | 禁止结论 |
|---|---:|---|---|
| H1 | 55/100 | 三位固定 development 患者存在跨窗口 predictive memory；2/3 有 strict time specificity | raw physiological slow state、autonomous state、队列普遍性 |
| H2a | 72/100 | persistent memory 改善 exact first subset 与 continuation prediction；旧患者特异图证据独立支持 | state 因果控制 IED、传播机制已证明 |
| H2b | 28/100 | 修复前产物中 339 次可评分发作的主层存在 exploratory frozen-readout shift；当前重算为 +0.4582 SD、21/27、p=0.00592 | 两处 pseudo-onset 仪器修复后尚未重跑；不得冻结为论文数字或写统一 preictal transition |
| H3a | 20/100 | 25–200 event STOP/extent antecedent screen；修复后的 synthetic instrument 可用 | IED 已塑造 state/network；长尺度阴性 |
| H3b | 5/100 | 尚无新支持 | IED-mediated seizure transition |

## 11. 下一阶段验收建议

### 11.1 R1.4：六患者 H1/H2a 复现

患者在新结果前固定为：`epilepsiae_620`、`epilepsiae_958`、`yuquan_huanghanwen`、`epilepsiae_922`、`yuquan_pengzihang`、`yuquan_hanyuxuan`。前三位是 formal R1.3 原队列，后二位扩展数据异质性，第六位韩宇轩按既有长记录支持事前指定，不根据本轮结果选择。

主 observer 固定为 explicit spectral + variance + autocorrelation；raw Transformer 仅作同 seed paired residual sensitivity，不再比较 encoder 家族，也不让 raw 结果阻断 H1/H2a。主要比较为：

1. persistent vs memoryless；
2. correct-time vs 每个 anchor 5–10 个 matched wrong-time donors；
3. timing、STOP/group size、first subset、later continuation、same-prefix continuation 分解。

wrong-time donor 限定在同一记录覆盖段，匹配 time of day、time since last IED、recent IED rate、last-event load/STOP、observation coverage 与 session position。seed 先在患者内取中位，患者是主要统计单位。

### 11.2 T2-R2.0：N=100 的最小 generator-edge 检验

超长 boxcar 不再作为下一主线。H3-S0 的稳定前置信号位于 25–200 events，故主尺度事前固定为 **N=100**，不按患者选择。优先运行当前 T1 最稳定的 `epilepsiae_620` 和 `epilepsiae_958`；新患者只有在 R1.4 出现可解释的 persistent/time-specific state 后再加入。

对事件属性 `phi(m_e)` 在 TRAIN 内 cross-fit：

`eta_e = phi(m_e) - E[phi(m_e) | z_e^-, r_e, o_e^-]`，

并按 `x_e = exp(-1/N) x_{e-1} + eta_e` 累积。事件 `e` 仍由 `z_e^-` 预测；事件结束后才允许 `z_e^+ = z_e^- + B x_e`。主 source 是 scalar load innovation；剔除 total load 后的 participation composition 独立作为 secondary source。

四个核心臂为：T1 no-edge、real cumulative exposure、state-matched 且历史不重叠的 donor exposure、current-event-only jump。fitted-intercept 只作为固定 offset 诊断。

一级端点是 next-event counterfactual：从相同 pre-event state 出发，关闭下一事件前 raw correction，比较下一事件 timing 与 exact mark。二级端点是 one-shot persistence：只在 anchor event 注入一次 real/placebo jump，之后关闭 raw correction 和新 T2 jump，保持相同真实 history covariates，比较 H5/H10。仅下一事件改善时称 `exposure-conditioned next-event prediction`；差异通过冻结 generator 延续时才称 `exposure-induced state update`。

训练前必须报告 `B=0` 处 gradient、exposure variance、design rank、edge 是否离开初始化，以及 positive/zero/reversed synthetic recovery。edge 结构零、T1 不稳定或 donor 不可构造时记为不可估计，不作人体阴性。

### 11.3 后续扩展顺序

只有 N=100 real edge 离开零且胜过 donor/current-event 后，才增加 N=50/200、与 TRAIN median IEI 匹配的 physical-time arm，以及 event merge/thinning sensitivity。N=1,000–2,000 与六小时 boxcar 退出当前主实验。

H2b 只做两件事：立即用修复后的 pseudo-onset 代码重跑旧 12 个产物并归档；待 R1.4/T2 冻结后，再以新 state 做 5/15/30/60/120 min patient-first probe。seizure loss 不反向训练 state，H3b 只在 T2 edge 可估计并冻结后运行。

## 12. 复审后的安全论文口径

> 在固定的 development 患者中，联合 IED timing 与 exact sequential mark 训练识别到跨 observation window 持续的预测记忆，该记忆改善下一次 IED 的触点 subset 与后续 continuation；正确时刻状态在两位患者中优于严格匹配的错误时刻状态。完整 raw temporal observer 已接受目标梯度，但尚未显示显式背景统计之外的稳定增量。近期多事件历史与下一事件 STOP/extent 存在小幅预测关联，但短尺度 generator edge 未被估计；长尺度人体实验又受到前状态退化、拟合发散、反事实暴露高度重叠和不足一个有效独立 validation 窗的共同限制。因此现阶段支持 state-dependent IED repertoire prediction，不支持 IED 已因果塑造长期状态或发作转换。H2b 的当前主层偏移仍是修复前探索性读数，须重跑后再冻结。

## 13. 修订验证

- `recent_goals_post_review_audit.json` 状态为 `COMPLETE`，当前估计器修订为 `t2_long_total_effect_decoder_space_v4_scaled_ridge_estimability_guarded`；机器审计确认 formal test partition 与 sealed partition 均未打开。
- 提交前联合运行 Raw-SEEG 五个测试文件、`tests/topic5_continuous_marked_state/`、`tests/topic5_continuous_marked_state_r1/` 与 `tests/topic5_epi_prssm/`，结果为 **295 passed，13 warnings**。warnings 来自 CuPy experimental API 与 PyTorch Transformer nested-tensor 的提示，不改变数值结果。
- 测试证明当前代码满足修订后的合同，不会把旧 H2b 产物、旧 ridge 人体结果或不足一个独立窗的数据自动升级成合格科学证据。

## 14. 权威证据路径

- R0.1 验收：`docs/archive/topic5/continuous_marked_state_r0_1_acceptance_2026-08-24.md`
- R1.2 coverage 更正：`docs/archive/topic5/r1_2_coverage_correction_2026-08-25.md`
- R1.2 报告：`results/epi_prssm/continuous_marked_state/r1/r1_2/reports/plain_report_2026-08-25.md`
- R1.2b 收口：`docs/archive/topic5/continuous_marked_state_r1_2b_closeout_2026-08-25.md`
- formal R1.3：`results/epi_prssm/continuous_marked_state/r1/r1_3/reports/r1_3_summary.json`
- long-total 更正：`docs/archive/topic5/continuous_marked_state_t2_long_total_post_review_corrections_2026-08-26.md`
- very-long 总报告：`results/epi_prssm/continuous_marked_state/r1/t2_very_long_overall/REPORT_TECHNICAL.md`
- 最新 long-patient triage：`results/epi_prssm/continuous_marked_state/r1/r1_3_long_triage_goal_report/REPORT_TECHNICAL.md`
- 当前 handoff：`results/epi_prssm/continuous_marked_state/r1/CURRENT_HANDOFF.md`
- 本轮二次复审：`docs/archive/topic5/recent_goals_integrated_review_post_review_corrections_2026-08-26.md`
- 本轮机器重算：`results/epi_prssm/continuous_marked_state/r1/final_reports/recent_goals_post_review_audit.json`
- 下一版冻结合同：`docs/archive/topic5/continuous_marked_state_r1_4_t2_r2_0_contract_2026-08-27.md`
