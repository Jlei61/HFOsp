# Continuous marked-state 当前接手说明

## 2026-08-26 R1.3 长患者 T1 分诊（当前最高优先级）

状态：固定三位患者、3 seed 的 target-trained explicit R1.3 已 9/9 完成；修正后的
完整长窗支持审计完成；没有患者同时满足 T1 稳定性与长窗独立支持，因此新人体 H3 按合同
调度 0 个作业。科学判定是
`H3_NOT_RUN_NO_PATIENT_MET_STATE_AND_INDEPENDENT_SUPPORT`，不是 H3 阴性。

- 韩宇轩：3/3 target alignment，只有 1/3 persistent 胜 memoryless；3/3
  correct-time 胜 matched wrong-time。中位 joint 为 +0.000233（timing +0.00566，
  mark -0.00493），即稳定存在 mark 方向的记忆，但不是稳定 joint persistent state；
  完整 N=1,000 + delayed-1,000 支持在 TRAIN/validation 仅 1/1 个不重叠窗。
- 陈子阳：3/3 选择 epoch 0，未形成候选 target-aligned state；完整 N=1,000
  支持仅 2/1 个不重叠窗。
- 程帅：只有 seed 1/3 训练启动并有利（joint -0.000210，correct−wrong
  -0.000229），另外 2/3 为 epoch 0。它是唯一有合格独立支持者：名义 N=1,000；
  real + delayed 对比实际需要 2,000 events，TRAIN/validation 为 8/3 个不重叠窗，
  validation 中位 1.15 h。稳定 T1 未达到 2/3，所以不运行 H3。

本轮复核还修正了一项会虚高独立样本量的问题：旧审计只按 real N-event 窗计数，
遗漏 causal-delayed arm 额外向前取的 1,000 events。N=2,000 的程帅支持由表面
7/3 降为完整对比 5/2，不再合格；当前最大合格尺度是 N=1,000（8/3）。

权威产物：

- 白话报告：`results/epi_prssm/continuous_marked_state/r1/r1_3_long_triage_goal_report/REPORT_PLAIN.md`
- 技术报告：`results/epi_prssm/continuous_marked_state/r1/r1_3_long_triage_goal_report/REPORT_TECHNICAL.md`
- 机器审计：`results/epi_prssm/continuous_marked_state/r1/r1_3_long_triage_goal_report/machine_audit.json`
- T1 汇总：`results/epi_prssm/continuous_marked_state/r1/r1_3_long_t1_triage/summary.json`
- 完整支持审计：`results/epi_prssm/continuous_marked_state/r1/r1_3_long_h3_followup/support_audit.json`
- 冻结合同：`docs/archive/topic5/continuous_marked_state_long_t1_triage_contract_2026-08-26.md`

全模块测试 67/67 通过（0 failure、0 error）；9 个 T1 均无 OOM；正式检验分区、
seizure probe 和 paper-ready figures 未打开。下一步不能继续给这三位补 seed 后再挑结果；
应先决定是扩展事前固定的长患者 T1 分诊，还是改进 T1 训练稳定性后再冻结新一轮。

> **先读两份更正记录**（都在 `docs/archive/topic5/`，优先于本文与各份报告）：
> 1. `recent_goals_integrated_review_post_review_corrections_2026-08-26.md`（第二轮，覆盖综合复审）
> 2. `continuous_marked_state_t2_long_total_post_review_corrections_2026-08-26.md`（第一轮，覆盖长尺度）
>
> 第二轮最要紧的一条：**H3 资格闸门原先按 `event_session` 分组、而真正建窗口的代码
> 按记录覆盖段分组**。下一步点名的 `epilepsiae_922` 正是受影响的患者——按正确分区它
> 只在 N=1,000 / N=2,000 上合格，N=5,000 的真实设计只有 204 个窗口、TRAIN 侧 0 个
> 完整不重叠窗。`epilepsiae_620` 任何 N 都建不出段内窗口，`epilepsiae_958` 也不合格。
> 已修，重算见 `final_reports/recent_goals_post_review_audit.json`。

状态：R1.2b 收口、formal R1.3 18/18、短尺度 T2-S1 12/12、长尺度 total-effect 与超长尺度双 kernel 队列全部工程完成。超长尺度科学判定为 `H3_LONG_UNRESOLVED_NO_CURRENT_SUPPORT`。

## 2026-08-26 超长尺度最终收口（优先于下文旧建议）

- 全 34 人支持度已按 TRAIN + validation、同一连续记录段、额外 1,000 次延迟历史重新审计。旧建议中的 620/958 没有严格 6 h 或 3,000 次连续支持，不再作为超长尺度主对象。
- 当前最大可训练尺度：程帅 15,000 次（545/5,515，validation 中位 11.73 h）；张佳琪 10,000 次（11,905/4,715，5.93 h，但 T1 3/3 退化）；彭子航 5,000 次；陈子阳 4,000 次；韩宇轩 2,000 次；E922 3,000 次。
- 5 位新发现患者各完成 7 seed T1，共 35 个 T1；近期加权和 whole-window boxcar 各完成 35 个 H3 artifacts，共 70 个。
- 旧生成器会把名义长窗口压回约 1–2 h，因此近期加权只作敏感性；boxcar 主实验让整个 N 次窗口等权，才直接检验几千到上万次累计。
- 原调度器只用 `selected_epochs > 0` 判定 T1 可用，科学门槛不足。更正后的规则同时要求 filtered 胜 no-state、persistent 胜 validation-correction-off；按此规则仅韩宇轩 7/7 seed 可进入探索性 H3，陈子阳 0/7，其他三人均停在 epoch 0。
- 韩宇轩 boxcar N=2000 的 real−intercept 为 +4.10（1/7 有利），6 h 为 +94.92（0/7 有利），当前不支持 exposure 增量；但每个条件的 validation 只有 1 个真正不重叠整窗，且 correct-time 仅 2/7 seed 有利，所以不能写成生物学阴性。
- 陈子阳 4,000 次/6 h 的表面有利方向因 T1 外层验证失败而不具可采信性；程帅的 15,000 次和张佳琪的 10,000 次均没有可用 T1，因此本轮实际上没有完成合格的万次 H3 检验。
- 新合同：`docs/archive/topic5/continuous_marked_state_t2_very_long_discovery_contract_2026-08-26.md`。输出根：`results/epi_prssm/continuous_marked_state/r1/t2_very_long_{discovery,boxcar}/`。

## 权威入口

- 超长尺度白话报告：`/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/r1/t2_very_long_overall/REPORT_PLAIN.md`
- 超长尺度技术报告：`/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/r1/t2_very_long_overall/REPORT_TECHNICAL.md`
- 超长尺度机器审计：`/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/r1/t2_very_long_overall/machine_audit.json`
- 最新白话总报告：`/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/r1/final_reports/r1_2b_r1_3_t2_long_total_plain_2026-08-26.md`
- 最新技术总报告：`/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/r1/final_reports/r1_2b_r1_3_t2_long_total_technical_2026-08-26.md`
- 最新机器审计：`/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/r1/final_reports/r1_2b_r1_3_t2_long_total_machine_audit.json`
- 复审更正记录：`/home/honglab/leijiaxin/HFOsp/docs/archive/topic5/continuous_marked_state_t2_long_total_post_review_corrections_2026-08-26.md`
- 事后补算的仪器诊断：`/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/r1/t2_long_total_effect/reports/post_review_audit.json`
- 长尺度合同：`/home/honglab/leijiaxin/HFOsp/docs/archive/topic5/continuous_marked_state_t2_long_total_effect_contract_2026-08-26.md`
- 长尺度聚合：`/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/r1/t2_long_total_effect/reports/summary.json`
- 先前 R1.3 汇总：`/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/r1/r1_3/reports/r1_3_summary.json`

## 当前科学结论

- H1：先前三位 development 患者 persistent-memoryless 3/3 有利，strict matched wrong-time 2/3 有利；最安全名称仍是跨窗口预测记忆。
- H2a：先前三位 first subset 与 continuation 均 3/3 有利，属于 development predictive evidence。
- H3（更正）：短尺度 N=1000 **不是 0/2**——两位患者的暴露边与 placebo 边在所有被检查的 epoch 都没有改善 TRAIN 内选择集，被留在零边，对比恒等于 0.0，分母是 0/0 而不是 0/2（同轮 `current_event_only` 臂确实训练起来了，所以机器本身没坏）。长尺度 synthetic 已按新判据重跑 10/10 通过（含两条「有均值 / 有漂移但无暴露」的假阳性校准）；张家齐三 seed T1 全部选择 epoch 0，读出秩 0，因此两个尺度的人体结果均为 `UNTESTABLE_T1_INSTRUMENT_DEGENERATE`，不是 H3 阴性。补充事实：这位患者的**整个状态模型（生成器、观测校正、三个读出）都停在构造函数默认值**，上游 R1.2 也是 epoch 0，所以不是「读出没学好」而是「从来没有过候选状态」。
- H3 超长尺度（最终）：更正 T1 门槛后，仅韩宇轩具 predictive+persistent memory；其 N=2000 和 6 h boxcar 均未胜 intercept。最长支持者没有合格 T1，且每个 boxcar validation 条件只有 1 个不重叠整窗。结论是“未决、当前无支持”，不是否定 H3。
- H2b/H3b：未运行。

## 本轮分母（更正）

窗口是逐事件滑动产生的，窗口计数不是样本量。真正的分母是端点跨度与记忆核长度：

- N=10000：11,905 TRAIN / 4,715 validation windows，中位 5.9275 h；
  但 validation 端点只跨 **1.63 h**，≈ **1.8 个有效独立窗口**；
- 约 6 h：7,922 TRAIN / 5,991 validation windows，中位 9,849 events；
  validation 端点只跨 **2.16 h**，≈ **2.4 个有效独立窗口**；
- 冻结生成器的时间常数是 **54.1 分钟**，所以两个「尺度」实际测的都是最近
  0.5–1.6 小时：有效加权事件数中位只有 2,409 / 2,285，不是一万次；
- 窗口跨未记录 gap：0（现在是算出来的，不再是硬编码字面量）；正式 test 打开：false。

## 已作废的旧下一工作包

下列建议已被超长尺度支持度审计取代，不应直接执行：把 total-effect 合同迁移到 `epilepsiae_620` 和 `epilepsiae_958`。两者没有严格 6 h 或 3,000 次连续支持。当前下一步应先在真正长序列患者上修复 T1，再以不重叠长块和 participation/repertoire exposure 重测。

历史上要求迁移前做的三件事仍保留为方法注意事项：

1. 主对比换成 `real_minus_intercept_matched` 与 `real_minus_causal_delayed`；
   `real_minus_no_edge` 只当伪迹量报——实测在完全无暴露信息的目标上它能给到 −445。
2. 先算这两位患者生成器的时间常数再定窗口长度。若同样是 54 分钟量级，就不要再叫
   「六小时总效应」。注：620 的生成器完全停在初始值（54.1 分钟），958 与黄瀚文的所有模态也都落在 53.5–55.1 分钟。
3. 事前报出端点跨度与有效独立窗口数。若又只有个位数，这个实验在单患者上本来
   就出不了结论，应先解决独立窗口再跑。

不继续给张家齐加 epoch、挑 seed 或扩大 exposure 网格。只有出现更多同时具备有效 T1 和 N=10000 支持的患者后，再做万次队列扩展。

## 复现

```bash
cd /home/honglab/leijiaxin/HFOsp
PYTHONPATH=. OMP_NUM_THREADS=1 /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/topic5_continuous_marked_state_r1/run_t2_long_total_synthetic.py
PYTHONPATH=. OMP_NUM_THREADS=1 /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/topic5_continuous_marked_state_r1/aggregate_t2_long_total.py
PYTHONPATH=. OMP_NUM_THREADS=1 /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest -q \
  tests/topic5_continuous_marked_state_r1 tests/topic5_continuous_marked_state
```

工作树原有无关修改全部保留。paper-ready Fig1–Fig4、seizure probe 与正式分区均未触碰。

复审后新增 / 重生成的复现步骤：

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 python \
  scripts/topic5_continuous_marked_state_r1/audit_t2_long_total_post_review.py
PYTHONPATH=. OMP_NUM_THREADS=1 python \
  scripts/topic5_continuous_marked_state_r1/aggregate_t2_s1_human.py
```

`t2_s1_long_scale` 的 12 个人体结果是 placebo donor 排除修复前跑的；
在重跑之前不得引用 placebo 对比。
