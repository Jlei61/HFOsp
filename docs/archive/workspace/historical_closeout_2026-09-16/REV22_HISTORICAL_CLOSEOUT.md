# rev22-postfit 历史归档验收（2026-09-16）

状态：**历史工作包归档验收完成；保留阶段结果，未完成的独立验证不补称通过。** 用户已明确将本路线作为历史收口，本次不再续跑或恢复旧队列。现有源码、候选冻结和失败记录保留，供后续追溯。

## 做了什么

rev22-DCI 将支持、顺序、时差和覆盖拆成四类训练目标，拟合训练响应面、按目标组合冻结候选，再执行资格、确认和结构null。源码提交836d1d2d包含验证输入根、工作树/结果根分离、目标对照图和嵌套输入检查修复；此前未推送的四个提交随本次归档提交一起保存。

## 得到什么、缺在哪里

现存机器记录：资格30单元、确认60单元、结构null 108单元的controller均为COMPLETE。训练阶段汇总90/90单元满足artifact integrity，但只有48/90满足primary estimability；五个冻结候选中，p030和p089的资格/确认条件分量均可估计，p000/p066/p075仍有CONDITIONAL_COMPONENT_NOT_ESTIMABLE。不能把文件齐全、训练距离降低或结构null跑完合并成患者传播机制已恢复。

响应面与frozen-stage汇总都明确是training-only。某些候选的时差/支持等分量相对参考改善，但分量有取舍，且存在条件样本支持缺口。旧各向同性参考守卫失败已经留在preliminary_archive；之后的修复不抹去原失败。

## 停在哪里

最新链状态停留在OPENING_SELECTION_BLIND_VALIDATION。核查时未找到audit/completion_audit.json和完整最终验证聚合，日志末尾仍有空切片警告；没有存活的rev22计算进程。此处归档为“冻结训练与仿真阶段完成，最终选择盲验证/总体科学验收未闭合”，不依据旧RUNNING文字推断它还会继续。本次只对已提交修复的相关回归测试做检查，不补跑科学实验。

## 去哪里找

- 原始结果：`/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/data_driven_dual_core_interictal_identifiability/`。
- 先读`frozen_stage_aggregate/frozen_stage_aggregate.json`、`response_fit/response_fit.json`和`response_fit/frozen_candidates.json`；再查`qualification/status/controller.json`、`confirmation/status/controller.json`及`structural_nulls/status/controller.json`。
- 方案：`docs/topic4_rev22_dci_interictal_parameter_identifiability_spec.md`与同名前缀的plan。
- 统一历史说明：main分支`docs/historical_results_closeout.md`；其中保存关键证据副本和本分支提交位置。

未来如重新研究参数可辨识性，应另建版本，先解决条件分量可估计性、模型/输入身份与独立验证合同；不能从本归档直接宣布参数唯一性、患者机制或正式图通过。后继患者几何先验与分布拟合路线有独立合同，不继承本版未完成的验收。
