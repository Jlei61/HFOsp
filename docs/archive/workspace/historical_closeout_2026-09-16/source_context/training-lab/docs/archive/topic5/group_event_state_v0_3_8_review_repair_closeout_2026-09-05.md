# v0.3.8 审阅修复验收与科学边界

**追加审阅后的正式收口（2026-09-05）：审阅修复工作包完成；原始核心科学闭环未建立。** 以下原验收记录保留，最新口径与图表见[第二轮白话报告](/data/hfosp_group_event_state_v0_3_8_review_repair/final_reports_round2/group_event_state_v0_3_8_core_closeout_plain.md)、[第二轮技术报告](/data/hfosp_group_event_state_v0_3_8_review_repair/final_reports_round2/group_event_state_v0_3_8_core_closeout_technical.md)和[机器汇总](/data/hfosp_group_event_state_v0_3_8_review_repair/final_reports_round2/summary_main.json)。不再使用“40/100”。

追加收口纳入常数/随机臂强基线封底（E1096原0.51–0.55降为0.078–0.122）、逐患者seed分组的代码版本审计、错时donor与独立物理窗诊断，以及最内层q初始回退。q选第0步不代表后面全部嵌套层都退回截距；也不自动等于优化失败。宽joint交集与另加学习/重放条件的交集不同，对二者矛盾的怀疑撤回。更新后115项回归通过，26498个登记hash通过，六张PNG重新目视检查，PDF同时再生成。[收口记录](/data/hfosp_group_event_state_v0_3_8_review_repair/final_reports_round2/closeout_record.json)。

后续按[v0.3.9合同](group_event_state_v0_3_9_transition_transfer_contract_2026-09-05.md)执行固定历史F、线性转移L、非线性转移N和同一冻结状态的未训练任务迁移。旧模型剩余优化受限路径保留边界，不继续无限补预算；原有数据病例仅作设计pilot。

**后续完成记录：** [v0.3.9有限pilot已收口](group_event_state_v0_3_9_transition_transfer_closeout_2026-09-05.md)。修正测量/背景后的新结果独立存放，不能用它给下面的旧回顾性结果补前瞻资格；原始科学闭环仍未建立。

**继续执行中新增的数据边界限制：** [v0.3.9输入审计](group_event_state_v0_3_9_input_boundary_audit_2026-09-05.md)查到旧触点选择汇总了整个记录目录，FIT-only重建后3/3病例的触点集合变化。所以下面的旧局部预测线索还必须限定为“在固定的全记录测量字典下的回顾性结果”，不具备严格前瞻输入资格。当前已转入新测量平行重建；审阅修复包收口不表示背书旧数据链为完全因果，也不因此认定任何生理假设阴性。

更新时间：2026-09-05T14:10:07.966885+08:00。本轮审阅、代码修复、有限补实验和产物核验完成；不表示原科学假设已经成立。原结果保留，未commit/push，根工作树未用于本轮修改。

## 科学结论

存在患者内预测线索和真实长历史梯度，但尚未建立由间期事件学习、具有非线性演化、能够统一解释多类事件表达和发作的病理状态。H3生理反馈未被当前观察器设计识别。

- E1096的event 0.5h、grid 0.5/2h保留同三个seed的完整H1预测对照线索；权重更新和端点分数复现有支持，但严格特征来源资格未满足。
- E1125 dual 6h有4/5同模型H1、事件输入更新、同checkpoint长程credit及固定community+mark端点链；但5/5事件分支用FIT均值更好，不能把整模型动态收益归给事件分支。
- E922旧event/grid发作证据同seed完整交集只有1/5，原可重复候选撤回。三次留出onset在6h聚集敏感性下为一个跨分区簇。
- H2a修复已知前缀混入后缀评分、缺少同子集常数对照及承重预算不足。固定端点身份后，无完整多端点重复候选；单端点方向性结果保留。
- 固定时间常数加输入矩阵学习不等于发现非线性生理演化；事件更新和历史删除不识别IED的生理因果反馈。

## 工程验收与资格限制

13个登记修复队列全部COMPLETE：611个完成、40个不可估、零失败、零待运行。单元包含同模型不同审计，不是独立样本。GPU由固定源码/输入hash监督器接续，最终空闲因无剩余登记任务。

- 回归113/113通过；26498个已登记文件hash一致，另56个新权重/评价包通过hash核验。
- H1共770条训练路径，仍6条background_current在2700步后触顶：E1096 dual seed20260907及E583 dual全部5个seed。有限预算复核已完成，这些继续为OPTIMIZATION_LIMITED，不能称训练充分或据此作生理阴性。
- H2a的360条承重stage均通过原耐心/预算资格；H2b的350个可估选中风险读出均通过最终梯度阈值，不等于统计功效或生理机制已成立。
- 原H1存档输入边界159/165端点分数复现、147/165严格状态与分数通过；新预处理144/165端点分数复现、0/165通过全部严格特征与分数容差。未保存的旧预处理历史无法补造，新缓存、变换与输入包已加锁。
- 逐层清单覆盖165个H1模型、120个可估H2a adapter与24个冻结decoder，参数和固定几何/mask分开计数。
- 六张图均检查PNG与PDF，最终图像等同已目视检查版本；未宣称作者接受。

## 交付

- [白话报告](/data/hfosp_group_event_state_v0_3_8_review_repair/final_reports/group_event_state_v0_3_8_core_closeout_plain.md)
- [技术报告](/data/hfosp_group_event_state_v0_3_8_review_repair/final_reports/group_event_state_v0_3_8_core_closeout_technical.md)
- [机器汇总](/data/hfosp_group_event_state_v0_3_8_review_repair/final_reports/summary_main.json)，SHA256 `5389cc4822f7806c2f595cfd5f3544629a1b9b79207c1c5aede6731219ec9f90`
- [状态来源](/data/hfosp_group_event_state_v0_3_8_review_repair/state_lineage.json)；[逐层清单](/data/hfosp_group_event_state_v0_3_8_review_repair/model_inventory.json)
- [图说明](/data/hfosp_group_event_state_v0_3_8_review_repair/final_reports/figures/README.md)；[分支归因图](/data/hfosp_group_event_state_v0_3_8_review_repair/final_reports/figures/group_event_state_v038_dual_branch_attribution.png)
- [产物核验](/data/hfosp_group_event_state_v0_3_8_review_repair/validation/artifact_integrity_final.json)；[回归记录](/data/hfosp_group_event_state_v0_3_8_review_repair/validation/regression_validation.json)
- [执行合同](/data/hfosp_group_event_state_v0_3_8_review_repair/execution_contract.md)

## 后续路线

优先解决仍触顶的基线和新的完整预处理训练来源；确认须使用未参与当前判断的数据。非线性演化需容量匹配转移对照；统一病理状态需同患者同权重贯穿事件预测、多表型和有足够独立发作的迁移。H3需要独立反馈识别设计。不能把延长原observer训练当成这些问题都已回答。
