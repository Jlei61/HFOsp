# v0.3.9事件状态与冻结迁移设计pilot收口

**v0.3.8审阅修复包已收口，v0.3.9有限设计pilot已执行完成；原始统一、非线性病理状态和IED生理反馈闭环仍未建立。** 三个病例E1096/E1125/E253均为已见设计资料，没有独立确认资格；没有打开development/sealed。

## 核心回答

- 主比较明确区分完整固定历史F、可学习线性转移L、非线性转移N，以及历史H=0.5/8h、预测提前量0/2/6h和目标宽度0.5h。三个病例的joint H8 INNER家族分别为N/F/L；主提前量2h的9条优化重复均无正的事件净贡献，不能用模型相对名次替代超过背景/常数的收益。
- 已测到真实训练更新和6–8h事件到实际留出损失的梯度；72个冻结状态的逐事件重放和有限差分均通过。270张人体卡中62张事件阶段预算受限、67张选择初始事件残差，不能统称训练充分或据此作科学阴性。
- 同一checkpoint的未训练表达探针有局部线索：E1096的N状态频带时间质心在3个优化seed上超过父基线、常数、完整F和初始化；E253计数功能读出到粗招募有不一致的跨任务收益。contact同prefix下一集合没有稳定完整支持，不把不同任务、病例或初始化信息拼成一个机制。
- 发作风险网格重新按过去支持生成。原始FIT/INNER/留出发作为E1096 3/1/4、E1125 6/0/0、E253 1/0/1；满足历史支持且对应下一发作的上界变为2/0/1、2/0/0、1/0/1。三例风险及早期空间任务均NOT_ESTIMABLE，没有拟合风险/空间读出，也不称发作关系阴性。
- 已知真值的强非线性转移场景中N超过L/F；独立原因和无事件反馈反例仍能出现原始latent迁移。这是单seed仪器诊断，不是患者条件下的全流程功效证明。H3的模型更新和生理反馈继续分开。

## 实际修复与执行记录

旧包整合对照封底、源码版本、错时/物理窗口和基础臂原点诊断，撤掉无定义总分。继续追源发现全记录触点选择影响前瞻边界，按FIT-only重建466个测量块和9个contact decoder。跨处理段事件保留可确定时钟与参与，细mark缺失；背景剔除60/215/540个受恢复事件core污染的窗口，并给E253一个零事件短块补6窗。所有依赖人体拟合使用修正版背景，新旧事件输入/历史/目标逐项一致。

人体有限清单为108张主比较、126张敏感性、36张单视图。按INNER冻结54个主状态及18个单视图状态，逐一完成导出、真实梯度审计、细表达及contact迁移。原配方不因收益正负扩大预算；GPU按队列接续，登记GPU工作完成后不运行占位任务。

两次初次CUDA运行失败保留：E1096 F rollout非法指令、E1125 F count-view索引断言，均发生于GPU 1。每张一次相同源码、输入、种子和预算的同步重跑成功；原队列失败状态与日志不覆盖，最终监督状态明确标记运行恢复。根因仍未定位，不称“零失败”或“硬件已排除”。

最终汇总核对5086项源码、输入和产物hash；各正式人体生产角色内部未混源码。参数清单逐层记录shape/数量，训练卡记录LR、AdamW、初始化、实际batch、归一化、更新和选中步。22项v039回归通过；旧v038修复包的115项回归另行保留，不合并成独立科学证据。

## 正式交付与下一步

- [白话报告](/data/hfosp_group_event_state_v0_3_9_transition_transfer_background_repaired/final_reports/group_event_state_v039_closeout_plain.md)
- [技术报告及每层参数索引](/data/hfosp_group_event_state_v0_3_9_transition_transfer_background_repaired/final_reports/group_event_state_v039_closeout_technical.md)
- [机器汇总](/data/hfosp_group_event_state_v0_3_9_transition_transfer_background_repaired/final_reports/summary_main.json)、[完整性核验](/data/hfosp_group_event_state_v0_3_9_transition_transfer_background_repaired/final_reports/artifact_integrity.json)
- [五张图说明](/data/hfosp_group_event_state_v0_3_9_transition_transfer_background_repaired/final_reports/figures/README.md)：PNG/PDF/SVG及生产代码，最终视觉核验记录随收口文件保存；不代替作者接受。
- [独立确认合同](group_event_state_v0_3_9_independent_confirmation_contract_2026-09-05.md)：先核验新资料的支持、发布延迟和全训练流程功效，再锁定确认病例与端点。IED反馈另以共同原因/无反馈、计数反馈、内容反馈模型检验，三者都必须允许观察更新。

正式人体根为`/data/hfosp_group_event_state_v0_3_9_transition_transfer_background_repaired`；旧背景结果和未修复输入队列保留在平行旧根，未混入最终人体汇总。没有提交、推送或发布代码。
