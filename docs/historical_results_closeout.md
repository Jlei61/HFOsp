# 历史结果归档与验收总说明

更新：2026-09-16。本文用于让后续协作者知道：旧路线回答了什么、为什么停、可以复用什么、从哪里找。按用户本次指令，这批历史研究**完成归档验收并结项，不再作为待执行任务挂账**。科学结论按下表分别收口：接受局部结果、限定范围的阴性、测量无效或未完成。归档结项不将它们统一改成阳性，不重新授予主图、患者机制或临床预测资格。

本次覆盖rev22-postfit，以及主目录、continuous-core、substrate和Topic5 training-lab中9月11日前的历史积压；9月11日起的传播优化、无标签拟合、core分岔v7/v8/v9与复发探索仍在[近期结果入口](recent_results_sync.md)，不因本文而被宣布停止。continuous-core和substrate也是新任务的源码依赖，本文关闭其旧实验包，不删除工作树或改动活跃运行。

## 1. 验收总表：做了什么、得到什么、停在哪里

| 历史路线 | 原问题与实际工作 | 结论及不足 | 本次归档决定与后续入口 |
|---|---|---|---|
| **rev22-postfit / DCI** | 将支持、顺序、时差、覆盖分成四类目标，训练响应面与组合候选，完成资格、确认及结构null运行 | 资格30、确认60、null108单元完成；训练汇总90/90完整但仅48/90 primary可估计，部分候选缺条件样本；最终选择盲验证没有完整交付证据 | **阶段结果接受，整体验证未完成而停止**。不宣称可辨识性、唯一机制或患者分布恢复；详见§2 |
| **旧Z分岔 / 旧ZM** | 在各自历史双核与慢变量定义下研究活动转变、分岔和轨迹，形成旧Fig5图件和绘图代码 | 是实际历史模型的条件性动力学结果；手动恢复不能称自主终止，混合阈值/输入版本不能替代新版纯降阈值core；旧操作性高活动也不自动成为临床发作 | **历史模型/图件接受归档**，不回填当前Fig5；源版本和41项旧图已在9月11日整合包保存，见§4 |
| **观测修复、多维筛查与同网分布学习v2.1** | 固定检测尺度、独立通道计数、孤立窗和质心/募集定义；按训练反馈迭代并在新拓扑/新噪声确认 | v2.1正式156单元完成，114非runaway、42物理runaway、0工程失败；候选确认L_off=0.979，对old-joint为1.079，但患者条件路线、支持和包络时序仍有缺口 | **工程结果与局部分布改善接受；机制恢复不接受**。停止旧搜索，转读出与时间表征审计；不继续靠标签比例解释传播恢复 |
| **native-activity正则与contact-native pilot** | 用粗原生场支持先验配合五项观测目标；首波12单元，整合pilot32单元（12继承、20新增） | 局部形态/原生场先验可改善，但招募和完整包络没有一致改善；继承首批不能倒称新版优化，2张网络与旧review数据不能冒充盲确认 | **有界pilot与取舍结果接受归档**，不冻结模型、不自动增加wave或进入Fig5 |
| **initial-state v1** | 固定结构与逐步配对输入，三臂×12噪声，以两核等量细胞的1mV冷启动偏置测持续模式偏好 | 36/36完成；晚期B1−B2约−0.9百分点，95%区间[−9.0,+6.3]，没有支持超过10点的持续偏好；三臂条件传播仍在患者匹配带之外 | **该工作点/幅度/窗口下的有界小效应结论接受**。第二图门槛未触发；回到结构/观测缺口，不外推为所有初态与慢状态无作用 |
| **continuous-core-state r1** | 固定一张历史网络，用五个固定状态和外源OU重分配两核I输入；18单元、3噪声重复 | 18/18完成、0runaway；响应可随状态变化，但固定状态事件稀少，连续OU事件数低于零状态；比例变化没有证明患者条件传播恢复 | **首轮干预响应接受归档**。不称内源双稳态、纯初态因果或宽状态域；不自动加剂量/换底物 |
| **state-S native-Z pilot** | 固定s与Z条件的18条历史执行与诊断，保留慢变量/底物身份 | 已有结果可重分析，不能由阶段完成推导自主终止、临床发作或通用状态机制；M关闭与固定s范围必须保留 | **有界pilot接受归档**，复用代码/结果而不升级主张；见9月11日整合验收 |
| **RNN contact-bridge v0.3.8** | 核对状态与触点读出的桥接接口，重汇总35个既有模型读出 | 父模型一致性与接口诊断可以复核；接口无误不等于丰富事件状态成立或临床解码可靠 | **接口诊断与重汇总接受归档**，保留35个读出及数据来源；不替代GES科学验收 |
| **Topic5 event-indexed v2.2** | 把时间步纠正为完整事件，审计block传播场，再比较匹配的历史/漂移/切换基线 | eligible pilot未见超出强基线的未来block增量；block均值仍混合模板占比变化 | **仅该block-mean合同的有界阴性接受**，旧ELR/RNN停止；不推翻稳定模板，后继stable-repertoire另立合同 |
| **stable-repertoire v2.3–v2.4** | 匹配历史量、recency与chronology null，预测未来repertoire | v2.4锁定扩展未选出额外PCA＋单衰减leaky state；较多历史主要提高稳定repertoire估计精度 | **固定低维滤波族阴性接受**。不能扩成所有可训练RNN阴性；v2.5–v2.7是后继不同模型族 |
| **stateful-event RNN v2.7** | 34人×3seed及reset、memory curve、shuffle、reverse、H40控制，检验事件历史状态跟踪 | 已有验收为ACCEPTED_REPAIR_ONLY_STATE_TRACKING_FINAL；支持记录内state tracking，不能推导事件改变网络、因果可塑性或图连接识别 | **沿用原有限科学验收**，不重复挂账；代码积压本次补存，后续机制问题见v3.0 |
| **innovation v3.0 / v3.1门** | 检验单事件残差与累积残差是否提供未来增量 | 两条路线均未跨预定门；累积正号对样本支持敏感；34人中仅17人满足innovation资格，未合格者不是科学阴性 | **Level 1 leaky observer收口接受**；v3.1人体transition为NOT_TRIGGERED，不事后放行 |
| **GES v0.3.5–v0.3.9** | 从旧门控RNN改到连续状态，修复训练对照、接口和FIT-only输入边界 | v035/036有效训练记忆不足以测6–8小时；v037–039修复后仍只有局部开发线索，未建立跨患者丰富事件状态、事件内传播调制和发作迁移 | **测量无效版本撤回；修复工作包与有界线索接受归档**。训练不足、标签支持不足不改称科学阴性 |
| **GES v0.3.10–v0.3.12** | 恢复丰富事件数据、检查历史长度/时间合同、前推与回顾性状态、冻结下游读出 | v0311数值原型可保留但科学链路不完整；v0312主计划67/67完成，丰富前推/共同状态NOT_ESTABLISHED，仅LOCAL_SUPPORT_COARSE_ONLY，S-C不可估计 | **工程工作包及有限粗状态线索接受**。功效未校准、细触点无增量，不扩大模型容量或写成临床风险阴性 |
| **GES v0.4.0** | 统一状态Q、共同条件C、发布边界、形态目标与消费者；G0合同检查完成 | 原interim_summary不是终态；现有queue文件29 COMPLETE、1 FAILED（t027缺old_summary_z），与原47项计划不等价；合成资格/完整人体科学闭环未完成 | **接口定义和实际已完成部分接受；剩余工作历史停止**。不能把旧RUNNING或G0通过写成人体结果，见§3 |
| **旧稿件、主图生产与临床描述分析积压** | 保存原论文版本、图登记、producer、SOZ/关联描述、参数与测试文件 | 同一路径存在版本差异和307项本地删除记录；4个冲突尚未解决；这些是版本材料，不构成新的科学实验 | **版本材料归档接受**，保留原科学/图件状态；不覆盖main正式图登记，不因历史归档宣称新增人工验图通过 |

以上具体数字来自本次归档的原报告/机器记录，不是本次重新训练或仿真结果。每种模型的阈值、噪声、观察器和数据版本只在自己的合同内成立；跨版本结果不能拼成一条已经验证的机制链。

## 2. rev22-postfit：这次明确结项的历史分支

旧分支在冻结训练阶段可确认：响应面有66条feasible row，四个目标分量均进入识别模型，冻结5个候选。阶段汇总为90个资格/确认单元，其中48个primary可估计。p030和p089条件分量完整，p000/p066/p075仍有CONDITIONAL_COMPONENT_NOT_ESTIMABLE。这解释了为什么不能把训练分数改善等同于完整患者模式恢复。

结构null的108单元controller虽已COMPLETE，最终链状态仍停在OPENING_SELECTION_BLIND_VALIDATION。核查没有找到最终completion_audit和完整最终验证交付；日志有空切片警告，未发现存活的rev22计算进程。**本次按用户要求停止在已有阶段成果，归档状态为“历史已结项／最终科学验证未完成”**，不重新启动该链。

- [独立分支结项说明副本](archive/workspace/historical_closeout_2026-09-16/REV22_HISTORICAL_CLOSEOUT.md)
- [训练阶段汇总](archive/workspace/historical_closeout_2026-09-16/evidence/rev22/frozen_stage_aggregate/frozen_stage_aggregate.json)
- [候选冻结](archive/workspace/historical_closeout_2026-09-16/evidence/rev22/response_fit/frozen_candidates.json)
- [响应面](archive/workspace/historical_closeout_2026-09-16/evidence/rev22/response_fit/response_fit.json)
- 原始结果：`/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/data_driven_dual_core_interictal_identifiability/`。
- 四个原未推送修复：836d1d2d、c5f0eca8、e0d87055、6e9ce3ac；已随`codex/topic4-rev22-postfit`的归档提交保存。

如未来重开，应先重新指定资格与独立验证合同，解决条件样本和输入身份；不得直接续写为本版已经通过。

## 3. Topic5：不要把旧状态文件当作最终验收

v0.3.12完整记录见[窗口收口](archive/workspace/historical_closeout_2026-09-16/evidence/topic5_ges/group_event_state_v0_3_12_window_closeout_2026-09-07.md)和[机器验收](archive/workspace/historical_closeout_2026-09-16/evidence/v0312_final/acceptance_decision.json)。合成扩展的56完成、2失败、2前置阻塞必须保留，不能只报成功任务。已有“未建立”不等于生物机制不存在。

v0.4.0的[旧interim快照](archive/workspace/historical_closeout_2026-09-16/evidence/v040/report/interim_summary.json)只记录11完成、2运行中、34未运行。更晚的[queue证据目录](archive/workspace/historical_closeout_2026-09-16/evidence/v040/queue_state)有30份记录（29完成、1失败），本次不把旧快照当最后进度。t027为synthetic_fit，实际失败为KeyError: old_summary_z；原计划47项不能由30份队列文件自动补足。G0六项接口检查成立，仍不能代替合成仪器资格、三种子人体与发作闭环。

收口方式：保留共同条件、合法发布和接口修复；完整科学目标标记为未完成并归档。未来若继承，先修合成oracle输入合同、核对计划/执行清单和验收来源，再决定是否另立新包。本次不修模型、不追加训练。

跨事件旧线请分别读：[v2.2](archive/workspace/historical_closeout_2026-09-16/evidence/topic5_sequence/event_indexed_evolving_rank_field_v2_2_review_2026-08-01.md)、[v2.4](archive/workspace/historical_closeout_2026-09-16/evidence/topic5_sequence/stable_repertoire_event_history_v2_4_acceptance_2026-08-02.md)、[v2.7](archive/workspace/historical_closeout_2026-09-16/evidence/topic5_sequence/stateful_event_sequence_rnn_v2_7_acceptance_2026-08-03.md)、[v3.0](archive/workspace/historical_closeout_2026-09-16/evidence/topic5_sequence/event_innovation_v3_0_acceptance_2026-08-03.md)。它们检验不同时间单位、模型族与证据层，不以“RNN成功/失败”总括。

## 4. 找回历史结果的方法

先读本文定位研究线，再读下面原报告；复现实验必须同时取得对应源码分支、配置与原始数据，不用main当前同名模块代替历史模型。

| 内容 | main内保存的入口 | 原始数据位置 |
|---|---|---|
| v2.1分布学习 | [科学审阅](archive/workspace/historical_closeout_2026-09-16/evidence/v21/scientific_review.md) | `.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/multievent_distribution_search_v2_1/` |
| 原生场与contact pilot | [整合审阅](archive/workspace/historical_closeout_2026-09-16/evidence/contact_native/scientific_review.md) | 同工作树`results/topic4_sef_hfo/contact_native_integrated_pilot/` |
| initial-state | [三臂结果](archive/workspace/historical_closeout_2026-09-16/evidence/prior_closeout/initial_state_conditioned_propagation_round_v1_2026-09-08.md) | 同工作树`results/topic4_sef_hfo/initial_state_conditioned_propagation_v1/` |
| 连续慢状态r1 | [科学报告](archive/workspace/historical_closeout_2026-09-16/evidence/continuous_state/scientific_report.md) | `/data/hfosp/topic4_sef_hfo/continuous_core_state_r1_20260909/` |
| 旧Z、ZM、state-S与contact-bridge | [9月11日整合记录](archive/workspace/integration_2026-09-11/README.md) | 该记录中的preserved_branches和file_mapping；旧图位于`results/paper-ready-figure/archive/2026-09-11_legacy_z_integration/` |
| GES v0312 | [最终报告集合](archive/workspace/historical_closeout_2026-09-16/evidence/v0312_final) | `/data/hfosp_group_event_state_observable_state_validation_v0312/final_reports/` |
| GES v040 | [合同与执行证据](archive/workspace/historical_closeout_2026-09-16/evidence/v040) | `/data/hfosp_group_event_state_epilepsy_state_v040/` |
| 所有历史修改与旧文档 | [源码保存清单](archive/workspace/historical_closeout_2026-09-16/source_manifest.json)、[文档副本](archive/workspace/historical_closeout_2026-09-16/source_context) | 各条记录的source、base、branch、commit和逐文件哈希 |

源码历史分支：`codex/archive-history-20260916-root`、`codex/archive-history-20260916-continuous-core`、`codex/archive-history-20260916-substrate`、`codex/archive-history-20260916-training-lab`；rev22沿用`codex/topic4-rev22-postfit`。固定提交见[验收记录](archive/workspace/historical_closeout_2026-09-16/acceptance.json)。按固定commit恢复，不假定分支名称永远指向同一版本。

本次把1096个历史修改文件保存为源码快照，部分内容曾在其他分支保存，不把它们都称为首次上传。root与continuous快照接在9月16日近期快照之后，以保留新旧代码背景，但**只将清单中9月11日前的工作认定为历史结项**。当前工作树和暂存区不改写，4个冲突与307项删除仍保留；删除对象可从原HEAD/远端历史恢复。本次没有清理数据盘、工作树或临床原始数据。

[证据清单](archive/workspace/historical_closeout_2026-09-16/evidence_manifest.json)记录本次保存的报告/表/状态原路径和SHA256。大型逐事件数组、模型权重与checkpoint留在源盘；可读文档有历史绝对路径时，以清单提供的归档副本为先。归档不等于数据盘备份，也不保证移走所有外部依赖后可独立重演。

## 5. 验收执行规则

本轮已核查研究问题、既有结果与停止边界，保存原始结论、失败与未完成记录，确认关键报告/机器证据存在，并提交远端。归档状态统一为HISTORICAL_ARCHIVE_ACCEPTED；科学状态逐行保留，不使用一个总分替代。

新增核验只覆盖文件完整性、文档链接、源码语法和rev22相关修复回归；不冒充重新跑过所有旧实验，不覆盖旧结果中的失败。候选图没有因这次文档验收新增“用户已目视通过”的标签。后续协作者可以引用有边界的历史结论或复用实现；任何重开、扩样本、换模型和进入主图均应有新的研究合同。
