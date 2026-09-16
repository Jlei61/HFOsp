# v0.3.9 延长训练复核与下一阶段决定

## 一句话判断

**延长训练后的 N/L 翻转属实；“因此已经充分收敛、人体正对照已校准”的说法不成立。** 上一部分可以按“审阅修复包与有限 pilot”收口，不能按“统一非线性病理状态已建立”收口。

本次直接读取了九张卡、保存的逐 episode 分数、checkpoint/源码 hash、旧480步训练曲线及另一任务的 contact 接口报告。修正结果写入新目录 `final_reports_training_review`，保留原 `final_reports` 与 `final_reports_support_audit`。

## 已经核实的数值

正值表示前者损失更低。

|优化seed|F损失|L损失|N损失|N-over-L|N-over-F|L-over-F|
|---|---:|---:|---:|---:|---:|---:|
|20260905|3.721426|3.640009|3.648195|−0.008185|+0.073232|+0.081417|
|20260906|3.737251|3.651015|3.650343|+0.000671|+0.086908|+0.086236|
|20260907|3.733516|3.629201|3.671062|−0.041861|+0.062454|+0.104315|
|配对中位数||||**−0.008185**|**+0.073232**|**+0.086236**|

九次均 COMPLETE，INNER_PATIENCE=true，选中步960–1600、实际停止步1200–1840；预算2400。L选中1280/1400/1600，确实晚于旧480步。seed20260905 的 F/L/N，旧新前480步曲线逐条相同；仅预算、输出路径及 L 的设备编号不同，输入与训练源码hash相同。

这支持：旧截断夸大了 N 相对 L 的优势。它不支持：这三个 seed 已覆盖训练配方不确定性。三者是**同一 synthetic data seed39001**的优化重复，不能叫三份独立真值样本。

## 本轮确认和修掉的问题

|等级|核验结论|处理|
|---|---|---|
|P1|`known_truth_margin` 用 `not all_limited` 标已收敛，一臂仍预算停止也会被放行|移除由停止原因推断收敛；分开任一预算停止、全部耐心停止、充分性未知|
|P1|用 `truth.seed` 统计训练seed，dict按family覆盖多seed，可能静默非配对|追溯source卡读取优化seed；数据seed/优化seed分别记录；不完整或重复配对拒绝聚合|
|P1|九卡复核读取器未核对status、recipe、源码、episode ID、评分与hash，正文硬写“全部收敛”|严格验证完整3×3配对、停止曲线、模型/评分hash和逐episode均值；停止描述由数据生成|
|P1|原报告一边说仪器能检出转移非线性，一边追加相反结论|旧480步仅保留历史读数；明确N-over-L校准未成立，N/F优势不自动代表人体功效|
|P1|迁移表补了锚点数，但锚点仍不是独立物理支持，跨视图表漏分母|补跨视图锚点数，撤去“三病例支持量与效应反向关系”的过度推断；下一版强制逐块/逐会话输出|
|待修P1|旧训练器无LR调度/精确resume，hidden未接CLI，单步梯度和选中checkpoint不足以认证训练充分|已落新spec；**本次没有宣称人体训练器这些问题已全部修复**|

原始 synthetic 输入hash只绑定 inputs+context，没绑定dt/targets/split。本次在训练源码完全一致条件下重新生成完整合成数据，补存全字段hash；这是事后重建核验，不是伪造为原运行已记录。旧checkpoint未保存optimizer/RNG，无法把重启称为精确继续训练。

自动测试：本轮报告配对/缺失/混源码/分数不一致等回归，加原汇总测试，**18/18通过**。这是审计代码正确性的证据，不是科学假设通过率。

## 模型是否真的小、是否没优化到位

旧人体主 L/N state16、readout hidden32，另有有限 L32 敏感性；并没有系统完成64/128及优化配方比较。AdamW、batch128、clip2和FIT标准化是有的，不能写成“完全没有训练设置”。缺的是公平达到平台后的复核、恢复状态、层级诊断和容量实验，而不只是配置字段。

本次在真实 E253 FIT 长H8样本上完成11个工程配置：L/N state16/32/64与hidden32/64/128、完整F的三种hidden。两卡反向传播无OOM。N64/hidden128=27798参数，L64/hidden128=25734，完整F/hidden128=175622（仅observer+读出）；大模型至少工程上可运行。小批预检不能代表整套训练峰值；不能因预检loss下降而宣称科学收益。

N的U/V初始化较小，需测量它是否在有效状态范围内真正使用了非线性。梯度非零与转移曲率不等于非线性改善留出预测。读出中的非线性也可能解释非线性生成数据；这个区分有受控建模依据。[DPAD原论文](https://www.nature.com/articles/s41593-024-01731-2)

## 之前的结论现在应怎么说

- **间期状态**：有旧版本局部探索线索；FIT-only修复后的本轮主比较没有建立稳定事件净贡献。不是“间期事件没有信息”。
- **可学习与非线性**：参数及长时计算路径可以学习、可追踪；合成数据上L/N均优于这一套F，但N必要性未建立。不能把人体未胜L写成非线性不存在。
- **多种事件表达**：保留局部细形态线索，尚未形成稳定的同状态多端点迁移。另一个任务的严格下一触点结果更弱，并暴露背景干扰和prefix-only基线的重要性。
- **发作关系**：旧E922候选撤回；v039按旧风险模型门槛不可估。不可估的是那个任务配置，不是所有关于发作状态的科学问题。
- **事件反馈**：observer更新、事件重放敏感性和held-out梯度都不是IED改变人体生理状态的证明。

## 发作实验的决定

不再把每患者FIT/INNER/SEL至少3/1/1发作和8h无发作作为所有分析的通用门槛。先让每一个有数据的发作进入轨迹与患者内匹配分析，按发作簇统计，训练期内描述与上游留出区间分开；再按实际支持进入簇外迁移和真正时间前推预测。

少量发作仍能检验明确的病例内关联与空间对应，但不能用大量5分钟窗口制造独立样本。患者自身匹配的设计思路参考[Maclure](https://pubmed.ncbi.nlm.nih.gov/1985444/)；严格前推模型所需的数据条件与[长期植入记录的预测研究](https://pubmed.ncbi.nlm.nih.gov/33341149/)不同。本版具体簇间隔、washout和容量均是预先固定的设计选择，不声称来自通用临床标准。

## 交付与下一步

下一版定为v0.3.10，按[冻结科学spec](group_event_state_v0_3_10_training_and_rare_seizure_spec_2026-09-05.md)及[Agent handoff](group_event_state_v0_3_10_agent_handoff_2026-09-05.md)执行。

优先顺序：现成合成LR复核占用GPU，同时修训练器 → 公平容量/优化比较 → 冻结状态的contact/形态与稀缺发作分析 → 有预算再补独立合成数据、16/24h历史和单视图迁移。不会把8–10小时内全部任务完成或科学阳性预先写进验收。

完整复核：[自动生成白话报告](/data/hfosp_group_event_state_v0_3_9_transition_transfer_background_repaired/final_reports_training_review/group_event_state_v039_closeout_plain.md)、[逐seed配对表](/data/hfosp_group_event_state_v0_3_9_transition_transfer_background_repaired/final_reports_training_review/instrument_budget_paired_seeds.csv)、[机器审计](/data/hfosp_group_event_state_v0_3_9_transition_transfer_background_repaired/final_reports_training_review/instrument_budget_audit.json)。
