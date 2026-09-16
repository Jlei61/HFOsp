# 间期事件状态：文献对下一版设计的约束

检索日期：2026-09-06。目的不是替现有模型寻找背书，而是区分：事件表达状态、长时动力学、发作关联、前瞻风险和事件反馈。以下筛选 15 篇原始研究/方法论文；可访问全文者核对方法与限制，其余只使用出版社或 PubMed 摘要明确支持的信息。这不是系统综述，没有声称穷尽文献。

## 1. 时间尺度与疾病表达

| 原始研究 | 支持什么 | 对本项目的约束 |
|---|---|---|
| [Fouad 等，2022，Brain Communications](https://pmc.ncbi.nlm.nih.gov/articles/PMC9724782/)：Interictal epileptiform discharges show distinct spatiotemporal and morphological patterns across wake and sleep | 睡眠/清醒、脑区及 SOZ 与 IED 发生率、形态及空间表达有关 | 背景与睡眠既可能是混杂因素，也可能是共同生理状态的观测。必须同时报告事件信息的总预测价值、给定背景后的增量；后者为零不能否定前者。时钟不能冒充睡眠标签。 |
| [Laminar organization of cellular microcircuits modulating human interictal epileptiform discharges，2026，Nature Neuroscience](https://www.nature.com/articles/s41593-026-02258-4) | 人体单神经元和皮层层次活动在约秒尺度组织 IED，并包含即将出现的 IED 及其特征的信息 | “事件背后有状态”不要求首先证明两小时提前预测。该研究的分辨率、记录场景与本项目宏电极/小时测量不同，不能据此规定本项目应得到相同效应。 |
| [Smith 等，2022，eLife](https://elifesciences.org/articles/73541)：Human interictal epileptiform discharges are bidirectional traveling waves echoing ictal discharges | 10 位患者的微电极记录揭示 IED 与 ictal discharges 的传播路径关系及双向结构 | 支持把触点、次序、方向及患者内网络联系设为独立端点；不证明跨小时状态、更不证明 IED 造成发作。 |
| [Diamond 等，2023，Brain](https://pubmed.ncbi.nlm.nih.gov/36729683/)：Interictal discharges in the human brain are travelling waves arising from an epileptogenic source | 40 位侵入式监测后接受手术患者；IED 波源与发作源/传播有关 | 宏电极研究与微电极的双向模式并不完全一致。模板/方向必须在本数据独立构建，不能把文献的双向图样设为必然真值。 |
| [Diamond 等，2024，Nature Communications](https://www.nature.com/articles/s41467-024-51338-1)：Focal seizures induce spatiotemporally organized spiking activity in the human cortex | 五位患者的皮层群体放电序列在发作中变得刻板，并偏离基线/IED 序列 | “共享网络骨架”允许存在，“整套事件内部序列完全统一”需另外检验。保留状态依赖表达与固定网络 trait 的区分。 |

## 2. 长时状态与发作预测

| 原始研究 | 支持什么 | 对本项目的约束 |
|---|---|---|
| [Baud 等，2018，Nature Communications](https://www.nature.com/articles/s41467-017-02577-y)：Multi-day rhythms modulate seizure risk in epilepsy | 37 位植入设备患者，长达多年记录中的 IEA 与发作呈日内/多日节律 | 小时计数可以携带慢信息，但数据时长远超短期术前监测。不能把 24h 历史窗口等同于可靠识别多日周期。 |
| [Proix 等，2021，Lancet Neurology](https://escholarship.org/uc/item/4x778199)：Forecasting seizure risk in adults with focal epilepsy | 长期设备记录的发作风险开发/验证；纳入要求包含至少约六个月的小时 IEA 和至少 20 次发作 | 这是风险预测研究的资料条件，不是所有“发作相关状态研究”的最低门槛。少量发作仍可做冻结状态的患者内关联与空间对应。 |
| [Rosch 等，2025，Advanced Science](https://pubmed.ncbi.nlm.nih.gov/40192017/)；[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12199362/)：Epileptiform Activity and Seizure Risk Follow Long-Term Non-Linear Attractor Dynamics | HAVOK 用延迟嵌入、近线性主系统及 forcing 表示慢 IEA 动态，并关联发作风险 | 非线性系统可以有有用的扩展线性表示；“N 没赢 L”不是系统线性的证据。forcing、嵌入和未来信息可用性必须审计，不能把 driven reconstruction 叫自主预测。 |
| [Yang 等，2026，Epilepsia](https://onlinelibrary.wiley.com/doi/10.1002/epi.70110)：Seizure forecasting with epilepsy cycles: On the causality of forecasting pipelines | 在同类长期资料上，改成只读过去的周期提取后，预测性能明显下降；时间滞后和边界变形是关键 | 每条输入记录 event time、raw support end、release time。按真实可用时刻截断重算是必要测试。论文中的 causal 指“不使用未来数据”，并非 IED 对发作的生理因果。 |

## 3. 非线性、潜状态与漂移

| 原始研究 | 支持什么 | 对本项目的约束 |
|---|---|---|
| [Sani 等，2024，Nature Neuroscience](https://www.nature.com/articles/s41593-024-01731-2)：DPAD | 分开神经输入、潜状态递归、神经读出、行为读出的非线性；在其运动任务中，很多非线性可定位于读出 | 至少比较线性/非线性转移 × 线性/非线性读出，并用已知来源的合成世界校准。单个 N/L 胜负不定位非线性来源。DPAD 的行为监督不能直接当作本项目冻结发作结局的训练目标。 |
| [Pandarinath 等，2018，Nature Methods](https://pubmed.ncbi.nlm.nih.gov/30224673/)；[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6380887/)：LFADS | 从单试次群体放电推断潜动态，并用行为关联等外部任务验证；论文讨论重建损失的局限 | 重建得好不是潜状态恢复成功的充分证据。回顾性序列编码不能未经改造就用于未来预测；预测时应只读过去。 |
| [NoMAD，2025，Nature Communications](https://www.nature.com/articles/s41467-025-59652-y) | 猴运动皮层记录中，通过固定动态模型并调整观测映射，实现跨天解码稳定 | 提供“观测通道变化”和“潜动态变化”可拆开的思路；不能证明癫痫的所有跨期变化都是技术漂移，更不能对齐掉真实发作前状态。 |
| [MARBLE，2025，Nature Methods](https://www.nature.com/articles/s41592-024-02582-2) | 通过局部流形几何/动态表示比较不同条件和系统的活动 | 比较动态应重视坐标无关或输出空间量。本文不是证明本项目需要换成 MARBLE；原 PCA 轴的正负号本来就不能承载生理方向。 |

## 4. 少量发作的可用设计

| 原始研究 | 支持什么 | 对本项目的约束 |
|---|---|---|
| [Maclure，1991，American Journal of Epidemiology](https://pubmed.ncbi.nlm.nih.gov/1985444/)：The case-crossover design | 用同一人的病例时刻与参照时刻研究短时暴露与急性事件的关联 | 适合本项目的患者内关联层；不需要先满足临床预测器的事件数。但病例/对照抽样概率不能直接当连续人时上的风险。 |
| [Shahn 等，2023，Biometrics](https://onlinelibrary.wiley.com/doi/full/10.1111/biom.13749)：A formal causal interpretation of the case-crossover design | 明确 case-crossover 的因果解释依赖额外假设，并分析异质性等影响 | 自身匹配并不自动控制时间变化的共同原因。这里先称关联；控制睡眠、记录阶段和既往发作也不等于已识别 IED 的生理反馈。 |

## 5. 三轮反思后留下的决定

**第一轮：多训、加宽、更多历史是否够？不够。** 对照尚有预算停止，固定历史读出饱和，新训练器没有自己的独立合成功效校准。先修这些是必要条件；修好仍只回答当前两小时代理任务。网络容量不是原始科学问题。

**第二轮：既然怀疑漂移，就对齐所有状态分布？不采纳。** PCA 中位数同侧不是范围外预测；坐标符号不是生理方向。即使真实存在分布变化，也可能是疾病状态本身。下一版先分技术观测变动、状态变化及读出失配，再做 FIT-only 可逆缩放和明确的观测映射敏感性。

**第三轮：回到“患者内事件表达的可预测状态”，再逐层问演化、发作和反馈。采纳。** 保留 2h 旧任务作为复现/次级任务；新增真实可用时间上的事件前状态与条件形态端点。冻结同一 checkpoint 跨视图验证。少量发作先做关联和固定空间预测，连续风险只在完整风险集就绪后报告。模型非线性、预测状态和生理反馈各自验收。

本轮执行了四个反例计算：中位数同侧但全部在训练范围内；潜坐标符号反转但预测完全不变；几乎完全线性的 tanh 分支仍有很大分支幅度；改变病例抽样比例会改变最优常数 Brier。见[独立审计数据](/data/hfosp_group_event_state_v0_3_10_independent_review/audit_summary.json)。它们检验推理有效性，不冒充新方案已通过完整训练或功效校准。
