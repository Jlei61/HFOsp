# Group-Event State v0.3.7 执行补充：训练性、慢链与长时反馈

**日期：** 2026-09-04  
**性质：** 在正式优化后人体结果产生前完成的执行缺口修补；不是事后改写阳性门槛。

## 1. 为什么需要这份补充

首轮 v0.3.7 已证明连续时间 SSM 的因果性和长记忆实现可以工作，但仍留下四个不能用初版人体数字替代的问题：

1. 只跑了一个旧学习率附近的配方，不能区分科学阴性和训练没动；
2. `S_grid` 只写进模型梯，尚未成为真实五分钟慢链；
3. H2a 缺少完整的 same-prefix、经验支持插值、局部敏感度和 joint interictal sensitivity；
4. H3 只估计一步五分钟关系，没有回答 2–6 小时的 persistent feedback-like dependence。

因此正式读取仍保持在 `FIT/INNER/SELECTION`，development 和 sealed 分区不打开；先补齐仪器和训练合同，再统一出科学报告。

## 2. 新增训练性合同

在四位固定患者、两 seed、三种 observer 上比较 11 个配方，真实改变：

- state 与 readout 学习率；
- Adam 动量；
- weight decay；
- warm-up；
- 非零 state-readout 初始化；
- 小/中状态宽度；
- 最大优化步数。

配方只用 INNER 选择。一个模型家族只有在至少 75% 单元不是第 0 步最优、INNER 中位改善为正、且不超过 25% 单元卡在预算边缘时，才允许进入正式 H1；双流模型的 background-state 与 event-state 两个阶段必须分别满足这三条，不能只检查最后的事件链。没有通过时停止该家族，不能把结果写成生物学阴性。

## 3. `S_grid` 的实现语义

`S_grid` 每五分钟完成一次更新，只汇入严格早于网格时刻、落在前一完整五分钟格内的事件。负荷用总量，grammar 用按事件数加权的条件组成并同时保存有效质量。状态只按真实时间衰减，不因插入零信息事件而额外遗忘。

它与逐事件 `S_event` 使用相同的 0.5/2/6/8 小时目标和评分，不以事件率高低改变物理记忆时间。

## 4. H2a 补充

正式 H2a 同时比较 `S_event/S_grid/S_dual`，producer 和通过嵌套时间审计的成熟 contact decoder 均冻结，只训练低秩 contextual adapter。状态在一次群体事件的每个 decoding step 持续调制，而不是只改变初始 hidden state。

除总体 contact/STOP/grammar 外，必须报告：

- 同一 session、相同前两个 tied groups、且在物理时间上至少相隔两小时的 same-prefix continuation state swap；
- correct-time 与远距离 state swap；
- 经验支持内 state 插值；
- decoder 局部 Jacobian/STOP sensitivity；
- 泄漏当前未来事件的 oracle 灵敏度。

支持性 joint 实验只允许间期 H2a grammar loss 更新 event observer；decoder 主体仍冻结。更新后的 observer 必须重新评分 H1，不能覆盖 primary checkpoint。

## 5. H2b 风险模型补充

旧实现给 72 个五分钟格各自一个自由基线，而部分患者 FIT 只有三次发作。正式版改为六个预先指定的 elapsed-time baseline bands：0–5、5–15、15–30、30–60、60–120、120–360 分钟，仍由同一个离散 survival likelihood 产生各 horizon 风险。

每个 state family 还要通过同一冻结风险读出比较：

- 正确时刻状态；
- 同一患者、同一评价期、至少相隔六小时的 circular wrong-time state；
- FIT 期平均状态。

这防止把患者后段的固定水平偏移写成时刻特异发作易感状态。风险和早期 ictal field 均以独立发作为分母；不足时记 `NOT_ESTIMABLE`。

此外，所有状态臂必须是严格嵌套的残差比较。风险任务先在 FIT/INNER 上一次性选择并冻结六段基础风险曲线及 `clinical/B_history/B_context` 系数，状态臂只拟合追加状态的系数；发作早期空间场也先冻结对应基础场读出，再拟合状态残差。不得让每条状态臂重新估计基础风险或平均发作场，否则几次发作下的截距漂移会再次伪装成跨任务增益。correct-time、错时和 FIT-mean 三臂使用同一个已拟合读出，只替换状态输入。

第一轮可直接评价的发作路径终点是：前 10 秒的群体事件参与场、首次到达时间场，以及有严格时间来源的早期宽带能量场。它们已经比单一 laterality 或 extent 标量保留更多空间信息。若患者缺少对应事件、通道映射或每阶段独立发作，明确记为不可估；不从通道编号伪造空间坐标，也不把不可获得的传播轴补成零。

## 6. H3 persistent 补充

H3 继续与 observer 分开。persistent 模型从冻结的一步 M0/M1/M2 出发，只增加长期 count 或 mark 特征；共同核心、截距和一步边保持冻结。长期边无偏置并显式包含零边候选。

物理尺度资格完全不看 outcome：每个尺度至少需要 FIT 4、INNER 2、SELECTION 3 个互不重叠窗口。当前人体只拟合通过资格的 2 小时和 6 小时；24/48 小时不够就不拟合。每条长期边必须比较同容量、同参数数目的 phase-contained fitted wrong-time placebo，并保留固定六小时延迟和 FIT-mean 对照。

允许结论仍只到 `feedback-like directional dependence`。observer 的事件后更新永远不是 IED 改变生理状态的证据。

## 7. 最终产物

完成后统一生成：

- `optimizer_search/summary.json`；
- `h1_optimized/{event,grid,dual}`；
- `h2a_optimized/{event,grid,dual}` 与 `h2a_joint_sensitivity_optimized`；
- `h2b_optimized`；
- `h3_persistent_feedback_v4`；
- `final_reports/integrated_summary_v2.json`；
- `final_reports/manifest_v3.json`；
- 四张核心科学图及逐图 README；
- 白话版与技术版收口报告。

所有阴性判读必须同时满足：实现检查通过、相应人体模型真实训练、强动态基线存在、常数与错时对照存在、且端点在物理时间上可估。
