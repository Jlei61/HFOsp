# Group-Event State v0.2-B：Seizure transfer（H2b）

开始前完整阅读共同科学合同和工程附录。本线不训练状态：它冻结并读取 registry 中全部 `B_multiscale`、`P_local`、`P_slow` producer，检验只用间期任务学到的状态能否跨任务预测发作。

## 1. 两个并列主任务

### B1：离发作还有多久

在每 5 min fixed-time grid 输出离散 survival/risk：

- 0–5 min；
- 5–15 min；
- 15–30 min；
- 30–60 min；
- 1–2 h；
- 2–6 h；
- 更远或 right censoring。

主指标为 held-out time-dependent log score/Brier、calibration 和 seizure-level ranking；event-conditioned distance probe 作为 secondary，用于与旧事件时刻分析衔接。

### B2：下一次发作早期空间场和传播入口

对每场 held-out seizure 定义前 5 s 的 per-contact normalized energy/recruitment field 为主连续 target；前 10 s 为敏感性。并行描述：

- first recruited group；
- laterality；
- spatial entropy/extent；
- early propagation axis/path；
- IED-to-ictal repertoire reuse score。

在同一发作前 6 h、2 h、30 min、5 min 读取冻结状态，预测同一个 seizure field。6 h 仅在连续 coverage 存在时计入，不用缺失 anchor 补零。

## 2. 为什么 B2 是承重终点

只预测“快发作了”可能来自时钟、近期 rate 或 postictal clustering。若完全由间期事件学习的状态在数小时到数分钟前持续指向同一次发作的早期空间能量场和入口路径，才更接近原始假设：间期网络 repertoire 与患者将进入的发作网络状态相连。

## 3. 冻结输入与 baseline

每个 producer 的 encoder、dynamics 和状态轨迹全部冻结；发作标签不能反向更新状态。对同一 risk-set/held-out seizure 比较：

1. clock/session/coverage baseline；
2. `B_multiscale`；
3. patient-average seizure field（B2）；
4. recent IED repertoire/current event；
5. `P_local` frozen functional state；
6. `P_slow` frozen functional state；
7. state + current event（secondary）。

所有 baseline 加入 time since previous seizure、postictal/refractory indicator、seizure-cluster status、day since admission。sleep/wake、ASM withdrawal/load、stimulation changes只在数据有可靠时间戳时加入；仅有 clock time 时不得写成控制了 vigilance。

主估计量是同一 rows 上 `baseline` vs `baseline + frozen state` 的嵌套增量。state producer 不能依据 seizure 结果选择。

## 4. 发作模式和 target 构造

- seizure ID 使用 recording-code crosswalk，逐发作核对 onset；禁止字符串 inner join 静默丢 Yuquan。
- early ictal field 必须从原始信号/既有经过验证的 seizure artifact 重新对齐，记录 channel order、reference、normalization 和 coverage。
- route/pattern 只能由 TRAIN seizures 或已有临床标签定义；held-out seizure 不参与 clustering、模板、阈值或归一化。
- 支持不足的 route 不强行合并；保留逐发作 field 预测和连续相似度。
- 基本统计分母是 held-out seizure；网格行不冒充独立样本。

## 5. 发作边界与评估

- 只使用 onset 前 trajectory；任何 predictor 不读取 onset 后信号。
- onset 时终止状态轨迹，不能删除 ictal event 后让 RNN 隔着发作继续衰减。
- seizure offset 后排除 60 min，再启动新 segment；对 30/120 min 排除作敏感性。
- rolling-origin：早期 seizure 只用于 TRAIN 目标/超参数，后期 held-out seizure 逐次评估。
- 每位患者报告总发作、可用 held-out 发作、各 lead 有 coverage 的发作、route 数和 censoring。

## 6. 时间特异性

主时间 null 是 within-session block circular shift frozen state，shift 大于对应 lead/target horizon。coarse matched donor 仅为敏感性，匹配 session、time-of-day、coverage、recent rate；不匹配 participation/repertoire，以免把真正跨任务信号消掉。

若 state 只在最后一个 event anchor 有效而 fixed grid 无效，结论是 preictal event phenotype，不是 persistent susceptibility state。

## 7. 执行计划

### B0：support 与 target closeout

1. 生成 patient→recording→seizure crosswalk 和逐 onset 零误差审计。
2. 建 fixed-grid risk sets、censoring、seizure boundary。
3. 构造 early 5 s/10 s field，目视检查至少每队列 3 场 seizure 的 onset/channel alignment。
4. 冻结 TRAIN-only route/field normalization。

### B1：仪器调通

可用 v0.1 trajectory 标 `plumbing_only` 验证 rows、lead、field、censoring 和输出 schema；不得报告承重人体效应。synthetic 只检验 survival/field scorer 能恢复已知信号，不作继续 gate。

### B2：三 producer 全量读取

读取 registry 全部合格 producer，不按 A 结果筛选；缺失显式报错。先固定 3 位发作支持较好的患者 × 3 seeds 验收，再扩所有有 held-out seizure 的 development 患者。

### B3：B1/B2 并行主分析

同批数据同时计算 survival 和 early field lead-time curve；一项不可估计不阻断另一项。按患者/发作 first 聚合，报告 seed spread 和 leave-one-seizure/leave-one-patient sensitivity。

### B4：收口

主图两部分：survival/Brier 增量随 lead time；early ictal field score 增量随 lead time。route、laterality、matched donor 和 event-conditioned distance 放辅助材料。

## 8. 验收和允许结论

- state 超过 `B_multiscale`，但只改善 B1：development seizure-risk transfer。
- state 在多 lead 改善 early field/path：development interictal-to-ictal network-state transfer。
- 只有 recent/current event 有效：preictal phenotype，不称慢易感状态。
- fixed-grid 阳性但 correct-time 不胜 shift：粗慢节律预测，不称时刻特异状态。
- 发作数/coverage/route 支持不足：assay not estimable，不作生物学阴性。
- 任何阳性都不是临床部署性能，也不证明 IED 导致发作。
