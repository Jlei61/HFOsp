# Group-Event State v0.2-C：H3 Scientific Spec 与执行计划

本文件可独立交给 Agent C。开始前必须先读
`group_event_state_v0_2_common_contract_2026-09-01.md`。H3 人体轨迹和反事实分析是主线；synthetic 只作代码校准。

## 1. 科学问题

在已经观测到的患者状态轨迹上，群体 IED 的发生时刻和事件内容是否对之后的慢预测状态留下增量影响？影响是否只持续几十次事件，还是累积到数千/上万次事件与小时尺度？

H3 不再以“rolling exposure 能否预测下一次 size”作为主问题，而是：

> 从同一个 pre-event state 出发，保留或扰动一段群体事件后，之后的 slow functional state 和未见未来事件分布是否系统改变？

## 2. 承重状态对象

原始 `z_slow` 坐标可旋转且每事件必然更新，只作诊断。承重轨迹至少包含：

- slow state 读出的未来 repertoire cluster distribution；
- future participation/extent/multiband field；
- H2b 可用时的 frozen seizure-distance/risk readout；
- fast 与 slow 的 pre/post-event update norm。

H3 只在功能读出上称“状态改变”；单维 latent 位移不能承担结论。

## 3. 观察来源分解

同一患者、相同 decoder 语义下比较：

1. event-only observer；
2. background-only observer；
3. event + background combined；
4. combined 但关闭 event update；
5. combined 但关闭 background correction。

保存两条 update contribution。当前 a4 的状态完全由事件更新加时间衰减，不能把“只有 IED 才更新”当发现；必须由 matched 模型和未来预测判定哪一路含有增量。

## 4. 尺度

事件数主网格：100/1,000/5,000/10,000；覆盖允许时增加更长。真实时间并行：5/30 min、2/6/12 h。每个患者报告实际时间分布、完整 recorded coverage 和非重叠独立窗数。

不为每位患者事后选择最好尺度；所有可支持尺度探索性并排报告。短尺度阴性不 gate 长尺度。

## 5. 人体 perturbation

每个 anchor 固定同一 pre-event state，构造 in-support 干预：

1. `real_sequence`：真实 event time + mark；
2. `no_event_update`：保留 elapsed time/background，关闭 event update；
3. `mark_shuffled`：保留 event times/count/rate，置换 state-matched event content；
4. `time_shifted`：保留 marks，在同 coverage/circadian block 内平移时刻；
5. `burst_removed`：删除预先定义高负荷 burst，同时用匹配 control window 校正缺少的 event count；
6. `state_matched_replacement`：以相同 pre-state、size/rate 的 donor events 替换；
7. `intercept/count_matched`：捕获固定 jump 饱和后变免费截距的旧失效。

扰动 exposure window 后，关闭真实未来 teacher forcing，从共同终态预测下一未来块。比较真实未见 future block 的分布损失，而不是只比较 latent 欧氏距离。

## 6. 自然实验与关联层

counterfactual 模型仍不是随机刺激。并行做患者内自然实验：在 pre-state、time of day、rate、coverage 匹配后，比高/低 IED burst 后的 future functional state。报告 exposure→future state 的关联；只有模型反事实与自然实验方向一致时，才称“支持 event-driven shaping”，仍避免临床因果措辞。

## 7. 执行计划

### C0：轨迹与 schema

先实现读取 shared state manifest，验证 pre/post fast/slow、session/gap、绝对时刻和功能读出对齐。可用 v0.1 做 plumbing，但不产生承重数字。

### C1：来源分解

实现 background-only 与 combined update attribution。先在3位长患者、3 seeds 检查训练和轨迹非退化；然后扩全部有支持患者，不以阳性 gate。

### C2：长尺度 support inventory

按 recorded session/coverage segment 真建窗，报告每尺度 TRAIN/validation/test 的非重叠窗数。滑窗数绝不写成样本量；支持少则保留 case series。

### C3：最小人体 perturbation

先跑 `real/no_event/mark_shuffled/state_matched/intercept-count-matched` 五臂，尺度 100/1,000/5,000/10,000，3 seeds。所有臂从逐位相同的 pre-state 开始、使用相同 decoder 和 future rows。

### C4：物理时间与 burst

并行扩 30 min/2 h/6 h/12 h、time-shift 与 burst removal。不要因 C3 阴性停止；但分别标清 event-count 与 physical-time 结果。

### C5：与 H2b 连接

若 H2b 有可用冻结 risk readout，将它作为一个 secondary functional trajectory；没有则不阻塞 H3 的 repertoire/field 结果。

### C6：报告

白话版用整条患者轨迹说明“删掉/替换这段 IED 后，模型认为之后的一片事件会怎样”；技术版给出每个窗口、尺度、独立分母、状态匹配质量和所有对照。

## 8. 允许结论

- event perturbation 稳定改变未见 future-block 功能读出并优于全部匹配对照：development support for event-driven state shaping。
- 只改变 raw latent：模型内部敏感性，不称状态塑形。
- 只有 rolling exposure 预测下一事件：antecedent association，不称 generator edge。
- event-only 与 combined 相同：背景未提供增量；不是背景生理无关。
- 长尺度不可估计：该患者/设计不可测，不作 H3 阴性。

