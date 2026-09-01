# Group-Event State v0.2-A：H1/H2a Scientific Spec 与执行计划

本文件可独立交给 Agent A。开始前必须先读同目录
`group_event_state_v0_2_common_contract_2026-09-01.md`；冲突时以共同合同为准。

## 1. 科学问题

### H1-short：跨事件预测记忆

过去的完整群体事件是否在模型中留下超过当前事件/近期手工统计的预测信息？

### H1-slow：真实时间慢状态

是否存在一个在几十分钟至数小时内持续、与真实时刻对应、能从同一 anchor 预测一片未来事件分布的状态？

### H2a：状态改变群体事件表达

在控制患者固定习惯、近期事件统计和当前时间后，状态是否改变未来事件的 participation、size/STOP、recruitment delay/order、same-prefix continuation 和 multiband field？

## 2. 状态与输出

保留当前 64-d fast + 32-d slow 架构，第一轮不扩维。每次完整群体事件为一步。保存每个 recorded session 的：

- `z_fast_pre/post`、`z_slow_pre/post`；
- event embedding 与所有一歩预测；
- fitted tau、event update norm、background correction norm；
- session/gap/split/absolute time/event index。

所有轨迹必须从 session 真起点因果 replay。保存原始 latent 供诊断，但承重状态使用冻结读出后的功能量。

## 3. 必做比较

1. `persistent_full`：a4 全输入、fast+slow carry。
2. `trained_memoryless_full`：同 encoder/输入/参数预算，从训练开始每事件 reset。
3. `recent_summary`：最近 1/5/20 事件统计；扩展线性与小 MLP 两个简单版本。
4. `fast_readout`、`slow_readout`、`fast+slow_readout`：同一冻结 checkpoint 上训练低容量 future-block decoder，防止把 head 容量误作状态。
5. `no_real_dt`：只用于区分事件计数与真实时间，不把阴性解释成无状态。
6. `n_streams=1` 对 `n_streams=8`：至少 3 位长患者、3 seeds，量化训练并行切段是否损失长状态。

## 4. H1 测量

### 4.1 reset 诊断

同一 checkpoint 做两套轴：

- 事件数：1/20/100/500/1,000/5,000/10,000；
- 真实时间：1/5/30/120/360/720 min。

reset 必须从 session 起点对齐，报告每个 K 在各患者对应的真实时间分布。它是模型诊断，不单独承担 slow-state 结论。

### 4.2 matched wrong-time

每个 anchor 从同一患者同一 session 选 5–10 个 donor；最少匹配：

- local clock/time of day；
- 最近 1/5/20/100 次事件率；
- time since last event；
- 最近 size/STOP 和 participation burden；
- coverage、gap distance、session quantile。

报告 donor 数、匹配残差和无 donor 比例。比较 correct state 与 donor 中位损失；随机全排列只保留作 easy-null 附录。

### 4.3 多未来预测（承重）

在 anchor (e) 冻结状态，不读取 (e+1\ldots e+h-1) 的真实事件，预测：

- event horizons：1/5/20/100/500；
- physical horizons：1/5/30/120 min；覆盖足够者加 6/12 h。

两类输出并行：

1. **direct horizon**：第 h 次未来事件的 mark/timing；
2. **future block distribution**：未来 H 次或 T 分钟内的 event count、TRAIN-only repertoire cluster occupancy、cluster transition、participation field、size/STOP、delay/order、band energy/peak/cross-band-lag 分布。

repertoire clusters 只用 TRAIN 事件建立；比较线性、Dirichlet/multinomial 与小神经 decoder。简单模型赢应保留为科学结果。

## 5. H2a 测量

分别报告：timing、participation NLL/AUC、group size/STOP、delay NLL、recruitment order、tied groups、same-prefix continuation、band energy/peak、cross-band lag。

H2a 至少分三层：

- 只改善 timing：rate-memory；
- 改善 size/participation：extent state；
- 改善 subset/order/continuation/multiband propagation，且跨多未来仍在：repertoire/network-expression state。

## 6. 执行计划

### A0：warm fix closeout（立即）

- 验证 split 两段显式 carry 与单次 uninterrupted pass 逐位一致；
- validation/test 从 recorded session 起点 replay；
- 结果写新 tag，旧 v0.1 结果不覆盖；
- 因 validation checkpoint selection 受旧 bug 影响，v0.2 承重模型必须重训，不能只重算 test。

### A1：8患者修复复现

固定原中期8位，3 seeds，先跑 persistent/memoryless/recent-summary。不是结果 gate；用于确认评估语义、资源和效应量。

### A2：matched wrong-time + fast/slow 分解

在 A1 checkpoint 上先做冻结后处理；匹配规则只看 anchor 以前。另做3位长患者 `n_streams=1/8` sensitivity。

### A3：多未来和 cluster 分布

先对8位×3 seeds 完成全部 horizon；随后不按结果筛患者，扩到27位可训练 development 患者。短记录患者自然缺失长 horizon，明确分母，不判阴性。

### A4：队列收口

5 seeds 用于预先指定的承重患者/端点复现；3 seeds 队列仍完整报告。输出白话、技术、机器 JSON、逐患者 CSV、fast/slow 功能读出和完整状态 manifest。

## 7. 允许结论

- reset/memoryless 阳性，一步为主：跨事件预测记忆。
- slow-only 在 matched-time、多未来、真实时间轴上有增量：time-specific slow predictive state。
- 多未来 subset/order/continuation 增量：slow repertoire state。
- 只在 teacher-forced 下一步改善：predictive filter，不称慢状态。
- K=100 与 full 未分开：未分辨更长记忆，不称饱和。

