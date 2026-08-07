# RNNv2.2 event-indexed evolving rank field 验收与收口

日期：2026-08-01  
状态：**开发期 bounded negative；停止 event-driven ELR/RNN**

> **后续边界更正**：这里的停止只适用于未按稳定模板分层的 block-mean ELR 合同。
> block mean 可因固定模板 occupancy 改变而变化，因此本报告不能推翻 PR-2.5 已证明的
> split-half / odd-even 稳定模板，也不能阻止以完整事件为 token、预测未来 repertoire 的
> stable-repertoire event RNN。后者已由 v2.3.1 独立实现和验收。

## 1. 一句话判断

v2.2 已把长期时间轴从事件内部 rank step 正确改成 event chronology，并先证明 block-wise
传播场变化可观测；但在最严格的两位 eligible pilot 中，当前事件历史方向没有提供超过
autonomous drift、persistence、离散切换与 time/IEI controls 的未来 block 增量。因此不再
实现更复杂 RNN，也不能写 activity-dependent shaping。

## 2. 完成度

**完成度：100/100（针对冻结 v2.2 合同）**。

完成的是一次可判决的开发实验，不是得到阳性机制：

- v2.1 已以 `ACCEPTED_AND_FROZEN` 归档；
- pilot 6/6、全 inventory 34/34 的 event time/source/block/rank/tie 字段审计通过；
- Phase 0 非参数 observability、low-rank eligibility 和 chronology null 完成；
- Phase 1 matched fixed/drift/switching/time/event-history 阶梯与 3 类 null 完成；
- 87 项相关测试通过；旧 heldout20、SNN 与病理标签均未进入分析。

## 3. 为什么此前会反复走错分支

根因不是单个 bug，而是把“RNN”当成目标，把科学对象留到模型跑完后再解释：

1. v0.1 的 recurrent time 是事件内 rank step，实际回答 next-rank，不可能回答跨事件网络演化；
2. v2.1 虽加入 contact feedback，仍在同一事件内生成 suffix，科学时间轴没有改变；
3. 评价一度让模板拥有显式 phase clock、让自主模型没有，比较混入信息集差异；
4. SNN compatibility、似然、perturbation 和 identifiability 被压成单一“G0 通过/失败”；
5. 在可观测动态信号尚未建立前投入模型容量，导致工程完成被误当成科学进展。

v2.2 的修复原则是：先审计时间和观测对象，再做非参数可观测性，最后只对 eligible 患者
运行最小、同信息集的模型比较。每一步失败都停止，不再靠架构扩容救援。

## 4. P0 / P1 修复记录

### P0-1：长期时间轴错误

已修为：完整事件是 token，event index 是 primary time；rank step 只描述单事件形状，IEI
只作 secondary nuisance。旧 within-event RNN 保留为冻结 bounded negative。

### P0-2：封顶抽样破坏真实 block adjacency

早期实现把“抽样后相邻”误作“原始 chronology 相邻”。已改为直接保存原始
`within_source_order`，封顶抽样保留真实相邻 pair，并在 Gate 内再次要求至少 10 个 lag=1
pairs。旧状态保存在 `EERF_V2_2_PHASE0_STATE_PRE_ADJACENCY_REPAIR.json`。

### P0-3：middle-contact Gate 计算但未参与授权

早期代码只要求 middle G0 variation，漏掉 middle low-rank gain/basis stability，曾把
`yuquan_zhangkexuan` 误标 eligible。已按原合同修正为六项全过；其正式状态降为
`G0_PASS_LOW_RANK_NOT_ESTABLISHED`，所有数值原样保留。

### P1-1：float32 lag 的假失败

输入审计最初使用绝对 `1e-6`，小于约 60 秒量级 float32 的有效 ULP。现按 canonical raw
值显式转 float32 后逐值比较；pilot 6/6、全 inventory 34/34 通过。这是 dtype 语义修复，
不是放宽任意容差。

### P1-2：middle sensitivity 对 eligible 患者无实际删点

两位 Phase 1 患者的 train mean rank 全部位于 0.2–0.8，故 middle mask 分别为 9/9 与
10/10 contacts。该 sensitivity 在两人中不具 endpoint-exclusion 信息量。它没有造成假阳性，
因为 Phase 1 主 Gate 本身 2/2 失败；但未来不能把这两人的 full=middle 当作独立稳健性证据。

## 5. Phase 0：动态是否可观测

| 患者 | N | field between/within | permutation p | 时间 lag rho | preliminary 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| epilepsiae_922 | 20 | 1.177 | 0.0050 | 0.113 | 有变化；basis null 未过 |
| epilepsiae_620 | 40 | 1.112 | 0.0100 | 0.155 | Phase 1 eligible |
| epilepsiae_1096 | 40 | 1.141 | 0.0050 | 0.170 | 有变化；basis null 未过 |
| yuquan_chenziyang | 40 | 1.162 | 0.0050 | 0.162 | Phase 1 eligible |
| yuquan_zhangjiaqi | — | — | — | — | block field 不可靠 |
| yuquan_zhangkexuan | 80 | 1.397 | 0.0050 | 0.320 | middle basis 未过 |

安全解释：5 位可靠患者的 block-to-block rank/participation field variation 超过 block 内估计
噪声，且有时间连续性。这只说明传播场并非全程完全固定；睡眠、药物、记录状态、event rate
或其他未观测输入仍可解释变化，不能归因于 HFO 事件塑造。

## 6. Phase 1：事件历史是否有增量

预测对象固定为同一 source record 中下一真实相邻 block 的完整 rank+participation field。
所有模型都读取当前 block field；event-history 模型唯一多出的量是同一批完整事件按真实顺序
计算的 `late-half − early-half`。三个 null 分别破坏 block 内顺序、delta-block 配对和正确
circular alignment。

| 患者 | 最强 baseline | event relative gain | order p | block-permutation p | circular p | 判决 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| epilepsiae_620 | autonomous | −0.89% | 0.577 | 0.607 | 0.517 | FAIL |
| yuquan_chenziyang | autonomous | +0.66% | 0.100 | 0.144 | 0.104 | FAIL |

IEI/time covariates没有改善这一结论；discrete switching 也未成为最强基线。两人均不满足
`p<=0.05` 的 chronology-specific Gate，正式决定为 `STOP_EVENT_DRIVEN_ELR`。

## 7. 科学结论边界

### 当前支持

> 在多数 pilot 中，患者特异 rank/participation field 在记录期间存在超过 block 内估计噪声
> 的缓慢变化；稳定 backbone 与时间变化可以并存。

### 当前不支持

> 给定当前传播场后，近期事件 chronology 能稳定预测下一传播场的增量变化。

因此不能写：

- interictal events shape / train the pathological network；
- RNN recovered an evolving pathological graph；
- IEI 揭示了恢复时间常数；
- 跨数据集或独立 cohort confirmation。

## 8. 工程验收

- exact source-record-safe equal-event blocks；
- true original adjacency preserved after subsampling；
- train-only rank prior 与 chronological inner split；
- calibration-only dimension/alpha/state-count selection；
- atomic JSON、依赖 SHA、fail-closed Gate；
- old heldout20 / SNN / A-B / axis / SOZ / ictal / geometry flags 均为 false；
- `pytest`：87 passed。

## 9. 最小下一步

本分支停止，不扩 34 人、不调 K、不加 GRU/过程噪声/一般 graph。论文中若保留，最多作为
Extended Data 或方法学开发阴性：block-wise variation 可观测，但 event-history shaping 未建立。
SNN 继续作为独立的生物物理充分机制，不作为 RNN Gate，也不因本结果重复运行。
