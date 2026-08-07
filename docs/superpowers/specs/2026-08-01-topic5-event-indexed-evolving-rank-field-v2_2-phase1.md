# Topic 5 EERF v2.2 Phase 1：最小事件历史增量合同

> **Closure note（2026-08-01）**：两位 eligible pilot 均未通过冻结 Gate；最终状态为
> `STOP_EVENT_DRIVEN_ELR`。不再增加 hidden dimension、history kernel、GRU、一般 graph
> 或过程噪声进行救援。

## 0. 开放依据与边界

Phase 0 修复版只授权两位 development pilot：`epilepsiae_620` 与
`yuquan_chenziyang`。授权表示 block-wise rank field 的变化、低维重建和 full/middle
basis stability 足以继续检验；它不是 event-driven shaping 证据。

Phase 1 不实现 GRU，不拟合任意 `C×C` graph，也不预测单场随机事件。它先回答一个更窄、
可判别的问题：

> 在已知当前 block 的完整 rank field 后，同一批事件内部沿 chronology 的变化方向，
> 是否还能增量预测下一 block 的传播分布？

## 1. 冻结时间与样本

- 一整场事件仍是 event-indexed update 的基本 token；
- block 只是估计未来传播分布的 observation window，不冒充 recurrent time step；
- 使用 Phase 0 冻结的 patient-specific block size；
- calibration / confirmation 沿用旧 train80 的前 75% / 后 25%；
- old heldout20、SNN、A/B、axis、SOZ、ictal、geometry 全部禁止；
- 本轮 confirmation 已参与 Phase 0 eligibility，因此只能称 development test，不能称
  independent confirmation。

## 2. 可观测状态与事件历史增量

每个 block 的 field 为：

`y_b = [rank_field_b, participation_b]`。

当前 field `y_b` 是所有模型共享的信息。将同一 block 按真实 event chronology 分成前后
两半，定义：

`delta_b = y_b(late half) - y_b(early half)`。

`delta_b` 只表示这一批完整事件中传播场变化的方向，不解释为已证明的 plastic update。
在 calibration blocks 上拟合 PCA basis，Phase 0 选出的 `K` 固定；confirmation 不重选 K。

## 3. 冻结模型阶梯

| 模型 | 输入 | 作用 |
| --- | --- | --- |
| F0 fixed | calibration mean | 固定网络 |
| F1 persistence | `y_b` | 最近状态延续 |
| F2 autonomous | low-rank `s_b` | 线性自主漂移 |
| F3 switching | `s_b` 的离散 Markov state | 少量模板切换 |
| F4 time/IEI | `s_b` + block time、duration、median IEI | 一般时间/速率 nuisance |
| E1 event-history | `s_b + delta_b` | 事件 chronology 的增量 |
| E2 event-history+IEI | E1 + time/IEI | secondary sensitivity |

F2、F4、E1、E2 使用同一 ridge estimator 和同一 calibration-only alpha grid。F3 的状态数
只在 calibration 内选择。所有比较预测同一 source record 中真实相邻的下一 block `y_{b+1}`。

## 4. Null 与 Gate

E1 的增量定义为：

`best(F1,F2,F3,F4) MSE - E1 MSE`，正值为好。

必须同时比较：

1. within-block order shuffle：保持每个 block 的完整事件集合与 `y_b` 不变，只打乱事件顺序
   后重算 `delta_b`；
2. block permutation：在 source record 内重新配对 `delta_b` 与当前 field；
3. non-zero circular shift：保持 `delta` 序列本身的局部结构，破坏正确配对。

每个 null 200 draws，阈值 `p<=0.05`。Phase 1 开放完整 state-space model 的必要条件：

- E1 在 confirmation 上优于最强 F baseline；
- full-contact 下三个 null 均通过；
- middle-contact sensitivity 下增量方向仍为正，且至少 order-shuffle 与 circular-shift 通过；
- E1 不得只靠 IEI/time covariates 获胜。

若失败，结论是“可见 evolving field，但当前事件 chronology 没有提供超过 state persistence /
drift / switching 的未来增量”，停止 ELR-RNN。若通过，才另立 linear state-space ELR 合同。

## 5. 允许表述

Phase 1 即使通过，也只能写：

> 当前 block 内事件历史的方向与下一 block 的传播场变化相关，并超过近期状态、一般漂移、
> 离散切换和 time/IEI-only controls。

不得写因果 plasticity、网络被 HFO 塑造、患者脑内通过学习形成连接。
