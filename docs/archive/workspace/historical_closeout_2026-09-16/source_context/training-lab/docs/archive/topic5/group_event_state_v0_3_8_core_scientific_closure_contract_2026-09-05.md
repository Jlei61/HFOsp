# Group-Event State v0.3.8 核心科学闭环合同

**日期：** 2026-09-05  
**状态：** 执行中  
**结果根：** `/data/hfosp_group_event_state_v0_3_8_core_expansion`

## 一句话目标

从连续群体间期事件和非事件背景中学习一个严格因果的预测状态，分别检验它是否随真实时间变化、是否真正使用多小时历史、是否同时预测多类 IED 病理表达，以及冻结后是否与发作风险、发作距离或发作早期传播场存在患者内关系。

## 四个必须回答的问题

### 1. 是否存在动态 predictive state

同一患者、同一批时间锚点上依次比较：

1. `B_rate`：强多尺度事件率和负荷历史；
2. `B_mark`：固定多尺度 marked-history EWMA；
3. `S_event`：学习型 event-only CTSSM；
4. `S_dual`：event history + 非事件背景双流 CTSSM；
5. `S_grid`：五分钟层级状态。

任何学习状态的增量都必须同时对照：

- FIT 期均值形成的常数状态；
- 同 session、按目标 horizon 做的时间平移状态；
- 同容量随机 observer；
- 已充分训练的 `B_rate` 和 `B_mark`。

只有“学习状态胜强基线、常数解释不了、正确时刻胜错时”同时成立，才称时刻特异动态 predictive state。常数改善单独报告为患者/阶段慢水平，不删去，也不升级为动态状态。

### 1.5 状态是否真的被学到，并使用了长历史

这分成三层，不能互相替代：

1. **架构能力：** CTSSM 的真实时间 impulse response、并行 scan 梯度和稳定性检查通过；
2. **训练发生：** 选中 checkpoint 不在初始化、第一步梯度非零、参数有实际更新、训练预算未触顶；
3. **人体利用：** 从 held-out 多端点损失反传到同一 carry segment 内的完整真实历史事件，逐锚点核对重建状态与冻结轨迹一致，再分别统计 `<0.5 h`、`0.5–2 h`、`2–6 h`、`6–8 h`、`8–16 h`、`16–32 h` 和 `≥32 h` 的梯度质量及每事件平均梯度。正式候选要求至少两个非同义端点在至少 3/5 seed 中有不低于总梯度 1% 的 `>6 h` credit，并且至少三个 held-out 锚点可审计；单纯数值非零不算通过。

“固定 tau 的公式允许长梯度”只属于第一层；只有选择后的人体 checkpoint 在第二、三层也成立，才能说模型实际学到并使用了多小时历史。旧事件具有 observer credit 不等于 IED 在生理上改变了脑状态。

### 2. 同一状态是否预测不止一种病理特性

H1 future-block 同时输出：

- event count 与 burden；
- community occupancy；
- cross-community coupling；
- repertoire mixture 与 continuous embedding；
- 条件 multiband、cross-band lag、连续 contact delay 和 waveform summary。

H2a 把冻结状态接入独立预训练、严格时间切分并冻结的 contact-sequence decoder，在整个事件内逐步调制：

- continue/STOP；
- positive group size；
- contact subset identity；
- route/order 与 later continuation；
- same-prefix 后续分叉；
- conditional rich mark。

主张“病理状态”至少需要 contact subset、continue/STOP、same-prefix continuation、conditional rich mark 中两个非同义端点，各自在至少 3/5 seed 中同时胜过 `B_mark`、FIT 常数和错时状态；同时要求 adapter 确实探索过且泄漏正对照可检出。aggregate grammar 作为总分报告，但不和其 contact/STOP 组成部分重复计数。仅 count/rate 阳性仍只称负荷状态。所有端点都报告，不按显著性挑选。

### 3. 冻结间期状态是否与发作有关

全部 H1 checkpoint 和间期特征先冻结并写入全局 marker，之后才允许打开发作结局。H2b 并列评价：

- 固定时间网格上的离散 hazard、censored log score，以及 person-period Brier/calibration-in-the-large；
- 距下一次发作的 survival/distance；
- 不同 lead time 的发作早期 ictal spatial field；
- 发作早期 route/path。

并行保留四条冻结来源：event-only state、五分钟层级 state、persistent non-event background state，以及 background + event 的双流 state。persistent background 的主比较以当前背景窗口 + 强 marked history 为固定基线，并同时对照同容量随机背景、FIT 常数和错时轨迹；这样不会因为只检验 event residual 而漏掉共同背景慢状态与发作的关系。

persistent background 的 frozen-random 容量对照统一使用 `dual_random_background_long_v2/` 与 `dual_random_background_medium_v2/`。旧目录是未解决旧 checkpoint 未保存 repertoire 字典所导致标签对称性的 v1 产物，不得与 v2 混用；v2 仅用 FIT 段对齐 repertoire 标签，并逐端点审计 frozen-parent parity。旧 checkpoint 同样没有保存 randomized-PCA 压缩后的 grammar target，因此 `mixture` 或 `mark` 若只有预设的小幅重建漂移，该端点必须标为不可审计且不得进入总分；漂移超过硬上限，或任何其他端点不一致，整张卡直接失败。

患者和 held-out seizure 是统计分母，五分钟 anchor 不是独立样本。任何一个预先登记的发作端点都可以形成跨任务候选证据，但同一 family 必须同时胜过其登记基线、FIT 常数和错时状态；双流还必须胜过随机容量对照，且该 lead time 至少有三场独立 held-out seizure。其他端点及不可估分母必须同时展示。发作结果不得反向选择 observer family、患者、tau、state width 或 checkpoint。

## 队列与可估性分层

### 长状态队列：共享 0.5/2/6/8 h heads

`E1096, E253, E958, E1077, E1125, E916`。

这些患者在不读取发作结局的前提下，满足共享 8 小时状态生产器所需的 FIT、INNER 和 SELECTION 有效观测时长。H2a 中 E916 缺少当前严格 rank/contact 评价产物，因此只进入 H1/H2b，不把缺失写成阴性。

### 中尺度扩展队列：共享 0.5/2 h heads

`E1146, E384, E548, E583, E922`。

它们不满足 8 小时共同切分，但可用于较短动态状态、H2a 和冻结 H2b。长、短两层不混成一个总体效应；每个 horizon 另报互不重叠的有效窗口数。

Yuquan 当前 24 小时记录在要求四倍目标窗 FIT、两倍 INNER、三倍 SELECTION 的共享 2 小时合同下仍不足，本轮记为结构性不可估，不启动无效 GPU 单元。

## 训练与执行锁

- 优化配方固定来自 v0.3.7 FIT/INNER 搜索；v0.3.8 新患者不参与回头选配方。
- 每个状态生产器跨其登记的所有 horizon 共享；只有 readout head 按 horizon 独立。
- `B_rate/B_mark` 与模型臂接受相同的可训练性闸门。
- H2a decoder 主体冻结，只训练低秩状态调制接口；正确状态、常数、错时和泄漏正对照同协议评分。
- H2b 先全队列 freeze，后 outcome；不允许边冻结边看结局。
- 所有失败区分为 `NOT_ESTIMABLE`、`ASSAY_LIMITED`、`OPTIMIZATION_FAILED`、`SCIENTIFIC_NULL`，不得互换。

## 完成定义

本轮不是以“找到显著结果”为工程完成条件。工程完成要求：

1. 两层 H1、两套严格 decoder、H2a、H2b 和训练后长时 credit 审计全部结束或给出可验证的不可估原因；
2. 每个患者保留五个 seed，患者为统计单位；
3. 四个核心问题各有自己的机器字段、对照和分母；
4. 白话报告与技术报告同时给出支持、反证和边界；
5. 只有 instrument checks 全过且训练充分后，阴性结果才具有科学解释力。

当前自动监督器：`scripts/supervise_group_event_state_v038_core_closure.py`。它负责在长队列完成后自动启动中尺度 H1，并在对应状态与 decoder 冻结后启动 H2a/H2b，避免 GPU 阶段之间空转。
