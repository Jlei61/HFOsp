# Topic 5 Event-indexed Evolving Rank Field v2.2 科学合同

> **Closure note（2026-08-01）**：Phase 0 已按修复后的真实 block-adjacency 与
> full+middle Gate 完成。6 位 pilot 中 5 位 block reliable、5 位 G0 PASS、2 位获得
> Phase 1 eligibility。Phase 1 的 matched event-history increment 在 2/2 均未通过，
> 因此停止 event-driven ELR / RNN 实现。该停止不否定稳定 backbone 或一般 block-wise
> variation，只说明当前 chronology-sensitive event increment 未超过 autonomous/state controls。

> **Supersession clarification（2026-08-01）**：该分析的观测量是未按稳定模板分层的
> block-mean rank/participation descriptor。固定模板的 occupancy 改变本身即可改变该均值；
> 因此 Phase 1 阴性不能否定已经由 split-half / odd-even block 证明的稳定传播模板，
> 也不再作为 stable-repertoire event RNN v2.3 的进入门。v2.2 只保留为 blockwise
> marginal descriptor 的历史审计。

## 0. 谱系与边界

v2.1 已正式验收并冻结。v2.2 改变的是科学对象，不是为阴性模型增加容量：

1. 一整场间期传播事件是一个 token；
2. event index 是长期状态演化的主时间轴；
3. rank step 只描述单事件内部形状；
4. IEI 仅作 secondary decay / nuisance，不定义主时间轴；
5. G0 未证明动态信号可观测前，禁止实现 ELR-RNN。

## 1. 冻结科学问题

> 患者特异传播场是否在稳定 backbone 周围呈现超过测量噪声的低维时间变化；若存在，
> 过去间期事件的完整传播内容是否能够预测随后传播场的变化，并超过 fixed network、
> autonomous drift、discrete switching 与 chronology-destroying null？

观察性人类数据最多支持 `associated with subsequent network-state change` 或
`supports activity-dependent shaping`，不能直接写因果 plasticity。

## 2. 三种时间必须分开

| 时间 | 用途 | 禁止误读 |
| --- | --- | --- |
| within-event rank/latency | 单事件传播形状与置信度 | 不是长期 recurrent time |
| event index | primary state time | 不等同真实小时/天 |
| IEI / absolute time | secondary decay、gap、day/rate control | 不单独证明 shaping |

## 3. 输入合同

Primary 输入只来自冻结 `dataset_v0_4`：

- `event_local_rank`：参与 contact 的 `[0,1]` masked local rank；
- `event_group_ids`：exact-centroid ties 的 set-valued rank；
- `event_participation`；
- `event_lag_raw`：within-event spectrogram centroid time，不冒充 peak time；
- `event_abs_time`：`packedTimes start + lagPat start_t`；
- `event_source_index`：回连原始 pooled event；
- `event_split`：旧 chronological train80 / heldout20；
- source record/block ID：由 `event_source_index` 回连 canonical loader。

必须逐患者验证时间、source index、block、mask、rank、lag 和 contact ordering 一致。
精确 contact peak time 当前不可用；只能用 centroid gap 定义 near-tie sensitivity。

主开发只使用旧 train80。train80 内再按 chronology 划分：前 75% 用于 block-size /
reliability calibration，后 25% 用于 development G0 confirmation。旧 heldout20 不进入
拟合、选择或评分。六患者 pilot 冻结为 v2.1 同一名单；全 34 人只做字段 inventory，
不做科学 Gate。

禁止输入 A/B、pathological axis、SOZ、ictal target、SNN、geometry 或 patient outcome。

## 4. 事件块合同

Primary block 是 source-record 内连续、互不重叠的等事件数窗口。不得跨 source record、
已知 gap 或 train/calibration 边界。候选大小冻结为 `N={20,40,80}`。

block size 只能由 calibration reliability 选择，不能按 G0 effect 选择。选择最小满足：

- calibration blocks ≥30；
- confirmation blocks ≥16；
- confirmation within-record adjacent pairs ≥10；
- median rank-field split-half Spearman ≥0.60；
- median participation split-half Spearman ≥0.70。

若无候选通过，该患者为 `BLOCK_FIELD_UNRELIABLE`，不训练 evolving model。固定真实
时间窗口只作后续 sensitivity。

为控制纯 bootstrap 计算量，每个候选按 source record 分层、沿 chronology 等距抽取
最多 128 个 calibration blocks 估计 reliability；G0 同样等距抽取最多 128 个
confirmation blocks。该上限在任何 human G0 结果写出前冻结，不按 effect 选 block，
不改变每个 block 的完整事件组成。若 G0 通过，preliminary low-rank audit 使用同一规则
另取最多 128 个 calibration blocks，并沿用 G0 的最多 128 个 confirmation blocks；不得
在 G0 后切回未封顶数据或按 low-rank effect 重选 blocks。

G0 的 confirmation 封顶抽样必须保留分散于 chronology 的真实相邻 block pairs；
`adjacent` 始终定义为同一 source record 中原始 `within_source_order` 相差 1，不能把
“抽样后相邻”冒充“原始相邻”。至少 10 个真实相邻 pair 的要求必须在封顶抽样后再次执行。

## 5. Block-wise 可观测对象

对事件 `e`、contact `c`：

`q[e,c] = normalized local rank`，非参与由 mask 表示。

block rank field 使用 train-only global mean 作弱收缩：

`mu[b,c] = (sum q + alpha * mu0[c]) / (n[b,c] + alpha)`。

同时估计：

- participation probability；
- source / sink probability；
- pairwise precedence probability，ties 贡献 0.5；
- 每个量的 split-half / bootstrap reliability。

Primary descriptor 为 rank field + participation；pairwise precedence 为 matched secondary。
所有动态结果另做 middle-contact sensitivity，防止只由稳定 source/sink anchors 驱动。

## 6. G0：动态信号可观测

在 confirmation blocks 上，以相同半样本大小比较：

- within-block split-half distance；
- within-source-record between-block distance；
- descriptor distance 与 block lag 的关系。

Null 在每个 source record 内打乱完整事件顺序后重新分 block，保持事件数、rank field、
participation、record identity 和 block size。阈值在 human G0 前冻结：

- primary between/within distance ratio ≥1.10；
- one-sided empirical permutation `p <= 0.05`；
- precedence secondary 与 primary 同方向；
- 至少 16 confirmation blocks、10 个 adjacent pairs。

仅 absolute block difference、PCA training variance 或显著的 event-rate drift 均不算 G0。
G0 只说明可观测变化，不能归因于事件塑造。

## 7. Preliminary low-rank eligibility

仅对 G0 通过患者，在 calibration blocks 拟合 `K=0..4` PCA/factor subspace，并在
chronologically later confirmation blocks 重建。候选 `K` 不能用 confirmation 选择。

开放 ELR 的必要条件：

- 某个 `K<=4` 的 held-out reconstruction gain 相对 `K=0` ≥0.10；
- gain 高于 95% event-order-shuffled null；
- basis 的 split stability 高于对应 null；
- middle-contact sensitivity 不完全消失。

若有动态但无低维性，结论为 `EVOLVING_BUT_NOT_LOW_RANK`，不实现 ELR-RNN。

## 8. G1–G7（只有 G0 后开放）

| Gate | 必须比较 | 允许结论 |
| --- | --- | --- |
| G1 evolving | fixed、recent persistence、static participation | graph/field 在变化 |
| G2 event-driven | autonomous drift vs `A s + B u` | event content 有增量 |
| G3 chronology | shuffle、block permutation、circular shift、time/IEI-only | 增量依赖真实顺序 |
| G4 low rank | K、subspace stability、matched null、middle-only | 变化近似低维 |
| G5 switching | HMM/template switching vs continuous state | 连续演化或离散切换 |
| G6 IEI | event-indexed vs IEI decay | 状态保持受间隔调节 |
| G7 replication | 冻结维度、block size、history、threshold | 独立确认 |

G2/G3 前不得使用 shaping；G7 前不得写 confirmatory mechanism claim。

## 9. Phase 0 决策矩阵

| 结果 | 决策 |
| --- | --- |
| 时间/source/block 合同失败 | 修数据，不做动态分析 |
| block reliability 失败 | 只保留稳定 template |
| G0 不超过 noise/null | 不实现 evolving model |
| G0 通过但 low-rank 失败 | 记录一般动态，不实现 ELR |
| G0 + low-rank eligibility 通过 | 新立最小 fixed/drift/HMM/event-driven 模型合同 |

任何结果都不自动启动 GRU、architecture zoo、general C×C graph 或 SNN comparison。
