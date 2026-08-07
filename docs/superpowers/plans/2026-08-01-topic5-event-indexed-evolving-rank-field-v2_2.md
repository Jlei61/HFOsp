# Topic 5 Event-indexed Evolving Rank Field v2.2 执行计划

## 1. 原则

先证明长期状态变化在数据中可观测，再讨论模型。完整事件是 token，event index 是
primary time。Phase 0 不训练 RNN，不把 next-event prediction 当主 endpoint。

## 2. 执行顺序

1. **v2.1 acceptance**：验证机器状态及依赖哈希，冻结 bounded negative；
2. **T0 time-field audit**：逐事件绝对时间、source index、block、mask/rank/lag、IEI、
   gap/day 与 centroid-time 语义；
3. **T1 block reliability calibration**：只在 train80 前 75% 选择 N=20/40/80；
4. **T2 G0 confirmation**：train80 后 25% 比较 between-block、within-block noise 和
   record-wise event-order shuffle；
5. **T3 low-rank eligibility**：只对 G0 PASS 患者做 held-out PCA/subspace null；
6. **Decision review**：仅 G0 与 low-rank 均通过时，另立最小 ELR 模型合同。

## 3. 工程交付

- `SIG_V2_1_ACCEPTANCE.json`；
- `EVENT_INDEXED_INPUT_AUDIT.json` + per-subject table；
- block descriptor/reliability library 与 synthetic positive/negative tests；
- `EERF_V2_2_PHASE0_STATE.json`；
- G0 per-subject/aggregate JSON、CSV；
- 一份中文多轮归档，明确 observed / not supported / unadjudicated。

## 4. 每轮强制反思

1. endpoint 测的是 estimate reliability、observable variation、low-dimensionality，
   还是 event-driven shaping？
2. block size、threshold 或 comparator 是否看过 confirmation effect？
3. source-record/gap/time-of-day 是否可能解释结果？
4. 当前证据是否真的授权增加模型容量？若否，停止。

## 5. 停止条件

- 任一逐事件时间/source mapping 不一致：停止并修数据；
- 无 block size 通过 reliability：该患者停止；
- G0 失败：不训练 evolving model；
- 动态只由 source/sink 或 event rate 驱动：降级为 nuisance result；
- low-rank held-out gain 不超过 shuffled null：不实现 ELR-RNN；
- 不读取旧 heldout20、SNN、A/B、SOZ、ictal target 或 geometry。
