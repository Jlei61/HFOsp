# Topic 5 Stable Interaction Identifiability v2.1 执行计划

## 1. 原则

先复用现有模型做结构判别，不开发更大的 SIG。seen-distribution predictive
ranking 只作描述；D0/D3/D4 才决定能否解释稳定 structure。

## 2. 执行顺序

1. **Gate/wording repair**：修订 v2 spec、plan、archive、index 和机器状态；旧
   oracle envelope 标为 stress test。
2. **D1 existing-artifact audit**：逐模型效应量、validation-selected baseline、
   within-start diversity、future-schedule shortcut。
3. **D2 M2 audit**：observable operator seed stability、component matching、共享
   backbone、occupancy 与 start/schedule predictability。
4. **D0 patient-matched synthetic design**：先冻结 sensitivity/specificity 数据
   生成器与阈值，再运行；不得用结果调门。
5. **D4 unseen-start**：当前 SIG1/M1/M2/M3 从头重训，不增加容量。
6. **D3 split stability**：early/late SIG1 与 phase/template matched null；比较
   supported-state `I_eff`。
7. **Decision review**：只有 D0 通过且 D3/D4 有信号，才决定是否新建
   modulated-backbone 模型。

## 3. 本轮停止条件

- D1 显示 route 主要由完整未来 schedule 决定：先重定义 conditioning，不训练
  新结构模型；
- D0 specificity 失败：停止 human structure interpretation；
- D4 和 D3 均无 real-over-null：停止 single graph；
- 任一阶段不得读取 SNN、A/B、SOZ、ictal、geometry 或 outer heldout20；
- 不以 test oracle、4/6 启发式或 raw weight similarity 宣布结构成立/失败。

## 4. 第一批交付

- `SIG_V2_1_IDENTIFIABILITY_STATE.json`；
- D1 predictive/envelope/diversity audit；
- D2 M2 operator audit；
- D0/D3/D4 的冻结可执行合同；
- 明确下一步是运行现有模型的结构实验，还是停止，不先实现新模型。

## 5. 每轮结束时的强制反思

每个任务结束后，在启动下一个任务前回答四个问题并写入 aggregate：

1. 本轮 endpoint 能排除哪个替代解释，不能排除哪个？
2. comparator 是否由 test 事后选择，或是否使用了未来 event envelope？
3. 该结果是 prediction、engineering identifiability，还是 human structural signal？
4. 下一步是否增加模型容量？若是，必须指出已经通过的 D0 与 D3/D4 证据；否则
   fail closed，不实现。

## 6. 执行状态

1. Gate/wording repair：完成；
2. D1 existing-artifact audit：完成；
3. D2 M2 seed/operator audit：完成；
4. D0 patient-matched sensitivity+specificity：完成，4/6 PASS；
5. D4 unseen-start：完成，NLL-only mixed signal；
6. D3 split stability：完成，0/6 real-over-null；另以未参与 checkpoint 选择的
   inner-test probe 重评分，仍为 0/6；
7. Decision review：完成，single fixed graph bounded negative；D5 与 full cohort
   不开放。
