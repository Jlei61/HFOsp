# Continuous Marked State H2b Cross-task Transfer v0.2 冻结合同

## 1. 唯一科学问题

只用连续背景与 IED timing/exact-mark 任务训练、并在读取发作标签前冻结的患者内状态，能否跨任务预测发作接近程度；重点是它是否在 recent IED history 与当前 observation 之外仍有增量。

这条线与 H1/H2a 的患者内 IED 验证及 H3/T2 分开运行。H1 是否阳性只作事前分层，不是 H2b 纳入门。任何增量都只能称 development cross-task transfer，不能称机制、因果或临床预测器。

## 2. 冻结输入与边界

- 上游固定为已经提交并推送的 R1.7B 17 人、5 seeds release。
- 17 人、85 个训练单元全部进入入口分母；75 个有 checkpoint，10 个非有限梯度单元保留为仪器失败。
- 所有实际存在且 SHA256 可复算的 checkpoint 都进入 H2b；`stable_checkpoint` 和 patient-level H1 标签只用于结果分层。
- observer、state update、generator、timing/mark decoder 全部冻结；seizure loss 只训练低容量 ridge conditional-risk probe。
- formal test 与 sealed partition 不打开；不运行 H3/T2，不修改 paper-ready figures。

## 3. 风险任务

- 主提前量：30 min。
- sensitivity：5、15、60、120 min，全部事前固定；不能用最漂亮的提前量替换主端点。
- arms：`B_history`、`B_observation`、`B_state`、`memoryless`、`matched_wrong_time`。
- 主量：held-out 30-min conditional log loss 的 `B_state - B_observation`；负值才有利。
- 两条必要解释量：`persistent - memoryless` 与 `correct-time - matched wrong-time`。
- 同一患者、lead、split 的全部 arms 使用相同 case/control risk sets。
- controls 和连续窗口不是额外患者；先在患者内合并 seeds，再以患者为 cohort 单位。

## 4. 发作支持层

| 30-min 合格发作数 | 运行与命名 |
|---:|---|
| `>=10` | primary chronological development |
| `5–9` | leave-one-seizure-out sensitivity |
| `2–4` | descriptive case series |
| `<2` | not estimable |

普通阴性和某一患者不可估计都不阻断其他患者。任何 checkpoint 缺失、原始挂载缺失、coverage/design 缺失都必须单列为仪器或运行环境分母，不能并入科学阴性。

## 5. 时间与数据合同

- 发作 onset 为 `s`、lead 为 `h` 时，state 只使用 `t <= s-h` 的数据。
- 绝对时间为 float64；状态在每个 recorded coverage segment 起点重置，不能跨 gap carry。
- case/control 必须来自同一患者的有效 recorded coverage；horizon 内不得有发作，ictal 与 120 min postictal 排除。
- 当前 observation 必须可从冻结 raw cache 精确读取且足够新鲜；训练期的 seizure guard 不得作为冻结推理分母。
- Epilepsiae 使用 SQL 派生冻结 inventory；Yuquan 必须通过记录码显式映射，禁止直接字符串连接。
- wrong-time donor 来自同患者、同 coverage segment，至少错开 30 min，并用 probe 显式调整 time-of-day、session position、recent IED 与 observation confounders。

## 6. 队列统计

- `>=10` 层使用患者内 chronological TRAIN/SELECT/TEST；epoch/regularization 只能由 TRAIN/SELECT 选择。
- 5–9 层用 nested leave-one-seizure-out；2–4 层只作描述。
- seeds 先取患者内中位；cohort 只按患者做方向计数和区间，不把 seeds、发作或 controls 当独立患者。
- 同时报告全 checkpoint-available 层与 H1-stable 分层；后者不能替代全分母。

## 7. Secondary phenotype

只允许使用分析前已存在、已冻结的 seizure subtype 或 early-recruitment target。缺失即记 `NOT_ESTIMABLE_FROZEN_TARGET_UNAVAILABLE`；不重新聚类、不看 state 后造阈值、不以 SOZ 或最高能量触点替代。

## 8. 承重仪器验证

1. positive synthetic 恢复负的 `state-observation`；
2. 患者内 time-label permutation 回到零附近；
3. 改变 anchor 之后的数据，anchor 之前输出逐位不变；
4. upstream result/checkpoint、coverage、design、raw-cache 与风险表逐层记录 SHA256；
5. 所有不合格状态 fail closed，但不把单患者普通阴性升级为 blocker。

## 9. 2026-08-30 启动审计

- R1.7B：17 位、85 cells、75 个 checkpoint 可复算，10 个仪器失败无 checkpoint；H1-stable 6 位。
- 冻结 seizure inventory：13/17 位有发作；其中 11 位在 30 min 仅按 coverage 初筛时至少有一次完整窗口。
- R1.7 上游 design/baseline 当前同步 10/17 位；新增 7 位的完整 design 留在已经消失的临时 worktree，需从冻结数据重建，不能猜 anchor 时间。
- 当前 `/mnt/yuquan_data` 与 `/mnt/epilepsia_data` 均未挂载，raw inference cache 为 0/17 可读。因此此刻是运行环境不可用，不是 H2b 阴性；队列数值不得在挂载恢复前生成。

本合同在查看任何 R1.7B×seizure transfer 结果前冻结。
