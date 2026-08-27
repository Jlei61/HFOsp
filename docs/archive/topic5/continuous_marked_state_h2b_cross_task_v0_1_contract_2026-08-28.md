# Continuous Marked State H2b Cross-task Transfer v0.1 冻结合同

## 1. 唯一科学问题

本工作流只问：一个只用连续背景和 IED timing/mark 任务训练、在接触 seizure label 前已经冻结的患者内状态，能否在真正未见的后段发作上，比 recent IED history 和当前 observation window 更好地判断患者是否接近发作，并进一步预测下一次发作如何展开。

这仍是 development 研究。预测增量不是机制、因果、临床预测器或 latent attractor 的证据。

## 2. 冻结边界

- 状态训练不读取 seizure label；seizure loss 不更新 observer、state update、generator 或 IED decoder。
- 发作任务只拟合低容量 ridge/conditional logistic readout。
- primary lead 固定为 30 min；5、15、60、120 min 全部运行，作为预先定义的 sensitivity。
- primary estimand 固定为 held-out 30-min conditional log loss 的 `B_state - B_observation`；负值才表示 frozen state 有增量。
- 同一患者、同一 lead、同一 split 的所有比较臂使用完全相同的 case/control risk sets。
- 发作是统计单位；controls 和连续窗口不是额外发作。seed 先在患者内取中位，患者才是 cohort 单位。
- formal test 和 sealed partition 始终关闭；不运行 H3、T2、physical clock，不修改 paper-ready figures。

## 3. 证据分层

- `checkpoint_available`：所有最终可用 checkpoint 患者，不能按 H1 结果挑选。
- `h1_stable`：R1.7 预先定义的解释层，不是 H2b 的运行 gate。
- 合格发作数不少于 10：患者内 60/20/20 chronological split，进入 primary development 层。
- 5–9：leave-one-seizure-out sensitivity。
- 2–4：descriptive case series。
- 少于 2：not estimable，但仍进入 exclusion funnel。

## 4. 术语锁

| 固定术语 | 含义 |
|---|---|
| frozen interictal state | 只由间期背景与 IED 目标训练、在发作任务前冻结的状态模型 |
| persistent state | 跨 observation anchor 传播并由当前 observation 更新的状态 |
| memoryless observation code | 每个 anchor 从固定初态独立编码，不携带上一窗口状态 |
| `B_history` | time of day、session position、recent IED timing/count/load/STOP/extent |
| `B_observation` | `B_history` 加当前 explicit spectral/variance/autocorrelation observation |
| `B_state` | `B_observation` 加 frozen persistent state |
| correct-time / wrong-time | 正确 anchor 状态与同患者、同记录段 risk-set donor 状态；混杂量在 probe 内显式调整 |
| phenotype transfer | frozen state 还能预测既有 subtype/early recruitment target；不等于机制或因果 |

## 5. 状态来源

第一阶段只使用 R1.6 `epilepsiae_384` stable seeds 1、3、4。checkpoint/result 路径和 SHA256 必须从 `continuous_marked_state_optimizer_identifiability_r1_6_machine_audit_2026-08-27.json` 追到逐 seed result，并在本轮复算。

第二阶段只读接入 R1.7。只有其 `reports/machine_audit.json` 为 `COMPLETE`、恰好 50 fits、formal/sealed 均为 false、源码和 checkpoint hash 可复算、对应提交已 commit 并 push 时才允许接入。未完成结果不得进入 H2b 支持清单或结论。

## 6. 风险集与因果性

- 发作 onset 为 `s`、lead 为 `h` 时，state 只能使用 `t <= s-h` 的数据。
- case/control 必须位于同患者有效 recorded coverage；不得跨 gap，且 observation availability 相同。
- control 对应 horizon 内不得有发作；ictal 和 120 min postictal 区间排除。
- wrong-time donor 在同患者、同 recorded coverage segment 中抽样；不再使用不可行的六维 0.5 SD 硬匹配。
- 同一 seizure 的所有 lead 位于同一 TRAIN/SELECT/TEST split；时间点不得同时进入训练和测试 risk set。
- 绝对时间保存为 float64 或 int64，不使用 float32。

## 7. 承重验证

1. positive synthetic 应恢复 `B_state` 增量；
2. 患者内 time-label permutation 后增量应回到零附近；
3. 改变 anchor 后数据时，anchor 前提取状态必须逐位不变。

上述验证只证明仪器符合合同，不证明人体 H2b 成立。

## 8. 当前执行边界（2026-08-28）

R1.7 仍在独立工作树运行且机器审计尚不存在，因此当前只运行 CPU inventory、crosswalk、纯函数 probe 开发和 E384 instrument pilot。不得与 R1.7 争抢 GPU。
