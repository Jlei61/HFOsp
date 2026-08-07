# Topic 5：Autonomous Latent-Trajectory Null v0.1 实施与收口计划

## 1. 目标

检验完整重复事件是否选择一个由第一 rank 初始化、输出不反馈的确定性自主
latent trajectory，
而不是静态 participation、局部转移、少量离散路径或低维时间模板。

开发结果必须先过公平信息集、训练充分性和完整事件自由生成三道门，才允许扩到
全人类队列或解释内部结构。

## 2. 已完成

### Phase A：数据与泄漏合同

- masked rank-event loader 与输入 SHA 校验
- 旧 train80 内 70/15/15 chronological train/monitor/development-test
- 旧 heldout20 完全排除
- target-blind 六患者 pilot
- A/B、axis、SOZ、geometry、ictal、IEI 均未读取

### Phase B：公平模型阶梯

- M0、M1、M1-phase、M2、M2-phase、M3、M4、M4-phase
- exact conditional `k`-subset likelihood
- static scaffold 在评分 likelihood 下极大似然拟合
- mixture 非同值初始化
- phase-matched Markov 和 field 对照
- NLL / decision、重复 IWAE、重复 prior-predictive、逐 step 诊断
- 4 次自由 rollout 与 repertoire 均值/SD

### Phase C：训练与 provenance

- 每模型稳定独立 seed
- validation early stopping 与 material-improvement threshold
- 过早最佳点自动从同一初始化降低学习率复核
- 最佳模型和 optimizer checkpoint
- config/source/input/split/checkpoint SHA
- launcher 等待并检查全部子进程
- aggregator 拒绝缺失 run、泄漏、混合 config 或混合 source

### Phase D：SNN 只读资产审计（历史探索性检查，不是 Gate）

- 未运行 simulator
- 审计既有 E1146 source-only、sink-only、paired source/sink families
- 生成三个统一 rank-event NPZ 及逐文件 SHA inventory
- 明确方向由低阈值 kernel/core 位置产生
- isotropic 仅作通道形状诊断，不作方向消失 null
- 该批旧文件跨条件不可池化且没有同条件 nested `N_min`；后续 Round 5 不能
  判 G0，已从 RNN Gate 删除

## 3. 已完成的开发运行

Canonical root：

`results/topic5_shared_propagation_field/development/ladder_pilot_v0_4/`

规模：

- 6 patients
- 3 seeds
- 8 models
- 18/18 shards 完成
- 144/144 fits 为 `CONVERGED` 或 `NO_FREE_PARAMETERS`
- 5 次自动低学习率复核，复核后全部合格

正式汇总：

- `ladder_cohort_summary.json`
- `ladder_per_patient.csv`
- `ladder_runs.csv`
- `LADDER_PILOT_STATE.json`

每个 run 另有 `summary.json`、`checkpoint.pt`、`conditioned_generation.npz`
和 `run_state.json`。

## 4. 开发判决

以患者内 seed-folded NLL / suffix decision 为准：

- M4 优于 M0：6/6
- M4 优于 stationary M1：5/6
- M4 优于 phase-matched M1：2/6
- M4 优于 stationary M2：3/6
- M4 优于 phase-matched M2：0/6
- M4 优于 M3：0/6
- M4-phase 优于 M3：2/6

pure prior-predictive sensitivity 给出同方向结果；MC SD 远小于主要差值。
自由生成 precedence 也没有给 M4 提供一致的第二 endpoint 优势。

因此：

> 进度时钟确实是旧比较中的混淆，但把相同进度信息加入 Markov/field 后，
> 输出不反馈的 autonomous latent-trajectory null 仍无稳定优势。当前数据支持的是完整事件具有
> 静态模型和普通 stationary first-order 模型之外的组织；它不能进一步把
> 该组织归因于由初态预先决定的低维自主轨迹。

M4-phase vs M3 为 2/6、患者中位差约 `+0.010 NLL/decision`，正式记为
`NON_SELECTION_TIE`；平局未通过“必须赢”的进入门，但不是统计方向性失败。

这是 development bounded negative，不是对稳定 contact interaction、所有
潜在动力系统或患者/SNN 网络结构的普遍否定。

## 5. SNN 只读兼容性检查已完成

不重跑 SNN。现有 artifact 已足以作为以后只读 calibration 的输入：

`results/topic5_shared_propagation_field/snn_positive_control/existing_artifact_reuse/`

按后续审阅要求，已在不运行 simulator 的前提下完成一项冻结 SNN-only
方法学验证：

1. 在 paired family 上训练；
2. 只通过第一 rank set 做 source-conditioned rollout；
3. 对 source-only / sink-only family 做 held-out 方向与 repertoire 预测；
4. 明确这不是一般 contact lesion operator。

该分析只复用现有 NPZ，没有生成新 SNN 事件。paired held-out
NLL/decision 中 M4 为 `0.528`，M1-phase 为 `0.113`、M3 为 `0.244`；
而方向 Brier 上 M4/M4-phase 最好。由于 pooled artifact 不满足同条件
`N_min` 合同，且 first-rank lookup 已获得 `100%/100%/78.4%` 的
source-only/sink-only/paired 方向准确率，这两个排序都不能判结构恢复。

正式状态：

`EXPLORATORY_COMPATIBILITY_CHECK_ONLY / G0_NOT_EVALUABLE`

## 6. 不再执行

- 不扩到 34 人正式多 seed cohort
- 不运行正式 G2/G3、Markov-surrogate cohort 或 mixture-of-fields
- 不用更多 architecture sweep 挽救 M4
- 不让 outer heldout20 进入拟合、选择或评分
- 不连接 early-ictal target
- 不重跑 SNN full/isotropic/axis/core/lesion 条件
- 不再把 SNN 作为任何 RNN Gate

## 7. Post-stop 判别诊断已完成

按用户要求补做了七轮相互区分的诊断：

1. likelihood sample/ESS 校准；
2. 患者内 event length/progress 分解；
3. `{10,20,40,60,80,100}%` target-blind nested learning curve；
4. observable stability 与 M0-subtracted dynamic residual；
5. 既有 SNN rank events 上的只读辨识；
6. `d={2,4,6}` sensitivity；
7. field latent/trajectory 非塌缩诊断。

learning curve 中 M4 相对 M3 在六档均为 `0/6`；autonomous M4 在三个
latent dimensions 也均为 `0/6`。这些人类诊断复用同一个 development test，
用于排除替代解释，不是独立统计复制，也不能推翻 full-data G1。

## 8. 最终交付

- 科学合同：
  `docs/superpowers/specs/2026-07-30-topic5-shared-propagation-field-rnn-v0_1.md`
- 结果归档：
  `docs/archive/topic5/shared_propagation_field_rnn_ladder_pilot_2026-07-30.md`
- 当前实现与测试入口见 spec §8
- 多轮最终审阅：
  `docs/archive/topic5/shared_propagation_field_rnn_multiround_review_2026-07-31.md`
- 机器判决：
  `results/topic5_shared_propagation_field/development/multiround_review_2026-07-31/MULTIROUND_VERDICT.json`

v0.1 当前最合适的论文地位是一个窄的 bounded null / Extended Data 或方法学
审计。RNN 与 SNN 独立开发，v0.1 不参与二者机制映射。新的 Stable
Interaction Graph RNN 另立合同。
