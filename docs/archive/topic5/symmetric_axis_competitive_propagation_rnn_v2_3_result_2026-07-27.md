# Symmetric-axis competitive propagation RNN v2.3 正式结果

> 日期：2026-07-27
> 状态：正式纯间期合同已执行至预注册停止点；不再调参，不开放发作期 target。
> 主图：
> `results/topic5_symmetric_axis_competitive_propagation_v2_3/figures/competitive_propagation_rnn_formal.png`

## 总判断

v2.3 修复了 v2.2 最主要的 observation-likelihood 问题：把独立 Bernoulli
next-set likelihood 改成条件于事件继续的 categorical next-contact likelihood 后，
模型稳定优于固定 node bias，22/22 患者均为正。历史状态相对无历史模型也有稳定
增益。

但这不是预设机制的完整阳性结果。第二个 delayed-competition state 没有增益；
physical-axis scaffold 相对 matched local-isotropic model 的增益置信区间跨零；
source-conditioned direction 同样失败。因此当前安全结论是：

> 人类间期 contact-rank sequence 包含稳定、可由低容量 recurrent history 利用的
> 下一触点信息；但当前数据不支持把这部分信息进一步归因于预设的 delayed
> competition、physical-axis anisotropy 或 source-conditioned reversal。

这条线可以作为自监督时序结构的计算补充，但不能承担“RNN 从人体间期数据恢复了
与 SNN 同类的共享病理轴，并迁移到发作早期”的 Figure 6 主机制结论。

## 1. 执行了什么

### 1.1 输入与 denominator

- dataset v0.4：34 人、864,163 个间期事件；
- 仅 25 个事件含 non-source tied rank，primary categorical task 排除这些事件；
- development：`epilepsiae_1077`、`epilepsiae_1146`、
  `yuquan_chengshuai`；
- formal physical-axis cohort：22 位 geometry-complete、
  development-excluded 患者；
- chronological train60 / validation20 / heldout20；
- axis 与 node bias 只用 train80，heldout20 不参与模型、epoch 或阈值选择；
- A/B、SOZ、IEI、seizure label 和 early-ictal target 均未输入。

输入审计：
`results/topic5_symmetric_axis_competitive_propagation_v2_3/input_audit/INPUT_AUDIT_STATUS.json`

### 1.2 Development freeze

三位 development patients 完成：

- 3 个 persistence pairs；
- 2 个 learning rates；
- 2 个 seeds；
- 共 36/36 runs，全部 finite、0 OOM、0 失败。

只根据三患者 validation20 的 patient-first categorical NLL 冻结：

- \(\rho_P=0.50\)；
- \(\rho_C=0.75\)；
- learning rate = 0.01；
- batch = 2048；
- AdamW，weight decay \(10^{-4}\)；
- maximum epochs = 200，patience = 20。

heldout20 未用于 development selection，也未写入 grid metrics。

### 1.3 Formal training

22 人 × 3 seeds × 5 trainable conditions：

1. `local_isotropic_two_state`
2. `axis_one_state_no_competition`
3. `axis_two_state_no_source`
4. `axis_instantaneous_no_history`
5. `axis_two_state_source_full`

共 66/66 patient-seed tasks、330/330 model fits 完成，0 失败。另以完全相同的
heldout categorical contract计算：

- `node_bias_categorical`
- `empirical_last_rank_markov`
- `empirical_ordered_history_markov`

正式 launcher wall time 为 1,194 s；全部 task 累计约 3.0 worker-hours。最大单
process RSS 约 1.62 GB，最大 CUDA allocation 约 0.08 GB，无 OOM。

## 2. 正式结果

正向 benefit 定义为：

\[
\mathrm{NLL}_{baseline}-\mathrm{NLL}_{model}.
\]

| Claim | 患者级比较 | median benefit [95% bootstrap CI] | 正向患者 | BH-FDR q | 结论 |
|---|---|---:|---:|---:|---|
| A predictive adequacy | node bias − full | 0.0781 [0.0427, 0.1132] | 22/22 | \(1.43\times10^{-6}\) | PASS |
| B1 ordered history | no-history − full | 0.0199 [0.00553, 0.0324] | 18/22 | \(1.81\times10^{-4}\) | PASS |
| B2 delayed competition | one-state − full | −0.000777 [−0.000943, −0.0000578] | 5/22 | 0.982 | FAIL |
| C axis bundle | local-isotropic − full | 0.000820 [−0.0000878, 0.00225] | 15/22 | 0.0269 | FAIL |
| C matched axis | local-isotropic − axis/no-source | 0.000385 [−0.0000685, 0.00103] | 14/22 | 0.0269 | FAIL |
| D source direction | axis/no-source − full | 0.0000306 [−0.0000660, 0.000264] | 13/22 | 0.255 | FAIL |

Claim C 的 Wilcoxon/FDR 项虽为 nominal positive，但预注册门同时要求 bootstrap
median CI 下界大于 0；两项 CI 均跨零，因此必须判为 FAIL，不能只选显著的
rank-based test 报告阳性。

## 3. Markov 对照说明了什么

ordered-history Markov 相对 node bias 的患者级 median benefit 为 0.1349；full
structured RNN 为 0.0781，即按 cohort median effect 约恢复 58% 的经验转移信号。
full 仅 3/22 优于 ordered-history Markov，二者 NLL 差的中位数为 −0.0489
（负值表示 Markov 更好）。

这不是要求可解释模型刷过 Markov，而是说明：

1. v2.3 的 categorical recurrence 已抓住一部分真实历史信息；
2. 剩余经验转移结构没有被预设的 symmetric-axis + competition + source
   参数化解释；
3. 不能因 Claim A 阳性就把预测性能升级成 physical scaffold recovery。

三 seed 的 full-model patient-level NLL SD 中位数为
\(1.60\times10^{-5}\)，最大为 \(4.26\times10^{-4}\)。这证明优化重复性高，
不证明模型结构可辨识。

## 4. 对核心科学目标的判断

### 支持的部分

- 间期群体事件的 contact-rank sequence 不是 node participation frequency 的简单
  重复；
- ordered history 在 heldout events 中有稳定增益；
- 以 raw contact rank 自监督训练的低容量 recurrent model 可预测下一触点；
- categorical competition 从模型层面消除了 v2.2 的 set-size overprediction。

### 不支持的部分

- 第二个 delayed-competition state 是必要动力学；
- physical-axis anisotropy 是预测增益来源；
- observed source 能沿共同轴产生可辨识的方向反转；
- 人体 rank sequence 已恢复 SNN 中的共享病理 scaffold；
- 当前模型可进入 early-ictal energy-field transfer。

因此不运行 latent-state mechanism analysis、不做 source-side reversal、不读取 A/B
read-back，也不创建 ictal dataloader。

## 5. 图中六块的科学含义

- **A**：唯一允许的模型合同——共同对称 scaffold、observed source、传播与
  competition states；
- **B**：full model 是否在 heldout events 上超过 node bias；
- **C**：历史 persistence 与第二 competition state 分别是否必要；
- **D**：axis bundle、matched axis 和 source term 是否有增量；
- **E**：可解释 model 恢复了多少 ordered-history Markov transition signal；
- **F**：预注册解释门；predictive 与 history 通过，但 competition、axis、
  source 失败。

## 6. 停止决定

按冻结合同：

- Claim A：PASS；
- Claim B：FAIL（B1 pass、B2 fail）；
- Claim C：FAIL；
- Claim D：FAIL；
- latent-state analysis：`LOCKED_NOT_RUN`；
- physical-axis interpretation：`NOT_ALLOWED`；
- source reversal：`LOCKED_NOT_RUN`；
- early-ictal transfer：
  `BLOCKED_INTERICTAL_GATES_AND_MISSING_EXACT_SOURCE_METADATA`；
- target values read：`false`。

不建议继续增加 seeds、调整 \(\rho\)、改变 loss 权重或把 competition state 改名后
重跑。若论文保留这一部分，最稳妥的位置是 supplementary computational result：
强调 rank sequence 的自监督可预测性及历史依赖，同时把 physical-axis
system-identification 明确写成阴性边界。

## 7. 主要产物

- 正式 gate：
  `results/topic5_symmetric_axis_competitive_propagation_v2_3/formal/FORMAL_GATE_STATUS.json`
- patient-level metrics：
  `results/topic5_symmetric_axis_competitive_propagation_v2_3/formal/patient_model_metrics.csv`
- Claims：
  `results/topic5_symmetric_axis_competitive_propagation_v2_3/formal/claim_comparisons.csv`
- Markov recovery：
  `results/topic5_symmetric_axis_competitive_propagation_v2_3/formal/benefit_recovery.csv`
- paper-ready figure：
  `results/topic5_symmetric_axis_competitive_propagation_v2_3/figures/competitive_propagation_rnn_formal.{png,pdf}`
- figure README：
  `results/topic5_symmetric_axis_competitive_propagation_v2_3/figures/README.md`
