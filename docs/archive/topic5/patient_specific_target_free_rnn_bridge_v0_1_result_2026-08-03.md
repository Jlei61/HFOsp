# 患者特异 target-free RNN 跨状态桥梁 v0.1（2026-08-03）

## 1. 科学问题

本轮不再用跨患者 readout，也不要求每名患者共享同一机制。每名患者只用自己的 masked
interictal contact-rank events 自监督训练，检验两个相互独立的问题：

1. RNN 能否在该患者 chronological heldout events 上恢复事件内 contact 传播顺序；
2. checkpoint 和自由 rollout 完全冻结后，模型汇总出的 patient-specific contact field 是否与
   同一患者 clinical-onset 后 0--10 s early-ictal energy field 对应。

训练阶段不使用其他患者事件、经验 A/B、SOZ、发作 source 或任何 ictal target。

## 2. 数据与训练合同

- strict clinical-onset cohort：16 人、33 次发作；`epilepsiae_1146` 为 development/supportive，
  其余 15 人、31 次发作为 primary。
- 每患者独立 chronological `fit60 / validation20 / test20`，`test20` 从未参与优化。
- 模型：`full_history_gru`、`linear_state`、`rank_shuffle_gru`；每模型 3 seeds。
- 冻结训练：hidden 32、7 次完整 fit60 coverage、每次 32 updates、batch 256、AdamW、
  lr `3e-4`、weight decay 0、gradient clipping 1。
- 每 unit 从模型自身联合分布自由生成 5000 个完整事件，导出 participation、early/late rank mass、
  endpoint mass 和 weighted earliness。
- 跨状态 primary：clinical-onset `[0,10] s`、`1--150 Hz`；`1--45 Hz` sensitivity。
- 评分：候选模型场的 sign-free max absolute Spearman；每个 null draw 内重新做候选场最大化。
  all-contact permutation 为论文既有 primary null；within-shaft permutation 为更严格几何 sensitivity。
- 静态对照使用 fit60 的完整 participation + conditional rank distribution，不是弱 participation-only 对照。

## 3. 间期结构学习结果

144/144 units 完成，0 failure、0 OOM、0 NaN。

- 全 16 人真实顺序 GRU 相对 rank-shuffle GRU 的 heldout NLL 改善中位数为
  `0.0603 nats/event`，15/16 同向。
- 15 人 primary 为 14/15 同向，精确配对 `P=0.0001221`。
- 自由 rollout 与真实 test20 的 pairwise contact precedence Spearman 中位数：
  GRU `0.775`、linear state `0.800`、rank-shuffle GRU `0.052`。
- 这支持 RNN/线性递归状态恢复患者自己的事件内传播结构；它不要求 AB 标签，也不是跨患者迁移。

## 4. 与发作早期场的联系

Primary 15 人、1--150 Hz：

| field | median max absolute rho | median all-contact margin | margin > 0 | exact P vs 0 |
|---|---:|---:|---:|---:|
| patient-only GRU | 0.584 | 0.167 | 13/15 | 0.0256 |
| linear state | 0.571 | 0.158 | 13/15 | 0.0125 |
| rank-shuffle GRU | 0.442 | 0.155 | 13/15 | 0.00537 |
| static fit60 participation + rank | 0.607 | 0.167 | 12/15 | 0.0157 |
| empirical test20 reference | 0.575 | 0.173 | 14/15 | 0.00836 |

GRU 相对完整 static fit60 的 all-contact margin 增量中位数 `+0.025`，9 正/4 负/2 并列，
`P=0.305`；相对 rank-shuffle GRU 为 `-0.008`，`P=0.572`。因此跨状态对应不是有序 GRU
独有的增量，而主要反映多个 target-free interictal estimators 都恢复出的患者空间 scaffold。

within-shaft sensitivity 下，GRU margin 中位 `+0.071`，10 正/4 负/1 并列，`P=0.149`；
不能声称完全排除了 electrode-shaft geometry。`1--45 Hz` sensitivity 与 primary 方向一致：
GRU all-contact margin `+0.214`，13/15，`P=0.0215`，但同样没有 GRU-specific 增量。

## 5. 最终科学判读

本轮支持：

> 仅用患者自身间期 contact-rank sequences 自监督训练的 recurrent model，能够恢复该患者
> heldout 间期事件的 recruitment/rank structure；模型恢复出的患者空间场与同一患者发作早期
> broadband energy field 在论文既有 all-contact null 下存在跨状态对应。

本轮不支持：

- 有序 RNN dynamics 比完整静态 interictal scaffold 额外解释 early-ictal field；
- 自动恢复唯一物理 A/B 轴；
- 逐次发作动态 replay 或发作路径预测；
- 每名患者共享同一个 RNN 机制。

## 6. 产物

- 汇总：`results/topic5_patient_specific_target_free_rnn_bridge_v0_1/PATIENT_SPECIFIC_RNN_BRIDGE_SUMMARY.json`
- 白话报告：`results/topic5_patient_specific_target_free_rnn_bridge_v0_1/PATIENT_SPECIFIC_RNN_BRIDGE_REPORT.md`
- patient tables：`interictal_patient_metrics.csv`、`early_ictal_patient_metrics.csv`
- 六联图：`figures/patient_specific_target_free_rnn_bridge_six_panel.{png,pdf}`
- 复现清单：`REPRODUCIBILITY_MANIFEST.json`
- 原始逐 unit checkpoint、heldout prediction、free rollout、training log 与 `DONE.json` 保留在本地
  result tree；launcher state 为 `COMPLETE`。

