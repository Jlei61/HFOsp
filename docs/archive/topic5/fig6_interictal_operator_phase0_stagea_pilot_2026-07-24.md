# Figure 6 间期传播算子：Phase 0 与 Stage-A 工程诊断

**日期**：2026-07-24
**合同**：v0.3
**当前状态**：Phase 0 通过；target-free one-SE 选择 h64；13 人 one-seed screen 在 suffix-static gate 停止；未启动正式三 seed gate

## 1. 一句话结论

masked contact-rank 的数据与泄漏合同已经闭合。target-free one-SE 在 13 个匹配
fold 上选择 h64；所选 contact-query GRU 在 13 人 one-seed screen 中对
participation-preserving rank-shuffle 的 next-set 与 suffix 指标均为 13/13
正向，并在 next-set NLL 上超过最强静态/一阶基线，但 suffix-rank concordance
没有超过最强静态基线。这个结果支持单事件前缀中存在可学习的局部下一招募信息，
但不支持进入 mode recovery、ictal readout，更不支持“发作早期路径重放”。

## 2. 冻结的科学问题

RNN 的 recurrence 只发生在单个间期群体事件的 recruitment pseudo-time：

```text
masked contact-rank prefix
    → next recruitment set / STOP
    → remaining participation
    → suffix contact utility/rank
```

Stage A 不输入 IEI、跨事件历史、seizure seed、clinical/EEG onset 或任何 ictal
energy。只有 Stage-A 正式 gate 和后续 Mode-recovery gate 都通过，才允许首次
读取已冻结的 clinical-onset `[0,10] s`、1–150 Hz 静态能量场。

完整合同：

- `docs/superpowers/specs/2026-07-24-topic5-interictal-operator-static-ictal-readout-design.md`
- `docs/archive/topic5/fig6_ictal_target_temporal_adjudication_2026-07-24.md`

## 3. Phase 0 数据合同

### 3.1 数据池

| 项目 | 结果 |
|---|---:|
| 合法患者 | 40/40 |
| Epilepsiae / Yuquan | 20 / 20 |
| 合法间期事件 | 532,793 |
| 候选发作患者 | 13 |
| frozen strict-BB150 seizures | 71 |
| 候选患者合法间期事件 | 71,847 |
| Stage-A 是否读取 ictal values | 否 |

每名患者按时间将合法间期事件拆为前 80% target-free calibration 和后 20%
held-out evaluation。共享初始化先做 dataset-balanced，再做 patient-balanced，
避免高事件数患者主导更新。

### 3.2 phantom rank 与 ties

所有 legacy `lagPatRank` 都先用 `eventsBool` 去除非参与触点的 phantom finite
rank，再只在参与触点内归一化。40 人全池中 participant count 和 recruitment
group count 的患者中位数均为 6；13 名候选患者中两者中位数均为 5。

候选患者的 tie/lag 审计：

| 指标 | 患者中位数 |
|---|---:|
| exact-tie fraction | 0 |
| adjacent lag ≤1 ms | 0.181 |
| adjacent lag ≤2 ms | 0.322 |
| adjacent lag ≤5 ms | 0.600 |
| adjacent lag median | 3.72 ms |

主数据只合并 exact ties。若使用 5 ms 容差，会改变约 60% 的相邻转移，已不再是
保守的测量分辨率处理；因此不能把 5 ms 作为未经裁决的默认生物时间窗。

### 3.3 数据集特异的 block 规则

首次 full-pool 构建发现 20 名 Yuquan 患者全部被 Epilepsiae 的 1 h
discontinuity 规则误杀。Yuquan 原始记录本身是连续 2 h blocks，正常相邻间隔
会超过 Epilepsiae 的 5,400 s 门限。

修复后：

- Epilepsiae 继续使用既有 fail-closed definite-interictal blocks；
- Yuquan 按真实 recording-block 与 seizure / 120 min post-ictal guard 的重叠
  做排除；
- 不把 day/night 或 block 间隔引入单事件内 RNN。

修复后 40/40 通过。旧失败日志保留，未覆盖：

- `results/topic5_interictal_operator_static_readout/runs/phase0_full_20260724/stdout.log`
- `results/topic5_interictal_operator_static_readout/runs/phase0_full_20260724_retry1/stdout.log`

### 3.4 sanity checks

| 检查 | 结果 |
|---|---:|
| synthetic forward/reverse label recovery | 1.0 |
| synthetic seed AMI | 1.0 |
| 两模板相关 | −1.0 |
| phantom values 是否在聚类前屏蔽 | 是 |
| rank-shuffle participation mask | exact preserved |
| rank-shuffle per-event group-size multiset | exact preserved |
| rank-shuffle contact support | exact preserved |
| original vs shuffle median Kendall τ | 0 |

`phase0_sanity_pass=true`。

## 4. Stage-A V1 与强制对照

V1 是 set-token contact-query GRU。contact 字符串 ID 不进入模型；允许的患者内
side information 只有 prefix participation support、杆内归一化位置和中心化
几何。输出联合 loss 为：

```text
L = L_next-set + 0.5 L_stop + 0.25 L_remaining + 0.5 L_suffix-rank
```

强制对照：

1. support-only；
2. first-order Markov；
3. calibration-prefix empirical K=2 templates；
4. unordered DeepSets；
5. matched feed-forward contact query；
6. participation-preserving within-event rank shuffle。

## 5. 三患者工程结果

### 5.1 短 pilot：8 shared + 5 calibration epochs

| gate quantity | patient median | bootstrap 95% CI |
|---|---:|---:|
| next NLL gain vs strongest static | +0.0156 | [−0.0041, +0.0285] |
| suffix concordance gain vs strongest static | −0.0025 | [−0.0077, +0.0008] |
| next NLL gain vs rank-shuffle | +0.0887 | [+0.0720, +0.0926] |
| suffix concordance gain vs rank-shuffle | +0.1471 | [+0.1386, +0.2349] |

共享 validation loss 在第 8 epoch 仍持续下降，因此该结果不能直接裁决架构阴性。

### 5.2 延长诊断：24 shared + 12 calibration epochs

患者为 E1077、E1084、E139；所有神经模型和 shuffle core 使用相同训练预算。

| gate quantity | patient median | bootstrap 95% CI | 正向患者 |
|---|---:|---:|---:|
| next NLL gain vs strongest static | +0.0130 | [+0.0113, +0.0231] | 3/3 |
| suffix concordance gain vs strongest static | +0.0023 | [−0.0092, +0.0031] | 2/3 |
| next NLL gain vs rank-shuffle | +0.0945 | [+0.0932, +0.0965] | 3/3 |
| suffix concordance gain vs rank-shuffle | +0.1180 | [+0.0851, +0.1762] | 3/3 |

E1077 的 suffix concordance 比 matched feed-forward 低 0.0092；E1084 和 E139
分别高于各自最强静态对照 0.0023 和 0.0031。这个幅度很小，不能用三人结果声称
RNN 已证明 event-order dynamics；但 next-set 3/3 正向和两项 shuffle 3/3
正向足以支持进入完整候选队列 screen。

三折墙钟时间约 33–34 min/折；峰值 RSS 约 1.46–1.48 GB/折，swap=0；三折同时
运行时总显存约 1.2 GB，无 OOM。

## 6. 完整候选队列工程筛查

### 6.1 h32：13 人 × 1 seed

| gate quantity | patient median | bootstrap 95% CI | 正向患者 |
|---|---:|---:|---:|
| next NLL gain vs strongest static | +0.01132 | [+0.00611, +0.01776] | 12/13 |
| suffix concordance gain vs strongest static | −0.00308 | [−0.00578, +0.00229] | 4/13 |
| next NLL gain vs rank-shuffle | +0.12871 | [+0.09446, +0.18782] | 13/13 |
| suffix concordance gain vs rank-shuffle | +0.19263 | [+0.16643, +0.24675] | 13/13 |

h32 清楚支持 next-set 局部动态，但 suffix-static gate 不支持。由于冻结合同要求
hidden size 只能由 Stage-A inner validation 选择，不能直接把 h32 当最终架构。

### 6.2 target-free hidden-size 选择

使用相同 13 个 subject-by-seed cells，只读取每折 `shared_initialization` 的最小
inner-validation loss；不读取 held-out last20 metrics，更不读取 ictal target。

| hidden size | mean inner-validation loss | SE |
|---|---:|---:|
| 32 | 2.26282 | 0.00231 |
| 64 | 2.23736 | 0.00216 |

h64 是最佳均值；其 one-SE 阈值为 2.23952，h32 不在阈值内，因此冻结 h64。

### 6.3 selected h64：13 人 × 1 seed

| gate quantity | patient median | bootstrap 95% CI | 正向患者 |
|---|---:|---:|---:|
| next NLL gain vs strongest static | +0.01022 | [+0.00179, +0.01540] | 10/13 |
| suffix concordance gain vs strongest static | −0.00164 | [−0.00447, +0.00201] | 6/13 |
| next NLL gain vs rank-shuffle | +0.13162 | [+0.09203, +0.18493] | 13/13 |
| suffix concordance gain vs rank-shuffle | +0.19665 | [+0.15320, +0.27490] | 13/13 |

h64 改善了 target-free shared loss，却没有修复 suffix-static gate；其 suffix
患者中位数仍为负，CI 跨 0。h64 的 next-set 结果继续为正，说明当前失败不是
“RNN 完全不可用”，而是更具体地指向：完整 event history 对局部下一招募有增量，
但当前 direct suffix-utility head 没有证明递归状态比强静态/无序模型更能恢复
完整剩余 rank。

这是一轮预先声明的 cheap-first engineering screen，而不是正式 13 人 × 3 seeds
gate。由于 selected architecture 已在必需的 suffix-static 条件上缺少方向支持，
没有继续消耗两个 seed 来制造形式上的正式检验；Stage B 按 stop rule 保持关闭。

## 7. 当前 gate 与下一步

### 已通过

- Phase 0 masked-rank / tie / block / fingerprint gate；
- synthetic forward/reverse 与 rank-shuffle sanity；
- 工程可运行性和资源安全；
- hidden size `{32,64}` 的 target-free one-SE 选择；
- selected h64 的 13 人 one-seed 完整 baseline screen；
- next-set 对 strongest static 和 rank-shuffle 的工程筛查方向。

### 尚未通过

- 13 人、三 seed、patient-level bootstrap 的正式 Event-dynamics gate；
- selected h64 的 suffix-rank concordance 对 strongest static gate；
- held-out interictal Mode-recovery gate；
- static ictal readout gate。

因此当前唯一安全结论是：

> 单事件内的 contact-rank prefix 含有超过 participation-preserving shuffle 的
> 可学习顺序信息；selected h64 对最强静态/一阶对照提供患者级 next-set 增益，
> 但 direct suffix-rank head 没有提供递归状态超越强静态模型的证据。

当前 stop decision：

1. 不启动另外两个 seeds 的正式 gate；
2. 不冻结 operator 做 two-mode recovery；
3. 不读取 clinical-onset `[0,10] s` BB150 target；
4. 不进入 V2 low-rank E/I core；
5. 若未来修改 suffix task，必须作为新版本重新预注册，不能用本轮 held-out
   结果回调 v0.3。

不得写：

- RNN 已预测发作；
- clinical onset 后存在 contact-order replay；
- 该模型证明了癫痫发作机制；
- Stage A 已正式通过或已完成正式阴性检验。

## 8. 产物

- 数据集：`results/topic5_interictal_operator_static_readout/dataset_v0_3/`
- 短 pilot：`results/topic5_interictal_operator_static_readout/stage_a/pilot3_seed20260724/`
- 延长诊断：`results/topic5_interictal_operator_static_readout/stage_a/convergence3_seed20260724/`
- 延长诊断图：`results/topic5_interictal_operator_static_readout/stage_a/convergence3_seed20260724/figures/stage_a_event_dynamics_gain.png`
- h32 13 人汇总：`results/topic5_interictal_operator_static_readout/stage_a/screen_h32_seed20260724/`
- hidden-size 选择：`results/topic5_interictal_operator_static_readout/stage_a/hidden_size_selection_seed20260724/`
- selected h64 13 人汇总：`results/topic5_interictal_operator_static_readout/stage_a/screen_h64_seed20260724/`
- selected h64 主图：`results/topic5_interictal_operator_static_readout/stage_a/screen_h64_seed20260724/figures/stage_a_event_dynamics_gain.png`
- h32 runs：`results/topic5_interictal_operator_static_readout/runs/stagea_screen_retry1_*_h32_seed20260724/`
- h64 selection-only runs：`results/topic5_interictal_operator_static_readout/runs/stagea_hiddenselect_*_h64_seed20260724/`
- h64 full-control runs：`results/topic5_interictal_operator_static_readout/runs/stagea_screen_h64_retry1_*_h64_seed20260724/`

13 人 screen 的首次外层会话在 shared epoch 12 收到 SIGTERM；三份不完整目录已写
`ABORTED.json`，没有 checkpoint/DONE，也不进入任何汇总。`retry1` 使用独立
tmux session 和全新 run 路径重启，避免覆盖失败证据。

h32/h64 完整 run 每折墙钟约 24–40 min，峰值 RSS 约 1.47–1.54 GB，三折并行
总显存约 1.5 GB，swap=0；没有 OOM。最终 CPU 回归为 16 passed、3 skipped，
CUDA 回归为 19 passed。
