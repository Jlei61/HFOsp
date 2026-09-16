# Continuous marked-state R1.2b 收口与 R1.3/T2-S1 执行合同

**冻结日期：** 2026-08-25
**上游权威报告：** `results/epi_prssm/continuous_marked_state/r1/r1_2b/reports/combined_route_audit_{plain,technical}_2026-08-25.md`

## 1. R1.2b 冻结结论

R1.2b 验收为 **limited target-alignment diagnostic**，不是 raw-informed
persistent-state 验收：

- 固定三位 development 患者、两臂、三 seed，共 18/18 fits；
- 958 与黄瀚文的 observation correction 主要改善 mark，620 三 seed 均选择
  epoch 0，属于 no-update，不解释为生物学阴性；
- `filtered - no-state` 与 `filtered - validation-off` 为 2/3 患者有利；
- `correct-time - matched wrong-time` 为 0/3 患者有利；
- raw 相对 explicit 近零，但 raw tokenizer 和全部 temporal blocks 未接受目标梯度，
  因而没有 raw 阴性解释力；
- 不扩 34 人，不追加 frozen-observer null，不升级 H2b/H3 结论。

允许的文字是：

> 有限 spatial-tail 联合对齐可在部分患者中改善 observation-conditioned mark
> filtering，但尚未识别正确时刻专属的跨窗口持续状态；raw 上游未被有效训练。

## 2. 收口后处理：不重训

在原 18 个 checkpoint、原 exact validation support 上完成：

1. **persistent vs memoryless**：memoryless 在每个 30 s anchor 从生成器均值
   `mu` 重新开始，只应用当前 observation correction；
2. **strict matched wrong-time**：同 session、至少相隔 30 min，匹配 time since
   last IED、30 s/2 min/10 min count、last/recent load、previous extent、time of
   day、session position 和 observation coverage；每个 anchor 固定 5 个 donor；
3. **mark 分解**：selecting group size、terminal STOP、first-group subset、
   later-group continuation，以及 exact repeated first-prefix continuation；
4. seed 先在患者内取中位数，再跨患者描述；普通阴性不阻断 R1.3 或 T2。

## 3. R1.3 正式问题

R1.3 只回答：

> 完整 raw patch tokenizer 与全部 temporal Transformer blocks 在 exact IED
> timing + sequential mark likelihood 下学习后，是否在 explicit/history 和
> 当前窗口之外增加可持续、时刻特异的预测状态？

固定三位患者 `epilepsiae_620`、`epilepsiae_958`、
`yuquan_huanghanwen`，固定 seeds 0/1/2。正式分区保持封闭。

### 3.1 三个比较臂

1. history + explicit，memoryless；
2. history + explicit，persistent；
3. history + explicit + fully target-trained raw，persistent。

### 3.2 训练覆盖

- raw residual gate 小非零初始化，不能从精确 0 阻断 raw 梯度；
- explicit Stage A/B 用于对齐 explicit observer、state readout 与 observation
  correction；generator 保持冻结；
- paired raw 从同 seed 已完成 explicit checkpoint 出发，只训练 raw tokenizer、
  全部 temporal blocks、raw projection/gate；共同 spatial/state/readout 全部冻结，
  防止把额外训练轮数误算成 raw 增量；
- 本轮不做 generator sensitivity；
- 唯一 loss 为 exact timing + tied-group sequential mark likelihood，不加频谱、
  waveform、seizure、contrastive、KL 或 latent-consistency 辅助目标；
- TRAIN 尾部 chronological inner-validation 选 epoch；development validation
  只做最终评分；TRAIN 到 validation 因果 warm-start。

### 3.3 结论分层

- persistent 不胜 memoryless：window-level predictive filter；
- persistent 胜 memoryless、swap 阴性：persistent predictive memory；
- persistent 胜 memoryless 且 correct 胜 matched wrong-time：time-specific
  persistent state estimate；
- raw 胜 explicit：raw waveform 在显式统计之外有增量；
- 仅 STOP/size 改善：termination/extent state；
- subset 或 later continuation 改善：recruitment/repertoire state。

这些是命名层级，不是核心内容的 AND gate。

## 4. H3 长尺度修订

H3 不再把 `N=100` 当唯一候选。先审计每个连续 session 对
`N in {10^2,10^3,10^4}` 的真实历史支持，并把历史不足者标为不可测。

连续覆盖审计后的固定三人事实是：

- 958 虽有 TRAIN 88,082、validation 30,884 个事件，但单个无缺口记录段最多只有
  1,599 个既往事件；validation 中 1,426 个下一事件可检验 1,000 次尺度；
- 620 的单个无缺口记录段最多 1,570 个既往事件；validation 中 779 个下一事件可
  检验 1,000 次尺度；
- 黄瀚文单段最多 285 个既往事件，不能检验 1,000 或 10,000 次尺度；
- 固定三人无人能检验 10,000 次尺度。六人 development 池中只有张家齐单段达到
  15,714，且有 5,715 个 validation 下一事件可检验 10,000 次尺度；但在张家齐具备
  同合同 target-trained T1 checkpoint 前，不运行该人体比较。

因此固定三人合同只约束 R1.3；长尺度 T2-S1 从既有六人 development 池中按
**预先可观测性**选择，而不是按效果选择。人体首轮使用递推 exposure accumulator，
不保存长窗口，不做完整 N/tau/seed 矩阵：

- `10^2` 作为短尺度参照；
- `10^3` 作为主要长尺度；
- `10^4` 只在同一无缺口记录段有完整历史且已有同合同 T1 checkpoint 时运行；
- 首轮 exposure 固定为 signed load innovation；participation 作为独立后续 secondary，
  不由 load 结果决定是否执行，也不在本最小仪器标定中临时选择表示；
- 四臂固定为 T1 no-edge、real cumulative exposure、state-matched innovation
  placebo、current-event-only jump；
- primary endpoint 为事件完成后关闭 raw correction、one-step 预测下一事件；
- 不按患者选择最佳 N，不把 recorded gap 当作观察到的 IED 历史。

## 5. 暂缓

- 34 人扩展；
- 正式检验分区；
- 完整 T2 physical-clock/scale 大网格；
- 新的频谱 forecast/Conformer 比较；
- state dimension 扩容；
- seizure probe 与 subtype 机制结论。

## 6. 最终交付

- R1.2b persistent/memoryless 与 strict-swap 机器结果；
- R1.3 代码、测试、3 人 x 3 seed 可恢复运行与患者优先汇总；
- T2 `10^2/10^3/10^4` 可观测性表、synthetic recovery 和最小 one-step pilot；
- 白话报告、技术报告、机器审计、`RUN_STATUS.json` 与
  `CURRENT_HANDOFF.md`；
- 明确区分工程完成、可测性、开发级结果和科学验收。
