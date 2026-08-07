# Stable-repertoire event-history v2.4：修复、锁定扩展与最终验收

日期：2026-08-02  
状态：**P0 修复完成；六人开发门通过；剩余 28 人锁定扩展完成；PCA + scalar-decay leaky state 阴性收口**

> **2026-08-02 scope correction：**V2.4 的 R1 不是一般可训练 RNN，而是
> `PCA projection + one scalar decay + ridge correction`。本报告只判决这一固定
> low-dimensional leaky-filter family；它不判决 trainable recurrent operator、GRU、
> LSTM、不同 history resolution 或经过 validation 调整的优化配置。下文中旧的宽泛
> “RNN 阴性”措辞均按此边界读取。

## 1. 一句话判断

稳定传播 repertoire 确实提供了建立跨事件预测模型的条件，但严格匹配历史事件数、
recency、distributed lag 和 coherent chronology null 后，锁定扩展没有选择出额外的
固定低维 leaky state。最稳定的结果是：使用过去 80 场事件，比随机抽取 20 场更准确地
估计未来稳定 repertoire；最近 20 场并不优于随机 20 场，因此主要是估计精度，不是普遍
recency，更不是事件顺序驱动的网络演化。

## 2. 五轮修复与验收

### Round 1：修复 chronology null

- 删除逐 forecasting-window 独立打乱的旧 null；
- 在每个 source raw event sequence 上只做一次 20-event block permutation，再重建全部
  overlapping histories；
- safe circular pairing 同步移动 target values、raw indices、positions、start/stop 和 times；
- 每一行强制 history-target 零重叠，并至少相隔一个完整 horizon；
- 保存 source permutation、origin/donor row 和 shifted raw indices。

### Round 2：补齐匹配历史基线

新增：

- first / last / random equal-count `H`-event controls；
- full-token EWMA；
- event-descriptor EWMA；
- four-bin distributed-lag ridge；
- time/IEI nuisance baseline。

所有基线只在 validation 选择，test 不做 oracle selection。低维 state 只与
validation-selected EWMA/lag comparator 比较。

### Round 3：重写可测量性和 endpoint

- score variance 只由 train targets 估计；
- primary 为 occupancy+rank propagation score；
- participation 是 secondary recruitment score；
- reliability 同时报 raw 和 train-mean-residualized 版本；
- 输出每位患者 80-event history 和 20/40-event target 的真实时长。

审计中发现 time-only baseline 曾读取 future target duration。它在扩展前被删除，并增加
future-blind 单测；六人全部重新运行。

### Round 4：六人 development freeze

修复后六人 H=20：

- state-minus-matched propagation 中位数 `-0.00357`；
- true-minus-block-null gain 中位数 `+0.000126`；
- true-minus-circular-null gain 中位数 `+0.00324`。

H=40 三个方向一致。依据事先冻结的“中位方向门”，锁定 runner/module/spec/config hashes，
开放剩余 28 人一次性扩展。六人不进入 primary P 值。

### Round 5：28 人 locked extension

主分析 H=20 有 19/28 位满足冻结的 source/window 合同；9 位因 source 太少、source 内不足
100 场事件或 validation/test 独立窗口不足而 fail closed。H=40 有 17/28 位。失败患者保留
在 denominator audit，未被静默删除。

所有可分析产物再次逐事件检查：旧 heldout20 索引为 0；safe circular history-target overlap
为 0；frozen hashes 与 development release 完全一致。

## 3. Primary 结果：固定 PCA + scalar-decay state 未复现

锁定 H=20、n=19：

| 比较 | 中位差 / gain | 方向患者 | 单侧 Wilcoxon p | 解释 |
| --- | ---: | ---: | ---: | --- |
| low-dimensional state − matched EWMA/lag | +0.000795 | 8/19 更好 | 0.879 | state 没有增量 |
| true gain − block-null gain | +0.000796 | 11/19 | 0.476 | chronology specificity 不成立 |
| true gain − circular-null gain | +0.000621 | 10/19 | 0.461 | chronology specificity 不成立 |

患者 bootstrap 的 state-minus-matched 中位数 95% CI 为 `[-0.00168, +0.00867]`。
Epilepsiae（n=12）和 Yuquan（n=7）的中位差都大于 0，没有相反的 dataset-specific rescue。

H=40 sensitivity 方向相同：state 仅 8/17 更好，中位差 `+0.000286`，Wilcoxon
`p=0.463`；两类 chronology null 也均不显著。

因此 V2.4 为这一固定 leaky-filter family 预设的 primary joint Gate 未通过。该结果不外推到
具有可学习 recurrent matrix/gates、不同归一化或不同训练超参数的 RNN family。

## 4. 真正稳定的结果：更多事件提高稳定 repertoire 的估计精度

H=20、n=19：

- unordered-80 优于 random-20：18/19，中位差 `-0.0498`，Wilcoxon
  `p=5.86e-4`，sign test `p=3.81e-5`；
- recent-20 不优于 random-20：8/19，中位差 `+0.0134`，Wilcoxon `p=0.813`；
- unordered-80 优于 static：14/19，中位差 `-0.122`，Wilcoxon `p=0.00412`；
- validation-selected EWMA/lag 不优于 unordered-80：8/19，中位差 `+0.00192`，
  Wilcoxon `p=0.824`。

19 位中没有一位由 binned-lag 被选为 H=20 最强匹配 baseline。EWMA 的 selected decay
主要为 0.95/0.99，进一步说明模型接近对较长历史做稳定平均，而不是强调最近几场。

正确解释是：

> 更多重复事件使患者特异稳定 repertoire 的估计更精确。

不能解释为：

> 更近的事件或精确事件顺序普遍改变未来传播状态。

## 5. Participation secondary endpoint

低维 state 相对 matched baseline 的 participation error 在 14/19 患者改善，中位差
`-0.00234`，Wilcoxon `p=0.0115`。但其 chronology-specific joint Gate 仍失败：

- true-minus-block-null recruitment gain：Wilcoxon `p=0.0180`，sign test不显著；
- true-minus-circular-null recruitment gain：Wilcoxon `p=0.105`。

因此这只是一个未通过双 null 的 secondary recruitment signal，不能写成稳定的
chronology-sensitive recruitment state，更不能替代阴性的 propagation primary endpoint。

## 6. 测量边界

H=20 可分析患者的 residualized reliability 中位数：

- occupancy：0.385；
- rank：0.498；
- participation：0.684。

这解释了 participation 更容易出现增量，也说明旧 raw reliability 确实被稳定 contact
main effect 抬高。

80-event history 的患者中位真实跨度约 518 秒，20-event target 约 115 秒。因此这里的
event-history state 即使存在，也主要是同一 recording 内分钟尺度，不能写成跨天或长期
网络塑造。source 是 canonical record/block；跨 source 生物连续性未验证，状态按合同重置。

train-only template read-back 在 H=20 的 19 位中 validation 为 14 strong、4 moderate、
1 weak；test 为 13 strong、5 moderate、1 weak。test grade 没有参与纳入选择，weak 患者也
没有按结果删除。

## 7. 最终科学结论

当前可成立：

> 同一患者的大量间期事件反复采样一个稳定的传播 repertoire。使用更多历史事件能够更
> 准确地估计随后事件窗口中的 mode occupancy 和 contact rank；在 V2.4 所检验的固定
> PCA + scalar-decay 参数化下，没有得到稳定的额外 chronology-sensitive 增量。

当前不成立：

- V2.4 固定 PCA + scalar-decay state 的 cohort-level 增量；
- 由该固定模型支持的 chronology-sensitive propagation state；
- event-driven evolving graph；
- 间期事件塑造或重写病理网络；
- RNN 恢复 contact-level biological connectivity。

## 8. 停止决定

按冻结停止规则，**V2.4 固定 leaky-filter 分支**在此收口：

\[
\boxed{\text{events reveal a stable repertoire; the fixed leaky state adds no cohort-level gain}}
\]

本结论不能作为停止一般 RNN 的依据。trainable recurrent operator 属于新的模型合同，必须
独立检验 cell、hidden size、history resolution、normalization、optimizer、learning rate、
batch size 和训练充分性；不得把它包装成对 V2.4 的事后调参 rescue。六人中出现的固定-state
增量仍只保留为 development heterogeneity，不用于反向定义 post hoc subtype。

## 9. 主要产物

- 冻结合同：`docs/superpowers/specs/2026-08-02-topic5-stable-repertoire-event-history-v2_4.md`
- development release：`results/topic5_stable_repertoire_event_history/v2_4/development_acceptance/LOCKED_EXTENSION_RELEASE.json`
- locked extension acceptance：`results/topic5_stable_repertoire_event_history/acceptance_v2_4/LOCKED_EXTENSION_ACCEPTANCE.json`
- patient table：`results/topic5_stable_repertoire_event_history/acceptance_v2_4/extension_patient_horizon_summary.csv`
- denominator audit：`results/topic5_stable_repertoire_event_history/acceptance_v2_4/denominator_audit.csv`
- artifact audit：`results/topic5_stable_repertoire_event_history/acceptance_v2_4/ARTIFACT_AUDIT.json`
