# Topic 5 external clinical-onset replication protocol v1.0

## 1. 状态

```text
READY_BUT_BLOCKED_NO_INDEPENDENT_PATIENT_COHORT
```

本合同只有在发现从未参与当前 target 读取的新患者后才能激活。当前 16 人的新 seizures
只能检验患者内稳定性，不能构成 patient-level external replication。

## 2. 要复制的科学对象

### Endpoint 1：orientation-free morphology

\[
H_{\mathrm{morphology}}:
|\rho(f_{\mathrm{interictal}},E_{\mathrm{ictal}})|
-
\operatorname{median}|\rho|_{\mathrm{within\text{-}shaft\ null}}
>0.
\]

primary field 为 raw train80 interictal participation。within-shaft circular 是全体新患者的
primary spatial null；all-contact 用于与当前结果连续对照，geometry-smooth 用于坐标完整子集。

这一 endpoint 检验同序或逆序的共同 contact organization，不检验 positive replay。

### Endpoint 2：signed direction and heterogeneity

\[
H_{\mathrm{signed}}:
\rho(f_{\mathrm{interictal}},E_{\mathrm{ictal}})>0.
\]

它作为层级第二 endpoint，用于判断固定正方向能否复制，并报告正向、反向和近零患者比例。
不得使用 ictal target 为患者翻转 polarity。

### Endpoint 3：GRU increment

在 Endpoint 1 的 frozen raw/best-regularized topography 之后，比较 full GRU 与：

- best target-free regularized field；
- rank-shuffle GRU；
- first-order model。

它不是 morphology replication 的前置门。

## 3. 独立单位

`independent replication` 必须是从未读取 early-ictal target 的新患者，而不是：

- 当前 16 人的新 seizures；
- 当前 16 人的重新分组；
- EEG onset 替代 clinical onset；
- SOZ、A/B source 或 energy-top contacts 替代 exact clinical-onset contract。

## 4. 冻结流程与可重新拟合部分

冻结的是流程，不是旧患者的 node-specific 权重。冻结项包括：

- GRU 结构、hidden size、loss 和 seeds；
- chronological train/validation split；
- field definition；
- regularizer candidate set 与 validation selection rule；
- target window/band；
- null、statistic、exclusion criteria；
- endpoint hierarchy 和 equivalence margin。

对每个新患者允许仅用其 interictal 数据：

- 估计 raw participation；
- 按冻结 train60/validation20 规则选择并在 train80 重拟合 regularized field；
- 按冻结训练器拟合 patient-specific GRU/contact calibration。

任何 ictal target 在上述 fit、selection 和 QC 完成前保持 sealed。

## 5. Target

- exact clinical-onset time；
- patient/contact exact join；
- `[0,10] s`；
- `1–150 Hz` baseline-normalized contact energy；
- patient-first statistic：seizure-level contact Spearman 后取 patient median；
- 预先记录每名患者 seizure 数、contact 数、shaft 数、坐标完整性和排除原因。

## 6. “无 GRU 有意义增量”的边界

未显著不能证明等效。若要写“静态 topography 已足够”，冻结最小有意义增量：

\[
\delta_{\mathrm{static}}=0.05
\]

单位为 orientation-free morphology margin。选择 0.05 是因为它约为当前 raw-field
cohort-median effect 的四分之一；低于此值不改变论文中 static-topography 结论。

只有 patient-bootstrap 单侧 95% 上界满足：

\[
\operatorname{UCB}_{95}
(\Delta_{\mathrm{full\ GRU-best\ static}})
<0.05
\]

才能写“未发现科学上有意义的 GRU 增量”。否则只能写 `INCONCLUSIVE_FOR_EQUIVALENCE`。

full GRU vs rank-shuffle 的 heldout NLL 必须报告 patient-bootstrap 95% CI，不得只报告
\(P>0.05\)。

## 7. 样本量与激活门

在读取任何新 target 前生成 `POWER_AND_PRECISION_FREEZE.json`：

- 使用当前 patient-level morphology-margin 分布估计方差；
- primary 目标为 Endpoint 1 的 patient-level power 与 CI precision；
- 显式报告预计排除率；
- 不用“现有多少人”反推便利阈值；
- 新患者数量不足时保持 blocked，不把 seizures 当独立样本扩充 power。

## 8. 数据获取路线

按优先级审计：

1. 为未进入当前 16 人的缓存患者补 exact clinical-onset annotation、同质 target 与 contact join；
2. 从 Epilepsiae 或医院原始资料增加新 clinical-onset 患者；
3. 前瞻性建立 seizure-level clinical-onset time、contact annotation、target producer 和
   interictal rank-event pipeline。

每条路线都必须记录：

- target 是否曾被任何当前设计读取；
- annotation source 与 reviewer；
- exact join；
- endpoint 同质性；
- 可用患者数。

三条路线均不可行时，状态改为：

```text
CLOSED_NO_FEASIBLE_INDEPENDENT_PATIENT_COHORT
```

不得无限保持 replication-ready。

## 9. 重开动态模型的门

只有独立患者同时显示：

1. full GRU 对 rank-shuffle/first-order 有稳定 heldout NLL 增益；
2. full GRU 对 strongest target-free static field 有跨状态增量；
3. 增量在 within-shaft 和 geometry-aware null 下保留；

才允许设计新的 hidden-state 或 dynamic seizure model。
