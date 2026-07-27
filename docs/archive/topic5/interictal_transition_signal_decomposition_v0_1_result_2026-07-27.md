# Interictal transition signal decomposition v0.1

> 日期：2026-07-27  
> 状态：31/31 正式运行完成；允许起草 v2.3 合同，不等于机制结论成立  
> 结果根：`results/topic5_interictal_transition_decomposition_v0_1/`

## 1. 一句话结论

Markov 的增益不只是同 shaft 或欧氏邻近：它主要包含一个患者特异、以对称成分为主、
可跨 shaft 泛化且依赖有序多步历史的 transition signal。物理轴相关 residual 和
source-conditioned modulation 均按冻结统计门通过，因此允许设计最小 v2.3
recurrent observation model；但 source 项效应很小，而且 14/22 患者的 axis
coefficient 为负，当前只能写成 **axis-aligned anisotropy**，不能写成“沿病理轴
增强传播”。

## 2. 合同与分母

- coordinate-free：31 位 development-excluded patients；
- physical-axis：22 位 geometry-complete patients；
- cross-shaft endpoint：20/22；`epilepsiae_139` 和
  `yuquan_zhangjiaqi` 因 heldout 中没有跨-shaft next transition 被预先排除；
- 每位患者 chronological train80 / heldout20；
- 所有 node hazard、pair residual、geometry coefficient、axis、source scalar 和
  history decay只使用 train80；
- 所有正式比较共用一个 11 项 BH-FDR family；
- A/B、IEI、SOZ、clinical/EEG onset 和 ictal values均未使用；
- early-ictal target seal 始终为 false。

## 3. Markov 到底学到了什么

### 3.1 Prefix 信息稳定存在

| 比较 | 中位 heldout NLL benefit | 95% bootstrap CI | 正效应患者 | BH-FDR q |
|---|---:|---:|---:|---:|
| probability Markov > node-bias | 0.01438 | 0.01075, 0.01720 | 30/31 | 2.56e-9 |
| directed-logit Markov > node-bias | 0.01438 | 0.01075, 0.01720 | 30/31 | 2.56e-9 |

这两种 Markov 写法在当前 last-rank arithmetic 下数学上等价，因此不是两个独立
replications。可以保留的结论是：当前 rank set 对下一 contact 的信息稳定超过
contact 的边际参与频率。

### 3.2 主要信号是对称 residual，不是任意有向网络

| 比较 | 中位 benefit | 正效应患者 | BH-FDR q |
|---|---:|---:|---:|
| symmetric residual > node-bias | 0.01479 | 30/31 | 2.56e-9 |
| skew increment > symmetric residual | -4.35e-5 | 15/31 | 0.252 |

因此经验 transition signal 并不要求一张任意 directed matrix。这个结果与“底层
scaffold 近似对称”相容，但仍然只是 effective transition residual，不是 anatomical
connectivity recovery。

### 3.3 局部植入几何解释不了主要增益

22 位 physical-axis patients中，`same_shaft + distance` 相对 node-bias 的中位
benefit 只有 0.00044（12/22，q=0.168）；完整 Markov 相对该 local geometry
control 的中位增益为 0.00981（21/22，q=1.24e-5）。

在正式 cross-shaft conditional likelihood 中，Markov 相对 local geometry 的中位
benefit 为 0.01570（95% CI 0.00893–0.02345；19/20；q=1.31e-5）。该评分同时包含
跨-shaft阳性和阴性 eligible contacts，不是只奖励 target contact。

### 3.4 有物理轴相关 residual，但符号不能过度解释

physical-axis residual 相对 local geometry 的中位 benefit 为 0.00215
（95% CI 0.00092–0.00381；20/22；q=4.37e-6）。

同时：

- local/axis feature Frobenius cosine 中位 0.972；
- selected axis 与 contact-cloud PCA1 absolute cosine 中位 0.413；
- axis-excess coefficient：8/22 为正，14/22 为负；
- 负 coefficient 患者的 axis benefit 中位 0.00321，明显高于正 coefficient 患者的
  0.00056。

因此数据支持的是局部几何之外的 **axis-aligned anisotropy**。它不能直接命名为
正值的轴向传播 scaffold；负 residual 可能反映 next-set competition、refractory
suppression 或 observation mapping，而不是负连接。

### 3.5 Source-conditioned modulation 通过统计门，但实际贡献很小

source-conditioned axis 相对 axis residual 的中位 benefit 为 1.73e-5
（15/22；q=0.00861）。其 bootstrap median CI 下界略低于 0，患者效应集中在接近零
的范围。去除两位没有跨-shaft transition 的患者后，descriptive 中位 benefit 为
8.20e-6（13/20；未校正 one-sided P=0.024）。

因此这一项只足以允许在 v2.3 中保留一个强约束、单 scalar 的 source-conditioned
方向项；不能把它写成主要预测来源或已恢复双向病理轴。

### 3.6 真正支持 recurrent state 的是多步历史

| 比较 | 中位 benefit | 正效应患者 | BH-FDR q |
|---|---:|---:|---:|
| last-rank > source-only | 0.01841 | 29/31 | 5.12e-9 |
| ordered full-prefix > last-rank | 0.00681 | 30/31 | 2.56e-9 |

`last_2_ranks` 和 `last_3_ranks` 相对 last-rank 的 descriptive 中位 benefit 分别为
0.00591 和 0.00591；ordered history 的 train-only decay 选择为 0.25（3 人）、
0.50（20 人）和 0.75（8 人）。

这一结果说明简单 first-order Markov 没有饱和，允许使用低维历史状态；它不要求
GRU，也不支持自由 dense hidden dynamics。

## 4. Go / no-go

冻结合同的五个条件均通过：

1. directed Markov 超过 local geometry，且正式 cross-shaft endpoint通过；
2. symmetric residual 超过 node-bias；
3. physical-axis residual 超过 local geometry；
4. source-conditioned modulation 超过 axis residual；
5. ordered history 超过 last-rank。

自动状态为 `GO_V2_3_RNN`。这里的 `GO` 只表示：

> 可以起草并工程验证一个“共享对称 scaffold + 极小 source-conditioned direction +
> 1–2 维历史/竞争状态”的 v2.3 合同。

它不表示可以读取 early-ictal target，也不表示 v2.3 已经通过。

## 5. v2.3 必须保留的边界

- 不用普通 GRU；
- 不允许 dense contact-to-contact bypass；
- 不引入 A/B label 或离散 path identity；
- positive symmetric scaffold 与 signed effective residual 分开；
- 显式建模 next-set cardinality/competition；
- source-conditioned方向项只允许一个低容量 scalar；
- recurrent state 的存在由 ordered-history benefit 支持，但维度固定为 1–2；
- v2.3 仍先做纯间期 heldout；early-ictal transfer 继续被 clinical source metadata
  阻断。

## 6. 图

### 审计图

`figures/transition_signal_decomposition.{png,pdf}`

保留自动 runner 的所有预注册比较。

### Paper-ready supplementary 图

`figures/transition_signal_decomposition_paper_ready.{png,pdf}`

- A：Markov 相对 node、local geometry 和正式 cross-shaft geometry control；
- B：symmetric residual 与额外 skew；
- C：axis residual 与 source modulation，并显示 axis coefficient 正负；
- D：source-only、last-rank 和 ordered history。

该图是下一版模型设计依据，不是主文 Figure 6 的完成版。

## 7. 对核心科学目标的审阅

### 没有偏移的部分

- 输入仍是 template-free contact-rank events；
- 核心问题仍是 interictal propagation structure，而不是 IEI 或 A/B 分类；
- 分解明确区分局部几何、对称 scaffold、source 和历史状态；
- 发作期数据没有被提前读取。

### 需要主动收窄的部分

- `axis residual pass` 不能改写成“恢复了正向病理轴”；
- `source modulation pass` 的效应量很小，不能作为主阳性结论；
- `GO_V2_3_RNN` 是模型开发许可，不是机制证据；
- clinical-onset exact source仍为 0/71，跨状态桥接继续 blocked。
