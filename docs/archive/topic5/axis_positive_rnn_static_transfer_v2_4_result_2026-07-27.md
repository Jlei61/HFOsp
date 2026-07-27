# Axis-positive RNN read-back 与 early-ictal static transfer v2.4 结果

> 日期：2026-07-27
> 状态：执行至冻结停止点；27/27 axis-search tasks、42/42 representation tasks 完成，
> 0 正式失败；Gate A/S/H/X 均已判定。
> 主图：
> `results/paper-ready-figure/fig6_rnn_axis_static_transfer_v2_4/figures/fig6_rnn_axis_static_transfer_v2_4.png`

## 总判断

这次补充实验直接检验了两个修正后的问题：

1. 只看原数据已经独立支持 A/B 轴共线的患者，RNN 的 next-contact objective 能否
   重新选出同一病例轴；
2. 把间期事件汇总成每个 contact 的 participation/rank distribution 后，能否
   source-free 地读出 clinical onset 后 `[0,10] s`、1–150 Hz 静态能量场。

两项主检验均未通过。轴阳性亚组并没有挽救 axis recovery：RNN 在三个 seed 中稳定
选择同一方向，但所选方向相对冻结 A/B shared axis 的 alignment margin 中位数为
−0.205。这个方向相对 isotropic model 的 next-contact NLL 有小幅增益，但它不是
原数据中的病例轴。静态迁移同样失败：full RNN 的患者级 all-contact null margin
中位数为 −0.153，history 和 physical-axis term 都没有提供跨状态增量。

因此，当前 RNN 最安全的定位是：

> 间期 contact-rank sequence 可由低容量 history-dependent model 做自监督预测，
> 而且模型能近似恢复每个 contact 的粗粒度 participation 和 rank order；但当前
> next-contact objective 学到的方向不能命名为 A/B 病理轴，模型生成的节点分布也
> 不能解释发作早期静态能量场。

这不否定论文已有的 empirical A/B axis 或 interictal–early-ictal field result。
它否定的是：当前结构化 RNN 已经把这两项经验发现统一成了一个可辨识内部机制。

## 1. 做了什么

### 1.1 输入与分母

- v2.3 formal physical-axis cohort：22 人；
- 原分析预先支持 A/B 轴共线：9/22；
- 其中 reversed：6/9；strict reversed：5/9；
- clinical-onset BB150 static target-ready：14/22；
- axis-positive 与 target-ready 交集：5 人，仅作描述性 sensitivity；
- 所有间期模型保持 chronological train60 / validation20 / heldout20；
- node bias、source distribution、event-length distribution 和 axis selection 均不读
  heldout20；
- early-ictal target 在五类间期 representation 完全冻结并写出 SHA256 后才读取；
- EEG onset 不替代 clinical onset。

### 1.2 Axis read-back

Stage A0 不重训，只把 v2.3 既有 train80 transition-selected axis 读回冻结 A/B
shared axis。Stage A1 才是真正的 RNN axis selection：

- 9 patients × 32 个 sign-invariant Fibonacci directions × 3 seeds；
- 每个方向训练同一个 `axis_two_state_no_source`；
- validation20 选择方向；
- 方向冻结后才读取 heldout20；
- reversed n=6 再训练同一方向上的 source term sensitivity；
- optimizer、persistence、batch、maximum epochs 和 patience 全部继承 v2.3。

正式共 27/27 patient-seed tasks、864 个 candidate fits 完成。资源恢复时短暂产生的
CPU/GPU mixed-device candidate 已移出 formal tree；全部正式 candidate 从 clean CPU
state 完成，未进入选择的隔离文件仅保留作审计。

### 1.3 节点级间期表征

对14位 target-ready patients，每个 v2.3 model seed free-rollout 5000 次：

- source set 和 event length 从 train80 empirical distribution 抽样；
- full/no-history/isotropic/node-only 共用随机流；
- 每个 contact 输出11维分布：不参与概率 + 10 个 normalized-rank joint bins；
- 三 seed 先逐 contact-feature 取中位数，再做 simplex closure；
- empirical_train80 直接由 train80 rank events 计算。

共 14 patients × 3 seeds = 42/42 tasks 完成。

### 1.4 Source-free early-ictal static readout

- target：clinical onset 后 `[0,10] s`，1–150 Hz baseline-robust-z energy；
- 每位患者先跨 seizure 取 contact-wise median；
- exact contact-name join，至少6个共同 contact；
- outer LOSO by patient；
- ridge `alpha=1.0`，只用 outer-training patients 做标准化；
- 每位训练患者总权重相等；
- 每位 heldout patient 计算 contact-level Spearman rho；
- primary null：5000次患者内 all-contact target-label permutation；
- within-shaft null 只作敏感性。

## 2. 结果

### 2.1 已有 transition-selected axis 只有趋势性 read-back

在 n=9 axis-positive subgroup 中：

- alignment margin median = 0.300；
- 95% bootstrap CI = [−0.141, 0.456]；
- 6/9 为正；
- one-sided Wilcoxon P=0.180。

它说明旧 transition decomposition 的选轴与 A/B axis 有一定趋势，但既不是 RNN
axis discovery，也没有通过患者级不确定性检验。该轴相对 isotropic model 的
heldout benefit 中位数接近0。

### 2.2 RNN 选择的方向可重复，但不是冻结 A/B 病理轴

Stage A1 的患者级结果：

| 指标 | median [95% bootstrap CI] | 正向患者 | P |
|---|---:|---:|---:|
| selected-axis alignment margin | −0.2047 [−0.4368, 0.0519] | 2/9 | 0.936 |
| selected axis − isotropic NLL benefit | 0.00798 [0.00226, 0.03096] | 8/9 | 0.00391 |
| reversed source − no-source benefit | 0.000041 [−0.00131, 0.00184] | 3/6 | 0.422 |
| seed-to-seed selected-axis cosine | 1.000 [1.000, 1.000] | 9/9 | — |

Gate A 要求 alignment 与 heldout predictive increment 同时为正，因此判定
`FAIL`。

这个组合很有解释价值：优化在三个 seed 中确实落到同一个方向，而且该方向对
next-contact prediction 有增量；但它没有对应原数据预先冻结的 A/B shared axis。
所以它更可能是一个有效 transition/implant direction，而不能命名为患者病理轴。
seed stability 在这里是优化重复性，不是结构可辨识性。

### 2.3 RNN 学到了粗粒度节点分布，但 axis information 几乎没有改变该分布

full RNN 与 empirical train80 distribution 的患者级中位关系为：

- participation probability Spearman rho = 0.922；
- conditional expected-rank Spearman rho = 0.742；
- mean per-contact total-variation distance = 0.106。

full 与各消融模型的 mean per-contact total-variation distance：

- full vs isotropic：0.00467；
- full vs no-history：0.0130；
- full vs node-only：0.0317。

因此，“每个 contact 是否参与、通常早还是晚”的粗粒度排序确实被模型保留了；但
full 与 isotropic 的完整概率分布几乎相同。轴项在 next-contact NLL 上的小增益没有
变成一个明显不同的节点级长期分布，这与后续 Gate X 失败一致。

### 2.4 Full RNN 不能 source-free 地读出 early-ictal static field

| Gate | 患者级 estimand | median [95% bootstrap CI] | 正向患者 | BH-FDR q | 结论 |
|---|---|---:|---:|---:|---|
| S | full all-contact null margin | −0.153 [−0.436, 0.601] | 6/14 | 0.520 | FAIL |
| H | full − no-history rho | 0.000 [−0.0357, 0.0456] | 5/14 | 0.520 | FAIL |
| X | full − isotropic rho | 0.000 [−0.0143, 0.0250] | 4/14 | 0.520 | FAIL |

axis-positive × target-ready n=5 sensitivity 也没有转正：

- full rho median = −0.0676；
- full all-contact margin median = −0.0603。

所以结果不能写成 RNN history 或 physical axis 在发作早期被重用。

### 2.5 Empirical distribution 的正向趋势只用于定位模型信息损失

未经 RNN free rollout 压缩的 empirical_train80 distribution：

- all-contact margin median = 0.368；
- 9/14 为正；
- one-sided P=0.0481；
- 95% bootstrap median CI = [−0.336, 0.619]；
- within-shaft margin median = 0.143，10/14 为正，未校正 P=0.0247。

它相对 full RNN 的 rho difference 中位数为0.0887（9/14为正，P=0.0309）。
这些都是冻结 gate 后的 descriptive diagnostics：没有进入 Gate S family，P 未做
多重比较校正，而且 bootstrap CI 跨0。因此不能把它升级成新的 cohort-positive
结果。

它能支持的更窄解释是：

> 原始间期 rank distribution 中可能仍保留与 early-ictal static field 相关的细粒度
> 信息；当前 next-contact model 的 free-rollout compression 没有保留这部分信息。

## 3. Gates 与停止决定

```text
Gate A  axis-positive construct validity: FAIL
Gate S  source-free static readout:         FAIL
Gate H  history contribution to transfer:  FAIL
Gate X  physical-axis contribution:        FAIL
Dynamic source-conditioned ictal rollout:
  BLOCKED_MISSING_EXACT_CLINICAL_ONSET_SOURCE_METADATA
```

按冻结合同：

- 不增加 seeds；
- 不重新选择 `[0,10] s` 窗口；
- 不改变1–150 Hz target；
- 不调 ridge、rollout 数、persistence 或 loss；
- 不追补 target 读取后的 `rnn_selected_axis` static sensitivity；
- 不继续扩展 learnable physical-axis RNN。

## 4. 对核心科学目标是否偏移

### 没有偏移的部分

- 输入仍是 template-free raw contact-rank events，不把 A/B label 当训练目标；
- RNN 学习的是下一触点和历史依赖，不是重新做 KMeans；
- axis-positive subgroup 由旧数据独立冻结，不按 RNN 结果重选患者；
- 发作期 target 与原论文一致：clinical onset、`[0,10] s`、1–150 Hz static energy
  field；
- 所有统计按患者折叠，没有把 contact、seizure 或 seed 当独立样本。

### 当前模型没有完成的核心目标

- 没有从间期 next-contact objective 恢复原数据 A/B 病理轴；
- 没有证明同一个 source-conditioned scaffold 产生反向传播；
- 没有把模型内部 history/axis 信息迁移到 early-ictal field；
- 没有获得可用于机制解释的跨状态 latent dynamics。

因此这条线没有变成“证明 RNN 有用”的性能竞赛，但也没有完成原定的机制桥接。

## 5. 论文定位

不建议把这张图作为主文 Figure 6 的中心机制结论。可以保留为 supplementary
computational closeout，回答三个窄问题：

1. 间期 rank sequence 包含超过 node frequency 的可预测历史信息；
2. 低容量 RNN 可近似恢复节点 participation/rank distribution；
3. 当前物理轴约束和 free-rollout representation 不足以解释 early-ictal field。

主文中原有 empirical A/B field 和 early-ictal energy-field reuse 不需要因这项阴性
结果撤回。相反，这一结果明确了经验关系与当前 system-identification model 之间仍缺
一个不会丢失 contact-level field information 的映射。

## 6. 主要产物

- spec：
  `docs/superpowers/specs/2026-07-27-topic5-axis-positive-rnn-static-transfer-v2_4.md`
- execution plan：
  `docs/superpowers/plans/2026-07-27-topic5-axis-positive-rnn-static-transfer-v2_4.md`
- input audit：
  `results/topic5_rnn_axis_positive_static_transfer_v2_4/input_audit/INPUT_AUDIT_STATUS.json`
- Stage A0：
  `results/topic5_rnn_axis_positive_static_transfer_v2_4/axis_readback_stage_a0/STAGE_A0_STATUS.json`
- Stage A1：
  `results/topic5_rnn_axis_positive_static_transfer_v2_4/formal/AXIS_SELECTION_GATE_STATUS.json`
- rank-distribution fidelity：
  `results/topic5_rnn_axis_positive_static_transfer_v2_4/representations/RANK_DISTRIBUTION_FIDELITY.json`
- static readout：
  `results/topic5_rnn_axis_positive_static_transfer_v2_4/static_readout/STATIC_READOUT_GATE_STATUS.json`
- static diagnostics：
  `results/topic5_rnn_axis_positive_static_transfer_v2_4/static_readout/STATIC_READOUT_DIAGNOSTICS.json`
- paper-ready PNG/PDF/metadata/README：
  `results/paper-ready-figure/fig6_rnn_axis_static_transfer_v2_4/figures/`
