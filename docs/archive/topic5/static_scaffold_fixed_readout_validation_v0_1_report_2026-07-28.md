# Topic 5 RNN 审阅后修复、验收与 static contact topography v1.0 总报告

日期：2026-07-28
状态：按预注册边界完成；RNN 冻结为 bounded supplementary analysis

## 1. 一句话结论

这轮补充实验确认了一个可保留但必须收窄的结果：

\[
\boxed{\text{间期事件可恢复患者特异的静态 contact morphology，并与发作早期能量场对应}}
\]

但没有证据表明这一对应需要 GRU 的有序长历史、自动辨识出的物理轴或方向性发作动力学：

\[
\boxed{\text{当前支持 orientation-free static contact correspondence，不支持 GRU-specific static increment}}
\]

这不是对“患者不存在病理轴”的否定。它只说明当前 rank-sequence GRU 没有提供超出强静态
估计器的独立跨状态增量。

## 2. 审阅意见如何落实

| 审阅要求 | 实际修改 | 验收结果 |
|---|---|---|
| participation 必须固定，不能每次选择最优 field | 唯一 primary field 冻结为 interictal participation | 完成 |
| primary 必须是 signed，而不能只看 `abs(rho)` | 固定正方向 signed Spearman；`abs(rho)` 降为形态 sensitivity | 完成 |
| all-contact null 太弱 | 新增 within-shaft circular、dihedral、equal-size shaft-profile 和 geometry-smooth null | 完成 |
| RNN 可能只是静态平滑器 | 增加 raw、beta-binomial、shaft/geometry Laplacian、Dirichlet rank 和低秩非递归估计器 | 完成 |
| baseline 不能看 ictal target 选 | 只用 chronological train60/validation20 选择，再在 train80 重拟合 | target seal 通过 |
| free rollout 可能自己制造静态场 | 导出 96 个 teacher-forced heldout-prefix fields，与 free rollout 配对比较 | fingerprint 与 target seal 通过 |
| contact confound 需控制 | 分别控制 shaft position、geometry、spacing/density、SOZ、baseline power 和 broadband power | 完成；GM/WM、artifact rate 缺失 |
| ordered history 不能只靠扰动实验 | 并列报告 formal heldout full-vs-rank-shuffle 和 matched perturbation | 完成，二者结论分开 |
| 当前 early-ictal target 已反复读取 | 全部标记为 same-dataset internal validation | 完成 |
| 不应继续调 structured-axis RNN | 冻结 GRU、axis、source、loss、seed 与 rollout | 未发生重训或事后选参 |

## 3. 数据和预测任务

### 3.1 间期自监督任务

- formal cohort：34 人；
- 每人 chronological train80 / heldout20；
- 3 个冻结 seeds；
- 输入：事件内已经观察到的 contact rank sets；
- 输出：下一 rank set、未来 participation 和 contact-rank distribution；
- full GRU 与 rank-shuffle GRU 使用同一数据分母和评分合同。

这里的 formal 问题是：

> 真实顺序是否比保持 membership、但打乱 rank order 的输入带来 heldout 预测增益？

### 3.2 跨状态静态任务

- strict cohort：Epilepsiae 16 人、106 seizures；
- target：clinical onset 后 `[0,10] s`、`1–150 Hz` baseline-normalized contact energy；
- 唯一 field：interictal participation；
- patient statistic：先在 seizure 内做 contact-wise Spearman，再取 seizure median；
- signed primary：更高 interictal participation 对应更高 early-ictal energy；
- sign-free sensitivity：`abs(rho)`，只判断空间形态是否对应。

这个任务预测的是患者级静态 contact field，不是逐秒 seizure trajectory、临床发作时间或
后续 recruitment order。

## 4. 模型和对照到底是什么

| 模型 | 使用的信息 | 科学作用 |
|---|---|---|
| raw train80 participation | 每个 contact 在训练事件中参与的频率 | 最直接的静态病理场 |
| best regularized | 对 raw field 做 beta-binomial、shaft/geometry smoothing 或低秩收缩 | 检查 GRU 是否只是去噪/平滑 |
| static contact hazard | contact bias，不读事件历史 | no-history 对照 |
| first-order | 只读最后一个 rank set | 一阶传播对照 |
| rank-shuffle GRU | 同样的 GRU，但训练顺序被打乱 | ordered-history 对照 |
| teacher-forced full GRU | 沿真实 heldout prefixes 聚合 one-step hazards | 检查自由 rollout 是否制造结果 |
| free full GRU | 从冻结模型自由 rollout 得到最终 participation field | 目标 RNN readout |

正则化模型完全没有读取 early-ictal target。16 人中，由 interictal validation 选择出的最优
估计器为：beta-binomial 7 人、低秩 logit 5 人、shaft Laplacian 2 人、geometry
Laplacian 2 人。

## 5. 结果

### 5.1 GRU 会不会使用事件顺序

两个看起来相近、但含义不同的实验必须分开：

| 实验 | n | 中位效应 | 正向患者 | P | 结论 |
|---|---:|---:|---:|---:|---|
| formal heldout full GRU 相对最佳 nonrecurrent prefix 模型的 NLL 增益 | 34 | 0.0010 | 17/34 | 不显著 | full history 未超过最佳非递归模型 |
| formal heldout full GRU 相对独立训练 rank-shuffle GRU 的 NLL 增益 | 34 | 0.0408 | 27/34 | \(P=2.51\times10^{-4}\) | 真实顺序相对打乱顺序有泛化增益 |
| 在同一 full GRU 中打乱顺序造成的 NLL 代价 | 34 | 0.0100 | 32/34 | \(1.79\times10^{-8}\) | 模型内部确实使用/编码了顺序 |

因此安全结论是：

> full GRU 对事件顺序敏感，真实顺序相对独立训练的 rank-shuffle GRU 具有稳定 heldout
> 增益；但 full-history GRU 没有超过最佳 nonrecurrent prefix 模型，因此不能把这一结果
> 解释为“无限历史递归状态是必要的”。

原报告把 `ordered_history_nll_gain` 误标为 full-vs-rank-shuffle；该字段实际定义为
`strongest_nonrecurrent_nll - full_history_gru_nll`。上表已按原始
`heldout_metrics.csv` 重新计算真正的 full-vs-rank-shuffle 配对比较。

### 5.2 固定正方向是否成立

full GRU participation 的结果：

| null | n | signed margin 中位数 | 正向患者 | P |
|---|---:|---:|---:|---:|
| all-contact | 16 | 0.243 | 11/16 | 0.126 |
| within-shaft circular | 16 | 0.113 | 11/16 | 0.163 |
| within-shaft dihedral | 16 | 0.108 | 11/16 | 0.150 |
| geometry-smooth | 13 | 0.241 | 9/13 | 0.122 |

预设正方向没有队列级支持。5/16 患者方向为负，另有患者接近零，因此不能写成“高间期
participation contact 一致具有更高发作早期能量”。

### 5.3 sign-free 静态形态是否存在

full GRU 的 `abs(rho)` morphology margin：

| null | n | 中位数 | 正向患者 | P |
|---|---:|---:|---:|---:|
| all-contact | 16 | 0.215 | 14/16 | 0.000153 |
| within-shaft circular | 16 | 0.136 | 13/16 | 0.00205 |
| within-shaft dihedral | 16 | 0.142 | 12/16 | 0.000944 |
| geometry-smooth | 13 | 0.183 | 9/13 | 0.00671 |

所以，静态 spatial morphology 不是单纯由任意 contact assignment、同 shaft 内位置互换或
平滑三维场自动产生。这个结果可以保留，但它不指定 polarity，也不等于方向性 replay。

### 5.4 这个结果是否需要 GRU

raw train80 participation 本身已经表现为：

- all-contact absolute margin 中位约 0.196，13/16 为正，\(P=0.000656\)；
- within-shaft absolute margin 中位约 0.079，12/16 为正，\(P=0.00226\)；
- geometry-smooth absolute margin 中位约 0.100，9/13 为正，\(P=0.0341\)。

full GRU 与 best regularized field 的 contact-wise Spearman 中位数为 0.941。full GRU
相对 best regularized 的 all-contact absolute-margin 增量仅 0.0119，\(P\approx0.11\)，
FDR \(q=0.122\)；相对 rank-shuffle、static hazard 和 first-order 也没有稳定独立增量。

因此：

> 当前跨状态阳性首先属于 interictal contact statistics，而不是 GRU 特有动力学。

### 5.5 teacher forcing 与 free rollout

teacher-forced full field 与：

- free full field 的 Spearman 中位数：0.802；
- best regularized field：0.807；
- teacher-forced rank-shuffle field：0.896。

teacher-forced full 的 signed readout 不显著；all-contact absolute margin 中位 0.024，
\(P=0.0719\)。free full 相对 teacher-forced 的 absolute-margin 增量为 0.129，
13/16 为正，exact \(P=0.00101\)，但 free full 同时没有超过 raw、best regularized 或
rank-shuffle。

所以 free-vs-teacher 差异只能写成：

> 自由 rollout 会增强静态形态 readout。

它不能写成 ordered recurrent dynamics 的证据，因为同样的最终形态不要求 ordered GRU。

### 5.6 contact confound

单混杂 partial-rank 分析使用 rank-space residualization，并以 Freedman–Lane residual
permutation 构建患者内 null。它不是多变量因果校正。

在 shaft position、geometry PC1 和 SOZ 分别控制后，full 与 raw 的 sign-free residual
margin 多数仍为正；控制 raw participation 后，full 的 residual morphology 仍有小幅信号，
但其与 rank-shuffle/best regularized 的配对增量是判断 RNN 特异性的关键，不能只看 full
相对零。

baseline band power 与 broadband power 已从原始 seizure windows 重建并纳入最终分析。
两者在 16/16 患者均与模型 contact 完整精确匹配。控制 baseline power 后：

- full GRU absolute residual margin 中位 0.281，15/16 为正；
- raw participation 中位 0.278，15/16 为正；
- full − rank-shuffle 仅 0.0025，\(P=0.188\)；
- full − best regularized 仅 0.0065，\(P=0.161\)。

控制 `1–250 Hz` broadband power 得到同样结论：full 和 raw 的 residual morphology 都
保留，但 full 不超过 rank-shuffle 或 best regularized。

在 within-shaft-position 单块中，full 相对 rank-shuffle/best regularized 有很小的配对
增量（中位 0.0085/0.0334，family \(q=0.0467\)）；这个增量没有在 geometry PC1、SOZ、
baseline power 或 broadband power 中重复，所以只能保留为单块 sensitivity，不能升为
GRU-specific robustness。

控制 raw participation 后，full 自身的 absolute residual margin 中位 0.109，
\(P=0.00459\)，但 full − rank-shuffle 仅 0.0156（family \(q=0.161\)），full − best
regularized 也未过 family 校正。因此这一结果仍不能绕过强基线。

完整数值和精确 contact join 见：

- `BASELINE_POWER_CONFOUND_AUDIT.json`
- `phase4_contact_confound_cohort_summary.csv`
- `phase4_contact_confound_paired_comparisons.csv`

当前仍不能控制：

- GM/WM；
- artifact/rejection rate。

这些变量没有可靠缓存，因此未插补，也没有宣称已排除。

## 6. RNN 结构是否合理

### 6.1 对自监督序列建模：基本合理

GRU 输入逐步 rank sets、预测 next contact/future rank，数据切分按患者内 chronological
train/heldout，并与 rank-shuffle、no-history、first-order 使用同一分母。这足以用于：

- 描述 contact participation/rank distribution；
- 检查模型是否对事件顺序敏感；
- 生成患者级 static field。

### 6.2 对论文的动力学机制：当前不够

当前结果没有证明：

- ordered long history 提高 heldout prediction；
- hidden state 含有超出 last-set/static field 的必要信息；
- PCA state 是患者病理轴、source 或生物 E/I state；
- 该模型能预测同一次 seizure 的时序传播。

因此不应继续对当前 GRU 或 structured-axis RNN 调超参数。若未来重开动态线，至少需要同时
满足：

1. formal full GRU 稳定优于 rank-shuffle/first-order；
2. free rollout 稳定优于 strongest target-free static baseline；
3. 增量在 within-shaft 与 geometry-aware null 下保留。

当前三个条件均未满足。

## 7. Figure 6 六块各自回答什么

| Panel | 科学含义 | 当前结论 |
|---|---|---|
| A | 固定、target-sealed 的跨状态 static-field 合同 | 预测的是 contact field，不是 seizure trajectory |
| B | 模型使用顺序，是否也获得 heldout 顺序增益 | 使用顺序，但无 formal heldout gain |
| C | 预设正方向的 signed correspondence | 未建立 |
| D | sign-free morphology 是否超过空间 null | 在 all-contact、within-shaft、smooth-field 下保留 |
| E | full GRU 是否超过强静态/一阶/打乱对照 | 未建立 GRU-specific gain |
| F | 控制单个 contact confound 后是否仍有 morphology | sensitivity 支持，但不是完整因果控制 |

该图的论文级别是 `supplementary_candidate`，不应替代主文中已经由真实数据建立的
interictal–early-ictal shared-field 结果。

## 8. 科学目标偏移审计

### 没有偏移的部分

- 输入仍是原始 SEEG 的 contact-rank event sequence；
- 发作 target 仍是论文已有的 clinical-onset 后 early energy field；
- 没有把 IEI、A/B 分类或 seizure-prefix completion 重新引入；
- 没有把 A/B 轴当作金标准；
- 没有为了让 RNN 阳性而重新选择 field、polarity、patient subgroup 或 seeds。

### 主动收窄的部分

- 从“RNN 找到跨状态机制”收窄为“间期数据含可迁移的静态 contact scaffold”；
- ordered-history 只保留为 model-usage diagnostic；
- PCA/internal-state read-back 降为 exploratory；
- physical axis、source reversal 和 dynamic seizure replay 标记为当前合同未证明。

这是依据对照结果做的结论收窄，不是把原假设改成更容易阳性的任务。

## 9. 验收

- static-scaffold utility 与相关 RNN 测试：29/29 通过；
- target-free baseline：16/16，target seal 通过；
- teacher-forced cells：96/96，NPZ fingerprint 与 target seal 通过；
- Phase 1/2/3 patient-metric 行数：480/560/160；
- baseline-power confound map：16 人合并后做 exact contact join audit；
- 最终 `FINAL_ACCEPTANCE.json`：`PASS_WITH_BOUNDED_STATIC_CONCLUSION`；
- 未发现 OOM、NaN、Traceback 或分母漂移。

执行合同到冻结结论：100%。没有运行新的 dynamic RNN，是科学停止条件的结果，不是欠项。

## 10. 下一 goal

下一合同已经冻结为：

> **Topic 5 RNN bounded closeout and replication readiness v0.1**

它不再训练 RNN，任务是：

1. 将当前结果作为 bounded Supplementary Results/Figure 收口，RNN 仅作为边界对照；
2. 保持 signed primary、`abs(rho)` sensitivity 和全部 baseline/null 不变；
3. 只在真正未参与本轮 target 读取的新 clinical-onset cohort 上做独立复制；
4. 若没有独立 cohort，停在 replication-ready handoff；
5. 在 formal order gain 与 GRU-specific static increment 出现前，不启动新的
   early-ictal dynamics model。

metadata-only inventory 显示当前缓存 20 人，其中 16 人已经进入本轮 target；剩余 4 人没有
相同 strict clinical-onset endpoint。当前独立复制状态为：

```text
BLOCKED_NO_UNTOUCHED_STRICT_CLINICAL_ONSET_COHORT
```

不能把 EEG onset、患者级 SOZ、A/B source 或同一批 target 的重新分组替代独立复制。

## 11. 审阅后论文定位

当前可靠对象不是“自动恢复病理轴”或“发作期重放”，而是患者特异的
`interictal contact topography` 与 clinical-onset 后 `[0,10] s`、`1–150 Hz`
early-ictal energy 在 contact 层面的 **orientation-free spatial
correspondence**。`abs(rho)` 允许相同或反向排列，因此不能解释为 positive replay。

ordered GRU 对 prefix 内顺序扰动敏感，且 formal heldout 比较显示真实顺序优于独立训练的
rank-shuffle GRU；但 full-history GRU 没有超过最强 non-recurrent prefix model。跨状态
静态对应也可由 raw/regularized non-recurrent fields 得到，未建立 GRU-specific static
increment。因此该线应放入 Supplementary，承担“事件顺序含有泛化信息，但无界 recurrent
history 和 recurrent-specific 跨状态增量未建立”的边界控制，而不是主文机制桥。

当前 manuscript-facing source：

- `docs/paper-draft/figure6_static_contact_topography_bounded_result.md`

独立复现必须使用本轮从未读取过 early-ictal target 的新患者。新患者可以只用其间期数据
拟合 patient-specific field，但不能复用旧患者权重，也不能把同一患者的新 seizure 当成
patient-level external replication。现有缓存没有满足该条件的独立 strict
clinical-onset cohort，因此 replication protocol 已冻结但暂时阻断。

## 12. 主要产物

- `results/topic5_static_scaffold_fixed_readout_validation/FINAL_ACCEPTANCE.json`
- `results/topic5_static_scaffold_fixed_readout_validation/BASELINE_FREEZE.json`
- `results/topic5_static_scaffold_fixed_readout_validation/BASELINE_POWER_CONFOUND_AUDIT.json`
- `results/topic5_static_scaffold_fixed_readout_validation/REPLICATION_INVENTORY.json`
- `results/paper-ready-figure/fig6_static_contact_topography/figures/fig6_static_contact_topography.png`
- `docs/superpowers/specs/2026-07-28-topic5-static-contact-topography-supplementary-closeout-v1_0.md`
- `docs/superpowers/plans/2026-07-28-topic5-static-contact-topography-supplementary-closeout-v1_0.md`
- `docs/superpowers/specs/2026-07-28-topic5-external-clinical-onset-replication-protocol-v1_0.md`
- `docs/superpowers/plans/2026-07-28-topic5-external-clinical-onset-replication-protocol-v1_0.md`
