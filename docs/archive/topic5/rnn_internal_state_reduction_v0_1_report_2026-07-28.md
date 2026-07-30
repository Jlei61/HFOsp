# Topic 5 RNN internal-state reduction v0.1 完整报告

日期：2026-07-28

## 1. 审阅结论

### 一句话判断

结果必须拆成三层。第一，患者特异的间期 static contact scaffold 与发作早期宽频能量
之间存在内部数据集上的空间对应。第二，full-history GRU 的预测确实对真实 rank 顺序
敏感，这是纯间期 heldout 扰动得到的可靠诊断结果。第三，PC1/PC2 residual contact
field 的 early-ictal 对应是在已经打开过的 target 上追加的机制拆解，只是探索性候选。

这不是“自动恢复患者病理轴”，也不是“RNN 预测发作传播”。修订后的证据等级为：

- **静态 contact scaffold：中等偏强的内部证据**，但尚未通过 signed fixed-readout、
  更强空间 null、简单平滑基线和完整 contact-confound 验证；
- **ordered-history sensitivity：较强的间期诊断证据**，来自匹配顺序扰动；
- **ordered latent state → early-ictal bridge：探索性证据**，不是独立确认，也不是
  当前主文优先结论。

### 完成度

**科学合同执行：100/100。**

- 34 人、3 seeds、full-history 与 rank-shuffle 两种 GRU 均完成；
- 102/102 hidden-state extraction、34/34 subject analysis、102/102 perturbation、
  102/102 random-subspace cell 完整；
- 所有 interictal direction 在读取本阶段 early-ictal target 前冻结；
- strict clinical-onset 16 人、106 次发作全部完成；
- all-contact primary null 与 within-shaft sensitivity 均为 5,000 次 coherent
  permutation；
- 20 项相关测试通过；
- 复现与完整性审计通过。

## 2. 这次具体做了什么

### 2.1 没有重新训练更复杂的网络

本阶段冻结并复用了既有 34 人 × 3 seeds GRU：

- `full_history_gru`：按真实 rank-set 顺序自监督学习下一 contact/STOP；
- `rank_shuffle_gru`：容量相同，但事件内 rank 顺序在训练时被破坏，同时保留事件
  participation、rank-set size 和 contact support。

目标不是比较谁的 AUC 更高，而是打开 GRU，检查它内部是否存在少数稳定状态，以及这些
状态到底承载了什么信息。

### 2.2 数据拆分

每位患者保持原来的 chronological train80/heldout20，再把 train80 固定拆成：

- `train60`：拟合 PCA、线性 probe 和方向；
- `validation20`：选择 probe 正则和温度；
- `heldout20`：只用于最终 interictal 指标。

大患者每个 split 使用固定、等距、可复现的事件抽样，避免内存随事件数失控；所有 seeds
使用同一批事件与 prefix。

### 2.3 四组验证

1. **低维性与稳定性**：PCA effective rank、k80/k90/k95、跨 seed CKA、
   chronological split-half。
2. **信息内容**：用 hidden PCs 预测下一 action、future participation 和 remaining
   rank，并和 unordered prefix、last-set、rank-shuffle 比较。
3. **因果扰动**：
   - 在同一个 full GRU 内打乱或反转已观察 prefix 的 rank-set 顺序；
   - 保持 prefix contact、候选 contact、下一步 target 和 STOP 完全不变；
   - 对 PCA 和 output-coupled directions 做 `±0.25/0.5/1 SD` 扰动。
4. **跨状态读回**：direction 全部冻结后，读取严格 clinical-onset 的 0–10 s、
   1–150 Hz early-ictal energy field。

## 3. 间期结果

### 3.1 GRU 状态是低维的

| 指标 | Ordered GRU | Rank-shuffled GRU |
|---|---:|---:|
| effective rank，中位数 | 1.878 | 1.336 |
| 90% variance 所需维数 | 3 | 2 |
| raw cross-seed CKA | 0.961 | 0.962 |
| residual cross-seed CKA | 0.924 | 0.931 |

Ordered GRU 的 2 个 PC 保留中位 85.5% heldout variance，4 个 PC 保留 96.5%，8 个
PC 保留 99.1%。相同维数的 PCA 子空间比随机子空间更能保留 decoder NLL，例如 k=2
的 PCA advantage 中位为 0.0796，32/34 患者为正。

**批注**：低维、跨 seed 稳定是真的，但 rank-shuffle GRU 也低维且稳定，所以“低维”
本身不能证明模型学到了传播顺序。

### 3.2 普通线性 probe 能读出信息，但不能单独证明顺序特异性

在 unordered prefix observables 之上加入 8 个 ordered hidden PCs：

- next-action NLL benefit 中位 0.0180，33/34 患者为正；
- future-participation Brier benefit 中位 0.00187；
- remaining-rank MSE benefit 中位 0.000692。

但 rank-shuffle hidden PCs 也给出相近增益；ordered-minus-shuffle 的通用线性 probe
差异很小，next-action k=8 的中位差为 0.000417，置信区间跨 0。

**批注**：hidden state 是有效压缩器，但线性 probe 会利用静态 contact prior、prefix
size 等通用结构，不能作为 ordered dynamics 的主要证据。

### 3.3 匹配的顺序扰动给出了更直接的阳性证据

对相同 heldout prefix 仅打乱已经观察到的 rank-set 顺序：

| 指标 | Ordered GRU | Rank-shuffled GRU | Ordered − shuffled |
|---|---:|---:|---:|
| order-shuffle NLL penalty | 0.01198 | 0.00160 | 0.01004 |
| 正向患者数 | 32/34 | 28/34 | 32/34 |
| 95% bootstrap CI | [0.00828, 0.02028] | [0.00107, 0.00273] | [0.00792, 0.01860] |
| paired P | — | — | \(1.79\times10^{-8}\) |

反转 prefix 顺序时，ordered-minus-shuffle NLL penalty 更大，中位为 0.0210，
95% CI [0.0127, 0.0318]。

**批注**：这是本阶段最可靠的 ordered-history 证据。它不依赖 A/B 标签或预定义病理
轴，只问同一批 contact 以不同顺序到达时，模型内部状态和预测是否改变。

### 3.4 contact field 的稳定性

所有扰动 field 先在单个 event 内平均，再跨 event 平均，避免长事件拥有更大权重。

- Ordered PC1 contact field 跨 seed Spearman 中位 0.611；
- Ordered PC2 中位 0.633；
- heldout chronological half stability 分别为 0.979 和 0.991；
- output-coupled direction 1 跨 seed为 0.772；
- output-coupled direction 2 不稳定，中位仅 0.118。

**批注**：PC1/PC2 可以保留；output-coupled direction 2 不应进入主要生物学解释。
很高的 split-half 稳定性表示同一模型的 decoder effect 稳定，不等于患者真实物理轴
已经被辨识。

## 4. Early-ictal 迁移

### 4.1 严格合同

- 数据集：Epilepsiae；
- anchor：逐发作 clinical onset；
- cohort：16 人、106 次发作；
- target：0–10 s、1–150 Hz contact-level energy；
- Yuquan EEG onset 不进入 primary；
- primary null：患者内 all-contact coherent permutation；
- anatomy sensitivity：within-shaft coherent permutation；
- 先 seizure、再 seed、最后 patient 折叠。

这里预测的是**静态 early-ictal spatial energy field**，不是后续 seizure recruitment
顺序，也不是 seizure onset 时间。

### 4.2 普通模型 fixed readout

Full GRU 的静态 participation field：

- all-contact null margin 中位 0.214，95% CI [0.129, 0.372]；
- 14/16 患者为正，P=0.000153。

Full GRU endpoint field：

- all-contact null margin 中位 0.214，95% CI [0.057, 0.250]；
- 12/16 患者为正，P=0.000656。

但 full GRU 相对 static、unordered、last-set 或 rank-shuffle 的 participation/endpoint
paired 优势均没有稳定通过 FDR。

**批注**：这说明间期 contact scaffold 与发作早期能量场的静态桥梁存在，但不能把这
一部分独占归因于 GRU 或 rank 顺序。当前正式 null 仍基于 absolute correlation；
`signed_rho` 虽已保存，但尚未配置相应 signed permutation/null，因此不能把这里写成
“模型正确预测了哪些 contact 能量更高”。

### 4.3 去 participation 后的内部状态方向

为了不让高参与 contact 同时在两种状态中都偏高造成假阳性，先在 interictal 数据内对
每个方向的 contact field 回归掉 participation，再比较 ordered 与 rank-shuffle。

| Direction | Null | Ordered − rank-shuffle margin | 95% CI | 正向患者 | FDR q |
|---|---|---:|---:|---:|---:|
| PC1 | all-contact | 0.176 | [0.067, 0.231] | 14/16 | 0.0054 |
| PC2 | all-contact | 0.098 | [0.009, 0.164] | 12/16 | 0.0090 |
| PC1 | within-shaft | 0.127 | [0.031, 0.171] | 12/16 | 0.0310 |
| PC2 | within-shaft | 0.095 | [0.000, 0.158] | 11/16 | 0.0356 |

原始、未去 participation 的 PC field 并不显著优于 rank-shuffle；阳性来自
participation-independent residual field。

把 seizures 按交替顺序拆成 A/B 两半：

- PC1：A 半中位 0.0868，12/16 为正；B 半中位 0.185，11/14 为正；
- PC2：A 半中位 0.104，13/16 为正；B 半中位 0.0781，11/14 为正。

但患者级 A/B effect size 的 Spearman 很低（PC1 −0.035；PC2 0.121）。

**批注**：阳性不是由单一 seizure half 驱动，但个体效应大小还不稳定。因此当前可以说
“cohort-level bridge 可重复出现”，不能说“已经得到稳定的患者级 biomarker”。

### 4.4 必须保留的审阅边界

1. v2.5 已经打开过同一个 early-ictal target，本阶段是同数据上的机制拆解，不是独立
   验证。
2. PCA direction 在 target-blind 阶段冻结；但 participation residualization 是本轮
   看过 target 后补入的更严格分析，因此这一项应明确标为 exploratory/post-target。
3. 结果支持的是 contact-level state field 与 early-ictal energy 的对应，不是物理
   病理轴恢复，也不是发作预测器。

## 5. 对核心科学目标有没有偏移

### 没有偏移的部分

- 输入始终是原始 SEEG 的简化 contact rank-set sequence；
- GRU 仍以自监督 next-contact/STOP 学习间期事件内部传播；
- 不把 A/B 模板作为监督 label，因此没有退化成聚类；
- early-ictal 端仍使用论文已有的静态 broadband energy field；
- 核心比较是 ordered history 与 rank-shuffle，而不是“GRU 是否比所有 baseline
  AUC 更高”。

### 尚未回答的部分

- 没有从全部患者自动恢复可信的物理病理轴；
- 没有证明 PC1/PC2 就是 SNN 中某个细胞级 E/I 变量；
- 没有预测 clinical onset，也没有预测 0–10 s 内逐秒传播；
- 没有证明个体级迁移强度在不同 seizures 中稳定。

因此本阶段不是整条 RNN 线的 no-go。它把问题拆成：

> 静态 scaffold 需要先证明其不只是 contact participation、空间平滑或测量偏差；
> ordered-state 的跨状态读回则保留为独立验证候选，不能覆盖静态主问题。

## 6. P0 / P1 审阅

### P0

无。完整性、target cohort、数值复现和测试均通过。

### P1-1：不是独立 target confirmation

**为什么严重**：同一 16 人/106 seizure target 已在 v2.5 查看，不能把当前显著性写成
新的独立复制。

**怎么处理**：论文中明确写 exploratory mechanism decomposition；下一步冻结 PC1/PC2
和 residualization，在新增病例或真正未使用 seizures 上确认。

### P1-2：患者级 seizure-half 效应不稳定

**为什么严重**：虽然 A/B 两半在 cohort level 都为正，但患者排序不一致，临床个体
预测价值尚未建立。

**怎么处理**：后续优先增加每患者 seizures 数，并报告 test-retest reliability，而不是
继续增加 GRU hidden size 或 seeds。

### P1-3：低维性并非 ordered-specific

**为什么严重**：rank-shuffle GRU 同样低维、稳定。只展示 spectrum 会高估机制含义。

**怎么处理**：Figure 和正文必须同时展示 matched order perturbation；不能只用
effective rank/CKA 下结论。

### P1-4：当前 fixed readout 仍以 absolute correlation 计分

**为什么严重**：正相关与负相关都会计为成功，不能直接解释成能量高低方向预测。

**怎么处理**：下一合同把 participation 的方向预先固定为正，primary 使用 signed
Spearman；absolute correlation 只作 morphology sensitivity。

### P1-5：GRU 尚未超过正则化非递归 estimator

**为什么严重**：free rollout 本身会平滑和收缩 contact distribution。当前对照没有
区分“RNN 学到长历史”与“RNN 是一个较好的平滑器”。

**怎么处理**：增加 Dirichlet/Laplace-smoothed participation、smoothed contact-rank
histogram、低秩非递归 estimator，以及 teacher-forced 聚合场。

## 7. 下一步建议

1. **冻结 structured-axis RNN 和当前 internal-state target read-back，不继续调
   hidden size、axis 或 source term。**
2. 下一主任务改为 `Interictal–early-ictal static scaffold fixed-readout validation
   v0.1`：唯一 primary field 为 participation，唯一方向为正相关。
3. 同时比较 raw/smoothed empirical、low-rank non-recurrent、last-set、unordered、
   rank-shuffle 和 full GRU，并拆分 free rollout 与 teacher-forced field。
4. primary 同时报告 all-contact、within-shaft circular shift/reversal、shaft-label
   permutation；geometry-smooth null 只在坐标完整且可辨识的患者中报告。
5. contact confound 分层处理：当前可直接使用 participation/HFO support、shaft
   position、contact spacing、coordinates 和部分 SOZ；baseline power、GM/WM、
   artifact rate 必须先做 availability audit，缺失时不得插补或宣称已控制。
6. 只有 fixed signed readout 在强 null 和简单平滑基线后仍保留增量，才把
   ordered-history necessity audit 升为下一主任务；PC1/PC2 read-back 等待新患者或
   真正未看过的 seizures 做 confirmation。

## 8. 可交付产物

- Figure：
  `results/paper-ready-figure/fig6_rnn_internal_state_reduction/figures/fig6_rnn_internal_state_reduction.png`
- Figure PDF：
  `results/paper-ready-figure/fig6_rnn_internal_state_reduction/figures/fig6_rnn_internal_state_reduction.pdf`
- Figure 说明：
  `results/paper-ready-figure/fig6_rnn_internal_state_reduction/figures/README.md`
- Interictal summary：
  `results/topic5_rnn_internal_state_reduction/INTERICTAL_SUMMARY.json`
- Perturbation/random-subspace summary：
  `results/topic5_rnn_internal_state_reduction/INTERICTAL_SENSITIVITY_SUMMARY.json`
- Event-first contact-field summary：
  `results/topic5_rnn_internal_state_reduction/INTERICTAL_EVENTFIRST_FIELD_SUMMARY.json`
- Strict early-ictal summary：
  `results/topic5_rnn_internal_state_reduction/EARLY_ICTAL_READBACK_SUMMARY.json`
- Reproduction audit：
  `results/topic5_rnn_internal_state_reduction/REPRODUCTION_AUDIT.json`
- Final status：
  `results/topic5_rnn_internal_state_reduction/FINAL_STATUS.json`
