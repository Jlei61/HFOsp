# Topic 5 / Figure 6：interictal–ictal shared-axis structured RNN v0.4

**状态**：implementation-ready，单次冻结执行  
**日期**：2026-08-05  
**继承**：v0.3 的患者内 chronological split、target seal、contact/STOP/cardinality 分解、ordinary GRU、static baseline、exact-k rollout、patient-first statistics 与 Figure 6 资产。  
**新结果根**：`results/topic5_interictal_ictal_shared_axis_rnn_v0_4/`  
**禁止覆盖**：`results/topic5_patient_specific_source_conditioned_rnn_v0_3_final/`

## 1. 唯一科学问题

本轮只回答：

> 患者内 structured RNN 能否仅从间期 contact-rank sequences 中恢复一个稳定、可双向读取的患者特异 effective propagation axis；在完全冻结模型后，该轴派生的空间场是否与同一患者 clinical-onset 后早期 1--150 Hz broadband energy field 形成 target-free 对应？

主逻辑为：

\[
\boxed{
\text{interictal rank sequences}
\rightarrow
\text{frozen patient-specific bidirectional axis}
\rightarrow
\text{early-ictal broadband field correspondence}
}
\]

本轮不再把“完整事件生成是否达到 data-vs-data 自相似度”作为主终点，也不训练跨患者的共享 readout。每位患者独立训练；cohort 统计只是患者级效应量的汇总。

## 2. 可支持与不可支持的结论

### 2.1 目标结论

若证据链完整，可写：

> A patient-specific structured RNN recovered a bidirectional effective propagation axis from interictal contact-rank sequences. Without exposure to ictal energy targets, spatial fields generated from this frozen axis aligned with the early-ictal broadband energy field in the same patient, supporting reuse of a shared pathological propagation axis across interictal and early ictal states.

中文口径：

> structured RNN 仅凭间期事件恢复了患者特异的双向有效传播轴；冻结轴产生的空间场与同患者发作早期 broadband energy field 对应，支持间期与发作早期对同一病理传播轴的跨状态读出。

### 2.2 解释边界

- `axis` 是 SEEG contact space 中的 effective propagation coordinate，不是解剖连接纤维束。
- `shared` 表示空间轴/场的跨状态对应，不表示间期与发作期具有完全相同的瞬时动力学。
- RNN state 是 rank-step state，不是秒级生物时间常数，也不表示细胞级 E/I 单元。
- early-ictal score 是 target-free association，不是 seizure forecasting。
- 经验 A/B 不是金标准，不参与训练、超参数选择、source 定义或 checkpoint 选择；它只用于冻结后的外部 read-back。

## 3. 队列与数据合同

### 3.1 间期队列

- 使用 v0.3 已审计的 34 位患者 rank dataset。
- 必须使用 masked participant-only rank 表示；不得读取 phantom-contaminated finite ranks。
- 每位患者按事件时间固定为 `fit60 / validation20 / test20`。
- 每场事件以 rank-set multi-hot 序列输入；不参与 contact 必须保持 masked，不可填成有限 rank。
- 每位患者单独拟合；不做 LOSO 权重训练，不共享 patient-specific node parameters。
- 三位 development patients 仅用于 interictal-only 学习率/预算选择：
  - `epilepsiae_1073`
  - `epilepsiae_1146`
  - `yuquan_chenziyang`
- 正式汇总同时报告：31 位 development-excluded confirmation 和 34 位全 cohort description。

### 3.2 发作期队列

- 复用 v0.3 已冻结的 early-ictal metadata inventory、clinical-onset anchor、exact contact join 和患者分母；不得看到 target 后增删患者。
- primary target：clinical onset 后 `0--10 s`、`1--150 Hz`、control-normalized contact broadband energy。
- EEG-onset 对齐只作 sensitivity；其不显著不推翻 clinical-onset 主结果。
- seizure 先在患者内汇总，再进行 patient-first cohort 统计。
- `epilepsiae_1146` 继承既有 target-seal incident，只能作 supportive representative，不进入 primary P 值或模型选择。
- 新 v0.4 的 primary 患者在 axis/field manifest 冻结前不得反序列化任何 ictal energy 数值。

### 3.3 两个预先定义的分析层

1. **All eligible**：所有能完成 exact contact join 的发作期患者，完整报告，不因模型质量删除。
2. **Interictal-axis-identifiable**：只使用间期数据、在 target seal 关闭时定义：
   - 两个 source sides 在 test20 各至少 20 场事件；
   - 三 seed sign-aligned axis 的最小两两绝对 Spearman `>=0.50`；
   - chronological split-half axis 的绝对 Spearman `>=0.50`；
   - source-excluded bidirectional matched-minus-swapped margin `>0`。

该子集用于回答“模型确实辨识出轴的患者中是否存在跨状态对应”。它不替代 all-eligible 结果，且不得依据 ictal target 修改阈值。

## 4. v0.4 唯一 structured RNN

### 4.1 学习到的 contact coordinate

每位患者学习一个标量 coordinate `s_i`。每次前向先中心化并按 population RMS 标准化：

\[
s_i\leftarrow
\frac{q_i-\bar q}{\sqrt{N^{-1}\sum_j(q_j-\bar q)^2+\epsilon}}.
\]

符号本身不可辨识；所有 seed/split 比较必须先做全局正负号对齐。禁止根据 A/B 或 ictal target 选择符号。

### 4.2 对称 scaffold 与同轴反对称 flow

固定 same-shaft local graph 为 `G_local`。沿 learned coordinate 构造：

\[
K^{axis}_{ij}=
\exp\left[-\frac{(s_i-s_j)^2}{2\ell_s^2}\right],
\qquad i\neq j.
\]

先分别把 `G_local` 与 `K_axis` 缩放到相同的非对角 Frobenius norm，再混合：

\[
A^S=(1-\gamma)\bar G_{local}+\gamma\bar K_{axis},
\qquad 0\leq\gamma\leq1.
\]

使用保对称的归一化：

\[
W^S=gD^{-1/2}A^SD^{-1/2},
\qquad W^S=(W^S)^\top.
\]

同一 coordinate 派生唯一方向算子：

\[
W^A_{ij}=W^S_{ij}
\tanh\left(\frac{s_i-s_j}{\delta}\right),
\qquad W^A=-(W^A)^\top.
\]

`W_A` 不是另一套自由连接，也不允许独立 forward/reverse 参数。

### 4.3 source-conditioned direction

每场事件只用已经观察到的第一 rank set `x_0` 定义方向：

\[
d_e=-\tanh\left[
\kappa\operatorname{mean}_{i\in x_0}(s_i)
\right].
\]

负端 source 得到向正端的 flow，正端 source 得到向负端的 flow。`d_e` 在事件剩余 rank steps 中冻结，event reset 时清零。不得用完整事件或最终长度计算方向。

### 4.4 rank-step recurrent state

输入按当前 rank set 的 active contact 数量归一：

\[
\tilde x_t=x_t/\max(1,\sum_i x_{i,t}).
\]

状态固定为：

\[
P_{t+1}=\rho_PP_t+W^S\tilde x_t+lambda_A d_eW^A\tilde x_t,
\]

\[
R_{t+1}=\rho_RR_t+W^S\tilde x_t,
\]

\[
z_{i,t+1}=b_i+\beta_PP_{i,t+1}-\beta_RR_{i,t+1}+m_{i,t}.
\]

其中 `m_i=-inf` mask 已参与 contacts。contact-choice、STOP、cardinality 继续复用 v0.3 的 likelihood contract；STOP/cardinality 只读 permutation-invariant summaries。

### 4.5 结构正则与禁止绕行

轴正则只使用 same-shaft graph：

\[
L_{smooth}=\lambda_s
\frac{\sum_{ij}G^{local}_{ij}(s_i-s_j)^2}
{\sum_{ij}G^{local}_{ij}+\epsilon}.
\]

`s` 已强制单位 RMS，不再添加可被尺度交换绕过的 axis-amplitude 参数。

正式 primary 的结构常数在读取任何 v0.4 ictal target 前固定为：

- `ell_s=1.0`；
- `delta=1.0`；
- `lambda_s=0.01`；
- `gamma=sigmoid(gamma_raw)`，初始化为 `0.50`；
- global operator gain 初始化为 `softplus(0)=0.693`；
- `rho_P` 初始化约 `0.378`，`rho_R` 初始化约 `0.689`，并保持 `rho_R>rho_P`；
- `beta_P=1.0`、`beta_R=0.25`；
- `lambda_A=0.5`；
- `kappa=2.0` 固定，不再与 axis scale 共同学习。

除 learning rate 外，这些值不进入网格搜索；如 synthetic/unit sanity 显示方向符号或数值稳定性错误，应修实现而不是调参数。

严禁：

- dense contact decoder、MLP contact mixer 或自由 `N x N` residual matrix；
- 两套 forward/reverse operators；
- A/B labels、template mean ranks、SOZ、clinical-onset contacts 或 ictal energy 进入训练；
- future event length、最终参与数等未来信息；
- 看到 ictal score 后调整 `lambda_s / gamma / delta / loss / source pool / horizon / checkpoint`。

## 5. 训练合同

### 5.1 目标

总 loss 沿用 v0.3：

\[
L=L_{contact\mid continue,k}
+w_{stop}L_{stop}
+w_kL_{cardinality}
+L_{smooth}.
\]

主要间期 endpoint 必须单独报告 `contact identity | continue,k`，不得用 STOP 或 set size 的改善冒充传播方向预测。

### 5.2 冻结超参数

- seeds：`11, 29, 47`。
- optimizer：沿用 v0.3 的 AdamW、`weight_decay=0`、原 gradient clipping。
- 初始候选 learning rate：`0.01, 0.03, 0.1`，只在三位 development patients、seed 11、validation20 上选最小 median contact NLL。
- 正式训练预算固定为 `28 cycles x 32 updates`。另在三位 development patients 跑 `84 cycles x 32 updates` 作为收敛 sensitivity；若其 median validation improvement `>=0.005 nats/decision`，必须在报告中标记预算边界，并可在剩余墙钟允许时追加长预算确认，但不得因此阻止 primary 链在 8--10 小时内完成。
- `lambda_s`、`ell_s`、`delta`、初始化与 loss weights 在正式执行前写死在 config，不做 ictal-guided sweep。
- ordinary GRU/static 的既有 v0.3 checkpoints 可复用，但必须验证 split、contact order、seed、runner/core SHA 与 checkpoint schema；ordinary fields 必须用 v0.4 source pools 和 rollout denominator 重新生成。

### 5.3 正式运行单位

- v0.4 structured main：34 patients x 3 seeds = 102 units。
- v0.4 structured within-event rank-shuffle：34 patients x seed 11 = 34 units。
- chronological split-half axis stability：34 patients x 2 halves x seed 11 = 68 units。
- ordinary GRU：优先复用 102 个 v0.3 checkpoints；不兼容时才重训。
- axis lesions 不重训，均从冻结 structured checkpoint 重新 rollout。

合计优先新增 204 units；设计目标是在一张 RTX 3090、12--14 个并发 worker 下于 8--10 小时内完成训练、readout、图和报告。

## 6. 间期证据

### 6.1 预测一致性

对 test20、每患者先跨 seed 汇总：

- contact-choice NLL；
- top-1 next-contact accuracy；
- structured minus static；
- structured minus ordinary GRU；
- structured true-order minus within-event rank-shuffle。

structured 不要求在 NLL 上超过容量大得多的 ordinary GRU；这里的主要用途是确认它不是退化到 static，且真实顺序被使用。

### 6.2 双向轴恢复

source pools 只由冻结 `s` 的两端分位数定义：每侧取 `max(2, ceil(0.2N))` 个 contacts。评估时：

- 给定 held-out 事件真实 first rank，score 只看后续 ranks；
- 排除被强制指定的 source contacts；
- 两个 source sides 分别报告 observed-vs-rollout expected-rank Spearman；
- 报告 matched-minus-swapped margin；
- 报告 `F+` 与 `F-` 的方向对比，不把“不同起点自然产生不同图”误写成模型自发发现两类。

### 6.3 轴稳定性

- 三 seed sign-aligned axis 两两绝对 Spearman；
- chronological split-half axis 绝对 Spearman；
- endpoint-set Jaccard；
- 在 geometry-complete 患者中，将 learned `s` 对 contact coordinates 做只读线性投影，报告可解释方差；此项只是 physical-geometry sensitivity，不决定 effective-axis 主分析。

### 6.4 经验 A/B 外部 read-back

从 masked train60/validation20 间期数据独立构造 empirical A/B templates，再在 test20 做 read-back。A/B 不作为真值门，只报告：

\[
M_{AB}=\frac{\rho(F^+,A)+\rho(F^-,B)}{2}
-\frac{\rho(F^+,B)+\rho(F^-,A)}{2},
\]

同时允许全局正负/标签交换后取最佳匹配。不得用全数据 A/B 选择模型 source pools。

## 7. 冻结场与跨状态检验

### 7.1 唯一 model fields

在 target seal 关闭时，从 learned `s` 两端 source pools 分别进行 exact-k rollouts。每方向、每 seed 5000 次；horizon、random seeds、candidate mask 和 source pool 对所有模型/lesion 相同。

唯一正式场为 participation-weighted first-arrival earliness：

\[
F_i^d=\sum_{t=1}^{H}
\left(1-\frac{t}{H}\right)
P(T_i=t\mid S^d),
\qquad d\in\{-,+\}.
\]

同时冻结轴对比场：

\[
G_i=z(rank(F_i^+))-z(rank(F_i^-)).
\]

`F-/F+/G`、source pools、horizon、checkpoint/config/code SHA256、contact order 和 field fingerprint 必须写入 immutable manifest。manifest 完成前不读取 early-ictal values。

### 7.2 跨状态主指标

对每次 seizure：

\[
S_p^{model}=
\max\left(
|\rho(F_p^-,Y_p^{ictal})|,
|\rho(F_p^+,Y_p^{ictal})|
\right).
\]

主 null 完全复用论文现有 contact-label permutation 口径：5000 次，每次都重新执行 absolute value 与 two-direction maximum。within-shaft permutation 为 sensitivity。

轴特异指标：

\[
A_p^{model}=|\rho(G_p,Y_p^{ictal})|.
\]

它回答 early-ictal field 是否沿模型两方向之差排列，而不是只由共同 participation 决定。

### 7.3 必须比较的模型和 lesion

1. `structured_full`：完整 v0.4。
2. `structured_flow_lesion`：`lambda_A=0`，保留同一 node bias、W_S、source pools 和训练 checkpoint。
3. `structured_axis_permutation`：shaft-preserving permutation `s` 后重建 `W_S/W_A`，其余不变；不少于 256 个轴 null draws。
4. `ordinary_gru`：同患者、同 split、同 source pools、同 exact-k rollout。
5. `static_participation`：train60/validation20 的 contact participation prior。
6. `empirical_AB`：冻结 masked interictal A/B fields；它是数据参照，不是要被 RNN 超越的训练基线。

### 7.4 统计

- 每 seizure 先得 score；每患者取 seizure median；患者是 cohort 统计单位。
- 报 observed score、null median、observed-minus-null margin、individual empirical P 和是否超过自身 p95。
- exact one-sided Wilcoxon/sign-rank 必须先按固定 `1e-9` tie tolerance 删除并列，再使用精确零分布。
- bootstrap 95% CI：5000 draws、patient resampling。
- 预先报告以下 paired effects，不把它们压成一个总 gate：
  - full minus its all-contact null；
  - full minus flow lesion；
  - full minus ordinary GRU；
  - full minus static participation；
  - full 与 empirical A/B 的差距；
  - axis-specific `A_p` 相对 shaft-preserving random-axis null。
- all-eligible 与 interictal-axis-identifiable 两层分别报告；不得只展示较好的一层。

## 8. 如何判读“共用轴”

这不是运行 gate，而是写作层级：

### Level 0：没有轴证据

interictal axis 不稳定，或 source-excluded 两侧传播不成立。只能报告 structured model 的局部顺序预测。

### Level 1：target-free 共同空间结构

`S_full` 超过 contact-label null，但 full 不优于 flow lesion/static。可写：模型间期场与 early-ictal 场对应；不能写方向轴被复用。

### Level 2：共享 effective axis

需要同时看到：

- target-independent axis-identifiable 患者中，interictal 双向 readout 与 split/seed stability 成立；
- `S_full` 超过 contact-label null；
- `S_full > S_flow_lesion`，或 axis-specific `A_p` 超过 random-axis null；
- 效应不由单一患者或单一数据集驱动。

此时可写“间期与发作早期共用同一 effective pathological axis”。

### Level 3：structured advantage

在 Level 2 基础上，`S_full > S_ordinary_GRU` 且超过 static participation。此时可进一步写 structured inductive bias 提供了普通无结构模型没有恢复出的跨状态轴信息。

## 9. 工程停止条件与非停止条件

### 9.1 只允许以下硬停止

- target seal 被 primary 患者提前破坏；
- test/target 泄漏；
- mixed config/core/runner hashes；
- 同一 output root 混入不兼容 checkpoint；
- 连续三次 OOM，降低 worker/chunk 后仍无法恢复；
- NaN/Inf 或单测失败无法定位。

### 9.2 以下不是停止条件

- structured NLL 不如 ordinary GRU；
- 只有部分患者轴稳定；
- full 没有超过某个对照；
- early-ictal 结果为阴性。

这些都必须走完分析、制图和报告；禁止看到 target 后继续调参追阳性。

## 10. Figure 6 六块

### A｜structured RNN

画真实计算结构：rank-set input、learned `s`、`W_S`、同轴 `W_A`、first-rank direction `d_e`、P/R state 与 contact/STOP/cardinality outputs。明确 `patient-specific`、`interictal-only training`、`no A/B labels`。

### B｜两方向间期传播恢复

固定 E1146 supportive representative。上下两行是 learned source-minus/source-plus；每行并排 observed test20 rank heatmap 与 model rollout heatmap。排除强制 source 后显示 expected-rank profile。标清“source-conditioned directions”，不写“模型无监督发现 A/B”除非 read-back 支持。

### C｜全 cohort 间期一致性

患者配对显示 contact-choice NLL：static、structured、rank-shuffle、ordinary GRU；附 source-excluded bidirectional matched-minus-swapped score。主图显示 31 人 confirmation，34 人 description 放同 panel 的浅色层或 inset。

### D｜轴稳定性与必要性

显示三 seed/split-half axis stability、两侧 source 的 heldout benefit，以及 full vs flow-lesion。不得只画 seed 重复性。

### E｜代表患者的跨状态共享轴

同一 contact geometry 上并排 `F-`、`F+`、axis contrast `G` 和 clinical-onset early-ictal 1--150 Hz field。model maps 的场尺度统一；early field 可单独 colorbar，但必须注明 within-map normalization。两张方向场都展示，不得只展示事后更像 target 的一个。

### F｜患者级跨状态统计

同患者配对展示 structured full、flow lesion、ordinary GRU、static participation 和 empirical A/B。主纵轴为 all-contact-null-corrected `S` margin；axis-identifiable 患者用预先固定符号标记。小 inset 展示 axis-specific `A_p` 相对 random-axis null。E1146 空心点单列，不进入 primary P 值。

输出 PDF、SVG、600-dpi PNG、每 panel source-data CSV、统计 JSON、完整 manifest 和中文 `figures/README.md`。所有图上数字必须由 source-data 自动派生，并完成逐 panel 目视 QA。

## 11. Definition of done

- v0.4 config/core/runner/rollout/analysis/plotter 与测试完成；不覆盖 v0.3。
- 相关单测全绿；解析检查证明 `W_S` 对称、`W_A` 反对称、方向只由 first rank 决定、无 dense bypass。
- development LR/budget audit 只读 interictal validation。
- 102 main + 34 rank-shuffle + 68 split-half units 完成或逐项给出明确失败原因；0 hidden failure。
- 31/34 间期统计、双向 source-excluded rollout 指标和 axis stability 完整。
- ordinary/static/empirical A/B 与 lesions 使用相同 denominator。
- 新 axis/field manifest 先冻结，之后才读取 primary early-ictal target。
- clinical-onset primary、EEG-onset sensitivity、all-contact primary null、within-shaft sensitivity 分层清楚。
- Figure 6 A--F、source data、README、中文白话报告和可复现 manifest 完成。
- 无论阳性或阴性都完成收口；不在 target unseal 后修改模型。
- 代码、文档和结果按逻辑分批 commit；未经用户明确要求不得 push。
