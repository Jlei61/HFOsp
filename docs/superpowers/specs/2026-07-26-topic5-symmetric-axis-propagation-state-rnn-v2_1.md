# Topic 5 / Figure 6：symmetric-axis propagation-state RNN v2.1

**日期**：2026-07-26
**状态**：**SUPERSEDED；禁止执行**
**替代合同**：
`2026-07-26-topic5-symmetric-axis-propagation-state-rnn-v2_2.md`

> v2.1 的 development/geometry 分母、restraint state、node-bias 概率尺度和
> non-absorbing rollout 存在阻断问题。以下内容只保留审计，不得据此启动训练。

**替代**：
`2026-07-26-topic5-symmetric-axis-ei-system-identification-v2_0.md`
**开发路线**：采用三位固定 development subjects；31 位 development-excluded
患者为 primary，34 位全体只作 supportive analysis。

## 1. 核心科学问题

> 人类间期 masked contact-rank sequences 能否辨识一个患者特异、近似对称且沿物理
> 病理轴各向异性的有效传播 scaffold；相反传播能否由同一 scaffold 在不同已观察起点
> 下解释；这一纯间期学习到的 scaffold 能否在冻结后预测论文已经定义的
> clinical-onset early-ictal broadband energy field？

科学对象固定为：

```text
shared symmetric effective scaffold
+ event-specific observed source
+ propagation drive / refractory-restraint traces
```

明确不是：

- 多条离散 path modes；
- 一场事件持续不变的 path identity；
- 两套独立 forward/reverse 主模型；
- 人脑真实 E→E anatomical connectivity；
- 可被解释为细胞级 E/I 的 latent states。

Topic 4 SNN 已经承担 forward-model 机制说明，本合同不重新导出 SNN 事件、不恢复 SNN
参数，也没有 SNN gate。

## 2. 队列、切分与 target seal

### 2.1 Development set

固定三位：

- `epilepsiae_1073`
- `epilepsiae_1146`
- `yuquan_chenziyang`

这三位用于有限的模型开发和优化选择，之后永久排除于 primary cohort。开发患者内按时间
切成：

- 前 60%：参数拟合；
- 中间 20%：候选选择与 early stopping；
- 最后 20%：冻结前 confirmation。

confirmation 结果只作过拟合审计；不得在查看后再次修改模型。

### 2.2 Formal interictal cohorts

输入固定为
`results/topic5_interictal_rank_distribution/dataset_v0_4`：

- 31 位 development-excluded 患者：formal sequence cohort；
- 其中预计 24 位满足
  `geometry_mapped == n_contacts`：physical-axis primary cohort；
- 其余患者不构造 latent axial coordinate，也不进入 physical-axis claim；
- 34 位全体分析只作 supportive，必须明确包含 3 位开发患者。

正式阶段每位患者使用 chronological train80 / heldout20。所有 patient-specific
参数、node bias 和 scaffold 只能使用 train80。

### 2.3 Target metadata audit

在建模前对全部 34 位患者做只读 metadata audit，记录但不读取 target 数值：

- clinical-onset 时间是否存在；
- clinical-onset contact set 是否存在；
- early-ictal `1–150 Hz` energy field 是否可用；
- contact-name exact join 是否完整；
- dynamic recruitment rank 是否可可靠导出；
- 每个 endpoint 的患者数、seizure 数和排除原因。

metadata audit 不解除 target seal。只有 §8 Claim 2 和 Claim 4 通过后，才允许读取
early-ictal energy values。

## 3. 事件输入与因果边界

第 \(e\) 个间期事件为 rank-set 序列：

\[
\mathcal E_e=(S_{e,1},\ldots,S_{e,T_e}),
\]

令 \(x_t\in\{0,1\}^{N_p}\) 表示当前真实 rank set。第一 rank set 是该事件已经观察到的
source；不引入 A/B 或 direction label。

模型只接收：

- 当前及过去的 rank sets；
- 当前已参与 mask；
- train partition 固定的 node bias；
- patient geometry。

禁止输入：

- \(t/T_e\)、最终事件长度、最终参与触点数；
- future rank、heldout20 构图信息；
- A/B label、IEI；
- early-ictal target values；
- contact-name 字符串 embedding。

本模型不使用实际 \(\Delta t\)。所有 persistence 参数只能称为
`rank-step persistence`，不能解释为毫秒或秒意义的生物学时间常数。

## 4. 唯一有效 scaffold

以下定义只用于 geometry-complete 患者。坐标在患者内中心化；距离尺度
\(s_p\) 固定为 implant geometry 中非零 nearest-neighbour distance 的中位数，不从
rank target 学习。

### 4.1 Local 与 axis kernels

对 \(i\neq j\)：

\[
K^{\mathrm{local}}_{ij}
=
\exp\left[-\frac{\|\mathbf x_i-\mathbf x_j\|^2}{2s_p^2}\right],
\qquad K_{ii}=0.
\]

患者物理轴为单位向量 \(\mathbf u_p\)，且
\(\mathbf u_p\equiv-\mathbf u_p\)。令：

\[
d_{\parallel,ij}
=
\left|\mathbf u_p^\top(\mathbf x_i-\mathbf x_j)\right|,
\]

\[
d_{\perp,ij}^2
=
\|\mathbf x_i-\mathbf x_j\|^2-d_{\parallel,ij}^2.
\]

axis kernel：

\[
K^{\mathrm{axis}}_{ij}
=
\exp\left[
-\frac{d_{\parallel,ij}^2}{2(r s_p)^2}
-\frac{d_{\perp,ij}^2}{2s_p^2}
\right],
\qquad K_{ii}=0,
\]

其中 shared anisotropy ratio \(r\in[1,4]\)；\(r=1\) 为 isotropic。

两个 kernel 分别作 Frobenius normalization：

\[
\bar K
=
\frac{K}{\|K\|_F+\epsilon}.
\]

随后：

\[
A_p
=
(1-\gamma_p)\bar K^{\mathrm{local}}_p
+\gamma_p\bar K^{\mathrm{axis}}_p,
\qquad 0\le\gamma_p\le1.
\]

### 4.2 保对称的传播算子

令 \(D_p=\operatorname{diag}(A_p\mathbf 1)\)。唯一允许的 normalization 是：

\[
W_p
=
g_pD_p^{-1/2}A_pD_p^{-1/2},
\qquad g_p>0.
\]

因此数值实现必须满足：

\[
W_p=W_p^\top.
\]

禁止 row normalization \(D^{-1}A\)、任意 \(UV^\top\)、eigen truncation、dense
contact mixing 或额外 learnable adjacency。

### 4.3 参数层级

患者特异参数仅有：

- 物理轴 \(\mathbf u_p\)；
- axis mixing \(\gamma_p\)；
- propagation gain \(g_p\)。

shared 参数：

- anisotropy ratio \(r\)；
- 两个 rank-step persistence；
- restraint strength。

local scale \(s_p\) 和 node bias 不学习。excitation readout coefficient 固定为 1，
避免 \(g_p\) 与额外 \(\beta_E\) 相互补偿。

## 5. 精确 recurrent state

状态名称固定为：

- \(P_t\)：propagation drive；
- \(R_t\)：refractory/restraint trace。

每个事件开始时：

\[
P_0=0,\qquad R_0=0.
\]

对真实或 rollout rank set \(x_t\)：

\[
P_{t+1}
=
\rho_P P_t+W_px_t,
\]

\[
R_{t+1}
=
\rho_R R_t+x_t,
\]

\[
z_{i,t+1}
=
b_i+P_{i,t+1}-\kappa R_{i,t+1}+m_{i,t},
\]

\[
h_{i,t+1}
=
\sigma(z_{i,t+1}).
\]

约束：

\[
0\le\rho_P\le\rho_R<1,\qquad \kappa>0.
\]

\(m_{i,t}= -\infty\) 表示已参与触点，确保每个触点最多首次到达一次。

实现中不允许：

- MLP、GRU/LSTM cell 或 attention；
- 自由 \(f_P/f_R\)；
- contact-to-contact decoder；
- 额外 future head；
- 跨事件 state carry-over。

node bias 使用 train partition 的 Laplace-smoothed participation：

\[
\pi_i=\frac{c_i+1}{n+2},
\qquad b_i=\operatorname{logit}(\pi_i),
\]

并在 full 与所有 controls 中共享同一 fingerprint。

## 6. 统一 hazard 与所有输出

### 6.1 Next-set / STOP likelihood

给定未参与节点集合 \(U_t\)，下一 rank set \(S_{t+1}\) 的概率为：

\[
p(S_{t+1}\mid S_{1:t})
=
\prod_{i\in U_t}
h_{i,t+1}^{y_i}(1-h_{i,t+1})^{1-y_i},
\]

其中 \(y_i=1\) 当且仅当 \(i\in S_{t+1}\)。最终 STOP 是所有
\(y_i=0\) 的空集，不增加独立 stop head。

### 6.2 Prefix-conditioned rollout

从任意真实 prefix 后开始 deterministic soft rollout。令初始 survival
\(s_i^{(0)}=1\)。第 \(k\) 步 first-arrival mass：

\[
q_i^{(k)}
=
s_i^{(k-1)}h_i^{(k)},
\]

\[
s_i^{(k)}
=
s_i^{(k-1)}(1-h_i^{(k)}).
\]

下一步状态输入固定为：

\[
x^{(k)}=q^{(k)}.
\]

由同一 hazard 唯一推导：

\[
P(i\text{ future participates})
=
\sum_{k=1}^{H}q_i^{(k)}
=
1-s_i^{(H)},
\]

\[
P(\text{remaining rank}_i=k\mid i\text{ participates})
=
\frac{q_i^{(k)}}{\sum_j q_i^{(j)}+\epsilon},
\]

\[
\widehat A_i
=
\sum_{k=1}^{H}q_i^{(k)}.
\]

next-set、future participation、remaining rank 和跨状态 cumulative activation
全部来自这一 rollout；不得建立独立 decoder。

## 7. Development contract

### 7.1 唯一允许选择的候选

精确模型结构、kernel、state update 和 controls 不参与选择。development set 只在以下
三个训练目标中选一个：

1. `next_only`：完整事件 next-set/STOP NLL；
2. `next_plus_rollout_h3`：next NLL + 权重 1.0 的 3-step first-arrival NLL；
3. `next_plus_rollout_h5`：next NLL + 权重 1.0 的 5-step first-arrival NLL。

first-arrival NLL 包含 \(1,\ldots,H\) 和 `not-arrived-within-H` 类，直接使用 §6 的
\(q\) 与 survival。

选择规则：

1. 在三位患者中间 20% 上，最大化 full 相对 local-isotropic 的
   patient-median future first-arrival NLL benefit；
2. 候选 benefit 相差不超过 0.5% 时，选择更简单者：
   `next_only` → `h3` → `h5`；
3. full 的 next-set NLL 不得比 local-isotropic 更差；否则该候选不可选；
4. 最后 20% confirmation 只审计，不再反向选择。

固定训练器：

- AdamW；
- learning rate `1e-3`；
- weight decay `1e-4`；
- gradient clipping `1.0`；
- maximum 200 epochs；
- patience 15；
- 三个固定 seeds。

开发结束必须写 `DEVELOPMENT_LOCK.json`，冻结 objective、\(H\)、参数边界、训练器、
input fingerprints 和代码 commit。之后不能根据 31 人结果修改。

### 7.2 Formal LOSO fitting

31 人 primary 中每个 heldout fold：

1. shared \(r,\rho_P,\rho_R,\kappa\) 只在其余 30 位 primary 患者的 train80 上拟合；
2. shared 参数冻结后，只在 heldout 患者 train80 上估计
   \(\mathbf u_p,\gamma_p,g_p\)；
3. heldout20 只评估；
4. 三位 development subjects 不进入 formal shared training；
5. 每个 nested control 独立拟合允许存在的参数，但必须复用相同 split、node bias、
   optimizer coverage 和 seed。

这是一种“跨患者共享动力学 + 患者内 train-only scaffold identification”，不是对完全
没有间期数据的新患者作零样本预测。

## 8. Controls 与分层 claims

### 8.1 Controls

1. `node_bias_no_history`：仅固定 \(b_i\)；
2. `source_distance_only`：\(b_i\) + 节点到 observed source set 的最小欧氏距离，
   无 recurrence；
3. `local_isotropic`：\(\gamma_p=0\)，其余 state 和 node bias 与 full 相同；
4. `empirical_first_order_markov`：train80 患者 transition graph，简单基准；
5. `symmetric_axis_full`：目标模型；
6. `two_direction_operator`：source-side 分开拟合两套对称 \(W\)，仅作共享性
   sensitivity。

不运行 dense GRU upper bound。

### 8.2 Claim 1：间期顺序可预测

问题：

\[
\text{full}>\text{node bias}.
\]

endpoint：heldout next-set NLL。它是 sanity/replication，不是新版最主要结论。

### 8.3 Claim 2：物理轴向结构提供额外信息

physical-axis primary cohort 中比较：

\[
\text{symmetric-axis full}>\text{local isotropic}.
\]

两个预注册 endpoint 独立报告：

- heldout next-set NLL；
- prefix-conditioned future first-arrival NLL。

每位患者先取 seed median；one-sided patient-level Wilcoxon；两 endpoint 内
BH-FDR。每个 endpoint 的 pass 分别要求 median benefit \(>0\)、改善患者过半、
\(q<0.05\)。不得压缩为一个覆盖所有 claims 的全局 boolean。

### 8.4 Claim 3：结果不是任意轴或 implant geometry

独立报告：

- train80 split-half 的 sign-invariant axis cosine；
- learned-axis heldout score 相对 256 个 random 3D directions；
- 相对 256 个 shaft-preserving coordinate permutations；
- contact-cloud PCA1 fixed-axis sensitivity。

每个 random/null orientation 固定 shared dynamics，只用 train80 重新估计
\(\gamma_p,g_p\)，再评估 heldout20。heldout20 不参与 axis 或 null 选择。

### 8.5 Claim 4：一个 scaffold 可跨两个 source sides 泛化

source side 只由 train80 learned axis 与第一 rank set 投影定义。eligibility：

- train80 每侧至少 100 个事件；
- heldout20 每侧至少 25 个事件；
- cohort eligible 患者不少于 physical-axis primary cohort 的一半；否则结论为
  `not_estimable`。

三个检验：

1. 同一个冻结 \(W_p\) 在 heldout source-left 和 source-right 两侧都优于
   local-isotropic；
2. `two_direction_operator` 相对 shared \(W\) 的 heldout NLL 增益，不得超过
   development 阶段冻结的 10% `full-vs-isotropic` benefit margin；
3. cross-side transfer：只用一侧 train events 估计 patient-specific
   \(\mathbf u_p,\gamma_p,g_p\)，冻结后测试另一侧；左右两方向分别报告。

检验 1 的 next-set 与 future first-arrival 分开报告；seed 先在患者内取中位数，
left/right 四项比较内 BH-FDR。Claim 4 的必要部分固定为两侧 next-set benefit 均满足
cohort median \(>0\)、改善患者过半、\(q<0.05\)。

检验 2 定义：

\[
\Delta_{\mathrm{two}}
=
\mathrm{NLL}_{\mathrm{shared}}-\mathrm{NLL}_{\mathrm{twoW}},
\qquad
\Delta_{\mathrm{axis}}
=
\mathrm{NLL}_{\mathrm{isotropic}}-\mathrm{NLL}_{\mathrm{shared}}.
\]

对患者 bootstrap 的
\(\operatorname{median}(\Delta_{\mathrm{two}})\) 95% upper bound 必须不超过
\(0.10\times\operatorname{median}(\Delta_{\mathrm{axis}})\)。否则 two-\(W\) 存在实质
heldout 增益，shared-scaffold claim 不通过。

Claim 4 pass 需要检验 1 的两侧 next-set 均通过，并且检验 2 不支持 two-\(W\) 的实质
增益。两侧 future-order 与 cross-side transfer 为加强证据，独立报告，不单独阻断。

### 8.6 All-subject sequence sensitivity

31 位 development-excluded 患者全部进入：

- node-bias next-set；
- empirical first-order Markov；
- participation/future-order 描述。

geometry-incomplete 患者不运行 latent-axis fallback，不与 physical-axis cohort
pool，也不得写成恢复了物理病理轴。34 人结果只作 supportive。

### 8.7 进入跨状态迁移的条件

没有全局 `formal_gate_pass`。只有：

- Claim 2 的两个 primary endpoints 均通过；
- Claim 4 通过；

才解除 early-ictal target value seal。Claim 1 和 Claim 3 独立报告；secondary
endpoint 的阴性不能取消已通过的独立 claim。

## 9. Frozen early-ictal transfer

### 9.1 Primary：existing energy-field bridge

对 metadata-eligible seizures：

1. 使用 clinical-onset contact set 作为 observed source；source 不能由 energy
   target 的 top contacts 反推；
2. 冻结 interictal \(W_p\)、shared dynamics、\(b_i\) 和 development-selected
   horizon；
3. 由 §6 rollout 得到 cumulative activation field \(\widehat A_i\)；
4. 预测论文当前已冻结的 clinical-onset `[0,10] s`、`1–150 Hz`
   baseline-robust-z contact energy field。

primary scoring：

- 在非 source contacts 上计算每次 seizure 的 Spearman correlation；
- seizure 先在患者内取中位数，再做 patient-level inference；
- full 与 local-isotropic、source-distance-only、node-bias 共用完全相同的 source、
  contact join、node bias 和 seizure denominator；
- all-contact scoring 作为 sensitivity。

主问题：

\[
\text{frozen interictal symmetric-axis rollout}
>
\text{frozen local-isotropic rollout}
\]

是否更能解释 early-ictal spatial energy ordering。

Epilepsiae clinical onset 为 primary anchor。EEG-onset-only 病例单独 sensitivity，
永不与 clinical-onset pool 合并。

### 9.2 Secondary：dynamic recruitment

只有 metadata audit 证明 recruitment-rank extractor 与分母可靠时，才测试：

- earliest ictal prefix → later participation；
- earliest ictal prefix → later recruitment rank。

它不取代 energy-field primary，不用于修改 interictal model。

### 9.3 解释边界

允许写：

- frozen interictal effective scaffold predicts/explains early-ictal contact-energy
  ordering；
- 同一有效传播 scaffold 与发作早期能量场相容。

禁止写：

- prospective seizure warning；
- 已经看到发作 prefix 后的 completion 等于发作预测；
- \(W_p\) 是真实 anatomical E→E matrix；
- \(P/R\) 是患者细胞级 excitation/inhibition；
- EEG-onset sensitivity 代替 clinical-onset primary。

## 10. Figure 6 六块合同

| Panel | 科学含义 |
|---|---|
| A | 同一近对称 effective scaffold，不同 observed source 产生相反方向 |
| B | 精确 recurrent equations、self-supervised next-set task 和统一 hazard rollout |
| C | physical-axis primary cohort 中 full vs local-isotropic 的 heldout next/future benefit |
| D | learned physical axis 的 split-half stability、random-axis 与 sampling-geometry null |
| E | 同一 \(W\) 在两个 source sides 上的 heldout/cross-direction generalization |
| F | frozen interictal rollout 对 existing clinical-onset early-ictal energy field 的迁移 |

SNN 机制图继续留在 Topic 4/Figure 4，不在 Figure 6 重做参数恢复。

## 11. 停止规则

- 不再执行 v2.0 SNN benchmark 或任何 SNN recovery gate。
- development 候选只限 §7 三项；不得扩展 hidden size、kernel、rank 或 loss grid。
- confirmation 后必须冻结；31 人结果不得反向改合同。
- geometry metadata 与预期分母不一致时先停下修订 denominator，不创建 latent-axis
  fallback。
- Claim 2 或 Claim 4 未通过时，early-ictal target values 保持封存。
- target metadata audit 若显示 energy-field bridge 不可执行，先报告
  `BLOCKED_INPUT`，不得用 dynamic recruitment 取代 primary。
