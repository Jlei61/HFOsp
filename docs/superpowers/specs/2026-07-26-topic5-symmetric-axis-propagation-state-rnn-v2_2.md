# Topic 5 / Figure 6：symmetric-axis propagation-state RNN v2.2

**日期**：2026-07-26
**状态**：待执行的冻结科学合同
**替代**：
`2026-07-26-topic5-symmetric-axis-propagation-state-rnn-v2_1.md`

> v2.2 修复 v2.1 的四个阻断问题：development 与 geometry 分母冲突、无效的
> restraint state、事件级 participation 被误用为逐步 hazard bias、以及没有吸收
> STOP 的 soft rollout。除 metadata inventory 发现分母漂移外，执行者不得再修改数学
> 模型、队列或 target-unlock 条件。

## 1. 核心科学问题

> 人类间期 masked contact-rank sequences 能否辨识一个患者特异、近似对称且沿物理
> 病理轴各向异性的有效传播 scaffold；相反传播能否由同一 scaffold 在不同已观察起点
> 下解释；这一纯间期学习到的 scaffold 能否在冻结后预测论文已经定义的
> clinical-onset early-ictal broadband energy field？

科学对象固定为：

```text
shared symmetric effective scaffold
+ event-specific observed source
+ one propagation state
+ one scalar STOP process
```

明确不是：

- 多条离散 path modes；
- 一场事件持续不变的 path identity；
- 两套独立 forward/reverse 主模型；
- 人脑真实 E→E anatomical connectivity；
- 可解释为细胞级 E/I 的 latent states；
- 以 IEI 为基础的新问题；
- 以 ictal-prefix completion 取代论文已有的 early-ictal energy bridge。

Topic 4 SNN 已经承担 forward-model 机制说明。本合同不重新导出 SNN 事件、不恢复 SNN
参数，也没有 SNN gate。

## 2. 队列、切分与 target seal

### 2.1 Geometry-complete development set

固定三位：

- `epilepsiae_1077`
- `epilepsiae_1146`
- `yuquan_chengshuai`

三位均在
`results/topic5_interictal_rank_distribution/dataset_v0_4/subject_audit.csv`
中满足 `geometry_mapped == n_contacts`。选择只依据 2026-07-26 已存在的 metadata：

- 保留原 development 中 geometry-complete 的 `epilepsiae_1146`；
- 用数值 ID 高于 1073 的最近一位 geometry-complete Epilepsiae 患者
  `epilepsiae_1077` 替代 `epilepsiae_1073`；
- 用 Yuquan 按 subject ID 排序的首位 geometry-complete 患者
  `yuquan_chengshuai` 替代 `yuquan_chenziyang`。

选择不读取本模型 score、A/B read-back 或任何 ictal target value。

患者内按事件时间切成：

- 前 60%：参数拟合；
- 中间 20%：objective 选择与 early stopping；
- 最后 20%：冻结前 confirmation。

confirmation 只作过拟合审计；查看后不得修改模型。

### 2.2 Formal cohorts

输入固定为
`results/topic5_interictal_rank_distribution/dataset_v0_4`。按当前 inventory：

- 全部：34 人；
- geometry-complete：25 人；
- development：3 位 geometry-complete；
- development-excluded sequence cohort：31 人；
- development-excluded physical-axis primary cohort：22 人；
- development-excluded geometry-incomplete sequence sensitivity：9 人。

正式阶段的两个 inventory 必须分开：

1. **Physical-axis formal**：22 个 LOSO folds，只运行需要三维物理轴的 full、
   isotropic、random-axis 和 two-\(W\) 分析；
2. **All-subject sequence sensitivity**：31 人，只运行 coordinate-free node-bias、
   Markov 和序列描述，不估计 \(\mathbf u_p\)。

其余患者不构造 latent axial coordinate，也不进入 physical-axis claim。34 人全体结果
只作包含 development cases 的 supportive analysis。

每位正式患者使用 chronological train80 / heldout20。patient-specific 参数、
node bias、source-side thresholds 和 scaffold 只能使用 train80。

若 Milestone A 实际 inventory 与上述数字不符，必须停止并修订 denominator；不得自动
更换 development subjects 或创建 topology fallback。

### 2.3 Target metadata audit 与 primary transfer denominator

建模前对 34 人做只读 metadata audit，只记录：

- clinical-onset anchor；
- clinical-onset contact set；
- early-ictal `1–150 Hz`、`[0,10] s` energy field artifact；
- contact-name exact join；
- dynamic recruitment rank 是否可可靠导出；
- 每个 endpoint 的患者数、seizure 数和排除原因。

不得读取 energy 或 recruitment 数值。primary transfer cohort 冻结为：

\[
\text{development-excluded}
\cap
\text{geometry-complete}
\cap
\text{clinical-onset energy eligible}.
\]

target-value seal 解除前冻结的是 **structural denominator**：

- 每位患者至少 1 次 eligible clinical-onset seizure；
- 每次 seizure 至少 1 个 exact-joined clinical-onset source contact；
- 排除 source 后至少 4 个 exact-joined target contacts。

seal 解除后才按预冻结的 value-QC 检查 target contact field 至少 4 个 finite 且
非恒定值，并形成 **analysis denominator**。这类预注册 value-QC exclusion 允许改变
最终数目；任何其他 attrition 均 hard fail。每位患者至少 2 次 analysis-eligible
seizure 作为 sensitivity，不替代 primary。EEG-onset-only 病例只作独立 sensitivity。

metadata audit 不解除 target seal。只有 §10.5 的四项 interictal 条件全部通过后，
才允许读取 early-ictal energy values。

## 3. 事件输入与因果边界

第 \(e\) 个间期事件为 rank-set 序列：

\[
\mathcal E_e=(S_{e,1},\ldots,S_{e,T_e}).
\]

\(x_t\in\{0,1\}^{N_p}\) 表示当前真实 rank set。第一 rank set 是已经观察到的 source；
不输入 A/B 或 direction label。

模型只接收：

- 当前及过去的 rank sets；
- 当前已参与 mask；
- train partition 固定的 node hazard bias；
- patient geometry。

禁止输入：

- \(t/T_e\)、最终事件长度、最终参与触点数；
- future rank、heldout20 构图信息；
- A/B label、IEI；
- early-ictal target values；
- contact-name 字符串 embedding。

本模型不使用实际 \(\Delta t\)。\(\rho_P\) 只能称为 `rank-step persistence`，不能
解释为毫秒或秒意义的生物学时间常数。

## 4. 唯一有效 scaffold

以下定义只用于 geometry-complete 患者。坐标在患者内中心化。距离尺度 \(s_p\) 固定为
implant geometry 中非零 nearest-neighbour distance 的中位数，不从 rank target
学习。

### 4.1 Local 与 axis kernels

对 \(i\neq j\)：

\[
K^{\mathrm{local}}_{ij}
=
\exp\left[-\frac{\|\mathbf x_i-\mathbf x_j\|^2}{2s_p^2}\right],
\qquad K^{\mathrm{local}}_{ii}=0.
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

\[
K^{\mathrm{axis}}_{ij}
=
\exp\left[
-\frac{d_{\parallel,ij}^2}{2(r s_p)^2}
-\frac{d_{\perp,ij}^2}{2s_p^2}
\right],
\qquad K^{\mathrm{axis}}_{ii}=0,
\]

其中 shared anisotropy ratio \(r\in[1,4]\)；\(r=1\) 为 isotropic。

两个 kernel 分别作 Frobenius normalization：

\[
\bar K=\frac{K}{\|K\|_F+\epsilon}.
\]

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

数值实现必须满足：

\[
W_p=W_p^\top.
\]

禁止 row normalization \(D^{-1}A\)、任意 \(UV^\top\)、eigen truncation、dense
contact mixing 或额外 learnable adjacency。

### 4.3 可辨识参数层级

患者特异参数仅有：

- \(\mathbf u_p\)；
- \(\gamma_p\)；
- \(g_p\)。

shared 参数仅有：

- \(r\)；
- \(\rho_P\)；
- scalar STOP 的 \(c_0,c_P,c_n\)。

约束：

\[
0\le\rho_P<1,\qquad c_P\le0,\qquad c_n\ge0.
\]

local scale \(s_p\) 和 node bias 不学习。readout coefficient 固定为 1，避免与
\(g_p\) 交换。v2.1 的 \(R_t,\rho_R,\kappa\) 全部删除。

## 5. 唯一 recurrent state 与正确的 node hazard bias

每个事件开始时：

\[
P_0=0.
\]

观察当前 rank set \(x_t\) 后：

\[
P_{t+1}
=
\rho_PP_t+W_px_t.
\]

令已出现触点集合为
\(\mathcal V_t=\bigcup_{\tau\le t}S_{e,\tau}\)，未出现集合为
\(U_t=\{1,\ldots,N_p\}\setminus\mathcal V_t\)。contact logits：

\[
z_{i,t+1}
=
b_i+P_{i,t+1}+m_{i,t},
\]

\[
h_{i,t+1}=\sigma(z_{i,t+1}),
\]

其中 \(m_{i,t}=-\infty\) 当 \(i\in\mathcal V_t\)，否则为 0。

不允许 MLP、GRU/LSTM、attention、自由 \(f_P\)、contact-to-contact decoder、
future head 或跨事件 state carry-over。

### 5.1 Node bias 的离散时间 hazard 定义

在每个 train partition 内，包含真实 terminal decision point，计算：

\[
n_i^{\mathrm{next}}
=
\sum_{e,t}
\mathbf 1[i\in S_{e,t+1}],
\]

\[
n_i^{\mathrm{eligible}}
=
\sum_{e,t}
\mathbf 1[i\notin\mathcal V_{e,t}].
\]

terminal decision 的 \(S_{e,T_e+1}=\emptyset\)。然后：

\[
\pi_i^{\mathrm{hazard}}
=
\frac{n_i^{\mathrm{next}}+1}
{n_i^{\mathrm{eligible}}+2},
\qquad
b_i=\operatorname{logit}(\pi_i^{\mathrm{hazard}}).
\]

原来的 event-level participation probability 只作描述性输出，禁止作为 step hazard
intercept。full 与所有 controls 必须共享同一个 node-bias fingerprint。

## 6. Scalar STOP 与 exact one-step likelihood

传播状态更新后，定义：

\[
\bar P_{t+1}
=
\frac{1}{|U_t|}
\sum_{i\in U_t}P_{i,t+1},
\qquad
f_{\mathrm{seen},t}
=
\frac{|\mathcal V_t|}{N_p}.
\]

若 \(U_t=\emptyset\)，强制 \(p_{\mathrm{stop},t+1}=1\)。否则：

\[
p_{\mathrm{stop},t+1}
=
\sigma\left(
c_0+c_P\bar P_{t+1}+c_nf_{\mathrm{seen},t}
\right).
\]

STOP 只能读取 eligible-node mean drive 与 seen fraction；禁止读取 contact identity、
coordinates、A/B 或任何 dense summary。

令：

\[
Z_t
=
1-\prod_{i\in U_t}(1-h_{i,t+1}).
\]

对非空 next set \(S_{t+1}\)：

\[
\Pr(S_{t+1}\mid S_{1:t})
=
(1-p_{\mathrm{stop},t+1})
\frac{
\prod_{i\in U_t}
h_{i,t+1}^{y_i}(1-h_{i,t+1})^{1-y_i}
}{
Z_t
}.
\]

对 terminal empty set：

\[
\Pr(S_{t+1}=\emptyset\mid S_{1:t})
=
p_{\mathrm{stop},t+1}.
\]

实现必须用 log-space 计算 \(Z_t\) 和 likelihood。独立 Bernoulli 的空集不再承担
STOP，因此不会在停止后继续 rollout。

## 7. 吸收 STOP 的 mean-field first-arrival rollout

该 rollout 是 exact one-step model 的明确 mean-field approximation，不得称为完整
joint sequence distribution。

从任意真实 prefix 开始，令：

\[
a^{(1)}=1,
\qquad
s_i^{(0)}=1\quad(i\in U_t).
\]

\(a^{(k)}\) 是第 \(k\) 个未来 decision 前事件仍存活的概率；
\(s_i^{(k-1)}\) 是 mean-field surviving trajectory 中触点 \(i\) 尚未到达的概率。

第 \(k\) 步以 expected seen fraction 和仍未到达节点的加权 mean drive 计算 STOP：

\[
\bar P^{(k)}
=
\frac{
\sum_{i\in U_t}s_i^{(k-1)}P_i^{(k)}
}{
\sum_{i\in U_t}s_i^{(k-1)}+\epsilon
},
\]

\[
f_{\mathrm{seen}}^{(k)}
=
\frac{
|\mathcal V_t|+\sum_{i\in U_t}(1-s_i^{(k-1)})
}{N_p}.
\]

由此按 §6 的受约束公式得到 \(p_{\mathrm{stop}}^{(k)}\)。同时由当前
conditional mean state 得到 contact hazard \(h_i^{(k)}\)。mean-field rollout 保留
§6 对非空 Bernoulli set 的同一归一化：

\[
Z^{(k)}
=
1-\prod_i(1-h_i^{(k)}),
\qquad
\widetilde h_i^{(k)}
=
\frac{h_i^{(k)}}{Z^{(k)}}.
\]

若 \(Z^{(k)}=0\)，该步强制 STOP。否则 surviving trajectory 中的 conditional
first-arrival probability 为：

\[
v_i^{(k)}
=
s_i^{(k-1)}\widetilde h_i^{(k)}.
\]

这是明确的 node-factorized mean-field approximation：\(s_i\) 负责该触点是否仍未
到达，\(Z\) 仍使用同一 conditional-nonempty contact hazard。它不声称恢复 contacts
之间的 joint arrival correlation。非条件 first-arrival mass：

\[
q_i^{(k)}
=
a^{(k)}
(1-p_{\mathrm{stop}}^{(k)})
v_i^{(k)}.
\]

STOP-before-arrival mass：

\[
d_i^{(k)}
=
a^{(k)}
p_{\mathrm{stop}}^{(k)}
s_i^{(k-1)}.
\]

更新：

\[
a^{(k+1)}
=
a^{(k)}(1-p_{\mathrm{stop}}^{(k)}),
\]

\[
s_i^{(k)}
=
s_i^{(k-1)}
(1-\widetilde h_i^{(k)}).
\]

送入下一步 recurrent state 的不是非条件 \(q\)，而是 surviving trajectory 的
conditional mean activation：

\[
x_i^{(k)}
=
v_i^{(k)}.
\]

因此下一步：

\[
P^{(k+1)}
=
\rho_PP^{(k)}+W_px^{(k)}.
\]

每个 rollout 必须通过两类质量守恒测试：

\[
\sum_{k=1}^{H}a^{(k)}p_{\mathrm{stop}}^{(k)}
+a^{(H+1)}
=1,
\]

\[
\sum_{k=1}^{H}q_i^{(k)}
+\sum_{k=1}^{H}d_i^{(k)}
+a^{(H+1)}s_i^{(H)}
=1.
\]

多触点可在同一 rank set 到达，因此 \(\sum_iq_i^{(k)}\) 不要求等于 1。

由同一 rollout 推导：

\[
\Pr(i\text{ future participates within }H)
=
\sum_{k=1}^{H}q_i^{(k)},
\]

\[
\Pr(i\text{ not arrived within }H)
=
1-\sum_{k=1}^{H}q_i^{(k)}
=
\sum_{k=1}^{H}d_i^{(k)}
+a^{(H+1)}s_i^{(H)}.
\]

\[
\Pr(\text{remaining rank}_i=k\mid i\text{ participates})
=
\frac{q_i^{(k)}}{\sum_jq_i^{(j)}+\epsilon},
\]

\[
\widehat A_i(H)
=
\sum_{k=1}^{H}q_i^{(k)}.
\]

不得建立独立 future decoder。

### 7.1 三类 horizon 完全分开

- \(H_{\mathrm{train}}\in\{0,3,5\}\)：只决定是否加入短期 auxiliary rollout loss；
- \(H_{\mathrm{eval}}=|U_t|\)：每个真实 prefix 的正式 future-order 评估上限；
- \(H_{\mathrm{transfer}}=N_p-|S_{\mathrm{source}}|\)：early-ictal source 后的迁移
  rollout 上限。

`not-arrived` 类包含 H 内 STOP-before-arrival 与 cap 后 residual。即使开发选择
`next_only`，\(H_{\mathrm{eval}}\) 与 \(H_{\mathrm{transfer}}\) 仍有唯一预定义。

\(H_{\mathrm{transfer}}\) 是 rank-step completion cap，不对应 clinical onset 后的秒数。
`[0,10] s` 只定义被预测的已冻结 energy target，不定义 RNN rollout 的时间轴。

## 8. 训练目标、训练器与 aggregation

### 8.1 Development 中唯一允许选择的 objective

1. `next_only`：one-step next-set/STOP NLL，\(H_{\mathrm{train}}=0\)；
2. `next_plus_rollout_h3`：next NLL + 权重 1.0 的 3-step first-arrival NLL；
3. `next_plus_rollout_h5`：next NLL + 权重 1.0 的 5-step first-arrival NLL。

first-arrival NLL 包含 \(1,\ldots,H_{\mathrm{train}}\) 和
`not-arrived-within-H_train` 类。

选择规则：

1. 在三位 development 患者中间 20% 上，最大化 full 相对 local-isotropic 的
   patient-median full-\(H_{\mathrm{eval}}\) future first-arrival benefit；
2. benefit 相差不超过 0.5% 时选更简单者：
   `next_only` → `h3` → `h5`；
3. full 的 normalized next-set NLL 不得差于 local-isotropic；
4. 最后 20% confirmation 只审计，不反向选择。

固定训练器：

- AdamW；
- learning rate `1e-3`；
- weight decay `1e-4`；
- gradient clipping `1.0`；
- maximum 200 epochs；
- patience 15；
- seeds：`17, 29, 43`。

开发结束写 `DEVELOPMENT_LOCK.json`，冻结 selected objective、
\(H_{\mathrm{train}}\)、参数边界、训练器、inputs 与代码 commit。

### 8.2 Event-first loss 与正式 metric

对每个 decision point，主要 log score 为：

\[
\ell_{e,t}
=
\frac{-\log\Pr(S_{e,t+1}\mid S_{e,1:t})}
{\max(1,|U_{e,t}|)}.
\]

训练和评估均按：

```text
eligible contacts normalization
→ within-event decision mean
→ patient-seed event mean
→ seed median within patient
→ patient-level cohort inference
```

future first-arrival metric 同样先在 eligible contacts 内平均，再做 prefix →
event → patient-seed → patient。prefix 和 event 不能作为 cohort-level 独立样本。

### 8.3 Formal LOSO fitting

22 个 physical-axis folds 中，每折严格执行：

1. shared \(r,\rho_P,c_0,c_P,c_n\) 只在其余 21 位 physical-axis formal 患者的
   train80 上联合拟合；
2. shared 参数冻结后，只在 heldout 患者的 train80 上估计
   \(\mathbf u_p,\gamma_p,g_p\)；
3. heldout20 只评估；
4. 三位 development subjects 不进入 shared training；
5. 每个 control 独立拟合其允许参数，但复用同一 split、bias、seed 和 aggregation。

这是一种“跨患者共享动力学 + 患者内 train-only scaffold identification”，不是对
完全没有间期数据的新患者作零样本预测。

## 9. Controls 的精确合同

所有 controls 复用相同 split、node bias、terminal labels、event-first aggregation 与
source/contact denominator。

### 9.1 `node_bias_no_history`

\[
h_i=\sigma(b_i).
\]

STOP 使用：

\[
p_{\mathrm{stop}}
=
\sigma(c_0+c_nf_{\mathrm{seen}}),
\]

不含传播状态。

### 9.2 `source_distance_only`

只使用第一 rank set \(S_{e,1}\)。定义：

\[
\delta_i
=
\frac{
\min_{j\in S_{e,1}}\|\mathbf x_i-\mathbf x_j\|
}{s_p},
\]

\[
z_i=b_i-\alpha_{d,p}\delta_i+m_i,
\qquad \alpha_{d,p}\ge0.
\]

\(\alpha_{d,p}\) 只从患者 train partition 拟合，整个事件不随 prefix 更新。STOP 与
node-bias control 相同。

### 9.3 `local_isotropic`

固定 \(\gamma_p=0\)，其余 \(P\)、STOP、bias 和训练代码与 full 完全相同。

### 9.4 `empirical_first_order_markov`

对 train partition 的相邻 rank sets，按 tie size 加权：

\[
C_{ji}
=
\sum_{e,t}
\frac{
\mathbf 1[j\in S_{e,t},\,i\in S_{e,t+1}]
}{|S_{e,t}|},
\]

\[
E_{ji}
=
\sum_{e,t}
\frac{
\mathbf 1[j\in S_{e,t},\,i\notin\mathcal V_{e,t}]
}{|S_{e,t}|}.
\]

使用 concentration \(\lambda=10\)、以 node hazard 为中心的 Beta smoothing
（即二分类 Dirichlet）：

\[
T_{ji}
=
\frac{C_{ji}+\lambda\pi_i^{\mathrm{hazard}}}
{E_{ji}+\lambda}.
\]

unseen edge 因而回退到 \(\pi_i^{\mathrm{hazard}}\)。当前 rank set 为 tie 时：

\[
h_i
=
\frac{1}{|S_t|}
\sum_{j\in S_t}T_{ji}.
\]

随后 mask 已出现触点。STOP 与 node-bias control 相同。Markov 不进入 physical-axis
primary gate，只作 all-subject sequence sensitivity。

### 9.5 `two_direction_operator`

先用全 train80 学一个共同 \(\mathbf u_p\)。source-left 与 source-right 只分别拟合：

\[
(\gamma_{p,L},g_{p,L}),
\qquad
(\gamma_{p,R},g_{p,R}).
\]

\(\mathbf u_p\)、shared \(r,\rho_P,c_0,c_P,c_n\)、node bias 和 decoder 保持共同。
禁止分别拟合 \(\mathbf u_{p,L},\mathbf u_{p,R}\) 或增加 direction-specific hidden
state。它只检验同一物理轴上是否还需要 direction-specific operator strength。

## 10. 分层 claims 与 target unlock

### 10.1 Claim 1：间期顺序可预测

\[
\text{full}>\text{node bias}.
\]

endpoint：heldout normalized next-set NLL。它是 sanity/replication，不是新版主要
结论。

### 10.2 Claim 2：物理轴向结构提供额外信息

22 人 physical-axis primary cohort 比较：

\[
\text{symmetric-axis full}>\text{local isotropic}.
\]

两个 endpoint：

- heldout normalized next-set NLL；
- full-\(H_{\mathrm{eval}}\) prefix-conditioned future first-arrival NLL。

每位患者先取 seed median；one-sided patient-level Wilcoxon；两个 endpoint 内
BH-FDR。每项 PASS 要求 median benefit \(>0\)、改善患者过半、\(q<0.05\)。

### 10.3 Claim 3：learned axis 不是任意方向

唯一阻断项是 learned-axis random-direction specificity。用固定 null seed
`20260726` 为每位患者预生成 256 个均匀球面随机方向，分别固定 \(\mathbf u\)，只在
train80 重新估计 \(\gamma_p,g_p\)，heldout20 评估。这里的 NLL 唯一指
event-first normalized next-set NLL。定义：

\[
\Delta_{\mathrm{random},p}
=
\operatorname{median}_{j}
\left(
\mathrm{NLL}_{p,\mathrm{random},j}
\right)
-\mathrm{NLL}_{p,\mathrm{learned}}.
\]

PASS 要求 patient median \(>0\)、正值患者过半、one-sided Wilcoxon \(p<0.05\)。

以下独立报告但不阻断：

- train80 split-half sign-invariant axis cosine；
- 256 个 shaft-preserving coordinate permutations；
- contact-cloud PCA1 fixed-axis sensitivity。

### 10.4 Claim 4：一个 scaffold 跨两个 source sides 泛化

事件 source centroid：

\[
s_e^{\mathrm{source}}
=
\frac{1}{|S_{e,1}|}
\sum_{i\in S_{e,1}}
\mathbf u_p^\top(\mathbf x_i-\bar{\mathbf x}_p).
\]

只用 train80 的 source projections 冻结：

\[
Q_{0.25,p},\qquad Q_{0.75,p}.
\]

分类：

- \(s_e^{\mathrm{source}}\le Q_{0.25,p}\)：source-left；
- \(s_e^{\mathrm{source}}\ge Q_{0.75,p}\)：source-right；
- 中间 50% 不进入 Claim 4。

heldout20 直接复用 train80 阈值。若两个分位数相同则该患者
`not_estimable`。事件数门槛在排除中间事件后计算：

- train80 每侧至少 100；
- heldout20 每侧至少 25；
- eligible 患者不少于 22 人的一半，否则 cohort `not_estimable`。

三个检验：

1. 同一个冻结 \(W_p\) 在 heldout left/right 两侧均优于 local-isotropic；
2. `two_direction_operator` 相对 shared \(W\) 没有实质增益；
3. cross-side transfer：只用一侧 train events 估计
   \(\mathbf u_p,\gamma_p,g_p\)，测试另一侧，两个方向分别报告。

检验 1 的 Claim-4 PASS 要求两侧 normalized next-set benefit 分别满足 cohort median
\(>0\)、改善患者过半、BH-FDR \(q<0.05\)。

检验 2 定义：

\[
\Delta_{\mathrm{two},p}
=
\mathrm{NLL}_{\mathrm{shared},p}
-\mathrm{NLL}_{\mathrm{twoW},p},
\]

\[
\Delta_{\mathrm{axis},p}
=
\mathrm{NLL}_{\mathrm{isotropic},p}
-\mathrm{NLL}_{\mathrm{shared},p},
\]

\[
M_p
=
\Delta_{\mathrm{two},p}
-0.10\Delta_{\mathrm{axis},p}.
\]

对患者 paired bootstrap，要求：

\[
\operatorname{upperCI}_{95\%}
\left[\operatorname{median}(M_p)\right]<0.
\]

否则 direction-specific operator 存在实质 heldout 增益，shared-scaffold claim
不通过。future-order 和 cross-side transfer 为加强证据，不单独阻断。

### 10.5 Target-unlock 条件

没有覆盖所有结果的全局科学 boolean。early-ictal value seal 只在以下四项全部通过时
解除：

- Claim 2 next-set PASS；
- Claim 2 future first-arrival PASS；
- Claim 3 random-axis specificity PASS；
- Claim 4 shared-scaffold PASS。

Claim 1、split-half、shaft-permutation、PCA1 和 A/B read-back 独立报告。

## 11. 冻结后的 A/B 病理轴 read-back

A/B 不进入训练、objective selection、axis fitting 或 target unlock。模型和 formal
scores 完全冻结后，读取：

`results/interictal_propagation_masked/template_gradient_fields/per_subject/<subject>.json`

中的：

```text
axis_definition = template_propagation_axis_v2
axis_pair.shared_axis.u
```

只在 `axis_pair.shared_axis.status == ok` 且 contact exact join 成功时计算：

\[
\left|
\widehat{\mathbf u}^{\mathrm{RNN}}_p
\cdot
\mathbf u^{A/B}_p
\right|,
\]

以及 sign-invariant contact projection Spearman：

\[
\left|
\rho\left(
\widehat{\mathbf u}^{\mathrm{RNN}\top}
(\mathbf x_i-\bar{\mathbf x}),
\mathbf u^{A/B\top}
(\mathbf x_i-\bar{\mathbf x})
\right)
\right|.
\]

同时相对同一患者 256 个 frozen random directions 报告 empirical null percentile。
它是连接论文前半部分的 secondary read-back，不是训练标签，也不是硬 gate。

## 12. Frozen early-ictal transfer

### 12.1 Primary energy-field bridge

只对 §2.3 冻结的 primary transfer cohort：

1. 使用原始 binary clinical-onset contact vector 作为 observed source；
2. 不根据 energy target 截取、扩张或重排 source；
3. 加载每位患者 formal LOSO fold 冻结的 shared dynamics、\(W_p\) 和 \(b_i\)；
4. 令 \(P_0=0\)，以 source vector 执行第一次 state update；
5. 使用 \(H_{\mathrm{transfer}}=N_p-|S_{\mathrm{source}}|\) 的 §7 rollout；
6. 得到 \(\widehat A_i=\sum_kq_i^{(k)}\)；
7. 在非 source contacts 上预测 clinical-onset `[0,10] s`、`1–150 Hz`
   baseline-robust-z contact energy ordering。

source-size-normalized：

\[
x_{\mathrm{source}}
\rightarrow
\frac{x_{\mathrm{source}}}{|S_{\mathrm{source}}|}
\]

只作预注册 sensitivity。

每次 seizure 计算 Spearman；seizure 先在患者内取中位数，再做 patient-level inference。
target field 恒定或 finite contacts 少于 4 时按预冻结 value-QC 规则排除。若某个模型
的 prediction 恒定，该模型该 seizure 的 Spearman 预先记为 0，不改变共同 denominator。

固定比较：

- full；
- local-isotropic；
- source-distance-only；
- node-bias。

主问题：

\[
\text{frozen interictal symmetric-axis rollout}
>
\text{frozen local-isotropic rollout}
\]

是否更能解释 early-ictal spatial energy ordering。

primary inference 为 one-sided paired patient-level Wilcoxon。PASS 要求
full-minus-isotropic 的 patient median \(>0\)、改善患者过半且 \(p<0.05\)；同时报告
patient bootstrap 95% CI。这里只有一个 primary transfer endpoint，不再额外 FDR。

### 12.2 Secondary dynamic recruitment

只有 metadata audit 证明 recruitment-rank extractor 和分母可靠时，才测试：

- earliest ictal prefix → later participation；
- earliest ictal prefix → later recruitment rank。

它不取代 energy-field primary，不用于修改 interictal model。

### 12.3 解释边界

允许写：

- frozen interictal effective scaffold predicts/explains early-ictal contact-energy
  ordering；
- 同一有效传播 scaffold 与发作早期能量场相容。

禁止写：

- prospective seizure warning；
- ictal-prefix completion 等于发作预测；
- \(W_p\) 是真实 anatomical E→E matrix；
- \(P_t\) 是细胞级 excitation；
- rank-step rollout 与 `[0,10] s` 存在真实时间一一映射；
- EEG-onset sensitivity 代替 clinical-onset primary。

## 13. Figure 6 六块合同

| Panel | 科学含义 |
|---|---|
| A | 同一近对称 effective scaffold，不同 observed source 产生相反方向 |
| B | 单一 propagation state、scalar STOP、self-supervised next-set 与 absorbing rollout |
| C | 22 人 physical-axis primary 中 full vs isotropic 的 heldout next/future benefit |
| D | learned axis 的 random-null specificity、稳定性及冻结后的 A/B axis read-back |
| E | train-only source quantiles 下同一 \(W\) 的 cross-direction generalization |
| F | frozen interictal rollout 对 clinical-onset early-ictal energy field 的迁移 |

若 target 未解封，Panel F 明确写 `target sealed`，不得用旧 SNN 或 dynamic-rank 图填补。

## 14. 停止规则

- metadata denominator 漂移：停止并修订合同；
- 任一 development subject 不再 geometry-complete：停止，不自动换人；
- confirmation 明显反向：报告 instability，停止，不扩 grid；
- Claim 2、Claim 3 random specificity 或 Claim 4 未满足：保持 target sealed；
- early-ictal primary 阴性：按阴性报告，不用 secondary rescue；
- 禁止增加 SNN benchmark、GRU upper bound、hidden-size sweep、额外 head、更多
  development cases 或看过 target 后改 source/horizon。
