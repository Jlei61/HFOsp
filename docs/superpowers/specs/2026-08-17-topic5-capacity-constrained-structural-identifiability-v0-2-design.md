# Topic 5.2D 容量约束的有序历史空间子空间与共享动力学可辨识性 v0.2：设计草案

> 状态：**AUTHORIZED FOR FORMAL EXECUTION（用户于 2026-08-17 正式授权；科学内容冻结，不再修订）**  
> 日期：2026-08-17  
> 修订依据：2026-08-17 科学审阅；已分离 predictive subspace 与 autonomous operator，并加入 bypass interaction、orderless/shaft/angle controls 和 fractional matrix。  
> Parent：`dynamical_motif_rnn_v0_1`、其 v0.2 修复，以及 `ecog_physical_neighborhood_rnn_v0_1`  
> 候选结果根：`results/topic5_capacity_constrained_history_motif_v0_2/`  
> 本文件不改写、覆盖或重新裁定既有 v0.1/v0.2 结果。

## 0. 决策摘要

### 0.1 是否适合作为下一步

适合，但它不是把上一轮阴性“救成阳性”的实验。它要区分两种仍未分开的解释：

\[
\boxed{
\text{提出的空间结构没有提供信息}
}
\]

与

\[
\boxed{
\text{空间结构有用，但被高容量通用历史通路和自由输出吸收}
}
\]

上一轮已经说明：真实事件顺序有预测价值；冻结 RNN 的局部递归在该 RNN 内部确实被使用；但是一个容量受控、没有 tissue propagation 的历史模型可以匹配下一触点预测，而新增走廊、方向推力和前馈接力没有提供 held-out 增量。这意味着当前 next-rank 任务没有唯一识别空间传播定律。

本阶段因此不再给原 full-tissue RNN 增加更多 motif，而把所有**有序历史**限制到一个低维、可交换、可审计的唯一通路中，并明确拆成两个子模块：

```text
5.2D1  PREDICTIVE_SUBSPACE_IDENTIFIABILITY
       低维 prefix state 能否直接解码多步未来？

5.2D2  AUTONOMOUS_SHARED_OPERATOR_IDENTIFIABILITY
       同一个低维算子能否自主推进并生成多步未来？
```

只有 D2 得到支持时，正文才使用“dynamical motif”或“shared propagation operator”。D1 阳性只能称为 predictive subspace 或低维预测坐标。

本阶段直接测量：

1. 患者训练序列与记录布局共同定义的结构，是否提高低容量模型的学习效率；
2. 这种结构只提供静态 suffix 字典，还是确实需要 rank 顺序；
3. 直接多步预测能否进一步收缩为同一个自主共享算子；
4. 结构优势是否在无序 contact-set 旁路减弱后增强；
5. 训练完成后，模型是否实际使用 rank 顺序和低维 ordered path；
6. 这种优势是否在训练样本减少或状态维数降低时更明显；
7. 部分电极覆盖是否足以让一个本来存在的结构优势在患者层或 cohort 层变弱。

### 0.2 不称“真实结构”

SEEG 只观察临床植入所覆盖的部分组织。每位患者的电极布局不同，也不保证完整覆盖 SOZ、传播源、传播终点或所有中继区。因此全文统一使用：

```text
PATIENT_ALIGNED_OBSERVED_STRUCTURE
患者记录布局对齐的结构
```

不得写成：

```text
TRUE_CONNECTIVITY
真实神经连接
完整病理网络
SOZ propagation graph
```

即使结果阳性，也只表示在**已记录触点构成的观测问题中**，某种结构先验提高了有序历史预测效率；它不证明未采样组织中的解剖连接。

### 0.3 两条平行实验线，不合并分母

#### SEEG 主队列

28 位患者，回答：

> 在弱、强两级无序 baseline 下，患者训练序列与记录布局共同定义的低维有序历史通路，是否比同容量错位结构更有效地预测未来；这种信息能否由同一个共享算子自主推进，而不依赖每个 horizon 的独立读出？

#### 高密度 ECoG 构造效度个案

E958 与 E1084，回答：

> 在物理上下左右邻接较明确、触点覆盖更密的网格上，少参数图递归是否表现出容量—结构—运行期使用关系？

ECoG 是两患者 case series，不与 28 人 SEEG 合并、不承担 cohort 复制，也不决定 SEEG 是否继续执行。

### 0.4 不设置科学结果 gate

除数据错配、未来泄漏、数值错误、checkpoint 损坏和不可复现外，所有预设分析都执行。下列结果均不是停止条件：

- aligned 结构没有优势；
- free model 也学不会；
- direct model 阳性但 autonomous model 阴性；
- aligned-orderless bag 与 aligned-ordered 相同；
- bypass interaction 接近零或方向相反；
- cohort 中位数接近零；
- E958 与 E1084 不一致；
- synthetic recovery 在低覆盖条件下很弱；
- train-time advantage 与 inference-time dependence 分离。

结果用 evidence matrix 和容量曲线表达，不用一串 pass/fail claim ladder。

---

## 1. 科学边界：癫痫患者的部分观测问题

### 1.1 可观测对象

模型只观测：

- 临床植入触点；
- 每次间期事件中这些触点的 rank-set 顺序；
- 触点三维位置、shaft 身份和冻结的记录布局；
- parent 已冻结的数据 split。

模型没有观测：

- 未植入组织；
- SOZ 的完整三维范围；
- 未被触点捕获的事件起点和中继区；
- 白质束、真实突触连接或轴突传导延迟；
- 发作期 target。

### 1.2 弱 cohort 效应的预期解释

患者间效应弱或异质，不自动等于机制不存在。至少有四种可区分解释：

1. 候选 motif 不适合；
2. 状态 rank 太低；
3. 观测触点没有覆盖关键传播区域；
4. 患者之间实际采用不同有效计算。

因此必须同时报告：

- 每位患者的完整效应；
- cohort 中位数及不确定性；
- 正、负、近零患者数；
- `n_contacts`、`n_shafts`、几何有效维数、事件数；
- 已记录触点中临床 SOZ 标注的比例，准确名称为 `recorded_SOZ_annotation_fraction`，不得称为 SOZ coverage；
- synthetic 中已知的 latent-source coverage 与可恢复率。

以上覆盖变量只作异质性描述，不按结果排除患者，也不进入 checkpoint selection。

### 1.3 患者级统计优先

event、horizon、null graph、seed 和 data fraction 先在患者内折叠，患者才是 cohort 推断单位。数百万个决策或 rollout 不能被当成独立样本。

---

## 2. 冻结输入、split 与防泄漏合同

### 2.1 复用对象

SEEG 主队列复用 `dynamical_motif_rnn_v0_1` 的 28 位患者级 `GEOMETRY_ONLY_PCA2` cache：

- contact identity 与顺序；
- rank-set event matrix；
- contact 三维坐标和 shaft；
- `split = 0/1/2/-1`；
- event、cache、geometry 和 parent-heldout hashes。

不重新提取事件，不重新定义 TA/TB，不更换 contact set，不读取 seizure target。

### 2.2 Split 语义

```text
split 0  = train
split 1  = validation / checkpoint / temperature
split 2  = development test
split -1 = parent model-unseen confirmation
```

`split -1` 在模型、rank、data fraction、basis、正则和 null family 冻结前不能进入任何选择。它仍称 model-unseen confirmation，不称 prospective validation。

`split -1` 只确认以下预先锁定的紧凑组合：

```text
r = 4
100% split-0 training events
U_FULL_SET
AUTONOMOUS_SHARED_OPERATOR
PATIENT_ALIGNED vs patient-median ANGLE_ROTATED_AXIS
FREE_LOW_RANK vs corresponding unordered baseline
DIRECT_HORIZON_UPPER_BOUND 作为预测上界
```

其他 rank、25/50% learning curves、`U_MINIMAL`、locality-rewired、nonlinear sensitivity、time proxy 和完整 transplant family 全部停留在 split 2。禁止把 split -1 变成反复查询的第二个 development test。

### 2.3 训练数据比例

在 split 0 内建立 block-stratified、严格嵌套的事件子集：

\[
25\% \subset 50\% \subset 100\%.
\]

同一患者、rank 和 seed 下，aligned、null 与 free 模型使用逐位相同的事件 ID。子集按 recording block 和事件长度分层，不按新模型结果、TA/TB 或 SOZ 选择。

完整 rank 曲线在 100% 数据上运行；25%/50% 学习曲线预先固定在 `r=4`，避免把全部矩阵扩大成没有额外科学含义的组合爆炸。

学习曲线分成两条，二者并行执行、互不作为 gate：

1. `END_TO_END_BASIS_CURVE`：basis 与动力学参数都只用对应 25/50/100% 子集；
2. `BASIS_PRETRAINED_CURVE`：basis 固定为仅由 100% split 0 构造的 `Q_100`，动力学参数只用对应子集训练。

第二条曲线不读取 split 1/2/-1，不构成 test leakage。两条曲线的差异用于区分“患者 axis 难以少样本估计”和“低维状态本身难以少样本学习”。

---

## 3. 冻结两级完全无序的 baseline

### 3.1 为什么必须直接操纵 bypass 强度

降低 ordered-state rank，并不等于降低全部替代路径。完整累计 contact set 本身可以携带强无序共现信息：哪些 contacts 已出现、哪些尚未出现、事件处于什么阶段，以及哪些 contacts 经常与当前集合共同出现。

因此正式实验不是“一个固定 baseline × state rank”，而是：

\[
\boxed{
\text{unordered bypass strength}
\times
\text{ordered-state capacity}
\times
\text{structure}
}
\]

### 3.2 两级 baseline

对事件 \(e\) 的 prefix step \(t\)：

\[
S_{e,t}=\bigvee_{q\le t}x_{e,q}
\]

是累计已出现触点集合。

#### `U_MINIMAL`

只允许读取：

\[
a^{min}_{e,t}=\left[x_{e,1},\ t,\ |S_{e,t}|/C\right]
\]

以及 contact-specific intercept。它解释静态 participation、起点和事件进程，但不能使用完整累计集合的无序共现。

#### `U_FULL_SET`

允许读取：

\[
a^{full}_{e,t}=\left[x_{e,1},\ S_{e,t},\ t,\ |S_{e,t}|/C\right].
\]

这是当前版本的强无序 baseline。

两者都不允许读取：

- 当前最后一个 rank set \(x_{e,t}\)；
- rank-set 的先后排列；
- prefix centroid displacement；
- TA/TB；
- future contacts、future cardinality 或 STOP label。

两级 baseline 使用同一模型 family：

\[
\ell^{base}_{e,t,h}
=b_h+U_hV_h^\top a_{e,t}.
\]

baseline rank 在任何 ordered model 训练前只用 split 0/1 冻结。每一级 baseline 在对应比较内部对所有结构、state rank、seed 和 use-phase 操作完全相同。

### 3.3 绕行路径交互

在 `r=4`、100% data 上定义：

\[
\Delta_{structure}^{U}
=L_{null}^{U}-L_{aligned}^{U},
\]

\[
I_{bypass}
=\Delta_{structure}^{U_{MINIMAL}}
-\Delta_{structure}^{U_{FULL\_SET}}.
\]

\(I_{bypass}>0\) 表示患者对齐结构的价值在无序 set 共现旁路减弱后增强。该交互是本阶段中心问题之一，不以任一 baseline 的单独显著性作为运行另一 baseline 的 gate。

### 3.4 防绕行合同

后续模型只学习：

\[
\ell_{e,t,h}
=\ell^{base,U}_{e,t,h}
+\Delta\ell^{ordered}_{e,t,h}.
\]

两级 baseline 都必须通过显式顺序置换测试：在起点、累计集合、prefix 长度和 recruited fraction 不变时，重新排列中间 ranks 后输出逐位不变。

不得再出现：

- 自由 \(C\times C\) contact transition table；
- 第二条读取有序 prefix 的 MLP/RNN；
- full-tissue recurrent residual；
- 将真实 future rank teacher-force 回状态；
- 结构模型专属的 contact bias、baseline 或输出温度。

---

## 4. 严格低维模型：预测子空间与自主共享算子分开

### 4.1 共同 prefix state

所有有序历史只通过：

\[
z_{e,q+1}=F_mz_{e,q}+B_m^\top x_{e,q},
\qquad z_{e,q}\in\mathbb R^r,
\]

进入未来预测，正式 state rank 为：

\[
r\in\{1,2,4,8\}.
\]

低维的是**有序历史对条件均值的贡献**，不是完整 contact field、观测噪声或未记录组织。

### 4.2 D1：直接多步预测上界

`DIRECT_HORIZON_UPPER_BOUND` 使用独立 horizon readout：

\[
\Delta\ell^{direct}_{e,t,h}
=R_{m,h}z_{e,t},
\qquad h\in\{1,2,3,4,5\}.
\]

cardinality 使用对应的 horizon-specific 低维 scalar head。它回答：给定相同低维 prefix state 和空间 basis，最多有多少未来信息可以被直接解码。因为每个 \(h\) 有独立 contact/cardinality readout，该模型不能承担“同一动力学自主产生多步未来”的结论。

### 4.3 D2：自主共享算子

`AUTONOMOUS_SHARED_OPERATOR` 在 prefix 结束后不再接收真实 future rank：

\[
z_{e,t+h}=F_m^h z_{e,t},
\]

\[
\Delta\ell^{auto}_{e,t,h}
=R_mz_{e,t+h}.
\]

所有 horizon 共享同一个 \(F_m\) 和同一个 contact readout \(R_m\)。只允许预冻结的 horizon intercept、scalar gain 或 availability mask，不允许每个 horizon 一个自由 contact readout。

cardinality 同样从 \(z_{e,t+h}\) 经过一个共享 scalar readout 产生，只允许 horizon-specific intercept；不得用独立 horizon MLP 绕过 \(F_m^h\)。

自主模型的 primary suffix field 从预先固定的未来 1–5 步概率累积得到：

\[
\widehat f^{suffix,5}_{e,t}
=1-\prod_{h\in\mathcal H}
\left(1-\widehat p_{e,t,h}\right),
\]

其中 \(\mathcal H=\{1,2,3,4,5\}\) 对所有事件固定，不使用真实事件剩余长度。应用 no-repeat mask；不存在的真实 horizon 只影响对应 target eligibility。完整事件 suffix 只能在 STOP head 冻结后由 closed-loop rollout 评估。独立 full-suffix head 只作为 `DIRECT_SUFFIX_UPPER_BOUND`，不能与自主 suffix 结果混写。

### 4.4 结构模型的反重参数化约束

患者对齐模型与所有结构 null 必须使用同一个冻结空间基同时约束输入和输出：

\[
B_m=Q_mC^{in}_m,
\]

\[
R_{m,h}=Q_mC^{out}_{m,h}
\quad\text{或}\quad
R_m=Q_mC^{out}_m.
\]

其中 \(Q_m\in\mathbb R^{C\times r}\) 列正交。可学习部分只在 \(r\) 维坐标内变化。不得为 aligned model 增加独立自由 contact readout。

### 4.5 候选空间 bases

#### Geometry layout

`Q^{geometry}` 只使用 contact 三维位置、shaft 和冻结 local kernel \(K_0\)，由 \([K_0,K_0^2]\) 的低频字典得到；不读取事件 rank、suffix、TA/TB 或时间。

#### Shaft gradient

`Q^{shaft}` 只使用每根 shaft 内的线性坐标、相邻关系和 shaft identity，完全不读取事件结果。它直接检验候选患者轴是否超越植入轴向采样。

#### Patient-aligned

对每位患者，仅用允许的 split-0 basis-estimation events：

1. 计算起点到 late-field centroid 的位移；
2. 由位移外积的主特征向量得到无向轴 \(u_p\)，其中 \(u_p\equiv-u_p\)；
3. sign 仅按三维几何确定，不按 held-out performance 或 TA/TB；
4. 在 observed-contact geometry 上构造 \(K_0,K_+,K_-\)；
5. 形成 \([K_0,K_+,K_-,K_+^2,K_-^2]\)；
6. 投影常数场和冻结 shaft indicators 后取前 \(r\) 个 left singular vectors，得到 \(Q^{align}_{p,r}\)。

该 basis 使用 train suffix 监督，因此是“训练序列定义的候选空间字典/归纳偏置”，不是 target-free anatomy。近一维患者保留在一维结果中，二维方向图按实际 denominator 报告。

### 4.6 与患者 axis 直接对应的 nulls

#### Angle-rotated axis：主方向 null

在患者冻结的二维 analysis plane 内，以预冻结角度旋转 \(u_p\)，保持同一 local kernel、各向异性强度、rank、参数量和 contact identity，再构造 \(Q^{angle}\)。正式主 null 为患者内 8 个预冻结旋转角；患者效应先相对该患者 angle-null median 计算。近一维或二维几何不合格患者不补造 angle null，保留在 aligned-vs-H0/free/identity 分析中，但不进入 angle-null denominator。

#### Identity-permuted：contact identity null

\[
Q^{perm}=PQ^{align}.
\]

置换在 shaft、径向距离和 degree bins 内进行，保留 rank、奇异值、列范数与参数量。Core 1 使用 4 张预冻结 null。

#### Locality-rewired：图结构 sensitivity

重连 observed-contact graph，尽量匹配 degree、edge-length、within/cross-shaft、connectedness 与一/两步可达规模，再生成 \(Q^{rewire}\)。它只在 `r=4`、100% data、`U_FULL_SET` 中使用 4 张预冻结 null；不能精确匹配的项目逐项报告。

### 4.7 Aligned-orderless control

`H1_ALIGNED_ORDERLESS_BAG` 使用与 ordered aligned 模型完全相同的 \(Q^{align}\) 和输出限制，但状态只由累计集合产生：

\[
z^{bag}_{e,t}
=C_{bag}^\top Q^{align\top}S_{e,t}.
\]

它不使用递归 \(F\)，不读取 rank 顺序。该对照区分：

- aligned basis 本身是好的 suffix 空间字典；
- aligned basis 中的有序历史在字典之上还有增量。

### 4.8 正式模型与读法

| 读者名称 | 代码 ID | 回答的问题 |
|---|---|---|
| 无有序历史 | `H0_UNORDERED_ONLY` | 对应 `U_MINIMAL` 或 `U_FULL_SET` 已解释多少？ |
| 只看植入几何 | `H1_GEOMETRY_LAYOUT` | 一般 contact cloud 与局部平滑是否足够？ |
| Shaft 方向 | `H1_SHAFT_GRADIENT` | 结果是否仅来自电极轴向采样？ |
| 患者训练序列对齐 | `H1_PATIENT_ALIGNED` | 患者对齐结构是否提高低维历史效率？ |
| 同 basis、无顺序 | `H1_ALIGNED_ORDERLESS_BAG` | 优势来自空间字典还是 rank 顺序？ |
| 方向旋转 | `H1_ANGLE_ROTATED_AXIS` | 患者 axis 是否优于其他同样平滑方向？ |
| 位置错位 | `H1_IDENTITY_PERMUTED` | 正确 contact identity 对齐是否重要？ |
| 局部性匹配假结构 | `H1_LOCALITY_REWIRED` | 优势是否只来自一般局部性？ |
| 自由低维历史 | `H1_FREE_LOW_RANK` | 当前 rank 与线性低维模型是否足够？ |

所有 ordered 结构均分别运行 direct 与 autonomous family。`H1_FREE_LOW_RANK` 是容量充分性上界，不承担严格等参数的结构比较。

### 4.9 Basis ceiling：训练前先回答“表示上是否可能”

对每个冻结 basis、两级 baseline 和 held-out residual field 分别计算：

\[
E_{oracle}(Q;U)
=\min_c
\left\|
y^{res}-Qc
\right\|_2^2,
\qquad
y^{res,U}=y-\widehat y^{base,U},
\qquad U\in\{U_{MINIMAL},U_{FULL\_SET}\}.
\]

比较 `aligned / geometry / shaft / angle-rotated / identity-permuted / train-only free-PCA upper bound`，并报告 principal angles。系数 \(c\) 可对每个 held-out field 单独最优化，因此它是 representation ceiling，不是可部署预测器。两级 baseline 的 ceiling 分开报告，用于判断强无序 set 是否已经从 residual 中移除了候选结构可表示的部分。

若 aligned ceiling 不优于 null，说明该 basis 本身缺乏表示优势；若 ceiling 阳性而训练模型阴性，问题在状态输入、共享动力学或优化；若 ceiling 与 orderless bag 都阳性而 ordered 无额外增益，主要证据是 suffix 字典而非 ordered motif。该分析不作为训练 gate。

### 4.10 可学习低维算子

Primary 对所有结构臂使用完整：

\[
F_m\in\mathbb R^{r\times r}.
\]

因为 \(r\le8\)，最多 64 个参数；状态仍不能离开固定 \(Q_m\) 子空间。完整 \(F\) 对低维 basis 内的坐标旋转保持协变，避免 SVD 列顺序人为偏爱 banded dynamics。

以下只作 sensitivity：

```text
DIAGONAL_ONLY
BANDWIDTH_1
STABLE_NORMAL
LOW_DIMENSIONAL_TANH
```

若 autonomous 模型只在不稳定 \(F\) 下改善，只能称 finite-horizon predictive operator，不能称有界传播动力学。

---

## 5. 训练目标：prefix-only、set likelihood、STOP 分离

### 5.1 Prefix 与 horizon denominator

Primary prefix 固定为真实前 3 个 rank sets；prefix=2 是预设困难 sensitivity。prefix 结束后不再输入真实 future ranks。

未来 horizon 定义为：

```text
h = 1,2,3  primary direct-horizon family
h = 1      顺序任务完整性诊断，不单独承担结构 claim
h = 2,3    核心多步结构终点
h = 4,5    长事件 sensitivity
suffix     所有存在 future suffix 的事件
```

每个 horizon 使用独立 eligibility denominator，不能先把所有 decision 混成一个总分。必须同时报告每位患者各 horizon 的 event 数与 rank-length 分布。

### 5.2 Rank-set likelihood

每个 horizon 同时预测下一 rank-set cardinality 和 contact logits：

\[
p(n_{e,t+h}\mid z_{e,t})
\]

与

\[
\ell_{e,t,h}\in\mathbb R^C.
\]

使用与 stochastic decoder 一致的 exact subset law：

\[
L_h
=-\log p(n_{e,t+h})
-\log p(S_{e,t+h}\mid n_{e,t+h},\ell_{e,t,h}).
\]

已经 recruited 的 contacts 从 available set 中 mask；缺失 horizon 只 mask 该 horizon，不伪造 STOP contact。独立 BCE 仅作兼容 sensitivity，不承担 primary next-set claim。

### 5.3 Suffix field

`DIRECT_SUFFIX_UPPER_BOUND` 可使用独立 full-suffix readout，并用 event-balanced BCE/Brier 评分。`AUTONOMOUS_SHARED_OPERATOR` 的 spatial primary 是由固定未来 1–5 步 probability 累积得到的 `suffix-5 field`，不能增加独立 suffix head；完整 suffix 只在 STOP 冻结后的 closed-loop rollout 中评分。

suffix 评分必须：

- 在事件内对 contacts 聚合；
- 报告 balanced Brier/BCE 与不平衡原始值；
- 对每个 event 等权；
- 不让长事件因可用 horizon 更多而获得更大权重。

正式聚合顺序固定为：

```text
horizon-specific decision
→ event mean
→ seed/null median
→ patient
→ cohort
```

### 5.4 空间 checkpoint

Direct 与 autonomous 分别用自己的 validation spatial objective 选择 checkpoint：

\[
L_{space}
=\sum_{h\in\{1,2,3\}}w_hL_h
+\lambda_f L_{suffix}.
\]

权重只用 split 1 冻结。h=4/5 不进入 primary checkpoint，STOP loss 也不进入空间 checkpoint。

### 5.5 STOP 单独拟合

空间模型冻结后，再单独拟合：

\[
p(\mathrm{STOP}_{t+h}\mid z_t,t,|S_t|).
\]

STOP 只回答事件时程是否受益于低维历史，不影响空间模型比较。不得用 STOP 阳性覆盖空间阴性。

### 5.6 Stochastic rollout

随机闭环 rollout 是 secondary evaluation。只有在 direct/autonomous 的 held-out 结果明确后才解释生成行为，但无论前述结果阳性或阴性均完成预设 rollout。

评价包括：

- 固定 3/5 步 cumulative field；
- late-field endpoint；
- 完整 suffix field；
- event length 与 STOP；
- 模板均值和模板内 covariance。

所有结构使用同一个 split-1 temperature 和 common random numbers。rollout 不稳定不影响 direct/autonomous 的执行与报告。

---

## 6. 容量、旁路、样本效率与训练后使用

### 6.1 容量曲线

主横轴为 ordered-state rank \(r=1,2,4,8\)。同时报告 ordered parameters、总 trainable parameters、validation/test loss、effective rank、wall time 和 seed variance。最终画性能—容量 Pareto 曲线，不要求低容量模型达到 parent full RNN 的绝对性能。

### 6.2 两条学习曲线

在 `r=4` 比较 25/50/100% train events：

1. `END_TO_END_BASIS_CURVE`：basis 与动力学都随数据量重估；
2. `BASIS_PRETRAINED_CURVE`：固定 split-0 100% basis，只改变动力学训练事件。

\[
\Delta_{structure}(f)
=L_{angle\ null}(f)-L_{aligned}(f).
\]

若 fixed-basis 少数据优势明显而 end-to-end 不明显，瓶颈主要是患者 axis 估计；若两条曲线都在少数据时增强，支持 sample-efficient inductive bias。

### 6.3 SEEG primary use-phase experiments

#### Prefix-order cost

代码 ID：`PREFIX_ORDER_COST`。

保持起点、累计 contact set、prefix 长度与 contact cardinality不变，只交换第二/第三 rank，或按预冻结规则反转中间顺序：

\[
\Delta_{order}
=L_{permuted\ prefix}-L_{original\ prefix}.
\]

两级 unordered baseline 必须严格不变。该实验直接回答训练后的 ordered model 是否实际读取 rank 顺序。

#### Ordered-path ablation cost

代码 ID：`ORDERED_PATH_ABLATION_COST`。

测试时令：

\[
z_t=0
\quad\text{或等价地}\quad
\Delta\ell^{ordered}=0,
\]

保持 unordered baseline、available mask、checkpoint 和 temperature 不变：

\[
\Delta_{ordered\ path}
=L_{ablated}-L_{intact}.
\]

若 aligned 优于 null，同时 aligned 的 \(\Delta_{order}\) 与 \(\Delta_{ordered\ path}\) 均更大，才允许写“该预测优势实际依赖有序历史通路”。

### 6.4 SEEG basis transplant：只测子空间特异性

保留 aligned/null 的 2×2 train–test basis transplant：

| | test aligned | test null |
|---|---:|---:|
| train aligned | `AA` | `AN` |
| train null | `NA` | `NN` |

报告：

\[
\Delta_{test|A}=L_{AN}-L_{AA},
\]

\[
\Delta_{test|N}=L_{NN}-L_{NA},
\]

\[
\Delta_{train|A}=L_{NA}-L_{AA},
\]

\[
\Delta_{train|N}=L_{NN}-L_{AN},
\]

\[
I_{transplant}
=(L_{AN}-L_{AA})-(L_{NN}-L_{NA}).
\]

SEEG 中统一称 `BASIS_TRANSPLANT_COST/TRANSFER`，不得称 runtime lesion、online necessity 或自然组织依赖。它更换了 encoder/readout 字典，天然包含坐标系不兼容成本。

transplant 后不重新训练、不重新校准、不改变 baseline。协变旋转恒等性测试必须证明同一子空间内部的坐标旋转不改变 logits；该测试只验证实现。

### 6.5 ECoG graph swap

ECoG 保持 contact identity、输入和输出不变，只更换 observed-grid graph operator，因此可以称 `RUNTIME_GRAPH_SWAP`。SEEG 与 ECoG 的 use-phase 证据不得共享同一术语或合并分母。

### 6.6 解释矩阵

| 结果 | 允许解释 |
|---|---|
| aligned 只在 direct 中优于 null | patient-aligned basis 是预测性压缩坐标，不支持共享传播算子 |
| aligned 在 autonomous 中也优于 null | 同一患者对齐算子可自主生成多步未来 |
| aligned ordered ≈ aligned bag > null | 主要是 suffix 空间字典，不是 ordered motif |
| aligned ordered > aligned bag > null | basis 有用，rank 顺序在其上仍有增量 |
| \(I_{bypass}>0\) | 结构优势在无序 set 旁路减弱后增强 |
| free direct 能学、free autonomous 不能 | 未来可低维预测，但不符合所测共享线性动力学 |
| free 也学不会 | 在该 rank 与低维模型族下不能学会；不自动归因于状态维数 |
| patient effects 正负并存 | 有效计算或观测覆盖异质，不强行形成 cohort 单机制 |

---

## 7. 主要与平行终点

### 7.1 预设核心对比

为了让探索性结果仍有清楚中心，核心展示固定为：

```text
r = 4
100% train events
prefix = first 3 rank sets
U_FULL_SET
DIRECT_HORIZON_UPPER_BOUND 与 AUTONOMOUS_SHARED_OPERATOR 分开
aligned vs patient-median angle-rotated null
aligned ordered vs aligned orderless bag
split 2
```

split -1 只确认 §2.2 的 compact autonomous comparison；angle-null comparison 按二维实际 denominator 报告，不能用其他 null 静默补齐。该固定对比不是科学 gate；其余 rank、data fraction 和 null family 按分数设计报告。

### 7.2 空间终点

1. h=1/2/3 的 cardinality NLL、exact subset NLL 与 total set NLL，逐 horizon 报告；
2. h=4/5 的 long-event sensitivity，使用独立 denominator；
3. autonomous 累积 suffix field 的 balanced Brier/BCE；
4. direct suffix upper bound，与 autonomous 分开；
5. predicted late-field centroid 与真实 late-field centroid 的距离；
6. 真实减 baseline 的方向持续性 residual；
7. 相似起点/累计集合下的 train-only TA/TB field separation，作为 secondary；
8. 固定 3/5 步 rollout cumulative field 与 covariance；
9. full STOP rollout 的长度和范围，单独解释。

`last contact` 不是唯一 endpoint。正式 endpoint 为后 20% ranks 的 probability-weighted late-field centroid；last-contact endpoint 只作 sensitivity。

### 7.3 患者级与 cohort 级报告

每个终点报告：

- 28 位患者完整 paired effects；
- median、bootstrap 95% CI、正负/近零计数；
- Wilcoxon 或 sign-flip p 作为描述，不作为是否运行后续实验的门；
- angle-rotated、identity-permuted 与 locality-rewired 各族 null 分开；
- split 2 与 split -1 分开；
- coverage descriptors 与效应的探索性关系，明确不作因果解释。

不得只报告 cohort p 值，也不得只挑显示最强的患者。

### 7.4 时间代理平行实验

上一轮最稳定的未利用线索，是控制 rank-step 后事件内谱质量中心时间代理仍与触点距离正相关。为避免再次只用 ordinal rank 约束 motif，在 `r=4`、100% train events、split 2 上平行运行：

```text
SPACE_ONLY
SPACE_PLUS_TIME_PROXY
```

time head 与空间 horizon 对齐，预测：

\[
\Delta\tau_{t\rightarrow t+h},
\qquad h\in\{1,2,3\},
\]

的粗时间 bin 与截尾连续 \(\log(1+\Delta\tau)\)。它与空间 loss 分开报告，并比较 aligned、geometry-only、angle-rotated 与 free。该变量始终称 `spectral-centroid latency proxy`，不得写成轴突传导延迟或速度。time head 不进入 split -1 compact confirmation，也不改变 SPACE_ONLY primary。

### 7.5 Goal 顺序：资源顺序，不是 gate

| Goal | 首先回答什么 | 核心输出 |
|---|---|---|
| G0 | future residual 在表示上是否可被候选低维 basis 覆盖？ | basis ceiling、principal angles、horizon denominator、两级 H0 |
| G1 | 控制无序 set 后，free low-rank 是否实际使用顺序？ | rank curve、prefix-order cost、ordered-path ablation |
| G2 | 未来是直接可解码，还是可由共享算子自主推进？ | direct vs autonomous |
| G3 | patient-aligned 是否超越 geometry、shaft 与 matched null？ | r=4、100%、U_FULL_SET |
| G4 | 无序旁路减弱后结构是否更重要？ | \(I_{bypass}\) |
| G5 | 结构是否省状态或省样本？ | capacity、end-to-end/fixed-basis curves |
| G6 | 训练后使用顺序、ordered path 或具体 basis 到什么程度？ | order perturbation、z ablation、basis transplant、ECoG graph swap |
| G7 | 改善发生在哪个行为量？ | horizon、suffix、endpoint、time、STOP matrix |
| G8 | 部分观测下能看见多少？ | compact synthetic surfaces、patient-matched detectability |
| G9 | 模型是否恢复模板均值和模板内方差？ | stochastic rollout 与补充图 |

G0–G9 都按预设执行；任何前序科学阴性不停止后序分析。

---

## 8. ECoG 平行构造效度实验

### 8.1 定位

E958/E1084 物理网格比 SEEG 的距离图更接近明确的观测邻接，但仍然不是完整皮层网络。既有结果显示真实四邻接在 E958 有 train-time 优势、E1084 未复制，且优势依赖两次内部更新；在线入边必要性未支持。

新实验不覆盖这些结果，而使用与 SEEG 相同的 `U_MINIMAL/U_FULL_SET + ordered residual` 合同，并分开 direct 与 autonomous family，避免 STOP 和静态共现再次主导。

### 8.2 少参数图递归容量

对冻结网格邻接 \(A\) 比较：

\[
G_1=\alpha_0I+\alpha_1\widetilde A,
\]

\[
G_2=\alpha_0I+\alpha_1\widetilde A+\alpha_2\widetilde A^2,
\]

\[
G_3=\alpha_0I+
\sum_b\alpha_b\widetilde A_b,
\]

其中 edge types 只包括一跳/两跳与预冻结的网格方向类别。`G4` 为既有逐边自由模型，只作高容量参照。

每个容量比较：

```text
OBSERVED_GRID
IDENTITY_PERMUTED_GRID
DEGREE_AND_DISTANCE_REWIRED_GRID
FREE_SAME_STATE_UPPER_BOUND
```

并运行 observed-grid runtime graph swap、25/50/100% 数据曲线、空间与 STOP 分离评分。ECoG 从 G3 开始与 SEEG 并行，不依赖 SEEG 的科学结果。

### 8.3 ECoG 解释边界

- 两位患者逐人报告，不形成二人 pooled p；
- E958 阳性不能称 ECoG cohort 机制；
- E1084 不一致不能否定 SEEG；
- 结果必须注明 microsteps、邻接定义和坏触点处理；
- 物理网格优势称为 observed-grid inductive bias，不称皮层突触图。

---

## 9. Synthetic identifiability：三个紧凑实验，不做全笛卡尔积

### 9.1 S0｜实现正确性

少量 canonical cells 验证：

- effect=0 时 false-positive 行为；
- aligned direct teacher 可被 aligned direct student 恢复；
- aligned autonomous teacher 的已知 \(F\) 与结构排序可被恢复；
- identity-permuted teacher 不被 aligned 错认；
- `U_MINIMAL/U_FULL_SET` 的 bypass 方向正确；
- prefix-order 与 ordered-path ablation 对已知 ordered teacher 有效。

S0 只验证实现；科学效应大小不决定真实数据是否运行。

### 9.2 S1｜经验设计附近的功效

选择 6 类代表 montage：小/中/大 contact，少/多 shaft，近一维/二维，source-near/source-far。扫描：

```text
effect = 0 / medium / strong
bypass = low / high
noise = 2–3 levels
model family = direct / autonomous
```

输出 false-positive rate、\(P(aligned>angle\ null)\)、orderless-vs-ordered recovery 和 autonomous operator recovery。

### 9.3 S2｜模型失配与部分观测

使用 Latin hypercube 扫描：

- unobserved-node fraction；
- extra latent state；
- direction jitter；
- bypass strength；
- contact-specific noise；
- source-to-electrode distance。

28 位真实 montage masks 只在一组冻结 canonical teachers 下全部运行，生成 patient-specific detectability descriptors；不与全部 synthetic 参数做笛卡尔积。另行比较 random、shaft-like 与 source-avoiding masks。

输出：

\[
P(\text{recover aligned structure})
=f(r,N_{event},C,noise,bypass,coverage),
\]

但只将其作为真实阴性解释范围。synthetic 结果不是科学 gate。

---

## 10. Evidence matrix 与措辞边界

### 10.1 独立证据层

| 层 | 问题 | 输出 |
|---|---|---|
| E0 Representation ceiling | 候选 basis 在表示上能否覆盖 held-out residual？ | oracle projection error、principal angles |
| E1 Ordered information | free low-rank 是否学得会且实际使用顺序？ | loss–rank、prefix-order、z-ablation |
| E2 Predictive vs dynamical | 未来是直接可解码还是共享算子可自主推进？ | direct–autonomous matrix |
| E3 Structure | aligned 是否优于 geometry、shaft 与 matched null？ | \(\Delta_{structure}\) |
| E4 Bypass interaction | 结构优势是否在无序 set 旁路减弱后增强？ | \(I_{bypass}\) |
| E5 Sample efficiency | 结构优势是否随数据减少增强？ | end-to-end/fixed-basis curves |
| E6 Use phase | 模型是否使用顺序、ordered path 或特定 basis？ | order cost、ablation cost、transplant cost |
| E7 Behavioral target | 改善落在 horizon、suffix、endpoint、time 还是 STOP？ | endpoint matrix |
| E8 Coverage | 部分观测下能否恢复？ | compact synthetic/empirical heterogeneity |
| E9 Dense-grid validity | ECoG graph swap 是否出现对应关系？ | 两患者 case series |

各层独立报告，不把未满足的一层改写成整个阶段失败。

### 10.2 允许措辞

若数据支持，可写：

> 在部分 SEEG 观测条件下，由患者训练序列与记录布局共同定义的低维有序历史 basis，以较少状态维数提高了 held-out suffix prediction；prefix-order perturbation 与 ordered-path ablation 表明该增量实际使用 rank 顺序。若 autonomous family 同时成立，可进一步写同一个低维共享算子能够生成多步未来。

若 cohort 弱但个体异质，可写：

> 结构效应在患者间高度异质，与临床植入造成的部分观测相容；当前结果不支持单一 cohort-level motif，但识别了可在个体层继续验证的低容量计算。

若 aligned 与 null 相同而 free 能学，可写：

> 低维有序历史足以预测未来，但提出的患者对齐空间结构没有提供额外约束价值。

### 10.3 禁止措辞

- 找到了患者真实 connectome；
- 电极完整覆盖了 SOZ 或传播网络；
- 结构阴性证明脑内不存在方向传播；
- 单一 ECoG 患者证明一般局部皮层机制；
- train-time advantage 自动等于在线必要性；
- test-time swap 损害等于自然组织 lesion；
- direct-horizon 阳性自动等于共享传播动力学；
- aligned bag 阳性等于有序历史 motif；
- SEEG basis transplant cost 等于 runtime graph dependence；
- 低维 state 等于癫痫特异神经轴；
- 该间期实验恢复了既有阴性的 seizure reuse；
- cohort 中位数不显著等于所有患者无效。

---

## 11. 图形合同

### 11.1 当前 Figure 6 正式候选

用户于 2026-08-17 确认的当前 Figure 6 正式候选是：

`results/paper-ready-figure/fig6_interictal_crossstate_response_r5_candidate/figures/topic5_figure6_interictal_crossstate_response_r5_candidate.{png,pdf,svg}`

它承担当前 RNN 主故事：真实事件顺序提高下一触点预测并恢复患者间期传播场；间期场与发作早期场存在无方向的形状对应；不同连接设计产生相似的有限时程组织扰动响应。新实验不得覆盖、改名或静默替换这张图。

`fig6_dynamical_motif_rnn_v0_2` 仅作为后续动力学 motif 的历史诊断资产，不是当前 Figure 6 正式候选。

### 11.2 新补充图候选的科学故事

容量约束实验默认先形成 Supplementary / Extended Data 候选，不自动替换当前 Figure 6。它只讲：

```text
弱/强两级无序 baseline
→ 所有顺序信息被压入低维状态
→ 直接预测与自主共享算子分开
→ 比较患者对齐、方向旋转、shaft 和自由状态
→ 看顺序使用、旁路交互、容量和样本效率
```

建议 2 行、6 个 panel：

#### Panel A：实验概念图

沿用现有 Figure 6 Panel A 的视觉语言：左侧画多个 observed rank sets，中间用一个简洁大圆框住 `Low-dimensional history state`，右侧分成两条输出：上方是“每个未来步直接读出”，下方是“同一状态算子逐步向前”。两级无序 baseline 画成灰色直通分支，`full set` 比 `start + progress` 多一个累计 contact-set 图标。圆框下方并列四个小型空间先验：

```text
Geometry only
Patient-trained spatial pattern
Direction rotated
Free low-dimensional
```

不画 tissue mesh 热图，不画长程连接，不在画布放方程、内部代码 ID 或防御性说明。输入必须画多个 rank sets，不能只画第一个。

#### Panel B：一个患者的直观例子

同一个 prefix 下显示真实 suffix field、aligned direct prediction、aligned autonomous prediction、angle-rotated autonomous prediction 和 free autonomous prediction。示例只帮助理解，不承担 cohort claim；默认示例预先固定为既有 Figure 6 的 E1146，只有输入损坏或本任务不合格时才按 contact 数中位、二维合格的冻结规则替换，不能按新模型效果选择。

#### Panel C：basis 表示上限

x = geometry、shaft、patient-trained、direction-rotated、free-PCA；y = held-out residual projection error。用患者配对点和 cohort median/CI，不用柱状图。它回答候选空间字典是否在训练模型前就有表示优势。

#### Panel D：直接预测与自主生成

x = direct / autonomous；y = angle-null minus aligned held-out error。每位患者两点相连，明确显示 aligned 优势是否只存在于直接解码，还是保留到共享算子。

#### Panel E：无序旁路交互

x = minimal baseline / full-set baseline；y = angle-null minus aligned error。每位患者配对，0 线明确。该 panel 直接回答结构优势是否被强无序 set 共现吸收。

#### Panel F：顺序使用与容量/样本效率

左半显示 prefix-order cost 与 ordered-path ablation cost；右半用小型曲线显示 r=1/2/4/8 或 25/50/100% 的 aligned-vs-null margin。不得把 SEEG basis transplant 画成 lesion 或 runtime necessity。

synthetic detectability surface、完整 learning curves、basis transplant 与 ECoG graph swap 分别进入其他 Extended Data panels，不强塞进同一张六面板图。

ECoG 容量—结构曲线进入 Extended Data，不与 28 人 SEEG cohort 合并到同一统计 panel。

### 11.3 视觉硬规则

- 字体：DejaVu Sans；panel label 粗体；最终 PDF 正文字号不低于 7 pt；
- 输入 rank 使用 `viridis`，early→late 方向与全项目一致；
- 模型颜色固定：unordered 灰、geometry-only 浅蓝、patient-trained 深蓝绿、location-shuffled 橙、free 紫；
- 同一模型在所有 panel 使用同一颜色；
- 一个 figure 只保留一套共享 legend；
- 不用大量 bar chart；优先 curve、paired points、patient scatter 和 heatmap；
- 统计图必须有患者点和不确定性；不能只用星号代替效应量；
- 主画布不用内部术语：`motif rank`、`TT/TF`、`null operator`、`C×C`、`gate`；
- PNG、PDF、SVG 同状态生成；PDF 栅格化逐 panel 目视；SVG 文本保留为文本；
- 每个 figure 目录必须有中文 `README.md`、source-data CSV/JSON、metadata 与 visual QA。

---

## 12. 工程 gate（仅这些可阻止相应单元）

1. event/contact/split hash 与 parent 不一致；
2. `U_MINIMAL/U_FULL_SET` 任一 baseline 对 rank 顺序不具置换不变性；
3. horizon target、basis 或 fixed-basis curve 读取 split 2/-1 future；
4. aligned/null rank、参数量、basis norm 或 event IDs 合同不匹配；
5. autonomous family 出现 horizon-specific contact readout 或独立 primary suffix head；
6. exact subset target、cardinality 或 no-repeat mask 实现错误；
7. prefix-order 操作改变累计 set、起点、长度或 baseline 输出；
8. ordered-path ablation 改变 unordered baseline；
9. basis transplant/graph swap 发生参数更新或 baseline 改变；
10. 完整 \(F\) 的协变坐标旋转不能逐位复现 logits；
11. split -1 被 compact confirmation 之外的任务访问；
12. NaN/Inf、维度错配、checkpoint/hash 损坏；
13. worker 输出不完整或无法从 manifest 恢复。

低 event 数、近一维几何、弱 cohort 效应、null matching 部分失败或 synthetic 恢复率低不阻止其他患者/分析；必须按实际 denominator 和资格原因报告。

---

## 13. 方法学依据与适用边界

本设计借鉴低秩 RNN、图卷积递归、latent-circuit 与 teacher–student 可辨识性分析的共同思想：限制动态状态、显式匹配结构对照、并在合成系统上绘制可恢复范围。但本项目的 SEEG 是强部分观测，不能从拟合良好直接推断内部机制唯一。相关工作包括：

- Mastrogiuseppe & Ostojic, *Linking Connectivity, Dynamics, and Computations in Low-Rank Recurrent Neural Networks*；
- Ruiz, Gama & Ribeiro, *Gated Graph Convolutional Recurrent Neural Networks*；
- Qian et al., *Partial observation can induce mechanistic mismatches in data-constrained models of neural dynamics*；
- Pagan et al., *Individual variability of neural computations underlying flexible decisions*；
- latent circuit 与 connectome-constrained recurrent network 的近期工作。

这些先例支持“为什么要做容量、结构和部分观测审计”，不替代本数据上的患者级证据。
