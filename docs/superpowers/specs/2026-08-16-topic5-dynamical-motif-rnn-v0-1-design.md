# Topic 5.2 动力学 motif RNN v0.1-r2：方向性传播与轴向前馈瞬态

> 状态：**AUTHORIZED AND EXECUTING（用户 2026-08-16 授权，覆盖 Phase B 后的暂停条款）**
> 日期：2026-08-16
> Parent：Topic 5.1 multiscale effective scaffold v0.5 与 Topic 5.2 frozen latent landscape v0.2
> 结果根：`results/topic5_dynamical_motif_rnn_v0_1/`
> 本修订版取代同路径的 v0.1 初稿。

## ERRATUM 2026-08-16（设计复核后的修订，实现按本节执行）

完整证据见 `results/topic5_dynamical_motif_rnn_v0_1/SCIENTIFIC_DESIGN_AUDIT.md`
与 `SPEC_IMPLEMENTATION_MAPPING.json`。以下 11 条覆盖正文对应段落。

**P0-1（§2.2）** 父代平面的第一坐标轴**就是** TA/TB 传播轴：
`interictal_field.planes.own_{a,b}.u` 与 `axis_pair.axis_{a,b}.u` 逐位相同，
且 `contacts_xy_mm = (coords_3d − origin) @ [u, w]`（42/42 fits，atol 1e-4）。
因此 `PARENT_FROZEN_FRAME` 里的"自由轴发现"部分循环。全队列默认
`GEOMETRY_ONLY_PCA2`；parent frame 的 G1 结果不得单独支持"超越布局的走廊"。

**P0-2（§5.2）** 几何 frame 中 contact-cloud PCA1 的 layout 轴按定义即 `θ≡0`
（实测 28/28）。因为 spec 把 layout 臂定义为"只在 M0 checkpoint 上做廉价拟合"，
`M1_free − M1_layout` 会混入"多训练一轮"。G1 头号比较改为三个量并列，
**不新增任何完整 RNN 单元**：(a) `M1_free − M1_layout(M0 上廉价拟合)` = 上界；
(b) `M1_free − M1_layout_replay`（把训练好的 M1 的 `θ` 换成 layout 轴、只重校准
decoder）= 下界；(c) 自由轴与 PCA1 / dominant-shaft 的夹角分布。
dominant-shaft 分支在几何 frame 中不退化（in-plane norm 0.72–1.00，28/28 可估）。

**P0-3（§4.1）** "cardinality head" 落地为**冻结递归之后拟合的共享 decoder**
（父代 `RolloutSizeHead` 合同：train continue decisions 拟合、calibration 选 epoch），
不是模型内部的 head。其上加三个温度。

**P0-4（§5.4）** `r_f` 冻结为 `r_local_mm`（`build_pool_contract` 已冻结的局部半径）。
实测前向锥含 **45.4%–49.2%** 的局部边（28/28，均值 48.1%），即"局部支持里朝前那一半"。
设计患者做 `0.5×` / `2×` 敏感性。

**P0-5（§5.5）** `DM3_AXIS_SHUFFLED_TRIANGULAR` 无法同时精确匹配四项。
精确保留：局部支持、核权重、严格三角性；按几何校准到精确匹配：非零数
（`calibrate_shuffle_radius`，只用几何）；只报告不匹配：被选中边的长度分布。

**P0-6（§5.4/§5.5）** `γ ≥ 0`（投影梯度），与 `δ_g, δ_κ ≥ 0` 对称。
G3 因此是单侧检验：数据若要负 `γ`，M3 停在 `γ=0` 并报告"无前馈增益"。

**P0-7（§5.3）** 三个 gate 实例化为同一族的嵌套阶梯：
`a_k = (r̄_{min(k,K)} − r̄_0) / min(k,K)`，`K = 1 / 2 / ∞` 对应
2RANK / 3RANK / ONLINE，三者在 `k=0` 时均为 0。

**P1-1（§5.3）** `σ_s` 取 split-1 事件 `‖r̄_2 − r̄_1‖` 的中位数（**位移模长**，
不投影到轴），使 gate 尺度与学到的 `θ` 无关。实测 1.40–15.91 mm。

**P1-2（§2.3）** 几何 frame 下近一维判定与父代 28/28 一致
（`epilepsiae_139` 0.0279、`yuquan_zhangjiaqi` 0.0230，阈值 0.05）。
二人照常训练评分，不进入任何二维方向结论。

**P1-3（§3.2）** `prefix_mode` 由 train-only 中心 + **前 3 个 rank** 得到；
只作评分目标，**不进入模型输入**（否则会把 rank 2 漏给只该看 rank 0–1 的 M2 gate）。

**P1-4（§5.2）** 三个 seed 的 `θ_init` 分别取 `{0, π/3, 2π/3}`，
避免"自由轴 = layout 轴"成为初始化产物，并让跨 seed 轴稳定性有意义。

**S1（§10.1）** 冻结 sidecar 只有真实 onset 的 0–10 s AUC。
same-block pseudo-onset 与 S2 都需要重建时间分辨 1–150 Hz 缓存；
生产者存在、原始数据已挂载、重建会逐位对拍已有 AUC 作为 provenance 校验。

## 0. 科学问题与总体设计

本阶段不再寻找“哪几条远程边是真实通路”，而直接研究：

\[
\boxed{
\text{什么类型的局部有效动力学，能同时产生稳定的宏观传播模板和有变异的微观 contact 顺序？}
}
\]

正式主链包含四类模型：

| 简称 | 代码 ID | 递归动力学 | 直接回答的问题 |
|---|---|---|---|
| M0 | `DM0_ISOTROPIC` | 各向同性局部扩散 | 患者几何与静态 participation 已经能解释多少？ |
| M1 | `DM1_FREE_AXIS` | 自由学习轴的各向异性局部扩散 | 是否存在超越植入布局的有效传播走廊？ |
| M2 | `DM2_LOCAL_DIRECTIONAL` | 早期位移连续控制的局部有偏传播 | 早期状态是否选择后续传播方向？ |
| M3 | `DM3_AXIS_FEEDFORWARD_TRANSIENT` | 沿传播轴排列的局部前馈 cascade | 轴向前馈成分是否额外解释传播范围和事件长度？ |

另有三个低成本 baseline：

- `LAYOUT_AXIS_ANISOTROPY`：轴固定为 contact cloud PCA1 或 dominant shaft，只拟合各向异性强度；
- `EARLY_DISPLACEMENT_KINEMATIC`：不含 RNN，只用起点和前两个 rank 的位移预测后续方向、终点和模式。
- `EVENT_VECTOR_DIRECTIONAL`：不假定稳定全局轴，直接沿每个事件最早位移向量构造局部方向核。

M3 的正式替代机制为：

- `DM3_GAIN_MEMORY`：更强 recurrent gain 或更慢 leak；
- `DM3_SYMMETRIC_MATCHED`：同一轴向 node pairs 上的对称耦合；
- `DM3_AXIS_SHUFFLED_TRIANGULAR`：保留三角性和权重分布，但打乱患者轴顺序。

原正交矩阵指数模型 `DM2_ORTHOGONAL_ROTATION` 与 normal-matched M3N 降为参数化实验或 Supplementary，不进入正式420-unit预算。

本阶段的组织原则是：

1. 尽量保持 parent 的输入、输出、事件任务和数据划分；
2. 坐标、训练方式和 decoder 改变分别做实验，不与 motif 变化混成一个结论；
3. G0–G6 全部是平行科学问题，不由前一项阳性与否决定是否运行；
4. 结果按连续效应、替代机制和不确定性解释，不压成串联式 pass/fail ladder。

## 1. 哪些对象保持不变

### 1.1 Parent 任务

继续使用现有患者内 rank-set sequence 任务：

- 输入是当前及之前已经观察到的 rank sets；
- 输出是下一 rank-set contacts、下一集合大小和 STOP；
- full-tissue latent nodes 通过固定 `H` 与 SEEG contacts 相连；
- TA/TB 不进入模型 forward、loss、方向门控或 checkpoint selection；
- seizure field 不进入 interictal 模型、motif 选择或 IED basis 构造。

现有最强证据仍然是：真实 prefix–suffix 对应能提高 held-out sequence likelihood。新阶段不是再次证明“历史有用”，而是解释这种历史依赖可能采用什么有效动力学。

### 1.2 Rank-step 是主时间变量

当前 `event_lag_raw` 是事件内谱质量中心时间代理，不是临床 recruitment time 或轴突传导延迟。M0–M3 主递推仍按 ordinal rank-step 更新。`event_lag_raw` 只进入 G6 的描述性 distance–lag sidecar，不进入主递归算子。

### 1.3 Patient-first 推断

患者是独立统计单位。event、rollout、fit view 和 seed 必须先在患者内折叠。任何图和显著性检验都不得把数百万条 rollout 当独立样本。

## 2. Frame experiment：坐标变化与 motif 变化分开

### 2.1 为什么不能只使用一个 frame

现有42个 parent fits 包括14位患者的 `own_a/own_b` 双视图和14位患者的 `shared` 视图。`own_a/own_b` 使用相同 events 和 contacts，但二维平面由完整 TA/TB earliness field 参与定义。它适合保持与旧 Figure 6 的工程连续性，却不适合独立发现一条新的传播轴。

纯几何 PCA2 不读取传播结果，但其第一轴可能主要反映 shaft 和 contact cloud 的植入方向。它也不能自动等同于病理传播轴。

因此启动正式全队列前先运行：

```text
FRAME_EXPERIMENT
    PARENT_FROZEN_FRAME
    GEOMETRY_ONLY_PCA2
```

### 2.2 `PARENT_FROZEN_FRAME`

该臂逐位复用 parent 的：

- tissue nodes；
- `H`；
- contact set；
- local support；
- `own_a/own_b/shared` plane。

它回答新 motif 在旧表示中能否工作。双视图必须折叠到患者，不能把 `own_a/own_b` 当两个独立样本。

### 2.3 `GEOMETRY_ONLY_PCA2`

该臂只使用冻结的 contact 三维物理坐标：

1. 在患者固定 contact set 上做 PCA2；
2. component 顺序按奇异值固定；
3. sign 按最大绝对三维 loading 固定，不读取 rank、TA/TB、suffix、seizure 或模型输出；
4. 重建 tissue mesh、距离、`H` 和 local support；
5. 合并同一患者重复的 `own_a/own_b`，得到28个患者级 fits；
6. 两个近一维患者保留一维结果，不进入二维正交方向结论。

### 2.4 Frame experiment 的判读

在预选的6–8位 sentinel patients 上，对两个 frame 使用相同事件、模型和随机数，比较：

- M0 的预测与闭环校准；
- M1−M0、M2−M1 的效应方向和大小；
- 轴的 seed/block stability；
- contact-layout PCA1、shaft direction 和自由轴的夹角；
- node count、`H` rank、zero-H fraction 和局部覆盖。

该实验不挑“阳性更多”的 frame。它报告 motif 结论是否依赖表示。当前420-unit资源公式只适用于28-fit frame；若用户最终要求42个 parent views 全队列训练，必须另行扩展预算，不能悄悄混入420单元。

旧结果与新结果之间的差异不得全部归因于 motif，因为 frame、mesh 或 `H` 可能变化。正式 motif 结论只来自同一冻结 frame 内的配对模型比较。

## 3. 数据划分与冻结边界

### 3.1 四层 split

沿用现有 cache：

```text
split 0  = model train
split 1  = calibration / checkpoint / hyperparameter selection
split 2  = development test
split -1 = original parent held-out events
```

split -1 必须先逐位证明对应 parent held-out events，且未进入新模型训练、温度、方向轴、basis 或阈值选择。由于历史 contact support 和部分空间表示见过全体 interictal 数据，它只称 `model-unseen confirmation`，不称 prospective validation。

### 3.2 模式标签

TA/TB 继续由 split 0 事件定义。split 1/2/-1 只允许通过 train-only centroids 获得模式概率或标签。所有连续 field 指标同时报告，不要求真实事件被强制二分。

### 3.3 静态 baseline

每位患者仅用 split 0 拟合一个无 recurrence 的 `STATIC_READOUT`，输入为起始 contacts、累计 participation 和固定 contact covariates。它量化静态 participation scaffold 已经能解释多少，不把 contact bias 或 node bias解释成组织易激性。

## 4. 共享 RNN 与 decoder

### 4.1 基础状态更新

继续使用 `state_dim=1` 的 full-tissue leaky RNN：

\[
u_k=(x_kH)\odot g_{in},
\]

\[
h_{k+1}=(1-\kappa)h_k+
\kappa\tanh\!\left(u_k+W_eh_k+b\right),
\]

\[
\ell_{k+1}=b_{contact}+g_{out}h_{k+1}H^\top.
\]

所有主模型共享 contact、STOP 和 cardinality heads、loss、optimizer family、训练预算、split、decoder rule 和 seed registry。

### 4.2 完整 closed-loop state

\[
q_k=(h_k,r_k,k,s_k),
\]

其中 `r_k` 是 recruited contact mask，`s_k` 是方向证据。decoder 必须保存 STOP precedence、cardinality probability、repeat mask、subset sampler、maximum-rank rule、absorbing STOP 和 RNG state。不存在只依赖 `h` 的完整闭环映射。

### 4.3 Stochastic decoder

主训练使用 teacher forcing；主闭环评价使用 split 1 冻结的随机 decoder：

1. 分别校准 contact、cardinality 和 STOP temperature；
2. 先采样 STOP；继续时采样下一 rank-set size；
3. 在未出现 contacts 中使用现有 exact fixed-cardinality subset sampler；
4. 不读取真实未来 cardinality；
5. 所有模型使用 common random numbers；
6. test 上不重新调温度、长度上限或模式比例。

## 5. 四个主动力学模型

以下模型使用同一冻结 local mask `m_ij`。令 `i` 为接收 node、`j` 为来源 node，并定义局部非负核的 column normalization：

\[
\mathcal P(K)_{ij}=
\frac{K_{ij}}{\sum_{i'}K_{i'j}+\epsilon}.
\]

该归一化保持固定局部支持，并使每个来源 node 的外发权重和为1。它不保证完整非线性 RNN 为 normal system；normality 不是 M2 的主问题。

### 5.1 M0：各向同性局部扩散

\[
K^{iso}_{ij}=m_{ij}
\exp\!\left[-\frac{\|r_i-r_j\|^2}{2\ell^2}\right],
\]

\[
W^{(0)}=g_0\mathcal P(K^{iso}).
\]

M0 不允许每条 local edge 独立学习权重。它回答 contact geometry、固定 `H`、静态偏置和一个低参数局部核已经能解释多少。

### 5.2 M1：自由轴各向异性走廊

令：

\[
u=(\cos\theta,\sin\theta),
\qquad
u_\perp=(-\sin\theta,\cos\theta),
\]

\[
\ell_\parallel=\ell e^\eta,
\qquad
\ell_\perp=\ell e^{-\eta},
\qquad \eta\ge0.
\]

\[
d^2_{u,ij}=
\frac{[(r_i-r_j)^\top u]^2}{\ell_\parallel^2}+
\frac{[(r_i-r_j)^\top u_\perp]^2}{\ell_\perp^2},
\]

\[
K^{axis}_{ij}=m_{ij}e^{-d^2_{u,ij}/2},
\qquad
W^{(1)}=g_1\mathcal P(K^{axis}).
\]

`eta=0` 且共享参数相同时严格退化为 M0。`theta` 是无向轴，`theta` 与 `theta+pi` 等价。

#### Implantation-layout baseline

`LAYOUT_AXIS_ANISOTROPY` 预先列出两条不读取事件顺序的物理轴：contact cloud 3D PCA1 与 dominant-shaft direction。dominant shaft 固定为有效contacts最多的shaft，并列时按shaft name排序；其方向由该shaft的3D PCA1给出。二者投影到当前 active frame；投影退化时明确标记不可估。只在每个 M0 checkpoint 上用 split 1 选择较好的固定轴并扫描 `eta` 和低容量 calibration，不另训练完整 RNN。这个较强的baseline回答植入布局本身造成的各向异性已经能解释多少。

重点比较：

\[
M1_{free}-M1_{layout},
\qquad
M1_{layout}-M0.
\]

只有自由轴超越 layout-axis baseline，且跨 seed/block 稳定，才支持“超越采样布局的患者条件性有效各向异性”。

Block stability 不重新训练完整RNN：在冻结shared checkpoint后，分别在每个train recording block只重新拟合`theta/eta`，形成无向角度分布和profile uncertainty。这样不会把大量block fits误计为独立患者。

### 5.3 M2：局部、非负、连续方向偏置

主方向证据由真实前两个 rank sets 产生：

\[
d_2=u^\top(\bar r_2-\bar r_1),
\qquad
s_2=\tanh(d_2/\sigma_s).
\]

第一个 rank 后令 `s_1=0`；观察第二个 rank 后才计算 `s_2`。不使用硬阈值。`sigma_s` 只在 split 1 上冻结。定义：

\[
K^{dir}_{ij}(s)=
K^{axis}_{ij}
\exp\!\left[
\beta s\frac{u^\top(r_i-r_j)}{\ell}
\right],
\]

\[
W^{(2)}(s)=g_2\mathcal P(K^{dir}(s)).
\]

该算子只在冻结 local support 上使用非负权重；`beta=0` 时严格退化为 M1。它不在一个 rank step 内通过 matrix exponential 执行任意多跳。M2 允许与 M1 相同的 shared parameters 小学习率适配，但必须另报一步 state/output norm；再做一个 split-1 gain-matched sensitivity，排除仅靠一步幅度取胜。

#### Gate-emergence experiment

在6–8位 design patients 上平行比较：

- `M2-2RANK`：第二 rank 后固定 `s_2`；
- `M2-3RANK`：累积前两次质心位移后固定；
- `M2-ONLINE`：每一步因果更新方向证据。

全队列主实现暂定 `M2-2RANK`，因为它最直接检验“很早的微小差异是否选择后续方向”。三种 gate 的预测随 prefix length 的 emergence curve 都必须报告。它们不是互相阻断的 gate。

此外，对全队列冻结的 M2 checkpoint 做 `3RANK/ONLINE gate replay`：不重新拟合参数，只替换因果 `s_k` 更新并使用相同随机数。该结果是实现敏感性，不冒充各 gate 的最优重新训练结果。

#### Kinematic baseline

`EARLY_DISPLACEMENT_KINEMATIC` 只用 split 0 拟合：

\[
\widehat r_{end}=\bar r_2+a(\bar r_2-\bar r_1)+b(start),
\]

并拟合 train-only mode probability。M2 必须与该 baseline 比较，而不能只与 M1 比较。若二者相当，允许结论是“存在早期方向惯性”，不升级为方向性递归计算。

#### Event-vector directional baseline

令：

\[
v_2=\frac{\bar r_2-\bar r_1}
{\|\bar r_2-\bar r_1\|+\epsilon},
\]

并在 M0 checkpoint 上构造：

\[
K^{event}_{ij}=
K^{iso}_{ij}
\exp\!\left[
\beta_e\frac{v_2^\top(r_i-r_j)}{\ell}
\right].
\]

`beta_e` 只做 split 1 scalar/grid fit，不完整重训RNN。它检验：即使没有稳定全局走廊，每个事件的早期位移方向是否已经足以形成局部传播偏置。

#### Orthogonal sensitivity

原 `Q_s=\exp(s\beta A)` 只在 design patients 上作为数学 sensitivity。它比较局部正向 transport 与 hidden-state orthogonal rotation，不能解释为生理活动搬运。

### 5.4 M3：沿轴排列的局部前馈瞬态

令 `q_i=u^\top r_i`，构造：

\[
F^+_{ij}=K^{axis}_{ij}
\mathbf 1(0<q_i-q_j<r_f),
\qquad
F^-=({F^+})^\top,
\]

\[
F(u,s)=\max(s,0)F^++\max(-s,0)F^-.
\]

M3 更新为：

\[
h_{k+1}=(1-\kappa)h_k+
\kappa\tanh\!\left[
u_k+W^{(2)}(s_k)h_k+
\gamma F(u,s_k)h_k+b
\right].
\]

`gamma=0` 时严格退化为 M2。模型名固定为 `DM3_AXIS_FEEDFORWARD_TRANSIENT`，避免把一种轴向前馈实现直接推广为所有非正规系统。

M2 本身的非对称核和非线性 Jacobian也可能非正规。M3 检验的是：显式、与患者轴对齐的 triangular feedforward component 是否提供增量解释。

### 5.5 M3 的三类正式替代机制

#### `DM3_GAIN_MEMORY`

不加入 `F`，定义：

\[
g_G=g_2e^{\delta_g},
\qquad
\kappa_G=\operatorname{sigmoid}
\left[\operatorname{logit}(\kappa_2)-\delta_\kappa\right],
\qquad
\delta_g,\delta_\kappa\ge0.
\]

它允许 recurrent gain 增大和 leak 变慢，并使用比 M3 更宽松的两个 scalar 参数。它回答传播范围改善是否只是系统更强、记得更久或更晚 STOP。

#### `DM3_SYMMETRIC_MATCHED`

\[
F_{sym}=|s|\frac{F^++F^-}{2}.
\]

它使用相同 node pairs、距离分布和权重尺度，但不形成单向 cascade。它回答增加轴向局部耦合是否已经足够。

#### `DM3_AXIS_SHUFFLED_TRIANGULAR`

在 train-only 冻结的 distance/degree bins 内打乱 node ordering，保留：

- triangular structure；
- 非零数；
- local distance distribution；
- weight distribution；
- 方向门控和参数量。

一个 hash-frozen permutation 承担正式 comparison；另做7个只重新拟合 scalar strength 的 permutation sensitivity，展示 null 分布。它回答任意三角非正规结构是否足够，还是必须与患者传播轴对齐。

M3 的科学结果拆开报告：预测、固定时程扩展、完整事件长度、有限时程 gain、峰值后变化、无输入返回。长期不返回但数值有限时，记为 `FINITE_NONRETURNING` 科学结果，不当作工程失败。

## 6. 训练实验：模型能力与成分作用分开

### 6.1 Primary：anchored joint fine-tuning

每一层从上一层 warm start，共享参数允许用较小学习率调整：

\[
L=L_{task}+\lambda_{anchor}
\|\theta_{shared}-\theta_{previous}\|_2^2.
\]

`lambda_anchor`、shared/new parameter learning-rate ratio 和最大 drift 在 design patients 上冻结，不根据全队列阳性结果修改。该训练回答模型 family 的最佳受约束解释能力。

### 6.2 Ablation：component isolation

在6–8位 design patients 中保留上一版冻结式训练：M1只释放 axis，M2只释放 direction，M3只释放 `gamma`。它回答单独加入一个成分时会发生什么，但不承担模型 family 主比较。

全队列不再复制一套 isolation training。对每个 joint checkpoint 做 `component-isolation replay`：恢复上一层共享参数，只保留本层新增 motif 参数，并只在 split 1 重新校准 decoder。必须同时保存 joint fit、isolation replay 和 parameter drift。若 joint 阳性、isolation replay阴性，解释为 motif 需要与局部尺度共同适配；若相反，则优先审计优化和过拟合。

### 6.3 Teacher forcing 与 self-feeding

正式主训练为 teacher forcing。3-step sampled self-feeding 只在6–8位 design patients 上作为训练 sensitivity，比较所有模型的：

- one-step calibration；
- fixed-horizon rollout；
- full STOP rollout；
- event-length calibration；
- seed stability。

self-feeding 不作为启动或停止 gate。是否扩展全队列必须在 design experiment 后单独修订资源合同并由用户审阅。

## 7. Stochastic rollout：宏观模板和微观方差

### 7.1 Monte Carlo 分层

- 所有 model-unseen events：32次 rollout；
- 每位患者在看结果前 hash-stratified 抽取20–30个 reference events：扩展到128次；
- 只有 Monte Carlo standard error 仍超出冻结阈值的 reference events 才增加到256次；
- 所有模型、对照和未扰动/扰动分支使用 common random numbers。

### 7.2 固定时程与完整事件分开

每个 prefix 同时运行：

1. `FIXED_H3/H5`：忽略 STOP，统一生成未来3步和5步；
2. `FULL_STOP`：保留 STOP、cardinality、repeat mask 和最大长度。

固定时程主要评价方向、传播范围和 contact field；完整事件再评价总长度和终止。如果 M3 只改善 FULL_STOP 而不改善 H3/H5，优先解释为 STOP/memory 效应。

### 7.3 两种终点

同时报告：

\[
r_{last}=\text{最后一个 rank set 的质心},
\]

\[
r_{late}=\frac{\sum_{k\in\text{last 20\%}}w_k\bar r_k}
{\sum_{k\in\text{last 20\%}}w_k}.
\]

last 20% 至少包含一个 rank set，主设置使用等权 `w_k=1`；rank-set size weighting 为 sensitivity。`r_late` 承担宏观终点，`r_last` 保留微观终止 contact 信息。

### 7.4 评价量

每条真实事件和生成事件转换为：

\[
S=(r_{last},r_{late},L_{axis},L_{orth},N_{rank},N_{contact},f_{contact}).
\]

报告：

- next-rank、STOP、cardinality NLL/calibration；
- endpoint–spread–length multivariate energy score；
- contact-field energy score；
- train-only TA/TB probability Brier/log score；
- observed coverage；
- 生成 covariance 与 held-out covariance 的 eigenvector/eigenvalue alignment。

模型不需要逐 contact 复制真实事件。主目标是恢复条件均值、模板身份、终点、范围和模板内方差。

## 8. 扰动：先编辑可观测 prefix，再分析 hidden dynamics

### 8.1 输入空间 counterfactual 是 primary

所有替换候选只用 split 0 建库，并匹配 contact 数、shaft relation、空间距离和相似 prefix support。不可匹配时保留患者并报告实际 denominator，不用 latent 扰动补齐。

#### A. 轴向 contact substitution

将第二 rank 中一个 contact 替换为：

- 沿 `+u` 最近且训练中可行的 contact；
- 沿 `-u` 的匹配 contact；
- 正交方向、距离和 shaft 匹配的 contact。

重新编码完整 prefix 后做 stochastic rollout。它检验小的、可实现的早期空间差异是否改变后续方向和宏观终点。

#### B. Tie-set 与局部顺序编辑

对相邻 ranks 执行：

- 顺序交换；
- 合并为 tie；
- 将 tie 拆成两个训练中出现过的可能顺序。

它检验微观顺序变化是否仍留在同一宏观模板盆地。

#### C. 传播范围编辑

在方向已经稳定的 prefix 中，增加或删除一个沿轴中段、训练中可支持的 contact input。比较模式身份、固定时程 spread、完整长度和 terminal field，区分方向选择与模板内范围调节。

### 8.2 Latent/Jacobian analysis 是第二层

在同一 reference states 上计算：

\[
P_H=J_{k+H-1}\cdots J_k,
\qquad
v_1^{(H)}=\arg\max_{\|v\|=1}\|P_Hv\|.
\]

沿 `±v1` 做小剂量 hidden perturbation，并与 norm-matched random、phase-shuffled 和 immediate-output-matched directions 比较。它用于解释 M3 的 finite-time gain，不把 hidden direction 当生理刺激坐标。

双重解离是可报告结果，不是 gate。任何一条腿单独成立，都按对应效应报告。

## 9. Synthetic identifiability 不是二元 gate

toy system 扫描：

- motif strength：`eta/beta/gamma`；
- contact 数和 shaft-like sampling；
- event 数；
- rank 数；
- observation noise；
- tie-set 大小；
- STOP variability。

输出每种设置下 M0/M1/M2/M3 的 recovery probability、bias 和 confusion matrix，形成可辨识性相图。真实数据阴性必须结合其所在的 synthetic power region 解释。

只有实现真值检查失败才是工程问题，例如 zero-equivalence 不成立、label 泄漏、sampler 读未来或 synthetic generator 与声明方程不一致。低信号条件不能恢复 motif 是预期科学结果，不阻止真实数据分析。

## 10. Seizure reuse：静态增量与动态方向拆开

### 10.1 S1：静态增量复用

现有0–10 s、1–150 Hz静态 field 可以回答较弱但明确的问题：

\[
\boxed{
\text{发作早期场是否包含静态 participation 无法解释、但 IED motif 可以解释的增量空间成分？}
}
\]

主静态场 `m_p` 固定为 split 0 的 start-removed mean contact-participation field；由 `STATIC_READOUT` 在 train-only起点分布上积分得到的期望场作为 sensitivity。对每条 train-only IED rollout field 做：

\[
f^{res}_{pe}=(I-\Pi_Z)f_{pe},
\qquad
Z=[1,m_p,\text{shaft/geometry covariates}],
\]

再从 residual rollout fields 中构造最多二维、完全 target-free 的 basis：

\[
U_p=\operatorname{PCA}_{1:2}\{f^{res}_{pe}\}.
\]

对每次真实 onset 或 same-block pseudo-onset 的标准化能量场 `y_s`，使用同一个 `Z` 比较：

\[
\mathcal S_0:y_s=Z a_s+\epsilon_s,
\]

\[
\mathcal S_1:y_s=Z a_s+U_pc_s+\epsilon_s.
\]

主指标为 leave-one-shaft-out；shaft 数不足时用预冻结 leave-one-contact-out：

\[
\Delta E_s=
CVError(\mathcal S_0)-CVError(\mathcal S_1).
\]

比较真实 onset 与同一患者、同一 recording block 的 pseudo-onset `Delta E`。这直接检验 IED motif 在静态 susceptibility 之外的增量解释。

### 10.2 A/Q 是直观辅助量

对正向 standardized log-power increment `y_s^+=max(y_s,0)`，报告：

\[
A_s=\|U_p^Ty_s^+\|_2^2,
\qquad
Q_s=\frac{\|U_p^Ty_s^+\|_2^2}
{\|y_s^+\|_2^2+\epsilon}.
\]

A/Q 用患者级二维联合置换检验和完整分量展示，不再要求两者分别显著才允许继续解释。主证据仍是相对静态模型的 `Delta E` 或交叉验证 likelihood gain。

### 10.3 S2：动态分支复用

若 time-resolved BB150 可靠性允许，再独立运行：早段估计方向系数，锁定后预测晚段 field。S2 不可用不阻止 S1，也不把 S1 改写成方向性复用。

Seizure extractor、same-block pseudo-onset 和频带 QC 可以与 interictal 训练并行开发；但 cohort scoring 只能在 interictal motif、static field、residual basis、维数和所有 scorer hashes 冻结后运行。

## 11. 平行 Goal 与 evidence matrix

G0–G6 全部在工程实现正确后运行，不串联停止：

| Goal | 核心问题 | 首要比较 | 主要输出 |
|---|---|---|---|
| G0 frame/layout | 结果是否依赖坐标或植入几何？ | parent vs geometry；free axis vs layout axis | frame dependence、layout-explained fraction |
| G1 anisotropy | 是否存在超越布局的走廊？ | M1 free−layout；layout−M0 | distribution effect、axis stability |
| G2 direction | 早期状态是否选择传播方向？ | M2−kinematic；M2−event-vector；M2−M1 | direction/endpoint effect、gate-emergence curve |
| G3 feedforward | 是否需要轴向前馈，而非更强、更慢或更多耦合？ | M3−gain/memory、symmetric、shuffled | fixed-H spread、length、gain time course |
| G4 variability | motif 是否解释模板稳定和细节随机？ | model vs observed covariance；input counterfactuals | mean/covariance、basin stability、extent modulation |
| G5 seizure | seizure 是否含静态场之外的 IED motif？ | static+IED vs static；real vs pseudo | `Delta E/DeltaNLL`、A/Q、动态预测 |
| G6 residual mechanisms | 是否残留低秩或距离–lag结构？ | recurrent vs readout residual rank；local-path lag residual | exploratory spectrum/sidecar，不训练 M4/M5 |

每个 Goal 分别填写以下证据列：

```text
predictive effect
distribution effect
dynamical signature
alternative-control comparison
observable perturbation consequence
denominator / CI / uncertainty
```

状态使用 `SUPPORTED / PARTIAL / NOT_DETECTED / UNDERPOWERED / NOT_IDENTIFIABLE`，不把不同问题合并成一个总分。

## 12. 统计合同

- 所有正式比较为 paired patient-level effect；
- 报告患者中位数、bootstrap 95% CI、正负号数和完整效应分布；
- 同一 Goal 内预注册 family 做 Holm correction；
- design patients 的实现比较明确标记 exploratory，不与 model-unseen cohort p-value混合；
- mode、endpoint、spread、length 和 covariance 全部报告，不运行后只挑最显著的一项；
- Monte Carlo uncertainty 与 patient variability 分开；
- 1D患者只进入可定义的预测/长度指标。

## 13. 正式资源合同

### 13.1 四个主模型

```text
28 patients × 4 main models × 3 seeds = 336 units
```

### 13.2 三个M3替代机制

```text
28 patients × 3 controls × 1 fixed seed chain = 84 units
```

控制模型全部从固定的 M2 seed chain 出发，不根据 validation 胜负挑 seed。总正式预算：

\[
336+84=420\ \text{units}.
\]

`LAYOUT_AXIS_ANISOTROPY`、kinematic/event-vector baselines、scalar shuffled-order sensitivities 和 static baseline 不计为完整 RNN unit，但仍需独立 manifest 和结果归档。Phase B 的 frame/operator/training design experiments 另列 pilot 资源，不伪装成420单元的一部分，也不进入正式 cohort inference。

## 14. Figure 6 与 Supplementary

正式结果出来前只冻结科学职责，不锁版面。

### Figure 6 候选主线

1. rank-set input、full-tissue RNN、随机 contact-set output；
2. 同一患者真实事件与随机 rollout 的模板均值和模板内方差；
3. M0→M1→M2→M3 对终点、范围和长度分布的影响；
4. free axis 相对 implantation-layout axis；
5. 可观测 prefix 的方向替换、局部顺序编辑和范围编辑；
6. M3 与 gain/memory、symmetric、axis-shuffled controls；
7. 若G5有信息量，展示 static-only 与 static+IED motif 的真实 onset/pseudo-onset增量。

每个结果 panel 优先采用直观 case + cohort statistic。图中文字使用白话，不用内部 adjudication 术语替代数据。

### Supplementary

- 旧 L0/L1/L2m/L3/C-suffix连接结构和 selected-nonlocal阴性结果；
- parent-frame与geometry-frame详细比较；
- orthogonal M2和normal M3N；
- self-feeding、component isolation、gain matching；
- full identifiability maps；
- Jacobian、hidden perturbation、decoder calibration；
- G6 low-rank与distance–lag sidecars；
- seizure A/Q、频带和空间 null sensitivities。

## 15. 仅保留的工程阻断条件

只有以下错误阻止对应运行：

1. split、event identity、contact join 或 provenance 错误；
2. TA/TB、未来 suffix、真实未来 cardinality 或 seizure target 泄漏；
3. `eta=0/beta=0/gamma=0` 数值等价失败；
4. sampler 重复 contact、STOP不吸收、closed-loop replay或RNG/resume不一致；
5. NaN/Inf、shape/device错误、checkpoint损坏或无法执行。

以下均为科学结果，不是工程 gate：

- frame effect不同；
- toy弱信号不可恢复；
- 轴不稳定；
- M2或M3不改善；
- gain没有峰或峰后不回落；
- 输入扰动没有选择性效应；
- S2动态数据不可辨识。

有限但长期不回落的模型必须保留并标记，不允许仅因不符合“有界瞬态”预期而删除。

## 16. 允许与禁止的解释

### 可以按证据逐层写

- 超越 implantation-layout baseline 的患者条件性各向异性；
- 早期位移提供方向惯性，或方向性 RNN 提供其上的增量；
- 轴向前馈 motif 额外解释固定时程扩展、事件长度或两者；
- 可观测 prefix 变化对应真实模板身份或模板内范围变化；
- seizure field 含有静态 participation 无法解释的 IED motif 成分。

### 不能写

- RNN恢复了真实 connectome、白质束或突触连接；
- geometry PCA1或自由轴就是病理解剖通路；
- M2非对称核证明了真实局部单向连接；
- M3阳性证明所有IED都由一般性非正规动力学产生；
- 大 singular value等于临界、不稳定或发作机制；
- 单条 rollout 相似等于重放真实事件；
- seizure与IED空间相似等于同一通路被重新调用；
- events、views、seeds或rollouts增加了患者样本量。

## 17. 本轮需要用户审阅的决定

1. 是否接受 `FRAME_EXPERIMENT`，而不是预先宣布唯一主 frame；
2. 是否接受局部非负 M2 为主、orthogonal rotation降为 sensitivity；
3. 是否接受 M3 的 gain/memory、symmetric、axis-shuffled 三类正式控制，M3N降为补充；
4. 是否接受 anchored joint fine-tuning 为主、component isolation为 ablation；
5. 是否接受 teacher forcing 为正式训练，self-feeding只做 design experiment；
6. 是否接受输入空间 counterfactual 为主扰动、hidden/Jacobian为第二层；
7. 是否接受 seizure S1 的 static-vs-static+residual-IED增量作为主 endpoint，A/Q为辅助量。
