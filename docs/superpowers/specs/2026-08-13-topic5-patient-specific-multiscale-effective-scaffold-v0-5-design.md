# Topic 5.1 患者特异多尺度有效传播 scaffold v0.5

> 状态：**SCIENTIFIC CLOSEOUT COMPLETE。A–H 全流程完成：531/531 正式训练单元、Stage F target-free field/attenuation freeze、
> 17 人/167 seizures locked internal benchmark、Figure 6 与 machine closeout audit 均完成；
> 仅余 commit/push 与主线 figure registry 集成。** 只承接已冻结的
> `topic5-lbss-full-tissue-v0.3-closeout`，不改写 v0.3 数值，不扩 architecture zoo。
> v0.5 是 target 已在项目历史中看过后的 `locked internal mechanistic follow-up`，不是
> independent validation、prospective confirmation 或 unseen-test confirmation。

## 1. 唯一核心问题与三级证据

v0.3 已证明 full-tissue recurrence 和真实间期顺序有预测价值，但没有证明 task-selected
nonlocal topology 普遍优于 matched controls。v0.5 不再问“哪张拓扑人人最好”，而问：

\[
\boxed{
\text{患者越偏离局部传播，task-selected nonlocal shortcuts 的增益是否越大？}
}
\]

### 1.1 Primary：target-free nonlocality interaction

在自动 census 冻结的 28 位 full-tissue spatial 患者上，以完全 heldout 的间期 distal contact prediction 为
primary。令高值表示更好：

\[
M^{II,distal}_{p,m}=-\operatorname{NLL}^{contact\mid continue,K}_{p,m,distal},
\]

所有 arms 使用相同 decisions、candidate masks、continue 条件和 next-set cardinality。定义：

\[
\Delta^{II}_p=M^{II,distal}_{p,L3}-M^{II,distal}_{p,L2m}.
\]

唯一 primary test 为：

\[
\boxed{\rho_S(J_p^{lat},\Delta^{II}_p)>0.}
\]

`L2m` 是从头训练的 macro-matched random-nonlocal control，定义见 §5。全 transition NLL、
free rollout 和 L3−L0/L1 是 secondary，不替代这个 primary。

### 1.2 Key information control：suffix order

\[
M^{II}_{L3}>M^{II}_{C\text{-suffix}}.
\]

它回答真实 prefix–suffix association 是否有信息，不回答 nonlocal topology specificity。除全体
heldout events 外，重点检验 RNN 相对 train-only prefix-template 的优势是否随连续 prefix
uncertainty 增大；ambiguous-prefix subset 只作直观展示。

### 1.3 Locked cross-state extension

在 Figure 3 的 17 位患者/167 次 seizures 中，定义：

\[
\Delta^{EI}_p=C^{EI}_{p,L3}-C^{EI}_{p,L2m},
\qquad
\boxed{\rho_S(J_p^{lat},\Delta^{EI}_p)>0.}
\]

这是 early-ictal family 的唯一 primary interaction。它是锁定后的内部跨状态 benchmark，不是
重新获得的独立确认。L3−C-suffix、L3−L0/L1 和其他 field metrics 都是预定义 secondary 或
robustness。

## 2. 术语与解释边界

允许：

- `patient-specific effective propagation scaffold`；
- `local recurrent backbone`；
- `task-selected nonlocal shortcut`；
- `event-internal relative centroid latency`；
- `early-ictal broadband energy field correspondence`。

禁止：

- anatomical、synaptic 或 white-matter connectivity；
- 正/负 RNN 权重等同兴奋/抑制；
- `event_lag_raw` 等同临床 recruitment 或轴突传导时延；
- 0–10 s broadband energy 等同 contact arrival order 或 ictal-core recruitment；
- RNN 恢复了真实 connectome。

所有 recurrent edges 在 ordinal rank update 内生效，模型不估计 conduction velocity 或 physical
axonal delay。

## 3. 冻结 parent result 与 cohort

### 3.1 v0.3 不再改变

- 21 人/31 fits、5 arms、3 seeds，465/465 正式单元完成；
- local-only 相对 no-recurrence 为 20/21 改善；
- L3 相对原 order-shuffle 为 21/21 改善；
- L3 未普遍优于 L0/L1/旧 L2；
- 12 人/141 seizures 的 L3 early-ictal canonical field 为正向内部对应，但无 nonlocal 特异性。

### 3.2 自动 cohort builder

正式 builder 必须对整个 masked-rank K=2 parent cohort 自动应用 `min_joint_contacts=6`，而不是只
手工恢复已知的 5 人。每位患者写出 inclusion/exclusion reason，并证明没有其他满足合同的患者被
遗漏。自动 builder 已在未读取 target values 的条件下冻结正式分母：

```text
parent K=2 cohort: 34 patients
spatial cohort: 28 patients / 42 fits
existing v0.3 overlap: 21 patients / 31 fits
new/recovery: 7 patients / 11 fits
early-ictal intersection: 17 patients / 167 seizures
```

新增于旧手工 recovery 清单之外的 `epilepsiae_583` 与 `yuquan_zhangjiaqi` 必须保留。自动几何
QC 将 `epilepsiae_139` 与 `yuquan_zhangjiaqi` 标为近一维；两者进入完整 census，另做去除
`DEGENERATE_ONE_DIMENSIONAL` 的固定 sensitivity，不因结果方向事后排除。

dataset、plane、H、split、`event_lag_raw` sidecar 均记录路径和 SHA256。

### 3.3 小 montage 观测性 QC

每 fit 输出：

- `effective_rank(H)` 与每 contact 的 H-support concentration；
- contact convex-hull coverage；
- prefix-to-contact distance spread；
- local-wave design matrix condition number；
- exact spatial permutation count；
- 6–7 contacts 与 >=8 contacts 分层标签。

shaft/exact permutation 覆盖所有合格患者；spectral/variogram null 只覆盖达到最低几何自由度者，
不得强行在 6-contact montage 上产生伪精确 null。

### 3.4 几何状态

```text
GEOMETRY_STATUS = RETROSPECTIVE_TEST_INFORMED_PROPAGATION_PLANE
EDGE_TIME_STATUS = ORDINAL_NO_PHYSICAL_DELAY
```

## 4. 模型与不变项

主 cell 不变：state-dim=1 leaky full-tissue RNN，contacts 只通过冻结 `H^T/H` 注入和读出；没有
dense contact-to-contact bypass。

| Arm | 固定 local backbone | Added edges | 作用 |
|---|---|---|---|
| L0 | 是 | 无 | local recurrence 基线 |
| L1 | 是 | K learned extra-local | 等容量较近边控制 |
| L2m | 是 | K macro-matched random nonlocal | primary topology null，必须重训 |
| L3 | 是 | K task-selected nonlocal | 主模型 |
| C-suffix | 与 L3 相同 | 在 suffix-pairing null 上选择 | key information control |

旧 v0.3 `L2 fixed random nonlocal` 只作 sensitivity，不承担 v0.5 primary。

所有 arms 共用：tissue nodes、H、local mask、K、cell、loss、decoder、optimizer、训练预算、splits、
checkpoint eligibility 和 seed registry。Local mask 必须使全部 H-supported nodes 位于同一个强连通
分量，pairwise directed reachability=1，minimum in/out degree>=1。

## 5. L2m：正式 topology control

### 5.1 从头训练，不用 frozen rewiring 代替

对每个 fit/seed，在完全 target-free 条件下读取该 seed 的最终 L3 added mask，构造一个新的
directed nonlocal mask，精确匹配：

- added-edge count K；
- 每 node added in-degree 与 out-degree；
- reciprocity count；
- 完整预冻结 distance-bin counts；
- nonlocal candidate constraint；
- 初始 added-weight distribution。

只随机破坏 source–target pairing。每个 model seed 使用独立 `graph_null_seed`，从头初始化权重，
使用与 L3 相同预算和 mask-freeze 后 checkpoint rule。匹配算法、尝试次数和不可行原因必须在训练前
写入 manifest；不得看 heldout labels 或 early target 后放宽条件。

Distance bins 使用该 fit 的 nonlocal candidate pool 在 train-only geometry 上冻结的 deciles。构造器
最多运行 100 个独立 restarts、每个 10,000 次 degree-preserving directed double-edge swaps；只有
同时满足 exact degree、reciprocity 和 bin counts 且 mask 不等于 L3 时才合格。全部 restarts 失败的
fit 标为 `GRAPH_NULL_NOT_CONSTRUCTIBLE`，保留 L3 描述结果但不进入 L3−L2m primary；不得事后合并
distance bins。Primary 报告全部 28 人 census 与实际 J/L2m identifiable denominator，并并列报告
去除两位近一维几何患者的预定义 sensitivity。

为避免只交换极少数 edges 的 cosmetic null，另冻结 `pairing_disruption_fraction >= 0.25`。该阈值只
要求四分之一 source–target pairs 被破坏，不替代上述 exact macro matching；若 exact constraints 下
达不到该阈值，同样标记 `GRAPH_NULL_NOT_CONSTRUCTIBLE`，不得放宽 degree、reciprocity 或 distance bins。

训练后 frozen macro rewiring 仍可作为 causal perturbation，但**不是** topology-selection null，也
不能决定是否补训 L2m。

### 5.2 L1/L3 candidate capacity 只审计，不在本轮重定义

输出 pool size、per-source/target candidate counts、unique candidates activated、exposure fraction
和 proposal frequency。若存在严重不平衡，L3−L1 不作机制主张；本合同不在运行中改写 L1/L3 pool。
任何完整 rematch 是另一个预估 507-unit 合同，不得由 v0.5 结果触发。

“严重不平衡”预先定义为：任一 H-supported node 仅在一个 arm 中有可选 added edge，或两个 pools 的
总 exposure fraction 比值落在 `[0.5,2.0]` 之外。该判据只约束 L3−L1 的解释，不影响 L3−L2m primary。

## 6. Train-only modes、template 与 suffix null

### 6.1 Mode discovery 必须 cross-fit

每个 outer training fold 内重新拟合 K=2 TA/TB templates；heldout suffix 不参与 clustering、template、
prefix posterior、mode-specific local-wave slope 或 flow bundle。跨 fold label alignment 只比较 train
templates。全数据 TA/TB labels 仅作 descriptive phenotype。

TA/TB labels 不再筛选任何 arm 的 train/validation/test events。`own_a` 与 `own_b` 是两种冻结
retrospective geometry views；两者都训练并评价该患者的全部合格事件。train-only modes 只用于
template baseline、prefix uncertainty、mode-specific flow 和 slope 分层。这样既不读取 heldout suffix，
也不会因后段缺少某一 mode 制造空 validation/test denominator。

这里必须区分两个不同聚合对象。Oracle repertoire candidates 对 shared 患者来自同一 fit 的两个
train-only modes；对 non-collinear 患者继续保留 `own_a` 与 `own_b` 两个 all-event geometry fields，
直到 per-seizure best-mode 评分后才聚合。Non-oracle train-prevalence mixture 则必须按 **mode** 聚合：
从 `own_a` 只取与 A 对齐的 train-only mode、从 `own_b` 只取与 B 对齐的 train-only mode，再按
train-only prevalence 加权。禁止把两个 all-event geometry fields 直接当成两个 mode components。
两条 own views 的 mode labels/templates/counts 必须逐位一致；修复 mixture 时四个 oracle A/B vector
hash 必须不变。

### 6.2 Primary suffix-pairing null

对每位患者、每个 outer fold，在相近 event length、missing-mask 和 mode-prevalence strata 内，将真实
suffix 在事件之间作 deranged reassignment：

- prefix 前 3 rank sets 原样保留；
- suffix tie blocks、contact-position marginal 和自然 suffix sequence 原样保留；
- 破坏 prefix–suffix pairing；
- 找不到 donor 时才使用 tie-block-preserving within-event derangement；
- T 太短不能破坏的事件原样保留并计数。

预生成 3 份确定性 mappings，每个 model seed 对应一份；`model_init_seed`、`suffix_null_seed` 和
`graph_null_seed` 分开记录。报告 effectively shuffled fraction、无法破坏比例、suffix-position
frequencies、pairwise precedence、mode prevalence、tie-block sizes 和 Kendall distance。

### 6.3 Prefix-template baselines

train-only prefix posterior 输出：

1. posterior-best TA/TB suffix template；
2. posterior-weighted template；
3. train-mode-prevalence mixture template。

这些模板同时进入 interictal 和冻结后的 early-ictal benchmark。连续 prefix uncertainty：

\[
H_e=-\sum_m p(m\mid prefix_e)\log p(m\mid prefix_e)
\]

是 primary shortcut diagnostic；检验 `RNN-template advantage ~ H_e`。二值 ambiguous subset 只作图。

### 6.4 时间块敏感性

保持 v0.3 retrospective event split 以保证主比较可比；另做 recording block/session-heldout
sensitivity，并明确它不替换原 split。

## 7. 患者 nonlocality index J

### 7.1 两种距离不能混用

Distal contact 的 primary 定义使用 prefix contacts 到新 contact 的直接二维前沿距离：

\[
d^{front}_{ei}=\operatorname{median}_{c\in N_{e,t+1}}
\min_{j\in S_{e,t}}\|r_c-r_j\|_2,
\]

且 `d_front > r_local`。Local-wave regression 使用 local graph path length。不得把单边 nonlocal
threshold 与累积 path length 当作同一个量。

训练单元为兼容 v0.3 仍可写出 train-distance q50/q80 三分箱，但它们只作描述性诊断。正式
`Delta_II` 和 attenuation double-dissociation 必须从每单元保存的相同 heldout decision rows 重新按
`d_front <= r_local` 与 `d_front > r_local` 二分；所有 arms/seeds 的 `(event, rank, distance)` support
hash 必须逐 fit 相同。每位患者至少 20 个 nonlocal heldout decisions 才进入 interaction inference。

H support 先按权重保留累计质量前 90%；primary path distance 使用支持对距离的 weighted 10th
quantile，raw minimum 仅作 sensitivity，避免极小 H 长尾权重制造虚假近路。

### 7.2 Cross-fitted local-wave baseline

`event_lag_raw` 每事件减去最早有限值，形成 `t'`。对 outer test fold f，`J_p^{(-f)}` 完全来自
非 f 事件：在非 f events 内再做 inner cross-fitting，每个 inner validation event 的 residual 来自
未见该 event 的 baseline。这样 primary `Delta_f` 与 J 不共享 test-event noise。

当前 v0.3 只有一个固定 heldout test split 时，primary J 只由 model-train events 内部的 5-fold OOF
residual 构成，正式 test events 完全不进入 J；若运行 block/session sensitivity，则每个 test block 的
J 只来自其他 blocks。

在 train-only K=2 modes 支持足够时拟合：

\[
t'_{ei}=b_e+\beta_{p,m(e)}d^{path}_{ei}+\epsilon_{ei},
\qquad \beta_{p,m}\ge 0.
\]

Heldout event 的 `b_e` 只由保留的前三个 prefix rank sets 估计为
`median(t'_ej-beta*d_path_ej)`；不得使用待评分 distal contact 的 latency 反推 event intercept。

mode event 数不足时退化为 patient-pooled `beta_p>=0`。残差尺度为 train-only robust MAD。若最优
beta=0，标记 `LOCAL_WAVE_UNSUPPORTED` 并保留；只有距离范围、finite latency、事件数不足或设计矩阵
退化时才标 `NOT_IDENTIFIABLE`。

Mode-specific slope 的最低支持为每 mode 20 个 train events 且至少 40 个 finite distal contact
observations；否则使用 pooled slope。`NOT_IDENTIFIABLE` 的固定门为：少于 40 个 finite distal
observations、front-distance 10–90% 范围小于 2 mm、或标准化设计矩阵 condition number >1e6。

### 7.3 Primary sparse early-arrival burden

\[
z_{ei}=-\frac{t'_{ei}-\hat t^{local}_{ei}}{\sigma_{p,train}},
\]

\[
\boxed{
J_p^{lat}=\operatorname{mean}_e
\left[\frac{1}{|D_e|}\sum_{i\in D_e}(z_{ei}-1)_+\right].
}
\]

它同时保留异常早到比例和幅度。原冻结的 event-median 版本在正式 RNN 训练前的 target-free
feasibility 中出现 28/28 患者精确为零，因此按 `J_ESTIMAND_PREFREEZE_REPAIR.json` 修订为 event-mean；
原 event-median 保留并明确标记为退化 sensitivity。10 个时间连续 block 的 burden-mean 再取中位数、
nonzero-event fraction、`1-tau_b` 和 near-late/far-early violation fraction 为 robustness。该修订发生在
任何 v0.5 RNN 结果和 early-ictal target 读取之前，不得再根据后续结果改变。

对 early-ictal interaction，J 使用全体 interictal events 的严格 out-of-fold residual 汇总，因为
early target 属另一状态；仍不得在拟合 J 时读取 early target。

## 8. 动力学与有效通路审计

### 8.1 Gain

除 `rho(W)` 和 `sigma_max(W)` 外，在 heldout trajectories 上计算：

\[
J_t=(1-\kappa)I+\kappa D_{\tanh'(z_t)}W,
\]

\[
G_K(t)=\max_{1\le k\le K}\|J_{t+k-1}\cdots J_t\|_2,
\qquad K=3.
\]

报告 median/p95 `G_K` 和标准化小扰动的 empirical output amplification。

Gain-adjusted sensitivity 预先定义为：对 seed-matched L2m/L3，取二者 validation median `G_K` 的
较小值为共同 reference；只对较高-gain arm 的 recurrent W 乘 `c in (0,1]`，用 validation-only
bisection 匹配 reference，再原样评价 heldout。它是 sensitivity，不替代 intact primary。

### 8.2 Dynamic flow

\[
\Phi_{ij}^{(m)}=
\mathbb E_{e\in m,t}
\left|\kappa\tanh'(z_i(t))W_{ij}h_j(t)\right|.
\]

同时保存 signed influence。TA/TB bundle attenuation 的 matched controls 必须匹配 edge number、
train-time total Phi、length、in/out degree、|W| 和 source/target endpoint density。只有 same-mode
损害大于 cross-mode 与 matched random 才支持 mode-specific effective routes。

### 8.3 Precedence 与稳定性

Pairwise precedence 只用共同出现次数达到 `max(5, ceil(0.01*n_train_events))` 的 pairs，并用
Beta-binomial shrinkage；
节点 q 为按 pair support 加权的平均，而非未经归一化的求和。Primary 稳定性对象是 contact-space
effective influence、source/target density field、distal reach 和 perturbation response；exact edge
overlap、raw weight 和 binary survival 只作 secondary。

### 8.4 Arm-specific attenuation

每个 arm 只 attenuate 自己实际存在的 active added edges：L1 的 extra-local、L2m 的 matched random
nonlocal、L3 的 task-selected nonlocal。固定：

\[
W_A\leftarrow(1-\alpha)W_A,
\qquad \alpha\in\{0.25,0.5,0.75,1.0\}.
\]

不得在 L3 中 attenuate 一个只存在于 L1/L2m 的 inactive edge。L3 内另抽 K 条 local-backbone edges，
匹配 total |W|、endpoint degree 与空间覆盖（不匹配 edge length，因为 local/nonlocal 正是被检验因素）。
对每个 target 以四档 dose-response AUC 为 primary perturbation summary，不为每个 alpha 单独追 P 值。

\[
S_p^m=\mathrm{AUC}_\alpha
\left[\Delta NLL_{distal}^{m}(\alpha)-\Delta NLL_{local}^{m}(\alpha)\right],
\]

并报告 L3−L2m、L3−L1 以及 L3 nonlocal-vs-local 的 double dissociation。Attenuated rollouts 与
fields 必须在 target unseal 前全部冻结。

## 9. Early-ictal broadband field 合同

Target 固定为 clinical onset 后 0–10 s、1–150 Hz broadband energy spatial field。它不提供 arrival
time。所有 RNN、template、mixture、attenuation、rewired fields 和 null index maps 必须在 target
unseal 前生成并冻结。

### 9.1 统一方向

将预测 rank 转为 earlyness：

\[
e_i=1-\frac{r_i-1}{N-1}.
\]

N=1 时该 endpoint 不可识别。Primary 不使用 `|rho|`。

N 是模型与 target 的 exact common valid contacts 数；ties 使用 midrank。所有 arms、templates 和 nulls
必须使用同一 common support，不能按各自生成 support 缩分母。

### 9.2 唯一 primary endpoint：oracle repertoire coverage

\[
C^{oracle}_{psa}=\max_{m\in\{TA,TB\}}
\rho_S(e_{pam},y_{ps}).
\]

它必须写作 `signed best-mode Spearman oracle repertoire correspondence`；maxAB 是 seizure target
选择 TA/TB candidate 的 oracle coverage，不是非 oracle 预测。

### 9.3 Key non-oracle endpoint

\[
\bar e_{pa}=\pi^{train}_{p,TA}e_{pa,TA}+\pi^{train}_{p,TB}e_{pa,TB},
\qquad
C^{mixture}_{psa}=\rho_S(\bar e_{pa},y_{ps}).
\]

`pi_train` 只由 interictal train events 决定。Prefix-template TA/TB fields 与 train-prevalence mixture
field 在 unseal 前一并冻结。对 non-collinear 患者，`e_TA/e_TB` 在该式中指 A/B-aligned train-mode
components，而不是未经 mode 分层的 own_a/own_b all-event oracle candidates；后者只用于 §9.2。

### 9.4 Robustness endpoints

- rank-weighted field concordance：令 observed energy 的降序 midrank 为 `q_i`、energy highness
  `v_i=1-(q_i-1)/(N-1)`，固定 `w_i=exp(-(q_i-1)/(0.2*N))`，对 `e_i` 与 `v_i` 计算 weighted
  correlation；
- predicted-early contacts 与 observed high-energy contacts 的 top-k overlap，其中
  `k=max(1,ceil(0.2*N))`；
- predicted earliest-field contact 到 observed peak-energy contact 的二维距离；ties 取各自 contact
  centroid 后计算距离；
- rank-normalized spatial Wasserstein distance：将 model earlyness 和 observed energy highness 归一为
  概率质量，以患者二维 contact distance/convex-hull diameter 为 transport cost。

不得再称 `early-weighted Kendall`、`earliest observed contact` 或 recruitment order。

### 9.5 Nulls

唯一 primary spatial null 是 synchronized all-contact permutation。每个 permutation 对所有 arms
同步，并重复相同 earlyness、best-mode oracle、missing-contact 和 patient aggregation。以下只作
robustness：shaft-preserving、distance-bin、spectral surrogate、variogram-matched surrogate；仅在
对应 montage QC 合格时运行。

Primary 使用 5,000 个固定、跨 arms 同步的 all-contact permutations；同一 seizure 的两个 mode
candidates 在每次 permutation 中共享同一 target permutation，且重新执行 best-mode oracle。

## 10. 统计单位与检验

聚合顺序唯一固定为：

```text
event/seizure metric
-> average within fit and model seed
-> aggregate seeds within fit
-> aggregate own-A/own-B fits within patient
-> patient-level inference
```

同患者 seizures、fits 和 seeds 绝不作为独立样本。

### 10.1 Primary target-free family

- primary：patient-level one-sided Spearman permutation test for `rho(J, Delta_II)>0`；
- 同时报 paired L3−L2m、bootstrap CI、正/负/并列数；
- patient bootstrap 每次同时重算 J 与 Delta；
- leave-one-patient-out slope、去除 6–7-contact patients、去除单个最高 J 患者为固定 sensitivity。

Patient-label permutation 使用固定 100,000 draws；若实际 eligible n 足以精确枚举则使用精确分布。

### 10.2 Information family

- L3 vs C-suffix all/distal；
- RNN-template advantage 与连续 H 的 interaction；
- ambiguous subset 只作可视化。

### 10.3 Early-ictal family

- 唯一 primary：`rho(J, Delta_EI)>0`，其中 C 是 signed best-mode Spearman oracle repertoire
  correspondence；patient-label permutation 检验患者级 J–增量关联，synchronized all-contact
  permutation 则在每个 draw 内重复 maxAB、患者内 seizure folding 和 L3−L2m，再形成该 interaction
  的 coherent spatial-null。两者是联合主判据，取两项单侧 P 的较大值，必须同时通过；
- non-oracle mixture、direct L3−L2m、L3−C-suffix/L0/L1 和四个其他 endpoints 为 secondary/
  robustness；
- 不为每个 endpoint/null 单独追逐星号；按预定义 claim family 报 Holm 或 simultaneous interval。

Early interaction 同样使用固定 100,000 次 patient-label permutation；seizure-level values 先在患者内
聚合，不进入 permutation 的独立样本单位。

Primary spatial null 使用冻结的 5,000 个 synchronized all-contact draws。对每一 draw，先在每个
seizure 内重做 A/B best-mode oracle，再在患者内逐 draw 取 seizure median，随后计算
`rho(J, null_L3-null_L2m)`。不得只减去各 arm 的 marginal null median，也不得在解封后选择两种
null 中较有利的一种。

## 11. 正式训练预算

自动全 parent census与旧 cache 逐位审计后，最小完整合同为 531 units。旧 20 个 `own_a/own_b`
fits 的 `keep` 曾使用全记录 adaptive-cluster labels，不能复用；11 个 shared fits 的 geometry、ranks
和 split 与 v0.3 逐位相同，可复用 L0/L1/L3：

| Scope | Arms | Units |
|---|---|---:|
| exact shared reuse 11 fits | C-suffix/L2m x3 | 66 |
| mandatory full retrain 31 fits | L0/L1/L2m/L3/C-suffix x3 | 465 |
| **total** |  | **531** |

旧 L2 保留 sensitivity。不得把 L2m 设成 frozen-rewiring 先阳性才运行的条件分支。若 L1/L3
candidate audit 严重不平衡，则删除 L3−L1 机制解释，不在当前 run 动态升级到 507 units。

## 12. 目标隔离、工程锁与完成定义

- 从明确 base commit 建 immutable execution worktree；launcher/model/trainer/scorer/builder 全部 hash；
- active scripts 复制到 `run_snapshot/` 后只读；
- target-free 阶段通过文件权限或独立 mount 使 energy values 不可读，仅允许 routing metadata；
- target unseal 后 source tree 不得变化，scorer 不得生成新 model field；
- best checkpoint 必须在 LR mask freeze 后；resume 保存 optimizer、RNG、mask、edge age、rewire counter；
- 0 unresolved OOM；所有 retry、nonfinite 和 excluded unit 有机器可读记录；
- aggregate 校验 producer code/config/input hashes 与 cohort revision；
- Figure source rows 可追溯到 patient table，figures 具有中文 README、600-dpi PNG、单页 PDF、SVG。

需要新增的承重测试：train-only modes 不读 heldout suffix；三类 seeds 独立；suffix mappings 不跨 split；
J cross-fit 无 self-event；H-support 90% 截断；L2m matching/refit 无 heldout label；signed/maxAB/null 使用同一
rule；A/B fits 不重复计数；small montage endpoint 不除零；target import/read hard-fail。

## 13. 结果解释与下一步

- `rho(J,Delta_II)>0` 且 attenuation distal-specific：支持患者特异 nonlocal effective shortcuts；
- 间期成立而 early interaction 不成立：只支持 interictal multiscale computation；
- L3≈L2m：任意 nonlocal capacity 足够，不支持 specific route；
- RNN≈prefix-template：任务主要是 early mode identification/template completion；
- 只有少量 state gain 才改善迁移：下一合同选 E2；
- susceptibility 解释超过 topology：下一合同选 E3；
- latency interaction 稳定：下一合同选 E1。

E1/E2/E3 只择一另立 spec，不在本合同中训练。任何结果都不允许把模型 shortcut 写成真实白质束。
