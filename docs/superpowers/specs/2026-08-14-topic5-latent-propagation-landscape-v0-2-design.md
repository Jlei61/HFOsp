# Topic 5.2 患者特异 latent propagation landscape 与 perturbation response v0.2

> 状态：**SCIENTIFIC SPEC — AUTHORIZED AND EXECUTED；2026-08-14 收口。**
> 执行结果：`docs/archive/topic5/latent_propagation_landscape_v0_2_closeout_2026-08-14.md`。
>
> ---
>
> ## ERRATUM（2026-08-15，不修改下方任何预注册文本）
>
> 下方 §7.9 / §7.10 之间存在一处**设计期就注定的符号冲突**，后续 spec 不得照抄：
> §4.2 复用 parent 的 **earlyness** producer（`1 − normalized_rank`，越大越早），因此
> §7.9 的 `v_data_prog` 是一张 earlyness 图；而 §5.5 的 `a_prog(s) = γ'(s)/‖γ'(s)‖` 指向**更晚** phase。
> 沿 `+a_prog` 扰动会抬高更晚触点的 logit，投影到 earlyness 图必然为负 ⇒ §7.10 预注册的
> `D_prog = R_prog←prog − |R_field←prog|` 在任何情况下都不可能为正。该 estimand 不可辨识，
> 本轮 C3 progress leg 与 C5 progress leg 的阴性判决因此**不构成对机制的证据**。
> 下一版必须先冻结 laterness（或 train-only phase-conditional common-field derivative）朝向。
>
> 另有两处执行缺口，已在收口 §14 记录：
> (a) §5.7 要求的 phase-shuffled / event-shuffled / high-variance-PCA transport controls 未实现，
> 只有 C-suffix arm 一族可用，且该族显示 transport 与 transverse contraction **不是 order-specific**；
> (b) §9.2 要求的 shaft / distance / autocorrelation 保结构 spatial nulls 已计算但未进入 C5 summary，
> 补齐后 progress 轴效应量由 `0.47` 降至 `0.18–0.20`。
>
> ---
>
> Parent：Topic 5.1 patient-specific multiscale effective propagation scaffold v0.5，
> `SCIENTIFIC CLOSEOUT COMPLETE`。
>
> 本阶段不重新训练主模型，不继续搜索 recurrent topology，不把 RNN weights 解释为解剖连接。
> 它由三个预注册 goal 组成；三者按数据依赖顺序执行，但**不以任一前序统计结果作为后序实验的
> 科学停止条件**。

## 0. 核心科学判断

Topic 5.1 已经说明：

\[
\text{真实 prefix-suffix association 有稳定总体信息，}
\]

但：

\[
\text{具体 static recurrent topology 在当前 SEEG 分辨率下不可辨识。}
\]

Topic 5.2 的核心假设改为：

\[
\boxed{
\text{有序 prefix 将冻结 RNN 写入一个由传播进程和未来传播场共同组织的低维状态空间。}
}
\]

换句话说：

```text
Static topology is not identifiable,
but a patient-specific dynamical computation may be.
```

本阶段不再把“是否存在 residual progress coordinate”当作整个科学问题，也不围绕 E3 建立统计停止树。
`E3_SMOOTH_SUSCEPTIBILITY` 只可能是 Topic 5.2 完成后，对已发现 control field 的低参数压缩模型；
不是 Topic 5.2 的预设终点。

## 1. 三个 goal

### Goal 5.2A：Latent propagation geometry and dynamical transport

回答：

> 冻结 RNN 如何把 partial prefix 编码成“事件进行到哪里”和“事件将趋向哪一种患者特异未来传播场”
> 两个连续坐标；局部 RNN dynamics 是否沿这些坐标运输状态，并约束偏离轨迹的扰动？

输出无论阳性阴性都必须为：

```text
LATENT_GEOMETRY_COMPLETE
```

### Goal 5.2B：Axis-specific state perturbation response

回答：

> progress 与 future-field coordinates 的局部扰动，是否分别改变未来传播进程和未来传播样式；
> empirical state transplantation 是否得到同方向结果？

输出无论阳性阴性都必须为：

```text
PERTURBATION_RESPONSE_COMPLETE
```

### Goal 5.2C：Patient-specific spatial control field and cross-model convergence

回答：

> RNN perturbation 产生的 finite-time functional response fields，是否优先匹配本患者的数据传播场，
> 并在输入资格允许时与本患者 SNN propagation/core fields 收敛；冻结后是否与 early-ictal field 呈探索性对应？

输出按实际输入资格为：

```text
SPATIAL_CONTROL_FIELD_COMPLETE
SNN_ALIGNMENT_COHORT_ELIGIBLE
SNN_ALIGNMENT_CASE_SERIES_ONLY
SNN_ALIGNMENT_NOT_IDENTIFIABLE
```

SNN 输入不足不能阻断数据场、spatial patch 或 early-ictal 分支。

## 2. 科学边界

本阶段最多允许形成以下 claim ladder：

1. task-aligned two-coordinate geometry；
2. model-internal propagation channel；
3. model-internal axis-specific control；
4. topology-convergent functional response；
5. patient-specific data-field alignment；
6. patient-specific cross-model convergence；
7. exploratory cross-state alignment。

即使全部成立，仍不能直接写：

- anatomical、white-matter 或 synaptic pathway；
- biological stimulation response；
- epilepsy-specific axis；
- TA/TB 是两个真实 attractors 或两条独立病理通路；
- early-ictal broadband energy 是 arrival/recruitment order；
- SNN 与 RNN 是统计独立的两份患者证据。

若 C1–C6 均成立，最强措辞是：

```text
patient-specific pathology-related recurrent state-control landscape
```

由于没有健康人或非病理事件对照，不使用 `epilepsy-specific`。

## 3. 冻结输入、索引与模型合同

### 3.1 分母和 checkpoint cells

复用 Topic 5.1 v0.5 的：

```text
patients: 28
fits: 42
arms: L0, L1, L2m, L3, C-suffix
seeds: 0, 1, 2
analysis checkpoint cells: 42 x 5 x 3 = 630
```

630 cells 包括：

- v0.5 新训练或正式调度的 531 units；
- 11 个 exact-reuse fits 上的 99 个 parent L0/L1/L3 units。

checkpoint resolver 必须记录 source、path、SHA256、parent commit/tag、config、split、H、node mask 和
decoder hashes。缺 cell 时不得复制另一 seed、另一 fit 或最近 checkpoint。

### 3.2 索引

| 符号 | 含义 |
|---|---|
| `p` | patient |
| `f` | patient 内 fit/geometry view：`shared/own_a/own_b` |
| `a` | arm：L0/L1/L2m/L3/C-suffix |
| `j` | seed |
| `e` | interictal event |
| `k` | event 内 ordinal rank-set step |
| `N_pf` | fit-specific tissue-node 数 |
| `C_pf` | fit-specific exact contact 数 |

\[
\mathbf h_{p,f,a,j,e,k}\in\mathbb R^{N_{pf}}.
\]

不同 fit 的 tissue-node indices 不是公共坐标，禁止逐 node 直接平均。跨 fit 比较必须先投影到 exact
common contacts，或进入冻结且通过审计的物理空间注册。

### 3.3 Hidden update

正式模型是 `state_dim=1` full-tissue leaky RNN。简化记号下：

\[
\mathbf u_k=(\mathbf x_kH)\odot\mathbf g_{in},
\]

\[
\mathbf h_k=(1-\kappa)\mathbf h_{k-1}
+\kappa\tanh(\mathbf u_k+W\mathbf h_{k-1}+\mathbf b),
\]

\[
\boldsymbol\ell_k=\mathbf b_{contact}+g_{out}\mathbf h_kH^\top.
\]

实际 tensor orientation、mask、dtype 和 decoder 以冻结实现及 checkpoint schema 为真值。

### 3.4 完整 decoder state

hidden state 不是完整 closed-loop state。定义：

\[
q_k=(\mathbf h_k,\mathbf r_k,k),
\]

其中 `r_k` 为已 recruited contact mask。STOP/cardinality decoder 还读取 hidden mean/max、
`k/(C-1)` 和 recruited fraction。冻结 closed-loop 必须写成：

\[
\widehat{\mathbf x}_{k+1}=G_\Theta(q_k),
\qquad
q_{k+1}=T_\Theta(q_k,\widehat{\mathbf x}_{k+1}).
\]

必须保存 repeat mask、STOP precedence、size decision、tie break、maximum-rank rule 和 absorbing STOP。
不存在可执行的 `G(h)` 简写；所有 perturbation branches 从同一个完整 `q_k` 分叉。

### 3.5 参数冻结

\[
\Theta=\{W,\text{node mask},H,\mathbf g_{in},\mathbf b,\kappa,
\mathbf b_{contact},g_{out},\text{STOP head},\text{size decoder},\text{decode rules}\}.
\]

所有分析中 optimizer 不存在，model 为 evaluation mode，perturbation 前后 parameter hashes 必须一致。

## 4. Split、target seal 与 field producer

### 4.1 Split

沿用每个 frozen checkpoint 的 chronological train/validation/test split，不重划原 RNN split。新增
manifold/decoder 的选择只使用 train/validation；test 只用于冻结后的最终评价。

若旧模型只有 train80/test20，则 train80 固定拆成：

```text
axis_train60
axis_validation20
heldout_test20
```

该 validation20 不是原 RNN 的独立验证集，只用于本阶段新增分析选择。

### 4.2 数据传播场

复用 v0.5 已接受的 event-to-field、train-only TA/TB alignment、mask、contact order 和 earlyness producer；
不另写一套 rank-field 规则。每个 event 的完整 contact-rank field 记为：

\[
\mathbf y_e\in\mathbb R^{C_{pf}}.
\]

primary 使用 start-removed event field，避免真实第一 rank 直接承重；canonical full field 为 sensitivity。
non-participating contact、ties 和 mask 的含义完全沿用 parent producer。

为避免构造与评价复用同一数据，字段分成两层：

- `axis-train fields`：只用于定义 `v_field/u_e`、符号和 C3 contact-space response axes；
- `heldout-test empirical fields`：只用于 C5 data alignment 和 identity null，不参与 axis、phase、dose、
  model 或 sign 选择。

heldout field 必须沿用 train-only mode alignment；禁止在 test events 上重命名 mode 或选择更高相关的
permutation。canonical TA/TB 命名只适用于 parent mapping 能在**同一个 fit/node space** 中把两个 train modes
一一映射到 A 与 B 的 `shared` fits。`own_a/own_b` 的两个 train modes 都属于同一个 geometry scope，只能称为
within-fit future-field variation；不得跨 `own_a` 与 `own_b` 的不同 node spaces 构造 hidden-state A↔B 轴。

### 4.3 Target seal

Goals 5.2A、5.2B 以及 5.2C 的数据场/spatial patch 部分禁止读取：

- early-ictal energy values；
- SOZ/PZ、resection、outcome；
- SNN response/core values。

SNN availability 只允许先读取路径、producer/hash、runtime、status 和 denominator metadata；field values
在 RNN/data response fields 完全冻结后才读取。early-ictal values 最后解封。

## 5. Goal 5.2A：两坐标 latent geometry

### 5.1 Primary state space 与 observable comparisons

分析顺序固定为：

1. **raw standardized hidden state manifold：主分析**；
2. observable-only baseline：comparison；
3. raw hidden + observables：incremental prediction；
4. observable-residualized hidden manifold：sensitivity。

显式 observables 为：

\[
o_k=\left[
\frac{k}{C-1},
\operatorname{mean}(r_k),
|x_k|,
x_k,
r_k
\right].
\]

`x_k/r_k` 是真实任务结构，不作为 nuisance 从 primary state 中删除。residualization 阴性不能否定 raw
manifold，只能说明 hidden coordinate 与当前 contact configuration 耦合。

在 incremental future-field decoder 中，先在每个 train-only phase bin 内拟合 `Z~O`，再将冻结的
`Z_resid=Z-O\hat B` 与 `O` 合并；validation/test 只应用 train 系数。这样 `O+hidden` 的增量项不由 raw
collinearity 承重。decoder dimension/ridge 在对应 phase 的 validation bin 内冻结，负 heldout `R2` 原样保留，
不得截断为零或按效果重选。

### 5.2 Progress phase

primary phase 改为 normalized rank-step：

\[
s^{step}_{e,k}=\frac{k-1}{K_e-1}.
\]

contact-cumulative phase 只作 sensitivity：

\[
s^{contact}_{e,k}=
\frac{\sum_{q=1}^{k}|x_{e,q}|-|x_{e,1}|}
{\sum_{q=1}^{K_e}|x_{e,q}|-|x_{e,1}|}.
\]

两者都是离线 geometry labels，不进入 RNN 或 decoder，也不解释为秒、传导速度或临床 recruitment。
STOP/remaining length 不再作为 progress geometry 的主要功能验证。

`K_e<2` 的 event 无法定义 `s_step`，从 progress-geometry branch 排除并记录，不作零填充；它仍可在不需要
phase 的描述性 field inventory 中保留。

### 5.3 Continuous future-field coordinate 与 identifiability tiers

在每个 `(p,f)` 的 axis-train events 内先冻结两个 train-only mode centroid fields：

\[
\overline{\mathbf y}_{m0},\qquad
\overline{\mathbf y}_{m1}.
\]

axis-train centroid 使用 parent 的 `full_train_mode`，不得误用只由前三 ranks 分配的 `prefix_mode`。validation/
test 的 continuous `u_e` 直接由完整 heldout event field 投影得到；binary mode sensitivity 如需标签，使用
train-only full-event centers 对 heldout full-event feature 的固定最近中心分配，并与 prefix-only assignment 明确
分栏，禁止把 prefix-derived label 当成独立 future outcome。

定义去均值、单位范数的患者特异 future-field axis：

\[
\mathbf v_{field}=
\frac{\operatorname{center}(\overline{\mathbf y}_{m+}-\overline{\mathbf y}_{m-})}
{\left\|\operatorname{center}(\overline{\mathbf y}_{m+}-\overline{\mathbf y}_{m-})\right\|_2}.
\]

event 的连续 future-field coordinate 为：

\[
u_e=\mathbf v_{field}^\top
\operatorname{center}(\mathbf y_e-\overline{\mathbf y}_{train}).
\]

方向与命名分两层冻结：

1. **canonical A/B tier**：仅 `shared` fits。parent train mapping 必须将 mode0/mode1 一一映射到 A/B；固定
   `m+=A, m-=B`，因此 `u_e>0/<0` 表示更偏向 train-defined A/B。
2. **within-fit mode tier**：`own_a/own_b` fits。固定 `m+=mode1, m-=mode0` 作为纯 train-only 数值符号；
   `u_e` 只表示该 geometry 内两个 train modes 之间的连续 future-field variation，不解释为 A↔B。

二值 mode label 只作 sensitivity/visualization，不是唯一状态形式。canonical TA/TB sensitivity、field-control
双重解离和 A/B 措辞只在 canonical tier 裁定；generic within-fit geometry/perturbation 可覆盖全部可辨识 fits，
但必须分别报告 denominator。`own_a/own_b` 只能在 contact-space response 已各自生成后做 patient-level 汇总，
不能把两个 fit 的 hidden vectors、tangents 或 state chords 拼接。

只有任一 train mode 缺失、没有共同 valid contacts、contrast nonfinite，或 standardized contrast norm
低于按 dtype/维度预冻结的 numerical tolerance 时，该 fit 标为 `FIELD_AXIS_NOT_IDENTIFIABLE`。事件分母小
或 split stability 差标为 `FIELD_AXIS_LOW_RELIABILITY` 并原样报告 uncertainty，不用任意 effect/P-value
threshold 删除；两者都不阻止 progress 或其他 fits 的分析。

### 5.4 两坐标状态模型

每个 `(p,f,a,j)` 在 event-first、phase-balanced weights 下拟合：

\[
\boxed{
\widetilde{\mathbf h}_{e,k}
\approx
\boldsymbol\gamma(s_{e,k})
+u_e\mathbf b(s_{e,k})
+\boldsymbol\varepsilon_{e,k}
}
\]

其中：

- `gamma(s)`：共享 progression trajectory；
- `b(s)`：相同 phase 下，对未来连续 field 的 state commitment direction；
- `epsilon`：未解释状态。

用 train-only weighted PCA 压缩到 `d in {2,4,8}`；PCA 只为估计和正则化，不把 PC 本身命名为机制。
cubic spline knots、dimension 和 smoothness 由 validation 选择，test 不改。

比较三个容量明确的模型：

```text
O: observable-only
P: gamma(s)
PF: gamma(s) + u*b(s)
```

PF 另有 `u_e` 在 event-length/participation strata 内 shuffle 的等容量 null。primary raw-state model 与
residual-state sensitivity 使用同一自由度预算。

### 5.5 两个 latent coordinates

下式中的 `gamma/b` 均指从 PCA/standardized estimation space 反投影到 raw hidden-state metric 后的曲线和
方向；禁止直接把不同 scale 的 PCA coefficient 当作 perturbation vector。

progress tangent：

\[
\mathbf a_{prog}(s)=
\frac{\boldsymbol\gamma'(s)}{\|\boldsymbol\gamma'(s)\|_2}.
\]

future-field direction 先去除 progress 分量：

\[
\mathbf a_{field}(s)=
\frac{\mathbf b(s)-
\mathbf a_{prog}(s)\mathbf a_{prog}(s)^\top\mathbf b(s)}
{\left\|\mathbf b(s)-
\mathbf a_{prog}(s)\mathbf a_{prog}(s)^\top\mathbf b(s)\right\|_2}.
\]

所有正交、norm 和 perturbation 在同一个 raw hidden-state metric 下定义。若 `b(s)` 与 progress tangent
近共线导致 denominator 低于 validation-frozen threshold，该 phase 只报告 `FIELD_AXIS_COLLINEAR`，
不强行构造 field axis。

坐标：

\[
\mathbf z_{e,k}=
\begin{bmatrix}
z_{prog}(\mathbf h_{e,k})\\
z_{field}(\mathbf h_{e,k})
\end{bmatrix}.
\]

`z_prog` 为到 progression curve 的局部坐标；`z_field` 为相同 phase 下沿 `a_field` 的连续投影。

### 5.6 Future-field commitment emergence

在每个 prefix step/phase 上，用当前 hidden state 预测冻结的 event-level `u_e`，比较：

```text
observable-only
raw hidden only
observable + raw hidden
residual hidden sensitivity
```

输出 heldout：

\[
R^2_{future\ field}(k),
\]

以及 Brier/AUC 的二值 TA/TB sensitivity。模型、regularization 和 step bins 在 validation 冻结。

关键比较：L0/L1/L2m/L3 与 C-suffix 的 early-prefix emergence curve。C-suffix 不要求缺少 progress axis；
它检验真实 prefix-suffix association 是否使 future-field commitment 更早、更强出现。

### 5.7 Teacher-forced local Jacobian

在真实下一 input 条件下：

\[
J_{e,k}=
\frac{\partial F_\Theta(\mathbf h_{e,k},\mathbf x_{e,k+1})}
{\partial\mathbf h_{e,k}}.
\]

不对离散 closed-loop decoder 求伪 Jacobian。

#### Tangent transport

\[
T_{prog}(s)=
\cos\left[
\mathbf a_{prog}(s_{k+1}),
J_{e,k}\mathbf a_{prog}(s_k)
\right],
\]

\[
T_{field}(s)=
\cos\left[
\mathbf a_{field}(s_{k+1}),
J_{e,k}\mathbf a_{field}(s_k)
\right].
\]

与 phase-shuffled axes、event-shuffled axes、high-variance PCA directions 和 C-suffix 对应方向比较。

#### Transverse contraction

令局部二维 tangent projector：

\[
P_{tan}(s)=
\mathbf a_{prog}\mathbf a_{prog}^\top+
\mathbf a_{field}\mathbf a_{field}^\top,
\qquad
P_\perp=I-P_{tan}.
\]

normal directions 从相同 phase、相近 observables 的 train-state local residual PCA 中取得，不在整个
`N_pf` 空间任意采样。定义：

\[
C_\perp(s)=
\operatorname{median}_{v\in\mathcal V_\perp}
\left\|P_\perp(s_{k+1})J_{e,k}v\right\|_2.
\]

同时报告 local tangent gain、normal singular spectrum 和 event-to-curve residual 在一步前后的变化。
只有 tangent transport 为正、`C_perp` 的 patient-level CI 支持绝对 gain `<1` 且低于 controls、同 phase
跨 event dispersion 下降三者共同成立，才称为 `model-internal propagation channel`。若只低于 controls 但
绝对 gain 不低于 1，只称 `relative transverse constraint`；不使用 basin/attractor。

### 5.8 Closed-loop empirical transition field

从完整 `q_k` 自由 rollout，将访问状态投影到 `(z_prog,z_field)`，估计：

\[
\mathbb E[\Delta\mathbf z\mid\mathbf z,o_k].
\]

它只称 empirical projected transition field，不称 `h` 上 autonomous flow。teacher-forced 与 closed-loop
分别报告，不用后者覆盖前者。

## 6. Goal 5.2A claim family

### C1：Two-coordinate geometry

独立报告：

1. P 相对 observable-only 的 heldout reconstruction/output fidelity；
2. PF 相对 P 与 shuffled-`u` PF 的 heldout增量；
3. real-order arms 相对 C-suffix 的 early-prefix future-field emergence；
4. raw-state primary 与 residual-state sensitivity。

以上结果必须同时给出 `(a)` all-identifiable-fit 的 generic within-fit tier 与 `(b)` shared-fit canonical A/B
tier；只有后者可承担“TA/TB future-field commitment”措辞。两层不互相替代，也不把 fit 当 cohort sample。

该 family 按预定义 endpoints 报 Holm，不以任何一项作为 Goal 5.2B 的统计准入门。

### C2：Dynamical transport

独立报告：

- progress/field tangent transport；
- transverse contraction；
- event-to-curve convergence；
- teacher-forced/closed-loop consistency。

C2 阴性不阻止 perturbation；它只禁止 `propagation channel/attracting trajectory` 措辞。

## 7. Goal 5.2B：两坐标 perturbation

### 7.1 共同原则

所有 perturbation：

- 在冻结 checkpoint 上运行；
- 从同一完整 `q=(h,r,k)` 分叉；
- 只改变 `h`，明确称 hidden-state intervention；
- `r,k` 保持不变；
- 不能声称完整 `q` 被移动到另一真实 phase；
- 只要该 axis 数值可定义且单次 perturbation 通过局部支持，就执行，不由 C1/C2 的 `P` 值决定。

reference phase 固定取每个 heldout event 最接近 `s={0.25,0.50,0.75}` 的合法 state，每 event/phase
至多一个；选择规则和 event IDs 在读取 perturbation response 前冻结。不得按 C1/C2 effect 或患者方向
重选 state。

### 7.2 Local progress-tangent perturbation

\[
\mathbf h^\pm=
\mathbf h\pm\epsilon\mathbf a_{prog}(s).
\]

primary dose 以相同 phase local residual SD 为单位：

```text
epsilon = 0.5 local SD
```

sensitivity：`0.25`、`1.0 local SD`。该 perturbation 只表示沿局部 progress direction 的小扰动，不称为
完整 state 前移到 `s+Delta s`。

对任一单位 raw-metric direction `a`，`local SD` 明确定义为：

\[
\sigma_a(s)=\sqrt{\mathbf a^\top\widehat\Sigma_{resid}(s,o_k)\mathbf a},
\]

其中 covariance 只由 axis-train/validation 的同 phase、相近 observables states 估计。若该值退化或邻居
不足，方向标记数值不可辨识，不改用全局 SD。

### 7.3 Continuous future-field perturbation

\[
\mathbf h^\pm=
\mathbf h\pm\lambda\mathbf a_{field}(s),
\]

primary `lambda=0.5 local SD`；`0.25/1.0` 为 sensitivity。正方向由 train-only `v_field` 的 TA 端冻结，
不能按 heldout response 翻转。

### 7.4 Empirical state transplantation

在 axis-train/validation 冻结 matching rule；在 heldout reference events 中寻找 A/B states：

- 相同 rank index 或相邻一个 step；
- recruited fraction、current-set size 相近；
- recruited-mask Jaccard 和 current-set overlap 高；
- phase 相近；
- future-field `u_A/u_B` 相差至少 validation-frozen quantile；
- 同 fit/arm/seed。

构造：

\[
\mathbf h_A'(\eta)=
\mathbf h_A+
\eta(\mathbf h_B-\mathbf h_A),
\qquad
\eta\in\{0.25,0.5,1.0\}.
\]

保留 `q_A` 的 `r_A,k_A`。该方法不称完整-state transplantation，而称 visited-state-derived hidden chord。
它与 local field-axis perturbation 独立报告，不能只保留方向更好的一个。

### 7.5 Control families

不再要求每个 state 找 20 个同时精确匹配 norm/logit/STOP/Jacobian 的“超级 control”。固定多套含义清楚
的 controls：

1. **norm-matched local-normal**：局部 residual PCA normal space 中固定 8 个方向；
2. **phase-shuffled axis**：使用 train phase-label shuffle 后重拟合的方向；
3. **PCA high-variance**：不使用 progress/field label 的前 3 条可辨识 PCA directions；
4. **C-suffix axes**：C-suffix 自身拟合的 progress/field axes 与 response；
5. **matched-observable empirical differences**：相似 `o_k`、但小 `u` difference 的 state chords；
6. **output/gain sensitivity**：记录即时 logit/STOP change 与 `||J delta h||`，做连续协变量调整和分层匹配。

axis-specific primary comparison只要求：

- raw perturbation norm matched；
- 同一局部 conditional-support 规则；
- positive/negative branches 同一 `q`。

即时 output/gain 不作为删除大量 states 的硬 caliper；它们必须逐 state 保存，并进入 robustness model。

### 7.6 Numerical support

support 在局部低维条件空间定义，不使用全维 robust Mahalanobis 作为唯一标准。每个 perturbed state
必须满足：

1. node-wise hidden bounds 在 train/validation empirical range 加冻结小容差内；
2. 在 `[z_prog,z_field,o_k]` 条件空间的 phase-matched kNN distance 不超过 validation q95；
3. 到 local manifold 的 residual norm 不超过 validation q95；
4. logits/state 全部 finite；
5. decoder 可执行；
6. 失败时不 clipping、不缩 dose、不换方向。

数值失败只排除该 state/dose/control，并记录原因；不停止其他 axes、control families 或 goals。

### 7.7 Open-loop

正、负、未扰动和 controls 在未来 `tau=1..3` 接收完全相同的真实 rank inputs。保存 hidden、logits、
STOP probability 和两个 latent coordinates。`tau=0` 只作 immediate-effect audit，不进入 finite-time
functional endpoint。

### 7.8 Closed-loop

从同一完整 `q` 分叉，后续只用冻结 decoder。保存：

- 每 step continuous contact logits；
- contact probabilities sensitivity；
- generated contact sets；
- cumulative generated field；
- STOP probability trajectory；
- discrete STOP length；
- `z_prog/z_field` trajectory。

如果某 branch 提前 STOP，之后进入 absorbing STOP；contact-logit response 不用复制最后一个活动 state。
per-step closed-loop response 只在两支均 active 的 risk set 上汇总并逐 `tau` 报 denominator；terminal
cumulative field 和 STOP trajectory 另作不依赖共同 active horizon 的 endpoints。

### 7.9 跨表征 contact-space endpoints

hidden axes 不用同一 hidden coordinate 作为唯一评价。定义 train-only、跨表征的 contact-space response
axes。它们与 hidden perturbation 不在同一坐标空间，但 `field` 轴仍由同一 axis-train data contract 提供，
因此不能称为独立数据复制：

\[
\mathbf v^{data}_{prog}=
\frac{\operatorname{center}
\left[(\overline{\mathbf y}_{TA}+\overline{\mathbf y}_{TB})/2\right]}
{\left\|\operatorname{center}
\left[(\overline{\mathbf y}_{TA}+\overline{\mathbf y}_{TB})/2\right]\right\|_2},
\]

\[
\mathbf v^{data}_{field}=\mathbf v_{field}.
\]

对 perturbation direction `beta` 和 future step `tau`，令 `d_beta` 为该方向的实际对称扰动剂量
（progress 为 `epsilon`，field 为 `lambda`）：

\[
\mathbf g^{\beta}_{e,k,\tau}=
\frac{\boldsymbol\ell^+_{e,k+\tau}-
\boldsymbol\ell^-_{e,k+\tau}}{2d_\beta}.
\]

这里的 `ell` 固定为 repeat-mask/STOP precedence 之前的 finite contact logits；post-mask decoder logits、
availability mask 和实际 generated set 另存。禁止用 `-inf` masked logits 做 central difference。

定义独立输出分数：

\[
S_{prog\leftarrow\beta}(\tau)=
\left\langle\operatorname{center}(\mathbf g^\beta_\tau),
\mathbf v^{data}_{prog}\right\rangle,
\]

\[
S_{field\leftarrow\beta}(\tau)=
\left\langle\operatorname{center}(\mathbf g^\beta_\tau),
\mathbf v^{data}_{field}\right\rangle.
\]

`v_data` 来自 train-only empirical fields；hidden axis 来自 hidden geometry。`z` persistence 降为动力学
diagnostic，不承担 primary functional endpoint。

### 7.10 Response matrix 与双重解离

对 `tau=1..3` event-first 聚合：

\[
R(s)=
\begin{bmatrix}
R_{prog\leftarrow prog} & R_{prog\leftarrow field}\\
R_{field\leftarrow prog} & R_{field\leftarrow field}
\end{bmatrix}.
\]

预注册两个 co-primary perturbation contrasts：

\[
D_{prog}=R_{prog\leftarrow prog}-
|R_{field\leftarrow prog}|,
\]

\[
D_{field}=R_{field\leftarrow field}-
|R_{prog\leftarrow field}|.
\]

两项在 perturbation family 内 Holm 校正，分别判定，不互相作为后续实验 gate。完整 `2x2` matrix、
effect size、CI、per-arm/per-phase values 同时报。

progress 预期推进/延迟 common propagation field；field perturbation 预期改变 TA↔TB continuous field，
而 progress/STOP spillover较小。STOP probability trajectory、remaining length、terminal mode score 都是
secondary，不再以 discrete STOP length 作为唯一 closed-loop primary。

### 7.11 Empirical transplantation endpoint

对 A→B chord，独立检验生成 future-field coordinate 是否朝 `u_B-u_A` 同方向变化。matched-observable
small-`u` chords 为 control。该实验是 model-internal state intervention，不是数据层面自然因果。

## 8. Goal 5.2B claim family

### C3：Axis-specific perturbation

独立判定：

- `D_prog`；
- `D_field`；
- empirical transplantation field response；
- local-normal/phase-shuffle/PCA/C-suffix control comparisons；
- output/gain-adjusted sensitivity。

C3 支持需要 co-primary family 校正后对应 contrast 成立，并在 output/gain sensitivity 中方向稳定。
某一 axis 阴性只限制该 axis claim，不删除另一 axis 或 Goal 5.2C。

`D_field` 分层裁定：all-identifiable-fit tier 只支持 generic within-fit field control；canonical A↔B field-control
claim 只使用 shared fits。两者都以 patient 为推断单位，并显式报告 shared/non-shared denominator。

### C4：Topology convergence

L0/L1/L2m/L3 分别构造 response matrix 和 finite-time response fields。比较：

- per-arm patient effects；
- leave-one-arm-out；
- arm-to-arm response-field similarity；
- real-order arms 内相似性相对 C-suffix；
- arm heterogeneity。

不要求每个 arm 单独显著才运行空间分析。只有多种 topology 的方向、response matrix 和 field geometry
均一致时，才称 topology-convergent computation。

## 9. Goal 5.2C：功能响应场与患者特异性

### 9.1 Finite-time contact response fields

对每个 fit/arm/seed、phase 和 `tau`，event-first 聚合：

\[
\mathbf g^{RNN}_{prog}(s,\tau)=
\operatorname{mean}_{events}\mathbf g^{prog}_{e,k,\tau},
\]

\[
\mathbf g^{RNN}_{field}(s,\tau)=
\operatorname{mean}_{events}\mathbf g^{field}_{e,k,\tau}.
\]

它们是 functional response fields。即时 `H a` 只作 readout sensitivity，不能替代 finite-time field。

### 9.2 与本患者数据场对齐

primary evaluation data fields 只由 heldout-test events 构造：

\[
\mathbf g^{data}_{prog}=
\operatorname{center}
\left[(\overline{\mathbf y}^{test}_{TA}+\overline{\mathbf y}^{test}_{TB})/2\right],
\]

\[
\mathbf g^{data}_{field}=
\operatorname{center}
(\overline{\mathbf y}^{test}_{TA}-\overline{\mathbf y}^{test}_{TB}).
\]

两者 primary 使用 start-removed fields；canonical full fields 为 sensitivity。检验：

\[
\mathbf g^{RNN}_{prog}\leftrightarrow\mathbf g^{data}_{prog},
\qquad
\mathbf g^{RNN}_{field}\leftrightarrow\mathbf g^{data}_{field}.
\]

使用 signed Spearman/cosine 并报告 shaft、distance、variogram/autocorrelation-preserving spatial nulls。
不得选择正负号最大者；符号由 train-only axis 冻结。

若 heldout 任一 mode 缺失、没有共同 valid contacts、field nonfinite 或 norm 数值退化，对应 C5 field 标记
`DATA_FIELD_NOT_IDENTIFIABLE`；事件少或 split reliability 低则标记 `DATA_FIELD_LOW_RELIABILITY` 并报告宽
CI，不按观察到的 alignment 删除。C3 的 train-defined output response 不因此删除，也不得冒充 C5 验证。

### 9.3 Cross-patient identity null

“本患者相关”不等于“患者特异”。预先构建不读取 field values 的 geometry-only transport：

\[
\mathcal T_{q\rightarrow p}^{geom},
\]

只使用冻结 contact coordinates、shaft metadata、contact spacing 和预定义 plane normalization，将患者
`q` 的 data field 映射到患者 `p` 的 contact support。禁止用 RNN/data field correlation 选择 rotation、
reflection、scale 或 contact matching。

patient-specific margin：

\[
I_p=
\operatorname{Align}(\mathbf g^{RNN}_p,\mathbf g^{data}_p)
-\operatorname{median}_{q\ne p}
\operatorname{Align}
(\mathbf g^{RNN}_p,
\mathcal T_{q\rightarrow p}^{geom}\mathbf g^{data}_q).
\]

映射必须先通过 geometry-registration audit。若公共方向/shaft/coordinate 信息不足，输出
`CROSS_PATIENT_IDENTITY_NOT_IDENTIFIABLE`，保留 within-patient alignment，不以 field-driven Procrustes
补救。

### 9.4 Tissue patch perturbation

在 Goal 5.2B 已冻结的 reference states 上扫描 Gaussian tissue patches，不重新选择 events。patch center
覆盖 fit tissue grid；width 以 train geometry 固定为局部 node spacing 的预注册倍数。对 center `c` 定义：

\[
p_c(i)=\exp\left[-\frac{\|r_i-r_c\|_2^2}{2\sigma_{patch}^2}\right],
\qquad
\widetilde{\mathbf p}_c=\frac{\mathbf p_c}{\|\mathbf p_c\|_2}.
\]

使用与 axis perturbation 相同的 raw-state norm 和 local-SD dose，对
`h +/- d_patch * p_tilde_c` 做 central difference，其中代码变量 `p_tilde_c` 对应
`\widetilde{\mathbf p}_c`；再把 finite-time pre-mask contact-logit response 投影到
train-defined progress/field contact axes，得到：

\[
\chi_{prog}(r,s,\tau),
\qquad
\chi_{field}(r,s,\tau).
\]

patch perturbation 使用与 axis perturbation 相同的局部支持与数值合法性规则，但不因 axis perturbation
统计阴性而取消。报告：

- progress-control field；
- field/mode-control field；
- sign stability；
- cross-seed/topology consistency；
- immediate-output 与 finite-time effect 分离。

它们仍是 model-internal tissue-node response fields，不是真实刺激图。

### 9.5 SNN input eligibility

在读取 SNN field values 前生成 `SNN_INPUT_ELIGIBILITY.json`。每个 candidate 必须记录：

- patient/geometry；
- producer path/hash；
- engine/baseline ID；
- `runtime_mode`；
- simulation duration；
- late-runaway status；
- network replication status；
- natural mode validation；
- field definitions；
- RNN↔SNN mapping；
- source status：`LOCKED_ACCEPTED/CANDIDATE_REPLICATED/DIAGNOSTIC_ONLY`。

资格层级：

```text
LOCKED_ACCEPTED with adequate patient denominator
  -> cohort C6 inference

CANDIDATE_REPLICATED or small denominator
  -> exploratory case-series alignment

DIAGNOSTIC_ONLY, unreplicated, runaway, missing runtime/provenance
  -> visual/descriptive audit only; no inferential C6 claim
```

当前 registry 中的 data-driven SNN 仍是 `SOURCE/CANDIDATE`，fresh-network replication 未闭合；因此本
spec 不预写其为 cohort-eligible。执行时按冻结 metadata 裁定，不因 RNN alignment 好看升级 status。

### 9.6 RNN–SNN functional alignment

比较功能场而非 edges/synapses：

\[
\mathbf g^{RNN}_{field}
\leftrightarrow
\mathbf g^{SNN}_{mode},
\]

\[
|\chi^{RNN}_{field}|
\leftrightarrow
\mathbf g^{SNN}_{core/susceptibility},
\]

\[
\mathbf g^{RNN}_{prog}
\leftrightarrow
\mathbf g^{SNN}_{propagation}.
\]

mapping、mode sign、core definition 和空间 registration 在读取 RNN–SNN alignment 前冻结，不能为了
相关性重新旋转或重新选 core。若多患者字段可用，使用同患者减 cross-patient identity null；若只有少量
患者，明确降为 case-series。

RNN 与 SNN 共享患者数据和几何，称 `cross-model convergence`，不称 independent replication。

### 9.7 Frozen early-ictal exploratory alignment

只有 RNN response fields、patch fields、data/SNN mappings、null maps 和 source hashes 全部冻结后，才
解封既有 17 位患者/167 seizures 的 clinical-onset 后 0–10 s、1–150 Hz broadband energy field。

预注册：

- progress/field response fields 与 early-ictal signed correspondence；
- patient-first seizure→patient 聚合；
- synchronized all-contact primary spatial null；
- geometry-eligible within-shaft/distance/variogram sensitivity；
- cross-patient identity sensitivity；
- no best-axis/best-phase oracle，除非每个 null draw 内完整重选且明确标为 omnibus exploratory。

该 target 已在项目历史中查看，因此结果只能是 locked internal exploratory convergence，不获得独立确认
地位，也不改变 Goals 5.2A/B 的 target-free claim。

## 10. Goal 5.2C claim family

### C5：Patient-specific data alignment

需要：

- within-patient functional-field alignment；
- cross-patient identity margin `I_p`；
- spatial null；
- cross-seed/fit稳定；
- 明确实际 denominator。

若 cross-patient registration 不可辨识，只能写 within-patient alignment，不能写 patient-specific。

### C6：Cross-model convergence

只有 SNN eligibility 为 inferential tier 才做 cohort claim；case-series 和 diagnostic-only 结果单列。
C6 与 C5 分别判定，SNN 输入不足不否定 RNN/data patient specificity。

### C7：Cross-state exploratory alignment

独立报告但不进入 confirmatory family，不因 nominal `P` 升级整条机制线。

## 11. 统计合同

### 11.1 聚合顺序

固定为：

```text
control draws within reference state
-> event
-> phase and tau
-> seed within arm/fit
-> arm-specific result
-> fit within patient
-> patient-level inference
```

跨 topology convergence 另外在 patient 内折叠 arms。event、step、seed、arm、control、patch 和
`own_a/own_b` fits 都不是独立 cohort samples。

### 11.2 Claim families

| Claim | 内容 | 校正/地位 |
|---|---|---|
| C1 | progress + continuous future-field geometry/emergence | family Holm |
| C2 | tangent transport + transverse contraction | family Holm |
| C3 | progress-control + field-control co-primary response | two-test Holm |
| C4 | topology convergence | secondary family |
| C5 | within-patient + identity-null data alignment | family Holm |
| C6 | SNN cross-model convergence | eligibility-dependent family |
| C7 | early-ictal | exploratory only |

不同 claim family 不建立“前一 family 显著才执行后一 family”的统计门。最终按支持到 C1–C7 哪一层写结论。

### 11.3 Denominator discipline

不设“少于 20 patients 就停止整个项目”的任意阈值。每个实验报告：

- scheduled/eligible/completed patients、fits、events、states；
- exclusion reasons；
- cohort inference denominator；
- descriptive-only denominator；
- uncertainty/CI。

不设置跨所有 branches 通用的患者数阈值。只要 estimand 在数学上可计算，就输出 patient-level effect、
exact/permutation uncertainty 和完整 denominator；claim 强度由 CI、source eligibility、patient coverage 和
预注册 estimand 是否可辨识共同决定。source 本身只有 case-series 资格时不得因 nominal `P` 升为 cohort
claim，也不把一个 branch 的小分母传播到其他 branches。

## 12. 仅保留两个 hard gate

### Gate E0：Engineering integrity

必须满足：

- 630 checkpoint cells 来源明确；
- checkpoint/config/split/H/node-mask/decoder hashes 完整；
- teacher-forced 与 closed-loop replay 正确；
- 完整 `q=(h,r,k)` 被复制；
- parameter hashes 不变；
- 无 target leak；
- 无 nonfinite；
- input/contact/node ordering一致；
- storage/resource preflight PASS。

E0 失败只停止受影响的执行阶段并修工程合同，不能用科学结果绕过。

### Gate N0：Per-perturbation numerical validity

每个 state/dose/control/patch 必须满足局部 conditional support、finite state/logits 和 decoder 可执行。
失败只排除该单元并记录，不停止其他 scientific analyses。

除此之外没有统计 hard stop。C1/C2 阴性不阻止 C3；C3 阴性不阻止 C4/C5/C6/C7；SNN 不可用不阻止
data/patch/early-ictal。

## 13. 工程数据流：两遍 extraction

### Pass 1：Streaming system identification

逐 unit 重放但不保存全量高维 archive。流式计算并保存：

- robust node mean/scale；
- event-first incremental PCA sufficient statistics；
- phase-binned state mean/covariance；
- raw/residual low-dimensional projections；
- progress/future-field decoders；
- P/PF curves；
- Jacobian transport/contraction summaries；
- closed-loop projected transitions；
- selected reference-state IDs 和重放 provenance。

必要时对同一 unit 做多次确定性 replay；以计算换存储，避免先复制全部 hidden/logits。

资源预审后固定采用 response-blind event sample：每个 fit 在 split 内按
`SHA256(patient, split, event_source_index, event_dataset_index)` 排序，axis-train 最多 1024 events、
axis-validation 最多 512、heldout-test 最多 512；少于 cap 时全纳入。同一 fit 的 event sample 在全部
arms/seeds 间完全共享；同一患者的 `own_a/own_b` 也使用相同 event identities。保存原始 denominator、inclusion
fraction 和 manifest hash。该 sampling 只限制 hidden
system-identification replay；future-field centroid/axis 仍读取全部 axis-train event fields。禁止按 mode、`u_e`、
hidden geometry、effect 或 response 改样本。Pass 2 reference events 再从已冻结 heldout sample 中按同一 identity
hash 最多取 64 个/fit，且在读取 perturbation response 前锁定。

### Pass 2：Selected-state extraction

只对预冻结 reference IDs 保存完整：

```text
q=(h,r,k)
current input
future open-loop inputs
logits/STOP/size state
local axes/basis
support neighbors
empirical transplantation pairs
```

随后运行 axis perturbation、controls、state chords 和 spatial patches。不得根据 Pass 1 effect direction
选择 reference patients、phases 或 axes；只能按预注册 phase/denominator/数值规则选择。

## 14. 结果与 provenance

结果根：

```text
results/topic5_latent_propagation_landscape_v0_2/
```

至少包含：

```text
CONTRACT.json
CHECKPOINT_MANIFEST.csv
INPUT_AUDIT.json
RESOURCE_BUDGET.json
PASS1_STREAMING_MANIFEST.json
LATENT_GEOMETRY_SUMMARY.json
DYNAMICAL_TRANSPORT_SUMMARY.json
REFERENCE_STATE_MANIFEST.csv
PASS2_PERTURBATION_MANIFEST.json
PERTURBATION_RESPONSE_MATRIX.json
FINITE_TIME_RESPONSE_FIELDS.npz
SPATIAL_PATCH_CONTROL_FIELDS.npz
DATA_ALIGNMENT_SUMMARY.json
SNN_INPUT_ELIGIBILITY.json
SNN_ALIGNMENT_SUMMARY.json
EARLY_ICTAL_EXPLORATORY_SUMMARY.json
COHORT_PATIENT_TABLE.csv
CLAIM_LADDER_ADJUDICATION.json
CLOSEOUT_AUDIT.json
figures/README.md
```

只有实际生成 figures 后才创建 `figures/README.md`。大矩阵使用 chunked Zarr/HDF5 或分块 NPZ；每个
unit/field 原子写入、带 hash 和 resume marker。不写 monolithic full-trajectory NPZ。

## 15. 必须实现的测试

1. 630-cell checkpoint resolver；
2. full-state `q` clone；
3. 同 `h`、不同 `r/k` 改变 decoder decision 的反例；
4. teacher-forced hidden/logit/STOP/size replay；
5. closed-loop 不读取真实 future suffix/set size；
6. event-first/phase-balanced weights；
7. raw-state primary、residual sensitivity 路由；
8. `s_step` primary、`s_contact` sensitivity；
9. future-field axis 只使用 train events；
10. heldout `u_e` 不进入 model input；
11. `own_a/own_b/shared` node-space guard；
12. field direction sign 在 train-only 冻结；
13. P/PF/shuffled-PF capacity matching；
14. Jacobian使用相同真实 next input；
15. local normal directions 来自条件 residual space；
16. local tangent perturbation 保持 `r/k` 不变并明确记录；
17. empirical transplantation pairs 只按冻结 observables 选；
18. support gate 在 `[z,o]` 条件空间执行；
19. `tau=0` 不进入 finite-time functional endpoint；
20. open-loop branches 使用相同 future inputs；
21. response endpoints 使用 train-only cross-representation contact-space axes；
22. `2x2` response matrix orientation 正确；
23. immediate-output/gain 作为 sensitivity covariates，不静默删 state；
24. cross-patient mapping 不读取 field values；
25. SNN mapping/core/mode sign 在 alignment 前冻结；
26. SNN eligibility status 不由 alignment 结果升级；
27. early-ictal target 在 target-free fields/nulls 冻结前不可读；
28. patient aggregation 不重复 event/seed/arm/fit；
29. parameter hashes before/after一致；
30. figures/source tables/claim ladder 一一对应。

## 16. Claim ladder 的安全措辞

### 只支持 C1

> Frozen full-tissue RNN states exhibit a within-patient, task-aligned geometry jointly organized by event phase and future propagation field.

不能写 patient-specific identity、control、channel 或 pathology-related。

### C1 + C2

> Local frozen-RNN dynamics transport task-aligned directions and constrain transverse deviations, consistent with a model-internal propagation channel.

不能写 biological attractor。

### C3

> Directional hidden-state interventions selectively alter future propagation progress and/or future-field identity within the frozen RNN.

必须分别说明 progress/field 哪一项成立。

### C4

> Distinct recurrent topologies converge on similar finite-time functional responses despite non-identifiable static connectivity.

### C5

> RNN functional response fields preferentially align with the same patient's empirical interictal propagation fields relative to geometry-registered cross-patient fields.

没有 identity null 时不得用 `patient-specific`。

### C6

> RNN and SNN functional fields show patient-level cross-model convergence under frozen mappings.

不能写 independent replication；必须说明 SNN eligibility tier。

### C7

> Frozen interictal control fields show exploratory correspondence with early-ictal broadband energy.

不能写 prediction、recruitment 或 confirmation。

## 17. 相对上一版的正式修订

保留：

- 完整 `q=(h,r,k)`；
- checkpoint/decoder/parameter hashes；
- teacher-forced 与 closed-loop 分离；
- target seal；
- fit-specific node-space；
- local support 与 numerical fail-closed；
- patient-first aggregation。

删除或降级：

- “E3 前窄门控审计”定位；
- residual hidden 作为 primary state；
- contact-cumulative phase 作为 primary；
- progress `P` 值控制 mode/spatial/SNN 是否运行；
- progress-only scientific中心；
- 同一 hidden coordinate 作为唯一 perturbation endpoint；
- 每 state 20 个四指标超级匹配 controls；
- discrete STOP length 作为唯一 closed-loop primary；
- 任意 20-patient/80%-matching 项目级停止阈值。

恢复并强化：

- continuous future-field coordinate `u_e`；
- raw-state two-coordinate manifold；
- C-suffix future-field emergence control；
- tangent transport/transverse contraction；
- local tangent + future-field perturbation；
- empirical state transplantation；
- `2x2` functional response matrix；
- finite-time contact response fields；
- spatial patch control fields；
- cross-patient identity null；
- eligibility-aware SNN convergence；
- frozen early-ictal exploratory alignment。
