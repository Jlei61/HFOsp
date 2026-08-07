# Topic 5：单模板梯度轴、共线分型与共享场读出合同

## 1. 科学问题与分析顺序

本分析回答两个依次发生、但统计上彼此隔离的问题：

1. 仅用间期 HFO 传播模板，模板 A 与 B 是否各自形成可重复的空间传播轴；两条轴是否近似共线，以及同向还是反向。
2. 在轴定义完全冻结后，发作早期 broadband energy 是否复现单模板场；对共线患者，把 A/B 放到同一共享平面后，场读出是否比各自平面更清楚。

发作数据不得参与轴拟合、轴质控、共线判定、共享轴选择或平面方向选择。当前分析属于 exploratory confirmation：阈值在查看本轮 ictal-field 结果前锁定，但不是历史预注册结果。

## 2. 输入与支持集

- 模板来自 masked rank-displacement `pair[0]`，仅纳入 `stable_k=2`。
- A、B 均在同一组 `joint_valid + coord-mapped + finite-rank` 触点上拟合。这里“模板自己的轴”指 A 与 B 使用各自的标量场，不指改变空间支持集。
- 单模板 earliness 定义为 `e_T=-z(rank_T)`。注意 `gradient(e_T)` 指向 earliness 增大的方向，即**晚→早**；正式传播轴的正方向必须另定义为**早→晚**。
- field 分析使用上述触点与 ictal cache channel 的交集。主分析中，A/B 各自的 smoothing support 为该触点在分到相应模板的间期 group events 中的 participation fraction；同一模板的 own/shared/发作场/null 共用该 support，因此 own 与 shared 之间只改变平面。两模板仍限制在同一组 joint-valid 节点上。全部间期事件的共同 support 作为敏感性分析，不能替代模板自己的 support。
- 主发作读出为 `bb_auc`：1–45 Hz、发作后 0–10 s 的 baseline-robust-z power 均值。统计单位为患者，不把触点或网格像素当独立样本。

### 2.1 纯间期冻结 artifact（2026-07-15 收口）

间期构建与发作读出拆成两个 producer。`scripts/build_topic5_interictal_template_fields.py`
只读取 masked stable-k=2 TA/TB joint-valid ranks、共同有效触点、三维坐标和间期 participation support，输出
`results/interictal_propagation_masked/template_gradient_fields/per_subject/<subject>.json`。
每个 artifact 固定保存 TA/TB early→late 轴、三层方向有效性、own plane/field/kernel weights；共线者
另存 shared plane/field。该 producer 禁止读取 seizure、onset、subtype 或能量数据。

所有后续发作分析必须按 channel name 对齐到 artifact 的 `interictal_field.contact_order`，直接复用
固定的 axis、plane、sigma、support、template field 和 kernel weights。发作 cache 缺少的触点保留为
missing，不得用剩余触点重新拟合轴或平面。`decision_k`、swap、endpoint 和 D_AB 不参与该 artifact；
它们只保留为补充方法校准。artifact 对上述冻结组件保存 SHA-256 指纹，下游加载时必须校验并在
不一致时 fail closed。

## 3. 单模板梯度轴

对每个模板拟合

`e_T(i) = alpha_T + beta_T^T (x_i-xbar) + epsilon_i`。

使用与既有 D_AB gradient 相同的截断最小二乘（相对奇异值阈值 0.05）。其中 `g_T=beta_T/||beta_T||` 是 earliness-gradient 正向（晚→早）；正式传播轴定义为

`u_T=-g_T=-beta_T/||beta_T||`，正方向固定为低 rank / 高 earliness 的早节点 → 高 rank / 低 earliness 的晚节点。

producer 必须把两者分字段保存：`u` 只表示 early→late propagation；原始梯度只允许写入 `earliness_gradient_u` / `earliness_gradient_beta`，不得再把 `gradient(e_T)` 直接称为传播正方向。必须区分三个层级：

1. **axis estimable**：至少 6 个共同且有坐标的触点，earliness 非常数且梯度非零。所有 estimable 患者均有 `u_A/u_B`，全部进入方向和共线分布；不能把后续稳定性门写成“是否有方向”。
2. **2D geometry supported**：至少 2 根电极杆且两轴 `effective_rank>=2`。这是统一 2D field 主分析的几何资格。单杆患者仍有沿杆方向，但其共线几乎由采样几何决定，只作标记结果。
3. **strict stability sensitivity**：两轴均满足 contact-bootstrap median `|cos(u_boot,u_full)|>=0.80`，且 leave-one-shaft-out median `|cos|>=0.50`、可计算。该层衡量重采样可靠性，只作高置信分层，不替代 estimable 或 2D geometry 主分母。

`R2`、condition number、Moran's I、within-shaft variance fraction 全部报告，但不作事后硬门槛。

## 4. A/B 共线与共享轴

- 线夹角：`phi=acos(|u_A^T u_B|)`，范围 0–90°。
- 宽松主定义：`|cos|>=0.50`（`phi<=60°`）。
- 敏感性：`|cos|>=0.707`（45°）和 `|cos|>=0.866`（30°）。
- `cos>0` 为传播方向同向，`cos<0` 为传播方向反向；未达到共线阈值称为不同向，不强建共享轴。由于 A/B 两轴同时从 earliness-gradient 翻为 propagation-positive，cosine、线夹角和三类分型数值不变。
- paired contact bootstrap 报告 `P(|cos_boot|>=0.5)` 与方向符号稳定率。二者均 `>=0.80` 定义为 robust-collinear sensitivity subset；它不替代宽松主定义。

对共线患者，把 B 对齐到 A：`u_B*=sign(cos)u_B`，共享轴固定为

`u_shared = normalize(u_A + u_B*)`。

因此同向时是 `u_A+u_B`，反向时是 `u_A-u_B`。共享平面的横轴由触点相对共享轴的残差第一主成分确定；其正负无解剖意义，所以 field correlation 对横轴镜像不变。

方向纠正只会把 own/shared 平面的 along 坐标整体反号；触点间距离、Gaussian 权重、field correlation、null 和 maxAB 均应保持不变。该不变量必须用回归测试和重跑结果核对。

## 5. Field 表示与相似性

所有场均在触点上用相同 Gaussian Nadaraya-Watson kernel 估计，带宽为归一化平面内 median nearest-neighbour spacing。主统计使用 contact-evaluated field，避免把插值网格像素当样本；2D grid 仅用于可视化。

### 5.1 间期 TA/TB shared-field 反向性

该问题只在上游已定义 shared axis 且 `geometry_2d_supported=true` 的患者中回答。TA、TB 必须投影到
同一个冻结 shared plane，并在同一组 contact-evaluated 位置计算 signed Pearson
`r_shared=corr(F_TA,F_TB)`；`r_shared<0` 表示两个传播场在共享平面上反向。主队列不再按
signed axis cosine 的正负、`same/reversed` 标签或 strict-stability 分组；这些字段只用于上游定轴审计。
不同轴患者没有合法 shared plane，不进入该 estimand，也不得用 own field 与 shared field 混合补齐分母。

为与既有 field-concordance 主分析保持同一随机化层级，主 null 固定为全触点 channel shuffle：
保持 shared axis、shared plane、contact set、bandwidth 和 TA field 不变，在全部触点间共同置换
TB earliness 与 participation support，再重建 TB field。within-shaft shuffle 作为更严格的
anatomy-controlled sensitivity，回答负相关是否超出杆内几何。统计单位为患者；主图展示每名患者的
observed signed `r` 与其 channel-null median 的配对 Data–Null 分布，并沿用既有 reversal panel 的
paired Wilcoxon `alternative='less'`。该检验条件于上游已定 shared axis，不重新检验 KMeans、建轴或
共线门本身。

每次发作同时计算：

- `own_A`：A 模板与发作能量都投影到 A 自己的平面；
- `own_B`：B 模板与发作能量都投影到 B 自己的平面；
- `own_maxAB=max(|r_A|,|r_B|)`；
- 共线患者另算 `shared_A/shared_B/shared_maxAB`：A、B 与发作能量均投影到同一个共享平面。

每个模板保留 signed r 与 absolute r；cohort 主读出沿用既有 A-line 的 polarity-free `maxAB`。identity 与 transverse-mirror 两个候选必须按 `|r|` 最大者选择，不能先取 signed maximum 再取绝对值。

## 6. Null、折叠和比较

- coarse null：在全部纳入触点间洗牌发作能量；
- anatomy-controlled null：仅在每根电极杆内洗牌；`effective_shuffle_n<4` 标记为 insufficient，不计为通过。
- 每个 permutation 对 A、B、own/shared 全部使用同一个洗牌向量；每次都重新计算 A、B 并重新选择 maxAB，完整包含选择效应。
- 先按发作计算，再以患者内 seizure median 折叠；第 b 个患者 null 是第 b 次洗牌在全部发作上的 median。
- cohort 以患者为单位。own vs shared 只在同一批共线患者内做 paired comparison，同时报告 observed difference 与 `observed-null median` margin difference。
- 方向/共线主分母使用全部 axis-estimable 患者；统一 2D shared-field 主分母使用 `2D geometry supported + |cos|>=0.5 + ictal cache`。strict-stability、robust-collinear、45°、30° 均为质量或阈值敏感性。same/reversed 仅分层描述，除非样本量足够。

## 7. 允许与禁止的结论

允许：全部可建轴患者的 A/B 方向分布；A/B 轴在部分患者近似共线；二维几何合格的共线患者中共享场能否复现早期发作能量；strict-stability 分层结果；own/shared 的患者内配对差异。

禁止：把共线解释为同一个生物通路；把 field correlation 写成传播因果；把 early-ictal readout 写成 early-ictal-specific（需独立时间窗对照）；把全 cohort 的轴结果与仅有 ictal cache 的 field 子队列混成同一分母。
