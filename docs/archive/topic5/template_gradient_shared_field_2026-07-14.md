# Topic 5：A/B 单模板梯度轴、共线分型与共享 2D field（2026-07-14）

## 1. 一句话结论

这次重构把“方向是否存在”“采样几何是否支持二维平面”和“方向是否重采样稳定”拆开了。28 名可建轴患者全部都有 A/B 两个方向：14 名宽泛共线（9 反向、5 同向），14 名不同向；其中 26 名具备二维几何，12 名共线、14 名不同向。严格稳定的 13 名只是高置信分层，不是方向存在的门槛。对有发作 cache 的 7 名二维共线患者，共享 broadband field 未超过 within-shaft null；阳性只出现在严格稳定的 n=6 子集，且 shared 没有提高患者内观测相关。因此当前成立的是 A/B 方向与共线分型，shared-field 的超细解剖证据仍依赖质量分层。

## 2. 为什么要重构

旧逻辑混合了三个不同问题：模板 A/B 的逐触点 rank correlation、由 swap endpoint 构造的方向、以及在该方向上的 2D field similarity。`rho_AB` 会随支持节点和节点数变化，也不直接回答两条空间梯度是否平行；把所有患者都投到 D_AB 或 swap 共享轴，又会把本来不同向的两条模式强行合并。

新逻辑严格按顺序执行：

1. 仅用间期模板 A、B 分别建轴；
2. 仅用这两条间期轴判断是否共线、同向或反向；
3. 只有共线患者才构造共享角平分轴和统一 2D 平面；
4. 轴、质控、共线门槛全部冻结后，才读发作早期能量场。

执行合同：`docs/superpowers/specs/2026-07-14-topic5-template-gradient-shared-field-design.md`。

## 3. 方法合同

### 3.1 单模板轴

输入为 masked rank-displacement `pair[0]` 的 `joint_valid` 共同节点。对每个模板定义 `e_T=-z(rank_T)`，先在 3D 触点坐标上拟合最小二乘 earliness gradient `g_T=gradient(e_T)/||gradient(e_T)||`。由于 `g_T` 指向晚→早，正式传播轴取其反向：`u_T=-g_T`，正方向固定为低 rank 的早节点→高 rank 的晚节点。producer 中 `u` 只保存该 propagation-positive 方向，原始梯度另存为 `earliness_gradient_u/beta`。A、B 使用同一组共同节点，但各用自己的标量场，因此这里的 own axis 不等于 own support mask。

轴可建条件只有：至少 6 个共同有坐标触点、earliness 非常数、梯度非零；满足后 A/B 方向均已定义并进入全体方向分析。至少 2 根杆且 `effective_rank>=2` 是二维 field 的几何资格。contact-bootstrap median cosine ≥0.80 和 leave-one-shaft-out median cosine ≥0.50 定义 strict-stability 高置信分层，只作敏感性，不能表述成其余患者“没有方向”。R2 只报告，不以拟合优度事后筛患者。

### 3.2 共线与共享平面

线关系用两个 early→late 传播向量的 `|cos(u_A,u_B)|`；符号另行区分传播方向同向/反向。宽松主门槛为 `|cos|>=0.5`（线夹角 ≤60°），45°、30° 为敏感性。paired contact bootstrap 同时报告共线概率和符号稳定率；二者均 ≥0.80 为 robust sensitivity subset。此次把 A/B 两轴同时从 `g_T` 改为 `-g_T`，因此 cosine、线夹角及 same/reversed/different 分型完全不变。

共线后先把 B 对齐到 A，再固定共享角平分轴：`u_shared=normalize(u_A+sign(cos)u_B)`。所以同向用和，反向用差；不同向患者不建共享轴。共享轴完全不看发作数据。

**2026-07-14 方向合同纠正**：旧 producer 曾把 `gradient(e_T)` 直接写入 `axis_a/b.u`，导致存储的正向实际为晚→早，尽管部分绘图代码又额外取负画成 early→late。现已在 producer 层统一纠正为 `u=-gradient(e_T)/||gradient(e_T)||`，并删除绘图层的二次翻转。该修正只把 along 坐标整体反号，不改变距离、field、null、maxAB 或 cohort 统计；重跑后应逐项验证科学数值不变。

### 3.3 Field 与 null

主 field 使用模板各自的事件参与率作为 Gaussian kernel support；own 与 shared 只改变平面，模板值和 support 不变。全部事件共同 support 另跑敏感性。正式相关在触点位置计算，不把 2D 插值像素当独立样本；identity/transverse-mirror 候选直接按 `|r|` 最大者选择。

每次发作计算 A、B 及 `maxAB=max(|r_A|,|r_B|)`；每次洗牌重新计算 A/B 并重新选 maxAB。先在患者内对发作取中位数，再形成 cohort median。主发作读出为 1–45 Hz、0–10 s `bb_auc`；channel shuffle 为 coarse null，within-shaft shuffle 为解剖控制。A/B 是患者内模板标识，未跨患者锚定，所以不把“B 比 A 强”写成 cohort 生物学结论。

## 4. 分母与结构结果

全 rank-displacement 输入为 40 名：

- 34 名 `stable_k=2`；
- 28 名可同时建立 A/B 梯度轴；其余包括 5 名 Yuquan 缺坐标、E1073 只有 3 个共同映射触点，以及 6 名 `stable_k!=2`；
- 28/28 可建轴者全部获得 `u_A/u_B` 与 signed cosine：反向 9、同向 5、不同向 14，即 14/28 宽泛共线；
- 26/28 具备至少两根杆和二维有效秩：反向 7、同向 5、不同向 14，即 12/26 共线；另外两名单杆患者的方向仍保留，但 `cos≈±1` 主要受杆方向约束；
- 13/28 双轴同时通过 strict-stability：反向 6、同向 2、不同向 5；其中 8/13 共线；
- paired-bootstrap robust 共线为 4 名，均为反向。

因此目前安全的结构结论是：**所有可建轴患者都有 A/B 双方向；全体中一半宽泛共线、一半不同向。统一 2D 平面只对二维几何且共线者构建，strict-stability 仅用于评估结果对方向可靠性的依赖。**

## 5. 早期发作 field 结果

### 5.1 分母

18 名有 ictal cache 且可建轴，17 名具备二维几何。二维且共线的 field 主分母为 7 名（E1084、E1146、E384、E548、E583、E590、E958）；strict-stability 子集为其中 6 名，E384 因 A 轴 contact-bootstrap median cosine=0.737 未达 0.80 而进入稳定性敏感性之外，但仍有可估计方向和合格二维几何。45° 子集 n=4，pair-bootstrap robust 子集 n=3。

### 5.2 Own-axis field

在全部 17 名二维几何且可做 field 的患者中：

- own `maxAB` cohort median |r|=0.798；
- within-shaft null median 0.807，upper-tail p=0.676。

因此全二维几何分母不支持 own-axis field 超越杆内解剖解释。

在 7 名二维共线患者中，own `maxAB` median |r|=0.790，对 within-shaft null 仅临界（null median 0.756，p=0.053）。strict-stability n=6 子集才达到 p=0.035。

### 5.3 Shared-axis field

7 名二维共线患者在共享平面上：

- shared `maxAB` median |r|=0.791；
- within-shaft null median 0.783，upper-tail p=0.346，未超过 null；
- strict-stability n=6 子集中 shared median 0.789、null median 0.720、p=0.002。该阳性是独立于发作结果定义的质量分层，但不能替代 n=7 二维主分母。

关键的 paired comparison 没有显示 shared 优于 own：

- `shared-own` 观测 |r| 差的患者中位数 = −0.002；
- bootstrap 95% CI = [−0.038, 0.025]；
- paired Wilcoxon p=0.938；
- within-shaft null-adjusted margin 差中位数 = −0.007。

因此正确解释是：**共享平面给二维共线 A/B 提供了统一表示，但 broadband field 在完整二维共线分母中未超过 within-shaft null，也没有提高原始相关；较强阳性局限于 strict-stability 子集。**

## 6. 敏感性与边界

- **Epilepsiae 逐人频段图（n=17）**：不按轴稳定性或共线性筛选，但仅保留 17 名 Epilepsiae 双轴可估患者。Yuquan 中仅 2 人各有 3 个缓存发作条目，其余无可用发作，因此本组发作 field 图全部排除 Yuquan；上游 28 人可建双轴的结构分母不变。own-axis maxAB 的患者中位数为 HFA 60–100 Hz 0.764、broadband 1–45 Hz 0.783、broadband 1–150 Hz 0.783。1–45 与 1–150 的跨患者 Spearman ρ=0.998、paired median difference≈0（Wilcoxon p=0.963）；HFA 与 1–45 也高度相关（ρ=0.794，paired p=0.963）。
- 共同 support 的二维 n=7 结果一致为阴性：shared within-shaft p=0.349；说明主结果不依赖 template-specific support。
- strict-stability n=6 shared p=0.002，45° n=4 p=0.010，但 pair-bootstrap robust n=3 p=0.067。阳性随质量/角度筛选出现而未在最严格小样本层过门，必须完整报告选择层级。
- HFA 二维 n=7 shared p=0.027，但 45°（p=0.254）和 robust n=3（p=0.648）不稳定；HFA 只作 sensitivity，不升级为频带特异结论。
- 这里是 early-ictal readout，不是 early-ictal-specific。已有时间分辨结果显示相似场可在发作前存在，本分析不能把 0–10 s 相关解释为 onset 时新出现。
- 共线不等于同一白质束或因果传播通路；gradient 是采样空间内的一阶趋势。单杆方向保留在 28 名结构分布中，但不进入二维 field；有限 SEEG 覆盖仍限制外推。

## 7. 当前可写口径

> Template-specific early-to-late propagation axes, defined as the negative spatial gradients of interictal earliness, yielded two estimable directions in 28 patients, with half showing broadly collinear A/B axes and half showing distinct directions. Among patients with two-dimensional sampling geometry, shared-plane broadband field similarity did not exceed within-shaft spatial shuffling in the full collinear subset, although a stronger association emerged in the independently defined strict-stability subset. The shared representation did not increase patient-level correlation relative to template-specific planes.

禁止写：

- “所有患者的 A/B 都是一条双向轴”；
- “共享平面优于模板自己的平面”；
- “模板 B 比 A 更病理”；
- “发作沿间期传播通路逐点重放”；
- “该场是发作起始时新出现的”。

## 8. 产出

- 代码：`src/topic5_template_axis_field.py`
- cohort runner：`scripts/run_topic5_template_axis_field.py`
- 图：`scripts/plot_topic5_template_axis_field.py`
- 测试：`tests/test_topic5_template_axis_field.py`
- 主结果：`results/topic5_ictal_recruitment/template_axis_field/`
- common-support sensitivity：`results/topic5_ictal_recruitment/template_axis_field_common_support/`
- HFA sensitivity：`results/topic5_ictal_recruitment/template_axis_field_hfa/`
- Epilepsiae n=17 三频段 OR-margin board、paired-field atlases 与逐人表：`results/topic5_ictal_recruitment/template_axis_field_frequency_panel/`（数据适配器 `scripts/render_topic5_maxab_field_concordance.py`；绘图直接调用既有 field-concordance 函数）
- 图说明：`results/topic5_ictal_recruitment/template_axis_field/figures/README.md`

所有 per-subject JSON 保留轴 QC、pair bootstrap、own/shared plane、逐发作观测和完整 B=1000 null 分布；`axis_cohort.csv` 是结构分母表，`cohort_summary.json` 是正式聚合入口。
