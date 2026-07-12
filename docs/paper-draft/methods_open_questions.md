# Methods 待确认问题

> 目的：把“作者需要确认”“需要改代码重跑”和“可直接写入”分开。P0 清零前不建议把 Methods 送入正式投稿稿件。

## P0：会改变方法或科学结论

### 1. MI 的文字定义与执行代码不一致 — ✅ 已解决（2026-07-12）

- **原问题（核实属实）**：旧 legacy MI 在事件得分时用完整有限 rank，纳入非参与触点的 phantom 伪秩（`compute_legacy_mi` 对全部通道对打分、非参与位置为有限伪值），与"仅共同参与触点"文字定义是不同统计量。
- **实测方向**：phantom 是稀释而非伪造——去掉后 36/40 患者 MI 反而更强、40/40 显著全部保持，故修正只加强主结论、不推翻它。
- **已实施**：`compute_legacy_mi` 主字段改为 masked shared-participant（事件仅在共同参与触点打分、置换零分布在参与集合内重排秩），旧全通道版保留为 `unmasked_sensitivity`；TDD 3 测试 + 51 回归全绿。
- **全量重跑**：`--augment-masked-mi --masked-features` 重算 40 例 → masked 40/40 显著（cohort median 0.228；unmasked sensitivity 同 40/40、median 0.188）。产物 `results/interictal_propagation_masked/`；Figure 1 paneld1 已按 masked 重画。

### 2. “共同 A–B contact plane”不是当前 field producer

- **现状**：附件用模板 A/B 的两个早端定义共同轴；当前 producer 分别用模板 A、B 各自的 source–sink 轴构建两个平面。
- **为什么严重**：轴坐标、核距离、mirror null 和 maxAB 的统计对象都会变化。
- **怎么关**：作者先二选一；随后统一 producer，重跑 per-subject、cohort、null 和 figure，并由 artifact 反写 Methods。

### 3. 发作场主合同未锁定

- **现状**：1–45 Hz 与 1–150 Hz、EEG onset 与 clinical onset、5 s 与 10 s 平滑同时存在；附件把它们揉成一个主分析。
- **为什么严重**：频带、时间零点和时间平滑都会直接改变早期场一致性。
- **怎么关**：预先锁定主版本，其余作为敏感性；同时锁定基线、支持阈值、all-contact/within-shaft null、患者内发作汇总和多重比较。

### 4. 慢变量不能写成已验证的主机制

- **现状**：当前主 SNN 为 slow=None；\(q_I/g_K\) 是探索性 screen，且没有得到可控扩展后恢复，\(g_K\) 主要表现为抑制。
- **为什么严重**：若与主模型连续书写，会把阴性探索误写成已支持的 ictal transition 机制。
- **怎么关**：正文主 Methods 只写静态低阈值核心 + 各向异性连接；慢变量移入 Supplementary exploratory methods，并报告 bounded negative。

## P0-轴：source/sink 命名、swap 定义与 A/B 病理轴构建（2026-07-12 讨论新增）

> 这是对上面 §2、§9 的深挖：不只是"命名统一"，而是"当前 source/sink 轴的强版本到底站不站得住"。核实过代码与实测 artifact 后写。

### A. 代码现状核实：是 source→sink 轴，但仓库里有两套不一致的"端点"定义

主几何确实按 source→sink 建轴：轴 = sink 质心 − source 质心（`src/propagation_skeleton_geometry.py::compute_axis_frame`）。但"端点"在仓库里有**两套定义**，服务不同问题、且不是同一批触点：

- **def-a（各模板自有端点）**：对单个模板取最早 3 / 最晚 3 个有效触点当 source/sink（`build_endpoint_cores(..., k_primary=3)`，n_eff∈{5,6} 时 fallback k=2）。paper 主图那条"同轴相反读取 cos=−0.977、7/10"就是它——dominant cluster 建一条轴、minority cluster 独立建另一条，比两条轴方向余弦（`run_propagation_skeleton_geometry.py::axes_cos_angle`）。descriptive-only，无显著性。
- **def-b（swap-k 端点）**：对 A/B 模板对取 decision_k 个"角色互换"触点（`compute_swap_score_sweep` → `derive_swap_endpoint`）。这条喂的是 SOZ set-relation（§9）和 broad-pool 验证，**不喂主几何轴**。

**两套并非同一批触点。** `results/lagpat_broad/geometry_validation.csv` 直接量了两者 Jaccard：strict-swap 病人普遍只有 ≈0.3，只有 decision_k 恰好=3 的少数病人到 1.0。

### B. "source / sink" 命名过强（overclaim 风险）

- **测量本体**：`t_M` 是事件窗内 50–300 Hz 谱能量的三次幂加权时间质心（母稿 §"事件内时间质心"）。母稿自己已写"不解释为神经元放电或病理传播的精确生物学起始时刻"。
- 因此 source/sink 的字面内容只是"这个触点的高频能量在共同事件里**先亮 / 后亮**"。叫 source/sink 会把"先亮/后亮"读成"发射源/汇聚点"或"驱动/被驱动"，测量撑不起这层因果或生成方向。
- **建议**：正文改中性词——leading / trailing endpoint、early / late endpoint、"事件内激活顺序轴的两端"；把 source/sink、propagation 这类方向/因果词留到 Discussion 并标为 hypothesis。

### C. swap 定义 + "A 的 source 是不是 B 的 sink"

同一个"swap"盖住了两个不同问题：

1. **方向共线（弱版本）**：两个模板各自建轴、方向近似反向（cos≈−1）。基本等价于"B 是 A 的整体倒序"（forward_reverse，Spearman r<−0.5）投到 3D。
2. **端点身份互换（强版本）**：A 里最早的那几个触点，是否正好就是 B 里最晚的那几个（`swap_node_groups_at_k`：source_in_A = bottom_k_A ∩ top_k_B）。这才是"source→sink 互换"的字面主张，有置换零假设。

**实测强版本站不住**：能过显著性（swap_class=strict）的病人 decision_k 几乎都饱和在 n_valid/2（`dk_saturated=True`），此时"swap 节点"≈全部参与触点、不是一对干净端点，def-a/def-b Jaccard≈0.3；反过来 decision_k 小、端点干净的病人（如 zhaojinrui/chengshuai）大多只到 candidate/none。也就是——**能过检验的 swap 本质是"整体倒序"，不是"某几个特定触点从源变汇"**。（与 topic4 H2b v1.0.2 降级、`swap 几何主张纪律` memory 一致：perm null 在 decision_k≈n/2 饱和，只证伪 orthogonal-source。）

**结论**：现在能写的是弱版本（两个主排序大致互为倒序、空间上近似共线）。"同一条病理轴、两个方向读取、且是同一批触点扮演互换角色"这个强版本，当前统计撑不住。

### D. 母稿事实错误：轴端点用的是固定 k=3，不是 decision_k

母稿 §"三维传播端点与传播轴"写"使用 decision_k 定义两个方向模式的 source 和 sink 触点"。但主几何 runner 用 `build_endpoint_cores(..., k_primary=3)` 固定 3（fallback 2），**不是 decision_k**。decision_k 只进 SOZ set-relation 和 broad 验证。作者定了下面 E 之后，这段必须按实际 producer 改写。

### E. 怎么构建病理轴：核心口径改用 D_AB（**必须补的逻辑**，2026-07-12 定向）

**这是一定要补的逻辑缺口。** 方向已定：把轴的核心定义从"source/sink 端点 + swap"改成**连续的 A/B 早晚偏好场 D_AB**。理由是 D_AB 才和真实信号对得上——上面 §C 已证明能过检验的"swap"本质是"B 整体倒序"，不是几个特定触点互换；D_AB = 每触点"在 A 里有多早"减"在 B 里有多早"，正是把这个整体倒序如实写成连续场，不强行局部化、不锁 k、不背 source/sink 的因果 overclaim。

D_AB 已存在于 V3d：`src/topic5_scaffold_ab_contrast.py::build_D_AB`（`eA=-zscore(rank_a)`，`eB=-zscore(rank_b)`，`D_AB=eA-eB`），约等于 z 标准化的 Δr = rank_b − rank_a。

**但 D_AB 本身不是轴——这才是本 open question 的承重部分：**

- D_AB 是每触点一个数（排名空间标量）。"轴"是脑内一个方向，必须把 D_AB 摊到三维坐标上才立得起来：轴 = D_AB 在空间里被组织的主方向（对坐标回归取梯度，或取 D_AB 两极质心连线）。
- **只有当 D_AB 空间上极化成两团分开的极**（偏 A、偏 B 的触点各自成片且空间分得开）才谈得上真"病理轴"。若两极其实是**同一根电极杆的两头**、或空间交错/散乱，那是杆内梯度或噪声，不是轴。
- 因此"swap"这个词与集合重合检验可以退休；但它当初想回答的问题（A/B 角色差异在空间上是不是一条真两极轴 vs 一根杆梯度）**不退休**，改用"D_AB 是否空间两极化"来问，并用同杆内打乱零假设把关（V3d `axis_present` 的 within-shaft shuffle 已是此闸）。

**两处代价（定案前看清）：**

1. D_AB 只在"互为倒序"的对（`rho_AB ≤ −0.5`，`template_pair_tier=reciprocal`）上有意义；换核心口径**不扩大**队列，仍是"有对立模板"子集。
2. "以及之后的分析"是一次真迁移：held-out 轴验证、V3c SOZ 覆盖等现在都吃 source/sink 端点，改 D_AB 口径要连带重跑，是小工程、需显式决定。

**降级项：** 原"AB 各自建轴 + 余弦 cos≈−1"（def-a，见 §A）降为**模板反向的补充性几何演示**，descriptive-only；它与 `rho_AB` 投到空间是同一反向事实的另一种画法，按 CLAUDE.md §7 不得当独立证据重复主张。

**两个角色别混（同一 D_AB，两种承重）：** Core 1 用 D_AB **定义间期轴** = 承重（必须证明空间两极化过同杆零假设）；V3d 那张 D_AB **触点上色图** = 补充插图、不承担检验（其检验在 C_AB 时序）。别让"只是插图"的定位渗到轴定义上。

**下一步（正在做）：** 先在信息量最足的被试上把 D_AB → 三维轴这步搭出来、可视化，目视判定两极到底分不分得开、还是一根杆——这是本 open question 成不成立的命门。

## P1：不一定推翻结果，但会削弱可复现性

### 5. Yuquan 伦理表述需原始文件核对

- 核对批准号 2016005、伦理委员会法定名称、批准范围和未成年人同意流程。
- EPILEPSIAE 三家委员会名称按原始数据库论文统一；没有来源时不加“法定监护人”。

### 6. EPILEPSIAE “30 选 20、采样率 >500 Hz”缺逐人证据

- 当前本地 inventory 不能直接复现这句话。
- 需要 Table S2 增加原始 cohort、采样率、artifact 可用性和排除原因。

### 7. 不能笼统写两类数据共享预处理

- Yuquan 为患者特异性 drop list + bipolar + 800 Hz。
- EPILEPSIAE v2 为 CAR + 原采样率，旧 sub_dropChns 已退出主合同。
- 发作期与间期来自不同 producer；逐项核对前不能写“完全相同”。

### 8. Group-event 患者特异参数尚未形成投稿表

- 需要逐人列出 pick_k、pack_win_sec、pack_top_n、特殊通道处理和 artifact 文件。
- 否则“预设比例/固定窗口”不可复现。

### 9. 空间轴的命名需要统一

- 当前可靠主合同是 rank-displacement swap-\(k\) source–sink 轴。
- “模板 A/B 占比命名”“共同病理轴”“两独立模板轴”是不同对象，不能互换。

### 10. SNN 参数需注明适用 artifact

- 20 mm、40,000 神经元、\(AR=2\)、\(l_{EE}=0.380\) mm、双核心阈值参数对应已锁定工作点，不应被写成所有 runner 的通用默认。
- 需要补 simulation seed、积分步长、总时长、突触权重、传导参数、OU 参数和软件版本表。

### 11. 虚拟 SEEG 的两种 \(k_{\rm dir}\) 要分层

- 规则密集读出用 3，患者稀疏读出可用 2。
- 两者的最小参与触点数和方向估计精度不同，结果中不能合并成一个无条件定义。

### 12. 两队列不能统一称为 SEEG cohort

- Yuquan 是 SEEG；EPILEPSIAE 纳入记录包含 depth、strip 和 grid 等不同植入形式。
- 总体层面应写 two intracranial EEG cohorts；只有数据集特异描述时才写 Yuquan SEEG。

## 建议的最短闭环

1. **补 Table S1 剩余临床源**：向医院补 Y19 介入/结局/随访、Y20 长期结局/随访，以及两例 implantation sheet。
2. **再锁统计对象**：Kendall \(\tau\) / masked MI 二选一；field plane、频带、onset 和时间窗一次性锁定。
3. **只重跑受影响链**：field + null + figure；若选择新 MI，再重跑 propagation summary。
4. **最后补复现表**：患者特异参数、SNN 参数、seed、版本和 artifact manifest。
5. **终稿清理**：清除全部 TBC，逐图核对 Methods、结果分母和图注。

## 作者需要直接回答的 11 个问题

1. 能否提供 Y19 的 intervention/outcome/follow-up、Y20 的 long-term outcome/follow-up，以及两例 implantation sheet？
2. Y19/Y20 的长期 outcome 应按 Engel、ILAE 还是原始自由文本报告？
3. MI 是否接受改成 shared-participant Kendall \(\tau\) 为主、legacy MI 为敏感性？
4. 二维场用每模板独立 source–sink plane，还是新建共同 A–B plane？
5. 发作场主频带是 1–45 Hz 还是 1–150 Hz？
6. 发作时间零点用 EEG onset 还是 clinical onset？
7. 慢变量是否同意只放 Supplementary exploratory methods，并明确阴性边界？
8. 能否提供 Yuquan 伦理批件或原始论文中的伦理原文用于最终核对？
9. 主文里 source/sink 是否换成中性的 leading/trailing endpoint（把源/汇的因果读法留到 Discussion）？（见 P0-轴 §B）
10. 【已定向，待命门验证】核心轴口径改用连续 D_AB 场（= V3d `build_D_AB`），退休 swap 集合检验、把"AB 各自建轴 + 余弦"降为补充反向演示。命门 = D_AB 是否空间两极化（vs 一根杆梯度），先在 E1146 上目视验证。（见 P0-轴 §E）
11. 轴端点大小锁 k=3 还是 decision_k？此决定同时修正母稿 §"三维传播端点与传播轴"的事实错误。（见 P0-轴 §D）
