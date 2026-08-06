# Methods 待确认问题

> 目的：把“作者需要确认”“需要改代码重跑”和“可直接写入”分开。P0 清零前不建议把 Methods 送入正式投稿稿件。

## P0：会改变方法或科学结论

### 1. MI 的文字定义与执行代码不一致 — ✅ 已解决（2026-07-12）

- **原问题（核实属实）**：旧 legacy MI 在事件得分时用完整有限 rank，纳入非参与触点的 phantom 伪秩（`compute_legacy_mi` 对全部通道对打分、非参与位置为有限伪值），与"仅共同参与触点"文字定义是不同统计量。
- **实测方向**：phantom 是稀释而非伪造——去掉后 36/40 患者 MI 反而更强、40/40 显著全部保持，故修正只加强主结论、不推翻它。
- **已实施**：`compute_legacy_mi` 主字段改为 masked shared-participant（事件仅在共同参与触点打分、置换零分布在参与集合内重排秩），旧全通道版保留为 `unmasked_sensitivity`；TDD 3 测试 + 51 回归全绿。
- **全量重跑**：`--augment-masked-mi --masked-features` 重算 40 例 → masked 40/40 显著（cohort median 0.228；unmasked sensitivity 同 40/40、median 0.188）。产物 `results/interictal_propagation_masked/`；Figure 1 paneld1 已按 masked 重画。

### 2. D_AB 三维梯度轴已定向，但尚未迁移到正式 producer

- **现状**：轴定义已改为全部有坐标联合触点上的 \(D_{AB}\) 三维最小二乘梯度。现有绘图脚本已有原型，但正式 geometry/readout producer 仍使用各模板 source–sink 端点轴。
- **为什么严重**：轴坐标、核距离、mirror null 和 maxAB 的统计对象都会变化。
- **怎么关**：将梯度轴抽成纯函数，迁移 geometry/readout producer，补退化与稳定性测试，然后重跑 per-subject、cohort、null 和 figure。替换方法见 methods_axis_gradient_rewrite.md。

### 3. 发作场主合同未锁定

- **现状**：1–45 Hz 与 1–150 Hz、EEG onset 与 clinical onset、5 s 与 10 s 平滑同时存在；附件把它们揉成一个主分析。
- **为什么严重**：频带、时间零点和时间平滑都会直接改变早期场一致性。
- **怎么关**：预先锁定主版本，其余作为敏感性；同时锁定基线、支持阈值、all-contact/within-shaft null、患者内发作汇总和多重比较。

### 4. 慢变量已进入 Figure 5 候选，但不能写成已验证的完整发作机制

- **现状**：Fig. 4 基础 SNN 仍为 `slow=None`；Figure 5 当前候选消费 \(q_I\) 耗竭进入 operational runaway 的连续轨迹，承担 `same scaffold, different state` 的 observation-layer bridge。\(g_K\) 未形成可靠 rescue，当前没有终止/恢复。
- **为什么严重**：若不区分 Fig. 4 baseline、Fig. 5 state-readout candidate 和完整 seizure-cycle 机制，会把单 seed runaway 可视化误写成已验证的 ictal transition。
- **怎么关**：正文可写 Figure 5 候选的 \(q_I\) 状态轨迹与读出定义；同时明确 runaway 是 sustained-rate 操作定义、不是临床发作或解析 separatrix，终止/恢复与机制 ablation 仍未闭合。\(g_K\) 阴性 screen 继续留在 Supplementary exploratory methods。

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

### E. 怎么构建病理轴：核心口径改用 D_AB（2026-07-13 已定向）

轴定义已经锁定为连续 D_AB 三维梯度：对 joint-valid 且有坐标的全部触点计算 \(D_{AB}=e_A-e_B\)，中心化坐标和值后拟合 \(y_c=X_c\beta\)，以 \(\beta/\|\beta\|\) 作为由 B-lead 指向 A-lead 的单位轴。该定义使用全部触点，不使用端点、source/sink、decision-k 或固定 k。

正负 D_AB 各三分之一触点的质心只用于显示两极和报告质心间距，不参与 beta 或轴方向估计。绘图中的轴箭头必须平行于 beta；当前原型连接两极质心画箭头，尚不符合这一边界。

现有原型已经完成 D_AB 构造、joint-valid/坐标筛选、三维最小二乘梯度、R²、Moran's I、杆内方差比例和极端三分位质心。仍缺：

1. 将梯度轴从绘图脚本抽成正式纯函数，并迁移 geometry/readout producer；
2. 输出坐标矩阵秩、条件数以及轴的 bootstrap/leave-one-shaft-out 稳定性；
3. 为该轴单独实现与 observed 完全同构的 null，而不是借用发作能量 axis-present 的 null；
4. 重跑轴投影、二维 field、held-out、SOZ 和后续所有轴依赖分析；
5. 给每个 artifact 写明 axis-definition，避免与旧 source–sink 轴混用。

D_AB 梯度可以对所有非退化 accepted A/B pair 计算；共同病理轴的强解释优先限制在 reciprocal pair（\(\rho_{AB}\le-0.5\)），其余 pair 分层报告。原 A/B 各自端点轴及轴余弦降为历史敏感性或补充性反向演示。
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

- 新正文目标是 D_AB gradient axis；旧 source–sink 轴和两独立模板轴只作为历史/敏感性定义。
- 输出必须记录 axis-definition，避免新旧 artifact 被同一字段名混用。
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

1. **先闭合 cohort crosswalk**：完成 Table S1/S2 与 analysis flow。
2. **迁移轴 producer**：实现 D_AB gradient axis 的纯函数、质量门和测试；同步锁定 field plane、频带、onset 和时间窗。
3. **重跑受影响链**：geometry、field、null、held-out、SOZ 和 figure。
4. **最后补复现表**：患者特异参数、SNN 参数、seed、版本和 artifact manifest。
5. **终稿清理**：清除全部 TBC，逐图核对 Methods、结果分母和图注。

## 作者需要直接回答的 7 个问题

1. 能否提供 Y19 的 intervention/outcome/follow-up、Y20 的 long-term outcome/follow-up，以及两例 implantation sheet？
2. Y19/Y20 的长期 outcome 应按 Engel、ILAE 还是原始自由文本报告？
3. 发作场主频带是 1–45 Hz 还是 1–150 Hz？
4. 发作时间零点用 EEG onset 还是 clinical onset？
5. Figure 5 最终定稿时，是否保留当前“\(q_I\) 轨迹进主图候选、完整慢变量机制与 \(g_K\) 阴性 screen 留 Supplementary”的分层？
6. 能否提供 Yuquan 伦理批件或原始论文中的伦理原文用于最终核对？
7. D_AB 梯度轴的强主张是否只限 reciprocal pair，其余 accepted pair 作为分层/敏感性？
