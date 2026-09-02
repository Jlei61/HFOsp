# Topic 5.2 动力学 motif RNN v0.1-r2：技术收口报告

> **修订通知（2026-08-16）**：本技术收口已被 `dynamical_motif_rnn_v0_2_repair_technical_report_2026-08-16.md` 取代。关键修复包括 contact-only checkpoint、三 seed 容量匹配历史基线、完整 2×2 状态路径消融、包含 \(\gamma\) 的真实前向算子可见度，以及 parity-only seizure 重算。

日期：2026-08-16
Spec：`docs/superpowers/specs/2026-08-16-topic5-dynamical-motif-rnn-v0-1-design.md`（含同日 ERRATUM）
Plan：`docs/superpowers/plans/2026-08-16-topic5-dynamical-motif-rnn-v0-1.md`（含同日 ERRATUM）
结果根：`results/topic5_dynamical_motif_rnn_v0_1/`
设计复核：`results/topic5_dynamical_motif_rnn_v0_1/SCIENTIFIC_DESIGN_AUDIT.md`
状态：**全队列执行完成（420/420 units, 0 failed）；合成可辨识性图 60/60 cell 已终态（14 个拟合失败，见 §10）**

---

## 1. 精确方程与参数化

所有模型共用父代（Topic 5.1 v0.5 / LBSS v0.2）的 full-tissue leaky RNN，`state_dim = 1`：

\[
u_k=(x_kH)\odot g_{in},\qquad
h_{k+1}=(1-\kappa)h_k+\kappa\tanh\!\big(u_k+W(s_k)h_k+b\big),\qquad
\ell_{k+1}=b_{c}+g_{out}h_{k+1}H^\top .
\]

`H` 为 (n_contacts, n_nodes) 观测算子；`x_k` 是第 `k` 个 rank set 的 contact 指示向量。

**索引约定（已对父代代码逐行核对）**：递归张量为 `W[i, j]`，`i` = 接收 node、`j` = 来源 node；
状态更新写作 `h @ W.T`，因此 `(Wh)_i = Σ_j W[i,j] h_j`。列归一化

\[\mathcal P(K)_{ij}=K_{ij}\Big/\big(\textstyle\sum_{i'}K_{i'j}+\epsilon\big),\qquad \epsilon=10^{-8}\]

对**行指标**求和，使每个来源 node 的外发权重和为 1。副作用（重要）：列归一化后
`g` 是唯一增益旋钮，**方向偏置不会顺带改变总注入量**。

### 1.1 四个主模型

| 代码 ID | 递归核 | 新增参数 |
|---|---|---|
| `DM0_ISOTROPIC` | `W=g_0\mathcal P(m_{ij}e^{-\|r_i-r_j\|^2/2\ell^2})` | — |
| `DM1_FREE_AXIS` | 各向异性：`ℓ_∥=ℓe^{η}`、`ℓ_⊥=ℓe^{-η}`，轴 `u=(\cosθ,\sinθ)` | `θ`, `η≥0` |
| `DM2_LOCAL_DIRECTIONAL` | `K^{dir}_{ij}(s)=K^{axis}_{ij}\exp[\beta s\,u^\top(r_i-r_j)/\ell]` | `β` |
| `DM3_AXIS_FEEDFORWARD_TRANSIENT` | 额外 `+\gamma F(u,s_k)h_k`，`F^+_{ij}=K^{axis}_{ij}\mathbf 1(0<q_i-q_j<r_f)`、`F^-=(F^+)^\top`、`F=\max(s,0)F^++\max(-s,0)F^-` | `γ≥0` |

三个 M3 替代机制：`DM3_GAIN_MEMORY`（`g_G=g_2e^{δ_g}`、`κ_G=σ(\text{logit}κ_2-δ_κ)`，`δ≥0`）、
`DM3_SYMMETRIC_MATCHED`（`F_{sym}=|s|(F^++F^-)/2`）、
`DM3_AXIS_SHUFFLED_TRIANGULAR`（轴向序号在冻结的半径/度数 bin 内置换）。

### 1.2 关键实现事实

- **列归一化让来源侧因子精确抵消**：
  `\mathcal P(K^{dir}(s))_{ij}=a_i(s)K_{ij}/\sum_{i'}a_{i'}(s)K_{i'j}`，`a_i(s)=e^{\beta s q_i/\ell}`，
  于是 `W^{(2)}(s)h=g\,a(s)\odot[K(h\oslash c(s))]`。逐事件逐步不同的 `W` 因此**从不显式构造**
  （B=1024、N=339 时那会是 470 MB），每步只要两次 `(B,N)×(N,N)` 矩阵乘。
  对拍稠密矩阵 atol 1e-4（`test_recurrent_drive_matches_the_dense_matrix`）。
- **嵌套等价是逐位的**：不拥有某个 motif 参数的层把它注册为 **buffer=0** 而不是走特判分支，
  因此上下层走完全相同的表达式，`torch.equal` 成立（6 条嵌套关系全部有测试）。
- **非负参数用投影梯度**（每次 optimizer step 后 `clamp_(min=0)`）而不是 `softplus`：
  `softplus` 取不到精确 0，warm start 就不可能逐位等价。
- **`r_f` 冻结为 `r_local_mm`**：实测前向锥包含 45.4%–49.2% 的局部边（28/28 患者，均值 48.1%）。
- **`σ_s`** = calibration split 上 `‖\bar r_2-\bar r_1‖` 的中位数（位移**模长**，
  与学到的 `θ` 无关）；跨患者 1.40–15.91 mm。
- **三个方向承诺规则是一族嵌套阶梯**：
  `a_k=(\bar r_{\min(k,K)}-\bar r_0)/\min(k,K)`，`K=1/2/∞` 对应 2RANK / 3RANK / ONLINE，
  三者在 `k=0` 时均为 0；闭环 rollout 与 teacher forcing 在真实序列上逐位一致（已测）。

---

## 2. Frame 与拟合分母

- **`GEOMETRY_ONLY_PCA2`（CORE_FRAME）**：仅由冻结的 3-D 触点坐标做 PCA2，
  分量按奇异值排序、符号由最大绝对 3-D loading 固定；随后用冻结的 v0.5 规则重建
  tissue mesh、`H`、距离和局部支持（`NODE_SEED=20260812`、`sigma` floor 0 mm）。
  **28 个患者级 fit，0 error**，全部强连通、contact-supported reachability = 1.0。
  节点数 64–339（父代 64–346），zero-`H` 比例 0.226–0.909（父代 0.225–0.919）。
- **`PARENT_FROZEN_FRAME`（敏感性）**：逐位复用 v0.5 的 42 个 fit。
  其第一坐标轴与 TA/TB 传播轴**逐位相同**（见 §3），所以只作 frame 依赖性记录。
- 14 位双视图患者的 `own_a`/`own_b` 的 `events.npz` / `events_raw.npz` SHA256 与 contact 列表
  **完全相同**（28/28 检查通过），因此折叠成患者级 fit 是精确操作。
- 近一维患者 2 位（`epilepsiae_139` S2/S1 = 0.0279、`yuquan_zhangjiaqi` = 0.0230，阈值 0.05），
  与父代分类 28/28 一致；照常训练评分，但不进入任何二维方向结论。

---

## 3. Split 与 target-free 边界

- `split == -1`（model-unseen）与父代 held-out **逐位相等**（`np.array_equal`，28/28）；
  `event_group_count < 2` 的 rank-不合格事件数为 **0**，因此两者没有混淆。
- 分母：train **394,561** / calibration **84,552** / development-test **84,566** /
  **model-unseen 140,938** 事件，28 位患者。
- 模式标签沿用父代 train-only KMeans（`train_only_modes.npz`），
  由**前 3 个 rank** 的 masked 特征 + train-only 中心得到；只作评分目标，**不进入模型输入**。
- 模型代码经 AST 静态检查，执行路径中不出现
  `earliness / seizure / ictal / bb150 / template_field / prefix_posterior / full_train_mode /
  prefix_mode / suffix` 任一标识符（`test_model_code_carries_no_template_or_seizure_dependency`）。
- **父代平面的第一坐标轴就是 TA/TB 传播轴**：`interictal_field.planes.own_{a,b}.u` 与
  `axis_pair.axis_{a,b}.u` 逐位相同，且 `contacts_xy_mm = (coords_3d − origin) @ [u, w]`
  （42/42 fits，atol 1e-4）。这是把全队列放在几何 frame 的数值理由。

---

## 4. 训练、checkpoint 与 decoder 合同

| 项 | 值 |
|---|---|
| optimizer | Adam；新参数 lr 6e-3，共享参数 lr 6e-3 × 0.2（warm start 时） |
| anchor | `λ=0.03` × **每元素均方漂移**（不是求和，故跨患者可比） |
| batch | `min(1024, ceil(n_train/8))`，每 epoch 最多 120 batch |
| 收敛 | `max_epochs=1500`、`patience=40`、`min_relative_improvement=0`、`max_seconds=7200` |
| checkpoint 选择 | calibration split 上 `next_bce + stop_bce`；**warm-start 状态作为 epoch −1 候选** |
| 梯度裁剪 | 5.0 |
| motif 初始化 | calibration split 上的冻结网格（含零点）；轴只在数值探针证明它对前向无影响时才进网格 |
| decoder | 递归冻结后拟合共享 size head（train continue decisions，calibration 选 epoch），再校准三个温度 |
| 采样顺序 | STOP → 下一 rank-set size → 未招募 contact 的精确固定基数子集 |
| 随机数 | 所有模型共用同一 uniform 流（common random numbers） |

**单例快路径**：队列中 tied-rank 比例 < 1.4e-4，绝大多数步骤只取一个 contact；
对这些行用**预抽取的固定形状 uniform 块**上的 Gumbel-max 代替 elementary-symmetric 动态规划，
与精确律的经验偏差在 N=40,000 时 ≤ 0.006（k=1/2/3、有/无 mask 共 6 组）。

---

## 5. 两个 motif 效应量（必须分开读）

1. **component-isolation replay（主）**：恢复上一层的共享参数，只保留本层学到的 motif 参数，
   在 calibration 与 model-unseen 上评分。它不含"共享解还没收敛"的混淆，且不需要训练。
2. **anchored joint fit（次）**：spec §6.1 的主训练方式，报告 family 的最佳受约束解释力；
   它把"motif 有用"和"共享解还能再优化"混在一起。

**为什么必须分开**：`epilepsiae_1077` 上的 λ 扫描显示，
`λ=0 / ratio=1.0` 能拿到 −0.0059 的 calibration 增益，但**最佳 epoch 处 `η` 仍精确为 0** ——
增益全部来自共享参数继续优化。同一诊断也暴露了原收敛判据过早停机
（`patience=20, min_rel=1e-4` 下 DM0 = 0.7428，收紧后 = 0.7257），
因此本轮把 DM0 训到真收敛后才让后续层接手。诊断日志：`run_logs/anchor_diag.log`。

---

## 6. 独立的 dose-response 剖面（不训练）

在每位患者已收敛的上一层解上，对 motif 参数做统一、加宽、加密的扫描：

- `η ∈ {0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.4}` × 12 个角度；
- `β ∈ {0, ±0.15, ±0.3, ±0.6, ±1, ±1.5, ±2, ±3}` × 12 个角度；
- `γ ∈ {0, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3}`。

**选择规则**：calibration split 单独选出**一个**点，held-out 值只在该点读取，
因此报告的 held-out 增益**不是网格最大值**。产物：
`DOSE_RESPONSE_PROFILE.csv`（完整剖面）、`DOSE_RESPONSE_PER_PATIENT.csv`、`DOSE_RESPONSE_SUMMARY.json`。

---

## 7. Monte Carlo 与随机生成

- 所有 model-unseen 事件：32 次 rollout；
- 每位患者 24 个 hash 分层 reference events（按事件长度三分位 × train-only 模板分层，看结果前选定）：128 次；
- 输入扰动每 branch 64 次；
- `FIXED_H3` / `FIXED_H5`（忽略 STOP）与 `FULL_STOP`（保留 STOP、基数、repeat mask、最大 rank 数）分别保存；
- 终点同时报告 `r_last` 与 `r_late`（末 20% rank set 等权均值）；
- 评价量 `S=(r_{last},r_{late},L_{axis},L_{orth},N_{rank},N_{contact},f_{contact})`，
  按该患者 held-out 真实事件的标准差标准化后计能量分数；Monte Carlo 误差由 4 个不相交子批估计。

---

## 8. 唯一工程阻断条件的状态

| 条件 | 状态 | 证据 |
|---|---|---|
| split / event / contact / provenance 错配 | CLEARED | `SPLIT_PROVENANCE_AUDIT.json`、`PARENT_VIEW_CENSUS.csv` |
| TA/TB、未来 suffix、未来基数、seizure 泄漏 | CLEARED | AST 标识符检查 + 因果性反证测试 |
| `η=0/β=0/γ=0` 数值等价 | CLEARED | 6 条嵌套关系 `torch.equal` |
| sampler 重复 contact / STOP 不吸收 / replay 不一致 | CLEARED | 精确律对拍 + 吸收性 + 可复现性测试 |
| NaN/Inf、shape/device、checkpoint 损坏 | 监控中 | 每单元 `numerical_audit` + `DONE.json` |
| geometry cache 不可构造 | CLEARED | 28/28 fits，0 error |

测试：`tests/test_topic5_dynamical_motif_rnn_v0_1.py`，**42 passed**。

---

## 9. 结果

### 9.1 执行完成度

| 项 | 值 |
|---|---|
| 正式 RNN 单元 | **420 / 420 完成，0 failed，0 nonfinite，0 time-limited** |
| 合计 GPU 时间 | **19.30 GPU-hours**（420 单元的 `DONE.json` 求和；中位 49.9 s/unit，最大 2011.8 s，`DM0` 中位 344.7 s）；低成本 baseline 另计 0.79 h |
| 低成本 arm（不计入 420） | `LAYOUT_AXIS_ANISOTROPY` 28、`LAYOUT_AXIS_REPLAY` 28、`EVENT_VECTOR_DIRECTIONAL` 28、`GAIN_MATCHED_*` 56、`STATIC_READOUT` 28、`EARLY_DISPLACEMENT_KINEMATIC` 28 |
| model-unseen 评分 | **560 / 560 checkpoint 全部评分** |
| 输入反事实 | **112 / 112**（28 患者 × 4 主模型，seed 0） |
| 患者数 | 28（几何 frame，患者级 fit） |

### 9.2 G1 各向异性

paired patient-level，正值表示第一臂更好（held-out 精确固定基数子集 NLL，nats）：

| 比较 | n | 中位 | 更好的患者数 | p |
|---|---:|---:|---:|---:|
| free axis − layout axis（M0 上廉价拟合，**上界**） | 28 | −0.000127 | 12/28 | 0.570 |
| free axis − layout axis replay（θ 换成 layout、只重校准 decoder，**下界**） | 28 | +0.000000 | 9/28 | 0.272 |
| layout axis − isotropic | 28 | +0.000127 | 15/28 | 0.352 |
| free axis − isotropic | 28 | +0.000000 | 7/28 | 0.097 |

上界与下界同号且都不显著。**没有任何一条各向异性优于各向同性。**

独立的 dose-response（不训练，calibration 选一个点、held-out 只在该点读）：
`η` 4/28 患者选中精确 0、7/28 落在网格边缘，held-out 增益中位 **0.000000**、12/28 为正、
范围 [−0.0340, +0.0408]。**注意 `η` 与 `β`/`γ` 不同**：12/28 患者的 |增益| > 0.005
（`β` 0/28、`γ` 1/28）。这些个体非零**不构成证据**——其量级落在合成零假设 band 的
95 分位（|增益| = 0.049，见 §10）以内，即"在完全没有 motif 的数据上这条流水线本就会
产生这么大的个体波动"。任何按患者挑出 `epilepsiae_1077`（+0.0408）说"这位有各向异性"
的读法都不成立，同一张表里 `epilepsiae_1150` 是 −0.0340。

### 9.3 G2 方向

| 比较 | n | 中位 | 更好的患者数 | p |
|---|---:|---:|---:|---:|
| directional − free axis | 28 | +0.000000 | 3/28 | 0.719 |
| directional − event-vector directional（无全局走廊） | 28 | −0.000107 | 11/28 | 0.520 |
| directional − 一步幅度匹配版 | 26 | −0.000013 | 7/26 | 0.514 |

dose-response：`β` **13/28 患者选中精确 0**，held-out 增益中位 **0.000000**、9/28 为正、
范围 [−0.0038, +0.0049]。剖面形状是**关于零点的碗**，两侧单调上升。

按触点数分层（此处必须精确，早先一版按 52/38/26/26/24/17/15 列举是**跳过了两位 16 触点患者的挑选式排序**，已更正）：

| 分层 | 选中精确 0 |
|---|---|
| 触点最多的 6 位（52/38/26/26/24/17） | **6/6** |
| 触点 ≥ 15 的 11 位 | **10/11**（例外 `epilepsiae_590`，16 触点，选 −0.15，held-out 增益 **−0.0033**） |
| 全队列 | 13/28 |

所有 `|β| ≥ 1.5` 的极端选择**全部出现在 7–10 触点的患者**
（`yuquan_huanghanwen` +3.0、`epilepsiae_922` −3.0、`epilepsiae_1125` −2.0、
`epilepsiae_139` −3.0、`epilepsiae_635` −2.0、`yuquan_zhangjiaqi` −3.0、`epilepsiae_1096` −1.5），
即**合成图上出现假阳性的那一档**。

早期位移运动学 baseline（无 RNN，只用前两个 rank-set 质心的闭式回归）：
终点误差中位 **11.69 mm**（4.60–41.79），预测方向与真实方向的余弦中位 **+0.915**、
方向同侧比例 **0.886**，模板 Brier 中位 0.173。
也就是说，**"沿着已经走的方向直线外推"本身就能把方向猜对约 89%**——
这正是 M2 想要加的那条信息，而它在序列预测上没有换来增益。

gate replay（冻结 checkpoint，只换方向承诺规则）：三种规则的 held-out contact NLL
队列中位**完全相同**（1.738682）。原因是 **23/28 患者的 `β` 精确为 0**；
仅有的 5 位非零患者中最大规则间差异 **0.000514**。
方向证据的**事件内符号翻转率中位为 0**。

### 9.4 G3 轴向前馈与三类替代机制

| 比较 | n | 中位 | 更好的患者数 | p | Holm |
|---|---:|---:|---:|---:|---:|
| feedforward − directional | 28 | +0.000000 | 2/28 | 0.375 | 0.914 |
| feedforward − gain/memory | 28 | +0.000000 | 11/28 | 0.339 | 0.861 |
| feedforward − symmetric matched | 28 | +0.000000 | 8/28 | 0.500 | 0.914 |
| feedforward − axis-shuffled triangular | 28 | +0.000000 | 8/28 | 0.420 | 0.914 |

dose-response：`γ` **23/28 患者选中精确 0**，held-out 增益中位 0.000000、4/28 为正。

### 9.4b 递归通路承担的是 STOP，不是空间选择（本轮最强的单条结果）

`STATIC_READOUT`（无递归；输入为起始 contacts、累计 participation 与固定 contact covariates，
外加同一个 STOP head）与 `DM0_ISOTROPIC` 在**同一批 model-unseen 事件**上、
经**同一个 `evaluate()`** 打分：

| 指标 | 谁更好 | 中位差 | 患者数 | p |
|---|---|---:|---:|---:|
| next-contact NLL | **STATIC_READOUT** | 0.0289 nats | **21/28** | 0.0247 |
| STOP BCE | **DM0_ISOTROPIC** | 0.0283 | **28/28** | 7.45e-09 |
| top-1 | STATIC 0.3961 vs RNN 0.3696 | — | — | — |

**解释性后果**：在拟合出来的解里，穿过组织的递归通路承担的是
**事件还剩多久（终止）**，而不是**下一步往哪（空间选择）**。
这直接解释了 G1–G3 的全零：三种 motif 都是在给一条**不负责空间方向**的通路加方向。

**必须同时报告的限制**：`STATIC_READOUT` 含两个 `C×C` 触点对矩阵，
在 **17/28 位患者上参数量多于**结构化 RNN（`yuquan_zhangbichen` 5541 vs 643）。
因此"直接触点对读出更准"里**结构与容量不可分**。可以确定的是：
**缺的那一块不是方向**。产物：`STATIC_VS_RECURRENT_PER_PATIENT.csv`。

### 9.5 学到的 motif 参数分布（"全零"是**不准确**的说法）

| 层 | n units | 非零 η | 非零 β | 非零 γ | 回退到 warm start | isolation 增益中位 |
|---|---:|---:|---:|---:|---:|---:|
| `DM1_FREE_AXIS` | 84 | 15 | 0 | 0 | 63 | 0.000000 |
| `DM2_LOCAL_DIRECTIONAL` | 84 | 20 | 17 | 0 | 67 | 0.000000 |
| `DM3_AXIS_FEEDFORWARD_TRANSIENT` | 84 | 21 | 20 | 7 | 74 | 0.000000 |
| `DM3_GAIN_MEMORY` | 28 | 7 | 9 | 0 | 22 | 0.000000 |
| `DM3_SYMMETRIC_MATCHED` | 28 | 6 | 6 | 2 | 25 | 0.000000 |
| `DM3_AXIS_SHUFFLED_TRIANGULAR` | 28 | 8 | 7 | 4 | 23 | 0.000000 |

正确表述：**多数单元的优化回到嵌套等价点；少数确实学到非零 motif，但那些 motif
在 component-isolation replay 下没有换来任何 held-out 增益。**

### 9.6 输入空间反事实（primary 扰动层）

均匀扩散模型（**无任何方向机制**）在 28 位患者上的响应中位：

| 编辑 | 生成终点沿轴位移 | Wilcoxon p | 未匹配参考态 |
|---|---:|---:|---:|
| 沿 `+u` 替换一个触点 | **−1.074 mm** | <0.0001 | 101 |
| 沿 `−u` 替换 | **+0.439 mm** | 0.017 | 125 |
| 正交替换 | −0.117 mm | 1.000 | 93 |
| 相邻 rank 对调 | +0.095 mm | 0.412 | 0 |
| 两 rank 合并 | +0.095 mm | 0.728 | 0 |
| 中段加一个触点 | −0.096 mm | 0.598 | 0 |

**符号是反的，且只在沿轴方向出现。** 这是 repeat mask 的记账后果：
提前消耗一个前方触点，后续可选集合就只剩后方。正交替换无效应正好证实这一点。

**减去均匀扩散模型的响应**（隔离"方向机制自身"）后：六种编辑的中位差
**全部精确为 0**；27 位患者中仅 9 位存在任何非零差异，且正是学到非零 motif 的那一小撮。

### 9.7 模型无关的方向持续性（相邻步位移向量夹角余弦）

- 相对**事件内顺序打乱**：28/28 患者显著为正，中位 excess **+0.0623**，中位 z **28.19**，
  28/28 p < 0.05。
- **但该零假设同时破坏空间连贯性**。合成标定证实这一点：
  在完全无方向的合成 cell（`β=0`）上，同一 excess 已经是 **+0.291**；
  加到 `β=1.2` 才升到 **+0.399**。真实队列的 0.011–0.288（中位 0.062）
  落在"清洁合成"与"高噪声合成（excess≈0）"之间。
- 决定性对照是**真实事件 vs 拟合模型自身生成的事件**：
  相对均匀扩散模型，中位 gap **+0.0139**，**21/28 为正，Wilcoxon p = 0.0008**；
  加了方向机制之后 gap **没有缩小**（directional +0.0169、feedforward +0.0169）。

**结论：数据中确有均匀扩散未复现的方向持续性成分，但本轮三种 motif 都不是它。**

### 9.8 G5 发作复用

时间分辨 1–150 Hz 缓存本轮新建：**17 位患者 / 283 次发作全部缓存**，
逐次发作与冻结 sidecar 的 0–10 s AUC parity 已记录（部分 > 1e-4，分析提供
`--parity-only` 子集开关）。伪发作取 `[-120, -20] s` 每 10 s 一个，共 3113 个。

设计矩阵按触点数预算（`≤ max(2, n_contacts//3)` 列）并加岭正则——
未预算的饱和设计曾在留一根杆的交叉验证下给出恰好 0 与 ±17 的无意义值。

| 判据 | n | 中位 | 正向患者 | p |
|---|---:|---:|---:|---:|
| 真实 onset 的 `ΔE`（>0 表示间期基底有帮助） | 17 | **−2.335** | 3/17 | 0.999 |
| 真实 − 同 block 伪 onset | 17 | +0.444 | 14/17 | **0.022** |

主判据明确阴性。次要项只能表述为"真实 onset 被拖累得较轻"。

### 9.9 G4 变异性

- 输入反事实：见 §9.6（六种编辑的"减去均匀扩散"中位差全部为 0）。
- 生成分布：`MODEL_UNSEEN_PER_PATIENT.csv` 保存 `FIXED_H3` / `FIXED_H5` / `FULL_STOP`
  三档的能量分数、contact-field 能量分数、覆盖率、模板 Brier/log score 与协方差对齐；
  四个主模型之间的所有 distribution 比较中位数均为 0（见 `EVIDENCE_MATRIX.json` 的
  `distribution` 列）。
- 方向持续性：见 §9.7。

### 9.10 G6 残余机制（探索性 sidecar，28/28 患者）

1. **残余场不是低秩的**：真实 contact 参与场减去模型生成场之后，
   有效秩占触点数的比例中位 **0.834**，残余方差占比中位 **0.869**。
   没有"少数几个缺失空间模式"可抓——缺的东西是弥散的。
2. **事件内时间代理携带 rank 顺序之外的距离信息**：
   同一事件内所有有序触点对的距离与 `event_lag_raw` 时间差，
   原始相关中位 **+0.237**；在**控制 rank 步长之后**（步长实测跨度 1–51，非退化控制）
   偏相关中位 **+0.132**，**27/28 位患者为正**。
   本轮所有模型都只用序号步长、完全丢弃这个时间变量，
   所以这是一条落在模型之外的、明确的下一步线索。
   **注意**：`event_lag_raw` 是事件内谱质量中心的时间代理，
   **不是**临床 recruitment time，也不是轴突传导延迟；不得写成传导速度。

产物：`G6_RESIDUAL_SIDECAR_PER_PATIENT.csv` / `G6_RESIDUAL_SIDECAR_SUMMARY.json`。

### 9.11 有限时程增益（`FINITE_NONRETURNING` 登记）

在冻结模型上沿真实轨迹计算 Jacobian 乘积的最大奇异值（112 个单元 = 28 患者 × 4 主模型）：

- 峰值出现在 **第 6 个 rank 步**（中位），峰值增益中位 **1.78**；
- **72/112 单元在峰后回落**（有界瞬态）；
- **40/112 单元有限但不回落**，按 spec §5.4 登记为 `FINITE_NONRETURNING` 科学结果，
  **未因不符合"有界瞬态"预期而删除**。

---

## 10. 失败、欠功率与不可辨识项

- **合成可辨识性图（已完成，60 cell × 2 sweep = 120 次读数，3 shard，
  `toy_identifiability/IDENTIFIABILITY_{GRID,PROFILE,SUMMARY}.{csv,json}`）**。
  生成规则是**触点级**椭圆输运（不是本轮模型的组织级扩散算子），
  拟合与打分走**真实数据完全相同**的流水线。

  **(a) 14/60 cell 拟合失败**（epoch 0 非有限分数），全部集中在
  `small`/`few_events` × 高噪声档：`dm{0,1,2,3}_*_{few_events,small}_noisy`。
  这 14 个是**缺口**，剩 92 次读数进入统计。

  **(b) 假阴性是主要限制**。66 次"真值非零"的读数里：
  **28 次选中精确 0**、符号仅 **31/66** 正确、held-out 增益为正仅 **18/66**、
  选中值/真值中位比 **0.00**。`medium` 档符号正确率反而最低（5/30），
  即大 cell 不是更准，而是**更坚决地一律返回零**。

  **(c) 假阳性存在且集中在 small 档**。24 次"真值为零"的读数里 13 次给出正增益，
  最坏一例 `dm0_s0.00_small_clean` 的 `η` 选中 +2.4、held-out 增益 **+0.3157**；
  该 band 的 95 分位 |增益| = **0.0489**、中位 +0.0006。
  `medium` 档最大仅 +0.0061。

  **(d) 对真实队列的校准结论**：`β` 的全队列范围 [−0.0038, +0.0049] 比该 band 小一个数量级；
  但 `η` 的范围 [−0.0340, +0.0408] **落在 band 之内**，因此 `η` 的逐患者非零
  不能读成信号。合并起来：G1/G2 只能报 `NOT_DETECTED_UNDERPOWERED`，
  **不能报排除**。

  **(e) 不可分离的混淆**：生成器是触点级规则、模型是组织级算子，
  "找不回"里既有精度不足也有型式不匹配，本轮**分不开**。
- **生成器两次失效已修**：第一版 logit sd 仅 0.17（近似均匀采样）；
  第二版对比度修好后仍无信号，因为**统计量选错**——沿固定轴的漂移在这一族模型下
  必然平均为 0（第一步无偏）。最终生成器改为透明的触点级椭圆输运规则。
- **`epilepsiae_442` 3 次发作、`epilepsiae_620` 1 次发作**等的重算 AUC 与冻结 sidecar
  parity > 1e-4，已逐次发作记录。
- **`yuquan_xuxinyi`** 的发作 `ΔE` 为 −24.2（3 次发作、15 触点、4 折），是中位数统计的离群点。
- **checkpoint 选择指标混合了 next-rank 与 STOP**。dose-response 与 grid landscape
  已全程另记 contact NLL，`epilepsiae_1096` 等患者上两者会选中不同的 `θ`。
- **frame experiment（parent frame）与 self-feeding 未执行**：
  `UNIT_MANIFEST_PARENT_FROZEN_FRAME.csv` 已生成（8 fits × 7 models = 56 unit），
  但 CORE_FRAME 已由用户在结果之前指定，且 spec ERRATUM P0-1 给出了数值理由，
  故未消耗额外算力。这是**记录在案的缺口**，不是已完成项。
- **S2 动态分支未运行**：时间分辨缓存已具备，但 S1 主判据明确阴性，
  按 spec §10.3"S2 不可用不阻止 S1"的对偶，本轮不投入；标记 `NOT_RUN`，非 `NOT_IDENTIFIABLE`。

---

## 11. 允许与禁止的论文措辞

### 可以写

- **穿过组织的递归通路在拟合解里承担终止判断而非空间选择**：
  相对无递归的触点对读出，STOP 在 **28/28 患者**上更好（p = 7.5e-9），
  next-contact 在 **21/28 患者**上更差（p = 0.025）。
  报告时必须同时给出参数量限制（17/28 患者上静态臂参数更多）。

- 在这一族局部传播算子里，**各向同性局部扩散在 28 位患者上未被任何各向异性、
  早期位移方向偏置或轴向前馈改进**；held-out 增益中位精确为 0，
  剂量曲线在零点取到内部极小。
- **该结论不依赖坐标系作弊**：几何 frame 完全不读传播方向，
  而父代 frame 的第一轴与模板传播轴逐位相同。
- **自由轴不优于植入布局轴**，上界与下界两种算法都不显著。
- **单向前馈与"更强更慢""对称耦合""打乱轴序三角"三种替代机制不可区分**，
  因为四者学到的强度都是零。
- **可观测早期输入的编辑确实改变生成**，但该效应可完全归因于"触点不可重复"，
  加入方向机制后中位变化为 0。
- **真实事件的方向持续性超过拟合模型自身生成的事件**（中位 +0.0139，21/28，p = 0.0008）。
- 发作早期场**未显示需要间期传播成分**；加入后留折误差中位变差 2.34。

### 不能写

- 不能说 RNN 恢复了真实 connectome、白质束或突触连接。
- 不能说几何主轴或自由轴是病理解剖通路。
- 不能把"未检出方向"写成"组织无方向性"——只在"局部算子预测下一触点"这一任务上成立。
- 不能把三种替代机制"分不开"写成"它们等价"。
- 不能把"真实事件比顺序打乱更持续"写成方向性传播证据——该零假设同时破坏空间连贯性，
  无方向合成 cell 的同一 excess 已达 +0.291。
- 不能写 seizure 场含间期成分。
- 不能把 events / rollouts / seeds / views 当作样本量；样本量是 **28 位患者**。
- 不能说"所有模型学出来的 motif 都是零"（见 §9.5）。
