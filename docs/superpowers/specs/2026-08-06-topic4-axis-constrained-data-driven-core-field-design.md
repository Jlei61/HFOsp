# Topic 4 — 轴约束的数据驱动病理场（data-driven core field）v0.1

- **状态：** rev2（2026-08-06 技术审阅 P0-1..P0-5 后）。Stage 0–1 授权执行；Stage 2 是否开启，由 Stage 1
  的预注册闸门裁定，**不由 Stage 2 的结果裁定**。rev1 从未执行，无结果作废。
- **被试：** E1146（`epilepsiae_1146`）单被试 MVP。队列层不在本合同内。
- **产出根：** `results/topic4_sef_hfo/data_driven_core_field/`
- **分支：** `codex/topic4-data-driven-core-field`
- **不是目标：** 重新发现传播轴；重新设计 SNN；拟合事件率 / 频谱 / 波形；发作期数据参与拟合；
  队列统计；证明"两个模板由两个低阈值核引起"。

---

## 0. 本合同对来源文稿做的更正

来源是一份未接触本仓库代码的设计文稿（含 Spec / Plan / Agent prompt 三份）。逐条对代码核验后，
来源文稿有 **19 处**必须更正（§0.1–0.3、§0.5）；rev1 自身又被技术审阅查出 **5 处**（§0.4）。
**每一处都已在下文对应章节落实**；不更正会导致事实错误、实验跑空或数值失效。

### 0.1 与代码不符的事实（A 类）

| # | 文稿说法 | 代码实际 | 落实于 |
|---|---|---|---|
| A1 | "现有 core 的中心、间距、半径及阈值分布都是预先指定的" | **中心不是预先指定的**。中心 = 患者两个模板各自最早 `k_early=3` 个触点的质心（`src/sef_hfo_subject_placement.py::template_source_foci`）。预先指定的是：个数=2、半径 `core_r=1.5` mm、阈值分布 `N(17.5, 1.0)` vs 基线 `18.0`、core-anchored 模式下的核间距 | §1、§3 |
| A2 | "文稿 N=4000 与 ρ=100 矛盾，不要从密度公式推导预算" | 密度公式是**对的**：`place_neurons` 用 `N = round(density·L²) = 100×20×20 = 40,000`（E 32,000 / I 8,000）。是手稿的 N 写错了，纠错方向反了 | §2.1 |
| A3 | "两个 core 生成正反两个方向的间期事件" | 已接受 Figure 4 Panel C 的 103/119 平衡事件，标签是 `driven_pooled` = 21 seed ×（4 s 只留 source 核 + 4 s 只留 sink 核）。**同一张网自发跑的 forward 事件数中位数 = 0** | §2.3、§5.3 |
| A4 | "MI 使用你们现有的共同参与触点规则" | 仓库里没有互信息。现有的是 **2×2 Spearman 秩相关矩阵**（`scripts/paper_figures/plot_fig_subject_snn_realvsmodel.py::_sim_matrix`） | §5 |
| A5 | `/mnt/data/Pasted text(46).txt`、`/mnt/data/*.png` | 不存在于本仓库 | — |

### 0.2 会让实验跑空的设计错误（B 类）

- **B1** 硬可行性门 `|E₊| ≥ K 且 |E₋| ≥ K`（K=5–10，自发无定向）会把**已接受的自发双核工作点本身**判为
  infeasible（`gradient_shared` seed 5 = 1 forward / 4 reverse）。叠加"禁止任何 rate penalty"⇒ 搜索面
  在绝大多数区域是常数，优化器无从移动。**已按用户批准放宽**：改为处处有定义的分级目标（§5.3），
  双方向移到 held-out 充分性门（§10.3）。
- **B2 最致命，文稿完全未意识到：成功标准没有参照系。** 文稿把"held-out rank score 优于 uniform field"
  写成成功标准，却从未问过**一个完全没有病理场的模型能拿多少分**。实测：仅按触点在轴上的投影排序
  （`axis_only`，零仿真）就有 `S_grad = 0.696`；而自发双核 `0.701 ± 0.173` —— **实质无差**。
  ⇒ 当前模型的分数几乎完全由"沿轴传播 + 模板沿轴排列"解释，绝对分数高不说明任何事。
  → §2.4、§5.4（axis-only 成为承重参照）、§7（整个 Stage 1 就是为这条设立的）

  > rev1 曾用"人工双核基线 0.805、天花板 0.906、头部空间 ≤0.10"论证此条 —— 那个 0.805 取自
  > `driven_pooled` 控制臂（非自发），0.906 是抽样最大值（非上界）。论证已作废并重写，见 §0.4 P0-1。
- **B3** 病理预算 `Σ(V_base − V_th) = D0` 约束的积分**不是机制敏感量**。该网络点火由局部平均阈值门控；
  同样的 D0 摊到 4 倍面积、深度 1/4 的场根本不点火。D0 防不住它想防的"作弊"，只会把搜索空间切歪。
  → §4.3（改为有效个数守恒）
- **B4** 层级顺序在 1D 阶段冻死 `σ_⊥`（横向宽度）。而横向宽度恰是决定"多少神经元被拉到点火阈值以下"
  的主自由度 —— 最关键的旋钮被锁在第一阶段之外。→ §8（纵向 + 宽度同时学）
- **B5** 队列层（文稿 Phase 10）建立在"两个模板相反"上，而这在队列水平是少数（此前正式关门：
  broad 6/26、narrow 9/26）。E1146 属于过关的那批（TA/TB 秩次 Spearman = −0.464），单被试 MVP 成立、
  队列推广不成立。→ §12（队列移出本合同）

### 0.3 数值 / 实现陷阱（C 类）

- **C1** D0 定义自相矛盾：Spec §5 用有符号 `d_i = V_base − V_core_i`（0.4999 mV/神经元）；Agent prompt
  用 `max(0, ·)`（0.6979 mV/神经元），**差 40%**。核内阈值 `N(17.5, 1.0)` vs 基线 `18.0` ⇒
  **31.0% 的核内神经元阈值比基线还高**。`max(0,·)` 在 h=1 时复现不出人工核（丢掉这 31%）；有符号版本
  让二分法 `Σ hᵢ(λ)dᵢ = D0` **非单调**（λ 升高时正负项反向移动），根可能不唯一或不存在。→ §4.3
- **C2** 内存真正的大头是尖峰布尔阵：**320 MB / 每秒仿真**（`(T/dt) × N_E` bool）。文稿只提"不保存
  membrane traces"。→ §6.3（流式包络）
- **C3** 目标模板有**两个来源且不一致**：`template_gradient_fields` 的 `rank_a/rank_b`（15 触点，与冻结
  共享平面同一 artifact）vs `propagation_geometry` 的 `typical_rank`（Panel C 在用）。二者 Spearman =
  0.959 / 0.943。现有 Fig 4 已经混着用（坐标取前者、核成员取后者），文稿完全没提。→ §5.2（两套并列）
- **C4** 事件门有两个且不一致：打分侧 `_model_templates` 用 `n_part ≥ 2·k_dir` = 4；仿真侧
  `subject_run` 用 `2·k_dir+1` = 5。→ §5.1（写死为 5，并回填打分侧）
- **C5** 读出极稀疏：15 个触点，单事件参与中位 5–6、最多 9；`gradient_shared` seed 5 的 34 个检出事件
  里只有 5 个能定方向。用 8–21 个参数拟合它，可辨识性是真问题。文稿把稳定性检查放在事后（其 Phase 5.3），
  本合同**前置为 Stage 1 闸门**。→ §7.4

### 0.4 rev1 自身的缺陷（技术审阅 P0-1..P0-5，2026-08-06）

rev1 从未执行。以下五条全部经代码核验成立，已在 rev2 落实。

- **P0-1 rev1 把 driven-pooled 误当成自发双核基线，整个 headroom 论证建在错误 regime 上。**
  冻结统计文件逐字写着 `"independent_unit": "paired network seed (source-only and sink-only arms)"`，
  每臂 `trace_duration_ms = 4000.1`。rev1 §2.3 已正确指出这一点，§2.4 却把同一批产物称作"人工双核基线"
  —— 自相矛盾。且 0.906 是 21 次抽样的最大值，**不是天花板**；Spearman 理论上界仍是 1，真正的数据上限需要
  患者模板的重复性估计，而我们没有。→ §2.4 全部重写，"天花板 / headroom ≤0.10"及由此推导的固定门全部删除
- **P0-2 评分存在"先按轴分方向、再证明事件像这条轴"的循环，且打分触点集合可被候选自己缩小。**
  已核验 `read_event`：`sign = np.sign(np.dot(ax, axis_unit))` —— 事件方向就是按其早→晚端点轴在 `u_C` 上的
  投影定的；随后再与本身沿 `u_C` 排列的患者模板求 Spearman。另已核验 `_model_templates` 的
  `names` 由该候选自己招募到的触点决定 ⇒ 少招募几个难匹配的触点即可抬高相关。→ §5.2a 冻结打分支撑集、
  §5.4 加 axis-only 参照、§5.5 报告参与覆盖率
- **P0-3 单方向候选可拿 0.75，接近双方向分数，会被描述成"恢复了两类传播"。** → §5.3 改为缺方向的格记 0
  （单方向上界自动 ≤0.5），§10.3 新增 held-out 双方向充分性门
- **P0-4 rev1 §4.3 的"逐 E 神经元预抽"不可能逐位复现人工双核。** 已核验 `_trunc_gauss` 用**拒绝重采样**，
  抽样次数依赖被拒绝多少、且只对 hard-mask 内神经元按 `cidx` 顺序抽；两个核用两个不同 seed 各抽一次再取
  `np.minimum`。rev1 的 TDD"h=1 全域时逐位相等"是**假命题**。且 Stage 1 拿 hard manual 比 soft projected
  会把"硬/软阈值赋值差异"混进"空间形状差异"。→ §4.3 改为统一的 latent-quantile 抽样合同、Stage 1 增加
  第四臂 `manual_projected`、等价性判据改为**行为等价**而非 h 空间相关
- **P0-5 Stage 1 裁决不完备，Stage 2 没有冻结的成功标准。** → §7.5 补全决策表并 fail-closed、
  §7.4 增加跨 seed 排序一致性（`sd(Δ)` 小不等于候选排序可靠）、§8.1 新增预注册结局分类

### 0.5 缺失项（D 类）

- **D1** 文稿没有 go/no-go 探针，第一个可证伪时刻在其 Phase 5，那时已烧掉几十小时。→ §7
- **D2** 文稿未利用最大工程杠杆：**建网 120 s 与病理场完全无关**，同种子可全程复用。→ §6.2
- **D3** 文稿 Phase 1.1 的"患者事件时间块切分"**用现有 artifact 做不到** —— 几何 JSON 只有聚合的
  `typical_rank` / `support` / `uncertainty_rank`，没有 per-event 秩次；且重切会产生**不同的轴**，破坏
  Panel A/B 依赖的冻结坐标合同。**用户已裁定：held-out 只做模型侧。** → §10.1
- **D4** 文稿 Phase 8 跨状态验证假定 z 慢变量能把间期推进发作。该线目前状态是"进得去、出不来、停在
  不动点"，不是现成可接的。→ §12（移出本合同）

---

## 1. 冻结的科学问题

**朴素话。** 患者的间期放电有两类，各自沿同一条方向线上的相反两头往外传。我们已有一个能跑出这种传播的
网络模型，但模型里那两块"容易点着的组织"是我们**手放上去的**：位置取自患者两类放电各自最早响应的
3 个触点，形状则是我们规定的两个半径 1.5 mm 的圆盘。

现在问的是：**如果不预先规定"两块、圆的、这么大"，而是让患者电极记录到的触点先后顺序自己去决定这块
组织长什么样、摊多宽、有几处，同一个网络还能不能跑出同样像患者的传播？**

**形式化。** 在下列全部冻结的前提下：

- 患者共享轴 `u_C`（无向线方向，`[u_C] = {u_C, −u_C}`），来自冻结的 `template_gradient_fields` 共享平面；
- E→E 各向异性长轴 = `u_C`（`theta_deg = −22.8°` for E1146）；
- LIF / 突触 / 适应变量 `m` / 抑制效能变量 `z` / 连接结构 / 延迟 / 空间均匀 OU 噪声 / 虚拟触点坐标 /
  虚拟 SEEG 包络与事件检测 —— 全部不动；

学习一个患者特异的连续空间场 `h_p(x) ∈ [0,1]`，替换掉当前手放的两个圆盘核，使同一个网络在**无定向刺激**
的自发运行下，产生的触点招募顺序尽可能接近患者的两类真实模板。

**唯一学习对象** = `h_p(x)` 的参数（§4）。**其余一切不学。**

---

## 2. 实测可行性事实（本合同的数字地基）

以下全部为本次实测或从已落盘 artifact 直接计算，非估计。任何与之冲突的排期一律以此为准。

### 2.1 网络规模

`place_neurons`：`N = round(density · L²) = 100 × 20 × 20 = 40,000`，`N_E = 32,000`、`N_I = 8,000`
（`f_E = 0.8`）。`V_th` 基线 18.0 mV，`V_reset` 11.0 mV。**手稿若写 N=4000 是手稿错**（A2）。

### 2.2 计算成本（实测，本机）

| 项 | 实测值 | 性质 |
|---|---|---|
| 建网（`place_neurons` + `build_connectivity_rot`） | **120.1 s** | **与病理场无关 → 可缓存复用** |
| 仿真 | **58.5 s / 每 1 秒生物时间** | 随 T 线性 |
| 尖峰布尔阵 `E_spk_bool` | **320 MB / 每秒仿真**（`(T/dt) × N_E`） | 内存主导项 |
| 峰值 RSS（1 s 跑） | 7.1 GB | |
| 一次 8 s 自发跑（网已缓存） | **≈ 7.8 min** | |

机器：80 核、251 GB RAM（218 GB 可用）、1× RTX 3090。**引擎是纯 NumPy CPU，GPU 用不上**
（`src/snn_engine/` 无 torch / cupy / numba）。单进程实测占用约 1.2 核 ⇒ 并行度由内存决定，不由核数决定。

### 2.3 事件产量（自发双核，E1146）

- `gradient_shared` seed 5（8 s）：检出 34 个事件，**只有 5 个**参与触点数够（≥5）能定方向
  → **1 forward / 4 reverse**
- `gradient_shared` seed 4：dir 1/0；`core_r=2.5` seed 4：dir 0/11
- E1146 全部 63 次 `template_source` 自发跑：**forward 事件数中位数 = 0**，33/63 曾出现双方向
- 已接受 Figure 4 Panel C 的 103/119：来自 `lesion="driven_pooled"`，21 seed ×（4 s 单 source 核 +
  4 s 单 sink 核），`duration_ms_per_seed = 8000`
- 触点总数 15；单事件参与中位 5–6、最多 9

**结论：自发双核方向平衡不是这个底物已有的行为。**`docs/paper-draft/figure4_subject_specific_snn.md`
自己写死了边界："控制臂支持'两个端点具有产生相反 readout 的能力'，**不支持**'双核同网长期自发且方向平衡'"。

### 2.4 三个参照分数（rev2 更正 P0-1、P0-2）

用 §5 的交换不变得分对已落盘 artifact 直接打分（免费，不重跑仿真）。**三者属于不同 regime，
不可互相替代**：

| 参照 | 是什么 | `S_grad` | `S_geom` | n |
|---|---|---|---|---|
| **axis-only 几何对照** | **完全没有病理场**：触点只按其在 `u_C` 上的投影排序，forward = +投影、reverse = −投影 | **0.696** | **0.763** | 解析，无仿真 |
| **自发双核 manual** | `twoend_equal` 自发跑、能同时读出两个方向的那些 | **0.701 ± 0.173**（范围 0.399–0.930） | 待 Stage 0 | 63 跑中 **30** 双方向、30 只单方向 |
| **driven-pooled 读出上参照** | 21 seed ×（4 s 只留 source 核 + 4 s 只留 sink 核）的合并读出 | 0.805 ± 0.141 | 待 Stage 0 | 21 |

**（a）driven-pooled 不是基线。** 冻结统计文件逐字写着
`"independent_unit": "paired network seed (source-only and sink-only arms)"`。它回答的是
**"两个端点分别被点亮后，这套 15 触点读出能不能各自还原一个模板"** —— 一个 *readout upper-reference /
endpoint sufficiency* 控制，不是"双核同网自发产生两类事件"的得分分布。本合同不再把它当基线。

**（b）自发双核几乎没有超过纯几何。** `0.701 ± 0.173` vs axis-only `0.696` —— 实质无差。
**当前自发模型的得分几乎全部由"事件沿轴传播 + 患者模板本来就沿轴排列"解释；病理场的形状没有贡献
可见的增量。** 这直接坐实 P0-2 的循环性担忧：`read_event` 先用 `sign(ax · u_C)` 给事件贴方向标签，
再拿它去匹配同样沿 `u_C` 排列的模板。

axis-only 的 2×2 矩阵 `[[0.971, −0.421], [−0.971, 0.421]]` 还说明：对角高达 0.971，非对角只有 −0.421 ——
因为患者两个模板彼此只有 Spearman −0.464，不是严格反向。**交换不变平均把 0.971 和 0.421 平均成 0.696。**

**（c）不使用"天花板"一词。** 0.906 / 0.930 只是有限次抽样的最大值。Spearman 理论上界是 1；
真正的数据上限需要患者模板自身的重复性（split-half / measurement reliability）估计，而现有 artifact
只有聚合 `typical_rank` 与逐触点 `uncertainty_rank`，不足以支撑一个模板级重复性上限（§10.1）。
**在拿到这样的估计之前，本合同不写 ceiling，也不由"天花板 − 基线"推导任何 headroom 或门限。**

**（d）噪声结构。** 在 driven-pooled 池里，`S` 与单跑事件数相关 `r = 0.042, p = 0.855`；事件数 ≥10 的跑
`sd = 0.099`（n=14），<10 的 `sd = 0.204`（n=7） ⇒ **噪声主要来自网络实现而非事件抽样，加长单跑到
~10 事件以上后收益就平了。** 自发池的边际 `sd = 0.173` 与之同量级。

**（e）由此定下本合同的真正参照系。**

> **要证明的不是"分数高"，而是"优化后的场超过 axis-only 与同预算 uniform-axial"。**
> 绝对 Spearman 高不构成证据 —— 纯几何就能拿到 0.696。

CMA-ES 是否可行，取决于 common random numbers（同代候选共用同一张网）能把**配对**方差压到多小，
以及**候选排序是否跨 seed 一致**。二者从未被测过。Stage 1 就是测它们（§7）。

### 2.5 病理预算的符号问题（C1 的证据）

核内 `V_th ~ TruncNormal(17.5, 1.0, lower=11.0)`，基线 18.0：

- `P(V_th,core > 18.0) = 0.310` —— **31% 的"核内"神经元阈值反而更高**
- 有符号：`E[d_i] = 0.4999` mV/神经元；`max(0,·)`：`0.6979` mV/神经元（**+40%**）
- 负值部分占正值部分的 28.4%

### 2.6 两套目标模板的一致性（C3 的证据）

E1146，15 个共同触点：`spearman(gradient rank_a, geometry typical_rank[t_a]) = 0.959`；
`t_b = 0.943`。高但不同 ⇒ 必须两套并列（§5.2）。

### 2.7 工作树的未提交改动（可复现性风险）

分支创建时，工作树含两处**改变行为**的未提交改动：

- `scripts/run_sef_hfo_subject_snn.py`：参与门从 `2·k_dir` = 4 改为 `2·k_dir+1` = 5；新增
  `gradient_shared` placement；修复"改 `cmrun.KDIR` / `cmrun.PART_MIN` 全局变量对 `read_event` 已绑定
  Python 默认值无效"的真 bug（改为显式传参）
- `src/sef_hfo_subject_placement.py`：新增 `gradient_shared_template_foci`

已快照到 `results/topic4_sef_hfo/data_driven_core_field/preexisting_worktree.patch`（8495 行）。
**Stage 0 必须确认已落盘的 `readout_*.json` 是哪个代码版本产出的**，否则 §2.4 的三个参照分数与新跑的
分数不是 like-for-like。

---

## 3. 冻结 vs 学习

**冻结（优化器不得触碰）**

1. 共享轴 `u_C` 及其无向性；2. E→E 各向异性长轴 = `u_C`；3. LIF / 突触 / `m` / `z` 全部方程与参数；
4. 连接结构、权重、延迟、邻接采样规则；5. 空间均匀背景输入与 OU 噪声模型；6. 患者虚拟触点坐标与
共享平面配准；7. 虚拟 SEEG 包络、事件检测、方向读出（`k_dir=2`）；8. 基线阈值 `V_base = 18.0`；
9. 核阈值抽样分布 `N(17.5, 1.0)`（即"病理组织有多深"不学，只学"摆在哪、摊多宽"）。

**学习（唯一）**

`h_p(x) ∈ [0,1]` 的低维参数（§4.1 / §4.2）。

**手放的、本合同要替换掉的**：核的**个数（2）**、**形状（半径 1.5 mm 硬圆盘）**、**空间范围**。
核**中心**原本已由患者数据决定（A1），本合同把"取最早 3 个触点质心"这个启发式替换为对整条秩次向量的拟合。

---

## 4. 病理场参数化

### 4.1 轴向坐标与基函数（Stage 2 主模型）

对每个兴奋性神经元，在共享平面里取轴向 / 横向坐标：

```
s_i = u_C · (x_i − c_p)        c_p = 两个模板源质心的中点（= register_to_sheet 的 center）
r_i = u_C⊥ · (x_i − c_p)
```

沿轴放 `M` 个固定 Gaussian 基（中心 `κ_m` 等距覆盖患者触点的轴向支撑 + margin）：

```
φ_m(s) = exp[ −(s − κ_m)² / (2 σ_s²) ]
π  = softmax(α)                                    α ∈ R^M，均值固定为 0（消 softmax 平移冗余）
q_i = exp[ −r_i² / (2 σ_⊥²) ] · ( ε + Σ_m π_m φ_m(s_i) )
```

**与文稿的差别（B4）**：`σ_⊥` **参与学习**（对数参数化 `log σ_⊥`），不是固定超参。`σ_s` 固定
（= 基间距的 1.2 倍），只控最小空间平滑尺度。⇒ Stage 2 参数量 = `M + 1`。

`M`、`σ_s`、`ε`、支撑 margin 写入 config，Stage 0 完成后冻结。

峰数不预设：均匀 `π` → 宽走廊；一簇 → 单峰；两簇 → 双峰。**峰数是结果的描述性性质，不是模型类别。**

### 4.2 二维轴约束场（Stage 3）

二维基中心 `(κ_m, η_n)`，椭圆长轴方向**固定为 `u_C`**：

```
φ_mn(x_i) = exp[ −(s_i−κ_m)²/(2σ_∥²) − (r_i−η_n)²/(2σ_⊥²) ]
π = softmax(β)         β ∈ R^{M_s × M_r}
q_i = ε + Σ_mn π_mn φ_mn(x_i)
```

初始网格 `M_s = 7`、`M_r = 3`。场可沿轴移动、可横向偏移、可单峰 / 多峰 / 连续走廊；**方向仍由 `u_C` 约束**。

### 4.3 抽样合同与预算投影（更正 B3 + C1；rev2 重写 P0-4）

**文稿的构造有两个独立缺陷**：有符号 `d_i` 让二分非单调；`max(0,·)` 让 h=1 复现不出人工核。
且 `Σ(V_base − V_th)` 约束的积分不是机制敏感量。

#### 4.3.1 统一的 per-neuron latent-quantile 抽样合同（rev2 更正 P0-4）

**为什么必须换掉 legacy 抽样。** 已核验 `_trunc_gauss` 用**拒绝重采样**：`rng.normal(...)` 循环抽、
丢掉 `< v_reset` 的，抽取次数依赖被拒绝多少；且 `sample_core_field` 只对当前 hard-mask 内的神经元
按 `cidx` 顺序抽 `cidx.size` 个值。人工双核是两次独立调用（`seed+7` / `seed+8`）再取 `np.minimum`。

⇒ 任何"逐 E 神经元预抽一次"的方案，抽取个数、顺序、RNG 流位置都与 legacy 不同，**逐位复现在数学上
不可能**。rev1 的 TDD"h=1 全域时逐位相等"是假命题，已删除。

**统一合同（Stage 0 冻结）：**

```
对每个 E 神经元抽一次分位数（固定 quantile seed，与位置无关、与场无关）：
    u_i ~ Uniform(0, 1)

核阈值由逆变换给出（与 legacy 同分布，但确定性且逐神经元可寻址）：
    V_core,i = Φ⁻¹_trunc( u_i ; μ = 17.5, σ = 1.0, lower = 11.0 )
    d_i      = V_base − V_core,i          # 保留符号，约 31% 为负
```

`Φ⁻¹_trunc` 用 `scipy.stats.truncnorm.ppf`。与拒绝重采样**同分布**，只是与 RNG 的耦合方式不同。

**三个场全部走这一套合同**：`manual_projected`（h = 人工双核指示函数）、`uniform_axial`、`learned`。
legacy 路径保留为 **`manual_hard`**，作为**外部参照臂**，不参与优化。

#### 4.3.2 预算投影

```
软成员度：
    h_i(θ) = sigmoid( [ log(q_i(θ) + ε) − λ_θ ] / τ_h )

约束（二分求 λ_θ）：
    Σ_i h_i(θ) = N_core^manual                 # 病理化神经元的"有效个数"守恒

最终阈值：
    V_th,i(θ) = V_base − h_i(θ) · d_i
```

**为什么这个构造更好：**

1. `Σ_i h_i(λ)` 对 `λ` **严格单调递减** ⇒ 二分法有唯一根，数值稳定（有符号 `d_i` 下 `Σ h_i d_i = D0`
   不单调，故不用它作约束）；
2. `d_i` 的抽样与位置独立 ⇒ `E[Σ h_i d_i] = N_core · E[d] = D0` **自动成立**。总阈值降低量仍守恒
   （在抽样误差内），作为**报告项**验证而非约束；
3. `h = manual 指示函数` 时精确给出 `manual_projected`，与 `manual_hard` **同分布**（但非逐位相同）；
4. 语义干净：**"同样多的病理组织，重新摆放"**，而不是"同样的积分"。摊薄到不点火仍会发生，但那是该候选
   **得分低**的结果，不是作弊 —— 不需要额外的深度约束，因为深度已由 `d_i` 的分布锁死。

#### 4.3.3 表达力与等价性判据（rev2 更正 P0-4）

**空间相关不能代替行为等价。** rev1 用 "h 空间相关 ≥ 0.9" 作等价判据是把可用信号当成了充分条件
（CLAUDE.md §6.1：判据要与问题匹配，不是与形状匹配）。rev2 改为两层：

- **前置廉价 sanity（纯计算，非判据）**：`manual_projected` 的 `h` 与人工双核 mask 空间相关 ≥ 0.9；
  双峰 `α` 生成的 `h` 与人工双核 mask 空间相关 ≥ 0.9。不满足说明参数族根本表达不了旧工作点，直接停。
- **正式判据 = 行为等价（Stage 1 第四臂）**：`manual_projected` 与 `manual_hard` 在**同一批网络种子上
  配对比较**，须同时满足：
  - `|mean(S_projected) − mean(S_hard)|` ≤ 配对差标准差 `sd(S_projected − S_hard)`（即无可检出的系统偏移）
  - 双方向出现率之差 ≤ 2/8 seeds
  - 可评分 seed 数之差 ≤ 1

  **只有行为等价成立，新参数族才算表达了旧工作点**，`uniform_axial` 与 `learned` 的比较才有意义。
  不成立 ⇒ Stage 1 判 `PARAMETERIZATION_MISMATCH`，停止报告。

**其余 TDD：**

- 二分收敛：`|Σ h_i − N_core^manual| / N_core^manual < 1e-6`；
- `h ∈ [0,1]` 逐元素成立；
- 报告项 `Σ h_i d_i` 与 `D0` 的相对偏差记录于每个候选的元数据；
- 轴翻转 `u_C → −u_C` 与 TA/TB 交换下最终 `S` 不变。

---

## 5. 目标函数

### 5.1 复用现有机器，不重造（CLAUDE.md §6.1）

**沿用 `scripts/paper_figures/plot_fig_subject_snn_realvsmodel.py` 的 `_model_templates` / `_sim_matrix`
的定义**：

- 模型 forward 模板 = 所有 `sign > 0` 事件的**逐触点平均事件内秩次**；reverse 同理（按事件符号，
  **不按 KMeans 簇标签**）；
- 2×2 Spearman，行 = 模型 forward/reverse，列 = 患者 t_a/t_b。

**但不能直接调用该文件的函数**：§5.2a 要求冻结支撑集与统一缺失规则，而 `_sim_matrix` 用的是
每个候选自己的共同触点（这正是 P0-2 指出的作弊通道）。**且不修改该文件**（它承载已发表数字，
CLAUDE.md §3 外科式改动）。⇒ 在本合同自己的评分模块中重新实现上述定义，并加上 §5.2a 的支撑集合同。
实现后须有一个回归测试：在"仅共同触点"模式下，本模块与 `_sim_matrix` 对同一批已落盘事件给出相同矩阵。

**更正 C4**：事件纳入门统一为 `n_part ≥ 2·k_dir + 1 = 5`（与 `subject_run` 一致），
并**同时**用门 = 4 各算一份落盘。两个门给出不同闸门裁定 ⇒ `SOURCE_DISAGREEMENT`（§7.4(E) / §7.5）。

### 5.2 交换不变得分（两套模板并列 —— 用户裁定）

共享轴无向 ⇒ TA/TB 标签必须允许交换：

```
S(target) = max( [M(fwd,t_a) + M(rev,t_b)] / 2 ,
                 [M(fwd,t_b) + M(rev,t_a)] / 2 )
L_rank    = 1 − S
```

**两套目标模板全程并列计算**（C3）：

- `S_grad`：`template_gradient_fields` 的 `rank_a` / `rank_b` —— **主口径**（与冻结共享平面同一 artifact，
  坐标框架自洽）
- `S_geom`：`propagation_geometry` 的 `typical_rank` —— **并列口径**（与已发表 Panel C 同源，数字可直接对比）

**优化只用 `S_grad`。** `S_geom` 全程计算并落盘。**任何主结论必须两套都成立**；只有一套成立时，
只能报告为"依赖模板源的结果"，不得写成主张。

### 5.2a 冻结打分支撑集（rev2 新增，更正 P0-2）

**问题。** `_model_templates` 的 `names` 由**该候选自己招募到的触点**决定，`_sim_matrix` 又只在
`set(model) ∩ set(data)` 上算 Spearman。⇒ 一个候选只要**少招募几个难匹配的触点**，就能在更小的支撑集上
拿到更高的相关，而不是真的更像完整模板。这是一条必须堵死的作弊通道。

**合同（Stage 0 冻结，写入 config）：**

1. **打分支撑集 `SUPPORT` 在优化开始前冻结**，来源 = 患者模板侧有定义的触点 ∩ 落在 sheet 内的虚拟触点。
   E1146 预期为全部 15 个（`valid_contacts = 15/15`），Stage 0 核实后写死。
2. **所有候选用同一个 `SUPPORT` 和同一条缺失规则**：候选未招募到的触点，其模型秩次记为**缺失**，
   按下述统一规则处理，**不得**把支撑集缩小到该候选招募到的子集。
3. **缺失规则（预注册）**：Spearman 在 `SUPPORT` 上计算，缺失触点参与但取该方向所有已招募触点的
   **平均秩次**（即"没被招募 = 无信息 = 排在中间"）。此规则对所有臂、所有候选一致。
4. **敏感性**：同时用"仅在共同触点上算"（= 现有 `_sim_matrix` 行为）各算一份落盘。两者若给出不同的
   闸门裁定，视为不稳定，按 §7.5 停止报告。

### 5.3 可行性 —— 分级，不是硬门（更正 B1；rev2 修正 P0-3）

文稿的 `|E₊| ≥ K 且 |E₋| ≥ K` 硬门取消。**分级规则改为不含任何发明常数：**

```
两方向都有 ≥1 个可定向事件   →  S = max over assignment of  (M_11 + M_22) / 2
只有一个方向有可定向事件     →  缺失方向的两格记为 0（不是 NaN、不是丢弃）
                                 S = max over assignment of  (M_present + 0) / 2   ⇒  S ≤ 0.5
一个可定向事件都没有         →  S = −1   （交换不变平均的理论下界）
```

**rev1 的 `−0.25` 惩罚与 `S_FLOOR = 0.294` 两个发明常数已删除。** 新规则的性质：

- 单方向候选的**上界自动是 0.5**，**低于 axis-only 参照 0.696**（§2.4）⇒ 不可能出现"稳定单方向场
  被当成成功"的情形（P0-3）；
- 在"只有一个方向"这一档内，`S` 仍随该方向的匹配质量单调上升 ⇒ 优化器有梯度可爬；
- 获得第二个方向时分数出现大幅跃升 ⇒ 搜索被正确地拉向双方向；
- 三档严格有序：`无方向 (−1) < 单方向 (≤0.5) < 双方向`。不需要任何调参。

**双方向不进入 per-candidate 损失**（读出太稀疏，逐 seed 强求会复活 B1 的平坦搜索面），
**但必须回到 held-out 充分性门**（§10.3）。每个候选仍记录 `dir_forward` / `dir_reverse` /
`n_directional` / `bidirectional`。

### 5.4 axis-only 参照（rev2 新增，更正 P0-2）

事件方向由 `sign(ax · u_C)` 定义（已核验 `read_event`），患者模板本身又主要沿 `u_C` 排列 ⇒
**得分的一部分由构造保证**。因此必须有一个"没有任何病理场信息"的参照：

```
axis_only.forward[n] = + proj(contact_n − center, u_C)
axis_only.reverse[n] = − proj(contact_n − center, u_C)
```

在冻结的 `SUPPORT` 上按 §5.2 打分。**已测得：`S_grad = 0.696`、`S_geom = 0.763`。**

**承重规则：**

> 任何"数据驱动病理场有效"的主张，必须是 **`S(learned) > S(axis_only)` 且 `S(learned) > S(uniform_axial)`**，
> 且差值跨 held-out 种子稳定。**绝对 Spearman 高不构成证据。**

**范围限制（必须写进任何对外表述）**：本设计能检验的是"轴上病理场的形状是否改善了触点招募顺序"，
**不能**独立检验"传播是否沿轴" —— 后者已被评分构造预设。

### 5.5 参与覆盖率（rev2 新增，更正 P0-2）

每个候选除 `S` 外必须记录：

- `coverage` = `SUPPORT` 中被该候选至少招募一次的触点比例；
- `mean_n_part` = 逐事件参与触点数均值；
- 两个方向各自的 `coverage`。

**报告规则**：任何被提名为"优于对照"的候选，其 `coverage` 不得低于 `manual_projected` 的
`coverage − 0.10`。低于则标注 `LOW_COVERAGE_WIN`，**不得计入主结论**，只能作为需要解释的观察。

---

## 6. Stage 0 — 复现、审计、工程杠杆

**估计 ~2 h。全部完成前不进 Stage 1。**

### 6.1 基线复现与版本对齐

1. 用当前工作树代码重跑 `gradient_shared` seed 5（T=8000），与已落盘
   `readout_epilepsiae_1146_gradient_shared_corefrozen_cr1p5_s5_20260722.json` 逐字段比对；
2. 判定 §2.7 的未提交改动是否已包含在已落盘 artifact 的产出版本中；
3. **用同一个冻结 scorer（§5.2a 的 `SUPPORT` + 缺失规则）重算三个参照**，得到与本合同同代码版本的数值：
   - `S_axis_only`（解析，无仿真）
   - **`S_manual_hard_spontaneous`** —— 自发双核基线。rev1 缺这一项，是 P0-1 的根因。
     现有 63 次 `twoend_equal` 自发跑（多数为旧 `template_source` 几何）给出 `0.701 ± 0.173`（30/63 双方向），
     **Stage 0 必须在冻结的 `gradient_shared` 几何上重取**，样本量与 Stage 1 的 8 个网络种子对齐；
   - `S_driven_pooled`（21 seed）—— 标注为 *readout upper-reference*，**不是基线**；
   - 三者各出 `S_grad` / `S_geom` 两套（§5.2），以及 §5.2a 的两种缺失规则各一份；
4. 冻结 §5.2a 的 `SUPPORT`、§4.3.1 的 quantile seed、§5.3 的分级规则，写入 `config/` 并落盘校验和；
5. 输出 `model_integrity_report.md`：`N / N_E / N_I / L / density / V_base / core 参数 / N_core^manual /
   D0 / theta_deg / 触点数 / 未提交改动清单 + patch 路径 / git commit`。

**不得**在本分支静默修改原模型的物理结构。发现的任何不一致只记录，不修（除非是确凿代码 bug，且单独提交）。

### 6.2 网络缓存层（D2 —— 最大工程杠杆）

`(network_seed, theta_EE, L, density, AR)` → 连接结构落盘（`.npz`，含 `pos` / `labels` / 各连接索引与延迟）。
命中则跳过 120 s 建网。

**验收：** 缓存命中与不命中的仿真结果在同 seed 下逐位相等（bit-parity 测试）。

### 6.3 流式包络（C2）

`simulate_kick` 目前返回完整 `E_spk_bool`（320 MB / 生物秒）。增加一条只在线累积
虚拟触点包络 + `active_fraction` + 事件摘要的路径，不保留全阵。

- **Stage 1 不强制**（1 s 跑实测峰值 7.1 GB，其中尖峰阵 0.32 GB ⇒ 8 s 跑约 9.4 GB/worker，
  8 worker ≈ 75 GB，在 218 GB 内装得下）；
- **Stage 2 之前必须完成**（更长的跑会 OOM）。
- **验收：** 流式与非流式在同 seed 下事件表逐位相等。

### 6.4 场模块与 TDD

实现 `src/topic4_data_driven_core_field.py`：坐标、基函数、softmax、二分预算投影、`V_th` 构造。
TDD 见 §4.3。**纯计算，不含仿真**，可独立测试。

---

## 7. Stage 1 — GO / NO-GO 配对方差探针

**这是整个合同的闸门。估计 ~25–40 min 墙钟（8 worker）。**

### 7.1 为什么必须先做（rev2 重写，更正 P0-1）

rev1 的理由（"头部空间 ≤0.10 小于噪声 0.141"）建在错误 regime 上，已作废。**真正的理由更强：**

§2.4 测得自发双核 `0.701 ± 0.173` vs axis-only 几何对照 `0.696` —— **实质无差**。当前模型的分数几乎
完全由"沿轴传播 + 模板沿轴排列"解释。这有两种可能，用现有数据**分不开**：

- **(a) 读出看不见场的形状** —— 15 个触点、单事件参与 5–6 个，秩次分辨不出病理场怎么摆 ⇒ 整个反演不可辨识；
- **(b) 手放的双核本来就没比纯几何多提供什么** ⇒ 换一个场有可能提供。

Stage 1 用不到 1 小时把 (a) 和 (b) 分开。**在分开之前投入任何搜索算力都是赌博。**

### 7.2 设计（rev2：4 个仿真臂，更正 P0-4）

固定 8 个网络种子 `{1..8}`，每个种子**建网一次**（缓存），在**同一张网、同一噪声种子**上依次跑 4 个场，
每个 8 s：

| 臂 | 场 | 抽样合同 | 问的是 |
|---|---|---|---|
| `manual_hard` | 已接受的两个半径 1.5 mm 圆盘核（legacy 路径原样） | `sample_core_field` × 2 + `np.minimum` | 外部参照，与已发表工作点同源 |
| `manual_projected` | 同样的双核指示函数，走新参数族 | §4.3.1 latent-quantile | **新参数族有没有表达出旧工作点**（行为等价，P0-4） |
| `uniform_axial` | `α = 0`（softmax 均匀）+ 同预算投影 | §4.3.1 | 摊成走廊后差多少 |
| `transverse_shift` | 双核形状沿 `u_C⊥` 平移 3 mm，同预算 | §4.3.1 | 轴的横向位置重不重要 |

外加 **`axis_only`（解析，零仿真成本）** 作为第五个参照（§5.4）。

共 **32 次仿真 = 256 生物秒 = 4.2 h 串行 ≈ 35–50 min（8 worker）**。内存：8 worker × ~9.4 GB ≈ 75 GB。

**`manual_hard` 与 `manual_projected` 分开是 P0-4 的核心**：不分开就会把"硬 mask vs 软投影的阈值赋值
差异"混进"空间形状差异"，让 `uniform_axial` 的对比失去意义。

### 7.3 测量量

对每个 (臂, 种子)：`S_grad` / `S_geom` × 两种缺失规则（§5.2a）、`coverage`、`mean_n_part`、
`dir_forward` / `dir_reverse`、`n_events`、`n_directional`、运行时、峰值内存。

对每一对臂 `(a,b)`，跨 8 个种子计算**配对差** `Δ_ab,k = S_a,k − S_b,k`，报告
`mean(Δ_ab)`、`sd(Δ_ab)`、**符号一致数** `n_same = #{k : sign(Δ_ab,k) = sign(mean Δ_ab)}`。

### 7.4 预注册判据（rev2 重写，更正 P0-5）

**判据一律用符号一致性（无分布假设、无发明的幅度常数）为主，幅度只报告不设门。**

**(A) 参数族等价（前置，必须先过）** —— §4.3.3 的行为等价三条件，比较 `manual_projected` vs `manual_hard`。
不过 ⇒ `PARAMETERIZATION_MISMATCH`。

**(B) 可辨识性（核心）** —— 在 6 个臂对中（含 `axis_only`），是否**至少存在一对**满足
`n_same ≥ 7/8`（两侧二项 p = 0.070；8/8 时 p = 0.0078）。
- 存在 ⇒ 读出**能**分辨这些场，反演有意义；
- 不存在 ⇒ `READOUT_INSENSITIVE`：这 15 个触点的秩次分不出场的形状，优化再多也没用。

**哪个臂赢不构成闸门，只作为结论记录。** `transverse_shift` 或 `uniform_axial` 胜过 `manual_*`
都是合法且有信息的结果（"手放的双核不是最优"），不是失败 —— 这补上了 rev1 未定义的两个分支。

**(C) 优化可行性 —— 跨 seed 排序一致性（rev1 只看 `sd(Δ)`，不够）**

`sd(Δ)` 小**不等于**候选排序可靠。定义
`concordance` = 8 个种子上、6 个臂对中"该种子内的符号与合并均值符号一致"的比例均值：

| `concordance` | 裁定 | Stage 2 预算 |
|---|---|---|
| ≥ 0.85 | `GO_SINGLE_SEED` | 每候选 1 seed；`M+1 = 8` 维、popsize 10、~40 代 |
| 0.65 – 0.85 | `GO_MULTI_SEED` | 每候选平均 4 seed；`M+1 ≤ 8` 维、≤ 150 次评估 |
| < 0.65 | `NO_GO_UNRESOLVABLE` | 停止报告 |

阈值 0.85 / 0.65 是**明示的工程判断**（不是从任何"天花板"推导的），Stage 1 运行前冻结，不得按结果调整。

**(D) 可评分性** —— 某种子若在 ≥2 个臂上得到 `S = −1`（无可定向事件），记为**无信息种子**。
无信息种子 ≥ 3/8 ⇒ `INSUFFICIENT_SCORABLE`（提示需加长单跑时长，而非改判据）。

**(E) 一致性** —— (B) 与 (C) 必须在**三个正交口径的全部组合**下给出相同裁定：
`S_grad` / `S_geom` 两套模板源（§5.2）× 两种缺失规则（§5.2a）× 事件门 `n_part ≥ 5` / `≥ 4`（§5.1）。
任一组合不一致 ⇒ `SOURCE_DISAGREEMENT`。

### 7.5 裁决器（纯函数，fail-closed）

`stage1_verdict(runs) -> verdict` 必须是**无副作用的纯函数**，按下列顺序短路，**任何 NaN / 缺字段 /
未覆盖分支一律返回 `FAIL_CLOSED` 而不是继续**：

```
1. 任何 (臂,种子) 缺失、S 为 NaN、或 config 校验和不匹配      -> FAIL_CLOSED
2. 无信息种子 >= 3/8                                          -> INSUFFICIENT_SCORABLE   [停]
3. (A) 参数族行为等价不成立                                    -> PARAMETERIZATION_MISMATCH [停]
4. (E) 模板源 x 缺失规则 x 事件门 的任一组合裁定不一致           -> SOURCE_DISAGREEMENT      [停]
5. (B) 不存在 n_same >= 7/8 的臂对                             -> READOUT_INSENSITIVE      [停]
6. (C) concordance < 0.65                                     -> NO_GO_UNRESOLVABLE       [停]
7. (C) concordance in [0.65, 0.85)                            -> GO_MULTI_SEED
8. (C) concordance >= 0.85                                    -> GO_SINGLE_SEED
```

**所有 `[停]` 分支：停止，写报告，等用户决定**（用户 2026-08-06 已裁定此分支）。
裁决器须有单元测试覆盖全部 8 条出口，包含 NaN、缺方向、模板源冲突三类构造用例。

### 7.6 交付物

`results/topic4_sef_hfo/data_driven_core_field/stage1_variance_probe/`：
`per_run.csv`（32 行 + `axis_only` 解析行）、`pairwise_deltas.csv`（6 臂对 × 8 种子）、
`gate_verdict.json`（(A)–(E) 全部数值 + 裁决器出口 + config 校验和）、
`figures/`（配对差点线图 + 五臂分布图 + 覆盖率图 + `figures/README.md` 中文说明）。

---

## 8. Stage 2 — 低维场优化

**仅在 Stage 1 裁决器返回 `GO_SINGLE_SEED` 或 `GO_MULTI_SEED` 时开启。预算按 §7.4(C) 的档位。**

- 参数：`α ∈ R^M`（均值固定 0）+ `log σ_⊥`，共 `M + 1` 维（**含横向宽度，更正 B4**）
- 优化器：CMA-ES ask–tell（非光滑、随机、低维、可并行）
- **Common random numbers**：同一代内所有候选共用同一组 `(network_seed, threshold_seed, noise_seed)`
- 初始化 ≥ 3 类：均匀 `α = 0`、弱随机、多次随机重启。**不从双峰初始化**
- 多保真：Stage A 短跑少 seed 淘汰；Stage B 前若干候选加长加 seed；Stage C 冻结最优参数用**未见过的
  噪声与网络种子**跑长仿真
- 每代 checkpoint：optimizer state、全部候选参数、两套 S、`coverage`、事件数、方向计数、运行时、内存、
  失败原因。支持中断续跑
- **必要对照四个**（不在首轮跑大量机制模型）：`axis_only` / `manual_projected` / `uniform_axial` / `learned`。
  `manual_hard` 作为外部参照一并报告

### 8.1 预注册结局分类（rev2 新增，更正 P0-5）

Stage 2 的"什么算成功"**在运行前冻结**。裁决同样是 fail-closed 纯函数，在 held-out 种子上评估，
且必须在 `S_grad` / `S_geom` 两套模板源下一致，否则 `SOURCE_DISAGREEMENT`。

| 结局 | 条件 | 允许的表述 |
|---|---|---|
| `RECOVERED_NONTRIVIAL_FIELD` | held-out 上同时：超过 `axis_only` **且**超过 `uniform_axial`（符号一致 ≥7/8）；对 `manual_projected` 非劣；**通过 §10.3 双方向充分性门**；跨重启场稳定；`coverage` 不低于 `manual_projected − 0.10` | "从触点秩次反演出的非平凡病理场，优于纯几何与同预算均匀走廊" |
| `UNIFORM_CORRIDOR_SUFFICIENT` | 优化结果不超过 `uniform_axial` | "在此读出分辨率下，一条同预算均匀轴向走廊已足够；数据不支持更结构化的场" |
| `MANUAL_HEURISTIC_RETAINED` | 优化场稳定低于 `manual_projected` | "手放的两端核仍是更好的工作点；反演未找到更优解" |
| `AXIS_ONLY_SUFFICIENT` | 没有任何臂稳定超过 `axis_only` | "得分可由沿轴传播的几何完全解释，病理场形状无可检出贡献" |
| `ONE_DIRECTION_ONLY` | 其余条件满足，但 §10.3 双方向门不过 | **只能说"恢复了一个传播方向"**，禁止写成"复现了两模板 repertoire" |
| `UNIDENTIFIABLE` | 跨 seed / 重启学到的场不稳定（场相关中位 < 0.5，或峰位置跨重启不一致） | "参数场在本读出下不可辨识；报告不稳定性，不展示最优一次" |
| `READOUT_INSENSITIVE` | 预注册的形状扰动（横向平移、峰数改变）在 held-out 上无法被 montage 区分 | "该 montage 分辨不出这些场差异" |

**`UNIDENTIFIABLE` 与 `READOUT_INSENSITIVE` 是合法结论，不是失败。** 按 §11 红线 14，不得只展示最优一次。

---

## 9. Stage 3 — 二维轴约束场与横向平移检验

**仅在 Stage 2 产出稳定结果后开启。**

- §4.2 的二维场，`β ∈ R^{7×3}`，同一预算投影
- 从 Stage 2 最优场映射到 `η = 0` 中心行做 warm start，**外加至少一次随机重启**
- 横向定位报告量：`r̄ = Σ_i (V_base − V_th,i) r_i / D0`；轴近质量
  `C_axis(δ) = Σ_i (V_base − V_th,i) · 1(|r_i| < δ) / D0`
- **横向平移检验**：冻结最优场形状与预算，沿 `u_C⊥` 平移若干 offset，**不重新优化**，用 held-out 种子
  重算 S，画 `S(δ)`。最大值出现在 `δ ≈ 0` 才算支持"共享轴同时约束了病理场的横向位置"

---

## 10. Held-out 与统计

### 10.1 held-out 定义（用户裁定：只做模型侧）

`held-out` = **优化过程中从未使用过的噪声种子 + 网络种子**。回答的是"有没有过拟合仿真器的随机性"。

**患者侧时间块切分不在本合同内**（D3）：现有 artifact 只有聚合 `typical_rank`，无 per-event 秩次；
重新切分需回到 masked lagPat 事件层重导模板（须走 phantom 秩次纪律），且会产生**与冻结共享平面不同的轴**，
破坏 Panel A/B 依赖的坐标合同。列为后续单独任务。

**因此本合同不得声称"在患者未见过的数据上验证"**，只能声称"在模型未见过的随机实现上验证"。

### 10.2 统计单位

本合同 n = 1 被试。**所有结果都是单被试可行性示例，不是队列证据。**
事件数与种子数只用于估计被试内稳定性，**不得当作独立样本扩大 n**。

### 10.3 held-out 双方向充分性门（rev2 新增，更正 P0-3）

**双方向不进 per-candidate 损失**（§5.3，读出太稀疏），**但必须在最终 held-out 上作为充分性门。**
在**跨 held-out 种子聚合后**的事件池上评估，四条全部满足才算通过：

1. **最低支持**：两个方向各自 `n_directional ≥ 10`（预注册；对齐 §2.4(d) 的"~10 事件后噪声收益变平"），
   且至少 `⌈2/3⌉` 的 held-out 种子各自贡献 ≥1 个双方向事件；
2. **两个模型模板彼此可区分**：`spearman(model_forward, model_reverse)` 在冻结 `SUPPORT` 上
   **显著为负**（对患者 TA/TB 自身的 −0.464 为参照，要求 ≤ −0.2），
   防止"两个方向其实是同一张图"；
3. **各自匹配到不同的患者模板**：2×2 矩阵的最优指派下，两个对角格都为正，两个非对角格都为负；
4. **交换不变**：`u_C → −u_C` 与 TA/TB 交换下裁定不变。

**不过门 ⇒ 结局只能是 `ONE_DIRECTION_ONLY`（§8.1）**，允许的表述仅为"恢复了一个传播方向"，
**禁止**写成 "data-driven field reproduces the two-template repertoire" 或任何等价说法。

---

## 11. 科学红线

1. 共享轴 `u_C` 固定，不得优化、旋转、重估、微调。
2. `u_C` 按无向轴处理；损失对 `u_C → −u_C` 与 TA/TB 交换保持不变（须有单元测试）。
3. E→E 长轴方向必须继续 = `u_C`。
4. 不改 LIF / 突触 / `m` / `z` / 噪声 / 虚拟 SEEG 的基础方程。
5. 不手工指定新场的峰数或位置。
6. 不对某一方向施加外部刺激来制造 forward/reverse；两个方向必须来自同一张网、同一噪声机制。
7. TA/TB 标签只能用于输出匹配，不得作为模型输入。
8. 不得从大量模拟事件中挑选最相似的事件（按时间先后取，不按相似度取）。
9. 所有候选场必须满足同一预算约束（§4.3）。
10. 优化只用 `S_grad`；不加 rate / baseline / 幅度 / runaway / 频谱损失。
11. 发作期数据不得参与病理场优化。
12. 不得静默降低正式模型的 `N`、连接数或仿真时长来解决 OOM。debug-small 结果必须与正式结果严格分开标注。
13. 学习到的阈值场**不得**直接解释为组织学病灶或静态抑制缺失。可说的是 effective excitability field /
    functional epileptogenic core-corridor。
14. 若不同切分或种子下学到的场不稳定，**报告不可辨识性**，不得只展示最优一次。
15. Stage 1 的全部判据与阈值在**运行前**冻结，不得按结果调整。
16. 单被试结果不得写成队列主张（B5）。

**rev2 新增（P0-1..P0-5）：**

17. **打分支撑集在优化开始前冻结**；所有候选、所有臂用同一 `SUPPORT` 与同一缺失规则。**不得**用候选
    自己招募到的子集打分（P0-2）。任何被提名的候选必须一并报告 `coverage`。
18. **任何"病理场有效"的主张必须同时超过 `axis_only` 与同预算 `uniform_axial`。绝对 Spearman 高不构成
    证据** —— 纯几何已经能拿到 0.696（P0-2）。
19. 21-seed `driven_pooled` 结果是 **readout upper-reference / endpoint sufficiency 控制**，
    **不得称为基线**，不得用于推导 headroom（P0-1）。
20. **不得使用"天花板 / ceiling / headroom"**，除非先有患者模板自身重复性（split-half / measurement
    reliability）的估计支撑（P0-1）。观测到的最大值不是上界。
21. **单方向结果只能表述为"恢复了一个传播方向"**，禁止写成"复现了两模板 repertoire"或等价说法（P0-3）。
22. `manual_projected` 与 `manual_hard` 的等价性**必须用行为判据**（§4.3.3），**不得**用 `h` 的空间相关
    代替（P0-4）。**不得声称新参数族逐位复现 legacy 抽样** —— 已核验数学上不可能。
23. 报告评分时必须**同时**说明"事件方向本身由 `sign(ax · u_C)` 定义"这一构造性事实；不得把评分结果
    表述为"证明了传播沿轴"（P0-2）。
24. Stage 1 / Stage 2 的裁决器必须是**纯函数且 fail-closed**：NaN、缺方向、模板源不一致一律返回失败出口，
    不得继续（P0-5）。

---

## 12. 明确推迟到后续 spec 的内容

各自需要独立的 spec → plan → 实现循环，**不在本合同内**：

- **recurrent-gain 版本**（用同一 `h_i` 调制已有 E→E 连接 + 突触后归一化）—— 仅在阈值场结果稳定后
- **二维完全自由场探索**（isotropic basis，事后测主轴与 `u_C` 夹角）
- **跨状态发作期验证** —— 依赖 z 慢变量把间期推进发作，而该线目前"进得去、出不来、停在不动点"（D4），
  不是现成可接的
- **队列推广** —— 前提"两个模板相反"在队列水平是少数（broad 6/26、narrow 9/26），需独立论证（B5）
- **患者侧时间块切分与模板重导** —— 见 §10.1

---

## 13. 产出物

`results/topic4_sef_hfo/data_driven_core_field/`：

```
preexisting_worktree.patch          已生成（分支创建时的未提交改动快照）
model_integrity_report.md           Stage 0
decision_log.md                     全程，保守决定的理由
config/                             冻结的超参与预注册常数
network_cache/                      (seed, theta) → 连接结构
stage1_variance_probe/
    per_run.csv  gate_verdict.json  figures/{*.pdf, README.md}
stage2_optimization/                （仅 Stage 1 通过后）
stage3_2d_field/                    （仅 Stage 2 通过后）
```

图目录必须有中文 `figures/README.md`（每图 `### filename` + 2–4 句 + 一行 `**关注点**：`），
在图实际生成后写，不得提前占位。

---

## 14. 发布检查表

- [ ] Stage 0 基线复现，版本对齐，`model_integrity_report.md` 已写
- [ ] **三个参照均用同一冻结 scorer 重算**：`axis_only` / **自发双核 `manual_hard`** / `driven_pooled`
      （后者标注为 readout upper-reference，非基线）
- [ ] `SUPPORT`、quantile seed、分级规则已冻结入 `config/` 并落盘校验和
- [ ] 网络缓存 bit-parity 测试通过
- [ ] 场模块 TDD 全绿（二分收敛 <1e-6 / `h ∈ [0,1]` / 轴翻转与 TA-TB 交换不变 / 双峰表达力 ≥0.9 作前置
      sanity）。**不含任何"逐位复现 legacy 抽样"的断言**（已核验不可能）
- [ ] Stage 1 四个仿真臂齐备，含 `manual_projected`（参数族行为等价，非 h 相关）
- [ ] Stage 1 裁决器为纯函数，8 条出口全部有单元测试（含 NaN / 缺方向 / 模板源冲突构造用例）
- [ ] Stage 1 判据在两套目标模板 × 两种缺失规则下裁定一致
- [ ] Stage 2 结局分类（§8.1）在 Stage 2 开跑前冻结
- [ ] 图目录有中文 README，图已目视检查
- [ ] 本 spec 与 plan 已 commit，分支自包含
- [ ] 主文档 / archive 未写入任何未过 Stage 1 闸门的数字
- [ ] 全文检索确认：无 "ceiling / 天花板 / headroom" 未加限定的用法（红线 20）
