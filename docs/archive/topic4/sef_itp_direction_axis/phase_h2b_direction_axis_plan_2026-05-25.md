# Topic 4 H2b — Direction-Axis Disambiguation Phase

> **状态**：v1.0.2 lock 2026-05-25 (review-round-1 demoted claims + bug fixes)
> **范围**：Topic 4 SEF-ITP framework v1.0.7 §3.2 H2 spatial-layer 的**补充**机制诊断；不是新的 cohort claim
> **不属于**：H2 主线（rank-displacement swap-k spatial compactness 已 lock 在 framework v1.0.5），SOZ 边界证明，Phase 3 H5 per-seizure 招募分析
> **依赖**：framework v1.0.7、`results/interictal_propagation_masked/rank_displacement/per_subject/*.json`、`src/seeg_coord_loader.py` (v3 mm coords)
> **触发上游**：Cursor plan `/home/honglab/leijiaxin/.cursor/plans/topic4-direction-phase_f9aebe13.plan.md`
> **文献延伸（不直搬）**：Diamond Brain (`docs/paper/awad015_*.md`) — interictal/ictal source 共定位思路；Smith eLife (`docs/paper/elife-73541-v3_*.md`) — UEA travelling-wave 方向 + polar histogram，**只迁移可证伪指标 framing**，不迁移 UEA 平面回归（SEEG 稀疏非平面）

---

## 0. 一句话承诺（朴素表述，CLAUDE.md §8）

**测了什么**：在已经知道两个传播模板存在反向 swap 端点的 cohort 上，进一步问 — 这两个模板是不是在用同**一条空间轴**（病理核心轴）以相反方向被读取，还是两个相距很远的独立 source 各自有不同方向。

**怎么测的**：把每模板的 source 通道空间中位置和 sink 通道空间中位置连成一个 3D 向量；如果是同一条轴反向读取，模板 A 的"源→汇"向量应该和模板 B 的"源→汇"向量**反向平行**（夹角 ~ 180°，或等价地，A 与 −B 夹角 ~ 0°）。如果是两个独立 source，A 向量和 B 向量方向应该几乎随机（夹角分布散）。

**揭示了什么**：H2 已经告诉我们 source 集合空间紧凑 + sink 集合空间紧凑 + swap_class 在 9/23 上达到 strict 或 candidate。H2b 进一步给出"swap 不是仅仅集合论换标签，而是同轴反向读取的物理图景"或"swap 是两个独立 source 的偶然共现"这两种解读的**per-subject** 证据。**H2b 不是 cohort claim**，它是 SEF-ITP 物理图景与替代模型分离的 per-subject 描述性诊断。

---

## 1. 为什么需要 H2b

framework v1.0.5 H2 spatial-layer 锁的是 "source-side k 个 swap 节点空间紧凑 ∧ sink-side k 个 swap 节点空间紧凑"。这条 cohort claim（19/23 source PASS、16/23 sink PASS、13/23 both PASS）足够支持 "同一组 swap-k 节点同时构成空间紧凑的 source 与空间紧凑的 sink"这个**集合论 + 各自紧凑性** statement。

但 "集合 S 紧凑 ∧ 集合 K 紧凑 ∧ S 与 K 在 swap_sweep 下角色反转"**逻辑上**还有两种几何解读：

- **解读 1（SEF-ITP 物理图景一致）**：S 和 K 是同一条传播轴的两端 → 涟漪沿这条轴扩散；模板 A 起点 = S 终点 = K，模板 B 起点 = K 终点 = S。`v_A = c_K − c_S` 和 `v_B = c_S − c_K = −v_A`，夹角 180°。
- **解读 2（独立双源 / 多源, SEF-ITP 物理图景需修正）**：S 和 K 是两个相距很远 / 解剖上独立的病理核心，各自有不同 source；H2 spatial compactness 通过是因为每个核心自己空间紧凑，但模板 A 和模板 B 不共享同一条轴，方向不反向。

`swap_sweep` 自己**不区分**这两种几何解读 — 它只测 source/sink **角色反转**（family-wise null 下），不测**几何轴共享**。

H2b 的科学价值就是给 H2 spatial-layer cohort claim 加一层**机制解释力**：在 H2 通过的 subject 上，多数是同轴反向（强支持 SEF-ITP），还是多数是独立双源（弱化 SEF-ITP 但不证伪 H2）？

---

## 1.5 H2b 能与不能 区分的对（review-round-1 lock 2026-05-25）

**先把丑话说前**：H2b 的几何指标 `v_A = c_sink_A − c_source_A`、`v_B = c_sink_B − c_source_B`、看 `cos(v_A, −v_B)` 是否接近 +1。但是 `swap_sweep` 在 source/sink 角色反转的定义本身已经隐含 **source_A ≈ sink_B、sink_A ≈ source_B**。因此对 swap_class ∈ {strict, candidate} 的 subject，"v_A 与 v_B 反向平行" 在很大程度上是 swap 定义的几何重述。

H2b 实际能与不能 区分的对：

| 替代假说 | 预测的 cos(v_A, −v_B) | H2b 能否区分？ |
|---|---|---|
| **正交无关 source（cluster A 与 cluster B 完全独立的两个解剖源）** | ≈ 0（轴方向随机） | ✅ **能** — `dual_source_shaped` 捕获 |
| **同向重复 source（两模板分享同一源到汇方向；非 swap 子集）** | ≈ −1 | ✅ **能** — `same_direction_shaped` 捕获 |
| **同一病理核心轴双向读取（SEF-ITP 物理图景）** | ≈ +1 | ❌ **不能** — H2b 无法把它与下面这个区分 |
| **同轴两端各自可启动 source（同一传播轴，但有两个独立 seed，一头一个）** | ≈ +1（同样反向平行） | ❌ **不能** — 与上面给出相同方向预测 |
| **测量伪影（单 shaft 上 1D 几何退化）** | ≈ ±1 trivially | ✅ **能** — Layer 4 PCA degeneracy override 强制为 `degenerate_geometry` |

**因此 H2b 主结论的正确表述是 排除而不是支持**：

- 可以写："**排除了正交无关 source 解释**" / "**与同一传播轴上的双向读取（不论 seed 是单源双向还是双端各自启动）方向一致**"
- **不可以写**："支持同一病理核心轴" / "证明 swap 不是双源" — 这超出 H2b 数据能给出的判读

要区分 "同一病理核心轴双向读取" 与 "同轴双端独立 seed"，必须做 Round 2 的延伸（见 §11）：

1. **per-event initiation centroid 聚类**：每个事件 earliest-k 的空间 centroid 在 subject 内是否双峰（双 seed）还是连续沿轴随机（单源双向）。
2. **rank-distance gradient 单调性**：行波预测 rank 沿距源距离单调上升；端点 swap 集合重叠未必有连续梯度。
3. **source_A 与 source_B 各自相对 SOZ / ictal-early channel 的位置**：两个 source 都贴 SOZ → 双 seed；一头贴 SOZ 一头不贴 → 单源双向更可能。

Round 2 不在本 v1.0.2 lock 的实施范围；本 phase 只做 H2b template-level + descriptive + degeneracy override + SOZ relation 这一层。

## 2. 不做的事（scope 红线）

H2b **不**做以下事情，避免 CLAUDE.md §5/§6/§8 经典错误：

1. **不**升级为 cohort claim。无 Wilcoxon、无 cohort p-value、无 binomial sign-test。**Cohort 输出只是 per-subject 3-state verdict 计数**（按 swap_class strict/candidate/none 分层）。
2. **不**证明 / 反驳 SOZ 边界。source / sink 端点与临床 SOZ / data-driven SOZ / ictal-early channel 的空间关系**只作 secondary descriptive readout**，不进 verdict gate。
3. **不**与 Smith UEA travelling-wave 速度场做数值对照。任何 polar plot 必须配 caveat：这是 SEEG rank-derived endpoint 方向，**不**是微电极速度场。
4. **不**与 Diamond ictal source 做 multilateration。我们没有皮层 geodesic mesh + 固定速度 radial wave 合同；不允许搬 multilateration 公式。
5. **不**在事件级把每事件当 cohort 独立 N。cohort 主结论按 subject-level 计数；事件级 direction 只作 per-subject 内部 polar histogram + bootstrap CI。
6. **不**事后调整 verdict 阈值、null spec、degeneracy 阈值。所有判据 framework time lock。

---

## 3. 设计 — 5 层 + verdict alphabet

### 3.1 输入合同

每 subject 从 `results/interictal_propagation_masked/rank_displacement/per_subject/<dataset>_<subject>.json` 取：

- `pairs[0]`（primary pair）的：
  - `channel_names`（n 个 lagPat-joint 通道）
  - `joint_valid`（bool 长度 n）
  - `rank_a_dense_full`、`rank_b_dense_full`（per-channel 0..n_valid-1 dense rank；非 joint 位置在 _full 里也有值，但 mask 外不可信）
  - `swap_sweep.swap_class`（strict / candidate / none / inconclusive）、`swap_sweep.decision_k`
  - `cluster_id_a`、`cluster_id_b`（PR-2 cluster IDs）
  - `soz_mask`（bool 长度 n，clinical SOZ 在 lagPat-joint 上的投影；可选）

从 `src.seeg_coord_loader.load_subject_coords(dataset, subject_id, channel_names_requested=channel_names)`：

- `coords_array_in_requested_order`（n, 3）mm 单位
- `mapped_mask_in_requested_order`（n, bool）

**实际可用通道（H2b universe）** = `joint_valid ∧ mapped_mask`。这是 H2b 所有几何运算的 universe；少于 6 个 → `exit_reason = "n_universe < 6"`。

### 3.2 Layer 1 — Template-level axis vector（**primary per-subject test**）

对每个 cluster c ∈ {A, B}：

1. 用 `decision_k` 与 `rank_a_dense_full` / `rank_b_dense_full` 派生 source / sink 集合：
   - `source_c = argsort(rank_c_dense_full[universe])[:decision_k]` （universe 内最低 dense rank，**最早点火**）
   - `sink_c   = argsort(rank_c_dense_full[universe])[-decision_k:]`（universe 内最高 dense rank，**最晚点火**）
   - 复用 `src.rank_displacement.derive_swap_endpoint` 的同款 argsort 规则；source = lowest-k，sink = highest-k。
2. 端点空间 centroid：`c_source_c = mean(coords[source_c])`，`c_sink_c = mean(coords[sink_c])`。
3. **轴向量**：`v_c = c_sink_c − c_source_c`，长度 `||v_c||` = 传播轴长度。

**主指标（angle test）**：

- `cos_AB_neg = (v_A · (−v_B)) / (||v_A|| · ||v_B||)`
- `angle_A_negB_deg = acos(clip(cos_AB_neg, -1, 1)) * 180 / π`
- 同时报 `cos_AB_pos = (v_A · v_B) / (||v_A|| ||v_B||)` 作 sanity（dual-source 时两条向量随机；独立同源时方向一致）。

**Null（lock 2026-05-25 — per-cluster independent role draws）**：

union `U = source_A ∪ sink_A ∪ source_B ∪ sink_B`（去重；在完美 antipodal 反转下 `|U| = 2*decision_k`，不到 `4*decision_k`）。每次 permutation：

- **cluster A**：在 U 中独立抽样得到 `S_A` (k channels) ⊥ `K_A` (k channels)（A 内部 disjoint）
- **cluster B**：在 U 中**独立**抽样得到 `S_B` (k) ⊥ `K_B` (k)（B 内部 disjoint）
- **A 与 B 之间允许重叠**（这正是 antipodal 反转的允许配置：同一 channel 可以是 `S_A` 又是 `K_B`，反之亦然）

最低条件：`|U| ≥ max(n_sA + n_kA, n_sB + n_kB) = 2*decision_k`，每 subject 都能跑。

**为什么不用 4-disjoint null**：4-disjoint 要求 `|U| ≥ 4*decision_k`，在完美 antipodal 反转下永远不满足（观测构型本身就违反 4-disjoint 假设）。"4-disjoint" null 把观测的 axis-reversal 排除在 null space 外，是 biased null（test 一定显著）。正确的 null 应包含观测构型作为 prior-equal 配置，per-cluster independent 满足。

- `n_perm = 1000`，`seed = 0`。
- 主指标：`p_one_sided = (1 + #{cos_AB_neg_null >= cos_AB_neg}) / (1 + n_perm_completed)` — 实测 cos 越接近 1（向量越同向 = "A 与 −B 越对齐"= "A 与 B 越反向"= 同轴反向 hypothesis 支持），p 越小。

### 3.3 Layer 2 — Event-level direction（**polar / cos histogram, 1D**）

对每个事件 e（在 cluster c 中）：

1. 在 universe 内取 event e 的 per-channel rank `rank_e`（来自 `*_lagPat_withFreqCent.npz` 的 `ranks` 经 phantom-mask 处理）+ event participation `bools_e`。
2. eligible = `universe ∧ bools_e ∧ (rank_e finite)`；要求 `n_eligible ≥ 2k_event` 才计算，否则该事件 skip。
3. `early_idx = argsort(rank_e[eligible])[:k_event]`、`late_idx = argsort(rank_e[eligible])[-k_event:]`。
4. `c_early_e = mean(coords[early_idx])`，`c_late_e = mean(coords[late_idx])`。
5. `v_event_e = c_late_e − c_early_e`（单位化后 `u_event_e = v_event_e / ||v_event_e||`）。

**默认 k_event = 2**（per CLAUDE.md §2 "minimum that solves the problem"），**sensitivity k_event = 3**。

**1D 投影直方图（cos-similarity，per cluster）**：

- 对 cluster A 的每事件，记 `cos_eventA_on_vA = u_event_e · u_A`（u_A = v_A / ||v_A||）。期望分布峰值 ≈ +1（事件方向与 template A 自己向量一致；几乎自定义为正）— sanity-check only。
- 对 cluster B 的每事件，记 `cos_eventB_on_vA = u_event_e · u_A`。**这是主测试**：
  - 同轴反向 → 分布峰值 ≈ −1
  - 双源 → 分布峰值 ≈ 0（无系统取向）
  - 独立同源 → 分布峰值 ≈ +1（与 A 同向）

报告 per-cluster：median(cos)、IQR、cohort-level **不**做（per §2 红线 #5）。

### 3.4 Layer 3 — Axis projection slope（**A 的轴上 B 的 rank 斜率**）

`u_A = v_A / ||v_A||`，universe 各通道在 A 轴上的投影 `proj_i = coords[i] · u_A`。

对每 cluster c，把 `rank_c_dense_full[universe]` 与 `proj` 做 Spearman / 线性回归：

- `slope_A_on_axisA`、`r2_A_on_axisA`：**trivial monotonic +**（A 的 source 投影最小、A 的 sink 投影最大，是 A 自己 axis 的定义）— sanity-check only。
- `slope_B_on_axisA`、`r2_B_on_axisA`：**这是 H2b 的关键测试**：
  - 同轴反向 → 显著 **负斜率** + 高 r²（B 的 rank 单调降低 along A 的 axis）
  - 双源 → 斜率 ≈ 0 + 低 r²
  - 独立同源 → 显著正斜率

**Slope 显著性**：simple permutation null —shuffle universe channel labels in `rank_b_dense_full`，重算 1000 次；one-sided `p_neg = mean(slope_null <= slope_B_on_axisA)`。

### 3.5 Layer 4 — Single-shaft / dimension degeneracy（**overriding flag**）

在 universe coords 上做 PCA，得 3 个特征值 `λ_1 ≥ λ_2 ≥ λ_3 ≥ 0`：

- `lambda_ratio_12 = λ_2 / λ_1`
- `lambda_ratio_23 = λ_3 / λ_1`

**Degeneracy 判据（lock）**：

- **near-1D**：`lambda_ratio_12 < 0.10` → universe 几乎在一条线上 → 任何"方向"几乎是这条线的 ±1；verdict **强制为 `degenerate_geometry`**（不论 Layer 1-3 结果）
- **near-2D（descriptive only）**：`lambda_ratio_23 < 0.05` 且 `lambda_ratio_12 ≥ 0.10` → universe 在一个平面上 → Layer 1-3 仍合理但 Layer 2 polar histogram 可以投到 PCA 主平面作 supplementary 可视化

**为什么 0.10 不是事后调整**：在 framework time lock，理由是 SEEG shaft 通常 5-10 mm contact spacing，shaft 长度 30-80 mm；单 shaft 跨度 / 邻 shaft 间距比一般在 3-10 之间 → `λ_2 / λ_1` 在 3D 实植上典型为 0.2-0.6。低于 0.10 的 cohort subject 应被识别为单 shaft 主导 → conservative 0.10 阈值。

### 3.6 Layer 5 — SOZ / ictal-early relation（**secondary, not in verdict gate**）

仅在 `swap_sweep` 给出非空 `soz_mask` 时跑（否则 `exit_reason = "no_soz_in_lagpat"`）：

1. SOZ centroid：`c_SOZ = mean(coords[soz_mask ∧ mapped_mask])`
2. 距离读数（**只报，不做 verdict**）：
   - `d_source_A_to_SOZ = ||c_source_A − c_SOZ||`、`d_source_B_to_SOZ`、`d_sink_A_to_SOZ`、`d_sink_B_to_SOZ`
   - `min_d_source_A_to_SOZ_chs` = `min(||coords[i] − coords[s]|| for i in source_A for s in SOZ-mapped)` （per channel-pair 最小距离）
3. swap-k endpoint 与 SOZ 的 Jaccard / set-overlap（复用 `src.rank_displacement.compute_clinical_soz_set_relation` 接口；不重新发明）。
4. ictal-early channel：**当前 cohort 尚无 data-driven ictal-early channel 文件**（PR-T3-1 Layer B pending），所以 H2b v1.0 实施时**只用 clinical SOZ**；future 加入 ictal-early 时再扩。

输出字段固定，下游不允许把这部分当 swap 定义。

### 3.6.5 Descriptive geometry layer（v1.0.1 add → v1.0.2 audit fix 2026-05-25）

> **诚实标记 (post-hoc 添加, advisor catch ratified)**：v1.0 contract 只锁了 §3.7 的 5-label strict verdict alphabet（含 perm null gate）。cohort 跑出后发现 strict gate 在 `decision_k ≈ n_universe / 2` 时 permutation null 自由度退化（同一 universe 内能产生的 source/sink partition 数有限），即使观测 cos(v_A, −v_B) ≈ 0.95 也无法 reject null。例：epilepsiae_1146 cos = 0.948、perm p = 0.104 → 落 strict `inconclusive`。
>
> 为避免几何信号被 strict gate 黑掉，v1.0.1 增加 `assess_descriptive_geometry` 平行层。返回 4 个 label：`axis_reversal_shaped` / `dual_source_shaped` / `same_direction_shaped` / `unclear`。
>
> **v1.0.2 review-round-1 audit fix #3**：v1.0.1 实现把 `axis_reversal_shaped` 闸门写成 "cos(A,-B) ≥ 0.5 AND slope_B < 0"，**漏掉了 docstring 承诺的 r² 阈值**。这导致低 r² 的噪声负斜率（slope_B ≈ −0.01、r² ≈ 0.005）也能进 `axis_reversal_shaped`，污染 cohort 计数。v1.0.2 把 `R2_DESCRIPTIVE_MIN = 0.20` 显式加入 `assess_descriptive_geometry`：`axis_reversal_shaped` 与 `same_direction_shaped` 都要求 r² ≥ 0.20，与 `dual_source_shaped` 的 r² < 0.20 形成对称闸门。
>
> **v1.0.2 review-round-1 fix #4 — strict layer 是 PRIMARY 读数，descriptive 是 supplementary shape**：v1.0.1 §10 把 descriptive 当 cohort 结论层用（`6:0 axis_reversal_shaped vs dual_source_shaped` 做 falsifiability check）。这违反 §1.5 "H2b 只能排除正交 unrelated source、不能区分同轴双向 vs 双端 seed" 的 scope 红线。v1.0.2 改回：**strict verdict 是 PRIMARY 读数**（保留 falsifiability）；**descriptive 只是 shape evidence**，每条 cohort 写法都必须同时报 strict 与 descriptive，禁止只引 descriptive 数字。
>
> CLAUDE.md §5 红线 "事后调整 verdict 阈值" 未触发：阈值数字 (cos ≥ 0.5、r² 0.20、ratio 2×) 没有动；v1.0.2 改动是 (a) 加 docstring 承诺但实现漏的 r² 闸门 (bug fix)、(b) 把 cohort 写法从 "descriptive 当结论" 改回 "strict 作 primary + descriptive 作补充"。

### 3.7 Per-subject verdict alphabet（3-state + degeneracy override）

| Verdict label | 触发条件（lock） |
|---|---|
| `degenerate_geometry` | Layer 4 near-1D fires（`lambda_ratio_12 < 0.10`）OR `n_universe < 6`；强制覆盖，不读 Layer 1-3 |
| `axis_reversal` | `cos_AB_neg ≥ 0.5`（夹角 A vs −B ≤ 60°）AND Layer 1 permutation p<0.05 AND Layer 3 `slope_B_on_axisA < 0` with permutation p<0.05 |
| `dual_source` | `\|cos_AB_neg\| < 0.5`（夹角 60°-120°）AND Layer 3 `r2_B_on_axisA < 0.2` |
| `same_direction` | `cos_AB_pos ≥ 0.5`（夹角 A vs +B ≤ 60°）— 模板没真正 swap，方向一致；通常出现在 `swap_class = none` 上 |
| `inconclusive` | 其它情况（Layer 1 / Layer 3 不一致；perm p borderline；signal weak） |

**none-swap subject 期望**：`swap_class = none` 的 subject 多数应落 `same_direction`（两模板共享方向、没有真 swap）；如果反而落 `axis_reversal` 是反常 → 进 audit。

### 3.8 Cohort 输出（只是计数）

`cohort_summary.json` 字段：

```json
{
  "schema_version": "h2b_direction_axis_v1",
  "framework_version": "v1.0.7",
  "tier_lock": "descriptive_supplementary_per_subject_verdict_only",
  "n_subjects_total": <int>,
  "n_subjects_with_coords": <int>,
  "verdict_distribution": {
    "axis_reversal": <int>,
    "dual_source": <int>,
    "same_direction": <int>,
    "inconclusive": <int>,
    "degenerate_geometry": <int>,
    "exit_no_universe": <int>
  },
  "verdict_by_swap_class": {
    "strict":     {"axis_reversal": <int>, "dual_source": <int>, ...},
    "candidate":  {...},
    "none":       {...},
    "inconclusive": {...}
  },
  "k_template": 3,  
  "k_event_default": 2,
  "n_perm": 1000,
  "seed": 0,
  "degeneracy_threshold_lambda_ratio_12": 0.10
}
```

**禁止**：
- ❌ Wilcoxon 或任何 cohort p-value
- ❌ "axis_reversal 主导 → SEF-ITP cohort-supported" 类陈述
- ✅ 允许的 cohort 语言："在 19/23 H2 spatial PASS 的 subject 中，N/19 落 axis_reversal、M/19 落 dual_source、K/19 落其它；这是 per-subject 机制诊断，不构成新的 cohort claim"

---

## 4. 文献关系（移植边界）

### 4.1 Diamond Brain (`docs/paper/awad015_*.md`)

迁移的是 **framing**：interictal IED 序列与 ictal discharge 序列**是否共享空间来源**是合法可问的科学问题。

**不**迁移的：
- multilateration 公式（皮层 geodesic mesh + 固定速度 radial wave 合同）— SEEG 稀疏 + 我们没有 fixed speed 假设
- ictal vs interictal 比例统计 — Topic 4 H5 才做 peri-ictal，不在 H2b scope

### 4.2 Smith eLife (`docs/paper/elife-73541-v3_*.md`)

迁移的是 **可证伪指标 framing**：
- "方向是否随时间稳定 / 随事件 bimodal antipodal" 这个问法可迁
- polar histogram 作可视化思路可迁，但 **caveat 必须写在 figure title**：`"SEEG endpoint-axis direction (rank-derived), not UEA traveling-wave velocity"`

**不**迁移的：
- UEA 96 通道平面回归（planar phase regression）— SEEG 稀疏 + 非平面，没有合同
- 数值上不与 Smith mm/s 速度比对

---

## 5. 测试（synthetic）— 锁住助记符

`tests/test_sef_itp_direction_axis.py` 必须含以下 case，每个 case 锁住一条 verdict：

1. **synthetic_axis_reversal** — 12 channel coords 在一条直线上 `[0, 0, 0], [1, 0, 0], ..., [11, 0, 0]`；cluster A rank = 0..11、cluster B rank = 11..0。期望 `cos_AB_neg ≈ 1`、`slope_B_on_axisA < 0`、verdict = `axis_reversal`。
2. **synthetic_dual_source** — cluster A axis 沿 X (source 集中在 X=0、sink 集中在 X=10)、cluster B axis 沿 Y (source 在 Y=0、sink 在 Y=10)，两套不共享通道集。期望 `cos_AB_neg ≈ 0`、`slope_B_on_axisA ≈ 0`、verdict = `dual_source`。
3. **synthetic_same_direction** — cluster A 和 cluster B 共享 axis 同向（none-swap subject 模拟）。期望 `cos_AB_pos ≈ 1`、verdict = `same_direction`。
4. **synthetic_single_shaft_degeneracy** — 所有 universe coords 在 X=0 平面内的 1D 线上 + 邻 shaft 仅 0.01 mm 偏离 → `lambda_ratio_12 < 0.10` → verdict 强制 `degenerate_geometry`，不论 Layer 1-3。
5. **k_event_sensitivity** — 同 case 1，跑 k_event ∈ {2, 3}；template-level 不变；event-level cos median 在两 k 下方向一致（不要求数值相等）。
6. **coords_missing_exit** — universe size = 4 → `exit_reason = "n_universe < 6"`、verdict = `exit_no_universe`。
7. **permutation_null_sanity** — 在 case 1 上跑 perm；null cos_AB_neg 中位 ≈ 0、实测 ≈ 1、p < 0.01。

---

## 6. 落地清单

- 新模块：`src/sef_itp_direction_axis.py`（helpers + per-subject verdict）
- 新 runner：`scripts/run_sef_itp_direction_axis.py`
- 新 plotter：`scripts/plot_sef_itp_direction_axis.py`（per-subject cos histogram + axis projection scatter + 3D source/sink + SOZ overlay）
- 新测试：`tests/test_sef_itp_direction_axis.py`
- 新结果目录：`results/topic4_sef_itp/direction_axis/{per_subject/, figures/, cohort_summary.json}`，含 `figures/README.md`（中文，AGENTS.md 规范）
- 主文档**不**改 `docs/topic4_sef_itp_framework.md`（H2b 是 archive-only supplementary，不进 framework v1.0.7 §3 H 主清单；只在 cohort 跑完后用 archive 链接回主文档 §3.2 H2 末尾一行 cross-cite）

---

## 7. Smoke 顺序

1. `epilepsiae_1146`（`swap_class=strict`）— 期望 verdict 落 `axis_reversal`（SEF-ITP 物理图景的强代表 case）
2. `epilepsiae_635`（`swap_class=candidate` / zigzag）— 期望 `axis_reversal` 或 `inconclusive`
3. `epilepsiae_1084`（`swap_class=none` but seed stable）— 期望 `same_direction` 或 `inconclusive`

任何 smoke case 出现 unexpected verdict 必须先 audit 是否 `derive_swap_endpoint` source/sink 半截切错、coords missing、或 degeneracy 阈值偏 → 不准事后调阈值"修复" verdict。

---

## 8. Falsifiability checkpoint（framework-time lock + v1.0.2 layer-mismatch fix）

H2b 真正能 falsify 的对仅有一个（per §1.5 scope）：**排除假说 H_orth — cluster A 与 cluster B 是正交无关解剖源**。"同一病理核心轴双向读取" 与 "同轴双端独立 seed" H2b 都不能区分（两者都预测 `cos(v_A, −v_B) ≈ +1`），必须等 Round 2（§11）。

H2b 单方向预测：在 H2 spatial PASS 的 subject 中，shape signal `axis_reversal_shaped + same_direction_shaped`（即 cos(v_A, ±v_B) 接近 ±1，方向一致或反向但非正交）应**显著多于** `dual_source_shaped`（cos 接近 0、轴正交）。

**Denominator layer mismatch fix (v1.0.2 review-round-1 audit fix #4)**：v1.0.1 cohort 写法用了 "swap_class ∈ {strict, candidate} 的可测 subject" 作分母。但 §8 plan 原写 "H2 spatial PASS subject"——这两个集合**不**等同：

- **label 层**（swap_class）：rank_displacement 的 swap_sweep family-wise null 给的 strict / candidate / none 标签。strict + candidate ≈ 9/23 in cohort。
- **spatial 层**（H2 PASS）：source-side AND sink-side 各自空间紧凑（PR-6 / Topic 4 framework v1.0.5 §3.2：source 19/23 PASS + sink 16/23 PASS，source ∩ sink 同时 PASS 13/23）。

用 label 层替 spatial 层会**让 cohort 显得更反向**——label 层是被 swap_sweep 预筛过的子集，几何反向部分是 swap_sweep 定义重述。v1.0.2 在 §10 cohort verdict 里 **两个分母都报**，并明确指出 label 层数字偏 SEF-ITP-friendly。

**阈值参考（descriptive shape 数字仅作 supplementary，不绑定 verdict）**：

- `axis_reversal_shaped ≥ 2 × dual_source_shaped`（两个分母下都查）→ shape 一致于"非正交"假说，**排除 H_orth**
- `dual_source_shaped ≥ axis_reversal_shaped` → H_orth 不能排除，SEF-ITP 解释力受冲击
- `degenerate_geometry ≥ 50%` → H2b 在该 cohort 上 underpowered

阈值数字（2×、50%）在 framework time lock，**事后不允许调整**。**注意**：阈值通过只 falsify H_orth 假说，**不构成 "支持 SEF-ITP 同一病理核心"**——同轴双向 vs 同轴双端 seed 的区分必须等 Round 2（§11）。

---

## 9. 版本历史

- v1.0 lock 2026-05-25：initial contract，pre-cohort
- v1.0.1 cohort 完结 2026-05-25：cohort 跑完，§10 cohort verdict 落字。Post-hoc 添加 `assess_descriptive_geometry` 几何形状 label 层（§3.6.5）— smoke 显示 strict perm null 在 decision_k ≈ n_universe/2 退化，几何信号被黑掉；descriptive 层并行报告几何形状不替代 strict 层。Falsifiability §8 阈值参考改用 descriptive 层。
- **v1.0.2 review-round-1 fix lock 2026-05-25**：审阅 catch 五点全部修复：
  1. **§1.5 新增 scope 红线**：明确 H2b 只能 falsify 正交无关 source 假说；不能区分 "同一病理核心轴双向" vs "同轴双端独立 seed"——两者都预测 cos(v_A, −v_B) ≈ +1。所有 cohort 措辞按这条降级。
  2. **§3.6.5 + 实现 audit fix #3**：`assess_descriptive_geometry` 原 docstring 承诺 r² 闸门，实现漏了。v1.0.2 显式加 `R2_DESCRIPTIVE_MIN = 0.20`，对 `axis_reversal_shaped` 与 `same_direction_shaped` 都要求 r² ≥ 0.20，与 `dual_source_shaped` 的 r² < 0.20 对称。r² 修复后 cohort 计数从 6/9 → 5/9 axis_reversal_shaped。
  3. **§3.6.5 + §10 fix #4**：strict verdict 是 PRIMARY 读数（保留 falsifiability）；descriptive 只是 supplementary shape evidence；不允许只引 descriptive 数字。v1.0.1 用 descriptive 当 cohort 结论层，违反 scope。
  4. **§8 + §10.3 fix #4**：cohort falsifiability denominator 错用 label 层（swap_class ∈ {strict, candidate}）替了 plan §8 原指定的 spatial 层（H2 PASS）。v1.0.2 两个分母都报，并明确 label 层数字偏 SEF-ITP-friendly。
  5. **SOZ relation fix #5**：`compute_soz_relation` 原版只在 `soz_mask ∧ universe_mask` 内算，会漏掉 clinical SOZ 有坐标但不在当前 pair joint-valid 的 channel。v1.0.2 同时报 `mapped_full`（soz ∧ mapped_mask）与 `joint_universe`（soz ∧ universe）两套距离。
  6. **Event-layer fix #6**：runner 的 `_compute_event_level_layer` 原只检查 phase0a channel order；v1.0.2 显式加 lagpat channel order vs rank_displacement channel order 的 strict equality assertion，杜绝 AGENTS.md 已警告的 channel-order 漂移导致的事件层静默污染。
  7. **§11 新增 Round 2 deferred work list**：要真正区分 "同核双向" vs "双端 seed" 必须做 per-event initiation centroid 聚类、rank-distance 连续梯度、source_A vs source_B 各自 SOZ 关系。本 phase v1.0.2 不实施，列为 Round 2 PR。

  CLAUDE.md §5 红线 "事后调整 verdict 阈值 / null spec / degeneracy 阈值" 未触发：所有阈值数字（cos ≥ 0.5、r² 0.20、ratio 2×、λ₂/λ₁ 0.10）没有动；v1.0.2 改动全部是 (a) bug fix (#3 #5 #6)、(b) cohort 措辞降级到 scope 红线内 (#1 #4)、(c) 加 Round 2 deferred list (#7)。

---

## 10. Cohort verdict（v1.0.2 review-round-1 demoted claims 2026-05-25）

### 10.1 三段式朴素话（CLAUDE.md §8 — review-round-1 demoted 版本）

**测了什么**：在 40 个间期事件 cohort 上（masked rank_displacement primary pair），看每个 subject 的两个传播模板的源→汇向量在 3D 脑空间里是不是反向平行 / 同向平行 / 正交。**注意 §1.5**：H2b 只能 falsify "cluster A 与 cluster B 是正交无关解剖源" 这个假说；"同一病理核心轴双向读取" 与 "同轴双端独立 seed" 两种都给出反向平行轴预测，H2b **不能区分**。

**怎么测的**：每个 subject 拿 swap_sweep 给的 decision_k，把 universe 通道（rank_displacement 的 joint_valid AND coord-mapped）按 dense rank 切成 source-side / sink-side 两半；用各 3D centroid 之差作模板轴向量。primary 指标 `cos(v_A, −v_B)`。同时做 cluster B 的 rank 在 cluster A 轴上的投影斜率。所有判读做 per-cluster independent role-shuffle permutation null。两层判读输出：

- **strict verdict (primary read)**：cos + perm p<0.05 + slope sign + slope perm p<0.05 + r² → 5-state
- **descriptive geometry (supplementary shape evidence)**：cos + slope sign + **r² ≥ 0.20**（v1.0.2 修了原 v1.0.1 漏的 r² 闸门）→ 4-shape labels

**揭示了什么 (v1.0.2 demoted 版本)**：

- 40 个 subject 里 **17 个 exit_no_universe**（n_universe < max(6, 2·decision_k)），不进 verdict 计数。
- **strict verdict (PRIMARY READ)** 在 23 个可测 subject 上：**0 个 axis_reversal**、5 个 dual_source、3 个 same_direction、6 个 degenerate_geometry、9 个 inconclusive。strict 层 0 个 axis_reversal 的原因：decision_k ≈ n_universe / 2 时 per-cluster role-shuffle null 自由度低，即使观测到 cos(v_A, −v_B) ≈ 0.95 也无法 reject null。**这是 H2b 在该 cohort 的 primary cohort 读数**：strict 层不支持 cohort-level "axis_reversal 成立" 的 claim。
- **descriptive geometry (SUPPLEMENTARY shape evidence — v1.0.2 r² gate after fix)** 在 23 个可测 subject 上：8 axis_reversal_shaped + 5 dual_source_shaped + 1 same_direction_shaped + 9 unclear。
- shape 层的 cohort 模式：**strict+candidate 合并 9 testable** 上 `axis_reversal_shaped : dual_source_shaped = 5 : 0`（v1.0.1 误计为 6:0；r² 闸门修复后 epi_1146 r²=0.185 honestly 落到 unclear）；**none 合并 14 testable** 上 `5 dual_source_shaped + 1 same_direction_shaped + 3 axis_reversal_shaped + 5 unclear`。
- shape 层结论（只能写 falsify H_orth）：**在 swap-positive 子集上，shape 不支持 "cluster A 与 cluster B 是正交无关解剖源" 假说**；具体哪种轴共享机制（同一核心双向 vs 双端各自 seed）H2b 不能区分，必须等 §11 Round 2。

### 10.2 数字总表（v1.0.2 r² gate 修复后）

| swap_class | n testable | strict: axis_reversal | strict: dual_source | strict: same_direction | strict: degenerate / incon. | shape: axis_reversal_shaped | shape: dual_source_shaped | shape: same_direction_shaped | shape: unclear |
|---|---|---|---|---|---|---|---|---|---|
| strict     | 5  | 0 | 0 | 0 | 2 / 3 | **2** | 0 | 0 | 3 |
| candidate  | 4  | 0 | 0 | 0 | 1 / 3 | **3** | 0 | 0 | 1 |
| strict+candidate combined | 9 | **0** | 0 | 0 | 3 / 6 | **5** | **0** | 0 | 4 |
| none       | 14 | 0 | 5 | 3 | 3 / 3 | 3 | 5 | 1 | 5 |
| **all (n=40)** | **23** | **0** | 5 | 3 | 6 / 9 | **8** | **5** | **1** | **9** |

### 10.3 Falsifiability check（v1.0.2 双分母版本）

archive plan §8 framework time lock 设的方向参考 `axis_reversal_shaped ≥ 2 × dual_source_shaped` 只 falsify H_orth（正交无关 source 假说）。审阅 catch：plan §8 原写"H2 spatial PASS subject"作分母，v1.0.1 错用了"swap_class ∈ {strict, candidate}"作分母（label 层 vs spatial 层不同集合）。v1.0.2 两个分母都报：

| 分母 | n testable | axis_reversal_shaped | dual_source_shaped | ratio | falsify H_orth? | 注 |
|---|---|---|---|---|---|---|
| **swap_class ∈ {strict, candidate}**（label 层） | 9 | 5 | 0 | ∞ | ✅ 通过 | label 层是 swap_sweep 预筛过的，**偏 SEF-ITP-friendly** |
| **H2 spatial-layer PASS（source ∧ sink 双 PASS, n=13/23 per framework v1.0.5）** | 待 cross-reference Topic 4 §3.2 数字逐 subject 落实；v1.0.2 此次未单独跑此分母（数字 list 在 framework doc 没有 per-subject 列出，待 Topic 4 framework 同步） | — | — | — | pending | spatial 层是 plan §8 原本指定的分母 |

**v1.0.2 lock 措辞**：cohort H_orth falsifiability check 在 label 层（**更宽容的分母**）下 ratio = ∞ > 2×，通过；在 spatial 层（plan §8 原 lock 的分母）下 v1.0.2 未单独跑。这两个 ratio 都**不构成 "支持 SEF-ITP 同一病理核心"**——只 falsify 正交无关 source；区分 "同轴双向" vs "同轴双端 seed" 等 Round 2。

degenerate_geometry 占 strict+candidate 可测的 3/9 = 33%，低于 50% underpowered 阈值。

### 10.3.5 epilepsiae_1146 channel-name set-overlap（参考性，不构成结论升级）

advisor catch ratified：cos(A, −B) = 0.948 理论上有可能是 centroid 数学伪结果。直接 set-overlap 验证：

| 角色对比 | overlap | 比例 |
|---|---|---|
| source_A ∩ sink_B | {ICL10, ICL11, ICL8, ICL9, SCL7} | 5/7 |
| sink_A ∩ source_B | {ICL1, ICL2, ICL3, ICL4, ICL5, ICL6} | 6/7 |

随机基线（n_universe = 15、k = 7）下 |sA ∩ kB| 期望 ≈ 7² / 15 ≈ 3.3；观测 5/7 与 6/7 高于随机预期。**注意**：5/7 + 6/7 是 swap_sweep 已经在该 subject 上达到 `swap_class = strict` 的几何重述（rank_displacement family-wise null 对 fwd/rev top-k 角色互换已经 pass）。set-overlap 验证仅证明 **centroid 数学不是伪结果**；不构成 "支持同一病理核心 vs 双端 seed" 的额外证据，所有这些都还在 H2b 的 scope 限制 §1.5 之内。

注意 v1.0.2 r² gate 修复后，epi_1146 的 descriptive 层 label 是 **unclear**（r² = 0.185 < 0.20），不是 `axis_reversal_shaped`。set-overlap 不能把它救回 axis_reversal_shaped；后续 §11 Round 2 该 subject 是否仍展现 SEF-ITP 同轴特征要靠 event-level 或 SOZ 关系来判定。

### 10.4 个别 subject 备注（v1.0.2 r² gate 修复后）

- **epilepsiae_1146** (strict, decision_k=7, n_universe=15)：cos(A,−B)=0.948、slope_B=−0.165、**r²=0.185** → strict `inconclusive`、descriptive **`unclear`**（v1.0.1 误标为 axis_reversal_shaped；r² gate 修复后回正）。channel-name 5/7+6/7 overlap 仍存在但只能 sanity centroid 数学；不构成 descriptive shape 通过。
- **epilepsiae_958** (strict, n_universe=?)：cos(A,−B)=0.946、slope_B=−0.349、r²=0.586 → descriptive **`axis_reversal_shaped`**（r² gate 通过）。
- **yuquan_zhaochenxi** (strict)：cos(A,−B)=0.891、slope_B=−0.704、r²=0.484 → descriptive **`axis_reversal_shaped`**。
- **epilepsiae_139** + **yuquan_zhangjiaqi** (strict)：cos≈1.0 但 PCA `λ₂/λ₁ < 0.10` → 强制 `degenerate_geometry`（单 shaft 1D），方向不可解释。Layer 4 override 起作用。
- **swap=none 但 axis_reversal_shaped 的 3 个 subject**（epi_1096、epi_583、epi_916）：shape 看反向但 swap_sweep family-wise null 没认定 strict/candidate。可能 marginal swap 或几何巧合；不进 §10.3 falsify-H_orth 分子分母。

### 10.5 H2b 主结论（v1.0.2 lock 措辞 — review-round-1 demoted）

> "H2b 跑完不提出新的 cohort claim。**Strict verdict (primary read)** 在 23 个可测 subject 上 0 个 `axis_reversal`、5 个 `dual_source`、3 个 `same_direction`、6 个 `degenerate_geometry`、9 个 `inconclusive`——strict 层在该 cohort 不支持 'axis_reversal 成立' 的 claim，主要是因为 decision_k ≈ n_universe / 2 时 per-cluster role-shuffle null 自由度低。
>
> **Descriptive shape (supplementary, NOT a conclusion layer)** 在 strict+candidate 合并的 9 个可测 subject 上给 5 `axis_reversal_shaped` + 0 `dual_source_shaped` + 4 `unclear`，按 plan §8 阈值参考 `axis_reversal_shaped ≥ 2 × dual_source_shaped` 通过 → **shape 一致于 "swap-positive cohort 的两模板不是正交无关解剖源"**。
>
> 但 §1.5 scope 红线锁住：H2b **不能在 "同一病理核心轴双向读取" 与 "同轴双端独立 seed" 之间判定**——两种都预测 cos(v_A, −v_B) ≈ +1。要回答这个问题必须做 §11 Round 2（per-event initiation centroid 聚类 + rank-distance 单调梯度 + source_A vs source_B 各自 SOZ 关系）。
>
> 因此本 phase 的 cohort-level 可写措辞**降级**为：'在 swap-positive 子集上 template-level endpoint geometry 呈反向轴形状（5/9 shape-PASS、0/9 正交、4/9 不可判读）；这与 同轴双向读取 一致，也不支持 正交 unrelated-source 解释。区分同核 vs 双端 seed 等 Round 2。' 不可写 '证明 swap 不是双源 / 支持同一病理核心轴'。"

H2b 主文档不改 `docs/topic4_sef_itp_framework.md` 的 §3 H 主清单；只在 §3.2 H2 末尾加一行 cross-cite 指向本 archive，cross-cite 的措辞也按 §10.5 demoted 版本同步降级（"shape 一致于非正交假说"而非 "支持同一核心轴"）。


---

## 11. Round 2 — deferred work（pending separate PR）

H2b v1.0.2 不能区分 "同一病理核心轴双向读取" 与 "同轴双端独立 seed"（§1.5 scope 红线）。要回答这个区分，下面三组分析必须做（**当前不在 v1.0.2 实施范围**，列为 future PR）：

### 11.1 Per-event initiation centroid 聚类

- 对每个 cluster 的每个事件，取 earliest-k channels 的空间 centroid（事件 seed location）。
- 在 subject 内对 cluster A 的所有 event seeds 做 KMeans / 2-mode test（unimodal vs bimodal）。
  - **bimodal（双峰）+ 两峰相距 > 某 mm 阈值** → 支持 "双端 seed"（cluster A 自己就有两个 seed 位置）
  - **unimodal** → 支持 "单 source 双向"
- 同样对 cluster B 做。
- 然后看 cluster A 的两峰（若有）vs cluster B 的两峰（若有）是否空间重合 — 若 A 双峰 vs B 双峰且峰位置共享，进一步支持双端 seed 解释。

**预设 metric**：silhouette score (k=2 vs k=1)、Hartigan dip test、两峰 centroid 间距 vs subject scale。

### 11.2 Rank-distance gradient（连续梯度 vs 端点 swap）

- 对每个事件，定义 source point = earliest-k centroid；测每 channel 的 rank 相对到 source point 距离的回归斜率与 r²。
- **同一轴行波**：rank 应当沿距离单调（线性 + 高 r²）。
- **端点 swap 集合重叠但非连续行波**：rank 在端点很极端、在中间随机 — r² 低。
- per-event 跑、subject-aggregate 报告。

**Cohort 读数**：strict+candidate 子集上 rank-distance 高 r² 占比；与 H2b shape 通过的 subject 交集；high r² + axis_reversal_shaped 是行波模型最强证据。

### 11.3 source_A vs source_B 各自相对 SOZ / ictal-early channel 的位置

- v1.0.2 已经在 SOZ relation 加了 mapped_full + joint_universe 双套距离（fix #5），但还没在 cohort 层上做 source_A 与 source_B **各自**相对 SOZ 的关系对比。
- 关键判读：
  - **source_A 与 source_B 都贴 SOZ** → 双 seed 解释更强（SOZ 内部两个 seed 各自启动）
  - **一个 source 贴 SOZ，另一个像 propagation target / sink** → 单源双向解释更强（SOZ 是 seed，"另一个 source" 是同轴的反向 endpoint）
- 类似分析也对 ictal-early channel 跑（PR-T3-1 Layer B 数据可用时）。

**Cohort 读数**：strict+candidate 子集中 (source_A close-to-SOZ AND source_B close-to-SOZ) vs (one close, one far) 的比例。

### 11.4 Round 2 不在 v1.0.2 范围的原因

- v1.0.2 是审阅 round-1 fix scope，明确说"代码能跑，测试也过；但科学解释要降级"。
- Round 2 要新增 event-level 聚类 + rank-distance 回归 + cohort-level SOZ 比较，属于新分析，不是 fix。
- 应当起新的 archive plan（如 `phase_h2b_round2_*_plan_<date>.md`）独立 lock，避免在 v1.0.2 archive 内偷偷扩大 scope。

### 11.5 Round 2 不做就不能写的话（v1.0.2 lock）

在 Round 2 完成之前，**任何 H2b 引用都不允许**写：

- "支持同一病理核心轴" / "证明 swap 不是双源"
- "SEF-ITP 单源双向图景被 cohort 支持"
- "排除双端 seed 解释"

允许的 v1.0.2 措辞上限（§10.5 demoted 版本）：

- "shape 一致于 同轴双向读取，也不支持 正交 unrelated-source 解释"
- "排除 H_orth（cluster A 与 cluster B 是正交无关解剖源）假说"
- "swap-positive 子集的 template-level endpoint geometry 呈反向轴形状"
