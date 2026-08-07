# Topic 4 数据驱动病理场 — Stage 3 实施计划：全局二维场（rev2，2026-08-08 技术审阅后）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把病理场从共享轴的坐标系里解放出来，在整张片子上自由定位，回答一个预注册的问题——
**在不给位置先验的情况下，固定预算的病理异质性放在片子的不同位置，得分如何分布。**
产出一张**结果中性**的全空间地图（结论由数据填），以及一份可与合作者审阅的 handoff。

**⚠️ 目标的表述边界（2026-08-08 审阅）**：不得把"最优是某个局部区域"预写进目标——
那是待检验的结论，不是计划的前提。

**Spec:** `docs/superpowers/specs/2026-08-06-topic4-axis-constrained-data-driven-core-field-design.md`
（rev4）§9 全节 + §8.3 收口记录。**每个 Task 开工前重读对应 §。**

**上一阶段的结论（决定了本计划的形状）：** Stage 2 学出来的是一条**贯通全轴的细丝**，
预注册的双方向门第 2 条不过（合同 15 触点上两模板相关 −0.130，需 ≤ −0.2）。
归因是两个**参数化缺口**（spec §8.3.5）：场的轴向范围从未是自由参数；目标里没有任何一项
看"招募到几个触点"。**本计划就是修这两个缺口。**

---

## 架构：两条独立的腿（本计划最重要的设计决定）

Stage 2 的教训是：结局分类按顺序短路，`SIMULATOR_OVERFIT` 在规则 1 就触发，
把后面所有科学问题都挡在门外。如果 Stage 3 只做自由优化，一旦再次判成过拟合，
**"最优异质性在哪"这个问题又会拿不到答案，那张图也出不来。**

因此本计划把 Stage 3 拆成两条**互不依赖**的腿：

| | Leg A — 位置扫描 | Leg B — 自由优化 |
|---|---|---|
| 问的问题 | **把一个固定形状的异质性放在片子的各个位置，哪里得分最高** | 完全自由的搜索会收敛到什么 |
| 有没有优化 | **没有**（穷举网格） | 有（CMA-ES，≥3 次重启） |
| 有没有**优化器**过拟合 | 没有优化器可过拟合 | 有 —— 所以要 held-out |
| **有没有 winner's curse** | **有！**（见下） | 有 |
| 承担什么 | **描述性主交付**：一张不依赖优化器的地图 | 回答"能不能学出来" |
| 成本 | 见"预算"节 | 见"预算"节 |

**⚠️ 更正（2026-08-08 审阅）：先前写的"没有优化器，所以不会过拟合"是错的。**
Leg A 仍然是**从 49 × 2 个格点里挑最大值，而每格只有 4 个网络** ——
这就是多重搜索 + winner's curse，最高格的分数必然乐观偏。

**因此 Leg A 的纪律（全部预注册，写进 Task 4）**：

1. 地图作为**描述性**主交付 —— 报告整张地图的形状，不报告"最优格"
2. **禁止**把最高格称为"确定的最优位置"。允许的表述是"得分较高的区域"
3. **最高区域必须用一批独立种子确认**（与建图的 4 个种子不相交），
   确认结果与建图值并列报告；两者的差就是 winner's curse 的直接估计
4. 图上**必须逐格显示有效分母**（几个种子能算出匹配分），见 Task 7

**Leg A 是这一轮的主交付**（描述性），Leg B 是能力检验，允许失败。

---

## Global Constraints

- 被试 `epilepsiae_1146`，montage `narrow`，placement `gradient_shared`。
- **`E→E` 各向异性 `theta_deg = -22.8` 仍然冻结**——这就是"只给一个方向的限制"。任何代码不得重估。
- **释放的只有场的位置与范围。** LIF / 突触 / 慢变量方程、虚拟触点与读出、
  冻结的 15 触点打分支撑集与缺失规则、预算 `Σ h_i = N_core_manual = 1129`——全部不变。
- **不修改** `src/snn_engine/`、`scripts/paper_figures/plot_fig_subject_snn*.py`。
  `scripts/run_sef_hfo_subject_snn.py` 只允许再加 opt-in 参数（默认关，已发表跑法字节不变）。
- **不得因为 §10.3 第 2 条没过就去改打分支撑集**（spec §9.1）。支撑集是冻结的 15 个触点。
- **禁止把 §10.3 第 2 条（两模板可区分度）放进目标函数**（spec §9.3）。它是门，不是目标；
  放进目标之后的"通过"没有证据力。
- **禁止从 Stage 2 的最优场 warm start**（spec §9.2）——那等于把"贯通细丝"这个答案喂回去。
- **held-out 种子集合必须在开跑前冻结进 config 并计入校验和**（spec §9.5）。
  Stage 2 的池子是事后扩大的，这一轮不许重演。
- 全部产出落 `results/topic4_sef_hfo/data_driven_core_field_stage3/`。
  含图目录**必须**有中文 `figures/README.md`（`### filename` + 2–4 句 + 一行 `**关注点**：`），**图看过之后**写。
- 纯计算测试必须秒级；仿真相关标 `@pytest.mark.integration` / `@pytest.mark.slow`。
- **实现与测试必须先提交，再跑任何仿真。**

## 文件结构

| 文件 | 职责 |
|---|---|
| `src/topic4_core_field_stage3.py` | 自由中心高斯混合参数化、边界、`h` 构造、空间诊断量（`r̄`、`C_axis`）、单块探针（Leg A 用） |
| `src/topic4_core_field_stage3_outcome.py` | Stage 3 结局分类（开跑前冻结）+ 位置稳定性 |
| `scripts/run_topic4_core_field_stage3_sweep.py` | **Leg A**：位置 × 尺寸网格扫描 |
| `scripts/run_topic4_core_field_stage3_optimize.py` | **Leg B**：CMA-ES，≥3 次重启，可续跑 |
| `scripts/run_topic4_core_field_stage3_heldout.py` | held-out 评估 + §10.3 四条门重测（冻结种子集） |
| `scripts/plot_topic4_core_field_stage3.py` | 新的 A 面板（全空间得分图）+ 四列图重出 |
| `tests/test_topic4_core_field_stage3.py` | Task 1 |
| `tests/test_topic4_core_field_stage3_key.py` | Task 2 |
| `tests/test_topic4_core_field_stage3_outcome.py` | Task 3 |

复用（不重写）：`src/topic4_core_field.py`（`project_to_budget` / `signed_depth` / `build_vth` /
`sample_core_quantiles` / `core_thresholds`）、`src/topic4_core_field_scoring.py`（模板、2×2、
交换不变得分）、`src/topic4_core_field_runner.py`（网络缓存、原子写、provenance）、
`src/topic4_core_field_cmaes.py`。

---

### Task 1: 自由中心高斯混合参数化 + 空间诊断量

**重读 spec：** §9.1、§9.2、§9.4

**Files:** Create `src/topic4_core_field_stage3.py`；Test `tests/test_topic4_core_field_stage3.py`

**Interfaces:**

```python
K_COMPONENTS = 3          # 不用 K=2 —— 那会把"两个核"写进先验
SIGMA_MIN_MM, SIGMA_MAX_MM = 0.4, 6.0
CENTER_MARGIN_MM = 2.0    # 与 AXIAL_MARGIN 同值，中心不许贴边

def n_params(K=K_COMPONENTS) -> int:
    """5K + K：每分量 (cx, cy, log s_par, log s_perp, phi) + K 个 softmax 权重。
    softmax 平移不变 ⇒ 自由度 5K + K - 1（K=3 时 17）。"""

def unpack(theta, K, L) -> list[dict]:
    """裁到物理边界后的分量列表。中心裁到 [margin, L-margin]，sigma 裁到 [SIGMA_MIN, SIGMA_MAX]。"""

def params_to_q(theta, pos_xy, K, L) -> np.ndarray:
    """sum_k w_k * exp(-0.5 (x-c_k)^T Sigma_k^{-1} (x-c_k)) + EPS，pos_xy 是 sheet 坐标。"""

def params_to_h(theta, pos_xy, K, L, target_count) -> np.ndarray:
    """params_to_q 之后走 project_to_budget，预算不变。"""

def probe_q(pos_xy, center_xy, sigma_mm) -> np.ndarray:
    """Leg A 的单块各向同性探针。只有位置和尺寸两个自由度。"""

def spatial_diagnostics(h, pos_xy, center, axis_unit, deltas=(1.0, 2.0, 3.0)) -> dict:
    """必须用 h 作空间质量，不得用 h*d（spec §9.4）。
    返回 r_bar（横向重心）、s_bar（轴向重心）、C_axis[delta]、rms_transverse、rms_axial、
    n_effective_lobes（h 的 2D 直方图上超过半高的连通分量数）。"""
```

- [ ] **Step 1: 写失败测试**

关键断言（每条对应一个能被违反的合同）：

1. **实际优化 17 维，不是 18 维**（2026-08-08 审阅更正）：`n_free(3) == 17`。
   softmax 的平移方向是冗余的，把它交给 CMA-ES 会浪费一个协方差方向。
   实现上：只存 `K-1` 个权重 logit，第 K 个固定为 0（`w = softmax([a_1..a_{K-1}, 0])`）。
   仍保留 `params_to_h` 对整体平移不变的测试（作为解析性质），但**优化器只见 17 维**
2. **中心越界必须被裁回**：`theta` 里放 `c = (-50, 100)`，解出的中心落在 `[2, 18]²` 内
3. **sigma 越界必须被裁回**：`log sigma = ±20` 解出 `sigma ∈ [0.4, 6.0]`
4. **预算恒成立**：任意随机 `theta`，`abs(params_to_h(...).sum() - 1129) < 1e-6`
5. **`h ∈ [0, 1]` 逐元素成立**（`project_to_budget` 的 expit 保证）
6. **不含方向先验（四条，缺一不可）**（2026-08-08 审阅补强：单独 90° 排除不了四重方向偏好）：
   - (a) 绕 `center` 旋转 **90°**，参数同步旋转 ⇒ `h` 逐元素不变
   - (b) 绕 `center` 旋转 **37°**（任意非特殊角）⇒ `h` 逐元素不变。
     **这条才是真正排除方向偏好的**；只测 90° 的实现可以有四重对称的偏好而测试仍绿
   - (c) `phi -> phi + π` ⇒ `h` 逐元素不变（椭圆的朝向以 π 为周期）
   - (d) **分量置换不变**：把 3 个分量的参数块任意重排 ⇒ `h` 逐元素不变
7. `spatial_diagnostics` 用 `h` 而非 `h*d`：构造 31% 为负的 `d`，断言 `r_bar` 与 `C_axis` 不依赖 `d`
8. `C_axis(delta)` 单调不减，且 `C_axis(inf) == 1`
9. **旋转不变性同样适用于诊断量**：`rms_transverse` 在把场整体旋转到轴上/轴外时按预期变化
10. `probe_q` 的中心与尺寸可辨识：两个不同中心的探针，`h` 的相关 < 0.9

- [ ] **Step 2: 实现，跑测试到绿**
- [ ] **Step 3: 提交** `feat(topic4-stage3): free-centre mixture parameterisation with no axis prior`

**验收：** 测试 6 的 (b) 任意角必须真的失败过一次——先写一个只在 90° 倍数下不变的实现
（例如把 `phi` 量化到 π/2 网格）看它红，再改对。这一条是本 Task 的存在理由。

---

### Task 2: 加一层触点招募档的候选键（补 spec §8.3.5 缺口 2）

**重读 spec：** §9.3、§5.3（分级不加权的理由）

**Files:** append `src/topic4_core_field_scoring.py`；Test `tests/test_topic4_core_field_stage3_key.py`

**Interfaces:**

**⚠️ 更正（2026-08-08 审阅）：招募指标必须是逐方向的最小者，不是并集。**
spec §5.5 规定并集只作参考，承重的是**正、反方向各自覆盖多少触点**。
用并集会错误奖励"一个方向覆盖 15 个、另一个只覆盖 3 个"的场 ——
而这恰恰就是 Stage 2 那条细丝的失效形态（正向 6.6 触点、反向 8.3 触点，但正向到不了远端）。

```python
def recruited_per_direction(events, support, part_min) -> tuple[int, int]:
    """(正向覆盖数, 反向覆盖数)：冻结支撑集里，在该方向的**任一**干净可定向事件中
    出现过的触点数。并集另作报告量，**不进键**。"""

def recruited_contacts(events, support, part_min) -> int:
    """承重量 = min(正向覆盖数, 反向覆盖数)。单方向运行返回 0。"""

def coverage_tier(n_recruited, step=3) -> int:
    """floor(n_recruited / step)，0..5。粗分级：读出太稀疏，连续权重会把噪声当信号。"""

def candidate_key3(n_dir, n_recruited, s_rank) -> tuple:
    """(n_dir, coverage_tier(min(fwd, rev)), S_rank)。S_rank 永远不跨档相减。"""
```

- [ ] **Step 1: 写失败测试**

1. `candidate_key3(2, 15, 0.1) > candidate_key3(2, 11, 0.9)` —— **招募档压倒秩次分**
2. `candidate_key3(2, 5, x) > candidate_key3(1, 15, y)` 对任意 x,y —— **方向数仍是最高位**
3. `n_dir=0 ⇒ S_rank=NaN ⇒ 键的第三位是 -inf`，且不与有限值比较出 NaN
4. `coverage_tier` 在 step 边界上的行为写死：`{0..2→0, 3..5→1, ..., 15→5}`
5. `recruited_contacts` 只数支撑集内的触点，支撑集外出现的触点不计
6. **不对称必须被惩罚**（本 Task 的存在理由）：构造一个正向覆盖 15、反向覆盖 3 的事件集，
   `recruited_contacts == 3`（不是 15，也不是并集的 15）；
   其 `coverage_tier` 必须**低于**一个正反各覆盖 9 个的事件集
7. 单方向运行（`n_dir == 1`）⇒ `recruited_contacts == 0` ⇒ `coverage_tier == 0`
8. **回归锁**：Stage 2 的贯通细丝那批 562 个事件喂进来，逐方向覆盖数应为
   （正向 11、反向 11 —— 但正向远端 3 个触点各只有 2–10 次观测），
   并集与 min 的差必须在测试里显式断言下来，防止将来有人改回并集

- [ ] **Step 2: 实现，跑测试到绿**
- [ ] **Step 3: 提交** `feat(topic4-stage3): rank recruitment above template match in the candidate key`

---

### Task 3: Stage 3 结局分类（开跑前冻结）

**重读 spec：** §9.4、§8.1（短路顺序本身是合同）

**Files:** Create `src/topic4_core_field_stage3_outcome.py`；Test `tests/test_topic4_core_field_stage3_outcome.py`

**Interfaces:**

```python
OUTCOME_ORDER = (
    "FAIL_CLOSED",              # 0 NaN / 缺产物 / 校验和不匹配
    "POSITION_UNIDENTIFIABLE",  # 1 跨重启 r_bar 标准差 >= 1mm，或等价最优场族两两相关中位 < 0.5
    "SIMULATOR_OVERFIT",        # 2 训练分提高、held-out 不提高
    "AXIS_REDISCOVERED",        # 3 |r_bar| < 1mm 且 C_axis(2mm) >= 0.7，跨重启一致
    "AXIS_NOT_REQUIRED",        # 4 稳定落在离轴 > 2mm，且分数不低于轴上解
    "AXIS_INCONCLUSIVE",        # 5 以上皆不触发
)
def classify_stage3(results) -> dict:
    """返回**全部**分量，不只返回一个短路标签（2026-08-08 审阅更正）。

    {
      "primary_outcome":          OUTCOME_ORDER 里的一个（短路顺序决定），
      "position_stable":          bool | None,   # 跨重启 r_bar 标准差 < 1mm 且场族相关中位 >= 0.5
      "transfers_to_heldout":     bool | None,   # 训练提高时 held-out 是否也提高
      "axis_relation":            "near" | "off" | "inconclusive",
      "all_triggered_conditions": [规则号...],   # 所有成立的条件，不止第一个
      "allowed_statement":        str,
      "measurements":             {...}          # r_bar / C_axis / 场相关 / 分数差，原样带出
    }
    """
```

**⚠️ 为什么要返回全量（2026-08-08 审阅）**：Stage 2 只返回一个短路标签，
结果 `SIMULATOR_OVERFIT` 一触发，后面所有科学问题都被遮住（spec §8.3.4 的教训）。
仅仅把顺序调换会**把同一个遮挡问题反向重演**。
`primary_outcome` 保留为主标签（用于一句话汇报），但**四个分量必须同时可读**。

**⚠️ 短路顺序相对 Stage 2 有一处刻意的改动，必须在 plan 里说明理由：**
`POSITION_UNIDENTIFIABLE` 排在 `SIMULATOR_OVERFIT` **之前**。
理由：位置不可辨识是比过拟合更基本的失败——若不同重启给出互不相似的场，
"训练/held-out 分数差"这个量本身没有稳定的对象可谈。
Stage 2 把过拟合放在第一位，结果把所有科学问题都挡在门外（spec §8.3.4 的教训）。

- [ ] **Step 1: 写失败测试** —— 每个结局至少一个 fixture，另加：
  - **测顺序**：同时满足"位置不稳"与"过拟合"时 `primary_outcome == "POSITION_UNIDENTIFIABLE"`
  - **测不遮挡**（本 Task 的存在理由）：同一个 fixture 下，
    `transfers_to_heldout is False` **且** `position_stable is False` **且**
    `len(all_triggered_conditions) == 2` —— 即主标签没有把另一条事实吃掉
  - `axis_relation` 在 `primary_outcome` 是失败标签时**仍然要被计算并返回**（可以是 "inconclusive"），
    不得返回 None 了事
- [ ] **Step 2: 实现，跑测试到绿**
- [ ] **Step 3: 提交** `feat(topic4-stage3): freeze the outcome taxonomy before any simulation`

**验收：** 提交时间戳必须早于任何 Stage 3 仿真产物的时间戳。

---

### Task 4: Leg A — 位置 × 尺寸扫描（描述性主交付）

**重读 spec：** §9.4（对照）、§9.0（为什么不能只做优化）；本 plan"架构"节的 winner's curse 纪律

**Files:** Create `scripts/run_topic4_core_field_stage3_sweep.py`

**设计（全部预注册，开跑前写进 config 并计入校验和）：**

- **网格**：片内 `7 × 7 = 49` 个中心，覆盖 `[2, 18]²`，间距 `2.67 mm`
- **尺寸**：`sigma = 1.2 mm` **预注册为主地图**；`sigma = 2.4 mm` **单独作为敏感性地图**
  （2026-08-08 审阅：**不得逐格取两者最大值** —— 那是又一层多重搜索）
- **建图种子**：`SWEEP_SEEDS = 4` 个，与 Leg B 的训练/held-out 种子**全不相交**
- **确认种子**：`CONFIRM_SEEDS = 6` 个，与建图种子**也不相交**，只用于确认得分较高的区域
- **预算不变**：每个探针都走 `project_to_budget(probe_q(...), 1129)`
  ⇒ 所有格点的病理细胞总数完全相同，**比较的是位置和形状，不是剂量**

**每格记录（缺一不可）：**

| 层 | 量 | 为什么需要 |
|---|---|---|
| 1 | `n_events`、`n_dir`、正/反事件数 | 单块探针可能只出一个方向；`S_rank` 会是 NaN，此时地图必须仍有内容 |
| 2 | `recruited_per_direction` 与 `min(fwd, rev)` | 直接对应 spec §8.3.5 缺口 2；**承重量是 min，不是并集** |
| 3 | `S_rank`（交换不变） | 只在 `n_dir=2` 的格点上有定义 |
| 4 | **`n_valid` = 该格 4 个种子里有几个能算出 `S_rank`** | winner's curse 纪律第 4 条；进图 |

**⚠️ winner's curse 的处理（预注册，不可事后改）：**

1. 主地图报告 `S_rank` 的**跨种子均值**与 `n_valid`。**`n_valid < 2` 的格点一律标为不可评估**，
   不参与任何排序，也不参与"最高区域"的定义
2. "得分较高的区域"定义为：主地图上 `S_rank` 均值排前 **10%** 且 `n_valid >= 3` 的**连通区**，
   **不是单个最高格**
3. 该区域的每个格点用 `CONFIRM_SEEDS`（6 个独立种子）重跑一遍，
   **建图值与确认值并列报告**，两者之差即 winner's curse 的直接估计
4. 报告里**禁止**出现"最优位置是 (x, y)"。允许的是"得分较高的区域覆盖 …，
   在独立种子上确认后的值为 …（比建图值低 …）"

- [ ] **Step 1**: 写 config 冻结器（网格、两个尺寸、建图种子、确认种子、校验和），提交
- [ ] **Step 2**: 跑主地图（`sigma=1.2`），每格原子写单独 JSON，支持中断续跑
- [ ] **Step 3**: 跑敏感性地图（`sigma=2.4`），同样落盘
- [ ] **Step 4**: 定"得分较高的区域"（按上面第 2 条的规则，纯函数，有测试），跑确认种子
- [ ] **Step 5**: 汇总 `sweep_summary.json`：每格四层量 + `n_valid` + 确认结果 + NaN 占比

**⚠️ 报告纪律：** 若某一层在大片格点上是 NaN，**必须在汇总里显式报告 NaN 的占比**，
不得只画有值的部分（等于静默截断，AGENTS.md 结果规范）。

**⚠️ 产物审计（2026-08-08 审阅新增）：** 汇总时必须断言
(a) 格点数 × 尺寸档 × 种子数 == 实际产物数；(b) 种子集合**恰好**等于 config 里冻结的集合、无重复；
(c) 每个产物带 `config_checksum` 与 `provenance`，且 checksum 与 config 一致。
任一条不满足 ⇒ **fail closed，不出图**。

### Task 5: Leg B — 自由优化（CMA-ES，≥3 次重启）

**重读 spec：** §9.2、§9.3、§9.7

**Files:** Create `scripts/run_topic4_core_field_stage3_optimize.py`

**⚠️ 预算必须写清"候选数"和"仿真次数"是两回事（2026-08-08 审阅更正）：**

```
每个候选评估用的网络数  SEEDS_PER_CANDIDATE = 1     （与 Stage 2 一致：训练期单种子，
                                                     靠 common random numbers 控方差）
popsize 10 × 15 代 × 3 次重启 = 450 个候选
⇒ 仿真次数 = 450 × 1 = 450 次
```

**若把 `SEEDS_PER_CANDIDATE` 提到 4，仿真次数就是 1800 次、约 30 小时** ——
本轮**不做**，理由：训练期的方差由 common random numbers 控制，
真正需要多网络的是 held-out（Task 6），把预算花在那里更划算。
**这是一个明确的取舍，写进报告。**

- 17 个自由参数（`K=3`，见 Task 1 第 1 条），复用 `src/topic4_core_field_cmaes.py`
- **初始化**：`c_k` 片内均匀随机，`sigma` 取中位尺度，权重 logit = 0。**禁止 warm start**
- **重启 ≥ 3 次**，每次独立随机初始化；三次的最优场都要保留（§8.2 的等价最优场族协议原样适用）
- Common random numbers：同一代内所有候选共用同一组种子
- 候选键用 Task 2 的 `candidate_key3`（承重量是 `min(fwd, rev)` 覆盖）
- 每代 checkpoint：优化器状态、全部候选 `theta`、四层量、`r_bar`、`C_axis`、运行时。可续跑

- [ ] **Step 1**: 驱动器 + checkpoint/续跑，提交
- [ ] **Step 2**: 跑三次重启
- [ ] **Step 3**: 记录跨重启的场相关矩阵与 `r_bar` 标准差（喂给 Task 3 的分类器）

### Task 6: held-out 评估 + §10.3 四条门重测

**重读 spec：** §9.5、§10.3、§8（必要对照四个）

**Files:** Create `scripts/run_topic4_core_field_stage3_heldout.py`

**⚠️ 必须评估的场（2026-08-08 审阅更正：先前只写了 3 个重启最优场，漏了 spec §8 要求的参照）：**

| # | 场 | 为什么必须在 held-out 上重算 |
|---|---|---|
| 1–3 | Leg B 三次重启各自的最优场 | 主对象；跨重启是否一致 |
| 4 | **Stage 2 的贯通细丝** | 上一阶段的解，必须能直接比较"有没有变好" |
| 5 | **手放的两个核** | 外部参照 |
| 6 | **同预算均匀走廊** | spec §8 的必要对照；"结构化的场有没有必要" |
| (7) | Leg A 得分较高区域的代表探针 | 若 Leg A 已完成则加入；两条腿的交叉验证 |

⇒ **6 个场 × 120 网络 = 720 次仿真**（含第 7 个则 840 次）。
先前写的 360 次是**只算了 3 个场**，低估了一半。

**⚠️ 相对 Stage 2 的强制改动（spec §9.5）：**

1. **held-out 种子集合在开跑前冻结进 config 并计入校验和。** 上一轮是事后扩池的，不许重演
2. **池子规模一次到位**：`120` 个种子 × 8 s
3. **§10.3 必须在冻结的 15 触点支撑集上算**，不得用任何显示过滤后的子集
4. **必须同时报告"每方向最少观测数"分层的敏感性**（spec §8.3.4 的教训：
   这个相关系数高度依赖纳入哪些触点，单一数字会误导）
5. 若第 2 条仍不过，**必须按 §8.3.5 的方式归因到具体的参数化缺口**，不得笼统写"做不到"

- [ ] **Step 1**: 冻结 held-out config（120 种子，与训练/扫描/确认种子全不相交），提交
- [ ] **Step 2**: 在 held-out 上跑全部 6（或 7）个场
- [ ] **Step 3**: 调 `scripts/audit_topic4_core_field_bidirectional_gate.py`（已存在），
      并扩成"逐观测数门槛的敏感性表"
- [ ] **Step 4**: 调 Task 3 的分类器，写 `stage3_outcome.json`（**返回全量分量**，不只主标签）

### Task 7: 新的 A 面板 —— 全空间得分图（用户点名的交付物）

**重读 spec：** §9.4；`docs/figure_style_guide.md` Topic 4 小节 + §0 全局硬规则

**Files:** Create `scripts/plot_topic4_core_field_stage3.py`

**替换关系（用户 2026-08-07 裁定）：** 现有 `learned_core_field_readout.png` 的第一格
（沿轴一维剖面）**换成**一张全空间二维图。后三格（forward / reverse 事件图 + 虚拟 SEEG）
沿用现有渲染函数。

**⚠️ 全部为 2026-08-08 审阅后的修订版规格：**

**主图 A 面板：**

- **底图** = **`sigma = 1.2 mm` 的主地图**（`S_rank` 跨建图种子均值）。
  `sigma = 2.4 mm` 出**单独的敏感性图**，**不得逐格取两者最大值**
- **逐格有效分母必须可见**：格内标注 `n_valid/4`（例如 `3/4`）。
  `0/4` 的格子画成灰色并在图例注明"单方向，匹配分无定义"；
  `1/4`–`2/4` 的格子加**网纹**表示估计不可靠。
  **不得**只对有限值求均值然后当成满格数据画出来
- **不做平滑插值**：用 `pcolormesh` / `imshow(interpolation="nearest")`。
  插值会制造并不存在的空间精度
- **等值线** = Leg B 三次重启各自最优场的 90% 质量等值区，三条不同线型
- **得分较高区域的独立确认值**：在该区域旁标注"确认 = …（建图 = …）"，
  **两个数并列**，不得只画一个
- **手放的两个核** = 小空心圈，图例写死 `reference, not an input to this run`
- **共享轴** = 细虚线。图例文字锁定为
  **`frozen E→E anisotropy axis; no field-location constraint`**
  （2026-08-08 审阅：先前写的"唯一保留的先验"不准确——评分目标同样来自患者秩次）
- **标题必须结果中性**：例如 `where a fixed-budget heterogeneity scores across the sheet`。
  **禁止**预写"最优的是某个局部区域"这类已含结论的标题

**第二张图（不塞进主图，CLAUDE.md §7）：**

- 左：**双方向网络比例**（该格 4 个种子里有几个跑出两个方向），
  **不画"平均方向数"**（含义模糊，2026-08-08 审阅）
- 右：**招募图画 `min(正向覆盖, 反向覆盖)`**，不画并集

**后三格用哪个场，必须在开跑前规定（2026-08-08 审阅新增）：**

- 预注册规则：**用 Leg B 三次重启中 held-out 得分最高的那个场**
- **若三次重启的场两两相关中位 < 0.5（即 `POSITION_UNIDENTIFIABLE`）**：
  **不得**展示任何一个"代表性学得场"。此时后三格改为**三列并排**，
  每列一次重启，标题写明"restart 1/2/3"，让不可辨识性直接可见

**⚠️ 表述纪律：** 图上不得出现 `S_rank` / `n_dir` / `coverage_tier` 这类内部字段名；
用读者语言（`template match`、`both directions readable`、`contacts recruited`）。

**⚠️ 可复现性（2026-08-08 审阅新增）：**

- `figures/README.md` 必须给出**逐字可粘贴的 producer 命令**（含全部参数），
  照 `data_driven_core_field/figures/README.md` 的格式
- 必须有 `tests/test_topic4_core_field_stage3_figure.py`：
  用小型合成数据断言 (a) `n_valid` 的计算、(b) 灰格/网纹的分类、
  (c) "得分较高区域"选取函数、(d) metadata 里 checksum 与 config 一致

- [ ] **Step 1**: 渲染 → **亲自目视** → 改 → 再渲染
- [ ] **Step 2**: 写 `figures/README.md`（中文，图看过之后写，含 producer 命令）
- [ ] **Step 3**: 提交

### Task 8: 收口报告与 handoff 更新

**Files:** append spec §9.9（Stage 3 实测收口）；update `docs/superpowers/handoffs/2026-08-07-topic4-data-driven-core-field.md`

- [ ] 按 §8.3 的同一格式写 Stage 3 收口：三个阶段结局、场的形状、门的逐条判定、
      失败归因的界定、允许/禁止的表述
- [ ] 更新 handoff 的"当前状态"与"下一步"
- [ ] **⚠️ 必查 1**：任何"最优在某处"的表述，都要写清是 Leg A（无优化的扫描）还是
      Leg B（优化）得到的。两条腿的证据强度不同，不得混为一谈
- [ ] **⚠️ 必查 2（2026-08-08 审阅新增）：Stage 3 并没有消除轴的循环性。**
      它只取消了"场必须沿轴铺"这一条。**连接的方向性仍然固定在由患者秩次拟合出的轴上，
      评分目标也来自同一批秩次。** 因此即使自由场回到轴附近，
      **禁止**写"独立地重新发现了传播轴"。唯一允许的表述是：

      > 在一个传播方向已按患者秩次设定的网络里，自由的场搜索是否也把兴奋性
      > 集中到该方向附近。

- [ ] **⚠️ 必查 3**：所有相关系数必须写明算在哪些触点上（spec §8.3.4 的教训）；
      所有"纯度"必须写明是 in-sample 还是交叉验证

---

## 预算与排期（2026-08-08 审阅后重算 —— 先前的 1200 次 / 24 h 明显低估）

实测吞吐：**约 1 分钟/次仿真**（10 并发，机器被其它作业占着；单次 8 s 仿真裸时 7–25 分钟）。

| 阶段 | 仿真次数 | 墙钟（10 并发） |
|---|---|---|
| Task 1–3（纯计算 + 测试） | 0 | ~1.5 h |
| Task 4 Leg A 主地图 `sigma=1.2` | 49 × 4 = **196** | ~3.3 h |
| Task 4 Leg A 敏感性 `sigma=2.4` | 49 × 4 = **196** | ~3.3 h |
| Task 4 Leg A 高分区确认 | ~5 格 × 6 = **30** | ~0.5 h |
| Task 5 Leg B（3 重启 × 150 候选 × 1 网络） | **450** | ~7.5 h |
| Task 6 held-out（**6 个场** × 120 网络） | **720** | ~12 h |
| Task 7–8 图与报告 | 0 | ~2.5 h |
| **合计** | **~1592** | **~30.6 h** |

（含 Leg A 代表探针的第 7 个 held-out 场则为 **1712 次 / ~32.6 h**。）

**⇒ 这不是一个 8–10 小时能跑完的计划。** 分批执行，每批结束后落盘可续跑。

### 本轮（8–10 小时窗口）执行的范围

| 做 | 不做 |
|---|---|
| Task 1–3：参数化、候选键、结局分类 + 全部单元测试 | Task 5 Leg B |
| Task 4：Leg A 主地图 + 敏感性图 + 高分区独立确认（422 次仿真） | Task 6 held-out |
| Task 7 的**第二张图**（双方向比例 / 招募图）+ 主 A 面板的**底图部分** | Task 7 的等值线（需要 Leg B） |

**本轮窗口内预计**：~1.5 h 测试 + ~7.1 h 仿真 ≈ **8.6 h**。
Leg B 与 held-out 留到下一批，因为它们合计还要 ~19.5 h。

**若时间超限的降级顺序**（必须在报告里写明降了什么）：
敏感性地图 `sigma=2.4` 整档砍掉（省 3.3 h）→ 高分区确认种子 6→4（省 0.2 h）。
**建图种子 4 个和高分区确认这两项都不许砍** —— 前者是 `n_valid` 的分母，
后者是 winner's curse 的唯一估计。

## 本计划刻意不做的事

- **不改打分支撑集**，即使 §10.3 第 2 条又不过（spec §9.1）
- **不把门的第 2 条放进目标函数**（spec §9.3）
- **不做 `K > 3`** —— 先看 `K=3` 够不够；`K` 是下一轮的自由度
- **不碰患者侧数据**：held-out 只在模型侧（spec §10.1 用户裁定）
- **不重跑已发表的 Fig4 系列产物**（k_dir 事实按用户裁定只记录不重跑，spec §2.4a）
