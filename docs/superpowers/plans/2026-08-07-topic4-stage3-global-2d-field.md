# Topic 4 数据驱动病理场 — Stage 3 实施计划：全局二维场（rev1）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把病理场从共享轴的坐标系里解放出来，在整张片子上自由定位，回答一个预注册的问题——
**在不给位置先验的情况下，最优的病理异质性落在哪里。** 产出一张能独立说明"全空间中最优是某个
学出来的局部区域"的图，以及一份可与合作者审阅的 handoff。

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
| 会不会过拟合优化器 | **不会**（没有优化器） | 会 —— 所以要 held-out |
| 承担什么 | **产出用户要的那张图**；给"最优在哪"一个不依赖优化器的答案 | 回答"能不能学出来" |
| 成本 | ~392 次仿真（~7 h @ 10 并发） | ~450 次评估（~8 h） |

**Leg A 是这一轮的主交付。** 它不需要优化成功，图和结论都成立。
Leg B 是能力检验，允许失败。

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

1. `n_params(3) == 18`，且 `params_to_h` 对 `theta` 加常数（softmax 平移）**逐元素不变**
2. **中心越界必须被裁回**：`theta` 里放 `c = (-50, 100)`，解出的中心落在 `[2, 18]²` 内
3. **sigma 越界必须被裁回**：`log sigma = ±20` 解出 `sigma ∈ [0.4, 6.0]`
4. **预算恒成立**：任意随机 `theta`，`abs(params_to_h(...).sum() - 1129) < 1e-6`
5. **`h ∈ [0, 1]` 逐元素成立**（`project_to_budget` 的 expit 保证）
6. **不含轴的先验**：把整套神经元坐标绕 `center` 旋转 90°，同时把 `theta` 里的中心与 `phi`
   做同样旋转，`h` 必须逐元素不变（**证明参数化本身没有偏好任何方向**——这是 Stage 2 做不到的）
7. `spatial_diagnostics` 用 `h` 而非 `h*d`：构造 31% 为负的 `d`，断言 `r_bar` 与 `C_axis` 不依赖 `d`
8. `C_axis(delta)` 单调不减，且 `C_axis(inf) == 1`
9. **旋转不变性同样适用于诊断量**：`rms_transverse` 在把场整体旋转到轴上/轴外时按预期变化
10. `probe_q` 的中心与尺寸可辨识：两个不同中心的探针，`h` 的相关 < 0.9

- [ ] **Step 2: 实现，跑测试到绿**
- [ ] **Step 3: 提交** `feat(topic4-stage3): free-centre mixture parameterisation with no axis prior`

**验收：** 测试 6（旋转不变）必须真的失败过一次——先写一个带轴先验的实现看它红，再改对。
这一条是本 Task 的存在理由。

---

### Task 2: 加一层触点招募档的候选键（补 spec §8.3.5 缺口 2）

**重读 spec：** §9.3、§5.3（分级不加权的理由）

**Files:** append `src/topic4_core_field_scoring.py`；Test `tests/test_topic4_core_field_stage3_key.py`

**Interfaces:**

```python
def recruited_contacts(events, support, part_min) -> int:
    """冻结支撑集里，在该次运行的**任一**干净可定向事件中出现过的触点数。
    预注册为"出现 >= 1 次"——粗分级会吸收噪声，连续比例另作报告量。"""

def coverage_tier(n_recruited, step=3) -> int:
    """floor(n_recruited / step)，0..5。粗分级：读出太稀疏，连续权重会把噪声当信号。"""

def candidate_key3(n_dir, n_recruited, s_rank) -> tuple:
    """(n_dir, coverage_tier(n_recruited), S_rank)。S_rank 永远不跨档相减。"""
```

- [ ] **Step 1: 写失败测试**

1. `candidate_key3(2, 15, 0.1) > candidate_key3(2, 11, 0.9)` —— **招募档压倒秩次分**
2. `candidate_key3(2, 5, x) > candidate_key3(1, 15, y)` 对任意 x,y —— **方向数仍是最高位**
3. `n_dir=0 ⇒ S_rank=NaN ⇒ 键的第三位是 -inf`，且不与有限值比较出 NaN
4. `coverage_tier` 在 step 边界上的行为写死：`{0..2→0, 3..5→1, ..., 15→5}`
5. `recruited_contacts` 只数支撑集内的触点，支撑集外出现的触点不计
6. **回归锁**：Stage 2 的贯通细丝那批事件喂进来，`recruited_contacts` 应为 11（沿轴那根杆），
   `coverage_tier == 3`；手放的核那批应更高。**这条锁住"新键确实能分开这两者"**

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
def classify_stage3(results) -> dict:  # 返回 outcome + allowed_statement + 触发规则号
```

**⚠️ 短路顺序相对 Stage 2 有一处刻意的改动，必须在 plan 里说明理由：**
`POSITION_UNIDENTIFIABLE` 排在 `SIMULATOR_OVERFIT` **之前**。
理由：位置不可辨识是比过拟合更基本的失败——若不同重启给出互不相似的场，
"训练/held-out 分数差"这个量本身没有稳定的对象可谈。
Stage 2 把过拟合放在第一位，结果把所有科学问题都挡在门外（spec §8.3.4 的教训）。

- [ ] **Step 1: 写失败测试** —— 每个结局至少一个 fixture，且**测顺序**：
  同时满足"位置不稳"与"过拟合"时必须返回 `POSITION_UNIDENTIFIABLE`
- [ ] **Step 2: 实现，跑测试到绿**
- [ ] **Step 3: 提交** `feat(topic4-stage3): freeze the outcome taxonomy before any simulation`

**验收：** 提交时间戳必须早于任何 Stage 3 仿真产物的时间戳。

---

### Task 4: Leg A — 位置 × 尺寸扫描（本轮主交付）

**重读 spec：** §9.4（对照）、§9.0（为什么不能只做优化）

**Files:** Create `scripts/run_topic4_core_field_stage3_sweep.py`

**设计（全部预注册，开跑前写进 config 并计入校验和）：**

- **网格**：片内 `7 × 7 = 49` 个中心，覆盖 `[2, 18]²`，间距 `2.67 mm`
- **尺寸**：`sigma ∈ {1.2, 2.4} mm` 两档（小/大局部异质性）
- **种子**：`4` 个，与 Leg B 的训练/held-out 种子**全不相交**
- **合计** `49 × 2 × 4 = 392` 次仿真，8 s/次，~7 h @ 10 并发
- **预算不变**：每个探针都走 `project_to_budget(probe_q(...), 1129)`，
  ⇒ 所有格点的病理细胞总数完全相同，**比较的是位置和形状，不是剂量**

**每格记录三层量（缺一不可）：**

| 层 | 量 | 为什么需要 |
|---|---|---|
| 1 | `n_events`、`n_dir`、正/反事件数 | 单块探针可能只出一个方向；`S_rank` 会是 NaN，此时地图必须仍有内容 |
| 2 | `n_recruited`（冻结支撑集上） | 直接对应 §8.3.5 缺口 2，也是本轮想改善的量 |
| 3 | `S_rank`（交换不变） | 只在 `n_dir=2` 的格点上有定义 |

- [ ] **Step 1**: 写 config 冻结器（网格、尺寸、种子、校验和），提交
- [ ] **Step 2**: 跑扫描，每格原子写单独 JSON，支持中断续跑（照 Stage 2 的 checkpoint 模式）
- [ ] **Step 3**: 汇总成 `sweep_summary.json`（每格三层量 + 跨种子均值与标准差）

**⚠️ 报告纪律：** 若某一层在大片格点上是 NaN，**必须在汇总里显式报告 NaN 的占比**，
不得只画有值的部分（等于静默截断，AGENTS.md 结果规范）。

---

### Task 5: Leg B — 自由优化（CMA-ES，≥3 次重启）

**重读 spec：** §9.2、§9.3、§9.7

**Files:** Create `scripts/run_topic4_core_field_stage3_optimize.py`

- 17 个自由参数（`K=3`），复用 `src/topic4_core_field_cmaes.py`
- **初始化**：`c_k` 片内均匀随机，`sigma` 取中位尺度，`alpha = 0`。**禁止 warm start**
- **重启 ≥ 3 次**，每次独立随机初始化；三次的最优场都要保留（§8.2 的等价最优场族协议原样适用）
- Common random numbers：同一代内所有候选共用同一组种子
- 候选键用 Task 2 的 `candidate_key3`
- 每代 checkpoint：优化器状态、全部候选 `theta`、三层量、`r_bar`、`C_axis`、运行时。可续跑
- **预算**：popsize 10 × 15 代 × 3 次重启 = 450 次评估，~8 h。时间封顶后按已完成代数收口

- [ ] **Step 1**: 驱动器 + checkpoint/续跑，提交
- [ ] **Step 2**: 跑三次重启
- [ ] **Step 3**: 记录跨重启的场相关矩阵与 `r_bar` 标准差（喂给 Task 3 的分类器）

---

### Task 6: held-out 评估 + §10.3 四条门重测

**重读 spec：** §9.5、§10.3

**Files:** Create `scripts/run_topic4_core_field_stage3_heldout.py`

**⚠️ 相对 Stage 2 的强制改动（spec §9.5）：**

1. **held-out 种子集合在开跑前冻结进 config 并计入校验和。**
   Stage 2 是在看到"4 个反向事件不够评估"之后才扩池的，这一轮不许重演
2. **池子规模一次到位**：`120` 个种子 × 8 s（Stage 2 的实测证明这个量在单次跑里估不稳）
3. **§10.3 必须在冻结的 15 触点支撑集上算**，不得用任何显示过滤后的子集
4. 若第 2 条仍不过，**必须按 §8.3.5 的方式归因到具体的参数化缺口**，不得笼统写"做不到"

- [ ] **Step 1**: 冻结 held-out config（120 种子，与训练/扫描种子全不相交），提交
- [ ] **Step 2**: 在 held-out 上重跑 Leg B 的三次重启最优场
- [ ] **Step 3**: 调 `scripts/audit_topic4_core_field_bidirectional_gate.py`（已存在，逐条判定）
- [ ] **Step 4**: 调 Task 3 的分类器，写 `stage3_outcome.json`

---

### Task 7: 新的 A 面板 —— 全空间得分图（用户点名的交付物）

**重读 spec：** §9.4；`docs/figure_style_guide.md` Topic 4 小节 + §0 全局硬规则

**Files:** Create `scripts/plot_topic4_core_field_stage3.py`

**替换关系（用户 2026-08-07 裁定）：** 现有 `learned_core_field_readout.png` 的第一格
（沿轴一维剖面：学出来的细丝 vs 手放的两个核）**换成**一张能独立说明
**"在整张片子上，最优的是某个学出来的局部区域存在异质性"** 的二维图。
后三格（forward / reverse 事件图 + 虚拟 SEEG）沿用现有渲染函数，不变。

**新 A 面板的内容：**

- **底图**：Leg A 的扫描得分图（`S_rank`，跨 4 个种子平均），`imshow` 到片子坐标。
  `S_rank` 无定义的格点必须**显式画成另一种视觉**（灰色 + 图例注明"单方向，无定义"），
  **不得**留白让读者误以为是低分
- **等值线**：Leg B 三次重启各自最优场的 90% 质量等值区，三条不同线型 ⇒
  读者能一眼看出三次重启是否落在同一处
- **触点**：白三角，沿用现有渲染的记号
- **手放的两个核**：只画小的空心圈，图例必须写明 **"reference, not an input to this run"**。
  理由：上一轮的教训是"数据驱动的图上画手放核会让读者以为核还在那儿"；
  但这一格的问题恰恰是"自由最优落在哪"，把旧位置作为参照标出是有信息的——**前提是标签写死**
- **共享轴**：画成一条细虚线，图例写"frozen E→E anisotropy (the only prior kept)"

**必须同时输出的第二张图（不塞进主图）：** 扫描的另两层量
（`n_dir` 图 + `n_recruited` 图）并排，回答"哪里能出两个方向"和"哪里能招募到更多触点"。
按 CLAUDE.md §7，这两层与得分图问的是不同问题，不能叠进同一格。

- [ ] **Step 1**: 渲染 → **亲自目视** → 改 → 再渲染
- [ ] **Step 2**: 写 `figures/README.md`（中文，图看过之后写）
- [ ] **Step 3**: 提交

**⚠️ 表述纪律：** 图上不得出现 `S_rank` / `n_dir` / `coverage_tier` 这类内部字段名；
用读者语言（"template match"、"both directions readable"、"contacts recruited"）。

---

### Task 8: 收口报告与 handoff 更新

**Files:** append spec §9.9（Stage 3 实测收口）；update `docs/superpowers/handoffs/2026-08-07-topic4-data-driven-core-field.md`

- [ ] 按 §8.3 的同一格式写 Stage 3 收口：三个阶段结局、场的形状、门的逐条判定、
      失败归因的界定、允许/禁止的表述
- [ ] 更新 handoff 的"当前状态"与"下一步"
- [ ] **⚠️ 必查**：任何"最优在某处"的表述，都要写清是 Leg A（无优化的扫描）还是
      Leg B（优化）得到的。两条腿的证据强度不同，不得混为一谈

---

## 预算与排期

| 阶段 | 仿真次数 | 墙钟（10 并发） |
|---|---|---|
| Task 1–3（纯计算 + 测试） | 0 | ~1 h |
| Task 4 Leg A 扫描 | 392 | **~7 h** |
| Task 5 Leg B 优化（3 重启） | 450 | **~8 h** |
| Task 6 held-out（3 场 × 120 种子） | 360 | **~6 h** |
| Task 7–8 图与报告 | 0 | ~2 h |
| **合计** | **~1200** | **~24 h** |

实测吞吐：单次 8 s 仿真 7–25 分钟（视机器负载）；120 种子 10 并发约 1.5 h。
**若时间受限的降级顺序**（必须在报告里写明降了什么）：
Leg A 尺寸档 2→1（省 3.5 h）→ Leg B 重启 3→2（省 2.7 h）→ held-out 120→60 种子
（**最后才降这个**，因为 Stage 2 已证明池子小会让门无法评估）。

## 本计划刻意不做的事

- **不改打分支撑集**，即使 §10.3 第 2 条又不过（spec §9.1）
- **不把门的第 2 条放进目标函数**（spec §9.3）
- **不做 `K > 3`** —— 先看 `K=3` 够不够；`K` 是下一轮的自由度
- **不碰患者侧数据**：held-out 只在模型侧（spec §10.1 用户裁定）
- **不重跑已发表的 Fig4 系列产物**（k_dir 事实按用户裁定只记录不重跑，spec §2.4a）
