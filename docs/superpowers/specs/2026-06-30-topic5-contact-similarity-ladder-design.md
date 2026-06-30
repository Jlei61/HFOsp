# Topic5 触点相似性几何阶梯（contact-similarity ladder）设计稿

- 日期：2026-06-30
- 定位：Topic 5 A-line **sensitivity / robustness** 层，不开新的队列级主张
- 状态：设计稿，待 spec review → writing-plans
- 评审来源：合作者一轮 review（P0 几何空间一致性 + 三处 P1 + 软化科学语句 + 硬合同），已逐条核实采纳，见 §11 评审契约记录

---

## 1. 背景与动机（朴素话）

我们现在判断"发作时的强放电在空间上是不是仍然贴着间期的传播轴"，用的是一张**场图**：把每个电极触点的值铺到它的空间位置上、做高斯平滑、再比较两张平滑后的"空间形状"像不像（`field similarity` / `maxAB`）。

合作者提出一个质疑：这张场图**没有引入任何新测量**——它完全是"触点值 + 触点位置"的确定性函数，所以"投影成场"这一步可能没必要，直接在电极触点上比就行；甚至担心"像不像"是平滑/网格这些**额外假设**造出来的，而不是真实空间组织。

这个质疑在信息论意义上是对的（场是触点值的固定线性变换），但"没有新信息"不等于"对结论没有影响"——场用了触点间的几何（位置、距离、密度），而最朴素的逐触点相关不用。所以正确的回应不是二选一，而是**把"用多少几何"这个旋钮显式拆开，看结论在哪一档出现或消失**。

本设计建一个三档对照：同一批被试、同一组输入，只让"几何处理"变化，从而把场结论里"几何 / 平滑 / 网格"各自的贡献量化出来。

### 1.1 已存在的部分（不要重复造）

- **几何无关的逐触点相似性已经做过** = echo gate 那条线（`src/topic5_dynamic_echo.py`、`src/topic5_echo_gate.py`）：把间期排名和发作激活按通道配对算 Spearman，配了打乱对照（含 within-shaft）。**但它用的间期排名是 k=2 聚类模板排名，不是本设计要对齐的 `typical_rank`，且 cohort/口径不同**，不能直接拿来对质 field。
- 恰好配 `typical_rank` × `bb_auc` 的逐触点版本 = `along_axis_sign`（`src/topic5_axis_alignment.py:158`），但**只给正负号、没有 null**。
- **场（field）这一档已存在** = `run_topic5_axis_alignment.py` 的 maxAB 主统计。

### 1.2 真正的空缺

- **中间一档**："带几何但不铺网格"的触点度量——不存在。
- 与场**严格 like-for-like** 的几何无关逐触点相似性（同 `typical_rank` × `bb_auc`、配 within-shaft null、极性自由 maxAB）——不存在。

本设计补这两块，并把现有场作为只读的第三档。

---

## 2. 科学定位（tier 纪律）

- **sensitivity / robustness 层**：本分析检验"现有 A-line 场结论稳不稳 / 几何里哪一步在起作用"，**不产生新的队列级主张**，符合 Topic 5 当前探索性边界与 CLAUDE.md §5 tier 纪律。
- **sign-free / mirror-invariant**：A-line 主结论判的是**共享轴**，不判 forward/reverse 方向。因此所有 pass/fail 判据用极性自由的 `|sim|`；带符号值只作 sidecar（报 source 端热还是 sink 端热），**不进 pass/fail**。
- **R3 锁定为静态 A-line 场统计**（`run_topic5_axis_alignment.py` 的 maxAB），**不是**发作内随进程变化的 `field-dynamics` 轨迹。理由：合作者质疑的是"场投影是否必要"，对应静态单数口径最干净；发作内动态是另一条线。

---

## 3. 设计：几何阶梯

### 3.1 field-like ablation ladder（primary，三档只差几何处理）

所有三档喂**完全相同的输入**（§4），极性处理完全相同（A/B 各算一次取 `max(|sim_A|, |sim_B|)`），相关类型完全相同（**Pearson**），唯一变化是几何/平滑/网格：

| 档 | 算法 | 隔离出什么 |
|---|---|---|
| **R1 — 无几何** | `(typical_rank, bb_auc)` 在配对触点上 **unweighted Pearson**（位置完全不进计算） | 0 几何、0 平滑：逐触点早晚是否根本对得上 |
| **R2 — 同平面触点核** | 复用 R3 的 `x_norm/y_norm`（2D 平面 = 沿轴 × 横向）+ 同一 `sigma_xy` + 同一 `support`；核在**触点位置**评估（非网格）；触点上 Pearson | 同平面平滑、**无网格** |
| **R3 — field（只读）** | 现有：网格（81×81）平滑 + support-masked 像素 Pearson + mirror/abs maxAB | 同平面平滑 **+ 网格** |

**R2 与 R3 的唯一差别必须是"网格 vs 触点评估"**，其余一律复刻 R3，否则减法不干净。具体：
- **极性处理复刻 R3**：R3 经 `corr_pair_mirror_invariant` 做 y-mirror（横向 PCA 轴符号本就任意）取 max，再 abs，再 A/B 取 max。R2 也用同一套——对 field-2 用 y-翻转后的 `y_norm` 重算"触点核平滑值"、与恒等取 max，再 abs，再 A/B maxAB。R1 无位置，mirror 无意义，只有 abs + A/B maxAB。
- **support 只进核、不进最终相关**：R2 的最终 Pearson 在触点上**等权**（每触点一次），support 仅参与核加权（邻居权重）。R3 的像素 Pearson 隐含密度加权（密集触点区像素多=权重大）。**这个"像素密度加权 vs 触点等权"之差，正是网格效应本身**，不是 bug。

**两个减法的科学含义**（仅在 R2 与 R3 同平面/同 σ/同 support/同极性处理时成立）：
- `R2 − R1` = **同平面平滑**（空间混合邻近触点）的贡献；
- `R3 − R2` = **网格对密集触点区的密度再加权 + support-mask 像素门**的贡献——即"网格做的事"。不再被"3D vs 2D 平面"差污染。

### 3.2 sensitivities（非核心减法层，可单独解读，不进上面两个减法）

- **R2b — native-3D 触点核**：把核里的距离换成 `load_subject_coords` 的真实 3D 欧氏距离（而非平面投影距离）。`R2b − R2` = "2D 平面投影 vs native 3D"的影响。**v1 可先留接口、按需再跑**（YAGNI；3D 坐标加载器只服务这一档）。
- **σ 扫描**：R2 在 `σ ∈ {0.5, 1, 2} × sigma_xy`。回答"对齐判据是不是平滑带宽调出来的"。`σ=1×` 为主版（= R3 的 `sigma_xy`），0.5/2 为 sensitivity。

### 3.3 sequence-sanity track（并列的另一条输出，回答不同问题）

`(typical_rank, bb_auc)` 在配对触点上的 **Spearman + Kendall**，无几何、极性自由 maxAB、同 null。

- 这条**不参与 field ablation 阶梯**，它回答的是独立问题："**触点序列本身**像不像"（最初合作者讨论里的"电极序列相似性"）。
- 与 R1（Pearson）分开报告，避免"相关类型差"混进"几何差"的减法解读。

---

## 4. 输入合同（三档共享，严格 like-for-like）

- **间期侧**：两个模板的 `typical_rank`，来自 axis 记录 `{ds_sid}_t_a.json` / `{ds_sid}_t_b.json`（`run_topic5_axis_alignment.py` 消费的同一批 observation_readout real_subjects 记录）。
  - `typical_rank` 定义：`contact_aggregates`（`src/propagation_contact_plane_readout.py:71-126`），`typ_rank = nanmedian_e(masked)`，`masked = mask_phantom_ranks(normalize=True)`（已做 phantom 伪秩屏蔽，符合 Topic 0 合同）。
  - 极性自由：对 A、B 各算一次相似性，取 `max(|sim_A|, |sim_B|)`，对齐 field 的 maxAB。
  - `t_b` 缺失：退化为单模板 `|sim_A|`，**null 同步退化为单模板**（无 max-选择），口径一致。
- **发作侧**：`bb_auc` 为主（发作头 [0,10]s 宽带基线 robust-z 的逐通道均值，`src/topic5_t0_features.py:34` `activation_mean`；cache `results/topic5_ictal_recruitment/t0_feature_cache/{ds_sid}.npz` key `bb_auc__{idx}`）。`hfa_auc` 为辅（同 cache）。
  - 与 `run_topic5_axis_alignment.py` 的 `ict_vals` 完全同源，保证 R1/R2/R3 喂的是同一份发作向量。
- **通道对齐**：复用 `matched_channels`（`src/topic5_axis_alignment.py:25`，按名 join、缺失丢弃、绝不按下标对齐）；bipolar 名归一复用 `bipolar_alias_label`（`src/topic5_ictal_recruitment.py:384`）。
- **R2/R3 的 `x_norm/y_norm/support/sigma_xy`**：R2 不重新估计，直接取 R3 在 `load_context` 里冻结的同一套（`make_field_record` 复制 `x_norm/y_norm/support`；`sigma_xy` 取 A 模板场的值并对 B、对所有窗复用——见 `run_topic5_axis_alignment.py` 的 σ 冻结逻辑）。

---

## 5. Null 与统计合同

### 5.1 主 null：within-shaft 打乱

- 复用 `within_shaft_shuffle`（`src/topic5_axis_alignment.py:54`，按 `parse_shaft` 分组、组内置换、保多重集、绝不跨杆）。
- 辅助 null：`channel_shuffle`（全打乱，弱对照）、`anchor_matched_shuffle`（按激活分位置换，控激活幅度）。

### 5.2 maxAB selection 进入每个 null draw（防选择偏置）

**硬合同**：每个 shuffle draw 对**同一份打乱后的发作向量**同时算 `sim_A`、`sim_B`，再取 `max(|sim_A|, |sim_B|)`。null 分布是这个 MAX 统计量的分布。**绝不允许**"真实值取 maxAB、null 只算单模板"——那会系统性放大假阳。

### 5.3 判据（sign-free）

- 每被试每档：观测 `obs = max(|sim_A|, |sim_B|)`；`pass = obs > null 的 95 分位`（单侧、极性自由）。
- `signed corr`（A/B 各自的带符号 Pearson）只作 sidecar：报 source 端热 / sink 端热，**不进 pass/fail**。
- `effective_shuffle_n < MIN`（`effective_shuffle_n`，`src/topic5_axis_alignment.py:137`，数实际移动的通道）→ 标 `INSUFFICIENT_NULL`，**不静默判通过**。
- **保存完整 null 分位数**（至少 p5/p50/p95/p99 + 观测分位），不仅保存 p 值。

### 5.4 阶梯减法的统计

- `R3 − R2`、`R2 − R1`：报 **paired Δ 分布 + CI**（被试为单位，bootstrap 或符号检验 CI）。
- **不预设 arbitrary δ_grid 阈值**；gate 围绕 paired Δ 的 CI 是否排除"有意义效应"来陈述（见 §6）。

---

## 6. 验收 gate + 坏数据回归（承重主张编码成数值门）

### 6.1 队列级陈述（每条都绑定一个数值判据）

1. **几何无关已足够**：R1 通过的被试占比。高 → "对主结论而言 field 非必要条件"（措辞见 §9）。
2. **网格惰性**（field 非网格依赖）：`R3 − R2` 的 paired Δ CI 是否包含 0 / 排除有意义效应；且 R2 在这些被试上通过。CI 贴 0 → "网格没加东西，R2 复现 field"。
3. **平滑稳健**：R2 的 pass/fail 判据在 `σ ∈ {0.5,1,2}×` 三档下是否翻转。不翻 → "不是带宽调出来的"。
4. **field 依赖平滑/网格读出**（最伤情形）：R3 通过但 R2 不通过的被试 → 计数点名。措辞为"field 依赖平滑/网格读出"，**不写"假象"**（且因 R2 已与 R3 同平面/同 σ/同 support/同 null，此解释才成立）。

### 6.2 坏数据回归（TDD 必含；sign-free 下重新设计）

- **空间无关激活 → 应判不对齐**：把发作 `bb_auc` 在触点间随机打乱（或人为堆到正交离轴方向），三档 `|sim|` 应贴 null、不通过。
  - 注意：**不能**用"倒置 rank"作失败对照——sign-free 下倒置=同轴反向=真阳性，会照样通过。
- **单杆 / 退化 → `INSUFFICIENT_NULL`**：只有 1 根杆 → within-shaft null 退化 → 标 `INSUFFICIENT_NULL`，不静默通过。
- **σ→0 退化**：R2 核在 σ→0 时应退化回 R1（自权重独大）→ 数值上 R2(σ→0) ≈ R1。
- **σ→∞ 退化**：所有触点被抹成同值 → 相关无定义 → 判 degenerate 而非乱给数。

---

## 7. 预注册参数（实现前锁定，写进运行配置）

| 参数 | 含义 | 暂定值（实现时核定后锁） |
|---|---|---|
| `MIN_CH` | 最少配对触点数（否则不入队列） | 与 axis_alignment 同口径（核定） |
| `MIN_SHAFTS` | 最少电极杆数（within-shaft null 不退化） | 2 |
| `MIN_EFFECTIVE_SHUFFLE_N` | 实际移动通道数下限 | 核定（否则 `INSUFFICIENT_NULL`） |
| `B` | null 置换次数 | 2000 |
| `seed` | 固定随机种子 | 固定（写入产物 provenance） |
| `sigma_xy` | R2 主版核宽 | = R3 冻结值（不另估） |
| `sigma_sweep` | R2 sensitivity | {0.5, 1, 2} × `sigma_xy` |
| 相关类型 | ladder primary / sequence-sanity | Pearson / (Spearman + Kendall) |

`require t_a`（必须）；`t_b` 可选（缺则单模板，null 同步单模板）。

---

## 8. 模块 / 脚本 / 产物结构

### 8.1 新建

- `src/topic5_contact_similarity.py`（核心新原件，单一职责：触点层相似性 + 同平面/3D 触点核 + null）
  - `median_nn_spacing` 复用现有（R2 主版直接取 R3 σ，不需重算；3D 版仅 R2b 用，按需加 `_median_nn_spacing_3d`）
  - `kernel_smooth_at_contacts(values, pts, support, sigma)` — 仿 `smooth_field`（`src/propagation_contact_plane_readout.py:230`）的核数学，但评估点 = 触点（含自权重），返回每触点平滑值
  - `contact_similarity(rank, value, *, mode)` — `mode='raw'`（R1，unweighted Pearson）/ `mode='kernel'`（R2，同平面触点核后 Pearson）；统一返回带符号 corr
  - `polarity_free_maxab(rank_a, rank_b, value, *, mode, ...)` — 对 A/B 各算取 `max(|·|)`；`mode='kernel'` 复刻 R3 的 y-mirror（对 field-2 用翻转 `y_norm` 重算后取 max），`mode='raw'` 无 mirror
  - `within_shaft_null(rank_a, rank_b, value, names, *, mode, B, seed)` — 每 draw 重算 maxAB，返回完整 null 分位 + 观测分位 + `effective_shuffle_n`
  - `sequence_similarity(rank, value)` — Spearman + Kendall（sequence-sanity track）；可复用 `_spearman_on_intersection`（`src/ictal_er_rank.py:596`）的名键交集模式
- `scripts/run_topic5_contact_similarity.py` — 遍历队列：载 `t_a/t_b` + `bb_auc`（+`hfa_auc`）；算 R1/R2/R2b(opt)/R3(只读)/σ 扫描/sequence-sanity；写 per-subject JSON + `cohort_summary.{json,csv}`
- `scripts/plot_topic5_contact_similarity.py` — 三面板（§8.3）+ 中文 `figures/README.md`
- `tests/test_topic5_contact_similarity.py` — TDD（§10）

### 8.2 只读复用

- R3 场 maxAB：读 `run_topic5_axis_alignment.py` 既有 per-subject 输出（不重算）。
- 几何/平滑原件：`smooth_field` / `_support_corr` / `corr_pair_mirror_invariant` / `make_field_record` / `matched_channels`（位置见 §11）。
- 打乱件：`within_shaft_shuffle` / `anchor_matched_shuffle` / `channel_shuffle` / `effective_shuffle_n`。
- 坐标（仅 R2b）：`load_subject_coords` + `assert_coord_result_is_mm_for_main_analysis` + `parse_shaft`。

### 8.3 产物

- 目录：`results/topic5_ictal_recruitment/contact_similarity/`（与 `field_dynamics/` 同级，合规）。
  - `figures/`（含**中文** `figures/README.md`，图生成后写）
  - `per_subject/{ds_sid}.json`
  - `cohort_summary.{json,csv}`
- 图三面板（§7 多面板纪律：各答一个独立问题，无冗余）：
  - **A**：每被试三档 `|sim|` 柱 + within-shaft null 95 带 → "每被试三档一致吗"
  - **B**：几何阶梯 slopegraph，R1→R2→R3 每被试一条线 → "平滑步、网格步各动多少"（一张图同时显两个减法，避免拆成两张冗余 scatter）
  - **C**：σ 扫描，R2 `|sim|` vs σ 每被试一条 → "带宽稳健吗"
  - sequence-sanity（Spearman/Kendall）另出小表或附panel，不挤进 ablation 三面板。
- `FIGURE_INDEX.md` append（新结论图目录约定）。

---

## 9. 科学语句模板（允许 / 禁止措辞）

**允许**：
- "把'用多少几何'拆成三档后，结论在 R_x 这一档出现/消失。"
- "对主结论而言，field 不是必要条件——R1（无几何）已能看到对齐；但 field 仍是**形态展示**与**不规则空间 support 读出**的工具。"
- "网格这一步对结论无贡献（`R3−R2` 的 CI 贴 0）。"
- "field 依赖平滑/网格读出（R3 过、R2 不过）。"

**禁止**：
- ✗ "`R3−R2` = 纯网格贡献"——除非明确 R2 与 R3 同 2D 平面 + 同 σ + 同 support（本设计已满足，但措辞需带这个前提）。
- ✗ "R1 过 ⇒ field 没用 / 是假象"——应说"field 非必要条件，但仍有形态/support 读出价值"。
- ✗ "R3 过 R2 不过 ⇒ 假象"——应说"依赖平滑/网格读出"，且须先确认 R2 坐标系/support/σ/null 与 R3 同构（本设计构造上同构）。
- ✗ 任何 forward/reverse 方向主张——本分析 sign-free，方向只在 sidecar 描述。
- ✗ 任何队列级新发现主张——本分析是 sensitivity 层。

---

## 10. TDD 测试清单（实现先写测试）

1. `kernel_smooth_at_contacts`：3 触点共线已知算例 → 校验权重 = `support·exp(-d²/2σ²)` 归一。
2. σ→0：R2 核退化回 R1（自权重独大）→ `R2(σ→0) ≈ R1`。
3. σ→∞：所有值抹平 → 相关无定义 → 判 degenerate（不乱给数）。
4. 极性自由：构造 `sim_B > sim_A` 算例 → maxAB 取到 B。
5. null selection：每 draw 重算 A/B 取 max → 与"只算单模板"的 null 对比，确认前者 null 上尾更高（防选择偏置）。
6. within-shaft 打乱：保多重集、绝不跨杆（断言 shaft 标签集合不变）。
7. 坏数据：空间打乱发作向量 → 三档 `|sim|` 不显著（贴 null）。
8. 退化：单杆 → `INSUFFICIENT_NULL`；缺 `t_b` → 单模板路径且 null 同步单模板。
9. like-for-like：R1/R2/R3 喂的 `matched` 通道集合一致（断言三档同一 channel 列表）。
10. provenance：产物含 `seed`、`B`、`sigma_xy`、输入文件路径、loader 版本。

---

## 11. 复用 vs 新建清单（精确 file:line）

**复用（只读 / 调用）**
- `src/propagation_contact_plane_readout.py`：`smooth_field` L230、`_support_corr` L273、`corr_pair_mirror_invariant` L285、`build_readout_record`（`x_norm/y_norm` L178-179）、`contact_aggregates`（`typical_rank` L117）、常量 `GRID_N`/`S_THRESH`/`OVERLAP_MIN` L14/17/18
- `src/topic5_axis_alignment.py`：`make_field_record` L31、`matched_channels` L25、`along_axis_sign` L158（参考其 corr 口径）、`within_shaft_shuffle` L54、`anchor_matched_shuffle` L70、`channel_shuffle` L48、`effective_shuffle_n` L137
- `scripts/run_topic5_axis_alignment.py`：场 maxAB 主统计 L97-127、`along_axis_sign` 调用 L148、σ 冻结逻辑（R2 取同 σ）
- `src/topic5_t0_features.py`：`activation_mean` L34（`bb_auc` 定义）；cache `results/topic5_ictal_recruitment/t0_feature_cache/{ds_sid}.npz`
- `src/ictal_er_rank.py`：`_spearman_on_intersection` L596（sequence-sanity 名键交集模式）
- `src/propagation_skeleton_geometry.py`：`parse_shaft` L21
- 仅 R2b：`src/seeg_coord_loader.py`：`load_subject_coords` L416、`assert_coord_result_is_mm_for_main_analysis` L882

**新建**
- `src/topic5_contact_similarity.py`（§8.1）：`kernel_smooth_at_contacts`、`contact_similarity`、`polarity_free_maxab`、`within_shaft_null`、`sequence_similarity`（+ R2b 用 `_median_nn_spacing_3d` 与 3D 距离矩阵 `cdist`，按需）
- `scripts/run_topic5_contact_similarity.py`、`scripts/plot_topic5_contact_similarity.py`、`tests/test_topic5_contact_similarity.py`

### 评审契约记录（2026-06-30 一轮 review，核实后采纳）
- **P0**：R2 用 native-3D 会使 `R3−R2` 混入 2D-plane-vs-3D 差。核实属实（field 跑在投影后的 `x_norm/y_norm` 2D 平面，`build_readout_record` L178-179）。改：R2 主版同平面（复用 R3 `x_norm/y_norm/sigma_xy/support`），native-3D 降 R2b sensitivity。
- **P1-a**：gate 要求 `signed>0` 违反 A-line sign-free 合同（field maxAB mirror+abs 极性自由）。改：primary = `max|sim|` vs null95；signed 降 sidecar。**连带**：坏数据回归改用"空间无关激活"而非"倒置 rank"（倒置在 sign-free 下是真阳性）。
- **P1-b**：null 必须每 draw 重算 maxAB（防 max-选择偏置）。锁为 surrogate 合同。
- **P1-c**：指标族锁死。拆两条独立输出：field-like Pearson ladder（专答 ablation）vs Spearman/Kendall sequence sanity（专答序列像不像）。
- **软化**：见 §9 措辞模板（"field 非必要但仍有形态/support 价值"；"依赖平滑/网格读出"非"假象"；`R3−R2=网格`须带同构前提）。
- **硬合同**：见 §5/§7（固定 B/seed、保存完整 null 分位、丢 arbitrary δ_grid 改报 paired Δ + CI、eligibility 门、R2 复用 field support、R1 用 unweighted raw 不混解释）。
