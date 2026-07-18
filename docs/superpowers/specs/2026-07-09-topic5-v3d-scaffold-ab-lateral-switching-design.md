# Topic5 V3d — 间期 scaffold 上的 A/B 侧向选择/切换（设计 spec）

> **状态（2026-07-09，rev2 — 首轮 review 10 条 + 二轮 2 条统计 patch（P2a per-seizure 枚举时间 null、P2b 闭式仅
> full-J 等价）全部并入，进 writing-plans）**：这是 V3p（preictal-only 非轴向轨迹，完整硬门阴性）的
> 重构继任线，也是 field_dynamics_signed peri-onset signed 读出的正式化。**不改 V3p 的旧结论**（"发作前
> 没有稳定的轴向→离轴单调重组"仍然成立且有价值），而是换一个更贴合数据的问题：能量多数时候仍落在间期
> scaffold 上，部分患者在 scaffold 的 A/B 两个传播态之间**侧向选择或切换**进入早期发作。
>
> **预注册纪律（两层，锁死，不许事后升级）**：
> - **已支持层（restate，非新主张）**：发作前后宽带能量常落在间期 scaffold 上（maxAB |r| 高）。
> - **待检验新假设层（本 spec 的 primary）**：部分患者在 scaffold 的 A/B 侧向态之间切换/择一进入早期发作。
> - **禁**：把它写成已经证明的 "preictal switching"。必须先让 `C_AB` + 状态空间图 + 时间 null 把这个假设坐实。
>
> 前序讨论：本 spec 由 2026-07-09 会话的四轮 brainstorm + 用户 review 收敛而来。审阅结论完成度 80/100，
> 四个必须锁死的点（`C_AB` 定义 / 镜像取向 / 幅度门控 / 时间 null）已全部并入下方 §2/§5/§6。

---

## 0. 一句话承诺（大白话，CLAUDE.md §8）

我们看：**这个病人平时（间期）高频事件在致痫区里反复走的那条传播路，其实有两个方向相反的版本——把它叫 A 走法和
B 走法。发作快开始的时候，这块组织的宽带能量（哪儿更亮）到底是贴着 A 走法那一头、还是 B 走法那一头？还是在两头
之间来回换？**

怎么测：我们不去问"能量有没有越来越强"，也不去问"能量是不是跑到路外面去了"（那是旧问题 V3p，答案是没有）。我们
问的是——把发作前后每一小段时间的能量分布图，拿去和"A 走法减 B 走法"这张差值图做带符号相关（记为 `C_AB`）：
偏 A 头 `C_AB` 就往 +1 走，偏 B 头就往 −1 走，两头没偏好就在 0 附近。**如果这纯粹是随机的时间巧合，把每次发作的
这条曲线在时间轴上整体平移打乱，临发作那段的"锁定到某一头"应该和实测一样强；实测如果临发作前那段明显比打乱后
更锁定，就说明是真的在往某一头收敛。**

揭示什么（预期，尚未验证）：在这个尺度上，能量大多数时候确实在这条路上（不是从无到有慢慢爬），但**一部分病人**
看起来是在 A/B 两头之间**换方向或选定一头**进入早期发作，而不是逐渐离开这条路。这只是一个**待检验的假设**，要过
了下面的幅度门 + 时间 null + 队列计数才能算数。

（内部归档代号：V3d, `C_AB`, `D_AB`, maxAB, axis_present, within_shaft_shuffle null, circular-shift 时间 null,
forward_reverse_reproduced 子集；前序 V3p / field_dynamics_signed / Fig3-B。）

---

## 1. 为什么做这个 / 和旧 V3p 的关系

### 1.1 旧 V3p 问错了问题（改写，不推翻）

- **旧 V3p 测的**：只看发作前两分钟，非轴向触点能量是否逐渐爬升、能量场是否从轴向走廊转向 off-axis（离轴单调重组）。
- **旧 V3p 结论**：n_perm=1000 完整硬门阴性（narrow 0/7、broad 0/13、0/9）。**这个阴性有效且有价值**——它证伪了
  "发作前逐渐离轴重组"。
- **改写口径（写进 topic5 主文档时用）**：发作前**没有**稳定的"轴向→非轴向单调重组"；相反，能量场多数时候**仍在**
  间期 scaffold 上，部分患者表现为沿 A/B scaffold 的**侧向选择或切换**。

### 1.2 数据为什么支持换这个问题

- Fig3-B 材料池（`fig3_peri_onset_subject_index.json`，20 个 ok 被试）：maxAB 的 |r| 中位数几乎都在 **0.6–0.9**、
  窗内方差小。说明能量不是"从无结构慢慢爬上来"，而是**一开始就在 scaffold 上**——旧 V3p 的"逐渐爬升"框架本就
  不该套在这。
- E548 peri-onset 图：发作前 template-B 相似度剧烈摆动、临 onset 前 ~40s 收敛到高位。这是"在已有 scaffold 上改变
  侧向态"，不是离轴。
- 现有 signed A/B 读出把 `r_A`、`r_B` 画成两条线；对反相关模板这两条线互为镜像（冗余）。**`C_AB` 是那一个不重复
  的量**（CLAUDE.md §7 多面板去冗余）。

### 1.3 命门（为什么现在的 signed 读出不能直接用）

现有 `corr_pair_mirror_invariant_signed`（`compute_topic5_signed_broadband_similarity._scorer` →
`maxab_batch`）对 A、B **各自、逐窗**按 |corr| 最大在 identity/mirror 两个取向里选（代码见
`run_topic5_fig3b_maxab_spatial_null.build_engine` 的 `idx_id`/`idx_mir` + `maxab_batch` 的 `np.nanmax`）。
这让 `r_A − r_B` 的符号在 |r|≈0 附近漂、会随时间翻面，制造**假切换**。**V3d 主分析必须换掉这套镜像选择**（§2.2）。

---

## 2. 核心量定义（锁死）

### 2.1 joint contacts 与固定的对比轴 `D_AB`

- 模板来源：`REAL_DIR/{ds_sid}_t_a.json`、`{ds_sid}_t_b.json`
  （`REAL_DIR = results/spatial_modulation/propagation_geometry/observation_readout/real_subjects`），
  每个 channel 带 `name` / `typical_rank` / `x_norm` / `y_norm` / `support`。
- 触点匹配复用 `src.topic5_axis_alignment.matched_channels(axis_a, raw_index)`（对齐到 EDF 的 bipolar 别名）。
- **subject-fixed joint set `J`**（一次算定，不随窗变）：满足
  1. `typical_rank_A` 有限，
  2. `typical_rank_B` 有限（按 name 对到 A 的匹配触点），
  3. 该触点在 ≥ `F_MIN_WIN = 0.9` 比例的时间窗里 ictal z 有限。
- **门槛**：`|J| >= N_JOINT_MIN = 6`（与现有 `len(matched) < 6` gate 一致）。`|J| < 6` → 该被试
  `insufficient_joint`，退出 H1/H2，只进 provenance 表。
- **rank 极性锁 + 标准化后再差分（P1，锁死；已核对代码）**：本仓库 `typical_rank` **低 = 早 = 源**
  （`src/topic5_field_extrapolation.py:3` "低=早=源"、`src/sef_hfo_subject_placement.py:61` "low = early"）。所以
  **必须先转成 earlyness 分数（大 = 早 = source-like）再差分**，否则 `C_AB` 的正负会被 rank 方向翻反：
  ```
  eA = -zscore(typical_rank_A over J)     # 大 = A 中越早 / 越 source-like
  eB = -zscore(typical_rank_B over J)     # 大 = B 中越早 / 越 source-like
  D_AB   = eA - eB                        # 固定的 A↔B 对比轴，一次算定
  rho_AB = pearson(eA, eB)                # = pearson(zA,zB)，符号翻转不改；模板对几何（§4 分层 + 退化守卫）
  ```
  于是 **`C_AB > 0` ⟺ 能量偏向"在 A 中更早、在 B 中更晚"的触点（A 源侧）**，`C_AB < 0` 偏 B 源侧。
  **唯一 source of truth = 直接 `C_AB(t) = corr(E_t, D_AB)`**（把固定的 `D_AB` 子集到该窗有限触点再算 Pearson）。
  等价闭式 `C_AB = (r_A − r_B) / sqrt(2·(1 − rho_AB))`（`r_A = corr(E, eA)`、`r_B = corr(E, eB)`）**只在 full-J
  全有限窗严格成立（P2b）**：若某窗只覆盖 J 的子集、而 `rho_AB` 用全 J，闭式有小偏差。所以闭式只作**解释等价**，
  实现和测试都以 direct `corr(E_t, D_AB)` 为准，**不要求缺触点窗满足闭式**。
  **不用 raw `A − B`**：Pearson 只在最后标准化 `D_AB`，不修正 A/B 差分前的相对尺度；earlyness 标准化后差分才对应上式。

### 2.2 逐窗对比量 `C_AB(t)` 与幅度 `maxAB(t)`（无镜像，固定取向）

对每个 seizure 的每个时间窗 `t`，取该窗 per-contact baseline robust-z 能量向量 `E_t`（来自
`compute_topic5_signed_broadband_similarity._compute_values` 的 `window_vals`，对齐到 `matched`）。在
`J ∩ {该窗 z 有限}` 上算**普通带符号 Pearson**（不做任何 y-flip / 镜像选择）：

```
C_AB(t)  = pearson(E_t, D_AB)                       # ∈ [-1, 1]；+ 偏 A 源侧，− 偏 B 源侧（§2.1 earlyness）
r_A(t)   = pearson(E_t, eA)                          # 用 earlyness eA，与 §2.1 闭式一致
r_B(t)   = pearson(E_t, eB)
maxAB(t) = max(|r_A(t)|, |r_B(t)|)                  # 在不在 scaffold 上（幅度，符号无关）
side01(t)= 0.5 + 0.5 * clip(C_AB(t), -1, 1)         # 仅上色用；统计一律用居中的 C_AB
```

- **主分析 = raw contact space**（上式，符号最硬）。
- **sensitivity = 平滑 field 版**：复用 `R_smooth_rank` + `make_field_record` 把 `eA`、`eB`、`E_t` 各自渲成场，
  **但 `D_AB` 场必须先固定 A/B 取向再相减**（`D_field = field(eA) − field(eB)`，一次算定），再算
  `C_AB_field(t) = pearson(E_field_t, D_field)`。**禁**让 A 和 B 分别选镜像。场版只作稳健性对照，不作主量。

### 2.3 退化守卫（P1）

当两个模板几乎相同（`rho_AB → +1`）时 `D_AB → 0`、`sd(D_AB) → 0`、Pearson 变 NaN。锁（分层完整定义见 §4 轴一，
这里只锁硬退出线）：

- `rho_AB >= RHO_DEGEN = +0.85` → 该被试标 `hard_degenerate`，**退出 H1/H2、不产出 `C_AB`**，只在 provenance 表记录。
  （`+0.5 <= rho_AB < +0.85` 的 `aligned` 层**不退出**但降级，见 §4；不要把 `aligned` 和 `hard_degenerate` 混叫
  "degenerate"。）
- 实现层另加：若某窗 `sd(E_t on J) < 1e-9`（能量全平） → 该窗 `C_AB = NaN`、`axis_present = False`。

### 2.4 幅度门控 axis_present(t)（P1）

`C_AB` 的符号只有在"能量确实落在 scaffold 上"时才有解释。**统计一律只在 axis_present 窗里做**（主图可画全时程）。

- `axis_present(t) = ( maxAB(t) 过该窗 within-shaft-shuffle null，pointwise 单边 p < ALPHA_PRESENT )`，
  `ALPHA_PRESENT = 0.05`，`n_perm = 1000`。
- null 构造复用 `src.topic5_axis_alignment.within_shaft_shuffle(vals, names, rng)` + `parse_shaft`：只在每根杆内打乱
  per-contact 能量值，保留"哪根杆热"的植入几何（与 `run_topic5_fig3b_maxab_spatial_null` 的 primary null 同族，但
  这里算在 **raw-contact maxAB** 上，与 `C_AB` 同表示，自洽）。
- **禁**硬用统一 0.5 当统计门。仅在 within-shaft null 不可测时（见下）允许 `maxAB > 0.5` 作**描述性**回退，并显式标注。
- **可测性前置**：需要 ≥ `N_MULTI_SHAFT_MIN = 2` 根"非单触点杆"（`shaft_size ≥ 2`），否则 within-shaft null 无自由度
  → 该被试 `axis_present` 不可判 → 归 §4 coarse-only tier，H1 不下结论。
- **单触点杆自由度（P7）**：within-shaft shuffle 里单触点杆不动、却仍进 maxAB 相关，会让 null 偏乐观/保守。
  **obs 和 null 一律算在同一批 joint 触点上**（不改观测统计量，避免 obs/null 触点集不一致），但落 QC：
  `n_contacts_shuffled`（在多触点杆上的触点数）、`fraction_contacts_shuffled = n_contacts_shuffled / n_joint`、
  `n_singleton_contacts`。**`fraction_contacts_shuffled < 0.6` → 标 `axis_present_low_dof`，该被试降级 coarse-only、
  H1 不下结论**（within-shaft null 此时几乎贴着观测，是诚实的弱自由度，不是通过）。

---

## 3. 预注册假设（tier 锁死）

三段式，数字判据 lock at spec time，不许事后调。

### H0 — 能量落在间期 scaffold 上（已支持层的 restate，非新主张）
- **测**：peri-onset 每窗 `maxAB(t)`。
- **判**：cohort 层面 maxAB 高（材料池中位 0.6–0.9）；**细 scaffold**（过 within-shaft null）只在过 gate 的被试成立，
  参考 `fig3b_maxab_spatial_null_index`（R2 现约 5/18）；**粗 scaffold**（过 all-contact / 杆级）更广。
- **报告语言**：这是"前提条件已具备"的事实陈述，不是 V3d 的核心证据。不得写成"发作早期特异招募"（§ topic5 §3.0 红线）。

### H1 — scaffold 上的 A/B 近-onset **侧向极化/选择**（**primary，待检验新假设**）
- **测**：在 axis_present 窗上，`C_AB(t)` 临 onset 是否比 far-pre 更偏向某一侧（侧向极化），且超过被试内时间 null。
- **判**：per-seizure locking 统计（§6.2）> circular-shift 时间 null 95 分位；subject-level 合并 + 队列二项计数（§6.3）。
- **⚠️ 用词纪律（P3）**：`locking = |mean_near| − |mean_far|` 证明的是"近-onset 比 far-pre 更极化到某一侧"，
  = **侧向极化 / side selection / locking**，**不等于"切换"**。举反例：far≈0→near+0.7 是"选择"非切换；far+0.5→near+0.6
  是"持续偏 A"非动态；near 内部先 +0.8 后 −0.8 时 `|mean|` 反而低、H1 不显著。所以 H1 只允许写"侧向极化/选择"，
  **禁**写"已证明 preictal switching / 模式切换"。
- **tier**：假设检验；允许写"在 k/m 个可检验病人上，发作临近出现超过时间 null 的侧向极化"。

### H2 — 侧向态的选择 vs 切换 vs 持续（**secondary，描述性 per-subject，非 cohort claim**）
- **测**：把 far-pre → near-onset 的侧向态显式分类（不与 H1 的"极化"混用）：
  ```
  far_side  = sign(mean_{far, axis_present} C_AB)  if |mean_far|  >= DELTA_SIDE else none
  near_side = sign(mean_{near,axis_present} C_AB)  if |mean_near| >= DELTA_SIDE else none
  selection_event  = far_side==none AND near_side labeled          # 从无侧向到选一侧
  switch_event     = far_side and near_side both labeled AND !=    # A↔B 真切换
  persistent_event = far_side == near_side != none                 # 一直偏一侧
  ```
- **判**：per-seizure 落上述四类（unlabeled→A/B selection / A→B / B→A switch / A→A / B→B persistent）+ side_probability
  + 状态转移矩阵（§7 表2/3），**只描述、不做 cohort 显著性**。
- **tier**：mechanism/descriptive。**禁**把"多数病人都切换/都单侧"写成 cohort 定量主张。

---

## 4. 被试分层（两条正交轴，都要落表）

一个被试同时带两个标签，报告时不得混为一谈（CLAUDE.md §6.3 pronoun 纪律）。

### 轴一：模板对几何（`rho_AB` on J）
- `reciprocal`（`rho_AB <= -0.5`）：`C_AB` 的正负有清楚物理含义 = 偏间期传播的**源端 vs 汇端**；可进一步连
  Topic-1 的 `forward_reverse_reproduced` 子集（约 8/9 或 13/14）。**A/B 双稳态叙事只在这一层允许。**
- `oblique`（`-0.5 < rho_AB < +0.5`）：A/B 非反相、两个不同模式，`C_AB` = 更像 A 还是 B，可讲 contrast、**不强讲互逆 source-sink**。
- `aligned`（`+0.5 <= rho_AB < +0.85`）：A/B 大量共享，只能谨慎讲"A/B 有差异分量"、**不适合双稳态叙事**（`D_AB` 变异小、`C_AB` 噪声大）；**不退出**但结论降一档。
- `hard_degenerate`（`rho_AB >= +0.85`）：`D_AB → 0`，**退出 H1/H2、不产出 `C_AB`**（§2.3）。

### 轴二：scaffold 精细度（within-shaft null）
- `fine`（过 within-shaft null）：细模板特异，`C_AB` 侧向可解释到轴内位置。
- `coarse-only`（maxAB 高但不过 within-shaft null，或 null 不可测）：只能讲"粗几何 scaffold / 哪根杆热"，
  **不得讲细模板特异**；这类被试 H1 侧向可作描述，但结论降一档。

### 队列计数规则（不 pool seizure、只在 subject 层做，沿用 topic5 tier 纪律）
报四个数：(a) `reciprocal` 被试数；(b) `axis_present` 可测且过 gate 的被试数；(c) H1 极化被试数；
(d) 其中 selection / switch / persistent 各几人（H2 描述）。
- **cohort 轻量判据（P6，不 pool seizure）**：设 `m` = H1-primary-eligible 被试数、`k` = subject-locked 被试数，
  做 **subject-count 单边精确二项检验** `Binom(k; m, p0=0.05)`（`p0` = 单被试 H1 的假阳率上界，因 subject_locked
  用 `p<0.05`）+ 报 exact CI。这是**被试计数检验**，不是把 seizure 当独立样本 pool，与"不 pool"原则不冲突。
- **abstract 语气随 k/m 走**：`k/m` 二项不显著 → 只能写"在部分可检验患者中观察到侧向极化"；显著且 `k/m` 高 →
  才可写"多数可检验患者"。**结果没跑完前禁写"均显著对齐"**（见 §11）。

---

## 5. 两个 null（互不替代）

| null | 问什么 | 怎么做 | 复用 |
|---|---|---|---|
| **空间 within-shaft**（gate 用） | 能量在不在细 scaffold 上（axis_present） | **每个 seizure × window 独立**：在杆内打乱 per-contact 能量值、重算该窗 maxAB，pointwise 单边 p → 该 seizure 该窗 `axis_present`。**不跨 seizure 取中位**（否则 per-seizure locking 无法定义）；跨 seizure 只在后面的 locking null（§6.3）聚合 | `within_shaft_shuffle` / `parse_shaft`（借 `run_topic5_fig3b_maxab_spatial_null` 的 shuffle/readout，但 fig3b 的 seizure-median 是 cohort 材料读出、V3d 不用） |
| **时间 circular-shift**（H1 用） | 近-onset 侧向极化是不是时间巧合 | 每 seizure 把**整张逐窗 state 表**环移同一个非零偏移，重算 locking | 新写，`src.topic5_scaffold_ab_contrast.circular_shift_null` |

- 环移是**保结构**的：破坏"和 onset 对齐"，但保住曲线自身的自相关和幅度分布——检验的是"临 onset 那段极化是否超过
  随机时间对齐"，不是"曲线有没有结构"。
- **必须环移整张 state 表，不能只移 `C_AB`（P2）**：locking 在 `axis_present` 窗上算，而 `axis_present`（∝ maxAB）与
  `|C_AB|`（reciprocal 层）高度相关、本身也带 onset 对齐信息。若只移 `C_AB`、把 `axis_present` mask 钉在真实 onset
  附近，null 就没打破 onset 对齐、不干净。所以对每个 seizure 的
  `[C_AB, r_A, r_B, maxAB, axis_present, side_label, within_shaft_p]` 施加**同一个**非零环移，再用移位后的 `C_AB` +
  移位后的 `axis_present` 重算 `polar_near/polar_far/locking`。
- **per-seizure = 枚举，不是抽样（P2a）**：一条 seizure 只有 `T−1 ≈ 65` 个唯一非零 shift，抽 1000 次只是重复抽这
  几十个、不产生 1000 个独立时间对齐。所以 per-seizure 层**穷举所有非零 shift**（`shift ∈ {1,…,T−1}`，禁 0-shift；
  移位后 near 或 far 的 axis_present 窗 `< 3` 的 shift 记 invalid），得**精确枚举** `locking_shift_p =
  (1 + #{valid shift: locking(shift) ≥ locking_obs}) / (n_valid_shift + 1)`。要求 `n_valid_shift ≥
  N_VALID_SHIFT_MIN = 40`，否则该 seizure 退出。**per-seizure `locking_shift_p` 只作描述**（精度 ~1/n_valid_shift），
  **不承担过精细 p**。
- **subject-level 才做 1000 次抽样**：H1 verdict 在 subject 层（§6.3）——每次 permutation 对每个 valid seizure
  **各独立抽一个 valid shift** 再取 median，联合空间 ≥ `40^{n_seizures}` 足够大，`N_PERM = 1000` 在这里才成立。
- 时间维多重比较：若要给"显著窗区间"，复用 `run_topic5_fig3b_maxab_spatial_null` 的 maxT / cluster 校正
  （Nichols-Holmes / Maris-Oostenveld），不新造。H1 的 subject-level 判据用 §6.3 的单点合并，不依赖逐窗校正。

---

## 6. 验收 gate（数字锁死 + 坏数据回归）

> 依 `MEMORY.md → feedback_acceptance_gate_encode_conclusion`：每个承重定性主张 → 数值阈值 + 坏数据回归；参数在
> spec 正文锁定。

### 6.1 锁定参数表（实现不得改，改需回本 spec）

| 参数 | 值 | 含义 |
|---|---|---|
| `START_SEC, STOP_SEC` | −120, +20 | peri-onset 窗（复用 Fig3-B locked 合同） |
| `WINDOW_SEC, STEP_SEC` | 10, 2 | 窗长 / 步长（复用 Fig3-B） |
| `BAND` | (1.0, 150.0) | 宽带 |
| `F_MIN_WIN` | 0.9 | 进 J 需要的窗内 z 有限比例 |
| `N_JOINT_MIN` | 6 | joint 触点下限 |
| `RHO_DEGEN` | +0.85 | 模板退化硬退出阈 |
| `DELTA_SIDE` | 0.2 | side 标注幅度地板（`|C_AB| < 0.2` → unlabeled） |
| `ALPHA_PRESENT` | 0.05 | axis_present pointwise p 阈 |
| `N_MULTI_SHAFT_MIN` | 2 | within-shaft null 可测所需非单触点杆数 |
| `FAR_PRE_SEC` | [−120, −60] | far-pre 窗（locking 基线） |
| `NEAR_ONSET_SEC` | [−30, +10] | near-onset 窗（H1 primary locking 目标） |
| `NEAR_PRE_SEC` | [−30, 0) | 描述窗：只发作前（表3/图分报，P10） |
| `EARLY_ICTAL_SEC` | [0, +10] | 描述窗：只发作早期（表3/图分报，P10） |
| `N_PERM` | 1000 | 空间 null 置换数 + **subject-level** 时间 null 组合抽样数（非 per-seizure，P2a） |
| `N_VALID_SHIFT_MIN` | 40 | 单 seizure 时间 null 穷举后需要的有效唯一非零 shift 下限（P2a；per-seizure 是枚举非抽样） |
| `N_VALID_SEIZURE_MIN` | 3 | subject 进 H1 需要的有效 seizure 数（P4；对齐 abstract "≥3 发作"） |
| `LOCK_ALPHA` | 0.05 | H1 单边显著阈 |

### 6.2 locking 统计（H1 承重量，锁死）

对每个 seizure：
```
polar_near = | mean_{t in NEAR_ONSET, axis_present} C_AB(t) |     # 近-onset 净侧向强度（符号无关，跨发作不抵消）
polar_far  = | mean_{t in FAR_PRE,   axis_present} C_AB(t) |
locking    = polar_near - polar_far                              # >0 = 向 onset 侧向收敛
```
- 用 `|mean|`（而非 mean）→ 一次发作偏 A、另一次偏 B **不会互相抵消**（用户明确点名的坑）。
- 用 `near − far`（差值）→ **静态高 `C_AB`**（一直偏一侧、不随时间变）`locking ≈ 0`，正确判为"非动态收敛"；只有
  **临 onset 才收敛**的才 `locking > 0`。
- 每侧窗数 < 3（axis_present 后）→ 该 seizure locking = NaN，退出。

### 6.3 H1 subject-level 判据 + 队列（锁死）

- **subject 进 H1 的前置门（P4，全满足才算 H1-primary-eligible，否则标 `insufficient_valid_seizures` 退出）**：
  ① `n_joint >= 6`；② `rho_AB < 0.85`（非 `hard_degenerate`）；③ `axis_present_testable`（≥2 非单触点杆 且
  `fraction_contacts_shuffled >= 0.6`，§2.4）；④ **≥ `N_VALID_SEIZURE_MIN = 3` 次 seizure 同时满足**：far-pre
  axis_present 窗 ≥3、near-onset axis_present 窗 ≥3、时间 null 穷举后有效唯一 shift ≥ `N_VALID_SHIFT_MIN = 40`。
- per-seizure：**穷举**所有 valid 非零 shift（§5 P2a），得精确枚举 `locking_shift_p`（描述用，精度 ~1/n_valid_shift）。
- **subject 统计（H1 verdict 在这层）**：`L_obs = median_{valid seizure} locking`。subject null：重复 `N_PERM = 1000`
  次，每次对每个 valid seizure **各独立从其 valid-shift 集抽一个** shift、整表移位后重算 locking，取 median → `L_null`
  （联合空间 ≥ `40^{n_seizures}`，1000 次抽样在此才成立）。
- **subject 锁定** ⟺ `L_obs > percentile(L_null, 95)`（单边，`LOCK_ALPHA`）。
- **队列报告**：`k` 锁定 / `m` H1-eligible，按 §4 两轴分层列出 + §4 的 subject-count 二项检验（不 pool seizure）。

### 6.4 坏数据回归（synthetic fixtures，必须先写先失败）

| fixture | 构造 | 必须的行为 |
|---|---|---|
| `flat_noise` | 每窗每触点独立高斯能量 | axis_present 绝大多数 False；H1 不锁定（无 scaffold 更谈不上侧向） |
| `static_on_axis` | 每窗 `E_t = 3*eA`（强、恒定、偏 A 源侧；用 earlyness `eA` 保证 `C_AB>0`=偏 A） | `C_AB≈+高` 且恒定、axis_present True，但 `locking≈0`、**H1 不显著**（整表环移对恒定曲线不变）——证明"在 scaffold 上但静态偏一侧"被正确判为非极化（H2 会记 `persistent`，非 `switch`/`selection`） |
| `ramp_to_onset` | `C_AB` 目标从 0 线性升到 +0.8 进 onset（正对照） | `locking>0` 且过环移 null → H1 显著 |
| `degenerate_AB` | `zB = zA`（`rho_AB=1`） | 标 `hard_degenerate`、退出 H1/H2、**不产出 `C_AB` 数值** |
| `mirror_invariance_gone` | 把触点坐标 y 翻转 | raw-contact `C_AB` **数值不变**（不依赖坐标）；且与旧 mirror-abs-max 值**不必相等**——固化"镜像漂移已移除" |

---

## 7. 输出三张表（不只出图；用户明确要求）

目录：`results/topic5_ictal_recruitment/scaffold_ab_switching/`
（`per_subject/` + `figures/` + `cohort_summary.json` + `figures/README.md` 中文）。跨 seizure 的 median `C_AB`
**只能看趋势、不能独自证切换**（相反侧发作互相抵消）——这句话进 README 与 summary 的 `caveats`。

### 表1 per-window continuous（`per_subject/<ds_sid>_scaffold_ab_per_window.csv`）
`subject, seizure_idx, window_center_sec, phase(pre|ictal|post), n_joint, C_AB, r_A, r_B, maxAB,
axis_present, within_shaft_pointwise_p, side01, side_label(A|B|unlabeled)`

### 表2 per-seizure state（`per_subject/<ds_sid>_scaffold_ab_per_seizure.csv`）
`subject, seizure_idx, n_axis_present_win, state_sequence(串), switch_count(axis_present 窗内 A↔B 转换数),
polar_far, polar_near_pre, polar_early_ictal, polar_near, locking, locking_shift_p(枚举精确 p,描述用), n_valid_shift,
far_side(A|B|none), near_side(A|B|none), event_class(selection|switch|persistent|none)`
（`event_class` 按 §3 H2 taxonomy；`polar_near_pre`/`polar_early_ictal` 用 §6.1 描述窗，P10）

### 表3 per-subject summary（`per_subject/<ds_sid>_scaffold_ab_summary.json` + cohort 汇总）
- 分层：`rho_AB`, `template_pair_tier(reciprocal|oblique|aligned|hard_degenerate)`,
  `scaffold_tier(fine|coarse-only|untestable|low_dof)`, `forward_reverse_reproduced`（连 Topic-1）。
- 侧向（**三段分报，P10**）：`side_probability` 对 far-pre / near-pre / early-ictal 各一组 A/B/unlabeled 概率、
  `side_entropy`（三段）、transition matrix（A→A/A→B/B→A/B→B，只在 axis_present 窗）、
  `event_class_counts`（selection/switch/persistent/none 各几次发作）。
- H1：`L_obs`, `L_null_p95`, `subject_locked(bool)`, `n_seizures_locked / n_valid_seizures`, `H1_eligible(bool)`。
- QC：`n_joint`, `n_shafts`, `n_singleton_shafts`, `n_contacts_shuffled`, `fraction_contacts_shuffled`,
  `insufficient_joint`, `hard_degenerate`, `axis_present_testable`, `axis_present_low_dof`,
  `insufficient_valid_seizures`。
- cohort 汇总另加：`k`, `m`, `binom_p`, `binom_ci`（§4/§6.3 subject-count 二项，不 pool seizure）。

---

## 8. 三张图（CLAUDE.md §7：一图一独立问题，不重画）

先读 `docs/figure_style_guide.md`；paper-grade 自包含（无 §X/括号轴标、共享图例、render→目视→改再定稿，
`MEMORY.md → feedback_figure_self_contained_paper_grade`）。

1. **`plot_topic5_scaffold_ab_contrast_timecourse.py` — `C_AB(t)`（承载 H1 统计）**
   问：**哪一侧、随时间怎么变、临 onset 极化不极化**。**不能用可能互相抵消的裸 signed median（P9）**——改三层：
   ① 细线 = 每次 seizure 的 `C_AB(t)`；② 粗线 = 按各 seizure 近-onset 主侧对齐后（即取 `sign(polar_near)·C_AB`）的
   median，直接对应 H1 的 `|mean|`；③ 底部 tick/底纹 = axis_present 窗；④ inset = `polar_far/polar_near/locking` +
   环移 null 带。0 线=无侧向、onset 竖线。**替换**旧 signed 双线面板（`r_A`/`r_B` 对反相层冗余）。
2. **`plot_topic5_scaffold_ab_state_space.py` — 侧向×scaffold 轨迹（描述/直觉，非独立证据）**
   问：**侧向坐标在可信 scaffold 窗里的时间轨迹**。x=`C_AB`(−1 B / +1 A)，y=scaffold-presence（`maxAB` 或
   `−log10(within_shaft_p)`），颜色=peri-onset 时间，每 seizure 一条轨迹。**⚠️ 不声称 y⊥x（P8）**：对理想 reciprocal
   模板 `zB≈−zA` → `D_AB≈2·eA` → `C_AB≈r_A`、`maxAB≈|C_AB|`，y 与 |x| 近乎数学绑定；所以 y 是**可解释强度 /
   scaffold-presence QC**，不是与 `C_AB` 独立的第二动力学维；本图**只作描述、不作独立统计证据**。措辞也改：发作前多数点
   已在**较高** scaffold-strength 区，临 onset 主要变化是 **x 轴侧向坐标的极化/稳定/反转**，不一定是 y 从低往高爬。
3. **`plot_topic5_scaffold_ab_cohort_raster.py` — 队列 raster（承载队列）**
   问：**谁锁/谁切换/谁无侧向、在什么时间**。行=被试（按 §4 两轴分层排序），列=peri-onset 时间，颜色=`C_AB`
   （发散、幅度门控外的窗置灰）。

**showcase 选择（用户拨正，按当前 fragile 读出，固定取向 `C_AB` 出来后必须复核）**：
- **正例**：E1146（过 within-shaft null + 教科书 3 子型）、E922（n=28、方差小）。
- **caution panel**：E548 / E635（|r| 高但几何/null 不稳——高相似 ≠ 稳侧向，放警示位，不当正例）。
- 复核动作：固定取向 `C_AB` 算完后，用 §6.3 的 subject_locked + §4 tier 重新排 showcase，避免沿用镜像漂移期的印象。

---

## 9. 文件结构 / 复用清单 / 目录

### 新建
- `src/topic5_scaffold_ab_contrast.py` — 核心：`derive_joint_contacts`, `build_D_AB`（earlyness：返回
  `eA,eB,D_AB,rho_AB`，§2.1）, `contrast_timecourse`（返回逐窗 `C_AB,r_A,r_B,maxAB`）, `label_sides`,
  `classify_event`（selection/switch/persistent，§3 H2）, `locking_statistic`,
  `circular_shift_null`（整表环移，§5）, `template_pair_tier`（4 档，§4）。
- `scripts/run_topic5_scaffold_ab_switching.py` — per-subject + `--all-ok` 批量；出表1/2/3 + npz + cohort 索引；
  fail-closed per seizure / per subject。
- `scripts/plot_topic5_scaffold_ab_contrast_timecourse.py`
- `scripts/plot_topic5_scaffold_ab_state_space.py`
- `scripts/plot_topic5_scaffold_ab_cohort_raster.py`
- `tests/test_topic5_scaffold_ab_contrast.py` — §6.4 五个 fixture + `C_AB` **direct `corr(E,D_AB)`** 定义测试
  （闭式只在 full-J 全有限时另测等价，**缺触点窗只测 direct**，P2b）+ 退化守卫 + per-seizure 时间 null **穷举**
  （不是抽样）+ 环移保自相关。

### 复用（不得重造 —— CLAUDE.md §6 "re-use don't re-invent"）
- `scripts.compute_topic5_signed_broadband_similarity`：`_compute_values`（逐窗 z 能量）, `_load_axis`, `_nan`。
- `scripts.plot_topic5_signed_broadband_similarity_timecourse`：`_eligible_idxs`, `_on_common_grid`。
- `src.topic5_axis_alignment`：`matched_channels`, `within_shaft_shuffle`, `channel_shuffle`, `make_field_record`。
- `src.propagation_skeleton_geometry.parse_shaft`。
- `src.propagation_contact_plane_readout`：`R_smooth_rank`, `make_plane_grid`（仅 field sensitivity 用）。
- `run_topic5_fig3b_maxab_spatial_null` 的 maxT/cluster 校正（若出显著区间图）。

### 目录（AGENTS.md Results Directory Standards）
```
results/topic5_ictal_recruitment/scaffold_ab_switching/
├── cohort_summary.json
├── figures/
│   ├── README.md                      ← 必须，中文，图实际生成后写
│   ├── <ds_sid>_scaffold_ab_timecourse.png
│   ├── <ds_sid>_scaffold_ab_state_space.png
│   └── cohort_scaffold_ab_raster.png
└── per_subject/
    ├── <ds_sid>_scaffold_ab_per_window.csv
    ├── <ds_sid>_scaffold_ab_per_seizure.csv
    └── <ds_sid>_scaffold_ab_summary.json
```

---

## 10. 失败合同（fail-closed，禁静默绕过）

- `insufficient_joint`（`|J| < 6`）/ `hard_degenerate`（`rho_AB ≥ 0.85`）/ `axis_present_untestable`
  （非单触点杆 < 2）→ 该被试**显式落 drop 记录**，退出 H1/H2，不得静默补默认值。
- 某 seizure 抛异常 → per-seizure drop 记录，不拖垮整被试（照 `run_topic5_fig3b_maxab_spatial_null` 的 fail-closed 批量）。
- `C_AB` 主分析**禁**调用任何带镜像选择的路径（`corr_pair_mirror_invariant*`）；只有 field sensitivity 用平滑，且
  `D_field` 必须先固定取向再相减。CI/测试里加一条：主路径 import 不引入 mirror 选择函数。
- **禁**用统一 0.5 当 axis_present 统计门（只允许 within-shaft null 不可测时的描述性回退，且显式标注）。

---

## 11. 范围边界（非目标）

- **不**声称 preictal switching 已被证明（预注册纪律）；H1 是假设检验、H2 是描述。
- **不**把 `C_AB` 的侧向解释成临床 SOZ 覆盖（那是 V3c，另 spec）。
- **不**碰模型层（Topic4）；本 spec 纯数据侧。
- **不**复活旧 V3p 的"离轴迁移"框架；只改写其结论口径。
- Yuquan 结构性合格被试少（Fig3-B 材料池里多数 Yuquan drop）→ 主队列仍以 Epilepsiae 为主，Yuquan 逐个报、不硬凑队列。

### 11.1 对 abstract 的措辞约束（与 spec tier 一致）
- **禁**在本线跑完前写"发作起始……宽带能量梯度**均显著**对齐于……模板之一"。`均显著` 隐含 per-subject 普遍性，
  而 V3d primary 尚待检验、fine scaffold 只在过 within-shaft null 的一部分被试成立（现约 5/18）。
- 结果没出前的稳妥写法：*"在拥有足够发作次数和可检验 scaffold 的患者中，发作起始附近的 1–150 Hz 宽带能量分布通常
  仍位于间期 HFO 定义的传播 scaffold 上；A/B 对比轴分析显示部分患者在发作临近/早期表现出超过时间环移 null 的侧向
  极化，提示发作招募并非任意扩散，而是在既有病理传播轴上发生状态选择/切换。"*
- 跑完后按 §4 的 `k/m` 二项：不显著 → 保持"部分患者"；显著且 `k/m` 高 → 才可升"多数可检验患者"。

---

## 12. 最小实现路线（有序；先接受设计再拆 bite-sized plan）

1. `src/topic5_scaffold_ab_contrast.py`：contact-space `D_AB`（固定取向）+ `contrast_timecourse`，先过 §6.4 恒等/退化/
   镜像三条 TDD。
2. axis_present：接 `within_shaft_shuffle` 算 raw-contact maxAB 的 within-shaft null（pointwise p）。
3. 出表1/2/3 + 三张图（先 E1146/E922 单被试目视，确认固定取向后切换形态是真的、不是镜像翻面）。
4. circular-shift 时间 null + `locking` 统计 + subject-level 判据（§6.2/6.3），过 `static_on_axis`/`ramp_to_onset`
   正反对照。
5. `--all-ok` 批量 + 队列计数（§4 两轴分层），写 cohort_summary + figures/README.md。
6. 结论只写到"H1 锁定 N/总数 被试、H2 描述"，同步 topic5 主文档改写 V3p 口径（§1.1）。

---

## 13. 待 review 确认点（rev1 后剩余；写给下一轮的自己/用户）

- **rev1 已锁**（首轮 review 10 条）：rank 极性 earlyness（§2.1）、去镜像（§2.2/§10）、整表环移时间 null（§5/§6.3）、
  H1=极化非切换 + switch/selection/persistent taxonomy（§3/§7）、`N_VALID_SEIZURE_MIN=3` + `N_VALID_NULL_MIN=800`
  + `axis_present_low_dof`（§6.1/§2.4）、4 档 `rho_AB`（§4）、subject-count 二项（§4/§6.3）、near-pre/early-ictal 分报
  （§6.1/§7）、状态空间图重解释（§8）、图1 去 signed-median（§8）、abstract 措辞约束（§11.1）。
- **仍待你拍板**：① `DELTA_SIDE=0.2`/`NEAR_ONSET=[−30,+10]`/`FAR_PRE=[−120,−60]`/`EARLY_ICTAL=[0,+10]` 数值；
  ② subject_locked 除前置门外，要不要**再加** `n_seizures_locked / n_valid_seizures ≥ 0.5` 的更保守判据；
  ③ 状态空间图 y 轴取 `maxAB` 还是 `−log10(within_shaft_p)`；④ showcase 正例 E1146/E922 + caution E548/E635 在固定
  取向 `C_AB` 出来后是否维持。
