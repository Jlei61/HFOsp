# Topic 5 C 线计划：发作子型 ↔ 发作激活方向（病人内 modifier，exploratory）

> 日期：2026-06-15 · 状态：**已执行（v2 审阅修复版，全队列 broadband+hfa）→ 结果见 `docs/archive/topic5/subtype_direction_cline_result_2026-06-15.md`** · 层级：**exploratory 病人内 modifier，不做队列主张**
> **结果一句话**：cohort-eligible 仅 2/14（合格=几何+成簇+子型三门同过），0/2 子型轴角分离显著，C↔A 机制连接无关系 → 队列不可行 + 无信号（非"证明无"，是没看清+凑不出队列）。
> 上游：A 线主线 `docs/archive/topic5/axis_alignment_AB_result_2026-06-14.md` +
> hfa×joint 复验 `docs/archive/topic5/hfa_joint_confirm_2026-06-15.md`
> 复用：`src/topic5_axis_direction.py`（gradient_angle / axial_mean / axial_resultant_length / circular_mean / resultant_length）+
> `src/topic1_topic5_bridge.py::load_topic5_subtype_labels` + `scripts/plot_topic5_axis_direction_rose.py` 的 `_seizure_angles` / `_electrode_kind` / `_load_frame`
>
> **v2 修订（审阅报告 5 点 + 1 改进方向）**：
> 1. 轴向统计抹掉 180° 反向 → **拆成两层**：Q_axis（轴角，连 A 线）+ Q_pol（极性/端点，仅描述）。
> 2. A 线连接从"目视讲故事" → **每被试三列数值同表的真检验**（A 线不稳标量 × 奇偶子型不平衡 × 子型轴角分离）。
> 3. 玫瑰图先验被过读 → 现玫瑰图两瓣是**间期事件模板方向**，不是发作方向成簇；pilot **先确认逐发作方向本身成簇**。
> 4. SEEG/近一维几何脆 → **加几何质量门**（电极类型 + 触点云二维展开纵横比），近一维只作 caveat/case-series。
> 5. 对齐别重造轮子 → 复用 `load_topic5_subtype_labels`（硬性 `status=="ok"`），按 **seizure_id** 经 T0 audit CSV（`seizure_idx`↔`seizure_id`）连 z-ER；执行前打印核对索引空间。

---

## 0. 白话摘要（§8 三段式）

**测了什么。** A 线发现"平时传播主轴 ↔ 发作早期激活"共享一根**粗轴**——注意 A 线是**符号自由**的，只看是不是
共线（同一根线），不看朝哪个端点。C 线问的是更细的一层：**同一个病人体内，不同种类的发作（子型），点亮脑子的
方向是不是不一样**。这里"方向不一样"有两种完全不同的含义，必须分开问：
(1) **轴本身转了吗**（子型 A 沿东西向、子型 B 沿南北向）——这种差异会改变那根粗轴，**能解释 A 线为什么有时强有时弱**；
(2) **轴没转、只是点亮的端点掉了个头**（都沿东西向，但 A 从东往西、B 从西往东）——这种纯 180° 掉头，因为 A 线
符号自由、**对 A 线毫无影响**，只能当"发作方向"的描述性读出。

**怎么测的。** 每次发作算一个激活方向 θ（从触点平面上的激活标量场拟合**梯度方向**，和玫瑰图同一套；θ ∈ [0, 2π)，
是有正负的真方向，不是 wavefront 速度）。读每次发作的"子型标签"（已有的 z-ER 子型聚类，只用 `status=="ok"` 的）。
把发作按 seizure_id 对齐到子型后，做**病人内置换检验**——把子型标签在该病人发作之间随机打乱，看两个量是否超过随机：
**Q_axis**：用轴向表示 u=(cos2θ, sin2θ)（θ 与 θ+π 同轴）测"子型间**轴角**分离"；**Q_pol**：用完整方向 (cosθ, sinθ)
测"子型间**端点极性**分离"。然后把"子型轴角分离 Q_axis"和"A 线奇偶分半的不稳程度"放进一张表里直接对照。

**揭示了什么（预期口径）。** 这是**探索性的病人内修饰量，不是队列主张**。最有价值的连接是：A 线最精细那层（快活动 ×
最严联合洗牌）**在全数据上能检出、但按发作奇偶分半时奇数半掉了**（不稳）。如果某些病人"子型不同 → 发作**轴角**不同"
（Q_axis 显著），那奇/偶分半就会把不同轴角的子型不均匀地拆到两半，自然让那根粗轴时强时弱——**这是 A 线 split-half
不稳的一个强候选机制**。注意：**只有 Q_axis（轴角）有资格解释 A 线，Q_pol（纯掉头）没资格**（A 线符号自由）。
反过来，如果轴角不随子型变，A 线的不稳就更可能只是样本量/噪声。**C 线不能反过来证明子型真实。**

（内部归档代号：C 线 = subtype × activation-direction；Q_axis = 轴向 axial_mean/axial_resultant_length；
Q_pol = 方向 circular_mean/resultant_length；子型源 = z-ER `per_band[band].subtype_label`（status=ok）；
A 线不稳标量 = hfa×joint `|real_median_abs_corr(even) − real_median_abs_corr(odd)|`；split_half_robust=False。）

---

## 1. 假设与 A 线挂钩（两层分开）

- **H_C-axis（主问题，exploratory，连 A 线）**：病人内，发作激活**轴向角度**随 z-ER 子型系统性变化（子型 → 不同轴），
  超过随机打乱子型标签所能达到的轴角分离。**这一层、且只有这一层，能作为 A 线 split-half 不稳的候选机制。**
- **H_C-pol（次级，纯描述）**：病人内，发作激活的**端点极性**（θ vs θ+π，同轴不同向）随子型变化。
  **因 A 线符号自由，本层与 A 线强弱无关**，仅作"发作往轴哪一端点亮"的方向读出。
- **H_C→A（连接，定量检验，不作因果）**：若 H_C-axis 在若干病人成立，则这些病人正是 A 线 hfa×joint 按发作奇偶
  分半时**最不稳**的来源。检验方式 = 每被试三个标量同表对照（§4），不是目视讲故事。
- **先验：弱（已下调）**。原计划把方向玫瑰图"两瓣"当成发作方向成簇的证据——**错**：现玫瑰图两个空心直方瓣是
  **间期事件**按模板 A/B 的方向，**发作方向只是图上的灰色细 tick + 黑色轴**。"逐发作方向本身是否成簇"**尚未目视确认**，
  必须在 pilot 第 0 步先确认（§8）。子型标签本身也是 exploratory（z-ER PR-1 自评 ~70% 命中、未过 sensitivity、epi-only）。

## 2. 数据源 + 对齐合同（§6.2 关键：两套 seizure 索引按 seizure_id 对齐，不按位置）

**已核实的索引空间事实（执行时仍须 assert）：**
- T0 特征缓存 `*.json` 的 `eligible_idxs` 是**位置索引** `[0,1,2,…]`（npz 键 `bb_auc__<idx>` / `hfa_auc__<idx>`）。
- z-ER `*_zer_binned.json` 的 `seizure_ids_kept` 是 **seizure_id 字符串**（如 `59000000102`），与 `subtype_label` 平行。
- 桥梁 = `results/topic5_ictal_recruitment/t0_eligibility_audit.csv`，每行同时有 `seizure_idx`（= cache idx）与 `seizure_id`。
- **已验证**：z-ER `seizure_ids_kept` 的 id 命名空间与 audit CSV `seizure_id` **一致**（590 七个 id 全部对得上）；
  且 z-ER 通常只保留 T0 子集（590：T0 cached 12，z-ER kept 7）→ 用 **seizure_id 内连接**自然取交集。
- **C 线 = epilepsiae-only**：z-ER 目前无 yuquan extractor（topic5 PR-1 epi-only），yuquan 不进 C 线。

**对齐合同（实现时硬执行）：**
1. 子型：`load_topic5_subtype_labels(subject, band, results_root)`（复用，不重造）。**硬性要求返回 `status=="ok"`**，
   否则该被试整体跳过（记 drop 原因）。得到 `seizure_id_to_subtype`（剔除 `subtype_label==-1` 的 outlier）。
2. 方向：从 T0 cache 取 `eligible_idxs` + 每发作激活，按 audit CSV 建 `seizure_idx → seizure_id` 映射。
3. **内连接 on seizure_id**：得到对齐三元组 `(seizure_id, θ_s, subtype)`。**执行前打印**：z-ER id 数、cache id 数、交集大小；
   **若交集为 0 → 不静默按位置盲配，raise（fail-loud）**，再人工检查是否需 onset 回退。
4. **onset 回退（仅当 seizure_id 命名空间不一致时启用，已记录为备选不是默认）**：z-ER 侧 onset 从其 JSON、T0 侧 onset 从
   inventory（`load_seizure_onsets`）各自取，按 onset 时间容差匹配。**严禁按位置盲配**（§6.2 / §6 paired-cohort key-match）。
5. 只保留：两边都有 + `subtype_label != -1` + 方向可算（≥3 有限触点）+ 过几何质量门（§3.5）的发作。

## 3. 方法（病人内；两问题两统计 + 几何质量门）

### 3.0 第 0 关（先于子型检验）：逐发作方向是否成簇？
对每被试，先算其全部对齐发作角度 {θ_s} 的轴向集中度 `axial_resultant_length` 与方向集中度 `resultant_length`，
与"均匀随机方向"零分布比（病人内置换/解析）。**若发作方向本身不成簇（R_axial 不超随机），子型检验无意义 → 记
`not_clustered`，仅描述不进 cohort。** 这一关挡掉审阅点 3 的"过读玫瑰图"。

### 3.1 每发作方向
θ_s = `gradient_angle(x, y, activation_field)`，平面与 A 线/玫瑰图同一个（`_subject_display_frame` / `_display_points`）。
保留 θ_s ∈ [0, 2π)（**有极性**），轴向与方向两层都从它派生。

### 3.2 Q_axis（轴角层，连 A 线）
- 轴向单位向量 u_s = (cos 2θ_s, sin 2θ_s)；每子型轴中心 φ_g = `axial_mean`、R_g = `axial_resultant_length`。
- 统计量 `T_axis`：k=2 → 两子型轴中心**轴向角距** `axial_distance(φ0, φ1) = min(|φ0−φ1|, π−|φ0−φ1|) ∈ [0, π/2]`；
  k>2 → 按子型大小加权两两轴向角距平均。
- 病人内置换：发作间随机打乱 subtype_label，B=2000 重算 T_axis；`p_axis = (1 + #{T_perm ≥ T_obs}) / (B+1)`。

### 3.3 Q_pol（极性层，仅描述）
- 方向单位向量 v_s = (cos θ_s, sin θ_s)；每子型方向中心 ψ_g = `circular_mean`、R = `resultant_length`。
- 统计量 `T_pol`：k=2 → 两子型方向中心**圆角距** `circular_distance(ψ0, ψ1) = min(|ψ0−ψ1|, 2π−|ψ0−ψ1|) ∈ [0, π]`；
  k>2 → 加权平均。同法病人内置换得 `p_pol`。
- **口径锁**：Q_pol 是"发作往轴哪端点亮"的描述性读出，**禁止**用它解释 A 线（A 线符号自由，纯掉头不影响 A 线）。

### 3.4 子型轴/向中心 + 置换（实现细节）
`axial_mean` / `circular_mean` 已 TDD。置换零分布按 §3.2/§3.3；坏数据门见 §3.6。

### 3.5 几何质量门（审阅点 4）
对每被试记录并据此分层：
- `electrode_kind`：`_electrode_kind(ds, subj, names)` → `ECoG` / `SEEG`（含 detail）。
- `coord_aspect`：触点显示坐标中心化后 (n×2) 矩阵的 **第二/第一奇异值比** s2/s1 ∈ [0,1]。近一维 → s2/s1→0 → 梯度角脆。
- **门**：`coord_aspect < ASPECT_MIN`（locked 0.15）或 `electrode_kind=="SEEG"` → 标 `geometry_caveat=True`，
  **该被试降为 case-series（仅描述，不进 cohort 计数）**；ECoG 网格且 aspect 充足才进 cohort。

### 3.6 eligibility 分层（locked）
- **小子型丢弃**：先把对齐后 size < 3 的子型**丢弃**（记 `dropped_subtypes`，它们不可靠、不进检验）。
- **cohort-test eligible**：`status=="ok"` + **丢弃小子型后仍 ≥2 个子型、各 ≥3 个对齐发作** + 方向成簇（§3.0）+ 过几何门（§3.5）→ 进 permutation。
- 否则 **case-series**（仅描述，eligibility=`insufficient_subtypes`）。全 C 线 exploratory；`subtype_size < 3` 的子型永不进检验。
- **已知（pilot 实测对齐前）**：590 子型 5/2 → 丢 2 → 仅剩 1 子型 → case-series；958 子型 7/3/2 → 丢 2 → 7/3 进检验；
  922 子型 18/6 → 两者进检验。合格队列预计很小，这是诚实范围，不是失败。

## 4. A 线连接 = 真检验（审阅点 2：每被试三标量同表，不目视讲故事）

对每个**有 A 线 split-half 数据**的被试，建一行三列（全部 exploratory、描述性）：

| 列 | 定义 | 来源 |
|---|---|---|
| `subtype_axis_sep` | C 线 Q_axis 的 T_axis（轴角分离，连 A 的唯一合法层）+ p_axis | §3.2 |
| `oddeven_subtype_imbalance` | 在**同一奇偶分半**下两半子型分布的总变差距离 TV(even, odd) ∈ [0,1] | `seizure_parity_subsets(eligible_idxs)` + §2 对齐子型 |
| `oddeven_aline_instability` | `|real_median_abs_corr(even) − real_median_abs_corr(odd)|`（hfa×joint） | `hfa_joint_confirm.json::arms.split_half_{even,odd}.per_subject` |

- **奇偶分半口径锁定**：必须复用 `seizure_parity_subsets`（位置奇偶）作用在 **A 线同一 eligible-seizure 顺序**上；
  子型不平衡只在"已对齐且有子型标签"的子集上算（590：12 eligible 里 7 有标签，按它们落在奇/偶的位置算 TV）。
- **读出（描述性，不做因果/不 pooled-p）**：看 `subtype_axis_sep` 高 & `oddeven_subtype_imbalance` 高的被试，
  是不是同时 `oddeven_aline_instability` 高（秩对照 / 散点）。**机制预测**：三者同高 = "子型轴角异质 → 奇偶拆分 →
  A 线细对齐时强时弱"链条成立的候选证据。**禁止**写成"证明子型导致 A 线不稳"。

## 5. 图（每类答一个独立问题，§7 figure discipline）

- **图 A：per-subject 方向玫瑰（子型上色）** —— 复用玫瑰骨架，但画的是**逐发作方向 tick 按子型上色**
  （不是间期模板两瓣！），黑线为发作轴（axial mean，0°/180°）。标 `electrode_kind` + `coord_aspect` + `geometry_caveat`。
  一眼看"逐发作方向是否按子型分簇"（直接回应审阅点 3）。
- **图 B：subtype × direction 圆散点** —— 角度=每发作 θ_s（画完整 [0,2π) 以显极性），按子型上色分列；
  并排显示 Q_axis（折到 [0,π)）与 Q_pol（完整）两个读出。配 T_axis/p_axis、T_pol/p_pol。
- **图 C：A 线连接三标量散点/表** —— 横 `subtype_axis_sep`、纵 `oddeven_aline_instability`、点大小/色 = `oddeven_subtype_imbalance`。
  直接呈现 §4 的三列关系。
- 强例/弱例各挑 1–2（先看 ECoG 网格、aspect 高的被试方向最实）。每图配 `figures/.../README.md`（中文，§关注点）。

## 6. 复用 + 新代码 + TDD

**复用（不重造）**：
- `src/topic5_axis_direction.py`：`gradient_angle` / `axial_mean` / `axial_resultant_length` / `circular_mean` /
  `resultant_length` / `rotate_to_reference`（已 TDD）。
- `src/topic1_topic5_bridge.py`：`load_topic5_subtype_labels`（硬性 status==ok）、`load_seizure_onsets`（onset 回退用）。
- `src/topic5_axis_alignment.py`：`seizure_parity_subsets`（A 线同一奇偶口径）。
- `scripts/plot_topic5_axis_direction_rose.py`：`_load_frame` / `_seizure_angles`（抽出逐发作版返回**每发作** θ）/ `_electrode_kind`。
- `hfa_joint_confirm.json`：A 线 split-half per_subject `real_median_abs_corr`（不重跑 A 线）。

**新代码（全部 TDD）**：
- `src/topic5_subtype_direction.py`（纯函数）：
  - `axial_distance(a, b)` → ∈ [0, π/2]（TDD：d(0,π)=0；d(0,π/2)=π/2；对称）。
  - `circular_distance(a, b)` → ∈ [0, π]（TDD：d(0,π)=π；d(0,π/2)=π/2；d(0,2π−ε)=ε；对称）。
  - `subtype_separation_stat(angles, labels, mode)` → T（mode∈{axis,pol}；k=2 角距 / k>2 加权；TDD：轴向 {θ,θ+π} 两子型 →
    T_axis=0 但 T_pol=π；正交两子型 → T_axis=π/2）。**这条测试就是审阅点 1 的回归保护。**
  - `within_subject_perm_p(angles, labels, mode, B, rng)` → p（TDD：完全分离→小 p；随机→p≈均匀；坏数据门：单子型 / 某子型<3 → case-series 标记不返 p）。
  - `direction_clustering(angles)` → R_axial / R_dir + 是否成簇（§3.0；TDD：集中→高 R，均匀→低 R）。
  - `coord_aspect_ratio(x, y)` → s2/s1（TDD：共线点→0；正方形格→≈1）。
  - `align_subtype_to_direction(zer_labels, audit_rows, eligible_idxs, angles_by_idx)` → 对齐 `(seizure_id, θ, subtype)`；
    **TDD：seizure_id 内连接正确；命名空间不一致 → raise（禁止位置盲配）；缺标签发作丢弃**（§6.2）。
  - `oddeven_subtype_imbalance(even_set, odd_set, idx_to_subtype)` → TV 距离（TDD：均衡→0；全偏一半→1）。
- `scripts/run_topic5_subtype_direction.py`：per-subject（Q_axis + Q_pol + 几何门 + 成簇关）+ cohort 计数 + §4 A 线连接表 → JSON。
- `scripts/plot_topic5_subtype_direction.py`：图 A/B/C + README。

## 7. 禁止 / 允许措辞（§6.3 pronoun discipline）

- **允许**："病人内，发作激活**轴向角度**随 z-ER 子型分离（置换 p_axis=…，描述性）"；"这是 A 线 hfa×joint
  分半不稳的候选解释（子型轴角异质 → 奇偶拆分），**非因果**"；"发作**端点极性**随子型变（p_pol=…，仅方向描述，与 A 线无关）"。
- **禁止**：把 Q_axis 与 Q_pol 混成一句"方向随子型变"（必须指明是**轴角**还是**极性**）；用 Q_pol 解释 A 线；
  "方向簇证明子型真实"；"子型 → 方向的**队列主张**"；把 C 线升成 primary；用一个 "p<0.05" 当 subtype ground-truth。
  **C 线全程 exploratory 病人内 modifier。**

## 8. 最小执行顺序（PILOT-FIRST，硬停）

1. **TDD 建 `src/topic5_subtype_direction.py` 纯函数**（含审阅点 1 的轴角/极性回归测试），全绿。
2. **Pilot（2–3 被试：590/958/922，且 z-ER status==ok）**：
   - 第 0 步：打印对齐合同核对（z-ER id 数 / cache id 数 / 交集 / 子型分布）+ 逐发作方向成簇关（§3.0）。
   - 第 1 步：Q_axis + Q_pol per-subject permutation + 几何门 + 图 A。
   - **先停下来人看**（对齐对不对、逐发作方向是否真成簇、簇是否对应子型、ECoG vs SEEG 是否要降 case-series）。
3. 人看 OK 再放全 eligible 队列 + cohort 计数 + §4 A 线连接三标量表 + 图 B/C。
4. 写 archive（`docs/archive/topic5/`）：per-subject 表（Q_axis/Q_pol 分列）+ A 线连接段（三标量） + 强弱例图；
   主文档只留摘要 + 链接。**结论口径严守 §7**。
