# Topic 5 C 线计划：发作子型 ↔ 发作激活方向（病人内 modifier，exploratory）

> 日期：2026-06-15 · 状态：**计划（未执行）** · 层级：**exploratory 病人内 modifier，不做队列主张**
> 上游：A 线主线 `docs/archive/topic5/axis_alignment_AB_result_2026-06-14.md` +
> hfa×joint 复验 `docs/archive/topic5/hfa_joint_confirm_2026-06-15.md`
> 复用：`src/topic5_axis_direction.py`（gradient_angle / axial_mean / axial_resultant_length）+
> 方向玫瑰图的 per-seizure 方向代码路径（`scripts/plot_topic5_axis_direction_rose.py`）

---

## 0. 白话摘要（§8 三段式）

**测了什么。** A 线发现"平时传播主轴 ↔ 发作早期激活"共享一根粗轴；方向玫瑰图又提示：同一个病人，不同
发作的激活方向不是死死指一个方向，**有时会分成几簇**（有的病人两个模板正好是一根双向轴的两端、有的同向）。
C 线问的是：**同一个病人体内，这些"发作方向的簇"是不是对应不同的发作种类（子型）**——也就是"这次发作是
哪一型，决定了它往轴的哪一端、朝哪个方向点亮"。

**怎么测的。** 每次发作算一个激活方向 θ（从触点平面上的激活标量场拟合梯度方向，和玫瑰图同一套；注意是
**梯度方向不是 wavefront 速度**）。因为 A 线是符号自由的，方向一律用**轴向表示** u=(cos2θ, sin2θ)，把
θ 和 θ+π 当成同一根轴。再读每次发作的"子型标签"（已有的 z-ER 子型聚类）。然后做**病人内置换检验**：
把子型标签在该病人的发作之间随机打乱，看"不同子型之间的方向分离"是否超过随机打乱能达到的程度。

**揭示了什么（预期口径）。** 这是**探索性的病人内修饰量，不是队列主张**。如果某些病人确实"子型不同→
方向不同"，它**不能**直接证明子型真实，但**是解释 A 线一个现象的强候选机制**：A 线最精细那层（快活动 ×
联合对照）按发作奇偶分半时不稳（奇数半不显著）——如果发作方向按子型分簇，奇/偶分半就会把不同子型不均匀地
混进两半，自然让最精细的对齐时强时弱。**所以 C 线既是"发作异质性"的直接探针，又是"为什么 A 线 split-half
不稳"的候选解释。** 反过来，如果方向簇和子型对不上，那 A 线的不稳就更可能只是样本量/噪声。

（内部归档代号：C 线 = subtype × activation-direction；轴向统计 axial_mean/axial_resultant_length；
子型源 = z-ER `per_band[band].subtype_label`；A 线不稳 = hfa×joint split_half_robust=False。）

---

## 1. 假设与 A 线挂钩

- **H_C（主问题，exploratory）**：病人内，发作的激活**轴向**方向随 z-ER 子型系统性变化（不同子型 → 不同
  轴端 / 不同方向），超过随机打乱子型标签所能达到的分离。
- **H_C→A（次级，连接 A 线）**：若 H_C 在若干病人成立，则这些病人正是 A 线 hfa×joint 按发作奇偶分半时
  最不稳的来源（奇/偶分半把子型不均匀地拆开）。**作为候选机制检验，不作因果断言。**
- **先验**：中等。方向玫瑰图已**目视**看到方向成簇（590/958 两模板呈 ~180° 对瓣、922 同向集中），但
  "簇 = 子型"未检验；子型标签本身是 exploratory（z-ER PR-1 自评 ~70% 命中、未过 sensitivity）。

## 2. 数据源 + 对齐合同（§6.2 关键：两套 seizure 索引必须按身份对齐）

- **发作激活方向**：`results/topic5_ictal_recruitment/t0_feature_cache/<ds_sid>.npz` 的逐发作激活
  （`bb_auc__<idx>` 主 / `hfa_auc__<idx>` 次），`.json` 的 `eligible_idxs` 给参与的发作。
  方向 = `gradient_angle(x, y, activation_field)`，平面与 A 线/玫瑰图同一个（`_subject_display_frame`）。
- **子型标签**：`results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/per_subject/<ds>_<subj>__zer_binned.json`
  的 `per_band[band].subtype_label`（与 `seizure_ids_kept` 平行；`-1` = outlier，剔除）。band 主用
  `gamma_ER`、`broad_ER` 作 sensitivity。
- **对齐合同（必须在实现时先验证，不能假设两套索引同序）**：
  - z-ER 的 `seizure_ids_kept[i]` ↔ `subtype_label[i]`；T0 的 `eligible_idxs[j]` ↔ 激活列 j。
  - **先核对 `seizure_id` 与 `eligible_idx` 是否同一索引空间**（都索引该被试的 seizure inventory）。
    若不确定 → 各自映射到**发作 onset 时刻**（z-ER 侧从 `onset/t_onset`，T0 侧从 seizure 检测/inventory
    的 onset），按 onset 时间（容差内）匹配。**严禁按位置盲配**（§6.2 / §6 paired-cohort key-match）。
  - 只保留两边都有、且 `subtype_label != -1`、且方向可算（≥3 有限触点）的发作。

## 3. 方法（病人内，轴向）

1. **每发作轴向方向**：θ_s = `gradient_angle(...)`；轴向单位向量 u_s = (cos 2θ_s, sin 2θ_s)。**全程轴向**
   （A 线符号自由 → θ 与 θ+π 同轴）。
2. **每子型轴中心**：用 `axial_mean` 求每个子型的轴向均值 φ_g、`axial_resultant_length` 求 R_g。
3. **统计量 T_obs（子型间轴向分离）**：
   - k=2（多数被试）：T = 两子型轴中心的**轴向角距** d_axial(φ_0, φ_1) = min(|φ0−φ1|, π−|φ0−φ1|) ∈ [0, π/2]。
   - k>2：T = 按子型大小加权的两两轴中心轴向角距平均。
4. **病人内置换检验**：在该被试发作间随机打乱 subtype_label，B=2000 次重算 T；
   p_subj = (1 + #{T_perm ≥ T_obs}) / (B + 1)。
5. **eligibility 分层（locked，照搬用户口径）**：
   - **cohort-test eligible**：≥2 子型、每子型 ≥3 个对齐发作且方向可算 → 进 permutation。
   - 否则 **case-series**（仅描述，不进 cohort 计数）。
   - 全 C 线 exploratory；`subtype_size < 3` 永不进 cohort 主张。

## 4. 队列汇总 + 层级纪律

- **cohort 读出（弱）**：报告 eligible 被试里 p_subj<0.05 的**个数**（binomial vs 5% 仅作参考，不作主张）+
  每个 eligible 被试一行（T_obs、p_subj、k、每子型 n、R_g）。**不写"队列证明子型决定方向"。**
- **A 线连接读出**：把"H_C eligible 且 p_subj<0.05"的被试，与 A 线 hfa×joint 复验里**奇偶半差异最大**的
  被试做**目视/秩**对照（是不是同一批）。**描述性，不做因果断言。**
- **层级（locked，§6.3 pronoun discipline）**：C 线 = **exploratory 病人内 modifier**。
  允许句式："被试 X 的发作激活轴向方向随 z-ER 子型分离，置换 p=…（病人内、描述性）"。
  **禁止**："方向簇证明子型真实"/"子型决定发作方向（队列）"/把方向簇当 subtype ground-truth。

## 5. 图（两类，每类答一个独立问题，§7 figure discipline）

- **图 A：per-subject 方向玫瑰（子型上色）** —— 复用玫瑰图骨架，但把灰色 per-seizure 方向线**按子型上色**
  （子型 0/1/… 各一色），黑线仍是发作轴（axial mean，0°/180°）。一眼看"方向簇是否按子型分开"。
- **图 B：subtype × direction 极坐标条带 / 圆散点** —— 横轴=子型，纵轴/角度=每发作 θ_s（轴向，画在 [0,π)），
  点按子型分列。配 T_obs、p_subj。看子型间方向是否分层。
- 强例/弱例各挑 1–2（先看 ECoG 网格被试方向最实）。每图配 `figures/.../README.md`（中文，§关注点）。

## 6. 复用 + 新代码 + TDD

**复用（不重造）**：
- `src/topic5_axis_direction.py`：`gradient_angle` / `axial_mean` / `axial_resultant_length` / `rotate_to_reference`（已 TDD）。
- `scripts/plot_topic5_axis_direction_rose.py`：`_load_frame` / `_interictal_event_vals`（不需要）/ per-seizure 激活方向路径（`_seizure_angles` 的逐发作版——抽出为可复用函数，返回**每发作**的 θ 而非聚合）。

**新代码（全部 TDD）**：
- `src/topic5_subtype_direction.py`（纯函数）：
  - `axial_distance(a, b)` → 轴向角距 ∈ [0, π/2]（TDD：d(0, π)=0；d(0, π/2)=π/2；对称）。
  - `subtype_separation_stat(angles, labels)` → T_obs（k=2 角距 / k>2 加权；TDD：两子型正交 → π/2；同向 → 0）。
  - `within_subject_perm_p(angles, labels, B, rng)` → p_subj（TDD：完全分离 → 小 p；标签随机 → p≈均匀；
    坏数据门：单子型 / 某子型 <3 → 返回 case-series 标记不返 p）。
  - `align_subtype_to_activation(zer_json, t0_meta, ...)` → 对齐后的 (per-seizure θ-index ↔ subtype)；
    **TDD：索引空间不一致时按 onset 匹配；不匹配的发作丢弃；按位置盲配必须被测试挡掉**（§6.2）。
- `scripts/run_topic5_subtype_direction.py`：per-subject + cohort 计数 + A 线连接对照 → JSON。
- `scripts/plot_topic5_subtype_direction.py`：图 A + 图 B + README。

## 7. 禁止 / 允许措辞

- **允许**："病人内，发作激活轴向方向随 z-ER 子型分离（置换 p=…，描述性）"；"这是 A 线 hfa×joint
  分半不稳的候选解释（发作异质性），非因果"。
- **禁止**：方向簇 = 子型证明；子型 → 方向的**队列主张**；把 C 线升成 primary；用一个 "p<0.05" 当 subtype
  ground-truth。**C 线全程 exploratory 病人内 modifier。**

## 8. 最小执行顺序（PILOT-FIRST）

1. 先在 2–3 个"方向已成簇"的被试（590/958/922，且有 z-ER 子型）跑通对齐合同 + per-subject permutation +
   图 A，**先停下来人看**（对齐对不对、方向簇跟子型是不是真的对得上）。
2. 人看 OK 再放全 eligible 队列 + cohort 计数 + A 线连接对照。
3. 写 archive（`docs/archive/topic5/`）：per-subject 表 + 强弱例图 + A 线连接段；主文档只留摘要 + 链接。
