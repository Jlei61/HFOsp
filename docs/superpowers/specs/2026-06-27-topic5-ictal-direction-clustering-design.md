# Topic 5 设计 spec：发作早期方向无监督两类聚类 ↔ 间期 A/B 模板方向（探索性，无预设假设）

> 日期：2026-06-27 · 状态：**设计 spec（待用户审 → writing-plans）** · 层级：**exploratory，描述性，非队列主张**
> 上游：A 线主线 `docs/archive/topic5/axis_alignment_AB_result_2026-06-14.md`（间期轴 ↔ 发作早期共线，**符号自由**）
> 失败前序：C 线 `docs/archive/topic5/subtype_direction_cline_result_2026-06-15.md`（用既有 z-ER 子型解释方向，合格被试太少）
> 代码归属：新纯函数模块 `src/topic5_directional_replay.py`（TDD）+ runner + 图；复用 `src/topic5_axis_direction.py`

---

## 0. 白话摘要（§8 三段式）

**测了什么。** 同一个病人，每次发作头十秒"往脑子哪个方向烧"（在电极平面上，把各触点的早期激活强度拟一个增长方向，得到一个方向角）。我们**不预设**任何"正/反"方向，也**先不看**平时的间期传播模板，纯粹按"方向像不像"把这个病人的多次发作分成两堆；分完后每堆有一个堆内平均方向。最后才把这两个方向，和这个病人平时那两条间期传播路线的方向，摆在一起看：对得上吗、怎么对。

**怎么测的。** 关键是不能被算法骗：在二维方向上硬分两类，**任何**一组角度都会被切成"主堆/次堆"，哪怕真相只有一个主方向加噪声。所以我们设了三道**事先锁死**的门：① 要谈"两类"，得发作数 ≥ 6、每堆 ≥ 3，且实测的两堆分离程度要超过"一个单峰方向 + 噪声随机切出来"的水平（拿单峰分布模拟两千次比）+ 自助重抽样里这个两类划分还得稳；过不了的病人只准说"一个主方向"，禁止说"两类"。② 间期两条路线本身得方向相反才谈得上"对上 A/B"——两方向夹角 <60° 不算方向对（只诊断）、60–120° 算弱、≥120° 才算可解释，这个阈值事先锁、不看结果再定。③ "两堆方向对上间期 A/B 有多近"必须有零假设：把发作两堆的整体朝向随机旋转两千次、每次重做最佳配对，看真实的贴合度在随机里排第几——否则"2 配 2 取最近"天生就显得好。

**揭示了什么（预期，先把诚实摆桌面）。** 快速原型显示：多数干净病人其实是"一个主方向独大 + 少数零散"，强行分两类多半过不了上面第①道门，只能写"主方向"；真正能谈"两类"的可能就一两个病人；而且个别病人两条间期路线方向几乎重合（根本不成一对）。所以这条线最可能的结论是**描述性的、按被试分档的**——"发作主方向≈某一条间期路线"在干净病人里看起来成立，"两类干净对上 A 和 B"较弱——**不下队列断言、不预设假设、不声称重放**。

（内部归档代号：θ=`gradient_angle`(头10s bb/hfa AUC, z-ER 式激活场)；ictal 聚类=`[cosθ,sinθ]` 上 `KMeans(k=2)` 盲于间期；两类资格门=count + 二模 null(主方向+背景散点) p_bimodal + bootstrap 标签稳定性(次级)；间期模板方向=`gradient_angle(typical_rank)` on `_t_{a,b}.json`；轴质量门=Δ_AB；对齐 null=旋转 best-pair percentile p_align；上游 A 线 `|corr_pair_mirror_invariant|` 符号自由；tier=exploratory 描述性。）

---

## 1. 背景与定位（这条线"新"在哪、为什么非循环）

- **A 线已确立**：每被试间期模板 A 的逐触点排名场（传播轴）与发作头 10 s 的逐触点激活场，**空间梯度共线**（符号自由 `|corr_pair_mirror_invariant|`，含反向共线，**不判方向重放**）。18 Epilepsiae 队列，粗骨架稳。
- **本条线问更细一层、但口径完全不同于失败的 C 线**：
  - C 线（失败）= 用**既有 z-ER 子型**去解释方向 → 合格被试太少（最终 2 个）。
  - 本条线 = **完全不用 z-ER 子型，也不用 SOZ 锚点**；只用**发作早期方向本身**无监督分两类，分完再去和间期 A/B 方向比关系。
- **非循环保证**：聚类只用发作方向（lab 坐标系平面梯度，独立于间期）；间期模板方向只用间期排名场；两者**先各自独立算出**，最后才比关系。**不预设"正方向"、不用 SOZ 定向、不把"对上"反过来定义分类**（这是用户明确口径：起始不区分正负、本来也没看间期模板）。
- **与 A 线不重复**：A 线用每被试**合并的**发作激活场 + 符号自由 |corr|；本条线用**逐发作**方向 + 无监督两类 + 保留符号的方向角，问的是 A 线主动没碰的"方向是否分两极、两极是否对上两条间期路线"。

---

## 2. 数据与队列（预锁）

- **主队列 = 几何干净的板状电极（ECoG）Epilepsiae 被试**：`电极类型==ECoG` **且** `coord_aspect ≥ 0.15`（触点云二维纵横比，PCA 次/主奇异值比）。当前命中 6 个：**442, 548, 583, 1084, 384, 958**。
- **深部电极（SEEG）/ 近一维（aspect<0.15）= caveat-only**：方向是投影/塌缩伪影，**不进主队列**；如做，单列"投影 caveat 灵敏度层"，不参与任何描述性汇总结论。
- **每发作激活场 = z-ER 式早期激活**：复用 A 线 `t0_feature_cache/{ds_sid}.npz` 的 `{feat}_auc__{sz}`（头 10 s AUC）。**band：broadband（`bb_auc`，1–45 Hz）为主；HFA（`hfa_auc`，60–100 Hz）作灵敏度**。两 band 分别跑、分别报，不合并解释。
- **坐标系**：复用 `_load_frame(ds_sid)`（由 `_t_a.json` 经 `_attach_real_coords`+`_subject_display_frame`+`_display_points` 得 x,y + 触点名）。发作方向、间期模板方向**同一坐标系**直接比（相对夹角对刚体旋转/翻转不变，保留符号干净）。
- **间期两模板场**：`results/spatial_modulation/propagation_geometry/observation_readout/real_subjects/{ds_sid}_t_{a,b}.json`，每触点带 `typical_rank` / `x_norm,y_norm` / `support`。已确认 6 个被试 t_a、t_b 均在。
- **不需要 SOZ**：本设计无锚点，`is_soz` 不参与（与上一版 SOZ-anchored 设计的区别）。

---

## 3. 方法（逐步）

对每被试、每 band：

**Step 1 — 逐发作方向。** 对每个合格发作 sz，取激活场 `vals = bb_auc[:, sz]`（对齐到坐标系触点序），`θ_sz = gradient_angle(x, y, vals)`（最小二乘平面拟合，值增长方向，[0,2π)）。丢 NaN（<3 有限点或无梯度）。得 `{θ_sz}`。

**Step 2 — ictal-only 两类聚类（盲于间期）。** 在单位向量 `V = [cosθ, sinθ]` 上 `KMeans(n_clusters=2, n_init=10, random_state=0)`。得标签、两类大小 `(n1,n2)`、两类堆内平均方向 `θ_c1,θ_c2`（`circular_mean`）、各类堆内集中度 `R1,R2`（`resultant_length`）。同时报全体 `R_dir`（`resultant_length`）、`R_axial`（`axial_resultant_length`）。

**Step 3 — 间期模板方向 + 质量。** `θ_A = plane_fit_direction(x, y, typical_rank_A)`，`θ_B = plane_fit_direction(x, y, typical_rank_B)`（typical_rank_B 按触点名对齐到坐标系序，缺则 NaN）。返回方向角 + **梯度可靠度**（拟合梯度向量范数 `||β||` 或平面拟合 R²）+ 有效触点数 `n_valid`。`Δ_AB = 全圆角距(θ_A, θ_B) ∈ [0,π]`。

**Step 4 — 描述性关系量（无锚、对称）。**
- `Δ_ictal = 角距(θ_c1, θ_c2)`（≈π=真两极；小=单峰被硬切）。
- `Δ_AB`（=间期轴质量，见 §4.3）。
- ictal 轴 vs 间期轴偏移 = `axial_distance(axial_mean({θ_sz}), axial_mean([θ_A,θ_B]))`。
- **best-pair 残差** `resid_obs`：在 {c1,c2}→{A,B} 的两种配对（straight `c1-A,c2-B` / crossed `c1-B,c2-A`）里取角距和最小者；记录配对方式 + 两个匹配角。

**Step 5 — 三道预锁门（§4）+ 分档报告（§5）。**

---

## 4. 三道预锁门（事先锁死，不看结果再定）

### 4.1 P0 —「两类资格」门（防算法制造 seizure type）

`two_class_eligible == True` 需**同时**满足：
1. **硬数量门**：`n_sz ≥ 6` 且 `min(n1,n2) ≥ 3`。
2. **二模 null 显著**：`p_bimodal < 0.05`。
   - 统计量 `S` = k=2 标签在 `V=[cosθ,sinθ]` 上的 **silhouette**。
   - null 的 H0 = **一个集中主方向 + 均匀背景散点**（**不是**纯单峰）：先 k=2 分出多数簇，对多数簇拟合 von Mises（μ=多数簇 circular_mean，κ=Mardia-Jupp `A⁻¹(R_多数)`），少数比例 `f=n_minor/n`；每次模拟 n 个角度——以概率 (1−f) 抽自该 von Mises、以 f 抽自 [0,2π) 均匀——再跑 k=2 取 `S_null`；`p_bimodal=(1+#{S_null≥S_obs})/(B+1)`，B=2000。
   - **理由（审阅 P1 修复，2026-06-27）**：纯单峰 null **太弱**——"主方向 + 少数散点"会被 k=2 切出一个远处小簇、silhouette 偏高、纯单峰 null 拦不住（实测 20 紧 + 4 散在纯单峰 null 下 p≈0.002 = 假阳，把噪声写成两类）。正确的 H0 是"主方向 + 散点"，要检出的信号是"出现第二个**集中**的方向模"。**已标定**：20 紧 + 4 散 p≈0.27（挡住）、纯单峰 p≈0.85（挡住）、真双峰（含偏斜 21:5 紧）p≈0.01–0.04（通过）；**真实数据只有 442 过门**（548/583/958 的"第二类"是主方向 + 散点，p≈0.2–0.58，正确降级为主方向）。
3. **自助标签稳定**：bootstrap 标签稳定性中位 `ARI ≥ 0.5`（**次级**检查）。
   - `B=500`，每次有放回重抽 n_sz 个发作 → k=2 → 把**全部**原始点指派到最近的重抽质心 → 与原始标签算 ARI（标签对换取大者）→ 取中位。
   - **注意**：stability 只测"划分是否可复现"，对**固定的散点**会给假高（同一批散点每次被一致分到一簇 → ARI≈1）；故**反散点主门是上面的二模 null，不是 stability**。stability 仅作"划分不稳"的额外排除。

**报告（无论是否过门，全报）**：`R_dir, R_axial, n1, n2, 大小比, R1, R2, Δ_ictal, S_obs, p_bimodal, stability_ARI`。
**过门 vs 不过门的措辞强约束见 §5**。

### 4.2 P1 —「best-pair 残差」旋转 null（消选择优势）

- `resid_obs` = §3 Step 4 的最小配对角距和。
- null：把 ictal 两类的**整体朝向**随机旋转 `φ ~ U[0,2π)`（即 `θ_c1+φ, θ_c2+φ`，**保 n_sz、保两类大小、保 Δ_ictal=保 ictal 形状**），对**固定的** `(θ_A,θ_B)` 重做 best-pair → `resid_null(φ)`；`B=2000`。
  - 说明：共同旋转两类均向 ≡ 刚体旋转全部角度后旋转等变地重聚类（partition 整体旋转 → 大小不变）→ 与"重复完整流程 + 固定 cluster size"等价；用闭式旋转均向实现，避开 k-means 种子不稳。
- `p_align = (1+#{resid_null ≤ resid_obs})/(B+1)`（残差越小越好，故数 ≤ 观测者）。**"对齐显著"= p_align < 0.05**。

### 4.3 P1 —「间期模板轴质量」门（防后验筛选）

按 `Δ_AB` 预锁三档：
- `Δ_AB < 60°` → **不成方向对，diagnostic-only**：剔出方向比较，只报诊断量（θ_A,θ_B,可靠度,n_valid）。
- `60° ≤ Δ_AB < 120°` → **weak axis，case-series**：可比较但全程打 case-series caveat。
- `Δ_AB ≥ 120°` → **direction-pair interpretable**：方向比较可解释。
- 同时报每模板 `gradient_reliability`（||β|| 或 R²）+ `n_valid`；任一模板 `n_valid < 6` 或拟合退化 → 降为 diagnostic-only + flag。

---

## 5. 报告口径（分档；门即结论，§6.3 代词纪律）

**一个被试支持"发作方向分两类、且两类分别对上两条间期路线"——当且仅当三条全绿：**
`Δ_AB ≥ 120°（interpretable）` ∧ `two_class_eligible` ∧ `p_align < 0.05`。

其余分档（强约束措辞）：
- interpretable ∧ two_class_eligible ∧ `p_align ≥ 0.05` → "存在两个发作方向类，但其与间期模板的对应未超随机"。
- interpretable ∧ **¬two_class_eligible** → **只准写"一个发作主方向，与某一条间期路线方向（轴向）对齐/不对齐"；禁止"两类"措辞**。
- weak axis（60–120°）→ 上述结论全部加 case-series caveat。
- diagnostic-only（<60° 或模板退化）→ 不进方向比较，仅诊断。
- SEEG/近一维 → 仅 caveat 灵敏度层，不进任何描述性汇总。

**队列层**：只出**描述性分档表**（各被试 tier + p_bimodal + p_align + 关系量）+ 每被试图。**不出 pooled p、不下队列断言**。可描述性报"interpretable∧eligible∧p_align<0.05 的被试数 vs 其余"，但明说 n 小、不作主张。**禁止**把任一被试或全队列写成"发作传播有两类 / 重放间期路线"。

---

## 6. 复用与新代码（不重造，§6/§6.1）

**复用（question-match 已核）**：`gradient_angle / circular_mean / resultant_length / axial_resultant_length / axial_mean / axial_distance / rotate_to_reference`（`src/topic5_axis_direction.py`）；`_load_frame / _seizure_angles / _electrode_kind`（`scripts/plot_topic5_axis_direction_rose.py`，runner import）；间期模板场 `_t_{a,b}.json`。

**新模块 `src/topic5_directional_replay.py`（纯函数）**：
- `plane_fit_direction(x,y,values) -> (angle, grad_norm, r2, n_valid)`（扩展 gradient_angle，多返回拟合质量；不改原函数）
- `cluster_directions_k2(angles, seed=0) -> dict(labels, means, sizes, class_R, R_dir, R_axial)`
- `silhouette_unit(angles, labels) -> float`
- `kappa_from_R(R) -> float`（Mardia-Jupp）
- `unimodal_null_pvalue(angles, B=2000, seed) -> (p_bimodal, S_obs)`（H0=主方向+均匀背景散点；信号=第二个集中模；纯单峰 null 太弱已弃）
- `bootstrap_label_stability(angles, B=500, seed) -> ari_median`
- `two_class_eligible(n_sz, sizes, p_bimodal, stability, *, bimodal_alpha=0.05, stab_min=0.5) -> (bool, reasons)`（硬数量门 n_sz≥6/min size≥3 + p_bimodal<bimodal_alpha + stability≥stab_min）
- `axis_quality_tier(delta_ab_rad, n_valid_a, n_valid_b, *, interp_min_deg=120, weak_min_deg=60, min_valid=6) -> str`（'interpretable'|'weak_axis'|'diagnostic_only'；任一模板 n_valid<min_valid → 'diagnostic_only'）
- `best_pair_residual(class_means, template_dirs) -> (resid, pairing)`（exchange-invariant，straight/crossed 取小）
- `best_pair_rotation_null(class_means, template_dirs, B=2000, seed) -> p_align`

**runner `scripts/run_topic5_directional_replay.py`**：`--subjects --activation {broadband,hfa} --include-seeg`；每被试出 per_subject JSON（含全部门量 + tier + provenance）。

**图**：每被试玫瑰（发作按 ictal 两类着色 + 叠 θ_A/θ_B 两条间期方向线 + tier 角注）；队列分档表（CSV/JSON + 一张 tier 摘要图）。**`figures/README.md` 中文必写**。

**输出目录**：`results/topic5_ictal_recruitment/directional_clustering/{per_subject/, cohort_summary_{broadband,hfa}.{json,csv}, figures/{README.md,...}}`（结果 gitignore，按仓库惯例）。

---

## 7. TDD 合同（实现前先写，红→绿）

1. **P0 回归（命脉）**：纯单峰 von Mises 合成（n=22）→ `two_class_eligible==False`（算法不得制造两类）。
2. 平衡双峰合成（15:7，±180°）→ `two_class_eligible==True`。
3. 偏斜双峰（17:5 真两极）→ True；单峰+少数散点（非真第二峰）→ False。
4. 旋转 null 标定：ictal 与模板随机无关朝向 → `p_align` 近均匀（不系统 <0.05）；ictal 贴模板 → `p_align` 小。
5. 轴质量阈边界：Δ_AB = 6/59/60/119/120/147° → 正确档（含边界）。
6. `best_pair_residual`：straight vs crossed 选对；c1↔c2、A↔B 对换不变。
7. 旋转保大小：刚体旋转后两类大小不变（容差 0）。
8. 数量门：n_sz=5 → 不合格；min class=2 → 不合格。
9. `kappa_from_R` 单调；R≈0/R≈1 退化不崩。
10. `plane_fit_direction`：退化场（<3 点/无梯度）→ angle=NaN + flag；正常场返回 grad_norm/r2/n_valid。
11. 复用核对：circular_mean/resultant 不重写（import 自 topic5_axis_direction）。

---

## 8. 验收（门即结论）

- 6 主队列被试 × 2 band 全跑出 per_subject JSON（含三道门全部量 + tier）。
- §7 全部测试绿（尤其测试 1：单峰不被判两类）。
- 每被试图 + `figures/README.md`（中文）生成且经用户目视。
- cohort 描述性分档表生成；**无任何 pooled 断言**；措辞符合 §5 分档约束（用户审）。
- 归档 doc 写入 `docs/archive/topic5/`，main doc `docs/topic5_seizure_subtyping.md` 仅加指针（不复制全表）。

---

## 9. 风险 / caveat / 层级

- **exploratory、描述性、无预设假设、无队列主张、不声称重放**。n 小（原型预估：interpretable 约 442/384/958，weak 约 548/583，diagnostic-only 1084）。
- 以往所有 replay 味检验（A 线、echo gate 1/2/2b）都收在"共享粗锚非逐点重放"；本条最可能多数落"单峰主方向"，少数才谈两类。
- von Mises 为单峰参照（假设）；真单峰若非 von Mises，p_bimodal 略有偏差——故 P0 用"count + 单峰 null + bootstrap 稳定"三重，不靠单一证据。
- SEEG 投影方向不可靠 → 主队列只 ECoG。
- §6.3 代词纪律：凡"两类"措辞必须三门全绿；否则只准"主方向"。

## 10. 不做什么（scope）

- 不做 SOZ 锚点 / 不预设正负方向 / 不做带符号"重放"检验（用户已否决）。
- 不用 z-ER 5-bin 张量作方向源（用 A 线激活场）。
- 不跨 band 合并解释。不作因果声称。不把 n 小的描述升格为队列断言。
