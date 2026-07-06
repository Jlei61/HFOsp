# TA/TB 间期传播场 signed 反向门 — 设计 spec

> 日期 2026-07-06 · Topic 5 network-axis (V1) supplement · 状态：design **rev1（已并入 spec-review：P0 shared-frame + P1×3 + 边界收紧 + 工程 TDD）**，待用户 spec review → writing-plans

## 0. 一句话

把每个病人的两套间期高频传播模板（A/B）各自摊成一张**空间场**，正式检验这两张场是不是**空间上一负一正地反过来**——而且反得**比「同杆内把触点重排一遍」能给的更狠**。真正想回答的上游问题是：**把离散电极序列铺成加权空间场，是不是对传播方向估计的一种「去噪」，因而更鲁棒？**

## 1. 动机（第一性原理，含 1146 机制示意）

离散电极秩**对跨杆物理邻近是瞎的**。举例（motivating case = 1146 布局）：A 杆头几个触点和 B 杆头几个触点在空间上挨着——都贴着两杆之间的入口区——但在秩空间里它们只是「A 的某 rank + B 的某 rank」，没有邻近信息。于是对原始电极序列拟合方向可能读成「A 杆 → B 杆」，而**几何一致的读法**是入口在两杆之间、主路径沿「A 头 → A 尾」。**带坐标的加权平滑知道 A-early 与 B-early 是邻居**，会把它们汇成两杆之间一个 early 热点，给出**几何一致的 candidate physical-axis readout**。这是秩做不到、场能做到的事。（措辞纪律：写「几何一致的 field 轴估计 / candidate physical-axis」，**不写「真轴 / 真梯度 / ground truth」**——见 §9。）

**为什么值得单独做**：A-line 主线只做了「间期模板 A vs 发作早期激活」，且刻意**符号自由**（不判方向重放，`topic5_seizure_subtyping.md:69`）。它从未把「TA 与 TB 两个间期模板彼此是不是反的」做成检验。现有 swap 节点图（event-resolved pilot 2026-06-25）明确是「纯展示、无统计」。本 spec 补这个上游 gate。

**与 contact-similarity ladder 的关系（防混淆）**：ladder 测的是**间期↔发作相似度的幅值**（加平滑→观测与 null 一起抬→过 null 数没涨），讲的是**幅值膨胀**。本 spec 的去噪假设是**另一回事**：平滑能不能把**方向估计的方差降下来**。两者正交；ladder 对去噪假设未证过。**但幅值膨胀这条在「反向 corr 幅值」上确实会转移**——所以去噪的干净证据用**方差/可复现性**（§6），不用幅值。

## 2. 科学问题、假设、结论边界（验收 gate 编码结论）

- **H_primary（反向门）**：per-subject，signed corr(TA_field, TB_field) 显著**为负**，且强于 within-shaft 重排 null。
- **H_supplement（去噪可复现性）**：场的 held-out 预测比触点更准（§6）。这是「更鲁棒」字面意义上、且**免疫幅值膨胀**的直接检验。

**三档结论 + 边界红线（写结果时逐字守）**：
- **过**（field 版 K/n 病人过自己 within-shaft null，分底物二项显著）：可写「TA/TB 间期传播场在共享空间轴上反向对齐，超出同杆重排——**TA/TB 是同一根空间 scaffold 的两个相反遍历方向**，membership-robust 的几何 readout」。
- **不过**：「反向对齐落在空间自相关预期内——forward/reverse 配对在场层部分是空间平滑 artifact」。**有价值的阴性**，非「无信号」。
- **「场更鲁棒吗」= 被测子问题**，由 (i) field vs contact 过 null 病人数差（§5）+ (ii) 可复现性配对（§6）回答。field 更差（ladder 先验 6→5→4）→ **如实写「场没买到鲁棒性、只抬了原始数值」**。
- **禁写（即使反向门过）**：❌「发作会选择 TA 或 TB 极性」；❌「field 证明真实传播方向 / ground truth 方向」；❌ 方向重放；❌ 证明两模板存在（PR-2 gap_perm 已干）；❌ 有效刻画病理网络。**上述极性/真方向类主张须 supplement（§6）AND 后续独立 ictal-polarity 检验都过才解锁**，本 spec 不含 ictal-polarity 检验。

## 3. 统计对象与度量（含 P0 shared-frame 硬合同）

### 3.1 【P0 硬合同】subject-level shared physical frame
主统计**必须先构造 subject 级共享物理 frame**，TA 与 TB 两张场都建在**同一组 contact x/y、同一 grid X/Y、同一 sigma、同一 s_thresh** 上。
- **禁止**：直接对 `plane_a` 与 `plane_b`（两个不同 normalized frame）各建场再 corr——`field_from_contact_values` 按传入 plane_record 自带坐标建场（`topic5_event_resolved_alignment.py:501`），A/B 用各自 template plane 会让负相关来自**坐标定义**而非生理反向。
- **合同（stat frame = normalized readout coords，非 plotting mm frame）**：`_subject_display_frame` 是 **plotting-only**——`plot_contact_plane_static.py:74` 明写「This is a plotting frame only; comparison metrics stay in normalized readout coords」，**不拿它当统计 frame**。统计共享 frame = **参考模板 t_a 的 normalized readout plane**（A/B 值都摆到 t_a 的 contact x_norm/y_norm）；更对称的替代 = 由 A∪B 触点真实坐标一次性投影出的 subject-level normalized frame（且 dovetail §8 R2b native-3D）——二选一由 plan 钉，**不变量 = A/B 落同一 normalized frame**。
- **stat 与 figure 锚同一 template-A 轴、但不同坐标实现**（stat=normalized readout，figure=mm display）——这是既有设计（`plot_contact_plane_static.py:74`），**不强求同一 builder**。
- **同一 channel 即使在 A/B record 里坐标不同，也必须落到这一个 shared frame 的同一 (x,y)**；参考 plane 缺席的通道如何处理（限交集 / 附真坐标重投影）由 plan 钉。

### 3.2 值与度量
- **TA/TB 逐触点值 = `class_aggregate_contact_values(bundle, label)[value]`**：= 该类事件上 masked normalized rank 的 class mean（`topic5_event_resolved_alignment.py:322`，非参与触点 NaN；`support` = 参与比例，作平滑权重）。**主值源 = class-aggregate 场**（与现有 TA/TB 场图同构，Route B 连贯）；`template_rank` 0.985 等价，仅 sensitivity 备选。
- **【P1】metric 用上面的 raw value（masked normalized rank 的 class mean），绝不用 display-time `_rank01`**（`_rank01` 只用于 figure display，`plot_...fields.py:15`）。误用图层 rank01 会改相关与 null，小通道 subject 尤其被离散重排放大。
- **度量 = signed Pearson corr(TA_field, TB_field)**，只在两场支撑都够（S≥s_thresh，默认 S_THRESH=0.15）的**共享网格 mask** 上算，测**负尾**。**无 mirror**（同一物理 frame、同一 y 约定，y-翻转会比较物理上不同的东西）。实现 = 复用 `_support_corr` 的 identity orientation（`corr_pair_mirror_invariant_signed` 因带 mirror-max 不直接用，只借内部 primitive）。
- **成员不匹配天然由支撑交集 mask 处理**：A 有 B 无的触点只进 TA 支撑，corr 在交集算；overlap < OVERLAP_MIN → 该被试 `insufficient_overlap`，不进推断计数。

## 4. Null 三件套 + 队列聚合

- **within-shaft（推断主 null）**：`within_shaft_shuffle` 打乱 **TB** 的触点值、**TA 固定、坐标/support/names 全不动**，重建 TB 场重算 signed corr；B 次抽。带 `effective_shuffle_n` **弱-null 守卫**——**按实际进入统计的 finite TB channels / shaft groups 算 effective_n，不按原始全部 channels 算**（P5）；退化被试标 `degenerate_null`，不进推断计数（仍描述报）。
  - *为何 within-shaft 恰是对仪器*：打乱杆**内**、保留杆**间**。**沿杆反向**（A 头→A 尾 翻成 A 尾→A 头）被 null 打散→真信号能赢；平滑把沿杆梯度估得更干净→场赢 null 比触点多，**直接量到沿杆那档去噪**。**跨杆几何纠正**（入口挪两杆中间）被 null 保留→这档 within-shaft 测不到，交给 §6。
- **channel-shuffle（粗底参照）**：`channel_shuffle` 全触点打乱，给「有没有任何粗共享结构」地板。
- **random re-split（仅描述对照，非推断）**：忽略 A/B 标签随机分两半、各建场、corr(half1,half2)；分布≈正，实测 A/B≈负，一眼看出反向不是「切两半」artifact。**标 non-inferential**（不付 KMeans「最分离一刀」选择成本，当推断 null 反保守；「存不存在两模板」是 PR-2 已答，不在此重打）。
- **per-subject**：观测 corr 的 null 左尾 percentile（`placement_in_distribution`）；pass = percentile < 5 **且** corr < 0。
- **cohort**：二项检验 pass 数 vs 5%；**broad / narrow 各报、永不 pool**；多模板被试 `subject_first_fold` 折成一代表值防 pseudo-replication。无 band sweep → **不需 selcorr 选择校正**。
- **报告铁律**：实测 corr 永远跟 within-shaft null 带一起出，**不出裸 corr**。

## 5. 表示层 head-to-head（Route B：场主 + 触点灵敏度）

同一 signed-corr + within-shaft null 流程跑两个表示：
- **field（头条）**：§3 的 shared-frame 平滑场版。
- **contact（灵敏度）**：无几何，直接对 TA/TB 逐触点值算 **signed Spearman**（rank 数据自然度量）+ 同构 within-shaft null。
- **「场更鲁棒吗」= field 过 null 病人数 − contact 过 null 病人数**（及 per-subject 超-null 余量的配对差）。**比较落在「各自超不超自己的 null」层，不比原始 corr 幅值**（故 field-Pearson vs contact-Spearman 度量不同不影响，与 ladder R1/R2/R3 同做法）。把你原来的假设从「预设」变成**逐层被测结果**。

## 6. Supplement：去噪可复现性（【P1】预锁定义，免疫幅值膨胀）

反向 corr **幅值**（场更接近 −1）单独不能证去噪——平滑机械地把幅值往 ±1 推。干净证据 = **held-out 预测精度**：

**预锁定义**（每个 class、每个 split）：
1. **train-half** 建两个预测器：(a) **raw contact vector** = train 半各触点 mean rank；(b) **shared-frame smoothed field**（§3 同 frame/grid/sigma/s_thresh）。
2. **held-out half** 求各触点 mean rank，作**预测目标**。
3. 比较两预测器对 held-out per-contact mean rank 的精度（跨触点 Spearman）：
   - **contact 预测** contact c = train-c raw mean rank；
   - **field 预测** contact c = **LOO 场值**——用 train 半**剔除 c 本身**重建的平滑场在 c 位置取值（**必须 LOO 排除目标 contact，杜绝自我平滑泄漏**）。
4. **两预测器的跨触点 Spearman 必须在「两者都有定义」的同一触点集上算**——LOO 场值在**空间孤立触点**（无近邻支撑、S<s_thresh）处为 NaN 被剔，若 field 只在子集算、contact 在全集算就是**静默偏置**。取交集触点。
5. **subject 内先折叠 A/B**（两类读数取代表值），**再 cohort 配对 Wilcoxon**（field 精度 > contact 精度?）。
- 结构可复用 `split_half_axis_validation`（`propagation_skeleton_geometry.py:535`）的分半 + bootstrap 骨架，但**预测器是 LOO 平滑场值、非 spatial-axis 投影**（那是不同的量，不照搬）。
- **tier = supplement**，回答「更鲁棒」的机制侧，非 primary cohort claim。

## 7. 带宽旋钮（回应「之前平滑范围和 field 有差别」）

- **主 sigma = median 最近邻触点间距**（`smooth_field` 默认 `_median_nn_spacing`）。
- **sensitivity 小扫** `{0.5, 1, 2} × median_nn`，报去噪结论对带宽稳不稳；**主结论只认主 sigma**（防多重比较 fishing）。
- 注：ladder R2 复用了场的带宽，故与本检验差的**更多是「问的问题」、不一定是尺度**；带宽仍由本 spec 显式钉。

## 8. 合格口径 / 队列风险

- **合格** = 同 event-resolved：需 A/B 两模板（k=2）+ 触点坐标。broad ≈ 9–11 / narrow ≈ 7（精确 n 由 run 时合格门定）。
- **⚠️ narrow = compact-core**（触点挤、可能 1–2 杆）→ within-shaft null **高风险大面积撞 `effective_shuffle_n` 守卫**。处理 = 守卫统一施加，**由非退化被试数自动定 narrow tier**：非退化太少 → narrow 自动降 case-series（结论编码进 QC 门，不预先拍板 run/不run）。narrow 另出 native-3D（R2b 式）对照防 2D 投影 artifact（sensitivity）。

## 9. 1146 机制个案图（【P1】措辞收紧）

挑一个原始电极方向被几何带偏、场给出几何一致 axis readout 的病人，单图展示：左 = 原始触点值 + 拟合方向；右 = shared-frame 平滑场 + **candidate physical-axis readout**（入口区落在两杆之间）。
- **措辞**：写「几何一致的 field 轴估计 / candidate physical-axis readout」，**禁写「真轴 / 真梯度 / ground truth」**。
- **⚠️ 1146 broad record 自身 `poor_planarity=True`**——只作**机制示意**，不作真值证明。**定稿前先拿真数据核现象**；若 1146 不支持，**换 subject 或删图，不保留预设叙事**。

## 10. 复用 / 新写（§6.1 已核，构造对齐非签名对齐）

**复用**：
- **shared-frame（§3.1 P0，只借坐标构造、不借平滑、不借 plotting mm frame）**：参考模板 normalized readout plane（reuse readout record 的 x_norm/y_norm；如需 A∪B 真坐标一次性投影，把该构造 lift 进 src、**不复制**）。
- **【P1 硬合同】统计 smoothing ≠ plotting smoothing**：主统计场必须走**参数化** `smooth_field`/`R_smooth_rank`（默认 `sigma=median_nn`、`s_thresh=S_THRESH=0.15`），**禁用** plotting 的 `_smooth_rank_field_mm` 及其 display 常量 `VIS_SIGMA_MULT=2.5` / `VIS_SIGMA_MIN_MM=6.0` / `VIS_MASK_REL=0.02`（`plot_contact_plane_static.py:24-28`）。否则主统计静默变成「2.5×median-nn + 6mm floor + display mask」，field/pass 数不可解释。plotting 图可继续用 display 常量。
- `smooth_field` / `R_smooth_rank`（sigma 默认 median-nn）+ `_support_corr`（signed 无-mirror primitive）— `propagation_contact_plane_readout`
- `within_shaft_shuffle` / `channel_shuffle` / `effective_shuffle_n` — `topic5_axis_alignment`
- `class_aggregate_contact_values` / `field_from_contact_values` / `load_event_labels_ranks` / `map_clusters_to_templates` — `topic5_event_resolved_alignment`
- `split_half_axis_validation` 分半骨架 — `propagation_skeleton_geometry`
- `subject_first_fold` / `placement_in_distribution` — `propagation_contact_plane_readout`

**新写（小，主要编排）**：shared-frame 双场 signed-corr（薄封装 `_support_corr`）；within-shaft 反向门 harness（**null 只 shuffle TB value，不改坐标/support/names**）；contact head-to-head + random-split 描述对照；§6 LOO-field-vs-contact 可复现性。

## 11. 工件

- `src/topic5_field_reversal.py` + `tests/test_topic5_field_reversal.py`
- `scripts/run_topic5_field_reversal.py`（`--substrate broad/narrow`、`--subjects`、`--sigma-sweep`；**拒绝隐式 cohort run**）
- `scripts/plot_topic5_field_reversal.py`（per-subject：TA场|TB场|corr+null 直方+random-split 对照；cohort null forest；1146 个案；head-to-head 与 supplement 图）
- `results/topic5_ictal_recruitment/field_reversal/`：`per_subject/*.json` + `cohort_summary_{substrate}.json` + `figures/README.md`（中文逐图说明）

## 12. TDD 不变量清单（每条一测，CLAUDE.md §6 + 工程 P5）

1. **shared-frame【P0】**：同一 channel 在 A/B record 坐标不同，也落到同一 shared frame 同一 (x,y)；TA/TB **共用同一 sigma、grid X/Y、s_thresh**。
2. **signed 无-mirror**：TA vs y-翻转-TB 的结果 ≠ TA vs TB；返回带符号值；仅共享支撑像素；overlap < min → `insufficient_overlap`。
3. **metric 用 raw value 非 rank01【P1】**：喂 raw class-mean 与喂 rank01 结果不同，实现走 raw。
4. **within-shaft**：保杆内多重集、never 跨杆；**只 shuffle TB value，坐标/support/names 不变**、TA 不动。
5. **effective_n【P5】**：按**实际进入统计的 finite TB channels / shaft groups** 算；单杆/单例重 → `degenerate_null` 剔出推断、仍描述报。
6. **负尾三情形**：观测强负 vs null≈0 → 低 percentile/pass；≈0 → ~50pct；正 → 高 pct/fail。
7. **成员不匹配**：ch 在 A 不在 B → 只进 TA 支撑，corr 在交集算，不崩。
8. **random-split**：合成单峰事件池 → 分布居正、观测 A/B 负、清晰分离；标 non-inferential。
9. **cohort**：二项 pass 数正确；`subject_first_fold` 折多模板不重复计数。
10. **supplement LOO【P1】**：合成「邻居比自身更可靠」案 → field-LOO held-out 精度 > contact 精度；LOO 确实剔除目标 contact（含目标 c 与不含 c 的 field 预测不同）。
11. **带宽**：多 sigma 跑通；主 = median-nn。
12. **统计 smoothing 不吃 display 常量【P1】**：主统计场 `sigma == median_nn`（**未乘 2.5**、无 6mm floor）、`s_thresh == 0.15`（统计合同值）；断言主统计路径**不经过** `_smooth_rank_field_mm` / `VIS_*`。

## 13. Logistics

- 科学上属 network-axis (V1) supplement。**建议 base = local `main`**（有 contact_similarity + event-resolved + 场机件 + readout normalized 坐标 + plotting mm frame〔仅图〕），别堆在 `topic5-v2-phase1`。plan 时确认 event-resolved + within-shaft null + split-half 骨架 + readout plane 构造在 main 齐，缺则 cherry-pick。

## 14. 我替你设的默认（请在 spec review 时确认或红线）

- **已锁为硬合同（非默认）**：P0 shared-frame（§3.1）；metric 用 raw class-mean 非 rank01（§3.2）；§6 supplement LOO 定义；边界红线（§2）。
- **仍待你拍板的默认**：
  - **narrow tier 由 effective_n 守卫自动定**（退化多则降 case-series）。
  - **主 sigma = median-nn**，sweep {0.5,1,2}× 仅 sensitivity。
  - **1146 定稿前先核现象**；不支持则换/删（§9）。
  - **primary = 反向门（§4–5），supplement = 可复现性（§6）**，结论开放不预设方向。
