# TA/TB 间期传播场 signed 反向门 — 设计 spec

> 日期 2026-07-06 · Topic 5 network-axis (V1) supplement · 状态：design **rev1（已并入 spec-review：P0 shared-frame + P1×3 + 边界收紧 + 工程 TDD）**，待用户 spec review → writing-plans

## 0. 一句话

把每个病人的两套间期高频传播模板（A/B）各自摊成一张**空间场**，正式检验这两张场是不是**空间上一负一正地反过来**——而且反得**比「同杆内把触点重排一遍」能给的更狠**。真正想回答的上游问题是：**把离散电极序列铺成加权空间场，是不是对传播方向估计的一种「去噪」，因而更鲁棒？**

## 1. 动机（第一性原理，含 1146 机制示意）

离散电极秩**对跨杆物理邻近是瞎的**。举例（motivating case = 1146 布局）：A 杆头几个触点和 B 杆头几个触点在空间上挨着——都贴着两杆之间的入口区——但在秩空间里它们只是「A 的某 rank + B 的某 rank」，没有邻近信息。于是对原始电极序列拟合方向可能读成「A 杆 → B 杆」，而**几何一致的读法**是入口在两杆之间、主路径沿「A 头 → A 尾」。**带坐标的加权平滑知道 A-early 与 B-early 是邻居**，会把它们汇成两杆之间一个 early 热点，给出**几何一致的 candidate physical-axis readout**。这是秩做不到、场能做到的事。（措辞纪律：写「几何一致的 field 轴估计 / candidate physical-axis」，**不写「真轴 / 真梯度 / ground truth」**——见 §9。）

**为什么值得单独做**：A-line 主线只做了「间期模板 A vs 发作早期激活」，且刻意**符号自由**（不判方向重放，`topic5_seizure_subtyping.md:69`）。它从未把「TA 与 TB 两个间期模板彼此是不是反的」做成检验。现有 swap 节点图（event-resolved pilot 2026-06-25）明确是「纯展示、无统计」。本 spec 补这个上游 gate。

**与 contact-similarity ladder 的关系（防混淆）**：ladder 测的是**间期↔发作相似度的幅值**（加平滑→观测与 null 一起抬→过 null 数没涨），讲的是**幅值膨胀**。本 spec 关心的「方向鲁棒性」是**另一回事**：读传播方向时，坐标读法是不是比坐标盲（杆折叠）读法更稳更对。两者正交；ladder 对此未证过。**幅值膨胀这条在「反向 corr 幅值」上会转移**——所以方向鲁棒性的干净证据用**轴的 held-out 方向预测（§6a）**；**per-contact LOO（§6b）只作逐点重建 sanity、不是方向检验**；都不用幅值。

## 2. 科学问题、假设、结论边界（验收 gate 编码结论）

- **H_primary（反向门）**：per-subject，signed corr(TA_field, TB_field) 显著**为负**，且强于 within-shaft 重排 null。
- **H_supplement（方向鲁棒性 = §6a axis-level）**：坐标读法的轴是否比坐标盲（杆折叠）读法在 held-out 上更好地预测发放方向——这才是「更鲁棒方向」的直接检验（免疫幅值膨胀）。**per-contact LOO（§6b）只是逐点重建 sanity，非方向鲁棒性。**

**三档结论 + 边界红线（写结果时逐字守）**：
- **过**（field 版 K/n 病人过自己 within-shaft null，分底物二项显著）：可写「TA/TB 间期传播场在共享空间轴上反向对齐，超出同杆重排——**TA/TB 是同一根空间 scaffold 的两个相反遍历方向**，membership-robust 的几何 readout」。
- **不过**：「反向对齐落在空间自相关预期内——forward/reverse 配对在场层部分是空间平滑 artifact」。**有价值的阴性**，非「无信号」。
- **「方向读法哪个更鲁棒」= 被测子问题**，由 **§6a 轴的 held-out 方向预测**（坐标 vs 坐标盲/杆折叠 vs 随机轴 null）回答。**per-contact LOO（§6b）不回答方向问题**（它测逐点重建，天然偏向高 SNR 的触点自均值）。
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
- **cohort**：二项检验 pass 数 vs 5%；**broad / narrow 各报、永不 pool（§8）**；每被试每底物一个 A/B 反向 corr（不需多模板折叠）。无 band sweep → **不需 selcorr 选择校正**。
- **报告铁律**：实测 corr 永远跟 within-shaft null 带一起出，**不出裸 corr**。

## 5. 表示层 head-to-head（Route B：场主 + 触点灵敏度）

同一 signed-corr + within-shaft null 流程跑两个表示：
- **field（头条）**：§3 的 shared-frame 平滑场版。
- **contact（灵敏度）**：无几何，直接对 TA/TB 逐触点值算 **signed Spearman**（rank 数据自然度量）+ 同构 within-shaft null。
- **「场更鲁棒吗」= field 过 null 病人数 − contact 过 null 病人数**（及 per-subject 超-null 余量的配对差）。**比较落在「各自超不超自己的 null」层，不比原始 corr 幅值**（故 field-Pearson vs contact-Spearman 度量不同不影响，与 ladder R1/R2/R3 同做法）。把你原来的假设从「预设」变成**逐层被测结果**。

## 6. Supplement：轴/方向鲁棒性（axis-level，robustness 正主）+ per-contact 重建 sanity（降级）

「场是否给更鲁棒的**传播方向**读出」= 原始动机的真问题。**两层不能混**：per-contact 重建（§6b，降级 sanity）≠ 轴/方向鲁棒性（§6a，robustness 正主）。**旧 §6（LOO）测的是前者，被误当后者——本 rev 纠正**（用户审阅 2026-07-06）。

### 6a. axis-level robustness（robustness 问题的正主）

**三种轴（同一 P0 共享 2D 平面）**：
- `sequence_axis`（**坐标盲**，专服务 1146 诊断）：只用电极序列 / 杆结构（**不用真坐标**）读出的方向——1146 失败模式（多个 early 触点分布两杆 → 误读「A 杆→B 杆」）就出在这。**具体读法在 pilot 定并在 1146 真数据上验证它确实复现 A→B 误读**（候选：按电极 identity 排 1D 序、rank~序位；或按 shaft 分组的 early/late 归属方向）。
- `raw_contact_axis`（**坐标 LS、不平滑**）：`rank_i ~ 1 + x_i + y_i` over contacts，加权 LS（权重 = 触点 support），β=(β_x,β_y)、early→late。
- `field_axis`（**先平滑再 LS**）：`T(x,y) ~ 1 + x + y` over supported grid pixels，加权 LS（权重 = field support S）。**同一估计器**——`raw_contact` 与 `field` **只差「是否先空间平滑」**，不差轴提取器。
- source/sink 质心差 = **diagnostic / 解释图**，非 primary（top/bottom quantile 易受少数触点 + 阈值影响）。

**【pilot 结论 2026-07-06（n=5 broad）→ 本 §6a 定向重构，走 Option-B】**
- **field vs raw_contact = 近平**（held-out 2/5 favor field、3/5 favor raw，最强反例 1077 是 raw 明显赢；`cos(raw_contact, field)` 0.94–1.00）——「平滑比坐标直线拟合更 robust」**不成立**（正中下方预警）。**保留为如实阴性**，不再作 primary。
- **`sequence_axis` 定义已锁** = 每触点折叠成其 shaft support-加权均值、再走同一加权 LS（「只看在哪根杆、丢杆内位置」= 坐标盲）；`epilepsiae_1146`、**尤其 1077** 上它与坐标-aware 轴分歧 49.6°/72.8°、70.4°/**148.8°**，其余 3 个 <20°。**`poor_planarity` 预测不了它**（1077 非 poor-planar 却最强；1125 是却没事）→ **个案按数据选（1077 为 primary case，不是 1146）**。

**Option-B cohort（本 §6a 实际主张；broad/narrow 分开、TA/TB 折叠、永不 pool）**：
- **primary = 坐标盲会不会误导 + 坐标-aware 能不能救**：(a) **分歧分布** = `angle(sequence_axis, raw_contact_axis)` 的 cohort 分布 + 大分歧（>45°、>90°）被试数/比例；(b) 坐标-aware 是否**泛化更好**（非只「不同」）= held-out ρ 配对 `raw_contact > sequence`（在大分歧子集上应显著）。held-out 分数 = `Spearman(沿 train-axis 投影, held-out per-contact mean rank)`。
- **secondary（如实阴性）= `field` vs `raw_contact` 近平** + `cos(raw_contact, field)` 高 → **「用坐标」就够，平滑不额外加分**。
- **结论红线**：主张 = **「读传播方向要用真实坐标；把每根杆内触点压成一个杆均值、丢掉杆内位置（坐标盲）再读，在部分被试（多杆入口几何）会严重误导」**（注：坐标盲 = **shaft-collapsed 后仍用真实坐标拟合**，**不是**「按电极名字/插入顺序排序」）；**不**主张「场平滑去噪 / 场更鲁棒 / 找到真实传播轴」。case 图 = `sequence_axis` vs `raw_contact_axis`（+field 叠加，示 field≈raw_contact），1077 primary + 1146。

**TA/TB 轴反平行（仅辅助，不替代 §3 反向主门）**：`cos(field_axis_TA, −field_axis_TB)` 中位 ≈ broad 0.61 / narrow 0.67——只是轴层面的反向补充读数，**不够强到单独承载「反向对」结论**；反向主门仍是 §3 的 per-pixel signed corr vs within-shaft null。per subject + cohort。

**帧**：共享 2D 平面主；**native-3D 作 sensitivity**（1146 poor-planarity 个案必须在图/supplement 标 3D，防审稿）。**tier = supplement**（robustness 侧），broad/narrow 分开、TA/TB 分开、永不 pool。

**预期风险（如实标 tier）**：逐触点 rank 已很稳（§6b），`raw_contact` 轴可能本来也稳 → held-out/角度有可能**近平**；真正判别力大概率在 **1146 型几何一致性（bias）**，但它无 ground truth、cohort 指标须小心。**pilot-first**：先 1146 + 3–5 典型 subject，看 `field_axis` held-out 是否真 > `raw_contact`，且 1146 是否 `sequence_axis` 错 / `field_axis` 几何一致；pilot 过再 cohort。

### 6b. per-contact LOO 重建 sanity check（**降级**，原 §6，改名，**非 axis test**）

（原预锁 LOO 定义，Task 6 已实现，保留不变）每 split：train-half 建 (a) raw contact vector、(b) shared-frame 平滑场；held-out-half 求 per-contact mean rank；**contact 预测** = train-c raw mean rank，**field 预测** = LOO 场值（train 半剔除 c 本身重建、c 位置取值，`den<s_thresh` 剔）；两者在交集触点上跨触点 Spearman；subject 内折 A/B 再 cohort 配对。
- **这个检验天然偏向触点 self-mean**——每个触点的间期均值是**大量事件平均出来的高 SNR 量**，拿掉它、用邻居插值估它本质是插值，必糊掉 contact-specific 信息。它测的是**逐触点重建**，**不是**轴 / 方向鲁棒性（1146 那种失败模式它根本没碰）。
- **tier = sanity check**；结果**不叫「去噪」**（不写「denoising refuted」）。「场是否更鲁棒」的答案在 §6a。

## 7. 带宽旋钮（回应「之前平滑范围和 field 有差别」）

- **主 sigma = median 最近邻触点间距**（`smooth_field` 默认 `_median_nn_spacing`）。
- **sensitivity 小扫** `{0.5, 1, 2} × median_nn`，报去噪结论对带宽稳不稳；**主结论只认主 sigma**（防多重比较 fishing）。
- 注：ladder R2 复用了场的带宽，故与本检验差的**更多是「问的问题」、不一定是尺度**；带宽仍由本 spec 显式钉。

## 8. 合格口径 / 队列范围（broad + narrow 双底物）

- **两底物都做，broad-vs-narrow = 核心 sensitivity**。narrow = compact-core：**若 broad 反向而 narrow 不反向，机制解释完全不同**（反向是粗/远场现象、非核心属性）；若 narrow 也反向，主张更强、深到 SOZ 核。**narrow 退化（compact/few-shaft 撞弱-null 守卫）本身是要报告的结果，不是预先跳过**——把「loader 没建」当「narrow 不可行」= 把工程现状写成数据结论，禁止。
- **narrow loader（Task 0）**：`load_event_labels_ranks(broad=False)` 原被 event-resolved pilot stub 成 `NotImplementedError`（pilot 自身 scoping，**非数据墙**）。**分层分母**：narrow planes=**35**（raw availability）；过 `stable_k==2` pre-map=**29**；broad 对应=**26**；**最终 inferential N 更小**（再过 cluster_map/overlap/degenerate，勿把 35 误读成 eligible N）。narrow labels=`results/interictal_propagation_masked/per_subject`，narrow geometry=`propagation_geometry/observation_readout/real_subjects/{ds_sid}_t_{a,b}.json`，narrow lagpat=canonical 池（`_subject_dir(dataset, root, subject)`，**dataset-specific root**：yuquan `YUQUAN_ROOT=/mnt/yuquan_data/yuquan_24h_edf`、epi `EPILEPSIAE_ROOT=/mnt/epilepsia_data/interilca_inter_results/all_data_lns`），**复用同一 C1 producer-template 证明作 loud 守卫**（错池 → C1 硬 raise，不静默）。
- **合格（每底物）** = 该底物 planes 存在 + `stable_k==chosen_k==2` + `map_clusters_to_templates` 不 ambiguous。**不需 ictal**（纯间期 TA-vs-TB）。broad / narrow **各报、永不 pool**。
- **退化守卫 + 逐被试问责**：`effective_shuffle_n` 按真正进统计的触点算；退化被试标 `degenerate_null`、剔出推断计数，但**每被试逐条报「为什么进/不进」**（`no_planes` / `c1_violation` / `cluster_map_ambiguous` / `insufficient_overlap` / `degenerate_null` / `ok`）——narrow 若大面积退化，这张问责表就是结果。

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
