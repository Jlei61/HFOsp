# SEF-HFO 间期传播几何骨架 — 一轮结果归档（2026-06-08）

> **状态**：探索性、**描述性测量（给模型喂尺度数），不是假设检验，不是 SOZ 定位器**。无 held-out，archive-only，不得把任何数字升级进 topic4 主文档的 paper 档。
> **上游**：`docs/topic4_sef_itp_framework.md`（SEF-HFO v0.2）；数据侧 plan `docs/superpowers/plans/2026-06-06-sef-hfo-soz-localization-rate-vs-geometry.md`；low-rate 读回结论 `docs/archive/topic4/sef_hfo/low_rate_template_stability_2026-06-07.md`。
> **硬前置**：`lagPatRank` phantom-mask（Topic 0 §3.1）——源/汇 core 只用 phantom-safe 的稠密排名（非参与通道为 NaN，无法被选进 core；实测 `phantom_core_violations=0`）。

---

## 朴素话摘要（测了什么 / 怎么测的 / 揭示了什么）

**测了什么。** 每个病人的间期 HFO（高频振荡）事件里，通道有一个稳定的放电先后顺序——谁先点着、谁最后停。这一轮我们纯几何地量这条"传播骨架"在脑子里长什么样：起点一小撮通道、终点一小撮通道各自离自己重心有多散（半径）、从起点到终点这条传播主轴有多长、参与通道除了沿主轴还往两侧离轴铺多宽（注意这是"参与通道"的铺展，是采样+发放率框住的**下界**，不是真实斑块/耦合宽度），以及沿着主轴往前走、放电先后这套刻板顺序能延伸多远。目的只有一个：把这几个空间尺度数（起止斑块的大小、传播轴的长度、横向铺展的尺度）喂给 SEF-HFO 这个易激场模型当输入参数。我们**没有**用它去定位致痫区（早前已确证稳定起止端≠临床致痫区），也**没有**在这一轮检验"几何顺序比发放计数更稳"（那是下一轮）。

**怎么测的。** 先在每个病人里找出最早放电的一小撮通道（起点）和最晚放电的一小撮通道（终点），用它们各自的几何重心连一条线当传播主轴，再把每个通道投影到这条轴上拆成两个分量：沿轴走多远、偏离轴多远。起止 core 的半径=这一小撮通道离自己重心的均方根距离（**注意：本轮没做"随机抽同样多通道"的空间采样 null，所以只报半径绝对值、不断言"紧凑"**——小半径可能部分来自电极触点间距）；参与通道的离轴铺展（participating-channel perpendicular spread）=所有参与通道偏离主轴的垂直距离的均方根——这是采样+发放率框住的**下界，不是真实斑块宽度或耦合宽度**。沿轴的刻板性这样比：如果通道放电先后纯属随机配对，刻板程度应该是某个随机基线值；我们报实测值减去这个随机基线（这个差是"超出随机的量"，跟事件数多少无关，所以不同通道、不同病人之间能直接比，不会被高发放率的通道虚高）。可测与否设了个数值门：既参与传播又有有效三维坐标的通道数够多（≥7）才进主档，少一点（5 或 6 个）进兜底档，太少（<5）只做描述、不算几何尺度。

**揭示了什么。** 在这个尺度上看，传播骨架是厘米级的——传播主轴长度的中位数大约 18–22 毫米；起止 core 的半径在 Yuquan 较小（中位约 3 毫米）、在 Epilepsiae 较大（约 7–9 毫米），但"是否比随机选通道更紧凑"**未经 null 检验**（小半径可能部分来自电极触点间距，见 §4），且有 4 个 Epilepsiae 被试起止 core 沿轴交错使轴长不可信（见 §2）；多数可分类被试（21/26）的传播不是沿一根电极杆的一维直线，而是跨多根杆的 distributed 采样（其中真正落在**单一电极杆**上的只有 **2 个**被试）；沿轴的刻板顺序整条轴上都为正（但这一条**部分是模板定义出来的**，是边界 sanity 不是独立证据）。这些都是给模型用的尺度数——为模型的传播长度参数提供厘米级的经验尺度约束（不是毫米点源、也不是全脑），但**不独立支持任何具体机制**，也**不是**结论性的临床主张。

（内部归档代号：SEF-HFO v0.2、rank-displacement `swap_class`、phantom-safe `rank_a_dense_full`、`eligibility_tier=primary/fallback/descriptive_only`、`sampling_geometry=1D/distributed`、`template_source=dominant_cluster_*/full_recording`、`stereotypy_excess=raw_excess_obs_minus_nullmean`、`perp_width_measurable`、PR-6 H1 endpoint≠SOZ NULL）

---

## 1. 队列漏斗（每一步都显式留痕，不藏）

| 步骤 | 数 | 说明 |
|---|---|---|
| rank-displacement 有 phantom-safe 端点文件的被试 | 40 | `results/interictal_propagation_masked/rank_displacement/per_subject/` |
| − 事件质量门剔除坏数据 | −1 | `yuquan:pengzihang`（`total_hours=2.0`，记入 `cohort_summary.excluded_bad_data`） |
| − 缺三维坐标（无 `chnXyzDict.npy`） | −9 | 全部 Yuquan，**显式记入 `cohort_summary.errors`**（不静默丢弃）：chenziyang / gaolan / hanyuxuan / huanghanwen / litengsheng / sunyuanxin / wangyiyang / xuxinyi / zhangjinhan |
| **= 有几何输出** | **30** | `n_ok=30` |

**eligibility 分档（`tiers`）**：

- **primary 23**：参与且有坐标的通道数 ≥7 → 取起/止各 3 个通道做 core（`k=3`）。
- **fallback 3**：通道数 ∈ {5,6} → core 退到 2 个通道（`k=2`）。
- **descriptive-only 4**：通道数 <5（`n_eff` 3–4，`k=0`）→ **不算任何几何尺度**（无轴长 / 无半径 / 无横展），只在队列里计数。下面四个几何尺度因此是在 **26 个有轴被试（23 primary + 3 fallback）** 上算的，那 4 个 descriptive-only 不是"被中位数剔掉"，而是从一开始就没算几何量。

**采样几何（`sampling_geometry`，在 26 个有轴被试上）**：21/26 distributed（跨多根杆，可测离轴铺展）/ 5/26 "1D 采样"（近共线，离轴铺展不可测、已排除出横展统计）。注意 **"1D 采样" ≠ "单一电极杆"**：这 5 个里真正落在单一电极杆上的只有 **2 个**（`epilepsiae:139`、`yuquan:zhangjiaqi`，`n_shafts=1`），另 3 个（`yuquan:chengshuai` 2 杆、`epilepsiae:1077` 3 杆、`epilepsiae:620` 2 杆）是跨杆但近共线（`p90_off < 触点间距`，故横向不可测）。所以"间期时序模板落在同一根电极杆上"的被试是 **2 个**，不是 5 个。（通道名解析率 0/353 失败，故此计数不是解析失误造成的；`classify_sampling_geometry` 现带 >20% 未解析→`shaft_parse_uncertain` 守卫，本队列触发 0。）

**起止端路由（swap routing）**：12 个 swap-positive 被试（7 个候选层 + 5 个严格层）路由到主导聚类那条轴（`template_source=dominant_cluster_*`：9 个落 dominant cluster 0 + 3 个落 dominant cluster 1）；其余 18 个无稳定互换的用全程那条轴（`template_source=full_recording`）。上面这 12 / 18 是对全部 30 个 ok 被试计数；其中真正算出几何轴（有 `axis_length_mm`）的子集是 10（swap）+ 16（non-swap）= 26 个。

**phantom 安全**：`phantom_core_violations=0`——没有任何非参与的幽灵通道混进起/止 core。

---

## 2. 四个几何尺度（**按数据集分层，绝不 pool**）

> 两套数据的毫米**不可混比**：Yuquan 是病人真解剖 native mm；Epilepsiae 坐标是 MNI152 模板空间 mm（每个头被缩放进模板，warp 类型未独立核验）。只看各数据集内部的尺度量级与分布，不做跨数据集绝对数值比较。

| 尺度（中位数） | Yuquan SEEG（n_geom=7） | Epilepsiae SEEG+ECoG（n_geom=19） |
|---|---|---|
| 传播轴长度 | **18.3 mm**（range 10.5–57.2） | **22.3 mm**（range 1.8–46.4） |
| 起点 core 半径 RMS | **2.9 mm** | **6.7 mm** |
| 终点 core 半径 RMS | **2.9 mm** | **8.9 mm** |
| 参与通道横向铺展 RMS（仅可测者） | **13.6 mm**（可测 n=5；p90 中位 22.2） | **15.5 mm**（可测 n=16；p90 中位 21.4） |

横向铺展只在 distributed（非 1D 采样）被试上可测，故可测数（5+16=21）= distributed 总数。

**模态分层（采样混杂，必读 limitation）**：电极类型按通道命名推断（`G`+行列号=栅格；非 SQL ground-truth）。26 个有轴被试分三层：**Yuquan 纯 SEEG 深部 7** / **Epilepsiae 纯深部 15** / **Epilepsiae 深部+栅格 4**（`epilepsiae:1084/548/922/958`，有参与的 ECoG 栅格通道）。所以**上表 Epilepsiae 的 19 个其实是 15 纯深部 + 4 含栅格混在一起**。ECoG 栅格（皮层面二维、间距 ~11mm）与 SEEG 深部杆（~3.5–4.6mm）空间采样方式完全不同，直接影响 core 半径 / 离轴铺展 / distributed 判定——因此 Yuquan 与 Epilepsiae 之间的 core 半径差异（2.9 vs 6.7–8.9mm）**不能解释成生物学差异**，很可能是植入策略/采样模态差异。完整三层分层敏感性比较留下一轮。

**起止 core 跨杆情况**：起点 core 跨 >1 根电极杆的有 10/26 个被试，终点 core 跨 >1 根杆的有 12/26 个——再次说明多数传播不是沿单一杆的一维链，而是跨杆铺开。

**"最大两两距离 > 2.5×RMS"这道裂核护栏结构性失效，不能据此断言紧凑**：core 只有 k≤3 个通道，k 点集的"最大两两距离 / RMS 半径"比值有个解析上界 √6≈2.449 < 2.5，所以这个比值**永远不会触发**——0/26 不是"数据相当紧凑/没有裂核"的证据，而是这个判据对 k≤3 的 core 根本无信息量。MEB 与各通道坐标仍按被试保留（`source/sink_radius`、各通道 `along_axis_mm`），但**本队列无法从这个判据断言 core 紧凑性**。

**真正的裂核探测：4/26 被试起止 core 沿轴交错（弱约束轴）**：用 `sink_min_along < source_max_along`（即某个汇 core 通道投影到的沿轴位置比某个源 core 通道更靠前）检测起止 core 沿轴交错——这会让源/汇重心互相靠拢、重心间距（=轴长）被抵消，axis_length 不可信。命中 **4/26：`epilepsiae:635 / 620 / 1150 / 583`（全部 Epilepsiae；Yuquan 0/7 干净）**。以 635 为例：终点 3 个通道里有一个沿轴坐标在 −20.4 mm（落在起点*后方*），把终点重心拉回到几乎与起点重合，两个重心间距塌成 1.8 mm。

把 Epilepsiae 轴长中位数**两种口径都报，作为敏感性、不作为更正**：**22.3 mm（全 19 个）** vs **26.3 mm（剔掉这 4 个弱约束轴后的 15 个）**。注意布尔 `degenerate_axis`（L<1e-9）**一个都没抓到**这 4 个（它们 `degenerate_axis=false`）——新增的 `weak_axis` 标志（写进 per_subject + `cohort_summary.weak_axis` / `weak_axis_subjects`）才是下游模型代码应该 gate 的字段。

---

## 3. 图（用户需亲自目视）

均在 `results/topic4_sef_hfo/skeleton_geometry/figures/`，配套中文 `README.md`：

- **`axis_frame_examples.png`** — 6 个 primary/distributed 被试的传播骨架散点（横=离源沿轴距离 mm，纵=离轴垂直距离 mm，颜色=该通道刻板性超出随机的量，对称居中于 0）。看一个骨架在每个病人里"长什么样"。
- **`skeleton_scalars_by_dataset.png`** — 四联散点（轴长 / 起点半径 / 终点半径 / 横展），**按数据集分两列、不混**，每点一个被试。
- **`along_axis_stereotypy_profile.png`** — pooled 所有 primary+fallback 被试，沿轴方向的刻板性剖面（纵轴=超出随机的量，虚线=随机水平 0）。这张是延展性 / 边界 sanity，不是独立检验（见 §4 警告）。

---

## 4. 诚实警告（这一轮的纪律红线，保护本轮不被过度解读）

1. **"横向铺展"不是"斑块真实宽度"。** 它是**参与通道**偏离主轴的垂直铺展，受采样 + 发放率共同圈住——几何只能在有通道参与（=高发放）的地方测，所以它是斑块宽度的**下界**，不是上界。每个被试存了 participation-threshold sweep（`perp_spread_participation_sweep`）作为对照，必须连同这个 sweep 一起读。
2. **跨数据集毫米不可 pool。** Yuquan native 解剖 mm vs Epilepsiae MNI152 模板 mm（warp 类型未独立核验）；只在各数据集内部比较。
3. **沿轴刻板性部分是同义反复。** 通道的排名本身≈它的沿轴位置（按构造），所以沿轴刻板剖面是**边界 sanity**（看刻板模式越过端点之后是不是掉下去），**不是独立检验**。我们用的是跟事件数无关的"超出随机的量"（不是会被高发放率虚高的 z），所以 pool 本身是干净的——但"沿轴为正"这件事本身别当独立证据。
4. **起止端 ≠ 致痫区。** 稳定起止端与临床致痫区无显著关系早前已确证（PR-6 H1 NULL）——本轮所有与致痫区的关系只作描述，不下任何显著性主张。
5. **"紧凑"未经空间采样 null 检验。** 本轮只报起止 core 半径绝对值，没在同一被试同一 eligible 通道集里随机抽同样多通道做对照。小半径（尤其 Yuquan ~3mm）可能部分来自 SEEG 触点间距/植入密度，**不能断言起止 core "比随机通道更紧凑"**。随机-k null 留下一轮。
6. **模态采样混杂。** Epilepsiae 是 SEEG+ECoG 混合（类型从命名推断，非 ground-truth），Yuquan 纯 SEEG；栅格与深部杆采样方式不同。core 半径 / 离轴铺展 / distributed 判定都可能受植入模态影响——跨数据集、跨模态的几何差异**不得解释成生物学差异**（见 §2 模态分层）。
7. **厘米级尺度是校准约束，不是机制验证。** 18–22mm 的轴长说明模型空间尺度该设在厘米级（非毫米点源、非全脑），为模型传播长度参数提供经验约束；它**不能单独支持** SEF-HFO 的任何具体机制（若机制参数本就用相近数据调出，这只是 calibration consistency）。
8. **探索性、无 held-out、archive-only。** 不得把任何一个数字升级进 topic4 主文档 paper 档。

---

## 5. 出界 / 下一轮

- **几何 vs 发放 的稳定性对比（split-half）**：把数据折半，比较"传播顺序几何"和"发放计数"哪个在两半之间更稳——本轮没做（本轮只描述尺度，不比稳定性）。
- **起止 core 紧凑性的空间采样 null**：同一被试同一 eligible 通道集内随机抽 k 个通道 ×1000，检验真实 source/sink core 半径是否显著小于随机——才能把"半径小"升级成"比随机更紧凑"（否则可能只是触点间距）。
- **沿轴方向的独立检验（split-half rank-axis monotonicity）**：一半事件定义主轴、另一半验证 rank 沿轴单调——把"沿轴刻板为正"从同义反复升级成独立证据。
- **模态分层敏感性**：Yuquan SEEG / Epilepsiae 纯深部 / Epilepsiae 含栅格 三层分别比尺度，排除采样模态混杂（本轮只报了三层计数 + limitation，未做分层比较）。
- **离轴时序耦合宽度作为门控的队列尺度量**：横向上时序还协同到多宽，作为一个带 eligibility 门的队列标量——留下一轮。

---

## 链接

- spec：`docs/superpowers/specs/2026-06-08-sef-hfo-propagation-skeleton-geometry-design.md`
- plan：`docs/superpowers/plans/2026-06-08-propagation-skeleton-geometry.md`
- 数据侧总 plan：`[[project_topic4_soz_localization_plan]]`（`docs/superpowers/plans/2026-06-06-sef-hfo-soz-localization-rate-vs-geometry.md`）
- 代码：`src/propagation_skeleton_geometry.py`（纯几何，无 I/O）、`scripts/run_propagation_skeleton_geometry.py`、`scripts/plot_propagation_skeleton_geometry.py`
- 结果：`results/topic4_sef_hfo/skeleton_geometry/{cohort_summary.json, per_subject/, figures/}`
