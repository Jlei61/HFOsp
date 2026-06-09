# SEF-HFO 间期传播几何骨架 — 一轮结果归档（2026-06-08）

> **状态**：探索性、archive-only，不得把任何数字升级进 topic4 主文档的 paper 档。这一轮**两部分**：(a) 描述性几何尺度数（喂模型，不是 SOZ 定位器）；(b) **每被试内的留出半检验**——把事件折半、一半定轴、另一半验证"沿轴位置能否预测放电先后"，回答"这条传播是不是真·共享刻板通路，而非平均出来的同义反复"。(b) 是本轮相对前一版的关键升级（前一版只有描述、被批同义反复风险）。仍为探索性：留出是**被试内**的、无跨被试 held-out、无多重比较校正。
> **上游**：`docs/topic4_sef_itp_framework.md`（SEF-HFO v0.2）；数据侧 plan `docs/superpowers/plans/2026-06-06-sef-hfo-soz-localization-rate-vs-geometry.md`；low-rate 读回结论 `docs/archive/topic4/sef_hfo/low_rate_template_stability_2026-06-07.md`。
> **硬前置**：`lagPatRank` phantom-mask（Topic 0 §3.1）——源/汇 core 只用 phantom-safe 的稠密排名（非参与通道为 NaN，无法被选进 core；实测 `phantom_core_violations=0`）。

---

## 朴素话摘要（测了什么 / 怎么测的 / 揭示了什么）

**测了什么。** 每个病人的间期 HFO（高频振荡）事件里，通道有一个稳定的放电先后顺序——谁先点着、谁最后停。这一轮我们纯几何地量这条"传播骨架"在脑子里长什么样：起点一小撮通道、终点一小撮通道各自离自己重心有多散（半径）、从起点到终点这条传播主轴有多长、参与通道除了沿主轴还往两侧离轴铺多宽（注意这是"参与通道"的铺展，是采样+发放率框住的**下界**，不是真实斑块/耦合宽度），以及沿着主轴往前走、放电先后这套刻板顺序能延伸多远。目的有两个：把这几个空间尺度数（起止斑块的大小、传播轴的长度、横向铺展的尺度）喂给 SEF-HFO 这个易激场模型当输入参数；以及**检验这条"传播路线"是不是真的**——把事件折半，用一半的事件定出主轴，再看**另一半（没参与定轴的）事件**里通道的放电先后能不能被沿轴位置预测。我们**没有**用它去定位致痫区（早前已确证稳定起止端≠临床致痫区），也**没有**在这一轮检验"几何顺序比发放计数更稳"（那是下一轮）。

**怎么测的。** 先在每个病人里找出最早放电的一小撮通道（起点）和最晚放电的一小撮通道（终点）——用的是**上游已经验收过的聚类模板**来给每个事件归簇（不是临时重新聚类；旧版临时重聚类对"上游认定 4 簇"的被试强行压成 2 簇是错的，换掉后有个被试的轴长从 1.8mm 变成 19mm）。用起/止各自的几何重心连一条线当传播主轴，把每个通道投影到轴上拆成"沿轴走多远、偏离轴多远"。**三个检验**：
> 1. **起止紧凑度有没有意义**：core 半径=通道离自己重心的均方根距离；并在**同一被试**里随机抽同样多通道 1000 次比——只有真实半径明显小于随机才算"比随机紧"。
> 2. **横向铺展**（participating-channel perpendicular spread）=参与通道偏离主轴的垂直距离均方根——这是采样+发放率框住的**下界，不是真实斑块/耦合宽度**。
> 3. **这条路线是不是真的（留出半检验，本轮核心）**：把事件折半，一半算出主轴+各通道沿轴位置，另一半（**没参与定轴**）算各通道的放电先后；看两者的相关（Spearman/Kendall + 自助置信区间）。相关高=空间轴能预测它没见过的事件的放电顺序=真·共享通路；相关≈0=只是平均出来的同义反复。
>
> 可测与否设了个数值门：既参与传播又有坐标的通道 ≥7 进主档，5–6 进兜底档，<5 只描述。

**揭示了什么。** 最重要的一条（留出半检验）：**26 个有轴被试里，16 个的空间轴强预测留出半事件的放电先后（相关 ρ≥0.7），5 个中等（0.4–0.7），5 个弱/无（<0.4）；中位 ρ=0.75，25/26 的置信区间下界在随机之上。** 也就是说**多数被试（约 21/26）确实存在一条能推广到"没见过的事件"的共享刻板传播通路**，不是平均出来的同义反复——而那 5 个弱的，恰恰大多是起止通道沿轴交错的退化几何（如 635 ρ=0.11、1150 ρ=0.08）。其次，几何尺度：传播骨架厘米级（主轴中位 Yuquan 18mm / Epilepsiae 26mm）；起止 core 半径"比同被试随机通道更紧"的，源端 10/26、汇端 9/26（**两端差不多，没有强的"源聚汇散"队列规律**——那是个别被试如 958 的现象，别外推）；多数（23/26）是跨多根杆的 distributed 采样，真正落在**单一电极杆**上的只有 2 个。正反模板被试里 7/10 是"同一空间轴、两个相反方向"（cos≤−0.85），3 个更像两条岔开的路径。这些几何数为模型的传播长度参数提供厘米级经验约束（不是毫米点源、也不是全脑），但**不独立支持任何具体机制**，也**不是**临床主张。

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

**采样几何（`sampling_geometry`，26 个有轴被试，用上游已验收聚类后）**：23/26 distributed（跨多根杆，可测离轴铺展）/ 3/26 "1D 采样"（近共线，离轴铺展不可测、排除出横展统计）。注意 **"1D 采样" ≠ "单一电极杆"**：这 3 个里真正落在单一电极杆上的只有 **2 个**（`epilepsiae:139`、`yuquan:zhangjiaqi`，`n_shafts=1`），另 1 个（`epilepsiae:620`）是跨 2 杆但近共线（`p90_off < 触点间距`，横向不可测）。所以"间期时序模板落在同一根电极杆上"的被试是 **2 个**。（通道名解析 0/353 失败，计数不是解析失误；`classify_sampling_geometry` 带 >20% 未解析→`shaft_parse_uncertain` 守卫，本队列触发 0。）

**聚类归簇 provenance**：每个事件用**上游已验收的聚类模板**按相关性归簇（`clustering_provenance=accepted_rankdisp_templates`），**不是临时重跑 KMeans**——这修掉了旧版对 2 个"上游认定 4 簇"的被试强压 2 簇的错误。

**起止端路由（swap routing）**：12 个 swap-positive 被试（7 候选 + 5 严格）用占多数那簇的轴（`template_source=dominant_cluster`，几何子集 10 个）；18 个无稳定互换的用全程轴（`full_recording`，几何子集 16 个）；10+16=26 个有轴。swap 被试还**并报次簇的轴**（`minority_axis` + `axes_cos_angle`），不藏 minority。

**phantom 安全**：`phantom_core_violations=0`——没有任何非参与的幽灵通道混进起/止 core。

---

## 2. 四个几何尺度（**按数据集分层，绝不 pool**）

> 两套数据的毫米**不可混比**：Yuquan 是病人真解剖 native mm；Epilepsiae 坐标是 MNI152 模板空间 mm（每个头被缩放进模板，warp 类型未独立核验）。只看各数据集内部的尺度量级与分布，不做跨数据集绝对数值比较。

| 尺度（中位数） | Yuquan SEEG（n_geom=7） | Epilepsiae SEEG+ECoG（n_geom=19） |
|---|---|---|
| 传播轴长度 | **18.3 mm**（range 3.7–70.1） | **25.8 mm**（range 9.8–41.6） |
| 起点 core 半径 RMS | **2.9 mm** | **7.5 mm** |
| 终点 core 半径 RMS | **7.6 mm** | **9.2 mm** |
| 参与通道横向铺展 RMS（仅可测者） | **12.3 mm**（可测 n=6） | **13.9 mm**（可测 n=17） |

横向铺展只在 distributed（非 1D 采样）被试上可测，故可测数（6+17=23）= distributed 总数。（数字为换用上游已验收聚类模板后的版本；与旧版临时 KMeans 相比有移动，例如某些被试的轴长/弱轴判定变了。）

**模态分层（采样混杂，必读 limitation）**：电极类型按通道命名推断（`G`+行列号=栅格；非 SQL ground-truth）。26 个有轴被试分三层：**Yuquan 纯 SEEG 深部 7** / **Epilepsiae 纯深部 15** / **Epilepsiae 深部+栅格 4**（`epilepsiae:1084/548/922/958`，有参与的 ECoG 栅格通道）。所以**上表 Epilepsiae 的 19 个其实是 15 纯深部 + 4 含栅格混在一起**。ECoG 栅格（皮层面二维、间距 ~11mm）与 SEEG 深部杆（~3.5–4.6mm）空间采样方式完全不同，直接影响 core 半径 / 离轴铺展 / distributed 判定——因此 Yuquan 与 Epilepsiae 之间的 core 半径差异（2.9 vs 7.5–9.2mm）**不能解释成生物学差异**，很可能是植入策略/采样模态差异。完整三层分层敏感性比较留下一轮。

**起止 core 跨杆情况**：起点 core 跨 >1 根电极杆的有 15/26 个被试，终点 core 跨 >1 根杆的也是 15/26——多数传播不是沿单一杆的一维链，而是跨杆铺开。

**"最大两两距离 > 2.5×RMS"这道裂核护栏结构性失效，不能据此断言紧凑**：core 只有 k≤3 个通道，k 点集的"最大两两距离 / RMS 半径"比值有个解析上界 √6≈2.449 < 2.5，所以这个比值**永远不会触发**——0/26 不是"数据相当紧凑/没有裂核"的证据，而是这个判据对 k≤3 的 core 根本无信息量。MEB 与各通道坐标仍按被试保留（`source/sink_radius`、各通道 `along_axis_mm`），但**本队列无法从这个判据断言 core 紧凑性**。

**真正的裂核探测：6/26 被试起止 core 沿轴交错（弱约束轴）**：用 `sink_min_along < source_max_along`（某个汇 core 通道投影到的沿轴位置比某个源 core 通道更靠前）检测——起止 core 沿轴交错会让源/汇重心互相靠拢、重心间距（=轴长）被抵消，axis_length 不可信。换用上游已验收聚类后命中 **6/26：`yuquan:chengshuai`（注意 Yuquan 也有了，旧版"Yuquan 0/7 干净"已失效）+ `epilepsiae:590/620/635/1125/1150`**。**关键诚实点**：弱轴标志与"留出半检验"大体一致——6 个里 1150（ρ=0.08）、635（ρ=0.11）、chengshuai（ρ=0.12）的通路确实没通过；但 **1125 起止 core 交错却仍 ρ=0.82**——所以"弱轴"是几何重心退化的标志，不等于"没有真通路"，最终该看留出半检验，不是只看弱轴标志。

把 Epilepsiae 轴长中位数**两种口径都报，作为敏感性、不作为更正**：**25.8 mm（全 19 个）** vs **27.0 mm（剔掉弱轴被试后）**。布尔 `degenerate_axis`（L<1e-9）**抓不到**这些（它们 `degenerate_axis=false`）——`weak_axis` 标志（写进 per_subject + `cohort_summary.weak_axis`/`weak_axis_subjects`）才是下游模型代码该 gate 的字段。

---

## 2.5 这条传播路线是真的吗？留出半检验 + 紧凑性 null + 正反模板（本轮核心新增）

**留出半检验（回答"是不是同义反复"）**：把每个被试的事件随机折半，一半算出主轴 + 各通道沿轴位置，另一半（**没参与定轴**）算各通道放电先后；报两者 Spearman ρ（+ 自助置信区间）。这检验的是"空间轴能不能预测它没见过的事件的放电顺序"——能=真共享通路，不能=平均出来的假象。

- 26 个有轴被试：**16 个强（ρ≥0.7）/ 5 个中等（0.4–0.7）/ 5 个弱（<0.4）**；中位 ρ=0.75，range [0.08, 1.00]；**25/26 的置信区间下界在 0 之上**。
- 强例：916 ρ=1.00、zhangjiaqi 0.96、1146 0.91、253 0.90、139 0.89。弱例：1150 0.08、635 0.11、chengshuai 0.12、zhangbichen 0.35、620 0.37。
- **判别成立**：弱的那几个大多是起止 core 沿轴交错的退化几何（见 §2 弱轴）——留出半检验正确地把它们标成"不是真通路"。
- **读法 caveat**：这个 ρ 是在**通道**上算的相关（n = 7–52 个通道，不是事件数），通道少的被试 ρ 噪声大、要连置信区间一起看；点估计取一个固定随机折半、CI 由自助折半得到，但**通道少时 CI 可能很窄甚至退化成一个值**（如 139 CI=[0.89,0.89]），别把窄 CI 当"高确定性"。

**起止紧凑性 null（"比随机更紧"还是只是触点间距）**：同一被试同一 eligible 通道集随机抽同样多通道 1000 次比 core 半径。结果：**源端 10/26、汇端 9/26 显著比随机紧（p<0.05）**——两端差不多，**没有"源聚汇散"的队列规律**（个别被试如 958 是源紧汇散，但不能外推成队列结论）。所以"core 比随机紧"只在约 1/3 多被试上成立，不是普遍现象。

**正反模板（不藏 minority mode）**：12 个 swap-positive 被试里 10 个能算出两条簇轴。**7/10 是"同一空间轴、两个相反方向"**（两轴夹角 cos≤−0.85，如 139/zhangjiaqi/620 cos=−1.00）；另 3 个更像**两条岔开的路径**（958 cos=−0.62、384 −0.69、liyouran −0.20）。所以"正反"在多数被试上是同一条轴反向走，但不是全部。

---

## 3. 图（用户需亲自目视）

均在 `results/topic4_sef_hfo/skeleton_geometry/figures/`，配套中文 `README.md`：

- **`axis_frame_examples.png`** — 6 个 primary/distributed 被试的传播骨架散点（横=离源沿轴距离 mm，纵=离轴垂直距离 mm，颜色=该通道刻板性超出随机的量，对称居中于 0）。看一个骨架在每个病人里"长什么样"。
- **`skeleton_scalars_by_dataset.png`** — 四联散点（轴长 / 起点半径 / 终点半径 / 横展），**按数据集分两列、不混**，每点一个被试。
- **`along_axis_stereotypy_profile.png`** — pooled 所有 primary+fallback 被试，沿轴方向的刻板性剖面（纵轴=超出随机的量，虚线=随机水平 0）。延展性 / 边界 sanity。
- **`per_subject/{ds}_{subj}_card.png`（26 张，每被试一张 path card）** — 5 面板：① 脑空间 3D（灰=全通道、彩=参与通道按放电先后着色、▲源/■汇 + 源→汇箭头）；② 轴坐标散点（点大小=参与事件数、色=中位放电先后）；③ **留出半稳定性散点（核心）**：沿轴位置 vs 留出半放电先后 + ρ/CI——强通路是清楚的单调云（如 958 ρ=0.85），退化的是散点（635 ρ=0.11）；④ 正反模板：主簇 + 次簇两条轴 + 夹角；⑤ 紧凑性 null inset：真实源/汇半径在随机分布里的位置 + p。**这是回答"看不看得见通路"的主图，cohort scalar 图只作索引。**

---

## 4. 诚实警告（这一轮的纪律红线，保护本轮不被过度解读）

1. **"横向铺展"不是"斑块真实宽度"。** 它是**参与通道**偏离主轴的垂直铺展，受采样 + 发放率共同圈住——几何只能在有通道参与（=高发放）的地方测，所以它是斑块宽度的**下界**，不是上界。每个被试存了 participation-threshold sweep（`perp_spread_participation_sweep`）作为对照，必须连同这个 sweep 一起读。
2. **跨数据集毫米不可 pool。** Yuquan native 解剖 mm vs Epilepsiae MNI152 模板 mm（warp 类型未独立核验）；只在各数据集内部比较。
3. **沿轴刻板剖面（图）部分是同义反复，但留出半检验不是。** 沿轴刻板剖面那张图里"沿轴为正"部分是按构造来的（rank≈沿轴位置），所以它只是边界 sanity。**真正回避同义反复的是 §2.5 的留出半检验**——轴用一半事件定、顺序用另一半事件算，两半独立。读结论请以 §2.5 留出半 ρ 为准，不要拿沿轴剖面图当独立证据。留出半 ρ 是在**通道**上算的（n 小的被试噪声大、CI 可能退化），也要带 n 和 CI 读。
4. **起止端 ≠ 致痫区。** 稳定起止端与临床致痫区无显著关系早前已确证（PR-6 H1 NULL）——本轮所有与致痫区的关系只作描述，不下任何显著性主张。
5. **"紧凑"现在有 null 了，但只在约 1/3 多被试上成立。** 本轮补了随机-k null（§2.5）：源端 10/26、汇端 9/26 显著比随机紧。所以"比随机紧"**不是普遍现象**，不能笼统说"起止 core 紧凑"；要按被试看 `source/sink_radius_null.p_value`。Yuquan 的小半径里，没过 null 的那些可能仍是触点间距而非生物结构紧。
6. **模态采样混杂。** Epilepsiae 是 SEEG+ECoG 混合（类型从命名推断，非 ground-truth），Yuquan 纯 SEEG；栅格与深部杆采样方式不同。core 半径 / 离轴铺展 / distributed 判定都可能受植入模态影响——跨数据集、跨模态的几何差异**不得解释成生物学差异**（见 §2 模态分层）。
7. **厘米级尺度是校准约束，不是机制验证。** 18–22mm 的轴长说明模型空间尺度该设在厘米级（非毫米点源、非全脑），为模型传播长度参数提供经验约束；它**不能单独支持** SEF-HFO 的任何具体机制（若机制参数本就用相近数据调出，这只是 calibration consistency）。
8. **留出是被试内的，不是跨被试 held-out；探索性、archive-only。** §2.5 的留出半是把**单个被试**的事件折半，证明该被试的轴能预测自己留出的事件——这是被试内的泛化，**不是**跨被试 held-out 验证，也没做多重比较校正（26 个被试各测一次）。"21/26 有通路"是描述性计数，不是一个经校正的队列检验。不得把任何数字升级进 topic4 主文档 paper 档。

---

## 5. 出界 / 下一轮

**本轮已补做**（相对上一版）：留出半通路检验（§2.5，回避同义反复）、起止 core 随机-k 紧凑性 null、正反模板并报次簇、聚类改用上游已验收模板、弱轴标志、fail-fast。

**仍留下一轮**：
- **几何 vs 发放 的稳定性对比（split-half）**：比较"传播顺序几何"和"发放计数"哪个在两半之间更稳——本轮没做（本轮验证的是"通路是否真"，不是"几何是否比率稳"）。
- **跨被试 held-out + 多重比较校正**：本轮留出是被试内的；要把"21/26 有通路"变成可发表的队列主张，需要跨被试留出设计或对 26 次检验做校正。
- **模态分层敏感性**：Yuquan SEEG / Epilepsiae 纯深部 / Epilepsiae 含栅格 三层分别比尺度（本轮只报了三层计数 + limitation，未做分层比较）。
- **离轴时序耦合宽度作为门控的队列尺度量**：横向上时序还协同到多宽，作为一个带 eligibility 门的队列标量。

---

## 链接

- spec：`docs/superpowers/specs/2026-06-08-sef-hfo-propagation-skeleton-geometry-design.md`
- plan：`docs/superpowers/plans/2026-06-08-propagation-skeleton-geometry.md`
- 数据侧总 plan：`[[project_topic4_soz_localization_plan]]`（`docs/superpowers/plans/2026-06-06-sef-hfo-soz-localization-rate-vs-geometry.md`）
- 代码：`src/propagation_skeleton_geometry.py`（纯几何 + 留出半检验 `split_half_axis_validation` + 紧凑性 null `core_radius_null` + 事件归簇 `assign_events_to_templates`，无 I/O）、`scripts/run_propagation_skeleton_geometry.py`、`scripts/plot_propagation_skeleton_geometry.py`（cohort scalar 图）、`scripts/plot_propagation_skeleton_card.py`（每被试 path card）
- 结果：`results/topic4_sef_hfo/skeleton_geometry/{cohort_summary.json, per_subject/, figures/}`
