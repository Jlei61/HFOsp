# SEF-HFO 间期传播几何骨架 — 一轮结果归档（2026-06-08）

> **状态**：探索性、**描述性测量（给模型喂尺度数），不是假设检验，不是 SOZ 定位器**。无 held-out，archive-only，不得把任何数字升级进 topic4 主文档的 paper 档。
> **上游**：`docs/topic4_sef_itp_framework.md`（SEF-HFO v0.2）；数据侧 plan `docs/superpowers/plans/2026-06-06-sef-hfo-soz-localization-rate-vs-geometry.md`；low-rate 读回结论 `docs/archive/topic4/sef_hfo/low_rate_template_stability_2026-06-07.md`。
> **硬前置**：`lagPatRank` phantom-mask（Topic 0 §3.1）——源/汇 core 只用 phantom-safe 的稠密排名（非参与通道为 NaN，无法被选进 core；实测 `phantom_core_violations=0`）。

---

## 朴素话摘要（测了什么 / 怎么测的 / 揭示了什么）

**测了什么。** 每个病人的间期 HFO（高频振荡）事件里，通道有一个稳定的放电先后顺序——谁先点着、谁最后停。这一轮我们纯几何地量这条"传播骨架"在脑子里长什么样：起点一小撮通道挤得有多紧、终点一小撮通道挤得有多紧、从起点到终点这条传播主轴有多长、这条传播除了沿主轴走还往两侧横向铺多宽，以及沿着主轴往前走、放电先后这套刻板顺序能延伸多远。目的只有一个：把这几个空间尺度数（起止斑块的大小、传播轴的长度、横向铺展的尺度）喂给 SEF-HFO 这个易激场模型当输入参数。我们**没有**用它去定位致痫区（早前已确证稳定起止端≠临床致痫区），也**没有**在这一轮检验"几何顺序比发放计数更稳"（那是下一轮）。

**怎么测的。** 先在每个病人里找出最早放电的一小撮通道（起点）和最晚放电的一小撮通道（终点），用它们各自的几何重心连一条线当传播主轴，再把每个通道投影到这条轴上拆成两个分量：沿轴走多远、偏离轴多远。起止斑块的紧凑度=这一小撮通道离自己重心的均方根半径；横向铺展=所有参与通道偏离主轴的垂直距离铺多宽。沿轴的刻板性这样比：如果通道放电先后纯属随机配对，刻板程度应该是某个随机基线值；我们报实测值减去这个随机基线（这个差是"超出随机的量"，跟事件数多少无关，所以不同通道、不同病人之间能直接比，不会被高发放率的通道虚高）。可测与否设了个数值门：既参与传播又有有效三维坐标的通道数够多（≥7）才进主档，少一点（5 或 6 个）进兜底档，太少（<5）只做描述、不算几何尺度。

**揭示了什么。** 在这个尺度上看，传播骨架是厘米级的——传播主轴长度的中位数大约 18–22 毫米；起点和终点这两小撮通道挤得相当紧凑（Yuquan 数据尤其紧，半径中位数约 3 毫米）；绝大多数病人的传播不是沿一根电极杆的一维直线，而是跨多根杆、铺成一片的分布式采样；沿轴的刻板顺序整条轴上都为正（但这一条**部分是模板定义出来的**，不是独立证据）。这些都是给模型用的尺度数，**不是**结论性的临床主张。

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

**采样几何（`sampling_geometry`）**：21 distributed（跨多根杆，可测横向铺展）/ 5 1D（单根电极杆，横向铺展不可测、已排除出横展统计）。

**起止端路由（swap routing）**：12 个被试在跨事件上能稳定看到起止角色互换（7 个候选层 + 5 个严格层）→ 这些用主导聚类那条轴（`template_source=dominant_cluster_*`，9+3=12）；其余 18 个无稳定互换 → 用全程那条轴（`template_source=full_recording`）。

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

横向铺展只在 distributed（非单杆 1D）被试上可测，故可测数（5+16=21）= distributed 总数。

**起止 core 跨杆情况**：起点 core 跨 >1 根电极杆的有 10/26 个被试，终点 core 跨 >1 根杆的有 12/26 个——再次说明多数传播不是沿单一杆的一维链，而是跨杆铺开。

**起止 core 没有 egregious 裂核**：26 个被试里没有任何一个的起点 core 满足"最大两两距离 > 2.5×RMS 半径"（0/26）——MEB + max-pairwise 这道裂核护栏在位，但数据本身已经相当紧凑，护栏没被触发。

**最短轴个案（不是紧凑模板，是弱约束轴）**：`epilepsiae:635` 轴长仅 1.8 mm。**这条轴短不是因为模板紧**，而是因为它的终点 core 跨在起点两侧——终点 3 个通道里有一个沿轴坐标在 −20.4 mm（落在起点*后方*），把终点重心拉回到几乎与起点重合，所以两个重心间距塌成 1.8 mm；但终点 core 自身 RMS 半径 23.5 mm、最大两两距离 53.3 mm，还有一个内部通道偏离主轴 52 mm。它是 distributed 采样、布尔 `degenerate_axis=false`（这个布尔没抓住"终点 core 空间上不连贯"这种情况），但**这是轴坐标系最不可信的一个个案，axis 长度在这里不能读成"模板紧凑"**。把它作为最短轴离群点保留，但明确改口：不是 compact 模板，是弱约束轴。
> 注：这一点与本轮上游 prompt 草稿里"a genuinely compact template, not degenerate"的措辞**冲突**；本归档以 per-subject 数据为准（`source/sink_radius`、各通道 `along_axis_mm`），不照搬 prompt 措辞——"compact" 在本文其它地方专指 core 半径（Yuquan 2.9 mm），用它形容 635 的短轴会发生代号坍缩（CLAUDE.md §6.3）。

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
5. **探索性、无 held-out、archive-only。** 不得把任何一个数字升级进 topic4 主文档 paper 档。

---

## 5. 出界 / 下一轮

- **几何 vs 发放 的稳定性对比（split-half）**：把数据折半，比较"传播顺序几何"和"发放计数"哪个在两半之间更稳——本轮没做（本轮只描述尺度，不比稳定性）。
- **离轴时序耦合宽度作为门控的队列尺度量**：横向上时序还协同到多宽，作为一个带 eligibility 门的队列标量——留下一轮。

---

## 链接

- spec：`docs/superpowers/specs/2026-06-08-sef-hfo-propagation-skeleton-geometry-design.md`
- plan：`docs/superpowers/plans/2026-06-08-propagation-skeleton-geometry.md`
- 数据侧总 plan：`[[project_topic4_soz_localization_plan]]`（`docs/superpowers/plans/2026-06-06-sef-hfo-soz-localization-rate-vs-geometry.md`）
- 代码：`src/propagation_skeleton_geometry.py`（纯几何，无 I/O）、`scripts/run_propagation_skeleton_geometry.py`、`scripts/plot_propagation_skeleton_geometry.py`
- 结果：`results/topic4_sef_hfo/skeleton_geometry/{cohort_summary.json, per_subject/, figures/}`
