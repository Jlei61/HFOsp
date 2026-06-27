# Topic 5 归档：发作早期方向无监督两类 ↔ 间期 A/B 方向（exploratory，描述性）

> 日期：2026-06-27 · 状态：**已执行**（broadband + hfa，n=6 干净 ECoG）· 层级：**exploratory，描述性，非队列主张，不声称重放**
> 设计 spec：`docs/superpowers/specs/2026-06-27-topic5-ictal-direction-clustering-design.md`
> 实现 plan：`docs/superpowers/plans/2026-06-27-topic5-ictal-direction-clustering.md`
> 代码：`src/topic5_directional_replay.py`（20 tests）+ `scripts/run_topic5_directional_replay.py`（3 tests）+ `scripts/plot_topic5_directional_replay.py`（2 tests）
> 上游：A 线主线 `axis_alignment_AB_result_2026-06-14.md`（符号自由共线）· 失败前序 C 线 `subtype_direction_cline_result_2026-06-15.md`

---

## 0. 白话摘要（§8 三段式）

**测了什么。** 每个几何干净的板状电极（ECoG）病人，把每次发作头十秒"往脑子哪个方向烧"取成一个方向角，
**不看间期、不预设正反**，纯按方向把这个病人的多次发作分成两堆；分完后看这两堆的平均方向，和病人平时那
两条间期高频传播路线（模板 A/B）的方向是什么关系。

**怎么测的。** 关键是不能被算法骗——二维方向上硬分两类，任何一组角度都会被切成"主堆/次堆"。三道事先锁死的门：
① 这真是两堆、还是"一个主方向加少数散点"?（拿"主方向+均匀背景散点"这个分布模拟两千次，看实测的两堆分离
有没有超过它）② 间期那两条路线本身方向够不够相反、成不成一根能比的轴?（夹角 <60°/60–120°/≥120° 三档，看结果前锁）
③ 两堆方向贴上 A/B，有没有超过"把两堆整体随机旋转两千次"?（消掉"2 个堆配 2 个模板取最近"的天生优势）

**揭示了什么。** **六个病人、宽带和快活动各跑一遍，没有一个能说"发作分两类、分别对上间期 A/B"。** 唯一真分成
两堆的是 442（宽带）：它的多数堆确实贴近间期模板 B 那条路线，但少数堆离间期 A 约 45°，两堆整体贴 A/B 没有超过
随机旋转——所以判"有两个方向类、但和间期模板对不上"。其余病人要么根本不是两堆（只是一个主方向加零散），要么
两条间期路线方向几乎重合、压根不成轴（1084，夹角仅 6°）。换到快活动一样：六个全是"主方向"或"不成轴"，连 442
都不再是两堆。**与本 topic 之前所有线一致：共性是粗的网络/解剖锚，不是细到"方向两类"可分的重放。**

（内部归档代号：`report_tier` two_class_mapped/unmapped/single_axis/diagnostic_only；P0=contaminated `unimodal_null_pvalue`
p_bimodal + bootstrap stability；P1=`axis_quality_tier` Δ_AB + `best_pair_rotation_null` p_align；上游 A 线
`|corr_pair_mirror_invariant|`；θ=`gradient_angle`(头10s bb/hfa AUC)。）

---

## 1. 方法（预锁口径，与 spec/plan 一致）

- **队列**：6 个几何干净 ECoG（`电极类型==ECoG` 且 `coord_aspect≥0.15`）：442, 548, 583, 1084, 384, 958。SEEG / 近一维仅 caveat，未跑。
- **激活场**：每发作头 10 s 逐触点 AUC；**broadband 主，hfa 灵敏度**（分别报）。方向 = 该场最小二乘平面梯度增长方向（`gradient_angle`）。
- **聚类**：发作方向单位向量 `[cosθ,sinθ]` 上 `KMeans(k=2)`，**盲于间期**。
- **P0 两类资格门（三条 AND）**：`n_sz≥6` 且每堆 `≥3`；`p_bimodal<0.05`（H0=**一个集中主方向+均匀背景散点**，B=2000，统计量=单位向量 silhouette；纯单峰 null 太弱已弃）；bootstrap 标签稳定中位 `ARI≥0.5`（B=500，次级）。
- **P1 轴质量门**：`Δ_AB≥120°` interpretable / `60–120°` weak_axis / `<60°`（或模板 n_valid<6）diagnostic_only。
- **P1 对齐 null**：旋转两类整体朝向 B=2000，`p_align<0.05` 才算"对齐显著"。
- **分档**：`two_class_mapped` 当且仅当 `geometry_clean ∧ interpretable ∧ two_class_eligible ∧ p_align<0.05`；否则逐级降到 unmapped / single_axis / diagnostic_only。

---

## 2. 队列结果

### Broadband（主）

| 病人 | n / 两堆 | p_bimodal | stab | Δ_AB | axis_tier | p_align | **report_tier** |
|---|---|---|---|---|---|---|---|
| 442 | 22 / [15,7] | **0.042** | 1.00 | 147° | interpretable | 0.314 | **two_class_unmapped** |
| 548 | 26 / [21,5] | 0.205 | 0.83 | 112° | weak_axis | 0.341 | single_axis |
| 583 | 22 / [18,4] | 0.579 | 1.00 | 60° | weak_axis | 0.209 | single_axis |
| 384 | 12 / [7,5] | 0.570 | 1.00 | 139° | interpretable | 0.053 | single_axis |
| 958 | 12 / [7,5] | 0.575 | 1.00 | 158° | interpretable | 0.126 | single_axis |
| 1084 | 72 / [45,27] | 0.383 | 1.00 | **6°** | diagnostic_only | 0.835 | diagnostic_only |

### HFA（灵敏度）

| 病人 | n / 两堆 | p_bimodal | Δ_AB | axis_tier | report_tier |
|---|---|---|---|---|---|
| 442 | 22 / [14,8] | 0.221 | 147° | interpretable | single_axis |
| 548 | 26 / [23,3] | 0.610 | 112° | weak_axis | single_axis |
| 583 | 22 / [19,3] | 0.734 | 60° | weak_axis | single_axis |
| 384 | 12 / [9,3] | 0.306 | 139° | interpretable | single_axis |
| 958 | 12 / [9,3] | 0.881 | 158° | interpretable | single_axis |
| 1084 | 72 / [37,35] | 0.836 | 6° | diagnostic_only | diagnostic_only |

**两 band 一致：`two_class_mapped` = 0/6。** 宽带唯一真两堆 = 442（p_bimodal=0.042），但两堆贴 A/B 未超随机（p_align=0.314）。
快活动里 442 都不再是两堆（p_bimodal=0.221）。

### 442 个案（图见 §4）

宽带：两个发作方向堆 = 类1（n=15，朝间期 B 一侧，离 B ~25°）+ 类2（n=7，离间期 A ~45°）。间期模板 A 事件方向集中
（R=0.80）但偏离发作轴约 45°，模板 B 事件方向弥散（R=0.32）。→ 真有两个方向类，但与间期两条路线方向对不齐到超随机水平。

---

## 3. 防自欺：为什么这套门是必须的

朴素 best-pair 残差（不设 null）在 548/583/958 给出 9–18° 的"看着像对上"——这是 review（2026-06-27）点出的假阳风险：
- **纯单峰 null 太弱**：实测 "20 紧 + 4 散" 在纯单峰 null 下 p≈0.002（被误判两类）。改 H0 为"主方向+均匀背景散点"后 p≈0.27（正确挡住）。
- **bootstrap stability 不防散点**：固定散点每次被一致分到一簇，ARI≈1（假高）。反散点主门是二模 null。
- **best-pair 必须有旋转 null**：2 堆配 2 模板取最近天生显得好；旋转 null 后 548/583/958 全部回落到 single_axis。
- **轴质量门预锁**：1084 间期两模板夹角仅 6°（不成轴），看结果前已锁 `<60°→diagnostic_only`，避免后验筛选。

回归测试固化：纯单峰 → 判非两类；"主方向+散点" → 判非两类（`tests/test_topic5_directional_replay.py`）。

---

## 4. 工件

- **代码**：`src/topic5_directional_replay.py`（geometry/clustering/P0+P1 门/旋转 null，20 tests）；
  `scripts/run_topic5_directional_replay.py`（runner，3 tests）；`scripts/plot_topic5_directional_replay.py`（图，2 tests）。
- **结果**（gitignored）：`results/topic5_ictal_recruitment/directional_clustering/`：
  `per_subject/*__dir_cluster_{broadband,hfa}.json`、`cohort_summary_{broadband,hfa}.{json,csv}`。
- **图**：`.../directional_clustering/figures/`：
  - `*__dir_cluster_{broadband,hfa}.png`（每被试：发作两类着色 + 间期模板方向线 + 分档角注）；
  - `epilepsiae_442__classes_vs_interictal_hist_{broadband,hfa}.png`（成熟 rose 风格：间期事件直方图 + 两色发作方向类，发作轴转到 0°/180°）；
  - `figures/README.md`（中文逐图说明）。
- **commits**（topic4-axial-intervention-probe 分支）：模块 `37070b0` / runner `3ff41fc` / 图+README `328c58b` / spec+plan `f82d5a6`。

---

## 5. 允许 / 禁止措辞

- **允许**："发作早期方向在干净病人里基本不分成与间期 A/B 对应的两类——6 个被试两 band 全无 two_class_mapped；唯一真分两堆的 442 其两类与间期模板对不齐到超随机水平（exploratory、描述性）"；
  "与 A 线及 echo gate 各线一致：共性是粗网络/解剖锚，不是方向两类可分的重放"。
- **禁止**：把任一被试或队列写成"发作传播分两类 / 重放间期 A/B 路线"；把"未过两类资格门"的被试写成"两类"（只准"主方向"）；把 n=6 写成队列断言；用朴素 best-pair 残差（无 null）声称"对上了"；"证明发作方向与间期无关"（实为没看清 + 凑不出队列）。
