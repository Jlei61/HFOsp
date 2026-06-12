# Subject-Specific 2D 传播触点平面读出 — real ↔ model 共同语言(2026-06-11)

桥接 topic3(空间 / 发作起始区)↔ topic4(传播建模)。Spec: `docs/superpowers/specs/2026-06-11-propagation-contact-plane-readout-design.md`;Plan: `docs/superpowers/plans/2026-06-11-propagation-contact-plane-readout.md`。

---

## 摘要(朴素话)

**测了什么** — 我们把真实病人和模型生成的"放电传播"事件,都压成同一张标准化的二维触点图:哪些触点参与、谁先谁后、传播往哪个方向、早端那团有多散。然后看模型生成的传播,在这张图上"像不像"真实病人的。这张图只表达传播顺序和采样支撑,不含原始电活动。

**怎么测的** — 模型那张图和每个真实病人的图算一个相似度;再看这个"模型对真实"的相似度,落在"真实病人彼此之间相似度"这把尺子的什么位置(只报落在哪个百分位,不报 p 值——模型没有病人身份,不能假装做队列检验)。然后做一层稳健性:只拿真实里那些二维铺得开的病人,再比一次,看结论变不变。

**揭示了什么** — 在这套二维读出下,**一个只往一个方向走的传播模型,看起来"像"真实病人**(相似度落在真实彼此之间的中上位置,只挑二维病人复测也一样);**一个来回双向走的传播模型,看起来"不像"任何单个真实模板**(相似度垫底、先后顺序排得最乱)。这符合常识:真实数据里每个模板就是一个方向,所以单向的像、双向的对不上单个模板。两个附带观察:模型的物理尺度偏小(虚拟厘米级贴片 vs 真实人脑数厘米电极);模型比真实更"刻板"(几乎没有事件间抖动)。这些都是描述,不是"模型验证了真实机制",也不涉及发作起始区。

(内部归档代号:`norm_scale_mm`, `rank_vs_xnorm_spearman`, `field_placement.{model_to_real_median_corr, placement.percentile, robust_z}`, `one_dimensional_sampling`, `out_of_field`, `snn_cm_spontaneous/{oneend_pos_s1, oneend_neg_s1, pooled_bidir/model_spont_bidir}`)

---

## 1. 读出管线(每模板一份标准化 record)

输入(每 subject/模板):`*_lagPat_withFreqCent.npz`(ranks/bools/lag_raw)+ 3D 触点坐标 + rank-displacement accepted 模板(`pairs[0]` 的 `rank_a/b_dense_full`)。phantom 屏蔽走 `mask_phantom_ranks(normalize=True)`(= 事件内归一化 masked rank,即 spec 的 rank_norm)。

每触点聚合:`typical_rank`(主,nanmedian 事件内归一化顺序)、`typical_time`(副,仅画图)、`support`(参与事件占比)、`uncertainty_rank/time`(IQR)。

二维平面:沿轴 = `compute_axis_frame` 的 source→sink 核方向;带符号横向 = 轴外残差 PCA 第一主方向(B1 确定性符号,仅显示)。归一化见 §2。

连续场:`smooth_field` 在归一化网格上做 support 加权高斯核 → T(顺序)/S(支撑)/U(不确定度),`S<S_THRESH` 灰掉不进相关。

flags:`one_dimensional_sampling`(单杆 / p90 off-axis < 间距)、`poor_planarity`(`transverse_pc1_variance_explained < 0.80`)、`low_contact_count`(<6)、`low_support`、`weak_axis`、`out_of_field`。

模块:`src/propagation_contact_plane_readout.py`(纯函数 + 26 green tests)。runner:`scripts/run_contact_plane_readout.py`(真实)、`scripts/run_model_contact_plane_readout.py`(模型,复用观测层 NPZ + sidecar montage,`--montage` 可 override)。比较:`scripts/run_real_vs_model_comparison.py`。图:`scripts/plot_contact_plane_static.py`。

## 2. 归一化合同(pilot audit 修正,2026-06-11)

**初版** spec §3 定 `x_norm = along / axis_length`。pilot 真实数据暴露:`axis_length`(源核↔汇核中心距)常远小于触点铺开范围(端点核是 k=3 紧子集,温度梯度不一定空间单调)。cohort audit:

- 旧归一化下 distributed(主二维)队列 **60% 记录有触点落到固定网格 [−0.5,1.5] 外,最坏 91% 触点出界、x_norm 到 5.68、11 条连发作起始区触点都出界**,场退化成无意义插值(如 `epilepsiae_1084_t_b` 11 个触点 10 个出界)。

**修正**:归一化分母改为参与触点 along 的鲁棒跨度 `norm_scale_mm = p97.5(along) − p2.5(along)`(x=0 仍源核中心;x/y 同 scale 各向同性)。`axis_length_mm` 留作独立物理标量,不进 field 归一化。

修正后全队列(52 usable record / 26 subject)重跑,**0 STALE,全部带 `norm_scale_mm` + `out_of_field`**:

| 层 | n records | 有外溢的记录 | 最大外溢触点数 | 强轴向(\|rho_x_rank\|≥0.5) |
|---|---|---|---|---|
| DISTRIB(主) | 45 | 5 | 4 | 30/45 |
| 1D(supplement) | 7 | 0 | 0 | 6/7 |

外溢从旧的 60%/最坏 91% 收到几乎为 0。`out_of_field` 字段仍保留(count/contacts/support_sum/soz_count),图上以红色空心三角 + 计数显式标注,绝不静默丢。

## 3. real-vs-model 描述性比较(spec §9,无 p 值)

模型 = 自发活动屏查级产物 3 个,各出一份 record(逐 template):

| 模型 | n_ch | 1D? | rho_x_rank | 含义 |
|---|---|---|---|---|
| `model_oneend_pos` | 8 | True | 0.90 | 单向行波(+) |
| `model_oneend_neg` | 8 | True | 0.99 | 单向行波(−) |
| `model_spont_bidir` | 8 | False | −0.14 | 双向自发(轴向排序弱) |

**场相似度 placement**(模型→真实场相似度,落在真实彼此相似度分布的百分位 / robust z;镜像不变 + support-gated;subject-first 折叠)。**稳健性**:全队列(n_real subjects 26) vs 仅二维真实(2D-only,n_real subjects 23):

| 模型 | field 百分位(全队列) | field 百分位(2D-only) | rho_x_rank 百分位(全 → 2D-only) |
|---|---|---|---|
| `model_oneend_neg` | 65.4 (corr 0.83) | 60.9 (corr 0.81) | 100 → 100 |
| `model_oneend_pos` | 73.1 (corr 0.85) | 78.3 (corr 0.82) | 88.5 → 91.3 |
| `model_spont_bidir` | **7.7** (corr 0.28) | **13.0** (corr 0.28) | 0 → 0 |

**方向对 1D 混入稳健**:剔除 7 个 1D 真实记录后,两个单向模型仍落 61–78 百分位、双向模型仍垫底(7.7→13.0,在噪声内),`oneend 像 / bidir 不像` 不变。

标量 placement(`axis_length_mm` 百分位 ~4%、`transverse_width_mm` ~8%)显示模型物理尺度系统性偏小(虚拟 cm 贴片 vs 真实人脑 SEEG)。模型 `uncertainty_rank` 场 ≈0(单向模型不确定度场全灰)= 模型比真实更刻板。

sensitivity sweep(`S_THRESH∈{0.10,0.15,0.20} × overlap_min∈{15,25,40}`,per-model 27 行)= `comparison/sensitivity.json`。

## 4. 接受的结论 / 口径锁

**能下的**(描述性):
- 在统一的触点平面读出下,单向行波模型的顺序场相似度落在真实病人彼此相似度分布的中上位置(2D-only 复测一致);双向自发模型在顺序场相似度和沿轴排序上均明显低于真实范围。
- 真实早端这团 + 传播场 与发作起始区的空间相对位置(描述性叠加,如 `yuquan_chengshuai` 整个模板落在标注核心内)。

**不能下的**(口径锁,写进 spec §14 + 图注):
- 不报 "p<0.05 模型匹配队列";不说 "模型验证真实机制";不说 "模型重现发作起始区";发作起始区永不进任何指标。
- 不说 "bidir 二维所以更真实"——它几何二维,但顺序场对单模板真实传播不匹配。
- 早端措辞:"有抖动的刻板传播通路 + 轮流当早端的小群",禁 "固定稳定早端 / 弹性入口区"。

**限制**:模型 n=3、屏查级、纯描述;模型物理尺度偏小、过于刻板;主比较为混合真实队列,2D-only 作稳健性检查。

## 5. 产物路径

```
results/spatial_modulation/propagation_geometry/observation_readout/
├── real_subjects/      52 usable record(26 subject;9 yuquan 无坐标良性跳过、若干 descriptive_only)
├── model_subjects/     3 model record
├── comparison/         real_vs_model_summary.json + sensitivity.json(per-model 27 行)+ sens_*/
├── comparison_2d_only/ 2D-only 稳健性
├── figures/
│   ├── README.md       中文逐图说明
│   ├── static_maps/    52 真实图
│   └── model_maps/     3 模型图
└── (results/ 全树 gitignored;本 archive doc + spec/plan + 模块/runner/tests tracked)
```

分支:`codex/topic3-topic4-2d-readout`(worktree,未合回)。
