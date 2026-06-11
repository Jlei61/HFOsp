# Subject-Specific 2D 传播触点平面读出 — 设计 spec

- 日期: 2026-06-11
- 状态: 设计已审阅（78/100 → 四锁补完），待用户复核 → writing-plans
- 关联:
  - 复用 `src/propagation_skeleton_geometry.py`（轴框架 / 端点核 / per-channel stereotypy）
  - 复用 `src/sef_hfo_observation.py`（`sample_envelopes` 高斯核 + 模型虚拟 SEEG 读出链）
  - 兄弟 spec: `docs/superpowers/specs/2026-06-08-sef-hfo-propagation-skeleton-geometry-design.md`、`docs/superpowers/specs/2026-06-06-sef-hfo-virtual-seeg-observation-layer-design.md`
  - 桥接: topic3（空间 / SOZ）↔ topic4（SEF-HFO 建模）

---

## 0. 一句话 + 范围

把每个 subject/模板的真实传播事件，压成一张**标准化 2D 触点平面图**：哪些触点参与、谁先谁后、方向往哪、早端这团有多散。**不含原始信号，只含传播顺序和采样支撑。** 模型输出走**完全相同**的读出后，落进真实 cohort 的指标分布里，描述性地看"像不像"。

**做（本 spec 范围）**：
- 标准化 2D 平面（双坐标系：物理 mm 标量平面 + normalized field plane）
- 带符号横向 y 轴 + 确定性符号约定 + 跨 subject 镜像不变比较规则
- 每触点时序聚合（rank 主、time 副）+ 连续场 T/S/U
- SOZ 描述性叠加层（first-contact alias）
- 模型侧复用观测层 + real-vs-model 描述性 posterior-predictive 比较
- 质量 flags + 静态图

**不做（非目标）**：
- 不做原始信号反演、不重建电活动
- 不做"模型解释 SOZ"的正式 cohort 检验（SOZ 仅描述性叠加，不进任何 metric）
- 不写 reverse_axis_cosine / heldout_rho（依赖正反双模板，观测层尚未接，列第二批）
- 动画 deferred（执行顺序最后一步）
- 不重写 source/sink core、axis length、perp spread、1D flag、participation sweep（骨架模块已有）

---

## 1. 背景与桥接定位

topic3 回答"传播往哪、SOZ 在哪"；topic4 用 SEF-HFO 模型生成传播。两边一直缺一套**共同语言**让"模型生成的传播"和"真实数据的传播"放在同一张图、同一组指标下比。本读出就是这套共同语言。

桥接到 topic3 的那一笔 = **SOZ 描述性叠加**：在 2D 平面上画出早端这团落哪、SOZ 落哪，只展示空间相对位置，不下"模型解释 SOZ"的结论（守 topic4 不把 template source 拟合 clinical SOZ 的红线）。

**复用，不重造**（骨架模块 `propagation_skeleton_geometry` 已覆盖）：
- `build_endpoint_cores(template_axis, eligible_mask)` → source/sink 核
- `compute_axis_frame(coords, source_idx, sink_idx)` → `along_axis` / `off_axis`（非负）/ `axis_length` / `degenerate_axis`
- `channel_stereotypy_components(...)` → per-channel 顺序稳定度
- 1D 采样 / poor-planarity / participation 判定

**新模块只补 4 件**：signed-transverse 坐标、normalized 2D field 插值、real-vs-model 比较 record、SOZ overlay renderer。

---

## 2. 输入契约

每个 real subject / 模板：
- `lagPatRaw`：每事件每触点时间位置；非参与 = NaN。**单位以 loader 输出为准**——真实 legacy on-disk 为**秒**，模型内部 sim 为 ms（observation writer 已转秒落盘）。本模块只做**事件内归一化**（§5 min 减完除 max），单位自动消掉；readout record 记录 `lag_time_unit` 供 time 副图轴标注，**禁止跨 record 比裸时间**
- `lagPatRank`：每事件每触点顺序；**必须经 phantom 屏蔽**（`mask_phantom_ranks` / masked-feature 路径），非参与 = NaN
- `eventsBool`：触点是否真的参与该事件
- `channel_names`
- 触点 3D 坐标：`src/seeg_coord_loader.py`，数据集分流（Epilepsiae MNI / Yuquan native RAS），mm
- cluster labels：每事件属于哪个模板（masked PR-2 路径）

硬约束：
- 非参与触点必须用 `eventsBool` 屏蔽
- 不直接使用未 mask 的 rank（守 lagPatRank phantom 合同）
- 不读 raw signal
- 来源 NPZ = `*_lagPat_withFreqCent.npz`（10ch 全集），不是 `*_lagPat.npz`（旧 7ch slice）

模型 subject：观测层已写出与真实同构的 `*_lagPat_withFreqCent.npz` + `*_montage.json`（2D 触点坐标），直接进同一读出（见 §8）。

---

## 3. 双坐标系（P1 lock — 不混 mm 与 normalized）

**关键澄清**：2D 平面的**空间坐标是 subject/模板固定的**（由触点 3D 坐标 + 模板轴一次性导出），**不随 event 变**。随 event 变的只有时间/顺序聚合（§5）。plan §3 的"event 内归一化"专指**时间归一化**，不是坐标。

拆成两个坐标系，分别服务两类输出：

**(a) 物理 mm 标量平面** — 服务标量几何指标，保留物理单位：
- `axis_length_mm`（= `compute_axis_frame` 的 `axis_length`）
- `transverse_width_mm`（signed-transverse 的稳健展宽，见 §4）
- 这些 mm 标量**不直接 pool**进场比较

**(b) normalized field plane** — 服务连续场与 field 比较，无量纲：
- `x_norm = along_axis / axis_length`
- `y_norm = signed_transverse / axis_length`
- field correlation **只在 normalized plane 上做**

理由：`compute_axis_frame` 给的是 3D mm 下的框架；real-vs-model 场比较若混用 raw mm 和 normalized plane，相关会被绝对尺度污染。

---

## 4. signed-transverse y 轴 + 符号约定 + 镜像规则（P0 lock）

现有 `off_axis` 是**非负**距离（`propagation_skeleton_geometry.py:107`），把左右两边折叠到一起。新增带符号横向：

1. 取参与触点相对 source centroid 的残差向量 `perp_vec`（= `compute_axis_frame` 内部的 `rel - along*u`，沿轴分量已去除）
2. 对 `perp_vec`（参与触点）做 PCA，取第一主方向 `v_perp`
3. `signed_transverse[c] = perp_vec[c] · v_perp`

**B1 符号约定（确定性，仅供画图）**：令"参与触点里 `|signed_transverse|` 最大的那个触点为正"，数据集内固定。

**P0 镜像规则（跨 subject/model 比较核心）**：B1 只保证确定性，**不保证跨 subject 左右同义** —— PCA 符号本就任意，signed-y 是"图上稳定显示坐标"，**不是解剖左/右**。所以跨 subject 与 model-vs-real 的场相关必须 **y-reflection invariant**：

```text
corr_pair = max(
  corr(F1(x, y),  F2(x,  y)),
  corr(F1(x, y),  F2(x, -y))
)
```

写进 spec 显眼处：signed-y direction is deterministic for visualization only; it is NOT anatomical left/right.

退化保护：参与触点 < 3 或 `degenerate_axis=True` → signed-transverse 全 NaN，触发 `weak_axis` flag（见 §10）。

**planarity 保护**：把轴外残差压成单个 y 方向，前提是残差近似一维。若残差本是二维散布，PC1 会丢掉另一半结构。所以记录 `transverse_pc1_variance_explained`（PC1 方差占残差总方差比）；**低于阈值 → `poor_planarity`**，该 subject 的 field comparison 进 supplement 或带 flag（阈值由 writing-plans 钉死）。

---

## 5. 每触点时序聚合（P1 lock — rank 主、time 副）

**张力一**：真实数据每触点时间 = 检测能量重心（`lag_raw`）；模型 = 首次越阈（observation spec 预注册的不对称）。两者不是同一物理量，**rank 对这个差异稳得多**。所以连续场主标量 = rank。

**typical_rank（主，进 real-vs-model 比较）—— 必须按事件参与数归一化**，否则 event size 污染场：

```text
rank_norm(c, e)  = rank(c, e) / max(n_participants(e) - 1, 1)
typical_rank(c)  = median_e rank_norm(c, e),  only over events where c participates
support(c)       = n_participating_events(c) / n_events
uncertainty(c)   = IQR_e rank_norm(c, e)      (rank 版；time 版仅副图)
```

`rank(c,e)` 来自 masked 归一化 rank（phantom 已屏蔽）。`rank_norm` 落在 [0,1]：0 = 最早，1 = 最晚。

**typical_time（副，仅画图，不进主比较）**：plan §3 的事件内时间归一化
```text
lag_rel(c, e)  = lag_raw(c, e) - min_{participating} lag_raw(·, e)
lag_norm(c, e) = lag_rel(c, e) / max_{participating} lag_rel(·, e)
typical_time(c)= median_e lag_norm(c, e)
```
typical_time 只用于人眼对照，**不进 real-vs-model 主比较**（守张力一）。

---

## 6. 连续场 T / S / U（kernel regression，support-gated）

把 `sample_envelopes` 的高斯核搬到 normalized 2D 平面（不是 mm 网格）。每个参与触点提供 `(x_norm_i, y_norm_i, typical_rank_i, support_i, uncertainty_i)`，在规则网格上做 **support 加权 kernel regression**：

```text
w_i(x,y) = support_i * exp(-((x-x_i)^2 + (y-y_i)^2) / (2 sigma_xy^2))
T(x,y)   = sum_i w_i * typical_rank_i / sum_i w_i      （主场 = 归一化 rank）
S(x,y)   = sum_i w_i                                    （支撑权重，NOT 事件率）
U(x,y)   = sum_i w_i * uncertainty_i / sum_i w_i        （不确定度场）
```

锁：
- `S(x,y)` 是**支撑权重**，不是事件发放率，不作率解读
- `sigma_xy` = normalized plane 上的最近邻触点间距中位数；写进 JSON
- 显示规则：`S(x,y) < S_thresh` 的像素**灰掉**，既不显色也**不参与任何相关**
- 空白/低支撑区域不解释成真实结构
- time 场 `T_time(x,y)`（用 typical_time）作并列副图，同样 support-gated，但不进比较

**待 writing-plans 钉死的数值门**（非结论门，属显示/重叠/质量参数；钉死时配一次敏感性检查，验证 §9 的 percentile/z 结论对其不敏感）：`S_thresh`（支撑显示阈）、`overlap-min`（§9 相关所需最少交集像素数）、`grid` 分辨率、`transverse_pc1_variance_explained` 的 `poor_planarity` 阈值（§4）。

---

## 7. SOZ 描述性叠加层（P1 lock — first-contact alias）

桥接 topic3 的那一笔，**仅描述性，不进任何 metric**：
- 从 `results/epilepsiae_soz_core_channels.json` / `results/yuquan_soz_core_channels.json` 取 SOZ core 触点
- **匹配政策锁死 = first-contact alias**（按当前 channel-universe 诊断口径），**不混用** topic3 PR-6 旧 helper 的 bipolar-any matching —— 两者科学含义不同，混用会把 SOZ 点画到错触点上
- 匹配上 → 投影到同一 normalized 平面，叠加一层标记
- 名字不唯一 / 对不上 → flag，SOZ 层留空，**不强配**
- 图注固定写: "SOZ overlay only, not metric input"

Yuquan 双极↔单触点桥接走 topic4 channel-universe 诊断里已验证的 first-contact 规则，不重新推导。

---

## 8. 模型侧：观测层复用（不另写读出链）

模型不重写读出。已有虚拟 SEEG 观测层把模型连续场放虚拟电极（endpoint-centroid 轴 + ≥2 非平行杆，2D montage）、写成真实格式 `*_lagPat_withFreqCent.npz` + `*_montage.json`。

模型就是"多一个 subject/template record"。**若模型输出已有一个或多个模板，本读出逐模板处理**；正反双模板、swap-k、`reverse_axis_cosine` 和 `heldout_rho` **不作为本 spec 的实现前提**，统一列第二批（守观测层"正反/swap 第二批"纪律）。它的 NPZ 走**完全相同**的 §3–§6 读出，产出 `model_subjects/<model_id>_<template>.json`。模型 `lag_raw` = 首次越阈（§5 已说明为何主比较用 rank 不用 time）。

---

## 9. real-vs-model 比较（F lock — 描述性 posterior-predictive，非正式检验）

模型没有病人身份。**禁止**报"p<0.05 model matches cohort"。统一口径 = 模型落在真实分布里的位置。

**聚合纪律（防多模板 subject 过度加权）**：比较的基本单位是 **subject-template record**，但 cohort 汇总时**先 subject 内聚合（一个 subject 多模板取其代表/中位），再 cohort 汇总**——否则模板多的 subject 在 cohort 分布里被重复计数。每个 template record 仍各自独立存盘，汇总层做 subject-first 折叠。

**两类标量分开（关键：避免把"模型内部验证"误当"real-vs-model cohort 比较"）**：

- **real-vs-model cohort scalar**（真实 subject 与模型都有定义；每个 real 一个值 → cohort 分布；模型一个值 → 百分位 / robust z）：
  - `axis_length_mm`、`transverse_width_mm`（mm 标量）
  - `early_zone_spread`、`late_zone_spread`（早/晚端这团的展宽，归一化）
  - `early_late_centroid_distance`（norm 与 mm 各报一个，已有 `endpoint_centroid_axis`）
  - `rank_vs_xnorm_spearman`（rank 顺序 vs 沿轴位置 x_norm 的单调性——**真实数据自身有定义**，不需 `theta_ref`）
- **model-only validation scalar**（依赖**已知真方向**，真实 subject 无 `theta_ref`，**不进** real-vs-model cohort 分布）：
  - `axis_angle_error`（模型估计轴 vs 已知 `theta_EE`，已有 `axis_angle_error_deg`）——仅模型内部 sanity，单独报

**(b) field_correlation_on_support**（本质成对，mirror-invariant，support-gated）：
```text
# 对每个 real subject i:
real_to_real_median_corr(i) = median_{j != i} corr_pair(F_i, F_j)
# 模型:
model_to_real_median_corr   = median_i corr_pair(F_model, F_i)
# 报告: model_to_real_median_corr 落在 {real_to_real_median_corr(i)} 分布的
#       percentile / robust z (median + MAD)。不报 p 值。
```
其中 `corr_pair` = §4 的 y-reflection invariant 相关，**只在双方 `S >= S_thresh` 像素的交集上算**；交集像素过少 → `insufficient_overlap_for_model_comparison` flag，该对不计入。

口径：模型对真实的相似度，是否落在真实彼此之间相似度的范围内 —— 描述，不是检验。

**locked block（原文进 spec / 代码 docstring）**：
```text
Real-vs-model field comparison is performed on a normalized subject/template
plane: x = along_axis / axis_length, y = signed_transverse / axis_length.
The signed y direction is deterministic for visualization only. Cross-subject
and model-vs-real field correlations are y-reflection invariant:
corr = max(corr(F1(x,y), F2(x,y)), corr(F1(x,y), F2(x,-y))).
Correlation is computed only on the intersection of sufficient-support pixels;
otherwise the comparison is flagged insufficient_overlap_for_model_comparison.
Primary scalar field = median event-wise normalized masked rank. Time field is
secondary visualization only.
```

第二批（依赖正反双模板，本 spec 不实现）：`reverse_axis_cosine`、`heldout_rho`。

---

## 10. 质量 flags + 口径锁

**flags**（plan §9 全保留，决定进主图 / supplement）：
`one_dimensional_sampling` / `poor_planarity` / `low_contact_count` / `low_support` / `weak_axis` / `rank_unstable` / `insufficient_overlap_for_model_comparison`。
1D / participation 判定复用骨架模块，不重写。**`poor_planarity` 具体来源 = §4 的 `transverse_pc1_variance_explained` 低于阈值**（轴外残差非近似一维，PC1 丢结构）；触发后该 subject field comparison 进 supplement 或带 flag。

**口径锁（写进 spec 显眼处 + 图注约束 + main-doc 引用约束）**：
- 早端必须画成"一团带不确定度的小入口群"（support 场 + 不确定度场显式呈现）
- **禁止**措辞: "固定稳定早端" / "弹性入口区" / "two fixed endpoints"
- **要求**措辞: "有抖动的刻板传播通路 + 轮流当早端的小群"
- 依据: entry-dispersion 那轮结论（95 个里 82 个早端分散，严格两固定端点被否决）

---

## 11. 代码结构 + 输出目录

**新模块** `src/propagation_contact_plane_readout.py`（只装新东西）：
- `signed_transverse_axis(perp_vec, participating_mask)` → `signed_transverse`, `v_perp`, B1 符号
- `build_readout_record(...)` → 每触点聚合 + normalized 坐标 + flags（标准化 record，real/model 同构）
- `smooth_field(record, sigma_xy, grid, scalar)` → T/S/U 网格场
- `corr_pair_mirror_invariant(F1, F2, S1, S2, S_thresh)` → §4/§9 相关
- `compare_model_to_cohort(model_record, real_records)` → §9 比较 record
- SOZ overlay 由 plotting 侧消费 record 的 `soz_overlay` 字段

轴 / 端点核 / stereotypy 调 `propagation_skeleton_geometry`；高斯核调 `sef_hfo_observation.sample_envelopes` 思路。

**runners / plotters**（`scripts/`）：
- `run_contact_plane_readout.py`（real subjects → `real_subjects/*.json`）
- `run_model_contact_plane_readout.py`（模型 NPZ 走同一读出 → `model_subjects/*.json`）
- `run_real_vs_model_comparison.py` → `comparison/real_vs_model_summary.json`
- `plot_contact_plane_static.py`（静态图 + SOZ overlay）
- `plot_contact_plane_animation.py`（deferred）

**输出目录**（AGENTS.md 规范；topic 分类不带 PR 号）：
```text
results/spatial_modulation/propagation_geometry/observation_readout/
├── real_subjects/
│   └── <dataset>_<subject>_<template>.json
├── model_subjects/
│   └── <model_id>_<template>.json
├── comparison/
│   └── real_vs_model_summary.json
├── figures/
│   ├── README.md            ← 必须存在，中文，逐图说明 + 关注点
│   ├── static_maps/
│   └── animations/          (deferred)
└── README.md
```

---

## 12. 测试（TDD — 每条一个会失败的测试）

1. **镜像不变相关**: 把一张场沿 y 翻面，`corr_pair` 必须 = 原值（不变）；普通相关会掉 → 证明 max-over-mirror 生效
2. **rank 事件内归一化**: 同一空间结构、两组 event size 不同 → `typical_rank` 场必须一致（不被 event size 污染）
3. **support gate**: `S < S_thresh` 像素既不显色也不进相关；构造一个高支撑 + 一个低支撑区，相关只算高支撑交集
4. **低 overlap flag**: 两 record support 交集像素 < 阈 → `insufficient_overlap_for_model_comparison`，该对不计入比较
5. **signed-transverse 符号确定性**: 同输入两次跑，B1 符号一致；左右镜像输入 → signed_transverse 整体变号（不是随机）
6. **SOZ 匹配政策**: first-contact alias 命中正确触点；名字对不上 → SOZ 层留空 + flag，不强配
7. **mm/normalized 不混**: 标量指标用 mm、field 用 normalized；构造缩放输入，field 相关不变、mm 标量随缩放变
8. **退化轴**: 参与触点 < 3 / `degenerate_axis` → signed-transverse NaN + `weak_axis`，不崩
9. **1D / poor-planarity flag**: 单维采样触发 `one_dimensional_sampling`（复用骨架判定）；二维散布残差 `transverse_pc1_variance_explained` 低于阈触发 `poor_planarity`
10. **单位不变时间归一化**: `lag_raw` 同结构、一份秒一份 ms（×1000）→ `typical_time` 场必须一致（事件内归一化消单位）；`lag_time_unit` 正确落盘
11. **多模板不过度加权**: 同一 subject 给 N 个模板 → cohort 汇总按 subject-first 折叠，该 subject 在分布里权重 = 1（不是 N）

---

## 13. 执行顺序（动画 deferred）

1. 2–3 个真实 subject 静态读出 → 目视验证投影 + 支撑 mask 合理
2. 连续场 T/S/U（含 mirror-invariant + support gate）
3. 模型走同一读出
4. real-vs-model 指标（标量 percentile + field descriptive posterior-predictive）
5. 扩展到全 cohort
6. 动画（最后）

每步后更新 archive doc（守"每次关键探索后更新 docs"）。

---

## 14. 预注册：这套读出能 / 不能下的结论

- **能**: "模型生成的传播时序，落在真实 cohort 传播时序相似度的范围内 / 之外"（描述性）
- **能**: "真实早端这团 + 传播场，与 SOZ 的空间相对位置"（描述性展示）
- **不能**: "模型解释 SOZ" / "p<0.05 model matches cohort" / "有固定稳定早端"
- 比较单位 = 模型 vs 真实 cohort 分布；SOZ 仅描述性叠加，永不进 metric
