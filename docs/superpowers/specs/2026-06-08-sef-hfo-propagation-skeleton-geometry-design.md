# SEF-HFO 间期传播几何骨架 — 设计 spec（2026-06-08）

> **状态**：探索性、**描述性测量（model-input descriptive），不是假设检验，不是 SOZ 定位器**。
> **上游**：`docs/topic4_sef_itp_framework.md`（SEF-HFO v0.2）、数据侧 plan `docs/superpowers/plans/2026-06-06-sef-hfo-soz-localization-rate-vs-geometry.md`、low-rate 读回结论 `docs/archive/topic4/sef_hfo/low_rate_template_stability_2026-06-07.md`。
> **硬前置**：`lagPatRank` phantom-mask（Topic 0 §3.1）；本 spec 的源/汇 core 只用 phantom-safe 的 `rank_a_dense_full`（非参与通道为 NaN，已实测 40/40 验证 rank=0 不可能是 phantom）。

---

## §0 一句话（朴素话）

每个病人的间期 HFO 事件有一条稳定的传播顺序（谁先放电、谁后放电）。这一轮我们**纯几何地量这条传播骨架长什么样**：起点斑块多紧凑、终点斑块多紧凑、从起点到终点这条轴有多长、这条传播"通道"有多粗（横向铺多宽），以及沿着这条轴走、时序刻板性能延多远。**目的是把这几个空间尺度数喂给 SEF-HFO 模型**（易激斑块尺寸、传播轴长度、横向耦合尺度），不是用来定位 SOZ（已确证稳定端点≠临床 SOZ，PR-6 H1 NULL），也不是这一轮就检验"几何比率更稳"（那是下一轮）。

## §1 为什么换成轴坐标系（不是源点球半径）

源点球半径会把两件**对模型不同**的事混在一起：**沿传播方向变远**（正常传播）和**偏离传播轴变远**（问斑块多宽、边界在哪）。轴坐标系把它们拆开：
- **沿轴坐标** a = 通道相对源 centroid 在轴方向的投影（0=源…L=汇，可 <0 或 >L）。
- **离轴坐标** d = 通道到轴线的垂直距离。

沿轴是传播本身；离轴才回答"这条传播是一条窄轴，还是一片散场"。

## §2 队列 + eligibility 分档（数值门，编码结论）

- 候选 = 同时有 phantom-safe 端点文件（`results/interictal_propagation_masked/rank_displacement/per_subject/{ds}_{subj}.json`，40 个）**且**有 3D 坐标的被试。实测可行 = **11 Yuquan(SEEG) + 20 Epilepsiae(SEEG+ECoG) = 31**，减事件质量门（如 pengzihang total_hours=2.0 坏数据）→ **~29–30**。
- `n_eff` = **既参与（joint-valid）又有有效 3D 坐标**（loader `mapped_mask=True`）的通道数。
- **分档（防止 2k 端点吃掉整个蒙太奇 → 伪源/汇轴）**：
  - `n_eff ≥ 7` → **k=3，primary cohort**。
  - `n_eff ∈ {5,6}` → **k=2，fallback 档，flagged，不并入 k=3 主统计**（k 不同→core 定义不同）。
  - `n_eff < 5`，或分配 core 后 interior 为空，或源/汇 centroid 重合 → **subject-descriptive only，排除出 cohort 主统计**。
- 每被试落档 + 落档计数都写进 cohort summary（不静默截断，AGENTS.md "no silent caps"）。

## §3 数据源 + phantom-safety

- **每事件 rank + 参与 bool**：`*_lagPat_withFreqCent.npz`（10ch 全集）+ packedTimes；走 low-rate 已有的 masked 加载（`src/low_rate_template_stability.py`）。
- **模板（定义源/汇）**：
  - **非-swap 被试** → 全程 phantom-masked 模板（所有事件 mean rank）。
  - **swap-positive 被试**（判据 = `rank_displacement` `pairs[0].swap_sweep.swap_class ∈ {strict, candidate}`，单一字段，不掺 decision_k 显著性）→ **用 dominant cluster（事件多的那簇）的 per-cluster 模板**作主轴；**绝不把反平行的 forward/reverse 平均成一条糊轴**。次簇轴只描述性报。（落实 advisor #4，但用 dominant-cluster 而非剔除，保住可 pool 性；糊平均数永不入池。）
- **源/汇 core（phantom-safe）**：从该模板的 dense rank（`rank_a_dense_full` 风格，非参与=NaN）取最小 k 个 = 源 core、最大 k 个 = 汇 core。**core 用 `nanargmin/nanargsort`**，NaN 不可能进 core。
- **坐标**：`src/seeg_coord_loader.py::load_subject_coords(ds, subj, channel_names)`；只在 `mapped_mask=True` 的通道上算几何。Yuquan native RAS mm；Epilepsiae MNI152 mm。

## §4 几何骨架（五个 deliverable 里的四个纯几何量）

复用 `src/sef_itp_phase1.py`：`pairwise_3d_euclidean`、`_centroid_distance`、`_mean_pairwise_distance`；MEB / max-pairwise 若 phase2/3 已有则复用，否则补最小实现。新写：轴投影 + 垂距。

1. **source radius / sink radius** = core 内通道到各自 centroid 的 **RMS 距离**（主）。
   - **advisor #3（双峰 core 坑）**：RMS-to-centroid 在 core 跨两根杆/两个亚灶时，centroid 落在中间空谷、半径量的是"间隙"不是紧凑度。所以**并报 min-enclosing-ball 半径 + max-pairwise**，并 flag "core 跨 >1 杆"。
2. **source-sink axis length** L = |汇 centroid − 源 centroid|。
3. **perpendicular width** = 参与通道垂距 d 的 **RMS（主）**；**p75 / p90 作稳健敏感性**（不用 MAD 当主——MAD 会吞掉少数远端参与通道，正好抹掉"散场"信号）。
   - **advisor #1（这一轮最关键的坑：垂宽重新引入 participation=rate 寄生）**：垂宽是**参与通道**的铺展，而"参与"=joint-valid=事件够多=高率。所以垂宽是被采样+率**圈住的下界**，和 temporal 层不同，**它本身没有对照**。强制两条纪律：
     - (a) **参与阈敏感性**：像 temporal 层带 matched-null 一样，给垂宽带"joint-valid 阈值扫描"——报垂宽随参与阈移动怎么变。
     - (b) **诚实命名**："**participating-channel perpendicular spread**（参与通道垂向铺展，采样+率条件下的下界）"，**不叫** "patch width / coupling width"。
     - (c) **可测性地板**（独立于 k-eligibility）：n_eff=7、k=3 时 interior 仅 1 通道，垂宽退化为近轴 core-scatter；这种被试垂宽标低可信度。
   - **1D 退化 flag**：参与通道全在一根杆/共线（解析 shaft id；或 p90(d) < 一个触点间距 ~4mm）→ 垂宽标 **NA(1D sampling)**，单独列；**轴长仍报**（沿杆 1D 传播是真的）。把"多少被试单杆"升级成"1D 链 vs 2D/3D 散布斑块"分类。

## §5 沿轴时序刻板剖面（第五个 deliverable，描述性）

- 每参与通道一个**刻板性**标量：它在参与事件里"事件内分数排名（0=最早…1=最晚）"的集中度。
- **participation-matched null（防"远=事件少"混淆）**：复用 low-rate 的时间打乱思路，每通道在其真实事件数下得随机基线，报 **excess（null-SD 单位 z）**，chance=0。
- **沿轴剖面（这一轮报）**：excess vs 沿轴坐标 a（含 a<0、a>L），看刻板模式沿轴延多远、端点是不是硬边界。
- **离轴剖面（这一轮只作描述性附图，不抽 gated cohort 标量）**：excess vs 垂距 d。注：把"离轴耦合宽度"提成有 null-crossing 阈的 cohort 标量、以及"几何 vs 率稳定性"，都**留到下一轮**（避免 spec 变胖、验收变糊）。
- 注意沿轴方向刻板性部分是模板**定义**出来的（rank≈沿轴位置），所以沿轴剖面作延展/边界 sanity，离轴才是真正新信息——但这一轮离轴只描述。

## §6 跨数据集 mm 不可直接 pool（advisor #2）

- Yuquan = native RAS = 真解剖 mm；Epilepsiae = MNI152 mm，loader 自带 tag `warp_type_unverified`。MNI 把每个头缩放到模板 → Epilepsiae MNI-mm 与 Yuquan native-mm **系统性不同**（若非线性则局部畸变）。
- **四个 mm deliverable 一律按数据集分层报**，Epilepsiae 明标 "MNI-space mm"。**within-dataset + Yuquan-native 数干净**；跨数据集 pooled mm 只在带此 caveat 时给。喂模型的是分层数，不是糊 pool 数。

## §7 SOZ = 标注 + 敏感性（描述性，无显著性主张）

- **主宇宙 = 所有参与通道**，不限 SOZ。
- 事后描述性报：源/汇 core、轴、垂宽 与临床 SOZ 的 set-relation（复用 `clinical_soz_set_relation`）。**不对 SOZ relation 下显著性结论**——这正是 advisor 确认的、避开 garden-of-forking-paths 的保护：本轮是描述性测量，不把模型约束绕回临床标签。

## §8 输出

- **per-subject JSON**：`{n_eff, k_used, eligibility_tier, swap_status, template_source, sampling_geometry(1D/distributed), source_radius_rms/meb/maxpair_mm, sink_radius_..., axis_length_mm, perp_spread_rms/p75/p90_mm, perp_width_participation_sweep[], perp_measurability_flag, along_axis_profile[], offaxis_profile_descriptive[], soz_relation, coord_space, missing_coords[]}`。
- **cohort summary JSON**：分档计数、分数据集分层的四个几何量分布、1D vs distributed 计数、swap-tier 计数。
- **figures/**（每图一问，CLAUDE.md §7；配中文 `figures/README.md`）：
  1. 每被试轴坐标散点（x=沿轴 a，y=离轴 d，色=刻板 excess）+ 源/汇 core + 轴 — 看骨架形状。
  2. cohort 四个几何量分布，**按数据集分层**（Epilepsiae 标 MNI-mm）+ 1D/distributed 拆分。
  3. 沿轴刻板 excess pooled 剖面（within-subject z 后可 pool）。

## §9 验收（描述性 deliverable + sanity 门，编码结论）

完成 = 全部成立：
1. **phantom-safe 中心**：每被试源/汇 core 全为 joint-valid 参与通道（NaN 不入 core）——脚本断言 + cohort 报 0 违例。
2. **分档数值门生效**：n_eff<5 / interior 空 / 退化轴的被试**确实**落 descriptive-only、不进主统计；分档计数写盘。
3. **垂宽双纪律**：(a) participation-sweep 报了；(b) 命名是 "participating-channel perpendicular spread"；(c) 1D 被试标 NA 单列。缺任一 = 不通过。
4. **跨数据集分层**：四个 mm 量分数据集报，Epilepsiae 标 MNI-mm；无静默跨集 pool。
5. **swap 路由**：swap-positive 用 dominant-cluster 轴，糊平均数不入池。
6. **双峰 core 守卫**：MEB+max-pairwise 并报，core 跨杆 flagged。
7. 图已生成 + 目视 + `figures/README.md` 中文写了关注点。
- 全程 TDD（RED→GREEN）；测试覆盖：轴投影/垂距数值正确、NaN core 守卫、1D 退化 flag、分档边界（n_eff=4/5/6/7）、swap 路由、跨数据集不 pool。

## §10 不在本轮（下一轮 / 显式 out-of-scope）

- "几何骨架是否比事件率定位更稳"（split-half 几何 vs 率稳定性）= **下一轮完整问题**。
- 离轴**时序耦合宽度**提成带 null-crossing 阈的 cohort 标量 = 下一轮（本轮只描述性附图）。
- 跨被试绝对坐标 pooling（Yuquan native 不可跨被试）= 不做。
- 把任何本轮量写进 topic4 主文档 paper 级口径 = 不做（探索性、无 held-out，archive-only）。
