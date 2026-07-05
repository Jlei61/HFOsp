# Cohort field-swap subject-SNN: coverage↔balance tradeoff + batch (2026-06-27)

## 朴素话 abstract（测了什么 / 怎么测的 / 揭示了什么）

**测了什么**：想给所有"间期有两类互为反向传播模板（stable_k=2）"的病人，各自摆一个
subject-SNN（两个低阈值灶放在两类模板各自的最早区，中间一条 E→E 长轴），看模型能不能
自发读出这两类模板。用户要求**把灶调大，让病人布局里大多数电极都能被传播波及**（之前
1146 每次事件只点亮约 40% 的电极）。

**怎么测的**：在 1146 上扫"灶大小"（core_r=1.5/2.5/4/6）和"背景驱动"（drive），量每次
事件覆盖的电极比例 + 正/反两个方向还在不在；又单独驱动一个灶看能不能既高覆盖又单向干净。

**揭示了什么（核心结论 = 一个模型固有的权衡）**：
- 灶越大 → 覆盖越高（per-event 0.41→0.67，union 0.73→1.0）✓ 用户直觉对。
- **但**两个灶都大 → 一个灶抢先点火压住另一个 → 自发活动几乎单向（cr6 = 1正/7反），
  **"两个模板互换"就消失了**（任何 seed 都这样）。
- 加大 drive 想救覆盖 → 直接把离散事件烧没（drive 0.75 → 0 个干净事件）。
- 把单个灶做大想"无竞争地高覆盖" → 单灶也失去方向干净度（source-only cr5 = 3正/3反混合，
  不是干净正向）——所以"分开驱动再合并"这条逃逸路线也失败。
- **结论（口径受限）：只在 E1146 上真扫了 core_r/drive**——所以严格说法是
  **"E1146 calibration + cohort screen 提示 高覆盖与可见互换存在张力"**，不是 cohort 层面证实。
  其余 24 个被试是**单 seed、单参数 screen**，没有逐被试 core_r/drive sweep。要 cohort 证实需对
  每个候选被试多 seed 或至少 core_r sweep 子集。与 [[project_topic4_sef_hfo_snn_stage3_plan]] 的
  twoend_equal one-core-dominance 是同一机制。

## 决策

图的科学目的就是展示两类模板互换（Fig4B 的 KMeans=2），所以**保互换优先，覆盖在不破坏
平衡的前提下尽量提高**，不为覆盖牺牲互换。

- core_r 规则（按几何缩放，per-subject）：`clamp(0.22 × inter_core_sheet, 2.5, 3.5)`，
  再 `cap 0.42×ic`（防两灶融合）。多数被试落在下限 2.5（cov≈0.44，平衡，direction
  purity 1.0）。
- 宽电极云被试（plane-fit 把核间距压小：xuxinyi ic1.6 / 922 / huangwanling / zhangbichen）
  标记 `cores_close_wide_cloud`，best-effort 跑但需人工复核。
- 仿真 = 自发 twoend，每被试 1 条 T=8000 seed3；**single-run Fig4A**（tempA/tempB 事件面板
  取自同一条 twoend 的 rep_fwd/rep_rev）+ montage-aware Fig4B。

## 校准数据（E1146, inter_core_sheet=13.05mm, T=4000）

| core_r | per-event cov | union cov | fwd/rev | minority frac | 备注 |
|--------|---------------|-----------|---------|---------------|------|
| 1.5 | 0.41 | 0.73 | 5/6 | 0.45 | 平衡，低覆盖 |
| 2.5 | 0.44 | 0.60 | 6/5 | 0.45 | 平衡（采用基准） |
| 4.0 | 0.47 | 0.73 | 1/3 | 0.25 | 失衡 |
| 6.0 | 0.67 | 1.00 | 1/7 | 0.12 | 高覆盖但单向 |
| probe cr3.5 d0.75 | 0.00 | 0.00 | 0/0 | — | drive 过高烧没事件 |
| probe cr2.5 d0.85 | 0.53 | 0.53 | 0/1 | — | 同上 |
| driven source-only cr5 | 0.43 | 0.80 | 3/3 | — | 单灶大→方向混合 |
| driven sink-only cr5 | 0.59 | 1.00 | 2/8 | — | 单灶大→方向混合 |

## 队列（stable_k=2 + 两个 distinct 模板源灶）

- **25 RUN**（见 `scratchpad/cohort_config.json`），narrow 默认，narrow 退化时用 broad。
- **8 SKIP**：583/1084/548/yuquan_huanghanwen/litengsheng/pengzihang/zhangkexuan
  （两模板最早区是同一批电极 = 没有可放两灶的端点互换）；1073（几何是 3D，未实现 3D→2D）。

## 队列结果（2026-06-27 批处理完成；分母不混淆"文件存在"与"科学可解释"）

- **25 run / 25 Fig4A / 21 Fig4B generated / 9 bidirectional（≥3 正 ∧ ≥3 反）/ 5 model-real PASS**
  （4 个 Fig4B 失败=事件太局部/太少：1125/384/635/1096）。
- verdict 分布：keep 4 + keep_geom_caveat 1 + bidir_no_real_match 4 + one_direction_only 12 +
  fig4b_failed 4。
- **keep（主图候选，= 双向 ∧ model-real swap 匹配）**：yuquan_zhaojinrui（14/13，对角 p0.005/0.001 最强）、
  yuquan_liyouran（4/4）、yuquan_zhaochenxi（20/6）、epilepsiae_916（fa 不显著 p0.11，边缘）、
  yuquan_songzishuo（geom caveat）。
- **关键纪律：主图不按 coverage 排名挑。** epilepsiae_620 覆盖 0.88 且双向，但 forward 不匹配 t_a
  （fa=−0.12）→ model-real 不过 → 落 bidir_no_real_match。
- **Fig4B 单向 gate（P0 修复 2026-06-27）**：model fwd/rev 模板按事件 SIGN 直接构造（不再用 cluster→方向
  的假映射）；右侧 fwd/rev × t_a/t_b 矩阵仅在 ≥MIN_DIR_EVENTS(3) 正 ∧ ≥3 反 时绘制，否则标 N/A
  "one-direction diagnostic only"。

## 复现（in-repo）

```
python scripts/paper_figures/build_cohort_field_swap_config.py        # -> _cohort_field_swap_snn/cohort_config.json (25 run/8 skip)
python scripts/paper_figures/run_cohort_field_swap_snn.py             # 内存自适应≥40GB,并发10; twoend T=8000 seed3 -> readouts + Fig4A/4B + cohort_index.json
python scripts/paper_figures/cohort_field_swap_summary.py             # -> cohort_summary.{csv,json} + README + 缩略图墙(verdict 排序)
```

- 仿真单线程 ~33min@T4000（CPU 非瓶颈，内存+墙钟是）；共享机器礼让用户并发 loop。
- 单图：`plot_fig_subject_snn.py`（single-run = 省略 --source-tag/--sink-tag）+
  `plot_fig_subject_snn_kmeans2.py --montage {narrow,broad}`。
- 覆盖优先对照样张：`results/paper-ready-figure/fig_subject_snn_epilepsiae_1146_COVERAGE_VARIANT/`
  （cr6，union 1.0 但 1正/7反）。

## 根因（用户验收时点出 + 几何验证 2026-06-27）：source 被多节点分散

用户判断："很多仿真不达标的根本是 source 的位置被很多节点分散了。" 用现有几何验证（earliest-3
source 触点的空间跨度 srcA_max/srcB_max，mm，按 verdict 分组均值）：

| verdict | srcA 跨度均值 | 解读 |
|---------|---------------|------|
| keep | 12.0mm（3/4 ≤8mm 紧凑） | 至少一侧是聚焦点火灶（zhaochenxi 4.6 / liyouran 6.5 / 916 7.8；zhaojinrui A散28.9 但 B紧5.9） |
| bidir_no_real_match | 36.0mm | 1150 双侧都散(88/62) 无干净源；620 A散(40) → forward 不匹配 t_a |
| one_direction_only | 25.0mm | 典型 一侧紧一侧散 → 只有紧侧干净点火（1146 A散23.6→0/17；huangwanling A散32→0/28） |
| fig4b_failed | 18.9mm | 源散 + 核近 |

**确认**：某方向要干净读出，**那一侧 source 必须是空间紧凑的点火灶**；earliest-k 触点空间散开
→ 核不是尖锐起点 → 该方向读不出（单向 / 不匹配 / 无事件）。接上 [[project_topic4_entry_dispersion_outcome]]
"真数据入口=小入口群、有抖动刻板通路、禁弹性入口区"。例外 xuxinyi（源紧但 22/0）= cores_close_wide_cloud
（核间距 1.6mm 几乎重叠），是另一个已标问题。

**下一轮修复方向**：placement 不用"按 typical_rank 取最早 k 个触点"（可能空间散），改选**空间紧凑早区**——
最早单触点 + 空间近邻 / 要求 earliest-k 空间相邻 / 自适应缩 k 至源聚焦。即"小入口群"代替"rank 前 k"。
`src/sef_hfo_subject_placement.py::template_source_foci` 是改造点。

## 待用户拍板

1. 主图从 5 个 keep 候选里挑（推荐 zhaojinrui / liyouran / zhaochenxi 三个对角双显著的）。
2. balanced-swap（当前）vs coverage-priority（高覆盖单向）是政策选择，不是 bug。
3. 是否救 4 个 fig4b_failed（1125/384/635/1096）——非最高优先级（救出来也只是扩大口径不清的池）。
4. 是否对 keep 候选做多 seed / core_r sweep 以把"E1146 calibration"升级为 cohort 证实。
5. 宽云被试（cores_close_wide_cloud）可考虑 core-anchored 注册作为第三条路线。
