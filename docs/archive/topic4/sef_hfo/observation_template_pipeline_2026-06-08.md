# 虚拟 SEEG 观测层 → 真实模板流水线（整合里程碑 (b)，2026-06-08）

## §0 揭示了什么（朴素话）

**测了什么** — 我们把模型（率场）产出的"事件"透过虚拟电极读成"每个事件里各通道的发放先后排名"，
再喂给我们分析真实病人 HFO 用的**同一套聚类流水线**，看它能不能跑通、能不能自动找出传播模板。

**怎么测的** — 造了一批模型事件，一半从连接长轴的一端点火（波往一个方向传）、一半从另一端
（波往反方向）；每个事件加一点观测噪声（均场模型本身是确定性的、没有事件间随机性，而真实
SEEG 有噪声）。把这些事件的排名矩阵交给真实聚类函数。如果流水线只认真实数据、吃不了模型产出，
就会报错或产出空结构；能吃的话，应自动分出若干模板并标出哪些是"一正一反"。

**揭示了什么** — 流水线**干净跑通**、产出了与真实病人**同样字段结构**的模板结果，自动分出
**2 个模板**、且这俩是"一正一反"（沿同一条轴方向相反），和真实病人看到的结构一样。但必须说清楚：
这只证明"**管道接通了、产出可解读**"（工程 + 合理性），**不**证明"模型重现了病人机制"——因为这个
模型只有**一条全局连接轴**，"一条轴两个方向 = 2 个模板"是被结构**强制**的，换任何点火方式都会得到 2。
真正的机制检验（异质连接能不能产出对的模板结构）是后面的异质核工作，不是这一步。

（内部归档代号：(b) = 把 model 写盘/读出走真实 PR-2 `compute_adaptive_cluster_stereotypy`；
estimator/观测层见 [[project_topic4_sef_hfo_observation_layer]]；(a) 真实几何注册
`from_real_geometry` 用户 2026-06-08 判定暂不需要。）

## §1 这一步是什么（advisor 2026-06-08 framing）

**整合里程碑**：模型产出**无特判**地穿过真实聚类流水线、落进真实 subject schema、字段对字段可与真实
病人并排比。报告口径 = 管道接通 + 结构可解读，**绝不**写成"模型重现病人机制"。原因：单一全局连接轴 →
模板空间 ~1 维 → stable_k≈2（正/反）是被强制的，不是被发现的。

## §2 方法

- **事件生成**（`scripts/run_sef_hfo_obs_template_pipeline.py`）：率场 C1 配方（`_integrate`，θ_EE=45°，
  AR=2），forward 臂踢负端（波 →+）、reverse 臂踢正端（波 →−），各 60 个事件。
- **读出**：一套固定虚拟 montage（3 根非平行杆 15/75/135°，真实 4 mm 间距，与四对照同），
  `rate_event_envelope` → `extract_lagpat` → 每事件一个排名向量 + 参与布尔。
- **臂内变异注入**：① 垂直种子抖动（perp jitter ±0.18×半片）；② 每事件观测噪声（env 上加
  `noise_sd×(env range)` 的高斯，默认 0.015）。
- **真实流水线**：把 (ranks, bools, channel_names) 直接喂
  `src.interictal_propagation.compute_adaptive_cluster_stereotypy(use_masked_features=True)`
  ——run_interictal_propagation 调的同一函数；输出 stable_k、每簇 stereotypy、簇间 Spearman
  相关矩阵 + 候选正/反对标注。

## §3 工程发现（confirmed，决定了为什么这样设计）

（这一节是 (b) **流水线的事件生成**发现；双电极**读出图**的读法 + 三个工程发现见 §6。）

1. **连接轴只能取 θ∈{0,45,90}**：周期网格上中间角度（37/41/49/53° 实测）**产不出可检测事件窗**
   （周期边界 × 各向异性相互作用）。→ 轴角抖动**不可用**作臂内变异；θ 固定 45°。
2. **率场是确定性的**：固定轴 + 仅 perp/工作点抖动 → 臂内排名**顺序**几乎不变（spread≈0，退化），
   聚类的稳定性/silhouette 闸门会被近重复事件噎住。→ 臂内变异**必须注入观测噪声**（real SEEG 有噪声），
   噪声抖动各触点首次越阈 → 排名顺序产生真实臂内 spread。
3. **正端（reverse）踢未抖动时正常**（基线 12 触点参与，有窗）；**沿轴抖动会把踢点推进周期边界 → 事件失败**。
   → 只用 perp 抖动，不用沿轴抖动。
4. **噪声-参与权衡**：noise_sd 太大（≥0.02）→ 噪声把所有触点推过参与阈（18/18 假参与）；太小（0.012）→
   参与真实（~12）但正/反对的簇间反相关不够强、自动标注不触发。取中（0.015 + 120 事件）兼顾。

## §4 结果（120 事件，noise=0.015）

产物 `results/topic4_sef_hfo/observation_layer/template_pipeline/model_subject_adaptive_cluster.json`。

- 120 事件（forward 60 / reverse 60），参与触点/事件：min 12 / 中位 15 / max 18（真实, ≥10）。
- 臂内排名 spread：forward 1.0 / reverse 1.07（非退化）。
- **stable_k = 2**（chosen_reason=stable_k），两簇各占 **50%**（cluster 0/1, fraction 0.5/0.5）。
- **簇间 Spearman = −0.762**（inter_cluster_corr_matrix [[1,−0.762],[−0.762,1]]）→ 自动标
  **candidate_forward_reverse**（cluster_a=0, cluster_b=1）= 两模板沿同一轴方向相反。
- 真实流水线完整字段都产出了：`scan / stable_k / clusters / inter_cluster_corr_matrix /
  candidate_forward_reverse_pairs / overall_tau / within_cluster_tau_mean / uplift / labels`。

**判定**：真实聚类函数 `compute_adaptive_cluster_stereotypy`（run_interictal_propagation 调的同一个）
**无特判地吃下模型产出**，产出 schema-valid 的 2-模板正/反结构。这是管道接通 + 可解读的工程里程碑
（见 §0/§5 边界）。**下一增量**（待用户）：把模型 artifact 写盘后跑**完整 `run_interictal_propagation`**
脚本（而非只调聚类函数）→ 产出与真实 subject 完整 schema 并排的 model-subject JSON。

## §5 诚实边界

- **是**：插件式接通真实流水线 + 产出 schema-valid、可解读的正/反结构。
- **不是**：模型重现病人机制（单轴强制 2 模板）；真实病人电极几何（(a) `from_real_geometry` 仍是
  loud-fail stub，用户判定暂不需要）；脉冲网络真实尺度（仍 3 mm 亚毫米，见
  [[project_topic4_sef_hfo_observation_layer]]）。
- **下一步真正的机制检验**：异质连接核 → 看模板结构是否随连接而变（Increment-3b / Step-3，仍锁"待机制锁"）。

## §6 双电极读出图的工程发现（同期交付，2026-06-08）

两张论文级双电极图（`results/topic4_sef_hfo/observation_layer/figures/two_electrode_readout_{rate,snn}.png`，
共享画图器 `src/sef_hfo_plot.py`，提交 e385dd1；重画 `python scripts/plot_sef_hfo_two_electrode_readout.py`）。
读法：电极**平行**连接长轴 → 各触点波峰**斜着扫过** = 读出方向；**垂直** → 波峰**对齐** = 读不出。
下面三条是用户复核（2026-06-08）问出来的，前两条改了图、改了画图代码。

1. **率场为什么只有一条长轴亮（不是 bug，是机制本身）**：E→E 兴奋沿 45° 长轴的连接范围
   `ELL_PAR=0.54` 是垂直向 `ELL_PERP=0.27` 的 **2 倍**，抑制短程各向同性（`L_INH=0.25`）。所以一处点着后，
   复发兴奋优先沿长轴接力、垂直向被抑制掐灭 → 过阈事件 footprint 成一条沿长轴的对角带。这条带就是
   "连接长轴被显形"，是 SEF-HFO 的核心机制（连接定方向）本身，不是图错。

2. **波峰窗口钳制假象（真 bug，已修 `src/sef_hfo_plot.py::_trace_panel`）**：原灰带 = `event_window_for_run`
   的**场事件窗**（整片像素过阈比例，率场实测 25.25–121.75 ms）。但这个窗比波**扫过整根电极**所需时间窄：
   平行电极各触点真实峰跨 **19–147 ms**（最近 P0 在 19 ms 在窗**前**，最远 P4/P5 在 145/147 ms 在窗**后**）。
   旧实现把波峰限制在窗内 argmax → 首/尾触点被**钉在窗边**，斜线两端是人为产物（内部 P1→P3 的
   33→70→105 ms 才是真扫过）。**修法**：波峰改在**整条波形**找真实到达（`peak_idx=argmax(sig[i])`），
   灰带改成"**这根电极把事件录全的时间跨度**"（首到尾参与触点峰）→ 峰不再钳边、斜线全真。

3. **SNN 上 P8（用户读成"O8"）为什么最早响应**：标注旧字号 5.5 太小被误读成 O8 → 已放大 + 白描边
   （SNN 触点密、两杆中心 `P4≡Q4` 坐标重合，改成只标杆端 P0/P8、Q0/Q8，中间触点按杆序对应 B/C 波形叠放）。
   P8 最早，是因为它**离种子最近**（距 end-kick 0.78 mm，P0 距 2.86 mm）：事件在种子点着、沿对角线往外传
   → 最近触点先峰，峰时随距离**单调**推移（P8 184.8 ms → P0 188.7 ms）。**证明是行波而非单纯远近**：
   平行电极 ~3.9 ms 单调扫过 vs 垂直电极同 ±1.04 mm 跨度仅 ~0.5 ms 几乎同时 = **8 倍反差**才是方向读出的
   证据。附注：幅度反而**非单调**（最近 P8 峰 peakZ 最小 ~380、最远 P0 最大 ~813，种子处去同步/募集未起）
   → 时序是干净读出、幅度不是。

读法口径 + colorbar 改相对时间（0 = 踢后第一个激活）+ 连接轴只画箭头不写文字，均同步进
`results/topic4_sef_hfo/observation_layer/figures/README.md`。技术上承接 Increment-3a rate parity /
LFP forward（见 [[increment3a_rate_parity_2026-06-07]] §3/§6）。
