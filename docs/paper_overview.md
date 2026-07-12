# 论文总论与文档索引

> 状态：当前正式入口
> 目的：给人和 Agent 一个稳定的总索引，先回答“这篇论文现在到底讲哪 3 个 topic、各自结论是什么、应该先读哪里”。

---

## 0. Topic 0：方法学审计与数据合同（**优先级最高**）

> 在读任何 Topic 1–5 的科学结论之前，必须先看本节是否有未结清问题。

- 正式入口：`docs/topic0_methodology_audits.md`
- 当前未结清问题：**0 完全未结清 + 2 个已基本结清（科学层 PASS，工程层 1 个仍在收口）**
  1. **（科学层 PASS，工程层 5i.6 进行中）** `lagPatRank` phantom pseudo-rank（2026-05-20 确诊；broad re-derivation 2026-05-22 **5a–5h + 5g 全部跑完，14/14 main step 完成**；Checkpoint A/B advisor consult 通过；**未发现任何 primary cohort verdict 翻转**；P3 framework-flip gate 在 like-for-like orig 6-cohort 上 verdict INCONCLUSIVE 完全保持。**工程层未封板**：`use_masked_features` / `mask_phantom` 默认仍是 False（5i.6 default flip 进行中），`src/cluster_geometry.py` PCA 嵌入路径还未走 masked features（5i.6 一起修），`scripts/run_interictal_propagation.py` PR-4* bootstrap 7 个 callsites 漏传 `use_masked_features`（5i.6 一起修）—— 直到 5i.6 落，新 runner 忘传 flag 会静默回到 phantom-contaminated 路径。详见 `docs/archive/topic0/lagpat_phantom_rank/rerun_results_2026-05-21.md` + Topic 0 §3.1 表 + §5 表）
  2. **（已落地）** SEEG 3D coord loader v3.1：`src/seeg_coord_loader.py` + 49 unit tests + 27-subject real-data smoke 全 GREEN。Yuquan 输出 `fs_native_ras_mm`，**Epilepsiae 自动发现 MRI + 应用 MNI152 affine 输出 `mni152_1mm`**（cohort-comparable）；当前 stable_k=2 cohort 约 22/34 已可直接进 Topic 4 H1/H2 主分析
- **Phase 0 解锁（2026-05-22）**：Topic 4 Phase 1 真实数据验收可以启动，剩下 = `load_subject_for_phase1()` integration PR（接 coord loader → Phase 1 runner）
- **影响范围**：Topic 1 / Topic 4 / PR-5 / PR-6 / PR-7 主结论方向全部保持；4 条 secondary metric flip（PR-5 share×2 + transition + PR-6 node anatomy h1_eligible Wilcoxon）；1 条 **exploratory/secondary loss**（PR-4B L3 高置信 n=8 Pearson r delta p=0.016→0.547，小样本脆弱性，原版 archive 已标 exploratory tier，不进 main evidence base — **不是 primary cohort verdict reversal**）；详见 Topic 0 §3.1
- **2026-05-21–22 重跑期间科学发现**：Step 5c 在 masked 上重跑发现"簇内 86% identity bias" → "92% identity bias"，PR-4 panel d **加强**而非削弱；Topic 4 attractor λ₂ orig 10/34 → masked **13/34** 实质加强；PR-5-B 核心 +65 events/h 信号 magnitude + direction 100% 保持；PR-6 Step 6 swap_class concordance 0.69 → 0.82 实质提升；PR-7 short-window mark-dependent 偏离信号 directionally stronger after fix (orig 10s median −0.018 → mask −0.045) 但 cohort-level verdict INCONCLUSIVE 保持，作 PR-7 v2 power-analysis 跟踪条目；Topic 4 "endpoint 是结构化锚点" 几何前提**被独立证据加强**

## 1. 论文现在的 4 个 topic

### Topic 1：间期事件内部时序结构

关注单个群体事件内部的时序组织，而不是事件与事件之间的间隔。

- 正式入口：`docs/topic1_within_event_dynamics.md`
- 核心数据对象：`lagPatRank`、`eventsBool`、`chnNames`、event-level synchrony rows
- 核心问题：
  - 单个群体事件内部的传播顺序是否刻板、是否多模态、是否与 SOZ 有关
  - 单个事件内部/事件级同步性是否支持“发作前后重置”叙事

### Topic 2：间期群体事件之间的时序分析

关注群体事件作为一个点过程，在事件与事件之间表现出什么时间结构。

- 正式入口：`docs/topic2_between_event_dynamics.md`
- 核心数据对象：group-event timestamps、IEI、PSD、rate trace、`n_participating`
- 核心问题：
  - `~2 Hz` 峰是不是内禀振荡
  - IEI 是否 power-law
  - 慢时间尺度调制是否存在、发生在什么时间尺度、是否与发作邻近有关

### Topic 3：Where / SOZ 空间归因

关注慢调制和时序差异在空间上发生在哪里，尤其是 SOZ / non-SOZ 的分离。

- 正式入口：`docs/topic3_spatial_soz_modulation.md`
- 核心数据对象：per-channel relaxed-refine events、SOZ labels、i/l/e labels
- 核心问题：
  - lagPat 框架为什么回答不好 where
  - per-channel 框架下 SOZ 与 non-SOZ 是否真的不同
  - 哪部分是全局调制，哪部分是 SOZ 的局部短程记忆

### Topic 4：模型层 —— SEF-HFO 空间易激场模型

关注间期事件机制建模层，目标是给 Topic 1 现象（稳定模板、正反共享 endpoint、模板选择近似随机、慢漂解耦）提供机制解释而非拟合。

- 正式入口：`docs/topic4_sef_hfo.md`（**v0.2 plan lock draft 2026-06-01**）
- 当前主模型计划：`docs/archive/topic4/sef_hfo_topic4_v2_plan_2026-06-01.md`
- 上游 SBA framework：`docs/paper1_framework_sba.md`（SEF-ITP 取代其 BHPN-toy 部分；保留 P1/P2/P3/P5 红线）
- 核心断言：间期群体 HFO = 局部低异质性、各向异性连接、近临界但仍亚阈值的 E-I 易激斑块，在噪声触发下产生的自限性瞬态传播事件；低异质性必须通过 effective gain 实际计算进入稳定性分析
- 6 条 pre-registered 预测：H1 endpoint 空间紧凑 / H2 source-sink 反向几何 / H3 mark independence + stable geometry / H4 rate-geometry 解耦 / H5 发作邻近 endpoint identity shift / H6 participation-field 空间分隔
- 建模路线：effective gain → linear dispersion map → finite-pulse response map → 2D rate field + geometry controls → LIF E-I SNN → 抽象慢变量 feasibility bridge；旧 HR/FHN Phase 4 route 降级为历史探索 / sensitivity
- **当前执行状态（2026-06-29，建模线收口为机制 screen）**：建模路线已执行到慢变量阶段。M0→M1→M2 在均质衬底上一致显示"空间自限难、不靠压死"→ 转 M3；M3 A 线（源空间逐细胞 onset 梯度）= 沿轴相干招募波（R²≈0.87、40k SNN），B 线（谱相图）§5 非正规瞬态 = 骨架特异自限轴向，两线互证；M3A 慢变量场探索（v2 / v2.1 clamp 复查 / v2.2 sustained+`h_G`）**一致 NEGATIVE** = 当前 SNN regime + 载体图景不足以把沿轴事件改道成离轴/全局发作样招募，下一杠杆 = 连接结构 `D_EE` / 衬底重做（非继续调慢变量）。**机制 screen 通过、发作机制未 validate。** M-stage 主文档：`docs/topic4_m3_stage.md`（A/B 分文档 + worktree 处置见其 §13 / §7）
- **M4 / M4-2 / M4-3A update（2026-07-08/10，分支 `topic4-m4-divisive-sg`，2026-07-10 合入 `main` 作为探索性 M4 线）**：M3 之后新增"除法共享抑制池"这一杠杆。**M4 pass-1** = 活动依赖的除法池能把 q_I 耗竭 runaway 打开成**窄窗口有界持续态**（bounded 第三态机制筛过），但空间宽、marginal、**不可撤回/不自终止**（池只 bound 不 terminate；archive `docs/archive/topic4/sef_hfo/m4_pass1_divisive_shared_pool_acceptance_2026-07-09.md`）。**M4-2** = 在该有界工作点加 E→E 短时程疲劳（STD）当终止器，扫削竭 × 恢复网格 + 3 seed：**全 map 0 terminate_clean**——STD 只把系统碎裂或压死，做不出"一次干净发作 → 可再触发间期"（scoped clean no-go，此衬底/工作点/网格/3-seed 内，非普适；archive `docs/archive/topic4/m4_2_std_termination_p1_sweep_2026-07-08.md`）。**结论 = 下一杠杆仍指向 `D_EE`/衬底异质 或更慢离子型（slow-K / gK）终止器，与 M3A 收口一致。** **M4-3A**（2026-07-10）= 换连续、活动驱动、**电导型 shunting** 恢复变量 `n→a` 当终止器（正是 M4-2 STD no-go 指向的"更慢离子型/shunting"方向），扫 shunt 强度 `α_A` × 负荷恢复时标 `τ_n` + 3 seed（+ 边界细化 α∈{5,6,7}）：**30 格 0 terminate_clean / 0 go**——**不加 shunt 是有界持续，弱 shunt（α≤4）反把有界态推成 runaway、强 shunt（α≥5）碎裂成 fragment，中间无干净终止窗**（scoped clean no-go，此衬底/工作点/网格/3-seed 内；`u_n0=0` 纠正 plan P0b 自败标定；archive `docs/archive/topic4/m4_3a_continuous_shunting_p1_discovery_2026-07-10.md`；spec/plan `docs/superpowers/{specs,plans}/2026-07-09-*m4-3*`）。**新机制线索（未证，需 dump `S_G` 才能查）**：弱-shunt-runaway 疑似 shunt 削稳态放电→除法共享刹车 `S_G` 积累不足→bound 松开。**下一杠杆（spec §5：no-go ≠ 已证 `D_EE`，`λ_K=0` 仍锁）= M4-3B graph-kernel smoke / deferred `g_K` arm / `D_EE` 衬底 三选一，外加 shunt-vs-`S_G` 相互作用这条新线索。** M4-3A build = 10-task subagent-driven 全过审（拦下 2 个真 plan-bug：多态 `slow=` + 弱自测），byte-parity 双证。

### Topic 5：Seizure-related analysis（subject 内 seizure subtype + 下游 pre-ictal/outcome）

关注以 ictal seizure 本身为研究对象的 within-subject heterogeneity carve-out 与下游关联。

- 正式入口：`docs/topic5_seizure_subtyping.md`
- 核心数据对象：v2.3 ictal ER timing atlas (PR-0)、z-ER (channel × time-bin) 张量、subtype_label
- 核心问题：
  - 每 subject 内多个 ictal seizure 是否需要按 within-subject pathway 切 subtype（PR-1）
  - subtype 在 pre-ictal / propagation / outcome 是否表现出系统差异（PR-2+，未启动）
  - subtype 与 SOZ propagation pattern 是否互相印证（待立 PR）
- **当前状态（network-skeleton 重定位 + 发作内 field pilot 2026-06-28）**：A 线已从"发作特异路径回放"重定位为"患者内稳定网络骨架读出"（间期/发作共用粗粒度锚点）；replay / subtype / load / drift 系列假设均阴性。最近做的发作内 field 动力学 pilot（§3.6）= **exploratory**：发作场随时间在变但整体仍贴间期主轴，"轴向走廊变弱 / 离轴变强"在 broad 队列有暗示、narrow 扩队列证否一半 → 非稳健、依队列/走廊几何，只进 supplementary。archive：`docs/archive/topic5/ictal_field_dynamics_pilot_2026-06-28.md`

---

## 2. 一句话总论

### Topic 1

间期群体事件内部存在稳定但多模态的传播结构；`k=2` 是主导压缩但不是普适真相，少数 subject 需要 `k=4` 到 `k=6` 才能更好描述。PR-2.5 显示这些模板在 split-half / blockwise 尺度上总体稳定（`23/30 strong`, `7/30 moderate`），forward/reverse 候选关系在 `11/12` subject 中复现。PR-5 进一步支持 post-ictal dominant-template 绝对招募率升高；但 rate 调制与 seizure-onset cluster 的共现具有明显异质性（strict 子群而非全体规律），需要新的 burst-level 指标继续刻画。PR-6A 截至 2026-04-23 仅完成 Step0-2 与 Step3-preview 审阅：ER 值得继续作为 clinical 前 electrographic recruitment 的候选特征，但 onset-rank 提取层尚未封板。cluster-aware 分析显示刻板性真实存在，但 SOZ 优势目前仍偏探索性。事件级同步性在线队列水平总体为 null，仅 extra-focal phase synchrony 出现探索性 `pre > post`。

### Topic 2

`~2 Hz` 群体事件峰不是内禀振荡器证据；现有证据支持“带不应期的兴奋性点过程 + 多时间尺度慢调制”。`21/21` 有 specparam 峰的 subject 已被 refractory renewal + slow modulation 解释。

### Topic 3

lagPat 群体事件框架的 SOZ / non-SOZ 对比受结构性选择偏差污染。per-channel relaxed-refine 分析显示：原始 serial correlation 没有 SOZ 差异，但去趋势后 SOZ 更像保留了额外的局部短程记忆。

### Topic 4

SEF-HFO v0.2（2026-06-01 plan lock draft）把 Topic 4 主模型收紧为一个可证伪机制闭环：低异质性不能直接等同近临界，必须先进入群体输入-输出曲线并改变局部 gain；线性稳定性只给小扰动地图，有限幅脉冲图才证明“能点燃但不失控”；稳定模板和高通道身份偏置必须高于电极几何 / 采样方式 controls。文献 framing 采用“具体细胞机制多样、中观动力学收敛”的安全口径：离子、泵、胶质、抑制和连接结构都可能改变易激性与恢复能力，但 SEF-HFO 只抽象检验 HFO 群体事件的自限传播、有限扰动响应、事件率调制和空间招募。旧 HR/FHN Phase 4 route 降级为历史探索 / sensitivity。真实数据验收合同仍沿用 v1 的 6 条预测（H1–H6），但 v0.2 建模主指标改为 held-out rank stability、split-half / odd-even stability、inter-template anti-correlation、self-limited pulse response 和 controls fail；`k=2` 与 raw identity bias 只作描述性输出。下一步主线是先做 effective gain + linear dispersion + finite-pulse response，再跑 2D rate field 和真实模板 pipeline。

**当前进展（2026-06-29，建模执行收口）**：我们把这个"易激斑块"模型从纸面推进到了仿真。先问最基本的——一片性质均匀的神经组织，能不能自己长出"一次放电点着、沿一条固定通路传一段、然后自己停下来"的事件？能，但它有个顽固脾气：要么整片一起亮、要么很快被压死，很难做到"只铺开一小片就停"。我们顺着这个脾气一路加机制（局部恢复、向前的抑制、会缓慢变化的局部易激/疲劳地形），想让放电学会"拐到旁边去"（对应真实发作里活动从原通路扩散到别处）。到目前为止没成功——不是这些机制想法本身错，而是当前这版仿真衬底的图景撑不起它：放电只会沿原轴越铺越大或直接失控，没有出现"受控地改道到旁边"。所以结论定为"机制层面的探针通过了（这种组织确实能点着、能沿轴传、能自限），但'像真实发作那样改道/全局招募'这一步还没在模型里验证出来"，下一步该动的是神经元之间的连接结构本身，而不是继续拧这两个慢变量旋钮。（内部归档代号：M0–M3 stage screen、M3A-v2/v2.1/v2.2 carrier exploration NEGATIVE、`D_EE` 下一杠杆；主文档 `docs/topic4_m3_stage.md`）

**H2b 方向轴诊断（archive-only supplementary，2026-05-25 v1.0.2）**：H2 spatial-layer cohort claim 之上 archive-only 的几何方向诊断 phase，问"swap 是否仅来自正交无关 source"。strict verdict 在 23 个可测 subject 上 0 个 `axis_reversal`（permutation null 在 `decision_k ≈ n_universe / 2` 时自由度退化）；descriptive shape 在 swap_class∈{strict, candidate} 合并 9 个可测 subject 上 5 个 `axis_reversal_shaped` + 0 个 `dual_source_shaped` + 4 个 `unclear`（含 3 个 PCA 单 shaft 退化）。**Scope 红线**：H2b 仅 falsify "正交无关 source" 假说，**不能** 区分 "同一病理核心轴双向读取" 与 "同轴双端独立 seed"（两者都预测 cos(v_A, −v_B) ≈ +1）。区分必须做 Round 2（per-event seed 聚类 + rank-distance 连续梯度 + source 各自相对 SOZ 关系，未实施）。**不进** framework v1.0.7 §3 H 主清单。详见 `docs/archive/topic4/sef_itp_direction_axis/{phase_h2b_direction_axis_plan_2026-05-25.md (v1.0.2), cohort_run_2026-05-25.md}`。

### Topic 5

PR-0 v2.3 ictal ER timing atlas + PR-1 z-ER subtyping 在 16 个 epilepsiae subject 上落地（exploratory，2026-05-10 audit-corrected）：约 64% subject-band 找到 ≥2 morphological subtypes，与 Schroeder 2020 *PNAS* within-patient pathway-variability 先验一致。442/548 sentinel 视觉支持 z-ER 抓得到 user 标的视觉异类（recall=100%）；548 gamma k=7 标为 high-heterogeneity / fine subdivision candidate（需 sensitivity）；916/1077 因 status filter / n_ok 失效不能作 sentinel。下游 PR 必须 per-subtype 不 per-subject。

**当前进展（network-skeleton 重定位 + 发作内 field pilot）**：这条线后来重定位了——我们原本以为间期网络轴是"发作时那条传播路径的预演/回放"，但证据指向它其实是患者自己一张**粗粒度的稳定网络锚点图**，间期和发作大体共用同一张图，而不是发作特异的路径回放（回放/子型/负荷/漂移一系列假设都没看到信号）。最近做的发作内 field pilot 想看：一次发作从头到尾，电活动的"空间形状"会不会从"贴着间期那条主轴"逐渐拐向"离轴/横向"。结果是：场确实随时间在变、但整体仍贴着间期主轴；"轴向变弱、离轴变强"这个方向性假设在小队列里有点苗头、扩到更多被试后被证否了一半——它依赖具体队列和走廊几何，不是稳健现象，只作 exploratory/supplementary，不进主结论。（内部归档代号：A-line network-skeleton pivot、ictal field dynamics pilot broad 暗示 / narrow 证否；archive `docs/archive/topic5/ictal_field_dynamics_pilot_2026-06-28.md`）

---

## 3. 先读哪份文档

### 如果你只想知道当前正式结论

1. `docs/paper_overview.md`
2. **`docs/topic0_methodology_audits.md`** ← 必读，决定下面结论的可信度
3. `docs/topic1_within_event_dynamics.md`
4. `docs/topic2_between_event_dynamics.md`
5. `docs/topic3_spatial_soz_modulation.md`
6. `docs/topic4_sef_hfo.md`
7. `docs/topic5_seizure_subtyping.md`

### 如果你要看历史证据链或审阅意见

- **Topic 0** 历史来源（`docs/archive/topic0/`）：
  - `docs/archive/topic0/INDEX.md`
  - `docs/archive/topic0/lagpat_phantom_rank/diagnostic_2026-05-20.md`
  - `docs/archive/topic0/lagpat_phantom_rank/plain_chinese_report_2026-05-20.md`
  - `docs/archive/topic0/lagpat_phantom_rank/rerun_roadmap_2026-05-20.md`
- Topic 1 历史来源（`docs/archive/topic1/`）：
  - `docs/archive/topic1/propagation/interictal_group_event_internal_propagation.md`
  - `docs/archive/topic1/synchrony/interictal_synchrony_preliminary_report_2026-04-03.md`
  - `docs/archive/topic1/pr6_template_anchoring/pr6a_step0-2_step3preview_review_2026-04-23.md`
- Topic 2 历史来源（`docs/archive/topic2/`）：
  - `docs/archive/topic2/event_periodicity_analysis.md`
  - `docs/archive/topic2/interictal_population_event_methodological_review.md`
  - `docs/archive/topic2/event_periodicity_phase2_review_2026-04-05.md`
- Topic 3 历史来源（`docs/archive/topic3/`）：
  - `docs/archive/topic3/pr1_spatial_modulation/spatial_modulation_soz_analysis.md`
- Topic 4 历史来源（`docs/archive/topic4/`）：
  - `docs/archive/topic4/sef_itp_direction_axis/phase_h2b_direction_axis_plan_2026-05-25.md` (v1.0.2 — H2b 方向轴诊断 contract + scope 红线 + Round 2 deferred list)
  - `docs/archive/topic4/sef_itp_direction_axis/cohort_run_2026-05-25.md` (H2b cohort 数字快照 + audit trail)
- Topic 5 历史来源（`docs/archive/topic5/`）：
  - `docs/archive/topic5/INDEX.md`
  - `docs/archive/topic5/pr1_seizure_clustering/pr1_zer_cohort_2026-05-10.md`

这些历史文档保留事实、审阅和阶段性推理，但不再是首选入口。

---

## 4. 结果与代码入口

### Topic 1

- 结果：`results/interictal_propagation/`
- 代码：`src/interictal_propagation.py`
- 脚本：`scripts/run_interictal_propagation.py`、`scripts/plot_interictal_propagation.py`

### Topic 2

- 结果：`results/event_periodicity/`、`results/event_periodicity/phase2/`
- 代码：`src/event_periodicity.py`
- 脚本：`scripts/run_event_periodicity.py`、`scripts/run_periodicity_phase2.py`、`scripts/plot_periodicity_phase2.py`

### Topic 3

- 结果：`results/spatial_modulation/`、`results/refine_soz_validation/`
- 代码：`src/event_periodicity.py` 中 per-channel / SOZ helpers，`src/group_event_analysis.py`
- 脚本：`scripts/audit_gpu_npz.py`、`scripts/run_spatial_modulation.py`、`scripts/plot_spatial_modulation.py`

### Topic 4

- 结果：`results/topic4_sef_itp/{phase1_spatial_geometry, phase2_temporal_x_geometry, phase3_ictal_adjacent, direction_axis}/`（真实数据验收）+ `results/topic4_sef_hfo/{linear_stability,rate_field,anisotropy,low_heterogeneity_patch,lif_snn,slow_variable_bridge,synthetic_vs_real}/`（v2 modeling）
- 代码：`src/sef_itp_phase1.py`、`src/sef_itp_phase2.py`、`src/sef_itp_phase3.py`、`src/sef_itp_phase3_trajectory.py`、`src/sef_itp_direction_axis.py`（H2b archive-only supplementary）
- 脚本：`scripts/{run,summarize,plot}_sef_itp_phase{1,2,3}*.py`、`scripts/{run,plot}_sef_itp_direction_axis.py`；v2 modeling scripts 待 Step 0 implementation plan 后新增

### Topic 5

- 结果：`results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/`、`results/data_driven_soz/layer_a_ictal_er_rank/atlas_v2_3/`
- 代码：`src/ictal_zer_features.py`、`src/ictal_seizure_clustering.py`、`src/ictal_seizure_plotting.py`、`src/atlas_loading.py`
- 脚本：`scripts/cluster_ictal_seizures.py`、`scripts/diagnostic_cluster_grid.py`、`scripts/plot_ictal_er_atlas.py`

---

## 5. 当前最稳的科学结论

- Topic 1：内部传播不是单一模板，而是多模态且多数以双模态为主的病理网络传播路径；legacy MI 可复现，cluster-aware τ 明显高于整体 τ，而且模板在 split-half / blockwise 尺度上总体稳定。
- Topic 1：interictal synchrony 在 cohort level 没有支持“post-ictal reset / pre-ictal resynchronization”；唯一值得继续追的是 extra-focal `phase_e` 的 `pre > post`。
- Topic 2：`~2 Hz` peak 不是 oscillator；IEI 是 lognormal，不是 power-law。
- Topic 2：IEI 相邻正相关是硬结果，支持慢率漂移；去趋势后仍保留短程依赖。
- Topic 2：rate trace 存在 seizure-centered broad elevation，但现在还不能诚实地叫作 pre-ictal biomarker。
- Topic 3：SOZ / non-SOZ 的 raw serial correlation 差异在 per-channel 框架下消失，说明旧 lagPat 结果部分混入了事件率与通道选择偏差。
- Topic 3：SOZ 更像是“全局调制之上叠加局部短程记忆”，而不是简单地“整体更同步”或“整体更周期”。
- Topic 5：v2.3 ictal ER atlas 显示 within-subject seizure 异质性是真现象；z-ER subtyping cohort 上 ~64% subject-band 找到 ≥2 morphological subtypes，与 Schroeder 2020 PNAS 先验一致。结论 commit 到 publication-grade 仍需 sensitivity（sentinel 442/548 已视觉过关；548 gamma k=7 / 916/1077 sentinel 失效是已知限制）。下游 PR 必须 per-subtype。
- Topic 5（临床结局收口 Track E，2026-06-13）：Yuquan 触点级"模板网络覆盖"预测变量侧已跑通、结局标签 gating 待医院随访（E1）；Epilepsiae 切除仅叶级、触点级覆盖构造不出 = no-go feasibility（E2）。详见 `docs/topic5_seizure_subtyping.md` §3.4。

---

## 6. 规则入口

- Topic 1 rule：`.cursor/rules/topic1-within-event-dynamics.mdc`
- Topic 2 rule：`.cursor/rules/topic2-between-event-dynamics.mdc`
- Topic 3 rule：`.cursor/rules/topic3-spatial-soz-modulation.mdc`

旧 rule：

- `.cursor/rules/interictal-propagation-pr-plan.mdc`
- `.cursor/rules/event-periodicity-pr-plan.mdc`

目前保留为过渡入口，防止旧引用失效。
