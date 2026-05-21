# 论文总论与文档索引

> 状态：当前正式入口
> 目的：给人和 Agent 一个稳定的总索引，先回答“这篇论文现在到底讲哪 3 个 topic、各自结论是什么、应该先读哪里”。

---

## 0. Topic 0：方法学审计与数据合同（**优先级最高**）

> 在读任何 Topic 1–5 的科学结论之前，必须先看本节是否有未结清问题。

- 正式入口：`docs/topic0_methodology_audits.md`
- 当前未结清问题：**1 个 + 1 个已基本结清**
  1. **（已基本结清）** `lagPatRank` phantom pseudo-rank（2026-05-20 确诊，broad re-derivation 2026-05-21 **5a–5h 全部完成**；5g PR-7 per-subject 28/30 剩 2 大 subject 还在跑；Checkpoint A/B advisor consult 通过；**未发现任何 primary cohort verdict 翻转**；详见 `docs/archive/topic0/lagpat_phantom_rank/rerun_results_2026-05-21.md` + Topic 0 §3.1 表）
  2. **（已落地）** SEEG 3D coord loader v3.1：`src/seeg_coord_loader.py` + 49 unit tests + 27-subject real-data smoke 全 GREEN。Yuquan 输出 `fs_native_ras_mm`，**Epilepsiae 自动发现 MRI + 应用 MNI152 affine 输出 `mni152_1mm`**（cohort-comparable）；当前 stable_k=2 cohort 约 22/34 已可直接进 SEF-ITP H1/H2 主分析
- **Phase 0 解锁（2026-05-21）**：SEF-ITP Phase 1 可以启动，剩下 = `load_subject_for_phase1()` integration PR（接 coord loader → Phase 1 runner）
- **影响范围**：Topic 1 / Topic 4 / PR-5 / PR-6 / PR-7 主结论方向全部保持；少量 secondary metric 弱化（详见 Topic 0 §3.1）
- **2026-05-21 重跑期间科学发现**：Step 5c 在 masked 上重跑发现"簇内 86% identity bias" → "92% identity bias"，PR-4 panel d **加强**而非削弱；Topic 4 attractor λ₂ orig 10/34 → masked **13/34** 实质加强；PR-5-B 核心 +65 events/h 信号 magnitude + direction 100% 保持；PR-6 Step 6 swap_class concordance 0.69 → 0.82 实质提升；SEF-ITP "endpoint 是结构化锚点" 几何前提**被独立证据加强**

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

### Topic 4：模型层 —— SEF-ITP 空间易激场模型

关注间期事件机制建模层，目标是给 Topic 1 现象（稳定模板、正反共享 endpoint、模板选择近似随机、慢漂解耦）提供机制解释而非拟合。

- 正式入口：`docs/topic4_sef_itp_framework.md`（**v1 framework lock 2026-05-20**）
- 上游 SBA framework：`docs/paper1_framework_sba.md`（SEF-ITP 取代其 BHPN-toy 部分；保留 P1/P2/P3/P5 红线）
- 核心断言：间期群体 HFO = 空间组织化病理易激区 θ(x) 被扩散物理反复采样的痕迹；θ(x) 是模型输入而非产出
- 6 条 pre-registered 预测：H1 endpoint 空间紧凑 / H2 source-sink 反向几何 / H3 mark independence + stable geometry / H4 rate-geometry 解耦 / H5 发作邻近 endpoint identity shift / H6 participation-field 空间分隔
- 硬前置：Topic 0 §3.1 phantom-rank 修复 + §5 broad re-derivation 必须完成才能启动 Phase 1+

### Topic 5：Seizure-related analysis（subject 内 seizure subtype + 下游 pre-ictal/outcome）

关注以 ictal seizure 本身为研究对象的 within-subject heterogeneity carve-out 与下游关联。

- 正式入口：`docs/topic5_seizure_subtyping.md`
- 核心数据对象：v2.3 ictal ER timing atlas (PR-0)、z-ER (channel × time-bin) 张量、subtype_label
- 核心问题：
  - 每 subject 内多个 ictal seizure 是否需要按 within-subject pathway 切 subtype（PR-1）
  - subtype 在 pre-ictal / propagation / outcome 是否表现出系统差异（PR-2+，未启动）
  - subtype 与 SOZ propagation pattern 是否互相印证（待立 PR）

---

## 2. 一句话总论

### Topic 1

间期群体事件内部存在稳定但多模态的传播结构；`k=2` 是主导压缩但不是普适真相，少数 subject 需要 `k=4` 到 `k=6` 才能更好描述。PR-2.5 显示这些模板在 split-half / blockwise 尺度上总体稳定（`23/30 strong`, `7/30 moderate`），forward/reverse 候选关系在 `11/12` subject 中复现。PR-5 进一步支持 post-ictal dominant-template 绝对招募率升高；但 rate 调制与 seizure-onset cluster 的共现具有明显异质性（strict 子群而非全体规律），需要新的 burst-level 指标继续刻画。PR-6A 截至 2026-04-23 仅完成 Step0-2 与 Step3-preview 审阅：ER 值得继续作为 clinical 前 electrographic recruitment 的候选特征，但 onset-rank 提取层尚未封板。cluster-aware 分析显示刻板性真实存在，但 SOZ 优势目前仍偏探索性。事件级同步性在线队列水平总体为 null，仅 extra-focal phase synchrony 出现探索性 `pre > post`。

### Topic 2

`~2 Hz` 群体事件峰不是内禀振荡器证据；现有证据支持“带不应期的兴奋性点过程 + 多时间尺度慢调制”。`21/21` 有 specparam 峰的 subject 已被 refractory renewal + slow modulation 解释。

### Topic 3

lagPat 群体事件框架的 SOZ / non-SOZ 对比受结构性选择偏差污染。per-channel relaxed-refine 分析显示：原始 serial correlation 没有 SOZ 差异，但去趋势后 SOZ 更像保留了额外的局部短程记忆。

### Topic 4

SEF-ITP framework v1（2026-05-20 lock）取代了 BHPN-toy 循环论证模型，把 Topic 4 模型层从"塞 Hebbian 矩阵让 Kuramoto 演化得到预设吸引子"换成"假设空间组织化病理易激区，看它被扩散物理反复采样后必然留下的几何指纹"。6 条 pre-registered 预测（H1–H6）覆盖空间紧凑性、source/sink 反向几何、mark independence + 几何稳定、rate-geometry 解耦、发作邻近 endpoint identity shift、participation-field 空间分隔。**Phase 0 已解锁（2026-05-21）**：phantom-rank 修复 5a–5h 完成 + coord loader v3.1 落地；Phase 1 可启动，仅剩 `load_subject_for_phase1()` integration PR。Phase 0 实质性**加强**了 SEF-ITP 几条关键证据（PR-3 identity bias 86%→92%、Topic 4 λ₂ 10/34→13/34、PR-5-B +65 ev/h direction 100% 保持、PR-6 Step 6 concordance 0.69→0.82）。

### Topic 5

PR-0 v2.3 ictal ER timing atlas + PR-1 z-ER subtyping 在 16 个 epilepsiae subject 上落地（exploratory，2026-05-10 audit-corrected）：约 64% subject-band 找到 ≥2 morphological subtypes，与 Schroeder 2020 *PNAS* within-patient pathway-variability 先验一致。442/548 sentinel 视觉支持 z-ER 抓得到 user 标的视觉异类（recall=100%）；548 gamma k=7 标为 high-heterogeneity / fine subdivision candidate（需 sensitivity）；916/1077 因 status filter / n_ok 失效不能作 sentinel。下游 PR 必须 per-subtype 不 per-subject。

---

## 3. 先读哪份文档

### 如果你只想知道当前正式结论

1. `docs/paper_overview.md`
2. **`docs/topic0_methodology_audits.md`** ← 必读，决定下面结论的可信度
3. `docs/topic1_within_event_dynamics.md`
4. `docs/topic2_between_event_dynamics.md`
5. `docs/topic3_spatial_soz_modulation.md`
6. `docs/topic4_sef_itp_framework.md`
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

---

## 6. 规则入口

- Topic 1 rule：`.cursor/rules/topic1-within-event-dynamics.mdc`
- Topic 2 rule：`.cursor/rules/topic2-between-event-dynamics.mdc`
- Topic 3 rule：`.cursor/rules/topic3-spatial-soz-modulation.mdc`

旧 rule：

- `.cursor/rules/interictal-propagation-pr-plan.mdc`
- `.cursor/rules/event-periodicity-pr-plan.mdc`

目前保留为过渡入口，防止旧引用失效。
