# Topic 5：Seizure-related analysis (per-subject subtype + 下游 pre-ictal / outcome)

> **🆕 V2 新重点（2026-07-01，design rev2 收紧完，待 writing-plans）：间期 HFO 几何 = 病理【候选】临界模态。**
> 三层框架 —— trait：间期 HFO timing 几何 = **candidate** 病理临界模态（升级需证据阶梯）；state：发作前临界性沿这条模态升高；
> expression：发作起始通过 **phenotype 特异的频带**沿这条模态招募（频带=发作机制的 readout，非技术参数）。
> 核心问题从"哪个频带最像间期几何"升级为"**哪种发作起始机制沿间期 HFO 通路被重新招募**"。
> **三个 gate（取代单一 go/no-go）**：A 空间平滑 null / B 频带特异（超 broadband）/ C HFO 特异（1-f+common-field 残差存活）；每层不过各有可发表结论（如 Gate B 不过 = "预测宽带招募非窄带 HFO"）。
> **防循环两条命门**：必须过 **HFO-rate-preserving timing-order null**（否则 timing geometry 退化成 HFO-rate topography）；`hfo_rich` phenotype 须**独立来源**（不能用同一 ripple-power outcome 定义再验 ripple 对齐）。
> 设计见 `docs/superpowers/specs/2026-07-01-topic5-v2-hfo-critical-mode-design.md`（rev2）；数值+文献 handoff 见 `docs/archive/topic5/ictal_band_sweep_handoff_2026-07-01.md`。
> V1（network-axis，下方 2026-06-14 主线）是 V2 前身/基础，未被推翻，且 V2 不默认把 stable scaffold 升级成 unstable mode。

> **主线结果（2026-06-14）：network-axis A+B 已执行**——间期传播轴 ↔ 发作早期激活共享一根粗网络骨架（18 Epilepsiae 队列稳），细对齐仅快活动量稳；符号自由共线，非逐点重放。见 §3.0 + 归档 `axis_alignment_AB_result_2026-06-14.md`。
> 状态：**探索性 (exploratory) — PR-0 + PR-1 落地，audit-rerun 完成 (2026-05-10)，yuquan 扩展 (2026-05-10 PR-0.1)**。PR-1 cohort z-ER 聚类有 1 张 cohort 主结果（`figures/per_subject/`，25 subjects），cohort-level "真子型断言" 仍依赖 sensitivity；PR-2+ 未启动。
> 范围：以 ictal seizure 本身为研究对象——subject 内的 seizure subtype carve-out + 下游 pre-ictal / outcome / propagation 关联。
> **不属于**：interictal 事件内部传播（topic1）、IEI/PSD（topic2）、spatial SOZ 归因（topic3）、模型层（topic4）。

---

## 1. 这个 topic 只回答什么问题

本 topic 回答以下三类问题：

1. **每 subject 内部，多个 ictal seizure 是否需要按 within-subject pathway 切分成 subtype？** (PR-1)
2. （未启动）切分出的 subtype 在 pre-ictal / propagation / outcome 层面是否表现出系统性差异？(PR-2+)
3. （未启动）subtype 与 SOZ propagation pattern 是否互相印证？

它**不**回答：

- interictal 群体事件内部传播的刻板性：那是 `docs/topic1_within_event_dynamics.md`
- 群体事件的 IEI / PSD：那是 `docs/topic2_between_event_dynamics.md`
- per-channel SOZ vs non-SOZ 慢调制：那是 `docs/topic3_spatial_soz_modulation.md`

---

## 2. 一句话当前结论

- **🟢 主线（network-axis pivot，A+B，2026-06-14 执行完毕）**：把每个病人**平时**那条间期高频传播轴
  （各触点平均发放先后）和**发作头十秒**的激活高地摆到同一张电极平面上，量两者空间梯度是否共线
  （符号自由）。**18 个 Epilepsiae 被试队列：粗层面的"共享网络主轴"稳——四种激活量都稳赢"全触点随机洗牌"，
  FDR + 留一都扛得住。**比电极杆/活跃度更细的对齐只在**快活动 60–100 Hz**上稳（过最严的同杆×活跃度联合洗牌，
  FDR q=0.029），主指标宽带功率止于粗层。**这是符号自由的轴/梯度共线，不是"发作沿间期路线逐点重放"。**
  primary 只有宽带一条，B 线（EI-like）/ hfa / ramp 是次级 / 灵敏度读出。
  详见 §3.0 + 归档 `docs/archive/topic5/axis_alignment_AB_result_2026-06-14.md`（含 handoff）。
- **PR-0 (v2.3 Layer A ictal ER timing atlas)**：cohort = 25 (15 epilepsiae audit_eligible + 9 yuquan audit_eligible + sentinel-only epilepsiae/916; topic5 PR-0.1 2026-05-10 yuquan extension)。每 subject v2.3 schema，per-seizure PNG 全 cohort 渲完。User 视觉巡视暴露 within-subject seizure pattern 异质性（442 sz=9 / 548 {13,14,24,25} / 916 {21,23,25} / 1077 sz=1），是 PR-1 的直接动机。详见 `docs/superpowers/specs/topic5_pr0_*` (待整理) + `results/data_driven_soz/layer_a_ictal_er_rank/atlas_v2_3/figures/`。
- **PR-1 z-ER subtype 聚类（2026-05-10 audit-corrected exploratory 版；2026-05-10 yuquan-extended）**：25 subjects (16 epilepsiae + 9 yuquan)，50 subject-band rows，33 ok / 17 insufficient_n。yuquan ok 子集 (n=5 cells, 4 subjects: litengsheng broad k=2, sunyuanxin broad k=1, zhangkexuan gamma k=2, zhaojinrui gamma k=2 + broad k=1) silhouette median 0.495 / gap_perm median 0.552 — 实际优于 epilepsiae ok 子集 (silhouette 0.418 / gap_perm 0.325)。整体 cohort silhouette median 0.444、gap_perm median 0.380。`over_split_flag` (AND 规则 `gap_perm < 0.10 AND ratio > 0.5`) cohort 命中 **0/33 ok**。Bug-fix 实测影响：pre-audit 28 个 ok rows 上 Δgap_perm 中位 −0.0007、abs_max 0.061，**0 个 over_split_flag flip**，0 个 sentinel jaccard 变化。
- **PR-1 sentinel 视觉裁定**（user 2026-05-09 / 2026-05-10）：
  - 442 (user=[9])：**最干净 sentinel** ✅，gamma+broad 都把 sz=9 单列
  - 548 broad k=3：**基本合理**，user-marked [13,14,24] 落同一 minority 家族
  - 548 gamma k=7：**high-heterogeneity / fine subdivision candidate** ⚠️ ——不是过切但也未确认真 7 类，需要 sensitivity (min_subtype_size=3 / 不同 bin / bootstrap stability)
  - 1146 broad k=3：**教科书级 3 子型分离** ✅
  - 916 (user=[21,23,25])：**不能作为 sentinel** ✗（user 标的 3 个全被 v2.3 status filter 过滤）
  - 1077 (user=[1])：**不能作为 sentinel** ✗（n_ok=3 < 5）
  - 有效 sentinel = 442 + 548（4 个 subject-band 全 recall=100% on user-marked outliers）
- **2026-05-10 audit fix（重要）**：发现并修复 3 个 bug
  - `channelwise_permutation_null` 在 z-ER 路径破坏 5-bin 协方差 → 加 `bins_per_channel` 参数走 channel-block coherent shuffle
  - `extract_zer_binned_for_subject` 缺少 channel-order consistency check → 加严格 equality check + `channel_order_mismatch` drop_reason
  - `over_split_flag` 旧 OR 规则 (`sil<0.2 OR ratio>0.4`) 在高维 Spearman 下产假阳性 → 改 AND 规则 + 用 gap_perm 替代 silhouette
  - cohort z-ER audit-rerun 已完成（2026-05-10 16:21）；cohort-level 结论与 sentinel 视觉裁定全部保持有效，gap_perm 数值微调（中位 −0.0007）
- **下游 PR contract**：subtype_label 是先验分组依据，PR-2+ 必须 per-subtype 不 per-subject。

---

## 3. 核心证据链

### 3.0 Network-axis 主线：间期传播轴 ↔ 发作早期激活（A 线 primary + B 线 secondary，2026-06-14）

**这是 network-axis pivot 阶段唯一有队列结论的部分，作为 topic5 现阶段主线结果。**

**测什么**：每被试取间期模板 A 的逐触点排名场（传播轴）与所有合格发作头 10 s 的逐触点激活均值，
算两者的镜像不变（符号自由）相关 `|corr_pair_mirror_invariant|` —— 只判共线/共轴，含反向共线，
**不判方向重放**。

**怎么测**：对四层独立 null 各比一遍——`channel`（全触点随机洗牌 = 有没有任何粗共享结构）、
`within_shaft`（同杆内 = 比解剖杆细吗）、`anchor_matched`（同活跃度箱 = 排活跃度混淆）、
`joint`（同杆×同活跃度 = 最严）。每被试 null = 跨发作取中位的 B 个重排实现的 95 分位；队列做
二项 + Wilcoxon + 留一（LOSO）+ BH-FDR。主队列 18 Epilepsiae / 354 合格发作；
Yuquan 结构性仅 1 被试合格，不成队列。

**结论**：
- **粗网络骨架共享 = 稳**：四种激活量都稳赢 `channel`（broadband FDR q=0.020 / LOSO p=0.015）。
- **细对齐 = 仅快活动量稳**：hfa（60–100 Hz）四层全稳赢含 `joint`（q=0.029）；broadband 只过粗层；
  ramp / ei 过粗 + 活跃度、不过 `joint`。
- **措辞红线**：符号自由共线 ≠ 逐点重放；primary 只有 broadband，B 线（ei）/ hfa / ramp 不得当 primary
  cohort claim。

**预注册纪律**：primary 单一终点 = broadband × channel；其余 exploratory / sensitivity。
"快活动细对齐"是唯一过最严联合 null 的发现，但属灵敏度档。**2026-06-15 已做 hfa×joint 冻结复验
（split-half + 负对照）：full 干净复现（joint Wilcox=0.022）、偶数半显著、奇数半不显著（0.078）→
split_half_robust=False；负对照四层全部非显著=非假阳性。结论 = real-but-not-robust，维持灵敏度档、
不升 primary，升格须独立第二队列**（`docs/archive/topic5/hfa_joint_confirm_2026-06-15.md`）。主线粗骨架不受影响。

完整方法 + 定稿数值表 + 工件清单 + handoff：
`docs/archive/topic5/axis_alignment_AB_result_2026-06-14.md`
（定稿表 `results/topic5_ictal_recruitment/axis_alignment/axis_alignment_FINAL.md`）。

**主图（论文级 field cohort，`figures/field_concordance/`）**：① **field concordance atlas**
`field_concordance_atlas_broadband.png` —— 18 病人 paired field 同屏（左间期 order 场 / 右发作 activation 场，
最佳符号镜像、深框=过自己的粗 null、按 margin 排序），**不做场平均**，一眼看到很多病人左右场对得上；
② **null forest** `field_concordance_null_forest_broadband.png` —— 实测 |r| vs 每病人自己的 channel-shuffle null
（5/18 黑=过自己 null95%，E590/E583 远超；cohort 二项 p=0.0016）。**这两张是 field 主结果图。**

**Supplement（方法审稿向）**：四层 null 阶梯图 `axis_alignment_null_ladder_B1000.png`、每被试场图
`fields/*`、方向玫瑰图 `rose/*`（A 线方向补充 + C 线主图）、干净被试方向版 A 线 `aline_direction/*`、
加固三图 `axis_hardening_fig*`。

**加固复验（2026-06-15，五项稳健性 + 关键负对照）—— 主结论收窄为"持续 scaffold readout"。**
归档 `docs/archive/topic5/axis_alignment_hardening_result_2026-06-15.md`。
- **轴自洽（反套套）**：用一半事件搭轴、预测另一半发放顺序，held-out ρ 中位 **0.76、18/18 病人 CI>0**
  → 间期轴是高度可复现的患者特异结构，不是某批事件的偶然。
- **砍两半重搭轴再对齐**：broadband × 粗层（主）在两种砍法 × 两半上**每个都赢过随机**（binom 全<0.05），
  跨被试两半对齐强度一致（ρ 0.54 / 0.94）→ 主结论不靠某批间期事件。（与上面 hfa×joint 复验一致：
  最严层细对齐两种分半都不稳，主线粗骨架两种分半都稳。）
- **⚠️ 时间负对照（关键，新）**：把"发作激活"换成**发作前 90–120 s 的远端窗**重测——粗对齐**不弱于、甚至
  强于发作窗**（broadband 远端 eff 0.111 > 发作 0–10 的 0.091；逐病人配对 post−distal 不显著 p=0.37）。
  → **这根粗网络骨架是持续存在的患者底座，不是发作早期特异招募。**
- **adequate 分母修正**：joint 层只 13 个有效病人；修正后 hfa×joint 效应量 CI 触 0（与"real-but-not-robust"
  一致，归 sensitivity）。broadband×粗层效应量 0.087、CI[0.006,0.129] 干净。
- **措辞收窄**：✅ 可写"间期传播轴是患者内稳定结构，读出与发作激活相关的粗网络骨架"；
  ❌ 不可写"该对齐是发作早期特异出现的现象 / ictal-early-specific"；❌ 不可写"HFA 干净过最严 null"。
- 图：`figures/axis_hardening_fig{1_patient_level,2_null_hierarchy,3_window_sensitivity}.png`（fig3 承载
  "非发作特异 = scaffold"判读）。**待用户目视复核。**

### 3.1 PR-0：v2.3 Layer A ictal ER timing atlas

每 subject 一张 (gamma+broad) 主 atlas + 每 seizure 一张 per-seizure PNG。
schema：`pr_t3_1_layer_a_v2_3_timing`，detection_window=[-120, 30]s，
`channel_onsets[ch] = {frame_idx, t_onset_sec}`。
cohort：25 subjects (15 epilepsiae audit_eligible + 9 yuquan audit_eligible + sentinel-only 916; topic5 PR-0.1 2026-05-10 yuquan extension; gaolan, huanghanwen, litengsheng, pengzihang, sunyuanxin, xuxinyi, zhangjinhan, zhangkexuan, zhaojinrui)。
关键发现：user 视觉巡视暴露 within-subject seizure pattern 异质性，
直接催生 PR-1 z-ER subtyping。

完整说明：`results/data_driven_soz/layer_a_ictal_er_rank/atlas_v2_3/figures/README.md`

### 3.2 PR-1：z-ER subtype 聚类

**Method**：

- Feature: per-channel × 5-bin z-ER 张量，bins `[(-200,-50), (-50,0), (0,50), (50,150), (150,200)]` s rel. clinical onset
- Distance: `1 − Spearman` over channel-bin features (min_overlap=5)
- Linkage: UPGMA
- k 选择: silhouette + min_cluster_size 守门
- Outlier vs subtype: cluster size < 2 → outlier (`subtype_label=-1`); rest → subtype 0..k-1 by descending size
- Permutation null (gap_perm_k): channel-block coherent shuffle (5 bins of a channel move together)

代码：`src/ictal_zer_features.py` + `src/ictal_seizure_clustering.py`
驱动：`scripts/cluster_ictal_seizures.py {per-subject, cohort, render}`

**Cohort 数值（n=25, 50 subject-band rows, post yuquan-extension 2026-05-10）**：

| 指标 | 中位 | 范围 |
|---|---|---|
| n_eff | 9.0 | [5, 40] |
| silhouette_k | **0.444** | [0.128, 0.597] |
| gap_perm_k | **0.380** | [0.094, 0.737] |
| n_subtypes | 2.0 | [1, 5] |
| ari_gamma_vs_broad | — | 多数 ≥ 0.6（双 band 一致度高） |

yuquan ok cells (n=5): silhouette 0.495, gap_perm 0.552（高于 epilepsiae ok 子集 silhouette 0.418 / gap_perm 0.325）。

`over_split_flag` 在 33 ok cells 中的命中数：**0/33** (AND 规则 `gap_perm<0.10 AND ratio>0.5`)；当前 `cohort_summary__zer_binned.csv` 已可查询 `over_split_flag` 列。

**形态层面**：约 33/50 (66%) subject-bands 落 ok 状态; ok 子集中 **23/33 (70%) 找到 ≥2 morphological subtypes** (基于 n_subtypes ≥ 2 的 subject-band 数 / ok 数) → within-subject morphological
异质性是 cohort-level 真现象（biological prior 与 Schroeder 2020 *PNAS* pathway-variability
一致）；cohort-level "真子型率" 的 publication-grade 断言仍需 sensitivity (intersection-only
mask / bin 设计变化 / bootstrap stability) 才能 commit。

**Sentinel 详细**：见 §2 五行表，与 archive doc §5。

完整 method + bug fix + sentinel 表 + per-subject 数值：
`docs/archive/topic5/pr1_seizure_clustering/pr1_zer_cohort_2026-05-10.md`

### 3.3 PR-1 视觉骨架

`results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/figures/per_subject/`
共 16 张 4-panel PNG（gamma 上 / broad 下）：

- 左：dendrogram + sorted pairwise (1−Spearman) heatmap
- 右上：MDS 2D 散点 colored by subtype（outlier 用 X marker + 灰色）
- 右下：cluster-stratified t_ER_onset matrix（rows=channel, cols=seizure 按 subtype 分组）

诊断 grid（`figures/diagnostic/`）：6 张以 v2.3 atlas per-seizure thumbnail 为单元
的 cluster-grouped grid，用于目视裁定。

### 3.4 Track E：临床结局收口（状态更新，2026-06-13）

问的是有真正临床新意的一句：**把平时高频小放电那条传播路线的网络切/毁得越完整，
术后越不容易复发吗——而且这个网络指标是不是比单纯 HFO 高发率区或临床 SOZ 更贴近结局。**
分两条、口径隔离：

- **E1（Yuquan 触点级，主线）= 预测变量侧已跑通，结局侧 gating 中。** 18 个有病例文档的
  Yuquan 病人，已算出"切/毁了多少网络"的六个 exploratory 覆盖量（3 个传播模板派生 + 3 个
  临床基线 SOZ/HFO/网络）+ swap 分层标签，全部 exploratory、discordant 个案领衔。**卡点：
  结局标签（Engel/ILAE/复发/随访月数）不在 repo、病例文档只到术后 24h，必须去医院补**；
  开颅切除区是图片抠不出 → 触点级"treated"这轮 = 热凝 only、`n_resected` 记 NA（不是 0）。
  已冻结空结局表 `results/template_resection_outcome/yuquan_outcome_labels.csv` 待填。
  spec：`docs/superpowers/specs/2026-06-13-yuquan-template-resection-outcome-design.md`。
- **E2（Epilepsiae 区域级）= no-go feasibility，已落盘。** 公开队列结局齐（18/20 有手术+随访，
  按主分析门随访 ≥ 12mo 达标 10 例、门内 Engel I=7 / II–IV=3），**但切除只记到叶级/区域级
  （17/18 叶级码 + 1 占位、无任何一例触点级）**，触点级"模板网络被切多少"**根本构造不出来**
  （粒度不足，未做逐触点→脑叶映射，不主张全切）→ 不能作 E1 外部验证、只区域级弱佐证。
  归档：`docs/archive/topic5/epilepsiae_e2_region_feasibility_2026-06-13.md`。

一句话：**E 线的限速步是 Yuquan 医院随访标签**；标签到位前不下任何覆盖-结局结论。
（Track A–E 全貌见 `docs/archive/topic5/network_axis_pivot_plan_2026-06-13.md`。）

### 3.5 发作早期方向无监督两类 ↔ 间期 A/B（2026-06-27，exploratory，negative）

**测什么**：每个几何干净（ECoG）病人，把每次发作头 10 s 的激活方向**盲于间期**无监督分两堆，再看这两堆方向与间期两条传播路线（模板 A/B）的关系——不预设正反，纯描述。

**怎么测**：三道预锁门防"算法硬造两类"：① 真两堆 vs 一个主方向+散点（H0="主方向+均匀背景散点"，比 silhouette）；② 间期两模板成不成一根可比的轴（Δ_AB <60/60–120/≥120° 三档，看结果前锁）；③ 两堆贴 A/B 超不超"把两堆整体随机旋转两千次"。

**结论**：**6 病人、宽带+快活动，0 个"两类对上 A/B"。** 唯一真分两堆的 442（宽带，p_bimodal=0.042）其两堆与间期模板对不齐到超随机（two_class_unmapped，p_align=0.31）；其余只有一个主方向，或间期两模板几乎同向不成轴（1084，Δ_AB=6°）。**与 A 线、echo gate 各线一致：共性是粗网络/解剖锚，不是细到"方向两类"可分的重放。** 朴素 best-pair 残差曾给 9–18° 的假"对上"，被二模 null + 旋转 null 正确打回。

**补充（方向角层面 cohort 检验，2026-06-27）**：发作主方向到**较近**那条间期模板方向的角距，broadband 中位 **12°**、旋转 null **p=0.005 显著**——即发作主方向**确实落在间期轴上**（符号自由），把 A 线粗轴结论在方向角层面印证一遍（caveat：旋转 null 不控同平面几何，严格版是 A 线 channel-shuffle）；但这是**符号自由**，**极性/哪一端仍阴性**（逐被试 p_align 全 ≥0.05）。hfa 中位 32°、p=0.15 不显著。**一句话：轴共享（确认），极性不重放。**

完整方法 + 数值表 + 措辞红线 + 442 图 + cohort 轴对齐图：`docs/archive/topic5/ictal_direction_clustering_2026-06-27.md`。

---

### 3.6 发作内 field 动力学：轴向走廊 vs 非轴向随发作进程（2026-06-28，exploratory，broad 暗示 / narrow 扩队列证否 → 不稳健）

**测什么**：一次发作从头到尾，电活动"空间形状"怎么变。按间期两模板把触点分块——两模板各自最早响应的小核（端点）、两核**之间**的"走廊"中段、离轴的横向触点——看随发作进程走廊（轴向）相对活动是否**减弱**、横向是否**增强**、整体是否更同步、场方向是否漂移。

**怎么测**：每 10s 窗算各触点相对发作前 baseline 的 robust-z（和 A 线同口径），比四分区的"正质量占比"；每次发作算 progress 与走廊占比的 Spearman ρ（期望<0）/ 非轴向占比的 ρ（期望>0），每被试再对各发作的 ρ 做 Wilcoxon（**发作=重复单位**）。队列=broad 9（8 swap + E916 非 swap）+ narrow 7 平行批（用每模板**端点 compact core** 构轴，证明**不必 swap**）；source 核 = 每模板最早 compact top-2-3，非 decision_k 整串。

**结论（broad + narrow 扩队列后）**：发作场随时间**在变**（GIF 可见）、整体**仍贴间期轴**（maxAB 不降反升）；但"轴向走廊变弱/非轴向变强"**不是稳健现象**——**broad 有暗示**（轴向 median ρ<0 在 5/8、非轴向 ρ>0 在 8/8），**narrow 扩队列不复现甚至反向**（轴向 ρ<0 仅 3/7、非轴向 ρ>0 仅 2/7；E1146 轴向 ρ=+0.52、E442 +0.37）；两队列各仅 1 被试 Wilcoxon 显著。**扩队列调查把 broad 的方向暗示证否了一半 → 依队列/substrate/走廊厚度而变。** 非 swap 确能构轴（E442/E916 swap=none 有走廊；E916 发作太短无趋势）。**与 A 线"轴共享"在时间维一致（maxAB 保持），但方向减弱假设阴性。** caveat：z-ER 中后期偏示意，场图/GIF 只作相对空间形状看。

**paper 用途**：本图模式（间期 **A|B 锚** + 发作场演化 + **GIF 直观看发作场的传播变化**）保留为 **supplementary**。

完整方法 + 数值表 + parity + 图/动画说明：`docs/archive/topic5/ictal_field_dynamics_pilot_2026-06-28.md`。

---

### 3.7 发作前 criticality/state 层是否投影到间期 HFO 几何（2026-07-01，V2a，exploratory，restricted-axial sanity check → 偏阴性，已 reframe 到 V3a）

**测什么**：真发作前最后约两分钟，几种"要失稳的迹象"——每触点变脆程度（易感场 K_t = variance/lag-1 autocorr/line-length rate 的 late−early 变化）、主导动力学模态落点（M_loading）、连锁激活前向流量（avalanche）——是否沿间期 HFO 传播轴 `G_HFO` 排布。

**怎么测**：state 信号 = 宽带能量 baseline-robust-z（`bb_zt`，只取发作前段）。subject 为单位（窗→发作→被试→队列中位数），broad/narrow 不 pool。动力学腿用 phase+block surrogate（各 1000 次，自建）；易感场/avalanche 的空间/顺序 null 当时依赖并发 session 的 Phase-1（未建）→ 挂 `pending_phase1`，**不假造 null**。

**结论**：**受限实现下没支持**——三条腿队列有符号中位数≈0 且被试间符号不一致（非"平均掉"）；唯一有真实（时间）null 的动力学腿 **M_loading 0/16、λ 趋势 0/16 显著**（两种 surrogate 都过）。**最耐用的产出 = 方法学定律**：raw λmax≈0.90–0.95 被 surrogate 解释掉（宽带 envelope 平滑本身造高自相关）→ **今后所有 λmax/VAR/DMD/Jacobian 一律报 `λ_surplus`（观测 − surrogate 中位数），不报 raw**。方法学副产：avalanche rank-coupling 0.64–0.91 但前向流量≈0（自持假象非传播）→ 主指标必须用前向位移，不用 rank-coupling。

**定位（重要）**：这是 **restricted axial preictal-only sanity check**（只看 −120~0s、两段窗、限 HFO 匹配触点、按 relt=0 而非 eeg_onset 锚定、无显式非轴向假设），**不判定**模型真正的预测（发作早期**轴向组织减弱 + 非轴向活动/流/模态放大**）。已 reframe 到 V3a。**不能写**："发作前没有临界性/state projection"、把 λmax≈0.95 当临界、把 rank-coupling 当传播。

完整方法 + 数值表 + 禁止 claim：`docs/archive/topic5/v2_phase2_criticality_state_layer_2026-07-01.md`；后继设计：`docs/superpowers/specs/2026-07-02-topic5-v3a-mode-transition-design.md`（V3a axis→non-axis mode transition）。**工具库存见 §7。**

---

### 3.8 发作"轴向→非轴向"模态转移（2026-07-04，V3a，exploratory，**脆弱阳性 · pending sensitivity/V3b**）

**测什么**：发作一启动，系统里最容易被放大的方向/连锁流向，会不会从那条固定的间期 HFO 小路搬到小路之外的触点上——非轴向连锁流放大（H3b）+ 密度归一奇异模态转向非轴（H3c）为 co-primary，轴向减弱（H3a）仅辅助。承重锚 发作前 30~10s（P3）→ 早发作 10~30s（I1），每发作按其脑电起始**在同一批发作内前后配对**，被试为单位，narrow 主 / broad 复制不合并。

**怎么测**：非轴向 = 纯间期 HFO 参与度定义（对发作全盲，防循环）；率保/密度归一/相位/块/标签自建零假设各 1000 次；p 算在 Δ(I1−P3) 置换分布上；λ 只报 surplus。

**结论（脆弱阳性，暂不可当确立结论）**：扣掉每触点放电率随机基线后的**非轴向净流增量**在主队列（Holm-p 0.031，刚过 0.05）和复制队列（0.008）都达到队列级显著 → 判读机械上到 tier 4 / `state_v3_supported=True`。**但五个来源使它很脆**：① 未校正的原始流大多在**下降**（主队列 5/7）——所谓"放大"是相对基线偏高、不是绝对增加；② 主队列 6/7 以**同时共激活**为主（lag1≈lag0），不是定向传导；③ 个体稳健性几乎全塌（流腿 0/7 过完整稳健门，`subject_support` 主队列 1/7、broad 0/9）；④ 更能代表"方向转移"的**模态腿(H3c)全阴**（Holm 0.89/0.63）；⑤ **主队列显著本身对单被试不稳健**——去掉主队列 7 个被试中任一（除 442）Holm 都翻到 0.0625 失败（6/7 drop 不通过），只有复制队列去一稳，即 tier-4 依赖 narrow 恰好这 7 个的配置。所以这是**数据侧候选信号，不是确立的"轴→非轴模态转移"**；机制升级要 V3b。**关键**：这个 tier-4 是"同一批发作内配对"修复后从 tier-2 翻上来的——未配对（P3、I1 拿不同发作子集相减）的错配噪声把一个符号一致的效应打散成不显著（如 253 −0.191→+0.035）；个别被试 1125 三腿方向最一致。

**不能写**："V3a 成立 / 发作发生模态转移"、"off-axis flux 增加"、把它当定向传导、上机制主张、把 broad 单独当主结论。完整方法+逐被试数值表+禁止 claim：`docs/archive/topic5/v3a_mode_transition_2026-07-04.md`；后继：V3p（preictal-only 轨迹）/ V3b（模型–数据一致性）。

### 3.9 V2 表达层三问收口（2026-07-04，Phase-1-v2，exploratory **candidate scaffold refinement**）

**测什么**：Phase-1 确认了"间期 HFO 几何 (G_HFO) ↔ 发作早期多频带能量场"的 cohort 层对齐（band-generic、非 ripple 独有）后，回答三个必要问题——**扛得住吗（survive：把宽带/1-f 扣掉后还在？）· 谁有（who：少数驱动？）· 什么时候（when：发作前已在还是点着才现？）**。

**结论（三问；tier = candidate scaffold，不越 formal/机制）**：
- **survive**：对齐**非纯宽带**（扣 LOBO 共有场后两池各 **4/7** primary 过 cohort-perm FWER、**α+γ 一致** = 频段特异残差层，Outcome B）；但**大半 1/f-可归因**（扣 1/f 后只 **gamma_LVFA** 两池稳；余塌）；**ripple 在 broad 扣 1/f 后"存活"= 多重比较天花板假象**（+ fs512 贴奈奎斯特被试带偏、绝对对齐反低于 raw、narrow 不显著、与 Phase-1 矛盾）→ **绝不 HFO/ripple 特异**。1/f 拟合 QC 干净（r²≈0.79、失败 1.7%）。
- **who**：cohort 6/7 是**聚合**、per-subject 弱（≥4/7 仅 **6/20**）、无单一表型预测（Spearman 全 <0.4 门；过闸特征都是同向量再描述）、band-generic。→ **subject-heterogeneous**。
- **when**：**发作前已高且平**（近前−远前≈0，无爬升）+ **起始处不大上抬**（broad 符号翻转 p≈0.005 支持、narrow 临界 p≈0.06/Wilcoxon<0.05）、**EEG 锚更清**（临床起始常早于 EEG 起始几十秒 → 临床锚抹平上抬）。→ **preictal-present + modest onset increment**。

**⚠️工程教训**：并行 numpy nulls worker 必须 `OMP_NUM_THREADS=1`（否则 26 worker×80 核 BLAS 超订 → 单被试 crawl，7h/40min 之差）；broad=16（1146=**强正向离群**（broad n_sig 7/7）、单被试 n_perm=1000 单线程 ~2h、OMP 不助（Python-loop permute）→ 排除；保守（matched raw 6/7→3/7 印证只降存活）、已在 narrow 全量、后台补跑；within_shaft_strong 仅 narrow 属性、broad 里是 subject_wide_weak）。

**不能写**：formal Gate A/B/C passed · HFO-/LVFA-/ripple-specific · timing-order replay · criticality/机制 · 过任何空间随机场。完整数值 + 承重 caveat + 禁 claim：`docs/archive/topic5/v2_phase1_v2_scaffold_refinement_2026-07-04.md`。

---

## 4. 已知 caveat

1. **gap_perm bug-fix 实测影响小**：cohort 28 个 ok rows 上 Δgap_perm 中位 −0.0007、
   abs_max 0.061，**0 个 over_split_flag flip**，0 个 sentinel jaccard 变化。bug 真实
   但本 cohort 上效应几乎为零；早先 sentinel 视觉裁定与 ARI 等所有 PR-1 结论保持有效。
2. **共同 channel mask 缺失**：z-ER feature 用 nanmean per bin 处理跨 seizure 缺通道，
   高 coverage subject 基本无害；低 coverage subject 应做 sensitivity (intersection-only mask)。
3. **916 / 1077 sentinel 失效**：是 v2.3 status filter / `n_ok < 5` 门的副产物，
   不是聚类失败；不能作 recall/precision evidence。
4. **548 gamma fine subdivision 不能确认真 7 类**：sensitivity battery
   (min_subtype_size=3 / 不同 bin / common channel mask / bootstrap stability)
   是 commit 前必跑项。
5. **Yuquan 部分覆盖**：cohort=25 含 9 yuquan audit_eligible (gaolan, huanghanwen, litengsheng, pengzihang, sunyuanxin, xuxinyi, zhangjinhan, zhangkexuan, zhaojinrui)。仍有 12 yuquan 因 n_seizures<2 被排除（chengshuai, dongyiming, hanyuxuan, huangwanling, liyouran, songzishuo, wangyiyang, zhangjiaqi, zhaochenxi, zhourongxuan, chenziyang, zhangbichen），ictal pool 不足以做 within-subject 聚类，无法补救。9 个 yuquan 中 4 个 (litengsheng, sunyuanxin, zhangkexuan, zhaojinrui) 在至少一个 band 上达到 ok 状态；其余 5 个落 insufficient_n（CUSUM 阈值 λ=100 cap 下 onset 未触发）。yuquan ok 子集的 z-ER 聚类质量 (silhouette 0.495 / gap_perm 0.552) 高于 epilepsiae ok 子集 (0.418 / 0.325)，但样本太小不足以做"yuquan vs epi"对比。
6. **方法溯源严格性**：Schroeder 2020 *PNAS* 是生物先验（within-patient pathway
   variability），**不是 pipeline 复现**。本 PR-1 聚类管道 (1−Spearman + UPGMA +
   silhouette + permutation null + outlier split) 全部本项目实现；Panagiotopoulou
   2022 *Brain Communications* 不能作为 pipeline 直接溯源。
7. **`over_split_flag` 是 descriptive flag，不是过切检验**：真正过切判定需要
   gap_perm（正确 null）+ 视觉 diagnostic + sensitivity 三方一致。
8. **z-ER 中后期偏示意 + 方向假设不稳健（§3.6 field 动力学）**：baseline-robust-z 相对发作前安静期
   归一，越往发作中后期越不可靠 → 场图/GIF/轨迹中后期只作**相对空间形状**看。"轴向走廊变弱"在 broad
   有暗示但 **narrow 扩队列证否**（多反向）→ 非稳健现象，不进 claim。"走廊"**不必 swap**（非 swap 端点
   也能构轴），但需非退化轴 + 中段有电极（broad 8/9、narrow 7/7 可测；253 双侧无中段电极不可测）。
9. **λ_surplus 方法学定律 + eeg-onset 锚定（§3.7 V2a）**：宽带 envelope 平滑本身造高自相关，raw λmax≈0.95
   被 phase/block surrogate 解释掉 → 今后所有 λmax/VAR/DMD/Jacobian **只报 λ_surplus（观测−surrogate 中位数），不报 raw**。
   avalanche **rank-coupling≠传播**（自持假象 0.64–0.91 但前向流≈0），主指标用前向位移。时间上 cache `relt=0`
   ≠电生理 onset（`eeg_onset_rel` 偏移数秒）→ 发作前/发作窗必须按 `eeg_onset_rel` 锚定，不用 relt=0。

---

## 5. 历史文档索引

- `docs/archive/topic5/INDEX.md` — topic5 archive 索引
- `docs/archive/topic5/pr1_seizure_clustering/pr1_zer_cohort_2026-05-10.md` — PR-1 主结果文档（audit-corrected）
- `results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/figures/README.md` — cohort 视觉骨架文档
- `docs/superpowers/specs/topic5_pr1_seizure_clustering*.md` — PR-1 plan v2
- `docs/superpowers/plans/2026-05-10-topic5-pr0_1-yuquan-ictal-cohort-extension.md` — yuquan cohort 扩展 plan (2026-05-10)
- `docs/archive/topic5/bridge_q1/bridge_q1_results_2026-05-10.md` — Topic 1 × Topic 5 Bridge Q1 cohort exploratory result (verdict: NULL-locked, n=9; power floor identified)
- `docs/archive/topic5/bridge_q1prime/bridge_q1prime_results_2026-05-10.md` — Topic 1 × Topic 5 Bridge Q1' PIVOT case-series result (verdict: INDETERMINATE, N=4 strict + 548 candidate; 4 strict subjects show consistent positive Cramér V 0.25–0.67 median 0.486 but underpowered on p; channel-rank correspondence × swap-subset)
- `docs/archive/topic5/bridge_q1prime/q1prime_overnight_exploration_2026-05-10.md` — Q1' overnight 探索：full 25-subject cohort + per-seizure feature × delta_rho/subtype 相关性分析 (verdict: INDETERMINATE/WEAK-SIGNAL; median_onset_latency_sec 有方向性倾向 sign_p=0.039 uncorrected; Stage C 全 NULL; 2 subjects 有大效量 subtype 区分)
- `docs/archive/topic5/echo_gate/stage1_proxy_triage_2026-06-08.md` — Stage 1 ictal-template-echo gate（ER 代理）：ER 代理 echo = **共享粗锚**非 specific-path-replay（coarse positive 但 anchor-matched FLAT + bad-data self-check clean）
- `docs/archive/topic5/dynamic_echo/stage2b_sentinel_2026-06-12.md` — **Stage 2b early-ictal dynamic-pattern echo sentinel（gate NOT PASSED，B=500 n=3，exploratory）**：早发作动态有模板相关结构（过 channel max-null）但**非稳定早期路径复演**——峰时偏晚、confirmatory 早窗方向逐发作变号、yuquan within-shaft 塌掉 → **共享粗解剖/杆级锚为主**，细化并一致于 Stage 1。**未进 cohort、不支持路径复演主张**；first-onset recruitment（Stage 2，量错失败）的接替。
- `docs/archive/topic5/ictal_direction_clustering_2026-06-27.md` — **发作早期方向无监督两类 ↔ 间期 A/B（exploratory，negative）**：6 个干净 ECoG、宽带+快活动全无 two_class_mapped；唯一真两堆的 442 其两堆与间期模板对不齐到超随机（two_class_unmapped）。防自欺=主方向+散点 null（弃纯单峰）+ best-pair 旋转 null + 预锁轴质量门；与 A 线一致=粗网络/解剖锚非方向两类重放。
- `docs/archive/topic5/ictal_field_dynamics_pilot_2026-06-28.md` — **发作内 field 动力学 pilot（exploratory，broad 暗示 / narrow 扩队列证否 → 不稳健）**：broad 9（轴向 ρ<0 5/8、非轴向 ρ>0 8/8 暗示）vs narrow 7 平行批（ρ<0 仅 3/7、非轴向 2/7，多反向，E1146 轴向 +0.52）→ 方向减弱假设非稳健、依队列/substrate；非 swap 也能用模板端点构轴（不必 swap）；图/GIF 模式 = paper supplementary；z-ER 中后期偏示意。
- `docs/archive/topic5/v2_phase2_criticality_state_layer_2026-07-01.md` — **V2a 发作前 criticality/state 层（exploratory，restricted-axial sanity check → 偏阴性）**：动力学腿 M_loading 0/16 + λ 趋势 0/16 显著（phase+block surrogate）；方法学定律 = 今后 λ 报 surplus 不报 raw；rank-coupling≠传播。不判定模型 axial-weakening/non-axial 预测 → reframe 到 V3a。
- `docs/superpowers/specs/2026-07-02-topic5-v3a-mode-transition-design.md` — **V3a 设计（axis→non-axis mode transition，data-side H3a/b/c）**：eeg-onset 锚定 phase grid、2D 投影算子 + 非正规 reactivity、纯间期 HFO 非轴向（防循环）、compartment flux、自建 null、narrow 主。姊妹 spec V3b（M3B 模型–数据一致性 H3d）待写。
- `docs/archive/topic5/v2_phase1_v2_scaffold_refinement_2026-07-04.md` — **V2 表达层三问收口（Phase-1-v2，exploratory candidate scaffold refinement）**：**survive** = 非纯宽带（common_resid 两池 4/7、α+γ 一致 = Outcome B）但大半 1/f-可归因（aperiodic 只 gamma_LVFA 稳）、ripple broad"存活"=多重比较天花板假象 → **NOT ripple-specific**；**who** = cohort 6/7 聚合、≥4/7 仅 6/20、无单一表型、band-generic（subject-heterogeneous）；**when** = 发作前已高且平 + 起始处小上抬（broad 符号翻转 p≈0.005、narrow 临界）、EEG 锚更清（preictal-present + modest onset increment）。broad=16（1146 计算受限 + 强正向离群排除，保守；within_shaft_strong 仅 narrow 属性）。⚠️并行 numpy nulls 必须 OMP_NUM_THREADS=1。

---

## 6. 下游 PR 必须遵守

1. **Per-subtype 不 per-subject**：从
   `results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/per_subject/<sid>__zer_binned.json`
   读 `result["per_band"][band]["subtype_label"]` 与 `outlier_flag`，按 subtype
   分别统计 pre-ictal / outcome / propagation 指标。**禁止**在 subject 内对所有 seizure 平均。
2. **`subtype_size < 3` 处理规则**：cohort 有不少 size=2 的 subtype，
   小子型 statistical power 不足。下游 PR 必须在自己的 README 标处理规则
   （pool with annotation / drop / 全 cohort CI）。
3. **band 选择**：`gamma_ER` vs `broad_ER` cohort 数值近似 (median ARI ≥ 0.6)；
   下游可任选其一作主分析，另一作 sensitivity；不能两 band 同时跑而不合并解释。
4. **t_onset feature 已被 z-ER 取代**：`per_subject/*.json` (无 `__zer_binned` 后缀)
   保留为历史归档，不再作为 PR-2+ 的 subtype 来源。

---

## 7. 文件清单

### 代码

- `src/ictal_zer_features.py` — z-ER tensor extraction + binning + channel-order check
- `src/ictal_seizure_clustering.py` — pairwise dissim, UPGMA, k selection, channel(-block) permutation null, outlier/subtype split, sentinel jaccard, EEG-realign helpers
- `src/ictal_seizure_plotting.py` — MDS, subtype color palette, sort orders
- `scripts/cluster_ictal_seizures.py` — CLI driver `per-subject / cohort / render`
- `scripts/diagnostic_cluster_grid.py` — cluster-grouped per-seizure thumbnail grid

### 测试

- `tests/test_ictal_seizure_clustering.py` (33 tests) + `tests/test_ictal_zer_features.py` (5 tests) + `tests/test_ictal_seizure_plotting.py` (8 tests) = **45 tests pass**

### 数据 / 图

- `results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/`
  - `per_subject/*__zer_binned.json` — 25 cohort z-ER cluster results (16 epilepsiae + 9 yuquan)
  - `cohort_summary__zer_binned.csv` — 50 subject-band rows
  - `cohort_summary__zer_binned__pre_audit_2026-05-10.csv` — pre-audit 快照 (n=16, 32 rows)
  - `figures/per_subject/*.png` — 25 per-subject 4-panel PNG
  - `figures/diagnostic/*.png` — 6 cluster grid 视觉诊断
  - `figures/README.md` — cohort 视觉骨架文档（中文）

### Run logs

- `results/run_logs/cohort_zer_20260509_2104.log` — pre-audit cohort run
- `results/run_logs/cohort_zer_audit_20260510_1045.log` — audit-rerun (channel-block null + ch-order check)

### V2a criticality/state 层辅助工具（2026-07-01，22 tests pass）

- `src/topic5_v2_criticality.py` — 纯函数：contact_susceptibility、ridge-VAR 套件（prepare_var_window 向量化 / var_window_ok / var1_ridge / spectral_radius / leading_eigvec / recovery_tau / cv_one_step_r2）、surrogate（block_shuffle / phase_randomize）、avalanche（activations_from_z / avalanche_atm / branching_ratio / atm_forward_displacement / atm_direction_index / atm_rank_coupling_spearman）
- `scripts/_topic5_v2_crit_io.py` — 共享 plumbing：load_context + ictal_field_long_cache 发作前段 → matched-contact 包络 E；state_prefix / shaft_of / window_index_range / get_contact_alignment（band_scan-or-shim）/ get_null_fns
- `scripts/run_topic5_v2_crit_{susceptibility,dynamics,avalanche,summary}.py` — 三腿 + 汇总 runner（subject 为单位、broad/narrow 不 pool、skip 记录）
- `scripts/plot_topic5_v2_crit_summary.py` — 结果双面板图（3-leg 对齐 + 前向流 vs 自相关方法学核验）
- `src/_topic5_v2_p1_contract_shim.py` — 临时 contact_alignment shim（逐字复刻 Phase-1 Task-5，band_scan 落地后自动退役）
- `config/topic5_v2_phase2.yaml` — 相位窗 / surrogate / seizure cap 配置
- 测试：`tests/test_topic5_v2_criticality.py` + `test_topic5_v2_crit_io.py` + `test_topic5_v2_crit_dynamics.py` + `test_topic5_v2_crit_legs.py` = **22 tests pass**
- 结果：`results/topic5_ictal_recruitment/v2_criticality/{broad,narrow}/phase2_*_subject.csv` + `figures/phase2_state_layer_alignment.png` + `figures/README.md`
- 复用边界（→ V3a）：io/surrogate/ATM primitives 直接继承；时间窗（改 eeg-onset 锚定 grid）、几何（改 signed axis + 非轴向）、null（自建 spatial/order/label/rate-preserving）需重写。
