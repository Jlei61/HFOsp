# Topic 5 Archive Index

> **主入口**：`docs/topic5_seizure_subtyping.md`（§5 历史文档索引含完整 backlink）
> **范围**：以 ictal seizure 本身为研究对象（subtype / pre-ictal / propagation / outcome）。
> **不属于**：interictal 事件内部传播（topic1）、IEI/PSD（topic2）、SOZ 空间归因（topic3）、模型层（topic4）。

## 主线（network-axis pivot）

### `axis_alignment_AB_result_2026-06-14.md` — **现阶段主线结果**：间期传播轴 ↔ 发作早期激活的轴对齐（A 线 primary + B 线 secondary）
- 18 Epilepsiae 队列：粗"共享网络主轴"稳（broadband 稳赢全通道 null，FDR + LOSO 扛住）；细对齐仅快活动（hfa）稳（过最严 joint）；符号自由共线，非逐点重放。
- 含完整方法 / 定稿数值表 / 工件清单 / handoff。计划全貌：`network_axis_pivot_plan_2026-06-13.md`（A/B 段已标 ✅ 执行）。
- 定稿表 `results/topic5_ictal_recruitment/axis_alignment/axis_alignment_FINAL.md`。

### `hfa_joint_confirm_2026-06-15.md` — hfa×joint 冻结复验（split-half + 负对照）
- 唯一过最严 joint null 的 hfa 细对齐：full 干净复现（Wilcox=0.022）但**奇数半不显著（0.078）→ 非 split-half 稳健**；负对照四层全部非显著=非假阳性。
- 结论 = real-but-not-robust，**维持灵敏度档、不升 primary**；升格须独立第二队列。主线粗骨架不受影响。

## PR 系列

### `pr1_seizure_clustering/` — Per-subject seizure subtyping (z-ER tensor + 1−Spearman + UPGMA)
- `pr1_zer_cohort_2026-05-10.md` — **主结果文档**：cohort z-ER subtyping，含 sentinel 视觉裁定、audit fix 历史、over_split 规则演化
- 见 `results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/figures/README.md` 的 cohort 视觉骨架
- 计划档：`docs/superpowers/specs/topic5_pr1_seizure_clustering.md`（plan v2）

### PR-0：v2.3 Layer A ictal ER timing atlas
（追授 topic5 PR-0；详细 spec 见 `docs/superpowers/specs/`）

### Bridge → Ictal-template echo 谱系（Topic 1 × Topic 5：间期传播模板是否在发作期复演）
- `bridge_q1/bridge_q1_results_2026-05-10.md` — Q1 cohort（verdict NULL-locked, n=9, power floor）
- `bridge_q1prime/bridge_q1prime_results_2026-05-10.md` + `q1prime_overnight_exploration_2026-05-10.md` — Q1′ case-series（INDETERMINATE）
- `echo_gate/stage1_proxy_triage_2026-06-08.md` — **Stage 1** ER 代理 echo gate：= 共享粗锚，非 specific-path-replay
- `dynamic_echo/stage2b_sentinel_2026-06-12.md` — **Stage 2b** early-ictal 动态模板 echo sentinel：**gate NOT PASSED**（B=500 n=3）；有模板相关结构但非稳定早期路径复演 → 粗解剖/杆级锚为主；未进 cohort。（Stage 2 first-onset recruitment 量错失败，未单独归档，见此文档"谱系"段）
