# Topic 5 Archive Index

> **主入口**：`docs/topic5_seizure_subtyping.md`（§5 历史文档索引含完整 backlink）
> **范围**：以 ictal seizure 本身为研究对象（subtype / pre-ictal / propagation / outcome）。
> **不属于**：interictal 事件内部传播（topic1）、IEI/PSD（topic2）、SOZ 空间归因（topic3）、模型层（topic4）。

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
