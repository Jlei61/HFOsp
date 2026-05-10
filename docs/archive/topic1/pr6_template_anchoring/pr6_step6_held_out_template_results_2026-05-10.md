# PR-6 Step 6 — Held-out Time Template Stability（results）

> Plan-of-record：`pr6_step6_held_out_template_plan_2026-05-10.md`
> Tier：Robustness / sensitivity，**不进 paper α，不开新 cohort claim**。
> 主文档：`docs/topic1_within_event_dynamics.md` §7.10 已加结果回链。

## 1. 一句话结论

在论文记录窗口内（Yuquan 24h / Epilepsiae 数日），**严格 train/test 不对称下 PR-6 主线 endpoint 与 §8 swap_class 总体稳健**：33/35 subject 落在 strong / moderate tier，0 subject fail，cohort 中位数 template Spearman = 0.922、endpoint position recall = 0.833、swap_class concordant 比例 = 68.6%（24/35）。**PR-6 主线 H1 cohort NULL 不是 PR-2 + PR-6 same-events double-dipping artifact**，weak 例外（`yuquan_litengsheng`、`yuquan_zhaochenxi`）单独说明，不波及主结论。

## 2. Cohort & 数据

### 2.1 入选 cohort

n = **35** stable_k=2 subject（PR-6 main cohort 23 与 rank_displacement v14 sensitivity n=35 的并集，去重）。

- **valid_mask provenance**：30 subject 走 PR-6 `per_template[k].valid_mask` union；5 subject（`epilepsiae_1125 / 384 / 620 / 916`、`yuquan_gaolan`）走 PR-2 `(template_rank ≠ −1)` sentinel union（PR-6 SOZ 缺失 / PR-6 anchoring 缺失 fallback；与 rank_displacement v14 同 provenance）。
- **dataset breakdown**：epilepsiae 19、yuquan 16。
- **不在 cohort 内**：stable_k ≠ 2 subject（818 k=4、zhangjinhan k=6、huanghanwen k=2 但 PR-6 cohort 之外的 case-series）已按 PR-2 / PR-6 既有规则单独处理，本 Step 不重做 stable_k 决策。

### 2.2 Inputs（已 read，未 mutate）

- `results/interictal_propagation/per_subject/<stem>.json`（PR-2 adaptive cluster + channel_names + template_rank）
- `results/interictal_propagation/template_anchoring/per_subject/<stem>.json`（PR-6 per_template valid_mask，可缺）
- `results/interictal_propagation/rank_displacement/cohort_summary.json`（§8 dual-tier swap_class label：strict 10 / candidate 8 / none 17）
- lagPat npz：`/mnt/yuquan_data/yuquan_24h_edf/<subject>/` 与 `/mnt/epilepsia_data/.../<subject>/all_recs/`
- `results/{epilepsiae,yuquan}_soz_core_channels.json`（H1 descriptive 报告，固定 endpoint 集合下不参与稳健性判定）

## 3. 方法（按 plan §6 执行）

每 subject 跑 `compute_held_out_endpoint_validation`（`src/interictal_propagation.py`，新加）：

1. 事件按 `start_t` 排序，median 切 first / second 两段（**严格按时间，不 recluster 第二半**）。block_time_ranges 边界由 `_split_events_by_time` 强制（CLAUDE.md §6 boundary 合同）。
2. **Train（仅 first half）**：KMeans (k=2, n_init=10, random_state=0) → labels_first → `build_cluster_templates` → templates_first → top-3 ∪ bottom-3 endpoint channel set（与 PR-6 主定义一致）。`compute_swap_score_sweep` (1000 perm, seed=0) 给 swap_class_first。
3. **Test（仅 second half）**：`assign_events_to_templates(ranks_second, bools_second, templates_first)` 投射，无 recluster；`build_cluster_templates` 在每 cluster 投射后的 events 上重算 mean rank → templates_second_projected。再次跑 `compute_swap_score_sweep` 给 swap_class_second_projected。
4. **四个稳健性量**：
   - `template_spearman` = mean over k of Spearman ρ(templates_first[k], templates_second_projected[k])
   - `endpoint_position_recall` = direction-preserving mean of (top→top, bottom→bottom) Jaccard between train endpoint and second-half top/bottom-3 of projected template
   - `assignment_coverage` = fraction of second-half events with non-(-1) cluster assignment（plan 原名 `cluster_assignment_purity`，但 nearest-vs-2nd-nearest gap purity 未实现，user 2026-05-10 review 改名以与实际语义一致；descriptive only，不进 tier）
   - `swap_class_concordant` = swap_class_first == swap_class_second_projected
5. **Per-subject tier**（plan §7.1）：strong（3/3 strict 阈值）/ moderate（2/3）/ weak（1/3）/ fail（0/3）。

代码改动只新加两个 helper（不改 `compute_time_split_reproducibility` / 不动 PR-6 主线 helper）；49 个 PR-2/PR-2.5/PR-3/PR-4B 既有测试全绿；12 个新加测试覆盖 split / pipeline / endpoint drift / stub 合同 / swap concordance。

## 4. 主结果

### 4.1 Tier 分布

| Tier | n | 比例 | 说明 |
|---|---|---|---|
| strong | **20** | 57.1% | 3/3 阈值都过；100% swap_class_concordant |
| moderate | **13** | 37.1% | 2/3 阈值过；4/13 (31%) swap_class_concordant |
| weak | **2** | 5.7% | 1/3 阈值过；0/2 swap_class_concordant |
| fail | **0** | 0% | 无 subject 三项全跌 |

**33/35 subject 在 strong + moderate**，整体几何稳定性良好。

### 4.2 cohort-level 量

| 量 | 中位数 | IQR | 阈值 | 通过 |
|---|---|---|---|---|
| `template_spearman` | 0.922 | [0.893, 0.958] | strong > 0.7 | ✓ |
| `endpoint_position_recall` | 0.833 | [0.750, 0.917] | strong > 0.6（baseline 0.30） | ✓ |
| `assignment_coverage` | 1.000 | [1.000, 1.000] | descriptive | — |
| `swap_class_concordant` 比例 | 68.6% | 24/35 | acceptance > 50% | ✓ |

`endpoint_position_recall` SE = 0.026（pilot 子集 0.048）—— 远低于 plan §7.3 halt 阈值 0.20。

### 4.3 Tier × §8 swap_class 交叉

| Tier \ swap | strict (10) | candidate (8) | none (17) |
|---|---|---|---|
| strong (20) | 4 | 4 | 12 |
| moderate (13) | 5 | 4 | 4 |
| weak (2) | 1 | 0 | 1 |

**关键观察**：strong tier 中 12/20 (60%) 是 §8 swap_class = none subject。意思是 hold-out 稳健性**不依赖**于 swap geometry —— 即使没有显著 forward/reverse swap 的 subject，其 first-half template 与 endpoint 集合在 second-half data 上仍稳。这跟 PR-6 主线 H1 cohort NULL（pooled p=0.42）的判读一致：稳健的是 endpoint 几何本身，不是它对 SOZ 的 anchoring。

### 4.4 Tier × dataset

| Tier \ dataset | epilepsiae (19) | yuquan (16) |
|---|---|---|
| strong | 12 | 8 |
| moderate | 7 | 6 |
| weak | 0 | **2** |

两个 weak subject 都在 yuquan：

| Subject | template_spearman | endpoint_position_recall | swap_concord | §8 swap_class |
|---|---|---|---|---|
| `yuquan_litengsheng` | 0.760 | 0.500 | False | none |
| `yuquan_zhaochenxi` | 0.721 | 0.583 | False | strict |

`zhaochenxi` 的 §8 strict 标签在 hold-out 下被推翻（first → none），说明 §8 dual-tier 标签在该 subject 内不是 within-recording 稳定的；考虑到 `zhaochenxi` 在 rank_displacement v14 加入时还是 PR-2.5 fwd/rev-reproduced 候选 subject，本 Step 6 提示它的 swap geometry 可能是 single-fold pattern。**应当作 case-series 不进 cohort claim**。

### 4.5 swap_class concordance 是最脆弱的量

按 tier 拆 swap_class concordant：

| Tier | concordant | total | 比例 |
|---|---|---|---|
| strong | 20 | 20 | 100% |
| moderate | 4 | 13 | 31% |
| weak | 0 | 2 | 0% |

swap_class 在 strong tier 100% concordant、在 moderate tier 跌到 31%。原因：variable-k swap_class 标签依赖 1000-perm max-null，N/2 events 下 perm-null 噪声放大，从而把一些 strict / candidate subject 在 second projected 上推到 none，反之亦然。这**不是** PR-6 主线 endpoint 几何不稳，而是 §8 dual-tier label 本身的功率边界 —— 已在 §8 plan 里写死 "mechanism sanity tier，不开新 cohort claim"，本结果不与 §8 既有结论矛盾。

## 4.6 Supportive null caveat（2026-05-10 post-hoc，epilepsiae_548 单 subject——**caveat 不是正式校准**）

完成主 cohort + figures 后，advisor review 指出 `template_spearman` 在 hold-out pipeline 下有结构性 selection-bias floor —— `assign_events_to_templates` 总把 second-half event 分到最近 template，导致即便 second-half ranks 完全打乱，cluster 内 mean rank 仍向 template 漂移。在 `epilepsiae_548`（pilot strong subject）上 per-event independent rank-shuffle 50 trials 跑完整 pipeline，得到 single-subject null 分布：

| 量 | null median (epi_548) | null p95 | null max | cohort median (n=35) | gain（仅 single-subject 标尺） |
|---|---|---|---|---|---|
| `template_spearman` | **0.747** | 0.845 | 0.906 | 0.922 | +0.175 |
| `endpoint_position_recall` | **0.667** | 0.796 | 0.917 | 0.833 | +0.166 |
| `swap_class_concordant` (fraction True) | **0.360** (18/50) | — | — | 0.686 (24/35) | **+33 pp** |

**重要约束（user 2026-05-10 review 写死）**：

> 用一个 subject 的 null 去支撑整个 n=35 的 calibrated gain 是**弱的**。本节只能写作 **"supportive calibration / caveat"**，不是全 cohort 校准完成。**正式 paper-level tier rebalancing 需要 per-subject null 或至少按 `n_valid` / `n_events` stratified null**——本批次未做，列入 Step 6 follow-up。

**关键含义**（在 supportive caveat 标尺下，**不**作 calibrated cohort claim）：

1. **plan §7.1 的"strong"阈值（spearman > 0.7、recall > 0.6）至少在 epi_548 single-subject null 下跨在分布之上**——意味着这两个阈值并不强；cohort 的 strong 20 / moderate 13 计数依 plan-of-record 保留，但 paper-level 表述不能把"strong 20"作为效应量声明。
2. **`swap_class_concordant` 在 epi_548 null 下的 36% baseline 与 cohort 的 68.6% 之间存在 +33 pp gap**——**至少在该 single subject 上** swap_class concordance 比 spearman / recall 更具判定力；这是一条值得在后续 per-subject null 下验证的 working hypothesis，不是已成立的 cohort 结论。
3. **正文 §4.5 中"swap_class concordance 是最脆弱的量"是基于 raw count 视角**——supportive caveat 给出反向直觉；两者**都不**应升级为 cohort claim，本节只记录两种视角并列存在的事实。
4. **不重新 re-tier**：tier 阈值是 plan-of-record，post-hoc 改阈值引入 fishing 风险。tier 计数保留，解读 caveat 已加。
5. **per-subject null 异质性**：不同 subject 的 `n_valid`、`n_events`、cluster size ratio 都会改变 null floor。Single-subject epi_548 的 null 不能外推到 n=35 全 cohort——这是本节最强的 caveat。

文件：`results/interictal_propagation/pr6_step6_held_out_template/null_calibration_epi_548.json`（标记 `single_subject_supportive_caveat`，不是 cohort calibration artifact）。

**Direction-preserving recall 与 plan 的协调（advisor 2026-05-10）**：plan §7.1 给的 baseline `≈ 6/n_valid` 是 direction-blind（任何端点是否仍处于任意 extreme）；实现中采用 direction-preserving（top→top + bottom→bottom 各自 Jaccard 平均），baseline 退到 `≈ 0.30`（n_valid=10）。Direction-preserving 是更严格的判定，cohort recall 0.833 在它下面意味着 endpoint 在保持端点身份与极性方向上稳健。Plan 的 baseline 公式按"direction-preserving 实现"修订为 `~3/n_valid` 是更准确的描述；图说 / 结果文档保持当前 0.30 baseline，并加 direction-preserving 注。

## 5. 与 PR-6 主线结论的对照

| PR-6 既有判读 | Step 6 是否冲击 | 理由 |
|---|---|---|
| H1 cohort NULL（pooled n=21 Wilcoxon p=0.42） | **不冲击** | endpoint 集合在 hold-out 下没有塌（recall median 0.83 above null floor）。**Step 6 仅证明 endpoint instability 不足以解释 H1 NULL；H1 NULL 在 within-recording held-out endpoint geometry 下保持稳健**。**不**证明 SOZ 标签 / clinical annotation / lagPat 通道选择没问题——这些维度本 Step 不覆盖。 |
| H2 forward/reverse swap 在 PR-2.5 reproduced 8 subject 的 sign-test p=0.031 | **不冲击** | strong tier 中 fwd/rev-reproduced subject 的 swap_class 100% concordant；H2 sign-test 仍是 mechanism sanity，本 Step 没改变 tier |
| §8 dual-tier swap_class（strict 10 / candidate 8 / none 17） | **不冲击** | label 在 strong tier 全 concordant；moderate tier 31% 反映 variable-k null 在 N/2 events 下的功率限制，与 §8 plan §8.7 channel-label 合同的"strict tier 才允许 channel-level label 且必须 split-half 验证"一致 |
| §9 swap × clinical SOZ set-relation NULL（strict ∩ informative n=5 sign p=0.500） | **不冲击** | endpoint 集合稳健 → §9 NULL 不是 endpoint-定义不稳定 artifact |

## 6. Caveats（写进 paper / 主文档时必须带）

1. **时间窗口 scope**：本 Step 只覆盖论文记录窗口内（Yuquan 24h / Epilepsiae 数日）。不能用作 H_drift 全盘否定；week-month timescale 的 drift 在本 Step 视野外。
2. **Power floor**：N/2 events 下 swap_class 是最噪的量；endpoint position recall 与 template Spearman 比 swap_class 更稳。后续若有 cohort claim 需求，应优先以这两个量为主轴。
3. **Day/night confound**：本 Step run 未使用 `--day-night-balanced` 切分模式。pilot 三个 subject 的两半 day/night 比例差均 < 20%，未触发 plan §6.5 sensitivity。Cohort 上未逐 subject 检查；若个别 weak subject 后续做 case-series follow-up，应先跑 day-night-balanced 重做。
4. **不能升级既有 tier**：本 Step 不改变 PR-6 既有 H1 / H2 / H3 / §8 / §9 的 tier 划分；只增加 robustness 一层。
5. **Channel selection 圆环 caveat**：与 PR-6 §9 同构 —— 本 Step 量化的是 lagPat / high-HI 通道集内部的稳定性，不是全脑信号。

## 7. Reproduction

```bash
# 12 unit tests (TDD coverage)
python -m pytest tests/test_held_out_template.py -v

# Pilot 3 sentinel subjects
python scripts/run_pr6_step6.py --pilot

# Cohort run (n=35 stable_k=2 union)
python scripts/run_pr6_step6.py --cohort

# Plan §6.5 day/night-balanced sensitivity (sibling output dir suffixed
# `_day_night_balanced`)：每段内 day events 与 night events 各自取 50%
# 重组成两半，控制 day/night 不平衡 confound
python scripts/run_pr6_step6.py --cohort --day-night-balanced

# Figures (4 panels + Chinese README)
python scripts/plot_pr6_step6.py
```

输出：
- `results/interictal_propagation/pr6_step6_held_out_template/per_subject/<stem>.json`
- `results/interictal_propagation/pr6_step6_held_out_template/cohort_summary.json`
- `results/interictal_propagation/pr6_step6_held_out_template/pilot_3subjects.json`
- `results/interictal_propagation/pr6_step6_held_out_template/figures/{tier_distribution_bar,template_spearman_recall_box,endpoint_position_recall_scatter,swap_class_transitions}.{png,pdf}`
- `results/interictal_propagation/pr6_step6_held_out_template/figures/README.md`

代码 entry：
- `src/interictal_propagation.py::_split_events_by_time` + `compute_held_out_endpoint_validation`
- `scripts/run_pr6_step6.py`
- `scripts/plot_pr6_step6.py`

## 8. 后续

- **Topic 4 SBA preflight (PR-6-sup1)** 是独立轨道，**不**与本 Step 6 共享 helper / output / 图。等 user 审阅 sup1 plan 后再起 TDD。
- **plan §15 next-step 第 1 / 3 / 4 / 5 项**（onset Jaccard、PR-4B HC × endpoint 联动、ictal propagation 直接对比、pre-ictal vs baseline endpoint）仍在 PR-6 §15 backlog，本 Step 不消耗这些项的预算。
