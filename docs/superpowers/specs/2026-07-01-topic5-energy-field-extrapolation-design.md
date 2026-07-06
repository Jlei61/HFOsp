# Topic 5 间期顺序场 → 发作早期【能量】空间外推 — design spec

- 日期：2026-07-01
- 状态：**Phase-2 cohort 已执行（2026-07-01）；Task5 补跑后 = 16-subject cohort**（E590/E1084/E1146 于 2026-07-01 补上游 broad propagation 纳入，见 `build_broad_lagpat_patch_epilepsiae.py`；其中 590/1146 nhid=4 低功率）。结论不变：延伸成立（9/16 过 channel null）、场必要性不立（F_core 不赢 C1）、1-3 vs 2-3 打平。runner=`scripts/run_topic5_energy_field_cohort.py --compute --n-null 2000`；输出 `results/topic5_ictal_recruitment/field_extrapolation/{energy_field_extrapolation_FINAL.{json,md}, cohort_per_subject/*.json, figures/cohort_energy_F_core_vs_baselines_{bb,hfa}.png}`。**结论=延伸成立、场必要性不立（F_core_only 从未显著赢过 C1）**，详见 archive。
- 前身：order-问题（间期顺序场预测发作**招募顺序**）已 CLOSED-NEGATIVE，见 `docs/superpowers/specs/2026-06-30-topic5-interictal-field-broad-extrapolation-design.md`。**本 spec 只装能量这一个问题。**
- 关联：pilot 结果 `docs/archive/topic5/field_extrapolation_pilot_2026-06-30.md`；plan `docs/superpowers/plans/2026-07-01-topic5-energy-field-extrapolation.md`。

---

## 0. 摘要（朴素话）

审稿人质疑：现有"间期传播方向↔发作早期一致"只覆盖间期有明显 HFO 群体事件的电极、信息增益不多。

**这是一个 pivot，不是修旧问题**：
- 旧问题（已 closed-negative）：间期传播**顺序**场能不能预测发作**招募顺序**？→ 不支持，因为发作招募顺序跨发作本就不稳。
- **新问题（本 spec）**：间期传播**顺序**场能不能预测**发作早期能量的空间分布**？把关注扩到那些间期"隐身"（放电太少被 narrow 速率阈值挡掉）但发作会被招募的电极。

能不能用空间场（带坐标）在这些电极上比"逐通道"预测得更好，是回答审稿人"场必要性"的关键。

**结论可写到什么份上（措辞天花板，review P1）**：最多写 **"间期传播 field 对部分 hidden/broad territory 的发作早期 activation energy 有外推能力"**。**不可**写"证明发作招募**顺序**延伸"（顺序问题已 negative）、**不可**写"发作早期特异"（加固时间负对照证其为持续骨架）。

（精度代号见正文。）

## 1. 测什么 / 怎么测

### 1.1 两侧量 + 操作（= 原 field_concordance 显著口径）
- **间期侧（predictor）**：间期传播**顺序**场 = broad 轴记录 `propagation_geometry_broad/.../{ds}_{t_a,t_b}.json` 的 `typical_rank`（低=早=源），support 加权 kernel 回归。**A/B 两间期模板都建场、per-seizure 取 max(A,B)**（review：A/B 无 a-priori 优劣，"用间期最好的"，镜像原 axis_alignment `max_ab`）。**selection 由同口径 max null 控制**（null 也 max(A,B)）。t_b 缺失则退化单模板。
- **发作侧（target）**：发作早期**整体能量**——发作后 [0,10]s 的 **broadband 1–45Hz 能量** `bb_auc`（baseline-robust-z；t0 cache）= **primary**；**HFA 60–100Hz `hfa_auc` = sensitivity**（tier 同原 field_concordance）。**不是 z-ER 招募顺序**。
- **聚合**：每发作算一次 |相关| → **对发作取中位数**（稳健，不需顺序稳定）。

### 1.2 新 territory = broad ∖ narrow
narrow 池 = 速率阈值卡的高发放核心；broad 池 = `pick_k=-2.0`+`top_n∈{20,40}`。**broad∖narrow** = 有间期事件但放电少被挡的"隐身"电极（精确字符串差集）= 本检验靶。

### 1.3 F（场，两版）+ 两条逐通道基线 C1/C2
- **F_core_only（contract 版，承重）** = 间期顺序场**只用 narrow/core 通道建**，评估到隐身电极位置 → 预测顺序，每发作对发作能量 |spearman|→中位数。**这才是"核心场预测隐身"的字面合同**：隐身电极完全不进场，预测纯来自核心。
- **F（broad-LOO 版，secondary/exploratory）** = 场用全 broad 通道、LOO 只排目标电极。**注意（review P1）：其它 hidden 电极仍参与平滑，且 hidden support 不一定低**，故 F 的"核心主导"是**假设不是合同**——若 F 赢基线但 F_core_only 不赢，说明优势部分来自 hidden 互借（如 583 实测 F_core_only 0.536 < C1 0.613）。**主张以 F_core_only 为准**，F 作覆盖更广的探索量并列报。
- 两版都 A/B max（§1.1）。
- **C1** = 隐身电极自身 broad 间期**顺序** `typical_rank` → 发作能量，per-seizure |spearman|→中位数（逐通道顺序基线）。
- **C2** = 隐身电极自身间期**能量 fingerprint**（baseline 活跃度 `bact`）→ 发作能量，per-seizure |spearman|→中位数。
  - **语义精确（review P1）**：C2 是"通道自身**能量空间指纹**能多大程度预测发作能量指纹"，**不是**单纯"活跃通道恒活跃"——用 |corr| 把**反相关也算成强基线**（保守，让 C2 更难被超过）。场赢 C2 = 传播几何带来的预测力超过"用通道自身能量指纹就能解释"。对应 axis_alignment 的 anchor/activity-matched 控制族。

**场的独特优势（卖点）**：间期侧是顺序/几何、发作侧是能量，两者不同量、不同电极覆盖，**只能靠空间场当连续图案来比**；且隐身电极自身间期信号太稀疏，场能用核心邻居外推。逐通道做不到 → F 赢 C1+C2 即此优势的实证。

## 2. 预注册判读 + margin（review P1，**Phase-2 前锁死**）

**per-subject screen（粗筛，margin 锁死 = δ_FC=0.03）**：一个被试算"场有独特优势"当且仅当
**F 过 null（p<α=0.05）AND (F−C1) > δ_FC AND (F−C2) > δ_FC，δ_FC=0.03**。
- δ_FC 必须在任何 cohort 跑之前锁；不锁则 3/13↔4/13 会漂（E583 在 0.025 处正好卡边 → 0.03 阈下不算、任意正 margin 下算）。锁 0.03 = "场要比逐通道**有意义地**高，不是差 0.001 也算"。
- screen 只用于挑展示被试/计数，**不是最终统计**。

**cohort 独立单位 = subject（review P1，防伪重复）**：**绝不把不同被试的 seizure 池化当独立样本。** 两级：
1. **per-subject 内**（seizure 作 within-subject 重复）：每被试得到**一个**汇总数 —— 对"F 赢基线"类用 `median_sz(F_sz − Y_sz)`（Y∈{C1,C2,radius,shaft}）+ per-subject paired Wilcoxon(F_sz, Y_sz) p；对"F 过 null"类用 per-subject permutation p。
2. **cohort（subject=单位）**：把每被试那一个汇总数做 across-subject 检验（sign / Wilcoxon / binomial），**N=被试数**。

**FDR hypothesis 表（review P1，每个 hypothesis 统计量 + p 来源不同，须显式）** —— 主张以 **F_core_only** 为准：

| Hypothesis | per-subject 统计量 | per-subject p 来源 | cohort 统计量(subject=单位) |
|---|---|---|---|
| H_channel | F_core_only median | permutation p（F vs channel-shuffle null） | n_pass(p<.05) binomial across subjects |
| H_withinshaft | F_core_only median | permutation p（vs within-shaft null） | binomial |
| H_anchor | F_core_only median | permutation p（vs anchor/activity-matched null） | binomial |
| H_C1 | median_sz(F_sz−C1_sz) | per-subject paired Wilcoxon(F_sz,C1_sz) | Wilcoxon/sign of per-subject median-diff across subjects |
| H_C2 | median_sz(F_sz−C2_sz) | per-subject paired Wilcoxon(F_sz,C2_sz) | across-subject |
| H_radius | median_sz(F_sz−radius_sz) | per-subject paired Wilcoxon | across-subject |
| H_pca1_geometry | median_sz(F_sz−pca1_sz) | per-subject paired Wilcoxon | across-subject |

**BH-FDR over {bb_auc, hfa_auc} × 上 7 hypothesis = 14 个 cohort p**。screen margin（δ_FC=0.03）只作 per-subject 展示计数，cohort claim 走上表。

## 3. Null 与基线（review P1：正式 claim 前**必须全做**）

pilot 只有 channel-shuffle null。**说"场独特优势"前必须补全**：
1. **channel-shuffle null**（已）。
2. **within-shaft null**（杆内 shuffle 发作能量）。
3. **anchor/activity-matched null**（按 `bact` 活跃度分箱内 shuffle）——与 C2 同源、控制活跃度。
4. **radius baseline**：沿轴距离 → 发作能量（场赢之 = 是方向不是"近的先亮"）。
5. **pca1_geometry baseline**（实现名；**非** shaft-direction）：沿隐身电极坐标**主 linear 轴**投影 → 发作能量（只在 pca_ratio≥0.05 即 ≥2D 时算）。场赢之 = 超过"沿主采样方向的位置"。**⚠️ review 修正：这不是"单杆方向 / ≥2 非平行杆"基线，不能声称排除了电极杆方向伪影**（真·per-shaft-direction baseline 未实现，multi-shaft hidden 集下该基线定义不清）。
6. **C1/C2 paired subject-level 统计 + BH-FDR**（见 §2）。

未补全前，结论只能写 **"pilot 正信号"**，不能写 cohort 级"场独特优势成立"。

## 4. Pilot 结果（已跑，描述性，非 cohort）

13 个有 broad 几何+t0 cache 的被试，channel-shuffle null B=2000：
- **bb_auc**：F 过 null **6/13**；screen（F−C1>0.03 且 F−C2>0.03）**3/13**（E253/E916/E620），若放宽到任意正 margin = 4/13（含 E583）→ **正是 §2 要锁 margin 的原因**。E253 最干净（F 0.745 vs C2 0.234）。
- **hfa_auc**：F 过 null **8/13**；场赢 C1（>0.03）7/13；但点亮被试与 bb 不同（E583 在 HFA 场优势 +0.266、E253 在 HFA 边缘 p=0.055）→ **band 依赖、异质**。HFA 的 C2 待 cohort 补。
- 与 narrow field_concordance 是否显著无干净对应（最强 E253/E620 narrow 不显著；narrow 显著 E922 不延伸）。

判读：**间期顺序场对部分隐身 territory 的发作早期能量有外推能力，且在 E253/E620 等被试上明显超过逐通道顺序+能量基线**；但约半数不显著、多数 F≈基线、band/被试异质、null 未补全 → **pilot 正信号，非确证**。

## 5. 数据源与复用

- 间期顺序场：`propagation_geometry_broad/.../{ds}_t_a.json`（含 x_norm/y_norm/typical_rank/support/coord_mm）。
- 发作能量：`results/topic5_ictal_recruitment/t0_feature_cache/{ds}.npz`（`bb_auc__/hfa_auc__/bact__{sz}`，全 montage 覆盖隐身电极）。
- broad/narrow 池：`interictal_propagation_masked{_broad,}/per_subject/{ds}.json::channel_names`。
- 模块 `src/topic5_field_extrapolation.py`（已实现：`ictal_bb_auc_by_seizure`/`compute_f_c_activation`/`null_F_activation`/`compute_c2_perchannel_energy`/`ictal_paired_features`；12 TDD）。runner `--ictal activation --activation {bb_auc,hfa_auc}`。

## 6. Pilot 选人 / cohort 扩展
- pilot：有 broad 几何 + t0 cache 的 13 个（已跑）。
- cohort 扩展：补 narrow field_concordance 显著但缺 broad 几何的 **E590/E1084/E1146** 产 broad 几何后纳入（关键样本）。

## 7. 不主张什么（措辞纪律）
- **不写"证明发作招募顺序延伸"**（顺序问题已 negative）。
- **不写"发作早期特异"**（持续骨架）。
- 区分两层主张：①**延伸**（F 过 null=场预测到隐身电极发作能量）vs ②**场独特优势**（F 赢 C1+C2）。后者承重更高，须 §2 margin + §3 全 null。
- pilot 描述性、不下 cohort 结论。
