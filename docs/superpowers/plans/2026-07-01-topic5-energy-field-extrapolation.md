# Topic 5 间期顺序场 → 发作早期【能量】空间外推 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development 或 executing-plans。Steps 用 checkbox。

**Goal:** 检验"间期传播**顺序**场能否预测间期隐身电极（broad∖narrow）发作早期**能量**空间分布"，并证明空间场比逐通道（顺序 C1 + 能量指纹 C2）更优（回应审稿人"场必要性"）。

**Architecture:** 复用 `src/propagation_contact_plane_readout.py` 场引擎 + `propagation_geometry_broad` 间期顺序场 + t0 cache 发作能量（bb_auc/hfa_auc/bact）。模块 `src/topic5_field_extrapolation.py` 已实现核心；本 plan 主要补 Phase-2 cohort 的 null/基线/统计。

**Spec:** `docs/superpowers/specs/2026-07-01-topic5-energy-field-extrapolation-design.md`

## Global Constraints

- 发作侧 = 能量（bb_auc primary / hfa_auc sensitivity），**非** z-ER 招募顺序（order 问题已 closed-negative）。
- 每发作 |spearman| → 对发作取中位数。
- **间期侧 A/B 两模板 max(A,B)**（无优劣，镜像 max_ab；null 也 max）。
- **主张量 = F_core_only（场只用 narrow core 建，隐身电极不进场）**；F（broad-LOO）作探索量并列，**因 LOO 只排目标、其它 hidden 仍参与平滑 + hidden support 不一定低 → "核心主导"是假设非合同**（583 实测 F_core_only 0.536 < C1 0.613 = 优势部分来自 hidden 互借）。
- **cohort 独立单位 = subject**（绝不池化 seizure）：per-subject 先汇总一个数，再 across-subject 检验，N=被试数。
- **margin 锁死 δ_FC=0.03**（per-subject screen 展示用）；cohort claim 走 FDR hypothesis 表（spec §2），不靠单一 margin。
- C2 = 逐通道能量 fingerprint 基线（|corr(bact, ictal_energy)|，abs 含反相关，保守）；**不写"活跃通道恒活跃"**。
- 措辞：结论最多 = "间期 field 对部分 hidden territory 发作早期 activation energy 有外推能力"；不写"招募顺序延伸"/"发作早期特异"。
- 间期 broad rank 已 phantom-masked，直接用。坐标 mm。broad∖narrow = 精确字符串差集。
- 产物 `results/topic5_ictal_recruitment/field_extrapolation/`；图目录配 `figures/README.md`。**不提交**（待用户定干净 Topic5 base）。

---

## Phase 1（DONE 2026-06-30/07-01）— 能量 pilot + C2 + A/B max + F_core_only

已实现并通过 14 TDD（`tests/test_topic5_field_extrapolation.py`）：
- `ictal_bb_auc_by_seizure` / `ictal_paired_features` — 读 t0 cache per-seizure 能量。
- `load_broad_axis_record(ds_sid, template="t_a"|"t_b")` — 两间期模板。
- `compute_f_c_activation(..., record_b=, core_names=)` — F（**A/B max**；core_names 给定=**F_core_only**）+ C1，per-seizure |corr|→中位数；`null_F_activation` 同口径 max-null + core_names。
- `compute_c2_perchannel_energy` — C2（能量 fingerprint bact）。
- runner `--ictal activation --activation {bb_auc,hfa_auc}` 输出 F / F_core_only / C1 / C2 / screen(δ=0.03)。

**⚠️ 早期 pilot 数字（单模板 A、无 core_only、margin 未锁）已废**：A/B max + F_core_only 重跑中。早期 583 单模板 F=0.544≈C1 screen 边缘；**A/B max 后 F=0.673、C1=0.613，但 F_core_only=0.536 < C1** → 暴露 broad-LOO 的 hidden 互借（= 为何主张走 F_core_only）。最终 pilot 数字以重跑 + spec §2 cohort 表为准。

---

## Phase 2（cohort 正式版 — **已执行 2026-07-01**）

> **状态**：Task 1–5 已实现并跑（runner `scripts/run_topic5_energy_field_cohort.py --compute --n-null 2000`，B=2000，channel/within_shaft/anchor null + radius/pca1_geometry 基线 + subject-unit 两级聚合 + 14-hypothesis BH-FDR）。**FINAL = 16-subject broad-geometry cohort**（Task 5 于 2026-07-01 补 E590/E1084/E1146 上游 broad propagation 后纳入；E590/E1146 nhid=4 低功率已标注）。**Task 6 部分**（cohort 图 + README done；主文档 §3.x 待用户定）。下方 Task 描述为实现记录，实际实现细节以 runner 代码 + spec §2/§3 为准（如 shaft-direction 基线实际实现为 pca1_geometry，见 Task 2）。

### Task 1: anchor/activity-matched null + within-shaft null（能量版）

**Files:** Modify `src/topic5_field_extrapolation.py`, `tests/test_topic5_field_extrapolation.py`

**Produces:** `null_F_activation(..., null_kind="channel"|"within_shaft"|"anchor")` —
- `within_shaft`：每发作在每根杆内 shuffle 发作能量。
- `anchor`：每发作按该发作 `bact` 的分位箱（n_bins=4）内 shuffle 发作能量。
需 `shafts`（从 record `channels[i]["shaft"]`）+ per-seizure `bact`（`ictal_paired_features` 取 `bact` 作 anchor 向量）。

- [ ] **Step 1: 失败测试**

```python
def test_null_within_shaft_and_anchor_run():
    rec = _rec_line_big()
    for c in rec["channels"]:
        c["shaft"] = c["name"][0]   # 同首字母同杆 (K*/H* 两杆)
    cache = ["H1","H2","H3","H4"]
    sz = [np.array([1.,2.,3.,4.]), np.array([1.5,2.,3.,3.5]), np.array([1.,2.5,2.8,4.])]
    bact = [np.array([1.,1.,1.,1.])*i for i in (1,2,3)]  # 占位 anchor
    nd_ws = null_F_activation(rec, ["H1","H2","H3","H4"], cache, sz, n=100, seed=1,
                              sigma_xy=0.3, null_kind="within_shaft", shafts={"H1":"H","H2":"H","H3":"H","H4":"H"})
    nd_an = null_F_activation(rec, ["H1","H2","H3","H4"], cache, sz, n=100, seed=1,
                              sigma_xy=0.3, null_kind="anchor", anchor_by_sz=bact)
    assert 0 <= nd_ws["p_value"] <= 1 and 0 <= nd_an["p_value"] <= 1
```

- [ ] **Step 2-4:** 实现 `null_kind` 分支（within_shaft：按 shaft 分组 permute 索引；anchor：按 bact 分位箱分组 permute）；跑测试过。

### Task 2: radius + pca1_geometry 基线（能量版）— **实现为 pca1_geometry，非 shaft-direction**

**实际实现（cohort runner `evaluate_subject`）**：radius = per-seizure |corr(沿轴距离, 发作能量)|→中位数；**pca1_geometry** = 沿隐身电极坐标**主 linear 轴**（PCA1）投影 → 同口径，只在 pca_ratio≥0.05（≥2D）算。**⚠️ 原计划写"shaft-direction / ≥2 非平行杆"未实现**——multi-shaft hidden 集下 per-shaft 方向基线定义不清；实际是 pca1_geometry（主采样方向）。**故不可声称"排除电极杆方向伪影"**。

- [ ] 失败测试（radius 用 _rec_line_big，沿 x 单调 → 与位置相关）；实现；过。

### Task 3: per-subject runner v2（全 null + 全基线 + screen margin）

**Files:** Modify `scripts/run_topic5_field_extrapolation_pilot.py`

**实际实现 = 新 runner `run_topic5_energy_field_cohort.py::evaluate_subject`**（非改 pilot runner）：per-subject 输出 `null_{channel,within_shaft,anchor}_p`、`radius`、`pca1_geometry`、`C1`、`C2`、`F_core_only`、`F_loo` + 逐发作 series + 两个 screen：`screen_channel_c1c2_only`（仅 channel null + C1/C2 margin，名字如实）与 `screen_strict`（三层 null 全过 ∧ 赢 C1/C2/radius/pca1 全 margin，才配叫 field-advantage）。两频段各跑。

- [ ] 跑 bb_auc + hfa_auc 全 13 被试 + 补的 E590/E1084/E1146（先产 broad 几何，见 Task 5）。

### Task 4: cohort 统计（subject=单位 + 显式 FDR hypothesis 表）

**Files:** Create `scripts/aggregate_topic5_energy_field_extrapolation.py`

**严格按 spec §2**：①cohort 独立单位 = **subject**（绝不池化 seizure）；②主张量 = **F_core_only**；③7 个 hypothesis 各自统计量+p 来源不同（H_channel/within_shaft/anchor = per-subject permutation p → binomial across subjects；H_C1/C2/radius/pca1_geometry = per-subject paired Wilcoxon(F_sz,Y_sz) → median-diff across-subject Wilcoxon）；④**BH-FDR over {bb,hfa}×7 = 14 cohort p**。输出 `energy_field_extrapolation_FINAL.{json,md}`（头锁 δ_FC=0.03、α=0.05、unit=subject、primary=F_core_only、cohort=13-subject subset）。

- [ ] 实现 cohort 两级聚合 + 14-hypothesis BH-FDR 表 + 跑。

### Task 5: 补 broad 几何（E590/E1084/E1146）— **DONE（2026-07-01）**

三者有 broad lagPat npz 但缺上游 broad propagation 产物。**已补**：写 `scripts/build_broad_lagpat_patch_epilepsiae.py`（复刻既有 13 的 build_broad_lagpat_patch 口径：dataset=epilepsiae、`lagpat_broad_epilepsiae`、top_n=20、`--masked-features`、平行 `_broad` dir、monkeypatch `_subject_dir`+`_has_propagation_inputs` 路由）跑 step1（propagation+pr25）+ step3（rank_displacement）；再 monkeypatch `run_contact_plane_readout._subject_dir` + `--rankdisp-dir` broad + `--out propagation_geometry_broad/observation_readout/real_subjects` 产几何（t_a+t_b，**n_channels=20 对齐既有口径**）；再能量 cohort（PYTHONHASHSEED=0）→ cohort_per_subject。**纳入后 16-subject，结论不变（1-3 vs 2-3 打平）**。⚠️ 590/1146 nhid=4（narrow 池本大、broad 扩得少）=低功率、基本 tie；1084 nhid=9。

- [x] 补 3 被试 → 16-subject cohort（tally + FINAL + 32 图已重出）。

### Task 6: paper figure + README + 主文档

- [ ] bb + hfa 两版 cohort F-vs-C1-vs-C2 散点 + E253/E620 场热图+隐身电极能量诊断。
- [ ] `figures/README.md`（中文逐图 + 关注点）。
- [ ] 进 `docs/topic5_seizure_subtyping.md` §3.x（措辞守 spec §7：写"对部分 hidden territory 发作能量有外推能力"，不写"顺序延伸"/"发作早期特异"；区分"延伸"vs"场赢逐通道"两层）。

## Self-Review 注记
- Spec 覆盖：F/C1/C2（Phase1）、margin 锁（Global+Task3）、全 null（Task1）、半径/杆向基线（Task2）、paired+FDR（Task4）、補几何（Task5）、措辞（Task6）。
- 一个 plan 一个问题（能量），order 问题在 closed-negative 旧 plan。
