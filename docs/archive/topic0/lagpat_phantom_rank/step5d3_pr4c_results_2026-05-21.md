# Step 5d.3 — PR-4C 发作邻近 修过版重跑结果（2026-05-21）

> 状态：Step 5d.3 完成。
> 主入口：`docs/topic0_methodology_audits.md`
> 路线图：`./rerun_roadmap_2026-05-20.md`
> 输出：`results/interictal_propagation_masked/pr4c_seizure_proximity.json`（40 subjects, 30 with usable seizure windows）
> run log：`logs/step5d3_pr4c_masked.log` + `logs/step5d3_pr4c_masked_resume.log`

---

## 1. 三段式朴素话

**测了什么** —— 把每个发作前后划成"基线 / 发作前 / 发作后"三段时间窗，看"传播模式 5 个指标"（簇内 raw τ / centered τ / L3 lag span / L3 Pearson r / dominant cluster fraction）在三段间是否系统不同。如果发作真的"重置"传播模式，应该看到 post 段和 baseline 段在某些指标上有 cohort-level 配对差异。

**怎么测的** —— 30 个有 usable seizure window 的 subject（10 个 yuquan 没标 seizure_times 被跳过），每个 subject 算 pre/post 与 baseline 的 paired delta，cohort Wilcoxon。两个配置：主 (4/1/1 h) + 辅 (2/0.5/1 h)。修过版用 5a 的 masked cluster labels。

**揭示了什么** —— **5/5 指标全 NULL，全部 3 个比较（pre vs base / post vs base / post vs pre）方向上都没有 cohort-level 信号**。与原版"5 indicators all NULL in main + auxiliary configs" 历史封板**完全同向**。**没有翻转**。

代号补注：PR-4C, `compute_seizure_proximity_coupling`, SEIZURE_PROXIMITY_CONFIGS["main"], `n_seizures_usable`, archive `interictal_group_event_internal_propagation.md` PR-4C 节。

---

## 2. Cohort 数字表

n = 30 usable subjects (有 seizure_times + 至少 1 个 usable seizure window)。

### 2.1 pre_vs_baseline (n=30)

| metric | n | n_pos | median Δ | Wilcoxon p |
|---|---:|---:|---:|---:|
| raw_tau | 30 | 18 | +0.0069 | 0.477 |
| centered_tau | 30 | 17 | +0.0006 | 0.627 |
| L3 lag_span | 30 | 12 | -0.0003 | 0.237 |
| L3 pearson_r | 28 | 13 | -0.0067 | 0.920 |
| dom_cluster_frac | 30 | 14 | -0.0031 | 0.543 |

### 2.2 post_vs_baseline (n=29-30)

| metric | n | n_pos | median Δ | Wilcoxon p |
|---|---:|---:|---:|---:|
| raw_tau | 29 | 14 | +0.0000 | 0.509 |
| centered_tau | 29 | 13 | +0.0000 | 0.810 |
| L3 lag_span | 29 | 13 | -0.0006 | 0.442 |
| L3 pearson_r | 28 | 20 | +0.0153 | 0.138 |
| dom_cluster_frac | 29 | 16 | +0.0168 | 0.670 |

### 2.3 post_vs_pre (n=27-29)

| metric | n | n_pos | median Δ | Wilcoxon p |
|---|---:|---:|---:|---:|
| raw_tau | 29 | 15 | +0.0026 | 0.733 |
| centered_tau | 29 | 12 | -0.0000 | 0.407 |
| L3 lag_span | 29 | 17 | +0.0016 | 0.183 |
| L3 pearson_r | 27 | 14 | +0.0016 | 0.611 |
| dom_cluster_frac | 29 | 18 | +0.0168 | 0.316 |

### 2.4 与原版对照

| metric | 原版历史 | 修过版 (n=30) | 方向 |
|---|---|---|---|
| propagation pattern 5 指标 cohort Wilcoxon (all windows) | NULL | NULL | **同向** |
| rate_by_template dominant_template_rate (post vs baseline) | p=0.0009 (主) / p=0.0067 (辅) | （由 PR-5-B 5e 重做） | 待 5e |

PR-4C 的 "rate_by_template" 信号属 PR-5-B 范畴（绝对事件率招募 shift），由 5e 重新评估。

---

## 3. 受影响 PR 状态更新

| 下游 | 5d.3 状态 |
|---|---|
| PR-4C 主结论"5 指标 cohort 全 NULL" | **重跑完，方向一致 = 仍 NULL** |
| PR-4C 历史封板（archive `pr4c_seizure_proximity_review_2026-04-17.md` §9） | 不需修订 |
| rate_by_template post>baseline | 由 5e PR-5-B 重新评估，本文档不下结论 |

---

## 4. Bug fix during this step

`_run_pr4c` 和 `_run_pr5_*` runner 在 `--subjects` filter + cohort summary 写入的语义上有一个边界 case：当 `--subjects` 只跑部分 subjects 时，per-subject summary 会与既有合并（merged.update），但 PR-specific cohort artifact (`pr4c_seizure_proximity.json`) 只含本次跑的 subjects——需要 `scripts/consolidate_pr_cohort_masked.py --pr pr4c` 后置 consolidator 重建全 cohort artifact。本 step 的最终 artifact 是 consolidator 输出，含 40 subjects。

---

## 5. 输出位置

- `results/interictal_propagation_masked/pr4c_seizure_proximity.json` — 40 subjects (consolidator rebuild)
- `results/interictal_propagation_masked/per_subject/<sid>.json` 顶层 `seizure_proximity_coupling` 字段（40/40 subjects）
- run logs: `logs/step5d3_pr4c_masked.log` (main, 34 subjects, 09:36 killed at boundary) + `logs/step5d3_pr4c_masked_resume.log` (6 subjects, 11:34 done)
