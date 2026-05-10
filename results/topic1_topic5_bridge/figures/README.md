# Topic 1 × Topic 5 Bridge — figures README

> Q1 + Q1b + Q3 cohort 验证图组（spec: `docs/superpowers/specs/2026-05-10-topic1-topic5-bridge-design.md`）

### q1_cohort_count_x_window.png

3 个 pre-ictal 窗口 (`[-15,-1]` / `[-30,-1]` / `[-60,-1]` min) 在 cohort 上各自的 subject-positive 计数 / denominator + binomial p-value。每柱顶 `count/denom` + `p=`；标题为 cohort 3-state verdict。
**关注点**：是否 ≥2/3 窗口 PER-WINDOW-PASS（决定 COHORT-EXPLORATORY-PASS）；某个窗口的 denom < 10 表示该 window 上有 subject 因 n_eligible_seizures < 4 退出。

### q1_effect_distribution.png

3 子图（每窗一个）显示 cohort 10 个 subject 的 feature-winner |effect|（ε² / |r| / Cramér V，per-subject 取最大者）。虚线 = `effect_min=0.10`。
**关注点**：cohort 整体 effect 是否大量集中在 dual gate effect 阈值之上；个别极高 effect 但低 p 的 subject 是否被 dual gate 排除（与单 p-value gate 对比检查）。

### q1_per_subject_strip.png

每 subject 一行的 frac_T0 strip（cohort 10 + 442 + 1084 broad sensitivity = 12 行）；点颜色按 ictal subtype label。442 的 sz=9 outlier 用错位高亮。
**关注点**：subject 内 subtype 间 frac_T0 是否目视分层；442 outlier 是否独立离群。

### q1b_442_sentinel.png

442 sz=9（outlier，subtype=-1）vs 其他 16 sz（main，subtype=0）在三个窗口上的 frac_T0、switch_rate effect bar；标题带 frac_T0 p-value。
**关注点**：descriptive case study，**不是** cohort claim。442 在 topic1 几何弱（swap=none, silhouette=0.255）的前提下，看 sz=9 是否有任何独立的 pre-ictal 指纹差异。

### q1_stratified_swap_silhouette.png

Q3 stratifier 散点：x 轴 = silhouette 二分（≤0.5 vs >0.5），y 轴 = primary window 的 effect_winner，颜色 = swap_class（real = strict ∪ candidate vs none）。每点标 subject id。
**关注点**：Q1 信号是否集中在 swap=real 与 silhouette=high 子集；descriptive only，不开 α。

---

## Q1' (PIVOT 2026-05-10) figures

### q1prime_per_subject_scatter.png
6-panel (2×3) per-subject (ρ_a, ρ_b) 散点；点颜色按 topic5 z-ER ictal subtype。
**关注点**：strict subject (1073/1146/635/958) 上 subtype-cluster 是否在 (ρ_a, ρ_b) 平面上分离；reverse-line 上下分布暗示 T0 vs T1 主导。

### q1prime_cohort_effect.png
6 subject 的 Cramér V + AMI bar；虚线 V_min=0.30 阈值；标题为 cohort verdict。
**关注点**：strict 4 subject 的 V/AMI 是否集中在阈值之上；548 candidate 的位置；442 (axis collapse) 应近 0。

### q1prime_assignment_x_subtype.png
strict + candidate (5 subject) 的 assignment × subtype 列联 stacked bar；标题带 Fisher/χ² p。
**关注点**：assignment {T0, T1} 是否在 subtype 间分布不均；p < 0.05 + V > 0.3 双 gate 通过的 subject。
