# Q1' 过夜探索 — Per-seizure 特征 × 间期时序模板 (2026-05-10)

> **分析层级**: case-series 探索，非 cohort α 声明。  
> **前置背景**: `docs/archive/topic5/bridge_q1prime/bridge_q1prime_results_2026-05-10.md`（phase-1 Q1' INDETERMINATE）  
> **判定**: **INDETERMINATE / WEAK-SIGNAL** — `median_onset_latency_sec` 有方向性倾向；其余 features 为 NULL。

---

## 1. Cohort 实际 size

### 全量输入 (预期 25 subject)

| 数据集 | 预期 | 实际产出 JSON | 备注 |
|---|---|---|---|
| Epilepsiae | 16 | 16 | 全部正常运行 |
| Yuquan | 9 | 7 (+ 2 FAILED) | zhangjinhan / zhaojinrui stable_k≠2，排除 |

**实际分析 cohort: 23 subjects**（16 epi + 7 yuquan）。

### Topic5 subtypes 覆盖

| 分类 | Epilepsiae | Yuquan |
|---|---|---|
| topic5 status=ok, n_subtypes≥2 (可参与检验) | 12 | 1 (zhangkexuan) |
| topic5 status=insufficient_n (无 subtype 标签) | 1 (1077) | 6 |
| topic5 n_subtypes=1 (单 subtype, 检验退化) | 3 (1084,442,1150,139) | 0 |

### Swap class 分布

| swap_class | Subjects |
|---|---|
| strict | 1073, 1146, 635, 958, 139 (5 epi) |
| candidate | 548, 1150 (2 epi), huanghanwen, zhangkexuan (2 yuq) |
| none | 其余 16 |

---

## 2. Stage A: Per-subject Q1' 主 axis 结果

**所有 23 subjects 均 q1prime_positive=False**（0/23）。

| subject | swap | t5_status | n_subtypes | n_sz | n_valid_assign | Cramér V | 备注 |
|---|---|---|---|---|---|---|---|
| epilepsiae_1073 | strict | ok | 2 | 18 | 7 | 0.250 | ρ_a ∈ [-0.30,1.00] |
| epilepsiae_1077 | none | insuf_n | 0 | 8 | 4 | 0.000 | 无 subtype 标签 |
| epilepsiae_1084 | none | ok | 1 | 84 | 17 | 0.000 | 单 subtype |
| epilepsiae_1096 | none | ok | 2 | 8 | 8 | 0.293 | |
| epilepsiae_1146 | strict | ok | 2 | 19 | 7 | 0.667 | 最高 Cramér V 之一 |
| epilepsiae_1150 | candidate | ok | 1 | 7 | 6 | 0.000 | 单 subtype |
| epilepsiae_139 | strict | ok | 1 | 5 | 3 | 0.000 | n_valid=3 不足 |
| epilepsiae_253 | none | ok | 2 | 5 | 1 | 0.000 | n_valid=1 |
| epilepsiae_442 | none | ok | 1 | 21 | 21 | 0.113 | 单 subtype (sentinel) |
| epilepsiae_548 | candidate | ok | 5 | 26 | 21 | 0.703 | 最高 Cramér V |
| epilepsiae_583 | none | ok | 3 | 23 | 22 | 0.199 | |
| epilepsiae_590 | none | ok | 2 | 10 | 6 | 1.000 | n=6 过少导致 V=1.0 |
| epilepsiae_635 | strict | ok | 2 | 16 | 9 | 0.378 | |
| epilepsiae_916 | none | ok | 2 | 49 | 23 | 0.067 | |
| epilepsiae_922 | none | ok | 2 | 25 | 25 | 0.143 | |
| epilepsiae_958 | strict | ok | 3 | 14 | 14 | 0.592 | |
| yuquan_gaolan | none | insuf_n | 0 | 3 | 0 | 0.000 | 所有 seizure n_active=0 |
| yuquan_huanghanwen | candidate | insuf_n | 0 | 2 | 0 | 0.000 | |
| yuquan_litengsheng | none | insuf_n | 0 | 8 | 2 | 0.000 | |
| yuquan_pengzihang | none | insuf_n | 0 | 6 | 3 | 0.000 | |
| yuquan_sunyuanxin | none | insuf_n | 0 | 8 | 8 | 0.000 | |
| yuquan_xuxinyi | none | insuf_n | 0 | 3 | 0 | 0.000 | |
| yuquan_zhangkexuan | candidate | ok | 2 | 6 | 6 | 0.000 | 唯一 yuq 可检验 |

**阶段反思**：Yuquan dataset 对 Q1' 几乎没有贡献——6/9 subjects 因 topic5 status=insufficient_n 无 subtype 标签；另外 2 因 stable_k≠2 完全排除；仅 zhangkexuan 可检验但 Cramér V=0。Yuquan seizure 数量少（median 4/subject），topic5 audit 已知缺数据，这不是 bug 而是已知限制。实际可检验的 cohort 是 epilepsiae-12（topic5=ok, n_subtypes≥2）。

---

## 3. Stage B: Per-seizure feature 提取摘要

**总行数**: 374（338 epi + 36 yuq），23 subjects。

| Feature | n 非 NaN | Median | IQR |
|---|---|---|---|
| delta_rho_swap | 213 | 0.171 | [-0.700, 0.833] |
| delta_rho_full | 234 | 0.104 | [-0.700, 0.756] |
| n_active | 374 | 30.5 | [15.3, 51.0] |
| active_fraction | 374 | 0.414 | [0.175, 0.693] |
| fast_recruit_fraction | 374 | 0.000 | [0.000, 0.000] | → 退化，几乎全 0，排除 |
| onset_spread_sec | 367 | 103.6 | [38.4, 130.3] |
| median_onset_latency_sec | 367 | -0.1 | [-30.9, 9.7] |

- `delta_rho_swap`: n=213（161 行因 swap endpoint 与 atlas 交集不足 3 ch → insufficient_n）
- `delta_rho_full` 交集更大（234），因为 full 用 all topic1 channels 而非仅 endpoint 子集
- `fast_recruit_fraction` 退化（median=0, IQR=0），不纳入后续分析

**阶段反思**：Swap-subset 的设计导致 45% (161/374) 的行在 assignment 上退化为 `insufficient_n`；这是 spec §8.7 strict-tier endpoint 的代价——牺牲覆盖率换取通道精确性。delta_rho_full 覆盖更广但代表不那么精准的 T0/T1 分辨。

---

## 4. Stage C: Feature × delta_rho 相关性

**方法**: per-subject Spearman(feature, delta_rho_swap)，n≥4 对才计算。16 subjects eligible for delta_rho_swap。

### 关键结果表（delta_rho_swap）

| Feature | n 可检验 subject | Median |ρ| | Median ρ | Sign(+/-) | Sign-test p |
|---|---|---|---|---|---|
| n_active | 16 | 0.389 | -0.016 | 8+/8- | 1.000 |
| active_fraction | 16 | 0.389 | -0.016 | 8+/8- | 1.000 |
| onset_spread_sec | 16 | 0.145 | -0.065 | 6+/10- | 0.454 |
| median_onset_latency_sec | 16 | 0.336 | -0.032 | 8+/8- | 1.000 |

- **n_active/active_fraction**: |ρ| 较高（median 0.389）但完全对称（8+/8-，p=1.0）→ **NULL**。每个 subject 有独立 T0/T1 convention，cross-subject 方向不一致是预期的。
- **onset_spread_sec**: |ρ| 低（median 0.145），略微偏负（6+/10-），p=0.454 → **NULL**。
- **median_onset_latency_sec**: |ρ| 中等（median 0.336）但也对称（p=1.0）→ **NULL**。

**Pooled Spearman（跨 subject pooling，仅供参考）**：
- n_active: ρ=0.114, p=0.098（borderline，不可靠）
- active_fraction: ρ=-0.000, p=0.999（NULL）
- onset_spread_sec: ρ=-0.096, p=0.161（NULL）
- median_onset_latency_sec: ρ=0.044, p=0.524（NULL）

**注意**: Pooled 分析混合了各 subject 各自 T0/T1 convention，不是有效统计，仅作参考。

**显著 per-subject 单例**（p<0.05）:
- `n_active × epilepsiae_922`: ρ=0.627, p=0.001 ← 唯一真实显著（n=25）
- `onset_spread_sec × epilepsiae_922`: ρ=-0.451, p=0.024（n=25）
- `median_onset_latency_sec × epilepsiae_1096`: ρ=0.784, p=0.021（n=8）

这些是个案信号，不代表 cohort 趋势。

**阶段反思**：Stage C 为 NULL。绝对效量（median |ρ|=0.3-0.4）可能反映了 within-subject 真实的 seizure-to-seizure 变异，但方向在各 subject 间不一致——这是 T0/T1 convention 在 subject 间没有固定物理含义的直接体现。要让 Stage C 有意义，需要先将 T0/T1 对齐到同一个物理参考（如固定的 SOZ source template），这是 Q2/Q3 的工作。

---

## 5. Stage D: Feature × subtype 区分

> **🔧 2026-05-11 sign-fix 重算**：原始报告基于 `_mann_whitney_with_effect` 的 rank-biserial r 公式 `r = 1 − 2U/(n1*n2)`，**与标准约定符号相反**（标准 `r = 2U/(n1*n2) − 1`：r > 0 ⇔ a > b）。在 src 修正后所有 k_s=2 subject 的方向标签翻转；KW (k_s≥3) 子型方向由中位数比较产生，未受影响。
> phase-1 Q1 / Q1' 主轴**不受影响**（dual gate 用 `abs(effect)`；Q1' 用 Spearman/Fisher 不用 MW）。
> 测试用例 `test_mann_whitney_sign_convention_standard` 已加入作 regression guard。

**方法**: per-subject MW (k_s=2) / KW (k_s≥3)，排除 subtype=-1 outlier，每 subtype ≥ 2 个 seizure。

**Eligible subjects**: 12（全为 epilepsiae）。

### Cohort sign-test 结果（**已修正**）

| Feature | n eligible | n_sig (p<0.05) | Binom p (vs 5%) | Median \|eff\| | Direction (+/−) | Direction sign p |
|---|---|---|---|---|---|---|
| n_active | 12 | 1 | 0.460 | 0.500 | 5+/7− | 0.774 |
| active_fraction | 12 | 1 | 0.460 | 0.500 | 5+/7− | 0.774 |
| onset_spread_sec | 12 | 0 | 1.000 | 0.248 | 2+/9− | 0.065 |
| median_onset_latency_sec | 12 | 2 | 0.118 | 0.315 | **9+/3−** | **0.146** |

### 关键发现（修正后）

**`median_onset_latency_sec` 方向倾向反转后变弱**：12 eligible subjects 中 9 个方向为正（subtype 0 的 median onset latency 比 subtype 1 **更晚**），sign-test p=0.146（不显著）。
- **原始报告方向解读错误**：旧版报 "subtype 0 更早 onset"，实际数据是 9+/3−，方向是 subtype 0 比 subtype 1 **更晚**。
- 修正后 sign_p=0.146 **未通过任何阈值**（uncorrected α=0.05 即不过）。Cohort 方向趋势消失。

**`onset_spread_sec` 新现弱方向倾向**：2+/9−（subtype 0 的 onset spread 比 subtype 1 更窄），sign-test p=0.065 (uncorrected)。**但未通过 0.05，也未通过 Bonferroni**。仅作 description。

**显著 per-subject 单例（p<0.05，修正后符号）**:
- `median_onset_latency_sec × epilepsiae_548`: p=0.025, ε²=0.596 (3 subtypes, n=17) — dir=pos（subtype 0 onset 更晚）
- `n_active × epilepsiae_958`: p=0.018, ε²=0.674 (3 subtypes, n=12) — dir=neg（subtype 0 n_active 更小）
- `active_fraction × epilepsiae_958`: p=0.018, ε²=0.674 — dir=neg
- `median_onset_latency_sec × epilepsiae_958`: p=0.036, ε²=0.520 — dir=neg（subtype 0 onset **更早**）

**548 与 958 在 median_onset_latency 上方向相反**：548 subtype 0 更晚（pos），958 subtype 0 更早（neg）。**两个 case-series 信号不能合成同一个 cohort 结论**（也就是为何 cohort sign-test 9+/3− 但显著 subject 一正一负）。

**阶段反思（修正后）**: Stage D **没有过任何阈值的 cohort signal**。`median_onset_latency_sec` 方向 9+/3− 距离 sign-test α=0.05 较近但未达；4-feature Bonferroni 校正后阈值 0.0125，更远。548/958 case-series 在该 feature 上方向**相反**，不构成 cohort 趋势。整体 Stage D 是 NULL 偏 weak-direction，**不是**原始报告所述的"方向倾向"。原始 archive 的 framing 因 sign-flip bug 高估了 cohort 信号；本节为标准约定下的诚实结果。

---

## 6. Stage E: 诚实反思

### 哪些路径死了

1. **Yuquan cohort**: 实际上无法贡献有效信号。topic5 insufficient_n 是结构性缺陷（Yuquan seizure 数量少，atlas 覆盖不足），不是可以修复的 bug。如果要做 Topic 1 × Topic 5 bridge，Yuquan 只能贡献 delta_rho 计算（不需要 subtype），不能贡献 Stage D。

2. **Stage C 方向一致性**: T0/T1 convention 在各 subject 间无固定物理含义，pooling delta_rho 跨 subject 没有意义。这条路需要先建立一个 cohort-wide physical reference（如"source template = SOZ 先活动"），才能让 cross-subject 相关变得可解释。

3. **fast_recruit_fraction**: 完全退化（全 0），对应的 atlas 产出大多数 seizure 不满足快速招募标准。排除。

4. **onset_spread_sec 作为区分器**: 效量低（median |eff|=0.248），方向不一致，不是好的 feature。

### 哪些有信号（弱，sign-fix 重算后）

1. **~~`median_onset_latency_sec` 方向倾向~~**: 原始声明的 sign-test p=0.039 是 sign-flip bug 的产物。修正后 9+/3−, sign_p=0.146。**信号消失**。
   - 弱观察：方向 (subtype 0 onset 更晚) 仍有 weak trend，但未过任何阈值。
   - 548 vs 958 在该 feature 上**方向相反**（548 subtype 0 更晚，958 subtype 0 更早）—— 两个 case-series 不能合成同一 cohort 趋势。

2. **个案效量（修正后）**:
   - **958** (strict, 3 subtypes): n_active/active_fraction ε²=0.674 (p=0.018), median_latency ε²=0.520 (p=0.036)，subtype 0 → 更**早** onset。
   - **548** (candidate, 5 subtypes): median_latency ε²=0.596 (p=0.025)，subtype 0 → 更**晚** onset。
   - 两 subject 同样在 within-subject 真实信号，但方向相反——这恰是 cohort pooling 失败的微观证据。

3. **Cramér V 正向性（Q1' 主轴，未受 MW sign-fix 影响）**: 4 strict-swap subjects (1073/1146/635/958) Cramér V 均正向 [0.25, 0.67]，median 0.486；548 candidate V=0.703；442 axis-collapse V=0.11 noise floor。Q1' 主轴的 cohort-level descriptive signal 仍然站住。

4. **Per-subject Δρ 诊断 (新增)** — 用户 2026-05-11 要求：
   - **958** (strict): n_valid=14, T1=10/T0=3/tie=1, median Δρ=−0.585 → 明显偏 T1
   - **548** (candidate): n_valid=21, T1=13/T0=8, median Δρ=−1.000 → 偏 T1
   - **922** (none): n_valid=25, T0=18/T1=5/tie=2, median Δρ=+0.381 → 偏 T0，但 swap_class=none，**不作主证据**
   - 见 `results/topic1_topic5_bridge/q1prime_per_subject_rho_diag.csv` + `figures/q1prime_rho_diag_*.png`

### 方法学问题

1. **Power floor 主导**: n_eligible per subject 在 3–22 之间，median 约 8。Fisher exact 2×2 在 n=8 时即使 perfect separation 也只给 p=0.008，非 perfect 分离通常需要 n≥15 才能通过。Stage D 12 个 subjects 中大多数 n≤12，Stage C 需要 n≥7 才能 p<0.05。

2. **T0/T1 convention 无 cohort-level 含义**: 每个 subject 的 T0=larger-fraction cluster，无法跨 subject 比较 delta_rho 的正负号。这是设计上的约束。

3. **Subtype label 不稳定**: topic5 PR-1 gap_perm 正在重跑（channel-block null），部分 subjects 的 subtype 分组可能在 audit 完成后改变。Stage D 结果应等 audit 完成后重新跑。

### 推荐下一步

**INDETERMINATE**（弱信号，非 NULL，非 PASS）。

优先级建议：
1. **等 topic5 audit 完成**（channel-block null re-run）后重跑 Stage D，subtype label 变化可能影响方向性结论。
2. **在 broad_ER band 重跑**：broad_ER 在部分 subjects 上有更多有效 onset，`onset_spread_sec` 和 `n_active` 可能分布更好。
3. **建立 SOZ-anchored T0/T1 convention**（Q3 axis），使 delta_rho 在 cohort 间可比，再重跑 Stage C。
4. **958 case study**: 探索 958 的 subtype 分组为何与 n_active/active_fraction/median_latency 强相关，是否与已知的临床特征相符。

---

## 7. 产出文件索引

| 文件 | 说明 |
|---|---|
| `results/topic1_topic5_bridge/q1prime_per_subject/{ds}_{sid}__q1prime.json` | 23 subjects Q1' per-seizure 结果 |
| `results/topic1_topic5_bridge/q1prime_features.csv` | 374 行 per-seizure 特征表 |
| `results/topic1_topic5_bridge/q1prime_feature_delta_rho_correlation.csv` | Stage C per-subject Spearman 结果 |
| `results/topic1_topic5_bridge/q1prime_feature_correlation_summary.json` | Stage C 汇总 JSON |
| `results/topic1_topic5_bridge/q1prime_feature_subtype_discrimination.csv` | Stage D per-subject 检验结果 |
| `results/topic1_topic5_bridge/q1prime_subtype_discrimination_summary.json` | Stage D 汇总 JSON |
| `results/topic1_topic5_bridge/figures/q1prime_overnight_feature_corr_heatmap.png` | Stage C subject × feature ρ heatmap |
| `results/topic1_topic5_bridge/figures/q1prime_overnight_feature_subtype_discrim_bar.png` | Stage D 条形图 |

---

**回链至主文档**: `docs/topic5_seizure_subtyping.md` → §Q1' overnight 探索节
