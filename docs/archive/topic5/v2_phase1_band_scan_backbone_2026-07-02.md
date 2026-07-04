# Topic 5 V2 Phase 1 — 频率扫描 backbone 建成 + dev-null Gate A 初判 (handoff)

date 2026-07-02 · 分支 `topic5-v2-phase1-build`（worktree `/home/honglab/leijiaxin/HFOsp-t5v2`，off `codex/topic4-m3a-v2-2` @ `e01c08b`，27 commits）· 状态：**backbone 全部建成 + 全部过 per-task review；dev n_perm=100 broad raw 已跑，Gate A 初判为「有前景的频带特异描述性信号，但形式化 Gate A 被稀疏杆几何卡住」**

> 计划：`docs/superpowers/plans/2026-07-01-topic5-v2-phase1-band-scan-backbone.md`；设计：`docs/superpowers/specs/2026-07-01-topic5-v2-hfo-critical-mode-design.md`。SDD 账本（含每个决策/局限的完整记录）：`.superpowers/sdd/progress.md`（该 worktree 内）。

---

## 口径修正 + tier 纪律（user review 2026-07-02，SUPERSEDES 下方 §0/§3 的措辞）

**tier（按 rev2 §1.1 证据阶梯，锁定）**：停在 **candidate scaffold / broadband-recruitment tier**。**禁止**升级到 timing-order mechanism / pathological critical mode。最稳的正结论只能写成：
> **HFO-derived core geometry may mark a shared, band-generic early-ictal recruitment scaffold**（不是 ripple/HFO-specific replay，也不是 timing-order mechanism）。

**措辞降级（承重）**：
- ❌"相似超过任何平滑空间场的巧合" / "沿着它铺开" / "共享底座"（肯定）
- ✅ **suggestive spatial co-structure**；**空间共定位 / 共享空间梯度**（不是 temporal propagation——现度量是 early-window 空间图相关，非时间传播方向，除非 Phase 3 做出 temporal recruitment）；**may mark a shared scaffold**。

**formal Gate A = UNEVALUABLE（非 failed-by-effect-size），且 weak-null 正证据很可能被高估**：预设的 within-shaft strong 空间 null 因 SEEG 杆稀疏无法满足（0/11），退化成 subject_wide_weak 后**不再保持杆内局部空间自相关**——在过大尺度打乱 → **让 observed map 看起来更显著（anti-conservative）**。所以现在只能写"formal within-shaft Gate A 未可评估 / unresolved；weak/global-null 下描述性为正，但该正证据可能因 null 过弱而偏高"，**不能**写"过了空间 null"。

**narrow ORDER null（补上——之前漏报，非缺失）**：narrow 跑了 order null，**4/7 被试 strong**（1146/253/384/958），3/7 weak_downgrade（gate-blocked）；obs **band-generic** 超出"保放电率、打乱时序"的 null（所有频带 Δ+0.16~+0.27，n_perm=100 分辨率下限 p=0.010，6-7/7）。→ narrow **不止贴 HFO-rich 触点拓扑**，strong 子集里对齐依赖**时序顺序**（超出放电率地形）。**但**：band-generic（非 HFO-specific）、dev n_perm=100（p 触底）、仅 4/7 strong、且 order null 有 T13 轻度 anti-conservative（obs=producer vs null=event-rebuild）→ 时序-order 主张只到"strong 子集描述性支持"，未闭合。

**cohort 显著性口径修正**：删掉我之前"改 per-band cohort permutation p 只会加强"的说法——**可能加强也可能变弱**（被试异质性 + maxT 校正）。正确做法：用 **subject-level cohort permutation of the median statistic + max-over-bands**（`cohort_stat_perm[band,perm]=median_over_subjects(subject_stat_perm)`；`maxT_perm=max_over_bands`），**不再用 median-of-per-subject-p 作主推断**。

**四个诚实标签（放正式报告首页）**：`legacy reproduction: PASS` · `weak/global spatial-null evidence: positive (likely inflated)` · `formal within-shaft Gate A: unresolved (null-strength gated, 0/11)` · `Gate B/C: pending`。

**下一步 spec（运行前锁定）**：见 `docs/superpowers/specs/2026-07-02-topic5-v2-phase1b-gate-closure-spec.md`（P1 空间-null hierarchy + subject-level cohort permutation + narrow order-null 闭合 + broad↔narrow channel-pool ablation；min_group=3 用 exact/enumerated 排列，不升级为 formal primary 除非写进修订 hierarchy）。

## 全 20 队列 maxAB 对齐表（观测层 · 2026-07-02 · 队列 13→20）

**队列扩充（补齐 field-similarity 全 20）**：Phase-1 backbone 原沿用发作内场动力学手挑的 13 被试（broad 9 / narrow 7），欠采样于 field-similarity（`axis_alignment_*_max_ab_B1000.json`，per_subject n=20 = 18 epilepsiae + 2 yuquan）。本轮补齐：**narrow 7→20、broad 9→17**（442/548/958 只有 narrow 几何、无 broad 几何，故不入 broad）。2 个 yuquan（xuxinyi/zhangkexuan）的发作只标 `eeg_onset_epoch`（无 `clin_onset_epoch`）→ 走 eeg-onset 锚（`iter_subject_seizure_windows` + `build_subject` 抽 `_anchor_epoch` fallback，TDD 锁 epilepsiae 逐位不变），且各仅 1–2 个真发作（余为零时长标记，被 ictal_fraction 门自然滤掉）→ 薄、混锚，标 `anchor=eeg_onset`。

**测了什么 / 怎么测的（朴素话）**：每个被试的间期 HFO 几何有两张"平时谁先谁后"的顺序场（A、B 模板），平滑成空间图；发作刚起头早窗里每个频带的能量按触点平均 → 空间图 → 同样平滑。单元格 = max( |corr(该频带早期能量场, 间期 A 场)|, |corr(…, B 场)| )（"maxAB 任意场"），窗→发作→被试中位数聚合。范围 0–1，越大越像。

**揭示了什么（描述层，未过 null；tier = candidate scaffold / broadband-recruitment）**：
- **频带上是平的（band-generic）**：narrow 各带中位数 0.68–0.75、broad 0.59–0.66，ripple 段并不更高——发作早期能量场贴间期 HFO 几何的程度不随频带（δ→ripple）系统变化，**不是 ripple/HFO 特异**。
- **narrow（核心几何）系统性高于 broad（扩展池）**：中位数 ~0.72 vs ~0.63。核心触点池的几何是更紧的 readout。
- **18-clean ≈ 20-mixed**（各带中位数差 ≤0.03）：2 个薄 eeg-onset yuquan 不改变队列结论，混锚 caveat 未扭曲描述结果。

#### narrow  (n=20 subjects) — cell = interictal-maxAB |corr| vs early-ictal band field

| subject | n_sz | δ 1-4 | θ 4-8 | α 8-13 | β 13-30 | γ 30-80 | hgR 80-150 | R 150-250 | low 1-13 | LVFA 13-80 | Rf 80-250 | Rs 80-220 | bb 1-45 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1077 | 8 | 0.63 | 0.71 | 0.72 | 0.88 | 0.79 | 0.80 | 0.69 | 0.69 | 0.87 | 0.74 | 0.77 | 0.75 |
| 1084 | 2 | 0.80 | 0.49 | 0.57 | 0.52 | 0.40 | 0.51 | 0.66 | 0.63 | 0.53 | 0.56 | 0.54 | 0.64 |
| 1096 | 8 | 0.88 | 0.76 | 0.74 | 0.91 | 0.77 | 0.78 | 0.81 | 0.79 | 0.86 | 0.74 | 0.73 | 0.66 |
| 1125 | 13 | 0.73 | 0.73 | 0.70 | 0.72 | 0.64 | 0.63 | 0.57 | 0.85 | 0.69 | 0.67 | 0.66 | 0.84 |
| 1146 | 23 | 0.78 | 0.75 | 0.75 | 0.54 | 0.49 | 0.59 | 0.70 | 0.76 | 0.52 | 0.58 | 0.59 | 0.78 |
| 1150 | 8 | 0.75 | 0.81 | 0.86 | 0.77 | 0.73 | 0.81 | 0.87 | 0.86 | 0.70 | 0.78 | 0.78 | 0.83 |
| 139 | 4 | 0.85 | 0.52 | 0.45 | 0.63 | 0.67 | 0.53 | 0.52 | 0.36 | 0.53 | 0.52 | 0.53 | 0.38 |
| 253 | 6 | 0.66 | 0.57 | 0.79 | 0.80 | 0.78 | 0.73 | 0.74 | 0.51 | 0.80 | 0.70 | 0.69 | 0.54 |
| 384 | 10 | 0.56 | 0.77 | 0.58 | 0.69 | 0.70 | 0.59 | 0.66 | 0.52 | 0.66 | 0.60 | 0.60 | 0.52 |
| 442 | 20 | 0.61 | 0.60 | 0.50 | 0.44 | 0.31 | 0.61 | 0.57 | 0.60 | 0.42 | 0.63 | 0.62 | 0.59 |
| 548 | 23 | 0.87 | 0.90 | 0.89 | 0.82 | 0.90 | 0.91 | 0.80 | 0.90 | 0.80 | 0.90 | 0.90 | 0.90 |
| 583 | 21 | 0.65 | 0.89 | 0.97 | 0.97 | 0.92 | 0.83 | 0.76 | 0.85 | 0.97 | 0.83 | 0.83 | 0.87 |
| 590 | 11 | 0.80 | 0.85 | 0.90 | 0.92 | 0.87 | 0.81 | 0.72 | 0.91 | 0.92 | 0.79 | 0.80 | 0.91 |
| 620 | 6 | 0.67 | 0.51 | 0.55 | 0.51 | 0.56 | 0.31 | 0.36 | 0.58 | 0.43 | 0.35 | 0.36 | 0.61 |
| 635 | 16 | 0.82 | 0.83 | 0.71 | 0.86 | 0.75 | 0.81 | 0.91 | 0.84 | 0.85 | 0.81 | 0.81 | 0.82 |
| 916 | 37 | 0.77 | 0.94 | 0.82 | 0.91 | 0.95 | 0.95 | 0.68 | 0.78 | 0.92 | 0.95 | 0.95 | 0.79 |
| 922 | 15 | 0.62 | 0.62 | 0.62 | 0.41 | 0.65 | 0.74 | 0.53 | 0.60 | 0.43 | 0.68 | 0.68 | 0.67 |
| 958 | 10 | 0.66 | 0.59 | 0.55 | 0.60 | 0.70 | 0.77 | 0.52 | 0.69 | 0.62 | 0.79 | 0.79 | 0.63 |
| xuxinyi | 1 | 0.89 | 0.83 | 0.54 | 0.59 | 0.51 | 0.65 | 0.51 | 0.86 | 0.56 | 0.63 | 0.63 | 0.83 |
| zhangkexuan | 2 | 0.72 | 0.61 | 0.60 | 0.74 | 0.55 | 0.43 | 0.71 | 0.71 | 0.66 | 0.51 | 0.49 | 0.71 |
| **median (all n=20)** |  | **0.74** | **0.74** | **0.70** | **0.73** | **0.70** | **0.73** | **0.68** | **0.74** | **0.68** | **0.69** | **0.69** | **0.73** |
| **median (epilepsiae-only n=18)** |  | **0.74** | **0.74** | **0.71** | **0.74** | **0.71** | **0.75** | **0.68** | **0.73** | **0.70** | **0.72** | **0.71** | **0.71** |

#### broad  (n=17 subjects) — cell = interictal-maxAB |corr| vs early-ictal band field

| subject | n_sz | δ 1-4 | θ 4-8 | α 8-13 | β 13-30 | γ 30-80 | hgR 80-150 | R 150-250 | low 1-13 | LVFA 13-80 | Rf 80-250 | Rs 80-220 | bb 1-45 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1077 | 8 | 0.66 | 0.64 | 0.64 | 0.89 | 0.85 | 0.56 | 0.56 | 0.60 | 0.89 | 0.51 | 0.55 | 0.54 |
| 1084 | 2 | 0.68 | 0.67 | 0.55 | 0.39 | 0.37 | 0.62 | 0.59 | 0.66 | 0.34 | 0.61 | 0.61 | 0.63 |
| 1096 | 8 | 0.57 | 0.67 | 0.64 | 0.80 | 0.58 | 0.67 | 0.69 | 0.65 | 0.78 | 0.69 | 0.69 | 0.67 |
| 1125 | 13 | 0.41 | 0.47 | 0.48 | 0.60 | 0.38 | 0.49 | 0.51 | 0.42 | 0.56 | 0.51 | 0.50 | 0.41 |
| 1146 | 23 | 0.74 | 0.67 | 0.75 | 0.70 | 0.69 | 0.66 | 0.68 | 0.68 | 0.60 | 0.61 | 0.63 | 0.70 |
| 1150 | 8 | 0.49 | 0.58 | 0.43 | 0.54 | 0.73 | 0.55 | 0.51 | 0.68 | 0.63 | 0.54 | 0.54 | 0.67 |
| 139 | 4 | 0.57 | 0.40 | 0.61 | 0.61 | 0.44 | 0.52 | 0.51 | 0.35 | 0.58 | 0.54 | 0.53 | 0.34 |
| 253 | 6 | 0.73 | 0.73 | 0.76 | 0.85 | 0.80 | 0.76 | 0.80 | 0.70 | 0.87 | 0.73 | 0.74 | 0.73 |
| 384 | 10 | 0.42 | 0.60 | 0.50 | 0.60 | 0.64 | 0.64 | 0.64 | 0.46 | 0.57 | 0.63 | 0.63 | 0.47 |
| 583 | 21 | 0.68 | 0.81 | 0.83 | 0.81 | 0.79 | 0.61 | 0.53 | 0.78 | 0.79 | 0.62 | 0.62 | 0.79 |
| 590 | 11 | 0.75 | 0.83 | 0.92 | 0.90 | 0.87 | 0.83 | 0.74 | 0.92 | 0.89 | 0.83 | 0.83 | 0.92 |
| 620 | 6 | 0.49 | 0.54 | 0.38 | 0.40 | 0.47 | 0.58 | 0.60 | 0.53 | 0.44 | 0.55 | 0.56 | 0.48 |
| 635 | 16 | 0.61 | 0.58 | 0.62 | 0.62 | 0.73 | 0.77 | 0.65 | 0.57 | 0.65 | 0.78 | 0.78 | 0.59 |
| 916 | 37 | 0.72 | 0.75 | 0.67 | 0.56 | 0.57 | 0.60 | 0.44 | 0.72 | 0.51 | 0.59 | 0.59 | 0.72 |
| 922 | 15 | 0.66 | 0.51 | 0.59 | 0.85 | 0.73 | 0.69 | 0.61 | 0.56 | 0.86 | 0.71 | 0.70 | 0.56 |
| xuxinyi | 1 | 0.81 | 0.69 | 0.49 | 0.40 | 0.63 | 0.69 | 0.45 | 0.74 | 0.48 | 0.67 | 0.67 | 0.69 |
| zhangkexuan | 2 | 0.62 | 0.47 | 0.62 | 0.73 | 0.47 | 0.40 | 0.58 | 0.60 | 0.63 | 0.45 | 0.44 | 0.61 |
| **median (all n=17)** |  | **0.66** | **0.64** | **0.62** | **0.62** | **0.64** | **0.62** | **0.59** | **0.65** | **0.63** | **0.61** | **0.62** | **0.63** |
| **median (epilepsiae-only n=15)** |  | **0.66** | **0.64** | **0.62** | **0.62** | **0.69** | **0.62** | **0.60** | **0.65** | **0.63** | **0.61** | **0.62** | **0.63** |

**读表纪律（勿过读）**：格子是**平滑场的 |corr| 幅度**，相对 0 天然偏高（平滑 + 取绝对值，随机场也给正值）——这是**描述性幅度 + 频带轮廓**，**不是显著性判决**。"是否真存在相关"须看能否超过空间 / 顺序 / 1-f null（= P1b spec §1–§3、§5，尚未跑，held on user go）。per-subject 异质大（548/583/590/916 ~0.85–0.95；620/442/139 低）；yuquan 两行 n_sz=1/2、eeg-onset 锚，个体值几乎无发作间稳健性。数据来源：`results/topic5_ictal_recruitment/v2_band_scan/{narrow,broad}/phase1_alignment_raw_subject_summary.csv`（`align_abs_maxab`，`used_fixed_mask=True`）。

## Formal-null 结果 + 验收（**full n_perm=1000 · narrow+broad · 2026-07-04**）

> 判读 frame = spec §EXP。**验收结论：可作为 Phase-1 exploratory candidate-scaffold result（cohort 层）验收；不可作 formal Gate A / 任何机制结论。** 四条 caveat 写死：per-subject 弱一致 · weak-null 反保守 · formal within-shaft Gate A 未评估 · Gate B/C 未跑。

**怎么测的**：发作 onset 后 0–20s 各频带能量空间场 vs 间期 HFO geometry 场对齐（§EXP primary endpoint = raw alignment），问能否超过 (A) 杆内洗牌 spatial null、(B) 保留 HFO 富集地形只打乱时序的 order null。统计 = subject-level cohort permutation of the median + max-over-bands（§2；**7 primary bands 为 FWER family**）。

**① cohort 层（primary endpoint，n_perm=1000）**

| | narrow (n=20) | broad (n=17) |
|---|---|---|
| primary 7 带过**空间 null**（FWER，p≈.001–.008） | **6/7** | **6/7** |
| primary 7 带过 **order null**（strong 子集 13） | **7/7** | **7/7** |
| 唯一不过 FWER 的带 | ripple_high | ripple_high |
| spatial null 强度（cohort weakest-wins） | subject_wide_weak | subject_wide_weak |

→ 跨频段超过弱空间 null + order null；**NOT ripple-specific**（ripple_high 两池最弱）。

**② per-subject 稳定性（承重 caveat，之前 cohort 6/7 掩盖了它）**：cohort 6/7 是**聚合**结果、**不是** per-subject 稳健。narrow **中位 subject 只 2/7 带显著**（δ>0 & 该 subject 自身 p<.05），**≥5/7 仅 3/20**（1096/1146/1150）、**≥4/7 仅 6/20**；broad 中位 3/7、≥5/7=7/17（触点更密 n_con~20）。跨两池都稳的只 **1146/1150/384**；**1146 是唯一 within_shaft_strong + 多带阳**（narrow 5/7、broad 7/7；n=1 非 cohort）。**无单一 phenotype**（strength/n_sz/n_con/maxab 混杂），且多带阳 subject 大多**不是** within_shaft_strong。

**③ 频带梯度**：narrow **基本平**（HF−low≈+0.000，真 band-generic）；broad **轻微 β(13–30) 峰**（cohort med delta 0.110，HF−low +0.014）——"fast/LVFA"只是 β 单带略高、非 HF 平台。ripple_high 两池最弱（叠加 256Hz Nyquist 丢带 + ripple 能量更局灶、场平滑对齐弱）。

**④ formal within-shaft Gate A = unresolved**：仅 **2/20**（narrow）达 within_shaft_strong → cohort subject_wide_weak；弱 null 破坏杆内局部自相关 → **反保守 → 正面很可能被抬高**。**SEEG 杆几何硬限制，不是样本量、不是 effect size**。

**⑤ Gate B/C = 未跑**（common_resid/aperiodic residual cache = 0）→ **宽带招募 / 1-f / confound 未排除**。

**验收口径（accepted wording，可直接引用）**：
> 在 20-subject（narrow）/17（broad）队列、n_perm=1000 下，发作 onset 后 0–20s 的多频带能量空间场与间期 HFO-derived geometry 的对齐，在 **cohort 层**跨频段超过弱/全局空间 null（FWER 后 6/7 primary，唯 ripple_high 不过）及 order null（strong 子集）。支持一个 **candidate early-ictal spatial recruitment scaffold（field-level suggestive spatial co-structure）**。**限定**：(i) 效应为 **cohort 统计倾向、per-subject 弱一致**（narrow 中位 2/7、≥5/7 仅 3/20）；(ii) 仅在 **weak/global spatial null** 下成立、**formal within-shaft Gate A 未可评估**（2/20），显著性**可能被抬高**；(iii) **Gate B/C 未跑**，未排除宽带招募/aperiodic；(iv) 仅 onset 后短窗、未覆盖 pre-ictal。**band-generic**（narrow 平 / broad 轻微 β 峰）、**NOT ripple-specific**。**不作** HFO-/LVFA-/ripple-specific、timing-order replay、formal Gate A passed。

**tags**：`legacy: PASS` · `cohort exploratory null-positive (6/7 primary FWER, weak null, likely inflated)` · `per-subject: WEAK (narrow median 2/7, ≥5/7 only 3/20)` · `formal within-shaft Gate A: unresolved (2/20)` · `order (strong 13): supportive` · `band-generic (narrow flat / broad mild β) · NOT ripple-specific` · `Gate B/C: NOT run` · `time: onset+0–20s only` · **`verdict: Phase-1 exploratory candidate-scaffold ACCEPTED (cohort-level); NOT formal/mechanism`**。

**图（paper-ready = Fig3-Sup1，Fig3-A field concordance 的多频带 supplement）**：3 panel 各答一个独立问题，遵 figure_style_guide §0。
- **A 观测层** `results/topic5_ictal_recruitment/v2_band_scan/figures/phase1_F1_observed_maxAB_heatmap.png` — subject×band maxAB 热图（红蓝 diverging 蓝<0.5<红、显著 cell 标白星=自身 null p<0.05、primary\|composite 黑虚线、末行 cohort 中位）：narrow>broad、band-generic。
- **B 形式化 null** `.../figures/phase1_F2_null_per_band.png` — 每 primary 带 per-subject Δ violin+背景点、cohort Δ 黑条、`*`=过 family-wise：两池 6/7 过、ripple_high n.s.（NOT ripple-specific）。
- **C per-subject caveat** `.../figures/phase1_F3_per_subject_stability.png` — 每 subject 显著频带数：cohort 6/7 是**聚合**、narrow 中位 2/7（承重 caveat）。
- **paper-ready 副本（Fig3-Sup1）**：`results/paper-ready-figure/fig3_sup1_multiband_field_alignment/figures/fig3sup1_{A,B,C}_*.{png,pdf}`；脚本 `scripts/paper_figures/plot_fig3_sup1_multiband_field_alignment.py`（复用 `scripts/plot_topic5_v2_phase1_figures.py`）。

**关键路径（整体验收索引）**：
- artifacts（n_perm=1000）：`results/topic5_ictal_recruitment/v2_band_scan/{narrow,broad}/phase1_{gate,null_raw,alignment_raw}_*`
- 代码：`src/topic5_v2_band_scan.py` + `scripts/run_topic5_v2_{alignment,nulls,gates}.py`（34 纯测绿）
- spec：`docs/superpowers/specs/2026-07-02-topic5-v2-phase1b-gate-closure-spec.md`（§EXP frame + §0–§8 锁）
- 下一步 plan：`docs/superpowers/plans/2026-07-04-topic5-v2-phase1-v2-scaffold-closure.md`

## 0. 摘要（朴素话，措辞见上方"口径修正"，频带特异描述已降级为 narrow band-generic）

**测了什么**：癫痫病人发作**刚起头 20 秒**里，每个频带（δ→ripple）的能量在电极阵列上"点亮"成一张空间图；间期时同一批触点各自有一条"平时谁先谁后"的 HFO 传播几何（G_HFO）。我们问：发作早期某频带的能量场，长得像不像间期这张顺序几何图，而且这种像**能不能扛过三道质疑**——(A) 只是空间涂抹？(B) 只是宽带整体招募？(C) 只是 1/f 背景？

**揭示了什么（初判，dev n_perm=100，只 broad raw，未过形式化 gate）**：
- **贴合强度 |maxab| 在所有频带上是平的**（~0.55-0.73），ripple 段并不更高——光看强度没有频带特异性。
- **但比"纯空间平滑"随机基线（spatial null）——中高频带明显超出，低频不超**：beta(13-30) Δ+0.116 p=0.010、LVFA_13_80 Δ+0.107 p=0.020、high-gamma Δ+0.112 p=0.030、ripple_full Δ+0.111 p=0.030（7/9 被试超出）；而 δ/α/宽带 Δ~+0.02-0.04 p~0.22-0.27（5/9）。即**中高频能量场贴 G_HFO 的程度超过空间平滑能给的，低频不超**——是频带特异、方向符合假设的信号。
- **过完 band 间多重比较（max-over-bands FWER）后，只有 beta + LVFA_13_80 稳住（~0.01-0.02），ripple 掉了（~0.75-0.91）**。→ 稳健信号是 **LVFA/fast(13-80Hz)，不是 ripple-specific**（对应 spec §8 的**中档**判读："HFO 几何标记一条以 LVFA/fast 招募表达的致痫通路，非 ripple-specific"）。
- **形式化 Gate A：0/11 全 weak_negative——但不是 p 值不够，是被"强度门"卡住**：min_group=4 下没有一个被试达到 within_shaft_strong（都 subject_wide_weak/distance_bin_fallback，因为 SEEG 杆稀疏），而 spec 的 P1-c 纪律规定只有 within_shaft_strong 能过形式化 Gate A。所以这是**几何/参数限制，不是干净的"对齐失败"**。

**一句话**：backbone 干净跑通，初判看到一个**有前景的、频带特异（LVFA/fast）、方向对的描述性信号**（中高频超空间平滑 null），但**形式化验证被稀疏杆几何卡住**，且**未过全量 n_perm、未跑 Gate B/C、未跑 narrow**——按 spec §1.1 证据阶梯，停在 **candidate mode，形式化 Gate A 未确立**。

---

## 1. 建成的东西（pipeline）

纯数学在 `src/topic5_v2_band_scan.py`；编排在 `scripts/run_topic5_v2_*.py` / `build_topic5_v2_*.py`；config 驱动 `config/topic5_v2_phase1.yaml`。

| 阶段 | 脚本 | 产物 |
|---|---|---|
| **硬门·legacy 复现** | `run_topic5_v2_legacy_repro.py` | 证明 v2 管线复现旧 bb/hfa align_maxab，**max\|delta\|=0.00 broad+narrow**（逐位相等） |
| **多频带 masked cache** | `build_topic5_v2_band_cache.py` | `v2_band_scan/cache/{sid}.npz`（12 band × 每 sz 的 baseline-robust-z 迹）+ sidecar（`analysis_channels` 固定掩膜 + 逐 band QC）。13 被试（broad9+narrow4）全建成 |
| **对齐表** | `run_topic5_v2_alignment.py --feature {raw\|common_resid\|aperiodic_resid}` | 窗→发作→被试中位数的 align_abs_maxab + signed 表 |
| **残差 cache（Gate B/C 输入）** | `build_topic5_v2_common_resid_cache.py`（LOBO 共场残差）、`build_topic5_v2_aperiodic_cache.py`（1/f 校正超量） | 同结构残差 cache |
| **混杂图** | `build_topic5_v2_confound_maps.py` | 每触点 hfo_rate / baseline_power / broadband / **shaft-order（非 along_axis_mm，避免自证循环）** / soz |
| **三层 null** | `run_topic5_v2_nulls.py --feature .. --n-perm ..` | perm-long parquet（max-over-bands 用）+ subject summary（null_z/empirical_p/strength） |
| **Gate 判读** | `run_topic5_v2_gates.py` | `phase1_gate_summary.csv`（Gate A/B/C flag + tier + max_over_bands_p） |

关键科学修正（详见账本）：
- **饱和质检从固定掩膜里剔除**：原设计按 band-power-z `|z|>12` 判"坏道"，会把发作时正在放电的高-ripple 通道当噪声剔掉（139 ripple 段 41→1）。改为 `analysis_channels` 只按**有效性**（有限、非 flatline）筛，饱和标记保留为旁路诊断——避免删掉要测的信号（循环论证）。cohort-wide 0 通道掉（数据干净）。
- **建 cache 提速 ~12×**：频谱图每发作只算一次、所有 band 共用（原来每 band 重算）——单被试 ~10-27min→48s；**legacy_bb 逐位仍与旧 cache 相等**（提速没动数字）。
- **order-null 用 producer nanmedian**（非 mean）匹配 G_HFO 几何（§6 边界参数一致性）。
- **shaft_position 用杆序索引**（非 along_axis_mm，后者是 G_HFO 派生的传播轴，做混杂会自证循环）。

---

## 2. 怎么跑全量（controller 未跑完的）

```bash
cd /home/honglab/leijiaxin/HFOsp-t5v2   # 该 worktree（results/ 软链到主 results）
# cache 已建（13 subj）。observed alignment：broad raw 已跑，narrow raw 已跑。
for ax in broad narrow; do
  python scripts/build_topic5_v2_confound_maps.py --substrate $ax          # 全量混杂图（dev 只 139）
  python scripts/build_topic5_v2_common_resid_cache.py --substrate $ax     # Gate B 残差 cache（未跑全量）
  python scripts/build_topic5_v2_aperiodic_cache.py --substrate $ax        # Gate C 残差 cache（~15-35min，内存 8-12GB/subj，顺序跑）
  for feat in raw common_resid aperiodic_resid; do
    python scripts/run_topic5_v2_alignment.py --substrate $ax --feature $feat
    python scripts/run_topic5_v2_nulls.py --substrate $ax --feature $feat --n-perm 1000   # ★慢：见下
  done
  python scripts/run_topic5_v2_gates.py --substrate $ax
done
```

**⚠️ 全量 null 极慢**：实测 n_perm=1000 ≈ **10.5 min/subject/feature**（E916 有 44 发作，≈2h 一个），broad ≈2.5-3.5h/feature，**全 feature×substrate ≈ 12-18h**。建议先 dev(100) 定性、final(1000) 后台排队。

---

## 3. dev-null Gate A 结果（broad raw n_perm=100，descriptive，未过形式化 gate）

见 §0 摘要 + `results/topic5_ictal_recruitment/v2_band_scan/broad/phase1_{gate_summary,null_subject_summary}.csv`。核心表（spatial null，中位 Δ=obs−null / 中位 empirical_p / #被试 obs>null）：

```
beta_LVFA_low        Δ+0.116  p=0.010  7/9   ← 超空间平滑 + family-wise 稳住(0.0099)
LVFA_13_80           Δ+0.107  p=0.020  7/9   ← 超 + family-wise 稳住(0.0198)
hg_low_ripple        Δ+0.112  p=0.030  7/9   ← 超，但 family-wise 掉(0.119)
ripple_full_80_250   Δ+0.111  p=0.030  7/9   ← 超，但 family-wise 掉(0.911)
ripple_safe_80_220   Δ+0.110  p=0.030  7/9   ← 同上
theta_preictal_PAC   Δ+0.066  p=0.040  6/9
delta/alpha/low_HYP/legacy_bb  Δ~+0.02-0.04  p~0.22-0.27  5-6/9   ← 不超
```
order null：所有频带都正（Δ+0.12-0.19）但 band-generic（非频带特异）、p 多 NS at n=100、且 gate-guarded + 轻度 anti-conservative（见 §4）→ 证据弱于 spatial。

**★ NARROW (n=7) spatial null（跨队列对照，重要）**：**所有频带都超出空间平滑 null**（Δ+0.07-0.16 p~0.01-0.06 5-6/7），**包括低频/宽带**（delta +0.132 p=0.010、low_HYP +0.162 p=0.010、legacy_bb +0.111 p=0.020）——是 **band-generic**，**不复现 broad 的中高频特异**。两点跨队列结论：
- **一致（robust）**：**两个队列 obs 都超出空间平滑 null**（broad 中高频超、narrow 全频段超，p<0.05，多数被试）→ **确实存在超出纯平滑的空间特异对齐**（描述性；形式化 Gate A 仍被 strength 门卡住）。注意"|maxab| 平"不等于"纯平滑"——null 中位数更低（~0.44），obs（~0.55）超出它。
- **不一致（NOT robust）**：**哪个频带**（频带特异性 = Gate B 的问题）——broad 频带特异（LVFA/fast），narrow band-generic（含宽带）。→ **频带特异主张不跨队列稳健**；narrow 的 band-generic 反而更像 spec 的 **broadband-recruitment 档**（G_HFO 预测宽带招募，非频带特异）。
- **诚实底线**：稳健的是"对齐超平滑"（Gate-A 描述性），不稳健的是"频带特异"（Gate-B）。整体倾向 **"G_HFO 标记一条超出平滑的空间招募通路，但频带特异性 + 形式化验证均未确立"**。

---

## 4. 已知局限 + 全量前待定项（承重，勿忽略）

1. **形式化 Gate A 被 min_group=4 卡死（0/9 within_shaft_strong）**。**关键 follow-up = min_group=3 灵敏度**（一个 flag：`nulls.min_group_for_shaft`）——密杆被试可能变 within_shaft_strong → 形式化 Gate A 才可评。现结论"形式化未过"是**几何/参数产物，非对齐失败**。
2. **Gate A cohort 显著性 = 逐被试 p 的中位数（median-of-p）——偏保守**（null 下逐被试 p~U(0,1)，中位数 ~0.5，n≥5 时 median-p<0.05 近乎不可达）。**更合适的每-band cohort permutation p 已在 `_max_over_bands_p` 里算好但没接到 spatial_p**——全量前应换（只会**加强**信号，不会削弱）。Task-14s review 的 Important 项。
3. **order-null 轻度 anti-conservative**：observed 用 producer typical_rank，null 用 event-rebuild rank（只到 corr ~0.95/0.80 复现 producer）→ strong 被试的 order-p 略乐观（gate-guarded：只 ≥0.90 的 strong 被试进决策，gap≤10%）。**spatial null 才是干净的主 Gate A 检验**；order 是 gate-保护的次要。
4. **max-T family 含 4 个与 primary 重叠的 composite**（保守，FWER 仍控住，但损 power）——全量可考虑只用 7 primary。
5. **未跑**：Gate B/C（残差 feature 全量 null）、narrow 全量、final n_perm=1000、全量 confound-adjusted。
6. **signed 方向度量不跨 substrate 稳健**：broad 低频正/高频负，narrow 多为负——因 signed 依赖 per-subject template-a 定向，跨 substrate 不可比。用**方向-不变的 |maxab|**（平的）+ **spatial-null Δ**（频带特异）作可信度量，不报 signed flip 为发现。
7. **composite band（LVFA_13_80 / ripple_full / ripple_safe）结构上封顶在 broadband_recruitment**：残差 cache 只建 7 个 primary，故 composite 的 common_resid_p/aperiodic_p 恒 NaN → 永远过不了 Gate B/C（whole-branch review 的 Important-interpretation 项）。读 gate_summary 时注意：composite 的 tier ≤ broadband_recruitment 是设计使然，非"频带不特异"的证据。Gate B/C 只在 7 primary 上判。

---

## 5. 工程状态

- 27 commits（`e01c08b..HEAD`），线性、全 `topic5-v2-`。纯测 `tests/test_topic5_v2_band_scan.py` **31 passed**；12 个 `@pytest.mark.integration` real-data smoke（各任务内验过）。
- 每个任务都由独立 subagent review（多数含手推导 + 边界测），journal 在 `.superpowers/sdd/`。cleanup pass（c9af042）补齐了饱和-nan robustness、legacy-repro teeth、null/gate/aperiodic/confound 回归测试。
- **worktree 隔离**：因主工作目录有并行 Phase-2 会话（多会话共享 HEAD 会撞车），Phase-1 全程在独立 worktree `HFOsp-t5v2` / 分支 `topic5-v2-phase1-build`，`results/` 软链共享数据。分支整合（merge/PR）待用户定。
- **未碰**：主工作目录的 Phase-2 criticality WIP + field-extrapolation WIP（互补、无文件冲突；本 Phase-1 产物正好满足 Phase-2 depcheck 的输入）。
