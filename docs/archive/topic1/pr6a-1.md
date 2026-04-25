> **SUPERSEDED 2026-04-25**：本文档（multi-anchor consensus probe）已被 PR-6 主线 pivot 取代。新主线把问题从“给 template 找 cohort-level ictal-onset anchor + 命名”改为“stable template 的 endpoint (source ∪ sink) 是否解剖锚定 SOZ / focus_rel”。**正式 plan-of-record**：[`docs/archive/topic1/pr6_template_endpoint_anchoring_plan_2026-04-25.md`](pr6_template_endpoint_anchoring_plan_2026-04-25.md)。本文档保留作为 pivot 决策证据：multi-anchor consensus 的设计动机仍可读，但**不再作为 PR-6-A 的执行路径**。

---

# PR-6-A-1: Multi-Anchor Consensus for Interictal Template Semantic Alignment — Probe Phase

> 状态：Probe phase plan，2026-04-25
> 范围：本文档定义 PR-6-A 启动前的硬前置 probe 阶段（PR-6-A-1）。Probe 通过后才进入 PR-6-A 正式合同的撰写。
> 性质：方法学 feasibility 验证，不是正式 inferential analysis。
> 上游：`docs/topic1_within_event_dynamics.md` §7（PR-6 计划占位）
> 下游：`docs/archive/topic1/pr6a_template_ictal_alignment_plan_2026-04-xx.md`（Probe 通过后撰写）

---

## 1. 这个 PR 解决的核心问题

### 1.1 PR-6 整体动机的回顾

Topic 1 已经在 PR-2 / PR-2.5 / PR-3 把 30/30 stable adaptive interictal templates 钉死。在 k=2 主导的 cohort 里（27/30），每个 subject 都有两个稳定的 template（C0 / C1），其中 9 个 subject 出现 `inter-cluster Spearman r < -0.5` 的 forward/reverse 对称结构。

但 **C0 / C1 这两个标签是 KMeans 任意初始化的产物，本身没有 cohort-level 物理含义**。要在 cohort 层面 claim "T_X 这一类 template 普遍具有某性质"，必须先给 C0 / C1 一个跨 subject 一致的语义命名。

**PR-6-A 的初衷就是给这两类 template 一个有物理意义的、cohort-consistent 的命名。** PR-6-A-1（本 plan）是它的可行性 probe。

### 1.2 单点问题：我们需要什么样的 anchor

要给 template 命名，必须有一个**外部参考时序**（external reference ordering），它满足：

1. **是 channel-level 的**（template 的物理身份就在 channel rank geometry 上，不能用 channel-set 做 anchor）
2. **跟 interictal templates 是独立采样的**（不同时间窗、不同信号类型），否则相关性 trivial
3. **在 cohort level 物理含义一致**（每个 subject 算出来的 anchor 都在同一种含义上）
4. **不需要是金标准**——只需要稳定到能给 template 命名

### 1.3 从 PR-6-A 旧设计到 PR-6-A-1 的转折

旧 PR-6-A 草案把"ictal channel-wise onset rank"作为 anchor，并赌它在每个 subject 内部稳定。Sentinel 测试（`548` / `916`）显示这个赌法不成立：

- `548` 两次 seizure 的 gamma_ER preclinical top-10 overlap = 0
- `916` 同一次 seizure 内 gamma_ER vs broad_ER 的 channel rank ρ = -0.21
- 多个最早被 ER detect 的通道是 `other`（既不在 SOZ 标签里，也不是 high-HFO 通道）

**如果继续把"单 anchor 跨 seizure 稳定"当作前提，PR-6-A 整条线在 sentinel 上就死了。** PR-6-A-1 的任务是检验：**多 anchor consensus + naming agreement 这一层抽象能不能把不稳的 single anchor 救起来**。

---

## 2. 文献已经告诉我们的事

不要假装这个领域问题没人碰过。以下是直接决定 PR-6-A-1 设计的文献事实：

### 2.1 Single ictal anchor 跨 seizure 不稳是 published consensus

Schroeder et al. 2020 PNAS: "In all subjects, we found variability in seizure paths through the space of possible network dynamics, producing either a spectrum or clusters of different dynamics... seizure pathways change on circadian and slower timescales in the majority of patients"

— 31 patients、平均 16.5 seizures/patient 的 within-patient seizure pathway 都在变。

Wenzel et al. 2017, Weaver et al. 2022 Epilepsy Currents: "the sequence of specific neurons that were activated during seizure initiation was highly variable... seizure initiation is stochastic at the single neuron level"

— Cellular-resolution calcium imaging 的标题就叫 "Ictogenesis? That's Random…"。

Pinto et al. 2023 Sci Reports: "Recent studies reporting preictal interval selection among a range of fixed intervals show inter- and intra-patient preictal interval variability... heterogeneity of the ictogenesis mechanisms among seizures (intra- and inter-patient) can contribute to the unsatisfactory performance of current seizure prediction models"

— 直接归因 ictogenesis variability 是 seizure prediction 长期不 work 的主因。

Balatskaya et al. 2020 Clin Neurophysiol (cEI): "the objective was not to measure intra-patient reproducibility but to compare two measures... we chose a single representative seizure per patient"

— Bartolomei 组在 cEI 论文里**显式回避 intra-patient reproducibility**，每个 patient 只用一次代表性 seizure。这是个 silent admission。

### 2.2 Interictal-ictal recapitulation 是已 establish 的现象

Korzeniewska et al. 2014: "Patterns of propagation during seizures are recapitulated in interictal recordings"

— SdDTF 在 HFO band 上展示 interictal-ictal propagation 高度重合。

Smith et al. 2022 eLife: "the majority of IEDs are traveling waves, traversing the same path as ictal discharges during seizures, and with a fixed direction relative to seizure propagation... bidirectional, with one predominant and a second, less frequent antipodal direction"

— μm-scale MEA 上的双向 traveling wave，直接预测我们 9-subject forward/reverse subset。

### 2.3 SOZ label 也不可靠但是 community 的不得不用

Spring et al. 2018 Nat Commun: "the precise localization is consistent in only 22% of patients. The remaining patients either have one intermittent source (16%), different sources varying over time (45%), or insufficient HFOs (17%)"

— HFO localization 跨时间，22% 严格一致。临床 SOZ label 综合多 seizure + 解剖 + 影像，已是 community 最好的 anchor，但仍然不是 ground truth。

### 2.4 文献给我们的 takeaway

- "Single ictal anchor 稳"是错误的预期——领域共识就是它不稳
- 多 seizure aggregation + 多 metric ensemble 是文献中默认的 robust 化策略
- Interictal-ictal spatial recapitulation 现象本身有强先验
- 我们不解决 SOZ localization；我们用"在 ictal 期得到的 channel-level 信息"作为 template naming 的一种 reference，且**不依赖任何单一 reference 准确**

---

## 3. 我们的修复方案：Multi-Anchor Consensus for Template Naming

### 3.1 核心 insight

**问题不是"anchor 之间一致"，问题是"template 命名在 anchor 之间一致"**。这两个完全不同。

- Anchor A1 排出来的 channel ordering 和 anchor A2 不同 → 可能（也是预期的）
- 但 "T0 和 A1 比 T1 更相关" 与 "T0 和 A2 比 T1 更相关" 这两件事是否一致 → **如果一致，T0 就是被一致命名的 T_aligned**

后者比前者要求宽松得多。Sentinel 上的不稳是 anchor-level 现象，不必然 propagate 到 naming-level。

### 3.2 多 anchor 设计

定义五个 anchor，每个对每个 subject 给一个 channel-level scalar：

| Anchor | 定义 | 物理含义（弱 claim） | 文献先验 |
|---|---|---|---|
| `A1: SIS_30s` | `[onset, onset+30s]` 内 channel-wise mean(z_ER)，跨 seizure median | 发作期总体能量参与度 | Bartolomei 2008 EI 的能量分量（去掉时延项） |
| `A2: SIS_early` | `[onset, onset+5s]` 早期能量，跨 seizure median | 发作早期招募 | EI 早期窗 |
| `A3: r_first_xover` | 每次 seizure 中 channel 第一次跨 z_ER 阈值的时间 rank，跨 seizure median rank | 粗 timing | Page-Hinkley alarm time |
| `A4: SIS_preictal_5s` | `[onset-5s, onset]` 即将发作时的能量参与，跨 seizure median | preictal recruitment | Aung 2026 preictal Sp-HFOs gradient |
| `A5: SOZ_label` | 临床二值标签（无 ordering）；用 0/1 作为 anchor scalar | presurgical 综合判断 | 所有 SOZ 工作的 ground truth |

#### 3.2a 关键合同

- **每个 anchor 都 cross-seizure aggregate**（median）：单次 seizure 不稳由 aggregation 处理
- **A1/A2/A4 用 continuous score 而非 rank**：避开 rank 在 score 接近时的 thrashing 问题
- **A3 仍用 rank 但 cross-seizure median rank**，作为 timing-based read-out（保留 PR-6-A 旧设计的味道）
- **A5 是无 ordering anchor**，用来检验"无 timing 信息时 template naming 是否还成立"

#### 3.2b ER 计算合同（A1/A2/A3/A4 共用）

- Bandpass: 4–250 Hz（Epilepsiae 主流 fs 1024 Hz 支持）
- `E_fast[n] = ∫_{60}^{100} |S(f,n)|² df`
- `E_slow[n] = ∫_{4}^{20} |S(f,n)|² df`
- `ER[n] = log(E_fast[n] / E_slow[n])`
- 滑窗 1s，步长 100ms
- Per-channel baseline z-score（baseline window `[-300s, -60s]`，排除 IED-dense epoch）→ `z_ER`
- A1/A2/A4 直接对 `z_ER` 求窗内 mean
- A3 用 Page-Hinkley CUSUM 找 first crossing，per-subject λ 由 baseline permutation 校定（false alarm <1/h）

**broad_ER (12–127 Hz) 不作为独立 anchor，作为 §5 sensitivity check 与 gamma 并报。**

### 3.3 Template naming via consensus voting

对每个 subject：

1. 对每个 anchor `A_k`，计算 `ρ_T0_k = Spearman(T0_centroid_rank, A_k)` 和 `ρ_T1_k`
2. 定义 `vote_k = sign(ρ_T0_k - ρ_T1_k)`：+1 表示 anchor A_k 支持 "T0 is aligned"，-1 表示支持 "T1 is aligned"
3. 跨 5 anchor 数票：`agreement_T0 = #{k : vote_k = +1}`
4. 命名规则：
   - `agreement_T0 ≥ 4/5` → T_aligned = T0, T_antipodal = T1, label `naming_strong`
   - `agreement_T0 = 3/5` → T_aligned = T0 但标 `naming_weak`，仍计入 cohort
   - `agreement_T0 = 2/5 or 3/5` 紧绷 → 标 `naming_ambiguous`，从主分析退出，case-series 另报

### 3.4 Stability 不是 gate，是 quality bin

每个 subject 同时产出两个 quality 维度：

- **Anchor-internal stability**: 5 个 anchor 各自的跨 seizure consistency（median pair-wise ρ across seizures）
- **Cross-anchor naming agreement**: §3.3 的 agreement 数

这两个量定义一个 2×2 quality plane：

| | High agreement | Low agreement |
|---|---|---|
| **High stability** | gold subjects, 主结论权重最大 | multi-modal seizure (Schroeder 2020 现象), case-series |
| **Low stability** | single-anchor 噪声但 consensus 救起来 → consensus 设计的核心证明点 | noise floor，退出 main，不剔除报告 |

**Cohort-level 主检验在所有非-ambiguous subject 上跑，但分层报告。**

### 3.5 与 SOZ 的差异化角色

A5 是 5 个 anchor 之一，参与 voting。但**额外**做一个独立检验：

- "用 anchor consensus 命名" vs "只用 SOZ overlap 命名" 的 naming concordance
- 这是 **PR-6-A-1 主 sanity check**：如果两种命名方式高度一致，PR-6-A 主线物理叙事立得住；如果显著不一致，说明 ER-based anchor 抓的不是 SOZ-related 的东西，PR-6-A 的物理解读必须重写

---

## 4. 为什么这个修复方案 work

### 4.1 直接对应 sentinel 失败的三个症状

| Sentinel 症状 | 原因 | 修复方案对应 |
|---|---|---|
| `548` 两次 seizure top-10 overlap = 0 | Rank ordering 在 score 接近时 thrash | A1/A2/A4 用 continuous score 而非 rank；多 seizure median 平滑 |
| `916` gamma vs broad ρ = -0.21 within-seizure | Single detector 单次 seizure 噪声 | 5 anchor consensus 不要求每对 anchor 之间高相关；naming-level agreement 而非 anchor-level agreement |
| early-detected 是 `other` channels | Early detection 的 false positive | 不挑 early detection；A1/A2 直接对全 window 求 mean(z_ER)，全通道连续打分；A5 SOZ 作为独立 channel-set anchor |

### 4.2 文献给的可行性预期

- Korzeniewska 2014：interictal-ictal recapitulation 在 cohort level 至少在某些子集上是真的——意味着 anchor 与 template 之间确实有共享信号可挖
- Schroeder 2020：seizure pathway variability 是 systematic（不是 random），有 circadian / multiday 结构 → cross-seizure median 不会把信号洗光
- Smith 2022：双向行波是普遍现象 → 我们 9-subject forward/reverse subset 不是统计 artifact，是机制现象 → naming consensus 在它们身上有清晰预测（T0 与 T1 在 anchor 上的方向应相反）

### 4.3 Statistical 角度

5 个 anchor 之间即便彼此独立 noise（worst case），naming agreement ≥ 4/5 在零假设下的概率只有 `2 × C(5,4) × 0.5^5 = 6/32 = 0.1875`。如果 cohort 50% subject 都达到 ≥4/5，这本身就是 strong evidence——naming 在 anchor 之间有 systematic 信号。

如果 anchor 之间相关（更现实的情况），相同的 naming agreement 比 0.1875 更易达到，但**正是这种相关本身证明 anchor 之间有共享 ictal 结构信号**——也是我们想要看到的。

### 4.4 与 PR-6 整体目标的匹配度

| 维度 | 旧 SOZ-only 设计 | 旧 single-anchor 设计 | 新 multi-anchor consensus |
|---|---|---|---|
| 给 T0/T1 命名能力 | 弱（SOZ 不带 ordering） | 强但脆 | 强且 robust |
| 对 ictal anchor 稳定性的依赖 | 无 | 高（已被 sentinel 证伪） | 弱（只要求 naming 稳定） |
| 与 PR-6 初衷匹配度 | 60% | 95% (脆) | 95% (robust) |
| 论文 framing 防守性 | 强 | 弱（容易被 SOZ open question 攻击） | 强（不 claim solve open question） |

---

## 5. 如果不 work 怎么办

### 5.1 三种失败模式与响应

| 失败模式 | 触发条件 | 响应 |
|---|---|---|
| **F1: Anchor 之间无共同信号** | Probe A 显示 5 anchor 之间 pair-wise ρ 中位数 < 0.2 | Anchor 抓的根本不是同一种东西 → multi-anchor consensus 失去意义 → 降级为只用 SOZ label naming，PR-6-A 改写为纯 spatial alignment paper（不涉及任何 timing claim） |
| **F2: Naming agreement 低** | Probe B 显示 sentinel + 扩展 5-8 subject 中 ≥4/5 agreement 的 subject 占比 < 50% | Anchor 之间有信号但不一致到能命名 → 退到 "naming_weak" 子集 only，cohort 缩小到 20-30%，主结论 underpowered → PR-6-A 降级为 case-series-driven exploratory |
| **F3: ER-anchor 与 SOZ 命名不一致** | Probe C 显示两种命名一致率 < 60% | ER anchor 抓的是非-SOZ-related 网络结构 → PR-6-A 物理叙事重写：不再 claim "templates align with ictal energy"，改 claim "templates capture network organization that is partially independent of clinical SOZ annotation" → 仍可发表但 framing 完全不同 |

### 5.2 Probe 阶段的 fail-safe 原则

**任何一个 Probe 给出明确负信号 → PR-6-A 不正式启动**，回到 Topic 1 §7 重新评估优先级（可能直接跳到 PR-6-C core node identification，绕过 ictal anchor 问题）。

**Probe 全部 pass → 正式撰写 PR-6-A archive plan**，把 multi-anchor consensus 设计、5 个 anchor 的完整定义、cohort 级别 statistical contracts、TDD 测试、与 PR-5 的 cohort 衔接全部锁死。

### 5.3 不允许的延伸行为

- **不允许在 Probe 中调 anchor 数量、参数到强行 pass**。Probe 是 feasibility 验证，不是 hyperparameter search。
- **不允许把 PR-6-A 主结论降到 sentinel-only**。如果 cohort 级别立不住就承认立不住。
- **不允许在 Probe 失败后偷偷换问题**。失败就回 Topic 1 §7 重新分配优先级。

---

## 6. Probe Phase: 三个 Probe 的合同

### 6.1 Probe A — Anchor Diversity & Internal Coherence

**目的**：验证 5 个 anchor 之间是否提供"基本相关但不完全冗余"的信号。如果全部高度相关（ρ > 0.85），多 anchor 集成是冗余；如果几乎无关（ρ < 0.2），anchor 之间没共同信号，方法学根基出问题。

**输入**：
- Sentinel subjects: `epilepsiae:548`、`epilepsiae:916`
- 扩展 subjects (Probe A 必须做)：随机抽取 PR-5-A retained main cohort 中另外 4 个 subject（剔除 sentinel，从 n=23 中均匀采样），共 6 个 subject
- 每个 subject 至少 3 次 seizure 有可用 onset annotation

**计算**：
1. 对每个 subject，按 §3.2 完整计算 5 个 anchor scalar (`A1` 到 `A5`)
2. 跨 channel 计算 anchor-pair 之间的 Spearman ρ（A5 是二值，用 point-biserial 转换或者 Mann-Whitney U-based proxy）
3. 报告：
   - 6×6 anchor pair-wise ρ matrix per subject (5 + diagonal handling)
   - Cohort median pair-wise ρ
   - `min(pair_rho)` 和 `max(pair_rho)` 跨 cohort 分布
4. 同时计算 anchor-internal stability：每个 anchor 跨 seizure 的 pair-wise Spearman median (`s_A1`, `s_A2`, ..., `s_A4`；A5 不需要因为不依赖 seizure)

**Pass 判据**：
- Cohort median pair-wise ρ ∈ [0.3, 0.85]（既不冗余也不无关）
- 至少 4/5 anchor 在 ≥ 4/6 subject 上 anchor-internal stability ≥ 0.5

**Fail 响应**：
- 全部 ρ > 0.85 → 用单 anchor + sensitivity，写更简单的 PR-6-A 替代 plan
- Cohort median ρ < 0.2 → F1，pivot 到 SOZ-only

**预算**：1.5 工作日

### 6.2 Probe B — Naming Convergence

**目的**：在 Probe A 通过的基础上，检验"naming-level agreement"在 sentinel + 扩展 cohort 上是否成立。这是 multi-anchor consensus 设计的最关键验证点。

**输入**：
- Probe A 的 6 个 subject（同样 cohort）
- 每个 subject 的 PR-2 stable templates（C0 / C1 centroid ranks）

**计算**：
1. 对每个 subject、每个 anchor `A_k`，计算 `ρ_T0_k = Spearman(T0_centroid_rank, A_k)`、`ρ_T1_k`
2. 计算 `vote_k = sign(ρ_T0_k − ρ_T1_k)`
3. 跨 5 anchor 数票，得 `agreement_T0`
4. 报告：
   - Per-subject 5×2 table of `(ρ_T0_k, ρ_T1_k)` for k in {1..5}
   - Per-subject `agreement_T0` 与 naming label (`strong/weak/ambiguous`)
   - Cohort 内 `naming_strong` 比例

**Pass 判据**：
- ≥ 4/6 subject 落在 `naming_strong` (agreement ≥ 4/5) 或 `naming_weak` (agreement = 3/5) 之一
- 且 `naming_strong` 占比 ≥ 50%

**Fail 响应**：
- `naming_strong` < 30% → F2，PR-6-A 降级为 exploratory case-series
- 命名翻转跨 anchor 严重（例如 1/4 split）→ 重新审视 anchor 选择，可能剔除 A3（rank-based）保留 4 anchor 重做

**预算**：1 工作日（Probe A 完成后）

### 6.3 Probe C — Anchor Consensus vs SOZ-only Naming Concordance

**目的**：检验 ER-derived anchor consensus 命名与单纯 SOZ-overlap 命名是否在物理含义上一致。这是 PR-6-A 物理叙事立得住的关键 sanity check。

**输入**：
- Probe B 的 6 个 subject（同样 cohort）
- 每个 subject 的 SOZ binary labels

**计算**：
1. **Method α (anchor consensus)**: §3.3 voting 得到 T_aligned
2. **Method β (SOZ-only)**: 对每个 template，计算 source channels (`rank_ch ≤ 33rd percentile`) 中 SOZ 通道的比例 `f_SOZ_T0_src`、`f_SOZ_T1_src`；T_aligned_β = arg max(f_SOZ_T_src)
3. 报告：
   - Per-subject `(T_aligned_α, T_aligned_β)` 一致性表
   - Cohort agreement rate
   - Disagreement subjects 的详细 anchor breakdown（识别哪个 anchor 与 SOZ 矛盾）

**Pass 判据**：
- α 与 β 命名一致率 ≥ 4/6 (66%)
- 非-`naming_ambiguous` subject 中一致率 ≥ 75%

**Fail 响应**：
- 一致率 < 50% → F3，物理叙事重写：ER-anchor 抓的是与 SOZ 解耦的网络结构。这本身可能是有趣的现象，但需要全新 framing
- 一致率在 50-66% 之间 → 边缘 pass，PR-6-A 写时显式承认这层 uncertainty

**预算**：0.5 工作日（Probe B 完成后；多数计算共享）

---

## 7. Probe Phase 工作量与时间线

| 任务 | 预算 | 前置 |
|---|---|---|
| ER pipeline 实现 (`src/ictal_anchor_extraction.py`) | 2 工作日 | 无 |
| Anchor 计算 (`src/anchor_compute.py`) | 1 工作日 | ER pipeline |
| Naming consensus 模块 (`src/template_naming.py`) | 0.5 工作日 | Anchor compute |
| Probe A 跑 + 报告 | 1.5 工作日 | 全部 src |
| Probe B 跑 + 报告 | 1 工作日 | Probe A pass |
| Probe C 跑 + 报告 | 0.5 工作日 | Probe B 共享 |
| Probe report 撰写 (`pr6a1_probe_report_2026-04-xx.md`) | 1 工作日 | Probe C |
| **Total** | **7.5 工作日** | |

**单 milestone**：Probe A/B/C 全部 pass → 正式启动 PR-6-A archive plan 撰写。

---

## 8. TDD 测试合同（Probe 阶段，硬锁 6 项）

```
tests/test_pr6a1_probe.py

T1. test_er_zscore_baseline_excludes_ied:
    构造 baseline window 含已知 IED epoch；z_ER 计算应自动排除该 epoch；
    剔除前后 σ_bl 比应 > 1.5。

T2. test_anchor_a1_a2_a4_continuous_score:
    构造 synthetic 3-channel signal: ch0 高能量 ictal, ch1 中等, ch2 baseline;
    A1/A2/A4 应给 ch0 > ch1 > ch2 的 score 顺序，**不论窗内具体 timing**.

T3. test_anchor_a3_first_crossing_rank:
    构造 ch0 在 onset+0.5s, ch1 在 onset+2s, ch2 在 onset+5s 跨阈值;
    A3 rank 应严格反映这个顺序 (rank 0 < 1 < 2).

T4. test_naming_voting_logic:
    构造 5 anchor 的 (ρ_T0, ρ_T1)：
    Case 1: 5 anchors 都 ρ_T0 > ρ_T1 → naming_strong, T_aligned=T0
    Case 2: 4/5 → naming_strong  
    Case 3: 3/5 → naming_weak
    Case 4: 2/3 split with 0 边界 → naming_ambiguous

T5. test_cross_seizure_aggregation_robust_to_outlier:
    构造 5 个 seizure，其中 1 个是 rapid-spread (所有 channel 同时活跃);
    median aggregation 后该 seizure 不应 dominate SIS。

T6. test_quality_bin_classification:
    构造 4 个 synthetic subject 覆盖 §3.4 的 2×2 plane;
    quality bin 函数应正确分到 4 个 bin，不漏不误。
```

---

## 9. 与其他 PR / Topic 的边界

- **PR-5-A retained cohort 是硬前置**：Probe 的 6 个 subject 必须从 main n=23 中抽
- **PR-4C cohort null 不影响**：PR-4C 是关于 template 几何变形，PR-6-A 是关于 template 命名
- **PR-6-B (π embedding) 在 Probe 后启动**：如果 PR-6-A-1 pass，π embedding 用来做 Obs 4 的 template 物理关系分析；PR-6-A-1 fail，π embedding 仍可独立做但叙事改
- **PR-6-C (core node identification) 与 Probe 解耦**：可以 Probe 阶段并行启动 PR-6-C，互不依赖

---

## 10. 主文档更新

Probe 完成后，无论 pass/fail，都在 Topic 1 主文档 `docs/topic1_within_event_dynamics.md` §7 增加一条：

```
- PR-6-A-1 (Probe phase): {DONE-PASS / DONE-FAIL / IN-PROGRESS}
  - 详见 docs/archive/topic1/pr6a1_probe_report_2026-04-xx.md
  - {一句话结论}
```

---

## 11. 已知 limitations & honest disclaimers

1. **6-subject probe cohort 功效有限**：Probe 不是 inferential analysis，是 feasibility 检查。任何"看起来 cohort-level 显著"的发现在 Probe 阶段都不发布
2. **A5 SOZ label 自身有 noise**：参考 Spring 2018 的 22% consistency。Probe C 的 disagreement 不 100% 归咎于 ER-anchor 错——可能 SOZ label 本身就漂
3. **A3 first-crossing 在 rapid-spread seizure 上仍然脆**：但作为 5 anchor 之一参与 voting，单点失败不致命
4. **跨 dataset (Yuquan) 暂不在 Probe 范围**：Yuquan seizure 数普遍 <5、annotation 质量我们已知较差。Probe pass 后正式 PR-6-A 再考虑 sensitivity replication

---

## 12. 一句话核心

**PR-6-A-1 不解决 SOZ localization 问题，也不解决"哪个 ictal anchor 最准"问题。它解决的是：在已知 single ictal anchor 不稳的前提下，能否通过 multi-anchor consensus 给两类 stable interictal templates 一个 cohort-consistent 的物理命名。**

Probe pass → PR-6-A 正式启动；Probe fail → 诚实承认这条路在我们 cohort 上走不通，回 Topic 1 §7 重新分配优先级。