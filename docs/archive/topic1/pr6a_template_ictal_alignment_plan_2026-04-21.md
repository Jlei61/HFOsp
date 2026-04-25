> **SUPERSEDED 2026-04-25**：本计划（single ictal anchor + Smith 2022 重现的 H1/H1' 主线）已被 PR-6 主线 pivot 取代。Sentinel `548/916` 已经把“跨 seizure 稳定 ictal onset rank”这条假设证伪（详见 `pr6a_step0-2_step3preview_review_2026-04-23.md` Step3-preview），并且文献（Schroeder 2020 / Wenzel 2017 / Pinto 2023 / Bailey 2021）也指出该方向在领域内本身高风险。**正式 plan-of-record**：[`docs/archive/topic1/pr6_template_endpoint_anchoring_plan_2026-04-25.md`](pr6_template_endpoint_anchoring_plan_2026-04-25.md)。本文档保留作为 pivot 前的科学背景与方法学讨论，但不再作为 PR-6-A 执行路径，§3-§9 中的 ER pipeline / CUSUM / sanity check 不再启动。

---

# PR-6-A：Interictal Template Semantic Alignment via Seizure Onset Propagation

先说一下文档定位：下面这份是对标 `docs/archive/topic1/pr5_template_recruitment_plan_2026-04-20.md` 的合同级计划，落盘后应放在 `docs/archive/topic1/pr6a_template_ictal_alignment_plan_2026-04-xx.md`。Topic 1 主文档 §7 增加 PR-6-A 条目；§7.9 的 KONWAC placeholder 已剥离为「未来模型层（不绑 PR 编号）」。

> 本计划占用 PR-6A 编号；PR-6 编号空间从此对应 PR-6A/B/C/D/E 数据发现序列；KONWAC v2 已剥离到主文档 §7.9 未来模型层，不绑 PR 编号。
>
> 2026-04-23 阶段性执行/审阅状态见 `docs/archive/topic1/pr6a_step0-2_step3preview_review_2026-04-23.md`：Step0-2（EEG-aware baseline clip + sentinel Step2 图）有条件验收通过；Step3 `t_ER_onset` 当前只接受为 preview-only，不进入 H1/H1' / sanity 正式叙事。

---

## 1. 科学问题与判断边界

### 1.1 核心问题

在 PR-2 / PR-2.5 已锁定的 `30/30` stable interictal templates 上，是否存在至少一个 template，其 channel rank geometry 在统计上复现 ictal onset 的 channel-wise earliest activation rank？

对 9 个 `inter-cluster r < -0.5` 的 forward/reverse subject，双 template 是否呈现 Smith 2022 双向行波的"一个同向、另一个反向"signature？

### 1.2 这个 PR **不**回答

- 模板是否随发作邻近发生几何变形 → PR-4C 已封板为 null
- 模板招募率是否在 post-ictal 升高 → PR-5-B 已 Bonferroni-pass
- SOZ 通道的慢调制优势 → Topic 3 PR-1/PR-2 的独立线

### 1.3 已发表先验（决定了假设的 prior strength）

Smith et al. 2022 eLife: "the majority of IEDs are traveling waves, traversing the same path as ictal discharges during seizures, and with a fixed direction relative to seizure propagation... were bidirectional, with one predominant and a second, less frequent antipodal direction" — MEA μm-scale 的双向行波直接预测我们 k=2 + 9/30 forward/reverse 的几何。

Korzeniewska et al. 2014: "In patients with focal ictal onsets, the patterns of propagation recorded during pre-ictal... and interictal... intervals were very similar to those recorded during seizures" — SdDTF 在 HFO band 展示 interictal-ictal propagation 高度重合。这是 H1 最强的方法论先验。

Liu et al. 2019: "Regions exhibited highly stable preferences to appear upstream, intermediate, or downstream in spike propagation sequences... DP-Stability was 0.88 ± 0.07" — 他们用 Degree Preference (Spearman) 衡量 interictal spike propagation 跨 30-min segment 的稳定性，方法学与我们 r_sz 跨 seizure 的稳定性检验同构。

Bailey et al. 2021: "Spike onset and seizure onset seemed to be distinct networks in most cases" — **负面文献**，显式挑战 H1。把 H1 cohort null 的可能性留在桌上，不预设方向。

Aung et al. 2026 Epilepsia: "Spectral similarity between Sp-HFOs and ictal HFA increases as seizure onset approaches, indicating dynamic preictal evolution" — 最新证据：HFO 在 interictal → ictal 是**渐进式**转变，而不是两态切换。支持"interictal template 里的 HFO 应与 ictal HFA 共用 generator"。

---

## 2. 数据合同

### 2.1 Cohort 定义（严格分层）

**主 cohort（Epilepsiae only，main config）**：

- 硬前置：属于 PR-5-A retained main cohort（n=23）
- 必须有 ≥3 个 seizure onset annotation 可用（Epilepsiae 的 `seizure_onset` 列）
- 必须 k=2（`stable_k = 2`）— 确保 T0/T1 二元对比语义清晰
- 预期 cohort 大小：n ≈ 15–18（PR-5-A main 23 中扣掉 k=4/k=6 的 2 个 + 少 seizure 的 3–5 个）

**辅 cohort（sensitivity）**：

- Yuquan PR-5-A retained (n≈15)：所有 subject 都走同样 pipeline 但不纳入 cohort-level inferential。作为**replication hint**，不参与 α 分配。
- 理由：Yuquan 录制更短、seizure 数普遍 <5、onset annotation 质量我们之前已知较差。强行 pool 会污染 main 的判读。

**Case-series 单独报告**：

- k=4 subject（`818`）
- k=6 subject（`zhangjinhan`）
- 9 个 forward/reverse subject 的子集（和主 cohort 重叠），单独跑 H1'

### 2.2 通道对齐合同

关键约束：interictal template 只在 `n_participating ≥ min_part`（当前 `min_part = 5`）的通道子集上有 rank；ictal onset rank 原则上在所有 recording channels 上可测。

- 定义 `C_interictal`：subject 的 PR-2 template 定义所用通道集合
- 定义 `C_ictal_reachable`：ictal onset pipeline 成功提取 onset time 的通道集合（见 §3）
- **对齐集合 `C_common = C_interictal ∩ C_ictal_reachable`**
- 硬门槛：`|C_common| ≥ 5`，否则该 subject 退出 main cohort（case-series 另报）
- 所有 template centroid rank 和 r_sz 都**在 C_common 上重新 re-rank**（保持 rank 连续性），不是在原始集合上截断

---

## 3. Ictal Onset Rank 提取 Pipeline（核心方法学）

这是你关心的核心问题："如何定义+提取 channel-wise earliest activation，不被传播带来的大范围同步高信号干扰"。答案是 **4 层防御**，每层针对一种污染源。

### 3.1 Layer 1 — Feature choice: Energy Ratio (ER) — 双配置并列

**为什么不用 amplitude crossing**：Seizure 的大 amplitude 是 propagation 后的事，amplitude crossing 会被晚起但大振幅的通道率先触发。

**为什么用 ER**："The Page-Hinkley algorithm provides a detection time Nd for each brain structure if involved in the generation of a rapid discharge" — ER 对频谱结构变化敏感（fast oscillation replacing slower background），不是对 amplitude 本身。"EI was developed by Bartolomei et al. to quantify the appearance of high frequency oscillations by averaging the energy ratio between high and low frequency bands over the lag in seizure onset time relative to a reference point"。

**两套配置写死，从 Step 2 起并行跑到底**（命名硬绑定，绝不接受运行时第三套自定义频段）：

| key | E_fast | E_slow | 定位 |
|---|---|---|---|
| `gamma_ER` | `∫_{60}^{100} \|S(f,n)\|² df` | `∫_{4}^{20} \|S(f,n)\|² df` | **主配置**（HFO-centered pipeline，LVFA 期望在 gamma 带最早出现） |
| `broad_ER` | `∫_{12}^{127} \|S(f,n)\|² df` | `∫_{4}^{20} \|S(f,n)\|² df` | **sensitivity 配置**（Bartolomei 2008 原版，覆盖 burst-suppression / delta brush 等非 LVFA onset 模式） |

通用约束（两套共享）：

- 滤波：bandpass 4–250 Hz（Epilepsiae 主流 fs 1024 Hz 可支持）
- `ER[n] = log(E_fast[n] / E_slow[n])`（log 稳定化，避免 slow band 低谷时 ratio 爆炸）
- 滑窗 1 s，步长 100 ms

**关键约束（合同级，不可松动）**：

- `gamma_ER` 与 `broad_ER` **从 Step 2 ER 提取起一直到 Step 7 H1' 全程并行跑**；所有下游统计（r_sz / sanity / H1 / H1'）在两套配置上各跑一遍并行报告。
- **禁止按 subject 后验切换主配置**。即不允许"这个 subject gamma_ER 失败、改用 broad_ER"。两套独立产出独立的 r_sz / s_sz / cohort 文件，独立做 H1 推断。
- 主配置切换只发生在 cohort-level Sanity gate（§4.3 / Step 5）：若 gamma cohort sanity FAIL 而 broad cohort sanity PASS，则在 archive 中正式切换主配置为 broad，**整体重跑下游**——不做 per-subject mix。

### 3.2 Layer 2 — Per-channel baseline z-score（抗 identity-bias）

**污染源**：有些通道 HFO 基线就高（Topic 1 §3.2 的 identity-bias 86% 在 seizure 里同样存在）。ER 绝对值大的通道会先触发绝对阈值，把它误判为 onset。

**解决**：每个 channel 独立 z-score normalize：

- `baseline window = [-300s, baseline_end_sec]` 相对 **clinical onset**，其中
  `baseline_end_sec = min(0, eeg_onset_rel_sec) − 60s`
  （即对 EEG-onset 与 clinical onset 中较早的一个再后退 60s buffer）。
  当 `eeg_onset_epoch` 缺失时退化为 legacy `[−300s, −60s]`。Epilepsiae 队列里
  `523/540` 个 seizure 拥有 `eeg_onset_epoch` 且其中 ~91% 早于 clinical onset，
  因此这条合同实际生效在大多数 seizure 上。
- 排除窗内所有 "known IED peaks"（用 Epilepsiae 已有 spike annotation 或从 HFO detection 反推）
- **Baseline-invalid 规则（不回退）**：若 EEG-aware clip 之后 baseline 有效长度 `< 60s`，
  该 seizure 整体记为 baseline-invalid，**不**回退到 legacy `[−300, −60]`
  窗（那等于把要规避的 pre-ictal / electrographic onset 重新吃回来）。
  下游 rank / CUSUM 必须直接丢弃该 seizure。
- 若某通道在合法 baseline 内 `< 60s` valid 样本 → 该 seizure 该通道退出（不参与 rank）
- `z_ER[n] = (ER[n] − μ_bl) / σ_bl`
- 对所有通道 baseline-normalize 后才进入下一步

这层是原始 EI 没做的，对我们尤其重要——因为我们已知 identity-bias 大。

### 3.3 Layer 3 — Page-Hinkley CUSUM 找 change-point

**为什么不用阈值穿越**：阈值穿越对噪声敏感、对渐进 onset 不稳健。"The key idea of the Page-Hinkley algorithm is to perform a test on the mean of ER statistic by building a quantity (cumulative sum U)... Decision that a significant change has occurred is taken at alarm time"。CUSUM 累积统计量，对早期缓慢增长敏感，并且一旦 detect 就给出**第一次偏离 baseline 的时点**（而不是 peak 的时点），这正好对应 onset。

**合同**：

- 对每个通道的 `z_ER[n]` 跑 Page-Hinkley CUSUM：
  - `U[n] = max(0, U[n-1] + z_ER[n] − bias)`
  - `bias = 0.5`（Bartolomei 2008 默认）
  - `M[n] = max_{k≤n} U[k]`；alarm 时刻 `n_d` = 第一个 `n` 满足 `M[n] − U[n] ≥ λ`
  - `λ` 阈值通过 per-subject permutation 定：在 baseline window 上滚动跑 CUSUM，取 `False positive rate < 1/hour` 的 λ
- detection window = `[onset_clinical − 5s, onset_clinical + 30s]`
  - 前 5s 容忍 clinical annotation 的不精确
  - 后 30s 覆盖大多数 onset pattern 的 recruitment 窗；>30s 通常是 propagation 阶段，不是 onset

### 3.4 Layer 4 — Tie-handling 与 "unresolved onset" 标记

**污染源**：部分 subject 的 seizure 会在 <500 ms 内招募 >60% 通道（rapid-spread pattern）。如果强行 ranking 会得到一堆 tied ranks，噪声占主导。

**合同**：

- 对 `n_d` 做 ranking（最小 `n_d` = rank 0）
- **Fractional rank for ties**：`Δt < 50 ms` 的通道赋予平均 rank
- 若某次 seizure 中 `>60%` 通道在 `[clinical_onset, clinical_onset + 1s]` 内被 detect → 该 seizure 标记 **`onset_tied`**：
  - 保留在 sensitivity analysis
  - 不参与主 r_sz 中位数计算
- 若某次 seizure 中 `<30%` 通道有 valid `n_d`（多数 channel "unreached"）→ 该 seizure 标记 **`onset_unreached`**，完全排除
- Subject 至少 3 个 non-tied non-unreached seizures 才 qualifies for main cohort

### 3.5 Per-subject r_sz 构造与 stability gate（双 cohort，无加权档）

- 对 qualifying subject，每次 seizure 得到一个 per-channel rank vector `r_{sz,s}`（s 为 seizure index）
- 主 r_sz = **median rank per channel across seizures**
- **Stability gate**：计算所有 seizure pair (s, s') 的 Spearman ρ on r：
  - `s_sz = median{ρ(r_{sz,s}, r_{sz,s'})}` over all pairs

**双 cohort 划分（合同级，不可加权）**：

| cohort key | 入选条件 | 用途 |
|---|---|---|
| `strict_cohort` | `s_sz >= 0.5` | **主分析**（H1 主结论、sanity 主判据均在 strict 上跑） |
| `relaxed_cohort` | `s_sz >= 0.3`（含 strict 全部成员） | **sensitivity 分析**（H1 / sanity 同步在 relaxed 上跑作为对比） |

- `s_sz < 0.3` → subject 完全退出，不进任何 cohort（"onset rank 本身都不稳，template 对比无意义"）
- **不存在加权档**：strict 与 relaxed 是两个独立 cohort，统计永远在 cohort 内 unweighted；从不在同一检验里把不同 stability 的 subject 加权汇总。
- Wilcoxon / sign test 在 strict 上跑主结论，在 relaxed 上跑 sensitivity，两份结果并列报告（每套 ER 配置 × 每个 cohort = 4 套数）。

这个门槛直接对标 Liu 2019 的 "DP-Stability was 0.88 ± 0.07"——他们的 stability 在 interictal spike propagation 上是 0.88；我们允许 ictal 场景降到 0.5（strict）/ 0.3（relaxed）是因为 ictal seizure 数通常远少于 interictal segment，统计上噪声更大。

### 3.6 Sentinel cohort（Step 2-4 共用，落定于 2026-04-21 Step 2）

PR-6-A pipeline 在进入全 cohort 复跑前，先在 2 个 sentinel subject 上做 visual inspection，避免 pipeline-level bug 污染整张 cohort 表。Sentinel 选择已在 Step 2 落定，后续 Step 3、Step 4 的 sentinel 报告（CUSUM alarm 表 / baseline false-alarm / per-band r_sz）必须复用同一对 subject，**禁止在 step 之间偷换 sentinel**。

| key | subject | k | PR-5-A retained main | 9-subset (forward/reverse) | inter-cluster r | n_seizures (PR-5-A gate-eligible) | 入选理由 |
|---|---|---|---|---|---|---|---|
| `sentinel_A` | `epilepsiae/548` | 2 | ✓ | ✓ | −0.62 | 31 | 9-subset 中 seizure 数最多，最直接对应 Smith 2022 双向行波 sanity |
| `sentinel_B` | `epilepsiae/916` | 2 | ✓ | ✗ | −0.37 | 51 | 普通 k=2、reproducibility grade=strong、不在 9-subset 内，作为"非双向行波 baseline" |

Sentinel 选择来源：交叉 [results/interictal_propagation/pr5a_novel_template_gate.json](../../../results/interictal_propagation/pr5a_novel_template_gate.json) 中 main retained 名单（n=23）与 [results/interictal_propagation/per_subject/](../../../results/interictal_propagation/per_subject/) 下每个 subject 的 `adaptive_cluster.inter_cluster_corr_matrix` + `time_split_reproducibility.full_data_forward_reverse_pairs`；筛选 `stable_k=2` 后按 9-subset / 非 9-subset 各取一个 seizure 数最大者。

Step 2 sentinel 已产出每个 sentinel × 每套 ER 配置 × 2 个 seizure 的 z-ER trace 图，路径 `results/interictal_propagation/ictal_alignment/_sentinel_step2/<subject>_<seizure_idx>_<gamma_ER|broad_ER>.png`，并附 [sentinel_step2_summary.json](../../../results/interictal_propagation/ictal_alignment/_sentinel_step2/sentinel_step2_summary.json)（含每张图 focal vs nonfocal 的 pre30s / post30s z-ER 中位数 max）。

---

## 4. 假设与统计合同

### 4.1 主假设 H1（Smith 2022 一般预测）

- **定义**：对每个 strict_cohort 内 subject（k=2, `s_sz >= 0.5`），计算：
  - `ρ_T0_sz = Spearman(T0_centroid_rank, r_sz)` on C_common
  - `ρ_T1_sz = Spearman(T1_centroid_rank, r_sz)` on C_common
  - `ρ_max = sign-preserving argmax by |ρ|`；取该值为 `ρ_aligned`
- **Cohort-level test（双 cohort × 双 ER 配置共 4 套数）**：
  - **主分析在 strict_cohort 上**，sensitivity 在 relaxed_cohort 上；从不在加权样本上跑（§3.5 双 cohort 合同）。
  - 主检验：one-sample Wilcoxon on `ρ_aligned` against 0（H1 预测 ρ_aligned 中位数 > 0）
  - 次检验：sign test（n subjects with ρ_aligned > 0.2 / total qualifying）
  - 报告 `|ρ_aligned|` 中位数和 IQR
- **α 分配**：
  - 主 cohort Bonferroni：α = 0.05 / 2 tests = 0.025（Wilcoxon + sign test 各一）
  - **PASS 判据（strict 上跑）**：Wilcoxon p < 0.025 且 sign-test 超过 70% 正向
  - relaxed cohort 不分独立 α，只作为方向一致性 sensitivity 报告
  - k≠2 subject（`818`、`zhangjinhan`）走 case-series，不进任何 cohort 推断（详见 Step 6 三态决策）

### 4.2 备择假设 H1'（Smith 2022 双向预测，9-subject subset） — pre-registered secondary

H1' 是**预先注册的次级检验（pre-registered secondary test）**：不进 H1 的 α 池、不分 α 阈值，只做方向性 / sign test 的强弱判读，**不承载主结论**。判读语言全程使用 "direction supported / sign count" 等方向性表达，绝不引用 H1 的 Bonferroni-corrected 阈值。

在 `inter-cluster r < -0.5` 的 9 个 subject 子集上（与 strict_cohort 取交集，即只在 strict 内的 9-subset 成员上跑，避免拿非 strict subject 拉读数）：

- 两 template 应与 r_sz 呈符号相反的相关
- 定义 **bidirectional score**：`B = −(ρ_T0_sz × ρ_T1_sz)`（若符号相反且绝对值都大，B 大）
- H1' 预测 B > 0 在大多数 9-subset subject 上成立
- Sign test：`# subjects with ρ_T0_sz × ρ_T1_sz < 0` / n（n = 9-subset ∩ strict_cohort 的实际样本数）
- 报告 individual `(ρ_T0, ρ_T1)` 散点图与 sign 计数；**不出 p-value 主结论**

这是 PR-6-A 里**最直接对应 Smith 2022 的检验**。9 个 subject 功效极有限（sign test 理论最小 p ≈ 0.004），所以从设计上就把 H1' 钉为方向性次级证据：不被 H1 alpha 池吸收，也不会被"显著/不显著"二分判读。

### 4.3 Sanity check（pipeline 可信度 gate） — i vs e only

**必须先通过这一关才能解读 H1/H1'**。用 Epilepsiae `focus_rel` 标注，**只看 `i` 与 `e` 两值**：

- `i` = focal（pipeline 预测 rank 更小 = 更早激活）
- `e` = extra-focal（pipeline 预测 rank 更大 = 更晚激活）
- `l` = lesion 通道**完全不进 sanity 集合、不进检验**（cohort 内 `l` 通道数普遍偏少，混入会污染 i vs e 的方向读数）

主 sanity 检验：

- Per-subject：`i` 通道的 r_sz rank 分布 vs `e` 通道的 r_sz rank 分布 → Mann-Whitney U
- Cohort-level：每个 subject 取 `median(r_sz | i) − median(r_sz | e)` 做 paired Wilcoxon
- **PASS 判据**：cohort Wilcoxon p < 0.05 且方向是 `i < e`（focal channel rank 更小 = 更早激活）

附录展示（不进判据）：

- 在 archive §11 附录里给一张 `l` 通道的 r_sz 分布描述图，便于观察 lesion 通道在 onset rank 中的相对位置；这张图**仅描述**，不进 sanity gate、不影响 PASS/FAIL。

**如果 sanity check FAIL**：pipeline 有 bug 或 SOZ label 有问题，**H1/H1' 结果不发布**。Step 5 hard gate 三态决策（gamma sanity / broad sanity）写在 step 计划 Step 5 节，本节不再展开。

### 4.4 失败合同

| 触发条件 | 响应 |
|---|---|
| `strict_cohort` 任一 ER 配置 `n < 8` | Step 4 HARD GATE 1 触发：PR6A 降级为 case-series；archive 写明并在主文档 §7 标注 underpowered |
| 任一 ER 配置 cohort 级 `tied_fraction > 0.4` | Step 4 HARD GATE 2 触发：把 cohort 拆 `tied / resolved` 分别上报，不能默认 pool |
| Sanity check (§4.3) gamma cohort null 但 broad cohort PASS | 在 archive 中正式切换主配置为 broad，重跑 Step 4 cohort 报告后再继续；不做 per-subject mix |
| Sanity check (§4.3) gamma + broad 两套都 null | PR6A 进入 BLOCKED 状态：archive 写明 sanity FAIL、不发布 H1，回 Step 2/3 重新调参（建议先调 baseline window，再调 λ） |
| H1 strict cohort Wilcoxon p > 0.025 且 `\|ρ_aligned\|` 中位数 < 0.25（两套 ER 配置均如此） | Smith 2022 在 HFO population event 层面不复现；诚实公开 null，不重跑 |
| H1 PASS 但 H1' 9-subset sign-count 不强 | H1' 仅作方向性描述，不作为主结论一部分；不再用"显著/不显著"语言 |
| `onset_tied` seizure 比例 > 40%（跨 cohort） | Epilepsiae 多数 seizure 是 rapid-spread 模式；rank extraction 时间分辨率限制；在 `onset_tied` vs `onset_resolved` 两层上各跑一次，分别报告 |
| H1 与 sanity 同时 PASS 但 H1' 显示 `ρ_T0` 和 `ρ_T1` 同号且都 > 0 | 两 template 都同向对应 ictal onset，inter-cluster 负相关是在非-onset-related 维度上发生；Smith bidirectional 不成立，但"模板锚 ictal"成立 → intermediate outcome，按方向性语言记录 |

---

## 5. 代码入口（新建文件）

```
src/ictal_onset_extraction.py        # §3 pipeline
    ├─ compute_er(signal, fs, fast_band, slow_band, win)
    ├─ baseline_zscore_er(er, baseline_window)
    ├─ page_hinkley_cusum(z_er, bias, threshold)
    ├─ extract_onset_rank_per_seizure(subject, seizure_idx) → (n_d dict, tied_flag, unreached_flag)
    └─ build_rsz_per_subject(subject) → (r_sz_median, s_sz_stability, qualifying_flag)

src/template_ictal_alignment.py      # §4 统计
    ├─ compute_rho_template_vs_rsz(template_centroid, r_sz, c_common)
    ├─ h1_cohort_test(rho_aligned_list) → (wilcoxon_p, sign_p, median_abs)
    ├─ h1prime_bidirectional_test(subset_9) → (B_list, sign_p)
    └─ sanity_check_focus_rel(r_sz, focus_rel_labels) → (mw_p_per_subject, cohort_wilcoxon_p)

scripts/run_pr6a_onset_rank.py       # 一次性批跑全 cohort
scripts/run_pr6a_alignment_stats.py  # 输出 archive 数值
scripts/plot_pr6a_alignment.py       # cohort 散点 + 9-subset individual + sanity panel
```

---

## 6. TDD 测试合同（锁 9 项，按 step 归属分三组）

测试文件：`tests/test_pr6a_ictal_onset.py`

| 组 | 归属 step | 测试 ID | 范围 |
|---|---|---|---|
| **数据层** | Step 2-4 | T1, T2, T3, T4, T5, T6 | ER 提取 / baseline z-score / CUSUM / tie 标记 / C_common re-rank / 双 cohort gate |
| **Sanity 层** | Step 5 | T8 | i vs e sanity gate |
| **推断层** | Step 6-7 | T9, T7 | H1 cohort 路由 / H1' 双向方向性 |

### 数据层（Step 2-4 共用）

```
T1. test_er_channel_independence:                          [Step 2]
    构造 2-channel synthetic signal，channel 0 在 onset+1s 注入 80Hz burst，
    channel 1 在 onset+3s 注入同等 burst。验证 n_d[ch0] < n_d[ch1].
    必须对 gamma_ER 与 broad_ER 两套 band 配置都通过.

T2. test_baseline_zscore_removes_identity_bias:            [Step 2]
    2 channels，ch0 baseline ER 始终比 ch1 高 5 dB，但 seizure 时两者都升高
    同等幅度。未做 z-score 时 ch0 n_d < ch1 n_d（错误）；做 z-score 后应
    两者 n_d 接近.

T3. test_page_hinkley_no_false_alarm_on_pure_baseline:     [Step 3]
    仅 baseline segment，CUSUM λ 按 §3.3 permutation null 设定后，
    false positive rate < 1/hour. λ 必须 per-subject + per-band 标定.

T4. test_onset_tied_flagging:                              [Step 3]
    synthetic seizure, 80% channels 在 onset+0.2s 同时 recruit.
    pipeline 必须标记 onset_tied=True.

T5. test_c_common_rebalancing:                             [Step 4]
    C_interictal 和 C_ictal_reachable 部分重合 (|common| = 7);
    template centroid 被正确 re-rank 到 7-channel space;
    r_sz 同样 re-rank 到 7-channel space.

T6. test_dual_cohort_gate:                                 [Step 4]
    构造 subject with 5 seizures:
      (a) 全部一致 → s_sz ≈ 0.85 → strict_cohort=True, relaxed_cohort=True
      (b) 3 一致 + 2 随机 → s_sz 落在 [0.3, 0.5) → strict_cohort=False,
          relaxed_cohort=True（不被剔除，但只进 sensitivity）
      (c) 全部随机 → s_sz < 0.3 → strict_cohort=False, relaxed_cohort=False
          （完全退出，禁止任何中间档加权标记）
    断言：不存在权重字段，仅有两个布尔 cohort 标记.
```

### Sanity 层（Step 5）

```
T8. test_sanity_check_i_vs_e_only:                         [Step 5]
    模拟 sanity check fail 情景（focal `i` vs extra-focal `e` rank 无差异），
    sanity 函数应返回 pass_flag=False，并阻止 H1/H1' 发布.
    断言：l 通道既不进 sanity 通道集合，也不进检验输入；其存在不会改变
    pass_flag 与方向输出. gamma_ER 与 broad_ER 各跑一份.
```

### 推断层（Step 6-7）

```
T9. test_k_neq_2_routed_to_caseseries:                     [Step 6]
    k=4 subject (818) 与 k=6 subject (zhangjinhan) 走 case-series path，
    不进入 strict_cohort 也不进入 relaxed_cohort 的 H1 推断；
    断言：H1 cohort 列表中不含这两个 subject id.

T7. test_h1prime_bidirectional:                            [Step 7]
    synthetic subject with ρ_T0_sz = +0.7, ρ_T1_sz = −0.6
    → B = 0.42 > 0, 计入 "bidirectional supported" 样本.
    断言：H1' 输出仅含 sign 计数与方向，不含 alpha-corrected p 阈值判读.
```

---

## 7. 与 PR-5 / PR-4C / Topic 3 的边界

- **PR-5-A retained cohort 是硬前置**：不重新做 novel-template gate；所有通道、template、rate 定义继承 PR-5 合同。
- **不动 PR-4C 的封板结论**：PR-4C 检验的是"模板内部几何随发作邻近变形"（cohort null），PR-6-A 检验的是"模板几何与 ictal onset geometry 的跨时态一致性"。两者是不同问题，PR-4C null 不预测 PR-6-A 结果。
- **不进入 Topic 3 PR-6-C core source/sink 工作**：PR-6-C 是从 identity-bias 出发挑选核心通道；PR-6-A 是验证已定义 template 是否锚 ictal。两者并行、结论独立可叠加（"锚 ictal 的 template 的 core source 是否也是 SOZ" 是 PR-6-A × PR-6-C 的交叉 bonus，不纳入任一方的主结论）。

---

## 8. 预期时间线（按 2026-04-21 起算）

> 注：详细的 step 间 hard checkpoint 与执行顺序写在 `.cursor/plans/pr6a_step_decomposition_*.plan.md`；本节仅给粗粒度 week 级时间预估。

| Week | 交付（对应 step） |
|---|---|
| 1 | `src/ictal_onset_extraction.py` 完成 T1-T3 + sentinel cohort 选定（Step 2-3）；2 个 sentinel subject 上 gamma_ER + broad_ER 双配置 visual PASS |
| 2 | T4-T6 + per-subject λ 标定 + 双 cohort gate（Step 3-4）；Step 4 hard gate（strict n>=8 + tied_fraction）必须通过才能进 Step 5 |
| 3 | T8 sanity i vs e（Step 5） + `src/template_ictal_alignment.py` H1 主分析（Step 6）；archive 中间报告 `pr6a_onset_extraction_validation_2026-04-xx.md` |
| 4 | H1' 双向次级 + 全 cohort 复跑 sensitivity（Step 7） + 归档与主文档回写（Step 8）|

验收条件：sanity check PASS（gamma 或 broad 至少一套）+ 9 项 TDD 测试三组全绿 + archive plan §11 复跑结论同步更新 + 主文档 §7.10 一句话结论更新。

---

## 9. 几个你可能追问的技术点，我先写答案

**Q1: 为什么不直接用 clinician 标的 SOZ 通道作为 "ictal onset channel set"，跳过 pipeline？**

因为 SOZ 是**二值标签**，不给 rank。我们要的是 per-channel timing rank，Smith 2022 / Korzeniewska 2014 的预测也是关于 ordering，不是关于 set membership。SOZ 用在 §4.3 sanity check 足够了。

**Q2: 为什么不用连续 `π` (PR-6-B) embedding 上的 template alignment，而要用离散 centroid rank？**

两个原因：(a) π embedding 还没做 TDD；PR-6-A 必须能在 PR-6-B 结果出来前独立运行；(b) centroid rank 是我们已经信任的量（PR-2.5 strong/moderate 全 cohort 覆盖），改成 π 会引入新的不确定性来源，妨碍归因。PR-6-B 完成后可以重跑 PR-6-A 做 robustness check。

**Q3: 如果某个 subject 的 template 是 k=2 但 baseline rate 把它驱成 `dom_frac > 0.85`（一个模板极其稀少），第二个 template 的 centroid rank 是否可信？**

不可信到需要门槛：`min events per cluster ≥ 200` 才能定义稳定 centroid。PR-2 本身有这个门槛但 PR-6-A 要显式重检。若某 subject T1 events < 200 → H1' 退出（9-subset 会损失 subject），但 H1 仍可用较大 cluster 的 centroid 做 ρ_aligned。

**Q4: seizure onset annotation 本身有 ±几秒的不准确，这会多大程度影响结果？**

§3.3 的 detection window `[onset−5s, onset+30s]` 就是为此而设。ER CUSUM 的 alarm time `n_d` 是相对 detection window 起点的，不依赖 annotation 精准到秒。Annotation 误差只会污染 baseline window——为此 §3.2 已把 baseline 末端改成 `min(0, eeg_onset_rel_sec) − 60s`（取 EEG/clinical onset 中较早者再后退 60s buffer），不可触达的 60s buffer 覆盖大多数残余 annotation 误差，而 EEG-onset 提前的情况由动态 clip 直接处理；不满足 60s 最小有效长度的 seizure 直接判 baseline-invalid，不回退。

**Q5: Epilepsiae 部分 subject 是 ECoG，seizure 有 DC shifts；gamma ER 会不会漏掉这些？**

"SEEG signature of LVFA was found to be largely observed (~70%) and the other onset patterns like burst-suppression and delta brush patterns were very rare"，即便非 LVFA onset 也有 20–30%。这正是 §3.1 双配置并列的动机：`gamma_ER`（60–100 Hz）瞄准 LVFA 主流模式，`broad_ER`（12–127 Hz Bartolomei 原版）覆盖 burst-suppression / delta brush 等非 LVFA onset。两套**从 Step 2 起一直并行跑到 Step 7**，不是"gamma 失败再切 broad"的 fallback；只有 cohort-level sanity gate 才允许在 archive 里正式切换主配置（详见 §4.4 与 step 计划 Step 5）。

---

## 10. 设计拍板记录（Step 1 已锁，不再开放）

下列设计点已在 Step 1 archive 重写中拍板，落入正文 §3.1 / §3.5 / §4.2 / §4.3。不再作为开放问题。

| # | 设计点 | 决定 | 落点 |
|---|---|---|---|
| 1 | ER fast band | `gamma_ER` (60–100 Hz) 主 + `broad_ER` (12–127 Hz) sensitivity，**全程并行**，禁止 per-subject 后验切换 | §3.1 / §4.4 / §6 T1 |
| 2 | Stability gate 阈值 | 双 cohort：`strict_cohort` (`s_sz >= 0.5`) 主 + `relaxed_cohort` (`s_sz >= 0.3`) sensitivity；**移除中间加权档** | §3.5 / §4.1 / §6 T6 |
| 3 | Sanity check 标签 | **i vs e only**；`l` 通道完全不进判据，仅在 §11 附录描述性展示 | §4.3 / §6 T8 |
| 4 | H1' 9-subset 性质 | **pre-registered secondary test**，不分 α，只出方向性 sign test 与散点；判读语言全程使用方向性表达，不再使用"非主结论的探索性试验"这类含糊措辞 | §4.2 / §6 T7 / §4.4 |