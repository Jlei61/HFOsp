# Topic 5 — Ictal-Template-Echo Gate (Stage 1) Design Spec (2026-06-08)

> **状态**：设计稿 **v2**（applied conditional-accept amendments 1–12, 2026-06-08：ER-proxy 正名 / LOO 去锚 / subject-level primary / 更强 null / MIN_CH=8 / atlas-quality gate / k=2 primary / H2 循环性 + tie / instrument-quality Stage-2 分支），**pending user re-review** before writing implementation plan。
> **Topic**：把"间期那条频繁出现、形状固定的高频传播通路"与"**当前可用的发作通道顺序代理**（ER/atlas 派生的 ictal rank proxy，**不是**新搭的真实招募时序图）"在 **cohort 层面合并**对齐——问"这个现成的发作顺序代理是不是富集了间期模板的排序"。
> **定位**：这是 topic5 下一步的 **Stage 1（闸门）**。它**不**搭新仪器；只把上一轮已算好的 per-seizure 相似度跨病人合并，决定值不值得投入 Stage 2（真正发作 EEG 的多特征招募时序图）。**Stage 1 是"要不要建更干净仪器"的闸门，不是对真实发作传播顺序的定论性检验**——measurement source 全程称 **ER/atlas-derived ictal rank proxy**，不直接称"真正发作通道点亮顺序"。
> **Owner（user-locked 2026-06-08）**：**topic5 拥有**"真正发作 EEG 招募层 + 间期↔发作桥接"。与 Topic 4 H5（发作邻近 HFO 端点招募）不重复——本层碰的是真正发作 EEG 信号本身的顺序，H5 碰的是间期高频事件端点。
> **前身**：bridge Q1（NULL-locked，state fingerprint）+ Q1'（INDETERMINATE，per-subject 列联，4 strict subjects median Cramér V 0.49 一致正向但 n_eligible 5–14 功效不足）。两轮的诊断：**信号方向一致为正，死在每病人样本量上，不在指标上**。归档：`docs/archive/topic5/bridge_q1/`、`docs/archive/topic5/bridge_q1prime/`。

---

## 0. 一个真正测什么的朴素话

每个病人有好几次发作。间期（没发作时）我们已经知道每个病人有一条形状固定、被高频事件反复扫到的传播通路（topic1 的"模板"）。Stage 1 问一个比上一轮更基本、且数据基本现成的问题：

> **用现成的发作通道顺序代理（ER/atlas 派生的排名，不是新仪器），各通道被点亮的先后整体上像不像这个病人间期那条固定通路的顺序？把所有病人合在一起看，这种"像"是不是真的、超过把通道身份打乱后该有的程度？**

**一个工程诚实声明（amendment 1）**：Stage 1 并没有搭新的招募时序图，它消费的是 topic5 PR-0/PR-0.1 v2.3 atlas 那套 **ER 派生的 ictal rank**（Stage 2 才会搭 line length / broadband / HFA / CUSUM / Page-Hinkley 的多特征仪器）。所以 Stage 1 测的是"**这个现成代理是否富集间期模板排序**"，是"要不要建更干净仪器"的闸门，**不是**对真实发作传播顺序的定论。由此有两条不能跨的红线：(i) 若代理本身质量差（见 §3.5 atlas-quality gate），Stage 1 的"不像"**不能**判成"间期模板不 echo 发作传播"——可能只是代理太烂；(ii) Stage 1 的"像"也不能单独判成真实传播路径 echo，可能只是代理与间期模板共享某个粗锚。

上一轮（Q1'）一上来就问最难的那个——"发作分成哪一型，是不是由它更像哪条通路决定的"——所以每个病人内部攒不够样本，撞了样本墙。Stage 1 先问最基本的"像不像"，把所有病人合起来打这堵墙。

**一个必须说清的混淆（否则闸门会因错误原因通过）**：发作顺序和间期模板顺序"像"，可能有两种来源——(a) 真的重现了那条**具体的**传播通路；(b) 两者其实都只是"**越靠近病灶的通道越先点亮**"这个粗锚，根本没重现具体通路。我们打乱通道身份的零假设会把 (b) 的粗锚也一起打掉，所以"打乱后变平、实测更像"**同时容纳 (a) 和 (b)**，不能单独证明 (a)。因此 Stage 1 闸门的主张被诚实地放宽为 (a)∪(b)（"发作招募贴合间期的通道优先级结构，含共享的病灶锚"）；**把 (a) 从 (b) 里剥出来**是一个更锐的第二层问题（§4.1b 的"去锚偏差"检验 + Stage 2），不是 Stage 1 闸门本身。

机制论文叙事：如果"像"站得住（哪怕只是 (a)∪(b)），就先支持"**间期刻板高频模板不是孤立 biomarker，而是被频繁采样到的病理网络结构；发作时同一组通道被持续招募**"；"重现的是**具体通路**还是**粗锚**"由 §4.1b 去锚检验进一步分辨，并决定 Stage 2 值不值得建。

---

## 1. 一句话主张（locked framing）

在一个扩展的 epilepsiae + yuquan 探索性 cohort 内，**每次发作的 ER/atlas-derived ictal rank proxy 与该病人间期传播模板的相似度，跨病人合并后系统性高于 within-seizure 通道身份打乱的零假设**（单侧）。

> **Measurement-source lock（amendment 1）**：Stage 1 tests whether the currently available **ER/atlas-derived ictal rank proxy** is enriched for interictal-template ordering. It is a **gate for whether a cleaner Stage 2 recruitment instrument is worth building, not a definitive test of true ictal recruitment order.**

- **PASS 许可的是 inclusive claim**：发作代理排名贴合间期的**通道优先级结构**（含 (a) 具体通路重现 ∪ (b) 共享的病灶锚）。**禁止**把 Stage 1 PASS 单独读成"具体通路被重现"——那需要 §4.1b 去锚检验**且**更强 null（§4.6 within-shaft / anchor-matched）也为正。
- 判定语言只允许 "**像 / 不像 / 没看清**"，**禁止** "predicts seizure" / "causes" / 任何 within-subject α-claim 升格为机制因果。
- "像"= ictal recruitment ordering 与 interictal template ordering 的 rank 相似度，**相对 within-seizure channel-shuffle null** 的超出量（dimensionless，解掉跨病人可比性前置题）。

---

## 2. 假设结构

### 2.1 H1（primary）— 回声

**H1**：per-seizure echo strength `e_k`（见 §4.1）跨 cohort 合并的均值 > 0（单侧），且 bad-data regression（§4.4）下零假设变平。

- 这是闸门的**唯一主统计**（user-locked 2026-06-08：只锁回声）。
- H1 是 **inclusive claim**（通路重现 ∪ 病灶锚，§0/§1）。**§4.1b LOO 去锚 + §4.6 更强 null 是必跑的解释层**，把 PASS 进一步分成"含具体通路"vs"稳定锚为主"——它们**不**是第二道 gate（尊重"只锁回声"），但方向决定 §4.3 / §7 里 Stage 2 的优先级。
- 不需要 subtype 标签即可计算；subtype 是 H2 的事。

### 2.2 H2（secondary）— 子型结构

**H2**：在 swap-positive 子集，"发作更像哪条模板"（`sign(ρ_a − ρ_b)`）与 topic5 PR-1 z-ER subtype label 跨病人**分层合并**后系统关联（即 Q1' 原主张，但合并而非逐病人）。

- **secondary**，因为 contingency 设计 intrinsic 功效更低（Q1' 已证）。报为伴随结果，**不**决定 Stage 1 闸门通过与否。
- 方向锁：沿用 Q1' 观察到的正向（一致 Cramér V）作单侧。
- **循环性 caveat（amendment 2）**：seizure rank 来自 ER-derived atlas，subtype 也来自 z-ER；二者共享构造来源，H2 因此**不是独立验证**。明确写：*H2 is not independent of the ictal-rank construction if both derive from z-ER / ER-atlas features; therefore H2 is descriptive consistency only until Stage 2 uses a feature-independent recruitment-time map.*
- **tie 处理（item 8）**：`assignment = T0 if (ρ_a−ρ_b)>τ; T1 if (ρ_b−ρ_a)>τ; else tie`（τ=0.10）。**primary H2 = exclude tie**；**sensitivity** = multinomial / ordinal logistic 把 tie 作第三类。

### 2.3 显式不在 Stage 1

- 多特征变点招募时序图（recruitment-time map）= **Stage 2，contingent**，只有 H1 站住才动（§7）。
- 有向连接（DTF/PDC/PTE）、贝叶斯/神经场反演 = 更重的独立子项目，**不在本 spec**。

---

## 3. 队列 gating

### 3.1 入选规则（扩，直接打样本墙）

subject 入选 ⇔ **同时**满足：

1. **有稳定间期模板**：topic1 `stable_k=2`（或 broad lagPat 扩展后的稳定模板），masked（phantom-safe）。**primary set 只用 `stable_k=2`（item 7）**；`k=1`、`k>2` 走 sensitivity / case-series，并在合并时把 `k` 作 covariate + 出 `effect vs k` 图。理由：k 越多 `max_m ρ_m` 越接近"有没有任意一条碰巧像"，生物解释更宽，且与 Topic 1"k=2 是主导压缩、少数 subject 高 k 多模态"一致。模板条数 k 的 ρ 处理见 §4.1。
2. **有发作影像图谱**：topic5 PR-0 / PR-0.1 v2.3 atlas 存在且 `n_seizures_with_atlas ≥ 2`，**且通过 §3.5 atlas-quality gate**。
3. 两个来源现在都比 Q1' 时大：
   - 通道池扩展（`results/lagpat_broad/`：yuquan 17/17 done，epi 多日 running）→ 候选模板病人数增加。
   - 发作图谱补了 yuquan 9 人（PR-0.1）。

> **B0 eligibility audit（必先跑，才锁 data gate；amendment 2 + item 10）**：先输出，再锁 gate。每 subject 至少枚举：`subject_id, dataset, n_seizures_total, n_seizures_with_atlas, n_seizures_eligible, n_channels_template, n_channels_ictal, n_channels_common_{min,median,max}, rank_tie_fraction, rank_missing_fraction, rank_dynamic_range, template_k, template_stability, swap_class, ictal_rank_source, atlas_quality_flag, reference_type, montage_type, channel_name_normalization_status, duplicate_channel_flag, phantom_mask_applied, alignment_guard_pass, clinical_onset_annotation_available, deanchor_eligible, deanchor_anchor_reliability`。**在 audit 完成前不写死最终 N、MIN_CH、atlas-quality 阈值**（当前估 ≫ Q1' 的 4 strict，但以 audit 为准）。**audit 后才进 full cohort inference，不得直接跳过 audit（item 12）。**

### 3.2 模板稳健度分层

- **primary inference set** = swap_class ∈ {strict, candidate}（真有稳定 swap-core 的病人）。
- **none subset** = 阴性对照（独立报；期望 echo ≈ 0；出现强 echo → 红 flag，提示模板不稳或漏洞）。
- **all-cohort** = sensitivity（独立报）。

沿用 Topic 4 / Q1' 既有 swap_class 合同，不新造分层。

### 3.3 数据集分层（mandatory sensitivity）

epi（ECoG+SEEG）与 yuquan（SEEG）montage/reference 不同。

- **primary** = 合并，dataset 作 covariate / stratum。
- **必报 sensitivity** = epi-only、yuquan-only 各自的合并估计。
- channel-shuffle null 是 within-seizure 的，已控每次发作自身的 montage；跨数据集差异由分层报告兜底。

### 3.4 per-seizure 资格门

- `n_channels_common(seizure, template) ≥ MIN_CH` 才能算 ρ + shuffle，否则该 seizure drop。**primary MIN_CH = 8（amendment 5）**；`MIN_CH ∈ {5,6}` 仅作 low-channel exploratory sensitivity，不进 primary。理由：Spearman 在 n=5 极离散、单通道换位即大幅改变结果，且 `max_m ρ_m` 放大偶然相似。若提到 8 后 cohort 掉太多，**明确标 "low-channel exploratory" 并报掉了多少**（不 silent truncation）。B0 audit 先出 `n_channels_common` 分布再定。
- 模板顺序用 masked（phantom-safe）rank；非参与通道 NaN，不进 ρ。
- channel-order：seizure rank 向量与 template rank 向量必须对齐到同一 channel 顺序后再求 ρ（沿用 Q1' 的 alignment guard）；**mismatch 必须 hard fail（item 10），不得 silent 截断**。

### 3.5 atlas-quality gate（amendment 6 — instrument-quality 守门）

ER/atlas-derived ictal rank 是个**可能很烂的代理**。在判 H1 之前先给每 subject 一个 `atlas_quality_flag`，由 B0 audit 列计算：

- `rank_tie_fraction` 过高（rank 大量并列 → 顺序信息少）；
- `rank_dynamic_range` 过低（几乎所有通道同时点亮 → 无顺序）；
- `n_channels_common` 过少；
- `ER_atlas_failure_flag`（CUSUM/onset 未触发、atlas 退化）。

阈值 audit 后锁。**用途**：质量差的 subject 的 H1 "不像" **不能**判生物学 NULL（→ §4.3 "proxy 不足" 档）；质量差也不该污染 primary，可降为 sensitivity。

---

## 4. 统计 contract

### 4.1 H1 primary — echo strength `e_k`（合同核心）

对每次合格 seizure k：

1. **观测相似度**：对该 subject 的每条间期模板 m（m=1..k），`ρ_m = Spearman(seizure_rank, template_m_rank)`，在 **full participating-channel set**（不是只 swap 子集）上计算。
   - `r_obs = max_m ρ_m`（取最像的那条模板；null 也取同样的 max-over-templates，控选择偏差）。
   - **k 的处理（消除 §3.1 的歧义，避免 implementation-time 静默决定）**：k=2 → `max(ρ_a, ρ_b)`；k=1 → 就是那一条 `ρ_1`；k>2 → `max` over k 条。三种情形 null 都对相同的 k 条模板取 max。
   - swap-subset 版的 ρ（Q1' 已落盘，25 份 JSON）作 §4.5 sensitivity，不作 primary。
2. **零假设（within-seizure channel-label shuffle）**：固定全部 k 条 template，把 **seizure rank 向量的通道身份**随机置换 `B` 次（建议 B=2000），每次重算 `max_m ρ_m`，得 null 分布 `{r_null^(b)}`。
3. **echo strength**：`e_k = (r_obs − mean(r_null)) / sd(r_null)`（standardized exceedance；dimensionless，跨病人可比）。
   - 同时记录 one-sided percentile `p_k = (#{r_null ≥ r_obs}+1)/(B+1)`。
   - **floor-robust 注意**：primary MIN_CH=8（perms 充足，B=2000 采样下 `sd(r_null)` 稳）。但 low-channel sensitivity tier（n_ch ∈ {5,6}，distinct perms ≤ 720）`sd` 退化、z 噪声大 → 该 tier 用 `p_k` 而非 z，合并走 percentile/Stouffer（见 §4.1.4）。
4. **合并（cohort）— 统计单位优先 subject，不是 seizure（amendment 3）**：
   - **subject 内先合并**：每 subject 得 `E_s = mean_k(e_sk)`，或 Stouffer `Z_s = (1/√n_s)·Σ_k Φ⁻¹(1 − p_sk)`。
   - **cohort 层再检验（primary）**：对 `{E_s}`（及 `{Z_s}` 作 floor-robust 平行口径）做 one-sample **Wilcoxon signed-rank** + sign test + subject-level bootstrap CI，单侧 median/mean > 0。
   - **sensitivity / effect-size estimator**：per-seizure `e_k ~ 1` cluster-robust SE by subject（statsmodels OLS `cov_type='cluster'`）+ mixed-effects（seizure nested in subject）。**primary（subject-level）与 sensitivity（per-seizure cluster）方向必须一致才算稳。**
   - 理由：subject 数接近 10 时 cluster-robust SE 不稳，且 per-seizure 口径会被 seizure-rich subject 主导、小通道下 z 尺度退化；subject-level 先合并消掉这些，与 H5"per-seizure reporting + subject-clustered cohort inference"一脉相承。

### 4.1b 去锚偏差检验（必跑解释层，区分"具体通路"vs"病灶锚"）

H1 的 channel-shuffle null 无法区分 (a) 具体通路重现 与 (b) 共享病灶锚（§0）。利用"每个 subject 有多次 seizure"剥离稳定锚——**用 leave-one-seizure-out anchor，杜绝当前 seizure 参与自己的锚（amendment 3 leakage fix）**：

1. **稳定锚（LOO）= 排除当前 seizure k 后，每通道跨该 subject 其余 seizure 的平均点亮 rank**
   `r̄_{c,−k} = (1/(N_s−1)) · Σ_{ℓ≠k} r_ℓ(c)`。
   （旧定义 `r̄_c` 含 seizure k 自身 → 把单次特异偏移吸进锚、压小 δ、造去锚假阴性；LOO 修掉。）
2. **per-seizure 去锚偏差** `δ_k(c) = r_k(c) − r̄_{c,−k}`。
3. **模板侧去锚**：`δ_template(c) = template_rank(c) − r̄_{c,−k}`（或 template 自身 anchor-free 余量）。检验 `ρ(δ_k, δ_template)` 是否仍 > within-seizure shuffle null（同 §4.1 机制，作用在 δ 上）；合并同 §4.1.4（subject-level primary）。
4. **anchor reliability 必报**：channel-wise rank 的 Kendall's W / ICC（跨该 subject 的 seizure）。reliability 低 → δ 不可信，不得据此判读。

**资格门（amendment 3）**：`n_seizures ≥ 4` 才做 **primary 去锚**（LOO 锚至少基于 3 次）；`n_seizures = 3` 仅 exploratory（LOO 锚只剩 2 次，太不稳）；`n_seizures < 3` 标 `deanchor_insufficient`，只进 H1、不拖累。

**判读（不另设 gate，只 route Stage 2；语言收紧）**：
- 去锚检验**也同向显著** → 证据**含 path-specific replay**（framing (a)）→ Stage 2 强 case。
- 去锚检验**变平** → **"Stage 1 evidence is not sufficient to isolate path-specific replay beyond the stable ictal-earliness anchor"**（**不写"仅病灶锚"**——稳定的真实通路本身也会被 `r̄_{c,−k}` 吸收，flat 不能证明无具体通路）→ Stage 2 优先级下调，但不否定具体通路存在的可能。

### 4.2 H2 secondary — 子型结构

- 仅 swap-positive 子集；per-seizure label = `T0 if (ρ_a−ρ_b)>τ; T1 if (ρ_b−ρ_a)>τ; else tie`（τ=0.10，沿用 Q1'）。
- 跨病人 **subject-stratified mixed-effects logistic**（resembles-T0 ~ subtype + (1|subject)）或 CMH stratified by subject；报 pooled OR + 单侧 p + per-subject Cramér V 分布。
- **不**决定 Stage 1 闸门。

### 4.3 判决合同（把结论写进数字 — 见 `feedback_acceptance_gate_encode_conclusion`）

**站住 = inclusive claim**（通路 ∪ 病灶锚）；"具体通路"由 §4.1b 去锚 + §4.6 更强 null 共同判读，并据此定 Stage 2 优先级。所有"站住"档前置 **§3.5 atlas-quality gate 通过**——质量差的 subject 不进 primary。

| Verdict | 条件 |
|---|---|
| **站住·inclusive echo** | subject-level H1 单侧 p<0.05（Wilcoxon signed-rank，§4.1.4）**AND** median/mean E_s > 0 **AND** percentile/Stouffer combine 同向 **AND** per-seizure cluster-robust sensitivity 同向 **AND** §4.4 bad-data regression 变平 **AND** ≥10 subjects **AND** atlas-quality gate 通过 |
| **站住·含具体通路** | inclusive echo 成立 **AND** §4.1b LOO 去锚同向显著 **AND** §4.6 至少一个更强 null（within-shaft **或** anchor-matched）仍同向显著 → framing (a) → **Stage 2 强 case** |
| **站住·稳定锚为主** | inclusive echo 成立 **但** LOO 去锚与更强 null 变平 → 记为 **shared ictal/interictal channel-priority anchor**（**非** specific path replay）→ Stage 2 优先级下调 |
| **proxy 不足** | H1 flat **但** atlas-quality gate 差 / `rank_tie_fraction` 高 / `n_channels_common` 少 / ER-rank 不稳 → **不能判生物学 NULL**；Stage 2 转为 instrument-repair（反而更必要，§7）|
| **平（生物学不支持）** | H1 p≥0.15，≥10 subjects，**atlas-quality gate 通过**，更强 null 与 sensitivity 都不支持 → echo 即便合并也站不住 |
| **没看清（UNDERPOWERED）** | <6 subjects eligible，或 B0 audit 显示数据结构不可解释 |

**临界**（H1 0.05≤p<0.15 或 6≤n_subjects<10）：触发 Stage 2"搭更干净仪器 + 继续扩队列"。

### 4.4 bad-data regression（防止合并机器自造显著）

- **channel-shuffle 自检**：把每个 `r_obs` 换成它自己 null 里的一个抽样（即"假装观测也是随机的"），重跑 §4.1.4 合并 → `mean(e)` 必须 ≈0、p NS。不变平 = 机器有 bug，先修再跑真数据。
- **subtype-shuffle 自检（for H2）**：within-subject 打乱 subtype 标签 → H2 pooled OR 必须 ≈1、p NS。

### 4.5 sensitivity battery（必报）

- swap-subset ρ（Q1' 落盘版）vs full-template ρ：两者方向应一致。
- epi-only / yuquan-only（§3.3）。
- B（shuffle 次数）= 1000 vs 2000 数值稳定性。
- MIN_CH = 8（primary）vs 10（stricter）vs {5,6}（low-channel exploratory）cohort 敏感性。
- template k：k=2 primary vs 含 k=1 / k>2，报 `effect vs k`（item 7）。

### 4.6 更强 negative controls（amendment 4 — channel-shuffle 太易被粗梯度骗过）

真实电极**不是 exchangeable**（同 shaft 空间连续、近病灶天然更早、深部 vs 皮层信号质量不同、相邻 contact rank 不独立）。全通道 shuffle 会被任何粗空间梯度打显著，因此 primary gate 之外**必跑**以下 null（每个都 subject-level 合并、报方向）：

| Null | 构造 | 它否证什么 |
|---|---|---|
| **A within-shaft shuffle** | 只在同一 shaft 内打乱 channel identity | 全通道显著但 A 不显著 → echo 主要来自 shaft 间差异 / 粗解剖锚，非细粒度顺序 |
| **B shaft-block shuffle** | 整条 shaft 当 block 打乱，shaft 内顺序保留 | shaft identity 是否已足够解释 echo |
| **C anchor/distance-matched shuffle** | 按 clinical-SOZ 距离 / endpoint 距离 / 平均 ictal earliness 分箱，bin 内 shuffle | 比 §4.1b 更直接打"病灶远近锚" |
| **D between-subject template control** | 用别人的 template rank（或 template label rotation）映射到同数量通道 | 别人模板也能显著 echo → 指标太粗 |
| **E none-subset control** | swap_class=none 子集（§3.2）| none 出现强 echo = red flag（模板/通道池有洞）|

**进 §4.3 的硬约束**：`站住·含具体通路` **要求 A（within-shaft）或 C（anchor-matched）也同向显著**；否则最多 `站住·inclusive echo` / `站住·稳定锚为主`。D 显著或 E 强 echo → primary 结论作废、回查。

---

## 5. 代码 / 数据架构

### 5.1 复用（不重造）

| 需要 | 复用来源 | 问题匹配？|
|---|---|---|
| per-seizure ρ_a/ρ_b + seizure rank 向量 | `src/topic1_topic5_bridge.py::_q1prime_per_seizure`（已有；改为 full-channel-set 调用 + 可选 swap-subset）| ✅ 同一相似度，换 channel scope |
| seizure channel-onset rank（**= ER-derived proxy**，amendment 1）| topic5 PR-0 v2.3 atlas `channel_onsets`（`results/data_driven_soz/layer_a_ictal_er_rank/atlas_v2_3/`）| ⚠️ proxy，非真实招募时序图；Stage 2 才替换 |
| interictal template rank（masked）| `src/lagpat_rank_audit.mask_phantom_ranks` + topic1 stable_k=2 / broad lagPat | ✅ phantom-safe |
| swap_class 分层 | `results/interictal_propagation_masked/rank_displacement/per_subject/*.json::swap_sweep.swap_class` | ✅ 同 Q1'/H5 |
| cluster-robust SE | `statsmodels OLS().fit(cov_type='cluster', cov_kwds={'groups':subject})`（同 H5）| ✅ |
| 已落盘 per-seizure JSON（25 subj）| `results/topic1_topic5_bridge/q1prime_per_subject/*.json` | ✅ swap-subset ρ 现成（sensitivity）|

### 5.2 新代码（最小）

- `src/topic1_topic5_bridge.py` 内新增：
  - `compute_echo_strength(seizure_rank, template_ranks, *, B, rng, null_mode, shaft_ids=None, anchor_bins=None)` → `(e_k, p_k, r_obs, r_null_{mean,sd,quantiles})`。`null_mode ∈ {channel, within_shaft, shaft_block, anchor_matched, between_subject}`（§4.6）；channel = primary，其余为更强 null。
  - `compute_deanchor_echo(per_seizure_ranks, template_rank, *, B, rng)` → LOO 锚 `r̄_{c,−k}`（§4.1b，**禁止**含当前 seizure）+ δ 上的 echo + anchor reliability（Kendall W / ICC）。
  - `pool_echo_subject_level(per_seizure_df)` → subject 内 `E_s`/`Z_s` 合并 → Wilcoxon signed-rank + sign + bootstrap（primary）；per-seizure cluster-robust OLS（sensitivity）；bad-data regression。
  - `compute_atlas_quality(subject_atlas)` → `atlas_quality_flag` + B0 audit 列（§3.5）。
- `scripts/run_topic5_echo_gate.py`：`audit / per-subject / cohort / figures` 子命令（沿用 `--masked-features` + `_apply_masked_paths()` 5-line 全局 path-swap 模式）。`audit` 先跑、产出 B0 csv，**人工看过再跑 cohort**。
- TDD：synthetic seizure/template（已知 echo / 已知 null）单测；**LOO 锚无 leakage 回归测试**（含当前 seizure 会改变 δ → 必须用 LOO）；channel-order mismatch **hard raise**；每个 null_mode 在已知粗梯度数据上的行为（A/C 应能把纯锚信号判平）；bad-data regression 必返 null。

### 5.3 输出

```
results/topic5_ictal_template_echo/
├── b0_eligibility_audit.csv            # §3.1 全量列 + atlas_quality_flag（先跑先看）
├── per_subject/<ds>_<sid>.json         # per-seizure e_k/p_k(每 null_mode) + δ-echo + anchor_reliability + swap_class + dataset
├── cohort_echo_summary.json            # subject-level verdict + 每 null_mode + 去锚 + H2 secondary + sensitivity + bad-data regression
└── figures/
    ├── README.md                       # 中文逐图说明（必须，图生成后写）
    ├── echo_strength_distribution.png  # per-seizure e_k by swap_class / dataset
    ├── null_mode_panel.png             # 每 null_mode 的合并方向对比（A/C 是否仍站住）
    └── cohort_pooled_forest.png        # subject-level E_s 估计 + pooled，阴性对照对照
```

**工程 locks（item 10）**：RNG seed 固定且落盘；B=2000 并行可复现（seed-per-seizure）；null 不只存 mean/sd，**存 quantiles**；Spearman ties = `'average'` 明确锁；NaN / non-recruited channel 一律不进 ρ（不 impute 0）；**channel-order mismatch hard fail**；duplicate channel / 命名规范化在 B0 audit 标 flag；每 subject forest 图**人工视觉巡视**后才进 cohort 结论。

---

## 6. Caveats & 显式 NOT-DO

### Caveats
1. **exploratory**：本 gate 是探索性，不写 paper-level cohort claim 直到 sensitivity battery（§4.5）+ 更强 null（§4.6）全过 + user 视觉巡视图。
2. **instrument = ER/atlas-derived proxy（amendment 1/6）**：Stage 1 用的是现成 ER-rank 代理，不是真实招募时序图。**H1 flat 在 atlas-quality 差时不能判生物学 NULL**（→ "proxy 不足"档）；H1 positive 也只许 inclusive claim。
3. **H2 循环性（amendment 2）**：seizure rank 与 subtype 同源 z-ER/ER-atlas，H2 是 descriptive consistency，**非独立验证**，直到 Stage 2 用 feature-independent 仪器。
4. **去锚 flat 不等于无具体通路（amendment 3）**：稳定的真实通路也会被 LOO 锚吸收；flat 只说"Stage 1 证据不足以把 path-specific replay 从稳定 earliness 锚里剥出来"。
5. **ρ 跨病人可比性**由 null standardize 解决，但 `e_k` 仍假设 null 的 sd 非退化；MIN_CH=8 门 + audit 排除退化 seizure。
6. **epi vs yuquan** 不做"哪个数据集更强"的对比（样本不均）；只作分层 sensitivity。
7. **none-subset 强 echo = 红 flag**，不是利好——提示模板/通道池有洞，需回 topic1 审。

### 显式 NOT-DO
- 不搭 recruitment-time map（Stage 2）。
- 不做 directed connectivity / Bayesian / neural-field（独立子项目）。
- 不把 H2（子型）升格为 Stage 1 闸门。
- 不重跑 topic1 PR-2 cluster pipeline；直接消费 masked 模板。
- 不在 within-subject 写 α-claim 当机制因果。

---

## 7. Stage 2（contingent — 只有 §4.3 "站住" 才动）

Stage 2 = topic5 搭**多特征变点招募时序图**（line length / broadband / HFA power / CUSUM / Page-Hinkley per contact → 每通道一个 ictal recruitment time → propagation rank / velocity / tree），作为**替代 ER 代理的更干净仪器**，再用它重测回声（full cohort，feature-independent，也解 H2 循环性）。这是 topic5 独有贡献（H5 只碰 HFO 端点，不碰真正发作 EEG 顺序）。Stage 2 另写 spec。

**Stage 2 触发由 H1 p 值 + instrument quality 共同决定（amendment 6 / item 9）**：

| Stage 1 结果 | Stage 2 决策 |
|---|---|
| H1 positive + atlas-quality OK | Stage 2 strong case（更强仪器确认 + 去循环）|
| H1 borderline | Stage 2（搭仪器 + 扩队列）|
| **H1 flat + atlas-quality 差（proxy 不足）** | **Stage 2 as instrument-repair — 反而更必要**，不判生物学否定 |
| H1 flat + atlas-quality OK | Stage 2 暂缓，回头审"指标是否本就是噪声" |

---

## 8. 来源文档

- `docs/topic5_seizure_subtyping.md` — topic5 主文档（PR-0/PR-1）
- `docs/archive/topic5/bridge_q1/bridge_q1_results_2026-05-10.md` — Q1 NULL-locked（弃案 phase-1）
- `docs/archive/topic5/bridge_q1prime/bridge_q1prime_results_2026-05-10.md` — Q1' INDETERMINATE（power-floor 诊断来源）
- `docs/superpowers/specs/2026-05-10-topic1-topic5-bridge-design.md` §10 — Q1' channel-rank correspondence 设计（ρ_a/ρ_b/τ 定义来源）
- `docs/superpowers/plans/2026-05-24-topic4-phase3-h5-per-seizure-recruitment-plan.md` — H5（cluster-robust SE 机器 + 分工边界来源）
- `results/lagpat_broad/COHORT_SUMMARY.md` — broad lagPat 扩展（扩队列输入）
- AGENTS.md Cross-PR：lagPatRank phantom（masked 必经）、swap_class 分层、`channel_names` ordering
