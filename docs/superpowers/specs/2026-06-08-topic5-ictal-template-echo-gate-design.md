# Topic 5 — Ictal-Template-Echo Gate (Stage 1) Design Spec (2026-06-08)

> **状态**：设计稿 **v4**（v3 + v4 scope-lock 2026-06-08：**主问题 = 通用模板回声**，primary = 所有稳定 masked 模板病人，swap_class 降为预注册分层，阴性对照换 between-subject template control + bad-data；§4.6 B 不等长 shaft fail-closed `insufficient_block_exchange`、D 升为正式阴性对照），plan 同步 patch 中。
> **Topic**：把"间期那条频繁出现、形状固定的高频传播通路"与"**当前可用的发作通道顺序代理**（ER/atlas 派生的 ictal rank proxy，**不是**新搭的真实招募时序图）"在 **cohort 层面合并**对齐——问"这个现成的发作顺序代理是不是富集了间期模板的排序"。
> **定位（P0-1 reframe）**：这是 topic5 下一步的 **Stage 1 = proxy triage（代理初筛）**，**不是 gate（闸门）**。它**不**搭新仪器；只把已算好的 per-seizure 相似度跨病人合并。**关键权力边界：proxy triage 只能为 Stage 2 加速 / 加强证据，绝不否决 Stage 2。** ER/atlas 排名本就不是传播路径仪器，用它判"暂缓建真仪器"逻辑不硬。Stage 2（真正发作 EEG 的招募时序图）由**科学价值**触发——若核心问题是"早期发作传播是否复用间期通路"，那它该建就建，**ER 代理的阴性无权否决**。measurement source 全程称 **ER/atlas-derived ictal rank proxy**，不直接称"真正发作通道点亮顺序"。
> **Owner（user-locked 2026-06-08）**：**topic5 拥有**"真正发作 EEG 招募层 + 间期↔发作桥接"。与 Topic 4 H5（发作邻近 HFO 端点招募）不重复——本层碰的是真正发作 EEG 信号本身的顺序，H5 碰的是间期高频事件端点。
> **前身**：bridge Q1（NULL-locked，state fingerprint）+ Q1'（INDETERMINATE，per-subject 列联，4 strict subjects median Cramér V 0.49 一致正向但 n_eligible 5–14 功效不足）。两轮的诊断：**信号方向一致为正，死在每病人样本量上，不在指标上**。归档：`docs/archive/topic5/bridge_q1/`、`docs/archive/topic5/bridge_q1prime/`。

---

## 0. 一个真正测什么的朴素话

每个病人有好几次发作。间期（没发作时）我们已经知道每个病人有一条形状固定、被高频事件反复扫到的传播通路（topic1 的"模板"）。Stage 1 问一个比上一轮更基本、且数据基本现成的问题：

> **用现成的发作通道顺序代理（ER/atlas 派生的排名，不是新仪器），各通道被点亮的先后整体上像不像这个病人间期那条固定通路的顺序？把所有病人合在一起看，这种"像"是不是真的、超过把通道身份打乱后该有的程度？**

**一个工程诚实声明（amendment 1 + P0-1）**：Stage 1 并没有搭新的招募时序图，它消费的是 topic5 PR-0/PR-0.1 v2.3 atlas 那套 **ER 派生的 ictal rank**（Stage 2 才会搭 line length / broadband / HFA / CUSUM / Page-Hinkley 的多特征仪器）。所以 Stage 1 是 **proxy triage**：测"**这个现成代理是否富集间期模板排序**"，**不是**对真实发作传播顺序的定论。由此有三条不能跨的红线：(i) 若代理本身质量差（见 §3.5 atlas-quality gate），Stage 1 的"不像"**不能**判成"间期模板不 echo 发作传播"——可能只是代理太烂；(ii) Stage 1 的"像"也不能单独判成真实传播路径 echo，可能只是代理与间期模板共享某个粗锚；(iii) **proxy triage 的任何结果都不能否决 Stage 2**——阳性加速它、阴性只说"现成代理没给出继续证据"，建不建真仪器由科学价值定。

上一轮（Q1'）一上来就问最难的那个——"发作分成哪一型，是不是由它更像哪条通路决定的"——所以每个病人内部攒不够样本，撞了样本墙。Stage 1 先问最基本的"像不像"，把所有病人合起来打这堵墙。

**一个必须说清的混淆（否则会因错误原因判"像"）**：发作顺序和间期模板顺序"像"，可能有两种来源——(a) 真的重现了那条**具体的**传播通路；(b) 两者其实都只是"**越靠近病灶的通道越先点亮**"这个粗锚，根本没重现具体通路。我们打乱通道身份的零假设会把 (b) 的粗锚也一起打掉，所以"打乱后变平、实测更像"**同时容纳 (a) 和 (b)**，不能单独证明 (a)。因此 Stage 1 的主张被诚实地放宽为 (a)∪(b)（"发作招募贴合间期的通道优先级结构，含共享的病灶锚"）；**把 (a) 从 (b) 里剥出来**是一个更锐的第二层问题（§4.1b 的"去锚偏差"检验 + Stage 2），不是 Stage 1 proxy triage 本身。

机制论文叙事：如果"像"站得住（哪怕只是 (a)∪(b)），就先支持"**间期刻板高频模板不是孤立 biomarker，而是被频繁采样到的病理网络结构；发作时同一组通道被持续招募**"；"重现的是**具体通路**还是**粗锚**"由 §4.1b 去锚检验进一步分辨，并决定 Stage 2 值不值得建。

---

## 1. 一句话主张（locked framing）

在一个扩展的 epilepsiae + yuquan 探索性 cohort 内，**每次发作的 ER/atlas-derived ictal rank proxy 与该病人间期传播模板的相似度，跨病人合并后系统性高于 within-seizure 通道身份打乱的零假设**（单侧）。

> **Measurement-source lock（amendment 1 + P0-1）**：Stage 1 tests whether the currently available **ER/atlas-derived ictal rank proxy** is enriched for interictal-template ordering. It is a **proxy triage that can only ACCELERATE / strengthen the Stage 2 case — it can NEVER veto Stage 2.** A flat result means "the ready-made proxy gave no continuation evidence," not "true ictal recruitment is not worth testing." Building the Stage 2 recruitment instrument is driven by scientific value, not by this proxy.

- **PASS 许可的是 inclusive claim**：发作代理排名贴合间期的**通道优先级结构**（含 (a) 具体通路重现 ∪ (b) 共享的病灶锚）。**禁止**把 Stage 1 PASS 单独读成"具体通路被重现"——那需要 §4.1b 去锚检验**且**更强 null（§4.6 within-shaft / anchor-matched）也为正。
- 判定语言只允许 "**像 / 不像 / 没看清**"，**禁止** "predicts seizure" / "causes" / 任何 within-subject α-claim 升格为机制因果。
- "像"= ictal recruitment ordering 与 interictal template ordering 的 rank 相似度，**相对 within-seizure channel-shuffle null** 的超出量（dimensionless，解掉跨病人可比性前置题）。

---

## 2. 假设结构

### 2.1 H1（primary）— 回声

**H1**：per-seizure echo strength `e_k`（见 §4.1）跨 cohort 合并的均值 > 0（单侧），且 bad-data regression（§4.4）下零假设变平。

- 这是 proxy triage 的**唯一主统计**（user-locked 2026-06-08：只锁回声）。
- H1 是 **inclusive claim**（通路重现 ∪ 病灶锚，§0/§1）。**§4.1b LOO 去锚 + §4.6 更强 null 是必跑的解释层**，把 PASS 进一步分成"含具体通路"vs"稳定锚为主"——它们**不**是第二道 gate（尊重"只锁回声"），但方向决定 §4.3 / §7 里 Stage 2 的优先级。
- 不需要 subtype 标签即可计算；subtype 是 H2 的事。

### 2.2 子型结构 — **移出假设结构（P1-3）**

原 H2（"发作更像哪条模板"是否关联 z-ER subtype）**不再作为 secondary hypothesis**。因为 subtype 与 ictal rank 同源 z-ER/ER-atlas，它**不是独立验证**，放进 hypothesis 结构会被误读成一个正经检验。**降到 Appendix §9，仅作 descriptive**，不进 verdict、不进 BH-FDR family。详见 §9。

### 2.3 显式不在 Stage 1

- 多特征变点招募时序图（recruitment-time map）= **Stage 2**，由**科学价值**触发；**proxy triage 不否决它**（§7，P0-1）。
- 有向连接（DTF/PDC/PTE）、贝叶斯/神经场反演 = 更重的独立子项目，**不在本 spec**。

---

## 3. 队列 gating

### 3.1 入选规则（扩，直接打样本墙）

subject 入选 ⇔ **同时**满足：

1. **有稳定间期模板（phantom-safe，P0-2 硬合同）**：topic1 `stable_k=2`（或 broad lagPat 扩展后的稳定模板），**模板 rank 必须从 `results/interictal_propagation_masked/` 读，且每条模板带 cluster-specific `valid_mask`**（见 §3.6）。**primary set 只用 `stable_k=2`（item 7）**；`k=1`、`k>2` 走 sensitivity / case-series，并在合并时把 `k` 作 covariate + 出 `effect vs k` 图。理由：k 越多 `max_m ρ_m` 越接近"有没有任意一条碰巧像"，生物解释更宽，且与 Topic 1"k=2 是主导压缩、少数 subject 高 k 多模态"一致。模板条数 k 的 ρ 处理见 §4.1。
2. **有发作影像图谱**：topic5 PR-0 / PR-0.1 v2.3 atlas 存在且 `n_seizures_with_atlas ≥ 2`，**且通过 §3.5 atlas-quality gate**。
3. 两个来源现在都比 Q1' 时大：
   - 通道池扩展（`results/lagpat_broad/`：yuquan 17/17 done，epi 多日 running）→ 候选模板病人数增加。
   - 发作图谱补了 yuquan 9 人（PR-0.1）。

> **B0 eligibility audit（必先跑；amendment 2 + item 10 + P1-1）**：每 subject 至少枚举：`subject_id, dataset, n_seizures_total, n_seizures_with_atlas, n_seizures_eligible, n_channels_template, n_channels_ictal, n_channels_common_{min,median,max}, rank_tie_fraction, rank_missing_fraction, rank_dynamic_range, template_k, template_stability, swap_class, ictal_rank_source, atlas_quality_flag, construct_validity_flag, reference_type, montage_type, channel_name_normalization_status, duplicate_channel_flag, phantom_mask_applied, valid_mask_source, alignment_guard_pass, clinical_onset_annotation_available, deanchor_eligible, deanchor_anchor_reliability`。
> **门槛锁死，audit 只报 drop（P1-1，解 §3.1↔§3.4 矛盾）**：`MIN_CH=8`、k=2 primary、atlas-quality 阈值都**在跑数据前锁死写进 spec**，B0 audit **只报告"按这些锁定门槛各掉了多少 subject/seizure"**，**禁止**看了 audit 结果再回调门槛（否则就是"看结果调门槛"）。**audit 必须在 full cohort inference 之前跑且人工看过（item 12）。**

### 3.2 主问题口径 + 分层（user-locked 2026-06-08：**通用模板回声**）

主问题 = "**通用模板回声**"：现成发作顺序代理是否贴合**任意稳定 masked 模板**的通道顺序。因此：

- **primary inference set = 所有有稳定 masked 模板的病人**（不分 swap_class；k=2 primary 见 §3.1/item 7）。这解掉了"Goal 说通用、gate 却锁 strict/candidate"的口径错位。
- **swap_class = 预注册分层（stratifier，非 primary gate）**：strict/candidate vs none 作 planned subgroup，看"有稳定 swap-core 的病人是否回声更强"——**描述性分组**，不是 primary 的入选门。
- **阴性对照不再用 none subset**（none 病人**也有合法稳定模板**，不是"无模板"，期望它也回声）。**真正的阴性对照 = §4.6 Null D between-subject template control（别人模板）+ §4.4 bad-data regression**。
- **all-cohort 含 k=1/k>2** = sensitivity（独立报）。

> **claim 边界**：primary 站住只许写"现成发作顺序代理贴合间期**通用**通道优先级结构"；**不许**写成"swap/双模板骨架在发作里回声"——后者只是 swap stratifier 的 subgroup 观察，且 swap 是 Topic 4 的轴。

### 3.3 数据集分层（mandatory sensitivity）

epi（ECoG+SEEG）与 yuquan（SEEG）montage/reference 不同。

- **primary** = 合并，dataset 作 covariate / stratum。
- **必报 sensitivity** = epi-only、yuquan-only 各自的合并估计。
- channel-shuffle null 是 within-seizure 的，已控每次发作自身的 montage；跨数据集差异由分层报告兜底。

### 3.4 per-seizure 资格门

- `n_channels_common(seizure, template) ≥ MIN_CH` 才能算 ρ + shuffle，否则该 seizure drop。**primary MIN_CH = 8，跑数据前锁死（amendment 5 + P1-1）**；`MIN_CH ∈ {5,6}` 仅作 low-channel exploratory sensitivity，不进 primary。理由：Spearman 在 n=5 极离散、单通道换位即大幅改变结果，且 `max_m ρ_m` 放大偶然相似。**B0 audit 只报 MIN_CH=8 下掉了多少（标 "low-channel exploratory" 那部分单独列），不回调门槛。**
- **模板 rank 必须 masked（phantom-safe，P0-2）**：见 §3.6 硬合同——非参与通道带 cluster `valid_mask`，**不进 ρ**；"full participating-channel set" = 模板 valid_mask 为真 ∩ ictal 有 rank 的通道，**不是**"所有有 channel name 的通道"。
- channel-order：seizure rank 向量与 template rank 向量必须对齐到同一 channel 顺序后再求 ρ（沿用 Q1' 的 alignment guard）；**mismatch 必须 hard fail（item 10），不得 silent 截断**。

### 3.5 atlas-quality gate（amendment 6 — instrument-quality 守门）

ER/atlas-derived ictal rank 是个**可能很烂的代理**。在判 H1 之前先给每 subject 一个 `atlas_quality_flag`，由 B0 audit 列计算：

- `rank_tie_fraction` 过高（rank 大量并列 → 顺序信息少）；
- `rank_dynamic_range` 过低（几乎所有通道同时点亮 → 无顺序）；
- `n_channels_common` 过少；
- `ER_atlas_failure_flag`（CUSUM/onset 未触发、atlas 退化）。

阈值跑数据前锁。**用途**：质量差的 subject 的 H1 "不像" **不能**判生物学 NULL（→ §4.3 "proxy 不足" 档）；质量差也不该污染 primary，可降为 sensitivity。

**construct-validity sentinel（P1-2 — "有形状" ≠ "是传播"）**：以上四项只验证 rank "有没有顺序信息"，**不**验证这个顺序捕捉的是不是传播路径（也可能只是低压快活动 / 病灶距离 / 参考方式 / shaft 覆盖）。因此**必加一道小门**：抽 N_sentinel（建议 ≥5）个 sentinel seizure，人工核对该 seizure 里"最早点亮"的通道，在 **line length / broadband / HFA / ER** 四种特征下是否大体同向。同向 → `construct_validity_flag=pass`；不同向 → **该 cohort 的 H1 negative 没有生物学解释权**（只能说"ER 代理与其它特征不一致，代理失真"），不得写成"间期模板不 echo 发作传播"。`construct_validity_flag` 进 B0 audit 列。

### 3.6 phantom-safe 模板加载硬合同（P0-2 — 禁止旧 unmasked loader）

这个仓库已经吃过一次 `lagPatRank` phantom 污染的亏（非参与通道带 finite int rank，AGENTS.md Cross-PR §lagPatRank）。**不能再靠"路径看起来对"蒙混。**

- **禁止**直接 as-is 复用 `src/topic1_topic5_bridge.py` 现有 loader / 现成 `q1prime_per_subject/*.json` 里的 `template_rank` 作为 primary 模板——它们读的是旧目录、JSON 里的 `template_rank` 没有显式 per-cluster `valid_mask`。它们只许作 §4.5 swap-subset sensitivity 的来源，**不许进 primary**。
- **primary 模板必须**：从 `results/interictal_propagation_masked/` 读；每条模板（每个 cluster）带自己的 `valid_mask`（从 raw cluster bools 派生，经 `src.lagpat_rank_audit.mask_phantom_ranks` / `build_masked_kmeans_features`）；非参与通道为 NaN/masked，**永不**进 ρ 或 shuffle。
- **"full participating-channel set" 的定义锁死**：= `template valid_mask==True` ∩ `ictal rank 非缺失` 的通道交集；**不是**"所有有 channel name 的通道"。runner 必须显式构造这个交集并落 `valid_mask_source` 到 B0 audit。
- runner 沿用 `--masked-features` + `_apply_masked_paths()` 5-line 全局 path-swap 模式（AGENTS.md Runner discipline）。
- TDD 必含一条 phantom 回归：构造一个带 phantom int rank 的非参与通道，断言它**不**进 ρ（若进了，ρ 会被污染 → 测试失败）。

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

### 4.2 子型一致性 → **见 Appendix §9（descriptive only，P1-3）**

子型↔模板的列联**不在统计 contract 内**（已移出假设结构）。计算细节、self-check、解读上限全在 §9，且**不进 verdict、不进 BH-FDR、不作 cohort claim**。

### 4.3 判决合同（把结论写进数字 — 见 `feedback_acceptance_gate_encode_conclusion`）

**站住 = inclusive claim**（通路 ∪ 病灶锚）；"具体通路"由 §4.1b 去锚 + §4.6 更强 null 共同判读，并据此定 Stage 2 优先级。所有"站住"档前置 **§3.5 atlas-quality gate 通过**——质量差的 subject 不进 primary。

> **P0-1 no-veto 原则贯穿全表**：没有任何一档能写成"Stage 2 暂缓 / 不必做"。proxy triage 的所有结果只调 Stage 2 的**优先级 / 证据强度**，**不**决定它做不做——那由科学价值定（§7）。

| Verdict | 条件 | 对 Stage 2 的（非否决性）影响 |
|---|---|---|
| **站住·inclusive echo** | subject-level H1 单侧 p<0.05（Wilcoxon signed-rank，§4.1.4）**AND** median/mean E_s > 0 **AND** percentile/Stouffer combine 同向 **AND** per-seizure cluster-robust sensitivity 同向 **AND** §4.4 bad-data regression 变平 **AND** ≥10 subjects **AND** atlas-quality + construct-validity gate 通过 | 加强 |
| **站住·含具体通路** | inclusive echo 成立 **AND** §4.1b LOO 去锚同向显著 **AND** §4.6 至少一个更强 null（within-shaft **或** anchor-matched）仍同向显著 → framing (a) | **强 case（加速）** |
| **站住·稳定锚为主** | inclusive echo 成立 **但** LOO 去锚与更强 null 变平 → 记为 **shared ictal/interictal channel-priority anchor**（**非** specific path replay）| 不额外加强（但**不**否决）|
| **proxy 没给继续证据** | H1 flat **AND** atlas-quality 差 / `construct_validity_flag` fail / `rank_tie` 高 / `n_channels_common` 少 / ER-rank 不稳 → **不能判生物学 NULL**（只说"现成代理失真/没给证据"）| **instrument-repair：反而更需要建真仪器** |
| **代理阴性（quality OK）** | H1 p≥0.15，≥10 subjects，**atlas-quality + construct-validity gate 通过**，更强 null 与 sensitivity 都不支持 → "**ER 代理在干净的情况下也没看到 echo**" | 不加速；**但 Stage 2 仍由科学价值定，不暂缓** |
| **没看清（UNDERPOWERED）** | <6 subjects eligible，或 B0 audit 显示数据结构不可解释 | 中性 |

**临界**（H1 0.05≤p<0.15 或 6≤n_subjects<10）：Stage 2"搭更干净仪器 + 继续扩队列"。

### 4.4 bad-data regression（防止合并机器自造显著）

- **channel-shuffle 自检**：把每个 `r_obs` 换成它自己 null 里的一个抽样（即"假装观测也是随机的"），重跑 §4.1.4 合并 → `mean(e)` 必须 ≈0、p NS。不变平 = 机器有 bug，先修再跑真数据。
- **subtype-shuffle 自检（for §9 appendix descriptive，非 verdict）**：within-subject 打乱 subtype 标签 → 附录描述量必须 ≈ 中性。

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
| **B shaft-block shuffle** | 整条 shaft 当 block 打乱，shaft 内顺序保留；**只能交换等长 shaft** | shaft identity 是否已足够解释 echo |
| **C anchor/distance-matched shuffle** | 按 clinical-SOZ 距离 / endpoint 距离 / 平均 ictal earliness 分箱，bin 内 shuffle | 比 §4.1b 更直接打"病灶远近锚" |
| **D between-subject template control（= 通用口径下的正式阴性对照）** | 用别人的 template rank（或 template label rotation）映射到同数量通道 | 别人模板也能显著 echo → 指标太粗（**通用口径下取代 none-subset 作主阴性对照，§3.2**）|

- **B 不等长 shaft 的硬合同（P1-B）**：`shaft_block` **只能交换等长 shaft**；不等长 shaft（真实 SEEG 常见）**原样不动**。当能交换的 shaft 对不足时，**必须显式记 `insufficient_block_exchange=True`**，**禁止**悄悄返回一个被原样不动通道稀释的弱 null（那会假装"shaft identity 已解释 echo"）。该 flag 进 per-subject JSON；B 在此 subject 上判 inconclusive，不进 §4.3 硬约束。
- **D 是通用口径的正式阴性对照（P1-B + §3.2）**：D 必须作为 formal task/test 跑（不是可选），合并方向应 ≈ 中性；D 显著 → primary 结论作废、回查。

**进 §4.3 的硬约束**：`站住·含具体通路` **要求 A（within-shaft）或 C（anchor-matched）也同向显著**；否则最多 `站住·inclusive echo` / `站住·稳定锚为主`。**D（between-subject）显著 → primary 作废。**

---

## 5. 代码 / 数据架构

### 5.1 复用（不重造）

| 需要 | 复用来源 | 问题匹配？|
|---|---|---|
| Spearman-on-common helper | 仿 `_q1prime_per_seizure` 的 ρ 逻辑，**在新模块重写**（full-channel-set + masked）| ✅ 同相似度，但 channel scope 与 mask 合同不同（§3.6）|
| seizure channel-onset rank（**= ER-derived proxy**，amendment 1）| topic5 PR-0 v2.3 atlas `channel_onsets`（`results/data_driven_soz/layer_a_ictal_er_rank/atlas_v2_3/`）| ⚠️ proxy，非真实招募时序图；Stage 2 才替换 |
| **interictal template rank（masked，P0-2 primary 唯一合法来源）** | `results/interictal_propagation_masked/` + 每 cluster `valid_mask`（`src.lagpat_rank_audit.mask_phantom_ranks` / `build_masked_kmeans_features`）| ✅ phantom-safe；**§3.6 硬合同** |
| swap_class 分层 | `results/interictal_propagation_masked/rank_displacement/per_subject/*.json::swap_sweep.swap_class` | ✅ 同 Q1'/H5 |
| cluster-robust SE（sensitivity 口径）| `statsmodels OLS().fit(cov_type='cluster', cov_kwds={'groups':subject})`（同 H5）| ✅ |
| shaft 解析 | `src.propagation_skeleton_geometry.parse_shaft` | ✅ 复用 |
| 旧 per-seizure JSON（25 subj）| `results/topic1_topic5_bridge/q1prime_per_subject/*.json` | ⚠️ **仅 §4.5 swap-subset sensitivity**，**禁进 primary**（unmasked，P0-2）|

### 5.2 新代码（focused 新模块，不塞进已 1250 行的 bridge）

- **新建 `src/topic5_echo_gate.py`（pure，no I/O）**：
  - `spearman_common(rank_a, rank_b, *, min_ch)` → 公共非-NaN 通道上的 Spearman；< min_ch → NaN。**masked 通道（valid_mask=False / NaN）永不进**。
  - `echo_r_obs(seizure_rank, template_ranks, *, min_ch)` → `max_m ρ_m`（k=1/2/>2，§4.1）。
  - `shuffle_null(seizure_rank, template_ranks, *, B, rng, null_mode, blocks=None)` → null 分布；`null_mode ∈ {channel, within_shaft, shaft_block, anchor_matched}`（§4.6；within_shaft/anchor_matched = block-within permute，shaft_block = block-between permute）。
  - `compute_echo_strength(...)` → `(e_k, p_k, r_obs, r_null_{mean,sd,quantiles})`。
  - `loo_anchor(per_seizure_ranks)` → `r̄_{c,−k}`（**禁含当前 seizure**）；`compute_deanchor_echo(...)`；`anchor_reliability(...)`（Kendall W / ICC）。
  - `pool_echo_subject_level(records)` → `E_s`/`Z_s` → Wilcoxon signed-rank + sign + bootstrap（primary）；per-seizure cluster-robust OLS（sensitivity）；bad-data regression。
  - `compute_atlas_quality(...)` → `atlas_quality_flag`；`construct_validity_sentinel(...)`（§3.5，P1-2）。
  - `between_subject_control(...)` → 用别人模板 / template rotation 重算 echo（§4.6 Null D）。
- **新建 `scripts/run_topic5_echo_gate.py`**：`audit / per-subject / cohort / figures` 子命令；**从 `results/interictal_propagation_masked/` 读模板 + per-cluster valid_mask（§3.6）**；沿用 `--masked-features` + `_apply_masked_paths()` 5-line path-swap。`audit` 先跑产出 B0 csv，**人工看过再跑 cohort**。
- **新建 `scripts/plot_topic5_echo_gate.py`** + `figures/README.md`。
- TDD（`tests/test_topic5_echo_gate.py`）：已知 echo / 已知 null synthetic；**LOO 无 leakage 回归**（含当前 seizure 改变 δ）；**phantom 回归**（带 phantom int rank 的非参与通道不得进 ρ，§3.6）；channel-order mismatch **hard raise**；A/C null 在纯粗梯度数据上把锚信号判平；bad-data regression 必返 null。

### 5.3 输出

```
results/topic5_ictal_template_echo/
├── b0_eligibility_audit.csv            # §3.1 全量列 + atlas_quality_flag（先跑先看）
├── per_subject/<ds>_<sid>.json         # per-seizure e_k/p_k(每 null_mode) + δ-echo + anchor_reliability + swap_class + dataset
├── cohort_echo_summary.json            # subject-level verdict + 每 null_mode + 去锚 + sensitivity + bad-data regression + §9 子型描述(appendix)
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
1. **exploratory**：本 triage 是探索性，不写 paper-level cohort claim 直到 sensitivity battery（§4.5）+ 更强 null（§4.6）全过 + user 视觉巡视图。
2. **proxy triage 不否决 Stage 2（P0-1，最重要）**：Stage 1 用的是现成 ER-rank 代理，不是真实招募时序图。它的任何结果**只调 Stage 2 优先级，不决定 Stage 2 做不做**。"H1 flat" 永远不能写成"Stage 2 暂缓 / 真实传播不值得测"。
3. **phantom-safe 硬合同（P0-2）**：模板 rank 必须走 §3.6（masked，per-cluster valid_mask，禁旧 loader as-is）。"full participating-channel set" ≠ "所有有名字的通道"。
4. **atlas-quality"有形状"≠"是传播"（P1-2）**：construct-validity sentinel 没过的 cohort，H1 negative 无生物学解释权。
5. **去锚 flat 不等于无具体通路（amendment 3）**：稳定的真实通路也会被 LOO 锚吸收；flat 只说"Stage 1 证据不足以把 path-specific replay 从稳定 earliness 锚里剥出来"。
6. **ρ 跨病人可比性**由 null standardize 解决，但 `e_k` 仍假设 null 的 sd 非退化；MIN_CH=8 门 + audit 排除退化 seizure。
7. **epi vs yuquan** 不做"哪个数据集更强"的对比（样本不均）；只作分层 sensitivity。
8. **通用口径下 none-subset 也期望回声**（none 病人仍有合法稳定模板）——它是 swap **分层**的一格，**不是**阴性对照。正式阴性对照 = §4.6 Null D between-subject + §4.4 bad-data（D 显著才是红 flag）。
9. **H2 = appendix descriptive（P1-3）**：seizure rank 与 subtype 同源，非独立验证；只在 §9 描述，不进 verdict / 不进 hypothesis 结构。

### 显式 NOT-DO
- 不搭 recruitment-time map（Stage 2）。
- 不做 directed connectivity / Bayesian / neural-field（独立子项目）。
- **不让 proxy triage 否决 Stage 2（P0-1）**。
- **不直接 as-is 复用旧 unmasked 模板 loader 作 primary（P0-2）**。
- 不重跑 topic1 PR-2 cluster pipeline；直接消费 masked 模板。
- 不在 within-subject 写 α-claim 当机制因果。

---

## 7. Stage 2（由科学价值触发 — proxy triage 不否决，P0-1）

Stage 2 = topic5 搭**多特征变点招募时序图**（line length / broadband / HFA power / CUSUM / Page-Hinkley per contact → 每通道一个 ictal recruitment time → propagation rank / velocity / tree），作为**替代 ER 代理的更干净仪器**，再用它重测回声（full cohort，feature-independent，也解附录 H2 循环性）。这是 topic5 独有贡献（H5 只碰 HFO 端点，不碰真正发作 EEG 顺序）。Stage 2 另写 spec。

**触发原则（P0-1）**：核心科学问题"早期发作传播是否复用间期通路"**本身**就值得用一个真仪器测；**Stage 1 的 proxy triage 只调节优先级 / 证据强度，绝不否决 Stage 2**。下表是"加速还是不加速"，**没有"暂缓 / 不做"这一格**：

| Stage 1 结果 | Stage 2 优先级（非否决）|
|---|---|
| 站住·含具体通路 | **强 case，加速**（更强仪器确认 + 去循环）|
| 站住·inclusive / 临界 | 加速（搭仪器 + 扩队列）|
| **proxy 没给继续证据**（flat + 代理失真 / construct-validity fail）| **instrument-repair：反而更需要建真仪器**（代理烂正是要换它）|
| 代理阴性（quality+construct-validity OK，flat）| **不加速，但不暂缓**——真仪器是否建由科学价值定；可顺带记"现成代理在干净情况下也没看到 echo"作背景 |

---

## 8. 来源文档

- `docs/topic5_seizure_subtyping.md` — topic5 主文档（PR-0/PR-1）
- `docs/archive/topic5/bridge_q1/bridge_q1_results_2026-05-10.md` — Q1 NULL-locked（弃案 phase-1）
- `docs/archive/topic5/bridge_q1prime/bridge_q1prime_results_2026-05-10.md` — Q1' INDETERMINATE（power-floor 诊断来源）
- `docs/superpowers/specs/2026-05-10-topic1-topic5-bridge-design.md` §10 — Q1' channel-rank correspondence 设计（ρ_a/ρ_b/τ 定义来源）
- `docs/superpowers/plans/2026-05-24-topic4-phase3-h5-per-seizure-recruitment-plan.md` — H5（cluster-robust SE 机器 + 分工边界来源）
- `results/lagpat_broad/COHORT_SUMMARY.md` — broad lagPat 扩展（扩队列输入）
- AGENTS.md Cross-PR：lagPatRank phantom（masked 必经）、swap_class 分层、`channel_names` ordering

---

## 9. Appendix（descriptive only，**非 hypothesis、非 verdict**）— 子型一致性

**降级说明（P1-3）**：原 H2 因 subtype 与 ictal rank 同源 z-ER/ER-atlas，**不是独立验证**，故移出假设结构，仅在此附录作描述性观察，**不进 §4.3 verdict、不进 BH-FDR、不作 cohort claim**。

- 观察量：swap-positive 子集内，每次发作"更像哪条模板"（`sign(ρ_a − ρ_b)`，τ=0.10，tie 排除）与 topic5 PR-1 z-ER subtype label 的列联（subject-stratified，描述 pooled OR + per-subject Cramér V 分布）。
- self-check（仅防自造显著，不是检验）：within-subject 打乱 subtype 标签 → 描述量应 ≈ 中性。
- 解读上限：只能写"在 ER 同源构造下，子型与'更像哪条模板'呈/不呈方向性一致"；**禁止**任何独立验证 / 因果 / cohort 主张。真正独立的子型↔模板检验留给 Stage 2 feature-independent 仪器。
